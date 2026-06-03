"""Session — per-frame conversation state.

Each Session owns an SR2 instance (session_id = frame_id), its own history,
and a per-frame asyncio.Lock serializing turns. The SR2 shares the Runtime's
tool registry and LLM callable but maintains independent conversation state.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer

from sr2.config.models import ToolLoopLimitError
from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.token_counting import CharacterTokenCounter

from sr2_spectre.config import SpectreConfig
from sr2_spectre.core import TurnResult
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)
from sr2_spectre.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Session:
    """Per-frame conversation state.

    Owns:
    - frame_id: stable identity (= SR2 session_id)
    - SR2 instance: constructed per-frame with shared provenance injected
    - history: list[Message] — only this frame's transcript
    - _lock: asyncio.Lock — serializes turns within this frame
    """

    def __init__(
        self,
        frame_id: str,
        config: SpectreConfig,
        llm: LiteLLMCallable,
        registry: ToolRegistry,
        tracer: "Tracer | None" = None,
    ) -> None:
        self.frame_id = frame_id
        self.config = config
        self._registry = registry
        self.history: list[Message] = []
        self._lock = asyncio.Lock()

        # SR2 owns context compilation, tool definition injection, and LLM calls
        self.sr2 = SR2(
            pipeline_config=config.pipeline,
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            session_id=frame_id,
            tool_source=self._registry,
            tracer=tracer,
            tool_executor=self._execute_tool,
        )

    async def _execute_tool(self, block: ToolUseBlock) -> ToolResultBlock:
        """SR2 tool_executor callback. Executes a tool via the shared registry.

        Truncates oversized results before they enter context.
        Special-cases ``complete_step``: on success, emits a
        ``plan_step_completed`` event on the SR2 bus for step-compaction.
        """
        max_bytes = self.config.agent.tool_result_max_bytes

        def _truncate(content: str, name: str) -> str:
            if len(content) <= max_bytes:
                return content
            truncated = content[:max_bytes]
            return (
                f"{truncated}\n\n"
                f"[TRUNCATED: output exceeded {max_bytes} bytes "
                f"(original size: {len(content)} bytes, tool: {name})]"
            )

        try:
            out = await self._registry.execute(block.name, block.input)
            content = _truncate(str(out), block.name)
            result = ToolResultBlock(tool_use_id=block.id, content=content)

            # Emit plan_step_completed event on successful complete_step
            if block.name == "complete_step":
                event_data = self._is_complete_step_success(result)
                if event_data is not None:
                    self.sr2.bus.queue(
                        Event(
                            name="plan_step_completed",
                            phase=EventPhase.COMPLETED,
                            source_layer="plan",
                            data=event_data,
                        )
                    )

            return result
        except Exception as exc:
            logger.warning("Tool %r failed: %s", block.name, exc)
            content = _truncate(f"ERROR: {exc}", block.name)
            return ToolResultBlock(tool_use_id=block.id, content=content, is_error=True)

    def _is_complete_step_success(self, result_block: ToolResultBlock) -> dict[str, Any] | None:
        """Check if a tool result is a successful complete_step output."""
        content = result_block.content
        if isinstance(content, list):
            texts = [b.text for b in content if hasattr(b, "text")]
            content = " ".join(texts)
        if not isinstance(content, str):
            return None
        try:
            data = _json.loads(content)
        except (_json.JSONDecodeError, ValueError):
            return None
        if data.get("success") is True:
            return {
                "frame": data.get("frame", ""),
                "plan": data.get("plan", ""),
                "task": data.get("task", ""),
                "order": data.get("order", 0),
            }
        return None

    async def stream_message(self, text: str) -> AsyncIterator[AgentEvent]:
        """Stream agent events for a user message, serialized by _lock."""
        async with self._lock:
            self.history.append(Message(role="user", content=[TextBlock(text=text)]))

            prior = self.history[:-1]
            increment = self.history[-1].content
            self.sr2.seed_session(prior)

            text_acc: list[str] = []
            total_tool_calls = 0
            tool_id_to_name: dict[str, str] = {}

            try:
                async for ev in self.sr2.turn(user_input=increment):
                    if ev.type == "text" and ev.text:
                        text_acc.append(ev.text)
                        yield AgentTextDelta(text=ev.text)
                    elif ev.type == "tool_use_emitted" and ev.tool_uses:
                        for tu in ev.tool_uses:
                            total_tool_calls += 1
                            tool_id_to_name[tu.id] = tu.name
                            yield AgentToolStart(tool_id=tu.id, name=tu.name, input=tu.input)
                    elif ev.type == "tool_result_received" and ev.tool_results:
                        for tr in ev.tool_results:
                            yield AgentToolResult(
                                tool_id=tr.tool_use_id,
                                name=tool_id_to_name.get(tr.tool_use_id, ""),
                                content=tr.content,
                                is_error=getattr(tr, "is_error", False),
                            )
            except ToolLoopLimitError:
                notice = "Tool iteration limit reached; stopping."
                text_acc.append(notice)
                yield AgentTextDelta(text=notice)

            last_text = "".join(text_acc)
            assistant_content = [TextBlock(text=last_text)] if last_text else []
            self.history.append(Message(role="assistant", content=assistant_content))

            logger.debug(
                "Turn complete, %d tool calls",
                total_tool_calls,
            )
            yield AgentDone(tool_calls_executed=total_tool_calls)

    async def handle_user_message(self, text: str) -> TurnResult:
        """Process a user message and return a TurnResult."""
        text_parts: list[str] = []
        total = 0
        async for ev in self.stream_message(text):
            if isinstance(ev, AgentTextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, AgentDone):
                total = ev.tool_calls_executed
        return TurnResult(text="".join(text_parts), tool_calls_executed=total)
