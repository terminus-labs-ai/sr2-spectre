"""Agent — SR2-powered spectre agent.

Owns:
- Session identity and authoritative conversation history (list[Message])
- Tool registry (execution side)
- SR2 instance (context compilation, LLM call, token budgets)

SR2 is invoked as a stateless context compiler + streamer.
Spectre drives the tool execution loop and owns all conversation state.

Design: spectre owns history; SR2 is stateless per-round.
Each round: seed prior history → turn(increment) → reconstruct assistant turn
→ execute tools → loop until no tool calls or max_tool_rounds exceeded.
"""

from __future__ import annotations

import logging
from typing import Any

from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2_spectre.config import SpectreConfig
from sr2_spectre.core.loop import TurnResult
from sr2_spectre.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """SR2 Spectre agent.

    Owns:
    - session_id and authoritative history (list[Message])
    - ToolRegistry (tool execution only — definitions injected via SR2 pipeline)
    - SR2 instance (pipeline, LLM, token budgets, context compilation)

    SR2 is a stateless per-round compiler. Agent seeds it with prior history
    and passes the newest message as user_input each turn.
    """

    def __init__(
        self,
        config: SpectreConfig,
        session_id: str | None = None,
    ) -> None:
        self.config = config
        self.session_id: str = session_id or f"{config.agent.name}-default"
        self.history: list[Message] = []

        # Tool registry — spectre owns tool *execution*
        self.registry = ToolRegistry()
        for tool_cfg in config.agent.tools:
            self.registry.register_from_class_path(tool_cfg.class_path, tool_cfg.config)

        # Build LLM callable — spectre constructs it, then hands it to SR2
        model_cfg = config.models["default"]
        llm_callable = LiteLLMCallable(
            model=model_cfg.model,
            base_url=model_cfg.base_url,
        )

        # SR2 owns context compilation, tool definition injection, and LLM calls
        self.sr2 = SR2(
            pipeline_config=config.pipeline,
            llm={"default": llm_callable},
            token_counter=CharacterTokenCounter(),
            session_id=self.session_id,
            extras={"tool_registry": self.registry},
        )

    async def handle_user_message(self, text: str) -> TurnResult:
        """Process a user message through the SR2-powered tool loop.

        Protocol (per spec OQ1):
        - Append user message to authoritative history
        - Loop up to max_tool_rounds:
            - seed SR2 with prior history (all but last message)
            - call sr2.turn(user_input=last_message.content) — stateless compile
            - reconstruct assistant turn from stream (text + tool_use blocks)
            - if no tool calls → return
            - execute each tool, catch errors, append ToolResultBlocks
            - append result message to history (becomes increment next round)
        - If max_tool_rounds exceeded → return last text with warning
        """
        self.history.append(Message(role="user", content=[TextBlock(text=text)]))

        max_rounds = self.config.agent.max_tool_rounds
        total_tool_calls = 0
        last_text = ""

        for _round in range(max_rounds):
            prior = self.history[:-1]
            increment = self.history[-1].content

            # Seed SR2 with correct prior — overwrites its lossy accumulation
            self.sr2.seed_session(prior)

            text_acc: list[str] = []
            tool_uses: list[ToolUseBlock] = []

            async for ev in self.sr2.turn(user_input=increment):
                if ev.type == "text" and ev.text:
                    text_acc.append(ev.text)
                elif ev.type == "tool_use":
                    tool_uses.append(
                        ToolUseBlock(
                            id=ev.tool_use_id,
                            name=ev.tool_name,
                            input=ev.tool_input,
                        )
                    )
                # "end" and "usage" events: fall through, generator terminates

            last_text = "".join(text_acc)
            assistant_content = (
                ([TextBlock(text=last_text)] if last_text else []) + tool_uses
            )
            self.history.append(Message(role="assistant", content=assistant_content))

            if not tool_uses:
                # No tool calls — final response
                logger.debug(
                    "Turn complete after %d round(s), %d tool calls",
                    _round + 1,
                    total_tool_calls,
                )
                return TurnResult(
                    text=last_text,
                    tool_calls_executed=total_tool_calls,
                )

            # Execute tools — FR13: errors caught and fed back, not raised
            result_blocks: list[ToolResultBlock] = []
            for tu in tool_uses:
                total_tool_calls += 1
                try:
                    out = await self.registry.execute(tu.name, tu.input)
                    result_blocks.append(
                        ToolResultBlock(tool_use_id=tu.id, content=str(out))
                    )
                except Exception as exc:
                    logger.warning("Tool %r failed: %s", tu.name, exc)
                    result_blocks.append(
                        ToolResultBlock(
                            tool_use_id=tu.id,
                            content=f"ERROR: {exc}",
                        )
                    )

            # Tool results become the next increment
            self.history.append(Message(role="user", content=result_blocks))

        # FR14: exceeded max_tool_rounds — return last text with surfaced warning
        warning = (
            f"\n\n[Warning: tool loop exceeded max_tool_rounds={max_rounds}. "
            "Stopping to prevent infinite execution.]"
        )
        logger.warning(
            "max_tool_rounds=%d exceeded for session %s", max_rounds, self.session_id
        )
        return TurnResult(
            text=last_text + warning,
            tool_calls_executed=total_tool_calls,
        )

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        fn: Any,
    ) -> None:
        """Register a tool at runtime (not from config)."""
        self.registry.register(name, description, input_schema, fn)

    def new_session(self, session_id: str | None = None) -> None:
        """Reset conversation history and start a new session."""
        self.session_id = session_id or f"{self.config.agent.name}-default"
        self.history = []
