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
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer

from sr2.config.models import ToolLoopLimitError
from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
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
from sr2_spectre.mcp.client import MCPClient, MCPConnectionError
from sr2_spectre.tools.registry import ToolRegistry

# Type alias for SR2's tool_executor callback
ToolExecutor = Callable[[ToolUseBlock], Awaitable[ToolResultBlock]]

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
        tracer: "Tracer | None" = None,
    ) -> None:
        self.config = config
        self.session_id: str = session_id or f"{config.agent.name}-default"
        self.history: list[Message] = []

        # Tool registry — spectre owns tool *execution*
        self.registry = ToolRegistry()
        for tool_cfg in config.agent.tools:
            self.registry.register_from_class_path(tool_cfg.class_path, tool_cfg.config)

        # MCP clients — one per mcp_servers entry; connected lazily via initialize()
        self._mcp_clients: list[MCPClient] = []
        for mcp_cfg in config.agent.mcp_servers:
            if mcp_cfg.type == "stdio":
                client = MCPClient(server_type="stdio", command=mcp_cfg.command, args=mcp_cfg.args, env=mcp_cfg.env)
            else:
                client = MCPClient(server_type="http", url=mcp_cfg.url)
            self._mcp_clients.append(client)

        # Build LLM callable — spectre constructs it, then hands it to SR2
        model_cfg = config.models["default"]
        llm_callable = LiteLLMCallable(
            model=model_cfg.model,
            base_url=model_cfg.base_url,
        )

        # agent.max_tool_rounds is the authoritative tool-loop limit: it wins
        # over whatever the pipeline config originally carried.
        config.pipeline.max_tool_iterations = config.agent.max_tool_rounds

        # SR2 owns context compilation, tool definition injection, and LLM calls
        # Spectre provides the tool_executor callback so SR2 can execute tools
        self.sr2 = SR2(
            pipeline_config=config.pipeline,
            llm={"default": llm_callable},
            token_counter=CharacterTokenCounter(),
            session_id=self.session_id,
            extras={"tool_registry": self.registry},
            tracer=tracer,
            tool_executor=self._execute_tool,
        )

    async def _execute_tool(self, block: ToolUseBlock) -> ToolResultBlock:
        """SR2 tool_executor callback. Executes a tool via the registry."""
        try:
            out = await self.registry.execute(block.name, block.input)
            return ToolResultBlock(tool_use_id=block.id, content=str(out))
        except Exception as exc:
            logger.warning("Tool %r failed: %s", block.name, exc)
            return ToolResultBlock(tool_use_id=block.id, content=f"ERROR: {exc}", is_error=True)

    async def initialize(self) -> None:
        """Connect all MCP clients and register their tool bridges into the registry.

        Failures for individual servers are caught and logged as warnings so that
        one bad server does not prevent the agent from starting.
        """
        for client in self._mcp_clients:
            try:
                bridges = await client.connect()
                for bridge in bridges:
                    self.registry.register(
                        name=bridge.name,
                        description=bridge.description,
                        input_schema=bridge.input_schema,
                        fn=bridge,
                    )
            except MCPConnectionError as exc:
                logger.warning("MCP server failed to connect: %s", exc)

    async def stream_message(self, text: str) -> AsyncIterator[AgentEvent]:
        """Stream agent events for a user message.

        SR2 handles the tool loop internally via tool_executor. We consume
        its stream events and translate them to AgentEvent types for the TUI.

        Yields AgentTextDelta, AgentToolStart, AgentToolResult events during
        processing, and always yields AgentDone as the final event.
        """
        self.history.append(Message(role="user", content=[TextBlock(text=text)]))

        prior = self.history[:-1]
        increment = self.history[-1].content
        self.sr2.seed_session(prior)

        text_acc: list[str] = []
        total_tool_calls = 0

        try:
            async for ev in self.sr2.turn(user_input=increment):
                if ev.type == "text" and ev.text:
                    text_acc.append(ev.text)
                    yield AgentTextDelta(text=ev.text)
                elif ev.type == "tool_use_emitted" and ev.tool_uses:
                    for tu in ev.tool_uses:
                        total_tool_calls += 1
                        yield AgentToolStart(tool_id=tu.id, name=tu.name, input=tu.input)
                elif ev.type == "tool_result_received" and ev.tool_results:
                    for tr in ev.tool_results:
                        yield AgentToolResult(
                            tool_id=tr.tool_use_id,
                            name="",  # SR2 doesn't carry name on result; that's OK for TUI
                            content=tr.content,
                            is_error=getattr(tr, "is_error", False),
                        )
                # Suppress iteration_complete, usage, end events from leaking to TUI
        except ToolLoopLimitError:
            # SR2 hit the tool-loop iteration limit. Stop consuming, surface a
            # short notice, and fall through to the normal AgentDone emission.
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
        """Process a user message through the SR2-powered tool loop.

        Re-implemented on top of stream_message() — collects all events and
        returns a TurnResult.
        """
        text_parts: list[str] = []
        total = 0
        async for ev in self.stream_message(text):
            if isinstance(ev, AgentTextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, AgentDone):
                total = ev.tool_calls_executed
        return TurnResult(text="".join(text_parts), tool_calls_executed=total)

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        fn: Any,
    ) -> None:
        """Register a tool at runtime (not from config)."""
        self.registry.register(name, description, input_schema, fn)

    async def aclose(self) -> None:
        """Close all MCP client transports. Safe to call even if initialize() was never called."""
        for client in self._mcp_clients:
            await client.close()

    def new_session(self, session_id: str | None = None) -> None:
        """Reset conversation history and start a new session."""
        self.session_id = session_id or f"{self.config.agent.name}-default"
        self.history = []
