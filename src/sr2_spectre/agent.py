"""Agent — owns session, history, tool loop, and system prompt.

This is the central orchestrator. Plugins plug into it via the Plugin protocol.
"""
from __future__ import annotations

import logging
from typing import Any

from sr2_spectre.config import AgentConfig
from sr2_spectre.core.client import RelayClient
from sr2_spectre.core.loop import TurnResult, run_tool_loop
from sr2_spectre.core.session import Session
from sr2_spectre.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """SR2 Spectre agent.

    Owns:
    - Session identity and history
    - Tool registry
    - Relay client
    - System prompt and model config
    """

    def __init__(
        self,
        config: AgentConfig,
        session_id: str | None = None,
    ) -> None:
        self.config = config
        self.registry = ToolRegistry()
        self.session = Session(
            session_id=session_id or f"{config.name}-default"
        )
        self.client = RelayClient(
            model=config.model,
            base_url=config.relay_base_url,
        )

        # Register tools from config
        for tool_cfg in config.tools:
            self.registry.register_from_class_path(
                tool_cfg.class_path, tool_cfg.config
            )

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def system_prompt(self) -> str:
        return self.config.system_prompt

    async def handle_user_message(self, text: str) -> TurnResult:
        """Process a user message through the tool loop.

        This is the main entry point for plugins:
        1. Append user message to session history
        2. Run tool loop via relay
        3. Return TurnResult
        """
        self.session.append_user(text)

        tools_defs = self.registry.to_definitions()

        logger.info(
            f"Running turn for session {self.session_id} "
            f"(tools={len(tools_defs)}, history={len(self.session.history)})"
        )

        result = await run_tool_loop(
            client=self.client,
            system_prompt=self.system_prompt,
            history=self.session.history,
            tools_definitions=tools_defs,
            tool_executor=self.registry,
        )

        logger.info(
            f"Turn complete: {result.tool_calls_executed} tool calls, "
            f"{len(result.text)} chars response"
        )

        return result

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
        """Create a fresh session."""
        self.session = Session(
            session_id=session_id or f"{self.config.name}-default"
        )
