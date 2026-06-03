"""Runtime — shared sub-runtime for all frames.

Holds config, LLM callable, MCP clients, tool registry, and shared stores.
One Runtime instance serves N per-frame Sessions, each with its own SR2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer

from sr2.integrations.litellm import LiteLLMCallable
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2_spectre.config import SpectreConfig
from sr2_spectre.mcp.client import MCPClient, MCPConnectionError
from sr2_spectre.session import Session
from sr2_spectre.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Runtime:
    """Shared sub-runtime for all frames.

    Owns:
    - SpectreConfig (single source of truth)
    - LiteLLMCallable (one LLM path)
    - ToolRegistry (tool definitions; stateless executors)
    - MCPClient instances (connected once)
    - Shared MemoryStore and ProvenanceStore (future — FR6)

    Creates per-frame Session instances via new_session().
    """

    def __init__(self, config: SpectreConfig) -> None:
        self.config = config
        self.registry = ToolRegistry()

        # Register tools from config
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

        # Build LLM callable
        model_cfg = config.models["default"]
        self.llm = LiteLLMCallable(
            model=model_cfg.model,
            base_url=model_cfg.base_url,
        )

    async def initialize(self) -> None:
        """Connect all MCP clients and register their tool bridges into the registry."""
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

    def new_session(
        self,
        frame_id: str,
        tracer: "Tracer | None" = None,
    ) -> Session:
        """Create a new per-frame Session with its own SR2 instance.

        The Session shares the Runtime's tool registry, LLM callable, and
        pipeline config, but has independent history and serialization.
        """
        return Session(
            frame_id=frame_id,
            config=self.config,
            llm=self.llm,
            registry=self.registry,
            tracer=tracer,
        )

    async def aclose(self) -> None:
        """Close all MCP client transports. Safe to call even if initialize() was never called."""
        for client in self._mcp_clients:
            await client.close()
