"""Spectre pipeline providers — registered via entry points into SR2.

SpectreToolProvider:
  - Entry point group: sr2.tool_providers
  - Entry point name:  spectre_tools
  - Reads ToolRegistry from deps.extras["tool_registry"]
  - Returns registry's ToolDefinitions to the SR2 tools compilation layer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sr2.pipeline.events import Event, EventSubscription

if TYPE_CHECKING:
    from sr2.config.models import ToolProviderConfig
    from sr2.models import ToolDefinition
    from sr2.pipeline.dependencies import Dependencies
    from sr2_spectre.tools.registry import ToolRegistry


class SpectreToolProvider:
    """SR2 ToolProvider that surfaces spectre's ToolRegistry into the pipeline.

    Registered as ``spectre_tools`` under the ``sr2.tool_providers`` entry-point
    group. SR2 discovers it lazily on the first call to ``_TOOL_PROVIDERS.get()``.

    Lifecycle:
    - ``build(config, deps)`` — called once at SR2 init; reads ToolRegistry from
      ``deps.extras["tool_registry"]``.
    - ``provide(events)`` — called each turn by the Layer when subscribed events
      arrive; returns the registry's current ToolDefinitions.
    """

    name: str = "spectre_tools"

    def __init__(self) -> None:
        # Populated by build()
        self._registry: ToolRegistry
        self.subscriptions: list[EventSubscription] = []
        self.max_executions: int = 1
        self.execution_count: int = 0

    @classmethod
    def build(
        cls,
        config: "ToolProviderConfig",
        deps: "Dependencies",
    ) -> "SpectreToolProvider":
        """Construct provider from pipeline config and injected dependencies.

        Raises KeyError if deps.extras["tool_registry"] is absent — the
        caller (Agent) is responsible for passing it via SR2(extras=...).
        """
        self = cls()
        # Raises KeyError if not present — fail fast at build time
        self._registry = deps.extras["tool_registry"]
        self.subscriptions = [EventSubscription(event_name="turn_start")]
        self.max_executions = config.max_executions
        self.execution_count = 0
        return self

    async def provide(self, events: list[Event]) -> list["ToolDefinition"]:
        """Return SR2 ToolDefinitions from the bound ToolRegistry."""
        self.execution_count += 1
        return self._registry.to_sr2_definitions()
