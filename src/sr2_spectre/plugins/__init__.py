"""Plugin system — protocols, registry, and plugin types.

Plugins extend spectre with I/O channels (single_shot, TUI, HTTP, etc.)
and lifecycle hooks (heartbeat, timer).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

__all__ = ["Plugin", "InputPlugin", "OutputPlugin", "PluginRegistry"]


@runtime_checkable
class Plugin(Protocol):
    """Base plugin protocol."""
    name: str

    async def start(self, agent: "Agent") -> None: ...
    async def stop(self) -> None: ...


@runtime_checkable
class InputPlugin(Plugin):
    """Plugin that receives user input and routes to agent."""
    async def run(self, agent: "Agent") -> None: ...


@runtime_checkable
class OutputPlugin(Plugin):
    """Plugin that formats and sends agent output."""
    async def handle_output(self, text: str, metadata: dict[str, Any]) -> None: ...


class PluginRegistry:
    """Load and manage plugins."""

    def __init__(self) -> None:
        self._plugins: list[Plugin] = []

    def register(self, plugin: Plugin) -> None:
        self._plugins.append(plugin)

    @property
    def plugins(self) -> list[Plugin]:
        return list(self._plugins)

    async def start_all(self, agent: "Agent") -> None:
        for plugin in self._plugins:
            await plugin.start(agent)

    async def stop_all(self) -> None:
        for plugin in reversed(self._plugins):
            await plugin.stop()
