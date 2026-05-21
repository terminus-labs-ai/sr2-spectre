"""TUI plugin — interactive loop with prompt-toolkit.

Usage: sr2-spectre --plugin tui
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent


class TUIPlugin:
    name = "tui"

    def __init__(self) -> None:
        self._running = False

    async def start(self, agent: "Agent") -> None:
        """Initialize TUI."""
        self._running = True
        print(f"Spectre TUI — session {agent.session_id}")
        print("Type a message (or 'quit' to exit):")

    async def stop(self) -> None:
        """Cleanup."""
        self._running = False

    async def run(self, agent: "Agent") -> None:
        """Interactive loop."""
        from prompt_toolkit import prompt

        while self._running:
            try:
                user_input = prompt("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break

            result = await agent.handle_user_message(user_input)
            print(f"\n{result.text}\n")
