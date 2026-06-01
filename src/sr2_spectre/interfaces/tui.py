"""TUI interface — interactive loop with prompt-toolkit.

Usage: sr2-spectre --interface tui
"""
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from sr2_spectre.events import AgentDone, AgentTextDelta, AgentToolResult, AgentToolStart

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

_HELP = """\
Commands:
  /quit   — exit the TUI
  /exit   — exit the TUI
  /reset  — start a new session
  /help   — show this help
  /tools  — list available tools
"""


class TUIInterface:
    name = "tui"

    def __init__(self) -> None:
        self._running = False
        self._session: PromptSession | None = None

    async def start(self, agent: "Agent") -> None:
        """Initialize TUI."""
        self._running = True

    async def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False

    async def run(self, agent: "Agent") -> None:
        """Interactive loop."""
        self._running = True
        session = PromptSession()

        # patch_stdout routes all output through prompt-toolkit's output
        # machinery, preventing keystrokes from echoing into streaming output.
        # Falls back to nullcontext when not connected to a real TTY (e.g. tests).
        stdout_ctx = patch_stdout() if sys.stdout.isatty() else nullcontext()
        with stdout_ctx:
            while self._running:
                # --- prompt ---
                try:
                    user_input = await session.prompt_async("> ")
                except KeyboardInterrupt:
                    print("\nInterrupted.")
                    break
                except EOFError:
                    print("\nEOF.")
                    break

                # --- empty / whitespace ---
                if not user_input or not user_input.strip():
                    continue

                # --- slash commands ---
                stripped = user_input.strip()
                if stripped in ("/quit", "/exit"):
                    print("Goodbye.")
                    break

                if stripped == "/reset":
                    agent.new_session()
                    print("Session reset.")
                    continue

                if stripped == "/help":
                    print(_HELP)
                    continue

                if stripped == "/tools":
                    print(str(agent.registry.list_names()))
                    continue

                # --- stream message ---
                try:
                    async for ev in agent.stream_message(stripped):
                        if isinstance(ev, AgentTextDelta):
                            sys.stdout.write(ev.text)
                            sys.stdout.flush()
                        elif isinstance(ev, AgentToolStart):
                            args_json = json.dumps(ev.input)
                            if len(args_json) > 60:
                                args_preview = args_json[:60] + "..."
                            else:
                                args_preview = args_json
                            print(f"\n⚙ {ev.name}({args_preview})...")
                        elif isinstance(ev, AgentToolResult):
                            if ev.is_error:
                                print(f"✗ {ev.name} failed")
                            else:
                                print(f"✓ {ev.name} done")
                        elif isinstance(ev, AgentDone):
                            print("\n\n", end="")
                except KeyboardInterrupt:
                    print("\n[Interrupted]")
                    continue
