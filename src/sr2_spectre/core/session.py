"""Session management — spectre owns session identity and conversation history."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Session"]


@dataclass
class Session:
    """A spectre session — identity, history, and per-session state.

    Spectre owns the session_id string and maintains conversation history
    as a list of messages. On each turn, the full history is sent to relay.
    """

    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append_user(self, text: str) -> None:
        """Append a user message to history."""
        self.history.append({"role": "user", "content": text})

    def append_assistant(self, content: list[dict[str, Any]]) -> None:
        """Append an assistant response (text + tool calls) to history."""
        self.history.append({"role": "assistant", "content": content})

    def append_tool_result(self, tool_use_id: str, content: str, is_error: bool = False) -> None:
        """Append a tool result to history."""
        self.history.append({
            "role": "tool",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        })

    def get_messages(self) -> list[dict[str, Any]]:
        """Return the full message history."""
        return self.history

    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    @property
    def turn_count(self) -> int:
        """Count user turns in this session."""
        return sum(1 for m in self.history if m["role"] == "user")
