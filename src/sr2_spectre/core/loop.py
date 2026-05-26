"""TurnResult — the return type for a completed agent turn.

The tool execution loop that previously lived here has been replaced by
Agent.handle_user_message() which delegates all LLM execution to SR2.
This module is retained for its TurnResult export, which plugins depend on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnResult:
    """Result of a complete agent turn."""
    text: str
    tool_calls_executed: int = 0
    total_tokens: int = 0
