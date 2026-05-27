"""Spectre core — TurnResult and shared loop types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnResult:
    """Result of a complete agent turn."""
    text: str
    tool_calls_executed: int = 0
    total_tokens: int = 0


__all__ = ["TurnResult"]
