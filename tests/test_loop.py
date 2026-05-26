"""Tests for TurnResult (core/loop.py).

The run_tool_loop() function has been removed — logic now lives in
Agent.handle_user_message(). This file tests only the TurnResult dataclass,
which is the public export plugins depend on.
"""

from sr2_spectre.core.loop import TurnResult


def test_turn_result_defaults() -> None:
    r = TurnResult(text="hello")
    assert r.text == "hello"
    assert r.tool_calls_executed == 0
    assert r.total_tokens == 0


def test_turn_result_with_values() -> None:
    r = TurnResult(text="answer", tool_calls_executed=2, total_tokens=200)
    assert r.text == "answer"
    assert r.tool_calls_executed == 2
    assert r.total_tokens == 200
