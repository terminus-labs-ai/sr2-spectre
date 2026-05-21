"""Tests for Session."""
from sr2_spectre.core.session import Session


def test_session_creation() -> None:
    s = Session(session_id="test-123")
    assert s.session_id == "test-123"
    assert s.history == []
    assert s.turn_count == 0


def test_append_user() -> None:
    s = Session(session_id="test")
    s.append_user("Hello")
    assert len(s.history) == 1
    assert s.history[0]["role"] == "user"
    assert s.history[0]["content"] == "Hello"
    assert s.turn_count == 1


def test_append_assistant() -> None:
    s = Session(session_id="test")
    s.append_assistant([{"type": "text", "text": "Hi there"}])
    assert len(s.history) == 1
    assert s.history[0]["role"] == "assistant"


def test_append_tool_result() -> None:
    s = Session(session_id="test")
    s.append_tool_result("tool-1", "42")
    assert len(s.history) == 1
    assert s.history[0]["role"] == "tool"
    assert s.history[0]["tool_use_id"] == "tool-1"
    assert s.history[0]["content"] == "42"
    assert not s.history[0]["is_error"]


def test_append_tool_result_error() -> None:
    s = Session(session_id="test")
    s.append_tool_result("tool-2", "timeout", is_error=True)
    assert s.history[0]["is_error"] is True


def test_clear() -> None:
    s = Session(session_id="test")
    s.append_user("a")
    s.append_assistant([{"type": "text", "text": "b"}])
    s.clear()
    assert s.history == []
    assert s.turn_count == 0


def test_turn_count() -> None:
    s = Session(session_id="test")
    s.append_user("first")
    s.append_assistant([{"type": "text", "text": "response"}])
    s.append_user("second")
    assert s.turn_count == 2


def test_metadata() -> None:
    s = Session(session_id="test", metadata={"source": "tui"})
    assert s.metadata["source"] == "tui"
