"""Tests for TUI session save/load and /history command.

Covers:
1. /save — serializes agent.history to JSON file
2. /load — deserializes JSON file into agent.history
3. /history — prints conversation history summary
4. Status bar after each turn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from sr2_spectre.events import (
    AgentDone,
    AgentTextDelta,
)
from sr2_spectre.interfaces.tui import TUIInterface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(events: list | None = None) -> MagicMock:
    """Return a mock agent whose stream_message() yields the supplied events."""
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.history = []
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=["tool_a"])

    if events is None:
        events = [AgentDone()]

    async def _stream(text: str) -> AsyncIterator:
        for ev in events:
            yield ev

    agent.stream_message = _stream
    return agent


def _prompt_sequence(*inputs: str | BaseException) -> MagicMock:
    """Build a prompt_async mock that returns inputs in order."""
    sequence = list(inputs) + [EOFError()]

    async def _side_effect(*args, **kwargs):
        item = sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    mock = MagicMock()
    mock.side_effect = _side_effect
    return mock


# ---------------------------------------------------------------------------
# /history command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slash_history_prints_history_summary(capsys: pytest.CaptureFixture) -> None:
    """/history must print a summary of the conversation history."""
    plugin = TUIInterface()
    agent = _make_agent()
    # Give the agent some history
    from sr2.models import Message, TextBlock
    agent.history = [
        Message(role="user", content=[TextBlock(text="Hello")]),
        Message(role="assistant", content=[TextBlock(text="Hi there!")]),
    ]

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/history", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "Hello" in out
    assert "Hi there!" in out


@pytest.mark.asyncio
async def test_slash_history_empty(capsys: pytest.CaptureFixture) -> None:
    """/history with empty history prints a message indicating no history."""
    plugin = TUIInterface()
    agent = _make_agent()
    agent.history = []

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/history", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "empty" in out.lower() or "no" in out.lower()


@pytest.mark.asyncio
async def test_slash_history_continues_loop(capsys: pytest.CaptureFixture) -> None:
    """/history must not stop the loop."""
    plugin = TUIInterface()
    events = [AgentTextDelta(text="response"), AgentDone()]
    agent = _make_agent(events)
    agent.history = []

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/history", "ping", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "response" in out


# ---------------------------------------------------------------------------
# /save command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slash_save_serializes_history_to_file(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    """/save path must serialize agent.history to a JSON file."""
    plugin = TUIInterface()
    agent = _make_agent()
    from sr2.models import Message, TextBlock
    agent.history = [
        Message(role="user", content=[TextBlock(text="Hello")]),
        Message(role="assistant", content=[TextBlock(text="Hi!")]),
    ]

    save_path = tmp_path / "session.json"
    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence(f"/save {save_path}", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    assert save_path.exists()
    data = json.loads(save_path.read_text())
    assert data["session_id"] == "test-session"
    assert len(data["history"]) == 2
    assert data["history"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_slash_save_default_path(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    """/save without a path uses a default filename (~/.sr2-spectre/session.json)."""
    plugin = TUIInterface()
    agent = _make_agent()
    from sr2.models import Message, TextBlock
    agent.history = [
        Message(role="user", content=[TextBlock(text="test")]),
    ]

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/save", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        with patch("pathlib.Path.home", return_value=tmp_path):
            await plugin.run(agent)

    # The save should succeed without error
    out = capsys.readouterr().out
    assert "Saved" in out or "saved" in out.lower()

    # Verify default path was created
    default_path = tmp_path / ".sr2-spectre" / "session.json"
    assert default_path.exists()


# ---------------------------------------------------------------------------
# /load command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slash_load_restores_history_from_file(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    """/load path must deserialize JSON file into agent.history."""
    plugin = TUIInterface()
    agent = _make_agent()
    agent.history = []

    # Pre-create the save file
    save_path = tmp_path / "session.json"
    save_data = {
        "session_id": "loaded-session",
        "history": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ],
    }
    save_path.write_text(json.dumps(save_data))

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence(f"/load {save_path}", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "Loaded" in out or "loaded" in out.lower()
    assert len(agent.history) == 2


@pytest.mark.asyncio
async def test_slash_load_missing_file_prints_error(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    """/load with a non-existent path must print an error and continue."""
    plugin = TUIInterface()
    agent = _make_agent()
    agent.history = []

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/load /nonexistent/path.json", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "error" in out.lower() or "not found" in out.lower()


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_bar_shown_after_turn(capsys: pytest.CaptureFixture) -> None:
    """After a turn completes, a status line should appear with session info."""
    plugin = TUIInterface()
    events = [AgentTextDelta(text="response"), AgentDone(tool_calls_executed=1)]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("hello", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    # Status bar should show session ID somewhere
    assert "test-session" in out


@pytest.mark.asyncio
async def test_status_bar_shows_tool_count(capsys: pytest.CaptureFixture) -> None:
    """Status bar should reflect tool_calls_executed from AgentDone."""
    plugin = TUIInterface()
    events = [AgentTextDelta(text="done"), AgentDone(tool_calls_executed=3)]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("go", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "3" in out  # tool count
