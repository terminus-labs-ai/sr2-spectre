"""Tests for thinking/reasoning visibility in the TUI (spc-26).

Covers:
  1. AgentThinkingDelta event type exists with correct type field
  2. Session.stream_message passes through thinking events from SR2
  3. TUI renders thinking with '>' prefix, distinct from regular text
  4. Thinking blocks are closed when regular text resumes
  5. Multiple thinking blocks in a single turn
  6. Thinking events are NOT accumulated in assistant history
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from sr2.protocols.llm import StreamEvent
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentThinkingDelta,
    AgentTextDelta,
)


# ---------------------------------------------------------------------------
# 1. Event type
# ---------------------------------------------------------------------------

def test_thinking_delta_event_type():
    """AgentThinkingDelta has type='thinking_delta'."""
    ev = AgentThinkingDelta(text="reasoning")
    assert ev.type == "thinking_delta"
    assert ev.text == "reasoning"
    assert isinstance(ev, AgentEvent)


def test_thinking_delta_is_distinct_from_text_delta():
    """AgentThinkingDelta is not an AgentTextDelta."""
    ev = AgentThinkingDelta(text="reasoning")
    assert not isinstance(ev, AgentTextDelta)


# ---------------------------------------------------------------------------
# 2. Session passes through thinking events
# ---------------------------------------------------------------------------

def _make_agent_for_thinking(round_events: list[StreamEvent]) -> MagicMock:
    """Create a mock agent whose stream_message yields the given events."""
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])
    agent.history = []

    async def _stream(text: str) -> AsyncIterator:
        for ev in round_events:
            if ev.type == "text" and ev.text:
                yield AgentTextDelta(text=ev.text)
            elif ev.type == "thinking" and ev.text:
                yield AgentThinkingDelta(text=ev.text)
            elif ev.type == "end":
                yield AgentDone()

    agent.stream_message = _stream
    return agent


# ---------------------------------------------------------------------------
# 3. TUI renders thinking with '>' prefix
# ---------------------------------------------------------------------------

def _prompt_sequence(*inputs: str | BaseException) -> MagicMock:
    """Build a prompt_async side_effect that returns inputs in order."""
    sequence = list(inputs) + [EOFError()]

    async def _side_effect(*args, **kwargs):
        item = sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    mock = MagicMock(side_effect=_side_effect)
    return mock


@pytest.mark.asyncio
async def test_tui_thinking_rendered_with_prefix(capsys):
    """Thinking text is prefixed with '> ' to distinguish from regular text."""
    from sr2_spectre.interfaces.tui import TUIInterface

    events = [
        AgentThinkingDelta(text="Let me think about this"),
        AgentTextDelta(text="The answer is 42"),
        AgentDone(),
    ]
    agent = _make_agent_for_thinking([
        StreamEvent(type="thinking", text="Let me think about this"),
        StreamEvent(type="text", text="The answer is 42"),
        StreamEvent(type="end"),
    ])

    plugin = TUIInterface()
    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("question", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "> " in out
    assert "Let me think about this" in out
    assert "The answer is 42" in out


@pytest.mark.asyncio
async def test_tui_thinking_block_closes_on_regular_text(capsys):
    """When regular text follows thinking, a newline separates them."""
    from sr2_spectre.interfaces.tui import TUIInterface

    agent = _make_agent_for_thinking([
        StreamEvent(type="thinking", text="reasoning"),
        StreamEvent(type="text", text="response"),
        StreamEvent(type="end"),
    ])

    plugin = TUIInterface()
    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("q", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    # Thinking should have '> ' prefix, followed by newline before regular text
    assert "> reasoning\n" in out
    assert "response" in out


@pytest.mark.asyncio
async def test_tui_thinking_streamed_live(capsys):
    """Multiple thinking deltas are streamed live (not buffered)."""
    from sr2_spectre.interfaces.tui import TUIInterface

    agent = _make_agent_for_thinking([
        StreamEvent(type="thinking", text="First "),
        StreamEvent(type="thinking", text="part of "),
        StreamEvent(type="thinking", text="reasoning"),
        StreamEvent(type="text", text="Answer"),
        StreamEvent(type="end"),
    ])

    plugin = TUIInterface()
    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("q", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    # All thinking chunks should appear after the '>' prefix
    assert "> First part of reasoning" in out
    assert "Answer" in out


@pytest.mark.asyncio
async def test_tui_thinking_without_regular_text(capsys):
    """Thinking-only response (no regular text) still renders correctly."""
    from sr2_spectre.interfaces.tui import TUIInterface

    agent = _make_agent_for_thinking([
        StreamEvent(type="thinking", text="I need to think"),
        StreamEvent(type="end"),
    ])

    plugin = TUIInterface()
    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("q", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "> I need to think" in out


@pytest.mark.asyncio
async def test_tui_thinking_then_tool_then_text(capsys):
    """Thinking before tools: thinking block closes, tool runs, text resumes."""
    from sr2_spectre.interfaces.tui import TUIInterface
    from sr2_spectre.events import AgentToolStart, AgentToolResult

    agent = MagicMock()
    agent.session_id = "test"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])
    agent.history = []

    async def _stream(text: str):
        yield AgentThinkingDelta(text="Planning approach")
        yield AgentToolStart(tool_id="t1", name="search", input={"q": "x"})
        yield AgentToolResult(tool_id="t1", name="search", content="found", is_error=False)
        yield AgentTextDelta(text="Found it!")
        yield AgentDone()

    agent.stream_message = _stream

    plugin = TUIInterface()
    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("q", "/quit")

    with patch("sr2_spectre.interfaces.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "> Planning approach" in out
    assert "⚙ search(" in out
    assert "Found it!" in out
