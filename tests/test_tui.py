"""Tests for TUIPlugin — streaming display and tool visibility.

Requirements tested:
1.  Plugin interface: name, start, run, stop async methods
2.  Input via prompt_toolkit.PromptSession.prompt_async() with "> " prompt
3.  Streaming output: AgentTextDelta.text written immediately to stdout
4.  Tool start display: "\\n⚙ {name}({args_preview})..."
5.  Tool result display: "✓ {name} done" / "✗ {name} failed"
6.  After response: "\\n\\n" printed after AgentDone
7.  Slash commands: /quit, /exit, /reset, /help, /tools
8.  Empty input: silently skipped
9.  KeyboardInterrupt during prompt: print "\\nInterrupted." and stop
10. EOFError during prompt: print "\\nEOF." and stop
11. KeyboardInterrupt during streaming: print "\\n[Interrupted]", re-prompt
12. stop() sets _running = False; run() exits
"""
from __future__ import annotations

import json
import sys
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.events import (
    AgentDone,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)
from sr2_spectre.plugins.tui import TUIPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(events: list | None = None) -> MagicMock:
    """Return a mock agent whose stream_message() yields the supplied events."""
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=["tool_a", "tool_b"])

    if events is None:
        events = [AgentDone()]

    async def _stream(text: str) -> AsyncIterator:
        for ev in events:
            yield ev

    agent.stream_message = _stream
    return agent


def _prompt_sequence(*inputs: str | BaseException) -> AsyncMock:
    """Build a prompt_async side_effect that returns inputs in order.

    Strings are returned as-is. BaseException subclasses are raised.
    After the sequence is exhausted the mock raises EOFError to stop the loop.
    """
    sequence = list(inputs) + [EOFError()]

    async def _side_effect(*args, **kwargs):
        item = sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item()
        return item

    mock = AsyncMock(side_effect=_side_effect)
    return mock


# ---------------------------------------------------------------------------
# Requirement 1: Plugin interface
# ---------------------------------------------------------------------------

def test_plugin_has_name_attribute() -> None:
    """TUIPlugin must have name = 'tui'."""
    plugin = TUIPlugin()
    assert plugin.name == "tui"


@pytest.mark.asyncio
async def test_start_is_async_and_callable() -> None:
    """start(agent) must be awaitable without raising."""
    plugin = TUIPlugin()
    agent = _make_agent()
    with patch("sr2_spectre.plugins.tui.PromptSession"):
        await plugin.start(agent)


@pytest.mark.asyncio
async def test_stop_is_async_and_sets_running_false() -> None:
    """stop() must be awaitable and set _running = False."""
    plugin = TUIPlugin()
    plugin._running = True
    await plugin.stop()
    assert plugin._running is False


@pytest.mark.asyncio
async def test_run_is_async_method() -> None:
    """run(agent) must be awaitable — terminates when loop ends."""
    plugin = TUIPlugin()
    agent = _make_agent()
    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/quit")
    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)


# ---------------------------------------------------------------------------
# Requirement 2: Input via prompt_toolkit.PromptSession.prompt_async()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_async_called_with_prompt_string(capsys: pytest.CaptureFixture) -> None:
    """prompt_async must be called with "> " as the prompt argument."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    prompt_mock = _prompt_sequence("/quit")
    mock_session.prompt_async = prompt_mock

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    prompt_mock.assert_called_with("> ")


# ---------------------------------------------------------------------------
# Requirement 3: Streaming output — AgentTextDelta
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_delta_written_to_stdout_immediately(capsys: pytest.CaptureFixture) -> None:
    """Each AgentTextDelta.text must appear in stdout without extra newlines between deltas."""
    plugin = TUIPlugin()
    events = [
        AgentTextDelta(text="Hello"),
        AgentTextDelta(text=", "),
        AgentTextDelta(text="world"),
        AgentDone(),
    ]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("say hello", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    # All three fragments must appear, concatenated without extra newlines between them
    assert "Hello, world" in out


@pytest.mark.asyncio
async def test_text_delta_no_buffering_no_extra_newline_between_deltas(
    capsys: pytest.CaptureFixture,
) -> None:
    """No newline must be inserted between adjacent AgentTextDelta events."""
    plugin = TUIPlugin()
    events = [AgentTextDelta(text="foo"), AgentTextDelta(text="bar"), AgentDone()]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("msg", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    # "foobar" must appear as a contiguous substring (no newline between)
    assert "foobar" in out


# ---------------------------------------------------------------------------
# Requirement 4: Tool start display
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_start_displays_name_and_args_preview(capsys: pytest.CaptureFixture) -> None:
    """AgentToolStart must print '\\n⚙ {name}({args_preview})...'."""
    plugin = TUIPlugin()
    tool_input = {"key": "value"}
    events = [
        AgentToolStart(tool_id="t1", name="search", input=tool_input),
        AgentToolResult(tool_id="t1", name="search", content="result", is_error=False),
        AgentDone(),
    ]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("find something", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "⚙ search(" in out
    assert json.dumps(tool_input)[:60] in out
    assert ")..." in out


@pytest.mark.asyncio
async def test_tool_start_args_preview_truncated_at_60_chars(
    capsys: pytest.CaptureFixture,
) -> None:
    """args_preview must be truncated to 60 chars with trailing '...' if longer."""
    plugin = TUIPlugin()
    # Construct an input that serialises to >60 chars
    long_input = {"query": "x" * 100}
    serialised = json.dumps(long_input)
    assert len(serialised) > 60

    events = [
        AgentToolStart(tool_id="t1", name="search", input=long_input),
        AgentToolResult(tool_id="t1", name="search", content="ok", is_error=False),
        AgentDone(),
    ]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("go", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    # The preview is the first 60 chars of the JSON, then "..."
    expected_preview = serialised[:60] + "..."
    assert expected_preview in out


@pytest.mark.asyncio
async def test_tool_start_args_not_truncated_when_short(
    capsys: pytest.CaptureFixture,
) -> None:
    """When args JSON is ≤60 chars, no truncation ellipsis is appended mid-preview."""
    plugin = TUIPlugin()
    short_input = {"k": "v"}
    serialised = json.dumps(short_input)
    assert len(serialised) <= 60

    events = [
        AgentToolStart(tool_id="t1", name="fetch", input=short_input),
        AgentToolResult(tool_id="t1", name="fetch", content="ok", is_error=False),
        AgentDone(),
    ]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("go", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    # Full JSON must appear, no truncation marker appended after it inside the parens
    assert f"fetch({serialised})..." in out


# ---------------------------------------------------------------------------
# Requirement 5: Tool result display
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_result_success_displays_checkmark(capsys: pytest.CaptureFixture) -> None:
    """AgentToolResult with is_error=False must print '✓ {name} done'."""
    plugin = TUIPlugin()
    events = [
        AgentToolStart(tool_id="t1", name="lookup", input={}),
        AgentToolResult(tool_id="t1", name="lookup", content="ok", is_error=False),
        AgentDone(),
    ]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("go", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "✓ lookup done" in out


@pytest.mark.asyncio
async def test_tool_result_error_displays_cross(capsys: pytest.CaptureFixture) -> None:
    """AgentToolResult with is_error=True must print '✗ {name} failed'."""
    plugin = TUIPlugin()
    events = [
        AgentToolStart(tool_id="t1", name="lookup", input={}),
        AgentToolResult(tool_id="t1", name="lookup", content="err", is_error=True),
        AgentDone(),
    ]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("go", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "✗ lookup failed" in out


# ---------------------------------------------------------------------------
# Requirement 6: After response — two newlines after AgentDone
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_newlines_printed_after_agent_done(capsys: pytest.CaptureFixture) -> None:
    """After AgentDone, '\\n\\n' must appear in stdout to separate from next prompt."""
    plugin = TUIPlugin()
    events = [AgentTextDelta(text="hi"), AgentDone()]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("hello", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "\n\n" in out
    assert out.index("hi") < out.index("\n\n")


# ---------------------------------------------------------------------------
# Requirement 7: Slash commands
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slash_quit_stops_loop_and_prints_goodbye(capsys: pytest.CaptureFixture) -> None:
    """/quit must print 'Goodbye.' and exit the loop."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "Goodbye." in out


@pytest.mark.asyncio
async def test_slash_exit_stops_loop_and_prints_goodbye(capsys: pytest.CaptureFixture) -> None:
    """/exit must behave identically to /quit."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/exit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "Goodbye." in out


@pytest.mark.asyncio
async def test_slash_quit_does_not_call_stream_message() -> None:
    """/quit must not invoke stream_message on the agent."""
    plugin = TUIPlugin()
    stream_called = False

    agent = MagicMock()
    agent.session_id = "test"
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])

    async def _stream(text: str):
        nonlocal stream_called
        stream_called = True
        yield AgentDone()

    agent.stream_message = _stream

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    assert not stream_called


@pytest.mark.asyncio
async def test_slash_reset_calls_new_session_and_prints_confirmation(
    capsys: pytest.CaptureFixture,
) -> None:
    """/reset must call agent.new_session() and print 'Session reset.'."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/reset", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    agent.new_session.assert_called_once()
    out = capsys.readouterr().out
    assert "Session reset." in out


@pytest.mark.asyncio
async def test_slash_reset_continues_loop_after_reset(capsys: pytest.CaptureFixture) -> None:
    """/reset must continue the loop — subsequent inputs are still processed."""
    plugin = TUIPlugin()
    events = [AgentTextDelta(text="response"), AgentDone()]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/reset", "hello", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "response" in out


@pytest.mark.asyncio
async def test_slash_help_prints_command_list(capsys: pytest.CaptureFixture) -> None:
    """/help must print a help string listing /quit, /reset, /help."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/help", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "/quit" in out
    assert "/reset" in out
    assert "/help" in out
    assert "/tools" in out
    assert "/exit" in out


@pytest.mark.asyncio
async def test_slash_help_continues_loop(capsys: pytest.CaptureFixture) -> None:
    """/help must not stop the loop."""
    plugin = TUIPlugin()
    events = [AgentTextDelta(text="pong"), AgentDone()]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/help", "ping", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "pong" in out


@pytest.mark.asyncio
async def test_slash_tools_prints_tool_names(capsys: pytest.CaptureFixture) -> None:
    """/tools must print str(agent.registry.list_names())."""
    plugin = TUIPlugin()
    agent = _make_agent()
    agent.registry.list_names.return_value = ["bash", "read_file"]

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/tools", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert str(["bash", "read_file"]) in out


@pytest.mark.asyncio
async def test_slash_tools_continues_loop(capsys: pytest.CaptureFixture) -> None:
    """/tools must not stop the loop."""
    plugin = TUIPlugin()
    events = [AgentTextDelta(text="done"), AgentDone()]
    agent = _make_agent(events)

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("/tools", "work", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "done" in out


# ---------------------------------------------------------------------------
# Requirement 8: Empty input silently skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_input_is_silently_skipped(capsys: pytest.CaptureFixture) -> None:
    """Empty string input must not trigger a stream_message call or any output."""
    plugin = TUIPlugin()
    stream_call_count = 0

    agent = MagicMock()
    agent.session_id = "test"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])

    async def _stream(text: str):
        nonlocal stream_call_count
        stream_call_count += 1
        yield AgentDone()

    agent.stream_message = _stream

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("", "   ", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    assert stream_call_count == 0


@pytest.mark.asyncio
async def test_whitespace_only_input_is_silently_skipped(capsys: pytest.CaptureFixture) -> None:
    """Whitespace-only input must also be skipped — same as empty."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("   ", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    # No response text in output since stream was never called
    out = capsys.readouterr().out
    assert "Goodbye." in out  # loop ended cleanly


# ---------------------------------------------------------------------------
# Requirement 9: KeyboardInterrupt during prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keyboard_interrupt_during_prompt_prints_interrupted(
    capsys: pytest.CaptureFixture,
) -> None:
    """KeyboardInterrupt during prompt must print '\\nInterrupted.' and stop the loop."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence(KeyboardInterrupt())

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "\nInterrupted." in out


@pytest.mark.asyncio
async def test_keyboard_interrupt_during_prompt_stops_loop() -> None:
    """KeyboardInterrupt during prompt must terminate run() cleanly (no crash)."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence(KeyboardInterrupt())

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        # Must not raise
        await plugin.run(agent)


# ---------------------------------------------------------------------------
# Requirement 10: EOFError during prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eoferror_during_prompt_prints_eof(capsys: pytest.CaptureFixture) -> None:
    """EOFError during prompt must print '\\nEOF.' and stop the loop."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence(EOFError())

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "\nEOF." in out


@pytest.mark.asyncio
async def test_eoferror_during_prompt_stops_loop_cleanly() -> None:
    """EOFError during prompt must terminate run() without raising."""
    plugin = TUIPlugin()
    agent = _make_agent()

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence(EOFError())

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)


# ---------------------------------------------------------------------------
# Requirement 11: KeyboardInterrupt during streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keyboard_interrupt_during_streaming_prints_interrupted_marker(
    capsys: pytest.CaptureFixture,
) -> None:
    """KeyboardInterrupt mid-stream must print '\\n[Interrupted]' in stdout."""
    plugin = TUIPlugin()

    agent = MagicMock()
    agent.session_id = "test"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])

    async def _stream_raises(text: str):
        yield AgentTextDelta(text="partial")
        raise KeyboardInterrupt()

    agent.stream_message = _stream_raises

    mock_session = MagicMock()
    mock_session.prompt_async = _prompt_sequence("go", "/quit")

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "\n[Interrupted]" in out


@pytest.mark.asyncio
async def test_keyboard_interrupt_during_streaming_loop_continues(
    capsys: pytest.CaptureFixture,
) -> None:
    """KeyboardInterrupt mid-stream must NOT stop the TUI loop — subsequent prompts work."""
    plugin = TUIPlugin()

    agent = MagicMock()
    agent.session_id = "test"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])

    call_count = 0

    async def _stream_raises_first(text: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield AgentTextDelta(text="partial")
            raise KeyboardInterrupt()
        else:
            yield AgentTextDelta(text="second response")
            yield AgentDone()

    agent.stream_message = _stream_raises_first

    mock_session = MagicMock()
    prompt_async = _prompt_sequence("first", "second", "/quit")
    mock_session.prompt_async = prompt_async

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await plugin.run(agent)

    out = capsys.readouterr().out
    assert "second response" in out
    assert call_count == 2
    assert prompt_async.call_count >= 3


# ---------------------------------------------------------------------------
# Requirement 12: stop() exits run() loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_exits_when_stop_called_between_prompts() -> None:
    """run() must exit its loop once _running is False after a prompt cycle."""
    plugin = TUIPlugin()

    agent = MagicMock()
    agent.session_id = "test"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])

    call_count = 0

    async def _stream(text: str):
        yield AgentDone()

    agent.stream_message = _stream

    # After first real input, stop() will be called; second prompt must not be reached
    second_prompt_called = False

    async def _prompt_side_effect(*args, **kwargs):
        nonlocal second_prompt_called, call_count
        call_count += 1
        if call_count == 1:
            return "hello"
        else:
            second_prompt_called = True
            raise EOFError()

    # Patch stop to be called right after the first message is processed
    original_run = plugin.run

    async def _patched_run(agent):
        # We'll set _running = False right before re-prompting by overriding the agent stream
        async def _stream_and_stop(text: str):
            yield AgentDone()
            plugin._running = False

        agent.stream_message = _stream_and_stop
        await original_run(agent)

    mock_session = MagicMock()
    mock_session.prompt_async = AsyncMock(side_effect=_prompt_side_effect)

    with patch("sr2_spectre.plugins.tui.PromptSession", return_value=mock_session):
        await _patched_run(agent)

    assert not second_prompt_called
