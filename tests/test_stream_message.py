"""Tests for Agent.stream_message() — streaming event API (obsidian-556).

Covers:
  A. Text-only response: yields AgentTextDelta(s) then AgentDone(tool_calls_executed=0)
  B. AgentDone is always last event, even on empty text
  C. Tool call sequence: AgentToolStart → AgentToolResult → ... → AgentDone
  D. tool_calls_executed counter in AgentDone is correct
  E. Tool errors yield AgentToolResult(is_error=True) — not raised
  F. max_tool_rounds exceeded: AgentDone still emitted
  G. History is updated the same as handle_user_message()
  H. handle_user_message() still works (re-implemented on top of stream_message())
  I. Multi-round: text + tool + text emitted in correct order

NOTE: Imports from sr2_spectre.events will fail until that module is implemented.
This is intentional — these are TDD tests written before the implementation exists.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from sr2.config.models import ToolLoopLimitError
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.protocols.llm import StreamEvent
from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import TurnResult
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)


# ---------------------------------------------------------------------------
# Shared helpers — same patterns as test_agent.py
# ---------------------------------------------------------------------------

def _minimal_pipeline_dict() -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
            },
            {
                "name": "tools",
                "target": "tools",
                "resolvers": [],
                "tool_providers": [{"type": "spectre_tools"}],
            },
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}, {"type": "input"}],
            },
        ]
    }


def _make_config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


def _mock_sr2_with_rounds(*round_event_lists: list[StreamEvent]) -> MagicMock:
    """Return a mock SR2 whose turn() yields successive rounds of events."""
    call_count = 0
    rounds = list(round_event_lists)

    mock_sr2 = MagicMock()
    mock_sr2.seed_session = MagicMock()

    async def _turn(user_input):
        nonlocal call_count
        events = rounds[call_count]
        call_count += 1
        for ev in events:
            yield ev

    mock_sr2.turn = _turn
    return mock_sr2


def _make_agent(mock_sr2: MagicMock, **config_kwargs) -> Agent:
    with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
        return Agent(config=_make_config(**config_kwargs), session_id="test-session")


async def _collect(gen: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [ev async for ev in gen]


# ---------------------------------------------------------------------------
# A. Text-only response
# ---------------------------------------------------------------------------

class TestStreamMessageTextOnly:
    @pytest.mark.asyncio
    async def test_text_only_yields_text_delta_then_done(self):
        """Text-only response: AgentTextDelta(s) followed by AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hello "),
            StreamEvent(type="text", text="world!"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Hi"))

        text_deltas = [e for e in events if isinstance(e, AgentTextDelta)]
        assert len(text_deltas) >= 1
        assert "".join(e.text for e in text_deltas) == "Hello world!"

        done_events = [e for e in events if isinstance(e, AgentDone)]
        assert len(done_events) == 1

        # AgentDone is last
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_text_only_done_has_zero_tool_calls(self):
        """AgentDone.tool_calls_executed is 0 for text-only response."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Question"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 0

    @pytest.mark.asyncio
    async def test_text_delta_type_field_is_text_delta(self):
        """AgentTextDelta.type == 'text_delta'."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Hello"))

        deltas = [e for e in events if isinstance(e, AgentTextDelta)]
        assert all(e.type == "text_delta" for e in deltas)


# ---------------------------------------------------------------------------
# B. AgentDone is always last — including empty text
# ---------------------------------------------------------------------------

class TestStreamMessageAgentDoneAlwaysLast:
    @pytest.mark.asyncio
    async def test_done_emitted_when_text_is_empty(self):
        """AgentDone is emitted even when the LLM yields no text."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Hi"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 0

    @pytest.mark.asyncio
    async def test_done_is_exactly_last_event(self):
        """No events come after AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Stuff"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Prompt"))

        done_indices = [i for i, e in enumerate(events) if isinstance(e, AgentDone)]
        assert len(done_indices) == 1
        assert done_indices[0] == len(events) - 1

    @pytest.mark.asyncio
    async def test_done_type_field_is_done(self):
        """AgentDone.type == 'done'."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="ok"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("test"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.type == "done"


# ---------------------------------------------------------------------------
# C. Tool call sequence
# ---------------------------------------------------------------------------

class TestStreamMessageToolSequence:
    @pytest.mark.asyncio
    async def test_tool_start_emitted_before_execution(self):
        """AgentToolStart is yielded when SR2 emits tool_use_emitted."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="add", input={"a": 1, "b": 2})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="3")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Result is 3"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("1+2?"))

        starts = [e for e in events if isinstance(e, AgentToolStart)]
        assert len(starts) == 1
        assert starts[0].type == "tool_start"
        assert starts[0].tool_id == "tu1"
        assert starts[0].name == "add"
        assert starts[0].input == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_tool_result_emitted_after_execution(self):
        """AgentToolResult is yielded when SR2 emits tool_result_received."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="greet", input={"name": "world"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="Hello world")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Greet"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 1
        assert results[0].type == "tool_result"
        assert results[0].tool_id == "tu1"
        assert results[0].is_error is False
        assert "Hello world" in results[0].content

    @pytest.mark.asyncio
    async def test_tool_start_before_tool_result_in_sequence(self):
        """AgentToolStart always precedes AgentToolResult for the same tool_id."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="ping", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="pong")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Pong"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Ping"))

        start_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolStart))
        result_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolResult))
        assert start_idx < result_idx

    @pytest.mark.asyncio
    async def test_tool_result_before_done(self):
        """AgentToolResult precedes AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="calc", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="ok"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        result_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolResult))
        done_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentDone))
        assert result_idx < done_idx


# ---------------------------------------------------------------------------
# D. tool_calls_executed counter
# ---------------------------------------------------------------------------

class TestStreamMessageToolCallsCounter:
    @pytest.mark.asyncio
    async def test_single_tool_call_counted(self):
        """AgentDone.tool_calls_executed == 1 after one tool call."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="t", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="result")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 1

    @pytest.mark.asyncio
    async def test_two_tool_calls_in_one_round_counted(self):
        """Two tool_use blocks in one iteration -> tool_calls_executed == 2."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[
                    ToolUseBlock(id="tu1", name="t1", input={}),
                    ToolUseBlock(id="tu2", name="t2", input={}),
                ],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[
                    ToolResultBlock(tool_use_id="tu1", content="r1"),
                    ToolResultBlock(tool_use_id="tu2", content="r2"),
                ],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="both done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 2

        starts = [e for e in events if isinstance(e, AgentToolStart)]
        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(starts) == 2
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_tool_calls_across_multiple_iterations_counted(self):
        """Tool calls across two iterations sum correctly in AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="t1", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="result")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="t1", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="result")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(type="text", text="final"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 2


# ---------------------------------------------------------------------------
# E. Tool errors
# ---------------------------------------------------------------------------

class TestStreamMessageToolErrors:
    @pytest.mark.asyncio
    async def test_tool_error_yields_tool_result_with_is_error_true(self):
        """Failing tool yields AgentToolResult(is_error=True), does not raise."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="boom", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="kaboom", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="recovered"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("trigger"))

        error_results = [
            e for e in events
            if isinstance(e, AgentToolResult) and e.is_error
        ]
        assert len(error_results) == 1
        assert "kaboom" in error_results[0].content

    @pytest.mark.asyncio
    async def test_unregistered_tool_name_yields_is_error_true(self):
        """Tool result with is_error=True yields AgentToolResult(is_error=True)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="no_such_tool", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="tool not found", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="recovered"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("trigger"))

        error_results = [e for e in events if isinstance(e, AgentToolResult) and e.is_error]
        assert len(error_results) == 1
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_error_does_not_raise_from_stream_message(self):
        """stream_message() must not raise even when tool result has error."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="crash", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="fatal", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="fine"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("trigger"))
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_successful_tool_yields_is_error_false(self):
        """Successful tool yields AgentToolResult(is_error=False)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="ok_tool", input={"x": 1})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="got 1")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("run"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 1
        assert results[0].is_error is False


# ---------------------------------------------------------------------------
# F. max_tool_rounds exceeded
# ---------------------------------------------------------------------------

class TestStreamMessageMaxToolRounds:
    @pytest.mark.asyncio
    async def test_done_emitted_when_max_tool_rounds_exceeded(self):
        """AgentDone is still the last event when SR2 stops after max iterations."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="still going")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="still going")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu3", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu3", content="still going")],
            ),
            StreamEvent(type="iteration_complete", iteration=2),
            StreamEvent(type="text", text="stopped"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2, max_tool_rounds=3)

        events = await _collect(agent.stream_message("start"))
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_calls_executed_reflects_max_rounds(self):
        """tool_calls_executed reflects the number of tool calls before SR2 stopped."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="ok")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="ok")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2, max_tool_rounds=2)

        events = await _collect(agent.stream_message("start"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 2


# ---------------------------------------------------------------------------
# F2. ToolLoopLimitError raised mid-stream -> graceful stop (obsidian-ydt, Behavior 2)
#
# When self.sr2.turn(...) raises ToolLoopLimitError partway through streaming,
# stream_message must catch it and end gracefully: emit the text/tool events
# seen before the raise, emit a final notice (an AgentTextDelta whose text
# indicates the tool/iteration limit was reached), and ALWAYS still emit
# AgentDone as the final event. No exception may escape stream_message.
# ---------------------------------------------------------------------------

def _mock_sr2_raising_loop_limit() -> MagicMock:
    """Mock SR2 whose turn() yields a tool round, then raises ToolLoopLimitError."""
    mock_sr2 = MagicMock()
    mock_sr2.seed_session = MagicMock()

    async def _turn(user_input):
        yield StreamEvent(
            type="tool_use_emitted",
            tool_uses=[ToolUseBlock(id="tu1", name="loop", input={})],
        )
        yield StreamEvent(
            type="tool_result_received",
            tool_results=[ToolResultBlock(tool_use_id="tu1", content="still going")],
        )
        raise ToolLoopLimitError("tool loop iteration limit reached")

    mock_sr2.turn = _turn
    return mock_sr2


class TestStreamMessageToolLoopLimitError:
    @pytest.mark.asyncio
    async def test_no_exception_escapes_and_done_is_last(self):
        """ToolLoopLimitError must be caught; AgentDone is still the final event."""
        agent = _make_agent(_mock_sr2_raising_loop_limit())

        events = await _collect(agent.stream_message("go"))

        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_calls_before_raise_are_reflected(self):
        """tool_calls_executed reflects the tool calls emitted before the raise (>=1)."""
        agent = _make_agent(_mock_sr2_raising_loop_limit())

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed >= 1

        # The tool events seen before the raise are still surfaced.
        starts = [e for e in events if isinstance(e, AgentToolStart)]
        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(starts) >= 1
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_final_notice_text_delta_emitted(self):
        """At least one AgentTextDelta carries a limit/iteration notice (not exact wording)."""
        agent = _make_agent(_mock_sr2_raising_loop_limit())

        events = await _collect(agent.stream_message("go"))

        delta_texts = [e.text.lower() for e in events if isinstance(e, AgentTextDelta)]
        # NOTE: wording is intentionally not pinned; accept any of these tokens.
        assert any(
            ("limit" in t) or ("iteration" in t) or ("stopped" in t)
            for t in delta_texts
        ), f"No limit/iteration notice found in text deltas: {delta_texts}"


# ---------------------------------------------------------------------------
# G. History
# ---------------------------------------------------------------------------

class TestStreamMessageHistory:
    @pytest.mark.asyncio
    async def test_history_updated_after_stream(self):
        """History has user + assistant messages after stream_message completes."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Response"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("Question"))

        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_tool_result_appended_to_history(self):
        """When tools are used, history still gets user + assistant (SR2 handles tool loop)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="lookup", input={"q": "x"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="The answer is 42"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("What is x?"))

        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_seed_session_called_each_turn(self):
        """SR2.seed_session is called once per stream_message call."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("Hello"))
        agent.sr2.seed_session.assert_called_once()


# ---------------------------------------------------------------------------
# H. handle_user_message on top of stream_message
# ---------------------------------------------------------------------------

class TestHandleUserMessageOnStreamMessage:
    @pytest.mark.asyncio
    async def test_handle_user_message_returns_turn_result(self):
        """handle_user_message still returns TurnResult."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        result = await agent.handle_user_message("Question")
        assert isinstance(result, TurnResult)
        assert result.text == "Answer"

    @pytest.mark.asyncio
    async def test_handle_user_message_with_tool_returns_correct_count(self):
        """handle_user_message counts tool calls from SR2 internal tool loop."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="calc", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="42"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        result = await agent.handle_user_message("calc")
        assert isinstance(result, TurnResult)
        assert result.tool_calls_executed == 1


# ---------------------------------------------------------------------------
# I. Multi-round ordering
# ---------------------------------------------------------------------------

class TestStreamMessageMultiRoundOrdering:
    @pytest.mark.asyncio
    async def test_text_then_tool_then_text_event_order(self):
        """Events appear in correct order: text, tool_start, tool_result, text, done."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Thinking..."),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="lookup", input={"q": "x"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="The answer is 42"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)
        agent.register_tool("lookup", "Lookup", {}, lambda q: "42")

        events = await _collect(agent.stream_message("What is x?"))

        types = [type(e).__name__ for e in events]

        assert "AgentTextDelta" in types
        assert "AgentToolStart" in types
        assert "AgentToolResult" in types
        assert "AgentDone" in types

        text_delta_indices = [i for i, t in enumerate(types) if t == "AgentTextDelta"]
        tool_start_idx = types.index("AgentToolStart")
        tool_result_idx = types.index("AgentToolResult")
        done_idx = types.index("AgentDone")

        assert min(text_delta_indices) < tool_start_idx
        assert tool_start_idx < tool_result_idx
        assert tool_result_idx < done_idx
