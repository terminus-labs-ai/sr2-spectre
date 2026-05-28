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
                "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
            },
            {
                "name": "tools",
                "resolvers": [],
                "tool_providers": [{"type": "spectre_tools"}],
            },
            {
                "name": "conversation",
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
        """AgentToolStart is yielded before the tool executes."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="add", tool_input={"a": 1, "b": 2}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="Result is 3"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("add", "Add", {}, lambda a, b: str(a + b))

        events = await _collect(agent.stream_message("1+2?"))

        starts = [e for e in events if isinstance(e, AgentToolStart)]
        assert len(starts) == 1
        assert starts[0].type == "tool_start"
        assert starts[0].tool_id == "tu1"
        assert starts[0].name == "add"
        assert starts[0].input == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_tool_result_emitted_after_execution(self):
        """AgentToolResult is yielded after the tool executes."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="greet", tool_input={"name": "world"}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="Done"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("greet", "Greet", {}, lambda name: f"Hello {name}")

        events = await _collect(agent.stream_message("Greet"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 1
        assert results[0].type == "tool_result"
        assert results[0].tool_id == "tu1"
        assert results[0].name == "greet"
        assert results[0].is_error is False
        assert "Hello world" in results[0].content

    @pytest.mark.asyncio
    async def test_tool_start_before_tool_result_in_sequence(self):
        """AgentToolStart always precedes AgentToolResult for the same tool_id."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="ping", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="Pong"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("ping", "Ping", {}, lambda: "pong")

        events = await _collect(agent.stream_message("Ping"))

        start_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolStart))
        result_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolResult))
        assert start_idx < result_idx

    @pytest.mark.asyncio
    async def test_tool_result_before_done(self):
        """AgentToolResult precedes AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="calc", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="ok"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("calc", "Calc", {}, lambda: "42")

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
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="t", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="done"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("t", "T", {}, lambda: "result")

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 1

    @pytest.mark.asyncio
    async def test_two_tool_calls_in_one_round_counted(self):
        """Two tool_use blocks in one LLM round → tool_calls_executed == 2."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="t1", tool_input={}),
                StreamEvent(type="tool_use", tool_use_id="tu2", tool_name="t2", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="both done"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("t1", "T1", {}, lambda: "r1")
        agent.register_tool("t2", "T2", {}, lambda: "r2")

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 2

        starts = [e for e in events if isinstance(e, AgentToolStart)]
        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(starts) == 2
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_tool_calls_across_multiple_rounds_counted(self):
        """Tool calls across two rounds sum correctly in AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="t1", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="tool_use", tool_use_id="tu2", tool_name="t1", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="final"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("t1", "T1", {}, lambda: "result")

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
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="boom", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="recovered"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)

        def _fail(**kw):
            raise ValueError("kaboom")

        agent.register_tool("boom", "Boom", {}, _fail)

        events = await _collect(agent.stream_message("trigger"))

        error_results = [
            e for e in events
            if isinstance(e, AgentToolResult) and e.is_error
        ]
        assert len(error_results) == 1
        assert "kaboom" in error_results[0].content

    @pytest.mark.asyncio
    async def test_unregistered_tool_name_yields_is_error_true(self):
        """LLM emitting a tool_use for an unregistered name yields AgentToolResult(is_error=True), not a raise."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="no_such_tool", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="recovered"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        # Deliberately register NO tool named "no_such_tool" — registry raises KeyError

        events = await _collect(agent.stream_message("trigger"))

        error_results = [e for e in events if isinstance(e, AgentToolResult) and e.is_error]
        assert len(error_results) == 1
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_error_does_not_raise_from_stream_message(self):
        """stream_message() must not raise even when a tool throws."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="crash", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="fine"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("crash", "Crash", {}, lambda: (_ for _ in ()).throw(Exception("fatal")))

        # Must not raise
        events = await _collect(agent.stream_message("trigger"))
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_successful_tool_yields_is_error_false(self):
        """Successful tool yields AgentToolResult(is_error=False)."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="ok_tool", tool_input={"x": 1}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="done"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("ok_tool", "OK", {}, lambda x: f"got {x}")

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
        """AgentDone is still the last event when max_tool_rounds is hit."""
        always_tool = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="loop", tool_input={}),
            StreamEvent(type="end"),
        ]
        mock_sr2 = _mock_sr2_with_rounds(*([always_tool] * 10))
        agent = _make_agent(mock_sr2, max_tool_rounds=3)
        agent.register_tool("loop", "Loop", {}, lambda: "still going")

        events = await _collect(agent.stream_message("start"))

        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_calls_executed_reflects_max_rounds(self):
        """tool_calls_executed counts all calls made up to max_tool_rounds."""
        always_tool = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="loop", tool_input={}),
            StreamEvent(type="end"),
        ]
        mock_sr2 = _mock_sr2_with_rounds(*([always_tool] * 10))
        agent = _make_agent(mock_sr2, max_tool_rounds=3)
        agent.register_tool("loop", "Loop", {}, lambda: "ok")

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 3
        # seed_session must have been called exactly max_tool_rounds times — no extra rounds
        assert mock_sr2.seed_session.call_count == 3

    @pytest.mark.asyncio
    async def test_stream_message_does_not_hang_at_max_rounds(self):
        """stream_message() terminates (not an infinite loop) when max_tool_rounds hit."""
        always_tool = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="inf", tool_input={}),
            StreamEvent(type="end"),
        ]
        mock_sr2 = _mock_sr2_with_rounds(*([always_tool] * 20))
        agent = _make_agent(mock_sr2, max_tool_rounds=2)
        agent.register_tool("inf", "Inf", {}, lambda: "x")

        # Will raise asyncio.TimeoutError if it hangs — normal path just collects events
        events = await _collect(agent.stream_message("run"))
        assert isinstance(events[-1], AgentDone)


# ---------------------------------------------------------------------------
# G. History updated same as handle_user_message()
# ---------------------------------------------------------------------------

class TestStreamMessageHistory:
    @pytest.mark.asyncio
    async def test_user_message_appended_to_history(self):
        """stream_message() appends the user message to agent.history."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("Hello"))

        assert len(agent.history) >= 1
        assert agent.history[0].role == "user"

    @pytest.mark.asyncio
    async def test_assistant_response_appended_to_history(self):
        """stream_message() appends the assistant response to agent.history."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Response"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("Question"))

        assert len(agent.history) == 2
        assert agent.history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_tool_result_appended_to_history(self):
        """Tool result is appended to history as user message with ToolResultBlock."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="echo", tool_input={"msg": "hi"}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="done"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("echo", "Echo", {}, lambda msg: msg)

        await _collect(agent.stream_message("Echo hi"))

        # history: user → assistant(tool_use) → user(tool_result) → assistant(text)
        assert len(agent.history) == 4
        tool_result_msg = agent.history[2]
        assert tool_result_msg.role == "user"
        assert any(isinstance(b, ToolResultBlock) for b in tool_result_msg.content)

    @pytest.mark.asyncio
    async def test_seed_session_called_each_round(self):
        """sr2.seed_session() is called on each round, same as handle_user_message()."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="t", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="ok"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("t", "T", {}, lambda: "r")

        await _collect(agent.stream_message("go"))

        # Two rounds → seed_session called twice
        assert mock_sr2.seed_session.call_count == 2


# ---------------------------------------------------------------------------
# H. handle_user_message() still works on top of stream_message()
# ---------------------------------------------------------------------------

class TestHandleUserMessageOnStreamMessage:
    @pytest.mark.asyncio
    async def test_handle_user_message_returns_turn_result(self):
        """handle_user_message() returns TurnResult with text and count."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hello "),
            StreamEvent(type="text", text="there!"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        result = await agent.handle_user_message("Hi")

        assert isinstance(result, TurnResult)
        assert result.text == "Hello there!"
        assert result.tool_calls_executed == 0

    @pytest.mark.asyncio
    async def test_handle_user_message_with_tool_returns_correct_count(self):
        """handle_user_message() reports correct tool_calls_executed from stream."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="t", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="done"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("t", "T", {}, lambda: "result")

        result = await agent.handle_user_message("go")

        assert isinstance(result, TurnResult)
        assert result.tool_calls_executed == 1

    @pytest.mark.asyncio
    async def test_handle_user_message_history_matches_stream_message_history(self):
        """Both APIs produce identical history structure for equivalent inputs."""
        events_round = [
            StreamEvent(type="text", text="Answer"),
            StreamEvent(type="end"),
        ]

        mock_sr2_a = _mock_sr2_with_rounds(events_round)
        agent_a = _make_agent(mock_sr2_a)
        await _collect(agent_a.stream_message("Question"))

        mock_sr2_b = _mock_sr2_with_rounds(events_round)
        agent_b = _make_agent(mock_sr2_b)
        await agent_b.handle_user_message("Question")

        assert len(agent_a.history) == len(agent_b.history)
        for msg_a, msg_b in zip(agent_a.history, agent_b.history):
            assert msg_a.role == msg_b.role
            assert len(msg_a.content) == len(msg_b.content)
            for blk_a, blk_b in zip(msg_a.content, msg_b.content):
                assert type(blk_a) == type(blk_b), (
                    f"Block type mismatch for role={msg_a.role}: "
                    f"{type(blk_a).__name__} vs {type(blk_b).__name__}"
                )


# ---------------------------------------------------------------------------
# I. Multi-round: correct event ordering
# ---------------------------------------------------------------------------

class TestStreamMessageMultiRoundOrdering:
    @pytest.mark.asyncio
    async def test_text_then_tool_then_text_event_order(self):
        """Full multi-round: text_delta → tool_start → tool_result → text_delta → done."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="text", text="Let me check."),
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="lookup", tool_input={"q": "x"}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="The answer is 42."),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("lookup", "Lookup", {}, lambda q: "42")

        events = await _collect(agent.stream_message("What is x?"))

        types = [type(e).__name__ for e in events]

        # Must contain these types in this relative order
        assert "AgentTextDelta" in types
        assert "AgentToolStart" in types
        assert "AgentToolResult" in types
        assert types[-1] == "AgentDone"

        delta_idx = types.index("AgentTextDelta")
        start_idx = types.index("AgentToolStart")
        result_idx = types.index("AgentToolResult")
        done_idx = len(types) - 1

        assert delta_idx < start_idx < result_idx < done_idx

    @pytest.mark.asyncio
    async def test_all_events_are_agent_event_instances(self):
        """Every yielded object is an instance of AgentEvent."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="t", tool_input={}),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="ok"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)
        agent.register_tool("t", "T", {}, lambda: "r")

        events = await _collect(agent.stream_message("go"))

        for ev in events:
            assert isinstance(ev, AgentEvent), f"Expected AgentEvent, got {type(ev)}"

    @pytest.mark.asyncio
    async def test_stream_message_is_async_generator(self):
        """stream_message() returns an async iterable (not a coroutine)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="hi"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        gen = agent.stream_message("test")
        # Must be an async iterator, not a coroutine
        import inspect
        assert hasattr(gen, "__aiter__"), "stream_message() must return an async iterable"
        await _collect(gen)
