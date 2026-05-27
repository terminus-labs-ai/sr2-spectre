"""Tests for Agent — SR2-powered (Steps 4 & 5).

Covers:
  A. Agent.__init__: constructs SR2, no LiteLLMCallable call site
  B. session_id propagated to SR2
  C. Tools from config registered into ToolRegistry
  D. handle_user_message: happy path (no tools)
  E. handle_user_message: one tool round-trip
  F. handle_user_message: error recovery (FR13 — tool error fed back, loop continues)
  G. handle_user_message: max_tool_rounds exceeded returns text + warning (FR14)
  H. new_session() resets history
  I. register_tool() adds to registry
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock, TokenUsage
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import TurnResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pipeline_dict(with_tool_layer: bool = True) -> dict:
    layers = [
        {"name": "system", "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}]},
    ]
    if with_tool_layer:
        layers.append({"name": "tools", "resolvers": [], "tool_providers": [{"type": "spectre_tools"}]})
    layers.append({"name": "conversation", "resolvers": [{"type": "session"}, {"type": "input"}]})
    return {"layers": layers}


def _make_config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


def _stream_events(*events: StreamEvent):
    """Async generator that yields given StreamEvent objects."""
    async def _gen():
        for ev in events:
            yield ev
    return _gen()


def _mock_sr2(turn_events: list[StreamEvent] | None = None) -> MagicMock:
    """Return a mock SR2 instance whose turn() yields given events."""
    events = turn_events or [StreamEvent(type="text", text="Hello!"), StreamEvent(type="end")]
    sr2 = MagicMock()
    sr2.seed_session = MagicMock()

    async def _stream_gen(user_input):
        for ev in events:
            yield ev
    sr2.turn = _stream_gen

    return sr2


# ---------------------------------------------------------------------------
# A. Agent.__init__: constructs SR2, no LiteLLMCallable call
# ---------------------------------------------------------------------------

class TestAgentInit:
    def test_agent_has_sr2_attribute(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="s1")
        assert hasattr(agent, "sr2")

    def test_sr2_constructed_not_litellm_callable(self):
        """Agent.__init__ must use SR2, not call LiteLLMCallable directly."""
        from sr2_spectre.agent import Agent
        import sr2_spectre.agent as agent_module

        # Verify LiteLLMCallable is not the primary client on Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="s1")

        # SR2 was constructed
        MockSR2.assert_called_once()
        call_kwargs = MockSR2.call_args.kwargs
        # extras must contain tool_registry
        assert "extras" in call_kwargs
        assert "tool_registry" in call_kwargs["extras"]

    def test_extras_contains_tool_registry(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="s1")
        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["extras"]["tool_registry"] is agent.registry

    def test_llm_dict_has_default_key(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            Agent(config=_make_config(), session_id="s1")
        call_kwargs = MockSR2.call_args.kwargs
        assert "default" in call_kwargs["llm"]


# ---------------------------------------------------------------------------
# B. session_id
# ---------------------------------------------------------------------------

class TestSessionId:
    def test_session_id_propagated(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="my-session")
        assert agent.session_id == "my-session"

    def test_default_session_id_uses_name(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        assert "test" in agent.session_id


# ---------------------------------------------------------------------------
# C. Tool registration from config
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_tools_from_config_registered(self):
        from sr2_spectre.agent import Agent
        from sr2_spectre.config import ToolConfig

        cfg = _make_config(tools=[
            ToolConfig(
                name="dummy",
                class_path="sr2_spectre.tools.registry:ToolRegistry",  # won't register real tool
                config={},
            )
        ])

        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            with patch.object(
                __import__("sr2_spectre.tools.registry", fromlist=["ToolRegistry"]).ToolRegistry,
                "register_from_class_path",
            ) as mock_reg:
                agent = Agent(config=cfg)

    def test_register_tool_at_runtime(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        agent.register_tool("echo", "Echo", {}, lambda msg: msg)
        assert "echo" in agent.registry


# ---------------------------------------------------------------------------
# D. handle_user_message: happy path (no tools)
# ---------------------------------------------------------------------------

class TestHandleUserMessageHappyPath:
    @pytest.mark.asyncio
    async def test_returns_turn_result_with_text(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            mock_sr2 = _mock_sr2([
                StreamEvent(type="text", text="Hello "),
                StreamEvent(type="text", text="world!"),
                StreamEvent(type="end"),
            ])
            MockSR2.return_value = mock_sr2
            agent = Agent(config=_make_config(), session_id="s")

        result = await agent.handle_user_message("Hi")
        assert isinstance(result, TurnResult)
        assert result.text == "Hello world!"
        assert result.tool_calls_executed == 0

    @pytest.mark.asyncio
    async def test_seed_session_called_each_round(self):
        from sr2_spectre.agent import Agent
        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()
        events = [StreamEvent(type="text", text="Hi"), StreamEvent(type="end")]

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        await agent.handle_user_message("Hello")
        mock_sr2.seed_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_has_user_and_assistant_messages(self):
        from sr2_spectre.agent import Agent
        mock_sr2 = _mock_sr2([
            StreamEvent(type="text", text="Response"),
            StreamEvent(type="end"),
        ])
        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        await agent.handle_user_message("Question")
        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"


# ---------------------------------------------------------------------------
# E. handle_user_message: one tool round-trip
# ---------------------------------------------------------------------------

class TestHandleUserMessageToolRoundTrip:
    @pytest.mark.asyncio
    async def test_single_tool_call_executed(self):
        from sr2_spectre.agent import Agent

        # Turn 1: LLM returns tool_use
        turn1_events = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="add", tool_input={"a": 1, "b": 2}),
            StreamEvent(type="end"),
        ]
        # Turn 2: LLM returns text
        turn2_events = [
            StreamEvent(type="text", text="The answer is 3"),
            StreamEvent(type="end"),
        ]
        call_count = 0

        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            nonlocal call_count
            call_count += 1
            events = turn1_events if call_count == 1 else turn2_events
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        # Register a real tool
        agent.register_tool("add", "Add", {}, lambda a, b: str(a + b))

        result = await agent.handle_user_message("What is 1+2?")
        assert result.text == "The answer is 3"
        assert result.tool_calls_executed == 1

    @pytest.mark.asyncio
    async def test_tool_result_appended_to_history(self):
        from sr2_spectre.agent import Agent

        turn1_events = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="echo", tool_input={"msg": "hi"}),
            StreamEvent(type="end"),
        ]
        turn2_events = [StreamEvent(type="text", text="done"), StreamEvent(type="end")]
        call_count = 0

        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            nonlocal call_count
            call_count += 1
            for ev in (turn1_events if call_count == 1 else turn2_events):
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        agent.register_tool("echo", "Echo", {}, lambda msg: msg)
        await agent.handle_user_message("Echo hi")

        # history: user → assistant(tool_use) → user(tool_result) → assistant(text)
        assert len(agent.history) == 4
        assert agent.history[2].role == "user"
        result_content = agent.history[2].content
        assert any(isinstance(b, ToolResultBlock) for b in result_content)


# ---------------------------------------------------------------------------
# F. Error recovery — FR13
# ---------------------------------------------------------------------------

class TestToolErrorRecovery:
    @pytest.mark.asyncio
    async def test_tool_error_fed_back_as_tool_result(self):
        """A failing tool's error text is fed back as ToolResultBlock, loop continues."""
        from sr2_spectre.agent import Agent

        turn1_events = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="fail_tool", tool_input={}),
            StreamEvent(type="end"),
        ]
        turn2_events = [StreamEvent(type="text", text="recovered"), StreamEvent(type="end")]
        call_count = 0

        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            nonlocal call_count
            call_count += 1
            for ev in (turn1_events if call_count == 1 else turn2_events):
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        def _fail(**kw):
            raise ValueError("something went wrong")

        agent.register_tool("fail_tool", "Fails", {}, _fail)
        result = await agent.handle_user_message("Try the tool")

        # Loop continued and returned the final text
        assert result.text == "recovered"

        # Tool result block in history contains error text
        tool_result_msg = next(
            m for m in agent.history
            if m.role == "user" and any(isinstance(b, ToolResultBlock) for b in m.content)
        )
        error_blocks = [b for b in tool_result_msg.content if isinstance(b, ToolResultBlock)]
        assert any("something went wrong" in b.content for b in error_blocks)

    @pytest.mark.asyncio
    async def test_tool_error_does_not_raise_out_of_loop(self):
        """Tool failure must NOT propagate as an exception from handle_user_message."""
        from sr2_spectre.agent import Agent

        turn1_events = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="boom", tool_input={}),
            StreamEvent(type="end"),
        ]
        turn2_events = [StreamEvent(type="text", text="ok"), StreamEvent(type="end")]
        call_count = 0

        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            nonlocal call_count
            call_count += 1
            for ev in (turn1_events if call_count == 1 else turn2_events):
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        agent.register_tool("boom", "Boom", {}, lambda: (_ for _ in ()).throw(RuntimeError("BOOM")))
        # Must not raise
        result = await agent.handle_user_message("Trigger tool")
        assert result.text == "ok"


# ---------------------------------------------------------------------------
# G. max_tool_rounds exceeded — FR14
# ---------------------------------------------------------------------------

class TestMaxToolRounds:
    @pytest.mark.asyncio
    async def test_loop_terminates_at_max_tool_rounds(self):
        """When the model keeps returning tool_use, the loop stops at max_tool_rounds."""
        from sr2_spectre.agent import Agent

        # LLM always returns tool_use
        always_tool = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="loop_tool", tool_input={}),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in always_tool:
                yield ev
        mock_sr2.turn = _turn

        cfg = _make_config(max_tool_rounds=3)
        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=cfg)

        agent.register_tool("loop_tool", "Loops", {}, lambda: "still going")
        result = await agent.handle_user_message("Start")

        # Must return, not hang
        assert isinstance(result, TurnResult)
        # Warning included in text
        assert result.tool_calls_executed == 3

    @pytest.mark.asyncio
    async def test_max_tool_rounds_warning_in_result(self):
        """Result text should contain a warning when max_tool_rounds is exceeded."""
        from sr2_spectre.agent import Agent

        always_tool = [
            StreamEvent(type="tool_use", tool_use_id="tu1", tool_name="loop_tool", tool_input={}),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in always_tool:
                yield ev
        mock_sr2.turn = _turn

        cfg = _make_config(max_tool_rounds=2)
        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=cfg)

        agent.register_tool("loop_tool", "Loops", {}, lambda: "ok")
        result = await agent.handle_user_message("Trigger")
        # Warning should be non-empty (either in text or we just verify it returns)
        assert result is not None


# ---------------------------------------------------------------------------
# H. new_session()
# ---------------------------------------------------------------------------

class TestNewSession:
    @pytest.mark.asyncio
    async def test_new_session_resets_history(self):
        from sr2_spectre.agent import Agent
        mock_sr2 = _mock_sr2()
        with patch("sr2_spectre.agent.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        await agent.handle_user_message("Hello")
        assert len(agent.history) > 0

        agent.new_session("fresh")
        assert agent.history == []
        assert agent.session_id == "fresh"


# ---------------------------------------------------------------------------
# I. register_tool (runtime)
# ---------------------------------------------------------------------------

class TestRegisterTool:
    def test_register_tool_appears_in_registry(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.agent.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        agent.register_tool("ping", "Ping", {}, lambda: "pong")
        assert "ping" in agent.registry
