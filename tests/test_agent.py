"""Tests for Agent."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig
from sr2_spectre.core.loop import TurnResult


def _make_agent() -> Agent:
    return Agent(
        config=AgentConfig(
            name="test",
            model="test-model",
            relay_base_url="http://test:8000",
            system_prompt="You are a test agent.",
        ),
        session_id="test-session",
    )


def test_agent_init() -> None:
    agent = _make_agent()
    assert agent.session_id == "test-session"
    assert agent.system_prompt == "You are a test agent."
    assert agent.config.model == "test-model"


@pytest.mark.asyncio
async def test_handle_user_message_happy_path() -> None:
    agent = _make_agent()

    # Mock the tool loop to return a fixed result
    mock_result = TurnResult(text="42", tool_calls_executed=0, total_tokens=100)

    with patch.object(agent, "handle_user_message", new=None):
        pass

    # Direct test: mock run_tool_loop
    with patch("sr2_spectre.agent.run_tool_loop", new_callable=AsyncMock) as mock_loop:
        mock_loop.return_value = mock_result
        result = await agent.handle_user_message("What is the answer?")

    assert result.text == "42"
    assert result.tool_calls_executed == 0
    # Verify user message was appended to session
    assert any(m["role"] == "user" for m in agent.session.history)


@pytest.mark.asyncio
async def test_handle_user_message_with_tools() -> None:
    agent = _make_agent()
    agent.register_tool(
        name="calculator",
        description="Calculate expressions",
        input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
        fn=lambda expr: str(eval(expr)),
    )

    mock_result = TurnResult(
        text="The answer is 42",
        tool_calls_executed=1,
        total_tokens=150,
    )

    with patch("sr2_spectre.agent.run_tool_loop", new_callable=AsyncMock) as mock_loop:
        mock_loop.return_value = mock_result
        result = await agent.handle_user_message("2+2*20")

    assert result.text == "The answer is 42"
    assert result.tool_calls_executed == 1


def test_new_session() -> None:
    agent = _make_agent()
    agent.new_session("fresh-session")
    assert agent.session_id == "fresh-session"
    assert agent.session.history == []


def test_register_tool_at_runtime() -> None:
    agent = _make_agent()
    agent.register_tool(
        name="echo",
        description="Echo back",
        input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        fn=lambda msg: msg,
    )
    assert "echo" in agent.registry
