"""Tests for tool execution loop."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from sr2_spectre.core.loop import (
    TurnResult,
    _build_system_prompt,
    run_tool_loop,
)


def test_build_system_prompt_empty() -> None:
    assert _build_system_prompt("") is None


def test_build_system_prompt_set() -> None:
    result = _build_system_prompt("You are helpful.")
    assert result is not None
    assert len(result) == 1
    assert result[0]["text"] == "You are helpful."


def test_turn_result() -> None:
    r = TurnResult(text="answer", tool_calls_executed=2, total_tokens=200)
    assert r.text == "answer"
    assert r.tool_calls_executed == 2
    assert r.total_tokens == 200


@pytest.mark.asyncio
async def test_run_tool_loop_no_tools() -> None:
    """Loop exits immediately when stop_reason != tool_use."""
    mock_client = AsyncMock()
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "Final answer"
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [mock_block]
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 30
    mock_client.complete.return_value = mock_response

    history: list = []
    result = await run_tool_loop(
        client=mock_client,
        system_prompt="Test",
        history=history,
        tools_definitions=[],
        tool_executor=AsyncMock(),
    )

    assert result.text == "Final answer"
    assert result.tool_calls_executed == 0
    assert mock_client.complete.call_count == 1
    # History has user + assistant
    assert len(history) == 1
    assert history[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_run_tool_loop_max_iterations() -> None:
    """Loop raises when max iterations exceeded."""
    mock_client = AsyncMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "loop_tool"
    mock_block.id = "tc1"
    mock_block.input = {}
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = [mock_block]
    mock_response.usage.input_tokens = 0
    mock_response.usage.output_tokens = 0
    mock_client.complete.return_value = mock_response

    mock_executor = AsyncMock()
    mock_executor.execute.return_value = "ok"

    with pytest.raises(RuntimeError, match="exceeded"):
        await run_tool_loop(
            client=mock_client,
            system_prompt="Test",
            history=[],
            tools_definitions=[],
            tool_executor=mock_executor,
            max_tool_iterations=2,
        )


@pytest.mark.asyncio
async def test_run_tool_loop_one_tool_then_text() -> None:
    """Loop executes one tool, then returns text on second call."""
    mock_client = AsyncMock()

    # First call: tool_use
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "calc"
    tool_block.id = "tc1"
    tool_block.input = {"expr": "2+2"}

    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.content = [tool_block]
    resp1.usage.input_tokens = 10
    resp1.usage.output_tokens = 5

    # Second call: text
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "The answer is 4"

    resp2 = MagicMock()
    resp2.stop_reason = "end_turn"
    resp2.content = [text_block]
    resp2.usage.input_tokens = 20
    resp2.usage.output_tokens = 10

    mock_client.complete.side_effect = [resp1, resp2]

    mock_executor = AsyncMock()
    mock_executor.execute.return_value = "4"

    history: list = []
    result = await run_tool_loop(
        client=mock_client,
        system_prompt="Test",
        history=history,
        tools_definitions=[],
        tool_executor=mock_executor,
    )

    assert result.text == "The answer is 4"
    assert result.tool_calls_executed == 1
    assert mock_client.complete.call_count == 2
    # History: tool_result + assistant(tool_use) + assistant(text)
    assert any(m["role"] == "tool" for m in history)
    assert any(m["role"] == "assistant" for m in history)
