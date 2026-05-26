"""Tests for SingleShotPlugin."""
import sys
import pytest
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

from sr2_spectre.plugins.single_shot import SingleShotPlugin
from sr2_spectre.core.loop import TurnResult


def _make_agent(response_text: str = "The answer is 4") -> MagicMock:
    agent = AsyncMock()
    agent.handle_user_message.return_value = TurnResult(
        text=response_text, tool_calls_executed=0, total_tokens=30
    )
    return agent


@pytest.mark.asyncio
async def test_start_stop_are_noops() -> None:
    plugin = SingleShotPlugin(prompt="hello")
    agent = _make_agent()
    await plugin.start(agent)
    await plugin.stop()
    agent.handle_user_message.assert_not_called()


@pytest.mark.asyncio
async def test_run_with_constructor_prompt(capsys: pytest.CaptureFixture) -> None:
    plugin = SingleShotPlugin(prompt="What is 2+2?")
    agent = _make_agent("4")

    await plugin.run(agent)

    agent.handle_user_message.assert_called_once_with("What is 2+2?")
    captured = capsys.readouterr()
    assert captured.out.strip() == "4"


@pytest.mark.asyncio
async def test_run_reads_from_stdin(capsys: pytest.CaptureFixture) -> None:
    plugin = SingleShotPlugin()
    agent = _make_agent("pong")

    with patch("sys.stdin", StringIO("ping\n")), patch("sys.argv", ["sr2-spectre"]):
        await plugin.run(agent)

    agent.handle_user_message.assert_called_once_with("ping")
    captured = capsys.readouterr()
    assert captured.out.strip() == "pong"


@pytest.mark.asyncio
async def test_run_empty_prompt_exits(capsys: pytest.CaptureFixture) -> None:
    plugin = SingleShotPlugin()
    agent = _make_agent()

    with patch("sys.stdin", StringIO("")), patch("sys.argv", ["sr2-spectre"]):
        with pytest.raises(SystemExit) as exc_info:
            await plugin.run(agent)

    assert exc_info.value.code == 1
    agent.handle_user_message.assert_not_called()


def test_load_plugin_via_cli() -> None:
    """_load_plugin('single_shot') must return a SingleShotPlugin instance."""
    from sr2_spectre.cli import _load_plugin
    plugin = _load_plugin("single_shot")
    assert isinstance(plugin, SingleShotPlugin)


def test_load_plugin_with_prompt_kwarg() -> None:
    from sr2_spectre.cli import _load_plugin
    plugin = _load_plugin("single_shot", prompt="hello")
    assert isinstance(plugin, SingleShotPlugin)
    assert plugin._prompt == "hello"
