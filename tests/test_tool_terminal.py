"""Tests for TerminalTool."""
import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_terminal_class_attributes() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    assert isinstance(TerminalTool.name, str) and TerminalTool.name
    assert isinstance(TerminalTool.description, str) and TerminalTool.description
    assert isinstance(TerminalTool.input_schema, dict)


def test_terminal_input_schema_requires_command() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    schema = TerminalTool.input_schema
    assert "command" in schema.get("properties", {})
    assert "command" in schema.get("required", [])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_terminal_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.terminal.TerminalTool")
    assert "terminal" in reg


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_returns_stdout() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    mock_proc = MagicMock()
    mock_proc.stdout = b"hello world\n"
    mock_proc.stderr = b""
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_shell", new=AsyncMock(return_value=mock_proc)):
        mock_proc.communicate = AsyncMock(return_value=(b"hello world\n", b""))
        tool = TerminalTool()
        result = await tool(command="echo hello world")

    assert "hello world" in result


@pytest.mark.asyncio
async def test_terminal_combines_stdout_and_stderr() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    with patch("asyncio.create_subprocess_shell", new=AsyncMock()) as mock_shell:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"out\n", b"err\n"))
        proc.returncode = 0
        mock_shell.return_value = proc

        tool = TerminalTool()
        result = await tool(command="cmd")

    assert "out" in result
    assert "err" in result


@pytest.mark.asyncio
async def test_terminal_empty_output_returns_empty_string() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    with patch("asyncio.create_subprocess_shell", new=AsyncMock()) as mock_shell:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 0
        mock_shell.return_value = proc

        tool = TerminalTool()
        result = await tool(command="true")

    assert result == ""


# ---------------------------------------------------------------------------
# Non-zero exit code — output returned, no exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_nonzero_exit_returns_output_not_raises() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    with patch("asyncio.create_subprocess_shell", new=AsyncMock()) as mock_shell:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", b"command not found\n"))
        proc.returncode = 127
        mock_shell.return_value = proc

        tool = TerminalTool()
        result = await tool(command="nosuchcmd")

    assert "command not found" in result


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_timeout_raises_timeout_error() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    with patch("asyncio.create_subprocess_shell", new=AsyncMock()) as mock_shell:
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.kill = MagicMock()
        mock_shell.return_value = proc

        tool = TerminalTool(timeout=1)
        with pytest.raises(TimeoutError, match="sleep 999"):
            await tool(command="sleep 999")


@pytest.mark.asyncio
async def test_terminal_timeout_message_contains_command() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    cmd = "some_long_running_command --flag"
    with patch("asyncio.create_subprocess_shell", new=AsyncMock()) as mock_shell:
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.kill = MagicMock()
        mock_shell.return_value = proc

        tool = TerminalTool(timeout=1)
        with pytest.raises(TimeoutError) as exc_info:
            await tool(command=cmd)

    assert cmd in str(exc_info.value)


# ---------------------------------------------------------------------------
# Custom timeout constructor arg
# ---------------------------------------------------------------------------

def test_terminal_custom_timeout_stored() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool(timeout=60)
    assert tool.timeout == 60
