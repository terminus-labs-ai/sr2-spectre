"""Tests for TUI text copy — obsidian-xije (FR12 regression fix).

Covers:
- Ctrl+Alt+C binding copies RichLog content to clipboard
- Clipboard fallback chain (pyperclip → subprocess)
- Help text documents the Shift+drag modifier
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sr2_spectre.interfaces.tui import SpectreTUI, _HELP


# ---------------------------------------------------------------------------
# Binding exists
# ---------------------------------------------------------------------------

def test_copy_binding_registered() -> None:
    """Ctrl+Alt+C must be bound to copy_output action."""
    binding_map = {b[0]: b[1] for b in SpectreTUI.BINDINGS}
    assert "ctrl+alt+c" in binding_map
    assert binding_map["ctrl+alt+c"] == "copy_output"


# ---------------------------------------------------------------------------
# Copy action extracts text from RichLog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_copy_output_extracts_richlog_text(tui_app: SpectreTUI) -> None:
    """action_copy_output must extract plain text from RichLog lines."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Input, RichLog

        # Write some content to the log
        log = tui_app.query_one("#output", RichLog)
        log.write("line one")
        log.write("line two")
        await pilot.pause()

        # Mock the clipboard write
        with patch.object(tui_app, "_copy_to_clipboard") as mock_copy:
            tui_app.action_copy_output()
            await pilot.pause()

            mock_copy.assert_called_once()
            copied_text = mock_copy.call_args[0][0]
            assert "line one" in copied_text
            assert "line two" in copied_text


@pytest.mark.asyncio
async def test_copy_output_skips_blank_lines(tui_app: SpectreTUI) -> None:
    """action_copy_output must skip blank lines from the RichLog."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import RichLog

        log = tui_app.query_one("#output", RichLog)
        log.write("content")
        log.write("")
        log.write("more content")
        await pilot.pause()

        with patch.object(tui_app, "_copy_to_clipboard") as mock_copy:
            tui_app.action_copy_output()
            await pilot.pause()

            copied_text = mock_copy.call_args[0][0]
            # Blank lines should be filtered out
            lines = copied_text.split("\n")
            assert all(line.strip() for line in lines)


@pytest.mark.asyncio
async def test_copy_output_triggers_notification(tui_app: SpectreTUI) -> None:
    """action_copy_output must notify the user."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import RichLog

        log = tui_app.query_one("#output", RichLog)
        log.write("test content")
        await pilot.pause()

        with patch.object(tui_app, "_copy_to_clipboard"):
            tui_app.action_copy_output()
            await pilot.pause()

        # Check notifications (Textual stores them in _notifications)
        notifications = list(tui_app._notifications)
        assert any("copied" in str(n.message).lower() for n in notifications)


# ---------------------------------------------------------------------------
# Clipboard fallback chain
# ---------------------------------------------------------------------------

def test_copy_to_clipboard_no_raise_on_missing_tools() -> None:
    """_copy_to_clipboard should not raise when no clipboard tools available."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()

        # pyperclip is not installed, all subprocess tools missing — should not raise
        SpectreTUI._copy_to_clipboard("test text")


def test_copy_to_clipboard_fallback_subprocess() -> None:
    """_copy_to_clipboard falls back to subprocess clipboard tools when pyperclip missing."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # pyperclip is not installed in test env, so it falls through to subprocess
        SpectreTUI._copy_to_clipboard("test text")
        # Should have tried at least one clipboard command
        assert mock_run.called


# ---------------------------------------------------------------------------
# Help text documents selection
# ---------------------------------------------------------------------------

def test_help_documents_shift_drag() -> None:
    """Help text must document the Shift+drag modifier for text selection."""
    assert "Shift" in _HELP
    assert "drag" in _HELP.lower() or "select" in _HELP.lower()


def test_help_documents_copy_binding() -> None:
    """Help text must document the copy keybinding."""
    assert "Ctrl+Alt+C" in _HELP or "ctrl+alt+c" in _HELP


# ---------------------------------------------------------------------------
# RichLog ALLOW_SELECT (regression guard for FR12)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_richlog_allow_select_not_overridden(tui_app: SpectreTUI) -> None:
    """RichLog must not have ALLOW_SELECT disabled (FR12 regression guard)."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import RichLog
        log = tui_app.query_one("#output", RichLog)
        assert log.ALLOW_SELECT is True
        assert log.allow_select is True
