"""Tests for CompleteStepTool (obsidian-aal.3).

Covers:
- Successful verify + status flip + findings
- Missing task file
- Task not pending
- Missing verify command
- Failed verify
- Event emission from agent on success
- No event emission on failure
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sr2.models import ToolResultBlock, ToolUseBlock

from sr2_spectre.tools.builtins.complete_step import (
    CompleteStepResult,
    CompleteStepTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plans_root(tmp_path: Path) -> Path:
    """Create a temporary plans directory."""
    root = tmp_path / "plans"
    root.mkdir()
    plan_dir = root / "test-plan"
    plan_dir.mkdir()
    return root


@pytest.fixture
def task_file(plans_root: Path) -> Path:
    """Create a standard pending task file."""
    task = plans_root / "test-plan" / "01-setup.md"
    task.write_text(
        textwrap.dedent(
            """\
            ---
            kind: task
            plan: test-plan
            order: 1
            status: pending
            verify: echo "tests pass"
            title: Setup infrastructure
            ---
            
            Set up the project infrastructure.
            """
        )
    )
    return task


@pytest.fixture
def tool(plans_root: Path) -> CompleteStepTool:
    return CompleteStepTool(plans_root=str(plans_root))


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestCompleteStepSuccess:
    async def test_successful_completion(self, tool: CompleteStepTool, task_file: Path) -> None:
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="none",
        )
        data = json.loads(result_str)

        assert data["success"] is True
        assert data["plan"] == "test-plan"
        assert data["task"] == "01-setup"
        assert data["order"] == 1
        assert data["frame"] == "plan:test-plan/01-setup"

        # Verify status flip
        content = task_file.read_text()
        assert "status: done" in content
        assert "status: pending" not in content

    async def test_findings_appended(self, tool: CompleteStepTool, task_file: Path, plans_root: Path) -> None:
        await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="The --plugin flag is also in tests/conftest.py",
        )

        findings_file = plans_root / "test-plan" / "_findings.md"
        assert findings_file.exists()
        content = findings_file.read_text()
        assert "01-setup" in content
        assert "The --plugin flag is also in tests/conftest.py" in content

    async def test_findings_none_is_noop(self, tool: CompleteStepTool, task_file: Path, plans_root: Path) -> None:
        await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="none",
        )

        findings_file = plans_root / "test-plan" / "_findings.md"
        assert not findings_file.exists()

    async def test_findings_appended_to_existing(self, tool: CompleteStepTool, task_file: Path, plans_root: Path) -> None:
        findings_file = plans_root / "test-plan" / "_findings.md"
        findings_file.write_text("### 00-previous\nPrevious finding\n")

        await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="New discovery",
        )

        content = findings_file.read_text()
        assert "00-previous" in content  # preserved
        assert "01-setup" in content  # appended

    async def test_flip_does_not_corrupt_body_mention(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """A 'status: pending' mention in the body must NOT be rewritten —
        only the frontmatter status flips."""
        task = plans_root / "test-plan" / "04-bodymention.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 4
                status: pending
                verify: echo ok
                title: Body mentions status
                ---

                Document the old behavior where `status: pending` stayed forever.
                """
            )
        )
        await tool(plan_slug="test-plan", task_slug="04-bodymention", findings="none")

        content = task.read_text()
        # Frontmatter flipped...
        assert "status: done" in content
        # ...but the body's literal mention is untouched.
        assert "`status: pending` stayed forever" in content


class TestCompleteStepFailures:
    async def test_missing_task_file(self, tool: CompleteStepTool) -> None:
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="99-nonexistent",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is False
        assert "not found" in data["message"].lower()

    async def test_task_not_pending(self, tool: CompleteStepTool, task_file: Path) -> None:
        task_file.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 1
                status: done
                verify: echo ok
                title: Already done
                ---
                
                Done task.
                """
            )
        )
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is False
        assert "not pending" in data["message"].lower()

    async def test_missing_verify_command(self, tool: CompleteStepTool, plans_root: Path) -> None:
        task = plans_root / "test-plan" / "02-noverify.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 2
                status: pending
                title: No verify
                ---
                
                Task with no verify command.
                """
            )
        )
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="02-noverify",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is False
        assert "no 'verify:" in data["message"] or "no" in data["message"].lower() and "verify" in data["message"].lower()

    async def test_failed_verify(self, tool: CompleteStepTool, plans_root: Path) -> None:
        task = plans_root / "test-plan" / "03-broken.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 3
                status: pending
                verify: exit 1
                title: Broken task
                ---
                
                This task's verify fails.
                """
            )
        )
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="03-broken",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is False
        assert "failed" in data["message"].lower()

        # Status should NOT be flipped
        content = task.read_text()
        assert "status: pending" in content


class TestCompleteStepToolSchema:
    def test_tool_has_required_schema_fields(self) -> None:
        tool = CompleteStepTool()
        assert tool.name == "complete_step"
        assert tool.description
        assert "findings" in tool.input_schema["required"]
        assert "plan_slug" in tool.input_schema["required"]
        assert "task_slug" in tool.input_schema["required"]
        assert tool.input_schema["type"] == "object"


# ---------------------------------------------------------------------------
# Agent event emission tests
# ---------------------------------------------------------------------------


class TestStatusFlipSpacingVariants:
    """spc-13: _flip_status must handle spacing/case variants that the
    parser already tolerates (status:pending, Status: Pending, etc.)."""

    async def test_uppercase_status_value(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """status: PENDING (uppercase value) must flip to done."""
        task = plans_root / "test-plan" / "05-uppercase-val.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 5
                status: PENDING
                verify: echo ok
                title: Uppercase value
                ---

                Task with uppercase status value.
                """
            )
        )
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="05-uppercase-val",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is True

        from sr2_spectre.planning import parse_file

        parsed = parse_file(task)
        assert parsed is not None
        assert parsed.status.value == "done"

    async def test_uppercase_status_key(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """Status: Pending (capital S) must flip correctly.

        Note: the frontmatter parser itself has a key-casing gap
        (data.get("status") misses "Status"), so we verify the flip
        by checking raw file content rather than re-parsing.
        """
        task = plans_root / "test-plan" / "06-uppercase.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 6
                Status: Pending
                verify: echo ok
                title: Uppercase
                ---

                Task with uppercase status key.
                """
            )
        )
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="06-uppercase",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is True

        # Verify the status line was actually changed in the file.
        content = task.read_text()
        assert "Status: done" in content
        assert "Pending" not in content

    async def test_mixed_spacing_and_case(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """STATUS : Pending (uppercase key, extra space) must flip.

        Note: parser key lookup is case-sensitive, so we verify via raw content.
        """
        task = plans_root / "test-plan" / "07-mixed.md"
        task.write_text(
            "---\n"
            "kind: task\n"
            "plan: test-plan\n"
            "order: 7\n"
            "STATUS : Pending\n"
            "verify: echo ok\n"
            "title: Mixed\n"
            "---\n"
            "\n"
            "Task with mixed spacing and case.\n"
        )
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="07-mixed",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is True

        # Verify the status line was actually changed.
        content = task.read_text()
        assert "STATUS: done" in content or "STATUS : done" in content
        assert "Pending" not in content and "pending" not in content

    async def test_flip_failure_returns_error(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """If the status line can't be matched, complete_step must FAIL
        instead of silently succeeding."""
        # Create a task where "status" appears but is embedded in a
        # non-standard way that the YAML parser still picks up (e.g.,
        # quoted value). The current broken code would no-op here.
        task = plans_root / "test-plan" / "08-quoted.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 8
                status: "pending"
                verify: echo ok
                title: Quoted
                ---

                Task with quoted status value.
                """
            )
        )
        # With the fix, this should still work because the YAML key in
        # the raw text is `status: "pending"` — the replace of
        # `status: pending` won't match but the new YAML-aware approach
        # should handle it. If the fix is correct, success + done.
        result_str = await tool(
            plan_slug="test-plan",
            task_slug="08-quoted",
            findings="none",
        )
        data = json.loads(result_str)
        assert data["success"] is True

        from sr2_spectre.planning import parse_file

        parsed = parse_file(task)
        assert parsed is not None
        assert parsed.status.value == "done"


# ---------------------------------------------------------------------------
# Agent event emission tests
# ---------------------------------------------------------------------------


class TestAgentCompleteStepEvent:
    async def test_agent_emits_event_on_success(self) -> None:
        """When complete_step succeeds, the agent emits plan_step_completed."""
        from sr2_spectre.agent import _is_complete_step_success

        success_result = ToolResultBlock(
            tool_use_id="tool-1",
            content=json.dumps({
                "success": True,
                "plan": "test-plan",
                "task": "01-setup",
                "order": 1,
                "frame": "plan:test-plan/01-setup",
                "message": "Task verified and marked complete.",
            }),
        )
        data = _is_complete_step_success(success_result)
        assert data is not None
        assert data["plan"] == "test-plan"
        assert data["task"] == "01-setup"
        assert data["order"] == 1
        assert data["frame"] == "plan:test-plan/01-setup"

    async def test_agent_no_event_on_failure(self) -> None:
        """When complete_step fails, no event data is extracted."""
        from sr2_spectre.agent import _is_complete_step_success

        failure_result = ToolResultBlock(
            tool_use_id="tool-1",
            content=json.dumps({
                "success": False,
                "plan": "test-plan",
                "task": "01-setup",
                "order": 1,
                "frame": "plan:test-plan/01-setup",
                "message": "Verification FAILED",
            }),
        )
        data = _is_complete_step_success(failure_result)
        assert data is None

    async def test_agent_no_event_on_non_json(self) -> None:
        """Non-JSON results don't trigger event emission."""
        from sr2_spectre.agent import _is_complete_step_success

        text_result = ToolResultBlock(
            tool_use_id="tool-1",
            content="Task completed!",
        )
        data = _is_complete_step_success(text_result)
        assert data is None

    async def test_agent_no_event_on_error(self) -> None:
        """Error results don't trigger event emission."""
        from sr2_spectre.agent import _is_complete_step_success

        error_result = ToolResultBlock(
            tool_use_id="tool-1",
            content="ERROR: Tool not found",
            is_error=True,
        )
        data = _is_complete_step_success(error_result)
        assert data is None
