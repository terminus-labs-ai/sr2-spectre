"""Complete Step tool — verify-gated task completion for planning workflows.

Runs the task's ``verify:`` command, flips status pending→done ONLY on green,
and emits a ``plan_step_completed`` event signal (returned as structured data
so the Agent's tool executor can broadcast it on the SR2 event bus).

Requires a ``findings`` argument (free text; ``"none"`` is valid) as the cheap
externalize-or-not gate before context is burned.

Registered as a builtin tool; takes ``plans_root`` via constructor config so
it can locate the plan/task files on disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sr2_spectre.planning import TaskFrontmatter, TaskStatus, parse_file

logger = logging.getLogger(__name__)


def frame_id(plan_slug: str, task_slug: str) -> str:
    """Canonical frame id for a plan task (see the ``frame`` primitive).

    Single source of truth for the ``plan:<plan>/<task>`` frame identifier so
    the transformer that burns a completed step's context matches exactly what
    this tool stamped.
    """
    return f"plan:{plan_slug}/{task_slug}"


@dataclass
class CompleteStepResult:
    """Structured return from a complete_step call."""

    success: bool
    plan: str
    task: str
    order: int
    frame: str
    message: str


class CompleteStepTool:
    """Verify-gated task completion tool.

    Schema:
        plan_slug (str): Plan directory name (e.g. "step-compaction").
        task_slug (str): Task file slug without extension (e.g. "02-cli-flag").
        findings (str): Free-text cross-step discoveries or "none".

    The tool reads the task file from ``<plans_root>/<plan_slug>/<task_slug>.md``,
    runs its ``verify:`` command, flips status to ``done`` on success, and
    appends findings to ``<plans_root>/<plan_slug>/_findings.md``.

    Returns a JSON-serializable dict with ``success``, ``plan``, ``task``,
    ``order``, and ``frame`` fields when verification passes — enabling the
    agent to emit a ``plan_step_completed`` event.
    """

    name = "complete_step"
    description = (
        "Complete a planning task: runs the task's verify command, flips "
        "status pending→done on success, and externalizes findings. "
        "Requires a findings argument (free text; 'none' is valid). "
        "Only use AFTER all work for the task is finished and verified."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "plan_slug": {
                "type": "string",
                "description": "Plan directory name (e.g. 'step-compaction').",
            },
            "task_slug": {
                "type": "string",
                "description": "Task file slug (e.g. '02-cli-flag').",
            },
            "findings": {
                "type": "string",
                "description": (
                    "Cross-step discoveries or 'none'. Forces a conscious "
                    "externalize-or-not decision before the step's context burns."
                ),
            },
        },
        "required": ["plan_slug", "task_slug", "findings"],
    }

    # Verify command wall-clock budget (seconds).
    _VERIFY_TIMEOUT = 120

    def __init__(self, plans_root: str | None = None) -> None:
        raw = plans_root or str(Path.home() / ".sr2" / "plans")
        self._plans_root = Path(raw).expanduser().resolve()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def __call__(
        self,
        plan_slug: str,
        task_slug: str,
        findings: str,
    ) -> str:
        """Execute the complete-step flow. Returns a JSON string."""
        result = await self._complete(plan_slug, task_slug, findings)
        return self._serialize(result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _complete(
        self,
        plan_slug: str,
        task_slug: str,
        findings: str,
    ) -> CompleteStepResult:
        frame = frame_id(plan_slug, task_slug)
        task_file = self._plans_root / plan_slug / f"{task_slug}.md"

        def fail(message: str, order: int = 0) -> CompleteStepResult:
            return CompleteStepResult(
                success=False,
                plan=plan_slug,
                task=task_slug,
                order=order,
                frame=frame,
                message=message,
            )

        # 1. Read + parse the task file (reusing the planning parser).
        task = self._read_task(task_file)
        if task is None:
            return fail(f"Task file not found or unparseable: {task_file}")

        # 2. Status must be pending.
        if task.status is not TaskStatus.PENDING:
            return fail(
                f"Task {task_slug} is not pending (status: {task.status.value})",
                order=task.order,
            )

        # 3. Verify command must exist.
        verify_cmd = task.verify.strip()
        if not verify_cmd:
            return fail(
                f"Task {task_slug} has no 'verify:' command in frontmatter",
                order=task.order,
            )

        # 4. Run verify — only a green run may flip status.
        verify_ok, verify_output = await self._run_verify(verify_cmd)
        if not verify_ok:
            return fail(
                f"Verification FAILED for {task_slug}:\n{verify_output}",
                order=task.order,
            )

        # 5. Flip status pending→done.
        self._flip_status(task_file)

        # 6. Externalize findings (unless explicitly "none").
        if findings.strip().lower() != "none":
            findings_file = self._plans_root / plan_slug / "_findings.md"
            self._append_findings(findings_file, task_slug, findings.strip())

        return CompleteStepResult(
            success=True,
            plan=plan_slug,
            task=task_slug,
            order=task.order,
            frame=frame,
            message=f"Task {task_slug} verified and marked complete.",
        )

    @staticmethod
    def _read_task(task_file: Path) -> TaskFrontmatter | None:
        """Parse the task file via the shared planning parser.

        Returns the ``TaskFrontmatter`` for a valid ``kind: task`` file, or
        None if the file is missing, unparseable, or not a task.
        """
        fm = parse_file(task_file)
        if not isinstance(fm, TaskFrontmatter):
            logger.warning("No valid task frontmatter in %s", task_file)
            return None
        return fm

    async def _run_verify(self, command: str) -> tuple[bool, str]:
        """Run the verify command in a subprocess. Returns (success, output)."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._VERIFY_TIMEOUT
            )
            output = (
                stdout.decode(errors="replace") + stderr.decode(errors="replace")
            ).strip()
            return proc.returncode == 0, output
        except (asyncio.TimeoutError, TimeoutError):
            return False, f"Verify command timed out: {command}"
        except Exception as exc:  # noqa: BLE001 — surface any spawn failure as a failed verify
            return False, f"Verify command error: {exc}"

    @staticmethod
    def _flip_status(task_file: Path) -> None:
        """Flip ``status: pending`` → ``status: done`` in the frontmatter only.

        The replace is bounded to the frontmatter block so a ``status: pending``
        mention in the task body can never be corrupted.
        """
        text = task_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            logger.warning("No leading frontmatter in %s — status not flipped", task_file.name)
            return
        body_start = text.find("\n---", 3)
        if body_start == -1:
            logger.warning("Unterminated frontmatter in %s — status not flipped", task_file.name)
            return

        frontmatter, body = text[:body_start], text[body_start:]
        new_frontmatter = frontmatter.replace(
            f"status: {TaskStatus.PENDING.value}",
            f"status: {TaskStatus.DONE.value}",
            1,
        )
        task_file.write_text(new_frontmatter + body, encoding="utf-8")
        logger.info("Flipped %s status to done", task_file.name)

    @staticmethod
    def _append_findings(findings_file: Path, task_slug: str, findings: str) -> None:
        """Append findings to _findings.md under a task heading."""
        existing = ""
        if findings_file.exists():
            try:
                existing = findings_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                existing = ""

        entry = f"\n### {task_slug}\n{findings}\n"
        content = existing.rstrip() + entry
        findings_file.write_text(content, encoding="utf-8")
        logger.info("Appended findings for %s to %s", task_slug, findings_file)

    @staticmethod
    def _serialize(result: CompleteStepResult) -> str:
        """Serialize result to a JSON string for tool output."""
        payload: dict[str, Any] = {
            "success": result.success,
            "plan": result.plan,
            "task": result.task,
            "order": result.order,
            "frame": result.frame,
            "message": result.message,
        }
        return json.dumps(payload)
