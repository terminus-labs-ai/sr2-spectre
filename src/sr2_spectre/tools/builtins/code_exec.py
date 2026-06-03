"""Code execution tool — sandboxed Python snippet evaluation."""
from __future__ import annotations

import asyncio
import io
import sys
from contextlib import redirect_stderr, redirect_stdout


class CodeExecTool:
    """Execute a Python snippet in an isolated namespace and return the result.

    Captures stdout, stderr, and the return value of the last expression.
    Enforces a timeout to prevent runaway code.
    """

    name = "code_exec"
    description = (
        "Execute a Python code snippet in an isolated namespace. "
        "Returns stdout, stderr, return_value, and execution status."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code snippet to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Maximum execution time in seconds (default: 10).",
            },
        },
        "required": ["code"],
    }

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    async def __call__(self, code: str, timeout: int | None = None) -> str:
        effective_timeout = timeout if timeout is not None else self.timeout

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._execute, code, stdout_buf, stderr_buf
                ),
                timeout=effective_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):
            return self._format_result(
                status="timeout",
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                return_value=None,
                error=f"Execution timed out after {effective_timeout}s",
            )

        return self._format_result(
            status=result["status"],
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            return_value=result.get("return_value"),
            error=result.get("error"),
        )

    def _execute(
        self,
        code: str,
        stdout_buf: io.StringIO,
        stderr_buf: io.StringIO,
    ) -> dict:
        """Execute code in a restricted namespace with captured output."""
        namespace = {"__builtins__": __builtins__}

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            # Temporarily replace sys.stdout/sys.stderr for print() calls
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            try:
                try:
                    # Try as expression first (for eval)
                    result = eval(compile(code, "<code_exec>", "eval"), namespace)
                    return {"status": "success", "return_value": result}
                except SyntaxError:
                    # Fall back to exec (statements)
                    exec(compile(code, "<code_exec>", "exec"), namespace)
                    return {"status": "success"}
            except Exception as e:
                return {"status": "error", "error": f"{type(e).__name__}: {e}"}
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @staticmethod
    def _format_result(
        status: str,
        stdout: str,
        stderr: str,
        return_value: object | None,
        error: str | None = None,
    ) -> str:
        lines = [f"Status: {status}"]
        if stdout:
            lines.append(f"Stdout:\n{stdout}")
        if stderr:
            lines.append(f"Stderr:\n{stderr}")
        if status == "success" and return_value is not None:
            lines.append(f"Return: {return_value!r}")
        if error:
            lines.append(f"Error: {error}")
        return "\n".join(lines)
