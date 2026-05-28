"""Terminal tool — run shell commands as a subprocess."""
from __future__ import annotations

import asyncio


class TerminalTool:
    """Execute shell commands and return combined stdout+stderr."""

    name = "terminal"
    description = "Run a shell command and return its output (stdout and stderr combined)."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    async def __call__(self, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            communicate_coro = proc.communicate()
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                communicate_coro, timeout=self.timeout
            )
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            raise TimeoutError(f"Command timed out: {command}")

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        return stdout + stderr
