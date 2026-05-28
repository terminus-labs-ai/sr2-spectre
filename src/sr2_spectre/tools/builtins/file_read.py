"""File read tool — read a file from disk."""
from __future__ import annotations

import asyncio
import os


class FileReadTool:
    """Read the contents of a file from the local filesystem."""

    name = "file_read"
    description = "Read the contents of a file from disk and return them as a string."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, max_bytes: int = 1_000_000) -> None:
        self.max_bytes = max_bytes

    async def __call__(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")

        size = os.path.getsize(path)
        if size > self.max_bytes:
            raise ValueError(
                f"File too large: {path} is {size} bytes (limit: {self.max_bytes})"
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_file, path)


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()
