"""File write tool — write content to a file on disk."""
from __future__ import annotations

import asyncio
import os


class FileWriteTool:
    """Write content to a file, creating parent directories as needed."""

    name = "file_write"
    description = "Write a string to a file on disk. Creates missing parent directories."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self) -> None:
        pass

    async def __call__(self, path: str, content: str) -> str:
        loop = asyncio.get_event_loop()
        n_bytes = await loop.run_in_executor(None, _write_file, path, content)
        return f"Written {n_bytes} bytes to {path}"


def _write_file(path: str, content: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoded = content.encode("utf-8")
    with open(path, "wb") as f:
        f.write(encoded)
    return len(encoded)
