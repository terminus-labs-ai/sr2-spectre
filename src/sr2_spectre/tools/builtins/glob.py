"""Glob tool — find files matching a shell-style pattern."""
from __future__ import annotations

import asyncio
import glob as _glob


class GlobTool:
    """Find files matching a glob pattern, relative to a root directory.

    Returns matching paths relative to ``path``, sorted ascending and
    newline-joined. Supports recursive ``**`` matching.
    """

    name = "glob"
    description = (
        "Find files matching a shell-style glob pattern. Returns paths "
        "relative to the search directory, sorted ascending. Use '**' for "
        "recursive matching across nested directories."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g. '*.py' or '**/*.md').",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search. Returned paths are relative to it.",
                "default": ".",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self) -> None:
        pass

    async def __call__(self, pattern: str, path: str = ".") -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _glob_search, pattern, path)


def _glob_search(pattern: str, path: str) -> str:
    matches = _glob.glob(pattern, root_dir=path, recursive=True)
    if not matches:
        return "No files found."
    return "\n".join(sorted(matches))
