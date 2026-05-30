"""Grep tool — search file contents for a pattern, line by line."""
from __future__ import annotations

import asyncio
import fnmatch
import os
import re


class GrepTool:
    """Search files for lines matching a pattern.

    Searches a single file or recursively walks a directory, emitting each
    matching line as ``{filepath}:{lineno}:{linetext}`` (1-based line numbers).
    Supports regex or literal matching and an optional basename glob filter.
    Binary (undecodable) files are skipped silently.
    """

    name = "grep"
    description = (
        "Search file contents for a pattern. Accepts a single file or a "
        "directory (searched recursively). Supports regex or literal matching "
        "and an optional glob to scope which files are searched."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Pattern to search for in each line.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search. Directories are searched recursively.",
                "default": ".",
            },
            "glob": {
                "type": "string",
                "description": "Only search files whose basename matches this glob (e.g. '*.py').",
            },
            "regex": {
                "type": "boolean",
                "description": "Treat pattern as a regular expression (default true). If false, match literally.",
                "default": True,
            },
        },
        "required": ["pattern"],
    }

    def __init__(self) -> None:
        pass

    async def __call__(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        regex: bool = True,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _grep, pattern, path, glob, regex
        )


def _grep(pattern: str, path: str, glob: str | None, regex: bool) -> str:
    if regex:
        compiled = re.compile(pattern)
    else:
        compiled = re.compile(re.escape(pattern))

    if os.path.isdir(path):
        files = []
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                if glob is not None and not fnmatch.fnmatch(filename, glob):
                    continue
                files.append(os.path.join(dirpath, filename))
    else:
        if glob is not None and not fnmatch.fnmatch(os.path.basename(path), glob):
            files = []
        else:
            files = [path]

    matches: list[str] = []
    for filepath in files:
        try:
            with open(filepath, encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    text = line.rstrip("\n")
                    if compiled.search(text):
                        matches.append(f"{filepath}:{lineno}:{text}")
        except UnicodeDecodeError:
            continue
        except OSError:
            continue

    if not matches:
        return "No matches found."
    return "\n".join(matches)
