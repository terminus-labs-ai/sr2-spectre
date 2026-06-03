"""Core skill model and registry.

A **Skill** is a loadable knowledge package — a named bundle of conventions,
workflows, and procedural knowledge that an agent can load on demand.

A **SkillRegistry** manages discovery, registration, and content resolution
of skills. Skills can be registered programmatically or loaded from disk.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    """A loadable knowledge package.

    Attributes:
        name: Unique identifier (e.g., "sr2-conventions").
        description: One-line description of what this skill teaches.
        version: Semantic version string (e.g., "0.1.0").
        content: The full text content injected when the skill is loaded.
        tags: Keywords for filtering/categorization (e.g., ["sr2", "planning"]).
    """

    name: str
    description: str
    version: str = "0.1.0"
    content: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Skill name must not be empty")
        if not self.description:
            raise ValueError("Skill description must not be empty")


def load_skill_from_path(
    name: str,
    path: str | Path,
    version: str = "0.1.0",
    description: str = "",
    tags: list[str] | None = None,
) -> Skill:
    """Load a skill's content from a file on disk.

    Args:
        name: Skill identifier.
        path: Path to the markdown/text file containing the skill content.
        version: Skill version.
        description: Override description (if empty, derived from filename).
        tags: Optional tags.

    Returns:
        A Skill with content populated from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Skill content file not found: {p}")

    if not description:
        description = f"Skill: {name}"

    content = p.read_text(encoding="utf-8")
    return Skill(
        name=name,
        description=description,
        version=version,
        content=content,
        tags=tags or [],
    )


class SkillRegistry:
    """Registry for discoverable, loadable skills.

    Skills are registered by name and retrieved by name. The registry
    supports programmatic registration, file-based loading, and
    entry-point discovery.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill by its name.

        Args:
            skill: The Skill instance to register.

        Raises:
            ValueError: If a skill with this name is already registered.
        """
        if skill.name in self._skills:
            logger.warning(
                "Skill '%s' already registered — overwriting",
                skill.name,
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Retrieve a skill by name.

        Args:
            name: The skill identifier.

        Returns:
            The Skill, or None if not found.
        """
        return self._skills.get(name)

    def get_content(self, name: str) -> str | None:
        """Get the content text for a skill.

        Convenience method: equivalent to ``registry.get(name).content``.

        Args:
            name: The skill identifier.

        Returns:
            The skill's content text, or None if not found.
        """
        skill = self._skills.get(name)
        if skill is None:
            return None
        return skill.content

    def list_names(self) -> list[str]:
        """Return all registered skill names, sorted alphabetically."""
        return sorted(self._skills.keys())

    def find_by_tag(self, tag: str) -> list[Skill]:
        """Find all skills that include the given tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of matching Skills, sorted by name.
        """
        return sorted(
            [s for s in self._skills.values() if tag in s.tags],
            key=lambda s: s.name,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)
