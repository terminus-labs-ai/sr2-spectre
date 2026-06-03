"""Planning subsystem for Spectre.

Durable, plan-aware context injection: a directory-per-plan layout on disk
plus a dynamic resolver that re-anchors the relevant plan/task/knowledge into
the LLM context each turn. Planning is a Spectre concern (not SR2 core); the
resolver plugs into SR2 via the ``sr2.resolvers`` entry point.

Public API:
  - ``LayerBudget`` — priority-aware token-budget allocator (pure logic).
  - ``PlanResolver`` / ``PlanResolverError`` — the dynamic resolver.
  - frontmatter parsing (``parse_file``, ``parse_frontmatter``,
    ``extract_raw_frontmatter``) and the frozen frontmatter models.
"""

from __future__ import annotations

from sr2_spectre.planning.budget import LayerBudget
from sr2_spectre.planning.frontmatter import (
    extract_raw_frontmatter,
    parse_file,
    parse_frontmatter,
    split_frontmatter,
)
from sr2_spectre.planning.models import (
    KnowledgeFrontmatter,
    PlanFileFrontmatter,
    PlanFrontmatter,
    PlanStatus,
    RECOGNIZED_KINDS,
    TaskFrontmatter,
    TaskStatus,
    get_frontmatter_class,
)
from sr2_spectre.planning.resolver import PlanResolver, PlanResolverError

__all__ = [
    "LayerBudget",
    "PlanResolver",
    "PlanResolverError",
    "extract_raw_frontmatter",
    "parse_file",
    "parse_frontmatter",
    "split_frontmatter",
    "KnowledgeFrontmatter",
    "PlanFileFrontmatter",
    "PlanFrontmatter",
    "PlanStatus",
    "RECOGNIZED_KINDS",
    "TaskFrontmatter",
    "TaskStatus",
    "get_frontmatter_class",
]
