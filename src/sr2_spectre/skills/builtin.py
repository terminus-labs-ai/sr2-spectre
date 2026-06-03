"""Built-in SR2 conventions skill.

Packages 'how to work with SR2' as a loadable skill so any agent can
pick up SR2's conventions on demand rather than having them always in
context.

This is the default skill that ships with sr2-spectre.
"""

from __future__ import annotations

from sr2_spectre.skills.core import Skill

# ---------------------------------------------------------------------------
# SR2 Conventions Skill Content
# ---------------------------------------------------------------------------

SR2_CONVENTIONS_CONTENT = """# SR2 Conventions — Loadable Skill

> **Purpose:** This skill teaches an agent how to work with the SR2 agent
> framework. Load it when you need to understand SR2's patterns, pipeline,
> and conventions for building or interacting with SR2-powered agents.

## Overview

SR2 is an agent orchestration framework. It owns context compilation, the
tool execution loop, token budgets, and LLM calls. Spectre (sr2-spectre)
is the agent runtime that wraps SR2 — it owns agent identity, session
management, tool execution, and interfaces.

**Key principle:** SR2 is a stateless per-round context compiler. The Agent
seeds it with prior history and passes the newest message each turn.

## Pipeline Architecture

SR2's pipeline is event-driven with resolvers, providers, and transformers:

- **Resolvers** — inject content into the prompt on every turn (e.g., plan
  files, knowledge bases). Subscribed to events like `turn_start`.
- **Providers** — supply tool definitions, system prompts, or other
  context elements each turn.
- **Transformers** — modify the event stream or context (e.g.,
  summarization, step compaction).

All pipeline components are registered via entry points:
- `sr2.resolvers` — knowledge/content resolvers
- `sr2.tool_providers` — tool definition providers
- `sr2.transformers` — context transformers

## Durable Plans

SR2 agents use durable markdown plans for multi-step work:

- Plans live in `~/.sr2/plans/<plan-slug>/` (one directory per plan).
- `_plan.md` — plan-shared file with goal, status, and constraints contract.
- `NN-slug.md` — one file per atomic task with frontmatter (kind, plan,
  order, status, verify, title).

**Plan workflow:**
1. Ground: read the affected code and understand the codebase.
2. Record contract: document understanding and constraints in `_plan.md`.
3. Write plan: create the directory and task files.
4. Execute current task (injected as "Current Task" by PlanResolver).
5. Verify: run the task's `verify:` command.
6. Advance: set `status: done` when verified green.
7. Final validation: full test suite + re-check constraints.

**PlanResolver** (dynamic, per-turn):
- Re-reads plan/knowledge directories each turn.
- Injects L1 (project knowledge), L2 (active plan body), L3 (current task).
- Configurable: `plans_root`, `knowledge_root`, `project`, `max_tokens`.

## Knowledge Layers

Three-layer injection keeps the working set tight:

- **L1 — Project Knowledge:** Durable, cross-plan files under
  `~/.sr2/knowledge/<project>/`. Filtered by `kind: project-knowledge`
  frontmatter with matching `project` field.
- **L2 — Plan Shared:** The `_plan.md` body of the currently open plan
  (goal + constraints).
- **L3 — Current Task:** The lowest-order `pending` task from the open plan.

## Frame Primitive (Advanced)

Every content block can be tagged with `meta["frame"]` — a purpose-scoped
span. Closing a frame releases its blocks from the prompt. This enables
step-compaction: when a plan step completes, the transformer burns the
now-redundant context for that step.

- Frame ID format: `plan:<plan-slug>/<task-slug>`
- Event: `plan_step_completed` carries the closed frame id.
- Transformer drops blocks whose frame matches; leaves a breadcrumb.

## Tool System

Spectre owns tool execution; SR2 owns tool definition injection.

- Tools are registered in `ToolRegistry` (spectre) and surfaced via
  `SpectreToolProvider` (SR2 pipeline entry point).
- Built-in tools: `terminal`, `file_read`, `file_write`, `edit`, `grep`,
  `glob`, `web_search`, `complete_step`.
- MCP tools are discovered dynamically via MCP client connections.

## Configuration

Spectre uses a 4-tier config merge:
1. `$SR2_HOME/config.yaml` — global defaults
2. `$SR2_HOME/spectre.yaml` — spectre-specific defaults
3. `<cwd>/.spectre.yaml` — project overrides
4. Positional file (e.g., `~/.sr2/agents/edi.yaml`) — wins over all

Config supports `extends:` for inheritance chains. Use `sr2-spectre config show --dry-run` for inspection.

## Key Conventions

- **One step at a time:** Never modify multiple plan tasks in one pass.
- **Verify before advancing:** A `done` task means verified green.
- **Don't skip grounding:** Understand the codebase before writing.
- **Tests are not optional:** Logic side gets unit tests; engine side gets run verification.
- **Decouple logic from engine:** Game rules are plain testable code; engine layer is a thin shell.
"""


def get_sr2_conventions_skill() -> Skill:
    """Return the built-in SR2 conventions skill."""
    return Skill(
        name="sr2-conventions",
        description="How to work with SR2: pipeline, plans, knowledge layers, tools, and conventions.",
        version="0.1.0",
        content=SR2_CONVENTIONS_CONTENT,
        tags=["sr2", "conventions", "pipeline", "planning"],
    )


# Default skill instances for registry bootstrap
DEFAULT_SKILLS = [
    get_sr2_conventions_skill(),
]
