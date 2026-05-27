---
project: sr2-spectre
status: milestone-complete
priority: high
last_touched: 2026-05-25
energy_at_close: spent
next_action: "Run smoke test live: cd ~/git/sr2-spectre && uv run sr2-spectre smoke.yaml 'Hello'"
tags: [sr2, agent, runtime, sr2-powered]
---

# Handoff — sr2-spectre

## Where I left off

**Spec fully implemented.** The SR2-powered agent spec (`specs/sr2-powered-agent.md`) is done end-to-end. Spectre no longer runs its own LLM loop — `Agent` delegates all context compilation, tool definition injection, and LLM calls to SR2 via `sr2.seed_session()` + `sr2.turn()`.

Both repos committed:
- `~/git/sr2` — `revamp` branch, 2 commits: plugin-registry work + execution_count reset fix
- `~/git/sr2-spectre` — `master`, commit `8722261`

## What was done this session

**SR2 fixes (prereq):**
- Committed the plugin-registry work (~1007 tests) — `PluginRegistry`, `ToolProvider` kind, `Dependencies.extras`, `SR2.seed_session()`
- Fixed `orchestrator.py`: `execution_count` reset now covers `tool_providers` + `transformers` (not just resolvers). Without this, tools vanished from `CompletionRequest` after turn 1.

**Spectre (7 steps, 76 tests):**
1. `ToolRegistry.to_sr2_definitions()` — returns `list[ToolDefinition]` (SR2 native shape)
2. `SpectreToolProvider` (`providers.py`) + `sr2.tool_providers` entry point in `pyproject.toml`
3. Config restructure — `ModelConfig` added; `AgentConfig` loses `model`/`base_url`/`system_prompt`, gains `max_tool_rounds`; `SpectreConfig` requires `models` + `pipeline`; `smoke.yaml` migrated
4. `Agent.__init__` constructs `SR2` (not `LiteLLMCallable` directly)
5. `Agent.handle_user_message` — `seed_session` + `turn` loop, tool error recovery (FR13), `max_tool_rounds` guard (FR14)
6. Deleted `run_tool_loop`, `ToolRegistry.to_definitions()`, updated `cli.py` (config arg now required)

## Next physical action

1. **Live smoke test** — endpoint is up (`http://localhost:11438/v1`, `qwen3.6:27b` available):
   ```bash
   cd ~/git/sr2-spectre
   uv run sr2-spectre smoke.yaml "What is the capital of France?"
   ```
   Expected: response starts with "Aye aye commander"

2. **Review the diff** — `git show 8722261` or `git diff HEAD~1`. ~1400 lines added, ~1600 deleted. Key files: `agent.py`, `providers.py`, `config.py`.

3. **After review** — decide what's next:
   - Wire a real tool into `smoke.yaml` (`agent.tools`) and test a tool round-trip
   - Fix TUI plugin (`plugins/tui.py` still has stub methods, not wired to the new Agent)
   - Merge SR2 `revamp` branch to `main`

## Open questions / known gaps

- **TUI plugin is untouched** — `plugins/tui.py` has stub methods and was written against the old dict-based session. It won't work with the new `list[Message]` history on Agent.
- **SR2 `revamp` branch** — still not merged to `main`. The `sr2-spectre` pyproject uses `sr2 = { path = "../sr2", editable = true }` so it picks up `revamp` locally. No issue for now.
- **`core/session.py`** — orphaned. `Session` class is dead code (Agent uses `list[Message]` now). Safe to delete along with `test_session.py` when ready to clean up.
- **No live tool test** — smoke.yaml has no tools configured in `agent.tools`, so the `tools` layer fires but returns `[]`. Everything works, but no actual tool round-trip has been exercised against a live model yet.

## Don't forget about

None.
