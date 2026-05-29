"""Spectre configuration models.

SpectreConfig is a superset of SR2's PipelineConfig:
  - agent:    spectre-owned concerns (name, tools, max_tool_rounds)
  - models:   dict[str, ModelConfig] — LLM endpoints
  - pipeline: SR2's native PipelineConfig — passed directly to SR2()
  - plugins:  list of plugin descriptors (unchanged)
  - heartbeat: optional heartbeat config (unchanged)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sr2.config.models import PipelineConfig

from sr2_spectre.config_merge import merge_configs
from sr2_spectre.path_resolution import resolve_path


class ToolConfig(BaseModel):
    """Configuration for a single tool."""
    name: str
    class_path: str  # e.g. "sr2_spectre.tools.builtins.web_search:WebSearch"
    config: dict[str, Any] = Field(default_factory=dict)


class PluginConfig(BaseModel):
    """Configuration for a plugin."""
    name: str
    class_path: str  # e.g. "sr2_spectre.plugins.single_shot:SingleShotPlugin"
    config: dict[str, Any] = Field(default_factory=dict)


class HeartbeatConfig(BaseModel):
    """Heartbeat plugin configuration."""
    interval_seconds: int = 60
    callback: str | None = None  # class path or inline prompt


class ModelConfig(BaseModel):
    """Configuration for a single LLM endpoint."""
    model: str
    base_url: str | None = None


class McpServerConfig(BaseModel):
    """Configuration for a single MCP server connection."""
    name: str
    type: str                     # "stdio" or "http"
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""


class AgentConfig(BaseModel):
    """Agent-level configuration — spectre-owned concerns only.

    model / base_url / system_prompt have been removed. They now live in
    the models and pipeline sections respectively.
    """
    name: str = "spectre"
    tools: list[ToolConfig] = Field(default_factory=list)
    max_tool_rounds: int = 10
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)


class SpectreConfig(BaseModel):
    """Top-level spectre configuration.

    models and pipeline are required: spectre cannot start without knowing
    which LLM to call and what pipeline to run.
    """
    agent: AgentConfig
    models: dict[str, ModelConfig]
    pipeline: PipelineConfig
    plugins: list[PluginConfig] = Field(default_factory=list)
    heartbeat: HeartbeatConfig | None = None


class CircularExtendsError(Exception):
    """Raised when a circular 'extends:' chain is detected in config files."""


def resolve_extends(
    config: dict,
    declaring_file: Path,
    env: dict[str, str] | None = None,
    _chain: list[Path] | None = None,
) -> dict:
    """Resolve extends: key recursively, building the full inheritance chain.

    Loads the extended file first, then merges the declaring config on top.
    Supports chains (A extends B extends C). Circular references raise
    CircularExtendsError.

    The full chain is built before merging: given A extends B extends C,
    the chain is [C, B, A] and configs are merged left to right (C is base).

    Args:
        config: The already-parsed config dict from the declaring file.
        declaring_file: Absolute path to the declaring config file.
        env: Environment variables for ${VAR} interpolation in extends paths.
        _chain: Internal — tracks the chain of resolved files for cycle detection.

    Returns:
        The fully-merged config dict with extends applied.

    Raises:
        CircularExtendsError: If a file appears twice in the extends chain.
        FileNotFoundError: If the extended file does not exist.
    """
    if _chain is None:
        _chain = []

    # Normalise declaring_file to an absolute resolved path for cycle detection.
    declaring_file = declaring_file.resolve()

    # Detect cycle: this file already appears in the chain being built.
    if declaring_file in _chain:
        chain_str = " -> ".join(str(p) for p in _chain) + f" -> {declaring_file}"
        raise CircularExtendsError(
            f"Circular 'extends:' detected: {chain_str}"
        )

    # Record this file in the chain.
    current_chain = _chain + [declaring_file]

    extends_raw = config.get("extends")
    if extends_raw is None:
        # No extends: return config unchanged.
        return config

    # Resolve the extends path via FR10 rules.
    parent_path = resolve_path(str(extends_raw), declaring_file, env)

    if not parent_path.exists():
        raise FileNotFoundError(
            f"Extended config file not found: {parent_path} "
            f"(referenced from {declaring_file})"
        )

    # Load the parent file.
    parent_raw = yaml.safe_load(parent_path.read_text())
    if parent_raw is None:
        parent_raw = {}

    # Recursively resolve the parent's own extends chain.
    parent_resolved = resolve_extends(
        parent_raw,
        declaring_file=parent_path,
        env=env,
        _chain=current_chain,
    )

    # Strip 'extends' from the declaring config before merging.
    child_without_extends = {k: v for k, v in config.items() if k != "extends"}

    # Merge: parent is base, child on top.
    return merge_configs(parent_resolved, child_without_extends)


def resolve_sr2_home(env: dict[str, str] | None = None) -> Path:
    """Resolve SR2_HOME. Defaults to ~/.sr2 if env var unset.

    Args:
        env: Environment variables dict. Defaults to os.environ.

    Returns:
        Absolute resolved Path for SR2_HOME.
    """
    if env is None:
        env = dict(os.environ)

    raw = env.get("SR2_HOME")
    if raw is None:
        return Path("~/.sr2").expanduser().resolve()
    return Path(raw).expanduser().resolve()


def load_merged_config(
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Load and merge config from all four tiers.

    Tier order (lowest to highest priority):
    1. $SR2_HOME/config.yaml      — user global defaults
    2. $SR2_HOME/spectre.yaml     — spectre-specific defaults
    3. <cwd>/.spectre.yaml        — project overrides
    4. CLI args (deferred — not implemented, reserved)

    Missing files at any tier are silently skipped.
    Returns the merged config dict.

    Args:
        cwd: Working directory for tier 3 lookup. Defaults to Path.cwd().
        env: Environment variables dict. Defaults to os.environ.
    """
    if cwd is None:
        cwd = Path.cwd()

    sr2_home = resolve_sr2_home(env)

    tier_paths = [
        sr2_home / "config.yaml",
        sr2_home / "spectre.yaml",
        cwd / ".spectre.yaml",
    ]

    result: dict = {}
    for path in tier_paths:
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text())
        if raw is None:
            raw = {}
        raw = resolve_extends(raw, declaring_file=path, env=env)
        result = merge_configs(result, raw)

    return result


def load_config(path: str | Path) -> SpectreConfig:
    """Load config from a YAML file.

    Raises FileNotFoundError if path does not exist.
    Raises pydantic.ValidationError if the YAML is structurally invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(p.read_text())
    return SpectreConfig(**raw)
