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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sr2.config.models import PipelineConfig

from sr2_spectre.config_merge import merge_configs
from sr2_spectre.path_resolution import resolve_path


@dataclass
class ProvenanceValue:
    """Tracks the winning source for a config value.

    Attributes:
        value: The actual config value (any type).
        source: Human-readable description of the file/tier that contributed
                this value. Examples:
                  "~/.sr2/config.yaml (global)"
                  "/project/.spectre.yaml (project)"
                  "/project/agents/base.yaml (extends)"
    """
    value: Any
    source: str


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
    max_tool_rounds: int = 40
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


class StartupConfigError(Exception):
    """Raised when config validation fails at startup."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


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


def _tier_label(path: Path, sr2_home: Path, cwd: Path) -> str:
    """Return a human-readable tier label for a config file path."""
    try:
        path = path.resolve()
        sr2_home = sr2_home.resolve()
        cwd = cwd.resolve()
    except Exception:
        pass

    # Check if it's under sr2_home
    try:
        path.relative_to(sr2_home)
        name = path.name
        if name == "config.yaml":
            return f"{path} (global)"
        return f"{path} (global-spectre)"
    except ValueError:
        pass

    # Check if it's under cwd
    try:
        path.relative_to(cwd)
        return f"{path} (project)"
    except ValueError:
        pass

    return f"{path} (extends)"


def _build_provenance(
    config: dict,
    source: str,
) -> dict:
    """Build a provenance map for a config dict with all top-level keys from source."""
    return {
        key: ProvenanceValue(value=val, source=source)
        for key, val in config.items()
    }


def _merge_provenance(
    parent_provenance: dict,
    child_provenance: dict,
    child_config: dict,
    parent_config: dict,
) -> dict:
    """Merge two provenance maps: child wins for keys it provides.

    For keys only in parent, keep parent provenance.
    For keys in child, use child provenance (child wins).
    """
    result = dict(parent_provenance)
    for key in child_config:
        if key in child_provenance:
            result[key] = child_provenance[key]
        else:
            # Key came from child but wasn't explicitly tracked — use child source
            # This shouldn't normally happen but handle it gracefully
            if child_provenance:
                first_source = next(iter(child_provenance.values())).source
            else:
                first_source = "unknown"
            result[key] = ProvenanceValue(value=child_config[key], source=first_source)
    return result


def _resolve_extends_with_provenance(
    config: dict,
    declaring_file: Path,
    declaring_source: str,
    env: dict[str, str] | None = None,
    _chain: list[Path] | None = None,
) -> tuple[dict, dict]:
    """Resolve extends: key recursively, tracking provenance.

    Returns:
        (resolved_config, provenance_map) where provenance_map has the same
        top-level keys as resolved_config, with ProvenanceValue leaves.
    """
    if _chain is None:
        _chain = []

    declaring_file = declaring_file.resolve()

    if declaring_file in _chain:
        chain_str = " -> ".join(str(p) for p in _chain) + f" -> {declaring_file}"
        raise CircularExtendsError(
            f"Circular 'extends:' detected: {chain_str}"
        )

    current_chain = _chain + [declaring_file]

    extends_raw = config.get("extends")
    if extends_raw is None:
        # No extends: provenance is declaring_file for all keys
        child_without_extends = {k: v for k, v in config.items() if k != "extends"}
        provenance = _build_provenance(child_without_extends, declaring_source)
        return child_without_extends, provenance

    # Resolve the extends path
    parent_path = resolve_path(str(extends_raw), declaring_file, env)

    if not parent_path.exists():
        raise FileNotFoundError(
            f"Extended config file not found: {parent_path} "
            f"(referenced from {declaring_file})"
        )

    # Load the parent file
    parent_raw = yaml.safe_load(parent_path.read_text())
    if parent_raw is None:
        parent_raw = {}

    parent_source = f"{parent_path} (extends)"

    # Recursively resolve the parent's own extends chain
    parent_resolved, parent_provenance = _resolve_extends_with_provenance(
        parent_raw,
        declaring_file=parent_path,
        declaring_source=parent_source,
        env=env,
        _chain=current_chain,
    )

    # Strip 'extends' from the declaring config
    child_without_extends = {k: v for k, v in config.items() if k != "extends"}
    child_provenance = _build_provenance(child_without_extends, declaring_source)

    # Merge: parent is base, child on top
    merged = merge_configs(parent_resolved, child_without_extends)

    # Merge provenance: child keys take child provenance, parent-only keys keep parent provenance
    merged_provenance = _merge_provenance(
        parent_provenance=parent_provenance,
        child_provenance=child_provenance,
        child_config=child_without_extends,
        parent_config=parent_resolved,
    )

    return merged, merged_provenance


def load_config_with_provenance(
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Load config tracking the winning source for each top-level key.

    Returns:
        (merged_config, provenance_map)
        provenance_map has the same top-level key structure as merged_config,
        with values as ProvenanceValue(value, source) objects.

    Args:
        cwd: Working directory for tier 3 lookup. Defaults to Path.cwd().
        env: Environment variables dict. Defaults to os.environ.
    """
    if cwd is None:
        cwd = Path.cwd()
    if env is None:
        env = dict(os.environ)

    sr2_home = resolve_sr2_home(env)

    tier_paths = [
        sr2_home / "config.yaml",
        sr2_home / "spectre.yaml",
        cwd / ".spectre.yaml",
    ]

    result: dict = {}
    result_provenance: dict = {}

    for path in tier_paths:
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text())
        if raw is None:
            raw = {}

        source = _tier_label(path, sr2_home, cwd)
        tier_config, tier_provenance = _resolve_extends_with_provenance(
            raw,
            declaring_file=path,
            declaring_source=source,
            env=env,
        )

        # Merge config
        result = merge_configs(result, tier_config)

        # Merge provenance: current tier wins for keys it provides
        result_provenance = _merge_provenance(
            parent_provenance=result_provenance,
            child_provenance=tier_provenance,
            child_config=tier_config,
            parent_config=result,
        )

    return result, result_provenance


def _resolve_tiers_with_provenance(
    paths: list[Path],
    sr2_home: Path,
    cwd: Path,
    env: dict[str, str] | None,
    require_last: bool = False,
) -> tuple[dict, dict]:
    """Resolve and merge an ordered list of tier paths, tracking provenance.

    Shared resolution core for both the tier-only and positional-file-aware
    loaders. Each path is yaml-loaded, extends-resolved (with provenance), then
    merged in order (later paths win). Missing files are silently skipped,
    EXCEPT the final path when ``require_last`` is True (the positional file
    must exist).

    De-dup: a path whose resolved absolute location was already processed is
    skipped, so passing a tier file as the positional file does not merge it
    twice.

    Args:
        paths: Ordered tier paths, lowest to highest priority.
        sr2_home: Resolved SR2_HOME (for tier labelling).
        cwd: Resolved working directory (for tier labelling).
        env: Environment for ${VAR} interpolation in extends paths.
        require_last: If True, the last path must exist (else FileNotFoundError).

    Returns:
        (merged_config, provenance_map)
    """
    result: dict = {}
    result_provenance: dict = {}
    seen: set[Path] = set()

    for index, path in enumerate(paths):
        is_last = index == len(paths) - 1
        if not path.exists():
            if is_last and require_last:
                raise FileNotFoundError(f"Config not found: {path}")
            continue

        resolved = path.resolve()
        if resolved in seen:
            # De-dup: already merged this file at a lower tier.
            continue
        seen.add(resolved)

        raw = yaml.safe_load(path.read_text())
        if raw is None:
            raw = {}

        source = _tier_label(path, sr2_home, cwd)
        tier_config, tier_provenance = _resolve_extends_with_provenance(
            raw,
            declaring_file=path,
            declaring_source=source,
            env=env,
        )

        result = merge_configs(result, tier_config)
        result_provenance = _merge_provenance(
            parent_provenance=result_provenance,
            child_provenance=tier_provenance,
            child_config=tier_config,
            parent_config=result,
        )

    return result, result_provenance


def load_lower_tiers(
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Resolve and merge tiers 1-3 only (no positional file).

    Tier order (lowest to highest):
    1. $SR2_HOME/config.yaml
    2. $SR2_HOME/spectre.yaml
    3. <cwd>/.spectre.yaml

    Missing files are silently skipped. Returns the merged dict (may be empty).
    Used by the CLI to overlay the positional file (tier 4) on top while keeping
    the positional file's own load behind the patchable load_config seam.
    """
    if cwd is None:
        cwd = Path.cwd()
    if env is None:
        env = dict(os.environ)

    sr2_home = resolve_sr2_home(env)
    paths = [
        sr2_home / "config.yaml",
        sr2_home / "spectre.yaml",
        cwd / ".spectre.yaml",
    ]
    config, _ = _resolve_tiers_with_provenance(
        paths, sr2_home=sr2_home, cwd=cwd, env=env, require_last=False
    )
    return config


def load_resolved_config_with_provenance(
    positional_path: str | Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Resolve the unified 4-tier config with the positional file at tier 4.

    Tier order (lowest to highest priority):
    1. $SR2_HOME/config.yaml
    2. $SR2_HOME/spectre.yaml
    3. <cwd>/.spectre.yaml
    4. extends-resolved(<positional_path>)  — wins over all

    Missing tier files (1-3) are silently skipped; the positional file must
    exist (FileNotFoundError otherwise). If the positional path resolves to the
    same file as a lower tier, it is not merged twice. Circular extends in any
    file raises CircularExtendsError.

    Args:
        positional_path: Path to the tier-4 config file (required).
        cwd: Working directory for tier 3 lookup. Defaults to Path.cwd().
        env: Environment variables dict. Defaults to os.environ.

    Returns:
        (merged_config, provenance_map) — provenance keys contributed by the
        positional file carry that file's source label.
    """
    if cwd is None:
        cwd = Path.cwd()
    if env is None:
        env = dict(os.environ)

    sr2_home = resolve_sr2_home(env)

    paths = [
        sr2_home / "config.yaml",
        sr2_home / "spectre.yaml",
        cwd / ".spectre.yaml",
        Path(positional_path),
    ]

    return _resolve_tiers_with_provenance(
        paths,
        sr2_home=sr2_home,
        cwd=cwd,
        env=env,
        require_last=True,
    )


def load_resolved_config(
    positional_path: str | Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Resolve the unified 4-tier config with the positional file at tier 4.

    Same resolution as load_resolved_config_with_provenance but returns only
    the merged config dict. See that function for tier order and semantics.

    Args:
        positional_path: Path to the tier-4 config file (required).
        cwd: Working directory for tier 3 lookup. Defaults to Path.cwd().
        env: Environment variables dict. Defaults to os.environ.

    Returns:
        The merged config dict.
    """
    config, _ = load_resolved_config_with_provenance(
        positional_path, cwd=cwd, env=env
    )
    return config


def format_dry_run(
    config: dict,
    provenance: dict,
    errors: list[str],
    include_content: bool = False,
    show_provenance: bool = True,
) -> str:
    """Format the dry-run report as a string.

    Outputs:
    - Merged YAML with inline provenance comments (if show_provenance=True)
    - Validation errors section
    - Exit-code hint (0 = clean, 1 = errors)

    Args:
        config: The merged config dict.
        provenance: The provenance map from load_config_with_provenance.
        errors: List of validation error strings from validate_config.
        include_content: If True, include raw file content (reserved, not used).
        show_provenance: If True, annotate each top-level key with its source.

    Returns:
        Formatted report string.
    """
    lines: list[str] = []

    if config:
        if show_provenance:
            # Output each top-level key as YAML with a provenance comment
            for key, value in config.items():
                key_yaml = yaml.dump({key: value}, default_flow_style=False).rstrip()
                pv = provenance.get(key)
                if pv is not None:
                    # Add comment to first line
                    first_line, *rest_lines = key_yaml.split("\n")
                    annotated = f"{first_line}  # ← {pv.source}"
                    if rest_lines:
                        key_yaml = "\n".join([annotated] + rest_lines)
                    else:
                        key_yaml = annotated
                lines.append(key_yaml)
        else:
            # Plain YAML — no annotations
            lines.append(yaml.dump(config, default_flow_style=False).rstrip())
    else:
        lines.append("{}")

    if errors:
        lines.append("")
        lines.append("errors:")
        for err in errors:
            lines.append(f"  - {err}")

    exit_code = 1 if errors else 0
    lines.append("")
    lines.append(f"# exit code: {exit_code}")

    return "\n".join(lines) + "\n"


def validate_config(config: dict) -> list[str]:
    """Return a list of validation error strings. Empty = valid.

    Rules:
    - If agent key present and has 'name': must be a non-empty string.
    - If models key present: must be a dict; each entry must have a non-empty 'model' field.
    - If pipeline key present and has 'layers': each layer must have a non-empty 'name';
      each resolver/transformer within a layer must be a dict.
    """
    errors: list[str] = []

    # --- agent ---
    agent = config.get("agent")
    if isinstance(agent, dict) and "name" in agent:
        name = agent["name"]
        if not isinstance(name, str) or not name:
            errors.append("agent.name must be a non-empty string")

    # --- models ---
    models = config.get("models")
    if models is not None:
        if not isinstance(models, dict):
            errors.append("models must be a dict mapping names to model configs")
        else:
            for entry_name, entry in models.items():
                if not isinstance(entry, dict):
                    errors.append(
                        f"models.{entry_name} must be a dict"
                    )
                    continue
                model_field = entry.get("model")
                if model_field is None:
                    errors.append(
                        f"models.{entry_name} is missing required 'model' field"
                    )
                elif not isinstance(model_field, str) or not model_field:
                    errors.append(
                        f"models.{entry_name}.model must be a non-empty string"
                    )

    # --- pipeline ---
    pipeline = config.get("pipeline")
    if isinstance(pipeline, dict) and "layers" in pipeline:
        layers = pipeline["layers"]
        if isinstance(layers, list):
            for i, layer in enumerate(layers):
                if not isinstance(layer, dict):
                    errors.append(f"pipeline.layers[{i}] must be a dict")
                    continue
                layer_name = layer.get("name")
                if layer_name is None:
                    errors.append(
                        f"pipeline.layers[{i}] is missing required 'name' field"
                    )
                elif not isinstance(layer_name, str) or not layer_name:
                    errors.append(
                        f"pipeline.layers[{i}].name must be a non-empty string"
                    )
                for section in ("resolvers", "transformers"):
                    items = layer.get(section)
                    if items is None:
                        continue
                    if isinstance(items, list):
                        for j, item in enumerate(items):
                            if not isinstance(item, dict):
                                errors.append(
                                    f"pipeline.layers[{i}].{section}[{j}] must be a dict"
                                )

    return errors


def load_and_validate(
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Load, merge, extend, and validate config. Raises StartupConfigError on any error.

    Calls load_merged_config first. Structural errors (CircularExtendsError,
    ConfigPathError) propagate directly — they abort immediately. Validation errors
    from validate_config are aggregated and raised together as StartupConfigError.

    Args:
        cwd: Working directory for tier 3 lookup. Defaults to Path.cwd().
        env: Environment variables dict. Defaults to os.environ.

    Returns:
        The validated merged config dict.

    Raises:
        CircularExtendsError: If a circular extends chain is detected.
        ConfigPathError: If an unresolved ${VAR} is found in a path.
        StartupConfigError: If validate_config returns any errors.
    """
    merged = load_merged_config(cwd=cwd, env=env)
    errors = validate_config(merged)
    if errors:
        raise StartupConfigError(errors)
    return merged


def load_config(source: str | Path | dict) -> SpectreConfig:
    """Build a SpectreConfig from a YAML file path or a pre-merged dict.

    - When ``source`` is a str/Path: load YAML from that file.
      Raises FileNotFoundError if the path does not exist.
    - When ``source`` is a dict: treat it as the already-merged config.

    Raises pydantic.ValidationError if the data is structurally invalid.
    """
    if isinstance(source, dict):
        return SpectreConfig(**source)

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {source}")

    raw = yaml.safe_load(p.read_text())
    return SpectreConfig(**raw)
