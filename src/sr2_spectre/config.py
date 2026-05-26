"""Spectre configuration models.

SpectreConfig is a superset of SR2's PipelineConfig:
  - agent:    spectre-owned concerns (name, tools, max_tool_rounds)
  - models:   dict[str, ModelConfig] — LLM endpoints
  - pipeline: SR2's native PipelineConfig — passed directly to SR2()
  - plugins:  list of plugin descriptors (unchanged)
  - heartbeat: optional heartbeat config (unchanged)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sr2.config.models import PipelineConfig


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


class AgentConfig(BaseModel):
    """Agent-level configuration — spectre-owned concerns only.

    model / base_url / system_prompt have been removed. They now live in
    the models and pipeline sections respectively.
    """
    name: str = "spectre"
    tools: list[ToolConfig] = Field(default_factory=list)
    max_tool_rounds: int = 10


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
