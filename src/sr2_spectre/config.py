"""Spectre configuration models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


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


class SpectreConfig(BaseModel):
    """Top-level spectre configuration."""

    agent: AgentConfig
    plugins: list[PluginConfig] = Field(default_factory=list)
    heartbeat: HeartbeatConfig | None = None


class AgentConfig(BaseModel):
    """Agent-level configuration."""
    name: str = "spectre"
    model: str = "default"
    relay_base_url: str = "http://localhost:8000"
    system_prompt: str = ""
    tools: list[ToolConfig] = Field(default_factory=list)


def load_config(path: str | Path | None = None) -> SpectreConfig:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        return SpectreConfig(agent=AgentConfig())

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(p.read_text())
    return SpectreConfig(**raw)
