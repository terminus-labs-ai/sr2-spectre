"""Tests for config loading."""
import pytest
from pydantic import BaseModel
from sr2_spectre.config import (
    AgentConfig,
    HeartbeatConfig,
    PluginConfig,
    SpectreConfig,
    ToolConfig,
    load_config,
)


def test_default_config() -> None:
    cfg = SpectreConfig(agent=AgentConfig())
    assert cfg.agent.name == "spectre"
    assert cfg.agent.model == "default"
    assert cfg.agent.relay_base_url == "http://localhost:8000"
    assert cfg.agent.system_prompt == ""
    assert cfg.agent.tools == []
    assert cfg.plugins == []


def test_config_with_values() -> None:
    cfg = SpectreConfig(
        agent=AgentConfig(
            name="edi",
            model="anthropic/claude-sonnet-4",
            relay_base_url="http://relay:8000",
            system_prompt="You are EDI.",
        )
    )
    assert cfg.agent.name == "edi"
    assert cfg.agent.model == "anthropic/claude-sonnet-4"


def test_config_with_tools() -> None:
    cfg = SpectreConfig(
        agent=AgentConfig(
            tools=[
                ToolConfig(
                    name="search",
                    class_path="mytools.Search:SearchTool",
                    config={"engine": "google"},
                )
            ]
        )
    )
    assert len(cfg.agent.tools) == 1
    assert cfg.agent.tools[0].name == "search"


def test_config_with_plugins() -> None:
    cfg = SpectreConfig(
        agent=AgentConfig(),
        plugins=[
            PluginConfig(
                name="tui",
                class_path="sr2_spectre.plugins.tui:TUIPlugin",
            )
        ],
    )
    assert len(cfg.plugins) == 1
    assert cfg.plugins[0].name == "tui"


def test_heartbeat_config_defaults() -> None:
    hb = HeartbeatConfig()
    assert hb.interval_seconds == 60
    assert hb.callback is None


def test_load_config_file(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "agent:\n"
        "  name: test-agent\n"
        "  model: test-model\n"
        "  relay_base_url: http://test:9000\n"
        "  system_prompt: Test prompt\n"
    )
    cfg = load_config(str(config_file))
    assert cfg.agent.name == "test-agent"
    assert cfg.agent.model == "test-model"
    assert cfg.agent.relay_base_url == "http://test:9000"


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
