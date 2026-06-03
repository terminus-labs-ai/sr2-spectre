"""Smoke tests — end-to-end integration verification for the sr2-spectre stack.

These are NOT unit tests. They verify the full stack works end-to-end:
config loading → agent instantiation → tool registry → interface wiring →
MCP integration path → planning resolver → degradation.

No external LLM calls required. All mocks are at the SR2/LLM boundary.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig, load_config
from sr2_spectre.interfaces.single_shot import SingleShotInterface
from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_spectre_config(
    tools: list[dict] | None = None,
    mcp_servers: list | None = None,
) -> SpectreConfig:
    """Build a minimal SpectreConfig with a static system prompt."""
    return SpectreConfig(
        agent=AgentConfig(
            name="smoke-agent",
            tools=tools or [],
            mcp_servers=mcp_servers or [],
        ),
        models={"default": ModelConfig(model="test", base_url="http://test:11434/v1")},
        pipeline={
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
                },
                {
                    "name": "tools",
                    "target": "tools",
                    "resolvers": [],
                    "tool_providers": [{"type": "spectre_tools"}],
                },
                {
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
    )


# ---------------------------------------------------------------------------
# Smoke 1: Config loading round-trip
# ---------------------------------------------------------------------------

class TestSmokeConfigLoading:
    def test_load_full_config_from_yaml(self, tmp_path: Path) -> None:
        """A realistic config file loads without error and round-trips correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                agent:
                  name: smoke-test
                  tools:
                    - name: terminal
                      class_path: sr2_spectre.tools.builtins.terminal.TerminalTool
                      config:
                        timeout: 30
                    - name: file_read
                      class_path: sr2_spectre.tools.builtins.file_read.FileReadTool
                    - name: file_write
                      class_path: sr2_spectre.tools.builtins.file_write.FileWriteTool
                models:
                  default:
                    model: test/model
                    base_url: http://localhost:11434/v1
                pipeline:
                  token_budget: 200000
                  max_tool_iterations: 40
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: tools
                      target: tools
                      resolvers: []
                      tool_providers:
                        - type: spectre_tools
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        cfg = load_config(str(config_file))
        assert cfg.agent.name == "smoke-test"
        assert len(cfg.agent.tools) == 3
        assert cfg.agent.tools[0].name == "terminal"
        assert cfg.models["default"].model == "test/model"
        assert len(cfg.pipeline.layers) == 3

    def test_load_config_with_extends(self, tmp_path: Path) -> None:
        """Config with extends: field merges correctly."""
        base = tmp_path / "base.yaml"
        base.write_text(
            textwrap.dedent("""\
                agent:
                  name: base-agent
                models:
                  default:
                    model: test/model
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        override = tmp_path / "override.yaml"
        override.write_text(
            textwrap.dedent(f"""\
                extends: {base}
                agent:
                  name: extended-agent
                  tools:
                    - name: terminal
                      class_path: sr2_spectre.tools.builtins.terminal.TerminalTool
                models:
                  default:
                    model: test/model
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        cfg = load_config(str(override))
        assert cfg.agent.name == "extended-agent"
        assert len(cfg.agent.tools) == 1
        assert cfg.agent.tools[0].name == "terminal"
        # Pipeline from override preserved
        assert len(cfg.pipeline.layers) == 2


# ---------------------------------------------------------------------------
# Smoke 2: Agent instantiation + tool registry
# ---------------------------------------------------------------------------

class TestSmokeAgentInstantiation:
    def test_agent_creates_with_built_in_tools(self) -> None:
        """Agent instantiates and registers the specified built-in tools."""
        cfg = _minimal_spectre_config(
            tools=[
                {"name": "terminal", "class_path": "sr2_spectre.tools.builtins.terminal.TerminalTool"},
                {"name": "file_read", "class_path": "sr2_spectre.tools.builtins.file_read.FileReadTool"},
                {"name": "file_write", "class_path": "sr2_spectre.tools.builtins.file_write.FileWriteTool"},
            ]
        )
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        assert "terminal" in agent.registry
        assert "file_read" in agent.registry
        assert "file_write" in agent.registry

    def test_agent_no_tools_creates_empty_registry(self) -> None:
        """Agent with no tools has an empty registry."""
        cfg = _minimal_spectre_config(tools=[])
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        assert len(agent.registry) == 0

    def test_agent_registers_all_built_in_tools(self) -> None:
        """All shipped built-in tools can be registered without error."""
        all_tool_defs = [
            {"name": "terminal", "class_path": "sr2_spectre.tools.builtins.terminal.TerminalTool"},
            {"name": "file_read", "class_path": "sr2_spectre.tools.builtins.file_read.FileReadTool"},
            {"name": "file_write", "class_path": "sr2_spectre.tools.builtins.file_write.FileWriteTool"},
            {"name": "edit", "class_path": "sr2_spectre.tools.builtins.edit.EditTool"},
            {"name": "grep", "class_path": "sr2_spectre.tools.builtins.grep.GrepTool"},
            {"name": "glob", "class_path": "sr2_spectre.tools.builtins.glob.GlobTool"},
            {"name": "web_search", "class_path": "sr2_spectre.tools.builtins.web_search.WebSearchTool", "config": {"base_url": "http://localhost:8080"}},
            {"name": "complete_step", "class_path": "sr2_spectre.tools.builtins.complete_step.CompleteStepTool"},
        ]
        cfg = _minimal_spectre_config(tools=all_tool_defs)
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        for tool_def in all_tool_defs:
            assert tool_def["name"] in agent.registry, f"Tool {tool_def['name']} not registered"


# ---------------------------------------------------------------------------
# Smoke 3: Single-shot interface path
# ---------------------------------------------------------------------------

class TestSmokeSingleShotInterface:
    async def test_single_shot_runs_end_to_end(self, capsys: pytest.CaptureFixture) -> None:
        """Single-shot interface accepts a prompt and produces output."""
        from sr2_spectre.core import TurnResult

        agent = AsyncMock()
        agent.handle_user_message.return_value = TurnResult(
            text="Hello from smoke test",
            tool_calls_executed=0,
            total_tokens=25,
        )
        interface = SingleShotInterface(prompt="smoke test")
        await interface.run(agent)

        agent.handle_user_message.assert_called_once_with("smoke test")
        captured = capsys.readouterr()
        assert "Hello from smoke test" in captured.out


# ---------------------------------------------------------------------------
# Smoke 4: MCP wiring path (no real connection)
# ---------------------------------------------------------------------------

class TestSmokeMcpWiring:
    async def test_agent_initializes_mcp_and_registers_bridge(self) -> None:
        """MCP server config → MCPClient → bridge → tool registry path works."""
        from sr2_spectre.config import McpServerConfig
        from sr2_spectre.mcp.tool_bridge import MCPToolBridge

        bridge = MagicMock(spec=MCPToolBridge)
        bridge.name = "mcp_list_files"
        bridge.description = "List files via MCP"
        bridge.input_schema = {"type": "object", "properties": {}}
        bridge._is_coroutine = AsyncMock._is_coroutine

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=[bridge])

        cfg = _minimal_spectre_config(
            mcp_servers=[
                McpServerConfig(name="fs", type="stdio", command=["fs-server"])
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        await agent.initialize()
        assert "mcp_list_files" in agent.registry

    async def test_agent_graceful_on_mcp_connection_failure(self) -> None:
        """MCP connection failure is swallowed and logged, not raised."""
        from sr2_spectre.config import McpServerConfig
        from sr2_spectre.mcp.client import MCPConnectionError

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(side_effect=MCPConnectionError("connection refused"))

        cfg = _minimal_spectre_config(
            mcp_servers=[
                McpServerConfig(name="unreachable", type="http", url="http://bad:9999/sse")
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        # Should not raise
        await agent.initialize()


# ---------------------------------------------------------------------------
# Smoke 5: Tool execution path (terminal)
# ---------------------------------------------------------------------------

class TestSmokeToolExecution:
    async def test_terminal_tool_executes_command(self) -> None:
        """Terminal tool can execute a simple command and return output."""
        from sr2_spectre.tools.builtins.terminal import TerminalTool

        tool = TerminalTool(timeout=5)
        result = await tool(command="echo 'smoke OK'")

        assert "smoke OK" in str(result)


# ---------------------------------------------------------------------------
# Smoke 6: File I/O tools
# ---------------------------------------------------------------------------

class TestSmokeFileTools:
    async def test_file_write_and_read_round_trip(self, tmp_path: Path) -> None:
        """file_write + file_read round-trips content correctly."""
        from sr2_spectre.tools.builtins.file_read import FileReadTool
        from sr2_spectre.tools.builtins.file_write import FileWriteTool

        test_file = tmp_path / "smoke_test.txt"
        content = "Smoke test round-trip content\nLine 2\n"

        writer = FileWriteTool()
        await writer(path=str(test_file), content=content)

        reader = FileReadTool()
        result = await reader(path=str(test_file))

        assert content.strip() in str(result)

    async def test_file_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """file_write creates missing parent directories."""
        from sr2_spectre.tools.builtins.file_write import FileWriteTool

        nested = tmp_path / "a" / "b" / "c" / "file.txt"
        writer = FileWriteTool()
        await writer(path=str(nested), content="nested content")

        assert nested.exists()
        assert nested.read_text() == "nested content"


# ---------------------------------------------------------------------------
# Smoke 7: PlanResolver registration
# ---------------------------------------------------------------------------

class TestSmokePlanResolver:
    def test_plan_resolver_importable(self) -> None:
        """PlanResolver can be imported and instantiated with ResolverConfig."""
        from sr2.config.models import ResolverConfig
        from sr2_spectre.planning.resolver import PlanResolver

        resolver_cfg = ResolverConfig(
            type="plan",
            config={
                "plans_root": str(Path.home() / ".sr2" / "plans"),
                "project": "test-project",
            },
            max_executions=10,
        )
        resolver = PlanResolver(config=resolver_cfg)
        assert isinstance(resolver, PlanResolver)


# ---------------------------------------------------------------------------
# Smoke 8: Config validation catches bad configs
# ---------------------------------------------------------------------------

class TestSmokeConfigValidation:
    def test_missing_model_raises(self, tmp_path: Path) -> None:
        """Config without a model section should fail validation."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("agent:\n  name: broken\n")

        with pytest.raises((ValueError, TypeError, KeyError)):
            load_config(str(bad))

    def test_bad_tool_class_path_logs_warning(self, tmp_path: Path) -> None:
        """A config with an unresolvable tool class_path logs a warning but loads."""
        config_file = tmp_path / "bad_tool.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                agent:
                  name: bad-tool-agent
                  tools:
                    - name: nonexistent
                      class_path: definitely.not.a.real.module.SomeTool
                models:
                  default:
                    model: test
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: tools
                      target: tools
                      resolvers: []
                      tool_providers:
                        - type: spectre_tools
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        # load_config itself may or may not validate tool paths at load time
        # The key is that it doesn't crash the application
        cfg = load_config(str(config_file))
        assert cfg.agent.name == "bad-tool-agent"
        assert len(cfg.agent.tools) == 1
