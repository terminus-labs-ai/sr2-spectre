"""Tests for per-tool-result truncation guard (obsidian-tqc, layer 1).

Prevents a single oversized tool result from flooding the context window
and crashing the SR2/LLM request with exceed_context_size_error.

Covers:
  A. AgentConfig.tool_result_max_bytes default value
  B. _execute_tool truncates oversized results
  C. _execute_tool does NOT truncate results under the cap
  D. Truncated content includes a clear marker with size info
  E. Error results also get truncated
  F. Configurable cap via AgentConfig
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import ToolResultBlock, ToolUseBlock
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={"layers": [
            {"name": "system", "target": "system", "resolvers": [
                {"type": "static", "config": {"text": "You are helpful."}}
            ]},
        ]},
    )


# ---------------------------------------------------------------------------
# A. AgentConfig.tool_result_max_bytes default
# ---------------------------------------------------------------------------

class TestToolResultMaxBytesDefault:
    def test_default_is_64kb(self):
        """Default tool_result_max_bytes is 64KB (65536)."""
        cfg = AgentConfig()
        assert cfg.tool_result_max_bytes == 65536

    def test_custom_value_accepted(self):
        """Custom tool_result_max_bytes is accepted."""
        cfg = AgentConfig(tool_result_max_bytes=1024)
        assert cfg.tool_result_max_bytes == 1024


# ---------------------------------------------------------------------------
# B. _execute_tool truncates oversized results
# ---------------------------------------------------------------------------

class TestExecuteToolTruncation:
    @pytest.mark.asyncio
    async def test_truncation_kicks_in_at_cap(self):
        """A tool result exactly at the cap is NOT truncated."""
        from sr2_spectre.agent import Agent

        cap = 100
        cfg = _make_config(tool_result_max_bytes=cap)

        # Register a tool that returns exactly cap bytes
        tool_output = "x" * cap

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        # Register a tool that returns known content
        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # Should NOT be truncated — exactly at cap
        assert "truncated" not in result.content.lower()
        assert len(result.content) == cap

    @pytest.mark.asyncio
    async def test_truncation_kicks_in_above_cap(self):
        """A tool result exceeding the cap IS truncated."""
        from sr2_spectre.agent import Agent

        cap = 100
        cfg = _make_config(tool_result_max_bytes=cap)

        # Register a tool that returns content exceeding the cap
        tool_output = "x" * 200

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # Result content must be truncated
        assert "truncated" in result.content.lower()
        assert len(result.content) <= cap + 150  # cap + margin for the marker text

    @pytest.mark.asyncio
    async def test_truncation_preserves_prefix(self):
        """Truncated content preserves the beginning of the original output."""
        from sr2_spectre.agent import Agent

        cap = 50
        cfg = _make_config(tool_result_max_bytes=cap)

        # Content with meaningful prefix
        tool_output = "HEADER: important data at the start" + "x" * 500

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # The beginning of the content should be preserved
        assert "HEADER: important data at the start" in result.content

    @pytest.mark.asyncio
    async def test_truncation_marker_includes_original_size(self):
        """The truncation marker includes the original output size for debugging."""
        from sr2_spectre.agent import Agent

        cap = 50
        cfg = _make_config(tool_result_max_bytes=cap)

        tool_output = "x" * 1000

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # Marker should reference the original size
        assert "1000" in result.content or "1024" in result.content or "bytes" in result.content

    @pytest.mark.asyncio
    async def test_error_results_also_truncated(self):
        """Error results from tools are also subject to truncation."""
        from sr2_spectre.agent import Agent

        cap = 50
        cfg = _make_config(tool_result_max_bytes=cap)

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        # Register a tool that raises a huge exception
        def failing_tool():
            raise ValueError("x" * 1000)

        agent.register_tool(
            "fail_tool",
            "Fails with huge error",
            {},
            failing_tool,
        )

        block = ToolUseBlock(id="tu1", name="fail_tool", input={})
        result = await agent._execute_tool(block)

        assert result.is_error is True
        assert "truncated" in result.content.lower() or len(result.content) <= cap + 50

    @pytest.mark.asyncio
    async def test_small_results_unchanged(self):
        """Results well under the cap pass through unchanged."""
        from sr2_spectre.agent import Agent

        cfg = _make_config(tool_result_max_bytes=65536)

        tool_output = "small result"

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "small_tool",
            "Returns small output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="small_tool", input={})
        result = await agent._execute_tool(block)

        assert result.content == "small result"
        assert "truncated" not in result.content.lower()

    @pytest.mark.asyncio
    async def test_regression_oversized_tool_result_continues(self):
        """Regression test for spc-3 scenario: oversized tool result must NOT crash the run.

        Before this fix, a ~2.16M-token grep blob injected raw into the next
        SR2/LLM request caused exceed_context_size_error and crashed the run.
        With truncation, the tool returns a bounded result and the loop continues.
        """
        from sr2_spectre.agent import Agent

        # Simulate a very tight cap to reproduce the overflow scenario
        cap = 1000
        cfg = _make_config(tool_result_max_bytes=cap)

        # Simulate a massive tool result (like the grep blob from spc-3)
        massive_output = "x" * (10 * 1024 * 1024)  # 10MB

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "mega_grep",
            "Returns massive output",
            {},
            lambda: massive_output,
        )

        block = ToolUseBlock(id="tu1", name="mega_grep", input={})
        result = await agent._execute_tool(block)

        # Must NOT raise — the run should continue
        assert result is not None
        # Result must be bounded
        assert len(result.content) <= cap + 200
        # Must have a truncation marker
        assert "truncated" in result.content.lower()
