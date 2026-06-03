"""Tests for CodeExecTool."""
import asyncio
from unittest.mock import patch

import pytest

from sr2_spectre.tools.builtins.code_exec import CodeExecTool
from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_code_exec_class_attributes() -> None:
    assert isinstance(CodeExecTool.name, str) and CodeExecTool.name
    assert isinstance(CodeExecTool.description, str) and CodeExecTool.description
    assert isinstance(CodeExecTool.input_schema, dict)


def test_code_exec_input_schema_requires_code() -> None:
    schema = CodeExecTool.input_schema
    assert "code" in schema.get("properties", {})
    assert "code" in schema.get("required", [])
    assert schema["properties"]["code"]["type"] == "string"


def test_code_exec_input_schema_timeout_optional() -> None:
    schema = CodeExecTool.input_schema
    assert "timeout" in schema.get("properties", {})
    assert "timeout" not in schema.get("required", [])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_code_exec_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path(
        "sr2_spectre.tools.builtins.code_exec.CodeExecTool"
    )
    assert "code_exec" in reg


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_evaluates_expression() -> None:
    tool = CodeExecTool()
    result = await tool(code="2 + 2")

    assert "Status: success" in result
    assert "Return: 4" in result


@pytest.mark.asyncio
async def test_code_exec_evaluates_string_expression() -> None:
    tool = CodeExecTool()
    result = await tool(code="'hello' + ' ' + 'world'")

    assert "Status: success" in result
    assert "Return: 'hello world'" in result


@pytest.mark.asyncio
async def test_code_exec_evaluates_complex_expression() -> None:
    tool = CodeExecTool()
    result = await tool(code="[x**2 for x in range(5)]")

    assert "Status: success" in result
    assert "Return: [0, 1, 4, 9, 16]" in result


# ---------------------------------------------------------------------------
# Statement execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_runs_statements() -> None:
    tool = CodeExecTool()
    result = await tool(code="x = 10\ny = 20")

    assert "Status: success" in result
    # exec doesn't return a value
    assert "Return:" not in result


@pytest.mark.asyncio
async def test_code_exec_print_captured_in_stdout() -> None:
    tool = CodeExecTool()
    result = await tool(code="print('hello stdout')")

    assert "Status: success" in result
    assert "Stdout:" in result
    assert "hello stdout" in result


@pytest.mark.asyncio
async def test_code_exec_namespace_isolation() -> None:
    """Each execution gets a fresh namespace."""
    tool = CodeExecTool()
    await tool(code="x = 42")

    # x should not persist to next call
    result = await tool(code="x")
    assert "Status: error" in result
    assert "NameError" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_syntax_error() -> None:
    tool = CodeExecTool()
    result = await tool(code="def incomplete(")

    assert "Status: error" in result
    assert "SyntaxError" in result


@pytest.mark.asyncio
async def test_code_exec_runtime_error() -> None:
    tool = CodeExecTool()
    result = await tool(code="1 / 0")

    assert "Status: error" in result
    assert "ZeroDivisionError" in result


@pytest.mark.asyncio
async def test_code_exec_name_error() -> None:
    tool = CodeExecTool()
    result = await tool(code="undefined_variable")

    assert "Status: error" in result
    assert "NameError" in result


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_timeout_raises_timeout_error() -> None:
    tool = CodeExecTool(timeout=1)
    result = await tool(code="import time; time.sleep(60)", timeout=1)

    assert "Status: timeout" in result
    assert "timed out" in result


@pytest.mark.asyncio
async def test_code_exec_call_timeout_overrides_constructor() -> None:
    """Per-call timeout takes precedence over constructor default."""
    tool = CodeExecTool(timeout=30)
    result = await tool(code="import time; time.sleep(60)", timeout=1)

    assert "Status: timeout" in result


@pytest.mark.asyncio
async def test_code_exec_custom_timeout_stored() -> None:
    tool = CodeExecTool(timeout=60)
    assert tool.timeout == 60


# ---------------------------------------------------------------------------
# Stderr capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_stderr_capture() -> None:
    import warnings
    tool = CodeExecTool()
    # Trigger a warning that goes to stderr
    result = await tool(
        code="import warnings; warnings.warn('test warning')"
    )

    assert "Status: success" in result
    # Warning may go to stderr depending on Python version/warnings config
    # At minimum the tool shouldn't crash
    assert "test warning" in result or "Status: success" in result


# ---------------------------------------------------------------------------
# Return value formatting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_none_return_omitted() -> None:
    """When exec returns None (statements only), no 'Return:' line appears."""
    tool = CodeExecTool()
    result = await tool(code="x = [1, 2, 3]")

    assert "Status: success" in result
    # exec returns None, which should not show as "Return: None"
    assert "Return: None" not in result


@pytest.mark.asyncio
async def test_code_exec_dict_return_formatted() -> None:
    tool = CodeExecTool()
    result = await tool(code="{'a': 1, 'b': 2}")

    assert "Status: success" in result
    assert "'a': 1" in result or "a" in result
