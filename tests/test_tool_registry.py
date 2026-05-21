"""Tests for ToolRegistry."""
import asyncio
import pytest
from sr2_spectre.tools.registry import ToolRegistry


def _sync_add(a: int, b: int) -> str:
    return str(a + b)


async def _async_greet(name: str) -> str:
    return f"Hello, {name}!"


def test_register_and_list() -> None:
    reg = ToolRegistry()
    reg.register(
        name="add",
        description="Add two numbers",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        fn=_sync_add,
    )
    assert "add" in reg
    assert len(reg) == 1
    assert "add" in reg.list_names()


def test_to_definitions() -> None:
    reg = ToolRegistry()
    reg.register(
        name="search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        fn=lambda q: q,
    )
    defs = reg.to_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "search"
    assert defs[0]["function"]["description"] == "Search the web"


def test_execute_sync_tool() -> None:
    reg = ToolRegistry()
    reg.register(
        name="add",
        description="Add",
        input_schema={},
        fn=_sync_add,
    )
    result = asyncio.get_event_loop().run_until_complete(
        reg.execute("add", {"a": 2, "b": 3})
    )
    assert result == "5"


@pytest.mark.asyncio
async def test_execute_async_tool() -> None:
    reg = ToolRegistry()
    reg.register(
        name="greet",
        description="Greet someone",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        fn=_async_greet,
    )
    result = await reg.execute("greet", {"name": "Diego"})
    assert result == "Hello, Diego!"


@pytest.mark.asyncio
async def test_execute_missing_tool() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        await reg.execute("nonexistent", {})


def test_empty_definitions() -> None:
    reg = ToolRegistry()
    assert reg.to_definitions() == []
