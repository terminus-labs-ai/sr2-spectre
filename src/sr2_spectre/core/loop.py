"""Tool execution loop — the core agent reasoning cycle.

Spectre owns this loop:
1. Build CompletionRequest with full history + tool definitions
2. Call RelayLLMCallable.complete()
3. If stop_reason == "tool_use", execute tools and loop
4. Otherwise, return the final response
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sr2_spectre.core.client import RelayClient

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a complete agent turn."""
    text: str
    tool_calls_executed: int = 0
    total_tokens: int = 0


def _build_system_prompt(system: str) -> list[dict[str, str]] | None:
    if not system:
        return None
    return [{"type": "text", "text": system}]


def run_tool_loop_sync_build(
    system_prompt: str,
    history: list[dict[str, Any]],
    tools_definitions: list[dict[str, Any]],
) -> Any:
    """Build a CompletionRequest from history.

    History uses spectre's internal dict format. This converts it to
    proper SR2 Message objects with ContentBlocks.
    """
    from sr2 import Message, TextBlock, ToolUseBlock, ToolResultBlock
    from sr2.protocols.llm import CompletionRequest

    messages: list[Message] = []
    i = 0
    while i < len(history):
        entry = history[i]
        role = entry["role"]

        if role == "user":
            messages.append(Message(
                role="user",
                content=[TextBlock(text=entry["content"])],
            ))

        elif role == "assistant":
            content_items = entry.get("content", [])
            blocks: list[Any] = []
            for item in content_items:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    blocks.append(ToolUseBlock(
                        id=item["id"],
                        name=item["name"],
                        input=item.get("input", {}),
                    ))
                elif isinstance(item, dict) and item.get("type") == "text":
                    blocks.append(TextBlock(text=item.get("text", "")))
            # If no structured items, treat as plain text
            if not blocks and content_items:
                # Might be raw string content from old format
                messages.append(Message(
                    role="assistant",
                    content=[TextBlock(text=str(content_items))],
                ))
            else:
                messages.append(Message(
                    role="assistant",
                    content=blocks if blocks else [TextBlock(text="")],
                ))

        elif role == "tool":
            # Tool results go as ToolResultBlock in assistant message
            messages.append(Message(
                role="assistant",
                content=[ToolResultBlock(
                    tool_use_id=entry["tool_use_id"],
                    content=entry["content"],
                    is_error=entry.get("is_error", False),
                )],
            ))

        i += 1

    return CompletionRequest(
        system=_build_system_prompt(system_prompt),
        messages=messages,
        tools=tools_definitions if tools_definitions else None,
    )


async def run_tool_loop(
    client: "RelayClient",
    system_prompt: str,
    history: list[dict[str, Any]],
    tools_definitions: list[dict[str, Any]],
    tool_executor: Any,
    max_tool_iterations: int = 20,
) -> TurnResult:
    """Run the agent tool loop.

    1. Send request to relay with full history + tools
    2. If LLM returns tool_use, execute tools, append results, repeat
    3. Return final text response with metadata
    """
    from sr2.models import TextBlock, ToolUseBlock

    iteration = 0
    total_tool_calls = 0

    while iteration < max_tool_iterations:
        iteration += 1

        request = run_tool_loop_sync_build(
            system_prompt, history, tools_definitions
        )

        response = await client.complete(request)

        if response.stop_reason == "tool_use":
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use" or isinstance(block, ToolUseBlock):
                    total_tool_calls += 1
                    tool_name = getattr(block, "name", "")
                    tool_id = getattr(block, "id", "")
                    tool_input = getattr(block, "input", {})

                    logger.debug(
                        f"Executing tool: {tool_name} (iteration {iteration})"
                    )

                    try:
                        exec_result = await tool_executor.execute(
                            tool_name, tool_input
                        )
                    except Exception as e:
                        logger.error(f"Tool {tool_name} failed: {e}")
                        exec_result = f"Error: {e}"
                        is_error = True
                    else:
                        is_error = False

                    result_str = (
                        exec_result
                        if isinstance(exec_result, str)
                        else str(exec_result)
                    )
                    history.append({
                        "role": "tool",
                        "tool_use_id": tool_id,
                        "content": result_str,
                        "is_error": is_error,
                    })

            tool_blocks = [
                b for b in response.content
                if getattr(b, "type", None) == "tool_use"
                or isinstance(b, ToolUseBlock)
            ]
            history.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": getattr(b, "id", ""),
                        "name": getattr(b, "name", ""),
                        "input": getattr(b, "input", {}),
                    }
                    for b in tool_blocks
                ],
            })
            continue

        # Extract text
        text_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text" or isinstance(block, TextBlock):
                text_parts.append(getattr(block, "text", ""))

        final_text = "".join(text_parts)
        history.append({
            "role": "assistant",
            "content": (
                [{"type": "text", "text": t} for t in text_parts]
                if text_parts
                else [{"type": "text", "text": ""}]
            ),
        })

        return TurnResult(
            text=final_text,
            tool_calls_executed=total_tool_calls,
            total_tokens=(
                (getattr(response.usage, "input_tokens", 0) or 0)
                + (getattr(response.usage, "output_tokens", 0) or 0)
            ),
        )

    raise RuntimeError(
        f"Tool loop exceeded {max_tool_iterations} iterations"
    )
