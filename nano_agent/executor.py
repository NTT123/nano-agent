"""Simple executor for running agent loops."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import httpx

from .api_base import APIError, APIProtocol
from .cancellation import CancellationToken
from .dag import DAG
from .data_structures import Response, TextContent

# Type alias for permission callback
# Takes (tool_name, tool_input) and returns True if allowed, False if denied
PermissionCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


async def run(
    api: APIProtocol,
    dag: DAG,
    cancel_token: CancellationToken | None = None,
    permission_callback: PermissionCallback | None = None,
) -> DAG:
    """Run agent loop until stop reason or cancellation.

    Args:
        api: ClaudeAPI client
        dag: Initial DAG with system prompt, tools, and user message
        cancel_token: Optional cancellation token for cooperative cancellation
        permission_callback: Optional async callback for tool permission checks.
            Called with (tool_name, tool_input). Returns True to allow, False to deny.
            Currently used for EditConfirm to require user confirmation.

    Returns:
        Final DAG with all messages and tool results
    """
    from .dag import Node
    from .data_structures import (
        Message,
        Role,
        StopReason,
        ToolExecution,
        ToolResultContent,
    )

    # Get tools from DAG
    tools = dag._tools or ()
    tool_map = {tool.name: tool for tool in tools}

    # Track cumulative usage
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    stop_reason = "end_turn"

    while True:
        # Check for cancellation before API call
        if cancel_token and cancel_token.is_cancelled:
            dag = dag.user("[Operation cancelled by user]")
            stop_reason = "cancelled"
            break

        # Retry on transient errors with exponential backoff
        for attempt in range(5):
            try:
                # Wrap API call if cancel_token provided
                if cancel_token:
                    response = await cancel_token.run(api.send(dag))
                else:
                    response = await api.send(dag)
                break
            except asyncio.CancelledError:
                # Cancellation requested - don't retry
                dag = dag.user("[Operation cancelled by user]")
                stop_reason = "cancelled"
                break
            except APIError as e:
                # Retry on rate limit (429) or server errors (5xx)
                if e.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue
                raise
            except (httpx.TimeoutException, asyncio.TimeoutError):
                if attempt < 4:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

        # If we cancelled during the retry loop, exit
        if stop_reason == "cancelled":
            break

        # Add assistant response to DAG
        dag = dag.assistant(response.content)

        # Accumulate usage
        total_usage["input_tokens"] += response.usage.input_tokens
        total_usage["output_tokens"] += response.usage.output_tokens
        total_usage[
            "cache_creation_input_tokens"
        ] += response.usage.cache_creation_input_tokens
        total_usage["cache_read_input_tokens"] += response.usage.cache_read_input_tokens
        stop_reason = response.stop_reason or "unknown"

        # Check for tool calls first (OpenAI returns "completed" even with tool calls)
        tool_calls = response.get_tool_use()
        if not tool_calls:
            break

        # Save current head before branching
        tool_use_head = dag.head

        # Execute tools and create branch nodes for visualization
        result_nodes = []
        tool_results = []

        cancelled = False
        for call in tool_calls:
            # Check for cancellation before each tool
            if cancel_token and cancel_token.is_cancelled:
                dag = dag.user("[Operation cancelled by user]")
                stop_reason = "cancelled"
                cancelled = True
                break

            tool = tool_map[call.name]

            # Check permission for EditConfirm
            if call.name == "EditConfirm" and permission_callback is not None:
                allowed = await permission_callback(call.name, call.input or {})
                if not allowed:
                    result = TextContent(
                        text="Permission denied: User rejected the edit operation. "
                        "The file was NOT modified."
                    )
                    result_list = [result]
                    result_node = tool_use_head.child(
                        ToolExecution(
                            tool_name=call.name,
                            tool_use_id=call.id,
                            result=result_list,
                            is_error=True,
                        )
                    )
                    result_nodes.append(result_node)
                    tool_results.append(
                        ToolResultContent(
                            tool_use_id=call.id,
                            content=result_list,
                            is_error=True,
                        )
                    )
                    continue  # Skip actual execution

            # Use execute() to convert dict input to typed dataclass
            try:
                if cancel_token:
                    result = await cancel_token.run(tool.execute(call.input))
                else:
                    result = await tool.execute(call.input)
            except asyncio.CancelledError:
                dag = dag.user("[Operation cancelled by user]")
                stop_reason = "cancelled"
                cancelled = True
                break

            # Normalize to list
            result_list = result if isinstance(result, list) else [result]

            # Create branch node with ToolExecution (for visualization)
            result_node = tool_use_head.child(
                ToolExecution(
                    tool_name=call.name,
                    tool_use_id=call.id,
                    result=result_list,
                )
            )
            result_nodes.append(result_node)

            # Collect tool results for API
            tool_results.append(
                ToolResultContent(
                    tool_use_id=call.id,
                    content=result_list,
                )
            )

        # If cancelled during tool execution, exit
        if cancelled:
            break

        # Merge all branches with combined results
        merged = Node.with_parents(
            result_nodes,
            Message(Role.USER, tool_results),
        )
        dag = dag._with_heads((merged,))

    # Add stop reason node
    dag = dag._with_heads(
        dag._append_to_heads(StopReason(reason=stop_reason, usage=total_usage))
    )

    return dag
