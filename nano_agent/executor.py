"""Simple executor for running agent loops."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx

from .api_base import APIError, APIProtocol
from .cancellation import CancellationToken
from .dag import DAG
from .data_structures import Response, TextContent

if TYPE_CHECKING:
    from .execution_context import ExecutionContext

# Type alias for permission callback
# Takes (tool_name, tool_input) and returns True if allowed, False if denied
PermissionCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


async def run(
    api: APIProtocol,
    dag: DAG,
    cancel_token: CancellationToken | None = None,
    permission_callback: PermissionCallback | None = None,
    execution_context: "ExecutionContext | None" = None,
) -> DAG:
    """Run agent loop until stop reason or cancellation.

    Args:
        api: ClaudeAPI client
        dag: Initial DAG with system prompt, tools, and user message
        cancel_token: Optional cancellation token for cooperative cancellation
        permission_callback: Optional async callback for tool permission checks.
            Called with (tool_name, tool_input). Returns True to allow, False to deny.
            Currently used for EditConfirm to require user confirmation.
        execution_context: Optional execution context for sub-agent support.
            If not provided, one will be created automatically.

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
    from .execution_context import ExecutionContext

    # Get tools from DAG
    tools = dag._tools or ()
    tool_map = {tool.name: tool for tool in tools}

    # Create execution context if not provided
    if execution_context is None:
        execution_context = ExecutionContext(
            api=api,
            dag=dag,
            cancel_token=cancel_token,
            permission_callback=permission_callback,
        )

    # Track cumulative usage
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    stop_reason = "end_turn"

    while True:
        # Check for cancellation before API call
        if cancel_token and cancel_token.is_cancelled:
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
        total_usage["reasoning_tokens"] += response.usage.reasoning_tokens
        total_usage["cached_tokens"] += response.usage.cached_tokens
        total_usage["total_tokens"] += response.usage.total_tokens
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
        cancelled_at_index: int | None = None
        for i, call in enumerate(tool_calls):
            # Check for cancellation before each tool
            if cancel_token and cancel_token.is_cancelled:
                cancelled_at_index = i  # Tool hasn't started yet
                cancelled = True
                break

            tool = tool_map.get(call.name)
            if tool is None:
                error_result = TextContent(text=f"Unknown tool: {call.name}")
                result_node = tool_use_head.child(
                    ToolExecution(
                        tool_name=call.name,
                        tool_use_id=call.id,
                        result=[error_result],
                        is_error=True,
                    )
                )
                result_nodes.append(result_node)
                tool_results.append(
                    ToolResultContent(
                        tool_use_id=call.id,
                        content=[error_result],
                        is_error=True,
                    )
                )
                continue

            # Check permission for EditConfirm
            if call.name == "EditConfirm" and permission_callback is not None:
                allowed = await permission_callback(call.name, call.input or {})
                if not allowed:
                    denied_result = TextContent(
                        text="Permission denied: User rejected the edit operation. "
                        "The file was NOT modified."
                    )
                    denied_result_list = [denied_result]
                    result_node = tool_use_head.child(
                        ToolExecution(
                            tool_name=call.name,
                            tool_use_id=call.id,
                            result=denied_result_list,
                            is_error=True,
                        )
                    )
                    result_nodes.append(result_node)
                    tool_results.append(
                        ToolResultContent(
                            tool_use_id=call.id,
                            content=denied_result_list,
                            is_error=True,
                        )
                    )
                    continue  # Skip actual execution

            # Create current execution context for this call
            # (DAG is updated as we process, so context needs fresh reference)
            current_context = ExecutionContext(
                api=api,
                dag=dag,
                cancel_token=cancel_token,
                permission_callback=permission_callback,
                depth=execution_context.depth,
                max_depth=execution_context.max_depth,
            )

            # Use execute() to convert dict input to typed dataclass
            try:
                if cancel_token:
                    tool_result = await cancel_token.run(
                        tool.execute(call.input, execution_context=current_context)
                    )
                else:
                    tool_result = await tool.execute(
                        call.input, execution_context=current_context
                    )
            except asyncio.CancelledError:
                cancelled_at_index = i  # Tool was running when cancelled
                cancelled = True
                break
            except Exception as e:
                error_result = TextContent(text=f"Tool error: {e}")
                result_node = tool_use_head.child(
                    ToolExecution(
                        tool_name=call.name,
                        tool_use_id=call.id,
                        result=[error_result],
                        is_error=True,
                    )
                )
                result_nodes.append(result_node)
                tool_results.append(
                    ToolResultContent(
                        tool_use_id=call.id,
                        content=[error_result],
                        is_error=True,
                    )
                )
                continue

            # Normalize content to list
            result = tool_result.content
            result_list = result if isinstance(result, list) else [result]

            # Check if tool returned a SubGraph (sub-agent execution)
            # SubGraph runs first, then produces the ToolExecution result
            sub_graph_node = None
            if tool_result.sub_graph is not None:
                # SubGraph is first (sub-agent runs)
                sub_graph_node = tool_use_head.child(tool_result.sub_graph)

            # Create ToolExecution node (the result)
            # If sub-agent ran, ToolExecution is child of SubGraph
            # Otherwise, ToolExecution is child of tool_use_head
            parent_node = sub_graph_node if sub_graph_node else tool_use_head
            result_node = parent_node.child(
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

        # If cancelled during tool execution, add results for cancelled/pending tools
        if cancelled:
            assert cancelled_at_index is not None
            for j in range(cancelled_at_index, len(tool_calls)):
                pending_call = tool_calls[j]
                if j == cancelled_at_index:
                    msg = "Operation cancelled by user."
                else:
                    msg = "Tool skipped due to cancellation."

                cancelled_result = TextContent(text=msg)
                result_node = tool_use_head.child(
                    ToolExecution(
                        tool_name=pending_call.name,
                        tool_use_id=pending_call.id,
                        result=[cancelled_result],
                        is_error=True,
                    )
                )
                result_nodes.append(result_node)
                tool_results.append(
                    ToolResultContent(
                        tool_use_id=pending_call.id,
                        content=[cancelled_result],
                        is_error=True,
                    )
                )

            # Merge all results (completed + cancelled + skipped)
            if result_nodes:
                merged = Node.with_parents(
                    result_nodes,
                    Message(Role.USER, tool_results),
                )
                dag = dag._with_heads((merged,))

            stop_reason = "cancelled"
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
