"""Simple executor for running agent loops."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, cast

import httpx

from .cancellation import CancellationToken
from .dag import DAG, Node
from .data_structures import (
    ImageContent,
    Message,
    Role,
    StopReason,
    TextContent,
    ToolExecution,
    ToolResultContent,
    ToolUseContent,
)
from .execution_context import ExecutionContext
from .providers.base import (
    APIError,
    APIProtocol,
    approx_token_count,
    responses_tool_result_item,
)
from .tools.base import Tool, ToolResult

# Type alias for permission callback
# Takes (tool_name, tool_input) and returns True if allowed, False if denied
PermissionCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass(frozen=True)
class _ToolCallOutcome:
    result_node: Node
    tool_result: ToolResultContent


def _error_tool_outcome(
    tool_use_head: Node,
    call: ToolUseContent,
    message: str,
) -> _ToolCallOutcome:
    error_result = TextContent(text=message)
    result_list: list[TextContent | ImageContent] = [error_result]
    result_node = tool_use_head.child(
        ToolExecution(
            tool_name=call.name,
            tool_use_id=call.id,
            result=result_list,
            is_error=True,
        )
    )
    return _ToolCallOutcome(
        result_node=result_node,
        tool_result=ToolResultContent(
            tool_use_id=call.id,
            content=result_list,
            is_error=True,
        ),
    )


def _successful_tool_outcome(
    tool_use_head: Node,
    call: ToolUseContent,
    tool_result: ToolResult,
) -> _ToolCallOutcome:
    result = tool_result.content
    result_list: list[TextContent | ImageContent] = (
        list(result) if isinstance(result, list) else [result]
    )

    parent_node = tool_use_head
    if tool_result.sub_graph is not None:
        parent_node = tool_use_head.child(tool_result.sub_graph)

    result_node = parent_node.child(
        ToolExecution(
            tool_name=call.name,
            tool_use_id=call.id,
            result=result_list,
        )
    )

    return _ToolCallOutcome(
        result_node=result_node,
        tool_result=ToolResultContent(
            tool_use_id=call.id,
            content=result_list,
        ),
    )


async def _execute_tool_call(
    *,
    call: ToolUseContent,
    tool_map: dict[str, Tool],
    tool_use_head: Node,
    turn_context: ExecutionContext,
) -> _ToolCallOutcome:
    tool = tool_map.get(call.name)
    if tool is None:
        return _error_tool_outcome(tool_use_head, call, f"Unknown tool: {call.name}")

    if call.name == "EditConfirm" and turn_context.permission_callback is not None:
        permission_input = cast(dict[str, Any], call.input or {})
        allowed = await turn_context.permission_callback(call.name, permission_input)
        if not allowed:
            return _error_tool_outcome(
                tool_use_head,
                call,
                "Permission denied: User rejected the edit operation. "
                "The file was NOT modified.",
            )

    try:
        raw_input = cast(dict[str, Any] | None, call.input)
        tool_result = await tool.execute(raw_input, execution_context=turn_context)
    except Exception as e:
        return _error_tool_outcome(tool_use_head, call, f"Tool error: {e}")

    return _successful_tool_outcome(tool_use_head, call, tool_result)


async def _gather_tool_outcomes(
    tasks: list[asyncio.Task[_ToolCallOutcome]],
) -> list[_ToolCallOutcome]:
    # Wrap gather in a coroutine so CancellationToken.run can drive it.
    return await asyncio.gather(*tasks)


async def _cancel_and_drain_tool_tasks(
    tasks: list[asyncio.Task[_ToolCallOutcome]],
) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _collect_cancelled_tool_outcomes(
    tool_calls: list[ToolUseContent],
    tasks: list[asyncio.Task[_ToolCallOutcome]],
    tool_use_head: Node,
) -> list[_ToolCallOutcome]:
    outcomes: list[_ToolCallOutcome] = []
    for call, task in zip(tool_calls, tasks, strict=True):
        if task.done() and not task.cancelled() and task.exception() is None:
            outcomes.append(task.result())
        else:
            outcomes.append(
                _error_tool_outcome(tool_use_head, call, "Operation cancelled by user.")
            )
    return outcomes


def _estimate_tool_result_tokens(tool_results: list[ToolResultContent]) -> int:
    # Mirrors codex-rs estimate_response_item_model_visible_bytes: serialize
    # each Responses-API item and apply 4-bytes/token ceiling division. The
    # wire wrapper (function_call_output, call_id, quoting) is what the next
    # API call gets billed for — not just the inner text.
    total = 0
    for result in tool_results:
        wire = responses_tool_result_item(result)
        total += approx_token_count(json.dumps(wire))
    return total


async def maybe_auto_compact(
    api: APIProtocol,
    dag: DAG,
    last_total_tokens: int,
    tool_results: list[ToolResultContent],
) -> DAG:
    """Compact the DAG mid-turn if projected tokens would cross the API's limit.

    Mirrors codex-rs (session/turn.rs:run_turn): compares
    ``last_total_tokens + estimated(tool_results)`` against
    ``api.auto_compact_token_limit``. Skipped when the API client doesn't
    opt in via ``auto_compact_token_limit`` / ``compact_dag``.
    """
    limit = getattr(api, "auto_compact_token_limit", None)
    compact_dag = getattr(api, "compact_dag", None)
    if limit is None or compact_dag is None:
        return dag
    projected = last_total_tokens + _estimate_tool_result_tokens(tool_results)
    if projected < limit:
        return dag
    compacted = await compact_dag(dag)
    return cast(DAG, compacted)


async def run(
    api: APIProtocol,
    dag: DAG,
    cancel_token: CancellationToken | None = None,
    permission_callback: PermissionCallback | None = None,
    execution_context: ExecutionContext | None = None,
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

        # Retry on transient errors with exponential backoff. A mid-stream
        # drop may mean the server fully processed the turn and only the body
        # was lost, so retrying can double-bill tokens — accepted tradeoff
        # vs. killing the worker on a flaky connection.
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
            except (httpx.TransportError, asyncio.TimeoutError):
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

        # All concurrent tasks share the same turn-start context.
        turn_context = dataclasses.replace(
            execution_context,
            api=api,
            dag=dag,
            cancel_token=cancel_token,
            permission_callback=permission_callback,
        )

        # asyncio.gather preserves order, so outcomes align with tool_calls.
        tasks = [
            asyncio.create_task(
                _execute_tool_call(
                    call=call,
                    tool_map=tool_map,
                    tool_use_head=tool_use_head,
                    turn_context=turn_context,
                )
            )
            for call in tool_calls
        ]

        cancelled = False
        try:
            if cancel_token:
                outcomes = await cancel_token.run(_gather_tool_outcomes(tasks))
            else:
                outcomes = await _gather_tool_outcomes(tasks)
        except asyncio.CancelledError:
            await _cancel_and_drain_tool_tasks(tasks)
            outcomes = _collect_cancelled_tool_outcomes(
                tool_calls, tasks, tool_use_head
            )
            cancelled = True

        result_nodes = [outcome.result_node for outcome in outcomes]
        tool_results = [outcome.tool_result for outcome in outcomes]
        merged = Node.with_parents(
            result_nodes,
            Message(Role.USER, tool_results),
        )
        dag = dag._with_heads((merged,))

        if cancelled:
            stop_reason = "cancelled"
            break

        dag = await maybe_auto_compact(
            api,
            dag,
            last_total_tokens=response.usage.total_tokens,
            tool_results=tool_results,
        )

    # Add stop reason node
    dag = dag._with_heads(
        dag._append_to_heads(StopReason(reason=stop_reason, usage=total_usage))
    )

    return dag
