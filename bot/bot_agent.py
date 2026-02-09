"""Agent loop and channel worker for the Discord bot."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from nano_agent import DAG, ExecutionContext, Role, TextContent
from nano_agent.dag import Node
from nano_agent.data_structures import (
    Message,
    StopReason,
    ToolExecution,
    ToolResultContent,
)
from nano_agent.providers.base import APIError

from .bot_state import BotState, serialize_content_blocks, serialize_text_contents

if TYPE_CHECKING:
    import discord
    from nano_agent.providers.base import APIProtocol


@dataclass
class AgentLoopStats:
    stop_reason: str = "end_turn"
    tool_calls: int = 0
    outbound_messages: int = 0
    dequeued_messages: int = 0


async def agent_loop(
    api: APIProtocol,
    state: BotState,
    channel: discord.abc.Messageable,
    channel_id: int,
    dag: DAG,
) -> tuple[DAG, AgentLoopStats]:
    """Run agent turns. Assistant text stays internal unless SendUserMessage is used."""
    tools = dag._tools or ()
    tool_map = {tool.name: tool for tool in tools}
    execution_context = ExecutionContext(api=api, dag=dag)
    stats = AgentLoopStats()

    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    state.active_run_stats = {"outbound_messages": 0, "dequeued_messages": 0}
    queue_end_turn_no_tool_count = 0
    try:
        while True:
            pending_before = len(state.get_channel_queue(channel_id))
            dag = state.sanitize_dag_for_api(dag, channel_id)
            queue_note = state.build_queue_runtime_note(channel_id)
            messages = dag.to_messages()
            last_is_user = bool(messages) and messages[-1].role == Role.USER
            if not last_is_user:
                user_msg = (
                    f"{queue_note}\n\n"
                    f"Runtime event: pending queued user messages={pending_before}. "
                    "Queue is managed outside conversation history. "
                    "Use PeekQueuedUserMessages or DequeueUserMessages to inspect/consume. "
                    "Only SendUserMessage sends text to Discord users."
                )
                dag = dag.user(user_msg)
                state.append_internal_log(
                    channel_id,
                    "runtime_user_event_added",
                    {"pending_queue_count": pending_before},
                )
            else:
                dag = dag.user(queue_note)
            state.append_internal_log(
                channel_id,
                "turn_started",
                {"pending_queue_count": pending_before},
            )

            # Show typing while waiting for API response
            async with channel.typing():
                response = None
                for attempt in range(5):
                    try:
                        response = await api.send(dag)
                        break
                    except APIError as e:
                        if e.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise
                    except (httpx.TimeoutException, asyncio.TimeoutError):
                        if attempt < 4:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise

            assert response is not None

            # Add assistant response to DAG (internal only).
            if response.content:
                dag = dag.assistant(response.content)
            else:
                state.append_internal_log(channel_id, "assistant_response_empty", {})
            state.append_internal_log(
                channel_id,
                "assistant_response",
                {
                    "stop_reason": response.stop_reason or "unknown",
                    "content": serialize_content_blocks(response.content),
                },
            )

            # Accumulate usage
            total_usage["input_tokens"] += response.usage.input_tokens
            total_usage["output_tokens"] += response.usage.output_tokens
            total_usage["cache_creation_input_tokens"] += response.usage.cache_creation_input_tokens
            total_usage["cache_read_input_tokens"] += response.usage.cache_read_input_tokens
            stats.stop_reason = response.stop_reason or "unknown"

            # Check for tool calls
            tool_calls = response.get_tool_use()
            if not tool_calls:
                if pending_before > 0:
                    queue_end_turn_no_tool_count += 1
                    if queue_end_turn_no_tool_count == 1:
                        dag = dag.user(state.format_queue_notification_for_dag(channel_id))
                        state.append_internal_log(
                            channel_id,
                            "queue_notification_injected",
                            {
                                "pending_queue_count": pending_before,
                                "reason": "no_tool_calls_with_pending_queue",
                            },
                        )
                        continue
                break
            queue_end_turn_no_tool_count = 0

            stats.tool_calls += len(tool_calls)
            print(f"[TOOLS] {[c.name for c in tool_calls]}")
            state.append_internal_log(
                channel_id,
                "tool_calls",
                {
                    "calls": [
                        {"id": call.id, "name": call.name, "input": call.input}
                        for call in tool_calls
                    ]
                },
            )

            # Execute tools (show typing while working)
            tool_use_head = dag.head
            result_nodes = []
            tool_results = []

            async with channel.typing():
                for call in tool_calls:
                    tool = tool_map.get(call.name)
                    if tool is None:
                        error_result = TextContent(text=f"Unknown tool: {call.name}")
                        result_nodes.append(
                            tool_use_head.child(
                                ToolExecution(
                                    tool_name=call.name,
                                    tool_use_id=call.id,
                                    result=[error_result],
                                    is_error=True,
                                )
                            )
                        )
                        tool_results.append(
                            ToolResultContent(
                                tool_use_id=call.id,
                                content=[error_result],
                                is_error=True,
                            )
                        )
                        state.append_internal_log(
                            channel_id,
                            "tool_result",
                            {
                                "tool_name": call.name,
                                "tool_use_id": call.id,
                                "is_error": True,
                                "result": [error_result.text],
                            },
                        )
                        continue

                    current_context = ExecutionContext(
                        api=api,
                        dag=dag,
                        depth=execution_context.depth,
                        max_depth=execution_context.max_depth,
                    )

                    try:
                        tool_result = await tool.execute(
                            call.input, execution_context=current_context
                        )
                    except Exception as e:
                        error_result = TextContent(text=f"Tool error: {e}")
                        result_nodes.append(
                            tool_use_head.child(
                                ToolExecution(
                                    tool_name=call.name,
                                    tool_use_id=call.id,
                                    result=[error_result],
                                    is_error=True,
                                )
                            )
                        )
                        tool_results.append(
                            ToolResultContent(
                                tool_use_id=call.id,
                                content=[error_result],
                                is_error=True,
                            )
                        )
                        state.append_internal_log(
                            channel_id,
                            "tool_result",
                            {
                                "tool_name": call.name,
                                "tool_use_id": call.id,
                                "is_error": True,
                                "result": [error_result.text],
                            },
                        )
                        continue

                    result = tool_result.content
                    result_list = result if isinstance(result, list) else [result]

                    sub_graph_node = None
                    if tool_result.sub_graph is not None:
                        sub_graph_node = tool_use_head.child(tool_result.sub_graph)

                    parent_node = sub_graph_node if sub_graph_node else tool_use_head
                    result_nodes.append(
                        parent_node.child(
                            ToolExecution(
                                tool_name=call.name,
                                tool_use_id=call.id,
                                result=result_list,
                            )
                        )
                    )
                    tool_results.append(
                        ToolResultContent(tool_use_id=call.id, content=result_list)
                    )

                    state.append_internal_log(
                        channel_id,
                        "tool_result",
                        {
                            "tool_name": call.name,
                            "tool_use_id": call.id,
                            "is_error": False,
                            "result": serialize_text_contents(result_list),
                        },
                    )

            # Merge tool results back into DAG
            merged = Node.with_parents(
                result_nodes, Message(Role.USER, tool_results)
            )
            dag = dag._with_heads((merged,))
    finally:
        if state.active_run_stats is not None:
            stats.outbound_messages = state.active_run_stats.get("outbound_messages", 0)
            stats.dequeued_messages = state.active_run_stats.get("dequeued_messages", 0)
        state.active_run_stats = None

    # Add stop reason
    dag = dag._with_heads(
        dag._append_to_heads(StopReason(reason=stats.stop_reason, usage=total_usage))
    )
    return dag, stats


def ensure_channel_worker(
    state: BotState,
    api: APIProtocol,
    channel: discord.abc.Messageable,
    system_prompt: str,
) -> None:
    """Ensure exactly one worker is running for this channel."""
    channel_id = channel.id  # type: ignore[attr-defined]
    existing = state.channel_worker_tasks.get(channel_id)
    if existing is not None and not existing.done():
        return

    task = asyncio.create_task(channel_worker(state, api, channel, system_prompt))
    state.channel_worker_tasks[channel_id] = task

    def _cleanup(done_task: asyncio.Task[None]) -> None:
        if state.channel_worker_tasks.get(channel_id) is done_task:
            state.channel_worker_tasks.pop(channel_id, None)

    task.add_done_callback(_cleanup)


async def channel_worker(
    state: BotState,
    api: APIProtocol,
    channel: discord.abc.Messageable,
    system_prompt: str,
) -> None:
    """Process queued messages for a channel until no immediate progress is made."""
    channel_id = channel.id  # type: ignore[attr-defined]
    passes = 0
    max_passes = 8

    while passes < max_passes:
        queue_before = len(state.get_channel_queue(channel_id))
        if queue_before == 0:
            return

        user_id = state.channel_last_user_id.get(channel_id)
        cwd = state.working_dirs.get(user_id) if user_id is not None else None

        # Import here to avoid circular dependency — get_tools needs state
        from .bot_tools import get_tools

        dag = state.get_session(
            channel_id, cwd, system_prompt=system_prompt, tools=get_tools(state)
        )

        async with state.agent_runtime_lock:
            old_cwd = os.getcwd()
            try:
                state.active_channel = channel
                state.active_channel_id = channel_id
                if cwd:
                    os.chdir(cwd)
                dag, stats = await agent_loop(api, state, channel, channel_id, dag)
            except Exception as e:
                print(f"[WORKER_ERROR] channel={channel_id} error={e}")
                state.append_internal_log(channel_id, "worker_error", {"error": str(e)})
                return
            finally:
                os.chdir(old_cwd)
                state.active_channel = None
                state.active_channel_id = None

        if channel_id in state.clear_context_requested:
            state.clear_context_requested.discard(channel_id)
            state.sessions[channel_id] = state.create_session(
                cwd, system_prompt=system_prompt, tools=get_tools(state)
            )
            state.clear_user_queue(channel_id)
            return

        state.set_session(channel_id, dag)
        queue_after = len(state.get_channel_queue(channel_id))
        state.append_internal_log(
            channel_id,
            "worker_pass_complete",
            {
                "pass": passes + 1,
                "queue_before": queue_before,
                "queue_after": queue_after,
                "tool_calls": stats.tool_calls,
                "dequeued_messages": stats.dequeued_messages,
                "outbound_messages": stats.outbound_messages,
                "stop_reason": stats.stop_reason,
            },
        )

        if queue_after >= queue_before and stats.dequeued_messages == 0:
            return

        passes += 1

    state.append_internal_log(
        channel_id,
        "worker_stopped_max_passes",
        {"max_passes": max_passes, "remaining_queue": len(state.get_channel_queue(channel_id))},
    )
