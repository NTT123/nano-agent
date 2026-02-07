"""Execution context for sub-agent support.

This module provides the ExecutionContext dataclass that is passed to
sub-agent-capable tools, giving them access to the LLM provider and
conversation context.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .cancellation import CancellationToken
from .data_structures import SubGraph

if TYPE_CHECKING:
    from .api_base import APIProtocol
    from .dag import DAG
    from .tools import Tool

# Type alias for permission callback
# Takes (tool_name, tool_input) and returns True if allowed, False if denied
PermissionCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass(frozen=True)
class ExecutionContext:
    """Context passed to sub-agent-capable tools.

    This provides tools with everything they need to spawn sub-agents:
    - api: The LLM API client for making calls
    - dag: The current conversation DAG (for context access)
    - cancel_token: For cooperative cancellation
    - permission_callback: For tool permission checks
    - depth: Current nesting depth (0 for top-level)
    - max_depth: Maximum allowed nesting depth

    Example usage in a sub-agent tool:
        async def __call__(self, input: MyInput) -> TextContent:
            if not self._execution_context:
                return TextContent(text="Error: No execution context")

            result_dag, sub_graph = await run_sub_agent(
                context=self._execution_context,
                system_prompt="You are a helpful assistant...",
                user_message="Do something...",
                tools=[ReadTool(), GrepTool()],
            )
            return TextContent(text=sub_graph.summary)
    """

    api: APIProtocol
    dag: DAG
    cancel_token: CancellationToken | None = None
    permission_callback: PermissionCallback | None = None
    depth: int = 0
    max_depth: int = 5

    def child_context(self, child_dag: DAG) -> ExecutionContext:
        """Create child context with incremented depth.

        Args:
            child_dag: The sub-agent's DAG

        Returns:
            New ExecutionContext with depth incremented

        Raises:
            RecursionError: If depth limit exceeded
        """
        if self.depth > self.max_depth:
            raise RecursionError(
                f"Sub-agent depth limit exceeded (max={self.max_depth})"
            )
        return ExecutionContext(
            api=self.api,
            dag=child_dag,
            cancel_token=self.cancel_token,
            permission_callback=self.permission_callback,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )


async def run_sub_agent(
    context: ExecutionContext,
    system_prompt: str,
    user_message: str,
    tools: Sequence[Tool] | None = None,
    tool_name: str = "SubAgent",
    tool_use_id: str | None = None,
) -> tuple[DAG, SubGraph]:
    """Helper to spawn a sub-agent from within a tool.

    This function handles the full lifecycle of a sub-agent:
    1. Creates a new DAG with the provided system prompt
    2. Adds tools if specified
    3. Adds the user message
    4. Runs the agent loop until completion
    5. Creates a SubGraph to encapsulate the results

    Args:
        context: The parent execution context
        system_prompt: System prompt for the sub-agent
        user_message: Initial user message/task for the sub-agent
        tools: Optional list of tools for the sub-agent (not inherited)
        tool_name: Name of the tool spawning this sub-agent
        tool_use_id: ID of the tool call (auto-generated if None)

    Returns:
        Tuple of (completed DAG, SubGraph for storage)

    Raises:
        RecursionError: If sub-agent depth limit exceeded

    Example:
        result_dag, sub_graph = await run_sub_agent(
            context=self._execution_context,
            system_prompt="You are a code reviewer...",
            user_message=f"Review the code in {file_path}",
            tools=[ReadTool(), GrepTool()],
            tool_name="CodeReviewer",
        )
    """
    # Import here to avoid circular dependency
    from .dag import DAG
    from .executor import run

    # Generate tool_use_id if not provided
    if tool_use_id is None:
        tool_use_id = f"subagent_{uuid.uuid4().hex[:12]}"

    # Create sub-agent DAG
    sub_dag = DAG(system_prompt=system_prompt)
    if tools:
        sub_dag = sub_dag.tools(*tools)
    sub_dag = sub_dag.user(user_message)

    # Create child context and run
    child_context = context.child_context(sub_dag)
    result_dag = await run(
        api=context.api,
        dag=sub_dag,
        cancel_token=context.cancel_token,
        permission_callback=context.permission_callback,
        execution_context=child_context,
    )

    # Extract summary from final response
    summary = _extract_summary(result_dag)

    # Create SubGraph for storage
    sub_graph = result_dag.to_sub_graph(
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        summary=summary,
        depth=context.depth + 1,
    )

    return result_dag, sub_graph


def _extract_summary(dag: DAG) -> str:
    """Extract a summary from the sub-agent's final response.

    Looks for the last assistant message with text content.
    """
    from .data_structures import Message, Role, TextContent

    if not dag._heads:
        return ""

    # Walk backwards through nodes to find last assistant message
    for node in reversed(dag.head.ancestors()):
        if isinstance(node.data, Message) and node.data.role == Role.ASSISTANT:
            content = node.data.content
            if isinstance(content, str):
                # Truncate if too long
                return content[:500] + "..." if len(content) > 500 else content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, TextContent):
                        text = block.text
                        return text[:500] + "..." if len(text) > 500 else text
    return ""
