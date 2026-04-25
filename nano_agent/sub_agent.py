"""Sub-agent runner (``run_sub_agent``) and the ``SubAgentTool`` base class."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .dag import DAG
from .data_structures import Message, Role, SubGraph, TextContent
from .execution_context import ExecutionContext
from .executor import run
from .tools.base import Tool, ToolResult


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
    if tool_use_id is None:
        tool_use_id = f"subagent_{uuid.uuid4().hex[:12]}"

    sub_dag = DAG().system(system_prompt)
    if tools:
        sub_dag = sub_dag.tools(*tools)
    sub_dag = sub_dag.user(user_message)

    child_context = context.child_context(sub_dag)
    result_dag = await run(
        api=context.api,
        dag=sub_dag,
        cancel_token=context.cancel_token,
        permission_callback=context.permission_callback,
        execution_context=child_context,
    )

    summary = _extract_summary(result_dag)
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
    if not dag._heads:
        return ""

    for msg in reversed(dag.to_messages()):
        if msg.role != Role.ASSISTANT:
            continue
        content = msg.content
        if isinstance(content, str):
            return content[:500] + "..." if len(content) > 500 else content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, TextContent):
                    text = block.text
                    return text[:500] + "..." if len(text) > 500 else text
    return ""


@dataclass
class SubAgentTool(Tool):
    """Base class for tools that spawn sub-agents (pure functional).

    This class provides a `spawn()` helper that handles the boilerplate of
    running sub-agents. Unlike the old mutation-based design, this is pure
    functional - the execution context is passed as a parameter to __call__,
    and spawn() returns both the summary and the SubGraph.

    Example:
        @dataclass
        class SecurityAuditInput:
            file_path: Annotated[str, Desc("Path to the file to audit")]

        @dataclass
        class SecurityAuditTool(SubAgentTool):
            name: str = "SecurityAudit"
            description: str = "Spawn a sub-agent to audit code for security issues"

            async def __call__(
                self,
                input: SecurityAuditInput,
                execution_context: ExecutionContext | None = None,
            ) -> ToolResult:
                if not execution_context:
                    return ToolResult(content=TextContent(text="Error: No context"))

                summary, sub_graph = await self.spawn(
                    context=execution_context,
                    system_prompt="You are an expert security auditor...",
                    user_message=f"Audit the file: {input.file_path}",
                    tools=[ReadTool()],
                )
                return ToolResult(
                    content=TextContent(text=summary),
                    sub_graph=sub_graph,
                )

    The base class provides:
    - spawn() helper for running sub-agents
    - Pure functional design (no mutation)
    """

    async def __call__(
        self,
        input: Any,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        """Execute the sub-agent tool. Override in subclasses.

        Args:
            input: The typed input dataclass
            execution_context: Execution context for spawning sub-agents

        Returns:
            ToolResult containing content and optionally sub_graph
        """
        raise NotImplementedError(f"{self.name} does not implement __call__()")

    async def spawn(
        self,
        context: ExecutionContext,
        system_prompt: str,
        user_message: str,
        tools: Sequence[Tool] | None = None,
        tool_name: str | None = None,
    ) -> tuple[str, SubGraph]:
        """Spawn a sub-agent and return its summary and graph.

        This is the main helper method that simplifies sub-agent creation.

        Args:
            context: Execution context (required, provides API access)
            system_prompt: System prompt for the sub-agent
            user_message: Initial user message/task for the sub-agent
            tools: Optional list of tools for the sub-agent
            tool_name: Name for the sub-agent (defaults to self.name)

        Returns:
            Tuple of (summary text, SubGraph)
        """
        _, sub_graph = await run_sub_agent(
            context=context,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            tool_name=tool_name or self.name,
        )

        return sub_graph.summary, sub_graph
