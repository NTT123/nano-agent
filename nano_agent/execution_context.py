"""Execution context for sub-agent support.

This module provides the ExecutionContext dataclass that is passed to
sub-agent-capable tools, giving them access to the LLM provider and
conversation context.

Kept dependency-light on purpose: ``ExecutionContext`` references
``APIProtocol`` and ``DAG`` but not ``Tool``. The sub-agent runner that
needs ``Tool`` lives in ``nano_agent.sub_agent`` so this module never
participates in a cycle with ``tools.base``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .cancellation import CancellationToken
from .dag import DAG
from .protocols import APIProtocol

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
        async def __call__(
            self,
            input: MyInput,
            execution_context: ExecutionContext | None = None,
        ) -> TextContent:
            if not execution_context:
                return TextContent(text="Error: No execution context")

            result_dag, sub_graph = await run_sub_agent(
                context=execution_context,
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
