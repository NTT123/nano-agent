"""Cancellation token for cooperative async operation cancellation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


# ============================================
# Tool Execution Tracking
# ============================================


class ToolExecutionStatus(Enum):
    """Status of a tool call during batch execution."""

    PENDING = "pending"  # Not started yet
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with error
    CANCELLED = "cancelled"  # User cancelled mid-execution
    SKIPPED = "skipped"  # User chose to skip after cancel


class CancellationChoice(Enum):
    """User's choice after cancellation."""

    RETRY = "retry"  # Retry the cancelled tool, continue with pending
    SKIP = "skip"  # Skip cancelled tool, continue with pending
    KEEP_COMPLETED = "keep"  # Keep completed only, stop execution
    UNDO_ALL = "undo"  # Rollback to before assistant message


@dataclass
class TrackedToolCall:
    """Track state of a single tool call during execution."""

    id: str
    name: str
    input: dict[str, Any]
    status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    result: Any = None
    error: str | None = None

    def display_input(self, max_len: int = 50) -> str:
        """Truncated input for display."""
        s = str(self.input)
        return s[: max_len - 3] + "..." if len(s) > max_len else s


@dataclass
class ToolExecutionBatch:
    """Track a batch of tool calls from a single assistant response.

    Used to provide context to users when cancellation occurs, allowing
    them to make informed decisions about how to proceed.
    """

    tool_calls: list[TrackedToolCall] = field(default_factory=list)
    cancelled_at_index: int | None = None

    def mark_running(self, index: int) -> None:
        """Mark a tool as currently running."""
        self.tool_calls[index].status = ToolExecutionStatus.RUNNING

    def mark_completed(self, index: int, result: Any) -> None:
        """Mark a tool as completed with result."""
        self.tool_calls[index].status = ToolExecutionStatus.COMPLETED
        self.tool_calls[index].result = result

    def mark_failed(self, index: int, error: str) -> None:
        """Mark a tool as failed with error."""
        self.tool_calls[index].status = ToolExecutionStatus.FAILED
        self.tool_calls[index].error = error

    def mark_cancelled(self, index: int) -> None:
        """Mark a tool as cancelled by user."""
        self.tool_calls[index].status = ToolExecutionStatus.CANCELLED
        self.cancelled_at_index = index

    def mark_skipped(self, index: int) -> None:
        """Mark a tool as skipped (user chose not to execute)."""
        self.tool_calls[index].status = ToolExecutionStatus.SKIPPED

    @property
    def completed(self) -> list[TrackedToolCall]:
        """Get all completed tool calls."""
        return [t for t in self.tool_calls if t.status == ToolExecutionStatus.COMPLETED]

    @property
    def pending(self) -> list[TrackedToolCall]:
        """Get all pending tool calls."""
        return [t for t in self.tool_calls if t.status == ToolExecutionStatus.PENDING]

    @property
    def cancelled_tool(self) -> TrackedToolCall | None:
        """Get the tool that was cancelled, if any."""
        if self.cancelled_at_index is not None:
            return self.tool_calls[self.cancelled_at_index]
        return None

    def summary_lines(self) -> list[str]:
        """Generate summary lines for display to user."""
        lines = []
        status_icons = {
            ToolExecutionStatus.PENDING: "⏳",
            ToolExecutionStatus.RUNNING: "🔄",
            ToolExecutionStatus.COMPLETED: "✅",
            ToolExecutionStatus.FAILED: "❌",
            ToolExecutionStatus.CANCELLED: "🚫",
            ToolExecutionStatus.SKIPPED: "⏭️",
        }
        for i, tc in enumerate(self.tool_calls, 1):
            icon = status_icons[tc.status]
            lines.append(
                f"  {i}. {icon} {tc.name}({tc.display_input(30)}) - {tc.status.value}"
            )
        return lines


# ============================================
# Cancellation Token
# ============================================


@dataclass
class CancellationToken:
    """Token for cooperative cancellation of async operations.

    This class provides a way to cancel async operations cooperatively.
    The token tracks a cancellation state and can wrap coroutines in
    cancellable tasks.

    Usage:
        token = CancellationToken()

        # In agent loop:
        try:
            result = await token.run(api.send(dag))
        except asyncio.CancelledError:
            # Handle cancellation
            pass

        # To cancel (from another coroutine/callback):
        token.cancel()

        # To reuse for new operation:
        token.reset()

    Integration with executor.run():
        from nano_agent import run, CancellationToken

        token = CancellationToken()
        # In another task: token.cancel()
        dag = await run(api, dag, cancel_token=token)
    """

    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _current_task: asyncio.Task[Any] | None = field(default=None, init=False)

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation - sets flag and cancels current task."""
        self._event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def reset(self) -> None:
        """Reset for reuse with new operation."""
        self._event.clear()
        self._current_task = None

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine with cancellation support.

        Wraps coroutine in asyncio.Task so it can be cancelled mid-flight.
        Raises asyncio.CancelledError if cancel() was called.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            asyncio.CancelledError: If cancel() was called during execution
        """
        if self.is_cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

        task: asyncio.Task[T] = asyncio.create_task(coro)
        self._current_task = task
        try:
            return await task
        finally:
            self._current_task = None
