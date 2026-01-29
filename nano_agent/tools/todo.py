"""Todo tool for task list management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, ClassVar

from ..data_structures import TextContent
from .base import Desc, Tool

# =============================================================================
# Todo Data Classes
# =============================================================================


class TodoStatus(str, Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Todo:
    """A single todo item with content, status, and active form."""

    content: str
    status: TodoStatus
    active_form: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Todo":
        """Create a Todo from a dictionary (e.g., from API input)."""
        return cls(
            content=data.get("content", ""),
            status=TodoStatus(data.get("status", "pending")),
            active_form=data.get("activeForm", ""),
        )

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "status": self.status.value,
            "activeForm": self.active_form,
        }


# =============================================================================
# Input Dataclasses
# =============================================================================


@dataclass
class TodoItemInput:
    """A single todo item for API input."""

    content: Annotated[str, Desc("Task description")]
    status: Annotated[str, Desc("pending, in_progress, or completed")]
    activeForm: Annotated[str, Desc("Present continuous form shown in spinner")]


@dataclass
class TodoWriteInput:
    """Input for TodoWriteTool."""

    todos: Annotated[list[TodoItemInput], Desc("The updated todo list")]


# =============================================================================
# Tool Class
# =============================================================================


@dataclass
class TodoWriteTool(Tool):
    """Create and manage a structured task list for your current coding session.

    This tool now includes state management (merged from TodoManager).
    It tracks todos internally and provides methods to query state.

    Example usage:
        tool = TodoWriteTool()
        await tool({"todos": [...]})  # Updates state and displays
        task = tool.get_current_task()  # Get in-progress task
        tool.display()  # Print formatted todo list
        tool.clear()  # Clear all todos
    """

    name: str = "TodoWrite"
    description: str = """Use this tool to create and manage a structured task list for your current coding session.

Task States:
- pending: Task not yet started
- in_progress: Currently working on (limit to ONE task at a time)
- completed: Task finished successfully

Task Management:
- Update task status in real-time as you work
- Mark tasks complete IMMEDIATELY after finishing
- Exactly ONE task must be in_progress at any time"""

    # State management (merged from TodoManager)
    _todos: list[Todo] = field(default_factory=list, repr=False)

    # Class constants for display
    STATUS_INDICATORS: ClassVar[dict[TodoStatus, str]] = {
        TodoStatus.PENDING: "[ ]",
        TodoStatus.IN_PROGRESS: "[~]",
        TodoStatus.COMPLETED: "[x]",
    }

    TOOL_RESULT_MESSAGE: ClassVar[str] = (
        "Todos have been modified successfully. "
        "Ensure that you continue to use the todo list to track your progress. "
        "Please proceed with the current tasks if applicable"
    )

    # ==========================================================================
    # State Management Methods (from TodoManager)
    # ==========================================================================

    @property
    def todos(self) -> list[Todo]:
        """Get all todos (returns a copy for safety)."""
        return self._todos.copy()

    def get_current_task(self) -> Todo | None:
        """Get the currently in-progress task, if any."""
        for todo in self._todos:
            if todo.status == TodoStatus.IN_PROGRESS:
                return todo
        return None

    def clear(self) -> None:
        """Clear all todos."""
        self._todos = []

    def display(self) -> None:
        """Print the todo list to console with status indicators."""
        if not self._todos:
            print("No todos.")
            return

        print("\n--- Todo List ---")
        for i, todo in enumerate(self._todos, 1):
            indicator = self.STATUS_INDICATORS[todo.status]
            print(f"{i}. {indicator} {todo.content}")

            # Show active form for in-progress items
            if todo.status == TodoStatus.IN_PROGRESS and todo.active_form:
                print(f"      {todo.active_form}...")

        # Summary
        total = len(self._todos)
        completed = sum(1 for t in self._todos if t.status == TodoStatus.COMPLETED)
        in_progress = sum(1 for t in self._todos if t.status == TodoStatus.IN_PROGRESS)
        pending = sum(1 for t in self._todos if t.status == TodoStatus.PENDING)

        print(f"\nProgress: {completed}/{total} completed", end="")
        if in_progress:
            print(f", {in_progress} in progress", end="")
        if pending:
            print(f", {pending} pending", end="")
        print("\n")

    # ==========================================================================
    # Tool Execution
    # ==========================================================================

    async def __call__(self, input: TodoWriteInput) -> TextContent:
        """Update the todo list, display it, and return a summary for the model."""
        # Convert TodoItemInput instances to Todo objects
        self._todos = [
            Todo(
                content=item.content,
                status=TodoStatus(item.status),
                active_form=item.activeForm,
            )
            for item in input.todos
        ]

        # Display to console
        self.display()

        # Return message for model
        return TextContent(text=self.TOOL_RESULT_MESSAGE)
