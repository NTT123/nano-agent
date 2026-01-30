"""Cancellation menu element.

Shows tool execution state and lets user choose how to proceed after cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nano_agent.cancellation import (
    CancellationChoice,
    ToolExecutionBatch,
    ToolExecutionStatus,
)

from .base import ActiveElement, InputEvent


@dataclass
class CancellationMenu(ActiveElement[CancellationChoice | None]):
    """Menu for handling cancellation with execution state context.

    Shows:
    - Which tools completed successfully
    - Which tool was cancelled
    - Which tools are still pending

    User can choose:
    - Retry: Re-execute the cancelled tool
    - Skip: Skip cancelled tool, continue with pending
    - Keep: Keep completed results only, stop execution
    - Undo: Rollback everything to before assistant message
    """

    batch: ToolExecutionBatch = field(default_factory=ToolExecutionBatch)
    selected: int = 0

    # Available choices based on state
    _choices: list[CancellationChoice] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize available choices based on batch state."""
        self._choices = [
            CancellationChoice.RETRY,
            CancellationChoice.SKIP,
            CancellationChoice.KEEP_COMPLETED,
            CancellationChoice.UNDO_ALL,
        ]

    def _get_choice_label(self, choice: CancellationChoice) -> str:
        """Get display label for a choice."""
        labels = {
            CancellationChoice.RETRY: "[R] Retry cancelled tool and continue",
            CancellationChoice.SKIP: "[S] Skip cancelled tool, continue with pending",
            CancellationChoice.KEEP_COMPLETED: "[K] Keep completed only, stop here",
            CancellationChoice.UNDO_ALL: "[U] Undo all (rollback to before response)",
        }
        return labels[choice]

    def _get_status_icon(self, status: ToolExecutionStatus) -> str:
        """Get icon for tool status."""
        icons = {
            ToolExecutionStatus.PENDING: "⏳",
            ToolExecutionStatus.RUNNING: "🔄",
            ToolExecutionStatus.COMPLETED: "✅",
            ToolExecutionStatus.FAILED: "❌",
            ToolExecutionStatus.CANCELLED: "🚫",
            ToolExecutionStatus.SKIPPED: "⏭️",
        }
        return icons.get(status, "?")

    def get_lines(self) -> list[str]:
        """Render the cancellation menu."""
        lines = []

        # Header
        lines.append("┌─ Operation Cancelled ─────────────────────────────┐")
        lines.append("│")

        # Show cancelled tool
        cancelled = self.batch.cancelled_tool
        if cancelled:
            lines.append(f"│  Cancelled during: {cancelled.name}")
            lines.append(f"│    Input: {cancelled.display_input(40)}")
        lines.append("│")

        # Show completed tools
        completed = self.batch.completed
        if completed:
            lines.append(f"│  ✅ Completed ({len(completed)}):")
            for tc in completed:
                lines.append(f"│    • {tc.name}({tc.display_input(30)})")
        else:
            lines.append("│  ✅ Completed: (none)")
        lines.append("│")

        # Show pending tools
        pending = self.batch.pending
        if pending:
            lines.append(f"│  ⏳ Pending ({len(pending)}):")
            for tc in pending:
                lines.append(f"│    • {tc.name}({tc.display_input(30)})")
        else:
            lines.append("│  ⏳ Pending: (none)")
        lines.append("│")

        # Separator
        lines.append("├─ What would you like to do? ──────────────────────┤")
        lines.append("│")

        # Choices
        for i, choice in enumerate(self._choices):
            cursor = "→ " if i == self.selected else "  "
            label = self._get_choice_label(choice)
            lines.append(f"│  {cursor}{label}")
        lines.append("│")

        # Footer
        lines.append("└───────────────────────────────────────────────────┘")
        lines.append("")
        lines.append("[↑/↓] move  [Enter] select  [Esc] cancel entirely")

        return lines

    def handle_input(self, event: InputEvent) -> tuple[bool, CancellationChoice | None]:
        """Handle keyboard input."""
        # Enter to select
        if event.key == "Enter":
            if self._choices:
                return (True, self._choices[self.selected])
            return (True, None)

        # Escape to cancel entirely (same as UNDO_ALL)
        if event.key == "Escape" or (event.ctrl and event.char == "c"):
            return (True, CancellationChoice.UNDO_ALL)

        # Shortcut keys
        if event.char:
            char_upper = event.char.upper()
            shortcuts = {
                "R": CancellationChoice.RETRY,
                "S": CancellationChoice.SKIP,
                "K": CancellationChoice.KEEP_COMPLETED,
                "U": CancellationChoice.UNDO_ALL,
            }
            if char_upper in shortcuts:
                return (True, shortcuts[char_upper])

        # Navigation
        if event.char == "j" or event.key == "Down":
            self.selected = min(self.selected + 1, len(self._choices) - 1)
            return (False, None)
        if event.char == "k" or event.key == "Up":
            self.selected = max(self.selected - 1, 0)
            return (False, None)

        return (False, None)
