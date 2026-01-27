"""Input handlers for the TUI message system.

This module provides concrete input handler implementations for different
message types in the TUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .messages import InputEvent, InputHandler


@dataclass
class TextInputHandler:
    """Input handler for text input (user query input).

    Handles character input, backspace, cursor movement, etc.
    """

    buffer: str = ""
    cursor_pos: int = 0
    on_submit: Callable[[str], None] | None = None

    async def handle_key(self, event: InputEvent) -> bool:
        """Handle key event for text input.

        Args:
            event: The input event

        Returns:
            True if the event was consumed
        """
        if event.key == "enter":
            if self.on_submit:
                self.on_submit(self.buffer)
            return True
        elif event.key == "backspace":
            if self.cursor_pos > 0:
                self.buffer = (
                    self.buffer[: self.cursor_pos - 1] + self.buffer[self.cursor_pos :]
                )
                self.cursor_pos -= 1
            return True
        elif event.key == "delete":
            if self.cursor_pos < len(self.buffer):
                self.buffer = (
                    self.buffer[: self.cursor_pos] + self.buffer[self.cursor_pos + 1 :]
                )
            return True
        elif event.key == "left":
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return True
        elif event.key == "right":
            if self.cursor_pos < len(self.buffer):
                self.cursor_pos += 1
            return True
        elif event.key == "home":
            self.cursor_pos = 0
            return True
        elif event.key == "end":
            self.cursor_pos = len(self.buffer)
            return True
        elif len(event.key) == 1 and not event.ctrl and not event.alt:
            # Regular character input
            self.buffer = (
                self.buffer[: self.cursor_pos]
                + event.key
                + self.buffer[self.cursor_pos :]
            )
            self.cursor_pos += 1
            return True
        return False

    def get_prompt_text(self) -> str:
        """Return current input buffer for display."""
        return self.buffer

    def clear(self) -> None:
        """Clear the input buffer."""
        self.buffer = ""
        self.cursor_pos = 0


@dataclass
class ConfirmationHandler:
    """Input handler for yes/no confirmation prompts.

    Used for EditConfirm and other permission prompts.
    """

    result: bool | None = None
    cancelled: bool = False
    on_result: Callable[[bool | None], None] | None = None

    async def handle_key(self, event: InputEvent) -> bool:
        """Handle key event for confirmation.

        Args:
            event: The input event

        Returns:
            True if the event was consumed
        """
        key = event.key.lower()
        if key == "y":
            self.result = True
            if self.on_result:
                self.on_result(True)
            return True
        elif key == "n":
            self.result = False
            if self.on_result:
                self.on_result(False)
            return True
        elif key == "escape":
            self.cancelled = True
            self.result = None
            if self.on_result:
                self.on_result(None)
            return True
        return False

    def get_prompt_text(self) -> str:
        """Return prompt text for display."""
        return "[y/n/Esc]"

    def reset(self) -> None:
        """Reset the handler state."""
        self.result = None
        self.cancelled = False


@dataclass
class SelectionHandler:
    """Input handler for selection from multiple options.

    Used for AskUserQuestion with multiple options.
    """

    options: list[str] = field(default_factory=list)
    selected_index: int = 0
    confirmed: bool = False
    on_select: Callable[[int], None] | None = None

    async def handle_key(self, event: InputEvent) -> bool:
        """Handle key event for selection.

        Args:
            event: The input event

        Returns:
            True if the event was consumed
        """
        if event.key == "up":
            if self.selected_index > 0:
                self.selected_index -= 1
            return True
        elif event.key == "down":
            if self.selected_index < len(self.options) - 1:
                self.selected_index += 1
            return True
        elif event.key == "enter":
            self.confirmed = True
            if self.on_select:
                self.on_select(self.selected_index)
            return True
        elif event.key == "escape":
            self.selected_index = -1
            self.confirmed = True
            if self.on_select:
                self.on_select(-1)
            return True
        elif event.key.isdigit():
            # Direct selection by number (1-indexed)
            num = int(event.key)
            if 1 <= num <= len(self.options):
                self.selected_index = num - 1
                self.confirmed = True
                if self.on_select:
                    self.on_select(self.selected_index)
                return True
        return False

    def get_prompt_text(self) -> str:
        """Return prompt text for display."""
        if not self.options:
            return ""
        lines = []
        for i, option in enumerate(self.options):
            prefix = ">" if i == self.selected_index else " "
            lines.append(f"{prefix} {i + 1}. {option}")
        return "\n".join(lines)

    def get_selected(self) -> str | None:
        """Return the selected option text, or None if cancelled."""
        if self.selected_index < 0 or self.selected_index >= len(self.options):
            return None
        return self.options[self.selected_index]

    def reset(self) -> None:
        """Reset the handler state."""
        self.selected_index = 0
        self.confirmed = False
