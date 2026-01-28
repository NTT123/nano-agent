"""Unified terminal footer with content area and status bar.

This module provides:
- StatusBarState: Dataclass holding status bar values
- TerminalFooter: Unified footer with content + status bar rendering
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .terminal import TerminalRegion


@dataclass
class StatusBarState:
    """State for the status bar display."""

    auto_accept: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    activity: str | None = None  # "Thinking...", "Running Bash...", etc.

    # Spinner animation
    _spinner_frame: int = field(default=0, init=False, repr=False)
    _spinner_chars: str = field(default="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", init=False, repr=False)

    def get_spinner_char(self) -> str:
        """Get current spinner character and advance frame."""
        char = self._spinner_chars[self._spinner_frame % len(self._spinner_chars)]
        self._spinner_frame += 1
        return char


class TerminalFooter:
    """Unified footer with content area + status bar.

    Layout:
        ┌─────────────────────────────────┐
        │ > user input█                   │  ← Content (optional, multiple lines)
        │ Auto-accept: off │ Tokens: 1,234│  ← Status bar (always visible)
        └─────────────────────────────────┘

    The footer manages a terminal region at the bottom of the screen.
    Content lines are optional (shown during input mode).
    Status bar is always visible.
    """

    def __init__(self) -> None:
        self._region = TerminalRegion()
        self._state: str = "inactive"  # inactive | active | paused
        self._content_lines: list[str] = []
        self._status = StatusBarState()

    @property
    def status(self) -> StatusBarState:
        """Get the current status bar state."""
        return self._status

    def _format_status_line(self) -> str:
        """Format the status bar line with ANSI colors."""
        parts = []

        # Activity with spinner (if active)
        if self._status.activity:
            spinner = self._status.get_spinner_char()
            parts.append(f"{spinner} \033[36m{self._status.activity}\033[0m")

        # Auto-accept status
        if self._status.auto_accept:
            parts.append("\033[32mAuto-accept: ON\033[0m")
        else:
            parts.append("\033[2mAuto-accept: off\033[0m")

        # Token stats
        total_tokens = (
            self._status.input_tokens
            + self._status.output_tokens
            + self._status.thinking_tokens
        )
        parts.append(
            f"\033[2mTokens (last): {total_tokens:,} "
            f"= {self._status.input_tokens:,}↓ "
            f"{self._status.output_tokens:,}↑ "
            f"{self._status.thinking_tokens:,}t\033[0m"
        )

        # Escape hint when activity is shown
        if self._status.activity:
            parts.append("\033[2m(Esc to cancel)\033[0m")

        return " • ".join(parts)

    def _get_all_lines(self) -> list[str]:
        """Get all lines to render (content + status bar)."""
        lines = list(self._content_lines) if self._content_lines else []
        lines.append(self._format_status_line())
        return lines

    def _render_lines(self, lines: list[str]) -> None:
        """Render the provided lines, resizing the region if needed."""
        if len(lines) > self._region.num_lines:
            self._region.update_size(len(lines))
        self._region.render(lines)

    def activate(self) -> None:
        """Activate the footer and reserve terminal region."""
        if self._state == "active":
            return
        if self._state == "paused":
            self.resume()
            return

        self._state = "active"
        # Start with just status bar (1 line)
        lines = self._get_all_lines()
        self._region.activate(len(lines))
        self._render_lines(lines)

    def deactivate(self) -> None:
        """Deactivate the footer and release terminal region."""
        if self._state == "inactive":
            return

        self._region.deactivate()
        self._state = "inactive"
        self._content_lines = []

    def render(self) -> None:
        """Render current content + status bar."""
        if self._state != "active":
            return

        self._render_lines(self._get_all_lines())

    def set_content(self, lines: list[str]) -> None:
        """Set the content area lines (above status bar)."""
        self._content_lines = list(lines)
        if self._state == "active":
            self.render()

    def clear_content(self) -> None:
        """Clear content area, keep status bar visible."""
        self._content_lines = []
        if self._state == "active":
            self.render()

    def update_status(
        self,
        auto_accept: bool | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
    ) -> None:
        """Update status bar values and re-render."""
        if auto_accept is not None:
            self._status.auto_accept = auto_accept
        if input_tokens is not None:
            self._status.input_tokens = input_tokens
        if output_tokens is not None:
            self._status.output_tokens = output_tokens
        if thinking_tokens is not None:
            self._status.thinking_tokens = thinking_tokens
        if self._state == "active":
            self.render()

    def set_activity(self, activity: str | None) -> None:
        """Set activity text (shows spinner when not None)."""
        self._status.activity = activity
        if self._state == "active":
            self.render()

    def is_active(self) -> bool:
        """Check if footer is currently active."""
        return self._state == "active"

    def pause(self) -> None:
        """Temporarily hide the footer to allow normal printing.

        Call this before using console.print() or other output.
        The footer region is cleared and cursor is positioned for normal output.
        Call resume() after printing to restore the footer.
        """
        if self._state != "active":
            return

        # Clear the region and show cursor for normal output.
        self._region.deactivate()
        self._state = "paused"

    def resume(self) -> None:
        """Restore the footer after pause().

        Call this after console.print() to show the footer again at the
        new cursor position.
        """
        if self._state != "paused":
            return

        # Re-activate at current cursor position
        lines = self._get_all_lines()
        self._region.activate(len(lines))
        self._region.render(lines)
        self._state = "active"

    def resync_position(self) -> None:
        """Re-sync the footer position after external content is printed.

        This is a convenience method that pauses and resumes the footer.
        Call this after using console.print() while the footer is active.
        """
        if self._state != "active":
            return

        self.pause()
        self.resume()
