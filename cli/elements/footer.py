"""Unified terminal footer with content area and status bar.

This module provides:
- FooterState: Enum for footer display states
- StatusBarState: Dataclass holding status bar values
- TerminalFooter: Unified footer with content + status bar rendering
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from .terminal import ANSI, TerminalRegion


class FooterState(Enum):
    """Footer display states with explicit values.

    Valid transitions:
        INACTIVE ──activate()──> ACTIVE
        ACTIVE ──deactivate()──> INACTIVE
        ACTIVE ──pause()──> PAUSED
        PAUSED ──resume()──> ACTIVE
        PAUSED ──deactivate()──> INACTIVE
    """

    INACTIVE = auto()  # Not rendered, no terminal region reserved
    ACTIVE = auto()  # Fully active, rendering content + status bar
    PAUSED = auto()  # Temporarily hidden for console output


@dataclass
class StatusBarState:
    """State for the status bar display."""

    auto_accept: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost: float = 0.0
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
        self._state: FooterState = FooterState.INACTIVE
        self._content_lines: list[str] = []
        self._status = StatusBarState()
        # Lock to prevent concurrent render operations and state transitions
        # (e.g., _refresh_loop and input handling both calling render())
        self._render_lock = threading.Lock()

    # Valid state transitions
    _VALID_TRANSITIONS: dict[FooterState, set[FooterState]] = {
        FooterState.INACTIVE: {FooterState.ACTIVE},
        FooterState.ACTIVE: {FooterState.INACTIVE, FooterState.PAUSED},
        FooterState.PAUSED: {FooterState.ACTIVE, FooterState.INACTIVE},
    }

    def _can_transition_to(self, new_state: FooterState) -> bool:
        """Check if transition to new_state is valid from current state."""
        return new_state in self._VALID_TRANSITIONS.get(self._state, set())

    def _transition_to(self, new_state: FooterState) -> bool:
        """Attempt state transition. Returns True if successful.

        Must be called while holding _render_lock.
        """
        if self._state == new_state:
            return True  # Already in target state
        if not self._can_transition_to(new_state):
            return False
        self._state = new_state
        return True

    def _assert_invariants(self) -> None:
        """Assert state invariants (for debugging).

        Call after transitions to verify state consistency.
        Raises AssertionError if invariants are violated.

        Invariants:
        - INACTIVE: content empty, region inactive
        - ACTIVE: region active, region.num_lines == len(content) + 1
        - PAUSED: region inactive, content may be preserved
        """
        if self._state == FooterState.INACTIVE:
            assert (
                not self._content_lines
            ), f"Content should be empty when inactive, got {len(self._content_lines)} lines"
            assert (
                not self._region._active
            ), "Region should be inactive when footer inactive"
        elif self._state == FooterState.ACTIVE:
            assert self._region._active, "Region should be active when footer active"
            expected_lines = len(self._content_lines) + 1  # +1 for status bar
            assert (
                self._region.num_lines == expected_lines
            ), f"Region lines {self._region.num_lines} != expected {expected_lines}"
        elif self._state == FooterState.PAUSED:
            assert not self._region._active, "Region should be inactive when paused"
            # Note: content_lines may be preserved when paused (for resume)

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
            parts.append(f"{spinner} {ANSI.CYAN}{self._status.activity}{ANSI.RESET}")

        # Auto-accept status
        if self._status.auto_accept:
            parts.append(f"{ANSI.GREEN}Auto-accept: ON{ANSI.RESET}")
        else:
            parts.append(f"{ANSI.DIM}Auto-accept: off{ANSI.RESET}")

        # Token stats
        total_tokens = (
            self._status.input_tokens
            + self._status.output_tokens
            + self._status.thinking_tokens
        )
        parts.append(
            f"{ANSI.DIM}Tokens (last): {total_tokens:,} "
            f"= {self._status.input_tokens:,}↓ "
            f"{self._status.output_tokens:,}↑ "
            f"{self._status.thinking_tokens:,}t{ANSI.RESET}"
        )

        # Cost display
        if self._status.cost > 0:
            from nano_agent.providers.cost import format_cost

            parts.append(f"{ANSI.YELLOW}{format_cost(self._status.cost)}{ANSI.RESET}")

        # Escape hint when activity is shown
        if self._status.activity:
            parts.append(f"{ANSI.DIM}(Esc to cancel){ANSI.RESET}")

        line = " • ".join(parts)

        # Truncate to terminal width
        return ANSI.truncate_to_width(line, ANSI.get_terminal_width())

    def _get_all_lines(self) -> list[str]:
        """Get all lines to render (content + status bar)."""
        # Content lines are already wrapped by FooterInput
        lines = list(self._content_lines) if self._content_lines else []
        lines.append(self._format_status_line())  # Status bar stays truncated
        return lines

    def _render_lines(self, lines: list[str]) -> None:
        """Render the provided lines, resizing the region if needed."""
        if len(lines) > self._region.num_lines:
            self._region.update_size(len(lines))
        self._region.render(lines)

    def activate(self) -> None:
        """Activate the footer and reserve terminal region.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            if self._state == FooterState.ACTIVE:
                return
            if self._state == FooterState.PAUSED:
                self._do_resume()
                return
            self._do_activate()

    def _do_activate(self) -> None:
        """Internal activate (must hold lock)."""
        self._transition_to(FooterState.ACTIVE)
        lines = self._get_all_lines()
        self._region.activate(len(lines))
        self._render_lines(lines)

    def deactivate(self) -> None:
        """Deactivate the footer and release terminal region.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            if self._state == FooterState.INACTIVE:
                return
            self._do_deactivate()

    def _do_deactivate(self) -> None:
        """Internal deactivate (must hold lock)."""
        self._region.deactivate()
        self._transition_to(FooterState.INACTIVE)
        self._content_lines = []

    def render(self) -> None:
        """Render current content + status bar.

        Thread-safe: uses lock to prevent concurrent render operations
        from corrupting terminal output.
        """
        with self._render_lock:
            if self._state != FooterState.ACTIVE:
                return
            self._render_lines(self._get_all_lines())

    def set_content(self, lines: list[str]) -> None:
        """Set the content area lines (above status bar).

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            self._content_lines = list(lines)
            if self._state == FooterState.ACTIVE:
                self._render_lines(self._get_all_lines())

    def clear_content(self) -> None:
        """Clear content area, keep status bar visible.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            self._content_lines = []
            if self._state == FooterState.ACTIVE:
                self._render_lines(self._get_all_lines())

    def overwrite_content_with_message(self) -> int:
        """Prepare to overwrite content area with a message.

        Moves cursor to start of content area so caller can print directly
        over the existing content. Returns number of content lines.

        After printing, caller should call finish_content_overwrite().
        """
        if self._state != FooterState.ACTIVE:
            return 0

        # Move cursor to top of region (content starts there)
        self._region._move_to_region_start()

        # Return number of content lines for caller to know
        return len(self._content_lines)

    def finish_content_overwrite(self, lines_printed: int) -> None:
        """Complete the overwrite transition after printing message.

        Clears any extra lines if old content was taller than new,
        then resizes region to just status bar.
        """
        if self._state != FooterState.ACTIVE:
            return

        old_content_lines = len(self._content_lines)

        # If we printed fewer lines than old content, clear the extras
        if lines_printed < old_content_lines:
            # Move to the line after what was printed
            lines_to_clear = old_content_lines - lines_printed
            for i in range(lines_to_clear):
                ANSI.clear_line()
                if i < lines_to_clear - 1:
                    ANSI.move_down(1)

        # Clear content state
        self._content_lines = []

        # Resize region to just status bar (1 line)
        # The cursor is currently somewhere after printed content
        # We need to position for just the status bar
        self._region.num_lines = 1
        self._region._cursor_at_line = 0

        # Render the status bar
        self.render()

    def transition_to_message(self, render_callback: Callable[[], int]) -> None:
        """Atomically transition from input content to rendered message.

        This is the seamless transition path: instead of pause/resume which
        causes visual flicker, we overwrite the footer content in place with
        the rendered message.

        Args:
            render_callback: Function that renders content and returns lines printed
        """
        if self._state != FooterState.ACTIVE or not self._content_lines:
            # Normal path - use pause/resume
            self._region.deactivate()
            render_callback()
            self._region.activate(1)
            self._render_lines(self._get_all_lines())
            self._state = FooterState.ACTIVE
            return

        # Seamless path: overwrite content in place
        old_count = len(self._content_lines)

        # Move cursor to top of region (content starts there)
        self._region.move_to_start()

        # Render the message
        lines_printed = render_callback()

        # Clear extra lines if old content was taller than new
        if lines_printed < old_count:
            lines_to_clear = old_count - lines_printed
            for i in range(lines_to_clear):
                ANSI.clear_line()
                if i < lines_to_clear - 1:
                    ANSI.move_down(1)

        # Reset footer state
        self._content_lines = []
        self._region.num_lines = 1
        self._region._cursor_at_line = 0

        # Render status bar
        self.render()

    def update_status(
        self,
        auto_accept: bool | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
        cost: float | None = None,
    ) -> None:
        """Update status bar values and re-render.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            if auto_accept is not None:
                self._status.auto_accept = auto_accept
            if input_tokens is not None:
                self._status.input_tokens = input_tokens
            if output_tokens is not None:
                self._status.output_tokens = output_tokens
            if thinking_tokens is not None:
                self._status.thinking_tokens = thinking_tokens
            if cost is not None:
                self._status.cost = cost
            if self._state == FooterState.ACTIVE:
                self._render_lines(self._get_all_lines())

    def set_activity(self, activity: str | None) -> None:
        """Set activity text (shows spinner when not None).

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            self._status.activity = activity
            if self._state == FooterState.ACTIVE:
                self._render_lines(self._get_all_lines())

    def is_active(self) -> bool:
        """Check if footer is currently active."""
        return self._state == FooterState.ACTIVE

    def is_paused(self) -> bool:
        """Check if footer is currently paused."""
        return self._state == FooterState.PAUSED

    def has_content(self) -> bool:
        """Check if footer has content lines."""
        return bool(self._content_lines)

    def get_state(self) -> str:
        """Get current footer state as string (backward compatible).

        Returns lowercase state name: 'inactive', 'active', or 'paused'.
        """
        return self._state.name.lower()

    def pause(self) -> None:
        """Temporarily hide the footer to allow normal printing.

        Call this before using console.print() or other output.
        The footer region is cleared and cursor is positioned for normal output.
        Call resume() after printing to restore the footer.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            if self._state != FooterState.ACTIVE:
                return
            self._do_pause()

    def _do_pause(self) -> None:
        """Internal pause (must hold lock)."""
        self._region.deactivate()
        self._transition_to(FooterState.PAUSED)

    def resume(self) -> None:
        """Restore the footer after pause().

        Call this after console.print() to show the footer again at the
        new cursor position.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            if self._state != FooterState.PAUSED:
                return
            self._do_resume()

    def _do_resume(self) -> None:
        """Internal resume (must hold lock)."""
        lines = self._get_all_lines()
        self._region.activate(len(lines))
        self._region.render(lines)
        self._transition_to(FooterState.ACTIVE)

    def resync_position(self) -> None:
        """Re-sync the footer position after external content is printed.

        This is a convenience method that pauses and resumes the footer.
        Call this after using console.print() while the footer is active.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._render_lock:
            if self._state != FooterState.ACTIVE:
                return
            self._do_pause()
            self._do_resume()
