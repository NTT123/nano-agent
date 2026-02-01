"""Footer-compatible text input with history support.

This module provides FooterInput - a multiline text prompt with:
- History navigation (up/down arrows)
- Load/save history file
- Integration with TerminalFooter via get_lines()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .base import ActiveElement, InputEvent
from .terminal import ANSI


@dataclass
class FooterInput(ActiveElement[str | None]):
    """Multiline text input with history support.

    Designed to work with FooterElementManager which renders via footer.set_content().

    Features:
    - Basic line editing (Backspace, Left/Right, Ctrl+A/E/K/U/W/Y)
    - Multiline input via backslash + Enter
    - History navigation (Up/Down arrows)
    - Persisted history file

    Returns:
    - Entered text on Enter
    - Empty string on Escape/Ctrl+C
    - None on Ctrl+D (EOF)
    """

    prompt: str = "> "
    buffer: str = ""
    cursor_pos: int = 0
    cursor_char: str = " "  # Shown in reverse video when cursor at end
    allow_multiline: bool = True
    kill_buffer: str = ""

    # History support
    history_file: str | None = None
    _history: list[str] = field(default_factory=list)
    _history_index: int = field(default=-1, init=False, repr=False)
    _history_loaded: bool = field(default=False, init=False, repr=False)
    _temp_buffer: str = field(default="", init=False, repr=False)

    # Callbacks
    on_toggle_auto_accept: Callable[[], None] | None = None

    def on_activate(self) -> None:
        """Load history when activated."""
        if not self._history_loaded and self.history_file:
            self._load_history()
            self._history_loaded = True
        self._history_index = -1
        self._temp_buffer = ""

    def on_deactivate(self) -> None:
        """Save history when deactivated (if there was input)."""
        # History is saved per-submission, not on deactivate
        pass

    def completion_delay(self) -> float:
        """No delay for text input - user already sees their input."""
        return 0.0

    def _load_history(self) -> None:
        """Load history from file."""
        if not self.history_file:
            return
        path = Path(os.path.expanduser(self.history_file))
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                # Each entry is separated by a special delimiter to handle multiline
                entries = content.split("\n\x00\n")
                self._history = [e for e in entries if e.strip()]
            except (IOError, OSError):
                self._history = []

    def _save_history(self, entry: str) -> None:
        """Save a new entry to history file."""
        if not self.history_file or not entry.strip():
            return

        # Add to in-memory history
        if entry not in self._history:
            self._history.append(entry)
            # Keep last 1000 entries
            if len(self._history) > 1000:
                self._history = self._history[-1000:]

        # Save to file
        path = Path(os.path.expanduser(self.history_file))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = "\n\x00\n".join(self._history)
            path.write_text(content, encoding="utf-8")
        except (IOError, OSError):
            pass

    def _insert_char(self, ch: str) -> None:
        self.buffer = (
            self.buffer[: self.cursor_pos] + ch + self.buffer[self.cursor_pos :]
        )
        self.cursor_pos += len(ch)

    def _insert_text(self, text: str) -> None:
        self.buffer = (
            self.buffer[: self.cursor_pos] + text + self.buffer[self.cursor_pos :]
        )
        self.cursor_pos += len(text)

    def _normalize_paste(self, text: str) -> str:
        """Normalize pasted text for insertion.

        Handles:
        - Line ending normalization (CRLF, CR -> LF)
        - Multiline restriction if disabled
        - Tab expansion (tabs display as variable width, convert to spaces)
        """
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        # Expand tabs to 4 spaces (tabs cause cursor positioning issues
        # since they count as 1 char but display as variable width)
        normalized = normalized.replace("\t", "    ")
        if not self.allow_multiline:
            normalized = normalized.replace("\n", " ")
        return normalized

    def _delete_before_cursor(self) -> None:
        if self.cursor_pos > 0:
            self.buffer = (
                self.buffer[: self.cursor_pos - 1] + self.buffer[self.cursor_pos :]
            )
            self.cursor_pos -= 1

    def _delete_prev_word(self) -> None:
        if self.cursor_pos == 0:
            return
        i = self.cursor_pos
        while i > 0 and self.buffer[i - 1].isspace():
            i -= 1
        while i > 0 and not self.buffer[i - 1].isspace():
            i -= 1
        self.kill_buffer = self.buffer[i : self.cursor_pos]
        self.buffer = self.buffer[:i] + self.buffer[self.cursor_pos :]
        self.cursor_pos = i

    def _history_prev(self) -> None:
        """Navigate to previous history entry."""
        if not self._history:
            return

        if self._history_index == -1:
            # Save current buffer before navigating
            self._temp_buffer = self.buffer

        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            # History is stored oldest-first, so go from end
            idx = len(self._history) - 1 - self._history_index
            self.buffer = self._history[idx]
            self.cursor_pos = len(self.buffer)

    def _history_next(self) -> None:
        """Navigate to next history entry."""
        if self._history_index > 0:
            self._history_index -= 1
            idx = len(self._history) - 1 - self._history_index
            self.buffer = self._history[idx]
            self.cursor_pos = len(self.buffer)
        elif self._history_index == 0:
            # Return to current input
            self._history_index = -1
            self.buffer = self._temp_buffer
            self.cursor_pos = len(self.buffer)

    def _can_move_left(self) -> bool:
        """Check if cursor can move left."""
        return self.cursor_pos > 0

    def _can_move_right(self) -> bool:
        """Check if cursor can move right."""
        return self.cursor_pos < len(self.buffer)

    def _move_cursor_left(self) -> None:
        """Move cursor left by one position if possible."""
        if self._can_move_left():
            self.cursor_pos -= 1

    def _move_cursor_right(self) -> None:
        """Move cursor right by one position if possible."""
        if self._can_move_right():
            self.cursor_pos += 1

    def _render_buffer_with_cursor(self) -> str:
        """Return buffer string with cursor position highlighted in reverse video."""
        if self.cursor_pos < len(self.buffer):
            # Cursor within text - show character at cursor in reverse video
            char_at_cursor = self.buffer[self.cursor_pos]
            return (
                self.buffer[: self.cursor_pos]
                + ANSI.REVERSE
                + char_at_cursor
                + ANSI.RESET
                + self.buffer[self.cursor_pos + 1 :]
            )
        # Cursor at end - show space in reverse video
        return self.buffer + ANSI.REVERSE + self.cursor_char + ANSI.RESET

    def get_lines(self) -> list[str]:
        """Get lines for rendering in footer content area."""
        terminal_width = ANSI.get_terminal_width()
        prompt_width = len(self.prompt)  # Visual width of prompt (no ANSI codes)

        display = self._render_buffer_with_cursor()

        # Split by logical newlines first
        logical_lines = display.split("\n")

        result = []
        for idx, logical_line in enumerate(logical_lines):
            if idx == 0:
                # First logical line: prompt prefix, wrap to fit
                # Guard against negative width on very narrow terminals
                first_width = max(1, terminal_width - prompt_width)
                first_wrapped = ANSI.wrap_to_width(logical_line, first_width)

                if first_wrapped:
                    result.append(self.prompt + first_wrapped[0])

                    # Continuation of first line uses full width (no prefix)
                    if len(first_wrapped) > 1:
                        remaining = "".join(first_wrapped[1:])
                        continuation_wrapped = ANSI.wrap_to_width(
                            remaining, max(1, terminal_width)
                        )
                        result.extend(continuation_wrapped)
            else:
                # Subsequent logical lines (from explicit newlines): no prefix, full width
                wrapped = ANSI.wrap_to_width(logical_line, max(1, terminal_width))
                result.extend(wrapped)

        if not result:
            return [self.prompt + ANSI.REVERSE + self.cursor_char + ANSI.RESET]

        return result

    def handle_input(self, event: InputEvent) -> tuple[bool, str | None]:
        # Handle Enter - submit or newline
        if event.key == "Enter":
            if (
                self.allow_multiline
                and self.cursor_pos == len(self.buffer)
                and self.cursor_pos > 0
                and self.buffer[self.cursor_pos - 1] == "\\"
            ):
                # Backslash + Enter inserts newline
                self._delete_before_cursor()
                self._insert_char("\n")
                return (False, None)
            result = self.buffer
            if result.strip():
                self._save_history(result)
            self.buffer = ""
            self.cursor_pos = 0
            self._history_index = -1
            return (True, result)
        if event.key == "ShiftEnter":
            if self.allow_multiline:
                self._insert_char("\n")
                return (False, None)
            return (False, None)

        # History navigation
        if event.key == "Up":
            self._history_prev()
            return (False, None)
        if event.key == "Down":
            self._history_next()
            return (False, None)

        # Shift+Tab toggles auto-accept
        if event.key == "BackTab":
            if self.on_toggle_auto_accept:
                self.on_toggle_auto_accept()
            return (False, None)

        # Bracketed paste
        if event.key == "Paste" and event.char:
            self._insert_text(self._normalize_paste(event.char))
            return (False, None)

        # Cursor movement
        if event.ctrl and event.char == "a":
            self.cursor_pos = 0
            return (False, None)
        if event.ctrl and event.char == "e":
            self.cursor_pos = len(self.buffer)
            return (False, None)
        if event.ctrl and event.char == "b":
            self._move_cursor_left()
            return (False, None)
        if event.ctrl and event.char == "f":
            self._move_cursor_right()
            return (False, None)

        # Editing
        if event.ctrl and event.char == "w":
            self._delete_prev_word()
            return (False, None)
        if event.ctrl and event.char == "k":
            self.kill_buffer = self.buffer[self.cursor_pos :]
            self.buffer = self.buffer[: self.cursor_pos]
            return (False, None)
        if event.ctrl and event.char == "u":
            self.kill_buffer = self.buffer[: self.cursor_pos]
            self.buffer = self.buffer[self.cursor_pos :]
            self.cursor_pos = 0
            return (False, None)
        if event.ctrl and event.char == "y":
            if self.kill_buffer:
                self._insert_char(self.kill_buffer)
            return (False, None)
        if event.ctrl and event.char == "l":
            self.buffer = ""
            self.cursor_pos = 0
            return (False, None)
        if event.ctrl and event.char == "j":
            # Ctrl+J - ignore (don't submit)
            return (False, None)

        # Cancel/exit
        if event.key == "Escape" or (event.ctrl and event.char == "c"):
            self.buffer = ""
            self.cursor_pos = 0
            return (True, "")
        if event.ctrl and event.char == "d":
            self.buffer = ""
            self.cursor_pos = 0
            return (True, None)

        # Backspace
        if event.key == "Backspace":
            self._delete_before_cursor()
            return (False, None)

        # Arrow keys
        if event.key == "Left":
            self._move_cursor_left()
            return (False, None)
        if event.key == "Right":
            self._move_cursor_right()
            return (False, None)

        # Printable characters
        if event.char and event.char.isprintable():
            self._insert_char(event.char)
            return (False, None)

        return (False, None)
