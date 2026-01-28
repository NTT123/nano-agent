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
    cursor_char: str = "█"
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
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
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

    def get_lines(self) -> list[str]:
        """Get lines for rendering in footer content area."""
        display = (
            self.buffer[: self.cursor_pos]
            + self.cursor_char
            + self.buffer[self.cursor_pos :]
        )
        parts = display.split("\n")
        if not parts:
            return [self.prompt + self.cursor_char]
        lines = [f"{self.prompt}{parts[0]}"]
        indent = " " * len(self.prompt)
        for part in parts[1:]:
            lines.append(f"{indent}{part}")
        return lines

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
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return (False, None)
        if event.ctrl and event.char == "f":
            if self.cursor_pos < len(self.buffer):
                self.cursor_pos += 1
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
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return (False, None)
        if event.key == "Right":
            if self.cursor_pos < len(self.buffer):
                self.cursor_pos += 1
            return (False, None)

        # Printable characters
        if event.char and event.char.isprintable():
            self._insert_char(event.char)
            return (False, None)

        return (False, None)
