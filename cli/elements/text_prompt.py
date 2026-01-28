"""Simple text input prompt element.

A basic text input with line editing. For richer features (history,
multi-line, etc.), use PromptToolkitInput instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ActiveElement, InputEvent


@dataclass
class TextPrompt(ActiveElement[str | None]):
    """Text input with basic line editing.

    Returns the entered text on Enter, or empty string on Escape/Ctrl+C.
    """

    prompt: str = "> "
    buffer: str = ""
    cursor_pos: int = 0
    cursor_char: str = "█"
    kill_buffer: str = ""

    def get_lines(self) -> list[str]:
        display = (
            self.buffer[: self.cursor_pos]
            + self.cursor_char
            + self.buffer[self.cursor_pos :]
        )
        return [f"{self.prompt}{display}"]

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
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")

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

    def handle_input(self, event: InputEvent) -> tuple[bool, str | None]:
        if event.key == "Enter":
            result = self.buffer
            self.buffer = ""
            self.cursor_pos = 0
            return (True, result)
        elif event.key == "ShiftEnter":
            return (False, None)
        elif event.key == "Paste" and event.char:
            self._insert_text(self._normalize_paste(event.char))
            return (False, None)
        elif event.ctrl and event.char == "a":
            self.cursor_pos = 0
            return (False, None)
        elif event.ctrl and event.char == "e":
            self.cursor_pos = len(self.buffer)
            return (False, None)
        elif event.ctrl and event.char == "b":
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return (False, None)
        elif event.ctrl and event.char == "f":
            if self.cursor_pos < len(self.buffer):
                self.cursor_pos += 1
            return (False, None)
        elif event.ctrl and event.char == "w":
            self._delete_prev_word()
            return (False, None)
        elif event.ctrl and event.char == "k":
            self.kill_buffer = self.buffer[self.cursor_pos :]
            self.buffer = self.buffer[: self.cursor_pos]
            return (False, None)
        elif event.ctrl and event.char == "u":
            self.kill_buffer = self.buffer[: self.cursor_pos]
            self.buffer = self.buffer[self.cursor_pos :]
            self.cursor_pos = 0
            return (False, None)
        elif event.ctrl and event.char == "y":
            if self.kill_buffer:
                self._insert_char(self.kill_buffer)
            return (False, None)
        elif event.ctrl and event.char == "l":
            self.buffer = ""
            self.cursor_pos = 0
            return (False, None)
        elif event.ctrl and event.char == "j":
            return (False, None)
        elif event.key == "Escape" or (event.ctrl and event.char == "c"):
            self.buffer = ""
            self.cursor_pos = 0
            return (True, "")
        elif event.key == "Backspace":
            self._delete_before_cursor()
            return (False, None)
        elif event.key == "Left":
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return (False, None)
        elif event.key == "Right":
            if self.cursor_pos < len(self.buffer):
                self.cursor_pos += 1
            return (False, None)
        elif event.char and event.char.isprintable():
            self._insert_char(event.char)
            return (False, None)
        return (False, None)


@dataclass
class MultiLinePrompt(ActiveElement[str | None]):
    """Text input with basic line editing and multiline support.

    Multiline input uses backslash + Enter to insert a newline.
    Returns the entered text on Enter, empty string on Escape/Ctrl+C,
    or None on Ctrl+D.
    """

    prompt: str = "> "
    buffer: str = ""
    cursor_pos: int = 0
    cursor_char: str = "█"
    allow_multiline: bool = True
    kill_buffer: str = ""

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
        return text.replace("\r\n", "\n").replace("\r", "\n")

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

    def get_lines(self) -> list[str]:
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
        if event.key == "Enter":
            # Backslash + Enter inserts a newline when cursor is at end.
            if (
                self.allow_multiline
                and self.cursor_pos == len(self.buffer)
                and self.cursor_pos > 0
                and self.buffer[self.cursor_pos - 1] == "\\"
            ):
                self._delete_before_cursor()
                self._insert_char("\n")
                return (False, None)
            result = self.buffer
            self.buffer = ""
            self.cursor_pos = 0
            return (True, result)
        if event.key == "ShiftEnter":
            if self.allow_multiline:
                self._insert_char("\n")
                return (False, None)
            return (False, None)
        if event.key == "Paste" and event.char:
            pasted = self._normalize_paste(event.char)
            if not self.allow_multiline:
                pasted = pasted.replace("\n", " ")
            self._insert_text(pasted)
            return (False, None)
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
            return (False, None)
        if event.key == "Escape" or (event.ctrl and event.char == "c"):
            self.buffer = ""
            self.cursor_pos = 0
            return (True, "")
        if event.ctrl and event.char == "d":
            self.buffer = ""
            self.cursor_pos = 0
            return (True, None)
        if event.key == "Backspace":
            self._delete_before_cursor()
            return (False, None)
        if event.key == "Left":
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return (False, None)
        if event.key == "Right":
            if self.cursor_pos < len(self.buffer):
                self.cursor_pos += 1
            return (False, None)
        if event.char and event.char.isprintable():
            self._insert_char(event.char)
            return (False, None)
        return (False, None)
