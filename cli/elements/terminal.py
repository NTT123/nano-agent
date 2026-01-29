"""Terminal control for interactive elements.

This module provides:
- ANSI: Centralized terminal escape sequences and helpers
- TerminalRegion: Controls a region at the bottom of the terminal
- RawInputReader: Reads single keystrokes in raw mode
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import re
import sys
import termios
import tty

from .base import InputEvent


class ANSI:
    """Centralized ANSI escape sequences and terminal control helpers.

    Usage:
        from .terminal import ANSI

        # Colors
        print(f"{ANSI.CYAN}colored text{ANSI.RESET}")

        # Cursor control (returns escape string)
        sys.stdout.write(ANSI.cursor_up(2))

        # Direct write (writes and flushes)
        ANSI.hide_cursor()
        ANSI.clear_line()
    """

    # Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    # Cursor visibility
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"

    # Cursor position
    SAVE_CURSOR = "\033[s"
    RESTORE_CURSOR = "\033[u"

    # Line control
    CLEAR_LINE = "\033[2K"
    CARRIAGE_RETURN = "\r"
    ENABLE_BRACKETED_PASTE = "\033[?2004h"
    DISABLE_BRACKETED_PASTE = "\033[?2004l"

    # Pattern to match ANSI escape sequences (for stripping)
    _ANSI_PATTERN = re.compile(r"\033\[[0-9;]*m")

    @classmethod
    def cursor_up(cls, n: int = 1) -> str:
        """Move cursor up n lines."""
        return f"\033[{n}A" if n > 0 else ""

    @classmethod
    def cursor_down(cls, n: int = 1) -> str:
        """Move cursor down n lines."""
        return f"\033[{n}B" if n > 0 else ""

    @classmethod
    def cursor_right(cls, n: int = 1) -> str:
        """Move cursor right n columns."""
        return f"\033[{n}C" if n > 0 else ""

    @classmethod
    def cursor_left(cls, n: int = 1) -> str:
        """Move cursor left n columns."""
        return f"\033[{n}D" if n > 0 else ""

    @classmethod
    def strip_ansi(cls, s: str) -> str:
        """Remove ANSI escape sequences from string."""
        return cls._ANSI_PATTERN.sub("", s)

    @classmethod
    def visual_len(cls, s: str) -> int:
        """Calculate visual length of string, excluding ANSI escape codes."""
        return len(cls.strip_ansi(s))

    # Direct write helpers (write to stdout and flush)
    @classmethod
    def _write(cls, s: str) -> None:
        """Write to stdout and flush."""
        sys.stdout.write(s)
        sys.stdout.flush()

    @classmethod
    def hide_cursor(cls) -> None:
        """Hide the terminal cursor."""
        cls._write(cls.HIDE_CURSOR)

    @classmethod
    def show_cursor(cls) -> None:
        """Show the terminal cursor."""
        cls._write(cls.SHOW_CURSOR)

    @classmethod
    def save_cursor(cls) -> None:
        """Save cursor position."""
        cls._write(cls.SAVE_CURSOR)

    @classmethod
    def restore_cursor(cls) -> None:
        """Restore cursor to saved position."""
        cls._write(cls.RESTORE_CURSOR)

    @classmethod
    def clear_line(cls) -> None:
        """Clear the current line and return to column 1."""
        cls._write(cls.CLEAR_LINE + cls.CARRIAGE_RETURN)

    @classmethod
    def move_up(cls, n: int = 1) -> None:
        """Move cursor up n lines."""
        if n > 0:
            cls._write(cls.cursor_up(n))

    @classmethod
    def move_down(cls, n: int = 1) -> None:
        """Move cursor down n lines."""
        if n > 0:
            cls._write(cls.cursor_down(n))

    @classmethod
    def move_right(cls, n: int = 1) -> None:
        """Move cursor right n columns."""
        if n > 0:
            cls._write(cls.cursor_right(n))

    @classmethod
    def enable_bracketed_paste(cls) -> None:
        """Enable bracketed paste mode."""
        cls._write(cls.ENABLE_BRACKETED_PASTE)

    @classmethod
    def disable_bracketed_paste(cls) -> None:
        """Disable bracketed paste mode."""
        cls._write(cls.DISABLE_BRACKETED_PASTE)


class RawInputReader:
    """Reads single keystrokes from terminal in raw mode."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old_settings: list | None = None

    def start(self) -> None:
        """Enter raw mode and flush any pending input."""
        self.old_settings = termios.tcgetattr(self.fd)
        # Flush any pending input to avoid stale keystrokes
        termios.tcflush(self.fd, termios.TCIFLUSH)
        tty.setraw(self.fd)
        # Re-enable output post-processing so '\n' moves to column 1.
        attrs = termios.tcgetattr(self.fd)
        attrs[1] |= termios.OPOST | termios.ONLCR
        termios.tcsetattr(self.fd, termios.TCSADRAIN, attrs)
        ANSI.enable_bracketed_paste()

    def stop(self) -> None:
        """Restore terminal settings."""
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            self.old_settings = None
        ANSI.disable_bracketed_paste()

    def flush(self) -> None:
        """Flush any pending input in the buffer."""
        termios.tcflush(self.fd, termios.TCIFLUSH)

    async def read(self) -> InputEvent:
        """Read a single input event (async-friendly)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_sync)

    def read_nonblocking(self, timeout: float = 0.0) -> InputEvent | None:
        """Attempt to read a single input event without blocking.

        Args:
            timeout: Seconds to wait for input before returning None.
        """
        import select

        if select.select([self.fd], [], [], timeout)[0]:
            return self._read_sync()
        return None

    def _read_sync(self) -> InputEvent:
        """Synchronous read of a single key."""
        # Read first character (blocking)
        ch = os.read(self.fd, 1).decode("utf-8", errors="ignore")

        # Handle special keys
        if ch == "\r":
            return InputEvent(key="Enter", char=None)
        elif ch == "\n":
            # Treat Ctrl+J (LF) as a control key, not Enter
            return InputEvent(key="j", char="j", ctrl=True)
        elif ch == "\x1b":  # Escape - check for sequences
            # Set non-blocking mode to check for more chars
            flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            try:
                seq = self._read_escape_sequence()
                if seq == "[A":
                    return InputEvent(key="Up", char=None)
                if seq == "[B":
                    return InputEvent(key="Down", char=None)
                if seq == "[C":
                    return InputEvent(key="Right", char=None)
                if seq == "[D":
                    return InputEvent(key="Left", char=None)
                if seq == "[Z":
                    # Shift+Tab - return as BackTab, not Escape
                    return InputEvent(key="BackTab", char=None)
                if seq in ("[13;2~", "[13;2u", "[27;2;13~"):
                    return InputEvent(key="ShiftEnter", char=None)
                if seq == "[200~":
                    # Bracketed paste start
                    fcntl.fcntl(self.fd, fcntl.F_SETFL, flags)
                    pasted = self._read_bracketed_paste()
                    return InputEvent(key="Paste", char=pasted)
                if seq and os.environ.get("NANO_CLI_DEBUG_KEYS") == "1":
                    sys.stdout.write(f"[debug] unknown escape seq: {seq!r}\n")
                    sys.stdout.flush()
                # No sequence found - plain Escape
                return InputEvent(key="Escape", char=None)
            finally:
                # Restore blocking mode
                fcntl.fcntl(self.fd, fcntl.F_SETFL, flags)

        elif ch == "\x7f" or ch == "\x08":  # Backspace
            return InputEvent(key="Backspace", char=None)
        elif ch == "\x03":  # Ctrl+C
            return InputEvent(key="c", char="c", ctrl=True)
        elif ch == "\x04":  # Ctrl+D
            return InputEvent(key="d", char="d", ctrl=True)
        elif ord(ch) < 32:  # Other control characters
            # Map Ctrl+<letter> to its letter (Ctrl+A -> "a", etc.)
            letter = chr(ord(ch) + 96)
            if "a" <= letter <= "z":
                return InputEvent(key=letter, char=letter, ctrl=True)
            return InputEvent(key=ch, char=None, ctrl=True)
        else:
            return InputEvent(key=ch, char=ch)

    def _read_escape_sequence(self) -> str | None:
        """Read an escape sequence after ESC in non-blocking mode."""
        try:
            ch2 = os.read(self.fd, 1)
        except (BlockingIOError, OSError):
            return None
        if ch2 == b"[":
            seq = bytearray()
            # CSI: read until final byte in 0x40..0x7E
            while True:
                try:
                    b = os.read(self.fd, 1)
                except (BlockingIOError, OSError):
                    break
                seq.extend(b)
                if 0x40 <= b[0] <= 0x7E:
                    break
                if len(seq) >= 12:
                    break
            return "[" + seq.decode("utf-8", errors="ignore")
        if ch2 == b"O":
            try:
                ch3 = os.read(self.fd, 1)
            except (BlockingIOError, OSError):
                return "O"
            return "O" + ch3.decode("utf-8", errors="ignore")
        return ch2.decode("utf-8", errors="ignore")

    def _read_bracketed_paste(self) -> str:
        """Read until bracketed paste terminator (ESC [ 201 ~)."""
        terminator = b"\x1b[201~"
        buf = bytearray()
        while True:
            b = os.read(self.fd, 1)
            if not b:
                break
            buf.extend(b)
            if len(buf) >= len(terminator) and buf[-len(terminator) :] == terminator:
                content = buf[: -len(terminator)]
                return content.decode("utf-8", errors="ignore")
        return buf.decode("utf-8", errors="ignore")


class TerminalRegion:
    """Controls a region at the bottom of the terminal.

    Handles:
    - Reserving space (scrolling content up)
    - Rendering lines to the region
    - Clearing the region
    - Restoring when done
    """

    def __init__(self) -> None:
        self.num_lines = 0
        self._active = False

    def _write(self, s: str) -> None:
        """Write to stdout without newline."""
        ANSI._write(s)

    def activate(self, num_lines: int) -> None:
        """Reserve space at bottom of terminal."""
        self.num_lines = num_lines
        self._active = True

        # Hide terminal cursor (we render our own cursor glyph)
        ANSI.hide_cursor()
        # Clear current line and use it as region start
        ANSI.clear_line()
        ANSI.save_cursor()

    def render(self, lines: list[str]) -> None:
        """Render lines to the region."""
        if not self._active:
            return

        # Restore to region start
        ANSI.restore_cursor()

        # Render each line
        for i in range(self.num_lines):
            # Clear line and move to column 1
            ANSI.clear_line()
            if i < len(lines):
                self._write(lines[i])
            if i < self.num_lines - 1:
                # Move down one row and to column 1
                self._write("\n")
                self._write(ANSI.CARRIAGE_RETURN)

        # Position cursor at end of last line (for visual feedback)
        ANSI.restore_cursor()  # Back to region start
        if lines:
            last_line_idx = min(len(lines), self.num_lines) - 1
            if last_line_idx > 0:
                ANSI.move_down(last_line_idx)
            # Use visual length to account for ANSI escape codes
            visual_width = ANSI.visual_len(lines[last_line_idx])
            if visual_width > 0:
                ANSI.move_right(visual_width)

    def update_size(self, num_lines: int) -> None:
        """Resize the region (add more lines if needed)."""
        if num_lines > self.num_lines:
            # Need more space - add lines at bottom
            extra = num_lines - self.num_lines

            # Move to end of current region
            ANSI.restore_cursor()
            ANSI.move_down(self.num_lines)

            # Add new lines (scrolls up)
            self._write("\n" * extra)

            # Move back to start and update saved position
            ANSI.move_up(num_lines)
            ANSI.save_cursor()

            self.num_lines = num_lines

    def deactivate(self) -> None:
        """Release the region and clean up."""
        if not self._active:
            return

        # Clear the region lines so prompts don't linger in scrollback
        ANSI.restore_cursor()
        for i in range(self.num_lines):
            ANSI.clear_line()
            if i < self.num_lines - 1:
                ANSI.move_down(1)
                self._write(ANSI.CARRIAGE_RETURN)

        # Return cursor to region start (cleared) without adding new lines
        ANSI.restore_cursor()
        ANSI.clear_line()

        # Show terminal cursor again
        ANSI.show_cursor()

        self._active = False
        self.num_lines = 0
