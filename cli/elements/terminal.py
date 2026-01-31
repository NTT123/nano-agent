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
import select
import sys
import termios
import time
import tty
from typing import Any

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

    # Screen control
    CLEAR_SCREEN = "\033[2J"  # Clear entire screen
    CLEAR_SCROLLBACK = "\033[3J"  # Clear scrollback buffer
    MOVE_HOME = "\033[H"  # Move cursor to home position (1,1)

    # iTerm2/WezTerm bookmark sequences (OSC 1337)
    SET_MARK = "\033]1337;SetMark\a"  # Mark current cursor position
    CLEAR_TO_MARK = "\033]1337;ClearToMark\a"  # Clear from mark to buffer end

    # Pattern to match ANSI escape sequences (for stripping)
    _ANSI_PATTERN = re.compile(r"\033\[[0-9;]*m")

    # Terminal capability detection (cached)
    _osc_1337_supported: bool | None = None
    _osc_1337_detection_done: bool = False

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

    @classmethod
    def _read_cursor_position(cls, timeout: float = 0.1) -> tuple[int, int] | None:
        """Read cursor position from CPR response (ESC[row;colR)."""
        response = ""
        deadline = time.time() + timeout
        fd = sys.stdin.fileno()

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if select.select([fd], [], [], min(0.01, remaining))[0]:
                try:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                except (BlockingIOError, OSError):
                    break
                if not ch:
                    break
                response += ch
                if ch == "R":
                    match = re.match(r".*\033\[(\d+);(\d+)R", response)
                    if match:
                        return (int(match.group(1)), int(match.group(2)))
        return None

    @classmethod
    def detect_osc_1337_support(cls) -> bool:
        """Detect OSC 1337 support by testing if ClearToMark works.

        Tests by checking if cursor position changes after SetMark/ClearToMark.
        Result is cached. Safe to call multiple times.
        """
        if cls._osc_1337_detection_done:
            return cls._osc_1337_supported or False

        cls._osc_1337_detection_done = True
        cls._osc_1337_supported = False

        # Must be a TTY
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False

        # Save terminal state
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except termios.error:
            return False
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)

        try:
            # Set raw mode for reading response
            tty.setraw(fd)
            attrs = termios.tcgetattr(fd)
            attrs[1] |= termios.OPOST | termios.ONLCR
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)

            # Flush any pending input
            termios.tcflush(fd, termios.TCIFLUSH)

            # Query initial cursor position
            sys.stdout.write("\033[6n")
            sys.stdout.flush()

            initial_pos = cls._read_cursor_position(0.1)
            if not initial_pos:
                return False

            # Set mark, write test char, clear to mark
            sys.stdout.write(cls.SET_MARK)
            sys.stdout.write("X")  # Test character
            sys.stdout.flush()

            sys.stdout.write(cls.CLEAR_TO_MARK)
            sys.stdout.flush()

            # Query final cursor position
            termios.tcflush(fd, termios.TCIFLUSH)
            sys.stdout.write("\033[6n")
            sys.stdout.flush()

            final_pos = cls._read_cursor_position(0.1)
            if not final_pos:
                # Clean up test character if ClearToMark failed
                sys.stdout.write("\b \b")
                sys.stdout.flush()
                return False

            # If cursor moved back, OSC 1337 worked
            if final_pos[1] <= initial_pos[1]:
                cls._osc_1337_supported = True
                return True

            # ClearToMark didn't work - clean up test character
            sys.stdout.write("\b \b")
            sys.stdout.flush()
            return False

        except Exception:
            return False
        finally:
            # Restore terminal state
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    @classmethod
    def supports_osc_1337(cls) -> bool:
        """Check if terminal supports OSC 1337 (cached result)."""
        if not cls._osc_1337_detection_done:
            cls.detect_osc_1337_support()
        return cls._osc_1337_supported or False

    @classmethod
    def clear_screen_and_scrollback(cls) -> None:
        """Clear entire screen and scrollback buffer, move cursor home."""
        cls._write(cls.CLEAR_SCREEN + cls.CLEAR_SCROLLBACK + cls.MOVE_HOME)

    @classmethod
    def set_mark(cls) -> None:
        """Set a bookmark at the current cursor position (if supported).

        Uses iTerm2/WezTerm OSC 1337 SetMark sequence. The bookmark persists
        even if content scrolls into the scrollback buffer. Subsequent calls
        overwrite the previous bookmark (only one bookmark at a time).

        On unsupported terminals, this is a no-op.
        """
        if cls.supports_osc_1337():
            cls._write(cls.SET_MARK)

    @classmethod
    def clear_to_mark(cls) -> None:
        """Clear from the bookmark to the end of buffer, or clear screen if unsupported.

        Uses iTerm2/WezTerm OSC 1337 ClearToMark sequence. If no bookmark was
        set, this has no effect. If the bookmark was purged from scrollback,
        this clears all content.

        On unsupported terminals, falls back to clear_screen_and_scrollback().
        """
        if cls.supports_osc_1337():
            cls._write(cls.CLEAR_TO_MARK)
        else:
            cls.clear_screen_and_scrollback()


class RawInputReader:
    """Reads single keystrokes from terminal in raw mode."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old_settings: list[Any] | None = None

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

    Uses explicit cursor movement (cursor_up) instead of save/restore
    sequences, which are more reliable across different terminals.
    """

    def __init__(self) -> None:
        self.num_lines = 0
        self._active = False
        self._cursor_at_line = 0  # Track which line cursor is on (0 = top of region)

    def _write(self, s: str) -> None:
        """Write to stdout without newline."""
        ANSI._write(s)

    def _move_to_region_start(self) -> None:
        """Move cursor to the start of the region (top-left)."""
        if self._cursor_at_line > 0:
            ANSI.move_up(self._cursor_at_line)
        self._write(ANSI.CARRIAGE_RETURN)
        self._cursor_at_line = 0

    def activate(self, num_lines: int) -> None:
        """Reserve space at bottom of terminal."""
        self.num_lines = num_lines
        self._active = True
        self._cursor_at_line = 0

        # Hide terminal cursor (we render our own cursor glyph)
        ANSI.hide_cursor()
        # Clear current line and use it as region start
        ANSI.clear_line()

    def render(self, lines: list[str]) -> None:
        """Render lines to the region."""
        if not self._active:
            return

        # Move to region start
        self._move_to_region_start()

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
                self._cursor_at_line = i + 1

        # Position cursor at end of last line (for visual feedback)
        self._move_to_region_start()
        if lines:
            last_line_idx = min(len(lines), self.num_lines) - 1
            if last_line_idx > 0:
                ANSI.move_down(last_line_idx)
                self._cursor_at_line = last_line_idx
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
            lines_to_move = self.num_lines - self._cursor_at_line
            if lines_to_move > 0:
                ANSI.move_down(lines_to_move)
                self._cursor_at_line = self.num_lines

            # Add new lines (scrolls up)
            self._write("\n" * extra)
            self._cursor_at_line += extra

            # Move back to start
            ANSI.move_up(num_lines)
            self._write(ANSI.CARRIAGE_RETURN)
            self._cursor_at_line = 0

            self.num_lines = num_lines

    def deactivate(self) -> None:
        """Release the region and clean up."""
        if not self._active:
            return

        # Clear the region lines so prompts don't linger in scrollback
        self._move_to_region_start()
        for i in range(self.num_lines):
            ANSI.clear_line()
            if i < self.num_lines - 1:
                ANSI.move_down(1)
                self._write(ANSI.CARRIAGE_RETURN)
                self._cursor_at_line = i + 1

        # Return cursor to region start (cleared) without adding new lines
        self._move_to_region_start()
        ANSI.clear_line()

        # Show terminal cursor again
        ANSI.show_cursor()

        self._active = False
        self.num_lines = 0
        self._cursor_at_line = 0
