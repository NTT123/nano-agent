"""Input handling using prompt_toolkit.

This module provides escape key detection during async operations using
prompt_toolkit's input system, which handles all the terminal complexity:
- Terminal raw mode management
- Escape sequence parsing (distinguishes ESC from arrow keys)
- Cross-platform support

Example:
    handler = InputHandler(on_escape=lambda: cancel_token.cancel())

    async with handler:
        # During this block, Escape key is detected
        await some_long_operation()
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator

from prompt_toolkit.input import create_input
from prompt_toolkit.input.vt100 import Vt100Input
from prompt_toolkit.keys import Keys

if TYPE_CHECKING:
    from prompt_toolkit.input import Input


@dataclass
class InputHandler:
    """Handles escape key detection during async operations.

    Uses prompt_toolkit's input system which properly handles:
    - Terminal raw mode
    - Escape sequence parsing (arrows, function keys, etc.)
    - Cross-platform support

    Usage:
        handler = InputHandler(on_escape=my_callback)

        async with handler:
            # Escape key detection active
            await long_running_operation()
        # Terminal restored automatically

    Attributes:
        on_escape: Callback invoked when Escape key is pressed.
                   Should be a simple, fast function (e.g., setting a flag).
    """

    on_escape: Callable[[], None] | None = None

    _input: Input | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _prompting: bool = field(default=False, init=False, repr=False)
    _raw_mode_ctx: Any = field(default=None, init=False, repr=False)
    _attach_ctx: Any = field(default=None, init=False, repr=False)

    def _is_tty(self) -> bool:
        """Check if stdin is a real terminal."""
        try:
            return sys.stdin.isatty()
        except Exception:
            return False

    def _on_input_ready(self) -> None:
        """Called by prompt_toolkit when input is available."""
        if not self._input or not self._running:
            return

        # Don't consume input if prompt_yn() is active
        if self._prompting:
            return

        # Force flush the parser to reduce escape sequence disambiguation delay
        if isinstance(self._input, Vt100Input) and hasattr(self._input, "vt100_parser"):
            self._input.vt100_parser.flush()

        for key_press in self._input.read_keys():
            if key_press.key == Keys.Escape and self.on_escape:
                self.on_escape()

    async def prompt_yn(self, prompt: str = "") -> bool | None:
        """Prompt for yes/no/escape using single character input.

        This method reads single characters to get y/n/Escape input.
        It's designed for permission prompts during agent execution.

        Args:
            prompt: The prompt message (caller should print this before calling).

        Returns:
            True if user pressed 'y' or 'Y'
            False if user pressed 'n' or 'N'
            None if user pressed Escape (caller should handle as cancel)
        """
        if not self._running or not self._input:
            # Fallback for non-TTY
            try:
                response = input(prompt).strip().lower()
                return response in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                return None

        # Pause callback-based input processing while prompting
        self._prompting = True
        try:
            while True:
                # Brief sleep to yield to event loop and allow input to arrive
                await asyncio.sleep(0.02)  # 20ms for faster response

                # Force flush to reduce escape sequence disambiguation delay
                if isinstance(self._input, Vt100Input) and hasattr(self._input, "vt100_parser"):
                    self._input.vt100_parser.flush()

                for key_press in self._input.read_keys():
                    if key_press.key == Keys.Escape:
                        return None
                    # key_press.data contains the actual character for regular keys
                    char = key_press.data.lower() if key_press.data else ""
                    if char == "y":
                        return True
                    elif char == "n":
                        return False
                    # Ignore other keys, keep waiting
        finally:
            self._prompting = False

    async def start(self) -> bool:
        """Start listening for escape key.

        Returns:
            True if started successfully, False if not possible (not a TTY, etc.)
        """
        if self._running:
            return True

        if not self._is_tty():
            return False

        try:
            self._input = create_input()
            self._raw_mode_ctx = self._input.raw_mode()
            self._raw_mode_ctx.__enter__()
            self._attach_ctx = self._input.attach(self._on_input_ready)
            self._attach_ctx.__enter__()
            self._running = True
            return True
        except Exception:
            self._cleanup()
            return False

    async def stop(self) -> None:
        """Stop listening and restore terminal."""
        self._running = False
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up contexts in reverse order."""
        if self._attach_ctx:
            try:
                self._attach_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._attach_ctx = None

        if self._raw_mode_ctx:
            try:
                self._raw_mode_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._raw_mode_ctx = None

        if self._input:
            try:
                self._input.close()
            except Exception:
                pass
            self._input = None

    async def __aenter__(self) -> "InputHandler":
        """Async context manager entry - starts listening."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - always restores terminal."""
        await self.stop()
