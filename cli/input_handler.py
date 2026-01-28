"""Input handling using raw terminal input.

This module provides escape key detection during async operations without
prompt_toolkit. It uses the same RawInputReader as the active-element system.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .elements.terminal import RawInputReader


@dataclass
class InputHandler:
    """Handles escape key detection during async operations.

    Usage:
        handler = InputHandler(on_escape=my_callback)

        async with handler:
            await long_running_operation()
    """

    on_escape: Callable[[], None] | None = None

    _reader: RawInputReader | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def _is_tty(self) -> bool:
        """Check if stdin is a real terminal."""
        try:
            return sys.stdin.isatty()
        except Exception:
            return False

    def is_running(self) -> bool:
        """Return True if the handler is currently running."""
        return self._running

    async def start(self) -> bool:
        """Start listening for escape key."""
        if self._running:
            return True
        if not self._is_tty():
            return False
        try:
            self._reader = RawInputReader()
            self._reader.start()
            self._running = True
            return True
        except Exception:
            await self.stop()
            return False

    async def stop(self) -> None:
        """Stop listening and restore terminal."""
        self._running = False
        if self._reader:
            try:
                self._reader.stop()
            finally:
                self._reader = None

    async def poll_escape(self, timeout: float = 0.02) -> None:
        """Poll for Escape and invoke callback if pressed."""
        if not self._running or not self._reader:
            return
        loop = asyncio.get_event_loop()
        event = await loop.run_in_executor(
            None, self._reader.read_nonblocking, timeout
        )
        if event and event.key == "Escape" and self.on_escape:
            self.on_escape()

    async def __aenter__(self) -> "InputHandler":
        """Async context manager entry - starts listening."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - always restores terminal."""
        await self.stop()
