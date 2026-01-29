"""Element manager for coordinating interactive UI elements.

This module provides ElementManager which:
- Coordinates active elements with terminal I/O
- Ensures only one element is active at a time
- Handles both standard and self-managed elements
"""

from __future__ import annotations

import asyncio
from typing import TypeVar

from .base import ActiveElement
from .terminal import RawInputReader, TerminalRegion

T = TypeVar("T")


class ElementManager:
    """Coordinates active elements with terminal I/O.

    Only one element can be active at a time. Supports two types of elements:

    1. Standard elements: Manager controls terminal region and input handling
       - get_lines() for rendering
       - handle_input() for keystroke processing

    2. Self-managed elements: Element handles its own I/O
       - run_async() for custom I/O handling

    TTY resources (TerminalRegion, RawInputReader) are lazily initialized on
    first use to allow instantiation in non-TTY environments (e.g., tests).
    """

    def __init__(self) -> None:
        self._region: TerminalRegion | None = None
        self._input: RawInputReader | None = None
        self._active: ActiveElement | None = None

    def _ensure_initialized(self) -> tuple[TerminalRegion, RawInputReader]:
        """Lazily initialize TTY resources on first use."""
        if self._region is None:
            self._region = TerminalRegion()
        if self._input is None:
            self._input = RawInputReader()
        return self._region, self._input

    async def run(self, element: ActiveElement[T]) -> T:
        """Run an element until it returns a result."""
        if self._active:
            raise RuntimeError("Another element is already active")

        self._active = element
        element.on_activate()

        # Self-managed elements handle their own I/O
        if element.is_self_managed():
            try:
                return await element.run_async()
            finally:
                element.on_deactivate()
                self._active = None

        # Standard elements use our terminal region and input handling
        region, input_reader = self._ensure_initialized()
        input_reader.start()

        try:
            # Initial render
            lines = element.get_lines()
            region.activate(len(lines))
            region.render(lines)

            # Flush any input that arrived during rendering
            input_reader.flush()

            # Input loop
            while True:
                event = await input_reader.read()
                done, result = element.handle_input(event)

                # Re-render (show response or state change)
                lines = element.get_lines()
                if len(lines) > region.num_lines:
                    region.update_size(len(lines))
                region.render(lines)

                if done:
                    # Brief pause so user can see their response
                    await asyncio.sleep(0.15)
                    return result

        finally:
            input_reader.stop()
            element.on_deactivate()
            region.deactivate()
            self._active = None
