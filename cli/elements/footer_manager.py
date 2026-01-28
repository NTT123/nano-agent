"""Element manager that coordinates elements with TerminalFooter.

This module provides FooterElementManager which:
- Renders element content through the footer's content area
- Keeps status bar visible while elements are active
- Handles input loop for non-self-managed elements
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from .base import ActiveElement
from .footer import TerminalFooter
from .terminal import RawInputReader

T = TypeVar("T")


class FooterElementManager:
    """Coordinates active elements with TerminalFooter rendering.

    Unlike the standard ElementManager, this manager:
    - Renders element.get_lines() via footer.set_content()
    - Keeps the status bar visible during element interaction
    - Clears content (but keeps status bar) when element completes
    """

    def __init__(self, footer: TerminalFooter) -> None:
        self._footer = footer
        self._input: RawInputReader | None = None
        self._active: ActiveElement[Any] | None = None

    def _ensure_input(self) -> RawInputReader:
        """Lazily initialize input reader."""
        if self._input is None:
            self._input = RawInputReader()
        return self._input

    async def run(self, element: ActiveElement[T]) -> T:
        """Run an element until it returns a result.

        The element's content is rendered in the footer's content area,
        while the status bar remains visible below.
        """
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

        # Standard elements use our footer and input handling
        input_reader = self._ensure_input()
        input_reader.start()

        try:
            # Ensure footer is active
            if not self._footer.is_active():
                self._footer.activate()

            # Initial render - set content lines
            lines = element.get_lines()
            self._footer.set_content(lines)

            # Flush any input that arrived during rendering
            input_reader.flush()

            # Input loop
            while True:
                event = await input_reader.read()
                done, result = element.handle_input(event)

                # Re-render content
                lines = element.get_lines()
                self._footer.set_content(lines)

                if done:
                    # Brief pause so user can see their response
                    await asyncio.sleep(0.15)
                    return result  # type: ignore[return-value]

        finally:
            input_reader.stop()
            element.on_deactivate()
            # Clear content but keep status bar visible
            self._footer.clear_content()
            self._active = None
