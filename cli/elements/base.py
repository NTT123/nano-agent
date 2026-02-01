"""Base classes for interactive UI elements.

This module provides the core abstractions:
- InputEvent: Keyboard input event
- ActiveElement: Protocol for interactive elements with exclusive I/O control
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class InputEvent:
    """A keyboard input event."""

    key: str  # 'Enter', 'Escape', 'Backspace', 'Up', 'Down', or the character
    char: str | None  # Printable character or None
    ctrl: bool = False


class ActiveElement(ABC, Generic[T]):
    """An interactive UI element with exclusive output/input control.

    Lifecycle for standard elements:
        1. on_activate() - setup
        2. get_lines() -> render
        3. handle_input() -> process key, return (done, result)
        4. on_deactivate() - cleanup

    Self-managed elements (is_self_managed=True) handle their own I/O
    via run_async() instead of get_lines/handle_input.
    """

    def is_self_managed(self) -> bool:
        """Return True if element handles its own I/O."""
        return False

    async def run_async(self) -> T:
        """Run self-managed element. Override for custom I/O handling."""
        raise NotImplementedError("Self-managed elements must implement run_async()")

    @abstractmethod
    def get_lines(self) -> list[str]:
        """Return lines to render in the region."""
        ...

    @abstractmethod
    def handle_input(self, event: InputEvent) -> tuple[bool, T | None]:
        """Handle input event.

        Returns:
            (done, result) - if done=True, element completes with result
        """
        ...

    def on_activate(self) -> None:
        """Called when element becomes active."""
        pass

    def on_deactivate(self) -> None:
        """Called when element completes."""
        pass

    def completion_delay(self) -> float:
        """Delay in seconds after completion before returning.

        Override in subclasses. Default is 0.15s for visual feedback
        on confirmations and menus. Text input should return 0.0.
        """
        return 0.15
