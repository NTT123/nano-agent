"""Message data structures for the TUI message list.

This module provides the core data structures for the message-list based TUI model:
- UIMessage: A message in the UI message list with its own output buffer
- RenderItem: A single renderable item in a message's output buffer
- MessageStatus: Status of a message (pending, active, complete, error)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from rich.console import RenderableType


class MessageStatus(Enum):
    """Status of a UIMessage."""

    PENDING = "pending"  # Still being generated/streamed
    ACTIVE = "active"  # Last message, has input control
    COMPLETE = "complete"  # Frozen, cannot change
    ERROR = "error"


@dataclass
class InputEvent:
    """Keyboard event passed to active message."""

    key: str
    ctrl: bool = False
    alt: bool = False


class InputHandler(Protocol):
    """Protocol for handling input in active message."""

    async def handle_key(self, event: InputEvent) -> bool:
        """Handle key event. Return True if consumed."""
        ...

    def get_prompt_text(self) -> str:
        """Return current input buffer for display."""
        ...


@dataclass
class RenderItem:
    """A single renderable item in a message's output buffer.

    Attributes:
        content: The renderable content (string or Rich renderable)
        style: Rich style string (only used if content is string)
        is_transient: If True, can be replaced (spinners, progress)
    """

    content: str | RenderableType
    style: str = ""
    is_transient: bool = False


@dataclass
class UIMessage:
    """A message in the UI message list.

    Each message owns its output buffer and can only modify its own section.
    Only the last (active) message can receive input events.

    Attributes:
        id: Unique identifier for this message
        message_type: Type of message (welcome, user, assistant, tool, error, etc.)
        output_buffer: List of render items for this message
        input_handler: Handler for input events (only active for last message)
        status: Current status of this message
        metadata: Additional metadata (tokens, timing, etc.)
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    message_type: str = ""

    # Output buffer - list of render items
    output_buffer: list[RenderItem] = field(default_factory=list)

    # Input handling (only active for last message)
    input_handler: InputHandler | None = None

    # Status
    status: MessageStatus = MessageStatus.PENDING

    # Metadata (tokens, timing, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, content: str | RenderableType, style: str = "") -> None:
        """Add content to this message's output buffer.

        Args:
            content: The text or Rich renderable to add
            style: Rich style string (only used if content is string)
        """
        self.output_buffer.append(RenderItem(content=content, style=style))

    def append_newline(self) -> None:
        """Add a blank line to the output buffer."""
        self.output_buffer.append(RenderItem(content=""))

    def set_transient(self, content: str | RenderableType, style: str = "") -> None:
        """Set transient content (replaces previous transient).

        Transient content is used for things like spinners that should be
        replaced on the next update rather than accumulated.

        Args:
            content: The text or Rich renderable to show
            style: Rich style string (only used if content is string)
        """
        # Remove previous transient items
        self.output_buffer = [r for r in self.output_buffer if not r.is_transient]
        self.output_buffer.append(
            RenderItem(content=content, style=style, is_transient=True)
        )

    def clear_transient(self) -> None:
        """Remove transient items (spinners, etc.)."""
        self.output_buffer = [r for r in self.output_buffer if not r.is_transient]

    def freeze(self) -> None:
        """Freeze this message (mark as complete, remove input handler)."""
        self.status = MessageStatus.COMPLETE
        self.input_handler = None
        self.clear_transient()

    def is_frozen(self) -> bool:
        """Check if this message is frozen (cannot be modified)."""
        return self.status == MessageStatus.COMPLETE
