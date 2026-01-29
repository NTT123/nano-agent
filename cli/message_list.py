"""Message list manager for the TUI.

This module provides the MessageList class that manages the UI message list.
The message list is the source of truth for UI rendering:
- Each message owns its output buffer
- Only the last message can receive input events
- Re-rendering is deterministic: render all messages in order
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from rich.console import Console, RenderableType
from rich.text import Text

from .messages import MessageStatus, RenderItem, UIMessage


@dataclass
class MessageList:
    """Manages the UI message list.

    The message list is the source of truth for rendering. Each message
    owns its output buffer, and only the last (active) message can
    modify itself or receive input events.

    Attributes:
        messages: List of UIMessage objects in display order
    """

    messages: list[UIMessage] = field(default_factory=list)

    def add(self, msg: UIMessage) -> UIMessage:
        """Add a message, freezing the previous last message.

        Args:
            msg: The message to add

        Returns:
            The added message (for chaining)
        """
        if self.messages:
            self.messages[-1].freeze()
        msg.status = MessageStatus.ACTIVE
        self.messages.append(msg)
        return msg

    def get_active(self) -> UIMessage | None:
        """Get the last (active) message.

        Returns:
            The active message, or None if the list is empty
        """
        return self.messages[-1] if self.messages else None

    def get_by_id(self, msg_id: str) -> UIMessage | None:
        """Get a message by its ID.

        Args:
            msg_id: The message ID to find

        Returns:
            The message, or None if not found
        """
        for msg in self.messages:
            if msg.id == msg_id:
                return msg
        return None

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def render_item(self, item: RenderItem) -> RenderableType:
        """Convert a RenderItem to a Rich renderable.

        Args:
            item: The render item to convert

        Returns:
            A Rich renderable object
        """
        if isinstance(item.content, str):
            if item.style:
                return Text(item.content, style=item.style)
            return Text(item.content)
        return item.content

    def render_message(self, msg: UIMessage, console: Console) -> None:
        """Render a single message to the console.

        Args:
            msg: The message to render
            console: The Rich console to render to
        """
        for item in msg.output_buffer:
            # Ensure we start at column 1 (avoid inherited cursor offsets)
            console.print("\r", end="")
            renderable = self.render_item(item)
            console.print(renderable)

    def render_all(self, console: Console) -> None:
        """Render all messages to the console.

        Args:
            console: The Rich console to render to
        """
        for msg in self.messages:
            self.render_message(msg, console)

    def full_redraw(self, console: Console) -> None:
        """Clear screen and re-render everything.

        Args:
            console: The Rich console to render to
        """
        console.clear()
        self.render_all(console)

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __iter__(self) -> "Iterator[UIMessage]":
        """Iterate over messages."""
        return iter(self.messages)
