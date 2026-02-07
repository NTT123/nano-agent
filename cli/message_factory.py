"""Factory functions for creating UIMessage instances.

This module provides factory functions to create UIMessage instances for
different message types (welcome, user, assistant, tool, error, etc.).
"""

from __future__ import annotations

from typing import Any

from rich.text import Text

from . import display
from .messages import MessageStatus, RenderItem, UIMessage
from .rendering import MessageRenderer

renderer = MessageRenderer()


def create_welcome_message() -> UIMessage:
    """Create a welcome message displayed at app startup.

    Returns:
        UIMessage configured as a welcome message
    """
    msg = UIMessage(message_type="welcome")
    msg.append(Text("nano-cli", style="bold cyan"))
    msg.append(
        Text(
            "Type your message. /help for commands. Shift+Tab toggles auto-accept. Ctrl+D to exit.",
            style="dim",
        )
    )
    msg.status = MessageStatus.COMPLETE
    return msg


def create_system_message(text: str) -> UIMessage:
    """Create a system message (status updates, info).

    Args:
        text: The system message text

    Returns:
        UIMessage configured as a system message
    """
    msg = UIMessage(message_type="system")
    msg.append(renderer.system(text))
    msg.status = MessageStatus.COMPLETE
    return msg


def create_user_message(text: str) -> UIMessage:
    """Create a user input message.

    Args:
        text: The user's input text

    Returns:
        UIMessage configured as a user message
    """
    msg = UIMessage(message_type="user")
    msg.append(renderer.user(text))
    msg.append_newline()
    msg.status = MessageStatus.COMPLETE
    return msg


def create_command_message(text: str) -> UIMessage:
    """Create a command input message (display-only, not sent to model).

    Args:
        text: The command text (e.g. "/help", "/save foo.json")

    Returns:
        UIMessage configured as a command message
    """
    msg = UIMessage(message_type="command")
    msg.append(renderer.user(text))
    msg.append_newline()
    msg.status = MessageStatus.COMPLETE
    return msg


def create_assistant_message() -> UIMessage:
    """Create an assistant response message (initially empty for streaming).

    Returns:
        UIMessage configured as an assistant message
    """
    msg = UIMessage(message_type="assistant")
    msg.status = MessageStatus.PENDING
    return msg


def add_thinking_to_assistant(msg: UIMessage, thinking: str) -> None:
    """Add thinking content to an assistant message.

    Args:
        msg: The assistant message to add thinking to
        thinking: The thinking content text
    """
    if thinking and thinking.strip():
        msg.append(renderer.thinking(thinking))
        msg.append_newline()


def add_text_to_assistant(
    msg: UIMessage,
    text: str,
    has_thinking: bool = False,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    cost: float = 0.0,
) -> None:
    """Add text response to an assistant message.

    Args:
        msg: The assistant message to add text to
        text: The response text
        has_thinking: Whether thinking was already added (adds separator)
        input_tokens: Input token count
        output_tokens: Output token count
        cache_creation_tokens: Cache creation token count
        cache_read_tokens: Cache read token count
        reasoning_tokens: Reasoning/thinking token count
        cached_tokens: Cached input token count
        cost: Cost in dollars
    """
    content_added = False
    if text and text.strip():
        if has_thinking:
            msg.append(renderer.thinking_separator())
            msg.append_newline()
        msg.append(renderer.assistant(text))
        content_added = True

    if input_tokens or output_tokens or cache_creation_tokens or cache_read_tokens:
        msg.append(
            renderer.token_count(
                input_tokens,
                output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                cost=cost,
            )
        )
        content_added = True

    if content_added:
        msg.append_newline()


def create_tool_call_message(name: str, params: dict[str, Any]) -> UIMessage:
    """Create a tool call message.

    Args:
        name: The tool name
        params: The tool parameters

    Returns:
        UIMessage configured as a tool call message
    """
    msg = UIMessage(message_type="tool_call")
    msg.append(renderer.tool_call(name, params))
    msg.status = MessageStatus.PENDING
    return msg


def create_tool_result_message(result: str, is_error: bool = False) -> UIMessage:
    """Create a tool result message.

    Args:
        result: The tool result text
        is_error: Whether the result is an error

    Returns:
        UIMessage configured as a tool result message
    """
    msg = UIMessage(message_type="tool_result")
    msg.append(renderer.tool_result(result, is_error=is_error))
    msg.status = MessageStatus.COMPLETE
    return msg


def create_question_message(text: str) -> UIMessage:
    """Create a question message for user prompts.

    Args:
        text: The question text

    Returns:
        UIMessage configured as a question message
    """
    msg = UIMessage(message_type="question")
    msg.append(renderer.tool_result(text))
    msg.status = MessageStatus.COMPLETE
    return msg


def create_error_message(text: str) -> UIMessage:
    """Create an error message.

    Args:
        text: The error message text

    Returns:
        UIMessage configured as an error message
    """
    msg = UIMessage(message_type="error", status=MessageStatus.ERROR)
    msg.append(renderer.error(text))
    return msg


def create_permission_message(
    file_path: str, match_count: int = 1, preview: str = ""
) -> UIMessage:
    """Create a permission request message for Edit tool.

    Args:
        file_path: The file being edited
        match_count: Number of occurrences being replaced
        preview: The diff preview text to display

    Returns:
        UIMessage configured as a permission message
    """
    palette = display.get_palette()
    msg = UIMessage(message_type="permission")
    msg.append_newline()
    msg.append(Text("--- Permission Required ---", style=palette.permission_header))
    msg.append(Text(f"File: {file_path}", style=palette.permission_file))
    if match_count > 1:
        msg.append(
            Text(
                f"(Replacing {match_count} occurrences)", style=palette.permission_note
            )
        )
    msg.append(Text("Confirm: press y/n/Esc", style=palette.permission_prompt))
    msg.append_newline()

    # Display the diff preview with background colors
    if preview:
        for line in preview.splitlines():
            if line.startswith("  -"):
                # Removed line - dark red background
                msg.append(Text(line, style=palette.permission_preview_remove))
            elif line.startswith("  +"):
                # Added line - dark green background
                msg.append(Text(line, style=palette.permission_preview_add))
            elif line.startswith("───"):
                # Header line
                msg.append(Text(line, style=palette.permission_preview_header))
            else:
                # Context line or other
                msg.append(Text(line, style=palette.permission_preview_context))
        msg.append_newline()

    msg.status = MessageStatus.ACTIVE
    return msg
