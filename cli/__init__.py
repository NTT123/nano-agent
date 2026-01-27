"""nano-agent Terminal UI (TUI) application.

A message-list based terminal app for interacting with Claude:
- Message-list architecture: UI is a sequential list of messages (source of truth)
- Each message owns its output buffer (can only modify its own section)
- The last message has exclusive control over input events
- Re-rendering is deterministic: clear screen -> render all messages in order

Usage:
    nano-cli

Features:
    - Rich Console for styled output
    - Rich Live for spinner while waiting for API response
    - prompt_toolkit for input with history
    - Escape key to cancel running operations
    - Ctrl+D to exit
    - All built-in tools (Bash, Read, Edit, Write, Glob, Grep, etc.)
    - Automatic authentication via Claude CLI
"""

from .app import TerminalApp, main
from .input_handler import InputHandler
from .input_handlers import ConfirmationHandler, SelectionHandler, TextInputHandler
from .message_factory import (
    add_text_to_assistant,
    add_thinking_to_assistant,
    create_assistant_message,
    create_error_message,
    create_input_prompt_message,
    create_permission_message,
    create_system_message,
    create_tool_call_message,
    create_tool_result_message,
    create_user_message,
    create_welcome_message,
)
from .message_list import MessageList
from .messages import InputEvent, MessageStatus, RenderItem, UIMessage

__all__ = [
    # Main app
    "TerminalApp",
    "main",
    # Input handlers
    "InputHandler",
    "TextInputHandler",
    "ConfirmationHandler",
    "SelectionHandler",
    # Message types
    "UIMessage",
    "RenderItem",
    "MessageStatus",
    "InputEvent",
    "MessageList",
    # Message factory functions
    "create_welcome_message",
    "create_system_message",
    "create_user_message",
    "create_assistant_message",
    "add_thinking_to_assistant",
    "add_text_to_assistant",
    "create_tool_call_message",
    "create_tool_result_message",
    "create_error_message",
    "create_permission_message",
    "create_input_prompt_message",
]
