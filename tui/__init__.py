"""nano-agent Terminal UI (TUI) application.

A Rich-based terminal app for interacting with Claude:
- Rich markup styling for messages
- Tool call visualization with panels
- All built-in tools (Bash, Read, Edit, Write, Glob, Grep, etc.)
- Automatic authentication via Claude CLI (auto-captures on first run)

Usage:
    nano-cli

Features:
    - Uses terminal's native scrollback for history
    - Rich Live context for dynamic status display
    - prompt_toolkit for input with history
    - Ctrl+C to cancel running operations
    - Ctrl+D to exit
    - Lightweight, minimal dependencies
"""

from .app import SimpleTerminalApp, main

__all__ = ["SimpleTerminalApp", "main"]
