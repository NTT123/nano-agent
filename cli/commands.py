"""Command routing for slash commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from rich.console import RenderableType
from rich.json import JSON

from nano_agent import DAG, Tool

from . import display
from .message_factory import create_error_message, create_system_message
from .messages import UIMessage
from .session import SessionStore


@dataclass
class CommandContext:
    """Dependencies and callbacks used by CommandRouter."""

    dag: DAG | None
    session_store: SessionStore
    tools: list[Tool]
    render_history: Callable[[], None]
    clear_and_reset: Callable[[], Awaitable[bool]]
    refresh_token: Callable[[], Awaitable[bool]]
    set_color_scheme: Callable[[str], bool]


class CommandRouter:
    """Parse and execute slash commands."""

    HELP_TEXT = """Commands:
  /quit, /exit, /q - Exit the application
  /clear - Reset conversation and clear screen
  /continue, /c - Continue agent execution without user message
  /renew - Refresh OAuth token (use when getting 401 errors)
  /render - Re-render history (useful after terminal resize)
  /theme [dark|light|auto] - Set UI color scheme (default: dark)
  /theme status - Show current scheme and auto-detection details
  /debug - Show DAG as JSON
  /save [filename] - Save session to file (default: session.json)
  /load [filename] - Load session from file (default: session.json)
  /help - Show this help message

Input:
  Enter - Send message
  \\ + Enter - Insert new line (for multiline input)
  Shift+Enter - Insert new line (prompt_toolkit only)
  Shift+Tab - Toggle auto-accept mode (auto-approve edit confirmations)
  Esc - Cancel input or current operation (during execution)
  Ctrl+D - Exit"""

    async def handle(
        self, command: str, ctx: CommandContext
    ) -> tuple[bool, list[UIMessage | RenderableType]]:
        """Handle a slash command.

        Returns:
            (should_continue, messages_to_add)
        """
        cmd = command.lower().strip()
        messages: list[UIMessage | RenderableType] = []

        if cmd in ("/quit", "/exit", "/q"):
            return False, messages

        if cmd == "/clear":
            await ctx.clear_and_reset()
            messages.append(create_system_message("Conversation reset."))
            return True, messages

        if cmd == "/render":
            ctx.render_history()
            return True, messages

        if cmd == "/help":
            messages.append(create_system_message(self.HELP_TEXT))
            return True, messages

        if cmd == "/debug":
            if ctx.dag and ctx.dag._heads:
                all_nodes: dict[str, object] = {}
                for node in ctx.dag.head.ancestors():
                    all_nodes[node.id] = node.to_dict()
                graph_data = {
                    "head_ids": [h.id for h in ctx.dag._heads],
                    "nodes": all_nodes,
                }
                dag_json = json.dumps(graph_data, indent=2, default=str)
                messages.append(JSON(dag_json))
            else:
                messages.append(create_system_message("No DAG initialized."))
            return True, messages

        if cmd.startswith("/theme"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) == 1:
                messages.append(
                    create_system_message(
                        "Usage: /theme [dark|light|auto] (example: /theme light)"
                    )
                )
                return True, messages
            scheme = parts[1].strip().lower()
            if scheme == "status":
                info = display.get_last_auto_detection()
                details = ", ".join(f"{k}={v}" for k, v in info.items())
                details_text = f"\nAuto details: {details}" if details else ""
                messages.append(
                    create_system_message(
                        f"Color scheme: {display.get_color_scheme()}{details_text}"
                    )
                )
                return True, messages
            if ctx.set_color_scheme(scheme):
                ctx.render_history()
                if scheme == "auto":
                    info = display.get_last_auto_detection()
                    details = ", ".join(f"{k}={v}" for k, v in info.items())
                    details_text = f" ({details})" if details else ""
                    messages.append(
                        create_system_message(
                            f"Color scheme set to: auto → {display.get_color_scheme()}{details_text}"
                        )
                    )
                else:
                    messages.append(
                        create_system_message(f"Color scheme set to: {scheme}")
                    )
            else:
                messages.append(
                    create_error_message(
                        "Unknown theme. Use: /theme dark, /theme light, or /theme auto"
                    )
                )
            return True, messages

        if cmd.startswith("/save"):
            parts = command.strip().split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session.json"
            try:
                filepath = await ctx.session_store.save(ctx.dag, filename)
                if filepath is None:
                    messages.append(create_system_message("No DAG to save."))
                else:
                    messages.append(
                        create_system_message(f"Session saved to: {filepath}")
                    )
            except Exception as e:
                messages.append(create_error_message(f"Failed to save: {e}"))
            return True, messages

        if cmd.startswith("/load"):
            parts = command.strip().split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session.json"
            try:
                loaded_dag, metadata, filepath = await ctx.session_store.load(filename)
                if ctx.tools:
                    loaded_dag = loaded_dag.tools(*ctx.tools)
                ctx.dag = loaded_dag
                session_id = metadata.get("session_id", "unknown")
                node_count = len(metadata.get("nodes", {}))
                ctx.render_history()
                messages.append(
                    create_system_message(
                        f"Session loaded from: {filepath}\n"
                        f"  Session ID: {session_id}\n"
                        f"  Nodes: {node_count}"
                    )
                )
            except Exception as e:
                messages.append(create_error_message(f"Failed to load: {e}"))
            return True, messages

        if cmd == "/renew":
            await ctx.refresh_token()
            return True, messages

        messages.append(create_error_message(f"Unknown command: {command}"))
        messages.append(create_system_message("Type /help for available commands."))
        return True, messages
