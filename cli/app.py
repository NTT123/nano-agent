"""Message-list based TUI application for nano-agent.

This module provides a terminal app with a message-list architecture:
- The UI is a sequential list of messages (source of truth for rendering)
- Each message owns its output buffer (can only modify its own section)
- The last message has exclusive control over input events
- Re-rendering is deterministic: clear screen -> render all messages in order

Architecture:
    MessageList (source of truth for UI)
    ├── WelcomeMessage (frozen)
    ├── UserMessage (frozen)
    ├── AssistantMessage (frozen)
    │   ├── thinking content
    │   ├── text response
    │   └── token count
    ├── ToolCallMessage (frozen)
    ├── ToolResultMessage (frozen)
    └── ActiveMessage (can re-render, has input control)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.vt100 import Vt100Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.console import Console, Group, RenderableType
from rich.json import JSON
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.theme import Theme

from nano_agent import (
    DAG,
    ClaudeCodeAPI,
    GeminiAPI,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from nano_agent.api_base import APIError
from nano_agent.cancellation import CancellationToken
from nano_agent.capture_claude_code_auth import get_config
from nano_agent.tools import (
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    PythonTool,
    ReadTool,
    TodoWriteTool,
    Tool,
    WebFetchTool,
    WriteTool,
)

from .display import (
    format_assistant_message,
    format_error_message,
    format_system_message,
    format_thinking_message,
    format_thinking_separator,
    format_token_count,
    format_tool_call,
    format_tool_result,
    format_user_message,
)
from .input_handler import InputHandler
from .message_factory import (
    add_text_to_assistant,
    add_thinking_to_assistant,
    create_assistant_message,
    create_error_message,
    create_permission_message,
    create_system_message,
    create_tool_call_message,
    create_tool_result_message,
    create_user_message,
    create_welcome_message,
)
from .message_list import MessageList
from .messages import UIMessage

# Auto-save session file in current directory
SESSION_FILE = ".nano-cli-session.json"


def build_system_prompt(model: str) -> tuple[str, bool]:
    """Build system prompt with dynamic context.

    Returns:
        A tuple of (system_prompt, claude_md_loaded) where claude_md_loaded
        indicates if CLAUDE.md was found and included in the prompt.
    """
    cwd = os.getcwd()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    claude_md_loaded = False

    base_prompt = f"""You are a helpful assistant.

## Context
- Model: {model}
- Working directory: {cwd}
- Current time: {now}

**Important**: Always use Edit for file modifications. Do not use Bash for file editing (e.g., echo >, sed -i, etc.).

Be concise but helpful. When using tools, explain briefly what you're doing."""

    # Check for CLAUDE.md in the current working directory
    claude_md_path = os.path.join(cwd, "CLAUDE.md")
    if os.path.isfile(claude_md_path):
        try:
            with open(claude_md_path, "r", encoding="utf-8") as f:
                claude_md_content = f.read()
            base_prompt += f"""

{claude_md_content}"""
            claude_md_loaded = True
        except (IOError, OSError):
            # If we can't read the file, just skip it silently
            pass

    return base_prompt, claude_md_loaded


@dataclass
class TerminalApp:
    """Message-list based terminal app for nano-agent.

    Uses a MessageList as the source of truth for UI rendering.
    Each message owns its output buffer, and only the last (active)
    message can receive input events.
    """

    # soft_wrap=True: Let terminal handle line wrapping naturally
    # Custom theme removes background from inline code (markdown.code)
    console: Console = field(
        default_factory=lambda: Console(
            soft_wrap=True,
            theme=Theme({"markdown.code": "cyan"}),
        )
    )
    api: ClaudeCodeAPI | GeminiAPI | None = None
    use_gemini: bool = False
    gemini_model: str = "gemini-3-pro-preview"
    thinking_level: str = "low"
    debug: bool = False
    continue_session: str | None = None
    dag: DAG | None = None
    tools: list[Tool] = field(default_factory=list)
    tool_map: dict[str, Tool] = field(default_factory=dict)
    session: PromptSession[str] | None = None
    # Cancellation support
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    input_handler: InputHandler = field(default=None, init=False)  # type: ignore
    # Message list for UI rendering
    message_list: MessageList = field(default_factory=MessageList)

    def __post_init__(self) -> None:
        """Initialize tools after dataclass creation."""
        if not self.tools:
            self.tools = [
                BashTool(),
                ReadTool(),
                WriteTool(),
                EditTool(permission_callback=self._edit_permission_callback),
                GlobTool(),
                GrepTool(),
                TodoWriteTool(),
                WebFetchTool(),
                PythonTool(),
            ]
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Initialize input handler with escape -> cancel binding
        self.input_handler = InputHandler(on_escape=self._on_escape_pressed)

    def _on_escape_pressed(self) -> None:
        """Called when Escape key is pressed during agent execution."""
        self.cancel_token.cancel()

    def add_message(self, msg: UIMessage) -> UIMessage:
        """Add a message to the list and render it.

        Args:
            msg: The message to add

        Returns:
            The added message
        """
        self.message_list.add(msg)
        self.message_list.render_message(msg, self.console)
        return msg

    def print_history(self, content: Text | Panel | RenderableType | str) -> None:
        """Print content to history (scrolls into terminal scrollback).

        This is a legacy method for compatibility. Prefer using add_message()
        with UIMessage instances for new code.
        """
        self.console.print(content)

    def print_blank(self) -> None:
        """Print a blank line for visual separation."""
        self.console.print()

    def _auto_save(self) -> None:
        """Automatically save the conversation to the session file.

        Saves silently to SESSION_FILE in the current directory after each
        conversation turn. Errors are ignored to not interrupt the user.
        """
        if not self.dag or not self.dag._heads:
            return
        try:
            filepath = Path.cwd() / SESSION_FILE
            self.dag.save(filepath, session_id=datetime.now().isoformat())
        except Exception:
            # Silently ignore save errors to not interrupt the user
            pass

    def _rebuild_message_list_from_dag(self) -> None:
        """Rebuild the message list from the DAG.

        This reconstructs the message list by iterating through the DAG
        and creating UIMessage instances for each node.
        """
        self.message_list.clear()

        if not self.dag or not self.dag._heads:
            return

        # Iterate through all nodes in causal order (parents before children)
        for node in self.dag.head.ancestors():
            if isinstance(node.data, Message):
                if node.data.role == Role.USER:
                    # User message - can be string or list of content blocks
                    if isinstance(node.data.content, str):
                        self.message_list.add(create_user_message(node.data.content))
                    else:
                        # Check what type of content blocks we have
                        text_parts = []
                        tool_results = []
                        for block in node.data.content:
                            if isinstance(block, TextContent):
                                text_parts.append(block.text)
                            elif isinstance(block, ToolResultContent):
                                tool_results.append(block)

                        # Create user message for text content
                        if text_parts:
                            self.message_list.add(
                                create_user_message("\n".join(text_parts))
                            )

                        # Create tool result messages
                        for tool_result in tool_results:
                            if isinstance(tool_result.content, str):
                                result_text = tool_result.content
                            else:
                                result_parts = []
                                for content_block in tool_result.content:
                                    if isinstance(content_block, TextContent):
                                        result_parts.append(content_block.text)
                                result_text = "\n".join(result_parts)
                            is_error = (
                                tool_result.is_error
                                if hasattr(tool_result, "is_error")
                                else False
                            )
                            self.message_list.add(
                                create_tool_result_message(
                                    result_text, is_error=is_error
                                )
                            )

                elif node.data.role == Role.ASSISTANT:
                    # Assistant message - may have thinking and text content
                    if isinstance(node.data.content, str):
                        if node.data.content.strip():
                            msg = create_assistant_message()
                            add_text_to_assistant(msg, node.data.content)
                            self.message_list.add(msg)
                    else:
                        # Process content blocks
                        thinking_blocks = []
                        text_parts = []
                        tool_uses = []

                        for block in node.data.content:
                            if isinstance(block, ThinkingContent):
                                if block.thinking:
                                    thinking_blocks.append(block.thinking)
                            elif isinstance(block, TextContent):
                                if block.text.strip():
                                    text_parts.append(block.text)
                            elif isinstance(block, ToolUseContent):
                                tool_uses.append(block)

                        # Create assistant message with thinking and text
                        if thinking_blocks or text_parts:
                            msg = create_assistant_message()
                            for thinking in thinking_blocks:
                                add_thinking_to_assistant(msg, thinking)
                            if text_parts:
                                add_text_to_assistant(
                                    msg,
                                    "\n".join(text_parts),
                                    has_thinking=bool(thinking_blocks),
                                )
                            self.message_list.add(msg)

                        # Create tool call messages
                        for tool_use in tool_uses:
                            self.message_list.add(
                                create_tool_call_message(
                                    tool_use.name, tool_use.input or {}
                                )
                            )

    def render_history(self) -> None:
        """Re-render all messages from the DAG.

        This rebuilds the message list from the DAG and performs a full redraw.
        Useful after loading a session or when terminal is resized.
        """
        if not self.dag or not self.dag._heads:
            self.add_message(
                create_system_message("No conversation history to render.")
            )
            return

        self._rebuild_message_list_from_dag()
        self.message_list.full_redraw(self.console)
        self.add_message(create_system_message("History rendered."))

    async def initialize_api(self) -> bool:
        """Initialize the API client (ClaudeCodeAPI or GeminiAPI).

        Returns:
            True if initialization successful, False otherwise.
        """
        if self.use_gemini:
            return await self._initialize_gemini_api()
        else:
            return await self._initialize_claude_api()

    async def _initialize_claude_api(self) -> bool:
        """Initialize Claude Code API client."""
        self.add_message(create_system_message("Connecting to Claude Code API..."))

        try:
            try:
                self.api = ClaudeCodeAPI()
            except ValueError:
                # No saved config - capture credentials first
                self.add_message(
                    create_system_message(
                        "No saved credentials, capturing from Claude CLI..."
                    )
                )
                if not await self._refresh_token():
                    return False

            # Set up main DAG
            model = self.api.model
            system_prompt, claude_md_loaded = build_system_prompt(model)
            self.dag = DAG().system(system_prompt).tools(*self.tools)

            self.add_message(create_system_message(f"Connected using {model}"))
            if claude_md_loaded:
                self.add_message(
                    create_system_message("Loaded CLAUDE.md into system prompt")
                )
            self.print_blank()
            return True

        except RuntimeError as e:
            self.add_message(create_error_message(str(e)))
            self.add_message(
                create_system_message(
                    "Claude CLI must be running to capture authentication."
                )
            )
            return False

    async def _initialize_gemini_api(self) -> bool:
        """Initialize Gemini API client."""
        self.add_message(create_system_message("Connecting to Gemini API..."))

        try:
            self.api = GeminiAPI(
                model=self.gemini_model,
                max_tokens=8192,
                thinking_level=self.thinking_level,
            )
            model = self.api.model
            system_prompt, claude_md_loaded = build_system_prompt(model)
            self.dag = DAG().system(system_prompt).tools(*self.tools)
            self.add_message(create_system_message(f"Connected using {model}"))
            if claude_md_loaded:
                self.add_message(
                    create_system_message("Loaded CLAUDE.md into system prompt")
                )
            self.print_blank()
            return True

        except ValueError as e:
            self.add_message(create_error_message(str(e)))
            self.add_message(
                create_system_message(
                    "Set GEMINI_API_KEY environment variable to use Gemini."
                )
            )
            return False

    async def _refresh_token(self) -> bool:
        """Refresh OAuth token by capturing from Claude CLI.

        Returns:
            True if refresh successful, False otherwise.
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: get_config(timeout=30))

            # Re-create API with new token
            self.api = ClaudeCodeAPI()
            self.add_message(create_system_message("Token refreshed."))
            return True

        except TimeoutError:
            self.add_message(
                create_error_message(
                    "Token refresh timed out. Is Claude CLI available?"
                )
            )
            return False
        except RuntimeError as e:
            self.add_message(create_error_message(f"Token refresh failed: {e}"))
            return False

    async def get_user_input(self) -> str | None:
        """Get user input using prompt_toolkit.

        Returns:
            User input string, or None if user wants to quit (Ctrl+D).

        Multiline support:
            - Enter: Submit message
            - Ctrl+J: Add new line
        """
        if self.session is None:
            # Create key bindings for multiline support
            bindings = KeyBindings()

            @bindings.add(Keys.ControlJ)
            def insert_newline(event: KeyPressEvent) -> None:
                """Insert a newline character (Ctrl+J)."""
                event.current_buffer.insert_text("\n")

            # Use FileHistory for persistent history across sessions
            history_path = Path.home() / ".nano-cli-history"
            self.session = PromptSession(
                history=FileHistory(str(history_path)),
                key_bindings=bindings,
            )

        try:
            # Run prompt in executor to avoid blocking asyncio
            loop = asyncio.get_event_loop()
            # Style to make bottom_toolbar invisible (no background)
            invisible_toolbar_style = Style.from_dict({"bottom-toolbar": "noreverse"})
            text = await loop.run_in_executor(
                None,
                lambda: self.session.prompt(  # type: ignore[union-attr]
                    "> ", bottom_toolbar=" ", style=invisible_toolbar_style
                ),
            )
            if text:
                # Clear the echoed input line(s) so we can reprint with formatting
                # Count lines: 1 for the prompt + any newlines in input + 1 for bottom_toolbar
                try:
                    line_count = 2 + text.count("\n")
                    for _ in range(line_count):
                        # Move cursor up and clear the line
                        self.console.file.write("\033[A\033[2K")
                    self.console.file.flush()
                except (OSError, IOError):
                    pass  # Terminal doesn't support ANSI - skip clearing
            return text.strip() if text else None
        except EOFError:
            # Ctrl+D pressed - signal to exit
            return None
        except KeyboardInterrupt:
            # Ctrl+C pressed - just return empty to continue loop
            return ""

    def handle_command(self, command: str) -> bool:
        """Handle a slash command.

        Args:
            command: The command string (including leading slash).

        Returns:
            True if app should continue, False if should quit.
        """
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            return False

        if cmd == "/clear":
            # Clear screen, reset conversation, and message list
            self.message_list.clear()
            self.console.clear()
            claude_md_loaded = False
            if self.dag and self.api:
                model = self.api.model
                system_prompt, claude_md_loaded = build_system_prompt(model)
                self.dag = DAG().system(system_prompt).tools(*self.tools)
            self.add_message(create_system_message("Conversation reset."))
            if claude_md_loaded:
                self.add_message(
                    create_system_message("Loaded CLAUDE.md into system prompt")
                )
            return True

        if cmd == "/render":
            # Re-render all history using message list (useful after terminal resize)
            self.message_list.full_redraw(self.console)
            return True

        if cmd == "/help":
            help_text = """Commands:
  /quit, /exit, /q - Exit the application
  /clear - Reset conversation and clear screen
  /continue, /c - Continue agent execution without user message
  /renew - Refresh OAuth token (use when getting 401 errors)
  /render - Re-render history (useful after terminal resize)
  /debug - Show DAG as JSON
  /save [filename] - Save session to file (default: session.json)
  /load [filename] - Load session from file (default: session.json)
  /help - Show this help message

Input:
  Enter - Send message
  Ctrl+J - Insert new line (for multiline input)
  Esc - Cancel current operation (during execution)
  Ctrl+D - Exit"""
            self.add_message(create_system_message(help_text))
            return True

        if cmd == "/debug":
            if self.dag and self.dag._heads:
                # Build graph structure similar to Node.save_graph
                all_nodes: dict[str, Any] = {}
                for node in self.dag.head.ancestors():
                    all_nodes[node.id] = node.to_dict()
                graph_data = {
                    "head_ids": [h.id for h in self.dag._heads],
                    "nodes": all_nodes,
                }
                dag_json = json.dumps(graph_data, indent=2, default=str)
                self.print_history(JSON(dag_json))
            else:
                self.add_message(create_system_message("No DAG initialized."))
            return True

        # /save [filename] - Save the current DAG to a file
        if cmd.startswith("/save"):
            parts = command.strip().split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session.json"
            if not filename.endswith(".json"):
                filename += ".json"
            if self.dag and self.dag._heads:
                try:
                    filepath = Path(filename).resolve()
                    self.dag.save(filepath, session_id=datetime.now().isoformat())
                    self.add_message(
                        create_system_message(f"Session saved to: {filepath}")
                    )
                except Exception as e:
                    self.add_message(create_error_message(f"Failed to save: {e}"))
            else:
                self.add_message(create_system_message("No DAG to save."))
            return True

        # /load [filename] - Load a DAG from a file
        if cmd.startswith("/load"):
            parts = command.strip().split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session.json"
            if not filename.endswith(".json"):
                filename += ".json"
            filepath = Path(filename).resolve()
            if not filepath.exists():
                self.add_message(create_error_message(f"File not found: {filepath}"))
                return True
            try:
                loaded_dag, metadata = DAG.load(filepath)
                # Restore tools to the loaded DAG
                if self.tools:
                    loaded_dag = loaded_dag.tools(*self.tools)
                self.dag = loaded_dag
                session_id = metadata.get("session_id", "unknown")
                node_count = len(metadata.get("nodes", {}))
                # Render the loaded session history
                self.render_history()
                self.add_message(
                    create_system_message(
                        f"Session loaded from: {filepath}\n"
                        f"  Session ID: {session_id}\n"
                        f"  Nodes: {node_count}"
                    )
                )
            except Exception as e:
                self.add_message(create_error_message(f"Failed to load: {e}"))
            return True

        self.add_message(create_error_message(f"Unknown command: {command}"))
        self.add_message(create_system_message("Type /help for available commands."))
        return True

    async def _edit_permission_callback(
        self, file_path: str, preview: str, match_count: int
    ) -> bool:
        """Permission callback for Edit tool.

        Args:
            file_path: The file being edited.
            preview: The edit preview text.
            match_count: Number of matches being replaced.

        Returns:
            True if the user allows the edit, False otherwise.
        """
        # Create permission message with the preview
        permission_msg = create_permission_message(file_path, match_count, preview)
        self.add_message(permission_msg)

        # Prompt for confirmation using single-character input
        self.console.print("Allow this edit? (y/n/esc): ", end="", style="yellow")

        result = await self.input_handler.prompt_yn()

        # Print newline after single-char input
        self.console.print()

        if result is None:
            # Escape pressed - cancel the operation
            self.cancel_token.cancel()
            return False

        return result

    async def _execute_tool(self, tool_call: ToolUseContent) -> None:
        """Execute a tool call with cancellation support."""
        if not self.dag:
            return

        # Add tool call message
        self.add_message(
            create_tool_call_message(tool_call.name, tool_call.input or {})
        )

        tool = self.tool_map.get(tool_call.name)
        if not tool:
            error_result = TextContent(text=f"Unknown tool: {tool_call.name}")
            self.add_message(
                create_tool_result_message(error_result.text, is_error=True)
            )
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id, content=[error_result], is_error=True
                )
            )
            return

        try:
            result = await self.cancel_token.run(tool.execute(tool_call.input))
            result_list = result if isinstance(result, list) else [result]
            result_text = "\n".join(r.text for r in result_list)
            self.add_message(create_tool_result_message(result_text))
            self.dag = self.dag.tool_result(
                ToolResultContent(tool_use_id=tool_call.id, content=result_list)
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_result = TextContent(text=f"Tool error: {e}")
            self.add_message(
                create_tool_result_message(error_result.text, is_error=True)
            )
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id, content=[error_result], is_error=True
                )
            )

    def _render_response(self, response: Response) -> UIMessage:
        """Render assistant response (thinking, text, token count).

        Args:
            response: The assistant response

        Returns:
            The created UIMessage for the response
        """
        # Debug: show raw content blocks
        if self.debug:
            self.add_message(
                create_system_message(
                    f"Response blocks: {[type(b).__name__ for b in response.content]}"
                )
            )
            for block in response.content:
                self.add_message(create_system_message(f"  {block}"))

        # Create assistant message
        msg = create_assistant_message()

        # Add thinking content (if any)
        has_thinking = False
        for thinking in response.get_thinking():
            if thinking.thinking:
                add_thinking_to_assistant(msg, thinking.thinking)
                has_thinking = True

        # Add text content with token count
        text_content = response.get_text()
        add_text_to_assistant(
            msg,
            text_content or "",
            has_thinking=has_thinking,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_tokens=response.usage.cache_creation_input_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens,
        )

        # Store metadata
        msg.metadata["input_tokens"] = response.usage.input_tokens
        msg.metadata["output_tokens"] = response.usage.output_tokens

        return self.add_message(msg)

    async def _check_for_escape(self) -> None:
        """Background task to poll for Escape key during API calls.

        prompt_toolkit's escape sequence disambiguation requires polling
        rather than pure callback-based detection, since the callback
        fires on raw input but read_keys() may not return the Escape
        until after the disambiguation timeout.
        """
        if not self.input_handler._input or not self.input_handler._running:
            return

        while not self.cancel_token.is_cancelled:
            await asyncio.sleep(0.02)  # Poll every 20ms for faster response
            if not self.input_handler._running:
                break
            # Don't consume input if prompt_yn() is active
            if self.input_handler._prompting:
                continue

            # Force flush to reduce escape sequence disambiguation delay
            inp = self.input_handler._input
            if isinstance(inp, Vt100Input) and hasattr(inp, "vt100_parser"):
                inp.vt100_parser.flush()

            for key_press in self.input_handler._input.read_keys():
                if key_press.key == Keys.Escape:
                    self.cancel_token.cancel()
                    return

    async def _agent_loop(self) -> None:
        """Run agent loop with cancellation support.

        Continuously sends requests to API and executes tool calls until
        no more tool calls are returned or operation is cancelled.

        Uses InputHandler to detect Escape key for cancellation.
        """
        if not self.api or not self.dag:
            return

        # Start input handler to detect Escape key during execution
        async with self.input_handler:
            while True:
                # API call with spinner, polling for Escape in background
                escape_task = asyncio.create_task(self._check_for_escape())
                try:
                    with Live(
                        Spinner("dots", text="Thinking... (Esc to cancel)"),
                        console=self.console,
                        refresh_per_second=10,
                        transient=True,
                    ):
                        response = await self.cancel_token.run(self.api.send(self.dag))
                finally:
                    escape_task.cancel()
                    try:
                        await escape_task
                    except asyncio.CancelledError:
                        pass

                # Update DAG and render response
                self.dag = self.dag.assistant(response.content)
                self._render_response(response)

                # Check for tool calls
                tool_calls = response.get_tool_use()
                if not tool_calls:
                    break

                # Execute tools
                for tool_call in tool_calls:
                    await self._execute_tool(tool_call)

    async def _run_agent_with_error_handling(self) -> None:
        """Run agent loop with error handling, auto-save, and visual formatting."""
        try:
            await self._agent_loop()
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.cancel_token.cancel()
            self.dag = self.dag.user("[Operation cancelled by user]")
            self.add_message(create_error_message("Operation cancelled."))
        except Exception as e:
            self.add_message(create_error_message(f"API error: {e}"))

        self.print_blank()
        self._auto_save()

    async def send_message(self, text: str) -> None:
        """Send a user message and handle the response with agent loop."""
        if not self.api or not self.dag:
            self.add_message(create_error_message("API not initialized"))
            return

        self.cancel_token.reset()
        self.add_message(create_user_message(text))
        self.dag = self.dag.user(text)

        await self._run_agent_with_error_handling()

    async def continue_agent(self) -> None:
        """Continue agent execution without adding a user message.

        Useful for resuming after cancellation or when the agent stopped
        mid-execution (e.g., due to tool call limits).
        """
        if not self.api or not self.dag:
            self.add_message(create_error_message("API not initialized"))
            return

        self.cancel_token.reset()
        self.add_message(create_system_message("Continuing execution..."))

        await self._run_agent_with_error_handling()

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.api and hasattr(self.api, "client"):
            await self.api.client.aclose()

    async def run(self) -> None:
        """Main application loop."""
        try:
            # Add welcome message
            self.add_message(create_welcome_message())

            # Initialize API
            if not await self.initialize_api():
                return

            # Load session if --continue was specified
            if self.continue_session:
                filename = self.continue_session
                if not filename.endswith(".json"):
                    filename += ".json"
                filepath = Path(filename).resolve()
                if not filepath.exists():
                    self.add_message(
                        create_error_message(f"Session file not found: {filepath}")
                    )
                else:
                    try:
                        loaded_dag, metadata = DAG.load(filepath)
                        # Restore tools to the loaded DAG
                        if self.tools:
                            loaded_dag = loaded_dag.tools(*self.tools)
                        self.dag = loaded_dag
                        session_id = metadata.get("session_id", "unknown")
                        node_count = len(metadata.get("nodes", {}))
                        # Render the loaded session history
                        self.render_history()
                        self.add_message(
                            create_system_message(
                                f"Continuing session from: {filepath}\n"
                                f"  Session ID: {session_id}\n"
                                f"  Nodes: {node_count}"
                            )
                        )
                    except Exception as e:
                        self.add_message(
                            create_error_message(f"Failed to load session: {e}")
                        )

            # Main input loop
            while True:
                user_input = await self.get_user_input()

                if user_input is None:
                    # User pressed Ctrl+D
                    self.print_blank()
                    self.add_message(create_system_message("Goodbye!"))
                    break

                if not user_input:
                    continue

                # Handle async commands (need special handling)
                cmd = user_input.lower().strip()
                if cmd in ("/continue", "/c"):
                    await self.continue_agent()
                    continue

                if cmd == "/renew":
                    await self._refresh_token()
                    continue

                # Handle sync slash commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        self.add_message(create_system_message("Goodbye!"))
                        break
                    continue

                # Send message to Claude
                await self.send_message(user_input)
        finally:
            await self.cleanup()


def main() -> None:
    """Entry point for the TUI application."""
    import argparse

    parser = argparse.ArgumentParser(description="nano-cli")
    parser.add_argument(
        "--gemini",
        metavar="MODEL",
        nargs="?",
        const="gemini-3-pro-preview",
        help="Use Gemini API instead of Claude (default model: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["off", "low", "medium", "high"],
        default="low",
        help="Gemini thinking level (default: low)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug output (raw response blocks)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_session",
        metavar="FILE",
        nargs="?",
        const="session.json",
        help="Continue from a saved session (default: session.json)",
    )
    args = parser.parse_args()

    app = TerminalApp(
        use_gemini=args.gemini is not None,
        gemini_model=args.gemini or "gemini-3-pro-preview",
        thinking_level=args.thinking_level,
        debug=args.debug,
        continue_session=args.continue_session,
    )
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
