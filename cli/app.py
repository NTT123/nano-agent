"""Simple Rich-based TUI application for nano-agent.

This module provides a minimal terminal app that uses:
- Rich Console.print() for all output (history and responses)
- Rich Live for spinner while waiting for API response
- prompt_toolkit for input handling with history navigation

Architecture:
    Terminal Scrollback (all output via Console.print())
    ├── User message 1
    ├── Assistant response 1 (thinking + text)
    ├── Tool calls and results
    └── ... (grows infinitely, terminal handles)

    Transient Spinner (Rich Live, removed after response)
    └── "Thinking..." indicator during API call
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
    EditConfirmTool,
    EditTool,
    GlobTool,
    GrepTool,
    PythonTool,
    ReadTool,
    TodoWriteTool,
    Tool,
    WebFetchTool,
    WriteTool,
    get_pending_edit,
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


def build_system_prompt(model: str) -> tuple[str, bool]:
    """Build system prompt with dynamic context.
    
    Returns:
        A tuple of (system_prompt, claude_md_loaded) where claude_md_loaded
        indicates if CLAUDE.md was found and included in the prompt.
    """
    cwd = os.getcwd()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    claude_md_loaded = False

    base_prompt = f"""You are a helpful terminal assistant.

## Context
- Model: {model}
- Working directory: {cwd}
- Current time: {now}

**Important**: Always use Edit/EditConfirm for file modifications. Do not use Bash for file editing (e.g., echo >, sed -i, etc.).

Be concise but helpful. When using tools, explain briefly what you're doing."""

    # Check for CLAUDE.md in the current working directory
    claude_md_path = os.path.join(cwd, "CLAUDE.md")
    if os.path.isfile(claude_md_path):
        try:
            with open(claude_md_path, "r", encoding="utf-8") as f:
                claude_md_content = f.read()
            base_prompt += f"""

## Project Documentation (CLAUDE.md)

The following document (CLAUDE.md) is the main document describing the project. The user prepared it to provide useful contextual information about the project and the practices that are important:

{claude_md_content}"""
            claude_md_loaded = True
        except (IOError, OSError):
            # If we can't read the file, just skip it silently
            pass

    return base_prompt, claude_md_loaded


@dataclass
class TerminalApp:
    """Simple Rich-based terminal app for nano-agent.

    Uses Rich Console for printing to history (terminal scrollback)
    and Rich Live for dynamic status/response display.
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

    def __post_init__(self) -> None:
        """Initialize tools after dataclass creation."""
        if not self.tools:
            self.tools = [
                BashTool(),
                ReadTool(),
                WriteTool(),
                EditTool(),
                EditConfirmTool(),
                GlobTool(),
                GrepTool(),
                TodoWriteTool(),
                WebFetchTool(),
                PythonTool(),
            ]
        self.tool_map = {tool.name: tool for tool in self.tools}

    def print_history(self, content: Text | Panel | RenderableType | str) -> None:
        """Print content to history (scrolls into terminal scrollback)."""
        self.console.print(content)

    def print_blank(self) -> None:
        """Print a blank line for visual separation."""
        self.console.print()

    def render_history(self) -> None:
        """Re-render all messages from the DAG.

        This clears the console and re-displays all user messages, assistant
        responses (including thinking), tool calls, and tool results from the
        current DAG. Useful after loading a session or when terminal is resized.
        """
        if not self.dag or not self.dag._heads:
            self.print_history(format_system_message("No conversation history to render."))
            return

        self.console.clear()

        # Iterate through all nodes in causal order (parents before children)
        for node in self.dag.head.ancestors():
            if isinstance(node.data, Message):
                if node.data.role == Role.USER:
                    # User message - can be string or list of content blocks
                    if isinstance(node.data.content, str):
                        self.print_history(format_user_message(node.data.content))
                    else:
                        # Check what type of content blocks we have
                        text_parts = []
                        tool_results = []
                        for block in node.data.content:
                            if isinstance(block, TextContent):
                                text_parts.append(block.text)
                            elif isinstance(block, ToolResultContent):
                                tool_results.append(block)

                        # Render text content as user message
                        if text_parts:
                            self.print_history(format_user_message("\n".join(text_parts)))

                        # Render tool results
                        for tool_result in tool_results:
                            if isinstance(tool_result.content, str):
                                result_text = tool_result.content
                            else:
                                result_parts = []
                                for content_block in tool_result.content:
                                    if isinstance(content_block, TextContent):
                                        result_parts.append(content_block.text)
                                result_text = "\n".join(result_parts)
                            is_error = tool_result.is_error if hasattr(tool_result, "is_error") else False
                            self.print_history(format_tool_result(result_text, is_error=is_error))

                elif node.data.role == Role.ASSISTANT:
                    # Assistant message - may have thinking and text content
                    if isinstance(node.data.content, str):
                        if node.data.content.strip():
                            self.print_history(format_assistant_message(node.data.content))
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

                        # Display thinking content
                        for thinking in thinking_blocks:
                            self.print_history(format_thinking_message(thinking))

                        # Display text content with separator if there was thinking
                        if text_parts:
                            if thinking_blocks:
                                self.print_history(format_thinking_separator())
                            self.print_history(format_assistant_message("\n".join(text_parts)))

                        # Display tool calls
                        for tool_use in tool_uses:
                            self.print_history(
                                format_tool_call(tool_use.name, tool_use.input or {})
                            )

        self.print_history(format_system_message("History rendered."))

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
        self.print_history(format_system_message("Connecting to Claude Code API..."))

        try:
            try:
                self.api = ClaudeCodeAPI()
            except ValueError:
                # No saved config - capture credentials first
                self.print_history(
                    format_system_message("No saved credentials, capturing from Claude CLI...")
                )
                if not await self._refresh_token():
                    return False

            # Validate token with warm-up (refreshes if needed)
            if not await self._warmup_and_validate_token():
                return False

            # Token is valid, set up main DAG
            model = self.api.model
            system_prompt, claude_md_loaded = build_system_prompt(model)
            self.dag = DAG().system(system_prompt).tools(*self.tools)

            self.print_history(format_system_message(f"Connected using {model}"))
            if claude_md_loaded:
                self.print_history(format_system_message("Loaded CLAUDE.md into system prompt"))
            self.print_blank()
            return True

        except RuntimeError as e:
            self.print_history(format_error_message(str(e)))
            self.print_history(
                format_system_message(
                    "Claude CLI must be running to capture authentication."
                )
            )
            return False

    async def _initialize_gemini_api(self) -> bool:
        """Initialize Gemini API client."""
        self.print_history(format_system_message("Connecting to Gemini API..."))

        try:
            self.api = GeminiAPI(
                model=self.gemini_model,
                max_tokens=8192,
                thinking_level=self.thinking_level,
            )
            model = self.api.model
            system_prompt, claude_md_loaded = build_system_prompt(model)
            self.dag = DAG().system(system_prompt).tools(*self.tools)
            self.print_history(format_system_message(f"Connected using {model}"))
            if claude_md_loaded:
                self.print_history(format_system_message("Loaded CLAUDE.md into system prompt"))
            self.print_blank()
            return True

        except ValueError as e:
            self.print_history(format_error_message(str(e)))
            self.print_history(
                format_system_message(
                    "Set GEMINI_API_KEY environment variable to use Gemini."
                )
            )
            return False

    async def _warmup_and_validate_token(self, max_retries: int = 3) -> bool:
        """Send warm-up message to validate token, refresh if needed.

        Args:
            max_retries: Maximum token refresh attempts.

        Returns:
            True if token is valid (after potential refresh), False if all retries failed.
        """
        for attempt in range(max_retries):
            try:
                # Create temporary DAG just for warm-up (not self.dag)
                warmup_dag = DAG().system("You are helpful.").user("hi")

                # Send warm-up request
                self.print_history(format_system_message("Validating token..."))
                await self.api.send(warmup_dag)  # type: ignore[union-attr]

                # Success - token is valid
                self.print_history(format_system_message("Token valid."))
                return True

            except APIError as e:
                if e.status_code == 401:
                    self.print_history(
                        format_system_message(
                            f"Token expired (attempt {attempt + 1}/{max_retries}), refreshing..."
                        )
                    )

                    # Refresh token
                    if not await self._refresh_token():
                        continue  # Refresh failed, try again

                    # Token refreshed, loop will retry warm-up
                else:
                    # Non-401 error, propagate
                    raise

        # All retries exhausted
        self.print_history(
            format_error_message("Failed to validate token after all attempts.")
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
            self.print_history(format_system_message("Token refreshed."))
            return True

        except TimeoutError:
            self.print_history(
                format_error_message("Token refresh timed out. Is Claude CLI available?")
            )
            return False
        except RuntimeError as e:
            self.print_history(format_error_message(f"Token refresh failed: {e}"))
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
            # Clear screen and reset conversation
            self.console.clear()
            claude_md_loaded = False
            if self.dag and self.api:
                model = self.api.model
                system_prompt, claude_md_loaded = build_system_prompt(model)
                self.dag = DAG().system(system_prompt).tools(*self.tools)
            self.print_history(format_system_message("Conversation reset."))
            if claude_md_loaded:
                self.print_history(format_system_message("Loaded CLAUDE.md into system prompt"))
            return True

        if cmd == "/render":
            # Re-render all history (useful after terminal resize)
            self.render_history()
            return True

        if cmd == "/help":
            help_text = """Commands:
  /quit, /exit, /q - Exit the application
  /clear - Reset conversation and clear screen
  /render - Re-render history (useful after terminal resize)
  /debug - Show DAG as JSON
  /save [filename] - Save session to file (default: session.json)
  /load [filename] - Load session from file (default: session.json)
  /help - Show this help message

Input:
  Enter - Send message
  Ctrl+J - Insert new line (for multiline input)
  Esc - Cancel current operation
  Ctrl+D - Exit"""
            self.print_history(format_system_message(help_text))
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
                self.print_history(format_system_message("No DAG initialized."))
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
                    self.print_history(
                        format_system_message(f"Session saved to: {filepath}")
                    )
                except Exception as e:
                    self.print_history(format_error_message(f"Failed to save: {e}"))
            else:
                self.print_history(format_system_message("No DAG to save."))
            return True

        # /load [filename] - Load a DAG from a file
        if cmd.startswith("/load"):
            parts = command.strip().split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session.json"
            if not filename.endswith(".json"):
                filename += ".json"
            filepath = Path(filename).resolve()
            if not filepath.exists():
                self.print_history(
                    format_error_message(f"File not found: {filepath}")
                )
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
                self.print_history(
                    format_system_message(
                        f"Session loaded from: {filepath}\n"
                        f"  Session ID: {session_id}\n"
                        f"  Nodes: {node_count}"
                    )
                )
            except Exception as e:
                self.print_history(format_error_message(f"Failed to load: {e}"))
            return True

        self.print_history(format_error_message(f"Unknown command: {command}"))
        self.print_history(format_system_message("Type /help for available commands."))
        return True

    async def _permission_callback(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> bool:
        """Interactive permission prompt for EditConfirm.

        Args:
            tool_name: The name of the tool being called.
            tool_input: The input parameters for the tool.

        Returns:
            True if the user allows the operation, False otherwise.
        """
        if tool_name != "EditConfirm":
            return True

        edit_id = tool_input.get("edit_id", "")
        if not edit_id:
            return True  # Let tool handle validation error

        pending = get_pending_edit(edit_id)
        if pending is None:
            return True  # Let tool handle expired/missing error

        # Display edit details
        self.print_blank()
        self.print_history(Text("─── Permission Required ───", style="yellow bold"))
        self.print_history(Text(f"File: {pending.file_path}", style="cyan"))

        if pending.replace_all and pending.match_count > 1:
            self.print_history(
                Text(
                    f"(Replacing {pending.match_count} occurrences)", style="yellow dim"
                )
            )

        self.print_blank()

        # Prompt for confirmation
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: input("Allow this edit? [y/N]: ").strip().lower()
            )
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    async def _execute_tool(
        self, tool_call: ToolUseContent, cancellable: bool = True
    ) -> None:
        """Execute a tool call, optionally with cancellation support."""
        if not self.dag:
            return

        # Check permission for EditConfirm before execution
        if tool_call.name == "EditConfirm":
            allowed = await self._permission_callback(
                tool_call.name, tool_call.input or {}
            )
            if not allowed:
                self.print_history(format_system_message("Edit rejected by user."))
                error_result = TextContent(
                    text="Permission denied: User rejected the edit operation. "
                    "The file was NOT modified."
                )
                self.print_history(format_tool_result(error_result.text, is_error=True))
                self.dag = self.dag.tool_result(
                    ToolResultContent(
                        tool_use_id=tool_call.id, content=[error_result], is_error=True
                    )
                )
                return

        self.print_history(format_tool_call(tool_call.name, tool_call.input or {}))

        tool = self.tool_map.get(tool_call.name)
        if not tool:
            error_result = TextContent(text=f"Unknown tool: {tool_call.name}")
            self.print_history(format_tool_result(error_result.text, is_error=True))
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id, content=[error_result], is_error=True
                )
            )
            return

        try:
            if cancellable:
                result = await self.cancel_token.run_with_escape(
                    tool.execute(tool_call.input)
                )
            else:
                result = await tool.execute(tool_call.input)
            result_list = result if isinstance(result, list) else [result]
            result_text = "\n".join(r.text for r in result_list)
            self.print_history(format_tool_result(result_text))
            self.dag = self.dag.tool_result(
                ToolResultContent(tool_use_id=tool_call.id, content=result_list)
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_result = TextContent(text=f"Tool error: {e}")
            self.print_history(format_tool_result(error_result.text, is_error=True))
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id, content=[error_result], is_error=True
                )
            )

    async def _agent_loop(self) -> None:
        """Run agent loop with cancellation support."""
        if not self.api or not self.dag:
            return

        while True:
            # Check for cancellation before API call
            if self.cancel_token.is_cancelled:
                raise asyncio.CancelledError()

            # Show status and wait for response (Esc to cancel)
            self.console.print(
                Spinner("dots", text="Thinking... (Esc to cancel)"),
                end="\r",
            )
            try:
                response = await self.cancel_token.run_with_escape(
                    self.api.send(self.dag)
                )
            finally:
                # Clear the spinner line
                self.console.print(" " * 40, end="\r")

            # Add response to DAG
            self.dag = self.dag.assistant(response.content)

            # Debug: show raw content blocks
            if self.debug:
                self.print_history(
                    format_system_message(
                        f"Response blocks: {[type(b).__name__ for b in response.content]}"
                    )
                )
                for block in response.content:
                    self.print_history(format_system_message(f"  {block}"))

            # Display thinking content (if any)
            thinking_blocks = response.get_thinking()
            has_thinking = False
            for thinking in thinking_blocks:
                if thinking.thinking:
                    self.print_history(format_thinking_message(thinking.thinking))
                    has_thinking = True

            # Display text content with token count
            text_content = response.get_text()
            if text_content and text_content.strip():
                # Add separator if there was thinking content
                if has_thinking:
                    self.print_history(format_thinking_separator())
                # Display assistant message and token count together
                self.print_history(
                    Group(
                        format_assistant_message(text_content),
                        format_token_count(
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            cache_creation_tokens=response.usage.cache_creation_input_tokens,
                            cache_read_tokens=response.usage.cache_read_input_tokens,
                        ),
                    )
                )
            else:
                # If no text content, just show token count
                self.print_history(
                    format_token_count(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        cache_creation_tokens=response.usage.cache_creation_input_tokens,
                        cache_read_tokens=response.usage.cache_read_input_tokens,
                    )
                )

            # Handle tool calls
            tool_calls = response.get_tool_use()
            if not tool_calls:
                break

            # Execute each tool with cancellation support
            for tool_call in tool_calls:
                # Check for cancellation before each tool
                if self.cancel_token.is_cancelled:
                    raise asyncio.CancelledError()
                await self._execute_tool(tool_call, cancellable=True)

    async def send_message(self, text: str) -> None:
        """Send a user message and handle the response with agent loop."""
        if not self.api or not self.dag:
            self.print_history(format_error_message("API not initialized"))
            return

        # Reset cancellation token for new message
        self.cancel_token.reset()

        # Display formatted user message with background color
        self.print_history(format_user_message(text))
        self.print_blank()

        # Add to DAG
        self.dag = self.dag.user(text)

        try:
            await self._agent_loop()
        except KeyboardInterrupt:
            # Ctrl+C pressed - fallback cancellation
            self.cancel_token.cancel()
            self.dag = self.dag.user("[Operation cancelled by user]")
            self.print_history(format_error_message("Operation cancelled."))
        except asyncio.CancelledError:
            # Cancelled via Escape key or token
            self.dag = self.dag.user("[Operation cancelled by user]")
            self.print_history(format_error_message("Operation cancelled."))
        except Exception as e:
            self.print_history(format_error_message(f"API error: {e}"))

        # Visual gap after complete response
        self.print_blank()

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.api and hasattr(self.api, "client"):
            await self.api.client.aclose()

    async def run(self) -> None:
        """Main application loop."""
        try:
            # Print header
            self.print_history(Text("nano-cli", style="bold cyan"))
            self.print_history(
                Text(
                    "Type your message. /help for commands. Esc to cancel. Ctrl+D to exit.",
                    style="dim",
                )
            )
            self.print_blank()

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
                    self.print_history(
                        format_error_message(f"Session file not found: {filepath}")
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
                        self.print_history(
                            format_system_message(
                                f"Continuing session from: {filepath}\n"
                                f"  Session ID: {session_id}\n"
                                f"  Nodes: {node_count}"
                            )
                        )
                    except Exception as e:
                        self.print_history(format_error_message(f"Failed to load session: {e}"))

            # Main input loop
            while True:
                user_input = await self.get_user_input()

                if user_input is None:
                    # User pressed Ctrl+D
                    self.print_blank()
                    self.print_history(format_system_message("Goodbye!"))
                    break

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        self.print_history(format_system_message("Goodbye!"))
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
