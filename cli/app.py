"""Simple Rich-based TUI application for nano-agent.

This module provides a minimal terminal app that uses:
- Rich Console.print() for history (terminal scrollback)
- Rich Live context for dynamic buffer (current thinking/response)
- prompt_toolkit for input handling with history navigation

Architecture:
    Terminal Scrollback (not controlled by app)
    ├── User message 1
    ├── Assistant response 1
    └── ... (grows infinitely, terminal handles)

    Dynamic Buffer (app controls via Rich Live)
    ├── Current thinking/response (re-renderable)
    └── Status indicators
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
from rich.console import Console, RenderableType
from rich.json import JSON
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

# Terminal size constraints
MIN_TERMINAL_WIDTH = 60
MIN_TERMINAL_HEIGHT = 10

from nano_agent import (
    DAG,
    ClaudeCodeAPI,
    DummyAPI,
    GeminiAPI,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
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
)

from nano_agent.cancellation import CancellationToken
from .display import (
    format_assistant_message,
    format_error_message,
    format_system_message,
    format_thinking_message,
    format_thinking_separator,
    format_tool_call,
    format_tool_result,
    format_user_message,
)


def build_system_prompt(model: str) -> str:
    """Build system prompt with dynamic context."""
    cwd = os.getcwd()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""You are a helpful terminal assistant.

## Context
- Model: {model}
- Working directory: {cwd}
- Current time: {now}

## Available Tools
- Bash: Run bash commands
- Read: Read files
- Write: Write files
- Edit, EditConfirm: Edit files
- Glob: Find files by pattern
- Grep: Search file contents
- TodoWrite: Manage tasks
- WebFetch: Fetch web content
- Python: Create, edit, and run Python scripts with dependencies

Be concise but helpful. When using tools, explain briefly what you're doing."""


@dataclass
class SimpleTerminalApp:
    """Simple Rich-based terminal app for nano-agent.

    Uses Rich Console for printing to history (terminal scrollback)
    and Rich Live for dynamic status/response display.
    """

    # soft_wrap=True: Let terminal handle line wrapping naturally
    console: Console = field(default_factory=lambda: Console(soft_wrap=True))
    api: ClaudeCodeAPI | GeminiAPI | DummyAPI | None = None
    use_dummy: bool = False
    use_gemini: bool = False
    gemini_model: str = "gemini-3-pro-preview"
    thinking_level: str = "low"
    debug: bool = False
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

    def check_terminal_size(self) -> tuple[bool, int, int]:
        """Check if terminal meets minimum size requirements.

        Returns:
            Tuple of (is_valid, current_width, current_height)
        """
        size = self.console.size
        is_valid = (
            size.width >= MIN_TERMINAL_WIDTH and size.height >= MIN_TERMINAL_HEIGHT
        )
        return is_valid, size.width, size.height

    async def wait_for_valid_terminal_size(self) -> None:
        """Block and display message until terminal is resized."""
        with Live(console=self.console, refresh_per_second=4) as live:
            while True:
                is_valid, width, height = self.check_terminal_size()
                if is_valid:
                    break

                # Display resize message
                message = Text()
                message.append("Terminal too small\n", style="bold red")
                message.append(f"Current: {width}×{height}\n", style="dim")
                message.append(
                    f"Required: {MIN_TERMINAL_WIDTH}×{MIN_TERMINAL_HEIGHT}\n"
                )
                message.append("\nPlease resize your terminal.", style="yellow")

                live.update(Panel(message, title="Resize Required"))
                await asyncio.sleep(0.25)

    def print_history(self, content: Text | Panel | RenderableType | str) -> None:
        """Print content to history (scrolls into terminal scrollback)."""
        self.console.print(content)

    def print_blank(self) -> None:
        """Print a blank line for visual separation."""
        self.console.print()

    async def initialize_api(self) -> bool:
        """Initialize the API client (ClaudeCodeAPI, GeminiAPI, or DummyAPI).

        Returns:
            True if initialization successful, False otherwise.
        """
        if self.use_dummy:
            return await self._initialize_dummy_api()
        elif self.use_gemini:
            return await self._initialize_gemini_api()
        else:
            return await self._initialize_claude_api()

    async def _initialize_claude_api(self) -> bool:
        """Initialize Claude Code API client."""
        self.print_history(format_system_message("Connecting to Claude Code API..."))

        try:
            self.api = ClaudeCodeAPI()
            model = self.api.model
            self.dag = DAG().system(build_system_prompt(model)).tools(*self.tools)
            self.print_history(format_system_message(f"Connected using {model}"))
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
            self.dag = DAG().system(build_system_prompt(model)).tools(*self.tools)
            self.print_history(format_system_message(f"Connected using {model}"))
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

    async def _initialize_dummy_api(self) -> bool:
        """Initialize Dummy API provider for testing."""
        self.print_history(format_system_message("Initializing Dummy API for testing..."))

        self.api = DummyAPI(
            model="dummy-model-v1",
            thinking_probability=0.3,
            tool_call_probability=0.5,
            max_tool_calls=2,
        )
        model = self.api.model
        self.dag = DAG().system(build_system_prompt(model)).tools(*self.tools)
        self.print_history(
            format_system_message(
                f"Dummy API initialized ({model})\n"
                "This is a test provider that generates random responses."
            )
        )
        self.print_blank()
        return True

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
                line_count = 2 + text.count("\n")
                for _ in range(line_count):
                    # Move cursor up and clear the line
                    self.console.file.write("\033[A\033[2K")
                self.console.file.flush()
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
            if self.dag and self.api:
                model = self.api.model
                self.dag = DAG().system(build_system_prompt(model)).tools(*self.tools)
            self.print_history(format_system_message("Conversation reset."))
            return True

        if cmd == "/help":
            help_text = """Commands:
  /quit, /exit, /q - Exit the application
  /clear - Reset conversation and clear screen
  /debug - Show DAG as JSON
  /help - Show this help message

Input:
  Enter - Send message
  Ctrl+J - Insert new line (for multiline input)
  Ctrl+C - Cancel current operation
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

        self.print_history(format_error_message(f"Unknown command: {command}"))
        self.print_history(format_system_message("Type /help for available commands."))
        return True

    async def execute_tool(self, tool_call: ToolUseContent) -> None:
        """Execute a tool call and add result to DAG."""
        if not self.dag:
            return

        # Display tool call
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
            result = await tool.execute(tool_call.input)
            result_list = result if isinstance(result, list) else [result]
            result_text = "\n".join(r.text for r in result_list)

            self.print_history(format_tool_result(result_text))

            self.dag = self.dag.tool_result(
                ToolResultContent(tool_use_id=tool_call.id, content=result_list)
            )

        except Exception as e:
            error_result = TextContent(text=f"Tool error: {e}")
            self.print_history(format_tool_result(error_result.text, is_error=True))
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id, content=[error_result], is_error=True
                )
            )

    async def _execute_tool_cancellable(self, tool_call: ToolUseContent) -> None:
        """Execute a tool call with cancellation support."""
        if not self.dag:
            return

        # Display tool call
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
            # Wrap tool execution in cancellable task
            result = await self.cancel_token.run(tool.execute(tool_call.input))
            result_list = result if isinstance(result, list) else [result]
            result_text = "\n".join(r.text for r in result_list)

            self.print_history(format_tool_result(result_text))

            self.dag = self.dag.tool_result(
                ToolResultContent(tool_use_id=tool_call.id, content=result_list)
            )

        except asyncio.CancelledError:
            # Don't add partial tool result - just re-raise
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

            # Show spinner in dynamic area while waiting
            with Live(
                Spinner("dots", text="Thinking... (Ctrl+C to cancel)"),
                console=self.console,
                refresh_per_second=10,
                transient=True,  # Remove spinner when done
            ) as live:
                # Wrap API call in cancellable task
                response = await self.cancel_token.run(self.api.send(self.dag))
                # Update with completion indicator briefly
                live.update(Text("Response received", style="green dim"))

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

            # Display text content
            text_content = response.get_text()
            if text_content and text_content.strip():
                # Add separator if there was thinking content
                if has_thinking:
                    self.print_history(format_thinking_separator())
                self.print_history(format_assistant_message(text_content))

            # Handle tool calls
            tool_calls = response.get_tool_use()
            if not tool_calls:
                break

            # Execute each tool with cancellation support
            for tool_call in tool_calls:
                # Check for cancellation before each tool
                if self.cancel_token.is_cancelled:
                    raise asyncio.CancelledError()
                await self._execute_tool_cancellable(tool_call)

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
            # Ctrl+C pressed - cancel the operation
            self.cancel_token.cancel()
            self.dag = self.dag.user("[Operation cancelled by user]")
            self.print_history(format_error_message("Operation cancelled."))
        except asyncio.CancelledError:
            # Cancelled via token (e.g., from tool execution)
            self.dag = self.dag.user("[Operation cancelled by user]")
            self.print_history(format_error_message("Operation cancelled."))
        except Exception as e:
            self.print_history(format_error_message(f"API error: {e}"))

        # Visual gap after complete response
        self.print_blank()

    async def run(self) -> None:
        """Main application loop."""
        # Ensure terminal is large enough before starting
        await self.wait_for_valid_terminal_size()

        # Print header
        self.print_history(Text("nano-cli", style="bold cyan"))
        self.print_history(
            Text(
                "Type your message. /help for commands. Ctrl+C to cancel. Ctrl+D to exit.",
                style="dim",
            )
        )
        self.print_blank()

        # Initialize API
        if not await self.initialize_api():
            return

        # Main input loop
        while True:
            # Check terminal size before each prompt
            await self.wait_for_valid_terminal_size()

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


def main() -> None:
    """Entry point for the TUI application."""
    import argparse

    parser = argparse.ArgumentParser(description="nano-cli")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy API provider for testing (no credentials needed)",
    )
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
    args = parser.parse_args()

    app = SimpleTerminalApp(
        use_dummy=args.dummy,
        use_gemini=args.gemini is not None,
        gemini_model=args.gemini or "gemini-3-pro-preview",
        thinking_level=args.thinking_level,
        debug=args.debug,
    )
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
