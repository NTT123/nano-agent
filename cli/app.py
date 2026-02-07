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
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import AsyncIterator

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from nano_agent import (
    DAG,
    ClaudeCodeAPI,
    CodexAPI,
    FireworksAPI,
    GeminiAPI,
    Response,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from nano_agent.cancellation import (
    CancellationChoice,
    CancellationToken,
    ToolExecutionBatch,
    ToolExecutionStatus,
    TrackedToolCall,
)
from nano_agent.providers.base import APIError
from nano_agent.providers.capture_claude_code_auth import async_get_config
from nano_agent.providers.cost import calculate_cost
from nano_agent.tools import (
    AskUserQuestionTool,
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
    convert_input,
)

from . import display
from .commands import CommandContext, CommandRouter
from .config import load_cli_config, save_cli_config
from .elements.terminal import ANSI
from .input_controller import InputController
from .mapper import DAGMessageMapper
from .message_factory import (
    add_text_to_assistant,
    add_thinking_to_assistant,
    create_assistant_message,
    create_command_message,
    create_error_message,
    create_permission_message,
    create_question_message,
    create_system_message,
    create_tool_call_message,
    create_tool_result_message,
    create_user_message,
    create_welcome_message,
)
from .message_list import MessageList
from .messages import UIMessage
from .session import SessionStore


class AppState(Enum):
    """Application lifecycle states for documentation and debugging.

    This enum tracks the high-level state of the CLI application.
    Used for logging/debugging only - no transition validation performed.

    State Diagram:
        INIT ──► IDLE ◄──► COMMAND
                  │
                  ▼
              THINKING ◄───┐
                  │        │
                  ▼        │
            EXECUTING_TOOL─┘
                  │
                  ▼
            AWAITING_USER ──► CANCELLATION
                                   │
                                   ▼
                               (back to IDLE or EXECUTING_TOOL)

        Any state ──► SHUTDOWN (on exit)
    """

    INIT = auto()  # App starting, API initialization
    IDLE = auto()  # Awaiting user input
    COMMAND = auto()  # Processing slash command
    THINKING = auto()  # API call in progress
    EXECUTING_TOOL = auto()  # Tool execution in progress
    AWAITING_USER = auto()  # Waiting for user response (confirm/select)
    CANCELLATION = auto()  # Showing cancellation menu
    SHUTDOWN = auto()  # Cleanup and exit


async def build_system_prompt_async(model: str) -> tuple[str, bool]:
    """Build system prompt with dynamic context (async version).

    Uses run_in_executor for non-blocking file I/O.

    Returns:
        A tuple of (system_prompt, claude_md_loaded) where claude_md_loaded
        indicates if CLAUDE.md was found and included in the prompt.
    """
    cwd = os.getcwd()
    claude_md_loaded = False

    base_prompt = """You are a helpful assistant.

**Important**: Always use Edit for file modifications. Do not use Bash for file editing (e.g., echo >, sed -i, etc.).

Be concise but helpful. When using tools, explain briefly what you're doing."""

    # Check for CLAUDE.md in the current working directory
    claude_md_path = os.path.join(cwd, "CLAUDE.md")
    if os.path.isfile(claude_md_path):
        try:
            # Use run_in_executor for non-blocking file read
            loop = asyncio.get_event_loop()
            claude_md_content = await loop.run_in_executor(
                None, lambda: Path(claude_md_path).read_text(encoding="utf-8")
            )
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
    api: ClaudeCodeAPI | GeminiAPI | CodexAPI | FireworksAPI | None = None
    use_gemini: bool = False
    use_codex: bool = False
    use_fireworks: bool = False
    codex_model: str = "gpt-5.2-codex"
    gemini_model: str = "gemini-3-pro-preview"
    fireworks_model: str = "accounts/fireworks/models/kimi-k2p5"
    thinking_level: str = "low"
    thinking_budget: int | None = 16000
    debug: bool = False
    continue_session: str | None = None
    dag: DAG | None = None
    tools: list[Tool] = field(default_factory=list)
    tool_map: dict[str, Tool] = field(default_factory=dict)
    command_router: CommandRouter = field(default_factory=CommandRouter)
    session_store: SessionStore = field(default_factory=SessionStore)
    # Cancellation support
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    input_controller: InputController = field(default_factory=InputController)
    dag_mapper: DAGMessageMapper = field(default_factory=DAGMessageMapper)
    # Message list for UI rendering
    message_list: MessageList = field(default_factory=MessageList)
    # Auto-accept mode: when enabled, automatically confirms edit operations
    auto_accept: bool = False
    # UI color scheme: "dark" or "light"
    color_scheme: str = "dark"
    # Accumulated token stats
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_thinking_tokens: int = 0
    total_cost: float = 0.0
    # Concurrent execution support
    pending_messages: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    _execution_task: asyncio.Task[None] | None = field(
        default=None, init=False, repr=False
    )
    # App state tracking (for debugging/logging)
    _state: AppState = field(default=AppState.INIT, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize tools after dataclass creation."""
        # Apply stored color scheme before rendering any messages
        display.set_color_scheme(self.color_scheme)
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
                AskUserQuestionTool(),
            ]
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Initialize input controller with escape -> cancel binding
        self.input_controller.on_escape = self._on_escape_pressed
        # Initialize input controller with Shift+Tab -> toggle auto-accept binding
        self.input_controller.on_toggle_auto_accept = self._toggle_auto_accept

    def _set_state(self, state: AppState) -> None:
        """Update app state (for debugging/logging).

        This is informational only - no validation or side effects.
        Enable debug logging to see state transitions.
        """
        self._state = state

    def get_state(self) -> AppState:
        """Get current app state."""
        return self._state

    @asynccontextmanager
    async def _state_activity(
        self, state: AppState, activity: str
    ) -> AsyncIterator[None]:
        """Context manager for state + activity updates.

        Sets state and activity on entry, clears activity on exit.
        Reduces boilerplate for the common pattern of:
            self._set_state(state)
            await self.input_controller.set_activity(activity)
            try:
                ...
            finally:
                await self.input_controller.set_activity(None)
        """
        self._set_state(state)
        await self.input_controller.set_activity(activity)
        try:
            yield
        finally:
            await self.input_controller.set_activity(None)

    def _on_escape_pressed(self) -> None:
        """Called when Escape key is pressed during agent execution."""
        self.cancel_token.cancel()

    def _toggle_auto_accept(self) -> None:
        """Toggle auto-accept mode (called by Shift+Tab)."""
        self.auto_accept = not self.auto_accept
        # Sync with footer status bar
        self.input_controller.update_status(auto_accept=self.auto_accept)

    def _set_bookmark(self) -> None:
        """Set a terminal bookmark at current cursor position.

        Uses iTerm2/WezTerm OSC 1337 SetMark sequence. The bookmark persists
        even in scrollback buffer. Used by /clear to preserve content from
        before the app started.
        """
        ANSI.set_mark()

    def _set_color_scheme(self, scheme: str) -> bool:
        """Set the UI color scheme and return True if applied."""
        if not display.set_color_scheme(scheme):
            return False
        self.color_scheme = scheme.strip().lower()
        # Persist to config
        config = load_cli_config()
        config["color_scheme"] = self.color_scheme
        save_cli_config(config)
        return True

    def _sync_status_to_footer(self) -> None:
        """Sync current state to footer status bar."""
        self.input_controller.update_status(
            auto_accept=self.auto_accept,
            input_tokens=self.last_input_tokens,
            output_tokens=self.last_output_tokens,
            thinking_tokens=self.last_thinking_tokens,
            cost=self.total_cost,
        )

    def add_message(self, msg: UIMessage) -> UIMessage:
        """Add a message to the list and render it.

        Args:
            msg: The message to add

        Returns:
            The added message
        """
        self.message_list.add(msg)

        # Use unified transition API - handles seamless overwrite when appropriate
        if (
            msg.message_type == "user"
            and self.input_controller.has_overwritable_content()
        ):
            # Seamless transition: overwrite footer content in place
            self.input_controller.transition_to_message(
                lambda: self.message_list.render_message(msg, self.console)
            )
        else:
            # Normal path: pause, render, resume
            self.input_controller.pause_footer()
            self.message_list.render_message(msg, self.console)
            self.input_controller.resume_footer()

        return msg

    def print_history(self, content: Text | Panel | RenderableType | str) -> None:
        """Print content to history (scrolls into terminal scrollback).

        This is a legacy method for compatibility. Prefer using add_message()
        with UIMessage instances for new code.
        """
        self.input_controller.pause_footer()
        self.console.print(content)
        self.input_controller.resume_footer()

    def print_blank(self) -> None:
        """Print a blank line for visual separation."""
        self.input_controller.pause_footer()
        self.console.print()
        self.input_controller.resume_footer()

    async def _auto_save_async(self) -> None:
        """Automatically save the conversation to the session file (async version)."""
        await self.session_store.auto_save(self.dag)

    def _rebuild_message_list_from_dag(self) -> None:
        """Rebuild the message list from the DAG."""
        self.message_list = self.dag_mapper.rebuild(self.dag)

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
        # Pause footer to avoid leaving blank reserved lines in scrollback
        self.input_controller.pause_footer()
        self.message_list.full_redraw(self.console)
        self.input_controller.resume_footer()
        self.add_message(create_system_message("History rendered."))

    async def initialize_api(self) -> bool:
        """Initialize the API client (ClaudeCodeAPI, GeminiAPI, CodexAPI, or FireworksAPI).

        Returns:
            True if initialization successful, False otherwise.
        """
        if self.use_codex:
            return await self._initialize_codex_api()
        if self.use_gemini:
            return await self._initialize_gemini_api()
        if self.use_fireworks:
            return await self._initialize_fireworks_api()
        else:
            return await self._initialize_claude_api()

    async def _initialize_claude_api(self) -> bool:
        """Initialize Claude Code API client."""
        self.add_message(create_system_message("Connecting to Claude Code API..."))

        try:
            try:
                self.api = ClaudeCodeAPI(thinking_budget=self.thinking_budget)
            except ValueError:
                # No saved config - capture credentials first
                self.add_message(
                    create_system_message(
                        "No saved credentials, capturing from Claude CLI..."
                    )
                )
                if not await self._refresh_token():
                    return False
                self.api = ClaudeCodeAPI(thinking_budget=self.thinking_budget)

            # Set up main DAG
            model = self.api.model
            claude_md_loaded = await self._build_dag_for_model(model)

            self.add_message(create_system_message(f"Connected using {model}"))
            if claude_md_loaded:
                self.add_message(
                    create_system_message("Loaded CLAUDE.md into system prompt")
                )
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
            claude_md_loaded = await self._build_dag_for_model(model)
            self.add_message(create_system_message(f"Connected using {model}"))
            if claude_md_loaded:
                self.add_message(
                    create_system_message("Loaded CLAUDE.md into system prompt")
                )
            return True

        except ValueError as e:
            self.add_message(create_error_message(str(e)))
            self.add_message(
                create_system_message(
                    "Set GEMINI_API_KEY environment variable to use Gemini."
                )
            )
            return False

    async def _initialize_codex_api(self) -> bool:
        """Initialize Codex API client."""
        self.add_message(create_system_message("Connecting to Codex API..."))

        try:
            self.api = CodexAPI(model=self.codex_model)
            model = self.api.model
            claude_md_loaded = await self._build_dag_for_model(model)
            self.add_message(create_system_message(f"Connected using {model}"))
            if claude_md_loaded:
                self.add_message(
                    create_system_message("Loaded CLAUDE.md into system prompt")
                )
            return True
        except ValueError as e:
            self.add_message(create_error_message(str(e)))
            self.add_message(
                create_system_message(
                    "Log in with Codex and ensure auth is stored in file mode."
                )
            )
            return False

    async def _initialize_fireworks_api(self) -> bool:
        """Initialize Fireworks API client."""
        self.add_message(create_system_message("Connecting to Fireworks API..."))

        try:
            # Map thinking_level to reasoning_effort (off = None)
            reasoning_effort = (
                self.thinking_level if self.thinking_level != "off" else None
            )
            self.api = FireworksAPI(
                model=self.fireworks_model,
                debug=self.debug,
                max_retries=3,
                session_id="nano-cli-session",
                reasoning_effort=reasoning_effort,
            )
            model = self.api.model
            claude_md_loaded = await self._build_dag_for_model(model)
            reasoning_info = (
                f", reasoning={reasoning_effort}" if reasoning_effort else ""
            )
            self.add_message(
                create_system_message(f"Connected using {model}{reasoning_info}")
            )
            if claude_md_loaded:
                self.add_message(
                    create_system_message("Loaded CLAUDE.md into system prompt")
                )
            return True
        except ValueError as e:
            self.add_message(create_error_message(str(e)))
            self.add_message(
                create_system_message(
                    "Set FIREWORKS_API_KEY environment variable to use Fireworks."
                )
            )
            return False

    async def _refresh_token(self) -> bool:
        """Refresh OAuth token by capturing from Claude CLI.

        Returns:
            True if refresh successful, False otherwise.
        """
        try:
            # Pause footer so async_get_config's stderr output doesn't
            # corrupt the terminal region.  add_message() below will
            # resume the footer on every exit path.
            self.input_controller.pause_footer()
            # Use async version directly - no thread pool overhead
            await async_get_config(timeout=30)

            # Re-create API with new token
            self.api = ClaudeCodeAPI(thinking_budget=self.thinking_budget)
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

    async def _build_dag_for_model(self, model: str) -> bool:
        """Create a fresh DAG for a given model and return CLAUDE.md status."""
        system_prompt, claude_md_loaded = await build_system_prompt_async(model)
        self.dag = DAG().system(system_prompt).tools(*self.tools)
        return claude_md_loaded

    async def get_user_input(self) -> str | None:
        """Get user input using the active-element system.

        Returns:
            User input string, or None if user wants to quit (Ctrl+D).

        Multiline support:
            - Enter: Submit message
            - \\ + Enter: Add new line
        """
        text = await self.input_controller.prompt_text(prompt="> ")
        if text is None:
            return None
        return text

    async def handle_command(self, command: str) -> bool:
        """Handle a slash command.

        Args:
            command: The command string (including leading slash).

        Returns:
            True if app should continue, False if should quit.
        """

        async def _clear_and_reset() -> bool:
            self.message_list.clear()
            # Pause footer first
            self.input_controller.pause_footer()
            # Clear to bookmark (preserves content before app started)
            ANSI.clear_to_mark()
            # Set new bookmark at current position for next clear
            self._set_bookmark()
            # Resume footer
            self.input_controller.resume_footer()
            # Re-render welcome message and connection status
            self.add_message(create_welcome_message())
            if self.api:
                model = self.api.model
                if self.use_fireworks:
                    reasoning = (
                        self.thinking_level if self.thinking_level != "off" else None
                    )
                    info = f", reasoning={reasoning}" if reasoning else ""
                    self.add_message(
                        create_system_message(f"Connected using {model}{info}")
                    )
                elif self.use_gemini:
                    self.add_message(create_system_message(f"Connected using {model}"))
                elif self.use_codex:
                    self.add_message(create_system_message(f"Connected using {model}"))
                else:
                    self.add_message(create_system_message(f"Connected using {model}"))
            # Rebuild DAG if needed
            if self.dag and self.api:
                model = self.api.model
                claude_md_loaded = await self._build_dag_for_model(model)
                if claude_md_loaded:
                    self.add_message(
                        create_system_message("Loaded CLAUDE.md into system prompt")
                    )
            ctx.dag = self.dag
            return True

        ctx = CommandContext(
            dag=self.dag,
            session_store=self.session_store,
            tools=self.tools,
            render_history=self.render_history,
            clear_and_reset=_clear_and_reset,
            refresh_token=self._refresh_token,
            set_color_scheme=self._set_color_scheme,
        )
        should_continue, messages = await self.command_router.handle(command, ctx)
        self.dag = ctx.dag
        for message in messages:
            if isinstance(message, UIMessage):
                self.add_message(message)
            else:
                self.print_history(message)
        return should_continue

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
        # Create permission message with the preview (displayed in message history)
        permission_msg = create_permission_message(file_path, match_count, preview)
        self.add_message(permission_msg)

        # If auto-accept mode is enabled, automatically approve the edit
        if self.auto_accept:
            self.console.print("[green]Auto-accepted[/green]")
            return True

        # Pause escape listener to avoid competing raw readers during prompt
        was_running = self.input_controller.is_running()
        prev_state = self._state
        if was_running:
            await self.input_controller.stop()
        try:
            self._set_state(AppState.AWAITING_USER)
            result = await self.input_controller.confirm(
                message=f"Allow edit to {file_path}?"
            )
        finally:
            self._set_state(prev_state)
            if was_running:
                await self.input_controller.start()

        if result is None:
            # Escape pressed - cancel the operation
            self.cancel_token.cancel()
            return False

        return result

    async def _handle_user_question(self, tool_call: ToolUseContent) -> None:
        """Handle AskUserQuestion tool calls with interactive prompts."""
        if not self.dag:
            return

        try:
            tool = self.tool_map.get(tool_call.name)
            typed_input = tool_call.input or {}
            if tool and getattr(tool, "_input_type", None):
                typed_input = convert_input(tool_call.input, tool._input_type)

            if not hasattr(typed_input, "questions"):
                error_result = TextContent(
                    text="AskUserQuestion: invalid input (missing questions)"
                )
                self.add_message(
                    create_tool_result_message(error_result.text, is_error=True)
                )
                self.dag = self.dag.tool_result(
                    ToolResultContent(
                        tool_use_id=tool_call.id,
                        content=[error_result],
                        is_error=True,
                    )
                )
                return

            questions = typed_input.questions
            if not questions:
                error_result = TextContent(
                    text="AskUserQuestion: no questions provided"
                )
                self.add_message(
                    create_tool_result_message(error_result.text, is_error=True)
                )
                self.dag = self.dag.tool_result(
                    ToolResultContent(
                        tool_use_id=tool_call.id,
                        content=[error_result],
                        is_error=True,
                    )
                )
                return

            question = questions[0]
            self.add_message(
                create_question_message(f"{question.header}: {question.question}")
            )

            options = [q.label for q in question.options]
            custom_label = None
            if getattr(question, "allowCustom", False):
                custom_label = getattr(question, "customLabel", "Other...")
                options = options + [custom_label]

            # Pause escape listener to avoid competing raw readers during prompt
            was_running = self.input_controller.is_running()
            prev_state = self._state
            if was_running:
                await self.input_controller.stop()
            try:
                self._set_state(AppState.AWAITING_USER)
                selection = await self.input_controller.select(
                    title=question.question,
                    options=options,
                    multi_select=question.multiSelect,
                )
                if selection is None:
                    # Escape pressed - cancel the operation
                    self.cancel_token.cancel()
                    cancelled = TextContent(text="Operation cancelled by user.")
                    self.add_message(
                        create_tool_result_message(cancelled.text, is_error=True)
                    )
                    self.dag = self.dag.tool_result(
                        ToolResultContent(
                            tool_use_id=tool_call.id,
                            content=[cancelled],
                            is_error=True,
                        )
                    )
                    return

                if question.multiSelect:
                    chosen = selection if isinstance(selection, list) else [selection]
                    custom_selected = custom_label and custom_label in chosen
                    if custom_selected:
                        chosen = [c for c in chosen if c != custom_label]
                        custom_prompt = getattr(
                            question, "customPrompt", "Your answer: "
                        )
                        custom_text = await self.input_controller.prompt_text(
                            prompt=custom_prompt
                        )
                        if custom_text is None:
                            self.cancel_token.cancel()
                            cancelled = TextContent(text="Operation cancelled by user.")
                            self.add_message(
                                create_tool_result_message(
                                    cancelled.text, is_error=True
                                )
                            )
                            self.dag = self.dag.tool_result(
                                ToolResultContent(
                                    tool_use_id=tool_call.id,
                                    content=[cancelled],
                                    is_error=True,
                                )
                            )
                            return
                        if custom_text:
                            chosen.append(custom_text)
                    if not chosen and custom_selected:
                        chosen.append(custom_label or "")
                    result_text = ", ".join(c for c in chosen if c)
                else:
                    selected = selection if isinstance(selection, str) else ""
                    if custom_label and selected == custom_label:
                        custom_prompt = getattr(
                            question, "customPrompt", "Your answer: "
                        )
                        custom_text = await self.input_controller.prompt_text(
                            prompt=custom_prompt
                        )
                        if custom_text is None:
                            self.cancel_token.cancel()
                            cancelled = TextContent(text="Operation cancelled by user.")
                            self.add_message(
                                create_tool_result_message(
                                    cancelled.text, is_error=True
                                )
                            )
                            self.dag = self.dag.tool_result(
                                ToolResultContent(
                                    tool_use_id=tool_call.id,
                                    content=[cancelled],
                                    is_error=True,
                                )
                            )
                            return
                        if custom_text:
                            selected = custom_text
                    result_text = selected
            finally:
                self._set_state(prev_state)
                if was_running:
                    await self.input_controller.start()

            result = TextContent(text=result_text)
            self.add_message(create_tool_result_message(result.text))
            self.dag = self.dag.tool_result(
                ToolResultContent(tool_use_id=tool_call.id, content=[result])
            )
        except asyncio.CancelledError:
            cancelled_result = TextContent(text="Operation cancelled by user.")
            self.add_message(
                create_tool_result_message(cancelled_result.text, is_error=True)
            )
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id,
                    content=[cancelled_result],
                    is_error=True,
                )
            )
            raise
        except Exception as e:
            error_result = TextContent(text=f"Tool error: {e}")
            self.add_message(
                create_tool_result_message(error_result.text, is_error=True)
            )
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id,
                    content=[error_result],
                    is_error=True,
                )
            )

    async def _execute_tool(self, tool_call: ToolUseContent) -> None:
        """Execute a tool call with cancellation support."""
        if not self.dag:
            return

        # Add tool call message
        self.add_message(
            create_tool_call_message(tool_call.name, tool_call.input or {})
        )

        if tool_call.name == "AskUserQuestion":
            await self._handle_user_question(tool_call)
            return

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
            tool_result = await self.cancel_token.run(tool.execute(tool_call.input))
            content = tool_result.content
            result_list = content if isinstance(content, list) else [content]
            result_text = "\n".join(r.text for r in result_list)
            self.add_message(create_tool_result_message(result_text))
            self.dag = self.dag.tool_result(
                ToolResultContent(tool_use_id=tool_call.id, content=result_list)
            )
        except asyncio.CancelledError:
            # User cancelled - add tool result to avoid API error about missing tool_result
            cancelled_result = TextContent(text="Operation cancelled by user.")
            self.add_message(
                create_tool_result_message(cancelled_result.text, is_error=True)
            )
            self.dag = self.dag.tool_result(
                ToolResultContent(
                    tool_use_id=tool_call.id, content=[cancelled_result], is_error=True
                )
            )
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

    async def _execute_tool_with_status(self, tool_call: ToolUseContent) -> None:
        """Execute a tool call with status bar showing tool name.

        Uses footer status bar to show the tool being executed.
        """
        if not self.dag:
            return

        # Show activity in footer status bar
        await self.input_controller.set_activity(f"Running {tool_call.name}...")
        escape_task = asyncio.create_task(self._check_for_escape())
        try:
            await self._execute_tool(tool_call)
        finally:
            await self.input_controller.set_activity(None)
            escape_task.cancel()
            try:
                await escape_task
            except asyncio.CancelledError:
                pass

    async def _execute_tools_with_tracking(
        self, tool_calls: list[ToolUseContent], checkpoint: DAG
    ) -> bool:
        """Execute tool calls with tracking and cancellation handling.

        Args:
            tool_calls: List of tool calls to execute
            checkpoint: DAG state before assistant response (for rollback)

        Returns:
            True if execution completed (with or without user intervention),
            False if user chose to undo all (rollback occurred).
        """
        if not self.dag:
            return True

        # Create batch tracker
        batch = ToolExecutionBatch(
            tool_calls=[
                TrackedToolCall(
                    id=tc.id,
                    name=tc.name,
                    input=tc.input or {},
                )
                for tc in tool_calls
            ]
        )

        i = 0
        while i < len(tool_calls):
            tool_call = tool_calls[i]
            batch.mark_running(i)

            # Show activity in footer
            await self.input_controller.set_activity(f"Running {tool_call.name}...")
            escape_task = asyncio.create_task(self._check_for_escape())

            try:
                await self._execute_tool(tool_call)
                batch.mark_completed(i, "success")
                i += 1

            except asyncio.CancelledError:
                batch.mark_cancelled(i)

                # Stop escape polling
                await self.input_controller.set_activity(None)
                escape_task.cancel()
                try:
                    await escape_task
                except asyncio.CancelledError:
                    pass

                # Reset cancel token for menu interaction
                self.cancel_token.reset()

                # Show cancellation menu and get user choice
                self._set_state(AppState.CANCELLATION)
                choice = await self.input_controller.show_cancellation_menu(batch)
                self._set_state(AppState.EXECUTING_TOOL)

                if choice == CancellationChoice.RETRY:
                    # Re-execute the same tool (don't increment i)
                    batch.tool_calls[i].status = ToolExecutionStatus.PENDING
                    continue

                elif choice == CancellationChoice.SKIP:
                    # Tool result was already added by _execute_tool's except handler
                    # Mark as skipped and continue to next
                    batch.tool_calls[i].status = ToolExecutionStatus.SKIPPED
                    i += 1
                    continue

                elif choice == CancellationChoice.KEEP_COMPLETED:
                    # Add skipped results for remaining pending tools
                    for j in range(i + 1, len(tool_calls)):
                        pending_call = tool_calls[j]
                        batch.mark_skipped(j)
                        skipped_result = TextContent(text="Tool skipped by user.")
                        self.add_message(
                            create_tool_call_message(
                                pending_call.name, pending_call.input or {}
                            )
                        )
                        self.add_message(
                            create_tool_result_message(
                                skipped_result.text, is_error=True
                            )
                        )
                        self.dag = self.dag.tool_result(
                            ToolResultContent(
                                tool_use_id=pending_call.id,
                                content=[skipped_result],
                                is_error=True,
                            )
                        )
                    return True

                elif choice == CancellationChoice.UNDO_ALL or choice is None:
                    # Rollback to checkpoint
                    self.dag = checkpoint
                    # Remove messages added after checkpoint
                    # (UI messages are harder to rollback, so just add a note)
                    self.add_message(
                        create_system_message(
                            "Rolled back to before assistant response."
                        )
                    )
                    return False

            finally:
                await self.input_controller.set_activity(None)
                if not escape_task.done():
                    escape_task.cancel()
                    try:
                        await escape_task
                    except asyncio.CancelledError:
                        pass

        return True

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

        # Calculate cost for this response
        cost_breakdown = calculate_cost(response.usage, model=response.model)
        self.total_cost += cost_breakdown.total_cost

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
            reasoning_tokens=response.usage.reasoning_tokens,
            cached_tokens=response.usage.cached_tokens,
            cost=cost_breakdown.total_cost,
        )

        # Store metadata
        msg.metadata["input_tokens"] = response.usage.input_tokens
        msg.metadata["output_tokens"] = response.usage.output_tokens

        # Track last-step token stats (for footer)
        self.last_input_tokens = response.usage.input_tokens
        self.last_output_tokens = response.usage.output_tokens
        self.last_thinking_tokens = response.usage.reasoning_tokens

        # Update footer status bar with new token counts
        self._sync_status_to_footer()

        return self.add_message(msg)

    async def _check_for_escape(self) -> None:
        """Background task to poll for Escape key during API calls.

        Uses raw input polling to detect Escape during long operations.
        """
        if not self.input_controller.is_running():
            return

        while not self.cancel_token.is_cancelled:
            if not self.input_controller.is_running():
                break
            await self.input_controller.poll_escape(timeout=0.05)
            if self.cancel_token.is_cancelled:
                return

    async def _agent_loop_concurrent(self) -> None:
        """Run agent loop suitable for concurrent input mode.

        Uses footer status bar for spinner during API calls and tool execution.
        Output is printed directly and appears above the prompt.

        Implements checkpoint-based rollback: saves DAG state before each API call
        so user can choose to undo if they cancel during tool execution.
        """
        if not self.api or not self.dag:
            return

        # Sync initial status to footer
        self._sync_status_to_footer()

        while True:
            # CHECKPOINT: Save DAG state before API call for potential rollback
            checkpoint = self.dag

            # API call with activity indicator and escape detection
            escape_task = asyncio.create_task(self._check_for_escape())
            try:
                async with self._state_activity(AppState.THINKING, "Thinking..."):
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
                self._set_state(AppState.IDLE)
                break

            # Execute tools with tracking (handles cancellation and user choices)
            self._set_state(AppState.EXECUTING_TOOL)
            completed = await self._execute_tools_with_tracking(tool_calls, checkpoint)

            # If user chose to undo all, exit loop - they need to provide new input
            if not completed:
                break

    async def _agent_loop(self) -> None:
        """Run agent loop with cancellation support.

        Continuously sends requests to API and executes tool calls until
        no more tool calls are returned or operation is cancelled.

        Uses InputController to detect Escape key for cancellation.
        Shows unified footer status bar with spinner, auto-accept status, and token counts.

        Implements checkpoint-based rollback: saves DAG state before each API call
        so user can choose to undo if they cancel during tool execution.
        """
        if not self.api or not self.dag:
            return

        # Start input controller to detect Escape key during execution
        async with self.input_controller:
            # Sync initial status to footer
            self._sync_status_to_footer()

            while True:
                # CHECKPOINT: Save DAG state before API call for potential rollback
                checkpoint = self.dag

                # API call with activity indicator and escape detection
                escape_task = asyncio.create_task(self._check_for_escape())
                try:
                    async with self._state_activity(AppState.THINKING, "Thinking..."):
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
                    self._set_state(AppState.IDLE)
                    break

                # Execute tools with tracking (handles cancellation and user choices)
                self._set_state(AppState.EXECUTING_TOOL)
                completed = await self._execute_tools_with_tracking(
                    tool_calls, checkpoint
                )

                # If user chose to undo all, exit loop - they need to provide new input
                if not completed:
                    break

    async def _run_agent_with_error_handling(self) -> None:
        """Run agent loop with error handling, auto-save, and visual formatting."""
        try:
            await self._agent_loop()
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.cancel_token.cancel()
            if self.dag:
                self.dag = self.dag.user("[Operation cancelled by user]")
            self.add_message(create_error_message("Operation cancelled."))
        except Exception as e:
            self.add_message(create_error_message(f"API error: {e}"))

        self.print_blank()
        # Fire-and-forget background auto-save
        asyncio.create_task(self._auto_save_async())

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

    async def _run_agent_concurrent(self) -> None:
        """Run agent loop with error handling (concurrent mode).

        Uses line-based status instead of Rich Live to allow concurrent input.
        """
        try:
            async with self.input_controller:
                await self._agent_loop_concurrent()
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.cancel_token.cancel()
            if self.dag:
                self.dag = self.dag.user("[Operation cancelled by user]")
            self.add_message(create_error_message("Operation cancelled."))
        except Exception as e:
            self.add_message(create_error_message(f"API error: {e}"))

        self.print_blank()
        # Fire-and-forget background auto-save
        asyncio.create_task(self._auto_save_async())

    async def _process_queue(self) -> None:
        """Process all queued messages (background execution task)."""
        while not self.pending_messages.empty():
            text = await self.pending_messages.get()
            if not self.dag:
                self.add_message(create_error_message("DAG not initialized"))
                continue
            self.cancel_token.reset()
            self.add_message(create_user_message(text))
            self.dag = self.dag.user(text)
            await self._run_agent_concurrent()

    def _is_execution_running(self) -> bool:
        """Check if execution task is currently running."""
        return self._execution_task is not None and not self._execution_task.done()

    async def _wait_for_execution(self) -> None:
        """Wait for current execution to complete."""
        if self._execution_task is not None:
            try:
                await self._execution_task
            except Exception:
                pass  # Errors already handled in _run_agent_concurrent
            self._execution_task = None

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.api and hasattr(self.api, "client"):
            await self.api.client.aclose()
        if self.api and hasattr(self.api, "_client"):
            await self.api._client.aclose()

    async def run(self) -> None:
        """Main application loop."""
        try:
            # Start input controller first to enter raw mode
            # This ensures consistent terminal behavior throughout the app
            await self.input_controller.start()

            self._set_bookmark()

            # Add welcome message
            self.add_message(create_welcome_message())

            # Initialize API
            if not await self.initialize_api():
                return

            # Load session if --continue was specified
            if self.continue_session:
                try:
                    loaded_dag, metadata, filepath = await self.session_store.load(
                        self.continue_session
                    )
                    if self.tools:
                        loaded_dag = loaded_dag.tools(*self.tools)
                    self.dag = loaded_dag
                    session_id = metadata.get("session_id", "unknown")
                    node_count = len(metadata.get("nodes", {}))
                    self.render_history()
                    self.add_message(
                        create_system_message(
                            f"Continuing session from: {filepath}\n"
                            f"  Session ID: {session_id}\n"
                            f"  Nodes: {node_count}"
                        )
                    )
                except FileNotFoundError as e:
                    self.add_message(create_error_message(str(e)))
                except Exception as e:
                    self.add_message(
                        create_error_message(f"Failed to load session: {e}")
                    )

            # Main input loop (prompt only when execution is idle)
            self._set_state(AppState.IDLE)
            while True:
                # If there's a pending execution and it's done, clean up
                if self._execution_task is not None and self._execution_task.done():
                    try:
                        await self._execution_task
                    except Exception:
                        pass  # Errors already handled
                    self._execution_task = None

                # Wait for any running execution before showing prompt
                if self._is_execution_running():
                    await self._wait_for_execution()

                # Get user input
                user_input = await self.get_user_input()

                if user_input is None:
                    # User pressed Ctrl+D - wait for execution to finish first
                    await self._wait_for_execution()
                    self._set_state(AppState.SHUTDOWN)
                    self.print_blank()
                    self.add_message(create_system_message("Goodbye!"))
                    break

                if not user_input:
                    continue

                # For slash commands, clear stale footer content immediately.
                # (User messages use the seamless transition in add_message() instead.)
                if user_input.startswith("/"):
                    self.input_controller.footer.clear_content()
                    self.add_message(create_command_message(user_input))

                # Handle async commands (need special handling)
                cmd = user_input.lower().strip()
                if cmd in ("/continue", "/c"):
                    # Wait for any running execution first
                    await self._wait_for_execution()
                    await self.continue_agent()
                    continue

                # Handle slash commands - wait for execution first
                if user_input.startswith("/"):
                    await self._wait_for_execution()
                    self._set_state(AppState.COMMAND)
                    if not await self.handle_command(user_input):
                        self._set_state(AppState.SHUTDOWN)
                        self.add_message(create_system_message("Goodbye!"))
                        break
                    self._set_state(AppState.IDLE)
                    continue

                # Queue message for processing
                await self.pending_messages.put(user_input)

                # Start execution task if not already running
                if not self._is_execution_running():
                    self._execution_task = asyncio.create_task(self._process_queue())
        finally:
            # Wait for execution to complete before cleanup
            await self._wait_for_execution()
            await self.input_controller.stop()
            await self.cleanup()


def main() -> None:
    """Entry point for the TUI application."""
    import argparse

    parser = argparse.ArgumentParser(description="nano-cli")
    parser.add_argument(
        "--codex",
        metavar="MODEL",
        nargs="?",
        const="gpt-5.2-codex",
        help="Use Codex ChatGPT OAuth API (default model: gpt-5.2-codex)",
    )
    parser.add_argument(
        "--gemini",
        metavar="MODEL",
        nargs="?",
        const="gemini-3-pro-preview",
        help="Use Gemini API instead of Claude (default model: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--fireworks",
        metavar="MODEL",
        nargs="?",
        const="accounts/fireworks/models/kimi-k2p5",
        help="Use Fireworks API (default model: accounts/fireworks/models/kimi-k2p5)",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["off", "low", "medium", "high"],
        default="low",
        help="Thinking/reasoning level for Gemini, Fireworks, OpenAI (default: low)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=16000,
        help="Claude thinking budget tokens (Claude Code only)",
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
    parser.add_argument(
        "--renew",
        action="store_true",
        help="Refresh OAuth token and exit (for Claude Code API)",
    )
    args = parser.parse_args()

    # Handle --renew: refresh token and exit immediately
    if args.renew:
        from nano_agent.providers.capture_claude_code_auth import async_get_config

        async def do_renew() -> int:
            try:
                print("Refreshing OAuth token...")
                await async_get_config(timeout=30)
                print("Token refreshed successfully.")
                return 0
            except TimeoutError:
                print("Error: Token refresh timed out. Is Claude CLI available?")
                return 1
            except RuntimeError as e:
                print(f"Error: Token refresh failed: {e}")
                return 1

        sys.exit(asyncio.run(do_renew()))

    use_codex = args.codex is not None
    use_gemini = args.gemini is not None and not use_codex
    use_fireworks = args.fireworks is not None and not use_codex and not use_gemini

    config = load_cli_config()
    initial_scheme = str(config.get("color_scheme", "dark"))

    app = TerminalApp(
        use_codex=use_codex,
        codex_model=args.codex or "gpt-5.2-codex",
        use_gemini=use_gemini,
        gemini_model=args.gemini or "gemini-3-pro-preview",
        use_fireworks=use_fireworks,
        fireworks_model=args.fireworks or "accounts/fireworks/models/kimi-k2p5",
        thinking_level=args.thinking_level,
        thinking_budget=args.thinking_budget,
        debug=args.debug,
        continue_session=args.continue_session,
        color_scheme=initial_scheme,
    )
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
