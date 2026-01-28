"""Tmux tool for managing tmux sessions, windows, and panes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

from ..data_structures import TextContent
from .base import Desc, Tool, TruncationConfig


@dataclass
class TmuxInput:
    """Input for TmuxTool."""

    operation: Annotated[
        str,
        Desc(
            "The operation to perform: 'list_sessions', 'new_session', 'send_keys', "
            "'capture_pane', 'kill_session', 'list_windows', 'new_window', 'split_pane'"
        ),
    ]
    session_name: Annotated[
        str, Desc("Session name for operations (required for most operations)")
    ] = ""
    window_name: Annotated[
        str, Desc("Window name (optional, for new_session, new_window, or targeting)")
    ] = ""
    window_index: Annotated[
        int, Desc("Window index to target (0-based, default 0)")
    ] = 0
    pane_index: Annotated[
        int, Desc("Pane index to target (0-based, default 0)")
    ] = 0
    keys: Annotated[
        str, Desc("Keys/command to send (for send_keys operation)")
    ] = ""
    enter: Annotated[
        bool, Desc("Whether to press Enter after sending keys (default True)")
    ] = True
    capture_lines: Annotated[
        int, Desc("Number of lines to capture from pane (default 100, -1 for all)")
    ] = 100
    vertical: Annotated[
        bool, Desc("Split pane vertically (default False = horizontal)")
    ] = False


@dataclass
class TmuxTool(Tool):
    """Manage tmux sessions, windows, and panes programmatically.

    This tool provides control over tmux terminal multiplexer sessions,
    allowing you to create sessions, send commands, and capture output.

    Operations:
    - list_sessions: List all active tmux sessions
    - new_session: Create a new tmux session
    - send_keys: Send keys/commands to a pane
    - capture_pane: Capture the content of a pane
    - kill_session: Kill/close a session
    - list_windows: List windows in a session
    - new_window: Create a new window in a session
    - split_pane: Split a pane horizontally or vertically

    Example usage:
        tool = TmuxTool()

        # List sessions
        result = await tool.execute({"operation": "list_sessions"})

        # Create a new session
        result = await tool.execute({
            "operation": "new_session",
            "session_name": "dev",
            "window_name": "editor"
        })

        # Send a command
        result = await tool.execute({
            "operation": "send_keys",
            "session_name": "dev",
            "keys": "ls -la"
        })

        # Capture output
        result = await tool.execute({
            "operation": "capture_pane",
            "session_name": "dev"
        })
    """

    name: str = "Tmux"
    description: str = """Manage tmux sessions, windows, and panes programmatically.

Use this tool to control tmux terminal multiplexer for running long-lived processes,
managing multiple terminal sessions, or executing commands in isolated environments.

Operations:
- list_sessions: List all active tmux sessions
  Returns: List of session names and their details

- new_session: Create a new tmux session
  Required: session_name
  Optional: window_name (name for the initial window)
  Returns: Confirmation with session details

- send_keys: Send keys/commands to a pane
  Required: session_name, keys
  Optional: window_index (default 0), pane_index (default 0), enter (default True)
  Returns: Confirmation

- capture_pane: Capture the content/output of a pane
  Required: session_name
  Optional: window_index (default 0), pane_index (default 0), capture_lines (default 100)
  Returns: The captured pane content

- kill_session: Kill/terminate a session
  Required: session_name
  Returns: Confirmation

- list_windows: List all windows in a session
  Required: session_name
  Returns: List of windows with indices and names

- new_window: Create a new window in a session
  Required: session_name
  Optional: window_name
  Returns: Confirmation with window details

- split_pane: Split a pane to create a new pane
  Required: session_name
  Optional: window_index, pane_index, vertical (default False = horizontal split)
  Returns: Confirmation with new pane details

Examples:
  # List all sessions
  {"operation": "list_sessions"}

  # Create session named "work"
  {"operation": "new_session", "session_name": "work", "window_name": "main"}

  # Run a command in the session
  {"operation": "send_keys", "session_name": "work", "keys": "python server.py"}

  # Check output
  {"operation": "capture_pane", "session_name": "work", "capture_lines": 50}

  # Split the pane vertically
  {"operation": "split_pane", "session_name": "work", "vertical": true}

Note: Requires tmux to be installed and libtmux Python package."""

    # Truncation config for captured pane output
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(
        max_chars=30000, max_lines=500, enabled=True
    )

    # Require tmux to be installed
    _required_commands: ClassVar[dict[str, str]] = {
        "tmux": "Install with: brew install tmux (macOS) or apt install tmux (Linux)"
    }

    async def __call__(self, input: TmuxInput) -> TextContent:
        """Execute a tmux operation."""
        try:
            import libtmux
        except ImportError:
            return TextContent(
                text="Error: libtmux is not installed. Install with: pip install libtmux"
            )

        operation = input.operation.lower().strip()

        try:
            server = libtmux.Server()

            if operation == "list_sessions":
                return self._list_sessions(server)
            elif operation == "new_session":
                return self._new_session(server, input)
            elif operation == "send_keys":
                return self._send_keys(server, input)
            elif operation == "capture_pane":
                return self._capture_pane(server, input)
            elif operation == "kill_session":
                return self._kill_session(server, input)
            elif operation == "list_windows":
                return self._list_windows(server, input)
            elif operation == "new_window":
                return self._new_window(server, input)
            elif operation == "split_pane":
                return self._split_pane(server, input)
            else:
                return TextContent(
                    text=f"Error: Unknown operation '{operation}'. "
                    f"Valid operations: list_sessions, new_session, send_keys, "
                    f"capture_pane, kill_session, list_windows, new_window, split_pane"
                )

        except Exception as e:
            return TextContent(text=f"Error: {type(e).__name__}: {e}")

    def _list_sessions(self, server: "libtmux.Server") -> TextContent:
        """List all tmux sessions."""
        sessions = server.sessions
        if not sessions:
            return TextContent(text="No active tmux sessions.")

        lines = ["Active tmux sessions:", ""]
        for session in sessions:
            window_count = len(session.windows)
            lines.append(
                f"  • {session.name} (id: {session.id}, windows: {window_count})"
            )
        return TextContent(text="\n".join(lines))

    def _new_session(
        self, server: "libtmux.Server", input: TmuxInput
    ) -> TextContent:
        """Create a new tmux session."""
        if not input.session_name:
            return TextContent(text="Error: session_name is required for new_session")

        # Check if session already exists
        existing = server.sessions.filter(session_name=input.session_name)
        if existing:
            return TextContent(
                text=f"Error: Session '{input.session_name}' already exists"
            )

        # Create the session
        kwargs: dict[str, str | bool] = {
            "session_name": input.session_name,
            "attach": False,  # Don't attach to the session
        }
        if input.window_name:
            kwargs["window_name"] = input.window_name

        session = server.new_session(**kwargs)
        window = session.active_window
        pane = session.active_pane

        return TextContent(
            text=f"Created session '{session.name}' (id: {session.id})\n"
            f"  Window: {window.name} (index: {window.index})\n"
            f"  Pane: {pane.id}"
        )

    def _send_keys(
        self, server: "libtmux.Server", input: TmuxInput
    ) -> TextContent:
        """Send keys to a pane."""
        if not input.session_name:
            return TextContent(text="Error: session_name is required for send_keys")
        if not input.keys:
            return TextContent(text="Error: keys is required for send_keys")

        session = self._get_session(server, input.session_name)
        if isinstance(session, TextContent):
            return session

        pane = self._get_pane(session, input.window_index, input.pane_index)
        if isinstance(pane, TextContent):
            return pane

        pane.send_keys(input.keys, enter=input.enter)

        return TextContent(
            text=f"Sent keys to {session.name}:{input.window_index}.{input.pane_index}"
            + (f" (with Enter)" if input.enter else " (no Enter)")
        )

    def _capture_pane(
        self, server: "libtmux.Server", input: TmuxInput
    ) -> TextContent:
        """Capture content from a pane."""
        if not input.session_name:
            return TextContent(text="Error: session_name is required for capture_pane")

        session = self._get_session(server, input.session_name)
        if isinstance(session, TextContent):
            return session

        pane = self._get_pane(session, input.window_index, input.pane_index)
        if isinstance(pane, TextContent):
            return pane

        # Capture pane content
        if input.capture_lines == -1:
            # Capture all available history
            content = pane.capture_pane(start="-")
        else:
            # Capture last N lines
            start = -input.capture_lines if input.capture_lines > 0 else 0
            content = pane.capture_pane(start=start)

        # content is a list of strings (lines)
        if isinstance(content, list):
            text = "\n".join(content)
        else:
            text = str(content)

        # Strip trailing empty lines
        text = text.rstrip()

        if not text:
            return TextContent(
                text=f"(Pane {session.name}:{input.window_index}.{input.pane_index} is empty)"
            )

        return TextContent(text=text)

    def _kill_session(
        self, server: "libtmux.Server", input: TmuxInput
    ) -> TextContent:
        """Kill a tmux session."""
        if not input.session_name:
            return TextContent(text="Error: session_name is required for kill_session")

        session = self._get_session(server, input.session_name)
        if isinstance(session, TextContent):
            return session

        session_name = session.name
        session.kill()

        return TextContent(text=f"Killed session '{session_name}'")

    def _list_windows(
        self, server: "libtmux.Server", input: TmuxInput
    ) -> TextContent:
        """List windows in a session."""
        if not input.session_name:
            return TextContent(text="Error: session_name is required for list_windows")

        session = self._get_session(server, input.session_name)
        if isinstance(session, TextContent):
            return session

        windows = session.windows
        if not windows:
            return TextContent(text=f"No windows in session '{session.name}'")

        lines = [f"Windows in session '{session.name}':", ""]
        for window in windows:
            pane_count = len(window.panes)
            active = " (active)" if window == session.active_window else ""
            lines.append(
                f"  [{window.index}] {window.name} ({pane_count} panes){active}"
            )
        return TextContent(text="\n".join(lines))

    def _new_window(
        self, server: "libtmux.Server", input: TmuxInput
    ) -> TextContent:
        """Create a new window in a session."""
        if not input.session_name:
            return TextContent(text="Error: session_name is required for new_window")

        session = self._get_session(server, input.session_name)
        if isinstance(session, TextContent):
            return session

        kwargs: dict[str, str | bool] = {"attach": False}
        if input.window_name:
            kwargs["window_name"] = input.window_name

        window = session.new_window(**kwargs)

        return TextContent(
            text=f"Created window '{window.name}' (index: {window.index}) "
            f"in session '{session.name}'"
        )

    def _split_pane(
        self, server: "libtmux.Server", input: TmuxInput
    ) -> TextContent:
        """Split a pane."""
        if not input.session_name:
            return TextContent(text="Error: session_name is required for split_pane")

        session = self._get_session(server, input.session_name)
        if isinstance(session, TextContent):
            return session

        pane = self._get_pane(session, input.window_index, input.pane_index)
        if isinstance(pane, TextContent):
            return pane

        # Split the pane
        new_pane = pane.split(vertical=input.vertical)
        direction = "vertically" if input.vertical else "horizontally"

        return TextContent(
            text=f"Split pane {direction} in {session.name}:{input.window_index}\n"
            f"  New pane: {new_pane.id}"
        )

    def _get_session(
        self, server: "libtmux.Server", session_name: str
    ) -> "libtmux.Session | TextContent":
        """Get a session by name, returning error TextContent if not found."""
        sessions = server.sessions.filter(session_name=session_name)
        if not sessions:
            available = [s.name for s in server.sessions]
            if available:
                return TextContent(
                    text=f"Error: Session '{session_name}' not found. "
                    f"Available sessions: {', '.join(available)}"
                )
            return TextContent(
                text=f"Error: Session '{session_name}' not found. No active sessions."
            )
        return sessions[0]

    def _get_pane(
        self,
        session: "libtmux.Session",
        window_index: int,
        pane_index: int,
    ) -> "libtmux.Pane | TextContent":
        """Get a pane by window and pane index, returning error TextContent if not found."""
        windows = session.windows.filter(window_index=str(window_index))
        if not windows:
            available = [str(w.index) for w in session.windows]
            return TextContent(
                text=f"Error: Window index {window_index} not found. "
                f"Available indices: {', '.join(available)}"
            )

        window = windows[0]
        panes = window.panes

        if pane_index >= len(panes) or pane_index < 0:
            return TextContent(
                text=f"Error: Pane index {pane_index} not found. "
                f"Window has {len(panes)} pane(s) (indices 0-{len(panes) - 1})"
            )

        return panes[pane_index]
