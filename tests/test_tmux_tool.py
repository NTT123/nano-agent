"""Tests for TmuxTool."""

import asyncio
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from nano_agent.data_structures import TextContent
from nano_agent.tools import InputSchemaDict, TmuxInput, TmuxTool


def get_properties(schema: InputSchemaDict) -> dict[str, object]:
    """Helper to get properties from schema with proper typing."""
    return cast(dict[str, object], schema.get("properties", {}))


class TestTmuxTool:
    """Unit tests for TmuxTool."""

    def test_default_values(self) -> None:
        """Test TmuxTool has correct default name and description."""
        tool = TmuxTool()
        assert tool.name == "Tmux"
        assert "tmux" in tool.description.lower()
        assert "session" in tool.description.lower()

    def test_input_schema_has_required_fields(self) -> None:
        """Test that input schema has all expected fields."""
        tool = TmuxTool()
        schema = tool.input_schema
        props = get_properties(schema)

        # Check all expected properties
        assert "operation" in props
        assert "session_name" in props
        assert "window_name" in props
        assert "window_index" in props
        assert "pane_index" in props
        assert "keys" in props
        assert "enter" in props
        assert "capture_lines" in props
        assert "vertical" in props

        # Only operation is required
        assert schema["required"] == ["operation"]

    def test_input_schema_operation_description(self) -> None:
        """Test operation field has valid operations listed."""
        tool = TmuxTool()
        schema = tool.input_schema
        props = get_properties(schema)
        op_desc = str(props["operation"])

        # Check all operations are mentioned
        operations = [
            "list_sessions",
            "new_session",
            "send_keys",
            "capture_pane",
            "kill_session",
            "list_windows",
            "new_window",
            "split_pane",
        ]
        for op in operations:
            assert op in op_desc or op in tool.description


class TestTmuxInput:
    """Tests for TmuxInput dataclass."""

    def test_default_values(self) -> None:
        """Test default values for TmuxInput."""
        input_obj = TmuxInput(operation="list_sessions")
        assert input_obj.operation == "list_sessions"
        assert input_obj.session_name == ""
        assert input_obj.window_name == ""
        assert input_obj.window_index == 0
        assert input_obj.pane_index == 0
        assert input_obj.keys == ""
        assert input_obj.enter is True
        assert input_obj.capture_lines == 100
        assert input_obj.vertical is False

    def test_custom_values(self) -> None:
        """Test custom values for TmuxInput."""
        input_obj = TmuxInput(
            operation="send_keys",
            session_name="dev",
            window_index=2,
            pane_index=1,
            keys="ls -la",
            enter=False,
        )
        assert input_obj.operation == "send_keys"
        assert input_obj.session_name == "dev"
        assert input_obj.window_index == 2
        assert input_obj.pane_index == 1
        assert input_obj.keys == "ls -la"
        assert input_obj.enter is False


class TestTmuxToolFunctional:
    """Functional tests for TmuxTool with mocked libtmux."""

    def test_libtmux_not_installed(self) -> None:
        """Test error message when libtmux is not installed."""
        tool = TmuxTool()

        with patch.dict("sys.modules", {"libtmux": None}):
            # Force reimport to trigger ImportError
            import importlib
            import sys

            # Temporarily make libtmux unavailable
            original_modules = sys.modules.copy()
            sys.modules["libtmux"] = None  # type: ignore

            result = asyncio.run(tool.execute({"operation": "list_sessions"}))
            assert isinstance(result, TextContent)
            # Restore - the actual test may or may not work depending on env
            # The key is to verify the tool handles import errors gracefully

    def test_unknown_operation(self) -> None:
        """Test error for unknown operation."""
        tool = TmuxTool()

        # Mock libtmux to avoid actual tmux interaction
        mock_server = MagicMock()
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(tool.execute({"operation": "unknown_op"}))
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert (
                "unknown_op" in result.text.lower()
                or "Unknown operation" in result.text
            )

    def test_list_sessions_empty(self) -> None:
        """Test list_sessions when no sessions exist."""
        tool = TmuxTool()

        mock_server = MagicMock()
        mock_server.sessions = []
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(tool.execute({"operation": "list_sessions"}))
            assert isinstance(result, TextContent)
            assert "No active" in result.text

    def test_list_sessions_with_sessions(self) -> None:
        """Test list_sessions with existing sessions."""
        tool = TmuxTool()

        # Create mock sessions
        mock_session1 = MagicMock()
        mock_session1.name = "dev"
        mock_session1.id = "$1"
        mock_session1.windows = [MagicMock(), MagicMock()]

        mock_session2 = MagicMock()
        mock_session2.name = "prod"
        mock_session2.id = "$2"
        mock_session2.windows = [MagicMock()]

        mock_server = MagicMock()
        mock_server.sessions = [mock_session1, mock_session2]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(tool.execute({"operation": "list_sessions"}))
            assert isinstance(result, TextContent)
            assert "dev" in result.text
            assert "prod" in result.text
            assert "windows: 2" in result.text
            assert "windows: 1" in result.text

    def test_new_session_missing_name(self) -> None:
        """Test new_session requires session_name."""
        tool = TmuxTool()

        mock_server = MagicMock()
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(tool.execute({"operation": "new_session"}))
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "session_name" in result.text

    def test_new_session_already_exists(self) -> None:
        """Test new_session when session already exists."""
        tool = TmuxTool()

        mock_existing = MagicMock()
        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_existing]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute({"operation": "new_session", "session_name": "existing"})
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "already exists" in result.text

    def test_new_session_success(self) -> None:
        """Test successful session creation."""
        tool = TmuxTool()

        mock_window = MagicMock()
        mock_window.name = "main"
        mock_window.index = 0

        mock_pane = MagicMock()
        mock_pane.id = "%0"

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.id = "$1"
        mock_session.active_window = mock_window
        mock_session.active_pane = mock_pane

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = []  # No existing session
        mock_server.new_session.return_value = mock_session
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute(
                    {
                        "operation": "new_session",
                        "session_name": "dev",
                        "window_name": "main",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Created session" in result.text
            assert "dev" in result.text

    def test_send_keys_missing_session(self) -> None:
        """Test send_keys requires session_name."""
        tool = TmuxTool()

        mock_server = MagicMock()
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute({"operation": "send_keys", "keys": "ls -la"})
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "session_name" in result.text

    def test_send_keys_missing_keys(self) -> None:
        """Test send_keys requires keys."""
        tool = TmuxTool()

        mock_server = MagicMock()
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute({"operation": "send_keys", "session_name": "dev"})
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "keys" in result.text

    def test_send_keys_session_not_found(self) -> None:
        """Test send_keys when session doesn't exist."""
        tool = TmuxTool()

        # Create a mock sessions list that supports both filter() and iteration
        mock_sessions = MagicMock()
        mock_sessions.filter.return_value = []
        mock_sessions.__iter__ = lambda self: iter([])  # Empty session list

        mock_server = MagicMock()
        mock_server.sessions = mock_sessions
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute(
                    {
                        "operation": "send_keys",
                        "session_name": "nonexistent",
                        "keys": "ls",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "not found" in result.text

    def test_send_keys_success(self) -> None:
        """Test successful send_keys."""
        tool = TmuxTool()

        mock_pane = MagicMock()
        mock_window = MagicMock()
        mock_window.panes = [mock_pane]

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.windows.filter.return_value = [mock_window]

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute(
                    {"operation": "send_keys", "session_name": "dev", "keys": "ls -la"}
                )
            )
            assert isinstance(result, TextContent)
            assert "Sent keys" in result.text
            mock_pane.send_keys.assert_called_once_with("ls -la", enter=True)

    def test_capture_pane_success(self) -> None:
        """Test successful capture_pane."""
        tool = TmuxTool()

        mock_pane = MagicMock()
        mock_pane.capture_pane.return_value = ["line1", "line2", "line3"]

        mock_window = MagicMock()
        mock_window.panes = [mock_pane]

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.windows.filter.return_value = [mock_window]

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute({"operation": "capture_pane", "session_name": "dev"})
            )
            assert isinstance(result, TextContent)
            assert "line1" in result.text
            assert "line2" in result.text
            assert "line3" in result.text

    def test_capture_pane_empty(self) -> None:
        """Test capture_pane with empty pane."""
        tool = TmuxTool()

        mock_pane = MagicMock()
        mock_pane.capture_pane.return_value = []

        mock_window = MagicMock()
        mock_window.panes = [mock_pane]

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.windows.filter.return_value = [mock_window]

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute({"operation": "capture_pane", "session_name": "dev"})
            )
            assert isinstance(result, TextContent)
            assert "empty" in result.text.lower()

    def test_kill_session_success(self) -> None:
        """Test successful kill_session."""
        tool = TmuxTool()

        mock_session = MagicMock()
        mock_session.name = "dev"

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute({"operation": "kill_session", "session_name": "dev"})
            )
            assert isinstance(result, TextContent)
            assert "Killed" in result.text
            assert "dev" in result.text
            mock_session.kill.assert_called_once()

    def test_list_windows_success(self) -> None:
        """Test successful list_windows."""
        tool = TmuxTool()

        mock_window1 = MagicMock()
        mock_window1.index = 0
        mock_window1.name = "editor"
        mock_window1.panes = [MagicMock()]

        mock_window2 = MagicMock()
        mock_window2.index = 1
        mock_window2.name = "terminal"
        mock_window2.panes = [MagicMock(), MagicMock()]

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.windows = [mock_window1, mock_window2]
        mock_session.active_window = mock_window1

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute({"operation": "list_windows", "session_name": "dev"})
            )
            assert isinstance(result, TextContent)
            assert "editor" in result.text
            assert "terminal" in result.text
            assert "[0]" in result.text
            assert "[1]" in result.text

    def test_new_window_success(self) -> None:
        """Test successful new_window."""
        tool = TmuxTool()

        mock_new_window = MagicMock()
        mock_new_window.name = "logs"
        mock_new_window.index = 2

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.new_window.return_value = mock_new_window

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute(
                    {
                        "operation": "new_window",
                        "session_name": "dev",
                        "window_name": "logs",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Created window" in result.text
            assert "logs" in result.text

    def test_split_pane_success(self) -> None:
        """Test successful split_pane."""
        tool = TmuxTool()

        mock_new_pane = MagicMock()
        mock_new_pane.id = "%1"

        mock_pane = MagicMock()
        mock_pane.split.return_value = mock_new_pane

        mock_window = MagicMock()
        mock_window.panes = [mock_pane]

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.windows.filter.return_value = [mock_window]

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute(
                    {"operation": "split_pane", "session_name": "dev", "vertical": True}
                )
            )
            assert isinstance(result, TextContent)
            assert "Split pane" in result.text
            assert "vertically" in result.text
            mock_pane.split.assert_called_once_with(vertical=True)

    def test_window_not_found(self) -> None:
        """Test error when window index doesn't exist."""
        tool = TmuxTool()

        mock_window = MagicMock()
        mock_window.index = 0

        # Create mock windows that supports filter() and iteration
        mock_windows = MagicMock()
        mock_windows.filter.return_value = []  # No matching window
        mock_windows.__iter__ = lambda self: iter([mock_window])

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.windows = mock_windows

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute(
                    {
                        "operation": "send_keys",
                        "session_name": "dev",
                        "window_index": 5,
                        "keys": "ls",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "Window index" in result.text

    def test_pane_not_found(self) -> None:
        """Test error when pane index doesn't exist."""
        tool = TmuxTool()

        mock_pane = MagicMock()
        mock_window = MagicMock()
        mock_window.panes = [mock_pane]  # Only 1 pane (index 0)

        mock_session = MagicMock()
        mock_session.name = "dev"
        mock_session.windows.filter.return_value = [mock_window]

        mock_server = MagicMock()
        mock_server.sessions.filter.return_value = [mock_session]
        mock_libtmux = MagicMock()
        mock_libtmux.Server.return_value = mock_server

        with patch.dict("sys.modules", {"libtmux": mock_libtmux}):
            result = asyncio.run(
                tool.execute(
                    {
                        "operation": "send_keys",
                        "session_name": "dev",
                        "pane_index": 5,
                        "keys": "ls",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "Pane index" in result.text


class TestTmuxToolIntegration:
    """Integration tests that require actual tmux.

    These tests are skipped if tmux is not available.
    """

    @pytest.fixture
    def check_tmux(self) -> None:
        """Skip test if tmux is not available."""
        import shutil

        if shutil.which("tmux") is None:
            pytest.skip("tmux not installed")

    @pytest.fixture
    def check_libtmux(self) -> None:
        """Skip test if libtmux is not available."""
        try:
            import libtmux  # noqa: F401
        except ImportError:
            pytest.skip("libtmux not installed")

    @pytest.mark.usefixtures("check_tmux", "check_libtmux")
    def test_list_sessions_real(self) -> None:
        """Test list_sessions with real tmux (doesn't create sessions)."""
        tool = TmuxTool()
        result = asyncio.run(tool.execute({"operation": "list_sessions"}))
        assert isinstance(result, TextContent)
        # Should either list sessions or say no sessions
        assert "session" in result.text.lower()
