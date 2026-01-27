"""Tests for CLI application."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.app import SESSION_FILE, TerminalApp, build_system_prompt
from nano_agent import DAG


class TestBuildSystemPrompt:
    """Tests for the build_system_prompt function."""

    def test_returns_tuple(self) -> None:
        """Test that build_system_prompt returns a tuple of (prompt, claude_md_loaded)."""
        result = build_system_prompt("test-model")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)

    def test_includes_model_in_prompt(self) -> None:
        """Test that the model name is included in the prompt."""
        prompt, _ = build_system_prompt("claude-3-opus")
        assert "claude-3-opus" in prompt

    def test_includes_working_directory(self) -> None:
        """Test that the working directory is included in the prompt."""
        prompt, _ = build_system_prompt("test-model")
        assert "Working directory:" in prompt

    def test_includes_current_time(self) -> None:
        """Test that current time is included in the prompt."""
        prompt, _ = build_system_prompt("test-model")
        assert "Current time:" in prompt


class TestTerminalAppAutoSave:
    """Tests for auto-save functionality."""

    def test_auto_save_creates_file(self, tmp_path: Path) -> None:
        """Test that _auto_save creates a session file."""
        app = TerminalApp()
        app.dag = DAG().system("test").user("hello")
        
        with patch("cli.app.Path.cwd", return_value=tmp_path):
            app._auto_save()
        
        session_file = tmp_path / SESSION_FILE
        assert session_file.exists()
        
        # Verify it's valid JSON
        with open(session_file) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "session_id" in data

    def test_auto_save_does_nothing_without_dag(self, tmp_path: Path) -> None:
        """Test that _auto_save does nothing when dag is None."""
        app = TerminalApp()
        app.dag = None
        
        with patch("cli.app.Path.cwd", return_value=tmp_path):
            app._auto_save()
        
        session_file = tmp_path / SESSION_FILE
        assert not session_file.exists()

    def test_auto_save_does_nothing_with_empty_dag(self, tmp_path: Path) -> None:
        """Test that _auto_save does nothing when dag has no heads."""
        app = TerminalApp()
        app.dag = DAG()  # Empty DAG
        
        with patch("cli.app.Path.cwd", return_value=tmp_path):
            app._auto_save()
        
        session_file = tmp_path / SESSION_FILE
        assert not session_file.exists()

    def test_auto_save_ignores_errors(self, tmp_path: Path) -> None:
        """Test that _auto_save silently ignores errors."""
        app = TerminalApp()
        app.dag = DAG().system("test").user("hello")
        
        # Use a path that will cause an error (directory doesn't exist)
        with patch("cli.app.Path.cwd", return_value=Path("/nonexistent/path")):
            # Should not raise
            app._auto_save()


class TestTerminalAppCommands:
    """Tests for command handling."""

    def test_handle_quit_commands(self) -> None:
        """Test that quit commands return False to signal exit."""
        app = TerminalApp()
        app.console = MagicMock()
        
        assert app.handle_command("/quit") is False
        assert app.handle_command("/exit") is False
        assert app.handle_command("/q") is False

    def test_handle_clear_command(self) -> None:
        """Test the /clear command."""
        app = TerminalApp()
        app.console = MagicMock()
        app.dag = DAG().system("old").user("message")
        app.api = MagicMock()
        app.api.model = "test-model"
        
        result = app.handle_command("/clear")
        
        assert result is True
        # DAG should be reset
        assert app.dag.get_system_prompt() != "old"
        # Should have new system prompt
        assert "test-model" in app.dag.get_system_prompt()

    def test_handle_render_command(self) -> None:
        """Test the /render command."""
        app = TerminalApp()
        app.console = MagicMock()
        app.dag = DAG().system("test")
        
        result = app.handle_command("/render")
        
        assert result is True

    def test_handle_help_command(self) -> None:
        """Test the /help command."""
        app = TerminalApp()
        app.console = MagicMock()
        
        result = app.handle_command("/help")
        
        assert result is True
        # Verify help was printed
        app.console.print.assert_called()

    def test_handle_debug_command_with_dag(self) -> None:
        """Test the /debug command with a DAG."""
        app = TerminalApp()
        app.console = MagicMock()
        app.dag = DAG().system("test").user("hello")
        
        result = app.handle_command("/debug")
        
        assert result is True
        app.console.print.assert_called()

    def test_handle_debug_command_without_dag(self) -> None:
        """Test the /debug command without a DAG."""
        app = TerminalApp()
        app.console = MagicMock()
        app.dag = DAG()  # Empty DAG
        
        result = app.handle_command("/debug")
        
        assert result is True

    def test_handle_save_command(self, tmp_path: Path) -> None:
        """Test the /save command."""
        app = TerminalApp()
        app.console = MagicMock()
        app.dag = DAG().system("test").user("hello")
        
        save_path = tmp_path / "test_session.json"
        result = app.handle_command(f"/save {save_path}")
        
        assert result is True
        assert save_path.exists()

    def test_handle_load_command(self, tmp_path: Path) -> None:
        """Test the /load command."""
        # Create a session file
        original_dag = DAG().system("loaded system").user("loaded message")
        save_path = tmp_path / "test_session.json"
        original_dag.save(save_path)
        
        app = TerminalApp()
        app.console = MagicMock()
        app.dag = DAG().system("old")
        
        result = app.handle_command(f"/load {save_path}")
        
        assert result is True
        assert app.dag.get_system_prompt() == "loaded system"

    def test_handle_unknown_command(self) -> None:
        """Test handling of unknown commands."""
        app = TerminalApp()
        app.console = MagicMock()
        
        result = app.handle_command("/unknown_command")
        
        assert result is True  # Unknown commands don't exit
        # Should print an error message
        app.console.print.assert_called()

    def test_continue_command_in_help(self) -> None:
        """Test that /continue command is documented in help."""
        app = TerminalApp()
        app.console = MagicMock()
        
        app.handle_command("/help")
        
        # Get the printed content
        call_args = app.console.print.call_args_list
        help_printed = False
        for call in call_args:
            args = call[0]
            if args and "/continue" in str(args[0]):
                help_printed = True
                break
        assert help_printed, "/continue should be in help text"


class TestTerminalAppContinue:
    """Tests for the /continue command functionality."""

    @pytest.mark.asyncio
    async def test_continue_agent_requires_api(self) -> None:
        """Test that continue_agent does nothing without API."""
        app = TerminalApp()
        app.console = MagicMock()
        app.api = None
        app.dag = DAG().system("test")
        
        await app.continue_agent()
        
        # Should print error message
        app.console.print.assert_called()

    @pytest.mark.asyncio
    async def test_continue_agent_requires_dag(self) -> None:
        """Test that continue_agent does nothing without DAG."""
        app = TerminalApp()
        app.console = MagicMock()
        app.api = MagicMock()
        app.dag = None
        
        await app.continue_agent()
        
        # Should print error message
        app.console.print.assert_called()


class TestSessionFile:
    """Tests for session file constants and behavior."""

    def test_session_file_is_hidden(self) -> None:
        """Test that SESSION_FILE is a hidden file (starts with dot)."""
        assert SESSION_FILE.startswith(".")

    def test_session_file_is_json(self) -> None:
        """Test that SESSION_FILE has .json extension."""
        assert SESSION_FILE.endswith(".json")
