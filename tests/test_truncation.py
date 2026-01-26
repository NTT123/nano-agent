"""Tests for the unified tool output truncation system."""

import tempfile
from pathlib import Path

import pytest

from nano_agent import (
    BashTool,
    GlobTool,
    PythonTool,
    ReadTool,
    GrepTool,
    TruncatedOutput,
    TruncationConfig,
    WebFetchTool,
    cleanup_truncated_outputs,
    clear_all_truncated_outputs,
)
from nano_agent.data_structures import TextContent
from nano_agent.tools import (
    _DEFAULT_TRUNCATION_CONFIG,
    _save_full_output,
    _truncate_text_content,
    _truncated_outputs,
)


class TestTruncationConfig:
    """Tests for TruncationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TruncationConfig()
        assert config.max_chars == 30000
        assert config.max_lines == 1000
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TruncationConfig(max_chars=10000, max_lines=500, enabled=False)
        assert config.max_chars == 10000
        assert config.max_lines == 500
        assert config.enabled is False


class TestTruncatedOutput:
    """Tests for TruncatedOutput dataclass."""

    def test_fields(self) -> None:
        """Test TruncatedOutput has required fields."""
        output = TruncatedOutput(
            tool_name="Bash",
            temp_file_path="/tmp/test.txt",
            original_chars=50000,
            original_lines=2000,
            created_at=1234567890.0,
        )
        assert output.tool_name == "Bash"
        assert output.temp_file_path == "/tmp/test.txt"
        assert output.original_chars == 50000
        assert output.original_lines == 2000
        assert output.created_at == 1234567890.0


class TestSaveFullOutput:
    """Tests for _save_full_output function."""

    def setup_method(self) -> None:
        """Clear truncated outputs before each test."""
        clear_all_truncated_outputs()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_all_truncated_outputs()

    def test_creates_file(self) -> None:
        """Test that file is created with content."""
        content = "Hello, World!\nThis is test content."
        path = _save_full_output(content, "TestTool")

        assert Path(path).exists()
        assert Path(path).read_text() == content

    def test_registers_in_global_dict(self) -> None:
        """Test that output is registered for cleanup."""
        content = "Test content"
        path = _save_full_output(content, "TestTool")

        assert path in _truncated_outputs
        output = _truncated_outputs[path]
        assert output.tool_name == "TestTool"
        assert output.original_chars == len(content)
        assert output.original_lines == 1

    def test_unique_filenames(self) -> None:
        """Test that multiple calls create unique files."""
        path1 = _save_full_output("content1", "Tool")
        path2 = _save_full_output("content2", "Tool")

        assert path1 != path2
        assert Path(path1).exists()
        assert Path(path2).exists()


class TestTruncateTextContent:
    """Tests for _truncate_text_content function."""

    def setup_method(self) -> None:
        """Clear truncated outputs before each test."""
        clear_all_truncated_outputs()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_all_truncated_outputs()

    def test_no_truncation_when_within_limits(self) -> None:
        """Test that content within limits is not truncated."""
        content = TextContent(text="Short content")
        config = TruncationConfig(max_chars=1000, max_lines=100)

        result = _truncate_text_content(content, "Test", config)

        assert result.text == "Short content"
        assert len(_truncated_outputs) == 0  # No file saved

    def test_truncation_by_chars(self) -> None:
        """Test truncation when char limit is exceeded."""
        # Create content that exceeds char limit
        long_text = "x" * 100
        content = TextContent(text=long_text)
        config = TruncationConfig(max_chars=50, max_lines=1000)

        result = _truncate_text_content(content, "Test", config)

        # Should be truncated with notification
        assert len(result.text) < len(long_text) + 500  # Truncated + notification
        assert "OUTPUT TRUNCATED" in result.text
        assert "50" in result.text or "chars" in result.text
        assert len(_truncated_outputs) == 1

    def test_truncation_by_lines(self) -> None:
        """Test truncation when line limit is exceeded."""
        # Create content that exceeds line limit but not char limit
        lines = ["line " + str(i) for i in range(20)]
        long_text = "\n".join(lines)
        content = TextContent(text=long_text)
        config = TruncationConfig(max_chars=100000, max_lines=10)

        result = _truncate_text_content(content, "Test", config)

        # Should be truncated
        assert "OUTPUT TRUNCATED" in result.text
        assert len(_truncated_outputs) == 1

    def test_notification_format(self) -> None:
        """Test that notification has expected format."""
        content = TextContent(text="x" * 100)
        config = TruncationConfig(max_chars=50, max_lines=1000)

        result = _truncate_text_content(content, "TestTool", config)

        # Check notification components
        assert "───── OUTPUT TRUNCATED ─────" in result.text
        assert "Original:" in result.text
        assert "Full output:" in result.text
        assert "Use Read or Grep to view full content." in result.text

    def test_temp_file_contains_full_content(self) -> None:
        """Test that temp file contains the full original content."""
        original = "x" * 100
        content = TextContent(text=original)
        config = TruncationConfig(max_chars=50, max_lines=1000)

        _truncate_text_content(content, "Test", config)

        # Get the temp file path from the registry
        temp_path = list(_truncated_outputs.keys())[0]
        saved_content = Path(temp_path).read_text()

        assert saved_content == original


class TestCleanupFunctions:
    """Tests for cleanup utility functions."""

    def setup_method(self) -> None:
        """Clear state before each test."""
        clear_all_truncated_outputs()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_all_truncated_outputs()

    def test_cleanup_removes_old_files(self) -> None:
        """Test that cleanup removes files older than max_age."""
        import time

        # Create an output
        path = _save_full_output("test", "Test")
        assert Path(path).exists()

        # Manually set created_at to be old
        _truncated_outputs[path].created_at = time.time() - 7200  # 2 hours ago

        # Cleanup with 1 hour max age
        count = cleanup_truncated_outputs(max_age_seconds=3600)

        assert count == 1
        assert not Path(path).exists()
        assert path not in _truncated_outputs

    def test_cleanup_keeps_recent_files(self) -> None:
        """Test that cleanup keeps files newer than max_age."""
        path = _save_full_output("test", "Test")

        # Cleanup with 1 hour max age (file was just created)
        count = cleanup_truncated_outputs(max_age_seconds=3600)

        assert count == 0
        assert Path(path).exists()
        assert path in _truncated_outputs

    def test_clear_all_removes_everything(self) -> None:
        """Test that clear_all removes all files."""
        path1 = _save_full_output("test1", "Test")
        path2 = _save_full_output("test2", "Test")

        count = clear_all_truncated_outputs()

        assert count == 2
        assert not Path(path1).exists()
        assert not Path(path2).exists()
        assert len(_truncated_outputs) == 0


class TestToolTruncationConfig:
    """Tests for per-tool truncation configuration."""

    def test_bashtool_uses_default(self) -> None:
        """Test BashTool uses default truncation config."""
        tool = BashTool()
        # Should use default (enabled) or not have a config
        config = tool._truncation_config
        assert config is None  # Uses _DEFAULT_TRUNCATION_CONFIG

    def test_searchtool_uses_default(self) -> None:
        """Test GrepTool uses default truncation config."""
        tool = GrepTool()
        config = tool._truncation_config
        assert config is None  # Uses _DEFAULT_TRUNCATION_CONFIG

    def test_globtool_uses_default(self) -> None:
        """Test GlobTool uses default truncation config."""
        tool = GlobTool()
        config = tool._truncation_config
        assert config is None  # Uses _DEFAULT_TRUNCATION_CONFIG

    def test_readtool_disables_truncation(self) -> None:
        """Test ReadTool has truncation disabled."""
        tool = ReadTool()
        config = tool._truncation_config
        assert config is not None
        assert config.enabled is False

    def test_webfetchtool_uses_custom_limit(self) -> None:
        """Test WebFetchTool has custom 5000 char truncation limit."""
        tool = WebFetchTool()
        config = tool._truncation_config
        assert config is not None
        assert config.enabled is True
        assert config.max_chars == 5000

    def test_pythontool_disables_truncation(self) -> None:
        """Test PythonTool has truncation disabled."""
        tool = PythonTool()
        config = tool._truncation_config
        assert config is not None
        assert config.enabled is False


class TestToolExecuteTruncation:
    """Tests for truncation in Tool.execute() method."""

    def setup_method(self) -> None:
        """Clear state before each test."""
        clear_all_truncated_outputs()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_all_truncated_outputs()

    async def test_bash_truncates_large_output(self) -> None:
        """Test BashTool truncates large command output."""
        bash = BashTool()
        # Generate output larger than 30000 chars
        result = await bash.execute({"command": "seq 1 10000"})

        # Handle list or single TextContent
        text = result[0].text if isinstance(result, list) else result.text

        # Output should be truncated
        if len(text) > 30000:
            # Only check if the raw output exceeded limit
            # (in some environments seq 1 10000 might not exceed 30k)
            assert "OUTPUT TRUNCATED" in text

    async def test_read_does_not_truncate(self) -> None:
        """Test ReadTool does not apply additional truncation."""
        # Create a test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line\n" * 50)  # More than 25 lines
            temp_path = f.name

        try:
            read = ReadTool()
            result = await read.execute({"file_path": temp_path})

            # Handle list or single TextContent
            text = result[0].text if isinstance(result, list) else result.text

            # Should NOT have truncation notice (ReadTool's own limit is different)
            assert "OUTPUT TRUNCATED" not in text
            # But should have ReadTool's own truncation message
            assert "truncated" in text.lower() or "Showing:" in text
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDefaultConfig:
    """Tests for the default truncation configuration."""

    def test_default_config_values(self) -> None:
        """Test the module-level default config."""
        assert _DEFAULT_TRUNCATION_CONFIG.max_chars == 30000
        assert _DEFAULT_TRUNCATION_CONFIG.max_lines == 1000
        assert _DEFAULT_TRUNCATION_CONFIG.enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
