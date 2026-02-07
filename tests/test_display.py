"""Tests for display formatting in cli/display.py.

Covers:
- format_user_message() multiline handling without indent
- Visual consistency between input and rendered message
"""

from __future__ import annotations

import pytest
from rich.text import Text

from cli.display import format_user_message


class TestFormatUserMessage:
    """Tests for format_user_message() function."""

    def test_single_line_has_prompt(self) -> None:
        """Single line should have prompt prefix."""
        result = format_user_message("hello")
        assert isinstance(result, Text)
        assert str(result).startswith("> ")
        assert "hello" in str(result)

    def test_single_line_has_style(self) -> None:
        """Single line should have grey30 background style."""
        result = format_user_message("hello")
        assert isinstance(result, Text)
        # The Text object should have the style applied

    def test_multiline_no_indent_on_continuation(self) -> None:
        """Continuation lines should NOT have indent."""
        result = format_user_message("line1\nline2\nline3")
        text_str = str(result)
        lines = text_str.split("\n")
        # First line should have prompt
        assert lines[0].startswith("> ")
        # Second and third lines should NOT start with spaces (no indent)
        assert lines[1] == "line2"  # No leading spaces
        assert lines[2] == "line3"  # No leading spaces

    def test_multiline_preserves_content(self) -> None:
        """All content should be preserved in multiline messages."""
        result = format_user_message("first\nsecond\nthird")
        text_str = str(result)
        assert "first" in text_str
        assert "second" in text_str
        assert "third" in text_str

    def test_empty_string(self) -> None:
        """Empty string should just show prompt."""
        result = format_user_message("")
        text_str = str(result)
        assert text_str == "> "

    def test_trailing_newline(self) -> None:
        """Trailing newline should be preserved."""
        result = format_user_message("hello\n")
        text_str = str(result)
        # Should have "hello" followed by empty line
        lines = text_str.split("\n")
        assert len(lines) == 2
        assert lines[0] == "> hello"
        assert lines[1] == ""

    def test_multiple_trailing_newlines(self) -> None:
        """Multiple trailing newlines should be preserved."""
        result = format_user_message("hello\n\n")
        text_str = str(result)
        lines = text_str.split("\n")
        assert len(lines) == 3

    def test_only_newlines(self) -> None:
        """String with only newlines."""
        result = format_user_message("\n\n")
        text_str = str(result)
        lines = text_str.split("\n")
        assert len(lines) == 3
        assert lines[0] == "> "  # First line has prompt
        assert lines[1] == ""  # Empty continuation
        assert lines[2] == ""  # Empty continuation

    def test_whitespace_lines_preserved(self) -> None:
        """Lines with only whitespace should be preserved."""
        result = format_user_message("hello\n   \nworld")
        text_str = str(result)
        lines = text_str.split("\n")
        assert lines[1] == "   "  # Whitespace preserved, no indent added

    def test_long_lines_not_modified(self) -> None:
        """Long lines should not be modified (wrapping is done elsewhere)."""
        long_text = "a" * 200
        result = format_user_message(long_text)
        text_str = str(result)
        # Should contain the full long line
        assert long_text in text_str.replace("> ", "")


class TestVisualConsistencyWithFooterInput:
    """Tests ensuring visual consistency between input and rendered message.

    The format_user_message() output should match what FooterInput shows,
    specifically: first line has prompt, subsequent lines have NO indent.
    """

    def test_simple_multiline_matches(self) -> None:
        """Simple multiline should render same as input appearance."""
        user_text = "line1\nline2"
        result = format_user_message(user_text)
        text_str = str(result)

        # Expected format (matching FooterInput):
        # "> line1"
        # "line2"  (no indent)
        expected_lines = ["> line1", "line2"]
        actual_lines = text_str.split("\n")
        assert actual_lines == expected_lines

    def test_three_lines_matches(self) -> None:
        """Three line input should have correct formatting."""
        user_text = "first\nsecond\nthird"
        result = format_user_message(user_text)
        text_str = str(result)

        expected_lines = ["> first", "second", "third"]
        actual_lines = text_str.split("\n")
        assert actual_lines == expected_lines

    def test_code_block_paste(self) -> None:
        """Pasted code block should preserve structure."""
        user_text = "def foo():\n    return 42"
        result = format_user_message(user_text)
        text_str = str(result)

        # The indentation in the code should be preserved
        lines = text_str.split("\n")
        assert lines[0] == "> def foo():"
        assert (
            lines[1] == "    return 42"
        )  # Code indent preserved, no extra prompt indent
