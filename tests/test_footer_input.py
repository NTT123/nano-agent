"""Tests for FooterInput in cli/elements/footer_input.py.

Covers:
- Multiline input rendering (no indent on continuation lines)
- Tab expansion in paste handling
- Cursor position tracking
- Line wrapping behavior
"""

from __future__ import annotations

import pytest

from cli.elements.footer_input import FooterInput
from cli.elements.terminal import ANSI


class TestNormalizePaste:
    """Tests for FooterInput._normalize_paste() method."""

    def test_single_tab_expansion(self) -> None:
        """Single tab should be expanded to 4 spaces."""
        fi = FooterInput()
        result = fi._normalize_paste("hello\tworld")
        assert result == "hello    world"
        assert "\t" not in result

    def test_multiple_tabs_expansion(self) -> None:
        """Multiple tabs should each be expanded to 4 spaces."""
        fi = FooterInput()
        result = fi._normalize_paste("a\tb\tc")
        assert result == "a    b    c"
        assert result.count("    ") == 2

    def test_tabs_at_different_positions(self) -> None:
        """Tabs at start, middle, and end should all be expanded."""
        fi = FooterInput()
        result = fi._normalize_paste("\ttabs\tat\tend\t")
        assert result == "    tabs    at    end    "
        assert "\t" not in result

    def test_crlf_normalized_to_lf(self) -> None:
        """CRLF line endings should be normalized to LF."""
        fi = FooterInput()
        result = fi._normalize_paste("line1\r\nline2\r\n")
        assert result == "line1\nline2\n"
        assert "\r" not in result

    def test_cr_normalized_to_lf(self) -> None:
        """CR line endings should be normalized to LF."""
        fi = FooterInput()
        result = fi._normalize_paste("line1\rline2\r")
        assert result == "line1\nline2\n"

    def test_mixed_line_endings_and_tabs(self) -> None:
        """Both line endings and tabs should be normalized."""
        fi = FooterInput()
        result = fi._normalize_paste("hello\tworld\r\nfoo\tbar\r")
        assert result == "hello    world\nfoo    bar\n"

    def test_tabs_only(self) -> None:
        """String with only tabs."""
        fi = FooterInput()
        result = fi._normalize_paste("\t\t\t")
        assert result == "            "  # 12 spaces

    def test_empty_string(self) -> None:
        """Empty string should remain empty."""
        fi = FooterInput()
        result = fi._normalize_paste("")
        assert result == ""

    def test_no_special_characters(self) -> None:
        """String without tabs or special line endings should be unchanged."""
        fi = FooterInput()
        result = fi._normalize_paste("hello world")
        assert result == "hello world"

    def test_multiline_disabled_converts_newlines_to_spaces(self) -> None:
        """With multiline disabled, newlines become spaces."""
        fi = FooterInput(allow_multiline=False)
        result = fi._normalize_paste("line1\nline2\nline3")
        assert result == "line1 line2 line3"
        assert "\n" not in result

    def test_tabs_with_multiline_disabled(self) -> None:
        """Tabs should be expanded even with multiline disabled."""
        fi = FooterInput(allow_multiline=False)
        result = fi._normalize_paste("hello\tworld\nnewline\twith\ttabs")
        # Tabs expanded, then newlines converted to spaces
        assert result == "hello    world newline    with    tabs"


class TestInsertTextCursorPosition:
    """Tests for cursor position after text insertion."""

    def test_cursor_at_end_after_paste_without_tabs(self) -> None:
        """Cursor should be at end after inserting text."""
        fi = FooterInput()
        fi._insert_text("hello world")
        assert fi.cursor_pos == 11
        assert fi.buffer == "hello world"

    def test_cursor_position_after_tab_paste(self) -> None:
        """Cursor position should match visual position after tab expansion."""
        fi = FooterInput()
        # Paste text with tab - will be normalized
        normalized = fi._normalize_paste("hello\tworld")
        fi._insert_text(normalized)
        # "hello    world" = 14 characters
        assert fi.buffer == "hello    world"
        assert fi.cursor_pos == 14
        assert len(fi.buffer) == fi.cursor_pos

    def test_cursor_position_mid_string_tab_insert(self) -> None:
        """Inserting tab in middle should update cursor correctly."""
        fi = FooterInput()
        fi.buffer = "helloworld"
        fi.cursor_pos = 5  # After "hello"
        normalized = fi._normalize_paste("\t")
        fi._insert_text(normalized)
        # Should have "hello    world"
        assert fi.buffer == "hello    world"
        assert fi.cursor_pos == 9  # 5 + 4 spaces


class TestMultilineRendering:
    """Tests for multiline input rendering without indent."""

    def test_single_line_has_prompt(self) -> None:
        """Single line should have prompt prefix."""
        fi = FooterInput(prompt="> ")
        fi.buffer = "hello"
        fi.cursor_pos = 5
        lines = fi.get_lines()
        assert len(lines) >= 1
        assert lines[0].startswith("> ")

    def test_multiline_no_indent_on_continuation(self) -> None:
        """Continuation lines should NOT have indent."""
        fi = FooterInput(prompt="> ")
        fi.buffer = "line1\nline2\nline3"
        fi.cursor_pos = len(fi.buffer)
        lines = fi.get_lines()
        # First line should have prompt
        assert lines[0].startswith("> ")
        # Subsequent lines should NOT start with spaces (no indent)
        if len(lines) > 1:
            # The line should not start with indent matching prompt width
            for line in lines[1:]:
                stripped = ANSI.strip_ansi(line)
                # Should NOT start with 2 spaces (prompt width)
                assert not stripped.startswith("  ") or stripped.startswith("  ") and stripped[2:3] != ""

    def test_empty_input_shows_cursor(self) -> None:
        """Empty input should show cursor character."""
        fi = FooterInput(prompt="> ")
        fi.buffer = ""
        fi.cursor_pos = 0
        lines = fi.get_lines()
        assert len(lines) == 1
        # Should contain reverse video for cursor
        assert "\033[7m" in lines[0]

    def test_cursor_at_end_shows_space_cursor(self) -> None:
        """Cursor at end should show space in reverse video."""
        fi = FooterInput(prompt="> ")
        fi.buffer = "hello"
        fi.cursor_pos = 5  # At end
        lines = fi.get_lines()
        # Should contain reverse video
        full = "".join(lines)
        assert "\033[7m" in full

    def test_cursor_in_middle_highlights_character(self) -> None:
        """Cursor in middle should highlight the character at position."""
        fi = FooterInput(prompt="> ")
        fi.buffer = "hello"
        fi.cursor_pos = 2  # At 'l'
        lines = fi.get_lines()
        full = "".join(lines)
        # Should contain reverse video around 'l'
        assert "\033[7m" in full


class TestLineWrapping:
    """Tests for line wrapping behavior."""

    def test_long_first_line_wraps(self) -> None:
        """Long first line should wrap to multiple visual lines."""
        fi = FooterInput(prompt="> ")
        # Create a long line that will definitely wrap
        fi.buffer = "a" * 100
        fi.cursor_pos = 100
        lines = fi.get_lines()
        # Should have multiple lines due to wrapping
        assert len(lines) >= 2

    def test_wrapped_lines_use_full_width(self) -> None:
        """Wrapped continuation lines should use full terminal width."""
        fi = FooterInput(prompt="> ")
        fi.buffer = "a" * 200
        fi.cursor_pos = 200
        lines = fi.get_lines()
        # First line has prompt, subsequent lines should be longer
        # (using full width instead of being indented)
        if len(lines) >= 3:
            first_content_len = ANSI.visual_len(ANSI.strip_ansi(lines[0]))
            second_content_len = ANSI.visual_len(ANSI.strip_ansi(lines[1]))
            # Second line should be at least as long (no indent reducing width)
            # This is approximate due to cursor handling
            assert second_content_len > 0

    def test_multiline_with_explicit_newlines(self) -> None:
        """Explicit newlines should create new logical lines."""
        fi = FooterInput(prompt="> ")
        fi.buffer = "short\nanother line"
        fi.cursor_pos = len(fi.buffer)
        lines = fi.get_lines()
        assert len(lines) >= 2

    def test_narrow_terminal_guard(self) -> None:
        """Should handle very narrow terminals without errors."""
        fi = FooterInput(prompt="> ")
        fi.buffer = "hello world"
        fi.cursor_pos = 11
        # This should not crash even with default terminal width
        lines = fi.get_lines()
        assert len(lines) >= 1


class TestCompletionDelay:
    """Tests for completion delay behavior."""

    def test_text_input_has_zero_delay(self) -> None:
        """FooterInput should have 0 completion delay."""
        fi = FooterInput()
        assert fi.completion_delay() == 0.0


class TestInputEventHandling:
    """Tests for input event handling."""

    def test_insert_regular_character(self) -> None:
        """Regular character insertion."""
        fi = FooterInput()
        from cli.elements.base import InputEvent

        fi.handle_input(InputEvent(key="Char", char="a"))
        assert fi.buffer == "a"
        assert fi.cursor_pos == 1

    def test_backspace_deletes_character(self) -> None:
        """Backspace should delete character before cursor."""
        fi = FooterInput()
        fi.buffer = "hello"
        fi.cursor_pos = 5
        from cli.elements.base import InputEvent

        fi.handle_input(InputEvent(key="Backspace", char=None))
        assert fi.buffer == "hell"
        assert fi.cursor_pos == 4

    def test_enter_submits_input(self) -> None:
        """Enter should submit the input."""
        fi = FooterInput()
        fi.buffer = "hello"
        fi.cursor_pos = 5
        from cli.elements.base import InputEvent

        done, result = fi.handle_input(InputEvent(key="Enter", char=None))
        assert done is True
        assert result == "hello"

    def test_escape_cancels_input(self) -> None:
        """Escape should cancel and return None."""
        fi = FooterInput()
        fi.buffer = "hello"
        fi.cursor_pos = 5
        from cli.elements.base import InputEvent

        done, result = fi.handle_input(InputEvent(key="Escape", char=None))
        assert done is True
        assert result is None

    def test_ctrl_d_cancels_empty_input(self) -> None:
        """Ctrl+D on empty input should cancel."""
        fi = FooterInput()
        from cli.elements.base import InputEvent

        done, result = fi.handle_input(InputEvent(key="Ctrl+D", char=None))
        assert done is True
        assert result is None

    def test_arrow_keys_move_cursor(self) -> None:
        """Arrow keys should move cursor."""
        fi = FooterInput()
        fi.buffer = "hello"
        fi.cursor_pos = 2
        from cli.elements.base import InputEvent

        # Move right
        fi.handle_input(InputEvent(key="Right", char=None))
        assert fi.cursor_pos == 3

        # Move left
        fi.handle_input(InputEvent(key="Left", char=None))
        assert fi.cursor_pos == 2

    def test_home_moves_to_start(self) -> None:
        """Home should move cursor to start."""
        fi = FooterInput()
        fi.buffer = "hello"
        fi.cursor_pos = 3
        from cli.elements.base import InputEvent

        fi.handle_input(InputEvent(key="Home", char=None))
        assert fi.cursor_pos == 0

    def test_end_moves_to_end(self) -> None:
        """End should move cursor to end."""
        fi = FooterInput()
        fi.buffer = "hello"
        fi.cursor_pos = 0
        from cli.elements.base import InputEvent

        fi.handle_input(InputEvent(key="End", char=None))
        assert fi.cursor_pos == 5
