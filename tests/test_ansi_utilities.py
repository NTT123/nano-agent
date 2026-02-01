"""Tests for ANSI utility functions in cli/elements/terminal.py.

Covers:
- visual_len() with wide characters (CJK, emoji, combining marks)
- truncate_to_width() with wide characters and ANSI codes
- wrap_to_width() with wide characters and ANSI code carry-over
- CSI escape sequence terminator parsing
"""

from __future__ import annotations

import pytest

from cli.elements.terminal import ANSI


class TestVisualLenWideCharacters:
    """Tests for ANSI.visual_len() with wcwidth support."""

    def test_ascii_text(self) -> None:
        """Basic ASCII text should have length equal to character count."""
        assert ANSI.visual_len("hello") == 5
        assert ANSI.visual_len("Hello World") == 11

    def test_cjk_characters_count_as_two(self) -> None:
        """CJK characters should count as 2 visual columns each."""
        # Chinese
        assert ANSI.visual_len("你好") == 4  # 2 chars × 2 width
        assert ANSI.visual_len("中文") == 4
        # Japanese hiragana
        assert ANSI.visual_len("こんにちは") == 10  # 5 chars × 2 width
        # Korean
        assert ANSI.visual_len("안녕") == 4  # 2 chars × 2 width

    def test_emoji_count_as_two(self) -> None:
        """Most emoji should count as 2 visual columns."""
        assert ANSI.visual_len("👍") == 2
        assert ANSI.visual_len("🎉") == 2
        assert ANSI.visual_len("😀😁😂") == 6  # 3 emoji × 2 width

    def test_mixed_ascii_and_wide_characters(self) -> None:
        """Mixed content should sum widths correctly."""
        # "Hello" (5) + "你好" (4) = 9
        assert ANSI.visual_len("Hello你好") == 9
        # "Hello" (5) + "👍" (2) + "World" (5) = 12
        assert ANSI.visual_len("Hello👍World") == 12
        # "H" (1) + "你" (2) + "好" (2) + "W" (1) = 6
        assert ANSI.visual_len("H你好W") == 6

    def test_ansi_codes_excluded_from_length(self) -> None:
        """ANSI escape codes should not contribute to visual length."""
        # Red text
        assert ANSI.visual_len("\033[31mred\033[0m") == 3
        # Bold + color
        assert ANSI.visual_len("\033[1m\033[32mbold green\033[0m") == 10
        # Wide chars with color
        assert ANSI.visual_len("\033[31m你好\033[0m") == 4

    def test_ansi_codes_with_wide_characters(self) -> None:
        """ANSI codes combined with wide characters."""
        # Colored CJK: "Hello" (5) + "World" (5) + "你好" (4) = 14
        assert ANSI.visual_len("\033[32mHello\033[0m你好World") == 14

    def test_empty_string(self) -> None:
        """Empty string should have zero length."""
        assert ANSI.visual_len("") == 0

    def test_only_ansi_codes(self) -> None:
        """String with only ANSI codes should have zero visual length."""
        assert ANSI.visual_len("\033[31m\033[0m") == 0
        assert ANSI.visual_len("\033[1m\033[32m\033[0m") == 0


class TestCSITerminatorParsing:
    """Tests for comprehensive CSI escape sequence terminator support."""

    def test_cursor_movement_codes_recognized(self) -> None:
        """Cursor movement codes (A/B/C/D) should be properly parsed."""
        # Cursor up - 'A' terminator
        assert ANSI.visual_len("text\033[5Amore") == 8  # "text" + "more"
        # Cursor down - 'B' terminator
        assert ANSI.visual_len("text\033[3Bmore") == 8
        # Cursor right - 'C' terminator
        assert ANSI.visual_len("text\033[10Cmore") == 8
        # Cursor left - 'D' terminator
        assert ANSI.visual_len("text\033[4Dmore") == 8

    def test_save_restore_cursor_codes(self) -> None:
        """Save/restore cursor codes (s/u) should be properly parsed."""
        # Save cursor - 's' terminator
        assert ANSI.visual_len("hello\033[sworld") == 10
        # Restore cursor - 'u' terminator
        assert ANSI.visual_len("hello\033[uworld") == 10
        # Both together
        assert ANSI.visual_len("start\033[smiddle\033[uend") == 14

    def test_mode_change_codes(self) -> None:
        """Mode change codes (h/l) should be properly parsed."""
        # Enable bracketed paste - 'h' terminator
        assert ANSI.visual_len("text\033[?2004hmore") == 8
        # Disable bracketed paste - 'l' terminator
        assert ANSI.visual_len("text\033[?2004lmore") == 8

    def test_erase_codes(self) -> None:
        """Erase codes (J/K) should be properly parsed."""
        # Erase display - 'J' terminator
        assert ANSI.visual_len("text\033[2Jmore") == 8
        # Erase line - 'K' terminator
        assert ANSI.visual_len("text\033[Kmore") == 8

    def test_cursor_position_codes(self) -> None:
        """Cursor position codes (H/f) should be properly parsed."""
        # Move to position - 'H' terminator
        assert ANSI.visual_len("text\033[10;20Hmore") == 8
        # Alternative position - 'f' terminator
        assert ANSI.visual_len("text\033[10;20fmore") == 8

    def test_other_csi_codes(self) -> None:
        """Other CSI codes should be properly parsed."""
        # Scroll up - 'S' terminator
        assert ANSI.visual_len("text\033[2Smore") == 8
        # Scroll down - 'T' terminator
        assert ANSI.visual_len("text\033[2Tmore") == 8
        # Insert line - 'L' terminator
        assert ANSI.visual_len("text\033[Lmore") == 8
        # Delete line - 'M' terminator
        assert ANSI.visual_len("text\033[Mmore") == 8

    def test_mixed_csi_codes(self) -> None:
        """Multiple different CSI codes in same string."""
        # Color + cursor movement
        s = "\033[31mred\033[0m \033[5Atext"
        assert ANSI.visual_len(s) == 8  # "red " + "text"


class TestTruncateToWidth:
    """Tests for ANSI.truncate_to_width() with wide characters."""

    def test_no_truncation_needed(self) -> None:
        """String shorter than max_width should not be truncated."""
        result = ANSI.truncate_to_width("hello", 10)
        assert "hello" in result
        assert "…" not in result

    def test_truncate_ascii(self) -> None:
        """Basic ASCII truncation."""
        result = ANSI.truncate_to_width("hello world", 8)
        # Should fit "hello w" (7) + "…" (1) = 8
        assert ANSI.visual_len(ANSI.strip_ansi(result.replace("…", ""))) <= 7
        assert "…" in result

    def test_truncate_with_wide_characters(self) -> None:
        """Wide characters should be properly accounted for."""
        # "你好世界" = 8 visual width
        result = ANSI.truncate_to_width("你好世界", 5)
        visible = ANSI.strip_ansi(result.replace("…", ""))
        # Should fit "你好" (4) + "…" (1) = 5
        assert ANSI.visual_len(visible) <= 4

    def test_wide_char_at_boundary_excluded(self) -> None:
        """Wide char that would exceed width should be excluded."""
        # "Hello" (5) + "你" (2) = 7, but if width=6, can't fit 你
        result = ANSI.truncate_to_width("Hello你", 6)
        visible = result.replace("…", "").replace("\033[0m", "")
        # "你" takes 2 columns, only 1 available, so excluded
        assert "你" not in visible
        assert ANSI.visual_len(visible) == 5

    def test_truncate_preserves_ansi_codes(self) -> None:
        """ANSI codes should be preserved in truncated output."""
        result = ANSI.truncate_to_width("\033[31mhello world\033[0m", 8)
        assert "\033[31m" in result

    def test_truncate_mixed_wide_and_ascii(self) -> None:
        """Mixed content truncation."""
        # "Hello你好World" = 5 + 4 + 5 = 14 visual width
        result = ANSI.truncate_to_width("Hello你好World", 10)
        visible = ANSI.strip_ansi(result.replace("…", ""))
        assert ANSI.visual_len(visible) <= 9

    def test_truncate_with_emoji(self) -> None:
        """Emoji should be properly handled."""
        # "Hi👍Bye" = 2 + 2 + 3 = 7 visual width
        result = ANSI.truncate_to_width("Hi👍Bye", 5)
        visible = ANSI.strip_ansi(result.replace("…", ""))
        assert ANSI.visual_len(visible) <= 4

    def test_truncate_cursor_movement_codes(self) -> None:
        """Cursor movement codes should be preserved."""
        result = ANSI.truncate_to_width("text\033[5Amore", 20)
        assert "\033[5A" in result
        assert "text" in result
        assert "more" in result

    def test_truncate_preserves_all_csi_terminators(self) -> None:
        """All CSI terminator types should be handled."""
        codes = [
            ("\033[5A", "cursor up"),
            ("\033[3B", "cursor down"),
            ("\033[s", "save cursor"),
            ("\033[u", "restore cursor"),
            ("\033[?2004h", "bracketed paste on"),
        ]
        for code, desc in codes:
            s = f"start{code}end"
            result = ANSI.truncate_to_width(s, 20)
            assert code in result, f"Failed to preserve {desc}"

    def test_truncate_empty_string(self) -> None:
        """Empty string truncation."""
        result = ANSI.truncate_to_width("", 10)
        # Should return ellipsis + reset
        assert "…" in result


class TestWrapToWidth:
    """Tests for ANSI.wrap_to_width() with wide characters."""

    def test_no_wrap_needed(self) -> None:
        """String shorter than max_width should not wrap."""
        lines = ANSI.wrap_to_width("hello", 10)
        assert len(lines) == 1
        assert "hello" in lines[0]

    def test_wrap_ascii(self) -> None:
        """Basic ASCII wrapping."""
        lines = ANSI.wrap_to_width("hello world test", 6)
        assert len(lines) >= 2
        for line in lines:
            assert ANSI.visual_len(line) <= 6

    def test_wrap_cjk_characters(self) -> None:
        """CJK characters should wrap at correct positions."""
        # "你好世界" = 8 visual width, wrap at 5
        lines = ANSI.wrap_to_width("你好世界", 5)
        # Should wrap as "你好" (4) and "世界" (4)
        assert len(lines) == 2
        for line in lines:
            assert ANSI.visual_len(ANSI.strip_ansi(line)) <= 5

    def test_wrap_mixed_content(self) -> None:
        """Mixed ASCII and wide characters."""
        # "Hello世界test" = 5 + 4 + 4 = 13 visual width
        lines = ANSI.wrap_to_width("Hello世界test", 8)
        assert len(lines) >= 2
        for line in lines:
            assert ANSI.visual_len(ANSI.strip_ansi(line)) <= 8

    def test_wrap_preserves_ansi_codes(self) -> None:
        """ANSI codes should be preserved across wrapped lines."""
        lines = ANSI.wrap_to_width("\033[31mhello world\033[0m", 6)
        # Color code should appear in output
        full = "".join(lines)
        assert "\033[31m" in full

    def test_wrap_carries_sgr_codes_across_lines(self) -> None:
        """SGR (style) codes should be carried to continuation lines."""
        lines = ANSI.wrap_to_width("\033[31mhello world test\033[0m", 6)
        # If there are multiple lines, color should be re-applied
        if len(lines) > 1:
            # Each line should have the color code or reset
            for line in lines:
                assert "\033[31m" in line or "\033[0m" in line

    def test_wrap_with_emoji(self) -> None:
        """Emoji wrapping should account for 2-column width."""
        # "Hello😀World" = 5 + 2 + 5 = 12 visual width
        lines = ANSI.wrap_to_width("Hello😀World", 8)
        for line in lines:
            assert ANSI.visual_len(ANSI.strip_ansi(line)) <= 8

    def test_wrap_empty_string(self) -> None:
        """Empty string should return empty list."""
        lines = ANSI.wrap_to_width("", 10)
        assert lines == []

    def test_wrap_very_narrow_width(self) -> None:
        """Wrapping with very narrow width."""
        # Each CJK char is 2 wide, so at width 2, each gets its own line
        lines = ANSI.wrap_to_width("你好世界", 2)
        assert len(lines) == 4
        for line in lines:
            assert ANSI.visual_len(ANSI.strip_ansi(line)) <= 2

    def test_wrap_cursor_codes_not_carried_over(self) -> None:
        """Cursor movement codes should NOT be carried to next line like SGR."""
        # Only SGR codes (ending in 'm') should be tracked
        lines = ANSI.wrap_to_width("\033[31mtext\033[5Along line here\033[0m", 10)
        # Cursor code \033[5A should appear once, not repeated
        full = "".join(lines)
        assert full.count("\033[5A") == 1


class TestStripAnsi:
    """Tests for ANSI.strip_ansi() method."""

    def test_strip_sgr_codes(self) -> None:
        """SGR (style) codes should be stripped."""
        assert ANSI.strip_ansi("\033[31mred\033[0m") == "red"
        assert ANSI.strip_ansi("\033[1m\033[32mbold\033[0m") == "bold"

    def test_strip_cursor_codes(self) -> None:
        """Cursor movement codes should be stripped."""
        assert ANSI.strip_ansi("text\033[5Amore") == "textmore"
        assert ANSI.strip_ansi("\033[s\033[u") == ""

    def test_strip_preserves_plain_text(self) -> None:
        """Plain text without codes should be unchanged."""
        assert ANSI.strip_ansi("hello world") == "hello world"
        assert ANSI.strip_ansi("你好世界") == "你好世界"


class TestEdgeCases:
    """Edge case tests for ANSI utilities."""

    def test_visual_len_control_characters(self) -> None:
        """Control characters should be handled."""
        # Newline and tab - wcwidth returns -1, treated as 0
        # Note: actual behavior depends on wcwidth implementation
        length = ANSI.visual_len("hello\x00world")
        assert length >= 10  # At minimum "helloworld"

    def test_truncate_exact_fit(self) -> None:
        """String that exactly fits should not be truncated."""
        result = ANSI.truncate_to_width("hello", 5)
        assert "hello" in result
        assert "…" not in result

    def test_wrap_single_wide_char_per_line(self) -> None:
        """Single wide character that fills entire line."""
        lines = ANSI.wrap_to_width("你", 2)
        assert len(lines) == 1
        assert "你" in lines[0]

    def test_nested_ansi_codes(self) -> None:
        """Multiple nested ANSI codes."""
        s = "\033[1m\033[31m\033[4mbold red underline\033[0m"
        assert ANSI.visual_len(s) == 18  # "bold red underline"

        result = ANSI.truncate_to_width(s, 10)
        # Should preserve codes
        assert "\033[1m" in result
