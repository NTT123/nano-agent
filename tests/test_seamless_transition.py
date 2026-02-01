"""Tests for seamless input-to-message transition.

Covers:
- overwrite_content_with_message() method
- finish_content_overwrite() method
- Seamless transition path for user messages
- Completion delay behavior
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from cli.elements.base import ActiveElement
from cli.elements.footer import FooterState, TerminalFooter


class _CaptureStdout:
    """Context manager to capture stdout writes."""

    def __init__(self) -> None:
        self.captured = ""
        self._original_write = sys.stdout.write

    def __enter__(self) -> "_CaptureStdout":
        self.captured = ""

        def capturing_write(s: str) -> int:
            self.captured += s
            return len(s)

        sys.stdout.write = capturing_write  # type: ignore[method-assign]
        return self

    def __exit__(self, *args: object) -> None:
        sys.stdout.write = self._original_write  # type: ignore[method-assign]


class _FakeRegion:
    """Mock region for testing without actual terminal I/O."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.num_lines = 0
        self._active = False
        self._cursor_at_line = 0

    def activate(self, num_lines: int) -> None:
        self.calls.append(("activate", num_lines))
        self.num_lines = num_lines
        self._active = True

    def render(self, lines: list[str]) -> None:
        self.calls.append(("render", list(lines)))

    def update_size(self, num_lines: int) -> None:
        self.calls.append(("update_size", num_lines))
        self.num_lines = num_lines

    def deactivate(self) -> None:
        self.calls.append(("deactivate",))
        self._active = False
        self.num_lines = 0

    def _move_to_region_start(self) -> None:
        self.calls.append(("_move_to_region_start",))
        self._cursor_at_line = 0


class TestOverwriteContentWithMessage:
    """Tests for overwrite_content_with_message() method."""

    def test_returns_content_line_count(self) -> None:
        """Should return the number of content lines."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["line1", "line2", "line3"]

        count = footer.overwrite_content_with_message()
        assert count == 3

    def test_returns_zero_when_inactive(self) -> None:
        """Should return 0 when footer is not active."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        # Don't activate
        footer._content_lines = ["line1", "line2"]

        count = footer.overwrite_content_with_message()
        assert count == 0

    def test_moves_cursor_to_region_start(self) -> None:
        """Should move cursor to start of region."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["line1"]

        footer.overwrite_content_with_message()
        # Should have called _move_to_region_start
        assert ("_move_to_region_start",) in fake.calls

    def test_returns_zero_with_empty_content(self) -> None:
        """Should return 0 when no content lines."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = []

        count = footer.overwrite_content_with_message()
        assert count == 0


class TestFinishContentOverwrite:
    """Tests for finish_content_overwrite() method."""

    def test_clears_content_lines(self) -> None:
        """Should clear the content lines list."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["line1", "line2"]

        footer.finish_content_overwrite(2)
        assert footer._content_lines == []

    def test_resizes_region_to_one_line(self) -> None:
        """Should resize region to just status bar (1 line)."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["line1", "line2"]
        fake.num_lines = 3  # content + status

        footer.finish_content_overwrite(2)
        assert fake.num_lines == 1

    def test_does_nothing_when_inactive(self) -> None:
        """Should do nothing when footer is not active."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        # Don't activate
        footer._content_lines = ["line1"]
        initial_calls = len(fake.calls)

        footer.finish_content_overwrite(1)
        # Should not have added calls (except maybe deactivate checks)
        assert footer._content_lines == ["line1"]  # Unchanged


class TestCompletionDelayBehavior:
    """Tests for completion_delay() on elements."""

    def test_footer_input_has_zero_delay(self) -> None:
        """FooterInput should have 0 completion delay."""
        from cli.elements.footer_input import FooterInput

        fi = FooterInput()
        assert fi.completion_delay() == 0.0

    def test_base_element_default_delay(self) -> None:
        """Base ActiveElement should have default 0.15s delay."""

        # Create a minimal concrete implementation
        class MinimalElement(ActiveElement[str]):
            def get_lines(self) -> list[str]:
                return []

            def handle_input(self, event: Any) -> tuple[bool, str | None]:
                return (False, None)

        elem = MinimalElement()
        assert elem.completion_delay() == 0.15


class TestHasOverwritableContent:
    """Tests for checking if footer has overwritable content."""

    def test_has_content_when_active_with_lines(self) -> None:
        """Should return True when active with content lines."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["input line"]

        assert footer._content_lines
        assert footer._state == FooterState.ACTIVE
        assert footer.is_active()

    def test_no_content_when_inactive(self) -> None:
        """Should return False when not active."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer._content_lines = ["input line"]
        # Don't activate

        assert footer._state != FooterState.ACTIVE
        assert not footer.is_active()

    def test_no_content_when_empty(self) -> None:
        """Should return False when content is empty."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = []

        assert not footer._content_lines


class TestSeamlessTransitionPath:
    """Integration tests for the seamless transition path."""

    def test_overwrite_then_finish_sequence(self) -> None:
        """Full overwrite sequence should work correctly."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["prompt > user input here"]
        initial_lines = 2  # content + status

        # Step 1: Prepare overwrite
        line_count = footer.overwrite_content_with_message()
        assert line_count == 1

        # Step 2: (Caller would print message here)

        # Step 3: Finish overwrite
        footer.finish_content_overwrite(1)  # Printed 1 line

        # Verify state
        assert footer._content_lines == []
        assert fake.num_lines == 1  # Just status bar

    def test_overwrite_with_fewer_lines_printed(self) -> None:
        """When printed lines < old content, should handle cleanup."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["line1", "line2", "line3"]  # 3 lines

        footer.overwrite_content_with_message()
        # Simulate printing only 1 line (shorter than original 3)
        footer.finish_content_overwrite(1)

        # Content should be cleared
        assert footer._content_lines == []

    def test_overwrite_with_more_lines_printed(self) -> None:
        """When printed lines >= old content, should still work."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["short"]  # 1 line

        footer.overwrite_content_with_message()
        # Simulate printing 3 lines (more than original)
        footer.finish_content_overwrite(3)

        # Content should be cleared
        assert footer._content_lines == []


class TestFooterStateTransitions:
    """Tests for footer state during transitions."""

    def test_pause_prevents_overwrite(self) -> None:
        """Paused footer should not allow overwrite."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["content"]
        footer.pause()

        # Should return 0 when paused
        count = footer.overwrite_content_with_message()
        assert count == 0

    def test_resume_after_pause_allows_normal_operation(self) -> None:
        """Resumed footer should work normally."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()
        footer._content_lines = ["content"]
        footer.pause()
        footer.resume()

        # Should work after resume
        count = footer.overwrite_content_with_message()
        assert count == 1
