from __future__ import annotations

import io
import sys

from cli.elements.footer import TerminalFooter
from cli.elements.terminal import ANSI, TerminalRegion


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
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.num_lines = 0
        self._active = False

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


def test_ansi_visual_len_strips_escape_codes() -> None:
    assert ANSI.visual_len("\033[31mred\033[0m") == 3
    assert ANSI.visual_len("plain") == 5


def test_footer_render_sequence() -> None:
    footer = TerminalFooter()
    fake = _FakeRegion()
    footer._region = fake  # type: ignore[assignment]

    footer.activate()
    assert footer.is_active()

    footer.pause()
    assert not footer.is_active()
    calls_after_pause = list(fake.calls)

    footer.render()
    assert fake.calls == calls_after_pause

    footer.resume()
    assert footer.is_active()
    assert fake.calls[-2][0] == "activate"
    assert fake.calls[-1][0] == "render"

    footer.deactivate()
    assert not footer.is_active()
    assert fake.calls[-1][0] == "deactivate"


# --- TerminalRegion tests ---
# These tests verify that TerminalRegion uses explicit cursor movement
# instead of save/restore sequences, which don't work in all terminals.


class TestTerminalRegionCursorTracking:
    """Tests for TerminalRegion cursor position tracking."""

    def test_initial_cursor_position_is_zero(self) -> None:
        region = TerminalRegion()
        assert region._cursor_at_line == 0

    def test_activate_resets_cursor_position(self) -> None:
        region = TerminalRegion()
        region._cursor_at_line = 5  # Simulate some previous state
        with _CaptureStdout():
            region.activate(2)
        assert region._cursor_at_line == 0

    def test_deactivate_resets_cursor_position(self) -> None:
        region = TerminalRegion()
        with _CaptureStdout():
            region.activate(2)
            region._cursor_at_line = 1
            region.deactivate()
        assert region._cursor_at_line == 0


class TestTerminalRegionNoSaveRestore:
    """Tests ensuring TerminalRegion doesn't use save/restore cursor sequences.

    The sequences \\033[s (save) and \\033[u (restore) don't work reliably
    in all terminals. We use explicit cursor movement (cursor_up) instead.
    """

    def test_activate_does_not_use_save_cursor(self) -> None:
        region = TerminalRegion()
        with _CaptureStdout() as cap:
            region.activate(2)
        # Should NOT contain save cursor sequence
        assert ANSI.SAVE_CURSOR not in cap.captured

    def test_render_does_not_use_restore_cursor(self) -> None:
        region = TerminalRegion()
        with _CaptureStdout() as cap:
            region.activate(2)
            cap.captured = ""  # Clear activation output
            region.render(["line1", "line2"])
        # Should NOT contain restore cursor sequence
        assert ANSI.RESTORE_CURSOR not in cap.captured

    def test_render_uses_cursor_up_to_return_to_start(self) -> None:
        region = TerminalRegion()
        with _CaptureStdout() as cap:
            region.activate(2)
            # Simulate cursor being on line 1 after first render
            region._cursor_at_line = 1
            cap.captured = ""
            region.render(["line1", "line2"])
        # Should contain cursor up sequence to move back
        assert ANSI.cursor_up(1) in cap.captured

    def test_deactivate_does_not_use_restore_cursor(self) -> None:
        region = TerminalRegion()
        with _CaptureStdout() as cap:
            region.activate(2)
            region._cursor_at_line = 1
            cap.captured = ""
            region.deactivate()
        # Should NOT contain restore cursor sequence
        assert ANSI.RESTORE_CURSOR not in cap.captured

    def test_update_size_does_not_use_save_restore(self) -> None:
        region = TerminalRegion()
        with _CaptureStdout() as cap:
            region.activate(2)
            cap.captured = ""
            region.update_size(4)
        # Should NOT contain save/restore cursor sequences
        assert ANSI.SAVE_CURSOR not in cap.captured
        assert ANSI.RESTORE_CURSOR not in cap.captured


class TestTerminalRegionRepeatedRenders:
    """Tests for repeated render behavior - the original bug scenario."""

    def test_repeated_renders_track_cursor_correctly(self) -> None:
        """Verify cursor tracking across multiple renders.

        This is the core test for the bug fix: without proper cursor tracking,
        repeated renders would print on new lines instead of overwriting.
        """
        region = TerminalRegion()
        with _CaptureStdout():
            region.activate(2)

            # First render - cursor should end up on line 1
            region.render(["line1", "line2"])
            # After render, cursor is positioned at end of last line
            # which is line index 1 (0-indexed)
            assert region._cursor_at_line == 1

            # Second render - should move cursor back up first
            region.render(["updated1", "updated2"])
            assert region._cursor_at_line == 1

    def test_single_line_render_cursor_stays_at_zero(self) -> None:
        region = TerminalRegion()
        with _CaptureStdout():
            region.activate(1)
            region.render(["single line"])
            # With only 1 line, cursor stays at line 0
            assert region._cursor_at_line == 0

    def test_move_to_region_start_uses_cursor_up(self) -> None:
        """Verify _move_to_region_start uses cursor_up for movement."""
        region = TerminalRegion()
        with _CaptureStdout() as cap:
            region.activate(3)
            region._cursor_at_line = 2  # Simulate cursor at bottom
            cap.captured = ""
            region._move_to_region_start()
        # Should move up 2 lines
        assert ANSI.cursor_up(2) in cap.captured
        assert region._cursor_at_line == 0

    def test_move_to_region_start_no_move_when_at_top(self) -> None:
        """Verify no cursor_up when already at top."""
        region = TerminalRegion()
        with _CaptureStdout() as cap:
            region.activate(3)
            region._cursor_at_line = 0  # Already at top
            cap.captured = ""
            region._move_to_region_start()
        # Should NOT contain cursor up (since already at line 0)
        assert "\033[" not in cap.captured or "A" not in cap.captured
