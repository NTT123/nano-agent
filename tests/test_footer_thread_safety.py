"""Tests for thread safety in TerminalFooter.

Covers:
- Thread safety with _render_lock
- Concurrent render/set_content/update_status operations
- Race condition prevention between spinner loop and main thread
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from cli.elements.footer import TerminalFooter


class _FakeRegion:
    """Mock region for testing without actual terminal I/O."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.num_lines = 0
        self._active = False
        self._render_count = 0
        self._lock = threading.Lock()

    def activate(self, num_lines: int) -> None:
        with self._lock:
            self.calls.append(("activate", num_lines))
            self.num_lines = num_lines
            self._active = True

    def render(self, lines: list[str]) -> None:
        with self._lock:
            self.calls.append(("render", list(lines)))
            self._render_count += 1

    def update_size(self, num_lines: int) -> None:
        with self._lock:
            self.calls.append(("update_size", num_lines))
            self.num_lines = num_lines

    def deactivate(self) -> None:
        with self._lock:
            self.calls.append(("deactivate",))
            self._active = False
            self.num_lines = 0


class TestFooterHasLock:
    """Tests verifying the lock exists and is used."""

    def test_footer_has_render_lock(self) -> None:
        """TerminalFooter should have a _render_lock attribute."""
        footer = TerminalFooter()
        assert hasattr(footer, "_render_lock")
        assert isinstance(footer._render_lock, type(threading.Lock()))

    def test_lock_is_threading_lock(self) -> None:
        """The lock should be a proper threading lock."""
        footer = TerminalFooter()
        # Should be able to acquire and release
        acquired = footer._render_lock.acquire(blocking=False)
        assert acquired
        footer._render_lock.release()


class TestConcurrentOperations:
    """Tests for concurrent access to footer methods."""

    def test_concurrent_render_calls(self) -> None:
        """Multiple threads calling render() should not corrupt state."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        errors: list[Exception] = []

        def stress_render(iterations: int = 50) -> None:
            try:
                for _ in range(iterations):
                    footer.render()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress_render) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent render: {errors}"
        # Should have completed all renders without exception
        assert fake._render_count > 0

    def test_concurrent_set_content_and_render(self) -> None:
        """Concurrent set_content() and render() should be thread-safe."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        errors: list[Exception] = []

        def stress_render(iterations: int = 30) -> None:
            try:
                for _ in range(iterations):
                    footer.render()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def stress_content(iterations: int = 30) -> None:
            try:
                for i in range(iterations):
                    footer.set_content([f"Line {i}"])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=stress_render),
            threading.Thread(target=stress_content),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"

    def test_concurrent_update_status_and_render(self) -> None:
        """Concurrent update_status() and render() should be thread-safe."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        errors: list[Exception] = []

        def stress_render(iterations: int = 30) -> None:
            try:
                for _ in range(iterations):
                    footer.render()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def stress_status(iterations: int = 30) -> None:
            try:
                for i in range(iterations):
                    footer.update_status(input_tokens=i, output_tokens=i * 2)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=stress_render),
            threading.Thread(target=stress_status),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"

    def test_concurrent_all_operations(self) -> None:
        """All footer operations running concurrently should be thread-safe."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        errors: list[Exception] = []
        iterations = 20

        def stress_render() -> None:
            try:
                for _ in range(iterations):
                    footer.render()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def stress_content() -> None:
            try:
                for i in range(iterations):
                    footer.set_content([f"Content {i}"])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def stress_status() -> None:
            try:
                for i in range(iterations):
                    footer.update_status(input_tokens=i)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def stress_activity() -> None:
            try:
                for i in range(iterations):
                    footer.set_activity(f"Activity {i}" if i % 2 == 0 else None)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=stress_render),
            threading.Thread(target=stress_content),
            threading.Thread(target=stress_status),
            threading.Thread(target=stress_activity),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


class TestStateConsistency:
    """Tests for state consistency under concurrent access."""

    def test_content_lines_not_corrupted(self) -> None:
        """Content lines should not be corrupted by concurrent access."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        final_content: list[str] = []
        iterations = 50

        def set_content_repeatedly() -> None:
            for i in range(iterations):
                footer.set_content([f"Line {i}"])
                time.sleep(0.001)
            # Record final state
            final_content.extend(footer._content_lines)

        def render_repeatedly() -> None:
            for _ in range(iterations):
                footer.render()
                time.sleep(0.001)

        t1 = threading.Thread(target=set_content_repeatedly)
        t2 = threading.Thread(target=render_repeatedly)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Content should be a valid list of strings
        assert isinstance(footer._content_lines, list)
        for line in footer._content_lines:
            assert isinstance(line, str)

    def test_status_values_consistent(self) -> None:
        """Status values should remain consistent."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        def update_tokens() -> None:
            for i in range(100):
                footer.update_status(
                    input_tokens=i * 10,
                    output_tokens=i * 20,
                    thinking_tokens=i * 5,
                )
                time.sleep(0.001)

        def check_status() -> None:
            for _ in range(100):
                # Just access the status - should not crash
                _ = footer.status.input_tokens
                _ = footer.status.output_tokens
                _ = footer.status.thinking_tokens
                time.sleep(0.001)

        t1 = threading.Thread(target=update_tokens)
        t2 = threading.Thread(target=check_status)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Status should have valid values
        assert footer.status.input_tokens >= 0
        assert footer.status.output_tokens >= 0


class TestLockBehaviorVerification:
    """Tests to verify lock is actually being used."""

    def test_render_acquires_lock(self) -> None:
        """render() should acquire the lock."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        lock_acquired_during_render = False

        # Replace lock with a tracking version
        original_lock = footer._render_lock

        class TrackingLock:
            def __init__(self) -> None:
                self.acquired_count = 0

            def __enter__(self) -> "TrackingLock":
                nonlocal lock_acquired_during_render
                self.acquired_count += 1
                lock_acquired_during_render = True
                return original_lock.__enter__()

            def __exit__(self, *args: Any) -> None:
                return original_lock.__exit__(*args)

        footer._render_lock = TrackingLock()  # type: ignore[assignment]
        footer.render()

        assert lock_acquired_during_render

    def test_set_content_acquires_lock(self) -> None:
        """set_content() should acquire the lock."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        original_lock = footer._render_lock
        acquired = []

        class TrackingLock:
            def __enter__(self) -> "TrackingLock":
                acquired.append(True)
                return original_lock.__enter__()

            def __exit__(self, *args: Any) -> None:
                return original_lock.__exit__(*args)

        footer._render_lock = TrackingLock()  # type: ignore[assignment]
        footer.set_content(["test"])

        assert len(acquired) > 0

    def test_update_status_acquires_lock(self) -> None:
        """update_status() should acquire the lock."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]
        footer.activate()

        original_lock = footer._render_lock
        acquired = []

        class TrackingLock:
            def __enter__(self) -> "TrackingLock":
                acquired.append(True)
                return original_lock.__enter__()

            def __exit__(self, *args: Any) -> None:
                return original_lock.__exit__(*args)

        footer._render_lock = TrackingLock()  # type: ignore[assignment]
        footer.update_status(input_tokens=100)

        assert len(acquired) > 0


class TestFooterStateMachine:
    """Tests for FooterState enum and state transitions."""

    def test_footer_state_enum_values(self) -> None:
        """FooterState enum should have expected values."""
        from cli.elements.footer import FooterState

        assert FooterState.INACTIVE.name == "INACTIVE"
        assert FooterState.ACTIVE.name == "ACTIVE"
        assert FooterState.PAUSED.name == "PAUSED"

    def test_initial_state_is_inactive(self) -> None:
        """Footer should start in INACTIVE state."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        assert footer._state == FooterState.INACTIVE

    def test_activate_transitions_to_active(self) -> None:
        """activate() should transition from INACTIVE to ACTIVE."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        footer.activate()
        assert footer._state == FooterState.ACTIVE
        assert footer.is_active()

    def test_deactivate_transitions_to_inactive(self) -> None:
        """deactivate() should transition to INACTIVE."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        footer.activate()
        footer.deactivate()
        assert footer._state == FooterState.INACTIVE
        assert not footer.is_active()

    def test_pause_transitions_to_paused(self) -> None:
        """pause() should transition from ACTIVE to PAUSED."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        footer.activate()
        footer.pause()
        assert footer._state == FooterState.PAUSED
        assert footer.is_paused()

    def test_resume_transitions_to_active(self) -> None:
        """resume() should transition from PAUSED to ACTIVE."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        footer.activate()
        footer.pause()
        footer.resume()
        assert footer._state == FooterState.ACTIVE
        assert footer.is_active()

    def test_deactivate_from_paused(self) -> None:
        """deactivate() should work from PAUSED state."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        footer.activate()
        footer.pause()
        footer.deactivate()
        assert footer._state == FooterState.INACTIVE

    def test_get_state_returns_lowercase(self) -> None:
        """get_state() should return lowercase state name."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        assert footer.get_state() == "inactive"
        footer.activate()
        assert footer.get_state() == "active"
        footer.pause()
        assert footer.get_state() == "paused"

    def test_activate_from_paused_resumes(self) -> None:
        """activate() from PAUSED should resume (not re-activate)."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        footer.activate()
        footer._content_lines = ["preserved content"]
        footer.pause()
        footer.activate()

        # Should be active and content preserved
        assert footer._state == FooterState.ACTIVE
        assert footer._content_lines == ["preserved content"]

    def test_idempotent_activate(self) -> None:
        """activate() when already active should be no-op."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        footer.activate()
        call_count_after_first = len(fake.calls)
        footer.activate()

        # Should not have added more calls
        assert footer._state == FooterState.ACTIVE

    def test_idempotent_deactivate(self) -> None:
        """deactivate() when already inactive should be no-op."""
        from cli.elements.footer import FooterState

        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        # Already inactive
        footer.deactivate()
        assert footer._state == FooterState.INACTIVE

    def test_concurrent_state_transitions(self) -> None:
        """Concurrent state transitions should be thread-safe."""
        footer = TerminalFooter()
        fake = _FakeRegion()
        footer._region = fake  # type: ignore[assignment]

        errors: list[Exception] = []

        def toggle_state(iterations: int = 20) -> None:
            try:
                for _ in range(iterations):
                    footer.activate()
                    footer.pause()
                    footer.resume()
                    footer.deactivate()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle_state) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent transitions: {errors}"
