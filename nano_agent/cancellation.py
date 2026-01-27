"""Cancellation token for cooperative async operation cancellation."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")

# Platform-specific imports for keyboard monitoring
_HAS_TERMIOS = False
if sys.platform != "win32":
    try:
        import termios
        import tty

        _HAS_TERMIOS = True
    except ImportError:
        pass


class KeyboardMonitor:
    """Async monitor for Escape key presses during operations.

    Uses asyncio's add_reader() to monitor stdin without threads.

    Usage:
        monitor = KeyboardMonitor(on_escape_callback)
        monitor.start()
        # ... do async work ...
        monitor.stop()
    """

    # Escape key character
    ESC = "\x1b"

    def __init__(self, callback: Callable[[], None]):
        """Initialize keyboard monitor.

        Args:
            callback: Function to call when Escape is pressed.
        """
        self._callback = callback
        self._running = False
        self._original_settings: list[Any] | None = None
        self._fd: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start async keyboard monitoring using loop.add_reader()."""
        if not _HAS_TERMIOS:
            return  # Not supported without termios

        if not sys.stdin.isatty():
            return  # Not a TTY, can't monitor

        if self._running:
            return  # Already running

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # No running loop

        self._fd = sys.stdin.fileno()
        try:
            self._original_settings = termios.tcgetattr(self._fd)
        except termios.error:
            return  # Can't get terminal settings

        try:
            # Set terminal to cbreak mode (character-at-a-time, signals still work)
            tty.setcbreak(self._fd)
            # Register async reader for stdin
            self._loop.add_reader(self._fd, self._on_stdin_ready)
            self._running = True
        except (termios.error, OSError):
            self._restore_terminal()

    def stop(self) -> None:
        """Stop keyboard monitoring and restore terminal."""
        if not self._running:
            return

        self._running = False

        # Remove the reader
        if self._loop and self._fd is not None:
            try:
                self._loop.remove_reader(self._fd)
            except (ValueError, OSError):
                pass  # Already removed or invalid

        # Restore terminal
        self._restore_terminal()

    def _on_stdin_ready(self) -> None:
        """Callback when stdin has data available."""
        if not self._running or self._fd is None:
            return

        try:
            char = os.read(self._fd, 1).decode("utf-8", errors="ignore")
            if char == self.ESC:
                # Escape pressed - trigger callback and stop monitoring
                self.stop()
                self._callback()
        except (IOError, OSError):
            self.stop()

    def _restore_terminal(self) -> None:
        """Restore original terminal settings."""
        if self._original_settings and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
            except termios.error:
                pass  # Best effort
        self._original_settings = None


@dataclass
class CancellationToken:
    """Token for cooperative cancellation of async operations.

    Supports multiple cancellation triggers:
    - Direct call to cancel()
    - SIGINT signal (Ctrl+C) when signal handler is installed
    - Escape key press when keyboard monitor is active

    Usage:
        token = CancellationToken()

        # Install SIGINT handler (optional, for cleaner Ctrl+C handling)
        token.install_signal_handler()

        # In agent loop:
        try:
            # Start keyboard monitoring during long operations
            token.start_keyboard_monitor()
            result = await token.run(api.send(dag))
        except asyncio.CancelledError:
            # Handle cancellation
            pass
        finally:
            token.stop_keyboard_monitor()

        # To cancel (from another coroutine/callback):
        token.cancel()

        # To reuse for new operation:
        token.reset()

        # Cleanup when done
        token.remove_signal_handler()

    Integration with executor.run():
        from nano_agent import run, CancellationToken

        token = CancellationToken()
        # In another task: token.cancel()
        dag = await run(api, dag, cancel_token=token)
    """

    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _current_task: asyncio.Task[Any] | None = field(default=None, init=False)
    _keyboard_monitor: KeyboardMonitor | None = field(default=None, init=False)
    _signal_handler_installed: bool = field(default=False, init=False)
    _original_sigint_handler: Any = field(default=None, init=False)

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation - sets flag and cancels current task."""
        self._event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def reset(self) -> None:
        """Reset for reuse with new operation."""
        self._event.clear()
        self._current_task = None

    def install_signal_handler(self) -> None:
        """Install SIGINT handler for Ctrl+C cancellation.

        This provides cleaner handling than catching KeyboardInterrupt,
        as it cancels the operation immediately via the event loop.
        """
        if self._signal_handler_installed:
            return

        if sys.platform == "win32":
            # Windows doesn't support add_signal_handler, use signal module
            self._original_sigint_handler = signal.signal(
                signal.SIGINT, lambda s, f: self.cancel()
            )
            self._signal_handler_installed = True
            return

        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, self.cancel)
            self._signal_handler_installed = True
        except RuntimeError:
            # No running loop, fall back to signal module
            self._original_sigint_handler = signal.signal(
                signal.SIGINT, lambda s, f: self.cancel()
            )
            self._signal_handler_installed = True

    def remove_signal_handler(self) -> None:
        """Remove SIGINT handler and restore original."""
        if not self._signal_handler_installed:
            return

        if sys.platform == "win32":
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._signal_handler_installed = False
            return

        try:
            loop = asyncio.get_running_loop()
            loop.remove_signal_handler(signal.SIGINT)
        except RuntimeError:
            # No running loop, restore via signal module
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)

        self._signal_handler_installed = False
        self._original_sigint_handler = None

    def start_keyboard_monitor(self) -> None:
        """Start monitoring for Escape key.

        Should be called before long-running async operations.
        Uses asyncio's add_reader() to detect Escape key asynchronously.
        """
        if self._keyboard_monitor is None:
            self._keyboard_monitor = KeyboardMonitor(self.cancel)
        self._keyboard_monitor.start()

    def stop_keyboard_monitor(self) -> None:
        """Stop keyboard monitoring.

        Should be called after async operations complete to restore
        normal terminal behavior.
        """
        if self._keyboard_monitor:
            self._keyboard_monitor.stop()

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine with cancellation support.

        Wraps coroutine in asyncio.Task so it can be cancelled mid-flight.
        Raises asyncio.CancelledError if cancel() was called.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            asyncio.CancelledError: If cancel() was called during execution
        """
        if self.is_cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

        task: asyncio.Task[T] = asyncio.create_task(coro)
        self._current_task = task
        try:
            return await task
        finally:
            self._current_task = None
