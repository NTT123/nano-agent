"""Cancellation token for cooperative async operation cancellation."""

from __future__ import annotations

import asyncio
import signal
import sys
from dataclasses import dataclass, field
from typing import Any, Coroutine, TypeVar

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

T = TypeVar("T")


@dataclass
class CancellationToken:
    """Token for cooperative cancellation of async operations.

    Supports multiple cancellation triggers:
    - Direct call to cancel()
    - SIGINT signal (Ctrl+C) when signal handler is installed
    - Escape key press when using run_with_escape()

    Usage:
        token = CancellationToken()

        # Option 1: Run with Escape key support (recommended)
        result = await token.run_with_escape(api.send(dag))

        # Option 2: Just wrap in cancellable task
        result = await token.run(some_coroutine())

        # To cancel from elsewhere:
        token.cancel()

        # To reuse for new operation:
        token.reset()

    Integration with executor.run():
        from nano_agent import run, CancellationToken

        token = CancellationToken()
        dag = await run(api, dag, cancel_token=token)
    """

    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _current_task: asyncio.Task[Any] | None = field(default=None, init=False)
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

    async def run_with_escape(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine with Escape key cancellation support.

        Uses prompt_toolkit to listen for Escape key while the operation runs.
        This is the recommended method for CLI operations as it provides
        clean terminal handling.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            asyncio.CancelledError: If Escape was pressed or cancel() was called
        """
        if self.is_cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

        kb = KeyBindings()

        @kb.add("escape")
        def on_escape(event: Any) -> None:
            self.cancel()
            event.app.exit()

        # Minimal invisible app - just captures Escape key
        app: Application[None] = Application(
            layout=Layout(Window(FormattedTextControl(""), height=0)),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
        )

        # Create the operation task
        operation: asyncio.Task[T] = asyncio.create_task(coro)
        self._current_task = operation

        async def exit_app_when_done() -> None:
            """Exit the prompt_toolkit app when operation completes."""
            try:
                await operation
            except (asyncio.CancelledError, Exception):
                pass
            finally:
                app.exit()

        # Start watcher task
        watcher = asyncio.create_task(exit_app_when_done())

        try:
            # Run app (listens for Escape) until operation completes or cancelled
            await app.run_async()

            # Return result or re-raise exception from operation
            return await operation

        finally:
            self._current_task = None
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
