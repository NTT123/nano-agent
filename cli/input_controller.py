"""Input orchestration using active elements with unified footer."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable

from nano_agent.cancellation import CancellationChoice, ToolExecutionBatch

from .elements import (
    CancellationMenu,
    ConfirmPrompt,
    FooterElementManager,
    FooterInput,
    MenuSelect,
    TerminalFooter,
)
from .elements.terminal import ANSI, RawInputReader


@dataclass
class InputController:
    """Unified input handling with footer-based status bar.

    Manages:
    - Text input prompts via FooterInput
    - Confirmation dialogs via ConfirmPrompt
    - Selection menus via MenuSelect
    - Escape key detection during execution
    - Status bar updates (activity, tokens, auto-accept)
    """

    footer: TerminalFooter = field(default_factory=TerminalFooter)
    elements: FooterElementManager = field(init=False, repr=False)
    on_escape: Callable[[], None] | None = None
    on_toggle_auto_accept: Callable[[], None] | None = None

    _reader: RawInputReader | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _refresh_task: asyncio.Task[None] | None = field(
        default=None, init=False, repr=False
    )
    _task_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize the footer element manager."""
        self.elements = FooterElementManager(self.footer)

    def _is_tty(self) -> bool:
        try:
            return sys.stdin.isatty()
        except Exception:
            return False

    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self) -> "InputController":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.stop()

    async def start(self) -> bool:
        """Start raw input reader for escape detection and activate footer."""
        if self._running:
            return True
        if not self._is_tty():
            return False
        self._reader = RawInputReader()
        self._reader.start()
        self._running = True
        # Hide terminal cursor for entire app lifetime
        # (FooterInput renders its own cursor via reverse video)
        ANSI.hide_cursor()
        # Activate footer for status bar display
        self.footer.activate()
        return True

    async def stop(self) -> None:
        """Stop raw input reader and deactivate footer."""
        # Stop refresh task if running
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        if self._reader:
            self._reader.stop()
        self._reader = None
        self._running = False
        # Deactivate footer even if we weren't running (prompt may have activated it)
        self.footer.deactivate()
        # Restore terminal cursor on exit
        ANSI.show_cursor()

    async def _refresh_loop(self) -> None:
        """Background task to refresh spinner animation during activity."""
        while True:
            await asyncio.sleep(0.1)
            if self.footer.is_active() and self.footer.status.activity:
                self.footer.render()

    async def poll_escape(self, timeout: float = 0.05) -> None:
        """Poll for escape key press and other hotkeys during execution."""
        if not self._running or not self._reader:
            return
        loop = asyncio.get_event_loop()
        try:
            event = await loop.run_in_executor(
                None, self._reader.read_nonblocking, timeout
            )
        except (OSError, ValueError):
            return  # Reader was stopped or fd invalid
        if event:
            if event.key == "Escape":
                if self.on_escape:
                    self.on_escape()
            elif event.key == "BackTab":
                # Shift+Tab toggles auto-accept during execution
                if self.on_toggle_auto_accept:
                    self.on_toggle_auto_accept()
                    # Update footer to reflect change
                    self.footer.render()

    def update_status(
        self,
        auto_accept: bool | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
        cost: float | None = None,
    ) -> None:
        """Update status bar values."""
        self.footer.update_status(
            auto_accept=auto_accept,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            cost=cost,
        )

    async def set_activity(self, activity: str | None) -> None:
        """Set activity text (shows spinner when not None).

        When activity is set:
        - Spinner shows with activity text
        - Status bar includes escape hint

        When activity is None:
        - No spinner
        - Just status bar with auto-accept and token counts

        Uses _task_lock to protect against concurrent task management.
        """
        self.footer.set_activity(activity)
        # Start/stop refresh task for spinner animation (protected by lock)
        async with self._task_lock:
            if activity and (self._refresh_task is None or self._refresh_task.done()):
                self._refresh_task = asyncio.create_task(self._refresh_loop())
            elif not activity and self._refresh_task and not self._refresh_task.done():
                self._refresh_task.cancel()

    def resync_footer(self) -> None:
        """Re-sync footer position after external content is printed.

        Call this after using console.print() or other output that moves
        the terminal cursor while the footer is active.
        """
        self.footer.resync_position()

    def pause_footer(self) -> None:
        """Temporarily hide footer to allow normal printing.

        Use this before a sequence of prints, then call resume_footer().
        More efficient than resync_footer() for multiple prints.
        """
        self.footer.pause()

    def resume_footer(self) -> None:
        """Restore footer after pause_footer()."""
        self.footer.resume()

    def has_overwritable_content(self) -> bool:
        """Check if footer has content that can be overwritten."""
        return self.footer.has_content() and self.footer.is_active()

    def prepare_content_overwrite(self) -> int:
        """Prepare to overwrite footer content. Returns line count."""
        return self.footer.overwrite_content_with_message()

    def finish_content_overwrite(self, lines_printed: int) -> None:
        """Complete the overwrite transition."""
        self.footer.finish_content_overwrite(lines_printed)

    def transition_to_message(self, render_callback: Callable[[], int]) -> None:
        """Atomically transition from input content to rendered message.

        This is the preferred API for seamless user input -> message transition.
        The render_callback should render content and return the number of lines printed.

        Args:
            render_callback: Function that renders content and returns lines printed
        """
        self.footer.transition_to_message(render_callback)

    async def prompt_text(self, prompt: str = "> ") -> str | None:
        """Prompt for user text input using FooterInput.

        Note: This method temporarily pauses the raw input reader to avoid
        conflicts with FooterInput's own input handling. The reader is
        restarted in the finally block. The `_running` flag remains True
        because this is a temporary pause, not a full stop.
        """
        try:
            # Stop raw reader while FooterInput handles input
            was_running = self._running
            if was_running and self._reader:
                self._reader.stop()

            history_file = os.path.expanduser("~/.nano-cli-history")
            try:
                text = await self.elements.run(
                    FooterInput(
                        prompt=prompt,
                        history_file=history_file,
                        on_toggle_auto_accept=self.on_toggle_auto_accept,
                    )
                )
            finally:
                # Restart raw reader
                if was_running and self._reader:
                    self._reader.start()

            return text
        except KeyboardInterrupt:
            return ""

    async def confirm(
        self, message: str, preview: Iterable[str] | None = None
    ) -> bool | None:
        """Prompt for confirmation."""
        preview_lines = list(preview) if preview else []
        return await self.elements.run(
            ConfirmPrompt(message=message, preview_lines=preview_lines)
        )

    async def select(
        self, title: str, options: list[str], *, multi_select: bool = False
    ) -> str | list[str] | None:
        """Prompt for selection from a list."""
        return await self.elements.run(
            MenuSelect(title=title, options=options, multi_select=multi_select)
        )

    async def show_cancellation_menu(
        self, batch: ToolExecutionBatch
    ) -> CancellationChoice | None:
        """Show cancellation menu with tool execution state.

        Args:
            batch: The tool execution batch with current state

        Returns:
            User's choice for how to proceed, or None if cancelled
        """
        return await self.elements.run(CancellationMenu(batch=batch))
