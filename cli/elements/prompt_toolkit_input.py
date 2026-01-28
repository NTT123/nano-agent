"""Rich text input using prompt_toolkit.

This element uses prompt_toolkit's PromptSession for:
- Multi-line editing (\\ + Enter for newline)
- History navigation (up/down arrows)
- Emacs/Vi key bindings
- Line editing (Ctrl+A/E, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import shutil
import math
from typing import Any, Callable

from .base import ActiveElement, InputEvent
from .terminal import ANSI


def _visual_len(s: str) -> int:
    """Return visual length accounting for wide chars."""
    from wcwidth import wcwidth

    total = 0
    for ch in s:
        w = wcwidth(ch)
        total += w if w > 0 else 0
    return total


@dataclass
class PromptToolkitInput(ActiveElement[str | None]):
    """Text input using prompt_toolkit.

    This is a self-managed element that uses prompt_toolkit's PromptSession
    for rich text editing features:
    - Multi-line editing (\\ + Enter for newline, Enter to submit)
    - History (up/down arrows)
    - Emacs/Vi key bindings
    - Syntax highlighting (if configured)

    Returns the entered text on Enter, or None on Ctrl+C/Ctrl+D/Escape.
    """

    prompt: str = "> "
    multiline: bool = True
    history_file: str | None = None
    on_toggle_auto_accept: Callable[[], None] | None = None
    get_bottom_toolbar: Callable[[], str] | None = None
    _session: Any = None  # PromptSession instance
    _prompt_len: int = 0

    def is_self_managed(self) -> bool:
        return True

    def get_lines(self) -> list[str]:
        # Not used for self-managed elements
        return []

    def handle_input(self, event: InputEvent) -> tuple[bool, str | None]:
        # Not used for self-managed elements
        return (True, None)

    async def run_async(self) -> str | None:
        """Run prompt_toolkit input."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys

        # Disable CPR to prevent terminal garbage like "9;1R" appearing as input
        os.environ["PROMPT_TOOLKIT_NO_CPR"] = "1"

        # Set up key bindings
        bindings = KeyBindings()
        cancelled = False

        @bindings.add(Keys.Escape)
        def handle_escape(event: Any) -> None:
            """Escape to cancel."""
            nonlocal cancelled
            cancelled = True
            event.app.exit(result="")

        @bindings.add(Keys.ControlJ)
        def handle_ctrl_j(event: Any) -> None:
            """Ignore Ctrl+J (do not submit)."""
            return

        shift_enter = getattr(Keys, "ShiftEnter", None)
        if shift_enter:
            @bindings.add(shift_enter)
            def handle_shift_enter(event: Any) -> None:
                """Shift+Enter inserts a newline."""
                event.current_buffer.insert_text("\n")

        # Shift+Tab to toggle auto-accept mode
        if self.on_toggle_auto_accept:
            @bindings.add(Keys.BackTab)
            def handle_shift_tab(event: Any) -> None:
                """Shift+Tab toggles auto-accept mode."""
                if self.on_toggle_auto_accept:
                    self.on_toggle_auto_accept()
                    # Force toolbar refresh to show updated status
                    event.app.invalidate()

        @bindings.add(Keys.Enter)
        def handle_enter(event: Any) -> None:
            """Enter: if line ends with \\, continue on next line; otherwise submit."""
            buffer = event.current_buffer
            text = buffer.text

            if text.endswith("\\"):
                # Remove the backslash and insert newline
                buffer.delete_before_cursor(1)
                buffer.insert_text("\n")
            else:
                # Submit
                buffer.validate_and_handle()

        # Set up session with optional history
        self._prompt_len = _visual_len(self.prompt)
        prompt_continuation = " " * self._prompt_len
        if self.history_file:
            history = FileHistory(self.history_file)
            self._session = PromptSession(
                history=history,
                key_bindings=bindings,
                prompt_continuation=prompt_continuation,
            )
        else:
            self._session = PromptSession(
                key_bindings=bindings,
                prompt_continuation=prompt_continuation,
            )

        try:
            # Use rprompt (right-side prompt) for status - works with CPR disabled
            from prompt_toolkit.formatted_text import HTML

            def get_rprompt():
                try:
                    if self.get_bottom_toolbar:
                        text = self.get_bottom_toolbar()
                        return HTML(f"<style bg='#333333' fg='#aaaaaa'> {text} </style>")
                    return None
                except Exception:
                    return None

            result = await self._session.prompt_async(
                self.prompt,
                multiline=self.multiline,
                rprompt=get_rprompt,
            )
            if cancelled:
                self._clear_prompt_lines("")
                return ""
            self._clear_prompt_lines(result)
            return result
        except (EOFError, KeyboardInterrupt):
            return None

    def _estimate_line_count(self, text: str) -> int:
        width = max(shutil.get_terminal_size().columns, 1)
        if width <= self._prompt_len:
            width = self._prompt_len + 1
        available = width - self._prompt_len
        lines = text.split("\n") if text is not None else [""]
        total = 0
        for line in lines:
            line_len = _visual_len(line)
            wraps = max(1, math.ceil(line_len / max(available, 1)))
            total += wraps
        # rprompt is on same line as prompt, no extra lines needed
        return max(total, 1)

    def _clear_prompt_lines(self, text: str) -> None:
        num_lines = self._estimate_line_count(text)
        if num_lines <= 0:
            return
        ANSI.move_up(num_lines)
        for i in range(num_lines):
            ANSI.clear_line()
            if i < num_lines - 1:
                ANSI.move_down(1)
        if num_lines > 1:
            ANSI.move_up(num_lines - 1)
        ANSI._write(ANSI.CARRIAGE_RETURN)
