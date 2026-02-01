"""Yes/No confirmation prompt element.

Used for permission prompts, confirmations, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ActiveElement, InputEvent
from .terminal import ANSI


@dataclass
class ConfirmPrompt(ActiveElement[bool | None]):
    """Yes/No confirmation prompt.

    Returns True (y), False (n), or None (Escape/cancel).

    Optionally displays preview lines above the prompt (useful for
    showing diffs, file contents, etc.).
    """

    message: str = "Confirm?"
    preview_lines: list[str] = field(default_factory=list)
    max_preview_lines: int = 10
    _response: str = ""  # Tracks what user typed for display

    def get_lines(self) -> list[str]:
        lines = []
        # Add preview with visual separator
        if self.preview_lines:
            lines.append(ANSI.DIM + "┌" + "─" * 50 + ANSI.RESET)
            for line in self.preview_lines[: self.max_preview_lines]:
                # Truncate long lines
                display_line = line[:48] if len(line) > 48 else line
                lines.append(ANSI.DIM + "│ " + ANSI.RESET + display_line)
            if len(self.preview_lines) > self.max_preview_lines:
                lines.append(
                    ANSI.DIM
                    + f"│ ... ({len(self.preview_lines) - self.max_preview_lines} more lines)"
                    + ANSI.RESET
                )
            lines.append(ANSI.DIM + "└" + "─" * 50 + ANSI.RESET)

        # Highlighted prompt: message in yellow, options in dim
        prompt_line = f"{ANSI.YELLOW}{ANSI.BOLD}{self.message}{ANSI.RESET} {ANSI.DIM}[y/n/esc]:{ANSI.RESET} {ANSI.GREEN}{self._response}{ANSI.RESET}"
        lines.append(prompt_line)
        return lines

    def handle_input(self, event: InputEvent) -> tuple[bool, bool | None]:
        char = (event.char or "").lower()
        if char == "y":
            self._response = "y"
            return (True, True)
        elif char == "n":
            self._response = "n"
            return (True, False)
        elif event.key == "Escape" or (event.ctrl and event.char == "c"):
            self._response = "esc"
            return (True, None)
        return (False, None)
