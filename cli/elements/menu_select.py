"""Menu selection element.

Allows user to select from a list of options using arrow keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ActiveElement, InputEvent


@dataclass
class MenuSelect(ActiveElement[str | list[str] | None]):
    """Menu selection with arrow keys.

    Navigate with j/k or up/down arrows.
    - Single-select: Enter selects the highlighted item.
    - Multi-select: Space toggles, Enter confirms selections.
    Returns selected item(s), or None on Escape.
    """

    title: str = "Select:"
    options: list[str] = field(default_factory=list)
    selected: int = 0
    multi_select: bool = False
    selected_indices: set[int] = field(default_factory=set)

    def get_lines(self) -> list[str]:
        lines = [self.title]
        for i, opt in enumerate(self.options):
            cursor = "→ " if i == self.selected else "  "
            if self.multi_select:
                mark = "[x]" if i in self.selected_indices else "[ ]"
                lines.append(f"{cursor}{mark} {opt}")
            else:
                lines.append(f"{cursor}{opt}")
        lines.append("")
        if self.multi_select:
            lines.append("[↑/↓] move  [Space] toggle  [Enter] confirm  [Esc] cancel")
        else:
            lines.append("[↑/↓] move  [Enter] select  [Esc] cancel")
        return lines

    def handle_input(self, event: InputEvent) -> tuple[bool, str | list[str] | None]:
        if event.key == "Enter":
            if not self.options:
                return (True, None)
            if self.multi_select:
                if not self.selected_indices:
                    self.selected_indices.add(self.selected)
                selected = [
                    self.options[i]
                    for i in range(len(self.options))
                    if i in self.selected_indices
                ]
                return (True, selected)
            return (True, self.options[self.selected])
        elif event.key == "Escape" or (event.ctrl and event.char == "c"):
            return (True, None)
        elif self.multi_select and event.char == " ":
            if self.selected in self.selected_indices:
                self.selected_indices.remove(self.selected)
            else:
                self.selected_indices.add(self.selected)
            return (False, None)
        elif event.char == "j" or event.key == "Down":
            self.selected = min(self.selected + 1, len(self.options) - 1)
            return (False, None)
        elif event.char == "k" or event.key == "Up":
            self.selected = max(self.selected - 1, 0)
            return (False, None)
        return (False, None)
