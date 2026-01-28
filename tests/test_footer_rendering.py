from __future__ import annotations

from cli.elements.footer import TerminalFooter
from cli.elements.terminal import ANSI


class _FakeRegion:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
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
