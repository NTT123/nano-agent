"""High-level message rendering helpers."""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text

from . import display


class MessageRenderer:
    """Renderer for UI message content."""

    def user(self, text: str) -> RenderableType:
        return display.format_user_message(text)

    def assistant(self, text: str) -> RenderableType:
        return display.format_assistant_message(text)

    def thinking(self, thinking: str) -> Text:
        return display.format_thinking_message(thinking)

    def thinking_separator(self) -> Text:
        return display.format_thinking_separator()

    def token_count(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
        cost: float = 0.0,
    ) -> Text:
        return display.format_token_count(
            input_tokens,
            output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
        )

    def tool_call(self, name: str, params: dict[str, object]) -> Text:
        return display.format_tool_call(name, params)

    def tool_result(self, result: str, is_error: bool = False) -> Text:
        return display.format_tool_result(result, is_error=is_error)

    def system(self, text: str) -> Text:
        return display.format_system_message(text)

    def error(self, text: str) -> Text:
        return display.format_error_message(text)
