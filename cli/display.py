"""Message display formatting for the TUI using Rich markup.

This module provides simple formatting functions that return Rich markup strings
for display in Textual's RichLog widget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.console import Group, RenderableType
from rich.markdown import ListItem, Markdown, MarkdownElement, TextElement
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.markdown import MarkdownContext, TableBodyElement, TableHeaderElement

from rich._loop import loop_first


class WideListItem(ListItem):
    """ListItem that doesn't truncate long lines.

    Rich's default ListItem constrains content to (max_width - 3), causing
    truncation of long bullet point text. This override uses a large width
    to prevent truncation, then strips trailing whitespace from each line.
    """

    @staticmethod
    def _strip_trailing_spaces(line: list[Segment]) -> list[Segment]:
        """Remove trailing whitespace segments from a line."""
        # Work backwards to find last non-whitespace segment
        result = list(line)
        while result and result[-1].text.isspace():
            result.pop()
        # Strip trailing spaces from the last segment if it exists
        if result and result[-1].text.endswith(" "):
            last = result[-1]
            result[-1] = Segment(last.text.rstrip(), last.style, last.control)
        return result

    def render_bullet(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        # Use large width to prevent truncation
        render_options = options.update(width=9999)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bullet_style = console.get_style("markdown.item.bullet", default="none")

        bullet = Segment(" • ", bullet_style)
        padding = Segment(" " * 3, bullet_style)
        new_line = Segment("\n")
        for first, line in loop_first(lines):
            yield bullet if first else padding
            yield from self._strip_trailing_spaces(line)
            yield new_line

    def render_number(
        self,
        console: "Console",
        options: "ConsoleOptions",
        number: int,
        last_number: int,
    ) -> "RenderResult":
        # Use large width to prevent truncation
        render_options = options.update(width=9999)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        number_style = console.get_style("markdown.item.number", default="none")

        number_width = len(str(last_number)) + 2
        number_str = f"{number}".rjust(number_width - 1) + " "
        number_seg = Segment(number_str, number_style)
        padding = Segment(" " * number_width, number_style)
        new_line = Segment("\n")
        for first, line in loop_first(lines):
            yield number_seg if first else padding
            yield from self._strip_trailing_spaces(line)
            yield new_line


class SimpleHeading(TextElement):
    """Renders headings with # prefix preserved, bold, left-aligned.

    Instead of Rich's default centered/underlined headers, this renders
    headers as plain bold text with the original markdown # prefix visible.
    Example: "# Header" renders as bold "# Header" (left-aligned).
    """

    @classmethod
    def create(cls, markdown: Markdown, token: object) -> "SimpleHeading":
        """Create a SimpleHeading from a markdown token."""
        # token.tag is "h1", "h2", etc.
        return cls(getattr(token, "tag", "h1"))

    def __init__(self, tag: str) -> None:
        self.tag = tag  # "h1", "h2", etc.
        self.level = int(tag[1])  # Extract level from "h1" → 1
        super().__init__()

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        # Prepend # symbols based on heading level
        prefix = "#" * self.level + " "
        text = Text(prefix, style="bold")
        text.append_text(self.text)
        text.stylize("bold")
        yield text


class SimpleCodeBlock(TextElement):
    """Renders code blocks without left padding.

    Rich's default CodeBlock uses padding=1, which adds a space on the left
    of each line. This causes issues when copying code - the extra space
    gets included, breaking Python indentation and other whitespace-sensitive code.
    """

    style_name = "markdown.code_block"

    @classmethod
    def create(cls, markdown: Markdown, token: object) -> "SimpleCodeBlock":
        """Create a SimpleCodeBlock from a markdown token."""
        node_info = getattr(token, "info", "") or ""
        lexer_name = node_info.partition(" ")[0]
        return cls(lexer_name or "text", markdown.code_theme)

    def __init__(self, lexer_name: str, theme: str) -> None:
        self.lexer_name = lexer_name
        self.theme = theme

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        code = str(self.text).rstrip()
        # padding=0 removes the left space that causes copy issues
        # word_wrap=False lets terminal handle wrapping for better copy-paste
        syntax = Syntax(
            code, self.lexer_name, theme=self.theme, word_wrap=False, padding=0
        )
        yield syntax


class SimpleTableElement(MarkdownElement):
    """Table element with MINIMAL box style for cleaner appearance."""

    def __init__(self) -> None:
        self.header: TableHeaderElement | None = None
        self.body: TableBodyElement | None = None

    def on_child_close(
        self, context: "MarkdownContext", child: MarkdownElement
    ) -> bool:
        from rich.markdown import TableBodyElement, TableHeaderElement

        if isinstance(child, TableHeaderElement):
            self.header = child
        elif isinstance(child, TableBodyElement):
            self.body = child
        return False

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        table = Table(box=box.SQUARE)

        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                table.add_column(column.content)

        if self.body is not None:
            for row in self.body.rows:
                row_content = [element.content for element in row.cells]
                table.add_row(*row_content)

        yield table


class LimitedMarkdown(Markdown):
    """Markdown with simplified header and code block styles.

    Overrides:
    - heading_open: Uses SimpleHeading (preserves # prefix, left-aligned, bold)
    - code_block: Uses SimpleCodeBlock (no left padding for clean copying)
    - fence: Uses SimpleCodeBlock (fenced code blocks like ```python)
    - table_open: Uses SimpleTableElement (MINIMAL box style)
    - list_item_open: Uses WideListItem (no truncation of long lines)
    """

    elements = {
        **Markdown.elements,
        "heading_open": SimpleHeading,
        "code_block": SimpleCodeBlock,
        "fence": SimpleCodeBlock,
        "table_open": SimpleTableElement,
        "list_item_open": WideListItem,
    }


def format_user_message(text: str) -> RenderableType:
    """Format a user message with prompt prefix, subtle background, and empty line before."""
    return Group(Text(""), Text(f"> {text}", style="on grey30"))


def format_assistant_message(text: str) -> RenderableType:
    """Format an assistant message with markdown rendering.

    Uses LimitedMarkdown to render with simplified header style:
    - Headers show as bold text with # prefix preserved (left-aligned)
    - Bold (**text**) and italic (*text*)
    - Code blocks with syntax highlighting
    - Lists (ordered and unordered)
    - Links and more
    """
    return LimitedMarkdown(text.rstrip(), code_theme="native")


def format_thinking_message(thinking: str) -> Text:
    """Format thinking content with magenta/dim styling."""
    # Truncate long thinking content
    lines = thinking.split("\n")
    max_lines = 15
    if len(lines) > max_lines:
        content = (
            "\n".join(lines[:max_lines])
            + f"\n... ({len(lines) - max_lines} more lines)"
        )
    else:
        content = thinking

    result = Text()
    result.append("Thinking: ", style="magenta italic")
    result.append(content, style="dim italic")
    return result


def format_tool_call(name: str, params: dict[str, object]) -> Text:
    """Format a tool call as a function call."""
    result = Text()
    result.append(name, style="yellow bold")
    result.append("(", style="dim")

    items = list(params.items())
    for i, (key, value) in enumerate(items):
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."

        result.append(key, style="cyan")
        result.append(": ", style="dim")
        result.append(val_str)

        if i < len(items) - 1:
            result.append(", ", style="dim")

    result.append(")", style="dim")
    return result


def format_tool_result(result: str, is_error: bool = False) -> Text:
    """Format a tool result with diff-aware styling."""
    lines = result.split("\n")
    max_lines = 100
    if len(lines) > max_lines:
        truncated = True
        truncate_count = len(lines) - max_lines
        lines = lines[:max_lines]
    else:
        truncated = False
        truncate_count = 0

    # overflow="fold" allows long lines to wrap at terminal width
    result_text = Text(overflow="fold")

    if is_error:
        content = "\n".join(lines)
        result_text.append(content, style="red dim")
    else:
        # Diff-aware coloring
        for i, line in enumerate(lines):
            if line.startswith("  -  "):
                result_text.append(line, style="on red")
            elif line.startswith("  +  "):
                result_text.append(line, style="on green")
            else:
                result_text.append(line, style="dim")
            if i < len(lines) - 1:
                result_text.append("\n")

    if truncated:
        result_text.append(f"\n... ({truncate_count} more lines)", style="dim")

    # Add empty line to separate from next tool call
    result_text.append("\n")

    return result_text


def format_system_message(text: str) -> Text:
    """Format a system message with dim styling."""
    result = Text()
    result.append(f"[{text}]", style="dim")
    return result


def format_error_message(text: str) -> Text:
    """Format an error message with red styling."""
    result = Text()
    result.append("Error: ", style="red bold")
    result.append(text, style="red")
    return result


def format_thinking_separator() -> Text:
    """Format a short separator line between thinking and response."""
    return Text("─" * 5, style="dim")


def format_token_count(
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> Text:
    """Format token usage statistics.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_creation_tokens: Number of cache creation tokens (optional)
        cache_read_tokens: Number of cache read tokens (optional)

    Returns:
        Formatted Text object with token statistics
    """
    result = Text()
    result.append("Tokens: ", style="dim")
    result.append(f"in={input_tokens}", style="cyan dim")
    result.append(" ", style="dim")
    result.append(f"out={output_tokens}", style="green dim")

    # Add cache info if present
    if cache_creation_tokens > 0:
        result.append(" ", style="dim")
        result.append(f"cache_write={cache_creation_tokens}", style="yellow dim")
    if cache_read_tokens > 0:
        result.append(" ", style="dim")
        result.append(f"cache_read={cache_read_tokens}", style="blue dim")

    # Add total
    total = input_tokens + output_tokens
    result.append(" ", style="dim")
    result.append(f"total={total}", style="magenta dim")

    return result
