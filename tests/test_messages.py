"""Tests for UIMessage in cli/messages.py.

Covers:
- visual_line_count() method for accurate line counting
- Handling of multiline content with embedded newlines
- Rich renderable content handling
"""

from __future__ import annotations

import pytest

from cli.messages import MessageStatus, RenderItem, UIMessage


class TestVisualLineCount:
    """Tests for UIMessage.visual_line_count() method."""

    def test_single_line_string(self) -> None:
        """Single line content should return count of 1."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="Hello"))
        assert msg.visual_line_count() == 1

    def test_multiline_string_two_lines(self) -> None:
        """Content with one newline should return count of 2."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="line1\nline2"))
        assert msg.visual_line_count() == 2

    def test_multiline_string_three_lines(self) -> None:
        """Content with two newlines should return count of 3."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="line1\nline2\nline3"))
        assert msg.visual_line_count() == 3

    def test_multiple_render_items(self) -> None:
        """Multiple RenderItems should sum their line counts."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="Hello"))  # 1 line
        msg.output_buffer.append(RenderItem(content="line1\nline2"))  # 2 lines
        msg.output_buffer.append(RenderItem(content="World"))  # 1 line
        assert msg.visual_line_count() == 4

    def test_empty_buffer(self) -> None:
        """Empty output buffer should return count of 0."""
        msg = UIMessage(message_type="user")
        assert msg.visual_line_count() == 0

    def test_empty_string_content(self) -> None:
        """Empty string content should still count as 1 line."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content=""))
        assert msg.visual_line_count() == 1

    def test_trailing_newline(self) -> None:
        """Trailing newline should add to line count."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="line1\nline2\n"))
        # "line1\nline2\n" has 2 newlines, so 3 lines
        assert msg.visual_line_count() == 3

    def test_multiple_consecutive_newlines(self) -> None:
        """Multiple consecutive newlines should each add a line."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="line1\n\n\nline2"))
        # 3 newlines = 4 lines
        assert msg.visual_line_count() == 4

    def test_only_newlines(self) -> None:
        """Content with only newlines."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="\n\n"))
        # 2 newlines = 3 lines (empty lines)
        assert msg.visual_line_count() == 3


class TestVisualLineCountWithRichContent:
    """Tests for visual_line_count() with Rich renderable content."""

    def test_rich_text_single_line(self) -> None:
        """Rich Text object without newlines should count as 1."""
        from rich.text import Text

        msg = UIMessage(message_type="assistant")
        msg.output_buffer.append(RenderItem(content=Text("Hello")))
        assert msg.visual_line_count() == 1

    def test_rich_text_multiline(self) -> None:
        """Rich Text object with newlines should count correctly."""
        from rich.text import Text

        msg = UIMessage(message_type="assistant")
        msg.output_buffer.append(RenderItem(content=Text("line1\nline2\nline3")))
        assert msg.visual_line_count() == 3

    def test_mixed_string_and_rich(self) -> None:
        """Mix of string and Rich content."""
        from rich.text import Text

        msg = UIMessage(message_type="assistant")
        msg.output_buffer.append(RenderItem(content="string line"))  # 1
        msg.output_buffer.append(RenderItem(content=Text("rich\ntext")))  # 2
        assert msg.visual_line_count() == 3


class TestUIMessageBasics:
    """Basic tests for UIMessage functionality."""

    def test_message_type(self) -> None:
        """Message type should be set correctly."""
        msg = UIMessage(message_type="user")
        assert msg.message_type == "user"

        msg2 = UIMessage(message_type="assistant")
        assert msg2.message_type == "assistant"

    def test_initial_status_is_active(self) -> None:
        """Initial status should be ACTIVE."""
        msg = UIMessage(message_type="user")
        assert msg.status == MessageStatus.ACTIVE

    def test_is_frozen_when_complete(self) -> None:
        """Message should be frozen when status is COMPLETE."""
        msg = UIMessage(message_type="user")
        assert not msg.is_frozen()

        msg.status = MessageStatus.COMPLETE
        assert msg.is_frozen()

    def test_append_content(self) -> None:
        """Content should be appendable to output buffer."""
        msg = UIMessage(message_type="user")
        msg.output_buffer.append(RenderItem(content="Hello"))
        assert len(msg.output_buffer) == 1
        assert msg.output_buffer[0].content == "Hello"

    def test_multiple_content_items(self) -> None:
        """Multiple content items should be stored."""
        msg = UIMessage(message_type="assistant")
        msg.output_buffer.append(RenderItem(content="Part 1"))
        msg.output_buffer.append(RenderItem(content="Part 2"))
        msg.output_buffer.append(RenderItem(content="Part 3"))
        assert len(msg.output_buffer) == 3


class TestRenderItem:
    """Tests for RenderItem dataclass."""

    def test_render_item_with_string(self) -> None:
        """RenderItem should hold string content."""
        item = RenderItem(content="test")
        assert item.content == "test"

    def test_render_item_with_style(self) -> None:
        """RenderItem should hold optional style."""
        item = RenderItem(content="test", style="bold red")
        assert item.content == "test"
        assert item.style == "bold red"

    def test_render_item_with_rich_content(self) -> None:
        """RenderItem should hold Rich content."""
        from rich.text import Text

        text = Text("styled", style="green")
        item = RenderItem(content=text)
        assert item.content == text
