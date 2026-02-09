"""Tests for the Discord bot modules."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nano_agent import DAG, Role, TextContent
from nano_agent.data_structures import Message, Response, ToolUseContent, Usage, ContentBlock

from .bot_state import (
    BotState,
    chunk_message,
    is_empty_assistant_message,
    serialize_content_blocks,
    serialize_text_contents,
    truncate,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockAttachment:
    url: str = "https://example.com/file.png"


@dataclass
class MockAuthor:
    id: int = 1001
    bot: bool = False
    name: str = "TestUser"

    def __str__(self) -> str:
        return self.name


@dataclass
class MockMessage:
    id: int = 9999
    author: MockAuthor = field(default_factory=MockAuthor)
    attachments: list[MockAttachment] = field(default_factory=list)
    content: str = "hello"
    webhook_id: int | None = None


class MockChannel:
    """Captures messages sent through channel.send()."""

    def __init__(self, channel_id: int = 42):
        self.id = channel_id
        self.sent_messages: list[str] = []
        self.name = "test-channel"

    async def send(self, content: str | None = None, **kwargs: Any) -> None:
        if content is not None:
            self.sent_messages.append(content)

    def typing(self):
        return _AsyncNoop()


class _AsyncNoop:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any):
        pass


class MockAPI:
    """Returns canned API responses."""

    def __init__(self, responses: list[Response] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0

    async def send(self, dag: DAG) -> Response:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = Response(
                id="mock-id",
                model="mock-model",
                role=Role.ASSISTANT,
                content=[TextContent(text="Done.")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        self._call_count += 1
        return resp


# ---------------------------------------------------------------------------
# Tier 1: Pure functions
# ---------------------------------------------------------------------------


class TestChunkMessage:
    def test_short_message_returns_single_chunk(self):
        assert chunk_message("hello") == ["hello"]

    def test_exact_limit_returns_single_chunk(self):
        text = "a" * 2000
        assert chunk_message(text) == [text]

    def test_long_message_splits(self):
        text = "line\n" * 500  # 2500 chars
        chunks = chunk_message(text, limit=100)
        assert all(len(c) <= 100 for c in chunks)
        # Recombined content should match original (modulo stripped newlines)
        assert "".join(chunks).replace("\n", "") == text.replace("\n", "")

    def test_no_line_break_splits_at_limit(self):
        text = "a" * 3000
        chunks = chunk_message(text, limit=2000)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 2000
        assert chunks[1] == "a" * 1000

    def test_custom_limit(self):
        text = "abcdef"
        chunks = chunk_message(text, limit=3)
        assert chunks == ["abc", "def"]


class TestTruncate:
    def test_short_text_unchanged(self):
        assert truncate("hi", 10) == "hi"

    def test_exact_limit_unchanged(self):
        assert truncate("hello", 5) == "hello"

    def test_long_text_truncated(self):
        assert truncate("hello world", 5) == "hello..."

    def test_default_limit(self):
        text = "x" * 300
        result = truncate(text)
        assert result == "x" * 200 + "..."


class TestIsEmptyAssistantMessage:
    def test_empty_string_content(self):
        msg = Message(role=Role.ASSISTANT, content="")
        assert is_empty_assistant_message(msg) is True

    def test_whitespace_string_content(self):
        msg = Message(role=Role.ASSISTANT, content="   ")
        assert is_empty_assistant_message(msg) is True

    def test_empty_list_content(self):
        msg = Message(role=Role.ASSISTANT, content=[])
        assert is_empty_assistant_message(msg) is True

    def test_nonempty_content(self):
        msg = Message(role=Role.ASSISTANT, content="hello")
        assert is_empty_assistant_message(msg) is False

    def test_user_message_always_false(self):
        msg = Message(role=Role.USER, content="")
        assert is_empty_assistant_message(msg) is False


class TestUtcNowIso:
    def test_returns_string(self):
        result = utc_now_iso()
        assert isinstance(result, str)
        assert "T" in result  # ISO format has T separator


class TestSerializeContentBlocks:
    def test_text_block(self):
        blocks = serialize_content_blocks([TextContent(text="hi")])
        assert blocks == [{"type": "text", "text": "hi"}]

    def test_tool_use_block(self):
        block = ToolUseContent(id="t1", name="Bash", input={"command": "ls"})
        result = serialize_content_blocks([block])
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "Bash"

    def test_unknown_block(self):
        result = serialize_content_blocks(["some_string"])
        assert result[0]["type"] == "str"


class TestSerializeTextContents:
    def test_text_content(self):
        assert serialize_text_contents([TextContent(text="a")]) == ["a"]

    def test_mixed(self):
        result = serialize_text_contents([TextContent(text="a"), 42])
        assert result == ["a", "42"]


# ---------------------------------------------------------------------------
# Tier 2: BotState
# ---------------------------------------------------------------------------


class TestBotStateQueue:
    def _make_state(self, tmp_path: Path) -> BotState:
        return BotState(state_root=tmp_path / "state")

    def test_enqueue_and_peek(self, tmp_path: Path):
        s = self._make_state(tmp_path)
        msg = MockMessage(content="hi")
        s.enqueue_user_message(1, msg, "hi")  # type: ignore[arg-type]
        items = s.peek_user_messages(1)
        assert len(items) == 1
        assert items[0]["content"] == "hi"

    def test_dequeue(self, tmp_path: Path):
        s = self._make_state(tmp_path)
        msg = MockMessage(content="one")
        s.enqueue_user_message(1, msg, "one")  # type: ignore[arg-type]
        s.enqueue_user_message(1, MockMessage(content="two"), "two")  # type: ignore[arg-type]
        items = s.dequeue_user_messages(1, count=1)
        assert len(items) == 1
        assert items[0]["content"] == "one"
        assert len(s.get_channel_queue(1)) == 1

    def test_dequeue_multiple(self, tmp_path: Path):
        s = self._make_state(tmp_path)
        for i in range(5):
            s.enqueue_user_message(1, MockMessage(content=f"m{i}"), f"m{i}")  # type: ignore[arg-type]
        items = s.dequeue_user_messages(1, count=3)
        assert len(items) == 3
        assert len(s.get_channel_queue(1)) == 2

    def test_clear_queue(self, tmp_path: Path):
        s = self._make_state(tmp_path)
        s.enqueue_user_message(1, MockMessage(), "x")  # type: ignore[arg-type]
        s.clear_user_queue(1)
        assert len(s.get_channel_queue(1)) == 0

    def test_queue_ids_monotonic(self, tmp_path: Path):
        s = self._make_state(tmp_path)
        ids = []
        for i in range(3):
            q = s.enqueue_user_message(1, MockMessage(content=f"m{i}"), f"m{i}")  # type: ignore[arg-type]
            ids.append(q["queue_id"])
        assert ids == sorted(ids)
        assert len(set(ids)) == 3

    def test_peek_empty(self, tmp_path: Path):
        s = self._make_state(tmp_path)
        assert s.peek_user_messages(999) == []

    def test_dequeue_empty(self, tmp_path: Path):
        s = self._make_state(tmp_path)
        assert s.dequeue_user_messages(999) == []


class TestBotStateSession:
    def test_get_creates_session(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        dag = s.get_session(1, tools=[])
        assert isinstance(dag, DAG)
        # Second call returns same instance
        assert s.get_session(1) is dag

    def test_set_session(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        dag = DAG().system("test")
        s.set_session(1, dag)
        assert s.sessions[1] is dag


class TestBotStateQueuePersistence:
    def test_save_and_reload(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        s.enqueue_user_message(1, MockMessage(content="persisted"), "persisted")  # type: ignore[arg-type]

        # Create a new state pointing at same directory
        s2 = BotState(state_root=tmp_path / "state")
        queue = s2.get_channel_queue(1)
        assert len(queue) == 1
        assert queue[0]["content"] == "persisted"

    def test_persisted_channel_ids(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        s.enqueue_user_message(10, MockMessage(), "a")  # type: ignore[arg-type]
        s.enqueue_user_message(20, MockMessage(), "b")  # type: ignore[arg-type]
        ids = s.persisted_channel_ids()
        assert set(ids) == {10, 20}

    def test_no_state_dir(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "nonexistent")
        assert s.persisted_channel_ids() == []


class TestBotStateSanitizeDag:
    def test_no_change_needed(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        dag = DAG().system("test").user("hello").assistant("world")
        result = s.sanitize_dag_for_api(dag, 1)
        assert result is dag  # Same object, nothing removed

    def test_removes_empty_assistant(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        dag = DAG().system("test").user("hello").assistant("").user("again")
        result = s.sanitize_dag_for_api(dag, 1)
        messages = result.to_messages()
        for msg in messages:
            if msg.role == Role.ASSISTANT:
                assert msg.content != ""


class TestBotStateBuildQueueRuntimeNote:
    def test_empty_queue(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        note = s.build_queue_runtime_note(1)
        assert "no queued user messages" in note

    def test_with_messages(self, tmp_path: Path):
        s = BotState(state_root=tmp_path / "state")
        s.enqueue_user_message(1, MockMessage(content="hi"), "hi")  # type: ignore[arg-type]
        note = s.build_queue_runtime_note(1)
        assert "Pending count: 1" in note
        assert "hi" in note


# ---------------------------------------------------------------------------
# Tier 3: Tools (with mock state + mock channel)
# ---------------------------------------------------------------------------


class TestSendUserMessageTool:
    @pytest.mark.asyncio
    async def test_send(self):
        from .bot_tools import SendUserMessageInput, SendUserMessageTool

        channel = MockChannel()
        s = BotState(active_channel=channel, active_channel_id=42)  # type: ignore[arg-type]
        s.active_run_stats = {"outbound_messages": 0}
        tool = SendUserMessageTool(state=s)
        result = await tool(SendUserMessageInput(message="hello"))
        assert "Sent 1 chunk" in result.content.text
        assert channel.sent_messages == ["hello"]
        assert s.active_run_stats["outbound_messages"] == 1

    @pytest.mark.asyncio
    async def test_empty_message_error(self):
        from .bot_tools import SendUserMessageInput, SendUserMessageTool

        s = BotState(active_channel=MockChannel(), active_channel_id=42)  # type: ignore[arg-type]
        tool = SendUserMessageTool(state=s)
        result = await tool(SendUserMessageInput(message="   "))
        assert "Error" in result.content.text

    @pytest.mark.asyncio
    async def test_no_channel_error(self):
        from .bot_tools import SendUserMessageInput, SendUserMessageTool

        s = BotState()
        tool = SendUserMessageTool(state=s)
        result = await tool(SendUserMessageInput(message="hi"))
        assert "Error" in result.content.text


class TestSendFileTool:
    @pytest.mark.asyncio
    async def test_file_not_found(self):
        from .bot_tools import SendFileInput, SendFileTool

        s = BotState(active_channel=MockChannel())  # type: ignore[arg-type]
        tool = SendFileTool(state=s)
        result = await tool(SendFileInput(file_path="/nonexistent/file.txt"))
        assert "File not found" in result.content.text

    @pytest.mark.asyncio
    async def test_file_too_large(self, tmp_path: Path):
        from .bot_tools import SendFileInput, SendFileTool

        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"\x00" * (9 * 1024 * 1024))
        s = BotState(active_channel=MockChannel())  # type: ignore[arg-type]
        tool = SendFileTool(state=s)
        result = await tool(SendFileInput(file_path=str(big_file)))
        assert "too large" in result.content.text

    @pytest.mark.asyncio
    async def test_no_channel_error(self):
        from .bot_tools import SendFileInput, SendFileTool

        s = BotState()
        tool = SendFileTool(state=s)
        result = await tool(SendFileInput(file_path="/some/file"))
        assert "Error" in result.content.text


class TestClearContextTool:
    @pytest.mark.asyncio
    async def test_sets_flag(self):
        from .bot_tools import ClearContextTool

        channel = MockChannel(channel_id=7)
        s = BotState(active_channel=channel)  # type: ignore[arg-type]
        tool = ClearContextTool(state=s)
        result = await tool()
        assert 7 in s.clear_context_requested
        assert "cleared" in result.content.text.lower()


class TestDequeueUserMessagesTool:
    @pytest.mark.asyncio
    async def test_dequeue(self, tmp_path: Path):
        from .bot_tools import DequeueUserMessagesInput, DequeueUserMessagesTool

        s = BotState(active_channel_id=1, state_root=tmp_path / "state")
        s.active_run_stats = {"dequeued_messages": 0}
        s.enqueue_user_message(1, MockMessage(content="hi"), "hi")  # type: ignore[arg-type]
        tool = DequeueUserMessagesTool(state=s)
        result = await tool(DequeueUserMessagesInput(count=1))
        payload = json.loads(result.content.text)
        assert payload["dequeued_count"] == 1
        assert payload["remaining_count"] == 0


class TestPeekQueuedUserMessagesTool:
    @pytest.mark.asyncio
    async def test_peek(self, tmp_path: Path):
        from .bot_tools import PeekQueuedUserMessagesInput, PeekQueuedUserMessagesTool

        s = BotState(active_channel_id=1, state_root=tmp_path / "state")
        s.enqueue_user_message(1, MockMessage(content="hi"), "hi")  # type: ignore[arg-type]
        tool = PeekQueuedUserMessagesTool(state=s)
        result = await tool(PeekQueuedUserMessagesInput(limit=5))
        payload = json.loads(result.content.text)
        assert payload["pending_count"] == 1
        assert len(payload["messages"]) == 1
        # Queue should still have the message
        assert len(s.get_channel_queue(1)) == 1


# ---------------------------------------------------------------------------
# Tier 4: Agent loop (mock API)
# ---------------------------------------------------------------------------


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_single_turn_no_tools(self, tmp_path: Path):
        from .bot_agent import agent_loop

        channel = MockChannel()
        s = BotState(
            active_channel=channel,  # type: ignore[arg-type]
            active_channel_id=42,
            state_root=tmp_path / "state",
        )
        dag = DAG().system("test").user("hello")
        mock_api = MockAPI()
        dag, stats = await agent_loop(mock_api, s, channel, 42, dag)  # type: ignore[arg-type]
        assert stats.stop_reason == "end_turn"
        assert stats.tool_calls == 0
        assert stats.outbound_messages == 0
