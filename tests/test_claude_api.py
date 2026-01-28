from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from nano_agent import (
    ClaudeAPI,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    Usage,
)


class TestRole:
    def test_user_role_value(self) -> None:
        assert Role.USER.value == "user"

    def test_assistant_role_value(self) -> None:
        assert Role.ASSISTANT.value == "assistant"


class TestTextContent:
    def test_default_values(self) -> None:
        content = TextContent(text="")
        assert content.type == "text"
        assert content.text == ""

    def test_with_text(self) -> None:
        content = TextContent(text="Hello")
        assert content.text == "Hello"

    def test_to_dict(self) -> None:
        content = TextContent(text="Hello")
        assert content.to_dict() == {"type": "text", "text": "Hello"}


class TestThinkingContent:
    def test_default_values(self) -> None:
        content = ThinkingContent(thinking="")
        assert content.type == "thinking"
        assert content.thinking == ""
        assert content.signature == ""

    def test_with_values(self) -> None:
        content = ThinkingContent(thinking="Let me think...", signature="sig123")
        assert content.thinking == "Let me think..."
        assert content.signature == "sig123"

    def test_to_dict(self) -> None:
        content = ThinkingContent(thinking="Thinking", signature="sig")
        assert content.to_dict() == {
            "type": "thinking",
            "thinking": "Thinking",
            "signature": "sig",
        }


class TestMessage:
    def test_string_content(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_string_content_to_dict(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.to_dict() == {"role": "user", "content": "Hello"}

    def test_list_content_to_dict(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextContent(text="Hi"), TextContent(text="there")],
        )
        assert msg.to_dict() == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "text", "text": "there"},
            ],
        }


class TestUsage:
    def test_required_fields(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=20)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    def test_optional_fields_default(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=20)
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_all_fields(self) -> None:
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=3,
        )
        assert usage.cache_creation_input_tokens == 5
        assert usage.cache_read_input_tokens == 3


class TestResponse:
    def test_from_dict_text_content(self) -> None:
        data: dict[str, Any] = {
            "id": "msg_123",
            "model": "claude-haiku-4-5-20251001",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = Response.from_dict(data)

        assert response.id == "msg_123"
        assert response.model == "claude-haiku-4-5-20251001"
        assert response.role == Role.ASSISTANT
        assert len(response.content) == 1
        assert isinstance(response.content[0], TextContent)
        assert response.content[0].text == "Hello"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_from_dict_thinking_content(self) -> None:
        data: dict[str, Any] = {
            "id": "msg_456",
            "model": "claude-haiku-4-5-20251001",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think...", "signature": "sig"},
                {"type": "text", "text": "Answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = Response.from_dict(data)

        assert len(response.content) == 2
        assert isinstance(response.content[0], ThinkingContent)
        assert response.content[0].thinking == "Let me think..."
        assert isinstance(response.content[1], TextContent)
        assert response.content[1].text == "Answer"

    def test_get_text(self) -> None:
        data: dict[str, Any] = {
            "id": "msg_789",
            "model": "claude-haiku-4-5-20251001",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Thinking..."},
                {"type": "text", "text": "The answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = Response.from_dict(data)
        assert response.get_text() == "The answer"

    def test_get_text_empty(self) -> None:
        data: dict[str, Any] = {
            "id": "msg_000",
            "model": "claude-haiku-4-5-20251001",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        response = Response.from_dict(data)
        assert response.get_text() == ""


class TestClaudeAPI:
    def test_init_default_values(self) -> None:
        api = ClaudeAPI(api_key="test-key")
        assert api.api_key == "test-key"
        assert api.model == "claude-sonnet-4-20250514"
        assert api.max_tokens == 16000
        assert api.temperature == 1.0
        assert api.thinking_budget == 10000
        assert api.base_url == "https://api.anthropic.com/v1/messages"

    def test_init_custom_values(self) -> None:
        api = ClaudeAPI(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            max_tokens=4096,
            temperature=0.5,
            thinking_budget=5000,
        )
        assert api.model == "claude-opus-4-5-20251101"
        assert api.max_tokens == 4096
        assert api.temperature == 0.5
        assert api.thinking_budget == 5000

    def test_init_requires_api_key(self) -> None:
        # Temporarily clear env var if set
        import os

        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                ClaudeAPI()
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_repr(self) -> None:
        api = ClaudeAPI(api_key="sk-ant-1234567890abcdef")
        repr_str = repr(api)
        assert "claude-sonnet-4-20250514" in repr_str
        assert "16000" in repr_str
        # First 15 chars of "sk-ant-1234567890abcdef" = "sk-ant-12345678"
        assert "sk-ant-12345678..." in repr_str

    async def test_send_makes_correct_request(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "msg_test",
            "model": "claude-sonnet-4-20250514",
            "role": "assistant",
            "content": [{"type": "text", "text": "Four"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        api = ClaudeAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="What is 2+2?")]
        response = await api.send(messages)

        api._client.post.assert_called_once()
        call_args = api._client.post.call_args

        # Verify URL and headers
        assert call_args[0][0] == "https://api.anthropic.com/v1/messages"
        assert call_args[1]["headers"]["x-api-key"] == "test-api-key"
        assert call_args[1]["headers"]["anthropic-version"] == "2023-06-01"
        assert call_args[1]["headers"]["content-type"] == "application/json"

        # Verify body
        body = call_args[1]["json"]
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["max_tokens"] == 16000
        assert body["temperature"] == 1.0
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert body["messages"] == [{"role": "user", "content": "What is 2+2?"}]

        assert isinstance(response, Response)
        assert response.get_text() == "Four"

    async def test_send_with_system_string(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "msg_test",
            "model": "claude-sonnet-4-20250514",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        api = ClaudeAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="Hi")]
        await api.send(messages, system="Be friendly.")

        call_args = api._client.post.call_args
        assert call_args[1]["json"]["system"] == "Be friendly."

    async def test_send_with_system_list(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "msg_test",
            "model": "claude-sonnet-4-20250514",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        api = ClaudeAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        system_list = [
            {"type": "text", "text": "Be helpful."},
            {"type": "text", "text": "Be concise."},
        ]
        messages = [Message(role=Role.USER, content="Hi")]
        await api.send(messages, system=system_list)

        call_args = api._client.post.call_args
        assert call_args[1]["json"]["system"] == system_list
