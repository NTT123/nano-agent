"""Tests for OpenAI API client."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nano_agent import (
    DAG,
    Message,
    Response,
    Role,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from nano_agent.openai_api import (
    OpenAIAPI,
    _map_status_to_stop_reason,
    _parse_arguments,
    _serialize_arguments,
)
from nano_agent.tools import Tool


class TestHelperFunctions:
    def test_serialize_arguments_none(self) -> None:
        assert _serialize_arguments(None) == "{}"

    def test_serialize_arguments_dict(self) -> None:
        result = _serialize_arguments({"key": "value", "num": 42})
        assert '"key"' in result
        assert '"value"' in result
        assert "42" in result

    def test_parse_arguments_valid_json(self) -> None:
        assert _parse_arguments('{"key": "value"}') == {"key": "value"}

    def test_parse_arguments_invalid_json(self) -> None:
        assert _parse_arguments("not json") == {}

    def test_parse_arguments_non_dict(self) -> None:
        assert _parse_arguments('"string"') == {}

    def test_map_status_completed(self) -> None:
        assert _map_status_to_stop_reason("completed") == "end_turn"

    def test_map_status_failed(self) -> None:
        assert _map_status_to_stop_reason("failed") == "error"

    def test_map_status_incomplete(self) -> None:
        assert _map_status_to_stop_reason("incomplete") == "max_tokens"

    def test_map_status_in_progress(self) -> None:
        assert _map_status_to_stop_reason("in_progress") is None

    def test_map_status_unknown(self) -> None:
        assert _map_status_to_stop_reason("unknown_status") == "unknown_status"


class TestOpenAIAPIInit:
    def test_init_with_api_key(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        assert api.api_key == "test-key"
        assert api.model == "gpt-5.2-codex"
        assert api.max_tokens == 4096
        assert api.temperature == 1.0
        assert api.base_url == "https://api.openai.com/v1/responses"

    def test_init_with_env_var(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            api = OpenAIAPI()
            assert api.api_key == "env-key"

    def test_init_without_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Also need to clear the key if it exists
            import os

            old_key = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="OpenAI API key required"):
                    OpenAIAPI()
            finally:
                if old_key:
                    os.environ["OPENAI_API_KEY"] = old_key

    def test_init_custom_values(self) -> None:
        api = OpenAIAPI(
            api_key="test-key",
            model="gpt-4-turbo",
            max_tokens=8192,
            temperature=0.5,
            base_url="https://custom.api.com/v1/responses",
        )
        assert api.model == "gpt-4-turbo"
        assert api.max_tokens == 8192
        assert api.temperature == 0.5
        assert api.base_url == "https://custom.api.com/v1/responses"

    def test_repr(self) -> None:
        api = OpenAIAPI(api_key="sk-test-key-12345678901234567890")
        repr_str = repr(api)
        assert "OpenAIAPI" in repr_str
        assert "gpt-5.2-codex" in repr_str
        assert "sk-test-key-123..." in repr_str  # Token truncated


class TestMessageConversion:
    def test_convert_user_string_message(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        msg = Message(role=Role.USER, content="Hello")
        items = api._convert_message_to_openai(msg)

        assert items == [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ]

    def test_convert_assistant_string_message(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        msg = Message(role=Role.ASSISTANT, content="Hi there")
        items = api._convert_message_to_openai(msg)

        assert items == [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}],
            }
        ]

    def test_convert_user_text_content(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        msg = Message(role=Role.USER, content=[TextContent(text="Hello")])
        items = api._convert_message_to_openai(msg)

        assert items == [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ]

    def test_convert_assistant_text_content(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        msg = Message(role=Role.ASSISTANT, content=[TextContent(text="Response")])
        items = api._convert_message_to_openai(msg)

        assert items == [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response"}],
            }
        ]

    def test_convert_tool_use_content(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolUseContent(
                    id="call_123", name="get_weather", input={"location": "NYC"}
                )
            ],
        )
        items = api._convert_message_to_openai(msg)

        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["call_id"] == "call_123"
        assert items[0]["name"] == "get_weather"
        assert '"location"' in items[0]["arguments"]

    def test_convert_tool_result_content(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(
                    tool_use_id="call_123",
                    content=[TextContent(text="72F and sunny")],
                )
            ],
        )
        items = api._convert_message_to_openai(msg)

        assert len(items) == 1
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_123"
        assert items[0]["output"] == "72F and sunny"


class TestToolConversion:
    def test_convert_tool_to_openai(self) -> None:
        from dataclasses import dataclass
        from typing import Annotated

        from nano_agent import TextContent
        from nano_agent.tools import Desc

        @dataclass
        class WeatherInput:
            location: Annotated[str, Desc("The location to get weather for")]

        @dataclass
        class GetWeatherTool(Tool):
            name: str = "get_weather"
            description: str = "Get the current weather"

            async def __call__(self, input: WeatherInput) -> TextContent:
                return TextContent(text=f"Weather for {input.location}")

        api = OpenAIAPI(api_key="test-key")
        tool = GetWeatherTool()
        result = api._convert_tool_to_openai(tool)

        assert result == {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        }


class TestResponseParsing:
    def test_parse_text_response(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        data: dict[str, Any] = {
            "id": "resp_123",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = api._parse_response(data)

        assert response.id == "resp_123"
        assert response.model == "gpt-5.2-codex"
        assert response.role == Role.ASSISTANT
        assert len(response.content) == 1
        assert isinstance(response.content[0], TextContent)
        assert response.content[0].text == "Hello!"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_parse_function_call_response(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        data: dict[str, Any] = {
            "id": "resp_456",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                }
            ],
            "usage": {"input_tokens": 15, "output_tokens": 20},
        }
        response = api._parse_response(data)

        assert len(response.content) == 1
        assert isinstance(response.content[0], ToolUseContent)
        assert response.content[0].id == "call_abc"
        assert response.content[0].name == "get_weather"
        assert response.content[0].input == {"location": "NYC"}

    def test_parse_refusal_response(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        data: dict[str, Any] = {
            "id": "resp_789",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "refusal", "refusal": "I cannot help with that."}
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = api._parse_response(data)

        assert len(response.content) == 1
        assert isinstance(response.content[0], TextContent)
        assert response.content[0].text == "I cannot help with that."

    def test_parse_usage_with_cached_and_reasoning(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        data: dict[str, Any] = {
            "id": "resp_usage",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 2000,
                "input_tokens_details": {"cached_tokens": 300},
                "output_tokens_details": {"reasoning_tokens": 200},
            },
        }
        response = api._parse_response(data)

        assert response.usage.input_tokens == 1000
        assert response.usage.output_tokens == 500
        assert response.usage.cached_tokens == 300
        assert response.usage.reasoning_tokens == 200
        assert response.usage.total_tokens == 2000


class TestSend:
    async def test_send_simple_message(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_test",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Four"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        api = OpenAIAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="What is 2+2?")]
        response = await api.send(messages)

        api._client.post.assert_called_once()
        call_args = api._client.post.call_args

        assert call_args[0][0] == "https://api.openai.com/v1/responses"
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key"
        assert call_args[1]["json"]["model"] == "gpt-5.2-codex"
        assert call_args[1]["json"]["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 2+2?"}],
            }
        ]

        assert isinstance(response, Response)
        assert response.get_text() == "Four"

    async def test_send_with_system_prompt(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_test",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        api = OpenAIAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="Hi")]
        await api.send(messages, system_prompt="You are helpful.")

        call_args = api._client.post.call_args
        # System prompt is now in input array, not instructions
        input_items = call_args[1]["json"]["input"]
        assert input_items[0] == {
            "role": "system",
            "content": [{"type": "input_text", "text": "You are helpful."}],
        }

    async def test_send_with_tools(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_test",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        api = OpenAIAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        from dataclasses import dataclass
        from typing import Annotated

        from nano_agent import TextContent
        from nano_agent.tools import Desc

        @dataclass
        class WeatherInput:
            location: Annotated[str, Desc("Location")] = ""

        @dataclass
        class GetWeatherTool(Tool):
            name: str = "get_weather"
            description: str = "Get weather"

            async def __call__(self, input: WeatherInput) -> TextContent:
                return TextContent(text="sunny")

        messages = [Message(role=Role.USER, content="What's the weather in NYC?")]
        tools = [GetWeatherTool()]
        response = await api.send(messages, tools=tools)

        call_args = api._client.post.call_args
        assert len(call_args[1]["json"]["tools"]) == 1
        assert call_args[1]["json"]["tools"][0]["name"] == "get_weather"

        assert response.has_tool_use()
        tool_calls = response.get_tool_use()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

    async def test_send_with_dag(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_test",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        api = OpenAIAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        dag = DAG().system("Be helpful.").user("Hi!")
        response = await api.send(dag)

        call_args = api._client.post.call_args
        input_items = call_args[1]["json"]["input"]
        # First item is system prompt
        assert input_items[0] == {
            "role": "system",
            "content": [{"type": "input_text", "text": "Be helpful."}],
        }
        # Second item is user message
        assert input_items[1] == {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hi!"}],
        }
        assert response.get_text() == "Hello!"

    async def test_send_error_handling(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key", "type": "invalid_request_error"}
        }

        api = OpenAIAPI(api_key="invalid-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="Hi")]

        with pytest.raises(RuntimeError, match="OpenAI API error: Invalid API key"):
            await api.send(messages)


class TestSendMethod:
    def test_send_method_is_async(self) -> None:
        """Verify send method is async."""
        api = OpenAIAPI(api_key="test-api-key")
        import asyncio

        assert asyncio.iscoroutinefunction(api.send)
