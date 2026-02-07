"""Tests for Gemini API client."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nano_agent import (
    DAG,
    APIError,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from nano_agent.providers.gemini import GeminiAPI, _map_finish_reason
from nano_agent.tools import Tool


class TestHelperFunctions:
    def test_map_finish_reason_stop(self) -> None:
        assert _map_finish_reason("STOP") == "end_turn"

    def test_map_finish_reason_max_tokens(self) -> None:
        assert _map_finish_reason("MAX_TOKENS") == "max_tokens"

    def test_map_finish_reason_safety(self) -> None:
        assert _map_finish_reason("SAFETY") == "safety"

    def test_map_finish_reason_malformed_function(self) -> None:
        assert _map_finish_reason("MALFORMED_FUNCTION_CALL") == "tool_use"

    def test_map_finish_reason_unknown(self) -> None:
        assert _map_finish_reason("UNKNOWN_REASON") == "unknown_reason"

    def test_map_finish_reason_empty(self) -> None:
        assert _map_finish_reason("") is None


class TestGeminiAPIInit:
    def test_init_with_api_key(self) -> None:
        api = GeminiAPI(api_key="test-key")
        assert api.api_key == "test-key"
        assert api.model == "gemini-3-flash-preview"
        assert api.max_tokens == 8192
        assert api.temperature == 1.0
        assert api.thinking_level == "low"
        assert api.base_url == "https://generativelanguage.googleapis.com/v1beta"

    def test_init_with_env_var(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
            api = GeminiAPI()
            assert api.api_key == "env-key"

    def test_init_without_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            import os

            old_key = os.environ.pop("GEMINI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="Gemini API key required"):
                    GeminiAPI()
            finally:
                if old_key:
                    os.environ["GEMINI_API_KEY"] = old_key

    def test_init_custom_values(self) -> None:
        api = GeminiAPI(
            api_key="test-key",
            model="gemini-3-pro-preview",
            max_tokens=16384,
            temperature=0.7,
            thinking_level="high",
            base_url="https://custom.api.com/v1beta",
        )
        assert api.model == "gemini-3-pro-preview"
        assert api.max_tokens == 16384
        assert api.temperature == 0.7
        assert api.thinking_level == "high"
        assert api.base_url == "https://custom.api.com/v1beta"

    def test_init_thinking_disabled(self) -> None:
        api = GeminiAPI(api_key="test-key", thinking_level=None)
        assert api.thinking_level is None

    def test_repr(self) -> None:
        api = GeminiAPI(api_key="AIzaSy-test-key-1234567890")
        repr_str = repr(api)
        assert "GeminiAPI" in repr_str
        assert "gemini-3-flash-preview" in repr_str
        assert "AIzaSy-test-key..." in repr_str  # Token truncated


class TestMessageConversion:
    def test_convert_user_string_message(self) -> None:
        api = GeminiAPI(api_key="test-key")
        msg = Message(role=Role.USER, content="Hello")
        tool_name_map: dict[str, str] = {}
        result = api._convert_message_to_gemini(msg, tool_name_map)

        assert result == {"role": "user", "parts": [{"text": "Hello"}]}

    def test_convert_assistant_string_message(self) -> None:
        api = GeminiAPI(api_key="test-key")
        msg = Message(role=Role.ASSISTANT, content="Hi there")
        tool_name_map: dict[str, str] = {}
        result = api._convert_message_to_gemini(msg, tool_name_map)

        # Role should be "model" for assistant
        assert result == {"role": "model", "parts": [{"text": "Hi there"}]}

    def test_convert_user_text_content(self) -> None:
        api = GeminiAPI(api_key="test-key")
        msg = Message(role=Role.USER, content=[TextContent(text="Hello")])
        tool_name_map: dict[str, str] = {}
        result = api._convert_message_to_gemini(msg, tool_name_map)

        assert result == {"role": "user", "parts": [{"text": "Hello"}]}

    def test_convert_tool_use_content(self) -> None:
        api = GeminiAPI(api_key="test-key")
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolUseContent(
                    id="call_123", name="get_weather", input={"location": "NYC"}
                )
            ],
        )
        tool_name_map: dict[str, str] = {}
        result = api._convert_message_to_gemini(msg, tool_name_map)

        assert result["role"] == "model"
        assert len(result["parts"]) == 1
        assert result["parts"][0] == {
            "function_call": {"name": "get_weather", "args": {"location": "NYC"}}
        }
        # Tool name should be tracked
        assert tool_name_map["call_123"] == "get_weather"

    def test_convert_tool_result_content(self) -> None:
        api = GeminiAPI(api_key="test-key")
        msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(
                    tool_use_id="call_123",
                    content=[TextContent(text="72F and sunny")],
                )
            ],
        )
        # Pre-populate tool name map
        tool_name_map = {"call_123": "get_weather"}
        result = api._convert_message_to_gemini(msg, tool_name_map)

        assert result["role"] == "user"
        assert len(result["parts"]) == 1
        assert result["parts"][0] == {
            "function_response": {
                "name": "get_weather",
                "response": {"output": "72F and sunny"},
            }
        }

    def test_convert_thinking_content_skipped(self) -> None:
        api = GeminiAPI(api_key="test-key")
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="Let me think..."),
                TextContent(text="The answer is 4"),
            ],
        )
        tool_name_map: dict[str, str] = {}
        result = api._convert_message_to_gemini(msg, tool_name_map)

        # Thinking should be skipped, only text remains
        assert result == {"role": "model", "parts": [{"text": "The answer is 4"}]}


class TestToolConversion:
    def test_convert_tool_to_gemini(self) -> None:
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

        api = GeminiAPI(api_key="test-key")
        tool = GetWeatherTool()
        result = api._convert_tool_to_gemini(tool)

        assert result == {
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
            },
        }


class TestResponseParsing:
    def test_parse_text_response(self) -> None:
        api = GeminiAPI(api_key="test-key")
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        response = api._parse_response(data)

        assert response.model == "gemini-3-flash-preview"
        assert response.role == Role.ASSISTANT
        assert len(response.content) == 1
        assert isinstance(response.content[0], TextContent)
        assert response.content[0].text == "Hello!"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_parse_function_call_response(self) -> None:
        api = GeminiAPI(api_key="test-key")
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "NYC"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 20},
        }
        response = api._parse_response(data)

        assert len(response.content) == 1
        assert isinstance(response.content[0], ToolUseContent)
        assert response.content[0].name == "get_weather"
        assert response.content[0].input == {"location": "NYC"}
        # ID should be generated
        assert response.content[0].id.startswith("call_")

    def test_parse_thinking_response(self) -> None:
        api = GeminiAPI(api_key="test-key")
        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": "Let me calculate..."},
                            {"text": "The answer is 4"},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 15,
            },
        }
        response = api._parse_response(data)

        assert len(response.content) == 2
        assert isinstance(response.content[0], ThinkingContent)
        assert response.content[0].thinking == "Let me calculate..."
        assert isinstance(response.content[1], TextContent)
        assert response.content[1].text == "The answer is 4"
        # Thinking tokens stored in reasoning_tokens
        assert response.usage.reasoning_tokens == 15
        assert response.usage.cache_creation_input_tokens == 0

    def test_parse_error_response(self) -> None:
        """Test that _parse_response raises when no candidates (error responses have no candidates).

        Note: HTTP error checking is handled by _check_response, not _parse_response.
        When _parse_response receives a response without candidates, it raises "blocked".
        """
        api = GeminiAPI(api_key="test-key")
        data: dict[str, Any] = {"error": {"message": "Invalid API key"}}

        with pytest.raises(RuntimeError, match="Gemini response blocked"):
            api._parse_response(data)

    def test_parse_blocked_response(self) -> None:
        api = GeminiAPI(api_key="test-key")
        data: dict[str, Any] = {
            "candidates": [],
            "promptFeedback": {"blockReason": "SAFETY"},
        }

        with pytest.raises(RuntimeError, match="Gemini response blocked: SAFETY"):
            api._parse_response(data)


class TestSend:
    async def test_send_simple_message(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Four"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        api = GeminiAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="What is 2+2?")]
        response = await api.send(messages)

        api._client.post.assert_called_once()
        call_args = api._client.post.call_args

        # Check URL includes API key
        assert "test-api-key" in call_args[0][0]
        assert "gemini-3-flash-preview:generateContent" in call_args[0][0]
        assert call_args[1]["json"]["contents"] == [
            {"role": "user", "parts": [{"text": "What is 2+2?"}]}
        ]

        assert isinstance(response, Response)
        assert response.get_text() == "Four"

    async def test_send_with_system_prompt(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        api = GeminiAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="Hi")]
        await api.send(messages, system_prompt="You are helpful.")

        call_args = api._client.post.call_args
        # System prompt uses systemInstruction field
        assert call_args[1]["json"]["systemInstruction"] == {
            "parts": [{"text": "You are helpful."}]
        }

    async def test_send_with_tools(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "NYC"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        api = GeminiAPI(api_key="test-api-key")
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
        # Tools use functionDeclarations (additionalProperties is stripped for Gemini)
        assert call_args[1]["json"]["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Location",
                                }
                            },
                            "required": [],
                        },
                    }
                ]
            }
        ]

        assert response.has_tool_use()
        tool_calls = response.get_tool_use()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

    async def test_send_with_thinking_enabled(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Result"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        api = GeminiAPI(api_key="test-api-key", thinking_level="medium")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="Think about this")]
        await api.send(messages)

        call_args = api._client.post.call_args
        assert call_args[1]["json"]["generationConfig"]["thinkingConfig"] == {
            "thinkingLevel": "MEDIUM"
        }

    async def test_send_with_thinking_disabled(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Result"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        api = GeminiAPI(api_key="test-api-key", thinking_level=None)
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="No thinking")]
        await api.send(messages)

        call_args = api._client.post.call_args
        assert "thinkingConfig" not in call_args[1]["json"]["generationConfig"]

    async def test_send_with_dag(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        api = GeminiAPI(api_key="test-api-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        dag = DAG().system("Be helpful.").user("Hi!")
        response = await api.send(dag)

        call_args = api._client.post.call_args
        # System prompt from DAG
        assert call_args[1]["json"]["systemInstruction"] == {
            "parts": [{"text": "Be helpful."}]
        }
        # User message
        assert call_args[1]["json"]["contents"] == [
            {"role": "user", "parts": [{"text": "Hi!"}]}
        ]
        assert response.get_text() == "Hello!"

    async def test_send_error_handling(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

        api = GeminiAPI(api_key="invalid-key")
        api._client = AsyncMock()
        api._client.post.return_value = mock_response

        messages = [Message(role=Role.USER, content="Hi")]

        with pytest.raises(APIError, match="Invalid API key"):
            await api.send(messages)


class TestSendMethod:
    def test_send_method_is_async(self) -> None:
        """Verify send method is async."""
        api = GeminiAPI(api_key="test-api-key")
        import asyncio

        assert asyncio.iscoroutinefunction(api.send)
