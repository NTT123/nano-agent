"""Tests for Codex API client."""

from dataclasses import dataclass
from typing import Annotated, Any

import pytest

from nano_agent import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from nano_agent.providers.codex import (
    CodexAPI,
    _map_status_to_stop_reason,
    _parse_arguments,
    _serialize_arguments,
)
from nano_agent.tools import Desc, Tool


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


class TestCodexAPIInit:
    def test_init_with_auth_token(self) -> None:
        api = CodexAPI(auth_token="test-token")
        assert api.auth_token == "test-token"
        assert api.model == "gpt-5.2-codex"
        assert api.base_url == "https://chatgpt.com/backend-api/codex/responses"

    def test_init_custom_values(self) -> None:
        api = CodexAPI(
            auth_token="test-token",
            model="gpt-4.1-mini",
            base_url="https://example.com/codex",
            parallel_tool_calls=False,
        )
        assert api.model == "gpt-4.1-mini"
        assert api.base_url == "https://example.com/codex"
        assert api.parallel_tool_calls is False

    def test_repr(self) -> None:
        api = CodexAPI(auth_token="token-12345678901234567890")
        repr_str = repr(api)
        assert "CodexAPI" in repr_str
        assert "gpt-5.2-codex" in repr_str
        assert "token-123456789..." in repr_str


class TestMessageConversion:
    def test_convert_user_string_message(self) -> None:
        api = CodexAPI(auth_token="test-token")
        msg = Message(role=Role.USER, content="Hello")
        items = api._convert_message_to_codex(msg)

        assert items == [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ]

    def test_convert_assistant_string_message(self) -> None:
        api = CodexAPI(auth_token="test-token")
        msg = Message(role=Role.ASSISTANT, content="Hi there")
        items = api._convert_message_to_codex(msg)

        assert items == [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}],
            }
        ]

    def test_convert_text_and_tool_use(self) -> None:
        api = CodexAPI(auth_token="test-token")
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="Calling tool"),
                ToolUseContent(
                    id="call_123", name="get_weather", input={"city": "NYC"}
                ),
            ],
        )
        items = api._convert_message_to_codex(msg)

        assert items[0] == {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Calling tool"}],
        }
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_123"
        assert items[1]["name"] == "get_weather"
        assert '"city"' in items[1]["arguments"]

    def test_convert_tool_result_content(self) -> None:
        api = CodexAPI(auth_token="test-token")
        msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(
                    tool_use_id="call_123",
                    content=[TextContent(text="part1"), TextContent(text="part2")],
                )
            ],
        )
        items = api._convert_message_to_codex(msg)

        assert items == [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "part1part2",
            }
        ]


class TestToolConversion:
    def test_convert_tool_to_codex(self) -> None:
        @dataclass
        class WeatherInput:
            location: Annotated[str, Desc("The location to get weather for")]

        @dataclass
        class GetWeatherTool(Tool):
            name: str = "get_weather"
            description: str = "Get the current weather"

            async def __call__(self, input: WeatherInput) -> TextContent:
                return TextContent(text=f"Weather for {input.location}")

        api = CodexAPI(auth_token="test-token")
        tool = GetWeatherTool()
        result = api._convert_tool_to_codex(tool)

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
    def test_parse_mixed_response(self) -> None:
        api = CodexAPI(auth_token="test-token")
        data: dict[str, Any] = {
            "id": "resp_123",
            "model": "gpt-5.2-codex",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "r1",
                    "encrypted_content": "enc",
                    "summary": [{"type": "summary_text", "text": "Summary text"}],
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello!"},
                        {"type": "refusal", "refusal": "No thanks."},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                    "id": "fc_1",
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = api._parse_response(data)

        assert response.id == "resp_123"
        assert response.model == "gpt-5.2-codex"
        assert response.role == Role.ASSISTANT
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

        assert isinstance(response.content[0], ThinkingContent)
        assert response.content[0].thinking == "Summary text"
        assert isinstance(response.content[1], TextContent)
        assert response.content[1].text == "Hello!"
        assert isinstance(response.content[2], TextContent)
        assert response.content[2].text == "No thanks."
        assert isinstance(response.content[3], ToolUseContent)
        assert response.content[3].id == "call_abc"
        assert response.content[3].name == "get_weather"
        assert response.content[3].input == {"location": "NYC"}


class TestCodexAPIInitErrors:
    def test_init_without_token_raises(self) -> None:
        from unittest.mock import patch

        with patch(
            "nano_agent.providers.codex.get_codex_access_token", return_value=None
        ):
            with pytest.raises(ValueError, match="Codex OAuth token required"):
                CodexAPI(auth_token=None)
