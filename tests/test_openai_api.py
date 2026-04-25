"""Tests for OpenAI API client."""

import json
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest

from nano_agent import (
    DAG,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from nano_agent.providers.openai import (
    OpenAIAPI,
    _map_status_to_stop_reason,
    _raise_openai_http_error,
)
from nano_agent.tools import Tool


class TestMapStatusToStopReason:
    def test_completed(self) -> None:
        assert _map_status_to_stop_reason("completed") == "end_turn"

    def test_failed(self) -> None:
        assert _map_status_to_stop_reason("failed") == "error"

    def test_incomplete(self) -> None:
        assert _map_status_to_stop_reason("incomplete") == "max_tokens"

    def test_in_progress(self) -> None:
        assert _map_status_to_stop_reason("in_progress") is None

    def test_unknown(self) -> None:
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

    def test_convert_drops_reasoning_without_encrypted_content(self) -> None:
        """Claude-style ``thinking`` blocks lack encrypted_content and would
        produce an invalid Responses-API item; they must be skipped even when
        ``include_reasoning=True``."""
        api = OpenAIAPI(api_key="test-key")
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="claude-style", signature="sig"),
                TextContent(text="answer"),
            ],
        )
        items = api._convert_message_to_openai(msg, include_reasoning=True)

        assert all(item.get("type") != "thinking" for item in items)
        assert all(item.get("type") != "reasoning" for item in items)

    def test_convert_replays_reasoning_with_encrypted_content(self) -> None:
        api = OpenAIAPI(api_key="test-key")
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(
                    thinking="t",
                    id="rs_abc",
                    encrypted_content="ENCRYPTED_BLOB",
                    summary=({"type": "summary_text", "text": "t"},),
                ),
                TextContent(text="answer"),
            ],
        )
        items = api._convert_message_to_openai(msg, include_reasoning=True)

        assert items[0] == {
            "type": "reasoning",
            "id": "rs_abc",
            "encrypted_content": "ENCRYPTED_BLOB",
            "summary": [{"type": "summary_text", "text": "t"}],
        }


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

    def test_convert_tool_to_openai_respects_non_strict_tool(self) -> None:
        from dataclasses import dataclass
        from typing import Annotated

        from nano_agent import TextContent
        from nano_agent.tools import Desc

        @dataclass
        class SearchInput:
            query: Annotated[str, Desc("Search query")]
            limit: Annotated[int, Desc("Maximum result count")] = 10

        @dataclass
        class SearchTool(Tool):
            name: str = "search"
            description: str = "Search"
            strict: ClassVar[bool] = False

            async def __call__(self, input: SearchInput) -> TextContent:
                return TextContent(text=input.query)

        api = OpenAIAPI(api_key="test-key")
        result = api._convert_tool_to_openai(SearchTool())

        assert result["strict"] is False
        assert result["parameters"]["required"] == ["query"]

    def test_strict_tool_promotes_default_fields_to_required(self) -> None:
        """With strict=True (the default), every property must appear in
        ``required`` or the OpenAI API rejects the tool."""
        from dataclasses import dataclass
        from typing import Annotated

        from nano_agent import TextContent
        from nano_agent.tools import Desc

        @dataclass
        class SearchInput:
            query: Annotated[str, Desc("Search query")]
            limit: Annotated[int, Desc("Maximum result count")] = 10

        @dataclass
        class SearchTool(Tool):
            name: str = "search"
            description: str = "Search"

            async def __call__(self, input: SearchInput) -> TextContent:
                return TextContent(text=input.query)

        api = OpenAIAPI(api_key="test-key")
        result = api._convert_tool_to_openai(SearchTool())

        assert result["strict"] is True
        assert sorted(result["parameters"]["required"]) == ["limit", "query"]
        assert result["parameters"]["additionalProperties"] is False

    def test_strict_rewrite_does_not_mutate_cached_input_schema(self) -> None:
        """``Tool._inferred_schema`` is a ``ClassVar`` shared across instances;
        the strict-mode rewrite must not mutate it or future conversions —
        even from a different provider — would see corrupted state."""
        from dataclasses import dataclass
        from typing import Annotated

        from nano_agent import TextContent
        from nano_agent.tools import Desc

        @dataclass
        class FooInput:
            x: Annotated[str, Desc("x")]
            y: Annotated[int, Desc("y")] = 1

        @dataclass
        class FooTool(Tool):
            name: str = "foo"
            description: str = "foo"

            async def __call__(self, input: FooInput) -> TextContent:
                return TextContent(text="")

        api = OpenAIAPI(api_key="test-key")
        tool = FooTool()
        before = list(tool.input_schema["required"])  # type: ignore[arg-type]

        api._convert_tool_to_openai(tool)
        api._convert_tool_to_openai(tool)

        assert list(tool.input_schema["required"]) == before  # type: ignore[arg-type]


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
        api = OpenAIAPI(api_key="test-api-key")
        api._stream_response = AsyncMock(  # type: ignore[method-assign]
            return_value={
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
        )

        messages = [Message(role=Role.USER, content="What is 2+2?")]
        response = await api.send(messages)

        api._stream_response.assert_awaited_once()
        request_body = api._stream_response.await_args.args[0]
        assert request_body["model"] == "gpt-5.2-codex"
        assert request_body["stream"] is True
        assert request_body["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 2+2?"}],
            }
        ]

        assert isinstance(response, Response)
        assert response.get_text() == "Four"

    async def test_send_includes_stable_prompt_cache_key(self) -> None:
        """Sending repeatedly from the same client must reuse the same
        ``prompt_cache_key`` so OpenAI prefix caching gets a stable bucket."""
        api = OpenAIAPI(api_key="test-api-key")
        api._stream_response = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "id": "r",
                "model": "gpt-5.2-codex",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

        await api.send([Message(role=Role.USER, content="hi")])
        await api.send([Message(role=Role.USER, content="hi again")])

        first_call, second_call = api._stream_response.await_args_list
        assert first_call.args[0]["prompt_cache_key"] == api._session_id
        assert second_call.args[0]["prompt_cache_key"] == api._session_id

    async def test_send_passes_reasoning_effort_and_summary(self) -> None:
        api = OpenAIAPI(
            api_key="test-api-key",
            reasoning=True,
            reasoning_effort="medium",
            reasoning_summary="detailed",
        )
        api._stream_response = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "id": "r",
                "model": "gpt-5.2-codex",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

        await api.send([Message(role=Role.USER, content="hi")])

        request_body = api._stream_response.await_args.args[0]
        assert request_body["reasoning"] == {
            "effort": "medium",
            "summary": "detailed",
        }
        assert request_body["include"] == ["reasoning.encrypted_content"]

    async def test_send_omits_summary_by_default(self) -> None:
        """``summary`` is only sent when explicitly opted in (``"auto"``
        requires org verification on the public API)."""
        api = OpenAIAPI(api_key="test-api-key", reasoning=True)
        api._stream_response = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "id": "r",
                "model": "gpt-5.2-codex",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

        await api.send([Message(role=Role.USER, content="hi")])

        request_body = api._stream_response.await_args.args[0]
        assert request_body["reasoning"] == {"effort": "high"}

    async def test_send_with_system_prompt(self) -> None:
        api = OpenAIAPI(api_key="test-api-key")
        api._stream_response = AsyncMock(  # type: ignore[method-assign]
            return_value={
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
        )

        messages = [Message(role=Role.USER, content="Hi")]
        await api.send(messages, system_prompt="You are helpful.")

        request_body = api._stream_response.await_args.args[0]
        # System prompt is in the input array as a developer message for
        # OpenAI prompt-cache friendliness.
        input_items = request_body["input"]
        assert input_items[0] == {
            "role": "developer",
            "content": [{"type": "input_text", "text": "You are helpful."}],
        }

    async def test_send_with_tools(self) -> None:
        api = OpenAIAPI(api_key="test-api-key")
        api._stream_response = AsyncMock(  # type: ignore[method-assign]
            return_value={
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
        )

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

        request_body = api._stream_response.await_args.args[0]
        assert len(request_body["tools"]) == 1
        assert request_body["tools"][0]["name"] == "get_weather"
        assert request_body["tool_choice"] == "auto"

        assert response.has_tool_use()
        tool_calls = response.get_tool_use()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

    async def test_send_with_dag(self) -> None:
        api = OpenAIAPI(api_key="test-api-key")
        api._stream_response = AsyncMock(  # type: ignore[method-assign]
            return_value={
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
        )

        dag = DAG().system("Be helpful.").user("Hi!")
        response = await api.send(dag)

        request_body = api._stream_response.await_args.args[0]
        input_items = request_body["input"]
        # First item is system prompt represented as a developer message.
        assert input_items[0] == {
            "role": "developer",
            "content": [{"type": "input_text", "text": "Be helpful."}],
        }
        # Second item is user message
        assert input_items[1] == {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hi!"}],
        }
        assert response.get_text() == "Hello!"

    async def test_stream_response_assembles_streamed_output_items(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.status_code = 200
                self.headers = {"content-type": "text/event-stream"}

            async def __aenter__(self) -> "FakeStream":
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            async def aiter_lines(self) -> Any:
                output_item = {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "streamed"}],
                    },
                }
                completed = {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_stream",
                        "model": "gpt-5.2-codex",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }
                yield f"data: {json.dumps(output_item)}"
                yield f"data: {json.dumps(completed)}"
                yield "data: [DONE]"

        class FakeClient:
            def stream(self, *args: Any, **kwargs: Any) -> FakeStream:
                return FakeStream()

        api = OpenAIAPI(api_key="test-api-key")
        api._client = FakeClient()  # type: ignore[assignment]

        result = await api._stream_response({})
        assert result is not None
        assert result["output"] == [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "streamed"}],
            }
        ]

    async def test_stream_response_raises_for_failed_event(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.status_code = 200
                self.headers = {"content-type": "text/event-stream"}

            async def __aenter__(self) -> "FakeStream":
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            async def aiter_lines(self) -> Any:
                failed = {
                    "type": "response.failed",
                    "response": {"error": {"message": "bad request"}},
                }
                yield f"data: {json.dumps(failed)}"

        class FakeClient:
            def stream(self, *args: Any, **kwargs: Any) -> FakeStream:
                return FakeStream()

        api = OpenAIAPI(api_key="test-api-key")
        api._client = FakeClient()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="bad request"):
            await api._stream_response({})


class TestRaiseOpenAIHttpError:
    def test_extracts_error_message_from_json(self) -> None:
        body = json.dumps(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
        ).encode()
        with pytest.raises(RuntimeError, match="OpenAI API error: Invalid API key"):
            _raise_openai_http_error(400, body)

    def test_falls_back_to_status_and_snippet(self) -> None:
        with pytest.raises(RuntimeError, match="OpenAI API HTTP 502: Bad gateway"):
            _raise_openai_http_error(502, b"Bad gateway")


class TestSendMethod:
    def test_send_method_is_async(self) -> None:
        """Verify send method is async."""
        api = OpenAIAPI(api_key="test-api-key")
        import asyncio

        assert asyncio.iscoroutinefunction(api.send)
