"""Tests for Codex API client."""

import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar
from unittest.mock import AsyncMock, Mock

import pytest

from nano_agent import (
    DAG,
    CompactionContent,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from nano_agent.data_structures import parse_content_block
from nano_agent.providers.codex import (
    CodexAPI,
    _convert_compacted_output_to_messages,
    _map_status_to_stop_reason,
)
from nano_agent.tools import Desc, Tool


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


class TestCodexAPIInit:
    def test_init_with_auth_token(self) -> None:
        api = CodexAPI(auth_token="test-token")
        assert api.auth_token == "test-token"
        assert api.model == "gpt-5.5"
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
        assert "gpt-5.5" in repr_str
        assert "token-123456789..." in repr_str

    def test_auto_compact_token_limit_disabled_by_default(self) -> None:
        api = CodexAPI(auth_token="test-token")
        assert api.context_window is None
        assert api.auto_compact_token_limit is None

    def test_auto_compact_token_limit_is_90_percent(self) -> None:
        api = CodexAPI(auth_token="test-token", context_window=100_000)
        assert api.context_window == 100_000
        assert api.auto_compact_token_limit == 90_000

    def test_auto_compact_token_limit_floor_division(self) -> None:
        api = CodexAPI(auth_token="test-token", context_window=128_001)
        assert api.auto_compact_token_limit == (128_001 * 9) // 10


class TestListModels:
    """Mirrors codex-rs's ``ModelsClient.list_models`` (codex-api/src/endpoint/models.rs)."""

    def test_models_url_is_root_plus_models(self) -> None:
        api = CodexAPI(
            auth_token="test-token",
            base_url="https://chatgpt.com/backend-api/codex/responses",
        )
        # codex-rs hits ``<root>/models`` where root is one path segment up
        # from ``/responses``.
        assert api._models_url() == "https://chatgpt.com/backend-api/codex/models"

    def test_models_url_strips_trailing_slash(self) -> None:
        api = CodexAPI(
            auth_token="test-token",
            base_url="https://chatgpt.com/backend-api/codex/responses/",
        )
        assert api._models_url() == "https://chatgpt.com/backend-api/codex/models"

    async def test_list_models_parses_envelope(self, tmp_path: Any) -> None:
        api = CodexAPI(auth_token="test-token", codex_home=tmp_path)
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json = Mock(
            return_value={
                "models": [
                    {"slug": "gpt-5.5", "context_window": 272_000},
                    {"slug": "gpt-5.5-mini", "context_window": 128_000},
                ]
            }
        )
        api._client.get = AsyncMock(return_value=mock_resp)

        models = await api.list_models()

        assert len(models) == 2
        assert models[0]["slug"] == "gpt-5.5"
        assert models[0]["context_window"] == 272_000
        # Verify URL + headers used.
        api._client.get.assert_awaited_once()
        kwargs = api._client.get.await_args.kwargs
        assert "x-codex-installation-id" in kwargs["headers"]
        assert kwargs["timeout"] == 5.0  # MODELS_REFRESH_TIMEOUT in codex-rs.
        # codex-rs appends ?client_version=X.Y.Z (codex-api/src/endpoint/models.rs).
        assert kwargs["params"] == {"client_version": "0.125.0"}

    async def test_list_models_returns_empty_on_malformed_envelope(
        self, tmp_path: Any
    ) -> None:
        api = CodexAPI(auth_token="test-token", codex_home=tmp_path)
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json = Mock(return_value={"unexpected": "shape"})
        api._client.get = AsyncMock(return_value=mock_resp)

        assert await api.list_models() == []

    async def test_list_models_raises_on_http_error(self, tmp_path: Any) -> None:
        api = CodexAPI(auth_token="test-token", codex_home=tmp_path)
        mock_resp = Mock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        api._client.get = AsyncMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="500"):
            await api.list_models()

    async def test_fetch_context_window_finds_model(self, tmp_path: Any) -> None:
        api = CodexAPI(auth_token="test-token", model="gpt-5.5", codex_home=tmp_path)
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json = Mock(
            return_value={
                "models": [
                    {"slug": "gpt-5.5-mini", "context_window": 128_000},
                    {"slug": "gpt-5.5", "context_window": 272_000},
                ]
            }
        )
        api._client.get = AsyncMock(return_value=mock_resp)

        assert await api.fetch_context_window() == 272_000

    async def test_fetch_context_window_falls_back_to_max(self, tmp_path: Any) -> None:
        # codex-rs ``ModelInfo.resolved_context_window`` falls back to
        # ``max_context_window`` when ``context_window`` is missing/null.
        api = CodexAPI(auth_token="test-token", model="gpt-5.5", codex_home=tmp_path)
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json = Mock(
            return_value={
                "models": [
                    {"slug": "gpt-5.5", "max_context_window": 272_000},
                ]
            }
        )
        api._client.get = AsyncMock(return_value=mock_resp)

        assert await api.fetch_context_window() == 272_000

    async def test_fetch_context_window_returns_none_when_slug_missing(
        self, tmp_path: Any
    ) -> None:
        api = CodexAPI(auth_token="test-token", model="gpt-7", codex_home=tmp_path)
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json = Mock(
            return_value={"models": [{"slug": "gpt-5.5", "context_window": 272_000}]}
        )
        api._client.get = AsyncMock(return_value=mock_resp)

        assert await api.fetch_context_window() is None


class TestSend:
    async def test_send_uses_codex_request_shape(self) -> None:
        api = CodexAPI(
            auth_token="test-token",
            model="gpt-test",
            installation_id="install-test",
        )
        api._stream_response = AsyncMock(
            return_value={
                "id": "resp_test",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

        response = await api.send([Message(role=Role.USER, content="Hi")])

        api._stream_response.assert_awaited_once()
        request_body = api._stream_response.await_args.args[0]
        assert request_body["model"] == "gpt-test"
        assert request_body["instructions"] == "You are a helpful assistant."
        assert request_body["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        ]
        assert request_body["tool_choice"] == "auto"
        assert request_body["parallel_tool_calls"] is True
        assert request_body["prompt_cache_key"] == api._session_id
        assert request_body["client_metadata"] == {
            "x-codex-installation-id": "install-test"
        }
        assert request_body["store"] is False
        assert request_body["stream"] is True
        assert response.get_text() == "Hello"

    async def test_send_persists_installation_id(self, tmp_path: Any) -> None:
        api = CodexAPI(auth_token="test-token", codex_home=tmp_path)
        api._stream_response = AsyncMock(
            return_value={
                "id": "resp_test",
                "model": "gpt-5.5",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

        await api.send([Message(role=Role.USER, content="Hi")])

        request_body = api._stream_response.await_args.args[0]
        installation_id = request_body["client_metadata"]["x-codex-installation-id"]
        assert (tmp_path / "installation_id").read_text() == installation_id

    async def test_stream_response_replays_turn_state(self) -> None:
        class FakeStream:
            def __init__(self, turn_state: str) -> None:
                self.status_code = 200
                self.headers = {
                    "content-type": "text/event-stream",
                    "x-codex-turn-state": turn_state,
                }

            async def __aenter__(self) -> "FakeStream":
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
                return None

            async def aiter_lines(self) -> Any:
                completed = {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_test",
                        "model": "gpt-5.5",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }
                yield f"data: {json.dumps(completed)}"
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self) -> None:
                self.headers: list[dict[str, str]] = []

            def stream(self, *args: Any, **kwargs: Any) -> FakeStream:
                self.headers.append(kwargs["headers"])
                return FakeStream(f"turn-{len(self.headers)}")

        api = CodexAPI(auth_token="test-token", installation_id="install-test")
        fake_client = FakeClient()
        api._client = fake_client

        await api._stream_response({})
        await api._stream_response({})

        assert "x-codex-turn-state" not in fake_client.headers[0]
        assert fake_client.headers[1]["x-codex-turn-state"] == "turn-1"


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

    def test_convert_replays_reasoning_with_encrypted_content(self) -> None:
        """Reasoning items returned by the backend must be replayed in the
        next turn so chain-of-thought continuity and prefix caching work."""
        api = CodexAPI(auth_token="test-token", reasoning=True)
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(
                    thinking="thought summary",
                    id="rs_abc",
                    encrypted_content="ENCRYPTED_BLOB",
                    summary=({"type": "summary_text", "text": "thought summary"},),
                ),
                TextContent(text="answer"),
            ],
        )
        items = api._convert_message_to_codex(msg)

        assert items[0] == {
            "type": "reasoning",
            "id": "rs_abc",
            "encrypted_content": "ENCRYPTED_BLOB",
            "summary": [{"type": "summary_text", "text": "thought summary"}],
        }
        assert items[1] == {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
        }

    def test_convert_drops_reasoning_when_disabled(self) -> None:
        api = CodexAPI(auth_token="test-token", reasoning=False)
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(
                    thinking="t",
                    id="rs_abc",
                    encrypted_content="ENCRYPTED_BLOB",
                ),
                TextContent(text="answer"),
            ],
        )
        items = api._convert_message_to_codex(msg)

        assert all(item.get("type") != "reasoning" for item in items)
        assert items == [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            }
        ]

    def test_convert_drops_reasoning_without_encrypted_content(self) -> None:
        """Claude-style ``thinking`` blocks without encrypted_content cannot be
        sent to the Responses API; they must be dropped even when reasoning
        is enabled."""
        api = CodexAPI(auth_token="test-token", reasoning=True)
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="claude-style", signature="sig"),
                TextContent(text="answer"),
            ],
        )
        items = api._convert_message_to_codex(msg)

        assert all(item.get("type") != "reasoning" for item in items)
        assert all(item.get("type") != "thinking" for item in items)


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

    def test_convert_tool_to_codex_respects_non_strict_tool(self) -> None:
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

        api = CodexAPI(auth_token="test-token")
        result = api._convert_tool_to_codex(SearchTool())

        assert result["strict"] is False
        assert result["parameters"]["required"] == ["query"]

    def test_strict_tool_promotes_default_fields_to_required(self) -> None:
        """With strict=True, every property must appear in ``required`` or
        the Codex backend rejects the tool."""

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

        api = CodexAPI(auth_token="test-token")
        result = api._convert_tool_to_codex(SearchTool())

        assert result["strict"] is True
        assert sorted(result["parameters"]["required"]) == ["limit", "query"]
        assert result["parameters"]["additionalProperties"] is False

    def test_strict_rewrite_does_not_mutate_cached_input_schema(self) -> None:
        """``Tool._inferred_schema`` is a ``ClassVar`` shared across instances;
        the strict-mode rewrite must not mutate it or future conversions —
        even from a different provider — would see corrupted state."""

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

        api = CodexAPI(auth_token="test-token")
        tool = FooTool()
        before = list(tool.input_schema["required"])  # type: ignore[arg-type]

        api._convert_tool_to_codex(tool)
        api._convert_tool_to_codex(tool)

        assert list(tool.input_schema["required"]) == before  # type: ignore[arg-type]


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


class TestSSEParsing:
    def test_parse_sse_text_attaches_collected_output_items(self) -> None:
        api = CodexAPI(auth_token="test-token")
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
                "model": "gpt-5.5",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }

        parsed = api._parse_sse_text(
            f"data: {json.dumps(output_item)}\n\n"
            f"data: {json.dumps(completed)}\n\n"
            "data: [DONE]\n\n"
        )

        assert parsed is not None
        assert parsed["output"] == [output_item["item"]]

    def test_parse_sse_text_raises_for_failed_response(self) -> None:
        api = CodexAPI(auth_token="test-token")
        failed = {
            "type": "response.failed",
            "response": {"error": {"message": "bad request"}},
        }

        with pytest.raises(RuntimeError, match="bad request"):
            api._parse_sse_text(f"data: {json.dumps(failed)}\n\n")


class TestCodexAPIInitErrors:
    def test_init_without_token_raises(self) -> None:
        from unittest.mock import patch

        with patch(
            "nano_agent.providers.codex.get_codex_access_token", return_value=None
        ):
            with pytest.raises(ValueError, match="Codex OAuth token required"):
                CodexAPI(auth_token=None)


class TestCompact:
    async def test_compact_request_body_shape(self) -> None:
        api = CodexAPI(auth_token="test-token", installation_id="install-test")
        api._post_compact = AsyncMock(  # type: ignore[method-assign]
            return_value={"output": []}
        )

        await api.compact(
            [Message(role=Role.USER, content="hello")],
            system_prompt="be terse",
        )

        api._post_compact.assert_awaited_once()
        body = api._post_compact.await_args.args[0]
        assert body["model"] == api.model
        assert body["instructions"] == "be terse"
        assert body["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ]
        assert body["parallel_tool_calls"] is True
        assert body["tools"] == []
        # Compaction body must NOT include any of these — they are accepted
        # only by /responses (codex-api/common.rs:25-36).
        for forbidden in (
            "stream",
            "store",
            "prompt_cache_key",
            "client_metadata",
            "include",
            "tool_choice",
        ):
            assert forbidden not in body, f"compact body must not include {forbidden}"

    async def test_compact_short_circuits_on_empty_input(self) -> None:
        api = CodexAPI(auth_token="test-token")
        api._post_compact = AsyncMock()  # type: ignore[method-assign]

        result = await api.compact([])

        assert result == []
        api._post_compact.assert_not_awaited()

    async def test_post_compact_uses_compact_url_and_header_installation_id(
        self,
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_post(
            url: str, *, headers: dict[str, str], json: dict[str, Any]
        ) -> Any:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json

            class FakeResp:
                status_code = 200
                text = ""
                headers: dict[str, str] = {}

                def json(self) -> dict[str, Any]:
                    return {"output": []}

            return FakeResp()

        api = CodexAPI(auth_token="test-token", installation_id="install-test")
        api._client = Mock()  # type: ignore[assignment]
        api._client.post = fake_post

        await api._post_compact({"foo": "bar"})

        assert captured["url"].endswith("/responses/compact")
        assert captured["url"] != api.base_url
        # x-codex-installation-id is an HTTP HEADER on /responses/compact
        # (client.rs:469-471), not a body field.
        assert captured["headers"]["x-codex-installation-id"] == "install-test"
        assert "x-codex-installation-id" not in captured["json"]
        assert captured["headers"]["Accept"] == "application/json"

    async def test_post_compact_raises_on_non_200(self) -> None:
        async def fake_post(*args: Any, **kwargs: Any) -> Any:
            class FakeResp:
                status_code = 500
                text = "boom"
                headers: dict[str, str] = {}

            return FakeResp()

        api = CodexAPI(auth_token="test-token")
        api._client = Mock()  # type: ignore[assignment]
        api._client.post = fake_post

        with pytest.raises(RuntimeError, match="Codex API HTTP 500"):
            await api._post_compact({})


class TestCompactedOutputFiltering:
    def test_drops_developer_messages(self) -> None:
        messages = _convert_compacted_output_to_messages(
            [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "instructions"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "summary"}],
                },
            ]
        )

        assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT]
        assert [m.content for m in messages] == ["hi", "summary"]

    def test_drops_non_message_items(self) -> None:
        messages = _convert_compacted_output_to_messages(
            [
                {"type": "reasoning", "id": "r1", "encrypted_content": "..."},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "tool",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "result",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "kept"}],
                },
            ]
        )

        assert len(messages) == 1
        assert messages[0].content == "kept"

    def test_drops_empty_user_content_wrappers(self) -> None:
        messages = _convert_compacted_output_to_messages(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "output_text", "text": "wrapper"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "real user"}],
                },
            ]
        )

        assert len(messages) == 1
        assert messages[0].content == "real user"


class TestCompactDag:
    async def test_compact_dag_returns_empty_dag_unchanged(self) -> None:
        api = CodexAPI(auth_token="test-token")
        api._post_compact = AsyncMock()  # type: ignore[method-assign]

        empty = DAG()
        result = await api.compact_dag(empty)

        assert result is empty
        api._post_compact.assert_not_awaited()

    async def test_compact_dag_preserves_system_and_tools(self) -> None:
        @dataclass
        class FooInput:
            x: Annotated[str, Desc("x")] = ""

        @dataclass
        class FooTool(Tool):
            name: str = "foo"
            description: str = "foo"

            async def __call__(self, input: FooInput) -> TextContent:
                return TextContent(text="")

        api = CodexAPI(auth_token="test-token")
        api._post_compact = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "output": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "summarized history"}
                        ],
                    },
                ]
            }
        )

        original = (
            DAG()
            .system("be brief")
            .tools(FooTool())
            .user("first message")
            .assistant("first reply")
            .user("second message")
        )

        compacted = await api.compact_dag(original)

        assert compacted._tools is not None
        assert len(compacted._tools) == 1
        assert compacted._tools[0].name == "foo"
        # System prompt is preserved
        assert "be brief" in compacted.head.get_system_prompts()[0]
        # Compacted history replaces the messages
        new_messages = compacted.to_messages()
        assert len(new_messages) == 1
        assert new_messages[0].role == Role.USER
        assert new_messages[0].content == "summarized history"

    async def test_compact_dag_retries_with_trimmed_history_on_cwe(self) -> None:
        """Mirrors codex-rs compact.rs:216-226: drop oldest, retry."""
        from nano_agent.providers.base import ContextWindowExceededError

        api = CodexAPI(auth_token="test-token")
        # First two attempts fail with CWE; third succeeds.
        success_payload: dict[str, Any] = {
            "output": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "summary"}],
                },
            ]
        }
        attempts: list[int] = []

        async def fake_post_compact(body: dict[str, Any]) -> dict[str, Any]:
            attempts.append(len(body["input"]))
            if len(attempts) < 3:
                raise ContextWindowExceededError("too big", provider="Codex API")
            return success_payload

        api._post_compact = fake_post_compact  # type: ignore[method-assign]

        original = (
            DAG()
            .system("sys")
            .user("oldest")
            .assistant("a1")
            .user("middle")
            .assistant("a2")
            .user("newest")
        )

        compacted = await api.compact_dag(original)

        # 3 attempts: full -> drop oldest -> drop two oldest.
        assert len(attempts) == 3
        # Each attempt sent strictly fewer items than the previous (trimming
        # from the front mirrors history.remove_first_item()).
        assert attempts[1] < attempts[0]
        assert attempts[2] < attempts[1]
        # Final compacted DAG carries the summary user message.
        new_messages = compacted.to_messages()
        assert len(new_messages) == 1
        assert new_messages[0].content == "summary"

    async def test_compact_dag_gives_up_when_only_one_message_remains(self) -> None:
        """Mirrors codex-rs compact.rs:227-230: stop trimming at 1 item."""
        from nano_agent.providers.base import ContextWindowExceededError

        api = CodexAPI(auth_token="test-token")
        api._post_compact = AsyncMock(  # type: ignore[method-assign]
            side_effect=ContextWindowExceededError("too big", provider="Codex API")
        )

        original = DAG().system("sys").user("only one")

        with pytest.raises(ContextWindowExceededError):
            await api.compact_dag(original)

        # Two attempts at most: full (1 message), then nothing left to trim.
        # codex-rs's check is ``turn_input_len > 1``; matching that, we only
        # attempt once when the history starts at 1 message.
        assert api._post_compact.await_count == 1


class TestCompactionEncryptedContentRoundTrip:
    """The opaque ``Compaction { encrypted_content }`` checkpoint emitted by
    /responses/compact must be replayed verbatim on the next /responses
    request — the server uses it to keep the compacted context alive
    (codex-rs ``compact_remote.rs:282`` keeps it; ``client_common.rs:65``
    forwards it through ``get_formatted_input`` to the next call).
    """

    def test_compaction_item_is_kept_in_output(self) -> None:
        messages = _convert_compacted_output_to_messages(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "history"}],
                },
                {"type": "compaction", "encrypted_content": "OPAQUE_BLOB_1"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "summary"}],
                },
            ]
        )

        assert len(messages) == 3
        assert messages[0].content == "history"
        assert messages[1].role == Role.ASSISTANT
        assert isinstance(messages[1].content, list)
        assert isinstance(messages[1].content[0], CompactionContent)
        assert messages[1].content[0].encrypted_content == "OPAQUE_BLOB_1"
        assert messages[2].content == "summary"

    def test_compaction_summary_alias_is_accepted(self) -> None:
        messages = _convert_compacted_output_to_messages(
            [{"type": "compaction_summary", "encrypted_content": "ALIAS"}]
        )

        assert len(messages) == 1
        assert isinstance(messages[0].content, list)
        assert isinstance(messages[0].content[0], CompactionContent)
        assert messages[0].content[0].encrypted_content == "ALIAS"

    def test_replays_compaction_as_top_level_item(self) -> None:
        api = CodexAPI(auth_token="test-token")
        msg = Message(
            role=Role.ASSISTANT,
            content=[CompactionContent(encrypted_content="REPLAY_ME")],
        )

        items = api._convert_message_to_codex(msg)

        assert items == [{"type": "compaction", "encrypted_content": "REPLAY_ME"}]

    def test_replays_compaction_between_text_blocks(self) -> None:
        """Compaction items must keep their position relative to text in the
        wire-level item list; otherwise the server's reconstruction order
        breaks."""
        api = CodexAPI(auth_token="test-token")
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="before"),
                CompactionContent(encrypted_content="MID"),
                TextContent(text="after"),
            ],
        )

        items = api._convert_message_to_codex(msg)

        assert items == [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "before"}],
            },
            {"type": "compaction", "encrypted_content": "MID"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "after"}],
            },
        ]

    async def test_compact_dag_preserves_compaction_for_next_send(self) -> None:
        """End-to-end: server returns compaction in /compact output,
        compact_dag stores it on the new DAG, next send() emits it back."""
        api = CodexAPI(auth_token="test-token")
        api._post_compact = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "output": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "earlier turn"}],
                    },
                    {"type": "compaction", "encrypted_content": "CHECKPOINT_42"},
                ]
            }
        )

        original = DAG().user("first").assistant("reply")
        compacted = await api.compact_dag(original)

        # Round-trip: serialize the compacted DAG's messages back into wire
        # items (what the next send() will POST in ``input``).
        wire_items: list[dict[str, Any]] = []
        for msg in compacted.to_messages():
            wire_items.extend(api._convert_message_to_codex(msg))

        compaction_items = [i for i in wire_items if i.get("type") == "compaction"]
        assert len(compaction_items) == 1
        assert compaction_items[0]["encrypted_content"] == "CHECKPOINT_42"

    def test_compaction_content_round_trips_through_parse_content_block(
        self,
    ) -> None:
        block = parse_content_block(
            {"type": "compaction", "encrypted_content": "FROM_DICT"}
        )

        assert isinstance(block, CompactionContent)
        assert block.encrypted_content == "FROM_DICT"
        assert block.to_dict() == {
            "type": "compaction",
            "encrypted_content": "FROM_DICT",
        }
