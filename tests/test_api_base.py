"""Tests for shared API infrastructure (providers.base module)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nano_agent.providers.base import (
    APPROX_BYTES_PER_TOKEN,
    APIClientMixin,
    APIError,
    ContextWindowExceededError,
    approx_token_count,
    flush_text_parts,
    is_context_window_error_payload,
    map_responses_status_to_stop_reason,
    parse_tool_arguments,
    responses_usage_from_dict,
    serialize_tool_arguments,
)


class TestSerializeToolArguments:
    def test_none_returns_empty_object(self) -> None:
        assert serialize_tool_arguments(None) == "{}"

    def test_dict_serializes_to_json(self) -> None:
        result = serialize_tool_arguments({"key": "value", "num": 42})
        assert '"key"' in result
        assert '"value"' in result
        assert "42" in result


class TestParseToolArguments:
    def test_valid_json(self) -> None:
        assert parse_tool_arguments('{"key": "value"}') == {"key": "value"}

    def test_invalid_json_returns_empty(self) -> None:
        assert parse_tool_arguments("not json") == {}

    def test_non_dict_returns_empty(self) -> None:
        assert parse_tool_arguments('"string"') == {}


class TestIsContextWindowErrorPayload:
    """Mirrors codex-rs ``is_context_window_error`` (codex-api/src/sse/responses.rs)."""

    def test_detects_context_length_exceeded_code(self) -> None:
        assert is_context_window_error_payload(
            {"code": "context_length_exceeded", "message": "too big"}
        )

    def test_does_not_match_other_codes(self) -> None:
        assert not is_context_window_error_payload(
            {"code": "rate_limit_exceeded", "message": "slow down"}
        )

    def test_does_not_match_when_code_missing(self) -> None:
        assert not is_context_window_error_payload({"message": "no code field"})

    def test_does_not_match_non_dict(self) -> None:
        assert not is_context_window_error_payload("just a string")
        assert not is_context_window_error_payload(None)


class TestContextWindowExceededError:
    def test_is_an_apierror(self) -> None:
        err = ContextWindowExceededError("too big", provider="Codex API")
        assert isinstance(err, APIError)
        assert err.status_code == 400
        assert err.error_type == "context_length_exceeded"
        assert err.provider == "Codex API"
        assert "too big" in str(err)


class TestApproxTokenCount:
    """Mirrors codex-rs's ``approx_token_count`` (utils/string/src/truncate.rs)."""

    def test_empty_string_is_zero_tokens(self) -> None:
        assert approx_token_count("") == 0

    def test_one_byte_rounds_up_to_one_token(self) -> None:
        assert approx_token_count("a") == 1

    def test_four_bytes_is_one_token(self) -> None:
        assert approx_token_count("abcd") == 1

    def test_five_bytes_rounds_up_to_two_tokens(self) -> None:
        assert approx_token_count("abcde") == 2

    def test_uses_utf8_byte_length_not_codepoints(self) -> None:
        # "é" is 2 bytes in UTF-8; with 4-bytes/token ceiling division this
        # is 1 token. Confirms we count bytes, not Python characters.
        assert approx_token_count("é") == 1
        # Four 2-byte chars = 8 bytes = 2 tokens.
        assert approx_token_count("éééé") == 2

    def test_byte_per_token_constant(self) -> None:
        assert APPROX_BYTES_PER_TOKEN == 4


class TestMapResponsesStatusToStopReason:
    def test_completed(self) -> None:
        assert map_responses_status_to_stop_reason("completed") == "end_turn"

    def test_failed(self) -> None:
        assert map_responses_status_to_stop_reason("failed") == "error"

    def test_incomplete(self) -> None:
        assert map_responses_status_to_stop_reason("incomplete") == "max_tokens"

    def test_in_progress(self) -> None:
        assert map_responses_status_to_stop_reason("in_progress") is None

    def test_unknown(self) -> None:
        assert map_responses_status_to_stop_reason("unknown_status") == "unknown_status"

    def test_empty(self) -> None:
        assert map_responses_status_to_stop_reason("") is None


class TestFlushTextParts:
    def test_appends_joined_message_and_clears(self) -> None:
        items: list[dict[str, Any]] = []
        parts = ["hello", "world"]
        flush_text_parts(items, "user", "input_text", parts)
        assert items == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello\nworld"}],
            }
        ]
        assert parts == []

    def test_noop_when_empty(self) -> None:
        items: list[dict[str, Any]] = []
        parts: list[str] = []
        flush_text_parts(items, "assistant", "output_text", parts)
        assert items == []
        assert parts == []


class TestResponsesUsageFromDict:
    def test_extracts_responses_api_fields(self) -> None:
        usage = responses_usage_from_dict(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_tokens_details": {"cached_tokens": 30},
                "output_tokens_details": {"reasoning_tokens": 20},
            }
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_tokens == 30
        assert usage.reasoning_tokens == 20
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_handles_missing_fields(self) -> None:
        usage = responses_usage_from_dict({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_handles_non_dict(self) -> None:
        usage = responses_usage_from_dict(None)
        assert usage.input_tokens == 0


class TestAPIError:
    """Tests for APIError exception class."""

    def test_basic_error(self) -> None:
        """Test basic APIError creation."""
        error = APIError("Something went wrong")
        assert str(error) == "[unknown] Something went wrong"
        assert error.status_code is None
        assert error.error_type is None
        assert error.provider == "unknown"

    def test_error_with_all_fields(self) -> None:
        """Test APIError with all fields populated."""
        error = APIError(
            message="Rate limit exceeded",
            status_code=429,
            error_type="rate_limit_error",
            provider="Claude",
        )
        assert str(error) == "[Claude] Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_type == "rate_limit_error"
        assert error.provider == "Claude"

    def test_error_repr(self) -> None:
        """Test APIError repr for debugging."""
        error = APIError(
            message="Invalid API key",
            status_code=401,
            error_type="authentication_error",
            provider="OpenAI",
        )
        repr_str = repr(error)
        assert "APIError" in repr_str
        assert "401" in repr_str
        assert "authentication_error" in repr_str
        assert "OpenAI" in repr_str

    def test_error_is_exception(self) -> None:
        """Test that APIError can be caught as Exception."""
        with pytest.raises(Exception):
            raise APIError("Test error")

    def test_error_inheritance(self) -> None:
        """Test that APIError inherits from Exception."""
        assert issubclass(APIError, Exception)


class ConcreteAPIClient(APIClientMixin):
    """Concrete implementation of APIClientMixin for testing."""

    def __init__(self) -> None:
        self._client = MagicMock(spec=httpx.AsyncClient)


class TestAPIClientMixin:
    """Tests for APIClientMixin shared functionality."""

    def test_check_response_success(self) -> None:
        """Test _check_response with successful response."""
        client = ConcreteAPIClient()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"id": "test", "content": "Hello"}

        result = client._check_response(response, provider="Test")
        assert result == {"id": "test", "content": "Hello"}

    def test_check_response_http_error(self) -> None:
        """Test _check_response with HTTP error status."""
        client = ConcreteAPIClient()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 401
        response.json.return_value = {
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key",
            }
        }

        with pytest.raises(APIError) as exc_info:
            client._check_response(response, provider="Claude")

        error = exc_info.value
        assert error.status_code == 401
        assert error.error_type == "authentication_error"
        assert error.provider == "Claude"
        assert "Invalid API key" in str(error)

    def test_check_response_http_error_no_message(self) -> None:
        """Test _check_response with HTTP error but no structured message."""
        client = ConcreteAPIClient()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.json.return_value = {"status": "error"}

        with pytest.raises(APIError) as exc_info:
            client._check_response(response, provider="Test")

        error = exc_info.value
        assert error.status_code == 500
        assert error.error_type == "unknown"

    def test_check_response_rate_limit(self) -> None:
        """Test _check_response with rate limit error."""
        client = ConcreteAPIClient()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 429
        response.json.return_value = {
            "error": {
                "type": "rate_limit_error",
                "message": "Too many requests",
            }
        }

        with pytest.raises(APIError) as exc_info:
            client._check_response(response, provider="Claude")

        error = exc_info.value
        assert error.status_code == 429

    def test_check_response_error_key_in_body(self) -> None:
        """Test _check_response with error key in 200 response (OpenAI pattern)."""
        client = ConcreteAPIClient()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "error": {
                "type": "invalid_request",
                "message": "Invalid model specified",
            }
        }

        with pytest.raises(APIError) as exc_info:
            client._check_response(response, provider="OpenAI")

        error = exc_info.value
        assert error.status_code is None  # HTTP was 200
        assert error.error_type == "invalid_request"
        assert error.provider == "OpenAI"

    def test_check_response_error_key_string(self) -> None:
        """Test _check_response with string error value."""
        client = ConcreteAPIClient()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"error": "Something went wrong"}

        with pytest.raises(APIError) as exc_info:
            client._check_response(response, provider="Test")

        error = exc_info.value
        assert "Something went wrong" in str(error)

    def test_check_response_non_dict_error_data(self) -> None:
        """Test _check_response with non-dict error in non-200 response."""
        client = ConcreteAPIClient()

        response = MagicMock(spec=httpx.Response)
        response.status_code = 400
        response.json.return_value = {"error": "Bad request string"}

        with pytest.raises(APIError) as exc_info:
            client._check_response(response, provider="Test")

        error = exc_info.value
        assert error.status_code == 400
        assert "Bad request string" in str(error)

    async def test_close(self) -> None:
        """Test close() calls aclose on httpx client."""
        client = ConcreteAPIClient()
        mock_aclose = AsyncMock()

        with patch.object(client._client, "aclose", mock_aclose):
            await client.close()
            mock_aclose.assert_called_once()

    async def test_context_manager(self) -> None:
        """Test async context manager support."""
        client = ConcreteAPIClient()
        mock_aclose = AsyncMock()

        with patch.object(client._client, "aclose", mock_aclose):
            async with client as ctx:
                assert ctx is client

            mock_aclose.assert_called_once()

    async def test_context_manager_on_exception(self) -> None:
        """Test context manager closes client even on exception."""
        client = ConcreteAPIClient()
        mock_aclose = AsyncMock()

        with patch.object(client._client, "aclose", mock_aclose):
            with pytest.raises(ValueError):
                async with client:
                    raise ValueError("Test error")

            mock_aclose.assert_called_once()


class TestAPIErrorInPractice:
    """Tests for using APIError in practice scenarios."""

    def test_catching_specific_status_codes(self) -> None:
        """Test that status codes can be used for control flow."""
        retryable_codes = {429, 500, 502, 503, 504}

        error = APIError("Rate limited", status_code=429, provider="Claude")
        assert error.status_code in retryable_codes

        error = APIError("Auth failed", status_code=401, provider="Claude")
        assert error.status_code not in retryable_codes

    def test_provider_specific_handling(self) -> None:
        """Test handling errors differently by provider."""
        claude_error = APIError("Test", provider="Claude")
        openai_error = APIError("Test", provider="OpenAI")
        gemini_error = APIError("Test", provider="Gemini")

        assert "[Claude]" in str(claude_error)
        assert "[OpenAI]" in str(openai_error)
        assert "[Gemini]" in str(gemini_error)
