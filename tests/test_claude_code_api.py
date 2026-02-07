"""Tests for ClaudeCodeAPI."""

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nano_agent import DAG, ClaudeCodeAPI, Message, Role

# Mock captured config for testing
MOCK_HEADERS: dict[str, str] = {
    "authorization": "Bearer sk-ant-test123456789",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
    "user-agent": "claude-cli/2.0.76 (external, cli)",
    "x-app": "cli",
}

MOCK_BODY_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 16384,
    "temperature": 1.0,
    "thinking": {"budget_tokens": 10000, "type": "enabled"},
    "user_id": "test-user-123",
    "system": [
        {
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    "url_path": "/v1/messages?beta=true",
}


@pytest.fixture
def mock_load_config() -> Generator[MagicMock, None, None]:
    """Mock load_config to return test config without reading from file."""
    with patch("nano_agent.providers.capture_claude_code_auth.load_config") as mock:
        mock.return_value = (MOCK_HEADERS.copy(), MOCK_BODY_PARAMS.copy())
        yield mock


class TestClaudeCodeAPIInit:
    """Test ClaudeCodeAPI initialization."""

    def test_init_loads_config(self, mock_load_config: MagicMock) -> None:
        """Verify config is loaded at init."""
        api = ClaudeCodeAPI()
        mock_load_config.assert_called_once()

    def test_init_uses_loaded_values(self, mock_load_config: MagicMock) -> None:
        """Verify loaded config values are used."""
        api = ClaudeCodeAPI()
        assert api.model == "claude-sonnet-4-20250514"
        assert api.max_tokens == 16384
        assert api.temperature == 1.0
        assert api.thinking_budget == 10000
        assert api.user_id == "test-user-123"

    def test_parameter_overrides(self, mock_load_config: MagicMock) -> None:
        """Verify model/max_tokens can be overridden."""
        api = ClaudeCodeAPI(
            model="claude-opus-4-5-20251101",
            max_tokens=4096,
            temperature=0.5,
            thinking_budget=5000,
        )
        assert api.model == "claude-opus-4-5-20251101"
        assert api.max_tokens == 4096
        assert api.temperature == 0.5
        assert api.thinking_budget == 5000

    def test_init_raises_without_auth(self) -> None:
        """Verify clear error if no auth token available."""
        with patch("nano_agent.providers.capture_claude_code_auth.load_config") as mock:
            mock.return_value = None  # No config file
            with pytest.raises(ValueError) as exc_info:
                ClaudeCodeAPI()
            assert "Auth token required" in str(exc_info.value)

    def test_init_with_explicit_auth_token(self) -> None:
        """Verify explicit auth_token works without config file."""
        with patch("nano_agent.providers.capture_claude_code_auth.load_config") as mock:
            mock.return_value = None  # No config file
            api = ClaudeCodeAPI(auth_token="sk-ant-explicit-token")
            assert api.auth_token == "sk-ant-explicit-token"

    def test_explicit_auth_token_overrides_config(
        self, mock_load_config: MagicMock
    ) -> None:
        """Verify explicit auth_token takes precedence over config."""
        api = ClaudeCodeAPI(auth_token="sk-ant-explicit-token")
        assert api.auth_token == "sk-ant-explicit-token"


class TestClaudeCodeAPIRepr:
    """Test ClaudeCodeAPI repr."""

    def test_repr(self, mock_load_config: MagicMock) -> None:
        """Verify clean representation."""
        api = ClaudeCodeAPI()
        repr_str = repr(api)
        assert "ClaudeCodeAPI(" in repr_str
        assert "model='claude-sonnet-4-20250514'" in repr_str


class TestClaudeCodeAPISend:
    """Test ClaudeCodeAPI send method."""

    @pytest.fixture
    def mock_httpx_response(self) -> MagicMock:
        """Create a mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        return mock_response

    @pytest.mark.asyncio
    async def test_send_uses_auth_header(
        self, mock_load_config: MagicMock, mock_httpx_response: MagicMock
    ) -> None:
        """Verify send() uses auth from config."""
        api = ClaudeCodeAPI()
        with patch.object(api._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_httpx_response
            messages = [Message(role=Role.USER, content="Hello")]
            await api.send(messages)
            headers = mock_post.call_args.kwargs["headers"]
            assert "authorization" in headers or "x-api-key" in headers

    @pytest.mark.asyncio
    async def test_send_with_dag(
        self, mock_load_config: MagicMock, mock_httpx_response: MagicMock
    ) -> None:
        """Verify send() works with DAG input."""
        api = ClaudeCodeAPI()
        with patch.object(api._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_httpx_response
            dag = DAG().system("Be helpful.").user("Hello")
            response = await api.send(dag)
            assert response.get_text() == "Hello!"

    @pytest.mark.asyncio
    async def test_send_uses_captured_system(
        self, mock_load_config: MagicMock, mock_httpx_response: MagicMock
    ) -> None:
        """Verify captured system prompt is preserved."""
        api = ClaudeCodeAPI()
        with patch.object(api._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_httpx_response
            messages = [Message(role=Role.USER, content="Hello")]
            await api.send(messages)
            json_body = mock_post.call_args.kwargs["json"]
            assert json_body["system"] == MOCK_BODY_PARAMS["system"]


class TestClaudeCodeAPIContextManager:
    """Test ClaudeCodeAPI async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_load_config: MagicMock) -> None:
        """Verify async context manager works."""
        async with ClaudeCodeAPI() as api:
            assert api.model == "claude-sonnet-4-20250514"


class TestClaudeCodeAPIDefaults:
    """Test default values when config is incomplete or missing."""

    def test_defaults_when_config_values_missing(self) -> None:
        """Verify sensible defaults when config values are None."""
        with patch("nano_agent.providers.capture_claude_code_auth.load_config") as mock:
            mock.return_value = (
                {"authorization": "Bearer test-token"},
                {
                    "model": None,
                    "max_tokens": None,
                    "temperature": None,
                    "thinking": None,
                    "user_id": None,
                    "system": None,
                    "url_path": None,
                },
            )
            api = ClaudeCodeAPI()
            # Should use built-in defaults
            assert api.model == "claude-sonnet-4-20250514"
            assert api.max_tokens == 16000
            assert api.temperature == 1.0

    def test_defaults_when_no_config_but_explicit_auth(self) -> None:
        """Verify defaults work when no config file but auth provided."""
        with patch("nano_agent.providers.capture_claude_code_auth.load_config") as mock:
            mock.return_value = None
            api = ClaudeCodeAPI(auth_token="sk-ant-test")
            assert api.model == "claude-sonnet-4-20250514"
            assert api.max_tokens == 16000
