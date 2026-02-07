"""Claude API client with Claude Code compatible headers.

This module provides ClaudeCodeAPI, which uses hardcoded headers matching
the Claude Code CLI format for API authentication.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from ..dag import DAG
from ..data_structures import Message, Response, convert_message_to_claude_format
from ..tools import Tool, ToolDict
from .base import APIClientMixin

__all__ = ["ClaudeCodeAPI", "convert_message_to_claude_format"]

# Default headers matching Claude Code CLI format
DEFAULT_HEADERS = {
    "accept": "application/json",
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14",
    "anthropic-dangerous-direct-browser-access": "true",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
    "user-agent": "claude-cli/2.1.17 (external, sdk-cli)",
    "x-app": "cli",
}

# Default system messages matching Claude Code CLI format
DEFAULT_SYSTEM = [
    {
        "type": "text",
        "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    },
]


class ClaudeCodeAPI(APIClientMixin):
    """Claude API client with Claude Code compatible headers.

    Credential resolution order:
    1. Explicit auth_token parameter
    2. Config file (~/.nano-agent.json)

    This client uses hardcoded headers matching the Claude Code CLI format.
    Pass your OAuth token directly to authenticate, or use the config file
    created by get_config().

    Supports async context manager for proper resource cleanup:
        >>> async with ClaudeCodeAPI() as api:
        ...     dag = DAG().system("Be helpful.").user("Hello!")
        ...     response = await api.send(dag)
    """

    def __init__(
        self,
        auth_token: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking_budget: int | None = None,
        temperature: float | None = None,
        user_id: str | None = None,
        base_url: str | None = None,
        config_path: Path | str | None = None,
    ):
        """Initialize ClaudeCodeAPI with an auth token.

        Args:
            auth_token: OAuth token (sk-ant-oat01-...) or API key (sk-ant-...).
                       If not provided, loads from config file.
            model: Model to use (default from config or claude-sonnet-4-20250514)
            max_tokens: Maximum tokens in response (default from config or 16000)
            thinking_budget: Token budget for thinking (default from config or 10000)
            temperature: Temperature for sampling (default from config or 1.0)
            user_id: User ID for metadata (default from config or "nano-claude")
            base_url: API endpoint URL
            config_path: Path to config file (default: ~/.nano-agent.json)
        """
        # Defaults
        defaults: dict[str, Any] = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 16000,
            "thinking_budget": 10000,
            "temperature": 1.0,
            "user_id": "nano-claude",
            "base_url": "https://api.anthropic.com/v1/messages?beta=true",
        }

        # Try to load config file
        config_headers: dict[str, str] | None = None
        config_body: dict[str, Any] | None = None
        self.captured_system: list[dict[str, Any]] | None = None
        try:
            from .capture_claude_code_auth import load_config

            config = load_config(config_path)
            if config:
                config_headers, config_body = config
                # Update defaults from config
                if config_body.get("model"):
                    defaults["model"] = config_body["model"]
                if config_body.get("max_tokens"):
                    defaults["max_tokens"] = config_body["max_tokens"]
                if config_body.get("temperature") is not None:
                    defaults["temperature"] = config_body["temperature"]
                if config_body.get("thinking"):
                    thinking = config_body["thinking"]
                    if isinstance(thinking, dict) and thinking.get("budget_tokens"):
                        defaults["thinking_budget"] = thinking["budget_tokens"]
                if config_body.get("user_id"):
                    defaults["user_id"] = config_body["user_id"]
                if config_body.get("url_path"):
                    # Reconstruct base URL from captured path
                    url_path = config_body["url_path"]
                    defaults["base_url"] = f"https://api.anthropic.com{url_path}"
                # Store captured system messages (preserves cache_control)
                if config_body.get("system"):
                    self.captured_system = config_body["system"]
        except ImportError:
            pass  # Config loading not available

        # Resolve auth token: explicit > config
        resolved_token = auth_token
        if not resolved_token and config_headers:
            auth = config_headers.get("authorization", "")
            if auth.startswith("Bearer "):
                resolved_token = auth[7:]

        if not resolved_token:
            raise ValueError(
                "Auth token required. Provide auth_token or create ~/.nano-agent.json "
                "(via get_config())"
            )

        self.auth_token = resolved_token
        self.model = model or defaults["model"]
        self.max_tokens = max_tokens or defaults["max_tokens"]
        self.thinking_budget = thinking_budget or defaults["thinking_budget"]
        self.temperature = (
            temperature if temperature is not None else defaults["temperature"]
        )
        self.user_id = user_id or defaults["user_id"]
        self.base_url = base_url or defaults["base_url"]

        # Build headers (use config headers if available, otherwise defaults)
        if config_headers:
            self.headers = dict(config_headers)
            # Ensure content-type is set
            self.headers["content-type"] = "application/json"
            # Update authorization with potentially different token
            self.headers["authorization"] = f"Bearer {self.auth_token}"
        else:
            self.headers = DEFAULT_HEADERS.copy()
            self.headers["authorization"] = f"Bearer {self.auth_token}"

        # Ensure oauth beta flag is present for OAuth tokens (sk-ant-oat*)
        if self.auth_token.startswith("sk-ant-oat"):
            beta = self.headers.get("anthropic-beta", "")
            if "oauth-2025-04-20" not in beta:
                # Add oauth flag to existing beta features
                if beta:
                    self.headers["anthropic-beta"] = f"{beta},oauth-2025-04-20"
                else:
                    self.headers["anthropic-beta"] = "oauth-2025-04-20"

        # Create reusable HTTP client
        self._client = httpx.AsyncClient(timeout=600.0)

    def __repr__(self) -> str:
        """Return a clean representation of the API client configuration."""
        token_preview = (
            self.auth_token[:15] + "..."
            if len(self.auth_token) > 15
            else self.auth_token
        )
        return (
            f"ClaudeCodeAPI(\n"
            f"  model={self.model!r},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  thinking_budget={self.thinking_budget},\n"
            f"  temperature={self.temperature},\n"
            f"  user_id={self.user_id!r},\n"
            f"  base_url={self.base_url!r},\n"
            f"  token={token_preview!r}\n"
            f")"
        )

    @staticmethod
    def _add_system_cache_control(system_messages: list[dict[str, Any]]) -> None:
        """Add cache_control to the last 2 system blocks (up to 2 breakpoints).

        Strips any existing cache_control first to avoid exceeding the
        4-breakpoint limit when captured system messages already contain
        cache_control from the original Claude Code request.
        """
        for block in system_messages:
            block.pop("cache_control", None)
        for block in system_messages[-2:]:
            block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

    @staticmethod
    def _add_message_cache_control(messages_dicts: list[dict[str, Any]]) -> None:
        """Add cache_control to the last content block of the last 2 messages.

        This mirrors the Claude Code CLI pattern: cache_control on the
        second-to-last message (typically the last assistant response) and
        on the last message (the current user message).  Combined with 2
        system breakpoints this uses all 4 allowed breakpoints.
        """
        for msg in messages_dicts[-2:]:
            content = msg.get("content")
            if isinstance(content, list) and content:
                content[-1]["cache_control"] = {"type": "ephemeral"}
            elif isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

    async def send(
        self,
        messages: list[Message] | DAG,
        tools: Sequence[Tool] | None = None,
    ) -> Response:
        """Send a request to the Claude API.

        Accepts either a list of Message objects OR a DAG directly.

        Args:
            messages: List of Message objects OR a DAG instance
            tools: Tool definitions (ignored if messages is DAG)

        Returns:
            Response object
        """
        # Type dispatch for messages and tools
        if isinstance(messages, DAG):
            dag = messages
            actual_messages = dag.to_messages()
            actual_tools: list[Tool] = list(dag._tools or [])
            dag_system_prompts = dag.head.get_system_prompts() if dag._heads else []
            tools_list: list[ToolDict] = [tool.to_dict() for tool in actual_tools]
            messages = actual_messages

            # Build system: captured OR default, then append DAG prompts
            if self.captured_system:
                system_messages: list[dict[str, Any]] = list(self.captured_system)
            else:
                system_messages = list(DEFAULT_SYSTEM)
            if dag_system_prompts:
                system_messages.extend(
                    {"type": "text", "text": prompt} for prompt in dag_system_prompts
                )
            # Cache system blocks (up to 2 breakpoints on last 2 blocks)
            self._add_system_cache_control(system_messages)
        else:
            tools_list = []
            if tools:
                tools_list = [tool.to_dict() for tool in tools]
            # Use captured system OR default
            if self.captured_system:
                system_messages = list(self.captured_system)
            else:
                system_messages = list(DEFAULT_SYSTEM)
            self._add_system_cache_control(system_messages)

        # Build request body - convert messages to Claude format
        # (handles sessions created with OpenAI/Codex APIs)
        messages_dicts: list[dict[str, Any]] = [
            dict(convert_message_to_claude_format(msg.to_dict())) for msg in messages
        ]

        # Prompt caching: use up to 4 breakpoints total (matching Claude Code):
        #   2 on system blocks (above) + 2 on last 2 messages (below)
        # This caches the conversation prefix incrementally across turns.
        self._add_message_cache_control(messages_dicts)

        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages_dicts,
            "system": system_messages,
            "metadata": {"user_id": self.user_id},
            "max_tokens": self.max_tokens,
            "thinking": {
                "budget_tokens": self.thinking_budget,
                "type": "enabled",
            },
        }

        # Add tools if present
        if tools_list:
            request_body["tools"] = tools_list

        http_response = await self._client.post(
            self.base_url,
            headers=self.headers,
            json=request_body,
        )

        response_json = self._check_response(http_response, provider="Claude")
        return Response.from_dict(response_json)
