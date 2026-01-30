"""Claude API client."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, TypedDict

import httpx

from .dag import DAG
from .data_structures import (
    ContentBlock,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
    convert_message_to_claude_format,
)
from .tools import Tool

# Re-export for backwards compatibility
__all__ = [
    "ClaudeAPI",
    "Message",
    "Role",
    "TextContent",
    "ThinkingContent",
    "ToolUseContent",
    "ToolResultContent",
    "Usage",
    "Response",
    "ContentBlock",
]


# =============================================================================
# API-specific types (TypedDicts for parsing)
# =============================================================================


class TextBlockDict(TypedDict, total=False):
    type: str
    text: str


class ThinkingBlockDict(TypedDict, total=False):
    type: str
    thinking: str
    signature: str


class ToolUseBlockDict(TypedDict, total=False):
    type: str
    id: str
    name: str
    input: dict[str, Any]


class UsageDict(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class ResponseDict(TypedDict, total=False):
    id: str
    model: str
    role: str
    content: list[TextBlockDict | ThinkingBlockDict | ToolUseBlockDict]
    stop_reason: str | None
    usage: UsageDict


class ClaudeAPI:
    """Claude API client - simple, curl-like interface to the Anthropic API.

    For CLI authentication (OAuth, captured headers), use ClaudeCodeAPI instead.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 16000,
        temperature: float = 1.0,
        thinking_budget: int = 10000,
        base_url: str = "https://api.anthropic.com/v1/messages",
    ):
        """
        Initialize Claude API client.

        Args:
            api_key: API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            model: Claude model to use.
            max_tokens: Maximum tokens in response.
            temperature: Temperature setting.
            thinking_budget: Thinking budget tokens for extended thinking.
            base_url: API base URL.
        """
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("API key required: pass api_key or set ANTHROPIC_API_KEY")
        self.api_key: str = resolved_key

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.base_url = base_url

        # Create reusable HTTP client for connection pooling
        self._client = httpx.AsyncClient()

    def __repr__(self) -> str:
        """Return a clean representation of the API client configuration."""
        token_preview = (
            self.api_key[:15] + "..." if len(self.api_key) > 15 else self.api_key
        )
        return (
            f"ClaudeAPI(model={self.model!r}, max_tokens={self.max_tokens}, "
            f"token={token_preview!r})"
        )

    async def send(
        self,
        messages: "list[Message] | DAG",
        tools: Sequence[Tool] | None = None,
        system: str | list[dict[str, Any]] | None = None,
    ) -> Response:
        """Send a request to the Claude API.

        Accepts either traditional arguments OR a DAG directly.

        Args:
            messages: List of Message objects OR a DAG instance
            tools: Tool definitions (ignored if messages is DAG)
            system: System prompt as string or list of dicts (ignored if messages is DAG)

        Returns:
            Response object
        """
        # Extract from DAG if provided
        if isinstance(messages, DAG):
            dag = messages
            messages = dag.to_messages()
            tools = list(dag._tools or [])
            dag_system = dag.head.get_system_prompts() if dag._heads else []
            if dag_system and system is None:
                system = "\n\n".join(dag_system)

        # Build simple headers
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        # Build request body - convert messages to Claude format
        # (handles sessions created with OpenAI/Codex APIs)
        messages_dicts = [
            convert_message_to_claude_format(msg.to_dict()) for msg in messages
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages_dicts,
            "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget},
        }

        if system:
            body["system"] = system
        if tools:
            body["tools"] = [tool.to_dict() for tool in tools]

        http_response = await self._client.post(
            self.base_url, headers=headers, json=body
        )
        return Response.from_dict(http_response.json())


if __name__ == "__main__":
    import asyncio

    from .tools import BashTool, ReadTool

    async def main() -> None:
        # Uses ANTHROPIC_API_KEY from environment
        client = ClaudeAPI()

        # Example with tools
        tools = [BashTool(), ReadTool()]
        messages = [Message(role=Role.USER, content="What is 2+2? Answer in one word.")]
        response = await client.send(messages, tools=tools)
        print(f"Response: {response.get_text()}")
        print(f"Usage: {response.usage}")

    asyncio.run(main())
