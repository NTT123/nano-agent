"""Gemini API client for the Generative Language API."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import httpx

from .api_base import APIClientMixin
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
)
from .tools import Tool

__all__ = ["GeminiAPI"]


class GeminiAPI(APIClientMixin):
    """Gemini API client using the Generative Language API.

    This client follows the same patterns as ClaudeAPI and OpenAIAPI, reusing
    existing data structures and providing a consistent interface.

    Gemini API differences from Claude/OpenAI:
    - Uses `model` role instead of `assistant`
    - Uses `parts` array instead of `content` array
    - Uses `functionCall`/`functionResponse` for tools (camelCase in response)
    - API key is passed as query parameter, not header
    - System prompt uses `systemInstruction` field
    - Thinking uses `thinkingConfig.thinkingLevel`

    Supports async context manager for proper resource cleanup:
        >>> async with GeminiAPI() as api:
        ...     response = await api.send(dag)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        max_tokens: int = 8192,
        temperature: float = 1.0,
        thinking_level: str | None = "low",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 120.0,
    ):
        """Initialize Gemini API client.

        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model: Model to use (default: gemini-3-flash-preview)
            max_tokens: Maximum tokens in response
            temperature: Temperature setting
            thinking_level: Thinking level (None, "off", "low", "medium", "high")
                           Set to None or "off" to disable thinking.
            base_url: API base URL
            timeout: Request timeout in seconds (default: 120s).
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key or set GEMINI_API_KEY env var."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_level = thinking_level
        self.base_url = base_url
        self.timeout = timeout

        # Create reusable HTTP client for connection pooling
        self._client = httpx.AsyncClient(timeout=timeout)

    def __repr__(self) -> str:
        """Return a clean representation of the API client configuration."""
        token_preview = (
            self.api_key[:15] + "..."
            if self.api_key and len(self.api_key) > 15
            else self.api_key
        )
        return (
            f"GeminiAPI(\n"
            f"  model={self.model!r},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  temperature={self.temperature},\n"
            f"  thinking_level={self.thinking_level!r},\n"
            f"  base_url={self.base_url!r},\n"
            f"  token={token_preview!r}\n"
            f")"
        )

    def _convert_message_to_gemini(
        self, msg: Message, tool_name_map: dict[str, str]
    ) -> dict[str, Any]:
        """Convert a Message to Gemini format.

        Gemini format:
        - Uses `model` role instead of `assistant`
        - Uses `parts` array with `text`, `functionCall`, `functionResponse`
        - functionResponse requires the function name, not just call_id

        Args:
            msg: The message to convert
            tool_name_map: Mapping from tool_use_id to tool name for functionResponse
        """
        # Role mapping: assistant -> model
        role = "model" if msg.role == Role.ASSISTANT else "user"
        parts: list[dict[str, Any]] = []

        if isinstance(msg.content, str):
            parts.append({"text": msg.content})
        else:
            for block in msg.content:
                if isinstance(block, TextContent):
                    parts.append({"text": block.text})
                elif isinstance(block, ThinkingContent):
                    # Gemini handles thinking via thinkingConfig, skip thinking blocks
                    pass
                elif isinstance(block, ToolUseContent):
                    # Assistant's function call (snake_case per official docs)
                    fc_part: dict[str, Any] = {
                        "function_call": {
                            "name": block.name,
                            "args": block.input or {},
                        }
                    }
                    # Include thought_signature for Gemini 3 models
                    if block.thought_signature:
                        fc_part["thought_signature"] = block.thought_signature
                    parts.append(fc_part)
                    # Track tool name for later function_response
                    tool_name_map[block.id] = block.name
                elif isinstance(block, ToolResultContent):
                    # Tool result - Gemini requires the function name (snake_case)
                    tool_name = tool_name_map.get(block.tool_use_id, block.tool_use_id)
                    result_text = "".join(tb.text for tb in block.content)
                    parts.append(
                        {
                            "function_response": {
                                "name": tool_name,
                                "response": {"output": result_text},
                            }
                        }
                    )

        return {"role": role, "parts": parts}

    def _strip_additional_properties(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively strip additionalProperties from JSON schema.

        Gemini doesn't support additionalProperties in tool schemas.
        """
        result: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue  # Skip this field
            elif isinstance(value, dict):
                result[key] = self._strip_additional_properties(value)
            elif isinstance(value, list):
                result[key] = [
                    (
                        self._strip_additional_properties(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def _convert_tool_to_gemini(self, tool: Tool) -> dict[str, Any]:
        """Convert a Tool to Gemini functionDeclaration format.

        Gemini uses:
        - `name`: function name
        - `description`: function description
        - `parameters`: JSON schema (same as input_schema, but without additionalProperties)
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": self._strip_additional_properties(tool.input_schema),
        }

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """Parse Gemini response to our Response format.

        Gemini response structure:
        {
            "candidates": [{
                "content": {
                    "parts": [{"text": "..."}, {"functionCall": {...}}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20
            }
        }

        Note: HTTP errors and "error" key are already handled by _check_response.
        """
        content: list[ContentBlock] = []

        candidates = data.get("candidates", [])
        if not candidates:
            # No candidates - could be blocked by safety filters
            block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
            raise RuntimeError(f"Gemini response blocked: {block_reason}")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])

        for part in parts:
            if "text" in part:
                content.append(TextContent(text=part["text"]))
            elif "functionCall" in part or "function_call" in part:
                # Handle both camelCase and snake_case formats
                fc = part.get("functionCall") or part.get("function_call")
                # Generate a unique ID for tool tracking
                call_id = f"call_{uuid4().hex[:8]}"
                # Capture thought_signature for Gemini 3 models (both formats)
                thought_sig = part.get("thoughtSignature") or part.get(
                    "thought_signature"
                )
                content.append(
                    ToolUseContent(
                        id=call_id,
                        name=fc["name"],
                        input=fc.get("args", {}),
                        thought_signature=thought_sig,
                    )
                )
            elif "thought" in part:
                # Gemini thinking output
                content.append(ThinkingContent(thinking=part["thought"]))

        # Parse usage
        usage_data = data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=usage_data.get("promptTokenCount", 0),
            output_tokens=usage_data.get("candidatesTokenCount", 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=usage_data.get("cachedContentTokenCount", 0),
            reasoning_tokens=usage_data.get("thoughtsTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )

        # Map finish reason
        finish_reason = candidate.get("finishReason", "")
        stop_reason = _map_finish_reason(finish_reason)

        return Response(
            id=f"gemini_{uuid4().hex[:8]}",
            model=self.model,
            role=Role.ASSISTANT,
            content=content,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def send(
        self,
        messages: list[Message] | DAG,
        tools: Sequence[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> Response:
        """Send a request to the Gemini API.

        Accepts either traditional arguments OR a DAG directly.

        Args:
            messages: List of Message objects OR a DAG instance
            tools: Tool definitions (ignored if messages is DAG)
            system_prompt: System prompt (ignored if messages is DAG)

        Returns:
            Response object
        """
        # Type dispatch
        if isinstance(messages, DAG):
            dag = messages
            actual_messages = dag.to_messages()
            actual_tools: list[Tool] = list(dag._tools or [])
            dag_system_prompts = dag.head.get_system_prompts() if dag._heads else []

            messages = actual_messages
            tools = actual_tools if actual_tools else None

            if dag_system_prompts:
                system_prompt = "\n\n".join(dag_system_prompts)

        # Track tool name mapping
        tool_name_map: dict[str, str] = {}

        # Build contents array
        contents: list[dict[str, Any]] = []
        for msg in messages:
            gemini_msg = self._convert_message_to_gemini(msg, tool_name_map)
            if gemini_msg["parts"]:
                contents.append(gemini_msg)

        # Build request body
        request_body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        if system_prompt:
            request_body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if self.thinking_level and self.thinking_level.lower() != "off":
            request_body["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.thinking_level.upper()
            }

        if tools:
            request_body["tools"] = [
                {
                    "functionDeclarations": [
                        self._convert_tool_to_gemini(t) for t in tools
                    ]
                }
            ]

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        response = await self._client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=request_body,
        )

        response_json = self._check_response(response, provider="Gemini")
        return self._parse_response(response_json)


def _map_finish_reason(finish_reason: str) -> str | None:
    """Map Gemini finishReason to Claude-style stop reason."""
    reason_map = {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "safety",
        "RECITATION": "recitation",
        "OTHER": "other",
        "BLOCKLIST": "blocklist",
        "PROHIBITED_CONTENT": "prohibited_content",
        "SPII": "spii",
        "MALFORMED_FUNCTION_CALL": "tool_use",  # Indicates tool use in progress
    }
    return reason_map.get(
        finish_reason, finish_reason.lower() if finish_reason else None
    )
