"""OpenAI API client using the Responses API."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence
from typing import Any

import httpx

from ..dag import DAG
from ..data_structures import (
    ContentBlock,
    ImageContent,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from ..tools import Tool
from .base import (
    DEFAULT_STREAM_IDLE_TIMEOUT,
    ReasoningEffort,
    ReasoningSummary,
    build_httpx_timeout,
    build_reasoning_block,
    consume_responses_sse_stream,
    force_required_all,
    parse_tool_arguments,
    responses_tool_result_item,
    responses_user_image_item,
    serialize_tool_arguments,
    unpack_dag_or_args,
)

__all__ = ["OpenAIAPI"]

_OPENAI_API_SOURCE = "OpenAI API"


class OpenAIAPI:
    """OpenAI API client using the Responses API.

    This client follows the same patterns as ClaudeAPI, reusing existing
    data structures and providing a consistent interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.2-codex",
        max_tokens: int = 4096,
        temperature: float = 1.0,
        reasoning: bool = True,
        reasoning_effort: ReasoningEffort = "high",
        reasoning_summary: ReasoningSummary | None = None,
        parallel_tool_calls: bool = True,
        base_url: str = "https://api.openai.com/v1/responses",
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
    ):
        """Initialize OpenAI API client.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model to use (default: gpt-5.2-codex)
            max_tokens: Maximum tokens in response
            temperature: Temperature setting
            reasoning: Enable reasoning/thinking mode
            reasoning_effort: ``low``/``medium``/``high`` — passed through to
                the model when ``reasoning`` is enabled.
            reasoning_summary: ``concise``/``detailed``/``auto`` or ``None``.
                Defaults to ``None`` because ``"auto"`` requires organization
                verification on the public API; opt in explicitly.
            parallel_tool_calls: Allow model to call multiple tools in one turn
            base_url: API base URL
            stream_idle_timeout: Per-chunk read timeout for the SSE stream.
                Defaults to 300s to match codex-rs.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key or set OPENAI_API_KEY env var."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.parallel_tool_calls = parallel_tool_calls
        self.base_url = base_url
        self.stream_idle_timeout = stream_idle_timeout
        self._client = httpx.AsyncClient(
            timeout=build_httpx_timeout(stream_idle_timeout)
        )
        # Stable cache key for the lifetime of this client; mirrors codex-rs's
        # use of conversation_id so prefix caching has a consistent bucket.
        self._session_id = str(uuid.uuid4())

    def __repr__(self) -> str:
        """Return a clean representation of the API client configuration."""
        token_preview = (
            self.api_key[:15] + "..."
            if self.api_key and len(self.api_key) > 15
            else self.api_key
        )
        return (
            f"OpenAIAPI(\n"
            f"  model={self.model!r},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  temperature={self.temperature},\n"
            f"  base_url={self.base_url!r},\n"
            f"  token={token_preview!r}\n"
            f")"
        )

    def _convert_message_to_openai(
        self, msg: Message, include_reasoning: bool = False
    ) -> list[dict[str, Any]]:
        """Convert a Message to OpenAI input format.

        OpenAI Responses API format:
        - User messages: role + content array with input_text items
        - Assistant messages: role + content array with output_text items
        - function_call for tool use
        - function_call_output for tool results
        - reasoning blocks are passed through only if include_reasoning=True
        """
        items: list[dict[str, Any]] = []

        # Determine text type based on role
        text_type = "input_text" if msg.role == Role.USER else "output_text"

        if isinstance(msg.content, str):
            # Simple string content
            items.append(
                {
                    "role": msg.role.value,
                    "content": [{"type": text_type, "text": msg.content}],
                }
            )
        else:
            # List of content blocks - collect text and handle tools separately
            text_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingContent):
                    # Only replay reasoning blocks that have encrypted_content
                    # (Responses API form). Skip Claude-style ``thinking`` blocks
                    # since the Responses API rejects that shape.
                    if include_reasoning and block.encrypted_content:
                        # Flush any accumulated text first
                        if text_parts:
                            items.append(
                                {
                                    "role": msg.role.value,
                                    "content": [
                                        {
                                            "type": text_type,
                                            "text": "\n".join(text_parts),
                                        }
                                    ],
                                }
                            )
                            text_parts = []
                        # Pass reasoning block through for multi-turn conversations
                        items.append(dict(block.to_dict()))
                elif isinstance(block, ToolUseContent):
                    # Flush any accumulated text first
                    if text_parts:
                        items.append(
                            {
                                "role": msg.role.value,
                                "content": [
                                    {"type": text_type, "text": "\n".join(text_parts)}
                                ],
                            }
                        )
                        text_parts = []
                    # Assistant's tool call
                    func_call: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": serialize_tool_arguments(block.input),
                    }
                    # Include item_id if present (required for OpenAI multi-turn)
                    if block.item_id:
                        func_call["id"] = block.item_id
                    items.append(func_call)
                elif isinstance(block, ToolResultContent):
                    items.append(responses_tool_result_item(block))
                elif isinstance(block, ImageContent):
                    if text_parts:
                        items.append(
                            {
                                "role": msg.role.value,
                                "content": [
                                    {"type": text_type, "text": "\n".join(text_parts)}
                                ],
                            }
                        )
                        text_parts = []
                    items.append(responses_user_image_item(msg, block))

            if text_parts:
                items.append(
                    {
                        "role": msg.role.value,
                        "content": [{"type": text_type, "text": "\n".join(text_parts)}],
                    }
                )

        return items

    def _convert_tool_to_openai(self, tool: Tool) -> dict[str, Any]:
        """Convert a Tool to OpenAI function format.

        OpenAI uses: {"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": true}

        Strict mode requires every property to also appear in ``required``,
        so we rewrite the schema when ``strict`` is True.
        """
        strict = bool(getattr(tool, "strict", True))
        parameters = tool.input_schema
        if strict and isinstance(parameters, dict):
            parameters = force_required_all(parameters)
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
            "strict": strict,
        }

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """Parse OpenAI response to our Response format.

        OpenAI Responses API returns:
        - output: list of content items (output_text, function_call, reasoning, etc.)
        - usage: {input_tokens, output_tokens, total_tokens}
        """
        content: list[ContentBlock] = []

        for item in data.get("output", []):
            item_type = item.get("type", "")

            if item_type == "reasoning":
                # Capture full reasoning block for multi-turn conversations
                # This includes encrypted_content needed for continuation
                summary = item.get("summary", [])
                # Extract thinking text from summary
                thinking_text = ""
                if summary:
                    thinking_text = " ".join(
                        s.get("text", "")
                        for s in summary
                        if s.get("type") == "summary_text"
                    )
                content.append(
                    ThinkingContent(
                        thinking=thinking_text,
                        id=item.get("id", ""),
                        encrypted_content=item.get("encrypted_content", ""),
                        summary=tuple(summary),
                    )
                )
            elif item_type == "message":
                # Handle message wrapper - extract nested content
                for nested in item.get("content", []):
                    nested_type = nested.get("type", "")
                    if nested_type == "output_text":
                        content.append(TextContent(text=nested.get("text", "")))
                    elif nested_type == "refusal":
                        content.append(TextContent(text=nested.get("refusal", "")))
            elif item_type == "function_call":
                # Tool call from assistant
                content.append(
                    ToolUseContent(
                        id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        input=parse_tool_arguments(item.get("arguments", "{}")),
                        item_id=item.get("id"),  # Store OpenAI item id (fc_...)
                    )
                )

        # Parse usage
        usage_data = data.get("usage", {})
        input_details = usage_data.get("input_tokens_details", {})
        output_details = usage_data.get("output_tokens_details", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            reasoning_tokens=output_details.get("reasoning_tokens", 0),
            cached_tokens=input_details.get("cached_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Determine stop reason from status
        status = data.get("status", "")
        stop_reason = _map_status_to_stop_reason(status)

        return Response(
            id=data.get("id", ""),
            model=data.get("model", self.model),
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
        """Send a request to the OpenAI API.

        Accepts either traditional arguments OR a DAG directly.

        Args:
            messages: List of Message objects OR a DAG instance
            tools: Tool definitions (ignored if messages is DAG)
            system_prompt: System prompt (ignored if messages is DAG)

        Returns:
            Response object
        """
        messages, tools, system_prompt = unpack_dag_or_args(
            messages, tools, system_prompt
        )

        # Put system prompt as developer message in input for caching support.
        # The `instructions` parameter does not participate in OpenAI prefix
        # caching — only the `input` array prefix is cached.
        input_items: list[dict[str, Any]] = []
        if system_prompt:
            input_items.append(
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )

        # Add conversation messages
        for msg in messages:
            input_items.extend(self._convert_message_to_openai(msg, self.reasoning))

        # Build request body
        request_body: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "store": False,
            "stream": True,
            "prompt_cache_key": self._session_id,
        }

        if self.reasoning:
            request_body["reasoning"] = build_reasoning_block(
                self.reasoning_effort, self.reasoning_summary
            )
            request_body["include"] = ["reasoning.encrypted_content"]

        # Add tools if provided
        if tools:
            request_body["tools"] = [self._convert_tool_to_openai(t) for t in tools]
            request_body["tool_choice"] = "auto"
            request_body["parallel_tool_calls"] = self.parallel_tool_calls
        else:
            request_body["tools"] = []

        response_data = await self._stream_response(request_body)
        if response_data is None:
            raise RuntimeError("No response received from OpenAI endpoint.")
        return self._parse_response(response_data)

    async def _stream_response(
        self, request_body: dict[str, Any]
    ) -> dict[str, Any] | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        async with self._client.stream(
            "POST",
            self.base_url,
            headers=headers,
            json=request_body,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                _raise_openai_http_error(resp.status_code, body)

            last_response, last_event_type = await consume_responses_sse_stream(
                resp.aiter_lines(), source=_OPENAI_API_SOURCE
            )
            if last_response is None and last_event_type:
                raise RuntimeError(
                    f"No response received from OpenAI endpoint (last event: {last_event_type})."
                )
        return last_response


def _raise_openai_http_error(status_code: int, body: bytes) -> None:
    """Raise ``RuntimeError`` for an OpenAI HTTP error response.

    Tries to extract ``error.message`` from a JSON body; falls back to a
    snippet of the raw response so transport-level failures are still
    diagnosable.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        snippet = body.decode("utf-8", errors="ignore")[:400]
        raise RuntimeError(
            f"OpenAI API HTTP {status_code}: {snippet or 'empty response'}"
        )
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            raise RuntimeError(f"OpenAI API error: {error.get('message', error)}")
        if error is not None:
            raise RuntimeError(f"OpenAI API error: {error}")
    raise RuntimeError(f"OpenAI API HTTP {status_code}: {payload!r}")


def _map_status_to_stop_reason(status: str) -> str | None:
    """Map OpenAI response status to Claude-style stop reason."""
    status_map = {
        "completed": "end_turn",
        "failed": "error",
        "incomplete": "max_tokens",
        "in_progress": None,
    }
    return status_map.get(status, status)
