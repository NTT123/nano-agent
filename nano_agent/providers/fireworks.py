"""Fireworks AI API client using the Responses API."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from typing import Any

import httpx

from ..dag import DAG
from ..data_structures import (
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
from ..tools import Tool
from .base import APIError

__all__ = ["FireworksAPI"]


class FireworksAPI:
    """Fireworks AI API client using the Responses API.

    This client follows the same patterns as OpenAI API, as Fireworks uses
    a compatible interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "accounts/fireworks/models/kimi-k2p5",
        max_tokens: int = 4096,
        temperature: float = 1.0,
        parallel_tool_calls: bool = True,
        base_url: str = "https://api.fireworks.ai/inference/v1/responses",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        debug: bool = True,
        session_id: str = "nano-agent-session",
        reasoning_effort: str | None = None,
    ):
        """Initialize Fireworks API client.

        Args:
            api_key: Fireworks API key. Falls back to FIREWORKS_API_KEY env var.
            model: Model to use (default: accounts/fireworks/models/kimi-k2p5)
            max_tokens: Maximum tokens in response
            temperature: Temperature setting
            parallel_tool_calls: Allow model to call multiple tools in one turn
            base_url: API base URL
            max_retries: Maximum number of retries on failure (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            debug: Print request/response details on error (default: True)
            session_id: Session ID for session affinity (default: "nano-agent-session")
            reasoning_effort: Reasoning effort level ("low", "medium", "high") or None to disable
        """
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fireworks API key required. Pass api_key or set FIREWORKS_API_KEY env var."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.parallel_tool_calls = parallel_tool_calls
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.debug = debug
        self.reasoning_effort = reasoning_effort

        # Session ID for session affinity (enables caching)
        self.session_id = session_id

        # Create reusable HTTP client for connection pooling
        self._client = httpx.AsyncClient(timeout=120.0)

    def __repr__(self) -> str:
        """Return a clean representation of the API client configuration."""
        token_preview = (
            self.api_key[:15] + "..."
            if self.api_key and len(self.api_key) > 15
            else self.api_key
        )
        return (
            f"FireworksAPI(\n"
            f"  model={self.model!r},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  temperature={self.temperature},\n"
            f"  base_url={self.base_url!r},\n"
            f"  token={token_preview!r},\n"
            f"  session_id={self.session_id!r}\n"
            f")"
        )

    def _convert_message_to_fireworks(self, msg: Message) -> list[dict[str, Any]]:
        """Convert a Message to Fireworks input format.

        Fireworks Responses API format (similar to OpenAI):
        - User messages: role + content array with input_text items
        - Assistant messages: role + content array with output_text items
        - function_call for tool use
        - function_call_output for tool results
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
                    # Skip thinking content - Fireworks doesn't use reasoning separately
                    pass
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
                        "arguments": _serialize_arguments(block.input),
                    }
                    # Include item_id if present
                    if block.item_id:
                        func_call["id"] = block.item_id
                    items.append(func_call)
                elif isinstance(block, ToolResultContent):
                    # Tool result
                    result_text = "".join(tb.text for tb in block.content)
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": result_text,
                        }
                    )

            # Flush remaining text
            if text_parts:
                items.append(
                    {
                        "role": msg.role.value,
                        "content": [{"type": text_type, "text": "\n".join(text_parts)}],
                    }
                )

        return items

    def _convert_tool_to_fireworks(self, tool: Tool) -> dict[str, Any]:
        """Convert a Tool to Fireworks function format.

        Fireworks uses: {"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": true}
        """
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": True,
        }

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """Parse Fireworks response to our Response format.

        Fireworks Responses API returns:
        - output: list of message items with content arrays
        - usage: {prompt_tokens, completion_tokens, total_tokens}
        - reasoning: {content: str, type: "reasoning"} (if reasoning enabled)
        """
        content: list[ContentBlock] = []

        # Add reasoning/thinking content if present
        reasoning = data.get("reasoning")
        if reasoning and isinstance(reasoning, dict):
            reasoning_text = reasoning.get("content", "")
            if reasoning_text:
                content.append(ThinkingContent(thinking=reasoning_text))

        for item in data.get("output", []):
            item_type = item.get("type", "")

            if item_type == "message":
                # Handle message wrapper - extract nested content
                # Only process assistant messages
                if item.get("role") == "assistant":
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
                        input=_parse_arguments(item.get("arguments", "{}")),
                        item_id=item.get("id"),
                    )
                )

        # Parse usage
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            # Fireworks doesn't have cache tokens, leave as 0
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
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
        """Send a request to the Fireworks API.

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

        # Build input array - start with system prompt if provided
        input_items: list[dict[str, Any]] = []
        if system_prompt:
            input_items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )

        # Add conversation messages
        for msg in messages:
            input_items.extend(self._convert_message_to_fireworks(msg))

        # Build request body
        request_body: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "store": False,
        }

        # Add reasoning if enabled
        if self.reasoning_effort:
            request_body["reasoning"] = {"effort": self.reasoning_effort}

        # Add tools if provided
        if tools:
            request_body["tools"] = [self._convert_tool_to_fireworks(t) for t in tools]
            request_body["parallel_tool_calls"] = self.parallel_tool_calls
        else:
            request_body["tools"] = []

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "x-session-affinity": self.session_id,
                    },
                    json=request_body,
                )

                # Handle empty or non-JSON responses
                try:
                    response_json = response.json()
                except json.JSONDecodeError as e:
                    response_text = response.text[:1000] if response.text else "(empty)"
                    error_msg = self._format_error(
                        response.status_code,
                        {"error": f"JSON decode error: {e}", "response": response_text},
                        request_body,
                    )
                    # Retry on server errors
                    if response.status_code >= 500 or response.status_code == 429:
                        last_error = APIError(
                            message=error_msg,
                            status_code=response.status_code,
                            provider="Fireworks",
                        )
                        if attempt < self.max_retries:
                            delay = self.retry_delay * (2**attempt)
                            await asyncio.sleep(delay)
                            continue
                    raise APIError(
                        message=error_msg,
                        status_code=response.status_code,
                        provider="Fireworks",
                    )

                # Check HTTP status code
                if response.status_code != 200:
                    error_msg = self._format_error(
                        response.status_code, response_json, request_body
                    )
                    # Retry on 5xx errors or rate limits (429)
                    if response.status_code >= 500 or response.status_code == 429:
                        last_error = APIError(
                            message=error_msg,
                            status_code=response.status_code,
                            provider="Fireworks",
                        )
                        if attempt < self.max_retries:
                            delay = self.retry_delay * (2**attempt)
                            await asyncio.sleep(delay)
                            continue
                    raise APIError(
                        message=error_msg,
                        status_code=response.status_code,
                        provider="Fireworks",
                    )

                # Check for errors in response body
                error = response_json.get("error")
                if error is not None:
                    error_msg = self._format_error(
                        response.status_code, response_json, request_body
                    )
                    raise APIError(message=error_msg, provider="Fireworks")

                return self._parse_response(response_json)

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                error_msg = self._format_error(None, {"error": str(e)}, request_body)
                raise APIError(message=f"Timeout: {error_msg}", provider="Fireworks")

            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                error_msg = self._format_error(None, {"error": str(e)}, request_body)
                raise APIError(
                    message=f"Request error: {error_msg}", provider="Fireworks"
                )

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise APIError(message="Unknown error after retries", provider="Fireworks")

    def _format_error(
        self,
        status_code: int | None,
        response_json: dict[str, Any],
        request_body: dict[str, Any],
    ) -> str:
        """Format error message with request details.

        Args:
            status_code: HTTP status code (if available)
            response_json: Response JSON body
            request_body: Request body that was sent

        Returns:
            Formatted error message
        """
        # Extract error message
        error = response_json.get("error", {})
        if isinstance(error, dict):
            error_msg = error.get("message", str(error))
        else:
            error_msg = str(error) if error else "Unknown error"

        # Build error details
        parts = []
        if status_code:
            parts.append(f"HTTP {status_code}")
        parts.append(error_msg)

        # Add request details if debug mode or always for visibility
        if self.debug:
            # Truncate input for readability
            request_summary = {
                "model": request_body.get("model"),
                "max_output_tokens": request_body.get("max_output_tokens"),
                "tools_count": len(request_body.get("tools", [])),
                "input_items_count": len(request_body.get("input", [])),
            }
            parts.append(f"\nRequest summary: {json.dumps(request_summary, indent=2)}")
            parts.append(f"\nFull response: {json.dumps(response_json, indent=2)}")

        return " - ".join(parts[:2]) + "".join(parts[2:])


def _serialize_arguments(input_dict: dict[str, Any] | None) -> str:
    """Serialize tool input to JSON string for Fireworks."""
    import json

    if input_dict is None:
        return "{}"
    return json.dumps(input_dict)


def _parse_arguments(arguments: str) -> dict[str, Any]:
    """Parse Fireworks function arguments JSON string to dict."""
    import json

    try:
        result = json.loads(arguments)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _map_status_to_stop_reason(status: str) -> str | None:
    """Map Fireworks response status to Claude-style stop reason."""
    status_map = {
        "completed": "end_turn",
        "failed": "error",
        "incomplete": "max_tokens",
        "in_progress": None,
    }
    return status_map.get(status, status)
