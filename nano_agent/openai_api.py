"""OpenAI API client using the Responses API."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

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
)
from .tools import Tool

__all__ = ["OpenAIAPI"]


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
        parallel_tool_calls: bool = True,
        base_url: str = "https://api.openai.com/v1/responses",
    ):
        """Initialize OpenAI API client.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model to use (default: gpt-5.2-codex)
            max_tokens: Maximum tokens in response
            temperature: Temperature setting
            reasoning: Enable reasoning/thinking mode
            parallel_tool_calls: Allow model to call multiple tools in one turn
            base_url: API base URL
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
        self.parallel_tool_calls = parallel_tool_calls
        self.base_url = base_url

        # Create reusable HTTP client for connection pooling
        self._client = httpx.AsyncClient()

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
                    # Only include reasoning blocks if explicitly enabled
                    if include_reasoning:
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
                        "arguments": _serialize_arguments(block.input),
                    }
                    # Include item_id if present (required for OpenAI multi-turn)
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
                # Skip ThinkingContent - OpenAI uses reasoning separately

            # Flush remaining text
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
        """
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": True,
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
                        input=_parse_arguments(item.get("arguments", "{}")),
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
            input_items.extend(self._convert_message_to_openai(msg, self.reasoning))

        # Build request body
        request_body: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "store": False,
        }

        # Add reasoning if enabled (for o3, o3-mini, etc.)
        # Note: 'summary': 'auto' requires organization verification
        if self.reasoning:
            request_body["reasoning"] = {}
            # Include encrypted_content for multi-turn reasoning
            request_body["include"] = ["reasoning.encrypted_content"]

        # Add tools if provided
        if tools:
            request_body["tools"] = [self._convert_tool_to_openai(t) for t in tools]
            request_body["parallel_tool_calls"] = self.parallel_tool_calls
        else:
            request_body["tools"] = []

        response = await self._client.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )

        response_json = response.json()

        # Check for errors (error key exists and is not None)
        error = response_json.get("error")
        if error is not None:
            if isinstance(error, dict):
                raise RuntimeError(f"OpenAI API error: {error.get('message', error)}")
            else:
                raise RuntimeError(f"OpenAI API error: {error}")

        return self._parse_response(response_json)


def _serialize_arguments(input_dict: dict[str, Any] | None) -> str:
    """Serialize tool input to JSON string for OpenAI."""
    import json

    if input_dict is None:
        return "{}"
    return json.dumps(input_dict)


def _parse_arguments(arguments: str) -> dict[str, Any]:
    """Parse OpenAI function arguments JSON string to dict."""
    import json

    try:
        result = json.loads(arguments)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _map_status_to_stop_reason(status: str) -> str | None:
    """Map OpenAI response status to Claude-style stop reason."""
    status_map = {
        "completed": "end_turn",
        "failed": "error",
        "incomplete": "max_tokens",
        "in_progress": None,
    }
    return status_map.get(status, status)
