"""Codex (ChatGPT OAuth) API client for the Codex Responses endpoint."""

from __future__ import annotations

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
from .codex_auth import get_codex_access_token

__all__ = ["CodexAPI"]


class CodexAPI:
    """Codex Responses API client using ChatGPT OAuth access tokens."""

    def __init__(
        self,
        auth_token: str | None = None,
        model: str = "gpt-5.2-codex",
        base_url: str = "https://chatgpt.com/backend-api/codex/responses",
        parallel_tool_calls: bool = True,
        reasoning: bool = True,
    ) -> None:
        resolved = auth_token or get_codex_access_token()
        if not resolved:
            raise ValueError(
                "Codex OAuth token required. Ensure Codex is configured to store "
                "auth in file mode (~/.codex/auth.json) or pass auth_token."
            )

        self.auth_token = resolved
        self.model = model
        self.base_url = base_url
        self.parallel_tool_calls = parallel_tool_calls
        self.reasoning = reasoning
        self._client = httpx.AsyncClient(timeout=60.0)

    def __repr__(self) -> str:
        token_preview = (
            self.auth_token[:15] + "..."
            if len(self.auth_token) > 15
            else self.auth_token
        )
        return (
            "CodexAPI(\n"
            f"  model={self.model!r},\n"
            f"  base_url={self.base_url!r},\n"
            f"  token={token_preview!r}\n"
            ")"
        )

    def _convert_message_to_codex(self, msg: Message) -> list[dict[str, Any]]:
        """Convert Message to Codex input items (Responses API format)."""
        items: list[dict[str, Any]] = []
        text_type = "input_text" if msg.role == Role.USER else "output_text"

        if isinstance(msg.content, str):
            items.append(
                {
                    "role": msg.role.value,
                    "content": [{"type": text_type, "text": msg.content}],
                }
            )
        else:
            text_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingContent):
                    continue
                elif isinstance(block, ToolUseContent):
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
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": _serialize_arguments(block.input),
                        }
                    )
                elif isinstance(block, ToolResultContent):
                    result_text = "".join(tb.text for tb in block.content)
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": result_text,
                        }
                    )

            if text_parts:
                items.append(
                    {
                        "role": msg.role.value,
                        "content": [{"type": text_type, "text": "\n".join(text_parts)}],
                    }
                )

        return items

    def _convert_tool_to_codex(self, tool: Tool) -> dict[str, Any]:
        schema = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": True,
        }
        params = schema.get("parameters")
        if isinstance(params, dict):
            self._force_required_all(params)
        return schema

    def _force_required_all(self, schema: dict[str, Any]) -> None:
        """Ensure required includes all properties (Codex strict schema)."""
        if schema.get("type") == "object":
            props = schema.get("properties")
            if isinstance(props, dict):
                schema["required"] = list(props.keys())
                for prop in props.values():
                    if isinstance(prop, dict):
                        self._force_required_all(prop)
            additional = schema.get("additionalProperties")
            if isinstance(additional, dict):
                self._force_required_all(additional)
        elif schema.get("type") == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                self._force_required_all(items)

    async def send(
        self,
        messages: list[Message] | DAG,
        tools: Sequence[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> Response:
        if isinstance(messages, DAG):
            dag = messages
            actual_messages = dag.to_messages()
            actual_tools: list[Tool] = list(dag._tools or [])
            dag_system_prompts = dag.head.get_system_prompts() if dag._heads else []

            messages = actual_messages
            tools = actual_tools if actual_tools else None

            if dag_system_prompts:
                system_prompt = "\n\n".join(dag_system_prompts)

        # Codex endpoint requires the `instructions` field.
        # Note: `instructions` does not participate in OpenAI prefix caching.
        # Caching with cached_tokens only works on the standard OpenAI API
        # (api.openai.com) using developer messages in the input array.
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            input_items.extend(self._convert_message_to_codex(msg))

        request_body: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt or "You are a helpful assistant.",
            "input": input_items,
            "store": False,
            "stream": True,
        }

        # Add reasoning if enabled
        if self.reasoning:
            request_body["reasoning"] = {"effort": "high", "summary": "detailed"}
            request_body["include"] = ["reasoning.encrypted_content"]

        if tools:
            request_body["tools"] = [self._convert_tool_to_codex(t) for t in tools]
            request_body["parallel_tool_calls"] = self.parallel_tool_calls
        else:
            request_body["tools"] = []

        response_data = await self._stream_response(request_body)
        if response_data is None:
            raise RuntimeError("No response received from Codex endpoint.")
        return self._parse_response(response_data)

    async def _stream_response(
        self, request_body: dict[str, Any]
    ) -> dict[str, Any] | None:
        last_response: dict[str, Any] | None = None
        async with self._client.stream(
            "POST",
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
            json=request_body,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                snippet = body.decode("utf-8", errors="ignore")[:400]
                raise RuntimeError(
                    f"Codex API HTTP {resp.status_code}: {snippet or 'empty response'}"
                )

            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                data = await resp.aread()
                text = data.decode("utf-8", errors="ignore")
                if os.environ.get("NANO_CLI_DEBUG_HTTP") == "1":
                    headers_preview = dict(resp.headers)
                    print(
                        f"\n[debug] Codex non-SSE content-type: {content_type!r}, "
                        f"headers={headers_preview}\n"
                    )
                if "data:" in text:
                    parsed = self._parse_sse_text(text)
                    if parsed is not None:
                        return parsed
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    snippet = text[:400]
                    raise RuntimeError(
                        f"Codex API non-stream response: {snippet or 'empty response'}"
                    )
                if isinstance(payload, dict) and payload.get("error"):
                    raise RuntimeError(f"Codex API error: {payload['error']}")
                if isinstance(payload, dict) and payload.get("response"):
                    return payload.get("response")
                return payload if isinstance(payload, dict) else None

            last_event_type: str | None = None
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                last_event_type = event.get("type")
                if event.get("type") == "response.completed":
                    last_response = event.get("response")
                if event.get("error"):
                    raise RuntimeError(f"Codex API error: {event['error']}")
            if last_response is None and last_event_type:
                raise RuntimeError(
                    f"No response received from Codex endpoint (last event: {last_event_type})."
                )
        return last_response

    def _parse_sse_text(self, text: str) -> dict[str, Any] | None:
        """Parse SSE-formatted text and return last response if present."""
        last_response: dict[str, Any] | None = None
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "response.completed":
                last_response = event.get("response")
            if event.get("error"):
                raise RuntimeError(f"Codex API error: {event['error']}")
        return last_response

    def _parse_response(self, data: dict[str, Any]) -> Response:
        content: list[ContentBlock] = []

        for item in data.get("output", []):
            item_type = item.get("type", "")

            if item_type == "reasoning":
                summary = item.get("summary", [])
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
                for nested in item.get("content", []):
                    nested_type = nested.get("type", "")
                    if nested_type == "output_text":
                        content.append(TextContent(text=nested.get("text", "")))
                    elif nested_type == "refusal":
                        content.append(TextContent(text=nested.get("refusal", "")))
            elif item_type == "function_call":
                content.append(
                    ToolUseContent(
                        id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        input=_parse_arguments(item.get("arguments", "{}")),
                        item_id=item.get("id"),
                    )
                )

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


def _serialize_arguments(input_dict: dict[str, Any] | None) -> str:
    if input_dict is None:
        return "{}"
    return json.dumps(input_dict)


def _parse_arguments(arguments: str) -> dict[str, Any]:
    try:
        result = json.loads(arguments)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _map_status_to_stop_reason(status: str) -> str | None:
    status_map = {
        "completed": "end_turn",
        "failed": "error",
        "incomplete": "max_tokens",
        "in_progress": None,
    }
    return status_map.get(status, status)
