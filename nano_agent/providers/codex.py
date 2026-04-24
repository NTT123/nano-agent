"""Codex (ChatGPT OAuth) API client for the Codex Responses endpoint."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
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
from .base import responses_tool_result_item, responses_user_image_item
from .codex_auth import (
    DEFAULT_CODEX_AUTH_PATH,
    REFRESH_AFTER_DAYS,
    get_codex_access_token,
    get_codex_account_id,
    load_codex_auth,
    maybe_refresh,
    parse_iso,
)

__all__ = ["CodexAPI"]


class CodexAPI:
    """Codex Responses API client using ChatGPT OAuth access tokens.

    Two modes:
      - Static: pass ``auth_token``. The token is used as-is; the caller is
        responsible for refresh. ``account_id`` may optionally be passed.
      - File-backed: pass ``auth_file`` (default: ``~/.codex/auth.json``). On
        each ``send()`` the file is reloaded and access_token is auto-refreshed
        when older than ``REFRESH_AFTER_DAYS``. This is the recommended mode
        for long-running processes.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        model: str = "gpt-5.5",
        base_url: str = "https://chatgpt.com/backend-api/codex/responses",
        parallel_tool_calls: bool = True,
        reasoning: bool = True,
        reasoning_effort: str = "high",
        auth_file: Path | str | None = None,
        account_id: str | None = None,
    ) -> None:
        self._auth_file: Path | None = None
        if (
            auth_token is None
            and auth_file is None
            and DEFAULT_CODEX_AUTH_PATH.exists()
        ):
            auth_file = DEFAULT_CODEX_AUTH_PATH

        self._last_refresh_at: datetime | None = None
        if auth_file is not None:
            self._auth_file = Path(auth_file)
            data = load_codex_auth(self._auth_file)
            if not data:
                raise ValueError(
                    f"Codex auth file missing or invalid: {self._auth_file}. "
                    "Run `codex login` or nano_agent.providers.codex_login first."
                )
            tokens = data.get("tokens") or {}
            resolved = tokens.get("access_token")
            if not resolved:
                raise ValueError(
                    f"No access_token in {self._auth_file}; re-login required."
                )
            self.account_id = account_id or tokens.get("account_id")
            self._last_refresh_at = parse_iso(data.get("last_refresh"))
        else:
            resolved = auth_token or get_codex_access_token()
            if not resolved:
                raise ValueError(
                    "Codex OAuth token required. Pass auth_token, auth_file, or "
                    "run `codex login` to populate ~/.codex/auth.json."
                )
            self.account_id = account_id or get_codex_account_id()

        self.auth_token = resolved
        self.model = model
        self.base_url = base_url
        self.parallel_tool_calls = parallel_tool_calls
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(timeout=60.0)
        # One session_id per client, mirroring the Codex CLI; a fresh UUID per
        # request would make every call look like a new session to the backend.
        self._session_id = str(uuid.uuid4())
        # Serializes refresh across concurrent send() calls so we don't double-
        # spend the rotating refresh_token when multiple workers hit the 8-day
        # threshold simultaneously.
        self._refresh_lock = asyncio.Lock()

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

    def _refresh_fresh_enough(self) -> bool:
        """True if the in-memory last_refresh timestamp is within the window."""
        if self._last_refresh_at is None:
            return False
        age_days = (
            datetime.now(timezone.utc) - self._last_refresh_at
        ).total_seconds() / 86400
        return age_days < REFRESH_AFTER_DAYS

    async def _refresh_if_file_backed(self) -> None:
        """If auth_file is set, reload + maybe-refresh tokens from disk.

        Short-circuits without disk I/O or locking when the in-memory
        ``_last_refresh_at`` is within the refresh window. This keeps send()
        off disk on the hot path; the lock is only taken when we actually
        need to re-check / rotate tokens.
        """
        if self._auth_file is None or self._refresh_fresh_enough():
            return
        async with self._refresh_lock:
            # Another coroutine may have refreshed while we waited.
            if self._refresh_fresh_enough():
                return
            data = load_codex_auth(self._auth_file)
            if not data:
                return
            data = await maybe_refresh(data, self._auth_file, self._client)
            tokens = data.get("tokens") or {}
            if tokens.get("access_token"):
                self.auth_token = tokens["access_token"]
            if tokens.get("account_id"):
                self.account_id = tokens["account_id"]
            self._last_refresh_at = parse_iso(data.get("last_refresh"))

    async def send(
        self,
        messages: list[Message] | DAG,
        tools: Sequence[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> Response:
        await self._refresh_if_file_backed()
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
            request_body["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "detailed",
            }
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
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "originator": "codex_cli_rs",
            "session_id": self._session_id,
            "OpenAI-Beta": "responses=experimental",
        }
        if self.account_id:
            headers["chatgpt-account-id"] = self.account_id
        async with self._client.stream(
            "POST",
            self.base_url,
            headers=headers,
            json=request_body,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                snippet = body.decode("utf-8", errors="ignore")[:400]
                raise RuntimeError(
                    f"Codex API HTTP {resp.status_code}: {snippet or 'empty response'}"
                )

            content_type = resp.headers.get("content-type") or ""
            # The chatgpt.com/backend-api/codex/responses endpoint omits
            # content-type on streamed responses; treat empty as SSE.
            if content_type and "text/event-stream" not in content_type:
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
            # The chatgpt.com/backend-api/codex/responses endpoint streams output
            # items via response.output_item.done but returns an empty `output`
            # array in the final response.completed event. Collect items from
            # output_item.done and attach them so _parse_response sees them.
            collected_items: list[dict[str, Any]] = []
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
                etype = event.get("type")
                last_event_type = etype
                if etype == "response.completed":
                    last_response = event.get("response")
                elif etype == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") in ("reasoning", "message", "function_call"):
                        collected_items.append(item)
                if event.get("error"):
                    raise RuntimeError(f"Codex API error: {event['error']}")

            if last_response is not None and not last_response.get("output"):
                if collected_items:
                    last_response = {**last_response, "output": collected_items}
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
