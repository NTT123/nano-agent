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
    CompactionContent,
    ContentBlock,
    ImageContent,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)
from ..tools import Tool
from .base import (
    DEFAULT_STREAM_IDLE_TIMEOUT,
    ContextWindowExceededError,
    ReasoningEffort,
    build_httpx_timeout,
    build_reasoning_block,
    consume_responses_sse_stream,
    flush_text_parts,
    force_required_all,
    is_context_window_error_payload,
    map_responses_status_to_stop_reason,
    parse_responses_sse_text,
    parse_tool_arguments,
    responses_tool_result_item,
    responses_usage_from_dict,
    responses_user_image_item,
    serialize_tool_arguments,
    unpack_dag_or_args,
)
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

_CODEX_INSTALLATION_ID_FILENAME = "installation_id"
_CODEX_INSTALLATION_ID_METADATA_KEY = "x-codex-installation-id"
_CODEX_INSTALLATION_ID_HEADER = "x-codex-installation-id"
_CODEX_TURN_STATE_HEADER = "x-codex-turn-state"
_CODEX_API_SOURCE = "Codex API"
_COMPACT_PATH_SUFFIX = "/compact"
# codex-rs uses MODELS_REFRESH_TIMEOUT = 5s (model-provider/src/models_endpoint.rs).
_MODELS_FETCH_TIMEOUT = 5.0
# codex-rs sends `?client_version=MAJOR.MINOR.PATCH` from its CARGO_PKG_VERSION
# (codex-api/src/endpoint/models.rs::append_client_version_query); the backend
# now requires the query param. Bump alongside the codex-rs version we mirror.
_CODEX_CLIENT_VERSION = "0.125.0"


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
        reasoning_effort: ReasoningEffort = "high",
        auth_file: Path | str | None = None,
        account_id: str | None = None,
        codex_home: Path | str | None = None,
        installation_id: str | None = None,
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
        context_window: int | None = None,
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
        self.stream_idle_timeout = stream_idle_timeout
        self.context_window = context_window
        self._client = httpx.AsyncClient(
            timeout=build_httpx_timeout(stream_idle_timeout)
        )
        self._codex_home = _resolve_codex_home(codex_home, auth_file)
        self._installation_id = installation_id
        self._turn_state: str | None = None
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

    @property
    def auto_compact_token_limit(self) -> int | None:
        """Token threshold above which ``executor.run`` auto-compacts the DAG.

        Mirrors codex-rs's ``ModelInfo.auto_compact_token_limit()``: 90 % of
        the configured context window, or ``None`` when the caller did not
        supply ``context_window`` (auto-compact disabled).
        """
        if self.context_window is None:
            return None
        return (self.context_window * 9) // 10

    def _get_installation_id(self) -> str:
        if self._installation_id is None:
            self._installation_id = _resolve_codex_installation_id(self._codex_home)
        return self._installation_id

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
            return items

        text_parts: list[str] = []
        for block in msg.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                # Only replay reasoning blocks that have encrypted_content
                # (Responses API form). Skip Claude-style ``thinking`` blocks
                # since the Responses API rejects that shape.
                if self.reasoning and block.encrypted_content:
                    flush_text_parts(items, msg.role.value, text_type, text_parts)
                    items.append(dict(block.to_dict()))
            elif isinstance(block, ToolUseContent):
                flush_text_parts(items, msg.role.value, text_type, text_parts)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": serialize_tool_arguments(block.input),
                    }
                )
            elif isinstance(block, ToolResultContent):
                items.append(responses_tool_result_item(block))
            elif isinstance(block, ImageContent):
                flush_text_parts(items, msg.role.value, text_type, text_parts)
                items.append(responses_user_image_item(msg, block))
            elif isinstance(block, CompactionContent):
                flush_text_parts(items, msg.role.value, text_type, text_parts)
                items.append(dict(block.to_dict()))

        flush_text_parts(items, msg.role.value, text_type, text_parts)
        return items

    def _convert_tool_to_codex(self, tool: Tool) -> dict[str, Any]:
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
        messages, tools, system_prompt = unpack_dag_or_args(
            messages, tools, system_prompt
        )

        # Codex endpoint requires the `instructions` field. The session-scoped
        # prompt_cache_key below mirrors the Codex CLI's cache-affinity hint.
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            input_items.extend(self._convert_message_to_codex(msg))

        request_body: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt or "You are a helpful assistant.",
            "input": input_items,
            "tool_choice": "auto",
            "parallel_tool_calls": self.parallel_tool_calls,
            "store": False,
            "stream": True,
            "prompt_cache_key": self._session_id,
            "client_metadata": {
                _CODEX_INSTALLATION_ID_METADATA_KEY: self._get_installation_id(),
            },
        }

        if self.reasoning:
            request_body["reasoning"] = build_reasoning_block(
                self.reasoning_effort, summary="detailed"
            )
            request_body["include"] = ["reasoning.encrypted_content"]

        if tools:
            request_body["tools"] = [self._convert_tool_to_codex(t) for t in tools]
        else:
            request_body["tools"] = []

        response_data = await self._stream_response(request_body)
        if response_data is None:
            raise RuntimeError("No response received from Codex endpoint.")
        return self._parse_response(response_data)

    async def compact(
        self,
        messages: list[Message] | DAG,
        tools: Sequence[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> list[Message]:
        """Compact a conversation via the Codex ``/responses/compact`` endpoint.

        Returns the new conversation history as a list of Messages. Reasoning,
        tool calls, and tool results are not part of the returned history —
        the server's compaction reduces them into user/assistant message
        items per ``should_keep_compacted_history_item`` in codex-rs.

        For the common case of compacting a DAG and continuing the
        conversation with the same system prompt and tools, prefer
        :meth:`compact_dag`.
        """
        await self._refresh_if_file_backed()
        messages, tools, system_prompt = unpack_dag_or_args(
            messages, tools, system_prompt
        )

        if not messages:
            return []

        input_items: list[dict[str, Any]] = []
        for msg in messages:
            input_items.extend(self._convert_message_to_codex(msg))

        # CompactionInput shape (codex-api/common.rs:25-36): NO stream/store/
        # prompt_cache_key/client_metadata/include/tool_choice. installation_id
        # moves to the HTTP header on this endpoint (client.rs:469-471).
        request_body: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt or "You are a helpful assistant.",
            "input": input_items,
            "parallel_tool_calls": self.parallel_tool_calls,
            "tools": ([self._convert_tool_to_codex(t) for t in tools] if tools else []),
        }
        if self.reasoning:
            request_body["reasoning"] = build_reasoning_block(
                self.reasoning_effort, summary="detailed"
            )

        response_data = await self._post_compact(request_body)
        output_items = response_data.get("output") or []
        return _convert_compacted_output_to_messages(output_items)

    async def compact_dag(self, dag: DAG) -> DAG:
        """Compact ``dag`` and return a fresh DAG with the same system prompt
        and tools but the compacted history as messages.

        Retries with the oldest history message dropped if the compaction call
        itself returns ``context_length_exceeded`` — mirrors codex-rs's
        ``compact.rs:216-226`` (trim from the front to preserve prefix cache,
        keep recent messages intact). Gives up when only one history message
        remains.

        An empty DAG round-trips unchanged.
        """
        if not dag._heads:
            return dag
        # ``head`` raises when there are multiple heads (parallel branches).
        # Compaction across parallel branches is undefined; fall back early
        # rather than picking a branch arbitrarily.
        if len(dag._heads) != 1:
            raise ValueError(
                "compact_dag requires a DAG with a single head; "
                f"got {len(dag._heads)} heads (parallel branches not supported)."
            )

        dag_system_prompts = dag.head.get_system_prompts()
        original_tools = dag._tools

        truncated = 0
        attempt_messages = dag.to_messages()
        compacted: list[Message] = []
        while True:
            attempt_dag = _rebuild_dag(
                dag_system_prompts, original_tools, attempt_messages
            )
            try:
                compacted = await self.compact(attempt_dag)
                break
            except ContextWindowExceededError:
                if len(attempt_messages) <= 1:
                    # Nothing left to drop; surface the error to the caller.
                    raise
                # Trim from the front to preserve the prefix cache, matching
                # codex-rs ``history.remove_first_item()``.
                attempt_messages = attempt_messages[1:]
                truncated += 1
                continue

        if truncated:
            # codex-rs notifies the user via a background event; we just log.
            print(
                f"[CodexAPI.compact_dag] trimmed {truncated} oldest history "
                "item(s) before compacting so the prompt fits the model "
                "context window."
            )

        return _rebuild_dag(dag_system_prompts, original_tools, compacted)

    async def _post_compact(self, request_body: dict[str, Any]) -> dict[str, Any]:
        compact_url = self.base_url.rstrip("/") + _COMPACT_PATH_SUFFIX
        headers = self._build_request_headers(accept="application/json")
        # On /responses/compact the installation_id moves from body
        # ``client_metadata`` (used by /responses) into an HTTP header.
        headers[_CODEX_INSTALLATION_ID_HEADER] = self._get_installation_id()

        resp = await self._client.post(
            compact_url,
            headers=headers,
            json=request_body,
        )
        if resp.status_code != 200:
            snippet = resp.text[:400]
            self._raise_if_context_window_exceeded_body(snippet)
            raise RuntimeError(
                f"Codex API HTTP {resp.status_code}: {snippet or 'empty response'}"
            )
        self._capture_turn_state(resp.headers)
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Codex API compact response was not JSON: {resp.text[:400]!r}"
            ) from exc
        if isinstance(payload, dict) and payload.get("error"):
            error = payload["error"]
            self._raise_if_context_window_exceeded_error(error)
            raise RuntimeError(f"Codex API error: {error}")
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _raise_if_context_window_exceeded_error(error: Any) -> None:
        """Raise :class:`ContextWindowExceededError` when ``error`` is a Codex
        error payload with ``code == "context_length_exceeded"``.

        ``is_context_window_error_payload`` already type-guards ``error`` to
        ``dict``, so the field access here is safe.
        """
        if is_context_window_error_payload(error):
            raise ContextWindowExceededError(
                str(error.get("message") or error),
                provider=_CODEX_API_SOURCE,
            )

    @staticmethod
    def _raise_if_context_window_exceeded_body(body: str) -> None:
        """Raise :class:`ContextWindowExceededError` when ``body`` is a Codex
        error envelope ``{"error": {"code": "context_length_exceeded", ...}}``.

        Used on non-200 HTTP responses where the body is a JSON string, not
        already-parsed dict.
        """
        if not body:
            return
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            CodexAPI._raise_if_context_window_exceeded_error(payload.get("error"))

    def _models_url(self) -> str:
        """Derive the ``/models`` URL from ``base_url``.

        ``base_url`` points at ``…/codex/responses``; codex-rs's ``/models``
        endpoint sits one path segment up at ``…/codex/models``
        (model-provider/src/models_endpoint.rs ``MODELS_ENDPOINT``).
        """
        root = self.base_url.rstrip("/").rsplit("/", 1)[0]
        return f"{root}/models"

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch the Codex backend's model catalog.

        Mirrors codex-rs's ``ModelsClient.list_models``: GET on the provider's
        ``/models`` path, parsing the ``{"models": [ModelInfo]}`` envelope.
        Each entry includes ``slug``, ``context_window``,
        ``max_context_window``, and the rest of ``ModelInfo``.

        Useful when you want to populate ``context_window`` from the source of
        truth instead of hardcoding it. Returns an empty list when the response
        envelope is malformed; raises on HTTP errors / timeouts.
        """
        await self._refresh_if_file_backed()
        headers = self._build_request_headers(accept="application/json")
        headers[_CODEX_INSTALLATION_ID_HEADER] = self._get_installation_id()
        resp = await self._client.get(
            self._models_url(),
            headers=headers,
            params={"client_version": _CODEX_CLIENT_VERSION},
            timeout=_MODELS_FETCH_TIMEOUT,
        )
        if resp.status_code != 200:
            snippet = resp.text[:400]
            raise RuntimeError(
                f"Codex API HTTP {resp.status_code} fetching /models: "
                f"{snippet or 'empty response'}"
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Codex API /models response was not JSON: {resp.text[:400]!r}"
            ) from exc
        if not isinstance(payload, dict):
            return []
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        return [m for m in models if isinstance(m, dict)]

    async def fetch_context_window(self) -> int | None:
        """Look up ``self.model``'s ``context_window`` from ``/models``.

        Returns the integer when the backend reports one, ``None`` when the
        slug is missing from the catalog or the field is absent. Falls back to
        ``max_context_window`` (codex-rs ``resolved_context_window`` order).

        Side-effect-free: does not mutate ``self.context_window``. The caller
        decides whether to apply it (typically at construction time).
        """
        models = await self.list_models()
        for entry in models:
            if entry.get("slug") != self.model:
                continue
            window = entry.get("context_window")
            if isinstance(window, int) and window > 0:
                return window
            fallback = entry.get("max_context_window")
            if isinstance(fallback, int) and fallback > 0:
                return fallback
            return None
        return None

    def _build_request_headers(self, *, accept: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": accept,
            "originator": "codex_cli_rs",
            "session_id": self._session_id,
            "x-client-request-id": self._session_id,
            "OpenAI-Beta": "responses=experimental",
        }
        if self._turn_state:
            headers[_CODEX_TURN_STATE_HEADER] = self._turn_state
        if self.account_id:
            headers["chatgpt-account-id"] = self.account_id
        return headers

    def _capture_turn_state(self, response_headers: httpx.Headers) -> None:
        turn_state = response_headers.get(_CODEX_TURN_STATE_HEADER)
        if turn_state:
            self._turn_state = turn_state

    async def _stream_response(
        self, request_body: dict[str, Any]
    ) -> dict[str, Any] | None:
        last_response: dict[str, Any] | None = None
        headers = self._build_request_headers(accept="text/event-stream")
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

            self._capture_turn_state(resp.headers)

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

            last_response, last_event_type = await consume_responses_sse_stream(
                resp.aiter_lines(), source=_CODEX_API_SOURCE
            )
            if last_response is None and last_event_type:
                raise RuntimeError(
                    f"No response received from Codex endpoint (last event: {last_event_type})."
                )
        return last_response

    def _parse_sse_text(self, text: str) -> dict[str, Any] | None:
        """Parse SSE-formatted text and return last response if present."""
        return parse_responses_sse_text(text, source=_CODEX_API_SOURCE)

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
                        input=parse_tool_arguments(item.get("arguments", "{}")),
                        item_id=item.get("id"),
                    )
                )

        usage = responses_usage_from_dict(data.get("usage"))

        status = data.get("status", "")
        stop_reason = map_responses_status_to_stop_reason(status)

        return Response(
            id=data.get("id", ""),
            model=data.get("model", self.model),
            role=Role.ASSISTANT,
            content=content,
            stop_reason=stop_reason,
            usage=usage,
        )


def _rebuild_dag(
    system_prompts: list[str],
    tools: tuple[Tool, ...] | None,
    messages: list[Message],
) -> DAG:
    """Construct a DAG with the given system prompt(s), tools, and messages.

    Used by :meth:`CodexAPI.compact_dag` to (a) build trim-and-retry attempt
    DAGs and (b) build the final DAG from the compacted message list. Keeps
    the rebuild logic in one place so the trim path and the success path agree
    on shape.
    """
    new_dag = DAG()
    if system_prompts:
        new_dag = new_dag.system("\n\n".join(system_prompts))
    if tools:
        new_dag = new_dag.tools(*tools)
    for msg in messages:
        if msg.role == Role.USER:
            new_dag = new_dag.user(msg.content)
        elif msg.role == Role.ASSISTANT:
            new_dag = new_dag.assistant(msg.content)
    return new_dag


def _should_keep_compacted_history_item(item: dict[str, Any]) -> bool:
    """Port of codex-rs's ``should_keep_compacted_history_item``.

    Drops developer-role messages and instruction wrappers; drops
    reasoning/tool-call/tool-result items. Keeps user/assistant messages
    and ``Compaction`` items — those are opaque encrypted checkpoints
    that must be replayed on the next request to preserve compacted
    context (codex-rs ``compact_remote.rs:282`` does the same).
    """
    item_type = item.get("type", "")
    if item_type in CompactionContent.WIRE_TYPES:
        return True
    if item_type != "message":
        return False
    role = item.get("role")
    if role == "user":
        for nested in item.get("content", []):
            if isinstance(nested, dict) and nested.get("type") == "input_text":
                return True
        return False
    if role == "assistant":
        return True
    return False


def _convert_compacted_output_to_messages(
    items: list[dict[str, Any]],
) -> list[Message]:
    """Convert the ``output`` array from a /responses/compact reply into
    nano-agent ``Message`` objects.

    Compaction items (opaque encrypted checkpoints) are wrapped as
    assistant messages containing a single :class:`CompactionContent`
    block; the role choice is arbitrary because the codex serializer
    flattens these to top-level wire items, but using ``ASSISTANT``
    keeps them in the model-output channel of the conversation.
    """
    messages: list[Message] = []
    for item in items:
        if not isinstance(item, dict) or not _should_keep_compacted_history_item(item):
            continue
        item_type = item.get("type", "")
        if item_type in CompactionContent.WIRE_TYPES:
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=[CompactionContent.from_dict(item)],
                )
            )
            continue
        role = Role.USER if item.get("role") == "user" else Role.ASSISTANT
        nested_text_keys = ("input_text", "output_text")
        text_chunks: list[str] = []
        for nested in item.get("content", []):
            if not isinstance(nested, dict):
                continue
            ntype = nested.get("type", "")
            if ntype in nested_text_keys:
                text = nested.get("text", "")
                if text:
                    text_chunks.append(text)
            elif ntype == "refusal":
                refusal = nested.get("refusal", "")
                if refusal:
                    text_chunks.append(refusal)
        if not text_chunks:
            continue
        messages.append(Message(role=role, content="\n".join(text_chunks)))
    return messages


def _resolve_codex_home(
    codex_home: Path | str | None, auth_file: Path | str | None
) -> Path:
    if codex_home is not None:
        return Path(codex_home).expanduser()
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        return Path(env_home).expanduser()
    if auth_file is not None:
        return Path(auth_file).expanduser().parent
    return DEFAULT_CODEX_AUTH_PATH.parent


def _resolve_codex_installation_id(codex_home: Path) -> str:
    path = codex_home / _CODEX_INSTALLATION_ID_FILENAME
    try:
        raw = path.read_text().strip()
        if raw:
            return str(uuid.UUID(raw))
    except (OSError, ValueError):
        pass

    installation_id = str(uuid.uuid4())
    try:
        codex_home.mkdir(parents=True, exist_ok=True)
        path.write_text(installation_id)
        try:
            path.chmod(0o644)
        except OSError:
            pass
    except OSError:
        # If the Codex home is not writable, keep the session usable. The ID
        # remains stable for this client instance via ``self._installation_id``.
        pass
    return installation_id
