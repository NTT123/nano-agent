"""Shared API infrastructure for all API clients.

This module provides:
- APIError: Unified exception with structured error context
- APIClientMixin: Shared functionality for HTTP error handling and resource cleanup
- APIProtocol: Type-safe protocol for API client polymorphism
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, Literal, Protocol, Self

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
ReasoningSummary = Literal["concise", "detailed", "auto"]

import httpx

from ..dag import DAG
from ..data_structures import (
    APPROX_BYTES_PER_TOKEN,
    ImageContent,
    Message,
    Response,
    Role,
    TextContent,
    ToolResultContent,
    approx_token_count,
)
from ..tools.base import Tool

__all__ = [
    "APIError",
    "APIClientMixin",
    "APIProtocol",
    "APPROX_BYTES_PER_TOKEN",
    "CONTEXT_LENGTH_EXCEEDED_CODE",
    "ContextWindowExceededError",
    "DEFAULT_STREAM_IDLE_TIMEOUT",
    "ReasoningEffort",
    "ReasoningSummary",
    "approx_token_count",
    "attach_collected_output",
    "build_httpx_timeout",
    "build_reasoning_block",
    "consume_responses_sse_stream",
    "force_required_all",
    "is_context_window_error_payload",
    "parse_responses_sse_text",
    "parse_tool_arguments",
    "raise_for_stream_event_error",
    "responses_tool_result_item",
    "responses_user_image_item",
    "serialize_tool_arguments",
    "unpack_dag_or_args",
]

# Wire-level error code emitted by the OpenAI/Codex Responses API when an
# input exceeds the model's context window. Codex-rs gates its
# ContextWindowExceeded mapping on this exact string
# (codex-api/src/sse/responses.rs ``is_context_window_error``).
CONTEXT_LENGTH_EXCEEDED_CODE = "context_length_exceeded"


_COLLECTABLE_OUTPUT_ITEM_TYPES = ("reasoning", "message", "function_call")

# Mirrors codex-rs's DEFAULT_STREAM_IDLE_TIMEOUT_MS (300_000). High-effort
# reasoning calls can leave an SSE stream silent for minutes between events,
# so the per-chunk read budget must well exceed httpx's 5s default.
DEFAULT_STREAM_IDLE_TIMEOUT = 300.0


def build_httpx_timeout(read: float = DEFAULT_STREAM_IDLE_TIMEOUT) -> httpx.Timeout:
    """Build the standard httpx.Timeout for Responses-API streaming clients.

    ``read`` is the per-chunk idle budget — the analog of codex-rs's
    ``stream_idle_timeout``. ``connect``/``write``/``pool`` stay short
    so non-stream failures fail fast.
    """
    return httpx.Timeout(connect=10.0, read=read, write=30.0, pool=10.0)


def build_reasoning_block(
    effort: ReasoningEffort, summary: ReasoningSummary | None = None
) -> dict[str, Any]:
    """Build the ``reasoning`` field for a Responses-API request body.

    ``summary`` is omitted when ``None`` because ``"auto"`` requires
    organization verification on the public OpenAI API; callers opt in
    explicitly. The Codex backend has no such restriction and passes
    ``"detailed"`` directly.
    """
    block: dict[str, Any] = {"effort": effort}
    if summary is not None:
        block["summary"] = summary
    return block


def serialize_tool_arguments(input_dict: dict[str, Any] | None) -> str:
    """Serialize a tool-call input dict to the JSON string the Responses API
    expects in ``function_call.arguments``."""
    if input_dict is None:
        return "{}"
    return json.dumps(input_dict)


def parse_tool_arguments(arguments: str) -> dict[str, Any]:
    """Parse a Responses-API ``function_call.arguments`` string back into a
    dict, returning ``{}`` for malformed or non-object payloads."""
    try:
        result = json.loads(arguments)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def unpack_dag_or_args(
    messages: list[Message] | DAG,
    tools: Sequence[Tool] | None,
    system_prompt: str | None,
) -> tuple[list[Message], Sequence[Tool] | None, str | None]:
    """Resolve a DAG-or-args call into ``(messages, tools, system_prompt)``.

    When ``messages`` is a DAG, its tools and system prompts override the
    caller's arguments — but only when the DAG actually supplies them, so
    a DAG with an empty tool tuple defers to the caller's ``tools``.
    """
    if not isinstance(messages, DAG):
        return messages, tools, system_prompt
    dag = messages
    dag_messages = dag.to_messages()
    dag_tools: list[Tool] = list(dag._tools or [])
    dag_system_prompts = dag.head.get_system_prompts() if dag._heads else []
    return (
        dag_messages,
        dag_tools if dag_tools else tools,
        "\n\n".join(dag_system_prompts) if dag_system_prompts else system_prompt,
    )


_strict_schema_cache: dict[int, dict[str, Any]] = {}


def force_required_all(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` with every property promoted to ``required``.

    OpenAI's Responses-API strict mode requires every key in ``properties``
    to also appear in ``required`` (and ``additionalProperties: false``).
    ``schema_from_dataclass`` only marks fields without defaults as required,
    so tools with optional fields would otherwise be rejected.

    The input is deep-copied because ``Tool.input_schema`` returns a
    ``ClassVar``-cached dict; mutating it would corrupt every subsequent use
    of the tool class. The result is memoized by ``id(schema)`` so we only
    pay the deepcopy + traversal cost once per tool class — every send()
    re-fetches the same dict object from the ClassVar.
    """
    cached = _strict_schema_cache.get(id(schema))
    if cached is not None:
        return cached
    result = copy.deepcopy(schema)
    _force_required_all_inplace(result)
    _strict_schema_cache[id(schema)] = result
    return result


def _force_required_all_inplace(schema: dict[str, Any]) -> None:
    if schema.get("type") == "object":
        props = schema.get("properties")
        if isinstance(props, dict):
            schema["required"] = list(props.keys())
            for prop in props.values():
                if isinstance(prop, dict):
                    _force_required_all_inplace(prop)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            _force_required_all_inplace(additional)
    elif schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            _force_required_all_inplace(items)


def attach_collected_output(
    response: dict[str, Any] | None, collected_items: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Merge ``response.output_item.done`` items into the final response.

    The Codex backend streams output items via ``response.output_item.done``
    but emits an empty ``output`` array on ``response.completed``; the public
    OpenAI Responses API populates ``output`` directly. This helper papers
    over the difference: if the final response has no ``output`` but items
    were collected separately, attach them.
    """
    if response is not None and not response.get("output") and collected_items:
        return {**response, "output": collected_items}
    return response


def raise_for_stream_event_error(
    event: dict[str, Any], *, source: str = "Responses API"
) -> None:
    """Raise ``RuntimeError`` if ``event`` represents a Responses-API failure.

    Handles three cases that the Responses API uses to signal errors mid-
    stream: a top-level ``error`` field, ``response.failed`` events, and
    ``response.incomplete`` events with a populated ``incomplete_details``.
    """
    if event.get("error"):
        raise RuntimeError(f"{source} error: {event['error']}")

    etype = event.get("type")
    if etype == "response.failed":
        response = event.get("response")
        error: object = None
        if isinstance(response, dict):
            error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error
        else:
            message = error or "response.failed event received"
        raise RuntimeError(f"{source} error: {message}")

    if etype == "response.incomplete":
        reason = "unknown"
        response = event.get("response")
        if isinstance(response, dict):
            incomplete_details = response.get("incomplete_details")
            if isinstance(incomplete_details, dict):
                raw_reason = incomplete_details.get("reason")
                if isinstance(raw_reason, str) and raw_reason:
                    reason = raw_reason
        raise RuntimeError(f"{source} incomplete response: {reason}")


async def consume_responses_sse_stream(
    line_aiter: AsyncIterator[str], *, source: str = "Responses API"
) -> tuple[dict[str, Any] | None, str | None]:
    """Drain a Responses-API SSE line stream and assemble the final response.

    Returns ``(response, last_event_type)``. ``response`` is the body of the
    final ``response.completed`` event with any separately-streamed output
    items attached; ``last_event_type`` is informational, used by callers
    to build a useful "no response received" message.

    Raises ``RuntimeError`` for ``response.failed`` / ``response.incomplete``
    via :func:`raise_for_stream_event_error`.
    """
    last_response: dict[str, Any] | None = None
    last_event_type: str | None = None
    collected_items: list[dict[str, Any]] = []
    async for line in line_aiter:
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        last_event_type = event.get("type")
        raise_for_stream_event_error(event, source=source)
        if last_event_type == "response.completed":
            last_response = event.get("response")
        elif last_event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") in _COLLECTABLE_OUTPUT_ITEM_TYPES:
                collected_items.append(item)
    return attach_collected_output(last_response, collected_items), last_event_type


def parse_responses_sse_text(
    text: str, *, source: str = "Responses API"
) -> dict[str, Any] | None:
    """Synchronous variant of :func:`consume_responses_sse_stream` for the
    case where a non-streaming endpoint returns a complete SSE-formatted
    body in one shot (the Codex backend occasionally does this on errors).
    """
    last_response: dict[str, Any] | None = None
    collected_items: list[dict[str, Any]] = []
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
        raise_for_stream_event_error(event, source=source)
        etype = event.get("type")
        if etype == "response.completed":
            last_response = event.get("response")
        elif etype == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") in _COLLECTABLE_OUTPUT_ITEM_TYPES:
                collected_items.append(item)
    return attach_collected_output(last_response, collected_items)


def responses_tool_result_item(block: ToolResultContent) -> dict[str, Any]:
    """Serialize a ``ToolResultContent`` into an OpenAI Responses-API
    ``function_call_output`` item.

    ``output`` is emitted as an array of ``input_text`` / ``input_image``
    items when the tool result contains any image; otherwise as a plain
    string for compactness.
    """
    output_items: list[dict[str, Any]] = []
    has_image = False
    text_chunks: list[str] = []
    for tb in block.content:
        if isinstance(tb, ImageContent):
            has_image = True
            output_items.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{tb.media_type};base64,{tb.data}",
                }
            )
        elif isinstance(tb, TextContent):
            text_chunks.append(tb.text)
            output_items.append({"type": "input_text", "text": tb.text})

    if not has_image:
        return {
            "type": "function_call_output",
            "call_id": block.tool_use_id,
            "output": "".join(text_chunks),
        }
    return {
        "type": "function_call_output",
        "call_id": block.tool_use_id,
        "output": output_items,
    }


def responses_user_image_item(msg: Message, image: ImageContent) -> dict[str, Any]:
    """Build a user-role message item containing an ``input_image`` block.

    Used when an ``ImageContent`` appears mid-message: the caller flushes
    pending text first, then emits this item to carry the image. OpenAI
    Responses only accepts ``input_image`` under a user role.
    """
    return {
        "role": Role.USER.value if msg.role == Role.USER else msg.role.value,
        "content": [
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{image.media_type};base64,{image.data}",
            }
        ],
    }


class APIError(Exception):
    """Unified API error with structured context.

    Provides consistent error handling across all API providers with:
    - HTTP status code (when applicable)
    - Provider-specific error type
    - Provider name for debugging

    Example:
        >>> try:
        ...     response = await api.send(dag)
        ... except APIError as e:
        ...     if e.status_code == 429:
        ...         # Handle rate limit
        ...         await asyncio.sleep(60)
        ...     elif e.status_code and e.status_code >= 500:
        ...         # Handle server error with retry
        ...         pass
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_type: str | None = None,
        provider: str = "unknown",
    ):
        """Initialize APIError.

        Args:
            message: Human-readable error description
            status_code: HTTP status code (e.g., 400, 401, 429, 500)
            error_type: Provider-specific error type (e.g., "invalid_api_key")
            provider: Name of the API provider (e.g., "Claude", "OpenAI", "Gemini")
        """
        self.status_code = status_code
        self.error_type = error_type
        self.provider = provider
        super().__init__(f"[{provider}] {message}")

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"APIError(message={str(self)!r}, "
            f"status_code={self.status_code}, "
            f"error_type={self.error_type!r}, "
            f"provider={self.provider!r})"
        )


class ContextWindowExceededError(APIError):
    """Raised when an API call's input exceeds the model's context window.

    Mirrors codex-rs's ``CodexErr::ContextWindowExceeded`` (protocol/src/error.rs).
    Detected from the wire-level ``error.code == "context_length_exceeded"``
    (codex-rs ``is_context_window_error`` in codex-api/src/sse/responses.rs).

    Callers — notably :meth:`CodexAPI.compact_dag` — catch this to drop the
    oldest history item and retry, mirroring ``compact.rs:216-226``.
    """

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_type=CONTEXT_LENGTH_EXCEEDED_CODE,
            provider=provider,
        )


def is_context_window_error_payload(error: Any) -> bool:
    """Return ``True`` when an API error payload signals context-window overflow.

    Mirrors codex-rs ``is_context_window_error`` (codex-api/src/sse/responses.rs):
    a single field check on ``error.code == "context_length_exceeded"``. The
    ``error`` argument is whatever the backend returned in its ``error`` slot
    (typically a dict with ``code``, ``message``, ``type``).
    """
    return isinstance(error, dict) and error.get("code") == CONTEXT_LENGTH_EXCEEDED_CODE


class APIProtocol(Protocol):
    """Protocol defining the interface for all API clients.

    All API clients (ClaudeAPI, ClaudeCodeAPI, OpenAIAPI, GeminiAPI) must
    implement this protocol to be usable with the executor.

    Example:
        >>> async def my_function(api: APIProtocol, dag: DAG) -> Response:
        ...     return await api.send(dag)
    """

    _client: httpx.AsyncClient

    async def send(self, dag: "DAG") -> "Response":
        """Send a request to the API.

        Args:
            dag: The conversation DAG to send

        Returns:
            Response from the API
        """
        ...


class APIClientMixin:
    """Mixin providing shared API client functionality.

    Provides:
    - Consistent HTTP error checking across providers
    - Async context manager support for proper resource cleanup
    - close() method for explicit cleanup

    Usage:
        >>> class MyAPI(APIClientMixin):
        ...     def __init__(self):
        ...         self._client = httpx.AsyncClient(timeout=120.0)
        ...
        ...     async def send(self, dag):
        ...         response = await self._client.post(...)
        ...         data = self._check_response(response, provider="MyProvider")
        ...         return Response.from_dict(data)
    """

    _client: httpx.AsyncClient

    def _check_response(
        self,
        response: httpx.Response,
        provider: str = "API",
    ) -> dict[str, Any]:
        """Check HTTP response and raise unified errors.

        Handles two error patterns:
        1. HTTP status code != 200 (standard REST error)
        2. "error" key in response body (OpenAI pattern)

        Args:
            response: The httpx Response object
            provider: Name of the API provider for error messages

        Returns:
            Parsed JSON response if successful

        Raises:
            APIError: If response indicates an error
        """
        response_json: dict[str, Any] = response.json()

        # Check HTTP status code first (most APIs return non-200 on error)
        if response.status_code != 200:
            error_data = response_json.get("error", {})
            if isinstance(error_data, dict):
                error_type = error_data.get("type", "unknown")
                error_msg = error_data.get("message", str(response_json))
            else:
                error_type = "unknown"
                error_msg = str(error_data or response_json)

            raise APIError(
                message=f"HTTP {response.status_code}: {error_msg}",
                status_code=response.status_code,
                error_type=error_type,
                provider=provider,
            )

        # Check for error key in body (OpenAI returns 200 with error key sometimes)
        error = response_json.get("error")
        if error is not None:
            if isinstance(error, dict):
                raise APIError(
                    message=error.get("message", str(error)),
                    error_type=error.get("type"),
                    provider=provider,
                )
            else:
                raise APIError(message=str(error), provider=provider)

        return response_json

    async def close(self) -> None:
        """Close the HTTP client and release resources.

        Should be called when the client is no longer needed to prevent
        connection leaks. Alternatively, use the async context manager.

        Example:
            >>> api = ClaudeAPI()
            >>> try:
            ...     response = await api.send(dag)
            ... finally:
            ...     await api.close()
        """
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Async context manager entry.

        Example:
            >>> async with ClaudeAPI() as api:
            ...     response = await api.send(dag)
            ... # Client automatically closed
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - closes the client."""
        await self.close()
