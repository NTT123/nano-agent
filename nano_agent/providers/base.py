"""Shared API infrastructure for all API clients.

This module provides:
- APIError: Unified exception with structured error context
- APIClientMixin: Shared functionality for HTTP error handling and resource cleanup
- APIProtocol: Type-safe protocol for API client polymorphism
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Protocol, Self

import httpx

from ..dag import DAG
from ..data_structures import Response

__all__ = ["APIError", "APIClientMixin", "APIProtocol"]


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
