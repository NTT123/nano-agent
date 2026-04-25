"""Protocol definitions shared across the package.

Lives outside ``providers/base`` so consumers can depend on ``APIProtocol``
without pulling in the ``Tool`` import that ``providers/base`` carries.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from .dag import DAG
from .data_structures import Response


class APIProtocol(Protocol):
    """Protocol defining the interface for all API clients.

    All API clients (ClaudeAPI, ClaudeCodeAPI, OpenAIAPI, GeminiAPI) must
    implement this protocol to be usable with the executor.

    Example:
        >>> async def my_function(api: APIProtocol, dag: DAG) -> Response:
        ...     return await api.send(dag)
    """

    _client: httpx.AsyncClient

    async def send(self, dag: DAG) -> Response:
        """Send a request to the API.

        Args:
            dag: The conversation DAG to send

        Returns:
            Response from the API
        """
        ...
