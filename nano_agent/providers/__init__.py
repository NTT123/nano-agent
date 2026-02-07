"""API providers for nano_agent.

This package contains all LLM API clients, auth helpers, and cost tracking.
"""

from .base import APIClientMixin, APIError, APIProtocol
from .capture_claude_code_auth import (
    DEFAULT_CONFIG_PATH,
    async_get_config,
    get_config,
    get_headers,
    load_config,
    save_config,
)
from .claude import ClaudeAPI
from .claude_code import ClaudeCodeAPI
from .codex import CodexAPI
from .codex_auth import (
    DEFAULT_CODEX_AUTH_PATH,
    get_codex_access_token,
    get_codex_refresh_token,
    load_codex_auth,
)
from .cost import CostBreakdown, ModelPricing, calculate_cost, format_cost, get_pricing
from .fireworks import FireworksAPI
from .gemini import GeminiAPI
from .openai import OpenAIAPI

__all__ = [
    # Base classes
    "APIError",
    "APIClientMixin",
    "APIProtocol",
    # API clients
    "ClaudeAPI",
    "ClaudeCodeAPI",
    "OpenAIAPI",
    "CodexAPI",
    "GeminiAPI",
    "FireworksAPI",
    # Auth capture utilities
    "get_config",
    "get_headers",
    "load_config",
    "save_config",
    "async_get_config",
    "DEFAULT_CONFIG_PATH",
    # Codex auth helpers
    "load_codex_auth",
    "get_codex_access_token",
    "get_codex_refresh_token",
    "DEFAULT_CODEX_AUTH_PATH",
    # Cost tracking
    "CostBreakdown",
    "ModelPricing",
    "calculate_cost",
    "format_cost",
    "get_pricing",
]
