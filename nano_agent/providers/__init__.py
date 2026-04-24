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
    OAUTH_CLIENT_ID,
    OAUTH_TOKEN_URL,
    REFRESH_AFTER_DAYS,
    get_codex_access_token,
    get_codex_account_id,
    get_codex_refresh_token,
    load_codex_auth,
    maybe_refresh,
    save_codex_auth,
)
from .codex_login import async_codex_login, codex_login
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
    "save_codex_auth",
    "get_codex_access_token",
    "get_codex_refresh_token",
    "get_codex_account_id",
    "maybe_refresh",
    "DEFAULT_CODEX_AUTH_PATH",
    "OAUTH_CLIENT_ID",
    "OAUTH_TOKEN_URL",
    "REFRESH_AFTER_DAYS",
    "codex_login",
    "async_codex_login",
    # Cost tracking
    "CostBreakdown",
    "ModelPricing",
    "calculate_cost",
    "format_cost",
    "get_pricing",
]
