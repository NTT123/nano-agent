"""Shared configuration + OAuth helpers for the Discord and Slack frontends.

Both bots read ``BOT_PROVIDER`` from the environment to pick an LLM backend,
and both expose a ``/renew`` slash command that re-runs the appropriate OAuth
flow. This module owns that logic so the frontends stay thin.
"""

from __future__ import annotations

import os
from typing import get_args

from nano_agent import ClaudeCodeAPI, CodexAPI
from nano_agent.providers import DEFAULT_CODEX_AUTH_PATH
from nano_agent.providers.base import APIProtocol, ReasoningEffort


def get_bot_provider() -> str:
    return os.getenv("BOT_PROVIDER", "claude").strip().lower()


def get_codex_model() -> str:
    return os.getenv("CODEX_MODEL", "gpt-5.5")


def get_codex_reasoning_effort() -> ReasoningEffort:
    valid = get_args(ReasoningEffort)
    raw = os.getenv("CODEX_REASONING_EFFORT", "high").strip().lower()
    if raw not in valid:
        raise ValueError(f"CODEX_REASONING_EFFORT must be one of {valid} (got {raw!r})")
    return raw  # type: ignore[return-value]


def get_codex_context_window() -> int | None:
    """Read the model context window override for auto-compaction.

    When set to a positive integer, takes precedence over the value fetched
    from the Codex ``/models`` endpoint. When unset, the bot auto-discovers
    the window from the backend on startup. Set ``CODEX_CONTEXT_WINDOW=0``
    (or any non-positive value) to disable auto-compaction entirely.
    """
    raw = os.getenv("CODEX_CONTEXT_WINDOW", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"CODEX_CONTEXT_WINDOW must be an integer (got {raw!r})"
        ) from exc
    return value if value > 0 else None


def build_api_from_env() -> APIProtocol:
    """Pick an LLM client based on ``BOT_PROVIDER``.

    ``codex`` → :class:`CodexAPI` reading ``~/.codex/auth.json`` (auto-refresh).
    Anything else → :class:`ClaudeCodeAPI` (Claude Code OAuth).

    Note: when ``CODEX_CONTEXT_WINDOW`` is unset, the returned ``CodexAPI``
    has ``context_window=None`` (auto-compact disabled). Call
    :func:`maybe_discover_context_window` from an async context (typically
    ``on_ready``) to populate it from the backend's ``/models`` endpoint.
    """
    if get_bot_provider() == "codex":
        return CodexAPI(
            auth_file=DEFAULT_CODEX_AUTH_PATH,
            model=get_codex_model(),
            reasoning_effort=get_codex_reasoning_effort(),
            context_window=get_codex_context_window(),
        )
    return ClaudeCodeAPI()


async def maybe_discover_context_window(api: APIProtocol) -> None:
    """Populate ``api.context_window`` from the backend if not yet set.

    Codex backend exposes a ``/models`` endpoint with per-model
    ``context_window`` values (mirrors codex-rs ``ModelsClient.list_models``);
    fetching once on bot startup avoids hardcoding a per-model registry. A
    failed fetch is logged but doesn't crash the bot — auto-compaction simply
    stays disabled. No-op for non-Codex API clients.
    """
    if not isinstance(api, CodexAPI) or api.context_window is not None:
        return
    try:
        api.context_window = await api.fetch_context_window()
    except Exception as exc:  # noqa: BLE001 — fail-open: log and continue.
        print(f"[bot_config] /models fetch failed; auto-compact disabled: {exc}")


async def async_renew_oauth() -> None:
    """Re-run the OAuth flow for the active provider.

    For ``codex`` this opens a browser and binds ``127.0.0.1:1455`` — it only
    works on a host where the bot operator can click through a browser, which
    rules out most remote deployments (log in locally and copy
    ``~/.codex/auth.json`` to the server instead).
    """
    if get_bot_provider() == "codex":
        from nano_agent.providers import async_codex_login

        await async_codex_login(
            DEFAULT_CODEX_AUTH_PATH, open_browser=True, print_url=True
        )
    else:
        from nano_agent.providers.capture_claude_code_auth import async_get_config

        await async_get_config(timeout=30)
