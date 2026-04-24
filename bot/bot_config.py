"""Shared configuration + OAuth helpers for the Discord and Slack frontends.

Both bots read ``BOT_PROVIDER`` from the environment to pick an LLM backend,
and both expose a ``/renew`` slash command that re-runs the appropriate OAuth
flow. This module owns that logic so the frontends stay thin.
"""

from __future__ import annotations

import os

from nano_agent import ClaudeCodeAPI, CodexAPI
from nano_agent.providers import DEFAULT_CODEX_AUTH_PATH
from nano_agent.providers.base import APIProtocol


def get_bot_provider() -> str:
    return os.getenv("BOT_PROVIDER", "claude").strip().lower()


def get_codex_model() -> str:
    return os.getenv("CODEX_MODEL", "gpt-5.5")


def get_codex_reasoning_effort() -> str:
    return os.getenv("CODEX_REASONING_EFFORT", "high").strip().lower()


def build_api_from_env() -> APIProtocol:
    """Pick an LLM client based on ``BOT_PROVIDER``.

    ``codex`` → :class:`CodexAPI` reading ``~/.codex/auth.json`` (auto-refresh).
    Anything else → :class:`ClaudeCodeAPI` (Claude Code OAuth).
    """
    if get_bot_provider() == "codex":
        return CodexAPI(
            auth_file=DEFAULT_CODEX_AUTH_PATH,
            model=get_codex_model(),
            reasoning_effort=get_codex_reasoning_effort(),
        )
    return ClaudeCodeAPI()


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
