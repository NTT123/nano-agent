"""Utilities for reading Codex (ChatGPT OAuth) credentials from disk.

File-mode only: reads ~/.codex/auth.json (or a provided path).
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
REFRESH_AFTER_DAYS = 8


def load_codex_auth(path: Path | str | None = None) -> dict[str, Any] | None:
    """Load Codex auth.json credentials from disk.

    Args:
        path: Auth file path (default: ~/.codex/auth.json)

    Returns:
        Parsed JSON dict or None if missing/invalid.
    """
    auth_path = Path(path) if path else DEFAULT_CODEX_AUTH_PATH
    if not auth_path.exists():
        return None

    # Warn if permissions are too open
    try:
        mode = auth_path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            import warnings

            warnings.warn(
                f"Codex auth file has open permissions: chmod 600 {auth_path}",
                stacklevel=2,
            )
    except OSError:
        pass

    try:
        raw = json.loads(auth_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if isinstance(raw, dict):
        result: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(key, str):
                result[key] = value
        return result
    return None


def get_codex_access_token(path: Path | str | None = None) -> str | None:
    """Return the Codex OAuth access token if present."""
    data = load_codex_auth(path)
    if not data:
        return None
    return _find_token_value(data, {"access_token", "accessToken"})


def get_codex_refresh_token(path: Path | str | None = None) -> str | None:
    """Return the Codex OAuth refresh token if present."""
    data = load_codex_auth(path)
    if not data:
        return None
    return _find_token_value(data, {"refresh_token", "refreshToken"})


def get_codex_account_id(path: Path | str | None = None) -> str | None:
    """Return the ChatGPT account_id from Codex auth.json if present."""
    data = load_codex_auth(path)
    if not data:
        return None
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else data
    account = tokens.get("account_id") if isinstance(tokens, dict) else None
    return account if isinstance(account, str) and account else None


def save_codex_auth(path: Path | str, data: dict[str, Any]) -> None:
    """Write Codex auth.json with 0600 permissions."""
    auth_path = Path(path)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(auth_path, 0o600)
    except OSError:
        pass


def parse_iso(ts: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string, tolerating ``Z`` suffix.

    Returns ``None`` for non-strings or unparseable input.
    """
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


async def maybe_refresh(
    auth: dict[str, Any],
    path: Path | str,
    client: httpx.AsyncClient,
    *,
    refresh_after_days: float = REFRESH_AFTER_DAYS,
) -> dict[str, Any]:
    """Refresh access_token if auth is older than refresh_after_days.

    Mirrors the Codex CLI refresh flow: POST JSON to auth.openai.com/oauth/token
    with grant_type=refresh_token. Updates auth dict and writes it back to disk
    on success. Returns the (possibly updated) auth dict.

    Raises RuntimeError if refresh is needed but no refresh_token is available
    or the network call fails.
    """
    parsed = parse_iso(auth.get("last_refresh"))
    if parsed is not None:
        age_days = (datetime.now(timezone.utc) - parsed).total_seconds() / 86400
        if age_days < refresh_after_days:
            return auth

    tokens = auth.get("tokens") or {}
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("No refresh_token in auth.json; re-login required.")

    resp = await client.post(
        OAUTH_TOKEN_URL,
        json={
            "client_id": OAUTH_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Codex token refresh failed: HTTP {resp.status_code}: {resp.text[:400]}"
        )
    body = resp.json()
    tokens["access_token"] = body["access_token"]
    for field in ("refresh_token", "id_token"):
        if field in body:
            tokens[field] = body[field]
    auth["tokens"] = tokens
    auth["last_refresh"] = datetime.now(timezone.utc).isoformat()
    save_codex_auth(path, auth)
    return auth


def _find_token_value(data: Any, keys: set[str]) -> str | None:
    """Recursively search for the first token value by key."""
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        for value in data.values():
            found = _find_token_value(value, keys)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_token_value(item, keys)
            if found:
                return found
    return None
