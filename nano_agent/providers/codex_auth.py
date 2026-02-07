"""Utilities for reading Codex (ChatGPT OAuth) credentials from disk.

File-mode only: reads ~/.codex/auth.json (or a provided path).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

DEFAULT_CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"


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
