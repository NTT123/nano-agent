from __future__ import annotations

import json
from pathlib import Path

from nano_agent.providers.codex_auth import (
    get_codex_access_token,
    get_codex_refresh_token,
    load_codex_auth,
)


def _write_auth(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_codex_auth_missing(tmp_path: Path) -> None:
    assert load_codex_auth(tmp_path / "missing.json") is None


def test_get_codex_access_token_from_tokens(tmp_path: Path) -> None:
    path = _write_auth(
        tmp_path,
        {"tokens": {"access_token": "access-123", "refresh_token": "refresh-456"}},
    )
    assert get_codex_access_token(path) == "access-123"
    assert get_codex_refresh_token(path) == "refresh-456"


def test_get_codex_access_token_from_root(tmp_path: Path) -> None:
    path = _write_auth(tmp_path, {"access_token": "root-access"})
    assert get_codex_access_token(path) == "root-access"


def test_get_codex_access_token_nested(tmp_path: Path) -> None:
    path = _write_auth(
        tmp_path,
        {"auth": {"oauth": {"accessToken": "nested-access"}}},
    )
    assert get_codex_access_token(path) == "nested-access"


def test_get_codex_access_token_missing(tmp_path: Path) -> None:
    path = _write_auth(tmp_path, {"tokens": {"refresh_token": "refresh-only"}})
    assert get_codex_access_token(path) is None
