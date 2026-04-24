"""Perform the Codex (ChatGPT) OAuth PKCE flow from scratch in Python.

Produces a ``~/.codex/auth.json`` file compatible with the Codex CLI and with
:mod:`nano_agent.providers.codex_auth`. Use this when you don't want to depend
on the ``codex`` CLI being installed to populate the auth file.

Notes:
- This reuses the public Codex CLI OAuth ``client_id`` (observable in the
  open-source Codex binary). OpenAI can rotate it or tighten scopes at any
  time; this is best-effort, not an official third-party integration.
- The flow binds ``127.0.0.1:1455`` and requires a local browser. On a
  headless/remote host, either SSH-forward port 1455 back to a local browser
  or use ``--no-open`` to print the URL and open it manually.

CLI:
    uv run python -m nano_agent.providers.codex_login
    uv run python -m nano_agent.providers.codex_login --out ~/.codex/auth.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import secrets
import socket
import sys
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx

from .codex_auth import (
    DEFAULT_CODEX_AUTH_PATH,
    OAUTH_CLIENT_ID,
    OAUTH_TOKEN_URL,
    save_codex_auth,
)

__all__ = ["codex_login", "async_codex_login"]

ISSUER = "https://auth.openai.com"
REDIRECT_PORT = 1455
REDIRECT_PATH = "/auth/callback"
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}{REDIRECT_PATH}"
SCOPE = "openid profile email offline_access api.connectors.read api.connectors.invoke"
CALLBACK_TIMEOUT = 300.0


def _b64url_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_pkce() -> tuple[str, str]:
    verifier = _b64url_nopad(secrets.token_bytes(64))
    challenge = _b64url_nopad(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _decode_jwt_claims(jwt: str) -> dict[str, Any]:
    parts = jwt.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padded = payload + "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != REDIRECT_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = dict(urllib.parse.parse_qsl(parsed.query))
        self.server.oauth_params = params  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        ok = "code" in params and "error" not in params
        title = "Success." if ok else "Error."
        msg = (
            "You can close this tab and return to the terminal."
            if ok
            else "Login failed."
        )
        self.wfile.write(
            f"<!doctype html><meta charset=utf-8><title>Codex login</title>"
            f"<h1>{title}</h1><p>{msg}</p>".encode()
        )

    def log_message(self, *_args: Any) -> None:  # silence default stderr logging
        return


def _wait_for_callback(timeout: float = CALLBACK_TIMEOUT) -> dict[str, str]:
    try:
        server = HTTPServer(("127.0.0.1", REDIRECT_PORT), _CallbackHandler)
    except OSError as e:
        raise RuntimeError(
            f"cannot bind 127.0.0.1:{REDIRECT_PORT} ({e}); "
            "another Codex login may be in progress — close it and retry"
        ) from e
    server.oauth_params = {}  # type: ignore[attr-defined]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    t.join(timeout=timeout)
    server.server_close()
    return getattr(server, "oauth_params", {}) or {}


def _build_auth_json(token_resp: dict[str, Any]) -> dict[str, Any]:
    id_token = token_resp.get("id_token", "")
    claims = _decode_jwt_claims(id_token)
    chatgpt = claims.get("https://api.openai.com/auth") or {}
    account_id = (
        chatgpt.get("chatgpt_account_id")
        or claims.get("chatgpt_account_id")
        or claims.get("account_id")
        or claims.get("sub")
    )
    return {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": id_token,
            "access_token": token_resp["access_token"],
            "refresh_token": token_resp.get("refresh_token", ""),
            "account_id": account_id,
        },
        "last_refresh": datetime.now(timezone.utc).isoformat(),
    }


async def async_codex_login(
    out: Path | str | None = None,
    *,
    open_browser: bool = True,
    print_url: bool = False,
) -> dict[str, Any]:
    """Run the Codex OAuth PKCE flow and write auth.json.

    Args:
        out: Destination path (default: ``~/.codex/auth.json``).
        open_browser: Open the authorize URL in the user's browser.
        print_url: Also print the authorize URL to stderr (useful for headless).

    Returns:
        The auth dict written to disk.
    """
    out_path = Path(out).expanduser() if out else DEFAULT_CODEX_AUTH_PATH
    verifier, challenge = _make_pkce()
    state = secrets.token_urlsafe(32)

    qs = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": OAUTH_CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": "codex_cli_rs",
        }
    )
    auth_url = f"{ISSUER}/oauth/authorize?{qs}"

    # Pre-bind the port so we don't race the browser redirect.
    try:
        probe = socket.socket()
        probe.bind(("127.0.0.1", REDIRECT_PORT))
        probe.close()
    except OSError as e:
        raise RuntimeError(
            f"port {REDIRECT_PORT} busy ({e}); close any running `codex login` and retry"
        ) from e

    if print_url or not open_browser:
        print(f"Open this URL in a browser:\n\n{auth_url}\n", file=sys.stderr)
    if open_browser:
        print(f"[browser] {auth_url[:120]}…", file=sys.stderr)
        webbrowser.open(auth_url)

    print(
        f"[waiting] listening on {REDIRECT_URI} for up to "
        f"{int(CALLBACK_TIMEOUT)} seconds…",
        file=sys.stderr,
    )
    loop = asyncio.get_running_loop()
    params = await loop.run_in_executor(None, _wait_for_callback, CALLBACK_TIMEOUT)

    if params.get("state") != state:
        raise RuntimeError(
            f"OAuth state mismatch — expected {state}, got {params.get('state')}"
        )
    if "error" in params:
        raise RuntimeError(
            f"authorize error: {params.get('error')}: {params.get('error_description', '')}"
        )
    if "code" not in params:
        raise RuntimeError(
            "no authorization code received (timeout or user closed window)"
        )

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": params["code"],
                "redirect_uri": REDIRECT_URI,
                "client_id": OAUTH_CLIENT_ID,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if r.status_code != 200:
        raise RuntimeError(
            f"token exchange failed: HTTP {r.status_code}: {r.text[:400]}"
        )
    token_resp = r.json()
    auth = _build_auth_json(token_resp)
    save_codex_auth(out_path, auth)
    print(
        f"[ok] wrote {out_path} (account_id={auth['tokens']['account_id']})",
        file=sys.stderr,
    )
    return auth


def codex_login(
    out: Path | str | None = None,
    *,
    open_browser: bool = True,
    print_url: bool = False,
) -> dict[str, Any]:
    """Synchronous wrapper for :func:`async_codex_login`."""
    return asyncio.run(
        async_codex_login(out, open_browser=open_browser, print_url=print_url)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Codex (ChatGPT) OAuth login")
    ap.add_argument(
        "--out",
        default=str(DEFAULT_CODEX_AUTH_PATH),
        help=f"Where to write auth.json (default: {DEFAULT_CODEX_AUTH_PATH})",
    )
    ap.add_argument(
        "--no-open",
        action="store_true",
        help="Print the URL instead of opening a browser",
    )
    args = ap.parse_args()
    try:
        codex_login(args.out, open_browser=not args.no_open, print_url=args.no_open)
    except RuntimeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
