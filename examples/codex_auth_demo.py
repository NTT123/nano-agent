"""Demo: load Codex OAuth token from ~/.codex/auth.json (file mode)."""

from __future__ import annotations

from nano_agent import get_codex_access_token


def _preview(token: str | None) -> str:
    if not token:
        return "<missing>"
    if len(token) <= 10:
        return token
    return f"{token[:6]}...{token[-4:]}"


def main() -> None:
    token = get_codex_access_token()
    print(f"Codex access token: {_preview(token)}")
    if token:
        headers = {"authorization": f"Bearer {token}"}
        print("Example headers:", headers)
    else:
        print("No token found. Ensure Codex is configured to store auth in file mode.")


if __name__ == "__main__":
    main()
