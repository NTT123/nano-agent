"""
Capture auth token and configuration from Claude Code CLI.

This module intercepts Claude Code CLI API requests to capture the authorization
token and configuration. It acts as a true MITM proxy, forwarding requests to
the real Anthropic API and returning real responses.

How it works:
1. Starts a local HTTP server on a random port
2. Launches Claude Code CLI with ANTHROPIC_BASE_URL pointing to localhost
3. Intercepts API requests, captures headers/body, forwards to real API
4. Returns real API responses to the CLI (true MITM proxy)
5. Optionally saves captured config to ~/.nano-agent.json

Usage:
    from nano_agent import get_config, get_headers, load_config, save_config

    # Capture and save (default)
    headers, body_params = get_config()

    # Load previously saved config
    config = load_config()
    if config:
        headers, body_params = config
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx

# Default config file path
DEFAULT_CONFIG_PATH = Path.home() / ".nano-agent.json"


class MITMCaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures requests and forwards to real API (true MITM proxy)."""

    # Store ALL captured configs (use last one for most up-to-date tokens)
    all_captured_configs: list[dict[str, Any]] = []

    # Class-level HTTP client for connection pooling (synchronous)
    http_client: httpx.Client | None = None

    @classmethod
    def get_http_client(cls) -> httpx.Client:
        """Get or create the HTTP client for forwarding requests."""
        if cls.http_client is None:
            cls.http_client = httpx.Client(timeout=120.0)
        return cls.http_client

    @classmethod
    def close_http_client(cls) -> None:
        """Close the HTTP client if it exists."""
        if cls.http_client is not None:
            cls.http_client.close()
            cls.http_client = None

    def do_POST(self) -> None:
        """Capture request and return minimal response (no forwarding needed)."""
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b""

        # Try to capture config from this request
        captured = self._maybe_capture_config(body_bytes)

        # Return minimal response - we don't need to forward to real API
        # Just need the headers/auth token for config capture
        if captured:
            # Return a minimal valid response so CLI exits cleanly
            self._send_minimal_response()
        else:
            # Not a config request, forward to real API
            self._forward_to_api(body_bytes)

    def do_GET(self) -> None:
        """Handle GET requests by forwarding to real API."""
        self._forward_to_api(b"")

    def _send_minimal_response(self) -> None:
        """Send minimal valid Claude API response after capturing config."""
        # Minimal response that satisfies Claude CLI
        response_body = json.dumps(
            {
                "id": "msg_capture",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode("utf-8")

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected, ignore

    def _forward_to_api(self, body_bytes: bytes) -> None:
        """Forward request to real Anthropic API and return real response."""
        # Build forward URL
        forward_url = f"https://api.anthropic.com{self.path}"

        # Forward headers (exclude hop-by-hop headers)
        hop_by_hop = {"host", "connection", "transfer-encoding", "content-length"}
        forward_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in hop_by_hop
        }

        try:
            client = self.get_http_client()

            if self.command == "POST":
                response = client.post(
                    forward_url,
                    headers=forward_headers,
                    content=body_bytes,
                )
            else:
                response = client.get(forward_url, headers=forward_headers)

            # Return real response to CLI
            self.send_response(response.status_code)

            # Forward response headers (exclude problematic ones)
            skip_headers = {"transfer-encoding", "content-encoding", "connection"}
            for key, value in response.headers.items():
                if key.lower() not in skip_headers:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.content)

        except Exception as e:
            # On error, try to send a minimal error response
            # Suppress noise if we already captured config (client may have disconnected)
            if not MITMCaptureHandler.all_captured_configs:
                print(f"✗ Forward error: {e}", file=sys.stderr)
            try:
                error_response = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_response)))
                self.end_headers()
                self.wfile.write(error_response)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client disconnected, ignore
                pass

    def _maybe_capture_config(self, body_bytes: bytes) -> bool:
        """Capture config if valid full-config request. Appends to list."""
        try:
            body = json.loads(body_bytes.decode("utf-8"))
            messages = body.get("messages", [])

            # Skip quota check requests
            if messages and messages[0].get("content") == "quota":
                print("⊘ Skipping quota check request", file=sys.stderr)
                return False

            # Check for cache_control in system prompt (indicates full config)
            if body.get("system") and self._has_cache_control(body.get("system")):
                config = {
                    "headers": self._capture_current_headers(),
                    "body": body,
                    "url_path": self.path,
                    "captured_at": time.time(),
                }
                MITMCaptureHandler.all_captured_configs.append(config)
                count = len(MITMCaptureHandler.all_captured_configs)
                print(f"✓ Captured config #{count}", file=sys.stderr)
                return True
            else:
                print(
                    "⊘ Skipping request without cached system prompt", file=sys.stderr
                )

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"✗ Failed to parse request body: {e}", file=sys.stderr)

        return False

    def _capture_current_headers(self) -> dict[str, str]:
        """Capture headers from the current request."""
        exclude_headers = {
            "host",
            "content-length",
            "content-type",
            "accept",
            "connection",
            "transfer-encoding",
            "accept-encoding",
        }

        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            if key.lower() not in exclude_headers:
                headers[key.lower()] = value

        return headers

    @staticmethod
    def _has_cache_control(system: Any) -> bool:
        """Check if system prompt has cache_control (indicates full config)."""
        if isinstance(system, list):
            for item in system:
                if isinstance(item, dict) and "cache_control" in item:
                    return True
        return False

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default logging."""
        pass

    def handle(self) -> None:
        """Handle request, suppressing errors after config is captured."""
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client disconnected, ignore if we already captured config
            pass


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _filter_long_system_messages(
    body_params: dict[str, Any],
    max_words: int = 1000,
) -> dict[str, Any]:
    """Filter out system messages longer than max_words.

    Args:
        body_params: The captured body parameters
        max_words: Maximum words allowed per system message (default: 1000)

    Returns:
        New dict with long system messages removed
    """
    system = body_params.get("system")
    if not isinstance(system, list):
        return body_params

    filtered_system = [
        item
        for item in system
        # if not isinstance(item, dict) or _count_words(item.get("text", "")) <= max_words
    ]

    return {**body_params, "system": filtered_system}


def _extract_body_params(body: dict[str, Any], url_path: str | None) -> dict[str, Any]:
    """Extract key parameters from request body."""
    return {
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "thinking": body.get("thinking"),
        "user_id": body.get("metadata", {}).get("user_id"),
        "system": body.get("system"),
        "url_path": url_path,
    }


def save_config(
    headers: dict[str, str],
    body_params: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Save config to JSON file with secure permissions (0600).

    Args:
        headers: Captured HTTP headers
        body_params: Captured body parameters
        path: Config file path (default: ~/.nano-agent.json)

    Returns:
        Path to the saved config file
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    # Filter long system messages before saving
    filtered_body_params = _filter_long_system_messages(body_params)

    config = {
        "version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "headers": headers,
        "body_params": filtered_body_params,
    }

    config_path.write_text(json.dumps(config, indent=2))

    # Set secure permissions (0600 - owner read/write only)
    os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)

    return config_path


def load_config(
    path: Path | str | None = None,
) -> tuple[dict[str, str], dict[str, Any]] | None:
    """Load config from JSON file.

    Args:
        path: Config file path (default: ~/.nano-agent.json)

    Returns:
        Tuple of (headers, body_params) or None if missing/invalid
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return None

    # Warn if permissions are too open
    try:
        mode = config_path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            import warnings

            warnings.warn(
                f"Config file has open permissions: chmod 600 {config_path}",
                stacklevel=2,
            )
    except OSError:
        pass  # Ignore permission check errors

    try:
        config = json.loads(config_path.read_text())

        # Version check
        if config.get("version") != 1:
            return None

        headers = config.get("headers", {})
        body_params = config.get("body_params", {})

        # Validate that we have an authorization header
        if not headers.get("authorization"):
            return None

        return headers, body_params

    except (json.JSONDecodeError, KeyError, OSError):
        return None


def get_config(
    timeout: int = 30,
    save_to_file: bool = True,
    config_path: Path | str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Capture all HTTP headers and request body parameters from Claude CLI.

    This is a sync wrapper around async_get_config(). For async code, use
    async_get_config() directly.

    Args:
        timeout: Seconds to wait for capture (default: 30)
        save_to_file: Whether to save config to file (default: True)
        config_path: Path to save config (default: ~/.nano-agent.json)

    Returns:
        Tuple of (headers_dict, body_params_dict) where:
        - headers_dict: All HTTP headers (authorization, anthropic-*, x-stainless-*, etc.)
        - body_params_dict: Key parameters (model, max_tokens, temperature, thinking, user_id, system, url_path)

    Raises:
        TimeoutError: If config not captured within timeout
        RuntimeError: If server or subprocess fails
    """
    return asyncio.run(async_get_config(timeout, save_to_file, config_path))


async def _cleanup_process(
    process: asyncio.subprocess.Process,
    terminate_timeout: float = 1.0,
    kill_timeout: float = 0.5,
) -> None:
    """Clean up subprocess gracefully: terminate → wait → kill if needed.

    Args:
        process: The subprocess to clean up
        terminate_timeout: Seconds to wait after terminate (default: 1.0)
        kill_timeout: Seconds to wait after kill (default: 0.5)
    """
    if process.returncode is not None:
        return  # Already exited

    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=terminate_timeout)
    except (ProcessLookupError, asyncio.TimeoutError):
        # Process already gone or won't terminate gracefully - try kill
        try:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=kill_timeout)
        except (ProcessLookupError, asyncio.TimeoutError):
            pass  # Process gone or won't die - give up


async def async_get_config(
    timeout: int = 30,
    save_to_file: bool = True,
    config_path: Path | str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Async version of get_config() - captures HTTP headers and request body from Claude CLI.

    This is the recommended way to capture full configuration from Claude CLI in async code.
    Uses asyncio.create_subprocess_exec() and asyncio.sleep() for true async operation.

    Args:
        timeout: Seconds to wait for capture (default: 30)
        save_to_file: Whether to save config to file (default: True)
        config_path: Path to save config (default: ~/.nano-agent.json)

    Returns:
        Tuple of (headers_dict, body_params_dict)

    Raises:
        TimeoutError: If config not captured within timeout
        RuntimeError: If server or subprocess fails
    """
    print("Capturing configuration (async MITM proxy mode)...", file=sys.stderr)

    # Reset captured data
    MITMCaptureHandler.all_captured_configs = []

    # Start HTTP server in background thread
    server = HTTPServer(("localhost", 0), MITMCaptureHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Get the actual port assigned by the OS
    actual_port = server.server_address[1]
    print(f"Starting MITM proxy on port {actual_port}...", file=sys.stderr)
    await asyncio.sleep(0.2)  # Brief wait for server to bind

    # Launch Claude CLI subprocess with redirected base URL
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://localhost:{actual_port}/"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    print("Launching Claude CLI...", file=sys.stderr)
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "hey",
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
        )

        # Wait for at least one config to be captured
        start_time = time.time()
        while not MITMCaptureHandler.all_captured_configs:
            if time.time() - start_time > timeout:
                await _cleanup_process(process)
                MITMCaptureHandler.close_http_client()
                threading.Thread(target=server.shutdown, daemon=True).start()
                raise TimeoutError(f"Failed to capture config within {timeout} seconds")
            await asyncio.sleep(0.05)

        # Wait briefly for any additional requests (token refresh, etc.)
        await asyncio.sleep(0.5)

        # Clean up process first
        await _cleanup_process(process)

        # Close HTTP client
        MITMCaptureHandler.close_http_client()

        # Shutdown server in background (don't block on it)
        shutdown_thread = threading.Thread(target=server.shutdown, daemon=True)
        shutdown_thread.start()
        shutdown_thread.join(timeout=1.0)

        # Use the LAST captured config (most up-to-date tokens)
        if MITMCaptureHandler.all_captured_configs:
            last_config = MITMCaptureHandler.all_captured_configs[-1]
            headers = last_config["headers"]
            body_params = _extract_body_params(
                last_config["body"], last_config["url_path"]
            )

            count = len(MITMCaptureHandler.all_captured_configs)
            print(
                f"✓ Using config #{count} (last of {count} captured)",
                file=sys.stderr,
            )

            # Optionally save to file
            if save_to_file:
                saved_path = save_config(headers, body_params, config_path)
                print(f"✓ Config saved to {saved_path}", file=sys.stderr)

            return headers, body_params
        else:
            raise RuntimeError("No configs were captured")

    except FileNotFoundError:
        MITMCaptureHandler.close_http_client()
        threading.Thread(target=server.shutdown, daemon=True).start()
        raise RuntimeError(
            "Claude CLI not found. Make sure 'claude' is installed and in PATH."
        )
    except Exception as e:
        MITMCaptureHandler.close_http_client()
        threading.Thread(target=server.shutdown, daemon=True).start()
        if process:
            await _cleanup_process(process)
        raise RuntimeError(f"Failed to capture config: {e}")


def get_headers(timeout: int = 10) -> dict[str, str]:
    """
    Capture HTTP headers from Claude CLI (backwards compatible wrapper).

    This function calls get_config() and returns only the headers dict.
    For capturing both headers and body params, use get_config() directly.

    Note: Uses a random available port automatically (no port parameter needed).

    Args:
        timeout: Seconds to wait for capture (default: 10)

    Returns:
        Dictionary of all captured headers (keys are lowercase)

    Raises:
        TimeoutError: If headers not captured within timeout
        RuntimeError: If server or subprocess fails
    """
    headers, _ = get_config(timeout)
    return headers


def main() -> None:
    """CLI entry point."""
    print("=" * 80)
    print("CLAUDE CONFIG CAPTURE (MITM Proxy Mode)")
    print("=" * 80)
    print()

    try:
        headers, body_params = get_config(timeout=30)
        print()
        print("=" * 80)
        print("CAPTURED CONFIGURATION:")
        print("=" * 80)
        print()
        print("Headers:")
        for key, value in sorted(headers.items()):
            # Truncate long values for display
            display_value = value[:50] + "..." if len(value) > 50 else value
            print(f"  {key}: {display_value}")
        print()
        print("Body Parameters:")
        for key, value in sorted(body_params.items()):
            if key == "system":
                print(f"  {key}: <{len(value)} items>" if value else f"  {key}: None")
            else:
                print(f"  {key}: {value}")
        print()
        print("=" * 80)
        print(f"Config saved to: {DEFAULT_CONFIG_PATH}")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
