"""WebFetch tool for fetching and rendering web content."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar

from ..data_structures import TextContent
from .base import Desc, Tool, TruncationConfig

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


@dataclass
class WebFetchInput:
    """Input for WebFetchTool."""

    url: Annotated[str, Desc("The URL to fetch content from")]
    prompt: Annotated[str, Desc("The prompt to run on the fetched content")]


@dataclass
class WebFetchTool(Tool):
    """Fetches content from a URL and renders it as clean text using lynx."""

    name: str = "WebFetch"
    description: str = """Fetches content from a URL and renders it as clean, readable text.

Uses lynx (text-mode browser) to render HTML pages as plain text, extracting
readable content without raw HTML tags.

Usage:
- URL must be a fully-formed valid URL (https://...)
- HTTP URLs are automatically upgraded to HTTPS
- Output is truncated to 5000 characters maximum
- The prompt parameter describes what to look for in the content

Examples:
  WebFetchInput(url="https://example.com", prompt="Summarize the page content")
  WebFetchInput(url="https://docs.python.org/3/", prompt="Find the tutorial section")

Note: Requires 'lynx' to be installed (brew install lynx)."""

    # Use centralized truncation with 5000 char limit (saves full output to temp file)
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(max_chars=5000)
    _required_commands: ClassVar[dict[str, str]] = {
        "lynx": (
            "Install lynx: brew install lynx (macOS), "
            "apt install lynx (Ubuntu), pacman -S lynx (Arch)"
        )
    }

    async def __call__(
        self,
        input: WebFetchInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Fetch URL content using lynx and return as text."""
        url = input.url

        # Upgrade HTTP to HTTPS
        if url.startswith("http://"):
            url = "https://" + url[7:]

        # Basic URL validation
        if not url.startswith("https://"):
            return TextContent(
                text="Error: Invalid URL. Must start with http:// or https://"
            )

        # Build lynx command
        # -dump: output to stdout
        # -nolist: don't print link list at bottom (cleaner output)
        # -width=120: reasonable line width
        # -accept_all_cookies: handle cookie prompts
        cmd = [
            "lynx",
            "-dump",
            "-nolist",
            "-width=120",
            "-accept_all_cookies",
            url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

            if process.returncode != 0:
                error_msg = stderr.decode(errors="replace").strip() or "Unknown error"
                return TextContent(text=f"Error fetching URL: {error_msg}")

            text = stdout.decode(errors="replace")

            if not text.strip():
                return TextContent(text="Error: Page returned empty content")

            # Build output with metadata
            header = f"─── WebFetch: {input.url} ───\n"
            header += f"Prompt: {input.prompt}\n\n"

            return TextContent(text=header + text)

        except asyncio.TimeoutError:
            return TextContent(text="Error: Request timed out after 30 seconds")
        except Exception as e:
            return TextContent(text=f"Error: {e}")
