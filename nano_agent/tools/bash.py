"""Bash tool for executing shell commands."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from ..data_structures import TextContent
from .base import Desc, Tool

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


@dataclass
class BashInput:
    """Input for BashTool."""

    command: Annotated[str, Desc("The command to execute")]
    timeout: Annotated[int, Desc("Optional timeout in seconds (max 600)")] = 120
    description: Annotated[
        str, Desc("Clear, concise description of what this command does in 5-10 words")
    ] = ""
    run_in_background: Annotated[
        bool, Desc("Set to true to run this command in the background.")
    ] = False


@dataclass
class BashTool(Tool):
    """Executes a given bash command in a persistent shell session."""

    name: str = "Bash"
    description: str = """Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.

Usage notes:
  - The command argument is required.
  - You can specify an optional timeout in seconds (up to 600 seconds / 10 minutes).
  - If the output exceeds 30000 characters, output will be truncated."""

    async def __call__(
        self,
        input: BashInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Execute a bash command with cancellation-safe subprocess handling."""
        if not input.command:
            return TextContent(text="Error: No command provided")

        process = await asyncio.create_subprocess_shell(
            input.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=input.timeout
            )
            output = stdout.decode() or stderr.decode() or "(no output)"
            return TextContent(text=output)
        except asyncio.CancelledError:
            # Terminate subprocess on cancellation
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            raise  # Re-raise to propagate cancellation
        except asyncio.TimeoutError:
            # Terminate subprocess on timeout
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            return TextContent(text=f"Error: Command timed out after {input.timeout}s")
        except Exception as e:
            # Clean up subprocess on any other error
            if process.returncode is None:
                process.terminate()
            return TextContent(text=f"Error: {e}")
