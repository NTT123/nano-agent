"""Glob tool for file pattern matching."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar

from ..data_structures import TextContent
from .base import Desc, Tool

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


@dataclass
class GlobInput:
    """Input for GlobTool."""

    pattern: Annotated[str, Desc("The glob pattern to match files against")]
    path: Annotated[
        str, Desc("The directory to search in. Defaults to current working directory.")
    ] = ""


@dataclass
class GlobTool(Tool):
    """Fast file pattern matching tool that works with any codebase size."""

    name: str = "Glob"
    description: str = """Fast file pattern matching tool using fd.

Supports glob patterns to find files by name. Results are sorted by modification time (most recent first).

Common patterns:
  "*.py"           - All Python files in current directory
  "**/*.py"        - All Python files recursively
  "test_*.py"      - All test files
  "*.{js,ts}"      - All JavaScript and TypeScript files
  "src/**/*.tsx"   - All TSX files under src/

Examples:
  GlobInput(pattern="*.py")                      # Find all .py files
  GlobInput(pattern="**/*.py", path="src/")      # Find .py files in src/
  GlobInput(pattern="test_*")                    # Find all test files
  GlobInput(pattern="*.{js,ts,jsx,tsx}")         # Find all JS/TS files

Note: Requires 'fd' to be installed (brew install fd)."""

    _required_commands: ClassVar[dict[str, str]] = {
        "fd": (
            "Install with: brew install fd (macOS), "
            "apt install fd-find (Ubuntu), pacman -S fd (Arch)"
        )
    }

    async def __call__(
        self,
        input: GlobInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Execute glob pattern matching using fd."""
        path = input.path or "."
        pattern = input.pattern

        # Build fd command
        # -g: glob mode, -t f: files only, -a: absolute paths
        cmd = ["fd", "-g", pattern, "-t", "f", "-a", path]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0 and not stdout:
                if stderr:
                    return TextContent(text=f"Error: {stderr.decode()}")
                return TextContent(text="No matches found")

            if not stdout:
                return TextContent(text="No matches found")

            # Parse results
            files = stdout.decode().strip().splitlines()

            # Sort by modification time (most recent first)
            files_with_mtime = []
            for f in files:
                try:
                    mtime = os.path.getmtime(f)
                    files_with_mtime.append((mtime, f))
                except OSError:
                    continue

            files_with_mtime.sort(reverse=True)  # Most recent first
            sorted_files = [f for _, f in files_with_mtime]

            return TextContent(text="\n".join(sorted_files) or "No matches found")

        except Exception as e:
            return TextContent(text=f"Error: {e}")
