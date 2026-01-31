"""Grep tool for content searching using ripgrep."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar

from ..data_structures import TextContent
from .base import Desc, Tool

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


@dataclass
class GrepInput:
    """Input for GrepTool.

    Note: The original API uses -B, -A, -C, -n, -i parameter names which are not
    valid Python identifiers. This dataclass uses Python-friendly names that map
    to the original parameters.
    """

    pattern: Annotated[
        str, Desc("The regular expression pattern to search for in file contents")
    ]
    path: Annotated[
        str,
        Desc("File or directory to search in. Defaults to current working directory."),
    ] = ""
    glob: Annotated[
        str, Desc('Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}")')
    ] = ""
    output_mode: Annotated[
        str, Desc("Output mode. Defaults to files_with_matches.")
    ] = "files_with_matches"
    context_before: Annotated[
        int, Desc("Number of lines to show before each match")
    ] = 0
    context_after: Annotated[int, Desc("Number of lines to show after each match")] = 0
    context: Annotated[
        int, Desc("Number of lines to show before and after each match")
    ] = 0
    line_numbers: Annotated[
        bool, Desc("Show line numbers in output. Defaults to true.")
    ] = True
    case_insensitive: Annotated[bool, Desc("Case insensitive search")] = False
    file_type: Annotated[
        str, Desc("File type to search (e.g., js, py, rust, go, java)")
    ] = ""
    head_limit: Annotated[
        int, Desc("Limit output to first N lines/entries. Defaults to 0 (unlimited).")
    ] = 0
    offset: Annotated[
        int,
        Desc("Skip first N lines/entries before applying head_limit. Defaults to 0."),
    ] = 0
    multiline: Annotated[bool, Desc("Enable multiline mode. Default: false.")] = False


@dataclass
class GrepTool(Tool):
    """A powerful search tool built on ripgrep."""

    name: str = "Grep"
    description: str = """A powerful search tool built on ripgrep.

Usage:
- Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
- Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
- Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts

Examples:
  # Find files containing a pattern (default: files_with_matches)
  GrepInput(pattern="async def", path="src/")

  # Show matching lines with line numbers
  GrepInput(pattern="class.*Tool", path="src/", output_mode="content")

  # Count matches per file
  GrepInput(pattern="TODO", output_mode="count")

  # Case-insensitive search with glob filter
  GrepInput(pattern="error", glob="*.py", case_insensitive=True)

  # Show context lines around matches
  GrepInput(pattern="def main", output_mode="content", context=2)

  # Limit results with pagination
  GrepInput(pattern="import", head_limit=10, offset=5)"""

    _required_commands: ClassVar[dict[str, str]] = {
        "rg": (
            "Install ripgrep: brew install ripgrep (macOS), "
            "apt install ripgrep (Ubuntu), pacman -S ripgrep (Arch)"
        )
    }

    async def __call__(
        self,
        input: GrepInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Execute grep search using ripgrep."""
        # Build ripgrep command
        cmd = ["rg"]

        # Output mode flags
        if input.output_mode == "files_with_matches":
            cmd.append("-l")
        elif input.output_mode == "count":
            cmd.append("-c")

        # Context flags: -C takes precedence over -B/-A
        if input.context > 0:
            cmd.extend(["-C", str(input.context)])
        else:
            if input.context_before > 0:
                cmd.extend(["-B", str(input.context_before)])
            if input.context_after > 0:
                cmd.extend(["-A", str(input.context_after)])

        # Line numbers (only meaningful for content output mode)
        if input.line_numbers and input.output_mode == "content":
            cmd.append("-n")

        # Case insensitive search
        if input.case_insensitive:
            cmd.append("-i")

        # Multiline mode (match across lines)
        if input.multiline:
            cmd.extend(["-U", "--multiline-dotall"])

        # Glob pattern filter
        if input.glob:
            cmd.extend(["--glob", input.glob])

        # File type filter
        if input.file_type:
            cmd.extend(["--type", input.file_type])

        # Pattern (required) and path (optional, defaults to current directory)
        cmd.append(input.pattern)
        cmd.append(input.path or ".")

        # Run ripgrep
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            output = stdout.decode()

            # Apply offset and head_limit if specified (post-processing)
            if input.offset > 0 or input.head_limit > 0:
                lines = output.splitlines()
                if input.offset > 0:
                    lines = lines[input.offset :]
                if input.head_limit > 0:
                    lines = lines[: input.head_limit]
                output = "\n".join(lines)

            # Handle no matches (ripgrep exit code 1 means no matches found)
            if process.returncode == 1 and not output:
                return TextContent(text="No matches found")

            # Handle actual errors (exit code 2+ indicates an error)
            if process.returncode is not None and process.returncode >= 2:
                return TextContent(text=f"Error: {stderr.decode() or 'ripgrep failed'}")

            return TextContent(text=output or "No matches found")

        except FileNotFoundError:
            return TextContent(text="Error: ripgrep (rg) not found. Please install it.")
        except Exception as e:
            return TextContent(text=f"Error: {e}")
