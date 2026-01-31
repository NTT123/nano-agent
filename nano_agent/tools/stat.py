"""Stat tool for getting file metadata."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from ..data_structures import TextContent
from .base import Desc, Tool, _format_size

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


@dataclass
class StatInput:
    """Input for StatTool."""

    file_path: Annotated[str, Desc("Absolute path to file or directory")]


@dataclass
class StatTool(Tool):
    """Get file metadata without reading content."""

    name: str = "Stat"
    description: str = """Get file or directory metadata without reading content.

Returns:
- File type (via 'file' command)
- Size in human-readable format
- Line count (for text files)
- Last modified timestamp
- Permissions

Useful for understanding files before deciding to read them, especially for
large files where you want to know the size first.

Examples:
  StatInput(file_path="/path/to/file.py")
  StatInput(file_path="/path/to/directory")

Note: Requires 'file' and 'wc' commands (standard on Unix systems)."""

    async def __call__(
        self,
        input: StatInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Get file metadata."""
        path = Path(input.file_path)

        if not path.exists():
            return TextContent(text=f"Error: Not found: {input.file_path}")

        try:
            stat_info = path.stat()
        except PermissionError:
            return TextContent(text=f"Error: Permission denied: {input.file_path}")

        # Basic metadata
        size = _format_size(stat_info.st_size)
        modified = datetime.fromtimestamp(stat_info.st_mtime).isoformat(sep=" ")
        permissions = oct(stat_info.st_mode)[-3:]

        # Determine type
        if path.is_dir():
            type_str = "directory"
            line_count = None
        elif path.is_symlink():
            target = path.resolve()
            type_str = f"symlink → {target}"
            line_count = None
        else:
            # Get file type via 'file' command
            try:
                process = await asyncio.create_subprocess_exec(
                    "file",
                    "-b",
                    str(path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate()
                type_str = stdout.decode().strip() or "unknown"
            except Exception:
                type_str = "file"

            # Get line count for text files
            line_count = None
            if "text" in type_str.lower() or path.suffix in (
                ".py",
                ".js",
                ".ts",
                ".json",
                ".yaml",
                ".yml",
                ".md",
                ".txt",
                ".html",
                ".css",
                ".sh",
                ".bash",
                ".zsh",
                ".toml",
                ".ini",
                ".cfg",
                ".xml",
                ".csv",
                ".sql",
                ".rs",
                ".go",
                ".java",
                ".c",
                ".cpp",
                ".h",
            ):
                try:
                    process = await asyncio.create_subprocess_exec(
                        "wc",
                        "-l",
                        str(path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await process.communicate()
                    # wc -l output: "  123 /path/to/file"
                    count_str = stdout.decode().strip().split()[0]
                    line_count = int(count_str)
                except Exception:
                    pass

        # Build output
        output_parts = [f"─── Stat: {input.file_path} ───"]
        output_parts.append(f"Type: {type_str}")
        output_parts.append(f"Size: {size}")
        if line_count is not None:
            output_parts.append(f"Lines: {line_count}")
        output_parts.append(f"Modified: {modified}")
        output_parts.append(f"Permissions: {permissions}")

        return TextContent(text="\n".join(output_parts))
