"""Write tool for creating and overwriting files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from ..data_structures import TextContent
from .base import Desc, Tool, _format_size

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


@dataclass
class WriteInput:
    """Input for WriteTool."""

    file_path: Annotated[
        str,
        Desc("The absolute path to the file to write (must be absolute, not relative)"),
    ]
    content: Annotated[str, Desc("The content to write to the file")]


@dataclass
class WriteTool(Tool):
    """Writes a file to the local filesystem."""

    name: str = "Write"
    description: str = """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Creates parent directories if they don't exist."""

    async def __call__(
        self,
        input: WriteInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Write content to file."""
        path = Path(input.file_path)

        # Create parent directories if needed
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return TextContent(
                text=f"Error: Permission denied creating directory: {path.parent}"
            )

        # Check if file exists (for info message)
        existed = path.exists()

        try:
            path.write_text(input.content)
        except PermissionError:
            return TextContent(text=f"Error: Permission denied: {input.file_path}")
        except Exception as e:
            return TextContent(text=f"Error writing file: {e}")

        # Count lines and size for confirmation
        line_count = input.content.count("\n") + (
            1 if input.content and not input.content.endswith("\n") else 0
        )
        size_str = _format_size(len(input.content.encode("utf-8")))

        action = "Overwritten" if existed else "Created"
        return TextContent(
            text=f"✓ {action}: {input.file_path}\n" f"  {line_count} lines, {size_str}"
        )
