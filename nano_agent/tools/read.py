"""Read tool for reading file contents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

from ..data_structures import TextContent
from .base import Desc, Tool, TruncationConfig, _format_size

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


@dataclass
class ReadInput:
    """Input for ReadTool."""

    file_path: Annotated[str, Desc("The absolute path to the file to read")]
    offset: Annotated[int, Desc("The line number to start reading from")] = 0
    limit: Annotated[int, Desc("The number of lines to read")] = 0


@dataclass
class ReadTool(Tool):
    """Reads a file from the local filesystem."""

    name: str = "Read"
    description: str = """Reads a file from the local filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- Maximum 25 lines per read (use offset to paginate through larger files)
- Output includes metadata: file path, size, total lines, range shown

Workflow for large files:
1. First read without offset to see file overview and metadata
2. Use Grep to find relevant line numbers for specific patterns
3. Use offset to read focused sections (e.g., offset=100 reads lines 101-125)

Examples:
  ReadInput(file_path="/path/to/file.py")              # Lines 1-25
  ReadInput(file_path="/path/to/file.py", offset=100)  # Lines 101-125
  ReadInput(file_path="/path/to/file.py", offset=50, limit=10)  # Lines 51-60

Note: For binary files (images, PDFs), content is processed differently."""

    # Constants for file reading
    MAX_LINES: ClassVar[int] = 25
    MAX_LINE_LENGTH: ClassVar[int] = 2000

    # Disable truncation - ReadTool already limits to MAX_LINES
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(enabled=False)

    async def __call__(
        self,
        input: ReadInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Read file contents with metadata and smart defaults."""
        path = Path(input.file_path)

        # Validate file exists
        if not path.exists():
            return TextContent(text=f"Error: File not found: {input.file_path}")
        if not path.is_file():
            return TextContent(text=f"Error: Not a file: {input.file_path}")

        try:
            # Get file metadata
            file_size = path.stat().st_size
            size_str = _format_size(file_size)

            # Read content
            content = path.read_text()
            lines = content.splitlines()
            total_lines = len(lines)

            # Apply offset and limit
            start = input.offset if input.offset > 0 else 0

            # Cap limit at MAX_LINES (use MAX_LINES if not specified or if exceeds max)
            effective_limit = (
                min(input.limit, self.MAX_LINES) if input.limit > 0 else self.MAX_LINES
            )
            end = min(start + effective_limit, total_lines)

            selected_lines = lines[start:end]

            # Truncate very long lines
            selected_lines = [
                (
                    line[: self.MAX_LINE_LENGTH] + "..."
                    if len(line) > self.MAX_LINE_LENGTH
                    else line
                )
                for line in selected_lines
            ]

            # Format with line numbers (1-indexed)
            numbered_lines = [
                f"{i + start + 1:6}\t{line}" for i, line in enumerate(selected_lines)
            ]

            # Build metadata header
            showing = f"{start + 1}-{end}"
            if end < total_lines:
                showing += f" of {total_lines}"

            header = f"Size: {size_str} | Lines: {total_lines} | Showing: {showing}"

            # Build output
            output_parts = [header, "\n".join(numbered_lines)]

            # Add truncation notice if applicable (only when using default limit)
            if end < total_lines and input.limit == 0:
                output_parts.append(
                    "\n<tool-warning>File truncated. Use offset/limit to read more, "
                    "or Grep to find specific sections.</tool-warning>"
                )

            return TextContent(text="\n".join(output_parts))

        except PermissionError:
            return TextContent(text=f"Error: Permission denied: {input.file_path}")
        except UnicodeDecodeError:
            return TextContent(
                text=f"Error: Cannot read binary file as text: {input.file_path}\n"
                "This appears to be a binary file (image, PDF, etc.)."
            )
        except Exception as e:
            return TextContent(text=f"Error reading file: {e}")
