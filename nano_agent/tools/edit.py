"""Edit tool for file modifications with preview and permission prompt."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from ..data_structures import TextContent
from .base import Desc, Tool


def _find_match_line(content: str, old_string: str) -> tuple[int, int]:
    """Find the line number where old_string starts.

    Returns:
        Tuple of (line_number, char_position) where line_number is 0-indexed.
    """
    pos = content.find(old_string)
    if pos == -1:
        return (-1, -1)
    # Count newlines before the match position
    line_num = content[:pos].count("\n")
    return (line_num, pos)


def _generate_preview(
    lines: list[str],
    match_line: int,
    old_string: str,
    new_string: str,
    context_lines: int = 3,
) -> tuple[str, list[str], list[str]]:
    """Generate a unified diff-style preview of the edit.

    Returns:
        Tuple of (preview_text, context_before, context_after)
    """
    import difflib

    total_lines = len(lines)
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()

    # Calculate the range we're working with
    start = max(0, match_line - context_lines)
    old_end = match_line + len(old_lines)
    end = min(total_lines, old_end + context_lines)

    context_before = lines[start:match_line]
    context_after = lines[old_end:end]

    # Build the old and new versions of the affected region
    old_region = lines[start:old_end]
    new_region = lines[start:match_line] + new_lines + lines[old_end:end]

    # Use difflib to generate unified diff
    diff = list(
        difflib.unified_diff(old_region, new_region, lineterm="", n=context_lines)
    )

    # Skip the header lines (--- and +++ and @@)
    # and format with line numbers
    preview_parts = []
    old_line_num = start + 1
    new_line_num = start + 1

    for line in diff:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        elif line.startswith("-"):
            preview_parts.append(f"  -  {old_line_num:6}\t{line[1:]}")
            old_line_num += 1
        elif line.startswith("+"):
            preview_parts.append(f"  +  {new_line_num:6}\t{line[1:]}")
            new_line_num += 1
        elif line.startswith(" "):
            preview_parts.append(f"     {old_line_num:6}\t{line[1:]}")
            old_line_num += 1
            new_line_num += 1
        else:
            # Empty line in diff
            preview_parts.append(f"     {old_line_num:6}\t{line}")
            old_line_num += 1
            new_line_num += 1

    return "\n".join(preview_parts), context_before, context_after


# =============================================================================
# Input Dataclasses
# =============================================================================


@dataclass
class EditInput:
    """Input for EditTool."""

    file_path: Annotated[str, Desc("The absolute path to the file to modify")]
    old_string: Annotated[str, Desc("The text to replace")]
    new_string: Annotated[
        str, Desc("The text to replace it with (must be different from old_string)")
    ]
    replace_all: Annotated[
        bool, Desc("Replace all occurences of old_string (default false)")
    ] = False


# =============================================================================
# Tool Classes
# =============================================================================

# Type alias for permission callback
PermissionCallback = Callable[[str, str, int], Awaitable[bool]]


@dataclass
class EditTool(Tool):
    """Performs exact string replacements in files with permission prompt.

    Shows a preview and asks for user permission before applying changes.
    If no permission_callback is provided, edits are auto-approved (non-interactive).
    """

    name: str = "Edit"
    description: str = """Performs exact string replacements in files.

Usage:
- You must use your Read tool at least once in the conversation before editing.
- Shows a preview and asks for user permission before applying.
- The edit will FAIL if old_string is not unique in the file (unless replace_all=True).
- Use replace_all=True for replacing all occurrences across the file."""

    # Optional callback for interactive permission prompts
    # Signature: async (file_path, preview, match_count) -> bool
    permission_callback: PermissionCallback | None = field(default=None, repr=False)

    async def __call__(self, input: EditInput) -> TextContent:
        """Validate edit, show preview, ask permission, and apply if approved."""
        path = Path(input.file_path)

        # Validate file exists
        if not path.exists():
            return TextContent(text=f"Error: File not found: {input.file_path}")
        if not path.is_file():
            return TextContent(text=f"Error: Not a file: {input.file_path}")

        try:
            content = path.read_text()
        except PermissionError:
            return TextContent(text=f"Error: Permission denied: {input.file_path}")
        except UnicodeDecodeError:
            return TextContent(
                text=f"Error: Cannot edit binary file: {input.file_path}"
            )

        # Validate old_string exists
        if input.old_string not in content:
            return TextContent(
                text=f"Error: old_string not found in {input.file_path}\n\n"
                "The text you're trying to replace does not exist in the file. "
                "Use the Read tool to verify the exact content."
            )

        # Check uniqueness (unless replace_all is True)
        match_count = content.count(input.old_string)
        if match_count > 1 and not input.replace_all:
            return TextContent(
                text=f"Error: old_string is not unique in {input.file_path}\n\n"
                f"Found {match_count} occurrences. Either:\n"
                "1. Provide more context to make old_string unique, or\n"
                "2. Set replace_all=True to replace all occurrences."
            )

        # Validate old_string != new_string
        if input.old_string == input.new_string:
            return TextContent(
                text="Error: old_string and new_string are identical. No changes to make."
            )

        # Find match location and generate preview
        lines = content.splitlines()
        match_line, _ = _find_match_line(content, input.old_string)

        preview, _, _ = _generate_preview(
            lines, match_line, input.old_string, input.new_string
        )

        # Build preview header
        header = f"─── Edit Preview: {input.file_path} ───\n"
        if match_count > 1:
            header += f"Replacing ALL {match_count} occurrences\n"
        else:
            header += f"Match found at line {match_line + 1}\n"
        header += "\n"

        full_preview = header + preview

        # Ask for permission if callback provided
        if self.permission_callback:
            allowed = await self.permission_callback(
                input.file_path, full_preview, match_count
            )
            if not allowed:
                return TextContent(text="Edit rejected by user.")

        # Apply the edit
        if input.replace_all:
            new_content = content.replace(input.old_string, input.new_string)
        else:
            new_content = content.replace(input.old_string, input.new_string, 1)

        try:
            path.write_text(new_content)
        except PermissionError:
            return TextContent(text=f"Error: Permission denied: {input.file_path}")
        except Exception as e:
            return TextContent(text=f"Error writing file: {e}")

        # Return success message
        if input.replace_all and match_count > 1:
            return TextContent(
                text=f"✓ Edit applied: {input.file_path}\n"
                f"  Replaced {match_count} occurrences."
            )
        else:
            return TextContent(
                text=f"✓ Edit applied: {input.file_path}\n"
                f"  Modified at line {match_line + 1}."
            )
