"""Codex-style apply_patch tool for editing files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

from ..data_structures import TextContent
from ..execution_context import ExecutionContext
from .base import Desc, Tool, TruncationConfig

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD = "*** Add File: "
_DELETE = "*** Delete File: "
_UPDATE = "*** Update File: "
_MOVE = "*** Move to: "
_EOF = "*** End of File"

_APPLY_PATCH_DESCRIPTION = """Use the `apply_patch` tool to edit files.
Your patch language is a stripped‑down, file‑oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high‑level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more "hunks", each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ \tdef method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- File references can only be relative, NEVER ABSOLUTE.
"""


@dataclass
class ApplyPatchInput:
    """Input for ApplyPatchTool."""

    input: Annotated[str, Desc("The entire contents of the apply_patch command")]


@dataclass
class _Chunk:
    context: str | None
    old_lines: list[str]
    new_lines: list[str]
    eof: bool = False


@dataclass
class _AddFile:
    path: Path
    lines: list[str]


@dataclass
class _DeleteFile:
    path: Path


@dataclass
class _UpdateFile:
    path: Path
    move_path: Path | None
    chunks: list[_Chunk]


_Operation = _AddFile | _DeleteFile | _UpdateFile


@dataclass
class ApplyPatchTool(Tool):
    """Apply a Codex apply_patch-formatted file patch."""

    name: str = "apply_patch"
    description: str = _APPLY_PATCH_DESCRIPTION

    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(enabled=False)

    async def __call__(
        self,
        input: ApplyPatchInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Parse and apply a Codex patch."""
        try:
            operations = _parse_patch(input.input)
            summary = _apply_operations(operations, Path.cwd())
            return TextContent(text=summary)
        except ValueError as exc:
            return TextContent(text=f"Invalid patch: {exc}")
        except OSError as exc:
            return TextContent(text=f"Error applying patch: {exc}")


def _parse_patch(raw_patch: str) -> list[_Operation]:
    patch = _strip_heredoc(raw_patch).strip()
    lines = patch.splitlines()
    if len(lines) < 3:
        raise ValueError(
            "expected begin marker, at least one operation, and end marker"
        )
    if lines[0].strip() != _BEGIN:
        raise ValueError(f"expected {_BEGIN!r}")
    if lines[-1].strip() != _END:
        raise ValueError(f"expected {_END!r}")

    operations: list[_Operation] = []
    index = 1
    end_index = len(lines) - 1
    while index < end_index:
        line = lines[index]
        operation: _Operation
        if line.startswith(_ADD):
            operation, index = _parse_add(lines, index, end_index)
        elif line.startswith(_DELETE):
            operation, index = _parse_delete(lines, index)
        elif line.startswith(_UPDATE):
            operation, index = _parse_update(lines, index, end_index)
        else:
            raise ValueError(f"expected file operation at line {index + 1}: {line}")
        operations.append(operation)

    if not operations:
        raise ValueError("no file operations found")
    return operations


def _strip_heredoc(raw_patch: str) -> str:
    lines = raw_patch.strip().splitlines()
    if len(lines) >= 4 and lines[0].startswith("<<") and lines[-1].strip() == "EOF":
        return "\n".join(lines[1:-1])
    return raw_patch


def _parse_add(
    lines: list[str],
    index: int,
    end_index: int,
) -> tuple[_AddFile, int]:
    path = _parse_relative_path(lines[index].removeprefix(_ADD))
    index += 1
    contents: list[str] = []
    while index < end_index and not _is_operation_line(lines[index]):
        line = lines[index]
        if not line.startswith("+"):
            raise ValueError(f"add file line {index + 1} must start with '+'")
        contents.append(line[1:])
        index += 1
    return _AddFile(path=path, lines=contents), index


def _parse_delete(lines: list[str], index: int) -> tuple[_DeleteFile, int]:
    return (
        _DeleteFile(path=_parse_relative_path(lines[index].removeprefix(_DELETE))),
        index + 1,
    )


def _parse_update(
    lines: list[str],
    index: int,
    end_index: int,
) -> tuple[_UpdateFile, int]:
    path = _parse_relative_path(lines[index].removeprefix(_UPDATE))
    index += 1
    move_path: Path | None = None
    if index < end_index and lines[index].startswith(_MOVE):
        move_path = _parse_relative_path(lines[index].removeprefix(_MOVE))
        index += 1

    chunks: list[_Chunk] = []
    while index < end_index and not _is_operation_line(lines[index]):
        line = lines[index]
        if line == "@@":
            context = None
        elif line.startswith("@@ "):
            context = line.removeprefix("@@ ")
        else:
            raise ValueError(f"expected @@ hunk marker at line {index + 1}: {line}")
        index += 1

        old_lines: list[str] = []
        new_lines: list[str] = []
        eof = False
        while index < end_index and not _is_operation_line(lines[index]):
            hunk_line = lines[index]
            if hunk_line == _EOF:
                eof = True
                index += 1
                break
            if hunk_line.startswith("@@"):
                break
            if not hunk_line:
                raise ValueError(f"hunk line {index + 1} is missing a prefix")
            prefix, text = hunk_line[0], hunk_line[1:]
            if prefix == " ":
                old_lines.append(text)
                new_lines.append(text)
            elif prefix == "-":
                old_lines.append(text)
            elif prefix == "+":
                new_lines.append(text)
            else:
                raise ValueError(f"hunk line {index + 1} has invalid prefix {prefix!r}")
            index += 1
        chunks.append(
            _Chunk(context=context, old_lines=old_lines, new_lines=new_lines, eof=eof)
        )

    if not chunks:
        raise ValueError(f"update file hunk for path {path!s} is empty")
    return _UpdateFile(path=path, move_path=move_path, chunks=chunks), index


def _is_operation_line(line: str) -> bool:
    return line.startswith((_ADD, _DELETE, _UPDATE))


def _parse_relative_path(value: str) -> Path:
    # Codex-rs accepts absolute paths because it runs inside a filesystem
    # sandbox; nano-agent has no sandbox layer, so reject absolute paths to
    # keep apply_patch contained to the working directory.
    if not value:
        raise ValueError("file path cannot be empty")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {value}")
    return path


def _apply_operations(operations: list[_Operation], cwd: Path) -> str:
    added: list[Path] = []
    modified: list[Path] = []
    deleted: list[Path] = []
    resolved_cwd = cwd.resolve()

    for operation in operations:
        if isinstance(operation, _AddFile):
            path = _resolve(resolved_cwd, operation.path)
            if path.exists():
                raise ValueError(f"file already exists: {operation.path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_lines_to_content(operation.lines))
            added.append(operation.path)
        elif isinstance(operation, _DeleteFile):
            path = _resolve(resolved_cwd, operation.path)
            if not path.is_file():
                raise ValueError(f"file to delete does not exist: {operation.path}")
            path.unlink()
            deleted.append(operation.path)
        else:
            path = _resolve(resolved_cwd, operation.path)
            if not path.is_file():
                raise ValueError(f"file to update does not exist: {operation.path}")
            original = path.read_text()
            new_content = _apply_chunks(original, operation.chunks, operation.path)
            if operation.move_path is not None:
                dest = _resolve(resolved_cwd, operation.move_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(new_content)
                path.unlink()
            else:
                path.write_text(new_content)
            modified.append(operation.path)

    return _format_summary(added=added, modified=modified, deleted=deleted)


def _resolve(resolved_cwd: Path, path: Path) -> Path:
    resolved_path = (resolved_cwd / path).resolve()
    if not resolved_path.is_relative_to(resolved_cwd):
        raise ValueError(f"path escapes working directory: {path}")
    return resolved_path


def _lines_to_content(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _content_to_lines(content: str) -> list[str]:
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _apply_chunks(content: str, chunks: list[_Chunk], display_path: Path) -> str:
    lines = _content_to_lines(content)
    replacements: list[tuple[int, int, list[str]]] = []
    line_index = 0

    for chunk in chunks:
        if chunk.context is not None:
            context_index = _seek_sequence(
                lines, [chunk.context], line_index, eof=False
            )
            if context_index is None:
                raise ValueError(
                    f"failed to find context {chunk.context!r} in {display_path}"
                )
            line_index = context_index + 1

        if not chunk.old_lines:
            replacements.append((len(lines), 0, chunk.new_lines))
            continue

        old_lines = chunk.old_lines
        new_lines = chunk.new_lines
        start = _seek_sequence(lines, old_lines, line_index, eof=chunk.eof)
        if start is None and old_lines and old_lines[-1] == "":
            old_lines = old_lines[:-1]
            new_lines = (
                new_lines[:-1] if new_lines and new_lines[-1] == "" else new_lines
            )
            start = _seek_sequence(lines, old_lines, line_index, eof=chunk.eof)
        if start is None:
            raise ValueError(
                f"failed to find expected lines in {display_path}:\n"
                + "\n".join(chunk.old_lines)
            )
        replacements.append((start, len(old_lines), new_lines))
        line_index = start + len(old_lines)

    for start, old_len, new_lines in sorted(replacements, reverse=True):
        lines[start : start + old_len] = new_lines
    return _lines_to_content(lines)


def _seek_sequence(
    lines: list[str],
    needle: list[str],
    start: int,
    eof: bool,
) -> int | None:
    if not needle:
        return start
    upper = len(lines) - len(needle)
    if upper < start:
        return None
    if eof:
        return upper if lines[upper : upper + len(needle)] == needle else None
    if len(needle) == 1:
        try:
            return lines.index(needle[0], start)
        except ValueError:
            return None
    for index in range(start, upper + 1):
        if lines[index : index + len(needle)] == needle:
            return index
    return None


def _format_summary(
    added: list[Path],
    modified: list[Path],
    deleted: list[Path],
) -> str:
    lines = ["Success. Updated the following files:"]
    lines.extend(f"A {path}" for path in added)
    lines.extend(f"M {path}" for path in modified)
    lines.extend(f"D {path}" for path in deleted)
    return "\n".join(lines)
