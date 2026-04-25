"""Codex-style exec_command tool for running shell commands."""

from __future__ import annotations

import asyncio
import itertools
import os
import pty
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar

from ..data_structures import APPROX_BYTES_PER_TOKEN, TextContent
from ..execution_context import ExecutionContext
from .base import Desc, Tool, TruncationConfig, terminate_process

_DEFAULT_MAX_OUTPUT_TOKENS = 10_000
_DEFAULT_WRITE_STDIN_YIELD_TIME_MS = 250
_MIN_YIELD_TIME_MS = 250
_MIN_EMPTY_WRITE_STDIN_YIELD_TIME_MS = 5_000
_MAX_YIELD_TIME_MS = 30_000
_MAX_BACKGROUND_TERMINAL_TIMEOUT_MS = 300_000
_READ_CHUNK_SIZE = 4096
_POST_WRITE_REACT_DELAY = 0.1
# Cap live sessions so a chatty agent can't grow the registry unbounded —
# zombies (process exited but never finalized) get reaped first; if still at
# the cap, new exec_command calls are rejected with an error.
_MAX_LIVE_SESSIONS = 64


@dataclass
class ExecCommandInput:
    """Input for ExecCommandTool."""

    cmd: Annotated[str, Desc("Shell command to execute.")]
    workdir: Annotated[
        str,
        Desc(
            "Optional working directory to run the command in; defaults to the turn cwd."
        ),
    ] = ""
    shell: Annotated[
        str, Desc("Shell binary to launch. Defaults to the user's default shell.")
    ] = ""
    tty: Annotated[
        bool,
        Desc(
            "Whether to allocate a TTY for the command. Defaults to false "
            "(plain pipes); set to true to open a PTY and access TTY process."
        ),
    ] = False
    yield_time_ms: Annotated[
        int, Desc("How long to wait (in milliseconds) for output before yielding.")
    ] = 10_000
    max_output_tokens: Annotated[
        int,
        Desc("Maximum number of tokens to return. Excess output will be truncated."),
    ] = 0
    login: Annotated[
        bool, Desc("Whether to run the shell with -l/-i semantics. Defaults to true.")
    ] = True
    sandbox_permissions: Annotated[
        str,
        Desc(
            'Sandbox permissions for the command. Set to "require_escalated" '
            'to request running without sandbox restrictions; defaults to "use_default".'
        ),
    ] = "use_default"
    justification: Annotated[
        str,
        Desc(
            'Only set if sandbox_permissions is "require_escalated". Request '
            "approval from the user to run this command outside the sandbox. "
            "Phrased as a simple question that summarizes the purpose of the "
            "command as it relates to the task at hand."
        ),
    ] = ""
    prefix_rule: Annotated[
        list[str],
        Desc(
            "Only specify when sandbox_permissions is `require_escalated`. "
            "Suggest a prefix command pattern that will allow you to fulfill "
            "similar requests from the user in the future."
        ),
    ] = field(default_factory=list)


@dataclass
class WriteStdinInput:
    """Input for WriteStdinTool."""

    session_id: Annotated[int, Desc("Identifier of the running unified exec session.")]
    chars: Annotated[str, Desc("Bytes to write to stdin (may be empty to poll).")] = ""
    yield_time_ms: Annotated[
        int, Desc("How long to wait (in milliseconds) for output before yielding.")
    ] = _DEFAULT_WRITE_STDIN_YIELD_TIME_MS
    max_output_tokens: Annotated[
        int,
        Desc("Maximum number of tokens to return. Excess output will be truncated."),
    ] = 0


@dataclass
class _ExecSession:
    session_id: int
    process: asyncio.subprocess.Process
    # Buffer of unreported output. Reader tasks append; _format_response drains
    # the consumed prefix on each call so a long polling loop stays O(n) total
    # rather than re-joining and re-decoding the full history every poll.
    chunks: list[bytes]
    reader_tasks: list[asyncio.Task[None]]
    started_at: float
    pty_master_fd: int | None = None

    def consume_pending(self) -> bytes:
        """Snapshot and drop the currently buffered chunks atomically.

        Reader tasks may append to ``chunks`` concurrently, so we capture the
        count first and only remove that many entries; anything appended after
        is preserved for the next call.
        """
        n = len(self.chunks)
        consumed = b"".join(self.chunks[:n])
        del self.chunks[:n]
        return consumed


_session_counter = itertools.count(1)
_exec_sessions: dict[int, _ExecSession] = {}


@dataclass
class ExecCommandTool(Tool):
    """Run a shell command using the Codex CLI exec_command shape."""

    name: str = "exec_command"
    description: str = (
        "Runs a command in a PTY, returning output or a session ID for ongoing "
        "interaction."
    )

    # exec_command applies Codex-style token truncation itself.
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(enabled=False)

    async def __call__(
        self,
        input: ExecCommandInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Execute a shell command and return Codex-style output text."""
        if not input.cmd.strip():
            return TextContent(text="Error: No command provided")

        permissions = input.sandbox_permissions or "use_default"
        if permissions not in {"use_default", "require_escalated"}:
            return TextContent(
                text=(
                    "Error: sandbox_permissions must be 'use_default' or "
                    f"'require_escalated', got {permissions!r}"
                )
            )

        try:
            cwd = _resolve_workdir(input.workdir)
        except ValueError as exc:
            return TextContent(text=f"Error: {exc}")

        _reap_finished_sessions()
        if len(_exec_sessions) >= _MAX_LIVE_SESSIONS:
            return TextContent(
                text=(
                    f"Error: too many live exec_command sessions "
                    f"(cap {_MAX_LIVE_SESSIONS}). Use write_stdin to interact "
                    "with existing sessions or wait for them to exit."
                )
            )

        command = _shell_command(input.cmd, input.shell, input.login)
        session_id = next(_session_counter)

        try:
            session = await _spawn_session(
                session_id=session_id,
                command=command,
                cwd=cwd,
                tty=input.tty,
            )
        except FileNotFoundError as exc:
            return TextContent(text=f"Error: shell not found: {exc.filename}")
        except Exception as exc:
            return TextContent(text=f"Error: failed to start command: {exc}")

        _exec_sessions[session_id] = session
        yield_seconds = _exec_yield_seconds(input.yield_time_ms)

        try:
            await asyncio.wait_for(
                asyncio.shield(session.process.wait()), timeout=yield_seconds
            )
            return TextContent(
                text=await _finalize_completed_session(
                    session,
                    max_output_tokens=input.max_output_tokens,
                )
            )
        except asyncio.TimeoutError:
            if session.process.returncode is not None:
                return TextContent(
                    text=await _finalize_completed_session(
                        session,
                        max_output_tokens=input.max_output_tokens,
                    )
                )
            # Leave the process and reader tasks alive; callers can inspect the
            # returned session id when a companion write_stdin tool is present.
            return TextContent(
                text=_format_running_response(
                    session,
                    max_output_tokens=input.max_output_tokens,
                )
            )
        except asyncio.CancelledError:
            await _terminate_session(session)
            _exec_sessions.pop(session_id, None)
            raise
        except Exception as exc:
            await _terminate_session(session)
            _exec_sessions.pop(session_id, None)
            return TextContent(text=f"Error: {exc}")


@dataclass
class WriteStdinTool(Tool):
    """Write to or poll a live Codex-style exec_command session."""

    name: str = "write_stdin"
    description: str = (
        "Writes characters to an existing unified exec session and returns recent "
        "output."
    )

    # write_stdin applies Codex-style token truncation itself.
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(enabled=False)

    async def __call__(
        self,
        input: WriteStdinInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        """Write bytes to a running exec_command TTY session or poll it."""
        session = _exec_sessions.get(input.session_id)
        if session is None:
            return TextContent(
                text=f"write_stdin failed: Unknown process id {input.session_id}"
            )

        started_at = time.monotonic()

        if input.chars:
            if session.pty_master_fd is None:
                return TextContent(
                    text=(
                        "write_stdin failed: stdin is closed for this session; "
                        "rerun exec_command with tty=true to keep stdin open"
                    )
                )
            try:
                await _write_pty(session.pty_master_fd, input.chars.encode())
                # Give the process a short chance to react before the poll window.
                await asyncio.sleep(_POST_WRITE_REACT_DELAY)
            except OSError as exc:
                if session.process.returncode is not None:
                    return TextContent(
                        text=await _finalize_completed_session(
                            session,
                            max_output_tokens=input.max_output_tokens,
                            wall_time_seconds=time.monotonic() - started_at,
                        )
                    )
                return TextContent(text=f"write_stdin failed: {exc}")

        yield_seconds = _write_stdin_yield_seconds(input.chars, input.yield_time_ms)
        try:
            await asyncio.wait_for(
                asyncio.shield(session.process.wait()), timeout=yield_seconds
            )
        except asyncio.TimeoutError:
            pass

        wall_time = time.monotonic() - started_at
        if session.process.returncode is not None:
            return TextContent(
                text=await _finalize_completed_session(
                    session,
                    max_output_tokens=input.max_output_tokens,
                    wall_time_seconds=wall_time,
                )
            )

        return TextContent(
            text=_format_running_response(
                session,
                max_output_tokens=input.max_output_tokens,
                wall_time_seconds=wall_time,
            )
        )


def _resolve_workdir(workdir: str) -> Path | None:
    if not workdir:
        return None
    path = Path(workdir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_dir():
        raise ValueError(f"working directory does not exist: {workdir}")
    return path


def _shell_command(cmd: str, shell: str, login: bool) -> list[str]:
    shell_path = shell or os.environ.get("SHELL") or "/bin/bash"
    if os.name == "nt":
        return [shell_path, "/C", cmd]
    return [shell_path, "-lc" if login else "-c", cmd]


async def _spawn_session(
    session_id: int,
    command: list[str],
    cwd: Path | None,
    tty: bool,
) -> _ExecSession:
    chunks: list[bytes] = []

    if tty and os.name != "nt":
        master_fd, slave_fd = pty.openpty()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(cwd) if cwd else None,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)
        pty_reader_tasks = [asyncio.create_task(_read_pty(master_fd, chunks))]
        return _ExecSession(
            session_id=session_id,
            process=process,
            chunks=chunks,
            reader_tasks=pty_reader_tasks,
            started_at=time.monotonic(),
            pty_master_fd=master_fd,
        )

    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        start_new_session=(os.name != "nt"),
    )
    reader_tasks: list[asyncio.Task[None]] = []
    if process.stdout is not None:
        reader_tasks.append(asyncio.create_task(_read_stream(process.stdout, chunks)))
    if process.stderr is not None:
        reader_tasks.append(asyncio.create_task(_read_stream(process.stderr, chunks)))
    return _ExecSession(
        session_id=session_id,
        process=process,
        chunks=chunks,
        reader_tasks=reader_tasks,
        started_at=time.monotonic(),
    )


async def _read_stream(
    stream: asyncio.StreamReader,
    chunks: list[bytes],
) -> None:
    while True:
        chunk = await stream.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)


async def _read_pty(fd: int, chunks: list[bytes]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            chunk = await loop.run_in_executor(None, os.read, fd, _READ_CHUNK_SIZE)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)


async def _write_pty(fd: int, data: bytes) -> None:
    loop = asyncio.get_running_loop()
    offset = 0
    while offset < len(data):
        written = await loop.run_in_executor(None, os.write, fd, data[offset:])
        if written == 0:
            raise OSError("pty write returned 0 bytes")
        offset += written


async def _finish_session(session: _ExecSession) -> None:
    if session.reader_tasks:
        done, pending = await asyncio.wait(session.reader_tasks, timeout=1.0)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*done, return_exceptions=True)
    _close_pty(session)


async def _terminate_session(session: _ExecSession) -> None:
    await terminate_process(session.process, kill_group=True)
    await _finish_session(session)


def _close_pty(session: _ExecSession) -> None:
    if session.pty_master_fd is not None:
        try:
            os.close(session.pty_master_fd)
        except OSError:
            pass
        session.pty_master_fd = None


def _effective_max_output_tokens(value: int) -> int:
    return value if value > 0 else _DEFAULT_MAX_OUTPUT_TOKENS


def _exec_yield_seconds(yield_time_ms: int) -> float:
    return min(max(yield_time_ms, _MIN_YIELD_TIME_MS), _MAX_YIELD_TIME_MS) / 1000


def _write_stdin_yield_seconds(chars: str, yield_time_ms: int) -> float:
    bumped = max(yield_time_ms, _MIN_YIELD_TIME_MS)
    if not chars:
        capped = max(
            _MIN_EMPTY_WRITE_STDIN_YIELD_TIME_MS,
            min(bumped, _MAX_BACKGROUND_TERMINAL_TIMEOUT_MS),
        )
    else:
        capped = min(bumped, _MAX_YIELD_TIME_MS)
    return capped / 1000


def _truncate_to_tokens(data: bytes, max_tokens: int) -> str:
    """Middle-truncate ``data`` to fit within ``max_tokens`` approximate tokens.

    Mirrors codex-rs ``formatted_truncate_text`` with ``TruncationPolicy::Tokens``:
    if the bytes fit in ``max_tokens * 4`` they decode and return unchanged.
    Otherwise prefix and suffix are preserved on UTF-8 boundaries, the middle
    is replaced with ``…N tokens truncated…``, and a ``Total output lines: M``
    header is prepended.

    Operating on bytes lets the caller skip a redundant encode/decode round
    trip when the source is already raw subprocess output.
    """
    if not data:
        return ""
    byte_budget = max_tokens * APPROX_BYTES_PER_TOKEN
    if max_tokens > 0 and len(data) <= byte_budget:
        return data.decode("utf-8", errors="replace")

    total_lines = len(data.splitlines())
    removed_bytes = len(data) - byte_budget
    removed_tokens = (
        removed_bytes + APPROX_BYTES_PER_TOKEN - 1
    ) // APPROX_BYTES_PER_TOKEN
    marker = f"…{removed_tokens} tokens truncated…"

    if byte_budget <= 0:
        return f"Total output lines: {total_lines}\n\n{marker}"

    left_budget = byte_budget // 2
    right_budget = byte_budget - left_budget
    prefix = _take_utf8_prefix(data, left_budget)
    suffix = _take_utf8_suffix(data, right_budget)
    return f"Total output lines: {total_lines}\n\n{prefix}{marker}{suffix}"


def _take_utf8_prefix(data: bytes, budget: int) -> str:
    if budget <= 0:
        return ""
    end = min(budget, len(data))
    while 0 < end < len(data) and (data[end] & 0xC0) == 0x80:
        end -= 1
    return data[:end].decode("utf-8", errors="replace")


def _take_utf8_suffix(data: bytes, budget: int) -> str:
    if budget <= 0:
        return ""
    start = max(0, len(data) - budget)
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    return data[start:].decode("utf-8", errors="replace")


def _generate_chunk_id() -> str:
    return "".join(f"{random.randrange(16):x}" for _ in range(6))


def _format_response(
    session: _ExecSession,
    max_output_tokens: int,
    exit_code: int | None,
    running_session_id: int | None,
    chunk_id: str,
    wall_time_seconds: float | None = None,
) -> str:
    output_bytes = session.consume_pending()
    token_count = (
        len(output_bytes) + APPROX_BYTES_PER_TOKEN - 1
    ) // APPROX_BYTES_PER_TOKEN
    max_tokens = _effective_max_output_tokens(max_output_tokens)
    truncated = _truncate_to_tokens(output_bytes, max_tokens)
    wall_time = (
        time.monotonic() - session.started_at
        if wall_time_seconds is None
        else wall_time_seconds
    )

    sections: list[str] = [
        f"Chunk ID: {chunk_id}",
        f"Wall time: {wall_time:.4f} seconds",
    ]
    if exit_code is not None:
        sections.append(f"Process exited with code {exit_code}")
    if running_session_id is not None:
        sections.append(f"Process running with session ID {running_session_id}")
    sections.append(f"Original token count: {token_count}")
    sections.append("Output:")
    sections.append(truncated)
    return "\n".join(sections)


async def _finalize_completed_session(
    session: _ExecSession,
    *,
    max_output_tokens: int,
    wall_time_seconds: float | None = None,
) -> str:
    """Drain readers, drop the registry entry, format the final response."""
    await _finish_session(session)
    _exec_sessions.pop(session.session_id, None)
    return _format_response(
        session=session,
        max_output_tokens=max_output_tokens,
        exit_code=session.process.returncode,
        running_session_id=None,
        wall_time_seconds=wall_time_seconds,
        chunk_id=_generate_chunk_id(),
    )


def _format_running_response(
    session: _ExecSession,
    *,
    max_output_tokens: int,
    wall_time_seconds: float | None = None,
) -> str:
    """Format a response for a still-running session (no cleanup)."""
    return _format_response(
        session=session,
        max_output_tokens=max_output_tokens,
        exit_code=None,
        running_session_id=session.session_id,
        wall_time_seconds=wall_time_seconds,
        chunk_id=_generate_chunk_id(),
    )


def _reap_finished_sessions() -> None:
    """Drop registry entries for sessions whose process has already exited.

    Sessions only stay in ``_exec_sessions`` while still running — once
    finalized they're popped. But a process can exit between polls without
    anyone calling write_stdin, leaving a zombie entry. Reap them so the cap
    counts only sessions the model could plausibly still interact with.
    """
    for sid in list(_exec_sessions.keys()):
        session = _exec_sessions[sid]
        if session.process.returncode is not None:
            _close_pty(session)
            _exec_sessions.pop(sid, None)


async def cleanup_exec_command_sessions() -> int:
    """Terminate all live exec_command sessions. Primarily useful for tests."""
    sessions = list(_exec_sessions.values())
    for session in sessions:
        await _terminate_session(session)
        _exec_sessions.pop(session.session_id, None)
    return len(sessions)
