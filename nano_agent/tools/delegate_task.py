"""DelegateTask tool for delegating work to a Codex CLI sub-agent."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from typing import Annotated, ClassVar

from ..data_structures import TextContent
from ..execution_context import ExecutionContext
from .base import Desc, Tool


@dataclass
class DelegateTaskInput:
    """Input for DelegateTaskTool."""

    prompt: Annotated[
        str,
        Desc(
            "Full task description for the sub-agent. The sub-agent has NO "
            "context from this conversation, so include: the goal, relevant "
            "files (paths), facts already established, expected output "
            "format, and any constraints. Be explicit and complete."
        ),
    ]
    model: Annotated[
        str,
        Desc("Codex model to use (e.g. 'gpt-5', 'gpt-5.5'). Defaults to 'gpt-5'."),
    ] = "gpt-5"
    sandbox: Annotated[
        str,
        Desc(
            "Sandbox mode for the sub-agent: 'read-only' (default, sub-agent "
            "can only read files), 'workspace-write' (can write inside the "
            "working dir), or 'danger-full-access' (no sandbox)."
        ),
    ] = "read-only"
    cwd: Annotated[
        str,
        Desc(
            "Working directory for the sub-agent. Empty string uses the "
            "current process directory."
        ),
    ] = ""
    timeout: Annotated[
        int,
        Desc("Timeout in seconds (max 1800 / 30 minutes). Defaults to 600."),
    ] = 600


@dataclass
class DelegateTaskTool(Tool):
    """Delegate a task to a Codex CLI sub-agent."""

    name: str = "DelegateTask"
    description: str = """Delegate a task to a sub-agent running on the Codex CLI.

Use this when you want to offload a research, analysis, or implementation
task to a fresh sub-agent with its own context window. The sub-agent runs in
a separate process via `codex exec` and returns its full output.

The sub-agent has NO awareness of this conversation. Provide ALL relevant
context in the prompt:
- The task / question / goal
- Relevant files (absolute paths it should read)
- Facts already established that it shouldn't re-derive
- Expected output format and where to save it (if applicable)
- Constraints (deliverable shape, depth of investigation, etc.)

Best for:
- Research tasks where you don't want to fill your own context
- Independent sub-tasks that benefit from a different model
- Long-running analysis whose output you'll then write to a file yourself

Defaults to a read-only sandbox so the sub-agent cannot modify files. If the
sub-agent needs to write its output to disk directly, set
`sandbox="workspace-write"` and tell it the target path in the prompt.

Returns: the sub-agent's stdout (its final answer plus any progress logs)."""

    _MAX_TIMEOUT: ClassVar[int] = 1800
    _ALLOWED_SANDBOXES: ClassVar[frozenset[str]] = frozenset(
        {"read-only", "workspace-write", "danger-full-access"}
    )

    async def __call__(
        self,
        input: DelegateTaskInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        if not input.prompt.strip():
            return TextContent(text="Error: prompt is required")
        if input.sandbox not in self._ALLOWED_SANDBOXES:
            return TextContent(
                text=(
                    f"Error: invalid sandbox '{input.sandbox}'. "
                    f"Allowed: {sorted(self._ALLOWED_SANDBOXES)}"
                )
            )
        timeout = min(input.timeout, self._MAX_TIMEOUT)

        codex_path = shutil.which("codex")
        if codex_path is None:
            return TextContent(
                text=(
                    "Error: codex CLI not found on PATH. "
                    "Install it from https://github.com/openai/codex."
                )
            )

        # --skip-git-repo-check: codex exec refuses non-git cwds by default;
        # delegated tasks should run wherever the caller points us, so suppress.
        cmd = [
            codex_path,
            "exec",
            "--model",
            input.model,
            "--sandbox",
            input.sandbox,
            "--skip-git-repo-check",
        ]
        if input.cwd:
            cmd.extend(["--cd", input.cwd])
        cmd.append(input.prompt)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.CancelledError:
            await self._terminate(process)
            raise
        except asyncio.TimeoutError:
            await self._terminate(process)
            return TextContent(
                text=f"Error: codex exec timed out after {timeout}s"
            )

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")

        if process.returncode != 0:
            parts = [
                f"Error: codex exec exited with code {process.returncode}",
            ]
            if stderr.strip():
                parts.append(f"stderr:\n{stderr.strip()}")
            if stdout.strip():
                parts.append(f"stdout:\n{stdout.strip()}")
            return TextContent(text="\n\n".join(parts))

        output = stdout.strip() or stderr.strip() or "(no output)"
        return TextContent(text=output)

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
