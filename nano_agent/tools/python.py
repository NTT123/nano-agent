"""Python tool for creating, editing, and running Python scripts."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar

from ..data_structures import TextContent
from .base import Desc, Tool, TruncationConfig

# =============================================================================
# Module-level state for Python tool
# =============================================================================


@dataclass
class PythonScript:
    """Tracks a Python script file for the Python tool."""

    file_id: str
    file_path: str
    content: str
    created_at: float
    last_run_at: float | None = None
    run_count: int = 0


# Global dict to store Python scripts by file_id
_python_scripts: dict[str, PythonScript] = {}

# Python tool configuration
_PYTHON_SCRIPT_PREFIX = "nano_python_"
_PYTHON_SCRIPT_EXPIRY_SECONDS = 1800  # 30 minutes


def _cleanup_expired_python_scripts() -> None:
    """Remove expired Python script files from registry and disk."""
    current_time = time.time()
    expired = [
        file_id
        for file_id, script in _python_scripts.items()
        if current_time - script.created_at > _PYTHON_SCRIPT_EXPIRY_SECONDS
    ]
    for file_id in expired:
        script = _python_scripts[file_id]
        # Try to delete the file from disk
        try:
            Path(script.file_path).unlink(missing_ok=True)
        except Exception:
            pass
        del _python_scripts[file_id]


def _get_python_script_dir() -> Path:
    """Get or create the Python scripts directory in temp."""
    script_dir = Path(tempfile.gettempdir()) / _PYTHON_SCRIPT_PREFIX
    script_dir.mkdir(exist_ok=True)
    return script_dir


def _get_uv_cache_dir() -> Path:
    """Get or create a writable uv cache directory in temp."""
    cache_dir = Path(tempfile.gettempdir()) / "nano_uv_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def list_python_scripts() -> list[PythonScript]:
    """List all active Python script files (utility function)."""
    _cleanup_expired_python_scripts()
    return list(_python_scripts.values())


def clear_python_scripts() -> int:
    """Clear all Python script files from registry and disk. Returns count of files cleared."""
    count = len(_python_scripts)
    for file_id, script in list(_python_scripts.items()):
        try:
            Path(script.file_path).unlink(missing_ok=True)
        except Exception:
            pass
    _python_scripts.clear()
    return count


# =============================================================================
# Input Dataclass
# =============================================================================


@dataclass
class PythonInput:
    """Input for PythonTool."""

    operation: Annotated[
        str, Desc("The operation to perform: 'create', 'edit', or 'run'")
    ]
    code: Annotated[str, Desc("Python code for create/edit operations")] = ""
    file_id: Annotated[str, Desc("Script file identifier for edit/run operations")] = ""
    dependencies: Annotated[
        list[str], Desc("Pip packages for uv run --with (e.g., ['numpy', 'pandas'])")
    ] = field(default_factory=list)
    timeout: Annotated[
        int, Desc("Execution timeout in milliseconds (default 30000, max 300000)")
    ] = 30000
    output_limit: Annotated[
        int, Desc("Maximum output characters (default 50000, max 100000)")
    ] = 50000
    filename: Annotated[
        str, Desc("Optional custom filename (without .py extension)")
    ] = ""


# =============================================================================
# Tool Class
# =============================================================================


@dataclass
class PythonTool(Tool):
    """Create, edit, and run Python scripts with automatic dependency management.

    A lightweight tool for executing Python code with on-the-fly dependency
    installation via uv.

    Operations:
    - create: Create a new Python script file
    - edit: Modify an existing script
    - run: Execute a script with optional dependencies

    Example usage:
        tool = PythonTool()

        # Create a script
        result = await tool.execute({
            "operation": "create",
            "code": "import numpy as np\\nprint(np.array([1,2,3]))"
        })

        # Run with dependencies
        result = await tool.execute({
            "operation": "run",
            "file_id": "py_abc123",
            "dependencies": ["numpy"]
        })
    """

    name: str = "Python"
    description: str = """Create, edit, run, and delete Python scripts with automatic dependency management.

Use this tool when you need to:
- Perform calculations or data processing that's easier in Python than bash
- Test code snippets or algorithms quickly
- Process JSON/CSV data with pandas or other libraries
- Make HTTP requests or interact with APIs
- Generate files, reports, or visualizations
- Run any Python code that needs external packages

Operations:
- create: Create a new Python script
  Required: code
  Optional: filename (custom name without .py)
  Returns: file_id for later operations

- edit: Modify an existing script
  Required: file_id, code
  Returns: confirmation with updated line count

- run: Execute a script with optional dependencies
  Required: file_id
  Optional: dependencies (list of pip packages), timeout, output_limit
  Returns: stdout/stderr, exit code, elapsed time

- delete: Remove a script when no longer needed
  Required: file_id
  Returns: confirmation of deletion

Dependencies are installed on-the-fly using 'uv run --with'.
Script files auto-expire after 30 minutes of inactivity, but use 'delete' to clean up immediately.

Examples:

1. Quick calculation:
   create: {"operation": "create", "code": "import math\\nprint(f'sqrt(2) = {math.sqrt(2):.6f}')"}
   run: {"operation": "run", "file_id": "py_xxx"}

2. Data processing with pandas:
   create: {"operation": "create", "code": "import pandas as pd\\ndf = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})\\nprint(df.describe())"}
   run: {"operation": "run", "file_id": "py_xxx", "dependencies": ["pandas"]}

3. HTTP request with requests:
   create: {"operation": "create", "code": "import requests\\nr = requests.get('https://api.github.com')\\nprint(r.status_code, r.headers['content-type'])"}
   run: {"operation": "run", "file_id": "py_xxx", "dependencies": ["requests"]}

4. JSON processing:
   create: {"operation": "create", "code": "import json\\ndata = {'name': 'test', 'values': [1,2,3]}\\nprint(json.dumps(data, indent=2))"}
   run: {"operation": "run", "file_id": "py_xxx"}

5. File generation:
   create: {"operation": "create", "code": "with open('output.txt', 'w') as f:\\n    f.write('Generated content')\\nprint('File created')"}
   run: {"operation": "run", "file_id": "py_xxx"}

6. Iterative development (edit and re-run):
   edit: {"operation": "edit", "file_id": "py_xxx", "code": "# Updated code\\nprint('v2')"}
   run: {"operation": "run", "file_id": "py_xxx"}

7. Clean up when done:
   delete: {"operation": "delete", "file_id": "py_xxx"}

Note: Requires 'uv' to be installed (pip install uv or brew install uv)."""

    # Class constants
    MAX_TIMEOUT_MS: ClassVar[int] = 300000  # 5 minutes max
    MAX_OUTPUT_CHARS: ClassVar[int] = 100000

    # Disable truncation - PythonTool already has output_limit parameter
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(enabled=False)
    _required_commands: ClassVar[dict[str, str]] = {
        "uv": (
            "Install uv: pip install uv, brew install uv (macOS), "
            "or curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    }

    async def _create(self, input: PythonInput) -> TextContent:
        """Create a new Python script file."""
        if not input.code.strip():
            return TextContent(
                text="Error: 'code' parameter is required for create operation.\n\n"
                "Provide the Python code you want to run."
            )

        # Generate file_id
        file_id = f"py_{uuid.uuid4().hex[:8]}"

        # Determine filename
        if input.filename:
            # Sanitize filename
            safe_name = "".join(
                c for c in input.filename if c.isalnum() or c in "_-"
            ).strip()
            if not safe_name:
                safe_name = file_id
            filename = f"{safe_name}.py"
        else:
            filename = f"{file_id}.py"

        # Create file
        script_dir = _get_python_script_dir()
        file_path = script_dir / filename

        try:
            file_path.write_text(input.code)
        except Exception as e:
            return TextContent(text=f"Error creating script file: {e}")

        # Register script
        script = PythonScript(
            file_id=file_id,
            file_path=str(file_path),
            content=input.code,
            created_at=time.time(),
        )
        _python_scripts[file_id] = script

        # Build response
        line_count = input.code.count("\n") + (
            1 if input.code and not input.code.endswith("\n") else 0
        )
        return TextContent(
            text=f"✓ Script created successfully\n"
            f"  file_id: {file_id}\n"
            f"  path: {file_path}\n"
            f"  lines: {line_count}\n\n"
            f"Use this file_id for 'edit' or 'run' operations."
        )

    async def _edit(self, input: PythonInput) -> TextContent:
        """Edit an existing Python script file."""
        if not input.file_id:
            return TextContent(
                text="Error: 'file_id' parameter is required for edit operation.\n\n"
                "Provide the file_id from a previous 'create' operation."
            )

        if not input.code.strip():
            return TextContent(
                text="Error: 'code' parameter is required for edit operation.\n\n"
                "Provide the new Python code content."
            )

        # Clean up expired scripts first
        _cleanup_expired_python_scripts()

        # Look up script
        if input.file_id not in _python_scripts:
            return TextContent(
                text=f"Error: Script '{input.file_id}' not found or expired.\n\n"
                "Script files expire after 30 minutes. "
                "Use 'create' operation to create a new script."
            )

        script = _python_scripts[input.file_id]

        # Verify file still exists
        file_path = Path(script.file_path)
        if not file_path.exists():
            del _python_scripts[input.file_id]
            return TextContent(
                text="Error: Script file was deleted from disk.\n\n"
                "Use 'create' operation to create a new script."
            )

        # Write new content
        try:
            file_path.write_text(input.code)
        except Exception as e:
            return TextContent(text=f"Error writing to script file: {e}")

        # Update registry
        script.content = input.code
        script.created_at = time.time()  # Reset expiry

        # Build response
        line_count = input.code.count("\n") + (
            1 if input.code and not input.code.endswith("\n") else 0
        )
        return TextContent(
            text=f"✓ Script updated successfully\n"
            f"  file_id: {input.file_id}\n"
            f"  lines: {line_count}\n"
            f"  path: {script.file_path}"
        )

    async def _run(self, input: PythonInput) -> TextContent:
        """Run an existing Python script file."""
        if not input.file_id:
            return TextContent(
                text="Error: 'file_id' parameter is required for run operation.\n\n"
                "Provide the file_id from a previous 'create' operation."
            )

        # Clean up expired scripts first
        _cleanup_expired_python_scripts()

        # Look up script
        if input.file_id not in _python_scripts:
            return TextContent(
                text=f"Error: Script '{input.file_id}' not found or expired.\n\n"
                "Script files expire after 30 minutes. "
                "Use 'create' operation to create a new script."
            )

        script = _python_scripts[input.file_id]
        file_path = Path(script.file_path)

        # Verify file still exists
        if not file_path.exists():
            del _python_scripts[input.file_id]
            return TextContent(
                text="Error: Script file was deleted from disk.\n\n"
                "Use 'create' operation to create a new script."
            )

        # Build command
        cmd = ["uv", "run"]

        # Add dependencies
        for dep in input.dependencies:
            cmd.extend(["--with", dep])

        cmd.append(str(file_path))

        # Apply limits
        timeout_ms = min(input.timeout, self.MAX_TIMEOUT_MS)
        timeout_s = timeout_ms / 1000
        output_limit = min(input.output_limit, self.MAX_OUTPUT_CHARS)

        # Execute
        start_time = time.time()
        try:
            env = os.environ.copy()
            cache_dir = _get_uv_cache_dir()
            env.setdefault("UV_CACHE_DIR", str(cache_dir))
            env.setdefault("XDG_CACHE_HOME", str(cache_dir))
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s
            )
            elapsed = time.time() - start_time
            exit_code = process.returncode

            # Update script stats
            script.last_run_at = time.time()
            script.run_count += 1
            script.created_at = time.time()  # Reset expiry on use

            # Decode output
            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")

            # Truncate if needed
            total_output = stdout_text + stderr_text
            truncated = False
            if len(total_output) > output_limit:
                truncated = True
                # Truncate proportionally
                if stdout_text and stderr_text:
                    stdout_text = stdout_text[: output_limit // 2]
                    stderr_text = stderr_text[: output_limit // 2]
                elif stdout_text:
                    stdout_text = stdout_text[:output_limit]
                else:
                    stderr_text = stderr_text[:output_limit]

            # Build output
            parts = [f"─── Run: {input.file_id} ───"]
            if input.dependencies:
                parts.append(f"Dependencies: {', '.join(input.dependencies)}")
            parts.append(f"Exit code: {exit_code}")
            parts.append(f"Elapsed: {elapsed:.2f}s")
            parts.append(f"Run count: {script.run_count}")
            parts.append("")

            if stdout_text.strip():
                parts.append("─── stdout ───")
                parts.append(stdout_text.rstrip())
                parts.append("")

            if stderr_text.strip():
                parts.append("─── stderr ───")
                parts.append(stderr_text.rstrip())
                parts.append("")

            if truncated:
                parts.append(f"\n⚠️ Output truncated at {output_limit} characters.")

            if not stdout_text.strip() and not stderr_text.strip():
                parts.append("(no output)")

            return TextContent(text="\n".join(parts))

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            return TextContent(
                text=f"Error: Execution timed out after {timeout_s:.1f}s\n\n"
                f"The script was killed. Consider:\n"
                f"1. Optimizing your code\n"
                f"2. Increasing timeout (max {self.MAX_TIMEOUT_MS}ms)"
            )
        except Exception as e:
            return TextContent(text=f"Error running script: {e}")

    async def _delete(self, input: PythonInput) -> TextContent:
        """Delete a Python script file."""
        if not input.file_id:
            return TextContent(
                text="Error: 'file_id' parameter is required for delete operation.\n\n"
                "Provide the file_id of the script to delete."
            )

        # Clean up expired scripts first
        _cleanup_expired_python_scripts()

        # Look up script
        if input.file_id not in _python_scripts:
            return TextContent(
                text=f"Error: Script '{input.file_id}' not found or already deleted.\n\n"
                "The script may have expired or been deleted previously."
            )

        script = _python_scripts[input.file_id]
        file_path = Path(script.file_path)

        # Delete from disk
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            return TextContent(text=f"Error deleting script file: {e}")

        # Remove from registry
        del _python_scripts[input.file_id]

        return TextContent(
            text=f"✓ Script deleted successfully\n"
            f"  file_id: {input.file_id}\n"
            f"  path: {script.file_path}"
        )

    async def __call__(self, input: PythonInput) -> TextContent:
        """Execute the requested Python operation."""
        operation = input.operation.lower().strip()

        if operation == "create":
            return await self._create(input)
        elif operation == "edit":
            return await self._edit(input)
        elif operation == "run":
            return await self._run(input)
        elif operation == "delete":
            return await self._delete(input)
        else:
            return TextContent(
                text=f"Error: Invalid operation '{input.operation}'\n\n"
                "Valid operations are:\n"
                "  - create: Create a new Python script\n"
                "  - edit: Modify an existing script\n"
                "  - run: Execute a script with optional dependencies\n"
                "  - delete: Remove a script when no longer needed"
            )
