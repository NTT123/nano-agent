"""Built-in tool definitions for Claude API."""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import tempfile
import time
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .data_structures import TextContent

# =============================================================================
# Module-level state for Edit tool confirmation workflow
# =============================================================================


@dataclass
class PendingEdit:
    """Stores a pending edit awaiting confirmation."""

    file_path: str
    old_string: str
    new_string: str
    match_line: int
    context_before: list[str]  # Lines before match
    context_after: list[str]  # Lines after match
    created_at: float
    replace_all: bool = False
    match_count: int = 1


# Global dict to store pending edits by edit_id
_pending_edits: dict[str, PendingEdit] = {}

# Expiry time for pending edits (5 minutes)
_EDIT_EXPIRY_SECONDS = 300


# =============================================================================
# Truncation Configuration and State
# =============================================================================


@dataclass
class TruncationConfig:
    """Configuration for tool output truncation.

    Attributes:
        max_chars: Maximum characters before truncation (default 30000)
        max_lines: Maximum lines before truncation (default 1000)
        enabled: Whether truncation is enabled for this tool (default True)
    """

    max_chars: int = 30000
    max_lines: int = 1000
    enabled: bool = True


@dataclass
class TruncatedOutput:
    """Metadata about a truncated tool output.

    Stored in _truncated_outputs for cleanup and retrieval.
    """

    tool_name: str
    temp_file_path: str
    original_chars: int
    original_lines: int
    created_at: float


# Global registry of truncated outputs for cleanup
_truncated_outputs: dict[str, TruncatedOutput] = {}

# Default truncation configuration
_DEFAULT_TRUNCATION_CONFIG = TruncationConfig()


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

# =============================================================================
# Type Extraction and Conversion Utilities
# =============================================================================


def get_call_input_type(cls: type) -> type | None:
    """Extract the input type from a Tool's __call__ method's 'input' parameter.

    Supports:
    - Direct dataclass types: `input: MyInput`
    - Optional types: `input: MyInput | None` or `input: Optional[MyInput]`
    - None for no-input tools: `input: None` or no input parameter at all

    Returns:
        The dataclass type for input, or None for no-input tools.

    Raises:
        TypeError: If annotation is invalid (not a dataclass).
    """
    # Get the __call__ method
    call_method = getattr(cls, "__call__", None)
    if call_method is None:
        raise TypeError(f"{cls.__name__} does not have a __call__ method")

    # Get type hints for the method
    try:
        hints = get_type_hints(call_method, include_extras=True)
    except Exception:
        raise TypeError(f"{cls.__name__}.__call__ has no valid type annotations")

    # No input parameter means no-input tool
    if "input" not in hints:
        return None

    input_type: Any = hints["input"]

    # Handle None type (for no-input tools)
    if input_type is type(None):
        return None  # type: ignore[no-any-return]

    # Handle Optional[T] / T | None (Union types)
    origin = get_origin(input_type)
    if origin is Union or origin is types.UnionType:
        args = get_args(input_type)
        # Filter out NoneType
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            input_type = non_none_args[0]
        elif len(non_none_args) == 0:
            return None
        else:
            raise TypeError(
                f"{cls.__name__}.__call__ input type must be a single dataclass "
                f"or Optional[dataclass], got Union of multiple types"
            )

    # Validate it's a dataclass
    if not dataclasses.is_dataclass(input_type):
        raise TypeError(
            f"{cls.__name__}.__call__ input type must be a dataclass, "
            f"got {input_type}"
        )

    # Cast to type for return
    return input_type  # type: ignore[return-value]


def _convert_value(value: Any, target_type: Any) -> Any:
    """Recursively convert a value to its target type.

    Handles:
    - Dataclasses: dict -> dataclass instance
    - Lists: list of dicts -> list of dataclass instances (if items are dataclasses)
    - Other types: passed through as-is
    """
    # Handle None
    if value is None:
        return None

    # Handle list types
    origin = get_origin(target_type)
    if origin is list:
        args = get_args(target_type)
        if args and isinstance(value, list):
            item_type = args[0]
            # If items are dataclasses, convert each one
            if dataclasses.is_dataclass(item_type) and isinstance(item_type, type):
                return [_convert_value(item, item_type) for item in value]
        return value

    # Handle dataclass types
    if dataclasses.is_dataclass(target_type) and isinstance(target_type, type):
        if isinstance(value, dict):
            # Get type hints for the dataclass fields
            hints = get_type_hints(target_type, include_extras=True)
            converted_kwargs = {}
            for field_name, field_value in value.items():
                if field_name in hints:
                    # Unwrap Annotated if present
                    field_type = hints[field_name]
                    if get_origin(field_type) is Annotated:
                        field_type = get_args(field_type)[0]
                    converted_kwargs[field_name] = _convert_value(
                        field_value, field_type
                    )
                else:
                    converted_kwargs[field_name] = field_value
            return target_type(**converted_kwargs)
        return value

    # Pass through other types
    return value


def convert_input(input_dict: dict[str, Any] | None, input_type: type | None) -> Any:
    """Convert a raw dict from API to a typed dataclass instance.

    Args:
        input_dict: The raw input dict from the API (may be None)
        input_type: The target dataclass type (or None for no-input tools)

    Returns:
        A dataclass instance, or None if input_type is None
    """
    if input_type is None:
        return None

    if input_dict is None:
        input_dict = {}

    return _convert_value(input_dict, input_type)


# =============================================================================
# Todo Data Classes (merged from tool_handlers.py)
# =============================================================================


class TodoStatus(str, Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Todo:
    """A single todo item with content, status, and active form."""

    content: str
    status: TodoStatus
    active_form: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Todo":
        """Create a Todo from a dictionary (e.g., from API input)."""
        return cls(
            content=data.get("content", ""),
            status=TodoStatus(data.get("status", "pending")),
            active_form=data.get("activeForm", ""),
        )

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "status": self.status.value,
            "activeForm": self.active_form,
        }


InputSchemaDict = dict[str, object]


class Desc:
    """Field description for JSON schema generation.

    Works in two ways:

    1. Modern (recommended): Inside Annotated[]
        @dataclass
        class Input:
            city: Annotated[str, Desc("City name like Tokyo")]
            year: Annotated[int, Desc("Birth year")] = 2000

    2. Legacy: As field default (creates dataclass field with metadata)
        @dataclass
        class Input:
            city: str = Desc("City name like Tokyo").field()
            year: int = Desc("Birth year").field(default=2000)
    """

    def __init__(self, description: str):
        self.description = description

    def field(self, **kwargs: Any) -> Any:
        """Create a dataclass field with this description in metadata."""
        metadata = {"description": self.description}
        return field(metadata=metadata, **kwargs)


# Alias for Pydantic-style naming
Field = Desc


# =============================================================================
# Tool Input Data Classes
# =============================================================================


@dataclass
class BashInput:
    """Input for BashTool."""

    command: Annotated[str, Desc("The command to execute")]
    timeout: Annotated[int, Desc("Optional timeout in seconds (max 600)")] = 120
    description: Annotated[
        str, Desc("Clear, concise description of what this command does in 5-10 words")
    ] = ""
    run_in_background: Annotated[
        bool, Desc("Set to true to run this command in the background.")
    ] = False


@dataclass
class TodoItemInput:
    """A single todo item for API input."""

    content: Annotated[str, Desc("Task description")]
    status: Annotated[str, Desc("pending, in_progress, or completed")]
    activeForm: Annotated[str, Desc("Present continuous form shown in spinner")]


@dataclass
class TodoWriteInput:
    """Input for TodoWriteTool."""

    todos: Annotated[list[TodoItemInput], Desc("The updated todo list")]


# =============================================================================
# Input Dataclasses for All Tools
# =============================================================================


@dataclass
class GlobInput:
    """Input for GlobTool."""

    pattern: Annotated[str, Desc("The glob pattern to match files against")]
    path: Annotated[
        str, Desc("The directory to search in. Defaults to current working directory.")
    ] = ""


@dataclass
class ReadInput:
    """Input for ReadTool."""

    file_path: Annotated[str, Desc("The absolute path to the file to read")]
    offset: Annotated[int, Desc("The line number to start reading from")] = 0
    limit: Annotated[int, Desc("The number of lines to read")] = 0


@dataclass
class WriteInput:
    """Input for WriteTool."""

    file_path: Annotated[
        str,
        Desc("The absolute path to the file to write (must be absolute, not relative)"),
    ]
    content: Annotated[str, Desc("The content to write to the file")]


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


@dataclass
class WebFetchInput:
    """Input for WebFetchTool."""

    url: Annotated[str, Desc("The URL to fetch content from")]
    prompt: Annotated[str, Desc("The prompt to run on the fetched content")]


@dataclass
class EditConfirmInput:
    """Input for EditConfirmTool."""

    edit_id: Annotated[str, Desc("The edit ID from preview to confirm or reject")]


@dataclass
class StatInput:
    """Input for StatTool."""

    file_path: Annotated[str, Desc("Absolute path to file or directory")]


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
class QuestionOption:
    """A single option for a question in AskUserQuestionTool."""

    label: Annotated[str, Desc("The display text for this option")]
    description: Annotated[str, Desc("Explanation of what this option means")]


@dataclass
class Question:
    """A single question for AskUserQuestionTool."""

    question: Annotated[str, Desc("The complete question to ask the user")]
    header: Annotated[
        str, Desc("Very short label displayed as a chip/tag (max 12 chars)")
    ]
    options: Annotated[
        list[QuestionOption], Desc("The available choices for this question")
    ]
    multiSelect: Annotated[bool, Desc("Set to true to allow multiple selections")]


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


def _get_json_type_for_python_type(
    py_type: Any, type_map: dict[type, str]
) -> dict[str, Any]:
    """Convert a Python type to JSON schema type definition.

    Handles:
    - Basic types (str, int, float, bool)
    - list[T] -> {"type": "array", "items": {...}}
    - dict[K, V] -> {"type": "object", "additionalProperties": {...}}
    - Nested dataclasses -> recursive schema
    """
    # Check for list/array types
    origin = get_origin(py_type)
    if origin is list:
        args = get_args(py_type)
        if args:
            item_type = args[0]
            # Check if item is a dataclass
            if dataclasses.is_dataclass(item_type) and isinstance(item_type, type):
                return {
                    "type": "array",
                    "items": schema_from_dataclass(item_type),
                }
            else:
                item_json_type = type_map.get(item_type, "string")
                return {"type": "array", "items": {"type": item_json_type}}
        else:
            return {"type": "array"}

    # Check for dict types
    if origin is dict:
        args = get_args(py_type)
        if args and len(args) >= 2:
            value_type = args[1]
            # Check if value is a dataclass
            if dataclasses.is_dataclass(value_type) and isinstance(value_type, type):
                return {
                    "type": "object",
                    "additionalProperties": schema_from_dataclass(value_type),
                }
            else:
                value_json_type = type_map.get(value_type, "string")
                return {
                    "type": "object",
                    "additionalProperties": {"type": value_json_type},
                }
        else:
            return {"type": "object"}

    # Check for nested dataclass
    if dataclasses.is_dataclass(py_type) and isinstance(py_type, type):
        return schema_from_dataclass(py_type)

    # Basic type
    json_type = type_map.get(py_type, "string")
    return {"type": json_type}


def schema_from_dataclass(cls: type) -> InputSchemaDict:
    """Generate JSON schema from a dataclass.

    Use Annotated with Desc() (or Field(), which is an alias) for descriptions:

        @dataclass
        class Input:
            city: Annotated[str, Desc("City name like Tokyo")]

    Example:
        @dataclass
        class HoroscopeInput:
            sign: Annotated[str, Desc("An astrological sign like Taurus")]

        schema = schema_from_dataclass(HoroscopeInput)
        # {
        #   "type": "object",
        #   "properties": {
        #     "sign": {"type": "string", "description": "An astrological sign..."}
        #   },
        #   "required": ["sign"]
        # }
    """
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    # Use include_extras=True to preserve Annotated metadata
    hints = get_type_hints(cls, include_extras=True)
    fields = dataclasses.fields(cls)

    properties = {}
    required = []

    for f in fields:
        hint = hints.get(f.name, str)
        description: str | None = None

        # Check if it's Annotated[type, Desc("...")]
        if get_origin(hint) is Annotated:
            base_type, *metadata = get_args(hint)
            py_type = base_type
            # Look for Desc metadata
            for m in metadata:
                if isinstance(m, Desc):
                    description = m.description
                    break
        else:
            py_type = hint

        # Build property dict using helper (handles list, nested dataclass, etc.)
        prop: dict[str, Any] = _get_json_type_for_python_type(py_type, type_map)

        # Add description from Annotated[..., Field(...)] if found
        if description:
            prop["description"] = description
        # Fallback: check field metadata (legacy Desc() approach)
        elif f.metadata and "description" in f.metadata:
            prop["description"] = f.metadata["description"]

        properties[f.name] = prop

        # Required if no default
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            required.append(f.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# =============================================================================
# Truncation Functions
# =============================================================================


def _save_full_output(content: str, tool_name: str) -> str:
    """Save full output to temp file, return path.

    Creates a temp file in /tmp/nano_tool_output/ with a unique name.
    Registers the file in _truncated_outputs for cleanup.
    """
    output_dir = Path(tempfile.gettempdir()) / "nano_tool_output"
    output_dir.mkdir(exist_ok=True)
    filename = f"{tool_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}.txt"
    file_path = output_dir / filename
    file_path.write_text(content)

    # Register for cleanup
    _truncated_outputs[str(file_path)] = TruncatedOutput(
        tool_name=tool_name,
        temp_file_path=str(file_path),
        original_chars=len(content),
        original_lines=content.count("\n") + 1,
        created_at=time.time(),
    )

    return str(file_path)


def _truncate_text_content(
    content: TextContent, tool_name: str, config: TruncationConfig
) -> TextContent:
    """Truncate TextContent if it exceeds limits.

    Returns the original content if within limits, otherwise:
    1. Saves full output to temp file
    2. Returns truncated content with notification footer
    """
    text = content.text
    line_count = text.count("\n") + 1

    # Check if truncation is needed
    if len(text) <= config.max_chars and line_count <= config.max_lines:
        return content  # No truncation needed

    # Save full output to temp file
    temp_path = _save_full_output(text, tool_name)

    # Determine truncation method (prefer char limit for consistency)
    if len(text) > config.max_chars:
        truncated = text[: config.max_chars]
    else:
        # Truncate by lines
        truncated = "\n".join(text.splitlines()[: config.max_lines])

    # Build notification
    notification = f"""

───── OUTPUT TRUNCATED ─────
Original: {line_count:,} lines, {len(text):,} chars
Full output: {temp_path}
Use Read or Grep to view full content.
─────────────────────────────"""

    return TextContent(text=truncated + notification)


def cleanup_truncated_outputs(max_age_seconds: int = 3600) -> int:
    """Remove expired temp files older than max_age_seconds.

    Args:
        max_age_seconds: Maximum age in seconds before cleanup (default 1 hour)

    Returns:
        Number of files removed
    """
    current_time = time.time()
    expired = [
        path
        for path, output in _truncated_outputs.items()
        if current_time - output.created_at > max_age_seconds
    ]

    count = 0
    for path in expired:
        try:
            Path(path).unlink(missing_ok=True)
            count += 1
        except Exception:
            pass
        del _truncated_outputs[path]

    return count


def clear_all_truncated_outputs() -> int:
    """Remove all temp files from truncated outputs.

    Returns:
        Number of files removed
    """
    count = 0
    for path in list(_truncated_outputs.keys()):
        try:
            Path(path).unlink(missing_ok=True)
            count += 1
        except Exception:
            pass

    _truncated_outputs.clear()
    return count


class ToolDict(TypedDict):
    name: str
    description: str
    input_schema: InputSchemaDict


@dataclass
class Tool:
    """Base class for all tools with automatic schema inference.

    The input_schema is automatically inferred from the __call__ method's
    'input' parameter type annotation. The input type must be a dataclass.

    Example:
        @dataclass
        class CalculatorInput:
            expr: Annotated[str, Desc("Math expression to evaluate")]

        @dataclass
        class Calculator(Tool):
            name: str = "calculator"
            description: str = "Evaluate a math expression"

            async def __call__(self, input: CalculatorInput) -> TextContent:
                return TextContent(text=str(eval(input.expr)))

    The framework will:
    1. Infer input_schema from CalculatorInput at class definition time
    2. Convert dict → CalculatorInput when execute() is called
    """

    name: str
    description: str

    # Class-level attributes set by __init_subclass__
    _input_type: ClassVar[type | None] = None
    _inferred_schema: ClassVar[InputSchemaDict | None] = None
    _no_input: ClassVar[bool] = False  # True if __call__ has no input parameter
    # Truncation config - override in subclasses to customize or disable
    _truncation_config: ClassVar[TruncationConfig | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Compute input schema from __call__ type annotation at class definition."""
        super().__init_subclass__(**kwargs)

        # Skip if this is an intermediate base class without __call__ override
        if "__call__" not in cls.__dict__:
            return

        try:
            # Check if __call__ has an 'input' parameter at all
            import inspect

            sig = inspect.signature(cls.__call__)
            has_input_param = "input" in sig.parameters

            input_type = get_call_input_type(cls)
            cls._input_type = input_type
            cls._no_input = not has_input_param  # True if no input parameter

            if input_type is not None:
                cls._inferred_schema = schema_from_dataclass(input_type)
            else:
                # No-input tool: empty schema
                cls._inferred_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }
        except TypeError:
            # __call__ doesn't have proper type annotations
            # This could be the base Tool class or old-style tools
            pass

    @property
    def input_schema(self) -> InputSchemaDict:
        """Get the input schema (inferred from __call__ type annotation)."""
        if self._inferred_schema is not None:
            return self._inferred_schema
        # Fallback for tools that don't use type annotations
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def to_dict(self) -> ToolDict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    async def execute(
        self, input: dict[str, Any] | None = None
    ) -> "TextContent | list[TextContent]":
        """Execute the tool with automatic output truncation.

        This is the recommended way to call tools from the framework.
        It handles:
        1. Conversion from raw API dict to typed input
        2. Automatic truncation of large outputs (saves full output to temp file)

        Truncation can be configured per-tool via _truncation_config class variable.
        """
        if self._no_input:
            # No-input tool: call without arguments
            result = await self.__call__()  # type: ignore[call-arg]
        else:
            typed_input = convert_input(input, self._input_type)
            result = await self.__call__(typed_input)

        # Apply truncation if enabled
        config = self._truncation_config or _DEFAULT_TRUNCATION_CONFIG
        if config.enabled:
            if isinstance(result, list):
                result = [
                    _truncate_text_content(tc, self.name, config) for tc in result
                ]
            else:
                result = _truncate_text_content(result, self.name, config)

        return result

    async def __call__(
        self, input: Any
    ) -> "TextContent | list[TextContent]":  # noqa: ARG002
        """Execute the tool with given input. Override in subclasses.

        For no-input tools, simply omit the input parameter in your override.
        """
        raise NotImplementedError(f"{self.name} does not implement __call__()")


@dataclass
class BashTool(Tool):
    """Executes a given bash command in a persistent shell session."""

    name: str = "Bash"
    description: str = """Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.

Usage notes:
  - The command argument is required.
  - You can specify an optional timeout in seconds (up to 600 seconds / 10 minutes).
  - If the output exceeds 30000 characters, output will be truncated."""

    async def __call__(self, input: BashInput) -> TextContent:
        """Execute a bash command with cancellation-safe subprocess handling."""
        import asyncio

        if not input.command:
            return TextContent(text="Error: No command provided")

        process = await asyncio.create_subprocess_shell(
            input.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=input.timeout
            )
            output = stdout.decode() or stderr.decode() or "(no output)"
            return TextContent(text=output)
        except asyncio.CancelledError:
            # Terminate subprocess on cancellation
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            raise  # Re-raise to propagate cancellation
        except asyncio.TimeoutError:
            # Terminate subprocess on timeout
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            return TextContent(text=f"Error: Command timed out after {input.timeout}s")
        except Exception as e:
            # Clean up subprocess on any other error
            if process.returncode is None:
                process.terminate()
            return TextContent(text=f"Error: {e}")


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

    async def __call__(self, input: GlobInput) -> TextContent:
        """Execute glob pattern matching using fd."""
        import asyncio
        import os
        import shutil

        path = input.path or "."
        pattern = input.pattern

        # Check if fd is available
        if shutil.which("fd") is None:
            return TextContent(
                text=(
                    "Error: 'fd' command not found.\n\n"
                    "GlobTool requires 'fd' (a fast file finder) to be installed.\n\n"
                    "Installation instructions:\n"
                    "  macOS:   brew install fd\n"
                    "  Ubuntu:  apt install fd-find  "
                    "(then: ln -s $(which fdfind) ~/.local/bin/fd)\n"
                    "  Arch:    pacman -S fd\n"
                    "  Windows: choco install fd\n\n"
                    "More info: https://github.com/sharkdp/fd"
                )
            )

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

    async def __call__(self, input: GrepInput) -> TextContent:
        """Execute grep search using ripgrep."""
        # Check if ripgrep is available
        if shutil.which("rg") is None:
            return TextContent(
                text=(
                    "Error: 'rg' (ripgrep) command not found.\n\n"
                    "GrepTool requires 'ripgrep' to be installed.\n\n"
                    "Installation instructions:\n"
                    "  macOS:   brew install ripgrep\n"
                    "  Ubuntu:  apt install ripgrep\n"
                    "  Arch:    pacman -S ripgrep\n"
                    "  Windows: choco install ripgrep\n\n"
                    "More info: https://github.com/BurntSushi/ripgrep"
                )
            )

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


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


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

    async def __call__(self, input: ReadInput) -> TextContent:
        """Read file contents with metadata and smart defaults."""
        from pathlib import Path

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


def _cleanup_expired_edits() -> None:
    """Remove expired pending edits."""
    current_time = time.time()
    expired = [
        edit_id
        for edit_id, edit in _pending_edits.items()
        if current_time - edit.created_at > _EDIT_EXPIRY_SECONDS
    ]
    for edit_id in expired:
        del _pending_edits[edit_id]


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
    """Generate a diff-style preview of the edit.

    Returns:
        Tuple of (preview_text, context_before, context_after)
    """
    total_lines = len(lines)

    # Get context lines
    start = max(0, match_line - context_lines)
    end = min(total_lines, match_line + context_lines + old_string.count("\n") + 1)

    context_before = lines[start:match_line]
    context_after = lines[match_line + old_string.count("\n") + 1 : end]

    # Build preview
    preview_parts = []

    # Show lines before
    for i, line in enumerate(lines[start:match_line], start=start + 1):
        preview_parts.append(f"     {i:6}\t{line}")

    # Show old lines (to be removed)
    old_lines = old_string.splitlines()
    for i, line in enumerate(old_lines):
        line_num = match_line + i + 1
        preview_parts.append(f"  -  {line_num:6}\t{line}")

    # Show new lines (to be added)
    new_lines = new_string.splitlines()
    for i, line in enumerate(new_lines):
        line_num = match_line + i + 1
        preview_parts.append(f"  +  {line_num:6}\t{line}")

    # Show lines after
    after_start = match_line + len(old_lines)
    for i, line in enumerate(lines[after_start:end], start=after_start + 1):
        preview_parts.append(f"     {i:6}\t{line}")

    return "\n".join(preview_parts), context_before, context_after


@dataclass
class EditTool(Tool):
    """Performs exact string replacements in files with two-step confirmation.

    This tool uses a preview → confirm workflow to prevent hallucination:
    1. Step 1 (Preview): Returns a diff preview, stores pending edit
    2. Step 2 (Confirm): Use EditConfirmTool to apply or reject

    This ensures the agent sees actual file content before changes are made.
    """

    name: str = "Edit"
    description: str = """Performs exact string replacements in files with preview confirmation.

Usage:
- You must use your Read tool at least once in the conversation before editing.
- Returns a PREVIEW of changes - edit is NOT applied immediately.
- After reviewing preview, call EditConfirm(edit_id="...") to apply.
- The edit will FAIL if old_string is not unique in the file (unless replace_all=True).
- Use replace_all=True for replacing all occurrences across the file.

Workflow:
1. Call Edit(file_path, old_string, new_string) → Returns preview with edit_id
2. Review the preview to verify it matches expectations
3. Call EditConfirm(edit_id="...") to apply, or let it expire to reject"""

    async def __call__(self, input: EditInput) -> TextContent:
        """Generate edit preview and store pending edit for confirmation."""
        # Cleanup expired edits first
        _cleanup_expired_edits()

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

        preview, context_before, context_after = _generate_preview(
            lines, match_line, input.old_string, input.new_string
        )

        # Generate unique edit_id and store pending edit
        edit_id = str(uuid.uuid4())[:8]
        _pending_edits[edit_id] = PendingEdit(
            file_path=input.file_path,
            old_string=input.old_string,
            new_string=input.new_string,
            match_line=match_line,
            context_before=context_before,
            context_after=context_after,
            created_at=time.time(),
            replace_all=input.replace_all,
            match_count=match_count,
        )

        # Build output
        header = f"─── Edit Preview: {input.file_path} ───\n"
        if match_count > 1:
            header += f"Replacing ALL {match_count} occurrences\n"
        else:
            header += f"Match found at line {match_line + 1}\n"
        header += "\n"

        footer = (
            f'\n\n⚠️ Edit NOT applied. Call EditConfirm(edit_id="{edit_id}") to apply.\n'
            f"Edit expires in {_EDIT_EXPIRY_SECONDS // 60} minutes."
        )

        return TextContent(text=header + preview + footer)


@dataclass
class EditConfirmTool(Tool):
    """Confirms and applies a pending edit from EditTool.

    This is the second step in the two-step edit confirmation workflow.
    Use the edit_id from the EditTool preview to apply the edit.
    """

    name: str = "EditConfirm"
    description: str = """Confirms and applies a pending edit from EditTool.

Usage:
- Provide the edit_id from the Edit tool preview
- The edit will be applied to the file
- If the edit_id is invalid or expired, an error is returned

Example:
  EditConfirm(edit_id="abc12345")"""

    async def __call__(self, input: EditConfirmInput) -> TextContent:
        """Apply a pending edit by its ID."""
        _cleanup_expired_edits()

        edit_id = input.edit_id

        if edit_id not in _pending_edits:
            return TextContent(
                text=f"Error: Edit ID '{edit_id}' not found or expired.\n\n"
                "Pending edits expire after 5 minutes. "
                "Use the Edit tool again to create a new preview."
            )

        pending = _pending_edits[edit_id]
        path = Path(pending.file_path)

        # Re-validate file exists
        if not path.exists():
            del _pending_edits[edit_id]
            return TextContent(
                text=f"Error: File no longer exists: {pending.file_path}"
            )

        try:
            content = path.read_text()
        except (PermissionError, UnicodeDecodeError) as e:
            del _pending_edits[edit_id]
            return TextContent(text=f"Error reading file: {e}")

        # Validate old_string still exists
        if pending.old_string not in content:
            del _pending_edits[edit_id]
            return TextContent(
                text=f"Error: The file has been modified since preview.\n"
                "The old_string no longer exists. Use Edit tool again."
            )

        # Apply the edit
        if pending.replace_all:
            new_content = content.replace(pending.old_string, pending.new_string)
        else:
            new_content = content.replace(pending.old_string, pending.new_string, 1)

        try:
            path.write_text(new_content)
        except PermissionError:
            del _pending_edits[edit_id]
            return TextContent(text=f"Error: Permission denied: {pending.file_path}")
        except Exception as e:
            del _pending_edits[edit_id]
            return TextContent(text=f"Error writing file: {e}")

        # Cleanup and confirm
        del _pending_edits[edit_id]

        if pending.replace_all and pending.match_count > 1:
            return TextContent(
                text=f"✓ Edit applied: {pending.file_path}\n"
                f"  Replaced {pending.match_count} occurrences."
            )
        else:
            return TextContent(
                text=f"✓ Edit applied: {pending.file_path}\n"
                f"  Modified at line {pending.match_line + 1}."
            )


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

    async def __call__(self, input: WriteInput) -> TextContent:
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


@dataclass
class WebFetchTool(Tool):
    """Fetches content from a URL and renders it as clean text using lynx."""

    name: str = "WebFetch"
    description: str = """Fetches content from a URL and renders it as clean, readable text.

Uses lynx (text-mode browser) to render HTML pages as plain text, extracting
readable content without raw HTML tags.

Usage:
- URL must be a fully-formed valid URL (https://...)
- HTTP URLs are automatically upgraded to HTTPS
- Output is truncated to 5000 characters maximum
- The prompt parameter describes what to look for in the content

Examples:
  WebFetchInput(url="https://example.com", prompt="Summarize the page content")
  WebFetchInput(url="https://docs.python.org/3/", prompt="Find the tutorial section")

Note: Requires 'lynx' to be installed (brew install lynx)."""

    # Use centralized truncation with 5000 char limit (saves full output to temp file)
    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(max_chars=5000)

    async def __call__(self, input: WebFetchInput) -> TextContent:
        """Fetch URL content using lynx and return as text."""
        # Check if lynx is available
        if shutil.which("lynx") is None:
            return TextContent(
                text=(
                    "Error: 'lynx' command not found.\n\n"
                    "WebFetchTool requires 'lynx' (text-mode browser) to be installed.\n\n"
                    "Installation instructions:\n"
                    "  macOS:   brew install lynx\n"
                    "  Ubuntu:  apt install lynx\n"
                    "  Arch:    pacman -S lynx\n"
                    "  Windows: choco install lynx\n\n"
                    "More info: https://lynx.invisible-island.net/"
                )
            )

        url = input.url

        # Upgrade HTTP to HTTPS
        if url.startswith("http://"):
            url = "https://" + url[7:]

        # Basic URL validation
        if not url.startswith("https://"):
            return TextContent(
                text="Error: Invalid URL. Must start with http:// or https://"
            )

        # Build lynx command
        # -dump: output to stdout
        # -nolist: don't print link list at bottom (cleaner output)
        # -width=120: reasonable line width
        # -accept_all_cookies: handle cookie prompts
        cmd = [
            "lynx",
            "-dump",
            "-nolist",
            "-width=120",
            "-accept_all_cookies",
            url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

            if process.returncode != 0:
                error_msg = stderr.decode(errors="replace").strip() or "Unknown error"
                return TextContent(text=f"Error fetching URL: {error_msg}")

            text = stdout.decode(errors="replace")

            if not text.strip():
                return TextContent(text="Error: Page returned empty content")

            # Build output with metadata
            header = f"─── WebFetch: {input.url} ───\n"
            header += f"Prompt: {input.prompt}\n\n"

            return TextContent(text=header + text)

        except asyncio.TimeoutError:
            return TextContent(text="Error: Request timed out after 30 seconds")
        except Exception as e:
            return TextContent(text=f"Error: {e}")


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

    async def __call__(self, input: StatInput) -> TextContent:
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


@dataclass
class TodoWriteTool(Tool):
    """Create and manage a structured task list for your current coding session.

    This tool now includes state management (merged from TodoManager).
    It tracks todos internally and provides methods to query state.

    Example usage:
        tool = TodoWriteTool()
        await tool({"todos": [...]})  # Updates state and displays
        task = tool.get_current_task()  # Get in-progress task
        tool.display()  # Print formatted todo list
        tool.clear()  # Clear all todos
    """

    name: str = "TodoWrite"
    description: str = """Use this tool to create and manage a structured task list for your current coding session.

Task States:
- pending: Task not yet started
- in_progress: Currently working on (limit to ONE task at a time)
- completed: Task finished successfully

Task Management:
- Update task status in real-time as you work
- Mark tasks complete IMMEDIATELY after finishing
- Exactly ONE task must be in_progress at any time"""

    # State management (merged from TodoManager)
    _todos: list[Todo] = field(default_factory=list, repr=False)

    # Class constants for display
    STATUS_INDICATORS: ClassVar[dict[TodoStatus, str]] = {
        TodoStatus.PENDING: "[ ]",
        TodoStatus.IN_PROGRESS: "[~]",
        TodoStatus.COMPLETED: "[x]",
    }

    TOOL_RESULT_MESSAGE: ClassVar[str] = (
        "Todos have been modified successfully. "
        "Ensure that you continue to use the todo list to track your progress. "
        "Please proceed with the current tasks if applicable"
    )

    # ==========================================================================
    # State Management Methods (from TodoManager)
    # ==========================================================================

    @property
    def todos(self) -> list[Todo]:
        """Get all todos (returns a copy for safety)."""
        return self._todos.copy()

    def get_current_task(self) -> Todo | None:
        """Get the currently in-progress task, if any."""
        for todo in self._todos:
            if todo.status == TodoStatus.IN_PROGRESS:
                return todo
        return None

    def clear(self) -> None:
        """Clear all todos."""
        self._todos = []

    def display(self) -> None:
        """Print the todo list to console with status indicators."""
        if not self._todos:
            print("No todos.")
            return

        print("\n--- Todo List ---")
        for i, todo in enumerate(self._todos, 1):
            indicator = self.STATUS_INDICATORS[todo.status]
            print(f"{i}. {indicator} {todo.content}")

            # Show active form for in-progress items
            if todo.status == TodoStatus.IN_PROGRESS and todo.active_form:
                print(f"      {todo.active_form}...")

        # Summary
        total = len(self._todos)
        completed = sum(1 for t in self._todos if t.status == TodoStatus.COMPLETED)
        in_progress = sum(1 for t in self._todos if t.status == TodoStatus.IN_PROGRESS)
        pending = sum(1 for t in self._todos if t.status == TodoStatus.PENDING)

        print(f"\nProgress: {completed}/{total} completed", end="")
        if in_progress:
            print(f", {in_progress} in progress", end="")
        if pending:
            print(f", {pending} pending", end="")
        print("\n")

    # ==========================================================================
    # Tool Execution
    # ==========================================================================

    async def __call__(self, input: TodoWriteInput) -> TextContent:
        """Update the todo list, display it, and return a summary for the model."""
        # Convert TodoItemInput instances to Todo objects
        self._todos = [
            Todo(
                content=item.content,
                status=TodoStatus(item.status),
                active_form=item.activeForm,
            )
            for item in input.todos
        ]

        # Display to console
        self.display()

        # Return message for model
        return TextContent(text=self.TOOL_RESULT_MESSAGE)


# =============================================================================
# Python Tool - Create, Edit, and Run Python scripts
# =============================================================================


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
    import tempfile

    script_dir = Path(tempfile.gettempdir()) / _PYTHON_SCRIPT_PREFIX
    script_dir.mkdir(exist_ok=True)
    return script_dir


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

        # Check if uv is available
        if shutil.which("uv") is None:
            return TextContent(
                text=(
                    "Error: 'uv' command not found.\n\n"
                    "PythonTool requires 'uv' for dependency management.\n\n"
                    "Installation instructions:\n"
                    "  pip:     pip install uv\n"
                    "  pipx:    pipx install uv\n"
                    "  macOS:   brew install uv\n"
                    "  curl:    curl -LsSf https://astral.sh/uv/install.sh | sh\n\n"
                    "More info: https://github.com/astral-sh/uv"
                )
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
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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


# Default set of all built-in tools
# Note: WebSearchTool is excluded because it is a stub that raises NotImplementedError
DEFAULT_TOOLS: list[Tool] = [
    BashTool(),
    GlobTool(),
    GrepTool(),
    ReadTool(),
    StatTool(),
    EditTool(),
    EditConfirmTool(),
    WriteTool(),
    WebFetchTool(),
    TodoWriteTool(),
    PythonTool(),
]
