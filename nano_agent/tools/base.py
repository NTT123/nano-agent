"""Base tool class and shared utilities."""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
import time
import types
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ..data_structures import SubGraph, TextContent

if TYPE_CHECKING:
    from ..execution_context import ExecutionContext


# =============================================================================
# Tool Result (pure functional return value)
# =============================================================================


@dataclass
class ToolResult:
    """Result from tool execution, optionally includes sub-agent graph.

    This is the pure functional return value from Tool.execute().
    For regular tools, only content is set.
    For SubAgentTool, both content and sub_graph are set.

    Attributes:
        content: The tool output (text content or list of text contents)
        sub_graph: Optional sub-agent execution graph (for SubAgentTool)
    """

    content: "TextContent | list[TextContent]"
    sub_graph: "SubGraph | None" = None


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
# Schema Generation Utilities
# =============================================================================

# Loose type for schema dicts (allows dynamic schema generation)
# For stricter typing, use JSONSchema from data_structures
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
# Question Data Classes (for AskUserQuestion-like tools)
# =============================================================================


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
    allowCustom: Annotated[
        bool, Desc("Set to true to allow a freeform custom response")
    ] = False
    customLabel: Annotated[str, Desc("Label for the custom response option")] = (
        "Other..."
    )
    customPrompt: Annotated[
        str, Desc("Prompt text for collecting a custom response")
    ] = "Your answer: "


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


# =============================================================================
# Helper Functions
# =============================================================================


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


# =============================================================================
# Tool Base Class
# =============================================================================


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
    # Required CLI commands - override in subclasses that need external CLI tools
    # Format: {"command": "install instructions"}
    _required_commands: ClassVar[dict[str, str]] = {}

    def __post_init__(self) -> None:
        """Validate that required CLI commands are available."""
        missing_commands = []
        for cmd, install_hint in self._required_commands.items():
            if shutil.which(cmd) is None:
                if cmd == "python" and shutil.which("python3") is not None:
                    continue
                missing_commands.append((cmd, install_hint))

        if missing_commands:
            error_parts = [f"Missing required CLI command(s) for {self.name}:"]
            for cmd, hint in missing_commands:
                error_parts.append(f"  - '{cmd}': {hint}")
            raise RuntimeError("\n".join(error_parts))

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
        self,
        input: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        """Execute the tool with automatic output truncation.

        This is the recommended way to call tools from the framework.
        It handles:
        1. Conversion from raw API dict to typed input
        2. Automatic truncation of large outputs (saves full output to temp file)
        3. Passing ExecutionContext to SubAgentTool.__call__

        Args:
            input: Raw input dict from API (will be converted to typed dataclass)
            execution_context: Optional context for sub-agent tools

        Returns:
            ToolResult containing content and optionally sub_graph

        Truncation can be configured per-tool via _truncation_config class variable.
        """
        # Call the tool
        result: TextContent | list[TextContent] | ToolResult
        if self._no_input:
            # No-input tool: call without arguments
            if isinstance(self, SubAgentTool):
                result = await self.__call__(execution_context=execution_context)  # type: ignore[call-arg]
            else:
                result = await self.__call__()  # type: ignore[call-arg]
        else:
            typed_input = convert_input(input, self._input_type)
            if isinstance(self, SubAgentTool):
                result = await self.__call__(
                    typed_input, execution_context=execution_context
                )
            else:
                result = await self.__call__(typed_input)

        # Wrap in ToolResult if not already
        tool_result: ToolResult
        if isinstance(result, ToolResult):
            tool_result = result
        else:
            tool_result = ToolResult(content=result)

        # Apply truncation if enabled
        config = self._truncation_config or _DEFAULT_TRUNCATION_CONFIG
        if config.enabled:
            content = tool_result.content
            if isinstance(content, list):
                content = [
                    _truncate_text_content(tc, self.name, config) for tc in content
                ]
            else:
                content = _truncate_text_content(content, self.name, config)
            # Create new ToolResult with truncated content (preserve sub_graph)
            tool_result = ToolResult(content=content, sub_graph=tool_result.sub_graph)

        return tool_result

    async def __call__(
        self,
        input: Any,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent | list[TextContent] | ToolResult:  # noqa: ARG002
        """Execute the tool with given input. Override in subclasses.

        For no-input tools, simply omit the input parameter in your override.
        Regular tools can ignore execution_context parameter.
        SubAgentTool implementations should use execution_context for spawning.
        """
        raise NotImplementedError(f"{self.name} does not implement __call__()")


# =============================================================================
# Sub-Agent Tool Base Class
# =============================================================================


@dataclass
class SubAgentTool(Tool):
    """Base class for tools that spawn sub-agents (pure functional).

    This class provides a `spawn()` helper that handles the boilerplate of
    running sub-agents. Unlike the old mutation-based design, this is pure
    functional - the execution context is passed as a parameter to __call__,
    and spawn() returns both the summary and the SubGraph.

    Example:
        @dataclass
        class SecurityAuditInput:
            file_path: Annotated[str, Desc("Path to the file to audit")]

        @dataclass
        class SecurityAuditTool(SubAgentTool):
            name: str = "SecurityAudit"
            description: str = "Spawn a sub-agent to audit code for security issues"

            async def __call__(
                self,
                input: SecurityAuditInput,
                execution_context: ExecutionContext | None = None,
            ) -> ToolResult:
                if not execution_context:
                    return ToolResult(content=TextContent(text="Error: No context"))

                summary, sub_graph = await self.spawn(
                    context=execution_context,
                    system_prompt="You are an expert security auditor...",
                    user_message=f"Audit the file: {input.file_path}",
                    tools=[ReadTool()],
                )
                return ToolResult(
                    content=TextContent(text=summary),
                    sub_graph=sub_graph,
                )

    The base class provides:
    - spawn() helper for running sub-agents
    - Pure functional design (no mutation)
    """

    async def __call__(
        self,
        input: Any,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        """Execute the sub-agent tool. Override in subclasses.

        Args:
            input: The typed input dataclass
            execution_context: Execution context for spawning sub-agents

        Returns:
            ToolResult containing content and optionally sub_graph
        """
        raise NotImplementedError(f"{self.name} does not implement __call__()")

    async def spawn(
        self,
        context: ExecutionContext,
        system_prompt: str,
        user_message: str,
        tools: Sequence[Tool] | None = None,
        tool_name: str | None = None,
    ) -> tuple[str, SubGraph]:
        """Spawn a sub-agent and return its summary and graph.

        This is the main helper method that simplifies sub-agent creation.

        Args:
            context: Execution context (required, provides API access)
            system_prompt: System prompt for the sub-agent
            user_message: Initial user message/task for the sub-agent
            tools: Optional list of tools for the sub-agent
            tool_name: Name for the sub-agent (defaults to self.name)

        Returns:
            Tuple of (summary text, SubGraph)
        """
        # Import here to avoid circular dependency
        from ..execution_context import run_sub_agent

        _, sub_graph = await run_sub_agent(
            context=context,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            tool_name=tool_name or self.name,
        )

        return sub_graph.summary, sub_graph
