"""
Core data structures for the agent conversation graph.

This module defines all the dataclasses used throughout the library,
providing type-safe alternatives to dictionaries.

Uses algebraic data types (sum types via Union) for type-safe
pattern matching and exhaustiveness checking.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Never, TypeAlias, TypedDict

# =============================================================================
# JSON Type Aliases
# =============================================================================

# Recursive JSON value type (for strict JSON typing)
JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)

# JSON object type
JSONObject: TypeAlias = dict[str, JSONValue]


# =============================================================================
# API-specific TypedDicts
# =============================================================================


class SummaryItem(TypedDict, total=False):
    """OpenAI reasoning summary item."""

    type: str  # e.g., "summary_text"
    text: str


class JSONSchemaProperty(TypedDict, total=False):
    """JSON Schema property definition."""

    type: str
    description: str
    enum: list[str]
    items: "JSONSchemaProperty"
    properties: dict[str, "JSONSchemaProperty"]
    required: list[str]
    additionalProperties: bool
    default: JSONValue


class JSONSchema(TypedDict, total=False):
    """JSON Schema for tool input validation."""

    type: str
    properties: dict[str, JSONSchemaProperty]
    required: list[str]
    additionalProperties: bool


# =============================================================================
# Serialization TypedDicts (for to_dict/from_dict methods)
# =============================================================================


class TextContentDict(TypedDict):
    """Serialized form of TextContent."""

    type: str
    text: str


class ThinkingContentDict(TypedDict, total=False):
    """Serialized form of ThinkingContent."""

    type: str
    thinking: str
    signature: str
    # OpenAI reasoning format
    id: str
    encrypted_content: str
    summary: list[SummaryItem]


class ToolUseContentDict(TypedDict):
    """Serialized form of ToolUseContent."""

    type: str
    id: str
    name: str
    input: dict[str, JSONValue] | None


class ToolResultContentDict(TypedDict):
    """Serialized form of ToolResultContent."""

    type: str
    tool_use_id: str
    content: list[TextContentDict]
    is_error: bool


# Forward reference for MessageDict (content can be string or list of content blocks)
ContentBlockDict = (
    TextContentDict | ThinkingContentDict | ToolUseContentDict | ToolResultContentDict
)


class MessageDict(TypedDict):
    """Serialized form of Message."""

    role: str
    content: str | list[ContentBlockDict]


class SystemPromptDict(TypedDict):
    """Serialized form of SystemPrompt."""

    type: str
    content: str


class ToolDefinitionDict(TypedDict):
    """Serialized form of ToolDefinition."""

    name: str
    description: str
    input_schema: dict[
        str, object
    ]  # JSON Schema (loosely typed for dynamic generation)


class ToolDefinitionsDict(TypedDict):
    """Serialized form of ToolDefinitions."""

    type: str
    tools: list[ToolDefinitionDict]


class ToolExecutionDict(TypedDict):
    """Serialized form of ToolExecution."""

    type: str
    tool_name: str
    tool_use_id: str
    result: list[TextContentDict]
    is_error: bool


class StopReasonDict(TypedDict):
    """Serialized form of StopReason."""

    type: str
    reason: str
    usage: dict[str, int]


class UsageDict(TypedDict, total=False):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    # OpenAI/Codex specific fields
    reasoning_tokens: int
    cached_tokens: int
    total_tokens: int


class ResponseDict(TypedDict, total=False):
    """Serialized form of Response."""

    id: str
    model: str
    role: str
    content: list[ContentBlockDict]
    stop_reason: str | None
    usage: UsageDict


# =============================================================================
# Exhaustiveness Helper
# =============================================================================


def assert_never(value: Never) -> Never:
    """Assert that a value is never reached (for exhaustive pattern matching).

    Usage:
        def handle(block: ContentBlock) -> str:
            match block:
                case TextContent(text=t):
                    return t
                case ThinkingContent(thinking=t):
                    return f"[{t}]"
                case ToolUseContent(name=n):
                    return f"tool:{n}"
                case ToolResultContent():
                    return "result"
                case _ as unreachable:
                    assert_never(unreachable)  # Type error if cases missed
    """
    raise AssertionError(f"Unexpected value: {value!r}")


# =============================================================================
# Enums
# =============================================================================


class Role(str, Enum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"


class NodeType(str, Enum):
    """Types of nodes in the conversation graph."""

    SYSTEM_PROMPT = "system_prompt"
    TOOL_DEFINITIONS = "tool_definitions"
    TOOL_EXECUTION = "tool_execution"
    MESSAGE = "message"
    STOP_REASON = "stop_reason"


# =============================================================================
# Content Blocks (for Message content)
# =============================================================================


@dataclass(frozen=True)
class TextContent:
    """Plain text content block."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")

    def to_dict(self) -> TextContentDict:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: TextContentDict) -> "TextContent":
        return cls(text=str(data.get("text", "")))


@dataclass(frozen=True)
class ThinkingContent:
    """Model's thinking/reasoning content block.

    Supports both Claude thinking (thinking + signature) and
    OpenAI reasoning (id + encrypted_content + summary for multi-turn).
    """

    thinking: str
    signature: str = ""
    # OpenAI reasoning fields (for multi-turn conversations)
    id: str = ""
    encrypted_content: str = ""
    summary: tuple[SummaryItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.thinking, str):
            raise TypeError(f"thinking must be str, got {type(self.thinking).__name__}")
        if not isinstance(self.signature, str):
            raise TypeError(
                f"signature must be str, got {type(self.signature).__name__}"
            )

    def to_dict(self) -> ThinkingContentDict:
        """Convert to dict for API calls."""
        # OpenAI reasoning format (has encrypted_content)
        if self.encrypted_content:
            return {
                "type": "reasoning",
                "id": self.id,
                "encrypted_content": self.encrypted_content,
                "summary": list(self.summary),
            }
        # Claude thinking format
        return {
            "type": "thinking",
            "thinking": self.thinking,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: ThinkingContentDict) -> "ThinkingContent":
        return cls(
            thinking=str(data.get("thinking", "")),
            signature=str(data.get("signature", "")),
            id=str(data.get("id", "")),
            encrypted_content=str(data.get("encrypted_content", "")),
            summary=tuple(data.get("summary", [])),
        )


@dataclass(frozen=True)
class ToolUseContent:
    """Tool invocation content block."""

    id: str  # Claude: tool_use id, OpenAI: call_id (for tool results)
    name: str
    input: dict[str, JSONValue] | None = None
    item_id: str | None = None  # OpenAI only: output item id (fc_...)
    thought_signature: str | None = None  # Gemini 3 only: encrypted reasoning context

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.name:
            raise ValueError("name cannot be empty")
        if self.input is not None and not isinstance(self.input, dict):
            raise TypeError(
                f"input must be dict or None, got {type(self.input).__name__}"
            )

    def to_dict(self) -> ToolUseContentDict:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }

    @classmethod
    def from_dict(cls, data: ToolUseContentDict) -> "ToolUseContent":
        tool_input = data.get("input")
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            input=tool_input if isinstance(tool_input, dict) else None,
        )


@dataclass(frozen=True)
class ToolResultContent:
    """Tool result content block."""

    tool_use_id: str
    content: list[TextContent]
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.tool_use_id:
            raise ValueError("tool_use_id cannot be empty")

    def to_dict(self) -> ToolResultContentDict:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": [block.to_dict() for block in self.content],
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: ToolResultContentDict) -> "ToolResultContent":
        raw_content = data.get("content", [])
        if isinstance(raw_content, str):
            content = [TextContent(text=raw_content)]
        elif isinstance(raw_content, list):
            content = [TextContent.from_dict(c) for c in raw_content]
        else:
            content = [TextContent(text=str(raw_content))]
        return cls(
            tool_use_id=str(data.get("tool_use_id", "")),
            content=content,
            is_error=bool(data.get("is_error", False)),
        )


# Sum type for content blocks (algebraic data type)
# Use class-based pattern matching: match block: case TextContent(): ...
ContentBlock = TextContent | ThinkingContent | ToolUseContent | ToolResultContent


# =============================================================================
# Message
# =============================================================================


@dataclass(frozen=True)
class Message:
    """A conversation message."""

    role: Role
    content: str | Sequence[ContentBlock]

    def to_dict(self) -> MessageDict:
        if isinstance(self.content, str):
            content_dict: str | list[ContentBlockDict] = self.content
        else:
            content_dict = [block.to_dict() for block in self.content]
        return {"role": self.role.value, "content": content_dict}


# =============================================================================
# Node Data Types (what goes in Node.data)
# =============================================================================


@dataclass(frozen=True)
class SystemPrompt:
    """System prompt for the conversation."""

    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, got {type(self.content).__name__}")

    def to_dict(self) -> SystemPromptDict:
        return {"type": "system_prompt", "content": self.content}

    @classmethod
    def from_dict(cls, data: SystemPromptDict) -> "SystemPrompt":
        return cls(content=str(data.get("content", "")))


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a single tool."""

    name: str
    description: str
    input_schema: Mapping[
        str, object
    ]  # JSON Schema (use JSONSchema TypedDict for manual construction)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")
        if not isinstance(self.description, str):
            raise TypeError(
                f"description must be str, got {type(self.description).__name__}"
            )
        if not isinstance(self.input_schema, Mapping):
            raise TypeError(
                f"input_schema must be Mapping, got {type(self.input_schema).__name__}"
            )

    def to_dict(self) -> ToolDefinitionDict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True)
class ToolDefinitions:
    """Collection of tool definitions."""

    tools: list[ToolDefinition]

    def to_dict(self) -> ToolDefinitionsDict:
        return {
            "type": "tool_definitions",
            "tools": [t.to_dict() for t in self.tools],
        }

    @classmethod
    def from_dict(cls, data: ToolDefinitionsDict) -> "ToolDefinitions":
        tools_raw = data.get("tools", [])
        tool_defs = [
            ToolDefinition(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("input_schema", {}),
            )
            for t in tools_raw
        ]
        return cls(tools=tool_defs)


@dataclass(frozen=True)
class ToolExecution:
    """Result of executing a tool (for visualization nodes)."""

    tool_name: str
    tool_use_id: str
    result: list[TextContent]
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name cannot be empty")
        if not self.tool_use_id:
            raise ValueError("tool_use_id cannot be empty")

    def to_dict(self) -> ToolExecutionDict:
        return {
            "type": "tool_execution",
            "tool_name": self.tool_name,
            "tool_use_id": self.tool_use_id,
            "result": [block.to_dict() for block in self.result],
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: ToolExecutionDict) -> "ToolExecution":
        raw_result = data.get("result", [])
        if isinstance(raw_result, str):
            result = [TextContent(text=raw_result)]
        elif isinstance(raw_result, list):
            result = [TextContent.from_dict(r) for r in raw_result]
        else:
            result = [TextContent(text=str(raw_result))]
        return cls(
            tool_name=str(data.get("tool_name", "")),
            tool_use_id=str(data.get("tool_use_id", "")),
            result=result,
            is_error=bool(data.get("is_error", False)),
        )


@dataclass(frozen=True)
class StopReason:
    """Stop reason marker (for visualization)."""

    reason: str
    usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("reason cannot be empty")
        if not isinstance(self.usage, dict):
            raise TypeError(f"usage must be dict, got {type(self.usage).__name__}")

    def to_dict(self) -> StopReasonDict:
        return {
            "type": "stop_reason",
            "reason": self.reason,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: StopReasonDict) -> "StopReason":
        return cls(
            reason=str(data.get("reason", "")),
            usage=dict(data.get("usage", {})),
        )


# Sum type for node data (algebraic data type)
# Use class-based pattern matching: match data: case Message(): ...
NodeData = Message | SystemPrompt | ToolDefinitions | ToolExecution | StopReason


# =============================================================================
# API Response Types
# =============================================================================


@dataclass
class Usage:
    """Token usage statistics from API responses."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # OpenAI/Codex specific fields
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
        }


class Response:
    """API response from Claude."""

    def __init__(
        self,
        id: str,
        model: str,
        role: Role,
        content: list[ContentBlock],
        stop_reason: str | None,
        usage: Usage,
    ):
        self.id = id
        self.model = model
        self.role = role
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage

    def get_text(self) -> str:
        """Extract text content from response."""
        for block in self.content:
            if isinstance(block, TextContent):
                return block.text
        return ""

    def get_tool_use(self) -> list[ToolUseContent]:
        """Extract all tool_use blocks from response."""
        return [block for block in self.content if isinstance(block, ToolUseContent)]

    def get_thinking(self) -> list[ThinkingContent]:
        """Extract all thinking blocks from response."""
        return [block for block in self.content if isinstance(block, ThinkingContent)]

    def has_tool_use(self) -> bool:
        """Check if response contains any tool_use blocks."""
        return any(isinstance(block, ToolUseContent) for block in self.content)

    @classmethod
    def from_dict(cls, data: ResponseDict) -> "Response":
        content: list[ContentBlock] = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content.append(TextContent(text=str(block.get("text", ""))))
            elif block["type"] == "thinking":
                content.append(
                    ThinkingContent(
                        thinking=str(block.get("thinking", "")),
                        signature=str(block.get("signature", "")),
                    )
                )
            elif block["type"] == "tool_use":
                tool_input = block.get("input")
                content.append(
                    ToolUseContent(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        input=tool_input if isinstance(tool_input, dict) else None,
                    )
                )

        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_creation_input_tokens=usage_data.get(
                "cache_creation_input_tokens", 0
            ),
            cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
        )

        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            role=Role(data.get("role", "assistant")),
            content=content,
            stop_reason=data.get("stop_reason"),
            usage=usage,
        )
