"""nano_agent: a minimalistic functional asynchronous agent

A Python library for building agent conversation systems using the Claude API.
This library implements a node-based conversation graph architecture where everything
(system prompts, tool definitions, messages, tool executions, stop reasons) is represented
as nodes in a directed acyclic graph (DAG).
"""

# API base classes (shared infrastructure)
from .api_base import APIClientMixin, APIError, APIProtocol

# Cancellation support
from .cancellation import (
    CancellationChoice,
    CancellationToken,
    ToolExecutionBatch,
    ToolExecutionStatus,
    TrackedToolCall,
)

# Auth capture utilities
from .capture_claude_code_auth import (
    DEFAULT_CONFIG_PATH,
    get_config,
    get_headers,
    load_config,
    save_config,
)

# API clients
from .claude_api import ClaudeAPI, Response, Usage
from .claude_code_api import ClaudeCodeAPI
from .codex_api import CodexAPI
from .codex_auth import (
    DEFAULT_CODEX_AUTH_PATH,
    get_codex_access_token,
    get_codex_refresh_token,
    load_codex_auth,
)

# Cost tracking
from .cost import CostBreakdown, ModelPricing, calculate_cost, format_cost, get_pricing

# Core graph primitives
from .dag import DAG, Node

# Data structures - Core types
from .data_structures import (  # Enums; Content blocks (sum type: ContentBlock); Messages; Node data (sum type: NodeData); Exhaustiveness helper; JSON type aliases; Serialization TypedDicts (for type-safe dict handling); Sub-agent support
    ContentBlock,
    ContentBlockDict,
    JSONObject,
    JSONSchema,
    JSONValue,
    Message,
    MessageDict,
    NodeData,
    ResponseDict,
    Role,
    StopReason,
    StopReasonDict,
    SubGraph,
    SummaryItem,
    SystemPrompt,
    SystemPromptDict,
    TextContent,
    TextContentDict,
    ThinkingContent,
    ThinkingContentDict,
    ToolDefinitionDict,
    ToolDefinitions,
    ToolDefinitionsDict,
    ToolExecution,
    ToolExecutionDict,
    ToolResultContent,
    ToolResultContentDict,
    ToolUseContent,
    ToolUseContentDict,
    UsageDict,
    assert_never,
)

# Execution context and sub-agent support
from .execution_context import ExecutionContext, run_sub_agent

# Executor
from .executor import run
from .fireworks_api import FireworksAPI
from .gemini_api import GeminiAPI
from .openai_api import OpenAIAPI

# Tools (including Todo data classes merged from tool_handlers)
from .tools import (
    AskUserQuestionInput,
    AskUserQuestionTool,
    BashInput,
    BashTool,
    Desc,
    EditInput,
    EditTool,
    Field,
    GlobInput,
    GlobTool,
    GrepInput,
    GrepTool,
    PermissionCallback,
    PythonInput,
    PythonScript,
    PythonTool,
    Question,
    QuestionOption,
    ReadInput,
    ReadTool,
    StatInput,
    StatTool,
    SubAgentTool,
    Todo,
    TodoItemInput,
    TodoStatus,
    TodoWriteInput,
    TodoWriteTool,
    Tool,
    ToolResult,
    TruncatedOutput,
    TruncationConfig,
    WebFetchInput,
    WebFetchTool,
    WriteInput,
    WriteTool,
    cleanup_truncated_outputs,
    clear_all_truncated_outputs,
    clear_python_scripts,
    convert_input,
    get_call_input_type,
    get_default_tools,
    list_python_scripts,
    schema_from_dataclass,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Node",
    "DAG",
    "ClaudeAPI",
    "ClaudeCodeAPI",
    "FireworksAPI",
    "GeminiAPI",
    "CodexAPI",
    "OpenAIAPI",
    "Response",
    # API base classes
    "APIError",
    "APIClientMixin",
    "APIProtocol",
    # Cancellation support
    "CancellationToken",
    # Auth capture utilities
    "get_config",
    "get_headers",
    "load_config",
    "save_config",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CODEX_AUTH_PATH",
    # Data structures
    "Message",
    "Role",
    "ContentBlock",
    "TextContent",
    "ThinkingContent",
    "ToolUseContent",
    "ToolResultContent",
    "NodeData",
    "SystemPrompt",
    "ToolDefinitions",
    "ToolExecution",
    "StopReason",
    "SubGraph",
    "Usage",
    # Execution context and sub-agent support
    "ExecutionContext",
    "run_sub_agent",
    # Tools
    "Tool",
    "SubAgentTool",
    "ToolResult",
    "schema_from_dataclass",
    "get_call_input_type",
    "convert_input",
    "Field",
    "Desc",
    "PermissionCallback",
    "BashTool",
    "BashInput",
    "GlobTool",
    "GlobInput",
    "GrepTool",
    "GrepInput",
    "ReadTool",
    "ReadInput",
    "StatTool",
    "StatInput",
    "EditTool",
    "EditInput",
    "WriteTool",
    "WriteInput",
    "WebFetchTool",
    "WebFetchInput",
    "TodoWriteTool",
    "TodoWriteInput",
    "TodoItemInput",
    "Question",
    "QuestionOption",
    "AskUserQuestionTool",
    "AskUserQuestionInput",
    "PythonTool",
    "PythonInput",
    "PythonScript",
    "list_python_scripts",
    "clear_python_scripts",
    # Truncation utilities
    "TruncationConfig",
    "TruncatedOutput",
    "cleanup_truncated_outputs",
    "clear_all_truncated_outputs",
    # Todo data classes (merged from tool_handlers)
    "Todo",
    "TodoStatus",
    # Cost tracking
    "CostBreakdown",
    "ModelPricing",
    "calculate_cost",
    "format_cost",
    "get_pricing",
    # Executor
    "run",
    # Default tools function
    "get_default_tools",
    # Codex auth helpers
    "load_codex_auth",
    "get_codex_access_token",
    "get_codex_refresh_token",
]
