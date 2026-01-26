"""nano_agent: a minimalistic functional asynchronous agent

A Python library for building agent conversation systems using the Claude API.
This library implements a node-based conversation graph architecture where everything
(system prompts, tool definitions, messages, tool executions, stop reasons) is represented
as nodes in a directed acyclic graph (DAG).
"""

# API base classes (shared infrastructure)
from .api_base import APIClientMixin, APIError, APIProtocol

# Cancellation support
from .cancellation import CancellationToken

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
from .dummy_api import DummyAPI

# Core graph primitives
from .dag import DAG, Node

# Data structures
from .data_structures import (  # Message and roles; Content blocks; Node data types
    ContentBlock,
    Message,
    NodeData,
    Role,
    StopReason,
    SystemPrompt,
    TextContent,
    ThinkingContent,
    ToolDefinitions,
    ToolExecution,
    ToolResultContent,
    ToolUseContent,
)

# Executor
from .executor import run
from .gemini_api import GeminiAPI
from .openai_api import OpenAIAPI

# Tools (including Todo data classes merged from tool_handlers)
from .tools import (
    BashInput,
    BashTool,
    Desc,
    EditConfirmInput,
    EditConfirmTool,
    EditInput,
    EditTool,
    Field,
    GlobInput,
    GlobTool,
    PendingEdit,
    PythonInput,
    PythonScript,
    PythonTool,
    Question,
    QuestionOption,
    ReadInput,
    ReadTool,
    GrepInput,
    GrepTool,
    StatInput,
    StatTool,
    Todo,
    TodoItemInput,
    TodoStatus,
    TodoWriteInput,
    TodoWriteTool,
    Tool,
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
    "DummyAPI",
    "GeminiAPI",
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
    "Usage",
    # Tools
    "Tool",
    "schema_from_dataclass",
    "get_call_input_type",
    "convert_input",
    "Field",
    "Desc",
    "PendingEdit",
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
    "EditConfirmTool",
    "EditConfirmInput",
    "WriteTool",
    "WriteInput",
    "WebFetchTool",
    "WebFetchInput",
    "TodoWriteTool",
    "TodoWriteInput",
    "TodoItemInput",
    "Question",
    "QuestionOption",
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
    # Executor
    "run",
]
