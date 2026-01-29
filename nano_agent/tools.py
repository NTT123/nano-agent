"""Built-in tool definitions for Claude API.

This module re-exports all tools from the nano_agent.tools package
for backward compatibility. New code should import directly from
nano_agent.tools instead.

Example:
    # Old style (still works):
    from nano_agent.tools import BashTool

    # Preferred style:
    from nano_agent.tools import BashTool
"""

# Re-export everything from the tools package
from nano_agent.tools import (
    _DEFAULT_TRUNCATION_CONFIG,
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
    InputSchemaDict,
    PythonInput,
    PythonScript,
    PythonTool,
    Question,
    QuestionOption,
    ReadInput,
    ReadTool,
    StatInput,
    StatTool,
    Todo,
    TodoItemInput,
    TodoStatus,
    TodoWriteInput,
    TodoWriteTool,
    Tool,
    ToolDict,
    TruncatedOutput,
    TruncationConfig,
    WebFetchInput,
    WebFetchTool,
    WriteInput,
    WriteTool,
    _python_scripts,
    _save_full_output,
    _truncate_text_content,
    _truncated_outputs,
    cleanup_truncated_outputs,
    clear_all_truncated_outputs,
    clear_python_scripts,
    convert_input,
    get_call_input_type,
    get_default_tools,
    list_python_scripts,
    schema_from_dataclass,
)

__all__ = [
    # Base classes and utilities
    "Tool",
    "ToolDict",
    "InputSchemaDict",
    "Desc",
    "Field",
    "TruncationConfig",
    "TruncatedOutput",
    "get_call_input_type",
    "convert_input",
    "schema_from_dataclass",
    "cleanup_truncated_outputs",
    "clear_all_truncated_outputs",
    # Question data classes
    "Question",
    "QuestionOption",
    # AskUserQuestion tool
    "AskUserQuestionTool",
    "AskUserQuestionInput",
    # Bash tool
    "BashTool",
    "BashInput",
    # Glob tool
    "GlobTool",
    "GlobInput",
    # Grep tool
    "GrepTool",
    "GrepInput",
    # Read tool
    "ReadTool",
    "ReadInput",
    # Edit tools
    "EditTool",
    "EditInput",
    # Write tool
    "WriteTool",
    "WriteInput",
    # WebFetch tool
    "WebFetchTool",
    "WebFetchInput",
    # Stat tool
    "StatTool",
    "StatInput",
    # Todo tool
    "TodoWriteTool",
    "TodoWriteInput",
    "TodoItemInput",
    "Todo",
    "TodoStatus",
    # Python tool
    "PythonTool",
    "PythonInput",
    "PythonScript",
    "list_python_scripts",
    "clear_python_scripts",
    # Default tools function
    "get_default_tools",
]
