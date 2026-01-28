"""Built-in tool definitions for Claude API.

This package provides a collection of tools that can be used with the Claude API.
Each tool is in its own module for better organization and maintainability.
"""

from .base import (
    Desc,
    Field,
    InputSchemaDict,
    Question,
    QuestionOption,
    Tool,
    ToolDict,
    TruncatedOutput,
    TruncationConfig,
    _DEFAULT_TRUNCATION_CONFIG,
    _save_full_output,
    _truncate_text_content,
    _truncated_outputs,
    cleanup_truncated_outputs,
    clear_all_truncated_outputs,
    convert_input,
    get_call_input_type,
    schema_from_dataclass,
)
from .ask_user_question import AskUserQuestionInput, AskUserQuestionTool
from .bash import BashInput, BashTool
from .edit import (
    EditInput,
    EditTool,
    PermissionCallback,
)
from .glob import GlobInput, GlobTool
from .grep import GrepInput, GrepTool
from .python import (
    PythonInput,
    PythonScript,
    PythonTool,
    _python_scripts,
    clear_python_scripts,
    list_python_scripts,
)
from .read import ReadInput, ReadTool
from .stat import StatInput, StatTool
from .todo import Todo, TodoItemInput, TodoStatus, TodoWriteInput, TodoWriteTool
from .tmux import TmuxInput, TmuxTool
from .webfetch import WebFetchInput, WebFetchTool
from .write import WriteInput, WriteTool

def get_default_tools() -> list[Tool]:
    """Get the default set of all built-in tools.

    Returns a new list of tool instances each time it's called.
    """
    return [
        BashTool(),
        GlobTool(),
        GrepTool(),
        ReadTool(),
        StatTool(),
        EditTool(),
        WriteTool(),
        WebFetchTool(),
        TodoWriteTool(),
        PythonTool(),
        AskUserQuestionTool(),
    ]

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
    "PermissionCallback",
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
    # Tmux tool
    "TmuxTool",
    "TmuxInput",
    # Default tools function
    "get_default_tools",
]
