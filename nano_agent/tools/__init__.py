"""Built-in tool definitions for Claude API.

This package provides a collection of tools that can be used with the Claude API.
Each tool is in its own module for better organization and maintainability.
"""

from .apply_patch import ApplyPatchInput, ApplyPatchTool
from .ask_user_question import AskUserQuestionInput, AskUserQuestionTool
from .base import (
    _DEFAULT_TRUNCATION_CONFIG,
    Desc,
    Field,
    InputSchemaDict,
    Question,
    QuestionOption,
    SubAgentTool,
    Tool,
    ToolDict,
    ToolResult,
    TruncatedOutput,
    TruncationConfig,
    _save_full_output,
    _truncate_text_content,
    _truncated_outputs,
    cleanup_truncated_outputs,
    clear_all_truncated_outputs,
    convert_input,
    get_call_input_type,
    schema_from_dataclass,
)
from .bash import BashInput, BashTool
from .download_skill import DownloadSkillInput, DownloadSkillTool, Skill
from .edit import (
    EditInput,
    EditTool,
    PermissionCallback,
)
from .exec_command import (
    ExecCommandInput,
    ExecCommandTool,
    WriteStdinInput,
    WriteStdinTool,
    cleanup_exec_command_sessions,
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
from .tmux import TmuxInput, TmuxTool
from .todo import Todo, TodoItemInput, TodoStatus, TodoWriteInput, TodoWriteTool
from .view_image import ViewImageInput, ViewImageTool
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


def get_codex_tools() -> list[Tool]:
    """Codex-style tool set, mirroring the codex-rs default tools.

    Drops the Claude-flavored Bash/Read/Glob/Grep/Edit/Write tools in favor of
    ``exec_command`` + ``write_stdin`` (shell), ``apply_patch`` (file edits),
    and ``view_image`` (image input). Codex-trained models (gpt-5.x) drive
    these via the canonical Codex schema and grammar.
    """
    return [
        ExecCommandTool(),
        WriteStdinTool(),
        ApplyPatchTool(),
        ViewImageTool(),
        StatTool(),
        WebFetchTool(),
        TodoWriteTool(),
        PythonTool(),
        AskUserQuestionTool(),
    ]


__all__ = [
    # Base classes and utilities
    "Tool",
    "SubAgentTool",
    "ToolDict",
    "ToolResult",
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
    "_DEFAULT_TRUNCATION_CONFIG",
    "_save_full_output",
    "_truncate_text_content",
    "_truncated_outputs",
    # Question data classes
    "Question",
    "QuestionOption",
    # AskUserQuestion tool
    "AskUserQuestionTool",
    "AskUserQuestionInput",
    # Codex-style apply_patch tool
    "ApplyPatchTool",
    "ApplyPatchInput",
    # DownloadSkill tool
    "DownloadSkillTool",
    "DownloadSkillInput",
    "Skill",
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
    # Codex-style exec_command tool
    "ExecCommandTool",
    "ExecCommandInput",
    "WriteStdinTool",
    "WriteStdinInput",
    "cleanup_exec_command_sessions",
    # Write tool
    "WriteTool",
    "WriteInput",
    # WebFetch tool
    "WebFetchTool",
    "WebFetchInput",
    # Codex-style view_image tool
    "ViewImageTool",
    "ViewImageInput",
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
    "_python_scripts",
    # Tmux tool
    "TmuxTool",
    "TmuxInput",
    # Default tools function
    "get_default_tools",
    # Codex-style tool set
    "get_codex_tools",
]
