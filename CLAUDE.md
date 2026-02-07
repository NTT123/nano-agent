# Guidance for working with this repository.

## Project Overview

**nano agent** - A minimalistic Python library for building AI agents using functional, immutable DAG operations.

- **Immutable DAG**: Every operation returns a new DAG instance
- **Node-based**: System prompts, messages, tool calls, results are all nodes
- **Tools are callable**: Tools define schema for API and implement `__call__` for execution

## Development Commands

```bash
# Testing
uv run pytest
uv run pytest tests/test_dag.py -v

# End-to-end tests (makes real API calls, not part of pre-commit)
uv run python e2e/run_all.py
uv run python e2e/test_executor_cancellation.py

# Type checking & formatting
uv run mypy .
uv run black .
uv run isort --profile black .
uv run pre-commit run --all-files

# Run examples
uv run python examples/hello_world.py
uv run python examples/simple_tool.py
```

## CLI Application

Interactive terminal UI for chatting with Claude/Gemini with full tool support.

**nano-cli** is a lightweight, terminal-based AI coding assistant similar to Claude Code or Cursor. It provides an agentic loop that can read files, execute commands, edit code, and browse the web—all from your terminal. The CLI automatically loads `CLAUDE.md` from your current directory as context, making it ideal for project-specific assistance.

Key capabilities:
- **Agentic execution**: Automatically handles tool calls in a loop until the task is complete
- **Session persistence**: Auto-saves conversations and can resume from where you left off
- **Multi-provider**: Works with Claude (via Claude Code OAuth) or Gemini APIs
- **Rich TUI**: Syntax-highlighted output, streaming responses, and interactive confirmations

### Architecture

Message-list based TUI where each message owns its output buffer:
- **WelcomeMessage** - Greeting and context info (frozen)
- **UserMessage** - User input (frozen)
- **AssistantMessage** - AI response with thinking, text, and token count (frozen)
- **ToolCallMessage** - Tool invocation details (frozen)
- **ToolResultMessage** - Tool execution output (frozen)
- **ActiveMessage** - Current input/streaming (has exclusive input control)

Re-rendering is deterministic: clear screen → render all messages in order.

#### Elements System

The `cli/elements/` module provides an abstraction for interactive terminal elements:

- **ActiveElement**: Protocol for elements with exclusive I/O control
- **ElementManager**: Coordinates active elements, ensures only one is active at a time
- **TerminalRegion**: Controls a region at the bottom of the terminal for element output
- **RawInputReader**: Reads single keystrokes in raw mode

Built-in elements:
- **ConfirmPrompt**: Yes/No confirmation with preview
- **MenuSelect**: Arrow-key menu selection
- **TextPrompt**: Simple text input
- **PromptToolkitInput**: Rich text input with history and editing

### Usage

```bash
# Run with Claude Code API (default)
uv run nano-cli

# Run with Gemini
uv run nano-cli --gemini
uv run nano-cli --gemini gemini-2.5-flash  # specific model

# Continue from saved session
uv run nano-cli --continue
uv run nano-cli --continue my-session.json

# Refresh OAuth token (for 401 errors)
uv run nano-cli --renew

# Debug mode (show raw response blocks)
uv run nano-cli --debug
```

### Commands

| Command | Description |
|---------|-------------|
| `/quit`, `/exit`, `/q` | Exit the application |
| `/clear` | Reset conversation and clear screen |
| `/continue`, `/c` | Continue agent execution without user message |
| `/save [filename]` | Save session to file (default: session.json) |
| `/load [filename]` | Load session from file |
| `/renew` | Refresh OAuth token (for 401 errors) |
| `/render` | Re-render history (after terminal resize) |
| `/debug` | Show DAG as JSON |
| `/help` | Show help message |

### Input Controls

| Key | Action |
|-----|--------|
| Enter | Send message |
| \\ + Enter | Insert new line (for multiline input) |
| Esc | Cancel current operation (during execution) |
| Ctrl+D | Exit |

Note: Ctrl+J and Shift+Enter are not supported.

### Features

- Message-list TUI with Rich rendering
- Auto-saves session to `.nano-cli-session.json`
- Built-in tools: Bash, Read, Write, Edit, Glob, Grep, Stat, TodoWrite, WebFetch, Python
- Edit tool prompts for user confirmation before applying changes

## Key Patterns

```python
# Immutable DAG operations
dag = DAG()
dag = dag.system("...")
dag = dag.tools(BashTool())
dag = dag.user("...")
dag = dag.assistant(response.content)

# Tool execution loop
dag = await run(api, dag)  # Handles tool calls automatically

# Manual tool handling
for call in response.get_tool_use():
    tool_result = await tool.execute(call.input)
    content = tool_result.content
    content_list = content if isinstance(content, list) else [content]
    dag = dag.tool_result(ToolResultContent(tool_use_id=call.id, content=content_list))
```

## Sub-Agents

Tools can spawn sub-agents using the `SubAgentTool` base class (pure functional):

```python
from nano_agent import SubAgentTool, ExecutionContext, TextContent, ReadTool
from nano_agent.tools.base import Desc, ToolResult

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
```

The `SubAgentTool` base class provides:
- Pure functional design (no mutation)
- `spawn()` helper that returns `(summary, sub_graph)` tuple
- `ToolResult` dataclass for returning content and sub_graph together

Sub-agents support recursive nesting (depth tracking) and parallel execution.

## Notes

- `ClaudeAPI()` uses `ANTHROPIC_API_KEY` env var
- `ClaudeCodeAPI()` uses Claude Code OAuth (no API key needed)
- `GeminiAPI()` uses `GEMINI_API_KEY` env var
- `Tool.execute()` returns `ToolResult` containing `content` and optional `sub_graph`
- `ToolResultContent.tool_name` is for display only - excluded from API calls via `to_dict()`
