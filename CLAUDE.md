# CLAUDE.md

Guidance for Claude Code when working with this repository.

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

# Type checking & formatting
uv run mypy .
uv run black .
uv run isort .
uv run pre-commit run --all-files

# Run examples
uv run python examples/hello_world.py
uv run python examples/simple_tool.py
```

## Project Structure

```
nano_agent/
├── nano_agent/
│   ├── dag.py              # Node and DAG classes
│   ├── data_structures.py  # Message, ContentBlock types
│   ├── claude_api.py       # ClaudeAPI client
│   ├── claude_code_api.py  # Claude Code OAuth client
│   ├── gemini_api.py       # Gemini API client
│   ├── executor.py         # Agent execution loop (run())
│   └── tools/              # Built-in tools package
│       ├── base.py         # Tool base class, schema utilities
│       ├── bash.py         # BashTool
│       ├── read.py         # ReadTool
│       ├── write.py        # WriteTool
│       ├── edit.py         # EditTool, EditConfirmTool
│       ├── glob.py         # GlobTool
│       ├── grep.py         # GrepTool
│       ├── stat.py         # StatTool
│       ├── python.py       # PythonTool
│       ├── todo.py         # TodoWriteTool
│       └── webfetch.py     # WebFetchTool
├── cli/
│   ├── __init__.py         # Entry point and exports
│   ├── app.py              # Main TerminalApp class
│   ├── display.py          # Rich rendering formatters
│   ├── messages.py         # UIMessage, MessageStatus, RenderItem
│   ├── message_list.py     # MessageList container
│   ├── message_factory.py  # Factory functions for UIMessage
│   ├── input_handler.py    # InputHandler protocol
│   ├── input_handlers.py   # TextInput, Confirmation, Selection handlers
│   └── elements/           # Interactive UI components
│       ├── base.py         # ActiveElement protocol, InputEvent
│       ├── manager.py      # ElementManager - coordinates elements
│       ├── terminal.py     # TerminalRegion, RawInputReader
│       ├── confirm_prompt.py   # ConfirmPrompt element
│       ├── menu_select.py      # MenuSelect element
│       ├── text_prompt.py      # TextPrompt element
│       └── prompt_toolkit_input.py  # PromptToolkitInput element
├── examples/
├── tests/
└── scripts/
    ├── viewer.py           # HTML visualization
    └── console_viewer.py   # ASCII visualization
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
| Ctrl+J | Insert new line (for multiline input) |
| Esc | Cancel current operation (during execution) |
| Ctrl+D | Exit |

### Features

- Message-list TUI with Rich rendering
- Auto-saves session to `.nano-cli-session.json`
- Loads `CLAUDE.md` from current directory as context
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
    result = await tool.execute(call.input)
    dag = dag.tool_result(ToolResultContent(tool_use_id=call.id, content=[result]))
```

## Notes

- `ClaudeAPI()` uses `ANTHROPIC_API_KEY` env var
- `ClaudeCodeAPI()` uses Claude Code OAuth (no API key needed)
- `GeminiAPI()` uses `GEMINI_API_KEY` env var
- Tools return `TextContent` or `list[TextContent]`
- `ToolResultContent.tool_name` is for display only - excluded from API calls via `to_dict()`
