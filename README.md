# nano agent

A minimalistic Python library for building AI agents using functional, immutable DAG operations.

## Features

**Functional & Immutable** - The DAG is immutable. Every operation returns a new instance. No hidden state, no mutations, easy to reason about.

```python
dag = DAG()
dag = dag.system("You are helpful.")  # New DAG
dag = dag.user("Hello")               # New DAG
dag = dag.assistant(response.content) # New DAG
```

**Conversation Graph** - Everything is a node in a directed acyclic graph: system prompts, messages, tool calls, results. Branch and merge for parallel tool execution.

**Built-in Tools** - `BashTool`, `ReadTool`, `WriteTool`, `EditTool`, `GlobTool`, `GrepTool`, `StatTool`, `PythonTool`, `TodoWriteTool`, `WebFetchTool`.

**Sub-Agents** - Tools can spawn their own agents using `SubAgentTool`. Supports recursive nesting and parallel execution.

**Visualization** - Print any DAG to see the conversation flow, or export to HTML:

```
SYSTEM: You are helpful.
    │
    ▼
USER: What files are here?
    │
    ▼
TOOL_USE: Bash
    │
    ▼
TOOL_RESULT: file1.py, file2.py
    │
    ▼
ASSISTANT: I found 2 Python files...
```

```python
dag.save("conversation.json")  # Save the graph
```

```bash
uv run nano-agent-viewer conversation.json  # Creates conversation.html
```

**Multi-Provider** - Works with Claude API, Claude Code OAuth, or Gemini API.

## Quick Start

```python
import asyncio
from nano_agent import ClaudeAPI, DAG, BashTool, run

async def main():
    api = ClaudeAPI()  # Uses ANTHROPIC_API_KEY
    dag = (
        DAG()
        .system("You are a helpful assistant.")
        .tools(BashTool())
        .user("What is the current date?")
    )
    dag = await run(api, dag)
    print(dag)

asyncio.run(main())
```

## Installation

```bash
git clone https://github.com/NTT123/nano-agent.git
cd nano-agent
uv sync
```

## Development

```bash
# Install pre-commit hooks (required for contributing)
uv run pre-commit install

# Run tests
uv run pytest

# Type checking
uv run mypy .

# Format code
uv run pre-commit run --all-files
```

## CLI

**nano-cli** is a lightweight, terminal-based AI coding assistant similar to Claude Code or Cursor. It provides an agentic loop that can read files, execute commands, edit code, and browse the web—all from your terminal.

### Features

- **Agentic execution**: Automatically handles tool calls in a loop until the task is complete
- **Session persistence**: Auto-saves conversations and can resume from where you left off
- **Multi-provider**: Works with Claude (via Claude Code OAuth) or Gemini APIs
- **Rich TUI**: Syntax-highlighted output, streaming responses, and interactive confirmations
- **Project context**: Automatically loads `CLAUDE.md` from your current directory as context
- **Built-in tools**: Bash, Read, Write, Edit, Glob, Grep, Stat, TodoWrite, WebFetch, Python

### Installation

Install the CLI globally using uv:

```bash
uv tool install git+https://github.com/NTT123/nano-agent.git
```

### Authentication

Capture your Claude Code auth credentials first:

```bash
nano-agent-capture-auth
```

### Usage

Once installed, you can use `nano-cli` from any project directory:

```bash
cd your-project
nano-cli
```

Additional options:

```bash
# Run with Gemini instead of Claude
nano-cli --gemini
nano-cli --gemini gemini-2.5-flash  # specific model

# Continue from saved session
nano-cli --continue
nano-cli --continue my-session.json

# Debug mode (show raw response blocks)
nano-cli --debug
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

## Sub-Agents

Create tools that spawn their own agents using `SubAgentTool`:

```python
from dataclasses import dataclass
from typing import Annotated
from nano_agent import SubAgentTool, TextContent, ReadTool
from nano_agent.tools.base import Desc

@dataclass
class CodeReviewInput:
    file_path: Annotated[str, Desc("Path to the file to review")]

@dataclass
class CodeReviewTool(SubAgentTool):
    name: str = "CodeReview"
    description: str = "Spawn a sub-agent to review code"

    async def __call__(self, input: CodeReviewInput) -> TextContent:
        summary = await self.spawn(
            system_prompt="You are an expert code reviewer...",
            user_message=f"Review: {input.file_path}",
            tools=[ReadTool()],
        )
        return TextContent(text=summary)
```

Features:
- **Recursive nesting**: Sub-agents can spawn their own sub-agents (with depth limits)
- **Parallel execution**: Multiple sub-agent tools can run concurrently
- **Graph visualization**: Sub-agent graphs are captured and viewable in HTML export

See `examples/parallel_sub_agents.py` and `examples/recursive_sub_agents.py` for complete examples.

## License

MIT
