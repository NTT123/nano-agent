# Quickstart Guide

Get started with nano-agent in under 5 minutes. This guide walks you through installation, your first conversation, and your first tool.

## Prerequisites

- **Python 3.10+**
- **uv** (Python package manager) - [Install uv](https://github.com/astral-sh/uv)
- **ANTHROPIC_API_KEY** environment variable set

## Installation

```bash
# Clone the repository
git clone https://github.com/NTT123/nano-agent.git
cd nano-agent

# Install dependencies
uv sync
```

With `uv`, examples run automatically without manual installation:

```bash
uv run python examples/hello_world.py
```

## Your First Conversation

Create a simple conversation with Claude:

```python
import asyncio
from nano_agent import DAG, ClaudeAPI

async def main():
    # Build conversation using the immutable DAG
    dag = (
        DAG()
        .system("You are a friendly assistant.")
        .user("Hello! What is your name?")
    )

    # Send request (uses ANTHROPIC_API_KEY env variable)
    api = ClaudeAPI()
    response = await api.send(dag)

    # Add response to DAG (returns new DAG)
    dag = dag.assistant(response.content)

    # Print the conversation visualization
    print(dag)

asyncio.run(main())
```

**Output:**

```
  SYSTEM: You are a friendly assistant.
      |
      v
  USER: Hello! What is your name?
      |
      v
  ASSISTANT: Hello! I'm Claude, an AI assistant...
```

## Understanding the DAG

The DAG (Directed Acyclic Graph) is nano-agent's core abstraction:

1. **Immutable**: Every operation returns a new DAG. The original is never modified.
2. **Traceable**: The full conversation history is preserved and visualizable.
3. **API-agnostic**: The same DAG works with Claude, OpenAI, or Gemini.

```python
dag1 = DAG().system("Be helpful.")
dag2 = dag1.user("Hi")      # dag1 unchanged!
dag3 = dag2.assistant("Hello!")  # dag1 and dag2 unchanged!
```

## Your First Tool

Tools let Claude interact with the real world. Here's a simple calculator:

```python
import asyncio
from dataclasses import dataclass
from nano_agent import DAG, ClaudeAPI, Tool, TextContent, ToolResultContent

# Define input schema using a dataclass
@dataclass
class CalculatorInput:
    expr: str  # Math expression to evaluate

# Define the tool
@dataclass
class Calculator(Tool):
    name: str = "calculator"
    description: str = "Evaluate a math expression"

    async def __call__(self, input: CalculatorInput) -> TextContent:
        result = eval(input.expr)
        return TextContent(text=str(result))

async def main():
    calc = Calculator()
    api = ClaudeAPI()

    # Build DAG with tool
    dag = (
        DAG()
        .tools(calc)  # Register the tool
        .user("What is 23 * 47?")
    )

    # First response: Claude requests tool use
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    # Execute tool calls and add results
    for call in response.get_tool_use():
        result = await calc.execute(call.input)
        content = result if isinstance(result, list) else [result]
        dag = dag.tool_result(
            ToolResultContent(tool_use_id=call.id, content=content)
        )

    # Second response: Claude provides final answer
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    print(dag)

asyncio.run(main())
```

**Output:**

```
  TOOLS: calculator
      |
      v
  USER: What is 23 * 47?
      |
      v
  TOOL_USE: calculator
      |
      v
  RESULT: 1081
      |
      v
  ASSISTANT: 23 × 47 = 1081
```

## Using Built-in Tools

nano-agent provides several ready-to-use tools:

```python
from nano_agent import BashTool, ReadTool, GlobTool

dag = (
    DAG()
    .tools(BashTool(), ReadTool(), GlobTool())
    .user("List Python files in the current directory")
)
```

Available built-in tools:
- **BashTool**: Execute shell commands
- **ReadTool**: Read files from filesystem
- **WriteTool**: Write files to filesystem
- **EditTool**: Perform string replacements in files
- **GlobTool**: Find files by pattern (uses `fd`)
- **SearchTool**: Search file contents (uses `ripgrep`)
- **PythonTool**: Create and run Python scripts with dependencies
- **WebFetchTool**: Fetch and render web content
- **StatTool**: Get file metadata
- **TodoWriteTool**: Manage task lists

## The run() Helper

For automatic tool execution, use the `run()` helper:

```python
from nano_agent import DAG, ClaudeAPI, run, BashTool

async def main():
    api = ClaudeAPI()
    dag = (
        DAG()
        .tools(BashTool())
        .user("What's the current date?")
    )

    # run() handles the tool execution loop automatically
    dag = await run(api, dag)
    print(dag)

asyncio.run(main())
```

## Multi-API Support

The same DAG works with different providers:

```python
from nano_agent import DAG, ClaudeAPI, OpenAIAPI, GeminiAPI

dag = DAG().system("Be helpful.").user("Hello!")

# Pick your provider
api = ClaudeAPI()       # Uses ANTHROPIC_API_KEY
# api = OpenAIAPI()     # Uses OPENAI_API_KEY
# api = GeminiAPI()     # Uses GEMINI_API_KEY

response = await api.send(dag)
```

## Authentication Options

nano-agent supports several authentication methods:

### 1. Environment Variable (Recommended)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
api = ClaudeAPI()  # Uses ANTHROPIC_API_KEY
```

### 2. Explicit API Key

```python
api = ClaudeAPI(api_key="sk-ant-...")
```

### 3. Config File (ClaudeCodeAPI)

```python
from nano_agent import ClaudeCodeAPI

api = ClaudeCodeAPI()  # Uses ~/.nano-agent.json
```

See [Configuration](configuration.md) for detailed auth setup.

## Next Steps

- **[Custom Tools Tutorial](custom-tools-tutorial.md)**: Build your own tools step-by-step
- **[Configuration](configuration.md)**: Auth setup and environment configuration
- **[Architecture](architecture.md)**: Understand the DAG/Node design
- **[Tools](tools.md)**: Complete tool reference
- **[API Clients](api-clients.md)**: Multi-provider support

## Running Examples

```bash
# Hello world
uv run python examples/hello_world.py

# Simple tool usage
uv run python examples/simple_tool.py

# Parallel tool execution
uv run python examples/parallel_tools.py

# Agent loop with executor
uv run python examples/simple_executor.py
```

---

**Next:** [Custom Tools Tutorial](custom-tools-tutorial.md) - Build your own tools
