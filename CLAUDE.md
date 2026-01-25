# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nano agent** (a minimalistic functional asynchronous agent) is a Python library for building agent conversation systems using the Claude API. It implements a **node-based conversation graph** architecture where everything (system prompts, tool definitions, messages, tool executions, stop reasons) is represented as nodes in a directed acyclic graph (DAG).

**Key Feature:** Configuration is fully automatic - no API keys or `.env` files needed. The library automatically captures all HTTP headers AND request body parameters (model, max_tokens, temperature, user_id) from the Claude CLI, ensuring perfect consistency with the official client.

## Design Philosophy

**Functional & Immutable**: The DAG is immutable - all operations return new DAG instances. No internal state is mutated.

```python
dag = DAG()
dag = dag.system("...")   # Returns NEW DAG
dag = dag.user("...")     # Returns NEW DAG
dag = dag.assistant(...)  # Returns NEW DAG
```

**Separation of Concerns**:
- **Framework manages**: Graph structure, traversal, serialization, visualization
- **You manage separately**: Tool execution (side effects), external environment, mutable state

**Tools are callable**: Tools define their schema for the API, and implement `__call__` for execution. The framework never executes tools - you call them explicitly, keeping side effects outside the DAG.

```python
bash = BashTool()
result = bash(tool_call.input)  # You execute, outside the DAG
dag = dag.tool_result(...)      # Record result in DAG
```

## Installation & Usage

### Installation

```bash
# Install from source (for development)
git clone https://github.com/NTT123/nano-agent.git
cd nano-agent
uv sync
```

**Note:** With `uv`, the package is automatically installed when you run examples with `uv run`. No need to manually install!

### Quick Start

```python
import asyncio
from nano_agent import ClaudeAPI, DAG

async def main():
    # No API key needed - auto-captured from Claude CLI
    api = ClaudeAPI()

    # Build conversation using immutable DAG
    dag = (
        DAG()
        .system("You are a friendly assistant.")
        .user("What is your name?")
    )

    # Send request (async)
    response = await api.send(dag)

    # Add response to DAG (for continuing conversation)
    dag = dag.assistant(response.content)

    # Print the DAG visualization
    print(dag)

asyncio.run(main())
```

## Development Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tracer.py

# Run with verbose output
uv run pytest -v
```

### Code Quality
```bash
# Type checking (strict mode enabled)
uv run mypy .

# Format code
uv run black .
uv run isort .

# Pre-commit hooks (runs black, isort, mypy)
uv run pre-commit run --all-files
```

### Running Examples

```bash
# Hello world example
uv run nano-agent-example-hello-world

# Or run examples directly with Python
uv run python examples/hello_world.py
uv run python examples/simple_tool.py
uv run python examples/parallel_tools.py

# View conversation graph (generates HTML)
uv run nano-agent-viewer graph.json

# View conversation graph in console (ASCII art)
uv run nano-agent-console-viewer graph.json

# Manually capture auth token
uv run nano-agent-capture-auth
```

## Core Architecture

### Node-Based Conversation Graph

The fundamental architecture revolves around the `Node` class ([nano_agent/dag.py](nano_agent/dag.py)), which represents everything in the conversation as nodes in a DAG:

- **System prompts** are nodes
- **Tool definitions** are nodes
- **Messages** (user/assistant) are nodes
- **Tool executions** (for visualization) are nodes
- **Stop reasons** (conversation termination) are nodes

Nodes form a directed acyclic graph where:
- Arrows indicate **forking** (1→N) when multiple tool calls execute in parallel
- Arrows indicate **merging** (N→1) when combining tool results back into the conversation

### Key Concepts

**1. Node Creation Pattern**
```python
# Start with system prompt (Claude Code identity is always prepended)
node = Node.system("Custom instructions")

# Add tool definitions
node = node.tools(BashTool(), ReadTool())

# Add conversation messages
node = node.child(Message(Role.USER, "Hello"))
node = node.child(Message(Role.ASSISTANT, "Hi!"))
```

**2. Graph Traversal**
- `node.ancestors()` returns all ancestors in causal order (parents before children)
- `node.to_messages()` extracts only Message nodes for API calls
- `node.get_system_prompt()` concatenates all system prompts
- `node.get_tools()` extracts tool definitions

**3. Tool Call Branching and Merging**

When Claude makes multiple tool calls in parallel:
1. Create tool_use_node as child with tool calls
2. Execute each tool, creating separate **branch nodes** (for visualization)
3. **Merge** all branches into a single node with combined tool results

```python
# Branch: one node per tool execution (visual)
result_nodes = []
for tool_call in tool_calls:
    result_node = tool_use_node.child(ToolExecution(...))
    result_nodes.append(result_node)

# Merge: combine all results into one message (what API sees)
node = Node.with_parents(result_nodes, Message(Role.USER, result_contents))
```

### Project Structure

```
nano_agent/
├── nano_agent/                     # Core library package
│   ├── __init__.py                 # Public API exports
│   ├── api_base.py                 # API client base classes and protocols
│   ├── cancellation.py             # Cancellation token support
│   ├── capture_claude_code_auth.py # Configuration capture from Claude CLI
│   ├── claude_api.py               # ClaudeAPI client - API communication
│   ├── claude_code_api.py          # ClaudeCodeAPI client - eager auth capture
│   ├── dag.py                      # Node and DAG classes - graph construction
│   ├── data_structures.py          # All dataclasses (Message, ContentBlock, etc.)
│   ├── executor.py                 # Agent execution loop
│   ├── gemini_api.py               # Gemini API client
│   ├── openai_api.py               # OpenAI API client
│   └── tools.py                    # Tool definitions + TodoWriteTool
├── examples/                       # Example scripts
│   ├── hello_world.py              # Simple hello world example
│   ├── simple_tool.py              # Tool usage demonstration
│   └── parallel_tools.py           # Parallel tool calls demonstration
├── tests/                          # Test suite
│   ├── test_dag.py                 # DAG builder tests
│   ├── test_tracer.py              # Node graph tests
│   ├── test_tools.py               # Tool definition tests
│   └── test_claude_api.py          # API client tests
├── scripts/                        # Utility scripts
│   ├── capture_claude_code_auth.py # Auth capture entry point
│   ├── viewer.py                   # HTML visualization generator
│   └── console_viewer.py           # Console ASCII visualization
├── docs/                           # Additional documentation
├── pyproject.toml                  # Package configuration
├── CLAUDE.md                       # This file (developer guide)
└── README.md                       # User documentation
```

### Module Organization

**Core Package (`nano_agent/`):**
- **[data_structures.py](nano_agent/data_structures.py)**: All dataclasses - Message, ContentBlock types (TextContent, ThinkingContent, ToolUseContent, ToolResultContent), NodeData types (SystemPrompt, ToolDefinitions, ToolExecution, StopReason), Usage
- **[dag.py](nano_agent/dag.py)**: Node and DAG classes - graph construction, traversal, serialization
- **[claude_api.py](nano_agent/claude_api.py)**: ClaudeAPI client - sends requests, auto-captures auth token if not provided, always prepends Claude Code identity to system prompts
- **[capture_claude_code_auth.py](nano_agent/capture_claude_code_auth.py)**: Configuration capture - intercepts Claude CLI requests to extract all HTTP headers AND request body parameters (auth tokens, client identity headers, model, max_tokens, temperature, user_id, URL path)
- **[claude_code_api.py](nano_agent/claude_code_api.py)**: ClaudeCodeAPI client - captures auth from Claude CLI at initialization (eager capture)
- **[tools.py](nano_agent/tools.py)**: Tool definitions - all built-in tools (Bash, Read, Edit, TodoWrite, etc.) with schemas. `TodoWriteTool` includes integrated state management (Todo, TodoStatus, display(), get_current_task(), clear())

**Utility Scripts (`scripts/`):**
- **[viewer.py](scripts/viewer.py)**: HTML visualization generator - creates interactive D3.js graph from saved JSON, includes stop reason visualization
- **[console_viewer.py](scripts/console_viewer.py)**: Console visualization - renders ASCII graph with colors and parallel execution patterns

### API Integration

The `ClaudeAPI` class ([nano_agent/claude_api.py](nano_agent/claude_api.py)) handles communication:
- **Automatic configuration capture**: If no `api_key` or `headers` provided, automatically captures all HTTP headers AND request body parameters from Claude CLI via `nano_agent/capture_claude_code_auth.py`
- Captures and uses: model, max_tokens, temperature, thinking budget, **user_id** (important for analytics!), **URL path** (ensures correct API endpoint)
- Always includes "You are Claude Code, Anthropic's official CLI for Claude" as the first system message (non-negotiable)
- Uses prompt caching for the Claude Code identity
- Supports extended thinking with captured `thinking.budget_tokens`
- Returns typed `Response` objects with structured content blocks

**Usage:**
```python
# Recommended: auto-capture all config (headers + body params) from Claude CLI
api = ClaudeAPI()  # Uses captured model, max_tokens, user_id, temperature, etc.

# Override specific parameters while keeping captured defaults
api = ClaudeAPI(model="claude-opus-4-5-20251101")  # Custom model, but captured user_id, etc.

# Backwards compatible: provide explicit API key
api = ClaudeAPI(api_key="sk-ant-...")

# Advanced: provide explicit headers
api = ClaudeAPI(headers={"authorization": "Bearer sk-ant-...", ...})
```

### ClaudeCodeAPI (Eager Auto-Auth)

The `ClaudeCodeAPI` class ([nano_agent/claude_code_api.py](nano_agent/claude_code_api.py)) provides automatic authentication by capturing credentials from Claude CLI at initialization:

```python
from nano_agent import ClaudeCodeAPI, DAG

# Captures auth immediately at init (eager) - fails fast if CLI unavailable
api = ClaudeCodeAPI()

# Use like ClaudeAPI
dag = DAG().system("Be helpful.").user("Hello!")
response = await api.send(dag)

# Override specific parameters while keeping captured defaults
api = ClaudeCodeAPI(model="claude-opus-4-5-20251101", max_tokens=4096)

# Async context manager for proper cleanup
async with ClaudeCodeAPI() as api:
    response = await api.send(dag)
```

**Key differences from ClaudeAPI:**
- **Eager capture**: Authentication is captured at `__init__`, not lazily on first request
- **Fail-fast**: Raises `RuntimeError` immediately if Claude CLI is not available
- **No manual auth options**: Always captures from CLI (use `ClaudeAPI` for explicit API keys)

### Graph Serialization

Graphs can be saved and loaded:
```python
# Save entire graph from head node
Node.save_graph(node, "graph.json")

# Load graph back
heads, metadata = Node.load_graph("graph.json")

# Generate HTML visualization
uv run python viewer.py graph.json  # Creates graph.html
```

## Important Implementation Details

### Configuration Capture System

The library supports three modes of operation:

**Mode 1: Auto-Capture (Default, Recommended)**
```python
api = ClaudeAPI()  # Automatically captures ALL headers AND body params from Claude CLI
```
- Captures all HTTP headers from a real Claude CLI request
- **NEW:** Also captures key request body parameters (model, max_tokens, temperature, user_id)
- Ensures perfect consistency with official Claude Code configuration
- Captured values are used as defaults but can be overridden
- Headers excluded: host, content-length, content-type, accept, connection, transfer-encoding

**Mode 2: Explicit API Key (Backwards Compatible)**
```python
api = ClaudeAPI(api_key="sk-ant-...")
```
- Uses hardcoded headers and parameters with provided API key
- Maintained for backwards compatibility
- May drift from official Claude CLI configuration over time

**Mode 3: Explicit Headers (Advanced)**
```python
headers = {"authorization": "Bearer sk-ant-...", "anthropic-version": "2023-06-01", ...}
api = ClaudeAPI(headers=headers)
```
- Full control over all HTTP headers
- Useful for testing or custom configurations

**Captured Headers Include:**
- `authorization` - OAuth token
- `anthropic-version`, `anthropic-beta`, `anthropic-dangerous-direct-browser-access`
- `user-agent`, `x-app`
- `x-stainless-*` - Platform info (arch, os, lang, runtime, package version, etc.)

**Captured Body Parameters Include:**
- `model` - Claude model being used (e.g., "claude-opus-4-5-20251101")
- `max_tokens` - Maximum token limit
- `temperature` - Temperature setting
- `thinking.budget_tokens` - Thinking budget for extended thinking
- **`metadata.user_id`** - User ID for tracking/analytics (very important!)
- **`system`** - Full system prompt array with cache_control (Claude Agent SDK identity + instructions)
- **`url_path`** - API endpoint path with query string (e.g., "/v1/messages?beta=true")

**How Auto-Capture Works:**
1. Starts local HTTP server on port 1235
2. Launches `claude` CLI subprocess with `ANTHROPIC_BASE_URL=http://localhost:1235/`
3. Intercepts multiple API requests from Claude CLI
4. Skips quota check requests and selects the request with cached system prompt (full config)
5. Extracts all headers, URL path, and body parameters from the full config request
6. Returns configuration for reuse in your API calls

**Parameter Override:**
```python
# Use captured defaults but override specific parameters
api = ClaudeAPI(model="claude-opus-4-5-20251101")  # Custom model, captured user_id/etc.
api = ClaudeAPI(temperature=0.5)  # Custom temperature, captured everything else
api = ClaudeAPI(base_url="https://custom-api.example.com/v1/messages")  # Custom URL
```

### Parallel Tool Execution
The library supports parallel tool calls (multiple tools in one API response). The graph structure captures this:
- One assistant message node with multiple ToolUseContent blocks
- Multiple execution branch nodes (one per tool)
- One merged user message node with all ToolResultContent blocks

This creates a visual "fan-out, fan-in" pattern in the graph while maintaining the correct API message sequence.

### Stop Reason Visualization
When a conversation ends, a special `StopReason` node is added to the graph:
- Shows why the conversation ended (e.g., "end_turn", "max_tokens")
- Displays cumulative token usage for the entire conversation
- Visualized as a red node at the end of the graph
- Helps debug unexpected conversation terminations

### Type Safety
The codebase uses strict mypy typing:
- All data structures are dataclasses with full type annotations
- String enums for Role, NodeType, TodoStatus
- TypedDict for API response shapes
- Union types for content blocks (ContentBlock = TextContent | ThinkingContent | ToolUseContent | ToolResultContent)

### Critical Bug Fix
**ToolResultContent.tool_name**: The `tool_name` field is for display/visualization only and must NOT be sent to the API. The API rejects requests with extra fields in `tool_result` blocks. Always use `to_dict()` which excludes this field when serializing for API calls.
