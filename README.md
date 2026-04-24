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

**Multi-Provider** - Works with Claude API, Claude Code OAuth, Gemini API, OpenAI API, or ChatGPT/Codex OAuth.

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

## Chat Bots (Discord + Slack)

Two chat frontends share the same agent core (queue/worker/session/tools):

- **nano-discord-bot** — runs in Discord channels and threads.
- **nano-slack-bot** — runs in Slack channels and threads via Socket Mode.

Both use the same `bot/bot_state.py`, `bot/bot_agent.py`, session persistence under `logs/`, and agent loop. Platform-specific code lives in `bot/discord_bot.py` + `bot/bot_tools.py` and `bot/slack_bot.py` + `bot/slack_tools.py`.

## Discord Bot

**nano-discord-bot** is an AI assistant that runs 24/7 as a Discord bot. It uses the same nano-agent core and tools to provide an agentic coding assistant directly in Discord channels and threads.

### Features

- **Queue-based messaging**: User messages are queued and processed asynchronously, so the bot handles concurrent messages gracefully
- **Channel workers**: Each channel gets a dedicated worker that processes messages in order
- **Discord tools**: Send messages, send files, create threads, explore guild/channel structure, and call the Discord REST API
- **Session persistence**: Message queues are persisted to disk and recovered on restart
- **All built-in tools**: Bash, Read, Write, Edit, Glob, Grep, Stat, Python, WebFetch, and more

### Setup

```bash
# Install with bot extras (discord.py, python-dotenv)
uv sync --extra bot

# Set your Discord bot token
export DISCORD_BOT_TOKEN=your-token-here

# Run the bot (default provider: Claude Code)
uv run nano-discord-bot
```

### Provider selection

The bot supports two LLM providers, selected via `BOT_PROVIDER`:

- **`claude`** (default) — Claude Code OAuth. Run `nano-agent-capture-auth` once to populate `~/.nano-agent.json`.
- **`codex`** — ChatGPT/Codex OAuth. Reads `~/.codex/auth.json` and auto-refreshes the access token (every 8 days). Populate it either with the official `codex login` CLI, or with the built-in Python PKCE flow:

  ```bash
  # One-time login (opens a browser, binds 127.0.0.1:1455)
  uv run python -m nano_agent.providers.codex_login

  # Run the bot against Codex
  BOT_PROVIDER=codex uv run nano-discord-bot

  # Optional: override the model (default: gpt-5.5)
  CODEX_MODEL=gpt-5.5 BOT_PROVIDER=codex uv run nano-discord-bot
  ```

  **Remote servers:** the OAuth flow needs a local browser and port 1455. If the bot runs on a headless host, log in locally first, then copy `~/.codex/auth.json` to the server. Auto-refresh keeps it alive from there; `/renew` only works on a host where the browser + port 1455 are reachable.

### Slash Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history in the current channel/thread |
| `/queue` | Show queued user messages |
| `/cd <path>` | Change working directory |
| `/cwd` | Show current working directory |
| `/thread <topic>` | Start a new conversation in a Discord thread |
| `/renew` | Re-run OAuth login for the active provider |
| `/explore` | Explore visible Discord context (guild/channels/threads) |

## Slack Bot

**nano-slack-bot** is the Slack counterpart. Same queue-based architecture, same `BOT_PROVIDER` switch (Claude Code or Codex), same agent loop — but driven by Slack events via [slack-bolt](https://slack.dev/bolt-python/) Socket Mode.

### Slack app setup

A ready-made manifest is at [`bot/slack_manifest.json`](bot/slack_manifest.json) — it declares every scope, event, and slash command the bot needs. Fastest setup:

1. Go to https://api.slack.com/apps → **Create New App** → **From an app manifest** → pick a workspace.
2. Paste the contents of `bot/slack_manifest.json` into the JSON tab and submit. Review and create.
3. **Basic Information** → **App-Level Tokens** → **Generate** with the `connections:write` scope. This gives you the `xapp-…` token.
4. **Install to Workspace**. Copy the Bot User OAuth Token (`xoxb-…`) from **OAuth & Permissions**.
5. Put both tokens in `.env`:
   - `SLACK_BOT_TOKEN=xoxb-…`
   - `SLACK_APP_TOKEN=xapp-…`

If you want to set things up manually instead, the manifest encodes:
- **Bot scopes**: `app_mentions:read`, `chat:write`, `channels:history`, `channels:read`, `groups:history`, `groups:read`, `im:history`, `im:read`, `im:write`, `mpim:history`, `mpim:read`, `files:write`, `users:read`, `commands`.
- **Event subscriptions**: `app_mention`, `message.channels`, `message.groups`, `message.im`, `message.mpim`.
- **Slash commands**: `/clear`, `/queue`, `/cd`, `/cwd`, `/thread`, `/renew`.
- **Socket Mode**: on (no public HTTP endpoint needed).

### Setup

```bash
# Install with bot extras (also installs discord.py, slack-bolt, slack-sdk)
uv sync --extra bot

# Add tokens to .env
echo "SLACK_BOT_TOKEN=xoxb-…" >> .env
echo "SLACK_APP_TOKEN=xapp-…" >> .env

# Run the bot (default provider: Claude Code)
uv run nano-slack-bot

# Or with Codex OAuth
BOT_PROVIDER=codex uv run nano-slack-bot
```

### Behavior

- **DMs** to the bot: always answered.
- **Channel messages**: the bot responds when **mentioned** (`@nano-bot`), or when a user replies in a thread the bot is already in. Set `SLACK_RESPOND_TO_ALL_MESSAGES=true` to respond to every message in channels the bot is in (not recommended for busy workspaces).
- **Threads**: each thread is its own conversation (keyed by `channel_id:thread_ts`). Channel-level messages reply in-thread by default so the main channel stays tidy.

### Slash Commands

Same surface as Discord — names differ only in platform flavor:

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history in the current channel |
| `/queue` | Show queued user messages |
| `/cd <path>` | Change working directory |
| `/cwd` | Show current working directory |
| `/thread <topic>` | Open a new thread in the current channel and start a conversation |
| `/renew` | Re-run OAuth login for the active provider |

### Slack-specific tools

The agent gets Slack analogues of the Discord tools: `SendUserMessage`, `SendFile` (`files_upload_v2`), `CreateThread`, `ExploreSlack`, `SlackAPI`, plus the shared queue tools. `SlackAPI` can call any Web API method — use `action=discover` to inspect context, then `action=request` with `method` and `body_json`.

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
