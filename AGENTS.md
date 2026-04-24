# Guidance for working with this repository

## Project Overview

**nano_agent** is a Python 3.12 library for building AI agents around a
functional, immutable conversation DAG.

- **Immutable graph model**: `DAG` and `Node` operations return new instances.
- **Node-based history**: system prompts, tool definitions, messages, tool
  executions, sub-graphs, and stop reasons are graph nodes.
- **Typed tools**: tools are dataclasses with typed `__call__` input dataclasses;
  schemas are inferred from annotations and `Desc`/`Field` metadata.
- **Provider-neutral loop**: providers implement `APIProtocol`; `executor.run`
  sends the DAG, executes tool calls, merges tool-result branches, and appends a
  stop reason.
- **Applications**: `cli/` implements the `nano-cli` terminal assistant, and
  `bot/` implements Discord and Slack bot frontends.

## Repository Layout

- `nano_agent/`: core library.
  - `dag.py`: immutable `Node` and `DAG` graph primitives, serialization,
    sub-graph conversion, console rendering.
  - `data_structures.py`: content blocks, messages, responses, usage, graph node
    payload types, and parsing helpers.
  - `executor.py`: generic agent loop with transient retry, tool execution,
    cancellation handling, usage accounting, and stop-reason nodes.
  - `execution_context.py`: sub-agent context and `run_sub_agent`.
  - `tools/`: built-in tools and `Tool`/`SubAgentTool` base classes.
  - `providers/`: Claude, Claude Code OAuth, Gemini, OpenAI, Fireworks, and
    Codex/ChatGPT OAuth clients plus auth helpers and cost tracking.
- `cli/`: message-list terminal UI, command router, footer input system, session
  persistence, and DAG-to-UI mapping.
- `bot/`: Discord and Slack frontends, shared queue/state management, shared
  channel worker, and platform-specific tools.
- `tests/`: unit tests for core, providers, tools, CLI UI pieces, and bot logic.
- `e2e/`: real-provider end-to-end tests; do not run as part of ordinary checks.
- `examples/`: runnable examples and generated sub-agent graph artifacts.
- `scripts/`: graph viewers and rendering helpers.

## Development Commands

```bash
# Install/update dependencies
uv sync
uv sync --extra bot
uv sync --extra tmux

# Unit tests
uv run pytest
uv run pytest --ignore=e2e
uv run pytest tests/test_dag.py -v
uv run pytest bot/test_bot.py -v

# End-to-end tests; these make real API calls
uv run python e2e/run_all.py
uv run python e2e/test_executor_cancellation.py

# Type checking and formatting
uv run mypy .
uv run black .
uv run isort --profile black .
uv run pre-commit run --all-files

# Examples
uv run python examples/hello_world.py
uv run python examples/simple_tool.py
uv run python examples/sub_agent.py
```

Pre-commit runs Black, isort, mypy with `tests/` and `e2e/` excluded, and
`uv run pytest --ignore=e2e`.

## Core Patterns

```python
from nano_agent import DAG, BashTool, ClaudeAPI, run

dag = (
    DAG()
    .system("You are helpful.")
    .tools(BashTool())
    .user("What is the current date?")
)
dag = await run(ClaudeAPI(), dag)
```

Important graph rules:

- Do not mutate `DAG`, `Node`, or content-block instances in place.
- Use fluent DAG methods (`system`, `tools`, `user`, `assistant`, `tool_result`,
  `sub_graph`) or internal helpers only where existing code already does so.
- `DAG._tools` stores live tool instances for execution; serialized graph data
  stores tool definitions for provider calls and visualization.
- Tool calls execute from one assistant head, create one branch per result, and
  merge through a synthetic user message containing all `ToolResultContent`
  blocks.
- Cancellation must leave every started or pending tool call with a matching
  tool result so provider conversations remain valid.

## Tool Patterns

Tools live under `nano_agent/tools/` and usually follow this shape:

```python
from dataclasses import dataclass
from typing import Annotated

from nano_agent import TextContent
from nano_agent.tools.base import Desc, Tool

@dataclass
class CalculatorInput:
    expr: Annotated[str, Desc("Expression to evaluate")]

@dataclass
class CalculatorTool(Tool):
    name: str = "Calculator"
    description: str = "Evaluate a small expression"

    async def __call__(self, input: CalculatorInput) -> TextContent:
        return TextContent(text=str(eval(input.expr)))
```

- Prefer `Annotated[T, Desc("...")]` for schema descriptions.
- Call tools through `tool.execute(...)` in framework code so raw JSON inputs are
  converted, output truncation is applied, and sub-agent context is passed.
- Return `TextContent`, `list[TextContent]`, or `ToolResult`.
- Configure large-output behavior with `_truncation_config`.
- Add external command checks with `_required_commands` when a tool depends on a
  CLI binary.
- `TmuxTool` is available but is not in `get_default_tools()`; it requires the
  `tmux` extra and the `tmux` binary.

## Sub-Agents

Use `SubAgentTool` when a tool needs to run another agent and preserve the child
graph:

```python
from dataclasses import dataclass
from typing import Annotated

from nano_agent import ExecutionContext, ReadTool, SubAgentTool, TextContent
from nano_agent.tools.base import Desc, ToolResult

@dataclass
class AuditInput:
    file_path: Annotated[str, Desc("Path to audit")]

@dataclass
class AuditTool(SubAgentTool):
    name: str = "Audit"
    description: str = "Spawn a sub-agent to audit a file"

    async def __call__(
        self,
        input: AuditInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        if execution_context is None:
            return ToolResult(content=TextContent(text="Error: no context"))

        summary, sub_graph = await self.spawn(
            context=execution_context,
            system_prompt="You are a focused code auditor.",
            user_message=f"Audit {input.file_path}",
            tools=[ReadTool()],
        )
        return ToolResult(content=TextContent(text=summary), sub_graph=sub_graph)
```

`ExecutionContext` carries the provider, current DAG, cancellation token,
permission callback, and depth limit. `run_sub_agent` and `SubAgentTool.spawn`
return a `SubGraph` that executor code stores before the tool result node.

## Providers and Auth

- `ClaudeAPI()` uses `ANTHROPIC_API_KEY`.
- `ClaudeCodeAPI()` uses captured Claude Code OAuth config in
  `~/.nano-agent.json`; refresh/capture with `uv run nano-agent-capture-auth` or
  `uv run nano-cli --renew`.
- `GeminiAPI()` uses `GEMINI_API_KEY`.
- `OpenAIAPI()` uses `OPENAI_API_KEY` and the Responses API.
- `FireworksAPI()` uses `FIREWORKS_API_KEY`.
- `CodexAPI()` uses ChatGPT/Codex OAuth. It defaults to
  `~/.codex/auth.json`, can refresh file-backed tokens, and can be initialized
  by `codex login` or `uv run python -m nano_agent.providers.codex_login`.

Provider tests mostly use mocked HTTP clients. E2E tests use real credentials.

## CLI Application

`nano-cli` is the terminal assistant in `cli/`. It uses a deterministic
message-list TUI:

- `TerminalApp`: application state, provider setup, agent loop, cancellation,
  auto-save, cost/tokens, and slash command dispatch.
- `MessageList` and `UIMessage`: append-only render model; adding a message
  freezes the previous one.
- `DAGMessageMapper`: rebuilds UI history from a saved DAG.
- `InputController` and `cli/elements/`: footer input, status bar, confirmation
  prompts, menu selection, cancellation menu, and raw-key handling.
- `SessionStore`: auto-saves to `.nano-cli-session.json`; `/save` and `/load`
  operate on JSON graph files.

Current CLI provider flags:

```bash
# Default: Claude Code OAuth
uv run nano-cli

# Codex/ChatGPT OAuth
uv run nano-cli --codex
uv run nano-cli --codex gpt-5.2-codex

# Gemini
uv run nano-cli --gemini
uv run nano-cli --gemini gemini-3-pro-preview

# Fireworks
uv run nano-cli --fireworks
uv run nano-cli --fireworks accounts/fireworks/models/kimi-k2p5

# Other useful flags
uv run nano-cli --thinking-level low
uv run nano-cli --thinking-budget 16000
uv run nano-cli --continue
uv run nano-cli --continue my-session.json
uv run nano-cli --renew
uv run nano-cli --debug
```

The CLI currently loads `CLAUDE.md` from the working directory into the system
prompt. It does not currently load `AGENTS.md`.

Slash commands:

| Command | Description |
| --- | --- |
| `/quit`, `/exit`, `/q` | Exit |
| `/clear` | Reset conversation and clear screen |
| `/continue`, `/c` | Continue agent execution without adding a user message |
| `/renew` | Refresh Claude Code OAuth token |
| `/render` | Re-render history from the DAG |
| `/theme dark\|light\|auto` | Set UI color scheme |
| `/theme status` | Show current color scheme detection info |
| `/debug` | Show DAG as JSON |
| `/save [filename]` | Save session JSON |
| `/load [filename]` | Load session JSON |
| `/help` | Show help |

Input controls:

| Key | Action |
| --- | --- |
| Enter | Submit |
| `\` + Enter | Insert newline |
| Shift+Enter | Insert newline when the terminal reports it |
| Shift+Tab | Toggle auto-accept for edit confirmations |
| Esc | Cancel input or current operation |
| Ctrl+D | Exit |

CLI tool set: Bash, Read, Write, Edit with confirmation callback, Glob, Grep,
TodoWrite, WebFetch, Python, and AskUserQuestion.

## Discord Bot

The Discord bot is in `bot/` and runs with:

```bash
uv sync --extra bot
DISCORD_BOT_TOKEN=... uv run nano-discord-bot
```

Provider selection:

```bash
# Default: Claude Code OAuth
uv run nano-discord-bot

# Codex/ChatGPT OAuth
BOT_PROVIDER=codex CODEX_MODEL=gpt-5.5 uv run nano-discord-bot
```

Important bot architecture:

- `discord_bot.py`: Discord events, slash commands, provider setup, and startup.
- `bot_agent.py`: platform-agnostic channel worker and agent loop.
- `bot_state.py`: sessions, working directories, queue persistence, internal
  JSONL logs, and DAG sanitization.
- `bot_tools.py`: Discord-specific tools and queue tools.
- State is persisted under `logs/discord_agent_state/channel_<id>/`.
- One worker runs per channel/thread; queued user messages are persisted and can
  be recovered on restart.
- Assistant text is internal-only. The agent must call `SendUserMessage` for the
  user to see text in Discord.
- Incoming messages are queued. The agent uses `PeekQueuedUserMessages` and
  `DequeueUserMessages` to decide when to consume them.
- `get_discord_tools()` uses default tools except `AskUserQuestion`, then adds
  SendUserMessage, queue tools, SendFile, CreateThread, ExploreDiscord,
  DiscordAPI, ClearContext, and RestartBot.

Discord slash commands:

| Command | Description |
| --- | --- |
| `/clear` | Clear channel/thread conversation and queue |
| `/queue [limit]` | Show queued user messages |
| `/cd <path>` | Set working directory and reset conversation |
| `/cwd` | Show current working directory |
| `/thread <topic>` | Create a new public thread conversation |
| `/renew` | Refresh OAuth for the active provider |
| `/explore` | Show visible Discord guild/channel/thread context |

## Slack Bot

The Slack bot is in `bot/slack_bot.py` and shares the same `BotState`,
`ensure_channel_worker`, queue model, and agent loop as the Discord bot. It runs
via Slack Bolt Socket Mode:

```bash
uv sync --extra bot
SLACK_BOT_TOKEN=... SLACK_APP_TOKEN=... uv run nano-slack-bot
```

Provider selection matches the Discord bot:

```bash
# Default: Claude Code OAuth
uv run nano-slack-bot

# Codex/ChatGPT OAuth
BOT_PROVIDER=codex CODEX_MODEL=gpt-5.5 uv run nano-slack-bot
```

Important Slack details:

- `slack_bot.py`: Socket Mode app, events, slash commands, provider setup, and
  startup.
- `slack_tools.py`: Slack-specific SendUserMessage, SendFile, CreateThread,
  ExploreSlack, SlackAPI, and RestartBot tools.
- `SlackContext` carries `(channel_id, thread_ts)` because Slack threads are
  reply chains keyed by parent message timestamp.
- Queue/session keys are `channel_id` for non-threaded contexts or
  `channel_id:thread_ts` for threaded contexts.
- Assistant text is internal-only. The agent must call `SendUserMessage` for the
  user to see text in Slack.
- Slack messages are split around 3500 characters; Slack uses mrkdwn, not
  Discord markdown.
- Slash commands declared by the app are `/clear`, `/queue`, `/cd`, `/cwd`,
  `/thread`, and `/renew`.
- Required app scopes are documented at the top of `bot/slack_bot.py`; keep that
  docstring in sync with tool and slash-command changes.

## Testing Notes

- Use focused tests for touched areas first, then broader `uv run pytest` when
  changes touch shared core behavior.
- Provider send-method tests are mocked; do not assume they cover real network
  auth or streaming behavior.
- `e2e/` tests create real API calls and should be run only when explicitly
  needed.
- CLI UI behavior has targeted tests for message rendering, footer input,
  seamless transitions, ANSI utilities, and thread safety.
- Bot tests are tiered around pure helpers, state/queue persistence, tool
  behavior, and agent loop behavior with fakes. Shared bot changes should be
  checked against both Discord and Slack assumptions, even when only Discord has
  direct tests.

## Contribution Notes for Agents

- Keep changes scoped; this repo may have dirty user work.
- Prefer existing DAG, tool, provider, and UI patterns over introducing new
  abstractions.
- Avoid editing generated artifacts such as `uv.lock` unless dependency changes
  require it.
- Do not put secrets from `.env`, OAuth files, or auth JSON into docs, tests, or
  logs.
- For file edits in app/tool behavior, preserve exact provider conversation
  validity: every tool use needs a matching tool result.
- When updating CLI docs, check actual argparse flags in `cli/app.py` and command
  help in `cli/commands.py`; README and `CLAUDE.md` may lag current code.
