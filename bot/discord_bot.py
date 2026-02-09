"""Discord bot frontend for nano-agent.

Connects to Discord and uses the nano-agent core (DAG, executor, tools)
to provide an AI assistant. Each user gets their own conversation session.

Usage:
    1. Set DISCORD_BOT_TOKEN in .env or environment
    2. uv run nano-discord-bot
"""

import json
import os
import sys
import traceback

import discord
from discord import app_commands
from dotenv import load_dotenv

from nano_agent import DAG, ClaudeCodeAPI

from .bot_agent import ensure_channel_worker
from .bot_state import BotState, chunk_message, truncate
from .bot_tools import build_discord_explore_payload, get_tools

load_dotenv()

SYSTEM_PROMPT = (
    "You are an assistant running 24/7 on a machine. "
    "You are connected with the user via Discord. "
    "You are a helpful AI assistant running inside a Discord bot. "
    "You have access to tools for reading files, running commands, "
    "editing code, searching, and browsing the web. "
    "Use the ExploreDiscord tool first when you need to inspect current "
    "Discord context (guild/channel/thread visibility). "
    "Use the DiscordAPI tool when you need to discover or execute Discord-specific "
    "REST API functionality. "
    "IMPORTANT: normal assistant text is internal-only and is not sent to the user. "
    "To send anything to Discord, you MUST call the SendUserMessage tool. "
    "Incoming user messages are queued. Use PeekQueuedUserMessages to inspect queue state "
    "and DequeueUserMessages to consume messages when you are ready. "
    "You may keep working on current tasks and process queued user messages later. "
    "Keep responses concise — Discord has a 2000 character message limit, "
    "so long responses will be split across multiple messages. "
    "Discord does NOT support markdown tables. For tabular data, "
    "use code blocks (```text) with monospace-aligned columns instead."
)

RESPOND_TO_ALL_MESSAGES = os.getenv(
    "DISCORD_RESPOND_TO_ALL_MESSAGES", "true"
).strip().lower() in {"1", "true", "yes", "on"}

# --- Discord bot setup ---

intents = discord.Intents.default()
intents.message_content = True

BOT_PERMISSIONS = discord.Permissions(
    send_messages=True,
    send_messages_in_threads=True,
    create_public_threads=True,
    read_message_history=True,
)

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

api = ClaudeCodeAPI()
state = BotState(bot=bot)


def _create_session(cwd: str | None = None) -> DAG:
    """Helper that wires SYSTEM_PROMPT and tools into state.create_session."""
    return state.create_session(
        cwd, system_prompt=SYSTEM_PROMPT, tools=get_tools(state)
    )


# --- Startup helpers ---


async def recover_pending_queues() -> None:
    """Resume workers for channels that have persisted queued messages."""
    for channel_id in state.persisted_channel_ids():
        pending = len(state.get_channel_queue(channel_id))
        if pending <= 0:
            continue

        channel = bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(channel_id)
            except Exception as e:
                print(f"[QUEUE_RECOVER] Failed to fetch channel {channel_id}: {e}")
                continue

        print(
            f"[QUEUE_RECOVER] Resuming queue for channel {channel_id} (pending={pending})"
        )
        ensure_channel_worker(state, api, channel, SYSTEM_PROMPT)


# --- Discord events ---


@bot.event
async def on_ready() -> None:
    print(f"[READY] Logged in as {bot.user}. Syncing slash commands...")
    try:
        synced = await tree.sync()
        print(f"[READY] Synced {len(synced)} slash commands.")
    except Exception as e:
        print(f"[READY] Slash command sync failed: {e}")
    await recover_pending_queues()


@tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
) -> None:
    """Return a visible error message instead of Discord's generic timeout banner."""
    print(f"[APP_CMD_ERROR] {type(error).__name__}: {error}")
    traceback.print_exception(type(error), error, error.__traceback__)
    message = f"Command failed: {error}"
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message)
        else:
            await interaction.response.send_message(message)
    except Exception as send_error:
        print(f"[APP_CMD_ERROR] Failed to send error message: {send_error}")


# --- Slash commands ---


@tree.command(
    name="clear", description="Clear conversation history in this channel/thread"
)
async def clear_command(interaction: discord.Interaction) -> None:
    cwd = state.working_dirs.get(interaction.user.id)
    state.sessions[interaction.channel_id] = _create_session(cwd)
    state.clear_user_queue(interaction.channel_id)
    await interaction.response.send_message("Conversation cleared.")


@tree.command(
    name="queue", description="Show queued user messages in this channel/thread"
)
@app_commands.describe(limit="How many queued messages to preview (1-20)")
async def queue_command(interaction: discord.Interaction, limit: int = 5) -> None:
    channel_id = interaction.channel_id
    if channel_id is None:
        await interaction.response.send_message("Cannot inspect queue in this context.")
        return

    bounded_limit = max(1, min(limit, 20))
    queue = state.get_channel_queue(channel_id)
    pending = len(queue)
    worker = state.channel_worker_tasks.get(channel_id)
    worker_running = worker is not None and not worker.done()

    if pending == 0:
        await interaction.response.send_message(
            "Queue is empty. Pending: 0.\n"
            f"Worker running: {'yes' if worker_running else 'no'}"
        )
        return

    preview = state.peek_user_messages(channel_id, bounded_limit)
    lines = [
        f"Pending queued messages: {pending}",
        f"Worker running: {'yes' if worker_running else 'no'}",
        f"Showing first {len(preview)}:",
    ]
    for item in preview:
        qid = item.get("queue_id", "?")
        author = item.get("author", "unknown")
        content = truncate(str(item.get("content", "")).replace("\n", " "), 120)
        lines.append(f"#{qid} {author}: {content}")

    text = "\n".join(lines)
    chunks = chunk_message(text)
    await interaction.response.send_message(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


@tree.command(name="cd", description="Change working directory")
@app_commands.describe(path="Directory path to change to")
async def cd_command(interaction: discord.Interaction, path: str) -> None:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(resolved):
        await interaction.response.send_message(f"Not a directory: `{resolved}`")
        return
    state.working_dirs[interaction.user.id] = resolved
    state.sessions[interaction.channel_id] = _create_session(resolved)
    await interaction.response.send_message(
        f"Working directory: `{resolved}` (conversation reset)"
    )


@tree.command(name="cwd", description="Show current working directory")
async def cwd_command(interaction: discord.Interaction) -> None:
    cwd = state.working_dirs.get(interaction.user.id, os.getcwd())
    await interaction.response.send_message(f"Working directory: `{cwd}`")


@tree.command(name="thread", description="Start a new conversation in a Discord thread")
@app_commands.describe(topic="Topic for the thread")
async def thread_command(
    interaction: discord.Interaction, topic: str = "New conversation"
) -> None:
    channel = interaction.channel
    if channel is None:
        await interaction.response.send_message("Cannot create thread here.")
        return
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Threads can only be created in text channels."
        )
        return
    thread = await channel.create_thread(
        name=topic,
        type=discord.ChannelType.public_thread,
    )
    cwd = state.working_dirs.get(interaction.user.id)
    state.sessions[thread.id] = _create_session(cwd)
    await interaction.response.send_message(f"Thread created: {thread.mention}")
    await thread.send(
        f"New conversation started: **{topic}**\nSend messages here to chat."
    )


@tree.command(name="renew", description="Refresh Claude Code OAuth token")
async def renew_command(interaction: discord.Interaction) -> None:
    await interaction.response.defer()
    try:
        from nano_agent.providers.capture_claude_code_auth import async_get_config

        await async_get_config(timeout=30)
        global api
        api = ClaudeCodeAPI()
        state.sessions.clear()
        await interaction.followup.send("OAuth token refreshed successfully.")
    except Exception as e:
        await interaction.followup.send(f"Failed to refresh token: {e}")


@tree.command(
    name="explore",
    description="Explore visible Discord context (guild/channels/threads)",
)
@app_commands.describe(
    include_channels="Include visible channels in the current guild",
    include_threads="Include active threads in the current guild",
)
async def explore_command(
    interaction: discord.Interaction,
    include_channels: bool = True,
    include_threads: bool = True,
) -> None:
    await interaction.response.defer()
    payload = build_discord_explore_payload(
        bot,
        channel=interaction.channel,
        include_channels=include_channels,
        include_threads=include_threads,
    )
    text = json.dumps(payload, indent=2)
    response = f"```json\n{text}\n```"
    for chunk in chunk_message(response):
        await interaction.followup.send(chunk)


# --- Message handler ---


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot or message.webhook_id is not None:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user is not None and bot.user in message.mentions
    is_thread = isinstance(message.channel, discord.Thread)

    should_respond = is_dm or is_mentioned or is_thread or RESPOND_TO_ALL_MESSAGES
    if not should_respond:
        return

    content = message.content
    if bot.user is not None:
        content = content.replace(f"<@{bot.user.id}>", "").strip()
        content = content.replace(f"<@!{bot.user.id}>", "").strip()

    if not content and message.attachments:
        content = "(message had only attachments)"

    if not content:
        return

    channel_id = message.channel.id
    user_id = message.author.id
    state.channel_last_user_id[channel_id] = user_id

    queued = state.enqueue_user_message(channel_id, message, content)
    print(
        f"[MSG_QUEUED] {message.author} (channel {channel_id}) "
        f"queue_id={queued['queue_id']} pending={len(state.get_channel_queue(channel_id))}"
    )
    ensure_channel_worker(state, api, message.channel, SYSTEM_PROMPT)


# --- Entry point ---


def main() -> None:
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("Error: Set DISCORD_BOT_TOKEN in .env or environment")
        sys.exit(1)
    bot.run(token)


if __name__ == "__main__":
    main()
