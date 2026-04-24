"""Slack bot frontend for nano-agent.

Mirrors the Discord bot (``bot/discord_bot.py``) — same queue/worker/tools
architecture — but drives the agent loop from Slack events via slack-bolt
Socket Mode.

Usage:
    1. Create a Slack app with Socket Mode enabled. Bot scopes needed:
       ``app_mentions:read``, ``chat:write``, ``channels:history``,
       ``channels:read``, ``groups:history``, ``im:history``, ``im:read``,
       ``files:write``, ``users:read``, ``commands``.
    2. Install the app to your workspace and copy the Bot User OAuth Token
       (``xoxb-…``) and App-Level Token with ``connections:write``
       (``xapp-…``) into .env as ``SLACK_BOT_TOKEN`` and ``SLACK_APP_TOKEN``.
    3. Register these slash commands in the app manifest (request URL is
       unused under Socket Mode — Slack just needs them declared):
       ``/clear``, ``/queue``, ``/cd``, ``/cwd``, ``/thread``, ``/renew``.
    4. (Optional) set ``BOT_PROVIDER=codex`` to use ChatGPT/Codex OAuth.
    5. ``uv run nano-slack-bot``
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from nano_agent import DAG
from nano_agent.providers.base import APIProtocol

from .bot_agent import ensure_channel_worker
from .bot_config import async_renew_oauth, build_api_from_env, get_bot_provider
from .bot_state import BotState, chunk_message, truncate
from .slack_tools import (
    SLACK_MESSAGE_LIMIT,
    SlackContext,
    composite_channel_key,
    get_slack_tools,
)

load_dotenv()

RESPOND_TO_ALL_MESSAGES = os.getenv(
    "SLACK_RESPOND_TO_ALL_MESSAGES", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# Subtypes we explicitly ignore — edits, deletes, joins/leaves, bot echoes.
# File uploads arrive as ``subtype="file_share"`` and must pass through.
_IGNORED_SUBTYPES = {
    "message_changed",
    "message_deleted",
    "message_replied",
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "bot_message",
    "tombstone",
}

SYSTEM_PROMPT = (
    "You are an assistant running 24/7 on a machine. "
    "You are connected with the user via Slack. "
    "You have access to tools for reading files, running commands, "
    "editing code, searching, and browsing the web. "
    "Use the ExploreSlack tool first when you need to inspect current "
    "Slack context (workspace/channel/thread visibility). "
    "Use the SlackAPI tool for Slack Web API operations beyond the built-in tools. "
    "User uploads appear in the [Attachments] section of the incoming message "
    "with a url_private URL. To read an upload, call SlackDownloadFile with "
    "that URL and a local save_path, then use Read (for text files) or other "
    "tools on the saved path. Do not use WebFetch on Slack file URLs — they "
    "require the bot token. "
    "IMPORTANT: normal assistant text is internal-only and is not sent to the user. "
    "To send anything to Slack, you MUST call the SendUserMessage tool. "
    "Incoming user messages are queued. Use PeekQueuedUserMessages to inspect "
    "queue state and DequeueUserMessages to consume messages when ready. "
    "You may keep working on current tasks and process queued user messages later. "
    "Keep responses concise — Slack messages are split at ~3500 characters, "
    "so long responses will be sent as multiple messages. "
    "Slack markdown is mrkdwn flavor: *bold*, _italic_, `code`, ```block```, "
    ">quote. Tables are not supported; use code blocks with aligned columns."
)


app = AsyncApp(token=os.getenv("SLACK_BOT_TOKEN"))
api: APIProtocol = build_api_from_env()
state = BotState(state_root=Path("logs/slack_agent_state"))
state.tools_factory = lambda: get_slack_tools(state, app.client)
_bot_user_id: str | None = None

# Slack renders user mentions as ``<@U01234>`` in raw text; strip the bot's
# own mention before passing to the agent.
_BOT_MENTION_RE = re.compile(r"<@[A-Z0-9]+>\s*")


def _strip_bot_mentions(text: str) -> str:
    return _BOT_MENTION_RE.sub("", text or "").strip()


def _describe_slack_file(f: dict[str, Any]) -> str:
    name = f.get("name") or f.get("title") or f.get("id") or "unknown"
    mimetype = f.get("mimetype") or f.get("filetype") or "?"
    size = f.get("size")
    url = f.get("url_private") or f.get("permalink") or ""
    size_str = f" {size}B" if isinstance(size, int) else ""
    return f"- {name} (type={mimetype}{size_str}) {url}".rstrip()


def _create_session(cwd: str | None = None) -> DAG:
    return state.create_session(
        cwd, system_prompt=SYSTEM_PROMPT, tools=get_slack_tools(state, app.client)
    )


# --- Adapter shim: channel_worker expects ``channel`` to have a platform-specific
#     handle. For Slack we pass the SlackContext since that carries both
#     channel_id and thread_ts; tools read ``state.active_channel`` directly.


def _make_channel_ref(channel_id: str, thread_ts: str | None) -> SlackContext:
    return SlackContext(channel_id=channel_id, thread_ts=thread_ts)


# --- Startup helpers ---


async def _recover_pending_queues() -> None:
    """Resume workers for channels with persisted queued messages."""
    for key in state.persisted_channel_ids():
        pending = len(state.get_channel_queue(key))
        if pending <= 0:
            continue
        # Composite keys are "<channel_id>:<thread_ts>"; plain channel ids
        # have no ":". Parse either form.
        if ":" in key:
            channel_id, thread_ts = key.split(":", 1)
        else:
            channel_id, thread_ts = key, None
        channel_ref = _make_channel_ref(channel_id, thread_ts)
        print(f"[QUEUE_RECOVER] Resuming queue for {key} (pending={pending})")
        ensure_channel_worker(state, api, channel_ref, key, SYSTEM_PROMPT)


# --- Events ---


@app.event("app_mention")
async def on_app_mention(event: dict[str, Any], client: Any) -> None:
    await _handle_user_message(event)


@app.event("message")
async def on_message(event: dict[str, Any], client: Any) -> None:
    if event.get("bot_id"):
        return
    if event.get("subtype") in _IGNORED_SUBTYPES:
        return
    channel_type = event.get("channel_type")
    if channel_type == "im":
        await _handle_user_message(event)
    elif RESPOND_TO_ALL_MESSAGES and channel_type in {"channel", "group"}:
        await _handle_user_message(event, force=True)


async def _handle_user_message(event: dict[str, Any], *, force: bool = False) -> None:
    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text") or ""
    ts = event.get("ts")
    thread_ts = event.get("thread_ts") or (ts if force else None)

    if not channel_id or not user_id:
        return
    if _bot_user_id and user_id == _bot_user_id:
        return

    cleaned = _strip_bot_mentions(text)
    files = event.get("files") or []
    if files:
        file_summary = "\n".join(_describe_slack_file(f) for f in files)
        prefix = f"{cleaned}\n\n" if cleaned else ""
        cleaned = f"{prefix}[Attachments]\n{file_summary}"
    if not cleaned:
        return

    # For channel messages we always reply in-thread so the channel stays tidy.
    # For DMs, thread_ts stays None (DMs don't thread).
    if event.get("channel_type") != "im" and thread_ts is None:
        thread_ts = ts

    key = composite_channel_key(str(channel_id), thread_ts)
    state.channel_last_user_id[key] = str(user_id)

    queued = state.enqueue_user_message(
        key,
        message_id=str(ts),
        author_id=str(user_id),
        author=str(user_id),
        content=cleaned,
        attachments=[f.get("url_private", "") for f in event.get("files") or []],
    )
    print(
        f"[MSG_QUEUED] user={user_id} key={key} "
        f"queue_id={queued['queue_id']} pending={len(state.get_channel_queue(key))}"
    )
    ensure_channel_worker(
        state,
        api,
        _make_channel_ref(str(channel_id), thread_ts),
        key,
        SYSTEM_PROMPT,
    )


# --- Slash commands ---


def _ack_text(text: str) -> dict[str, Any]:
    """Return a payload Bolt will send via ack() for a slash command."""
    return {"response_type": "ephemeral", "text": text}


@app.command("/clear")
async def cmd_clear(ack: Any, command: dict[str, Any]) -> None:
    channel_id = str(command.get("channel_id") or "")
    user_id = str(command.get("user_id") or "")
    # Slash commands don't carry thread_ts; /clear applies to the channel.
    key = channel_id
    cwd = state.working_dirs.get(user_id)
    state.set_session(key, _create_session(cwd))
    state.clear_user_queue(key)
    await ack(_ack_text("Conversation cleared."))


@app.command("/queue")
async def cmd_queue(ack: Any, command: dict[str, Any]) -> None:
    channel_id = str(command.get("channel_id") or "")
    if not channel_id:
        await ack(_ack_text("Cannot inspect queue here."))
        return
    queue = state.get_channel_queue(channel_id)
    pending = len(queue)
    worker = state.channel_worker_tasks.get(channel_id)
    worker_running = worker is not None and not worker.done()
    if pending == 0:
        await ack(
            _ack_text(
                f"Queue is empty. Worker running: {'yes' if worker_running else 'no'}"
            )
        )
        return
    preview = state.peek_user_messages(channel_id, 5)
    lines = [
        f"Pending queued messages: {pending}",
        f"Worker running: {'yes' if worker_running else 'no'}",
    ]
    for item in preview:
        qid = item.get("queue_id", "?")
        author = item.get("author", "unknown")
        content = truncate(str(item.get("content", "")).replace("\n", " "), 120)
        lines.append(f"#{qid} {author}: {content}")
    await ack(_ack_text("\n".join(lines)))


@app.command("/cd")
async def cmd_cd(ack: Any, command: dict[str, Any]) -> None:
    user_id = str(command.get("user_id") or "")
    channel_id = str(command.get("channel_id") or "")
    path = (command.get("text") or "").strip()
    if not path:
        await ack(_ack_text("Usage: /cd <path>"))
        return
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(resolved):
        await ack(_ack_text(f"Not a directory: `{resolved}`"))
        return
    state.working_dirs[user_id] = resolved
    state.set_session(channel_id, _create_session(resolved))
    await ack(_ack_text(f"Working directory: `{resolved}` (conversation reset)"))


@app.command("/cwd")
async def cmd_cwd(ack: Any, command: dict[str, Any]) -> None:
    user_id = str(command.get("user_id") or "")
    cwd = state.working_dirs.get(user_id, os.getcwd())
    await ack(_ack_text(f"Working directory: `{cwd}`"))


@app.command("/thread")
async def cmd_thread(ack: Any, command: dict[str, Any], client: Any) -> None:
    channel_id = str(command.get("channel_id") or "")
    user_id = str(command.get("user_id") or "")
    topic = (command.get("text") or "New conversation").strip()
    try:
        parent = await client.chat_postMessage(channel=channel_id, text=f"*{topic}*")
    except Exception as e:
        await ack(_ack_text(f"Failed to open thread: {e}"))
        return
    new_ts = parent.get("ts") if isinstance(parent, dict) else None
    if not new_ts:
        await ack(_ack_text("Thread open returned no ts."))
        return
    key = composite_channel_key(channel_id, new_ts)
    cwd = state.working_dirs.get(user_id)
    state.set_session(key, _create_session(cwd))
    await ack(
        _ack_text(
            f"Thread started in <#{channel_id}> (thread_ts={new_ts}). "
            "Reply in the thread to continue."
        )
    )
    try:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=new_ts,
            text=f"New conversation started: *{topic}*\nReply here to chat.",
        )
    except Exception:
        pass


@app.command("/renew")
async def cmd_renew(ack: Any, command: dict[str, Any]) -> None:
    await ack(_ack_text("Refreshing OAuth… check the bot logs for browser prompt."))
    global api
    try:
        await async_renew_oauth()
        api = build_api_from_env()
        state.delete_all_session_files()
        state.sessions.clear()
    except Exception as e:
        print(f"[RENEW] failed: {e}")
        traceback.print_exc()


# --- Entry point ---


async def _async_main() -> None:
    global _bot_user_id
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        print("Error: set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in .env or environment")
        sys.exit(1)

    auth = await app.client.auth_test()
    _bot_user_id = auth.get("user_id")
    print(
        f"[READY] Slack bot user={_bot_user_id} team={auth.get('team')} "
        f"provider={get_bot_provider()}"
    )
    await _recover_pending_queues()
    handler = AsyncSocketModeHandler(app, app_token)
    await handler.start_async()  # type: ignore[no-untyped-call]


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
