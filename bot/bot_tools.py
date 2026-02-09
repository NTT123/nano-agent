"""Discord-specific tool classes for the bot."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Annotated, Any

import discord
import httpx

from nano_agent import ExecutionContext, TextContent, get_default_tools
from nano_agent.tools.base import Desc, Tool, ToolResult

from .bot_state import BotState, chunk_message


# --- Helpers ---


def _format_channel_type(channel_obj: Any) -> str:
    """Convert a Discord channel type enum/object into a readable string."""
    channel_type = getattr(channel_obj, "type", None)
    if channel_type is None:
        return type(channel_obj).__name__
    return getattr(channel_type, "name", str(channel_type))


def _summarize_channel(channel_obj: Any) -> dict[str, Any]:
    """Summarize a Discord channel/thread object for exploration output."""
    payload: dict[str, Any] = {
        "id": getattr(channel_obj, "id", None),
        "name": getattr(channel_obj, "name", None),
        "type": _format_channel_type(channel_obj),
    }

    if hasattr(channel_obj, "position"):
        payload["position"] = getattr(channel_obj, "position")
    if hasattr(channel_obj, "parent_id"):
        payload["parent_id"] = getattr(channel_obj, "parent_id")
    if hasattr(channel_obj, "category_id"):
        payload["category_id"] = getattr(channel_obj, "category_id")
    if hasattr(channel_obj, "topic"):
        payload["topic"] = getattr(channel_obj, "topic")
    if isinstance(channel_obj, discord.Thread):
        payload["archived"] = channel_obj.archived
        payload["locked"] = channel_obj.locked
        payload["message_count"] = channel_obj.message_count
        payload["member_count"] = channel_obj.member_count

    return payload


def build_discord_explore_payload(
    bot: discord.Client,
    *,
    channel: discord.abc.Messageable | None = None,
    include_channels: bool = True,
    include_threads: bool = True,
    channel_limit: int = 50,
    thread_limit: int = 50,
) -> dict[str, Any]:
    """Build an exploration payload for the current Discord context."""
    try:
        parsed_channel_limit = int(channel_limit)
    except (TypeError, ValueError):
        parsed_channel_limit = 50
    try:
        parsed_thread_limit = int(thread_limit)
    except (TypeError, ValueError):
        parsed_thread_limit = 50
    clamped_channel_limit = max(1, min(parsed_channel_limit, 200))
    clamped_thread_limit = max(1, min(parsed_thread_limit, 200))

    payload: dict[str, Any] = {
        "mode": "explore",
        "bot": {
            "id": getattr(bot.user, "id", None),
            "username": getattr(bot.user, "name", None),
            "display_name": getattr(bot.user, "display_name", None),
        },
        "context": {
            "active_channel_id": getattr(channel, "id", None),
            "active_channel_name": getattr(channel, "name", None),
            "active_channel_type": type(channel).__name__
            if channel is not None
            else None,
            "active_guild_id": None,
            "active_guild_name": None,
            "active_thread_parent_id": None,
        },
        "known_guilds": [
            {
                "id": guild.id,
                "name": guild.name,
                "channels": len(guild.channels),
                "active_threads": len(guild.threads),
            }
            for guild in bot.guilds[:50]
        ],
        "current_guild": None,
    }

    if isinstance(channel, discord.Thread):
        payload["context"]["active_thread_parent_id"] = channel.parent_id

    guild = None
    if isinstance(channel, (discord.TextChannel, discord.Thread)):
        guild = channel.guild
        payload["context"]["active_guild_id"] = guild.id
        payload["context"]["active_guild_name"] = guild.name

    if guild is not None:
        guild_payload: dict[str, Any] = {
            "id": guild.id,
            "name": guild.name,
            "member_count": guild.member_count,
        }

        if include_channels:
            visible_channels = sorted(
                guild.channels,
                key=lambda channel_obj: (
                    getattr(channel_obj, "position", 0),
                    str(getattr(channel_obj, "name", "")),
                ),
            )
            guild_payload["channels"] = [
                _summarize_channel(channel_obj)
                for channel_obj in visible_channels[:clamped_channel_limit]
            ]

        if include_threads:
            active_threads = sorted(
                guild.threads,
                key=lambda thread_obj: (
                    str(getattr(thread_obj, "name", "")),
                    getattr(thread_obj, "id", 0),
                ),
            )
            guild_payload["active_threads"] = [
                _summarize_channel(thread_obj)
                for thread_obj in active_threads[:clamped_thread_limit]
            ]

        payload["current_guild"] = guild_payload

    payload["notes"] = [
        "Use this tool to discover IDs and visibility before making DiscordAPI requests.",
        "For deeper operations, call DiscordAPI with action='request'.",
    ]
    return payload


# --- Tool input dataclasses ---


@dataclass
class SendFileInput:
    file_path: Annotated[str, Desc("Absolute or relative path to the file to send")]
    message: Annotated[str, Desc("Optional message to send with the file")] = ""


@dataclass
class SendUserMessageInput:
    message: Annotated[str, Desc("Message to send to the user on Discord")]


@dataclass
class PeekQueuedUserMessagesInput:
    limit: Annotated[int, Desc("How many queued messages to show (1-50)")] = 5


@dataclass
class DequeueUserMessagesInput:
    count: Annotated[int, Desc("How many queued messages to consume (1-20)")] = 1


@dataclass
class CreateThreadInput:
    topic: Annotated[str, Desc("Topic / title for the new thread")]
    message: Annotated[str, Desc("Initial message to send in the thread")] = ""


@dataclass
class ExploreDiscordInput:
    include_channels: Annotated[
        bool,
        Desc("Include visible channels from the active guild"),
    ] = True
    include_threads: Annotated[
        bool,
        Desc("Include visible active threads from the active guild"),
    ] = True
    channel_limit: Annotated[
        int,
        Desc("Maximum number of channels to include (1-200)"),
    ] = 50
    thread_limit: Annotated[
        int,
        Desc("Maximum number of active threads to include (1-200)"),
    ] = 50


@dataclass
class DiscordAPIInput:
    action: Annotated[
        str,
        Desc("`discover` to inspect capabilities, or `request` to call Discord REST API"),
    ]
    method: Annotated[
        str,
        Desc("HTTP method for `request` mode: GET, POST, PATCH, PUT, DELETE"),
    ] = "GET"
    path: Annotated[
        str,
        Desc("Discord REST path for `request`, e.g. /channels/{channel_id}/messages"),
    ] = ""
    query_json: Annotated[
        str,
        Desc("JSON object for query params in `request` mode (default: {})"),
    ] = "{}"
    body_json: Annotated[
        str,
        Desc("JSON payload for `request` mode (default: {})"),
    ] = "{}"
    reason: Annotated[str, Desc("Optional short reason for auditability")] = ""


@dataclass
class RestartBotInput:
    reason: Annotated[str, Desc("Reason for restarting the bot")] = "User requested restart"


# --- Tool classes ---


@dataclass
class SendFileTool(Tool):
    name: str = "SendFile"
    description: str = (
        "Send a file to the user via Discord. Use this when the user asks you "
        "to share, send, or show them a file. The file will be uploaded as a "
        "Discord attachment. Max file size is 25MB."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: SendFileInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        channel = self.state.active_channel
        if channel is None:
            return ToolResult(content=TextContent(text="Error: No active Discord channel"))

        path = os.path.expanduser(input.file_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        if not os.path.isfile(path):
            return ToolResult(content=TextContent(text=f"Error: File not found: {path}"))

        size = os.path.getsize(path)
        if size > 8 * 1024 * 1024:
            mb = size / (1024 * 1024)
            return ToolResult(
                content=TextContent(
                    text=f"Error: File too large ({mb:.1f}MB). "
                    "Discord limit is 8MB for unboosted servers (25MB/50MB for boosted). "
                    "Consider compressing the file first."
                )
            )

        try:
            file = discord.File(path, filename=os.path.basename(path))
            await channel.send(content=input.message or None, file=file)
            return ToolResult(
                content=TextContent(text=f"File sent: {os.path.basename(path)} ({size} bytes)")
            )
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Error sending file: {e}"))


@dataclass
class SendUserMessageTool(Tool):
    name: str = "SendUserMessage"
    description: str = (
        "Send a message to the user on Discord. "
        "Use this tool whenever you want the user to see text output."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: SendUserMessageInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        channel = self.state.active_channel
        channel_id = self.state.active_channel_id
        if channel is None or channel_id is None:
            return ToolResult(content=TextContent(text="Error: No active Discord channel"))

        if not input.message.strip():
            return ToolResult(content=TextContent(text="Error: message cannot be empty"))

        sent_chunks = 0
        for chunk in chunk_message(input.message):
            await channel.send(chunk)
            sent_chunks += 1

        if self.state.active_run_stats is not None:
            self.state.active_run_stats["outbound_messages"] = (
                self.state.active_run_stats.get("outbound_messages", 0) + 1
            )

        self.state.append_internal_log(
            channel_id,
            "assistant_message_sent",
            {"chunks": sent_chunks, "message_preview": input.message[:300]},
        )
        return ToolResult(content=TextContent(text=f"Sent {sent_chunks} chunk(s) to user."))


@dataclass
class PeekQueuedUserMessagesTool(Tool):
    name: str = "PeekQueuedUserMessages"
    description: str = (
        "Inspect queued user messages without consuming them. "
        "Use this to see what users have sent while you were working."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: PeekQueuedUserMessagesInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        channel_id = self.state.active_channel_id
        if channel_id is None:
            return ToolResult(content=TextContent(text="Error: No active Discord channel"))

        queue = self.state.get_channel_queue(channel_id)
        items = self.state.peek_user_messages(channel_id, input.limit)
        payload = {
            "pending_count": len(queue),
            "messages": items,
        }
        return ToolResult(content=TextContent(text=json.dumps(payload, indent=2)))


@dataclass
class DequeueUserMessagesTool(Tool):
    name: str = "DequeueUserMessages"
    description: str = (
        "Consume queued user messages from oldest to newest and return them. "
        "Use this when you are ready to process user input."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: DequeueUserMessagesInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        channel_id = self.state.active_channel_id
        if channel_id is None:
            return ToolResult(content=TextContent(text="Error: No active Discord channel"))

        items = self.state.dequeue_user_messages(channel_id, input.count)
        if self.state.active_run_stats is not None:
            self.state.active_run_stats["dequeued_messages"] = (
                self.state.active_run_stats.get("dequeued_messages", 0) + len(items)
            )

        payload = {
            "dequeued_count": len(items),
            "remaining_count": len(self.state.get_channel_queue(channel_id)),
            "messages": items,
        }
        return ToolResult(content=TextContent(text=json.dumps(payload, indent=2)))


@dataclass
class CreateThreadTool(Tool):
    name: str = "CreateThread"
    description: str = (
        "Create a new Discord thread in the current channel and start a "
        "conversation there. Use this when the discussion would benefit from "
        "a dedicated thread (e.g. a sub-topic, a long task, or to keep the "
        "main channel tidy). The bot will automatically respond to messages "
        "in the new thread."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: CreateThreadInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        channel = self.state.active_channel
        if channel is None:
            return ToolResult(content=TextContent(text="Error: No active Discord channel"))

        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return ToolResult(
                content=TextContent(
                    text="Error: Threads can only be created in text channels"
                )
            )

        target = channel.parent if isinstance(channel, discord.Thread) else channel
        if not isinstance(target, discord.TextChannel):
            return ToolResult(
                content=TextContent(text="Error: Cannot create thread here")
            )

        try:
            thread = await target.create_thread(
                name=input.topic,
                type=discord.ChannelType.public_thread,
            )
            self.state.sessions[thread.id] = self.state.create_session()
            initial = input.message or f"New conversation started: **{input.topic}**"
            await thread.send(initial)
            return ToolResult(
                content=TextContent(
                    text=f"Thread created: #{thread.name} (id: {thread.id}). "
                    "The user can now continue the conversation there."
                )
            )
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Error creating thread: {e}"))


@dataclass
class ExploreDiscordTool(Tool):
    name: str = "ExploreDiscord"
    description: str = (
        "Explore Discord context available to the bot, including current guild/channel "
        "visibility and IDs. Use this before DiscordAPI requests when you need to "
        "discover what is accessible."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: ExploreDiscordInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        bot = self.state.bot
        if bot is None:
            return ToolResult(content=TextContent(text="Error: Bot not initialized"))

        payload = build_discord_explore_payload(
            bot,
            channel=self.state.active_channel,
            include_channels=input.include_channels,
            include_threads=input.include_threads,
            channel_limit=input.channel_limit,
            thread_limit=input.thread_limit,
        )
        text = json.dumps(payload, indent=2)
        if len(text) > 20000:
            text = text[:20000] + "\n... (truncated)"
        return ToolResult(content=TextContent(text=text))


@dataclass
class DiscordAPITool(Tool):
    name: str = "DiscordAPI"
    description: str = (
        "Discover and execute Discord REST API operations with the bot token. "
        "Use action='discover' to inspect context/capabilities first, then "
        "action='request' with method/path/query_json/body_json to call endpoints. "
        "This can access broad Discord functionality, limited by the bot's permissions."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: DiscordAPIInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
        if not token:
            return ToolResult(
                content=TextContent(text="Error: DISCORD_BOT_TOKEN is not configured.")
            )

        action = input.action.strip().lower()
        if action == "discover":
            return await self._discover(token)
        if action == "request":
            return await self._request(input, token)

        return ToolResult(
            content=TextContent(
                text="Error: action must be `discover` or `request`."
            )
        )

    async def _discover(self, token: str) -> ToolResult:
        bot = self.state.bot
        channel = self.state.active_channel
        context: dict[str, Any] = {
            "active_channel_id": getattr(channel, "id", None),
            "active_channel_type": type(channel).__name__ if channel else None,
            "active_guild_id": None,
            "active_guild_name": None,
            "active_thread_parent_id": None,
        }

        if isinstance(channel, discord.Thread):
            context["active_thread_parent_id"] = channel.parent_id

        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            context["active_guild_id"] = channel.guild.id
            context["active_guild_name"] = channel.guild.name

        known_guilds = []
        if bot is not None:
            known_guilds = [
                {
                    "id": g.id,
                    "name": g.name,
                    "channels": len(g.channels),
                    "roles": len(g.roles),
                }
                for g in bot.guilds[:20]
            ]

        headers = {"Authorization": f"Bot {token}"}
        auth_status = None
        bot_identity: dict[str, Any] | None = None
        try:
            async with httpx.AsyncClient(
                base_url="https://discord.com/api/v10",
                timeout=20,
            ) as client:
                me_resp = await client.get("/users/@me", headers=headers)
            auth_status = me_resp.status_code
            if me_resp.status_code == 200:
                me_json = me_resp.json()
                bot_identity = {
                    "id": me_json.get("id"),
                    "username": me_json.get("username"),
                    "global_name": me_json.get("global_name"),
                }
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Discord discover failed: {e}"))

        payload = {
            "mode": "discover",
            "auth_check_status": auth_status,
            "bot_identity": bot_identity,
            "context": context,
            "known_guilds": known_guilds,
            "common_endpoint_templates": [
                {"method": "GET", "path": "/users/@me"},
                {"method": "GET", "path": "/guilds/{guild_id}/channels"},
                {"method": "GET", "path": "/guilds/{guild_id}/roles"},
                {"method": "GET", "path": "/channels/{channel_id}/messages"},
                {"method": "POST", "path": "/channels/{channel_id}/messages"},
                {"method": "POST", "path": "/channels/{channel_id}/threads"},
                {"method": "PATCH", "path": "/channels/{channel_id}"},
                {"method": "GET", "path": "/guilds/{guild_id}/members/{user_id}"},
                {"method": "PUT", "path": "/guilds/{guild_id}/members/{user_id}/roles/{role_id}"},
            ],
            "request_mode_usage": {
                "action": "request",
                "method": "GET",
                "path": "/channels/{channel_id}/messages",
                "query_json": '{"limit": 10}',
                "body_json": "{}",
            },
            "notes": [
                "Use numeric IDs in path parameters.",
                "This tool can access Discord REST functionality supported by bot token permissions.",
                "Discord rate limits apply; inspect rate_limit headers in tool results.",
            ],
        }
        return ToolResult(content=TextContent(text=json.dumps(payload, indent=2)))

    async def _request(self, input: DiscordAPIInput, token: str) -> ToolResult:
        method = input.method.strip().upper() or "GET"
        if method not in {"GET", "POST", "PATCH", "PUT", "DELETE"}:
            return ToolResult(
                content=TextContent(
                    text=f"Error: unsupported method `{method}`. Use GET/POST/PATCH/PUT/DELETE."
                )
            )

        path = input.path.strip()
        if not path.startswith("/"):
            return ToolResult(
                content=TextContent(text="Error: path must start with `/`.")
            )

        try:
            query_obj = (
                json.loads(input.query_json) if input.query_json.strip() else {}
            )
        except json.JSONDecodeError as e:
            return ToolResult(
                content=TextContent(text=f"Error: query_json is invalid JSON: {e}")
            )
        if query_obj is None:
            query_obj = {}
        if not isinstance(query_obj, dict):
            return ToolResult(
                content=TextContent(text="Error: query_json must decode to a JSON object.")
            )

        try:
            body_obj = json.loads(input.body_json) if input.body_json.strip() else {}
        except json.JSONDecodeError as e:
            return ToolResult(
                content=TextContent(text=f"Error: body_json is invalid JSON: {e}")
            )

        headers = {"Authorization": f"Bot {token}"}
        if method in {"POST", "PATCH", "PUT"}:
            headers["Content-Type"] = "application/json"

        try:
            async with httpx.AsyncClient(
                base_url="https://discord.com/api/v10",
                timeout=30,
            ) as client:
                resp = await client.request(
                    method,
                    path,
                    headers=headers,
                    params=query_obj or None,
                    json=body_obj if method in {"POST", "PATCH", "PUT"} else None,
                )
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Discord request failed: {e}"))

        content_type = resp.headers.get("content-type", "")
        response_data: Any
        if "application/json" in content_type:
            try:
                response_data = resp.json()
            except Exception:
                response_data = resp.text
        else:
            response_data = resp.text

        payload = {
            "mode": "request",
            "reason": input.reason,
            "request": {
                "method": method,
                "path": path,
                "query": query_obj,
                "body": body_obj if method in {"POST", "PATCH", "PUT"} else None,
            },
            "response": {
                "status_code": resp.status_code,
                "rate_limit": {
                    "limit": resp.headers.get("x-ratelimit-limit"),
                    "remaining": resp.headers.get("x-ratelimit-remaining"),
                    "reset_after": resp.headers.get("x-ratelimit-reset-after"),
                    "bucket": resp.headers.get("x-ratelimit-bucket"),
                },
                "data": response_data,
            },
        }
        text = json.dumps(payload, indent=2)
        if len(text) > 20000:
            text = text[:20000] + "\n... (truncated)"
        return ToolResult(content=TextContent(text=text))


@dataclass
class ClearContextTool(Tool):
    name: str = "ClearContext"
    description: str = (
        "Clear the conversation context/history in the current channel or thread. "
        "This resets the session so the bot starts fresh with no memory of previous messages. "
        "Use this when the user asks to clear context, reset the conversation, or start over."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        channel = self.state.active_channel
        if channel is None:
            return ToolResult(content=TextContent(text="Error: No active Discord channel"))
        self.state.clear_context_requested.add(channel.id)
        return ToolResult(
            content=TextContent(
                text="Context cleared. The next message will start a fresh conversation."
            )
        )


@dataclass
class RestartBotTool(Tool):
    name: str = "RestartBot"
    description: str = (
        "Restart the Discord bot process. This kills the current bot process and "
        "starts a new one in the background. The bot will briefly go offline and "
        "come back. Use this when the user asks to restart, reload, or reset the bot."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: RestartBotInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        channel = self.state.active_channel
        reason = input.reason

        if channel is not None:
            try:
                await channel.send(f"Restarting bot... Reason: {reason}")
            except Exception:
                pass

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_file = os.path.join(project_root, "logs", "discord_bot.nohup.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        restart_cmd = (
            f"cd {project_root} && sleep 2 && "
            f"nohup uv run nano-discord-bot "
            f">> {log_file} 2>&1 &"
        )
        subprocess.Popen(
            restart_cmd,
            shell=True,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"[RESTART] Shutting down for restart. Reason: {reason}")
        os._exit(0)

        # Unreachable, but satisfies type checker
        return ToolResult(content=TextContent(text="Restarting..."))


def get_tools(state: BotState) -> list[Tool]:
    """Get tools suitable for Discord (excludes AskUserQuestion, adds Discord tools)."""
    tools: list[Tool] = [t for t in get_default_tools() if t.name != "AskUserQuestion"]
    tools.append(SendUserMessageTool(state=state))
    tools.append(PeekQueuedUserMessagesTool(state=state))
    tools.append(DequeueUserMessagesTool(state=state))
    tools.append(SendFileTool(state=state))
    tools.append(CreateThreadTool(state=state))
    tools.append(ExploreDiscordTool(state=state))
    tools.append(DiscordAPITool(state=state))
    tools.append(ClearContextTool(state=state))
    tools.append(RestartBotTool(state=state))
    return tools
