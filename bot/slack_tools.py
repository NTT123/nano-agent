"""Slack-specific tool classes for the bot.

Mirrors :mod:`bot.bot_tools` (which hosts the Discord tools). Shared
platform-agnostic tools (``PeekQueuedUserMessagesTool``,
``DequeueUserMessagesTool``, ``ClearContextTool``) are imported from
``bot_tools`` and reused verbatim.

Slack has no first-class "thread object" the way Discord does — threads are
reply chains keyed by ``thread_ts`` of the parent message. ``SlackContext``
holds the ``(channel_id, thread_ts)`` pair and is stored on
``state.active_channel``. The composite queue/session key used in
``BotState`` is the string ``f"{channel_id}:{thread_ts}"`` when threaded,
otherwise just ``channel_id``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from slack_sdk.web.async_client import AsyncWebClient

from nano_agent import ExecutionContext, TextContent, get_default_tools
from nano_agent.tools.base import Desc, Tool, ToolResult

from .bot_introspect import InspectBotStateTool
from .bot_state import BotState, chunk_message
from .bot_tools import (
    ClearContextTool,
    DequeueUserMessagesTool,
    PeekQueuedUserMessagesTool,
)

# Slack chat.postMessage soft limit is 40,000 chars, but ~3,500 keeps messages
# well under mobile truncation and matches coworker's convention.
SLACK_MESSAGE_LIMIT = 3500


class _NoopTyping:
    """Stand-in for ``discord.abc.Messageable.typing()`` on platforms without a
    typing indicator — lets the agent loop keep the ``async with channel.typing():``
    pattern unchanged."""

    async def __aenter__(self) -> "_NoopTyping":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


@dataclass
class SlackContext:
    """Slack channel handle carried on ``state.active_channel`` so tools know
    where to reply (which channel, and whether to post in a thread)."""

    channel_id: str
    thread_ts: str | None = None

    def typing(self) -> _NoopTyping:
        """Slack has no typing indicator; return a no-op so the agent loop's
        ``async with channel.typing():`` works for both platforms."""
        return _NoopTyping()


def composite_channel_key(channel_id: str, thread_ts: str | None) -> str:
    """Queue/session key used by BotState for Slack contexts."""
    return f"{channel_id}:{thread_ts}" if thread_ts else channel_id


# --- Tool input dataclasses ---


@dataclass
class SlackSendUserMessageInput:
    message: Annotated[str, Desc("Text to send to the user on Slack.")]


@dataclass
class SlackSendFileInput:
    file_path: Annotated[str, Desc("Absolute or home-relative path to the file.")]
    message: Annotated[
        str, Desc("Optional message to send with the file (''=no message).")
    ] = ""


@dataclass
class SlackCreateThreadInput:
    topic: Annotated[str, Desc("Topic for the thread (used as the opening message).")]
    message: Annotated[
        str, Desc("Optional first body message ('' means use topic only).")
    ] = ""


@dataclass
class ExploreSlackInput:
    include_channels: Annotated[
        bool, Desc("Include visible public channels in the workspace.")
    ] = True
    include_users: Annotated[bool, Desc("Include a summary of workspace members.")] = (
        False
    )
    channel_limit: Annotated[int, Desc("Max channels to list (1-200).")] = 50
    user_limit: Annotated[int, Desc("Max users to list (1-200).")] = 50


@dataclass
class SlackAPIInput:
    action: Annotated[
        str,
        Desc(
            "'discover' to inspect context, 'request' to call a Slack Web API method."
        ),
    ]
    method: Annotated[
        str,
        Desc(
            "Slack API method name (e.g. 'conversations.list'). Required for 'request'."
        ),
    ] = ""
    body_json: Annotated[
        str,
        Desc("JSON object with method arguments. '{}' means no args."),
    ] = "{}"
    reason: Annotated[str, Desc("Short reason for the call (for audit logs).")] = ""


@dataclass
class SlackRestartBotInput:
    reason: Annotated[str, Desc("Why the bot is being restarted.")] = "user requested"


@dataclass
class SlackDownloadFileInput:
    url: Annotated[
        str,
        Desc(
            "Slack url_private (or url_private_download) for a user-uploaded file. "
            "This is the URL that appears in the [Attachments] section of an "
            "incoming user message."
        ),
    ]
    save_path: Annotated[
        str,
        Desc(
            "Local filesystem path where the file should be written. "
            "Relative paths are resolved against the current working directory; "
            "parent directories are created as needed."
        ),
    ]


# --- Tools ---


@dataclass
class SlackSendUserMessageTool(Tool):
    name: str = "SendUserMessage"
    description: str = (
        "Send a message to the user on Slack. Use this tool whenever you want "
        "the user to see text output. Replies post into the active thread if "
        "the conversation started from a threaded message."
    )
    state: BotState = field(default_factory=BotState, repr=False)
    client: AsyncWebClient | None = field(default=None, repr=False)

    async def __call__(
        self,
        input: SlackSendUserMessageInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        ctx = self.state.active_channel
        channel_id = self.state.active_channel_id
        if (
            not isinstance(ctx, SlackContext)
            or channel_id is None
            or self.client is None
        ):
            return ToolResult(
                content=TextContent(text="Error: No active Slack channel")
            )

        if not input.message.strip():
            return ToolResult(
                content=TextContent(text="Error: message cannot be empty")
            )

        sent = 0
        for chunk in chunk_message(input.message, limit=SLACK_MESSAGE_LIMIT):
            await self.client.chat_postMessage(
                channel=ctx.channel_id,
                text=chunk,
                thread_ts=ctx.thread_ts,
            )
            sent += 1

        if self.state.active_run_stats is not None:
            self.state.active_run_stats["outbound_messages"] = (
                self.state.active_run_stats.get("outbound_messages", 0) + 1
            )
        self.state.append_internal_log(
            channel_id,
            "assistant_message_sent",
            {"chunks": sent, "message_preview": input.message[:300]},
        )
        return ToolResult(content=TextContent(text=f"Sent {sent} chunk(s) to user."))


@dataclass
class SlackSendFileTool(Tool):
    name: str = "SendFile"
    description: str = (
        "Upload a file to the user on Slack. The file is uploaded to the "
        "active channel (and thread, if any). Use this when the user asks you "
        "to share, send, or show a file."
    )
    state: BotState = field(default_factory=BotState, repr=False)
    client: AsyncWebClient | None = field(default=None, repr=False)

    async def __call__(
        self,
        input: SlackSendFileInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        ctx = self.state.active_channel
        if not isinstance(ctx, SlackContext) or self.client is None:
            return ToolResult(
                content=TextContent(text="Error: No active Slack channel")
            )

        path = os.path.expanduser(input.file_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.isfile(path):
            return ToolResult(
                content=TextContent(text=f"Error: File not found: {path}")
            )

        try:
            resp = await self.client.files_upload_v2(
                channel=ctx.channel_id,
                thread_ts=ctx.thread_ts,
                file=path,
                filename=os.path.basename(path),
                initial_comment=input.message or None,
            )
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Error sending file: {e}"))

        size = os.path.getsize(path)
        file_id = (resp.get("file") or {}).get("id") if isinstance(resp, dict) else None
        return ToolResult(
            content=TextContent(
                text=f"File uploaded: {os.path.basename(path)} ({size} bytes) id={file_id}"
            )
        )


@dataclass
class SlackCreateThreadTool(Tool):
    name: str = "CreateThread"
    description: str = (
        "Start a new Slack thread in the current channel with an opening message. "
        "The bot will respond to messages in this thread when users mention it "
        "or reply in the thread."
    )
    state: BotState = field(default_factory=BotState, repr=False)
    client: AsyncWebClient | None = field(default=None, repr=False)

    async def __call__(
        self,
        input: SlackCreateThreadInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        ctx = self.state.active_channel
        if not isinstance(ctx, SlackContext) or self.client is None:
            return ToolResult(
                content=TextContent(text="Error: No active Slack channel")
            )

        # Post a top-level message in the parent channel (not in the current
        # thread) to start a fresh thread.
        try:
            parent = await self.client.chat_postMessage(
                channel=ctx.channel_id,
                text=f"*{input.topic}*",
            )
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Error creating thread: {e}"))

        new_ts = parent.get("ts") if isinstance(parent, dict) else None
        if not new_ts:
            return ToolResult(
                content=TextContent(text=f"Thread creation returned no ts: {parent!r}")
            )

        if input.message:
            try:
                await self.client.chat_postMessage(
                    channel=ctx.channel_id,
                    text=input.message,
                    thread_ts=new_ts,
                )
            except Exception as e:
                return ToolResult(
                    content=TextContent(
                        text=f"Thread opened but failed to post first message: {e}"
                    )
                )

        new_key = composite_channel_key(ctx.channel_id, new_ts)
        self.state.sessions[new_key] = self.state.create_session()
        return ToolResult(
            content=TextContent(
                text=f"Thread started in channel {ctx.channel_id} (thread_ts={new_ts}). "
                "The user can reply in the thread to continue the conversation."
            )
        )


@dataclass
class ExploreSlackTool(Tool):
    name: str = "ExploreSlack"
    description: str = (
        "Explore Slack workspace context: list visible channels, threads, and "
        "users. Use before SlackAPI requests to discover IDs."
    )
    state: BotState = field(default_factory=BotState, repr=False)
    client: AsyncWebClient | None = field(default=None, repr=False)

    async def __call__(
        self,
        input: ExploreSlackInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        if self.client is None:
            return ToolResult(
                content=TextContent(text="Error: Slack client not configured")
            )

        ctx = self.state.active_channel
        active = {
            "channel_id": ctx.channel_id if isinstance(ctx, SlackContext) else None,
            "thread_ts": ctx.thread_ts if isinstance(ctx, SlackContext) else None,
        }

        payload: dict[str, Any] = {"mode": "explore", "active": active}

        channel_limit = max(1, min(int(input.channel_limit), 200))
        user_limit = max(1, min(int(input.user_limit), 200))

        # Fire all three Slack API calls in parallel. ``return_exceptions=True``
        # keeps one failure from cancelling the others.
        tasks: list[Any] = [self.client.auth_test()]
        if input.include_channels:
            tasks.append(
                self.client.conversations_list(
                    limit=channel_limit,
                    types="public_channel,private_channel",
                )
            )
        if input.include_users:
            tasks.append(self.client.users_list(limit=user_limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        results_iter = iter(results)

        auth_result = next(results_iter)
        if isinstance(auth_result, BaseException):
            payload["bot"] = {"error": str(auth_result)}
        else:
            payload["bot"] = {
                "user_id": auth_result.get("user_id"),
                "user": auth_result.get("user"),
                "team_id": auth_result.get("team_id"),
                "team": auth_result.get("team"),
                "url": auth_result.get("url"),
            }

        if input.include_channels:
            channels_result = next(results_iter)
            if isinstance(channels_result, BaseException):
                payload["channels_error"] = str(channels_result)
            else:
                channels = (
                    channels_result.get("channels", [])
                    if isinstance(channels_result, dict)
                    else []
                )
                payload["channels"] = [
                    {
                        "id": c.get("id"),
                        "name": c.get("name"),
                        "is_private": c.get("is_private"),
                        "is_member": c.get("is_member"),
                    }
                    for c in channels[:channel_limit]
                ]

        if input.include_users:
            users_result = next(results_iter)
            if isinstance(users_result, BaseException):
                payload["users_error"] = str(users_result)
            else:
                members = (
                    users_result.get("members", [])
                    if isinstance(users_result, dict)
                    else []
                )
                payload["users"] = [
                    {
                        "id": u.get("id"),
                        "name": u.get("name"),
                        "real_name": u.get("real_name"),
                        "is_bot": u.get("is_bot"),
                    }
                    for u in members[:user_limit]
                ]

        payload["notes"] = [
            "Use this tool to discover IDs before making SlackAPI requests.",
            "For deeper operations, call SlackAPI with action='request'.",
        ]
        text = json.dumps(payload, indent=2)
        if len(text) > 20000:
            text = text[:20000] + "\n... (truncated)"
        return ToolResult(content=TextContent(text=text))


@dataclass
class SlackAPITool(Tool):
    name: str = "SlackAPI"
    description: str = (
        "Call any Slack Web API method with the bot token. "
        "Use action='discover' first to inspect context, then action='request' "
        "with method (e.g. 'conversations.history') and body_json."
    )
    state: BotState = field(default_factory=BotState, repr=False)
    client: AsyncWebClient | None = field(default=None, repr=False)

    async def __call__(
        self,
        input: SlackAPIInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        if self.client is None:
            return ToolResult(
                content=TextContent(text="Error: Slack client not configured")
            )

        action = input.action.strip().lower()
        if action == "discover":
            return await self._discover()
        if action == "request":
            return await self._request(input)
        return ToolResult(
            content=TextContent(text="Error: action must be `discover` or `request`.")
        )

    async def _discover(self) -> ToolResult:
        ctx = self.state.active_channel
        context = {
            "channel_id": ctx.channel_id if isinstance(ctx, SlackContext) else None,
            "thread_ts": ctx.thread_ts if isinstance(ctx, SlackContext) else None,
        }
        try:
            auth = await self.client.auth_test()  # type: ignore[union-attr]
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Slack discover failed: {e}"))

        payload = {
            "mode": "discover",
            "auth": {
                "user_id": auth.get("user_id"),
                "team_id": auth.get("team_id"),
                "team": auth.get("team"),
            },
            "context": context,
            "common_methods": [
                "auth.test",
                "conversations.list",
                "conversations.history",
                "conversations.replies",
                "conversations.info",
                "chat.postMessage",
                "chat.update",
                "chat.delete",
                "users.list",
                "users.info",
                "files.upload",
                "reactions.add",
                "pins.add",
            ],
            "request_mode_usage": {
                "action": "request",
                "method": "conversations.history",
                "body_json": '{"channel": "C…", "limit": 20}',
            },
            "notes": [
                "body_json is passed as kwargs to client.api_call(method, **body).",
                "Slack rate limits apply per method tier.",
            ],
        }
        return ToolResult(content=TextContent(text=json.dumps(payload, indent=2)))

    async def _request(self, input: SlackAPIInput) -> ToolResult:
        method = input.method.strip()
        if not method:
            return ToolResult(
                content=TextContent(text="Error: method is required for 'request'.")
            )
        try:
            body = json.loads(input.body_json) if input.body_json.strip() else {}
        except json.JSONDecodeError as e:
            return ToolResult(
                content=TextContent(text=f"Error: body_json is invalid JSON: {e}")
            )
        if not isinstance(body, dict):
            return ToolResult(
                content=TextContent(
                    text="Error: body_json must decode to a JSON object."
                )
            )

        try:
            resp = await self.client.api_call(method, params=body)  # type: ignore[union-attr]
        except Exception as e:
            return ToolResult(content=TextContent(text=f"Slack request failed: {e}"))

        data: Any = getattr(resp, "data", resp)
        payload = {
            "mode": "request",
            "reason": input.reason,
            "request": {"method": method, "body": body},
            "response": data,
        }
        text = json.dumps(payload, indent=2, default=str)
        if len(text) > 20000:
            text = text[:20000] + "\n... (truncated)"
        return ToolResult(content=TextContent(text=text))


@dataclass
class SlackRestartBotTool(Tool):
    name: str = "RestartBot"
    description: str = (
        "Restart the Slack bot process. Kills the current process and starts "
        "a new one in the background. Use when the user asks to restart or reset."
    )
    state: BotState = field(default_factory=BotState, repr=False)
    client: AsyncWebClient | None = field(default=None, repr=False)

    async def __call__(
        self,
        input: SlackRestartBotInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        ctx = self.state.active_channel
        if isinstance(ctx, SlackContext) and self.client is not None:
            try:
                await self.client.chat_postMessage(
                    channel=ctx.channel_id,
                    text=f"Restarting bot... Reason: {input.reason}",
                    thread_ts=ctx.thread_ts,
                )
            except Exception:
                pass

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_file = os.path.join(project_root, "logs", "slack_bot.nohup.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        restart_cmd = (
            f"cd {project_root} && sleep 2 && "
            f"nohup uv run nano-slack-bot "
            f">> {log_file} 2>&1 &"
        )
        subprocess.Popen(
            restart_cmd,
            shell=True,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"[RESTART] Shutting down for restart. Reason: {input.reason}")
        os._exit(0)
        return ToolResult(content=TextContent(text="Restarting..."))


_SLACK_FILE_HOSTS = frozenset({"files.slack.com", "slack-files.com"})


def _is_slack_file_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and host in _SLACK_FILE_HOSTS


@dataclass
class SlackDownloadFileTool(Tool):
    name: str = "SlackDownloadFile"
    description: str = (
        "Download a user-uploaded file from Slack to the local filesystem. "
        "Slack's url_private URLs require the bot token; use this tool instead "
        "of WebFetch or curl. After downloading, use Read (for text) or other "
        "tools to inspect the content. Requires the files:read scope."
    )
    state: BotState = field(default_factory=BotState, repr=False)
    client: AsyncWebClient | None = field(default=None, repr=False)

    async def __call__(
        self,
        input: SlackDownloadFileInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        if self.client is None:
            return ToolResult(
                content=TextContent(text="Error: Slack client not configured.")
            )
        token = getattr(self.client, "token", None)
        if not token:
            return ToolResult(
                content=TextContent(text="Error: Slack client has no bot token.")
            )
        if not _is_slack_file_url(input.url):
            return ToolResult(
                content=TextContent(
                    text=(
                        f"Error: URL must point to a Slack file host "
                        f"({', '.join(sorted(_SLACK_FILE_HOSTS))}); got {input.url!r}."
                    )
                )
            )

        path = Path(input.save_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path

        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(
                input.url,
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=True,
            )
            if resp.status_code != 200:
                body = resp.text[:300] if resp.text else "(empty)"
                return ToolResult(
                    content=TextContent(
                        text=f"HTTP {resp.status_code} from Slack: {body}"
                    )
                )
            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type and b"<html" in resp.content[:100].lower():
                return ToolResult(
                    content=TextContent(
                        text=(
                            "Slack returned an HTML page instead of the file "
                            "(usually means the bot lacks files:read scope, or "
                            "the URL has expired). Reinstall the app after "
                            "adding files:read to the manifest."
                        )
                    )
                )
            data = resp.content

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        payload = {
            "saved_to": str(path),
            "bytes": len(data),
            "content_type": content_type,
        }
        return ToolResult(content=TextContent(text=json.dumps(payload, indent=2)))


def get_slack_tools(state: BotState, client: AsyncWebClient) -> list[Tool]:
    """Slack tool set: default tools + Slack-specific tools + shared queue tools."""
    tools: list[Tool] = [t for t in get_default_tools() if t.name != "AskUserQuestion"]
    tools.append(SlackSendUserMessageTool(state=state, client=client))
    tools.append(PeekQueuedUserMessagesTool(state=state))
    tools.append(DequeueUserMessagesTool(state=state))
    tools.append(SlackSendFileTool(state=state, client=client))
    tools.append(SlackCreateThreadTool(state=state, client=client))
    tools.append(ExploreSlackTool(state=state, client=client))
    tools.append(SlackAPITool(state=state, client=client))
    tools.append(ClearContextTool(state=state))
    tools.append(SlackRestartBotTool(state=state, client=client))
    tools.append(SlackDownloadFileTool(state=state, client=client))
    tools.append(InspectBotStateTool(state=state))
    return tools
