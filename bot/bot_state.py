"""Bot state management, queue operations, session persistence, and pure helpers."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord

from nano_agent import DAG, Role, TextContent, ThinkingContent
from nano_agent.data_structures import Message, ToolUseContent


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def chunk_message(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks respecting Discord's message limit.

    Tries to split at line boundaries for readability.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def is_empty_assistant_message(message: Message) -> bool:
    """Whether a message is an assistant message with empty content."""
    if message.role != Role.ASSISTANT:
        return False
    if isinstance(message.content, str):
        return message.content.strip() == ""
    return len(message.content) == 0


def serialize_content_blocks(content: list[Any]) -> list[dict[str, Any]]:
    """Serialize response content blocks for internal logs."""
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            blocks.append({"type": "thinking", "thinking": block.thinking})
        elif isinstance(block, ToolUseContent):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        else:
            blocks.append({"type": type(block).__name__, "raw": str(block)})
    return blocks


def serialize_text_contents(content: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in content:
        if isinstance(item, TextContent):
            texts.append(item.text)
        else:
            texts.append(str(item))
    return texts


def truncate(text: str, limit: int = 200) -> str:
    """Truncate text to *limit* characters, appending '...' if trimmed."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@dataclass
class BotState:
    bot: discord.Client | None = None
    sessions: dict[int, DAG] = field(default_factory=dict)
    working_dirs: dict[int, str] = field(default_factory=dict)
    active_channel: discord.abc.Messageable | None = None
    active_channel_id: int | None = None
    clear_context_requested: set[int] = field(default_factory=set)
    channel_message_queues: dict[int, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    channel_queue_seq: dict[int, int] = field(default_factory=dict)
    channel_last_user_id: dict[int, int] = field(default_factory=dict)
    channel_worker_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict)
    active_run_stats: dict[str, int] | None = None
    agent_runtime_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state_root: Path = Path("logs/discord_agent_state")

    # -- path helpers --

    def _channel_state_dir(self, channel_id: int) -> Path:
        return self.state_root / f"channel_{channel_id}"

    def _queue_file(self, channel_id: int) -> Path:
        return self._channel_state_dir(channel_id) / "queue.json"

    def _internal_log_file(self, channel_id: int) -> Path:
        return self._channel_state_dir(channel_id) / "internal_events.jsonl"

    def _session_file(self, channel_id: int) -> Path:
        return self._channel_state_dir(channel_id) / "session.json"

    def _ensure_state_dir(self, channel_id: int) -> None:
        self._channel_state_dir(channel_id).mkdir(parents=True, exist_ok=True)

    # -- session management --

    def get_session(
        self,
        channel_id: int,
        cwd: str | None = None,
        system_prompt: str = "",
        tools: list[Any] | None = None,
    ) -> DAG:
        if channel_id in self.sessions:
            return self.sessions[channel_id]
        loaded = self._load_session(channel_id, tools=tools)
        if loaded is not None:
            self.sessions[channel_id] = loaded
            return loaded
        self.sessions[channel_id] = self.create_session(
            cwd, system_prompt=system_prompt, tools=tools
        )
        return self.sessions[channel_id]

    def set_session(self, channel_id: int, dag: DAG) -> None:
        self.sessions[channel_id] = dag
        self._save_session(channel_id, dag)

    def create_session(
        self,
        cwd: str | None = None,
        system_prompt: str = "",
        tools: list[Any] | None = None,
    ) -> DAG:
        """Create a fresh conversation session.

        ``tools`` must be provided by the caller to avoid circular imports.
        """
        prompt = system_prompt
        if cwd:
            prompt += f"\n\nThe user's working directory is: {cwd}"
        dag = DAG().system(prompt or "<empty>")
        if tools:
            dag = dag.tools(*tools)
        return dag

    # -- queue operations --

    def get_channel_queue(self, channel_id: int) -> list[dict[str, Any]]:
        if channel_id not in self.channel_message_queues:
            self.channel_message_queues[channel_id] = self._load_channel_queue(
                channel_id
            )
        return self.channel_message_queues[channel_id]

    def _next_queue_id(self, channel_id: int) -> int:
        self.channel_queue_seq[channel_id] = (
            self.channel_queue_seq.get(channel_id, 0) + 1
        )
        return self.channel_queue_seq[channel_id]

    def enqueue_user_message(
        self, channel_id: int, message: discord.Message, content: str
    ) -> dict[str, Any]:
        """Queue an inbound Discord user message for deferred consumption."""
        queue = self.get_channel_queue(channel_id)
        queued: dict[str, Any] = {
            "queue_id": self._next_queue_id(channel_id),
            "message_id": message.id,
            "author_id": message.author.id,
            "author": str(message.author),
            "content": content,
            "created_at": utc_now_iso(),
            "attachments": [a.url for a in message.attachments],
        }
        queue.append(queued)
        self._save_channel_queue(channel_id)
        self.append_internal_log(channel_id, "user_message_enqueued", queued)
        return queued

    def peek_user_messages(
        self, channel_id: int, limit: int = 5
    ) -> list[dict[str, Any]]:
        queue = self.get_channel_queue(channel_id)
        return [dict(item) for item in queue[: max(1, min(limit, 50))]]

    def dequeue_user_messages(
        self, channel_id: int, count: int = 1
    ) -> list[dict[str, Any]]:
        queue = self.get_channel_queue(channel_id)
        take = max(1, min(count, 20))
        items = [queue.pop(0) for _ in range(min(take, len(queue)))]
        self._save_channel_queue(channel_id)
        if items:
            self.append_internal_log(
                channel_id,
                "user_messages_dequeued",
                {
                    "count": len(items),
                    "queue_ids": [i.get("queue_id") for i in items],
                },
            )
        return items

    def clear_user_queue(self, channel_id: int) -> None:
        self.channel_message_queues[channel_id] = []
        self._save_channel_queue(channel_id)
        self.append_internal_log(channel_id, "user_queue_cleared", {})

    # -- notifications --

    def _format_queue_preview(
        self, channel_id: int, limit: int = 5, max_content_len: int = 140
    ) -> list[str]:
        """Shared preview rendering for queue notifications."""
        queue = self.get_channel_queue(channel_id)
        lines: list[str] = []
        for item in queue[:limit]:
            qid = item.get("queue_id")
            author = item.get("author", "unknown")
            content = str(item.get("content", "")).replace("\n", " ")
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."
            lines.append(f"- #{qid} from {author}: {content}")
        return lines

    def build_queue_runtime_note(self, channel_id: int) -> str:
        """Build per-turn runtime note so agent can choose when to consume messages."""
        queue = self.get_channel_queue(channel_id)
        if not queue:
            return (
                "Runtime queue status: no queued user messages. "
                "If you need to communicate with the user, call SendUserMessage."
            )

        preview_text = "\n".join(
            self._format_queue_preview(channel_id, limit=5, max_content_len=140)
        )
        return (
            "Runtime queue status: there are queued user messages waiting.\n"
            f"Pending count: {len(queue)}.\n"
            "Use PeekQueuedUserMessages to inspect details without consuming.\n"
            "Use DequeueUserMessages to consume messages when you are ready.\n"
            "You may continue current work and leave messages queued.\n"
            "Queued previews:\n"
            f"{preview_text}"
        )

    def format_queue_notification_for_dag(
        self, channel_id: int, limit: int = 10
    ) -> str:
        """Render queued-message notification as a synthetic user message."""
        queue = self.get_channel_queue(channel_id)
        bounded = max(1, min(limit, 20))
        lines = [
            f"Runtime notification: there are {len(queue)} queued user messages pending.",
            "Queued message preview:",
        ]
        lines.extend(
            self._format_queue_preview(channel_id, limit=bounded, max_content_len=160)
        )
        lines.append(
            "Use DequeueUserMessages to consume messages, or PeekQueuedUserMessages for details."
        )
        return "\n".join(lines)

    # -- DAG sanitization --

    def sanitize_dag_for_api(self, dag: DAG, channel_id: int) -> DAG:
        """Drop invalid empty assistant messages from history before API calls."""
        messages = dag.to_messages()
        filtered = [m for m in messages if not is_empty_assistant_message(m)]
        removed = len(messages) - len(filtered)
        if removed == 0:
            return dag

        rebuilt = DAG()
        prompts = dag.head.get_system_prompts() if dag.heads else []
        for prompt in prompts:
            rebuilt = rebuilt.system(prompt)
        if dag._tools:
            rebuilt = rebuilt.tools(*dag._tools)

        for msg in filtered:
            if msg.role == Role.USER:
                rebuilt = rebuilt.user(msg.content)
            else:
                rebuilt = rebuilt.assistant(msg.content)

        self.append_internal_log(
            channel_id,
            "dag_sanitized",
            {
                "removed_empty_assistant_messages": removed,
                "original_message_count": len(messages),
                "message_count_after": len(filtered),
            },
        )
        return rebuilt

    # -- session persistence --

    def _save_session(self, channel_id: int, dag: DAG) -> None:
        try:
            self._ensure_state_dir(channel_id)
            dag.save(self._session_file(channel_id), session_id=f"channel_{channel_id}")
        except Exception:
            pass  # In-memory state is source of truth

    def _load_session(
        self, channel_id: int, tools: list[Any] | None = None
    ) -> DAG | None:
        path = self._session_file(channel_id)
        if not path.exists():
            return None
        try:
            loaded_dag, _meta = DAG.load(path)
            if not loaded_dag._heads:
                return None
            if tools:
                loaded_dag = loaded_dag._with_heads(loaded_dag._heads, tools=tools)
            return loaded_dag
        except Exception:
            return None

    def delete_session_file(self, channel_id: int) -> None:
        try:
            path = self._session_file(channel_id)
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def delete_all_session_files(self) -> None:
        for channel_id in self.persisted_channel_ids():
            self.delete_session_file(channel_id)

    # -- queue persistence --

    def _save_channel_queue(self, channel_id: int) -> None:
        self._ensure_state_dir(channel_id)
        queue = self.channel_message_queues.get(channel_id, [])
        with self._queue_file(channel_id).open("w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)

    def _load_channel_queue(self, channel_id: int) -> list[dict[str, Any]]:
        qfile = self._queue_file(channel_id)
        if not qfile.exists():
            return []
        try:
            raw = json.loads(qfile.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        queue: list[dict[str, Any]] = [item for item in raw if isinstance(item, dict)]
        max_id = 0
        for item in queue:
            msg_id = item.get("queue_id")
            if isinstance(msg_id, int):
                max_id = max(max_id, msg_id)
        self.channel_queue_seq[channel_id] = max(
            self.channel_queue_seq.get(channel_id, 0), max_id
        )
        return queue

    def persisted_channel_ids(self) -> list[int]:
        """Return channel IDs discovered from persisted queue directories."""
        if not self.state_root.exists():
            return []
        ids: list[int] = []
        for entry in self.state_root.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith("channel_"):
                continue
            raw = name.removeprefix("channel_")
            if raw.isdigit():
                ids.append(int(raw))
        return ids

    # -- logging --

    def append_internal_log(
        self, channel_id: int, event: str, payload: dict[str, Any]
    ) -> None:
        """Append an internal event for this channel to persistent jsonl logs."""
        self._ensure_state_dir(channel_id)
        entry = {
            "ts": utc_now_iso(),
            "event": event,
            "payload": payload,
        }
        with self._internal_log_file(channel_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
