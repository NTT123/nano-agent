"""Self-inspection tool for the bot — exposes ``BotState`` internals read-only.

Shared by the Discord and Slack frontends. Only touches ``BotState`` (no
platform clients), so a single implementation serves both bots.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Annotated, Any

from nano_agent import ExecutionContext, TextContent
from nano_agent.tools.base import Desc, Tool, ToolResult

from .bot_state import BotState

_MAX_REPR_LEN = 4000


@dataclass
class InspectBotStateInput:
    field: Annotated[
        str,
        Desc(
            "BotState field name to inspect (e.g. 'sessions', 'channel_worker_tasks'). "
            "Empty string returns a top-level overview of all fields."
        ),
    ] = ""
    key: Annotated[
        str,
        Desc(
            "If the chosen field is a dict, the key to drill into (e.g. a "
            "channel id like 'C123:1777.275209'). If a list, the integer index "
            "as a string. Empty means no drill-down."
        ),
    ] = ""
    max_items: Annotated[
        int,
        Desc("Maximum dict/list items to include in summaries (1-200)."),
    ] = 20


def _type_name(obj: Any) -> str:
    return type(obj).__name__


def _truncate_repr(obj: Any, limit: int = _MAX_REPR_LEN) -> str:
    try:
        text = repr(obj)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(text) > limit:
        return text[:limit] + f"... [truncated, {len(text)} chars total]"
    return text


def _summarize_task(task: Any) -> str:
    done = getattr(task, "done", lambda: None)()
    if done is False:
        return "running"
    if done is True:
        if getattr(task, "cancelled", lambda: False)():
            return "cancelled"
        exc = None
        try:
            exc = task.exception()
        except Exception:
            exc = None
        return f"done(error={type(exc).__name__})" if exc else "done"
    return _type_name(task)


def _summarize_value(value: Any, max_items: int) -> dict[str, Any]:
    if isinstance(value, dict):
        keys = list(value.keys())
        sample = keys[:max_items]
        summary: dict[str, Any] = {
            "type": f"dict[{len(keys)}]",
            "keys_preview": [str(k) for k in sample],
        }
        if keys and all(
            hasattr(v, "done") and callable(v.done) for v in value.values()
        ):
            statuses = [_summarize_task(v) for v in value.values()]
            summary["task_statuses"] = {
                "running": sum(s == "running" for s in statuses),
                "done": sum(s.startswith("done") for s in statuses),
                "cancelled": sum(s == "cancelled" for s in statuses),
            }
        return summary
    if isinstance(value, (list, tuple, set, frozenset)):
        return {
            "type": f"{_type_name(value)}[{len(value)}]",
            "preview": _truncate_repr(list(value)[:max_items], 800),
        }
    return {"type": _type_name(value), "repr": _truncate_repr(value, 800)}


def _overview(state: BotState, max_items: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(state):
        try:
            value = getattr(state, f.name)
        except Exception as exc:
            out[f.name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if value is None:
            out[f.name] = {"type": "None"}
        elif isinstance(value, (str, int, float, bool)):
            out[f.name] = {"type": _type_name(value), "value": value}
        else:
            out[f.name] = _summarize_value(value, max_items)
    return out


def _drill(value: Any, key: str, max_items: int) -> dict[str, Any]:
    if isinstance(value, dict):
        if key in value:
            item = value[key]
        else:
            return {
                "error": f"Key {key!r} not found",
                "available_keys": [str(k) for k in list(value.keys())[:max_items]],
            }
    elif isinstance(value, (list, tuple)):
        try:
            idx = int(key)
        except ValueError:
            return {"error": f"Index must be an integer, got {key!r}"}
        if idx < -len(value) or idx >= len(value):
            return {"error": f"Index {idx} out of range for length {len(value)}"}
        item = value[idx]
    else:
        return {"error": f"Cannot drill into {_type_name(value)} with key {key!r}"}
    return {
        "type": _type_name(item),
        "repr": _truncate_repr(item),
    }


@dataclass
class InspectBotStateTool(Tool):
    name: str = "InspectBotState"
    description: str = (
        "Inspect the bot's own BotState (sessions, queues, worker tasks, "
        "working dirs, etc.) via read-only reflection. "
        "Call with no arguments for an overview of all fields. "
        "Pass `field` to dump a single field's summary, plus `key` to drill "
        "into a dict entry or list index. Output is truncated; use this to "
        "debug bot behavior, not to exfiltrate large state."
    )
    state: BotState = field(default_factory=BotState, repr=False)

    async def __call__(
        self,
        input: InspectBotStateInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        max_items = max(1, min(input.max_items, 200))

        if not input.field:
            return _json_result(_overview(self.state, max_items))

        field_names = {f.name for f in dataclasses.fields(self.state)}
        if input.field not in field_names:
            return _json_result(
                {
                    "error": f"Unknown field {input.field!r}",
                    "available_fields": sorted(field_names),
                }
            )

        value = getattr(self.state, input.field)
        if input.key:
            return _json_result(_drill(value, input.key, max_items))
        return _json_result(_summarize_value(value, max_items))


def _json_result(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        content=TextContent(text=json.dumps(payload, indent=2, default=str))
    )
