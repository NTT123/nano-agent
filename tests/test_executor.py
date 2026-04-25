"""Tests for executor agent loop behavior."""

import asyncio
from dataclasses import dataclass
from typing import Annotated, ClassVar

import httpx
from pytest import MonkeyPatch

from nano_agent import DAG, Response, Role, TextContent, ToolUseContent, Usage, run
from nano_agent.data_structures import ToolResultContent
from nano_agent.tools import Desc, Tool


@dataclass
class WaitInput:
    value: Annotated[str, Desc("Value to echo")]


@dataclass
class WaitForSiblingTool(Tool):
    name: str = "WaitForSibling"
    description: str = "Wait until both sibling tool calls have started"

    started: ClassVar[int] = 0
    all_started: ClassVar[asyncio.Event]

    async def __call__(self, input: WaitInput) -> TextContent:
        type(self).started += 1
        if type(self).started == 2:
            type(self).all_started.set()
        await asyncio.wait_for(type(self).all_started.wait(), timeout=0.5)
        return TextContent(text=input.value)


class ParallelToolAPI:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, dag: DAG) -> Response:
        self.calls += 1
        if self.calls == 1:
            return Response(
                id="resp_tools",
                model="test-model",
                role=Role.ASSISTANT,
                content=[
                    ToolUseContent(
                        id="call_1",
                        name="WaitForSibling",
                        input={"value": "first"},
                    ),
                    ToolUseContent(
                        id="call_2",
                        name="WaitForSibling",
                        input={"value": "second"},
                    ),
                ],
                stop_reason="tool_use",
                usage=Usage(input_tokens=1, output_tokens=1),
            )
        return Response(
            id="resp_done",
            model="test-model",
            role=Role.ASSISTANT,
            content=[TextContent(text="done")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


class FlakyStreamAPI:
    """First send() raises a mid-stream peer-close, then succeeds.

    Models the Codex SSE failure: ``peer closed connection without sending
    complete message body`` is raised by httpx as ``RemoteProtocolError``.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def send(self, dag: DAG) -> Response:
        self.calls += 1
        if self.calls == 1:
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body "
                "(incomplete chunked read)"
            )
        return Response(
            id="resp_done",
            model="test-model",
            role=Role.ASSISTANT,
            content=[TextContent(text="done")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_executor_retries_on_remote_protocol_error(
    monkeypatch: MonkeyPatch,
) -> None:
    # Skip the exponential backoff so the test runs instantly.
    monkeypatch.setattr("nano_agent.executor.asyncio.sleep", _no_sleep)

    api = FlakyStreamAPI()
    dag = DAG().user("hi")
    result = await run(api, dag)

    assert api.calls == 2
    messages = result.to_messages()
    assert messages[-1].role == Role.ASSISTANT
    assert isinstance(messages[-1].content, list)
    text_block = messages[-1].content[0]
    assert isinstance(text_block, TextContent)
    assert text_block.text == "done"


async def test_executor_runs_parallel_tool_calls_concurrently() -> None:
    WaitForSiblingTool.started = 0
    WaitForSiblingTool.all_started = asyncio.Event()

    dag = DAG().tools(WaitForSiblingTool()).user("Run both tools")
    result = await run(ParallelToolAPI(), dag)

    tool_result_messages = [
        msg
        for msg in result.to_messages()
        if isinstance(msg.content, list)
        and any(isinstance(block, ToolResultContent) for block in msg.content)
    ]
    assert len(tool_result_messages) == 1
    tool_results = [
        block
        for block in tool_result_messages[0].content
        if isinstance(block, ToolResultContent)
    ]
    assert [block.content[0].text for block in tool_results] == ["first", "second"]
    assert [block.is_error for block in tool_results] == [False, False]


@dataclass
class EchoInput:
    value: Annotated[str, Desc("Value to echo")]


@dataclass
class EchoTool(Tool):
    name: str = "Echo"
    description: str = "Echo the value back as text"

    async def __call__(self, input: EchoInput) -> TextContent:
        return TextContent(text=input.value)


class CompactingAPI:
    """Mock API exercising the auto-compact path.

    First ``send`` returns a tool call with ``total_tokens`` over the configured
    threshold; ``compact_dag`` is expected to be called before the second
    ``send`` (which returns ``end_turn``). Captures the DAG passed to
    ``compact_dag`` so the test can assert it carried the tool result.
    """

    def __init__(self, *, auto_compact_token_limit: int | None) -> None:
        self.auto_compact_token_limit = auto_compact_token_limit
        self.calls = 0
        self.compact_calls = 0
        self.compact_input_messages: list[Role] = []

    async def send(self, dag: DAG) -> Response:
        self.calls += 1
        if self.calls == 1:
            return Response(
                id="resp_tools",
                model="test-model",
                role=Role.ASSISTANT,
                content=[
                    ToolUseContent(
                        id="call_1",
                        name="Echo",
                        input={"value": "hello"},
                    ),
                ],
                stop_reason="tool_use",
                usage=Usage(input_tokens=900, output_tokens=100, total_tokens=1_000),
            )
        return Response(
            id="resp_done",
            model="test-model",
            role=Role.ASSISTANT,
            content=[TextContent(text="done")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20),
        )

    async def compact_dag(self, dag: DAG) -> DAG:
        self.compact_calls += 1
        self.compact_input_messages = [m.role for m in dag.to_messages()]
        # Return a fresh, small DAG that preserves system+tools but drops history.
        new_dag = DAG()
        prompts = dag.head.get_system_prompts() if dag._heads else []
        if prompts:
            new_dag = new_dag.system("\n\n".join(prompts))
        if dag._tools:
            new_dag = new_dag.tools(*dag._tools)
        return new_dag.user("[compacted summary]")


async def test_executor_auto_compacts_when_threshold_crossed() -> None:
    api = CompactingAPI(auto_compact_token_limit=500)
    dag = DAG().tools(EchoTool()).user("Echo hello")

    await run(api, dag)

    assert api.calls == 2
    assert api.compact_calls == 1
    # Compaction must run AFTER tool execution: the DAG handed to compact_dag
    # carries both the assistant's tool call and the user-role tool result.
    assert api.compact_input_messages.count(Role.USER) >= 2
    assert Role.ASSISTANT in api.compact_input_messages


async def test_executor_skips_compact_when_below_threshold() -> None:
    api = CompactingAPI(auto_compact_token_limit=10_000)
    dag = DAG().tools(EchoTool()).user("Echo hello")

    await run(api, dag)

    assert api.calls == 2
    assert api.compact_calls == 0


async def test_executor_skips_compact_when_limit_is_none() -> None:
    api = CompactingAPI(auto_compact_token_limit=None)
    dag = DAG().tools(EchoTool()).user("Echo hello")

    await run(api, dag)

    assert api.calls == 2
    assert api.compact_calls == 0
