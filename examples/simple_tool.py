"""Simple Tool Example: Custom tool with async execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nano_agent import (
    DAG,
    ClaudeAPI,
    TextContent,
    Tool,
    ToolResultContent,
)

if TYPE_CHECKING:
    from nano_agent import ExecutionContext


@dataclass
class CalculatorInput:
    expr: str


@dataclass
class Calculator(Tool):
    name: str = "calculator"
    description: str = "Evaluate a math expression"

    async def __call__(
        self,
        input: CalculatorInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        return TextContent(text=str(eval(input.expr)))  # noqa: S307


async def main() -> None:
    calc = Calculator()
    api = ClaudeAPI()
    dag = DAG().tools(calc).user("What is 23 * 47?")

    response = await api.send(dag)
    dag = dag.assistant(response.content)

    for call in response.get_tool_use():
        tool_result = await calc.execute(call.input)
        content = tool_result.content
        content_list = content if isinstance(content, list) else [content]
        dag = dag.tool_result(
            ToolResultContent(tool_use_id=call.id, content=content_list)
        )

    # Get final answer
    response = await api.send(dag)
    dag = dag.assistant(response.content)
    print(dag)


if __name__ == "__main__":
    asyncio.run(main())
