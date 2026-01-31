"""Sub-Agent Example: Tools that spawn their own agent loops.

This example demonstrates how to create tools that can spawn sub-agents,
giving them access to the LLM API and their own set of tools.

Key concepts:
- SubAgentTool: Base class for tools that spawn sub-agents (pure functional)
- ExecutionContext: Provides api, dag, cancel_token (passed as parameter)
- spawn(): Helper method to spawn a sub-agent and capture results
- ToolResult: Pure functional return value with content and optional sub_graph

Output:
- sub_agent_graph.json: Serialized DAG for inspection
- sub_agent_graph.html: Interactive HTML visualization (via nano-agent-viewer)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from nano_agent import (
    DAG,
    ClaudeCodeAPI,
    ExecutionContext,
    ReadTool,
    SubAgentTool,
    TextContent,
    run,
)
from nano_agent.tools.base import Desc, ToolResult

# Import the existing viewer for HTML generation
from scripts.viewer import view_file

# =============================================================================
# Code Review Tool (spawns a sub-agent)
# =============================================================================


@dataclass
class CodeReviewInput:
    """Input for the CodeReview tool."""

    file_path: Annotated[str, Desc("Path to the file to review")]
    focus_areas: Annotated[
        str, Desc("Specific areas to focus on (e.g., 'security', 'performance')")
    ] = "general"


@dataclass
class CodeReviewTool(SubAgentTool):
    """A tool that spawns a sub-agent to review code (pure functional)."""

    name: str = "CodeReview"
    description: str = "Spawn a code review sub-agent to analyze a file"

    async def __call__(
        self,
        input: CodeReviewInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        if not execution_context:
            return ToolResult(content=TextContent(text="Error: No execution context"))

        system_prompt = f"""You are an expert code reviewer. Your task is to review code
with a focus on: {input.focus_areas}

Provide constructive feedback on:
- Code quality and readability
- Potential bugs or issues
- Best practices
- Suggestions for improvement

Be concise but thorough. Use the Read tool to examine the file."""

        summary, sub_graph = await self.spawn(
            context=execution_context,
            system_prompt=system_prompt,
            user_message=f"Please review the file: {input.file_path}",
            tools=[ReadTool()],
            tool_name="CodeReviewer",
        )

        return ToolResult(
            content=TextContent(
                text=f"Code Review Complete\n"
                f"File: {input.file_path}\n"
                f"Focus: {input.focus_areas}\n"
                f"---\n"
                f"{summary}"
            ),
            sub_graph=sub_graph,
        )


# =============================================================================
# Main Demo
# =============================================================================


async def main() -> None:
    """Run code review demo and export graph."""
    print("=" * 60)
    print("Sub-Agent Demo: Code Review with Graph Export")
    print("=" * 60)

    api = ClaudeCodeAPI()
    code_review_tool = CodeReviewTool()

    # Create DAG with CodeReviewTool
    dag = DAG(
        system_prompt="You are a helpful assistant with access to code review tools."
    )
    dag = dag.tools(code_review_tool)
    dag = dag.user("Please review the file examples/hello_world.py for code quality.")

    print("\nRunning agent loop...")
    dag = await run(api, dag)
    # SubGraph is automatically captured by the executor alongside ToolExecution
    print("Agent loop complete (SubGraph auto-captured by executor)")

    # Export paths
    output_dir = Path(__file__).parent
    json_path = output_dir / "sub_agent_graph.json"
    html_path = output_dir / "sub_agent_graph.html"

    # Save JSON
    dag.save(json_path, session_id="sub_agent_demo")
    print(f"\nJSON exported to: {json_path}")

    # Generate HTML using the existing viewer
    view_file(str(json_path), str(html_path))

    # Print DAG visualization
    print("\n" + "=" * 60)
    print("DAG Text Visualization:")
    print("=" * 60)
    print(dag)

    print("\n" + "=" * 60)
    print(f"Open {html_path} in a browser to view the interactive graph!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
