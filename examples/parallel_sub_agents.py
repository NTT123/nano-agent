"""Parallel Sub-Agents Example: Two sub-agents executing in parallel.

This example demonstrates how to create multiple SubAgentCapable tools that
can be invoked in parallel by the model in a single response.

Key concepts:
- Fork-join pattern: Model calls multiple tools, each spawns a sub-agent
- Parallel execution: Both sub-agents run concurrently
- Graph visualization: Shows the fork-join DAG structure

Expected Graph Structure:
    TOOL_USE: SecurityAudit + DocSummary (parallel call)
          │
          ├─────────────────────────────┐
          │                             │
          ▼                             ▼
    🔄SecurityAuditor               🔄DocSummarizer
          │                             │
          ▼                             ▼
    ⚡SecurityAudit                 ⚡DocSummary
          │                             │
          └─────────────────────────────┘
                        │
                        ▼
                  USER: ←results

Output:
- parallel_sub_agents_graph.json: Serialized DAG for inspection
- parallel_sub_agents_graph.html: Interactive HTML visualization
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
# Security Audit Tool (spawns a sub-agent)
# =============================================================================


@dataclass
class SecurityAuditInput:
    """Input for the SecurityAudit tool."""

    file_path: Annotated[str, Desc("Path to the file to audit for security issues")]


@dataclass
class SecurityAuditTool(SubAgentTool):
    """A tool that spawns a sub-agent to perform security audits."""

    name: str = "SecurityAudit"
    description: str = "Spawn a sub-agent to audit code for security vulnerabilities"

    async def __call__(
        self,
        input: SecurityAuditInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        if not execution_context:
            return ToolResult(content=TextContent(text="Error: No execution context"))

        system_prompt = """You are an expert security auditor. Your task is to analyze code
for potential security vulnerabilities.

Look for:
- Input validation issues
- Injection vulnerabilities (SQL, command, etc.)
- Authentication/authorization flaws
- Sensitive data exposure
- Insecure dependencies or practices

Use the Read tool to examine the file, then provide a concise security assessment."""

        summary, sub_graph = await self.spawn(
            context=execution_context,
            system_prompt=system_prompt,
            user_message=f"Perform a security audit on: {input.file_path}",
            tools=[ReadTool()],
            tool_name="SecurityAuditor",
        )

        return ToolResult(
            content=TextContent(
                text=f"Security Audit Complete\n"
                f"File: {input.file_path}\n"
                f"---\n"
                f"{summary}"
            ),
            sub_graph=sub_graph,
        )


# =============================================================================
# Documentation Summary Tool (spawns a sub-agent)
# =============================================================================


@dataclass
class DocSummaryInput:
    """Input for the DocSummary tool."""

    file_path: Annotated[str, Desc("Path to the file to summarize documentation for")]


@dataclass
class DocSummaryTool(SubAgentTool):
    """A tool that spawns a sub-agent to summarize documentation."""

    name: str = "DocSummary"
    description: str = "Spawn a sub-agent to summarize code documentation and comments"

    async def __call__(
        self,
        input: DocSummaryInput,
        execution_context: ExecutionContext | None = None,
    ) -> ToolResult:
        if not execution_context:
            return ToolResult(content=TextContent(text="Error: No execution context"))

        system_prompt = """You are an expert technical writer. Your task is to summarize
the documentation and comments in code files.

Provide:
- Overview of what the code does
- Summary of docstrings and comments
- Description of key functions/classes
- Any usage examples found in the code

Use the Read tool to examine the file, then provide a concise documentation summary."""

        summary, sub_graph = await self.spawn(
            context=execution_context,
            system_prompt=system_prompt,
            user_message=f"Summarize the documentation in: {input.file_path}",
            tools=[ReadTool()],
            tool_name="DocSummarizer",
        )

        return ToolResult(
            content=TextContent(
                text=f"Documentation Summary Complete\n"
                f"File: {input.file_path}\n"
                f"---\n"
                f"{summary}"
            ),
            sub_graph=sub_graph,
        )


# =============================================================================
# Main Demo
# =============================================================================


async def main() -> None:
    """Run parallel sub-agents demo and export graph."""
    print("=" * 60)
    print("Parallel Sub-Agents Demo: Fork-Join Pattern")
    print("=" * 60)

    api = ClaudeCodeAPI()
    security_tool = SecurityAuditTool()
    doc_tool = DocSummaryTool()

    # Create DAG with both tools
    dag = DAG(
        system_prompt="""You are a helpful assistant with access to security and documentation tools.

IMPORTANT: When asked to perform multiple analyses, you MUST call all the tools
in a SINGLE response (parallel tool calls). Do NOT call them sequentially."""
    )
    dag = dag.tools(security_tool, doc_tool)
    dag = dag.user(
        "Analyze the file examples/hello_world.py:\n"
        "1. Perform a security audit using the SecurityAudit tool\n"
        "2. Summarize the documentation using the DocSummary tool\n\n"
        "Call BOTH tools in your response."
    )

    print("\nRunning agent loop (expecting parallel tool calls)...")
    dag = await run(api, dag)
    print("Agent loop complete (SubGraphs auto-captured by executor)")

    # Export paths
    output_dir = Path(__file__).parent
    json_path = output_dir / "parallel_sub_agents_graph.json"
    html_path = output_dir / "parallel_sub_agents_graph.html"

    # Save JSON
    dag.save(json_path, session_id="parallel_sub_agents_demo")
    print(f"\nJSON exported to: {json_path}")

    # Generate HTML using the existing viewer
    view_file(str(json_path), str(html_path))

    # Print DAG visualization
    print("\n" + "=" * 60)
    print("DAG Text Visualization:")
    print("=" * 60)
    print(dag)

    print("\n" + "=" * 60)
    print("Verification Checklist:")
    print("=" * 60)
    print("1. Open the HTML file in a browser")
    print("2. Verify: Graph shows fork after TOOL_USE (two parallel branches)")
    print("3. Verify: Each branch has SubGraph → ToolExecution")
    print("4. Verify: Branches merge into single RESULT node")
    print("5. Click each SubGraph node → 'Zoom Into Sub-Agent' button works")
    print("6. Test breadcrumb navigation for both sub-agents")
    print(f"\nOpen: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
