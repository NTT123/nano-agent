"""Recursive Sub-Agents Example: Multi-level sub-agent nesting.

This example demonstrates **recursive sub-agent calling** where a sub-agent
spawns its own sub-agents, creating a multi-level hierarchy.

Key concepts:
- Nested sub-agents: Tools that spawn sub-agents can themselves be given to sub-agents
- Multi-level depth: Depth increments at each nesting level (d=1, d=2, etc.)
- Breadcrumb navigation: HTML visualization shows the nesting path

Expected Graph Structure:
    USER: "Analyze project"
        │
        ▼
    TOOL_USE: ProjectAnalyzer
        │
        ▼
    🔄ProjectAnalyzer[d=1]
        │
        ├── TOOL_USE: SecurityAudit + DocSummary (Level 1 agent calls 2 tools)
        │       │
        │       ├─────────────────────────────┐
        │       │                             │
        │       ▼                             ▼
        │   🔄SecurityAuditor[d=2]       🔄DocSummarizer[d=2]
        │       │                             │
        │       ▼                             ▼
        │   ⚡SecurityAudit              ⚡DocSummary
        │       │                             │
        │       └─────────────────────────────┘
        │                   │
        ▼                   ▼
    ⚡ProjectAnalyzer (result from sub-agents)
        │
        ▼
    ASSISTANT: Final analysis

Output:
- recursive_sub_agents_graph.json: Serialized DAG for inspection
- recursive_sub_agents_graph.html: Interactive HTML visualization
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from nano_agent import (
    DAG,
    ClaudeCodeAPI,
    ReadTool,
    SubAgentTool,
    TextContent,
    run,
)
from nano_agent.tools.base import Desc

# Import the existing viewer for HTML generation
from scripts.viewer import view_file

# =============================================================================
# Level 2 Tools: SecurityAuditTool and DocSummaryTool (spawn depth=2 sub-agents)
# =============================================================================


@dataclass
class SecurityAuditInput:
    """Input for the SecurityAudit tool."""

    file_path: Annotated[str, Desc("Path to the file to audit for security issues")]


@dataclass
class SecurityAuditTool(SubAgentTool):
    """A tool that spawns a sub-agent to perform security audits.

    When called from within a sub-agent (e.g., ProjectAnalyzer's sub-agent),
    this will create a nested sub-agent at depth=2.
    """

    name: str = "SecurityAudit"
    description: str = "Spawn a sub-agent to audit code for security vulnerabilities"

    async def __call__(self, input: SecurityAuditInput) -> TextContent:
        system_prompt = """You are an expert security auditor. Your task is to analyze code
for potential security vulnerabilities.

Look for:
- Input validation issues
- Injection vulnerabilities (SQL, command, etc.)
- Authentication/authorization flaws
- Sensitive data exposure
- Insecure dependencies or practices

Use the Read tool to examine the file, then provide a concise security assessment."""

        summary = await self.spawn(
            system_prompt=system_prompt,
            user_message=f"Perform a security audit on: {input.file_path}",
            tools=[ReadTool()],
            tool_name="SecurityAuditor",
        )

        return TextContent(
            text=f"Security Audit Complete\n"
            f"File: {input.file_path}\n"
            f"---\n"
            f"{summary}"
        )


@dataclass
class DocSummaryInput:
    """Input for the DocSummary tool."""

    file_path: Annotated[str, Desc("Path to the file to summarize documentation for")]


@dataclass
class DocSummaryTool(SubAgentTool):
    """A tool that spawns a sub-agent to summarize documentation.

    When called from within a sub-agent (e.g., ProjectAnalyzer's sub-agent),
    this will create a nested sub-agent at depth=2.
    """

    name: str = "DocSummary"
    description: str = "Spawn a sub-agent to summarize code documentation and comments"

    async def __call__(self, input: DocSummaryInput) -> TextContent:
        system_prompt = """You are an expert technical writer. Your task is to summarize
the documentation and comments in code files.

Provide:
- Overview of what the code does
- Summary of docstrings and comments
- Description of key functions/classes
- Any usage examples found in the code

Use the Read tool to examine the file, then provide a concise documentation summary."""

        summary = await self.spawn(
            system_prompt=system_prompt,
            user_message=f"Summarize the documentation in: {input.file_path}",
            tools=[ReadTool()],
            tool_name="DocSummarizer",
        )

        return TextContent(
            text=f"Documentation Summary Complete\n"
            f"File: {input.file_path}\n"
            f"---\n"
            f"{summary}"
        )


# =============================================================================
# Level 1 Tool: ProjectAnalyzerTool (spawns depth=1 sub-agent with nested tools)
# =============================================================================


@dataclass
class ProjectAnalyzerInput:
    """Input for the ProjectAnalyzer tool."""

    file_path: Annotated[str, Desc("Path to the file to analyze")]


@dataclass
class ProjectAnalyzerTool(SubAgentTool):
    """A tool that spawns a sub-agent which can spawn its own sub-agents.

    The ProjectAnalyzer creates a sub-agent at depth=1 that has access to
    SecurityAuditTool and DocSummaryTool. When that sub-agent uses those tools,
    they spawn their own sub-agents at depth=2, creating a recursive structure.
    """

    name: str = "ProjectAnalyzer"
    description: str = (
        "Spawn a sub-agent to perform comprehensive project analysis using "
        "security auditing and documentation summarization tools"
    )

    async def __call__(self, input: ProjectAnalyzerInput) -> TextContent:
        system_prompt = """You are a project analyst that coordinates comprehensive code analysis.

You have access to two specialized tools:
1. SecurityAudit - For security vulnerability analysis
2. DocSummary - For documentation and code structure analysis

IMPORTANT: When asked to analyze a file, you MUST use BOTH tools in a SINGLE response
(parallel tool calls). Do NOT call them sequentially.

After receiving results from both tools, provide a brief combined analysis."""

        # Note: SecurityAuditTool and DocSummaryTool are SubAgentTool!
        # When the sub-agent calls them, they will spawn their own sub-agents.
        summary = await self.spawn(
            system_prompt=system_prompt,
            user_message=f"Analyze {input.file_path}. Use BOTH the SecurityAudit and DocSummary tools.",
            tools=[SecurityAuditTool(), DocSummaryTool()],
            tool_name="ProjectAnalyzer",
        )

        return TextContent(
            text=f"Project Analysis Complete\n"
            f"File: {input.file_path}\n"
            f"---\n"
            f"{summary}"
        )


# =============================================================================
# Main Demo
# =============================================================================


async def main() -> None:
    """Run recursive sub-agents demo and export graph."""
    print("=" * 60)
    print("Recursive Sub-Agents Demo: Multi-Level Nesting")
    print("=" * 60)

    api = ClaudeCodeAPI()
    analyzer_tool = ProjectAnalyzerTool()

    # Create DAG with the top-level tool
    dag = DAG(
        system_prompt="""You are a helpful assistant with a powerful project analysis tool.

When asked to analyze a project or file, use the ProjectAnalyzer tool.
This tool will spawn specialized sub-agents to perform comprehensive analysis."""
    )
    dag = dag.tools(analyzer_tool)
    dag = dag.user(
        "Analyze the file examples/hello_world.py using the ProjectAnalyzer tool."
    )

    print("\nRunning agent loop (expecting recursive sub-agent calls)...")
    print("Expected structure:")
    print("  Main → ProjectAnalyzer[d=1] → SecurityAuditor[d=2] + DocSummarizer[d=2]")
    dag = await run(api, dag)
    print("Agent loop complete (SubGraphs auto-captured by executor)")

    # Export paths
    output_dir = Path(__file__).parent
    json_path = output_dir / "recursive_sub_agents_graph.json"
    html_path = output_dir / "recursive_sub_agents_graph.html"

    # Save JSON
    dag.save(json_path, session_id="recursive_sub_agents_demo")
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
    print("2. Verify: Main graph shows ProjectAnalyzer tool call")
    print("3. Click 'Zoom Into Sub-Agent' → see ProjectAnalyzer[d=1] graph")
    print("4. Inside d=1: See SecurityAudit + DocSummary tool calls (parallel)")
    print(
        "5. Click SecurityAuditor → breadcrumb shows: Main › ProjectAnalyzer › SecurityAuditor"
    )
    print(
        "6. Click DocSummarizer → breadcrumb shows: Main › ProjectAnalyzer › DocSummarizer"
    )
    print("7. Navigate back using breadcrumbs at each level")
    print("8. Depth labels show [d=1] and [d=2] correctly")
    print(f"\nOpen: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
