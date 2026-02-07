"""
DAG: Node-based conversation graph with fluent builder API.

Everything is a node: system prompts, tool definitions, messages, etc.
Nodes form a DAG - arrows indicate forking (1→N) and merging (N→1).

This module contains:
- Node: The fundamental unit of agent conversation graphs
- DAG: Fluent builder for constructing Node-based conversation graphs

Usage:
    from nano_agent import Node, DAG

    # Low-level: Build with Node directly
    root = Node.system("You are helpful.")
    root = root.tools(BashTool())
    n1 = root.child(Message(Role.USER, "Hello"))

    # High-level: Build with DAG fluent API
    dag = DAG().system("You are helpful.").tools(BashTool()).user("Hello")
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data_structures import (
    ContentBlock,
    Message,
    NodeData,
    Response,
    Role,
    StopReason,
    SubGraph,
    SystemPrompt,
    TextContent,
    ThinkingContent,
    ToolDefinition,
    ToolDefinitionDict,
    ToolDefinitions,
    ToolExecution,
    ToolResultContent,
    ToolUseContent,
    parse_message_content,
    parse_stop_reason,
    parse_sub_graph,
    parse_tool_definitions,
    parse_tool_execution,
)
from .tools import Tool


def _generate_id() -> str:
    """Generate a unique node ID."""
    return uuid.uuid4().hex[:12]


# =============================================================================
# Node: The fundamental unit of agent conversation graphs
# =============================================================================


@dataclass(frozen=True)
class Node:
    """A single node in the conversation graph.

    Everything is a node: system prompts, tool definitions, messages, etc.
    Nodes form a DAG - arrows indicate forking (1→N) and merging (N→1).

    Examples:
        # Start with system prompt and tools (they're nodes too)
        root = Node.system("You are helpful.")
        root = root.tools(BashTool(), ReadTool())

        # Add conversation
        n1 = root.child(Message(Role.USER, "Hello"))
        n2 = n1.child(Message(Role.ASSISTANT, "Hi!"))

        # Get everything for API call
        messages = n2.to_messages()
        system = n2.get_system_prompt()
        tools = n2.get_tools()

        # Merge multiple branches (e.g., tool results)
        merged = Node.with_parents([branch1, branch2], message)
    """

    parents: tuple["Node", ...]
    data: NodeData

    id: str = field(default_factory=_generate_id)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Node creation
    # -------------------------------------------------------------------------

    @classmethod
    def system(cls, prompt: str) -> "Node":
        """Create a system prompt node as root."""
        return cls(
            parents=(),
            data=SystemPrompt(content=prompt),
        )

    def tools(self, *tools: "Tool") -> "Node":
        """Add tool definitions as a child node.

        Args:
            *tools: Tool objects to add

        Returns:
            New Node with tool definitions
        """
        tool_defs = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in tools
        ]
        return Node(
            parents=(self,),
            data=ToolDefinitions(tools=tool_defs),
        )

    @classmethod
    def root(
        cls,
        data: NodeData,
        metadata: dict[str, Any] | None = None,
    ) -> "Node":
        """Create a root node (no parents)."""
        return cls(
            parents=(),
            data=data,
            metadata=metadata or {},
        )

    def child(
        self,
        data: NodeData,
        metadata: dict[str, Any] | None = None,
    ) -> "Node":
        """Create a child node."""
        return Node(
            parents=(self,),
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def with_parents(
        cls,
        parents: list["Node"] | tuple["Node", ...],
        data: NodeData,
        metadata: dict[str, Any] | None = None,
    ) -> "Node":
        """Create a node with multiple parents (for merging branches)."""
        return cls(
            parents=tuple(parents) if isinstance(parents, list) else parents,
            data=data,
            metadata=metadata or {},
        )

    # -------------------------------------------------------------------------
    # Graph traversal
    # -------------------------------------------------------------------------

    def ancestors(self) -> list["Node"]:
        """Get ancestors in causal order (parents before children)."""
        result: list[Node] = []
        visited: set[str] = set()

        def visit(node: Node) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            for parent in node.parents:
                visit(parent)
            result.append(node)

        visit(self)
        return result

    def to_messages(self) -> list[Message]:
        """Get Message nodes."""
        return [n.data for n in self.ancestors() if isinstance(n.data, Message)]

    def get_system_prompt(self) -> str:
        """Get all system prompts from ancestors, concatenated."""
        return "\n\n".join(self.get_system_prompts())

    def get_system_prompts(self) -> list[str]:
        """Get all system prompts from ancestors as a list."""
        prompts = []
        for node in self.ancestors():
            if isinstance(node.data, SystemPrompt):
                prompts.append(node.data.content)
        return prompts

    def get_tools(self) -> list[ToolDefinitionDict]:
        """Get tool definitions from ancestors."""
        for node in self.ancestors():
            if isinstance(node.data, ToolDefinitions):
                return [t.to_dict() for t in node.data.tools]
        return []

    @property
    def is_root(self) -> bool:
        return len(self.parents) == 0

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data_dict = self.data.to_dict() if hasattr(self.data, "to_dict") else self.data
        return {
            "id": self.id,
            "parent_ids": [p.id for p in self.parents],
            "data": data_dict,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any], node_map: dict[str, "Node"]) -> "Node":
        node_data_raw = data.get("data", {})
        node_data: NodeData | None = None

        if isinstance(node_data_raw, dict):
            node_type = node_data_raw.get("type", "")

            # Reconstruct based on type
            if node_type == "system_prompt":
                content = node_data_raw.get("content")
                if isinstance(content, str):
                    node_data = SystemPrompt(content=content)
            elif node_type == "tool_definitions":
                node_data = parse_tool_definitions(node_data_raw)
            elif node_type == "tool_execution":
                node_data = parse_tool_execution(node_data_raw)
            elif node_type == "stop_reason":
                node_data = parse_stop_reason(node_data_raw)
            elif node_type == "sub_graph":
                node_data = parse_sub_graph(node_data_raw)

            if node_data is None and "role" in node_data_raw:
                # It's a Message
                role = Role(node_data_raw.get("role", "user"))
                content_raw = node_data_raw.get("content", "")
                node_data = Message(
                    role=role,
                    content=parse_message_content(content_raw),
                )
            elif node_data is None:
                # Unknown dict type - create a minimal SystemPrompt as fallback
                node_data = SystemPrompt(content=str(node_data_raw))
        else:
            # Fallback for non-dict data
            node_data = SystemPrompt(content=str(node_data_raw))

        parents_tuple = tuple(
            node_map[pid] for pid in data.get("parent_ids", []) if pid in node_map
        )

        node = cls(
            parents=parents_tuple,
            data=node_data,
            metadata=data.get("metadata", {}),
        )
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(node, "id", data.get("id", node.id))
        object.__setattr__(node, "timestamp", data.get("timestamp", node.timestamp))
        return node

    @staticmethod
    def save_graph(
        heads: list["Node"] | "Node",
        filepath: str | Path,
        session_id: str | None = None,
        **metadata: Any,
    ) -> None:
        """Save graph to JSON."""
        if isinstance(heads, Node):
            heads = [heads]

        all_nodes: dict[str, Node] = {}
        for head in heads:
            for node in head.ancestors():
                all_nodes[node.id] = node

        output = {
            "session_id": session_id or f"session_{uuid.uuid4().hex[:8]}",
            "created_at": time.time(),
            "metadata": metadata,
            "head_ids": [h.id for h in heads],
            "nodes": {nid: node.to_dict() for nid, node in all_nodes.items()},
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

    @staticmethod
    def load_graph(filepath: str | Path) -> tuple[list["Node"], dict[str, Any]]:
        """Load graph from JSON."""
        with open(filepath) as f:
            data = json.load(f)

        nodes_data = data.get("nodes", {})
        head_ids = data.get("head_ids", [])

        node_map: dict[str, Node] = {}
        remaining = dict(nodes_data)

        while remaining:
            ready = [
                nid
                for nid, ndata in remaining.items()
                if all(pid in node_map for pid in ndata.get("parent_ids", []))
            ]
            if not ready:
                raise ValueError("Cycle or missing parents")
            for nid in ready:
                node_map[nid] = Node._from_dict(remaining.pop(nid), node_map)

        heads = [node_map[hid] for hid in head_ids if hid in node_map]
        if not heads and node_map:
            has_children = {p.id for n in node_map.values() for p in n.parents}
            heads = [n for n in node_map.values() if n.id not in has_children]

        meta = {
            "session_id": data.get("session_id", ""),
            "created_at": data.get("created_at"),
            **data.get("metadata", {}),
        }
        return heads, meta

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_usage_totals(self) -> dict[str, int]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        for node in self.ancestors():
            for key in totals:
                totals[key] += node.metadata.get("usage", {}).get(key, 0)
        return totals

    def __repr__(self) -> str:
        data_type = type(self.data).__name__
        return f"Node(id={self.id!r}, data={data_type}, parents={len(self.parents)})"


# =============================================================================
# DAG: Fluent builder for Node-based conversation graphs
# =============================================================================


@dataclass(frozen=True)
class DAG:
    """Immutable fluent builder for Node-based conversation graphs.

    The DAG class provides a high-level interface for constructing conversation
    graphs while maintaining the Node-based architecture internally.

    All builder methods return NEW DAG instances, leaving the original unchanged.

    Examples:
        # Basic usage (chaining creates new instances)
        dag = DAG(system_prompt="You are helpful.")
        dag = dag.tools(BashTool(), ReadTool()).user("What is 2+2?")
        response = api.send(dag)

        # Tool execution (returns new DAG)
        dag = dag.execute_tools(response.get_tool_use(), handle_tool)

        # Agent loop (using executor.run)
        from nano_agent import run
        dag = await run(api, dag)  # Runs until stop reason
    """

    _heads: tuple[Node, ...] = field(default_factory=tuple)
    _tools: tuple[Tool, ...] | None = None

    def __init__(
        self,
        system_prompt: str | None = None,
        _heads: tuple[Node, ...] | None = None,
        _tools: tuple[Tool, ...] | None = None,
    ):
        """Create a new DAG, optionally starting with a system prompt.

        Args:
            system_prompt: Optional system prompt to start with
            _heads: Internal - tuple of head nodes
            _tools: Internal - tuple of tools
        """
        if _heads is not None:
            # Internal constructor for creating modified copies
            object.__setattr__(self, "_heads", _heads)
            object.__setattr__(self, "_tools", _tools)
        elif system_prompt:
            # User constructor with system prompt
            node = Node.system(system_prompt)
            object.__setattr__(self, "_heads", (node,))
            object.__setattr__(self, "_tools", None)
        else:
            # Empty DAG
            object.__setattr__(self, "_heads", ())
            object.__setattr__(self, "_tools", None)

    def _with_heads(
        self,
        heads: tuple[Node, ...] | list[Node],
        tools: Sequence[Tool] | None = None,
    ) -> DAG:
        """Internal: Create new DAG with updated heads.

        Args:
            heads: New head nodes
            tools: Optional new tools (or keep existing if None)

        Returns:
            New DAG instance
        """
        heads_tuple = tuple(heads) if isinstance(heads, list) else heads
        tools_tuple = tuple(tools) if tools is not None else self._tools
        return DAG(
            _heads=heads_tuple,
            _tools=tools_tuple,
        )

    def append_to(
        self,
        data: NodeData,
        *,
        heads: Sequence[Node],
    ) -> DAG:
        """Append data to a specific subset of heads.

        Args:
            data: NodeData to append
            heads: Subset of current heads to append to

        Returns:
            New DAG instance with only the selected heads advanced

        Raises:
            ValueError: If heads is empty or contains nodes not in current heads
        """
        if not heads:
            raise ValueError("heads must be a non-empty subset of current heads")
        if not self._heads:
            raise ValueError("Cannot append_to on an empty DAG")

        head_ids = {h.id for h in self._heads}
        selected_ids = {h.id for h in heads}
        if not selected_ids.issubset(head_ids):
            raise ValueError("heads must be a subset of current heads")

        new_heads: list[Node] = []
        for head in self._heads:
            if head.id in selected_ids:
                new_heads.append(head.child(data))
            else:
                new_heads.append(head)

        return self._with_heads(tuple(new_heads))

    # -------------------------------------------------------------------------
    # Core builder methods
    # -------------------------------------------------------------------------

    def system(self, prompt: str) -> DAG:
        """Add system prompt node(s).

        Note: If this is the first call on an empty DAG, it will include
        the Claude Code identity automatically (via Node.system).

        Returns:
            New DAG instance with system prompt added
        """
        if not self._heads:
            # First system prompt - use Node.system (includes identity)
            node = Node.system(prompt)
            return self._with_heads((node,))
        else:
            # Additional system prompt - just append
            return self._with_heads(self._append_to_heads(SystemPrompt(content=prompt)))

    def tools(self, *tools: Tool) -> DAG:
        """Add tool definitions.

        Args:
            *tools: Tool objects to add

        Returns:
            New DAG instance with tools added
        """
        tool_defs = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in tools
        ]
        data = ToolDefinitions(tools=tool_defs)
        if not self._heads:
            node = Node.root(data)
            return self._with_heads((node,), tools=tools)
        new_dag = self.append_to(data, heads=self._heads)
        return new_dag._with_heads(new_dag._heads, tools=tools)

    def user(
        self, content: str | ContentBlock | Sequence[ContentBlock], *more: ContentBlock
    ) -> DAG:
        """Add user message.

        Args:
            content: First content item (string, content block, or sequence of blocks)
            *more: Additional content blocks (only when first arg is a single block)

        Returns:
            New DAG instance with user message added

        Examples:
            dag.user("Hello")                    # String
            dag.user(block1, block2)             # Varargs
            dag.user(response.content)           # List (backward compat)
        """
        if isinstance(content, str):
            if more:
                raise ValueError("Cannot mix string with ContentBlock arguments")
            data = Message(Role.USER, content)
            if not self._heads:
                return self._with_heads((Node.root(data),))
            return self.append_to(data, heads=self._heads)
        elif isinstance(content, Sequence) and not isinstance(content, str):
            # content is a sequence of blocks (backward compat)
            if more:
                raise ValueError(
                    "Cannot mix sequence with additional ContentBlock arguments"
                )
            blocks: list[ContentBlock] = []
            for block in content:
                if isinstance(
                    block,
                    (TextContent, ThinkingContent, ToolUseContent, ToolResultContent),
                ):
                    blocks.append(block)
            data = Message(Role.USER, blocks)
            if not self._heads:
                return self._with_heads((Node.root(data),))
            return self.append_to(data, heads=self._heads)
        else:
            # content is a single ContentBlock
            if not isinstance(
                content,
                (TextContent, ThinkingContent, ToolUseContent, ToolResultContent),
            ):
                raise TypeError("Invalid content type for user message")
            block = content
            all_content: list[ContentBlock] = [block, *more]
            data = Message(Role.USER, all_content)
            if not self._heads:
                return self._with_heads((Node.root(data),))
            return self.append_to(data, heads=self._heads)

    def assistant(
        self, content: str | ContentBlock | Sequence[ContentBlock], *more: ContentBlock
    ) -> DAG:
        """Add assistant message.

        Args:
            content: First content item (string, content block, or sequence of blocks)
            *more: Additional content blocks (only when first arg is a single block)

        Returns:
            New DAG instance with assistant message added

        Examples:
            dag.assistant("Hi there")            # String
            dag.assistant(block1, block2)        # Varargs
            dag.assistant(response.content)      # List (backward compat)
        """
        if isinstance(content, str):
            if more:
                raise ValueError("Cannot mix string with ContentBlock arguments")
            data = Message(Role.ASSISTANT, content)
            if not self._heads:
                return self._with_heads((Node.root(data),))
            return self.append_to(data, heads=self._heads)
        elif isinstance(content, Sequence) and not isinstance(content, str):
            # content is a sequence of blocks (backward compat)
            if more:
                raise ValueError(
                    "Cannot mix sequence with additional ContentBlock arguments"
                )
            blocks: list[ContentBlock] = []
            for block in content:
                if isinstance(
                    block,
                    (TextContent, ThinkingContent, ToolUseContent, ToolResultContent),
                ):
                    blocks.append(block)
            data = Message(Role.ASSISTANT, blocks)
            if not self._heads:
                return self._with_heads((Node.root(data),))
            return self.append_to(data, heads=self._heads)
        else:
            # content is a single ContentBlock
            if not isinstance(
                content,
                (TextContent, ThinkingContent, ToolUseContent, ToolResultContent),
            ):
                raise TypeError("Invalid content type for assistant message")
            block = content
            all_content: list[ContentBlock] = [block, *more]
            data = Message(Role.ASSISTANT, all_content)
            if not self._heads:
                return self._with_heads((Node.root(data),))
            return self.append_to(data, heads=self._heads)

    def tool_result(self, *tool_results: ToolResultContent) -> DAG:
        """Add tool results.

        Args:
            *tool_results: Tool result content blocks

        Returns:
            New DAG instance with tool results added
        """
        data = Message(Role.USER, list(tool_results))
        if not self._heads:
            return self._with_heads((Node.root(data),))
        return self.append_to(data, heads=self._heads)

    def sub_graph(self, sub_graph: SubGraph) -> DAG:
        """Add a sub-graph node (encapsulated sub-agent execution).

        Args:
            sub_graph: SubGraph containing the sub-agent's execution

        Returns:
            New DAG instance with sub-graph added
        """
        if not self._heads:
            return self._with_heads((Node.root(sub_graph),))
        return self.append_to(sub_graph, heads=self._heads)

    # -------------------------------------------------------------------------
    # Tool execution helpers (the key innovation)
    # -------------------------------------------------------------------------

    def execute_tools(
        self,
        tool_calls: list[ToolUseContent],
        handler: Callable[[ToolUseContent], TextContent | list[TextContent]],
    ) -> DAG:
        """High-level helper: execute tools and update graph automatically.

        This is the "magic" method that replaces 30+ lines of boilerplate.

        Args:
            tool_calls: List of ToolUseContent from API response
            handler: Function that takes ToolUseContent and returns ContentBlock(s)

        Returns:
            New DAG instance with tool execution results

        Example:
            def handle_tool(tool_call: ToolUseContent) -> TextContent:
                if tool_call.name == "Bash":
                    return bash_tool(tool_call.input)
                ...

            dag = dag.execute_tools(response.get_tool_use(), handle_tool)
        """
        if not tool_calls:
            return self

        # Begin: add assistant message with tool calls
        dag_with_calls = self.assistant(list(tool_calls))

        # Save current head before branching
        tool_use_head = dag_with_calls.head

        # Execute and branch for each tool
        result_nodes = []
        tool_results = []
        for tool_call in tool_calls:
            result = handler(tool_call)
            # Normalize to list
            result_list = result if isinstance(result, list) else [result]

            # Create branch node with ToolExecution (visualization only)
            result_node = tool_use_head.child(
                ToolExecution(
                    tool_name=tool_call.name,
                    tool_use_id=tool_call.id,
                    result=result_list,
                )
            )
            result_nodes.append(result_node)

            # Collect tool results for API
            tool_results.append(
                ToolResultContent(
                    tool_use_id=tool_call.id,
                    content=result_list,
                )
            )

        # End: merge all branches with results
        merged = Node.with_parents(
            result_nodes,
            Message(Role.USER, tool_results),
        )
        return dag_with_calls._with_heads((merged,))

    def add_response(
        self,
        response: Response,
        tool_handler: (
            Callable[[ToolUseContent], TextContent | list[TextContent]] | None
        ) = None,
    ) -> DAG:
        """Add API response to graph, with optional automatic tool handling.

        This method intelligently processes the response:
        - Extracts text/thinking blocks -> adds as assistant message
        - If tool_handler provided and response has tool calls:
          - Automatically executes tools via execute_tools()
        - Otherwise, adds tool calls as assistant message for manual handling

        Args:
            response: API response
            tool_handler: Optional function to execute tools automatically

        Returns:
            New DAG instance with response added
        """
        # Separate text/thinking from tool calls
        text_blocks: list[ContentBlock] = [
            b for b in response.content if isinstance(b, (TextContent, ThinkingContent))
        ]
        tool_calls = [b for b in response.content if isinstance(b, ToolUseContent)]

        # Start with current DAG
        dag = self

        # Add text response if any
        if text_blocks:
            dag = dag.assistant(text_blocks)

        # Handle tool calls
        if tool_calls:
            if tool_handler:
                # Automatic tool execution
                dag = dag.execute_tools(tool_calls, tool_handler)
            else:
                # Manual - just add the tool calls for user to handle
                # Cast needed because list is invariant
                tool_call_blocks: list[ContentBlock] = list(tool_calls)
                dag = dag.assistant(tool_call_blocks)

        return dag

    def add_stop_reason(self, response: Response) -> DAG:
        """Add stop reason node (for visualization).

        Args:
            response: API response with stop_reason

        Returns:
            New DAG instance with stop reason added
        """
        usage_totals = self.head.get_usage_totals()
        data = StopReason(
            reason=response.stop_reason or "unknown",
            usage=usage_totals,
        )
        if not self._heads:
            return self._with_heads((Node.root(data),))
        return self.append_to(data, heads=self._heads)

    # -------------------------------------------------------------------------
    # Properties and accessors
    # -------------------------------------------------------------------------

    @property
    def head(self) -> Node:
        """Get the single head node.

        Raises:
            ValueError: If there are multiple heads (use .heads instead)
        """
        if len(self._heads) != 1:
            raise ValueError(
                f"Expected single head, got {len(self._heads)}. "
                "Use .heads property instead."
            )
        return self._heads[0]

    @property
    def heads(self) -> tuple[Node, ...]:
        """Get all current head nodes (as tuple for immutability)."""
        return self._heads

    def to_messages(self) -> list[Message]:
        """Extract messages for API call.

        Returns:
            List of Message objects
        """
        if not self._heads:
            return []
        return self.head.to_messages()

    def get_system_prompt(self) -> str:
        """Get concatenated system prompt.

        Returns:
            Concatenated system prompt string
        """
        if not self._heads:
            return ""
        return self.head.get_system_prompt()

    def get_tools(self) -> list[ToolDefinitionDict]:
        """Get tool definitions.

        Returns:
            List of tool definition dicts
        """
        if not self._heads:
            return []
        return self.head.get_tools()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def save(
        self,
        filepath: str | Path,
        session_id: str | None = None,
        **metadata: Any,
    ) -> None:
        """Save graph to JSON file.

        Args:
            filepath: Path to save JSON file
            session_id: Optional session ID
            **metadata: Additional metadata to save
        """
        Node.save_graph(list(self._heads), filepath, session_id, **metadata)

    @classmethod
    def load(cls, filepath: str | Path) -> tuple[DAG, dict[str, Any]]:
        """Load graph from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Tuple of (DAG instance, metadata dict)
        """
        heads, metadata = Node.load_graph(filepath)
        dag = cls(_heads=tuple(heads))
        return dag, metadata

    # -------------------------------------------------------------------------
    # SubGraph conversion
    # -------------------------------------------------------------------------

    def to_sub_graph(
        self,
        tool_name: str,
        tool_use_id: str,
        summary: str = "",
        depth: int = 0,
    ) -> SubGraph:
        """Create SubGraph from this DAG instance.

        Args:
            tool_name: Name of the tool that spawned this sub-agent
            tool_use_id: ID of the tool call
            summary: Summary of the sub-agent's results
            depth: Nesting depth of this sub-agent

        Returns:
            SubGraph containing serialized DAG
        """
        all_nodes: dict[str, object] = {}
        head_ids: list[str] = []

        if self._heads:
            for head in self._heads:
                head_ids.append(head.id)
                for node in head.ancestors():
                    all_nodes[node.id] = node.to_dict()

        return SubGraph(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            system_prompt=self.get_system_prompt() if self._heads else "",
            nodes=all_nodes,
            head_ids=head_ids,
            summary=summary,
            depth=depth,
        )

    @classmethod
    def from_sub_graph(cls, sub_graph: SubGraph) -> "DAG":
        """Reconstruct DAG from a SubGraph.

        Args:
            sub_graph: SubGraph containing serialized nodes

        Returns:
            DAG instance with reconstructed nodes
        """
        if not sub_graph.nodes:
            return cls()

        node_map: dict[str, Node] = {}
        remaining = dict(sub_graph.nodes)

        while remaining:
            ready = [
                nid
                for nid, ndata in remaining.items()
                if isinstance(ndata, dict)
                and all(pid in node_map for pid in ndata.get("parent_ids", []))
            ]
            if not ready:
                raise ValueError("Cycle or missing parents in SubGraph")
            for nid in ready:
                ndata = remaining.pop(nid)
                if isinstance(ndata, dict):
                    node_map[nid] = Node._from_dict(ndata, node_map)

        heads = [node_map[hid] for hid in sub_graph.head_ids if hid in node_map]
        if not heads and node_map:
            has_children = {p.id for n in node_map.values() for p in n.parents}
            heads = [n for n in node_map.values() if n.id not in has_children]

        return cls(_heads=tuple(heads))

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _append_to_heads(self, data: NodeData) -> tuple[Node, ...]:
        """Helper to append data to all current heads.

        Args:
            data: NodeData to append

        Returns:
            New tuple of head nodes with data appended
        """
        if not self._heads:
            # No heads yet - create root
            node = Node.root(data)
            return (node,)
        # Delegate to append_to for consistent head selection logic
        return self.append_to(data, heads=self._heads).heads

    def __repr__(self) -> str:
        """Return string representation of DAG."""
        return (
            f"DAG(heads={len(self._heads)}, "
            f"tools={len(self._tools) if self._tools else 0})"
        )

    def __str__(self) -> str:
        """Return console visualization of the DAG."""
        if not self._heads:
            return "DAG(empty)"

        lines: list[str] = []

        # Get all nodes in order
        all_nodes = self.head.ancestors()

        # Build children map
        children_map: dict[str, list[Node]] = {n.id: [] for n in all_nodes}
        for node in all_nodes:
            for parent in node.parents:
                if parent.id in children_map:
                    children_map[parent.id].append(node)

        for i, node in enumerate(all_nodes):
            type_label, content = self._format_node(node)

            # Check for fan-in (multiple parents)
            is_fanin = len(node.parents) > 1

            # Check for fan-out (multiple children)
            children = children_map.get(node.id, [])
            is_fanout = len(children) > 1
            is_last = i == len(all_nodes) - 1

            if is_fanin:
                lines.append("  └────────┬─────────┘")
                lines.append("           │")

            lines.append(f"  {type_label}: {content}")

            if not is_last and not is_fanout:
                lines.append("      │")
                lines.append("      ▼")

        return "\n".join(lines)

    @staticmethod
    def _format_node(node: Node) -> tuple[str, str]:
        """Format a node for display. Returns (type_label, content)."""
        data = node.data

        def truncate(text: str, max_len: int = 40) -> str:
            text = text.replace("\n", " ").strip()
            return text[: max_len - 3] + "..." if len(text) > max_len else text

        if isinstance(data, SystemPrompt):
            return "SYSTEM", truncate(data.content)

        elif isinstance(data, ToolDefinitions):
            tool_names = [t.name for t in data.tools]
            return "TOOLS", ", ".join(tool_names)

        elif isinstance(data, ToolExecution):
            result_text = data.result[0].text if data.result else ""
            return "EXEC", f"{data.tool_name} -> {truncate(result_text, 30)}"

        elif isinstance(data, StopReason):
            return "STOP", data.reason

        elif isinstance(data, SubGraph):
            depth_str = f"[d={data.depth}]" if data.depth > 0 else ""
            summary = truncate(data.summary, 30) if data.summary else "(no summary)"
            return "SUBGRAPH", f"{data.tool_name}{depth_str} -> {summary}"

        elif isinstance(data, Message):
            role = data.role.value.upper()
            content = data.content

            if isinstance(content, str):
                return role, truncate(content)
            elif isinstance(content, list):
                if not content:
                    return role, "(empty)"

                # Check content block types
                tool_uses = [c for c in content if isinstance(c, ToolUseContent)]
                tool_results = [c for c in content if isinstance(c, ToolResultContent)]
                text_blocks = [c for c in content if isinstance(c, TextContent)]
                thinking_blocks = [c for c in content if isinstance(c, ThinkingContent)]

                if tool_results:
                    # Show first result content
                    first_content = tool_results[0].content
                    if first_content:
                        return "RESULT", truncate(first_content[0].text)
                    return "RESULT", "(empty)"
                elif tool_uses:
                    tools = [t.name for t in tool_uses]
                    return "TOOL_USE", " + ".join(tools)
                elif text_blocks:
                    return role, truncate(text_blocks[0].text)
                elif thinking_blocks:
                    return role, "(thinking)"
                return role, f"({len(content)} blocks)"

        return "UNKNOWN", str(data)[:40]
