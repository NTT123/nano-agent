"""Tests for DAG builder class."""

from __future__ import annotations

from pathlib import Path

import pytest

from nano_agent import DAG, Message, Role, TextContent
from nano_agent.data_structures import ToolUseContent
from nano_agent.tools import BashTool, ReadTool


class TestDAGBasics:
    """Test basic DAG creation and builder methods."""

    def test_empty_dag(self) -> None:
        """Test creating an empty DAG."""
        dag = DAG()
        assert len(dag.heads) == 0
        assert dag.to_messages() == []
        assert dag.get_system_prompt() == ""
        assert dag.get_tools() == []

    def test_dag_with_system_prompt(self) -> None:
        """Test DAG with initial system prompt."""
        dag = DAG().system("You are helpful.")
        assert len(dag.heads) == 1
        assert len(dag.head.ancestors()) == 1
        system = dag.get_system_prompt()
        assert "You are helpful" in system

    def test_system_method(self) -> None:
        """Test .system() builder method."""
        dag = DAG()
        dag = dag.system("Test prompt")
        assert len(dag.heads) == 1
        assert "Test prompt" in dag.get_system_prompt()

    def test_tools_method(self) -> None:
        """Test .tools() builder method."""
        dag = DAG().system("Test")
        dag = dag.tools(BashTool(), ReadTool())

        tool_defs = dag.get_tools()
        assert len(tool_defs) == 2
        assert tool_defs[0]["name"] == "Bash"
        assert tool_defs[1]["name"] == "Read"

    def test_user_method(self) -> None:
        """Test .user() builder method."""
        dag = DAG().system("Test")
        dag = dag.user("Hello world")

        messages = dag.to_messages()
        assert len(messages) == 1
        assert messages[0].role == Role.USER
        assert messages[0].content == "Hello world"

    def test_assistant_method(self) -> None:
        """Test .assistant() builder method."""
        dag = DAG().system("Test")
        dag = dag.user("Hello")
        dag = dag.assistant("Hi there!")

        messages = dag.to_messages()
        assert len(messages) == 2
        assert messages[1].role == Role.ASSISTANT
        assert messages[1].content == "Hi there!"

    def test_method_chaining(self) -> None:
        """Test fluent method chaining."""
        dag = DAG().system("Test")
        result = dag.tools(BashTool()).user("Hello").assistant("Hi")

        # Should return new DAG (immutable)
        assert result is not dag
        messages = result.to_messages()
        assert len(messages) == 2


class TestDAGToolExecution:
    """Test tool execution helpers."""

    def test_execute_tools_basic(self) -> None:
        """Test basic tool execution with branching/merging."""
        dag = DAG().system("Test")
        dag = dag.user("Run command")

        # Simulate tool calls
        tool_calls = [
            ToolUseContent(id="call_1", name="Bash", input={"command": "ls"}),
        ]

        def handler(call: ToolUseContent) -> TextContent:
            return TextContent(text=f"Result of {call.name}")

        dag = dag.execute_tools(tool_calls, handler)

        # Should have added assistant message, branched for execution, merged with results
        messages = dag.to_messages()
        assert messages[-2].role == Role.ASSISTANT  # Tool calls
        assert messages[-1].role == Role.USER  # Tool results

    def test_execute_tools_parallel(self) -> None:
        """Test parallel tool execution."""
        dag = DAG().system("Test")
        dag = dag.user("Run multiple commands")

        # Multiple tool calls
        tool_calls = [
            ToolUseContent(id="call_1", name="Bash", input={"command": "ls"}),
            ToolUseContent(id="call_2", name="Read", input={"file": "test.txt"}),
        ]

        results = []

        def handler(call: ToolUseContent) -> TextContent:
            result = f"Result of {call.name}"
            results.append(result)
            return TextContent(text=result)

        dag = dag.execute_tools(tool_calls, handler)

        # Should have executed both tools
        assert len(results) == 2
        assert results[0] == "Result of Bash"
        assert results[1] == "Result of Read"

    def test_execute_tools_empty_list(self) -> None:
        """Test execute_tools with empty list does nothing."""
        dag = DAG().system("Test")
        dag = dag.user("Hello")

        initial_heads = len(dag.heads)
        dag = dag.execute_tools([], lambda x: TextContent(text="result"))

        # Should be no-op
        assert len(dag.heads) == initial_heads


class TestDAGSerialization:
    """Test DAG serialization and deserialization."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading DAG."""
        dag = DAG().system("Test").tools(BashTool()).user("Hello").assistant("Hi")

        # Save
        filepath = tmp_path / "test_graph.json"
        dag.save(filepath)

        # Load
        loaded_dag, metadata = DAG.load(filepath)

        # Verify
        assert len(loaded_dag.heads) == len(dag.heads)
        assert loaded_dag.to_messages() == dag.to_messages()
        assert loaded_dag.get_system_prompt() == dag.get_system_prompt()


class TestDAGEdgeCases:
    """Test edge cases and error handling."""

    def test_head_property_single_head(self) -> None:
        """Test .head property with single head."""
        dag = DAG().system("Test")
        # Should not raise
        head = dag.head
        assert head is not None

    def test_head_property_no_heads_raises(self) -> None:
        """Test .head property with no heads raises error."""
        dag = DAG()
        with pytest.raises(ValueError, match="Expected single head, got 0"):
            _ = dag.head

    def test_heads_property_returns_tuple(self) -> None:
        """Test .heads property returns a tuple."""
        dag = DAG().system("Test")
        heads = dag.heads
        assert isinstance(heads, tuple)
        assert len(heads) == 1

    def test_repr(self) -> None:
        """Test __repr__ method."""
        dag = DAG().system("Test")
        dag = dag.tools(BashTool())
        repr_str = repr(dag)
        assert "DAG" in repr_str
        assert "heads=1" in repr_str
        assert "tools=1" in repr_str

    def test_deep_chain_does_not_recurse(self) -> None:
        """Long linear chains must not blow Python's recursion limit.

        Why: a real Slack session with ~981 sequential nodes hit
        ``RecursionError: maximum recursion depth exceeded`` because
        ``Node.ancestors`` was implemented recursively.
        """
        import sys

        depth = sys.getrecursionlimit() * 3
        dag = DAG().system("Test")
        for i in range(depth):
            dag = dag.user(f"u{i}").assistant(f"a{i}")

        ancestors = dag.head.ancestors()
        assert len(ancestors) == 2 * depth + 1
        messages = dag.to_messages()
        assert len(messages) == 2 * depth


class TestDAGIntegration:
    """Test DAG integration with ClaudeAPI."""

    def test_dag_to_messages_format(self) -> None:
        """Test that DAG produces correct message format for API."""
        dag = DAG().system("Test").tools(BashTool()).user("Hello").assistant("Hi there")

        messages = dag.to_messages()
        assert isinstance(messages, list)
        assert all(isinstance(m, Message) for m in messages)
        assert all(hasattr(m, "role") and hasattr(m, "content") for m in messages)

    def test_dag_get_system_prompt_format(self) -> None:
        """Test that system prompt is correctly formatted."""
        dag = DAG().system("Custom instructions")
        system = dag.get_system_prompt()
        assert isinstance(system, str)
        assert len(system) > 0

    def test_dag_get_tools_format(self) -> None:
        """Test that tools are correctly formatted."""
        dag = DAG().system("Test").tools(BashTool(), ReadTool())
        tools = dag.get_tools()
        assert isinstance(tools, list)
        assert all(isinstance(t, dict) for t in tools)
        assert all("name" in t and "input_schema" in t for t in tools)


class TestImmutability:
    """Test that frozen dataclasses are truly immutable."""

    def test_node_is_frozen(self) -> None:
        """Test that Node attributes cannot be mutated after construction."""
        from nano_agent.dag import Node

        node = Node.system("test")
        with pytest.raises(AttributeError):
            node.id = "hacked"  # type: ignore[misc]

    def test_dag_is_frozen(self) -> None:
        """Test that DAG attributes cannot be mutated after construction."""
        dag = DAG()
        with pytest.raises(AttributeError):
            dag._heads = ()  # type: ignore[misc]
