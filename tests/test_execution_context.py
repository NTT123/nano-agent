"""Tests for ExecutionContext and sub-agent support."""

import pytest

from nano_agent import (
    DAG,
    ExecutionContext,
    SubAgentCapable,
    SubGraph,
    TextContent,
    Tool,
)
from nano_agent.data_structures import parse_sub_graph


class MockAPI:
    """Mock API for testing."""

    async def send(self, dag):
        pass


class TestExecutionContext:
    """Tests for ExecutionContext dataclass."""

    def test_create_context(self):
        """Test basic context creation."""
        api = MockAPI()
        dag = DAG(system_prompt="Test")

        ctx = ExecutionContext(api=api, dag=dag)

        assert ctx.api is api
        assert ctx.dag is dag
        assert ctx.cancel_token is None
        assert ctx.permission_callback is None
        assert ctx.depth == 0
        assert ctx.max_depth == 5

    def test_child_context_increments_depth(self):
        """Test that child_context increments depth."""
        api = MockAPI()
        dag = DAG(system_prompt="Test")
        ctx = ExecutionContext(api=api, dag=dag, depth=0)

        child_dag = DAG(system_prompt="Child")
        child_ctx = ctx.child_context(child_dag)

        assert child_ctx.depth == 1
        assert child_ctx.dag is child_dag
        assert child_ctx.api is api

    def test_child_context_raises_on_max_depth(self):
        """Test that child_context raises RecursionError at max depth."""
        api = MockAPI()
        dag = DAG(system_prompt="Test")
        ctx = ExecutionContext(api=api, dag=dag, depth=5, max_depth=5)

        with pytest.raises(RecursionError, match="Sub-agent depth limit exceeded"):
            ctx.child_context(dag)

    def test_custom_max_depth(self):
        """Test custom max_depth setting."""
        api = MockAPI()
        dag = DAG(system_prompt="Test")
        ctx = ExecutionContext(api=api, dag=dag, max_depth=3)

        # Can create children up to max_depth
        child1 = ctx.child_context(dag)
        assert child1.depth == 1

        child2 = child1.child_context(dag)
        assert child2.depth == 2

        child3 = child2.child_context(dag)
        assert child3.depth == 3

        # Fails at max_depth
        with pytest.raises(RecursionError):
            child3.child_context(dag)


class TestSubGraph:
    """Tests for SubGraph dataclass."""

    def test_create_subgraph(self):
        """Test basic SubGraph creation."""
        sg = SubGraph(
            tool_name="TestTool",
            tool_use_id="test_123",
            system_prompt="Test prompt",
            nodes={},
            head_ids=[],
            summary="Test summary",
            depth=1,
        )

        assert sg.tool_name == "TestTool"
        assert sg.tool_use_id == "test_123"
        assert sg.system_prompt == "Test prompt"
        assert sg.summary == "Test summary"
        assert sg.depth == 1

    def test_subgraph_to_dict(self):
        """Test SubGraph.to_dict serialization."""
        sg = SubGraph(
            tool_name="TestTool",
            tool_use_id="test_123",
            system_prompt="Test prompt",
            nodes={"node1": {"id": "node1"}},
            head_ids=["node1"],
            summary="Test summary",
            depth=2,
        )

        d = sg.to_dict()

        assert d["type"] == "sub_graph"
        assert d["tool_name"] == "TestTool"
        assert d["tool_use_id"] == "test_123"
        assert d["system_prompt"] == "Test prompt"
        assert d["nodes"] == {"node1": {"id": "node1"}}
        assert d["head_ids"] == ["node1"]
        assert d["summary"] == "Test summary"
        assert d["depth"] == 2

    def test_subgraph_from_dict(self):
        """Test SubGraph.from_dict deserialization."""
        data = {
            "type": "sub_graph",
            "tool_name": "TestTool",
            "tool_use_id": "test_123",
            "system_prompt": "Test prompt",
            "nodes": {"node1": {"id": "node1"}},
            "head_ids": ["node1"],
            "summary": "Test summary",
            "depth": 2,
        }

        sg = SubGraph.from_dict(data)

        assert sg.tool_name == "TestTool"
        assert sg.tool_use_id == "test_123"
        assert sg.system_prompt == "Test prompt"
        assert sg.nodes == {"node1": {"id": "node1"}}
        assert sg.head_ids == ["node1"]
        assert sg.summary == "Test summary"
        assert sg.depth == 2

    def test_subgraph_from_dag(self):
        """Test SubGraph.from_dag creates SubGraph from DAG."""
        dag = DAG(system_prompt="Test system prompt")
        dag = dag.user("Hello")
        dag = dag.assistant("Hi there!")

        sg = SubGraph.from_dag(
            dag=dag,
            tool_name="TestAgent",
            tool_use_id="agent_123",
            summary="Completed task",
            depth=1,
        )

        assert sg.tool_name == "TestAgent"
        assert sg.tool_use_id == "agent_123"
        assert sg.summary == "Completed task"
        assert sg.depth == 1
        assert "Test system prompt" in sg.system_prompt
        assert len(sg.nodes) > 0
        assert len(sg.head_ids) > 0

    def test_subgraph_to_dag_roundtrip(self):
        """Test SubGraph -> DAG -> SubGraph roundtrip."""
        original_dag = DAG(system_prompt="Test system prompt")
        original_dag = original_dag.user("User message")
        original_dag = original_dag.assistant("Assistant response")

        sg = SubGraph.from_dag(
            dag=original_dag,
            tool_name="TestAgent",
            tool_use_id="agent_123",
        )

        # Convert back to DAG
        restored_dag = sg.to_dag()

        # Verify messages are preserved
        messages = restored_dag.to_messages()
        assert len(messages) == 2
        assert messages[0].content == "User message"
        assert messages[1].content == "Assistant response"

    def test_subgraph_validation(self):
        """Test SubGraph validation."""
        with pytest.raises(ValueError, match="tool_name cannot be empty"):
            SubGraph(
                tool_name="",
                tool_use_id="test_123",
                system_prompt="",
                nodes={},
                head_ids=[],
            )

        with pytest.raises(ValueError, match="tool_use_id cannot be empty"):
            SubGraph(
                tool_name="TestTool",
                tool_use_id="",
                system_prompt="",
                nodes={},
                head_ids=[],
            )


class TestParseSubGraph:
    """Tests for parse_sub_graph function."""

    def test_parse_valid_subgraph(self):
        """Test parsing a valid SubGraph dict."""
        data = {
            "type": "sub_graph",
            "tool_name": "TestTool",
            "tool_use_id": "test_123",
            "system_prompt": "Test prompt",
            "nodes": {},
            "head_ids": [],
        }

        sg = parse_sub_graph(data)

        assert sg is not None
        assert sg.tool_name == "TestTool"
        assert sg.tool_use_id == "test_123"

    def test_parse_missing_tool_name(self):
        """Test parsing with missing tool_name returns None."""
        data = {
            "type": "sub_graph",
            "tool_use_id": "test_123",
        }

        sg = parse_sub_graph(data)
        assert sg is None

    def test_parse_non_dict(self):
        """Test parsing non-dict returns None."""
        assert parse_sub_graph("not a dict") is None
        assert parse_sub_graph(None) is None
        assert parse_sub_graph([]) is None


class TestSubAgentCapable:
    """Tests for SubAgentCapable protocol."""

    def test_protocol_check(self):
        """Test that SubAgentCapable is a runtime checkable protocol."""
        from dataclasses import dataclass, field

        @dataclass
        class MockSubAgentTool(Tool, SubAgentCapable):
            name: str = "MockSubAgent"
            description: str = "A mock sub-agent tool"
            _execution_context: ExecutionContext | None = field(
                default=None, repr=False
            )

            def set_execution_context(self, ctx):
                object.__setattr__(self, "_execution_context", ctx)

            async def __call__(self, input=None):
                return TextContent(text="result")

        tool = MockSubAgentTool()
        assert isinstance(tool, SubAgentCapable)

    def test_non_subagent_tool(self):
        """Test that regular tools are not SubAgentCapable."""
        from dataclasses import dataclass

        @dataclass
        class RegularTool(Tool):
            name: str = "Regular"
            description: str = "A regular tool"

            async def __call__(self, input=None):
                return TextContent(text="result")

        tool = RegularTool()
        assert not isinstance(tool, SubAgentCapable)


class TestDAGSubGraph:
    """Tests for DAG.sub_graph method."""

    def test_add_subgraph_to_dag(self):
        """Test adding SubGraph to DAG."""
        dag = DAG(system_prompt="Test")
        dag = dag.user("Hello")

        sg = SubGraph(
            tool_name="TestAgent",
            tool_use_id="agent_123",
            system_prompt="Sub prompt",
            nodes={},
            head_ids=[],
            summary="Done",
        )

        dag = dag.sub_graph(sg)

        # Verify SubGraph is in the DAG
        ancestors = dag.head.ancestors()
        subgraph_nodes = [n for n in ancestors if isinstance(n.data, SubGraph)]
        assert len(subgraph_nodes) == 1
        assert subgraph_nodes[0].data.tool_name == "TestAgent"
