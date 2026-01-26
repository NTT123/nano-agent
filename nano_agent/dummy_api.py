"""Dummy API provider for testing the CLI.

This module provides a mock API that randomly generates responses including
thinking, tool calls, and text content, useful for testing the CLI without
needing actual API credentials.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from .dag import DAG
from .data_structures import (
    ContentBlock,
    Message,
    Response,
    Role,
    TextContent,
    ThinkingContent,
    ToolUseContent,
    Usage,
)
from .tools import Tool


class DummyAPI:
    """Dummy API provider that generates random responses for testing.

    This provider simulates real API behavior by randomly generating:
    - Thinking content (extended reasoning)
    - Tool calls (randomly selected from available tools)
    - Text responses

    Useful for testing the CLI interface without API credentials.
    """

    def __init__(
        self,
        model: str = "dummy-model-v1",
        thinking_probability: float = 0.3,
        tool_call_probability: float = 0.4,
        max_tool_calls: int = 2,
    ):
        """Initialize dummy API.

        Args:
            model: Model name to report
            thinking_probability: Probability of including thinking content (0-1)
            tool_call_probability: Probability of making tool calls (0-1)
            max_tool_calls: Maximum number of tool calls per response
        """
        self.model = model
        self.thinking_probability = thinking_probability
        self.tool_call_probability = tool_call_probability
        self.max_tool_calls = max_tool_calls
        self._response_count = 0

    async def send(
        self,
        messages: list[Message] | DAG,
        tools: list[Tool] | None = None,
        system: str | list[dict[str, Any]] | None = None,
    ) -> Response:
        """Generate a dummy response.

        Args:
            messages: List of Message objects OR a DAG instance
            tools: Available tools (extracted from DAG if provided)
            system: System prompt (ignored for dummy)

        Returns:
            Response object with randomly generated content
        """
        # Extract from DAG if provided
        if isinstance(messages, DAG):
            dag = messages
            messages = dag.to_messages()
            tools = list(dag._tools or [])

        self._response_count += 1
        content: list[ContentBlock] = []

        # Maybe add thinking
        if random.random() < self.thinking_probability:
            thinking_samples = [
                "Let me analyze this request carefully...",
                "I need to think about the best approach here.",
                "Considering the available tools and the user's request...",
                "Let me break this down step by step.",
                "I should consider what information I need first.",
            ]
            content.append(
                ThinkingContent(thinking=random.choice(thinking_samples))
            )

        # Maybe make tool calls
        should_call_tools = (
            tools is not None
            and len(tools) > 0
            and random.random() < self.tool_call_probability
        )

        if should_call_tools and tools:
            num_calls = random.randint(1, min(self.max_tool_calls, len(tools)))
            selected_tools = random.sample(tools, num_calls)

            for tool in selected_tools:
                # Generate dummy input for the tool
                tool_input = self._generate_tool_input(tool)
                content.append(
                    ToolUseContent(
                        id=f"toolu_{uuid.uuid4().hex[:16]}",
                        name=tool.name,
                        input=tool_input,
                    )
                )
        else:
            # Add text response if not calling tools
            text_samples = [
                "I've completed the analysis.",
                "Here's what I found based on the information provided.",
                "Let me provide a detailed response to your query.",
                "Based on my understanding, here's the answer.",
                f"This is response #{self._response_count} from the dummy provider.",
            ]
            content.append(TextContent(text=random.choice(text_samples)))

        return Response(
            id=f"msg_{uuid.uuid4().hex[:16]}",
            model=self.model,
            role=Role.ASSISTANT,
            content=content,
            stop_reason="end_turn" if not should_call_tools else "tool_use",
            usage=Usage(
                input_tokens=random.randint(100, 500),
                output_tokens=random.randint(50, 200),
            ),
        )

    def _generate_tool_input(self, tool: Tool) -> dict[str, Any]:
        """Generate dummy input for a tool based on its schema.

        Args:
            tool: Tool to generate input for

        Returns:
            Dictionary of dummy input parameters
        """
        schema = tool.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if not isinstance(properties, dict):
            properties = {}
        if not isinstance(required, list):
            required = []

        dummy_input: dict[str, Any] = {}

        for prop_name, prop_schema in properties.items():
            # Only include required properties to keep it simple
            if prop_name not in required:
                continue

            prop_type = prop_schema.get("type", "string")

            if prop_type == "string":
                # Generate appropriate dummy strings based on property name
                if "pattern" in prop_name:
                    dummy_input[prop_name] = "test.*"
                elif "path" in prop_name:
                    dummy_input[prop_name] = "."
                elif "command" in prop_name:
                    dummy_input[prop_name] = "echo 'Hello from dummy API'"
                elif "content" in prop_name or "text" in prop_name:
                    dummy_input[prop_name] = "Dummy content generated by test API"
                else:
                    dummy_input[prop_name] = "dummy_value"
            elif prop_type == "integer":
                dummy_input[prop_name] = random.randint(1, 10)
            elif prop_type == "number":
                dummy_input[prop_name] = round(random.uniform(0, 1), 2)
            elif prop_type == "boolean":
                dummy_input[prop_name] = random.choice([True, False])
            elif prop_type == "array":
                dummy_input[prop_name] = []
            elif prop_type == "object":
                dummy_input[prop_name] = {}

        return dummy_input
