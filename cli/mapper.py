"""DAG to UI message list mapping."""

from __future__ import annotations

from nano_agent import (
    DAG,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)

from .message_factory import (
    add_text_to_assistant,
    add_thinking_to_assistant,
    create_assistant_message,
    create_tool_call_message,
    create_tool_result_message,
    create_user_message,
)
from .message_list import MessageList


class DAGMessageMapper:
    """Rebuilds UI message list from DAG nodes."""

    def rebuild(self, dag: DAG | None) -> MessageList:
        message_list = MessageList()
        if not dag or not dag._heads:
            return message_list
        for node in dag.head.ancestors():
            if isinstance(node.data, Message):
                if node.data.role == Role.USER:
                    self._handle_user(node.data, message_list)
                elif node.data.role == Role.ASSISTANT:
                    self._handle_assistant(node.data, message_list)
        return message_list

    def _handle_user(self, message: Message, message_list: MessageList) -> None:
        if isinstance(message.content, str):
            message_list.add(create_user_message(message.content))
            return

        text_parts = []
        tool_results = []
        for block in message.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ToolResultContent):
                tool_results.append(block)

        if text_parts:
            message_list.add(create_user_message("\n".join(text_parts)))

        for tool_result in tool_results:
            if isinstance(tool_result.content, str):
                result_text = tool_result.content
            else:
                result_parts = []
                for content_block in tool_result.content:
                    if isinstance(content_block, TextContent):
                        result_parts.append(content_block.text)
                result_text = "\n".join(result_parts)
            is_error = (
                tool_result.is_error if hasattr(tool_result, "is_error") else False
            )
            message_list.add(create_tool_result_message(result_text, is_error=is_error))

    def _handle_assistant(self, message: Message, message_list: MessageList) -> None:
        if isinstance(message.content, str):
            if message.content.strip():
                msg = create_assistant_message()
                add_text_to_assistant(msg, message.content)
                message_list.add(msg)
            return

        thinking_blocks = []
        text_parts = []
        tool_uses = []
        for block in message.content:
            if isinstance(block, ThinkingContent):
                if block.thinking:
                    thinking_blocks.append(block.thinking)
            elif isinstance(block, TextContent):
                if block.text.strip():
                    text_parts.append(block.text)
            elif isinstance(block, ToolUseContent):
                tool_uses.append(block)

        if thinking_blocks or text_parts:
            msg = create_assistant_message()
            for thinking in thinking_blocks:
                add_thinking_to_assistant(msg, thinking)
            if text_parts:
                add_text_to_assistant(
                    msg,
                    "\n".join(text_parts),
                    has_thinking=bool(thinking_blocks),
                )
            message_list.add(msg)

        for tool_use in tool_uses:
            message_list.add(
                create_tool_call_message(tool_use.name, tool_use.input or {})
            )
