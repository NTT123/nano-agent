from typing import cast

from nano_agent.tools import (
    DEFAULT_TOOLS,
    BashTool,
    EditConfirmTool,
    EditTool,
    GlobTool,
    InputSchemaDict,
    PythonTool,
    ReadTool,
    GrepTool,
    StatTool,
    TodoWriteTool,
    Tool,
    WebFetchTool,
    WriteTool,
)


def get_properties(schema: InputSchemaDict) -> dict[str, object]:
    """Helper to get properties from schema with proper typing."""
    return cast(dict[str, object], schema.get("properties", {}))


class TestToolBase:
    def test_tool_has_required_fields(self) -> None:
        # Tool base class returns empty schema when no __call__ is defined
        tool = Tool(name="TestTool", description="A test tool")
        assert tool.name == "TestTool"
        assert tool.description == "A test tool"
        # Empty schema fallback when no __call__ defined
        assert tool.input_schema == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def test_tool_to_dict(self) -> None:
        tool = Tool(name="TestTool", description="A test tool")
        result = tool.to_dict()
        assert result["name"] == "TestTool"
        assert result["description"] == "A test tool"
        assert result["input_schema"] == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }


class TestBashTool:
    def test_default_values(self) -> None:
        tool = BashTool()
        assert tool.name == "Bash"
        assert "bash command" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = BashTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "command" in props
        assert "timeout" in props
        assert "run_in_background" in props
        assert schema["required"] == ["command"]


class TestGlobTool:
    def test_default_values(self) -> None:
        tool = GlobTool()
        assert tool.name == "Glob"
        assert "pattern matching" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = GlobTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "pattern" in props
        assert "path" in props
        assert schema["required"] == ["pattern"]


class TestGrepTool:
    def test_default_values(self) -> None:
        tool = GrepTool()
        assert tool.name == "Grep"
        assert "ripgrep" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = GrepTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "pattern" in props
        assert "path" in props
        assert "glob" in props
        assert "output_mode" in props
        # New Python-friendly parameter names
        assert "context_before" in props
        assert "context_after" in props
        assert "context" in props
        assert "line_numbers" in props
        assert "case_insensitive" in props
        assert "file_type" in props
        assert "head_limit" in props
        assert "offset" in props
        assert "multiline" in props
        assert schema["required"] == ["pattern"]


class TestReadTool:
    def test_default_values(self) -> None:
        tool = ReadTool()
        assert tool.name == "Read"
        assert "Reads a file" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = ReadTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "file_path" in props
        assert "offset" in props
        assert "limit" in props
        assert schema["required"] == ["file_path"]


class TestEditTool:
    def test_default_values(self) -> None:
        tool = EditTool()
        assert tool.name == "Edit"
        assert "string replacements" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = EditTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "file_path" in props
        assert "old_string" in props
        assert "new_string" in props
        assert "replace_all" in props
        assert schema["required"] == ["file_path", "old_string", "new_string"]


class TestWriteTool:
    def test_default_values(self) -> None:
        tool = WriteTool()
        assert tool.name == "Write"
        assert "Writes a file" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = WriteTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "file_path" in props
        assert "content" in props
        assert schema["required"] == ["file_path", "content"]


class TestWebFetchTool:
    def test_default_values(self) -> None:
        tool = WebFetchTool()
        assert tool.name == "WebFetch"
        assert "Fetches content" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = WebFetchTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "url" in props
        assert "prompt" in props
        assert schema["required"] == ["url", "prompt"]


class TestTodoWriteTool:
    def test_default_values(self) -> None:
        tool = TodoWriteTool()
        assert tool.name == "TodoWrite"
        assert "task list" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = TodoWriteTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "todos" in props
        assert schema["required"] == ["todos"]


class TestStatTool:
    def test_default_values(self) -> None:
        tool = StatTool()
        assert tool.name == "Stat"
        assert "metadata" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = StatTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "file_path" in props
        assert schema["required"] == ["file_path"]


class TestEditConfirmTool:
    def test_default_values(self) -> None:
        tool = EditConfirmTool()
        assert tool.name == "EditConfirm"
        assert "pending edit" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = EditConfirmTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "edit_id" in props
        assert schema["required"] == ["edit_id"]


class TestPythonTool:
    def test_default_values(self) -> None:
        tool = PythonTool()
        assert tool.name == "Python"
        assert "python" in tool.description.lower()

    def test_input_schema_has_required_fields(self) -> None:
        tool = PythonTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "operation" in props
        assert "code" in props
        assert "file_id" in props
        assert "dependencies" in props
        assert "timeout" in props
        assert "output_limit" in props
        assert "filename" in props
        assert schema["required"] == ["operation"]


class TestDefaultTools:
    def test_default_tools_count(self) -> None:
        # 11 tools (excludes WebSearch stub)
        assert len(DEFAULT_TOOLS) == 11

    def test_default_tools_are_tool_instances(self) -> None:
        for tool in DEFAULT_TOOLS:
            assert isinstance(tool, Tool)

    def test_default_tools_have_unique_names(self) -> None:
        names = [tool.name for tool in DEFAULT_TOOLS]
        assert len(names) == len(set(names))

    def test_default_tools_all_have_to_dict(self) -> None:
        for tool in DEFAULT_TOOLS:
            result = tool.to_dict()
            assert "name" in result
            assert "description" in result
            assert "input_schema" in result

    def test_default_tools_names(self) -> None:
        # Excludes stub tool: WebSearch
        expected_names = {
            "Bash",
            "Glob",
            "Grep",
            "Read",
            "Stat",
            "Edit",
            "EditConfirm",
            "Write",
            "WebFetch",
            "TodoWrite",
            "Python",
        }
        actual_names = {tool.name for tool in DEFAULT_TOOLS}
        assert actual_names == expected_names


# =============================================================================
# Tests for Automatic Schema Inference
# =============================================================================

import asyncio
from dataclasses import dataclass
from typing import Annotated

from nano_agent import TextContent
from nano_agent.tools import (
    BashInput,
    Desc,
    TodoItemInput,
    TodoWriteInput,
    convert_input,
    get_call_input_type,
    schema_from_dataclass,
)


class TestSchemaInference:
    """Tests for automatic schema inference from __call__ type annotations."""

    def test_get_call_input_type_extracts_dataclass(self) -> None:
        @dataclass
        class TestInput:
            value: str

        @dataclass
        class TestTool(Tool):
            name: str = "test"
            description: str = "test"

            async def __call__(self, input: TestInput) -> TextContent:
                return TextContent(text=input.value)

        assert TestTool._input_type is TestInput

    def test_inferred_schema_has_correct_properties(self) -> None:
        @dataclass
        class TestInput:
            value: Annotated[str, Desc("A test value")]

        @dataclass
        class TestTool(Tool):
            name: str = "test"
            description: str = "test"

            async def __call__(self, input: TestInput) -> TextContent:
                return TextContent(text=input.value)

        tool = TestTool()
        schema = tool.input_schema
        props = cast(dict[str, object], schema["properties"])
        value_prop = cast(dict[str, object], props["value"])
        assert value_prop["type"] == "string"
        assert value_prop["description"] == "A test value"
        required = cast(list[str], schema["required"])
        assert "value" in required

    def test_none_input_type_creates_empty_schema(self) -> None:
        @dataclass
        class PingTool(Tool):
            name: str = "ping"
            description: str = "ping"

            async def __call__(self, input: None) -> TextContent:
                return TextContent(text="pong")

        assert PingTool._input_type is None
        tool = PingTool()
        assert tool.input_schema == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def test_execute_converts_dict_to_typed_input(self) -> None:
        @dataclass
        class TestInput:
            value: str

        @dataclass
        class TestTool(Tool):
            name: str = "test"
            description: str = "test"

            async def __call__(self, input: TestInput) -> TextContent:
                # Verify we receive a typed dataclass instance
                assert isinstance(input, TestInput)
                return TextContent(text=input.value)

        tool = TestTool()
        result = asyncio.run(tool.execute({"value": "hello"}))
        assert isinstance(result, TextContent)
        assert result.text == "hello"


class TestConvertInput:
    """Tests for the convert_input utility function."""

    def test_convert_simple_dataclass(self) -> None:
        @dataclass
        class SimpleInput:
            name: str
            count: int

        result = convert_input({"name": "test", "count": 5}, SimpleInput)
        assert isinstance(result, SimpleInput)
        assert result.name == "test"
        assert result.count == 5

    def test_convert_nested_list_of_dataclasses(self) -> None:
        # Test with the actual TodoWriteInput structure
        input_dict = {
            "todos": [
                {"content": "Task 1", "status": "pending", "activeForm": "Doing 1"},
                {"content": "Task 2", "status": "completed", "activeForm": "Doing 2"},
            ]
        }
        result = convert_input(input_dict, TodoWriteInput)
        assert isinstance(result, TodoWriteInput)
        assert len(result.todos) == 2
        assert isinstance(result.todos[0], TodoItemInput)
        assert result.todos[0].content == "Task 1"
        assert result.todos[1].status == "completed"

    def test_convert_none_input_type(self) -> None:
        result = convert_input({"any": "data"}, None)
        assert result is None

    def test_convert_none_input_dict(self) -> None:
        @dataclass
        class EmptyInput:
            pass

        result = convert_input(None, EmptyInput)
        assert isinstance(result, EmptyInput)


class TestBashInputSchema:
    """Tests for BashInput automatic schema inference."""

    def test_bash_input_schema_matches_expected(self) -> None:
        tool = BashTool()
        schema = tool.input_schema
        props = cast(dict[str, object], schema.get("properties", {}))

        assert "command" in props
        assert "timeout" in props
        assert "description" in props
        assert "run_in_background" in props
        assert schema["required"] == ["command"]

    def test_bash_tool_execute_converts_dict(self) -> None:
        tool = BashTool()
        # The execute method should convert dict to BashInput
        result = asyncio.run(tool.execute({"command": "echo hello"}))
        assert isinstance(result, TextContent)
        assert "hello" in result.text


# =============================================================================
# Functional Tests for New Tools
# =============================================================================

import os
import tempfile
from pathlib import Path

from nano_agent.tools import (
    EditConfirmInput,
    EditInput,
    StatInput,
    WebFetchInput,
    WriteInput,
    _pending_edits,
)


class TestEditToolFunctional:
    """Functional tests for the two-step EditTool workflow."""

    def test_edit_file_not_found(self) -> None:
        tool = EditTool()
        result = asyncio.run(
            tool.execute(
                {
                    "file_path": "/nonexistent/file.txt",
                    "old_string": "hello",
                    "new_string": "world",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "not found" in result.text

    def test_edit_old_string_not_found(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world\n")
            temp_path = f.name

        try:
            tool = EditTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": temp_path,
                        "old_string": "nonexistent",
                        "new_string": "replacement",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "not found" in result.text
        finally:
            os.unlink(temp_path)

    def test_edit_non_unique_string_error(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\nhello\nhello\n")
            temp_path = f.name

        try:
            tool = EditTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": temp_path,
                        "old_string": "hello",
                        "new_string": "world",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "not unique" in result.text
            assert "3 occurrences" in result.text
        finally:
            os.unlink(temp_path)

    def test_edit_replace_all_multiple_occurrences(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\nhello\nhello\n")
            temp_path = f.name

        try:
            tool = EditTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": temp_path,
                        "old_string": "hello",
                        "new_string": "world",
                        "replace_all": True,
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Edit Preview" in result.text
            assert "edit_id=" in result.text
            # Should mention all 3 occurrences
            assert "3 occurrences" in result.text
        finally:
            os.unlink(temp_path)

    def test_edit_preview_returns_edit_id(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("def foo():\n    pass\n")
            temp_path = f.name

        try:
            tool = EditTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": temp_path,
                        "old_string": "def foo():",
                        "new_string": "def bar():",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Edit Preview" in result.text
            assert "NOT applied" in result.text
            assert 'edit_id="' in result.text
        finally:
            os.unlink(temp_path)

    def test_edit_same_string_error(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            temp_path = f.name

        try:
            tool = EditTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": temp_path,
                        "old_string": "hello",
                        "new_string": "hello",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "Error" in result.text
            assert "identical" in result.text
        finally:
            os.unlink(temp_path)


class TestEditConfirmToolFunctional:
    """Functional tests for EditConfirmTool."""

    def test_confirm_invalid_edit_id(self) -> None:
        tool = EditConfirmTool()
        result = asyncio.run(tool.execute({"edit_id": "invalid123"}))
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "not found or expired" in result.text

    def test_full_edit_confirm_workflow(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("def foo():\n    pass\n")
            temp_path = f.name

        try:
            # Step 1: Create edit preview
            edit_tool = EditTool()
            preview_result = asyncio.run(
                edit_tool.execute(
                    {
                        "file_path": temp_path,
                        "old_string": "def foo():",
                        "new_string": "def bar():",
                    }
                )
            )

            # Extract edit_id from result
            import re

            assert isinstance(preview_result, TextContent)
            match = re.search(r'edit_id="([^"]+)"', preview_result.text)
            assert match is not None
            edit_id = match.group(1)

            # Step 2: Confirm the edit
            confirm_tool = EditConfirmTool()
            confirm_result = asyncio.run(confirm_tool.execute({"edit_id": edit_id}))
            assert isinstance(confirm_result, TextContent)
            assert "✓ Edit applied" in confirm_result.text

            # Step 3: Verify the file was actually modified
            content = Path(temp_path).read_text()
            assert "def bar():" in content
            assert "def foo():" not in content
        finally:
            os.unlink(temp_path)


class TestWriteToolFunctional:
    """Functional tests for WriteTool."""

    def test_write_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "new_file.txt")

            tool = WriteTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": file_path,
                        "content": "Hello, World!\nLine 2\n",
                    }
                )
            )

            assert isinstance(result, TextContent)
            assert "Created" in result.text
            assert "2 lines" in result.text

            # Verify content
            assert Path(file_path).read_text() == "Hello, World!\nLine 2\n"

    def test_write_overwrite_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("old content\n")
            temp_path = f.name

        try:
            tool = WriteTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": temp_path,
                        "content": "new content\n",
                    }
                )
            )

            assert isinstance(result, TextContent)
            assert "Overwritten" in result.text

            # Verify content
            assert Path(temp_path).read_text() == "new content\n"
        finally:
            os.unlink(temp_path)

    def test_write_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "nested", "dir", "file.txt")

            tool = WriteTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "file_path": file_path,
                        "content": "content\n",
                    }
                )
            )

            assert isinstance(result, TextContent)
            assert "Created" in result.text
            assert Path(file_path).exists()


class TestStatToolFunctional:
    """Functional tests for StatTool."""

    def test_stat_file_not_found(self) -> None:
        tool = StatTool()
        result = asyncio.run(tool.execute({"file_path": "/nonexistent/file.txt"}))
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "Not found" in result.text

    def test_stat_text_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    pass\n")
            temp_path = f.name

        try:
            tool = StatTool()
            result = asyncio.run(tool.execute({"file_path": temp_path}))

            assert isinstance(result, TextContent)
            assert "Stat:" in result.text
            assert "Type:" in result.text
            assert "Size:" in result.text
            assert "Lines:" in result.text
            assert "Modified:" in result.text
            assert "Permissions:" in result.text
        finally:
            os.unlink(temp_path)

    def test_stat_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = StatTool()
            result = asyncio.run(tool.execute({"file_path": temp_dir}))

            assert isinstance(result, TextContent)
            assert "Stat:" in result.text
            assert "Type: directory" in result.text


class TestWebFetchToolFunctional:
    """Functional tests for WebFetchTool."""

    def test_webfetch_invalid_url(self) -> None:
        import shutil

        tool = WebFetchTool()
        result = asyncio.run(
            tool.execute(
                {
                    "url": "not-a-valid-url",
                    "prompt": "summarize",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        # If lynx isn't installed, we get the dependency error first
        # If lynx is installed, we get the invalid URL error
        if shutil.which("lynx") is not None:
            assert "Invalid URL" in result.text
        else:
            assert "'lynx'" in result.text

    def test_webfetch_dependency_check(self) -> None:
        """Test that missing lynx dependency is reported clearly."""
        import shutil

        if shutil.which("lynx") is None:
            tool = WebFetchTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "url": "https://example.com",
                        "prompt": "summarize",
                    }
                )
            )
            assert isinstance(result, TextContent)
            assert "'lynx'" in result.text
            assert "brew install lynx" in result.text


# =============================================================================
# Functional Tests for PythonTool
# =============================================================================

from nano_agent.tools import (
    PythonInput,
    _python_scripts,
    clear_python_scripts,
    list_python_scripts,
)


class TestPythonToolFunctional:
    """Functional tests for PythonTool operations."""

    def setup_method(self) -> None:
        """Clear scripts before each test."""
        clear_python_scripts()

    def teardown_method(self) -> None:
        """Clean up scripts after each test."""
        clear_python_scripts()

    def test_create_empty_code_error(self) -> None:
        """Test that create with empty code returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "'code'" in result.text

    def test_create_whitespace_only_code_error(self) -> None:
        """Test that create with whitespace-only code returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "   \n\t  ",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text

    def test_create_returns_file_id(self) -> None:
        """Test that create operation returns a file_id."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "print('hello')",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "file_id:" in result.text
        assert "py_" in result.text
        assert "✓" in result.text

    def test_create_with_custom_filename(self) -> None:
        """Test create with custom filename."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "print('test')",
                    "filename": "my_script",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "my_script.py" in result.text

    def test_edit_missing_file_id_error(self) -> None:
        """Test that edit without file_id returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "edit",
                    "code": "print('updated')",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "'file_id'" in result.text

    def test_edit_nonexistent_file_error(self) -> None:
        """Test that edit with invalid file_id returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "edit",
                    "file_id": "nonexistent_123",
                    "code": "print('updated')",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "not found or expired" in result.text

    def test_run_missing_file_id_error(self) -> None:
        """Test that run without file_id returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "run",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "'file_id'" in result.text

    def test_run_nonexistent_file_error(self) -> None:
        """Test that run with invalid file_id returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "run",
                    "file_id": "nonexistent_456",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "not found or expired" in result.text

    def test_invalid_operation_error(self) -> None:
        """Test that invalid operation returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "invalid_op",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "Invalid operation" in result.text
        assert "create" in result.text
        assert "edit" in result.text
        assert "run" in result.text
        assert "delete" in result.text

    def test_delete_missing_file_id_error(self) -> None:
        """Test that delete without file_id returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "delete",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "'file_id'" in result.text

    def test_delete_nonexistent_file_error(self) -> None:
        """Test that delete with invalid file_id returns error."""
        tool = PythonTool()
        result = asyncio.run(
            tool.execute(
                {
                    "operation": "delete",
                    "file_id": "nonexistent_789",
                }
            )
        )
        assert isinstance(result, TextContent)
        assert "Error" in result.text
        assert "not found" in result.text

    def test_delete_success(self) -> None:
        """Test successful delete operation."""
        tool = PythonTool()

        # Create a script first
        create_result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "print('to be deleted')",
                }
            )
        )
        assert isinstance(create_result, TextContent)

        import re

        match = re.search(r"file_id: (py_\w+)", create_result.text)
        assert match is not None
        file_id = match.group(1)

        # Verify it exists
        assert file_id in _python_scripts

        # Delete it
        delete_result = asyncio.run(
            tool.execute(
                {
                    "operation": "delete",
                    "file_id": file_id,
                }
            )
        )
        assert isinstance(delete_result, TextContent)
        assert "deleted successfully" in delete_result.text
        assert file_id in delete_result.text

        # Verify it's gone
        assert file_id not in _python_scripts

    def test_full_create_run_workflow(self) -> None:
        """Test complete create → run workflow (requires uv)."""
        import shutil

        if shutil.which("uv") is None:
            # Skip if uv not installed
            return

        tool = PythonTool()

        # Step 1: Create
        create_result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "print('hello from sandbox')",
                }
            )
        )
        assert isinstance(create_result, TextContent)
        assert "file_id:" in create_result.text

        # Extract file_id
        import re

        match = re.search(r"file_id: (py_\w+)", create_result.text)
        assert match is not None
        file_id = match.group(1)

        # Step 2: Run
        run_result = asyncio.run(
            tool.execute(
                {
                    "operation": "run",
                    "file_id": file_id,
                }
            )
        )
        assert isinstance(run_result, TextContent)
        assert "hello from sandbox" in run_result.text
        assert "Exit code: 0" in run_result.text

    def test_create_edit_run_workflow(self) -> None:
        """Test complete create → edit → run workflow (requires uv)."""
        import shutil

        if shutil.which("uv") is None:
            return

        tool = PythonTool()

        # Step 1: Create
        create_result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "print('original')",
                }
            )
        )
        assert isinstance(create_result, TextContent)

        import re

        match = re.search(r"file_id: (py_\w+)", create_result.text)
        assert match is not None
        file_id = match.group(1)

        # Step 2: Edit
        edit_result = asyncio.run(
            tool.execute(
                {
                    "operation": "edit",
                    "file_id": file_id,
                    "code": "print('modified')",
                }
            )
        )
        assert isinstance(edit_result, TextContent)
        assert "updated successfully" in edit_result.text

        # Step 3: Run
        run_result = asyncio.run(
            tool.execute(
                {
                    "operation": "run",
                    "file_id": file_id,
                }
            )
        )
        assert isinstance(run_result, TextContent)
        assert "modified" in run_result.text
        assert "original" not in run_result.text

    def test_run_with_dependencies(self) -> None:
        """Test running with dependencies (requires uv)."""
        import shutil

        if shutil.which("uv") is None:
            return

        tool = PythonTool()

        # Create script that uses numpy
        create_result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "import numpy as np\nprint(np.array([1, 2, 3]))",
                }
            )
        )
        assert isinstance(create_result, TextContent)

        import re

        match = re.search(r"file_id: (py_\w+)", create_result.text)
        assert match is not None
        file_id = match.group(1)

        # Run with numpy dependency
        run_result = asyncio.run(
            tool.execute(
                {
                    "operation": "run",
                    "file_id": file_id,
                    "dependencies": ["numpy"],
                }
            )
        )
        assert isinstance(run_result, TextContent)
        assert "Exit code: 0" in run_result.text
        assert "[1 2 3]" in run_result.text

    def test_run_count_increments(self) -> None:
        """Test that run_count increments on each run."""
        import shutil

        if shutil.which("uv") is None:
            return

        tool = PythonTool()

        # Create
        create_result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "print('test')",
                }
            )
        )
        assert isinstance(create_result, TextContent)

        import re

        match = re.search(r"file_id: (py_\w+)", create_result.text)
        assert match is not None
        file_id = match.group(1)

        # Run twice
        asyncio.run(tool.execute({"operation": "run", "file_id": file_id}))
        run_result = asyncio.run(tool.execute({"operation": "run", "file_id": file_id}))
        assert isinstance(run_result, TextContent)

        assert "Run count: 2" in run_result.text

    def test_list_python_scripts(self) -> None:
        """Test list_python_scripts utility function."""
        tool = PythonTool()

        # Create two scripts
        asyncio.run(tool.execute({"operation": "create", "code": "print(1)"}))
        asyncio.run(tool.execute({"operation": "create", "code": "print(2)"}))

        scripts = list_python_scripts()
        assert len(scripts) == 2

    def test_clear_python_scripts(self) -> None:
        """Test clear_python_scripts utility function."""
        tool = PythonTool()

        # Create scripts
        asyncio.run(tool.execute({"operation": "create", "code": "print(1)"}))
        asyncio.run(tool.execute({"operation": "create", "code": "print(2)"}))

        assert len(_python_scripts) == 2

        # Clear
        count = clear_python_scripts()
        assert count == 2
        assert len(_python_scripts) == 0

    def test_dependency_check_uv(self) -> None:
        """Test that missing uv dependency is reported clearly (if uv not installed)."""
        import shutil

        # This test only meaningful if uv is not installed
        # If uv IS installed, we verify the tool works
        tool = PythonTool()

        # Create a sandbox first
        create_result = asyncio.run(
            tool.execute(
                {
                    "operation": "create",
                    "code": "print('test')",
                }
            )
        )
        assert isinstance(create_result, TextContent)

        import re

        match = re.search(r"file_id: (py_\w+)", create_result.text)
        if match:
            file_id = match.group(1)

            if shutil.which("uv") is None:
                # Test dependency error message
                run_result = asyncio.run(
                    tool.execute({"operation": "run", "file_id": file_id})
                )
                assert isinstance(run_result, TextContent)
                assert "'uv'" in run_result.text
                assert "pip install uv" in run_result.text
