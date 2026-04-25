import asyncio
import base64
import os
import re
import shlex
import stat
import sys
from typing import cast

import pytest

from nano_agent.tools import (
    ApplyPatchTool,
    BashTool,
    DelegateTaskTool,
    DownloadSkillTool,
    EditTool,
    ExecCommandTool,
    GlobTool,
    GrepTool,
    InputSchemaDict,
    PythonTool,
    ReadTool,
    Skill,
    StatTool,
    TodoWriteTool,
    Tool,
    ViewImageTool,
    WebFetchTool,
    WriteStdinTool,
    WriteTool,
    get_default_tools,
)


def get_properties(schema: InputSchemaDict) -> dict[str, object]:
    """Helper to get properties from schema with proper typing."""
    return cast(dict[str, object], schema.get("properties", {}))


def python_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def extract_session_id(text: str) -> int:
    match = re.search(r"Process running with session ID (\d+)", text)
    assert match is not None
    return int(match.group(1))


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

    def test_tool_required_commands_missing_raises_error(self) -> None:
        """Test that missing required CLI commands raise RuntimeError on init."""
        from dataclasses import dataclass
        from typing import ClassVar

        import pytest

        @dataclass
        class ToolWithMissingCommand(Tool):
            name: str = "TestToolMissingCmd"
            description: str = "A test tool requiring a non-existent command"
            _required_commands: ClassVar[dict[str, str]] = {
                "nonexistent_command_xyz123": "This command does not exist"
            }

        with pytest.raises(RuntimeError) as exc_info:
            ToolWithMissingCommand()

        assert "nonexistent_command_xyz123" in str(exc_info.value)
        assert "This command does not exist" in str(exc_info.value)
        assert "TestToolMissingCmd" in str(exc_info.value)

    def test_tool_required_commands_present_succeeds(self) -> None:
        """Test that tools with available CLI commands instantiate correctly."""
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass
        class ToolWithPresentCommand(Tool):
            name: str = "TestToolPresentCmd"
            description: str = "A test tool requiring a common command"
            _required_commands: ClassVar[dict[str, str]] = {
                "python": "Python should be available"  # Python is always available
            }

        # Should not raise
        tool = ToolWithPresentCommand()
        assert tool.name == "TestToolPresentCmd"


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


class TestExecCommandTool:
    def test_default_values(self) -> None:
        tool = ExecCommandTool()
        assert tool.name == "exec_command"
        assert "session ID" in tool.description

    def test_input_schema_matches_codex_shape(self) -> None:
        tool = ExecCommandTool()
        schema = tool.input_schema
        props = get_properties(schema)
        expected_props = {
            "cmd",
            "workdir",
            "shell",
            "tty",
            "yield_time_ms",
            "max_output_tokens",
            "login",
            "sandbox_permissions",
            "justification",
            "prefix_rule",
        }
        assert expected_props <= set(props)
        assert schema["required"] == ["cmd"]


class TestWriteStdinTool:
    def test_default_values(self) -> None:
        tool = WriteStdinTool()
        assert tool.name == "write_stdin"
        assert "unified exec session" in tool.description

    def test_input_schema_matches_codex_shape(self) -> None:
        tool = WriteStdinTool()
        schema = tool.input_schema
        props = get_properties(schema)
        expected_props = {"session_id", "chars", "yield_time_ms", "max_output_tokens"}
        assert expected_props <= set(props)
        assert schema["required"] == ["session_id"]


class TestApplyPatchTool:
    def test_default_values(self) -> None:
        tool = ApplyPatchTool()
        assert tool.name == "apply_patch"
        assert "edit files" in tool.description

    def test_input_schema_matches_codex_json_shape(self) -> None:
        tool = ApplyPatchTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "input" in props
        assert schema["required"] == ["input"]


class TestViewImageTool:
    def test_default_values(self) -> None:
        tool = ViewImageTool()
        assert tool.name == "view_image"
        assert "local image" in tool.description

    def test_input_schema_matches_codex_shape(self) -> None:
        tool = ViewImageTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert {"path", "detail"} <= set(props)
        assert schema["required"] == ["path"]


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


class TestDownloadSkillTool:
    def test_default_values_empty_skills(self) -> None:
        tool = DownloadSkillTool()
        assert tool.name == "DownloadSkill"
        assert "Download a skill" in tool.description
        assert "No skills are currently registered" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = DownloadSkillTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "name" in props
        assert schema["required"] == ["name"]

    def test_description_compiles_skill_descriptions(self) -> None:
        tool = DownloadSkillTool(
            skills=[
                Skill(
                    name="kung_fu",
                    description="Use when you need to fight",
                    knowledge="Block, strike, sweep",
                ),
                Skill(
                    name="helicopter",
                    description="Use when you need to fly a chopper",
                    knowledge="Pull collective, push cyclic",
                ),
            ]
        )
        assert "kung_fu: Use when you need to fight" in tool.description
        assert "helicopter: Use when you need to fly a chopper" in tool.description

    def test_returns_knowledge_for_known_skill(self) -> None:
        tool = DownloadSkillTool(
            skills=[
                Skill(
                    name="kung_fu",
                    description="Use when you need to fight",
                    knowledge="Block, strike, sweep",
                ),
            ]
        )
        result = asyncio.run(tool.execute({"name": "kung_fu"}))
        assert isinstance(result.content, TextContent)
        assert result.content.text == "Block, strike, sweep"

    def test_returns_error_for_unknown_skill(self) -> None:
        tool = DownloadSkillTool(
            skills=[
                Skill(name="kung_fu", description="d", knowledge="k"),
            ]
        )
        result = asyncio.run(tool.execute({"name": "missing"}))
        assert isinstance(result.content, TextContent)
        assert "not found" in result.content.text
        assert "kung_fu" in result.content.text

    def test_returns_error_when_skills_empty(self) -> None:
        tool = DownloadSkillTool()
        result = asyncio.run(tool.execute({"name": "anything"}))
        assert isinstance(result.content, TextContent)
        assert "not found" in result.content.text
        assert "(none)" in result.content.text

    def test_description_argument_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            DownloadSkillTool(description="custom")  # type: ignore[call-arg]


class FakeCodex:
    """Helper that writes an executable shell script to a tmp dir on PATH."""

    def __init__(self, script_path: "os.PathLike[str]") -> None:
        self.script_path = script_path

    def set_script(self, body: str) -> None:
        from pathlib import Path

        path = Path(self.script_path)
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """Provide a stand-in `codex` binary on PATH; tests fill in its body."""
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return FakeCodex(tmp_path / "codex")


class TestDelegateTaskTool:
    def test_default_values(self) -> None:
        tool = DelegateTaskTool()
        assert tool.name == "DelegateTask"
        assert "Delegate a task" in tool.description

    def test_input_schema_has_required_fields(self) -> None:
        tool = DelegateTaskTool()
        schema = tool.input_schema
        props = get_properties(schema)
        assert "prompt" in props
        assert "model" in props
        assert "sandbox" in props
        assert "cwd" in props
        assert "timeout" in props
        assert schema["required"] == ["prompt"]

    def test_empty_prompt_returns_error(self) -> None:
        tool = DelegateTaskTool()
        result = asyncio.run(tool.execute({"prompt": "   "}))
        assert isinstance(result.content, TextContent)
        assert "prompt is required" in result.content.text

    def test_invalid_sandbox_returns_error(self) -> None:
        tool = DelegateTaskTool()
        result = asyncio.run(tool.execute({"prompt": "do x", "sandbox": "wide-open"}))
        assert isinstance(result.content, TextContent)
        assert "invalid sandbox" in result.content.text

    def test_missing_codex_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _name: None)
        tool = DelegateTaskTool()
        result = asyncio.run(tool.execute({"prompt": "anything"}))
        assert isinstance(result.content, TextContent)
        assert "codex CLI not found" in result.content.text

    def test_successful_execution(self, fake_codex: "FakeCodex") -> None:
        fake_codex.set_script(
            "#!/bin/sh\n" 'echo "ARGS: $*"\n' "echo SUB_AGENT_OUTPUT\n"
        )
        tool = DelegateTaskTool()
        result = asyncio.run(
            tool.execute(
                {
                    "prompt": "Research X",
                    "model": "gpt-5",
                    "sandbox": "read-only",
                }
            )
        )
        assert isinstance(result.content, TextContent)
        text = result.content.text
        assert "SUB_AGENT_OUTPUT" in text
        assert "exec" in text
        assert "--model gpt-5" in text
        assert "--sandbox read-only" in text
        assert "--skip-git-repo-check" in text
        assert "Research X" in text

    def test_failure_exit_code_returns_error(self, fake_codex: "FakeCodex") -> None:
        fake_codex.set_script("#!/bin/sh\n" "echo 'something broke' >&2\n" "exit 2\n")
        tool = DelegateTaskTool()
        result = asyncio.run(tool.execute({"prompt": "anything"}))
        assert isinstance(result.content, TextContent)
        assert "exit" in result.content.text.lower()
        assert "code 2" in result.content.text
        assert "something broke" in result.content.text

    def test_passes_cwd_when_set(self, fake_codex: "FakeCodex") -> None:
        fake_codex.set_script('#!/bin/sh\necho "ARGS: $*"\n')
        tool = DelegateTaskTool()
        result = asyncio.run(
            tool.execute(
                {
                    "prompt": "p",
                    "cwd": "/tmp/work",
                }
            )
        )
        assert isinstance(result.content, TextContent)
        assert "--cd /tmp/work" in result.content.text


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
        # 11 tools (excludes WebSearch stub, EditConfirm removed)
        tools = get_default_tools()
        assert len(tools) == 11

    def test_default_tools_are_tool_instances(self) -> None:
        tools = get_default_tools()
        for tool in tools:
            assert isinstance(tool, Tool)

    def test_default_tools_have_unique_names(self) -> None:
        tools = get_default_tools()
        names = [tool.name for tool in tools]
        assert len(names) == len(set(names))

    def test_default_tools_all_have_to_dict(self) -> None:
        tools = get_default_tools()
        for tool in tools:
            result = tool.to_dict()
            assert "name" in result
            assert "description" in result
            assert "input_schema" in result

    def test_default_tools_names(self) -> None:
        # Excludes stub tool: WebSearch; EditConfirm removed
        expected_names = {
            "Bash",
            "Glob",
            "Grep",
            "Read",
            "Stat",
            "Edit",
            "Write",
            "WebFetch",
            "TodoWrite",
            "Python",
            "AskUserQuestion",
        }
        tools = get_default_tools()
        actual_names = {tool.name for tool in tools}
        assert actual_names == expected_names


# =============================================================================
# Tests for Automatic Schema Inference
# =============================================================================

import asyncio
from dataclasses import dataclass
from typing import Annotated

from nano_agent import ImageContent, TextContent
from nano_agent.tools import (
    BashInput,
    Desc,
    TodoItemInput,
    TodoWriteInput,
    cleanup_exec_command_sessions,
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


class TestExecCommandToolFunctional:
    """Functional tests for the Codex-style exec_command tool."""

    def teardown_method(self) -> None:
        asyncio.run(cleanup_exec_command_sessions())

    def test_exec_command_runs_command(self) -> None:
        tool = ExecCommandTool()
        result = asyncio.run(
            tool.execute(
                {
                    "cmd": python_cmd("print('hello')"),
                    "shell": "/bin/sh",
                    "login": False,
                }
            )
        )

        assert isinstance(result.content, TextContent)
        assert re.search(r"^Chunk ID: [0-9a-f]{6}$", result.content.text, re.M)
        assert "Process exited with code 0" in result.content.text
        assert "Original token count:" in result.content.text
        assert "hello" in result.content.text

    def test_exec_command_uses_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ExecCommandTool()
            result = asyncio.run(
                tool.execute(
                    {
                        "cmd": python_cmd(
                            "from pathlib import Path; print(Path.cwd())"
                        ),
                        "workdir": temp_dir,
                        "shell": "/bin/sh",
                        "login": False,
                    }
                )
            )

        assert isinstance(result.content, TextContent)
        assert "Process exited with code 0" in result.content.text
        assert temp_dir in result.content.text

    def test_exec_command_returns_session_id_when_still_running(self) -> None:
        tool = ExecCommandTool()
        result = asyncio.run(
            tool.execute(
                {
                    "cmd": python_cmd("import time; time.sleep(5)"),
                    "shell": "/bin/sh",
                    "login": False,
                    "yield_time_ms": 1,
                }
            )
        )

        assert isinstance(result.content, TextContent)
        assert "Process running with session ID" in result.content.text

    def test_exec_command_rejects_unknown_sandbox_permissions(self) -> None:
        tool = ExecCommandTool()
        result = asyncio.run(
            tool.execute(
                {
                    "cmd": "echo nope",
                    "sandbox_permissions": "unknown",
                }
            )
        )

        assert isinstance(result.content, TextContent)
        assert "sandbox_permissions must be" in result.content.text

    def test_write_stdin_writes_to_tty_session(self) -> None:
        if os.name == "nt":
            import pytest

            pytest.skip("PTY-backed write_stdin is only supported on POSIX")

        exec_tool = ExecCommandTool()
        write_tool = WriteStdinTool()
        start = asyncio.run(
            exec_tool.execute(
                {
                    "cmd": python_cmd(
                        "import sys; "
                        "print('ready', flush=True); "
                        "line = sys.stdin.readline(); "
                        "print('got:' + line.strip(), flush=True)"
                    ),
                    "shell": "/bin/sh",
                    "login": False,
                    "tty": True,
                    "yield_time_ms": 250,
                }
            )
        )

        assert isinstance(start.content, TextContent)
        assert "ready" in start.content.text
        session_id = extract_session_id(start.content.text)
        result = asyncio.run(
            write_tool.execute(
                {
                    "session_id": session_id,
                    "chars": "ping\n",
                    "yield_time_ms": 250,
                }
            )
        )

        assert isinstance(result.content, TextContent)
        assert "ready" not in result.content.text
        assert "Process exited with code 0" in result.content.text
        assert "got:ping" in result.content.text

    def test_write_stdin_rejects_non_tty_session_input(self) -> None:
        exec_tool = ExecCommandTool()
        write_tool = WriteStdinTool()
        start = asyncio.run(
            exec_tool.execute(
                {
                    "cmd": python_cmd("import time; time.sleep(5)"),
                    "shell": "/bin/sh",
                    "login": False,
                    "yield_time_ms": 250,
                }
            )
        )

        assert isinstance(start.content, TextContent)
        session_id = extract_session_id(start.content.text)
        result = asyncio.run(
            write_tool.execute({"session_id": session_id, "chars": "hello\n"})
        )

        assert isinstance(result.content, TextContent)
        assert "rerun exec_command with tty=true" in result.content.text


class TestApplyPatchToolFunctional:
    """Functional tests for the Codex-style apply_patch tool."""

    def test_apply_patch_adds_and_updates_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                tool = ApplyPatchTool()
                result = asyncio.run(
                    tool.execute(
                        {
                            "input": (
                                "*** Begin Patch\n"
                                "*** Add File: sample.txt\n"
                                "+hello\n"
                                "*** Update File: sample.txt\n"
                                "@@\n"
                                "-hello\n"
                                "+hello world\n"
                                "*** End Patch"
                            )
                        }
                    )
                )
            finally:
                os.chdir(previous_cwd)

            assert isinstance(result.content, TextContent)
            assert "A sample.txt" in result.content.text
            assert "M sample.txt" in result.content.text
            assert Path(temp_dir, "sample.txt").read_text() == "hello world\n"

    def test_apply_patch_rejects_absolute_paths(self) -> None:
        tool = ApplyPatchTool()
        result = asyncio.run(
            tool.execute(
                {
                    "input": (
                        "*** Begin Patch\n"
                        "*** Add File: /tmp/nope.txt\n"
                        "+hello\n"
                        "*** End Patch"
                    )
                }
            )
        )

        assert isinstance(result.content, TextContent)
        assert "absolute paths are not allowed" in result.content.text

    def test_apply_patch_rejects_parent_paths_outside_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "cwd"
            temp_path.mkdir()
            outside_path = Path(temp_dir) / "outside.txt"

            previous_cwd = os.getcwd()
            os.chdir(temp_path)
            try:
                tool = ApplyPatchTool()
                result = asyncio.run(
                    tool.execute(
                        {
                            "input": (
                                "*** Begin Patch\n"
                                "*** Add File: ../outside.txt\n"
                                "+hello\n"
                                "*** End Patch"
                            )
                        }
                    )
                )
            finally:
                os.chdir(previous_cwd)

            assert isinstance(result.content, TextContent)
            assert "path escapes working directory" in result.content.text
            assert not outside_path.exists()


class TestViewImageToolFunctional:
    """Functional tests for the Codex-style view_image tool."""

    def test_view_image_returns_image_content(self) -> None:
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "pixel.png"
            image_path.write_bytes(png_data)

            tool = ViewImageTool()
            result = asyncio.run(
                tool.execute({"path": str(image_path), "detail": "original"})
            )

        assert isinstance(result.content, ImageContent)
        assert result.content.media_type == "image/png"
        assert result.content.data

    def test_view_image_rejects_unknown_detail(self) -> None:
        tool = ViewImageTool()
        result = asyncio.run(tool.execute({"path": "missing.png", "detail": "high"}))

        assert isinstance(result.content, TextContent)
        assert "only supports `original`" in result.content.text


# =============================================================================
# Functional Tests for New Tools
# =============================================================================

import os
import tempfile
from pathlib import Path

from nano_agent.tools import (
    EditInput,
    StatInput,
    WebFetchInput,
    WriteInput,
)


class TestEditToolFunctional:
    """Functional tests for EditTool (auto-approves without permission_callback)."""

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
            # Without permission_callback, edit is auto-approved
            assert "✓ Edit applied" in result.text
            assert "3 occurrences" in result.text

            # Verify file was modified
            content = Path(temp_path).read_text()
            assert content == "world\nworld\nworld\n"
        finally:
            os.unlink(temp_path)

    def test_edit_applies_without_callback(self) -> None:
        """Edit tool auto-approves when no permission_callback is set."""
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
            assert "✓ Edit applied" in result.text

            # Verify file was modified
            content = Path(temp_path).read_text()
            assert "def bar():" in content
            assert "def foo():" not in content
        finally:
            os.unlink(temp_path)

    def test_edit_rejected_by_callback(self) -> None:
        """Edit tool respects permission_callback rejection."""

        async def reject_callback(
            file_path: str, preview: str, match_count: int
        ) -> bool:
            return False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("def foo():\n    pass\n")
            temp_path = f.name

        try:
            tool = EditTool(permission_callback=reject_callback)
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
            assert "rejected" in result.text.lower()

            # Verify file was NOT modified
            content = Path(temp_path).read_text()
            assert "def foo():" in content
            assert "def bar():" not in content
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

        # Skip if lynx is not installed (tool instantiation will raise RuntimeError)
        if shutil.which("lynx") is None:
            import pytest

            pytest.skip(
                "lynx not installed - covered by test_webfetch_dependency_check"
            )
            return

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
        assert "Invalid URL" in result.text

    def test_webfetch_dependency_check(self) -> None:
        """Test that missing lynx dependency raises RuntimeError on init."""
        import shutil

        if shutil.which("lynx") is None:
            # When lynx is not installed, tool instantiation should raise RuntimeError
            import pytest

            with pytest.raises(RuntimeError) as exc_info:
                WebFetchTool()
            assert "'lynx'" in str(exc_info.value)
            assert "Install lynx" in str(exc_info.value)


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
        """Test that missing uv dependency raises RuntimeError on init."""
        import shutil

        if shutil.which("uv") is None:
            # When uv is not installed, tool instantiation should raise RuntimeError
            import pytest

            with pytest.raises(RuntimeError) as exc_info:
                PythonTool()
            assert "'uv'" in str(exc_info.value)
            assert "Install uv" in str(exc_info.value)
        else:
            # If uv IS installed, verify the tool instantiates correctly
            tool = PythonTool()
            assert tool.name == "Python"
