from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

from PIL import Image

import nano_agent.core as core
import nano_agent.tools as tools
from nano_agent.tools import (
    _data_url_to_anthropic_source,
    _file_mtime,
    build_anthropic_messages,
    chat_followup_image_message,
)

from .helpers import assert_equal, assert_true


def test_parse_tool_call_arguments() -> None:
    assert_equal(tools.parse_tool_call_arguments(None), {})
    assert_equal(tools.parse_tool_call_arguments("{}"), {})
    assert_equal(tools.parse_tool_call_arguments('{"x": 1}'), {"x": 1})
    try:
        tools.parse_tool_call_arguments("[]")
    except ValueError:
        return
    raise AssertionError("parse_tool_call_arguments should reject non-object JSON")


def test_validate_bash_args_accepts_quoted_operators() -> None:
    command = 'printf "%s\\n" "a && b" && pwd'
    validated, description, timeout = tools.validate_bash_args(
        command,
        "accept quoted operators",
        10,
    )
    assert_equal(validated, command)
    assert_equal(description, "accept quoted operators")
    assert_equal(timeout, 10)


def test_validate_bash_args_accepts_multiline_commands() -> None:
    command = "printf 'hello\\n'\npwd"
    validated, description, timeout = tools.validate_bash_args(
        command,
        "accept multiline commands",
        10,
    )
    assert_equal(validated, command)
    assert_equal(description, "accept multiline commands")
    assert_equal(timeout, 10)


def test_validate_bash_args_rejects_banned_commands() -> None:
    try:
        tools.validate_bash_args("echo hi\nless file.txt", "reject banned commands", 10)
    except ValueError as exc:
        assert_equal(str(exc), "command is banned: less")
        return
    raise AssertionError("validate_bash_args should reject banned commands")


def test_apply_patch_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        target = cwd / "demo.txt"
        target.write_text("alpha\nbeta\n")
        result = tools.execute_apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: demo.txt",
                    "@@",
                    " alpha",
                    "-beta",
                    "+gamma",
                    "*** End Patch",
                ]
            ),
            cwd,
        )
        assert_true(result["success"], "apply_patch should succeed")
        assert_equal(target.read_text(), "alpha\ngamma\n")


def test_apply_patch_absolute_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        target = cwd / "absolute.txt"
        target.write_text("alpha\nbeta\n")
        result = tools.execute_apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    f"*** Update File: {target}",
                    "@@",
                    " alpha",
                    "-beta",
                    "+gamma",
                    "*** End Patch",
                ]
            ),
            cwd,
        )
        assert_true(result["success"], "apply_patch should accept absolute paths")
        assert_equal(target.read_text(), "alpha\ngamma\n")


def test_validate_fetch_url() -> None:
    tools.validate_fetch_url("https://example.com/test")
    tools.validate_fetch_url("http://localhost:8080/test")
    try:
        tools.validate_fetch_url("http://example.com/test")
    except ValueError:
        return
    raise AssertionError("validate_fetch_url should reject external http URLs")


def test_build_anthropic_messages() -> None:
    system, messages = build_anthropic_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"echo hi"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"success": true}'},
        ]
    )
    assert_equal(system, "sys")
    assert_equal(messages[0], {"role": "user", "content": "hello"})
    assistant_blocks = messages[1]["content"]
    assert_true(
        isinstance(assistant_blocks, list), "assistant content should be block list"
    )
    assert_equal(assistant_blocks[1]["type"], "tool_use")
    assert_equal(messages[2]["content"][0]["type"], "tool_result")

    cached_system, cached_messages = build_anthropic_messages(
        [
            {
                "role": "system",
                "content": "cached-sys",
                "copilot_cache_control": {"type": "ephemeral"},
            },
            {
                "role": "user",
                "content": "cached-user",
                "copilot_cache_control": {"type": "ephemeral"},
            },
            {
                "role": "assistant",
                "content": "cached-assistant",
                "copilot_cache_control": {"type": "ephemeral"},
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "content": "tool-output",
                "copilot_cache_control": {"type": "ephemeral"},
            },
        ]
    )
    assert_true(
        isinstance(cached_system, list),
        "system cache control should force block-based system payload",
    )
    cached_system_blocks = cast(list[dict[str, Any]], cached_system)
    assert_equal(
        cast(dict[str, Any], cached_system_blocks[0].get("cache_control")).get("type"),
        "ephemeral",
    )
    assert_equal(cached_messages[0]["content"][0]["cache_control"]["type"], "ephemeral")
    assert_equal(
        cached_messages[1]["content"][-1]["cache_control"]["type"], "ephemeral"
    )
    assert_equal(cached_messages[2]["content"][0]["cache_control"]["type"], "ephemeral")

    image_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
    _system, image_messages = build_anthropic_messages(
        [
            {
                "role": "tool",
                "tool_call_id": "call-image",
                "content": [{"type": "input_image", "image_url": image_data_url}],
            }
        ]
    )
    tool_result = image_messages[0]["content"][0]
    assert_equal(tool_result["type"], "tool_result")
    image_block = tool_result["content"][0]
    assert_equal(image_block["type"], "image")
    assert_equal(image_block["source"]["media_type"], "image/png")


def test_view_image_schema_and_helpers() -> None:
    basic_schema = tools.view_image_schema("gpt-5")
    assert_true(
        "detail" not in cast(dict[str, Any], basic_schema["properties"]),
        "non-supported models should omit detail from view_image schema",
    )

    codex_schema = tools.view_image_schema("gpt-5.3-codex")
    assert_true(
        "detail" in cast(dict[str, Any], codex_schema["properties"]),
        "gpt-5.3-codex should expose detail in view_image schema",
    )

    image_result = {
        "success": True,
        "path": "/tmp/test.png",
        "image_url": "data:image/png;base64,AAA",
        "detail": "original",
    }
    response_output = tools.responses_function_call_output(image_result)
    assert_true(
        isinstance(response_output, list), "responses output should allow image parts"
    )
    response_parts = cast(list[dict[str, Any]], response_output)
    assert_equal(response_parts[0]["type"], "input_image")
    assert_equal(response_parts[0]["detail"], "original")

    chat_content = tools.chat_tool_message_content(image_result)
    assert_true(isinstance(chat_content, str), "chat tool content should stay textual")
    assert_true(
        "[omitted data URL]" in cast(str, chat_content),
        "chat tool content should elide the data URL",
    )

    followup = chat_followup_image_message(image_result)
    assert_true(
        followup is not None,
        "chat completions should synthesize a follow-up image message",
    )
    followup_message = cast(dict[str, Any], followup)
    content = cast(list[dict[str, Any]], followup_message["content"])
    assert_equal(content[0]["type"], "image_url")
    assert_equal(content[0]["image_url"]["detail"], "high")

    anthropic_source = _data_url_to_anthropic_source(image_result["image_url"])
    assert_equal(anthropic_source["type"], "base64")
    assert_equal(anthropic_source["media_type"], "image/png")


def test_view_image_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "example.png"
        Image.new("RGBA", (2304, 864), (255, 0, 0, 255)).save(image_path, format="PNG")

        resized = tools.make_view_image_result(
            image_path,
            detail=None,
            model="gpt-5",
            provider="openai",
        )
        assert_equal(resized["mime_type"], "image/png")
        assert_true(
            resized["width"] <= core.VIEW_IMAGE_MAX_WIDTH,
            "resized width should be bounded",
        )
        assert_true(
            resized["height"] <= core.VIEW_IMAGE_MAX_HEIGHT,
            "resized height should be bounded",
        )
        assert_true(resized["width"] < 2304, "resized image should shrink width")
        assert_equal(resized.get("detail"), None)

        original = tools.make_view_image_result(
            image_path,
            detail="original",
            model="gpt-5.3-codex",
            provider="openai",
        )
        assert_equal(original["width"], 2304)
        assert_equal(original["height"], 864)
        assert_equal(original["detail"], "original")

        try:
            tools.make_view_image_result(
                image_path,
                detail="low",
                model="gpt-5.3-codex",
                provider="openai",
            )
        except ValueError as exc:
            assert_true(
                "only supports `original`" in str(exc),
                "unsupported detail should produce a clear error",
            )
        else:
            raise AssertionError("view_image should reject unsupported detail values")

        text_path = Path(tmp) / "example.json"
        text_path.write_text('{"hello": true}')
        try:
            tools.make_view_image_result(
                text_path,
                detail=None,
                model="gpt-5",
                provider="openai",
            )
        except ValueError as exc:
            assert_true(
                "unsupported image" in str(exc),
                "non-image files should produce an unsupported image error",
            )
        else:
            raise AssertionError("view_image should reject non-image files")


def test_truncate_tool_output() -> None:
    passthrough, passthrough_truncated = tools.truncate_tool_output("hello world")
    assert_equal(passthrough, "hello world")
    assert_true(not passthrough_truncated, "short output should not be truncated")

    large_output = "a" * (core.MAX_TOOL_OUTPUT_BYTES + 256)
    truncated_output, was_truncated = tools.truncate_tool_output(large_output)
    assert_true(was_truncated, "large output should be truncated")
    assert_true(
        len(truncated_output.encode("utf-8")) <= core.MAX_TOOL_OUTPUT_BYTES,
        "truncated output should respect the byte limit",
    )
    assert_true(
        truncated_output.endswith(
            f"... [truncated due to max output bytes limit of {core.MAX_TOOL_OUTPUT_BYTES}]"
        ),
        "truncated output should include a clear truncation marker",
    )


def test_truncate_bash_output_for_model() -> None:
    passthrough, passthrough_truncated = tools.truncate_bash_output_for_model(
        "hello world\n"
    )
    assert_equal(passthrough, "hello world\n")
    assert_true(not passthrough_truncated, "short bash output should not be truncated")

    lines = [f"line {index}: {'x' * 40}" for index in range(5000)]
    large_output = "\n".join(lines) + "\n"
    truncated_output, was_truncated = tools.truncate_bash_output_for_model(large_output)

    assert_true(was_truncated, "large bash output should be truncated")
    assert_true(
        truncated_output.startswith(f"Total output lines: {len(lines)}\n\nline 0:"),
        "truncated bash output should report total lines and preserve the head",
    )
    assert_true(
        "tokens truncated" in truncated_output,
        "truncated bash output should include the token truncation marker",
    )
    assert_true(
        f"line {len(lines) - 1}:" in truncated_output,
        "truncated bash output should preserve the tail",
    )
    assert_true(
        "line 2500:" not in truncated_output,
        "truncated bash output should drop the middle of very large output",
    )


def test_select_tool_names_defaults() -> None:
    assert_equal(
        tools.select_tool_names("gpt-4.1", "copilot"),
        core.DEFAULT_CHAT_TOOL_NAMES,
    )
    assert_equal(
        tools.select_tool_names("claude-sonnet", "copilot"),
        core.DEFAULT_CHAT_TOOL_NAMES,
    )
    assert_equal(
        tools.select_tool_names("gpt-5", "openai"),
        core.DEFAULT_RESPONSE_TOOL_NAMES,
    )


def test_select_tool_names_allowed_override() -> None:
    override = tools.normalize_allowed_tools("bash,read_file,grep")
    assert_equal(tools.select_tool_names("gpt-5", "openai", override), override)


def test_file_tool_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        path = cwd / "demo.txt"
        created = tools.execute_write_file(
            str(path), "alpha\nbeta\n", cwd, last_read_time=None
        )
        assert_true(created["success"], "write_file should create a file")

        read = tools.execute_read_file(str(path), cwd)
        assert_true("1 | alpha" in read["content"], "read_file should number lines")
        last_read = _file_mtime(path)

        edited = tools.execute_edit_file(
            str(path),
            "beta",
            "gamma",
            cwd,
            replace_all=False,
            last_read_time=last_read,
        )
        assert_true(edited["success"], "edit_file should edit the file")
        assert_equal(path.read_text(), "alpha\ngamma\n")


def test_search_tools_roundtrip() -> None:
    fd_path = shutil.which("fd")
    rg_path = shutil.which("rg")
    if not fd_path or not rg_path:
        return
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        src = cwd / "src"
        src.mkdir()
        target = src / "demo.py"
        target.write_text("def hello():\n    return 'hi'\n")

        glob_result = tools.execute_glob("**/*.py", cwd, path=str(cwd))
        assert_true(str(target) in glob_result["content"], "glob should find file")

        grep_result = tools.execute_grep("hello", cwd, path=str(cwd), include="*.py")
        assert_true(
            str(target) in grep_result["content"],
            "grep should report matching file",
        )
