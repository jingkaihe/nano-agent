from __future__ import annotations

from typing import Any, cast

import nano_agent.core as core
from nano_agent.tools._output_impl import truncate_bash_output_for_model, truncate_tool_output
from nano_agent.tools._stream_impl import build_anthropic_messages
from nano_agent.tools._tool_results_impl import parse_tool_call_arguments

from .helpers import assert_equal, assert_true


def test_parse_tool_call_arguments() -> None:
    assert_equal(parse_tool_call_arguments(None), {})
    assert_equal(parse_tool_call_arguments("{}"), {})
    assert_equal(parse_tool_call_arguments('{"x": 1}'), {"x": 1})
    try:
        parse_tool_call_arguments("[]")
    except ValueError:
        return
    raise AssertionError("parse_tool_call_arguments should reject non-object JSON")


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
        isinstance(assistant_blocks, list),
        "assistant content should be block list",
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
        cached_messages[1]["content"][-1]["cache_control"]["type"],
        "ephemeral",
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


def test_truncate_tool_output() -> None:
    passthrough, passthrough_truncated = truncate_tool_output("hello world")
    assert_equal(passthrough, "hello world")
    assert_true(not passthrough_truncated, "short output should not be truncated")

    large_output = "a" * (core.MAX_TOOL_OUTPUT_BYTES + 256)
    truncated_output, was_truncated = truncate_tool_output(large_output)
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
    passthrough, passthrough_truncated = truncate_bash_output_for_model("hello world\n")
    assert_equal(passthrough, "hello world\n")
    assert_true(not passthrough_truncated, "short bash output should not be truncated")

    lines = [f"line {index}: {'x' * 40}" for index in range(5000)]
    large_output = "\n".join(lines) + "\n"
    truncated_output, was_truncated = truncate_bash_output_for_model(large_output)

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
