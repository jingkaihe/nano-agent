from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

from PIL import Image

import nano_agent.core as core
from nano_agent.tools._images_impl import _data_url_to_anthropic_source
from nano_agent.tools._tool_results_impl import chat_followup_image_message
from nano_agent.tools.results import chat_tool_message_content, responses_function_call_output
from nano_agent.tools.view_image import make_view_image_result, view_image_schema

from .helpers import assert_equal, assert_true


def test_view_image_schema_and_helpers() -> None:
    basic_schema = view_image_schema("gpt-5")
    assert_true(
        "detail" not in cast(dict[str, Any], basic_schema["properties"]),
        "non-supported models should omit detail from view_image schema",
    )

    codex_schema = view_image_schema("gpt-5.3-codex")
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
    response_output = responses_function_call_output(image_result)
    assert_true(
        isinstance(response_output, list),
        "responses output should allow image parts",
    )
    response_parts = cast(list[dict[str, Any]], response_output)
    assert_equal(response_parts[0]["type"], "input_image")
    assert_equal(response_parts[0]["detail"], "original")

    chat_content = chat_tool_message_content(image_result)
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

        resized = make_view_image_result(
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

        original = make_view_image_result(
            image_path,
            detail="original",
            model="gpt-5.3-codex",
            provider="openai",
        )
        assert_equal(original["width"], 2304)
        assert_equal(original["height"], 864)
        assert_equal(original["detail"], "original")

        try:
            make_view_image_result(
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
            make_view_image_result(
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
