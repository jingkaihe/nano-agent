from __future__ import annotations

import json
from typing import Any


def tool_result_content(
    result: dict[str, Any], *, include_image_url: bool = True
) -> str:
    payload = dict(result)
    if not include_image_url and isinstance(payload.get("image_url"), str):
        payload["image_url"] = "[omitted data URL]"
    return json.dumps(payload, ensure_ascii=False)


def tool_result_content_parts(result: dict[str, Any]) -> list[dict[str, Any]] | None:
    if (
        result.get("success")
        and isinstance(result.get("image_url"), str)
        and result.get("image_url")
    ):
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": result["image_url"],
        }
        if result.get("detail") == "original":
            image_part["detail"] = "original"
        return [image_part]
    return None


def chat_tool_message_content(result: dict[str, Any]) -> str | list[dict[str, Any]]:
    return tool_result_content(result, include_image_url=False)


def responses_function_call_output(
    result: dict[str, Any],
) -> str | list[dict[str, Any]]:
    parts = tool_result_content_parts(result)
    if parts is not None:
        return parts
    return tool_result_content(result)


def _copilot_cache_control(value: dict[str, Any]) -> dict[str, Any] | None:
    existing = value.get("copilot_cache_control")
    if not isinstance(existing, dict):
        return None
    return dict(existing)


def _apply_copilot_cache_control_to_block(
    block: dict[str, Any], cache_control: dict[str, Any] | None
) -> dict[str, Any]:
    if cache_control is None:
        return block
    updated = dict(block)
    existing = updated.get("copilot_cache_control")
    updated["copilot_cache_control"] = {
        **(existing if isinstance(existing, dict) else {}),
        **cache_control,
    }
    return updated


def _anthropic_block_from_internal_block(block: dict[str, Any]) -> dict[str, Any]:
    updated = dict(block)
    existing = updated.pop("copilot_cache_control", None)
    if isinstance(existing, dict):
        existing_cache_control = updated.get("cache_control")
        current_cache_control: dict[str, Any] = (
            dict(existing_cache_control)
            if isinstance(existing_cache_control, dict)
            else {}
        )
        updated["cache_control"] = {
            **current_cache_control,
            **existing,
        }
    return updated


def _apply_copilot_cache_control_to_anthropic_block(
    block: dict[str, Any], cache_control: dict[str, Any] | None
) -> dict[str, Any]:
    updated = _anthropic_block_from_internal_block(block)
    if cache_control is None:
        return updated
    existing = updated.get("cache_control")
    updated["cache_control"] = {
        **(existing if isinstance(existing, dict) else {}),
        **cache_control,
    }
    return updated


def _apply_message_copilot_cache_control_to_last_anthropic_block(
    blocks: list[dict[str, Any]], message: dict[str, Any]
) -> list[dict[str, Any]]:
    updated = [_anthropic_block_from_internal_block(block) for block in blocks]
    cache_control = _copilot_cache_control(message)
    if cache_control is None or not updated:
        return updated
    updated[-1] = _apply_copilot_cache_control_to_anthropic_block(
        updated[-1], cache_control
    )
    return updated
