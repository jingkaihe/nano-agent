from __future__ import annotations

from ._common import *
from ._images_impl import _data_url_to_anthropic_source
from .results import (
    _anthropic_block_from_internal_block,
    _apply_copilot_cache_control_to_anthropic_block,
    _apply_message_copilot_cache_control_to_last_anthropic_block,
    _copilot_cache_control,
)

def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return value


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _extract_usage_metrics(value: Any) -> dict[str, int] | None:
    plain = _to_plain_data(value)
    if not isinstance(plain, dict):
        return None

    if "input_tokens" in plain or "output_tokens" in plain:
        input_tokens = _coerce_usage_int(plain.get("input_tokens"))
        output_tokens = _coerce_usage_int(plain.get("output_tokens"))
        total_tokens = _coerce_usage_int(plain.get("total_tokens")) or (
            input_tokens + output_tokens
        )
        raw_input_details = plain.get("input_tokens_details")
        input_details: dict[str, Any] = (
            raw_input_details if isinstance(raw_input_details, dict) else {}
        )
        raw_output_details = plain.get("output_tokens_details")
        output_details: dict[str, Any] = (
            raw_output_details if isinstance(raw_output_details, dict) else {}
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": _coerce_usage_int(
                input_details.get("cached_tokens")
            ),
            "reasoning_tokens": _coerce_usage_int(
                output_details.get("reasoning_tokens")
            ),
        }

    if "prompt_tokens" in plain or "completion_tokens" in plain:
        input_tokens = _coerce_usage_int(plain.get("prompt_tokens"))
        output_tokens = _coerce_usage_int(plain.get("completion_tokens"))
        total_tokens = _coerce_usage_int(plain.get("total_tokens")) or (
            input_tokens + output_tokens
        )
        raw_prompt_details = plain.get("prompt_tokens_details")
        prompt_details: dict[str, Any] = (
            raw_prompt_details if isinstance(raw_prompt_details, dict) else {}
        )
        raw_completion_details = plain.get("completion_tokens_details")
        completion_details: dict[str, Any] = (
            raw_completion_details if isinstance(raw_completion_details, dict) else {}
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": _coerce_usage_int(
                prompt_details.get("cached_tokens")
            ),
            "reasoning_tokens": _coerce_usage_int(
                completion_details.get("reasoning_tokens")
            ),
        }

    return None


def _extract_text_fragments(value: Any) -> list[str]:
    plain = _to_plain_data(value)
    if plain is None:
        return []
    if isinstance(plain, str):
        return [plain]
    if isinstance(plain, list):
        fragments: list[str] = []
        for item in plain:
            fragments.extend(_extract_text_fragments(item))
        return fragments
    if isinstance(plain, dict):
        fragments: list[str] = []
        for key in ("text", "content", "value"):
            if isinstance(plain.get(key), str):
                fragments.append(plain[key])
        return fragments
    return []


def _get_delta_fragments(delta: Any, keys: list[str]) -> list[str]:
    plain = _to_plain_data(delta)
    if not isinstance(plain, dict):
        return []
    fragments: list[str] = []
    for key in keys:
        fragments.extend(_extract_text_fragments(plain.get(key)))
    return fragments


def _merge_stream_tool_call(
    accumulator: dict[int, dict[str, Any]], tool_call: Any
) -> None:
    plain = _to_plain_data(tool_call)
    if not isinstance(plain, dict):
        return
    index = plain.get("index", 0)
    current = accumulator.setdefault(
        int(index),
        {
            "id": plain.get("id"),
            "type": plain.get("type", "function"),
            "function": {"name": "", "arguments": ""},
        },
    )
    if plain.get("id"):
        current["id"] = plain["id"]
    if plain.get("type"):
        current["type"] = plain["type"]
    function = plain.get("function") or {}
    if isinstance(function, dict):
        if function.get("name"):
            current["function"]["name"] += function["name"]
        if function.get("arguments"):
            current["function"]["arguments"] += function["arguments"]


def _ordered_tool_calls(accumulator: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [accumulator[index] for index in sorted(accumulator)]


def chat_tools_to_responses_tools(
    chat_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in chat_tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        tools.append(
            {
                "type": "function",
                "name": function.get("name") or "",
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {},
            }
        )
    return tools


def chat_tools_to_anthropic_tools(
    chat_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in chat_tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        tools.append(
            {
                "name": function.get("name") or "",
                "description": function.get("description") or "",
                "input_schema": function.get("parameters") or {},
            }
        )
    return tools


def anthropic_thinking_config(reasoning_effort: str | None) -> dict[str, Any] | None:
    if not reasoning_effort or reasoning_effort == "none":
        return None
    normalized = reasoning_effort.strip().lower()
    if normalized in {"minimal", "low"}:
        return {"type": "enabled", "budget_tokens": 2048}
    if normalized == "medium":
        return {"type": "enabled", "budget_tokens": 4096}
    if normalized == "high":
        return {"type": "enabled", "budget_tokens": 8192}
    return {"type": "adaptive"}


def build_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    system_blocks: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                system_blocks.append(
                    _apply_copilot_cache_control_to_anthropic_block(
                        {"type": "text", "text": content},
                        _copilot_cache_control(message),
                    )
                )
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        system_blocks.append(_anthropic_block_from_internal_block(item))
            continue

        if role == "user":
            content = message.get("content")
            if isinstance(content, str):
                cache_control = _copilot_cache_control(message)
                if cache_control is None:
                    anthropic_messages.append({"role": "user", "content": content})
                else:
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": [
                                _apply_copilot_cache_control_to_anthropic_block(
                                    {"type": "text", "text": content},
                                    cache_control,
                                )
                            ],
                        }
                    )
            elif isinstance(content, list):
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            (
                                _anthropic_block_from_internal_block(item)
                                if isinstance(item, dict)
                                else item
                            )
                            for item in content
                        ],
                    }
                )
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            content = message.get("content")
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        blocks.append(_anthropic_block_from_internal_block(item))

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    raw_function = call.get("function")
                    function: dict[str, Any] = (
                        raw_function if isinstance(raw_function, dict) else {}
                    )
                    arguments = function.get("arguments")
                    try:
                        parsed_input = (
                            json.loads(arguments)
                            if isinstance(arguments, str) and arguments
                            else {}
                        )
                    except json.JSONDecodeError:
                        parsed_input = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id") or "tool-use",
                            "name": function.get("name") or "",
                            "input": parsed_input
                            if isinstance(parsed_input, dict)
                            else {},
                        }
                    )

            blocks = _apply_message_copilot_cache_control_to_last_anthropic_block(
                blocks, message
            )
            anthropic_messages.append({"role": "assistant", "content": blocks or ""})
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            content = message.get("content")
            if isinstance(content, list):
                tool_result_blocks: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "input_image":
                        image_url = item.get("image_url")
                        if isinstance(image_url, str) and image_url:
                            tool_result_blocks.append(
                                {
                                    "type": "image",
                                    "source": _data_url_to_anthropic_source(image_url),
                                }
                            )
                    elif item_type == "text":
                        text = item.get("text")
                        if isinstance(text, str):
                            tool_result_blocks.append({"type": "text", "text": text})
                if tool_result_blocks:
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": [
                                _apply_copilot_cache_control_to_anthropic_block(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_call_id or "",
                                        "content": tool_result_blocks,
                                    },
                                    _copilot_cache_control(message),
                                )
                            ],
                        }
                    )
                    continue
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        _apply_copilot_cache_control_to_anthropic_block(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id or "",
                                "content": content
                                if isinstance(content, str)
                                else json.dumps(content, ensure_ascii=False),
                            },
                            _copilot_cache_control(message),
                        )
                    ],
                }
            )

    if not system_blocks:
        return None, anthropic_messages
    if len(system_blocks) == 1 and set(system_blocks[0].keys()) <= {"type", "text"}:
        return system_blocks[0]["text"], anthropic_messages
    return system_blocks, anthropic_messages
