from __future__ import annotations

from ._common import *
from .results import chat_tool_message_content, tool_result_content, tool_result_content_parts


def _tool_error_from_json(exc: json.JSONDecodeError) -> dict[str, Any]:
    return {"success": False, "error": f"invalid JSON arguments: {exc}"}


def parse_tool_call_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to an object")
    return parsed


def _tool_result_message_content(
    result: dict[str, Any], *, role: str, api: str
) -> str | list[dict[str, Any]]:
    if role == "tool":
        if api == "chat.completions":
            return chat_tool_message_content(result)
        if api == "messages":
            parts = tool_result_content_parts(result)
            if parts is not None:
                return parts
    return tool_result_content(result)


def chat_followup_image_message(result: dict[str, Any]) -> dict[str, Any] | None:
    image_url = result.get("image_url")
    if not (result.get("success") and isinstance(image_url, str) and image_url):
        return None
    detail = "high" if result.get("detail") == "original" else "auto"
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": detail},
            }
        ],
    }
