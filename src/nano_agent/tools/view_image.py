from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ToolCallContext, ToolDefinition
from ._common import JINJA
from ._images_impl import (
    _data_url_to_anthropic_source,
    make_view_image_result,
    supports_image_inputs,
    supports_view_image_original_detail,
)
from ._tool_results_impl import chat_followup_image_message
from .results import (
    chat_tool_message_content,
    responses_function_call_output,
    tool_result_content_parts,
)


def view_image_schema(model: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "path": {
            "type": "string",
            "description": "Local filesystem path to an image file. Absolute paths are preferred.",
        }
    }
    if supports_view_image_original_detail(model):
        properties["detail"] = {
            "type": "string",
            "description": "Optional detail override. The only supported value is `original`; omit this field for default resized behavior.",
        }
    return {
        "type": "object",
        "properties": properties,
        "required": ["path"],
        "additionalProperties": False,
    }


def view_image_description(model: str) -> str:
    detail_text = (
        "The optional `detail` field is available for this model and supports only `original`. Use it when high-fidelity image perception or precise localization is needed."
        if supports_view_image_original_detail(model)
        else "This model does not support the optional `detail` field; omit it."
    )
    return JINJA.from_string(
        "View a local image from the filesystem (only use if given a full filepath by the user, and the image isn't already attached in the conversation context).\n\n{{ detail_text }}"
    ).render(detail_text=detail_text)


class ViewImageTool(ToolDefinition):
    name = "view_image"

    def description(self, agent: Any) -> str:
        return view_image_description(agent.model)

    def schema(self, agent: Any) -> dict[str, Any]:
        return view_image_schema(agent.model)

    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        raw_path = context.arguments.get("path")
        detail = context.arguments.get("detail")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("path is required")
        if detail is not None and not isinstance(detail, str):
            raise ValueError("detail must be a string when provided")
        candidate = Path(raw_path.strip())
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (context.cwd / candidate).resolve()
        )
        return make_view_image_result(
            resolved,
            detail=detail,
            model=context.model,
            provider=context.provider,
        )


view_image_tool = ViewImageTool()

__all__ = [
    "_data_url_to_anthropic_source",
    "chat_followup_image_message",
    "chat_tool_message_content",
    "make_view_image_result",
    "responses_function_call_output",
    "supports_image_inputs",
    "supports_view_image_original_detail",
    "tool_result_content_parts",
    "view_image_tool",
    "view_image_description",
    "view_image_schema",
]
