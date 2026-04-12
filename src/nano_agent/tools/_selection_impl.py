from __future__ import annotations

from ._common import *


def normalize_allowed_tools(value: Any) -> list[str] | None:
    if value is None:
        return None

    items: list[str]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    else:
        raise ValueError("allowed_tools must be a comma-separated string or list")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        if item not in ALL_TOOL_NAMES:
            available = ", ".join(ALL_TOOL_NAMES)
            raise ValueError(f"unknown tool: {item}. Available tools: {available}")
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def select_tool_names(
    model: str, provider: str, allowed_tools: list[str] | None = None
) -> list[str]:
    if allowed_tools is not None:
        return list(allowed_tools)
    if should_use_anthropic_messages_api(provider, model):
        return list(DEFAULT_CHAT_TOOL_NAMES)
    if should_use_responses_api(model):
        return list(DEFAULT_RESPONSE_TOOL_NAMES)
    return list(DEFAULT_CHAT_TOOL_NAMES)


def tool_guidance_text(
    model: str, provider: str, allowed_tools: list[str] | None = None
) -> str:
    tool_names = set(select_tool_names(model, provider, allowed_tools))
    search_text = (
        "For filesystem search activities, prefer the `glob` and `grep` tools."
        if {"grep", "glob"}.issubset(tool_names)
        else "For filesystem search activities, use `fd` for file discovery and `rg` for content search via the `bash` tool only."
    )
    edit_text = (
        "Use `read_file`, `write_file`, and `edit_file` for file changes instead of `apply_patch`."
        if {"read_file", "write_file", "edit_file"}.issubset(tool_names)
        else "Use `apply_patch` for file edits."
    )
    return f"{search_text}\n{edit_text}"
