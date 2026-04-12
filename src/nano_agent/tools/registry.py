from __future__ import annotations

from .apply_patch import apply_patch_tool
from .base import ToolDefinition, function_tool_spec
from .bash import bash_tool
from .files import edit_file_tool, read_file_tool, write_file_tool
from .search import glob_tool, grep_tool
from .skill import skill_tool
from .view_image import view_image_tool
from .web_fetch import web_fetch_tool


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    bash_tool,
    apply_patch_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    grep_tool,
    glob_tool,
    skill_tool,
    web_fetch_tool,
    view_image_tool,
)

TOOL_DEFINITIONS_BY_NAME: dict[str, ToolDefinition] = {
    tool.name: tool for tool in TOOL_DEFINITIONS
}


def tool_definition(name: str) -> ToolDefinition:
    try:
        return TOOL_DEFINITIONS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown tool: {name}") from exc


def tool_spec(name: str, agent: object) -> dict[str, object]:
    return function_tool_spec(tool_definition(name), agent)
