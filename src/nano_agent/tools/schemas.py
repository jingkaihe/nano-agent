from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._common import GREP_MAX_RESULTS, MAX_READ_LINE_LIMIT
from ._images_impl import supports_view_image_original_detail

class BashArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(description="The bash command to run")
    description: str = Field(description="A description of the command to run")
    timeout: int = Field(description="Timeout in seconds (10-120)")


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(description="The absolute path of the file to read")
    offset: int | None = Field(default=None, description=f"The 1-indexed line number to start reading from. Default: 1")
    line_limit: int | None = Field(default=None, description=f"The maximum number of lines to read from the offset. Default: {MAX_READ_LINE_LIMIT}. Max: {MAX_READ_LINE_LIMIT}")


class WriteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(description="The absolute path of the file to write")
    text: str = Field(description="The text to write to the file")


class EditFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(description="The absolute path of the file to edit")
    old_text: str = Field(description="The text to be replaced")
    new_text: str = Field(description="The text to replace the old text with")
    replace_all: bool | None = Field(default=None, description="If true, replace all occurrences of old_text; otherwise old_text must be unique")


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="The pattern to search for")
    path: str | None = Field(default=None, description="Absolute file or directory path to search in")
    include: str | None = Field(default=None, description="Optional glob to filter files when searching a directory")
    ignore_case: bool | None = Field(default=None, description="Case-insensitive search if true")
    fixed_strings: bool | None = Field(default=None, description="Treat pattern as a literal string if true")
    surround_lines: int | None = Field(default=None, description="Number of context lines before and after each match")
    max_results: int | None = Field(default=None, description=f"Maximum number of files to return. Max: {GREP_MAX_RESULTS}")


class GlobArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="The glob pattern to match files")
    path: str | None = Field(default=None, description="Absolute directory path to search in")
    ignore_gitignore: bool | None = Field(default=None, description="If true, do not respect .gitignore rules")


class ApplyPatchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(description="The entire contents of the apply_patch command")


class SkillArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(description="The name of the skill to invoke")


class WebFetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="The URL to fetch content from")
    prompt: str | None = Field(default=None, description="Information to extract from HTML/Markdown content (optional)")


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


def bash_schema() -> dict[str, Any]:
    return _strict_schema(BashArgs)


def read_file_schema() -> dict[str, Any]:
    return _strict_schema(ReadFileArgs)


def write_file_schema() -> dict[str, Any]:
    return _strict_schema(WriteFileArgs)


def edit_file_schema() -> dict[str, Any]:
    return _strict_schema(EditFileArgs)


def grep_schema() -> dict[str, Any]:
    return _strict_schema(GrepArgs)


def glob_schema() -> dict[str, Any]:
    return _strict_schema(GlobArgs)


def apply_patch_schema() -> dict[str, Any]:
    return _strict_schema(ApplyPatchArgs)


def skill_schema() -> dict[str, Any]:
    return _strict_schema(SkillArgs)


def web_fetch_schema() -> dict[str, Any]:
    return _strict_schema(WebFetchArgs)

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
