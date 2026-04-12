from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._common import GLOB_MAX_RESULTS, GREP_MAX_LINE_LENGTH, GREP_MAX_OUTPUT_BYTES, GREP_MAX_RESULTS, JINJA, SEARCH_TOOL_TIMEOUT_SECONDS
from .base import ToolCallContext, ToolDefinition
from .files import _ensure_absolute_path
from ._output_impl import truncate_tool_output


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="The pattern to search for")
    path: str | None = Field(
        default=None, description="Absolute file or directory path to search in"
    )
    include: str | None = Field(
        default=None,
        description="Optional glob to filter files when searching a directory",
    )
    ignore_case: bool | None = Field(
        default=None, description="Case-insensitive search if true"
    )
    fixed_strings: bool | None = Field(
        default=None, description="Treat pattern as a literal string if true"
    )
    surround_lines: int | None = Field(
        default=None, description="Number of context lines before and after each match"
    )
    max_results: int | None = Field(
        default=None,
        description=f"Maximum number of files to return. Max: {GREP_MAX_RESULTS}",
    )


class GlobArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="The glob pattern to match files")
    path: str | None = Field(
        default=None, description="Absolute directory path to search in"
    )
    ignore_gitignore: bool | None = Field(
        default=None, description="If true, do not respect .gitignore rules"
    )


def grep_schema() -> dict[str, Any]:
    return GrepArgs.model_json_schema()


def glob_schema() -> dict[str, Any]:
    return GlobArgs.model_json_schema()


def grep_description() -> str:
    return JINJA.from_string(
        f"""Search for a pattern in the codebase using regex.

## Important Notes
* Prefer this tool over raw grep/egrep shell commands for content search.
* Files matching .gitignore patterns are automatically excluded by the underlying rg defaults.
* Hidden files/directories (starting with .) are skipped by default for directory searches.
* Results are sorted by modification time (newest first), returning at most {GREP_MAX_RESULTS} files by default.
* To get the best result, use `glob` first to narrow the files and then use this tool for targeted content search.

## Input
- pattern: The pattern to search for (regex by default, or literal string if fixed_strings is true)
- path: The absolute path to search in. Can be a directory or a single file. Defaults to the current working directory.
- include: Optional glob pattern to filter files, for example `*.go` or `*.{{go,py}}`
- ignore_case: If true, use case-insensitive search
- fixed_strings: If true, treat pattern as a literal string instead of regex
- surround_lines: Number of lines of context to show before and after each match
- max_results: Number of files to return results from (1-{GREP_MAX_RESULTS})
"""
    ).render()


def glob_description() -> str:
    return JINJA.from_string(
        f"""Find files matching a glob pattern in the filesystem.

## Important Notes
* By default, .gitignore patterns are respected.
* Hidden files/directories (starting with .) are excluded by default.
* Results return at most {GLOB_MAX_RESULTS} files sorted by modification time (newest first).
* This tool matches filenames, not file contents. For content search, use `grep`.

## Input
- pattern: The glob pattern to match files, for example `*.go`, `**/*.py`, or `cmd/*.ts`
- path: The absolute path to a directory to search in. Defaults to the current working directory.
- ignore_gitignore: If true, do not respect .gitignore rules.
"""
    ).render()


def _find_search_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise ValueError(f"{name} not found on PATH")
    return binary


def _validate_glob_path(path: str | None, cwd: Path) -> Path:
    resolved = cwd if not path else _ensure_absolute_path(cwd, path, field_name="path")
    if not resolved.exists():
        raise ValueError(f"invalid path {resolved!r}: does not exist")
    if not resolved.is_dir():
        raise ValueError(
            f"path {str(resolved)!r} is not a directory - glob searches directories, not individual files"
        )
    return resolved


def execute_glob(
    pattern: str,
    cwd: Path,
    *,
    path: str | None = None,
    ignore_gitignore: bool = False,
) -> dict[str, Any]:
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern is required")
    search_path = _validate_glob_path(path, cwd)
    fd_path = _find_search_binary("fd")
    args = [fd_path, "--glob", "--type", "f", "--absolute-path"]
    if ignore_gitignore:
        args.extend(["--no-ignore", "--hidden"])
    args.extend([pattern, str(search_path)])
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=SEARCH_TOOL_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode not in {0, 1}:
        stderr = (result.stderr or result.stdout or "").strip()
        raise ValueError(f"fd error: {stderr or 'unknown error'}")
    files = [line for line in result.stdout.splitlines() if line.strip()]
    files.sort(
        key=lambda item: Path(item).stat().st_mtime if Path(item).exists() else 0,
        reverse=True,
    )
    truncated = len(files) > GLOB_MAX_RESULTS
    limited = files[:GLOB_MAX_RESULTS]
    content = "\n".join(limited)
    if truncated:
        content += "\n\n[Results truncated to 100 files. Please refine your pattern to narrow down the results.]"
    return {
        "success": True,
        "pattern": pattern,
        "path": str(search_path),
        "truncated": truncated,
        "content": content,
        "files": limited,
    }


def execute_grep(
    pattern: str,
    cwd: Path,
    *,
    path: str | None = None,
    include: str | None = None,
    ignore_case: bool = False,
    fixed_strings: bool = False,
    surround_lines: int = 0,
    max_results: int = GREP_MAX_RESULTS,
) -> dict[str, Any]:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern is required")
    if max_results > GREP_MAX_RESULTS:
        raise ValueError(f"max_results cannot exceed {GREP_MAX_RESULTS}")
    search_path = cwd if not path else _ensure_absolute_path(cwd, path, field_name="path")
    if not search_path.exists():
        raise ValueError(f"invalid path {str(search_path)!r}")
    rg_path = _find_search_binary("rg")

    args = [
        rg_path,
        "--no-heading",
        "--line-number",
        "--with-filename",
        "--color",
        "never",
    ]
    if search_path.is_dir():
        args.extend(["--glob", "!.*"])
        try:
            args.extend(["--sort", "path"])
        except Exception:
            pass
    if ignore_case:
        args.append("-i")
    if fixed_strings:
        args.append("-F")
    if surround_lines > 0:
        args.extend(["-C", str(surround_lines)])
    if include and search_path.is_dir():
        args.extend(["-g", include])
    args.append(pattern)
    args.append(str(search_path))

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=SEARCH_TOOL_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode not in {0, 1}:
        stderr = (result.stderr or result.stdout or "").strip()
        raise ValueError(f"ripgrep error: {stderr or 'unknown error'}")
    if result.returncode == 1 or not result.stdout.strip():
        return {
            "success": True,
            "pattern": pattern,
            "path": str(search_path),
            "include": include or "",
            "truncated": False,
            "content": f"No matches found for pattern '{pattern}'",
        }

    file_blocks: dict[str, list[str]] = {}
    file_order: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^(.*?)([:\-])(\d+)([:\-])(.*)$", line)
        if not match:
            continue
        filename = match.group(1)
        if filename not in file_blocks:
            file_blocks[filename] = []
            file_order.append(filename)
        content = match.group(5)
        if len(content) > GREP_MAX_LINE_LENGTH:
            content = content[:GREP_MAX_LINE_LENGTH] + "... [truncated]"
        file_blocks[filename].append(f"{match.group(3)}{match.group(2)}{content}")

    ordered_files = sorted(
        file_order,
        key=lambda item: Path(item).stat().st_mtime if Path(item).exists() else 0,
        reverse=True,
    )
    truncated = False
    if len(ordered_files) > max_results:
        ordered_files = ordered_files[:max_results]
        truncated = True

    parts = [f"Search results for pattern '{pattern}':"]
    for filename in ordered_files:
        parts.append(f"\nPattern found in file {filename}:\n")
        parts.extend(file_blocks.get(filename, []))
    content = "\n".join(parts).strip()
    if len(content.encode("utf-8")) > GREP_MAX_OUTPUT_BYTES:
        content = truncate_tool_output(content, GREP_MAX_OUTPUT_BYTES)[0]
        truncated = True
    if truncated:
        content += (
            "\n\n[TRUNCATED DUE TO MAXIMUM 100 FILE LIMIT - refine your search pattern or use include filter]"
            if len(file_order) > max_results
            else ""
        )
    return {
        "success": True,
        "pattern": pattern,
        "path": str(search_path),
        "include": include or "",
        "truncated": truncated,
        "content": content,
    }


class GrepTool(ToolDefinition):
    name = "grep"

    def description(self, agent: Any) -> str:
        return grep_description()

    def schema(self, agent: Any) -> dict[str, Any]:
        return grep_schema()

    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        return execute_grep(
            str(context.arguments.get("pattern") or ""),
            context.cwd,
            path=context.arguments.get("path"),
            include=context.arguments.get("include"),
            ignore_case=bool(context.arguments.get("ignore_case", False)),
            fixed_strings=bool(context.arguments.get("fixed_strings", False)),
            surround_lines=int(context.arguments.get("surround_lines") or 0),
            max_results=int(context.arguments.get("max_results") or GREP_MAX_RESULTS),
        )


class GlobTool(ToolDefinition):
    name = "glob"

    def description(self, agent: Any) -> str:
        return glob_description()

    def schema(self, agent: Any) -> dict[str, Any]:
        return glob_schema()

    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        return execute_glob(
            str(context.arguments.get("pattern") or ""),
            context.cwd,
            path=context.arguments.get("path"),
            ignore_gitignore=bool(context.arguments.get("ignore_gitignore", False)),
        )


grep_tool = GrepTool()
glob_tool = GlobTool()

__all__ = [
    "GlobTool",
    "GrepTool",
    "execute_glob",
    "execute_grep",
    "glob_tool",
    "glob_description",
    "glob_schema",
    "grep_tool",
    "grep_description",
    "grep_schema",
]
