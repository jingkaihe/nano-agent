from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import ToolCallContext, ToolDefinition
from .descriptions import apply_patch_description
from .schemas import apply_patch_schema


@dataclass
class Hunk:
    header: str
    lines: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class FilePatch:
    op: str
    path: str
    move_to: str | None = None
    add_lines: list[str] = field(default_factory=list)
    hunks: list[Hunk] = field(default_factory=list)


def _validate_patch_path(path: str) -> str:
    if not path:
        raise ValueError("File path is required")
    return path


def resolve_patch_path(cwd: Path, patch_path: str) -> Path:
    path = Path(patch_path)
    if path.is_absolute():
        return path
    return (cwd / path).resolve()


def parse_apply_patch(patch_text: str) -> list[FilePatch]:
    lines = patch_text.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError(
            "Patch must start with '*** Begin Patch' and end with '*** End Patch'"
        )

    ops: list[FilePatch] = []
    i = 1
    while i < len(lines) - 1:
        line = lines[i]
        if line.startswith("*** Add File: "):
            path = _validate_patch_path(line.removeprefix("*** Add File: ").strip())
            i += 1
            add_lines: list[str] = []
            while i < len(lines) - 1 and not lines[i].startswith("*** "):
                current = lines[i]
                if not current.startswith("+"):
                    raise ValueError(f"Add File lines must start with '+': {current}")
                add_lines.append(current[1:])
                i += 1
            ops.append(FilePatch(op="add", path=path, add_lines=add_lines))
            continue
        if line.startswith("*** Delete File: "):
            path = _validate_patch_path(line.removeprefix("*** Delete File: ").strip())
            ops.append(FilePatch(op="delete", path=path))
            i += 1
            continue
        if line.startswith("*** Update File: "):
            path = _validate_patch_path(line.removeprefix("*** Update File: ").strip())
            patch = FilePatch(op="update", path=path)
            i += 1
            if i < len(lines) - 1 and lines[i].startswith("*** Move to: "):
                patch.move_to = _validate_patch_path(
                    lines[i].removeprefix("*** Move to: ").strip()
                )
                i += 1
            while i < len(lines) - 1 and not lines[i].startswith("*** "):
                if not lines[i].startswith("@@"):
                    raise ValueError(f"Expected hunk header, got: {lines[i]}")
                hunk = Hunk(header=lines[i][2:].strip())
                i += 1
                while (
                    i < len(lines) - 1
                    and not lines[i].startswith("@@")
                    and not lines[i].startswith("*** ")
                ):
                    if lines[i] == "*** End of File":
                        i += 1
                        continue
                    prefix = lines[i][:1]
                    if prefix not in {" ", "+", "-"}:
                        raise ValueError(f"Invalid hunk line: {lines[i]}")
                    hunk.lines.append((prefix, lines[i][1:]))
                    i += 1
                patch.hunks.append(hunk)
            ops.append(patch)
            continue
        if not line.strip():
            i += 1
            continue
        raise ValueError(f"Unknown patch operation: {line}")

    return ops


def _find_hunk_position(source_lines: list[str], needle: list[str], start: int) -> int | None:
    if not needle:
        return start
    max_start = len(source_lines) - len(needle)
    for candidate in range(start, max_start + 1):
        if source_lines[candidate : candidate + len(needle)] == needle:
            return candidate
    for candidate in range(0, max_start + 1):
        if source_lines[candidate : candidate + len(needle)] == needle:
            return candidate
    stripped_needle = [line.rstrip() for line in needle]
    for candidate in range(start, max_start + 1):
        if [line.rstrip() for line in source_lines[candidate : candidate + len(needle)]] == stripped_needle:
            return candidate
    return None


def apply_hunks_to_text(original: str, hunks: list[Hunk], path: str) -> str:
    newline = "\r\n" if "\r\n" in original else "\n"
    source_lines = original.splitlines()
    output: list[str] = []
    cursor = 0
    for hunk in hunks:
        source_fragment = [text for prefix, text in hunk.lines if prefix in {" ", "-"}]
        target_fragment = [text for prefix, text in hunk.lines if prefix in {" ", "+"}]
        pos = _find_hunk_position(source_lines, source_fragment, cursor)
        if pos is None:
            raise ValueError(f"Failed to apply hunk to {path}: context not found")
        output.extend(source_lines[cursor:pos])
        output.extend(target_fragment)
        cursor = pos + len(source_fragment)
    output.extend(source_lines[cursor:])
    result = newline.join(output)
    if original.endswith(("\n", "\r\n")) and output:
        result += newline
    return result


def _unified_diff(old: str, new: str, fromfile: str, tofile: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(diff)


def execute_apply_patch(patch_text: str, cwd: Path) -> dict[str, Any]:
    ops = parse_apply_patch(patch_text)
    changes: list[dict[str, Any]] = []
    diffs: list[str] = []

    for op in ops:
        target = resolve_patch_path(cwd, op.path)
        if op.op == "add":
            target.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(op.add_lines)
            if op.add_lines:
                content += "\n"
            target.write_text(content)
            changes.append({"action": "add", "path": str(target)})
            diffs.append(_unified_diff("", content, f"a/{op.path}", f"b/{op.path}"))
            continue

        if op.op == "delete":
            if not target.exists():
                raise ValueError(f"Cannot delete missing file: {op.path}")
            old = target.read_text(errors="replace")
            target.unlink()
            changes.append({"action": "delete", "path": str(target)})
            diffs.append(_unified_diff(old, "", f"a/{op.path}", f"b/{op.path}"))
            continue

        if op.op == "update":
            if not target.exists():
                raise ValueError(f"Cannot update missing file: {op.path}")
            old = target.read_text(errors="replace")
            new = apply_hunks_to_text(old, op.hunks, op.path)
            write_target = resolve_patch_path(cwd, op.move_to or op.path)
            write_target.parent.mkdir(parents=True, exist_ok=True)
            write_target.write_text(new)
            if op.move_to and write_target.resolve() != target.resolve():
                target.unlink()
            changes.append(
                {
                    "action": "update",
                    "path": str(target),
                    "move_to": str(write_target) if op.move_to else None,
                }
            )
            diffs.append(_unified_diff(old, new, f"a/{op.path}", f"b/{op.move_to or op.path}"))
            continue

    return {
        "success": True,
        "summary": f"Applied patch to {len(changes)} file(s)",
        "changes": changes,
        "diff": "\n".join(diff for diff in diffs if diff).strip(),
    }


class ApplyPatchTool(ToolDefinition):
    name = "apply_patch"

    def description(self, agent: Any) -> str:
        return apply_patch_description()

    def schema(self, agent: Any) -> dict[str, Any]:
        return apply_patch_schema()

    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        patch_input = context.arguments.get("input")
        if not isinstance(patch_input, str) or not patch_input.strip():
            raise ValueError("input is required")
        return execute_apply_patch(patch_input, context.cwd)


apply_patch_tool = ApplyPatchTool()

__all__ = [
    "ApplyPatchTool",
    "apply_patch_description",
    "apply_patch_schema",
    "apply_patch_tool",
    "execute_apply_patch",
    "parse_apply_patch",
]
