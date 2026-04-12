from __future__ import annotations

from ._common import *
from ._web_fetch_impl import with_numbered_lines

def _ensure_absolute_path(cwd: Path, raw_path: str, *, field_name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field_name} is required")
    path = Path(raw_path.strip())
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    return path.resolve()


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime_ns / 1_000_000_000


def _truncate_line_for_read(line: str) -> str:
    if len(line) <= MAX_READ_LINE_CHARACTERS:
        return line
    return line[:MAX_READ_LINE_CHARACTERS] + "..."


def execute_read_file(
    file_path: str,
    cwd: Path,
    *,
    offset: int = 1,
    line_limit: int = MAX_READ_LINE_LIMIT,
) -> dict[str, Any]:
    path = _ensure_absolute_path(cwd, file_path, field_name="file_path")
    if offset < 0:
        raise ValueError("offset must be a positive integer")
    if line_limit == 0:
        line_limit = MAX_READ_LINE_LIMIT
    if line_limit < 0:
        raise ValueError("line_limit must be a positive integer")
    if line_limit > MAX_READ_LINE_LIMIT:
        raise ValueError(f"line_limit cannot exceed {MAX_READ_LINE_LIMIT}")

    start_line = 1 if offset == 0 else offset
    lines = path.read_text(errors="replace").splitlines()
    total_lines = len(lines)
    if start_line > total_lines and start_line != 1:
        raise ValueError(
            f"File has only {total_lines} lines, which is less than the requested offset {start_line}"
        )

    selected: list[str] = []
    bytes_read = 0
    end_index = min(total_lines, start_line - 1 + line_limit)
    current_index = start_line - 1
    while current_index < end_index:
        line = _truncate_line_for_read(lines[current_index])
        line_bytes = len(line.encode("utf-8"))
        if bytes_read + line_bytes > MAX_READ_OUTPUT_BYTES:
            break
        selected.append(line)
        bytes_read += line_bytes
        current_index += 1

    remaining_lines = total_lines - current_index
    if current_index < end_index and remaining_lines > 0:
        selected.append(
            f"... [truncated due to max output bytes limit of {MAX_READ_OUTPUT_BYTES}]"
        )
    elif current_index == end_index and remaining_lines > 0:
        selected.append(
            f"... [{remaining_lines} lines remaining - use offset={current_index + 1} to continue reading]"
        )

    return {
        "success": True,
        "file_path": str(path),
        "offset": start_line,
        "line_limit": line_limit,
        "remaining_lines": max(remaining_lines, 0),
        "content": with_numbered_lines(selected, start_line),
    }


def execute_write_file(
    file_path: str,
    text: str,
    cwd: Path,
    *,
    last_read_time: float | None,
) -> dict[str, Any]:
    path = _ensure_absolute_path(cwd, file_path, field_name="file_path")
    if not isinstance(text, str) or text == "":
        raise ValueError(
            "text is required. run 'touch' command to create an empty file"
        )
    if (
        path.exists()
        and last_read_time is not None
        and _file_mtime(path) > last_read_time
    ):
        raise ValueError(
            f"file {path} has been modified since the last read either by another tool or by the user, please read the file again"
        )
    path.write_text(text)
    preview_lines = text.splitlines()
    return {
        "success": True,
        "file_path": str(path),
        "content": f"file {path} has been written successfully\n\n{with_numbered_lines(preview_lines, 0)}"
        if preview_lines
        else f"file {path} has been written successfully",
    }


def _find_line_numbers(content: str, old_text: str) -> tuple[int, int]:
    lines = content.split("\n")
    old_lines = old_text.split("\n")
    for index in range(0, max(0, len(lines) - len(old_lines)) + 1):
        if lines[index : index + len(old_lines)] == old_lines:
            start_line = index + 1
            return start_line, start_line + len(old_lines) - 1
    before = content.split(old_text, 1)[0]
    start_line = before.count("\n") + 1
    return start_line, start_line + old_text.count("\n")


def execute_edit_file(
    file_path: str,
    old_text: str,
    new_text: str,
    cwd: Path,
    *,
    replace_all: bool = False,
    last_read_time: float | None,
) -> dict[str, Any]:
    path = _ensure_absolute_path(cwd, file_path, field_name="file_path")
    if not path.exists():
        raise ValueError(
            f"file {path} does not exist, use the 'write_file' tool to create instead"
        )
    if not isinstance(old_text, str) or old_text == "":
        raise ValueError("old_text is required")
    if not isinstance(new_text, str):
        raise ValueError("new_text must be a string")

    original = path.read_text(errors="replace")
    if old_text not in original:
        raise ValueError(
            "old text not found in the file, please ensure the text exists"
        )
    occurrences = original.count(old_text)
    if not replace_all and occurrences > 1:
        raise ValueError(
            f"old text appears {occurrences} times in the file, please ensure the old text is unique or set replace_all to true"
        )
    if not replace_all:
        if last_read_time is None:
            raise ValueError("failed to get the last access time of the file")
        if _file_mtime(path) > last_read_time:
            raise ValueError(
                f"file {path} has been modified since the last read either by another tool or by the user, please read the file again"
            )

    start_line, end_line = _find_line_numbers(original, old_text)
    if replace_all:
        content = original.replace(old_text, new_text)
        replaced_count = occurrences
        summary = f"File {path} has been edited successfully. Replaced {replaced_count} occurrences"
    else:
        content = original.replace(old_text, new_text, 1)
        replaced_count = 1
        summary = f"File {path} has been edited successfully"
    path.write_text(content)

    edited_block = (
        with_numbered_lines(new_text.splitlines(), start_line) if new_text else ""
    )
    return {
        "success": True,
        "file_path": str(path),
        "replace_all": replace_all,
        "replaced_count": replaced_count,
        "content": (
            summary
            if replace_all and replaced_count > 1
            else f"{summary}\n\nEdited code block:\n{edited_block}".rstrip()
        ),
        "start_line": start_line,
        "end_line": end_line,
    }
