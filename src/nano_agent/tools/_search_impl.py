from __future__ import annotations

from ._common import *
from ._files_impl import _ensure_absolute_path

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
    search_path = (
        cwd if not path else _ensure_absolute_path(cwd, path, field_name="path")
    )
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
