from __future__ import annotations

class BashRunner:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def close(self) -> None:
        return

    def run(self, command: str, timeout: int) -> tuple[int, str, bool]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            output, _truncated = truncate_bash_output_for_model(stdout + stderr)
            if output:
                raise TimeoutError(
                    f"command timed out after {timeout} seconds\n{output}"
                ) from exc
            raise TimeoutError(f"command timed out after {timeout} seconds") from exc

        output, truncated = truncate_bash_output_for_model(result.stdout)
        return result.returncode, output, truncated


def validate_bash_args(
    command: Any, description: Any, timeout: Any
) -> tuple[str, str, int]:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command is required")
    command = command.replace("\r\n", "\n").replace("\r", "\n")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")
    if not isinstance(timeout, int) or timeout < 10 or timeout > 120:
        raise ValueError("timeout must be between 10 and 120 seconds")

    # Keep bash validation intentionally shallow: enforce required fields and
    # banned top-level commands, but let bash itself handle shell syntax.
    # Deep parsing here caused false positives for quoted strings like
    # `"a && b"` and rejected multiline commands unnecessarily.
    for part in re.split(r"&&|\|\||[;&|\n]", command):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        if not tokens:
            continue
        if tokens[0] in BANNED_COMMANDS:
            raise ValueError(f"command is banned: {tokens[0]}")

    return command, description, timeout


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


def _find_hunk_position(
    source_lines: list[str], needle: list[str], start: int
) -> int | None:
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
        if [
            line.rstrip() for line in source_lines[candidate : candidate + len(needle)]
        ] == stripped_needle:
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
            diffs.append(
                _unified_diff(
                    old,
                    new,
                    f"a/{op.path}",
                    f"b/{op.move_to or op.path}",
                )
            )
            continue

    return {
        "success": True,
        "summary": f"Applied patch to {len(changes)} file(s)",
        "changes": changes,
        "diff": "\n".join(diff for diff in diffs if diff).strip(),
    }


def is_local_host(hostname: str) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def validate_fetch_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("invalid URL")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and is_local_host(parsed.hostname or ""):
        return
    raise ValueError(
        "only HTTPS scheme is supported for external domains, HTTP is allowed for localhost/internal addresses"
    )


async def fetch_with_same_domain_redirects(raw_url: str) -> tuple[str, str]:
    validate_fetch_url(raw_url)
    async with httpx.AsyncClient(follow_redirects=False, timeout=30) as client:
        current = raw_url
        original_host = (urlparse(raw_url).hostname or "").lower()
        for _ in range(10):
            response = await client.get(current, headers={"User-Agent": "nano-agent"})
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("redirect response missing location header")
                current = urljoin(current, location)
                new_host = (urlparse(current).hostname or "").lower()
                if new_host != original_host:
                    raise ValueError(
                        "redirects are only followed within the same domain"
                    )
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "text/plain").lower()
            if (
                "application/zip" in content_type
                or "application/pdf" in content_type
                or content_type.startswith("image/")
                or content_type.startswith("audio/")
                or content_type.startswith("video/")
                or "application/octet-stream" in content_type
            ):
                raise ValueError(
                    f"binary content type is not supported: {content_type}"
                )
            return response.text, content_type
    raise ValueError("too many redirects")


def is_markdown_like(raw_url: str, content_type: str) -> bool:
    path = urlparse(raw_url).path.lower()
    return (
        "text/html" in content_type
        or "text/markdown" in content_type
        or path.endswith(".md")
        or path.endswith(".markdown")
    )


def archive_filename(raw_url: str, content_type: str) -> Path:
    parsed = urlparse(raw_url)
    domain = parsed.hostname or "unknown"
    name = Path(parsed.path).name or "index"
    if "." not in name:
        if "json" in content_type:
            name += ".json"
        elif "xml" in content_type:
            name += ".xml"
        elif "javascript" in content_type:
            name += ".js"
        elif "html" in content_type:
            name += ".html"
        else:
            name += ".txt"
    return ARCHIVE_DIR / domain / name


def add_line_numbers(text: str, max_lines: int = 400, max_chars: int = 30000) -> str:
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    numbered = "\n".join(f"{idx + 1:>4} | {line}" for idx, line in enumerate(lines))
    if len(numbered) > max_chars:
        numbered = numbered[:max_chars]
        truncated = True
    if truncated:
        numbered += "\n... [truncated]"
    return numbered


def with_numbered_lines(lines: list[str], start_line: int) -> str:
    if not lines:
        return ""
    width = max(4, len(str(start_line + len(lines) - 1)))
    return "\n".join(
        f"{start_line + index:>{width}} | {line}" for index, line in enumerate(lines)
    )


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


def truncate_tool_output(
    text: str, max_bytes: int = MAX_TOOL_OUTPUT_BYTES
) -> tuple[str, bool]:
    output_bytes = text.encode("utf-8")
    if len(output_bytes) <= max_bytes:
        return text, False

    suffix = f"\n... [truncated due to max output bytes limit of {max_bytes}]"
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        return suffix, True

    truncated_bytes = output_bytes[: max_bytes - len(suffix_bytes)]
    truncated_text = truncated_bytes.decode("utf-8", errors="ignore").rstrip()
    return f"{truncated_text}{suffix}", True


def truncate_bash_output_for_model(content: str) -> tuple[str, bool]:
    max_bytes = approx_bytes_for_tokens(BASH_MAX_OUTPUT_TOKENS)
    if len(content.encode("utf-8")) <= max_bytes:
        return content, False

    total_lines = count_output_lines(content)
    truncated = truncate_middle_with_token_budget(content, BASH_MAX_OUTPUT_TOKENS)
    return f"Total output lines: {total_lines}\n\n{truncated}", True


def count_output_lines(content: str) -> int:
    if not content:
        return 0

    line_count = content.count("\n")
    if not content.endswith("\n"):
        line_count += 1
    return line_count


def truncate_middle_with_token_budget(content: str, max_tokens: int) -> str:
    if not content:
        return ""

    max_bytes = approx_bytes_for_tokens(max_tokens)
    if max_tokens > 0 and len(content.encode("utf-8")) <= max_bytes:
        return content

    return truncate_middle_by_bytes_estimate(content, max_bytes, use_tokens=True)


def approx_bytes_for_tokens(tokens: int) -> int:
    if tokens <= 0:
        return 0
    return tokens * BASH_APPROX_BYTES_PER_TOKEN


def approx_tokens_from_byte_count(byte_count: int) -> int:
    if byte_count <= 0:
        return 0
    return (byte_count + BASH_APPROX_BYTES_PER_TOKEN - 1) // BASH_APPROX_BYTES_PER_TOKEN


def truncate_middle_by_bytes_estimate(
    content: str, max_bytes: int, *, use_tokens: bool
) -> str:
    if not content:
        return ""

    content_bytes = len(content.encode("utf-8"))
    if max_bytes <= 0:
        return format_bash_truncation_marker(
            use_tokens,
            removed_units(use_tokens, content_bytes, len(content)),
        )

    if content_bytes <= max_bytes:
        return content

    left_budget, right_budget = split_bash_budget(max_bytes)
    prefix_end, suffix_start, removed_chars = split_bash_string(
        content, left_budget, right_budget
    )
    prefix = content[:prefix_end]
    suffix = content[suffix_start:]
    removed_bytes = (
        content_bytes - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
    )
    marker = format_bash_truncation_marker(
        use_tokens,
        removed_units(use_tokens, removed_bytes, removed_chars),
    )
    return prefix + marker + suffix


def split_bash_budget(budget: int) -> tuple[int, int]:
    left = budget // 2
    return left, budget - left


def split_bash_string(
    content: str, beginning_bytes: int, end_bytes: int
) -> tuple[int, int, int]:
    content_bytes = len(content.encode("utf-8"))
    tail_start_target = max(content_bytes - end_bytes, 0)
    prefix_end = 0
    suffix_start = len(content)
    removed_chars = 0
    suffix_started = False
    byte_index = 0

    for char_index, char in enumerate(content):
        char_len = len(char.encode("utf-8"))
        char_end = byte_index + char_len
        if char_end <= beginning_bytes:
            prefix_end = char_index + 1
            byte_index = char_end
            continue

        if byte_index >= tail_start_target:
            if not suffix_started:
                suffix_start = char_index
                suffix_started = True
            byte_index = char_end
            continue

        removed_chars += 1
        byte_index = char_end

    if suffix_start < prefix_end:
        suffix_start = prefix_end

    return prefix_end, suffix_start, removed_chars


def removed_units(use_tokens: bool, removed_bytes: int, removed_chars: int) -> int:
    if use_tokens:
        return approx_tokens_from_byte_count(removed_bytes)
    return removed_chars


def format_bash_truncation_marker(use_tokens: bool, removed_count: int) -> str:
    if use_tokens:
        return f"…{removed_count} tokens truncated…"
    return f"…{removed_count} chars truncated…"

def _tool_error_from_json(exc: json.JSONDecodeError) -> dict[str, Any]:
    return {"success": False, "error": f"invalid JSON arguments: {exc}"}


def parse_tool_call_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to an object")
    return parsed


def tool_result_content(
    result: dict[str, Any], *, include_image_url: bool = True
) -> str:
    payload = dict(result)
    if not include_image_url and isinstance(payload.get("image_url"), str):
        payload["image_url"] = "[omitted data URL]"
    return json.dumps(payload, ensure_ascii=False)


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
