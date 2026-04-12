from __future__ import annotations

from ._common import *


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
