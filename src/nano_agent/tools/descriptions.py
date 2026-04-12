from __future__ import annotations

from typing import Any

from ._common import JINJA, MAX_READ_LINE_LIMIT, MAX_READ_OUTPUT_BYTES, GREP_MAX_RESULTS, GLOB_MAX_RESULTS
def bash_description(tool_names: list[str] | None = None) -> str:
    available = set(tool_names or [])
    edit_hint = (
        "write_file or edit_file"
        if {"write_file", "edit_file"}.issubset(available)
        else "apply_patch"
    )
    return JINJA.from_string(
        """Run a bash command in an isolated shell process.

# Restrictions
Banned commands:
- vim
- view
- less
- more
- cd

# Input
- command: required bash command
- description: required, 5-10 words
- timeout: required, 10-120

# Rules
- Use parallel tool calling for independent commands.
- Do not run interactive commands.
- For multiple commands, use ';' or '&&' on one line.
- Avoid direct cd; use absolute paths or subshell: (cd /path && cmd).
- {% if prefer_search_tools %}Prefer grep/glob over grep/find in bash.{% else %}For filesystem search activities, use fd and rg via this tool only.{% endif %}
- Do not use heredoc; use {{ edit_hint }} instead.

Examples:
- (cd /repo && mise run test)
"""
    ).render(
        prefer_search_tools={"grep", "glob"}.issubset(available), edit_hint=edit_hint
    )


def read_file_description() -> str:
    return JINJA.from_string(
        f"""Reads a file and returns its contents with line numbers.

This tool takes three parameters:
- file_path: The absolute path of the file to read
- offset: The 1-indexed line number to start reading from (default: 1, minimum: 1)
- line_limit: The maximum number of lines to read from the offset (default: {MAX_READ_LINE_LIMIT}, minimum: 1, maximum: {MAX_READ_LINE_LIMIT})

For most files, omit offset and line_limit to read the entire file. Use these parameters only for large files when you need specific sections.

The result includes line numbers. If there are more lines beyond the line limit, a continuation hint is shown. Very long lines are truncated and total output is capped at {MAX_READ_OUTPUT_BYTES} bytes.

If you need to read multiple files, use parallel tool calling to read multiple files simultaneously.
"""
    ).render()


def write_file_description() -> str:
    return JINJA.from_string(
        """Writes a file with the given text. If the file already exists, its contents will be overwritten.

This tool takes two parameters:
- file_path: The absolute path of the file to write
- text: The text to be written to the file. It must not be empty.

IMPORTANT: If you want to create an empty file, use the `bash` tool to run `touch` instead.
IMPORTANT: If you are overwriting an existing file, read it using `read_file` first. If the file changed after the last read, this tool will ask you to read it again.
IMPORTANT: Make sure the directory already exists before writing to it.
"""
    ).render()


def edit_file_description() -> str:
    return JINJA.from_string(
        """Edit a file by replacing old text with new text.

If you are creating a new file, use `write_file` instead.

This tool takes four parameters:
- file_path: The absolute path of the file to edit
- old_text: The text to be replaced. It must exactly match the text in the file including whitespace.
- new_text: The replacement text
- replace_all: Optional, default false. If true, replace all occurrences of old_text; otherwise old_text must be unique.

# RULES
## Read before editing
You must read the file using `read_file` before making non-replace_all edits.

## Unique matching
When replace_all is false, make old_text unique by including 3-5 lines before and after the target block where needed.

## Validate after edit
If the edit changes code or configuration, validate it with the relevant checks via `bash`.
"""
    ).render()


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


def apply_patch_description() -> str:
    return JINJA.from_string(
        """Use the `apply_patch` tool to edit files.
Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more "hunks", each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@   def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- File references may be relative or absolute. Relative paths are resolved from the current working directory.
"""
    ).render()


def web_fetch_description() -> str:
    return JINJA.from_string(
        """Fetch content from a public URL.

# Input
- url: required URL to fetch
- prompt: optional instruction for extracting info from HTML/Markdown pages

# Rules
- Use HTTPS for external domains.
- HTTP is allowed only for localhost/internal addresses.
- Redirects are followed only within the same domain (max 10).
- Binary content types (zip/pdf/image/audio/video/octet-stream) are rejected.

# Behavior
- Code/text/JSON/XML/etc:
  - Save to ~/.nano-agent/web-archives/{domain}/{filename}.{ext}
  - Return content with line numbers (truncated if output is too large)
- HTML/Markdown without prompt:
  - Return full page content as Markdown (HTML is converted)
- HTML/Markdown with prompt:
  - Run AI extraction against page content and return only the extracted result

# Prompt guidance
Use prompt when:
- You need specific facts/sections from an HTML/Markdown page
- The page is large and you do not want full-page output

Examples:
- url: https://docs.example.com/api-reference
  prompt: List all endpoints with HTTP methods
- url: https://company.example.com/changelog
  prompt: Summarize breaking changes in the latest release

Do not use prompt when:
- You want raw file contents with line numbers
- You want full-page output

Examples:
- url: https://raw.githubusercontent.com/user/repo/main/config.yaml
- url: https://example.com/data.json

# Notes
- Only public URLs are supported (no auth/session handling).
- Prompt is ignored for non-HTML/Markdown responses (code/text/JSON/XML).
"""
    ).render()


def view_image_description(model: str) -> str:
    detail_text = (
        "The optional `detail` field is available for this model and supports only `original`. Use it when high-fidelity image perception or precise localization is needed."
        if supports_view_image_original_detail(model)
        else "This model does not support the optional `detail` field; omit it."
    )
    return (
        "View a local image from the filesystem (only use if given a full filepath by the user, and the image isn't already attached in the conversation context).\n\n"
        + detail_text
    )
