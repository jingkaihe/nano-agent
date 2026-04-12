from __future__ import annotations

from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from .core import BASH_MAX_OUTPUT_TOKENS, SessionUsage
from .platforms import model_context_window_limit

class ChatUI:
    def __init__(self) -> None:
        self.console = Console()
        self._current_stream: str | None = None
        self._stream_open = False
        self._prompt_session = PromptSession(
            multiline=False,
            history=InMemoryHistory(),
            key_bindings=self._build_prompt_bindings(),
        )

    def _build_prompt_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-x", "c-e")
        def _edit_in_editor(event: Any) -> None:
            event.current_buffer.open_in_editor(validate_and_handle=False)

        @bindings.add("c-d")
        def _exit_on_empty(event: Any) -> None:
            if not event.current_buffer.text:
                event.app.exit(exception=EOFError)
                return

            event.current_buffer.delete(count=1)

        return bindings

    def _separator(self, label: str, style: str) -> None:
        self.console.print()
        self.console.print(Rule(f"[{style}]{label}[/{style}]", style=style))

    def startup(
        self,
        provider: str,
        model: str,
        reasoning_effort: str | None,
        session_id: str,
        resumed: bool = False,
    ) -> None:
        subtitle = f"{provider} / {model}"
        if reasoning_effort:
            subtitle += f" / reasoning={reasoning_effort}"
        self.console.print(
            Rule(f"[bold cyan]nano-agent[/bold cyan] [dim]{subtitle}[/dim]")
        )
        prefix = "Resumed session" if resumed else "Session"
        self.console.print(f"[dim]{prefix}: {session_id}[/dim]")
        self.console.print("[dim]Type /exit to quit.[/dim]")
        self.console.print(
            "[dim]Press Enter to submit. Press Ctrl+X Ctrl+E to edit in $EDITOR. Ctrl+D exits on empty input.[/dim]"
        )

    async def prompt(self) -> str:
        self.console.print()
        return (
            await self._prompt_session.prompt_async(
                HTML("<b><ansigreen>You</ansigreen></b> > "),
            )
        ).strip()

    def show_user(self, text: str) -> None:
        self._separator("user", "green")
        self.console.print(text, soft_wrap=True)

    def show_debug(self, text: str) -> None:
        self.console.print(f"[dim][debug][/dim] {text}")

    def begin_assistant(self) -> None:
        self._current_stream = None
        self._stream_open = False

    def update_assistant(
        self, text_fragment: str = "", reasoning_fragment: str = ""
    ) -> None:
        if reasoning_fragment:
            if self._current_stream != "thinking":
                if self._stream_open:
                    self.console.print()
                self._separator("thinking", "magenta")
                self.console.print("thinking: ", style="magenta", end="")
                self._current_stream = "thinking"
                self._stream_open = True
            self.console.print(
                reasoning_fragment,
                style="dim italic",
                end="",
                soft_wrap=True,
                markup=False,
            )

        if text_fragment:
            if self._current_stream != "assistant":
                if self._stream_open:
                    self.console.print()
                self._separator("assistant", "blue")
                self.console.print("assistant: ", style="blue", end="")
                self._current_stream = "assistant"
                self._stream_open = True
            self.console.print(text_fragment, end="", soft_wrap=True, markup=False)

    def end_assistant(self) -> None:
        if self._stream_open:
            self.console.print()
        self._current_stream = None
        self._stream_open = False

    def _preview_text(
        self, text: str, *, max_lines: int = 18, max_chars: int = 4000
    ) -> str:
        lines = text.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
        preview = "\n".join(lines)
        if len(preview) > max_chars:
            preview = preview[:max_chars].rstrip()
            truncated = True
        if truncated:
            preview += "\n… [truncated]"
        return preview

    def _tool_metadata_rows(
        self, name: str, result: dict[str, Any]
    ) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = [("Tool", name)]
        rows.append(("Status", "ok" if result.get("success") else "error"))

        if name == "bash":
            command = str(result.get("command") or "")
            if command:
                rows.append(("Command", command))

            exit_code = result.get("exit_code")
            if exit_code is not None:
                rows.append(("Exit code", str(exit_code)))

            if result.get("truncated"):
                rows.append(
                    (
                        "Output",
                        f"preserved head/tail context (~{BASH_MAX_OUTPUT_TOKENS} tokens)",
                    )
                )

            if not result.get("success"):
                output = str(result.get("output") or "")
                error = str(result.get("error") or "")
                if not output:
                    if error:
                        rows.append(("Error", error))
                    elif exit_code is not None:
                        rows.append(("Error", f"command exited with code {exit_code}"))
            return rows

        if not result.get("success"):
            rows.append(("Error", str(result.get("error") or "unknown error")))
            return rows

        if name in {"read_file", "write_file", "edit_file"}:
            rows.append(("Path", str(result.get("file_path") or "")))
            if name == "read_file":
                rows.append(("Offset", str(result.get("offset") or 1)))
                rows.append(("Line limit", str(result.get("line_limit") or "")))
            if name == "edit_file" and result.get("replace_all"):
                rows.append(("Replaced", str(result.get("replaced_count") or 0)))
            return rows

        if name in {"grep", "glob"}:
            if result.get("path"):
                rows.append(("Path", str(result.get("path") or "")))
            if name == "grep":
                rows.append(("Pattern", str(result.get("pattern") or "")))
            else:
                rows.append(("Pattern", str(result.get("pattern") or "")))
            if result.get("truncated"):
                rows.append(("Output", "truncated"))
            return rows

        if name == "apply_patch":
            rows.append(("Summary", str(result.get("summary") or "")))
            changes = (
                result.get("changes") if isinstance(result.get("changes"), list) else []
            )
            if changes:
                rendered = []
                for change in changes[:8]:
                    if not isinstance(change, dict):
                        continue
                    action = str(change.get("action") or "update")
                    path = str(change.get("move_to") or change.get("path") or "")
                    rendered.append(f"{action} {path}".strip())
                if rendered:
                    rows.append(("Files", ", ".join(rendered)))
            return rows

        if name == "web_fetch":
            rows.append(("URL", str(result.get("url") or "")))
            if result.get("file_path"):
                rows.append(("Saved", str(result.get("file_path"))))
            if result.get("prompt"):
                rows.append(("Extracted", "yes"))
            return rows

        if name == "view_image":
            rows.append(("Path", str(result.get("path") or "")))
            mime_type = str(result.get("mime_type") or "")
            if mime_type:
                rows.append(("Type", mime_type))
            width = result.get("width")
            height = result.get("height")
            if isinstance(width, int) and isinstance(height, int):
                rows.append(("Size", f"{width}x{height}"))
            if result.get("detail"):
                rows.append(("Detail", str(result.get("detail"))))
            return rows

        if name == "skill":
            rows.append(("Skill", str(result.get("skill_name") or "")))
            rows.append(("Directory", str(result.get("directory") or "")))
            return rows

        return rows

    def _tool_preview(
        self, name: str, result: dict[str, Any]
    ) -> tuple[str, str] | None:
        if name == "bash":
            output = str(result.get("output") or "")
            if output:
                return ("Output", self._preview_text(output))

            if not result.get("success"):
                error = str(result.get("error") or "")
                if not error and result.get("exit_code") is not None:
                    error = f"command exited with code {result['exit_code']}"
                return (
                    ("Error", self._preview_text(error, max_lines=8, max_chars=1200))
                    if error
                    else None
                )

            return None

        if not result.get("success"):
            error = str(result.get("error") or "")
            return (
                ("Error", self._preview_text(error, max_lines=8, max_chars=1200))
                if error
                else None
            )

        if name == "apply_patch":
            diff = str(result.get("diff") or "")
            return (
                ("Diff", self._preview_text(diff, max_lines=24, max_chars=5000))
                if diff
                else None
            )

        if name in {
            "read_file",
            "write_file",
            "edit_file",
            "grep",
            "glob",
            "web_fetch",
            "skill",
        }:
            content = str(result.get("content") or "")
            return (
                ("Preview", self._preview_text(content, max_lines=16, max_chars=3000))
                if content
                else None
            )

        if name == "view_image":
            return None

        return None

    def show_tool_result(self, name: str, result: dict[str, Any]) -> None:
        style = "green" if result.get("success") else "red"
        self._separator(f"tool:{name}", style)
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(style="bold")
        table.add_column()
        for label, value in self._tool_metadata_rows(name, result):
            table.add_row(label, value)
        self.console.print(table)

        preview = self._tool_preview(name, result)
        if preview:
            title, content = preview
            self.console.print(f"[bold]{title}[/bold]")
            self.console.print(content, soft_wrap=True, markup=False)

    def newline(self) -> None:
        self.console.print()

    def show_session_exit(self, session_id: str) -> None:
        self.console.print(Rule(style="dim"))
        self.console.print(f"[dim]Session saved: {session_id}[/dim]")

    def show_usage_summary(
        self, usage: SessionUsage, model: str, provider: str
    ) -> None:
        if usage.request_count <= 0:
            return

        self._separator("usage", "yellow")
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(style="yellow")
        table.add_column(style="white")

        table.add_row("Model calls", _format_int(usage.request_count))
        table.add_row(
            "Session tokens",
            f"input {_format_int(usage.input_tokens)} / output {_format_int(usage.output_tokens)} / total {_format_int(usage.total_tokens)}",
        )
        table.add_row(
            "Session cache",
            f"read {_format_int(usage.cached_input_tokens)} / reasoning {_format_int(usage.reasoning_tokens)}",
        )

        uncached_last_input = max(
            usage.last_input_tokens - usage.last_cached_input_tokens, 0
        )
        table.add_row(
            "Last request",
            f"input {_format_int(usage.last_input_tokens)} (uncached {_format_int(uncached_last_input)}, cached {_format_int(usage.last_cached_input_tokens)}) / output {_format_int(usage.last_output_tokens)}",
        )

        limit = _context_window_limit(usage.last_model or model, provider=provider)
        label = "Last context"
        value = _format_ratio(usage.last_input_tokens, limit)
        if limit is None:
            value += " (model limit unavailable)"
        table.add_row(label, value)

        if usage.last_api:
            table.add_row("Last API", usage.last_api)
        if usage.last_model:
            table.add_row("Last model", usage.last_model)

        self.console.print(table)

def _format_int(value: int) -> str:
    return f"{value:,}"


def _context_window_limit(model: str, provider: str = "copilot") -> int | None:
    return model_context_window_limit(model, provider=provider)


def _format_ratio(value: int, total: int | None) -> str:
    if not total or total <= 0:
        return _format_int(value)
    percentage = value / total * 100
    return f"{_format_int(value)} / {_format_int(total)} ({percentage:.1f}%)"

