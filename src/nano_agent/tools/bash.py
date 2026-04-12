from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from .base import ToolCallContext, ToolDefinition
from ._common import BANNED_COMMANDS
from .descriptions import bash_description
from ._output_impl import truncate_bash_output_for_model
from .schemas import bash_schema


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


class BashTool(ToolDefinition):
    name = "bash"

    def description(self, agent: Any) -> str:
        return bash_description(agent.active_tool_names())

    def schema(self, agent: Any) -> dict[str, Any]:
        return bash_schema()

    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        command, _description, timeout = validate_bash_args(
            context.arguments.get("command"),
            context.arguments.get("description"),
            context.arguments.get("timeout"),
        )
        exit_code, output, truncated = context.agent.bash.run(command, timeout)
        return {
            "success": exit_code == 0,
            "command": command,
            "exit_code": exit_code,
            "output": output,
            "truncated": truncated,
        }


bash_tool = BashTool()

__all__ = [
    "BashRunner",
    "BashTool",
    "bash_description",
    "bash_schema",
    "bash_tool",
    "validate_bash_args",
]
