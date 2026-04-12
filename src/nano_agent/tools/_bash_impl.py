from __future__ import annotations

from ._common import *


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
