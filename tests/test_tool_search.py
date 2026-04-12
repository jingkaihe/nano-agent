from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from nano_agent.tools.search import execute_glob, execute_grep

from .helpers import assert_true


def test_search_tools_roundtrip() -> None:
    fd_path = shutil.which("fd")
    rg_path = shutil.which("rg")
    if not fd_path or not rg_path:
        return

    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        src = cwd / "src"
        src.mkdir()
        target = src / "demo.py"
        target.write_text("def hello():\n    return 'hi'\n")

        glob_result = execute_glob("**/*.py", cwd, path=str(cwd))
        assert_true(str(target) in glob_result["content"], "glob should find file")

        grep_result = execute_grep("hello", cwd, path=str(cwd), include="*.py")
        assert_true(
            str(target) in grep_result["content"],
            "grep should report matching file",
        )
