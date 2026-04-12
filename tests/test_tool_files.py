from __future__ import annotations

import tempfile
from pathlib import Path

from nano_agent.tools.files import _file_mtime, execute_edit_file, execute_read_file, execute_write_file

from .helpers import assert_equal, assert_true


def test_file_tool_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        path = cwd / "demo.txt"
        created = execute_write_file(
            str(path),
            "alpha\nbeta\n",
            cwd,
            last_read_time=None,
        )
        assert_true(created["success"], "write_file should create a file")

        read = execute_read_file(str(path), cwd)
        assert_true("1 | alpha" in read["content"], "read_file should number lines")
        last_read = _file_mtime(path)

        edited = execute_edit_file(
            str(path),
            "beta",
            "gamma",
            cwd,
            replace_all=False,
            last_read_time=last_read,
        )
        assert_true(edited["success"], "edit_file should edit the file")
        assert_equal(path.read_text(), "alpha\ngamma\n")
