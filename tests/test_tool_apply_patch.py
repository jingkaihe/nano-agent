from __future__ import annotations

import tempfile
from pathlib import Path

from nano_agent.tools.apply_patch import execute_apply_patch

from .helpers import assert_equal, assert_true


def test_apply_patch_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        target = cwd / "demo.txt"
        target.write_text("alpha\nbeta\n")
        result = execute_apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: demo.txt",
                    "@@",
                    " alpha",
                    "-beta",
                    "+gamma",
                    "*** End Patch",
                ]
            ),
            cwd,
        )
        assert_true(result["success"], "apply_patch should succeed")
        assert_equal(target.read_text(), "alpha\ngamma\n")


def test_apply_patch_accepts_absolute_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        target = cwd / "absolute.txt"
        target.write_text("alpha\nbeta\n")
        result = execute_apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    f"*** Update File: {target}",
                    "@@",
                    " alpha",
                    "-beta",
                    "+gamma",
                    "*** End Patch",
                ]
            ),
            cwd,
        )
        assert_true(result["success"], "apply_patch should accept absolute paths")
        assert_equal(target.read_text(), "alpha\ngamma\n")
