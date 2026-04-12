from __future__ import annotations

from nano_agent.tools.bash import validate_bash_args

from .helpers import assert_equal


def test_validate_bash_args_accepts_quoted_operators() -> None:
    command = 'printf "%s\\n" "a && b" && pwd'
    validated, description, timeout = validate_bash_args(
        command,
        "accept quoted operators",
        10,
    )
    assert_equal(validated, command)
    assert_equal(description, "accept quoted operators")
    assert_equal(timeout, 10)


def test_validate_bash_args_accepts_multiline_commands() -> None:
    command = "printf 'hello\\n'\npwd"
    validated, description, timeout = validate_bash_args(
        command,
        "accept multiline commands",
        10,
    )
    assert_equal(validated, command)
    assert_equal(description, "accept multiline commands")
    assert_equal(timeout, 10)


def test_validate_bash_args_rejects_banned_commands() -> None:
    try:
        validate_bash_args("echo hi\nless file.txt", "reject banned commands", 10)
    except ValueError as exc:
        assert_equal(str(exc), "command is banned: less")
        return
    raise AssertionError("validate_bash_args should reject banned commands")
