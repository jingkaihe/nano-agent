from __future__ import annotations

import nano_agent.core as core
from nano_agent.tools._selection_impl import normalize_allowed_tools, select_tool_names

from .helpers import assert_equal


def test_select_tool_names_defaults() -> None:
    assert_equal(
        select_tool_names("gpt-4.1", "copilot"),
        core.DEFAULT_CHAT_TOOL_NAMES,
    )
    assert_equal(
        select_tool_names("claude-sonnet", "copilot"),
        core.DEFAULT_CHAT_TOOL_NAMES,
    )
    assert_equal(
        select_tool_names("gpt-5", "openai"),
        core.DEFAULT_RESPONSE_TOOL_NAMES,
    )


def test_select_tool_names_allowed_override() -> None:
    override = normalize_allowed_tools("bash,read_file,grep")
    assert_equal(select_tool_names("gpt-5", "openai", override), override)
