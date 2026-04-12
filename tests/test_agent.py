from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, cast

import httpx
import openai

import nano_agent.core as core
from nano_agent.agent import NanoAgent
from nano_agent.ui import ChatUI

from .helpers import assert_equal, assert_true


def test_prompt_cache_keys() -> None:
    agent = NanoAgent(
        model="gpt-5",
        cwd=Path.cwd(),
        provider="openai",
        api_key="test",
        session_id="session-openai",
    )
    try:
        assert_equal(
            agent.completion_kwargs().get("prompt_cache_key"),
            "session-openai",
        )
        assert_equal(agent.responses_prompt_cache_key(), "session-openai")
    finally:
        asyncio.run(agent.close())

    copilot_agent = cast(Any, object.__new__(NanoAgent))
    copilot_agent.provider = "copilot"
    copilot_agent.session_id = "session-copilot"
    copilot_agent.reasoning_effort = None
    assert_true(
        "prompt_cache_key" not in NanoAgent.completion_kwargs(copilot_agent),
        "copilot chat completions should not use OpenAI prompt_cache_key",
    )
    assert_equal(
        NanoAgent.responses_prompt_cache_key(copilot_agent),
        "session-copilot",
    )


def test_copilot_initiator_headers() -> None:
    agent = cast(Any, object.__new__(NanoAgent))
    agent.provider = "copilot"

    assert_equal(
        NanoAgent.copilot_extra_headers(agent, initiator="agent"),
        {"x-initiator": "agent"},
    )
    assert_equal(
        NanoAgent.copilot_extra_headers(agent, initiator="user"),
        {"x-initiator": "user"},
    )

    openai_agent = cast(Any, object.__new__(NanoAgent))
    openai_agent.provider = "openai"
    assert_equal(
        NanoAgent.copilot_extra_headers(openai_agent, initiator="user"),
        None,
    )


def test_call_with_provider_auth_retry_retries_auth_error() -> None:
    async def _run() -> None:
        agent = NanoAgent.__new__(NanoAgent)
        agent.provider = "copilot"
        agent.api_key = None
        agent.base_url = None
        refresh_calls: list[bool] = []
        attempts = {"count": 0}

        async def fake_refresh(*, force: bool = False) -> bool:
            refresh_calls.append(force)
            return True

        async def fake_call() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise openai.AuthenticationError(
                    "unauthorized",
                    response=httpx.Response(
                        401,
                        request=httpx.Request("GET", "https://example.com"),
                    ),
                    body=None,
                )
            return "ok"

        agent._refresh_copilot_clients = fake_refresh  # type: ignore[attr-defined]
        result = await NanoAgent._call_with_provider_auth_retry(agent, fake_call)
        assert_equal(result, "ok")
        assert_equal(refresh_calls, [False, True])
        assert_equal(attempts["count"], 2)

    asyncio.run(_run())


def test_call_with_provider_auth_retry_retries_stream_auth_error() -> None:
    async def _run() -> None:
        agent = NanoAgent.__new__(NanoAgent)
        agent.provider = "copilot"
        agent.api_key = None
        agent.base_url = None
        refresh_calls: list[bool] = []
        enter_attempts: list[str] = []
        exit_calls: list[str] = []
        call_count = 0

        async def fake_refresh(*, force: bool = False) -> bool:
            refresh_calls.append(force)
            return True

        class FakeStreamManager:
            def __init__(self, name: str, *, fail_auth: bool) -> None:
                self.name = name
                self.fail_auth = fail_auth

            async def __aenter__(self) -> str:
                enter_attempts.append(self.name)
                if self.fail_auth:
                    raise openai.AuthenticationError(
                        "unauthorized",
                        response=httpx.Response(
                            401,
                            request=httpx.Request("GET", "https://example.com"),
                        ),
                        body=None,
                    )
                return f"stream:{self.name}"

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                exit_calls.append(self.name)
                return False

        def fake_call() -> FakeStreamManager:
            nonlocal call_count
            call_count += 1
            return FakeStreamManager(f"attempt-{call_count}", fail_auth=call_count == 1)

        agent._refresh_copilot_clients = fake_refresh  # type: ignore[attr-defined]
        stream_cm = await NanoAgent._call_with_provider_auth_retry(agent, fake_call)

        async with stream_cm as stream:
            assert_equal(stream, "stream:attempt-2")

        assert_equal(refresh_calls, [False, True])
        assert_equal(enter_attempts, ["attempt-1", "attempt-2"])
        assert_equal(exit_calls, ["attempt-2"])
        assert_equal(call_count, 2)

    asyncio.run(_run())


def test_call_with_provider_auth_retry_leaves_async_iterables_unwrapped() -> None:
    async def _run() -> None:
        agent = NanoAgent.__new__(NanoAgent)
        agent.provider = "copilot"
        agent.api_key = None
        agent.base_url = None
        refresh_calls: list[bool] = []

        async def fake_refresh(*, force: bool = False) -> bool:
            refresh_calls.append(force)
            return True

        class FakeAsyncStream:
            def __init__(self) -> None:
                self.chunks = ["chunk-1", "chunk-2"]

            def __aiter__(self) -> "FakeAsyncStream":
                return self

            async def __anext__(self) -> str:
                if not self.chunks:
                    raise StopAsyncIteration
                return self.chunks.pop(0)

            async def __aenter__(self) -> "FakeAsyncStream":
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

        stream = FakeAsyncStream()

        def fake_call() -> FakeAsyncStream:
            return stream

        agent._refresh_copilot_clients = fake_refresh  # type: ignore[attr-defined]
        result = await NanoAgent._call_with_provider_auth_retry(agent, fake_call)
        assert_true(
            result is stream,
            "iterator-style streams should not be wrapped as context managers",
        )

        chunks: list[str] = []
        async for chunk in result:
            chunks.append(chunk)

        assert_equal(chunks, ["chunk-1", "chunk-2"])
        assert_equal(refresh_calls, [False])

    asyncio.run(_run())


def test_call_with_provider_auth_retry_non_copilot_no_retry() -> None:
    async def _run() -> None:
        agent = NanoAgent.__new__(NanoAgent)
        agent.provider = "openai"
        agent.api_key = None
        agent.base_url = None
        attempts = {"count": 0}

        async def fake_call() -> str:
            attempts["count"] += 1
            raise openai.AuthenticationError(
                "unauthorized",
                response=httpx.Response(
                    401,
                    request=httpx.Request("GET", "https://example.com"),
                ),
                body=None,
            )

        try:
            await NanoAgent._call_with_provider_auth_retry(agent, fake_call)
        except openai.AuthenticationError:
            pass
        else:
            raise AssertionError("non-copilot auth errors should not be retried")
        assert_equal(attempts["count"], 1)

    asyncio.run(_run())


def test_build_system_prompt_variants() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        context_path = cwd / core.CONTEXT_FILENAME
        context_path.write_text("# Local repo guidance\nAlways run focused checks.\n")

        default_prompt = core.build_system_prompt(cwd)
        gpt_41_prompt = core.build_system_prompt(cwd, "gpt-4.1")

        for prompt in (default_prompt, gpt_41_prompt):
            assert_true(
                "# Context" in prompt,
                "system prompt should include the shared context guidance",
            )
            assert_true(
                f"contains a `{core.CONTEXT_FILENAME}` file" in prompt,
                "system prompt should mention AGENTS-style context loading",
            )
            assert_true(
                f'<context filename="{context_path}"' in prompt,
                "system prompt should embed collected context files",
            )
            assert_true(
                "Always run focused checks." in prompt,
                "system prompt should include the context file contents",
            )

        assert_true(
            "You are an agent - please keep going until the user’s query is completely resolved"
            in gpt_41_prompt,
            "gpt-4.1 prompt should include beast-style operating guidance",
        )
        assert_true(
            "`web_fetch`" in gpt_41_prompt,
            "gpt-4.1 prompt should reference the real web_fetch tool",
        )


def test_build_system_prompt_tool_guidance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        prompt = core.build_system_prompt(
            cwd,
            "claude-sonnet",
            "copilot",
            ["bash", "read_file", "write_file", "edit_file", "grep", "glob"],
        )
        assert_true(
            "prefer the `glob` and `grep` tools" in prompt,
            "chat-style prompt should prefer grep/glob when available",
        )
        assert_true(
            "Use `read_file`, `write_file`, and `edit_file` for file changes" in prompt,
            "chat-style prompt should prefer file tools over apply_patch",
        )

        responses_prompt = core.build_system_prompt(cwd, "gpt-5", "openai", None)
        assert_true(
            "Use `apply_patch` for file edits." in responses_prompt,
            "responses prompt should keep apply_patch guidance",
        )


def test_tool_specs_respect_allowed_tools() -> None:
    agent = NanoAgent(
        model="gpt-4.1",
        cwd=Path.cwd(),
        provider="openai",
        api_key="test",
        allowed_tools=["bash"],
    )
    try:
        specs = [spec["function"]["name"] for spec in agent.tool_specs()]
        assert_equal(specs, ["bash"])
    finally:
        asyncio.run(agent.close())


def test_tool_specs_include_view_image_when_enabled() -> None:
    agent = NanoAgent(
        model="gpt-5.3-codex",
        cwd=Path.cwd(),
        provider="openai",
        api_key="test",
        allowed_tools=["view_image"],
    )
    try:
        specs = agent.tool_specs()
        assert_equal(specs[0]["function"]["name"], "view_image")
        properties = specs[0]["function"]["parameters"]["properties"]
        assert_true(
            "detail" in properties,
            "gpt-5.3-codex view_image tool should expose detail",
        )
    finally:
        asyncio.run(agent.close())


def test_execute_tool_calls_only_adds_chat_followup_image_message() -> None:
    async def _run() -> None:
        class FakeImageAgent(NanoAgent):
            async def run_tool(
                self, name: str, arguments: dict[str, Any]
            ) -> dict[str, Any]:
                assert_equal(name, "view_image")
                assert_equal(arguments, {"path": "/tmp/demo.png"})
                return {
                    "success": True,
                    "path": "/tmp/demo.png",
                    "image_url": "data:image/png;base64,AAA",
                    "mime_type": "image/png",
                    "width": 10,
                    "height": 10,
                }

        agent = FakeImageAgent(
            model="claude-haiku-4.5",
            cwd=Path.cwd(),
            provider="openai",
            api_key="test",
        )
        try:
            tool_call = {
                "id": "call-1",
                "function": {
                    "name": "view_image",
                    "arguments": '{"path":"/tmp/demo.png"}',
                },
            }

            agent.messages = []
            await agent.execute_tool_calls(
                [tool_call], call_id_key="id", api="messages"
            )
            assert_equal(len(agent.messages), 1)
            assert_equal(agent.messages[0]["role"], "tool")
            assert_true(
                isinstance(agent.messages[0]["content"], list),
                "anthropic tool result should keep structured image content",
            )
            assert_equal(agent.messages[0]["content"][0]["type"], "input_image")

            agent.messages = []
            await agent.execute_tool_calls(
                [tool_call], call_id_key="id", api="chat.completions"
            )
            assert_equal(len(agent.messages), 2)
            assert_equal(agent.messages[0]["role"], "tool")
            assert_equal(agent.messages[1]["role"], "user")
            assert_equal(agent.messages[1]["content"][0]["type"], "image_url")
        finally:
            await agent.close()

    asyncio.run(_run())


def test_bash_failure_preview_uses_output() -> None:
    ui = ChatUI()
    result = {
        "success": False,
        "command": "ls /definitely-missing",
        "exit_code": 2,
        "output": "ls: cannot access '/definitely-missing': No such file or directory",
        "truncated": False,
    }

    rows = ui._tool_metadata_rows("bash", result)
    assert_equal(
        rows,
        [
            ("Tool", "bash"),
            ("Status", "error"),
            ("Command", "ls /definitely-missing"),
            ("Exit code", "2"),
        ],
    )
    assert_equal(
        ui._tool_preview("bash", result),
        (
            "Output",
            "ls: cannot access '/definitely-missing': No such file or directory",
        ),
    )
