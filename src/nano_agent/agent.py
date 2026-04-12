from __future__ import annotations

import inspect
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from .core import *
from .core import (
    _extract_text_fragments,
    _extract_usage_metrics,
    _get_delta_fragments,
    _is_copilot_claude_chat,
    _is_copilot_provider,
    _merge_stream_tool_call,
    extract_function_calls_from_response,
)
from .platforms import *
from .platforms import _close_async_client, _is_async_context_manager
from .tools import *
from .tools import (
    _file_mtime,
    _ordered_tool_calls,
    _tool_error_from_json,
    _tool_result_message_content,
    _to_plain_data,
    anthropic_thinking_config,
    chat_followup_image_message,
    responses_function_call_output,
    tool_definition,
    tool_spec,
    ToolCallContext,
)
from .ui import ChatUI


class _CopilotRefreshAsyncContextManager:
    def __init__(self, agent: NanoAgent, call: Callable[[], Any], manager: Any) -> None:
        self.agent = agent
        self.call = call
        self.manager = manager

    async def __aenter__(self) -> Any:
        try:
            return await self.manager.__aenter__()
        except Exception as exc:
            if not _is_copilot_provider(
                self.agent.provider
            ) or not is_copilot_auth_error(exc):
                raise

        await self.agent._refresh_copilot_clients(force=True)
        self.manager = await self.agent._resolve_copilot_call_result(self.call)
        return await self.manager.__aenter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return await self.manager.__aexit__(exc_type, exc, tb)


@dataclass
class AssistantTurn:
    message: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    responses_items: list[dict[str, Any]] = field(default_factory=list)


class NanoAgent:
    def __init__(
        self,
        model: str,
        cwd: Path,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        weak_model: str | None = None,
        session_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        responses_items: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        debug_tools: bool = False,
    ) -> None:
        self.model = model
        self.cwd = cwd.resolve()
        self.provider = provider
        self.session_id = (
            normalize_session_id(session_id) if session_id else str(uuid.uuid4())
        )
        self.history_path = conversation_history_path(self.session_id)
        self.reasoning_effort = (
            None if reasoning_effort in {None, "none"} else reasoning_effort
        )
        resolved_weak_model = weak_model.strip() if isinstance(weak_model, str) else ""
        self.weak_model = resolved_weak_model or default_weak_model(provider, model)
        self.created_at = iso_now()
        if self.history_path.exists():
            try:
                existing = load_conversation_history(self.session_id)
                self.created_at = cast(str, existing["created_at"])
            except Exception:
                pass
        self.api_key = api_key
        self.base_url = base_url
        self._copilot_token: str | None = None
        self.client, self.anthropic_client = create_runtime_clients(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        if _is_copilot_provider(provider) and api_key is None and base_url is None:
            creds = load_copilot_credentials()
            copilot_token, _ = refresh_copilot_token(creds)
            self._copilot_token = copilot_token
        self.messages: list[dict[str, Any]] = list(messages or [])
        self.responses_items: list[dict[str, Any]] = list(responses_items or [])
        self.usage = SessionUsage.from_dict(usage)
        self.allowed_tools = normalize_allowed_tools(allowed_tools)
        self.debug_tools = debug_tools
        self._file_access_times: dict[str, float] = {}
        self._file_locks: dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()
        self.bash = BashRunner(self.cwd)
        self.active_skills: set[str] = set()
        self.ui = ChatUI()
        if self.debug_tools:
            self.ui.show_debug("resolved tools: " + ", ".join(self.active_tool_names()))
        self.save_history()

    def completion_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.provider.strip().lower() == "openai":
            kwargs["prompt_cache_key"] = self.session_id
        return kwargs

    def responses_prompt_cache_key(self) -> str | None:
        if self.provider.strip().lower() not in {"openai", "copilot"}:
            return None
        return self.session_id

    def copilot_extra_headers(self, *, initiator: str) -> dict[str, str] | None:
        if not _is_copilot_provider(self.provider):
            return None
        return {"x-initiator": initiator}

    def prepare_chat_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return apply_copilot_ephemeral_cache(
            messages,
            provider=self.provider,
            model=self.model,
            endpoint="/chat/completions",
        )

    def chat_extra_body(self) -> dict[str, Any] | None:
        if _is_copilot_claude_chat(self.provider, self.model):
            return {"thinking_budget": 4000}
        return None

    def record_usage(
        self, usage_value: Any, *, api: str, model: str | None = None
    ) -> None:
        metrics = _extract_usage_metrics(usage_value)
        if not metrics:
            return
        self.usage.record(metrics, api=api, model=model or self.model)

    def responses_reasoning(self) -> dict[str, Any] | None:
        if not self.reasoning_effort:
            return None
        return {"effort": self.reasoning_effort, "summary": "auto"}

    def anthropic_thinking(self) -> dict[str, Any] | None:
        return anthropic_thinking_config(self.reasoning_effort)

    def system_prompt(self) -> str:
        return build_system_prompt(
            self.cwd,
            self.model,
            self.provider,
            self.allowed_tools,
        )

    def active_tool_names(self) -> list[str]:
        return select_tool_names(self.model, self.provider, self.allowed_tools)

    def _get_file_lock(self, path: Path) -> threading.Lock:
        key = str(path)
        with self._file_locks_guard:
            lock = self._file_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[key] = lock
            return lock

    def mark_file_read(self, path: Path) -> None:
        try:
            self._file_access_times[str(path)] = _file_mtime(path)
        except OSError:
            self._file_access_times[str(path)] = time.time()

    def last_read_time(self, path: Path) -> float | None:
        return self._file_access_times.get(str(path))

    def require_anthropic_client(self) -> AsyncAnthropic:
        if self.anthropic_client is None:
            raise ValueError("Anthropic messages client is not configured")
        return self.anthropic_client

    def require_openai_client(self) -> AsyncOpenAI:
        if self.client is None:
            raise ValueError("OpenAI-compatible client is not configured")
        return self.client

    async def _refresh_copilot_clients(self, *, force: bool = False) -> bool:
        if (
            not _is_copilot_provider(self.provider)
            or self.api_key is not None
            or self.base_url is not None
        ):
            return False

        creds = load_copilot_credentials()
        previous_token = str(creds.get("copilot_token") or self._copilot_token or "")
        copilot_token, refreshed = refresh_copilot_token(creds, force=force)
        token_changed = force or copilot_token != previous_token
        self._copilot_token = copilot_token
        if not token_changed:
            return False

        old_client = self.client
        old_anthropic_client = self.anthropic_client
        self.client = AsyncOpenAI(
            api_key=copilot_token,
            base_url=copilot_base_url(),
            default_headers={
                "User-Agent": COPILOT_OPENAI_USER_AGENT,
                "Editor-Version": COPILOT_EDITOR_VERSION,
            },
        )
        self.anthropic_client = (
            AsyncAnthropic(
                auth_token=copilot_token,
                base_url=copilot_base_url(),
                default_headers=copilot_anthropic_headers(),
            )
            if should_use_anthropic_messages_api(self.provider, self.model)
            else None
        )

        if old_client is not None and old_client is not self.client:
            await _close_async_client(old_client)
        if (
            old_anthropic_client is not None
            and old_anthropic_client is not self.anthropic_client
        ):
            await _close_async_client(old_anthropic_client)
        return refreshed.get("copilot_token") != previous_token or force

    async def _resolve_copilot_call_result(self, call: Callable[[], Any]) -> Any:
        result = call()
        return await result if inspect.isawaitable(result) else result

    async def _call_with_provider_auth_retry(self, call: Callable[[], Any]) -> Any:
        if _is_copilot_provider(self.provider):
            await self._refresh_copilot_clients()

        try:
            result = await self._resolve_copilot_call_result(call)
            if _is_async_context_manager(result):
                return _CopilotRefreshAsyncContextManager(self, call, result)
            return result
        except Exception as exc:
            if not _is_copilot_provider(self.provider) or not is_copilot_auth_error(
                exc
            ):
                raise

            await self._refresh_copilot_clients(force=True)
            retry_result = await self._resolve_copilot_call_result(call)
            if _is_async_context_manager(retry_result):
                return _CopilotRefreshAsyncContextManager(self, call, retry_result)
            return retry_result

    async def close(self) -> None:
        self.save_history()
        self.bash.close()
        if self.client is not None:
            await _close_async_client(self.client)
        if self.anthropic_client is not None:
            await _close_async_client(self.anthropic_client)

    def serialized_history(self) -> dict[str, Any]:
        return {
            "id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "weak_model": self.weak_model,
            "allowed_tools": self.allowed_tools,
            "created_at": self.created_at,
            "updated_at": iso_now(),
            "history": {
                "messages": self.messages,
                "responses_items": self.responses_items,
                "usage": self.usage.to_dict(),
            },
        }

    def save_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps(self.serialized_history(), indent=2) + "\n"
        )

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        call_id_key: str,
        api: str,
    ) -> list[dict[str, Any]]:
        tool_outputs: list[dict[str, Any]] = []
        for call in tool_calls:
            try:
                args = parse_tool_call_arguments(call["function"].get("arguments"))
            except json.JSONDecodeError as exc:
                result = _tool_error_from_json(exc)
            except ValueError as exc:
                result = {"success": False, "error": str(exc)}
            else:
                result = await self.run_tool(call["function"]["name"], args)

            self.ui.show_tool_result(call["function"]["name"], result)
            tool_call_id = call[call_id_key]
            message_tool_content = _tool_result_message_content(
                result, role="tool", api=api
            )
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": message_tool_content,
                }
            )
            followup_chat_message = (
                chat_followup_image_message(result)
                if api == "chat.completions"
                else None
            )
            if followup_chat_message is not None:
                self.messages.append(followup_chat_message)
            response_output = responses_function_call_output(result)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": response_output,
                }
            )

        return tool_outputs

    def _finalize_response_turn(self, final_response: Any) -> AssistantTurn:
        output_text = getattr(final_response, "output_text", None)
        assistant_text = (
            output_text if isinstance(output_text, str) and output_text else None
        )
        plain_response = _to_plain_data(final_response) or {}
        output_items = plain_response.get("output") or []
        responses_items = output_items if isinstance(output_items, list) else []
        return AssistantTurn(
            message={"role": "assistant", "content": assistant_text},
            tool_calls=extract_function_calls_from_response(final_response),
            responses_items=list(responses_items),
        )

    def _finalize_chat_turn(
        self,
        assistant_content_parts: list[str],
        assistant_reasoning_parts: list[str],
        tool_call_parts: dict[int, dict[str, Any]],
        reasoning_opaque: str | None,
    ) -> AssistantTurn:
        message_tool_calls = _ordered_tool_calls(tool_call_parts)
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(assistant_content_parts) or None,
        }
        if assistant_reasoning_parts:
            assistant_message["reasoning_text"] = "".join(assistant_reasoning_parts)
        if reasoning_opaque:
            assistant_message["reasoning_opaque"] = reasoning_opaque
        if message_tool_calls:
            assistant_message["tool_calls"] = message_tool_calls
        return AssistantTurn(message=assistant_message, tool_calls=message_tool_calls)

    def _finalize_anthropic_turn(self, final_message: Any) -> AssistantTurn:
        final_content = list(getattr(final_message, "content", []) or [])
        assistant_text_parts: list[str] = []
        assistant_thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in final_content:
            item_type = getattr(item, "type", None)
            if item_type == "thinking":
                thinking_text = cast(str | None, getattr(item, "thinking", None))
                if thinking_text:
                    assistant_thinking_parts.append(thinking_text)
            elif item_type == "text":
                text = cast(str | None, getattr(item, "text", None))
                if text:
                    assistant_text_parts.append(text)
            elif item_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(item, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": json.dumps(getattr(item, "input", {}) or {}),
                        },
                    }
                )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(assistant_text_parts) or None,
        }
        if assistant_thinking_parts:
            assistant_message["reasoning_text"] = "".join(assistant_thinking_parts)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return AssistantTurn(message=assistant_message, tool_calls=tool_calls)

    def current_skills(self) -> dict[str, SkillInfo]:
        return discover_skills(self.cwd)

    def tool_specs(self) -> list[dict[str, Any]]:
        return [tool_spec(name, self) for name in self.active_tool_names()]

    async def extract_from_markdown(self, prompt: str, markdown: str) -> str:
        limited = markdown[:50000]
        summary_model = self.weak_model
        if _is_copilot_provider(self.provider) and should_use_anthropic_messages_api(
            self.provider, summary_model
        ):
            kwargs: dict[str, Any] = {
                "model": summary_model,
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Request: {prompt}\n\nWeb content:\n\n{limited}"
                                ),
                            }
                        ],
                    }
                ],
            }
            extra_headers = self.copilot_extra_headers(initiator="user")
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            response = await self._call_with_provider_auth_retry(
                lambda: self.require_anthropic_client().messages.create(**kwargs)
            )
            self.record_usage(
                getattr(response, "usage", None),
                api="messages",
                model=getattr(response, "model", None) or summary_model,
            )
            turn = self._finalize_anthropic_turn(response)
            return cast(str, turn.message.get("content") or "")

        if should_use_responses_api(summary_model):
            kwargs: dict[str, Any] = {
                "model": summary_model,
                "instructions": "You extract only the information requested from supplied web content. If the answer is not present, say so plainly.",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Request: {prompt}\n\nWeb content:\n\n{limited}",
                            }
                        ],
                    }
                ],
            }
            extra_headers = self.copilot_extra_headers(initiator="user")
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            prompt_cache_key = (
                self.session_id
                if self.provider.strip().lower() in {"openai", "copilot"}
                else None
            )
            if prompt_cache_key:
                kwargs["prompt_cache_key"] = prompt_cache_key
            response = await self._call_with_provider_auth_retry(
                lambda: self.require_openai_client().responses.create(**kwargs)
            )
            self.record_usage(
                getattr(response, "usage", None),
                api="responses",
                model=getattr(response, "model", None) or summary_model,
            )
            plain = _to_plain_data(response) or {}
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text:
                return output_text
            fragments: list[str] = []
            for item in plain.get("output") or []:
                if not isinstance(item, dict):
                    continue
                for content_item in item.get("content") or []:
                    if isinstance(content_item, dict) and isinstance(
                        content_item.get("text"), str
                    ):
                        fragments.append(content_item["text"])
            return "".join(fragments)

        chat_messages = cast(
            list[ChatCompletionMessageParam],
            self.prepare_chat_messages(
                [
                    {
                        "role": "system",
                        "content": "You extract only the information requested from supplied web content. If the answer is not present, say so plainly.",
                    },
                    {
                        "role": "user",
                        "content": f"Request: {prompt}\n\nWeb content:\n\n{limited}",
                    },
                ]
            ),
        )
        chat_kwargs: dict[str, Any] = {
            "model": summary_model,
            "messages": chat_messages,
            "extra_headers": self.copilot_extra_headers(initiator="user"),
            "extra_body": self.chat_extra_body(),
        }
        if self.provider.strip().lower() == "openai":
            chat_kwargs["prompt_cache_key"] = self.session_id
        response = await self._call_with_provider_auth_retry(
            lambda: self.require_openai_client().chat.completions.create(**chat_kwargs)
        )
        self.record_usage(
            getattr(response, "usage", None),
            api="chat.completions",
            model=getattr(response, "model", None) or summary_model,
        )
        return response.choices[0].message.content or ""

    async def send_via_responses(self, *, initiator: str = "agent") -> None:
        while True:
            self.ui.begin_assistant()
            chat_tools = self.tool_specs()
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": self.responses_items,
                "instructions": self.system_prompt(),
                "tools": chat_tools_to_responses_tools(chat_tools),
                "tool_choice": "auto",
            }
            extra_headers = self.copilot_extra_headers(initiator=initiator)
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            prompt_cache_key = self.responses_prompt_cache_key()
            if prompt_cache_key:
                kwargs["prompt_cache_key"] = prompt_cache_key
            reasoning = self.responses_reasoning()
            if reasoning:
                kwargs["reasoning"] = reasoning

            stream_cm = await self._call_with_provider_auth_retry(
                lambda: self.require_openai_client().responses.stream(**kwargs)
            )
            async with stream_cm as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = cast(str, getattr(event, "delta", ""))
                        if delta:
                            self.ui.update_assistant(text_fragment=delta)
                    elif event_type in {
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta",
                    }:
                        delta = cast(str, getattr(event, "delta", ""))
                        if delta:
                            self.ui.update_assistant(reasoning_fragment=delta)

                final_response = await stream.get_final_response()
                self.record_usage(
                    getattr(final_response, "usage", None),
                    api="responses",
                    model=getattr(final_response, "model", None),
                )

            self.ui.end_assistant()

            turn = self._finalize_response_turn(final_response)
            self.messages.append(turn.message)
            self.responses_items.extend(turn.responses_items)

            if not turn.tool_calls:
                return

            tool_outputs = await self.execute_tool_calls(
                turn.tool_calls, call_id_key="call_id", api="responses"
            )
            self.responses_items.extend(tool_outputs)
            initiator = "agent"

    async def send_via_chat_completions(self, *, initiator: str = "agent") -> None:
        while True:
            assistant_content_parts: list[str] = []
            assistant_reasoning_parts: list[str] = []
            tool_call_parts: dict[int, dict[str, Any]] = {}
            self.ui.begin_assistant()
            chat_messages = cast(
                list[ChatCompletionMessageParam],
                self.prepare_chat_messages(
                    [
                        {"role": "system", "content": self.system_prompt()},
                        *self.messages,
                    ]
                ),
            )
            chat_tools = cast(list[ChatCompletionToolParam], self.tool_specs())

            stream = await self._call_with_provider_auth_retry(
                lambda: self.require_openai_client().chat.completions.create(
                    model=self.model,
                    messages=chat_messages,
                    tools=chat_tools,
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_headers=self.copilot_extra_headers(initiator=initiator),
                    extra_body=self.chat_extra_body(),
                    **self.completion_kwargs(),
                )
            )

            final_usage: Any = None
            response_model: str | None = None
            reasoning_opaque: str | None = None
            async for chunk in stream:
                response_model = getattr(chunk, "model", response_model)
                if getattr(chunk, "usage", None) is not None:
                    final_usage = getattr(chunk, "usage", None)
                choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
                if choice is None:
                    continue

                delta = _to_plain_data(choice.delta) or {}

                opaque = delta.get("reasoning_opaque")
                if isinstance(opaque, str) and opaque:
                    reasoning_opaque = opaque

                for fragment in _get_delta_fragments(
                    delta, ["reasoning_text", "reasoning", "thinking"]
                ):
                    if fragment:
                        self.ui.update_assistant(reasoning_fragment=fragment)
                        assistant_reasoning_parts.append(fragment)

                content = delta.get("content")
                if isinstance(content, str):
                    self.ui.update_assistant(text_fragment=content)
                    assistant_content_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        item_plain = _to_plain_data(item)
                        if isinstance(item_plain, dict):
                            for fragment in _extract_text_fragments(item_plain):
                                self.ui.update_assistant(text_fragment=fragment)
                                assistant_content_parts.append(fragment)

                for streamed_tool_call in delta.get("tool_calls") or []:
                    _merge_stream_tool_call(tool_call_parts, streamed_tool_call)

            self.record_usage(final_usage, api="chat.completions", model=response_model)

            self.ui.end_assistant()

            turn = self._finalize_chat_turn(
                assistant_content_parts,
                assistant_reasoning_parts,
                tool_call_parts,
                reasoning_opaque,
            )

            self.messages.append(turn.message)

            if not turn.tool_calls:
                return

            await self.execute_tool_calls(
                turn.tool_calls, call_id_key="id", api="chat.completions"
            )
            initiator = "agent"

    async def send_via_anthropic_messages(self, *, initiator: str = "agent") -> None:
        if self.anthropic_client is None:
            raise ValueError("Anthropic messages client is not configured")

        while True:
            self.ui.begin_assistant()
            assistant_text_parts: list[str] = []
            assistant_thinking_parts: list[str] = []

            system_prompt = self.system_prompt()
            prepared_messages = apply_copilot_ephemeral_cache(
                [{"role": "system", "content": system_prompt}, *self.messages],
                provider=self.provider,
                model=self.model,
                endpoint="/v1/messages",
            )
            system, anthropic_messages = build_anthropic_messages(prepared_messages)

            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 16384,
                "messages": anthropic_messages,
                "tools": chat_tools_to_anthropic_tools(self.tool_specs()),
            }
            if system is not None:
                kwargs["system"] = system
            thinking = self.anthropic_thinking()
            if thinking is not None:
                kwargs["thinking"] = thinking
            extra_headers = self.copilot_extra_headers(initiator=initiator)
            if extra_headers:
                kwargs["extra_headers"] = extra_headers

            stream_cm = await self._call_with_provider_auth_retry(
                lambda: self.require_anthropic_client().messages.stream(**kwargs)
            )
            async with stream_cm as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "thinking_delta":
                            fragment = cast(str, getattr(delta, "thinking", ""))
                            if fragment:
                                assistant_thinking_parts.append(fragment)
                                self.ui.update_assistant(reasoning_fragment=fragment)
                        elif delta_type == "text_delta":
                            fragment = cast(str, getattr(delta, "text", ""))
                            if fragment:
                                assistant_text_parts.append(fragment)
                                self.ui.update_assistant(text_fragment=fragment)

                final_message = await stream.get_final_message()

            usage_value = getattr(final_message, "usage", None)
            if usage_value is not None:
                self.record_usage(
                    {
                        "input_tokens": _coerce_usage_int(
                            getattr(usage_value, "input_tokens", 0)
                        ),
                        "output_tokens": _coerce_usage_int(
                            getattr(usage_value, "output_tokens", 0)
                        ),
                        "total_tokens": _coerce_usage_int(
                            getattr(usage_value, "input_tokens", 0)
                        )
                        + _coerce_usage_int(getattr(usage_value, "output_tokens", 0)),
                        "cached_input_tokens": _coerce_usage_int(
                            getattr(usage_value, "cache_read_input_tokens", 0)
                        ),
                        "reasoning_tokens": 0,
                    },
                    api="messages",
                    model=getattr(final_message, "model", None) or self.model,
                )

            self.ui.end_assistant()

            turn = self._finalize_anthropic_turn(final_message)
            if assistant_thinking_parts and not turn.message.get("reasoning_text"):
                turn.message["reasoning_text"] = "".join(assistant_thinking_parts)
            if assistant_text_parts and not turn.message.get("content"):
                turn.message["content"] = "".join(assistant_text_parts)

            self.messages.append(turn.message)

            if not turn.tool_calls:
                return

            await self.execute_tool_calls(
                turn.tool_calls, call_id_key="id", api="messages"
            )
            initiator = "agent"

    async def run_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            active_tools = set(self.active_tool_names())
            if self.debug_tools:
                self.ui.show_debug(
                    f"tool call requested: {name}; active tools: {', '.join(sorted(active_tools))}"
                )
            if name not in active_tools:
                if self.debug_tools:
                    self.ui.show_debug(f"tool call rejected: {name}")
                raise ValueError(f"tool {name!r} is not enabled for this run")
            tool = tool_definition(name)
            context = ToolCallContext(
                agent=self,
                cwd=self.cwd,
                model=self.model,
                provider=self.provider,
                arguments=arguments,
            )
            return await tool.execute(context)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def send(self, user_input: str) -> None:
        self.ui.show_user(user_input)
        self.messages.append({"role": "user", "content": user_input})
        if should_use_anthropic_messages_api(self.provider, self.model):
            await self.send_via_anthropic_messages(initiator="user")
        elif should_use_responses_api(self.model):
            self.responses_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )
            await self.send_via_responses(initiator="user")
        else:
            await self.send_via_chat_completions(initiator="user")
        self.save_history()


async def run_chat(
    model: str,
    prompt: str | None,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    weak_model: str | None = None,
    session_id: str | None = None,
    resumed: bool = False,
    messages: list[dict[str, Any]] | None = None,
    responses_items: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    created_at: str | None = None,
    allowed_tools: list[str] | None = None,
    debug_tools: bool = False,
) -> None:
    agent = NanoAgent(
        model=model,
        cwd=Path.cwd(),
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        weak_model=weak_model,
        session_id=session_id,
        messages=messages,
        responses_items=responses_items,
        usage=usage,
        allowed_tools=allowed_tools,
        debug_tools=debug_tools,
    )
    if created_at:
        agent.created_at = created_at
    try:
        if prompt is not None:
            await agent.send(prompt)
            return

        agent.ui.startup(
            provider, model, reasoning_effort, agent.session_id, resumed=resumed
        )
        while True:
            try:
                user_input = await agent.ui.prompt()
            except EOFError:
                agent.ui.newline()
                break
            if not user_input:
                continue
            if user_input in {"/exit", "exit", "quit"}:
                break
            await agent.send(user_input)
    finally:
        agent.ui.show_usage_summary(agent.usage, agent.model, agent.provider)
        agent.ui.show_session_exit(agent.session_id)
        await agent.close()
