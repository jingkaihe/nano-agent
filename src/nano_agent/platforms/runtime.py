from __future__ import annotations

from ._common import AsyncAnthropic, AsyncOpenAI, should_use_anthropic_messages_api
from .anthropic import (
    create_direct_anthropic_client,
    resolve_anthropic_api_key,
    resolve_anthropic_base_url,
)
from .auth import _close_async_client, _is_async_context_manager, is_copilot_auth_error
from .copilot import (
    create_anthropic_copilot_client,
    create_openai_copilot_client,
)
from .openai import (
    create_direct_openai_client,
    resolve_openai_api_key,
    resolve_openai_base_url,
)


def create_client(
    provider: str, api_key: str | None = None, base_url: str | None = None
) -> AsyncOpenAI:
    normalized = provider.strip().lower()
    if normalized == "copilot":
        return create_openai_copilot_client()
    if normalized == "openai":
        return create_direct_openai_client(
            resolve_openai_api_key(api_key),
            resolve_openai_base_url(base_url),
        )
    if normalized == "anthropic":
        raise ValueError(
            "provider anthropic uses the Anthropic Messages API and does not expose an OpenAI-compatible client"
        )
    raise ValueError(f"unsupported provider: {provider}")


def create_runtime_clients(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[AsyncOpenAI | None, AsyncAnthropic | None]:
    normalized = provider.strip().lower()
    if normalized == "copilot":
        openai_client = create_openai_copilot_client()
        anthropic_client = (
            create_anthropic_copilot_client()
            if should_use_anthropic_messages_api(provider, model)
            else None
        )
        return openai_client, anthropic_client
    if normalized == "openai":
        return create_direct_openai_client(
            resolve_openai_api_key(api_key),
            resolve_openai_base_url(base_url),
        ), None
    if normalized == "anthropic":
        return None, create_direct_anthropic_client(
            resolve_anthropic_api_key(api_key),
            resolve_anthropic_base_url(base_url),
        )
    raise ValueError(f"unsupported provider: {provider}")


__all__ = [
    "_close_async_client",
    "_is_async_context_manager",
    "create_client",
    "create_runtime_clients",
    "is_copilot_auth_error",
]
