from __future__ import annotations

from typing import Any

from ._common import AsyncOpenAI, os


def create_direct_openai_client(
    api_key: str, base_url: str | None = None
) -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def resolve_openai_api_key(api_key: str | None) -> str:
    resolved_api_key = (
        api_key or os.getenv("OPENAI_API_KEY") or os.getenv("NANO_AGENT_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY or pass --api-key."
        )
    return resolved_api_key


def resolve_openai_base_url(base_url: str | None) -> str | None:
    return base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("NANO_AGENT_BASE_URL")

__all__ = [
    "create_direct_openai_client",
    "resolve_openai_api_key",
    "resolve_openai_base_url",
]
