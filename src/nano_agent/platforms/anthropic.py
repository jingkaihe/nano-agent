from __future__ import annotations

from typing import Any

from ._common import AsyncAnthropic, os


def resolve_anthropic_api_key(api_key: str | None) -> str:
    resolved_api_key = (
        api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("NANO_AGENT_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY or pass --api-key."
        )
    return resolved_api_key


def resolve_anthropic_base_url(base_url: str | None) -> str | None:
    return (
        base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("NANO_AGENT_BASE_URL")
    )


def create_direct_anthropic_client(
    api_key: str, base_url: str | None = None
) -> AsyncAnthropic:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncAnthropic(**kwargs)


def _normalize_anthropic_model(model: dict[str, Any]) -> dict[str, Any]:
    capabilities = model.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    existing_limits = capabilities.get("limits")
    limits_payload: dict[str, Any] = (
        dict(existing_limits) if isinstance(existing_limits, dict) else {}
    )
    existing_supports = capabilities.get("supports")
    supports_payload: dict[str, Any] = (
        dict(existing_supports) if isinstance(existing_supports, dict) else {}
    )

    limits = {
        "max_context_window_tokens": model.get("max_input_tokens"),
        "max_output_tokens": model.get("max_tokens"),
    }
    supports = {
        "reasoning_effort": ["minimal", "low", "medium", "high"],
    }

    return {
        "id": model.get("id"),
        "name": model.get("display_name") or model.get("id"),
        "vendor": "anthropic",
        "version": "",
        "preview": False,
        "supported_endpoints": ["/v1/messages"],
        "capabilities": {
            **capabilities,
            "family": model.get("id"),
            "type": model.get("type") or capabilities.get("type") or "model",
            "limits": {
                **limits_payload,
                **{k: v for k, v in limits.items() if isinstance(v, (int, float))},
            },
            "supports": {
                **supports_payload,
                **supports,
            },
            "input_modalities": ["text", "image"],
        },
    }


__all__ = [
    "create_direct_anthropic_client",
    "resolve_anthropic_api_key",
    "resolve_anthropic_base_url",
]
