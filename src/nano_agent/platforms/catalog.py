from __future__ import annotations

from typing import Any

from ._common import (
    DEFAULT_ANTHROPIC_BASE_URL,
    Console,
    Table,
    _legacy_context_window_limit,
    core,
    httpx,
    load_cached_provider_models,
    load_copilot_credentials,
    refresh_copilot_token,
    save_cached_provider_models,
    time,
)
from .anthropic import (
    _normalize_anthropic_model,
    resolve_anthropic_api_key,
    resolve_anthropic_base_url,
)
from .auth import _anthropic_headers, _set_copilot_models_cache, copilot_api_headers
from .copilot import copilot_base_url
from .openai import resolve_openai_api_key, resolve_openai_base_url


def list_provider_models(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    force_reload: bool = False,
) -> list[dict[str, Any]]:
    normalized = provider.strip().lower()

    if normalized == "copilot":
        if (
            not force_reload
            and api_key is None
            and base_url is None
            and core.COPILOT_MODELS_CACHE is not None
            and isinstance(core.COPILOT_MODELS_CACHE_EXPIRES_AT, (int, float))
            and time.time() < core.COPILOT_MODELS_CACHE_EXPIRES_AT
        ):
            return [dict(item) for item in core.COPILOT_MODELS_CACHE]
        if not force_reload and api_key is None and base_url is None:
            cached_models, expires_at = load_cached_provider_models("copilot")
            if cached_models is not None:
                _set_copilot_models_cache(cached_models, expires_at=expires_at)
                return [dict(item) for item in cached_models]
        creds = load_copilot_credentials()
        copilot_token, _ = refresh_copilot_token(creds)
        response = httpx.get(
            f"{copilot_base_url()}/models",
            headers=copilot_api_headers(copilot_token),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Copilot models response is invalid")
        models = [item for item in payload["data"] if isinstance(item, dict)]
        if api_key is None and base_url is None:
            expires_at = save_cached_provider_models("copilot", models)
            _set_copilot_models_cache(models, expires_at=expires_at)
        return models

    if normalized == "openai":
        resolved_api_key = resolve_openai_api_key(api_key)
        resolved_base_url = (
            resolve_openai_base_url(base_url) or "https://api.openai.com/v1"
        )
        response = httpx.get(
            f"{resolved_base_url.rstrip('/')}/models",
            headers={
                "Authorization": f"Bearer {resolved_api_key}",
                "Accept": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("OpenAI models response is invalid")
        return [item for item in payload["data"] if isinstance(item, dict)]

    if normalized == "anthropic":
        resolved_api_key = resolve_anthropic_api_key(api_key)
        resolved_base_url = (
            resolve_anthropic_base_url(base_url) or DEFAULT_ANTHROPIC_BASE_URL
        )
        response = httpx.get(
            f"{resolved_base_url.rstrip('/')}/v1/models",
            headers=_anthropic_headers(resolved_api_key),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Anthropic models response is invalid")
        return [
            _normalize_anthropic_model(item)
            for item in payload["data"]
            if isinstance(item, dict)
        ]

    raise ValueError(f"unsupported provider: {provider}")


def _model_catalog_entry(
    model: str, provider: str = "copilot"
) -> dict[str, Any] | None:
    normalized_model = model.strip().lower()
    for item in list_provider_models(provider):
        model_id = item.get("id")
        version = item.get("version")
        family = (
            item.get("capabilities", {}).get("family")
            if isinstance(item.get("capabilities"), dict)
            else None
        )
        candidates = [model_id, version, family]
        if any(
            isinstance(candidate, str) and candidate.strip().lower() == normalized_model
            for candidate in candidates
        ):
            return item
    return None


def model_supports_endpoint(
    endpoint: str, model: str, provider: str = "copilot"
) -> bool:
    try:
        entry = _model_catalog_entry(model, provider=provider)
    except Exception:
        return False
    supported_endpoints = (
        entry.get("supported_endpoints") if isinstance(entry, dict) else None
    )
    if supported_endpoints is None:
        return True
    if not isinstance(supported_endpoints, list):
        return False
    if not supported_endpoints:
        return True
    normalized_endpoint = endpoint.strip().lower()
    return any(
        isinstance(candidate, str) and candidate.strip().lower() == normalized_endpoint
        for candidate in supported_endpoints
    )


def model_context_window_limit(model: str, provider: str = "copilot") -> int | None:
    try:
        entry = _model_catalog_entry(model, provider=provider)
    except Exception:
        entry = None
    limits = (
        entry.get("capabilities", {}).get("limits")
        if isinstance(entry, dict) and isinstance(entry.get("capabilities"), dict)
        else {}
    )
    value = (
        limits.get("max_context_window_tokens") if isinstance(limits, dict) else None
    )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return _legacy_context_window_limit(model)


def model_supports_thinking(model: dict[str, Any]) -> str:
    raw_capabilities = model.get("capabilities")
    capabilities: dict[str, Any] = (
        raw_capabilities if isinstance(raw_capabilities, dict) else {}
    )
    raw_supports = capabilities.get("supports")
    supports: dict[str, Any] = raw_supports if isinstance(raw_supports, dict) else {}
    reasoning_effort = supports.get("reasoning_effort")
    adaptive_thinking = supports.get("adaptive_thinking")
    min_budget = supports.get("min_thinking_budget")
    max_budget = supports.get("max_thinking_budget")

    if isinstance(reasoning_effort, list) and reasoning_effort:
        levels = "/".join(str(level) for level in reasoning_effort)
        budget = ""
        if isinstance(min_budget, (int, float)) and isinstance(
            max_budget, (int, float)
        ):
            budget = f" [{int(min_budget):,}-{int(max_budget):,}]"
        mode = " adaptive" if adaptive_thinking is True else ""
        return f"yes ({levels}){budget}{mode}"

    if (
        adaptive_thinking is True
        or isinstance(min_budget, (int, float))
        or isinstance(max_budget, (int, float))
    ):
        budget = ""
        if isinstance(min_budget, (int, float)) and isinstance(
            max_budget, (int, float)
        ):
            budget = f" [{int(min_budget):,}-{int(max_budget):,}]"
        return f"yes{budget}"

    return ""


def render_models_table(models: list[dict[str, Any]], provider: str) -> None:
    console = Console()
    if not models:
        console.print("No models returned.")
        return

    normalized = provider.strip().lower()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", style="green")
    table.add_column("Name")
    table.add_column("Vendor")
    table.add_column("Type")
    table.add_column("Context")
    table.add_column("Thinking")
    table.add_column("Version")
    table.add_column("Preview", style="dim")

    def sort_key(item: dict[str, Any]) -> str:
        model_id = item.get("id")
        return model_id if isinstance(model_id, str) else ""

    for model in sorted(models, key=sort_key):
        model_id = model.get("id")
        name = model.get("name")
        vendor = model.get("vendor")
        version = model.get("version")
        preview = model.get("preview")
        raw_capabilities = model.get("capabilities")
        capabilities: dict[str, Any] = (
            raw_capabilities if isinstance(raw_capabilities, dict) else {}
        )
        model_type = capabilities.get("type")
        raw_limits = capabilities.get("limits")
        limits: dict[str, Any] = raw_limits if isinstance(raw_limits, dict) else {}
        context_limit = limits.get("max_context_window_tokens")
        context_text = (
            f"{int(context_limit):,}" if isinstance(context_limit, (int, float)) else ""
        )
        thinking = model_supports_thinking(model)

        if normalized == "openai":
            vendor = model.get("owned_by")
            version = ""
            preview = ""
            model_type = model.get("object")

        table.add_row(
            model_id if isinstance(model_id, str) else "",
            name if isinstance(name, str) else "",
            vendor if isinstance(vendor, str) else "",
            model_type if isinstance(model_type, str) else "",
            context_text,
            thinking,
            version if isinstance(version, str) else "",
            str(preview).lower() if isinstance(preview, bool) else "",
        )

    console.print(table)

__all__ = [
    "_model_catalog_entry",
    "list_provider_models",
    "model_context_window_limit",
    "model_supports_endpoint",
    "model_supports_thinking",
    "render_models_table",
]
