from __future__ import annotations

from typing import Any

from ._common import AsyncAnthropic, AsyncOpenAI, os
from .auth import copilot_anthropic_headers, copilot_base_url
from ._common import load_copilot_credentials, refresh_copilot_token
def create_openai_copilot_client() -> AsyncOpenAI:
    base_url = copilot_base_url()
    creds = load_copilot_credentials()
    copilot_token, _ = refresh_copilot_token(creds)
    return AsyncOpenAI(
        api_key=copilot_token,
        base_url=base_url,
        default_headers={
            "User-Agent": COPILOT_OPENAI_USER_AGENT,
            "Editor-Version": COPILOT_EDITOR_VERSION,
        },
    )


def create_anthropic_copilot_client() -> AsyncAnthropic:
    creds = load_copilot_credentials()
    copilot_token, _ = refresh_copilot_token(creds)
    return AsyncAnthropic(
        auth_token=copilot_token,
        base_url=copilot_base_url(),
        default_headers=copilot_anthropic_headers(),
    )


def create_direct_openai_client(
    api_key: str, base_url: str | None = None
) -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


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
        base_url
        or os.getenv("ANTHROPIC_BASE_URL")
        or os.getenv("NANO_AGENT_BASE_URL")
    )


def create_direct_anthropic_client(
    api_key: str, base_url: str | None = None
) -> AsyncAnthropic:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncAnthropic(**kwargs)


def copilot_base_url() -> str:
    return (
        "https://api.business.githubcopilot.com"
        if os.getenv("BUSINESS_COPILOT") == "true"
        else "https://api.githubcopilot.com"
    )


def copilot_token_exchange_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "Accept": "application/json",
    }


def copilot_api_headers(copilot_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {copilot_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": COPILOT_CHAT_USER_AGENT,
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "Editor-Plugin-Version": COPILOT_CHAT_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "OpenAI-Intent": "conversation-panel",
        "X-GitHub-Api-Version": COPILOT_GITHUB_API_VERSION,
        "X-Request-Id": str(uuid.uuid4()),
        "X-Vscode-User-Agent-Library-Version": COPILOT_VSCODE_USER_AGENT_LIBRARY_VERSION,
    }


def copilot_anthropic_headers() -> dict[str, str]:
    return {
        "User-Agent": COPILOT_CHAT_USER_AGENT,
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "Editor-Plugin-Version": COPILOT_CHAT_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "OpenAI-Intent": "conversation-panel",
        "X-GitHub-Api-Version": COPILOT_GITHUB_API_VERSION,
        "X-Vscode-User-Agent-Library-Version": COPILOT_VSCODE_USER_AGENT_LIBRARY_VERSION,
    }


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


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "Accept": "application/json",
    }


def _normalize_anthropic_model(model: dict[str, Any]) -> dict[str, Any]:
    capabilities = model.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}

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
                **(
                    capabilities.get("limits")
                    if isinstance(capabilities.get("limits"), dict)
                    else {}
                ),
                **{k: v for k, v in limits.items() if isinstance(v, (int, float))},
            },
            "supports": {
                **(
                    capabilities.get("supports")
                    if isinstance(capabilities.get("supports"), dict)
                    else {}
                ),
                **supports,
            },
            "input_modalities": ["text", "image"],
        },
    }


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


def create_client(
    provider: str, api_key: str | None = None, base_url: str | None = None
) -> AsyncOpenAI:
    normalized = provider.strip().lower()
    if normalized == "copilot":
        return create_openai_copilot_client()
    if normalized == "openai":
        resolved_api_key = (
            api_key or os.getenv("OPENAI_API_KEY") or os.getenv("NANO_AGENT_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass --api-key."
            )
        resolved_base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("NANO_AGENT_BASE_URL")
        )
        return create_direct_openai_client(resolved_api_key, resolved_base_url)
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
            resolve_openai_api_key(api_key), resolve_openai_base_url(base_url)
        ), None
    if normalized == "anthropic":
        return None, create_direct_anthropic_client(
            resolve_anthropic_api_key(api_key), resolve_anthropic_base_url(base_url)
        )
    raise ValueError(f"unsupported provider: {provider}")
