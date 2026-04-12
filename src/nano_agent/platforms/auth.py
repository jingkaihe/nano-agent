from __future__ import annotations

from typing import Any

from ._common import anthropic, inspect, openai
def _set_copilot_models_cache(
    models: list[dict[str, Any]], *, expires_at: float | None
) -> None:
    import nano_agent.core as core

    core.COPILOT_MODELS_CACHE = [dict(item) for item in models]
    core.COPILOT_MODELS_CACHE_EXPIRES_AT = expires_at

async def _close_async_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _copilot_auth_error_status(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _looks_like_copilot_auth_error(exc: Exception) -> bool:
    status = _copilot_auth_error_status(exc)
    if status in {401, 403}:
        return True
    if status != 400:
        return False

    message = str(exc).strip().lower()
    return (
        "authorization header is badly formatted" in message
        or "authorization header" in message
        or "invalid authorization" in message
    )


def is_copilot_auth_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
        ),
    ):
        return True

    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return _looks_like_copilot_auth_error(exc)

    return False


def _is_async_context_manager(value: Any) -> bool:
    if callable(getattr(value, "__aiter__", None)):
        return False
    return callable(getattr(value, "__aenter__", None)) and callable(
        getattr(value, "__aexit__", None)
    )


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
