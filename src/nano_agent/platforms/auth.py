from __future__ import annotations

from typing import Any

from ._common import (
    ANTHROPIC_API_VERSION,
    COPILOT_CHAT_PLUGIN_VERSION,
    COPILOT_CHAT_USER_AGENT,
    COPILOT_EDITOR_VERSION,
    COPILOT_GITHUB_API_VERSION,
    COPILOT_INTEGRATION_ID,
    COPILOT_VSCODE_USER_AGENT_LIBRARY_VERSION,
    anthropic,
    inspect,
    openai,
    os,
    uuid,
)


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


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "Accept": "application/json",
    }


__all__ = [
    "copilot_anthropic_headers",
    "copilot_api_headers",
    "copilot_base_url",
    "copilot_token_exchange_headers",
    "is_copilot_auth_error",
]
