from __future__ import annotations

from ._common import (
    AsyncAnthropic,
    AsyncOpenAI,
    COPILOT_EDITOR_VERSION,
    COPILOT_OPENAI_USER_AGENT,
    load_copilot_credentials,
    refresh_copilot_token,
)
from .auth import (
    copilot_anthropic_headers,
    copilot_api_headers,
    copilot_base_url,
    copilot_token_exchange_headers,
)
from .login import (
    copilot_login,
    exchange_for_copilot_token,
    generate_device_flow,
    poll_for_token,
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

__all__ = [
    "copilot_anthropic_headers",
    "copilot_api_headers",
    "copilot_base_url",
    "copilot_login",
    "copilot_token_exchange_headers",
    "create_anthropic_copilot_client",
    "create_openai_copilot_client",
    "exchange_for_copilot_token",
    "generate_device_flow",
    "poll_for_token",
]
