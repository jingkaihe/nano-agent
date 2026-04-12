from __future__ import annotations

from pathlib import Path
from typing import Any

from ._common import (
    COPILOT_CLIENT_ID,
    COPILOT_DEVICE_URL,
    COPILOT_EXCHANGE_URL,
    COPILOT_SCOPES,
    COPILOT_TOKEN_URL,
    CopilotAuthError,
    httpx,
    time,
    save_copilot_credentials,
)
from .auth import copilot_token_exchange_headers
def generate_device_flow() -> dict[str, Any]:
    response = httpx.post(
        COPILOT_DEVICE_URL,
        data={
            "client_id": COPILOT_CLIENT_ID,
            "scope": " ".join(COPILOT_SCOPES),
        },
        headers={
            "Accept": "application/json",
            "User-Agent": "nano-agent",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def poll_for_token(device_code: str, interval: int, expires_in: int) -> dict[str, Any]:
    timeout_at = time.time() + expires_in
    current_interval = interval
    while time.time() < timeout_at:
        time.sleep(current_interval)
        response = httpx.post(
            COPILOT_TOKEN_URL,
            data={
                "client_id": COPILOT_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        error = data.get("error")
        if not error:
            return data
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            current_interval += 5
            continue
        raise CopilotAuthError(
            f"Authentication failed: {error} - {data.get('error_description', '')}"
        )
    raise CopilotAuthError("Authentication timed out")


def exchange_for_copilot_token(access_token: str) -> dict[str, Any]:
    response = httpx.get(
        COPILOT_EXCHANGE_URL,
        headers=copilot_token_exchange_headers(access_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def copilot_login(output_path: Path | None = None) -> dict[str, Any]:
    print("=" * 60)
    print("GitHub Copilot OAuth Login")
    print("=" * 60)
    device_response = generate_device_flow()
    print()
    print("To authenticate with GitHub Copilot:")
    print(f"  1. Open this URL in your browser: {device_response['verification_uri']}")
    print(f"  2. Enter this code when prompted: {device_response['user_code']}")
    print()
    print("Waiting for authentication to complete...")

    token_response = poll_for_token(
        device_response["device_code"],
        int(device_response["interval"]),
        int(device_response["expires_in"]),
    )
    access_token = token_response["access_token"]
    copilot_response = exchange_for_copilot_token(access_token)
    credentials = {
        "access_token": access_token,
        "copilot_token": copilot_response["token"],
        "scope": token_response.get("scope", ""),
        "copilot_expires_at": copilot_response["expires_at"],
    }
    saved = save_copilot_credentials(credentials, output_path)
    print()
    print(f"✓ Authentication successful. Credentials saved to {saved}")
    return credentials
