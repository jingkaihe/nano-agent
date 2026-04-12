from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import openai
import nano_agent.core as core
import nano_agent.platforms as platforms

from .helpers import assert_equal, assert_true


def test_copilot_headers() -> None:
    exchange_headers = platforms.copilot_token_exchange_headers("token")
    api_headers = platforms.copilot_api_headers("copilot")
    assert_equal(exchange_headers["Editor-Version"], core.COPILOT_EDITOR_VERSION)
    assert_equal(api_headers["Copilot-Integration-Id"], core.COPILOT_INTEGRATION_ID)
    assert_true("X-Request-Id" in api_headers, "api headers should include request id")


def test_is_copilot_auth_error() -> None:
    auth_exc = openai.AuthenticationError(
        "unauthorized",
        response=httpx.Response(401, request=httpx.Request("GET", "https://example.com")),
        body=None,
    )
    forbidden_exc = openai.PermissionDeniedError(
        "forbidden",
        response=httpx.Response(403, request=httpx.Request("GET", "https://example.com")),
        body=None,
    )
    bad_request_exc = openai.APIStatusError(
        "bad request",
        response=httpx.Response(400, request=httpx.Request("GET", "https://example.com")),
        body=None,
    )
    bad_auth_header_exc = openai.BadRequestError(
        "bad request: Authorization header is badly formatted",
        response=httpx.Response(400, request=httpx.Request("GET", "https://example.com")),
        body=None,
    )

    assert_true(platforms.is_copilot_auth_error(auth_exc), "401 auth errors should be retried")
    assert_true(
        platforms.is_copilot_auth_error(forbidden_exc),
        "403 permission errors should be retried",
    )
    assert_true(
        not platforms.is_copilot_auth_error(bad_request_exc),
        "non-auth API errors should not be treated as refreshable",
    )
    assert_true(
        platforms.is_copilot_auth_error(bad_auth_header_exc),
        "badly formatted auth headers should trigger Copilot re-auth",
    )


def test_refresh_copilot_token_skips_valid_token() -> None:
    creds = {
        "access_token": "github-token",
        "copilot_token": "copilot-token",
        "copilot_expires_at": time.time() + 3600,
    }
    token, refreshed = core.refresh_copilot_token(creds)
    assert_equal(token, "copilot-token")
    assert_true(refreshed is creds, "refresh should reuse provided credentials dict")
    assert_equal(refreshed["copilot_token"], "copilot-token")


def test_refresh_copilot_token_renews_expiring_token(monkeypatch: Any) -> None:
    saved_creds: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        assert_equal(url, core.COPILOT_EXCHANGE_URL)
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "token": "fresh-copilot-token",
                "expires_at": "2099-01-01T00:00:00Z",
            },
            request=request,
        )

    def fake_save(credentials: dict[str, Any], output_path: Path | None = None) -> Path:
        saved_creds.append(dict(credentials))
        return output_path or Path("/tmp/copilot.json")

    monkeypatch.setattr(core.httpx, "get", fake_get)
    monkeypatch.setattr(core, "save_copilot_credentials", fake_save)

    creds = {
        "access_token": "github-token",
        "copilot_token": "stale-copilot-token",
        "copilot_expires_at": time.time() + 60,
    }
    token, refreshed = core.refresh_copilot_token(creds)
    assert_equal(token, "fresh-copilot-token")
    assert_equal(refreshed["copilot_token"], "fresh-copilot-token")
    assert_equal(refreshed["copilot_expires_at"], "2099-01-01T00:00:00Z")
    assert_equal(len(saved_creds), 1)
    assert_equal(saved_creds[0]["copilot_token"], "fresh-copilot-token")


def test_refresh_copilot_token_wraps_http_error(monkeypatch: Any) -> None:
    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("GET", url)
        return httpx.Response(401, request=request)

    monkeypatch.setattr(core.httpx, "get", fake_get)

    creds = {
        "access_token": "github-token",
        "copilot_token": "stale-copilot-token",
        "copilot_expires_at": time.time() + 60,
    }

    try:
        core.refresh_copilot_token(creds)
    except core.CopilotAuthError as exc:
        assert_true(
            "re-authenticate" in str(exc),
            "refresh failures should instruct the user to log in again",
        )
        return

    raise AssertionError("refresh_copilot_token should wrap HTTP errors")
