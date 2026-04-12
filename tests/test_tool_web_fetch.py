from __future__ import annotations

from nano_agent.tools.web_fetch import validate_fetch_url


def test_validate_fetch_url() -> None:
    validate_fetch_url("https://example.com/test")
    validate_fetch_url("http://localhost:8080/test")
    try:
        validate_fetch_url("http://example.com/test")
    except ValueError:
        return
    raise AssertionError("validate_fetch_url should reject external http URLs")
