from __future__ import annotations

from typing import Any


def assert_equal(actual: Any, expected: Any, message: str = "") -> None:
    if actual != expected:
        detail = message or f"expected {expected!r}, got {actual!r}"
        raise AssertionError(detail)


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)
