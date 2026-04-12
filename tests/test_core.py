from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import nano_agent.core as core

from .helpers import assert_equal, assert_true


def test_normalize_session_id() -> None:
    assert_equal(core.normalize_session_id("abc.json"), "abc")
    assert_equal(core.normalize_session_id(" 123 "), "123")
    try:
        core.normalize_session_id("a/b")
    except ValueError:
        return
    raise AssertionError("normalize_session_id should reject paths")


def test_latest_conversation_session_id() -> None:
    original_history_dir = core.HISTORY_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            core.HISTORY_DIR = Path(tmp)

            try:
                core.latest_conversation_session_id()
            except ValueError:
                pass
            else:
                raise AssertionError(
                    "latest_conversation_session_id should fail when history is empty"
                )

            history_dir = cast(Path, core.HISTORY_DIR)
            history_dir.mkdir(parents=True, exist_ok=True)
            (history_dir / "older.json").write_text(
                json.dumps(
                    {
                        "id": "older",
                        "provider": "copilot",
                        "model": "claude-sonnet-4.5",
                        "created_at": "2026-03-10T00:00:00Z",
                        "updated_at": "2026-03-10T00:00:00Z",
                        "history": {"messages": []},
                    }
                )
            )
            (history_dir / "newer.json").write_text(
                json.dumps(
                    {
                        "id": "newer",
                        "provider": "copilot",
                        "model": "claude-sonnet-4.5",
                        "created_at": "2026-03-11T00:00:00Z",
                        "updated_at": "2026-03-12T00:00:00Z",
                        "history": {"messages": []},
                    }
                )
            )

            assert_equal(core.latest_conversation_session_id(), "newer")
    finally:
        core.HISTORY_DIR = original_history_dir


def test_default_weak_model() -> None:
    assert_equal(
        core.default_weak_model("copilot", "claude-sonnet-4.5"), "claude-haiku-4.5"
    )
    assert_equal(core.default_weak_model("copilot", "gpt-5"), "gpt-4.1")
    assert_equal(core.default_weak_model("openai", "gpt-5"), "gpt-4.1")


def test_models_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models.json"
        now = datetime(2026, 3, 13, tzinfo=timezone.utc).timestamp()
        models = [
            {"id": "claude-sonnet-4.5", "supported_endpoints": ["/chat/completions"]}
        ]

        expires_at = core.save_cached_provider_models(
            "copilot",
            models,
            fetched_at="2026-03-13T00:00:00Z",
            path=path,
        )
        assert_equal(expires_at, now + core.MODELS_CACHE_TTL_SECONDS)

        loaded_models, loaded_expires_at = core.load_cached_provider_models(
            "copilot",
            ttl_seconds=core.MODELS_CACHE_TTL_SECONDS,
            path=path,
            now=now + 60,
        )
        assert_equal(loaded_models, models)
        assert_equal(loaded_expires_at, expires_at)

        stale_models, stale_expires_at = core.load_cached_provider_models(
            "copilot",
            ttl_seconds=core.MODELS_CACHE_TTL_SECONDS,
            path=path,
            now=expires_at,
        )
        assert_equal(stale_models, None)
        assert_equal(stale_expires_at, expires_at)


def test_apply_copilot_ephemeral_cache() -> None:
    original_cache = core.COPILOT_MODELS_CACHE
    original_cache_expires_at = core.COPILOT_MODELS_CACHE_EXPIRES_AT
    core.COPILOT_MODELS_CACHE = [
        {
            "id": "claude-chat",
            "version": "claude-chat",
            "supported_endpoints": ["/v1/messages", "/chat/completions"],
        },
        {
            "id": "claude-messages-only",
            "version": "claude-messages-only",
            "supported_endpoints": ["/v1/messages"],
        },
        {
            "id": "claude-unspecified-endpoints",
            "version": "claude-unspecified-endpoints",
            "supported_endpoints": [],
        },
        {
            "id": "claude-missing-endpoints",
            "version": "claude-missing-endpoints",
        },
    ]
    core.COPILOT_MODELS_CACHE_EXPIRES_AT = time.time() + core.MODELS_CACHE_TTL_SECONDS

    try:
        messages = [
            {"role": "system", "content": "sys-1"},
            {"role": "system", "content": [{"type": "text", "text": "sys-2"}]},
            {"role": "user", "content": "user-1"},
            {"role": "assistant", "content": "assistant-1"},
            {"role": "user", "content": [{"type": "text", "text": "user-2"}]},
        ]

        patched = core.apply_copilot_ephemeral_cache(
            messages,
            provider="copilot",
            model="claude-chat",
            endpoint="/chat/completions",
        )
        assert_true(
            patched is not messages,
            "cache application should return a copied message list",
        )
        assert_equal(patched[0]["copilot_cache_control"]["type"], "ephemeral")
        assert_equal(
            patched[1]["content"][-1]["copilot_cache_control"]["type"], "ephemeral"
        )
        assert_equal(patched[3]["copilot_cache_control"]["type"], "ephemeral")
        assert_equal(
            patched[4]["content"][-1]["copilot_cache_control"]["type"], "ephemeral"
        )
        assert_true(
            "copilot_cache_control" not in messages[0],
            "original messages should remain unchanged",
        )

        unchanged = core.apply_copilot_ephemeral_cache(
            messages,
            provider="copilot",
            model="claude-messages-only",
            endpoint="/chat/completions",
        )
        assert_true(unchanged is messages, "messages-only models should not be patched")

        unspecified = core.apply_copilot_ephemeral_cache(
            messages,
            provider="copilot",
            model="claude-unspecified-endpoints",
            endpoint="/chat/completions",
        )
        assert_true(
            unspecified is not messages,
            "empty supported_endpoints should allow patching",
        )

        missing = core.apply_copilot_ephemeral_cache(
            messages,
            provider="copilot",
            model="claude-missing-endpoints",
            endpoint="/chat/completions",
        )
        assert_true(
            missing is not messages,
            "missing supported_endpoints should allow patching",
        )

        wrong_provider = core.apply_copilot_ephemeral_cache(
            messages,
            provider="openai",
            model="claude-chat",
            endpoint="/chat/completions",
        )
        assert_true(
            wrong_provider is messages, "non-copilot providers should not be patched"
        )

        messages_api = core.apply_copilot_ephemeral_cache(
            messages,
            provider="copilot",
            model="claude-messages-only",
            endpoint="/v1/messages",
        )
        assert_true(
            messages_api is not messages,
            "messages endpoint should also receive copilot cache markers",
        )
        assert_equal(messages_api[0]["copilot_cache_control"]["type"], "ephemeral")
    finally:
        core.COPILOT_MODELS_CACHE = original_cache
        core.COPILOT_MODELS_CACHE_EXPIRES_AT = original_cache_expires_at
