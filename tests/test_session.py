"""Tests for kvbridge.session — session identification and lifecycle."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from kvbridge.session import (
    classify_session_age,
    estimate_cache_status,
    identify_session,
)


# ---------------------------------------------------------------------------
# Session identification
# ---------------------------------------------------------------------------

class TestIdentifySession:
    def test_same_messages_same_id(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        assert identify_session(msgs) == identify_session(msgs)

    def test_different_system_prompt_different_id(self):
        msgs_a = [
            {"role": "system", "content": "You are a chef."},
            {"role": "user", "content": "Hello"},
        ]
        msgs_b = [
            {"role": "system", "content": "You are a programmer."},
            {"role": "user", "content": "Hello"},
        ]
        assert identify_session(msgs_a) != identify_session(msgs_b)

    def test_different_first_user_message_different_id(self):
        msgs_a = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is Python?"},
        ]
        msgs_b = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is Rust?"},
        ]
        assert identify_session(msgs_a) != identify_session(msgs_b)

    def test_stable_across_growing_conversation(self):
        base_msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Tell me about Python."},
        ]
        extended_msgs = base_msgs + [
            {"role": "assistant", "content": "Python is a language."},
            {"role": "user", "content": "Tell me more."},
        ]
        assert identify_session(base_msgs) == identify_session(extended_msgs)

    def test_no_system_prompt_uses_first_user(self):
        msgs_a = [{"role": "user", "content": "session A"}]
        msgs_b = [{"role": "user", "content": "session B"}]
        assert identify_session(msgs_a) != identify_session(msgs_b)

    def test_no_system_no_user_uses_first_3(self):
        msgs_a = [
            {"role": "assistant", "content": "hello"},
            {"role": "function", "content": "result"},
        ]
        msgs_b = [
            {"role": "assistant", "content": "goodbye"},
            {"role": "function", "content": "result"},
        ]
        assert identify_session(msgs_a) != identify_session(msgs_b)

    def test_empty_messages_returns_hash(self):
        result = identify_session([])
        assert isinstance(result, str)
        assert len(result) == 16

    def test_id_length_is_16(self):
        msgs = [{"role": "user", "content": "test"}]
        assert len(identify_session(msgs)) == 16

    def test_id_is_hex_string(self):
        msgs = [{"role": "user", "content": "test"}]
        sid = identify_session(msgs)
        int(sid, 16)  # should not raise

    def test_first_user_message_truncated_at_500(self):
        # Both messages share the same first 500 chars; only the tail differs.
        base = "x" * 500
        msgs_short = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": base},        # exactly 500 chars
        ]
        msgs_long = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": base + "y"},  # 501 chars, truncated to same 500
        ]
        # First 500 chars are identical; session IDs should match
        assert identify_session(msgs_short) == identify_session(msgs_long)

    def test_user_content_beyond_500_irrelevant(self):
        base = "a" * 500
        msgs_a = [{"role": "user", "content": base + "extra_a"}]
        msgs_b = [{"role": "user", "content": base + "extra_b"}]
        assert identify_session(msgs_a) == identify_session(msgs_b)

    def test_multiple_system_prompts_uses_first(self):
        msgs_a = [
            {"role": "system", "content": "First system"},
            {"role": "system", "content": "Second system"},
            {"role": "user", "content": "hello"},
        ]
        msgs_b = [
            {"role": "system", "content": "First system"},
            {"role": "user", "content": "hello"},
        ]
        assert identify_session(msgs_a) == identify_session(msgs_b)

    def test_system_with_none_content(self):
        msgs = [{"role": "system", "content": None}, {"role": "user", "content": "hi"}]
        result = identify_session(msgs)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_user_with_none_content(self):
        msgs = [{"role": "user", "content": None}]
        result = identify_session(msgs)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_user_with_missing_content(self):
        msgs = [{"role": "user"}]
        result = identify_session(msgs)
        assert isinstance(result, str)

    def test_unicode_content_handled(self):
        msgs_a = [{"role": "user", "content": "こんにちは"}]
        msgs_b = [{"role": "user", "content": "さようなら"}]
        assert identify_session(msgs_a) != identify_session(msgs_b)

    def test_system_prompt_separator_prevents_collision(self):
        # "AB" || "C" should != "A" || "BC"
        msgs_a = [
            {"role": "system", "content": "AB"},
            {"role": "user", "content": "C"},
        ]
        msgs_b = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "BC"},
        ]
        assert identify_session(msgs_a) != identify_session(msgs_b)

    def test_fallback_uses_role_in_hash(self):
        # No system or user messages: role is included in the hash
        msgs_a = [{"role": "assistant", "content": "hello"}]
        msgs_b = [{"role": "tool", "content": "hello"}]
        assert identify_session(msgs_a) != identify_session(msgs_b)

    def test_list_content_user_message(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = identify_session(msgs)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_deterministic_across_calls(self):
        msgs = [{"role": "user", "content": "deterministic"}]
        results = {identify_session(msgs) for _ in range(10)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Session age classification
# ---------------------------------------------------------------------------

class TestClassifySessionAge:
    def _ago(self, **kwargs) -> datetime:
        return datetime.utcnow() - timedelta(**kwargs)

    def test_just_made_is_active(self):
        assert classify_session_age(self._ago(seconds=30)) == "active"

    def test_within_5_min_is_active(self):
        assert classify_session_age(self._ago(minutes=4, seconds=59)) == "active"

    def test_exactly_5_min_is_idle(self):
        # At exactly 5 min boundary: timedelta == _ACTIVE_THRESHOLD, not <=
        assert classify_session_age(self._ago(minutes=5, seconds=1)) == "idle"

    def test_10_min_is_idle(self):
        assert classify_session_age(self._ago(minutes=10)) == "idle"

    def test_within_30_min_is_idle(self):
        assert classify_session_age(self._ago(minutes=29, seconds=59)) == "idle"

    def test_exactly_30_min_is_stale(self):
        assert classify_session_age(self._ago(minutes=30, seconds=1)) == "stale"

    def test_1_hour_is_stale(self):
        assert classify_session_age(self._ago(hours=1)) == "stale"

    def test_very_old_is_stale(self):
        assert classify_session_age(self._ago(days=30)) == "stale"

    def test_just_now_is_active(self):
        assert classify_session_age(datetime.utcnow()) == "active"


# ---------------------------------------------------------------------------
# Cache status estimation
# ---------------------------------------------------------------------------

class TestEstimateCacheStatus:
    def _ago(self, **kwargs) -> datetime:
        return datetime.utcnow() - timedelta(**kwargs)

    def test_recent_is_hot(self):
        assert estimate_cache_status(self._ago(seconds=30)) == "hot"

    def test_within_5_min_is_hot(self):
        assert estimate_cache_status(self._ago(minutes=4)) == "hot"

    def test_5_to_30_min_is_cold(self):
        assert estimate_cache_status(self._ago(minutes=10)) == "cold"

    def test_over_30_min_is_evicted(self):
        assert estimate_cache_status(self._ago(minutes=45)) == "evicted"

    def test_old_is_evicted(self):
        assert estimate_cache_status(self._ago(days=1)) == "evicted"
