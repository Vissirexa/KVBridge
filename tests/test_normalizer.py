"""Tests for kvbridge.normalizer — covers normalization rules and edge cases."""
from __future__ import annotations

import pytest

from kvbridge.normalizer import (
    normalize_messages,
    _normalize_content,
    _normalize_string,
    _normalize_tool_calls,
)


# ---------------------------------------------------------------------------
# Rule 1 — Strip non-standard fields
# ---------------------------------------------------------------------------

class TestStripNonStandardFields:
    def test_strips_timestamp(self):
        msgs = [{"role": "user", "content": "hi", "timestamp": "2024-01-01"}]
        result = normalize_messages(msgs)
        assert "timestamp" not in result[0]

    def test_strips_finish_reason(self):
        msgs = [{"role": "assistant", "content": "ok", "finish_reason": "stop"}]
        result = normalize_messages(msgs)
        assert "finish_reason" not in result[0]

    def test_strips_logprobs(self):
        msgs = [{"role": "assistant", "content": "ok", "logprobs": {"tokens": []}}]
        result = normalize_messages(msgs)
        assert "logprobs" not in result[0]

    def test_strips_created(self):
        msgs = [{"role": "user", "content": "hello", "created": 1700000000}]
        result = normalize_messages(msgs)
        assert "created" not in result[0]

    def test_strips_multiple_non_standard(self):
        msgs = [{
            "role": "user",
            "content": "hello",
            "timestamp": "x",
            "finish_reason": "stop",
            "logprobs": None,
            "created": 1,
            "unknown_field": "discard",
        }]
        result = normalize_messages(msgs)
        assert set(result[0].keys()) == {"role", "content"}

    def test_preserves_all_standard_fields(self):
        msgs = [{
            "role": "assistant",
            "content": "response",
            "name": "assistant_name",
            "tool_call_id": "call_abc",
        }]
        result = normalize_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "response"
        assert result[0]["name"] == "assistant_name"
        assert result[0]["tool_call_id"] == "call_abc"

    def test_empty_messages_list(self):
        assert normalize_messages([]) == []

    def test_message_with_only_role(self):
        msgs = [{"role": "system"}]
        result = normalize_messages(msgs)
        assert result[0] == {"role": "system"}

    def test_extra_fields_stripped_even_with_unusual_names(self):
        msgs = [{"role": "user", "content": "hi", "x_custom_field": 42, "_internal": True}]
        result = normalize_messages(msgs)
        assert "x_custom_field" not in result[0]
        assert "_internal" not in result[0]


# ---------------------------------------------------------------------------
# Rule 2 — Normalize tool_calls
# ---------------------------------------------------------------------------

class TestNormalizeToolCalls:
    def _make_tool_call(self, id: str, extra: bool = False) -> dict:
        tc: dict = {
            "id": id,
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
        if extra:
            tc["extra_field"] = "should be removed"
            tc["function"]["extra"] = "also removed"
        return tc

    def test_strips_extra_fields_from_tool_call(self):
        msgs = [{
            "role": "assistant",
            "tool_calls": [self._make_tool_call("c1", extra=True)],
        }]
        result = normalize_messages(msgs)
        tc = result[0]["tool_calls"][0]
        assert set(tc.keys()) == {"id", "type", "function"}

    def test_strips_extra_fields_from_function(self):
        msgs = [{
            "role": "assistant",
            "tool_calls": [self._make_tool_call("c1", extra=True)],
        }]
        result = normalize_messages(msgs)
        fn = result[0]["tool_calls"][0]["function"]
        assert set(fn.keys()) == {"name", "arguments"}

    def test_tool_calls_sorted_by_id(self):
        msgs = [{
            "role": "assistant",
            "tool_calls": [
                self._make_tool_call("c3"),
                self._make_tool_call("c1"),
                self._make_tool_call("c2"),
            ],
        }]
        result = normalize_messages(msgs)
        ids = [tc["id"] for tc in result[0]["tool_calls"]]
        assert ids == ["c1", "c2", "c3"]

    def test_tool_calls_already_sorted_unchanged(self):
        msgs = [{
            "role": "assistant",
            "tool_calls": [
                self._make_tool_call("a"),
                self._make_tool_call("b"),
            ],
        }]
        result = normalize_messages(msgs)
        ids = [tc["id"] for tc in result[0]["tool_calls"]]
        assert ids == ["a", "b"]

    def test_empty_tool_calls_list(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": []}]
        result = normalize_messages(msgs)
        assert result[0]["tool_calls"] == []

    def test_tool_calls_none_not_included(self):
        msgs = [{"role": "user", "content": "hi", "tool_calls": None}]
        # tool_calls=None: key should be present with None (included as standard field)
        result = normalize_messages(msgs)
        # When tool_calls is None we skip normalization; key still present
        assert result[0].get("tool_calls") is None

    def test_single_tool_call_preserved(self):
        msgs = [{
            "role": "assistant",
            "tool_calls": [self._make_tool_call("only")],
        }]
        result = normalize_messages(msgs)
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "only"

    def test_tool_calls_with_identical_ids_stable(self):
        tc = self._make_tool_call("same")
        msgs = [{"role": "assistant", "tool_calls": [tc, tc.copy()]}]
        result = normalize_messages(msgs)
        assert len(result[0]["tool_calls"]) == 2


# ---------------------------------------------------------------------------
# Rule 3 — Normalize whitespace in content
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_strips_trailing_whitespace(self):
        msgs = [{"role": "user", "content": "hello   "}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "hello"

    def test_strips_trailing_whitespace_multiline(self):
        msgs = [{"role": "user", "content": "line1   \nline2  \nline3"}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "line1\nline2\nline3"

    def test_normalizes_multiple_spaces(self):
        msgs = [{"role": "user", "content": "a  b   c    d"}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "a b c d"

    def test_normalizes_crlf_to_lf(self):
        msgs = [{"role": "user", "content": "line1\r\nline2\r\nline3"}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "line1\nline2\nline3"

    def test_normalizes_cr_to_lf(self):
        msgs = [{"role": "user", "content": "line1\rline2"}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "line1\nline2"

    def test_preserves_leading_whitespace(self):
        msgs = [{"role": "user", "content": "    indented code"}]
        result = normalize_messages(msgs)
        assert result[0]["content"].startswith("    ")

    def test_preserves_leading_whitespace_multiline(self):
        code = "def foo():\n    return 1\n    "
        msgs = [{"role": "user", "content": code}]
        result = normalize_messages(msgs)
        # Trailing spaces on the last line are stripped, but the newline itself remains
        assert result[0]["content"] == "def foo():\n    return 1\n"

    def test_tabs_between_words_not_collapsed(self):
        # Only multiple spaces/tabs are collapsed; single tabs are preserved
        msgs = [{"role": "user", "content": "a\tb"}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "a\tb"

    def test_empty_content_string_unchanged(self):
        msgs = [{"role": "user", "content": ""}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == ""

    def test_whitespace_only_content(self):
        msgs = [{"role": "user", "content": "   "}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == ""

    def test_same_content_produces_identical_output(self):
        content = "Hello   world\r\n  how are you?  "
        msgs1 = [{"role": "user", "content": content}]
        msgs2 = [{"role": "user", "content": content}]
        assert normalize_messages(msgs1) == normalize_messages(msgs2)

    def test_different_formatting_same_logical_content(self):
        a = [{"role": "user", "content": "Hello   world\r\n"}]
        b = [{"role": "user", "content": "Hello world\n"}]
        assert normalize_messages(a) == normalize_messages(b)


# ---------------------------------------------------------------------------
# Rule 4 — Normalize content types
# ---------------------------------------------------------------------------

class TestNormalizeContentTypes:
    def test_null_content_becomes_empty_string(self):
        msgs = [{"role": "assistant", "content": None}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == ""

    def test_list_content_sorted_by_type(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "http://img"}},
                {"type": "text", "text": "describe this"},
            ],
        }]
        result = normalize_messages(msgs)
        types = [p["type"] for p in result[0]["content"]]
        assert types == sorted(types)

    def test_list_content_text_normalized(self):
        msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": "hello   world  "}],
        }]
        result = normalize_messages(msgs)
        assert result[0]["content"][0]["text"] == "hello world"

    def test_list_content_non_text_parts_preserved(self):
        img = {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}
        msgs = [{"role": "user", "content": [img]}]
        result = normalize_messages(msgs)
        assert result[0]["content"][0] == img

    def test_empty_list_content(self):
        msgs = [{"role": "user", "content": []}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == []

    def test_missing_content_key_not_added(self):
        msgs = [{"role": "system"}]
        result = normalize_messages(msgs)
        assert "content" not in result[0]


# ---------------------------------------------------------------------------
# Rule 5 — Field ordering (idempotency)
# ---------------------------------------------------------------------------

class TestFieldOrdering:
    def test_normalize_is_idempotent(self):
        msgs = [{
            "role": "user",
            "content": "hello   world\r\n",
            "timestamp": "x",
        }]
        once = normalize_messages(msgs)
        twice = normalize_messages(once)
        assert once == twice

    def test_multiple_messages_all_normalized(self):
        msgs = [
            {"role": "system", "content": "You are helpful.", "extra": "drop"},
            {"role": "user", "content": "Hello  world"},
            {"role": "assistant", "content": "Hi  there  "},
        ]
        result = normalize_messages(msgs)
        assert "extra" not in result[0]
        assert result[1]["content"] == "Hello world"
        assert result[2]["content"] == "Hi there"


# ---------------------------------------------------------------------------
# Rule 6 — Normalize reasoning fields
# ---------------------------------------------------------------------------

class TestNormalizeReasoningFields:
    def test_reasoning_renamed_to_reasoning_content(self):
        msgs = [{"role": "assistant", "content": "ans", "reasoning": "I thought..."}]
        result = normalize_messages(msgs)
        assert "reasoning_content" in result[0]
        assert result[0]["reasoning_content"] == "I thought..."
        assert "reasoning" not in result[0]

    def test_reasoning_content_preserved_when_alone(self):
        msgs = [{"role": "assistant", "content": "ans", "reasoning_content": "I thought..."}]
        result = normalize_messages(msgs)
        assert result[0]["reasoning_content"] == "I thought..."

    def test_both_reasoning_fields_uses_reasoning_content(self):
        msgs = [{
            "role": "assistant",
            "content": "ans",
            "reasoning": "old value",
            "reasoning_content": "preferred value",
        }]
        result = normalize_messages(msgs)
        assert result[0]["reasoning_content"] == "preferred value"

    def test_no_reasoning_fields_unaffected(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = normalize_messages(msgs)
        assert "reasoning" not in result[0]
        assert "reasoning_content" not in result[0]


# ---------------------------------------------------------------------------
# Edge cases — complex / combined scenarios
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unicode_content_preserved(self):
        content = "こんにちは   世界"
        msgs = [{"role": "user", "content": content}]
        result = normalize_messages(msgs)
        assert "こんにちは" in result[0]["content"]
        assert "世界" in result[0]["content"]

    def test_very_long_content(self):
        long_content = "word " * 10000
        msgs = [{"role": "user", "content": long_content}]
        result = normalize_messages(msgs)
        # Multiple spaces collapsed; trailing space removed
        assert "  " not in result[0]["content"]

    def test_message_with_newlines_only(self):
        msgs = [{"role": "user", "content": "\n\n\n"}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "\n\n\n"

    def test_code_block_leading_whitespace_preserved(self):
        code = "```python\n    x = 1\n    y = 2\n```"
        msgs = [{"role": "user", "content": code}]
        result = normalize_messages(msgs)
        assert "    x = 1" in result[0]["content"]

    def test_system_message_stripped_properly(self):
        msgs = [{
            "role": "system",
            "content": "  You are helpful.  ",
            "timestamp": 123,
            "created": 456,
        }]
        result = normalize_messages(msgs)
        # Leading whitespace preserved per spec; only trailing stripped
        assert result[0]["content"] == "  You are helpful."
        assert "timestamp" not in result[0]
        assert "created" not in result[0]

    def test_tool_message_preserved(self):
        msgs = [{
            "role": "tool",
            "content": '{"result": 42}',
            "tool_call_id": "call_xyz",
        }]
        result = normalize_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_xyz"
        assert result[0]["content"] == '{"result": 42}'

    def test_list_content_multiple_text_parts(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "part1   "},
                {"type": "text", "text": "part2  "},
            ],
        }]
        result = normalize_messages(msgs)
        texts = [p["text"] for p in result[0]["content"]]
        assert texts == ["part1", "part2"]

    def test_normalize_string_mixed_line_endings(self):
        s = "a\r\nb\rc\nd  "
        result = _normalize_string(s)
        assert "\r" not in result
        assert result == "a\nb\nc\nd"

    def test_normalize_preserves_single_space(self):
        msgs = [{"role": "user", "content": "a b c"}]
        result = normalize_messages(msgs)
        assert result[0]["content"] == "a b c"

    def test_all_roles_normalized(self):
        for role in ("system", "user", "assistant", "tool", "function"):
            msgs = [{"role": role, "content": "test  content  ", "extra": "drop"}]
            result = normalize_messages(msgs)
            assert result[0]["role"] == role
            assert result[0]["content"] == "test content"
            assert "extra" not in result[0]

    def test_name_field_not_modified(self):
        msgs = [{"role": "user", "content": "hi", "name": "alice  "}]
        result = normalize_messages(msgs)
        # name field is NOT whitespace-normalized (only content is)
        assert result[0]["name"] == "alice  "

    def test_large_number_of_messages(self):
        msgs = [{"role": "user", "content": f"message {i}  ", "extra": i} for i in range(1000)]
        result = normalize_messages(msgs)
        assert len(result) == 1000
        assert all("extra" not in m for m in result)
        assert all(not m["content"].endswith(" ") for m in result)

    def test_tool_calls_function_arguments_preserved_exactly(self):
        args = '{"key": "value with   spaces", "num": 42}'
        msgs = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "fn", "arguments": args},
            }],
        }]
        result = normalize_messages(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == args
