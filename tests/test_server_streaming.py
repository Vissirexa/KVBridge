"""Tests for the streaming path of the chat-completions proxy.

Regression coverage for token tracking on streaming requests: the upstream
emits a final ``usage`` chunk (requested via ``stream_options.include_usage``)
and those token counts must reach the recorded metric so cache classification
and calibration work for streaming traffic.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import kvbridge.server as server
from kvbridge.server import _extract_usage, _has_content, create_app
from kvbridge.tracker import TTFTTracker


def test_has_content_detects_delta():
    chunk = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    assert _has_content(chunk) is True


def test_has_content_false_without_delta():
    chunk = b'data: {"choices":[],"usage":{"prompt_tokens":5}}\n\n'
    assert _has_content(chunk) is False


def test_extract_usage_returns_last_usage_block():
    chunk = (
        b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":200,"completion_tokens":50}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert _extract_usage(chunk) == {"prompt_tokens": 200, "completion_tokens": 50}


def test_extract_usage_none_when_absent():
    chunk = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
    assert _extract_usage(chunk) is None


def test_streaming_records_token_counts(monkeypatch, tmp_path):
    """The recorded metric should carry token counts parsed from the stream."""
    captured: dict = {}

    async def fake_forward_streaming(payload, upstream_url, timeout):
        captured["payload"] = payload
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        yield (
            b'data: {"choices":[],"usage":'
            b'{"prompt_tokens":200,"completion_tokens":50}}\n\n'
        )
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(server, "forward_streaming", fake_forward_streaming)

    tracker = TTFTTracker(data_dir=str(tmp_path))
    app = create_app(tracker=tracker)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    # Consume the body so the generator finishes and records the metric.
    body = resp.content
    assert b"[DONE]" in body

    # Proxy asked upstream to include usage in the stream.
    assert captured["payload"]["stream_options"]["include_usage"] is True

    metrics = tracker.get_all_metrics()
    assert len(metrics) == 1
    assert metrics[0].input_tokens == 200
    assert metrics[0].output_tokens == 50
