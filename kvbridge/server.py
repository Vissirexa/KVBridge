from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .forwarder import forward_request, forward_streaming, passthrough_get, passthrough_post
from .logger import get_logger
from .models import ChatCompletionRequest, RequestMetric
from .normalizer import normalize_messages
from .session import identify_session
from .tracker import TTFTTracker

_SSE_DATA_PREFIX = b"data: "
_SSE_DONE = b"data: [DONE]"


def _iter_sse_objects(chunk: bytes) -> Any:
    for line in chunk.split(b"\n"):
        line = line.strip()
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        data = line[len(_SSE_DATA_PREFIX):]
        if data == b"[DONE]":
            continue
        try:
            yield json.loads(data)
        except (json.JSONDecodeError, AttributeError):
            continue


def _has_content(chunk: bytes) -> bool:
    for obj in _iter_sse_objects(chunk):
        for choice in obj.get("choices", []):
            if choice.get("delta", {}).get("content"):
                return True
    return False


def _extract_usage(chunk: bytes) -> dict[str, int] | None:
    """Return the last usage block found in an SSE chunk, if any.

    OpenAI-compatible servers emit a final chunk carrying ``usage`` when
    ``stream_options.include_usage`` is set.
    """
    found: dict[str, int] | None = None
    for obj in _iter_sse_objects(chunk):
        usage = obj.get("usage")
        if isinstance(usage, dict):
            found = usage
    return found


def create_app(
    upstream_url: str = "http://localhost:1234",
    upstream_timeout: float = 300.0,
    tracker: TTFTTracker | None = None,
    normalization_enabled: bool = True,
) -> FastAPI:
    app = FastAPI(title="KVBridge", version="0.1.0")
    logger = get_logger()
    _tracker = tracker or TTFTTracker()

    # ------------------------------------------------------------------
    # Primary endpoint
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        req = ChatCompletionRequest.model_validate(body)

        # Normalize messages
        raw_messages = [m.model_dump(exclude_none=True) for m in req.messages]
        if normalization_enabled:
            normalized_msgs = normalize_messages(raw_messages)
        else:
            normalized_msgs = raw_messages

        session_id = identify_session(normalized_msgs)
        prefix_length = len(normalized_msgs[0].get("content", "")) if normalized_msgs else 0

        # Build forwarding payload from original body, replacing messages
        payload = dict(body)
        payload["messages"] = normalized_msgs

        logger.debug("Forwarding request session=%s stream=%s", session_id, req.stream)

        start_time = time.monotonic()

        if req.stream:
            ttft_holder: dict[str, float | None] = {"ttft": None}

            # Ask the upstream to emit a final usage chunk so we can record
            # token counts (required for cache classification & calibration).
            stream_payload = dict(payload)
            stream_options = dict(stream_payload.get("stream_options") or {})
            stream_options["include_usage"] = True
            stream_payload["stream_options"] = stream_options

            async def instrumented_stream() -> Any:
                total_start = time.monotonic()
                input_tokens = 0
                output_tokens = 0
                model = req.model

                try:
                    async for chunk in forward_streaming(stream_payload, upstream_url, upstream_timeout):
                        if ttft_holder["ttft"] is None and _has_content(chunk):
                            ttft_holder["ttft"] = (time.monotonic() - start_time) * 1000
                        usage = _extract_usage(chunk)
                        if usage is not None:
                            input_tokens = usage.get("prompt_tokens", input_tokens)
                            output_tokens = usage.get("completion_tokens", output_tokens)
                        yield chunk
                except httpx.HTTPError as exc:
                    logger.error("Upstream error: %s", exc)
                    yield b"data: [DONE]\n\n"
                    return

                ttft_ms = ttft_holder["ttft"]
                total_ms = (time.monotonic() - total_start) * 1000
                cache_status = _tracker.classify_cache_status(ttft_ms or 0, input_tokens)

                metric = RequestMetric(
                    session_id=session_id,
                    timestamp=datetime.now(timezone.utc),
                    ttft_ms=ttft_ms,
                    cache_status=cache_status,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_latency_ms=total_ms,
                    message_count=len(normalized_msgs),
                    prefix_length=prefix_length,
                    model=model,
                )
                _tracker.record(metric)
                logger.info(
                    "session=%s ttft=%.1fms cache=%s",
                    session_id,
                    ttft_ms or 0,
                    cache_status,
                )

            return StreamingResponse(
                instrumented_stream(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )

        else:
            try:
                response_data = await forward_request(payload, upstream_url, upstream_timeout)
            except httpx.HTTPStatusError as exc:
                raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

            total_ms = (time.monotonic() - start_time) * 1000
            ttft_ms = total_ms  # Non-streaming: TTFT ≈ total latency

            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cache_status = _tracker.classify_cache_status(ttft_ms, input_tokens)

            metric = RequestMetric(
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                ttft_ms=ttft_ms,
                cache_status=cache_status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_latency_ms=total_ms,
                message_count=len(normalized_msgs),
                prefix_length=prefix_length,
                model=req.model,
            )
            _tracker.record(metric)
            logger.info(
                "session=%s ttft=%.1fms cache=%s",
                session_id,
                ttft_ms,
                cache_status,
            )
            return JSONResponse(content=response_data)

    # ------------------------------------------------------------------
    # Passthrough endpoints
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models() -> Any:
        try:
            return await passthrough_get("/v1/models", upstream_url)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        try:
            return await passthrough_post("/v1/completions", body, upstream_url)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    # ------------------------------------------------------------------
    # KVBridge-specific endpoints
    # ------------------------------------------------------------------

    @app.get("/kvbridge/status")
    async def status() -> Any:
        sessions = _tracker.get_all_sessions()
        total_requests = sum(s.total_requests for s in sessions)
        total_hits = sum(s.cache_hits for s in sessions)
        overall_hit_rate = total_hits / total_requests if total_requests else 0.0
        return {
            "status": "running",
            "upstream": upstream_url,
            "total_sessions": len(sessions),
            "total_requests": total_requests,
            "overall_hit_rate": overall_hit_rate,
            "prefill_tps": _tracker.prefill_tps,
        }

    @app.get("/kvbridge/sessions")
    async def list_sessions() -> Any:
        sessions = _tracker.get_all_sessions()
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "total_requests": s.total_requests,
                    "cache_hits": s.cache_hits,
                    "cache_misses": s.cache_misses,
                    "hit_rate": s.hit_rate,
                    "avg_ttft_hit_ms": s.avg_ttft_hit_ms,
                    "avg_ttft_miss_ms": s.avg_ttft_miss_ms,
                    "last_request_time": s.last_request_time.isoformat(),
                    "estimated_cache_status": s.estimated_cache_status,
                }
                for s in sessions
            ]
        }

    @app.get("/kvbridge/metrics")
    async def metrics() -> Any:
        all_metrics = _tracker.get_all_metrics()
        return {
            "metrics": [
                {
                    "session_id": m.session_id,
                    "timestamp": m.timestamp.isoformat(),
                    "ttft_ms": m.ttft_ms,
                    "cache_status": m.cache_status,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_latency_ms": m.total_latency_ms,
                    "message_count": m.message_count,
                    "model": m.model,
                }
                for m in all_metrics
            ]
        }

    @app.post("/kvbridge/reset")
    async def reset() -> Any:
        _tracker.reset()
        logger.info("Tracking data reset")
        return {"status": "reset"}

    return app
