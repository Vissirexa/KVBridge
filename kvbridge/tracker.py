from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .models import RequestMetric, SessionMetrics
from .session import estimate_cache_status


class TTFTTracker:
    def __init__(
        self,
        data_dir: str = "./data",
        hit_threshold: float = 0.3,
        miss_threshold: float = 0.7,
        calibration_requests: int = 5,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._data_dir / "metrics.json"

        self._hit_threshold = hit_threshold
        self._miss_threshold = miss_threshold
        self._calibration_requests = calibration_requests

        self._lock = Lock()
        self._metrics: list[RequestMetric] = []
        self._prefill_samples: list[float] = []  # tokens_per_ms samples
        self._prefill_tps: float | None = None   # calibrated tokens/ms

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, metric: RequestMetric) -> None:
        with self._lock:
            self._metrics.append(metric)
            self._update_calibration(metric)
            self._persist()

    def classify_cache_status(
        self, ttft_ms: float, input_tokens: int
    ) -> str:
        if self._prefill_tps is None or self._prefill_tps == 0:
            return "unknown"

        expected_miss_ms = input_tokens / self._prefill_tps
        if expected_miss_ms == 0:
            return "unknown"

        ratio = ttft_ms / expected_miss_ms
        if ratio < self._hit_threshold:
            return "hit"
        if ratio > self._miss_threshold:
            return "miss"
        return "uncertain"

    def get_session_metrics(self, session_id: str) -> SessionMetrics | None:
        with self._lock:
            return self._build_session_metrics(session_id)

    def get_all_sessions(self) -> list[SessionMetrics]:
        with self._lock:
            session_ids = {m.session_id for m in self._metrics}
            return [
                sm
                for sid in session_ids
                if (sm := self._build_session_metrics(sid)) is not None
            ]

    def get_all_metrics(self) -> list[RequestMetric]:
        with self._lock:
            return list(self._metrics)

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()
            self._prefill_samples.clear()
            self._prefill_tps = None
            self._persist()

    @property
    def prefill_tps(self) -> float | None:
        return self._prefill_tps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_calibration(self, metric: RequestMetric) -> None:
        if (
            metric.ttft_ms is not None
            and metric.ttft_ms > 0
            and metric.input_tokens > 0
            and metric.cache_status in ("miss", "unknown")
        ):
            sample = metric.input_tokens / metric.ttft_ms
            self._prefill_samples.append(sample)
            # Keep a rolling window of the last N calibration samples
            window = max(self._calibration_requests, 10)
            if len(self._prefill_samples) > window:
                self._prefill_samples = self._prefill_samples[-window:]
            self._prefill_tps = sum(self._prefill_samples) / len(
                self._prefill_samples
            )

    def _build_session_metrics(self, session_id: str) -> SessionMetrics | None:
        session_reqs = [m for m in self._metrics if m.session_id == session_id]
        if not session_reqs:
            return None

        hits = [m for m in session_reqs if m.cache_status == "hit"]
        misses = [m for m in session_reqs if m.cache_status == "miss"]
        total = len(session_reqs)

        hit_rate = len(hits) / total if total else 0.0
        avg_hit = (
            sum(m.ttft_ms for m in hits if m.ttft_ms is not None) / len(hits)
            if hits
            else 0.0
        )
        avg_miss = (
            sum(m.ttft_ms for m in misses if m.ttft_ms is not None) / len(misses)
            if misses
            else 0.0
        )
        last_time = max(m.timestamp for m in session_reqs)
        cache_status = estimate_cache_status(last_time)

        return SessionMetrics(
            session_id=session_id,
            total_requests=total,
            cache_hits=len(hits),
            cache_misses=len(misses),
            hit_rate=hit_rate,
            avg_ttft_hit_ms=avg_hit,
            avg_ttft_miss_ms=avg_miss,
            last_request_time=last_time,
            estimated_cache_status=cache_status,
        )

    def _persist(self) -> None:
        records = []
        for m in self._metrics:
            records.append(
                {
                    "session_id": m.session_id,
                    "timestamp": m.timestamp.isoformat(),
                    "ttft_ms": m.ttft_ms,
                    "cache_status": m.cache_status,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_latency_ms": m.total_latency_ms,
                    "message_count": m.message_count,
                    "prefix_length": m.prefix_length,
                    "model": m.model,
                }
            )
        self._metrics_path.write_text(json.dumps(records, indent=2))

    def _load(self) -> None:
        if not self._metrics_path.exists():
            return
        try:
            raw: list[dict[str, Any]] = json.loads(self._metrics_path.read_text())
            for r in raw:
                self._metrics.append(
                    RequestMetric(
                        session_id=r["session_id"],
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        ttft_ms=r.get("ttft_ms"),
                        cache_status=r.get("cache_status", "unknown"),
                        input_tokens=r.get("input_tokens", 0),
                        output_tokens=r.get("output_tokens", 0),
                        total_latency_ms=r.get("total_latency_ms", 0.0),
                        message_count=r.get("message_count", 0),
                        prefix_length=r.get("prefix_length", 0),
                        model=r.get("model", ""),
                    )
                )
        except Exception:
            self._metrics = []
