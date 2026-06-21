"""Tests for kvbridge.tracker — TTFT tracking and cache classification."""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from kvbridge.models import RequestMetric
from kvbridge.tracker import TTFTTracker


def _make_metric(
    session_id: str = "sess1",
    ttft_ms: float = 100.0,
    input_tokens: int = 200,
    output_tokens: int = 50,
    cache_status: str = "miss",
    total_latency_ms: float = 500.0,
    message_count: int = 3,
    prefix_length: int = 50,
    model: str = "test-model",
) -> RequestMetric:
    return RequestMetric(
        session_id=session_id,
        timestamp=datetime.utcnow(),
        ttft_ms=ttft_ms,
        cache_status=cache_status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_latency_ms=total_latency_ms,
        message_count=message_count,
        prefix_length=prefix_length,
        model=model,
    )


@pytest.fixture
def tmp_tracker(tmp_path: Path) -> TTFTTracker:
    return TTFTTracker(
        data_dir=str(tmp_path),
        hit_threshold=0.3,
        miss_threshold=0.7,
        calibration_requests=5,
    )


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestCacheClassification:
    def test_unknown_when_uncalibrated(self, tmp_tracker):
        assert tmp_tracker.classify_cache_status(50.0, 200) == "unknown"

    def test_classified_as_hit(self, tmp_tracker):
        # Calibrate: inject a miss metric so prefill_tps is known
        # 200 tokens in 100ms → 2 tok/ms
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        # TTFT = 5ms for 200 tokens → ratio = 5/100 = 0.05 < 0.3 → hit
        assert tmp_tracker.classify_cache_status(5.0, 200) == "hit"

    def test_classified_as_miss(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        # TTFT = 100ms for 200 tokens → ratio = 1.0 > 0.7 → miss
        assert tmp_tracker.classify_cache_status(100.0, 200) == "miss"

    def test_classified_as_uncertain(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        # TTFT = 50ms for 200 tokens → ratio = 0.5 → between 0.3 and 0.7 → uncertain
        assert tmp_tracker.classify_cache_status(50.0, 200) == "uncertain"

    def test_boundary_at_hit_threshold(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        # ratio = 0.3 exactly: NOT < 0.3, so uncertain
        assert tmp_tracker.classify_cache_status(30.0, 200) == "uncertain"

    def test_boundary_at_miss_threshold(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        # ratio = 0.7 exactly: NOT > 0.7, so uncertain
        assert tmp_tracker.classify_cache_status(70.0, 200) == "uncertain"

    def test_zero_input_tokens_returns_unknown(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        assert tmp_tracker.classify_cache_status(50.0, 0) == "unknown"

    def test_very_small_ttft_is_hit(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=1000.0, input_tokens=500, cache_status="miss"))
        # Calibrated at 0.5 tok/ms, expected miss = 500/0.5 = 1000ms
        # actual ttft = 1ms → ratio = 0.001 → hit
        assert tmp_tracker.classify_cache_status(1.0, 500) == "hit"

    def test_custom_thresholds(self, tmp_path):
        tracker = TTFTTracker(
            data_dir=str(tmp_path / "custom"),
            hit_threshold=0.1,
            miss_threshold=0.9,
            calibration_requests=3,
        )
        # Calibrate: 100 tokens in 100ms → 1 tok/ms
        tracker.record(_make_metric(ttft_ms=100.0, input_tokens=100, cache_status="miss"))
        # ratio=0.05 < 0.1 → hit
        assert tracker.classify_cache_status(5.0, 100) == "hit"
        # ratio=0.95 > 0.9 → miss
        assert tracker.classify_cache_status(95.0, 100) == "miss"
        # ratio=0.5 → uncertain (between 0.1 and 0.9)
        assert tracker.classify_cache_status(50.0, 100) == "uncertain"


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_no_calibration_initially(self, tmp_tracker):
        assert tmp_tracker.prefill_tps is None

    def test_calibration_after_miss_metric(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        assert tmp_tracker.prefill_tps is not None
        assert tmp_tracker.prefill_tps == pytest.approx(2.0)  # 200/100

    def test_calibration_not_updated_by_hit(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        tps_before = tmp_tracker.prefill_tps
        tmp_tracker.record(_make_metric(ttft_ms=10.0, input_tokens=200, cache_status="hit"))
        assert tmp_tracker.prefill_tps == tps_before

    def test_rolling_average_calibration(self, tmp_tracker):
        # 3 samples: 1 tok/ms, 2 tok/ms, 3 tok/ms → avg 2 tok/ms
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=100, cache_status="miss"))
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=200, cache_status="miss"))
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=300, cache_status="miss"))
        assert tmp_tracker.prefill_tps == pytest.approx(2.0)

    def test_zero_ttft_not_used_for_calibration(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=0.0, input_tokens=200, cache_status="miss"))
        assert tmp_tracker.prefill_tps is None

    def test_none_ttft_not_used_for_calibration(self, tmp_tracker):
        m = _make_metric()
        m.ttft_ms = None
        tmp_tracker.record(m)
        assert tmp_tracker.prefill_tps is None

    def test_zero_input_tokens_not_used_for_calibration(self, tmp_tracker):
        tmp_tracker.record(_make_metric(ttft_ms=100.0, input_tokens=0, cache_status="miss"))
        assert tmp_tracker.prefill_tps is None


# ---------------------------------------------------------------------------
# Session metrics tests
# ---------------------------------------------------------------------------

class TestSessionMetrics:
    def test_session_metrics_none_for_unknown_session(self, tmp_tracker):
        assert tmp_tracker.get_session_metrics("nonexistent") is None

    def test_session_metrics_basic(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="miss", ttft_ms=200.0))
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="hit", ttft_ms=20.0))
        sm = tmp_tracker.get_session_metrics("s1")
        assert sm is not None
        assert sm.session_id == "s1"
        assert sm.total_requests == 2
        assert sm.cache_hits == 1
        assert sm.cache_misses == 1
        assert sm.hit_rate == pytest.approx(0.5)

    def test_session_avg_ttft_hit(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="hit", ttft_ms=10.0))
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="hit", ttft_ms=30.0))
        sm = tmp_tracker.get_session_metrics("s1")
        assert sm.avg_ttft_hit_ms == pytest.approx(20.0)

    def test_session_avg_ttft_miss(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="miss", ttft_ms=100.0))
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="miss", ttft_ms=200.0))
        sm = tmp_tracker.get_session_metrics("s1")
        assert sm.avg_ttft_miss_ms == pytest.approx(150.0)

    def test_session_no_hits_avg_hit_zero(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="miss"))
        sm = tmp_tracker.get_session_metrics("s1")
        assert sm.avg_ttft_hit_ms == 0.0

    def test_session_no_misses_avg_miss_zero(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="hit"))
        sm = tmp_tracker.get_session_metrics("s1")
        assert sm.avg_ttft_miss_ms == 0.0

    def test_get_all_sessions(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1"))
        tmp_tracker.record(_make_metric(session_id="s2"))
        tmp_tracker.record(_make_metric(session_id="s1"))
        sessions = tmp_tracker.get_all_sessions()
        ids = {s.session_id for s in sessions}
        assert ids == {"s1", "s2"}

    def test_session_metrics_isolated(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="hit", ttft_ms=10.0))
        tmp_tracker.record(_make_metric(session_id="s2", cache_status="miss", ttft_ms=500.0))
        s1 = tmp_tracker.get_session_metrics("s1")
        s2 = tmp_tracker.get_session_metrics("s2")
        assert s1.cache_hits == 1 and s1.cache_misses == 0
        assert s2.cache_hits == 0 and s2.cache_misses == 1

    def test_uncertain_not_counted_as_hit_or_miss(self, tmp_tracker):
        tmp_tracker.record(_make_metric(session_id="s1", cache_status="uncertain"))
        sm = tmp_tracker.get_session_metrics("s1")
        assert sm.cache_hits == 0
        assert sm.cache_misses == 0
        assert sm.total_requests == 1


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_metrics_persisted_to_file(self, tmp_path):
        t = TTFTTracker(data_dir=str(tmp_path))
        t.record(_make_metric(session_id="s1"))
        assert (tmp_path / "metrics.json").exists()

    def test_metrics_loaded_on_restart(self, tmp_path):
        t1 = TTFTTracker(data_dir=str(tmp_path))
        t1.record(_make_metric(session_id="persist_test"))
        # Create new instance — should load from file
        t2 = TTFTTracker(data_dir=str(tmp_path))
        sm = t2.get_session_metrics("persist_test")
        assert sm is not None
        assert sm.total_requests == 1

    def test_multiple_metrics_persisted(self, tmp_path):
        t = TTFTTracker(data_dir=str(tmp_path))
        for i in range(5):
            t.record(_make_metric(session_id=f"s{i}"))
        t2 = TTFTTracker(data_dir=str(tmp_path))
        assert len(t2.get_all_metrics()) == 5

    def test_reset_clears_data(self, tmp_tracker):
        tmp_tracker.record(_make_metric())
        tmp_tracker.reset()
        assert tmp_tracker.get_all_metrics() == []
        assert tmp_tracker.get_all_sessions() == []
        assert tmp_tracker.prefill_tps is None

    def test_reset_persists_empty_state(self, tmp_path):
        t = TTFTTracker(data_dir=str(tmp_path))
        t.record(_make_metric())
        t.reset()
        t2 = TTFTTracker(data_dir=str(tmp_path))
        assert len(t2.get_all_metrics()) == 0

    def test_corrupted_metrics_file_handled(self, tmp_path):
        (tmp_path / "metrics.json").write_text("not valid json{{{")
        t = TTFTTracker(data_dir=str(tmp_path))
        assert t.get_all_metrics() == []

    def test_data_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        t = TTFTTracker(data_dir=str(nested))
        t.record(_make_metric())
        assert nested.exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTrackerEdgeCases:
    def test_none_ttft_metric_recorded(self, tmp_tracker):
        m = _make_metric()
        m.ttft_ms = None
        tmp_tracker.record(m)
        metrics = tmp_tracker.get_all_metrics()
        assert len(metrics) == 1
        assert metrics[0].ttft_ms is None

    def test_very_high_input_tokens(self, tmp_tracker):
        # Should not crash or produce wrong classification
        tmp_tracker.record(_make_metric(ttft_ms=1000.0, input_tokens=100000, cache_status="miss"))
        result = tmp_tracker.classify_cache_status(1.0, 100000)
        assert result in ("hit", "miss", "uncertain", "unknown")

    def test_many_sessions(self, tmp_tracker):
        for i in range(100):
            tmp_tracker.record(_make_metric(session_id=f"session_{i}"))
        sessions = tmp_tracker.get_all_sessions()
        assert len(sessions) == 100

    def test_get_all_metrics_returns_all(self, tmp_tracker):
        for i in range(20):
            tmp_tracker.record(_make_metric(session_id="s1" if i % 2 == 0 else "s2"))
        assert len(tmp_tracker.get_all_metrics()) == 20
