"""Unit tests for the E02-06 cold/warm cost-separation logic (pure Python).

Every expected value is hand-computed from the event timeline and hardcoded,
never produced by the function under test. The anti-examples the protocol cares
about each get an explicit test: a duplicate event (silently overwritten
timestamp), an out-of-order timeline, a sub-span sum masquerading as the total,
a failed reset that the state check must see, and a "steady" verdict that must
not be reached while the window is still drifting.
"""

from __future__ import annotations

import pytest

from hqsb.benchmark.cold_warm import (
    EventTimeline,
    compilation_evidence,
    derive_windows,
    kv_reset_verdict,
    process_start_monotonic_ns,
    read_meminfo_fields,
    read_process_start_ticks,
    read_proc_io,
    read_uptime_seconds,
    stability_reached,
    startup_attribution,
    summarize_series,
    validate_event_ordering,
)

# ───────────────────────────── EventTimeline ─────────────────────────────


def test_event_timeline_records_and_spans():
    timeline = EventTimeline()
    timeline.record("a", ns=1_000_000_000)
    timeline.record("b", ns=1_250_000_000)
    assert timeline.span_ms("a", "b") == pytest.approx(250.0)
    assert timeline.names == ["a", "b"]
    assert timeline.as_dict()["events"] == {"a": 1_000_000_000, "b": 1_250_000_000}


def test_event_timeline_rejects_duplicate_name():
    timeline = EventTimeline()
    timeline.record("model_ready", ns=10)
    with pytest.raises(ValueError):
        timeline.record("model_ready", ns=20)


def test_span_missing_event_returns_none():
    timeline = EventTimeline()
    timeline.record("a", ns=1)
    assert timeline.span_ms("a", "b") is None
    assert timeline.get("b") is None


# ───────────────────────── process-clock helpers ─────────────────────────


def test_process_start_monotonic_ns_hand_computed():
    # now = 10 s after boot; uptime = 5 s; process started at 200 ticks / 100 Hz
    # = 2 s after boot -> the process is 3 s old.
    value = process_start_monotonic_ns(
        now_mono_ns=10_000_000_000,
        uptime_s=5.0,
        starttime_ticks=200,
        clk_tck=100.0,
    )
    assert value == 7_000_000_000


def test_process_start_monotonic_ns_none_when_inputs_missing():
    assert process_start_monotonic_ns(10, None, 200, 100.0) is None
    assert process_start_monotonic_ns(10, 5.0, None, 100.0) is None
    assert process_start_monotonic_ns(10, 5.0, 200, None) is None


def test_read_uptime_seconds(tmp_path):
    path = tmp_path / "uptime"
    path.write_text("12345.67 98765.43\n", encoding="utf-8")
    assert read_uptime_seconds(str(path)) == pytest.approx(12345.67)


def test_read_uptime_seconds_missing_file_returns_none(tmp_path):
    assert read_uptime_seconds(str(tmp_path / "nope")) is None


def test_read_process_start_ticks_handles_spaces_in_comm(tmp_path):
    # 22 fields: pid (comm) state ppid ... starttime; starttime is field 22 and
    # the 20th token after the last ')'.
    tail = [str(i) for i in range(3, 23)]
    tail[19] = "200"
    line = "123 (my proc) " + " ".join(tail) + "\n"
    path = tmp_path / "stat"
    path.write_text(line, encoding="utf-8")
    assert read_process_start_ticks(stat_path=str(path)) == 200


def test_read_process_start_ticks_missing_file_returns_none(tmp_path):
    assert read_process_start_ticks(stat_path=str(tmp_path / "nope")) is None


def test_read_proc_io(tmp_path):
    path = tmp_path / "io"
    path.write_text(
        "rchar: 100\nwchar: 20\nsyscr: 3\nsyscw: 1\n"
        "read_bytes: 40\nwrite_bytes: 5\ncancelled_write_bytes: 0\n",
        encoding="utf-8",
    )
    counters = read_proc_io(str(path))
    assert counters["rchar"] == 100
    assert counters["read_bytes"] == 40


def test_read_proc_io_missing_file_returns_empty(tmp_path):
    assert read_proc_io(str(tmp_path / "nope")) == {}


def test_read_meminfo_fields_converts_kb_to_bytes(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:       1000 kB\nCached:          250 kB\n"
        "MemAvailable:     64 kB\nSwapFree:         10 kB\n",
        encoding="utf-8",
    )
    fields = read_meminfo_fields(
        ("MemTotal", "Cached", "Shmem"), path=str(path)
    )
    assert fields["MemTotal"] == 1000 * 1024
    assert fields["Cached"] == 250 * 1024
    assert "Shmem" not in fields  # absent field is omitted, not reported as 0


# ───────────────────────── cost windows ─────────────────────────

# Timeline (ns) with hand-computed ms spans:
#   interpreter 0.5 s | import 3.0 s | verify 1.0 s | device 0.1 s | load 12 s
_EVENTS = {
    "process_start": 0,
    "imports_begin": 500_000_000,
    "imports_ready": 3_500_000_000,
    "artifact_verify_begin": 3_600_000_000,
    "artifact_verified": 4_600_000_000,
    "device_init_begin": 4_700_000_000,
    "device_init_end": 4_800_000_000,
    "load_begin": 4_900_000_000,
    "model_ready": 16_900_000_000,
    "first_request_begin": 17_000_000_000,
    "first_request_end": 30_000_000_000,
    "steady_request_begin": 200_000_000_000,
    "steady_request_end": 200_013_000_000,
    "cache_reset_begin": 210_000_000_000,
    "cache_reset_end": 210_000_100_000,
}


def test_derive_windows_hand_computed():
    windows = derive_windows(_EVENTS)
    assert windows["interpreter_startup_ms"] == pytest.approx(500.0)
    assert windows["framework_import_ms"] == pytest.approx(3000.0)
    assert windows["pre_import_ms"] == pytest.approx(3500.0)
    assert windows["artifact_verify_ms"] == pytest.approx(1000.0)
    assert windows["device_init_ms"] == pytest.approx(100.0)
    assert windows["load_ms"] == pytest.approx(12000.0)
    assert windows["prep_overhead_ms"] == pytest.approx(100.0)
    assert windows["startup_to_model_ready_ms"] == pytest.approx(16900.0)
    assert windows["first_request_latency_ms"] == pytest.approx(13000.0)
    assert windows["startup_to_first_result_ms"] == pytest.approx(30000.0)
    # 200.013 s - 200.000 s = 13 ms (the span, not the request's 13 s latency)
    assert windows["steady_request_latency_ms"] == pytest.approx(13.0)
    assert windows["reset_latency_ms"] == pytest.approx(0.1)


def test_derive_windows_missing_events_are_none_not_zero():
    windows = derive_windows({"process_start": 0, "model_ready": 1_000_000_000})
    assert windows["startup_to_model_ready_ms"] == pytest.approx(1000.0)
    assert windows["first_request_latency_ms"] is None
    assert windows["reset_latency_ms"] is None


def test_startup_attribution_parts_sum_to_total():
    attribution = startup_attribution(derive_windows(_EVENTS))
    # named = 500 + 3000 + 1000 + 100 + 12000; total = 16900 -> overhead = 300.
    assert attribution["named_parts_total_ms"] == pytest.approx(16600.0)
    assert attribution["total_startup_to_model_ready_ms"] == pytest.approx(16900.0)
    assert attribution["unattributed_overhead_ms"] == pytest.approx(300.0)
    assert attribution["non_overlapping"] is True
    assert attribution["complete"] is True


def test_validate_event_ordering_detects_violation():
    assert validate_event_ordering(_EVENTS) == []
    broken = dict(_EVENTS)
    broken["model_ready"] = 1_000_000_000  # before load_begin
    violations = validate_event_ordering(broken)
    assert violations and "model_ready" in violations[0]


# ───────────────────────── compile evidence ─────────────────────────


def test_compilation_evidence_not_enabled_states_it_explicitly():
    evidence = compilation_evidence(enabled=False)
    assert evidence["enabled"] is False
    assert evidence["backend"] is None
    assert evidence["recompile_events"] == 0
    assert "compile not enabled" in evidence["statement"]


def test_compilation_evidence_enabled_records_backend():
    evidence = compilation_evidence(
        enabled=True, backend="inductor", recompile_events=2
    )
    assert evidence["enabled"] is True
    assert evidence["backend"] == "inductor"
    assert evidence["recompile_events"] == 2


# ───────────────────────── KV reset verdict ─────────────────────────


def _negative(detected: bool = True):
    return {"detected": detected, "kv_filled_before": 128, "kv_filled_after": 256}


def test_kv_reset_verdict_passes_when_independent_and_control_detected():
    verdict = kv_reset_verdict(
        cache_filled_before_reset=159,
        request_b_filled_after_prefill=128,
        expected_filled_after_prefill=128,
        request_b_matches_reference=True,
        negative_control=_negative(True),
    )
    assert verdict["passed"] is True
    assert verdict["reasons"] == []


def test_kv_reset_verdict_detects_failed_reset():
    # B started from A's cache (filled 128 + 128) -> not an empty start.
    verdict = kv_reset_verdict(
        cache_filled_before_reset=159,
        request_b_filled_after_prefill=256,
        expected_filled_after_prefill=128,
        request_b_matches_reference=False,
        negative_control=_negative(True),
    )
    assert verdict["passed"] is False
    assert len(verdict["reasons"]) == 2


def test_kv_reset_verdict_requires_negative_control():
    verdict = kv_reset_verdict(
        cache_filled_before_reset=159,
        request_b_filled_after_prefill=128,
        expected_filled_after_prefill=128,
        request_b_matches_reference=True,
        negative_control=_negative(False),
    )
    assert verdict["passed"] is False
    assert "negative control" in verdict["reasons"][0]


# ───────────────────────── steady-state rule ─────────────────────────


def test_stability_reached_when_window_is_flat():
    result = stability_reached([100.0, 102.0, 101.0], window=3, rel_tol=0.10)
    assert result["reached"] is True
    assert result["median_ms"] == pytest.approx(101.0)
    assert result["spread_ratio"] == pytest.approx(2.0 / 101.0)


def test_stability_not_reached_when_drifting():
    result = stability_reached([100.0, 200.0, 400.0], window=3, rel_tol=0.10)
    assert result["reached"] is False
    assert result["reason"] == "still_drifting"


def test_stability_requires_minimum_samples():
    result = stability_reached([100.0, 101.0], window=3, rel_tol=0.10)
    assert result["reached"] is False
    assert result["reason"] == "insufficient_samples"


# ───────────────────────── series summary ─────────────────────────


def test_summarize_series_hand_computed():
    summary = summarize_series([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p50"] == pytest.approx(2.5)
    # pos = (4-1)*0.95 = 2.85 -> 3*0.15 + 4*0.85 = 3.85
    assert summary["p95"] == pytest.approx(3.85)


def test_summarize_series_empty_is_not_a_fake_zero():
    summary = summarize_series([])
    assert summary["count"] == 0
    assert summary["max"] is None
