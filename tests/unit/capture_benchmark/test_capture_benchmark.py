"""Meaningful oracle and schedule invariants; executed only on remote CPU."""

import json

from benchmarks.capture.oracle import quantiles, verify
from benchmarks.capture.schedule import (
    latency_sample,
    offset_ns,
    payload,
    plan,
    producer_for,
)
from benchmarks.capture.worker import RECEIPT


def test_open_schedule_not_rescheduled_after_backpressure():
    c = {"duration_s": 1, "rate": 10_000, "seed": 100}
    assert offset_ns(0, c) == 0
    assert offset_ns(9999, c) == 999_900_000
    c.update(arrival="burst", active_ms=1, period_ms=100)
    assert offset_ns(999, c) < 1_000_000
    assert offset_ns(1000, c) == 100_000_000
    assert offset_ns(3000, c) == 300_000_000
    assert offset_ns(6000, c) == 600_000_000
    for i in range(1, 10_000):
        assert offset_ns(i, c) >= offset_ns(i - 1, c)


def test_phased_load_has_exact_duration_and_original_schedule():
    c = {"duration_s": 3, "rate": 1000, "phases": [{"duration_s": 1, "rate": 10}, {"duration_s": 2, "rate": 100}]}
    schedule = plan(c)
    assert len(schedule) == 210
    assert schedule[10] == 1_000_000_000
    assert schedule[-1] == 2_990_000_000


def test_random_arrivals_are_seeded_and_inside_input_window():
    for arrival in ("poisson", "onoff"):
        c = {"duration_s": 1, "rate": 1000, "seed": 100, "arrival": arrival}
        a = plan(c)
        assert a == plan(c)
        assert all(0 <= t < 1_000_000_000 for t in a)
        assert a == sorted(a)


def test_tail_quantile_is_not_claimed_from_small_samples():
    assert quantiles(list(range(100)))["p99_ns"] is None
    assert quantiles(list(range(10_000)))["p99_ns"] == 9900


def test_skewed_source_assigns_hot_90_percent():
    c = {"producers": 8, "skew": True}
    assignments = [producer_for(i, c) for i in range(1000)]
    assert assignments.count(0) == 900
    assert set(assignments) == set(range(8))


def test_sampler_does_not_alias_periodic_snapshot_identity():
    selected = [i for i in range(999, 1_000_000, 1000) if latency_sample(i, 100)]
    assert 5 < len(selected) < 40


def fixture_capture(tmp_path):
    c = {"id": "test", "workload": "W01", "run_id": "run_test", "seed": 100,
         "duration_s": 1, "rate": 2, "producers": 1, "body_bytes": 64}
    raw = {"case": c, "adapter": "otel", "scheduled": 2, "track": "PYTHON",
           "H_ns": 1_000_000_000, "t0_ns": 100, "endpoint_end_ns": 1_000_000_100,
           "acknowledgement": "unavailable_void_api", "errors": [],
           "terminal": {"endpoint": "D2-close", "close_end_ns": 1_000_000_100},
           "cpu_seconds": 0.001, "maxrss_KiB": 100, "startup_ns": 1,
           "minor_faults": 0, "major_faults": 0, "voluntary_switches": 0, "involuntary_switches": 0}
    (tmp_path / "worker.json").write_text(json.dumps(raw))
    (tmp_path / "receipts-0.bin").write_bytes(RECEIPT.pack(0, 0, 100, 0, 1) + RECEIPT.pack(1, 0, 500_000_100, 0, 1))
    rows = [{"run_id": "run_test", "source_seq": i, "producer_id": 0,
             "source_timestamp": 100 + i * 500_000_000, "body": payload(i, c)} for i in range(2)]
    return rows


def test_oracle_does_not_count_duplicate_or_corrupt_as_goodput(tmp_path):
    rows = fixture_capture(tmp_path)
    rows.append(dict(rows[0]))
    rows[1]["body"] = "corrupted"
    (tmp_path / "otel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = verify(tmp_path, tmp_path)
    assert result["status"] == "correctness_fail"
    assert result["valid_unique"] == 1
    assert result["duplicate_records"] == 1
    assert result["semantic_corruption"] == 1
    assert result["accepted"] is None
    assert result["scheduled_delivery_ratio"] == 0.5


def test_success_keeps_absent_completion_timing_unavailable(tmp_path):
    rows = fixture_capture(tmp_path)
    (tmp_path / "otel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = verify(tmp_path, tmp_path)
    assert result["status"] == "complete"
    assert result["cohort_completion_rate"] == 2
    assert "per_event_ready_durable_latency_and_deadline_goodput" in result["unavailable"]


def test_structured_oracle_preserves_types_and_allows_key_reordering(tmp_path):
    rows = fixture_capture(tmp_path)
    raw = json.loads((tmp_path / "worker.json").read_text())
    raw["case"].update(structured=True, depth=1, list_entries=2)
    (tmp_path / "worker.json").write_text(json.dumps(raw))
    for i, row in enumerate(rows):
        row["body"] = json.loads(payload(i, raw["case"]))
    rows[0]["body"]["metadata"]["nested"][0] = False  # False == 0 in Python, but types differ.
    (tmp_path / "otel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = verify(tmp_path, tmp_path)
    assert result["semantic_corruption"] == 1
    assert result["valid_unique"] == 1
