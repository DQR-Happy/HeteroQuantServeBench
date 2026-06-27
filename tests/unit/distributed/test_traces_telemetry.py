"""E10-09 interface tests: trace pairing, clocks, attribution and C6/C7 projection."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import telemetry as tm
from hqsb.distributed import traces as tr


def _event(rank: int, seq: int, *, arrival_ns: int, tail_ns: int = 100) -> tr.DistributedTraceEvent:
    return tr.DistributedTraceEvent(
        run_id="r",
        global_rank=rank,
        event_type="collective",
        phase="decode",
        clock_domain="host_monotonic",
        host_start_ns=arrival_ns,
        host_end_ns=arrival_ns + tail_ns,
        mapped_global_time_ns=arrival_ns,
        group_id="tp",
        collective_seq=seq,
        op="all_reduce",
        request_id="req-1",
    )


@pytest.mark.unit
class TestTraceEventsAndClocks:
    def test_negative_duration_is_refused(self):
        with pytest.raises(ConfigError):
            tr.DistributedTraceEvent(
                run_id="r", global_rank=0, event_type="kernel", phase="decode",
                clock_domain="host_monotonic", host_start_ns=10, host_end_ns=5,
            )

    def test_unknown_clock_domain_is_refused(self):
        with pytest.raises(ConfigError):
            tr.DistributedTraceEvent(
                run_id="r", global_rank=0, event_type="kernel", phase="decode",
                clock_domain="wall_clock_guess",
            )

    def test_calibration_uncertainty_limits_microsecond_claims(self):
        calibration = tr.calibrate_offsets(
            host_clock_ns={"n0": 1000, "n1": 1200}, reference_host="n0", uncertainty_ns=50
        )
        assert calibration.host_offsets_ns["n1"] == 200
        assert calibration.can_compare(1000) is True
        assert calibration.can_compare(40) is False

    def test_metric_manifest_requires_a_reason_for_unavailable(self):
        with pytest.raises(ConfigError):
            tr.MetricManifest(
                platform="cuda_nccl", profiler_version="2026.1", fields=("collective",),
                units={"collective": "us"}, sampling_rate="100%", rank_coverage="all",
                clock_domain="device", unavailable={"hccl_transit": ""},
            )
        manifest = tr.MetricManifest(
            platform="cuda_nccl", profiler_version="2026.1", fields=("collective",),
            units={"collective": "us"}, sampling_rate="100%", rank_coverage="all",
            clock_domain="device", unavailable={"nvlink_errors": "not exposed by this driver"},
        )
        assert manifest.unavailable


@pytest.mark.unit
class TestPairingAndSkew:
    def test_pairing_requires_every_rank(self):
        events = [_event(0, 0, arrival_ns=1000), _event(1, 0, arrival_ns=1300, tail_ns=50)]
        paired = tr.pair_collective_events(events, expected_ranks=[0, 1])
        assert paired["complete"] is True
        item = paired["paired"][0]
        assert item.arrival_skew_ns == 300

        incomplete = tr.pair_collective_events([_event(0, 0, arrival_ns=1000)], expected_ranks=[0, 1])
        assert incomplete["complete"] is False
        assert incomplete["incomplete"][0]["missing_ranks"] == [1]

    def test_skew_rows_warn_about_victims(self):
        paired = tr.pair_collective_events(
            [_event(0, 0, arrival_ns=1000), _event(1, 0, arrival_ns=5000)], expected_ranks=[0, 1]
        )["paired"]
        rows = tr.arrival_completion_skew(paired)
        assert rows[0]["arrival_skew_ns"] == 4000
        assert rows[0]["early_rank_wait_ns"][0] == 4000
        assert "root cause" in rows[0]["victim_warning"]

    def test_phase_breakdown_balance(self):
        good = tr.PhaseBreakdown(
            rank=0, host_ms=1.0, compute_ms=5.0, comm_ms=2.0, overlap_ms=1.0, wait_ms=1.0,
            idle_ms=1.0, unknown_ms=0.0, wall_ms=9.0,
        )
        report = tr.phase_breakdown([good])
        assert report["ok"] is True
        bad = tr.PhaseBreakdown(
            rank=1, host_ms=1.0, compute_ms=1.0, comm_ms=1.0, overlap_ms=0.0, wait_ms=0.0,
            idle_ms=0.0, unknown_ms=0.0, wall_ms=10.0,
        )
        assert tr.phase_breakdown([bad])["ok"] is False


@pytest.mark.unit
class TestAttribution:
    def _baseline(self) -> tr.VariabilityBaseline:
        baseline = tr.VariabilityBaseline()
        for _ in range(10):
            baseline.record(rank=0, item="attn_kernel", duration_ms=10.0)
            baseline.record(rank=1, item="attn_kernel", duration_ms=10.0)
        return baseline

    def test_first_deviation_and_classification(self):
        classifier = tr.RootCauseClassifier(self._baseline(), uncertainty_ns=0)
        observations = [
            {"rank": 1, "item": "attn_kernel", "duration_ms": 40.0, "start_ns": 2000},
            {"rank": 0, "item": "attn_kernel", "duration_ms": 12.0, "start_ns": 1000},
        ]
        first = classifier.first_deviation(observations)
        assert first is not None and first["rank"] == 1
        attribution = classifier.classify(first=first, evidence={"COMPUTE_SLOW": True})
        assert attribution.fault_class == "COMPUTE_SLOW"
        assert attribution.evidence_used

    def test_unknown_is_preferred_over_a_guess(self):
        classifier = tr.RootCauseClassifier(self._baseline())
        first = {"rank": 0, "item": "attn_kernel", "duration_ms": 40.0, "start_ns": 1}
        attribution = classifier.classify(first=first, evidence={})
        assert attribution.fault_class == "UNKNOWN"
        assert "misdiagnosis" in attribution.notes

    def test_classifier_scores_report_unknown_rate(self):
        scores = tr.classifier_scores(
            {("COMPUTE_SLOW", "COMPUTE_SLOW"): 2, ("UNKNOWN", "LINK_SLOW"): 1,
             ("LINK_SLOW", "LINK_SLOW"): 1}
        )
        assert scores["accuracy"] == pytest.approx(0.75)
        assert scores["unknown_rate"] == pytest.approx(0.25)

    def test_amplification_metrics(self):
        metrics = tr.amplification_metrics(
            delta_local_ms=2.0, delta_job_ms=8.0, victim_wait_ms=[1.0, 2.0, 0.0],
            propagation_depth=3,
        )
        assert metrics["amplification"] == pytest.approx(4.0)
        assert metrics["fanout"] == 2

    def test_unmapped_ratio_gate(self):
        events = [_event(0, 0, arrival_ns=1000), _event(1, 0, arrival_ns=1000)]
        assert tr.unmapped_ratio(events, threshold=0.5)["conclusive"] is True
        unmapped = tr.DistributedTraceEvent(
            run_id="r", global_rank=0, event_type="collective", phase="decode",
            clock_domain="host_monotonic", unmapped=True,
        )
        report = tr.unmapped_ratio([unmapped] * 10, threshold=0.05)
        assert report["conclusive"] is False


@pytest.mark.unit
class TestInjectionPlans:
    def test_injection_plan_is_marker_delimited(self):
        plan = tr.injection_plan(
            kind="rank_arrival_delay", target_rank=1, strength="2 ms", duration_s=10.0
        )
        assert plan["markers"]["start"] and plan["markers"]["end"]
        assert "network transit" in plan["note"]
        with pytest.raises(ConfigError):
            tr.injection_plan(kind="unknown", target_rank=0, strength="x", duration_s=1.0)

    def test_link_shaping_requires_isolation_and_approval(self):
        assert tr.link_shaping_plan(isolated_network=False)["status"] == "NOT_RUN"
        assert tr.link_shaping_plan(isolated_network=True, approval_reference="")["status"] == "NOT_RUN"
        planned = tr.link_shaping_plan(isolated_network=True, approval_reference="CHG-2")
        assert planned["status"] == "PLANNED"

    def test_runbook_and_public_summary(self):
        runbook = tr.straggler_runbook_lines()
        assert runbook[0]["symptom"]
        rows = tr.public_summary_rows(
            rows=[{"compute_ms": 1.0, "comm_ms": 2.0}], native_counters={"nccl_retrans": 3}
        )
        assert rows[0]["compute_ms"] == 1.0
        assert rows[-1]["native_counters"]["nccl.nccl_retrans"] == 3

    def test_trace_verdict_requires_the_full_chain(self):
        blocked = tr.trace_verdict(
            all_ranks_covered=False, correlation_ok=False, clock_documented=False,
            breakdown_balanced=False, injections_classified=False, unknown_ratio_ok=False,
            confirmation_ok=False,
        )
        assert blocked["status"] == "INCONCLUSIVE" and blocked["blockers"]
        passed = tr.trace_verdict(
            all_ranks_covered=True, correlation_ok=True, clock_documented=True,
            breakdown_balanced=True, injections_classified=True, unknown_ratio_ok=True,
            confirmation_ok=True,
        )
        assert passed["status"] == "PASS"


@pytest.mark.unit
class TestTelemetryProjection:
    def test_c6_projection_uses_the_frozen_contract(self):
        fields = tm.S10ResultFields(run_id="r1", world_size=2, tp_degree=2, backend="nccl")
        result = tm.project_c6("r1", fields)
        assert result.summary[tm.C6_NAMESPACE]["world_size"] == 2
        assert result.correctness is not None and result.correctness.passed is False

    def test_c7_projection_maps_kinds_and_rejects_unknown_kinds(self):
        fields = tm.S10ResultFields(run_id="r1", world_size=2)
        records = [
            {"kind": "collective", "start_ns": 100, "group_id": "tp", "collective_seq": 0},
            {"kind": "transport", "start_ns": 200, "transport": "nvlink"},
        ]
        events = tm.project_c7(fields, records=records, run_id="r1")
        assert [event.event_type.value for event in events] == ["collective", "network"]
        assert events[0].attributes["distributed"]["group_id"] == "tp"
        with pytest.raises(ConfigError):
            tm.project_c7(fields, records=[{"kind": "mystery"}], run_id="r1")

    def test_coverage_summary_and_table_validation(self):
        coverage = tm.coverage_summary()
        assert coverage["c6"]["ok"] is True and coverage["c6"]["missing"] == []
        row = {name: None for name in tm.SCALING_ROW_FIELDS}
        assert tm.validate_table_row("scaling_row", row)["ok"] is True
        incomplete = dict(row)
        incomplete.pop("world_size")
        report = tm.validate_table_row("scaling_row", incomplete)
        assert report["ok"] is False and "world_size" in report["missing"]
        with pytest.raises(ConfigError):
            tm.validate_table_row("unknown_table", row)

    def test_table_hashes_are_stable(self):
        rows = {"scaling_row": [{name: 0 for name in tm.SCALING_ROW_FIELDS}]}
        first = tm.table_hashes(rows)
        second = tm.table_hashes(rows)
        assert first == second
