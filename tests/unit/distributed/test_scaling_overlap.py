"""E10-05 / E10-06 interface tests: scaling honesty and interval-based overlap."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import overlap as ov
from hqsb.distributed import scaling as sc


@pytest.mark.unit
class TestScalingPrereg:
    def test_preregistration_needs_hypothesis_and_non_claims(self):
        with pytest.raises(ConfigError):
            sc.ScalingPreregistration(
                primary_metric="latency_ms", secondary_metrics=(), non_claims=(),
                workload_id="w", hypothesis="h",
            )
        with pytest.raises(ConfigError):
            sc.ScalingPreregistration(
                primary_metric="latency_ms", secondary_metrics=(), non_claims=("no 4-card claim",),
                workload_id="w", hypothesis="",
            )
        record = sc.ScalingPreregistration(
            primary_metric="latency_ms", secondary_metrics=("throughput",),
            non_claims=("no 4-card claim",), workload_id="balanced", hypothesis="TP=2 lowers latency",
        )
        assert record.independent_runs == 3

    def test_weak_definition_must_be_named(self):
        with pytest.raises(ConfigError):
            sc.WeakWorkUnit(
                definition="token_weak", held_constant="per-device tokens", per_device_work=1.0,
                global_formula="p × tokens", name="",
            )
        unit = sc.WeakWorkUnit(
            definition="token_weak", held_constant="per-device tokens", per_device_work=1.0,
            global_formula="p × tokens", name="token-weak",
        )
        assert unit.global_work_for(4) == 4.0

    def test_capacity_protocol_requires_criteria_and_margin(self):
        with pytest.raises(ConfigError):
            sc.CapacityProtocol(ladder=(1, 2), oom_margin_fraction=1.0, max_runnable_criteria=("correct",))
        protocol = sc.CapacityProtocol(
            ladder=(512, 1024), oom_margin_fraction=0.1, max_runnable_criteria=("correctness", "stability")
        )
        assert protocol.ladder == (512, 1024)


@pytest.mark.unit
class TestResourceMatrixAndBaseline:
    def test_missing_cells_are_not_zero(self):
        matrix = sc.ResourceMatrix(
            cells=(
                sc.ResourceMatrixCell(device_count=2, node_count=1, tp_degree=2, available=True),
                sc.ResourceMatrixCell(
                    device_count=4, node_count=2, tp_degree=4, available=False,
                    missing_reason="no second node in scope", resource_date="2026-09-19",
                ),
            )
        )
        rows = matrix.as_rows()
        missing = [row for row in rows if row["status"] == "MISSING"][0]
        assert missing["latency_ms"] is None and missing["reason"]

    def test_mixed_node_families_are_refused(self):
        rows = [
            {"topology_family": "single_node"},
            {"topology_family": "multi_node_2"},
        ]
        assert sc.refuse_mixed_families(rows)

    def test_no_t1_refuses_strong_speedup(self):
        baseline = sc.baseline_calibration(t1_latency_ms=None, reason="model does not fit")
        result = sc.strong_speedup(baseline, degree=2, latency_ms=100.0)
        assert result.metric_name == "speedup_from_p0"
        assert result.speedup is None and result.efficiency is None
        assert "must not be called" in result.note

    def test_real_t1_computes_speedup_and_efficiency(self):
        baseline = sc.baseline_calibration(t1_latency_ms=200.0)
        result = sc.strong_speedup(baseline, degree=2, latency_ms=100.0)
        assert result.speedup == pytest.approx(2.0)
        assert result.efficiency == pytest.approx(1.0)


@pytest.mark.unit
class TestDecompositionAndPairability:
    def test_decomposition_balances_and_reports_unexplained_idle(self):
        good = sc.decomposition(
            compute_ms=60.0, comm_ms=30.0, overlap_ms=10.0, wait_ms=10.0,
            host_gap_ms=5.0, sync_ms=5.0, total_ms=100.0,
        )
        assert good.exposed_comm_ms == pytest.approx(20.0)
        assert good.balanced is True
        leaky = sc.decomposition(
            compute_ms=10.0, comm_ms=10.0, overlap_ms=0.0, wait_ms=0.0,
            host_gap_ms=0.0, sync_ms=0.0, total_ms=100.0,
        )
        assert leaky.balanced is False
        assert leaky.unexplained_idle_ms == pytest.approx(80.0)

    def test_pairability_detects_identity_mismatch(self):
        left = {"model_artifact_hash": "m", "precision": "fp16", "global_batch": 4}
        right = {"model_artifact_hash": "m", "precision": "fp16", "global_batch": 8}
        verdict = sc.pairability(left, right)
        assert verdict.ok is False
        assert "global_batch" in verdict.differing_fields
        split = sc.split_pairable([left, right])
        assert len(split["pairable"]) == 1 and len(split["non_pairable"]) == 1

    def test_fit_refuses_extrapolation(self):
        fit = sc.fit_scaling_model(
            [
                {"degree": 1, "latency_ms": 100.0},
                {"degree": 2, "latency_ms": 60.0},
                {"degree": 4, "latency_ms": 45.0},
            ]
        )
        assert fit.measured_degrees == (1, 2, 4)
        with pytest.raises(ConfigError):
            fit.predict_ms(8)

    def test_flag_anomalies_keeps_raw_and_refuses_fastest_picking(self):
        rows = [
            {"run_id": "a", "world_size": 2, "latency_ms": 100.0},
            {"run_id": "b", "world_size": 2, "latency_ms": 100.5},
            {"run_id": "c", "world_size": 2, "latency_ms": 400.0},
        ]
        flagged = sc.flag_anomalies(rows, variability={2: 2.0})
        assert flagged and flagged[0]["kept_in_raw"] is True
        with pytest.raises(ConfigError):
            sc.pick_fastest_is_forbidden("use the fastest run per degree")

    def test_verdict_ordering(self):
        assert sc.scaling_verdict(
            correctness_ok=False, pairable=True, resource_grid_complete=True, repeat_runs=3,
            latency_effect=0.5, effect_threshold=0.1,
        )["status"] == "FAIL"
        assert sc.scaling_verdict(
            correctness_ok=True, pairable=True, resource_grid_complete=True, repeat_runs=3,
            latency_effect=0.5, effect_threshold=0.1, capacity_gain=True, capacity_stable=True,
        )["status"] == "PASS_POSITIVE_LATENCY"
        assert sc.scaling_verdict(
            correctness_ok=True, pairable=True, resource_grid_complete=True, repeat_runs=3,
            latency_effect=0.0, effect_threshold=0.1,
        )["status"] == "PASS_NEGATIVE"
        assert sc.scaling_verdict(
            correctness_ok=True, pairable=True, resource_grid_complete=False, repeat_runs=1,
            latency_effect=None, effect_threshold=0.1,
        )["status"] == "INCONCLUSIVE"


@pytest.mark.unit
class TestIntervalAlgebra:
    def test_union_merges_overlapping_intervals(self):
        intervals = [ov.Interval(0, 10), ov.Interval(5, 20), ov.Interval(30, 40)]
        assert ov.union_measure_ns(intervals) == 30

    def test_intersection_is_the_true_overlap(self):
        compute = [ov.Interval(0, 100)]
        comm = [ov.Interval(60, 200)]
        assert ov.intersection_measure_ns(compute, comm) == 40

    def test_overlap_cannot_exceed_min_side(self):
        with pytest.raises(ConfigError):
            ov.OverlapMetrics(
                compute_active_ms=10.0, comm_active_ms=10.0, overlap_ms=20.0, exposed_comm_ms=-10.0,
                overlap_fraction_comm=None, net_gain_ms=None,
            )

    def test_metrics_and_uncertainty(self):
        compute = [ov.Interval(0, 1_000_000)]
        comm = [ov.Interval(500_000, 1_500_000)]
        metrics = ov.overlap_metrics_from_intervals(
            compute, comm, no_overlap_total_ms=1.5, overlap_total_ms=1.0, clock_uncertainty_ms=0.01
        )
        assert metrics.overlap_ms == pytest.approx(0.5)
        assert metrics.exposed_comm_ms == pytest.approx(0.5)
        assert metrics.net_gain_ms == pytest.approx(0.5)
        assert metrics.conclusive is True

    def test_negative_duration_is_refused(self):
        with pytest.raises(ConfigError):
            ov.Interval(start_ns=10, end_ns=5)


@pytest.mark.unit
class TestOverlapSchedule:
    def test_legal_window_requires_independent_compute(self):
        dag = ov.TensorDependencyDAG(
            nodes=(
                ov.DagNode(
                    tensor_role="mlp_output", producer="mlp", consumers=("residual",),
                    collective_seq=0, independent_compute=("next_layer_qkv",),
                ),
                ov.DagNode(
                    tensor_role="attention_output", producer="attention", consumers=("residual",),
                    collective_seq=1, independent_compute=(),
                ),
            )
        )
        feasible = ov.legal_overlap_window(
            dag, collective_seq=0, comm_ms=10.0, compute_window_ms={"next_layer_qkv": 4.0}
        )
        assert feasible.feasible is True and feasible.theoretical_max_overlap_ms == pytest.approx(4.0)
        blocked = ov.legal_overlap_window(
            dag, collective_seq=1, comm_ms=10.0, compute_window_ms={"next_layer_qkv": 4.0}
        )
        assert blocked.feasible is False
        assert "not allowed" in blocked.reason

    def test_schedule_identity_requires_one_variable(self):
        arms = [
            ov.ScheduleIdentity("blocking_baseline", "h", "schedule", "blocking", "blocking"),
            ov.ScheduleIdentity("overlap", "h", "schedule", "overlap", "overlap"),
        ]
        assert ov.validate_only_variable(arms)["ok"] is True
        mixed = [
            ov.ScheduleIdentity("blocking_baseline", "h", "schedule", "blocking", "blocking"),
            ov.ScheduleIdentity("overlap", "h2", "chunk", "overlap", "overlap"),
        ]
        assert ov.validate_only_variable(mixed)["ok"] is False

    def test_fallback_schedule_needs_a_reason(self):
        with pytest.raises(ConfigError):
            ov.ScheduleIdentity("overlap", "h", "schedule", "overlap", "blocking_baseline")

    def test_chunk_candidates_filter_unaligned_sizes(self):
        candidates = ov.chunk_candidates(
            message_bytes=1000, natural_boundaries=(300,), alignment_bytes=16
        )
        assert any(not candidate.supported for candidate in candidates)
        assert all(
            candidate.filtered_reason for candidate in candidates if not candidate.supported
        )

    def test_work_lifetime_and_actual_vs_planned(self):
        completion = ov.CompletionEvent("t", 0, "comm", 1000, 5000)
        assert ov.validate_work_lifetime(completion=completion, consumer_wait_ns=2000)["ok"] is True
        assert ov.validate_work_lifetime(completion=completion, consumer_wait_ns=6000)["ok"] is False
        planned = [ov.ScheduleTraceEntry(0, (0, 10), "p", "c", "x", "s", "e", "w", 0, 0, 10)]
        assert ov.compare_actual_vs_planned(planned, planned)["ok"] is True

    def test_race_probe_and_global_sync_audit(self):
        plan = ov.RaceProbePlan()
        assert plan.repeats >= 16
        results = [{"deterministic": True, "error": ""}, {"deterministic": False, "error": "wrong token"}]
        assert ov.evaluate_race_probe(results)["ok"] is False
        audit = ov.global_sync_audit(
            unrelated_stream_events=[{"blocked_by_comm": True}],
            device_sync_calls=3,
            necessary_tensor_waits=1,
        )
        assert audit["ok"] is False

    def test_contention_and_bound_comparison(self):
        contention = ov.contention_report(
            solo_compute_ms=10.0, solo_comm_ms=10.0, concurrent_compute_ms=12.0,
            concurrent_comm_ms=12.0,
        )
        assert contention["both_slow"] is True
        bound = ov.OverlapBound(
            collective_seq=0, independent_compute_ms=5.0, comm_ms=10.0,
            theoretical_max_overlap_ms=5.0, feasible=True,
        )
        comparison = ov.compare_to_bound(net_gain_ms=2.0, bound=bound, ready_late_ms=2.0)
        assert comparison["shortfall_ms"] == pytest.approx(3.0)

    def test_overlap_verdict_chain(self):
        assert ov.overlap_verdict(
            correctness_ok=True, no_race=True, timeline_proves_overlap=False,
            exposed_comm_reduced=False, e2e_improved=False, confirmation_significant=False,
        )["status"] == "FAIL"
        assert ov.overlap_verdict(
            correctness_ok=True, no_race=True, timeline_proves_overlap=True,
            exposed_comm_reduced=False, e2e_improved=False, confirmation_significant=False,
        )["status"] == "PASS_NEGATIVE"
        assert ov.overlap_verdict(
            correctness_ok=True, no_race=True, timeline_proves_overlap=True,
            exposed_comm_reduced=True, e2e_improved=True, confirmation_significant=True,
        )["status"] == "PASS_POSITIVE"
        assert ov.overlap_verdict(
            correctness_ok=True, no_race=True, timeline_proves_overlap=True,
            exposed_comm_reduced=True, e2e_improved=True, confirmation_significant=True,
            conclusive=False,
        )["status"] == "INCONCLUSIVE"

    def test_phase_policy_separates_prefill_and_decode(self):
        policies = ov.phase_policy(prefill_verdict="PASS_POSITIVE", decode_verdict="PASS_NEGATIVE")
        by_phase = {policy.phase: policy.schedule for policy in policies}
        assert by_phase["prefill"] == "overlap"
        assert by_phase["decode"] == "async_no_overlap"

    def test_abba_confirmation_requires_repeats(self):
        with pytest.raises(ConfigError):
            ov.AbbaConfirmation(
                order=("A", "B"), paired_effect_ms=1.0, ci_low_ms=0.1, ci_high_ms=2.0,
                runs_per_arm=1,
            )
        confirmation = ov.abba_confirmation(
            paired_effect_ms=1.0, ci_low_ms=0.1, ci_high_ms=2.0, runs_per_arm=4
        )
        assert confirmation.significant is True
