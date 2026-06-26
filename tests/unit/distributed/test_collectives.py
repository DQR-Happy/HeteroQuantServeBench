"""E10-02 interface tests: semantics, oracles, bandwidth formulas, grids, fits."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import backend as be
from hqsb.distributed import collectives as co


@pytest.mark.unit
class TestSpecsAndPayload:
    def test_count_semantics_is_frozen_per_op(self):
        assert set(co.COUNT_SEMANTICS) == set(co.OPS)
        spec = co.spec_for("reduce_scatter", dtype="fp32")
        assert "total_input_count" in spec.count_semantics

    def test_broadcast_needs_a_root_and_others_must_not_have_one(self):
        with pytest.raises(ConfigError):
            co.spec_for("broadcast", dtype="fp16")
        with pytest.raises(ConfigError):
            co.CollectiveSpec(op="all_reduce", count_semantics="x", root=0)

    def test_payload_bytes_uses_dtype_width(self):
        spec = co.spec_for("all_reduce", dtype="fp16")
        assert spec.payload_bytes(numel_per_rank=8) == 16


@pytest.mark.unit
class TestBandwidthFormulas:
    def test_nccltests_corrections(self):
        assert co.bus_correction("all_reduce", 2) == pytest.approx(1.0)
        assert co.bus_correction("all_gather", 4) == pytest.approx(0.75)
        assert co.bus_correction("broadcast", 8) == pytest.approx(1.0)

    def test_hqsb_row_carries_formula_ids(self):
        row = co.bandwidth_row(
            op="all_reduce", world_size=2, numel_per_rank=1024, dtype="fp32",
            latency_us=100.0, algorithm="ring", protocol="Simple",
        )
        assert row["algbw_formula_id"] == "hqsb.algbw.v1"
        assert row["busbw_GBps"] == pytest.approx(row["algbw_GBps"] * 1.0)

    def test_vendor_native_row_keeps_derived_column_named(self):
        row = co.bandwidth_row(
            op="all_reduce", world_size=2, numel_per_rank=1024, dtype="fp32",
            latency_us=100.0, algorithm="auto", protocol="auto", source="hccl_native",
        )
        assert row["busbw_GBps"] == ""
        assert row["normalized_nccltests_bus_correction"] > 0


@pytest.mark.unit
class TestOracles:
    def test_all_reduce_sum_and_max(self):
        inputs = [[1, 2], [3, 4]]
        assert co.oracle_all_reduce(inputs) == [4, 6]
        assert co.oracle_all_reduce(inputs, reduce_op="max") == [3, 4]

    def test_all_gather_follows_group_rank_order(self):
        inputs = [[10, 11], [20, 21]]
        assert co.oracle_all_gather(inputs) == [10, 11, 20, 21]
        # a non-natural group order must change the concatenation order
        assert co.oracle_all_gather(inputs, ordered_ranks=[1, 0]) == [20, 21, 10, 11]
        with pytest.raises(ConfigError):
            co.oracle_all_gather(inputs, ordered_ranks=[0, 0])

    def test_reduce_scatter_keeps_the_rank_segment(self):
        inputs = [[1, 1, 1, 1], [2, 2, 2, 2]]
        assert co.oracle_reduce_scatter(inputs, 0) == [3, 3]
        assert co.oracle_reduce_scatter(inputs, 1) == [3, 3]
        with pytest.raises(ConfigError):
            co.oracle_reduce_scatter([[1, 1, 1], [2, 2, 2]], 0)

    def test_all_to_all_permutation_and_conservation(self):
        chunks = [[[0, 1], [2]], [[3], [4, 5]]]
        received = co.oracle_all_to_all(chunks)
        assert received[1][0] == [2]
        assert received[0][1] == [3]

    def test_all_to_all_v_conservation_is_checked(self):
        sends = {(0, 1): [1, 2], (1, 0): [3]}
        received = co.oracle_all_to_all_v(sends, world_size=2)
        assert received[(1, 0)] == [1, 2]
        with pytest.raises(ConfigError):
            co.oracle_all_to_all_v({(0, 9): [1]}, world_size=2)

    def test_rank_coded_pattern_identifies_rank_and_index(self):
        assert co.rank_coded_pattern(0, 0) != co.rank_coded_pattern(1, 0)
        values = co.vector_for_rank(3, 4)
        assert len(values) == 4 and all(isinstance(value, int) for value in values)


@pytest.mark.unit
class TestComparison:
    def test_compare_vectors_locates_the_first_error(self):
        comparison = co.compare_vectors([1.0, 2.0, 3.0], [1.0, 2.5, 3.0], tolerance=0.1)
        assert comparison.ok is False
        assert comparison.first_bad_index == 1

    def test_nan_and_inf_fail_the_gate(self):
        comparison = co.compare_vectors([1.0], [float("nan")], tolerance=1e9)
        assert comparison.ok is False and comparison.nan_count == 1

    def test_length_mismatch_is_reported(self):
        assert co.compare_vectors([1.0], [1.0, 2.0], tolerance=0.0).ok is False


@pytest.mark.unit
class TestGuards:
    def test_guard_region_detects_an_overrun(self):
        buffer = co.GuardedBuffer(name="out", values=[1, 2, 3], guard_slots=2, guard_value=-1)
        poisoned = buffer.build()
        poisoned[0] = 7
        report = buffer.check_guard(poisoned)
        assert report["ok"] is False and report["overruns"]

    def test_alias_rules(self):
        assert co.alias_check(True, "buf", "buf")["ok"] is True
        with pytest.raises(ConfigError):
            co.alias_check(True, "a", "b")
        with pytest.raises(ConfigError):
            co.alias_check(False, "a", "a")


@pytest.mark.unit
class TestGrids:
    def test_size_grid_covers_latency_transition_and_model_neighbourhood(self):
        cases = co.message_size_grid(dtype="fp16", model_payloads=[4096])
        categories = {case.category for case in cases}
        assert "latency" in categories
        assert "model_neighbourhood" in categories
        assert all(case.payload_bytes > 0 for case in cases)

    def test_filtered_case_must_carry_a_reason(self):
        cases = co.message_size_grid(dtype="fp32", max_bytes=1 << 20, memory_limit_bytes=1024)
        filtered = [case for case in cases if case.status == "FILTERED"]
        assert filtered and all(case.filtered_reason for case in filtered)

    def test_rank_grid_marks_missing_cells_with_a_reason(self):
        cases = co.rank_grid(
            available_ranks=[2, 4],
            node_families=[(1, 8)],
            placement_hashes={2: "h2", 4: "h4"},
            missing_reason="only one node in scope",
        )
        assert all(case.available for case in cases)
        assert all(case.missing_reason == "" for case in cases)

    def test_dtype_capability_probe_returns_unknown_for_unrecorded_cells(self):
        matrix = be.CapabilityMatrix()
        matrix.record("all_reduce", "fp16", "nccl", state="SUPPORTED", reason_code="probe.ok")
        rows = co.dtype_capability_probe(matrix, op="all_reduce", dtypes=["fp16", "bf16"], backend="nccl")
        assert rows[0]["state"] == "SUPPORTED"
        assert rows[1]["state"] == "UNKNOWN"
        assert rows[1]["allows_execution"] is False


@pytest.mark.unit
class TestSweepAndFit:
    def test_profiler_runs_are_separate_and_barriers_must_be_declared(self):
        with pytest.raises(ConfigError):
            co.LatencySweepPlan(profiler_enabled=True)
        with pytest.raises(ConfigError):
            co.LatencySweepPlan(per_case_barrier=True, barrier_note="")
        with pytest.raises(ConfigError):
            co.LatencySweepPlan(independent_jobs=1)

    def test_correctness_failure_stops_performance(self):
        spec = co.spec_for("all_reduce", dtype="fp16")
        inputs = co.vector_for_rank(0, 4), co.vector_for_rank(1, 4)
        bad_row = co.correctness_row(
            spec=spec, rank=0, inputs=inputs, actual=[0.0, 0.0, 0.0, 0.0], tolerance=0.0
        )
        decision = co.should_stop_performance([bad_row])
        assert decision["stop"] is True

    def test_alpha_beta_fit_needs_two_segments(self):
        with pytest.raises(ConfigError):
            co.fit_alpha_beta_segments([], segment_boundaries=[8])

    def test_alpha_beta_segments_predict_and_keep_residuals(self):
        points = [
            {"payload_bytes": 8, "latency_us": 10.0},
            {"payload_bytes": 16, "latency_us": 14.0},
            {"payload_bytes": 4096, "latency_us": 30.0},
            {"payload_bytes": 8192, "latency_us": 50.0},
        ]
        segments = co.fit_alpha_beta_segments(points, segment_boundaries=[8, 64, 1 << 20])
        assert len(segments) == 2
        assert segments[0].residual_rms_us >= 0.0
        assert segments[1].predict_us(4096) > 0

    def test_crossover_detection_finds_the_slope_change(self):
        points = [
            {"payload_bytes": 100, "latency_us": 10.0, "algorithm": "tree"},
            {"payload_bytes": 200, "latency_us": 11.0, "algorithm": "tree"},
            {"payload_bytes": 400, "latency_us": 40.0, "algorithm": "ring"},
            {"payload_bytes": 800, "latency_us": 80.0, "algorithm": "ring"},
        ]
        crossovers = co.detect_crossover(points)
        assert crossovers and crossovers[0]["needs_denser_sampling"] is True

    def test_projection_refuses_extrapolation(self):
        segments = [
            co.AlphaBetaSegment(
                lower_bytes=8, upper_bytes=64, alpha_us=5.0, beta_us_per_byte=0.01,
                residual_rms_us=0.1, points=3,
            )
        ]
        with pytest.raises(ConfigError):
            co.project_model_messages(
                [{"phase": "decode", "layer": 0, "collective": "all_reduce",
                  "payload_bytes": 4096}],
                segments,
            )

    def test_crosscheck_flags_unexplained_deltas(self):
        hqsb_rows = [
            {"op": "all_reduce", "world_size": 2, "payload_bytes": 1024, "latency_us": 200.0}
        ]
        official = [
            {"op": "all_reduce", "world_size": 2, "payload_bytes": 1024, "latency_us": 100.0}
        ]
        report = co.crosscheck_with_official_tools(hqsb_rows, official)
        assert report["ok"] is False and report["unexplained"]


@pytest.mark.unit
class TestStopRulesAndLoopback:
    def test_stop_rules_trigger_with_reasons(self):
        decision = co.apply_stop_rules({"correctness_failure": True})
        assert decision["stop"] is True
        assert "correctness_failure" in decision["reasons"]

    def test_loopback_executor_is_simulated_and_never_a_claim(self):
        executor = co.LoopbackCollectiveExecutor(world_size=2)
        inputs = {0: [1, 2], 1: [3, 4]}
        result = executor.execute(co.spec_for("all_reduce", dtype="fp16"), inputs=inputs)
        assert result.outputs_by_rank[0] == [4, 6]
        assert result.simulated is True
        assert result.claim_allowed() is False

    def test_loopback_reports_wrong_rank_count(self):
        executor = co.LoopbackCollectiveExecutor(world_size=2)
        result = executor.execute(co.spec_for("all_reduce", dtype="fp16"), inputs={0: [1]})
        assert result.error

    def test_requested_actual_requires_a_reason(self):
        with pytest.raises(ConfigError):
            be.RequestedActual(requested="ring", actual="auto")
        record = be.RequestedActual(
            requested="ring", actual="auto", reason_code="fallback.auto",
            reason="the locked backend rejected the forced algorithm",
        )
        assert record.degraded is True

    def test_timing_calibration_refuses_host_return_time(self):
        with pytest.raises(ConfigError):
            be.TimingCalibration(
                enqueue_ns=0, completion_ns=10, method="host", host_api_return_used=True
            ).validate()
        calibration = be.TimingCalibration(
            enqueue_ns=100, completion_ns=350, method="event"
        )
        assert calibration.completion_latency_ns == 250
