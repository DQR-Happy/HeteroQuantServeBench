"""Unit tests for the E03-02 RMSNorm shape-performance primitives.

Deliberately GPU-free: they pin the *measurement* contract — the shape matrix
and its source tags, the S02 call matrix, the byte/FLOP conventions, the
statistics and guard-band winner rule, the region labels and the heatmap
renderer. If any of that drifts, the on-device verdict in
``scripts/audit/run_e03_02_shape_heatmap.py`` would silently change meaning, so
it is locked here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hqsb.benchmark import rmsnorm_perf as rp
from hqsb.benchmark.hotspot_decision import GUARD_BAND


@pytest.mark.unit
class TestShapePlan:
    def test_mandatory_axes_present(self):
        plan = rp.shape_plan()
        rows = {entry["rows"] for entry in plan}
        hidden = {entry["hidden"] for entry in plan}
        assert set(rp.MANDATORY_ROWS).issubset(rows)
        assert set(rp.MANDATORY_H).issubset(hidden)
        assert set(rp.BOUNDARY_H).issubset(hidden)

    def test_shape_ids_unique(self):
        plan = rp.shape_plan()
        ids = [entry["shape_id"] for entry in plan]
        assert len(ids) == len(set(ids))
        assert all(entry["sources"] for entry in plan)

    def test_odd_hidden_values_are_marked_as_boundary(self):
        plan = {entry["shape_id"]: entry for entry in plan_entries()}
        for hidden in rp.BOUNDARY_H:
            entry = plan[rp.shape_id(1, hidden)]
            assert "DESIGN_BOUNDARY" in entry["sources"]

    def test_6144_and_8192_are_synthetic_not_runtime(self):
        plan = {entry["shape_id"]: entry for entry in rp.shape_plan()}
        for hidden in (6144, 8192):
            entry = plan[rp.shape_id(1, hidden)]
            assert "S02_RUNTIME" not in entry["sources"]
            assert "DESIGN_MANDATORY" in entry["sources"]

    def test_real_shapes_are_tagged_and_weighted(self):
        plan = {entry["shape_id"]: entry for entry in rp.shape_plan()}
        real = [e for e in plan.values() if "S02_RUNTIME" in e["sources"]]
        assert real, "the S02 runtime shapes must be part of the matrix"
        assert all(entry["call_weight"] > 0 for entry in real)


def plan_entries():
    return rp.shape_plan()


@pytest.mark.unit
class TestCallMatrix:
    def test_decode_calls_are_output_tokens_minus_one(self):
        assert rp.s02_decode_calls(16) == 15
        assert rp.s02_decode_calls(1) == 0

    def test_matrix_uses_per_module_and_per_request_counts(self):
        matrix = rp.s02_call_matrix()
        for row in matrix:
            assert row["calls_per_request"] == (
                rp.S02_LAYERS * row["calls_per_module_instance"]
            )
            if row["phase"] == "prefill":
                assert row["calls_per_module_instance"] == 1
            else:
                assert row["calls_per_module_instance"] == row["output_tokens"] - 1

    def test_hidden_and_head_dim_are_not_merged(self):
        matrix = rp.s02_call_matrix()
        hidden_widths = {row["hidden"] for row in matrix}
        assert hidden_widths == {rp.S02_HIDDEN, rp.S02_HEAD_DIM}

    def test_weights_aggregate_per_shape(self):
        weights = rp.shape_call_weights()
        assert weights[rp.shape_id(1, rp.S02_HIDDEN)]["calls_per_request_total"] > 0
        assert weights[rp.shape_id(rp.S02_Q_HEADS, rp.S02_HEAD_DIM)][
            "calls_per_request_total"
        ] > 0


@pytest.mark.unit
class TestConventions:
    def test_logical_bytes_use_three_element_traversals(self):
        assert rp.logical_bytes(4, 100, "fp32") == 4 * 100 * 3 * 4
        assert rp.logical_bytes(4, 100, "fp16") == 4 * 100 * 3 * 2

    def test_declared_flops_follow_the_frozen_convention(self):
        assert rp.declared_flops(1, 100) == 402
        assert rp.declared_flops(3, 4) == 3 * 18

    def test_derived_metrics_are_self_consistent(self):
        metrics = rp.derived_metrics(
            rows=2, hidden=100, dtype="fp32", device_seconds=1e-3
        )
        assert metrics["logical_bytes"] == 2 * 100 * 3 * 4
        assert metrics["effective_GBps"] == pytest.approx(
            metrics["logical_bytes"] / 1e-3 / 1e9
        )
        assert metrics["effective_GFLOPs"] == pytest.approx(
            metrics["declared_flops"] / 1e-3 / 1e9
        )
        assert metrics["arithmetic_intensity_flop_per_byte"] == pytest.approx(
            metrics["declared_flops"] / metrics["logical_bytes"]
        )

    def test_host_metrics_never_override_device_metrics(self):
        metrics = rp.derived_metrics(
            rows=2, hidden=100, dtype="fp32",
            device_seconds=1e-3, host_seconds=2e-3, submit_seconds=1.5e-3,
        )
        assert metrics["effective_GBps"] != metrics["host_effective_GBps"]
        assert metrics["device_seconds"] == 1e-3

    def test_zero_device_time_is_rejected(self):
        with pytest.raises(ValueError):
            rp.derived_metrics(rows=1, hidden=8, dtype="fp32", device_seconds=0.0)


@pytest.mark.unit
class TestStatistics:
    def test_summary_reports_spread(self):
        stats = rp.summarize_samples([1.0, 1.0, 1.0, 1.0])
        assert stats["median"] == pytest.approx(1.0)
        assert stats["mad"] == pytest.approx(0.0)
        assert stats["cv"] == pytest.approx(0.0)

    def test_summary_ignores_non_finite_values(self):
        stats = rp.summarize_samples([1.0, float("nan"), float("inf"), 3.0])
        assert stats["count"] == 2
        assert stats["median"] == pytest.approx(2.0)

    def test_bootstrap_ci_is_deterministic_and_brackets_the_median(self):
        samples = [1.0, 1.05, 0.95, 1.02, 0.98, 1.01]
        first = rp.bootstrap_ci(samples)
        second = rp.bootstrap_ci(samples)
        assert first == second
        assert first["low"] <= np.median(samples) <= first["high"]

    def test_single_sample_ci_is_flagged(self):
        ci = rp.bootstrap_ci([2.0])
        assert ci["n"] == 1
        assert "not a CI" in ci["note"]

    def test_paired_speedup_requires_equal_lengths(self):
        with pytest.raises(ValueError):
            rp.paired_speedup([1.0, 2.0], [1.0])

    def test_paired_speedup_is_ratio_of_pairs(self):
        result = rp.paired_speedup([2.0, 4.0], [1.0, 2.0])
        assert result["ratios"] == [2.0, 2.0]
        assert result["median"] == pytest.approx(2.0)


@pytest.mark.unit
class TestGuardBandVerdicts:
    def test_clear_win_above_the_guard_band(self):
        ci = {"low": 1.20, "high": 1.40}
        assert rp.verdict_from_ci(ci) == "CANDIDATE_WINS"

    def test_within_guard_band_is_a_tie(self):
        ci = {"low": 1.01, "high": 1.20}
        assert rp.verdict_from_ci(ci) == "TIE_OR_WITHIN_GUARD_BAND"

    def test_clear_regression(self):
        ci = {"low": 0.70, "high": 0.95}
        assert rp.verdict_from_ci(ci) == "REGRESSION"

    def test_small_non_significant_difference_is_no_gain(self):
        ci = {"low": 0.99, "high": 1.02}
        assert rp.verdict_from_ci(ci) == "NO_GAIN"

    def test_guard_band_default_matches_e02_09(self):
        assert GUARD_BAND["micro_min_relative_improvement"] == pytest.approx(0.05)


@pytest.mark.unit
class TestWinnerRow:
    def test_candidate_win_requires_ci_above_guard_band(self):
        row = rp.winner_row(
            shape_id_value="r512_h2048", rows=512, hidden=2048, dtype="fp32",
            paired_by_variant={"v2_vectorized": [1.5, 1.6, 1.45, 1.55]},
        )
        assert row["decision"] == "v2_vectorized"
        assert row["decision_kind"] == "CANDIDATE_WINS"
        assert "v0_shared" in row["tie_group"] or len(row["tie_group"]) == 1

    def test_no_candidate_crossing_the_band_retains_the_baseline(self):
        row = rp.winner_row(
            shape_id_value="r1_h128", rows=1, hidden=128, dtype="fp32",
            paired_by_variant={"v2_vectorized": [1.01, 1.02, 0.99, 1.0]},
        )
        assert row["decision"] == "v0_shared"
        assert row["decision_kind"] == "BASELINE_RETAINED"

    def test_regression_is_reported_not_hidden(self):
        row = rp.winner_row(
            shape_id_value="r1_h128", rows=1, hidden=128, dtype="fp32",
            paired_by_variant={"v1_warp_shuffle": [0.8, 0.82, 0.79, 0.81]},
        )
        assert row["decision_kind"] == "BASELINE_RETAINED"
        verdicts = {c["variant"]: c["verdict"] for c in row["candidates"]}
        assert verdicts["v1_warp_shuffle"] == "REGRESSION"


@pytest.mark.unit
class TestRegions:
    def test_rows_one_is_flagged_as_single_cta(self):
        region = rp.classify_region(
            rows=1, hidden=2048, dtype="fp16", effective_gbps=5.0,
            ceiling_gbps=50.0, sm_count=8, variant="v2_vectorized",
        )
        assert "launch_or_single_cta_limited_candidate" in region["labels"]

    def test_low_bandwidth_is_not_declared_memory_bound(self):
        region = rp.classify_region(
            rows=64, hidden=2048, dtype="fp32", effective_gbps=1.0,
            ceiling_gbps=50.0, sm_count=8, variant="v0_shared",
        )
        assert "memory_throughput_sensitive_candidate" not in region["labels"]
        assert any("counter" in reason for reason in region["reasons"])

    def test_odd_hidden_takes_the_tail_path(self):
        region = rp.classify_region(
            rows=16, hidden=129, dtype="fp32", effective_gbps=20.0,
            ceiling_gbps=50.0, sm_count=8, variant="v2_vectorized",
        )
        assert "tail_or_scalar_path" in region["labels"]
        assert "vectorization_eligible" not in region["labels"]

    def test_aligned_fp32_is_vectorization_eligible(self):
        region = rp.classify_region(
            rows=64, hidden=2048, dtype="fp32", effective_gbps=40.0,
            ceiling_gbps=50.0, sm_count=8, variant="v2_vectorized",
        )
        assert "vectorization_eligible" in region["labels"]


@pytest.mark.unit
class TestCasePlan:
    def test_case_ids_are_unique_and_statuses_preassigned(self):
        cases, meta = rp.build_case_plan()
        ids = [case["case_id"] for case in cases]
        assert len(ids) == len(set(ids))
        assert {case["status"] for case in cases} <= {"ELIGIBLE", "EXPECTED_UNSUPPORTED"}
        assert meta["shape_count"] > 0

    def test_fp16_only_claims_v2_and_keeps_the_unsupported_cells(self):
        cases, _ = rp.build_case_plan()
        fp16 = [case for case in cases if case["dtype"] == "fp16"]
        eligible_variants = {
            c["variant"]
            for c in fp16
            if c["status"] == "ELIGIBLE" and c["variant"] != rp.FRAMEWORK_VARIANT
        }
        assert eligible_variants == {"v2_vectorized"}
        assert "v0_shared" in {c["variant"] for c in fp16}
        assert "v1_warp_shuffle" in {c["variant"] for c in fp16}
        assert all(
            c["status"] == "EXPECTED_UNSUPPORTED"
            for c in fp16
            if c["variant"] in ("v0_shared", "v1_warp_shuffle")
        )

    def test_fp32_claims_all_three_variants(self):
        cases, _ = rp.build_case_plan()
        fp32 = [
            c["variant"] for c in cases
            if c["dtype"] == "fp32" and c["status"] == "ELIGIBLE"
        ]
        assert set(fp32) == {"v0_shared", "v1_warp_shuffle", "v2_vectorized"}

    def test_framework_baseline_is_reduced_and_separate(self):
        cases, _ = rp.build_case_plan()
        framework = [c for c in cases if c["variant"] == rp.FRAMEWORK_VARIANT]
        assert framework, "the real shapes must also get a framework baseline"
        assert all(c["dtype"] == "fp16" for c in framework)
        assert all(c["group"] == "framework_baseline" for c in framework)

    def test_expected_case_ids_are_sorted_and_complete(self):
        cases, _ = rp.build_case_plan()
        ids = rp.expected_case_ids(cases)
        assert ids == sorted(ids)
        assert len(ids) == len(cases)


@pytest.mark.unit
class TestMatricesAndHeatmaps:
    def _summaries(self):
        rows = []
        for variant, median in (("v0_shared", 1.0), ("v1_warp_shuffle", 2.0)):
            for hidden in (128, 2048):
                for shape_rows in (1, 16):
                    rows.append(
                        {
                            "case_id": f"r{shape_rows}_h{hidden}|fp32|v|{variant}",
                            "shape_id": rp.shape_id(shape_rows, hidden),
                            "rows": shape_rows,
                            "hidden": hidden,
                            "dtype": "fp32",
                            "variant": variant,
                            "status": "MEASURED",
                            "group": "main",
                            "device_ms": {"median": median},
                        }
                    )
        return rows

    def test_matrix_keeps_missing_cells_missing(self):
        summaries = self._summaries()
        matrix = rp.build_matrix(
            summaries, dtype="fp32", variant="v0_shared", value_key="device_ms",
            row_labels=[128, 2048, 4096], col_labels=[1, 16, 4096],
        )
        assert matrix["row_labels"] == [128, 2048, 4096]
        assert matrix["col_labels"] == [1, 16, 4096]
        assert matrix["values"][0][2] is None
        assert matrix["values"][2][0] is None

    def test_matrix_widths_are_derived_when_axes_are_omitted(self):
        matrix = rp.build_matrix(
            self._summaries(), dtype="fp32", variant="v0_shared", value_key="device_ms"
        )
        assert matrix["row_labels"] == [128, 2048]
        assert matrix["col_labels"] == [1, 16]
        assert all(value is not None for row in matrix["values"] for value in row)

    def test_heatmap_svg_is_renderable_and_marks_unsupported(self):
        matrix = rp.build_matrix(
            self._summaries(), dtype="fp32", variant="v0_shared",
            value_key="device_ms",
        )
        matrix["status"][0][0] = "EXPECTED_UNSUPPORTED"
        matrix["values"][0][0] = None
        svg = rp.render_heatmap_svg(title="t", matrix=matrix, unit="ms")
        assert svg.startswith("<svg")
        assert "N/A" in svg

    def test_categorical_renderer_labels_every_cell(self):
        svg = rp.render_categorical_svg(
            title="t",
            row_labels=[128],
            col_labels=[1, 16],
            labels=[["MEASURED", "FAIL_CORRECTNESS"]],
        )
        assert "ok" in svg and "FAIL" in svg

    def test_matrix_csv_round_trips(self, tmp_path):
        matrix = rp.build_matrix(
            self._summaries(), dtype="fp32", variant="v0_shared",
            value_key="device_ms",
        )
        path = tmp_path / "m.csv"
        rp.write_matrix_csv(path, matrix)
        text = path.read_text(encoding="utf-8")
        assert "hidden\\rows" in text
        assert text.count("\n") == len(matrix["row_labels"]) + 1

    def test_canonical_hash_is_order_independent(self):
        assert rp.canonical_sha256({"a": 1, "b": 2}) == rp.canonical_sha256(
            {"b": 2, "a": 1}
        )
        assert json.loads(json.dumps({"a": 1})) == {"a": 1}
