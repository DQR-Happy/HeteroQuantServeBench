"""Tests for the E05-10 decision layer: registry, gates, Pareto, scenarios."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant import decision as dec


def _candidate(cid, *, hardware="h100", workload="short", memory=None, metrics=None, quality="PASS", execution=None, offline=0.0):
    metrics = dict(metrics or {"ttft_ms": 10.0, "tpot_ms": 1.0, "tpot_p95_ms": 2.0})
    metrics.setdefault("measurement_gate", {"passed": True})
    return dec.Candidate(
        identity={
            "candidate_id": cid,
            "model_revision": "rev",
            "method": "rtn",
            "config_hash": "c" * 64,
            "bits": 4,
            "scheme_hash": "s" * 64,
            "module_coverage": 0.9,
            "calibration_hash": "cal",
            "canonical_artifact_hash": "ca",
            "packed_variant_hash": "pv",
            "runtime": "triton",
            "kernel_provider": "triton",
            "hardware_fingerprint": hardware,
            "workload": workload,
            "quality_result_hash": "qr",
            "performance_result_hash": "pr",
            "fallback_policy": "fail_closed",
            "known_limitations": [],
        },
        gates={
            "correctness": {"passed": True},
            "artifact": {"passed": True},
        },
        quality={"gate": {"verdict": quality}},
        execution=execution or {
            "low_bit_executed": True,
            "observed_kernel": "hqsb_dequant_gemm_kernel",
            "claimed_bits": 4,
        },
        memory={"device_peak_bytes": memory if memory is not None else 1000},
        metrics=metrics,
        lifecycle={"offline_cost_s": offline},
        energy={"joules_per_token": 0.1},
    )


@pytest.mark.unit
class TestCompleteness:
    def test_missing_identity_field_marks_incomplete(self):
        candidate = _candidate("x")
        candidate.identity["calibration_hash"] = ""
        audit = candidate.completeness()
        assert audit["status"] == dec.INCOMPLETE
        assert "calibration_hash" in audit["missing_identity_fields"]

    def test_complete_candidate(self):
        audit = _candidate("x").completeness()
        assert audit["status"] == dec.COMPLETE

    def test_registry_rejects_duplicate_id(self):
        registry = dec.CandidateRegistry()
        registry.add(_candidate("x"))
        with pytest.raises(ConfigError):
            registry.add(_candidate("x"))


@pytest.mark.unit
class TestGates:
    def test_evaluate_gates_classifies(self):
        registry = dec.CandidateRegistry()
        registry.add(_candidate("good"))
        registry.add(_candidate("bad_quality", quality="FAIL"))
        rows = {row["candidate_id"]: row for row in dec.evaluate_gates(registry)["rows"]}
        assert rows["good"]["classification"] == dec.RECOMMENDED
        assert rows["bad_quality"]["classification"] == dec.REJECTED_CLAIM

    def test_execution_gate_fails_without_observed_symbol(self):
        registry = dec.CandidateRegistry()
        registry.add(_candidate("no_symbol", execution={"low_bit_executed": False}))
        rows = {row["candidate_id"]: row for row in dec.evaluate_gates(registry)["rows"]}
        assert "execution" in rows["no_symbol"]["failed_gates"]


@pytest.mark.unit
class TestPareto:
    def test_dominates(self):
        assert dec.dominates({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 2.0}, objectives=("a", "b"))
        assert not dec.dominates({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 2.0}, objectives=("a", "b"))

    def test_dominates_requires_all_objectives(self):
        with pytest.raises(ConfigError):
            dec.dominates({"a": 1.0}, {"a": 2.0}, objectives=("a", "missing"))

    def test_point_pareto_splits_front(self):
        registry = dec.CandidateRegistry()
        # "fast" dominates "slow" on both ttft and tpot.
        registry.add(_candidate("fast", metrics={"ttft_ms": 5.0, "tpot_ms": 0.5, "tpot_p95_ms": 1.0}, memory=1000))
        registry.add(_candidate("slow", metrics={"ttft_ms": 10.0, "tpot_ms": 1.0, "tpot_p95_ms": 2.0}, memory=2000))
        # memory candidate: not dominated because its memory is better.
        registry.add(_candidate("tiny", metrics={"ttft_ms": 20.0, "tpot_ms": 2.0, "tpot_p95_ms": 3.0}, memory=500))
        report = dec.point_pareto(
            registry, objectives=("ttft_ms", "tpot_ms", "device_peak_bytes")
        )
        front = report["fronts"][0]["pareto_front"]
        assert "fast" in front and "tiny" in front
        assert "slow" not in front

    def test_uncertainty_pareto_empty(self):
        registry = dec.CandidateRegistry()
        front = dec.uncertainty_pareto(registry)
        assert front.resamples == 0


@pytest.mark.unit
class TestScenarios:
    def test_no_feasible_candidate(self):
        registry = dec.CandidateRegistry()
        registry.add(_candidate("x", memory=100000, metrics={"ttft_ms": 10.0, "tpot_ms": 1.0, "tpot_p95_ms": 2.0}))
        scenario = dec.Scenario(name="mem", constraints={"device_peak_bytes_max": 100.0}, minimize=("tpot_ms",))
        report = dec.apply_scenarios(registry, [scenario])
        assert report["scenarios"][0]["outcome"] == "no_feasible_candidate"

    def test_feasible_candidate(self):
        registry = dec.CandidateRegistry()
        registry.add(_candidate("x", memory=50, metrics={"ttft_ms": 10.0, "tpot_ms": 1.0, "tpot_p95_ms": 2.0}))
        scenario = dec.Scenario(name="mem", constraints={"device_peak_bytes_max": 100.0}, minimize=("tpot_ms",))
        report = dec.apply_scenarios(registry, [scenario])
        assert report["scenarios"][0]["feasible"] == ["x"]

    def test_quality_gate_blocks_feasibility(self):
        registry = dec.CandidateRegistry()
        registry.add(_candidate("x", memory=50, quality="FAIL", metrics={"ttft_ms": 10.0, "tpot_ms": 1.0, "tpot_p95_ms": 2.0}))
        scenario = dec.Scenario(name="mem", constraints={"device_peak_bytes_max": 100.0}, minimize=("tpot_ms",))
        report = dec.apply_scenarios(registry, [scenario])
        assert report["scenarios"][0]["outcome"] == "no_feasible_candidate"

    def test_unknown_constraint_suffix(self):
        with pytest.raises(ConfigError):
            dec.apply_scenarios(
                dec.CandidateRegistry(),
                [dec.Scenario(name="x", constraints={"ttft_ms_avg": 1.0}, minimize=())],
            )


@pytest.mark.unit
class TestOfflineAmortization:
    def test_amortization_over_volumes(self):
        report = dec.offline_amortization(120.0, request_volumes=[10, 100, 1000])
        assert report["offline_cost_s"] == 120.0
        assert report["sensitivity"][0]["amortized_cost_s_per_request"] == pytest.approx(12.0)

    def test_non_positive_volume_refused(self):
        with pytest.raises(ConfigError):
            dec.offline_amortization(1.0, request_volumes=[0])


@pytest.mark.unit
class TestRecommendations:
    def test_recommendation_matrix_lists_rejected(self):
        registry = dec.CandidateRegistry()
        registry.add(_candidate("good", memory=50, metrics={"ttft_ms": 10.0, "tpot_ms": 1.0, "tpot_p95_ms": 2.0}))
        registry.add(_candidate("bad", quality="FAIL", memory=50, metrics={"ttft_ms": 10.0, "tpot_ms": 1.0, "tpot_p95_ms": 2.0}))
        pareto = dec.point_pareto(registry, objectives=("ttft_ms", "tpot_ms", "device_peak_bytes"))
        matrix = dec.recommendation_matrix(registry, pareto=pareto)
        classes = {row["candidate_id"]: row["recommendation"] for row in matrix["rows"]}
        assert "bad" in classes

    def test_release_bundle_has_hash(self):
        manifest = dec.release_bundle_manifest(
            run_id="r", artifacts=[{"canonical_hash": "c"}], policies=[{"name": "p"}],
            compatibility_matrix_hash="cm", result_registry_hash="rr",
        )
        assert manifest["bundle_hash"]
        assert manifest["kind"] == "hqsb.s05.release_bundle"

    def test_regression_thresholds_include_baselines(self):
        report = dec.regression_thresholds(
            quality_metrics={"ppl": {"baseline": 10.0}},
            memory_metrics={"device": 1000},
            performance_metrics={"tpot": 1.0},
        )
        assert report["quality"]["ppl"]["baseline"] == 10.0

    def test_figure_points_keep_row_index(self):
        report = dec.figure_points_from_table(
            [{"run_id": "r", "x": 1.0, "y": 2.0, "candidate_id": "a"}],
            x_key="x", y_key="y",
        )
        assert report["points"][0]["row_index"] == 0
        assert report["points"][0]["x"] == 1.0
