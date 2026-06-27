"""Unit tests for the E03-01 RMSNorm correctness primitives.

These are deliberately GPU-free: they pin the *numerical contract* (tolerance
table, epsilon position, NaN/Inf classification policy, metric edge cases) that
the on-device experiment in ``scripts/audit/run_e03_01_rmsnorm_semantics.py``
asserts against real kernels. If the contract logic drifts, the experiment's
verdict would silently change meaning — so it is locked here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hqsb.benchmark import rmsnorm_correctness as rc
from ops import cuda_bridge


@pytest.mark.unit
class TestToleranceTable:
    def test_fp16_uses_the_e02_09_ceiling(self):
        tol = rc.tolerance_for("fp16", 2048)
        assert tol["atol"] == pytest.approx(8.0 * 2.0 ** -11)
        assert tol["rtol"] == pytest.approx(2.0e-2)
        assert tol["l2rel"] == pytest.approx(1.0e-3)
        assert tol["cosine_min"] == pytest.approx(0.9999)

    def test_fp32_has_two_shape_classes(self):
        small = rc.tolerance_for("fp32", 2048)
        large = rc.tolerance_for("fp32", 2049)
        assert small["atol"] == pytest.approx(5.0e-4)
        assert small["shape_class"] == "small_h"
        assert large["atol"] == pytest.approx(2.0e-3)
        assert large["shape_class"] == "large_h"

    def test_reference_budget_is_half_the_gate(self):
        gate = rc.tolerance_for("fp32", 512)
        budget = rc.reference_budget("fp32", 512)
        assert budget["atol"] == pytest.approx(gate["atol"] * 0.5)
        assert budget["l2rel"] == pytest.approx(gate["l2rel"] * 0.5)

    def test_unknown_dtype_rejected(self):
        with pytest.raises(ValueError):
            rc.tolerance_for("bf16", 128)


@pytest.mark.unit
class TestSeedsAndHashing:
    def test_seed_is_deterministic_and_case_sensitive(self):
        assert rc.derive_seed("a") == rc.derive_seed("a")
        assert rc.derive_seed("a") != rc.derive_seed("b")
        assert 0 <= rc.derive_seed("a") < 2 ** 32

    def test_canonical_hash_is_key_order_independent(self):
        assert rc.sha256_hex({"a": 1, "b": 2}) == rc.sha256_hex({"b": 2, "a": 1})
        assert rc.sha256_hex({"a": 1}) != rc.sha256_hex({"a": 2})


@pytest.mark.unit
class TestGeneration:
    def test_same_seed_same_bytes(self):
        a = rc.generate_case_arrays("random_normal", 4, 8, "fp32", 7)
        b = rc.generate_case_arrays("random_normal", 4, 8, "fp32", 7)
        assert a["x"].tobytes() == b["x"].tobytes()
        assert a["w"].tobytes() == b["w"].tobytes()

    def test_different_seed_different_bytes(self):
        a = rc.generate_case_arrays("random_normal", 4, 8, "fp32", 7)
        b = rc.generate_case_arrays("random_normal", 4, 8, "fp32", 8)
        assert a["x"].tobytes() != b["x"].tobytes()

    def test_all_modes_generate_finite_shapes(self):
        for mode in rc.INPUT_MODES:
            out = rc.generate_case_arrays(mode, 3, 5, "fp32", 1)
            assert out["x"].shape == (3, 5)
            assert out["w"].shape == (5,)

    def test_zeros_and_constant_modes(self):
        zeros = rc.generate_case_arrays("zeros", 2, 4, "fp32", 1)
        assert np.all(zeros["x"] == 0.0)
        constant = rc.generate_case_arrays("constant", 2, 4, "fp32", 1)
        assert np.all(constant["x"] == np.float32(0.75))

    def test_weight_zero_mode_puts_exact_zeros_in_w(self):
        out = rc.generate_case_arrays("weight_zero", 2, 32, "fp32", 3)
        assert np.count_nonzero(out["w"] == 0.0) == 2
        assert out["w_mode"] == "zero"

    def test_weight_signed_mode_alternates_sign(self):
        out = rc.generate_case_arrays("weight_signed", 1, 8, "fp32", 3)
        assert out["w"][0] < 0 and out["w"][1] > 0

    def test_smallest_fp16_is_not_silently_zero(self):
        out = rc.generate_case_arrays("tiny", 1, 64, "fp16", 5)
        assert np.count_nonzero(out["x"] == 0.0) == 0

    def test_large_safe_square_sum_stays_finite_in_fp32(self):
        out = rc.generate_case_arrays("large_safe", 1, 8192, "fp16", 5)
        squares = out["x"].astype(np.float32) ** 2
        assert np.all(np.isfinite(squares))
        assert float(np.sum(squares, dtype=np.float64)) < 3.4e38

    def test_overflow_probe_overflows_fp32_square_only_for_fp32_input(self):
        fp32_case = rc.generate_case_arrays("overflow_probe", 1, 4, "fp32", 9)
        assert rc.fp32_square_sum_overflow(fp32_case["x"], 1, 4)[0]
        fp16_case = rc.generate_case_arrays("overflow_probe", 1, 4, "fp16", 9)
        assert not rc.fp32_square_sum_overflow(fp16_case["x"], 1, 4)[0]

    def test_injections_are_recorded_with_indices(self):
        out = rc.generate_case_arrays("nan_inject", 3, 8, "fp16", 2)
        assert len(out["injections"]) == 2
        flat = out["x"].reshape(-1)
        for entry in out["injections"]:
            assert math.isnan(float(flat[entry["flat_index"]]))
        assert out["x"][-1, -1] != out["x"][-1, -1]  # tail site is NaN too

    def test_posinf_and_neginf_injection_signs(self):
        pos = rc.generate_case_arrays("posinf", 1, 4, "fp32", 2)
        assert np.isposinf(pos["x"][0, 0])
        neg = rc.generate_case_arrays("neginf", 1, 4, "fp32", 2)
        assert np.isneginf(neg["x"][0, 0])


@pytest.mark.unit
class TestClassificationOracle:
    def test_classify_all_four_classes(self):
        values = np.array([1.0, np.inf, -np.inf, np.nan])
        classes = rc.classify(values)
        assert list(classes) == [
            rc.CLASS_FINITE, rc.CLASS_POSINF, rc.CLASS_NEGINF, rc.CLASS_NAN,
        ]

    def test_nan_row_makes_every_element_nan(self):
        x = np.zeros((2, 4), dtype=np.float32)
        x[0, 2] = np.nan
        expected = rc.expected_classes(x, 2, 4)
        assert list(expected[:4]) == [rc.CLASS_NAN] * 4
        assert list(expected[4:]) == [rc.CLASS_FINITE] * 4

    def test_inf_row_marks_inf_lanes_nan_and_rest_finite(self):
        x = np.zeros((1, 4), dtype=np.float32)
        x[0, 1] = np.inf
        expected = rc.expected_classes(x, 1, 4)
        assert list(expected) == [
            rc.CLASS_FINITE, rc.CLASS_NAN, rc.CLASS_FINITE, rc.CLASS_FINITE,
        ]

    def test_inf_row_finite_lanes_must_be_exactly_zero(self):
        x = np.zeros((2, 3), dtype=np.float32)
        x[1, 0] = -np.inf
        mask = rc.inf_row_finite_lanes_exact_zero(x, 2, 3)
        assert list(mask) == [False] * 3 + [False, True, True]

    def test_class_counts(self):
        counts = rc.class_counts(rc.classify(np.array([1.0, np.inf, np.nan])))
        assert counts == {"finite": 1, "posinf": 1, "neginf": 0, "nan": 1}


@pytest.mark.unit
class TestFp64Oracle:
    def test_matches_hand_computed_values(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        w = np.array([1.0, 1.0], dtype=np.float32)
        out = rc.fp64_oracle(x, w, 1, 2, 0.0)
        # 1/sqrt(25/2) = 0.28284271247461906
        assert out[0] == pytest.approx(3.0 * 0.28284271247461906, rel=1e-12)
        assert out[1] == pytest.approx(4.0 * 0.28284271247461906, rel=1e-12)

    def test_epsilon_is_added_inside_the_rsqrt_after_the_mean(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        w = np.array([1.0, 1.0], dtype=np.float32)
        # mean(x^2) = 12.5; eps = 12.5 -> 1/sqrt(25) = 0.2
        out = rc.fp64_oracle(x, w, 1, 2, 12.5)
        assert out[0] == pytest.approx(0.6, rel=1e-12)
        assert out[1] == pytest.approx(0.8, rel=1e-12)

    def test_weight_is_broadcast_along_the_last_axis_only(self):
        x = np.ones((2, 3), dtype=np.float32)
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = rc.fp64_oracle(x, w, 2, 3, 1e-12).reshape(2, 3)
        assert out[0] == pytest.approx(out[1])
        assert out[0] == pytest.approx([1.0, 2.0, 3.0], rel=1e-9)

    def test_inf_input_gives_zero_for_finite_lanes_and_nan_for_inf_lane(self):
        x = np.array([[1.0, np.inf]], dtype=np.float32)
        w = np.array([1.0, 1.0], dtype=np.float32)
        out = rc.fp64_oracle(x, w, 1, 2, 1e-6)
        assert out[0] == 0.0
        assert math.isnan(out[1])

    def test_nan_input_makes_the_whole_row_nan(self):
        x = np.array([[1.0, np.nan, 2.0]], dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        out = rc.fp64_oracle(x, w, 1, 3, 1e-6)
        assert all(math.isnan(v) for v in out)


@pytest.mark.unit
class TestMetrics:
    def test_perfect_match(self):
        a = np.array([1.0, 2.0, 3.0])
        metrics = rc.error_metrics(a, a)
        assert metrics["max_abs"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["cosine"] == pytest.approx(1.0)
        assert metrics["l2rel"] == 0.0

    def test_cosine_is_not_applicable_on_zero_norm(self):
        ref = np.zeros(4)
        cand = np.zeros(4)
        metrics = rc.error_metrics(cand, ref)
        assert metrics["cosine"] is None
        assert metrics["cosine_applicable"] is False
        assert metrics["l2rel"] == 0.0

    def test_l2rel_uses_the_pre_registered_floor(self):
        ref = np.zeros(1)
        cand = np.full(1, rc.L2_FLOOR_ABS / 2.0)
        metrics = rc.error_metrics(cand, ref)
        assert metrics["l2rel"] == pytest.approx(0.5, rel=1e-9)

    def test_inapplicable_when_mask_is_empty(self):
        metrics = rc.error_metrics([1.0], [1.0], np.array([False]))
        assert metrics["applicable"] is False
        assert metrics["max_abs"] is None

    def test_max_and_mean_and_rmse(self):
        ref = np.array([0.0, 0.0])
        cand = np.array([3.0, 4.0])
        metrics = rc.error_metrics(cand, ref)
        assert metrics["max_abs"] == 4.0
        assert metrics["mean_abs"] == 3.5
        assert metrics["rmse"] == pytest.approx(math.sqrt(12.5))


@pytest.mark.unit
class TestViolationsAndFirstMismatch:
    def test_mixed_criterion_accepts_relative_slack(self):
        tol = {"atol": 1e-3, "rtol": 1e-2}
        ref = np.array([1000.0])
        cand = np.array([1005.0])  # 5 abs, but 0.5% relative
        result = rc.elementwise_violations(cand, ref, tol, np.array([True]))
        assert result["passed"] is True

    def test_mixed_criterion_rejects_beyond_both_bounds(self):
        tol = {"atol": 1e-3, "rtol": 1e-2}
        ref = np.array([1.0])
        cand = np.array([1.5])
        result = rc.elementwise_violations(cand, ref, tol, np.array([True]))
        assert result["passed"] is False
        assert result["worst"]["flat_index"] == 0

    def test_first_mismatch_reports_flat_row_col(self):
        ref = np.zeros((2, 3))
        cand = np.zeros((2, 3))
        cand[1, 2] = 5.0
        out = rc.first_mismatch(
            candidate=cand,
            reference=ref,
            candidate_classes=rc.classify(cand),
            expected_class_array=rc.classify(ref),
            tolerance={"atol": 1e-3, "rtol": 1e-3},
            hidden=3,
        )
        assert out["kind"] == "numeric"
        assert out["flat_index"] == 5
        assert out["row"] == 1 and out["col"] == 2

    def test_first_mismatch_detects_a_class_mismatch(self):
        ref = np.zeros((1, 2))
        cand = np.array([[0.0, np.nan]])
        out = rc.first_mismatch(
            candidate=cand,
            reference=ref,
            candidate_classes=rc.classify(cand),
            expected_class_array=rc.classify(ref),
            tolerance={"atol": 1e-3, "rtol": 1e-3},
            hidden=2,
        )
        assert out["kind"] == "classification"
        assert out["candidate_class"] == "nan"
        assert out["expected_class"] == "finite"

    def test_first_mismatch_none_when_clean(self):
        a = np.ones((1, 2))
        assert rc.first_mismatch(
            candidate=a, reference=a,
            candidate_classes=rc.classify(a), expected_class_array=rc.classify(a),
            tolerance={"atol": 1e-3, "rtol": 1e-3}, hidden=2,
        ) is None


@pytest.mark.unit
class TestJudgeCase:
    def test_all_true_is_pass(self):
        assert rc.judge_case({"a": True, "b": True})["status"] == "PASS"

    def test_false_fails(self):
        verdict = rc.judge_case({"a": True, "b": False})
        assert verdict["status"] == "FAIL"
        assert verdict["failed_checks"] == ["b"]

    def test_undocumented_not_applicable_fails(self):
        verdict = rc.judge_case({"a": None})
        assert verdict["status"] == "FAIL"

    def test_documented_not_applicable_passes(self):
        verdict = rc.judge_case({"a": None}, {"a": "declared reason"})
        assert verdict["status"] == "PASS"


@pytest.mark.unit
class TestCoverageMatrix:
    def test_unrun_combination_is_absent_not_assumed(self):
        records = [
            {"dtype": "fp32", "mode": "zeros", "variant": "v0_shared",
             "shape_class": "s1", "status": "PASS"},
            {"dtype": "fp16", "mode": "nan_inject", "variant": "v2_vectorized",
             "shape_class": "s1", "status": "FAIL"},
        ]
        matrix = rc.coverage_matrix(records)
        assert matrix["total_cases"] == 2
        assert matrix["total_pass"] == 1
        assert matrix["total_fail"] == 1
        assert "fp32|nan_inject|v0_shared|s1" not in matrix["combinations"]
        assert matrix["per_axis"]["mode"]["zeros"]["pass"] == 1
        assert matrix["per_axis"]["mode"]["nan_inject"]["fail"] == 1


@pytest.mark.unit
class TestDeclaredDivergence:
    def test_only_the_fp32_overflow_probe_cell_diverges(self):
        assert ("fp32", "overflow_probe") in rc.DECLARED_DIVERGENCE
        assert ("fp16", "overflow_probe") not in rc.DECLARED_DIVERGENCE
        assert ("fp32", "random_normal") not in rc.DECLARED_DIVERGENCE


@pytest.mark.unit
class TestBridgeContractValidation:
    """Pure validation logic of the bridge (no CUDA call is made)."""

    def test_reason_code_set_is_frozen(self):
        assert "UNSUPPORTED_DTYPE" in cuda_bridge.RMSNORM_REASON_CODES
        assert "NON_CONTIGUOUS_INPUT" in cuda_bridge.RMSNORM_REASON_CODES

    def test_unknown_reason_code_rejected(self):
        with pytest.raises(ValueError):
            cuda_bridge.RmsNormContractError("NOT_A_CODE", "x")

    def test_variant_name_and_code_resolution(self):
        assert cuda_bridge._resolve_variant_code("v0_shared") == 2
        assert cuda_bridge._resolve_variant_code(4) == 4
        with pytest.raises(cuda_bridge.RmsNormContractError) as excinfo:
            cuda_bridge._resolve_variant_code("v9")
        assert excinfo.value.reason_code == "UNSUPPORTED_VARIANT"
        with pytest.raises(cuda_bridge.RmsNormContractError):
            cuda_bridge._resolve_variant_code(9)

    def test_reference_variant_is_encodable_so_the_c_layer_can_reject_it(self):
        # If the bridge refused it locally the C-layer rejection could never be
        # exercised; the code must survive validation and be rejected by C++.
        assert cuda_bridge._resolve_variant_code("reference") == 1
