"""Unit tests for the E02-09 hotspot decision primitives.

These tests are the *implementation* check of the formulas E02-09 reports:
the Amdahl table is compared against hand-computed numbers, the illegal
inputs the protocol demands be refused are asserted to raise, and the
Roofline/scoring helpers are exercised on synthetic numbers so the report can
claim "verified by an independent oracle" rather than "looks right".
"""

from __future__ import annotations

import pytest

from hqsb.benchmark.hotspot_decision import (
    AMDAHL_ORACLE_EXPECTED,
    DEFAULT_SCORE_WEIGHTS,
    GUARD_BAND,
    NOMINAL_ENVELOPE,
    S_SCENARIOS,
    amdahl_ceiling,
    amdahl_oracle_check,
    amdahl_oracle_table,
    candidate_dimensions,
    classify_roofline,
    combine_amdahl,
    decision_record,
    request_share_bound,
    roofline_point,
    s03_protocol,
    shape_weighted_speedup,
    validate_fractions,
    weight_sensitivity,
    weighted_total_scores,
)
from hqsb.benchmark.roofline import amdahl_speedup
from hqsb.core.errors import ConfigError


def _candidate(
    name: str,
    *,
    share_high: float,
    neutral: float = 2.0,
    risk: str = "low",
    reuse: str = "high",
    coverage: float = 1.0,
) -> dict:
    return candidate_dimensions(
        name=name,
        phase="decode",
        share_low=share_high / 2,
        share_high=share_high,
        call_count=100,
        shapes=["m=1,n=2048,k=2048"],
        bottleneck="synthetic",
        expected_speedup={"conservative": 1.25, "neutral": neutral, "optimistic": 4.0},
        ceiling=amdahl_ceiling(share_high),
        reference_complexity="low",
        integration_risk=risk,
        library_baseline="synthetic",
        reuse_value=reuse,
        workload_coverage=coverage,
        evidence=["synthetic"],
    )


@pytest.mark.unit
class TestAmdahlOracle:
    def test_oracle_passes(self):
        result = amdahl_oracle_check()
        assert result["passed"] is True
        assert result["mismatches"] == []

    def test_table_matches_hand_computed_values(self):
        table = amdahl_oracle_table()
        rows = {row["f"]: row for row in table["rows"]}
        assert rows[0.01]["s2"] == pytest.approx(1.00503, rel=5e-4)
        assert rows[0.05]["s4"] == pytest.approx(1.03896, rel=5e-4)
        assert rows[0.20]["s_infinite"] == pytest.approx(1.25, rel=5e-4)
        assert table["extra"]["speedup"] == pytest.approx(1.04167, rel=5e-4)

    def test_oracle_expectations_cover_the_documented_table(self):
        assert len(AMDAHL_ORACLE_EXPECTED) == 10

    def test_ceiling_matches_single_implementation(self):
        for fraction in (0.0, 0.01, 0.2, 0.9):
            assert amdahl_ceiling(fraction) == pytest.approx(1.0 / (1.0 - fraction))

    def test_zero_fraction_and_unit_speedup(self):
        assert amdahl_speedup(0.0, 100.0) == pytest.approx(1.0)
        assert amdahl_speedup(0.3, 1.0) == pytest.approx(1.0)


@pytest.mark.unit
class TestValidateFractions:
    def test_sum_returned(self):
        assert validate_fractions([0.2, 0.3, 0.1]) == pytest.approx(0.6)

    def test_negative_rejected(self):
        with pytest.raises(ConfigError):
            validate_fractions([-0.01])

    def test_greater_than_one_rejected(self):
        with pytest.raises(ConfigError):
            validate_fractions([1.5])

    def test_overlapping_partition_rejected(self):
        with pytest.raises(ConfigError):
            validate_fractions([0.7, 0.6])


@pytest.mark.unit
class TestCombineAmdahl:
    def test_single_fraction_equals_plain_amdahl(self):
        combined = combine_amdahl([(0.2, 4.0)])
        assert combined["speedup"] == pytest.approx(amdahl_speedup(0.2, 4.0))
        assert combined["untouched_fraction"] == pytest.approx(0.8)

    def test_multiple_fractions_use_one_baseline(self):
        combined = combine_amdahl([(0.2, 2.0), (0.1, 4.0)])
        expected = 1.0 / ((1.0 - 0.3) + 0.2 / 2.0 + 0.1 / 4.0)
        assert combined["speedup"] == pytest.approx(expected)
        assert combined["covered_fraction"] == pytest.approx(0.3)

    def test_speedup_below_one_rejected(self):
        with pytest.raises(ConfigError):
            combine_amdahl([(0.2, 0.9)])

    def test_overlapping_fractions_rejected(self):
        with pytest.raises(ConfigError):
            combine_amdahl([(0.6, 2.0), (0.5, 2.0)])

    def test_empty_input_is_identity(self):
        assert combine_amdahl([])["speedup"] == pytest.approx(1.0)


@pytest.mark.unit
class TestRequestShareBound:
    def test_product(self):
        assert request_share_bound(0.3, 0.5) == pytest.approx(0.15)

    def test_bad_share_rejected(self):
        with pytest.raises(ConfigError):
            request_share_bound(1.2, 0.5)

    def test_bad_weight_rejected(self):
        with pytest.raises(ConfigError):
            request_share_bound(0.5, -0.1)


@pytest.mark.unit
class TestShapeWeightedSpeedup:
    def test_weighting_follows_call_counts(self):
        result = shape_weighted_speedup(
            calls=[1000, 1],
            t_old_us=[10.0, 1000.0],
            t_new_us=[5.0, 1000.0],
        )
        # The one expensive call is not accelerated, so the weighted speedup is
        # far from the arithmetic mean of the two shape speedups.
        assert result["t_old_target_us"] == pytest.approx(11000.0)
        assert result["t_new_target_us"] == pytest.approx(6000.0)
        assert result["s_weighted"] == pytest.approx(11000.0 / 6000.0)

    def test_all_shapes_accelerated(self):
        result = shape_weighted_speedup([2, 3], [10.0, 20.0], [5.0, 5.0])
        assert result["s_weighted"] == pytest.approx((20.0 + 60.0) / (10.0 + 15.0))

    def test_length_mismatch_rejected(self):
        with pytest.raises(ConfigError):
            shape_weighted_speedup([1, 2], [1.0], [1.0, 1.0])

    def test_empty_rejected(self):
        with pytest.raises(ConfigError):
            shape_weighted_speedup([], [], [])

    def test_non_positive_latency_rejected(self):
        with pytest.raises(ConfigError):
            shape_weighted_speedup([1], [0.0], [1.0])

    def test_non_positive_calls_rejected(self):
        with pytest.raises(ConfigError):
            shape_weighted_speedup([0], [1.0], [1.0])


@pytest.mark.unit
class TestRooflinePoint:
    def test_bandwidth_bound_classification(self):
        point = roofline_point(
            bytes_moved=1e6,
            duration_us=1000.0,
            measured_memory_throughput_pct=90.0,
            measured_compute_throughput_pct=20.0,
        )
        assert point["classification"] == "throughput_memory_limited"
        assert point["achieved_bytes_per_s"] == pytest.approx(1e9)

    def test_compute_bound_classification(self):
        point = roofline_point(
            bytes_moved=1e6,
            duration_us=1000.0,
            useful_flops=1e12,
            measured_memory_throughput_pct=30.0,
            measured_compute_throughput_pct=88.0,
        )
        assert point["classification"] == "throughput_compute_limited"
        assert point["arithmetic_intensity_flop_per_byte"] == pytest.approx(1e6)

    def test_launch_limited_when_neither_roof_is_touched(self):
        point = roofline_point(
            bytes_moved=1e6,
            duration_us=1000.0,
            measured_memory_throughput_pct=17.3,
            measured_compute_throughput_pct=29.9,
        )
        assert point["classification"] == "launch_or_latency_limited"

    def test_insufficient_evidence_without_percentages(self):
        assert (
            classify_roofline(
                arithmetic_intensity=1.0,
                measured_memory_throughput_pct=None,
                measured_compute_throughput_pct=None,
            )
            == "insufficient_evidence"
        )

    def test_no_flops_model_is_reported_not_invented(self):
        point = roofline_point(bytes_moved=1e6, duration_us=1000.0)
        assert point["useful_flops"] is None
        assert point["arithmetic_intensity_flop_per_byte"] is None
        assert "arithmetic_intensity_note" in point

    def test_memory_level_is_preserved(self):
        point = roofline_point(bytes_moved=1e6, duration_us=1000.0, memory_level="l2")
        assert point["memory_level"] == "l2"

    def test_negative_inputs_rejected(self):
        with pytest.raises(ConfigError):
            roofline_point(bytes_moved=1e6, duration_us=0.0)
        with pytest.raises(ConfigError):
            roofline_point(bytes_moved=-1.0, duration_us=1.0)

    def test_nominal_envelope_is_labelled(self):
        assert NOMINAL_ENVELOPE["peak_fp16_flops"] == pytest.approx(67e12)
        assert "not measured" in NOMINAL_ENVELOPE["sources"]["peak_fp16_flops"]


@pytest.mark.unit
class TestCandidateDimensions:
    def test_basic_record(self):
        record = _candidate("x", share_high=0.3)
        assert record["amdahl_ceiling"] == pytest.approx(1.0 / 0.7)
        assert record["bottleneck"] == "synthetic"

    def test_inverted_share_range_rejected(self):
        with pytest.raises(ConfigError):
            candidate_dimensions(
                name="x",
                phase="decode",
                share_low=0.5,
                share_high=0.1,
                call_count=1,
                shapes=["s"],
                bottleneck="b",
                expected_speedup={"neutral": 2.0},
                ceiling=1.1,
                reference_complexity="low",
                integration_risk="low",
                library_baseline="lib",
                reuse_value="high",
                workload_coverage=1.0,
                evidence=["e"],
            )

    def test_coverage_out_of_range_rejected(self):
        with pytest.raises(ConfigError):
            candidate_dimensions(
                name="x",
                phase="decode",
                share_low=0.1,
                share_high=0.2,
                call_count=1,
                shapes=["s"],
                bottleneck="b",
                expected_speedup={"neutral": 2.0},
                ceiling=1.25,
                reference_complexity="low",
                integration_risk="low",
                library_baseline="lib",
                reuse_value="high",
                workload_coverage=1.5,
                evidence=["e"],
            )

    def test_ceiling_below_one_rejected(self):
        with pytest.raises(ConfigError):
            candidate_dimensions(
                name="x",
                phase="decode",
                share_low=0.1,
                share_high=0.2,
                call_count=1,
                shapes=["s"],
                bottleneck="b",
                expected_speedup={"neutral": 2.0},
                ceiling=0.9,
                reference_complexity="low",
                integration_risk="low",
                library_baseline="lib",
                reuse_value="high",
                workload_coverage=1.0,
                evidence=["e"],
            )


@pytest.mark.unit
class TestScoring:
    def test_weights_are_normalised(self):
        result = weighted_total_scores([_candidate("a", share_high=0.3)])
        assert sum(result["weights"].values()) == pytest.approx(
            sum(DEFAULT_SCORE_WEIGHTS.values())
        )

    def test_totals_are_bounded(self):
        result = weighted_total_scores(
            [_candidate("a", share_high=0.9), _candidate("b", share_high=0.01)]
        )
        assert 0.0 <= result["ranking"][-1]["total"] <= result["ranking"][0]["total"] <= 1.0

    def test_unknown_weight_dimension_rejected(self):
        with pytest.raises(ConfigError):
            weighted_total_scores(
                [_candidate("a", share_high=0.3)], weights={"not_a_dimension": 1.0}
            )

    def test_sensitivity_detects_a_flip(self):
        strong_share = _candidate("share", share_high=0.9, risk="high", reuse="low")
        strong_safe = _candidate("safe", share_high=0.2, risk="low", reuse="high")
        result = weight_sensitivity(
            [strong_share, strong_safe],
            variants=[
                {"evidence_share": 0.9, "expected_end_to_end_gain": 0.1},
                {"safe_reintegration_probability": 0.9, "evidence_share": 0.05},
            ],
        )
        assert result["top_flips"] is True
        assert "close" in result["verdict"]

    def test_sensitivity_stable_when_one_candidate_dominates(self):
        dominant = _candidate("dominant", share_high=0.95, neutral=4.0)
        weak = _candidate("weak", share_high=0.01, neutral=1.1, risk="high", reuse="low")
        result = weight_sensitivity([dominant, weak])
        assert result["top_flips"] is False
        assert result["base_top"] == "dominant"


@pytest.mark.unit
class TestS03Protocol:
    def test_guard_band_exceeds_the_measured_noise(self):
        assert GUARD_BAND["micro_min_relative_improvement"] > GUARD_BAND["noise_floor_relative"]
        assert GUARD_BAND["independent_processes_min"] >= 3

    def test_protocol_sections_present(self):
        protocol = s03_protocol()
        assert set(protocol) >= {
            "correctness",
            "performance",
            "dispatcher",
            "stop_criteria",
            "pass_negative_scope",
        }
        assert protocol["correctness"]["hard_gates"]
        assert protocol["correctness"]["correctness_never_tradeable_for_speed"] is True

    def test_two_lines_have_distinct_tolerance_classes(self):
        correctness = s03_protocol()["correctness"]
        assert correctness["rmsnorm"]["tolerance_ceiling"]["l2rel"] != (
            correctness["second_hotspot"]["tolerance_ceiling"]["l2rel"]
        )

    def test_stop_criteria_mention_pass_negative_scope(self):
        protocol = s03_protocol()
        assert any("PASS_NEGATIVE" in item for item in protocol["stop_criteria"])
        assert "FAIL" in protocol["pass_negative_scope"]


@pytest.mark.unit
class TestDecisionRecord:
    def _entry(self, name: str) -> dict:
        return {
            "name": name,
            "phase": "decode",
            "share_range": [0.1, 0.3],
            "amdahl_ceiling": 1.43,
            "route": "synthetic",
            "stop_criteria": ["stop"],
        }

    def test_record_fields(self):
        record = decision_record(
            selected=[self._entry("a")],
            deferred=[{"name": "b", "reason": "out of scope"}],
            rank_one_handling={"handling": "library"},
            provenance={"input_count": 1},
        )
        assert record["selected"][0]["name"] == "a"
        assert record["deferred_or_not_selected"][0]["reason"] == "out of scope"
        assert record["provenance"]["input_count"] == 1
        assert "re-profile" in record["revisitable"]

    def test_empty_selection_rejected(self):
        with pytest.raises(ConfigError):
            decision_record(
                selected=[],
                deferred=[],
                rank_one_handling={},
                provenance={},
            )

    def test_missing_selected_field_rejected(self):
        broken = self._entry("a")
        broken.pop("stop_criteria")
        with pytest.raises(ConfigError):
            decision_record(
                selected=[broken],
                deferred=[],
                rank_one_handling={},
                provenance={},
            )


@pytest.mark.unit
class TestScenarioTable:
    def test_every_class_has_ordered_scenarios(self):
        for name, payload in S_SCENARIOS.items():
            assert payload["conservative"] <= payload["neutral"] <= payload["optimistic"], name
            assert payload["source"], name

    def test_library_route_claims_less_than_a_hand_written_kernel(self):
        assert S_SCENARIOS["gemm_library"]["optimistic"] < S_SCENARIOS["attention_softmax"]["optimistic"]
