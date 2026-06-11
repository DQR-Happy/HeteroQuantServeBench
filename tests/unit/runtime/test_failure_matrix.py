"""E07-09 failure matrix, cancel timelines, OOM policy and recovery checks."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import failure as F
from hqsb.runtime import kv


@pytest.mark.unit
class TestFailureMatrix:
    def test_every_documented_case_is_present(self):
        matrix = F.frozen_matrix()
        assert {case.case_id for case in matrix} == set(F.ALL_FAILURE_CASES)
        assert len(matrix) == len(F.ALL_FAILURE_CASES)

    def test_every_case_declares_an_action_and_an_output_policy(self):
        for case in F.frozen_matrix():
            assert case.expected_action in F.EXPECTED_ACTIONS
            assert case.extra_output_policy in F.EXTRA_OUTPUT_POLICIES
            assert case.invariants == F.COMMON_INVARIANTS

    def test_unknown_case_id_refused(self):
        with pytest.raises(ConfigError):
            F.FailureCase(
                case_id="cancel_everything",
                category="request_control",
                injection_point="nowhere",
                expected_action="cancel_and_release",
                extra_output_policy="no_extra_token",
            )

    def test_unknown_expected_action_refused(self):
        with pytest.raises(ConfigError):
            F.FailureCase(
                case_id="cancel_waiting",
                category="request_control",
                injection_point="x",
                expected_action="retry_forever",
                extra_output_policy="no_extra_token",
            )

    def test_over_long_context_is_rejected_before_allocation(self):
        cases = {case.case_id: case for case in F.frozen_matrix()}
        for case_id in (
            "context_beyond_model_max",
            "context_beyond_runtime_max",
            "context_beyond_kv_capacity",
        ):
            assert cases[case_id].expected_action == "reject_before_allocation"

    def test_client_stops_consuming_is_a_cancel_case(self):
        cases = {case.case_id: case for case in F.frozen_matrix()}
        assert cases["client_stops_consuming"].expected_action == "cancel_and_release"


@pytest.mark.unit
class TestCancelTimeline:
    def _timeline(self, **overrides) -> F.CancelTimeline:
        payload = {
            "request_id": "r0",
            "t_cancel_requested_ns": 0,
            "t_cancel_seen_scheduler_ns": 1_000_000,
            "t_last_kernel_for_request_ns": 2_000_000,
            "t_last_token_ns": 2_000_000,
            "t_blocks_released_or_cached_ns": 3_000_000,
            "t_cleanup_done_ns": 4_000_000,
        }
        payload.update(overrides)
        return F.CancelTimeline(**payload)

    def test_lag_metrics_are_derived(self):
        timeline = self._timeline()
        assert timeline.observation_latency_ms == pytest.approx(1.0)
        assert timeline.block_release_lag_ms == pytest.approx(2.0)
        assert timeline.cleanup_time_ms == pytest.approx(4.0)

    def test_non_monotonic_timeline_refused(self):
        with pytest.raises(ConfigError):
            self._timeline(t_cleanup_done_ns=0)

    def test_extra_tokens_violate_a_no_extra_token_contract(self):
        with pytest.raises(ConfigError):
            self._timeline(extra_tokens_emitted=1)

    def test_in_flight_tokens_are_allowed_when_declared(self):
        timeline = self._timeline(
            extra_tokens_emitted=1, extra_output_policy="in_flight_allowed_discarded"
        )
        assert timeline.as_dict()["allowed_extra_tokens"] is None


@pytest.mark.unit
class TestOomPolicy:
    def test_oom_actions_are_bounded(self):
        assert F.oom_sequence("actual_leak") == ["fail"]
        sequence = F.oom_sequence("execution_oom")
        assert sequence[-1] == "fail"
        assert len(sequence) <= kv.MAX_OOM_ATTEMPTS

    def test_attempt_budget_cannot_exceed_the_code_bound(self):
        with pytest.raises(ConfigError):
            F.oom_sequence("execution_oom", max_attempts=kv.MAX_OOM_ATTEMPTS + 1)

    def test_unknown_kind_refused(self):
        with pytest.raises(ConfigError):
            F.oom_sequence("mystery")

    def test_prefix_pressure_evicts_once_then_fails(self):
        assert F.oom_sequence("prefix_cache_pressure") == ["evict_eligible_cache", "fail"]


@pytest.mark.unit
class TestContextAbuse:
    def test_early_rejection_within_budget_passes(self):
        check = F.ContextAbuseCheck(
            requested_tokens=100000,
            rejected_at_layer="model_max",
            tokens_allocated_before_reject=0,
            kv_capacity_tokens=16000,
        )
        assert check.ok

    def test_late_rejection_fails(self):
        check = F.ContextAbuseCheck(
            requested_tokens=100000,
            rejected_at_layer="admission",
            tokens_allocated_before_reject=15000,
            kv_capacity_tokens=16000,
        )
        assert not check.ok

    def test_missing_rejection_layer_fails(self):
        check = F.ContextAbuseCheck(
            requested_tokens=10,
            rejected_at_layer="",
            tokens_allocated_before_reject=0,
            kv_capacity_tokens=16000,
        )
        assert not check.ok

    def test_layer_limits_use_the_earliest_bound(self):
        report = F.context_limit_layers(
            requested_tokens=900, model_max=4096, runtime_max=1024, kv_capacity_tokens=512
        )
        assert report["rejects"]
        assert report["earliest_rejecting_layer"] == "kv_capacity_tokens"


@pytest.mark.unit
class TestRecoveryAndLongRun:
    def test_healthy_probe_requires_several_rounds(self):
        calls = {"count": 0}

        def probe() -> bool:
            calls["count"] += 1
            return True

        report = F.healthy_request_probe(probe, rounds=3)
        assert report["ok"]
        assert calls["count"] == 3

    def test_a_failing_round_blocks_recovery(self):
        results = iter([True, False, True])
        report = F.healthy_request_probe(lambda: next(results), rounds=3)
        assert not report["ok"]

    def test_raising_probe_counts_as_failure(self):
        def boom() -> bool:
            raise RuntimeError("unhealthy")

        assert not F.healthy_request_probe(boom, rounds=1)["ok"]

    def test_segments_separate_warmup_from_steady(self):
        slopes = F.resource_slope_report(
            {"kv_blocks": [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]},
            warmup_cycles=3,
            tolerance=0.1,
        )
        assert not slopes.growing
        assert F.leak_blocks_pass(slopes)["pass_allowed"]

    def test_growing_steady_segment_blocks_a_pass(self):
        slopes = F.resource_slope_report(
            {"kv_blocks": [1.0, 1.0, 2.0, 4.0, 8.0, 16.0]},
            warmup_cycles=2,
            tolerance=0.1,
        )
        assert "kv_blocks@steady" in slopes.growing
        verdict = F.leak_blocks_pass(slopes)
        assert not verdict["pass_allowed"]
        assert "kv" in verdict["reason"]

    def test_series_without_a_steady_segment_refused(self):
        with pytest.raises(ConfigError):
            F.resource_slope_report({"kv": [1.0]}, warmup_cycles=1, tolerance=1.0)

    def test_run_separation_forbids_mixing_modes(self):
        plan = F.run_separation_plan()
        assert "profiler" in plan and "sanitizer" in plan
        assert any("latency" in rule for rule in plan["rules"])


@pytest.mark.unit
class TestOutcomeTable:
    def _outcome(self, **overrides) -> F.FailureOutcome:
        payload = {
            "case_id": "cancel_during_decode",
            "expected_action": "cancel_and_release",
            "observed_action": "cancel_and_release",
            "bounded": True,
            "extra_tokens_after_request": 0,
            "cleanup_ms": 12.0,
            "other_request_impact_ms": 0.5,
            "resources_released": True,
            "recovered": True,
        }
        payload.update(overrides)
        return F.FailureOutcome(**payload)

    def test_matching_outcome_passes(self):
        assert self._outcome().ok

    def test_action_mismatch_fails(self):
        assert not self._outcome(observed_action="fail_request").ok

    def test_extra_tokens_fail_the_case(self):
        assert not self._outcome(extra_tokens_after_request=1).ok

    def test_matrix_requires_every_case_exactly_once(self):
        table = F.failure_matrix_table([self._outcome()])
        assert not table["ok"]
        assert table["missing"]
        assert table["failures"] == []

    def test_duplicate_cases_are_reported(self):
        table = F.failure_matrix_table([self._outcome(), self._outcome()])
        assert table["duplicates"] == ["cancel_during_decode"]
