"""S07 time metrics, token denominators and paired statistics."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import metrics


def _timeline(**overrides) -> metrics.RequestTimeline:
    payload = {
        "t_submit": 0.0,
        "t_prefill_start": 0.1,
        "t_prefill_end": 0.3,
        "t_first_token_ready": 0.4,
        "t_final_token_ready": 1.4,
        "output_tokens": 3,
        "token_ready_times": (0.4, 0.9, 1.4),
        "logical_input_tokens": 32,
    }
    payload.update(overrides)
    return metrics.RequestTimeline(**payload)


@pytest.mark.unit
class TestTimeline:
    def test_derived_metrics_follow_the_documented_formulas(self):
        timeline = _timeline()
        assert timeline.queue_delay == pytest.approx(0.1)
        assert timeline.runtime_ttft == pytest.approx(0.4)
        assert timeline.prefill_service == pytest.approx(0.3)
        assert timeline.e2e_core == pytest.approx(1.4)
        assert timeline.tpot == pytest.approx(0.5)
        assert timeline.itl_ms == pytest.approx([500.0, 500.0], abs=1e-6)

    def test_non_monotonic_timeline_refused(self):
        with pytest.raises(ConfigError):
            _timeline(t_final_token_ready=0.2)

    def test_missing_first_token_refused_when_tokens_exist(self):
        with pytest.raises(ConfigError):
            _timeline(t_first_token_ready=None)

    def test_token_ready_times_must_match_output_tokens(self):
        with pytest.raises(ConfigError):
            _timeline(token_ready_times=(0.4, 1.4))

    def test_missing_stamp_is_reported_not_defaulted(self):
        timeline = _timeline(t_prefill_start=None)
        with pytest.raises(ConfigError):
            timeline.queue_delay

    def test_cancel_latency_and_cleanup_are_optional(self):
        timeline = _timeline(
            t_cancel_requested=1.5,
            t_cancel_observed=1.6,
            t_cleanup_done=1.7,
            cancelled=True,
        )
        assert timeline.cancel_observation_latency == pytest.approx(0.1)
        assert timeline.cleanup_time == pytest.approx(0.1)
        assert _timeline().cancel_observation_latency is None

    def test_completed_request_measures_cleanup_from_the_final_token(self):
        timeline = _timeline(t_cleanup_done=1.5)
        assert timeline.cleanup_time == pytest.approx(0.1)


@pytest.mark.unit
class TestTokenLedger:
    def test_three_denominators_are_distinct(self):
        ledger = metrics.TokenLedger(
            logical_input_tokens=64,
            committed_output_tokens=16,
            cached_prefix_tokens=32,
            speculative_verified_positions=2,
            preemption_recompute_positions=4,
            cancelled_or_failed_positions=8,
        )
        assert ledger.useful_committed_tokens == 80
        # 32 fresh prefill + 16 committed decode + 2 spec + 4 recompute + 8 cancelled
        assert ledger.model_computed_positions == 62

    def test_cached_prefix_cannot_exceed_the_prompt(self):
        with pytest.raises(ConfigError):
            metrics.TokenLedger(logical_input_tokens=8, committed_output_tokens=1, cached_prefix_tokens=9)

    def test_rates_use_the_right_denominator(self):
        ledger = metrics.TokenLedger(
            logical_input_tokens=32, committed_output_tokens=8, cached_prefix_tokens=32
        )
        assert ledger.output_tps(1.0) == pytest.approx(8.0)
        assert ledger.logical_token_tps(1.0) == pytest.approx(40.0)
        assert ledger.compute_position_tps(1.0) == pytest.approx(8.0)

    def test_zero_window_refused(self):
        ledger = metrics.TokenLedger(logical_input_tokens=1, committed_output_tokens=1)
        with pytest.raises(ConfigError):
            ledger.output_tps(0.0)

    def test_acceptance_cannot_exceed_proposed_drafts(self):
        ledger = metrics.TokenLedger(
            logical_input_tokens=1,
            committed_output_tokens=1,
            speculative_verified_positions=5,
            draft_tokens_proposed=3,
        )
        assert not ledger.audit().ok

    def test_audit_require_ok_raises_on_empty_ledger(self):
        with pytest.raises(ConfigError):
            metrics.TokenLedger(logical_input_tokens=0, committed_output_tokens=0).audit().require_ok()

    def test_request_records_aggregate_including_failures(self):
        ledger = metrics.token_accounting_from_requests(
            [
                {
                    "logical_input_tokens": 16,
                    "committed_output_tokens": 4,
                    "cached_prefix_tokens": 8,
                },
                {
                    "logical_input_tokens": 32,
                    "committed_output_tokens": 0,
                    "computed_positions": 12,
                    "cancelled": True,
                },
            ]
        )
        assert ledger.logical_input_tokens == 48
        assert ledger.cancelled_or_failed_positions == 12
        assert ledger.useful_committed_tokens == 52


@pytest.mark.unit
class TestStatistics:
    def test_percentile_interpolates(self):
        assert metrics.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
        assert metrics.percentile([5.0], 0.9) == 5.0

    def test_percentile_rejects_bad_input(self):
        with pytest.raises(ConfigError):
            metrics.percentile([], 0.5)
        with pytest.raises(ConfigError):
            metrics.percentile([1.0], 1.5)

    def test_distribution_summary_reports_spread_and_tails(self):
        summary = metrics.distribution_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert summary.count == 5
        assert summary.median == pytest.approx(3.0)
        assert summary.p95 == pytest.approx(4.8)
        assert summary.spread == pytest.approx(4.0)

    def test_paired_effect_is_deterministic_for_a_seed(self):
        left = [10.0, 12.0, 11.0, 9.0, 13.0]
        right = [9.0, 11.0, 10.0, 8.0, 12.0]
        first = metrics.paired_effect(left, right, seed=7)
        second = metrics.paired_effect(left, right, seed=7)
        assert first.as_dict() == second.as_dict()
        assert first.mean_difference == pytest.approx(1.0)

    def test_paired_effect_requires_equal_lengths(self):
        with pytest.raises(ConfigError):
            metrics.paired_effect([1.0, 2.0], [1.0])

    def test_identical_series_crosses_zero(self):
        effect = metrics.paired_effect([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert effect.crosses_zero

    def test_guard_band_requires_the_whole_interval(self):
        effect = metrics.paired_effect([10.0, 11.0, 12.0], [1.0, 1.0, 1.0])
        assert effect.exceeds_guard_band(0.5)
        assert not effect.exceeds_guard_band(100.0)

    def test_equivalence_needs_a_positive_bound(self):
        effect = metrics.paired_effect([1.0], [1.0])
        with pytest.raises(ConfigError):
            metrics.EquivalenceCheck(bound=0.0, effect=effect)

    def test_equivalence_uses_the_pre_registered_bound(self):
        effect = metrics.paired_effect([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert metrics.EquivalenceCheck(bound=0.1, effect=effect).equivalent


@pytest.mark.unit
class TestRunLevelSamples:
    def test_run_medians_are_the_repetition_unit(self):
        samples = metrics.RunLevelSamples(metric="ttft_ms")
        samples.record("run-1", [10.0, 12.0, 11.0])
        samples.record("run-2", [20.0, 22.0, 21.0])
        assert samples.independent_runs() == 2
        assert samples.run_medians() == [11.0, 21.0]
        assert samples.summary().count == 2

    def test_summary_needs_at_least_one_run(self):
        with pytest.raises(ConfigError):
            metrics.RunLevelSamples(metric="ttft_ms").summary()
