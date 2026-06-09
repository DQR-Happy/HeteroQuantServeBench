"""E07-04 scheduling: static / continuous / chunked prefill, budgets, fairness."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import scheduler as S


def _spec(**overrides) -> S.SchedulerSpec:
    payload = {
        "mode": S.CONTINUOUS,
        "max_batched_tokens": 64,
        "max_sequences": 4,
        "block_size": 16,
        "kv_blocks": 64,
        "watermark": 1.0,
    }
    payload.update(overrides)
    return S.SchedulerSpec(**payload)


def _trace(count: int = 3, prompt: int = 32, new_tokens: int = 8) -> S.RequestTrace:
    return S.homogeneous_trace(
        "t", count=count, prompt_tokens=prompt, max_new_tokens=new_tokens
    )


@pytest.mark.unit
class TestSchedulerSpec:
    def test_chunked_mode_requires_a_chunk_size(self):
        with pytest.raises(ConfigError):
            _spec(mode=S.CHUNKED_PREFILL, chunk_size=0)

    def test_chunk_size_outside_chunked_mode_refused(self):
        with pytest.raises(ConfigError):
            _spec(mode=S.CONTINUOUS, chunk_size=16)

    def test_unknown_mode_refused(self):
        with pytest.raises(ConfigError):
            _spec(mode="speculative")

    def test_token_capacity_applies_the_watermark(self):
        spec = _spec(kv_blocks=10, block_size=16, watermark=0.5)
        assert spec.kv_token_capacity == 80


@pytest.mark.unit
class TestTraceConstruction:
    def test_trace_hash_is_stable_and_order_sensitive(self):
        left = _trace()
        right = _trace()
        assert left.trace_hash == right.trace_hash
        reordered = S.RequestTrace(name="t", requests=tuple(reversed(left.requests)))
        assert reordered.trace_hash != left.trace_hash

    def test_duplicate_request_ids_refused(self):
        request = S.SimRequest(request_id="dup", prompt_tokens=4, max_new_tokens=1)
        with pytest.raises(ConfigError):
            S.RequestTrace(name="dup", requests=(request, request))

    def test_shared_prefix_longer_than_prompt_refused(self):
        with pytest.raises(ConfigError):
            S.SimRequest(
                request_id="r", prompt_tokens=8, max_new_tokens=1, shared_prefix_tokens=9
            )

    def test_mixed_trace_respects_the_requested_order(self):
        long_first = S.mixed_trace(
            "m",
            short_count=1,
            long_count=1,
            short_prompt=8,
            long_prompt=32,
            max_new_tokens=2,
            long_first=True,
        )
        assert long_first.requests[0].prompt_tokens == 32
        short_first = S.mixed_trace(
            "m",
            short_count=1,
            long_count=1,
            short_prompt=8,
            long_prompt=32,
            max_new_tokens=2,
            long_first=False,
        )
        assert short_first.requests[0].prompt_tokens == 8

    def test_staggered_trace_offsets_submits(self):
        base = _trace(count=3)
        staggered = S.staggered_trace("s", requests=base.requests, spacing=2)
        assert [r.submit_iteration for r in staggered.requests] == [0, 2, 4]


@pytest.mark.unit
class TestSimulationInvariants:
    def test_continuous_run_satisfies_token_conservation(self):
        result = S.simulate(_spec(), _trace(), max_iterations=256)
        assert result.conservation()["ok"]
        assert result.makespan_iterations > 0

    def test_chunked_prefill_satisfies_token_conservation(self):
        spec = _spec(mode=S.CHUNKED_PREFILL, chunk_size=8, max_batched_tokens=24)
        result = S.simulate(spec, _trace(prompt=64, new_tokens=4), max_iterations=256)
        assert result.conservation()["ok"]

    def test_static_mode_records_padding_slots(self):
        spec = _spec(mode=S.STATIC)
        trace = S.RequestTrace(
            name="static",
            requests=(
                S.SimRequest(request_id="short", prompt_tokens=4, max_new_tokens=1),
                S.SimRequest(request_id="long", prompt_tokens=4, max_new_tokens=8),
            ),
        )
        result = S.simulate(spec, trace, max_iterations=128)
        assert result.padding_slots > 0

    def test_every_request_reaches_a_terminal_state(self):
        result = S.simulate(_spec(), _trace(count=4), max_iterations=512)
        for outcome in result.outcomes:
            assert outcome.finish_iteration >= 0 or outcome.cancelled

    def test_kv_capacity_is_respected(self):
        spec = _spec(kv_blocks=4, block_size=16, max_sequences=8)
        result = S.simulate(spec, _trace(count=4, prompt=32, new_tokens=4), max_iterations=512)
        assert result.kv_peak_blocks <= spec.kv_blocks

    def test_preemption_counts_recomputed_positions(self):
        # 6 blocks × 16 slots = 96 slots; each request reserves 32 + 8 = 40 tokens
        # (3 blocks), so only two requests fit and the third forces a preemption.
        spec = _spec(
            kv_blocks=6,
            block_size=16,
            max_sequences=8,
            preemption_policy="recompute_longest",
        )
        result = S.simulate(spec, _trace(count=4, prompt=32, new_tokens=8), max_iterations=512)
        assert result.conservation()["ok"]
        assert any(outcome.preemptions > 0 for outcome in result.outcomes)
        assert result.kv_peak_blocks <= spec.kv_blocks

    def test_full_isl_reservation_prevents_over_admission(self):
        # Without the reservation policy a runtime over-admits and thrashes; with it
        # the run is refused instead of being admitted and preempted.
        spec = _spec(
            kv_blocks=3,
            block_size=16,
            max_sequences=8,
            admission_reserve_full_isl=True,
        )
        result = S.simulate(spec, _trace(count=4, prompt=32, new_tokens=8), max_iterations=512)
        assert result.max_concurrent_running <= 1

    def test_cancel_marks_the_request_as_cancelled(self):
        trace = S.RequestTrace(
            name="cancel",
            requests=(
                S.SimRequest(
                    request_id="r0",
                    prompt_tokens=64,
                    max_new_tokens=16,
                    cancel_at_iteration=1,
                ),
            ),
        )
        result = S.simulate(_spec(), trace, max_iterations=64)
        assert result.outcomes[0].cancelled
        assert result.conservation()["ok"]

    def test_simulation_payload_is_labelled_simulated(self):
        payload = S.simulate(_spec(), _trace(count=1), max_iterations=64).as_dict()
        assert payload["simulated"] is True
        assert payload["conservation"]["ok"] is True


@pytest.mark.unit
class TestChunksAndFairness:
    def test_chunks_tile_the_prompt_exactly_once(self):
        assert S.chunk_boundaries(10, 4) == ((0, 4), (4, 8), (8, 10))
        audit = S.chunk_coverage_audit(10, 4, [(0, 4), (4, 4), (8, 2)])
        assert audit["ok"]

    def test_chunk_gap_is_detected(self):
        audit = S.chunk_coverage_audit(10, 4, [(0, 4), (5, 5)])
        assert not audit["ok"]

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ConfigError):
            S.chunk_boundaries(10, 0)

    def test_jain_index_is_one_for_uniform_values(self):
        assert S.fairness_jain([2.0, 2.0, 2.0]) == pytest.approx(1.0)

    def test_jain_index_drops_for_skewed_values(self):
        assert S.fairness_jain([10.0, 0.0, 0.0]) < 0.5

    def test_jain_index_rejects_negative_values(self):
        with pytest.raises(ConfigError):
            S.fairness_jain([1.0, -1.0])


@pytest.mark.unit
class TestCongestionAndCurves:
    def test_congestion_detection_reports_a_reason(self):
        spec = _spec(max_batched_tokens=8, max_sequences=1)
        trace = S.mixed_trace(
            "c",
            short_count=4,
            long_count=2,
            short_prompt=32,
            long_prompt=128,
            max_new_tokens=8,
        )
        result = S.simulate(spec, trace, max_iterations=256)
        report = S.detect_congestion(result)
        assert report.congested
        assert report.reasons

    def test_strategy_curve_needs_two_configurations(self):
        with pytest.raises(ConfigError):
            S.strategy_curve([_spec()], _trace())

    def test_strategy_curve_labels_every_row_simulated(self):
        rows = S.strategy_curve(
            [_spec(), _spec(mode=S.STATIC, chunk_size=0)], _trace(count=2), max_iterations=128
        )
        assert len(rows) == 2
        assert all(row["simulated"] for row in rows)

    def test_outcome_reports_queue_and_wait_iterations(self):
        result = S.simulate(
            _spec(max_sequences=1), _trace(count=2, prompt=16, new_tokens=2), max_iterations=128
        )
        waits = [outcome.wait_iterations for outcome in result.outcomes]
        assert all(wait >= 0 for wait in waits)
        assert max(waits) > 0
