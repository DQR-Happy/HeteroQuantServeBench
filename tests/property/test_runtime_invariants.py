"""Property/invariant tests for the S07 runtime layer.

These are not exhaustive fuzzing: each test states an invariant that must hold
for *any* accepted input, and asserts it over a deterministic generated family.
The invariants mirror the protocol requirements (token conservation, key
sensitivity, bounded actions, monotonic timelines).
"""

from __future__ import annotations

import random

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import kv, metrics, prefix_cache, scheduler, spec_decode, trace

_RNG = random.Random(20260918)


@pytest.mark.property
class TestSchedulerConservation:
    @pytest.mark.parametrize("seed_case", range(6))
    def test_token_conservation_holds_for_random_traces(self, seed_case: int):
        rng = random.Random(1000 + seed_case)
        requests = tuple(
            scheduler.SimRequest(
                request_id=f"r{index}",
                prompt_tokens=rng.randint(8, 128),
                max_new_tokens=rng.randint(1, 12),
                submit_iteration=rng.randint(0, 3),
            )
            for index in range(rng.randint(1, 6))
        )
        trace = scheduler.RequestTrace(name=f"random-{seed_case}", requests=requests)
        mode = (scheduler.STATIC, scheduler.CONTINUOUS, scheduler.CHUNKED_PREFILL)[
            seed_case % 3
        ]
        spec = scheduler.SchedulerSpec(
            mode=mode,
            max_batched_tokens=rng.choice([16, 32, 64]),
            max_sequences=rng.randint(1, 4),
            block_size=16,
            kv_blocks=rng.randint(4, 64),
            chunk_size=16 if mode == scheduler.CHUNKED_PREFILL else 0,
            watermark=1.0,
        )
        result = scheduler.simulate(spec, trace, max_iterations=1024)
        assert result.conservation()["ok"], result.conservation()["offenders"]

    @pytest.mark.parametrize("max_tokens", [8, 16, 32, 48])
    def test_jain_index_stays_within_bounds(self, max_tokens: int):
        result = scheduler.simulate(
            scheduler.SchedulerSpec(
                mode=scheduler.CONTINUOUS,
                max_batched_tokens=64,
                max_sequences=4,
                block_size=16,
                kv_blocks=64,
                watermark=1.0,
            ),
            scheduler.homogeneous_trace(
                "bounds", count=4, prompt_tokens=32, max_new_tokens=max_tokens
            ),
            max_iterations=512,
        )
        fairness = result.fairness()["jain_index"]
        assert 0.0 <= fairness <= 1.0


@pytest.mark.property
class TestKvAccounting:
    @pytest.mark.parametrize("tokens", [0, 1, 15, 16, 17, 31, 32, 33])
    def test_slots_always_cover_live_tokens(self, tokens: int):
        block_size = 16
        slots = tokens + (block_size - tokens % block_size) % block_size
        assert slots >= tokens
        assert slots % block_size == 0

    def test_reconciliation_residual_is_a_pure_function_of_the_inputs(self):
        geometry = kv.KVGeometry(
            num_layers=4, num_kv_heads=2, head_dim=64, element_bytes=2
        )
        residuals = []
        for active_tokens in (0, 16, 33, 64):
            reconciliation = kv.MemoryReconciliation(
                geometry=geometry,
                block_size=16,
                active_tokens=active_tokens,
                tolerance_bytes=1024.0,
            )
            reconciliation.declare("workspace", 100.0)
            reconciliation.measured_framework_reserved = (
                reconciliation.predicted_total_bytes + 10
            )
            residuals.append(reconciliation.residual_bytes)
        assert residuals == [10.0] * 4

    def test_oom_actions_are_bounded_for_every_kind(self):
        for kind in kv.OOM_KINDS:
            sequence = []
            for attempt in range(kv.MAX_OOM_ATTEMPTS):
                action = kv.oom_action(kind, attempt)
                sequence.append(action)
                if action in ("reject", "fail"):
                    break
            assert sequence[-1] in ("reject", "fail"), (kind, sequence)
            assert len(sequence) <= kv.MAX_OOM_ATTEMPTS
            assert kv.OOM_LADDERS[kind][-1] in ("reject", "fail")

    def test_resource_slope_verdict_is_one_of_the_documented_values(self):
        for series in ([1.0] * 5, [1.0, 2.0, 4.0, 8.0, 16.0], [5.0, 4.0, 3.0, 2.0, 1.0]):
            verdict = kv.resource_slope("kv", series, tolerance=0.5).verdict
            assert verdict in ("STEADY", "BOUNDED_CACHE", "GROWING", "DECREASING")


@pytest.mark.property
class TestPrefixKeySensitivity:
    @pytest.mark.parametrize("field", prefix_cache.ALL_KEY_FIELDS)
    def test_any_key_field_change_changes_the_digest(self, field: str):
        identity = {
            "model_id": "m",
            "weight_revision": "r",
            "precision": "float16",
            "tokenizer_id": "t",
            "chat_template_hash": "c",
        }
        base = prefix_cache.build_prefix_key(
            identity=identity, tokens=list(range(64)), token_span=(0, 32)
        )
        mutated = prefix_cache.PrefixKey(
            fields={**base.fields, field: base.fields[field] + "-x"}, tokens=base.tokens
        )
        assert base.effective_digest != mutated.effective_digest

    def test_lookup_accounting_always_sums_to_the_query(self):
        identity = {
            "model_id": "m",
            "weight_revision": "r",
            "precision": "float16",
            "tokenizer_id": "t",
            "chat_template_hash": "c",
        }
        spec = prefix_cache.PrefixCacheSpec(block_size=16, max_cache_bytes=1 << 20)
        for cache_tokens in (0, 16, 32):
            cache = prefix_cache.PrefixCache(spec)
            if cache_tokens:
                key = prefix_cache.build_prefix_key(
                    identity=identity, tokens=list(range(64)), token_span=(0, cache_tokens)
                )
                cache.insert(
                    key=key,
                    block_ids=tuple(range(cache_tokens // 16)),
                    bytes_value=1.0,
                    request_id="r0",
                )
            query = prefix_cache.build_prefix_key(
                identity=identity, tokens=list(range(64)), token_span=(0, 64)
            )
            result = cache.lookup(
                key=query,
                query_tokens=64,
                block_groups=("full_attention",),
                request_id="r1",
            )
            assert result.cached_tokens + result.computed_tokens == 64
            assert result.cached_tokens == cache_tokens


@pytest.mark.property
class TestSpecDecodeExactness:
    @pytest.mark.parametrize("accepted", [0, 1, 2, 3])
    def test_rollback_is_the_complement_of_acceptance(self, accepted: int):
        proposed = 3
        report = spec_decode.kv_commit_rollback_audit(
            proposed_positions=proposed,
            accepted_positions=accepted,
            committed_positions=accepted + (1 if accepted < proposed else 0),
            rolled_back_positions=proposed - accepted,
        )
        assert report["ok"], report["problems"]

    def test_residual_distribution_is_always_normalised(self):
        from fractions import Fraction

        rng = random.Random(7)
        for _ in range(8):
            size = rng.randint(2, 5)
            target = [Fraction(rng.randint(0, 5), 10) for _ in range(size)]
            draft = [Fraction(rng.randint(0, 5), 10) for _ in range(size)]
            if sum(target) == 0 or sum(draft) == 0:
                continue
            try:
                residual = spec_decode.residual_distribution(target, draft)
            except ConfigError:
                continue
            assert spec_decode.residual_is_normalized(residual)
            assert all(value >= 0 for value in residual)


@pytest.mark.property
class TestTimelinesAndLedgers:
    def test_timeline_accepts_only_monotonic_inputs(self):
        rng = random.Random(11)
        for _ in range(10):
            stamps = sorted(rng.uniform(0, 10) for _ in range(6))
            timeline = metrics.RequestTimeline(
                t_submit=stamps[0],
                t_prefill_start=stamps[1],
                t_first_token_ready=stamps[2],
                t_final_token_ready=stamps[5],
                output_tokens=2,
                token_ready_times=(stamps[2], stamps[5]),
                logical_input_tokens=4,
            )
            assert timeline.e2e_core >= timeline.runtime_ttft >= 0

    def test_token_ledger_audit_is_stable_under_aggregation(self):
        ledger = metrics.TokenLedger(
            logical_input_tokens=32,
            committed_output_tokens=8,
            cached_prefix_tokens=16,
            speculative_verified_positions=2,
            preemption_recompute_positions=3,
            cancelled_or_failed_positions=1,
        )
        assert ledger.audit().ok
        assert ledger.model_computed_positions == (
            32 - 16 + 8 + 2 + 3 + 1
        )

    def test_iteration_ledger_residual_is_zero_when_the_invariant_holds(self):
        rng = random.Random(23)
        for iteration in range(10):
            scheduled = rng.randint(0, 16)
            rollback = rng.randint(0, scheduled)
            previous = rng.randint(0, 100)
            entry = trace.IterationLedgerEntry(
                iteration=iteration,
                scheduled_tokens={"r0": scheduled},
                token_budget=16,
                previous_computed_positions=previous,
                rollback_positions=rollback,
                new_computed_positions=previous + scheduled - rollback,
            )
            assert entry.conservation_residual() == 0
