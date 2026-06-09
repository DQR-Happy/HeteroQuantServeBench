"""E07-05 prefix cache: key binding, collision policy, refcounts, eviction."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import prefix_cache as P


IDENTITY = {
    "model_id": "Qwen/Qwen3-1.7B",
    "weight_revision": "rev-1",
    "precision": "float16",
    "quant_artifact_hash": "",
    "adapter_hash": "",
    "tokenizer_id": "Qwen/Qwen3-1.7B",
    "chat_template_hash": "tpl",
    "rope_config_hash": "rope",
    "attention_config_hash": "attn",
}


def _spec(**overrides) -> P.PrefixCacheSpec:
    payload = {"block_size": 16, "max_cache_bytes": 1 << 20}
    payload.update(overrides)
    return P.PrefixCacheSpec(**payload)


def _key(tokens=None, span=(0, 32), **kwargs) -> P.PrefixKey:
    return P.build_prefix_key(
        identity=kwargs.pop("identity", IDENTITY),
        tokens=tokens if tokens is not None else list(range(64)),
        token_span=span,
        **kwargs,
    )


@pytest.mark.unit
class TestPrefixKey:
    def test_key_requires_every_identity_field(self):
        with pytest.raises(ConfigError):
            P.PrefixKey(fields={"model_id": "m"}, tokens=(1, 2))

    def test_digest_covers_identity_span_and_content(self):
        base = _key()
        other_identity = _key(identity={**IDENTITY, "precision": "bfloat16"})
        assert base.effective_digest != other_identity.effective_digest
        other_content = _key(tokens=list(range(1, 65)))
        assert base.effective_digest != other_content.effective_digest

    def test_span_fields_change_the_digest(self):
        base = _key()
        mutated = P.PrefixKey(
            fields={**base.fields, "parent_chain_digest": "0" * 64}, tokens=base.tokens
        )
        assert base.effective_digest != mutated.effective_digest

    def test_block_size_truncates_to_complete_blocks(self):
        key = _key(span=(0, 30), block_size=16)
        assert key.length == 16

    def test_span_outside_the_sequence_refused(self):
        with pytest.raises(ConfigError):
            _key(span=(0, 128))


@pytest.mark.unit
class TestLookupSemantics:
    def test_full_hit_accounts_for_every_query_token(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        result = cache.lookup(
            key=key, query_tokens=32, block_groups=("full_attention",), request_id="r1"
        )
        assert result.cached_tokens == 32
        assert result.computed_tokens == 0
        assert result.saved_model_tokens == 32
        assert result.hit_fraction == pytest.approx(1.0)
        assert result.rejected_reason == ""
        assert result.partial_tail_tokens == 0

    def test_partial_hit_reports_the_tail(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        longer_prompt = list(range(96))
        query = _key(tokens=longer_prompt, span=(0, 96))
        result = cache.lookup(
            key=query, query_tokens=96, block_groups=("full_attention",), request_id="r1"
        )
        assert result.cached_tokens == 32
        assert result.computed_tokens == 64
        assert result.partial_tail_tokens == 64

    def test_identity_mismatch_never_hits(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        foreign = _key(identity={**IDENTITY, "model_id": "other/model"})
        result = cache.lookup(
            key=foreign, query_tokens=32, block_groups=("full_attention",), request_id="r1"
        )
        assert result.cached_tokens == 0
        assert result.rejected_reason == "identity_mismatch"

    def test_content_difference_never_hits(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        other = _key(tokens=list(range(1, 65)))
        result = cache.lookup(
            key=other, query_tokens=32, block_groups=("full_attention",), request_id="r1"
        )
        assert result.cached_tokens == 0

    def test_block_groups_take_the_intersection(self):
        cache = P.PrefixCache(_spec())
        key = _key(block_group="full_attention")
        cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        grouped = cache.lookup(
            key=_key(),
            query_tokens=32,
            block_groups=("full_attention", "sliding_window"),
            request_id="r1",
        )
        assert grouped.cached_tokens == 0

    def test_query_tokens_beyond_the_key_refused(self):
        cache = P.PrefixCache(_spec())
        with pytest.raises(ConfigError):
            cache.lookup(
                key=_key(),
                query_tokens=64,
                block_groups=("full_attention",),
                request_id="r1",
            )


@pytest.mark.unit
class TestCollisionPolicy:
    def test_forced_collision_rejected_by_the_frozen_policy(self):
        report = P.collision_is_detected(_spec(), IDENTITY)
        assert report["correct"]
        assert report["collision_detected"]
        assert report["served_tokens"] == 0

    def test_digest_only_policy_serves_the_forged_record(self):
        spec = _spec(collision_policy="strong_digest", verify_token_equality=False)
        report = P.collision_is_detected(spec, IDENTITY)
        assert not report["correct"]

    def test_strict_policy_requires_token_verification(self):
        with pytest.raises(ConfigError):
            _spec(
                collision_policy="digest_plus_token_equality",
                verify_token_equality=False,
            )

    def test_fixture_describes_what_is_being_tested(self):
        case = P.forced_collision_case(IDENTITY)
        assert case["digest_equal_by_construction"]
        assert not case["token_equality_holds"]
        assert case["expected"] == "REJECT"


@pytest.mark.unit
class TestNegativeFixtures:
    def test_every_key_field_mutation_makes_the_fixture_refusable(self):
        fixtures = P.identity_negative_fixtures(IDENTITY)
        assert len(fixtures) >= len(P.ALL_KEY_FIELDS)
        for fixture in fixtures:
            if fixture["must_hit"]:
                continue
            assert (not fixture["identity_matches"]) or (
                not fixture["span_matches"]
            ) or fixture["digest_differs"], fixture

    def test_shorter_prefix_is_a_legitimate_partial_hit(self):
        fixtures = P.identity_negative_fixtures(IDENTITY)
        shorter = next(
            item for item in fixtures if item["case"] == "prefix_shorter_by_one_token"
        )
        assert shorter["must_hit"]
        assert shorter["max_reusable_tokens"] == 31
        assert shorter["must_not_claim"] == 32


@pytest.mark.unit
class TestLifecycleAndEviction:
    def test_active_entries_are_never_dropped(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        entry = cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        with pytest.raises(ConfigError):
            cache._drop(entry.entry_id, reason="test")
        assert cache.entries

    def test_eviction_respects_the_byte_cap(self):
        cache = P.PrefixCache(_spec(max_cache_bytes=2048.0))
        for index in range(3):
            key = _key(tokens=list(range(index, index + 64)))
            cache.insert(
                key=key, block_ids=(index, index + 1), bytes_value=1024.0, request_id=f"r{index}"
            )
            cache.release(f"r{index}", retain=True)
        cache.release("r0", retain=True)
        evicted = cache.evict(target_bytes=2048.0)
        assert evicted
        assert cache.total_bytes() <= 2048.0

    def test_release_without_retain_drops_zero_ref_entries(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        cache.release("r0", retain=False)
        assert not cache.entries

    def test_invariant_report_checks_refcounts(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        entry = cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        assert cache.invariant_report()["ok"]
        entry.refcount = 5
        assert not cache.invariant_report()["ok"]

    def test_reset_reports_what_it_dropped(self):
        cache = P.PrefixCache(_spec())
        key = _key()
        cache.insert(key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0")
        report = cache.reset(reason="model changed")
        assert report["dropped_entries"] == 1
        assert report["reason"] == "model changed"

    def test_incomplete_block_insert_refused(self):
        cache = P.PrefixCache(_spec())
        with pytest.raises(ConfigError):
            cache.insert(
                key=_key(span=(0, 30)), block_ids=(0,), bytes_value=1.0, request_id="r0"
            )

    def test_unknown_eviction_policy_refused(self):
        with pytest.raises(ConfigError):
            _spec(eviction_policy="oracle")


@pytest.mark.unit
class TestSavingModel:
    def test_net_saving_subtracts_lookup_and_side_effects(self):
        model = P.NetSavingModel(
            baseline_prefill_ms=100.0,
            cached_request_prefill_ms=40.0,
            lookup_hash_ms=5.0,
            eviction_recompute_ms=10.0,
            other_requests_impact_ms=5.0,
            saved_tokens=512,
        )
        assert model.net_saved_time_ms() == 40.0
        assert model.naive_saving_ms(0.1) == pytest.approx(51.2)

    def test_zero_refused_by_the_comparison_gate(self):
        with pytest.raises(ConfigError):
            P.CacheCorrectnessComparison(
                request_id="r",
                token_sequence_equal=False,
                next_token_logits_max_abs_diff=1.0,
                kv_boundary_equal=False,
                resumed_position_equal=False,
                finish_reason_equal=False,
                long_generation_tokens=8,
                quality_gate_passed=False,
            )

    def test_equivalent_comparison_reports_equivalence(self):
        comparison = P.CacheCorrectnessComparison(
            request_id="r",
            token_sequence_equal=True,
            next_token_logits_max_abs_diff=0.0,
            kv_boundary_equal=True,
            resumed_position_equal=True,
            finish_reason_equal=True,
            long_generation_tokens=64,
            quality_gate_passed=True,
        )
        assert comparison.semantically_equivalent
