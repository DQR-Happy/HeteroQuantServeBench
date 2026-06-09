"""E07-03 KV geometry, block lifecycle, fragmentation and capacity."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import kv


def _geometry(**overrides) -> kv.KVGeometry:
    payload = {
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
        "element_bytes": 2,
    }
    payload.update(overrides)
    return kv.KVGeometry(**payload)


@pytest.mark.unit
class TestGeometry:
    def test_bytes_per_token_matches_the_documented_formula(self):
        geometry = _geometry()
        assert geometry.payload_bytes_per_token == 2 * 28 * 8 * 128 * 2
        assert geometry.bytes_per_token == float(geometry.payload_bytes_per_token)

    def test_quant_side_bytes_are_counted_separately(self):
        geometry = _geometry(quant_scale_bytes_per_token=8.0)
        assert geometry.bytes_per_token == geometry.payload_bytes_per_token + 8.0

    def test_blocks_for_rounds_up(self):
        assert kv.blocks_for(0, 16) == 0
        assert kv.blocks_for(1, 16) == 1
        assert kv.blocks_for(16, 16) == 1
        assert kv.blocks_for(17, 16) == 2

    def test_negative_tokens_refused(self):
        with pytest.raises(ConfigError):
            kv.blocks_for(-1, 16)

    def test_boundary_points_cover_p_minus_one_and_two_p(self):
        points = kv.block_boundary_points(16, blocks=2)
        assert (15, 16, 17, 31, 32, 33) == points

    def test_invalid_geometry_refused(self):
        with pytest.raises(ConfigError):
            _geometry(num_layers=0)
        with pytest.raises(ConfigError):
            _geometry(quant_scale_bytes_per_token=-1)


@pytest.mark.unit
class TestRequestAccounting:
    def test_internal_fragment_is_slot_waste(self):
        account = kv.KVRequestAccounting(
            request_id="r0", live_tokens=20, reserved_tokens=20, block_size=16, blocks=2, slots=32
        )
        assert account.internal_fragment_tokens == 12

    def test_reserved_must_cover_live_tokens(self):
        with pytest.raises(ConfigError):
            kv.KVRequestAccounting(
                request_id="r0",
                live_tokens=32,
                reserved_tokens=16,
                block_size=16,
                blocks=1,
                slots=16,
            )

    def test_block_count_must_match_reservation(self):
        with pytest.raises(ConfigError):
            kv.KVRequestAccounting(
                request_id="r0",
                live_tokens=16,
                reserved_tokens=16,
                block_size=16,
                blocks=3,
                slots=48,
            )


@pytest.mark.unit
class TestMemoryReconciliation:
    def test_unknown_class_refused(self):
        reconciliation = kv.MemoryReconciliation(
            geometry=_geometry(), block_size=16, active_tokens=16, tolerance_bytes=1.0
        )
        with pytest.raises(ConfigError):
            reconciliation.declare("some_fragmentation", 1024.0)

    def test_residual_outside_tolerance_is_not_explained(self):
        reconciliation = kv.MemoryReconciliation(
            geometry=_geometry(), block_size=16, active_tokens=16, tolerance_bytes=128.0
        )
        reconciliation.measured_framework_reserved = (
            reconciliation.predicted_total_bytes + 256
        )
        assert not reconciliation.explained()
        rows = {row["class"] for row in reconciliation.as_rows()}
        assert "unexplained_residual" in rows

    def test_tolerance_must_be_positive(self):
        reconciliation = kv.MemoryReconciliation(
            geometry=_geometry(), block_size=16, active_tokens=16
        )
        with pytest.raises(ConfigError):
            reconciliation.explained()

    def test_named_classes_close_the_books(self):
        reconciliation = kv.MemoryReconciliation(
            geometry=_geometry(), block_size=16, active_tokens=16, tolerance_bytes=64.0
        )
        reconciliation.declare("block_table_metadata", 256.0)
        reconciliation.declare("allocator_reserve", 512.0)
        reconciliation.measured_framework_reserved = reconciliation.predicted_total_bytes
        assert reconciliation.explained()


@pytest.mark.unit
class TestBlockPool:
    def _pool(self, **kwargs) -> kv.BlockPool:
        return kv.BlockPool(total_blocks=kwargs.pop("blocks", 8), block_size=kwargs.pop("block_size", 4))

    def test_allocate_write_and_invariants(self):
        pool = self._pool()
        blocks = pool.allocate("r0", token_start=0, token_count=9)
        assert len(blocks) == 3
        assert pool.invariant_report()["ok"]

    def test_pool_exhaustion_is_reported_not_silent(self):
        pool = self._pool(blocks=1)
        pool.allocate("r0", token_start=0, token_count=4)
        with pytest.raises(ConfigError):
            pool.allocate("r1", token_start=0, token_count=8)

    def test_share_increments_refcount_and_release_keeps_cache(self):
        pool = self._pool()
        blocks = pool.allocate("r0", token_start=0, token_count=4)
        pool.share(blocks[0], "r1")
        pool.mark_cached(blocks[0], "r0")
        pool.release("r0")
        record = pool.blocks[blocks[0]]
        assert record.refcount == 1
        assert "r1" in record.readers
        pool.release("r1")
        assert pool.blocks[blocks[0]].state == kv.BlockState.EVICTABLE

    def test_refusing_to_evict_an_active_block(self):
        pool = self._pool()
        blocks = pool.allocate("r0", token_start=0, token_count=4)
        with pytest.raises(ConfigError):
            pool.evict(blocks[0])

    def test_evict_only_eligible_blocks(self):
        pool = self._pool()
        blocks = pool.allocate("r0", token_start=0, token_count=4)
        pool.mark_cached(blocks[0], "r0")
        pool.release("r0")
        pool.evict(blocks[0])
        assert pool.blocks[blocks[0]].state == kv.BlockState.FREE

    def test_double_free_is_absorbed_without_corrupting_the_pool(self):
        pool = self._pool()
        pool.allocate("r0", token_start=0, token_count=4)
        assert pool.inject_double_free("r0") == "DOUBLE_FREE_ABSORBED"
        assert pool.invariant_report()["ok"]

    def test_refcount_error_is_detected_by_the_invariants(self):
        pool = self._pool()
        blocks = pool.allocate("r0", token_start=0, token_count=4)
        pool.inject_refcount_error(blocks[0])
        assert not pool.invariant_report()["ok"]

    def test_stale_id_rejected_after_eviction(self):
        pool = self._pool()
        blocks = pool.allocate("r0", token_start=0, token_count=4)
        pool.mark_cached(blocks[0], "r0")
        pool.release("r0")
        pool.evict(blocks[0])
        assert pool.inject_stale_id_access(blocks[0]) == "STALE_ID_REJECTED"

    def test_live_block_id_is_not_a_stale_id(self):
        pool = self._pool()
        blocks = pool.allocate("r0", token_start=0, token_count=4)
        assert pool.inject_stale_id_access(blocks[0]) == "BLOCK_STILL_LIVE"

    def test_illegal_state_transition_refused(self):
        record = kv.BlockRecord(block_id=0)
        with pytest.raises(ConfigError):
            record.transition(kv.BlockState.SHARED_CACHED, "skip-from-free")


@pytest.mark.unit
class TestEvictionSelection:
    def test_lru_prefers_the_oldest(self):
        candidates = [
            kv.EvictionCandidate(block_id=1, last_use_iteration=10, reuse_count=5, bytes_value=1.0),
            kv.EvictionCandidate(block_id=2, last_use_iteration=1, reuse_count=0, bytes_value=1.0),
        ]
        chosen = kv.select_evictions(candidates, "lru", 1)
        assert [item.block_id for item in chosen] == [2]

    def test_reuse_aware_prefers_the_least_reused(self):
        candidates = [
            kv.EvictionCandidate(block_id=1, last_use_iteration=1, reuse_count=9, bytes_value=1.0),
            kv.EvictionCandidate(block_id=2, last_use_iteration=10, reuse_count=0, bytes_value=1.0),
        ]
        chosen = kv.select_evictions(candidates, "lru_reuse_aware", 1)
        assert [item.block_id for item in chosen] == [2]

    def test_pinned_candidates_are_never_selected(self):
        candidates = [
            kv.EvictionCandidate(
                block_id=1, last_use_iteration=0, reuse_count=0, bytes_value=1.0, pinned=True
            )
        ]
        assert kv.select_evictions(candidates, "lru", 5) == []

    def test_unknown_policy_refused(self):
        with pytest.raises(ConfigError):
            kv.select_evictions([], "oracle", 1)


@pytest.mark.unit
class TestCapacityAndOOM:
    def test_safe_admission_applies_the_watermark(self):
        model = kv.KVCapacityModel(
            geometry=_geometry(),
            block_size=16,
            blocks_total=100,
            non_kv_resident_bytes=0.0,
            watermark=0.5,
        )
        assert model.theoretical_max_tokens() == 1600
        assert model.safe_admission_tokens() == 800

    def test_bisection_finds_the_boundary(self):
        boundary = kv.find_capacity_boundary(lambda tokens: tokens <= 37, low=1, high=64)
        assert boundary.safe_tokens == 37
        assert boundary.oom_tokens == 38

    def test_bisection_refuses_a_dirty_lower_bound(self):
        with pytest.raises(ConfigError):
            kv.find_capacity_boundary(lambda tokens: False, low=1, high=8)

    def test_bisection_refuses_an_unbounded_capacity(self):
        with pytest.raises(ConfigError):
            kv.find_capacity_boundary(lambda tokens: True, low=1, high=8)

    def test_oom_actions_are_bounded_and_deterministic(self):
        assert kv.oom_action("actual_leak", 0) == "fail"
        assert kv.oom_action("predictable_admission_reject", 0) == "reject"
        with pytest.raises(ConfigError):
            kv.oom_action("execution_oom", kv.MAX_OOM_ATTEMPTS)

    def test_unknown_oom_kind_refused(self):
        with pytest.raises(ConfigError):
            kv.oom_action("mystery", 0)

    def test_context_limit_uses_the_earliest_layer(self):
        check = kv.ContextLimitCheck(
            requested_tokens=100,
            model_max=4096,
            runtime_max=64,
            kv_capacity_tokens=2048,
        )
        assert check.rejects
        assert check.earliest_rejecting_layer == "runtime_max"
        assert check.as_dict()["headroom_tokens"] == -36


@pytest.mark.unit
class TestResourceSlope:
    def test_steady_series_is_steady(self):
        slope = kv.resource_slope("kv", [100.0, 100.0, 100.0, 100.0], tolerance=1.0)
        assert slope.verdict in ("STEADY", "BOUNDED_CACHE")

    def test_growing_series_is_detected(self):
        slope = kv.resource_slope("kv", [1.0, 2.0, 4.0, 8.0], tolerance=0.5)
        assert slope.verdict == "GROWING"

    def test_slope_needs_three_samples(self):
        with pytest.raises(ConfigError):
            kv.resource_slope("kv", [1.0, 2.0], tolerance=1.0)

    def test_slope_needs_a_positive_tolerance(self):
        with pytest.raises(ConfigError):
            kv.resource_slope("kv", [1.0, 2.0, 3.0], tolerance=0.0)
