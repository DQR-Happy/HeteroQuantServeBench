"""Unit tests for memory and KV-cache accounting."""

from __future__ import annotations

import pytest

from hqsb.benchmark.memory import (
    DEFAULT_RESERVE_MB,
    DEFAULT_RESERVE_RATIO,
    KvCacheInfo,
    compute_kv_cache_info,
    memory_budget_bytes,
    model_kv_cache_config,
)


@pytest.mark.unit
class TestKvCacheInfo:
    def test_known_config(self):
        info = compute_kv_cache_info(
            num_layers=28,
            num_kv_heads=2,
            head_dim=128,
            context_length=128,
            dtype_bytes=2,
        )
        # per-token = 2 * 28 * 2 * 128 * 2 = 28672 bytes
        assert info.per_token_bytes() == 2 * 28 * 2 * 128 * 2
        assert info.total_bytes() == info.per_token_bytes() * 128

    def test_validation_rejects_non_positive(self):
        with pytest.raises(ValueError):
            compute_kv_cache_info(
                num_layers=0, num_kv_heads=2, head_dim=128,
                context_length=128, dtype_bytes=2,
            )


@pytest.mark.unit
class TestModelKvCacheConfig:
    class _Config:
        num_hidden_layers = 28
        num_attention_heads = 12
        num_key_value_heads = 2
        hidden_size = 1536
        head_dim = 128

    def test_extracts_gqa_params(self):
        class Model:
            config = self._Config()

        result = model_kv_cache_config(Model())
        assert result == {
            "num_layers": 28,
            "num_kv_heads": 2,
            "head_dim": 128,
        }

    def test_missing_config_returns_empty(self):
        class Model:
            pass

        assert model_kv_cache_config(Model()) == {}

    def test_derives_head_dim_from_hidden(self):
        class Config:
            num_hidden_layers = 4
            num_attention_heads = 8
            hidden_size = 1024
            # no head_dim, no num_key_value_heads -> fall back to attn heads

        class Model:
            config = Config()

        result = model_kv_cache_config(Model())
        assert result["head_dim"] == 128  # 1024 // 8
        assert result["num_kv_heads"] == 8


@pytest.mark.unit
class TestMemoryBudgetBytes:
    """Budget must be derived from measured free memory, never hardcoded.

    Regression guard: a previous implementation hard-coded a 6 GB GPU budget,
    which overshoots the real headroom on unified-memory devices (Jetson
    8 GB reports ~7.6 GiB total but only ~4 GiB free).
    """

    GIB = 1024**3

    def test_absolute_reserve_dominates_on_large_free(self):
        free = 64 * self.GIB
        budget = memory_budget_bytes(free, reserve_ratio=0.15, reserve_mb=512)
        # proportional reserve = 9.6 GiB > absolute 512 MiB -> proportional wins
        assert budget == int(free * 0.85)

    def test_absolute_reserve_dominates_on_small_free(self):
        free = int(2 * self.GIB)
        budget = memory_budget_bytes(free, reserve_ratio=0.15, reserve_mb=512)
        # proportional reserve = 0.3 GiB < absolute 512 MiB -> absolute wins
        assert budget == free - 512 * 1024**2

    def test_never_exceeds_free(self):
        for free_mb in (256, 512, 1024, 4096, 8192):
            free = free_mb * 1024**2
            assert 0 <= memory_budget_bytes(free) <= free

    def test_tight_headroom_yields_zero_not_negative(self):
        # free smaller than the absolute reserve -> clamped to 0
        assert memory_budget_bytes(0) == 0
        assert memory_budget_bytes(100 * 1024**2, reserve_mb=512) == 0

    def test_jetson_8gb_scenario_stays_within_free(self):
        """Realistic case: 7.6 GiB total, ~4.2 GiB free."""
        free = int(4.2 * self.GIB)
        budget = memory_budget_bytes(free)
        assert budget < free
        assert budget / self.GIB == pytest.approx(4.2 * 0.85, abs=0.05)
        # A 3.4 GiB FP16 Qwen3-1.7B must still fit.
        assert budget > int(3.4 * self.GIB)

    def test_defaults_are_conservative(self):
        assert 0 < DEFAULT_RESERVE_RATIO < 1
        assert DEFAULT_RESERVE_MB > 0

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            memory_budget_bytes(-1)
        with pytest.raises(ValueError):
            memory_budget_bytes(1024, reserve_ratio=1.0)
        with pytest.raises(ValueError):
            memory_budget_bytes(1024, reserve_ratio=-0.1)
