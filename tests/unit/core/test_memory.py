"""Unit tests for memory and KV-cache accounting."""

from __future__ import annotations

import pytest

from hqsb.benchmark.memory import (
    KvCacheInfo,
    compute_kv_cache_info,
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
