"""Unit tests for the E02-04 batch-capacity primitives.

``batch_token_budgets`` (E02-04 §4), ``compute_batch_kv_cache_info`` and
``batch_throughput`` (E02-04 §6) are pure helpers — tested without a model or
GPU, matching the E02-03 ``request_summary`` test style. ``benchmark_model_core_batch``
is covered only for its argument validation (no forward pass).
"""

from __future__ import annotations

import pytest
import torch

from hqsb.benchmark.batch_core import (
    BatchBenchmarkOOM,
    batch_throughput,
    batch_token_budgets,
    benchmark_model_core_batch,
    compute_batch_kv_cache_info,
)


class TestBatchTokenBudgets:
    def test_three_budgets(self):
        b = batch_token_budgets(batch_size=4, isl=128, osl=32)
        assert b["input_compute_tokens"] == 4 * 128
        assert b["reserved_token_capacity"] == 4 * (128 + 32)
        assert b["final_kv_tokens"] == 4 * (128 + 32 - 1)

    def test_batch_one(self):
        b = batch_token_budgets(batch_size=1, isl=512, osl=128)
        assert b["final_kv_tokens"] == 512 + 128 - 1

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            batch_token_budgets(batch_size=0, isl=128, osl=32)
        with pytest.raises(ValueError):
            batch_token_budgets(batch_size=2, isl=0, osl=32)
        with pytest.raises(ValueError):
            batch_token_budgets(batch_size=2, isl=128, osl=0)


class TestComputeBatchKvCacheInfo:
    def test_scales_linearly_with_batch(self):
        single = compute_batch_kv_cache_info(
            num_layers=2, num_kv_heads=8, head_dim=128,
            batch_size=1, context_length=10, dtype_bytes=2,
        )
        four = compute_batch_kv_cache_info(
            num_layers=2, num_kv_heads=8, head_dim=128,
            batch_size=4, context_length=10, dtype_bytes=2,
        )
        # per_token_bytes = 2 * L * Hkv * Dh * bytes = 2*2*8*128*2 = 8192.
        assert single["per_token_bytes"] == 8192
        assert single["per_sequence_bytes"] == 8192 * 10
        assert four["per_sequence_bytes"] == single["per_sequence_bytes"]
        assert four["total_bytes"] == 4 * single["total_bytes"]

    def test_context_length_scales(self):
        a = compute_batch_kv_cache_info(
            num_layers=2, num_kv_heads=8, head_dim=128,
            batch_size=2, context_length=5, dtype_bytes=2,
        )
        b = compute_batch_kv_cache_info(
            num_layers=2, num_kv_heads=8, head_dim=128,
            batch_size=2, context_length=10, dtype_bytes=2,
        )
        assert b["total_bytes"] == 2 * a["total_bytes"]


class TestBatchThroughput:
    def test_prefill_tps_uses_batch_work(self):
        # B*I / prefill_time = 4*128 / 0.1s = 5120.
        t = batch_throughput(
            batch_size=4, isl=128, osl=32,
            prefill_ms=100.0, decode_total_ms=3100.0, e2e_ms=3200.5,
        )
        assert t["batch_prefill_tokens_per_s"] == pytest.approx(5120.0)

    def test_decode_tps_uses_g_minus_1(self):
        # B*(G-1) / decode = 4*31 / 3.1s = 40.
        t = batch_throughput(
            batch_size=4, isl=128, osl=32,
            prefill_ms=100.0, decode_total_ms=3100.0, e2e_ms=3200.5,
        )
        assert t["batch_decode_tokens_per_s"] == pytest.approx(40.0)

    def test_output_tps(self):
        # B*G / e2e = 4*32 / 3.2005s.
        t = batch_throughput(
            batch_size=4, isl=128, osl=32,
            prefill_ms=100.0, decode_total_ms=3100.0, e2e_ms=3200.5,
        )
        assert t["batch_output_tokens_per_s"] == pytest.approx(128 / 3.2005)

    def test_g_eq_1_decode_zero(self):
        t = batch_throughput(
            batch_size=2, isl=128, osl=1,
            prefill_ms=100.0, decode_total_ms=0.0, e2e_ms=100.0,
        )
        assert t["batch_decode_tokens_per_s"] == 0.0

    def test_zero_prefill_zero_tps(self):
        t = batch_throughput(
            batch_size=2, isl=128, osl=32,
            prefill_ms=0.0, decode_total_ms=3100.0, e2e_ms=3100.0,
        )
        assert t["batch_prefill_tokens_per_s"] == 0.0


class TestBatchBenchmarkOOM:
    def test_carries_stage_and_step(self):
        err = BatchBenchmarkOOM("decode", 42, "CUDA out of memory")
        assert err.stage == "decode"
        assert err.step == 42
        assert "out of memory" in str(err)


class TestBenchmarkModelCoreBatchValidation:
    def test_rejects_non_2d_input(self):
        model = object()  # validation happens before any forward pass
        with pytest.raises(ValueError):
            benchmark_model_core_batch(
                model,
                input_ids=torch.ones(1, 2, 3, dtype=torch.long),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
                output_tokens=1,
            )

    def test_rejects_bad_osl(self):
        model = object()
        with pytest.raises(ValueError):
            benchmark_model_core_batch(
                model,
                input_ids=torch.ones(1, 2, dtype=torch.long),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
                output_tokens=0,
            )
