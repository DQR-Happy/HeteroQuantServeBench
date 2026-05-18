"""Unit tests for the E02-05 predictive memory model.

Every expected value here is **hand-computed** from the formula and hardcoded,
never produced by the function under test (protocol §8 step 2: "expected 与
预测器独立，不能调用同一个函数生成预期"). The anti-examples of protocol §8
step 8 (``Hq`` instead of ``Hkv``, ``MiB`` treated as ``MB``, tied parameters
double counted) each get an explicit test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hqsb.benchmark.memory_model import (
    DeviceMemorySampler,
    activation_lower_bound_bytes,
    bytes_to_decimal_mb,
    bytes_to_gib,
    bytes_to_mib,
    decompose,
    eager_attention_workspace_bytes,
    format_bytes,
    kv_cache_metadata,
    kv_ledger,
    kv_ledger_from_model,
    memory_snapshot,
    model_memory_inventory,
    storage_dedup_summary,
    storage_groups,
    summarize_samples,
    tensor_inventory,
)

# Qwen3-1.7B frozen structure (config.json + E02-02 census).
_L = 28
_HKV = 8
_HQ = 16
_DH = 128
_HIDDEN = 2048
_INTER = 6144


@pytest.mark.unit
class TestKvLedgerHandComputed:
    """Protocol §4/§4.1 conditional worked examples, hardcoded."""

    def test_per_token_per_layer_is_4_kib(self):
        ledger = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=128,
        )
        # 2 (K+V) * 8 * 128 * 2 bytes = 4096 bytes = 4 KiB
        assert ledger.per_token_per_layer_bytes == 4096
        assert ledger.per_token_per_layer_bytes / 1024 == 4

    def test_per_token_all_layers_is_112_kib(self):
        ledger = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=128,
        )
        # 28 * 4096 = 114688 bytes = 112 KiB
        assert ledger.per_token_all_layers_bytes == 114688
        assert ledger.per_token_all_layers_bytes / 1024 == 112

    def test_b1_t2048_is_exactly_224_mib(self):
        ledger = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=2048,
        )
        # 114688 * 2048 = 234881024 bytes; / 2^20 = 224.0 MiB exactly
        assert ledger.total_bytes == 234_881_024
        assert bytes_to_mib(ledger.total_bytes) == pytest.approx(224.0, abs=1e-9)

    def test_b2_doubles_kv(self):
        ledger = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=2, context_length=2048,
        )
        assert bytes_to_mib(ledger.total_bytes) == pytest.approx(448.0, abs=1e-9)

    def test_filled_vs_capacity_off_by_one_is_not_fragmentation(self):
        """I=2048, G=128: filled=2175 -> 237.890625 MiB, capacity=2176 -> 238."""
        filled = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=2175,
        )
        capacity = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=2176,
        )
        assert bytes_to_mib(filled.total_bytes) == pytest.approx(237.890625, abs=1e-6)
        assert bytes_to_mib(capacity.total_bytes) == pytest.approx(238.0, abs=1e-9)
        # The two lengths are one token apart; using capacity as filled is a
        # recording bug, not allocator fragmentation.
        assert capacity.total_bytes - filled.total_bytes == 114688

    def test_rejects_non_positive_dimensions(self):
        for kwargs in (
            {"num_layers": 0}, {"num_kv_heads": 0}, {"head_dim": 0},
            {"element_bytes": 0}, {"batch_size": 0}, {"context_length": 0},
        ):
            base = {
                "num_layers": _L, "num_kv_heads": _HKV, "head_dim": _DH,
                "element_bytes": 2, "batch_size": 1, "context_length": 8,
            }
            base.update(kwargs)
            with pytest.raises(ValueError):
                kv_ledger(**base)


@pytest.mark.unit
class TestKvHeadsNotQueryHeads:
    """Protocol §8 step 8 anti-example: using ``Hq`` doubles persistent KV."""

    def test_kv_heads_uses_hkv(self):
        ledger = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=2048,
        )
        assert ledger.per_token_per_layer_bytes == 2 * _HKV * _DH * 2

    def test_wrong_hq_prediction_is_twice_the_correct_one(self):
        correct = kv_ledger(
            num_layers=_L, num_kv_heads=_HKV, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=2048,
        )
        wrong = kv_ledger(
            num_layers=_L, num_kv_heads=_HQ, head_dim=_DH,
            element_bytes=2, batch_size=1, context_length=2048,
        )
        assert wrong.total_bytes == 2 * correct.total_bytes
        assert bytes_to_mib(wrong.total_bytes) == pytest.approx(448.0)
        # The Hq-based number must not be accepted as the persistent KV value.
        assert wrong.total_bytes != correct.total_bytes

    def test_kv_ledger_from_model_uses_kv_heads_field(self):
        class _Config:
            num_hidden_layers = _L
            num_attention_heads = _HQ
            num_key_value_heads = _HKV
            head_dim = _DH
            hidden_size = _HIDDEN

        class _Model:
            config = _Config()

        ledger = kv_ledger_from_model(
            _Model(), batch_size=1, context_length=2048, element_bytes=2
        )
        assert ledger.num_kv_heads == _HKV
        assert bytes_to_mib(ledger.total_bytes) == pytest.approx(224.0, abs=1e-9)


@pytest.mark.unit
class TestUnitConversion:
    """Protocol §8 step 8 anti-example: MiB treated as MB."""

    def test_one_gib_is_not_one_gb(self):
        one_gib = 1024**3
        assert bytes_to_gib(one_gib) == pytest.approx(1.0)
        # 1 GiB = 1073741824 B = 1073.741824 MB (decimal)
        assert bytes_to_decimal_mb(one_gib) == pytest.approx(1073.741824, abs=1e-9)
        # 1 GiB expressed in decimal GB is 1.073741824, i.e. 7.37% larger; the
        # unit a report writes matters (protocol §2).
        one_gb = 1000**3
        assert one_gib / one_gb == pytest.approx(1.073741824, abs=1e-12)

    def test_mib_and_decimal_mb_differ_by_about_4_86_percent(self):
        nbytes = 224 * 1024**2  # the 224 MiB KV example
        assert bytes_to_mib(nbytes) == pytest.approx(224.0)
        assert bytes_to_decimal_mb(nbytes) == pytest.approx(234.881024, abs=1e-6)
        assert (
            bytes_to_decimal_mb(nbytes) / bytes_to_mib(nbytes) - 1.0
        ) == pytest.approx(0.048576, abs=1e-9)

    def test_format_bytes_names_both_units(self):
        text = format_bytes(1024**2)
        assert "MiB" in text and "MB" in text and "B" in text
        assert "1.000 MiB" in text
        assert "1.049 MB" in text


@pytest.mark.unit
class TestEagerWorkspaceAndActivation:
    def test_long_prefill_attention_workspace(self):
        """S=2048, Hq=16: fp16 score (128 MiB) + fp32 softmax (256 MiB)."""
        ws = eager_attention_workspace_bytes(
            batch_size=1, num_query_heads=_HQ, seq_len=2048
        )
        assert ws["score_cells"] == 16 * 2048 * 2048
        assert ws["score_matrix_bytes"] == 134_217_728  # 128 MiB
        assert ws["softmax_matrix_bytes"] == 268_435_456  # 256 MiB
        assert ws["total_bytes"] == 402_653_184  # 384 MiB

    def test_b4_long_prefill_matches_e02_04_derivation(self):
        ws = eager_attention_workspace_bytes(
            batch_size=4, num_query_heads=_HQ, seq_len=2048
        )
        assert bytes_to_mib(ws["score_matrix_bytes"]) == pytest.approx(512.0)
        assert bytes_to_mib(ws["softmax_matrix_bytes"]) == pytest.approx(1024.0)
        assert bytes_to_mib(ws["total_bytes"]) == pytest.approx(1536.0)

    def test_workspace_scales_quadratically_in_seq(self):
        a = eager_attention_workspace_bytes(
            batch_size=1, num_query_heads=_HQ, seq_len=512
        )
        b = eager_attention_workspace_bytes(
            batch_size=1, num_query_heads=_HQ, seq_len=1024
        )
        assert b["total_bytes"] == 4 * a["total_bytes"]

    def test_activation_lower_bound_hand_computed(self):
        # 1 * 2048 * (2048 + 6144) * 2 = 33554432 B = 32 MiB
        assert (
            activation_lower_bound_bytes(
                batch_size=1, seq_len=2048,
                hidden_size=_HIDDEN, intermediate_size=_INTER, element_bytes=2,
            )
            == 33_554_432
        )

    def test_workspace_rejects_zero(self):
        with pytest.raises(ValueError):
            eager_attention_workspace_bytes(
                batch_size=1, num_query_heads=0, seq_len=128
            )


class _TiedModule(nn.Module):
    """Two parameter names sharing one storage, plus a separate buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(4, 8, dtype=torch.float16))
        # Distinct Parameter object that shares embed's storage: the tied
        # embedding/LM-head case, which named_parameters(remove_duplicate=True)
        # would hide.
        self.head = nn.Parameter(self.embed)
        self.scale = nn.Parameter(torch.zeros(8, dtype=torch.float16))
        self.register_buffer("inv_freq", torch.zeros(4, dtype=torch.float32))


@pytest.mark.unit
class TestStorageInventory:
    """Protocol §8 step 8 anti-example: tied parameters counted twice."""

    def test_tied_parameter_detected_as_alias_group(self):
        inventory = tensor_inventory(_TiedModule())
        summary = inventory["parameter_summary"]
        # 3 names, but embed/head share a storage -> 2 unique storages
        assert summary["num_tensors"] == 3
        assert summary["num_unique_storages"] == 2
        assert summary["alias_group_count"] == 1
        group = summary["alias_groups"][0]
        assert group["num_members"] == 2
        assert set(group["member_names"]) == {"embed", "head"}

    def test_logical_total_double_counts_dedup_does_not(self):
        inventory = tensor_inventory(_TiedModule())
        summary = inventory["parameter_summary"]
        # logical: (4*8 + 4*8 + 8) * 2 B = 144 B ; dedup: (4*8 + 8) * 2 B = 80 B
        assert summary["logical_total_bytes"] == 144
        assert summary["dedup_total_bytes"] == 80
        assert summary["duplicate_counted_bytes"] == 64

    def test_buffer_dtype_counted_by_real_size_not_fp16_assumption(self):
        inventory = tensor_inventory(_TiedModule())
        buffer_summary = inventory["buffer_summary"]
        # inv_freq: 4 * 4 B (fp32) = 16 B, not 4 * 2 B
        assert buffer_summary["dedup_total_bytes"] == 16
        assert buffer_summary["dtype_histogram"] == {"float32": 1}

    def test_distinct_storages_not_merged(self):
        module = nn.Module()
        module.a = nn.Parameter(torch.zeros(4, dtype=torch.float16))
        module.b = nn.Parameter(torch.zeros(4, dtype=torch.float16))
        summary = tensor_inventory(module)["parameter_summary"]
        assert summary["num_unique_storages"] == 2
        assert summary["alias_group_count"] == 0
        assert summary["duplicate_counted_bytes"] == 0

    def test_group_ids_are_stable_labels_not_addresses(self):
        records = [
            {"name": "a", "storage_ptr": 111, "storage_nbytes": 16, "storage_offset": 0},
            {"name": "b", "storage_ptr": 111, "storage_nbytes": 16, "storage_offset": 0},
            {"name": "c", "storage_ptr": 999, "storage_nbytes": 32, "storage_offset": 0},
        ]
        groups = storage_groups(records)
        assert [g["group_id"] for g in groups] == ["storage_0", "storage_1"]
        assert groups[0]["member_names"] == ["a", "b"]
        assert groups[0]["is_alias_group"] is True
        assert groups[1]["is_alias_group"] is False

    def test_zero_size_storages_are_not_aliased_together(self):
        records = [
            {"name": "x", "storage_ptr": 0, "storage_nbytes": 0, "storage_offset": 0},
            {"name": "y", "storage_ptr": 0, "storage_nbytes": 0, "storage_offset": 0},
        ]
        groups = storage_groups(records)
        assert len(groups) == 2
        assert all(not g["is_alias_group"] for g in groups)

    def test_empty_inventory_is_all_zero(self):
        summary = storage_dedup_summary([])
        assert summary["num_tensors"] == 0
        assert summary["dedup_total_bytes"] == 0
        assert summary["duplicate_counted_bytes"] == 0


class _FakeCache:
    def __init__(self, keys, values) -> None:
        self.key_cache = keys
        self.value_cache = values


@pytest.mark.unit
class TestKvCacheMetadata:
    def _cache_two_layers(self, context: int = 128):
        keys = [
            torch.zeros(1, _HKV, context, _DH, dtype=torch.float16)
            for _ in range(2)
        ]
        values = [
            torch.zeros(1, _HKV, context, _DH, dtype=torch.float16)
            for _ in range(2)
        ]
        return _FakeCache(keys, values)

    def test_reports_layers_shapes_and_bytes(self):
        meta = kv_cache_metadata(self._cache_two_layers(128))
        assert meta["source_path"] == "key_cache_value_cache_lists"
        assert meta["num_layers"] == 2
        assert meta["layers"][0]["key_shape"] == [1, _HKV, 128, _DH]
        # per layer: 2 (K+V) * 8 * 128 * 128 * 2 B = 524288 B
        assert meta["layers"][0]["logical_bytes"] == 524_288
        assert meta["total_logical_bytes"] == 2 * 524_288
        assert meta["per_token_per_layer_bytes_observed"] == 4096
        assert meta["kv_heads_observed"] == [_HKV]
        assert meta["element_bytes_observed"] == [2]

    def test_reports_filled_context_length(self):
        meta = kv_cache_metadata(self._cache_two_layers(160))
        assert meta["context_lengths_observed"] == [160]
        assert meta["context_filled"] == 160

    def test_preallocated_view_is_flagged(self):
        """A view into a larger buffer exposes Tcapacity > Tfilled."""
        key_buffer = torch.zeros(1, _HKV, 256, _DH, dtype=torch.float16)
        value_buffer = torch.zeros(1, _HKV, 256, _DH, dtype=torch.float16)
        key_view = key_buffer[:, :, :128, :]
        value_view = value_buffer[:, :, :128, :]
        meta = kv_cache_metadata(_FakeCache([key_view], [value_view]))
        layer = meta["layers"][0]
        # logical 2 * 262144 B; storage 2 * 524288 B -> 2.0x preallocated
        assert layer["logical_bytes"] == 524_288
        assert layer["storage_over_logical"] == pytest.approx(2.0)
        assert meta["appears_preallocated"] is True

    def test_contiguous_cache_not_flagged_as_preallocated(self):
        meta = kv_cache_metadata(self._cache_two_layers(64))
        assert meta["appears_preallocated"] is False

    def test_empty_cache_reports_zero_without_crashing(self):
        meta = kv_cache_metadata(_FakeCache([], []))
        assert meta["num_layers"] == 0
        assert meta["total_logical_bytes"] == 0
        assert meta["source_path"] == "none"


@pytest.mark.unit
class TestDecompose:
    def test_residual_and_unresolved_split(self):
        result = decompose(
            predicted={"weights": 100, "kv": 200},
            observed_bytes=400,
            explained={"attention_workspace": 50},
        )
        assert result["predicted_total_bytes"] == 300
        assert result["residual_bytes"] == 100
        assert result["explained_total_bytes"] == 50
        assert result["unresolved_bytes"] == 50
        assert result["residual_ratio"] == pytest.approx(0.25)
        assert result["unresolved_ratio"] == pytest.approx(0.125)

    def test_fully_explained_residual_leaves_nothing_unresolved(self):
        result = decompose(
            predicted={"weights": 100},
            observed_bytes=150,
            explained={"cuda_context": 50},
        )
        assert result["unresolved_bytes"] == 0

    def test_missing_observation_yields_none_not_zero(self):
        result = decompose(predicted={"weights": 100}, observed_bytes=None)
        assert result["residual_bytes"] is None
        assert result["unresolved_bytes"] is None
        assert result["observed_mib"] is None

    def test_prediction_can_exceed_observation(self):
        result = decompose(predicted={"weights": 1000}, observed_bytes=800)
        assert result["residual_bytes"] == -200


@pytest.mark.unit
class TestSamplerSummary:
    def test_summarize_hand_computed(self):
        summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
        assert summary["count"] == 4
        assert summary["min"] == 1.0
        assert summary["max"] == 4.0
        assert summary["mean"] == pytest.approx(2.5)
        assert summary["p50"] == pytest.approx(2.5)
        assert summary["p95"] == pytest.approx(3.85)

    def test_empty_samples_do_not_fabricate_a_peak(self):
        summary = summarize_samples([])
        assert summary == {
            "count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0,
        }

    def test_sampler_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            DeviceMemorySampler(interval_s=0)


@pytest.mark.unit
class TestPeakModelIsPhaseMax:
    """The prefill peak is a max over mutually exclusive phases, not a sum.

    The fp16 score / fp32 softmax buffers are released before ``lm_head``
    materializes the full-sequence logits, so adding them double-counts. The
    run-0 development data refuted the sum form (predicted 1233.5 MiB vs
    observed 817.0 MiB at I=2048) and the max form reproduces the observation.
    """

    DIMS = {
        "num_layers": 28, "num_query_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048, "intermediate_size": 6144,
        "vocab_size": 151936, "element_bytes": 2, "tie_word_embeddings": True,
    }

    def _theory(self, isl: int, osl: int, batch: int):
        from hqsb.benchmark.memory_experiment import point_theory

        return point_theory(self.DIMS, isl, osl, batch)

    def test_staged_point_hand_computed(self):
        theory = self._theory(2048, 32, 1)
        peak = theory["peak_prefill_model"]
        # kv 234881024 + ws 402653184 + act 33554432 = 671088640
        assert peak["attention_branch_bytes"] == 671_088_640
        # kv 234881024 + logits 622329856 = 857210880
        assert peak["lm_head_branch_bytes"] == 857_210_880
        assert peak["total_bytes"] == 857_210_880
        assert peak["winning_branch"] == "lm_head"

    def test_naive_sum_double_counts_by_477_mib(self):
        theory = self._theory(2048, 32, 1)
        gap = (
            theory["naive_sum_peak_prefill_bytes"]
            - theory["peak_prefill_model"]["total_bytes"]
        )
        assert gap == 436_207_616
        assert bytes_to_mib(gap) == pytest.approx(416.0, abs=1e-6)

    def test_attention_branch_wins_for_very_long_context(self):
        # At I=4096 the fp16+fp32 attention matrices (6 bytes/cell) overtake the
        # fp16 logits (2 bytes per position per vocab).
        theory = self._theory(4096, 8, 1)
        peak = theory["peak_prefill_model"]
        assert peak["attention_branch_bytes"] > peak["lm_head_branch_bytes"]
        assert peak["winning_branch"] == "attention_workspace"

    def test_decode_peak_is_kv_plus_a_small_term(self):
        theory = self._theory(2048, 32, 1)
        kv_final = theory["kv_final"]["total_bytes"]
        peak = theory["peak_decode_model"]["total_bytes"]
        assert peak >= kv_final
        # The decode-side extra term must stay far below the prefill peak.
        assert peak - kv_final < 0.01 * theory["peak_prefill_model"]["total_bytes"]


@pytest.mark.unit
class TestMemorySnapshot:
    def test_without_cuda_does_not_claim_device_numbers(self):
        snapshot = memory_snapshot(label="M0", with_cuda=False)
        assert snapshot["label"] == "M0"
        assert snapshot["cuda"]["allocated_mb"] is None
        assert snapshot["device"]["free_mb"] is None
        assert snapshot["device_used_mb"] is None
        # Process/host view is still available pre-context.
        assert snapshot["process_rss_bytes"] >= 0
        assert "available_mb" in snapshot["host"]

    def test_model_inventory_reports_resident_total(self):
        inventory = model_memory_inventory(_TiedModule())
        # parameters 80 B (dedup) + buffer 16 B = 96 B resident
        assert inventory["resident_bytes"] == 96
        assert inventory["resident_mib"] == pytest.approx(96 / 1024**2)
