"""E02-05 measurement protocol and point-level measurement engine.

This module holds everything E02-05 needs that is more than pure arithmetic but
less than a CLI: the pre-registered protocol, the staged/sweep measurement
point, the multi-request and hook A/B probes, the OOM cross-reference and the
independent hand-computed unit self-test.

Splitting it out of the runner keeps ``scripts/audit/run_e02_05_memory_model.py``
a thin CLI, so the measurement semantics can be unit-tested without a GPU.

Protocol source: ``docs/stage_experiments/details/S02/E02-05_memory_model_validation.md``
and ``docs/stage_experiments/S02_实验清单.md`` (E02-05).
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.memory import process_rss_bytes, process_swap_bytes
from hqsb.benchmark.memory_model import (
    GIB,
    MIB,
    DeviceMemorySampler,
    activation_lower_bound_bytes,
    bytes_to_decimal_mb,
    bytes_to_gib,
    bytes_to_mib,
    decompose,
    eager_attention_workspace_bytes,
    kv_cache_metadata,
    kv_ledger,
    memory_snapshot,
    snapshot_used_bytes,
)
from hqsb.benchmark.metrics import model_core_timings
from hqsb.benchmark.workload import make_fixed_token_input

logger = logging.getLogger("e02_05")

MODEL_DEFAULT = "~/models/hqsb/Qwen3-1.7B"

# ───────────────────────── pre-registered protocol ─────────────────────────
# Frozen *before* any data collection; changing these invalidates the run id.
PROTOCOL: Dict[str, Any] = {
    "staged": {"input_tokens": 2048, "output_tokens": 32, "batch_size": 1},
    "context_sweep": {
        "output_tokens": 16,
        "batch_size": 1,
        "input_tokens": [32, 128, 512, 1024, 2048],
    },
    "batch_sweep": [
        {"input_tokens": 128, "output_tokens": 16, "batch_sizes": [1, 2, 4, 8, 16]},
        {"input_tokens": 512, "output_tokens": 16, "batch_sizes": [1, 2, 4]},
    ],
    "multi_request": {
        "input_tokens": 128,
        "output_tokens": 8,
        "batch_size": 1,
        "requests": 3,
    },
    "hook_ab": {"input_tokens": 512, "output_tokens": 8, "batch_size": 1},
    # Shape-keyed library workspace (cuBLAS/cuDNN keep per-problem-shape
    # buffers) makes the *first* point at a new (I, G, B) pay a one-off
    # allocation that never returns. Each measured point therefore runs one
    # unmeasured pass at exactly its own shape first, so the measured baseline
    # already has that workspace paid for. See the noise probe below, which
    # quantifies what remains.
    "shape_warmup_passes": 1,
    "noise_probe": {
        "known_alloc_mib": [16, 64, 256],
        "repeats": 2,
        "idle_seconds": 2.0,
        "idle_samples": 40,
        "idle_interval_s": 0.05,
    },
    "repeat_probe": {
        "input_tokens": 512,
        "output_tokens": 8,
        "batch_size": 1,
        "repeats": 2,
    },
    # Calibration vs held-out split (protocol §7: never fit and validate on the
    # same points). Held-out points are excluded from the fixed-overhead fit.
    "calibration_context_lengths": [32, 128, 512, 2048],
    "heldout_context_lengths": [1024],
    "calibration_batch_points": [
        [128, 1], [128, 2], [128, 4], [128, 8], [128, 16], [512, 1], [512, 2],
    ],
    "heldout_batch_points": [[512, 4]],
    "sampler_interval_s": 0.01,
    "input_mode": "duplicate_fixed_token_input",
    "dtype": "float16",
    "attention_backend": "eager",
    "clock_domain": "host_monotonic_ns + torch.cuda.synchronize",
    "kv_state_definition": "DynamicCache filled length = I + k at decode step k",
}

# Pre-registered tolerances and guards (protocol §8 step 7 / §9).
TOLERANCES: Dict[str, float] = {
    "kv_metadata_rel_error": 0.01,
    "unit_rel_error": 1e-9,
    # Allocator-level observation vs the analytic KV/peak model. Restricting the
    # *pass* check to points whose predicted KV exceeds 64 MiB keeps a ~30 MiB
    # system-memory wobble from dominating a 3.5 MiB prediction.
    "min_kv_for_observation_check_bytes": 64 * MIB,
    "steady_residual_ratio": 0.35,
    # Device free/total deltas on a unified-memory device carry a measured
    # observation floor (see ``noise_probe``); a point is only used for the
    # allocator-level cross-check when its predicted signal clears that floor,
    # and the residual allowance is ``ratio * predicted + noise_floor``.
    "min_signal_over_noise": 2.0,
    # A peak row counts as *structurally demonstrated* when the analytic
    # phase-max peak reproduces the device-sampled peak this closely. The check
    # requires at least ``min_demonstrated_points`` such rows, so the model must
    # be shown to work somewhere, not merely fail to be refuted everywhere.
    "peak_unresolved_ratio": 0.15,
    "min_demonstrated_points": 2,
    # Weight ledger (analytic storage-de-duplicated bytes) vs the device delta
    # M2-M1. The device delta is system-wide on unified memory and includes the
    # load transient and page tables, so a one-digit-percent band is expected.
    "weight_rel_error": 0.25,
    "heldout_rel_error": 0.40,
    "cross_run_rel_spread": 0.15,
    "cleanup_growth_bytes": 64 * MIB,
}

SAFETY = {
    "min_device_free_mb": 256.0,
    "thermal_suspect_c": 95.0,
}

# Frozen E02-04 OOM points (E02-04 report §4.4/§4.5). Re-used as the capacity
# cross-reference; E02-05 does not provoke new OOMs.
E02_04_OOM_POINTS: List[Dict[str, Any]] = [
    {"workload": "balanced", "input_tokens": 512, "output_tokens": 128, "batch_size": 16},
    {"workload": "long_prefill", "input_tokens": 2048, "output_tokens": 32, "batch_size": 4},
    {"workload": "long_balanced", "input_tokens": 2048, "output_tokens": 128, "batch_size": 4},
]


# ───────────────────────────── small helpers ─────────────────────────────


def device_used_bytes() -> Optional[int]:
    """Device bytes currently in use (unified memory: whole-system DRAM)."""
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(total_bytes - free_bytes)


def relative_error(predicted: float, observed: float) -> Optional[float]:
    if predicted == 0:
        return None if observed == 0 else float("inf")
    return abs(observed - predicted) / abs(predicted)


def allocator_view() -> Dict[str, Any]:
    """Allocator counters, which may all be zero under no-caching.

    Recorded (not assumed) so the report can state whether
    ``allocated/reserved/peak`` came from the allocator or had to be replaced by
    device free/total deltas.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    def _memory_mb() -> Dict[str, float]:
        return {
            "allocated_mb": torch.cuda.memory_allocated() / MIB,
            "reserved_mb": torch.cuda.memory_reserved() / MIB,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / MIB,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / MIB,
        }

    snapshot = _memory_mb()
    stats: Dict[str, Any] = {}
    try:
        raw = torch.cuda.memory_stats()
        for key in (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "active_bytes.all.current",
            "inactive_split_bytes.all.current",
            "num_alloc_retries",
            "num_ooms",
        ):
            if key in raw:
                stats[key] = raw[key]
    except Exception as exc:  # noqa: BLE001
        stats = {"error": str(exc)}
    snapshot["stats"] = stats
    snapshot["available"] = True
    snapshot["counters_nonzero"] = bool(
        snapshot["allocated_mb"] or snapshot["reserved_mb"]
    )
    return snapshot


def allocator_probe() -> Dict[str, Any]:
    """Functional check of whether the caching allocator pools memory.

    Allocates 64 MiB, frees it, then allocates a tiny tensor and re-reads the
    reserved counter. ``reserved`` staying > 0 after the free means the block was
    returned to a pool (caching enabled); ``reserved == 0`` throughout means
    either caching is disabled or the counters are unavailable — the raw numbers
    are kept so the report never guesses.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    result: Dict[str, Any] = {"available": True}
    try:
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        result["baseline"] = allocator_view()
        big = torch.empty(32 * 1024 * 1024, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        result["after_big_alloc"] = allocator_view()
        del big
        gc.collect()
        torch.cuda.synchronize()
        result["after_free"] = allocator_view()
        small = torch.empty(1, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        result["after_small_realloc"] = allocator_view()
        del small
        result["reserved_retained_after_free_mb"] = result["after_free"]["reserved_mb"]
        result["caching_pools_memory"] = bool(result["after_free"]["reserved_mb"] > 1.0)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def model_dims(model: Any) -> Dict[str, Any]:
    """Structural dimensions read off the loaded model's config."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        int(cfg.hidden_size) // int(cfg.num_attention_heads)
    )
    return {
        "num_layers": int(cfg.num_hidden_layers),
        "num_query_heads": int(cfg.num_attention_heads),
        "num_kv_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        "head_dim": int(head_dim),
        "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(cfg.intermediate_size),
        "vocab_size": int(cfg.vocab_size),
        "element_bytes": 2,
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
    }


def build_inputs(
    tokenizer: Any, isl: int, batch_size: int, device: str
) -> Dict[str, torch.Tensor]:
    """Real ``[B, I]`` copies via ``.repeat`` (never a broadcast view)."""
    single = make_fixed_token_input(tokenizer, isl, device=device)
    return {
        "input_ids": single["input_ids"].repeat(batch_size, 1),
        "attention_mask": single["attention_mask"].repeat(batch_size, 1),
    }


def point_theory(
    dims: Dict[str, Any], isl: int, osl: int, batch_size: int
) -> Dict[str, Any]:
    """Analytic memory terms for one (I, G, B) point, all in bytes."""
    layers = dims["num_layers"]
    hkv = dims["num_kv_heads"]
    hq = dims["num_query_heads"]
    dh = dims["head_dim"]
    element_bytes = dims["element_bytes"]

    kv_prefill = kv_ledger(
        num_layers=layers, num_kv_heads=hkv, head_dim=dh,
        element_bytes=element_bytes, batch_size=batch_size, context_length=isl,
    )
    kv_final = kv_ledger(
        num_layers=layers, num_kv_heads=hkv, head_dim=dh,
        element_bytes=element_bytes, batch_size=batch_size,
        context_length=isl + osl - 1,
    )
    workspace_prefill = eager_attention_workspace_bytes(
        batch_size=batch_size, num_query_heads=hq, seq_len=isl,
        score_bytes=element_bytes, softmax_bytes=4,
    )
    # The decode-time score matrix is [B, Hq, 1, T], so it is not square and
    # cannot reuse eager_attention_workspace_bytes' S x S shape.
    decode_cells = batch_size * hq * 1 * (isl + osl - 1)
    workspace_decode_tail = {
        "score_cells": decode_cells,
        "score_matrix_bytes": decode_cells * element_bytes,
        "softmax_matrix_bytes": decode_cells * 4,
        "total_bytes": decode_cells * (element_bytes + 4),
    }
    activation_prefill = activation_lower_bound_bytes(
        batch_size=batch_size, seq_len=isl,
        hidden_size=dims["hidden_size"], intermediate_size=dims["intermediate_size"],
        element_bytes=element_bytes,
    )
    activation_decode = activation_lower_bound_bytes(
        batch_size=batch_size, seq_len=1,
        hidden_size=dims["hidden_size"], intermediate_size=dims["intermediate_size"],
        element_bytes=element_bytes,
    )
    logits_full = batch_size * isl * dims["vocab_size"] * element_bytes
    logits_last = batch_size * 1 * dims["vocab_size"] * element_bytes

    # Peak = max over *mutually exclusive* moments, never a sum of every term
    # (protocol §5: "不能把每一层所有 activation 全部相加作为同时驻留量").
    # The reference forward frees the attention score/softmax buffers before
    # ``lm_head`` materializes the full-sequence logits, so the two are never
    # live together. Adding them double-counts and was refuted by the run-0
    # development data (see the E02-05 report §"peak model correction").
    kv_prefill_bytes = int(kv_prefill.total_bytes)
    kv_final_bytes = int(kv_final.total_bytes)
    prefill_attention_branch = kv_prefill_bytes + workspace_prefill["total_bytes"] + activation_prefill
    prefill_lm_head_branch = kv_prefill_bytes + logits_full
    decode_activation_branch = (
        kv_final_bytes + activation_decode + workspace_decode_tail["total_bytes"]
    )
    decode_lm_head_branch = kv_final_bytes + logits_last

    return {
        "kv_prefill": kv_prefill.as_dict(),
        "kv_final": kv_final.as_dict(),
        "attention_workspace_prefill": workspace_prefill,
        "attention_workspace_decode_tail": workspace_decode_tail,
        "activation_lower_bound_prefill_bytes": activation_prefill,
        "activation_lower_bound_prefill_mib": bytes_to_mib(activation_prefill),
        "activation_lower_bound_decode_bytes": activation_decode,
        "logits_full_prefill_bytes": logits_full,
        "logits_full_prefill_mib": bytes_to_mib(logits_full),
        "logits_last_bytes": logits_last,
        "peak_prefill_model": {
            "attention_branch_bytes": prefill_attention_branch,
            "lm_head_branch_bytes": prefill_lm_head_branch,
            "total_bytes": max(prefill_attention_branch, prefill_lm_head_branch),
            "winning_branch": (
                "lm_head"
                if prefill_lm_head_branch >= prefill_attention_branch
                else "attention_workspace"
            ),
        },
        "peak_decode_model": {
            "activation_branch_bytes": decode_activation_branch,
            "lm_head_branch_bytes": decode_lm_head_branch,
            "total_bytes": max(decode_activation_branch, decode_lm_head_branch),
            "winning_branch": (
                "lm_head"
                if decode_lm_head_branch >= decode_activation_branch
                else "activation_workspace"
            ),
        },
        "naive_sum_peak_prefill_bytes": (
            kv_prefill_bytes + logits_full + workspace_prefill["total_bytes"] + activation_prefill
        ),
        "token_budgets": {
            "input_compute_tokens": batch_size * isl,
            "reserved_token_capacity": batch_size * (isl + osl),
            "final_kv_tokens": batch_size * (isl + osl - 1),
        },
    }


def point_predictions(theory: Dict[str, Any]) -> Dict[str, int]:
    """Model predictions expressed as **increments** over the point baseline.

    The baseline already holds the resident weights, so none of these terms
    repeats ``W``. Every prediction is a conditional model of the *frozen*
    implementation (eager attention), never a general activation law.
    """
    return {
        "steady_prefill_bytes": int(theory["kv_prefill"]["total_bytes"]),
        "steady_final_bytes": int(theory["kv_final"]["total_bytes"]),
        "peak_prefill_bytes": int(theory["peak_prefill_model"]["total_bytes"]),
        "peak_decode_bytes": int(theory["peak_decode_model"]["total_bytes"]),
        "naive_sum_peak_prefill_bytes": int(theory["naive_sum_peak_prefill_bytes"]),
    }


# ─────────────────────────── one measurement point ───────────────────────────


@torch.inference_mode()
def measure_point(
    model: Any,
    tokenizer: Any,
    isl: int,
    osl: int,
    batch_size: int,
    device: str,
    sampler_interval_s: float,
    *,
    attach_hook: bool = False,
    label: str = "",
    warmup_passes: int = 0,
) -> Dict[str, Any]:
    """Run one prefill + ``G-1`` decode point with staged memory snapshots.

    Lifecycle (protocol §8 steps 3/4/5)::

        before → prefill_with_logits (M3: full-sequence logits live)
               → prefill_steady      (M3b: logits/activations released, KV live)
               → decode_end          (M4/M5: KV at I+G-1, outputs released)
               → released            (M6: request objects dropped)

    The ``prefill_with_logits`` → ``prefill_steady`` delta is the direct evidence
    for protocol §5.1: this implementation materializes logits for *every*
    prefill position, and releasing that tensor is measurable.

    Args:
        warmup_passes: Unmeasured passes at exactly this shape before the
            measured baseline. cuBLAS/cuDNN keep per-problem-shape buffers for
            the process lifetime, so the first point at a new shape pays a
            one-off allocation that would otherwise be misread as KV growth
            (protocol §5: "库 workspace 可能随 shape 离散跳变").
    """
    if osl < 1:
        raise ValueError(f"output_tokens must be >= 1, got {osl}")

    for index in range(int(warmup_passes)):
        logger.debug("shape warmup %d/%d for %s", index + 1, warmup_passes, label)
        measure_point(
            model, tokenizer, isl, osl, batch_size, device, sampler_interval_s,
            attach_hook=False, label=f"{label}#shape_warmup", warmup_passes=0,
        )
        gc.collect()

    inputs = build_inputs(tokenizer, isl, batch_size, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    collector = None
    if attach_hook:
        from hqsb.benchmark.shape_census import ShapeCensusCollector

        collector = ShapeCensusCollector()
        collector.attach(model)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    used_before = device_used_bytes()
    snapshots: Dict[str, Any] = {"before": memory_snapshot(label="before")}
    sampler = DeviceMemorySampler(sampler_interval_s)

    # ── Prefill ───────────────────────────────────────────────────
    sampler.start()
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_forward_ms = (time.perf_counter() - prefill_start) * 1000.0
    sampler.stop()
    prefill_peak = sampler.summary()
    sampler.reset()

    logits = outputs.logits
    logits_meta = {
        "shape": [int(s) for s in logits.shape],
        "dtype": str(logits.dtype).removeprefix("torch."),
        "contiguous": bool(logits.is_contiguous()),
        "observed_bytes": int(logits.numel() * logits.element_size()),
    }
    peak_allocated_mb = (
        torch.cuda.max_memory_allocated() / MIB if device == "cuda" else 0.0
    )
    peak_reserved_mb = (
        torch.cuda.max_memory_reserved() / MIB if device == "cuda" else 0.0
    )

    snapshots["prefill_with_logits"] = memory_snapshot(label="prefill_with_logits")
    kv_prefill_meta = kv_cache_metadata(outputs.past_key_values)

    # First-token selection: argmax only. Keeping the ``logits[:, -1, :]`` *view*
    # would pin the whole [B, I, vocab] buffer alive, which is exactly the
    # retention trap protocol §5.1 warns about.
    selection_start = time.perf_counter()
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    if device == "cuda":
        torch.cuda.synchronize()
    first_token_selection_ms = (time.perf_counter() - selection_start) * 1000.0
    generated: List[List[int]] = [[int(t)] for t in next_token.flatten().tolist()]

    del logits, outputs
    gc.collect()
    snapshots["prefill_steady"] = memory_snapshot(label="prefill_steady")

    # ── Decode ────────────────────────────────────────────────────
    itl_ms: List[float] = []
    step_ledger: List[Dict[str, Any]] = []
    decode_error: Optional[Dict[str, Any]] = None
    sampler.start()
    step = 0
    try:
        for step in range(1, osl):
            current_length = isl + step
            decode_mask = torch.ones(
                (batch_size, current_length), dtype=torch.long, device=device
            )
            if device == "cuda":
                torch.cuda.synchronize()
            decode_start = time.perf_counter()
            outputs = model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            if device == "cuda":
                torch.cuda.synchronize()
            itl_ms.append((time.perf_counter() - decode_start) * 1000.0)
            for row_index, token in enumerate(next_token.flatten().tolist()):
                generated[row_index].append(int(token))
            del outputs
            step_ledger.append(
                {
                    "step": step,
                    "filled_context": current_length,
                    "device_used_bytes": device_used_bytes(),
                }
            )
    except Exception as exc:  # noqa: BLE001 - record a failure, never discard it
        decode_error = {
            "step": step,
            "stage": "decode",
            "type": type(exc).__name__,
            "is_oom": "out of memory" in str(exc).lower(),
            "message": str(exc),
        }
        logger.warning("decode failed at step %d: %s", step, exc)
    sampler.stop()
    decode_peak = sampler.summary()

    kv_final_meta = kv_cache_metadata(past_key_values)
    snapshots["decode_end"] = memory_snapshot(label="decode_end")
    end_peak_allocated_mb = (
        torch.cuda.max_memory_allocated() / MIB if device == "cuda" else 0.0
    )
    end_peak_reserved_mb = (
        torch.cuda.max_memory_reserved() / MIB if device == "cuda" else 0.0
    )

    # M5: the frozen protocol retains generated token ids only (no per-position
    # logits, no hidden states), hence M5 == M4 by design.
    snapshots["outputs_retained"] = memory_snapshot(label="outputs_retained")

    del next_token, past_key_values, input_ids, attention_mask, inputs
    gc.collect()
    snapshots["released"] = memory_snapshot(label="released")
    if collector is not None:
        collector.detach()

    used_after_release = device_used_bytes()

    # ── Derived observations (increments over the point baseline) ──
    def _increment(snapshot: Dict[str, Any]) -> Optional[int]:
        used = snapshot_used_bytes(snapshot)
        if used is None or used_before is None:
            return None
        return used - used_before

    def _peak_increment(summary: Dict[str, float]) -> Optional[int]:
        if used_before is None or not summary.get("count"):
            return None
        return int(round((summary["max"] - used_before / MIB) * MIB))

    observations = {
        "device_used_before_bytes": used_before,
        "device_used_after_release_bytes": used_after_release,
        "prefill_with_logits_increment_bytes": _increment(
            snapshots["prefill_with_logits"]
        ),
        "prefill_steady_increment_bytes": _increment(snapshots["prefill_steady"]),
        "decode_end_increment_bytes": _increment(snapshots["decode_end"]),
        "released_increment_bytes": _increment(snapshots["released"]),
        "prefill_peak_increment_bytes": _peak_increment(prefill_peak),
        "decode_peak_increment_bytes": _peak_increment(decode_peak),
        "prefill_peak_summary": prefill_peak,
        "decode_peak_summary": decode_peak,
    }
    observations["logits_release_delta_bytes"] = (
        None
        if observations["prefill_with_logits_increment_bytes"] is None
        else observations["prefill_with_logits_increment_bytes"]
        - (observations["prefill_steady_increment_bytes"] or 0)
    )

    phases = model_core_timings(prefill_forward_ms, first_token_selection_ms, itl_ms)
    row_hashes = [hash_token_sequence(row) for row in generated]

    point: Dict[str, Any] = {
        "label": label,
        "spec": {
            "input_tokens": isl,
            "output_tokens": osl,
            "batch_size": batch_size,
            "input_mode": PROTOCOL["input_mode"],
        },
        "theory": point_theory(
            dims=model_dims(model), isl=isl, osl=osl, batch_size=batch_size
        ),
        "snapshots": snapshots,
        "observations": observations,
        "kv_metadata": {"prefill": kv_prefill_meta, "final": kv_final_meta},
        "logits": logits_meta,
        "timings": {
            "prefill_forward_ms": prefill_forward_ms,
            "first_token_selection_ms": first_token_selection_ms,
            **phases,
            "itl_count": len(itl_ms),
            "itl_p50_ms": statistics.median(itl_ms) if itl_ms else None,
            "itl_mean_ms": (sum(itl_ms) / len(itl_ms)) if itl_ms else None,
            "raw_itl_ms": itl_ms,
        },
        "generated": {
            "rows": len(generated),
            "row_lengths": [len(row) for row in generated],
            "row_hashes": row_hashes,
            "distinct_row_hashes": len(set(row_hashes)),
            "token_ids": (
                generated if batch_size * max(osl, 1) <= 4096 else "omitted"
            ),
        },
        "decode": {"step_ledger": step_ledger, "error": decode_error},
        "allocator": {
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "end_peak_allocated_mb": end_peak_allocated_mb,
            "end_peak_reserved_mb": end_peak_reserved_mb,
            "snapshot": allocator_view(),
        },
        "rss_swap": {
            "process_rss_bytes": process_rss_bytes(),
            "process_swap_bytes": process_swap_bytes(),
        },
        "protocol": {
            "sampler_interval_s": sampler_interval_s,
            "attach_hook": attach_hook,
        },
    }

    predictions = point_predictions(point["theory"])
    point["predictions"] = predictions
    point["decomposition"] = {
        "steady_prefill": decompose(
            predicted={"kv_prefill": predictions["steady_prefill_bytes"]},
            observed_bytes=observations["prefill_steady_increment_bytes"],
        ),
        "steady_final": decompose(
            predicted={"kv_final": predictions["steady_final_bytes"]},
            observed_bytes=observations["decode_end_increment_bytes"],
        ),
        "peak_prefill": decompose(
            predicted={
                "kv_prefill": predictions["steady_prefill_bytes"],
                "exclusive_phase_peak": (
                    predictions["peak_prefill_bytes"]
                    - predictions["steady_prefill_bytes"]
                ),
            },
            observed_bytes=observations["prefill_peak_increment_bytes"],
        ),
        "peak_decode": decompose(
            predicted={
                "kv_final": predictions["steady_final_bytes"],
                "exclusive_phase_peak": (
                    predictions["peak_decode_bytes"]
                    - predictions["steady_final_bytes"]
                ),
            },
            observed_bytes=observations["decode_peak_increment_bytes"],
        ),
    }
    point["peak_model"] = {
        "prefill_attention_branch_bytes": point["theory"]["peak_prefill_model"][
            "attention_branch_bytes"
        ],
        "prefill_lm_head_branch_bytes": point["theory"]["peak_prefill_model"][
            "lm_head_branch_bytes"
        ],
        "prefill_winning_branch": point["theory"]["peak_prefill_model"][
            "winning_branch"
        ],
        "decode_winning_branch": point["theory"]["peak_decode_model"][
            "winning_branch"
        ],
        "naive_sum_peak_prefill_bytes": point["theory"][
            "naive_sum_peak_prefill_bytes"
        ],
        "naive_sum_minus_phase_max_bytes": (
            point["theory"]["naive_sum_peak_prefill_bytes"]
            - point["theory"]["peak_prefill_model"]["total_bytes"]
        ),
    }
    point["kv_checks"] = {
        "prefill_metadata_rel_error": relative_error(
            point["theory"]["kv_prefill"]["total_bytes"],
            kv_prefill_meta["total_logical_bytes"],
        ),
        "final_metadata_rel_error": relative_error(
            point["theory"]["kv_final"]["total_bytes"],
            kv_final_meta["total_logical_bytes"],
        ),
        "metadata_layers": kv_final_meta["num_layers"],
        "metadata_kv_heads": kv_final_meta["kv_heads_observed"],
        "metadata_element_bytes": kv_final_meta["element_bytes_observed"],
        "metadata_per_token_per_layer_bytes": kv_final_meta[
            "per_token_per_layer_bytes_observed"
        ],
        "metadata_preallocated": kv_final_meta["appears_preallocated"],
        "observed_logits_bytes": logits_meta["observed_bytes"],
        "theory_logits_bytes": point["theory"]["logits_full_prefill_bytes"],
        "logits_rel_error": relative_error(
            point["theory"]["logits_full_prefill_bytes"], logits_meta["observed_bytes"]
        ),
        "output_tokens_exact": all(len(row) == osl for row in generated),
        "rows_present": len(generated) == batch_size,
        "decode_error": decode_error,
    }
    return point


def point_key(point: Dict[str, Any]) -> str:
    spec = point["spec"]
    return f"I{spec['input_tokens']}_G{spec['output_tokens']}_B{spec['batch_size']}"


def iter_points(record: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(section, point)`` for every measured point of one run record."""
    if record.get("staged_point"):
        yield "staged", record["staged_point"]
    for point in record.get("context_sweep", []):
        yield "context_sweep", point
    for point in record.get("batch_sweep", []):
        yield "batch_sweep", point


# ───────────────────── multi-request and hook A/B probes ─────────────────────


def run_multi_request(
    model: Any,
    tokenizer: Any,
    device: str,
    spec: Dict[str, Any],
    sampler_interval_s: float,
) -> Dict[str, Any]:
    """Repeat one identical request, releasing state between repetitions.

    Protocol §8 step 9: ``reserved`` not returning to its initial value is not
    automatically a leak — growth of *live* objects and ``allocated`` across
    requests is the real signal. On unified memory the device free/total delta is
    the equivalent observable.
    """
    records: List[Dict[str, Any]] = []
    for index in range(int(spec["requests"])):
        point = measure_point(
            model, tokenizer,
            spec["input_tokens"], spec["output_tokens"], spec["batch_size"],
            device, sampler_interval_s, label=f"multi_request_{index}",
        )
        records.append(
            {
                "request_index": index,
                "device_used_before_bytes": point["observations"][
                    "device_used_before_bytes"
                ],
                "device_used_after_release_bytes": point["observations"][
                    "device_used_after_release_bytes"
                ],
                "released_increment_bytes": point["observations"][
                    "released_increment_bytes"
                ],
                "prefill_steady_increment_bytes": point["observations"][
                    "prefill_steady_increment_bytes"
                ],
                "decode_end_increment_bytes": point["observations"][
                    "decode_end_increment_bytes"
                ],
                "predicted_steady_final_bytes": point["predictions"][
                    "steady_final_bytes"
                ],
                "process_rss_bytes": point["rss_swap"]["process_rss_bytes"],
                "process_swap_bytes": point["rss_swap"]["process_swap_bytes"],
                "generated_row_hashes": point["generated"]["row_hashes"],
            }
        )

    growth = None
    if len(records) >= 2:
        first = records[0]["device_used_before_bytes"]
        last = records[-1]["device_used_before_bytes"]
        if first is not None and last is not None:
            growth = last - first
    return {
        "spec": spec,
        "records": records,
        "device_used_growth_bytes_first_to_last": growth,
        "regression_to_baseline": (
            None if growth is None else abs(growth) <= TOLERANCES["cleanup_growth_bytes"]
        ),
    }


def run_hook_ab(
    model: Any,
    tokenizer: Any,
    device: str,
    spec: Dict[str, Any],
    sampler_interval_s: float,
) -> Dict[str, Any]:
    """Protocol §8 step 8: does the census hook retain tensors?

    Runs the same point with and without ``ShapeCensusCollector`` attached. The
    collector stores shape/stride *strings*, so a near-zero delta is expected; a
    large delta would mean the hook is itself the memory event under study.
    """
    plain = measure_point(
        model, tokenizer, spec["input_tokens"], spec["output_tokens"],
        spec["batch_size"], device, sampler_interval_s,
        attach_hook=False, label="hook_ab_without",
    )
    hooked = measure_point(
        model, tokenizer, spec["input_tokens"], spec["output_tokens"],
        spec["batch_size"], device, sampler_interval_s,
        attach_hook=True, label="hook_ab_with",
    )

    def _delta(key: str) -> Optional[int]:
        a = plain["observations"].get(key)
        b = hooked["observations"].get(key)
        if a is None or b is None:
            return None
        return b - a

    return {
        "spec": spec,
        "without_hook": {
            "prefill_peak_increment_bytes": plain["observations"][
                "prefill_peak_increment_bytes"
            ],
            "prefill_steady_increment_bytes": plain["observations"][
                "prefill_steady_increment_bytes"
            ],
        },
        "with_hook": {
            "prefill_peak_increment_bytes": hooked["observations"][
                "prefill_peak_increment_bytes"
            ],
            "prefill_steady_increment_bytes": hooked["observations"][
                "prefill_steady_increment_bytes"
            ],
        },
        "peak_delta_bytes": _delta("prefill_peak_increment_bytes"),
        "steady_delta_bytes": _delta("prefill_steady_increment_bytes"),
    }


def noise_probe(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Quantify the device free/total observation floor (protocol §8 step 4).

    Two components are measured, both needed to read the sweep honestly:

    1. **Idle span** — how much ``total - free`` wanders while the process does
       nothing. On a unified-memory device this absorbs page-cache reclaim from
       unrelated processes, so it is a floor on any single-shot delta.
    2. **Known-size recovery** — allocate a known tensor, free it, and read the
       delta. ``rise - size`` is the visibility error; ``net`` (post-free minus
       pre-alloc) is a *bias*, not noise: a persistent positive net means the
       unified-memory accounting does not hand the pages straight back.

    ``noise_floor_bytes`` is the maximum of the idle span and the median absolute
    net bias, so a residual smaller than it can never be called a model error.
    """
    result: Dict[str, Any] = {"available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return result

    idle_values: List[float] = []
    deadline = time.monotonic() + float(spec["idle_seconds"])
    while time.monotonic() < deadline and len(idle_values) < int(spec["idle_samples"]):
        used = device_used_bytes()
        if used is not None:
            idle_values.append(used / MIB)
        time.sleep(float(spec["idle_interval_s"]))
    idle_span = (max(idle_values) - min(idle_values)) if idle_values else 0.0

    # Each allocation is done twice per repetition: once as a bare
    # ``torch.empty`` (virtual reservation only) and once written with
    # ``fill_`` (physical commit). On a unified-memory device ``cudaMemGetInfo``
    # reports *committed* pages, so the two cases differ enormously and the
    # contrast is the evidence for how coarse the device view is.
    allocations: List[Dict[str, Any]] = []
    for size_mib in spec["known_alloc_mib"]:
        for _ in range(int(spec["repeats"])):
            elements = int(size_mib * MIB / 2)  # fp16
            expected = int(size_mib * MIB)

            before = device_used_bytes()
            tensor = torch.empty(elements, dtype=torch.float16, device="cuda")
            torch.cuda.synchronize()
            after_empty = device_used_bytes()
            tensor.fill_(0)
            torch.cuda.synchronize()
            after_fill = device_used_bytes()
            del tensor
            gc.collect()
            torch.cuda.synchronize()
            freed = device_used_bytes()
            if None in (before, after_empty, after_fill, freed):
                continue
            allocations.append(
                {
                    "size_mib": size_mib,
                    "empty_rise_bytes": after_empty - before,
                    "fill_rise_bytes": after_fill - before,
                    "fall_bytes": after_fill - freed,
                    "net_bytes": freed - before,
                    "empty_rise_error_bytes": (after_empty - before) - expected,
                    "fill_rise_error_bytes": (after_fill - before) - expected,
                }
            )

    net_values = [abs(a["net_bytes"]) for a in allocations]
    fill_errors = [a["fill_rise_error_bytes"] for a in allocations]
    empty_errors = [a["empty_rise_error_bytes"] for a in allocations]
    noise_floor = max(
        idle_span * MIB,
        statistics.median(net_values) if net_values else 0.0,
    )
    result.update(
        {
            "idle_samples": len(idle_values),
            "idle_span_bytes": int(idle_span * MIB),
            "allocations": allocations,
            "median_abs_net_bytes": (
                int(statistics.median(net_values)) if net_values else None
            ),
            "median_abs_fill_rise_error_bytes": (
                int(statistics.median([abs(e) for e in fill_errors]))
                if fill_errors
                else None
            ),
            "median_abs_empty_rise_error_bytes": (
                int(statistics.median([abs(e) for e in empty_errors]))
                if empty_errors
                else None
            ),
            # The floor used to excuse a residual is the *idle* wander plus the
            # return-to-baseline bias only. The rise error is reported
            # separately: it characterises how well the device view sees an
            # allocation at all, and is quoted as a limitation, not used to
            # loosen the model check.
            "noise_floor_bytes": int(noise_floor),
        }
    )
    return result


def run_repeat_probe(
    model: Any,
    tokenizer: Any,
    device: str,
    spec: Dict[str, Any],
    sampler_interval_s: float,
    *,
    warmup_passes: int = 0,
) -> Dict[str, Any]:
    """Run the same point twice; the repeatability of the observation is the
    per-point error bar the sweep must be judged against."""
    points = [
        measure_point(
            model, tokenizer, spec["input_tokens"], spec["output_tokens"],
            spec["batch_size"], device, sampler_interval_s,
            label=f"repeat_{index}", warmup_passes=warmup_passes,
        )
        for index in range(int(spec["repeats"]))
    ]

    def _values(key: str) -> List[Optional[int]]:
        return [point["observations"][key] for point in points]

    steady = _values("prefill_steady_increment_bytes")
    decode = _values("decode_end_increment_bytes")
    peak = _values("prefill_peak_increment_bytes")

    def _spread(values: List[Optional[int]]) -> Optional[int]:
        clean = [v for v in values if v is not None]
        return (max(clean) - min(clean)) if len(clean) >= 2 else None

    return {
        "spec": spec,
        "warmup_passes": warmup_passes,
        "prefill_steady_increment_bytes": steady,
        "decode_end_increment_bytes": decode,
        "prefill_peak_increment_bytes": peak,
        "prefill_steady_spread_bytes": _spread(steady),
        "decode_end_spread_bytes": _spread(decode),
        "prefill_peak_spread_bytes": _spread(peak),
        "predicted_steady_final_bytes": points[0]["predictions"]["steady_final_bytes"],
        "max_spread_bytes": max(
            v for v in (
                _spread(steady), _spread(decode), _spread(peak),
            ) if v is not None
        )
        if any(v is not None for v in (_spread(steady), _spread(decode), _spread(peak)))
        else None,
    }


def oom_cross_reference(
    dims: Dict[str, Any], available_after_load_bytes: Optional[int]
) -> Dict[str, Any]:
    """Map the frozen E02-04 OOM points onto this memory model (protocol step 11).

    No OOM is provoked here. The analytic per-point work set is compared with the
    memory actually available after loading the model in *this* run, plus a
    falsifiable variant where eager attention stops materializing [B, Hq, S, S].
    """
    rows: List[Dict[str, Any]] = []
    for frozen in E02_04_OOM_POINTS:
        isl = frozen["input_tokens"]
        osl = frozen["output_tokens"]
        batch = frozen["batch_size"]
        theory = point_theory(dims, isl, osl, batch)
        peak = point_predictions(theory)["peak_prefill_bytes"]
        without_eager = int(
            theory["kv_prefill"]["total_bytes"]
            + theory["logits_full_prefill_bytes"]
            + theory["activation_lower_bound_prefill_bytes"]
        )
        rows.append(
            {
                **frozen,
                "kv_prefill_bytes": theory["kv_prefill"]["total_bytes"],
                "attention_workspace_prefill_bytes": theory[
                    "attention_workspace_prefill"
                ]["total_bytes"],
                "activation_lower_bound_bytes": theory[
                    "activation_lower_bound_prefill_bytes"
                ],
                "logits_full_bytes": theory["logits_full_prefill_bytes"],
                "predicted_prefill_peak_increment_bytes": peak,
                "predicted_peak_without_eager_softmax_bytes": without_eager,
                "available_after_load_bytes": available_after_load_bytes,
                "predicted_over_available": (
                    None
                    if available_after_load_bytes is None
                    else peak > available_after_load_bytes
                ),
                "without_eager_would_fit": (
                    None
                    if available_after_load_bytes is None
                    else without_eager <= available_after_load_bytes
                ),
            }
        )
    return {
        "source": (
            "docs/stage_experiments/S02/E02-04/E02-04_实验报告.md §4.4/§4.5 "
            "(frozen values; E02-05 provokes no new OOM)"
        ),
        "rows": rows,
    }


# ───────────────────────── independent unit self-test ─────────────────────────


def unit_selftest() -> Dict[str, Any]:
    """Hand-computed checks of the predictor (protocol §8 step 2).

    Expected values come from the protocol's worked examples and hardcoded
    literals — *not* from a call to the function under test.
    """
    checks: Dict[str, Any] = {}
    ledger = kv_ledger(
        num_layers=28, num_kv_heads=8, head_dim=128,
        element_bytes=2, batch_size=1, context_length=2048,
    )
    checks["per_token_per_layer_bytes"] = {
        "expected": 4096, "actual": ledger.per_token_per_layer_bytes,
        "ok": ledger.per_token_per_layer_bytes == 4096,
    }
    checks["per_token_all_layers_bytes"] = {
        "expected": 114688, "actual": ledger.per_token_all_layers_bytes,
        "ok": ledger.per_token_all_layers_bytes == 114688,
    }
    checks["b1_t2048_total_bytes"] = {
        "expected": 234881024, "actual": ledger.total_bytes,
        "ok": ledger.total_bytes == 234881024,
    }
    checks["b1_t2048_mib"] = {
        "expected": 224.0, "actual": bytes_to_mib(ledger.total_bytes),
        "ok": abs(bytes_to_mib(ledger.total_bytes) - 224.0) < 1e-9,
    }
    filled = kv_ledger(
        num_layers=28, num_kv_heads=8, head_dim=128,
        element_bytes=2, batch_size=1, context_length=2175,
    )
    checks["filled_2175_mib"] = {
        "expected": 237.890625, "actual": bytes_to_mib(filled.total_bytes),
        "ok": abs(bytes_to_mib(filled.total_bytes) - 237.890625) < 1e-6,
    }
    wrong = kv_ledger(
        num_layers=28, num_kv_heads=16, head_dim=128,
        element_bytes=2, batch_size=1, context_length=2048,
    )
    checks["hq_would_double"] = {
        "expected": 2 * ledger.total_bytes, "actual": wrong.total_bytes,
        "ok": wrong.total_bytes == 2 * ledger.total_bytes,
    }
    checks["mib_not_mb"] = {
        "expected": 234.881024,
        "actual": bytes_to_decimal_mb(ledger.total_bytes),
        "ok": abs(bytes_to_decimal_mb(ledger.total_bytes) - 234.881024) < 1e-6,
    }
    checks["one_gib_in_units"] = {
        "expected": [1.0, 1073.741824],
        "actual": [bytes_to_gib(GIB), bytes_to_decimal_mb(GIB)],
        "ok": abs(bytes_to_gib(GIB) - 1.0) < 1e-12
        and abs(bytes_to_decimal_mb(GIB) - 1073.741824) < 1e-9,
    }
    workspace = eager_attention_workspace_bytes(
        batch_size=4, num_query_heads=16, seq_len=2048
    )
    checks["eager_workspace_b4_s2048_bytes"] = {
        "expected": 1610612736, "actual": workspace["total_bytes"],
        "ok": workspace["total_bytes"] == 1610612736,
    }

    # Peak model structure: the staged point is the protocol's worked example.
    # Hand-computed (B=1, I=2048, G=32, Qwen3-1.7B):
    #   kv_prefill = 114688 * 2048 = 234881024
    #   attention  = kv + 402653184 (fp16 score + fp32 softmax) + 33554432 = 671088640
    #   lm_head    = kv + 2048 * 151936 * 2 = 234881024 + 622329856  = 857210880
    #   phase max  = 857210880 (lm_head branch wins)
    #   naive sum  = 1293418496 (refuted: 477 MiB of double counting)
    staged = point_theory(
        {
            "num_layers": 28, "num_query_heads": 16, "num_kv_heads": 8,
            "head_dim": 128, "hidden_size": 2048, "intermediate_size": 6144,
            "vocab_size": 151936, "element_bytes": 2, "tie_word_embeddings": True,
        },
        2048, 32, 1,
    )
    checks["staged_peak_phase_max_bytes"] = {
        "expected": 857210880,
        "actual": staged["peak_prefill_model"]["total_bytes"],
        "ok": staged["peak_prefill_model"]["total_bytes"] == 857210880,
    }
    checks["staged_peak_winning_branch"] = {
        "expected": "lm_head",
        "actual": staged["peak_prefill_model"]["winning_branch"],
        "ok": staged["peak_prefill_model"]["winning_branch"] == "lm_head",
    }
    checks["staged_naive_sum_overprediction_bytes"] = {
        "expected": 436207616,  # 1293418496 - 857210880
        "actual": staged["naive_sum_peak_prefill_bytes"]
        - staged["peak_prefill_model"]["total_bytes"],
        "ok": staged["naive_sum_peak_prefill_bytes"]
        - staged["peak_prefill_model"]["total_bytes"]
        == 436207616,
    }
    return {"checks": checks, "all_ok": all(c["ok"] for c in checks.values())}
