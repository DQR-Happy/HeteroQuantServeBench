"""E14-F3 — long context: KV, chunked prefill, capacity × quality × latency.

Conditional P0: the main branch only when ``E14-05`` selects F3, otherwise
``N/A_BY_ADR``.  "Supports 128K" usually means a configuration number, not that
the machine can run it, that every token is processed, that the information is
retrievable, or that short requests are not starved.  The experiment is therefore
five-dimensional:

```text
real token semantics × KV/attention capacity × long-context quality
× prefill/decode latency × mixed-service fairness
```

Two rules drive the interfaces here:

* **no truncation, ever.** ``submitted`` / ``accepted`` / ``processed`` /
  ``truncated`` are separate counters, and a non-zero ``truncated`` disqualifies
  the episode as a capacity result (§10: 截断/滑窗/fallback 冒充完整上下文);
* **KV bytes are three numbers**, not one: ``logical`` (token geometry),
  ``physical`` (blocks/alignment) and ``metadata`` (scales/zeros).  Reporting a
  single total is how a metadata-heavy quantisation looks like a saving (§3.1).

Interfaces provided:

* :func:`kv_bytes_per_token` / :func:`kv_capacity` — the theoretical model that
  must be reconciled against the allocator (steps 10, 15);
* :class:`LongContextEpisode` — the §7 record with all four token counters and
  the three KV numbers (steps 5, 14);
* :class:`KvAllocatorTimeline` / :func:`calibrate_bytes_per_token` — measured vs
  theoretical, with block waste and fragmentation (steps 15–16);
* :func:`check_no_truncation` — the proof obligation, from trace/positions/mask
  rather than from an API's success (step 14);
* :func:`verify_position_and_mask` — chunk boundaries, RoPE positions and causal
  mask (steps 18, 21);
* :func:`kv_quant_report` / :func:`eviction_audit` / :func:`chunked_prefill_audit`
  (steps 19–21);
* :func:`quality_by_length_position` — needle depth × document length (step 26);
* :func:`mixed_load_fairness` / :func:`admission_budget` (steps 29–31);
* :func:`over_limit_rejection`, :func:`corrupted_kv_metadata_check`,
  :func:`cancel_release` (steps 33, 35–36);
* :class:`F3AdoptionDecision` (step 40).

Nothing here allocates KV or runs attention; the geometry is computed from
supplied shapes so the allocator report can be checked against the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import AdoptionDecision
from hqsb.experimental.identity import SCHEMA_PREFIX

EXPERIMENT_ID = "E14-F3"
TITLE = "长上下文 KV、Chunked Prefill、容量—质量—延迟"
LEVEL = "条件 P0"

CLAIM_BOUNDARY = (
    "只在 F3 为唯一主分支时成立；容量提升必须同时给出质量与延迟，截断/滑窗/OOM fallback 不算支持。"
    "允许合法的“容量显著改善但性能无收益”结论（E14-F3 §12）。"
)

#: The mechanisms F3 must choose exactly one of (§1).
INTERVENTIONS: Tuple[str, ...] = ("kv_quant", "kv_compress", "kv_evict", "chunked_prefill")

#: Byte sizes a KV storage element may use.
KV_STORAGE_BYTES: Tuple[str, ...] = ("fp16", "bf16", "fp8", "int8", "int4", "int2", "mixed")

#: Quality tasks that long context must use, instead of short-input perplexity.
LONG_CONTEXT_TASKS: Tuple[str, ...] = ("needle_retrieval", "long_qa", "summarisation", "code_repo", "perplexity")

#: Length buckets; `near_limit` is where OOM margin matters (step 24).
LENGTH_BUCKETS: Tuple[str, ...] = ("short", "medium", "long", "near_limit")

#: Admission/budget dimensions a capacity model must update when KV shrinks (step 31).
ADMISSION_BUDGETS: Tuple[str, ...] = ("max_tokens", "max_kv_bytes", "max_concurrency", "safety_margin_bytes")


def kv_bytes_per_token(
    *,
    layers: int,
    kv_heads_per_rank: int,
    head_dim: int,
    storage: str,
    storage_bytes_override: Optional[int] = None,
    scale_group_size: int = 0,
    metadata_bytes_per_group: int = 0,
) -> Dict[str, Any]:
    """The per-token KV geometry of §3.1, with the metadata made explicit.

    ``layers × 2(K,V) × kv_heads × head_dim × bytes_per_element`` is the standard
    formula; the metadata term is what a quantised layout forgets.  The function
    returns both, so ``bytes_per_token`` cannot be quoted without the metadata.
    """
    sizes = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 1, "int2": 1}
    if storage not in KV_STORAGE_BYTES:
        raise ConfigError(f"unknown storage {storage!r}; known: {', '.join(KV_STORAGE_BYTES)}")
    if storage_bytes_override is not None:
        if storage_bytes_override <= 0:
            raise ConfigError("storage_bytes_override must be positive")
        element_bytes = int(storage_bytes_override)
    elif storage == "mixed":
        # A mixed layout (e.g. int4 keys + int2 values) has no single element size,
        # so it must state its own bytes rather than silently inheriting one of them.
        raise ConfigError(
            "storage 'mixed' has no single element size; pass storage_bytes_override "
            "(and record the per-key/value layout in kv_quant_report)"
        )
    else:
        element_bytes = sizes[storage]
    if layers <= 0 or kv_heads_per_rank <= 0 or head_dim <= 0:
        raise ConfigError("layers, kv_heads_per_rank and head_dim must be positive")
    # int4/int2 are packed: two/four values share a byte, so the per-element cost
    # is fractional and must not be rounded down to 1.
    if storage in ("int4", "int2") and storage_bytes_override is None:
        divisor = 2 if storage == "int4" else 4
        element_bytes_float = 1.0 / divisor
    else:
        element_bytes_float = float(element_bytes)
    payload = layers * 2 * kv_heads_per_rank * head_dim * element_bytes_float
    metadata = 0.0
    if scale_group_size > 0 and metadata_bytes_per_group > 0:
        groups = (layers * 2 * kv_heads_per_rank * head_dim) / scale_group_size
        metadata = groups * metadata_bytes_per_group
    return {
        "storage": storage,
        "payload_bytes_per_token": payload,
        "metadata_bytes_per_token": metadata,
        "bytes_per_token": payload + metadata,
        "metadata_ratio": metadata / (payload + metadata) if (payload + metadata) else 0.0,
        "note": "只看 payload 会把 scale/zero 的代价藏起来（step 19）",
    }


def kv_capacity(
    *,
    bytes_per_token: float,
    available_bytes: int,
    batch: int,
    block_tokens: int = 16,
    block_overhead_ratio: float = 0.0,
) -> Dict[str, Any]:
    """Step 24: context × concurrency from the *measured* bytes/token, not the model's."""
    if bytes_per_token <= 0:
        raise ConfigError("bytes_per_token must be positive")
    if batch <= 0:
        raise ConfigError("batch must be positive")
    usable = available_bytes * (1.0 - max(0.0, block_overhead_ratio))
    max_tokens_total = usable / bytes_per_token
    max_context_per_request = max_tokens_total / batch
    blocks = int(max_tokens_total // block_tokens) if block_tokens > 0 else 0
    return {
        "bytes_per_token": bytes_per_token,
        "available_bytes": available_bytes,
        "usable_bytes": usable,
        "batch": batch,
        "max_tokens_total": max_tokens_total,
        "max_context_per_request": max_context_per_request,
        "blocks": blocks,
        "block_tokens": block_tokens,
        "block_overhead_ratio": block_overhead_ratio,
        "note": "单请求最大长度不等于服务容量（step 24）",
    }


@dataclass
class LongContextEpisode:
    """``E14-F3`` §7 ``LongContextEpisode`` (schema ``…e14-f3.episode.v1``)."""

    episode_id: str
    model_artifact_id: str
    candidate_config_id: str
    submitted_tokens: int = 0
    accepted_tokens: int = 0
    processed_tokens: int = 0
    truncated_tokens: int = 0
    output_tokens: int = 0
    kv_logical_bytes: int = 0
    kv_physical_bytes: int = 0
    kv_metadata_bytes: int = 0
    prefill_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    quality: Mapping[str, float] = field(default_factory=dict)
    actual_backend: str = ""
    length_bucket: str = ""
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e14-f3.episode.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        for name in ("episode_id", "model_artifact_id", "candidate_config_id"):
            if not getattr(self, name):
                findings.append(f"LongContextEpisode: {name} is required")
        if self.submitted_tokens <= 0:
            findings.append("LongContextEpisode: submitted_tokens must be positive")
        if self.accepted_tokens > self.submitted_tokens:
            findings.append("LongContextEpisode: accepted_tokens exceeds submitted_tokens")
        if self.truncated_tokens:
            findings.append(
                f"LongContextEpisode: {self.truncated_tokens} tokens were truncated — "
                "截断/滑窗/fallback 冒充完整上下文是 FAIL（§10）"
            )
        if self.processed_tokens != self.accepted_tokens - self.truncated_tokens:
            findings.append(
                f"LongContextEpisode: processed_tokens {self.processed_tokens} != accepted - truncated "
                f"({self.accepted_tokens - self.truncated_tokens})"
            )
        if self.kv_physical_bytes and self.kv_logical_bytes > self.kv_physical_bytes:
            findings.append("LongContextEpisode: logical KV exceeds physical KV")
        if not self.kv_physical_bytes and self.status == rec.STATUS_PASS:
            findings.append("LongContextEpisode: PASS without a physical KV measurement")
        if not self.actual_backend:
            findings.append("LongContextEpisode: actual_backend is required (缺失即 INVALID_IDENTITY)")
        if self.length_bucket and self.length_bucket not in LENGTH_BUCKETS:
            findings.append(
                f"LongContextEpisode: length_bucket {self.length_bucket!r} must be one of "
                f"{', '.join(LENGTH_BUCKETS)}"
            )
        total = self.kv_logical_bytes + self.kv_metadata_bytes
        if self.kv_physical_bytes and total and self.kv_metadata_bytes / max(1, total) > 1.0:
            findings.append("LongContextEpisode: metadata exceeds the payload; the KV format is a regression")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "model_artifact_id": self.model_artifact_id,
            "candidate_config_id": self.candidate_config_id,
            "submitted_tokens": self.submitted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "processed_tokens": self.processed_tokens,
            "truncated_tokens": self.truncated_tokens,
            "output_tokens": self.output_tokens,
            "kv_logical_bytes": self.kv_logical_bytes,
            "kv_physical_bytes": self.kv_physical_bytes,
            "kv_metadata_bytes": self.kv_metadata_bytes,
            "prefill_ms": self.prefill_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "quality": dict(sorted(self.quality.items())),
            "actual_backend": self.actual_backend,
            "length_bucket": self.length_bucket,
            "status": self.status,
        }


def check_no_truncation(episode: LongContextEpisode, *, trace_processed_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Step 14: prove from the trace that every accepted token reached attention.

    The counters are compared against the runtime trace, because an API returning
    success while a sliding window dropped history is exactly the failure this
    step exists to catch (§10).
    """
    problems: List[str] = []
    if episode.truncated_tokens:
        problems.append("truncated_tokens is non-zero")
    if trace_processed_tokens is None:
        problems.append("no runtime trace token count supplied: the claim is unverified")
    elif trace_processed_tokens != episode.processed_tokens:
        problems.append(
            f"trace processed {trace_processed_tokens} tokens but the episode reports "
            f"{episode.processed_tokens}"
        )
    if episode.accepted_tokens != episode.submitted_tokens and episode.truncated_tokens == 0:
        problems.append("accepted < submitted with no truncation recorded: the gap is unexplained")
    return {
        "submitted": episode.submitted_tokens,
        "accepted": episode.accepted_tokens,
        "processed": episode.processed_tokens,
        "truncated": episode.truncated_tokens,
        "verified": not problems,
        "problems": problems,
    }


@dataclass(frozen=True)
class KvAllocatorTimeline:
    """Step 16: logical vs physical vs metadata, plus block waste and margin."""

    episode_id: str
    samples: Tuple[Mapping[str, Any], ...] = ()
    block_tokens: int = 16
    block_bytes: int = 0

    REQUIRED_SAMPLE_KEYS: Tuple[str, ...] = (
        "t_ns",
        "logical_bytes",
        "physical_bytes",
        "metadata_bytes",
        "blocks",
        "free_blocks",
        "fragmentation",
        "oom_margin_bytes",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.samples:
            findings.append("allocator timeline has no samples")
        for index, sample in enumerate(self.samples):
            missing = [key for key in self.REQUIRED_SAMPLE_KEYS if key not in sample]
            if missing:
                findings.append(f"allocator sample {index} is missing {', '.join(missing)}")
            if sample.get("logical_bytes", 0) > sample.get("physical_bytes", 0):
                findings.append(f"allocator sample {index}: logical > physical")
        return findings

    def tail_waste(self) -> Dict[str, Any]:
        """Block-granularity waste: the last block of a request is usually partial."""
        if not self.samples or not self.block_bytes:
            return {"error": "no samples or unknown block size"}
        total_blocks = sum(int(sample.get("blocks", 0)) for sample in self.samples)
        logical = sum(int(sample.get("logical_bytes", 0)) for sample in self.samples)
        capacity = total_blocks * self.block_bytes
        return {
            "blocks": total_blocks,
            "block_bytes": self.block_bytes,
            "capacity_bytes": capacity,
            "logical_bytes": logical,
            "waste_ratio": (capacity - logical) / capacity if capacity else 0.0,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "samples": [dict(sample) for sample in self.samples],
            "block_tokens": self.block_tokens,
            "block_bytes": self.block_bytes,
            "tail_waste": self.tail_waste(),
            "problems": self.problems(),
        }


def calibrate_bytes_per_token(
    *, longer_logical_bytes: int, shorter_logical_bytes: int,
    longer_tokens: int, shorter_tokens: int, theoretical: float,
) -> Dict[str, Any]:
    """Step 15: estimate the *actual* bytes/token from two adjacent lengths.

    ``总显存 / token`` mixes weights and workspace into the KV number, so the
    calibration uses the incremental memory between two lengths of the same
    configuration instead.
    """
    if longer_tokens <= shorter_tokens:
        raise ConfigError("longer_tokens must exceed shorter_tokens")
    if longer_logical_bytes < shorter_logical_bytes:
        raise ConfigError("longer logical bytes must not be smaller than shorter")
    measured = (longer_logical_bytes - shorter_logical_bytes) / (longer_tokens - shorter_tokens)
    ratio = measured / theoretical if theoretical else float("inf")
    return {
        "measured_bytes_per_token": measured,
        "theoretical_bytes_per_token": theoretical,
        "ratio": ratio,
        "consistent": 0.8 <= ratio <= 1.5 if theoretical else False,
        "note": "偏离理论值必须由 block/对齐/metadata 解释，而不是记成随机噪声",
    }


# ── position / mask / chunk boundaries (steps 18, 21) ──────────────────────


def verify_position_and_mask(
    *,
    positions: Sequence[int],
    chunk_boundaries: Sequence[int],
    attention_mask: Sequence[int],
    total_tokens: int,
) -> List[str]:
    """Steps 18/21: an off-by-one at a chunk boundary silently drops history.

    ``positions`` must be the exact ``0..n-1`` sequence (RoPE is position based),
    the mask must be causal and complete, and every chunk boundary must be
    strictly inside the sequence.
    """
    problems: List[str] = []
    if list(positions) != list(range(total_tokens)):
        problems.append(
            f"positions are not 0..{total_tokens - 1} (got {list(positions)[:6]}…) — RoPE 位置错位"
        )
    if len(attention_mask) != total_tokens:
        problems.append(f"mask length {len(attention_mask)} != {total_tokens}")
    if attention_mask and not all(value in (0, 1) for value in attention_mask):
        problems.append("attention mask is not binary")
    if attention_mask and sum(attention_mask) != total_tokens:
        problems.append("the attention mask drops tokens: this is a sliding window, not full context")
    for boundary in chunk_boundaries:
        if not 0 < boundary < total_tokens:
            problems.append(f"chunk boundary {boundary} is not strictly inside the sequence")
    if len(set(chunk_boundaries)) != len(chunk_boundaries):
        problems.append("duplicate chunk boundaries")
    return problems


def chunked_prefill_audit(
    *,
    chunk_sizes: Sequence[int],
    token_budget: int,
    decode_inserted: Sequence[int],
    unchunked_logits_delta: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Step 21: chunked prefill changes *scheduling*, not the mathematics.

    ``E14-F3`` §3.4 requires the chunk boundary, position/mask, KV commit and
    inserted decodes to be recorded, and the resulting logits to be compared with
    an unchunked reference; a schedule change that alters the output is a
    correctness failure, not a tuning choice.
    """
    problems: List[str] = []
    if not chunk_sizes:
        problems.append("no chunk sizes recorded")
    if any(size <= 0 for size in chunk_sizes):
        problems.append("a chunk size is not positive")
    if token_budget <= 0:
        problems.append("token_budget must be positive")
    elif any(size > token_budget for size in chunk_sizes):
        problems.append(f"a chunk exceeds the iteration token budget {token_budget}")
    if not decode_inserted:
        problems.append(
            "no decode was inserted between chunks: the point of chunked prefill is decoder interleaving"
        )
    if unchunked_logits_delta is None or tolerance is None:
        problems.append("no unchunked logits comparison supplied: the semantic claim is unverified")
    elif abs(unchunked_logits_delta) > tolerance:
        problems.append(
            f"chunked logits differ from unchunked by {unchunked_logits_delta:.3e} > {tolerance:.3e}"
        )
    return {
        "chunks": len(chunk_sizes),
        "token_budget": token_budget,
        "total_tokens": sum(chunk_sizes),
        "inserted_decodes": len(decode_inserted),
        "problems": problems,
        "output_preserved": not problems,
    }


def kv_quant_report(
    *,
    key_granularity: str,
    value_granularity: str,
    group_size: Optional[int],
    residual_window: int,
    scale_metadata_bytes: float,
    payload_bytes_per_token: float,
    actual_kernel: str,
) -> Dict[str, Any]:
    """Step 19: granularity, group, residual window and the *actual* kernel.

    ``E14-F3`` §3.2 notes that key/value distributions differ, so a single
    granularity is a claim to be checked rather than a default; a format whose
    metadata exceeds its saving is reported instead of being called a win.
    """
    problems: List[str] = []
    if key_granularity not in ("per_token", "per_channel", "per_group", "per_head_dim"):
        problems.append(f"unknown key granularity {key_granularity!r}")
    if value_granularity not in ("per_token", "per_channel", "per_group", "per_head_dim"):
        problems.append(f"unknown value granularity {value_granularity!r}")
    if key_granularity == "per_group" or value_granularity == "per_group":
        if not group_size or group_size <= 0:
            problems.append("a per_group layout needs a positive group_size")
    if residual_window < 0:
        problems.append("residual_window must be >= 0")
    if not actual_kernel:
        problems.append(
            "no actual kernel recorded: metadata 比 KV 节省还大或走 FP fallback 就抓不到（step 19）"
        )
    total = payload_bytes_per_token + scale_metadata_bytes
    return {
        "key_granularity": key_granularity,
        "value_granularity": value_granularity,
        "group_size": group_size,
        "residual_window": residual_window,
        "metadata_bytes_per_token": scale_metadata_bytes,
        "payload_bytes_per_token": payload_bytes_per_token,
        "total_bytes_per_token": total,
        "metadata_ratio": scale_metadata_bytes / total if total else 0.0,
        "actual_kernel": actual_kernel,
        "problems": problems,
    }


def eviction_audit(
    *,
    events: Sequence[Mapping[str, Any]],
    budget_tokens: int,
    total_tokens: int,
) -> Dict[str, Any]:
    """Step 20: each token's keep/evict/recompute decision, and no read of freed blocks.

    ``E14-F3`` §3.3: eviction is not equivalent to full attention, so the quality
    oracle must be long-range retrieval — but the *accounting* has to be right
    first, which is what this checks.
    """
    problems: List[str] = []
    if budget_tokens <= 0:
        problems.append("eviction budget must be positive")
    if budget_tokens > total_tokens:
        problems.append("eviction budget exceeds the sequence: nothing would be evicted")
    kept = evicted = recomputed = 0
    for index, event in enumerate(events):
        action = str(event.get("action", ""))
        if action not in ("keep", "evict", "recompute"):
            problems.append(f"eviction event {index} has unknown action {action!r}")
            continue
        if action == "keep":
            kept += 1
        elif action == "evict":
            evicted += 1
        else:
            recomputed += 1
        if event.get("read_after_free"):
            problems.append(f"eviction event {index} read a released block")
    if not events:
        problems.append("no eviction events recorded")
    return {
        "events": len(events),
        "kept": kept,
        "evicted": evicted,
        "recomputed": recomputed,
        "budget_tokens": budget_tokens,
        "total_tokens": total_tokens,
        "problems": problems,
        "note": "eviction 统计必须与真实 KV 一致（step 20）",
    }


# ── quality and service (steps 26–32) ──────────────────────────────────────


def quality_by_length_position(
    rows: Sequence[Mapping[str, Any]], *, guard_band: float
) -> Dict[str, Any]:
    """Step 26: report by needle depth and document length, not one mean.

    ``E14-F3`` §9: needle 质量随位置下降 → 必须限制支持范围，不能只报平均.
    """
    problems: List[str] = []
    if not rows:
        return {"error": "no quality rows supplied"}
    buckets: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        for key in ("length_bucket", "needle_depth", "task", "reference", "candidate"):
            if key not in row:
                problems.append(f"quality row is missing {key!r}")
                break
        else:
            if row["length_bucket"] not in LENGTH_BUCKETS:
                problems.append(f"unknown length bucket {row['length_bucket']!r}")
                continue
            bucket = buckets.setdefault((str(row["length_bucket"]), str(row["needle_depth"])), {"n": 0, "ref": 0.0, "cand": 0.0})
            bucket["n"] += 1
            bucket["ref"] += float(row["reference"])
            bucket["cand"] += float(row["candidate"])
    table = {
        f"{length}:{depth}": {
            "n": int(values["n"]),
            "reference": values["ref"] / values["n"],
            "candidate": values["cand"] / values["n"],
            "delta": (values["cand"] - values["ref"]) / values["n"],
        }
        for (length, depth), values in sorted(buckets.items())
    }
    worst = min(table.items(), key=lambda item: item[1]["delta"]) if table else None
    if worst and worst[1]["delta"] < -abs(guard_band):
        problems.append(
            f"stratum {worst[0]} falls {abs(worst[1]['delta']):.4f} below the reference, beyond the guard band"
        )
    return {"strata": table, "worst_stratum": worst[0] if worst else None, "problems": problems}


def short_context_regression(
    *, baseline_delta: float, candidate_delta: float, guard_band: float, latency_ratio: float
) -> Dict[str, Any]:
    """Step 27: a globally enabled mechanism must not degrade the ordinary workload."""
    problems: List[str] = []
    quality_delta = candidate_delta - baseline_delta
    if quality_delta < -abs(guard_band):
        problems.append(f"short-context quality fell by {abs(quality_delta):.4f}")
    if latency_ratio > 1.0 + abs(guard_band):
        problems.append(f"short-context latency ratio {latency_ratio:.4f} exceeds the guard band")
    return {
        "quality_delta": quality_delta,
        "latency_ratio": latency_ratio,
        "guard_band": guard_band,
        "problems": problems,
        "ok": not problems,
    }


def mixed_load_fairness(
    *, short_rows: Sequence[Mapping[str, Any]], long_rows: Sequence[Mapping[str, Any]],
    slo_ratio: float, protected: str = "short",
) -> Dict[str, Any]:
    """Step 29: long and short requests run together; the short ones must survive.

    ``E14-F3`` §10: 长请求饿死短请求 is a FAIL, and independently running the two
    lengths and adding the numbers cannot show it.
    """
    problems: List[str] = []
    target_rows = short_rows if protected == "short" else long_rows
    if not target_rows:
        problems.append(f"no {protected}-request rows supplied")
    worst = 0.0
    for row in target_rows:
        ratio = row.get("p99_ratio_to_slo")
        if ratio is None:
            problems.append(f"protected {protected} row has no p99_ratio_to_slo")
            continue
        worst = max(worst, float(ratio))
    if worst > slo_ratio:
        problems.append(f"protected {protected} requests reached {worst:.3f}× the SLO ratio (limit {slo_ratio})")
    return {
        "protected": protected,
        "short_requests": len(short_rows),
        "long_requests": len(long_rows),
        "worst_slo_ratio": worst,
        "problems": problems,
        "ok": not problems,
    }


def admission_budget(
    *, bytes_per_token: float, usable_bytes: int, safety_margin_bytes: int, updated_from: Optional[float] = None
) -> Dict[str, Any]:
    """Step 31: when the KV format changes, the admission budget must be recomputed.

    ``E14-F3`` §10: 容量模型仍使用旧 bytes/token 会让准入放行一个必然 OOM 的请求.
    Retaining the old value is reported rather than silently accepted.
    """
    problems: List[str] = []
    if bytes_per_token <= 0:
        problems.append("bytes_per_token must be positive")
    budget = max(0, usable_bytes - safety_margin_bytes)
    max_tokens = budget / bytes_per_token if bytes_per_token > 0 else 0.0
    if updated_from is not None and abs(updated_from - bytes_per_token) > 1e-9:
        problems.append(
            f"admission still uses the previous bytes/token ({updated_from}) instead of {bytes_per_token}"
        )
    return {
        "bytes_per_token": bytes_per_token,
        "usable_bytes": usable_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "budget_bytes": budget,
        "max_tokens": max_tokens,
        "budgets": list(ADMISSION_BUDGETS),
        "problems": problems,
    }


def over_limit_rejection(
    *, requested_tokens: int, max_context_tokens: int, structurally_rejected: bool, truncated_instead: bool
) -> Dict[str, Any]:
    """Step 33: an over-limit request must be refused *before* a large allocation.

    ``E14-F3`` §10: 内部截断后返回 200 is the failure; the expected outcome is a
    structured rejection, and the status code carries that meaning.
    """
    problems: List[str] = []
    over = requested_tokens > max_context_tokens
    if over and not structurally_rejected:
        problems.append("an over-limit request was not structurally rejected")
    if truncated_instead:
        problems.append("the request was truncated instead of refused (截断后返回成功 = FAIL)")
    return {
        "requested_tokens": requested_tokens,
        "max_context_tokens": max_context_tokens,
        "over_limit": over,
        "structurally_rejected": structurally_rejected,
        "status": rec.STATUS_NOT_RUN if not problems else rec.STATUS_FAIL_CORRECTNESS,
        "problems": problems,
    }


def corrupted_kv_metadata_check(
    *, block_table_valid: bool, scale_format_matches: bool, version_matches: bool, fallback_action: str
) -> Dict[str, Any]:
    """Step 35: corrupt KV metadata must be detected, isolated and never used.

    ``E14-F3`` §10: 错误历史被静默用于生成 is the failure, so the check requires a
    *positive* isolation/fallback decision rather than merely "no crash".
    """
    problems: List[str] = []
    if block_table_valid is False and fallback_action == "":
        problems.append("an invalid block table had no isolation or fallback action")
    if scale_format_matches is False:
        problems.append("scale metadata format mismatch was not rejected")
    if version_matches is False:
        problems.append("KV cache version mismatch was not rejected")
    if not fallback_action:
        problems.append("no fallback/isolation action recorded")
    return {
        "block_table_valid": block_table_valid,
        "scale_format_matches": scale_format_matches,
        "version_matches": version_matches,
        "fallback_action": fallback_action,
        "problems": problems,
        "isolated": not problems,
    }


def cancel_release(events: Sequence[Mapping[str, Any]]) -> List[str]:
    """Step 36: cancelling a long prefill/decode must release chunk, KV and queue."""
    problems: List[str] = []
    for event in events:
        stage = str(event.get("stage", "<unnamed>"))
        if stage not in ("prefill_chunk", "decode", "queue", "eviction"):
            problems.append(f"cancel stage {stage!r} is not documented")
        if event.get("chunk_work_remaining"):
            problems.append(f"cancel at {stage}: chunk work continued after cancellation")
        if event.get("kv_blocks_leaked"):
            problems.append(f"cancel at {stage}: KV blocks leaked")
        if event.get("refcount_mismatch"):
            problems.append(f"cancel at {stage}: block refcount mismatch")
    if not events:
        problems.append("no cancellation events recorded")
    return problems


def reconcile_prediction(
    *,
    predicted_max_context: float,
    measured_max_context: float,
    predicted_ttft_ms: float,
    measured_ttft_ms: float,
    bytes_per_token_used: float,
) -> Dict[str, Any]:
    """Step 37: rebuild capacity and latency from the measured bytes/token."""
    if predicted_max_context <= 0 or predicted_ttft_ms <= 0:
        return {"error": "predicted values must be positive"}
    return {
        "capacity_residual": measured_max_context - predicted_max_context,
        "capacity_ratio": measured_max_context / predicted_max_context,
        "ttft_residual_ms": measured_ttft_ms - predicted_ttft_ms,
        "ttft_ratio": measured_ttft_ms / predicted_ttft_ms,
        "bytes_per_token_used": bytes_per_token_used,
        "explained": abs(measured_ttft_ms / predicted_ttft_ms - 1.0) <= 0.2,
        "note": "“显存少了所以快”不是解释；必须用 bytes/token、block waste 与 kernel profile 重建（step 37）",
    }


# ── adoption (step 40) ─────────────────────────────────────────────────────


def f3_adoption(
    *,
    decision_id: str,
    intervention: str,
    no_truncation: Mapping[str, Any],
    quality: Mapping[str, Any],
    short_regression: Mapping[str, Any],
    fairness: Mapping[str, Any],
    capacity_gain: Mapping[str, Any],
    meta_problems: Tuple[str, ...],
    evidence_refs: Sequence[str],
) -> AdoptionDecision:
    """Step 40: a single maximum length never decides adoption (§10).

    A capacity improvement with no latency benefit is a *legitimate* outcome, so
    it maps to ``ADOPT_EXPERIMENTAL`` with an explicit boundary rather than being
    rejected for lacking a speedup.
    """
    problems: List[str] = []
    if intervention not in INTERVENTIONS:
        problems.append(f"unknown intervention {intervention!r}")
    if not no_truncation.get("verified"):
        problems.append("truncation/omission was not ruled out")
    if quality.get("problems"):
        problems.append("long-context quality gate failed")
    if not short_regression.get("ok"):
        problems.append("short-context regression gate failed")
    if not fairness.get("ok"):
        problems.append("mixed-load fairness gate failed")
    problems.extend(meta_problems)
    gain = capacity_gain.get("max_context_per_request") or 0.0
    if problems:
        decision = rec.REJECT_QUALITY
        allowed: Tuple[str, ...] = ()
    elif gain <= 0:
        decision = rec.REJECT_NO_BENEFIT
        allowed = ("容量与延迟模型已闭合，但未观察到容量提升",)
    else:
        decision = rec.ADOPT_EXPERIMENTAL
        allowed = (
            f"{intervention} 在预注册长度/并发域内提升容量至 {gain}",
            "容量提升与延迟收益是不同轴，不合并成单一 speedup",
        )
    return AdoptionDecision(
        decision_id=decision_id,
        experiment_id=EXPERIMENT_ID,
        decision=decision,
        allowed_claims=allowed,
        forbidden_claims=(
            "配置最大长度等于真实支持",
            "字符数代替 token 数",
            "短 perplexity 代替长文质量",
            "baseline OOM 时计算无限 speedup",
        ),
        quality_status=rec.STATUS_PASS if not problems else rec.STATUS_FAIL_QUALITY,
        performance_status=rec.STATUS_NOT_RUN,
        maturity=rec.MATURITY_QUALITY_VERIFIED if not problems else rec.MATURITY_SOURCE_INTEGRATED,
        evidence_refs=tuple(evidence_refs),
        limitations=tuple(problems) + ("支持范围必须按 needle 深度限定",),
        reopened_if=("得到更长的原生位置训练模型或新的 attention backend 时可重开",),
    )


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-F3 interfaces (labelled smoke, not an experiment)."""
    geometry = kv_bytes_per_token(
        layers=28, kv_heads_per_rank=4, head_dim=128, storage="int8",
        scale_group_size=64, metadata_bytes_per_group=2,
    )
    capacity = kv_capacity(bytes_per_token=geometry["bytes_per_token"], available_bytes=6 << 30, batch=8)
    episode = LongContextEpisode(
        episode_id="e0", model_artifact_id="serving::sha256:" + "0" * 64, candidate_config_id="cfg",
        submitted_tokens=32768, accepted_tokens=32768, processed_tokens=32768, truncated_tokens=0,
        kv_physical_bytes=1 << 30, kv_logical_bytes=1 << 30, actual_backend="paged_attn",
    )
    no_trunc = check_no_truncation(episode, trace_processed_tokens=32768)
    mask_problems = verify_position_and_mask(
        positions=list(range(4)), chunk_boundaries=[2], attention_mask=[1, 1, 1, 1], total_tokens=4
    )
    short = short_context_regression(baseline_delta=0.0, candidate_delta=0.0, guard_band=0.02, latency_ratio=1.01)
    fairness = mixed_load_fairness(
        short_rows=[{"p99_ratio_to_slo": 1.1}], long_rows=[{"p99_ratio_to_slo": 0.9}], slo_ratio=1.0
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "bytes_per_token": round(geometry["bytes_per_token"], 3),
        "metadata_ratio": round(geometry["metadata_ratio"], 4),
        "mask_problems": mask_problems,
        "no_truncation_verified": no_trunc["verified"],
        "capacity_per_request": int(capacity["max_context_per_request"]),
        "episode_problems": episode.validate(),
        "short_regression_ok": short["ok"],
        "fairness_violation_detected": not fairness["ok"],
        "interventions": list(INTERVENTIONS),
    }


# ── result accessors ───────────────────────────────────────────────────────

def kv_actual_kernel(report: Mapping[str, Any]) -> str:
    """Step 28: the kernel that really ran (an FP fallback must not be invisible)."""
    return str(report.get("actual_kernel", ""))


def truncation_is_impossible(check: Mapping[str, Any]) -> bool:
    """Step 14: the claim that every accepted token was processed."""
    return bool(check.get("verified"))


# ── protocol step table (40 steps of details/S14/E14-F3) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "绑定 E14-05 协议", ("frontier:issue_contract", "frontier:contract_hash")),
    (2, "冻结 long-context ModelArtifact", ("long_context:kv_bytes_per_token", "contracts:ServingModelArtifact")),
    (3, "冻结 full-KV/full-attention baseline", ("frontier:BaselinePair", "long_context:kv_capacity")),
    (4, "冻结 candidate artifact/config", ("frontier:BaselinePair.intended_difference", "long_context:INTERVENTIONS")),
    (5, "冻结真实输入 token ledger", ("long_context:LongContextEpisode.submitted_tokens",
                                       "identity:file_inventory")),
    (6, "冻结长上下文质量集", ("long_context:LONG_CONTEXT_TASKS", "frontier:QualityGate")),
    (7, "冻结生成与评分语义", ("frontier:TimingBoundaries", "speculative:GenerationContract")),
    (8, "冻结长度/并发扫描", ("long_context:LENGTH_BUCKETS", "frontier:WorkloadStrata")),
    (9, "冻结容量与 OOM 安全边界", ("long_context:ADMISSION_BUDGETS", "frontier:StopRules")),
    (10, "建立理论 KV/attention 模型", ("long_context:kv_bytes_per_token", "frontier:PredictionModel")),
    (11, "运行 capability/actual-path probe", ("contracts:check_actual_path_recorded", "long_context:kv_quant_report")),
    (12, "建立短上下文 correctness baseline", ("parity:evaluate_gates", "long_context:short_context_regression")),
    (13, "建立 full-KV 长度基线", ("long_context:LongContextEpisode", "long_context:kv_capacity")),
    (14, "验证无截断/无滑窗偷换", ("long_context:check_no_truncation", "long_context:LongContextEpisode.truncated_tokens")),
    (15, "校准 KV bytes/token", ("long_context:calibrate_bytes_per_token", "long_context:kv_bytes_per_token")),
    (16, "建立 block/fragmentation 基线", ("long_context:KvAllocatorTimeline", "long_context:KvAllocatorTimeline.tail_waste")),
    (17, "执行 candidate 小规模语义测试", ("parity:evaluate_gates", "long_context:LongContextEpisode.quality")),
    (18, "验证 position/mask/chunk boundary", ("long_context:verify_position_and_mask",
                                                 "long_context:chunked_prefill_audit")),
    (19, "验证 KV quant/dequant（若适用）", ("long_context:kv_quant_report", "long_context:KV_STORAGE_BYTES")),
    (20, "验证 eviction/recompute（若适用）", ("long_context:eviction_audit", "long_context:INTERVENTIONS")),
    (21, "验证 chunked prefill（若适用）", ("long_context:chunked_prefill_audit",
                                             "long_context:verify_position_and_mask")),
    (22, "运行 context-length 扫描", ("long_context:LENGTH_BUCKETS", "long_context:LongContextEpisode")),
    (23, "运行 output-length 扫描", ("long_context:LongContextEpisode.output_tokens",
                                      "long_context:LongContextEpisode.tpot_ms")),
    (24, "运行 batch/concurrency 扫描", ("long_context:kv_capacity", "long_context:admission_budget")),
    (25, "运行 prefix reuse 分层", ("posttraining:check_kv_reuse_identity", "long_context:KvAllocatorTimeline")),
    (26, "运行长上下文质量门", ("long_context:quality_by_length_position", "frontier:QualityGate.paired")),
    (27, "运行短上下文回归门", ("long_context:short_context_regression", "contracts:check_quality_before_performance")),
    (28, "采集 attention/KV kernel profile", ("long_context:kv_actual_kernel", "long_context:kv_quant_report",
                                               "records:PROFILE_LAYER_FIELDS")),
    (29, "运行长短混合 closed-loop", ("long_context:mixed_load_fairness", "frontier:StatisticsPlan")),
    (30, "运行 open-loop 到达率曲线", ("long_context:admission_budget", "records:PROFILE_LAYERS")),
    (31, "验证 chunk/admission 调度协同", ("long_context:admission_budget", "long_context:chunked_prefill_audit")),
    (32, "测显存、主存、能耗和成本", ("contracts:check_resource_ledger", "frontier:CostDenominator")),
    (33, "执行受控 over-limit 负例", ("long_context:over_limit_rejection", "records:STATUS_FAIL_CORRECTNESS")),
    (34, "执行受控 OOM/fragmentation 边界", ("long_context:kv_capacity", "campaign:isolation_clause")),
    (35, "注入损坏 KV metadata/cache", ("long_context:corrupted_kv_metadata_check", "campaign:isolation_clause")),
    (36, "验证取消、超时和 eviction 并发", ("long_context:cancel_release", "long_context:eviction_audit")),
    (37, "对账预测与实测", ("long_context:reconcile_prediction", "long_context:calibrate_bytes_per_token")),
    (38, "在 holdout 长文/负载确认", ("frontier:WorkloadStrata.holdout_id", "long_context:quality_by_length_position")),
    (39, "跨 run/time block 重复", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan.minimum_repeats")),
    (40, "形成 F3 AdoptionDecision", ("long_context:f3_adoption", "contracts:AdoptionDecision.validate")),
)
