"""KV-cache quantization: representation, capacity and attention path (E05-08).

P1 scope. KV quantization differs from weight quantization in ways this module
makes explicit (E05-08 §1):

* K and V have different statistics and are configured independently;
* K may be cached pre- or post-RoPE — the spec records the point, and mixing
  them is a semantic error that cannot be repaired later;
* the cache is written once and read many times, so the *write* path
  (range → scale → quantize → pack → page write) and the *read* path
  (page lookup → unpack → dequant → attention) are measured separately;
* real capacity includes scale/zero metadata, page headers, alignment,
  the residual high-precision window and workspace — the "INT4 = 1/4 of FP16"
  arithmetic is explicitly insufficient (E05-08 §3/§4);
* quality is only meaningful with the cache enabled and reused over a long
  generation, so the module records per-step evidence.

Everything is pure Python except the optional attention comparison, which
uses the same small pure-Python attention so the oracle works without torch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from hqsb.core.errors import ConfigError
from hqsb.quant.stats import summarize_distribution

#: Where K is quantized relative to RoPE (E05-08 §6).
ROPE_PRE = "pre_rope"
ROPE_POST = "post_rope"

ROPE_POINTS = (ROPE_PRE, ROPE_POST)

#: KV granularities.
KV_PER_TENSOR = "per_tensor"
KV_PER_TOKEN_HEAD = "per_token_per_head"
KV_PER_CHANNEL = "per_channel"
KV_PER_GROUP = "per_group"
KV_PER_BLOCK = "per_block"

KV_GRANULARITIES = (
    KV_PER_TENSOR,
    KV_PER_TOKEN_HEAD,
    KV_PER_CHANNEL,
    KV_PER_GROUP,
    KV_PER_BLOCK,
)


@dataclass
class KvQuantSpec:
    """Frozen KV quantization spec (E05-08 §14 step 1)."""

    k_bits: int = 8
    v_bits: int = 8
    k_granularity: str = KV_PER_TOKEN_HEAD
    v_granularity: str = KV_PER_TOKEN_HEAD
    k_group_size: Optional[int] = None
    v_group_size: Optional[int] = None
    rope_point: str = ROPE_POST
    residual_window: int = 0
    page_size: int = 16
    scale_dtype: str = "float32"
    k_zero_point: bool = False
    v_zero_point: bool = False
    clip_percentile: Optional[float] = None
    label: str = "kv8"

    def __post_init__(self) -> None:
        for name, value in (("k_granularity", self.k_granularity), ("v_granularity", self.v_granularity)):
            if value not in KV_GRANULARITIES:
                raise ConfigError(
                    f"unknown {name} {value!r}; supported: {list(KV_GRANULARITIES)}"
                )
        if self.rope_point not in ROPE_POINTS:
            raise ConfigError(
                f"unknown rope_point {self.rope_point!r}; supported: {list(ROPE_POINTS)}"
            )
        for name, bits in (("k_bits", self.k_bits), ("v_bits", self.v_bits)):
            if not 1 <= bits <= 16:
                raise ConfigError(f"{name}={bits} outside [1, 16]")
        if self.residual_window < 0:
            raise ConfigError("residual_window must be >= 0")
        if self.page_size <= 0:
            raise ConfigError("page_size must be positive")
        if self.k_granularity == KV_PER_GROUP and self.k_group_size is None:
            raise ConfigError("K per-group granularity needs k_group_size")
        if self.v_granularity == KV_PER_GROUP and self.v_group_size is None:
            raise ConfigError("V per-group granularity needs v_group_size")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "k_granularity": self.k_granularity,
            "v_granularity": self.v_granularity,
            "k_group_size": self.k_group_size,
            "v_group_size": self.v_group_size,
            "rope_point": self.rope_point,
            "residual_window": self.residual_window,
            "page_size": self.page_size,
            "scale_dtype": self.scale_dtype,
            "k_zero_point": self.k_zero_point,
            "v_zero_point": self.v_zero_point,
            "clip_percentile": self.clip_percentile,
            "label": self.label,
        }


def _scale_count(granularity: str, *, tokens: int, heads: int, group_size: Optional[int], head_dim: int) -> int:
    """Number of scales for one of K/V under a granularity (E05-08 §4)."""
    if tokens == 0:
        return 0
    if granularity == KV_PER_TENSOR:
        return 1
    if granularity == KV_PER_TOKEN_HEAD:
        return tokens * heads
    if granularity == KV_PER_CHANNEL:
        return head_dim
    if granularity == KV_PER_GROUP:
        if group_size is None:
            raise ConfigError("per-group KV needs a group size")
        return tokens * heads * math.ceil(head_dim / group_size)
    if granularity == KV_PER_BLOCK:
        raise ConfigError(
            "per-block granularity needs the block size from the runtime; use "
            "KV_PER_TOKEN_HEAD or KV_PER_GROUP here and describe blocks as pages"
        )
    raise ConfigError(f"unknown granularity {granularity!r}")


@dataclass
class KvCapacity:
    """Predicted KV capacity breakdown (E05-08 §3/§4/§13)."""

    layers: int
    kv_heads: int
    head_dim: int
    tokens: int
    payload_bytes: int
    scale_bytes: int
    zero_bytes: int
    page_header_bytes: int
    alignment_bytes: int
    residual_bytes: int
    workspace_bytes: int
    duplicated_cache_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.payload_bytes
            + self.scale_bytes
            + self.zero_bytes
            + self.page_header_bytes
            + self.alignment_bytes
            + self.residual_bytes
            + self.workspace_bytes
            + self.duplicated_cache_bytes
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "tokens": self.tokens,
            "payload_bytes": self.payload_bytes,
            "scale_bytes": self.scale_bytes,
            "zero_bytes": self.zero_bytes,
            "page_header_bytes": self.page_header_bytes,
            "alignment_bytes": self.alignment_bytes,
            "residual_bytes": self.residual_bytes,
            "workspace_bytes": self.workspace_bytes,
            "duplicated_cache_bytes": self.duplicated_cache_bytes,
            "total_bytes": self.total_bytes,
            "bytes_per_token": self.total_bytes / self.tokens if self.tokens else 0.0,
        }

    def reconcile(self, measured_bytes: int, *, tolerance_bytes: int) -> Dict[str, Any]:
        residual = int(measured_bytes) - self.total_bytes
        return {
            "predicted_total_bytes": self.total_bytes,
            "measured_total_bytes": int(measured_bytes),
            "residual_bytes": residual,
            "within_tolerance": abs(residual) <= int(tolerance_bytes),
            "parts": self.as_dict(),
            "note": (
                "allocator reserved above the active bytes is normal, but an "
                "unexplained residual must be reported, not rounded away"
            ),
        }


def kv_capacity(
    spec: KvQuantSpec,
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    tokens: int,
    batch: int = 1,
    workspace_bytes: int = 0,
    duplicated_cache_bytes: int = 0,
    dtype_bytes: int = 2,
) -> KvCapacity:
    """Compute the full capacity model for one KV configuration.

    ``2`` (K and V) is applied per component rather than to a naive total, so
    asymmetric K/V bits, different granularities and a residual window are all
    representable. ``dtype_bytes`` is the residual-window element size and
    defaults to FP16. A 16-bit K or V payload is unquantized FP16 and has no
    scale/zero-point planes, while page bookkeeping still applies.
    """
    if min(layers, kv_heads, head_dim, tokens) < 1:
        raise ConfigError(
            f"KV dimensions must be positive: layers={layers} heads={kv_heads} "
            f"head_dim={head_dim} tokens={tokens}"
        )
    sequences = max(1, batch)
    residual = min(spec.residual_window, tokens)
    quantized_tokens = tokens - residual

    def _payload(bits: int) -> int:
        return (
            layers * kv_heads * head_dim * bits * quantized_tokens * sequences // 8
        )

    payload = _payload(spec.k_bits) + _payload(spec.v_bits)
    scale_bytes_per = {"float16": 2, "float32": 4, "float64": 8}[spec.scale_dtype]
    k_scales = 0 if spec.k_bits == 16 else sequences * layers * _scale_count(
        spec.k_granularity,
        tokens=quantized_tokens,
        heads=kv_heads,
        group_size=spec.k_group_size,
        head_dim=head_dim,
    )
    v_scales = 0 if spec.v_bits == 16 else sequences * layers * _scale_count(
        spec.v_granularity,
        tokens=quantized_tokens,
        heads=kv_heads,
        group_size=spec.v_group_size,
        head_dim=head_dim,
    )
    scale_bytes = (k_scales + v_scales) * scale_bytes_per
    zero_bytes = 0
    if spec.k_zero_point:
        zero_bytes += k_scales * 1
    if spec.v_zero_point:
        zero_bytes += v_scales * 1

    pages = math.ceil(tokens / spec.page_size) * sequences * layers * 2
    page_header_bytes = pages * 8  # page id + length (documented layout)
    per_page_alignment = 32  # bytes per page, from the runtime layout
    alignment_bytes = pages * per_page_alignment
    residual_bytes = (
        layers * kv_heads * head_dim * 2 * dtype_bytes * residual * sequences
    )
    return KvCapacity(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        tokens=tokens,
        payload_bytes=payload,
        scale_bytes=scale_bytes,
        zero_bytes=zero_bytes,
        page_header_bytes=page_header_bytes,
        alignment_bytes=alignment_bytes,
        residual_bytes=residual_bytes,
        workspace_bytes=workspace_bytes,
        duplicated_cache_bytes=duplicated_cache_bytes,
    )


def fp16_baseline_bytes(
    *, layers: int, kv_heads: int, head_dim: int, tokens: int, batch: int = 1, dtype_bytes: int = 2
) -> int:
    """FP16 KV payload for the same shape (capacity comparison baseline)."""
    return layers * kv_heads * head_dim * 2 * dtype_bytes * tokens * max(1, batch)


def capacity_ratio(spec: KvQuantSpec, *, layers: int, kv_heads: int, head_dim: int, tokens: int, batch: int = 1, dtype_bytes: int = 2) -> Dict[str, Any]:
    """Capacity ratio against FP16, with an explicit residual element size.

    ``dtype_bytes`` only changes the residual window; the comparison baseline
    stays FP16 so requesting an FP32 residual cannot silently change both sides.
    """
    capacity = kv_capacity(
        spec, layers=layers, kv_heads=kv_heads, head_dim=head_dim, tokens=tokens, batch=batch,
        dtype_bytes=dtype_bytes,
    )
    baseline = fp16_baseline_bytes(
        layers=layers, kv_heads=kv_heads, head_dim=head_dim, tokens=tokens, batch=batch
    )
    return {
        "quantized_total_bytes": capacity.total_bytes,
        "fp16_payload_bytes": baseline,
        "ratio_vs_fp16_payload": baseline / capacity.total_bytes if capacity.total_bytes else float("nan"),
        "payload_only_ratio": baseline / capacity.payload_bytes if capacity.payload_bytes else float("nan"),
        "metadata_fraction": (
            (capacity.total_bytes - capacity.payload_bytes) / capacity.total_bytes
            if capacity.total_bytes
            else float("nan")
        ),
        "parts": capacity.as_dict(),
    }


# ── representation quality (E05-08 §9) ────────────────────────────────────


def quantize_kv_tensor(
    values: Sequence[float], spec: KvQuantSpec, *, bits: int, group_size: Optional[int]
) -> Dict[str, Any]:
    """Quantize one K or V tensor with the spec's granularity (per token/head).

    Returns reconstruction metrics so the representation error can be reported
    per layer/head/token rather than as one aggregate number.
    """
    from hqsb.benchmark.metrics import numerical_diff_summary
    from hqsb.quant.rtn import quantize
    from hqsb.quant.spec import QuantScheme

    if bits == 8:
        scheme = QuantScheme(bits=8, granularity="per_channel", axis=-1, label="kv8")
    elif bits == 4:
        scheme = QuantScheme(
            bits=4,
            granularity="per_group" if group_size else "per_channel",
            group_size=group_size,
            axis=-1,
            label=f"kv4_g{group_size}",
        )
    else:
        raise ConfigError(f"KV quantization supports 8/4 bits here, got {bits}")
    flat = [float(value) for value in values]
    quantized = quantize(flat, scheme)
    summary = numerical_diff_summary(flat, quantized.values_dequant)
    saturated = sum(
        1 for code in quantized.q if code in (scheme.qmin, scheme.qmax)
    )
    return {
        "metrics": summary,
        "saturation_rate": saturated / len(quantized.q) if quantized.q else float("nan"),
        "scales": list(quantized.scales),
        "spec": spec.as_dict(),
        "bits": bits,
    }


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = math.fsum(exps)
    return [value / total for value in exps]


def attention_reference(
    q_rows: Sequence[Sequence[float]],
    k_rows: Sequence[Sequence[float]],
    v_rows: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Small, dependency-free scaled dot-product attention reference.

    Used as the attention oracle for the fake-KV path: quantized K/V are
    injected here and the QK scores, softmax and output are compared with the
    FP16 path (E05-08 §9.2). Real runs use the model's attention; this
    reference exists so the check is available before any runtime exists.
    """
    head_dim = len(k_rows[0]) if k_rows and k_rows[0] else 0
    if head_dim == 0:
        raise ConfigError("attention reference needs non-empty K rows")
    scale = 1.0 / math.sqrt(head_dim)
    outputs: List[List[float]] = []
    for q_row in q_rows:
        scores = [
            sum(a * b for a, b in zip(q_row, k_row)) * scale for k_row in k_rows
        ]
        weights = softmax(scores)
        out_row = []
        for dim in range(head_dim):
            out_row.append(
                math.fsum(
                    weight * v_row[dim] for weight, v_row in zip(weights, v_rows)
                )
            )
        outputs.append(out_row)
    return outputs


def attention_comparison_report(
    q_rows: Sequence[Sequence[float]],
    k_reference: Sequence[Sequence[float]],
    v_reference: Sequence[Sequence[float]],
    k_quantized: Sequence[Sequence[float]],
    v_quantized: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Compare attention with FP16 vs quantized K/V (E05-08 §9.2).

    Reports the QK-score error, the softmax distribution shift (KL/JS) and the
    output error, because a small KV reconstruction error does not imply a
    stable attention distribution.
    """
    from hqsb.benchmark.metrics import numerical_diff_summary
    from hqsb.quant.quality import jensen_shannon, kl_divergence

    head_dim = len(k_reference[0])
    scale = 1.0 / math.sqrt(head_dim)
    score_rows_ref: List[List[float]] = []
    score_rows_q: List[List[float]] = []
    attention_rows_ref: List[List[float]] = []
    attention_rows_q: List[List[float]] = []
    for q_row in q_rows:
        scores_ref = [sum(a * b for a, b in zip(q_row, k)) * scale for k in k_reference]
        scores_q = [sum(a * b for a, b in zip(q_row, k)) * scale for k in k_quantized]
        score_rows_ref.append(scores_ref)
        score_rows_q.append(scores_q)
        attention_rows_ref.append(
            _attend(scores_ref, v_reference)
        )
        attention_rows_q.append(_attend(scores_q, v_quantized))
    flat_scores_ref = [value for row in score_rows_ref for value in row]
    flat_scores_q = [value for row in score_rows_q for value in row]
    flat_out_ref = [value for row in attention_rows_ref for value in row]
    flat_out_q = [value for row in attention_rows_q for value in row]
    js_values = [
        jensen_shannon(softmax(ref), softmax(q))
        for ref, q in zip(score_rows_ref, score_rows_q)
    ]
    kl_values = [
        kl_divergence(softmax(ref), softmax(q))
        for ref, q in zip(score_rows_ref, score_rows_q)
    ]
    return {
        "qk_score": numerical_diff_summary(flat_scores_ref, flat_scores_q),
        "attention_output": numerical_diff_summary(flat_out_ref, flat_out_q),
        "softmax_js_mean": sum(js_values) / len(js_values) if js_values else float("nan"),
        "softmax_kl_mean": sum(kl_values) / len(kl_values) if kl_values else float("nan"),
        "q_norm": summarize_distribution([abs(value) for row in q_rows for value in row]).as_dict(),
        "k_quantized_error": numerical_diff_summary(
            [value for row in k_reference for value in row],
            [value for row in k_quantized for value in row],
        ),
        "v_quantized_error": numerical_diff_summary(
            [value for row in v_reference for value in row],
            [value for row in v_quantized for value in row],
        ),
    }


def _attend(scores: Sequence[float], v_rows: Sequence[Sequence[float]]) -> List[float]:
    weights = softmax(scores)
    head_dim = len(v_rows[0])
    return [
        math.fsum(weight * v_row[dim] for weight, v_row in zip(weights, v_rows))
        for dim in range(head_dim)
    ]


# ── lifecycle, sweeps and max-context search (E05-08 §10/§12) ─────────────


@dataclass
class KvStepRecord:
    """Per-step cache state during generation (E05-08 §9.3)."""

    step: int
    context_length: int
    kv_bytes: int
    k_error_rmse: float = float("nan")
    v_error_rmse: float = float("nan")
    attention_output_rmse: float = float("nan")
    logit_kl: float = float("nan")
    token: int = -1
    diverged: bool = False
    tpot_ms: float = float("nan")
    quant_dequant_ms: float = float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "context_length": self.context_length,
            "kv_bytes": self.kv_bytes,
            "k_error_rmse": self.k_error_rmse,
            "v_error_rmse": self.v_error_rmse,
            "attention_output_rmse": self.attention_output_rmse,
            "logit_kl": self.logit_kl,
            "token": self.token,
            "diverged": self.diverged,
            "tpot_ms": self.tpot_ms,
            "quant_dequant_ms": self.quant_dequant_ms,
        }


def generation_sweep(
    steps: Sequence[KvStepRecord], *, requested_output_tokens: int
) -> Dict[str, Any]:
    """Summarize a KV-quantized generation: drift, first divergence, slowdown."""
    if not steps:
        return {"steps": 0, "first_divergence_step": -1, "kv_bytes_final": 0}
    first_divergence = next(
        (record.step for record in steps if record.diverged), -1
    )
    kl_values = [
        record.logit_kl for record in steps if not math.isnan(record.logit_kl)
    ]
    tpot_values = [record.tpot_ms for record in steps if not math.isnan(record.tpot_ms)]
    return {
        "steps": len(steps),
        "requested_output_tokens": requested_output_tokens,
        "completed": len(steps) >= requested_output_tokens,
        "first_divergence_step": first_divergence,
        "mean_kl": (sum(kl_values) / len(kl_values)) if kl_values else float("nan"),
        "max_kl": max(kl_values) if kl_values else float("nan"),
        "mean_tpot_ms": (
            sum(tpot_values) / len(tpot_values) if tpot_values else float("nan")
        ),
        "tpot_last_decile_ms": (
            sum(tpot_values[int(len(tpot_values) * 0.9) :]) / max(1, len(tpot_values) - int(len(tpot_values) * 0.9))
            if tpot_values
            else float("nan")
        ),
        "kv_bytes_final": steps[-1].kv_bytes,
        "records": [record.as_dict() for record in steps],
    }


def context_generation_matrix(
    *, context_lengths: Sequence[int], output_lengths: Sequence[int]
) -> List[Dict[str, int]]:
    """Build the context × generation matrix (E05-08 §10).

    Both directions are represented: long-context/short-output and
    short-context/long-output answer different questions, and at least one
    configuration must have a long output to expose accumulation.
    """
    if not context_lengths or not output_lengths:
        raise ConfigError("both lengths must be non-empty")
    matrix = []
    for context in sorted(set(int(value) for value in context_lengths)):
        for output in sorted(set(int(value) for value in output_lengths)):
            if context <= 0 or output <= 0:
                raise ConfigError(f"lengths must be positive, got {context}/{output}")
            matrix.append(
                {
                    "input_context_length": context,
                    "output_generation_length": output,
                    "total_cache_length": context + output,
                }
            )
    return matrix


def max_context_search(
    probe,
    *,
    low: int,
    high: int,
    safety_margin_bytes: int,
    tolerance: int = 0,
) -> Dict[str, Any]:
    """Binary-search the maximum context that fits with a fixed safety margin.

    ``probe(context)`` must attempt an allocation and return
    ``{"fits": bool, "allocated_bytes": int, "reserved_bytes": int,
    "free_bytes": int}``. The margin is subtracted *before* deciding, and every
    probed point is recorded (E05-08 §12.3: an accidental OOM on a single try
    is not a capacity measurement).
    """
    if low <= 0 or high < low:
        raise ConfigError(f"invalid search range [{low}, {high}]")
    if safety_margin_bytes < 0:
        raise ConfigError("safety margin must be non-negative")
    trace: List[Dict[str, Any]] = []
    best = 0
    lo, hi = low, high
    while lo <= hi:
        mid = (lo + hi) // 2
        result = dict(probe(mid))
        free = int(result.get("free_bytes", 0))
        fits = bool(result.get("fits", False)) and free >= safety_margin_bytes
        trace.append(
            {
                "context": mid,
                "fits_reported": bool(result.get("fits", False)),
                "fits_with_margin": fits,
                "free_bytes": free,
                "allocated_bytes": int(result.get("allocated_bytes", 0)),
                "reserved_bytes": int(result.get("reserved_bytes", 0)),
                "safety_margin_bytes": safety_margin_bytes,
            }
        )
        if fits:
            best = mid
            lo = mid + 1 + tolerance
        else:
            hi = mid - 1
    return {
        "max_context": best,
        "safety_margin_bytes": safety_margin_bytes,
        "probes": trace,
        "note": (
            "capacity is reported with the margin and allocation policy; a "
            "single OOM point without this trace is not a capacity result"
        ),
    }


def kv_spec_json(spec: KvQuantSpec) -> str:
    return json.dumps(spec.as_dict(), sort_keys=True, indent=2)


__all__ = [
    "KV_GRANULARITIES",
    "KV_PER_BLOCK",
    "KV_PER_CHANNEL",
    "KV_PER_GROUP",
    "KV_PER_TENSOR",
    "KV_PER_TOKEN_HEAD",
    "KvCapacity",
    "KvQuantSpec",
    "KvStepRecord",
    "ROPE_POST",
    "ROPE_PRE",
    "ROPE_POINTS",
    "attention_comparison_report",
    "attention_reference",
    "capacity_ratio",
    "context_generation_matrix",
    "fp16_baseline_bytes",
    "generation_sweep",
    "kv_capacity",
    "kv_spec_json",
    "max_context_search",
    "quantize_kv_tensor",
    "softmax",
]
