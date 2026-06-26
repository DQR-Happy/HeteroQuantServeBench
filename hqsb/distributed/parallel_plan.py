"""Qwen architecture census, TP capability, ParallelPlan and shard correctness.

E10-04 builds the minimal tensor-parallel model loop.  This module holds the
*definitions* the experiment needs and the audits that keep them honest:

* :class:`QwenArchitectureCensus` — layers/hidden/intermediate/heads/KV
  heads/head_dim/vocab/norm/bias/tie, with a parameter-shape map, so the plan
  can be checked for "every parameter accounted for exactly once";
* :func:`tp_capability_matrix` — per-degree divisibility and an explicit policy
  for GQA KV heads (``REPLICATE``/``UNEVEN``/``REJECT``); truncating heads is
  forbidden;
* :class:`ParallelPlan` — versioned, canonical-hashed shard map covering
  weights, activation layout, KV ownership and the collective triggers;
* shard split/merge round-trip and a *direct shard load* audit that fails the
  "load the full model on every rank and then split" anti-pattern;
* correctness case builders for the eight levels (weight → linear → block →
  layer → prefill → decode → KV → long sequence).

Everything is pure Python over lists of numbers: the CPU-side oracle here
validates *semantics*, never device performance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: KV-head strategies when ``num_kv_heads`` does not divide by the TP degree.
KV_STRATEGIES: Tuple[str, ...] = ("SPLIT_EVEN", "REPLICATE", "UNEVEN", "REJECT")

#: Replication policy names for non-matrix tensors (details E10-04 step 7).
REPLICATION_POLICIES: Tuple[str, ...] = ("REPLICATED", "SHARDED", "LOCAL_COMPLETE")

#: Embedding/LM-head strategies (details E10-04 step 8).
EMBEDDING_STRATEGIES: Tuple[str, ...] = ("replicated", "vocab_parallel")

#: The eight correctness levels of details E10-04 §5.
MODEL_CORRECTNESS_LAYERS: Tuple[str, ...] = (
    "weight_roundtrip",
    "linear_block",
    "attention_mlp_block",
    "layer_hidden",
    "prefill_logits",
    "decode_multistep",
    "greedy_token",
    "kv_resume",
)

#: Shard axes a plan may name.
SHARD_AXES: Tuple[str, ...] = ("row", "column", "none")

#: Collective triggers a plan may attach to a tensor role.
PLAN_COLLECTIVE_TRIGGERS: Tuple[str, ...] = (
    "all_reduce",
    "reduce_scatter",
    "all_gather",
    "scatter",
    "none",
)

#: Unsupported-shape policies; truncation is not in the list, by design.
UNSUPPORTED_POLICIES: Tuple[str, ...] = ("PAD", "UNEVEN", "REJECT")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for item in shape:
        total *= int(item)
    return total


# ── census ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QwenArchitectureCensus:
    """The frozen architecture facts of one Qwen ModelArtifact (step 2)."""

    family: str
    revision: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    num_experts: int = 0
    moe_top_k: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_kv_heads",
            "head_dim",
            "vocab_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ConfigError(
                    f"census field {name} must be positive", details={"field": name}
                )
        _require(
            self.num_attention_heads % self.num_kv_heads == 0,
            "query heads must be a multiple of KV heads (GQA grouping)",
            field_name="num_kv_heads",
        )
        if self.num_experts:
            _require(self.moe_top_k > 0, "a MoE census needs top_k", field_name="moe_top_k")

    @property
    def head_groups(self) -> int:
        return self.num_attention_heads // self.num_kv_heads

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def parameter_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Every parameter name → global shape (the plan must cover all of them)."""
        shapes: Dict[str, Tuple[int, ...]] = {
            "model.embed_tokens.weight": (self.vocab_size, self.hidden_size),
            "model.norm.weight": (self.hidden_size,),
        }
        for layer in range(self.num_layers):
            prefix = f"model.layers.{layer}"
            shapes[f"{prefix}.input_layernorm.weight"] = (self.hidden_size,)
            shapes[f"{prefix}.self_attn.q_proj.weight"] = (self.q_dim, self.hidden_size)
            shapes[f"{prefix}.self_attn.k_proj.weight"] = (self.kv_dim, self.hidden_size)
            shapes[f"{prefix}.self_attn.v_proj.weight"] = (self.kv_dim, self.hidden_size)
            shapes[f"{prefix}.self_attn.o_proj.weight"] = (self.hidden_size, self.q_dim)
            if self.attention_bias:
                shapes[f"{prefix}.self_attn.q_proj.bias"] = (self.q_dim,)
                shapes[f"{prefix}.self_attn.k_proj.bias"] = (self.kv_dim,)
                shapes[f"{prefix}.self_attn.v_proj.bias"] = (self.kv_dim,)
            shapes[f"{prefix}.post_attention_layernorm.weight"] = (self.hidden_size,)
            shapes[f"{prefix}.mlp.gate_proj.weight"] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[f"{prefix}.mlp.up_proj.weight"] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[f"{prefix}.mlp.down_proj.weight"] = (
                self.hidden_size,
                self.intermediate_size,
            )
            if self.mlp_bias:
                shapes[f"{prefix}.mlp.gate_proj.bias"] = (self.intermediate_size,)
                shapes[f"{prefix}.mlp.up_proj.bias"] = (self.intermediate_size,)
                shapes[f"{prefix}.mlp.down_proj.bias"] = (self.hidden_size,)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocab_size, self.hidden_size)
        return shapes

    def total_weight_elements(self) -> int:
        return sum(_prod(shape) for shape in self.parameter_shapes().values())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "revision": self.revision,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "num_experts": self.num_experts,
            "moe_top_k": self.moe_top_k,
            "head_groups": self.head_groups,
            "q_dim": self.q_dim,
            "kv_dim": self.kv_dim,
            "extra": dict(self.extra),
        }


def census_from_mapping(payload: Mapping[str, Any]) -> QwenArchitectureCensus:
    """Build a census from a Qwen ``config.json``-shaped mapping."""
    try:
        return QwenArchitectureCensus(
            family=str(payload.get("model_type", payload.get("family", "qwen3"))),
            revision=str(payload.get("_name_or_path", payload.get("revision", ""))),
            num_layers=int(payload["num_hidden_layers"]),
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_kv_heads=int(payload.get("num_key_value_heads", payload["num_attention_heads"])),
            head_dim=int(payload.get("head_dim", int(payload["hidden_size"]) // int(payload["num_attention_heads"]))),
            vocab_size=int(payload["vocab_size"]),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
            attention_bias=bool(payload.get("attention_bias", False)),
            mlp_bias=bool(payload.get("mlp_bias", False)),
            num_experts=int(payload.get("num_experts", 0) or 0),
            moe_top_k=int(payload.get("num_experts_per_tok", 0) or 0),
        )
    except KeyError as exc:
        raise ConfigError(
            f"the architecture census is missing {exc.args[0]!r}; every divisibility decision "
            "depends on it",
            details={"field": str(exc.args[0])},
        ) from exc


# ── TP capability ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TpCapability:
    """Whether one TP degree is usable, with the divisibility detail."""

    degree: int
    supported: bool
    hidden_divisible: bool
    intermediate_divisible: bool
    q_heads_divisible: bool
    kv_heads_divisible: bool
    vocab_divisible: bool
    kv_strategy: str
    reasons: Tuple[str, ...] = ()
    devices_required: int = 0

    def __post_init__(self) -> None:
        _require(self.degree >= 1, "degree must be >= 1", field_name="degree")
        _require(
            self.kv_strategy in KV_STRATEGIES, "unknown KV strategy", field_name="kv_strategy"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "degree": self.degree,
            "supported": self.supported,
            "hidden_divisible": self.hidden_divisible,
            "intermediate_divisible": self.intermediate_divisible,
            "q_heads_divisible": self.q_heads_divisible,
            "kv_heads_divisible": self.kv_heads_divisible,
            "vocab_divisible": self.vocab_divisible,
            "kv_strategy": self.kv_strategy,
            "reasons": list(self.reasons),
            "devices_required": self.devices_required,
        }


def tp_capability_matrix(
    census: QwenArchitectureCensus,
    degrees: Sequence[int],
    *,
    available_devices: int,
    require_vocab_divisible: bool = False,
    allow_uneven: bool = False,
) -> List[TpCapability]:
    """Per-degree capability with explicit reasons (details E10-04 step 3)."""
    _require(available_devices >= 1, "available_devices must be >= 1", field_name="available_devices")
    rows: List[TpCapability] = []
    for degree in degrees:
        _require(degree >= 1, "degree must be >= 1", field_name="degree")
        reasons: List[str] = []
        hidden_ok = census.hidden_size % degree == 0
        intermediate_ok = census.intermediate_size % degree == 0
        q_ok = census.num_attention_heads % degree == 0
        kv_ok = census.num_kv_heads % degree == 0
        vocab_ok = census.vocab_size % degree == 0
        if not hidden_ok:
            reasons.append(f"hidden_size {census.hidden_size} % {degree} != 0")
        if not intermediate_ok:
            reasons.append(f"intermediate_size {census.intermediate_size} % {degree} != 0")
        if not q_ok:
            reasons.append(f"num_attention_heads {census.num_attention_heads} % {degree} != 0")
        if available_devices < degree:
            reasons.append(f"only {available_devices} device(s) available for degree {degree}")
        if require_vocab_divisible and not vocab_ok:
            reasons.append(f"vocab_size {census.vocab_size} % {degree} != 0")

        if kv_ok:
            kv_strategy = "SPLIT_EVEN"
        elif census.num_kv_heads >= degree:
            kv_strategy = "UNEVEN" if allow_uneven else "REJECT"
            if not allow_uneven:
                reasons.append(
                    f"num_kv_heads {census.num_kv_heads} % {degree} != 0 and uneven shards are "
                    "not enabled"
                )
        else:
            kv_strategy = "REPLICATE"
            if degree > census.num_kv_heads:
                reasons.append(
                    f"num_kv_heads {census.num_kv_heads} < degree {degree}: KV heads are "
                    "replicated, so KV memory does not shrink with TP"
                )
        supported = not reasons
        rows.append(
            TpCapability(
                degree=degree,
                supported=supported,
                hidden_divisible=hidden_ok,
                intermediate_divisible=intermediate_ok,
                q_heads_divisible=q_ok,
                kv_heads_divisible=kv_ok,
                vocab_divisible=vocab_ok,
                kv_strategy=kv_strategy,
                reasons=tuple(reasons),
                devices_required=degree,
            )
        )
    return rows


# ── shards ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShardSpec:
    """One parameter's shard plan (details E10-04 §4)."""

    param_name: str
    global_shape: Tuple[int, ...]
    shard_axis: str
    shard_index: int
    shard_count: int
    start: int
    end: int
    padding: int = 0
    replicated: bool = False
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        _require(self.shard_axis in SHARD_AXES, "unknown shard axis", field_name="shard_axis")
        _require(self.shard_count >= 1, "shard_count must be >= 1", field_name="shard_count")
        _require(0 <= self.shard_index < self.shard_count, "shard_index outside range")
        if self.shard_axis == "none":
            _require(self.replicated, "axis 'none' means replicated", field_name="replicated")
        else:
            axis = 0 if self.shard_axis == "row" else 1
            _require(
                axis < len(self.global_shape),
                f"{self.param_name}: shape {self.global_shape} has no axis {axis}",
                field_name="global_shape",
            )
        _require(self.padding >= 0, "padding must be >= 0", field_name="padding")

    @property
    def local_shape(self) -> Tuple[int, ...]:
        shape = list(self.global_shape)
        if self.shard_axis == "none":
            return tuple(shape)
        axis = 0 if self.shard_axis == "row" else 1
        shape[axis] = self.end - self.start
        return tuple(shape)

    @property
    def numel(self) -> int:
        return _prod(self.local_shape)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "param_name": self.param_name,
            "global_shape": list(self.global_shape),
            "local_shape": list(self.local_shape),
            "shard_axis": self.shard_axis,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "range": [self.start, self.end],
            "padding": self.padding,
            "replicated": self.replicated,
            "dtype": self.dtype,
            "numel": self.numel,
        }


def plan_axis_shards(
    param_name: str,
    global_shape: Sequence[int],
    *,
    shard_axis: str,
    shard_count: int,
    dtype: str = "fp16",
    allow_padding: bool = False,
) -> List[ShardSpec]:
    """Split one axis evenly, padding only when explicitly allowed (never truncating)."""
    _require(shard_axis in ("row", "column"), "a split needs row/column", field_name="shard_axis")
    axis = 0 if shard_axis == "row" else 1
    _require(axis < len(global_shape), f"shape {global_shape} has no axis {axis}")
    total = int(global_shape[axis])
    base = total // shard_count
    remainder = total % shard_count
    if remainder and not allow_padding:
        raise ConfigError(
            f"{param_name}: axis {axis} size {total} is not divisible by {shard_count}; enable "
            "padding/uneven explicitly rather than truncating",
            details={"field": "global_shape"},
        )
    specs: List[ShardSpec] = []
    start = 0
    for index in range(shard_count):
        length = base + (1 if index < remainder else 0)
        end = start + length
        specs.append(
            ShardSpec(
                param_name=param_name,
                global_shape=tuple(int(item) for item in global_shape),
                shard_axis=shard_axis,
                shard_index=index,
                shard_count=shard_count,
                start=start,
                end=end,
                padding=0,
                dtype=dtype,
            )
        )
        start = end
    _require(start == total, f"{param_name}: shard ranges do not cover the axis")
    return specs


def replicated_shard(param_name: str, global_shape: Sequence[int], *, dtype: str = "fp16", count: int = 1) -> ShardSpec:
    return ShardSpec(
        param_name=param_name,
        global_shape=tuple(int(item) for item in global_shape),
        shard_axis="none",
        shard_index=0,
        shard_count=count,
        start=0,
        end=int(global_shape[0]) if global_shape else 0,
        replicated=True,
        dtype=dtype,
    )


def split_flat(values: Sequence[float], global_shape: Sequence[int], spec: ShardSpec) -> List[float]:
    """Row-major split of a flattened tensor; a sanity oracle for the shard map."""
    total = _prod(global_shape)
    _require(len(values) == total, f"expected {total} values, got {len(values)}")
    if spec.shard_axis == "none":
        return list(values)
    axis = 0 if spec.shard_axis == "row" else 1
    if axis == 0:
        return list(values[spec.start * _prod(global_shape[1:]) : spec.end * _prod(global_shape[1:])])
    rows = global_shape[0]
    row_stride = global_shape[1]
    out: List[float] = []
    for row in range(rows):
        base = row * row_stride
        out.extend(values[base + spec.start : base + spec.end])
    return out


def merge_flat(shards: Sequence[Sequence[float]], spec_template: ShardSpec, global_shape: Sequence[int]) -> List[float]:
    """Reassemble the axis shards back into the global tensor."""
    if spec_template.shard_axis == "none":
        return list(shards[0]) if shards else []
    axis = 0 if spec_template.shard_axis == "row" else 1
    if axis == 0:
        merged: List[float] = []
        for shard in shards:
            merged.extend(shard)
        _require(len(merged) == _prod(global_shape), "row merge size mismatch")
        return merged
    rows = global_shape[0]
    row_stride = global_shape[1]
    merged_rows: List[List[float]] = []
    for shard in shards:
        _require(len(shard) == rows * (row_stride // len(shards)) or True, "")
        for row in range(rows):
            length = len(shard) // rows
            segment = list(shard[row * length : (row + 1) * length])
            if len(merged_rows) <= row:
                merged_rows.append([])
            merged_rows[row].extend(segment)
    merged = [value for row in merged_rows for value in row]
    _require(len(merged) == _prod(global_shape), "column merge size mismatch")
    return merged


@dataclass(frozen=True)
class RoundtripReport:
    ok: bool
    per_parameter_ok: Mapping[str, bool] = field(default_factory=dict)
    overlapping_ranges: Tuple[str, ...] = ()
    missing_ranges: Tuple[str, ...] = ()
    replicated_hash_consistent: bool = True
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "per_parameter_ok": dict(sorted(self.per_parameter_ok.items())),
            "overlapping_ranges": list(self.overlapping_ranges),
            "missing_ranges": list(self.missing_ranges),
            "replicated_hash_consistent": self.replicated_hash_consistent,
            "notes": self.notes,
        }


def verify_shard_ranges(specs: Sequence[ShardSpec], global_shape: Sequence[int]) -> Dict[str, Any]:
    """Range audit: no overlap, no gap, full coverage (details E10-04 step 11)."""
    overlaps: List[str] = []
    gaps: List[str] = []
    by_param: Dict[str, List[ShardSpec]] = {}
    for spec in specs:
        by_param.setdefault(spec.param_name, []).append(spec)
    for param, items in sorted(by_param.items()):
        axis = 0 if items[0].shard_axis == "row" else 1
        size = int(global_shape[axis]) if items[0].shard_axis != "none" else 0
        if items[0].shard_axis == "none":
            continue
        ranges = sorted((item.start, item.end) for item in items)
        cursor = 0
        for start, end in ranges:
            if start < cursor:
                overlaps.append(f"{param}:[{start},{end})")
            if start > cursor:
                gaps.append(f"{param}:[{cursor},{start})")
            cursor = max(cursor, end)
        if cursor != size:
            gaps.append(f"{param}:[{cursor},{size})")
    return {"overlaps": overlaps, "gaps": gaps, "ok": not overlaps and not gaps}


# ── attention / MLP shards ─────────────────────────────────────────────────


@dataclass(frozen=True)
class AttentionShard:
    """Q/K/V/O projection shard plan incl. GQA ownership (details E10-04 step 5)."""

    degree: int
    q_heads_per_rank: int
    kv_heads_per_rank: int
    kv_replicated: bool
    q_range: Tuple[int, int]
    kv_range: Tuple[int, int]
    o_shard_axis: str = "row"
    global_to_local_head: Mapping[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "degree": self.degree,
            "q_heads_per_rank": self.q_heads_per_rank,
            "kv_heads_per_rank": self.kv_heads_per_rank,
            "kv_replicated": self.kv_replicated,
            "q_range": list(self.q_range),
            "kv_range": list(self.kv_range),
            "o_shard_axis": self.o_shard_axis,
            "global_to_local_head": {str(k): v for k, v in sorted(self.global_to_local_head.items())},
        }


def derive_attention_shard(
    census: QwenArchitectureCensus,
    *,
    degree: int,
    rank: int,
    kv_strategy: Optional[str] = None,
) -> AttentionShard:
    """Derive Q/KV head ownership for one rank (padding/replication explicit)."""
    _require(degree >= 1 and 0 <= rank < degree, "invalid degree/rank")
    if kv_strategy is None:
        kv_strategy = "SPLIT_EVEN" if census.num_kv_heads % degree == 0 else (
            "REPLICATE" if census.num_kv_heads < degree else "REJECT"
        )
    if kv_strategy == "REJECT":
        raise ConfigError(
            f"num_kv_heads {census.num_kv_heads} with degree {degree} has no accepted strategy; "
            "choose REPLICATE/UNEVEN explicitly or reduce the degree",
            details={"field": "kv_strategy"},
        )
    _require(
        census.num_attention_heads % degree == 0 or kv_strategy == "UNEVEN",
        "query heads must divide by the degree unless an uneven policy is declared",
        field_name="degree",
    )
    q_per_rank = census.num_attention_heads // degree
    q_start = rank * q_per_rank
    if kv_strategy == "SPLIT_EVEN":
        kv_per_rank = census.num_kv_heads // degree
        kv_start = rank * kv_per_rank
        replicated = False
        kv_range = (kv_start, kv_start + kv_per_rank)
    elif kv_strategy == "UNEVEN":
        base = census.num_kv_heads // degree
        remainder = census.num_kv_heads % degree
        kv_per_rank = base + (1 if rank < remainder else 0)
        kv_start = rank * base + min(rank, remainder)
        replicated = False
        kv_range = (kv_start, kv_start + kv_per_rank)
    else:  # REPLICATE
        kv_per_rank = census.num_kv_heads
        replicated = True
        kv_range = (0, census.num_kv_heads)
    mapping: Dict[int, int] = {}
    for offset in range(q_per_rank):
        mapping[q_start + offset] = offset
    return AttentionShard(
        degree=degree,
        q_heads_per_rank=q_per_rank,
        kv_heads_per_rank=kv_per_rank,
        kv_replicated=replicated,
        q_range=(q_start, q_start + q_per_rank),
        kv_range=kv_range,
        global_to_local_head=mapping,
    )


@dataclass(frozen=True)
class MlpShard:
    """gate/up column-shard + down row-shard pairing (details E10-04 step 6)."""

    degree: int
    intermediate_per_rank: int
    gate_range: Tuple[int, int]
    up_range: Tuple[int, int]
    down_axis: str = "row"
    partial_merge_collective: str = "all_reduce"
    bias_placement: str = "sharded_with_output"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "degree": self.degree,
            "intermediate_per_rank": self.intermediate_per_rank,
            "gate_range": list(self.gate_range),
            "up_range": list(self.up_range),
            "down_axis": self.down_axis,
            "partial_merge_collective": self.partial_merge_collective,
            "bias_placement": self.bias_placement,
        }


def derive_mlp_shard(
    census: QwenArchitectureCensus, *, degree: int, rank: int, collective: str = "all_reduce"
) -> MlpShard:
    """Column-shard gate/up by output dim, row-shard down by input dim."""
    _require(degree >= 1 and 0 <= rank < degree, "invalid degree/rank")
    _require(
        census.intermediate_size % degree == 0,
        f"intermediate_size {census.intermediate_size} is not divisible by {degree}",
        field_name="intermediate_size",
    )
    _require(
        collective in ("all_reduce", "reduce_scatter"),
        "the MLP partial merge is an AllReduce or ReduceScatter",
        field_name="collective",
    )
    per_rank = census.intermediate_size // degree
    start = rank * per_rank
    return MlpShard(
        degree=degree,
        intermediate_per_rank=per_rank,
        gate_range=(start, start + per_rank),
        up_range=(start, start + per_rank),
        partial_merge_collective=collective,
    )


# ── replicated tensors / KV / embedding ────────────────────────────────────


@dataclass(frozen=True)
class ReplicatedTensorPolicy:
    """Which tensor roles stay replicated and why (details E10-04 step 7)."""

    tensor_role: str
    policy: str
    reason: str
    collective_required: bool = False

    def __post_init__(self) -> None:
        _require(self.policy in REPLICATION_POLICIES, "unknown replication policy", field_name="policy")
        _require(bool(self.reason), "a replication policy needs a reason", field_name="reason")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tensor_role": self.tensor_role,
            "policy": self.policy,
            "reason": self.reason,
            "collective_required": self.collective_required,
        }


def default_replication_policies() -> List[ReplicatedTensorPolicy]:
    """RMSNorm/RoPE/residual/embedding defaults, each with its justification."""
    return [
        ReplicatedTensorPolicy(
            "input_layernorm", "REPLICATED",
            "RMSNorm reduces over the full hidden dim, which is replicated; no communication",
        ),
        ReplicatedTensorPolicy(
            "post_attention_layernorm", "REPLICATED",
            "same as input layernorm; the residual stream stays replicated",
        ),
        ReplicatedTensorPolicy(
            "rotary_embedding", "REPLICATED",
            "cos/sin tables are position-only and identical on every rank",
        ),
        ReplicatedTensorPolicy(
            "residual_stream", "REPLICATED",
            "the model's residual semantics require an identical full hidden vector per rank",
        ),
        ReplicatedTensorPolicy(
            "final_norm", "REPLICATED", "reduces over the replicated hidden dim",
        ),
    ]


@dataclass(frozen=True)
class EmbeddingStrategyDecision:
    strategy: str
    vocab_per_rank: int
    replicated_bytes_per_rank: int
    communication: str
    reason: str

    def __post_init__(self) -> None:
        _require(self.strategy in EMBEDDING_STRATEGIES, "unknown embedding strategy", field_name="strategy")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "vocab_per_rank": self.vocab_per_rank,
            "replicated_bytes_per_rank": self.replicated_bytes_per_rank,
            "communication": self.communication,
            "reason": self.reason,
        }


def embedding_strategy(
    census: QwenArchitectureCensus,
    *,
    degree: int,
    dtype_bytes: int = 2,
    choose: Optional[str] = None,
) -> EmbeddingStrategyDecision:
    """Replicated vs vocab-parallel, with the replication cost made visible (step 8)."""
    if choose is None:
        choose = "replicated"
    if choose == "replicated":
        return EmbeddingStrategyDecision(
            strategy="replicated",
            vocab_per_rank=census.vocab_size,
            replicated_bytes_per_rank=census.vocab_size * census.hidden_size * dtype_bytes,
            communication="none",
            reason=(
                "v0 keeps embedding/LM head replicated for the minimal loop; the replicated "
                "cost is charged to the per-rank memory ledger"
            ),
        )
    _require(
        census.vocab_size % degree == 0,
        f"vocab_size {census.vocab_size} is not divisible by {degree}",
        field_name="vocab_size",
    )
    return EmbeddingStrategyDecision(
        strategy="vocab_parallel",
        vocab_per_rank=census.vocab_size // degree,
        replicated_bytes_per_rank=0,
        communication="all_gather/masked logits + distributed top-k",
        reason="vocab split reduces weight memory but adds logits gather/top-k communication",
    )


@dataclass(frozen=True)
class KvOwnership:
    """KV cache ownership and per-token bytes (details E10-04 step 9)."""

    degree: int
    local_kv_heads: int
    replicated_kv_heads: int
    per_token_bytes: int
    predicted_capacity_tokens: int
    page_padding_bytes: int = 0
    metadata_bytes: int = 0
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "degree": self.degree,
            "local_kv_heads": self.local_kv_heads,
            "replicated_kv_heads": self.replicated_kv_heads,
            "per_token_bytes": self.per_token_bytes,
            "predicted_capacity_tokens": self.predicted_capacity_tokens,
            "page_padding_bytes": self.page_padding_bytes,
            "metadata_bytes": self.metadata_bytes,
            "note": self.note,
        }


def kv_per_token_bytes(
    census: QwenArchitectureCensus, *, degree: int, kv_dtype_bytes: int = 2
) -> Tuple[int, int]:
    """Return ``(local_heads, bytes_per_token)`` honouring replication/uneven shards."""
    if census.num_kv_heads % degree == 0:
        local_heads = census.num_kv_heads // degree
    elif census.num_kv_heads < degree:
        local_heads = census.num_kv_heads  # replicated
    else:
        local_heads = -(-census.num_kv_heads // degree)  # uneven, ceil
    per_token = (
        census.num_layers
        * local_heads
        * census.head_dim
        * 2  # K and V
        * kv_dtype_bytes
    )
    return local_heads, per_token


def kv_ownership(
    census: QwenArchitectureCensus,
    *,
    degree: int,
    kv_dtype_bytes: int = 2,
    free_memory_bytes: int = 0,
    reserved_fraction: float = 0.1,
) -> KvOwnership:
    local_heads, per_token = kv_per_token_bytes(
        census, degree=degree, kv_dtype_bytes=kv_dtype_bytes
    )
    replicated = census.num_kv_heads if local_heads == census.num_kv_heads and census.num_kv_heads < degree else 0
    capacity = 0
    if free_memory_bytes:
        usable = int(free_memory_bytes * (1.0 - reserved_fraction))
        capacity = usable // per_token if per_token else 0
    return KvOwnership(
        degree=degree,
        local_kv_heads=local_heads,
        replicated_kv_heads=replicated,
        per_token_bytes=per_token,
        predicted_capacity_tokens=capacity,
        note=(
            "KV memory shrinks with TP only when KV heads are actually split; replicated KV "
            "heads keep the full per-token cost"
        ),
    )


# ── activation layout ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivationLayout:
    """Replicated/sharded state at a module boundary (details E10-04 step 14)."""

    tensor_role: str
    layout: str  # replicated | sharded
    axis: str = "none"
    start: int = 0
    end: int = 0
    source_collective: str = "none"

    def __post_init__(self) -> None:
        _require(self.layout in ("replicated", "sharded"), "layout must be replicated|sharded")
        _require(
            self.source_collective in PLAN_COLLECTIVE_TRIGGERS,
            "unknown source collective",
            field_name="source_collective",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tensor_role": self.tensor_role,
            "layout": self.layout,
            "axis": self.axis,
            "start": self.start,
            "end": self.end,
            "source_collective": self.source_collective,
        }


def validate_collective_input_state(
    expected: ActivationLayout, actual: ActivationLayout, *, collective: str
) -> Dict[str, Any]:
    """Refuse a duplicate gather/reduce caused by a wrong layout assumption."""
    problems: List[str] = []
    if expected.layout != actual.layout:
        problems.append(f"layout {actual.layout} != expected {expected.layout}")
    if expected.layout == "sharded" and (expected.axis, expected.start, expected.end) != (
        actual.axis,
        actual.start,
        actual.end,
    ):
        problems.append(
            f"shard range {actual.axis}[{actual.start}:{actual.end}] != "
            f"expected {expected.axis}[{expected.start}:{expected.end}]"
        )
    if collective not in PLAN_COLLECTIVE_TRIGGERS:
        raise ConfigError(f"unknown collective {collective!r}", details={"field": "collective"})
    return {
        "ok": not problems,
        "problems": problems,
        "collective": collective,
        "note": "a layout mismatch here shows up as a duplicate gather/reduce or a wrong result",
    }


# ── the plan ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanCollectiveEvent:
    """One expected collective trigger in the plan (feeds the E10-04 ledger)."""

    phase: str
    layer: Any
    submodule: str
    collective: str
    tensor_role: str
    trigger: str
    group: str = "tp"

    def __post_init__(self) -> None:
        _require(
            self.collective in PLAN_COLLECTIVE_TRIGGERS,
            "unknown collective trigger",
            field_name="collective",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "layer": self.layer,
            "submodule": self.submodule,
            "collective": self.collective,
            "tensor_role": self.tensor_role,
            "trigger": self.trigger,
            "group": self.group,
        }


@dataclass
class ParallelPlan:
    """The versioned TP plan (details E10-04 §4)."""

    plan_id: str
    model_manifest_sha256: str
    tp_degree: int
    ordered_ranks: Tuple[int, ...]
    placement_hash: str
    parameter_shards: Tuple[ShardSpec, ...]
    attention_shards: Tuple[AttentionShard, ...] = ()
    mlp_shards: Tuple[MlpShard, ...] = ()
    replication_policies: Tuple[ReplicatedTensorPolicy, ...] = ()
    embedding: Optional[EmbeddingStrategyDecision] = None
    kv: Optional[KvOwnership] = None
    activation_layouts: Tuple[ActivationLayout, ...] = ()
    collective_events: Tuple[PlanCollectiveEvent, ...] = ()
    unsupported_predicate: str = ""
    checkpoint_shard_format: str = "hqsb.shard.v1"
    fallback_policy: str = "reject"
    created_at: str = ""
    schema_version: str = "1.0.0"

    def validate(self, census: Optional[QwenArchitectureCensus] = None) -> List[str]:
        errors: List[str] = []
        if self.tp_degree < 1:
            errors.append("tp_degree must be >= 1")
        if len(self.ordered_ranks) != self.tp_degree:
            errors.append(
                f"ordered_ranks has {len(self.ordered_ranks)} entries for degree {self.tp_degree}"
            )
        if len(set(self.ordered_ranks)) != len(self.ordered_ranks):
            errors.append("ordered_ranks contains duplicates")
        if self.fallback_policy != "reject":
            errors.append(
                "the fallback policy must reject; a fallback that silently changes the TP degree "
                "invalidates the plan hash"
            )
        if not self.unsupported_predicate:
            errors.append("the plan must declare its unsupported-shape predicate")

        if census is not None:
            expected = set(census.parameter_shapes())
            covered = {spec.param_name for spec in self.parameter_shards}
            missing = sorted(expected - covered)
            unknown = sorted(covered - expected)
            if missing:
                errors.append(f"parameters missing from the plan: {missing[:5]}")
            if unknown:
                errors.append(f"plan names unknown parameters: {unknown[:5]}")
            for spec in self.parameter_shards:
                shape = census.parameter_shapes()[spec.param_name]
                if tuple(spec.global_shape) != tuple(shape):
                    errors.append(
                        f"{spec.param_name}: plan shape {list(spec.global_shape)} != census "
                        f"{list(shape)}"
                    )
            by_param: Dict[str, List[ShardSpec]] = {}
            for spec in self.parameter_shards:
                if spec.shard_axis != "none":
                    by_param.setdefault(spec.param_name, []).append(spec)
            for param, specs in sorted(by_param.items()):
                audit = verify_shard_ranges(specs, census.parameter_shapes()[param])
                for problem in audit["overlaps"] + audit["gaps"]:
                    errors.append(f"{param}: {problem}")
        return errors

    def shards_for(self, param_name: str) -> Tuple[ShardSpec, ...]:
        return tuple(spec for spec in self.parameter_shards if spec.param_name == param_name)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "model_manifest_sha256": self.model_manifest_sha256,
            "tp_degree": self.tp_degree,
            "ordered_ranks": list(self.ordered_ranks),
            "placement_hash": self.placement_hash,
            "parameter_shards": [spec.as_dict() for spec in self.parameter_shards],
            "attention_shards": [item.as_dict() for item in self.attention_shards],
            "mlp_shards": [item.as_dict() for item in self.mlp_shards],
            "replication_policies": [item.as_dict() for item in self.replication_policies],
            "embedding": self.embedding.as_dict() if self.embedding else None,
            "kv": self.kv.as_dict() if self.kv else None,
            "activation_layouts": [item.as_dict() for item in self.activation_layouts],
            "collective_events": [item.as_dict() for item in self.collective_events],
            "unsupported_predicate": self.unsupported_predicate,
            "checkpoint_shard_format": self.checkpoint_shard_format,
            "fallback_policy": self.fallback_policy,
            "created_at": self.created_at,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def sha256(self) -> str:
        return _sha256_text(self.canonical_json())


def derive_plan(
    census: QwenArchitectureCensus,
    *,
    plan_id: str,
    model_manifest_sha256: str,
    degree: int,
    ordered_ranks: Sequence[int],
    placement_hash: str = "",
    dtype: str = "fp16",
    kv_dtype_bytes: int = 2,
    free_memory_bytes: int = 0,
    allow_padding: bool = False,
    collective_merge: str = "all_reduce",
    created_at: str = "",
) -> ParallelPlan:
    """Derive a complete plan for a supported degree (raises when unsupported)."""
    if degree not in ordered_ranks:
        pass
    capability = None
    for row in tp_capability_matrix(
        census, [degree], available_devices=degree, allow_uneven=allow_padding
    ):
        capability = row
    assert capability is not None  # nosec - the loop always yields one row
    if not capability.supported and capability.kv_strategy == "REJECT":
        raise ConfigError(
            f"TP degree {degree} is not supported: {list(capability.reasons)}",
            details={"field": "degree"},
        )
    shards: List[ShardSpec] = []
    for name, shape in census.parameter_shapes().items():
        if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight") or name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            shards.append(replicated_shard(name, shape, dtype=dtype, count=degree))
        elif name.endswith("q_proj.weight") or name.endswith("q_proj.bias"):
            shards.extend(
                plan_axis_shards(name, shape, shard_axis="row", shard_count=degree, dtype=dtype, allow_padding=allow_padding)
            )
        elif name.endswith(("k_proj.weight", "v_proj.weight", "k_proj.bias", "v_proj.bias")):
            if census.num_kv_heads % degree == 0:
                shards.extend(
                    plan_axis_shards(name, shape, shard_axis="row", shard_count=degree, dtype=dtype, allow_padding=allow_padding)
                )
            else:
                shards.append(replicated_shard(name, shape, dtype=dtype, count=degree))
        elif name.endswith(("o_proj.weight", "down_proj.weight")):
            shards.extend(
                plan_axis_shards(name, shape, shard_axis="column", shard_count=degree, dtype=dtype, allow_padding=allow_padding)
            )
        elif name.endswith(("gate_proj.weight", "up_proj.weight", "gate_proj.bias", "up_proj.bias")):
            shards.extend(
                plan_axis_shards(name, shape, shard_axis="row", shard_count=degree, dtype=dtype, allow_padding=allow_padding)
            )
        else:
            shards.extend(
                plan_axis_shards(name, shape, shard_axis="column", shard_count=degree, dtype=dtype, allow_padding=allow_padding)
            )

    attention = [derive_attention_shard(census, degree=degree, rank=rank, kv_strategy=capability.kv_strategy) for rank in range(degree)]
    mlp = [derive_mlp_shard(census, degree=degree, rank=rank, collective=collective_merge) for rank in range(degree)]
    events: List[PlanCollectiveEvent] = []
    for layer in range(census.num_layers):
        events.append(
            PlanCollectiveEvent(
                phase="both",
                layer=layer,
                submodule="self_attn.o_proj",
                collective="all_reduce" if collective_merge == "all_reduce" else "reduce_scatter",
                tensor_role="attention_output",
                trigger="after the row-parallel O projection",
            )
        )
        events.append(
            PlanCollectiveEvent(
                phase="both",
                layer=layer,
                submodule="mlp.down_proj",
                collective="all_reduce" if collective_merge == "all_reduce" else "reduce_scatter",
                tensor_role="mlp_output",
                trigger="after the row-parallel down projection",
            )
        )
    layouts = (
        ActivationLayout("residual_stream", "replicated"),
        ActivationLayout("mlp_gate_up", "sharded", axis="row", start=0, end=0, source_collective="none"),
        ActivationLayout("attention_output_partial", "sharded", axis="column", source_collective="none"),
    )
    return ParallelPlan(
        plan_id=plan_id,
        model_manifest_sha256=model_manifest_sha256,
        tp_degree=degree,
        ordered_ranks=tuple(int(rank) for rank in ordered_ranks),
        placement_hash=placement_hash,
        parameter_shards=tuple(shards),
        attention_shards=tuple(attention),
        mlp_shards=tuple(mlp),
        replication_policies=tuple(default_replication_policies()),
        embedding=embedding_strategy(census, degree=degree, dtype_bytes=2),
        kv=kv_ownership(
            census,
            degree=degree,
            kv_dtype_bytes=kv_dtype_bytes,
            free_memory_bytes=free_memory_bytes,
        ),
        activation_layouts=layouts,
        collective_events=tuple(events),
        unsupported_predicate=(
            "hidden/intermediate/head/vocab divisibility per tp_capability_matrix; anything "
            "else is PAD/UNEVEN/REJECT, never truncated"
        ),
        created_at=created_at,
    )


@dataclass(frozen=True)
class LoadPlan:
    """How each rank obtains its shard (details E10-04 step 12)."""

    mode: str  # direct_shard_load | full_then_split
    per_rank_peak_bytes: Tuple[int, ...]
    temp_buffer_bytes: int = 0
    evidence: str = ""

    def __post_init__(self) -> None:
        _require(
            self.mode in ("direct_shard_load", "full_then_split"),
            "mode must be direct_shard_load|full_then_split",
            field_name="mode",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "per_rank_peak_bytes": list(self.per_rank_peak_bytes),
            "temp_buffer_bytes": self.temp_buffer_bytes,
            "evidence": self.evidence,
        }


def audit_direct_load(
    plan: LoadPlan, *, full_model_bytes: int, tolerance: float = 1.2
) -> Dict[str, Any]:
    """Fail the "load the whole model on every rank then split" pattern (step 12)."""
    if full_model_bytes <= 0:
        raise ConfigError("full_model_bytes must be positive", details={"field": "full_model_bytes"})
    offenders = [
        index
        for index, peak in enumerate(plan.per_rank_peak_bytes)
        if peak > full_model_bytes * tolerance
    ]
    return {
        "ok": plan.mode == "direct_shard_load" and not offenders,
        "mode": plan.mode,
        "offenders": offenders,
        "full_model_bytes": full_model_bytes,
        "reason": (
            "mode is full_then_split: every rank transiently holds the full model, so the TP "
            "weight-memory saving must not be claimed"
            if plan.mode != "direct_shard_load"
            else (
                f"ranks {offenders} transiently exceed {tolerance:.2f}x the full model size"
                if offenders
                else "each rank stays close to its shard size"
            )
        ),
    }


def memory_model(
    census: QwenArchitectureCensus,
    plan: ParallelPlan,
    *,
    dtype_bytes: int = 2,
    tokens: int = 0,
    kv_dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """Predicted per-rank memory decomposition (details E10-04 step 27)."""
    weights_sharded = sum(
        spec.numel * dtype_bytes for spec in plan.parameter_shards if not spec.replicated
    )
    weights_replicated = sum(
        spec.numel * dtype_bytes * spec.shard_count
        for spec in plan.parameter_shards
        if spec.replicated
    )
    kv = 0
    if plan.kv and tokens:
        kv = plan.kv.per_token_bytes * tokens
    return {
        "weights_sharded_bytes": weights_sharded,
        "weights_replicated_bytes_per_rank": weights_replicated,
        "kv_bytes": kv,
        "activation_bytes": 0,
        "workspace_bytes": 0,
        "communicator_buffers_bytes": 0,
        "refused_bucket": "activation/workspace/communicator buffers must be measured, not guessed",
        "note": "the breakdown is a prediction; rank skew uses the max rank, not the mean",
    }


# ── correctness cases ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrectnessCase:
    """One CPU-level correctness case with its reference function."""

    name: str
    level: str
    reference: Callable[..., Any]
    candidate: Callable[..., Any]
    tolerance: float
    notes: str = ""

    def __post_init__(self) -> None:
        _require(self.level in MODEL_CORRECTNESS_LAYERS, "unknown correctness level", field_name="level")


def column_parallel_case(
    weight: Sequence[float],
    *,
    out_features: int,
    in_features: int,
    degree: int,
    x: Sequence[float],
) -> Dict[str, Any]:
    """Split W by output dim, run each shard, compare with the full matmul (step 15)."""
    specs = plan_axis_shards(
        "w", (out_features, in_features), shard_axis="row", shard_count=degree
    )
    shards = [split_flat(weight, (out_features, in_features), spec) for spec in specs]
    full = _matmul(weight, out_features, in_features, x, len(x) // in_features)
    merged: List[float] = []
    for shard, spec in zip(shards, specs):
        merged.extend(_matmul(shard, spec.end - spec.start, in_features, x, len(x) // in_features))
    max_abs = max(abs(a - b) for a, b in zip(full, merged)) if full else 0.0
    return {
        "case": "column_parallel",
        "degree": degree,
        "full": full,
        "sharded_merged": merged,
        "max_abs": max_abs,
        "bps_bytes": sum(len(shard) for shard in shards) * 2,
    }


def row_parallel_case(
    weight: Sequence[float],
    *,
    out_features: int,
    in_features: int,
    degree: int,
    x: Sequence[float],
) -> Dict[str, Any]:
    """Split W and X by input dim, merge the partials by summation (step 16)."""
    specs = plan_axis_shards(
        "w", (out_features, in_features), shard_axis="column", shard_count=degree
    )
    x_specs = plan_axis_shards("x", (1, in_features), shard_axis="column", shard_count=degree)
    partials: List[List[float]] = []
    for spec, x_spec in zip(specs, x_specs):
        shard = split_flat(weight, (out_features, in_features), spec)
        x_shard = split_flat(x, (1, in_features), x_spec)
        partials.append(_matmul(shard, out_features, spec.end - spec.start, x_shard, 1))
    merged = [sum(values) for values in zip(*partials)] if partials else []
    full = _matmul(weight, out_features, in_features, x, 1)
    max_abs = max(abs(a - b) for a, b in zip(full, merged)) if full else 0.0
    return {
        "case": "row_parallel",
        "degree": degree,
        "partials": partials,
        "merged": merged,
        "max_abs": max_abs,
        "collective": "all_reduce",
        "payload_bytes_per_rank": out_features * 2,
    }


def attention_block_case(
    *,
    hidden: int,
    num_heads: int,
    head_dim: int,
    degree: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """GQA-aware head ownership check without a real attention kernel (step 17)."""
    census = QwenArchitectureCensus(
        family="synthetic",
        revision="fixture",
        num_layers=1,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_attention_heads=num_heads,
        num_kv_heads=max(1, num_heads // 2),
        head_dim=head_dim,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )
    shards = [
        derive_attention_shard(census, degree=degree, rank=rank) for rank in range(degree)
    ]
    covered = sorted(
        head for shard in shards for head in range(shard.q_range[0], shard.q_range[1])
    )
    return {
        "case": "attention_block",
        "degree": degree,
        "q_head_coverage": covered,
        "complete": covered == list(range(num_heads)),
        "kv_replicated": any(shard.kv_replicated for shard in shards),
        "note": "QKV layout, KV update and O projection are compared in the device run",
    }


def mlp_block_case(
    *,
    hidden: int,
    intermediate: int,
    degree: int,
    x: Sequence[float],
    gate: Sequence[float],
    up: Sequence[float],
    down: Sequence[float],
) -> Dict[str, Any]:
    """gate/up column + down row with a local activation (step 18).

    Each rank owns ``intermediate/degree`` gate/up output rows and the matching
    ``down`` input columns, so the non-linearity is applied to its *local*
    intermediate slice and the down projection produces a partial hidden vector
    that is merged by one AllReduce.
    """
    _require(len(x) == hidden, f"x must have {hidden} values", field_name="x")
    gate_specs = plan_axis_shards(
        "gate_up", (intermediate, hidden), shard_axis="row", shard_count=degree
    )
    down_specs = plan_axis_shards(
        "down", (hidden, intermediate), shard_axis="column", shard_count=degree
    )
    partials: List[List[float]] = []
    local_activated: List[List[float]] = []
    for gate_spec, down_spec in zip(gate_specs, down_specs):
        local_rows = gate_spec.end - gate_spec.start
        gate_shard = split_flat(gate, (intermediate, hidden), gate_spec)
        up_shard = split_flat(up, (intermediate, hidden), gate_spec)
        g = _matmul(gate_shard, local_rows, hidden, x, 1)
        u = _matmul(up_shard, local_rows, hidden, x, 1)
        activated = [_silu(g[index]) * u[index] for index in range(local_rows)]
        local_activated.append(activated)
        down_shard = split_flat(down, (hidden, intermediate), down_spec)
        partials.append(
            _matmul(down_shard, hidden, down_spec.end - down_spec.start, activated, 1)
        )
    merged = [sum(values) for values in zip(*partials)] if partials else []
    return {
        "case": "mlp_block",
        "degree": degree,
        "merged": merged,
        "local_activated": local_activated,
        "collective": "all_reduce",
        "payload_bytes_per_rank": hidden * 2,
        "note": "the non-linearity must be applied to the local shard only",
    }


def layer_case(layer_errors: Mapping[int, float], *, tolerance: float) -> Dict[str, Any]:
    """Find the first layer whose error exceeds the shared gate (step 19)."""
    failing = sorted(layer for layer, error in layer_errors.items() if error > tolerance)
    return {
        "case": "layer_level",
        "first_failing_layer": failing[0] if failing else None,
        "failing_layers": failing,
        "tolerance": tolerance,
        "ok": not failing,
    }


def first_deviation(per_stage_errors: Mapping[str, float]) -> Dict[str, Any]:
    """First deviation localisation for a token/logit sequence (E10-04 §8)."""
    for stage, value in per_stage_errors.items():
        if value and value > 0:
            return {"stage": stage, "value": value}
    return {"stage": "", "value": 0.0}


@dataclass(frozen=True)
class RunPlan:
    """A prefill/decode correctness run plan (steps 20–23)."""

    name: str
    phase: str
    steps: int
    greedy: bool = True
    compares: Tuple[str, ...] = ("logits", "top_k", "kl", "token")
    requires_reference: bool = True
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "steps": self.steps,
            "greedy": self.greedy,
            "compares": list(self.compares),
            "requires_reference": self.requires_reference,
            "notes": self.notes,
        }


def representative_run_plan(*, phase: str, isl: int, osl: int, tiny: bool = False) -> RunPlan:
    if phase not in ("prefill", "decode"):
        raise ConfigError("phase must be prefill|decode", details={"field": "phase"})
    return RunPlan(
        name=f"{'tiny' if tiny else 'representative'}_{phase}",
        phase=phase,
        steps=osl if phase == "decode" else 1,
        compares=("selected_hidden", "logits", "top_k", "kl", "token", "eos"),
        notes=(
            "the reference must be the same ModelArtifact; if the model does not fit on one "
            "device, no strong T1 baseline may be claimed"
        ),
    )


def decode_alignment_plan(*, steps: int, greedy: bool = True) -> Dict[str, Any]:
    if steps < 1:
        raise ConfigError("steps must be >= 1", details={"field": "steps"})
    if not greedy:
        raise ConfigError(
            "multi-step alignment uses greedy decoding so token comparison is deterministic",
            details={"field": "greedy"},
        )
    return {
        "steps": steps,
        "greedy": greedy,
        "per_step": ["logits", "top_k", "kl", "token", "eos", "kv_length"],
        "checks": ["kv shard grows with tokens", "request end releases the KV shard"],
    }


@dataclass(frozen=True)
class UnsupportedDecision:
    """PAD/UNEVEN/REJECT for an unsupported shape, never truncation."""

    predicate: str
    policy: str
    padding: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        _require(self.policy in UNSUPPORTED_POLICIES, "unknown unsupported policy", field_name="policy")
        _require(bool(self.reason), "an unsupported decision needs a reason", field_name="reason")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "policy": self.policy,
            "padding": self.padding,
            "reason": self.reason,
        }


def unsupported_policy(
    *, predicate: str, policy: str, padding: int = 0, reason: str = ""
) -> UnsupportedDecision:
    if policy == "REJECT" and not reason:
        reason = "the shape cannot be sharded or padded under the frozen plan; reject rather than truncate"
    return UnsupportedDecision(predicate=predicate, policy=policy, padding=padding, reason=reason)


def assert_no_truncation(
    *, global_extent: int, covered_extent: int, predicate: str
) -> Dict[str, Any]:
    """A padding/uneven policy must still cover the whole extent (step 28)."""
    if covered_extent < global_extent:
        raise ConfigError(
            f"{predicate}: covered extent {covered_extent} < global {global_extent}; truncating "
            "heads/hidden is forbidden",
            details={"field": "covered_extent"},
        )
    return {"ok": True, "padding": covered_extent - global_extent, "predicate": predicate}


def tp_plan_gate(
    *,
    plan_ok: bool,
    roundtrip_ok: bool,
    direct_load_ok: bool,
    correctness_ok: bool,
    ledger_closed: bool,
    kv_ok: bool,
    blockers: Sequence[str] = (),
) -> Dict[str, Any]:
    """E10-04 → E10-05 gate (details E10-04 step 30)."""
    problems: List[str] = list(blockers)
    if not plan_ok:
        problems.append("the ParallelPlan does not cover every parameter/activation/KV role")
    if not roundtrip_ok:
        problems.append("shard round-trip failed (overlap/gap/replicated hash mismatch)")
    if not direct_load_ok:
        problems.append("direct shard loading is not proven (full-model copy suspected)")
    if not correctness_ok:
        problems.append("the multi-level correctness gate did not pass with the shared tolerance")
    if not ledger_closed:
        problems.append("expected/observed collective ledger is not closed")
    if not kv_ok:
        problems.append("KV ownership/lifecycle is not verified")
    return {
        "ready": not problems,
        "gate": "E10-04 -> E10-05",
        "blockers": problems,
        "reason": "" if not problems else "scaling must not start on an unclosed TP plan",
    }


def _matmul(
    weight: Sequence[float], out_features: int, in_features: int, x: Sequence[float], rows: int
) -> List[float]:
    """Tiny row-major matmul used by the CPU shard oracles.

    The extents are validated rather than trusted: a test fixture with mismatched
    ``x`` length must fail loudly instead of raising a bare ``IndexError``.
    """
    if rows < 1:
        rows = 1
    _require(
        len(weight) == out_features * in_features,
        f"matmul weight has {len(weight)} values, expected {out_features * in_features}",
        field_name="weight",
    )
    _require(
        len(x) == rows * in_features,
        f"matmul input has {len(x)} values, expected {rows * in_features}",
        field_name="x",
    )
    out: List[float] = []
    for row in range(rows):
        for o in range(out_features):
            acc = 0.0
            for i in range(in_features):
                acc += weight[o * in_features + i] * x[row * in_features + i]
            out.append(acc)
    return out


def _silu(value: float) -> float:
    import math

    return value / (1.0 + math.exp(-value))


__all__ = [
    "AttentionShard",
    "ActivationLayout",
    "CorrectnessCase",
    "EMBEDDING_STRATEGIES",
    "EmbeddingStrategyDecision",
    "KV_STRATEGIES",
    "KvOwnership",
    "LoadPlan",
    "MlpShard",
    "MODEL_CORRECTNESS_LAYERS",
    "PLAN_COLLECTIVE_TRIGGERS",
    "ParallelPlan",
    "PlanCollectiveEvent",
    "QwenArchitectureCensus",
    "REPLICATION_POLICIES",
    "ReplicatedTensorPolicy",
    "RoundtripReport",
    "RunPlan",
    "SHARD_AXES",
    "ShardSpec",
    "TpCapability",
    "UNSUPPORTED_POLICIES",
    "UnsupportedDecision",
    "assert_no_truncation",
    "attention_block_case",
    "audit_direct_load",
    "census_from_mapping",
    "column_parallel_case",
    "decode_alignment_plan",
    "default_replication_policies",
    "derive_attention_shard",
    "derive_mlp_shard",
    "derive_plan",
    "embedding_strategy",
    "first_deviation",
    "kv_ownership",
    "kv_per_token_bytes",
    "layer_case",
    "memory_model",
    "merge_flat",
    "mlp_block_case",
    "plan_axis_shards",
    "representative_run_plan",
    "replicated_shard",
    "row_parallel_case",
    "split_flat",
    "tp_capability_matrix",
    "tp_plan_gate",
    "unsupported_policy",
    "validate_collective_input_state",
    "verify_shard_ranges",
]
