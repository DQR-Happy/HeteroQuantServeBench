"""E03-02 performance primitives: shape matrix, byte/FLOP conventions,
statistics, mechanism-region labelling and heatmap rendering.

This module is the *pure* half of ``E03-02`` (RMSNorm shape 性能热力图与适用区间).
It freezes everything that must not be decided after seeing a number:

* the **pre-registered shape matrix** (:func:`shape_plan`) with an explicit
  ``source`` tag per cell (``S02_RUNTIME`` / ``DESIGN_MANDATORY`` /
  ``DESIGN_BOUNDARY`` / ``SYNTHETIC_SCALING``);
* the **S02 call matrix** (:func:`s02_call_matrix`) derived from the E02-02
  workload definitions, used to weight shapes by real call counts instead of
  averaging them arithmetically;
* the frozen **byte / FLOP conventions** (:data:`BYTE_CONVENTION_ID`,
  :data:`FLOP_CONVENTION_ID`) and their derived metrics;
* the **case plan** (:func:`build_case_plan`) with the pre-launch status of
  every cell (``ELIGIBLE`` vs ``EXPECTED_UNSUPPORTED``) so a cell can never be
  relabelled after it fails;
* **statistics** (median / P95 / MAD / CV / IQR + bootstrap CI) and the
  guard-band **winner / tie** rule seeded by E02-09;
* **mechanism-region labels** that are *candidates* for E03-04 to confirm with
  hardware counters (a low effective GB/s alone is never labelled
  "memory-bound");
* a dependency-free **SVG heatmap renderer** so the deliverables can be produced
  without matplotlib.

It imports neither ``torch`` nor any HQSB backend, so it cannot share a code
path with the kernels under test and stays unit-testable without a GPU. Do not
add CUDA/torch imports here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hqsb.benchmark.hotspot_decision import GUARD_BAND, NOMINAL_ENVELOPE

# ── identity ────────────────────────────────────────────────────────────

EXPERIMENT_ID = "E03-02"
STAGE = "S03"

# ── frozen conventions (protocol §6) ────────────────────────────────────

#: Identifies the logical-byte accounting used by every derived bandwidth in
#: this experiment: one read of ``x``, one mathematical read of ``w`` and one
#: write of ``y`` per element. It is *not* a claim about DRAM traffic (``w`` is
#: reused across rows and may be served from cache).
BYTE_CONVENTION_ID = "hqsb.s03.rmsnorm.logical_bytes.3x_elem.v1"

#: Identifies the declared-FLOP accounting: per row ``H`` squares,
#: ``H-1`` reduction adds, one multiply by ``1/H``, one ``+eps``, one ``rsqrt``
#: and ``2H`` multiply-accumulates (``x*rstd`` and ``*w``) => ``4H + 2``.
#: ``rsqrt`` counts as one operation, which does *not* reflect its real
#: throughput cost; the same convention is used for every variant.
FLOP_CONVENTION_ID = "hqsb.s03.rmsnorm.declared_flops.4H_plus_2.v1"

BYTES_PER_ELEMENT: Dict[str, int] = {"fp32": 4, "fp16": 2}

DTYPES: Tuple[str, ...] = ("fp32", "fp16")

#: Frozen claim handed over by E03-01 §12 (which variants support which dtype).
VARIANT_SUPPORT: Dict[str, Tuple[str, ...]] = {
    "fp32": ("v0_shared", "v1_warp_shuffle", "v2_vectorized"),
    "fp16": ("v2_vectorized",),
}
UNSUPPORTED_VARIANT_PAIRS: Dict[str, Tuple[str, ...]] = {
    "fp32": (),
    "fp16": ("v0_shared", "v1_warp_shuffle"),
}

#: The C ABI variant codes (ops/cuda/rmsnorm/src/rmsnorm_c_api.cu).
VARIANT_CODE: Dict[str, int] = {
    "auto": 0,
    "reference": 1,
    "v0_shared": 2,
    "v1_warp_shuffle": 3,
    "v2_vectorized": 4,
}
DTYPE_CODE: Dict[str, int] = {"fp32": 0, "fp16": 1}

#: Launch configuration used by the dispatcher for every timed case
#: (``kDefaultBlockSize`` in rmsnorm_dispatcher.cu). E03-04 owns the sweep.
DISPATCH_BLOCK_SIZE = 256

#: Default RMSNorm epsilon of the frozen model config (S00/S02), used only as
#: a fallback when ``config.json`` is unavailable.
DEFAULT_EPSILON = 1.0e-6

#: Input mode used for every timed case. E03-01 already covered 14 modes for
#: correctness; timing uses one well-conditioned mode so the numbers stay
#: comparable, and the mode is recorded per case.
TIMED_INPUT_MODE = "random_normal"

# ── pre-registered shape matrix (protocol §4) ───────────────────────────

#: ``rows`` anchors required by the checklist: decode (1), small batch
#: (3/7/15 non-powers-of-two so conclusions are not倍数-only), power-of-two
#: small batch (16), synthetic scaling (64..4096).
MANDATORY_ROWS: Tuple[int, ...] = (1, 3, 7, 15, 16, 64, 256, 1024, 4096)

#: ``H`` values the checklist makes mandatory (100/128/512/2048/6144/8192).
MANDATORY_H: Tuple[int, ...] = (100, 128, 512, 2048, 6144, 8192)

#: Boundary / odd ``H`` (tail + partial warp + non-vectorizable).
BOUNDARY_H: Tuple[int, ...] = (31, 33, 101, 129, 511, 513, 2047, 2049)

#: ``H`` used for the full ``rows`` cross-product (the "core" tier).
CORE_H: Tuple[int, ...] = (100, 128, 512, 2048)

#: ``rows`` anchors used for the extended-``H`` tier (large H is expensive to
#: materialise, so the row axis is reduced there - explicitly, not silently).
H_EXT_ROWS: Tuple[int, ...] = (1, 16, 512)

# ── S02 runtime facts (E02-02 shape census + workload YAML) ─────────────

S02_LAYERS = 28
S02_HIDDEN = 2048
S02_HEAD_DIM = 128
S02_Q_HEADS = 16
S02_KV_HEADS = 8

#: ``name -> (input_tokens, output_tokens)`` frozen by
#: ``configs/benchmarks/jetson_qwen3_fp16.yaml`` and read back from the E02-02
#: census (one request per workload run, batch size 1, greedy).
S02_WORKLOADS: Dict[str, Tuple[int, int]] = {
    "tiny": (32, 16),
    "short": (128, 32),
    "balanced": (512, 128),
    "long_prefill": (2048, 32),
    "decode_heavy": (128, 256),
    "long_balanced": (2048, 128),
}


def s02_decode_calls(output_tokens: int) -> int:
    """Decode-step RMSNorm invocations for one request (``OSL - 1`` steps)."""
    return max(int(output_tokens) - 1, 0)


#: ``(module_role, kind)`` pairs of one Qwen3-1.7B layer that run RMSNorm.
#: ``kind`` selects the shape family: the two layer norms run over the hidden
#: dimension of every token, the Q/K norms over ``heads x head_dim``.
S02_ROLES: Tuple[Tuple[str, str], ...] = (
    ("layers.*.input_layernorm", "hidden"),
    ("layers.*.post_attention_layernorm", "hidden"),
    ("layers.*.self_attn.q_norm", "head_q"),
    ("layers.*.self_attn.k_norm", "head_k"),
)


def s02_role_shape(kind: str, input_tokens: int, phase: str) -> Tuple[int, int]:
    """``(rows, hidden)`` of one module instance call."""
    if kind == "hidden":
        return (int(input_tokens) if phase == "prefill" else 1, S02_HIDDEN)
    if kind == "head_q":
        return (
            int(input_tokens) * S02_Q_HEADS if phase == "prefill" else S02_Q_HEADS,
            S02_HEAD_DIM,
        )
    if kind == "head_k":
        return (
            int(input_tokens) * S02_KV_HEADS if phase == "prefill" else S02_KV_HEADS,
            S02_HEAD_DIM,
        )
    raise ValueError(f"unknown role kind {kind!r}")


def s02_call_matrix() -> List[Dict[str, Any]]:
    """Per-workload RMSNorm call matrix derived from S02.

    One row per ``(workload, phase, module-role)`` and therefore per exact
    ``(rows, hidden)``. Two call counts are kept apart on purpose:

    * ``calls_per_module_instance`` - how often *one* module instance runs in
      that phase (1 during prefill, ``OSL-1`` during decode). This is the
      quantity the E02-02 census records, so it can be validated against it;
    * ``calls_per_request`` - ``layers x instances x calls_per_module_instance``,
      the weight used by the shape-weighted speedup (never an arithmetic mean).

    ``rows x hidden`` follows the layer-norm / Q-norm / K-norm geometry, so the
    hidden-width norm (H=2048) and the head-dim norm (H=128) stay separate.
    """
    rows: List[Dict[str, Any]] = []
    for name, (isl, osl) in S02_WORKLOADS.items():
        steps = s02_decode_calls(osl)
        for role, kind in S02_ROLES:
            prefill_rows, prefill_hidden = s02_role_shape(kind, isl, "prefill")
            rows.append(
                _call_matrix_row(
                    workload=name, isl=isl, osl=osl, role=role, phase="prefill",
                    shape=(prefill_rows, prefill_hidden),
                    calls_per_module=1,
                )
            )
            if steps > 0:
                decode_rows, decode_hidden = s02_role_shape(kind, isl, "decode")
                rows.append(
                    _call_matrix_row(
                        workload=name, isl=isl, osl=osl, role=role, phase="decode",
                        shape=(decode_rows, decode_hidden),
                        calls_per_module=steps,
                    )
                )
    return rows


def _call_matrix_row(
    *, workload: str, isl: int, osl: int, role: str, phase: str,
    shape: Tuple[int, int], calls_per_module: int,
) -> Dict[str, Any]:
    rows, hidden = shape
    return {
        "workload": workload,
        "input_tokens": isl,
        "output_tokens": osl,
        "phase": phase,
        "module_role": role,
        "rows": int(rows),
        "hidden": int(hidden),
        "dtype": "fp16",
        "calls_per_module_instance": int(calls_per_module),
        "calls_per_layer": int(calls_per_module),
        "calls_per_request": int(S02_LAYERS * calls_per_module),
        "shape_id": shape_id(int(rows), int(hidden)),
        "source": (
            "E02-02 census (module/phase/shape/call_count) x "
            "configs/benchmarks/jetson_qwen3_fp16.yaml (ISL/OSL) x "
            f"num_hidden_layers={S02_LAYERS}"
        ),
    }


def shape_call_weights() -> Dict[str, Dict[str, Any]]:
    """Aggregate the call matrix into ``shape_id -> call weight``."""
    weights: Dict[str, Dict[str, Any]] = {}
    for row in s02_call_matrix():
        entry = weights.setdefault(
            row["shape_id"],
            {
                "shape_id": row["shape_id"],
                "rows": row["rows"],
                "hidden": row["hidden"],
                "dtype": row["dtype"],
                "calls_per_request_total": 0,
                "calls_by_workload": {},
                "phases": set(),
                "module_roles": set(),
            },
        )
        entry["calls_per_request_total"] += row["calls_per_request"]
        entry["calls_by_workload"][row["workload"]] = (
            entry["calls_by_workload"].get(row["workload"], 0) + row["calls_per_request"]
        )
        entry["phases"].add(row["phase"])
        entry["module_roles"].add(row["module_role"])
    for entry in weights.values():
        entry["phases"] = sorted(entry["phases"])
        entry["module_roles"] = sorted(entry["module_roles"])
    return weights


def shape_id(rows: int, hidden: int) -> str:
    return f"r{int(rows)}_h{int(hidden)}"


def shape_plan() -> List[Dict[str, Any]]:
    """Pre-registered shape matrix with one ``source`` tag per cell.

    A shape that appears in more than one tier keeps **all** its tags (e.g.
    ``r1_h2048`` is both the mandatory decode anchor and the S02 runtime decode
    shape), so the runtime coverage map cannot be inflated by relabelling.
    """
    shapes: Dict[str, Dict[str, Any]] = {}

    def add(rows: int, hidden: int, source: str, note: str) -> None:
        sid = shape_id(rows, hidden)
        entry = shapes.setdefault(
            sid,
            {
                "shape_id": sid,
                "rows": int(rows),
                "hidden": int(hidden),
                "sources": [],
                "notes": [],
                "call_weight": 0,
            },
        )
        if source not in entry["sources"]:
            entry["sources"].append(source)
        if note not in entry["notes"]:
            entry["notes"].append(note)

    # Tier CORE - full rows x H cross product.
    for hidden in CORE_H:
        for rows in MANDATORY_ROWS:
            add(rows, hidden, "DESIGN_MANDATORY", "core tier: mandatory rows x H cross product")
    # Tier H_EXT - mandatory large H + boundary H at reduced row anchors.
    for hidden in tuple(MANDATORY_H[4:]) + BOUNDARY_H:
        for rows in H_EXT_ROWS:
            src = "DESIGN_MANDATORY" if hidden in MANDATORY_H else "DESIGN_BOUNDARY"
            add(rows, hidden, src, "extended tier: large/odd H at rows in (1, 16, 512)")
    # Synthetic scaling rows on the mandatory H=2048/128 to expose the saturation knee.
    for hidden in (128, 2048):
        for rows in (2, 4, 8, 8192):
            add(rows, hidden, "SYNTHETIC_SCALING", "saturation-knee row anchors")

    # Tier S02_RUNTIME - every real RMSNorm shape class of the six workloads.
    for isl, _osl in S02_WORKLOADS.values():
        add(isl, S02_HIDDEN, "S02_RUNTIME", "layers.*.input_layernorm / post_attention_layernorm prefill")
        add(isl * S02_Q_HEADS, S02_HEAD_DIM, "S02_RUNTIME", "layers.*.self_attn.q_norm prefill")
        add(isl * S02_KV_HEADS, S02_HEAD_DIM, "S02_RUNTIME", "layers.*.self_attn.k_norm prefill")
    add(1, S02_HIDDEN, "S02_RUNTIME", "layers.*.input_layernorm / post_attention_layernorm decode")
    add(S02_Q_HEADS, S02_HEAD_DIM, "S02_RUNTIME", "layers.*.self_attn.q_norm decode")
    add(S02_KV_HEADS, S02_HEAD_DIM, "S02_RUNTIME", "layers.*.self_attn.k_norm decode")

    weights = shape_call_weights()
    for sid, entry in shapes.items():
        if sid in weights:
            entry["call_weight"] = int(weights[sid]["calls_per_request_total"])
            entry["sources"].append("S02_CALL_WEIGHTED")
            entry["sources"] = sorted(set(entry["sources"]))
        entry["call_weight"] = int(entry.get("call_weight", 0))
    return sorted(shapes.values(), key=lambda e: (int(e["hidden"]), int(e["rows"])))


#: Shapes on which the *framework* baseline (plain torch expression) is also
#: timed. Explicitly reduced: S02 real shapes plus the three row anchors on the
#: two real widths. The framework baseline is stored separately and never mixed
#: into the V0/V1/V2 heatmaps.
FRAMEWORK_BASELINE_ROWS: Tuple[int, ...] = (1, 16, 512)
FRAMEWORK_BASELINE_H: Tuple[int, ...] = (128, 2048)
FRAMEWORK_VARIANT = "framework_baseline"


def framework_baseline_shapes() -> List[str]:
    """Pre-registered shapes that get an extra framework-baseline measurement."""
    ids = set()
    for entry in shape_plan():
        if "S02_RUNTIME" in entry["sources"]:
            ids.add(entry["shape_id"])
    for hidden in FRAMEWORK_BASELINE_H:
        for rows in FRAMEWORK_BASELINE_ROWS:
            if any(
                e["rows"] == rows and e["hidden"] == hidden for e in shape_plan()
            ):
                ids.add(shape_id(rows, hidden))
    return sorted(ids)


# ── case plan ───────────────────────────────────────────────────────────


def build_case_plan() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pre-registered case table (one row per forced (shape, dtype, variant)).

    Status is decided *before* any launch:

    * ``ELIGIBLE`` - the variant is claimed to support the dtype and the shape
      is inside the canonical contiguous/aligned domain that E03-01 verified;
    * ``EXPECTED_UNSUPPORTED`` - a claimed-unsupported (dtype, variant) pair is
      kept as a cell (never deleted) so it can be *verified* rejected.

    A later failure can never be relabelled as "expected unsupported".
    """
    shapes = shape_plan()
    framework_shapes = set(framework_baseline_shapes())
    cases: List[Dict[str, Any]] = []
    for entry in shapes:
        for dtype in DTYPES:
            for variant in VARIANT_SUPPORT[dtype]:
                cases.append(
                    _case(
                        entry,
                        dtype,
                        variant,
                        status="ELIGIBLE",
                        group="main",
                    )
                )
            for variant in UNSUPPORTED_VARIANT_PAIRS[dtype]:
                cases.append(
                    _case(
                        entry,
                        dtype,
                        variant,
                        status="EXPECTED_UNSUPPORTED",
                        group="unsupported",
                    )
                )
            if dtype == "fp16" and entry["shape_id"] in framework_shapes:
                cases.append(
                    _case(
                        entry,
                        dtype,
                        FRAMEWORK_VARIANT,
                        status="ELIGIBLE",
                        group="framework_baseline",
                    )
                )
    meta = {
        "shape_count": len(shapes),
        "framework_baseline_shapes": sorted(framework_shapes),
        "block_size": DISPATCH_BLOCK_SIZE,
        "input_mode": TIMED_INPUT_MODE,
        "variants_per_dtype": {k: list(v) for k, v in VARIANT_SUPPORT.items()},
        "unsupported_pairs": {k: list(v) for k, v in UNSUPPORTED_VARIANT_PAIRS.items()},
        "byte_convention_id": BYTE_CONVENTION_ID,
        "flop_convention_id": FLOP_CONVENTION_ID,
        "tier_reduction_note": (
            "The extended-H tier uses rows in (1, 16, 512) because materialising "
            "H=6144/8192 for every row anchor multiplies memory traffic without "
            "adding a new mechanism question. The reduction is recorded here "
            "before execution and is visible as missing cells in the matrices."
        ),
    }
    return cases, meta


def _case(
    entry: Mapping[str, Any],
    dtype: str,
    variant: str,
    *,
    status: str,
    group: str,
) -> Dict[str, Any]:
    rows = int(entry["rows"])
    hidden = int(entry["hidden"])
    input_key = f"{entry['shape_id']}|{dtype}|{TIMED_INPUT_MODE}"
    return {
        "case_id": f"{input_key}|{variant}",
        "input_key": input_key,
        "shape_id": entry["shape_id"],
        "rows": rows,
        "hidden": hidden,
        "dtype": dtype,
        "variant": variant,
        "group": group,
        "status": status,
        "shape_sources": list(entry["sources"]),
        "call_weight": int(entry.get("call_weight", 0) or 0),
        "block_size": DISPATCH_BLOCK_SIZE,
    }


def expected_case_ids(cases: Sequence[Mapping[str, Any]]) -> List[str]:
    """Sorted case-id set of the claimed domain (used to prove "not run" != "pass")."""
    return sorted(str(case["case_id"]) for case in cases)


# ── bytes / FLOPs / derived metrics (protocol §6) ───────────────────────


def logical_bytes(rows: int, hidden: int, dtype: str) -> int:
    """``rows x H x 3 x sizeof(dtype)`` under :data:`BYTE_CONVENTION_ID`."""
    return int(rows) * int(hidden) * 3 * BYTES_PER_ELEMENT[dtype]


def physical_tensor_bytes(rows: int, hidden: int, dtype: str) -> int:
    """Bytes of a single ``(rows, hidden)`` tensor in ``dtype``."""
    return int(rows) * int(hidden) * BYTES_PER_ELEMENT[dtype]


def declared_flops(rows: int, hidden: int) -> int:
    """``rows x (4H + 2)`` under :data:`FLOP_CONVENTION_ID`."""
    return int(rows) * (4 * int(hidden) + 2)


def derived_metrics(
    *,
    rows: int,
    hidden: int,
    dtype: str,
    device_seconds: float,
    host_seconds: Optional[float] = None,
    submit_seconds: Optional[float] = None,
    bridge_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Turn raw durations into the frozen derived table for one case.

    ``device_seconds`` must be the **device-event** latency; host-side numbers
    are derived with their own keys so a device bandwidth can never be computed
    from a host time.
    """
    if device_seconds is None or device_seconds <= 0:
        raise ValueError(f"device_seconds must be > 0, got {device_seconds}")
    nbytes = logical_bytes(rows, hidden, dtype)
    flops = declared_flops(rows, hidden)
    metrics: Dict[str, Any] = {
        "byte_convention_id": BYTE_CONVENTION_ID,
        "flop_convention_id": FLOP_CONVENTION_ID,
        "logical_bytes": nbytes,
        "physical_tensor_bytes": physical_tensor_bytes(rows, hidden, dtype),
        "declared_flops": flops,
        "arithmetic_intensity_flop_per_byte": flops / nbytes,
        "device_seconds": device_seconds,
        "effective_GBps": nbytes / device_seconds / 1e9,
        "effective_GFLOPs": flops / device_seconds / 1e9,
        "nominal_dram_ceiling_fraction": (
            (nbytes / device_seconds) / NOMINAL_ENVELOPE["peak_dram_bandwidth"]
        ),
    }
    if host_seconds is not None:
        metrics["host_seconds"] = host_seconds
        metrics["host_effective_GBps"] = nbytes / host_seconds / 1e9
    if submit_seconds is not None:
        metrics["submit_seconds"] = submit_seconds
        metrics["submit_effective_GBps"] = nbytes / submit_seconds / 1e9
    if bridge_seconds is not None:
        metrics["bridge_seconds"] = bridge_seconds
    return metrics


# ── statistics (protocol §5.3/§7) ───────────────────────────────────────


def summarize_samples(values: Sequence[float]) -> Dict[str, Any]:
    """Median / P95 / MAD / CV / IQR / min / max of one sample vector."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p95": None, "mad": None, "cv": None,
                "iqr": None, "min": None, "max": None, "mean": None}
    median = float(np.median(arr))
    q1, q3 = (float(v) for v in np.percentile(arr, [25, 75]))
    mad = float(np.median(np.abs(arr - median)))
    mean = float(np.mean(arr))
    return {
        "count": int(arr.size),
        "median": median,
        "p95": float(np.percentile(arr, 95)),
        "mad": mad,
        "cv": (float(np.std(arr, ddof=1) / mean) if arr.size > 1 and mean else 0.0),
        "iqr": q3 - q1,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": mean,
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 4000,
    seed: int = 20260918,
) -> Dict[str, Any]:
    """Percentile bootstrap CI of the median (deterministic given ``seed``)."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"level": level, "low": None, "high": None, "n": 0, "note": "no samples"}
    if arr.size == 1:
        return {
            "level": level,
            "low": float(arr[0]),
            "high": float(arr[0]),
            "n": 1,
            "note": "single sample: the interval collapses and is not a CI",
        }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    stats = np.median(arr[draws], axis=1)
    low, high = (float(v) for v in np.percentile(stats, [50 * (1 - level), 50 * (1 + level)]))
    return {
        "level": level,
        "low": low,
        "high": high,
        "n": int(arr.size),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def paired_speedup(
    baseline: Sequence[float], candidate: Sequence[float]
) -> Dict[str, Any]:
    """Paired ``baseline/candidate`` ratios, sample by sample.

    Pairing is by *sample index* inside one process/group block, so the ratio
    inherits the local temperature/clock state instead of comparing the median
    of one version against the median of another.
    """
    a = [float(v) for v in baseline]
    b = [float(v) for v in candidate]
    if len(a) != len(b):
        raise ValueError(
            f"paired_speedup needs equal-length samples, got {len(a)} and {len(b)}"
        )
    ratios = [x / y for x, y in zip(a, b) if y > 0 and x > 0]
    summary = summarize_samples(ratios)
    return {
        "paired_count": len(ratios),
        "ratios": ratios,
        "median": summary["median"],
        "p95": summary["p95"],
        "cv": summary["cv"],
        "ci": bootstrap_ci(ratios),
        "summary": summary,
    }


def verdict_from_ci(
    ci: Mapping[str, Any], *, guard: float = GUARD_BAND["micro_min_relative_improvement"]
) -> str:
    """Classify one paired speedup against the frozen guard band.

    * ``CANDIDATE_WINS`` - CI lower bound above ``1 + guard``;
    * ``TIE_OR_WITHIN_GUARD_BAND`` - CI upper bound above ``1 + guard`` but the
      lower bound is not, i.e. the gain is not distinguishable from noise;
    * ``REGRESSION`` - CI upper bound below ``1``;
    * ``NO_GAIN`` - everything else (a small, non-significant difference).
    """
    low, high = ci.get("low"), ci.get("high")
    if low is None or high is None:
        return "INSUFFICIENT_SAMPLES"
    if low > 1.0 + guard:
        return "CANDIDATE_WINS"
    if high < 1.0:
        return "REGRESSION"
    if high > 1.0 + guard:
        return "TIE_OR_WITHIN_GUARD_BAND"
    return "NO_GAIN"


def winner_row(
    *,
    shape_id_value: str,
    rows: int,
    hidden: int,
    dtype: str,
    paired_by_variant: Mapping[str, Sequence[float]],
    guard: float = GUARD_BAND["micro_min_relative_improvement"],
) -> Dict[str, Any]:
    """Winner / tie / baseline-retention decision for one shape block.

    ``paired_by_variant`` maps candidate variant name to its paired speedup
    sample vector *relative to v0_shared*. The decision never uses a single
    best sample, never averages latencies across variants and never drops a
    regression.
    """
    entries: List[Dict[str, Any]] = []
    for variant, ratios in sorted(paired_by_variant.items()):
        if not ratios:
            continue
        ci = bootstrap_ci(list(ratios))
        entries.append(
            {
                "variant": variant,
                "paired_count": len(list(ratios)),
                "median_speedup": float(np.median(np.asarray(list(ratios), dtype=float))),
                "ci": ci,
                "verdict": verdict_from_ci(ci, guard=guard),
            }
        )
    baseline = {
        "variant": "v0_shared",
        "paired_count": 0,
        "median_speedup": 1.0,
        "ci": {"level": 0.95, "low": 1.0, "high": 1.0, "n": 0, "note": "baseline"},
        "verdict": "BASELINE",
    }
    ranking = sorted(entries, key=lambda e: (e["median_speedup"] or 0.0), reverse=True)
    winners = [e for e in ranking if e["verdict"] == "CANDIDATE_WINS"]
    best = winners[0] if winners else baseline
    tie: List[str] = [best["variant"]]
    best_median = best["median_speedup"] or 0.0
    for entry in [baseline] + ranking:
        if entry["variant"] == best["variant"]:
            continue
        if entry["median_speedup"] is None:
            continue
        if entry["median_speedup"] >= best_median / (1.0 + guard):
            tie.append(entry["variant"])
    return {
        "shape_id": shape_id_value,
        "rows": int(rows),
        "hidden": int(hidden),
        "dtype": dtype,
        "guard_band": guard,
        "decision": best["variant"],
        "decision_kind": (
            "BASELINE_RETAINED" if best["variant"] == "v0_shared" else "CANDIDATE_WINS"
        ),
        "tie_group": sorted(set(tie)),
        "candidates": ranking,
        "baseline": baseline,
        "note": (
            "decision_kind=BASELINE_RETAINED means no candidate crossed the guard "
            "band; it is not a claim that v0 is optimal"
        ),
    }


# ── mechanism-region labelling (protocol §9, candidates only) ───────────


def classify_region(
    *,
    rows: int,
    hidden: int,
    dtype: str,
    effective_gbps: float,
    ceiling_gbps: Optional[float],
    sm_count: Optional[int],
    variant: Optional[str] = None,
    block_size: int = DISPATCH_BLOCK_SIZE,
) -> Dict[str, Any]:
    """Pre-registered *candidate* labels for E03-04 to confirm with counters.

    The rules are deliberately conservative: a low effective GB/s is never
    turned into "memory-bound" here, because the protocol requires counters to
    distinguish "not enough work to fill the GPU" from "memory path saturated".
    """
    labels: List[str] = []
    reasons: List[str] = []
    grid = int(rows)
    if grid == 1:
        labels.append("launch_or_single_cta_limited_candidate")
        reasons.append("grid has a single block: fixed launch + one-CTA latency dominate")
    elif sm_count and grid < sm_count:
        labels.append("insufficient_grid_parallelism_candidate")
        reasons.append(f"grid ({grid}) < SM count ({sm_count}): SMs idle by construction")
    elif sm_count and grid < 4 * sm_count:
        labels.append("grid_parallelism_sensitive_candidate")
        reasons.append(f"grid ({grid}) < 4 x SM count ({4 * sm_count})")

    vec_ok = (hidden % 4 == 0) if dtype == "fp32" else (hidden % 2 == 0)
    if not vec_ok:
        labels.append("tail_or_scalar_path")
        reasons.append(
            f"hidden={hidden} is not a multiple of "
            f"{4 if dtype == 'fp32' else 2}: the vector path cannot be used for whole rows"
        )
    elif variant in ("v2_vectorized",):
        labels.append("vectorization_eligible")
        reasons.append("hidden satisfies the vector width condition for this dtype")

    if ceiling_gbps and ceiling_gbps > 0:
        frac = effective_gbps / ceiling_gbps
        if frac >= 0.80:
            labels.append("memory_throughput_sensitive_candidate")
            reasons.append(f"effective GB/s reaches {frac:.2%} of the measured device ceiling")
        elif frac < 0.25:
            reasons.append(
                f"effective GB/s is only {frac:.2%} of the device ceiling; a counter "
                "check is required before calling this memory-bound"
            )
    if not labels:
        labels.append("insufficient_evidence")
    return {"labels": sorted(set(labels)), "reasons": reasons, "grid_blocks": grid,
            "block_size": int(block_size)}


# ── matrix / heatmap rendering ──────────────────────────────────────────


def build_matrix(
    summaries: Iterable[Mapping[str, Any]],
    *,
    dtype: str,
    variant: str,
    value_key: str,
    aggregate: str = "median",
    row_labels: Optional[Sequence[int]] = None,
    col_labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Matrix view ``H (rows) x rows (columns)`` of one metric/variant/dtype.

    Missing cells (shape never measured for this variant) stay ``None`` and are
    rendered as such - they are never interpolated from a neighbour. Pass the
    dtype-wide ``row_labels``/``col_labels`` to make a variant that lacks a
    shape visibly empty instead of silently shrinking the axes.
    """
    picked: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for rec in summaries:
        if rec.get("dtype") != dtype or rec.get("variant") != variant:
            continue
        stats = rec.get(value_key) or {}
        value = stats.get(aggregate) if isinstance(stats, Mapping) else None
        picked[(int(rec["hidden"]), int(rec["rows"]))] = {
            "value": value,
            "status": rec.get("status"),
            "case_id": rec.get("case_id"),
        }
    if row_labels is None:
        row_labels = sorted({h for h, _ in picked})
    else:
        row_labels = sorted(int(v) for v in row_labels)
    if col_labels is None:
        col_labels = sorted({r for _, r in picked})
    else:
        col_labels = sorted(int(v) for v in col_labels)
    values: List[List[Optional[float]]] = []
    status: List[List[Optional[str]]] = []
    for hidden in row_labels:
        value_row: List[Optional[float]] = []
        status_row: List[Optional[str]] = []
        for rows in col_labels:
            cell = picked.get((hidden, rows))
            value_row.append(None if cell is None else cell["value"])
            status_row.append(None if cell is None else cell["status"])
        values.append(value_row)
        status.append(status_row)
    return {
        "dtype": dtype,
        "variant": variant,
        "value_key": value_key,
        "aggregate": aggregate,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "values": values,
        "status": status,
    }


def _lerp_color(t: float, low: tuple, mid: tuple, high: tuple) -> str:
    t = min(max(float(t), 0.0), 1.0)
    if t < 0.5:
        a, b, u = low, mid, t / 0.5
    else:
        a, b, u = mid, high, (t - 0.5) / 0.5
    rgb = [int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


_COLOR_SCHEMES = {
    # ascending "hotter is worse" for latency / time
    "sequential": ((0x1F, 0x77, 0xB4), (0xFE, 0xE0, 0x8B), (0xB2, 0x22, 0x22)),
    # diverging around 1.0 for speedup (blue = slower, red = faster)
    "diverging": ((0x21, 0x66, 0xAC), (0xF7, 0xF7, 0xF7), (0xB2, 0x18, 0x2B)),
}

_STATUS_FILL = {
    "EXPECTED_UNSUPPORTED": "#9e9e9e",
    "BLOCKED_BY_CORRECTNESS": "#000000",
    "FAIL": "#000000",
    "BLOCKED_BY_LAYOUT": "#616161",
    "NOT_RUN": "#e0e0e0",
}


def render_heatmap_svg(
    *,
    title: str,
    matrix: Mapping[str, Any],
    unit: str = "",
    scale: str = "sequential",
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cell_w: int = 62,
    cell_h: int = 26,
    decimals: int = 3,
    legend_note: str = "",
) -> str:
    """Render one heatmap as a standalone SVG string (no third-party deps).

    Colour limits are computed from robust percentiles unless the caller passes
    fixed ones, so a single extreme cell cannot flatten the whole palette. The
    exact numeric value is always printed inside the cell, and the raw numeric
    table is stored next to the image.
    """
    row_labels = list(matrix["row_labels"])
    col_labels = list(matrix["col_labels"])
    values = matrix["values"]
    status = matrix.get("status") or [[None] * len(col_labels) for _ in row_labels]

    finite = [v for r in values for v in r if v is not None]
    if vmin is None or vmax is None:
        if finite:
            lo_pct, hi_pct = (2.0, 98.0) if scale == "sequential" else (5.0, 95.0)
            auto_min = float(np.percentile(finite, lo_pct))
            auto_max = float(np.percentile(finite, hi_pct))
            if center is not None:
                span = max(abs(auto_min - center), abs(auto_max - center), 1e-9)
                vmin = center - span if vmin is None else vmin
                vmax = center + span if vmax is None else vmax
            else:
                vmin = auto_min if vmin is None else vmin
                vmax = auto_max if vmax is None else vmax
        else:
            vmin, vmax = (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1e-9

    left, top = 96, 74
    width = left + cell_w * max(len(col_labels), 1) + 24
    height = top + cell_h * max(len(row_labels), 1) + 58

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="16" font-weight="bold">{_xml(title)}</text>',
        f'<text x="{left}" y="46" font-size="11" fill="#444">'
        f'dtype={matrix["dtype"]} variant={matrix["variant"]} metric={matrix["value_key"]} '
        f'unit={_xml(unit or "-")} scale[{vmin:.4g}, {vmax:.4g}]</text>',
    ]
    if legend_note:
        parts.append(
            f'<text x="{left}" y="61" font-size="10" fill="#666">{_xml(legend_note)}</text>'
        )
    for ci, col in enumerate(col_labels):
        x = left + ci * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 8}" font-size="10" text-anchor="middle" '
            f'fill="#333">{_xml(f"r={col}")}</text>'
        )
    for ri, rlab in enumerate(row_labels):
        y = top + ri * cell_h + cell_h / 2 + 3.5
        parts.append(
            f'<text x="{left - 6}" y="{y}" font-size="10" text-anchor="end" '
            f'fill="#333">H={rlab}</text>'
        )
    for ri, rlab in enumerate(row_labels):
        for ci, clab in enumerate(col_labels):
            x = left + ci * cell_w
            y = top + ri * cell_h
            value = values[ri][ci]
            cell_status = status[ri][ci]
            if value is None:
                fill = _STATUS_FILL.get(cell_status or "NOT_RUN", "#e0e0e0")
                text = {
                    "EXPECTED_UNSUPPORTED": "N/A",
                    "BLOCKED_BY_CORRECTNESS": "FAIL",
                    "FAIL": "FAIL",
                    "NOT_RUN": "-",
                }.get(cell_status or "NOT_RUN", "-")
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" '
                    f'stroke="#ffffff" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 3.5}" font-size="10" '
                    f'text-anchor="middle" fill="#ffffff">{_xml(text)}</text>'
                )
                continue
            if scale == "diverging" and center is not None:
                span = max(abs(vmax - center), abs(center - vmin), 1e-9)
                t = (float(value) - center) / span
                t = 0.5 + t / 2.0
            else:
                t = (float(value) - vmin) / (vmax - vmin)
            fill = _lerp_color(t, *_COLOR_SCHEMES[scale])
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" '
                f'stroke="#ffffff" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 3.5}" font-size="9.5" '
                f'text-anchor="middle" fill="#111">{_fmt(value, decimals)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _xml(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fmt(value: Optional[float], decimals: int) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 10 ** (-decimals):
        return f"{v:.3g}"
    return f"{v:.{decimals}f}"


#: Fill colours for categorical / status heatmaps. A cell that is *missing*
#: (never measured) and a cell that is *unsupported* must never look alike, so
#: they get different colours and keep their raw status text in the cell.
CATEGORY_FILL: Dict[str, str] = {
    "MEASURED": "#cfe8f3",
    "EXPECTED_UNSUPPORTED": "#9e9e9e",
    "FAIL_CORRECTNESS": "#000000",
    "BLOCKED_BY_CORRECTNESS": "#000000",
    "BLOCKED_BY_LAYOUT": "#616161",
    "EXPECTED_UNSUPPORTED_NOT_REJECTED": "#8e24aa",
    "NOT_RUN": "#e0e0e0",
    "MISSING": "#f5f5f5",
}

CATEGORY_LABEL: Dict[str, str] = {
    "MEASURED": "ok",
    "EXPECTED_UNSUPPORTED": "N/A",
    "FAIL_CORRECTNESS": "FAIL",
    "BLOCKED_BY_CORRECTNESS": "FAIL",
    "BLOCKED_BY_LAYOUT": "LAYOUT",
    "EXPECTED_UNSUPPORTED_NOT_REJECTED": "NOT-REJ",
    "NOT_RUN": "-",
    "MISSING": "-",
}


def render_categorical_svg(
    *,
    title: str,
    row_labels: Sequence[int],
    col_labels: Sequence[int],
    labels: Sequence[Sequence[Optional[str]]],
    palette: Optional[Mapping[str, str]] = None,
    cell_w: int = 62,
    cell_h: int = 26,
    legend_note: str = "",
) -> str:
    """Render a status / winner heatmap (no numeric colour scale)."""
    palette = dict(palette or CATEGORY_FILL)
    n_rows, n_cols = len(row_labels), len(col_labels)
    left, top = 96, 74
    width = left + cell_w * max(n_cols, 1) + 24
    height = top + cell_h * max(n_rows, 1) + 82
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="16" font-weight="bold">{_xml(title)}</text>',
    ]
    if legend_note:
        parts.append(
            f'<text x="{left}" y="46" font-size="11" fill="#444">{_xml(legend_note)}</text>'
        )
    for ci, col in enumerate(col_labels):
        x = left + ci * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 8}" font-size="10" text-anchor="middle" '
            f'fill="#333">r={col}</text>'
        )
    for ri, rlab in enumerate(row_labels):
        y = top + ri * cell_h + cell_h / 2 + 3.5
        parts.append(
            f'<text x="{left - 6}" y="{y}" font-size="10" text-anchor="end" '
            f'fill="#333">H={rlab}</text>'
        )
    for ri in range(n_rows):
        for ci in range(n_cols):
            x = left + ci * cell_w
            y = top + ri * cell_h
            category = (labels[ri][ci] if ri < len(labels) else None) or "MISSING"
            fill = palette.get(category, "#dddddd")
            text = CATEGORY_LABEL.get(category, category)
            ink = "#111" if fill.lower() in ("#cfe8f3", "#e0e0e0", "#f5f5f5") else "#ffffff"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" '
                f'stroke="#ffffff" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 3.5}" font-size="9.5" '
                f'text-anchor="middle" fill="{ink}">{_xml(text)}</text>'
            )
    legend_y = top + cell_h * max(n_rows, 1) + 18
    lx = left
    for category in sorted(palette):
        fill = palette[category]
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 9}" width="12" height="12" fill="{fill}" '
            'stroke="#999" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{legend_y}" font-size="10" fill="#333">'
            f'{_xml(category)}</text>'
        )
        lx += 18 + 7 * len(category)
    parts.append("</svg>")
    return "\n".join(parts)


def write_matrix_csv(path, matrix: Mapping[str, Any]) -> None:
    """Numeric table that must accompany every heatmap (protocol §8)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hidden\\rows"] + [str(c) for c in matrix["col_labels"]])
        for ri, rlab in enumerate(matrix["row_labels"]):
            row = [str(rlab)]
            for value in matrix["values"][ri]:
                row.append("" if value is None else repr(float(value)))
            writer.writerow(row)


def canonical_sha256(obj: Any) -> str:
    """SHA256 over a canonical (sorted-key, compact) JSON rendering."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percentile_of(value: float, values: Sequence[float]) -> Optional[float]:
    """Fraction of ``values`` at or below ``value`` (robust-position helper)."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None
    return float(np.count_nonzero(arr <= value) / arr.size)


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def geometric_mean(values: Sequence[float]) -> Optional[float]:
    arr = [float(v) for v in values if v is not None and float(v) > 0]
    if not arr:
        return None
    return float(math.exp(sum(math.log(v) for v in arr) / len(arr)))


__all__ = [
    "BOUNDARY_H",
    "BYTES_PER_ELEMENT",
    "BYTE_CONVENTION_ID",
    "CATEGORY_FILL",
    "CATEGORY_LABEL",
    "CORE_H",
    "DEFAULT_EPSILON",
    "DISPATCH_BLOCK_SIZE",
    "DTYPES",
    "DTYPE_CODE",
    "EXPERIMENT_ID",
    "FLOP_CONVENTION_ID",
    "FRAMEWORK_BASELINE_H",
    "FRAMEWORK_BASELINE_ROWS",
    "FRAMEWORK_VARIANT",
    "H_EXT_ROWS",
    "MANDATORY_H",
    "MANDATORY_ROWS",
    "S02_ROLES",
    "S02_WORKLOADS",
    "STAGE",
    "TIMED_INPUT_MODE",
    "UNSUPPORTED_VARIANT_PAIRS",
    "VARIANT_CODE",
    "VARIANT_SUPPORT",
    "bootstrap_ci",
    "build_case_plan",
    "build_matrix",
    "canonical_sha256",
    "classify_region",
    "declared_flops",
    "derived_metrics",
    "expected_case_ids",
    "framework_baseline_shapes",
    "geometric_mean",
    "is_power_of_two",
    "logical_bytes",
    "paired_speedup",
    "percentile_of",
    "physical_tensor_bytes",
    "render_categorical_svg",
    "render_heatmap_svg",
    "s02_call_matrix",
    "s02_decode_calls",
    "s02_role_shape",
    "safe_div",
    "shape_call_weights",
    "shape_id",
    "shape_plan",
    "summarize_samples",
    "verdict_from_ci",
    "winner_row",
    "write_matrix_csv",
]
