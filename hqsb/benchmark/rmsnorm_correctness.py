"""RMSNorm correctness primitives for E03-01.

This module is the *pure* half of ``E03-01`` (RMSNorm 数学语义、独立 Reference 与
正确性边界): it freezes the math contract, derives the pre-registered tolerance
table, generates deterministic inputs, provides an **FP64 oracle independent of
the CUDA kernels**, classifies special values, computes the error metrics and
decides a per-case verdict.

It deliberately imports neither ``torch`` nor any HQSB backend: the oracle here
cannot share a code path with the implementation under test (V0/V1/V2), and the
logic stays unit-testable without a GPU. Only ``numpy`` is required.

The two oracles required by the protocol §4 are split as follows:

* FP64 CPU oracle → :func:`fp64_oracle` (here, numpy float64, CPU);
* independent FP32 framework oracle → built by the experiment runner out of
  plain ``torch`` tensor ops (``x.float()``/``*``/``mean``/``rsqrt``), never
  V0/V1/V2, never the dispatcher;
* a third, analytically-derived **classification oracle**
  (:func:`expected_classes`) predicts the IEEE class of every output element
  directly from the input classes, without evaluating the formula numerically.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

# ── frozen math contract ────────────────────────────────────────────────

SEMANTIC_VERSION = "1.0.0"
EQUATION = (
    "y[r,j] = cast_out( cast_acc(x[r,j]) * cast_acc(w[j]) * "
    "rsqrt( (sum_k cast_acc(x[r,k])**2) / H + eps ) )"
)
REDUCTION_AXIS = "last dimension (H); leading dimensions are flattened into rows"
EPSILON_POSITION = "added to the mean-square, *inside* the rsqrt; scalar, finite, > 0"
WEIGHT_BROADCAST = "w is broadcast along the last dimension only"
OUT_SHAPE = "y.shape == x.shape (row-major, contiguous)"

#: NumPy dtype used for the declared input/weight/output dtype per contract key.
DTYPE_NUMPY = {"fp32": np.float32, "fp16": np.float16}

#: Accumulation dtype per declared input dtype (frozen: always float32 on device).
ACCUM_DTYPE = {"fp32": "float32", "fp16": "float32"}

#: Element classes used for the classification comparison (never ``equal_nan``).
CLASS_FINITE = 0
CLASS_POSINF = 1
CLASS_NEGINF = 2
CLASS_NAN = 3
CLASS_NAMES = {
    CLASS_FINITE: "finite",
    CLASS_POSINF: "posinf",
    CLASS_NEGINF: "neginf",
    CLASS_NAN: "nan",
}

#: The 14 pre-registered input modes (protocol §5.3).
INPUT_MODES = (
    "random_normal",
    "random_uniform",
    "zeros",
    "constant",
    "alternating",
    "tiny",
    "large_safe",
    "overflow_probe",
    "sparse",
    "weight_zero",
    "weight_signed",
    "nan_inject",
    "posinf",
    "neginf",
)

#: Modes that deliberately inject non-finite values after generation.
INJECTION_MODES = ("nan_inject", "posinf", "neginf")

#: ``(dtype, mode)`` cells whose numeric comparison against the FP64 oracle is a
#: **pre-registered divergence**, not a failure. The reason is frozen here so it
#: cannot be invented after seeing a result.
DECLARED_DIVERGENCE: Dict[tuple, str] = {
    ("fp32", "overflow_probe"): (
        "The frozen contract accumulates the square-sum in FP32. For *finite* "
        "inputs whose square (or sum of squares) overflows FP32, the device "
        "square-sum becomes +Inf, inv_rms = rsqrt(Inf) = 0 and the kernel "
        "returns exactly 0, while an FP64 oracle stays finite. The contract "
        "keeps the FP32 accumulation, so this cell is compared against the "
        "independent FP32 framework oracle instead, and the FP64 comparison is "
        "recorded as NOT_APPLICABLE_DECLARED_DIVERGENCE."
    ),
}

#: Absolute floor used in the L2 relative metric denominator (pre-registered;
#: never adjusted after seeing a result).
L2_FLOOR_ABS = 1.0e-6

#: Numeric bounds used by the input generators, with their derivation.
GENERATOR_BOUNDS = {
    "tiny": {
        "fp32": [1.0e-7, 1.0e-6],
        "fp16": [6.0e-5, 6.0e-4],
        "derivation": (
            "FP16 smallest normal is 6.10e-5, so an fp16 'tiny' draw must stay "
            "at/above it to avoid silently becoming a subnormal or zero; the "
            "fp32 row uses an even smaller range to stress the relative metric."
        ),
    },
    "large_safe": {
        "fp32": [512.0, 1024.0],
        "fp16": [512.0, 1024.0],
        "derivation": (
            "Upper bound for the largest tested H: x_max^2 * H_max = 1024^2 * "
            "8192 = 8.6e9 << FLT_MAX 3.4e38, so the FP32 square-sum stays "
            "finite for every tested shape (safety factor > 4e28)."
        ),
    },
    "overflow_probe": {
        "fp32": 1.0e20,
        "fp16": 65504.0,
        "derivation": (
            "FP32: 1e20 is finite in FP32 but its square is 1e40 -> +Inf, "
            "reaching the declared square-overflow cell. FP16: the largest "
            "finite fp16 (65504) is used, whose square is only 4.29e9 and "
            "therefore *cannot* overflow an FP32 accumulation, so the fp16 "
            "cell is a pure max-magnitude test and is NOT a divergence cell."
        ),
    },
}


def _tolerance_derivation() -> Dict[str, str]:
    return {
        "fp16": (
            "ceiling pre-registered by E02-09 s03_protocol: atol = 8 * 2^-11 "
            "(= 3.90625e-3, the FP16 output quantum at |y|<=1 with an 8-quantum "
            "budget), rtol = 2e-2, l2rel = 1e-3, cosine >= 0.9999. E03-01 keeps "
            "the ceiling (tightening is allowed, loosening is forbidden) and "
            "adds the shape class: H > 2048 may not widen atol (the FP32 "
            "accumulation order budget is already covered by the 8-quantum "
            "term), so the same atol applies to every fp16 shape."
        ),
        "fp32": (
            "derived here: FP32 output rounding contributes |y|*2^-24; the FP32 "
            "square-sum reordering contributes about H*u*mean(x^2) with "
            "u = 2^-24 = 5.96e-8, which is halved by the rsqrt. For H <= 2048 "
            "the worst-case term is 2048*5.96e-8 = 1.22e-4, so atol = 5e-4 is "
            "chosen (matches the frozen C3 tolerance in "
            "configs/operators/rmsnorm_v0.json); for H > 2048 the term grows to "
            "8192*5.96e-8 = 4.88e-4 and atol is widened to 2e-3 with an explicit "
            "documented derivation instead of silently relaxing one global "
            "number. rtol = 1e-5, l2rel = 5e-4, cosine >= 0.9999."
        ),
        "reference_budget_factor": (
            "the two oracles must agree within 50% of the operator tolerance on "
            "finite elements; otherwise the reference pair itself is not tight "
            "enough to gate a candidate."
        ),
    }


#: Pre-registered per-dtype/shape-class tolerance table (protocol §6).
TOLERANCE_TABLE = {
    "fp16": {
        "atol": 8.0 * (2.0 ** -11),
        "rtol": 2.0e-2,
        "l2rel": 1.0e-3,
        "cosine_min": 0.9999,
        "shape_class": "all_h",
    },
    "fp32": {
        "small_h": {
            "atol": 5.0e-4,
            "rtol": 1.0e-5,
            "l2rel": 5.0e-4,
            "cosine_min": 0.9999,
        },
        "large_h": {
            "atol": 2.0e-3,
            "rtol": 1.0e-5,
            "l2rel": 5.0e-4,
            "cosine_min": 0.9999,
        },
        "small_h_max": 2048,
    },
}


def tolerance_for(dtype: str, hidden: int) -> Dict[str, Any]:
    """Return the pre-registered tolerance entry for ``dtype``/``hidden``."""
    if dtype == "fp16":
        entry = dict(TOLERANCE_TABLE["fp16"])
        entry["derivation"] = _tolerance_derivation()["fp16"]
        return entry
    if dtype != "fp32":
        raise ValueError(f"unknown dtype {dtype!r}")
    table = TOLERANCE_TABLE["fp32"]
    key = "small_h" if int(hidden) <= table["small_h_max"] else "large_h"
    entry = dict(table[key])
    entry["shape_class"] = key
    entry["derivation"] = _tolerance_derivation()["fp32"]
    return entry


def reference_budget(dtype: str, hidden: int) -> Dict[str, Any]:
    """Tolerance applied to the *oracle-vs-oracle* comparison (50% of the gate)."""
    entry = tolerance_for(dtype, hidden)
    return {
        "atol": entry["atol"] * 0.5,
        "rtol": entry["rtol"] * 0.5,
        "l2rel": entry["l2rel"] * 0.5,
        "cosine_min": entry["cosine_min"],
        "derivation": _tolerance_derivation()["reference_budget_factor"],
    }


# ── hashing / seeds ─────────────────────────────────────────────────────


def canonical_json(obj: Any) -> str:
    """Canonical (sorted-key, compact) JSON serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(obj: Any) -> str:
    """SHA256 over the canonical JSON of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def derive_seed(case_id: str) -> int:
    """Derive a 32-bit seed from the case id (deterministic, machine-independent)."""
    digest = hashlib.sha256(f"hqsb-e03-01|{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ── dtype casting ───────────────────────────────────────────────────────


def cast_to_declared(values: np.ndarray, dtype: str) -> Dict[str, Any]:
    """Cast ``values`` to the declared dtype, recording any overflow events.

    The returned array is the **actual byte pattern** the oracle consumes: the
    protocol requires the cast-to-input-dtype values to be recorded, because a
    "tiny" float64 draw may already have become zero (or Inf) in the target
    dtype.
    """
    np_dtype = DTYPE_NUMPY[dtype]
    source = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        casted = source.astype(np_dtype)
    overflow = int(np.count_nonzero(np.isinf(casted) & np.isfinite(source)))
    underflow = int(np.count_nonzero((casted == 0) & (source != 0)))
    return {
        "array": casted,
        "cast_overflow_count": overflow,
        "cast_underflow_to_zero_count": underflow,
        "cast_warnings": sorted({str(w.message) for w in caught}),
    }


# ── deterministic generation ────────────────────────────────────────────


def _split_mode(mode: str) -> tuple:
    if mode == "weight_zero":
        return "random_normal", "zero"
    if mode == "weight_signed":
        return "random_normal", "signed"
    return mode, "normal"


def _generate_x(mode: str, rows: int, hidden: int, dtype: str, rng) -> np.ndarray:
    if mode in ("nan_inject", "posinf", "neginf"):
        mode = "random_normal"
    shape = (rows, hidden)
    if mode == "random_normal":
        return rng.standard_normal(shape)
    if mode == "random_uniform":
        return rng.uniform(-1.0, 1.0, shape)
    if mode == "zeros":
        return np.zeros(shape, dtype=np.float64)
    if mode == "constant":
        return np.full(shape, 0.75, dtype=np.float64)
    if mode == "alternating":
        pattern = np.array([1.5, -1.5], dtype=np.float64)
        return np.tile(pattern, (rows * hidden + 1) // 2)[: rows * hidden].reshape(shape)
    if mode == "tiny":
        lo, hi = GENERATOR_BOUNDS["tiny"][dtype]
        return rng.uniform(lo, hi, shape)
    if mode == "large_safe":
        lo, hi = GENERATOR_BOUNDS["large_safe"][dtype]
        magnitude = rng.uniform(lo, hi, shape)
        sign = np.where(rng.integers(0, 2, shape) == 0, -1.0, 1.0)
        return magnitude * sign
    if mode == "overflow_probe":
        probe = GENERATOR_BOUNDS["overflow_probe"][dtype]
        sign = np.where(rng.integers(0, 2, shape) == 0, -1.0, 1.0)
        return np.full(shape, probe, dtype=np.float64) * sign
    if mode == "sparse":
        dense = np.zeros(shape, dtype=np.float64)
        flat = dense.reshape(-1)
        idx = np.arange(0, rows * hidden, 97)
        flat[idx] = rng.uniform(-2.0, 2.0, idx.size)
        return flat.reshape(shape)
    raise ValueError(f"unknown input mode {mode!r}")


def _generate_w(mode: str, hidden: int, rng) -> np.ndarray:
    base = rng.uniform(0.5, 1.5, hidden)
    if mode == "zero":
        base[::16] = 0.0
    elif mode == "signed":
        sign = np.where(np.arange(hidden) % 2 == 0, -1.0, 1.0)
        base = base * sign
    return base


def generate_case_arrays(
    mode: str, rows: int, hidden: int, dtype: str, seed: int
) -> Dict[str, Any]:
    """Generate the frozen ``(x, w)`` pair for one case.

    The same bytes are handed to every variant; no variant ever calls a random
    number generator of its own (protocol §8 step 3).
    """
    rng = np.random.default_rng(np.random.PCG64(seed))
    x_mode, w_mode = _split_mode(mode)

    x_raw = _generate_x(x_mode, rows, hidden, dtype, rng)
    w_raw = _generate_w(w_mode, hidden, rng)

    x_cast = cast_to_declared(x_raw, dtype)
    w_cast = cast_to_declared(w_raw, dtype)
    x = x_cast["array"]
    w = w_cast["array"]

    injections: List[Dict[str, Any]] = []
    if x_mode in INJECTION_MODES:
        value = {
            "nan_inject": math.nan,
            "posinf": math.inf,
            "neginf": -math.inf,
        }[x_mode]
        # Two injection sites: the head of row 0 and the last element of the
        # last row (so the vector tail is exercised too).
        sites = [(0, 0)]
        if rows > 1 or hidden > 1:
            sites.append((rows - 1, hidden - 1))
        for row, col in sites:
            x[row, col] = np.array(value, dtype=DTYPE_NUMPY[dtype])
            injections.append(
                {
                    "flat_index": int(row * hidden + col),
                    "row": int(row),
                    "col": int(col),
                    "value": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf"),
                }
            )

    return {
        "x": x,
        "w": w,
        "injections": injections,
        "x_mode": x_mode,
        "w_mode": w_mode,
        "cast": {
            "x": {k: v for k, v in x_cast.items() if k != "array"},
            "w": {k: v for k, v in w_cast.items() if k != "array"},
        },
    }


# ── classification ──────────────────────────────────────────────────────


def classify(values: np.ndarray) -> np.ndarray:
    """Per-element IEEE class as an ``int8`` array (NaN / ±Inf / finite)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(v.shape, CLASS_FINITE, dtype=np.int8)
    pos = np.isposinf(v)
    neg = np.isneginf(v)
    nan = np.isnan(v)
    out[pos] = CLASS_POSINF
    out[neg] = CLASS_NEGINF
    out[nan] = CLASS_NAN
    return out


def class_counts(classes: np.ndarray) -> Dict[str, int]:
    classes = np.asarray(classes).reshape(-1)
    return {
        name: int(np.count_nonzero(classes == code))
        for code, name in CLASS_NAMES.items()
    }


def expected_classes(x: np.ndarray, rows: int, hidden: int) -> np.ndarray:
    """Analytic classification oracle derived from the frozen formula alone.

    * a row containing NaN → every element of that row is NaN (the reduction
      propagates NaN through ``inv_rms``);
    * else a row containing ±Inf → ``square-sum = +Inf`` → ``inv_rms = 0`` →
      finite lanes are exactly ``0`` and the ±Inf lanes become ``Inf * 0 = NaN``;
    * else every element is finite.

    No arithmetic is performed, so this oracle cannot share a bug with either
    numerical reference or with the kernels.
    """
    in_cls = classify(x).reshape(rows, hidden)
    out = np.full((rows, hidden), CLASS_FINITE, dtype=np.int8)
    for row in range(rows):
        cells = in_cls[row]
        if np.any(cells == CLASS_NAN):
            out[row] = CLASS_NAN
        elif np.any((cells == CLASS_POSINF) | (cells == CLASS_NEGINF)):
            out[row] = np.where(
                (cells == CLASS_POSINF) | (cells == CLASS_NEGINF),
                CLASS_NAN,
                CLASS_FINITE,
            )
    return out.reshape(-1)


def inf_row_finite_lanes_exact_zero(x: np.ndarray, rows: int, hidden: int) -> np.ndarray:
    """Boolean mask of elements that must be *exactly* 0.0 by construction.

    Those are the finite lanes of a row that contains ±Inf (``x * 0 = 0`` is
    exact in IEEE arithmetic, in the kernel and in the oracle alike).
    """
    in_cls = classify(x).reshape(rows, hidden)
    has_inf = np.any((in_cls == CLASS_POSINF) | (in_cls == CLASS_NEGINF), axis=1)
    has_nan = np.any(in_cls == CLASS_NAN, axis=1)
    finite = in_cls == CLASS_FINITE
    mask = finite & (has_inf & ~has_nan)[:, None]
    return mask.reshape(-1)


# ── oracles ─────────────────────────────────────────────────────────────


def fp64_oracle(
    x: np.ndarray, w: np.ndarray, rows: int, hidden: int, epsilon: float
) -> np.ndarray:
    """Independent FP64 CPU oracle of the frozen equation.

    Consumes the *cast* input bytes (exact in float64) and performs every
    operation in float64, so no kernel, no dispatcher and no CUDA code is
    reached. IEEE behaviour is preserved (``inf * 0 -> nan``).
    """
    xd = np.asarray(x, dtype=np.float64).reshape(rows, hidden)
    wd = np.asarray(w, dtype=np.float64).reshape(hidden)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        square_sum = np.sum(xd * xd, axis=1, keepdims=True)
        inverse_rms = 1.0 / np.sqrt(square_sum / float(hidden) + float(epsilon))
        out = xd * inverse_rms * wd
    return out.reshape(-1)


def fp32_square_sum_overflow(x: np.ndarray, rows: int, hidden: int) -> np.ndarray:
    """Per-row flag: would the FP32 square-sum overflow to +Inf?

    Evaluated on the *cast* inputs with a sequential float32 accumulation
    (``np.cumsum`` with ``dtype=float32``), which is the most overflow-prone
    order; it is therefore an upper bound on when the declared FP32
    accumulation of this row can saturate. No warning is raised for the
    deliberate overflow.
    """
    x32 = np.asarray(x, dtype=np.float64).reshape(rows, hidden).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squares = x32 * x32
            running = np.cumsum(squares, axis=1, dtype=np.float32)
    return ~np.isfinite(running[:, -1])


# ── metrics ─────────────────────────────────────────────────────────────


def error_metrics(
    candidate: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Multi-metric error report over ``mask`` (protocol §6).

    ``cosine`` is ``None`` (NOT_APPLICABLE) when either vector has zero norm —
    it is never coerced to 1. ``l2rel`` divides by
    ``max(||ref||, L2_FLOOR_ABS)`` with the pre-registered floor.
    """
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if mask is None:
        mask = np.ones(ref.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "applicable": False,
            "element_count": 0,
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
            "cosine": None,
            "cosine_applicable": False,
            "l2rel": None,
            "l2_floor_abs": L2_FLOOR_ABS,
        }

    c = cand[mask]
    r = ref[mask]
    with np.errstate(invalid="ignore", over="ignore"):
        diff = c - r
        abs_diff = np.abs(diff)
        norm_ref = float(np.linalg.norm(r))
        norm_cand = float(np.linalg.norm(c))
        norm_diff = float(np.linalg.norm(diff))
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        cosine = None
        if norm_ref > 0.0 and norm_cand > 0.0:
            cosine = float(np.dot(c, r) / (norm_cand * norm_ref))
        l2rel = norm_diff / max(norm_ref, L2_FLOOR_ABS)
    return {
        "applicable": True,
        "element_count": count,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "cosine": cosine,
        "cosine_applicable": cosine is not None,
        "l2rel": float(l2rel),
        "norm_reference": norm_ref,
        "norm_candidate": norm_cand,
        "l2_floor_abs": L2_FLOOR_ABS,
    }


def elementwise_violations(
    candidate: np.ndarray,
    reference: np.ndarray,
    tolerance: Mapping[str, Any],
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Mixed criterion ``abs <= atol OR abs <= rtol*|ref|`` over ``mask``."""
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    with np.errstate(invalid="ignore"):
        abs_diff = np.abs(cand - ref)
        allowed = np.maximum(
            float(tolerance["atol"]), float(tolerance["rtol"]) * np.abs(ref)
        )
        ok = (abs_diff <= allowed) | ~mask
        bad = ~ok
    bad_idx = np.flatnonzero(bad)
    worst = None
    if bad_idx.size:
        excess = abs_diff[bad_idx] - allowed[bad_idx]
        pick = int(bad_idx[int(np.argmax(excess))])
        worst = {
            "flat_index": pick,
            "candidate": float(cand[pick]),
            "reference": float(ref[pick]),
            "abs_error": float(abs_diff[pick]),
            "allowed": float(allowed[pick]),
        }

    # Tolerance headroom: the largest |error| / allowed ratio over the mask.
    # This is the meaningful "how close was the gate" number (an absolute
    # max_abs can exceed atol and still pass through the relative term).
    ratio = None
    ratio_index = None
    if np.any(mask):
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = np.where(mask, abs_diff, 0.0)
            ratio_map = scaled / np.maximum(allowed, np.finfo(np.float64).tiny)
        ratio_index = int(np.argmax(ratio_map))
        ratio = float(ratio_map[ratio_index])
    return {
        "violation_count": int(bad_idx.size),
        "passed": bool(bad_idx.size == 0),
        "worst": worst,
        "max_allowed_ratio": ratio,
        "max_allowed_ratio_index": ratio_index,
    }


def first_mismatch(
    *,
    candidate: np.ndarray,
    reference: np.ndarray,
    candidate_classes: np.ndarray,
    expected_class_array: np.ndarray,
    tolerance: Mapping[str, Any],
    hidden: int,
) -> Optional[Dict[str, Any]]:
    """Locate the first (lowest flat index) classification or numeric failure."""
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand_cls = np.asarray(candidate_classes).reshape(-1)
    exp_cls = np.asarray(expected_class_array).reshape(-1)

    class_bad = np.flatnonzero(cand_cls != exp_cls)
    with np.errstate(invalid="ignore"):
        abs_diff = np.abs(cand - ref)
        allowed = np.maximum(
            float(tolerance["atol"]), float(tolerance["rtol"]) * np.abs(ref)
        )
        finite_both = np.isfinite(ref) & np.isfinite(cand)
        numeric_bad = np.flatnonzero(finite_both & (abs_diff > allowed))
    candidates = [idx for idx in (class_bad, numeric_bad) if idx.size]
    if not candidates:
        return None
    flat = int(min(int(idx[0]) for idx in candidates))
    kind = "classification" if class_bad.size and flat == int(class_bad[0]) else "numeric"
    with np.errstate(invalid="ignore"):
        abs_err = abs(float(cand[flat]) - float(ref[flat]))
        allowed_v = float(
            max(float(tolerance["atol"]), float(tolerance["rtol"]) * abs(float(ref[flat])))
        )
    return {
        "kind": kind,
        "flat_index": flat,
        "row": flat // int(hidden),
        "col": flat % int(hidden),
        "candidate": float(cand[flat]),
        "reference": float(ref[flat]),
        "abs_error": abs_err,
        "allowed": allowed_v,
        "candidate_class": CLASS_NAMES[int(cand_cls[flat])],
        "expected_class": CLASS_NAMES[int(exp_cls[flat])],
    }


# ── per-case verdict ────────────────────────────────────────────────────


def judge_case(
    checks: Mapping[str, Optional[bool]], na_reasons: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:
    """Apply the pre-registered per-case gate.

    ``checks`` values are ``True`` (passed), ``False`` (failed) or ``None``
    (not applicable — which *must* carry a declared reason, otherwise a None is
    treated as a failure so that "not run" can never be counted as a pass).
    """
    na_reasons = dict(na_reasons or {})
    failed = []
    undocumented_na = []
    for name, value in checks.items():
        if value is False:
            failed.append(name)
        elif value is None:
            if name in na_reasons:
                continue
            undocumented_na.append(name)
    failed.extend(undocumented_na)
    return {
        "status": "FAIL" if failed else "PASS",
        "failed_checks": sorted(failed),
        "checks": {k: v for k, v in checks.items()},
        "na_reasons": na_reasons,
    }


# ── coverage matrix ─────────────────────────────────────────────────────


def coverage_matrix(
    records: Iterable[Mapping[str, Any]],
    axes: Sequence[str] = ("dtype", "mode", "variant", "shape_class"),
) -> Dict[str, Any]:
    """Aggregate case records into a coverage matrix on the given axes.

    A combination that was never executed simply has no entry (it is *not*
    inferred from a neighbouring cell), so an empty corner shows up as a
    missing key rather than as a silent pass.
    """
    materialized = [dict(rec) for rec in records]
    per_axis: Dict[str, Dict[str, Dict[str, int]]] = {axis: {} for axis in axes}
    combos = [tuple(str(rec.get(axis, "?")) for axis in axes) for rec in materialized]
    combo_stats: Dict[str, Dict[str, int]] = {}
    for rec, key in zip(materialized, combos):
        status = str(rec.get("status", "NOT_RUN"))
        combo_key = "|".join(key)
        stats = combo_stats.setdefault(combo_key, {"total": 0, "pass": 0, "fail": 0})
        stats["total"] += 1
        if status == "PASS":
            stats["pass"] += 1
        elif status == "FAIL":
            stats["fail"] += 1
        for axis, value in zip(axes, key):
            bucket = per_axis[axis].setdefault(value, {"total": 0, "pass": 0, "fail": 0})
            bucket["total"] += 1
            if status == "PASS":
                bucket["pass"] += 1
            elif status == "FAIL":
                bucket["fail"] += 1
    return {
        "axes": list(axes),
        "per_axis": per_axis,
        "combinations": combo_stats,
        "total_cases": len(materialized),
        "total_pass": sum(1 for r in materialized if r.get("status") == "PASS"),
        "total_fail": sum(1 for r in materialized if r.get("status") == "FAIL"),
    }


__all__ = [
    "ACCUM_DTYPE",
    "CLASS_FINITE",
    "CLASS_NAN",
    "CLASS_NAMES",
    "CLASS_NEGINF",
    "CLASS_POSINF",
    "DECLARED_DIVERGENCE",
    "DTYPE_NUMPY",
    "EPSILON_POSITION",
    "EQUATION",
    "GENERATOR_BOUNDS",
    "INPUT_MODES",
    "L2_FLOOR_ABS",
    "OUT_SHAPE",
    "REDUCTION_AXIS",
    "SEMANTIC_VERSION",
    "TOLERANCE_TABLE",
    "canonical_json",
    "cast_to_declared",
    "class_counts",
    "classify",
    "coverage_matrix",
    "derive_seed",
    "elementwise_violations",
    "error_metrics",
    "expected_classes",
    "first_mismatch",
    "fp32_square_sum_overflow",
    "fp64_oracle",
    "generate_case_arrays",
    "inf_row_finite_lanes_exact_zero",
    "judge_case",
    "reference_budget",
    "sha256_bytes",
    "sha256_hex",
    "tolerance_for",
]
