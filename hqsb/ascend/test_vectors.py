"""Shared case matrix, deterministic test vectors and guard/poison detection.

Every backend reads **one** artifact.  "Each backend generates its own random
input" is an explicit prohibition (E09-03 §11) because it makes a correctness
comparison unfalsifiable — so vectors are generated once, hashed, and the hash
travels with every result row.

Two mechanisms do the work a plain ``allclose`` cannot:

* **guard/poison regions** (E09-02 step 5): bytes before and after the legal
  range are filled with a fixed pattern and re-checked afterwards.  A kernel
  that writes the right values *and* overruns the buffer still fails, which is
  the difference between "the answer looks right" and "the answer is right".
* **splits** (E09-03 step 3, E09-04 §5): development / tuning / confirmation /
  adversarial.  :meth:`CaseMatrix.seal_confirmation` freezes the confirmation
  split and records the seal hash, so "we kept tuning on the holdout" becomes
  detectable rather than a matter of trust.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from hqsb.core.errors import ConfigError, UsageError
from hqsb.core.fingerprint import canonical_json, sha256_hex
from hqsb.ascend.operators import (
    OPERATOR_ADD,
    OPERATOR_ROW_REDUCE_SUM,
    OPERATOR_RMSNORM,
    oracle_add,
    oracle_rmsnorm,
    oracle_row_reduce_sum,
)

GENERATOR_VERSION = "hqsb.ascend.test_vectors/1.0.0"

SPLIT_DEVELOPMENT = "development"
SPLIT_TUNING = "tuning"
SPLIT_CONFIRMATION = "confirmation"
SPLIT_ADVERSARIAL = "adversarial"

SPLITS: Tuple[str, ...] = (
    SPLIT_DEVELOPMENT,
    SPLIT_TUNING,
    SPLIT_CONFIRMATION,
    SPLIT_ADVERSARIAL,
)

# ── shape categories (E09-02 §5.1) ───────────────────────────────────────────

SHAPE_ZERO = "zero"
SHAPE_ONE = "one"
SHAPE_LESS_THAN_BLOCK = "less_than_block"
SHAPE_EXACT_ALIGNED = "exact_aligned"
SHAPE_TAIL_MINUS_ONE = "tail_minus_one"
SHAPE_TAIL_PLUS_ONE = "tail_plus_one"
SHAPE_MULTI_TILE = "multi_tile"
SHAPE_CORE_TAIL = "core_tail"
SHAPE_FEWER_THAN_CORES = "fewer_work_than_cores"
SHAPE_LARGE = "large"
SHAPE_MODEL_HIDDEN = "model_hidden"
SHAPE_PREFILL_FLAT = "prefill_flat"
SHAPE_DECODE_SMALL = "decode_small"

SHAPE_CATEGORIES: Tuple[str, ...] = (
    SHAPE_ZERO,
    SHAPE_ONE,
    SHAPE_LESS_THAN_BLOCK,
    SHAPE_EXACT_ALIGNED,
    SHAPE_TAIL_MINUS_ONE,
    SHAPE_TAIL_PLUS_ONE,
    SHAPE_MULTI_TILE,
    SHAPE_CORE_TAIL,
    SHAPE_FEWER_THAN_CORES,
    SHAPE_LARGE,
    SHAPE_MODEL_HIDDEN,
    SHAPE_PREFILL_FLAT,
    SHAPE_DECODE_SMALL,
)

# ── data categories (E09-02 §5.2, E09-03 §5.2) ───────────────────────────────

DATA_RAMP = "ramp"
DATA_ALTERNATING = "alternating_sign"
DATA_ZEROS = "all_zero"
DATA_ONES = "all_one"
DATA_CONSTANT = "constant"
DATA_TINY = "very_small"
DATA_HUGE = "very_large"
DATA_HIGH_DYNAMIC_RANGE = "high_dynamic_range"
DATA_CANCELLATION = "cancellation_prone"
DATA_RANDOM_NORMAL = "random_normal"
DATA_RANDOM_UNIFORM = "random_uniform"
DATA_MODEL_HIDDEN_STATE = "model_hidden_state"
DATA_NAN = "nan"
DATA_POS_INF = "pos_inf"
DATA_NEG_INF = "neg_inf"

DATA_CATEGORIES: Tuple[str, ...] = (
    DATA_RAMP,
    DATA_ALTERNATING,
    DATA_ZEROS,
    DATA_ONES,
    DATA_CONSTANT,
    DATA_TINY,
    DATA_HUGE,
    DATA_HIGH_DYNAMIC_RANGE,
    DATA_CANCELLATION,
    DATA_RANDOM_NORMAL,
    DATA_RANDOM_UNIFORM,
    DATA_MODEL_HIDDEN_STATE,
    DATA_NAN,
    DATA_POS_INF,
    DATA_NEG_INF,
)

#: Categories whose behaviour the OperatorSpec must define before they may be
#: generated at all (E09-02 §5.2: "NaN/±Inf only after the spec defines them").
NON_FINITE_CATEGORIES: Tuple[str, ...] = (DATA_NAN, DATA_POS_INF, DATA_NEG_INF)

GAMMA_ONES = "gamma_all_one"
GAMMA_RANDOM = "gamma_random"
GAMMA_EXTREME = "gamma_with_extremes"

GAMMA_CATEGORIES: Tuple[str, ...] = (GAMMA_ONES, GAMMA_RANDOM, GAMMA_EXTREME)

#: Poison byte pattern written into the guard regions.
POISON_PATTERN = 0xA5

#: Value written into an output before the kernel runs, so "not written at all"
#: is distinguishable from "written correctly by accident".
OUTPUT_SENTINEL = float("nan")


@dataclass(frozen=True)
class CaseSpec:
    """One point of the frozen case matrix."""

    case_id: str
    operator: str
    rows: int
    hidden: int
    dtype: str
    split: str
    shape_category: str
    data_category: str
    gamma_category: str = GAMMA_ONES
    eps: float = 1e-6
    layout: str = "contiguous_nd"
    seed: int = 0
    block_dim: int = 1
    tile_elems: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ConfigError(
                f"case {self.case_id!r} has unknown split {self.split!r}; allowed {list(SPLITS)}",
                details={"case_id": self.case_id, "split": self.split},
            )
        if self.shape_category not in SHAPE_CATEGORIES:
            raise ConfigError(
                f"case {self.case_id!r} has unknown shape category {self.shape_category!r}",
                details={"case_id": self.case_id},
            )
        if self.data_category not in DATA_CATEGORIES:
            raise ConfigError(
                f"case {self.case_id!r} has unknown data category {self.data_category!r}",
                details={"case_id": self.case_id},
            )
        if self.gamma_category not in GAMMA_CATEGORIES:
            raise ConfigError(
                f"case {self.case_id!r} has unknown gamma category {self.gamma_category!r}",
                details={"case_id": self.case_id},
            )
        if self.rows < 0 or self.hidden < 0:
            raise ConfigError(
                f"case {self.case_id!r} has a negative dimension", details={"case_id": self.case_id}
            )
        if self.operator not in (OPERATOR_ADD, OPERATOR_ROW_REDUCE_SUM, OPERATOR_RMSNORM):
            raise ConfigError(
                f"case {self.case_id!r} names an operator S09 has not frozen: {self.operator!r}",
                details={"case_id": self.case_id, "operator": self.operator},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operator": self.operator,
            "rows": self.rows,
            "hidden": self.hidden,
            "dtype": self.dtype,
            "split": self.split,
            "shape_category": self.shape_category,
            "data_category": self.data_category,
            "gamma_category": self.gamma_category,
            "eps": self.eps,
            "layout": self.layout,
            "seed": self.seed,
            "block_dim": self.block_dim,
            "tile_elems": self.tile_elems,
            "notes": self.notes,
        }

    @property
    def hash(self) -> str:
        return sha256_hex(canonical_json(self.as_dict()))


@dataclass(frozen=True)
class CaseMatrix:
    """The frozen matrix plus the confirmation seal."""

    cases: Tuple[CaseSpec, ...]
    sealed: bool = False
    seal_hash: str = ""
    sealed_at: str = ""

    def __post_init__(self) -> None:
        ids = [case.case_id for case in self.cases]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ConfigError(
                f"duplicate case_id(s) {duplicates}; two cases with one id cannot be reported separately",
                details={"duplicates": duplicates},
            )
        if self.sealed and not self.seal_hash:
            raise ConfigError("a sealed matrix must carry its seal hash", details={"field": "seal_hash"})

    def __len__(self) -> int:
        return len(self.cases)

    def by_split(self, split: str) -> Tuple[CaseSpec, ...]:
        return tuple(case for case in self.cases if case.split == split)

    def by_category(self, **filters: str) -> Tuple[CaseSpec, ...]:
        selected = self.cases
        for key, value in filters.items():
            selected = tuple(case for case in selected if getattr(case, key) == value)
        return selected

    def coverage(self) -> Dict[str, Any]:
        def census(attribute: str) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for case in self.cases:
                key = str(getattr(case, attribute))
                counts[key] = counts.get(key, 0) + 1
            return dict(sorted(counts.items()))

        return {
            "total": len(self.cases),
            "by_split": census("split"),
            "by_shape_category": census("shape_category"),
            "by_data_category": census("data_category"),
            "by_dtype": census("dtype"),
            "by_operator": census("operator"),
            "missing_shape_categories": sorted(set(SHAPE_CATEGORIES) - set(census("shape_category"))),
            "sealed": self.sealed,
            "seal_hash": self.seal_hash,
        }

    def split_manifest(self) -> Dict[str, Any]:
        """The ``cases/split_manifest.json`` artifact of E09-03 §9."""
        return {
            "generator_version": GENERATOR_VERSION,
            "sealed": self.sealed,
            "seal_hash": self.seal_hash,
            "sealed_at": self.sealed_at,
            "coverage": self.coverage(),
            "cases": [case.as_dict() for case in self.cases],
        }

    def seal_confirmation(self, sealed_at: str) -> "CaseMatrix":
        """Freeze the confirmation split (E09-03 step 27, E09-04 step 24).

        Sealing is what makes "run the holdout once" checkable: after this, any
        change to a confirmation case changes :attr:`seal_hash`, and
        :func:`confirm_seal_intact` reports the drift instead of trusting memory.
        """
        if not sealed_at:
            raise UsageError("sealing needs a timestamp; an undated seal cannot be ordered against tuning")
        confirmation = [case.as_dict() for case in self.by_split(SPLIT_CONFIRMATION)]
        if not confirmation:
            raise UsageError(
                "there is no confirmation case to seal; a holdout that does not exist cannot "
                "protect against tuning overfit"
            )
        return CaseMatrix(
            cases=self.cases,
            sealed=True,
            seal_hash=sha256_hex(canonical_json(confirmation)),
            sealed_at=sealed_at,
        )

    def confirm_seal_intact(self) -> Dict[str, Any]:
        recomputed = sha256_hex(
            canonical_json([case.as_dict() for case in self.by_split(SPLIT_CONFIRMATION)])
        )
        return {
            "sealed": self.sealed,
            "seal_hash": self.seal_hash,
            "recomputed": recomputed,
            "intact": bool(self.sealed) and recomputed == self.seal_hash,
            "note": (
                ""
                if (self.sealed and recomputed == self.seal_hash)
                else "the confirmation split changed after sealing; it is no longer a holdout and "
                "must be re-declared as a new experiment revision"
            ),
        }


# ── deterministic data generation ────────────────────────────────────────────


def _round_to_dtype(value: float, dtype: str) -> float:
    """Emulate the target dtype's finite precision without numpy.

    The golden must be the *logical* FP64 value; the input, however, has to be
    representable in the dtype under test, or the case silently measures cast
    error instead of kernel error.
    """
    if dtype == "fp32":
        return float.fromhex(float(value).hex())
    if dtype == "fp16":
        try:
            import struct as _struct

            return _struct.unpack("<e", _struct.pack("<e", value))[0]
        except (OverflowError, ValueError):
            return math.copysign(float("inf"), value)
    if dtype == "bf16":
        import struct as _struct

        packed = _struct.pack("<f", value)
        truncated = _struct.unpack("<I", packed)[0] & 0xFFFF0000
        return _struct.unpack("<f", _struct.pack("<I", truncated))[0]
    raise UsageError(f"unsupported dtype {dtype!r}", details={"dtype": dtype})


def generate_data(category: str, count: int, *, seed: int, dtype: str = "fp32") -> List[float]:
    """Deterministic data for one category.  Same inputs ⇒ same bytes."""
    if count < 0:
        raise UsageError(f"count must be non-negative, got {count}", details={"count": count})
    if category in NON_FINITE_CATEGORIES:
        value = {
            DATA_NAN: float("nan"),
            DATA_POS_INF: float("inf"),
            DATA_NEG_INF: float("-inf"),
        }[category]
        return [value] * count
    rng = random.Random((seed, category, count, dtype).__hash__() & 0xFFFFFFFF)
    if category == DATA_RAMP:
        raw = [float(index) for index in range(count)]
    elif category == DATA_ALTERNATING:
        raw = [float(index) * (1.0 if index % 2 == 0 else -1.0) for index in range(count)]
    elif category == DATA_ZEROS:
        raw = [0.0] * count
    elif category == DATA_ONES:
        raw = [1.0] * count
    elif category == DATA_CONSTANT:
        raw = [3.5] * count
    elif category == DATA_TINY:
        raw = [1e-8 + index * 1e-10 for index in range(count)]
    elif category == DATA_HUGE:
        raw = [1e4 + index for index in range(count)]
    elif category == DATA_HIGH_DYNAMIC_RANGE:
        raw = [(1e6 if index % 3 == 0 else (1e-6 if index % 3 == 1 else 1.0)) for index in range(count)]
    elif category == DATA_CANCELLATION:
        # +v, -v, +v/2, -v/2 … : the exact sum is 0, so any residual is pure error.
        raw = []
        for index in range(count):
            magnitude = 1024.0 / (1 + index // 2)
            raw.append(magnitude if index % 2 == 0 else -magnitude)
    elif category == DATA_RANDOM_NORMAL:
        raw = [rng.gauss(0.0, 1.0) for _ in range(count)]
    elif category == DATA_RANDOM_UNIFORM:
        raw = [rng.uniform(-1.0, 1.0) for _ in range(count)]
    elif category == DATA_MODEL_HIDDEN_STATE:
        # Stand-in for a de-identified captured hidden state.  A real capture must
        # record model/run/layer/token provenance and its hash (E09-03 §5.2); this
        # generator only produces the deterministic shape of one.
        raw = [rng.gauss(0.0, 2.0) * (1.0 + 0.25 * math.sin(index / 7.0)) for index in range(count)]
    else:
        raise UsageError(
            f"unknown data category {category!r}; allowed {list(DATA_CATEGORIES)}",
            details={"category": category},
        )
    return [_round_to_dtype(value, dtype) for value in raw]


def generate_gamma(category: str, hidden: int, *, seed: int, dtype: str = "fp32") -> List[float]:
    if hidden <= 0:
        raise UsageError(f"gamma needs hidden >= 1, got {hidden}", details={"hidden": hidden})
    rng = random.Random((seed, "gamma", category, hidden).__hash__() & 0xFFFFFFFF)
    if category == GAMMA_ONES:
        raw = [1.0] * hidden
    elif category == GAMMA_RANDOM:
        raw = [rng.uniform(0.5, 1.5) for _ in range(hidden)]
    elif category == GAMMA_EXTREME:
        raw = []
        for index in range(hidden):
            if index % 5 == 0:
                raw.append(1e-7)
            elif index % 5 == 1:
                raw.append(-2.0)
            else:
                raw.append(rng.uniform(0.1, 10.0))
    else:
        raise UsageError(f"unknown gamma category {category!r}", details={"category": category})
    return [_round_to_dtype(value, dtype) for value in raw]


# ── guard / poison ───────────────────────────────────────────────────────────


@dataclass
class GuardedBuffer:
    """A logical payload wrapped in poison-filled guard regions.

    ``verify()`` answers three separate questions, because conflating them is how
    a buffer overrun survives review:

    1. did the guard regions change?            → out-of-bounds write
    2. did the inputs change?                  → in-place mutation
    3. is any output element still the sentinel? → output not fully written
    """

    payload: List[float]
    guard_elements: int = 16
    sentinel: float = OUTPUT_SENTINEL
    _prefix: List[float] = field(default_factory=list, init=False, repr=False)
    _suffix: List[float] = field(default_factory=list, init=False, repr=False)
    _input_snapshot: List[float] = field(default_factory=list, init=False, repr=False)
    _is_output: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.guard_elements < 0:
            raise UsageError("guard_elements must be non-negative", details={"guard_elements": self.guard_elements})
        poison = _poison_float()
        self._prefix = [poison] * self.guard_elements
        self._suffix = [poison] * self.guard_elements
        self._input_snapshot = list(self.payload)

    def as_output(self) -> "GuardedBuffer":
        """Mark this buffer as an output: fill it with the sentinel."""
        self.payload = [self.sentinel] * len(self.payload)
        self._is_output = True
        return self

    def storage(self) -> List[float]:
        """The contiguous storage a device would see: guard + payload + guard."""
        return self._prefix + self.payload + self._suffix

    def verify(self) -> Dict[str, Any]:
        poison = _poison_float()
        prefix_ok = all(value == poison for value in self._prefix)
        suffix_ok = all(value == poison for value in self._suffix)
        first_bad_guard = -1
        for index, value in enumerate(self._prefix + self._suffix):
            if value != poison:
                first_bad_guard = index
                break
        unwritten = (
            [index for index, value in enumerate(self.payload) if _is_sentinel(value, self.sentinel)]
            if self._is_output
            else []
        )
        inputs_mutated = (
            []
            if self._is_output
            else [
                index
                for index, (before, after) in enumerate(zip(self._input_snapshot, self.payload))
                if before != after and not (math.isnan(before) and math.isnan(after))
            ]
        )
        return {
            "guard_intact": prefix_ok and suffix_ok,
            "first_bad_guard_index": first_bad_guard,
            "guard_elements": self.guard_elements,
            "unwritten_output_indices": unwritten[:16],
            "unwritten_output_count": len(unwritten),
            "input_mutated_indices": inputs_mutated[:16],
            "input_mutated_count": len(inputs_mutated),
            "ok": prefix_ok and suffix_ok and not unwritten and not inputs_mutated,
        }


def _poison_float() -> float:
    """The poison pattern as a float, byte-replicated so it survives any dtype."""
    byte = bytes([POISON_PATTERN]) * 4
    return float(int.from_bytes(byte, "little"))


def _is_sentinel(value: float, sentinel: float) -> bool:
    if math.isnan(sentinel):
        return math.isnan(value)
    return value == sentinel


# ── test vector artifact ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TestVectorArtifact:
    """One case's frozen inputs, golden and intermediates.

    ``intermediates`` carries ``mean_square``/``rstd`` because a final-output
    comparison cannot localise an epsilon or accumulation error (E09-03 step 4).
    """

    case: CaseSpec
    inputs: Mapping[str, Any]
    golden: Any
    intermediates: Mapping[str, Any]
    generator_version: str = GENERATOR_VERSION
    guard_elements: int = 16

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case.as_dict(),
            "generator_version": self.generator_version,
            "guard_elements": self.guard_elements,
            "inputs": {key: _summarize(value) for key, value in self.inputs.items()},
            "golden_summary": _summarize(self.golden),
            "intermediates": {key: _summarize(value) for key, value in self.intermediates.items()},
            "sha256": self.sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "case": self.case.as_dict(),
                    "generator_version": self.generator_version,
                    "inputs": self.inputs,
                    "golden": self.golden,
                    "intermediates": self.intermediates,
                }
            )
        )

    def guarded_inputs(self) -> Dict[str, GuardedBuffer]:
        """Wrap every input in guard regions (E09-02 step 5)."""
        buffers: Dict[str, GuardedBuffer] = {}
        for key, value in self.inputs.items():
            flat = _flatten(value)
            buffers[key] = GuardedBuffer(payload=flat, guard_elements=self.guard_elements)
        return buffers

    def guarded_output(self, count: Optional[int] = None) -> GuardedBuffer:
        length = count if count is not None else len(_flatten(self.golden))
        return GuardedBuffer(
            payload=[OUTPUT_SENTINEL] * length, guard_elements=self.guard_elements
        ).as_output()


def _flatten(values: Any) -> List[float]:
    flat: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
        else:
            flat.append(float(item))
    return flat


def _summarize(values: Any) -> Dict[str, Any]:
    flat = _flatten(values)
    finite = [value for value in flat if math.isfinite(value)]
    return {
        "count": len(flat),
        "nan_count": sum(1 for value in flat if math.isnan(value)),
        "inf_count": sum(1 for value in flat if math.isinf(value)),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "sha256": sha256_hex(canonical_json(flat)),
    }


def build_vector(case: CaseSpec) -> TestVectorArtifact:
    """Generate one case's inputs and golden from the frozen spec."""
    total = case.rows * case.hidden
    if case.operator == OPERATOR_ADD:
        x1 = generate_data(case.data_category, total, seed=case.seed, dtype=case.dtype)
        x2 = generate_data(case.data_category, total, seed=case.seed + 1, dtype=case.dtype)
        inputs = {"x1": x1, "x2": x2}
        golden = oracle_add(x1, x2)
        intermediates: Dict[str, Any] = {}
    elif case.operator == OPERATOR_ROW_REDUCE_SUM:
        flat = generate_data(case.data_category, total, seed=case.seed, dtype=case.dtype)
        rows = [flat[start : start + case.hidden] for start in range(0, total, case.hidden)]
        inputs = {"x": rows}
        golden = oracle_row_reduce_sum(rows)
        intermediates = {"kahan": oracle_row_reduce_sum(rows, kahan=True)}
    else:
        flat = generate_data(case.data_category, total, seed=case.seed, dtype=case.dtype)
        rows = [flat[start : start + case.hidden] for start in range(0, total, case.hidden)]
        gamma = generate_gamma(case.gamma_category, case.hidden, seed=case.seed + 2, dtype=case.dtype)
        outputs, mean_squares, rstds = oracle_rmsnorm(
            rows, gamma, case.eps, return_intermediates=True
        )
        inputs = {"x": rows, "gamma": gamma}
        golden = outputs
        intermediates = {"mean_square": mean_squares, "rstd": rstds}
    return TestVectorArtifact(case=case, inputs=inputs, golden=golden, intermediates=intermediates)


def vector_index(artifacts: Iterable[TestVectorArtifact]) -> Dict[str, Any]:
    """The ``test_vectors/index.json`` artifact."""
    rows = [artifact.as_dict() for artifact in artifacts]
    return {
        "generator_version": GENERATOR_VERSION,
        "count": len(rows),
        "index_sha256": sha256_hex(canonical_json(rows)),
        "vectors": rows,
    }


def content_hash_of(payload: Any) -> str:
    """Convenience wrapper so callers cannot pick a different hash recipe."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def non_finite_cases_require_spec(cases: Iterable[CaseSpec], spec_non_finite_policy: str) -> Dict[str, Any]:
    """E09-02 §5.2: NaN/Inf cases may only exist once the spec defines them."""
    offenders = [
        case.case_id
        for case in cases
        if case.data_category in NON_FINITE_CATEGORIES
        and spec_non_finite_policy in ("", "undefined", "unspecified")
    ]
    return {
        "ok": not offenders,
        "policy": spec_non_finite_policy,
        "offending_cases": offenders,
        "note": (
            ""
            if not offenders
            else "a NaN/Inf case without a defined behaviour cannot be judged; define it in the "
            "OperatorSpec or drop the case"
        ),
    }


__all__ = [
    "CaseMatrix",
    "CaseSpec",
    "DATA_CATEGORIES",
    "GAMMA_CATEGORIES",
    "GENERATOR_VERSION",
    "GuardedBuffer",
    "NON_FINITE_CATEGORIES",
    "OUTPUT_SENTINEL",
    "POISON_PATTERN",
    "SHAPE_CATEGORIES",
    "SPLITS",
    "SPLIT_ADVERSARIAL",
    "SPLIT_CONFIRMATION",
    "SPLIT_DEVELOPMENT",
    "SPLIT_TUNING",
    "TestVectorArtifact",
    "build_vector",
    "content_hash_of",
    "generate_data",
    "generate_gamma",
    "non_finite_cases_require_spec",
    "vector_index",
]
