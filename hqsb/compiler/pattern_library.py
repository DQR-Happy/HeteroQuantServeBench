"""Frozen pattern contracts, predicate catalogue and CPU reference oracles.

Protocol anchor: ``details/S11/E11-02_semantic_pattern_rewrite_idempotence.md``
§4 (pattern contract), §6 (positive/negative/near-miss matrix) and §8 steps
2–4, 16–18 (contract, semantic pattern id, positive and mutated corpora).

Everything here is *declarative*: contracts, the predicate order, the mutation
axes that produce near-misses, and pure-Python numeric references used as the
correctness oracle interface.  The matcher/rewrite engine lives in
``hqsb.compiler.rewrite``.

The v0 pattern is ``residual + RMSNorm`` because HQSB already has CUDA/Triton
assets for it (details README §1).  ``rmsnorm_quantize`` is registered as the
second declared pattern contract (the stage gate requires two real patterns to
be recognised; the second may complete with rewrite+fallback only).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text

# ── pattern contracts (E11-02 §4) ──────────────────────────────────────────

PATTERN_RESIDUAL_RMSNORM = "hqsb.pattern.residual_add_rmsnorm"
PATTERN_RMSNORM_QUANTIZE = "hqsb.pattern.rmsnorm_quantize"
PATTERN_VERSION = "1.0.0"


@dataclass
class PatternContract:
    """Inputs/outputs/math/effects/supported/unsupported of one pattern."""

    pattern_id: str
    version: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    math: Tuple[str, ...]
    effects: str  # functional | mutable
    supported: Tuple[str, ...]
    unsupported: Tuple[str, ...]
    fused_op: str
    tolerance_policy_id: str = "common_s06"
    notes: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.pattern_id or not self.version:
            problems.append("pattern contract needs id and version")
        if self.effects not in ("functional", "mutable"):
            problems.append(f"unknown effect contract {self.effects!r}")
        if not self.math:
            problems.append("pattern contract must state the math")
        if not self.unsupported:
            problems.append(
                "pattern contract must enumerate unsupported cases (an open-ended contract "
                "cannot be fail-closed)"
            )
        if not self.fused_op.startswith("hqsb::"):
            problems.append("fused op must live in the hqsb:: namespace")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "version": self.version,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "math": list(self.math),
            "effects": self.effects,
            "supported": list(self.supported),
            "unsupported": list(self.unsupported),
            "fused_op": self.fused_op,
            "tolerance_policy_id": self.tolerance_policy_id,
            "notes": self.notes,
        }


def residual_add_rmsnorm_contract() -> PatternContract:
    return PatternContract(
        pattern_id=PATTERN_RESIDUAL_RMSNORM,
        version=PATTERN_VERSION,
        inputs=("x", "residual", "weight", "eps"),
        outputs=("normalized", "residual_new"),
        math=(
            "residual_new = cast_semantics(x + residual)",
            "variance = reduce_mean(accum_cast(residual_new)^2, last_dim)",
            "normalized = residual_new * rsqrt(variance + eps) * weight",
        ),
        effects="functional",
        supported=(
            "add is out-of-place",
            "mean over the last dimension with keepdim",
            "eps is a literal/buffer in the allowed set",
            "weight broadcasts along the normalised axis with input dtype",
            "residual_new has no users outside the pattern, or the replacement "
            "returns it bit-identically",
        ),
        unsupported=(
            "unknown alias/effect",
            "in-place add / write-back to residual",
            "different reduction axis/keepdim/divisor",
            "cast reordering that changes rounding",
            "weight shape or dtype mismatch",
            "extra semantic users that cannot be preserved",
        ),
        fused_op="hqsb::fused_add_rms_norm",
        notes="v0 hero pattern; semantics frozen against S03/S04.5 kernels",
    )


def rmsnorm_quantize_contract() -> PatternContract:
    return PatternContract(
        pattern_id=PATTERN_RMSNORM_QUANTIZE,
        version=PATTERN_VERSION,
        inputs=("x", "weight", "eps", "scale", "zero_point"),
        outputs=("quantized", "dequantized"),
        math=(
            "y = x * rsqrt(mean(x^2, last_dim) + eps) * weight",
            "q = quantize(y, scale, zero_point); dequant = dequantize(q, scale, zero_point)",
        ),
        effects="functional",
        supported=(
            "per-tensor or per-group scale with a frozen QuantArtifact",
            "rounding mode declared by the artifact",
        ),
        unsupported=(
            "dynamic/row-wise scale that changes with the input",
            "packing that changes the semantic dtype without an artifact",
            "quality gate not covering the current shape",
        ),
        fused_op="hqsb::rms_norm",
        notes=(
            "second declared pattern; may complete with rewrite+fallback only "
            "(stage gate: two real patterns recognised, one full lower)"
        ),
    )


def frozen_contracts() -> Tuple[PatternContract, ...]:
    return (residual_add_rmsnorm_contract(), rmsnorm_quantize_contract())


def contract_by_id(pattern_id: str) -> PatternContract:
    for contract in frozen_contracts():
        if contract.pattern_id == pattern_id:
            return contract
    raise ConfigError(f"unknown pattern id {pattern_id!r}")


@dataclass
class PatternSignature:
    """Class-name-independent signature (E11-02 step 4)."""

    pattern_id: str
    version: str
    ordered_roles: Tuple[str, ...]
    constants_policy: str
    effect_contract: str
    shape_relations: Tuple[str, ...]
    schema_versions: Mapping[str, str]
    decomposition_variants: Tuple[str, ...]

    def digest(self) -> str:
        return sha256_text(canonical_json(self.as_dict()))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "version": self.version,
            "ordered_roles": list(self.ordered_roles),
            "constants_policy": self.constants_policy,
            "effect_contract": self.effect_contract,
            "shape_relations": list(self.shape_relations),
            "schema_versions": dict(sorted(self.schema_versions.items())),
            "decomposition_variants": list(self.decomposition_variants),
            "note": "must not depend on Qwen Python class names or temporary FX node names",
        }


ROLE_ORDER_RESIDUAL_RMSNORM: Tuple[str, ...] = (
    "weight_mul",
    "scale",
    "rsqrt",
    "eps_add",
    "mean",
    "pow",
    "add_residual",
)

#: Declarative role chain: ``(role, accepted_op_names, operand_value_roles,
#: consumer_link)``.  ``consumer_link`` names the role that consumes this
#: role's result and at which operand index (``None`` for the output root).
#: The matcher walks the chain from its *anchor* (``add_residual``, whose
#: operands must be graph inputs) forward to the root; a shared value role
#: (``residual_new`` consumed by both ``pow`` and ``scale``) is expressed by
#: the same link target appearing twice, which the matcher verifies.
ROLE_CHAIN_RESIDUAL_RMSNORM: Tuple[
    Tuple[str, Tuple[str, ...], Tuple[Tuple[str, int], ...], Optional[Tuple[str, int]]], ...
] = (
    ("add_residual", ("aten.add", "aten.add_"), (("x", 0), ("residual", 1)), ("pow", 0)),
    ("pow", ("aten.pow",), (("residual_new", 0), ("exponent", 1)), ("mean", 0)),
    ("mean", ("aten.mean",), (("sq", 0),), ("eps_add", 0)),
    ("eps_add", ("aten.add",), (("variance", 0), ("eps_const", 1)), ("rsqrt", 0)),
    ("rsqrt", ("aten.rsqrt",), (("var_eps", 0),), ("scale", 1)),
    ("scale", ("aten.mul",), (("residual_new", 0), ("inv_std", 1)), ("weight_mul", 0)),
    ("weight_mul", ("aten.mul",), (("scaled", 0), ("weight", 1)), None),
)

#: Value roles that must resolve to graph inputs.
INPUT_VALUE_ROLES: Tuple[str, ...] = ("x", "residual", "weight", "eps_const")

#: The anchor role of the chain (its operands must be graph inputs).
ANCHOR_ROLE = "add_residual"

#: Alternative decompositions accepted as equivalent expressions.
DECOMPOSITION_VARIANTS: Tuple[str, ...] = (
    "canonical",
    "explicit_cast_before_reduce",
    "rsqrt_as_pow_neg_half",
    "mean_via_sum_div",
)


def residual_rmsnorm_signature() -> PatternSignature:
    return PatternSignature(
        pattern_id=PATTERN_RESIDUAL_RMSNORM,
        version=PATTERN_VERSION,
        ordered_roles=ROLE_ORDER_RESIDUAL_RMSNORM,
        constants_policy="eps value must be in the frozen allowed set; exponent must be 2",
        effect_contract="functional; residual_new has no writes",
        shape_relations=(
            "mean over the last dimension with keepdim",
            "weight broadcasts to the normalised axis",
            "hidden size divisible by the kernel vector width or a tail variant is chosen",
        ),
        schema_versions={"hqsb::fused_add_rms_norm": "1.0.0", "aten.add": "1.0.0"},
        decomposition_variants=DECOMPOSITION_VARIANTS,
    )


# ── predicate catalogue (steps 6–13) ───────────────────────────────────────

PREDICATE_ORDER: Tuple[str, ...] = (
    "structural_match",
    "dtype_cast",
    "shape_reduction",
    "epsilon",
    "users_liveness",
    "alias_mutation_effect",
    "layout",
    "return_contract",
)

PREDICATE_CATALOG: Mapping[str, Mapping[str, str]] = {
    "structural_match": {
        "purpose": "candidate discovery only (roles/edges/users/constants)",
        "reject_code": "PATTERN_NEAR_MISS",
    },
    "dtype_cast": {
        "purpose": "operand/output/accumulator dtype and autocast state",
        "reject_code": "PATTERN_NEAR_MISS",
    },
    "shape_reduction": {
        "purpose": "normalised axis, weight broadcast, rank, hidden relation",
        "reject_code": "PATTERN_NEAR_MISS",
    },
    "epsilon": {
        "purpose": "literal/buffer/runtime eps value, dtype and position",
        "reject_code": "PATTERN_NEAR_MISS",
    },
    "users_liveness": {
        "purpose": "in/out users of intermediate values; preserved outputs",
        "reject_code": "PATTERN_NEAR_MISS",
    },
    "alias_mutation_effect": {
        "purpose": "PROVED_SAFE / PROVED_UNSAFE / UNKNOWN_UNSAFE",
        "reject_code": "ALIAS_UNSAFE",
    },
    "layout": {
        "purpose": "layout/stride class: semantic (alias) vs candidate-capability",
        "reject_code": "PATTERN_NEAR_MISS",
    },
    "return_contract": {
        "purpose": "which values must be returned and with which rounding",
        "reject_code": "PATTERN_NEAR_MISS",
    },
}

#: Predicate outcomes.
OUTCOME_PASS = "PASS"
OUTCOME_REJECT = "REJECT"
OUTCOME_UNKNOWN = "UNKNOWN"

# ── match classes (E11-02 §6) ──────────────────────────────────────────────

MATCH = "MATCH"
REJECT = "REJECT"
NO_CHANGE = "NO_CHANGE"
EPSILON_PROFILE = "EPSILON_PROFILE"

MATCH_CLASS_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("positive_canonical", MATCH, ""),
    ("positive_decomposed", MATCH, ""),
    ("epsilon_difference", REJECT, "PATTERN_NEAR_MISS"),
    ("reduction_difference", REJECT, "PATTERN_NEAR_MISS"),
    ("accumulation_difference", REJECT, "PATTERN_NEAR_MISS"),
    ("order_difference", REJECT, "PATTERN_NEAR_MISS"),
    ("extra_user", MATCH, "must preserve residual_new exactly, otherwise REJECT"),
    ("mutation", REJECT, "MUTATION_UNSAFE"),
    ("alias_view", REJECT, "ALIAS_UNSAFE"),
    ("weight_broadcast", REJECT, "PATTERN_NEAR_MISS"),
    ("stride_layout", "DEFER", "semantic pass keeps the site; lowering guard/fallback decides"),
    ("dtype_autocast", REJECT, "PATTERN_NEAR_MISS"),
    ("nan_inf_edge", "CORRECTNESS_TEST", "matcher cannot decide; differential test must cover"),
    ("already_fused", NO_CHANGE, ""),
)

NEAR_MISS_MUTATION_AXES: Tuple[Tuple[str, str, str], ...] = (
    ("epsilon_value", "epsilon", "PATTERN_NEAR_MISS"),
    ("epsilon_position", "epsilon", "PATTERN_NEAR_MISS"),
    ("reduction_axis", "shape_reduction", "PATTERN_NEAR_MISS"),
    ("reduction_kind", "shape_reduction", "PATTERN_NEAR_MISS"),
    ("cast_order", "dtype_cast", "PATTERN_NEAR_MISS"),
    ("accumulation_dtype", "dtype_cast", "PATTERN_NEAR_MISS"),
    ("operation_order", "structural_match", "PATTERN_NEAR_MISS"),
    ("extra_user", "users_liveness", "PATTERN_NEAR_MISS"),
    ("inplace_add", "alias_mutation_effect", "MUTATION_UNSAFE"),
    ("alias_view", "alias_mutation_effect", "ALIAS_UNSAFE"),
    ("weight_shape", "shape_reduction", "PATTERN_NEAR_MISS"),
    ("return_values", "return_contract", "PATTERN_NEAR_MISS"),
)


def mutation_axes_table() -> List[Dict[str, str]]:
    """One mutation changes exactly one semantic factor (step 17)."""
    return [
        {"axis": axis, "expected_predicate": predicate, "reject_code": code}
        for axis, predicate, code in NEAR_MISS_MUTATION_AXES
    ]


def expected_outcome(match_class: str) -> Dict[str, Any]:
    for name, outcome, note in MATCH_CLASS_TABLE:
        if name == match_class:
            return {"class": name, "expected": outcome, "note": note}
    raise ConfigError(f"unknown match class {match_class!r}")


# ── CPU reference oracles (correctness interface) ──────────────────────────


def fp16_round(value: float) -> float:
    """Round a Python float to IEEE-754 binary16 (eager FP16 storage)."""
    return struct.unpack("e", struct.pack("e", value))[0]


def bf16_round(value: float) -> float:
    """Round a Python float to bfloat16 (truncating mantissa, round-half-even)."""
    packed = struct.pack(">f", value)
    as_int = int.from_bytes(packed, "big")
    rounded = (as_int + 0x8000) & 0xFFFF0000
    return struct.unpack(">f", rounded.to_bytes(4, "big"))[0]


CAST_ROUNDERS = {"fp16": fp16_round, "bf16": bf16_round, "fp32": lambda value: value}


def cast_semantics(value: float, dtype: str) -> float:
    if dtype not in CAST_ROUNDERS:
        raise ConfigError(f"unknown cast dtype {dtype!r}")
    return CAST_ROUNDERS[dtype](value)


def _rms_norm(values: Sequence[float], weight: Sequence[float], eps: float) -> List[float]:
    if not values:
        raise ConfigError("empty row")
    squares = [value * value for value in values]
    variance = sum(squares) / len(squares)
    inv_std = 1.0 / math.sqrt(variance + eps)
    return [values[index] * inv_std * weight[index] for index in range(len(values))]


def composed_add_rmsnorm_reference(
    x: Sequence[float],
    residual: Sequence[float],
    weight: Sequence[float],
    *,
    eps: float = 1e-6,
    storage_dtype: str = "fp16",
    accumulate_dtype: str = "fp32",
) -> Dict[str, Any]:
    """Eager composition: ``r`` is *stored* (rounded) before the reduction."""
    if not (len(x) == len(residual) == len(weight)):
        raise ConfigError("x/residual/weight must have the same hidden size")
    residual_new = [cast_semantics(a + b, storage_dtype) for a, b in zip(x, residual)]
    reduced = [cast_semantics(value, accumulate_dtype) for value in residual_new]
    normalized = _rms_norm(reduced, weight, eps)
    normalized = [cast_semantics(value, storage_dtype) for value in normalized]
    return {"normalized": normalized, "residual_new": residual_new, "path": "composed"}


def fused_add_rmsnorm_reference(
    x: Sequence[float],
    residual: Sequence[float],
    weight: Sequence[float],
    *,
    eps: float = 1e-6,
    storage_dtype: str = "fp16",
    keep_residual_in: str = "fp32",
) -> Dict[str, Any]:
    """Fused lowering: the reduction consumes the *unrounded* residual.

    This is the accumulation difference the near-miss table requires to be
    rejected unless the numerical contract explicitly declares the domain
    where the two paths are equivalent.
    """
    if not (len(x) == len(residual) == len(weight)):
        raise ConfigError("x/residual/weight must have the same hidden size")
    residual_new = [cast_semantics(a + b, storage_dtype) for a, b in zip(x, residual)]
    reduction_input = [cast_semantics(a + b, keep_residual_in) for a, b in zip(x, residual)]
    normalized = _rms_norm(reduction_input, weight, eps)
    normalized = [cast_semantics(value, storage_dtype) for value in normalized]
    return {"normalized": normalized, "residual_new": residual_new, "path": "fused"}


def compare_reference_paths(
    x: Sequence[float],
    residual: Sequence[float],
    weight: Sequence[float],
    *,
    eps: float = 1e-6,
    storage_dtype: str = "fp16",
) -> Dict[str, Any]:
    """Differential oracle used to document (not to excuse) the near-miss."""
    composed = composed_add_rmsnorm_reference(
        x, residual, weight, eps=eps, storage_dtype=storage_dtype
    )
    fused = fused_add_rmsnorm_reference(x, residual, weight, eps=eps, storage_dtype=storage_dtype)
    diffs = [
        abs(left - right) for left, right in zip(composed["normalized"], fused["normalized"])
    ]
    residual_diffs = [
        abs(left - right)
        for left, right in zip(composed["residual_new"], fused["residual_new"])
    ]
    # The structural difference between the two paths is the reduction input:
    # the composed path feeds the *stored* (rounded) residual, the fused path
    # feeds the unrounded sum.  This difference is deterministic, unlike the
    # final rounding, which may coincidentally agree.
    raw_sum = [a + b for a, b in zip(x, residual)]
    rounded = [cast_semantics(value, storage_dtype) for value in raw_sum]
    reduction_diffs = [abs(p - q) for p, q in zip(raw_sum, rounded)]
    return {
        "max_abs_error": max(diffs) if diffs else 0.0,
        "mean_abs_error": sum(diffs) / len(diffs) if diffs else 0.0,
        "max_residual_error": max(residual_diffs) if residual_diffs else 0.0,
        "paths_identical": all(diff == 0.0 for diff in diffs),
        "reduction_inputs_differ": any(diff > 0.0 for diff in reduction_diffs),
        "max_reduction_input_error": max(reduction_diffs) if reduction_diffs else 0.0,
        "note": (
            "a numerically small difference on one sample does not make the rewrite legal; "
            "the semantics predicate decides — but the reduction-input difference is what the "
            "accumulation predicate is protecting against"
        ),
    }


# ── fixture graphs (canonical / near-miss / negatives) ─────────────────────


def _base_inputs(*, weight_operand: str = "ph_w", extra: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = [
        {"value_id": "ph_x", "dtype": "fp16", "dims": ("B", "H")},
        {"value_id": "ph_r", "dtype": "fp16", "dims": ("B", "H")},
        {"value_id": weight_operand, "dtype": "fp16", "dims": ("H",)},
        {"value_id": "ph_eps", "dtype": "fp32", "dims": ()},
    ]
    inputs.extend(dict(item) for item in extra)
    return inputs


def residual_add_rmsnorm_graph(
    *,
    graph_id: str = "fixture_canonical",
    variant: str = "canonical",
    extra_users: bool = False,
) -> Dict[str, Any]:
    """Declarative op list for one corpus member.

    ``variant`` selects the mutation axis: ``canonical``, ``decomposed_cast``,
    ``eps_value``, ``reduction_axis``, ``reduction_kind``, ``cast_order``,
    ``order_different``, ``inplace_add``, ``alias_view``, ``weight_shape``,
    ``return_only_y``, ``already_fused`` or ``hard_negative``.
    """
    ops: List[Dict[str, Any]] = []
    add_spec: Dict[str, Any] = {
        "op_id": "o_add",
        "semantic_op": "aten.add",
        "operands": ["ph_x", "ph_r"],
        "results": ["v_r"],
        "attrs": {"alpha": 1.0},
        "source": {"module_path": "fixture.block", "source_nodes": ["add"], "source_line": 10},
        "mutation": "none",
        "effect_evidence": "schema",
        "dtype": "fp16",
        "dims": ("B", "H"),
    }
    if variant == "inplace_add":
        add_spec["semantic_op"] = "aten.add_"
        add_spec["mutation"] = "inplace"
        add_spec["writes"] = ("ph_r",)
    if variant == "alias_view":
        add_spec["effect_evidence"] = "unknown"
        add_spec["effect_reason"] = "residual may be a view of x"
        add_spec["mutation"] = "unknown"
    ops.append(add_spec)

    if variant == "order_different":
        # norm applied to x only, then residual added afterwards
        ops = [
            {
                "op_id": "o_pow",
                "semantic_op": "aten.pow",
                "operands": ["ph_x", "ph_two"],
                "results": ["v_pow"],
                "attrs": {"exponent": 2},
                "source": {"module_path": "fixture.block", "source_nodes": ["pow"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o_mean",
                "semantic_op": "aten.mean",
                "operands": ["v_pow"],
                "results": ["v_mean"],
                "attrs": {"dim": -1, "keepdim": True},
                "source": {"module_path": "fixture.block", "source_nodes": ["mean"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o_eps",
                "semantic_op": "aten.add",
                "operands": ["v_mean", "ph_eps"],
                "results": ["v_epsadd"],
                "source": {"module_path": "fixture.block", "source_nodes": ["add"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o_rsqrt",
                "semantic_op": "aten.rsqrt",
                "operands": ["v_epsadd"],
                "results": ["v_rsqrt"],
                "source": {"module_path": "fixture.block", "source_nodes": ["rsqrt"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o_scale",
                "semantic_op": "aten.mul",
                "operands": ["ph_x", "v_rsqrt"],
                "results": ["v_scaled"],
                "source": {"module_path": "fixture.block", "source_nodes": ["mul"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o_weight",
                "semantic_op": "aten.mul",
                "operands": ["v_scaled", "ph_w"],
                "results": ["v_y"],
                "source": {"module_path": "fixture.block", "source_nodes": ["mul"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o_add_out",
                "semantic_op": "aten.add",
                "operands": ["v_y", "ph_r"],
                "results": ["v_out"],
                "source": {"module_path": "fixture.block", "source_nodes": ["add"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
        ]
        return {
            "graph_id": graph_id,
            "inputs": _base_inputs(),
            "ops": ops,
            "outputs": ["v_out"],
            "constraints": [
                {"constraint_id": "c_hidden", "expression": "H % 8 == 0", "category": "variant", "origin": "kernel"},
                {"constraint_id": "c_batch", "expression": "1 <= B <= 8", "category": "semantic", "origin": "user"},
            ],
        }

    # ``decomposed_cast`` is the *declared* decomposition (cast to the
    # accumulation dtype); ``cast_order`` re-casts to storage precision and is
    # therefore an accumulation near-miss.
    cast_variant = variant in ("cast_order", "decomposed_cast")
    pow_operand = "v_r_cast" if cast_variant else "v_r"
    pow_ops: List[Dict[str, Any]] = []
    if cast_variant:
        cast_dtype = "fp16" if variant == "cast_order" else "fp32"
        pow_ops.append(
            {
                "op_id": "o_cast",
                "semantic_op": "aten.to",
                "operands": ["v_r"],
                "results": ["v_r_cast"],
                "attrs": {"dtype": cast_dtype, "rounding": "rn"},
                "source": {"module_path": "fixture.block", "source_nodes": ["to"]},
                "mutation": "none",
                "effect_evidence": "schema",
            }
        )
    pow_ops.append(
        {
            "op_id": "o_pow",
            "semantic_op": "aten.pow",
            "operands": [pow_operand, "ph_two"],
            "results": ["v_pow"],
            "attrs": {"exponent": 2 if variant != "accumulation_dtype" else 2.0, "accumulate_dtype": "fp32"},
            "source": {"module_path": "fixture.block", "source_nodes": ["pow"]},
            "mutation": "none",
            "effect_evidence": "schema",
        }
    )
    reduction_op = "aten.mean" if variant != "reduction_kind" else "aten.sum"
    mean_attrs: Dict[str, Any] = {"dim": -1, "keepdim": True}
    if variant == "reduction_axis":
        mean_attrs = {"dim": -2, "keepdim": True}
    if variant == "reduction_kind":
        mean_attrs = {"dim": -1, "keepdim": True, "divisor": "N-1"}
    pow_ops.append(
        {
            "op_id": "o_mean",
            "semantic_op": reduction_op,
            "operands": ["v_pow"],
            "results": ["v_mean"],
            "attrs": mean_attrs,
            "source": {"module_path": "fixture.block", "source_nodes": ["mean"]},
            "mutation": "none",
            "effect_evidence": "schema",
        }
    )
    eps_value = 1e-5 if variant != "eps_value" else 1e-3
    pow_ops.append(
        {
            "op_id": "o_rsqrt",
            "semantic_op": "aten.rsqrt",
            "operands": ["v_epsadd"],
            "results": ["v_rsqrt"],
            "source": {"module_path": "fixture.block", "source_nodes": ["rsqrt"]},
            "mutation": "none",
            "effect_evidence": "schema",
        }
    )
    ops.extend(pow_ops)
    eps_add: Dict[str, Any] = {
        "op_id": "o_eps",
        "semantic_op": "aten.add",
        "operands": ["v_mean", "ph_eps"],
        "results": ["v_epsadd"],
        "attrs": {"value": eps_value},
        "source": {"module_path": "fixture.block", "source_nodes": ["add"]},
        "mutation": "none",
        "effect_evidence": "schema",
    }
    scale_operand = "v_r_cast" if cast_variant else "v_r"
    tail_ops: List[Dict[str, Any]] = [
        eps_add,
        {
            "op_id": "o_scale",
            "semantic_op": "aten.mul",
            "operands": [scale_operand, "v_rsqrt"],
            "results": ["v_scaled"],
            "source": {"module_path": "fixture.block", "source_nodes": ["mul"]},
            "mutation": "none",
            "effect_evidence": "schema",
        },
        {
            "op_id": "o_weight",
            "semantic_op": "aten.mul",
            "operands": ["v_scaled", "ph_w_shape" if variant == "weight_shape" else "ph_w"],
            "results": ["v_y"],
            "source": {"module_path": "fixture.block", "source_nodes": ["mul"]},
            "mutation": "none",
            "effect_evidence": "schema",
        },
    ]
    ops.extend(tail_ops)
    outputs = ["v_y"] if variant == "return_only_y" else ["v_y", "v_r"]
    if variant == "already_fused":
        ops = [
            {
                "op_id": "o_fused",
                "semantic_op": "hqsb::fused_add_rms_norm",
                "operands": ["ph_x", "ph_r", "ph_w", "ph_eps"],
                "results": ["v_y", "v_r_out"],
                "attrs": {"eps": 1e-5},
                "source": {"module_path": "fixture.block", "source_nodes": ["fused"]},
                "mutation": "none",
                "effect_evidence": "schema",
                "pattern_id": PATTERN_RESIDUAL_RMSNORM,
            }
        ]
        outputs = ["v_y", "v_r_out"]
    if variant == "hard_negative":
        ops = [
            {
                "op_id": "o_mm",
                "semantic_op": "aten.mm",
                "operands": ["ph_x", "ph_r"],
                "results": ["v_mm"],
                "source": {"module_path": "fixture.other", "source_nodes": ["mm"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o_relu",
                "semantic_op": "aten.relu",
                "operands": ["v_mm"],
                "results": ["v_out"],
                "source": {"module_path": "fixture.other", "source_nodes": ["relu"]},
                "mutation": "none",
                "effect_evidence": "schema",
            },
        ]
        outputs = ["v_out"]
    inputs = _base_inputs(
        weight_operand="ph_w_shape" if variant == "weight_shape" else "ph_w",
        extra=(
            {"value_id": "ph_two", "dtype": "fp32", "dims": ()},
            {"value_id": "ph_rsqrt_placeholder", "dtype": "fp16", "dims": ("B", "1")},
        ),
    )
    constraints = [
        {"constraint_id": "c_batch", "expression": "1 <= B <= 8", "category": "semantic", "origin": "user"},
        {"constraint_id": "c_hidden", "expression": "H % 8 == 0", "category": "variant", "origin": "kernel"},
    ]
    if variant == "weight_shape":
        # weight is a 2-D matrix that does not broadcast along the hidden axis
        inputs = [
            item if item["value_id"] != "ph_w_shape" else {**item, "dims": ("H1", "H2")}
            for item in inputs
        ]
    return {
        "graph_id": graph_id,
        "inputs": inputs,
        "ops": ops,
        "outputs": outputs,
        "constraints": constraints,
        "notes": f"variant={variant}",
        "extra_users": extra_users,
    }


CORPUS_CLASSES: Tuple[str, ...] = (
    "positive_canonical",
    "positive_decomposed",
    "epsilon_difference",
    "reduction_difference",
    "accumulation_difference",
    "order_difference",
    "extra_user",
    "mutation",
    "alias_view",
    "weight_broadcast",
    "dtype_autocast",
    "already_fused",
    "hard_negative",
)


def corpus_plan() -> List[Dict[str, Any]]:
    """Pre-registered corpus with the expected decision per class (step 19)."""
    return [
        {"class": "positive_canonical", "variant": "canonical", "expected": MATCH, "reject_code": ""},
        {"class": "positive_decomposed", "variant": "decomposed_cast", "expected": MATCH, "reject_code": ""},
        {"class": "epsilon_difference", "variant": "eps_value", "expected": REJECT, "reject_code": "PATTERN_NEAR_MISS"},
        {"class": "reduction_difference", "variant": "reduction_axis", "expected": REJECT, "reject_code": "PATTERN_NEAR_MISS"},
        {"class": "accumulation_difference", "variant": "cast_order", "expected": REJECT, "reject_code": "PATTERN_NEAR_MISS"},
        {"class": "order_difference", "variant": "order_different", "expected": REJECT, "reject_code": "PATTERN_NEAR_MISS"},
        {"class": "extra_user", "variant": "canonical", "expected": MATCH, "reject_code": ""},
        {"class": "mutation", "variant": "inplace_add", "expected": REJECT, "reject_code": "MUTATION_UNSAFE"},
        {"class": "alias_view", "variant": "alias_view", "expected": REJECT, "reject_code": "ALIAS_UNSAFE"},
        {"class": "weight_broadcast", "variant": "weight_shape", "expected": REJECT, "reject_code": "PATTERN_NEAR_MISS"},
        {"class": "already_fused", "variant": "already_fused", "expected": NO_CHANGE, "reject_code": ""},
        {"class": "hard_negative", "variant": "hard_negative", "expected": REJECT, "reject_code": "PATTERN_NEAR_MISS"},
    ]


def contract_digest() -> str:
    """Digest over all frozen contracts (enters the pass identity)."""
    return sha256_text(canonical_json([item.as_dict() for item in frozen_contracts()]))
