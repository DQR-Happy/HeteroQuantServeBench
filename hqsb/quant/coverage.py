"""Quantization coverage enumeration (E05-02 §10 step 3, E05-10 §12).

"W4 model" is not an answer to *which tensors* were quantized. This module
walks a real module tree, applies an explicit include/exclude policy, and
produces one row per candidate weight with its scheme, group axis/size,
packed variant, quantized flag and — for every exclusion — a reason.

Three coverages are reported, because parameter coverage alone can hide the
fact that the hot modules fall back (E05-10 §12):

``parameter_coverage``
    quantized parameters / candidate parameters;
``low_bit_call_coverage``
    calls that actually executed a low-bit kernel / total calls;
``low_bit_time_coverage``
    time spent in low-bit kernels / total kernel time.

The last two need runtime data and are computed by
:func:`coverage_from_execution` from observed dispatch records; they are
``None`` (not 0) until such data exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.quant.spec import QuantScheme

#: Weight-name suffixes that are never quantized by default (their semantics
#: are not a plain linear projection). The policy is data, not a hidden rule.
DEFAULT_EXCLUDE_SUFFIXES = (
    "lm_head",
    "embed_tokens",
    "norm",
    "layernorm",
    "ln_",
)

DEFAULT_EXCLUDE_TYPES = ("LayerNorm", "RMSNorm", "Embedding")


@dataclass
class CoveragePolicy:
    """Explicit include/exclude policy for weight quantization (E05-02 §4.2).

    ``include``/``exclude`` are glob-ish suffix patterns matched against the
    full module path; ``exclude_types`` matches the module class name. Anything
    matched by neither ``include`` nor ``exclude`` is *quantized by default*
    when it is a linear-like weight, which is recorded in the row as
    ``selected_by = 'default_linear'`` so the decision is auditable.
    """

    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = DEFAULT_EXCLUDE_SUFFIXES
    exclude_types: Tuple[str, ...] = DEFAULT_EXCLUDE_TYPES
    include_bias: bool = False
    include_embeddings: bool = False
    #: Quantize only tensors whose name ends with one of these (empty = all).
    linear_suffixes: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def to_json(self) -> str:
        return json.dumps(
            {
                "include": list(self.include),
                "exclude": list(self.exclude),
                "exclude_types": list(self.exclude_types),
                "include_bias": self.include_bias,
                "include_embeddings": self.include_embeddings,
                "linear_suffixes": list(self.linear_suffixes),
            },
            sort_keys=True,
            indent=2,
        )


@dataclass
class WeightRow:
    """One candidate weight with its quantization decision."""

    name: str
    module_type: str
    shape: Tuple[int, ...]
    dtype: str
    num_parameters: int
    quantized: bool
    selected_by: str
    exclusion_reason: str = ""
    scheme_hash: str = ""
    group_axis: int = -1
    group_size: Optional[int] = None
    packed_variant: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module_type": self.module_type,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "num_parameters": self.num_parameters,
            "quantized": self.quantized,
            "selected_by": self.selected_by,
            "exclusion_reason": self.exclusion_reason,
            "scheme_hash": self.scheme_hash,
            "group_axis": self.group_axis,
            "group_size": self.group_size,
            "packed_variant": self.packed_variant,
        }


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(name.endswith(pattern) or pattern in name for pattern in patterns)


def classify_weight(
    name: str,
    module_type: str,
    policy: CoveragePolicy,
    *,
    is_bias: bool = False,
) -> Tuple[bool, str, str]:
    """Return ``(quantized, selected_by, exclusion_reason)`` for one weight.

    Order of decisions (first match wins, all of them recorded):

    1. explicit ``include`` patterns → quantize (``policy.include``);
    2. excluded type (norm/embedding) → skip (``excluded_type``);
    3. excluded suffix pattern → skip (``excluded_suffix``);
    4. bias without ``include_bias`` → skip (``bias_excluded``);
    5. embedding without ``include_embeddings`` → skip (``embedding_excluded``);
    6. name not ending in a configured linear suffix → skip
       (``not_linear_projection``);
    7. otherwise → quantize (``default_linear``).
    """
    # Linear-suffix matching works against the *module* path (a weight named
    # ``...down_proj.weight`` belongs to the ``down_proj`` linear projection).
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    if _matches(name, policy.include):
        return True, "explicit_include", ""
    if any(module_type.endswith(pattern) for pattern in policy.exclude_types):
        return False, "", f"excluded_type:{module_type}"
    if _matches(name, policy.exclude):
        return False, "", "excluded_suffix"
    if is_bias and not policy.include_bias:
        return False, "", "bias_excluded"
    if "embed" in name.lower() and not policy.include_embeddings:
        return False, "", "embedding_excluded"
    if policy.linear_suffixes and not (
        base.endswith(policy.linear_suffixes) or name.endswith(policy.linear_suffixes)
    ):
        return False, "", "not_linear_projection"
    return True, "default_linear", ""


def enumerate_weight_rows(
    model,
    policy: CoveragePolicy,
    *,
    scheme: Optional[QuantScheme] = None,
    packed_variant: str = "",
) -> List[WeightRow]:
    """Enumerate weight rows for a torch model (lazy torch import).

    Only ``torch.nn.Module`` traversal is used; no forward pass, no device
    movement, no quantization. The function is safe to call on a freshly
    loaded model to answer "what would be quantized".
    """
    import torch  # noqa: F401  (optional dependency, imported lazily)

    rows: List[WeightRow] = []
    for module_name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "shape"):
            continue
        module_type = type(module).__name__
        full_name = f"{module_name}.weight" if module_name else "weight"
        quantized, selected_by, reason = classify_weight(
            full_name, module_type, policy
        )
        rows.append(
            WeightRow(
                name=full_name,
                module_type=module_type,
                shape=tuple(int(dim) for dim in weight.shape),
                dtype=str(weight.dtype),
                num_parameters=int(weight.numel()),
                quantized=quantized,
                selected_by=selected_by,
                exclusion_reason=reason,
                scheme_hash=scheme.scheme_hash() if quantized and scheme else "",
                group_axis=(scheme.axis if quantized and scheme else -1),
                group_size=(scheme.group_size if quantized and scheme else None),
                packed_variant=packed_variant if quantized else "",
            )
        )
        if getattr(module, "bias", None) is not None:
            bias_name = f"{module_name}.bias" if module_name else "bias"
            bias_quantized, bias_selected_by, bias_reason = classify_weight(
                bias_name, module_type, policy, is_bias=True
            )
            rows.append(
                WeightRow(
                    name=bias_name,
                    module_type=module_type,
                    shape=tuple(int(dim) for dim in module.bias.shape),
                    dtype=str(module.bias.dtype),
                    num_parameters=int(module.bias.numel()),
                    quantized=bias_quantized,
                    selected_by=bias_selected_by,
                    exclusion_reason=bias_reason,
                )
            )
    return rows


def coverage_summary(
    rows: Sequence[WeightRow], *, runtime_facing_suffixes: Sequence[str] = ()
) -> Dict[str, Any]:
    """Aggregate rows into parameter coverage and exclusion breakdown.

    ``runtime_facing_suffixes`` restricts the denominator to weights a kernel
    can actually consume (linear projections); a high parameter coverage over
    norms/embeddings would be misleading otherwise.
    """
    total_params = sum(row.num_parameters for row in rows)
    quantized_params = sum(row.num_parameters for row in rows if row.quantized)
    reasons: Dict[str, int] = {}
    for row in rows:
        if not row.quantized:
            key = row.exclusion_reason or "unspecified"
            reasons[key] = reasons.get(key, 0) + 1
    relevant = [
        row
        for row in rows
        if not runtime_facing_suffixes
        or row.name.endswith(tuple(runtime_facing_suffixes))
    ]
    relevant_params = sum(row.num_parameters for row in relevant)
    return {
        "candidate_tensors": len(rows),
        "quantized_tensors": sum(1 for row in rows if row.quantized),
        "candidate_parameters": total_params,
        "quantized_parameters": quantized_params,
        "parameter_coverage": (quantized_params / total_params) if total_params else None,
        "runtime_facing_parameters": relevant_params,
        "runtime_facing_coverage": (
            quantized_params / relevant_params if relevant_params else None
        ),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "low_bit_call_coverage": None,
        "low_bit_time_coverage": None,
        "note": (
            "call/time coverage are None until observed dispatch records exist; "
            "a parameter-only number must not be reported as low-bit coverage"
        ),
    }


def coverage_from_execution(
    summary: Mapping[str, Any],
    dispatch_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Augment a coverage summary with observed low-bit call/time coverage.

    ``dispatch_records`` are the runtime records produced by
    :mod:`hqsb.quant.execution` (one per module invocation). Records missing
    the kernel-identity fields are *not* counted as low-bit — an unknown path
    is not a low-bit path.
    """
    total_calls = 0
    low_bit_calls = 0
    total_time = 0.0
    low_bit_time = 0.0
    unknown_records = 0
    for record in dispatch_records:
        calls = int(record.get("call_count", 1) or 1)
        elapsed = float(record.get("duration_ms", 0.0) or 0.0)
        total_calls += calls
        total_time += elapsed
        label = str(record.get("execution_label", ""))
        if not label:
            unknown_records += 1
            continue
        if record.get("low_bit_executed"):
            low_bit_calls += calls
            low_bit_time += elapsed
    updated = dict(summary)
    updated["low_bit_call_coverage"] = low_bit_calls / total_calls if total_calls else None
    updated["low_bit_time_coverage"] = low_bit_time / total_time if total_time else None
    updated["observed_calls"] = total_calls
    updated["records_without_label"] = unknown_records
    return updated


def coverage_rows_json(rows: Sequence[WeightRow]) -> str:
    return json.dumps([row.as_dict() for row in rows], sort_keys=True, indent=2)


__all__ = [
    "CoveragePolicy",
    "DEFAULT_EXCLUDE_SUFFIXES",
    "DEFAULT_EXCLUDE_TYPES",
    "WeightRow",
    "classify_weight",
    "coverage_from_execution",
    "coverage_rows_json",
    "coverage_summary",
    "enumerate_weight_rows",
]
