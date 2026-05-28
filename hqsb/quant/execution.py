"""Execution reality: what actually ran, and what may be claimed.

E05-02 §9.3 and E05-06 §2/§11 make one rule non-negotiable: a performance
number is only a *low-bit* number when the observed execution path really
decoded packed low-bit weights inside the kernel. This module defines the
closed set of execution labels, the record every module invocation must
carry, and the classifier that decides whether a low-bit claim is allowed.

Labels (E05-06 §2):

``fp16_reference``
    plain FP16/BF16 GEMM on FP16 weights;
``fake_quant``
    quantize→dequantize in memory, then FP16 GEMM (algorithm quality only);
``storage_only``
    packed artifact on disk/RAM, weights materialized to FP16 before the
    GEMM (storage/load claims only);
``fused_dequant_weight_only``
    kernel reads packed W4/W8 and dequantizes inside the kernel, activation
    stays 16-bit (a real low-bit *weight bandwidth* path);
``integer_tensor_op``
    both operands are integer and the dot product is integral (requires
    instruction-level evidence);
``explicit_fallback``
    a label for any path entered after a compatibility refusal, with the
    fallback reason attached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hqsb.core.errors import ConfigError

FP16_REFERENCE = "fp16_reference"
FAKE_QUANT = "fake_quant"
STORAGE_ONLY = "storage_only"
FUSED_DEQUANT_WEIGHT_ONLY = "fused_dequant_weight_only"
INTEGER_TENSOR_OP = "integer_tensor_op"
EXPLICIT_FALLBACK = "explicit_fallback"

EXECUTION_LABELS = (
    FP16_REFERENCE,
    FAKE_QUANT,
    STORAGE_ONLY,
    FUSED_DEQUANT_WEIGHT_ONLY,
    INTEGER_TENSOR_OP,
    EXPLICIT_FALLBACK,
)

#: Labels that may support a *low-bit execution* claim (E05-02 §9.3).
LOW_BIT_LABELS = (FUSED_DEQUANT_WEIGHT_ONLY, INTEGER_TENSOR_OP)

#: Labels that may support a *storage/load* claim only.
STORAGE_LABELS = (STORAGE_ONLY,)

#: Labels usable for algorithm-quality comparison only.
QUALITY_ONLY_LABELS = (FAKE_QUANT, FP16_REFERENCE)


@dataclass
class ExecutionRecord:
    """Per-module-invocation execution evidence (E05-02 §9.3 field list)."""

    module: str
    phase: str
    workload_id: str = ""
    batch: int = 0
    m: int = 0
    n: int = 0
    k: int = 0
    expected_kernel: str = ""
    observed_kernel_symbol: str = ""
    kernel_provider: str = ""
    kernel_version: str = ""
    packed_layout_id: str = ""
    canonical_hash: str = ""
    capability_check: str = ""
    fallback_reason: str = ""
    profiler_trace_hash: str = ""
    weights_materialized_to_fp16: bool = False
    artifact_bytes_read: int = 0
    workspace_bytes: int = 0
    call_count: int = 1
    duration_ms: float = 0.0
    require_observed_symbol: bool = True
    #: Executor-declared label. It is only accepted when it is *consistent*
    #: with the recorded evidence (see :func:`classify_execution`); a claim
    #: that contradicts the evidence is ignored, never trusted.
    declared_label: str = ""

    def execution_label(self) -> str:
        """Classify the observed path from the recorded evidence."""
        return classify_execution(
            observed_kernel_symbol=self.observed_kernel_symbol,
            kernel_provider=self.kernel_provider,
            packed_layout_id=self.packed_layout_id,
            weights_materialized_to_fp16=self.weights_materialized_to_fp16,
            fallback_reason=self.fallback_reason,
            integer_dot_evidence=bool(self.integer_dot_evidence),
            require_observed_symbol=self.require_observed_symbol,
            declared_label=self.declared_label,
        )

    #: Set by the caller when instruction-level integer-dot evidence exists.
    integer_dot_evidence: bool = False

    @property
    def low_bit_executed(self) -> bool:
        return self.execution_label() in LOW_BIT_LABELS

    def as_dict(self) -> Dict[str, Any]:
        label = self.execution_label()
        return {
            "module": self.module,
            "phase": self.phase,
            "workload_id": self.workload_id,
            "batch": self.batch,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "expected_kernel": self.expected_kernel,
            "observed_kernel_symbol": self.observed_kernel_symbol,
            "kernel_provider": self.kernel_provider,
            "kernel_version": self.kernel_version,
            "packed_layout_id": self.packed_layout_id,
            "canonical_hash": self.canonical_hash,
            "capability_check": self.capability_check,
            "fallback_reason": self.fallback_reason,
            "profiler_trace_hash": self.profiler_trace_hash,
            "weights_materialized_to_fp16": self.weights_materialized_to_fp16,
            "artifact_bytes_read": self.artifact_bytes_read,
            "workspace_bytes": self.workspace_bytes,
            "call_count": self.call_count,
            "duration_ms": self.duration_ms,
            "declared_label": self.declared_label,
            "execution_label": label,
            "low_bit_executed": label in LOW_BIT_LABELS,
            "observed_kernel_matches_expected": self.kernel_matches_expected(),
        }

    def kernel_matches_expected(self) -> Optional[bool]:
        """True/False when both symbols are known, ``None`` when unverifiable."""
        if not self.expected_kernel or not self.observed_kernel_symbol:
            return None
        return self.expected_kernel in self.observed_kernel_symbol or (
            self.observed_kernel_symbol in self.expected_kernel
        )


def classify_execution(
    *,
    observed_kernel_symbol: str = "",
    kernel_provider: str = "",
    packed_layout_id: str = "",
    weights_materialized_to_fp16: bool = False,
    fallback_reason: str = "",
    integer_dot_evidence: bool = False,
    require_observed_symbol: bool = True,
    declared_label: str = "",
) -> str:
    """Classify an execution path from its evidence.

    Precedence:

    1. an explicit ``fallback_reason`` → ``explicit_fallback``;
    2. weights materialized to FP16 (or a packed layout that is not the
       executed one) → ``storage_only``;
    3. an observed kernel symbol from a provider plus a packed layout →
       ``fused_dequant_weight_only``;
    4. no packed layout at all → ``fp16_reference`` if a symbol was observed,
       otherwise ``fp16_reference`` as well (unknown execution is never
       promoted to low-bit);
    5. ``integer_dot_evidence`` is required (and only then) to classify an
       integral path as ``integer_tensor_op``.

    ``require_observed_symbol`` enforces "no symbol, no low-bit claim": when
    True (the default), a packed layout without an observed kernel is
    *unverifiable*, which maps to ``fp16_reference`` for claim purposes and
    must be recorded as a missing-evidence finding by the caller.
    """
    if fallback_reason:
        return EXPLICIT_FALLBACK
    if declared_label and declared_label in EXECUTION_LABELS:
        # Accept a declared label only when the evidence cannot contradict it.
        if declared_label == FAKE_QUANT and not packed_layout_id and not weights_materialized_to_fp16:
            return FAKE_QUANT
        if declared_label == STORAGE_ONLY and weights_materialized_to_fp16:
            return STORAGE_ONLY
        if declared_label == FP16_REFERENCE and not packed_layout_id:
            return FP16_REFERENCE
    if weights_materialized_to_fp16:
        return STORAGE_ONLY
    if packed_layout_id:
        if not observed_kernel_symbol and require_observed_symbol:
            # A declared layout without an observed symbol is exactly the
            # "API said int4" trap; do not label it low-bit.
            return FP16_REFERENCE
        if integer_dot_evidence:
            return INTEGER_TENSOR_OP
        if observed_kernel_symbol and kernel_provider:
            return FUSED_DEQUANT_WEIGHT_ONLY
        return STORAGE_ONLY
    return FP16_REFERENCE


def claim_audit(records: Sequence[ExecutionRecord]) -> Dict[str, Any]:
    """Audit a set of records for claim eligibility (E05-10 §5.4).

    Returns the coverage and every record that would *not* support the claim
    it was collected for, so a performance table cannot quietly mix a
    storage-only path into a low-bit speedup.
    """
    counts: Dict[str, int] = {label: 0 for label in EXECUTION_LABELS}
    missing_symbols: List[str] = []
    mismatches: List[Dict[str, Any]] = []
    fallback_records: List[Dict[str, Any]] = []
    for record in records:
        label = record.execution_label()
        counts[label] = counts.get(label, 0) + 1
        if record.packed_layout_id and not record.observed_kernel_symbol:
            missing_symbols.append(record.module)
        match = record.kernel_matches_expected()
        if match is False:
            mismatches.append(
                {
                    "module": record.module,
                    "expected_kernel": record.expected_kernel,
                    "observed_kernel_symbol": record.observed_kernel_symbol,
                }
            )
        if label == EXPLICIT_FALLBACK:
            fallback_records.append(
                {"module": record.module, "fallback_reason": record.fallback_reason}
            )
    low_bit_records = [
        record for record in records if record.execution_label() in LOW_BIT_LABELS
    ]
    return {
        "total_records": len(records),
        "label_counts": {key: value for key, value in counts.items() if value},
        "low_bit_records": len(low_bit_records),
        "low_bit_eligible": bool(low_bit_records),
        "missing_observed_symbol": sorted(set(missing_symbols)),
        "kernel_symbol_mismatches": mismatches,
        "fallback_records": fallback_records,
    }


def records_to_jsonl(records: Sequence[ExecutionRecord]) -> str:
    return "\n".join(
        json.dumps(record.as_dict(), sort_keys=True, ensure_ascii=False)
        for record in records
    )


def parse_records(payloads: Sequence[Mapping[str, Any]]) -> List[ExecutionRecord]:
    """Rebuild records from their serialized form (round-trip safe)."""
    records: List[ExecutionRecord] = []
    known = set(ExecutionRecord.__dataclass_fields__)  # type: ignore[attr-defined]
    for payload in payloads:
        filtered = {key: value for key, value in payload.items() if key in known}
        unknown = set(payload) - known - {
            "execution_label",
            "low_bit_executed",
            "observed_kernel_matches_expected",
        }
        if unknown:
            raise ConfigError(
                f"execution record has unknown field(s): {sorted(unknown)}"
            )
        records.append(ExecutionRecord(**filtered))
    return records


__all__ = [
    "EXECUTION_LABELS",
    "EXPLICIT_FALLBACK",
    "ExecutionRecord",
    "FAKE_QUANT",
    "FP16_REFERENCE",
    "FUSED_DEQUANT_WEIGHT_ONLY",
    "INTEGER_TENSOR_OP",
    "LOW_BIT_LABELS",
    "QUALITY_ONLY_LABELS",
    "STORAGE_LABELS",
    "STORAGE_ONLY",
    "claim_audit",
    "classify_execution",
    "parse_records",
    "records_to_jsonl",
]
