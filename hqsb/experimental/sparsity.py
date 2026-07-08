"""E14-F4 — structured sparsity / sparse attention and the gap to real speedup.

Conditional P0: the main branch only when ``E14-05`` selects F4, otherwise
``N/A_BY_ADR``.  Two routes exist and only one may be primary (§2):

* **route A** — structured *weight* sparsity (N:M, commonly 2:4), where the
  artefacts are prune/retrain, a compressed format, sparse GEMM, shapes and
  library/hardware support;
* **route B** — *sparse/linear attention*, where the mathematical object changes
  and the oracle must be long-range quality rather than a small tensor diff.

Three quantities must never be conflated (§4.1):

```text
logical_sparsity    = zero_elements / total_elements
pattern_compliance  = valid_sparse_groups / total_groups
effective_sparse_work = work actually executed by the sparse kernel
```

and the total cost is ``convert + metadata/index/mask + sparse_kernel +
reorder/restore + fallback`` — a conversion that is amortised offline still has
to be reported (§4.2).

Interfaces provided:

* :class:`SparseTransform` / :class:`SparseArtifactSchema` (steps 3–4);
* :func:`pattern_compliance` — per-tensor N:M compliance with exceptions (step 14);
* :func:`compressed_roundtrip` — values/metadata order and padding (step 15);
* :func:`sparse_gemm_reference` — a dense execution of the *sparse* weights, so
  pruning error is separated from kernel error (steps 16–17);
* :func:`check_actual_dispatch` — the kernel that really ran, with fallback
  reasons (steps 11, 19, 33);
* :func:`metadata_cost` / :func:`conversion_cost` (steps 21–22);
* :func:`coverage_and_amdahl` — layer coverage and the residual that micro
  speedups cannot cross (steps 23–24, 27);
* :func:`attention_quality_by_distance` — route B's long-range oracle (step 26);
* :func:`unsupported_capability`, :func:`corrupted_artifact_check`,
  :func:`version_mismatch_check`, :func:`fallback_policy` (steps 33–36);
* :func:`reconcile_prediction`, :class:`F4AdoptionDecision` (steps 37, 40).

Nothing here calls cuSPARSELt or compiles anything; the arithmetic is over
supplied mask/value shapes, so a library's claimed speedup can be checked against
an independent cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import AdoptionDecision
from hqsb.experimental.identity import SCHEMA_PREFIX, is_digest

EXPERIMENT_ID = "E14-F4"
TITLE = "结构化稀疏/Sparse Attention 的真实硬件加速"
LEVEL = "条件 P0"

CLAIM_BOUNDARY = (
    "只在 F4 为唯一主分支时成立；零元素比例不等于硬件稀疏，micro speedup 不等于模型收益。"
    "严格证明“当前硬件/shape 不值得采用”同样是有价值的结果（E14-F4 §12）。"
)

#: Route A/B — exactly one is primary (§2).
ROUTES: Tuple[str, ...] = rec.SPARSITY_ROUTES

#: Weight patterns route A may use.
WEIGHT_PATTERNS: Tuple[str, ...] = ("2:4", "4:8", "1:2", "block", "unstructured")

#: Attention patterns route B may use.
ATTENTION_PATTERNS: Tuple[str, ...] = ("local_window", "block_sparse", "linear", "low_rank", "streaming")

#: Dtypes a sparse kernel may accept.
SPARSE_DTYPES: Tuple[str, ...] = ("fp16", "bf16", "tf32", "fp8", "int8")

#: Fallback reasons a dispatch decision may carry (step 33).
DISPATCH_REASONS: Tuple[str, ...] = (
    "PATTERN_UNSUPPORTED",
    "DTYPE_UNSUPPORTED",
    "SHAPE_UNSUPPORTED",
    "VERSION_UNSUPPORTED",
    "ALIGNMENT_MISMATCH",
    "AVAILABLE",
)

#: Cost components of §4.2.
SPARSE_COST_COMPONENTS: Tuple[str, ...] = (
    "convert_or_compress",
    "metadata_index_mask",
    "sparse_kernel",
    "reorder_restore",
    "fallback",
)

#: Quality metrics route B must report by distance (step 26).
DISTANCE_METRICS: Tuple[str, ...] = ("retrieval_recall", "long_qa_accuracy", "perplexity", "exact_match")


@dataclass(frozen=True)
class SparseTransform:
    """Step 3: how the sparse artefact was produced — algorithm, version, randomness."""

    route: str
    pattern: str
    algorithm: str
    version: str
    sparsity_target: float
    retrained: bool = False
    retrain_steps: int = 0
    seed: Optional[int] = None
    axis: int = 0
    per_layer: Mapping[str, float] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.route not in ROUTES:
            findings.append(f"SparseTransform: route {self.route!r} must be one of {', '.join(ROUTES)}")
        allowed = WEIGHT_PATTERNS if self.route == ROUTES[0] else ATTENTION_PATTERNS
        if self.pattern not in allowed:
            findings.append(
                f"SparseTransform: pattern {self.pattern!r} is not valid for route {self.route!r}; "
                f"allowed: {', '.join(allowed)}"
            )
        for name in ("algorithm", "version"):
            if not getattr(self, name):
                findings.append(f"SparseTransform: {name} is required (最终零值比例无法重现产生过程)")
        if not 0.0 <= self.sparsity_target < 1.0:
            findings.append(f"SparseTransform: sparsity_target {self.sparsity_target} must be in [0, 1)")
        if self.retrained and self.retrain_steps <= 0:
            findings.append("SparseTransform: retrained without recording retrain_steps")
        if self.route == ROUTES[0] and self.pattern == "unstructured":
            findings.append(
                "SparseTransform: unstructured sparsity cannot claim a hardware sparse path "
                "(2:4 是硬件契约，不是任意 50% 稀疏)"
            )
        if self.route == ROUTES[1] and self.per_layer:
            findings.append("SparseTransform: per_layer sparsity belongs to route A; route B changes the mask")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "pattern": self.pattern,
            "algorithm": self.algorithm,
            "version": self.version,
            "sparsity_target": self.sparsity_target,
            "retrained": self.retrained,
            "retrain_steps": self.retrain_steps,
            "seed": self.seed,
            "axis": self.axis,
            "per_layer": {key: self.per_layer[key] for key in sorted(self.per_layer)},
        }


@dataclass(frozen=True)
class SparseArtifactSchema:
    """Step 4: the schema a kernel consumes, not a library's private state."""

    sparse_artifact_id: str
    source_artifact_id: str
    pattern: str
    dtype: str
    axis: int
    shape: Tuple[int, ...]
    value_count: int
    metadata_count: int
    alignment: int = 16
    endianness: str = "little"
    format_version: str = ""
    compatibility_digest: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("sparse_artifact_id", "source_artifact_id", "format_version"):
            if not getattr(self, name):
                findings.append(f"SparseArtifactSchema: {name} is required")
        if self.dtype not in SPARSE_DTYPES:
            findings.append(
                f"SparseArtifactSchema: dtype {self.dtype!r} must be one of {', '.join(SPARSE_DTYPES)}"
            )
        if len(self.shape) < 2:
            findings.append(f"SparseArtifactSchema: shape {self.shape} has fewer than two axes")
        if self.value_count <= 0:
            findings.append("SparseArtifactSchema: value_count must be positive")
        if self.metadata_count < 0:
            findings.append("SparseArtifactSchema: metadata_count must be >= 0")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            findings.append(f"SparseArtifactSchema: alignment {self.alignment} is not a power of two")
        if self.endianness not in ("little", "big"):
            findings.append(f"SparseArtifactSchema: unknown endianness {self.endianness!r}")
        if not is_digest(self.source_artifact_id):
            findings.append("SparseArtifactSchema: source_artifact_id must be sha256:<hex>")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sparse_artifact_id": self.sparse_artifact_id,
            "source_artifact_id": self.source_artifact_id,
            "pattern": self.pattern,
            "dtype": self.dtype,
            "axis": self.axis,
            "shape": list(self.shape),
            "value_count": self.value_count,
            "metadata_count": self.metadata_count,
            "alignment": self.alignment,
            "endianness": self.endianness,
            "format_version": self.format_version,
            "compatibility_digest": self.compatibility_digest,
        }


def pattern_compliance(
    masks: Mapping[str, Sequence[Sequence[int]]], *, n: int = 2, m: int = 4
) -> Dict[str, Any]:
    """Step 14: per-tensor N:M compliance, so a global 50% cannot hide a bad matrix.

    The pattern ``n:m`` means *m consecutive elements along the axis contain
    exactly n non-zeros*; a matrix that is 50% sparse overall but violates the
    grouping does **not** qualify for the hardware path (§4.3).
    """
    if n <= 0 or m <= 0 or n > m:
        raise ConfigError(f"invalid N:M pattern {n}:{m}")
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    total_groups = compliant_groups = 0
    for name, matrix in sorted(masks.items()):
        tensor_groups = tensor_compliant = zeros = elements = 0
        for row, values in enumerate(matrix):
            if len(values) % m:
                problems.append(
                    f"{name} row {row}: length {len(values)} is not a multiple of the group size {m}"
                )
            elements += len(values)
            zeros += sum(1 for value in values if value == 0)
            for start in range(0, len(values) - m + 1, m):
                group = values[start : start + m]
                tensor_groups += 1
                if sum(1 for value in group if value != 0) == n:
                    tensor_compliant += 1
        total_groups += tensor_groups
        compliant_groups += tensor_compliant
        rows.append(
            {
                "tensor": name,
                "elements": elements,
                "zeros": zeros,
                "logical_sparsity": zeros / elements if elements else 0.0,
                "groups": tensor_groups,
                "compliant_groups": tensor_compliant,
                "pattern_compliance": tensor_compliant / tensor_groups if tensor_groups else 0.0,
            }
        )
    overall = compliant_groups / total_groups if total_groups else 0.0
    if rows and any(row["pattern_compliance"] < 1.0 for row in rows):
        problems.append(
            "at least one tensor is not fully pattern compliant: 全局 50% 掩盖部分矩阵完全不合规（step 14）"
        )
    return {
        "pattern": f"{n}:{m}",
        "rows": rows,
        "pattern_compliance": overall,
        "total_groups": total_groups,
        "compliant_groups": compliant_groups,
        "hardware_eligible": bool(rows) and overall == 1.0,
        "problems": problems,
    }


def compressed_roundtrip(
    *,
    values: Sequence[float],
    metadata: Sequence[int],
    expected_values: Sequence[float],
    expected_metadata: Sequence[int],
    padding_tokens: int = 0,
) -> Dict[str, Any]:
    """Step 15: value order and metadata must survive a compress→decompress round trip.

    A compression that is smaller but interpreted in the wrong order produces
    plausible-looking numbers, which is why the round trip compares both arrays
    rather than the byte count.
    """
    problems: List[str] = []
    if list(values) != list(expected_values):
        problems.append(f"value order changed: {list(values)[:4]}… != {list(expected_values)[:4]}…")
    if list(metadata) != list(expected_metadata):
        problems.append(f"metadata changed: {list(metadata)[:4]}… != {list(expected_metadata)[:4]}…")
    if padding_tokens < 0:
        problems.append("padding_tokens must be >= 0")
    if not values:
        problems.append("no values supplied")
    return {
        "values": len(values),
        "metadata": len(metadata),
        "padding_tokens": padding_tokens,
        "roundtrip_ok": not problems,
        "problems": problems,
    }


def sparse_gemm_reference(
    *,
    dense_a: Sequence[Sequence[float]],
    sparse_b_values: Sequence[Sequence[float]],
    mask_b: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Steps 16–17: dense execution of the *sparse* weights, before any kernel.

    ``E14-F4`` §6 step 17 requires the pruning algorithm's error and the kernel's
    error to be separated; this reference is the first, so a later comparison
    against a sparse kernel cannot attribute a kernel bug to the algorithm.
    """
    if len(sparse_b_values) != len(mask_b):
        raise ConfigError("sparse_b_values and mask_b must have the same number of rows")
    width = len(dense_a[0]) if dense_a else 0
    if any(len(row) != width for row in dense_a):
        raise ConfigError("dense_a rows have inconsistent width")
    if len(sparse_b_values) != width:
        raise ConfigError(
            f"sparse_b_values has {len(sparse_b_values)} rows but dense_a has width {width}"
        )
    columns = len(sparse_b_values[0]) if sparse_b_values else 0
    result: List[List[float]] = []
    for row in dense_a:
        out_row: List[float] = []
        for column in range(columns):
            total = 0.0
            for index, value in enumerate(row):
                if mask_b[index][column]:
                    total += value * sparse_b_values[index][column]
            out_row.append(total)
        result.append(out_row)
    zeros = sum(1 for row in mask_b for value in row if value == 0)
    elements = sum(len(row) for row in mask_b)
    return {
        "shape": [len(dense_a), columns],
        "result": result,
        "logical_sparsity": zeros / elements if elements else 0.0,
        "note": "这是算法参考（dense 执行稀疏权重），不是 kernel 结果",
    }


def check_actual_dispatch(
    *,
    requested: str,
    actual_kernel: str,
    fallback: bool,
    reason_code: str,
    evidence: str,
    pattern_compliance_value: float,
) -> Dict[str, Any]:
    """Steps 11/19/33: the sparse path must be *hit*, and a fallback must be reasoned.

    ``E14-F4`` §10: sparse artifact 实际走 dense is a FAIL even when the numbers
    look good, because the claim would then be about the dense kernel.
    """
    problems: List[str] = []
    if not actual_kernel:
        problems.append(
            "no actual kernel recorded: 环境变量或配置名不是执行证据（step 11）"
        )
    if requested and requested == actual_kernel and fallback:
        problems.append("the dispatch reports both hit and fallback for the same kernel")
    if fallback and not reason_code:
        problems.append("a fallback without a reason code is a silent degradation")
    if reason_code and reason_code not in DISPATCH_REASONS:
        problems.append(f"unknown dispatch reason {reason_code!r}")
    if not evidence:
        problems.append("no profiler/trace/library evidence for the dispatch decision")
    if pattern_compliance_value < 1.0 and not fallback:
        problems.append(
            f"dispatch reports the sparse path although pattern compliance is {pattern_compliance_value:.4f}"
        )
    return {
        "requested": requested,
        "actual_kernel": actual_kernel,
        "fallback": fallback,
        "reason_code": reason_code,
        "evidence": evidence,
        "sparse_path_hit": bool(actual_kernel) and not fallback,
        "problems": problems,
    }


def metadata_cost(
    *, metadata_count: int, metadata_bytes_per_entry: int, value_count: int, value_bytes: int
) -> Dict[str, Any]:
    """Step 21: metadata is part of the size, and it is often the majority."""
    if metadata_count < 0 or value_count <= 0:
        raise ConfigError("metadata_count >= 0 and value_count > 0 are required")
    metadata_bytes = metadata_count * metadata_bytes_per_entry
    value_bytes_total = value_count * value_bytes
    total = metadata_bytes + value_bytes_total
    return {
        "metadata_bytes": metadata_bytes,
        "value_bytes": value_bytes_total,
        "total_bytes": total,
        "metadata_ratio": metadata_bytes / total if total else 0.0,
        "note": "只计算 values 大小会漏掉 index/mask/padding（step 21）",
    }


@dataclass(frozen=True)
class SparseExecutionRecord:
    """``E14-F4`` §8 ``SparseExecutionRecord`` (schema ``…e14-f4.exec.v1``)."""

    operator_id: str
    source_artifact_id: str
    sparse_artifact_id: str
    pattern: str
    shape: Tuple[int, ...] = ()
    dtype: str = ""
    logical_sparsity: Optional[float] = None
    pattern_compliance: Optional[float] = None
    compressed_value_bytes: int = 0
    metadata_bytes: int = 0
    actual_kernel: str = ""
    fallback: bool = False
    latency_ms: Optional[float] = None
    correctness_status: str = rec.STATUS_NOT_RUN
    quality_eligible: bool = False

    schema_version = f"{SCHEMA_PREFIX}.e14-f4.exec.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        for name in ("operator_id", "source_artifact_id", "sparse_artifact_id", "pattern"):
            if not getattr(self, name):
                findings.append(f"SparseExecutionRecord: {name} is required")
        if self.logical_sparsity is None:
            findings.append("SparseExecutionRecord: logical_sparsity is unmeasured")
        if self.pattern_compliance is None:
            findings.append(
                "SparseExecutionRecord: pattern_compliance is unmeasured (逻辑稀疏不是执行稀疏)"
            )
        if not self.actual_kernel:
            findings.append("SparseExecutionRecord: actual_kernel is required")
        if self.latency_ms is None:
            findings.append("SparseExecutionRecord: latency is unmeasured")
        if self.quality_eligible and self.correctness_status != rec.STATUS_PASS:
            findings.append(
                "SparseExecutionRecord: quality_eligible requires correctness to have passed first"
            )
        if self.fallback and not self.quality_eligible:
            pass  # a reasoned fallback that never claimed quality eligibility is consistent
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operator_id": self.operator_id,
            "source_artifact_id": self.source_artifact_id,
            "sparse_artifact_id": self.sparse_artifact_id,
            "pattern": self.pattern,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "logical_sparsity": self.logical_sparsity,
            "pattern_compliance": self.pattern_compliance,
            "compressed_value_bytes": self.compressed_value_bytes,
            "metadata_bytes": self.metadata_bytes,
            "actual_kernel": self.actual_kernel,
            "fallback": self.fallback,
            "latency_ms": self.latency_ms,
            "correctness_status": self.correctness_status,
            "quality_eligible": self.quality_eligible,
        }


def conversion_cost(
    *, offline_ms: float, online_ms: float, amortisation_requests: Optional[int], peak_bytes: int
) -> Dict[str, Any]:
    """Step 22/§4.2: offline conversion is separated from the per-request cost.

    ``E14-F4`` §10: 在线 conversion 被排除 is the failure; an offline conversion
    that is never amortised is reported as a one-off publication cost instead of
    disappearing from the account.
    """
    problems: List[str] = []
    if offline_ms < 0 or online_ms < 0:
        problems.append("conversion times must be >= 0")
    per_request = online_ms
    amortised: Optional[float] = None
    if amortisation_requests:
        if amortisation_requests <= 0:
            problems.append("amortisation_requests must be positive when supplied")
        else:
            amortised = offline_ms / amortisation_requests
            per_request = online_ms + amortised
    elif offline_ms > 0:
        problems.append(
            "offline conversion is not amortised; report it as a one-off publication cost rather than "
            "removing it from the account"
        )
    return {
        "offline_ms": offline_ms,
        "online_ms": online_ms,
        "amortisation_requests": amortisation_requests,
        "amortised_offline_ms": amortised,
        "per_request_ms": per_request,
        "peak_bytes": peak_bytes,
        "components": list(SPARSE_COST_COMPONENTS),
        "problems": problems,
    }


def coverage_and_amdahl(
    *, layer_times_ms: Mapping[str, float], sparse_kernel_speedup: float, eligible_layers: Sequence[str]
) -> Dict[str, Any]:
    """Steps 23–24/27: coverage decides how far a micro speedup can travel.

    ``E14-F4`` §10: micro speedup 直接外推模型 is the classic error, so the model
    prediction is computed from the covered fraction and the residual is what an
    explanation has to account for.
    """
    if sparse_kernel_speedup <= 0:
        raise ConfigError("sparse_kernel_speedup must be positive")
    total = sum(float(value) for value in layer_times_ms.values())
    if total <= 0:
        return {"error": "layer times sum to zero"}
    eligible = [name for name in eligible_layers if name in layer_times_ms]
    if not eligible:
        return {"error": "no eligible layer appears in the layer times"}
    covered = sum(float(layer_times_ms[name]) for name in eligible)
    coverage = covered / total
    predicted_speedup = 1.0 / ((1.0 - coverage) + coverage / sparse_kernel_speedup)
    return {
        "coverage": coverage,
        "covered_ms": covered,
        "total_ms": total,
        "sparse_kernel_speedup": sparse_kernel_speedup,
        "predicted_model_speedup": predicted_speedup,
        "residual_uncovered": 1.0 - coverage,
        "note": "hotspot 覆盖之外的 Amdahl 残差必须显式报告（step 27）",
    }


def attention_quality_by_distance(
    rows: Sequence[Mapping[str, Any]], *, guard_band: float
) -> Dict[str, Any]:
    """Step 26: route B must be judged by distance, not by a small tensor diff.

    ``E14-F4`` §10: sparse attention 不测长距离质量 is a FAIL; a mean over all
    distances can hide a collapse beyond the window.
    """
    problems: List[str] = []
    if not rows:
        return {"error": "no quality rows supplied"}
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        if "distance_bucket" not in row or "delta" not in row:
            problems.append("quality row is missing distance_bucket or delta")
            continue
        buckets.setdefault(str(row["distance_bucket"]), []).append(float(row["delta"]))
    summary = {name: sum(values) / len(values) for name, values in sorted(buckets.items())}
    worst = min(summary.items(), key=lambda item: item[1]) if summary else None
    if worst and worst[1] < -abs(guard_band):
        problems.append(
            f"distance bucket {worst[0]} falls {abs(worst[1]):.4f} below dense, beyond the guard band"
        )
    return {
        "by_distance": summary,
        "worst_bucket": worst[0] if worst else None,
        "guard_band": guard_band,
        "problems": problems,
    }


# ── capability, shapes, compile/cache, service (steps 18, 28–33) ──────────


@dataclass(frozen=True)
class SupportDomain:
    """Step 5: the compute capability/library/dtype/layout/shape envelope."""

    compute_capability: str
    library: str
    library_version: str
    dtypes: Tuple[str, ...] = ()
    min_shape: Tuple[int, int, int] = (0, 0, 0)
    alignment: int = 16
    notes: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("compute_capability", "library", "library_version"):
            if not getattr(self, name):
                findings.append(f"SupportDomain: {name} is required (版本必须绑定执行时版本)")
        unknown = [dtype for dtype in self.dtypes if dtype not in SPARSE_DTYPES]
        if unknown:
            findings.append(f"SupportDomain: unsupported dtypes declared: {', '.join(unknown)}")
        if not self.dtypes:
            findings.append("SupportDomain: no dtypes declared")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            findings.append(f"SupportDomain: alignment {self.alignment} is not a power of two")
        if len(self.min_shape) != 3:
            findings.append("SupportDomain: min_shape must be (M, N, K)")
        return findings

    def supports(self, *, M: int, N: int, K: int, dtype: str) -> Dict[str, Any]:
        """Step 33: an unsupported request must be refused or explicitly fallen back."""
        reasons: List[str] = []
        if dtype not in self.dtypes:
            reasons.append("DTYPE_UNSUPPORTED")
        m_min, n_min, k_min = self.min_shape
        if M < m_min or N < n_min or K < k_min:
            reasons.append("SHAPE_UNSUPPORTED")
        for axis, value in (("M", M), ("N", N), ("K", K)):
            if value % self.alignment:
                reasons.append("ALIGNMENT_MISMATCH")
                break
        return {
            "supported": not reasons,
            "reason_code": reasons[0] if reasons else "AVAILABLE",
            "all_reasons": sorted(set(reasons)),
            "shape": {"M": M, "N": N, "K": K},
            "dtype": dtype,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compute_capability": self.compute_capability,
            "library": self.library,
            "library_version": self.library_version,
            "dtypes": list(self.dtypes),
            "min_shape": list(self.min_shape),
            "alignment": self.alignment,
            "notes": self.notes,
        }


def shape_scan(
    rows: Sequence[Mapping[str, Any]], *, domain: SupportDomain, dtype: str
) -> Dict[str, Any]:
    """Step 20: interleaved dense/sparse over the *real* shape ledger.

    ``E14-F4`` §10: 只测友好整除 shape is how a tail-shape regression is missed, so
    the scan reports the unsupported tails separately instead of averaging them.
    """
    supported: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    for row in rows:
        for key in ("M", "N", "K", "dense_ms", "sparse_ms"):
            if key not in row:
                unsupported.append({"row": dict(row), "problem": f"missing {key}"})
                break
        else:
            verdict = domain.supports(M=int(row["M"]), N=int(row["N"]), K=int(row["K"]), dtype=dtype)
            if verdict["supported"]:
                dense = float(row["dense_ms"])
                sparse = float(row["sparse_ms"])
                supported.append(
                    {
                        "M": row["M"], "N": row["N"], "K": row["K"],
                        "dense_ms": dense, "sparse_ms": sparse,
                        "speedup": dense / sparse if sparse else float("inf"),
                    }
                )
            else:
                unsupported.append({"shape": verdict["shape"], "reason_code": verdict["reason_code"]})
    return {
        "supported": supported,
        "unsupported": unsupported,
        "dense_only_shapes": len(unsupported),
        "note": "不支持/非对齐 shape 单列，不参与平均（step 18/20）",
    }


def compile_cache_check(
    *, dynamic_guards: Sequence[str], compiled_artifact_digest: str, cache_hit: bool,
    recompile_time_ms: Optional[float], unsupported_fallback: bool,
) -> Dict[str, Any]:
    """Step 28: guards, cache identity and recompile time must not hide the cost."""
    problems: List[str] = []
    if not dynamic_guards:
        problems.append("no dynamic guards recorded: the compiled artifact may be reused on an invalid shape")
    if not compiled_artifact_digest:
        problems.append("compiled artifact has no identity (engine 复用不可追踪)")
    elif not is_digest(compiled_artifact_digest):
        problems.append("compiled_artifact_digest must be sha256:<hex>")
    if recompile_time_ms is None:
        problems.append("recompile time is unmeasured (重新编译时间不得被隐藏)")
    if unsupported_fallback and not dynamic_guards:
        problems.append("an unsupported shape fell back without any guard being recorded")
    return {
        "guards": list(dynamic_guards),
        "cache_hit": cache_hit,
        "recompile_time_ms": recompile_time_ms,
        "compiled_artifact_digest": compiled_artifact_digest,
        "unsupported_fallback": unsupported_fallback,
        "problems": problems,
        "ok": not problems,
    }


def runtime_dynamic_workload(
    *, fixed_batch_speedup: float, dynamic_batch_speedup: float, fallback_fraction: float
) -> Dict[str, Any]:
    """Step 29: a fixed-shape kernel speedup can vanish under dynamic batching.

    ``E14-F4`` §10: 固定 batch kernel speedup 在动态 Runtime 消失却不报告 — the
    degradation is reported, and a fallback fraction above zero means the sparse
    path covered less work than the micro benchmark suggested.
    """
    problems: List[str] = []
    if fixed_batch_speedup <= 0:
        problems.append("fixed_batch_speedup must be positive")
    if fallback_fraction < 0 or fallback_fraction > 1:
        problems.append(f"fallback_fraction {fallback_fraction} is outside [0, 1]")
    if dynamic_batch_speedup < 1.0 < fixed_batch_speedup:
        problems.append(
            f"the speedup reverses under dynamic batching ({fixed_batch_speedup:.3f}× → "
            f"{dynamic_batch_speedup:.3f}×) and must be reported, not averaged away"
        )
    return {
        "fixed_batch_speedup": fixed_batch_speedup,
        "dynamic_batch_speedup": dynamic_batch_speedup,
        "fallback_fraction": fallback_fraction,
        "problems": problems,
        "adaptive_dispatch_recommended": problems and fallback_fraction > 0,
    }


def service_curve(rows: Sequence[Mapping[str, Any]], *, required: Sequence[str]) -> Dict[str, Any]:
    """Step 30: the service result must keep the device count and SLO fixed."""
    problems: List[str] = []
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            problems.append(f"service row {row.get('variant', '<unnamed>')} is missing {', '.join(missing)}")
    return {
        "rows": len(rows),
        "fields": list(required),
        "problems": problems,
        "note": "峰值离线 throughput 不能代替服务收益（step 30）",
    }


def memory_and_capacity(
    *, dense_weight_bytes: int, sparse_value_bytes: int, sparse_metadata_bytes: int,
    workspace_bytes: int, engine_bytes: int,
) -> Dict[str, Any]:
    """Step 31/§9: a smaller file does not imply smaller resident memory.

    ``E14-F4`` §9: 文件小、显存不降 often comes from workspace, engine or a
    decompressed resident copy; all five contributions are returned together.
    """
    total = sparse_value_bytes + sparse_metadata_bytes + workspace_bytes + engine_bytes
    return {
        "dense_weight_bytes": dense_weight_bytes,
        "sparse_value_bytes": sparse_value_bytes,
        "sparse_metadata_bytes": sparse_metadata_bytes,
        "workspace_bytes": workspace_bytes,
        "engine_bytes": engine_bytes,
        "resident_bytes": total,
        "weight_only_ratio": (sparse_value_bytes / dense_weight_bytes) if dense_weight_bytes else 0.0,
        "resident_ratio": (total / dense_weight_bytes) if dense_weight_bytes else 0.0,
        "note": "只引用磁盘压缩率会漏掉 workspace/engine/解压驻留（step 31）",
    }


def energy_and_cost(
    *, variant_energy_j: float, requests: int, ammortised_conversion_j: Optional[float]
) -> Dict[str, Any]:
    """Step 32: energy per request, with the offline conversion amortisation named."""
    problems: List[str] = []
    if requests <= 0:
        problems.append("requests must be positive")
    if variant_energy_j < 0:
        problems.append("energy must be >= 0")
    per_request = variant_energy_j / requests if requests else 0.0
    with_conversion: Optional[float] = None
    if ammortised_conversion_j is not None:
        if requests <= 0:
            problems.append("cannot amortise conversion over zero requests")
        else:
            with_conversion = (variant_energy_j + ammortised_conversion_j) / requests
    else:
        problems.append(
            "offline conversion energy not supplied: report it or state explicitly that it is unmeasured"
        )
    return {
        "requests": requests,
        "energy_j": variant_energy_j,
        "energy_per_request_j": per_request,
        "energy_per_request_with_conversion_j": with_conversion,
        "problems": problems,
    }


# ── faults and reconciliation (steps 33–37) ────────────────────────────────


def unsupported_capability(case: Mapping[str, Any], *, domain: SupportDomain) -> Dict[str, Any]:
    """Step 33: an unsupported dtype/arch/alignment/shape must not silently succeed."""
    verdict = domain.supports(
        M=int(case.get("M", 0)), N=int(case.get("N", 0)), K=int(case.get("K", 0)),
        dtype=str(case.get("dtype", "")),
    )
    problems: List[str] = []
    if not verdict["supported"] and case.get("dense_fallback") is False and not case.get("rejected"):
        problems.append(
            "an unsupported request neither fell back nor was rejected (silent wrong result 或未记录回退)"
        )
    if case.get("dense_fallback") and not case.get("reason_code"):
        problems.append("a dense fallback without a reason code is a silent degradation")
    return {"verdict": verdict, "problems": problems, "acceptable": not problems}


def corrupted_artifact_check(
    *, checksum_valid: bool, index_monotonic: bool, mask_shape_matches: bool, rejected_before_execution: bool
) -> Dict[str, Any]:
    """Step 34: corrupt metadata/index/mask must be refused *before* execution.

    ``E14-F4`` §10: 只靠数值质量偶然发现损坏 means an out-of-bounds access was
    possible; the check therefore requires a pre-execution rejection.
    """
    problems: List[str] = []
    if not checksum_valid and not rejected_before_execution:
        problems.append("a checksum mismatch reached execution")
    if not index_monotonic and not rejected_before_execution:
        problems.append("a non-monotonic index reached execution (越界访问风险)")
    if not mask_shape_matches and not rejected_before_execution:
        problems.append("a mask shape mismatch reached execution")
    if not rejected_before_execution:
        problems.append(
            "no pre-execution validation recorded: 依赖数值质量偶然发现损坏是不合格的"
        )
    return {
        "checksum_valid": checksum_valid,
        "index_monotonic": index_monotonic,
        "mask_shape_matches": mask_shape_matches,
        "rejected_before_execution": rejected_before_execution,
        "problems": problems,
        "safe": not problems,
    }


def version_mismatch_check(
    *, artifact_format_version: str, runtime_format_version: str,
    device_compute_capability: str, required_compute_capability: str,
) -> Dict[str, Any]:
    """Step 35: a serialisable file is not a cross-version universal artefact."""
    problems: List[str] = []
    if artifact_format_version != runtime_format_version:
        problems.append(
            f"format version mismatch: artifact {artifact_format_version!r} vs runtime "
            f"{runtime_format_version!r}"
        )
    if device_compute_capability != required_compute_capability:
        problems.append(
            f"compute capability mismatch: device {device_compute_capability!r} vs required "
            f"{required_compute_capability!r}"
        )
    return {
        "artifact_format_version": artifact_format_version,
        "runtime_format_version": runtime_format_version,
        "device_compute_capability": device_compute_capability,
        "required_compute_capability": required_compute_capability,
        "compatible": not problems,
        "problems": problems,
        "status": rec.STATUS_NOT_RUN if not problems else rec.STATUS_INVALID_IDENTITY,
    }


def fallback_policy(
    *, sparse_backend_failed: bool, policy_allows_dense: bool, dense_output_matches: bool,
    trace_updated: bool, capacity_model_updated: bool,
) -> Dict[str, Any]:
    """Step 36: recovery after a sparse failure must be explicit and complete.

    ``E14-F4`` §10: 错误后继续使用部分状态 is a FAIL, so switching to dense
    requires the trace, the capacity model and the output to be consistent.
    """
    problems: List[str] = []
    if not sparse_backend_failed:
        return {"fallback_applied": False, "problems": [], "status": rec.STATUS_NOT_RUN}
    if not policy_allows_dense:
        if not dense_output_matches:
            problems.append("the policy forbids dense but the output was produced anyway")
        return {
            "fallback_applied": False,
            "action": "fail_closed",
            "problems": problems,
            "status": rec.STATUS_NOT_RUN if not problems else rec.STATUS_FAIL_CORRECTNESS,
        }
    if not dense_output_matches:
        problems.append("the dense fallback output differs from the dense reference")
    if not trace_updated:
        problems.append("the trace still reports the sparse path after a fallback")
    if not capacity_model_updated:
        problems.append("the capacity model still uses the sparse footprint after a fallback")
    return {
        "fallback_applied": True,
        "action": "dense",
        "reason_code": "IMPLEMENTATION_ERROR",
        "problems": problems,
        "status": rec.STATUS_NOT_RUN if not problems else rec.STATUS_FAIL_RECOVERY,
    }


def reconcile_prediction(
    *, predicted_model_speedup: float, measured_model_speedup: float,
    coverage: float, kernel_speedup: float, conversion_ratio: float,
) -> Dict[str, Any]:
    """Step 37: rebuild the model effect from kernel speedup, coverage and conversion."""
    if predicted_model_speedup <= 0:
        return {"error": "predicted_model_speedup must be positive"}
    residual = measured_model_speedup - predicted_model_speedup
    return {
        "predicted_model_speedup": predicted_model_speedup,
        "measured_model_speedup": measured_model_speedup,
        "residual": residual,
        "residual_ratio": residual / predicted_model_speedup,
        "mediators": {"coverage": coverage, "kernel_speedup": kernel_speedup, "conversion_ratio": conversion_ratio},
        "explained": abs(residual) <= 0.15 * predicted_model_speedup,
        "note": "micro 2× 不得直接写成模型 2×；残差必须由 coverage/conversion/fallback 解释（step 37）",
    }


# ── adoption (step 40) ─────────────────────────────────────────────────────


def f4_adoption(
    *,
    decision_id: str,
    route: str,
    compliance: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    quality: Mapping[str, Any],
    cost: Mapping[str, Any],
    coverage: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> AdoptionDecision:
    """Step 40: theory being right is not a reason to adopt (§13).

    A rigorous "this hardware/shape is not worth it" is a high-value outcome, so
    an unmet pattern contract or a speedup that does not survive its own costs
    maps to ``REJECT_*`` with the reason named rather than to a forced positive.
    """
    problems: List[str] = []
    if route not in ROUTES:
        problems.append(f"unknown route {route!r}")
    if not compliance.get("hardware_eligible"):
        problems.append("pattern compliance does not meet the hardware contract")
    if not dispatch.get("sparse_path_hit"):
        problems.append("the actual sparse kernel was not hit")
    if quality.get("problems"):
        problems.append("quality gate failed")
    problems.extend(cost.get("problems") or ())
    if not coverage.get("coverage"):
        problems.append("no hotspot coverage recorded")
    if reconciliation.get("explained") is False:
        problems.append("the model effect is not explained by the mediators")

    if not compliance.get("hardware_eligible"):
        decision = rec.REJECT_PORTABILITY
    elif not dispatch.get("sparse_path_hit"):
        decision = rec.REJECT_NO_BENEFIT
    elif quality.get("problems"):
        decision = rec.REJECT_QUALITY
    elif problems:
        decision = rec.RESEARCH_ONLY
    else:
        decision = rec.ADOPT_EXPERIMENTAL
    allowed: Tuple[str, ...] = ()
    if decision == rec.ADOPT_EXPERIMENTAL:
        allowed = (
            f"{route} 在预注册 shape/覆盖率 {coverage.get('coverage')} 内成立",
            "收益已扣除 metadata/conversion 成本",
        )
    elif decision == rec.REJECT_NO_BENEFIT:
        allowed = ("真实 sparse kernel 未命中或收益无法覆盖转换成本，属于可信负结论",)
    return AdoptionDecision(
        decision_id=decision_id,
        experiment_id=EXPERIMENT_ID,
        decision=decision,
        allowed_claims=allowed,
        forbidden_claims=(
            "零元素比例等于硬件稀疏",
            "2:4 与任意 50% 稀疏混淆",
            "micro speedup 直接外推模型",
            "sparse attention 不测长距离质量",
        ),
        quality_status=rec.STATUS_PASS if not quality.get("problems") else rec.STATUS_FAIL_QUALITY,
        performance_status=rec.STATUS_NOT_RUN,
        maturity=rec.MATURITY_KERNEL_PROFILED if not problems else rec.MATURITY_SOURCE_INTEGRATED,
        evidence_refs=tuple(evidence_refs),
        limitations=tuple(problems) + ("适用域必须按 shape/pattern/覆盖率限定",),
        reopened_if=("出现支持该 pattern/dtype/shape 的库或硬件时可重开",),
    )


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-F4 interfaces (labelled smoke, not an experiment)."""
    compliance = pattern_compliance({"w": [[1, 2, 0, 0], [0, 3, 4, 0]]}, n=2, m=4)
    broken = pattern_compliance({"w": [[1, 2, 3, 0]]}, n=2, m=4)
    reference = sparse_gemm_reference(
        dense_a=[[1.0, 2.0]], sparse_b_values=[[1.0], [2.0]], mask_b=[[1], [1]],
    )
    domain = SupportDomain(
        compute_capability="sm_86", library="cusparselt", library_version="0.6.2",
        dtypes=("fp16", "bf16"), min_shape=(16, 16, 16), alignment=16,
    )
    dispatch = check_actual_dispatch(
        requested="sparse_gemm", actual_kernel="cusparseLtMatmul", fallback=False,
        reason_code="AVAILABLE", evidence="nsys kernel name", pattern_compliance_value=1.0,
    )
    cost = metadata_cost(metadata_count=8, metadata_bytes_per_entry=4, value_count=64, value_bytes=2)
    amdahl = coverage_and_amdahl(
        layer_times_ms={"attn": 2.0, "mlp": 6.0, "norm": 2.0},
        sparse_kernel_speedup=2.0, eligible_layers=("mlp",),
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "compliant": compliance["hardware_eligible"],
        "noncompliant_rejected": not broken["hardware_eligible"],
        "reference_result": reference["result"],
        "unsupported_reason": domain.supports(M=8, N=64, K=64, dtype="fp16")["reason_code"],
        "sparse_path_hit": dispatch["sparse_path_hit"],
        "metadata_ratio": round(cost["metadata_ratio"], 4),
        "coverage": amdahl["coverage"],
        "predicted_model_speedup": round(amdahl["predicted_model_speedup"], 4),
        "routes": list(ROUTES),
    }


# ── result accessors ───────────────────────────────────────────────────────

def dispatch_fell_back(result: Mapping[str, Any]) -> bool:
    """Step 29: whether the dynamic workload actually reached the sparse kernel."""
    return bool(result.get("fallback"))


def compliance_is_eligible(result: Mapping[str, Any]) -> bool:
    """Step 14: the artefact may use the hardware sparse path at all."""
    return bool(result.get("hardware_eligible"))


# ── protocol step table (40 steps of details/S14/E14-F4) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "绑定 E14-05 协议", ("frontier:issue_contract", "frontier:contract_hash")),
    (2, "冻结 dense ModelArtifact", ("contracts:ServingModelArtifact", "sparsity:SparseArtifactSchema.source_artifact_id")),
    (3, "冻结 sparse transform", ("sparsity:SparseTransform", "sparsity:ROUTES")),
    (4, "冻结 sparse artifact schema", ("sparsity:SparseArtifactSchema", "sparsity:SPARSE_DTYPES")),
    (5, "冻结硬件/runtime 支持域", ("sparsity:SupportDomain", "sparsity:SupportDomain.supports")),
    (6, "冻结 dense baseline", ("frontier:BaselinePair", "sparsity:SparseTransform.problems")),
    (7, "冻结质量门", ("frontier:QualityGate", "sparsity:DISTANCE_METRICS")),
    (8, "冻结 shape/workload 矩阵", ("sparsity:shape_scan", "frontier:WorkloadStrata")),
    (9, "冻结成本与摊销口径", ("sparsity:conversion_cost", "sparsity:SPARSE_COST_COMPONENTS")),
    (10, "建立理论 FLOPs/bytes/Amdahl 模型", ("sparsity:coverage_and_amdahl", "frontier:PredictionModel")),
    (11, "运行 capability probe", ("sparsity:check_actual_dispatch", "sparsity:SupportDomain.supports")),
    (12, "建立 dense correctness/performance baseline", ("sparsity:sparse_gemm_reference", "frontier:BaselinePair")),
    (13, "执行 sparse transform", ("sparsity:SparseTransform.retrained", "identity:content_address_aggregate")),
    (14, "验证 sparsity/pattern compliance", ("sparsity:pattern_compliance", "sparsity:WEIGHT_PATTERNS")),
    (15, "验证 compressed representation", ("sparsity:compressed_roundtrip", "sparsity:SparseArtifactSchema.alignment")),
    (16, "运行 tensor-level correctness", ("sparsity:sparse_gemm_reference", "parity:evaluate_gates")),
    (17, "分离算法误差与实现误差", ("sparsity:sparse_gemm_reference", "sparsity:SparseExecutionRecord")),
    (18, "运行特殊 shape/tail/非对齐", ("sparsity:shape_scan", "sparsity:SupportDomain.supports")),
    (19, "验证 actual sparse kernel", ("sparsity:check_actual_dispatch", "sparsity:DISPATCH_REASONS")),
    (20, "做 kernel shape 扫描", ("sparsity:shape_scan", "frontier:StatisticsPlan")),
    (21, "测 metadata/index/mask 开销", ("sparsity:metadata_cost", "sparsity:SparseExecutionRecord.metadata_bytes")),
    (22, "测 conversion/compression 开销", ("sparsity:conversion_cost", "frontier:CostDenominator")),
    (23, "运行 layer/block 集成", ("sparsity:coverage_and_amdahl", "parity:evaluate_gates")),
    (24, "扫描覆盖层选择", ("sparsity:coverage_and_amdahl", "sparsity:runtime_dynamic_workload")),
    (25, "运行 full-model correctness", ("parity:evaluate_gates", "sparsity:SparseExecutionRecord.correctness_status")),
    (26, "运行任务质量门", ("sparsity:attention_quality_by_distance", "contracts:check_quality_before_performance")),
    (27, "运行 model-core 性能", ("sparsity:coverage_and_amdahl", "records:PROFILE_LAYER_FIELDS")),
    (28, "运行编译/graph/cache 测试", ("sparsity:compile_cache_check", "records:REASON_CODES")),
    (29, "运行 Runtime 动态 workload", ("sparsity:runtime_dynamic_workload", "sparsity:dispatch_fell_back")),
    (30, "运行 Service 到达率曲线", ("sparsity:service_curve", "records:PROFILE_LAYERS")),
    (31, "测内存和容量", ("sparsity:memory_and_capacity", "contracts:check_resource_ledger")),
    (32, "测能耗和成本", ("sparsity:energy_and_cost", "frontier:CostDenominator")),
    (33, "测试 unsupported capability/shape", ("sparsity:unsupported_capability", "sparsity:DISPATCH_REASONS")),
    (34, "注入损坏 metadata/index/mask", ("sparsity:corrupted_artifact_check", "campaign:isolation_clause")),
    (35, "测试版本/engine mismatch", ("sparsity:version_mismatch_check", "records:STATUS_INVALID_IDENTITY")),
    (36, "执行故障后的 fallback/recovery", ("sparsity:fallback_policy", "records:STATUS_FAIL_RECOVERY")),
    (37, "对账预测与实测", ("sparsity:reconcile_prediction", "sparsity:coverage_and_amdahl")),
    (38, "在 holdout shape/workload 确认", ("frontier:WorkloadStrata.holdout_id", "sparsity:shape_scan")),
    (39, "跨 run/设备条件重复", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan.minimum_repeats")),
    (40, "形成 F4 AdoptionDecision", ("sparsity:f4_adoption", "contracts:AdoptionDecision.validate")),
)
