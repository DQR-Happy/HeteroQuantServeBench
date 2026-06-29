"""Second compiler stack (TVM Relax/TensorIR or MLIR): portable lowering.

Protocol anchor: ``details/S11/E11-09_tvm_mlir_portable_lowering.md``.
E11-09 does not migrate the project main line; it reproduces the *same*
verified semantic op in a second stack to show the concepts transfer:

    frozen semantic contract → high-level graph → legalisation → loop IR →
    schedule trace → target lowering → runtime → correctness/perf/cost

Instruments provided (all declarative — nothing here executes TVM/MLIR, so the
package stays importable on the CPU-minimal installation):

* primary stack selection with frozen versions and an explicit scope ceiling;
* the semantic mapping table (FX/HQSB ↔ TVM ↔ MLIR) with must-preserve fields
  and blockers for unmapped semantics;
* legalisation target with ``legal`` / ``dynamic_legal`` / ``illegal`` and an
  analysis-only mode so "unsupported" is discovered *before* codegen;
* loop-IR structural checks (bounds, read/write regions, reduction
  init/update, tail, type) and a replayable schedule trace;
* the bridge contract (ownership/stride/device/stream/sync/error) plus a copy
  and sync audit that refuses an unverified "zero-copy" claim;
* pass pipeline trace, generated-code/binary lineage, development-cost rubric,
  compiler-role comparison and the adoption decision with its four allowed
  conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text, version_is_frozen

# ── stack selection and scope (steps 1–3) ──────────────────────────────────

STACKS: Tuple[str, ...] = ("tvm", "mlir")

STACK_ROLES: Mapping[str, str] = {
    "tvm": "Relax (graph) + TensorIR (tensor program) + MetaSchedule + runtime",
    "mlir": "dialect conversion, legality analysis, progressive lowering, pass manager",
}

LEGALITY_MODES: Tuple[str, ...] = ("analysis_only", "partial", "full")

LEGAL_STATUSES: Tuple[str, ...] = ("legal", "dynamic_legal", "illegal", "external_fallback")

UNMAPPED_POLICIES: Tuple[str, ...] = ("block", "external_call")


@dataclass
class StackSelection:
    """Primary stack + version freeze + the consciously limited scope."""

    primary_stack: str
    versions: Mapping[str, str]
    scope: str
    secondary_stack: str = ""
    secondary_status: str = "NOT_RUN_SCOPE_LIMITED"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.primary_stack not in STACKS:
            problems.append(f"unknown primary stack {self.primary_stack!r}")
        if self.secondary_stack and self.secondary_stack not in STACKS:
            problems.append(f"unknown secondary stack {self.secondary_stack!r}")
        if not self.versions:
            problems.append("stack versions must be frozen")
        for name, value in self.versions.items():
            if not version_is_frozen(value):
                problems.append(f"version {name}={value!r} is not frozen")
        if not self.scope:
            problems.append("a scope ceiling is mandatory (P0 is one operator/subgraph loop)")
        if self.primary_stack == "tvm" and "relax" not in " ".join(self.versions):
            # not fatal, but the version set should name the components it locks
            problems.append("TVM selection should lock relax/tir components explicitly")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "primary_stack": self.primary_stack,
            "secondary_stack": self.secondary_stack,
            "secondary_status": self.secondary_status,
            "versions": dict(sorted(self.versions.items())),
            "scope": self.scope,
            "roles": dict(sorted(STACK_ROLES.items())),
        }


SCOPE_CEILING = (
    "P0: one real operator/subgraph runnable loop (no full Qwen importer, no production "
    "runtime); the second stack never replaces the FX/Inductor main line without evidence"
)


# ── semantic mapping (steps 4, 31) ─────────────────────────────────────────

REQUIRED_MAPPING_FIELDS: Tuple[str, ...] = (
    "hqsb_or_fx",
    "tvm",
    "mlir",
    "must_preserve",
)

MAPPING_ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("semantic fused op", "Relax call/composite", "custom/linalg/func op", "schema/version/effects"),
    ("tensor type", "TensorStructInfo/buffer", "ranked tensor/memref", "shape/dtype/layout"),
    ("symbolic shape", "symbolic vars/constraints", "dynamic dims + attrs/asserts",
     "range/relations/guards"),
    ("reduction", "TensorIR block/reduce axis", "linalg/scf/vector reduction",
     "accumulation/order"),
    ("alias/mutation", "explicit buffer/call semantics", "memory effects/memref",
     "ownership/write order"),
    ("target capability", "Target/pass policy", "ConversionTarget/attrs", "arch/features"),
    ("fallback", "external packed/PyTorch path", "partial conversion/runtime call",
     "source/provenance"),
)


def semantic_mapping_table(
    stack: str, *, unmapped: Sequence[str] = (), unmapped_policy: str = "block"
) -> Dict[str, Any]:
    """Every row must map and state what must be preserved; unmapped ⇒ blocker."""
    if stack not in STACKS:
        raise ConfigError(f"unknown stack {stack!r}")
    if unmapped_policy not in UNMAPPED_POLICIES:
        raise ConfigError(f"unknown unmapped policy {unmapped_policy!r}")
    rows = [
        {
            "hqsb_or_fx": row[0],
            "tvm": row[1],
            "mlir": row[2],
            "must_preserve": row[3],
            "primary_mapping": row[1] if stack == "tvm" else row[2],
        }
        for row in MAPPING_ROWS
    ]
    blockers = []
    for item in unmapped:
        blockers.append(
            {
                "semantic": item,
                "action": "block" if unmapped_policy == "block" else "explicit external call",
                "reason": "unmapped semantics silently disappearing is a correctness bug",
            }
        )
    return {
        "stack": stack,
        "rows": rows,
        "unmapped": blockers,
        "ok": not (blockers and unmapped_policy == "block"),
        "rule": "unknown ops are never automatically legal; they are blocked or called externally",
    }


# ── legalisation (steps 9–13) ──────────────────────────────────────────────


@dataclass
class LegalizationTarget:
    """Legal / dynamic-legal / illegal op sets for a conversion."""

    legal_ops: Tuple[str, ...]
    dynamic_legal_ops: Tuple[str, ...] = ()
    illegal_ops: Tuple[str, ...] = ()
    external_calls: Tuple[str, ...] = ()
    mode: str = "analysis_only"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.mode not in LEGALITY_MODES:
            problems.append(f"unknown legality mode {self.mode!r}")
        overlap = set(self.legal_ops) & set(self.illegal_ops)
        if overlap:
            problems.append(f"ops both legal and illegal: {sorted(overlap)}")
        if not self.legal_ops:
            problems.append("a legalisation target needs a non-empty legal set")
        return problems

    def classify(self, op: str) -> str:
        if op in self.legal_ops:
            return "legal"
        if op in self.dynamic_legal_ops:
            return "dynamic_legal"
        if op in self.external_calls:
            return "external_fallback"
        return "illegal"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "legal_ops": list(self.legal_ops),
            "dynamic_legal_ops": list(self.dynamic_legal_ops),
            "illegal_ops": list(self.illegal_ops),
            "external_calls": list(self.external_calls),
            "mode": self.mode,
        }


def analysis_only_legality(module_ops: Sequence[str], target: LegalizationTarget) -> Dict[str, Any]:
    """Discover unsupported ops *before* codegen (step 10)."""
    problems = target.validate()
    rows = [
        {"op": op, "status": target.classify(op), "reason": "" if target.classify(op) != "illegal" else "NO_LEGAL_LOWERING"}
        for op in module_ops
    ]
    illegal = [row["op"] for row in rows if row["status"] == "illegal"]
    return {
        "rows": rows,
        "illegal": illegal,
        "analysis_ok": not illegal and not problems,
        "problems": problems,
        "rule": (
            "legalisation (can it run?) and optimisation (which schedule?) are different "
            "questions: confuse them and a schedule failure is reported as an unsupported op"
        ),
    }


# ── loop IR, schedule and passes (steps 12–19) ─────────────────────────────


@dataclass
class LoopIRSpec:
    """Minimal loop/block IR: buffers, loops, bounds, reduction, tail."""

    block_id: str
    buffers: Tuple[str, ...]
    loops: Tuple[Tuple[str, int, int], ...]  # (var, extent, step)
    reads: Mapping[str, Tuple[str, ...]]
    writes: Mapping[str, Tuple[str, ...]]
    reduction: Mapping[str, Any] = field(default_factory=dict)
    type_check: str = "elemwise"
    tail_policy: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.buffers:
            problems.append("loop IR without buffers")
        for var, extent, step in self.loops:
            if extent <= 0 or step <= 0:
                problems.append(f"loop {var}: non-positive extent/step")
        for name in self.reads:
            if name not in self.buffers:
                problems.append(f"read region references unknown buffer {name!r}")
        for name in self.writes:
            if name not in self.buffers:
                problems.append(f"write region references unknown buffer {name!r}")
        if self.reduction:
            for key in ("axis", "init", "update"):
                if key not in self.reduction:
                    problems.append(f"reduction missing {key!r}")
        if not self.tail_policy:
            problems.append(
                "tail policy missing: a schedule that assumes divisibility must say what happens "
                "at the boundary"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "buffers": list(self.buffers),
            "loops": [list(item) for item in self.loops],
            "reads": {key: list(value) for key, value in sorted(self.reads.items())},
            "writes": {key: list(value) for key, value in sorted(self.writes.items())},
            "reduction": dict(sorted(self.reduction.items())),
            "type_check": self.type_check,
            "tail_policy": self.tail_policy,
        }


def verify_loop_ir(spec: LoopIRSpec) -> Dict[str, Any]:
    problems = spec.validate()
    return {
        "block_id": spec.block_id,
        "ok": not problems,
        "problems": problems,
        "checks": [
            "bounds",
            "read_write_regions",
            "reduction_init_update",
            "tail",
            "types",
        ],
    }


SCHEDULE_STEPS: Tuple[str, ...] = (
    "split",
    "tile",
    "reorder",
    "fuse",
    "cache_read",
    "cache_write",
    "vectorize",
    "thread_binding",
    "unroll",
    "reduction_strategy",
)


@dataclass
class ScheduleStep:
    kind: str
    params: Mapping[str, Any]
    hardware_reason: str
    before_hash: str = ""
    after_hash: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.kind not in SCHEDULE_STEPS:
            problems.append(f"unknown schedule step {self.kind!r}")
        if not self.hardware_reason:
            problems.append(f"schedule step {self.kind!r} must state its hardware reason")
        if self.before_hash and self.after_hash and self.before_hash == self.after_hash:
            problems.append(f"schedule step {self.kind!r} did not change the IR hash")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "params": dict(sorted(self.params.items())),
            "hardware_reason": self.hardware_reason,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
        }


@dataclass
class ScheduleTrace:
    """Ordered schedule transformations; replayable in a new process."""

    trace_id: str
    steps: Tuple[ScheduleStep, ...]
    seed: int = 0

    def validate(self) -> List[str]:
        problems = [problem for step in self.steps for problem in step.validate()]
        if not self.steps:
            problems.append("a schedule trace without steps cannot be replayed")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "seed": self.seed,
            "steps": [step.as_dict() for step in self.steps],
            "digest": sha256_text(canonical_json([step.as_dict() for step in self.steps])),
        }


def replay_schedule(trace: ScheduleTrace, *, apply: Any = None) -> Dict[str, Any]:
    """Replay a schedule trace; without an executor this only checks structure."""
    problems = trace.validate()
    if apply is None:
        return {
            "trace_id": trace.trace_id,
            "structural_ok": not problems,
            "problems": problems,
            "status": "STRUCTURE_ONLY",
            "note": "a real replay needs the stack executor in the environment",
        }
    hashes: List[str] = []
    for step in trace.steps:
        hashes.append(str(apply(step)))
    return {
        "trace_id": trace.trace_id,
        "replayed_hashes": hashes,
        "structural_ok": not problems,
        "problems": problems,
        "status": "REPLAYED",
    }


@dataclass
class PassTraceEntry:
    name: str
    options: Mapping[str, Any]
    before_hash: str
    after_hash: str
    statistics: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pass": self.name,
            "options": dict(sorted(self.options.items())),
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "statistics": dict(sorted(self.statistics.items())),
            "status": self.status,
        }


def pass_pipeline_trace(entries: Sequence[PassTraceEntry]) -> Dict[str, Any]:
    broken = [entry.name for entry in entries if entry.status != "ok"]
    chained = all(
        left.after_hash == right.before_hash
        for left, right in zip(entries, entries[1:])
    )
    return {
        "entries": [entry.as_dict() for entry in entries],
        "failed_passes": broken,
        "hash_chain_consistent": chained,
        "ok": not broken and chained and bool(entries),
        "rule": "saving only the final code loses which pass produced what",
    }


# ── bridge contract (steps 21–22) ──────────────────────────────────────────

BRIDGE_KINDS: Tuple[str, ...] = ("dlpack", "packed_func_c_abi", "torch_library", "file_exchange")


@dataclass
class BridgeContract:
    kind: str
    ownership: str
    stride_policy: str
    device_policy: str
    stream_policy: str
    sync_policy: str
    error_policy: str
    zero_copy_claimed: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.kind not in BRIDGE_KINDS:
            problems.append(f"unknown bridge kind {self.kind!r}")
        for name in (
            "ownership",
            "stride_policy",
            "device_policy",
            "stream_policy",
            "sync_policy",
            "error_policy",
        ):
            if not getattr(self, name):
                problems.append(f"bridge contract missing {name!r}")
        if self.zero_copy_claimed and self.stride_policy.startswith("copy_to_contiguous"):
            problems.append("zero-copy claimed while the bridge copies to contiguous memory")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ownership": self.ownership,
            "stride_policy": self.stride_policy,
            "device_policy": self.device_policy,
            "stream_policy": self.stream_policy,
            "sync_policy": self.sync_policy,
            "error_policy": self.error_policy,
            "zero_copy_claimed": self.zero_copy_claimed,
        }


def bridge_audit_plan(contract: BridgeContract) -> Dict[str, Any]:
    problems = contract.validate()
    return {
        "contract": contract.as_dict(),
        "measurements": [
            "device→device copy bytes (if any)",
            "host staging bytes and time",
            "device synchronizations per call",
            "stream identity of the callable vs the caller",
        ],
        "claim_rule": (
            "a zero-copy/zero-sync claim requires timeline/counter evidence; the interface name "
            "proves nothing"
        ),
        "problems": problems,
        "ok": not problems,
    }


def bridge_overhead(
    *, end_to_end_ms: Optional[float], device_kernel_ms: Optional[float]
) -> Dict[str, Any]:
    if end_to_end_ms is None or device_kernel_ms is None:
        return {
            "bridge_overhead_ms": None,
            "status": "UNAVAILABLE",
            "reason": "copy/sync accounting requires a timeline; missing evidence is not zero",
        }
    return {
        "bridge_overhead_ms": round(end_to_end_ms - device_kernel_ms, 6),
        "status": "MEASURED",
        "note": "kernel-only speedups must be reported next to the integrated subgraph",
    }


# ── development cost and role comparison (steps 30–32) ─────────────────────

COST_RUBRIC_ITEMS: Tuple[str, ...] = (
    "implementation_loc",
    "integration_effort",
    "debugging_hours",
    "build_and_dependency_weight",
    "artifact_footprint_bytes",
    "replay_success",
    "maintenance_risk",
)


def development_cost_rubric(
    *,
    implementation_loc: int,
    integration_effort: str,
    debugging_hours: float,
    dependency_weight: str,
    artifact_footprint_bytes: int,
    replay_success: bool,
    maintenance_risk: str,
) -> Dict[str, Any]:
    payload = {
        "implementation_loc": implementation_loc,
        "integration_effort": integration_effort,
        "debugging_hours": debugging_hours,
        "build_and_dependency_weight": dependency_weight,
        "artifact_footprint_bytes": artifact_footprint_bytes,
        "replay_success": replay_success,
        "maintenance_risk": maintenance_risk,
    }
    return {
        "rubric": payload,
        "complete": all(value not in ("", None) for value in payload.values()),
        "rule": "engineering effort is reported with its own rubric, never folded into latency",
    }


ROLE_DIMENSIONS: Tuple[str, ...] = (
    "capture",
    "IR",
    "legality",
    "schedule",
    "codegen",
    "dynamic_shape",
    "runtime",
    "cache",
)


def role_comparison(
    *, hqsb: Mapping[str, str], second_stack: Mapping[str, str]
) -> Dict[str, Any]:
    missing = [name for name in ROLE_DIMENSIONS if name not in hqsb or name not in second_stack]
    rows = [
        {
            "dimension": name,
            "hqsb_fx_inductor": hqsb.get(name, ""),
            "second_stack": second_stack.get(name, ""),
        }
        for name in ROLE_DIMENSIONS
    ]
    return {
        "rows": rows,
        "missing": missing,
        "complete": not missing,
        "note": "the comparison is about who does what per stage, not which stack is 'better'",
    }


ADOPTION_OPTIONS: Tuple[str, ...] = (
    "second_stack_as_target_backend",
    "schedule_oracle_offline",
    "concept_validation_only",
    "insufficient_data",
)


def adoption_decision(
    *,
    option: str,
    correctness_ok: bool,
    performance_evidence: bool,
    dev_cost_rubric: Mapping[str, Any],
    reasons: Sequence[str] = (),
) -> Dict[str, Any]:
    if option not in ADOPTION_OPTIONS:
        raise ConfigError(f"unknown adoption option {option!r}")
    blockers: List[str] = []
    if not correctness_ok:
        blockers.append("correctness not established")
    if option == "second_stack_as_target_backend" and not performance_evidence:
        blockers.append("no performance evidence for a production backend")
    return {
        "option": option,
        "reasons": list(reasons),
        "blockers": blockers,
        "decided": not blockers,
        "allowed_public_conclusions": [
            "verified concept transfer for a target/research backend",
            "schedule/search usable as an offline oracle while the main line stays FX/Inductor",
            "integration/compile cost not worth it: keep the concept validation",
            "insufficient data to compare",
        ],
        "rule": (
            "a runnable operator alone never implies a general MLIR/TVM compiler or portable "
            "performance"
        ),
    }


def negative_capability_case(*, kind: str, detail: str) -> Dict[str, Any]:
    return {
        "kind": kind,
        "detail": detail,
        "action": "explicit failure or fallback",
        "rule": "unsupported shape/dtype/layout/target must not generate a misused binary",
    }


def concept_summary_rows() -> List[Dict[str, str]]:
    """Concept-transfer summary used by the report (not a result table)."""
    return [
        {"concept": "graph abstraction & fusion", "tvm": "Relax", "mlir": "dialect patterns"},
        {"concept": "operator legalisation", "tvm": "Relax → TIR", "mlir": "conversion patterns"},
        {"concept": "schedule", "tvm": "TensorIR schedule", "mlir": "affine/linalg transforms"},
        {"concept": "target legality", "tvm": "Target policy", "mlir": "ConversionTarget"},
        {"concept": "verifier/pass manager", "tvm": "IRModule checks", "mlir": "PassManager"},
        {"concept": "runtime bridge", "tvm": "PackedFunc/DLPack", "mlir": "custom ABI + runtime"},
    ]
