"""Lowering registry, candidate selection, dispatch evidence and fallback safety.

Protocol anchor: ``details/S11/E11-03_graph_ir_lowering_custom_kernel.md``.
This is the S11 hero path: it must distinguish four facts (§1) — pattern
matched, graph rewritten, lowering selected, kernel executed — and it must
never confuse a debug backend with a compiled path.

Implemented instruments:

* registry entries carrying semantic op/schema, implementation/build ids,
  supported dtype/layout/shape/arch/features, alignment/workspace/stream
  semantics, mutation contract, guard builder, artifact locator, evidence
  scope, priority, feature adapter and fallback (§10, steps 9–11);
* a selection pipeline that emits the *full* candidate table with a structured
  reject reason per filter (steps 12–14, §3.1);
* deterministic policies ``reference`` / ``forced:<id>`` / ``auto_heuristic``
  where a forced-but-unsupported candidate fails loudly instead of silently
  switching (step 14, §12);
* materialisation of the selected candidate plus the always-correct reference
  fallback (steps 10, 15);
* runtime dispatch telemetry with requested/eligible/selected/actual/fallback
  (step 16) and a two-independent-evidence confirmation rule (§3.5, step 24);
* pre-launch side-effect ordering and post-failure policy (§3.4, step 30);
* compile-time breakdown separated from steady state (§14, step 26).

Nothing here imports ``ops``: kernels are addressed by capability/provider
names and artifact hashes (module ownership rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text
from hqsb.compiler.records import (
    COMPILER_COST_KEYS,
    STATUS_FAIL_COMPILE,
    STATUS_FAIL_LOWERING,
    STATUS_PASS,
)
from hqsb.compiler.targets import (
    CapabilityOutcome,
    CapabilityRequirement,
    TargetSnapshot,
)

# ── evidence index ─────────────────────────────────────────────────────────

EVIDENCE_LEVELS: Tuple[str, ...] = ("operator", "block", "model", "fallback")


@dataclass
class CorrectnessEvidence:
    """Correctness evidence covering a shape/dtype/arch scope (steps 11, 20)."""

    evidence_id: str
    level: str
    implementation_id: str
    dtypes: Tuple[str, ...]
    shapes: Tuple[str, ...] = ()
    archs: Tuple[str, ...] = ()
    tolerance_policy_id: str = ""
    status: str = "not_run"
    raw_ref: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.level not in EVIDENCE_LEVELS:
            problems.append(f"unknown evidence level {self.level!r}")
        if self.status not in ("pass", "fail", "not_run"):
            problems.append(f"unknown evidence status {self.status!r}")
        if self.status == "pass" and not self.raw_ref:
            problems.append(f"{self.evidence_id}: a passing evidence row needs a raw reference")
        if not self.tolerance_policy_id:
            problems.append(f"{self.evidence_id}: tolerance policy must be named (no ad-hoc tolerance)")
        if not self.dtypes:
            problems.append(f"{self.evidence_id}: evidence must declare the dtype scope")
        return problems

    def covers(self, *, dtype: str, shape: str, arch: str) -> bool:
        if self.status != "pass":
            return False
        if self.dtypes and dtype not in self.dtypes:
            return False
        if self.shapes and shape not in self.shapes and "*" not in self.shapes:
            return False
        if self.archs and arch not in self.archs and "*" not in self.archs:
            return False
        return True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "level": self.level,
            "implementation_id": self.implementation_id,
            "dtypes": list(self.dtypes),
            "shapes": list(self.shapes),
            "archs": list(self.archs),
            "tolerance_policy_id": self.tolerance_policy_id,
            "status": self.status,
            "raw_ref": self.raw_ref,
        }


class EvidenceIndex:
    """Lookup: does a candidate have correctness evidence for *this* point?"""

    def __init__(self, rows: Iterable[CorrectnessEvidence] = ()) -> None:
        self._rows: List[CorrectnessEvidence] = []
        for row in rows:
            self.register(row)

    def register(self, row: CorrectnessEvidence) -> None:
        problems = row.validate()
        if problems:
            raise ConfigError("invalid evidence row: " + "; ".join(problems))
        self._rows.append(row)

    def rows(self) -> List[CorrectnessEvidence]:
        return list(self._rows)

    def lookups(
        self, implementation_id: str, *, dtype: str, shape: str, arch: str, level: str
    ) -> Dict[str, Any]:
        matches = [
            row
            for row in self._rows
            if row.implementation_id == implementation_id
            and row.level == level
            and row.covers(dtype=dtype, shape=shape, arch=arch)
        ]
        return {
            "implementation_id": implementation_id,
            "level": level,
            "covered": bool(matches),
            "evidence_ids": [row.evidence_id for row in matches],
            "reason": "" if matches else "EVIDENCE_MISSING",
        }

    def covered(
        self,
        implementation_id: str,
        *,
        dtype: str,
        shape: str,
        arch: str,
        required_levels: Sequence[str] = ("operator",),
    ) -> bool:
        return all(
            self.lookups(implementation_id, dtype=dtype, shape=shape, arch=arch, level=level)[
                "covered"
            ]
            for level in required_levels
        )


# ── registry (E11-03 steps 9–11) ───────────────────────────────────────────

EXECUTION_KINDS: Tuple[str, ...] = ("reference", "custom_kernel", "generated_kernel")


@dataclass
class LoweringEntry:
    """One lowering candidate with everything needed to prove/execute it."""

    candidate_id: str
    semantic_op: str
    schema_version: str
    execution_kind: str
    implementation_id: str
    build_id: str = ""
    backend: str = "cuda"
    requirement: Optional[CapabilityRequirement] = None
    artifact_locator: str = ""
    artifact_hash: str = ""
    signature: str = ""
    mutation_contract: str = "functional"
    stream_semantics: str = "current_stream"
    workspace_bytes: int = 0
    guard_ids: Tuple[str, ...] = ()
    evidence_scope: Tuple[str, ...] = ("operator",)
    performance_scope: str = ""
    priority: int = 0
    feature_adapter: str = ""
    fallback_id: str = ""
    notes: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("candidate_id", "semantic_op", "schema_version", "implementation_id"):
            if not getattr(self, name):
                problems.append(f"lowering entry missing {name!r}")
        if self.execution_kind not in EXECUTION_KINDS:
            problems.append(f"unknown execution kind {self.execution_kind!r}")
        if self.requirement is None:
            problems.append(f"{self.candidate_id}: capability requirement is mandatory")
        else:
            problems.extend(self.requirement.validate())
        if self.mutation_contract not in ("functional", "inplace", "stateful"):
            problems.append(f"{self.candidate_id}: unknown mutation contract {self.mutation_contract!r}")
        if self.stream_semantics not in ("current_stream", "own_stream", "default_stream"):
            problems.append(f"{self.candidate_id}: unknown stream semantics {self.stream_semantics!r}")
        if self.execution_kind in ("custom_kernel", "generated_kernel"):
            if not (self.artifact_locator and self.artifact_hash):
                problems.append(
                    f"{self.candidate_id}: a compiled candidate must carry an artifact "
                    "locator and hash (no anonymous pointers)"
                )
            if not self.build_id:
                problems.append(f"{self.candidate_id}: compiled candidate needs a build id")
        if not self.evidence_scope:
            problems.append(f"{self.candidate_id}: evidence scope must be declared")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "semantic_op": self.semantic_op,
            "schema_version": self.schema_version,
            "execution_kind": self.execution_kind,
            "implementation_id": self.implementation_id,
            "build_id": self.build_id,
            "backend": self.backend,
            "requirement": self.requirement.as_dict() if self.requirement else None,
            "artifact_locator": self.artifact_locator,
            "artifact_hash": self.artifact_hash,
            "signature": self.signature,
            "mutation_contract": self.mutation_contract,
            "stream_semantics": self.stream_semantics,
            "workspace_bytes": self.workspace_bytes,
            "guard_ids": list(self.guard_ids),
            "evidence_scope": list(self.evidence_scope),
            "performance_scope": self.performance_scope,
            "priority": self.priority,
            "feature_adapter": self.feature_adapter,
            "fallback_id": self.fallback_id,
            "notes": self.notes,
        }

    def identity_digest(self) -> str:
        payload = self.as_dict()
        payload.pop("notes", None)
        return sha256_text(canonical_json(payload))


class LoweringRegistry:
    """Registry of lowering candidates; registration compiles nothing."""

    def __init__(self) -> None:
        self._entries: Dict[str, LoweringEntry] = {}

    def register(self, entry: LoweringEntry) -> None:
        problems = entry.validate()
        if problems:
            raise ConfigError("invalid lowering entry: " + "; ".join(problems))
        if entry.candidate_id in self._entries:
            raise ConfigError(f"duplicate candidate id {entry.candidate_id!r}")
        if entry.fallback_id and entry.fallback_id not in self._entries and entry.execution_kind != "reference":
            # allow forward declaration only for the reference entry
            raise ConfigError(
                f"{entry.candidate_id}: fallback {entry.fallback_id!r} must be registered before use"
            )
        self._entries[entry.candidate_id] = entry

    def get(self, candidate_id: str) -> LoweringEntry:
        try:
            return self._entries[candidate_id]
        except KeyError as exc:
            raise ConfigError(f"unknown candidate id {candidate_id!r}") from exc

    def entries_for(self, semantic_op: str, schema_version: str = "") -> List[LoweringEntry]:
        rows = [
            entry
            for entry in self._entries.values()
            if entry.semantic_op == semantic_op
            and (not schema_version or entry.schema_version == schema_version)
        ]
        return sorted(rows, key=lambda item: (-item.priority, item.candidate_id))

    def snapshot(self) -> Dict[str, Any]:
        rows = [self._entries[key].as_dict() for key in sorted(self._entries)]
        return {
            "candidates": rows,
            "count": len(rows),
            "digest": sha256_text(canonical_json(rows)),
            "note": "registration is side-effect free: no compile, no device context, no download",
        }


def reference_lowering_entry(
    *, candidate_id: str = "reference", semantic_op: str = "hqsb::fused_add_rms_norm",
    fallback_id: str = "",
) -> LoweringEntry:
    """The always-available reference lowering (steps 10 and 19).

    The reference path composes framework-level operations, so it declares no
    device/dtype restriction of its own: whatever dtype/layout the input has,
    the reference is defined by the semantic contract (and its evidence row).
    """
    return LoweringEntry(
        candidate_id=candidate_id,
        semantic_op=semantic_op,
        schema_version="1.0.0",
        execution_kind="reference",
        implementation_id="hqsb.reference.fused_add_rms_norm",
        backend="reference",
        requirement=CapabilityRequirement(
            requirement_id=f"{candidate_id}:capability",
            backends=(),
            dtypes=(),
            evidence_scope="reference_selected_tensors",
            reason_on_failure="TARGET_UNSUPPORTED",
        ),
        mutation_contract="functional",
        stream_semantics="current_stream",
        guard_ids=(),
        evidence_scope=("operator",),
        priority=0,
        fallback_id=fallback_id,
        notes="always semantically correct; may be slower than a kernel",
    )


def custom_kernel_entry(
    *,
    candidate_id: str,
    implementation_id: str,
    build_id: str,
    artifact_locator: str,
    artifact_hash: str,
    archs: Sequence[str],
    dtypes: Sequence[str] = ("fp16",),
    features: Sequence[str] = (),
    shared_memory_bytes: int = 0,
    registers_per_thread: int = 0,
    workspace_bytes: int = 0,
    alignment_bytes: int = 0,
    evidence_scope: Sequence[str] = ("operator", "block"),
    priority: int = 10,
    fallback_id: str = "reference",
    semantic_op: str = "hqsb::fused_add_rms_norm",
    performance_scope: str = "",
) -> LoweringEntry:
    """A compiled custom kernel candidate bound to a real artifact (step 11)."""
    return LoweringEntry(
        candidate_id=candidate_id,
        semantic_op=semantic_op,
        schema_version="1.0.0",
        execution_kind="custom_kernel",
        implementation_id=implementation_id,
        build_id=build_id,
        backend="cuda",
        requirement=CapabilityRequirement(
            requirement_id=f"{candidate_id}:capability",
            backends=("cuda",),
            dtypes=tuple(dtypes),
            archs=tuple(archs),
            features=tuple(features),
            alignment_bytes=alignment_bytes,
            shared_memory_bytes=shared_memory_bytes,
            registers_per_thread=registers_per_thread,
            workspace_bytes=workspace_bytes,
            evidence_scope=",".join(evidence_scope),
            reason_on_failure="TARGET_UNSUPPORTED",
        ),
        artifact_locator=artifact_locator,
        artifact_hash=artifact_hash,
        mutation_contract="functional",
        stream_semantics="current_stream",
        workspace_bytes=workspace_bytes,
        evidence_scope=tuple(evidence_scope),
        performance_scope=performance_scope,
        priority=priority,
        fallback_id=fallback_id,
    )


# ── selection pipeline (steps 12–14, §6) ───────────────────────────────────

POLICY_REFERENCE = "reference"
POLICY_AUTO_HEURISTIC = "auto_heuristic"
POLICY_FORCED_PREFIX = "forced:"

POLICIES: Tuple[str, ...] = (POLICY_REFERENCE, POLICY_AUTO_HEURISTIC)


@dataclass
class CandidateEvaluation:
    """The full filter chain for one candidate (§6 canonical order)."""

    candidate_id: str
    semantic_legal: bool
    capability_outcome: Optional[CapabilityOutcome]
    artifact_compatible: bool
    guard_outcome: Optional[bool]
    evidence_covered: bool
    predicted_cost: Optional[float]
    selected: bool = False
    reject_reason: str = ""
    fallback_id: str = ""

    @property
    def first_failure(self) -> str:
        if not self.semantic_legal:
            return "NO_LEGAL_LOWERING"
        if self.capability_outcome is not None and not self.capability_outcome.available:
            return self.capability_outcome.reason_code
        if not self.artifact_compatible:
            return "BINARY_INCOMPATIBLE"
        if self.guard_outcome is False:
            return "GUARD_FALSE"
        if not self.evidence_covered:
            return "EVIDENCE_MISSING"
        return self.reject_reason

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "semantic_legal": self.semantic_legal,
            "capability": self.capability_outcome.as_dict() if self.capability_outcome else None,
            "artifact_compatible": self.artifact_compatible,
            "guard_outcome": self.guard_outcome,
            "evidence_covered": self.evidence_covered,
            "predicted_cost": self.predicted_cost,
            "selected": self.selected,
            "reject_reason": self.reject_reason,
            "first_failure": self.first_failure if (self.reject_reason or not self.selected) else "",
            "fallback_id": self.fallback_id,
        }


@dataclass
class LoweringDecision:
    """The lowered decision record (E11-03 §10 field list)."""

    compile_id: str
    op_instance_id: str
    semantic_op: str
    schema_version: str
    source_ir_id: str
    targeted_ir_id: str
    target_snapshot_id: str
    candidates: Tuple[CandidateEvaluation, ...]
    selection_policy: str
    selected: str = ""
    fallback_id: str = ""
    reject_reason: str = ""
    predicted_cost: Optional[float] = None
    guard_set_id: str = ""
    materialize_status: str = "not_started"
    actual_dispatch_id: str = ""
    forced_unsupported: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("compile_id", "op_instance_id", "semantic_op", "target_snapshot_id"):
            if not getattr(self, name):
                problems.append(f"lowering decision missing {name!r}")
        if self.selected and not self.fallback_id:
            problems.append("a selected candidate must name its fallback")
        if self.forced_unsupported and self.selected:
            problems.append(
                "a forced candidate that failed capability must not be reported as selected"
            )
        if not self.candidates:
            problems.append(
                "the full candidate table is mandatory: only saving the winner cannot explain "
                "why others lost"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compile_id": self.compile_id,
            "op_instance_id": self.op_instance_id,
            "semantic_op": self.semantic_op,
            "schema_version": self.schema_version,
            "source_ir_id": self.source_ir_id,
            "targeted_ir_id": self.targeted_ir_id,
            "target_snapshot_id": self.target_snapshot_id,
            "candidates": [row.as_dict() for row in self.candidates],
            "selection_policy": self.selection_policy,
            "selected": self.selected,
            "fallback_id": self.fallback_id,
            "reject_reason": self.reject_reason,
            "predicted_cost": self.predicted_cost,
            "guard_set_id": self.guard_set_id,
            "materialize_status": self.materialize_status,
            "actual_dispatch_id": self.actual_dispatch_id,
            "forced_unsupported": self.forced_unsupported,
        }


def parse_policy(policy: str) -> Tuple[str, str]:
    """``reference`` | ``auto_heuristic`` | ``forced:<candidate_id>``."""
    if policy in POLICIES:
        return policy, ""
    if policy.startswith(POLICY_FORCED_PREFIX) and len(policy) > len(POLICY_FORCED_PREFIX):
        return POLICY_FORCED_PREFIX, policy[len(POLICY_FORCED_PREFIX):]
    raise ConfigError(
        f"unknown selection policy {policy!r}; expected one of {list(POLICIES)} or forced:<id>"
    )


def evaluate_candidates(
    *,
    registry: LoweringRegistry,
    semantic_op: str,
    schema_version: str,
    target: TargetSnapshot,
    evidence: EvidenceIndex,
    inputs: Mapping[str, Any],
    artifact_compatible: Callable[[LoweringEntry], bool] = lambda entry: True,
    guard_evaluator: Callable[[LoweringEntry, Mapping[str, Any]], bool] = lambda entry, values: True,
    dtype: str = "fp16",
    shape: str = "*",
    policy: str = POLICY_REFERENCE,
    compile_id: str = "compile",
    op_instance_id: str = "op",
    source_ir_id: str = "",
    targeted_ir_id: str = "",
    predicted_costs: Optional[Mapping[str, float]] = None,
    require_levels: Sequence[str] = ("operator",),
) -> LoweringDecision:
    """Run the full filter chain and return the decision with its candidate table."""
    policy_kind, forced_id = parse_policy(policy)
    entries = registry.entries_for(semantic_op, schema_version)
    if not entries:
        raise ConfigError(f"no lowering entry registered for {semantic_op!r}")
    rows: List[CandidateEvaluation] = []
    for entry in entries:
        capability = entry.requirement.check(target) if entry.requirement else None
        artifact_ok = (
            True
            if entry.execution_kind == "reference"
            else bool(artifact_compatible(entry) and entry.artifact_hash)
        )
        guard_true = guard_evaluator(entry, inputs)
        covered = evidence.covered(
            entry.implementation_id,
            dtype=dtype,
            shape=shape,
            arch=target.arch,
            required_levels=require_levels if entry.execution_kind != "reference" else ("operator",),
        )
        costs = dict(predicted_costs or {})
        rows.append(
            CandidateEvaluation(
                candidate_id=entry.candidate_id,
                semantic_legal=True,  # E11-02 already proved legality upstream
                capability_outcome=capability,
                artifact_compatible=artifact_ok,
                guard_outcome=guard_true,
                evidence_covered=covered,
                predicted_cost=costs.get(entry.candidate_id),
                fallback_id=entry.fallback_id,
            )
        )
    eligible = [
        row
        for row in rows
        if row.semantic_legal
        and (row.capability_outcome is None or row.capability_outcome.available)
        and row.artifact_compatible
        and row.guard_outcome is not False
        and row.evidence_covered
    ]
    forced_unsupported = False
    selected = ""
    reject_reason = ""
    if policy_kind == POLICY_FORCED_PREFIX:
        if forced_id not in {row.candidate_id for row in rows}:
            raise ConfigError(f"forced candidate {forced_id!r} is not registered for {semantic_op!r}")
        forced_row = next(row for row in rows if row.candidate_id == forced_id)
        if forced_row in eligible:
            selected = forced_id
        else:
            forced_unsupported = True
            reject_reason = f"FORCED_UNSUPPORTED:{forced_row.first_failure}"
    elif policy_kind == POLICY_REFERENCE:
        selected = "reference" if any(row.candidate_id == "reference" for row in eligible) else ""
        if not selected:
            raise ConfigError("reference lowering is not eligible; infrastructure is broken")
    else:  # auto_heuristic
        heuristic = [row for row in eligible if row.candidate_id != "reference"]
        if heuristic:
            heuristic.sort(
                key=lambda row: (
                    row.predicted_cost if row.predicted_cost is not None else float("inf"),
                    row.candidate_id,
                )
            )
            selected = heuristic[0].candidate_id
        elif any(row.candidate_id == "reference" for row in eligible):
            selected = "reference"
    if not selected and not forced_unsupported:
        reject_reason = "NO_LEGAL_LOWERING"
    selected_entry = registry.get(selected) if selected else None
    for row in rows:
        row.selected = row.candidate_id == selected
        if not row.selected and not row.reject_reason:
            row.reject_reason = "" if row.candidate_id == selected else row.first_failure
    decision = LoweringDecision(
        compile_id=compile_id,
        op_instance_id=op_instance_id,
        semantic_op=semantic_op,
        schema_version=schema_version,
        source_ir_id=source_ir_id,
        targeted_ir_id=targeted_ir_id,
        target_snapshot_id=target.target_id,
        candidates=tuple(rows),
        selection_policy=policy_kind if policy_kind != POLICY_FORCED_PREFIX else policy,
        selected=selected,
        fallback_id=selected_entry.fallback_id or "reference" if selected_entry else "",
        reject_reason=reject_reason,
        predicted_cost=(
            next((row.predicted_cost for row in rows if row.selected), None)
        ),
        guard_set_id="",
        forced_unsupported=forced_unsupported,
    )
    problems = decision.validate()
    if problems:
        raise ConfigError("invalid lowering decision: " + "; ".join(problems))
    return decision


# ── materialisation (steps 15, 18–19) ──────────────────────────────────────

MATERIALIZE_STATUSES: Tuple[str, ...] = (
    "not_started",
    "materialized",
    "fallback_only",
    "failed",
)


@dataclass
class MaterializedPlan:
    """What will actually be called, plus the fallback that must stay reachable."""

    compile_id: str
    selected: str
    fallback: str
    selected_implementation: str
    fallback_implementation: str
    status: str = "materialized"
    python_per_call_selection: bool = False
    workspace_bytes: int = 0
    stream_semantics: str = "current_stream"
    error: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.status not in MATERIALIZE_STATUSES:
            problems.append(f"unknown materialize status {self.status!r}")
        if self.status == "materialized" and not self.selected_implementation:
            problems.append("materialized plan needs an implementation")
        if not self.fallback_implementation:
            problems.append("a plan without a reachable fallback violates fail-closed design")
        if self.python_per_call_selection:
            problems.append(
                "expensive selection must not run per call; materialization must be hoisted"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compile_id": self.compile_id,
            "selected": self.selected,
            "fallback": self.fallback,
            "selected_implementation": self.selected_implementation,
            "fallback_implementation": self.fallback_implementation,
            "status": self.status,
            "python_per_call_selection": self.python_per_call_selection,
            "workspace_bytes": self.workspace_bytes,
            "stream_semantics": self.stream_semantics,
            "error": self.error,
        }


def materialize(
    decision: LoweringDecision,
    registry: LoweringRegistry,
    *,
    fallback_override: str = "",
) -> MaterializedPlan:
    """Turn a decision into a callable plan; refusal is explicit, never silent."""
    fallback_id = fallback_override or decision.fallback_id
    fallback_entry = None
    if fallback_id:
        fallback_entry = registry.get(fallback_id)
    if decision.selected:
        entry = registry.get(decision.selected)
        return MaterializedPlan(
            compile_id=decision.compile_id,
            selected=decision.selected,
            fallback=fallback_id,
            selected_implementation=entry.implementation_id,
            fallback_implementation=(
                fallback_entry.implementation_id
                if fallback_entry
                else "original_subgraph"
            ),
            workspace_bytes=entry.workspace_bytes,
            stream_semantics=entry.stream_semantics,
            status="materialized",
        )
    if decision.forced_unsupported:
        return MaterializedPlan(
            compile_id=decision.compile_id,
            selected="",
            fallback=fallback_id,
            selected_implementation="",
            fallback_implementation=(
                fallback_entry.implementation_id if fallback_entry else "original_subgraph"
            ),
            status="fallback_only",
            error=decision.reject_reason,
        )
    return MaterializedPlan(
        compile_id=decision.compile_id,
        selected="",
        fallback=fallback_id,
        selected_implementation="",
        fallback_implementation=(
            fallback_entry.implementation_id if fallback_entry else "original_subgraph"
        ),
        status="failed",
        error=decision.reject_reason or "NO_LEGAL_LOWERING",
    )


def rebuild_plan(
    registry: LoweringRegistry,
    *,
    semantic_op: str,
    selected: str,
    expected_registry_digest: str,
    expected_plan_digest: str,
) -> Dict[str, Any]:
    """Round-trip: re-materialise from the registry snapshot in a new process."""
    snapshot = registry.snapshot()
    entry = registry.get(selected)
    plan = {
        "semantic_op": semantic_op,
        "selected": selected,
        "implementation_id": entry.implementation_id,
        "artifact_hash": entry.artifact_hash,
        "registry_digest": snapshot["digest"],
    }
    plan_digest = sha256_text(canonical_json(plan))
    return {
        "plan": plan,
        "plan_digest": plan_digest,
        "registry_matches": snapshot["digest"] == expected_registry_digest,
        "plan_matches": plan_digest == expected_plan_digest,
        "ok": snapshot["digest"] == expected_registry_digest
        and plan_digest == expected_plan_digest,
    }


# ── dispatch telemetry + actual-dispatch evidence (steps 16, 24) ───────────

EVIDENCE_KINDS: Tuple[str, ...] = (
    "compiled_code_symbol",
    "profiler_kernel_symbol",
    "kernel_telemetry_counter",
    "controlled_fail_build",
    "candidate_disable_ablation",
)

MIN_EVIDENCE_KINDS = 2


@dataclass
class DispatchRow:
    """requested/eligible/selected/actual + fallback (handbook §5.7)."""

    dispatch_id: str
    compile_id: str
    variant_id: str
    requested: str
    eligible: Tuple[str, ...]
    selected: str
    actual: str
    actual_candidate_id: str = ""
    fallback_reason: str = ""
    artifact_hash: str = ""
    guard_outcome: str = ""
    trace_id: str = ""
    latency_us: Optional[float] = None

    def validate(self) -> List[str]:
        problems: List[str] = []
        if (
            self.actual_candidate_id
            and self.selected
            and self.actual_candidate_id != self.selected
            and not self.fallback_reason
        ):
            problems.append(
                f"{self.dispatch_id}: actual candidate != selected without a fallback reason "
                "(silent fallback)"
            )
        if self.selected and self.selected not in self.eligible and self.selected != "reference":
            problems.append(f"{self.dispatch_id}: selected candidate is not eligible")
        if not self.actual:
            problems.append(f"{self.dispatch_id}: actual implementation is not recorded")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "compile_id": self.compile_id,
            "variant_id": self.variant_id,
            "requested": self.requested,
            "eligible": list(self.eligible),
            "selected": self.selected,
            "actual": self.actual,
            "actual_candidate_id": self.actual_candidate_id,
            "fallback_reason": self.fallback_reason,
            "artifact_hash": self.artifact_hash,
            "guard_outcome": self.guard_outcome,
            "trace_id": self.trace_id,
            "latency_us": self.latency_us,
        }


class DispatchTelemetry:
    """Append-only dispatch log with C6/C7 linkage."""

    def __init__(self) -> None:
        self._rows: List[DispatchRow] = []

    def record(self, row: DispatchRow) -> None:
        problems = row.validate()
        if problems:
            raise ConfigError("invalid dispatch row: " + "; ".join(problems))
        self._rows.append(row)

    def rows(self) -> List[DispatchRow]:
        return list(self._rows)

    def summary(self) -> Dict[str, Any]:
        actual_custom = sum(1 for row in self._rows if row.actual.startswith("hqsb."))
        fallbacks = [row for row in self._rows if row.fallback_reason]
        return {
            "dispatches": len(self._rows),
            "custom_actual": actual_custom,
            "fallback_count": len(fallbacks),
            "fallback_rate": round(len(fallbacks) / max(1, len(self._rows)), 4),
            "fallback_reasons": sorted({row.fallback_reason for row in fallbacks}),
            "note": "fallback rows stay in the table; removing them would hide capability limits",
        }


@dataclass
class ActualDispatchEvidence:
    """At least two independent evidence kinds confirm the kernel really ran."""

    dispatch_id: str
    kinds: Mapping[str, str] = field(default_factory=dict)
    artifact_hash: str = ""

    def confirm(self) -> Dict[str, Any]:
        present = {key: value for key, value in self.kinds.items() if value}
        unknown = sorted(set(present) - set(EVIDENCE_KINDS))
        enough = len(present) >= MIN_EVIDENCE_KINDS
        return {
            "dispatch_id": self.dispatch_id,
            "kinds": dict(sorted(present.items())),
            "unknown_kinds": unknown,
            "enough_evidence": enough and not unknown,
            "artifact_hash": self.artifact_hash,
            "rule": (
                "a Python log line 'selected=cuda_v2' is not evidence: the implementation may "
                "still have fallen back"
            ),
        }


def disable_ablation_plan(candidate_id: str) -> Dict[str, Any]:
    """Ablation: disabling the candidate must change trace/latency/kernel count."""
    return {
        "action": f"disable {candidate_id}",
        "expect": {
            "trace": "target kernel symbol disappears",
            "kernel_count": "increases by the number of fused launches",
            "latency": "moves toward the reference path",
            "output": "still correct (fallback path)",
        },
        "note": "a change in only one of these signals is not confirmation",
    }


# ── side-effect safety (E11-03 §3.4, step 30) ──────────────────────────────

PRE_LAUNCH_CHECKS: Tuple[str, ...] = (
    "capability",
    "artifact_hash",
    "abi",
    "guard",
    "workspace",
    "stream",
    "evidence_coverage",
)

RUNTIME_ERROR_POLICIES: Tuple[str, ...] = (
    "FAIL_REQUEST",
    "RETRY_SAFE_ONLY",
)


@dataclass
class SideEffectGate:
    """Decide fallback *before* any side effect; fail requests otherwise."""

    mutates_state: bool
    transactional: bool = False

    def plan(self, candidate_id: str) -> Dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "pre_launch_checks": list(PRE_LAUNCH_CHECKS),
            "order": "checks → launch → commit (never launch → discover unsupported → rerun)",
            "mutates_state": self.mutates_state,
            "transactional": self.transactional,
        }

    def on_runtime_error(self) -> Dict[str, Any]:
        if self.mutates_state and not self.transactional:
            return {
                "policy": "FAIL_REQUEST",
                "reason": (
                    "a state-mutating kernel may already have written residual/KV; re-running the "
                    "reference path would double-apply the update"
                ),
                "retry_allowed": False,
            }
        return {
            "policy": "RETRY_SAFE_ONLY",
            "reason": "no mutation (or transactional scratch/commit) proven: fallback is safe",
            "retry_allowed": True,
        }


def preflight_checks(
    decision: LoweringDecision,
    *,
    guard_true: bool,
    artifact_hash_matches: bool,
    evidence_covered: bool,
    workspace_fits: bool,
    stream_matches: bool,
) -> Dict[str, Any]:
    """The pre-launch checklist result (any failure ⇒ fallback before effects)."""
    checks = {
        "capability": decision.selected != ""
        and all(
            row.capability_outcome is None or row.capability_outcome.available
            for row in decision.candidates
            if row.selected
        ),
        "artifact_hash": artifact_hash_matches,
        "abi": True,
        "guard": guard_true,
        "workspace": workspace_fits,
        "stream": stream_matches,
        "evidence_coverage": evidence_covered,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    return {
        "checks": checks,
        "failed": failed,
        "proceed": not failed,
        "action": "launch" if not failed else "fallback_before_launch",
    }


# ── failure injection (steps 29–30) ────────────────────────────────────────

INJECTABLE_LOWERING_FAILURES: Tuple[str, ...] = (
    "wrong_arch",
    "missing_symbol",
    "schema_mismatch",
    "abi_mismatch",
    "guard_false",
    "compile_error",
    "runtime_error",
)


def inject_lowering_failure(
    kind: str,
    *,
    entry: LoweringEntry,
    target: TargetSnapshot,
    mutate_state: bool = False,
) -> Dict[str, Any]:
    """Simulate a metadata-level incompatibility; nothing is loaded or launched."""
    if kind not in INJECTABLE_LOWERING_FAILURES:
        raise ConfigError(f"unknown injection kind {kind!r}")
    attempted_load = False
    reason = {
        "wrong_arch": "TARGET_UNSUPPORTED",
        "missing_symbol": "BINARY_INCOMPATIBLE",
        "schema_mismatch": "ABI_MISMATCH",
        "abi_mismatch": "ABI_MISMATCH",
        "guard_false": "GUARD_FALSE",
        "compile_error": "COMPILE_ERROR",
        "runtime_error": "ERROR_NO_SAFE_FALLBACK",
    }[kind]
    action = {
        "wrong_arch": "not_registered",
        "missing_symbol": "rejected_at_load_validation",
        "schema_mismatch": "rejected_at_load_validation",
        "abi_mismatch": "rejected_at_load_validation",
        "guard_false": "fallback_before_launch",
        "compile_error": "fallback_reference_no_cache_pollution",
        "runtime_error": "fail_request" if mutate_state else "retry_reference",
    }[kind]
    return {
        "kind": kind,
        "candidate_id": entry.candidate_id,
        "target_id": target.target_id,
        "attempted_load": attempted_load,
        "attempted_launch": False,
        "reason_code": reason,
        "action": action,
        "status": STATUS_FAIL_LOWERING if kind != "compile_error" else STATUS_FAIL_COMPILE,
        "ok": not attempted_load,
    }


# ── correctness/performance ordering (§11) ─────────────────────────────────

CORRECTNESS_ORDER: Tuple[str, ...] = (
    "ir_verifier",
    "reference_lowering_vs_original",
    "custom_operator_vs_reference",
    "qwen_block_selected_tensors",
    "prefill_hidden_logits_topk_kl",
    "multi_step_decode_logits_token_kv",
    "unsupported_fallback_correctness",
)

PERFORMANCE_ORDER: Tuple[str, ...] = (
    "compile_breakdown",
    "operator_steady",
    "block_steady",
    "phase_steady",
    "model_end_to_end",
    "break_even_call_weighting",
)


def next_gate(completed: Sequence[str], order: Sequence[str]) -> str:
    """The next gate in the frozen order (no skipping)."""
    for gate in order:
        if gate not in completed:
            return gate
    return ""


def gate_sequence_ok(completed: Sequence[str], order: Sequence[str]) -> Dict[str, Any]:
    """Completed gates must be a prefix of the frozen order."""
    prefix = list(order[: len(completed)])
    return {
        "completed": list(completed),
        "expected_prefix": prefix,
        "ok": list(completed) == prefix,
        "note": "correctness gates precede performance; a failure stops the performance claim",
    }


def performance_allowed(correctness_rows: Mapping[str, str]) -> Dict[str, Any]:
    """Performance may only start after every correctness gate passes."""
    failing = sorted(
        gate
        for gate, status in correctness_rows.items()
        if gate in CORRECTNESS_ORDER and status != "pass"
    )
    return {
        "correctness": dict(sorted(correctness_rows.items())),
        "failing": failing,
        "allowed": not failing,
        "reason": (
            "" if not failing else "correctness gate not passed: " + ", ".join(failing)
        ),
    }


def compile_breakdown(
    *,
    capture_time: float,
    graph_transform_time: float,
    lowering_selection_time: float,
    codegen_time: float,
    native_compile_link_time: float,
    artifact_write_time: float,
    cache_lookup_and_load_time: float = 0.0,
    autotune_search_time: float = 0.0,
) -> Dict[str, Any]:
    """Compile cost decomposition that must never be mixed into steady state."""
    breakdown = {
        "capture_time": capture_time,
        "graph_transform_time": graph_transform_time,
        "lowering_selection_time": lowering_selection_time,
        "autotune_search_time": autotune_search_time,
        "codegen_time": codegen_time,
        "native_compile_link_time": native_compile_link_time,
        "artifact_write_time": artifact_write_time,
        "cache_lookup_and_load_time": cache_lookup_and_load_time,
    }
    unknown = sorted(set(breakdown) - set(COMPILER_COST_KEYS))
    return {
        "breakdown": {key: round(value, 9) for key, value in breakdown.items()},
        "unknown_keys": unknown,
        "cold_total_s": round(sum(breakdown.values()), 9),
        "first_execution_note": (
            "first_execution_lazy_init_time and warm_steady_runtime are measured separately; "
            "they are not part of the compile breakdown"
        ),
    }


def success_status_for(decision: LoweringDecision, plan: MaterializedPlan) -> str:
    if plan.status == "failed":
        return STATUS_FAIL_LOWERING
    if plan.status == "fallback_only":
        return STATUS_PASS  # correct fallback is a success path, reported separately
    return STATUS_PASS
