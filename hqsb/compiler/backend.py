"""The HQSB compiler backend contract and the CompileRun assembly.

Protocol anchor: ``details/S11/E11-03`` §4 (the backend must expose the nine
steps below and the registry must not compile at import time) and §13 (returning
``gm.forward`` while claiming a compiled backend is forbidden).

``HQSBBackend`` is deliberately dependency-free: every step is an injected
callable, so the same object can be driven from the real driver, from a unit
test, or from a ``torch.compile`` entry point without importing torch here.
The class records the *facts* (was the graph rewritten? was a lowering
selected? was a kernel materialised?) so a debug backend cannot be mistaken
for a compiled path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text
from hqsb.compiler.lowering import LoweringDecision, MaterializedPlan
from hqsb.compiler.records import CompileRun, merge_step_timings

#: The nine backend steps (E11-03 §4), in execution order.
BACKEND_STEPS: Tuple[str, ...] = (
    "capture_adapter",
    "canonicalize_and_verify",
    "run_semantic_passes",
    "analyze_target",
    "enumerate_lowerings",
    "build_guards",
    "select_candidate",
    "materialize_callable",
    "emit_artifact_manifest",
)

STEP_STATUSES: Tuple[str, ...] = ("ok", "skipped", "failed", "not_run")


@dataclass
class StepRecord:
    """One backend step: status, timing, output reference and reason."""

    step: str
    status: str = "not_run"
    output_ref: str = ""
    elapsed_s: float = 0.0
    reason: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.step not in BACKEND_STEPS:
            problems.append(f"unknown backend step {self.step!r}")
        if self.status not in STEP_STATUSES:
            problems.append(f"unknown step status {self.status!r}")
        if self.status == "failed" and not self.reason:
            problems.append(f"{self.step}: a failed step needs a structured reason")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "output_ref": self.output_ref,
            "elapsed_s": self.elapsed_s,
            "reason": self.reason,
        }


@dataclass
class BackendPlan:
    """The result of one compile attempt through the backend."""

    compile_id: str
    run_id: str
    case_id: str
    steps: Tuple[StepRecord, ...]
    graph_id: str = ""
    canonical_ir_id: str = ""
    targeted_ir_id: str = ""
    rewrite_count: int = 0
    lowering: Optional[LoweringDecision] = None
    materialized: Optional[MaterializedPlan] = None
    capture_only: bool = False
    graph_break_count: int = 0
    guard_set_id: str = ""
    artifact_manifest_id: str = ""
    trace_id: str = ""
    error: str = ""

    @property
    def is_debug_backend(self) -> bool:
        """True when nothing was lowered and the original callable would run."""
        return self.rewrite_count == 0 and (self.lowering is None or not self.lowering.selected)

    def validate(self) -> List[str]:
        problems: List[str] = []
        for record in self.steps:
            problems.extend(record.validate())
        if self.capture_only and not self.is_debug_backend:
            problems.append("capture_only backend reported lowering work")
        if self.materialized and self.materialized.status == "materialized" and not self.artifact_manifest_id:
            problems.append("a materialised candidate must emit an artifact manifest id")
        return problems

    def as_compile_run(self) -> CompileRun:
        lowering_id = self.lowering.selected if self.lowering else ""
        fallback_id = self.lowering.fallback_id if self.lowering else ""
        return CompileRun(
            run_id=self.run_id,
            compile_id=self.compile_id,
            case_id=self.case_id,
            capture_mode="dynamo_debug_backend" if self.capture_only else "compile_default",
            compiler_stack_versions={},
            source_graph_id=self.graph_id,
            canonical_ir_id=self.canonical_ir_id,
            targeted_ir_id=self.targeted_ir_id,
            target_id=self.lowering.target_snapshot_id if self.lowering else "",
            guard_set_id=self.guard_set_id,
            selected_lowering_id=lowering_id,
            fallback_id=fallback_id,
            compile_status=(
                "PASS" if self.error == "" and (self.materialized and self.materialized.status != "failed")
                else "FAIL_COMPILE"
            ),
            runtime_status="PLANNED",
            correctness_status="not_run",
            timing_breakdown=merge_step_timings(
                {record.step: record.elapsed_s for record in self.steps if record.elapsed_s}
            ),
            artifact_manifest_id=self.artifact_manifest_id,
            trace_id=self.trace_id,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compile_id": self.compile_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "steps": [record.as_dict() for record in self.steps],
            "graph_id": self.graph_id,
            "canonical_ir_id": self.canonical_ir_id,
            "targeted_ir_id": self.targeted_ir_id,
            "rewrite_count": self.rewrite_count,
            "lowering": self.lowering.as_dict() if self.lowering else None,
            "materialized": self.materialized.as_dict() if self.materialized else None,
            "capture_only": self.capture_only,
            "is_debug_backend": self.is_debug_backend,
            "graph_break_count": self.graph_break_count,
            "guard_set_id": self.guard_set_id,
            "artifact_manifest_id": self.artifact_manifest_id,
            "trace_id": self.trace_id,
            "error": self.error,
        }


StepFn = Callable[..., Any]


@dataclass
class BackendHooks:
    """Injected implementations of the nine steps (real driver or test double)."""

    capture_adapter: Optional[StepFn] = None
    canonicalize_and_verify: Optional[StepFn] = None
    run_semantic_passes: Optional[StepFn] = None
    analyze_target: Optional[StepFn] = None
    enumerate_lowerings: Optional[StepFn] = None
    build_guards: Optional[StepFn] = None
    select_candidate: Optional[StepFn] = None
    materialize_callable: Optional[StepFn] = None
    emit_artifact_manifest: Optional[StepFn] = None

    def hook(self, step: str) -> Optional[StepFn]:
        if step not in BACKEND_STEPS:
            raise ConfigError(f"unknown backend step {step!r}")
        return getattr(self, step)


class HQSBBackend:
    """A PyTorch-compatible custom backend with explicit step accounting."""

    def __init__(
        self,
        *,
        hooks: BackendHooks,
        compile_id: str = "compile",
        run_id: str = "run",
        case_id: str = "case",
        capture_only: bool = False,
        timer: Optional[Callable[[], float]] = None,
    ) -> None:
        self.hooks = hooks
        self.compile_id = compile_id
        self.run_id = run_id
        self.case_id = case_id
        self.capture_only = capture_only
        self._timer = timer
        self.plans: List[BackendPlan] = []
        self._last_state: Dict[str, Any] = {}

    # -- the torch custom-backend contract ---------------------------------

    def __call__(self, graph_module: Any, example_inputs: Sequence[Any]) -> Any:
        """``(GraphModule, example_inputs) -> callable`` (PyTorch contract).

        In ``capture_only`` mode the original ``forward`` is returned and the
        plan is flagged ``is_debug_backend``: that is a *capture* backend, not
        a compiled path.
        """
        plan = self.compile(graph_module, example_inputs)
        if plan.error:
            raise ConfigError(f"HQSB backend failed: {plan.error}")
        return self._callable_for(plan)

    def _callable_for(self, plan: BackendPlan) -> Any:
        factory = plan.materialized.selected_implementation if plan.materialized else ""
        original = self._last_state.get("forward")
        if not factory or plan.is_debug_backend:
            return original if original is not None else (lambda *args, **kwargs: None)
        builder = self._last_state.get("callable_builder")
        if builder is None:
            raise ConfigError(
                "a materialised candidate needs a callable builder; returning the original "
                "forward here would silently turn the compiled path back into eager"
            )
        return builder(factory)

    # -- orchestration ------------------------------------------------------

    def compile(self, graph_module: Any, example_inputs: Sequence[Any]) -> BackendPlan:
        if not self.capture_only and self.hooks.materialize_callable is None:
            raise ConfigError(
                "a compiled backend requires materialize_callable; refusing to return the "
                "original forward as if it were compiled (E11-03 §13)"
            )
        records: List[StepRecord] = []
        state: Dict[str, Any] = {
            "graph_module": graph_module,
            "example_inputs": example_inputs,
            "forward": getattr(graph_module, "forward", None),
            "capture_only": self.capture_only,
        }
        error = ""
        for step in BACKEND_STEPS:
            hook = self.hooks.hook(step)
            if hook is None:
                records.append(
                    StepRecord(
                        step=step,
                        status="failed" if step in ("capture_adapter", "materialize_callable") else "skipped",
                        reason="hook not provided",
                    )
                )
                if step in ("capture_adapter", "materialize_callable"):
                    error = f"missing hook for required step {step!r}"
                    break
                continue
            started = self._now()
            try:
                output = hook(state)
            except ConfigError as exc:
                records.append(
                    StepRecord(step=step, status="failed", elapsed_s=self._now() - started, reason=str(exc))
                )
                error = f"{step}: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                records.append(
                    StepRecord(
                        step=step,
                        status="failed",
                        elapsed_s=self._now() - started,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                error = f"{step}: {type(exc).__name__}: {exc}"
                break
            elapsed = self._now() - started
            if isinstance(output, Mapping):
                state.update(output)
            status = "ok"
            reason = ""
            if step == "materialize_callable" and state.get("materialize_status") == "fallback_only":
                status = "ok"
                reason = "forced candidate unsupported: fallback materialised (reported, not hidden)"
            records.append(
                StepRecord(
                    step=step,
                    status=status,
                    output_ref=str(state.get(f"{step}_ref", "")),
                    elapsed_s=elapsed,
                    reason=reason,
                )
            )
        self._last_state = state
        plan = self._plan_from_state(records, state, error)
        problems = plan.validate()
        if problems and not error:
            plan = BackendPlan(**{**plan.__dict__, "error": "; ".join(problems)})
        self.plans.append(plan)
        return plan

    def _now(self) -> float:
        if self._timer is not None:
            return self._timer()
        import time

        return time.perf_counter()

    def _plan_from_state(
        self, records: Sequence[StepRecord], state: Mapping[str, Any], error: str
    ) -> BackendPlan:
        lowering = state.get("lowering")
        materialized = state.get("materialized")
        return BackendPlan(
            compile_id=self.compile_id,
            run_id=self.run_id,
            case_id=self.case_id,
            steps=tuple(records),
            graph_id=str(state.get("graph_id", "")),
            canonical_ir_id=str(state.get("canonical_ir_id", "")),
            targeted_ir_id=str(state.get("targeted_ir_id", "")),
            rewrite_count=int(state.get("rewrite_count", 0) or 0),
            lowering=lowering if isinstance(lowering, LoweringDecision) else None,
            materialized=materialized if isinstance(materialized, MaterializedPlan) else None,
            capture_only=self.capture_only,
            graph_break_count=int(state.get("graph_break_count", 0) or 0),
            guard_set_id=str(state.get("guard_set_id", "")),
            artifact_manifest_id=str(state.get("artifact_manifest_id", "")),
            trace_id=str(state.get("trace_id", "")),
            error=error,
        )


def assert_not_debug_backend(plan: BackendPlan) -> Dict[str, Any]:
    """Refuse to call a capture-only plan a compiled path (E11-03 §13)."""
    if plan.capture_only or plan.is_debug_backend:
        return {
            "ok": False,
            "reason": (
                "the plan returned the original callable: this is a capture/debug backend, "
                "not a compiled graph→kernel path"
            ),
            "plan": plan.as_dict(),
        }
    return {"ok": True, "reason": "", "graph_id": plan.graph_id}


def emit_artifact_manifest(
    plan: BackendPlan, *, artifacts: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Build the artifact manifest for a compile (step 17/step 9 payload)."""
    rows = [dict(item) for item in artifacts]
    payload = {
        "compile_id": plan.compile_id,
        "run_id": plan.run_id,
        "graph_id": plan.graph_id,
        "canonical_ir_id": plan.canonical_ir_id,
        "targeted_ir_id": plan.targeted_ir_id,
        "selected": plan.lowering.selected if plan.lowering else "",
        "fallback": plan.lowering.fallback_id if plan.lowering else "",
        "artifacts": rows,
    }
    manifest_id = sha256_text(canonical_json(payload))[:16]
    return {
        "manifest_id": manifest_id,
        "payload": payload,
        "artifact_count": len(rows),
        "lineage_complete": all(
            item.get("ir_level") and item.get("canonical_hash") for item in rows
        ),
    }


def backend_contract_document() -> Dict[str, Any]:
    """Human-readable contract summary used by the development report."""
    return {
        "steps": list(BACKEND_STEPS),
        "signature": "(GraphModule, example_inputs) -> callable",
        "rules": [
            "the returned callable must be equivalent to the original graph",
            "returning gm.forward is a debug backend and must be flagged as such",
            "registry import must not compile, create device contexts or download",
            "capability probe and materialisation are separate steps",
            "every step records status/timing/output reference",
        ],
    }


def torch_compile_invocation(
    *, model_name: str, backend_ref: str = "hqsb.compiler.backend.HQSBBackend", mode: str = "default"
) -> Dict[str, Any]:
    """Describe how the driver wires the backend into ``torch.compile``.

    Importing torch happens inside the *driver*, not here (CPU-minimal rule);
    this function only returns the command description that the driver prints
    and stores.
    """
    return {
        "model": model_name,
        "call": f"torch.compile(model, backend={backend_ref}, mode={mode!r}, dynamic=None)",
        "notes": (
            "fullgraph=True is used for the boundary probe; default mode keeps graph breaks "
            "visible in the census"
        ),
        "precondition": (
            "the driver must have recorded TORCH_VERSION/Inductor version; without them the "
            "compile identity is incomplete"
        ),
    }
