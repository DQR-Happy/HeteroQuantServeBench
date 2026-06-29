"""Zero-trust quality gates for AI/Agent generated kernel candidates (E11-10).

Protocol anchor: ``details/S11/E11-10_ai_assisted_kernel_quality_gate.md``.
AI/Agent output is *untrusted code*: it may hardcode inputs, skip computation,
read out of bounds, exploit asynchronous timing or read secrets.  The gate
chain is ordered and strictly cumulative — a later gate can never overrule an
earlier failure (a very fast candidate that fails G5 correctness is rejected).

Everything is data-driven: candidates are mappings, gates are pure functions
over an injected evaluation harness, and the sandbox/harness contracts are
declared objects so the driver cannot accidentally give a candidate write
access to the real reference or timer.

The module implements (not just documents) the gate chain, the wrong-candidate
corpus model, hidden/metamorphic test design, measurement-integrity checks, the
``fast_p`` metric, the admission package schema and the iteration lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.compiler.identity import canonical_json, sha256_text

# ── task package, generation config, provenance (steps 2–4) ────────────────


@dataclass
class TaskPackage:
    """Frozen task: spec, allowed APIs, examples, forbidden actions, limits."""

    task_id: str
    operator_spec_id: str
    allowed_apis: Tuple[str, ...]
    public_examples: Tuple[Mapping[str, Any], ...]
    forbidden_actions: Tuple[str, ...]
    resource_limits: Mapping[str, Any]
    output_contract: Mapping[str, Any]
    reference_hash: str = ""
    harness_hash: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("task_id", "operator_spec_id"):
            if not getattr(self, name):
                problems.append(f"task package missing {name!r}")
        if not self.allowed_apis:
            problems.append("allowed APIs must be enumerated (no implicit anything-goes)")
        if not self.forbidden_actions:
            problems.append("forbidden actions must be enumerated (the generator cannot modify them)")
        if not self.resource_limits:
            problems.append("resource limits are mandatory for untrusted code")
        if not (self.reference_hash and self.harness_hash):
            problems.append(
                "reference and harness hashes must be frozen before generation (G0 immutability)"
            )
        return problems

    def digest(self) -> str:
        payload = {
            "task_id": self.task_id,
            "operator_spec_id": self.operator_spec_id,
            "allowed_apis": list(self.allowed_apis),
            "public_examples": [dict(item) for item in self.public_examples],
            "forbidden_actions": list(self.forbidden_actions),
            "resource_limits": dict(sorted(self.resource_limits.items())),
            "output_contract": dict(sorted(self.output_contract.items())),
            "reference_hash": self.reference_hash,
            "harness_hash": self.harness_hash,
        }
        return sha256_text(canonical_json(payload))


@dataclass
class GenerationConfig:
    """Frozen generation configuration (step 3)."""

    provider: str
    model: str
    model_version: str
    prompt_template_hash: str
    temperature: float
    seed: int
    max_iterations: int
    max_tokens: int
    feedback_policy: str
    generated_at: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("provider", "model", "model_version", "prompt_template_hash", "feedback_policy"):
            if not getattr(self, name):
                problems.append(f"generation config missing {name!r}")
        if self.max_iterations <= 0 or self.max_tokens <= 0:
            problems.append("generation budgets must be positive and frozen")
        if not 0.0 <= self.temperature <= 2.0:
            problems.append("temperature out of range")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "prompt_template_hash": self.prompt_template_hash,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_iterations": self.max_iterations,
            "max_tokens": self.max_tokens,
            "feedback_policy": self.feedback_policy,
            "generated_at": self.generated_at,
        }


PROVENANCE_FIELDS: Tuple[str, ...] = (
    "candidate_id",
    "parent_id",
    "prompt_hash",
    "model",
    "model_version",
    "generation_index",
    "patch",
    "tool_calls",
    "human_edits",
    "license_declaration",
    "source_sha256",
    "binary_sha256",
)


@dataclass
class ProvenanceRecord:
    candidate_id: str
    prompt_hash: str = ""
    model: str = ""
    model_version: str = ""
    parent_id: str = ""
    generation_index: int = 0
    patch: str = ""
    tool_calls: Tuple[str, ...] = ()
    human_edits: Tuple[str, ...] = ()
    license_declaration: str = ""
    source_sha256: str = ""
    binary_sha256: str = ""

    def missing_fields(self) -> List[str]:
        values = {
            "candidate_id": self.candidate_id,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "model_version": self.model_version,
            "patch": self.patch,
            "license_declaration": self.license_declaration,
            "source_sha256": self.source_sha256,
        }
        return sorted(name for name, value in values.items() if not value)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "model_version": self.model_version,
            "generation_index": self.generation_index,
            "patch": self.patch,
            "tool_calls": list(self.tool_calls),
            "human_edits": list(self.human_edits),
            "license_declaration": self.license_declaration,
            "source_sha256": self.source_sha256,
            "binary_sha256": self.binary_sha256,
        }


# ── candidate classes, wrong corpus, sandbox (steps 5–8) ───────────────────

CANDIDATE_CLASSES: Mapping[str, str] = {
    "C0_control_correct": "known-correct reference/handwritten: the gates must not kill it",
    "C1_syntax_compile_wrong": "syntax/type/link/unsupported API",
    "C2_math_wrong": "eps/axis/cast/reduction/order errors",
    "C3_shape_hardcode": "supports only the public shapes/tails",
    "C4_value_hardcode_skip": "detects fixed inputs, writes partial outputs",
    "C5_memory_unsafe": "out-of-bounds, misalignment, uninitialised reads",
    "C6_race_nondeterministic": "missing synchronisation / data races",
    "C7_timing_exploit": "no sync, modified timer, async return",
    "C8_state_corrupt": "mutates input/residual/KV/RNG",
    "C9_slow_resource_bomb": "huge workspace, compile/run timeouts",
    "C10_provenance_violation": "unlicensed copying, secrets/paths",
    "C11_genuine_ai_proposal": "real generated candidate to be judged",
}

EXPECTED_GATE_BY_CLASS: Mapping[str, str] = {
    "C0_control_correct": "G10_review",
    "C1_syntax_compile_wrong": "G2_isolated_compile",
    "C2_math_wrong": "G5_hidden_correctness",
    "C3_shape_hardcode": "G5_hidden_correctness",
    "C4_value_hardcode_skip": "G5_hidden_correctness",
    "C5_memory_unsafe": "G3_sanitizer_memory",
    "C6_race_nondeterministic": "G6_state_concurrency",
    "C7_timing_exploit": "G7_measurement_integrity",
    "C8_state_corrupt": "G6_state_concurrency",
    "C9_slow_resource_bomb": "G8_performance_resource",
    "C10_provenance_violation": "G1_static_policy",
    "C11_genuine_ai_proposal": "G10_review",
}


@dataclass
class WrongCorpusEntry:
    candidate_id: str
    candidate_class: str
    expected_gate: str
    expected_reason: str
    human_reviewed: bool = False
    isolated_from_hidden_tests: bool = True

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.candidate_class not in CANDIDATE_CLASSES:
            problems.append(f"unknown candidate class {self.candidate_class!r}")
        elif EXPECTED_GATE_BY_CLASS[self.candidate_class] != self.expected_gate:
            problems.append(
                f"{self.candidate_id}: expected gate {self.expected_gate!r} != class gate "
                f"{EXPECTED_GATE_BY_CLASS[self.candidate_class]!r}"
            )
        if not self.human_reviewed:
            problems.append(f"{self.candidate_id}: expected failure must be human-reviewed")
        if not self.isolated_from_hidden_tests:
            problems.append(
                f"{self.candidate_id}: wrong corpus must stay isolated from the final hidden tests"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_class": self.candidate_class,
            "expected_gate": self.expected_gate,
            "expected_reason": self.expected_reason,
            "human_reviewed": self.human_reviewed,
            "isolated_from_hidden_tests": self.isolated_from_hidden_tests,
        }


def wrong_corpus_template() -> Dict[str, Any]:
    """One pre-registered entry per class C0–C10 (step 7)."""
    entries = [
        WrongCorpusEntry(
            candidate_id=f"wrong_{cls.split('_', 1)[1]}",
            candidate_class=cls,
            expected_gate=EXPECTED_GATE_BY_CLASS[cls],
            expected_reason=CANDIDATE_CLASSES[cls],
        )
        for cls in sorted(CANDIDATE_CLASSES)
        if cls != "C11_genuine_ai_proposal"
    ]
    coverage = sorted({entry.candidate_class for entry in entries})
    missing = sorted(set(CANDIDATE_CLASSES) - set(coverage) - {"C11_genuine_ai_proposal"})
    return {
        "entries": [entry.as_dict() for entry in entries],
        "classes_covered": coverage,
        "missing_classes": missing,
        "complete": not missing,
        "rule": "each class changes exactly one failure mechanism and has a reviewed expectation",
    }


@dataclass
class SandboxPolicy:
    """Least-privilege evaluator contract (step 5)."""

    network: bool = False
    secrets_visible: bool = False
    reference_read_only: bool = True
    registry_writable: bool = False
    official_cache_writable: bool = False
    scratch_dir: str = ""
    cpu_seconds_limit: float = 0.0
    gpu_seconds_limit: float = 0.0
    memory_mb_limit: int = 0
    process_limit: int = 0
    file_size_mb_limit: int = 0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.network:
            problems.append("the evaluator must run without network access")
        if self.secrets_visible:
            problems.append("the evaluator must not see secrets")
        if not self.reference_read_only:
            problems.append("reference/tests/harness must be read-only")
        if self.registry_writable or self.official_cache_writable:
            problems.append("the official registry/cache must not be writable from the sandbox")
        if not self.scratch_dir:
            problems.append("an isolated scratch directory is mandatory")
        if min(
            self.cpu_seconds_limit,
            self.gpu_seconds_limit,
            self.memory_mb_limit,
            self.process_limit,
            self.file_size_mb_limit,
        ) <= 0:
            problems.append("all CPU/GPU/memory/process/file quotas must be positive")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network,
            "secrets_visible": self.secrets_visible,
            "reference_read_only": self.reference_read_only,
            "registry_writable": self.registry_writable,
            "official_cache_writable": self.official_cache_writable,
            "scratch_dir": self.scratch_dir,
            "cpu_seconds_limit": self.cpu_seconds_limit,
            "gpu_seconds_limit": self.gpu_seconds_limit,
            "memory_mb_limit": self.memory_mb_limit,
            "process_limit": self.process_limit,
            "file_size_mb_limit": self.file_size_mb_limit,
        }


@dataclass
class HarnessLock:
    """Trusted harness identity; the candidate API never receives these paths."""

    runner_hash: str
    reference_hash: str
    tests_hash: str
    timer_hash: str
    build_wrapper_hash: str
    candidate_api_surface: Tuple[str, ...] = ("tensors", "config")

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("runner_hash", "reference_hash", "tests_hash", "timer_hash", "build_wrapper_hash"):
            if not getattr(self, name):
                problems.append(f"harness lock missing {name!r}")
        forbidden = {"timer", "test_paths", "expected_outputs", "reference_paths", "runner"}
        leaked = sorted(forbidden & set(self.candidate_api_surface))
        if leaked:
            problems.append(f"candidate API surface leaks trusted components: {leaked}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runner_hash": self.runner_hash,
            "reference_hash": self.reference_hash,
            "tests_hash": self.tests_hash,
            "timer_hash": self.timer_hash,
            "build_wrapper_hash": self.build_wrapper_hash,
            "candidate_api_surface": list(self.candidate_api_surface),
        }


# ── hidden tests (steps 16–18) ─────────────────────────────────────────────


@dataclass
class HiddenShapeDistribution:
    """Generator knows the domain, never the values (step 16)."""

    interpolation: Tuple[Mapping[str, int], ...] = ()
    boundary: Tuple[Mapping[str, int], ...] = ()
    extrapolation: Tuple[Mapping[str, int], ...] = ()
    batch_layout: Tuple[Mapping[str, int], ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.interpolation:
            problems.append("hidden distribution needs interpolation shapes")
        if not self.boundary:
            problems.append("hidden distribution needs boundary/tail shapes")
        if not self.extrapolation:
            problems.append("hidden distribution needs extrapolation shapes")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "interpolation": [dict(item) for item in self.interpolation],
            "boundary": [dict(item) for item in self.boundary],
            "extrapolation": [dict(item) for item in self.extrapolation],
            "batch_layout": [dict(item) for item in self.batch_layout],
        }


@dataclass
class HiddenValueDistribution:
    seeds: Tuple[int, ...] = ()
    include_zeros: bool = True
    include_constants: bool = True
    include_large_magnitude: bool = True
    include_cancellation: bool = True
    include_nan_inf: bool = False
    include_signed_zero: bool = False
    contract_allows_nan_inf: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if len(self.seeds) < 3:
            problems.append("at least three unpredictable seeds are required")
        if self.include_nan_inf and not self.contract_allows_nan_inf:
            problems.append("NaN/Inf cases are only meaningful if the contract defines them")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "include_zeros": self.include_zeros,
            "include_constants": self.include_constants,
            "include_large_magnitude": self.include_large_magnitude,
            "include_cancellation": self.include_cancellation,
            "include_nan_inf": self.include_nan_inf,
            "include_signed_zero": self.include_signed_zero,
        }


def metamorphic_tests() -> List[Dict[str, Any]]:
    """Property relations that do not need a golden output (step 18)."""
    return [
        {
            "name": "batch_permutation",
            "relation": "f(permute(x)) == permute(f(x))",
            "applies_when": "the op is batch-independent",
        },
        {
            "name": "scale_invariance_of_normalised_output",
            "relation": "rmsnorm(alpha*x) == rmsnorm(x) for alpha > 0 (same weight)",
            "applies_when": "RMSNorm semantics",
        },
        {
            "name": "chunk_concatenate",
            "relation": "f([a; b]) matches concatenated row-wise results",
            "applies_when": "row-independent ops",
        },
    ]


# ── gate chain (steps 9–13, 19–28) ─────────────────────────────────────────

GATE_ORDER: Tuple[str, ...] = (
    "G0_provenance_immutability",
    "G1_static_policy",
    "G2_isolated_compile",
    "G3_sanitizer_memory",
    "G4_public_correctness",
    "G5_hidden_correctness",
    "G6_state_concurrency",
    "G7_measurement_integrity",
    "G8_performance_resource",
    "G9_compiler_integration",
    "G10_review",
)

GATE_OUTCOMES: Tuple[str, ...] = ("accept", "reject", "not_run")

BANNED_IMPORTS: Tuple[str, ...] = (
    "os.system",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "ctypes",
    "shutil.rmtree",
    "pickle",
)

BANNED_PATTERNS: Tuple[str, ...] = (
    "torch.cuda.synchronize",   # candidates must not control measurement
    "time.time",                # wall-clock timing inside the candidate
    "os.environ",
    "/etc/",
    "ssh",
)


@dataclass
class GateResult:
    gate: str
    outcome: str
    reason_code: str = ""
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    earliest: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.gate not in GATE_ORDER:
            problems.append(f"unknown gate {self.gate!r}")
        if self.outcome not in GATE_OUTCOMES:
            problems.append(f"unknown gate outcome {self.outcome!r}")
        if self.outcome == "reject" and not self.reason_code:
            problems.append(f"{self.gate}: a rejection needs a structured reason")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "evidence": dict(sorted(self.evidence.items())),
            "earliest": self.earliest,
        }


REJECT_PREFIX = "REJECTED_"


def candidate_verdict(chain: Sequence[GateResult], candidate_id: str, *, faster: bool = False) -> Dict[str, Any]:
    """``REJECTED_<GATE>`` | ``QUALIFIED_NOT_FASTER`` | ``QUALIFIED_EXPERIMENTAL`` | ``ADMITTED``."""
    failures = [row for row in chain if row.outcome == "reject"]
    if failures:
        first = failures[0]
        return {
            "candidate_id": candidate_id,
            "status": f"{REJECT_PREFIX}{first.gate}",
            "earliest_failure": first.gate,
            "all_failures": [row.gate for row in failures],
            "reason": first.reason_code,
        }
    if not faster:
        return {
            "candidate_id": candidate_id,
            "status": "QUALIFIED_NOT_FASTER",
            "earliest_failure": "",
            "all_failures": [],
            "reason": "correct and safe but no measured advantage over the strong baseline",
        }
    review = next((row for row in chain if row.gate == "G10_review"), None)
    if review is None or review.outcome != "accept":
        return {
            "candidate_id": candidate_id,
            "status": "QUALIFIED_EXPERIMENTAL",
            "earliest_failure": "",
            "all_failures": [],
            "reason": "all automated gates passed; independent human review pending",
        }
    return {
        "candidate_id": candidate_id,
        "status": "ADMITTED",
        "earliest_failure": "",
        "all_failures": [],
        "reason": "all gates and human review passed (still bound to target/domain/version)",
    }


def gate_g0_provenance(provenance: ProvenanceRecord, task: TaskPackage) -> GateResult:
    missing = provenance.missing_fields()
    return GateResult(
        gate="G0_provenance_immutability",
        outcome="reject" if missing else "accept",
        reason_code="PROVENANCE_INCOMPLETE" if missing else "",
        detail=f"missing: {missing}" if missing else "provenance complete; task/harness frozen",
        evidence={"task_digest": task.digest(), "missing": missing},
    )


def gate_g1_static_policy(
    *, source_text: str, imports: Sequence[str] = (), secrets_found: Sequence[str] = (),
    license_declaration: str = "",
) -> GateResult:
    violations: List[str] = []
    for name in imports:
        if any(name.startswith(banned) for banned in BANNED_IMPORTS):
            violations.append(f"banned import: {name}")
    for pattern in BANNED_PATTERNS:
        if pattern in source_text:
            violations.append(f"banned pattern: {pattern}")
    if secrets_found:
        violations.append(f"possible secrets: {list(secrets_found)}")
    if not license_declaration:
        violations.append("missing license declaration")
    return GateResult(
        gate="G1_static_policy",
        outcome="reject" if violations else "accept",
        reason_code="STATIC_POLICY_VIOLATION" if violations else "",
        detail="; ".join(violations),
        evidence={"violations": violations},
    )


def gate_g2_isolated_compile(
    *,
    compile_status: str,
    stdout: str = "",
    stderr: str = "",
    artifact_hash: str = "",
    timeout: bool = False,
    oom: bool = False,
    sandbox: Optional[SandboxPolicy] = None,
) -> GateResult:
    problems: List[str] = []
    if sandbox is not None:
        problems.extend(sandbox.validate())
    if timeout:
        problems.append("compile timeout")
    if oom:
        problems.append("compile OOM")
    if compile_status != "ok":
        problems.append(f"compile status {compile_status!r}")
    if not artifact_hash:
        problems.append("no artifact hash emitted")
    return GateResult(
        gate="G2_isolated_compile",
        outcome="reject" if problems else "accept",
        reason_code="COMPILE_ERROR" if problems else "",
        detail="; ".join(problems),
        evidence={"stdout_tail": stdout[-200:], "stderr_tail": stderr[-200:]},
    )


def gate_g3_sanitizer_memory(
    *,
    sanitizer_status: str,
    findings: Sequence[str] = (),
    canary_ok: bool = False,
    tool_available: bool = True,
) -> GateResult:
    problems = list(findings)
    if sanitizer_status not in ("clean", "unavailable"):
        problems.append(f"sanitizer status {sanitizer_status!r}")
    if sanitizer_status == "unavailable":
        if not canary_ok:
            problems.append(
                "sanitizer unavailable and no canary/guard-page evidence: the claim must be lowered"
            )
        else:
            return GateResult(
                gate="G3_sanitizer_memory",
                outcome="accept",
                reason_code="SANITIZER_UNAVAILABLE_CANARY_ONLY",
                detail="tool unavailable: accepted with canary evidence and a reduced claim",
                evidence={"tool_available": tool_available},
            )
    return GateResult(
        gate="G3_sanitizer_memory",
        outcome="reject" if problems else "accept",
        reason_code="MEMORY_SAFETY" if problems else "",
        detail="; ".join(problems),
        evidence={"findings": list(findings), "tool_available": tool_available},
    )


def gate_g4_public_correctness(
    *, max_abs_error: Optional[float], tolerance: float, cases: int
) -> GateResult:
    if max_abs_error is None or cases <= 0:
        return GateResult(
            gate="G4_public_correctness",
            outcome="reject",
            reason_code="CORRECTNESS_NOT_MEASURED",
            detail="public correctness produced no measurement",
        )
    ok = max_abs_error <= tolerance
    return GateResult(
        gate="G4_public_correctness",
        outcome="accept" if ok else "reject",
        reason_code="" if ok else "PUBLIC_CORRECTNESS_FAIL",
        detail=f"max_abs_error={max_abs_error} tolerance={tolerance}",
        evidence={"cases": cases, "max_abs_error": max_abs_error},
    )


def gate_g5_hidden_correctness(
    *,
    hidden_cases: int,
    failures: Sequence[Mapping[str, Any]] = (),
    metamorphic_failures: Sequence[str] = (),
    hardcode_suspected: bool = False,
) -> GateResult:
    problems: List[str] = []
    if hidden_cases <= 0:
        problems.append("no hidden cases were executed")
    problems.extend(f"hidden case failed: {dict(item)}" for item in failures)
    problems.extend(f"metamorphic failure: {name}" for name in metamorphic_failures)
    if hardcode_suspected:
        problems.append("hardcoded-input detection triggered")
    return GateResult(
        gate="G5_hidden_correctness",
        outcome="reject" if problems else "accept",
        reason_code="HIDDEN_CORRECTNESS_FAIL" if problems else "",
        detail="; ".join(problems),
        evidence={"hidden_cases": hidden_cases, "failures": [dict(item) for item in failures]},
    )


def gate_g6_state_concurrency(
    *,
    repeated_runs: int,
    nondeterministic: bool = False,
    state_mutated: Sequence[str] = (),
    race_detected: bool = False,
    partial_write: bool = False,
) -> GateResult:
    problems: List[str] = []
    if repeated_runs < 3:
        problems.append("fewer than three repeated runs")
    if nondeterministic:
        problems.append("nondeterministic output across repetitions")
    problems.extend(f"unauthorised mutation: {name}" for name in state_mutated)
    if race_detected:
        problems.append("race detected")
    if partial_write:
        problems.append("partial output written")
    return GateResult(
        gate="G6_state_concurrency",
        outcome="reject" if problems else "accept",
        reason_code="STATE_UNSAFE" if problems else "",
        detail="; ".join(problems),
        evidence={"repeated_runs": repeated_runs, "mutated": list(state_mutated)},
    )


def gate_g7_measurement_integrity(
    *,
    timer_hash_unchanged: bool,
    inputs_unchanged: bool,
    reference_unchanged: bool,
    device_sync_used: bool,
    outputs_validated_after_timing: bool,
    async_exploit_detected: bool = False,
) -> GateResult:
    problems: List[str] = []
    if not timer_hash_unchanged:
        problems.append("timer modified by the candidate")
    if not inputs_unchanged:
        problems.append("inputs modified by the candidate")
    if not reference_unchanged:
        problems.append("reference modified by the candidate")
    if not device_sync_used:
        problems.append("no device synchronisation: wall-clock numbers are meaningless")
    if not outputs_validated_after_timing:
        problems.append("outputs were not validated after timing (async exploit)")
    if async_exploit_detected:
        problems.append("asynchronous timing exploit detected")
    return GateResult(
        gate="G7_measurement_integrity",
        outcome="reject" if problems else "accept",
        reason_code="MEASUREMENT_INTEGRITY" if problems else "",
        detail="; ".join(problems),
        evidence={},
    )


def gate_g8_performance_resource(
    *,
    correct_and_faster: bool,
    speedup: Optional[float],
    min_speedup: float,
    resource_regression: Optional[str] = None,
    compile_time_s: Optional[float] = None,
    compile_time_limit_s: Optional[float] = None,
) -> GateResult:
    problems: List[str] = []
    if speedup is None:
        problems.append("no measured speedup against the strong baseline")
    elif not correct_and_faster:
        problems.append("candidate is not correct-and-faster")
    elif speedup < min_speedup:
        problems.append(f"speedup {speedup} below threshold {min_speedup}")
    if resource_regression:
        problems.append(f"resource regression: {resource_regression}")
    if (
        compile_time_s is not None
        and compile_time_limit_s is not None
        and compile_time_s > compile_time_limit_s
    ):
        problems.append(f"compile time {compile_time_s}s exceeds the limit")
    return GateResult(
        gate="G8_performance_resource",
        outcome="reject" if problems else "accept",
        reason_code="PERFORMANCE_RESOURCE" if problems else "",
        detail="; ".join(problems),
        evidence={"speedup": speedup, "min_speedup": min_speedup},
    )


def gate_g9_compiler_integration(
    *,
    registry_schema_ok: bool,
    guards_ok: bool,
    fallback_ok: bool,
    cache_identity_ok: bool,
    actual_dispatch_ok: bool,
) -> GateResult:
    problems: List[str] = []
    if not registry_schema_ok:
        problems.append("registry schema/capability/evidence incomplete")
    if not guards_ok:
        problems.append("guard domain/fallback not honoured")
    if not fallback_ok:
        problems.append("unsupported inputs do not fall back safely")
    if not cache_identity_ok:
        problems.append("cache key/invalidation identity incomplete")
    if not actual_dispatch_ok:
        problems.append("actual dispatch not confirmed for the candidate artifact")
    return GateResult(
        gate="G9_compiler_integration",
        outcome="reject" if problems else "accept",
        reason_code="INTEGRATION_FAIL" if problems else "",
        detail="; ".join(problems),
        evidence={},
    )


def gate_g10_review(
    *, reviewer_id: str, checklist: Mapping[str, bool], approved_commit: str
) -> GateResult:
    missing = sorted(name for name, ok in checklist.items() if not ok)
    problems: List[str] = []
    if not reviewer_id:
        problems.append("no reviewer id")
    if missing:
        problems.append(f"checklist items not passed: {missing}")
    if not approved_commit:
        problems.append("no approved commit")
    return GateResult(
        gate="G10_review",
        outcome="reject" if problems else "accept",
        reason_code="REVIEW_NOT_APPROVED" if problems else "",
        detail="; ".join(problems),
        evidence={"reviewer": reviewer_id, "missing": missing},
    )


def run_gate_chain(
    candidate_id: str, gate_results: Sequence[GateResult]
) -> Dict[str, Any]:
    """Run/record the chain; later gates never override an earlier failure."""
    ordered = sorted(
        gate_results, key=lambda row: GATE_ORDER.index(row.gate) if row.gate in GATE_ORDER else 999
    )
    problems = [problem for row in ordered for problem in row.validate()]
    first_reject = next((row for row in ordered if row.outcome == "reject"), None)
    executed: List[str] = []
    for row in ordered:
        if first_reject is not None and row is first_reject:
            executed.append(row.gate)
            break
        executed.append(row.gate)
    overruled = [
        row.gate
        for row in ordered
        if row.outcome == "accept" and first_reject is not None
        and GATE_ORDER.index(row.gate) > GATE_ORDER.index(first_reject.gate)
    ]
    return {
        "candidate_id": candidate_id,
        "results": [row.as_dict() for row in ordered],
        "executed": executed,
        "first_reject": first_reject.gate if first_reject else "",
        "reason": first_reject.reason_code if first_reject else "",
        "later_accepts_after_reject": overruled,
        "chain_ok": not problems,
        "rule": "a later gate can never overrule an earlier failure (speed never covers correctness)",
    }


# ── metrics and admission (steps 25–31) ────────────────────────────────────


def fast_p(rows: Sequence[Mapping[str, Any]], *, p: float) -> Dict[str, Any]:
    """``fast_p`` = correct-and-faster fraction; correctness failures are excluded."""
    considered = [row for row in rows if row.get("correct") is not None]
    if not considered:
        return {"p": p, "fast_p": None, "reason": "no correctness-evaluated candidates"}
    fast = sum(
        1
        for row in considered
        if row.get("correct") is True and row.get("speedup") is not None and row["speedup"] >= p
    )
    return {
        "p": p,
        "candidates": len(considered),
        "correct": sum(1 for row in considered if row.get("correct") is True),
        "fast_p": round(fast / len(considered), 4),
        "denominator": "all evaluated candidates (correctness failures are not silently dropped)",
    }


def resource_pareto(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Non-dominated candidates over speedup / memory / compile / code size."""
    keys = ("speedup", "peak_memory_mb", "compile_time_s", "code_size_kb")
    best: List[Mapping[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = all(
                (other.get(key) is not None and row.get(key) is not None and other[key] <= row[key])
                or other.get(key) == row.get(key)
                for key in ("peak_memory_mb", "compile_time_s", "code_size_kb")
            ) and (other.get("speedup") or 0) >= (row.get("speedup") or 0)
            strictly_better = (
                (other.get("speedup") or 0) > (row.get("speedup") or 0)
                or any(
                    other.get(key) is not None
                    and row.get(key) is not None
                    and other[key] < row[key]
                    for key in ("peak_memory_mb", "compile_time_s", "code_size_kb")
                )
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            best.append(row)
    return {
        "dimensions": list(keys),
        "pareto_front": [dict(row) for row in best],
        "note": "a faster candidate with exploding memory is a trade-off, not a free optimisation",
    }


ADMISSION_REQUIRED_FIELDS: Tuple[str, ...] = (
    "candidate_id",
    "source_sha256",
    "binary_sha256",
    "provenance",
    "semantic_contract",
    "target_capability",
    "guard_domain",
    "correctness_public",
    "correctness_hidden",
    "sanitizer_results",
    "performance_vs_baselines",
    "resource_tradeoffs",
    "integration_results",
    "cache_identity",
    "known_limitations",
    "fallback",
    "reviewers",
    "review_commit",
    "license",
)

ADMISSION_DRAFT_STATUS = "DRAFT"
ADMISSION_STATUSES: Tuple[str, ...] = ("DRAFT", "SCHEMA_FAIL", "QUALIFIED", "ADMITTED")


def validate_admission(package: Mapping[str, Any]) -> Dict[str, Any]:
    """A missing required field fails the schema (no silent admission)."""
    missing = sorted(name for name in ADMISSION_REQUIRED_FIELDS if package.get(name) in (None, "", [], {}))
    return {
        "candidate_id": package.get("candidate_id", ""),
        "missing_fields": missing,
        "status": "QUALIFIED" if not missing else "SCHEMA_FAIL",
        "admitted": False,
        "rule": (
            "admission is a separate permission from passing tests: it needs provenance, license, "
            "hash, support domain and an independent reviewer"
        ),
    }


def human_review_checklist() -> List[Dict[str, str]]:
    return [
        {"item": "math", "detail": "derivation matches the operator contract"},
        {"item": "bounds", "detail": "every access proven in range for the declared domain"},
        {"item": "synchronisation", "detail": "streams/events correct, no implicit global sync"},
        {"item": "alias_effect", "detail": "no unauthorised mutation; outputs have the right ownership"},
        {"item": "guards", "detail": "variant/guard domain matches the performed checks"},
        {"item": "maintainability", "detail": "readable, tested, no magic constants copied from tests"},
        {"item": "license", "detail": "no unlicensed copied code; attribution complete"},
        {"item": "generated_code_marking", "detail": "AI provenance recorded in the artifact"},
    ]


def iteration_lineage(
    *, one_shot: Sequence[Mapping[str, Any]], feedback: Sequence[Mapping[str, Any]],
    equal_budget_ok: bool,
) -> Dict[str, Any]:
    """One-shot vs feedback comparison with an equal-budget precondition."""
    def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {
            "candidates": len(rows),
            "generation_seconds": sum(float(row.get("generation_seconds", 0.0)) for row in rows),
            "compile_seconds": sum(float(row.get("compile_seconds", 0.0)) for row in rows),
            "device_seconds": sum(float(row.get("device_seconds", 0.0)) for row in rows),
            "accepted": sum(1 for row in rows if row.get("accepted")),
            "regressions": sum(1 for row in rows if row.get("regression")),
        }

    return {
        "one_shot": _summary(one_shot),
        "feedback": _summary(feedback),
        "equal_budget": equal_budget_ok,
        "comparable": equal_budget_ok,
        "rule": (
            "the feedback loop must not see the final hidden tests; each round records parent "
            "diff, fixes and regressions"
        ),
    }


def adversarial_detection_matrix(
    chain_rows: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Did each wrong candidate get rejected, and at which gate (step 7)?"""
    observed = {str(row.get("candidate_id")): row for row in chain_rows}
    rows: List[Dict[str, Any]] = []
    for entry in expected:
        candidate_id = str(entry["candidate_id"])
        actual = observed.get(candidate_id, {})
        rejected = str(actual.get("status", "")).startswith(REJECT_PREFIX)
        actual_gate = str(actual.get("earliest_failure", ""))
        expected_gate = str(entry.get("expected_gate", ""))
        rows.append(
            {
                "candidate_id": candidate_id,
                "expected_gate": expected_gate,
                "actual_gate": actual_gate,
                "rejected": rejected,
                "at_expected_gate": actual_gate == expected_gate,
                "false_accept": not rejected,
            }
        )
    return {
        "rows": rows,
        "false_accept": [row["candidate_id"] for row in rows if row["false_accept"]],
        "early_gate_gaps": [
            row["candidate_id"] for row in rows if row["rejected"] and not row["at_expected_gate"]
        ],
        "ok": not any(row["false_accept"] for row in rows),
        "rule": (
            "detection at a later gate is safe but wasteful; record earliest vs actual detection "
            "and improve G1–G3"
        ),
    }


def claim_boundary(*, admitted_candidates: int, methodology_pass: bool) -> Dict[str, Any]:
    """The exact wording the project is allowed to use (step 32/§15)."""
    if not methodology_pass:
        return {
            "allowed": False,
            "text": "the gate protocol is incomplete: no AI-assisted claim may be made",
        }
    if admitted_candidates == 0:
        return {
            "allowed": True,
            "text": (
                "established a zero-trust generate→verify protocol for AI kernel candidates and "
                "showed that the pre-registered wrong candidates are rejected; no candidate "
                "reached admission in this round"
            ),
        }
    return {
        "allowed": True,
        "text": (
            "candidate(s) passed all gates and independent review; the claim is bound to the "
            "stated semantic op, shape/dtype/target, baselines and limitations"
        ),
    }
