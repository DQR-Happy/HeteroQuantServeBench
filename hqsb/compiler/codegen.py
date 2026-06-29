"""Codegen artifacts, cross-layer ledgers and IR→binary→hardware attribution.

Protocol anchor: ``details/S11/E11-04_ir_codegen_hardware_attribution.md``.
The experiment may not accept "fewer kernels so it is faster" or "higher
occupancy so it is faster": at least one optimisation *or* regression must be
explained by the joint chain

    pass/config → graph/IR diff → generated source → binary resources →
    hardware counters → phase/model latency

Implemented instruments (all of them record *unavailable* layers explicitly):

* generated-source artifacts with raw/canonical hashes and symbol mapping;
* the toolchain export plan (Triton IR, LLVM IR, PTX, cubin, SASS) which turns
  a missing tool into ``NOT_RUN_TOOL_UNAVAILABLE`` instead of an estimate;
* a tolerant ``cuobjdump``-style resource parser and binary resource report;
* memory-plan and launch ledgers with expected/observed diffs;
* the mechanism hypothesis table with SUPPORTED/REFUTED verdicts and
  alternative explanations;
* single-factor ablation planning and evaluation;
* Amdahl attribution from kernel savings to phase/model latency with residual;
* profiler plan with replay-risk handling, and artifact reconstruction checks.

No tool is executed here: commands are emitted for the driver/audit runner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, content_hash, sha256_text
from hqsb.compiler.targets import TargetSnapshot, toolchain_export_capability

# ── generated source artifacts ─────────────────────────────────────────────

SOURCE_LANGUAGES: Tuple[str, ...] = ("triton", "cuda_cpp", "wrapper_python", "ascend_c", "mlir", "tir")


@dataclass
class GeneratedSourceArtifact:
    """Generated code plus the mapping back to the IR that produced it."""

    artifact_id: str
    language: str
    source_text: str
    entry_symbol: str
    launch_grid: Tuple[int, ...] = ()
    block_size: Tuple[int, ...] = ()
    constexpr: Mapping[str, Any] = field(default_factory=dict)
    compiler_flags: Tuple[str, ...] = ()
    source_map: Mapping[str, str] = field(default_factory=dict)
    parent_ir_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.language not in SOURCE_LANGUAGES:
            problems.append(f"unknown language {self.language!r}")
        if not self.entry_symbol:
            problems.append(f"{self.artifact_id}: generated source must declare its entry symbol")
        if not self.parent_ir_id:
            problems.append(f"{self.artifact_id}: generated source must link back to its IR")
        if not self.source_text.strip():
            problems.append(f"{self.artifact_id}: empty source text")
        return problems

    def hashes(self) -> Dict[str, Any]:
        raw = sha256_text(self.source_text)
        canonical, stripped = content_hash(
            {
                "language": self.language,
                "entry_symbol": self.entry_symbol,
                "launch_grid": list(self.launch_grid),
                "block_size": list(self.block_size),
                "constexpr": dict(self.constexpr),
                "flags": list(self.compiler_flags),
                "source": self.source_text,
            }
        )
        return {"raw_hash": raw, "canonical_hash": canonical, "stripped_keys": stripped}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "language": self.language,
            "entry_symbol": self.entry_symbol,
            "launch_grid": list(self.launch_grid),
            "block_size": list(self.block_size),
            "constexpr": dict(sorted(self.constexpr.items())),
            "compiler_flags": list(self.compiler_flags),
            "source_map": dict(sorted(self.source_map.items())),
            "parent_ir_id": self.parent_ir_id,
            **self.hashes(),
        }


# ── toolchain export plan ──────────────────────────────────────────────────

EXPORT_LAYERS: Tuple[str, ...] = (
    "inductor_scheduler_ir",
    "triton_ir",
    "llvm_ir",
    "ptx",
    "cubin",
    "sass",
    "resource_report",
    "nsys_timeline",
    "ncu_report",
)


def toolchain_export_plan(snapshot: TargetSnapshot) -> Dict[str, Any]:
    """Commands per layer plus an explicit status for every unavailable tool."""
    capability = toolchain_export_capability(snapshot)
    tools = capability["tools"]
    layers: List[Dict[str, Any]] = []

    def add(layer: str, command: Sequence[str], tool: str, note: str = "") -> None:
        available = bool(tools.get(tool, "")) if tool else True
        layers.append(
            {
                "layer": layer,
                "tool": tool or "in_process",
                "command": list(command),
                "available": available,
                "status": "READY" if available else "NOT_RUN_TOOL_UNAVAILABLE",
                "reason": "" if available else f"{tool} not found on PATH",
                "note": note,
            }
        )

    add("inductor_scheduler_ir", ["TORCH_COMPILE_DEBUG=1", "<driver>"], "", "env-driven dump")
    add("triton_ir", ["python3", "scripts/bench/dump_triton_ir.py"], "python3")
    add("llvm_ir", ["<inductor-cache>/*.ll"], "llvm-dis", "search the Inductor cache")
    add("ptx", ["cuobjdump", "-ptx", "<binary>"], "cuobjdump")
    add("cubin", ["cuobjdump", "-xelf", "all", "<binary>"], "cuobjdump")
    add("sass", ["cuobjdump", "-sass", "<binary>"], "cuobjdump")
    add("resource_report", ["cuobjdump", "-res-usage", "<binary>"], "cuobjdump")
    add("nsys_timeline", ["nsys", "profile", "-o", "<trace>", "--", "<driver>"], "nsys")
    add("ncu_report", ["ncu", "--set", "full", "-k", "<symbol>", "-o", "<report>", "<driver>"], "ncu")
    return {
        "target_id": snapshot.target_id,
        "layers": layers,
        "missing_tools": capability["missing"],
        "sass_binding_rule": (
            "every disassembly must be produced from the formal run binary hash; recompiling a "
            "'similar' artifact is not evidence"
        ),
    }


# ── binary resource report ─────────────────────────────────────────────────

RESOURCE_FIELDS: Tuple[str, ...] = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "local_bytes",
    "stack_frame_bytes",
    "spill_stores",
    "spill_loads",
    "code_size_bytes",
    "arch",
    "symbol",
)

_INT_RE = re.compile(r"(\d+)")


def _first_int(text: str) -> Optional[int]:
    match = _INT_RE.search(text)
    return int(match.group(1)) if match else None


def parse_resource_usage(text: str, *, symbol: str = "", arch: str = "") -> Dict[str, Any]:
    """Tolerant parser for ``cuobjdump -res-usage`` style output.

    Unparsable fields stay ``None`` with a reason; they are never defaulted to
    zero (a missing register count is very different from zero registers).
    """
    fields: Dict[str, Any] = {name: None for name in RESOURCE_FIELDS}
    fields["symbol"] = symbol
    fields["arch"] = arch
    patterns = {
        "registers_per_thread": r"REG:(\d+)",
        "static_shared_bytes": r"SMEM:(\d+)",
        "local_bytes": r"LOCAL:(\d+)",
        "stack_frame_bytes": r"STACK:(\d+)",
        "spill_stores": r"STL:(\d+)",
        "spill_loads": r"LDL:(\d+)",
    }
    for field_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            fields[field_name] = int(match.group(1))
    shared_match = re.search(r"shared=(\d+)", text)
    if shared_match:
        fields["static_shared_bytes"] = int(shared_match.group(1))
    if "SHARED" in text and fields["static_shared_bytes"] is None:
        match = re.search(r"SHARED:(\d+)", text)
        if match:
            fields["static_shared_bytes"] = int(match.group(1))
    missing = sorted(name for name in RESOURCE_FIELDS if fields.get(name) is None)
    return {
        "fields": fields,
        "missing_fields": missing,
        "parse_status": "OK" if len(missing) <= 2 else "PARTIAL",
        "note": "missing values are 'unknown', not zero",
    }


def occupancy_limits(
    *,
    registers_per_thread: Optional[int],
    static_shared_bytes: Optional[int],
    dynamic_shared_bytes: Optional[int] = None,
    shared_per_sm_bytes: Optional[int] = None,
    max_threads_per_sm: Optional[int] = None,
    max_registers_per_sm: Optional[int] = None,
    block_size: int = 0,
) -> Dict[str, Any]:
    """Occupancy *limits* only (occupancy is not an objective by itself)."""
    limits: List[Dict[str, Any]] = []
    if registers_per_thread and max_registers_per_sm and block_size:
        per_block = registers_per_thread * block_size
        blocks = max_registers_per_sm // max(1, per_block)
        limits.append({"limiter": "registers", "blocks_per_sm": blocks})
    if static_shared_bytes and shared_per_sm_bytes:
        shared_total = static_shared_bytes + int(dynamic_shared_bytes or 0)
        if shared_total > 0:
            limits.append(
                {"limiter": "shared_memory", "blocks_per_sm": shared_per_sm_bytes // shared_total}
            )
    if max_threads_per_sm and block_size:
        limits.append(
            {"limiter": "threads", "blocks_per_sm": max_threads_per_sm // block_size}
        )
    return {
        "limits": limits,
        "estimated_blocks_per_sm": min(
            (row["blocks_per_sm"] for row in limits), default=None
        ),
        "warning": (
            "higher occupancy is not automatically faster; interpret together with stall and "
            "throughput counters"
        ),
    }


# ── ledgers ────────────────────────────────────────────────────────────────


@dataclass
class MemoryLedgerRow:
    buffer_id: str
    producer: str
    consumers: Tuple[str, ...]
    bytes: int
    layout: str
    lifetime: str = "kernel"
    allocated: bool = True
    eliminated: bool = False
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "buffer_id": self.buffer_id,
            "producer": self.producer,
            "consumers": list(self.consumers),
            "bytes": self.bytes,
            "layout": self.layout,
            "lifetime": self.lifetime,
            "allocated": self.allocated,
            "eliminated": self.eliminated,
            "reason": self.reason,
        }


def memory_plan_diff(
    before: Sequence[MemoryLedgerRow], after: Sequence[MemoryLedgerRow]
) -> Dict[str, Any]:
    before_map = {row.buffer_id: row for row in before}
    after_map = {row.buffer_id: row for row in after}
    eliminated = sorted(set(before_map) - set(after_map))
    added = sorted(set(after_map) - set(before_map))
    before_bytes = sum(row.bytes for row in before if row.allocated)
    after_bytes = sum(row.bytes for row in after if row.allocated)
    return {
        "eliminated_buffers": eliminated,
        "added_buffers": added,
        "bytes_before": before_bytes,
        "bytes_after": after_bytes,
        "bytes_saved": before_bytes - after_bytes,
        "saving_ratio": round((before_bytes - after_bytes) / before_bytes, 4)
        if before_bytes
        else None,
        "rows": [row.as_dict() for row in after],
    }


@dataclass
class LaunchLedgerRow:
    sequence: int
    kind: str  # kernel | copy | allocation | sync
    symbol: str
    grid: Tuple[int, ...] = ()
    stream: str = "current"
    phase: str = ""
    observed: bool = False
    duration_us: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "symbol": self.symbol,
            "grid": list(self.grid),
            "stream": self.stream,
            "phase": self.phase,
            "observed": self.observed,
            "duration_us": self.duration_us,
        }


def launch_ledger_diff(
    expected: Sequence[LaunchLedgerRow], observed: Sequence[LaunchLedgerRow]
) -> Dict[str, Any]:
    """Compare expected launches with the profiled timeline (step 15)."""
    expected_counts = _count_by_symbol(expected)
    observed_counts = _count_by_symbol(observed)
    missing = sorted(set(expected_counts) - set(observed_counts))
    extra = sorted(set(observed_counts) - set(expected_counts))
    changed = sorted(
        symbol
        for symbol in set(expected_counts) & set(observed_counts)
        if expected_counts[symbol] != observed_counts[symbol]
    )
    return {
        "expected": expected_counts,
        "observed": observed_counts,
        "missing_symbols": missing,
        "extra_symbols": extra,
        "count_mismatch": changed,
        "balanced": not missing and not extra and not changed,
        "rows_observed": [row.as_dict() for row in observed],
    }


def _count_by_symbol(rows: Iterable[LaunchLedgerRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        if row.kind == "kernel":
            counts[row.symbol] = counts.get(row.symbol, 0) + 1
    return dict(sorted(counts.items()))


# ── mechanism hypotheses and ablation ──────────────────────────────────────

MECHANISM_VERDICTS: Tuple[str, ...] = (
    "SUPPORTED",
    "SUPPORTED_WITH_LIMITS",
    "INCONCLUSIVE",
    "REFUTED",
)

MECHANISM_TEMPLATE: Tuple[str, ...] = (
    "mechanism_id",
    "case_pair",
    "changed_factor",
    "ir_prediction",
    "generated_code_observation",
    "binary_observation",
    "counter_prediction",
    "counter_observation",
    "runtime_observation",
    "ablation_observation",
    "alternative_explanations",
    "confidence",
    "verdict",
    "artifact_refs",
)


@dataclass
class MechanismHypothesis:
    """One mechanism row (E11-04 §10)."""

    mechanism_id: str
    case_pair: str
    changed_factor: str
    ir_prediction: str
    generated_code_observation: str = ""
    binary_observation: str = ""
    counter_prediction: str = ""
    counter_observation: str = ""
    runtime_observation: str = ""
    ablation_observation: str = ""
    alternative_explanations: Tuple[str, ...] = ()
    refuting_observation: str = ""
    confidence: str = "low"
    verdict: str = "INCONCLUSIVE"
    artifact_refs: Tuple[str, ...] = ()

    def evaluate(self) -> "MechanismHypothesis":
        """Apply the minimum-evidence rule for each verdict.

        A mechanism is only *refuted* when a counter-observation is recorded;
        "no evidence yet" is INCONCLUSIVE, not a negative result.
        """
        layers = [
            bool(self.generated_code_observation),
            bool(self.binary_observation),
            bool(self.counter_observation),
            bool(self.runtime_observation),
        ]
        strong = sum(layers) >= 3 and bool(self.ablation_observation)
        partial = sum(layers) >= 2
        if self.refuting_observation:
            self.verdict = "REFUTED"
            self.confidence = "medium"
        elif strong:
            self.verdict = "SUPPORTED"
            self.confidence = "high"
        elif partial and self.alternative_explanations:
            self.verdict = "SUPPORTED_WITH_LIMITS"
            self.confidence = "medium"
        else:
            self.verdict = "INCONCLUSIVE"
            self.confidence = "low"
        return self

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.verdict not in MECHANISM_VERDICTS:
            problems.append(f"unknown verdict {self.verdict!r}")
        if self.verdict == "SUPPORTED" and not self.alternative_explanations:
            problems.append(
                f"{self.mechanism_id}: SUPPORTED requires the alternative explanations it ruled out"
            )
        if self.verdict in ("SUPPORTED", "SUPPORTED_WITH_LIMITS") and not self.artifact_refs:
            problems.append(f"{self.mechanism_id}: a supported mechanism needs artifact refs")
        if self.verdict == "REFUTED" and not self.refuting_observation:
            problems.append(f"{self.mechanism_id}: REFUTED requires the refuting observation")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "case_pair": self.case_pair,
            "changed_factor": self.changed_factor,
            "ir_prediction": self.ir_prediction,
            "generated_code_observation": self.generated_code_observation,
            "binary_observation": self.binary_observation,
            "counter_prediction": self.counter_prediction,
            "counter_observation": self.counter_observation,
            "runtime_observation": self.runtime_observation,
            "ablation_observation": self.ablation_observation,
            "alternative_explanations": list(self.alternative_explanations),
            "refuting_observation": self.refuting_observation,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "artifact_refs": list(self.artifact_refs),
        }


def mechanism_verdict_table(rows: Sequence[MechanismHypothesis]) -> Dict[str, Any]:
    evaluated = [row.evaluate() for row in rows]
    problems = [problem for row in evaluated for problem in row.validate()]
    supported = [row.mechanism_id for row in evaluated if row.verdict == "SUPPORTED"]
    return {
        "rows": [row.as_dict() for row in evaluated],
        "supported": supported,
        "at_least_one_supported": bool(supported),
        "problems": problems,
        "rule": (
            "correlation across one counter is not causality: each mechanism needs the "
            "IR→binary→counter→runtime chain plus an ablation and its alternatives"
        ),
    }


@dataclass
class AblationCase:
    """One single-factor configuration (steps 3, 27)."""

    case_id: str
    factors: Mapping[str, Any]
    latency_ms: Optional[float] = None
    correctness_ok: bool = False
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "factors": dict(sorted(self.factors.items())),
            "latency_ms": self.latency_ms,
            "correctness_ok": self.correctness_ok,
            "note": self.note,
        }


def ablation_matrix(
    base: Mapping[str, Any], *, factors: Mapping[str, Sequence[Any]]
) -> List[AblationCase]:
    """Enumerate single-factor changes from a frozen base configuration."""
    rows: List[AblationCase] = []
    rows.append(AblationCase(case_id="base", factors=dict(base)))
    for name, values in factors.items():
        for value in values:
            factors_new = dict(base)
            factors_new[name] = value
            rows.append(AblationCase(case_id=f"{name}={value}", factors=factors_new))
    return rows


def evaluate_ablation(cases: Sequence[AblationCase]) -> Dict[str, Any]:
    """Check that only one factor moved and collect the response trend."""
    base = next((case for case in cases if case.case_id == "base"), None)
    if base is None:
        raise ConfigError("ablation matrix needs a 'base' case")
    trends: List[Dict[str, Any]] = []
    problems: List[str] = []
    for case in cases:
        if case is base:
            continue
        changed = [key for key in base.factors if base.factors[key] != case.factors.get(key)]
        if len(changed) != 1:
            problems.append(f"{case.case_id}: {len(changed)} factors changed (need exactly 1)")
        delta = (
            None
            if case.latency_ms is None or base.latency_ms is None
            else round(case.latency_ms - base.latency_ms, 6)
        )
        trends.append(
            {
                "case_id": case.case_id,
                "changed_factor": changed[0] if len(changed) == 1 else ",".join(changed),
                "delta_ms": delta,
                "correctness_ok": case.correctness_ok,
            }
        )
    return {
        "base": base.as_dict(),
        "trends": trends,
        "problems": problems,
        "rule": "ablation supports a mechanism only if the single-factor trend matches its prediction",
    }


# ── Amdahl attribution and profiling plan ──────────────────────────────────


def amdahl_attribution(
    *,
    kernel_saving_ms: float,
    call_share_in_phase: float,
    phase_latency_ms: float,
    measured_phase_saving_ms: Optional[float],
    fallback_rate: float = 0.0,
    unrelated_launch_overhead_ms: float = 0.0,
) -> Dict[str, Any]:
    """Predict the phase saving from a kernel saving and report the residual."""
    predicted = kernel_saving_ms * call_share_in_phase * (1.0 - fallback_rate)
    predicted -= unrelated_launch_overhead_ms
    residual = (
        None if measured_phase_saving_ms is None else round(measured_phase_saving_ms - predicted, 6)
    )
    return {
        "kernel_saving_ms": kernel_saving_ms,
        "call_share_in_phase": call_share_in_phase,
        "fallback_rate": fallback_rate,
        "predicted_phase_saving_ms": round(predicted, 6),
        "measured_phase_saving_ms": measured_phase_saving_ms,
        "residual_ms": residual,
        "phase_latency_ms": phase_latency_ms,
        "attribution_ok": (
            residual is not None
            and abs(residual) <= max(0.05 * abs(predicted), 1e-3)
        )
        if residual is not None
        else None,
        "note": (
            "a kernel that is faster while the model is unchanged is allowed; it must be "
            "explained by call share, graph breaks, fallback or Amdahl limits — not reported "
            "as a model speedup"
        ),
    }


@dataclass
class ProfilePlan:
    """Nsight/target profiler commands plus replay-risk handling (steps 16–21)."""

    target: str
    symbols: Tuple[str, ...]
    mutates_state: bool = False
    metric_set: str = "full"
    report_timing_is_primary: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.symbols:
            problems.append("profiling without a kernel symbol cannot map counters to a case")
        if self.report_timing_is_primary:
            problems.append(
                "profiler timing must not be the primary performance number (replay overhead)"
            )
        return problems

    def commands(self) -> List[List[str]]:
        commands: List[List[str]] = [
            ["nsys", "profile", "-o", "<run>/profile/timeline", "--", "<driver>"]
        ]
        for symbol in self.symbols:
            commands.append(
                [
                    "ncu",
                    "--set",
                    self.metric_set,
                    "-k",
                    symbol,
                    "-o",
                    f"<run>/profile/{symbol}",
                    "<driver>",
                ]
            )
        if not self.mutates_state:
            for symbol in self.symbols:
                commands.append(
                    [
                        "ncu",
                        "--set",
                        self.metric_set,
                        "--replay-mode",
                        "kernel",
                        "-k",
                        symbol,
                        "<driver>",
                    ]
                )
        return commands

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "symbols": list(self.symbols),
            "mutates_state": self.mutates_state,
            "metric_set": self.metric_set,
            "commands": self.commands(),
            "replay_risk": (
                "state-mutating kernels need isolated inputs/reset per replay; otherwise the "
                "profile changes the result"
                if self.mutates_state
                else "no explicit replay hazard declared"
            ),
            "problems": self.validate(),
        }


# ── reconstruction (step 31, §11 H3) ──────────────────────────────────────


def reconstruction_plan(
    *, manifest_id: str, toolchain_versions: Mapping[str, str], source_artifacts: Sequence[str]
) -> Dict[str, Any]:
    return {
        "manifest_id": manifest_id,
        "toolchain": dict(sorted(toolchain_versions.items())),
        "source_artifacts": list(source_artifacts),
        "steps": [
            "re-read manifest and verify hashes",
            "rebuild from IR + config + toolchain flags",
            "compare semantic/canonical hashes of generated code",
            "compare resource report and representative timings",
        ],
        "binary_nondeterminism_note": (
            "bytewise-identical binaries are not guaranteed; a rebuild must explain differences "
            "in compiler metadata/paths and confirm the machine code is equivalent"
        ),
    }


def reconstruction_check(
    *,
    manifest_id: str,
    original: Mapping[str, Any],
    rebuilt: Mapping[str, Any],
    tolerance_fields: Sequence[str] = ("canonical_hash", "resource_report", "timing_class"),
) -> Dict[str, Any]:
    """Compare a rebuild against the original artifact at declared tolerance."""
    diffs = [
        {
            "field": field,
            "original": original.get(field),
            "rebuilt": rebuilt.get(field),
        }
        for field in tolerance_fields
        if original.get(field) != rebuilt.get(field)
    ]
    return {
        "manifest_id": manifest_id,
        "compared_fields": list(tolerance_fields),
        "diffs": diffs,
        "semantic_rebuild_ok": not any(
            item["field"] in ("canonical_hash",) for item in diffs
        ),
        "bitwise_identical": original.get("raw_hash") == rebuilt.get("raw_hash"),
        "note": (
            "a different binary hash with identical semantics/resource/timing class is a "
            "semantic rebuild, not a bitwise reproduction"
        ),
    }


def cross_layer_diff_rows(
    *,
    graph_diff: Mapping[str, Any],
    generated_source: Sequence[GeneratedSourceArtifact],
    resource_before: Mapping[str, Any],
    resource_after: Mapping[str, Any],
    timing_before_ms: Optional[float],
    timing_after_ms: Optional[float],
) -> List[Dict[str, Any]]:
    """One table row per observed change, layer by layer (step 13)."""
    rows: List[Dict[str, Any]] = []
    for key in ("ops_added", "ops_removed", "semantic_op_changes", "constraints_added"):
        value = graph_diff.get(key)
        if value:
            rows.append({"layer": "ir", "change": key, "value": value})
    rows.append(
        {
            "layer": "generated_source",
            "change": "artifacts",
            "value": [item.artifact_id for item in generated_source],
        }
    )
    for key in ("registers_per_thread", "local_bytes", "spill_loads"):
        if resource_before.get(key) != resource_after.get(key):
            rows.append(
                {
                    "layer": "binary",
                    "change": key,
                    "value": {"before": resource_before.get(key), "after": resource_after.get(key)},
                }
            )
    if timing_before_ms is not None and timing_after_ms is not None:
        rows.append(
            {
                "layer": "runtime",
                "change": "latency_ms",
                "value": {"before": timing_before_ms, "after": timing_after_ms},
            }
        )
    return rows


def code_size_metrics(artifacts: Sequence[GeneratedSourceArtifact]) -> Dict[str, Any]:
    sizes = {item.artifact_id: len(item.source_text) for item in artifacts}
    return {
        "sizes_chars": sizes,
        "total_chars": sum(sizes.values()),
        "artifact_count": len(artifacts),
    }


def hash_all_sources(artifacts: Sequence[GeneratedSourceArtifact]) -> Dict[str, Any]:
    digests = {item.artifact_id: item.hashes()["canonical_hash"] for item in artifacts}
    return {
        "digests": digests,
        "combined": sha256_text(canonical_json(digests)),
    }
