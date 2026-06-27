"""Graph capture, graph breaks, source/shape/effect metadata and coverage.

Protocol anchor: ``details/S11/E11-01_qwen_graph_capture_break_guard.md`` — the
front-end fact floor of S11.  It builds the instruments that turn "the model
compiles" into an explainable compiler boundary:

* capture requests over the region/mode/workload/layout/backend axes (steps 1–4);
* an eager execution census that supplies the *denominator* of every coverage
  metric (step 5);
* a no-rewrite debug backend that isolates capture failure from codegen failure
  (step 6);
* graph serialisation/canonicalisation (step 7), source lineage (step 8),
  tensor metadata (step 9) and effect metadata (step 10);
* the pre-registered ordered shape trace (step 18) and native framework traces
  (step 26) with a correlation join (step 27);
* coverage and metadata-completeness accounting (step 28);
* determinism/cross-process repeatability checks (steps 29–30) and the hero
  graph handoff manifest (step 31).

This module never executes a model.  The ``DebugBackend`` is a *contract* that
the driver can hand to ``torch.compile``; the recording callbacks are injected,
so the module stays importable without torch (CPU-minimal rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text

# ── capture axes (E11-01 §5.1 / §7) ────────────────────────────────────────

MODE_DYNAMO_DEBUG = "dynamo_debug_backend"
MODE_COMPILE_DEFAULT = "compile_default"
MODE_FULLGRAPH = "fullgraph_probe"
MODE_EXPORT_STRICT = "export_strict"
MODE_EXPORT_NON_STRICT = "export_non_strict"

CAPTURE_MODES: Tuple[str, ...] = (
    MODE_DYNAMO_DEBUG,
    MODE_COMPILE_DEFAULT,
    MODE_FULLGRAPH,
    MODE_EXPORT_STRICT,
    MODE_EXPORT_NON_STRICT,
)

REGIONS: Tuple[str, ...] = ("operator", "block", "prefill", "decode_step", "decode_multi_step")

WORKLOADS: Tuple[str, ...] = (
    "tiny",
    "short",
    "balanced",
    "long_prefill",
    "decode_heavy",
    "long_balanced",
)

LAYOUT_CASES: Tuple[str, ...] = ("contiguous", "valid_non_contiguous", "unsupported_layout")

BACKEND_MODES: Tuple[str, ...] = ("eager", "custom_op_enabled", "quantized")

REPEAT_CASES: Tuple[str, ...] = ("same_process_repeat", "cold_process_repeat")

# ── break taxonomy (E11-01 §3.2 + step 25) ─────────────────────────────────

BREAK_REASON_CATALOG: Mapping[str, str] = {
    "CAPTURE_UNSUPPORTED_PYTHON": "Python construct the tracer cannot record",
    "CAPTURE_GRAPH_BREAK": "generic graph break raised by the tracer",
    "EXPORT_CONSTRAINT": "export cannot prove the input satisfies its constraints",
    "DATA_DEPENDENT_BRANCH": "control flow depends on tensor values",
    "MISSING_FAKE_META": "custom op lacks a fake/meta implementation",
    "SIDE_EFFECT_ORDER": "state/IO side effect would be reordered",
    "EXPLICIT_DISABLE": "capture explicitly disabled for this region",
    "COMPILER_BUG_WORKAROUND": "known compiler limitation worked around by a break",
    "UNKNOWN_BREAK": "break could not be attributed (never acceptable in a report)",
}

BREAK_FALLBACK_ACTIONS: Tuple[str, ...] = ("eager_island", "reference_kernel", "error")


@dataclass
class CaptureRequest:
    """One cell of the E11-01 capture matrix (frozen before running)."""

    request_id: str
    region: str
    capture_mode: str
    workload: str
    layout_case: str = "contiguous"
    backend_mode: str = "eager"
    repeat: str = "same_process_repeat"
    dtype: str = "fp16"
    batch: int = 1
    isl: int = 128
    kv_len: int = 0
    decode_step: int = 0
    notes: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        checks = (
            ("region", self.region, REGIONS),
            ("capture_mode", self.capture_mode, CAPTURE_MODES),
            ("workload", self.workload, WORKLOADS),
            ("layout_case", self.layout_case, LAYOUT_CASES),
            ("backend_mode", self.backend_mode, BACKEND_MODES),
            ("repeat", self.repeat, REPEAT_CASES),
        )
        for name, value, allowed in checks:
            if value not in allowed:
                problems.append(f"unknown {name} {value!r}")
        if self.region == "operator":
            problems.append("operator-only cells are debugging aids, not P0 acceptance cells")
        return problems


def capture_matrix(
    *, regions: Sequence[str] = REGIONS, modes: Sequence[str] = CAPTURE_MODES
) -> List[CaptureRequest]:
    """Enumerate the required matrix cells (success *and* negative cells)."""
    rows: List[CaptureRequest] = []
    for region in regions:
        for mode in modes:
            for workload in WORKLOADS:
                if region == "operator" and workload != "tiny":
                    continue
                rows.append(
                    CaptureRequest(
                        request_id=f"cap_{region}_{mode}_{workload}",
                        region=region,
                        capture_mode=mode,
                        workload=workload,
                    )
                )
    for layout in ("valid_non_contiguous", "unsupported_layout"):
        rows.append(
            CaptureRequest(
                request_id=f"cap_layout_{layout}",
                region="block",
                capture_mode=MODE_DYNAMO_DEBUG,
                workload="short",
                layout_case=layout,
            )
        )
    return rows


# ── break records and audit ────────────────────────────────────────────────


@dataclass
class GraphBreak:
    """One graph break with full localisation (E11-01 §10 ``graph_breaks``)."""

    break_id: str
    frame_id: str
    reason_code: str
    native_reason: str = ""
    unsupported_feature: str = ""
    source_file: str = ""
    source_line: int = 0
    function: str = ""
    module_path: str = ""
    graph_before_id: str = ""
    graph_after_id: str = ""
    eager_region_id: str = ""
    state_boundary: str = ""
    fallback_action: str = "eager_island"
    raw_trace_ref: str = ""
    workload: str = ""
    phase: str = ""
    step: int = 0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.reason_code not in BREAK_REASON_CATALOG:
            problems.append(f"unknown break reason {self.reason_code!r}")
        if self.reason_code == "UNKNOWN_BREAK":
            problems.append(
                "UNKNOWN_BREAK is not an acceptable report row: a break must be localised"
            )
        if not (self.source_file and self.source_line):
            problems.append("break without file/line cannot be traced back to source")
        if not self.native_reason:
            problems.append("native tracer reason string must be preserved verbatim")
        if self.fallback_action not in BREAK_FALLBACK_ACTIONS:
            problems.append(f"unknown fallback action {self.fallback_action!r}")
        if not self.graph_before_id and not self.graph_after_id:
            problems.append("break must record the graph boundary on at least one side")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "break_id": self.break_id,
            "frame_id": self.frame_id,
            "workload": self.workload,
            "phase": self.phase,
            "step": self.step,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "function": self.function,
            "module_path": self.module_path,
            "reason_code": self.reason_code,
            "native_reason": self.native_reason,
            "unsupported_feature": self.unsupported_feature,
            "graph_before_id": self.graph_before_id,
            "graph_after_id": self.graph_after_id,
            "eager_region_id": self.eager_region_id,
            "state_boundary": self.state_boundary,
            "fallback_action": self.fallback_action,
            "raw_trace_ref": self.raw_trace_ref,
        }


@dataclass
class BreakAudit:
    """Break localisation audit: ``unlocated`` must be empty for a PASS."""

    rows: Tuple[GraphBreak, ...]

    def problems(self) -> List[Dict[str, str]]:
        return [
            {"break_id": row.break_id, "problem": problem}
            for row in self.rows
            for problem in row.validate()
        ]

    def by_reason(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self.rows:
            counts[row.reason_code] = counts.get(row.reason_code, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> Dict[str, Any]:
        return {
            "break_count": len(self.rows),
            "by_reason": self.by_reason(),
            "unlocated": [row.break_id for row in self.rows if not (row.source_file and row.source_line)],
            "unknown_reason": [
                row.break_id for row in self.rows if row.reason_code == "UNKNOWN_BREAK"
            ],
            "problems": self.problems(),
            "passes_localisation": not self.problems(),
        }


# ── metadata records (steps 8–10) ──────────────────────────────────────────

METADATA_DIMENSIONS: Tuple[str, ...] = ("source", "shape", "layout", "effect")

MISSING_REASONS: Tuple[str, ...] = (
    "not_exposed_by_framework",
    "custom_op_without_schema",
    "broken_stack_trace",
    "symbolic_only",
    "not_collected",
)


@dataclass
class SourceLineage:
    """Per-node source lineage (step 8)."""

    node_id: str
    module_qualified_name: str = ""
    file: str = ""
    line: int = 0
    function: str = ""
    native_op: str = ""
    frame_id: str = ""
    graph_id: str = ""
    stack: Tuple[str, ...] = ()
    missing_reason: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not (self.file and self.line):
            if self.missing_reason not in MISSING_REASONS:
                problems.append(
                    "missing source location requires an explicit missing_reason "
                    "(silence is not a metadata state)"
                )
        if not self.native_op:
            problems.append("original ATen/custom op target must be recorded")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "module_qualified_name": self.module_qualified_name,
            "file": self.file,
            "line": self.line,
            "function": self.function,
            "native_op": self.native_op,
            "frame_id": self.frame_id,
            "graph_id": self.graph_id,
            "stack": list(self.stack),
            "missing_reason": self.missing_reason,
        }


@dataclass
class TensorMetadataRecord:
    """Per-tensor metadata (step 9) with fake/runtime cross-check."""

    tensor_id: str
    dtype: str = ""
    shape: Tuple[Optional[int], ...] = ()
    symbolic_expression: Tuple[str, ...] = ()
    symbolic_range: Tuple[Tuple[Optional[int], Optional[int]], ...] = ()
    stride: Tuple[Optional[int], ...] = ()
    layout: str = ""
    device: str = ""
    requires_grad: bool = False
    storage_offset: Optional[int] = None
    fake_source: str = ""
    runtime_sampled: bool = False

    def completeness(self) -> Dict[str, bool]:
        return {
            "dtype": bool(self.dtype),
            "shape": bool(self.shape) or bool(self.symbolic_expression),
            "symbolic_range": all(
                lower is not None and upper is not None for lower, upper in self.symbolic_range
            )
            if self.symbolic_range
            else False,
            "stride": bool(self.stride),
            "layout": bool(self.layout),
            "device": bool(self.device),
            "runtime_sampled": self.runtime_sampled,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "symbolic_expression": list(self.symbolic_expression),
            "symbolic_range": [list(item) for item in self.symbolic_range],
            "stride": list(self.stride),
            "layout": self.layout,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "storage_offset": self.storage_offset,
            "fake_source": self.fake_source,
            "runtime_sampled": self.runtime_sampled,
        }


def compare_fake_runtime(
    fake: TensorMetadataRecord, runtime: TensorMetadataRecord, *, strict_stride: bool = True
) -> Dict[str, Any]:
    """Compare fake metadata against a real runtime sample (step 9).

    Stride is compared only when the caller declares it is contractually equal
    (some ops legitimately return different but compatible strides).
    """
    fields = ("dtype", "shape", "layout", "device")
    mismatches = [
        {"field": name, "fake": getattr(fake, name), "runtime": getattr(runtime, name)}
        for name in fields
        if getattr(fake, name) != getattr(runtime, name)
    ]
    if strict_stride and fake.stride != runtime.stride:
        mismatches.append({"field": "stride", "fake": fake.stride, "runtime": runtime.stride})
    return {
        "tensor_id": fake.tensor_id,
        "mismatches": mismatches,
        "ok": not mismatches,
        "stride_compared": strict_stride,
    }


@dataclass
class MetadataCompleteness:
    """Metadata completeness by dimension (step 28)."""

    source_ok: int = 0
    source_total: int = 0
    shape_ok: int = 0
    shape_total: int = 0
    layout_ok: int = 0
    layout_total: int = 0
    effect_ok: int = 0
    effect_total: int = 0
    unknown_effects: Tuple[str, ...] = ()
    missing_reasons: Mapping[str, int] = field(default_factory=dict)

    def rates(self) -> Dict[str, Any]:
        def rate(ok: int, total: int) -> Optional[float]:
            return None if total == 0 else round(ok / total, 4)

        return {
            "source": rate(self.source_ok, self.source_total),
            "shape": rate(self.shape_ok, self.shape_total),
            "layout": rate(self.layout_ok, self.layout_total),
            "effect": rate(self.effect_ok, self.effect_total),
            "effect_denominator_note": (
                "effect_total counts nodes with a schema/analysis record; nodes with "
                "UNKNOWN_UNSAFE are reported explicitly, not silently counted as complete"
            ),
            "unknown_effects": list(self.unknown_effects),
            "missing_reasons": dict(sorted(self.missing_reasons.items())),
        }

    def meets(
        self, thresholds: Mapping[str, float]
    ) -> Dict[str, Any]:
        rates = self.rates()
        failures = [
            {"dimension": name, "rate": rates[name], "threshold": threshold}
            for name, threshold in sorted(thresholds.items())
            if rates.get(name) is not None and rates[name] < threshold
        ]
        return {"ok": not failures, "failures": failures, "rates": rates}

    @classmethod
    def from_records(
        cls,
        *,
        lineages: Sequence[SourceLineage] = (),
        tensors: Sequence[TensorMetadataRecord] = (),
        effect_states: Sequence[str] = (),
    ) -> "MetadataCompleteness":
        report = cls()
        report.source_total = len(lineages)
        report.source_ok = sum(1 for row in lineages if not row.validate())
        missing: Dict[str, int] = {}
        for row in lineages:
            if row.missing_reason:
                missing[row.missing_reason] = missing.get(row.missing_reason, 0) + 1
        report.missing_reasons = missing
        report.shape_total = len(tensors)
        report.shape_ok = sum(1 for row in tensors if row.completeness()["shape"])
        report.layout_total = len(tensors)
        report.layout_ok = sum(1 for row in tensors if row.completeness()["layout"])
        report.effect_total = len(effect_states)
        report.effect_ok = sum(1 for state in effect_states if state != "UNKNOWN_UNSAFE")
        report.unknown_effects = tuple(
            f"effect[{index}]" for index, state in enumerate(effect_states) if state == "UNKNOWN_UNSAFE"
        )
        return report


# ── coverage (E11-01 §6) ───────────────────────────────────────────────────

COVERAGE_METRICS: Tuple[str, ...] = (
    "op_count_coverage",
    "weighted_time_coverage",
    "hotspot_coverage",
    "phase_coverage",
    "pattern_opportunity_coverage",
)


@dataclass
class CoverageReport:
    """Coverage metrics; every ratio requires an explicit denominator."""

    captured_ops: int = 0
    observed_ops: int = 0
    captured_region_time_s: float = 0.0
    eager_baseline_time_s: float = 0.0
    captured_hotspot_time_s: float = 0.0
    total_hotspot_time_s: float = 0.0
    captured_phase_time: Mapping[str, float] = field(default_factory=dict)
    phase_time: Mapping[str, float] = field(default_factory=dict)
    capturable_sites: int = 0
    semantic_sites: int = 0
    unavailable: Mapping[str, str] = field(default_factory=dict)

    def _ratio(self, numerator: float, denominator: float, key: str) -> Optional[float]:
        if key in self.unavailable:
            return None
        if denominator <= 0:
            return None
        return round(numerator / denominator, 4)

    def metrics(self) -> Dict[str, Any]:
        rows = {
            "op_count_coverage": self._ratio(self.captured_ops, self.observed_ops, "op_count_coverage"),
            "weighted_time_coverage": self._ratio(
                self.captured_region_time_s, self.eager_baseline_time_s, "weighted_time_coverage"
            ),
            "hotspot_coverage": self._ratio(
                self.captured_hotspot_time_s, self.total_hotspot_time_s, "hotspot_coverage"
            ),
            "pattern_opportunity_coverage": self._ratio(
                self.capturable_sites, self.semantic_sites, "pattern_opportunity_coverage"
            ),
        }
        return rows

    def phase_coverage(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for phase, total in sorted(self.phase_time.items()):
            out[phase] = self._ratio(self.captured_phase_time.get(phase, 0.0), total, "phase_coverage")
        return out

    def claim_rules(self) -> Dict[str, Any]:
        """Structural reminder of what each metric may and may not support."""
        return {
            "op_count_coverage": "may not be substituted for time/hotspot coverage",
            "weighted_time_coverage": "attribution of eager baseline time, not a speedup",
            "hotspot_coverage": "requires S02 hotspot table (else UNAVAILABLE)",
            "phase_coverage": "prefill and decode are reported separately",
            "pattern_opportunity_coverage": "requires the semantic candidate census",
            "unavailable": dict(sorted(self.unavailable.items())),
        }


# ── eager census (step 5) ──────────────────────────────────────────────────


@dataclass
class EagerCensusEntry:
    module_path: str
    op_name: str
    dtype: str
    shape: Tuple[Optional[int], ...]
    stride: Tuple[Optional[int], ...]
    device: str
    time_us: float
    source_file: str = ""
    source_line: int = 0
    call_index: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module_path": self.module_path,
            "op_name": self.op_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "device": self.device,
            "time_us": self.time_us,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "call_index": self.call_index,
        }


@dataclass
class EagerCensus:
    """Eager execution denominator: ops, time and hotspots (step 5)."""

    entries: List[EagerCensusEntry] = field(default_factory=list)
    measurement_method: str = ""

    def add(self, entry: EagerCensusEntry) -> None:
        self.entries.append(entry)

    @property
    def ready(self) -> bool:
        return bool(self.entries) and bool(self.measurement_method)

    def to_coverage(self, captured_ops: int, captured_time_s: float) -> CoverageReport:
        if not self.ready:
            raise ConfigError(
                "eager census incomplete: coverage without a measured denominator is UNAVAILABLE"
            )
        return CoverageReport(
            captured_ops=captured_ops,
            observed_ops=len(self.entries),
            captured_region_time_s=captured_time_s,
            eager_baseline_time_s=sum(row.time_us for row in self.entries) / 1e6,
        )

    def hotspot_rows(self, top_k: int = 20) -> List[Dict[str, Any]]:
        ranked = sorted(self.entries, key=lambda row: row.time_us, reverse=True)[:top_k]
        return [row.as_dict() for row in ranked]


# ── debug backend (step 6) ─────────────────────────────────────────────────

BACKEND_CONTRACT_NOTE = (
    "PyTorch custom backends receive (GraphModule, example_inputs) and must return "
    "a callable equivalent to the original graph.  The debug backend records and "
    "returns gm.forward unchanged; it must never be reported as a compiled path."
)


@dataclass
class DebugBackendResult:
    graph_id: str
    recorded: bool
    returned_callable: str
    is_debug_backend: bool = True
    rewrites_applied: int = 0
    lowerings_selected: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "recorded": self.recorded,
            "returned_callable": self.returned_callable,
            "is_debug_backend": self.is_debug_backend,
            "rewrites_applied": self.rewrites_applied,
            "lowerings_selected": self.lowerings_selected,
            "note": BACKEND_CONTRACT_NOTE,
        }


class DebugBackend:
    """Capture-only backend: records the graph and returns ``gm.forward``.

    Callables are injected (``capture_fn``, ``metadata_fn``) so this object is
    usable from the driver and from tests without importing torch.  The result
    is explicitly flagged ``is_debug_backend=True``: returning ``gm.forward``
    is *not* a compiled backend (E11-03 §13).
    """

    def __init__(
        self,
        *,
        capture_fn: Optional[Callable[[Any, Sequence[Any]], Dict[str, Any]]] = None,
        metadata_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> None:
        self.capture_fn = capture_fn
        self.metadata_fn = metadata_fn
        self.captures: List[Dict[str, Any]] = []

    def __call__(self, graph_module: Any, example_inputs: Sequence[Any]) -> Any:
        record: Dict[str, Any] = {"module": type(graph_module).__name__}
        if self.capture_fn is not None:
            record.update(self.capture_fn(graph_module, example_inputs))
        if self.metadata_fn is not None:
            record["metadata"] = self.metadata_fn(graph_module)
        self.captures.append(record)
        return getattr(graph_module, "forward", graph_module)

    def result(self, graph_id: str) -> DebugBackendResult:
        return DebugBackendResult(
            graph_id=graph_id,
            recorded=bool(self.captures),
            returned_callable="gm.forward",
        )


# ── ordered shape trace (step 18) and native traces (step 26) ──────────────

TRACE_PHASES: Tuple[str, ...] = (
    "tiny",
    "short",
    "tiny_again",
    "balanced",
    "long_prefill",
    "decode_multi_step",
    "non_contiguous",
    "unsupported_dtype",
    "restore_compiled_shape",
    "cold_process_replay",
)

TRACE_EXPECTATIONS: Mapping[str, str] = {
    "tiny": "first static specialisation / cold compile",
    "short": "warm variant reuse or one new bucket",
    "tiny_again": "must reuse the tiny variant (no recompile)",
    "balanced": "phase switch prefill→decode",
    "long_prefill": "bucket/variant for long sequences",
    "decode_multi_step": "past-length growth must not recompile per token",
    "non_contiguous": "layout guard / copy / fallback",
    "unsupported_dtype": "explicit rejection or reference fallback before launch",
    "restore_compiled_shape": "return to an earlier shape must hit the existing variant",
    "cold_process_replay": "cross-process cache behaviour",
}


@dataclass
class ShapeTraceStep:
    phase: str
    shapes: Mapping[str, int]
    expect: str = ""
    request_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "request_id": self.request_id,
            "shapes": dict(sorted(self.shapes.items())),
            "expect": self.expect or TRACE_EXPECTATIONS.get(self.phase, ""),
        }


def ordered_shape_trace(steps: Sequence[ShapeTraceStep]) -> List[ShapeTraceStep]:
    """Validate the pre-registered order (an edited order invalidates the run)."""
    phases = [step.phase for step in steps]
    if tuples_equal(phases, TRACE_PHASES):
        return list(steps)
    missing = [phase for phase in TRACE_PHASES if phase not in phases]
    extra = [phase for phase in phases if phase not in TRACE_PHASES]
    raise ConfigError(
        "ordered shape trace must follow the pre-registered sequence",
        details={"missing": missing, "unexpected": extra, "expected": list(TRACE_PHASES)},
    )


def tuples_equal(left: Sequence[str], right: Sequence[str]) -> bool:
    return list(left) == list(right)


@dataclass
class NativeTracePlan:
    """Framework-native trace commands (step 26) with availability reasons."""

    torch_version: str
    trace_dir: str
    commands: Tuple[Tuple[str, ...], ...] = ()
    gaps: Tuple[str, ...] = ()

    @classmethod
    def for_torch(cls, torch_version: str, trace_dir: str = "native_trace/torch_trace") -> "NativeTracePlan":
        if not torch_version:
            return cls(torch_version="", trace_dir=trace_dir, gaps=("torch_version_unknown",))
        major = int(torch_version.split(".")[0])
        if major < 2:
            return cls(
                torch_version=torch_version,
                trace_dir=trace_dir,
                gaps=("TORCH_TRACE requires torch>=2.0",),
            )
        commands = (
            ("TORCH_TRACE", trace_dir, "python3", "<driver>", "--execute"),
            ("TORCH_LOGS", "graph_breaks,guards,recompiles,dynamic", "python3", "<driver>", "--execute"),
            ("tlparse", trace_dir, "--output", "native_trace/tlparse"),
        )
        return cls(torch_version=torch_version, trace_dir=trace_dir, commands=commands)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "trace_dir": self.trace_dir,
            "commands": [list(cmd) for cmd in self.commands],
            "gaps": list(self.gaps),
            "raw_note": (
                "native traces contain model source; they are raw evidence and must be "
                "audited before sharing"
            ),
        }


def join_native_and_hqsb(
    native_rows: Sequence[Mapping[str, Any]],
    hqsb_rows: Sequence[Mapping[str, Any]],
    *,
    key: str = "correlation_id",
) -> Dict[str, Any]:
    """Correlation join between framework trace rows and HQSB events (step 27)."""
    native_keys = {str(row.get(key, "")) for row in native_rows if row.get(key)}
    hqsb_keys = {str(row.get(key, "")) for row in hqsb_rows if row.get(key)}
    unmapped = sorted(hqsb_keys - native_keys)
    return {
        "native_rows": len(native_rows),
        "hqsb_rows": len(hqsb_rows),
        "matched": len(native_keys & hqsb_keys),
        "unmapped_hqsb": unmapped,
        "unmapped_ratio": round(len(unmapped) / max(1, len(hqsb_keys)), 4),
        "ok": not unmapped,
        "note": "every normalized row must be traceable back to the raw native trace",
    }


# ── repeatability (steps 29–30) and hero manifest (step 31) ────────────────


def repeatability_check(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Canonical hashes must agree; raw hashes may differ on noise only."""
    canonical = {str(row.get("canonical_hash", "")) for row in rows}
    raw = {str(row.get("raw_hash", "")) for row in rows}
    break_sets = {canonical_json(row.get("breaks", [])) for row in rows}
    guard_sets = {canonical_json(row.get("guards", [])) for row in rows}
    missing = [index for index, row in enumerate(rows) if not row.get("canonical_hash")]
    return {
        "runs": len(rows),
        "canonical_hashes": sorted(canonical),
        "raw_hashes_differ": len(raw) > 1,
        "canonical_stable": len(canonical) == 1 and not missing,
        "breaks_stable": len(break_sets) == 1,
        "guards_stable": len(guard_sets) == 1,
        "missing_canonical_hash": missing,
        "ok": len(canonical) == 1 and not missing and len(break_sets) == 1,
        "limitation": (
            "same-process repeats cannot prove cross-process stability; a cold-process "
            "repeat is required for H1"
        ),
    }


@dataclass
class HeroGraphManifest:
    """Handoff artifact for E11-02/E11-03 (step 31, §14)."""

    graph_id: str
    source_graph_id: str
    canonical_hash: str
    capture_mode: str
    region: str
    workload: str
    dtype: str
    constraints: Tuple[Mapping[str, Any], ...] = ()
    pattern_sites: Tuple[Mapping[str, Any], ...] = ()
    near_miss_sites: Tuple[Mapping[str, Any], ...] = ()
    extra_user_sites: Tuple[Mapping[str, Any], ...] = ()
    known_breaks: Tuple[str, ...] = ()
    unsupported_ops: Tuple[str, ...] = ()
    compiler_versions: Mapping[str, str] = field(default_factory=dict)
    eager_reference_uri: str = ""
    capture_scope_claim: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("graph_id", "source_graph_id", "canonical_hash", "capture_mode", "region"):
            if not getattr(self, name):
                problems.append(f"hero manifest missing {name!r}")
        if not self.pattern_sites:
            problems.append("hero graph must carry at least one real pattern site")
        if not self.constraints:
            problems.append("hero graph must carry its symbolic constraints")
        if not self.compiler_versions:
            problems.append("hero manifest must freeze compiler versions")
        if not self.capture_scope_claim:
            problems.append(
                "hero manifest must state the capture scope that may be claimed externally"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "source_graph_id": self.source_graph_id,
            "canonical_hash": self.canonical_hash,
            "capture_mode": self.capture_mode,
            "region": self.region,
            "workload": self.workload,
            "dtype": self.dtype,
            "constraints": [dict(row) for row in self.constraints],
            "pattern_sites": [dict(row) for row in self.pattern_sites],
            "near_miss_sites": [dict(row) for row in self.near_miss_sites],
            "extra_user_sites": [dict(row) for row in self.extra_user_sites],
            "known_breaks": list(self.known_breaks),
            "unsupported_ops": list(self.unsupported_ops),
            "compiler_versions": dict(sorted(self.compiler_versions.items())),
            "eager_reference_uri": self.eager_reference_uri,
            "capture_scope_claim": self.capture_scope_claim,
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.as_dict()))


# ── negative cases (steps 24–25) ───────────────────────────────────────────


def missing_fake_negative(
    *, op_name: str, frame_id: str = "neg_frame"
) -> GraphBreak:
    """Fixture: a custom op without fake/meta must fail capture explicitly."""
    return GraphBreak(
        break_id=f"neg_missing_fake_{op_name}",
        frame_id=frame_id,
        reason_code="MISSING_FAKE_META",
        native_reason=f"{op_name}: no fake/meta implementation registered",
        unsupported_feature="fake_tensor",
        source_file="tests/fixtures/negative_missing_fake.py",
        source_line=1,
        function="forward",
        module_path="fixture.negative",
        graph_before_id="g_before",
        fallback_action="error",
    )


def data_dependent_negative(*, source_file: str, source_line: int) -> GraphBreak:
    """Fixture: a value-dependent branch must be reported with its source line."""
    return GraphBreak(
        break_id="neg_data_dependent_branch",
        frame_id="neg_frame",
        reason_code="DATA_DEPENDENT_BRANCH",
        native_reason="tensor value used in `if` condition",
        unsupported_feature="data_dependent_control_flow",
        source_file=source_file,
        source_line=source_line,
        function="forward",
        module_path="fixture.negative",
        graph_after_id="g_after",
        fallback_action="eager_island",
    )


def coverage_claim_guard(metrics: Mapping[str, Any], *, claim: str) -> Dict[str, Any]:
    """Refuse a coverage claim that is not carried by the matching metric."""
    allowed = {
        "op_count": "op_count_coverage",
        "time": "weighted_time_coverage",
        "hotspot": "hotspot_coverage",
        "phase": "phase_coverage",
        "pattern": "pattern_opportunity_coverage",
    }
    if claim not in allowed:
        raise ConfigError(f"unknown coverage claim {claim!r}")
    key = allowed[claim]
    value = metrics.get(key)
    if value is None and claim == "phase":
        value = metrics.get("phase_coverage", {})
    blocked = value is None or (isinstance(value, Mapping) and not value)
    return {
        "claim": claim,
        "metric": key,
        "value": value,
        "allowed": not blocked,
        "reason": "" if not blocked else f"{key} is UNAVAILABLE; the claim cannot be made",
    }
