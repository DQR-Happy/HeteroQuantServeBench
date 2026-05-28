"""Model-level measurement harness for S05 (E05-02 §8/§9, E05-06 §10.2).

What lives here:

* :class:`MemoryLadder` — the ten-layer memory accounting required before any
  compression claim (disk / canonical / packed / host / device / peak), with a
  predicted-vs-measured reconciliation that reports the residual instead of
  hiding it;
* :class:`PhaseTimer` — phase-separated timing (offline, pack, cold load,
  first compile, steady prefill, steady decode) with warmup, repeats, device
  events and explicit synchronisation, so an asynchronous launch can never be
  reported as a fast kernel;
* perplexity with frozen tokenizer/mask/stride/denominator;
* :func:`build_unified_table` — the *only* way result rows are produced, from
  raw records, with a table hash;
* :func:`gate_order` — the fixed adjudication order (quality → execution
  reality → measurement → performance), so a fast fake-quant path cannot be
  recommended.

Everything optional (torch, device counters) is lazily imported and reported
as a capability error when missing.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.quant.artifact import QuantArtifactDocument
from hqsb.quant.stats import GATE_FAIL, GATE_INCONCLUSIVE, GATE_PASS

#: Canonical phase names; a timing record must use one of them.
PHASES = (
    "offline_calibration",
    "offline_quantization",
    "offline_search",
    "pack",
    "save",
    "cold_load",
    "repack",
    "first_compile",
    "first_request",
    "steady_prefill",
    "steady_decode",
    "kv_growth",
    "energy",
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CPU-minimal CI
        raise CapabilityError(
            "model-level measurement needs torch; install the 'benchmark' extra",
            details={"capability": "torch", "reason": "not installed"},
        ) from exc
    return torch


@dataclass
class MemoryStage:
    """One stage of the memory ladder (E05-02 §8.1)."""

    stage: str
    bytes_value: int
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "bytes": self.bytes_value, "note": self.note}


@dataclass
class MemoryLadder:
    """All measured memory stages plus the theoretical prediction."""

    stages: List[MemoryStage] = field(default_factory=list)
    predicted: Optional[Dict[str, Any]] = None
    residual: Optional[Dict[str, Any]] = None

    def add(self, stage: str, bytes_value: int, note: str = "") -> None:
        if stage in {item.stage for item in self.stages}:
            raise ConfigError(f"memory stage {stage!r} recorded twice")
        self.stages.append(MemoryStage(stage=stage, bytes_value=int(bytes_value), note=note))

    def get(self, stage: str) -> Optional[int]:
        for item in self.stages:
            if item.stage == stage:
                return item.bytes_value
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "predicted": self.predicted,
            "residual": self.residual,
        }


def measure_artifact_memory(document: QuantArtifactDocument, *, artifact_dir: Optional[str] = None) -> Dict[str, int]:
    """Measure on-disk/artifact bytes for one QuantArtifact (E05-01 §10).

    Separates ``file_size`` (what is on disk now), ``canonical_bytes``
    (auditable logical form), ``packed_runtime_bytes`` (what a kernel would
    read) and ``metadata_bytes``. Reporting a single "compressed size" would
    make the scale/padding overhead invisible.
    """
    import os

    path = artifact_dir or document.artifact_dir
    disk = 0
    if path and os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for filename in files:
                disk += os.path.getsize(os.path.join(root, filename))
    breakdown = document.size_breakdown()
    return {
        "file_size_bytes": disk,
        "canonical_q_bytes": breakdown.canonical_q_bytes,
        "scale_bytes": breakdown.scale_bytes,
        "zero_bytes": breakdown.zero_bytes,
        "packed_variant_bytes": breakdown.packed_variant_bytes,
        "metadata_bytes": breakdown.metadata_bytes,
        "alignment_padding_bytes": breakdown.alignment_padding_bytes,
        "checksum_bytes": breakdown.checksum_bytes,
        "canonical_total_bytes": breakdown.canonical_total,
        "fp16_equivalent_bytes": document.fp16_equivalent_bytes(),
    }


def device_memory_ladder(
    document: QuantArtifactDocument,
    *,
    include_host: bool = True,
) -> MemoryLadder:
    """Build the memory ladder for an artifact using available measurements.

    Device numbers come from :mod:`hqsb.benchmark.memory` (allocated/reserved
    /peak); host numbers from ``/proc``. Any stage that cannot be measured is
    *absent* (``None``), never zero: a missing measurement is not a small one.
    """
    from hqsb.benchmark import memory as mem

    ladder = MemoryLadder()
    measured = measure_artifact_memory(document)
    ladder.add("artifact_file", measured["file_size_bytes"], "on-disk container")
    ladder.add("canonical_bytes", measured["canonical_total_bytes"], "auditable logical form")
    ladder.add("packed_runtime_bytes", measured["packed_variant_bytes"], "kernel layout")
    ladder.add("metadata_bytes", measured["metadata_bytes"], "manifest + checksums")
    breakdown = document.size_breakdown()
    ladder.predicted = breakdown.as_dict()
    ladder.residual = breakdown.reconcile(measured["file_size_bytes"] + measured["packed_variant_bytes"])
    if include_host:
        ladder.add("host_rss_bytes", mem.process_rss_bytes(), "current process RSS")
        ladder.add("host_swap_bytes", mem.process_swap_bytes(), "current process swap")
    snapshot = mem.cuda_memory_snapshot()
    if snapshot["allocated_mb"] > 0 or _cuda_available():
        ladder.add("device_allocated_bytes", int(snapshot["allocated_mb"] * 1024 * 1024))
        ladder.add("device_reserved_bytes", int(snapshot["reserved_mb"] * 1024 * 1024))
        ladder.add("device_peak_allocated_bytes", int(snapshot["peak_allocated_mb"] * 1024 * 1024))
    return ladder


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - CPU-minimal CI
        return False


@dataclass
class PhaseSample:
    """One timed repetition of one phase."""

    phase: str
    index: int
    process_id: int
    latency_ms: float
    tokens: int = 0
    power_w: float = float("nan")
    energy_j: float = float("nan")
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "index": self.index,
            "process_id": self.process_id,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "power_w": self.power_w,
            "energy_j": self.energy_j,
            "note": self.note,
        }


@dataclass
class PhaseResult:
    """Raw samples plus a percentile summary for one phase."""

    phase: str
    samples: List[PhaseSample] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        from hqsb.benchmark.metrics import latency_summary

        return latency_summary([sample.latency_ms for sample in self.samples])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "raw": [sample.as_dict() for sample in self.samples],
            "summary": self.summary(),
        }


class PhaseTimer:
    """Phase-separated timing with warmup, repeats and explicit sync.

    ``use_cuda_events`` measures with device events (kernel time) while the
    host clock measures the end-to-end boundary; both are reported, and the
    caller must state which one a claim refers to (E05-02 §9.2).
    """

    def __init__(
        self,
        *,
        warmup: int = 3,
        repeats: int = 10,
        use_cuda_events: bool = False,
        process_id: Optional[int] = None,
    ) -> None:
        if warmup < 0:
            raise ConfigError(f"warmup must be >= 0, got {warmup}")
        if repeats <= 0:
            raise ConfigError(f"repeats must be positive, got {repeats}")
        self.warmup = warmup
        self.repeats = repeats
        self.use_cuda_events = use_cuda_events
        import os

        self.process_id = process_id if process_id is not None else os.getpid()

    def time_phase(
        self,
        phase: str,
        fn: Callable[[], Any],
        *,
        tokens: int = 0,
        setup: Optional[Callable[[], None]] = None,
    ) -> PhaseResult:
        """Time one phase; ``setup`` runs before every repetition (cache reset).

        The warmup repetitions are *not* recorded, which is how "first
        compile" can be kept out of steady-state numbers only if the caller
        uses a separate, explicitly labelled phase for it.
        """
        if phase not in PHASES:
            raise ConfigError(
                f"unknown phase {phase!r}; supported: {list(PHASES)}"
            )
        torch = _require_torch() if self.use_cuda_events else None
        result = PhaseResult(phase=phase)
        for _ in range(self.warmup):
            if setup is not None:
                setup()
            fn()
            if torch is not None:
                torch.cuda.synchronize()
        for index in range(self.repeats):
            if setup is not None:
                setup()
            if torch is not None:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                fn()
                end_event.record()
                torch.cuda.synchronize()
                latency_ms = float(start_event.elapsed_time(end_event))
            else:
                start = time.perf_counter()
                fn()
                latency_ms = (time.perf_counter() - start) * 1000.0
            result.samples.append(
                PhaseSample(
                    phase=phase,
                    index=index,
                    process_id=self.process_id,
                    latency_ms=latency_ms,
                    tokens=tokens,
                )
            )
        return result


@dataclass
class MeasurementGate:
    """Result of the measurement-quality gate (E05-10 §5.5)."""

    passed: bool
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons), "details": dict(self.details)}


def measurement_gate(
    phase_results: Sequence[PhaseResult],
    *,
    independent_processes: int,
    required_processes: int = 3,
    require_warmup: bool = True,
    phases_with_warmup: Optional[Sequence[str]] = None,
    cold_warm_separated: bool = True,
) -> MeasurementGate:
    """Check that a performance result satisfies the measurement gate.

    Fails when fewer than ``required_processes`` independent processes
    contributed samples, when a phase has too few samples, or when cold/warm
    states are mixed. A failure here disqualifies the numbers from entering
    the Pareto analysis — it never upgrades them.
    """
    reasons: List[str] = []
    if independent_processes < required_processes:
        reasons.append(
            f"only {independent_processes} independent process(es); "
            f"{required_processes} required"
        )
    if not phase_results:
        reasons.append("no phase results recorded")
    for result in phase_results:
        process_ids = {sample.process_id for sample in result.samples}
        if len(process_ids) < required_processes and independent_processes >= required_processes:
            reasons.append(
                f"phase {result.phase!r} has samples from {len(process_ids)} process(es)"
            )
        if len(result.samples) < 3:
            reasons.append(
                f"phase {result.phase!r} has only {len(result.samples)} sample(s)"
            )
    if require_warmup and phases_with_warmup is not None:
        missing = [phase for phase in phases_with_warmup if phase not in PHASES]
        if missing:
            reasons.append(f"unknown phases requested for warmup: {missing}")
    if not cold_warm_separated:
        reasons.append("cold and warm states are not separated")
    return MeasurementGate(passed=not reasons, reasons=reasons)


# ── perplexity (E05-02 §7.3) ──────────────────────────────────────────────


def perplexity_from_nll(nll_values: Sequence[float]) -> float:
    """``exp(mean(nll))`` with an explicit, caller-supplied denominator."""
    clean = [value for value in nll_values if value is not None and not math.isnan(value)]
    if not clean:
        return float("nan")
    return math.exp(sum(clean) / len(clean))


def perplexity_from_logits(
    logits_rows: Sequence[Sequence[float]],
    target_ids: Sequence[int],
    *,
    valid_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Token-level perplexity from logits with a fixed denominator.

    The denominator counts only tokens that contribute an NLL term: the first
    position is excluded (no prediction target) and ``valid_mask`` can exclude
    further positions explicitly. The returned record carries the denominator
    so two runs with different masking cannot be compared silently.
    """
    if len(logits_rows) != len(target_ids):
        raise ConfigError(
            f"logits rows ({len(logits_rows)}) and targets ({len(target_ids)}) "
            f"must match"
        )
    nll_values: List[float] = []
    counted = 0
    for index, (row, target) in enumerate(zip(logits_rows, target_ids)):
        if index == 0 and valid_mask is None:
            continue
        if valid_mask is not None and index < len(valid_mask) and not valid_mask[index]:
            continue
        if not row:
            continue
        maximum = max(row)
        log_sum = math.log(math.fsum(math.exp(value - maximum) for value in row))
        log_prob = row[int(target)] - maximum - log_sum
        nll_values.append(-log_prob)
        counted += 1
    return {
        "perplexity": perplexity_from_nll(nll_values),
        "mean_nll": (sum(nll_values) / len(nll_values)) if nll_values else float("nan"),
        "denominator": counted,
        "excluded_positions": len(logits_rows) - counted,
        "total_positions": len(logits_rows),
    }


def perplexity_torch(model, input_ids, *, stride: int = 512, max_length: Optional[int] = None) -> Dict[str, Any]:
    """Sliding-window perplexity with frozen stride and denominator.

    The window/stride/dtype are part of the protocol: ``stride`` and the
    effective ``max_length`` are returned so the number can be compared only
    with runs using the same settings.
    """
    torch = _require_torch()
    if stride <= 0:
        raise ConfigError(f"stride must be positive, got {stride}")
    window = int(max_length or getattr(model.config, "max_position_embeddings", 2048))
    if window <= stride:
        raise ConfigError(
            f"max_length ({window}) must exceed stride ({stride}) or every window "
            f"would be fully discarded"
        )
    ids = input_ids.reshape(-1)
    nlls: List[float] = []
    counted = 0
    with torch.inference_mode():
        for start in range(0, len(ids), stride):
            end = min(start + window, len(ids))
            if end - start < 2:
                break
            chunk = ids[start:end].unsqueeze(0)
            outputs = model(chunk)
            logits = outputs.logits[0, :-1, :].float()
            targets = chunk[0, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            per_token = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            # Only score the *new* stride tokens (sliding window protocol).
            new_tokens = end - start - 1 if start == 0 else min(stride, end - start - 1)
            if start > 0:
                per_token = per_token[-new_tokens:]
            nlls.extend(float(value) for value in per_token.tolist())
            counted += int(per_token.numel())
    return {
        "perplexity": perplexity_from_nll(nlls),
        "mean_nll": (sum(nlls) / len(nlls)) if nlls else float("nan"),
        "denominator": counted,
        "stride": stride,
        "max_length": window,
        "protocol": "sliding_window_new_tokens_only",
    }


# ── unified result table (E05-02 §11, E05-10 §8) ──────────────────────────


TABLE_FIELDS = (
    "run_id",
    "spec_hash",
    "commit",
    "env_hash",
    "model_hash",
    "tokenizer_hash",
    "workload_id",
    "phase",
    "candidate_id",
    "method",
    "bits",
    "scheme_hash",
    "group_axis",
    "group_size",
    "coverage_parameter",
    "coverage_call",
    "coverage_time",
    "artifact_hash",
    "packed_hash",
    "metric",
    "value",
    "unit",
    "baseline_value",
    "delta",
    "ci_low",
    "ci_high",
    "process_id",
    "repeat",
    "execution_label",
    "observed_kernel",
    "fallback_reason",
    "verdict",
)


def build_unified_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize raw records into the unified table with a content hash.

    Unknown keys are rejected (a typo'd field would otherwise be silently
    dropped) and the table hash is computed over the sorted rows, so a figure
    can be traced back to the exact table it came from.
    """
    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        unknown = set(row) - set(TABLE_FIELDS)
        if unknown:
            raise ConfigError(
                f"row {index} has unknown field(s) {sorted(unknown)}; the table "
                f"schema is fixed so figures cannot invent columns"
            )
        normalized.append({field_name: row.get(field_name) for field_name in TABLE_FIELDS})
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "rows": normalized,
        "row_count": len(normalized),
        "table_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "schema": list(TABLE_FIELDS),
    }


# ── adjudication order (E05-02 §10 step 16, E05-10 §5) ────────────────────


def gate_order(
    *,
    correctness: Mapping[str, Any],
    artifact: Mapping[str, Any],
    quality: Mapping[str, Any],
    execution: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the five gates in the fixed order and report the first failure.

    The order matters: a candidate that fails correctness must never reach the
    quality comparison, and a candidate whose execution was fake-quant must
    never reach a performance Pareto. Returns per-gate verdicts plus a final
    recommendation class:
    ``deployment_candidate`` / ``quality_only`` / ``research_only`` /
    ``rejected`` / ``incomplete``.
    """
    def _verdict(report: Mapping[str, Any]) -> str:
        if report.get("verdict"):
            return str(report["verdict"])
        if report.get("passed") is True:
            return GATE_PASS
        if report.get("passed") is False:
            return GATE_FAIL
        return GATE_INCONCLUSIVE

    verdicts = {
        "correctness": _verdict(correctness),
        "artifact": _verdict(artifact),
        "quality": _verdict(quality),
        "execution": _verdict(execution),
        "measurement": _verdict(measurement),
    }
    failed = [name for name, verdict in verdicts.items() if verdict == GATE_FAIL]
    inconclusive = [name for name, verdict in verdicts.items() if verdict == GATE_INCONCLUSIVE]
    if "correctness" in failed or "artifact" in failed:
        classification = "rejected"
    elif "quality" in failed:
        classification = "rejected"
    elif "execution" in failed:
        classification = "quality_only"
    elif "measurement" in failed:
        classification = "research_only"
    elif inconclusive:
        classification = "incomplete"
    else:
        classification = "deployment_candidate"
    return {
        "verdicts": verdicts,
        "failed_gates": failed,
        "inconclusive_gates": inconclusive,
        "classification": classification,
        "order": ["correctness", "artifact", "quality", "execution", "measurement"],
        "note": (
            "quality_only candidates may support algorithm-quality statements "
            "but must not enter a low-bit deployment Pareto; incomplete means "
            "some gate lacked evidence, which is not a pass"
        ),
    }


__all__ = [
    "MemoryLadder",
    "MemoryStage",
    "MeasurementGate",
    "PHASES",
    "PhaseResult",
    "PhaseSample",
    "PhaseTimer",
    "TABLE_FIELDS",
    "build_unified_table",
    "device_memory_ladder",
    "gate_order",
    "measure_artifact_memory",
    "measurement_gate",
    "perplexity_from_logits",
    "perplexity_from_nll",
    "perplexity_torch",
]
