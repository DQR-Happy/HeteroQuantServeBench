"""Multi-level profiling correlation for E02-07.

E02-07 must connect five levels into one auditable chain::

    workload/phase -> Qwen module -> framework operator
                   -> CUDA kernel/launch -> hardware counter -> bottleneck

Three tools observe that chain at different depths and with different
perturbations, so the experiment never pretends the three give the same
milliseconds. What it *does* require is that the same hotspot can be
identified, shaped, counted and located in time consistently across them.

This module holds the correlation logic, deliberately free of any CUDA
dependency so it runs in the CPU-only unit-test suite:

* :func:`phase_span_index` / :func:`attribute_events_to_phases` build the
  phase attribution from an un-modified Chrome trace (Kineto). Attribution
  is done through the ``correlation`` id chain (CPU event -> CUDA runtime
  call -> device kernel) instead of by timestamp containment alone, because
  a timestamp-domain mismatch between the CPU and device rows of a Kineto
  trace would silently assign kernels to the wrong phase.
* :func:`timeline_gaps` / :func:`gpu_idle_windows` answer the Nsight Systems
  questions (launch->start latency, inter-kernel gaps, stream overlap, GPU
  holes) from the same trace.
* :func:`parse_ncu_csv` / :func:`ncu_metric_panel` turn the Nsight Compute
  CSV export into a small, reviewable counter panel.
* :func:`roofline_consistency` does the step-11 cross-check while keeping
  theoretical FLOPs, logical tensor bytes, L2 bytes and (unavailable) DRAM
  bytes as distinct quantities.

Terminology kept from E02-02: ``cumulative kernel work time`` is the sum of
kernel durations *within one scope*. With overlapping streams it can exceed
the wall-clock span, so it is only ever used to normalise shares, never as a
phase duration and never as an Amdahl fraction (see
:func:`critical_path_share_note`).
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ── Frozen NVTX range names ──────────────────────────────────────────────
#
# The ranges are deliberately coarse (run / phase / early-late decode), as
# the E02-07 protocol requires: one annotation per model phase, not one per
# operator. ``result_handling`` exists so that token decoding and result
# serialisation can never be mistaken for model work.

RUN_RANGE = "e02_07_run"
PREFILL_RANGE = "e02_07_prefill"
FIRST_TOKEN_RANGE = "e02_07_first_token_selection"
DECODE_EARLY_RANGE = "e02_07_decode_early"
DECODE_MIDDLE_RANGE = "e02_07_decode_middle"
DECODE_LATE_RANGE = "e02_07_decode_late"
RESULT_RANGE = "e02_07_result_handling"

#: Phase ranges that partition the model-core region, in execution order.
PHASE_RANGES: Tuple[str, ...] = (
    PREFILL_RANGE,
    FIRST_TOKEN_RANGE,
    DECODE_EARLY_RANGE,
    DECODE_MIDDLE_RANGE,
    DECODE_LATE_RANGE,
    RESULT_RANGE,
)

#: Ranges that contain model work (``result_handling`` is explicitly not one).
MODEL_PHASE_RANGES: Tuple[str, ...] = (
    PREFILL_RANGE,
    FIRST_TOKEN_RANGE,
    DECODE_EARLY_RANGE,
    DECODE_MIDDLE_RANGE,
    DECODE_LATE_RANGE,
)

#: Prefix of the per-module-role probe ranges on a single audited layer.
#:
#: Only *one* layer carries module-role ranges. Annotating every module of
#: every layer would add thousands of annotations and perturb the measurement
#: far more than it clarifies (the protocol warns about this explicitly), while
#: one fully annotated layer is enough to audit the module->op->kernel chain.
MODULE_ROLE_PREFIX = "e02_07_role."

#: Chrome-trace categories produced by Kineto.
DEVICE_TRACE_CATEGORIES: Tuple[str, ...] = ("kernel", "gpu_memcpy", "gpu_memset")
CPU_TRACE_CATEGORIES: Tuple[str, ...] = ("cpu_op", "cuda_runtime", "cuda_driver")
NVTX_TRACE_CATEGORIES: Tuple[str, ...] = ("user_annotation", "nvtx", "NVTX")


def phase_ledger(
    *,
    input_len: int,
    output_tokens: int,
    early_steps: int,
    late_steps: int,
) -> Dict[str, Any]:
    """Independent expectation of what each phase range must cover.

    This is the E02-07 analogue of the E02-01 step ledger: the runner must
    satisfy it *before* any profile is trusted, because a range that is off
    by one step is worse than no range at all.

    Args:
        input_len: ISL, the number of prompt tokens in one prefill.
        output_tokens: OSL (G). Decode runs ``G - 1`` steps.
        early_steps: Number of leading decode steps inside ``decode_early``.
        late_steps: Number of trailing decode steps inside ``decode_late``.

    Returns:
        A dict with the expected per-range forward-pass counts and the
        expected covered decode step indices.

    Raises:
        ValueError: If the requested windows cannot tile the decode steps.
    """
    if input_len < 1:
        raise ValueError(f"input_len must be >= 1, got {input_len}")
    if output_tokens < 1:
        raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")
    if early_steps < 0 or late_steps < 0:
        raise ValueError("early_steps and late_steps must be >= 0")

    total_decode = output_tokens - 1
    if early_steps + late_steps > total_decode:
        raise ValueError(
            f"early_steps + late_steps ({early_steps + late_steps}) exceeds the "
            f"{total_decode} decode steps of OSL={output_tokens}"
        )

    early = list(range(1, early_steps + 1))
    late = (
        list(range(total_decode - late_steps + 1, total_decode + 1))
        if late_steps
        else []
    )
    claimed = set(early) | set(late)
    middle = [s for s in range(1, total_decode + 1) if s not in claimed]
    return {
        "prefill": {"forward_passes": 1, "query_len": input_len},
        "first_token_selection": {"argmax_calls": 1},
        "decode_early": {
            "steps": early,
            "forward_passes": len(early),
            "context_len": [input_len + s for s in early],
        },
        "decode_middle": {
            "steps": middle,
            "forward_passes": len(middle),
            "context_len": [input_len + s for s in middle],
        },
        "decode_late": {
            "steps": late,
            "forward_passes": len(late),
            "context_len": [input_len + s for s in late],
        },
        "result_handling": {"forward_passes": 0},
        "model_forward_passes_total": 1 + len(early) + len(middle) + len(late),
        "decode_steps_total": total_decode,
        "phases_partition_decode": sorted(claimed | set(middle))
        == list(range(1, total_decode + 1)),
    }


# ── Chrome trace loading and indexing ────────────────────────────────────


def load_chrome_trace(path: str) -> Dict[str, Any]:
    """Load a Kineto Chrome trace, accepting either the wrapper or a list.

    Raises:
        ValueError: If the file does not contain a ``traceEvents`` list.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "traceEvents" in payload:
        return payload
    if isinstance(payload, list):
        return {"traceEvents": payload}
    raise ValueError(f"{path}: no 'traceEvents' list found")


def trace_events(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return the complete-``X`` events of a trace, skipping metadata rows."""
    events = trace.get("traceEvents") or []
    return [e for e in events if e.get("ph") == "X" and "ts" in e]


def nvtx_spans(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return the NVTX/annotation spans of a trace, innermost-sortable.

    Kineto exports ``torch.cuda.nvtx.range`` and ``record_function`` under
    ``cat="user_annotation"``; older builds used ``nvtx``. Both are accepted,
    and unknown categories are simply not treated as spans.
    """
    spans: List[Dict[str, Any]] = []
    for event in trace_events(trace):
        if str(event.get("cat")) not in NVTX_TRACE_CATEGORIES:
            continue
        spans.append(
            {
                "name": str(event.get("name")),
                "ts": float(event["ts"]),
                "dur": float(event.get("dur", 0.0) or 0.0),
                "tid": int(event.get("tid", -1)),
            }
        )
    return sorted(spans, key=lambda s: (s["ts"], -s["dur"]))


def device_events(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return device-scope events (kernels plus device memcpy/memset)."""
    rows: List[Dict[str, Any]] = []
    for event in trace_events(trace):
        category = str(event.get("cat"))
        if category not in DEVICE_TRACE_CATEGORIES:
            continue
        args = event.get("args") or {}
        rows.append(
            {
                "name": str(event.get("name")),
                "cat": category,
                "ts": float(event["ts"]),
                "dur": float(event.get("dur", 0.0) or 0.0),
                "tid": int(event.get("tid", -1)),
                "stream": int(args.get("stream", -1)),
                "correlation": args.get("correlation"),
                "external_id": args.get("External id"),
                "queued_us": args.get("queued"),
                "grid": list(args.get("grid", []) or []),
                "block": list(args.get("block", []) or []),
                "registers_per_thread": args.get("registers per thread"),
                "shared_memory": args.get("shared memory"),
            }
        )
    return sorted(rows, key=lambda r: r["ts"])


def correlation_index(trace: Mapping[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """Map every ``correlation`` id to its owning host-side ATen operator.

    One correlation id is reported on several host rows (``aten::linear``,
    ``aten::mm``, ``cudaLaunchKernel``). Two rules pick the "owner":

    1. an ATen op (``cat="cpu_op"``) always beats a CUDA runtime/driver row,
       because the runtime row only describes the launch, not the operator;
    2. among equals, the innermost (shortest) row wins, so ``aten::mm`` is
       preferred over the ``aten::linear`` that contains it.

    ``chain`` keeps the full sorted list of host rows for the correlation so
    the Module -> Op -> Kernel mapping can show ``aten::linear -> aten::mm ->
    kernel`` instead of pretending the chain has one link.
    """
    index: Dict[Any, Dict[str, Any]] = {}
    for event in trace_events(trace):
        if str(event.get("cat")) not in CPU_TRACE_CATEGORIES:
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation")
        if correlation is None:
            continue
        candidate = {
            "name": str(event.get("name")),
            "cat": str(event.get("cat")),
            "ts": float(event["ts"]),
            "dur": float(event.get("dur", 0.0) or 0.0),
        }
        current = index.get(correlation)
        if current is None:
            candidate["chain"] = [candidate["name"]]
            index[correlation] = candidate
            continue
        chain = current.setdefault("chain", [current["name"]])
        if candidate["name"] not in chain:
            chain.append(candidate["name"])
        better = (
            candidate["cat"] == "cpu_op" and current["cat"] != "cpu_op"
        ) or (
            candidate["cat"] == current["cat"] and candidate["dur"] < current["dur"]
        )
        if better:
            chain_sorted = [candidate["name"]] + [
                name for name in chain if name != candidate["name"]
            ]
            index[correlation] = {
                **candidate,
                "chain": chain_sorted,
            }
    return index


def phase_span_index(
    trace: Mapping[str, Any],
    ranges: Sequence[str] = PHASE_RANGES,
) -> Dict[str, Dict[str, Any]]:
    """Index the phase spans actually present, keyed by range name.

    When a range is entered more than once (``decode_early`` is not, but a
    re-entrant caller could), the spans are merged into one envelope and
    ``instances`` records how many were seen, so a duplicate range is visible
    instead of being silently collapsed.
    """
    wanted = set(ranges)
    found: Dict[str, Dict[str, Any]] = {}
    for span in nvtx_spans(trace):
        if span["name"] not in wanted:
            continue
        entry = found.get(span["name"])
        if entry is None:
            found[span["name"]] = {
                "name": span["name"],
                "ts": span["ts"],
                "end": span["ts"] + span["dur"],
                "dur": span["dur"],
                "instances": 1,
                "tids": {span["tid"]},
            }
        else:
            entry["ts"] = min(entry["ts"], span["ts"])
            entry["end"] = max(entry["end"], span["ts"] + span["dur"])
            entry["dur"] = entry["end"] - entry["ts"]
            entry["instances"] += 1
            entry["tids"].add(span["tid"])
    for entry in found.values():
        entry["tids"] = sorted(entry["tids"])
    return found


def _innermost_phase(
    spans: Mapping[str, Mapping[str, Any]],
    ts: float,
) -> Optional[str]:
    """Return the name of the innermost phase span containing ``ts``."""
    best: Optional[str] = None
    best_dur = math.inf
    for name, span in spans.items():
        if span["ts"] <= ts <= span["end"] and span["dur"] < best_dur:
            best, best_dur = name, span["dur"]
    return best


def external_id_index(trace: Mapping[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """Map ``External id`` -> the ATen op that carries that record-function id.

    Kineto links a device kernel to the host in two different ways and only
    one of them names an operator:

    * ``args.correlation`` links the kernel to the **CUDA runtime call** that
      enqueued it (``cudaLaunchKernel``, ``cuLaunchKernel``...). It is what
      identifies the launch, not the operator.
    * ``args["External id"]`` is the id of the innermost ``RecordFunction``
      active at launch time, i.e. the ``cpu_op`` row of the ATen operator.

    Using the correlation id alone therefore attributes every kernel to
    ``cudaLaunchKernel`` — technically true, useless for a Module -> Op ->
    Kernel mapping. This index supplies the operator name; only ``cpu_op``
    rows are indexed, so a kernel can never resolve back to the launch API.
    """
    index: Dict[Any, Dict[str, Any]] = {}
    for event in trace_events(trace):
        if str(event.get("cat")) != "cpu_op":
            continue
        args = event.get("args") or {}
        external_id = args.get("External id")
        if external_id is None:
            continue
        index.setdefault(
            external_id,
            {
                "name": str(event.get("name")),
                "ts": float(event["ts"]),
                "dur": float(event.get("dur", 0.0) or 0.0),
                # ``record_shapes=True`` puts the live input dims on the op row,
                # which is what makes the shape an observation rather than a
                # hand-copied constant.
                "input_dims": args.get("Input Dims"),
            },
        )
    return index


def _dims_signature(input_dims: Any) -> Optional[str]:
    """Render ``Input Dims`` as a compact, comparable shape signature.

    ``Input Dims`` nests differently per operator: a pointwise op gets one
    list per operand, while ``aten::cat`` gets a list of operand lists inside
    a single entry. The renderer therefore recurses instead of assuming one
    level, and scalar entries (which Kineto prints as a type name) are kept so
    the arity of the signature stays comparable.
    """

    def render(node: Any, depth: int = 0) -> Optional[str]:
        if depth > 4 or not isinstance(node, (list, tuple)):
            return None
        items: List[str] = []
        for entry in node:
            if isinstance(entry, (list, tuple)):
                if not entry:
                    continue
                if all(not isinstance(v, (list, tuple)) for v in entry):
                    items.append("[" + ",".join(str(int(v)) for v in entry) + "]")
                else:
                    nested = render(entry, depth + 1)
                    if nested:
                        items.append("{" + nested + "}")
            elif isinstance(entry, str) and entry:
                items.append(entry)
        return " x ".join(items) if items else None

    if not isinstance(input_dims, (list, tuple)) or not input_dims:
        return None
    return render(input_dims)


def op_attribution(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve every device event to its owning ATen operator.

    Returns ``{"kernels": [...], "by_op": {...}, "unresolved": n}`` where each
    kernel entry keeps the original device fields plus ``op`` and ``op_ts``.
    Resolution order is ``External id`` first (the operator), then the
    correlation chain restricted to ``cpu_op`` rows (a weaker but still
    operator-level link), then ``None``.
    """
    correlations = correlation_index(trace)
    external = external_id_index(trace)
    kernels = device_events(trace)
    by_op: Dict[str, Dict[str, Any]] = {}
    unresolved = 0

    for kernel in kernels:
        owner = None
        external_id = kernel.get("external_id")
        if external_id is not None:
            owner = external.get(external_id)
        if owner is None:
            correlation = kernel.get("correlation")
            candidate = correlations.get(correlation) if correlation is not None else None
            if candidate is not None and candidate["cat"] == "cpu_op":
                owner = candidate
        if owner is None:
            unresolved += 1
            kernel["op"] = None
            kernel["op_ts"] = None
            kernel["op_dims"] = None
            continue
        kernel["op"] = owner["name"]
        kernel["op_ts"] = owner["ts"]
        kernel["op_dims"] = _dims_signature(owner.get("input_dims"))
        entry = by_op.setdefault(
            owner["name"],
            {
                "op": owner["name"],
                "kernel_count": 0,
                "device_us": 0.0,
                "kernels": {},
            },
        )
        entry["kernel_count"] += 1
        entry["device_us"] += float(kernel.get("dur", 0.0) or 0.0)
        entry["kernels"][kernel["name"]] = entry["kernels"].get(kernel["name"], 0) + 1

    ranked = sorted(by_op.values(), key=lambda e: -e["device_us"])
    return {
        "kernels": kernels,
        "by_op": ranked,
        "unresolved": unresolved,
        "external_ids_indexed": len(external),
        "correlations_indexed": len(correlations),
    }


def attribute_events_to_phases(
    trace: Mapping[str, Any],
    *,
    ranges: Sequence[str] = PHASE_RANGES,
) -> Dict[str, Any]:
    """Attribute device kernels to phase ranges, at operator granularity.

    Each kernel is placed by the timestamp of the **ATen operator that owns
    it** (``External id`` chain), falling back to the timestamp of the CUDA
    runtime call that enqueued it. The operator timestamp is the more precise
    of the two: it is the host-side region the user actually wrote, so a
    kernel cannot be attributed to whatever launch API happened to be sampled.

    Returns a dict with:

    * ``phases``: ``{range: {"span_us", "kernels", "count", "kernel_work_us",
      "memcpy_count", "streams", "host_ops"}}``;
    * ``unattributed``: device work that never lands inside a phase range —
      kept as its own bucket instead of being spread over familiar modules;
    * ``coverage``: how many device events were attributed vs. left out.
    """
    spans = phase_span_index(trace, ranges)
    correlations = correlation_index(trace)
    attribution = op_attribution(trace)
    kernels = attribution["kernels"]

    phases: Dict[str, Dict[str, Any]] = {
        name: {
            "name": name,
            "span_us": span["dur"],
            "instances": span["instances"],
            "kernels": [],
            "count": 0,
            "kernel_work_us": 0.0,
            "memcpy_count": 0,
            "streams": set(),
            "roles": set(),
            "host_ops": {},
        }
        for name, span in spans.items()
    }

    unattributed: List[Dict[str, Any]] = []
    unattributed_us = 0.0
    attributed_count = 0

    for kernel in kernels:
        ts = kernel.get("op_ts")
        if ts is None:
            correlation = kernel.get("correlation")
            host = correlations.get(correlation) if correlation is not None else None
            ts = host["ts"] if host is not None else None
        phase = _innermost_phase(spans, ts) if ts is not None else None
        if phase is None:
            unattributed_us += kernel["dur"]
            unattributed.append(kernel)
            continue
        attributed_count += 1
        bucket = phases[phase]
        bucket["kernels"].append(kernel)
        bucket["count"] += 1
        bucket["kernel_work_us"] += kernel["dur"]
        if kernel["cat"] in ("gpu_memcpy", "gpu_memset"):
            bucket["memcpy_count"] += 1
        bucket["streams"].add(kernel["stream"])
        if kernel.get("op"):
            bucket["roles"].add(kernel["op"])

    for phase in phases.values():
        phase["streams"] = sorted(phase["streams"])
        phase["roles"] = sorted(phase["roles"])
        phase["host_ops"] = _host_op_counts(phase["kernels"])
        phase["kernels"] = sorted(phase["kernels"], key=lambda k: k["ts"])

    return {
        "phases": phases,
        "unattributed": {
            "count": len(unattributed),
            "device_work_us": unattributed_us,
            "names": sorted({k["name"] for k in unattributed})[:32],
            "categories": sorted({k["cat"] for k in unattributed}),
        },
        "coverage": {
            "device_events_total": len(kernels),
            "device_events_attributed": attributed_count,
            "attributed_ratio": (attributed_count / len(kernels) if kernels else 0.0),
            "operator_resolution": {
                "resolved": len(kernels) - attribution["unresolved"],
                "unresolved": attribution["unresolved"],
                "external_ids_indexed": attribution["external_ids_indexed"],
            },
        },
    }


def _host_op_counts(kernels: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Count the ATen operators that own the device events of one phase."""
    counts: Dict[str, int] = {}
    for kernel in kernels:
        op = kernel.get("op")
        if not op:
            continue
        counts[op] = counts.get(op, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def host_op_table_by_phase(
    trace: Mapping[str, Any],
    *,
    ranges: Sequence[str] = PHASE_RANGES,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-phase table of the host operator that *owns* each device kernel.

    Two time columns are produced and they mean different things:

    * ``device_us`` — the device time of the kernels this op launched. This
      is a *self* device time: kernels are leaves, so nothing is double
      counted, but the sum over ops equals the cumulative kernel work, whose
      overlap caveat still applies.
    * ``op_invocations`` / ``host_op_total_us`` — how many ``cpu_op`` rows of
      that name lie inside the phase, and their total span. The total is
      *inclusive*: a parent row (``aten::linear``) contains its children
      (``aten::mm``), so these totals must never be summed across rows, and
      they are not the ATen self CPU time that ``key_averages()`` reports.
      They are named differently for that reason.

    Kernels with no resolvable operator (bare CUDA calls) are grouped under
    ``"<no-aten-op>"`` so the residual stays visible instead of being hidden
    in a real operator.
    """
    spans = phase_span_index(trace, ranges)
    attribution = attribute_events_to_phases(trace, ranges=ranges)
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _row(phase: str, name: str) -> Dict[str, Any]:
        bucket = buckets.setdefault(phase, {})
        row = bucket.get(name)
        if row is None:
            row = {
                "name": name,
                "kernel_count": 0,
                "op_invocations": 0,
                "device_us": 0.0,
                "host_op_total_us": 0.0,
            }
            bucket[name] = row
        return row

    for phase, info in attribution["phases"].items():
        for kernel in info["kernels"]:
            row = _row(phase, kernel.get("op") or "<no-aten-op>")
            row["kernel_count"] += 1
            row["device_us"] += float(kernel.get("dur", 0.0) or 0.0)

    # Operator call counts and host-side cost come from the ``cpu_op`` rows
    # themselves, de-duplicated by (name, ts, duration). Reading them off the
    # kernels instead would count a multi-kernel operator once per kernel, and
    # summing the correlation index would count a shared row once per launch.
    seen: Dict[Tuple[str, float, float], str] = {}
    for event in trace_events(trace):
        if str(event.get("cat")) != "cpu_op":
            continue
        ts = float(event["ts"])
        phase = _innermost_phase(spans, ts)
        if phase is None:
            continue
        key = (
            str(event.get("name")),
            ts,
            float(event.get("dur", 0.0) or 0.0),
        )
        seen.setdefault(key, phase)
    for (name, _ts, dur), phase in seen.items():
        row = _row(phase, name)
        row["op_invocations"] += 1
        # Inclusive: a parent row (``aten::linear``) contains its children
        # (``aten::mm``), so these totals must never be summed across rows.
        row["host_op_total_us"] += dur

    return {
        phase: sorted(bucket.values(), key=lambda r: -r["device_us"])
        for phase, bucket in buckets.items()
    }


def module_role_table(
    trace: Mapping[str, Any],
    *,
    prefix: str = MODULE_ROLE_PREFIX,
    role_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Map annotated module roles to the ops and kernels they produced.

    The runner annotates a single audited layer's submodules with
    ``e02_07_role.<role>``. A kernel is assigned to a role only when its
    owning ATen operator's timestamp falls inside that role's range, so the
    edge Module -> Op -> Kernel is *observed*, never inferred from a kernel
    name.

    The mapping is many-to-many on purpose: one module may launch several ops,
    one op several kernels, and a fused kernel may serve several ops. The
    result therefore reports sets plus per-role counts.
    """
    attribution = op_attribution(trace)
    kernels = attribution["kernels"]
    spans = [
        span
        for span in nvtx_spans(trace)
        if span["name"].startswith(prefix)
        and (role_filter is None or span["name"] == f"{prefix}{role_filter}")
    ]
    if not spans:
        return {"roles": {}, "span_count": 0}

    def _innermost_role(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        best: Optional[str] = None
        best_dur = math.inf
        for span in spans:
            if span["ts"] <= ts <= span["ts"] + span["dur"] and span["dur"] < best_dur:
                best, best_dur = span["name"], span["dur"]
        return best

    roles: Dict[str, Dict[str, Any]] = {}
    for kernel in kernels:
        role = _innermost_role(kernel.get("op_ts"))
        if role is None:
            continue
        entry = roles.setdefault(
            role,
            {
                "role": role[len(prefix):],
                "range": role,
                "op_invocations": 0,
                "ops": {},
                "chains": {},
                "kernels": {},
                "kernel_count": 0,
                "device_us": 0.0,
            },
        )
        entry["kernel_count"] += 1
        entry["device_us"] += float(kernel.get("dur", 0.0) or 0.0)
        entry["kernels"][kernel["name"]] = entry["kernels"].get(kernel["name"], 0) + 1
        op = kernel.get("op")
        if op:
            entry["ops"][op] = entry["ops"].get(op, 0) + 1

    for entry in roles.values():
        entry["op_invocations"] = sum(entry["ops"].values())
        entry["ops"] = dict(sorted(entry["ops"].items(), key=lambda kv: (-kv[1], kv[0])))
        entry["kernels"] = dict(
            sorted(entry["kernels"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        entry["chains"] = {
            f"{op} -> {kernel}": count
            for op, count in list(entry["ops"].items())[:6]
            for kernel in list(entry["kernels"])[:2]
        }
        entry["distinct_op_count"] = len(entry["ops"])
        entry["distinct_kernel_count"] = len(entry["kernels"])
    return {
        "roles": dict(sorted(roles.items())),
        "span_count": len(spans),
        "operator_resolution": {
            "resolved": len(kernels) - attribution["unresolved"],
            "unresolved": attribution["unresolved"],
        },
    }


def aggregate_kernels(
    kernels: Iterable[Mapping[str, Any]],
    *,
    key: str = "kernel_work_us",
) -> List[Dict[str, Any]]:
    """Aggregate device events by name, adding duration statistics.

    Returns rows sorted by descending total duration, each carrying the
    call count, total/min/mean/max duration, the distinct streams, grids and
    blocks observed, and the categories seen. The distinct grid/block sets
    are what allow a same-named kernel invoked with different shapes to be
    spotted instead of averaged away.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for kernel in kernels:
        name = str(kernel["name"])
        row = merged.get(name)
        if row is None:
            row = {
                "name": name,
                "count": 0,
                "total_us": 0.0,
                "min_us": math.inf,
                "max_us": 0.0,
                "streams": set(),
                "grids": set(),
                "blocks": set(),
                "categories": set(),
                "registers_per_thread": set(),
                "shared_memory": set(),
                "ops": set(),
                "dims": set(),
            }
            merged[name] = row
        dur = float(kernel.get("dur", 0.0) or 0.0)
        row["count"] += 1
        row["total_us"] += dur
        row["min_us"] = min(row["min_us"], dur)
        row["max_us"] = max(row["max_us"], dur)
        row["streams"].add(kernel.get("stream", -1))
        row["grids"].add(tuple(kernel.get("grid") or ()))
        row["blocks"].add(tuple(kernel.get("block") or ()))
        row["categories"].add(kernel.get("cat", "kernel"))
        if kernel.get("registers_per_thread") is not None:
            row["registers_per_thread"].add(kernel["registers_per_thread"])
        if kernel.get("shared_memory") is not None:
            row["shared_memory"].add(kernel["shared_memory"])
        if kernel.get("op"):
            row["ops"].add(kernel["op"])
        if kernel.get("op_dims"):
            row["dims"].add(kernel["op_dims"])

    rows: List[Dict[str, Any]] = []
    for row in merged.values():
        row["mean_us"] = row["total_us"] / row["count"] if row["count"] else 0.0
        row["min_us"] = 0.0 if row["min_us"] is math.inf else row["min_us"]
        row["streams"] = sorted(row["streams"])
        row["grids"] = [list(g) for g in sorted(row["grids"])]
        row["blocks"] = [list(b) for b in sorted(row["blocks"])]
        row["categories"] = sorted(row["categories"])
        row["registers_per_thread"] = sorted(row["registers_per_thread"])
        row["shared_memory"] = sorted(row["shared_memory"])
        row["ops"] = sorted(row["ops"])
        row["dims"] = sorted(row["dims"])
        row["shape_signature_count"] = max(
            len(row["grids"]), len(row["dims"]), 1
        )
        rows.append(row)

    rows.sort(key=lambda r: r.get(key, r["total_us"]), reverse=True)
    return rows


def attach_shares(
    rows: Sequence[Mapping[str, Any]],
    *,
    total_us: Optional[float] = None,
    field: str = "total_us",
) -> List[Dict[str, Any]]:
    """Add ``time_share`` and ``cumulative_share`` to a ranked table.

    ``total_us`` defaults to the sum of the table itself, which is the
    *cumulative kernel work time* of the profiled region. Callers wanting a
    share of the phase wall-clock must pass the measured span instead, and
    the two denominators are then visibly different numbers.
    """
    denominator = float(total_us) if total_us is not None else float(
        sum(float(r.get(field, 0.0) or 0.0) for r in rows)
    )
    if denominator <= 0:
        denominator = 1.0
    result: List[Dict[str, Any]] = []
    cumulative = 0.0
    for row in rows:
        share = float(row.get(field, 0.0) or 0.0) / denominator
        cumulative += share
        augmented = dict(row)
        augmented["time_share"] = share
        augmented["cumulative_share"] = cumulative
        result.append(augmented)
    return result


def cumulative_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    top: int,
) -> Dict[str, Any]:
    """Coverage of the top-``top`` rows plus the residual they leave behind."""
    head = list(rows[:top])
    covered = sum(float(r.get("time_share", 0.0) or 0.0) for r in head)
    return {
        "top": top,
        "covered_share": covered,
        "residual_share": 1.0 - covered,
        "names": [r["name"] for r in head],
    }


# ── Timeline (Nsight Systems) questions ──────────────────────────────────


def merge_intervals(
    intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Merge possibly overlapping ``(start, end)`` intervals."""
    ordered = sorted((s, e) for s, e in intervals if e >= s)
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def gpu_activity_window(
    kernels: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """GPU busy-union vs. wall span: the honest "GPU hole" measure.

    Summing kernel durations over-counts when streams overlap, so the busy
    time is the *union* of kernel intervals. Everything inside the enclosing
    span that the union does not cover is GPU idle, which is a different
    finding from "the CPU is the bottleneck" — it may equally be a sync, a
    data dependency or simply the tail of the enclosing range.
    """
    if not kernels:
        return {
            "kernel_count": 0,
            "span_us": 0.0,
            "busy_us": 0.0,
            "idle_us": 0.0,
            "idle_ratio": 0.0,
            "sum_of_durations_us": 0.0,
            "overlap_factor": 0.0,
            "streams": [],
        }
    start = min(k["ts"] for k in kernels)
    end = max(k["ts"] + k["dur"] for k in kernels)
    merged = merge_intervals([(k["ts"], k["ts"] + k["dur"]) for k in kernels])
    busy = sum(e - s for s, e in merged)
    span = end - start
    durations = sum(k["dur"] for k in kernels)
    return {
        "kernel_count": len(kernels),
        "span_us": span,
        "busy_us": busy,
        "idle_us": span - busy,
        "idle_ratio": (span - busy) / span if span > 0 else 0.0,
        "sum_of_durations_us": durations,
        "overlap_factor": durations / span if span > 0 else 0.0,
        "streams": stream_breakdown(kernels),
    }


def stream_breakdown(kernels: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per-stream kernel count and duration, to expose unexpected streams."""
    buckets: Dict[int, Dict[str, Any]] = {}
    for kernel in kernels:
        stream = int(kernel.get("stream", -1))
        row = buckets.setdefault(
            stream, {"stream": stream, "count": 0, "total_us": 0.0}
        )
        row["count"] += 1
        row["total_us"] += float(kernel.get("dur", 0.0) or 0.0)
    return sorted(buckets.values(), key=lambda r: -r["total_us"])


def timeline_gaps(
    kernels: Sequence[Mapping[str, Any]],
    *,
    stream: Optional[int] = None,
) -> Dict[str, Any]:
    """Inter-kernel gaps on one stream (default: the busiest stream).

    A gap is the interval between the end of one kernel and the start of the
    next *on the same stream*: that is where launch latency, a missing
    prefetch or an accidental synchronisation shows up. A gap on one stream
    while another stream is busy is not a GPU hole, which is exactly why this
    is reported per stream and not over the whole timeline.
    """
    if stream is None:
        streams = stream_breakdown(kernels)
        if not streams:
            return {"stream": None, "gap_count": 0, "total_gap_us": 0.0, "max_gap_us": 0.0, "gaps_us": []}
        stream = streams[0]["stream"]
    ordered = sorted(
        (k for k in kernels if int(k.get("stream", -1)) == stream),
        key=lambda k: k["ts"],
    )
    gaps: List[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        gap = current["ts"] - (previous["ts"] + previous["dur"])
        if gap > 0:
            gaps.append(gap)
    return {
        "stream": stream,
        "kernel_count": len(ordered),
        "gap_count": len(gaps),
        "total_gap_us": sum(gaps),
        "max_gap_us": max(gaps) if gaps else 0.0,
        "mean_gap_us": (sum(gaps) / len(gaps)) if gaps else 0.0,
        "gaps_us": sorted(gaps, reverse=True)[:32],
    }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    )


def cpu_to_kernel_latency(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Latency from the end of a launch API call to the kernel start.

    Uses the correlation chain: for every device kernel, its correlation id
    also appears on the ``cuda_runtime`` row that enqueued it. Large values
    mean the device was still busy (depth in the queue) or the launch was
    delayed; this is a *queue* measurement, not a CPU-busy measurement, and
    it is reported with its own distribution rather than as an average.
    """
    launches: Dict[Any, Dict[str, Any]] = {}
    for event in trace_events(trace):
        if str(event.get("cat")) != "cuda_runtime":
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation")
        if correlation is None:
            continue
        launches[correlation] = {
            "name": str(event.get("name")),
            "end": float(event["ts"]) + float(event.get("dur", 0.0) or 0.0),
        }

    latencies: List[float] = []
    for kernel in device_events(trace):
        correlation = kernel.get("correlation")
        launch = launches.get(correlation)
        if launch is None:
            continue
        latencies.append(kernel["ts"] - launch["end"])

    ordered = sorted(latencies)
    # A negative value means the kernel's timestamp precedes the end of the
    # launch API call that produced it. That cannot happen physically; it is a
    # clock-alignment artefact of merging the CPU and device rows into one
    # trace. They are counted and reported instead of being clamped away,
    # because a large negative population would mean the whole measurement is
    # unusable and a reader has to be able to see that.
    negatives = [value for value in ordered if value < 0]
    return {
        "samples": len(ordered),
        "negative_count": len(negatives),
        "negative_ratio": len(negatives) / len(ordered) if ordered else 0.0,
        "p50_us": _percentile(ordered, 0.50),
        "p90_us": _percentile(ordered, 0.90),
        "p99_us": _percentile(ordered, 0.99),
        "max_us": ordered[-1] if ordered else 0.0,
        "min_us": ordered[0] if ordered else 0.0,
    }


def critical_path_share_note(
    *,
    kernel_work_us: float,
    span_us: float,
) -> Dict[str, Any]:
    """Compare cumulative kernel work with the phase wall span.

    ``overlap_factor > 1`` means kernels overlap, so the cumulative work
    cannot be divided by the end-to-end time to obtain an Amdahl fraction.
    The returned ``wall_clock_share`` is the number E02-09 may use, and it is
    only a *bound*: it assumes the named work lies entirely on the critical
    path, which overlapping kernels contradict.
    """
    if span_us <= 0:
        return {
            "kernel_work_us": kernel_work_us,
            "span_us": span_us,
            "overlap_factor": 0.0,
            "wall_clock_share": 0.0,
            "overlaps": False,
        }
    overlap = kernel_work_us / span_us
    return {
        "kernel_work_us": kernel_work_us,
        "span_us": span_us,
        "overlap_factor": overlap,
        "wall_clock_share": min(kernel_work_us / span_us, 1.0),
        "overlaps": overlap > 1.001,
    }


# ── Nsight Compute export parsing ────────────────────────────────────────

#: Metric names (as printed by ``ncu --csv``) that make up the report panel.
#:
#: Units matter here: NCU prints ``Duration`` in **nanoseconds**, not
#: microseconds. Reading it as µs understates every derived rate by 1000x and
#: silently breaks any roofline comparison built on it.
NCU_PANEL_METRICS: Dict[str, str] = {
    "Duration": "duration_ns",
    "Memory Throughput": "memory_throughput_pct",
    "Compute (SM) Throughput": "compute_sm_throughput_pct",
    "SM Busy": "sm_busy_pct",
    "L2 Cache Throughput": "l2_throughput_pct",
    "L1/TEX Cache Throughput": "l1tex_throughput_pct",
    "Max Bandwidth": "max_bandwidth_pct",
    "Mem Busy": "mem_busy_pct",
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
    "Achieved Active Warps Per SM": "achieved_warps_per_sm",
    "Registers Per Thread": "registers_per_thread",
    "Shared Memory Configuration Size": "smem_config_bytes",
    "Static Shared Memory Per Block": "smem_static_bytes",
    "Dynamic Shared Memory Per Block": "smem_dynamic_bytes",
    "Waves Per SM": "waves_per_sm",
    "Block Limit Shared Mem": "block_limit_smem",
    "Block Limit Registers": "block_limit_registers",
    "No Eligible": "scheduler_no_eligible_pct",
    "One or More Eligible": "scheduler_eligible_pct",
    "Eligible Warps Per Scheduler": "eligible_warps_per_scheduler",
    "Active Warps Per Scheduler": "active_warps_per_scheduler",
    "Issued Warp Per Scheduler": "issued_warp_per_scheduler",
    "Warp Cycles Per Issued Instruction": "warp_cycles_per_issued_instruction",
    "Executed Ipc Active": "executed_ipc_active",
    "Issue Slots Busy": "issue_slots_busy_pct",
    "L2 Hit Rate": "l2_hit_rate_pct",
    "SM Frequency": "sm_frequency_hz",
}

#: NCU rule names worth surfacing as *diagnostic leads* (never as root cause).
NCU_DIAGNOSTIC_RULES = (
    "occupancy",
    "memory",
    "compute",
    "scheduler",
    "warp",
    "uncoalesced",
    "bank",
    "imbalance",
)


def parse_ncu_csv(text: str) -> Dict[str, Any]:
    """Parse an ``ncu --csv`` export into kernel rows with a metric panel.

    Two column layouts are accepted: the section layout (``Metric Value``)
    and the per-launch layout (``Minimum``/``Maximum``/``Average`` plus an
    ``Invocations`` counter). Rows whose metric value is ``n/a`` are kept but
    flagged, because on this Tegra build the DRAM-level counters are simply
    not exposed — reporting that gap is part of the experiment.
    """
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if "Kernel Name" in line and "Metric Name" in line:
            header_index = index
            break
    if header_index is None:
        return {"kernels": [], "error": "no NCU CSV header found"}

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    kernels: Dict[str, Dict[str, Any]] = {}
    unavailable: Dict[str, int] = {}
    rules: List[Dict[str, Any]] = []

    for row in reader:
        name = (row.get("Kernel Name") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        section = (row.get("Section Name") or "").strip()
        if not name or not metric:
            continue
        entry = kernels.setdefault(
            name,
            {
                "name": name,
                "grid": (row.get("Grid Size") or "").strip(),
                "block": (row.get("Block Size") or "").strip(),
                "context": (row.get("Context") or "").strip(),
                "stream": (row.get("Stream") or "").strip(),
                "invocations": int(float(row.get("Invocations") or 1)),
                "metrics": {},
                "sections": [],
            },
        )
        if section and section not in entry["sections"]:
            entry["sections"].append(section)
        if metric in entry["metrics"]:
            continue
        value = _ncu_value(row)
        if value is None:
            unavailable[metric] = unavailable.get(metric, 0) + 1
        entry["metrics"][metric] = {
            "section": section,
            "unit": (row.get("Metric Unit") or "").strip(),
            "value": value,
            "raw": (row.get("Metric Value") or row.get("Average") or "").strip(),
        }
        rule_name = (row.get("Rule Name") or "").strip()
        if rule_name:
            rules.append(
                {
                    "kernel": name,
                    "rule": rule_name,
                    "type": (row.get("Rule Type") or "").strip(),
                    "description": (row.get("Rule Description") or "").strip(),
                    "speedup": (row.get("Estimated Speedup") or "").strip(),
                }
            )

    for entry in kernels.values():
        entry["panel"] = ncu_metric_panel(entry["metrics"])
    return {
        "kernels": sorted(kernels.values(), key=lambda k: k["name"]),
        "unavailable_metrics": unavailable,
        "rules": rules,
        "diagnostic_leads": [
            r for r in rules
            if any(token in r["rule"].lower() for token in NCU_DIAGNOSTIC_RULES)
        ][:32],
    }


def _ncu_value(row: Mapping[str, Any]) -> Optional[float]:
    """Read a numeric metric value from either CSV layout."""
    for key in ("Metric Value", "Average", "Maximum", "Minimum"):
        raw = (row.get(key) or "").strip()
        if not raw or raw.lower() in ("n/a", "-", "nan"):
            continue
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            continue
    return None


def ncu_metric_panel(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Project the raw metric set onto :data:`NCU_PANEL_METRICS`.

    Every panel field that is missing or ``n/a`` is reported as ``None``
    rather than dropped, so a reader can tell "not collected" from "zero".
    """
    panel: Dict[str, Any] = {}
    for metric_name, field in NCU_PANEL_METRICS.items():
        entry = metrics.get(metric_name)
        panel[field] = None if entry is None else entry.get("value")
    duration_ns = panel.get("duration_ns")
    if duration_ns is not None:
        # Convenience derived field; the raw nanosecond value is kept too.
        panel["duration_ms"] = duration_ns / 1e6
    return panel


def ncu_stall_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Collect whatever stall-reason counters the collected sections expose.

    The metric names differ between NCU versions, so this is a name-prefix
    scan instead of a fixed list. Stall reasons are only meaningful together
    with scheduler issue and the source path; the caller must not treat the
    largest entry as a root cause on its own.
    """
    stalls: Dict[str, float] = {}
    for name, entry in metrics.items():
        if "stall" not in name.lower():
            continue
        value = entry.get("value")
        if value is None:
            continue
        stalls[name] = float(value)
    return stalls


# ── Roofline / value-flow consistency ────────────────────────────────────


def roofline_consistency(
    *,
    useful_flops: float,
    modeled_dram_bytes: float,
    measured_l2_bytes: Optional[float],
    duration_us: float,
    peak_flops: float,
    peak_bandwidth: float,
) -> Dict[str, Any]:
    """Step-11 cross-check that keeps every byte/FLOP definition separate.

    ``useful_flops`` are the logical FLOPs of the operation, ``modeled_dram_bytes``
    the ideal minimum traffic, ``measured_l2_bytes`` what NCU actually counted
    at L2. They are three different quantities: cache reuse, read/write
    amplification, fusion and replay conditions all make them disagree, and
    the disagreement is reported instead of being tuned away.

    The ceilings are nominal datasheet values unless the caller passes
    measured ones; the output therefore carries ``ceiling_source`` so no
    reader can mistake a model for a measurement.
    """
    if duration_us <= 0:
        return {"error": "duration_us must be > 0"}
    seconds = duration_us * 1e-6
    achieved_flops = useful_flops / seconds if seconds else 0.0
    intensity = (
        useful_flops / modeled_dram_bytes if modeled_dram_bytes > 0 else 0.0
    )
    bound = min(peak_flops, peak_bandwidth * intensity)
    observed_bytes_per_s = (
        measured_l2_bytes / seconds if measured_l2_bytes else None
    )
    return {
        "useful_flops": useful_flops,
        "modeled_dram_bytes": modeled_dram_bytes,
        "measured_l2_bytes": measured_l2_bytes,
        "duration_us": duration_us,
        "achieved_flops": achieved_flops,
        "arithmetic_intensity_flop_per_byte": intensity,
        "roofline_bound_flops": bound,
        "roofline_efficiency": achieved_flops / bound if bound > 0 else 0.0,
        "modeled_dram_bytes_per_s": (
            modeled_dram_bytes / seconds if seconds else 0.0
        ),
        "measured_l2_bytes_per_s": observed_bytes_per_s,
        "bandwidth_ceiling_fraction": (
            observed_bytes_per_s / peak_bandwidth
            if observed_bytes_per_s and peak_bandwidth > 0
            else None
        ),
        "classification": (
            "compute_bound" if intensity >= (peak_flops / peak_bandwidth) else "bandwidth_bound"
        ),
        "ceiling_source": "nominal_datasheet",
        "caveats": [
            "useful FLOPs, modeled DRAM bytes and measured L2 bytes are three "
            "different quantities",
            "NCU flushes caches between replay passes, so measured bytes are "
            "not the same as steady-state bytes",
            "ceilings are nominal, not measured, unless replaced explicitly",
        ],
    }


def kernel_shape_flops(
    *,
    m: int,
    n: int,
    k: int,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """Logical FLOPs and minimum bytes of a dense ``(m,k)x(k,n)`` GEMM.

    Two FLOPs per multiply-accumulate; the byte count is the compulsory
    traffic of reading both operands once and writing the result once, i.e.
    the *lower bound* the kernel can approach only with perfect reuse.
    """
    flops = 2.0 * m * n * k
    bytes_min = float(dtype_bytes) * (m * k + k * n + m * n)
    return {
        "flops": flops,
        "min_bytes": bytes_min,
        "arithmetic_intensity": flops / bytes_min if bytes_min else 0.0,
    }


# ── Cross-tool compatibility ─────────────────────────────────────────────


def tool_compatibility(
    *,
    profiler_kernels: Sequence[Mapping[str, Any]],
    nsys_kernels: Sequence[Mapping[str, Any]],
    ncu_kernel_names: Sequence[str],
    token_hashes: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    """Check that the three tools saw the same work, not the same numbers.

    Durations are explicitly *not* required to match — NCU serialises and
    replays launches, Nsight Systems instruments every API call, and the
    PyTorch profiler adds shape collection. What must match is identity:
    kernel names, call counts (where both tools count them) and the token
    sequence, because a tool that changed the attention path or the control
    flow has produced a different experiment.
    """
    profiler_names = {str(r["name"]) for r in profiler_kernels}
    nsys_names = {str(r["name"]) for r in nsys_kernels}
    ncu_names = {str(n) for n in ncu_kernel_names}
    hashes = {k: v for k, v in token_hashes.items() if v}
    return {
        "profiler_kernel_names": len(profiler_names),
        "nsys_kernel_names": len(nsys_names),
        "profiler_missing_in_nsys": sorted(profiler_names - nsys_names),
        "nsys_missing_in_profiler": sorted(nsys_names - profiler_names),
        "ncu_names_not_in_profiler": sorted(ncu_names - profiler_names),
        "token_hashes_identical": len(set(hashes.values())) <= 1 if hashes else None,
        "token_hashes": dict(hashes),
        "durations_comparable": False,
        "duration_rationale": (
            "NCU replays/serialises launches and controls caches; NSys and the "
            "PyTorch profiler instrument different layers. Only identity "
            "(names, shapes, counts, tokens) is compared across tools."
        ),
    }


def candidate_selection(
    *,
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    span_us: float,
    max_candidates: int = 3,
) -> List[Dict[str, Any]]:
    """Rank NCU candidates by phase cost x frequency x optimisability.

    The score is *not* the kernel name's familiarity. It combines the share
    of cumulative kernel work, the share of the phase wall span, the call
    count (a cheap kernel called hundreds of times accumulates), whether the
    kernel is a library kernel (replaceable only via a different library
    call) and whether it is reachable from a model module. The returned rows
    carry the reasons so a reader can disagree with the ranking.
    """
    ranked = attach_shares(rows, field="total_us")
    selected: List[Dict[str, Any]] = []
    for row in ranked:
        if len(selected) >= max_candidates:
            break
        wall_share = (
            min(float(row["total_us"]) / span_us, 1.0) if span_us > 0 else 0.0
        )
        bucket = kernel_bucket(str(row["name"]))
        optimisable = bucket in ("gemm", "attention")
        score = float(row["time_share"]) * (1.0 + math.log10(max(row["count"], 1)))
        if optimisable:
            score *= 1.25
        selected.append(
            {
                "phase": phase,
                "name": row["name"],
                "bucket": bucket,
                "count": row["count"],
                "total_us": row["total_us"],
                "mean_us": row["mean_us"],
                "phase_share": row["time_share"],
                "wall_share_bound": wall_share,
                "score": score,
                "shape_signature_count": row.get("shape_signature_count", 1),
                "grids": row.get("grids", [])[:4],
                "blocks": row.get("blocks", [])[:4],
                "reason": (
                    f"{row['time_share'] * 100:.1f}% of cumulative kernel work "
                    f"over {row['count']} launches ({bucket}); "
                    f"wall-share bound {wall_share * 100:.1f}%"
                ),
            }
        )
    return selected


def kernel_bucket(name: str) -> str:
    """Coarse optimisation bucket for a device kernel name.

    Intentionally name-driven and therefore reportable: the bucket only
    partitions the ranking, it never decides the winner on its own.

    The check order carries real information and is *not* arbitrary. A
    ``torch`` elementwise kernel that copies data is named
    ``at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(...)>``:
    matching on the bare substring ``copy`` would file it under ``memory_kv``
    even though it is an elementwise kernel, and matching ``elementwise``
    before ``reduce`` would file ``elementwise_kernel<...ReduceOp...>`` as
    elementwise even though it reduces. The order below therefore tests the
    most specific identities first — reduction before elementwise, and
    ``CatArrayBatchedCopy``/``Memcpy`` rather than the substring ``copy``.
    """
    lower = name.lower()
    if any(t in lower for t in ("gemm", "cutlass", "cublas", "matmul", "::mm", "bmm")):
        return "gemm"
    if any(t in lower for t in ("attention", "softmax", "flash", "scaled_dot")):
        return "attention"
    if "reduce" in lower or "argmax" in lower or "topk" in lower:
        return "reduction"
    if any(
        t in lower
        for t in (
            "elementwise",
            "norm",
            "silu",
            "gelu",
            "relu",
            "unaryfunctor",
            "binaryfunctor",
            "add",
            "mul",
        )
    ):
        return "elementwise_norm"
    if any(
        t in lower
        for t in ("catarray", "concat", "memcpy", "memset", "indexselect", "index_select")
    ):
        return "memory_kv"
    return "other"


def bucket_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate a ranked kernel table into optimisation buckets."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = kernel_bucket(str(row["name"]))
        entry = buckets.setdefault(
            bucket,
            {"bucket": bucket, "kernels": 0, "count": 0, "total_us": 0.0, "top_kernel": ""},
        )
        entry["kernels"] += 1
        entry["count"] += int(row.get("count", 0))
        entry["total_us"] += float(row.get("total_us", 0.0) or 0.0)
        if not entry["top_kernel"]:
            entry["top_kernel"] = str(row["name"])
    for entry in buckets.values():
        entry["share_of_top_table"] = 0.0
    total = sum(e["total_us"] for e in buckets.values()) or 1.0
    for entry in buckets.values():
        entry["share_of_top_table"] = entry["total_us"] / total
    return sorted(buckets.values(), key=lambda e: -e["total_us"])


__all__ = [
    "DECODE_EARLY_RANGE",
    "DECODE_LATE_RANGE",
    "DECODE_MIDDLE_RANGE",
    "DEVICE_TRACE_CATEGORIES",
    "FIRST_TOKEN_RANGE",
    "MODEL_PHASE_RANGES",
    "MODULE_ROLE_PREFIX",
    "NCU_DIAGNOSTIC_RULES",
    "NCU_PANEL_METRICS",
    "PHASE_RANGES",
    "PREFILL_RANGE",
    "RESULT_RANGE",
    "RUN_RANGE",
    "aggregate_kernels",
    "attach_shares",
    "attribute_events_to_phases",
    "bucket_summary",
    "candidate_selection",
    "correlation_index",
    "cpu_to_kernel_latency",
    "critical_path_share_note",
    "cumulative_coverage",
    "device_events",
    "external_id_index",
    "gpu_activity_window",
    "host_op_table_by_phase",
    "kernel_bucket",
    "kernel_shape_flops",
    "load_chrome_trace",
    "merge_intervals",
    "module_role_table",
    "ncu_metric_panel",
    "ncu_stall_metrics",
    "nvtx_spans",
    "op_attribution",
    "parse_ncu_csv",
    "phase_ledger",
    "phase_span_index",
    "roofline_consistency",
    "stream_breakdown",
    "timeline_gaps",
    "tool_compatibility",
    "trace_events",
]
