#!/usr/bin/env python3
"""E03-02 — RMSNorm shape 性能热力图与适用区间.

Implements the twelve protocol steps of
``docs/stage_experiments/details/S03/E03-02_shape_performance_heatmap.md`` and
records every field of §8: raw device-event and host submit+completion samples,
per-case correctness against the E03-01 reference, per-case resources, the
bytes/FLOPs conventions, the shape heatmaps (including winners, ties,
unsupported cells and regressions), the E03-04 NCU candidate selection and the
E03-08 routing candidates.

Design constraints honoured here
-------------------------------
* Every timed call is **forced** through the C ABI variant code; ``auto`` is
  never used, because E03-01 proved the resolved variant is not observable.
* Timing loops call the C ABI directly (``ctypes``), so the measured
  ``host_ms`` is the C++ API + launch + sync cost. The Python bridge cost
  (validation included) is measured separately as ``bridge_ms`` and is never
  used as the operator latency.
* Every case is compared against the E03-01 reference primitives
  (:mod:`hqsb.benchmark.rmsnorm_correctness`) before it is timed, and its
  output hash is re-checked after the timing loops.
* The protocol (warmup, launches per group, group count, case order) is
  frozen into ``protocol.json`` by :func:`run_pilot` and *refuses to run* if it
  is missing: no number is chosen after seeing a result.
* Unsupported (dtype, variant) pairs stay in the matrix as cells and must be
  demonstrably rejected; a later failure is never relabelled "unsupported".

Subcommands::

    python3 scripts/audit/run_e03_02_shape_heatmap.py pilot     --output-dir <dir>
    python3 scripts/audit/run_e03_02_shape_heatmap.py collect   --output-dir <dir>
    python3 scripts/audit/run_e03_02_shape_heatmap.py verify    --output-dir <dir>
    python3 scripts/audit/run_e03_02_shape_heatmap.py summarize --output-dir <dir>
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AUDIT_DIR = _REPO_ROOT / "scripts" / "audit"
if str(_AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(_AUDIT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_e03_01_rmsnorm_semantics as e01  # noqa: E402
from hqsb.benchmark import rmsnorm_correctness as rc  # noqa: E402
from hqsb.benchmark import rmsnorm_perf as rp  # noqa: E402
from hqsb.benchmark.hotspot_decision import GUARD_BAND  # noqa: E402
from hqsb.benchmark.resource_monitor import TegrastatsMonitor  # noqa: E402
from hqsb.benchmark.tegrastats_parser import (  # noqa: E402
    compute_resource_summary,
    parse_tegrastats_line,
    slice_records,
)
from ops import cuda_bridge  # noqa: E402
from ops.capability import detect_capabilities  # noqa: E402

EXPERIMENT_ID = "E03-02"
STAGE = "S03"
DEFAULT_OUTPUT_DIR = "docs/stage_experiments/S03/E03-02/raw"
DEFAULT_S02_CENSUS = "docs/stage_experiments/S02/E02-02/raw_v2"
S02_CENSUS_DEFAULT_ABS = _REPO_ROOT / DEFAULT_S02_CENSUS

#: Frozen protocol defaults (the pilot may re-derive the launch counts, but the
#: window target and the clamps are part of this file, not of the results).
PROCESSES_DEFAULT = 3
GROUPS_DEFAULT = 4
WARMUP_DEFAULT = 15
TARGET_WINDOW_MS_DEFAULT = 3.0
MIN_LAUNCHES_DEFAULT = 30
MAX_LAUNCHES_DEFAULT = 200
PILOT_GROUPS = 1
PILOT_WARMUP = 5
PILOT_LAUNCHES = 20
BRIDGE_SAMPLE_LAUNCHES = 40
ORDER_SEED_BASE = 20260918

#: Sentinel written into the output buffer before every launch (same value the
#: E03-01 contract tests used), so "the kernel never ran" is visible.
OUTPUT_SENTINEL = 1234.0

GPU_CLOCK_GLOB = "/sys/class/devfreq/*gpu*/cur_freq"

STATUS_MEASURED = "MEASURED"
STATUS_FAIL_CORRECTNESS = "FAIL_CORRECTNESS"
STATUS_UNSUPPORTED = "EXPECTED_UNSUPPORTED"
STATUS_UNSUPPORTED_NOT_REJECTED = "EXPECTED_UNSUPPORTED_NOT_REJECTED"

#: Required heatmap views (protocol §8). ``(name, kind)``.
HEATMAP_VIEWS: Tuple[Tuple[str, str], ...] = (
    ("device_latency", "numeric"),
    ("host_submit_completion", "numeric"),
    ("effective_gbps", "numeric"),
    ("speedup_vs_v0", "numeric"),
    ("status_mask", "categorical"),
    ("winner_and_tie", "categorical"),
    ("s02_source_coverage", "categorical"),
)


# ══════════════════════════════════════════════════════════════════════
# small helpers
# ══════════════════════════════════════════════════════════════════════


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def _read_gpu_clock_hz() -> Optional[int]:
    for pattern in (GPU_CLOCK_GLOB, "/sys/devices/gpu.0/devfreq/*/cur_freq"):
        for path in sorted(glob.glob(pattern)):
            try:
                raw = Path(path).read_text(encoding="utf-8").strip()
                if raw:
                    return int(raw)
            except Exception:
                continue
    return None


def _run(cmd: Sequence[str], *, timeout: float = 20.0) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout
        )
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": " ".join(cmd), "error": repr(exc)}


def _to_cuda(array: np.ndarray, dtype: str) -> "torch.Tensor":
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    return torch.from_numpy(np.ascontiguousarray(array)).to(
        device="cuda", dtype=torch_dtype
    )


# ══════════════════════════════════════════════════════════════════════
# step 1 — frozen identity and machine state
# ══════════════════════════════════════════════════════════════════════


def freeze_machine_state() -> Dict[str, Any]:
    """Device UUID/CC/SMs/driver/runtime plus power and clock policy."""
    state: Dict[str, Any] = {
        "gpu_clock_hz_observed_at_freeze": _read_gpu_clock_hz(),
        "gpu_clock_source": GPU_CLOCK_GLOB,
        "nvpmodel": _run(["nvpmodel", "-q"]),
        "jetson_clocks": _run(["jetson_clocks", "--show"]),
        "cuda_driver": _run(
            [
                "python3",
                "-c",
                "import torch;print(torch.cuda.get_device_properties(0))",
            ]
        ),
        "nvidia_smi": _run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv"]
        ),
        "thermal_zones": {},
    }
    for zone in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*")):
        try:
            name = Path(zone, "type").read_text(encoding="utf-8").strip()
            raw = Path(zone, "temp").read_text(encoding="utf-8").strip()
            if raw:
                state["thermal_zones"][name] = int(raw) / 1000.0
        except Exception:
            # some sysfs entries raise or return an empty read on this kernel;
            # a missing zone must not abort the run
            continue
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        state["device"] = {
            "name": prop.name,
            "major": int(prop.major),
            "minor": int(prop.minor),
            "multi_processor_count": int(prop.multi_processor_count),
            "total_memory_bytes": int(prop.total_memory),
            "uuid": str(getattr(prop, "uuid", "")) or None,
        }
        free_b, total_b = torch.cuda.mem_get_info()
        state["memory_free_bytes"] = int(free_b)
        state["memory_total_bytes"] = int(total_b)
    state["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return state


def static_resource_usage(so_path: Optional[str]) -> Dict[str, Any]:
    """Register/shared/stack usage per kernel, read from the built binary.

    ``cuobjdump`` is a *static* read of the ELF; it does not run the kernel and
    cannot disturb a measurement (it runs before the timing loops). If the tool
    is absent the gap is recorded instead of being papered over.
    """
    result: Dict[str, Any] = {
        "so_path": so_path,
        "cuobjdump_available": False,
        "per_kernel": {},
        "raw_excerpt": None,
    }
    cuobjdump = None
    for candidate in (
        "/usr/local/cuda/bin/cuobjdump",
        "/usr/local/cuda-12.6/bin/cuobjdump",
    ):
        if Path(candidate).is_file():
            cuobjdump = candidate
            break
    if cuobjdump is None:
        which = _run(["which", "cuobjdump"])
        cuobjdump = which.get("stdout") or None
    if not cuobjdump or not so_path or not Path(so_path).is_file():
        result["note"] = (
            "cuobjdump or the shared library is unavailable; per-kernel register "
            "counts are not recorded (occupancy is still recorded via the C ABI)"
        )
        return result

    run = _run([cuobjdump, "--dump-resource-usage", so_path], timeout=120.0)
    result["cuobjdump_available"] = run.get("returncode") == 0
    text = run.get("stdout") or ""
    result["raw_excerpt"] = text[:20000]
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function "):
            current = stripped[len("Function ") :].strip()
            result["per_kernel"].setdefault(current, {})
        elif current and "REG:" in stripped:
            fields = {}
            for token in stripped.split():
                if ":" in token:
                    key, _, value = token.partition(":")
                    fields[key] = value
            result["per_kernel"][current] = fields
    return result


def occupancy_matrix(block_sizes: Sequence[int] = (32, 64, 128, 256, 512, 1024)) -> Dict[str, Any]:
    """Theoretical max active blocks/SM per (dtype, variant, block size).

    Recorded for the dispatch configuration used by the timed cases *and* for
    the sweep range E03-04 will own.
    """
    out: Dict[str, Any] = {"block_sizes": list(block_sizes), "entries": {}}
    for dtype in rp.DTYPES:
        for variant in rp.VARIANT_SUPPORT[dtype]:
            for block in block_sizes:
                key = f"{dtype}|{variant}|{block}"
                try:
                    value = cuda_bridge.rmsnorm_occupancy(
                        variant=variant, dtype=dtype, block_size=block
                    )
                except Exception as exc:  # pragma: no cover - environment dependent
                    value = None
                    out["entries"][key] = {"occupancy": None, "error": repr(exc)}
                    continue
                out["entries"][key] = {"occupancy": value}
    for dtype, variants in rp.UNSUPPORTED_VARIANT_PAIRS.items():
        for variant in variants:
            for block in (256,):
                key = f"{dtype}|{variant}|{block}"
                out["entries"][key] = {
                    "occupancy": cuda_bridge.rmsnorm_occupancy(
                        variant=variant, dtype=dtype, block_size=block
                    ),
                    "note": "claimed-unsupported pair: C ABI must return 0",
                }
    return out


# ══════════════════════════════════════════════════════════════════════
# step 2 — import the S02 runtime shape ledger
# ══════════════════════════════════════════════════════════════════════


def import_s02_shapes(census_dir: Path) -> Dict[str, Any]:
    """Import the real shapes *and* validate the derived call matrix against it."""
    census = e01.import_s02_shape_census(census_dir)
    plan = {entry["shape_id"]: entry for entry in rp.shape_plan()}
    real_shapes = sorted(
        {entry["shape_id"] for entry in plan.values() if "S02_RUNTIME" in entry["sources"]}
    )
    census_shapes = sorted(
        {
            rp.shape_id(int(_shape_rows(sig)), int(_shape_hidden(sig)))
            for sig in census["unique_signatures"]
            if _shape_rows(sig) and _shape_hidden(sig)
        }
    )
    missing_from_census = sorted(set(real_shapes) - set(census_shapes))
    extra_in_census = sorted(set(census_shapes) - set(real_shapes))

    call_matrix = rp.s02_call_matrix()
    validation: List[Dict[str, Any]] = []
    for sig in census["unique_signatures"]:
        rows, hidden = _shape_rows(sig), _shape_hidden(sig)
        if not rows or not hidden:
            continue
        sid = rp.shape_id(rows, hidden)
        workload = (sig.get("workload") or {}).get("name")
        expected = sorted(
            {
                row["calls_per_module_instance"]
                for row in call_matrix
                if row["shape_id"] == sid
                and row["workload"] == workload
                and row["phase"] == sig.get("phase")
            }
        )
        observed = sig.get("call_count")
        validation.append(
            {
                "module": sig.get("module"),
                "phase": sig.get("phase"),
                "workload": workload,
                "shape_id": sid,
                "census_call_count": observed,
                "derived_calls_per_module_instance": expected,
                "consistent": (
                    bool(expected) and observed in expected
                    if observed is not None
                    else None
                ),
            }
        )
    mismatches = [row for row in validation if row["consistent"] is False]
    return {
        "census_dir": str(census_dir),
        "census_exists": census["exists"],
        "runs_imported": len(census["runs"]),
        "unique_signature_count": len(census["unique_signatures"]),
        "census_shapes": census_shapes,
        "planned_real_shapes": real_shapes,
        "missing_from_census": missing_from_census,
        "extra_in_census": extra_in_census,
        "call_matrix": call_matrix,
        "shape_call_weights": rp.shape_call_weights(),
        "call_matrix_validation": validation,
        "call_matrix_mismatch_count": len(mismatches),
    }


def _shape_rows(sig: Mapping[str, Any]) -> Optional[int]:
    shape = _shape_tuple(sig.get("input_shape"))
    if not shape:
        return None
    if len(shape) == 3:
        return shape[0] * shape[1]
    if len(shape) == 4:
        return shape[0] * shape[1] * shape[2]
    return None


def _shape_hidden(sig: Mapping[str, Any]) -> Optional[int]:
    shape = _shape_tuple(sig.get("input_shape"))
    return shape[-1] if shape else None


def _shape_tuple(raw: Any) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().strip("[]()")
        if not text:
            return None
        try:
            return tuple(int(part.strip()) for part in text.split(","))
        except ValueError:
            return None
    if isinstance(raw, (list, tuple)):
        try:
            return tuple(int(v) for v in raw)
        except (TypeError, ValueError):
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# raw C ABI launcher (timing path, no Python-side validation)
# ══════════════════════════════════════════════════════════════════════


class RawLauncher:
    """Direct ``ctypes`` handle on ``hqsb_rmsnorm_forward_ex_c``.

    The bridge's Python-side validation is *not* on the timing path: it is
    measured separately so "host submit + completion" means "C ABI + launch +
    sync" and not "CPython argument marshalling".
    """

    def __init__(self, lib_path: str) -> None:
        self.lib = ctypes.CDLL(lib_path)
        if not hasattr(self.lib, "hqsb_rmsnorm_forward_ex_c"):
            raise RuntimeError(
                f"{lib_path} does not export hqsb_rmsnorm_forward_ex_c; rebuild "
                "the operator library first"
            )
        fn = self.lib.hqsb_rmsnorm_forward_ex_c
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
        self.fn = fn

    def bind(
        self,
        *,
        x: "torch.Tensor",
        w: "torch.Tensor",
        out: "torch.Tensor",
        rows: int,
        hidden: int,
        epsilon: float,
        dtype: str,
        variant: str,
        stream_ptr: int,
    ):
        fn = self.fn
        xp, wp, op = x.data_ptr(), w.data_ptr(), out.data_ptr()
        rows_i, hidden_i = int(rows), int(hidden)
        eps = float(epsilon)
        dtype_code = rp.DTYPE_CODE[dtype]
        variant_code = rp.VARIANT_CODE[variant]
        stream = ctypes.c_void_p(stream_ptr)

        def launch() -> int:
            return int(
                fn(xp, wp, op, rows_i, hidden_i, eps, dtype_code, variant_code, stream)
            )

        return launch


# ══════════════════════════════════════════════════════════════════════
# timing (protocol §5)
# ══════════════════════════════════════════════════════════════════════


def time_configuration(
    *,
    launch,
    bridge_launch,
    stream: "torch.cuda.Stream",
    warmup: int,
    n_launches: int,
    groups: int,
) -> Dict[str, Any]:
    """Device-event and host submit+completion samples for one configuration.

    The device loop records one CUDA event pair around ``n_launches`` back to
    back launches and divides by ``n`` (no per-iteration synchronisation). The
    host loop starts a CPU monotonic clock *before* submitting the same ``n``
    calls, takes a pure-submit timestamp, then synchronises once and divides by
    ``n`` - so both "submit cost" and "submit + completion cost" are reported
    and neither is a per-iteration-synchronised number.
    """
    device_ms: List[float] = []
    host_ms: List[float] = []
    submit_ms: List[float] = []
    immediate_status: List[int] = []
    errors: List[str] = []
    last = 0

    for _ in range(groups):
        for _ in range(warmup):
            last = launch()
        if last != 0:
            errors.append(f"warmup launch returned cudaError={last}")
            break
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        last = 0
        for _ in range(n_launches):
            last = launch()
        stop.record(stream)
        stop.synchronize()
        immediate_status.append(int(last))
        device_ms.append(float(start.elapsed_time(stop)) / float(n_launches))
        if last != 0:
            errors.append(f"device loop returned cudaError={last}")
            break

        t0 = time.perf_counter_ns()
        last = 0
        for _ in range(n_launches):
            last = launch()
        t1 = time.perf_counter_ns()
        torch.cuda.synchronize()
        t2 = time.perf_counter_ns()
        immediate_status.append(int(last))
        submit_ms.append((t1 - t0) / float(n_launches) / 1e6)
        host_ms.append((t2 - t0) / float(n_launches) / 1e6)
        if last != 0:
            errors.append(f"host loop returned cudaError={last}")
            break

    bridge_ms: List[float] = []
    if bridge_launch is not None and not errors:
        for _ in range(warmup):
            bridge_launch()
        torch.cuda.synchronize()
        b0 = time.perf_counter_ns()
        for _ in range(BRIDGE_SAMPLE_LAUNCHES):
            bridge_launch()
        torch.cuda.synchronize()
        b1 = time.perf_counter_ns()
        bridge_ms.append((b1 - b0) / float(BRIDGE_SAMPLE_LAUNCHES) / 1e6)

    return {
        "device_ms_groups": device_ms,
        "host_ms_groups": host_ms,
        "submit_ms_groups": submit_ms,
        "bridge_ms_samples": bridge_ms,
        "immediate_cuda_status": immediate_status,
        "errors": errors,
    }


def measure_device_ceiling(stream: "torch.cuda.Stream", *, mib: int = 64) -> Dict[str, Any]:
    """Library-level steady-state bandwidth reference for the heatmap context.

    A plain ``copy_`` (read+write) and a plain reduction (read) give an order of
    magnitude for what the memory path can do on this board *at the observed
    clock*, so "effective GB/s" can be read relative to something measured
    rather than only against the datasheet. It is a library measurement: it is
    never used as the operator's DRAM traffic.

    The copies run on the **default stream** and the events are recorded on the
    **default stream**, so the bracket actually wraps the measured work (the
    operator's timed loops use the explicit ``stream``, but this reference has
    no reason to).
    """
    elements = (mib * 1024 * 1024) // 4
    result: Dict[str, Any] = {
        "mib_per_buffer": mib,
        "elements": elements,
        "note": (
            "library-level reference (torch copy_/sum), measured on the default "
            "stream at the observed clock; NOT the operator's DRAM traffic and "
            "not a hardware peak"
        ),
    }
    try:
        a = torch.empty(elements, dtype=torch.float32, device="cuda").normal_()
        b = torch.empty_like(a)
        for _ in range(5):
            b.copy_(a)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        iterations = 20
        start.record()
        for _ in range(iterations):
            b.copy_(a)
        stop.record()
        torch.cuda.synchronize()
        copy_ms = float(start.elapsed_time(stop)) / iterations
        result["copy_ms"] = copy_ms
        result["copy_GBps"] = 2.0 * elements * 4 / (copy_ms / 1e3) / 1e9

        start.record()
        acc = None
        for _ in range(iterations):
            acc = torch.sum(a)
        stop.record()
        torch.cuda.synchronize()
        read_ms = float(start.elapsed_time(stop)) / iterations
        result["read_ms"] = read_ms
        result["read_GBps"] = elements * 4 / (read_ms / 1e3) / 1e9
        result["sample_value"] = float(acc) if acc is not None else None
    except Exception as exc:  # pragma: no cover - environment dependent
        result["error"] = repr(exc)
    finally:
        try:
            del a, b  # type: ignore[possibly-undefined]
            torch.cuda.empty_cache()
        except Exception:
            pass
    return result


# ══════════════════════════════════════════════════════════════════════
# step 5 — correctness probe against the E03-01 reference
# ══════════════════════════════════════════════════════════════════════


def correctness_probe(
    *,
    case: Mapping[str, Any],
    x: "torch.Tensor",
    w: "torch.Tensor",
    out: "torch.Tensor",
    x_np: np.ndarray,
    reference_fp64: np.ndarray,
    epsilon: float,
    stream: "torch.cuda.Stream",
    input_hash: str,
    weight_hash: str,
) -> Dict[str, Any]:
    """Forced execution + E03-01 reference comparison for one case."""
    rows, hidden, dtype = int(case["rows"]), int(case["hidden"]), str(case["dtype"])
    out.fill_(OUTPUT_SENTINEL)
    torch.cuda.synchronize()
    launched = False
    reason = None
    reason_code = None
    try:
        cuda_bridge.rmsnorm_forward(
            x,
            w,
            dtype=dtype,
            variant=str(case["variant"]),
            epsilon=float(epsilon),
            out=out,
            stream=stream,
        )
        launched = True
    except cuda_bridge.RmsNormContractError as exc:
        reason, reason_code = str(exc), exc.reason_code
    except RuntimeError as exc:
        reason, reason_code = str(exc), "CUDA_LAUNCH_ERROR"
    except Exception as exc:  # pragma: no cover - defensive
        reason, reason_code = repr(exc), "UNEXPECTED_ERROR"

    completion_status: Optional[int] = None
    if launched:
        try:
            torch.cuda.synchronize()
            completion_status = 0
        except RuntimeError as exc:
            completion_status = 1
            reason = (reason or "") + f" | async: {exc}"

    if not launched:
        return {
            "launched": False,
            "api_reason": reason,
            "api_reason_code": reason_code,
            "completion_cuda_status": completion_status,
            "correctness_status": "FAIL_CORRECTNESS",
            "failed_checks": ["launched"],
        }

    cand_np = out.detach().cpu().numpy().reshape(rows, hidden)
    output_hash = rc.sha256_bytes(np.ascontiguousarray(cand_np).tobytes())
    input_after = rc.sha256_bytes(x.detach().cpu().numpy().tobytes())
    weight_after = rc.sha256_bytes(w.detach().cpu().numpy().tobytes())

    cand_cls = rc.classify(cand_np)
    exp_cls = rc.expected_classes(x_np, rows, hidden)
    class_mismatch = int(np.count_nonzero(cand_cls != exp_cls))
    cand64 = cand_np.astype(np.float64).reshape(-1)
    finite_mask = np.isfinite(reference_fp64) & np.isfinite(cand64)

    tolerance = rc.tolerance_for(dtype, hidden)
    metrics = rc.error_metrics(cand_np, reference_fp64, finite_mask)
    violations = rc.elementwise_violations(cand_np, reference_fp64, tolerance, finite_mask)
    mismatch = rc.first_mismatch(
        candidate=cand_np,
        reference=reference_fp64,
        candidate_classes=cand_cls,
        expected_class_array=exp_cls,
        tolerance=tolerance,
        hidden=hidden,
    )
    zero_mask = rc.inf_row_finite_lanes_exact_zero(x_np, rows, hidden)
    zero_violations = int(np.count_nonzero(cand64[zero_mask] != 0.0))

    cosine_ok: Optional[bool] = None
    if metrics["cosine_applicable"]:
        cosine_ok = bool(metrics["cosine"] >= tolerance["cosine_min"])
    na_reasons: Dict[str, str] = {}
    if not metrics["applicable"]:
        na_reasons["numeric_within_tolerance"] = "no finite element pair"
    if cosine_ok is None:
        na_reasons["cosine_within_min"] = "cosine denominator is zero (NOT_APPLICABLE)"

    checks: Dict[str, Optional[bool]] = {
        "launched": True,
        "completion_cuda_status_ok": completion_status == 0,
        "output_shape_ok": tuple(cand_np.shape) == (rows, hidden),
        "output_dtype_ok": cand_np.dtype
        == (np.float16 if dtype == "fp16" else np.float32),
        "classification_matches_analytic_oracle": class_mismatch == 0,
        "numeric_within_tolerance": bool(violations["passed"]),
        "inf_row_finite_lanes_exactly_zero": zero_violations == 0,
        "cosine_within_min": cosine_ok,
        "l2rel_within_tolerance": (
            bool(metrics["l2rel"] <= tolerance["l2rel"]) if metrics["applicable"] else None
        ),
        "input_unchanged": input_after == input_hash,
        "weight_unchanged": weight_after == weight_hash,
    }
    verdict = rc.judge_case(checks, na_reasons)
    return {
        "launched": True,
        "api_reason": reason,
        "api_reason_code": reason_code,
        "completion_cuda_status": completion_status,
        "output_hash": output_hash,
        "input_after_hash": input_after,
        "weight_after_hash": weight_after,
        "metrics": metrics,
        "numeric_violations": violations,
        "tolerance": tolerance,
        "classification_mismatch_count": class_mismatch,
        "exact_zero_required_count": int(zero_mask.sum()),
        "exact_zero_violation_count": zero_violations,
        "max_allowed_ratio": violations.get("max_allowed_ratio"),
        "first_mismatch": mismatch,
        "correctness_status": verdict["status"],
        "failed_checks": verdict["failed_checks"],
        "checks": checks,
    }


def probe_unsupported(
    *,
    case: Mapping[str, Any],
    x: "torch.Tensor",
    w: "torch.Tensor",
    epsilon: float,
    stream: "torch.cuda.Stream",
) -> Dict[str, Any]:
    """A claimed-unsupported pair must be rejected *and* leave output untouched."""
    out = torch.full_like(x, OUTPUT_SENTINEL)
    torch.cuda.synchronize()
    launched = False
    reason = None
    reason_code = None
    try:
        cuda_bridge.rmsnorm_forward(
            x,
            w,
            dtype=str(case["dtype"]),
            variant=str(case["variant"]),
            epsilon=float(epsilon),
            out=out,
            stream=stream,
        )
        launched = True
    except cuda_bridge.RmsNormContractError as exc:
        reason, reason_code = str(exc), exc.reason_code
    except RuntimeError as exc:
        reason, reason_code = str(exc), "CUDA_LAUNCH_ERROR"
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    untouched = bool(
        torch.equal(
            out.detach().cpu(), torch.full_like(x, OUTPUT_SENTINEL).detach().cpu()
        )
    )
    confirmed = (not launched) and untouched
    return {
        "launched": launched,
        "api_reason": reason,
        "api_reason_code": reason_code,
        "output_untouched": untouched,
        "correctness_status": STATUS_UNSUPPORTED if confirmed else STATUS_UNSUPPORTED_NOT_REJECTED,
        "failed_checks": [] if confirmed else ["rejected_before_launch"],
    }


# ══════════════════════════════════════════════════════════════════════
# step 3 — pilot, then freeze the protocol
# ══════════════════════════════════════════════════════════════════════


def pilot_estimate_ms(
    *, raw: RawLauncher, x, w, out, case: Mapping[str, Any], epsilon: float,
    stream, stream_ptr: int,
) -> Optional[float]:
    launch = raw.bind(
        x=x, w=w, out=out, rows=int(case["rows"]), hidden=int(case["hidden"]),
        epsilon=epsilon, dtype=str(case["dtype"]), variant=str(case["variant"]),
        stream_ptr=stream_ptr,
    )
    timing = time_configuration(
        launch=launch,
        bridge_launch=None,
        stream=stream,
        warmup=PILOT_WARMUP,
        n_launches=PILOT_LAUNCHES,
        groups=PILOT_GROUPS,
    )
    if timing["errors"] or not timing["device_ms_groups"]:
        return None
    return float(np.median(timing["device_ms_groups"]))


def run_pilot(output_dir: Path) -> int:
    """Measure one quick estimate per (shape, dtype, variant) and freeze N."""
    caps = detect_capabilities()
    if not caps.cuda_rmsnorm_available:
        print("BLOCKED: CUDA RMSNorm library unavailable", file=sys.stderr)
        return 2
    raw = RawLauncher(caps.cuda_rmsnorm_lib)  # type: ignore[arg-type]
    epsilon = float(e01.read_model_config().get("rms_norm_eps") or rp.DEFAULT_EPSILON)
    cases, meta = rp.build_case_plan()
    stream = torch.cuda.Stream()
    stream_ptr = int(stream.cuda_stream)

    estimates: Dict[str, float] = {}
    for case in cases:
        if case["status"] != "ELIGIBLE":
            continue
        rows, hidden, dtype = int(case["rows"]), int(case["hidden"]), str(case["dtype"])
        seed = rc.derive_seed(case["input_key"])
        generated = rc.generate_case_arrays(
            rp.TIMED_INPUT_MODE, rows, hidden, dtype, seed
        )
        x = _to_cuda(generated["x"], dtype)
        w = _to_cuda(generated["w"], dtype)
        out = torch.empty_like(x)
        torch.cuda.synchronize()
        try:
            if str(case["variant"]) == rp.FRAMEWORK_VARIANT:
                continue
            value = pilot_estimate_ms(
                raw=raw, x=x, w=w, out=out, case=case, epsilon=epsilon,
                stream=stream, stream_ptr=stream_ptr,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            value = None
            print(f"pilot failure {case['case_id']}: {exc!r}", file=sys.stderr)
        if value is not None:
            estimates[case["case_id"]] = value
        del x, w, out
        torch.cuda.empty_cache()

    launches: Dict[str, int] = {}
    for case in cases:
        if case["status"] != "ELIGIBLE":
            continue
        estimate = estimates.get(case["case_id"])
        if estimate is None:
            launches[case["case_id"]] = MIN_LAUNCHES_DEFAULT
            continue
        n = int(math.ceil(TARGET_WINDOW_MS_DEFAULT / max(estimate, 1e-6)))
        launches[case["case_id"]] = int(
            min(max(n, MIN_LAUNCHES_DEFAULT), MAX_LAUNCHES_DEFAULT)
        )

    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "state": "FROZEN",
        "frozen_at_utc": _utc_now(),
        "groups": GROUPS_DEFAULT,
        "warmup": WARMUP_DEFAULT,
        "target_window_ms": TARGET_WINDOW_MS_DEFAULT,
        "min_launches": MIN_LAUNCHES_DEFAULT,
        "max_launches": MAX_LAUNCHES_DEFAULT,
        "bridge_sample_launches": BRIDGE_SAMPLE_LAUNCHES,
        "order_seed_base": ORDER_SEED_BASE,
        "processes": PROCESSES_DEFAULT,
        "case_launches": launches,
        "pilot_estimates_ms": estimates,
        "rules": {
            "launch_count": (
                "n = clamp(ceil(target_window_ms / pilot_device_ms), min, max) for "
                "every case; the same rule is applied to every variant, so a noisy "
                "variant never gets more samples than another"
            ),
            "device_time": (
                "one CUDA event pair around n back-to-back launches on the explicit "
                "stream; elapsed/n; no per-iteration synchronisation (warm "
                "steady-state, device-event latency)"
            ),
            "host_time": (
                "CPU monotonic clock from before submitting n calls to after one "
                "final synchronisation, divided by n; pure-submit time is recorded "
                "separately"
            ),
            "order": (
                "case blocks are shuffled with order_seed_base + process_index; "
                "within a block the variant order is rotated by block index so no "
                "version is systematically last (= hottest)"
            ),
        },
        "case_plan_meta": meta,
    }
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(
        output_dir / "pilot.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pilot_groups": PILOT_GROUPS,
            "pilot_warmup": PILOT_WARMUP,
            "pilot_launches": PILOT_LAUNCHES,
            "estimates_ms": estimates,
            "note": "pilot numbers are diagnostics only and never enter a conclusion",
            "estimated_total_launches": int(
                sum(launches.values()) * (GROUPS_DEFAULT * 2 + WARMUP_DEFAULT)
            ),
        },
    )
    print(
        f"pilot: {len(estimates)} estimates, "
        f"launch counts in [{min(launches.values(), default=0)}, "
        f"{max(launches.values(), default=0)}]"
    )
    return 0


# ══════════════════════════════════════════════════════════════════════
# steps 4-7 — one independent process
# ══════════════════════════════════════════════════════════════════════


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ordered_blocks(
    cases: Sequence[Mapping[str, Any]], *, process_index: int
) -> List[Tuple[Tuple[str, str], List[Dict[str, Any]]]]:
    blocks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for case in cases:
        blocks.setdefault((str(case["shape_id"]), str(case["dtype"])), []).append(dict(case))
    keys = sorted(blocks)
    rng = np.random.default_rng(ORDER_SEED_BASE + process_index)
    order = rng.permutation(len(keys))
    return [(keys[i], blocks[keys[i]]) for i in order]


def run_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _read_json(output_dir / "protocol.json")
    if not protocol or protocol.get("state") != "FROZEN":
        print(
            "REFUSING to collect: protocol.json is missing or not FROZEN. "
            "Run the `pilot` subcommand first.",
            file=sys.stderr,
        )
        return 2

    process_index = int(args.process_index)
    processes = int(args.processes)
    if not 0 <= process_index < processes:
        print(f"invalid --process-index {process_index} for --processes {processes}",
              file=sys.stderr)
        return 2

    caps = detect_capabilities()
    if not caps.cuda_rmsnorm_available:
        print("BLOCKED: CUDA RMSNorm library unavailable", file=sys.stderr)
        return 2

    environment = e01.collect_environment()
    model_config = e01.read_model_config()
    epsilon = float(model_config.get("rms_norm_eps") or rp.DEFAULT_EPSILON)
    source_identity = e01.freeze_operator_source()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    stream = torch.cuda.Stream()
    stream_ptr = int(stream.cuda_stream)
    raw = RawLauncher(caps.cuda_rmsnorm_lib)  # type: ignore[arg-type]

    sm_count = None
    if torch.cuda.is_available():
        sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)

    cases, plan_meta = rp.build_case_plan()

    monitor = TegrastatsMonitor(interval_ms=100)
    telemetry_available = True
    telemetry_note = None
    try:
        monitor.start()
    except Exception as exc:  # pragma: no cover - environment dependent
        telemetry_available = False
        telemetry_note = repr(exc)
        print(f"WARNING: tegrastats unavailable: {exc!r}", file=sys.stderr)

    records: List[Dict[str, Any]] = []
    process_started = _utc_now()
    t_process0 = time.monotonic_ns()
    t_process1 = t_process0
    ceiling: Dict[str, Any] = {}
    resources: Dict[str, Any] = {}
    case_path = output_dir / f"cases_proc{process_index}.jsonl"
    existing = _read_jsonl(case_path)
    existing_ids = {str(rec["case_id"]) for rec in existing}
    try:
        ceiling = measure_device_ceiling(stream, mib=args.ceiling_mib)
        resources = {
            "occupancy": occupancy_matrix(),
            "static": static_resource_usage(caps.cuda_rmsnorm_lib),
        }
        if process_index == 0:
            _write_json(output_dir / "device_ceiling.json", ceiling)
            _write_json(output_dir / "resources.json", resources)
            _write_json(
                output_dir / "provenance.json",
                {
                    "experiment_id": EXPERIMENT_ID,
                    "stage": STAGE,
                    "environment": environment,
                    "model_config": model_config,
                    "epsilon_used": epsilon,
                    "operator_source": source_identity,
                    "s02_shapes": import_s02_shapes(Path(args.s02_census)),
                    "case_plan_meta": plan_meta,
                    "heatmap_views_required": [name for name, _ in HEATMAP_VIEWS],
                    "machine_state": freeze_machine_state(),
                    "protocol_sha256": rp.canonical_sha256(protocol),
                },
            )
            _write_json(
                output_dir / "case_plan.json",
                {"cases": cases, "meta": plan_meta,
                 "expected_case_ids": rp.expected_case_ids(cases)},
            )

        per_case_launches = protocol["case_launches"]
        blocks = _ordered_blocks(cases, process_index=process_index)
        global_index = 0
        for block_index, ((_shape_key, dtype), block_cases) in enumerate(blocks):
            rows = int(block_cases[0]["rows"])
            hidden = int(block_cases[0]["hidden"])
            seed = rc.derive_seed(block_cases[0]["input_key"])
            generated = rc.generate_case_arrays(
                rp.TIMED_INPUT_MODE, rows, hidden, dtype, seed
            )
            x_np, w_np = generated["x"], generated["w"]
            input_hash = rc.sha256_bytes(x_np.tobytes())
            weight_hash = rc.sha256_bytes(w_np.tobytes())
            reference_fp64 = rc.fp64_oracle(x_np, w_np, rows, hidden, epsilon)
            x = _to_cuda(x_np, dtype)
            w = _to_cuda(w_np, dtype)
            out = torch.empty_like(x)
            torch.cuda.synchronize()

            # rotate the variant order within the block (no version is always last)
            rotation = block_index % max(len(block_cases), 1)
            ordered_cases = block_cases[rotation:] + block_cases[:rotation]
            for case in ordered_cases:
                if str(case["case_id"]) in existing_ids:
                    continue
                case_id = str(case["case_id"])
                is_framework = str(case["variant"]) == rp.FRAMEWORK_VARIANT
                record: Dict[str, Any] = {
                    "experiment_id": EXPERIMENT_ID,
                    "process_index": process_index,
                    "processes": processes,
                    "process_started_utc": process_started,
                    "pid": os.getpid(),
                    "execution_order_index": global_index,
                    "block_index": block_index,
                    "variant_rotation_offset": rotation,
                    "case_id": case_id,
                    "input_key": case["input_key"],
                    "shape_id": case["shape_id"],
                    "rows": rows,
                    "hidden": hidden,
                    "dtype": dtype,
                    "mode": rp.TIMED_INPUT_MODE,
                    "variant": case["variant"],
                    "group": case["group"],
                    "planned_status": case["status"],
                    "shape_sources": case["shape_sources"],
                    "call_weight": case["call_weight"],
                    "epsilon": epsilon,
                    "seed": seed,
                    "input_hash": input_hash,
                    "weight_hash": weight_hash,
                    "block_size": rp.DISPATCH_BLOCK_SIZE,
                    "grid_blocks": rows,
                    "requested_variant_code": rp.VARIANT_CODE.get(str(case["variant"])),
                    "stream_kind": "explicit_non_default",
                    "stream_ptr_nonzero": bool(stream_ptr),
                    "byte_convention_id": rp.BYTE_CONVENTION_ID,
                    "flop_convention_id": rp.FLOP_CONVENTION_ID,
                    "logical_bytes": rp.logical_bytes(rows, hidden, dtype),
                    "declared_flops": rp.declared_flops(rows, hidden),
                    "threads_per_block": rp.DISPATCH_BLOCK_SIZE,
                    "warps_per_block": rp.DISPATCH_BLOCK_SIZE // 32,
                    "shared_bytes_per_block": (
                        rp.DISPATCH_BLOCK_SIZE * 4
                        if str(case["variant"]) == "v0_shared"
                        else (rp.DISPATCH_BLOCK_SIZE // 32) * 4
                    ),
                }
                slot = f"{dtype}|{case['variant']}|{rp.DISPATCH_BLOCK_SIZE}"
                record["occupancy_max_blocks_per_sm"] = (
                    ((resources.get("occupancy") or {}).get("entries") or {})
                    .get(slot, {})
                    .get("occupancy")
                )
                window_begin = time.monotonic_ns()
                clock_begin = _read_gpu_clock_hz()
                should_time = case["status"] == "ELIGIBLE"

                if case["status"] != "ELIGIBLE":
                    probe = probe_unsupported(
                        case=case, x=x, w=w, epsilon=epsilon, stream=stream
                    )
                    record.update(probe)
                    record["status"] = probe["correctness_status"]
                    record["timed"] = False
                elif is_framework:
                    # The framework baseline is the *reference* for the real
                    # shapes, not a candidate: it is timed without an operator
                    # correctness gate (it cannot be launched through the C ABI).
                    record["correctness_status"] = "REFERENCE_NOT_GATED"
                    record["launched"] = False
                    record["output_hash"] = None
                else:
                    probe = correctness_probe(
                        case=case, x=x, w=w, out=out, x_np=x_np,
                        reference_fp64=reference_fp64, epsilon=epsilon,
                        stream=stream, input_hash=input_hash, weight_hash=weight_hash,
                    )
                    record.update(probe)
                    if probe["correctness_status"] != "PASS":
                        record["status"] = STATUS_FAIL_CORRECTNESS
                        record["timed"] = False
                        should_time = False

                if should_time:
                    n_launches = int(
                        per_case_launches.get(case_id, MIN_LAUNCHES_DEFAULT)
                    )
                    record["n_launches_per_group"] = n_launches
                    if is_framework:
                        launch = _framework_launch(x, w, out, epsilon, stream)
                        bridge_launch = None
                    else:
                        launch = raw.bind(
                            x=x, w=w, out=out, rows=rows, hidden=hidden,
                            epsilon=epsilon, dtype=dtype,
                            variant=str(case["variant"]), stream_ptr=stream_ptr,
                        )
                        bridge_launch = _bridge_launch(
                            x, w, out, dtype, str(case["variant"]), epsilon, stream
                        )
                    timing = time_configuration(
                        launch=launch,
                        bridge_launch=bridge_launch,
                        stream=stream,
                        warmup=int(protocol["warmup"]),
                        n_launches=n_launches,
                        groups=int(protocol["groups"]),
                    )
                    record.update(timing)
                    # post-timing re-check: idempotence, pollution, async error
                    out.fill_(OUTPUT_SENTINEL)
                    torch.cuda.synchronize()
                    launch()
                    torch.cuda.synchronize()
                    post_np = out.detach().cpu().numpy()
                    record["post_timing_output_hash"] = rc.sha256_bytes(
                        np.ascontiguousarray(post_np).tobytes()
                    )
                    record["output_hash_stable_after_timing"] = bool(
                        record["post_timing_output_hash"] == record.get("output_hash")
                    ) if not is_framework else None
                    record["input_hash_after_timing"] = rc.sha256_bytes(
                        x.detach().cpu().numpy().tobytes()
                    )
                    record["weight_hash_after_timing"] = rc.sha256_bytes(
                        w.detach().cpu().numpy().tobytes()
                    )
                    record["input_unchanged_after_timing"] = bool(
                        record["input_hash_after_timing"] == input_hash
                    )
                    record["weight_unchanged_after_timing"] = bool(
                        record["weight_hash_after_timing"] == weight_hash
                    )
                    device_stats = rp.summarize_samples(timing["device_ms_groups"])
                    host_stats = rp.summarize_samples(timing["host_ms_groups"])
                    submit_stats = rp.summarize_samples(timing["submit_ms_groups"])
                    bridge_stats = rp.summarize_samples(timing["bridge_ms_samples"])
                    device_seconds = (device_stats["median"] or 0.0) / 1e3
                    derived = rp.derived_metrics(
                        rows=rows, hidden=hidden, dtype=dtype,
                        device_seconds=max(device_seconds, 1e-12),
                        host_seconds=(
                            (host_stats["median"] or 0.0) / 1e3
                            if host_stats["median"]
                            else None
                        ),
                        submit_seconds=(
                            (submit_stats["median"] or 0.0) / 1e3
                            if submit_stats["median"]
                            else None
                        ),
                        bridge_seconds=(
                            (bridge_stats["median"] or 0.0) / 1e3
                            if bridge_stats["median"]
                            else None
                        ),
                    )
                    record["derived"] = derived
                    record["device_ms_stats"] = device_stats
                    record["host_ms_stats"] = host_stats
                    record["submit_ms_stats"] = submit_stats
                    record["bridge_ms_stats"] = bridge_stats
                    record["region_candidates"] = rp.classify_region(
                        rows=rows, hidden=hidden, dtype=dtype,
                        effective_gbps=derived["effective_GBps"],
                        ceiling_gbps=ceiling.get("copy_GBps"),
                        sm_count=sm_count,
                        variant=str(case["variant"]),
                    )
                    if is_framework:
                        record["status"] = STATUS_MEASURED
                    elif timing["errors"]:
                        record["status"] = STATUS_FAIL_CORRECTNESS
                        record["timing_errors"] = timing["errors"]
                    else:
                        record["status"] = STATUS_MEASURED
                    record["timed"] = True

                window_end = time.monotonic_ns()
                clock_end = _read_gpu_clock_hz()
                record["window_begin_ns"] = window_begin
                record["window_end_ns"] = window_end
                record["window_ms"] = (window_end - window_begin) / 1e6
                record["gpu_clock_hz_begin"] = clock_begin
                record["gpu_clock_hz_end"] = clock_end
                records.append(record)
                global_index += 1
                if global_index % 20 == 0:
                    print(
                        f"proc{process_index}: {global_index}/{len(cases)} cases, "
                        f"last={case_id}",
                        flush=True,
                    )
            del x, w, out, x_np, w_np, reference_fp64
            torch.cuda.empty_cache()
    finally:
        t_process1 = time.monotonic_ns()
        if telemetry_available:
            monitor.stop()

    parsed = _parse_telemetry(monitor) if telemetry_available else []
    for record in records:
        window = slice_records(
            parsed, int(record["window_begin_ns"]), int(record["window_end_ns"])
        )
        record["telemetry"] = (
            compute_resource_summary(window) if window else {"num_samples": 0}
        )

    telemetry_path = output_dir / f"telemetry_proc{process_index}.jsonl"
    _write_jsonl(telemetry_path, parsed)

    if args.append and existing:
        records = existing + records
    _write_jsonl(case_path, records)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "process_index": process_index,
        "processes": processes,
        "pid": os.getpid(),
        "started_at_utc": process_started,
        "finished_at_utc": _utc_now(),
        "wall_seconds": (t_process1 - t_process0) / 1e9,
        "case_records": len(records),
        "measured": sum(1 for r in records if r.get("status") == STATUS_MEASURED),
        "failed_correctness": sum(
            1 for r in records if r.get("status") == STATUS_FAIL_CORRECTNESS
        ),
        "expected_unsupported": sum(
            1 for r in records if r.get("status") == STATUS_UNSUPPORTED
        ),
        "unsupported_not_rejected": sum(
            1 for r in records if r.get("status") == STATUS_UNSUPPORTED_NOT_REJECTED
        ),
        "telemetry_available": telemetry_available,
        "telemetry_note": telemetry_note,
        "telemetry_samples": len(parsed),
        "gpu_clock_hz_at_start": _read_gpu_clock_hz(),
        "device_ceiling": ceiling,
        "resources": {
            "occupancy_queried": bool((resources.get("occupancy") or {}).get("entries")),
            "cuobjdump_available": bool(
                (resources.get("static") or {}).get("cuobjdump_available")
            ),
        },
        "environment": {
            "python": environment.get("python_version"),
            "torch": environment.get("torch_version"),
            "device": environment.get("device_name"),
        },
        "protocol_sha256": rp.canonical_sha256(protocol),
        "library_sha256": source_identity.get("cuda_rmsnorm_lib_sha256"),
    }
    _write_json(output_dir / f"collect_proc{process_index}.json", summary)
    print(
        f"proc{process_index}: {summary['measured']} measured, "
        f"{summary['failed_correctness']} correctness failures, "
        f"{summary['expected_unsupported']} unsupported confirmed, "
        f"wall {summary['wall_seconds']:.1f}s"
    )
    return 0


def _framework_launch(x, w, out, epsilon: float, stream):
    """Framework (plain torch) baseline, kept out of the V0/V1/V2 heatmaps."""

    def launch() -> int:
        with torch.cuda.stream(stream):
            xf = x.float()
            wf = w.float()
            mean = (xf * xf).mean(dim=-1, keepdim=True)
            inv = torch.rsqrt(mean + float(epsilon))
            result = (xf * inv * wf).to(out.dtype)
            out.copy_(result)
        return 0

    return launch


def _bridge_launch(x, w, out, dtype: str, variant: str, epsilon: float, stream):
    def launch():
        return cuda_bridge.rmsnorm_forward(
            x, w, dtype=dtype, variant=variant, epsilon=float(epsilon),
            out=out, stream=stream,
        )

    return launch


def _parse_telemetry(monitor: TegrastatsMonitor) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in list(monitor.records):
        parsed = parse_tegrastats_line(rec.get("raw", "") or "")
        if not parsed:
            continue
        parsed["time_ns"] = rec.get("time_ns")
        rows.append(parsed)
    return rows


# ══════════════════════════════════════════════════════════════════════
# aggregation helpers shared by verify / summarize
# ══════════════════════════════════════════════════════════════════════


def load_process_records(output_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for path in sorted(output_dir.glob("cases_proc*.jsonl")):
        index = int(path.stem.replace("cases_proc", ""))
        out[index] = _read_jsonl(path)
    return out


def load_plan(output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    doc = _read_json(output_dir / "case_plan.json") or {}
    return doc.get("cases", []), doc.get("meta", {})


def build_case_summaries(
    process_records: Mapping[int, Sequence[Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    """Aggregate per-process raw records into one summary row per case."""
    by_case: Dict[str, List[Mapping[str, Any]]] = {}
    for index in sorted(process_records):
        for rec in process_records[index]:
            by_case.setdefault(str(rec["case_id"]), []).append(rec)

    summaries: List[Dict[str, Any]] = []
    for case_id, recs in sorted(by_case.items()):
        base = recs[0]
        statuses = sorted({str(r.get("status")) for r in recs})
        device: List[float] = []
        host: List[float] = []
        submit: List[float] = []
        bridge: List[float] = []
        per_process_device: Dict[str, float] = {}
        per_process_host: Dict[str, float] = {}
        for rec in recs:
            index = str(rec.get("process_index"))
            device.extend(rec.get("device_ms_groups") or [])
            host.extend(rec.get("host_ms_groups") or [])
            submit.extend(rec.get("submit_ms_groups") or [])
            bridge.extend(rec.get("bridge_ms_samples") or [])
            stats = rec.get("device_ms_stats") or {}
            if stats.get("median") is not None:
                per_process_device[index] = stats["median"]
            host_stats = rec.get("host_ms_stats") or {}
            if host_stats.get("median") is not None:
                per_process_host[index] = host_stats["median"]

        device_stats = rp.summarize_samples(device)
        derived: Optional[Dict[str, Any]] = None
        if device_stats["median"]:
            derived = rp.derived_metrics(
                rows=int(base["rows"]),
                hidden=int(base["hidden"]),
                dtype=str(base["dtype"]),
                device_seconds=device_stats["median"] / 1e3,
                host_seconds=(
                    (rp.summarize_samples(host)["median"] or 0.0) / 1e3
                    if host and rp.summarize_samples(host)["median"]
                    else None
                ),
                submit_seconds=(
                    (rp.summarize_samples(submit)["median"] or 0.0) / 1e3
                    if submit and rp.summarize_samples(submit)["median"]
                    else None
                ),
                bridge_seconds=(
                    (rp.summarize_samples(bridge)["median"] or 0.0) / 1e3
                    if bridge and rp.summarize_samples(bridge)["median"]
                    else None
                ),
            )
        gbps_samples = [
            rp.logical_bytes(int(base["rows"]), int(base["hidden"]), str(base["dtype"]))
            / (ms / 1e3)
            / 1e9
            for ms in device
            if ms and ms > 0
        ]
        row: Dict[str, Any] = {
            "case_id": case_id,
            "shape_id": base["shape_id"],
            "rows": int(base["rows"]),
            "hidden": int(base["hidden"]),
            "dtype": base["dtype"],
            "variant": base["variant"],
            "group": base.get("group"),
            "planned_status": base.get("planned_status"),
            "status": statuses[0] if len(statuses) == 1 else "INCONSISTENT_ACROSS_PROCESSES",
            "statuses_by_process": {str(r.get("process_index")): r.get("status") for r in recs},
            "process_count": len(recs),
            "correctness_status": sorted(
                {str(r.get("correctness_status")) for r in recs}
            ),
            "output_hash": sorted({str(r.get("output_hash")) for r in recs}),
            "classification_mismatch_count_max": max(
                (int(r.get("classification_mismatch_count") or 0) for r in recs),
                default=0,
            ),
            "max_allowed_ratio_max": max(
                (float(r.get("max_allowed_ratio") or 0.0) for r in recs), default=0.0
            ),
            "device_ms": device_stats,
            "host_ms": rp.summarize_samples(host),
            "submit_ms": rp.summarize_samples(submit),
            "bridge_ms": rp.summarize_samples(bridge),
            "per_process_device_ms_median": per_process_device,
            "per_process_host_ms_median": per_process_host,
            "effective_GBps": rp.summarize_samples(gbps_samples),
            "derived": derived,
            "logical_bytes": rp.logical_bytes(
                int(base["rows"]), int(base["hidden"]), str(base["dtype"])
            ),
            "declared_flops": rp.declared_flops(int(base["rows"]), int(base["hidden"])),
            "byte_convention_id": rp.BYTE_CONVENTION_ID,
            "flop_convention_id": rp.FLOP_CONVENTION_ID,
            "block_size": base.get("block_size"),
            "n_launches_per_group": base.get("n_launches_per_group"),
            "occupancy_max_blocks_per_sm": base.get("occupancy_max_blocks_per_sm"),
            "shared_bytes_per_block": base.get("shared_bytes_per_block"),
            "warps_per_block": base.get("warps_per_block"),
            "gpu_clock_hz": sorted(
                {
                    r.get("gpu_clock_hz_begin")
                    for r in recs
                    if r.get("gpu_clock_hz_begin") is not None
                }
                | {
                    r.get("gpu_clock_hz_end")
                    for r in recs
                    if r.get("gpu_clock_hz_end") is not None
                }
            ),
            "gpu_temp_c": sorted(
                {
                    round(float((r.get("telemetry") or {}).get("avg_gpu_temp_c")), 3)
                    for r in recs
                    if (r.get("telemetry") or {}).get("avg_gpu_temp_c") is not None
                }
            ),
            "power_w": sorted(
                {
                    round(float((r.get("telemetry") or {}).get("avg_power_w")), 3)
                    for r in recs
                    if (r.get("telemetry") or {}).get("avg_power_w") is not None
                }
            ),
            "call_weight": base.get("call_weight"),
            "shape_sources": base.get("shape_sources"),
            "region_candidates": base.get("region_candidates"),
            "ncu_relevant": bool(base.get("group") == "main"),
        }
        if device:
            row["device_ms"]["ci"] = rp.bootstrap_ci(device)
        if host:
            row["host_ms"]["ci"] = rp.bootstrap_ci(host)
        row["effective_GBps"]["ci"] = rp.bootstrap_ci(gbps_samples)
        summaries.append(row)
    return summaries


def attach_paired_speedups(
    summaries: List[Dict[str, Any]],
    process_records: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Paired ``v0/candidate`` latency ratios per (shape, dtype, candidate).

    Pairing is by *group index inside the same process block*, so a ratio
    inherits the local clock/thermal state; the pooled vector is then taken over
    all ``processes x groups`` samples (protocol §7).
    """
    pairs: Dict[str, List[float]] = {}
    for index in sorted(process_records):
        by_block: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = {}
        for rec in process_records[index]:
            if rec.get("group") != "main":
                continue
            if rec.get("status") != STATUS_MEASURED:
                continue
            by_block.setdefault((str(rec["shape_id"]), str(rec["dtype"])), {})[
                str(rec["variant"])
            ] = rec
        for (sid, dtype), variants in by_block.items():
            baseline = variants.get("v0_shared")
            if not baseline:
                continue
            base_groups = baseline.get("device_ms_groups") or []
            for variant, rec in variants.items():
                if variant == "v0_shared":
                    continue
                cand_groups = rec.get("device_ms_groups") or []
                bucket = pairs.setdefault(f"{sid}|{dtype}|{variant}", [])
                for a, b in zip(base_groups, cand_groups):
                    if a and b and a > 0 and b > 0:
                        bucket.append(float(a) / float(b))

    for summary in summaries:
        if summary.get("group") != "main" or summary["variant"] == "v0_shared":
            continue
        key = f"{summary['shape_id']}|{summary['dtype']}|{summary['variant']}"
        ratios = pairs.get(key)
        if not ratios:
            continue
        stats = rp.summarize_samples(ratios)
        stats["ci"] = rp.bootstrap_ci(ratios)
        stats["paired_count"] = len(ratios)
        stats["paired_samples"] = list(ratios)
        stats["verdict"] = rp.verdict_from_ci(stats["ci"])
        summary["speedup_vs_v0"] = stats
    return {"pair_count": len(pairs)}


def winner_rows(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Winner / tie decision per (shape, dtype), from paired speedup vectors."""
    base_info: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    samples: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for summary in summaries:
        if summary.get("group") != "main":
            continue
        key = (str(summary["shape_id"]), str(summary["dtype"]))
        base_info.setdefault(key, summary)
        if summary.get("variant") == "v0_shared":
            continue
        vector = (summary.get("speedup_vs_v0") or {}).get("paired_samples")
        if vector:
            samples.setdefault(key, {})[str(summary["variant"])] = list(vector)

    rows: List[Dict[str, Any]] = []
    for key, variants in samples.items():
        info = base_info[key]
        rows.append(
            rp.winner_row(
                shape_id_value=key[0],
                rows=int(info["rows"]),
                hidden=int(info["hidden"]),
                dtype=key[1],
                paired_by_variant=variants,
            )
        )
    return sorted(rows, key=lambda r: (r["hidden"], r["rows"], r["dtype"]))


def regression_rows(
    summaries: Sequence[Mapping[str, Any]], winners: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Every case slower than the baseline (protocol §12 step 12).

    The table is built from the *measured* median speedup, not from the winner
    decision, so a case that fails the guard band but is still slower than v0
    cannot disappear.
    """
    winner_lookup = {
        (row["shape_id"], row["dtype"]): row for row in winners
    }
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        if summary.get("group") != "main" or summary.get("variant") == "v0_shared":
            continue
        speedup = summary.get("speedup_vs_v0") or {}
        median = speedup.get("median")
        if median is None or median >= 1.0:
            continue
        winner = winner_lookup.get((summary["shape_id"], summary["dtype"]), {})
        rows.append(
            {
                "shape_id": summary["shape_id"],
                "rows": summary["rows"],
                "hidden": summary["hidden"],
                "dtype": summary["dtype"],
                "variant": summary["variant"],
                "block_size": summary.get("block_size"),
                "median_speedup_vs_v0": median,
                "slow_ratio": 1.0 / median if median else None,
                "ci": speedup.get("ci"),
                "paired_count": speedup.get("paired_count"),
                "verdict": speedup.get("verdict"),
                "correctness_status": summary.get("correctness_status"),
                "status": summary.get("status"),
                "mechanism_hypothesis": (summary.get("region_candidates") or {}).get(
                    "labels"
                ),
                "mechanism_reasons": (summary.get("region_candidates") or {}).get(
                    "reasons"
                ),
                "fallback_decision": (
                    "routing falls back to v0_shared for this cell unless a "
                    "stable neighbour contradicts it"
                ),
                "e03_04_tracking": True,
                "e03_04_role": (
                    "regression/no-gain representative"
                    if (winner.get("decision") == "v0_shared")
                    else "slower than baseline but not selected"
                ),
            }
        )
    return sorted(rows, key=lambda r: r["median_speedup_vs_v0"])


def ncu_candidates(
    winners: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 10: representative fast/slow cases for E03-04, chosen by rule.

    The rules are applied in a fixed order so the selection cannot be tuned to
    the prettiest point:

    1. biggest verified acceleration (``CANDIDATE_WINS``, real S02 shape first);
    2. worst regression / no-gain cell;
    3. ``rows=1`` at the real hidden width (H=2048);
    4. prefill-width rows at the real hidden width (H=2048, rows>=512);
    5. an odd/tail shape that is still correct;
    6. a cell whose neighbour contradicts it (dispatcher boundary candidate).
    """
    candidates: Dict[str, Any] = {}
    measured = [
        s for s in summaries
        if s.get("group") == "main" and s.get("status") == STATUS_MEASURED
        and (s.get("speedup_vs_v0") or {}).get("median") is not None
    ]

    def pack(summary: Mapping[str, Any], role: str) -> Dict[str, Any]:
        speedup = summary.get("speedup_vs_v0") or {}
        return {
            "role": role,
            "case_id": summary["case_id"],
            "shape_id": summary["shape_id"],
            "rows": summary["rows"],
            "hidden": summary["hidden"],
            "dtype": summary["dtype"],
            "variant": summary["variant"],
            "median_speedup_vs_v0": speedup.get("median"),
            "speedup_ci": speedup.get("ci"),
            "verdict": speedup.get("verdict"),
            "device_ms_median": (summary.get("device_ms") or {}).get("median"),
            "effective_GBps_median": (summary.get("effective_GBps") or {}).get("median"),
            "region_candidates": summary.get("region_candidates"),
            "counters_to_request": [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__bytes.sum (if the Tegra build exposes it, else record as unavailable)",
                "lts__t_bytes.sum",
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
                "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            ],
        }

    fast = [
        s for s in measured
        if (s.get("speedup_vs_v0") or {}).get("verdict") == "CANDIDATE_WINS"
    ]
    fast.sort(
        key=lambda s: (
            0 if "S02_RUNTIME" in (s.get("shape_sources") or []) else 1,
            -float((s.get("speedup_vs_v0") or {}).get("median") or 0.0),
        )
    )
    if fast:
        candidates["acceleration"] = pack(fast[0], "verified acceleration")

    slow = sorted(measured, key=lambda s: float((s.get("speedup_vs_v0") or {}).get("median") or 9e9))
    if slow:
        candidates["regression_or_no_gain"] = pack(
            slow[0], "slowest measured cell (regression or no gain)"
        )

    def pick(predicate) -> Optional[Mapping[str, Any]]:
        hits = [s for s in measured if predicate(s)]
        if not hits:
            return None
        hits.sort(key=lambda s: float((s.get("speedup_vs_v0") or {}).get("median") or 0.0))
        return hits[0]

    decode = pick(lambda s: s["rows"] == 1 and s["hidden"] == rp.S02_HIDDEN)
    if decode:
        candidates["decode_rows1_real_hidden"] = pack(decode, "rows=1 at the real H")
    prefill = pick(
        lambda s: s["hidden"] == rp.S02_HIDDEN and s["rows"] >= 512
    )
    if prefill:
        candidates["prefill_many_rows_real_hidden"] = pack(
            prefill, "prefill-width rows at the real H"
        )
    tail = pick(
        lambda s: s["hidden"] in rp.BOUNDARY_H and s["hidden"] % 4 != 0
    )
    if tail:
        candidates["odd_tail_correct"] = pack(tail, "odd/tail shape that is still correct")

    flip = _dispatcher_boundary_candidate(winners)
    if flip:
        candidates["dispatcher_boundary"] = flip
    return {
        "selection_rules": (
            "fixed order: acceleration -> slowest cell -> rows=1 real H -> prefill "
            "real H -> odd/tail correct -> neighbour contradiction; thresholds were "
            "not adjusted after seeing the numbers"
        ),
        "cases": candidates,
    }


def _dispatcher_boundary_candidate(
    winners: Sequence[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """A cell whose decision differs from its nearest smaller-``rows``/``H`` neighbour."""
    by_key = {(w["shape_id"], w["dtype"]): w for w in winners}
    for winner in sorted(winners, key=lambda w: (w["hidden"], w["rows"])):
        for other_rows, other_h in ((winner["rows"] // 2, winner["hidden"]),
                                    (winner["rows"], winner["hidden"] - 1),
                                    (winner["rows"], winner["hidden"] + 1)):
            other = by_key.get((rp.shape_id(other_rows, other_h), winner["dtype"]))
            if other and other["decision"] != winner["decision"]:
                return {
                    "role": "dispatcher boundary (neighbour contradiction)",
                    "shape_id": winner["shape_id"],
                    "dtype": winner["dtype"],
                    "decision": winner["decision"],
                    "neighbour_shape_id": other["shape_id"],
                    "neighbour_decision": other["decision"],
                    "note": (
                        "an isolated cell that beats its neighbour the other way is a "
                        "HYPOTHESIS for E03-08, not a routing rule"
                    ),
                }
    return None


def routing_candidates(
    winners: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 11: candidate routing domains with guard band and forbidden zones."""
    domains = [
        ("decode_rows1", lambda w: w["rows"] == 1),
        ("small_rows_2_16", lambda w: 2 <= w["rows"] <= 16),
        ("mid_rows_17_512", lambda w: 17 <= w["rows"] <= 512),
        ("large_rows_513_plus", lambda w: w["rows"] > 512),
    ]
    h_buckets = [
        ("head_dim_h_le_128", lambda w: w["hidden"] <= 128),
        ("hidden_129_2048", lambda w: 128 < w["hidden"] <= 2048),
        ("wide_h_gt_2048", lambda w: w["hidden"] > 2048),
    ]
    out: List[Dict[str, Any]] = []
    for dtype in rp.DTYPES:
        for row_name, row_pred in domains:
            for h_name, h_pred in h_buckets:
                cells = [
                    w for w in winners
                    if w["dtype"] == dtype and row_pred(w) and h_pred(w)
                ]
                if not cells:
                    continue
                tally: Dict[str, Dict[str, Any]] = {}
                for cell in cells:
                    for cand in cell["candidates"]:
                        entry = tally.setdefault(
                            cand["variant"],
                            {"wins": 0, "regressions": 0, "medians": []},
                        )
                        if cand["verdict"] == "CANDIDATE_WINS":
                            entry["wins"] += 1
                        if cand["verdict"] == "REGRESSION":
                            entry["regressions"] += 1
                        entry["medians"].append(cand["median_speedup"])
                viable = {
                    variant: entry
                    for variant, entry in tally.items()
                    if entry["wins"] > 0 and entry["regressions"] == 0
                }
                if viable:
                    best = max(
                        viable.items(),
                        key=lambda item: float(np.median(item[1]["medians"])),
                    )
                    proposed, kind = best[0], "CANDIDATE_RULE"
                else:
                    proposed, kind = "v0_shared", "FALLBACK_TO_BASELINE"
                out.append(
                    {
                        "dtype": dtype,
                        "domain": {
                            "rows": row_name,
                            "hidden": h_name,
                            "shape_count": len(cells),
                        },
                        "tally": {
                            variant: {
                                "wins": entry["wins"],
                                "regressions": entry["regressions"],
                                "median_speedup": float(np.median(entry["medians"])),
                            }
                            for variant, entry in sorted(tally.items())
                        },
                        "proposed": proposed,
                        "proposal_kind": kind,
                        "guard_band": GUARD_BAND["micro_min_relative_improvement"],
                        "rules": (
                            "a candidate may be routed only if its paired speedup CI "
                            "lower bound exceeds 1+guard band and no cell in the domain "
                            "regresses; isolated single-cell wins are HYPOTHESIS"
                        ),
                    }
                )
    forbidden = []
    for dtype, variants in rp.UNSUPPORTED_VARIANT_PAIRS.items():
        for variant in variants:
            forbidden.append(f"{dtype} x {variant}: rejected by the C ABI (E03-01)")
    forbidden.append(
        "any non-contiguous / misaligned / non-canonical layout: E03-03 owns those "
        "domains and this experiment never measured them"
    )
    forbidden.append(
        "boundary dtype codes and variant codes outside the frozen set: rejected "
        "before launch (E03-01)"
    )
    return {
        "domains": out,
        "forbidden_domains": forbidden,
        "guard_band": GUARD_BAND,
        "handoff": "E03-08 must replay every declared path with forced + auto cases",
    }


def call_weighted_view(
    summaries: Sequence[Mapping[str, Any]], winners: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Shape-weighted speedup on the S02 call matrix (never an arithmetic mean).

    The real model dtype is fp16, where V0/V1 are unsupported, so the reference
    baseline is the **framework expression** (plain torch ops, i.e. what the
    model actually runs) and the candidate is the hand-written ``v2_vectorized``
    kernel. The comparison is therefore "CUDA kernel vs framework reference",
    shape-weighted by the S02 call counts. It is an input to E03-10, not a
    model-level claim.
    """
    from hqsb.benchmark.hotspot_decision import shape_weighted_speedup

    lookup = {
        (s["shape_id"], s["dtype"], s["variant"]): s for s in summaries
    }
    matrix = rp.s02_call_matrix()
    results: List[Dict[str, Any]] = []
    by_workload: Dict[str, List[Dict[str, Any]]] = {}
    for row in matrix:
        sid = row["shape_id"]
        base = lookup.get((sid, "fp16", rp.FRAMEWORK_VARIANT))
        selected = lookup.get((sid, "fp16", "v2_vectorized"))
        if not base or not selected:
            continue
        t_old = (base.get("device_ms") or {}).get("median")
        t_new = (selected.get("device_ms") or {}).get("median")
        if not t_old or not t_new:
            continue
        entry = {
            "shape_id": sid,
            "rows": row["rows"],
            "hidden": row["hidden"],
            "phase": row["phase"],
            "module_role": row["module_role"],
            "calls_per_request": row["calls_per_request"],
            "baseline": rp.FRAMEWORK_VARIANT,
            "candidate": "v2_vectorized",
            "t_old_ms": t_old,
            "t_new_ms": t_new,
            "shape_speedup": t_old / t_new,
        }
        results.append(entry)
        by_workload.setdefault(row["workload"], []).append(entry)

    def compute(entries: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not entries:
            return None
        return shape_weighted_speedup(
            [e["calls_per_request"] for e in entries],
            [e["t_old_ms"] for e in entries],
            [e["t_new_ms"] for e in entries],
        )

    pooled: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in results:
        key = (entry["shape_id"], entry["candidate"])
        slot = pooled.setdefault(
            key,
            {
                "shape_id": entry["shape_id"],
                "candidate": entry["candidate"],
                "calls_per_request": 0,
                "t_old_ms": entry["t_old_ms"],
                "t_new_ms": entry["t_new_ms"],
            },
        )
        slot["calls_per_request"] += entry["calls_per_request"]

    return {
        "dtype": "fp16",
        "baseline": rp.FRAMEWORK_VARIANT,
        "candidate": "v2_vectorized",
        "by_workload": {
            name: compute(entries) for name, entries in sorted(by_workload.items())
        },
        "pooled_across_workloads": compute(list(pooled.values())),
        "rows": results,
        "note": (
            "shape-weighted micro speedup of the CUDA kernel against the framework "
            "reference, on the S02 call matrix; E03-10 owns the phase-share / "
            "Amdahl step and no model-level claim is made here"
        ),
    }


def build_heatmap_artifacts(
    output_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Write every required heatmap view as SVG + numeric CSV."""
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    def emit(*, view: str, dtype: str, variant: str, kind: str, svg: str, matrix=None):
        name = f"{view}__{dtype}__{variant}".replace("/", "_")
        svg_path = heatmap_dir / f"{name}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        entry = {
            "view": view,
            "dtype": dtype,
            "variant": variant,
            "kind": kind,
            "svg": str(svg_path.relative_to(output_dir)),
            "svg_bytes": svg_path.stat().st_size,
        }
        if matrix is not None:
            csv_path = heatmap_dir / f"{name}.csv"
            rp.write_matrix_csv(csv_path, matrix)
            json_path = heatmap_dir / f"{name}.json"
            _write_json(json_path, matrix)
            entry["csv"] = str(csv_path.relative_to(output_dir))
            entry["json"] = str(json_path.relative_to(output_dir))
            entry["csv_bytes"] = csv_path.stat().st_size
        manifest.append(entry)

    for dtype in rp.DTYPES:
        variants = list(rp.VARIANT_SUPPORT[dtype])
        dtype_cells = [
            (int(s["hidden"]), int(s["rows"]))
            for s in summaries
            if s.get("dtype") == dtype
        ]
        axis_rows = sorted({h for h, _ in dtype_cells})
        axis_cols = sorted({r for _, r in dtype_cells})
        for variant in variants:
            for view, value_key, unit, scale, center in (
                ("device_latency", "device_ms", "ms", "sequential", None),
                ("host_submit_completion", "host_ms", "ms", "sequential", None),
                ("effective_gbps", "effective_GBps", "GB/s", "sequential", None),
                ("speedup_vs_v0", "speedup_vs_v0", "x", "diverging", 1.0),
            ):
                if view == "speedup_vs_v0" and variant == "v0_shared":
                    continue
                if view == "speedup_vs_v0" and dtype == "fp16":
                    # v0_shared is unsupported for fp16, so "speedup vs v0" is
                    # undefined; the fp16 comparison is kernel vs framework
                    # reference (call_weighted_view), never this axis.
                    continue
                matrix = rp.build_matrix(
                    summaries, dtype=dtype, variant=variant, value_key=value_key,
                    row_labels=axis_rows, col_labels=axis_cols,
                )
                if not matrix["row_labels"]:
                    continue
                svg = rp.render_heatmap_svg(
                    title=f"E03-02 {view} — {dtype} {variant}",
                    matrix=matrix,
                    unit=unit,
                    scale=scale,
                    center=center,
                    legend_note=(
                        "colour range from robust percentiles; exact values in cells "
                        "and in the accompanying CSV"
                    ),
                )
                emit(view=view, dtype=dtype, variant=variant,
                     kind="numeric", svg=svg, matrix=matrix)

        status_matrix = _status_matrix(summaries, dtype)
        emit(
            view="status_mask",
            dtype=dtype,
            variant="all",
            kind="categorical",
            svg=rp.render_categorical_svg(
                title=f"E03-02 status mask — {dtype}",
                row_labels=status_matrix["row_labels"],
                col_labels=status_matrix["col_labels"],
                labels=status_matrix["labels"],
                legend_note="MEASURED / EXPECTED_UNSUPPORTED / FAIL / not-run are distinct cells",
            ),
        )
        emit(
            view="s02_source_coverage",
            dtype=dtype,
            variant="all",
            kind="categorical",
            svg=rp.render_categorical_svg(
                title=f"E03-02 S02 source coverage — {dtype}",
                row_labels=status_matrix["row_labels"],
                col_labels=status_matrix["col_labels"],
                labels=status_matrix["source_labels"],
                palette={
                    "S02_RUNTIME": "#ffb74d",
                    "S02_RUNTIME+DESIGN": "#ff8a65",
                    "DESIGN+SYNTHETIC": "#90caf9",
                    "DESIGN": "#e0e0e0",
                    "MISSING": "#f5f5f5",
                },
                legend_note="orange cells are real Qwen3-1.7B RMSNorm shapes (E02-02 census)",
            ),
        )
        winner_matrix = _winner_matrix(winners, dtype)
        labels_present = sorted(
            {cell for row in winner_matrix["labels"] for cell in row if cell is not None}
        )
        colours = ["#bbdefb", "#c8e6c9", "#ffe0b2", "#d1c4e9", "#f8bbd0", "#b2dfdb"]
        palette = {
            label: (
                "#eeeeee" if label in ("MISSING", "NOT_RUN") else colours[i % len(colours)]
            )
            for i, label in enumerate(labels_present)
        }
        emit(
            view="winner_and_tie",
            dtype=dtype,
            variant="all",
            kind="categorical",
            svg=rp.render_categorical_svg(
                title=f"E03-02 winner / tie — {dtype}",
                row_labels=winner_matrix["row_labels"],
                col_labels=winner_matrix["col_labels"],
                labels=winner_matrix["labels"],
                palette=palette,
                legend_note="a label lists every variant inside the guard band, not only the argmax",
            ),
        )
    return manifest


def _status_matrix(summaries: Sequence[Mapping[str, Any]], dtype: str) -> Dict[str, Any]:
    priority = {
        STATUS_MEASURED: 0,
        "REFERENCE_NOT_GATED": 0,
        STATUS_UNSUPPORTED: 1,
        STATUS_FAIL_CORRECTNESS: 2,
        STATUS_UNSUPPORTED_NOT_REJECTED: 3,
    }
    cells: Dict[Tuple[int, int], str] = {}
    sources: Dict[Tuple[int, int], str] = {}
    for summary in summaries:
        if summary["dtype"] != dtype:
            continue
        key = (int(summary["hidden"]), int(summary["rows"]))
        status = str(summary.get("status"))
        if key not in cells or priority.get(status, 9) > priority.get(cells[key], -1):
            cells[key] = status
        tags = set(summary.get("shape_sources") or [])
        is_real = "S02_RUNTIME" in tags
        is_design = bool({"DESIGN_MANDATORY", "SYNTHETIC_SCALING"} & tags)
        if is_real and is_design:
            sources[key] = "S02_RUNTIME+DESIGN"
        elif is_real:
            sources[key] = "S02_RUNTIME"
        elif is_design:
            sources[key] = "DESIGN+SYNTHETIC"
        else:
            sources.setdefault(key, "DESIGN")
    row_labels = sorted({h for h, _ in cells})
    col_labels = sorted({r for _, r in cells})
    labels = [
        [cells.get((h, r)) for r in col_labels] for h in row_labels
    ]
    source_labels = [
        [sources.get((h, r), "MISSING") for r in col_labels] for h in row_labels
    ]
    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "labels": labels,
        "source_labels": source_labels,
    }


def _winner_matrix(winners: Sequence[Mapping[str, Any]], dtype: str) -> Dict[str, Any]:
    cells: Dict[Tuple[int, int], str] = {}
    for winner in winners:
        if winner["dtype"] != dtype:
            continue
        cells[(int(winner["hidden"]), int(winner["rows"]))] = "+".join(
            winner["tie_group"]
        )
    row_labels = sorted({h for h, _ in cells})
    col_labels = sorted({r for _, r in cells})
    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "labels": [[cells.get((h, r)) for r in col_labels] for h in row_labels],
    }


# ══════════════════════════════════════════════════════════════════════
# step 8 — summaries, heatmaps and handoffs
# ══════════════════════════════════════════════════════════════════════


def run_summarize(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    process_records = load_process_records(output_dir)
    if not process_records:
        print("no cases_proc*.jsonl found; run `collect` first", file=sys.stderr)
        return 2
    plan_cases, plan_meta = load_plan(output_dir)
    protocol = _read_json(output_dir / "protocol.json") or {}
    ceiling = _read_json(output_dir / "device_ceiling.json") or {}
    resources = _read_json(output_dir / "resources.json") or {}

    summaries = build_case_summaries(process_records)
    attach_paired_speedups(summaries, process_records)
    _write_jsonl(output_dir / "case_summary.jsonl", summaries)

    winners = winner_rows(summaries)
    _write_json(
        output_dir / "winner_table.json",
        {
            "guard_band": GUARD_BAND,
            "decision_rule": (
                "CANDIDATE_WINS requires the paired speedup CI lower bound above "
                "1 + guard band; ties list every variant inside the guard band of "
                "the best median; no candidate crossing the band means "
                "BASELINE_RETAINED, not 'v0 is optimal'"
            ),
            "rows": winners,
        },
    )
    regressions = regression_rows(summaries, winners)
    _write_json(
        output_dir / "regression_table.json",
        {
            "rows": regressions,
            "count": len(regressions),
            "rule": "every measured cell with median speedup < 1.0 relative to v0",
        },
    )
    ncu = ncu_candidates(winners, summaries)
    _write_json(output_dir / "ncu_candidates.json", ncu)
    routing = routing_candidates(winners, summaries)
    _write_json(output_dir / "routing_candidates.json", routing)
    call_weighted = call_weighted_view(summaries, winners)
    _write_json(output_dir / "call_weighted_speedup.json", call_weighted)
    heatmaps = build_heatmap_artifacts(output_dir, summaries, winners)

    coverage: Dict[str, Any] = {
        "total_cases": len(plan_cases),
        "eligible_cases": sum(1 for c in plan_cases if c["status"] == "ELIGIBLE"),
        "expected_unsupported_cases": sum(
            1 for c in plan_cases if c["status"] != "ELIGIBLE"
        ),
        "shapes": plan_meta.get("shape_count"),
        "by_dtype_variant": {},
        "by_status": {},
        "per_process": {},
    }
    for summary in summaries:
        key = f"{summary['dtype']}|{summary['variant']}"
        bucket = coverage["by_dtype_variant"].setdefault(
            key, {"cases": 0, "measured": 0, "failed": 0, "unsupported": 0}
        )
        bucket["cases"] += 1
        if summary["status"] == STATUS_MEASURED:
            bucket["measured"] += 1
        elif summary["status"] == STATUS_FAIL_CORRECTNESS:
            bucket["failed"] += 1
        elif summary["status"] in (STATUS_UNSUPPORTED, STATUS_UNSUPPORTED_NOT_REJECTED):
            bucket["unsupported"] += 1
        coverage["by_status"][summary["status"]] = (
            coverage["by_status"].get(summary["status"], 0) + 1
        )
    for index, records in process_records.items():
        collect = _read_json(output_dir / f"collect_proc{index}.json") or {}
        coverage["per_process"][str(index)] = {
            "pid": collect.get("pid"),
            "records": len(records),
            "measured": collect.get("measured"),
            "wall_seconds": collect.get("wall_seconds"),
            "telemetry_available": collect.get("telemetry_available"),
            "gpu_clock_hz_at_start": collect.get("gpu_clock_hz_at_start"),
        }

    slowest = sorted(
        (
            {
                "case_id": s["case_id"],
                "median_speedup_vs_v0": (s.get("speedup_vs_v0") or {}).get("median"),
                "verdict": (s.get("speedup_vs_v0") or {}).get("verdict"),
            }
            for s in summaries
            if s.get("group") == "main"
            and (s.get("speedup_vs_v0") or {}).get("median") is not None
        ),
        key=lambda row: row["median_speedup_vs_v0"],
    )
    fastest = list(reversed(slowest))

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "generated_at_utc": _utc_now(),
        "protocol": {
            "groups": protocol.get("groups"),
            "warmup": protocol.get("warmup"),
            "target_window_ms": protocol.get("target_window_ms"),
            "processes": protocol.get("processes"),
            "order_seed_base": protocol.get("order_seed_base"),
            "sha256": rp.canonical_sha256(protocol) if protocol else None,
        },
        "conventions": {
            "byte_convention_id": rp.BYTE_CONVENTION_ID,
            "flop_convention_id": rp.FLOP_CONVENTION_ID,
            "block_size": rp.DISPATCH_BLOCK_SIZE,
            "input_mode": rp.TIMED_INPUT_MODE,
        },
        "coverage": coverage,
        "device_ceiling": ceiling,
        "resources": {
            "occupancy_entries": len((resources.get("occupancy") or {}).get("entries") or {}),
            "cuobjdump_available": (resources.get("static") or {}).get("cuobjdump_available"),
            "per_kernel": (resources.get("static") or {}).get("per_kernel"),
        },
        "fastest_cells": fastest[:10],
        "slowest_cells": slowest[:10],
        "regression_count": len(regressions),
        "winner_rows": len(winners),
        "ncu_candidates": sorted((ncu.get("cases") or {}).keys()),
        "routing_domains": len(routing.get("domains") or []),
        "heatmaps": heatmaps,
        "s02_call_weighted_speedup": call_weighted.get("pooled_across_workloads"),
        "failures": [
            s["case_id"]
            for s in summaries
            if s["status"] == STATUS_FAIL_CORRECTNESS
            or s["status"] == STATUS_UNSUPPORTED_NOT_REJECTED
        ],
    }
    _write_json(output_dir / "summary.json", summary)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": _utc_now(),
        "files": [],
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            manifest["files"].append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": e01._sha256_file(path),
                }
            )
    manifest["file_count"] = len(manifest["files"])
    _write_json(output_dir / "EVIDENCE_MANIFEST.json", manifest)
    print(
        f"summarize: {len(summaries)} case summaries, {len(winners)} winner rows, "
        f"{len(regressions)} regressions, {len(heatmaps)} heatmaps"
    )
    return 0


# ══════════════════════════════════════════════════════════════════════
# step 11 of §11 — the verdict
# ══════════════════════════════════════════════════════════════════════


def run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    process_records = load_process_records(output_dir)
    plan_cases, plan_meta = load_plan(output_dir)
    provenance = _read_json(output_dir / "provenance.json") or {}
    protocol = _read_json(output_dir / "protocol.json") or {}
    summary = _read_json(output_dir / "summary.json") or {}
    winners = (_read_json(output_dir / "winner_table.json") or {}).get("rows", [])
    regressions = (_read_json(output_dir / "regression_table.json") or {}).get("rows", [])
    ncu = _read_json(output_dir / "ncu_candidates.json") or {}
    routing = _read_json(output_dir / "routing_candidates.json") or {}
    ceiling = _read_json(output_dir / "device_ceiling.json") or {}
    resources = _read_json(output_dir / "resources.json") or {}
    collect_summaries = [
        _read_json(output_dir / f"collect_proc{index}.json") or {}
        for index in sorted(process_records)
    ]

    expected_ids = set(rp.expected_case_ids(plan_cases)) if plan_cases else set()
    eligible = [c for c in plan_cases if c["status"] == "ELIGIBLE"]
    unsupported = [c for c in plan_cases if c["status"] != "ELIGIBLE"]

    checks: Dict[str, Any] = {}

    def check(name: str, passed: bool, detail: Any = None, note: str = "") -> None:
        checks[name] = {"passed": bool(passed), "detail": detail, "note": note}

    all_records = [rec for index in sorted(process_records) for rec in process_records[index]]
    measured = [r for r in all_records if r.get("status") == STATUS_MEASURED]
    failed = [r for r in all_records if r.get("status") == STATUS_FAIL_CORRECTNESS]
    not_rejected = [
        r for r in all_records if r.get("status") == STATUS_UNSUPPORTED_NOT_REJECTED
    ]

    # 1 ── identity is bound
    env = provenance.get("environment") or {}
    source = provenance.get("operator_source") or {}
    check(
        "identity_and_machine_state_bound",
        bool(env.get("git_commit") and source.get("cuda_rmsnorm_lib_sha256")
             and provenance.get("machine_state")),
        {
            "git_commit": env.get("git_commit"),
            "git_dirty": env.get("git_dirty"),
            "library_sha256": source.get("cuda_rmsnorm_lib_sha256"),
            "source_sha256_count": len(source.get("source_sha256") or {}),
            "nvpmodel": (provenance.get("machine_state") or {}).get("nvpmodel", {}).get("stdout"),
            "gpu_clock_hz_at_freeze": (
                provenance.get("machine_state") or {}
            ).get("gpu_clock_hz_observed_at_freeze"),
        },
    )

    # 2 ── the protocol was frozen before the run
    frozen_launches = protocol.get("case_launches") or {}
    missing_launch_counts = [
        c["case_id"]
        for c in eligible
        if c["case_id"] not in frozen_launches
    ]
    check(
        "protocol_frozen_before_execution",
        protocol.get("state") == "FROZEN" and not missing_launch_counts,
        {
            "state": protocol.get("state"),
            "frozen_at_utc": protocol.get("frozen_at_utc"),
            "cases_without_frozen_launch_count": missing_launch_counts[:10],
            "missing_count": len(missing_launch_counts),
        },
        "every eligible case has a launch count frozen by the pilot before any timed run",
    )

    # 3 ── S02 runtime shapes available
    s02 = provenance.get("s02_shapes") or {}
    check(
        "s02_runtime_shapes_available",
        bool(s02.get("census_exists")) and int(s02.get("unique_signature_count") or 0) > 0,
        {
            "census_dir": s02.get("census_dir"),
            "runs_imported": s02.get("runs_imported"),
            "unique_signature_count": s02.get("unique_signature_count"),
            "planned_real_shapes": s02.get("planned_real_shapes"),
            "missing_from_census": s02.get("missing_from_census"),
            "extra_in_census": s02.get("extra_in_census"),
        },
        "without the S02 shape ledger the verdict would be BLOCKED, not PASS",
    )

    # 4 ── the derived call matrix matches the census
    check(
        "s02_call_matrix_validated_against_census",
        int(s02.get("call_matrix_mismatch_count") or 0) == 0
        and len(s02.get("call_matrix_validation") or []) > 0,
        {
            "validated_signatures": len(s02.get("call_matrix_validation") or []),
            "mismatch_count": s02.get("call_matrix_mismatch_count"),
        },
    )

    # 5 ── three independent processes
    pids = [cs.get("pid") for cs in collect_summaries]
    check(
        "three_independent_processes",
        len(process_records) >= 3 and len({p for p in pids if p}) >= 3,
        {"processes": sorted(process_records), "pids": pids},
        "3 is the minimum evidence gate of protocol §5.3",
    )

    # 6 ── the claimed domain was fully executed
    per_process_ok = {}
    for index, records in process_records.items():
        ids = {r["case_id"] for r in records}
        per_process_ok[str(index)] = {
            "records": len(ids),
            "missing": sorted(expected_ids - ids)[:5],
            "missing_count": len(expected_ids - ids),
            "unexpected": sorted(ids - expected_ids)[:5],
            "unexpected_count": len(ids - expected_ids),
        }
    check(
        "every_planned_case_executed_and_no_not_run_counted",
        all(
            item["missing_count"] == 0 and item["unexpected_count"] == 0
            for item in per_process_ok.values()
        )
        and len(per_process_ok) >= 1,
        per_process_ok,
    )

    # 7 ── claimed-supported variants were actually hit (forced)
    forced_violations = [
        r["case_id"]
        for r in all_records
        if r.get("group") == "main"
        and int(r.get("requested_variant_code") or 0) not in (
            rp.VARIANT_CODE["v0_shared"],
            rp.VARIANT_CODE["v1_warp_shuffle"],
            rp.VARIANT_CODE["v2_vectorized"],
        )
    ]
    variant_coverage: Dict[str, int] = {}
    for record in measured:
        if record.get("group") != "main":
            continue
        variant_coverage[record["variant"]] = variant_coverage.get(record["variant"], 0) + 1
    check(
        "all_forced_variants_hit_in_eligible_domain",
        not forced_violations
        and all(variant_coverage.get(v, 0) > 0 for v in ("v0_shared", "v2_vectorized")),
        {
            "measured_counts_by_variant": variant_coverage,
            "records_with_non_forced_code": forced_violations[:10],
            "auto_never_used": True,
        },
    )

    # 8 ── unsupported pairs were rejected, not silently executed
    unsupported_records = [
        r for r in all_records if r.get("group") == "unsupported"
    ]
    check(
        "unsupported_pairs_rejected_and_output_untouched",
        bool(unsupported_records)
        and not not_rejected
        and all(r.get("launched") is False and r.get("output_untouched") for r in unsupported_records),
        {
            "unsupported_records": len(unsupported_records),
            "not_rejected": [r["case_id"] for r in not_rejected][:10],
            "expected_unsupported_cases_in_plan": len(unsupported),
        },
    )

    # 9 ── no silent wrong result in the performance conclusion
    wrong = [
        r["case_id"]
        for r in all_records
        if r.get("group") == "main"
        and r.get("status") == STATUS_MEASURED
        and (
            r.get("correctness_status") not in ("PASS", "REFERENCE_NOT_GATED")
            or int(r.get("classification_mismatch_count") or 0) != 0
            or not (r.get("numeric_violations") or {}).get("passed", True)
        )
    ]
    check(
        "no_silent_wrong_result_entered_performance_conclusion",
        not wrong and not failed,
        {
            "measured_cases": len(measured),
            "correctness_failures": len(failed),
            "wrong_but_measured": wrong[:10],
        },
    )

    # 10 ── every measured case has device AND host raw samples
    bad_samples = [
        r["case_id"]
        for r in measured
        if len(r.get("device_ms_groups") or []) < 2
        or len(r.get("host_ms_groups") or []) < 2
        or len(r.get("submit_ms_groups") or []) < 2
    ]
    bridge_missing = [
        r["case_id"]
        for r in measured
        if r.get("group") == "main" and not (r.get("bridge_ms_samples") or [])
    ]
    check(
        "per_case_raw_device_and_host_samples_retained",
        not bad_samples,
        {
            "cases_with_lt_2_groups": bad_samples[:10],
            "cases_without_bridge_sample": len(bridge_missing),
            "broker_note": "bridge samples are a separate diagnostic and may be absent",
        },
    )

    # 11 ── telemetry coverage
    telem_available = all(cs.get("telemetry_available") for cs in collect_summaries)
    telem_samples = sum(int(cs.get("telemetry_samples") or 0) for cs in collect_summaries)
    with_window = sum(
        1
        for r in measured
        if int((r.get("telemetry") or {}).get("num_samples") or 0) > 0
    )
    check(
        "temperature_power_clock_recorded",
        telem_available and telem_samples > 0,
        {
            "telemetry_available_all_processes": telem_available,
            "total_samples": telem_samples,
            "measured_cases_with_window_samples": with_window,
            "measured_cases": len(measured),
            "window_coverage_fraction": (with_window / len(measured)) if measured else None,
            "gpu_clock_sources": sorted(
                {str(r.get("gpu_clock_hz_begin")) for r in measured}
            )[:6],
        },
        "temperature/power come from tegrastats; the GPU clock from devfreq sysfs",
    )

    # 12 ── resources recorded
    occ_entries = (resources.get("occupancy") or {}).get("entries") or {}
    occ_ok = True
    for dtype in rp.DTYPES:
        for variant in rp.VARIANT_SUPPORT[dtype]:
            key = f"{dtype}|{variant}|{rp.DISPATCH_BLOCK_SIZE}"
            value = (occ_entries.get(key) or {}).get("occupancy")
            if not value:
                occ_ok = False
    unsupported_zero = all(
        ((occ_entries.get(f"{dtype}|{variant}|256") or {}).get("occupancy") == 0)
        for dtype, variants in rp.UNSUPPORTED_VARIANT_PAIRS.items()
        for variant in variants
    )
    check(
        "per_case_resources_recorded",
        occ_ok
        and unsupported_zero
        and all(
            r.get("threads_per_block") and r.get("warps_per_block") is not None
            and r.get("shared_bytes_per_block") is not None
            for r in measured
        ),
        {
            "occupancy_ok": occ_ok,
            "unsupported_pairs_report_zero_occupancy": unsupported_zero,
            "static_per_kernel_available": bool(
                (resources.get("static") or {}).get("per_kernel")
            ),
            "sample": {
                k: v for k, v in list(occ_entries.items())[:6]
            },
        },
    )

    # 13 ── conventions applied and reproducible
    convention_bad = []
    for record in measured:
        expected_bytes = rp.logical_bytes(
            int(record["rows"]), int(record["hidden"]), str(record["dtype"])
        )
        expected_flops = rp.declared_flops(int(record["rows"]), int(record["hidden"]))
        if (
            record.get("byte_convention_id") != rp.BYTE_CONVENTION_ID
            or record.get("flop_convention_id") != rp.FLOP_CONVENTION_ID
            or int(record.get("logical_bytes") or -1) != expected_bytes
            or int(record.get("declared_flops") or -1) != expected_flops
        ):
            convention_bad.append(record["case_id"])
    check(
        "byte_and_flop_conventions_declared_and_recomputable",
        not convention_bad and bool(measured),
        {
            "byte_convention_id": rp.BYTE_CONVENTION_ID,
            "flop_convention_id": rp.FLOP_CONVENTION_ID,
            "violations": convention_bad[:10],
        },
        "logical bytes are not DRAM bytes; the DRAM/L2 level is never mixed in here",
    )

    # 14 ── statistics reported per run and across runs
    summary_rows = _read_jsonl(output_dir / "case_summary.jsonl")
    stats_bad = [
        s["case_id"]
        for s in summary_rows
        if s.get("status") == STATUS_MEASURED
        and s.get("group") == "main"
        and (
            not (s.get("device_ms") or {}).get("ci")
            or len(s.get("per_process_device_ms_median") or {}) < 3
        )
    ]
    check(
        "per_run_and_cross_run_statistics_reported",
        not stats_bad and bool(summary_rows),
        {
            "case_summary_rows": len(summary_rows),
            "cases_missing_ci_or_process_medians": stats_bad[:10],
        },
    )

    # 15 ── heatmaps + numeric tables
    heatmaps = summary.get("heatmaps") or []
    missing_files = [
        entry for entry in heatmaps
        if not (output_dir / entry["svg"]).is_file()
        or (entry.get("csv") and not (output_dir / entry["csv"]).is_file())
    ]
    views = sorted({entry["view"] for entry in heatmaps})
    required_views = {name for name, _ in HEATMAP_VIEWS}
    check(
        "heatmaps_and_numeric_tables_generated",
        bool(heatmaps)
        and not missing_files
        and required_views.issubset(set(views)),
        {
            "heatmap_count": len(heatmaps),
            "views": views,
            "missing_files": [e["svg"] for e in missing_files][:5],
            "required_views": sorted(required_views),
        },
    )

    # 16 ── winner / tie recorded, degradations kept
    regression_expected = {
        (s["shape_id"], s["dtype"], s["variant"])
        for s in summary_rows
        if s.get("group") == "main"
        and (s.get("speedup_vs_v0") or {}).get("median") is not None
        and float((s.get("speedup_vs_v0") or {}).get("median")) < 1.0
    }
    regression_recorded = {
        (r["shape_id"], r["dtype"], r["variant"]) for r in regressions
    }
    check(
        "winners_ties_and_every_degradation_recorded",
        bool(winners)
        and regression_expected == regression_recorded
        and all("tie_group" in w for w in winners),
        {
            "winner_rows": len(winners),
            "regression_expected": len(regression_expected),
            "regression_recorded": len(regression_recorded),
            "missing_regressions": sorted(regression_expected - regression_recorded)[:5],
            "extra_regressions": sorted(regression_recorded - regression_expected)[:5],
        },
        "a degradation may never be deleted from the table",
    )

    # 17 ── E03-04 / E03-08 handoffs
    roles = sorted((ncu.get("cases") or {}).keys())
    check(
        "ncu_representative_cases_selected",
        {"acceleration", "regression_or_no_gain"}.issubset(set(roles))
        and bool(roles),
        {"roles": roles},
        "the selection rule is fixed in ncu_candidates() before the numbers are read",
    )
    check(
        "routing_candidates_and_forbidden_domains_recorded",
        bool(routing.get("domains")) and bool(routing.get("forbidden_domains")),
        {
            "domains": len(routing.get("domains") or []),
            "forbidden_domains": len(routing.get("forbidden_domains") or []),
            "guard_band": (routing.get("guard_band") or {}).get(
                "micro_min_relative_improvement"
            ),
        },
    )

    # 18 ── measured device ceiling (context for effective GB/s)
    check(
        "device_bandwidth_ceiling_measured",
        float(ceiling.get("copy_GBps") or 0.0) > 0,
        {
            "copy_GBps": ceiling.get("copy_GBps"),
            "read_GBps": ceiling.get("read_GBps"),
            "note": ceiling.get("note"),
        },
    )

    # 19 ── pollution / idempotence after timing
    pollution = [
        r["case_id"]
        for r in measured
        if r.get("input_unchanged_after_timing") is False
        or r.get("weight_unchanged_after_timing") is False
    ]
    unstable = [
        r["case_id"]
        for r in measured
        if r.get("group") == "main" and r.get("output_hash_stable_after_timing") is False
    ]
    check(
        "inputs_unchanged_and_output_hash_stable_after_timing",
        not pollution and not unstable,
        {"pollution": pollution[:10], "unstable_output_hash": unstable[:10]},
    )

    # 20 ── mandatory axes covered
    planned_rows = {c["rows"] for c in plan_cases}
    planned_h = {c["hidden"] for c in plan_cases}
    missing_rows = [r for r in rp.MANDATORY_ROWS if r not in planned_rows]
    missing_h = [h for h in rp.MANDATORY_H if h not in planned_h]
    check(
        "mandatory_rows_and_hidden_columns_present",
        not missing_rows and not missing_h,
        {
            "missing_rows": missing_rows,
            "missing_hidden": missing_h,
            "planned_shape_count": plan_meta.get("shape_count"),
            "planned_case_count": len(plan_cases),
        },
    )

    failed_conditions = [name for name, item in checks.items() if not item["passed"]]
    verdict = {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "generated_at_utc": _utc_now(),
        "overall": "PASS" if not failed_conditions else "FAIL",
        "condition_count": len(checks),
        "passed_condition_count": len(checks) - len(failed_conditions),
        "failed_conditions": failed_conditions,
        "checks": checks,
        "scope_note": (
            "The performance conclusions hold inside the canonical contiguous/aligned "
            "domain verified by E03-01. Misalignment/non-contiguous layouts (E03-03), "
            "stream semantics (E03-06) and sanitizer safety (E03-07) are NOT covered "
            "here, and the S03 stage gate that requires them before a performance "
            "conclusion stays open until those experiments pass."
        ),
        "failures": summary.get("failures"),
    }
    _write_json(output_dir / "verdict.json", verdict)
    print(
        f"verify: {verdict['overall']} "
        f"({verdict['passed_condition_count']}/{verdict['condition_count']})"
    )
    for name in failed_conditions:
        print(f"  FAILED: {name} -> {checks[name]['detail']}")
    return 0 if verdict["overall"] == "PASS" else 1


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "collect", "verify", "summarize"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--s02-census", default=str(S02_CENSUS_DEFAULT_ABS))
    parser.add_argument("--process-index", type=int, default=None)
    parser.add_argument("--processes", type=int, default=PROCESSES_DEFAULT)
    parser.add_argument("--ceiling-mib", type=int, default=64)
    parser.add_argument("--append", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pilot":
        return run_pilot(Path(args.output_dir))
    if args.command == "collect":
        if args.process_index is None:
            return _collect_all_processes(args)
        return run_collect(args)
    if args.command == "summarize":
        return run_summarize(args)
    if args.command == "verify":
        return run_verify(args)
    return 2


def _collect_all_processes(args: argparse.Namespace) -> int:
    """Run the pre-registered number of *independent* processes."""
    processes = int(args.processes)
    for index in range(processes):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "collect",
            "--output-dir",
            str(args.output_dir),
            "--s02-census",
            str(args.s02_census),
            "--process-index",
            str(index),
            "--processes",
            str(processes),
            "--ceiling-mib",
            str(args.ceiling_mib),
        ]
        print(f"==> collect process {index + 1}/{processes}", flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
