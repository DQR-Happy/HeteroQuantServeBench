#!/usr/bin/env python3
"""E03-04: launch configuration sweep and Nsight Compute explanation.

This program is remote-only.  It keeps ordinary timing and NCU replay in
separate commands, binds every measurement to the exact shared library, and
stores both NCU native reports and raw CSV exports.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from hqsb.benchmark import multilevel_profiling as mp  # noqa: E402

EXPERIMENT_ID = "E03-04"
SCHEMA = "hqsb.s03.e03_04.v1"
EPSILON = 1e-6
SEED = 20260918
BLOCKS = (32, 64, 128, 256, 512, 1024)
VARIANTS = ("v0_shared", "v1_warp_shuffle", "v2_vectorized")
VARIANT_CODE = {"v0_shared": 2, "v1_warp_shuffle": 3, "v2_vectorized": 4}
SHAPES = (
    {"shape_id": "r512_h128", "rows": 512, "hidden": 128,
     "role": "speedup_case_A", "source": "E03-02 acceleration candidate; S02 real Q/K width"},
    {"shape_id": "r1_h128", "rows": 1, "hidden": 128,
     "role": "no_gain_case_B", "source": "E03-02 no-gain candidate"},
    {"shape_id": "r1_h2048", "rows": 1, "hidden": 2048,
     "role": "decode_case_C", "source": "S02 real hidden; decode rows=1"},
    {"shape_id": "r1024_h2048", "rows": 1024, "hidden": 2048,
     "role": "prefill_case_D", "source": "S02 real hidden; prefill BxI rows"},
    {"shape_id": "r512_h101", "rows": 512, "hidden": 101,
     "role": "odd_tail_case_E", "source": "E03-02 design boundary (synthetic)"},
)
GROUPS = 4
WARMUP = 12
PROCESSES = 3
GUARD_BAND = 0.05
NCU_SECTIONS = (
    "SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis",
    "SchedulerStats", "WarpStateStats", "SourceCounters",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        p = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                           timeout=timeout, check=False)
        return {"command": command, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:
        return {"command": command, "returncode": 127, "stdout": "",
                "stderr": repr(exc)}


def find_library(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("HQSB_CUDA_RMSNORM_LIB"):
        candidates.append(Path(os.environ["HQSB_CUDA_RMSNORM_LIB"]))
    candidates.extend(sorted(REPO.glob("build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("libhqsb_rmsnorm_shared.so not found")


class LaunchApi:
    def __init__(self, library: Path):
        self.library = library
        self.lib = ctypes.CDLL(str(library))
        if not hasattr(self.lib, "hqsb_rmsnorm_forward_config_ex_c"):
            raise RuntimeError("library lacks E03-04 explicit-config ABI; rebuild required")
        self.forward = self.lib.hqsb_rmsnorm_forward_config_ex_c
        self.forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.forward.restype = ctypes.c_int
        self.occupancy = self.lib.hqsb_rmsnorm_occupancy_c
        self.occupancy.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.occupancy.restype = ctypes.c_int

    def bind(self, x, weight, output, rows: int, hidden: int, variant: str,
             block: int, stream) -> Any:
        fn = self.forward
        args = (x.data_ptr(), weight.data_ptr(), output.data_ptr(), rows, hidden,
                EPSILON, 0, VARIANT_CODE[variant], block, int(stream.cuda_stream))

        def launch() -> int:
            return int(fn(*args))
        return launch


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def sample_count(rows: int, hidden: int) -> int:
    if rows * hidden >= 1024 * 2048:
        return 20
    if rows * hidden >= 512 * 128:
        return 50
    return 200


def correctness(output, reference) -> dict[str, Any]:
    diff = (output.float() - reference.float()).reshape(-1)
    abs_diff = diff.abs()
    denom = max(float(reference.float().norm().item()), 1e-30)
    return {"max_abs": float(abs_diff.max().item()),
            "mean_abs": float(abs_diff.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
            "l2_relative": float(diff.norm().item() / denom),
            "passed": bool(abs_diff.max().item() <= 2e-5 and
                           diff.norm().item() / denom <= 2e-5)}


def time_launch(launch, stream, launches: int) -> dict[str, Any]:
    for _ in range(WARMUP):
        if launch() != 0:
            return {"error": "warmup launch failed"}
    stream.synchronize()
    device, submit, completion = [], [], []
    for _ in range(GROUPS):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(stream)
        for _ in range(launches):
            rc = launch()
            if rc:
                return {"error": f"launch returned cudaError={rc}"}
        stop.record(stream)
        stop.synchronize()
        device.append(float(start.elapsed_time(stop) / launches))

        t0 = time.monotonic_ns()
        for _ in range(launches):
            rc = launch()
            if rc:
                return {"error": f"host timing launch returned cudaError={rc}"}
        t1 = time.monotonic_ns()
        stream.synchronize()
        t2 = time.monotonic_ns()
        submit.append((t1 - t0) / 1e6 / launches)
        completion.append((t2 - t0) / 1e6 / launches)
    return {"device_ms_samples": device, "host_submit_ms_samples": submit,
            "host_submit_completion_ms_samples": completion}


def device_resources() -> dict[str, Any]:
    p = torch.cuda.get_device_properties(0)
    names = ("name", "major", "minor", "multi_processor_count", "total_memory",
             "warp_size", "max_threads_per_block", "max_threads_per_multi_processor",
             "regs_per_multiprocessor", "shared_memory_per_block",
             "shared_memory_per_multiprocessor")
    values = {name: getattr(p, name, None) for name in names}
    # Some Jetson PyTorch builds expose a non-UTF8 byte in the verbose
    # ``_CudaDeviceProperties.__repr__``.  The individual numeric fields are
    # authoritative and avoid that environment-specific decoding failure.
    values["memory_bus_width_bits"] = getattr(p, "memory_bus_width", None)
    values["memory_clock_rate_khz"] = getattr(p, "memory_clock_rate", None)
    return values


def static_resources(library: Path) -> dict[str, Any]:
    tool = "/usr/local/cuda/bin/cuobjdump"
    result = run([tool, "--dump-resource-usage", str(library)])
    result["library_sha256"] = sha256_file(library)
    result["compile_contract"] = {"cmake_target": "hqsb_rmsnorm_shared",
                                  "cuda_flags": ["-O3", "-lineinfo"],
                                  "fast_math": False}
    return result


def make_plan(output: Path) -> int:
    prior = load(REPO / "docs/stage_experiments/S03/E03-02/raw/ncu_candidates.json", {})
    plan = {
        "schema_version": SCHEMA, "experiment_id": EXPERIMENT_ID,
        "state": "FROZEN_BEFORE_E03_04_MEASUREMENT", "frozen_at_utc": utc_now(),
        "processes": PROCESSES, "groups": GROUPS, "warmup": WARMUP,
        "block_sizes": list(BLOCKS), "variants": list(VARIANTS), "dtype": "fp32",
        "epsilon": EPSILON, "guard_band": GUARD_BAND, "shapes": list(SHAPES),
        "profile_selection_rule": {
            "acceleration_pair": "r512_h128: best V0 versus best V1 by pooled ordinary median",
            "degradation_pair": "r1_h128 V2: fastest versus slowest block by pooled ordinary median",
            "regime_pair": "best V1 config independently for r1_h2048 and r1024_h2048",
            "selection_time": "rules frozen now; concrete blocks resolved only after ordinary sweep",
        },
        "ncu_sections_requested": list(NCU_SECTIONS),
        "hypotheses": {
            "acceleration": "V1 reduces shared traffic/barriers relative to V0",
            "degradation": "oversized blocks waste lanes on H=128; occupancy alone will not predict latency",
            "regime": "rows=1 is grid-underfilled while prefill supplies enough CTAs to expose throughput behavior",
        },
        "e03_02_frozen_input": prior,
    }
    dump(output / "protocol.json", plan)
    print(json.dumps({"status": "FROZEN", "configurations_per_process":
                      len(SHAPES) * len(VARIANTS) * len(BLOCKS)}))
    return 0


def collect(args) -> int:
    output = Path(args.output_dir).resolve()
    protocol = load(output / "protocol.json")
    if not protocol or protocol.get("state") != "FROZEN_BEFORE_E03_04_MEASUREMENT":
        raise RuntimeError("run plan before collect")
    library = find_library(args.library)
    api = LaunchApi(library)
    resources = device_resources()
    stream = torch.cuda.Stream()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + args.process_index)
    records = []
    configs = [(shape, variant, block) for shape in SHAPES
               for variant in VARIANTS for block in BLOCKS]
    random.Random(SEED + args.process_index).shuffle(configs)
    for shape, variant, block in configs:
        rows, hidden = int(shape["rows"]), int(shape["hidden"])
        x = torch.randn((rows, hidden), generator=generator, device="cuda",
                        dtype=torch.float32) * 0.25
        weight = torch.randn((hidden,), generator=generator, device="cuda",
                             dtype=torch.float32) * 0.1 + 1.0
        output_tensor = torch.empty_like(x)
        reference = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1,
                                                        keepdim=True) + EPSILON) * weight.float()
        # Tensor initialisation above runs on PyTorch's current/default stream,
        # whereas the audited kernel deliberately uses an explicit non-default
        # stream.  Establish the producer -> consumer dependency before the
        # first launch; otherwise the first configuration in a fresh process
        # can race its input initialisation.
        torch.cuda.synchronize()
        launch = api.bind(x, weight, output_tensor, rows, hidden, variant, block, stream)
        rc = launch()
        stream.synchronize()
        numeric = correctness(output_tensor, reference) if rc == 0 else {"passed": False}
        occ_blocks = int(api.occupancy(VARIANT_CODE[variant], 0, block))
        max_threads_sm = int(resources.get("max_threads_per_multi_processor") or 1536)
        theoretical_occ = min(100.0, 100.0 * occ_blocks * block / max_threads_sm)
        capacity_blocks = occ_blocks * int(resources.get("multi_processor_count") or 0)
        record = {
            "schema_version": SCHEMA, "experiment_id": EXPERIMENT_ID,
            "process_index": args.process_index, "case_id":
                f"{shape['shape_id']}|fp32|{variant}|b{block}",
            **shape, "dtype": "fp32", "variant": variant, "block_size": block,
            "threads_per_block": block, "warps_per_block": block // 32,
            "grid_blocks": rows,
            "dynamic_shared_bytes": block * 4 if variant == "v0_shared" else block // 32 * 4,
            "occupancy_max_blocks_per_sm": occ_blocks,
            "theoretical_occupancy_pct": theoretical_occ,
            "grid_resident_capacity_blocks": capacity_blocks,
            "grid_fill_fraction_first_wave": min(1.0, rows / max(capacity_blocks, 1)),
            "epsilon": EPSILON, "cuda_return_code": rc, "correctness": numeric,
            "actual_kernel_path": ("scalar_safe" if variant == "v2_vectorized" and hidden % 4
                                   else variant),
            "launches_per_group": sample_count(rows, hidden),
            "library_sha256": sha256_file(library),
        }
        if rc == 0 and numeric.get("passed"):
            record.update(time_launch(launch, stream, record["launches_per_group"]))
            record["status"] = "MEASURED" if "error" not in record else "FAIL_TIMING"
        else:
            record["status"] = "FAIL_CORRECTNESS"
        records.append(record)
        print(record["case_id"], record["status"], flush=True)
    path = output / f"cases_proc{args.process_index}.jsonl"
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.process_index == 0:
        dump(output / "device_resources.json", resources)
        dump(output / "binary_resources.json", static_resources(library))
        git = run(["git", "status", "--short"])
        dump(output / "provenance.json", {
            "schema_version": SCHEMA, "timestamp_utc": utc_now(),
            "git_commit": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
            "git_dirty": bool(git["stdout"].strip()),
            "git_status_short": git["stdout"].splitlines(),
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "library": str(library.relative_to(REPO)), "library_sha256": sha256_file(library),
            "source_sha256": {str(path.relative_to(REPO)): sha256_file(path) for path in (
                REPO / "ops/cuda/rmsnorm/src/rmsnorm_v0.cu",
                REPO / "ops/cuda/rmsnorm/src/rmsnorm_v1.cu",
                REPO / "ops/cuda/rmsnorm/src/rmsnorm_v2.cu",
                REPO / "ops/cuda/rmsnorm/src/rmsnorm_c_api.cu",
                Path(__file__).resolve(),
            )},
        })
    return 0


def read_records(output: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output.glob("cases_proc*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["case_id"], []).append(record)
    table = []
    for case_id, group in sorted(grouped.items()):
        device = [v for r in group for v in r.get("device_ms_samples", [])]
        host = [v for r in group for v in r.get("host_submit_completion_ms_samples", [])]
        base = group[0]
        median = statistics.median(device) if device else math.nan
        table.append({
            "case_id": case_id, "shape_id": base["shape_id"], "role": base["role"],
            "rows": base["rows"], "hidden": base["hidden"], "dtype": base["dtype"],
            "variant": base["variant"], "block_size": base["block_size"],
            "threads_per_block": base["threads_per_block"],
            "warps_per_block": base["warps_per_block"],
            "dynamic_shared_bytes": base["dynamic_shared_bytes"],
            "occupancy_max_blocks_per_sm": base["occupancy_max_blocks_per_sm"],
            "theoretical_occupancy_pct": base["theoretical_occupancy_pct"],
            "grid_fill_fraction_first_wave": base["grid_fill_fraction_first_wave"],
            "processes": len(group), "device_samples": len(device),
            "all_correct": all(r.get("correctness", {}).get("passed") for r in group),
            "device_ms_median": median, "device_ms_p95": percentile(device, .95),
            "device_ms_cv": statistics.pstdev(device) / median if len(device) > 1 and median else None,
            "host_ms_median": statistics.median(host) if host else math.nan,
            "effective_gbps": (base["rows"] * base["hidden"] * 12) / (median / 1000) / 1e9
                              if median else None,
        })
    return table


def best(table: list[dict[str, Any]], shape: str, variant: str) -> dict[str, Any]:
    rows = [r for r in table if r["shape_id"] == shape and r["variant"] == variant]
    return min(rows, key=lambda r: r["device_ms_median"])


def worst(table: list[dict[str, Any]], shape: str, variant: str) -> dict[str, Any]:
    rows = [r for r in table if r["shape_id"] == shape and r["variant"] == variant]
    return max(rows, key=lambda r: r["device_ms_median"])


def profile_plan(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for profile_id, relation, row in (
        ("accel_v0", "acceleration_baseline", best(table, "r512_h128", "v0_shared")),
        ("accel_v1", "acceleration_candidate", best(table, "r512_h128", "v1_warp_shuffle")),
        ("nogain_v2_fast", "degradation_fast", best(table, "r1_h128", "v2_vectorized")),
        ("nogain_v2_slow", "degradation_slow", worst(table, "r1_h128", "v2_vectorized")),
        ("decode_v1", "regime_decode", best(table, "r1_h2048", "v1_warp_shuffle")),
        ("prefill_v1", "regime_prefill", best(table, "r1024_h2048", "v1_warp_shuffle")),
    ):
        candidates.append({"profile_id": profile_id, "relation": relation,
                           **{k: row[k] for k in ("case_id", "shape_id", "rows", "hidden",
                                                  "variant", "block_size", "device_ms_median")}})
    return candidates


def parse_ncu_csv(text: str) -> dict[str, Any]:
    parsed = mp.parse_ncu_csv(text)
    kernels = parsed.get("kernels") or []
    if not kernels:
        return {"error": parsed.get("error", "no kernel rows"), "kernels": []}
    kernel = kernels[0]
    tokens = ("dram", "lts", "l2", "shared", "bank", "barrier", "spill",
              "local", "eligible", "issued", "occupancy", "register")
    diagnostic_metrics = {
        name: item for name, item in kernel.get("metrics", {}).items()
        if any(token in name.lower() for token in tokens)
    }
    return {"kernel_name": kernel.get("name"), "grid": kernel.get("grid"),
            "block": kernel.get("block"), "panel": kernel.get("panel", {}),
            "stall_metrics": mp.ncu_stall_metrics(kernel.get("metrics", {})),
            "diagnostic_metrics": diagnostic_metrics,
            "metric_count": len(kernel.get("metrics", {})),
            "unavailable_metrics": sorted(name for name, item in kernel.get("metrics", {}).items()
                                           if item.get("value") is None),
            "kernels": len(kernels)}


def parse_ncu_raw_csv(text: str) -> dict[str, Any]:
    """Extract causal counters from NCU's one-row wide raw export."""
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return {"error": "raw CSV has fewer than header/unit/value rows", "counters": {}}
    header, units, values = rows[0], rows[1], rows[2]
    selected: dict[str, Any] = {}
    tokens = ("dram__", "lts__t_sector", "l1tex__data_bank_conflicts",
              "memory_l1_wavefronts_shared", "derived__memory_l1_conflicts_shared",
              "warps_issue_stalled", "warps_eligible", "warps_active",
              "launch__registers", "launch__shared_mem", "launch__occupancy",
              "memory_l2_theoretical_sectors_local")
    for index, name in enumerate(header):
        if not any(token in name for token in tokens):
            continue
        raw = values[index] if index < len(values) else ""
        try:
            value: Any = float(raw.replace(",", ""))
        except ValueError:
            value = None if raw.lower() in ("", "n/a", "nan", "-") else raw
        selected[name] = {"value": value,
                          "unit": units[index] if index < len(units) else "",
                          "raw": raw}
    return {"counters": selected, "counter_count": len(selected),
            "dram_counter_count": sum(name.startswith("dram__") for name in selected),
            "note": "zero DRAM counters means this Tegra NCU build did not expose them; L2 counters remain explicit"}


def parse_device_limits_raw_csv(text: str) -> dict[str, Any]:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return {"error": "raw CSV unavailable"}
    header, units, values = rows[0], rows[1], rows[2]
    wanted = (
        "device__attribute_warp_size", "device__attribute_multiprocessor_count",
        "device__attribute_max_threads_per_block",
        "device__attribute_max_threads_per_multiprocessor",
        "device__attribute_max_warps_per_multiprocessor",
        "device__attribute_max_blocks_per_multiprocessor",
        "device__attribute_max_registers_per_multiprocessor",
        "device__attribute_max_shared_memory_per_block",
        "device__attribute_max_shared_memory_per_block_optin",
        "device__attribute_max_shared_memory_per_multiprocessor",
        "device__attribute_global_memory_bus_width",
        "device__attribute_memory_clock_rate",
        "device__attribute_max_mem_frequency_khz",
    )
    result = {}
    for name in wanted:
        if name not in header:
            result[name] = None
            continue
        index = header.index(name)
        raw = values[index]
        try:
            value: Any = float(raw.replace(",", ""))
            if value.is_integer():
                value = int(value)
        except ValueError:
            value = raw
        result[name] = {"value": value, "unit": units[index], "source": "NCU raw export"}
    return result


def ncu_capture(args) -> int:
    output = Path(args.output_dir).resolve()
    library = find_library(args.library)
    table = load(output / "ordinary_summary.json", {}).get("configurations", [])
    if not table:
        raise RuntimeError("summarize ordinary sweep before NCU")
    ncu = "/usr/local/cuda/bin/ncu"
    query = {"version": run([ncu, "--version"]),
             "sections": run([ncu, "--list-sections"], timeout=180),
             "requested_sections": list(NCU_SECTIONS)}
    dump(output / "ncu" / "tool_query.json", query)
    candidates = profile_plan(table)
    dump(output / "ncu_candidates.json", {"selection_rule": load(output / "protocol.json")
         ["profile_selection_rule"], "candidates": candidates})
    outcomes = []
    for c in candidates:
        stem = output / "ncu" / c["profile_id"]
        report = stem.with_suffix(".ncu-rep")
        command = ["sudo", "-n", "-E", ncu]
        for section in NCU_SECTIONS:
            command += ["--section", section]
        command += ["--kernel-name", "regex:rmsnorm_.*_kernel", "--launch-skip", "3",
                    "--launch-count", "1", "--export", str(report), "--force-overwrite",
                    sys.executable, str(Path(__file__).resolve()), "profile",
                    "--library", str(library), "--rows", str(c["rows"]),
                    "--hidden", str(c["hidden"]), "--variant", c["variant"],
                    "--block", str(c["block_size"])]
        captured = run(command, timeout=900)
        (stem.with_suffix(".stdout.txt")).write_text(captured["stdout"])
        (stem.with_suffix(".stderr.txt")).write_text(captured["stderr"])
        exported = run(["sudo", "-n", ncu, "--import", str(report), "--csv",
                        "--page", "details", "--print-units", "base"],
                       timeout=300) if report.exists() else {"returncode": 1, "stdout": "", "stderr": "no report"}
        csv_path = stem.with_suffix(".csv")
        csv_path.write_text(exported["stdout"])
        stem.with_suffix(".export.stderr.txt").write_text(exported.get("stderr", ""))
        raw_exported = run(["sudo", "-n", ncu, "--import", str(report), "--csv",
                            "--page", "raw", "--print-units", "base"],
                           timeout=300) if report.exists() else {
                               "returncode": 1, "stdout": "", "stderr": "no report"}
        raw_csv_path = stem.with_suffix(".raw.csv")
        raw_csv_path.write_text(raw_exported["stdout"])
        run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(report)])
        outcome = {**c, "command": command, "capture_returncode": captured["returncode"],
                   "export_returncode": exported["returncode"], "report_exists": report.is_file(),
                   "report_sha256": sha256_file(report) if report.is_file() else None,
                   "csv_sha256": sha256_file(csv_path),
                   "raw_csv_sha256": sha256_file(raw_csv_path),
                   "parsed": parse_ncu_csv(exported["stdout"]),
                   "raw_counters": parse_ncu_raw_csv(raw_exported["stdout"])}
        dump(stem.with_suffix(".json"), outcome)
        outcomes.append(outcome)
        print(c["profile_id"], captured["returncode"], outcome["parsed"].get("metric_count"), flush=True)
    dump(output / "ncu" / "capture_summary.json", {"outcomes": outcomes})
    return 0 if all(o["capture_returncode"] == 0 and not o["parsed"].get("error") for o in outcomes) else 2


def profile(args) -> int:
    library = find_library(args.library)
    api = LaunchApi(library)
    torch.manual_seed(SEED)
    x = torch.randn((args.rows, args.hidden), device="cuda", dtype=torch.float32) * .25
    w = torch.randn((args.hidden,), device="cuda", dtype=torch.float32) * .1 + 1
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    launch = api.bind(x, w, out, args.rows, args.hidden, args.variant, args.block, stream)
    for _ in range(4):
        if launch() != 0:
            return 2
    stream.synchronize()
    return 0


def counter_comparisons(output: Path) -> dict[str, Any]:
    captures = load(output / "ncu" / "capture_summary.json", {}).get("outcomes", [])
    by_id = {c["profile_id"]: c for c in captures}
    pairs = []
    for pair_id, base_id, candidate_id, hypothesis in (
        ("acceleration", "accel_v0", "accel_v1", "shuffle reduces shared/barrier work"),
        ("degradation", "nogain_v2_fast", "nogain_v2_slow", "oversized/undersized block hurts despite occupancy"),
        ("rows_regime", "decode_v1", "prefill_v1", "grid fill changes utilization; workloads differ so not causal A/B"),
    ):
        if base_id not in by_id or candidate_id not in by_id:
            continue
        a, b = by_id[base_id], by_id[candidate_id]
        pa, pb = a["parsed"].get("panel", {}), b["parsed"].get("panel", {})
        keys = sorted(set(pa) | set(pb))
        delta = {key: {"baseline": pa.get(key), "candidate": pb.get(key)} for key in keys}
        pairs.append({"pair_id": pair_id, "baseline": base_id, "candidate": candidate_id,
                      "hypothesis": hypothesis,
                      "ordinary_latency_ms": {"baseline": a["device_ms_median"],
                                              "candidate": b["device_ms_median"],
                                              "speedup": a["device_ms_median"] / b["device_ms_median"]},
                      "panel": delta, "stall_baseline": a["parsed"].get("stall_metrics", {}),
                      "stall_candidate": b["parsed"].get("stall_metrics", {}),
                      "diagnostic_metrics_baseline": a["parsed"].get("diagnostic_metrics", {}),
                      "diagnostic_metrics_candidate": b["parsed"].get("diagnostic_metrics", {}),
                      "raw_causal_counters_baseline": a.get("raw_counters", {}).get("counters", {}),
                      "raw_causal_counters_candidate": b.get("raw_counters", {}).get("counters", {}),
                      "dram_counter_availability": {
                          "baseline_count": a.get("raw_counters", {}).get("dram_counter_count", 0),
                          "candidate_count": b.get("raw_counters", {}).get("dram_counter_count", 0),
                          "interpretation": "zero means unavailable on this Tegra NCU build, not zero DRAM traffic",
                      }})
    return {"pairs": pairs}


def manifest(output: Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    dump(output / "EVIDENCE_MANIFEST.json", {"schema_version": SCHEMA,
         "generated_at_utc": utc_now(), "files": files})


def summarize(args) -> int:
    output = Path(args.output_dir).resolve()
    records = read_records(output)
    table = aggregate(records)
    dump(output / "ordinary_summary.json", {"schema_version": SCHEMA,
         "records": len(records), "configurations": table})
    policies = []
    for shape in SHAPES:
        choices = [r for r in table if r["shape_id"] == shape["shape_id"]]
        choices.sort(key=lambda r: r["device_ms_median"])
        winner, runner = choices[0], choices[1]
        policies.append({"shape_id": shape["shape_id"], "role": shape["role"],
                         "config_id": f"{winner['variant']}|b{winner['block_size']}",
                         "variant": winner["variant"], "block_size": winner["block_size"],
                         "latency_ms": winner["device_ms_median"],
                         "speedup_vs_runner_up": runner["device_ms_median"] / winner["device_ms_median"],
                         "guard_band_clear": runner["device_ms_median"] / winner["device_ms_median"] > 1 + GUARD_BAND,
                         "fallback": "production default b256 until E03-08 routing is validated",
                         "boundary": shape["source"]})
    dump(output / "config_policy.json", {"guard_band": GUARD_BAND, "policies": policies})
    comparisons = counter_comparisons(output)
    dump(output / "counter_comparisons.json", comparisons)
    ncu_outcomes = load(output / "ncu" / "capture_summary.json", {}).get("outcomes", [])
    raw_anchor = output / "ncu" / "accel_v0.raw.csv"
    dump(output / "device_limits_resolved.json", {
        "schema_version": SCHEMA, "captured_at_utc": utc_now(),
        "ncu_device_attributes": parse_device_limits_raw_csv(
            raw_anchor.read_text() if raw_anchor.is_file() else ""),
        "nvpmodel": run(["nvpmodel", "-q"]),
        "jetson_clocks": run(["sudo", "-n", "jetson_clocks", "--show"]),
        "note": "read after ordinary/NCU collection; no power or clock setting was changed",
    })
    conditions = {
        "matrix_complete": len(table) == len(SHAPES) * len(VARIANTS) * len(BLOCKS),
        "three_processes_each": bool(table) and all(row["processes"] == PROCESSES for row in table),
        "all_configurations_correct": bool(table) and all(row["all_correct"] for row in table),
        "ordinary_latency_present": bool(table) and all(row["device_samples"] == PROCESSES * GROUPS for row in table),
        "resource_fields_present": bool(table) and all(row["occupancy_max_blocks_per_sm"] > 0 for row in table),
        "ncu_six_candidates_present": len(ncu_outcomes) == 6,
        "ncu_native_and_csv_valid": len(ncu_outcomes) == 6 and all(
            row.get("report_exists") and row.get("parsed", {}).get("metric_count", 0) > 0
            for row in ncu_outcomes),
        "ncu_launch_faithful": len(ncu_outcomes) == 6 and all(
            row.get("parsed", {}).get("block") == f"({row['block_size']}, 1, 1)" and
            row.get("parsed", {}).get("grid") == f"({row['rows']}, 1, 1)"
            for row in ncu_outcomes),
        "raw_causal_counters_present": len(ncu_outcomes) == 6 and all(
            row.get("raw_counters", {}).get("counter_count", 0) > 0 for row in ncu_outcomes),
        "acceleration_explained": any(p["pair_id"] == "acceleration" for p in comparisons["pairs"]),
        "degradation_explained": any(p["pair_id"] == "degradation" for p in comparisons["pairs"]),
        "config_policy_covers_all_shapes": len(policies) == len(SHAPES),
    }
    overall = "PASS" if all(conditions.values()) else (
        "BLOCKED" if not conditions["ncu_native_and_csv_valid"] else "FAIL")
    verdict = {"schema_version": SCHEMA, "experiment_id": EXPERIMENT_ID,
               "verified_at_utc": utc_now(), "overall": overall,
               "conditions": conditions,
               "failed_conditions": [key for key, value in conditions.items() if not value]}
    dump(output / "summary.json", {"configurations": len(table), "policies": policies,
         "counter_pairs": comparisons["pairs"], "verdict_preview": verdict})
    dump(output / "verdict.json", verdict)
    manifest(output)
    print(json.dumps(verdict, ensure_ascii=False))
    return 0 if overall == "PASS" else 2


def verify(args) -> int:
    verdict = load(Path(args.output_dir).resolve() / "verdict.json", {})
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict.get("overall") == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "summarize", "verify", "ncu"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", required=True)
        if name == "ncu":
            p.add_argument("--library")
    p = sub.add_parser("collect")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--process-index", type=int, required=True, choices=range(PROCESSES))
    p.add_argument("--library")
    p = sub.add_parser("profile")
    p.add_argument("--library", required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--block", type=int, choices=BLOCKS, required=True)
    args = parser.parse_args()
    return {"plan": lambda: make_plan(Path(args.output_dir).resolve()),
            "collect": lambda: collect(args), "ncu": lambda: ncu_capture(args),
            "profile": lambda: profile(args), "summarize": lambda: summarize(args),
            "verify": lambda: verify(args)}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
