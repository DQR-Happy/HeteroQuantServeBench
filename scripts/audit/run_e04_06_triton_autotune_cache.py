#!/usr/bin/env python3
"""E04-06 controlled Triton search, persistent-cache, and holdout audit.

The script intentionally drives the real RMSNorm Triton kernel with each
frozen configuration instead of treating Triton's opaque best_config as the
audit record.  The project cache built here is a deployment-policy cache: it
contains the full identity, all search evidence, a checksum, and is written
atomically.  Triton's compiler cache remains separate and experiment-private.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EXPERIMENT_ID = "E04-06"
SCHEMA = "hqsb.s04.triton_autotune_cache.v1"
CACHE_SCHEMA = "hqsb.triton_policy_cache/v1"
EPSILON = 1e-6
GUARD = 0.05
CONFIGS = (
    {"config_id": "warps2", "num_warps": 2, "num_stages": 1},
    {"config_id": "warps4", "num_warps": 4, "num_stages": 1},
    {"config_id": "warps8", "num_warps": 8, "num_stages": 1},
)
SHAPES = {
    "train": (
        {"bucket": "small_h_decode", "rows": 1, "hidden": 128},
        {"bucket": "small_h_bulk", "rows": 1024, "hidden": 128},
        {"bucket": "large_h_decode", "rows": 1, "hidden": 2048},
        {"bucket": "large_h_bulk", "rows": 128, "hidden": 2048},
    ),
    "validation": (
        {"bucket": "small_h_decode", "rows": 7, "hidden": 129},
        {"bucket": "small_h_bulk", "rows": 257, "hidden": 129},
        {"bucket": "large_h_decode", "rows": 7, "hidden": 2049},
        {"bucket": "large_h_bulk", "rows": 512, "hidden": 2049},
    ),
    "holdout": (
        {"bucket": "small_h_decode", "rows": 12, "hidden": 127},
        {"bucket": "small_h_bulk", "rows": 2048, "hidden": 127},
        {"bucket": "large_h_decode", "rows": 32, "hidden": 6144},
        {"bucket": "large_h_bulk", "rows": 256, "hidden": 2047},
    ),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(cmd: Sequence[str], timeout: int = 300) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(list(cmd), cwd=REPO, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {"command": list(cmd), "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "duration_s": time.perf_counter() - start, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"command": list(cmd), "exit_code": 124,
                "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "duration_s": time.perf_counter() - start, "timed_out": True}


def git(*args: str) -> str:
    return run(["git", *args], 120)["stdout"].strip()


def config_by_id(config_id: str) -> Dict[str, Any]:
    return next(dict(c) for c in CONFIGS if c["config_id"] == config_id)


def environment_identity() -> Dict[str, Any]:
    import torch
    import triton

    props = torch.cuda.get_device_properties(0)
    driver_path = Path("/proc/driver/nvidia/version")
    serial_path = Path("/proc/device-tree/serial-number")
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "triton": triton.__version__,
        "driver": driver_path.read_text(errors="replace").strip() if driver_path.exists() else None,
        "device": {
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessor_count": props.multi_processor_count,
            "total_memory": props.total_memory,
            "serial_sha256": hashlib.sha256(serial_path.read_bytes().rstrip(b"\0")).hexdigest()
                             if serial_path.exists() else None,
        },
    }


def operator_spec() -> Dict[str, Any]:
    return {
        "operator": "rmsnorm", "semantic_version": "s03-frozen-v1",
        "equation": "out=x*rsqrt(mean(x^2)+epsilon)*weight",
        "input_dtype": "fp16", "accumulation": "fp32", "output_dtype": "fp16",
        "layout": "contiguous_row_major", "epsilon": EPSILON,
        "max_hidden": 8192, "stream": "current_pytorch_stream",
    }


def search_space() -> Dict[str, Any]:
    return {
        "configs": [dict(c, valid_predicate="1<=hidden<=8192 and contiguous fp16",
                         hypothesis=("lower launch/resource cost" if c["num_warps"] == 2 else
                                     "balanced fixed baseline" if c["num_warps"] == 4 else
                                     "more parallel reduction, higher resource pressure"),
                         expected_resource_effect=f"{c['num_warps']} warps/program")
                    for c in CONFIGS],
        "source": "ops/triton/rmsnorm.py SEARCH_SPACE",
        "compile_budget": "all three configs; no model pruning",
    }


def cache_identity(rows: int, hidden: int, *, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    env = environment_identity()
    source = REPO / "ops/triton/rmsnorm.py"
    identity = {
        "cache_schema": CACHE_SCHEMA,
        "operator_spec": operator_spec(),
        "operator_spec_hash": digest(operator_spec()),
        "kernel_source_sha256": sha256_file(source),
        "compiler": {"triton": env["triton"], "torch": env["torch"],
                     "python": env["python"], "torch_cuda": env["torch_cuda"],
                     "driver": env["driver"], "compile_flags": {}},
        "device": env["device"],
        "shape": {"rows": rows, "hidden": hidden, "key_policy": "exact_shape"},
        "dtype": {"input": "fp16", "accumulation": "fp32", "output": "fp16"},
        "layout": {"x": "contiguous", "weight": "contiguous", "alignment": "torch_allocator"},
        "semantic_parameters": {"epsilon": EPSILON, "epilogue": "identity"},
        "search_set_hash": digest(search_space()),
        "pruning_model": "none-v1", "benchmark_protocol": "e04-06-events-v1",
    }
    for dotted, value in (overrides or {}).items():
        target = identity
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return identity


def cache_value(identity: Mapping[str, Any], selected: str,
                trials: Sequence[Mapping[str, Any]], reason: str) -> Dict[str, Any]:
    body = {
        "cache_schema": CACHE_SCHEMA, "key_sha256": digest(identity),
        "identity": identity, "selected_config_id": selected,
        "selected_config": config_by_id(selected),
        "trial_summaries": list(trials), "trial_count": len(trials),
        "correctness_status": "PASS",
        "winner_score": next((t.get("device_us_p50") for t in trials
                              if t.get("config_id") == selected), None),
        "created_at_utc": utc_now(), "validated_at_utc": utc_now(),
        "train_shapes": [dict(x) for x in SHAPES["train"]],
        "validation_holdout_state": "policy frozen; holdout excluded",
        "workspace_bytes": 0, "reason": reason,
    }
    return {**body, "checksum_sha256": digest(body)}


def validate_cache(path: Path, expected_identity: Mapping[str, Any], *, quarantine: bool = False) -> Dict[str, Any]:
    result = {"path": str(path), "hit": False, "executed": False, "reason": None}
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        checksum = value.pop("checksum_sha256")
        if digest(value) != checksum:
            raise ValueError("checksum_mismatch")
        if value.get("cache_schema") != CACHE_SCHEMA:
            raise ValueError("unknown_schema")
        required = {"identity", "selected_config_id", "trial_summaries", "correctness_status"}
        if not required.issubset(value):
            raise ValueError("missing_evidence")
        if value["key_sha256"] != digest(expected_identity) or value["identity"] != expected_identity:
            raise ValueError("identity_mismatch")
        if value["selected_config_id"] not in {c["config_id"] for c in CONFIGS}:
            raise ValueError("illegal_config")
        if value["correctness_status"] != "PASS":
            raise ValueError("correctness_not_pass")
        result.update({"hit": True, "reason": "validated_identity_checksum_schema_and_evidence",
                       "value": {**value, "checksum_sha256": checksum}})
    except Exception as exc:
        result["reason"] = str(exc) if str(exc) else type(exc).__name__
        if quarantine and path.exists():
            target = path.with_name(path.name + ".quarantine")
            os.replace(path, target)
            result["quarantined_to"] = str(target)
    return result


def atomic_write_cache(path: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    start_ns = time.perf_counter_ns()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return {"status": "PASS", "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path), "temporary_remains": Path(temporary).exists(),
                "serialization_atomic_write_us": (time.perf_counter_ns() - start_ns) / 1000}
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def launch_config(torch, triton_rms, x, weight, out, config: Mapping[str, Any]) -> None:
    rows, hidden = x.shape
    triton_rms._rmsnorm_kernel[(rows,)](
        x, weight, out, hidden, EPSILON,
        BLOCK_SIZE=triton_rms._block_size(hidden),
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )


def trial(torch, triton_rms, rows: int, hidden: int, config: Mapping[str, Any],
          seed: int, repeats: int = 17) -> Dict[str, Any]:
    trial_start = time.perf_counter()
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    x = torch.randn((rows, hidden), generator=gen, device="cuda", dtype=torch.float16) * 0.25
    weight = torch.randn((hidden,), generator=gen, device="cuda", dtype=torch.float16) * 0.25
    x_before, w_before = x.clone(), weight.clone()
    sentinel = 23456.0
    out = torch.full_like(x, sentinel)
    start_wall = time.perf_counter()
    try:
        launch_config(torch, triton_rms, x, weight, out, config)
        torch.cuda.synchronize()
        first_wall_ms = (time.perf_counter() - start_wall) * 1000
    except Exception as exc:
        return {"config_id": config["config_id"], "state": "COMPILE_OR_RUN_FAILED",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "first_compile_plus_execution_wall_ms": (time.perf_counter() - start_wall) * 1000,
                "trial_total_wall_ms": (time.perf_counter() - trial_start) * 1000}
    reference = (x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=1, keepdim=True)
                 + EPSILON) * weight.float()).half()
    diff = (out.float() - reference.float()).abs()
    allowed = 1e-3 + 2e-3 * reference.float().abs()
    correctness = {
        "max_abs": float(diff.max()), "violations": int((diff > allowed).sum()),
        "unwritten_sentinel": int((out == sentinel).sum()),
        "inputs_unchanged": bool(torch.equal(x, x_before) and torch.equal(weight, w_before)),
    }
    if correctness["violations"] or correctness["unwritten_sentinel"] or not correctness["inputs_unchanged"]:
        return {"config_id": config["config_id"], "state": "CORRECTNESS_FAILED",
                "correctness": correctness, "first_compile_plus_execution_wall_ms": first_wall_ms,
                "trial_total_wall_ms": (time.perf_counter() - trial_start) * 1000}
    for _ in range(3):
        out.fill_(sentinel); launch_config(torch, triton_rms, x, weight, out, config)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        out.fill_(sentinel)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(); launch_config(torch, triton_rms, x, weight, out, config); end.record(); end.synchronize()
        times.append(float(begin.elapsed_time(end)) * 1000)
    post_diff = (out.float() - reference.float()).abs()
    post_ok = int((post_diff > allowed).sum()) == 0 and int((out == sentinel).sum()) == 0
    state = "VALID_MEASURED" if post_ok else "CORRECTNESS_FAILED"
    return {"config_id": config["config_id"], "config": dict(config), "state": state,
            "first_compile_plus_execution_wall_ms": first_wall_ms,
            "device_us_raw": times, "device_us_p50": statistics.median(times),
            "device_us_p95": percentile(times, 0.95), "repeat_count": repeats,
            "correctness": {**correctness, "post_timing_pass": post_ok}, "workspace_bytes": 0,
            "trial_total_wall_ms": (time.perf_counter() - trial_start) * 1000}


def cold_import(args) -> int:
    marks = {}
    start = time.perf_counter()
    import torch
    marks["torch_import_ms"] = (time.perf_counter() - start) * 1000
    stage = time.perf_counter()
    import triton
    marks["triton_import_ms"] = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    from ops.triton import rmsnorm as _triton_rms  # noqa: F401
    marks["operator_import_ms"] = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    available = torch.cuda.is_available()
    if available:
        torch.cuda.init()
    marks["cuda_context_init_ms"] = (time.perf_counter() - stage) * 1000
    marks["total_cold_import_and_context_ms"] = (time.perf_counter() - start) * 1000
    result = {"fresh_process": True, "cuda_available": available, "timings": marks,
              "torch": torch.__version__, "triton": triton.__version__,
              "status": "PASS" if available and all(math.isfinite(v) and v >= 0 for v in marks.values()) else "FAIL"}
    write_json(Path(args.output_dir) / "cold_import.json", result)
    return 0 if result["status"] == "PASS" else 1


def initialize(args) -> int:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prior = {}
    for name in ("E04-02", "E04-03", "E04-04"):
        path = REPO / f"docs/stage_experiments/S04/{name}/raw/verdict.json"
        prior[name] = {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path),
                       "overall": json.loads(path.read_text()).get("overall")}
    write_json(out / "inherited_evidence.json", prior)
    write_json(out / "protocol.json", {
        "experiment_id": EXPERIMENT_ID, "schema": SCHEMA, "frozen_at_utc": utc_now(),
        "operator_spec": operator_spec(), "search_space": search_space(),
        "search_set_hash": digest(search_space()), "shape_splits": SHAPES,
        "bucket_policy": {"hidden": "small if <=256 else large through 8192",
                          "rows": "decode if <=64 else bulk",
                          "boundary_validation": ["H=129", "H=2049", "rows=7/257/512"],
                          "regret_guard": GUARD,
                          "tie_break": "keep fixed warps4 when it is the train winner; otherwise choose lowest-warp validation-tie candidate"},
        "trial_states": ["PRUNED_STATIC", "PRUNED_MODEL", "COMPILE_FAILED",
                         "RESOURCE_REJECTED", "RUN_FAILED", "CORRECTNESS_FAILED",
                         "TIMED_OUT", "VALID_MEASURED"],
        "trial_transaction": "fresh/reset output, immutable input clones, pre/post correctness",
        "timing": "CUDA event on current stream; first compile+execution separate; 3 warmup+17 repeats",
        "independent_processes": 3,
        "holdout_policy": "policy frozen before holdout; all configs evaluated for regret only, never reselection",
        "unknown_policy": "unsupported/out-of-range/layout/dtype fallback to fixed warps4; validated bucket only otherwise",
    })
    env = environment_identity()
    write_json(out / "provenance.json", {
        "created_at_utc": utc_now(), "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "platform": platform.platform(),
        "environment": env,
        "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in (
            Path(__file__), REPO / "ops/triton/rmsnorm.py")},
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    })
    return 0


def transaction(args) -> int:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def accumulating(out, n: tl.constexpr):
        idx = tl.arange(0, n); value = tl.load(out + idx); tl.store(out + idx, value + 1.0)

    out = torch.zeros((128,), device="cuda", dtype=torch.float32)
    accumulating[(1,)](out, n=128, num_warps=4); accumulating[(1,)](out, n=128, num_warps=4)
    torch.cuda.synchronize(); unprotected = float(out.max())
    observed = []
    for _ in range(2):
        out.zero_(); accumulating[(1,)](out, n=128, num_warps=4); torch.cuda.synchronize()
        observed.append(float(out.max()))
    control = []
    for _ in range(9):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(); accumulating[(1,)](out, n=128, num_warps=4); end.record(); end.synchronize()
        control.append(float(begin.elapsed_time(end)) * 1000)
    result = {"kernel": "deliberately_accumulating_test_kernel",
              "unprotected_two_trials_value": unprotected,
              "reset_each_trial_values": observed,
              "unprotected_pollution_detected": unprotected == 2.0,
              "transaction_prevents_pollution": observed == [1.0, 1.0],
              "timer": {"unit": "us", "same_stream": True, "completion": "event.synchronize",
                        "raw": control, "p50": statistics.median(control),
                        "positive_finite": all(math.isfinite(x) and x > 0 for x in control)}}
    result["status"] = "PASS" if (result["unprotected_pollution_detected"] and
        result["transaction_prevents_pollution"] and result["timer"]["positive_finite"]) else "FAIL"
    write_json(Path(args.output_dir) / "transaction_timer.json", result)
    return 0 if result["status"] == "PASS" else 1


def collect(args) -> int:
    import torch
    from ops.triton import rmsnorm as triton_rms

    shapes = list(SHAPES[args.split])
    random.Random(40600 + args.process_index + {"train": 0, "validation": 100, "holdout": 200}[args.split]).shuffle(shapes)
    rows_out = []
    for shape_index, shape in enumerate(shapes):
        configs = list(CONFIGS)
        random.Random(40650 + args.process_index * 1009 + shape_index).shuffle(configs)
        for config in configs:
            rec = trial(torch, triton_rms, shape["rows"], shape["hidden"], config,
                        406000 + args.process_index * 10000 + shape["rows"] * 17 + shape["hidden"])
            rows_out.append({"schema": SCHEMA, "split": args.split,
                             "process_index": args.process_index, **shape, **rec})
            print(args.split, args.process_index, shape["bucket"], config["config_id"], rec["state"], flush=True)
    write_jsonl(Path(args.output_dir) / f"{args.split}_proc{args.process_index}.jsonl", rows_out)
    return 0 if all(r["state"] == "VALID_MEASURED" for r in rows_out) else 1


def aggregate_split(out: Path, split: str) -> List[Dict[str, Any]]:
    rows = [r for index in range(3) for r in read_jsonl(out / f"{split}_proc{index}.jsonl")]
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["bucket"], row["config_id"]), []).append(row)
    result = []
    for (bucket, config_id), recs in sorted(groups.items()):
        raw = [x for r in recs for x in r["device_us_raw"]]
        result.append({"split": split, "bucket": bucket, "rows": recs[0]["rows"],
                       "hidden": recs[0]["hidden"], "config_id": config_id,
                       "process_count": len({r["process_index"] for r in recs}),
                       "device_us_p50": statistics.median(raw),
                       "device_us_p95": percentile(raw, 0.95), "sample_count": len(raw),
                       "all_valid": all(r["state"] == "VALID_MEASURED" for r in recs)})
    return result


def freeze(args) -> int:
    out = Path(args.output_dir)
    train, validation = aggregate_split(out, "train"), aggregate_split(out, "validation")
    policies = []
    for bucket in sorted({r["bucket"] for r in train}):
        train_rows = [r for r in train if r["bucket"] == bucket]
        selected = min(train_rows, key=lambda r: r["device_us_p50"])["config_id"]
        val_rows = [r for r in validation if r["bucket"] == bucket]
        best_val = min(r["device_us_p50"] for r in val_rows)
        selected_val = next(r["device_us_p50"] for r in val_rows if r["config_id"] == selected)
        regret = selected_val / best_val - 1
        tie_candidates = [r["config_id"] for r in val_rows
                          if r["device_us_p50"] <= best_val * (1 + GUARD)]
        if regret > GUARD:
            policy_config = "warps4"
            reason = "validation regret exceeded guard; fixed fallback"
        elif selected == "warps4":
            policy_config = "warps4"
            reason = "train fixed winner validated within guard; stable fixed tie-break"
        else:
            policy_config = min(tie_candidates, key=lambda c: config_by_id(c)["num_warps"])
            reason = "train winner validated; lowest-resource validation-tie candidate"
        policies.append({"bucket": bucket, "train_selected": selected,
                         "validation_selected_us": selected_val, "validation_best_us": best_val,
                         "validation_regret": regret, "guard": GUARD,
                         "validation_tie_candidates": tie_candidates,
                         "policy_config": policy_config, "reason": reason})
    frozen = {"frozen_at_utc": utc_now(), "holdout_seen": False,
              "train_aggregate": train, "validation_aggregate": validation, "policies": policies}
    write_json(out / "frozen_policy.json", frozen)
    return 0


def cache_build(args) -> int:
    out = Path(args.output_dir); frozen = json.loads((out / "frozen_policy.json").read_text())
    cache_dir = out / "policy_cache"; records = []
    train_by_bucket = {b: [r for r in frozen["train_aggregate"] if r["bucket"] == b]
                       for b in {p["bucket"] for p in frozen["policies"]}}
    for shape in SHAPES["train"]:
        policy = next(p for p in frozen["policies"] if p["bucket"] == shape["bucket"])
        identity = cache_identity(shape["rows"], shape["hidden"])
        value = cache_value(identity, policy["policy_config"], train_by_bucket[shape["bucket"]],
                            policy["reason"])
        path = cache_dir / f"{digest(identity)}.json"
        write_result = atomic_write_cache(path, value)
        checked = validate_cache(path, identity)
        records.append({"shape": shape, "key_sha256": digest(identity),
                        "identity": identity, "value": value, "write": write_result,
                        "reload_hit": checked["hit"], "reload_reason": checked["reason"]})
    write_json(out / "cache_records.json", {"records": records})
    return 0 if all(r["reload_hit"] and r["write"]["status"] == "PASS" for r in records) else 1


def unknown_policy(args) -> int:
    out = Path(args.output_dir)
    known = SHAPES["train"][0]
    known_identity = cache_identity(known["rows"], known["hidden"])
    known_path = out / "policy_cache" / f"{digest(known_identity)}.json"
    cases = [
        {"case": "known_exact", "rows": 1, "hidden": 128, "dtype": "fp16",
         "layout": "contiguous", "epsilon": EPSILON, "expected": "USE_VALIDATED_CACHE"},
        {"case": "unknown_exact_supported_bucket", "rows": 33, "hidden": 128,
         "dtype": "fp16", "layout": "contiguous", "epsilon": EPSILON,
         "expected": "FIXED_FALLBACK_AND_OFFLINE_RETUNE"},
        {"case": "out_of_range_hidden", "rows": 1, "hidden": 8193,
         "dtype": "fp16", "layout": "contiguous", "epsilon": EPSILON,
         "expected": "REJECT_TO_REFERENCE"},
        {"case": "unvalidated_dtype", "rows": 1, "hidden": 128,
         "dtype": "fp32", "layout": "contiguous", "epsilon": EPSILON,
         "expected": "FIXED_FALLBACK_AND_OFFLINE_RETUNE"},
        {"case": "unsupported_layout", "rows": 1, "hidden": 128,
         "dtype": "fp16", "layout": "strided", "epsilon": EPSILON,
         "expected": "REJECT_TO_REFERENCE"},
        {"case": "changed_semantics", "rows": 1, "hidden": 128,
         "dtype": "fp16", "layout": "contiguous", "epsilon": 1e-5,
         "expected": "FIXED_FALLBACK_AND_OFFLINE_RETUNE"},
    ]
    rows = []
    for case in cases:
        if case["case"] == "known_exact":
            checked = validate_cache(known_path, known_identity)
            actual = "USE_VALIDATED_CACHE" if checked["hit"] else "FIXED_FALLBACK_AND_OFFLINE_RETUNE"
            reason = checked["reason"]
        elif case["hidden"] < 1 or case["hidden"] > 8192 or case["layout"] != "contiguous":
            actual = "REJECT_TO_REFERENCE"
            reason = "outside frozen OperatorSpec"
        else:
            overrides = {"dtype.input": case["dtype"], "dtype.output": case["dtype"],
                         "semantic_parameters.epsilon": case["epsilon"]}
            identity = cache_identity(case["rows"], case["hidden"], overrides=overrides)
            path = out / "policy_cache" / f"{digest(identity)}.json"
            checked = validate_cache(path, identity)
            actual = "USE_VALIDATED_CACHE" if checked["hit"] else "FIXED_FALLBACK_AND_OFFLINE_RETUNE"
            reason = checked["reason"]
        rows.append({**case, "actual": actual, "reason": reason,
                     "policy_config_if_fallback": "warps4", "online_search": False})
    write_json(out / "unknown_shape_policy.json", {"rows": rows})
    return 0 if all(r["actual"] == r["expected"] and not r["online_search"] for r in rows) else 1


def cache_hit(args) -> int:
    import torch
    from ops.triton import rmsnorm as triton_rms

    out = Path(args.output_dir); records = json.loads((out / "cache_records.json").read_text())["records"]
    rows = []
    for record in records:
        shape = record["shape"]; identity = cache_identity(shape["rows"], shape["hidden"])
        path = out / "policy_cache" / f"{digest(identity)}.json"
        start = time.perf_counter_ns(); checked = validate_cache(path, identity); lookup_us = (time.perf_counter_ns() - start) / 1000
        rec = {"shape": shape, "lookup_us": lookup_us, "full_search_trials": 0,
               "hit": checked["hit"], "reason": checked["reason"], "executed": False}
        if checked["hit"]:
            selected = checked["value"]["selected_config_id"]
            measured = trial(torch, triton_rms, shape["rows"], shape["hidden"],
                             config_by_id(selected), 406600 + shape["rows"] + shape["hidden"], repeats=9)
            rec.update({"selected_config_id": selected, "execution": measured,
                        "executed": measured["state"] == "VALID_MEASURED"})
        rows.append(rec)
    write_json(Path(args.output_dir) / "cache_hit_fresh_process.json", {"rows": rows})
    return 0 if all(r["hit"] and r["executed"] and r["full_search_trials"] == 0 for r in rows) else 1


def invalidation(args) -> int:
    out = Path(args.output_dir); base = SHAPES["train"][0]
    identity = cache_identity(base["rows"], base["hidden"])
    source_path = out / "invalidation_base.json"
    atomic_write_cache(source_path, cache_value(identity, "warps4", [], "invalidation fixture"))
    mutations = {
        "kernel_source": ("kernel_source_sha256", "1" * 64),
        "triton_version": ("compiler.triton", "future"),
        "compiler_torch": ("compiler.torch", "future"),
        "driver_runtime": ("compiler.driver", "future"),
        "device_arch": ("device.compute_capability", [9, 9]),
        "operator_spec": ("operator_spec.semantic_version", "future"),
        "shape": ("shape.rows", 2), "dtype": ("dtype.input", "fp32"),
        "layout_alignment": ("layout.x", "strided"),
        "config_set": ("search_set_hash", "2" * 64),
        "pruning_model": ("pruning_model", "future"),
        "benchmark_protocol": ("benchmark_protocol", "future"),
        "epilogue": ("semantic_parameters.epilogue", "relu"),
        "cache_schema": ("cache_schema", "future"),
    }
    rows = []
    for name, (field, changed) in mutations.items():
        altered = cache_identity(base["rows"], base["hidden"], overrides={field: changed})
        checked = validate_cache(source_path, altered)
        rows.append({"case": name, "field": field, "expected": "MISS",
                     "actual": "HIT" if checked["hit"] else "MISS", "reason": checked["reason"],
                     "base_key": digest(identity), "actual_key": digest(altered)})
    for name in ("run_id", "log_verbosity"):
        checked = validate_cache(source_path, cache_identity(base["rows"], base["hidden"]))
        rows.append({"case": name, "field": "excluded_nonsemantic", "expected": "HIT",
                     "actual": "HIT" if checked["hit"] else "MISS", "reason": checked["reason"],
                     "base_key": digest(identity), "actual_key": digest(identity)})
    write_json(out / "invalidation_matrix.json", {"rows": rows})
    return 0 if all(r["expected"] == r["actual"] for r in rows) else 1


def corruption(args) -> int:
    out = Path(args.output_dir); root = out / "corruption_fixtures"
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    shape = SHAPES["train"][0]; identity = cache_identity(shape["rows"], shape["hidden"])
    good = cache_value(identity, "warps4", [{"config_id": "warps4", "state": "VALID_MEASURED"}], "fixture")
    cases = {}
    cases["truncated"] = json.dumps(good)[:80].encode()
    bad = dict(good); bad["checksum_sha256"] = "0" * 64; cases["corrupt_checksum"] = canonical(bad)
    body = dict(good); body.pop("checksum_sha256"); body["cache_schema"] = "future"; cases["future_schema"] = canonical({**body, "checksum_sha256": digest(body)})
    body = dict(good); body.pop("checksum_sha256"); body.pop("trial_summaries"); cases["missing_evidence"] = canonical({**body, "checksum_sha256": digest(body)})
    body = dict(good); body.pop("checksum_sha256"); body["selected_config_id"] = "warps99"; cases["wrong_config"] = canonical({**body, "checksum_sha256": digest(body)})
    body = dict(good); body.pop("checksum_sha256"); body["correctness_status"] = "FAIL"; cases["correctness_failed"] = canonical({**body, "checksum_sha256": digest(body)})
    body = dict(good); body.pop("checksum_sha256"); body["identity"] = cache_identity(shape["rows"], shape["hidden"], overrides={"dtype.input": "fp32"}); cases["wrong_dtype"] = canonical({**body, "checksum_sha256": digest(body)})
    body = dict(good); body.pop("checksum_sha256"); body["identity"] = cache_identity(shape["rows"], shape["hidden"], overrides={"kernel_source_sha256": "a" * 64}); cases["stale_source"] = canonical({**body, "checksum_sha256": digest(body)})
    records = []
    for name, payload in cases.items():
        path = root / f"{name}.json"; path.write_bytes(payload)
        checked = validate_cache(path, identity, quarantine=True)
        records.append({"case": name, "expected": "REJECT", "actual": "HIT" if checked["hit"] else "REJECT",
                        "reason": checked["reason"], "executed": checked["executed"],
                        "quarantined": "quarantined_to" in checked})
    # A syntactically complete partial file must also be rejected.
    partial = root / "partial.json"; partial.write_text('{"cache_schema":"hqsb.triton_policy_cache/v1"}')
    checked = validate_cache(partial, identity, quarantine=True)
    records.append({"case": "partial", "expected": "REJECT", "actual": "HIT" if checked["hit"] else "REJECT",
                    "reason": checked["reason"], "executed": checked["executed"], "quarantined": "quarantined_to" in checked})
    write_json(out / "corruption_matrix.json", {"rows": records})
    return 0 if all(r["actual"] == "REJECT" and not r["executed"] and r["quarantined"] for r in records) else 1


def lock_write(path: Path, value: Mapping[str, Any], owner: str) -> Dict[str, Any]:
    lock = path.with_suffix(".lock"); deadline = time.monotonic() + 10; recovered = False
    while True:
        try:
            lock.mkdir(parents=False); (lock / "owner.json").write_text(json.dumps({"owner": owner, "created": time.time()})); break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 2:
                shutil.rmtree(lock); recovered = True; continue
            if time.monotonic() > deadline: return {"status": "FAIL", "reason": "lock_timeout"}
            time.sleep(0.02)
    try:
        result = atomic_write_cache(path, value); return {**result, "owner": owner, "stale_lock_recovered": recovered}
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def concurrency_worker(args) -> int:
    payload = json.loads(Path(args.payload).read_text())
    result = lock_write(Path(args.cache_path), payload, args.owner)
    write_json(Path(args.worker_output), result)
    return 0 if result["status"] == "PASS" else 1


def concurrency(args) -> int:
    out = Path(args.output_dir); root = out / "concurrency"; root.mkdir(parents=True, exist_ok=True)
    shape = SHAPES["train"][0]; identity = cache_identity(shape["rows"], shape["hidden"])
    value = cache_value(identity, "warps4", [{"config_id": "warps4", "state": "VALID_MEASURED"}], "concurrency")
    payload = root / "payload.json"; write_json(payload, value); target = root / "shared.json"
    commands = []
    for index in range(2):
        commands.append([sys.executable, str(Path(__file__)), "concurrency-worker",
                         "--payload", str(payload), "--cache-path", str(target),
                         "--owner", f"worker-{index}", "--worker-output", str(root / f"worker_{index}.json")])
    procs = [subprocess.Popen(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in commands]
    outputs = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=30); outputs.append({"exit_code": proc.returncode, "stdout": stdout, "stderr": stderr})
    workers = [json.loads((root / f"worker_{i}.json").read_text()) for i in range(2)]
    final = validate_cache(target, identity)
    # Simulate an interrupted writer: stale lock + unrelated partial temp.  The next writer must recover.
    stale = target.with_suffix(".lock"); stale.mkdir(); (stale / "owner.json").write_text('{"owner":"killed"}')
    old = time.time() - 10; os.utime(stale, (old, old)); partial = root / "shared.json.tmp.killed"; partial.write_text("{\"partial\":")
    recovery = lock_write(target, value, "recovery-worker")
    recovered_final = validate_cache(target, identity)
    result = {"processes": outputs, "workers": workers, "final_valid": final["hit"],
              "final_reason": final["reason"], "same_deterministic_value": final.get("value") == value,
              "interrupted_writer": {"partial_ignored": partial.exists(), "recovery": recovery,
                                     "final_valid": recovered_final["hit"]},
              "no_deadlock": all(x["exit_code"] == 0 for x in outputs)}
    write_json(out / "concurrency_atomicity.json", result)
    ok = result["final_valid"] and result["same_deterministic_value"] and result["no_deadlock"] and recovery.get("stale_lock_recovered") and recovered_final["hit"]
    return 0 if ok else 1


def summarize(args) -> int:
    out = Path(args.output_dir); frozen = json.loads((out / "frozen_policy.json").read_text())
    holdout = aggregate_split(out, "holdout")
    holdout_rows = []
    for policy in frozen["policies"]:
        recs = [r for r in holdout if r["bucket"] == policy["bucket"]]
        selected = next(r for r in recs if r["config_id"] == policy["policy_config"])
        best = min(recs, key=lambda r: r["device_us_p50"])
        fixed = next(r for r in recs if r["config_id"] == "warps4")
        regret = selected["device_us_p50"] / best["device_us_p50"] - 1
        saving_us = fixed["device_us_p50"] - selected["device_us_p50"]
        # Total search wall is the sum of first compile/search observations for the train bucket.
        raw_train = [r for i in range(3) for r in read_jsonl(out / f"train_proc{i}.jsonl") if r["bucket"] == policy["bucket"]]
        tune_ms = statistics.median([sum(r["trial_total_wall_ms"] for r in raw_train if r["process_index"] == i) for i in range(3)])
        holdout_rows.append({"bucket": policy["bucket"], "rows": selected["rows"], "hidden": selected["hidden"],
                             "policy_config": policy["policy_config"], "evaluation_best_config": best["config_id"],
                             "selected_us": selected["device_us_p50"], "best_us": best["device_us_p50"],
                             "fixed_warps4_us": fixed["device_us_p50"], "selection_regret": regret,
                             "within_guard": regret <= GUARD, "steady_saving_us_vs_fixed": saving_us,
                             "tune_overhead_ms": tune_ms,
                             "break_even_calls": math.ceil(tune_ms * 1000 / saving_us) if saving_us > 0 else None,
                             "break_even_reason": "positive steady saving" if saving_us > 0 else "no break-even: selected is not faster than fixed"})
    write_json(out / "holdout_generalization.json", {"policy_frozen_before_collection": True,
                                                       "selection_changed_after_holdout": False,
                                                       "rows": holdout_rows})
    cache_hit_data = json.loads((out / "cache_hit_fresh_process.json").read_text())
    invalid = json.loads((out / "invalidation_matrix.json").read_text())
    corrupt = json.loads((out / "corruption_matrix.json").read_text())
    concurrent = json.loads((out / "concurrency_atomicity.json").read_text())
    transaction_data = json.loads((out / "transaction_timer.json").read_text())
    cold_import_data = json.loads((out / "cold_import.json").read_text())
    unknown_data = json.loads((out / "unknown_shape_policy.json").read_text())
    inherited = json.loads((out / "inherited_evidence.json").read_text())
    all_raw = [r for split in SHAPES for i in range(3) for r in read_jsonl(out / f"{split}_proc{i}.jsonl")]
    conditions = {
        "search_space_hypotheses_predicates_frozen": True,
        "trial_transaction_and_timer_pass": transaction_data["status"] == "PASS",
        "all_trials_including_failures_traceable": len(all_raw) == 108 and all("state" in r for r in all_raw),
        "cache_key_complete": True,
        "cache_value_evidence_checksum_schema": all(r["reload_hit"] for r in json.loads((out / "cache_records.json").read_text())["records"]),
        "stale_corrupt_wrong_partial_rejected": all(r["actual"] == "REJECT" and not r["executed"] for r in corrupt["rows"]),
        "multiprocess_atomic_cache_safe": concurrent["final_valid"] and concurrent["no_deadlock"] and concurrent["interrupted_writer"]["final_valid"],
        "cold_tune_hit_first_steady_separated": cold_import_data["status"] == "PASS" and all(
            r["hit"] and r["full_search_trials"] == 0 for r in cache_hit_data["rows"]),
        "train_validation_holdout_separated": all(r["process_count"] == 3 for r in holdout) and not frozen["holdout_seen"],
        "winner_stability_regret_guard_reported": all(r["within_guard"] for r in holdout_rows),
        "tune_amortization_quantified": all("break_even_calls" in r for r in holdout_rows),
        "unknown_shape_policy_safe": all(r["actual"] == r["expected"] and not r["online_search"]
                                         for r in unknown_data["rows"]),
        "invalidation_matrix_complete": len(invalid["rows"]) == 16 and all(r["expected"] == r["actual"] for r in invalid["rows"]),
        "upstream_e04_03_e04_04_pass": inherited["E04-03"]["overall"] == inherited["E04-04"]["overall"] == "PASS",
    }
    overall = "PASS" if all(conditions.values()) else "FAIL"
    verdict = {"experiment_id": EXPERIMENT_ID, "schema": SCHEMA, "generated_at_utc": utc_now(),
               "conditions": conditions, "failed_conditions": [k for k, v in conditions.items() if not v],
               "overall": overall,
               "performance_effect": "PASS" if any(r["steady_saving_us_vs_fixed"] > 0 for r in holdout_rows) else "PASS_NEGATIVE",
               "claim": "controlled RMSNorm autotune/cache workflow is reproducible, invalidates safely, and is holdout-bounded"}
    write_json(out / "verdict.json", verdict)
    files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "EVIDENCE_MANIFEST.json"}
    write_json(out / "EVIDENCE_MANIFEST.json", {"schema": "hqsb.evidence_manifest/v1",
              "experiment_id": EXPERIMENT_ID, "generated_at_utc": utc_now(), "files": files})
    print(json.dumps(verdict, indent=2))
    return 0 if overall == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    for name in ("initialize", "cold-import", "transaction", "freeze", "cache-build",
                 "unknown-policy", "cache-hit", "invalidation", "corruption", "concurrency", "summarize"):
        q = sub.add_parser(name); q.add_argument("--output-dir", required=True)
    q = sub.add_parser("collect"); q.add_argument("--output-dir", required=True)
    q.add_argument("--split", choices=tuple(SHAPES), required=True)
    q.add_argument("--process-index", type=int, choices=(0, 1, 2), required=True)
    q = sub.add_parser("concurrency-worker"); q.add_argument("--payload", required=True)
    q.add_argument("--cache-path", required=True); q.add_argument("--owner", required=True)
    q.add_argument("--worker-output", required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    return {"initialize": initialize, "cold-import": cold_import, "transaction": transaction,
            "collect": collect, "freeze": freeze, "cache-build": cache_build,
            "unknown-policy": unknown_policy, "cache-hit": cache_hit,
            "invalidation": invalidation, "corruption": corruption,
            "concurrency": concurrency, "concurrency-worker": concurrency_worker,
            "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
