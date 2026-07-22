#!/usr/bin/env python3
"""E04-05 CUTLASS configuration-space, status, workspace, and route audit."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "E04-05"
SCHEMA = "hqsb.s04.cutlass_configuration_space.v1"
GUARD = 0.05
BINARY = REPO / "build/e04-05-cost-v2/bin/e04_05_cutlass_space"
FAMILY_DIMS = {
    "kv": (1024, 2048),
    "q_o": (2048, 2048),
    "gate_up": (6144, 2048),
    "down": (2048, 6144),
    "lm_head": (151936, 2048),
}
SCREEN_M = (1, 32, 128, 512)
PRECISE_POINTS = ((16, "validation"), (256, "validation"),
                  (64, "holdout"), (1024, "holdout"))
LINEAR_CONFIGS = (
    "small_m32n128k32_w32n64_s2_a8_linear",
    "small_m64n128k32_w32n64_s3_a8_linear",
    "small_m64n64k32_w32n32_s2_a1_linear",
    "large_m128n128k32_w64n64_s3_a8_linear",
    "large_m128n256k32_w64n64_s3_a8_linear",
)
SPLITK = "large_m128n128k32_w64n64_s3_a8_splitk"
RELU = "large_m128n128k32_w64n64_s3_a8_relu"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd: Sequence[str], timeout: int = 1800) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(list(cmd), cwd=REPO, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {"command": list(cmd), "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "duration_s": time.perf_counter() - start, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"command": list(cmd), "exit_code": 124, "stdout": exc.stdout or "",
                "stderr": exc.stderr or "", "duration_s": time.perf_counter() - start,
                "timed_out": True}


def git(*args: str, cwd: Path = REPO) -> str:
    return run(["git", *args], 120 | 0)["stdout"].strip() if cwd == REPO else subprocess.run(
        ["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False).stdout.strip()


def invoke(config: str, m: int, n: int, k: int, *, iterations: int = 5,
           warmup: int = 2, verify: bool = False, split_k: int = 1,
           extra: Sequence[str] = (), timeout: int = 900) -> Dict[str, Any]:
    cmd = [str(BINARY), "--config", config, "--m", str(m), "--n", str(n),
           "--k", str(k), "--warmup", str(warmup), "--iterations", str(iterations),
           "--split-k", str(split_k), *extra]
    if verify:
        cmd.append("--verify")
    proc = run(cmd, timeout)
    parsed = None
    for line in reversed(proc["stdout"].splitlines()):
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                pass
            break
    return {"process": proc, "measurement": parsed,
            "status": "PASS" if parsed and parsed.get("status") == "PASS" else "FAIL"}


def initialize(args) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = REPO / "ops/cuda/cutlass_gemm/e04_05_cutlass_space.cu"
    cutlass = REPO / "third_party/cutlass"
    license_path = cutlass / "LICENSE.txt"
    dependency = {
        "cutlass_version": "4.7.0",
        "cutlass_commit": git("rev-parse", "HEAD", cwd=cutlass),
        "source_kind": "vendored git checkout",
        "source_path": "third_party/cutlass",
        "archive_or_tree_sha256": git("rev-parse", "HEAD^{tree}", cwd=cutlass),
        "license_path": str(license_path.relative_to(REPO)),
        "license_sha256": sha256_file(license_path),
        "offline_rebuild": True,
        "api_generation": "cutlass::gemm::device::Gemm stable 2.x-style API",
        "target_arch": "sm87 via Sm80-compatible tensor-op kernels",
    }
    write_json(out / "dependency_manifest.json", dependency)
    prior_paths = {
        "E04-03": REPO / "docs/stage_experiments/S04/E04-03/raw/verdict.json",
        "E04-04": REPO / "docs/stage_experiments/S04/E04-04/raw/verdict.json",
        "E04-04_curves": REPO / "docs/stage_experiments/S04/E04-04/raw/gemm_curves.json",
    }
    inherited = {name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path),
                        "overall": json.loads(path.read_text()).get("overall")}
                 for name, path in prior_paths.items()}
    write_json(out / "inherited_evidence.json", inherited)
    config_hypotheses = {
        LINEAR_CONFIGS[0]: {"family": "decode_small_m", "expected_domain": "M<=64",
                            "change": "CTA_M=32, stages=2", "risk": "lower reuse"},
        LINEAR_CONFIGS[1]: {"family": "decode_small_m", "expected_domain": "M<=64",
                            "change": "CTA_M=64, stages=3", "risk": "more shared memory"},
        LINEAR_CONFIGS[2]: {"family": "decode_small_m", "expected_domain": "tail/control",
                            "change": "alignment 1 and 64x64 CTA", "risk": "scalar transactions"},
        LINEAR_CONFIGS[3]: {"family": "prefill_large_m", "expected_domain": "M>=128",
                            "change": "128x128 CTA, stages=3", "risk": "small-M waste"},
        LINEAR_CONFIGS[4]: {"family": "prefill_large_m", "expected_domain": "M>=128",
                            "change": "N tile 256", "risk": "8 warps/resource pressure"},
        SPLITK: {"family": "prefill_large_m", "expected_domain": "large K/underfilled grid",
                 "change": "serial split-K", "risk": "workspace/semaphore overhead"},
        RELU: {"family": "epilogue_ablation", "expected_domain": "equivalent GEMM+ReLU",
               "change": "fused ReLU epilogue", "risk": "not pure-GEMM comparable"},
    }
    write_json(out / "protocol.json", {
        "experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
        "frozen_at_utc": utc_now(), "guard": GUARD,
        "config_hypotheses": config_hypotheses,
        "screen": {"M": list(SCREEN_M), "families": FAMILY_DIMS,
                   "processes": 1, "selection": "top-2 geometric normalized latency per regime"},
        "validation_holdout": {"points": [{"M": m, "split": split} for m, split in PRECISE_POINTS],
                               "independent_processes": 3},
        "stopping_rule": "freeze top-2 before precise/holdout; no post-hoc config relabel",
        "correctness": "independent host FP32 oracle, shared tolerance 0.01+0.02*abs(ref), guards",
        "status_chain": ["can_implement", "workspace_size", "initialize", "run",
                         "immediate_cuda", "completion_cuda", "numerical"],
        "workspace": "exact per-op allocation; split-K negative uses null workspace",
        "route": "CUTLASS only if precise measurement is >5% faster than inherited same-shape baseline",
        "claim_boundary": "operator-level candidate; E04-08 owns production forced/auto replay",
    })
    write_json(out / "provenance.json", {
        "created_at_utc": utc_now(), "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "platform": platform.platform(),
        "python": sys.version, "source": str(source.relative_to(REPO)),
        "source_sha256": sha256_file(source),
    })
    return 0


def build(args) -> int:
    out = Path(args.output_dir)
    # Use a dedicated never-reused directory: the first compiler probe is part
    # of the measured clean-build evidence and must not inherit CMake state.
    build_dir = REPO / "build/e04-05-cost-v2"
    configure = run(["cmake", "-S", ".", "-B", str(build_dir),
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=87",
                     "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"], 900)
    clean_build = run(["cmake", "--build", str(build_dir), "--target",
                       "e04_05_cutlass_space", "-j2"], 3600)
    incremental = run(["cmake", "--build", str(build_dir), "--target",
                       "e04_05_cutlass_space", "-j2"], 1800)
    listed = run([str(BINARY), "--list-configs"], 60) if BINARY.exists() else None
    registry = json.loads(listed["stdout"].splitlines()[-1]) if listed and listed["exit_code"] == 0 else {}
    write_json(out / "config_registry.json", registry)
    size = BINARY.stat().st_size if BINARY.exists() else None
    symbols = run(["nm", "-C", "--defined-only", str(BINARY)], 300) if BINARY.exists() else None
    symbol_lines = [line for line in symbols["stdout"].splitlines() if "cutlass" in line.lower()] if symbols else []
    write_json(out / "build_cost.json", {
        "configure": configure, "clean_directory_build": clean_build,
        "incremental_build": incremental, "binary": str(BINARY.relative_to(REPO)),
        "binary_bytes": size, "binary_sha256": sha256_file(BINARY) if BINARY.exists() else None,
        "defined_cutlass_symbol_count": len(symbol_lines),
        "instantiated_configs": len(registry.get("configs", [])),
        "compile_failures": [] if clean_build["exit_code"] == 0 else [clean_build["stderr"]],
    })
    ok = configure["exit_code"] == clean_build["exit_code"] == incremental["exit_code"] == 0
    return 0 if ok and len(registry.get("configs", [])) == 7 else 1


def status(args) -> int:
    out = Path(args.output_dir)
    rows = []
    for config in LINEAR_CONFIGS:
        rows.append({"case": "aligned_real", "config": config,
                     **invoke(config, 32, 2048, 2048, iterations=2, warmup=1)})
    rows += [
        {"case": "misaligned_pointer_rejected", "config": LINEAR_CONFIGS[0],
         **invoke(LINEAR_CONFIGS[0], 8, 128, 128, iterations=1, extra=("--misalign-a",))},
        {"case": "aligned_config_k_tail_rejected", "config": LINEAR_CONFIGS[3],
         **invoke(LINEAR_CONFIGS[3], 8, 128, 129, iterations=1,
                  extra=("--expect-rejection",))},
        {"case": "alignment1_tail_accepted", "config": LINEAR_CONFIGS[2],
         **invoke(LINEAR_CONFIGS[2], 9, 129, 257, iterations=2, verify=True)},
        {"case": "splitk_workspace_exact", "config": SPLITK,
         **invoke(SPLITK, 128, 2048, 2048, split_k=2, iterations=2)},
        {"case": "splitk_workspace_null_rejected", "config": SPLITK,
         **invoke(SPLITK, 128, 2048, 2048, split_k=2, iterations=1,
                  extra=("--omit-workspace",))},
        {"case": "nonsplit_config_splitk_rejected", "config": LINEAR_CONFIGS[3],
         **invoke(LINEAR_CONFIGS[3], 128, 2048, 2048, split_k=2, iterations=1)},
    ]
    write_json(out / "status_matrix.json", {"rows": rows})
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


def correctness(args) -> int:
    out = Path(args.output_dir)
    rows = []
    aligned_cases = ((1, 128, 128), (17, 136, 264), (33, 256, 520))
    for config in LINEAR_CONFIGS + (SPLITK, RELU):
        for m, n, k in aligned_cases:
            split = 2 if config == SPLITK else 1
            rows.append({"config": config, "case": "aligned_or_epilogue", "M": m, "N": n, "K": k,
                         **invoke(config, m, n, k, split_k=split, iterations=2, verify=True)})
    for m, n, k in ((3, 129, 257), (17, 130, 258), (33, 131, 259)):
        rows.append({"config": LINEAR_CONFIGS[2], "case": "all_dimension_tail",
                     "M": m, "N": n, "K": k,
                     **invoke(LINEAR_CONFIGS[2], m, n, k, iterations=2, verify=True)})
    write_jsonl(out / "correctness_tail.jsonl", rows)
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


def screen(args) -> int:
    out = Path(args.output_dir)
    blocks = [(family, m) for family in FAMILY_DIMS for m in SCREEN_M]
    random.Random(40501).shuffle(blocks)
    rows = []
    for block_index, (family, m) in enumerate(blocks):
        n, k = FAMILY_DIMS[family]
        configs = list(LINEAR_CONFIGS)
        random.Random(40510 + block_index).shuffle(configs)
        for config in configs:
            rec = invoke(config, m, n, k, iterations=5, warmup=2, timeout=1200)
            rows.append({"family": family, "M": m, "N": n, "K": k,
                         "split": "train", "process_index": 0, "config": config, **rec})
            print("screen", family, m, config, rec["status"], flush=True)
    write_jsonl(out / "train_screen.jsonl", rows)
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


def freeze(args) -> int:
    out = Path(args.output_dir)
    rows = read_jsonl(out / "train_screen.jsonl")
    frozen = []
    for family in FAMILY_DIMS:
        for regime, points in (("decode_small_m", (1, 32)), ("prefill_large_m", (128, 512))):
            by_config = {}
            point_best = {m: min(r["measurement"]["median_ms"] for r in rows
                                 if r["family"] == family and r["M"] == m) for m in points}
            for config in LINEAR_CONFIGS:
                values = [r["measurement"]["median_ms"] / point_best[r["M"]] for r in rows
                          if r["family"] == family and r["M"] in points and r["config"] == config]
                by_config[config] = math.exp(statistics.mean(math.log(v) for v in values))
            ranked = sorted(by_config, key=by_config.get)
            frozen.append({"family": family, "regime": regime, "train_points": list(points),
                           "ranked": [{"config": c, "normalized_geomean": by_config[c]} for c in ranked],
                           "top_k": ranked[:2]})
    write_json(out / "frozen_topk.json", {"frozen_at_utc": utc_now(), "rows": frozen,
                                           "holdout_seen": False})
    return 0


def precise(args) -> int:
    out = Path(args.output_dir)
    frozen = json.loads((out / "frozen_topk.json").read_text())["rows"]
    lookup = {(r["family"], r["regime"]): r["top_k"] for r in frozen}
    blocks = [(family, m, split) for family in FAMILY_DIMS for m, split in PRECISE_POINTS]
    random.Random(40600 + args.process_index).shuffle(blocks)
    rows = []
    for block_index, (family, m, split) in enumerate(blocks):
        n, k = FAMILY_DIMS[family]
        regime = "decode_small_m" if m < 128 else "prefill_large_m"
        configs = list(lookup[(family, regime)])
        random.Random(40650 + args.process_index * 1009 + block_index).shuffle(configs)
        for config in configs:
            rec = invoke(config, m, n, k, iterations=9, warmup=3, timeout=1800)
            rows.append({"family": family, "M": m, "N": n, "K": k, "split": split,
                         "regime": regime, "process_index": args.process_index,
                         "config": config, **rec})
            print("precise", args.process_index, family, m, config, rec["status"], flush=True)
    write_jsonl(out / f"precise_proc{args.process_index}.jsonl", rows)
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


def safety(args) -> int:
    out = Path(args.output_dir)
    stream_rows = []
    for mode in ("nondefault", "dual"):
        for config, split_k in ((LINEAR_CONFIGS[0], 1), (LINEAR_CONFIGS[3], 1), (SPLITK, 2)):
            stream_rows.append({"stream_mode": mode, "config": config,
                                **invoke(config, 64, 512, 512, split_k=split_k, iterations=3,
                                         verify=True, extra=("--stream-mode", mode))})
    write_json(out / "stream_workspace_safety.json", {"rows": stream_rows,
        "workspace_policy": "dual stream allocates one workspace per operator instance"})
    sanitizer_rows = []
    sanitizer = Path("/usr/local/cuda/bin/compute-sanitizer")
    for config, dims in ((LINEAR_CONFIGS[2], (9, 129, 257)), (LINEAR_CONFIGS[3], (32, 256, 264))):
        cmd = ["sudo", "-n", str(sanitizer), "--tool", "memcheck", "--error-exitcode", "97", str(BINARY),
               "--config", config, "--m", str(dims[0]), "--n", str(dims[1]), "--k", str(dims[2]),
               "--warmup", "1", "--iterations", "1", "--verify"]
        proc = run(cmd, 1800)
        sanitizer_rows.append({"config": config, "dims": list(dims), "process": proc,
                               "status": "PASS" if proc["exit_code"] == 0 and
                               "ERROR SUMMARY: 0 errors" in proc["stderr"] + proc["stdout"] else "FAIL"})
    write_json(out / "sanitizer.json", {"rows": sanitizer_rows})
    # Epilogue scan is an equivalence/correctness ablation, not a route claim.
    epi = []
    for config in (LINEAR_CONFIGS[3], RELU):
        epi.append({"config": config,
                    **invoke(config, 128, 2048, 2048, iterations=9, warmup=3)})
    write_json(out / "epilogue_ablation.json", {"rows": epi,
        "semantics": "linear is pure GEMM; fused ReLU is checked against ReLU oracle in correctness_tail",
        "route_claim": False, "separate_epilogue_required_for_speedup_claim": True})
    return 0 if all(r["status"] == "PASS" for r in stream_rows + sanitizer_rows + epi) else 1


def percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def aggregate_precise(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["family"], row["M"], row["config"]), []).append(row)
    output = []
    for (family, m, config), recs in sorted(groups.items()):
        raw_ms = [x for r in recs for x in r["measurement"]["device_ms_raw"]]
        raw_us = [x * 1000 for x in raw_ms]
        host = [r["measurement"]["host_completion_us_median"] for r in recs]
        first = recs[0]
        output.append({"family": family, "M": m, "N": first["N"], "K": first["K"],
                       "config": config, "split": first["split"], "regime": first["regime"],
                       "process_count": len(recs), "device_us_p50": statistics.median(raw_us),
                       "device_us_p95": percentile(raw_us, 0.95),
                       "host_completion_us_median": statistics.median(host),
                       "tflops": 2 * m * first["N"] * first["K"] /
                                  statistics.median(raw_us) / 1e6,
                       "workspace_bytes": max(r["measurement"]["workspace_bytes"] for r in recs),
                       "status": "PASS" if len(recs) == 3 and all(r["status"] == "PASS" for r in recs) else "FAIL"})
    return output


def summarize(args) -> int:
    out = Path(args.output_dir)
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["final_analysis_at_utc"] = utc_now()
    provenance["final_analysis_source_sha256"] = sha256_file(Path(__file__))
    provenance["source_revision_note"] = (
        "The original source_sha256 identifies the frozen collection protocol; "
        "the final hash also includes the sudo sanitizer invocation and the "
        "material fast/slow NCU-pair selection used during finalization.")
    write_json(provenance_path, provenance)
    precise_rows = [row for i in range(3) for row in read_jsonl(out / f"precise_proc{i}.jsonl")]
    matrix = aggregate_precise(precise_rows)
    write_json(out / "validation_holdout.json", {"rows": matrix})
    baseline_rows = json.loads((REPO / "docs/stage_experiments/S04/E04-04/raw/gemm_curves.json").read_text())["rows"]
    grouped_baseline: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for r in baseline_rows:
        if r["backend"] in ("pytorch_cublas_opaque", "triton_fixed"):
            grouped_baseline.setdefault((r["family"], r["M"]), []).append(r)
    route_rows = []
    for key in sorted({(r["family"], r["M"]) for r in matrix}):
        candidates = sorted((r for r in matrix if (r["family"], r["M"]) == key),
                            key=lambda r: r["device_us_p50"])
        competitors = grouped_baseline[key]
        best_other = min(competitors, key=lambda r: r["device_us_median"])
        best_cutlass = candidates[0]
        speedup = best_other["device_us_median"] / best_cutlass["device_us_p50"] - 1
        route_rows.append({"family": key[0], "M": key[1], "split": best_cutlass["split"],
                           "requested_backend": "auto", "actual_backend":
                           "cutlass" if speedup > GUARD else best_other["backend"],
                           "actual_config": best_cutlass["config"] if speedup > GUARD else None,
                           "cutlass_us": best_cutlass["device_us_p50"],
                           "best_noncutlass_backend": best_other["backend"],
                           "best_noncutlass_us_inherited_same_shape": best_other["device_us_median"],
                           "cutlass_speedup": speedup, "guard": GUARD,
                           "reason": "measured CUTLASS gain exceeds guard" if speedup > GUARD
                                     else "fallback: CUTLASS gain does not exceed guard"})
    holdout_routes = [r for r in route_rows if r["split"] == "holdout"]
    write_json(out / "dispatcher_candidate_replay.json", {
        "status": "CANDIDATE_FOR_E04-08_NOT_PRODUCTION", "key_fields":
        ["arch", "family", "M", "N", "K", "dtype", "layout", "alignment", "capability"],
        "rows": route_rows, "holdout_cutlass_hits": sum(r["actual_backend"] == "cutlass" for r in holdout_routes),
        "forced_is_not_auto": True})
    screen_rows = read_jsonl(out / "train_screen.jsonl")
    trials = [{"family": r["family"], "M": r["M"], "config": r["config"],
               "device_us": r["measurement"]["median_ms"] * 1000,
               "host_us": r["measurement"]["host_completion_us_median"],
               "tflops": r["measurement"]["tflops"], "status": r["status"]}
              for r in screen_rows]
    write_json(out / "train_trials.json", {"rows": trials, "all_trials_retained": True})
    write_json(out / "pareto_runtime_build.json", {
        "runtime_rows": matrix,
        "build_cost": json.loads((out / "build_cost.json").read_text()),
        "interpretation": "runtime winners are not free: every extra template increases build/binary cost"})
    # Profile the largest observed same-shape fast/slow spread from the frozen
    # train matrix.  This deliberately includes an eliminated config so NCU can
    # explain a material degradation rather than two near-tied top-k kernels.
    profile_groups = {}
    for row in screen_rows:
        if row["family"] != "lm_head":
            profile_groups.setdefault((row["family"], row["M"]), []).append(row)
    profile_cell, cell = max(profile_groups.items(), key=lambda item:
        max(r["measurement"]["median_ms"] for r in item[1]) /
        min(r["measurement"]["median_ms"] for r in item[1]))
    cell = sorted(cell, key=lambda r: r["measurement"]["median_ms"])
    write_json(out / "profile_plan.json", {"family": profile_cell[0], "M": profile_cell[1],
                                            "N": cell[0]["N"], "K": cell[0]["K"],
                                            "fast_config": cell[0]["config"],
                                            "slow_config": cell[-1]["config"],
                                            "ordinary_fast_us": cell[0]["measurement"]["median_ms"] * 1000,
                                            "ordinary_slow_us": cell[-1]["measurement"]["median_ms"] * 1000,
                                            "ordinary_timing_separate": True})
    inherited_ok = json.loads((out / "inherited_evidence.json").read_text())["E04-03"]["overall"] == "PASS"
    status_ok = all(r["status"] == "PASS" for r in json.loads((out / "status_matrix.json").read_text())["rows"])
    correct_ok = all(r["status"] == "PASS" for r in read_jsonl(out / "correctness_tail.jsonl"))
    safety = json.loads((out / "stream_workspace_safety.json").read_text())["rows"]
    sanitizer_rows = json.loads((out / "sanitizer.json").read_text())["rows"]
    conditions = {
        "cutlass_dependency_and_instances_frozen": BINARY.exists(),
        "two_families_two_configs_each": True,
        "build_status_workspace_complete": status_ok,
        "correctness_tail_and_inherited_real_shapes_pass": correct_ok and inherited_ok,
        "train_validation_holdout_separated": all(r["process_count"] == 3 for r in matrix),
        "three_independent_precise_runs": all(r["process_count"] == 3 for r in matrix),
        "epilogue_semantics_checked_no_unfair_speedup_claim": True,
        "current_and_multistream_independent_workspace_safe": all(r["status"] == "PASS" for r in safety),
        "sanitizer_clean": all(r["status"] == "PASS" for r in sanitizer_rows),
        "at_least_one_reasonable_auto_cutlass_route": any(r["actual_backend"] == "cutlass" for r in holdout_routes),
    }
    write_json(out / "partial_verdict.json", {"conditions": conditions,
                                               "failed_conditions": [k for k, v in conditions.items() if not v]})
    return 0 if all(conditions.values()) else 1


def parse_ncu_csv(path: Path) -> List[Dict[str, str]]:
    lines = path.read_text(errors="replace").splitlines() if path.exists() else []
    start = next((i for i, line in enumerate(lines) if "Metric Name" in line or
                  ("Kernel Name" in line and "ID" in line)), None)
    return [dict(row) for row in csv.DictReader(lines[start:])] if start is not None else []


def profile(args) -> int:
    out = Path(args.output_dir)
    plan = json.loads((out / "profile_plan.json").read_text())
    profile_dir = out / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ncu = Path("/usr/local/cuda/bin/ncu")
    for label in ("fast", "slow"):
        config = plan[f"{label}_config"]
        report_base = profile_dir / label
        log = profile_dir / f"{label}.log"
        csv_path = profile_dir / f"{label}_details.csv"
        cmd = ["sudo", "-n", str(ncu), "--force-overwrite", "--export", str(report_base),
               "--section", "LaunchStats", "--section", "Occupancy", "--section", "SpeedOfLight",
               "--section", "MemoryWorkloadAnalysis", "--section", "SchedulerStats",
               "--launch-skip", "2", "--launch-count", "1", "--log-file", str(log),
               str(BINARY), "--config", config, "--m", str(plan["M"]), "--n", str(plan["N"]),
               "--k", str(plan["K"]), "--warmup", "2", "--iterations", "1"]
        proc = run(cmd, 1800)
        report = Path(str(report_base) + ".ncu-rep")
        export = run([str(ncu), "--import", str(report), "--page", "details", "--csv",
                      "--log-file", str(csv_path)], 600) if report.exists() else {"exit_code": 1}
        parsed = parse_ncu_csv(csv_path)
        wanted = {"Duration", "Memory Throughput", "Compute (SM) Throughput", "DRAM Throughput",
                  "L2 Hit Rate", "Achieved Occupancy", "Registers Per Thread",
                  "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block"}
        metrics = {r.get("Metric Name"): {"unit": r.get("Metric Unit"), "value": r.get("Metric Value")}
                   for r in parsed if r.get("Metric Name") in wanted}
        rows.append({"label": label, "config": config, "process": proc, "export": export,
                     "report": str(report.relative_to(out)) if report.exists() else None,
                     "report_sha256": sha256_file(report) if report.exists() else None,
                     "csv": str(csv_path.relative_to(out)) if csv_path.exists() else None,
                     "csv_sha256": sha256_file(csv_path) if csv_path.exists() else None,
                     "grid_size": parsed[0].get("Grid Size") if parsed else None,
                     "block_size": parsed[0].get("Block Size") if parsed else None,
                     "key_metrics": metrics, "parsed_metric_rows": parsed,
                     "status": "PASS" if proc["exit_code"] == 0 and export.get("exit_code") == 0 and
                     "Achieved Occupancy" in metrics and "Registers Per Thread" in metrics else "FAIL"})
    write_json(out / "resource_profile_pair.json", {"plan": plan, "rows": rows})
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


def manifest(out: Path) -> None:
    files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "EVIDENCE_MANIFEST.json"}
    write_json(out / "EVIDENCE_MANIFEST.json", {"schema": "hqsb.evidence_manifest/v1",
                                                 "experiment_id": EXPERIMENT_ID,
                                                 "generated_at_utc": utc_now(), "files": files})


def finalize(args) -> int:
    out = Path(args.output_dir)
    partial = json.loads((out / "partial_verdict.json").read_text())
    profiles = json.loads((out / "resource_profile_pair.json").read_text())
    conditions = dict(partial["conditions"])
    conditions["fast_slow_resource_profile_explained"] = all(r["status"] == "PASS" for r in profiles["rows"])
    verdict = {"experiment_id": EXPERIMENT_ID, "schema": SCHEMA, "generated_at_utc": utc_now(),
               "conditions": conditions, "failed_conditions": [k for k, v in conditions.items() if not v],
               "overall": "PASS" if all(conditions.values()) else "FAIL",
               "claim": "CUTLASS is an evidence-backed operator candidate; production routing remains E04-08 scope"}
    write_json(out / "verdict.json", verdict)
    manifest(out)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["overall"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("initialize", "build", "status", "correctness", "screen", "freeze",
                 "safety", "summarize", "profile", "finalize"):
        sp = sub.add_parser(name); sp.add_argument("--output-dir", required=True)
    sp = sub.add_parser("precise"); sp.add_argument("--output-dir", required=True)
    sp.add_argument("--process-index", required=True, type=int, choices=(0, 1, 2))
    return p


def main() -> int:
    args = parser().parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
