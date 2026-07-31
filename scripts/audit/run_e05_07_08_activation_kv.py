#!/usr/bin/env python3
"""Remote-only S05 P1 activation/KV capability, contract and micro evidence.

No model candidate is selected here. Synthetic tensors and saved calibration
statistics are labelled explicitly; missing model evidence remains NOT_EXECUTED.
Run through scripts/remote_run.sh, in cpu/gpu/finalize/verify phases.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "docs/stage_experiments/S05"
SEED = 50708


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def rows_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False) + "\n" for row in rows))


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def shell_output(argv):
    return subprocess.check_output(argv, cwd=ROOT, text=True).strip()


def metric(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    delta = b - a
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {"max_abs": float(np.max(np.abs(delta))) if a.size else 0.,
            "rmse": float(np.sqrt(np.mean(delta ** 2))) if a.size else 0.,
            "relative_l2": float(np.linalg.norm(delta) / max(na, 1e-30)),
            "cosine": float(np.dot(a, b) / (na * nb)) if na * nb else (1. if np.array_equal(a, b) else 0.)}


def prereq():
    result = []
    for exp in ("E05-01", "E05-02", "E05-03", "E05-04"):
        path = BASE / exp / "raw/verdict.json"
        value = read(path) if path.exists() else {}
        result.append({"id": exp, "path": str(path.relative_to(ROOT)),
                       "sha256": digest(path) if path.exists() else None,
                       "overall": value.get("overall", "MISSING")})
    handoff = BASE / "E05-03/raw/selection/e05_04_handoff.json"
    return {"captured_at_utc": now(), "upstream": result,
            "calibration_handoff_exists": handoff.is_file(),
            "calibration_handoff_path": str(handoff.relative_to(ROOT)),
            "activation_model_runtime_integrated": False,
            "quantized_kv_model_runtime_integrated": False,
            "fp16_cache_runtime_exists": True,
            "source_inspected": ["hqsb/quant/activation.py", "hqsb/quant/kv.py",
                                 "hqsb/benchmark/model_core.py", "hqsb/benchmark/shape_census.py"],
            "reason": "FP16 cache-enabled model paths exist; activation INT8 and packed low-bit KV model adapters are absent. No frozen calibration candidate is available."}


def specs():
    common = {"seed": SEED, "frozen_at_utc": now(),
              "scope": "synthetic contract and target-device microbenchmark; no model quality or deployment claim",
              "final_evaluation_consumed": False,
              "warmup": 5, "repeats": 5, "inner_iterations": 10,
              "timing": "CUDA events per operation; host launch-to-synchronize wall time measured separately; profiler excluded from latency",
              "model_candidate_selection": "NOT_EXECUTED",
              "energy": "NOT_MEASURED for short microbenchmarks; no energy/token claim",
              "source_sha256": digest(__file__)}
    a = dict(common, experiment_id="E05-07", bits=8, qrange=[-127, 127],
             rounding="nearest_even", scale_dtype="float32", accumulator_dtype="int32",
             overflow_policy="reject K when K*127*127 exceeds int32 max; verify exact integer oracle",
             nan_inf="reject", mask_policy="caller removes masked rows before reduction",
             clip=None, smoothquant_alpha=None,
             calibration_policy="all saved calibration sample absmax merged only for a diagnostic per-tensor scale; not selected for deployment",
             static_runtime_state="separate synthetic calibration tensor fixed before timing",
             granularities={"per_tensor": "all M,K", "per_token": "K reduction for each row",
                            "per_channel": "M reduction for each input K channel; unsupported by single int32 epilogue",
                            "per_group": "groups of 32 along K per row; unsupported by single int32 epilogue"},
             micro_shapes=[{"m": m, "n": 2048, "k": 2048, "phase": "decode" if m == 1 else "prefill"}
                           for m in (1, 16, 128)],
             matrix=["fp16", "w8a16_dequant_fp16", "fake_w8a8", "static_w8a8", "dynamic_w8a8"],
             scale_utilization_definition="fraction abs(q) >= qmax/2; boundary occupancy is separately named",
             oracle_max_abs_tolerance=0.0)
    k = dict(common, experiment_id="E05-08", k_v_bits=[[8, 8], [4, 4], [8, 4]],
             granularity="per_token_per_head", optional_group=32, scale_dtype="float32",
             qrange={"8": [-127, 127], "4": [-7, 7]}, zero_point=0,
             rope_point="synthetic post-RoPE-labelled tensor; no actual model RoPE executed",
             layout="synthetic [layers, kv_heads, tokens, head_dim], dense payload; INT4 low nibble first signed two's complement",
             page_size=16, synthetic_page_header_bytes=8, synthetic_page_alignment_bytes=32,
             residual_windows=[0, 32], context_lengths=[1, 15, 16, 17, 128, 512, 2048, 8192],
             gpu_context_lengths=[17, 128, 512, 2048],
             gpu_layers=2, kv_heads=8, head_dim=128,
             tensor_storage_tolerance_bytes=0,
             allocator_tolerance="rounded active allocation >= exact tensor bytes; report residual, never hide it",
             max_context="NOT_EXECUTED; no low-bit model runtime and no model resident-state capacity protocol",
             generation_lengths=[32, 128, 512], generation_sweep_status="NOT_EXECUTED")
    return a, k


def activation_contract(out):
    import numpy as np
    from hqsb.quant.activation import ActivationQuantSpec, quantize_activation, w8a8_reference, compute_smooth_scale
    fixtures = {
        "zero": np.zeros((2, 7)), "constant": np.full((2, 7), 3.),
        "signed_extrema": np.array([[-127, 127, -1, 0, 1, -64, 64]] * 2),
        "all_positive": np.arange(1, 15).reshape(2, 7),
        "all_negative": -np.arange(1, 15).reshape(2, 7),
        "outlier_tail": np.array([[.1, -.1, .2, -.2, .5, -.5, 128], [1, 2, 3, 4, 5, 6, 7]])}
    rows = []
    for gran in ("per_tensor", "per_token", "per_channel", "per_group"):
        spec = ActivationQuantSpec(granularity=gran, group_size=4 if gran == "per_group" else None)
        for name, x in fixtures.items():
            q = quantize_activation(x.ravel().tolist(), spec, shape=x.shape)
            expected_count = {"per_tensor": 1, "per_token": 2, "per_channel": 7, "per_group": 4}[gran]
            rows.append({"fixture": name, "granularity": gran, "shape": list(x.shape),
                         "scale_count": len(q.scales), "expected_scale_count": expected_count,
                         "q": q.q, "scales": q.scales, "error": metric(x, q.dequant),
                         "boundary_occupancy": q.saturation_rate,
                         "reported_scale_utilization": q.scale_utilization,
                         "independent_half_range_utilization": sum(abs(v) >= 63.5 for v in q.q)/len(q.q),
                         "passed": len(q.scales) == expected_count and all(-127 <= v <= 127 for v in q.q),
                         "scope": "pure-Python quantizer synthetic contract"})
    rows_write(out / "correctness/granularity.jsonl", rows)
    negative = []
    for name, values, mode, scales in (
        ("nan_dynamic", [0., float("nan")], "dynamic", None),
        ("inf_dynamic", [0., float("inf")], "dynamic", None),
        ("static_missing_scale", [0., 1.], "static", None),
        ("dynamic_reject_static_scale", [0., 1.], "dynamic", [1.]),
        ("static_zero_scale", [0., 1.], "static", [0.])):
        try:
            quantize_activation(values, ActivationQuantSpec(mode=mode, granularity="per_tensor"),
                                shape=(1, 2), static_scales=scales)
            negative.append({"fixture": name, "rejected": False})
        except Exception as exc:
            negative.append({"fixture": name, "rejected": True,
                             "exception": type(exc).__name__, "message": str(exc)})
    tie = quantize_activation([-.5, .5, 1.5, 2.5], ActivationQuantSpec(mode="static", granularity="per_tensor"),
                              shape=(1, 4), static_scales=[1.])
    negative.append({"fixture": "nearest_even_ties", "expected": [0, 0, 2, 2], "actual": tie.q,
                     "passed": tie.q == [0, 0, 2, 2]})
    # Tensor-only contracts cannot validate masks/strides; no implicit support claim.
    negative.extend([{"fixture": n, "status": "NOT_SUPPORTED_BY_LIST_API"}
                     for n in ("noncontiguous_strides", "empty_masked_rows")])
    write(out / "correctness/boundaries.json", negative)
    xq = [1, 2, 3, 4]; wq = [2, 1, 1, 2]
    actual = w8a8_reference(xq, wq, x_scales=[.5, 2.], w_scales=[.25, 3.], m=2, n=2, k=2)
    expected = (np.array(xq).reshape(2, 2) @ np.array(wq).reshape(2, 2)) * np.array([.5, 2.])[:, None] * np.array([.25, 3.])[None, :]
    mismatch = metric(expected, actual)
    write(out / "correctness/reference_broadcast_audit.json", {
        "scope": "existing w8a8_reference default API audit; not the GPU oracle used in this runner",
        "x_scales": [.5, 2.], "w_scales": [.25, 3.], "actual": actual,
        "independent_expected": expected.tolist(), "error": mismatch,
        "passed": mismatch["max_abs"] == 0,
        "finding": ("default x_units_per_row/w_units_per_row=1 broadcasts scale[0]; it cannot represent per-token/per-output-channel scale vectors at those defaults"
                    if mismatch["max_abs"] else "per-token/per-output-channel scale vectors match independent broadcast oracle")})
    rng = np.random.default_rng(SEED)
    x, w = rng.normal(size=(7, 17)), rng.normal(size=(13, 17))
    smooth = compute_smooth_scale(np.max(abs(x), axis=0).tolist(), np.max(abs(w), axis=0).tolist(), alpha=.5)
    xs = np.asarray(smooth.apply_to_activation(x.ravel(), cols=17)).reshape(x.shape)
    ws = np.asarray(smooth.apply_to_weight(w.ravel(), cols=17)).reshape(w.shape)
    e = metric(x @ w.T, xs @ ws.T)
    write(out / "smooth_transforms/primitive_equivalence.json", {
        "scope": "synthetic float64 algebra only; no model graph/norm/residual fold",
        "alpha": .5, "error": e, "tolerance_max_abs": 1e-12, "passed": e["max_abs"] <= 1e-12})
    drift = []
    calibration = rng.normal(size=(16, 128))
    fixed_scale = max(abs(calibration).ravel())/127
    for domain, factor in (("same_distribution", 1.), ("scale_shift_x4", 4.), ("outlier_heavy", 1.)):
        x = rng.normal(size=(16, 128)) * factor
        if domain == "outlier_heavy":
            x[:, 0] *= 64
        for mode in ("static", "dynamic"):
            spec = ActivationQuantSpec(mode=mode, granularity="per_tensor" if mode == "static" else "per_token")
            q = quantize_activation(x.ravel(), spec, shape=x.shape, static_scales=[fixed_scale] if mode == "static" else None)
            drift.append({"scope": "synthetic drift; not workload or model quality", "domain": domain,
                          "mode": mode, "granularity": spec.granularity, "error": metric(x, q.dequant),
                          "boundary_occupancy": q.saturation_rate,
                          "true_clipping_rate": float(np.mean(abs(x) > fixed_scale * 127)) if mode == "static" else 0.})
    rows_write(out / "drift/synthetic.jsonl", drift)


def saved_activation_stats(out):
    path = BASE / "E05-03/raw/stats/activation/index.jsonl"
    grouped, calibration, split_counts = {}, {}, {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        split = row["sample_id"].split(":")[0]
        if split == "final-evaluation":
            continue
        split_counts[split] = split_counts.get(split, 0) + 1
        for module in row["modules"]:
            key = (split, row["domain"], row["length_bucket"], module["module"])
            grouped.setdefault(key, []).append(module)
            if split == "calibration":
                calibration[module["module"]] = max(calibration.get(module["module"], 0), module["absmax"])
    summary = []
    for (split, domain, bucket, module), records in sorted(grouped.items()):
        summary.append({"scope": "reused real FP16 module statistics from E05-03; not a new W8A8 model evaluation",
                        "split": split, "domain": domain, "length_bucket": bucket, "module": module,
                        "sample_count": len(records), "valid_tokens": sum(r["valid_tokens"] for r in records),
                        "absmax_max": max(r["absmax"] for r in records),
                        "mean_p99_abs": statistics.mean(r["p99_abs"] for r in records),
                        "mean_outlier_rate": statistics.mean(r["outlier_rate"] for r in records),
                        "position": "full sample aggregate; per-position raw activations unavailable"})
    rows_write(out / "activation_stats/reused_fp16_summary.jsonl", summary)
    write(out / "static_scales/diagnostic.json", {
        "status": "DIAGNOSTIC_NOT_SELECTED", "scale_granularity": "per_tensor_per_module",
        "source": str(path.relative_to(ROOT)), "source_sha256": digest(path),
        "split_counts": split_counts, "used_for_scale": "calibration only",
        "scales": {name: value / 127 for name, value in calibration.items()},
        "count": len(calibration), "calibration_handoff_exists": prereq()["calibration_handoff_exists"],
        "not_for_deployment": "No chosen E05-03 calibration candidate or activation model runtime."})


def numpy_pack_quant(x, bits, group=128):
    import numpy as np
    x = np.asarray(x, np.float32)
    d = x.shape[-1]; groups = math.ceil(d/group); pad = groups*group-d
    xp = np.pad(x, [(0, 0)] * (x.ndim-1) + [(0, pad)]).reshape(*x.shape[:-1], groups, group)
    limit = (1 << (bits-1))-1
    s = np.maximum(np.max(abs(xp), axis=-1, keepdims=True), np.float32(1e-12))/limit
    q = np.clip(np.rint(xp/s), -limit, limit).astype(np.int8)
    deq = (q.astype(np.float32)*s).reshape(*x.shape[:-1], groups*group)[..., :d]
    codes = q.ravel()
    if bits == 4:
        if codes.size % 2:
            codes = np.pad(codes, (0, 1))
        packed = ((codes[::2] & 15).astype(np.uint8) | ((codes[1::2] & 15).astype(np.uint8) << 4))
        uq = np.empty(codes.size, np.int8)
        lo, hi = (packed & 15).astype(np.int8), (packed >> 4).astype(np.int8)
        uq[::2], uq[1::2] = np.where(lo >= 8, lo-16, lo), np.where(hi >= 8, hi-16, hi)
    else:
        packed, uq = codes.copy(), codes.copy()
    return packed, s, deq, bool(np.array_equal(codes, uq)), pad


def kv_cpu(out):
    import numpy as np
    from hqsb.quant.kv import KvQuantSpec, kv_capacity, fp16_baseline_bytes, attention_comparison_report
    model = read(ROOT / "docs/benchmark/qwen3_model_manifest.json")
    write(out / "cache_schema.json", {
        "schema_scope": "model config plus historical FP16 cache-enabled evidence; low-bit schema below is micro-only",
        "layers": model["num_hidden_layers"], "kv_heads": model["num_key_value_heads"], "head_dim": model["head_dim"],
        "model_config_source": "docs/benchmark/qwen3_model_manifest.json", "model_config_sha256": digest(ROOT / "docs/benchmark/qwen3_model_manifest.json"),
        "fp16_runtime_source": "hqsb/benchmark/model_core.py (use_cache=True)",
        "historical_capacity_evidence": "docs/stage_experiments/S02/E02-05/raw/run_0.json",
        "historical_capacity_sha256": digest(ROOT / "docs/stage_experiments/S02/E02-05/raw/run_0.json"),
        "low_bit_model_runtime": "NOT_IMPLEMENTED", "model_rope_point": "NOT_RECAPTURED",
        "micro_layout": "[L,H,T,D] packed symmetric per-token-per-head; explicit synthetic headers and alignment buffers",
        "micro_rope_point": "post-RoPE-labelled synthetic data; no model RoPE claim",
        "no_persistent_fp16_shadow_in_packed_state": True,
        "temporary_dequant_buffer": "full readback FP16 buffer, counted separately; diagnostic attention path"})
    theory, exact = [], []
    for kb, vb in ((8, 8), (4, 4), (8, 4)):
        for residual in (0, 32):
            spec = KvQuantSpec(k_bits=kb, v_bits=vb, residual_window=residual, label=f"k{kb}v{vb}_r{residual}")
            for t in (1, 15, 16, 17, 128, 512, 2048, 8192):
                cap = kv_capacity(spec, layers=28, kv_heads=8, head_dim=128, tokens=t, dtype_bytes=2)
                base = fp16_baseline_bytes(layers=28, kv_heads=8, head_dim=128, tokens=t)
                theory.append({"spec": spec.as_dict(), "scope": "model-dimension capacity prediction only",
                               **cap.as_dict(), "fp16_payload_bytes": base,
                               "ratio_vs_fp16_payload": base/cap.total_bytes,
                               "residual_dtype_bytes": 2,
                               "model_runtime_measured": False})
            for t in (17, 128):
                r = min(t, residual); qtokens = t-r
                # Allocate real independent CPU tensors matching the declared micro layout.
                payloads = [np.empty((28, 8, qtokens, 128*bits//8), np.uint8) for bits in (kb, vb)]
                scales = [np.empty((28, 8, qtokens), np.float32) for _ in (0, 1)]
                pages = math.ceil(t/16)*28*2
                header, align = np.empty((pages, 8), np.uint8), np.empty((pages, 32), np.uint8)
                residual_array = np.empty((2, 28, 8, r, 128), np.float16)
                measured = sum(a.nbytes for a in payloads+scales+[header, align, residual_array])
                cap = kv_capacity(spec, layers=28, kv_heads=8, head_dim=128, tokens=t, dtype_bytes=2)
                exact.append({"label": spec.label, "tokens": t, "scope": "real CPU tensor storage for synthetic layout, not model allocator",
                              "scale_count": sum(a.size for a in scales), "zero_count": 0,
                              "page_count": pages, "residual_dtype": "float16",
                              **cap.reconcile(measured, tolerance_bytes=0)})
    rows_write(out / "capacity/theoretical.jsonl", theory)
    rows_write(out / "capacity/cpu_tensor_storage.jsonl", exact)
    rng = np.random.default_rng(SEED)
    rep, att = [], []
    for d in (127, 128, 129):
        q, k, v = rng.normal(size=(3, d)), rng.normal(size=(17, d)), rng.normal(size=(17, d))
        k[0, 0] *= 16; v[-1, -1] *= 8
        for kb, vb in ((8, 8), (4, 4), (8, 4)):
            for group in (32, d):
                kp, ks, kd, okk, padk = numpy_pack_quant(k, kb, group)
                vp, vs, vd, okv, padv = numpy_pack_quant(v, vb, group)
                rep.append({"scope": "synthetic KV tensor", "shape": [17, d], "k_bits": kb, "v_bits": vb,
                            "group_size": group, "k_error": metric(k, kd), "v_error": metric(v, vd),
                            "payload_bytes": kp.nbytes+vp.nbytes, "scale_bytes": ks.nbytes+vs.nbytes,
                            "scale_count": ks.size+vs.size, "tail_padding_values_per_row": padk,
                            "pack_unpack_exact": okk and okv, "zero_point": 0,
                            "k_scale_min": float(ks.min()), "k_scale_max": float(ks.max()),
                            "v_scale_min": float(vs.min()), "v_scale_max": float(vs.max())})
                quality = attention_comparison_report(q.tolist(), k.tolist(), v.tolist(), kd.tolist(), vd.tolist())
                att.append({"scope": "synthetic fake-KV attention oracle; not model quality",
                            "head_dim": d, "context": 17, "k_bits": kb, "v_bits": vb,
                            "group_size": group, **quality})
    rows_write(out / "correctness/representation.jsonl", rep)
    rows_write(out / "correctness/attention.jsonl", clean(att))
    # Record the existing helper's default FP32 residual contract explicitly.
    s = KvQuantSpec(k_bits=4, v_bits=4, residual_window=32)
    a = kv_capacity(s, layers=28, kv_heads=8, head_dim=128, tokens=128)
    b = kv_capacity(s, layers=28, kv_heads=8, head_dim=128, tokens=128, dtype_bytes=2)
    write(out / "capacity/residual_dtype_audit.json", {
        "scope": "helper API compatibility audit", "default_dtype_bytes": 4,
        "declared_fp16_dtype_bytes": 2, "default_total_bytes": a.total_bytes,
        "explicit_fp16_total_bytes": b.total_bytes,
        "difference_bytes": a.total_bytes-b.total_bytes,
        "finding": "kv_capacity defaults residual storage to float32; all FP16 calculations in this runner pass dtype_bytes=2 explicitly. capacity_ratio has no dtype_bytes argument."})


def cpu():
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("This collector must run on Jetson via scripts/remote_run.sh")
    a, k = specs()
    env = {"captured_at_utc": now(), "platform": platform.platform(), "machine": platform.machine(),
           "python": platform.python_version(), "hostname": platform.node(),
           "git_commit": shell_output(["git", "rev-parse", "HEAD"]),
           "git_dirty": bool(shell_output(["git", "status", "--porcelain"])),
           "runner_sha256": digest(__file__), "execution": "Jetson remote CPU contract phase"}
    for exp, spec in (("E05-07", a), ("E05-08", k)):
        out = BASE/exp/"raw"
        write(out/"spec.json", spec)
        write(out/"environment_cpu.json", env)
        write(out/"prerequisites.json", prereq())
    activation_contract(BASE/"E05-07/raw")
    saved_activation_stats(BASE/"E05-07/raw")
    kv_cpu(BASE/"E05-08/raw")
    print(json.dumps({"phase": "cpu", "completed": True, "at": now()}), flush=True)


def event_bench(fn, torch):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    events, walls = [], []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        start.record()
        for _ in range(10):
            fn()
        end.record(); end.synchronize()
        walls.append((time.perf_counter()-t0)*1000/10)
        events.append(start.elapsed_time(end)/10)
    return {"cuda_event_ms": events, "median_ms": statistics.median(events),
            "host_wall_ms": walls, "host_median_ms": statistics.median(walls),
            "repeats": 5, "inner_iterations": 10}


def profile(fn, path, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace = read(path)
    names = sorted({r["name"] for r in trace.get("traceEvents", []) if r.get("cat") == "kernel"})
    return {"trace": str(path.relative_to(ROOT)), "observed_cuda_symbols": names,
            "cuda_kernel_count": sum(r.get("cat") == "kernel" for r in trace.get("traceEvents", [])),
            "provider": "PyTorch NVIDIA wheel / ATen / CUDA", "fallback": "none inside profiled function"}


def activation_gpu(out, torch):
    import numpy as np
    rng = np.random.default_rng(SEED)
    capability = {"api": "torch._int_mm", "available": hasattr(torch, "_int_mm"),
                  "accumulator_dtype": "int32", "kernel_oracle": [], "scope": "actual CUDA integer GEMM microkernel"}
    timings = []
    for m in (1, 16, 128):
        n, k = 2048, 2048
        x = torch.randn((m, k), device="cuda", dtype=torch.float16)
        w = torch.randn((n, k), device="cuda", dtype=torch.float16)
        bias = torch.zeros(n, device="cuda", dtype=torch.float32)
        sx = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)/127
        sw = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)/127
        calibration = torch.randn((m, k), device="cuda", dtype=torch.float16)
        static_sx = calibration.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)/127
        qx = torch.round(x.float()/sx).clamp(-127, 127).to(torch.int8)
        qw = torch.round(w.float()/sw).clamp(-127, 127).to(torch.int8)
        wt = qw.t()  # _int_mm expects column-major B on this provider.
        granularity = []
        for gran in ("per_tensor", "per_token", "per_channel", "per_group"):
            xf = x.float()
            def gran_fn(gran=gran):
                if gran == "per_tensor":
                    scale = xf.abs().amax().clamp_min(1e-12)/127
                    return torch.round(xf/scale).clamp(-127, 127).to(torch.int8)
                if gran == "per_token":
                    scale = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12)/127
                    return torch.round(xf/scale).clamp(-127, 127).to(torch.int8)
                if gran == "per_channel":
                    scale = xf.abs().amax(0, keepdim=True).clamp_min(1e-12)/127
                    return torch.round(xf/scale).clamp(-127, 127).to(torch.int8)
                xx = xf.reshape(m, k//32, 32)
                scale = xx.abs().amax(-1, keepdim=True).clamp_min(1e-12)/127
                return torch.round(xx/scale).clamp(-127, 127).to(torch.int8)
            granularity.append({"granularity": gran,
                                "integer_gemm_supported_in_this_runner": gran in ("per_tensor", "per_token"),
                                "time": event_bench(gran_fn, torch)})
        row = {"scope": "synthetic CUDA microbench; no model TTFT/TPOT", "m": m, "n": n, "k": k,
               "phase": "decode" if m == 1 else "prefill", "granularity_cost": granularity}
        try:
            acc = torch._int_mm(qx, wt)
            torch.cuda.synchronize()
            # FP64 CPU dot is an independent exact integer oracle for this bounded K.
            oracle = qx.cpu().numpy().astype(np.int64) @ qw.cpu().numpy().astype(np.int64).T
            actual = acc.cpu().numpy()
            check = {"m": m, "n": n, "k": k, "success": True,
                     "max_int_abs_difference": int(abs(oracle-actual).max()),
                     "accumulator_dtype": str(acc.dtype), "b_stride": list(wt.stride()),
                     "overflow_upper_bound": k*127*127, "overflow_safe": k*127*127 <= 2**31-1}
            capability["kernel_oracle"].append(check)
            xf = x.float(); ranges = xf.abs().amax(dim=1, keepdim=True)
            def dynamic_total():
                ss = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12)/127
                qq = torch.round(x.float()/ss).clamp(-127, 127).to(torch.int8)
                aa = torch._int_mm(qq, wt)
                return (aa.float()*ss*sw.t()+bias).half()
            def static_total():
                qq = torch.round(x.float()/static_sx).clamp(-127, 127).to(torch.int8)
                return (torch._int_mm(qq, wt).float()*static_sx*sw.t()+bias).half()
            wdeq = (qw.float()*sw).half()
            xdeq = (qx.float()*sx).half()
            fns = {"range_reduction_fp32": lambda: xf.abs().amax(1, keepdim=True),
                   "scale_compute": lambda: ranges.clamp_min(1e-12)/127,
                   "activation_quantize": lambda: torch.round(x.float()/sx).clamp(-127, 127).to(torch.int8),
                   "integer_gemm": lambda: torch._int_mm(qx, wt),
                   "output_dequant_scale": lambda: acc.float()*sx*sw.t(),
                   "epilogue_bias_cast": lambda: (acc.float()+bias).half(),
                   "dynamic_total": dynamic_total, "static_total": static_total,
                   "fp16_gemm": lambda: x @ w.t(),
                   "w8a16_dequant_fp16_total": lambda: x @ (qw.float()*sw).half().t(),
                   "fake_w8a8_total": lambda: (torch.round(x.float()/sx).clamp(-127, 127)*sx).half() @ (qw.float()*sw).half().t()}
            row["cost"] = {name: event_bench(fn, torch) for name, fn in fns.items()}
            row["launch_sync"] = {"measured_separately": False,
                                   "reason": "CUDA events include stream execution; host wall includes launch+synchronize. No subtraction is presented as a launch estimate."}
            y = (acc.float()*sx*sw.t()).cpu().numpy()
            ref = oracle.astype(np.float64)*sx.cpu().numpy()*sw.t().cpu().numpy()
            row["epilogue_oracle"] = metric(ref, y)
            row["float16_reference_error"] = metric((x.float() @ w.float().t()).cpu().numpy(), y)
            row["profile"] = profile(dynamic_total, out/f"profiler/dynamic_m{m}.json", torch)
            row["actual_integer_kernel_observed"] = bool(row["profile"]["observed_cuda_symbols"])
            row["status"] = "MEASURED"
        except Exception as exc:
            row["status"] = "UNSUPPORTED"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            capability["kernel_oracle"].append({"m": m, "n": n, "k": k, "success": False, **row["error"]})
        timings.append(row)
        rows_write(out/"microbench/phase_cost.jsonl", timings)
        write(out/"correctness/integer_kernel.json", capability)
        print(json.dumps({"phase": "activation_gpu", "m": m, "status": row["status"]}), flush=True)
    return capability


def torch_quant(x, bits, torch):
    lim = (1 << (bits-1))-1
    s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12)/lim
    q = torch.round(x.float()/s).clamp(-lim, lim).to(torch.int8)
    if bits == 4:
        packed = (q[..., ::2].to(torch.uint8) & 15) | ((q[..., 1::2].to(torch.uint8) & 15) << 4)
    else:
        packed = q
    return packed, s


def torch_dequant(p, s, bits, torch):
    if bits == 4:
        lo = (p & 15).to(torch.int8); hi = (p >> 4).to(torch.int8)
        lo = torch.where(lo >= 8, lo-16, lo); hi = torch.where(hi >= 8, hi-16, hi)
        q = torch.stack((lo, hi), -1).flatten(-2)
    else:
        q = p
    return (q.float()*s).half()


def kv_gpu(out, torch):
    from hqsb.quant.kv import KvQuantSpec, kv_capacity
    rows = []
    for t in (17, 128, 512, 2048):
        for kb, vb in ((8, 8), (4, 4), (8, 4)):
            torch.cuda.empty_cache()
            initial = torch.cuda.memory_allocated()
            shape = (2, 8, t, 128)
            k = torch.randn(shape, device="cuda", dtype=torch.float16)
            v = torch.randn(shape, device="cuda", dtype=torch.float16)
            q = torch.randn((2, 8, 1, 128), device="cuda", dtype=torch.float16)
            kp, ks = torch_quant(k, kb, torch); vp, vs = torch_quant(v, vb, torch)
            pages = math.ceil(t/16)*2*2
            header = torch.zeros((pages, 8), device="cuda", dtype=torch.uint8)
            align = torch.zeros((pages, 32), device="cuda", dtype=torch.uint8)
            exact_bytes = sum(a.numel()*a.element_size() for a in (kp, ks, vp, vs, header, align))
            expected = kv_capacity(KvQuantSpec(k_bits=kb, v_bits=vb), layers=2, kv_heads=8, head_dim=128, tokens=t, dtype_bytes=2)
            kd, vd = torch_dequant(kp, ks, kb, torch), torch_dequant(vp, vs, vb, torch)
            representation = {"k_error": metric(k.cpu().numpy(), kd.cpu().numpy()),
                              "v_error": metric(v.cpu().numpy(), vd.cpu().numpy())}
            def attn(kk, vv):
                return torch.softmax((q.float() @ kk.float().transpose(-1, -2))/math.sqrt(128), -1) @ vv.float()
            expected_attention = attn(k, v)
            actual_attention = attn(kd, vd)
            attention_error = metric(expected_attention.cpu().numpy(), actual_attention.cpu().numpy())
            def read_fn():
                return torch_dequant(kp, ks, kb, torch), torch_dequant(vp, vs, vb, torch)
            def read_attention():
                kk, vv = read_fn()
                return attn(kk, vv)
            def write_full():
                pp, ss = torch_quant(k, kb, torch); pp2, ss2 = torch_quant(v, vb, torch)
                kp.copy_(pp); ks.copy_(ss); vp.copy_(pp2); vs.copy_(ss2)
            def write_one():
                pp, ss = torch_quant(k[:, :, -1:], kb, torch)
                pp2, ss2 = torch_quant(v[:, :, -1:], vb, torch)
                kp[:, :, -1:].copy_(pp); ks[:, :, -1:].copy_(ss)
                vp[:, :, -1:].copy_(pp2); vs[:, :, -1:].copy_(ss2)
            measured = {"full_write_quant_pack_copy": event_bench(write_full, torch),
                        "one_token_write_quant_pack_copy": event_bench(write_one, torch),
                        "read_unpack_dequant": event_bench(read_fn, torch),
                        "dequantized_attention_only": event_bench(lambda: attn(kd, vd), torch),
                        "read_plus_attention": event_bench(read_attention, torch),
                        "fp16_attention": event_bench(lambda: attn(k, v), torch)}
            row = {"scope": "synthetic 2-layer CUDA packed tensor microbench; not model TPOT/capacity",
                   "context": t, "shape": list(shape), "k_bits": kb, "v_bits": vb,
                   "payload_bytes": kp.numel()*kp.element_size()+vp.numel()*vp.element_size(),
                   "scale_count": ks.numel()+vs.numel(), "scale_bytes": (ks.numel()+vs.numel())*4,
                   "header_bytes": header.numel(), "alignment_bytes": align.numel(),
                   "persistent_cache_bytes": exact_bytes, "persistent_bytes_per_token": exact_bytes/t,
                   "fp16_payload_bytes": k.numel()*k.element_size()+v.numel()*v.element_size(),
                   "temporary_full_dequant_bytes": kd.numel()*kd.element_size()+vd.numel()*vd.element_size(),
                   "shadow_cache": "reference and dequant buffers retained only for correctness/perf comparison; excluded from persistent-state claim; all live memory reported",
                   "tensor_capacity": expected.reconcile(exact_bytes, tolerance_bytes=0),
                   "allocator_all_live_delta_bytes": torch.cuda.memory_allocated()-initial,
                   "allocator_reserved_bytes": torch.cuda.memory_reserved(),
                   "representation": representation, "attention_output_error": attention_error, "perf": measured,
                   "max_context": None, "energy_token": None,
                   "page_lookup": "NOT_APPLICABLE: dense micro layout; no paged allocator claim"}
            if t == 512 and kb == vb:
                row["profile"] = profile(lambda: (write_one(), read_attention()), out/f"profiler/k{kb}v{vb}.json", torch)
            # Isolate packed-state allocation delta after deleting all reference/read buffers.
            del k, v, q, kd, vd, expected_attention, actual_attention
            torch.cuda.synchronize()
            row["packed_state_allocator_delta_bytes"] = torch.cuda.memory_allocated()-initial
            row["allocator_rounding_residual_bytes"] = row["packed_state_allocator_delta_bytes"]-exact_bytes
            rows.append(row)
            rows_write(out/"perf/packed_microbench.jsonl", rows)
            del kp, ks, vp, vs, header, align
            print(json.dumps({"phase": "kv_gpu", "context": t, "bits": [kb, vb]}), flush=True)


def gpu():
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("GPU collector requires remote Jetson")
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    free, total = torch.cuda.mem_get_info()
    if free < 512*1024**2:
        raise RuntimeError("Need at least 512 MiB free for bounded P1 microbenchmark")
    env = {"captured_at_utc": now(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
           "free_bytes_before": free, "total_bytes": total, "runner_sha256": digest(__file__),
           "process_memory_budget_bytes": 1024**3, "scope": "synthetic GPU micro only"}
    for exp in ("E05-07", "E05-08"):
        write(BASE/exp/"raw/environment_gpu.json", env)
    activation_gpu(BASE/"E05-07/raw", torch)
    torch.cuda.empty_cache()
    kv_gpu(BASE/"E05-08/raw", torch)
    env["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    env["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    env["completed_at_utc"] = now()
    for exp in ("E05-07", "E05-08"):
        write(BASE/exp/"raw/environment_gpu.json", env)
    print(json.dumps({"phase": "gpu", "completed": True, "peak_allocated_bytes": env["peak_allocated_bytes"]}), flush=True)


def profiles():
    """Separate profiler replay; elevated replay never replaces latency samples."""
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Profiler replay requires remote Jetson")
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    role = "sudo" if os.geteuid() == 0 else "user"
    aout, kout = BASE/"E05-07/raw", BASE/"E05-08/raw"
    x = torch.randn((128, 2048), device="cuda", dtype=torch.float16)
    w = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    sw = w.float().abs().amax(1, keepdim=True).clamp_min(1e-12)/127
    qw = torch.round(w.float()/sw).clamp(-127, 127).to(torch.int8)
    def dynamic_total():
        sx = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12)/127
        qx = torch.round(x.float()/sx).clamp(-127, 127).to(torch.int8)
        return (torch._int_mm(qx, qw.t()).float()*sx*sw.t()).half()
    dynamic_total(); torch.cuda.synchronize()
    result = profile(dynamic_total, aout/f"profiler/{role}_replay_m128.json", torch)
    result.update({"effective_uid": os.geteuid(), "captured_at_utc": now(),
                   "scope": "profiler-only replay; not latency data", "m": 128, "n": 2048, "k": 2048,
                   "integer_symbol_observed": any("igemm" in s.lower() or "int8" in s.lower() or "_s8_" in s.lower() for s in result["observed_cuda_symbols"]),
                   "system_profiler_configuration_changed": False})
    write(aout/f"profiler/{role}_replay_summary.json", result)
    del x, w, sw, qw
    for bits in (8, 4):
        k = torch.randn((2, 8, 512, 128), device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        q = torch.randn((2, 8, 1, 128), device="cuda", dtype=torch.float16)
        kp, ks = torch_quant(k, bits, torch); vp, vs = torch_quant(v, bits, torch)
        def write_read_attn():
            pp, ss = torch_quant(k[:, :, -1:], bits, torch)
            pp2, ss2 = torch_quant(v[:, :, -1:], bits, torch)
            kp[:, :, -1:].copy_(pp); ks[:, :, -1:].copy_(ss)
            vp[:, :, -1:].copy_(pp2); vs[:, :, -1:].copy_(ss2)
            kk = torch_dequant(kp, ks, bits, torch)
            vv = torch_dequant(vp, vs, bits, torch)
            return torch.softmax((q.float() @ kk.float().transpose(-1, -2))/math.sqrt(128), -1) @ vv.float()
        write_read_attn(); torch.cuda.synchronize()
        result = profile(write_read_attn, kout/f"profiler/{role}_replay_k{bits}v{bits}.json", torch)
        result.update({"effective_uid": os.geteuid(), "captured_at_utc": now(),
                       "scope": "profiler-only synthetic packed write/read/dequant + FP32 attention; not model latency",
                       "k_bits": bits, "v_bits": bits, "fused_quantized_attention": False,
                       "system_profiler_configuration_changed": False})
        write(kout/f"profiler/{role}_replay_k{bits}v{bits}_summary.json", result)
        del k, v, q, kp, ks, vp, vs
    print(json.dumps({"phase": "profiles", "role": role, "completed": True}), flush=True)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finalize():
    for exp in ("E05-07", "E05-08"):
        out = BASE/exp/"raw"
        is_a = exp == "E05-07"
        environment = read(out/"environment.json") if (out/"environment.json").exists() else read(out/"environment_cpu.json")
        environment["gpu"] = read(out/"environment_gpu.json") if (out/"environment_gpu.json").exists() else None
        environment["analysis_runner_sha256"] = digest(__file__)
        current_sources = [{"path": p, "sha256": digest(ROOT/p)} for p in
                           ("hqsb/quant/activation.py", "hqsb/quant/kv.py", "hqsb/quant/rtn.py", "hqsb/quant/spec.py")]
        environment.setdefault("source_files", current_sources)
        environment["analysis_source_files_current"] = current_sources
        environment["timing_limitations"] = "Clock/power/thermal state was not locked or sampled; console model was resident; five batches show timing jitter. These are diagnostic micro times, not publication-grade model speedups."
        write(out/"environment.json", environment)
        blocked = ["e05_03_frozen_calibration_handoff", "quantized_activation_model_runtime" if is_a else "quantized_kv_model_runtime",
                   "six_workload_model_quality_and_performance", "long_generation_and_drift_model_quality",
                   "graph_or_runtime_state_new_process_reload"]
        unexecuted = {"status": "NOT_EXECUTED", "reason": "Upstream frozen candidate and quantized model runtime are absent; microbench evidence cannot satisfy model gates.",
                      "required": ["six workloads", "logits/PPL/task", "free-running tokens and divergence", "TTFT/TPOT/TPS",
                                   "end-to-end memory and energy/token", "final quality with confidence intervals", "graph/artifact new-process reload"],
                      "final_evaluation_consumed": False}
        write(out/("model/not_executed.json" if is_a else "quality/not_executed.json"), unexecuted)
        if not is_a:
            historical = []
            for idx in range(3):
                p = ROOT/f"docs/stage_experiments/S02/E02-05/raw/run_{idx}.json"
                data = read(p)
                for point in data.get("context_sweep", []):
                    for phase in ("prefill", "final"):
                        meta = point.get("kv_metadata", {}).get(phase, {})
                        if not meta.get("layers"):
                            continue
                        theory = point["theory"][f"kv_{phase}"]
                        historical.append({"scope": "reused historical real cache-enabled FP16 model evidence, not a new KV quant run",
                                           "source": str(p.relative_to(ROOT)), "source_sha256": digest(p),
                                           "run_id": data["run_id"], "phase": phase, "spec": point["spec"],
                                           "theoretical_bytes": theory["total_bytes"],
                                           "observed_logical_bytes": meta["total_logical_bytes"],
                                           "observed_storage_bytes": meta["total_unique_storage_bytes"],
                                           "bytes_per_token_all_layers": theory["per_token_all_layers_bytes"],
                                           "context_filled": meta["context_filled"], "first_layer_schema": meta["layers"][0],
                                           "all_layers": meta["num_layers"],
                                           "logical_matches_formula": meta["total_logical_bytes"] == theory["total_bytes"],
                                           "storage_matches_formula": meta["total_unique_storage_bytes"] == theory["total_bytes"]})
            rows_write(out/"capacity/historical_fp16_model.jsonl", historical)
            write(out/"capacity/max_context.json", {"status": "NOT_EXECUTED", "max_context": None,
                                                    "reason": "No low-bit model cache path; synthetic allocation is not a model context capacity result.",
                                                    "safety_margin_bytes": None, "probes": []})
            write(out/"correctness/generation.json", unexecuted)
        if is_a:
            cpu_rows = jsonl(out/"correctness/granularity.jsonl")
            broadcast = read(out/"correctness/reference_broadcast_audit.json")
            micro = jsonl(out/"microbench/phase_cost.jsonl") if (out/"microbench/phase_cost.jsonl").exists() else []
            kernel = read(out/"correctness/integer_kernel.json") if (out/"correctness/integer_kernel.json").exists() else {}
            observed = read(out/"profiler/sudo_replay_summary.json") if (out/"profiler/sudo_replay_summary.json").exists() else {}
            # Classification is derived from the saved trace symbols. The original
            # profiler summary remains unchanged even if its first classifier was narrow.
            observed = dict(observed)
            observed["integer_symbols"] = [s for s in observed.get("observed_cuda_symbols", [])
                                           if "igemm" in s.lower() or "int8" in s.lower() or "_s8_" in s.lower()]
            observed["integer_symbol_observed"] = bool(observed["integer_symbols"])
            observed["classification_rule"] = "CUDA symbol contains igemm, int8, or _s8_; actual observed CUTLASS i16832gemm_s8 symbol is retained"
            observed["source_summary"] = "profiler/sudo_replay_summary.json"
            write(out/"profiler/kernel_identity.json", observed)
            findings = []
            if any(abs(r["reported_scale_utilization"]-r["independent_half_range_utilization"]) > 1e-12 for r in cpu_rows):
                findings.append("collected implementation scale_utilization does not match half-range usage")
            if not broadcast["passed"]:
                findings.append("collected w8a8_reference default scale broadcast fails per-row/per-column vector oracle")
            criteria = [
                ("activation数学和scale状态完全可复现", not findings, "contract vectors saved; defect list reflects the collected source version only"),
                ("static/dynamic/granularity对照公平", bool(micro), "synthetic micro only"),
                ("calibration/final隔离", True, "saved calibration only, final not consumed"),
                ("SmoothQuant模型变换等价（若启用）", False, "not enabled; primitive-only diagnostic"),
                ("quantizer和W8A8 kernel oracle通过", False, "GPU independent oracle recorded; existing helper defects remain"),
                ("observed真实W8A8 kernel", observed.get("integer_symbol_observed", False), "sudo profiler-only CUDA microkernel replay; no model coverage"),
                ("在线阶段成本可归因", bool(micro), "independent stage and total CUDA event times; no inferred launch time"),
                ("六workload质量和phase性能完整", False, "NOT_EXECUTED"),
                ("分布漂移与长生成边界明确", False, "synthetic drift only; model NOT_EXECUTED"),
                ("artifact/graph重载通过", False, "NOT_EXECUTED")]
            summary = {"contract_cases": len(cpu_rows), "contract_cases_pass": sum(r["passed"] for r in cpu_rows),
                       "existing_reference_broadcast_pass": broadcast["passed"], "known_defects": findings,
                       "micro_shapes": [{"m": r["m"], "status": r["status"]} for r in micro],
                       "integer_kernel": kernel, "observed_kernel_replay": observed}
            expected = "判断 activation quant 是否值得进入主线"
            standard = "质量可接受且 actual low-bit kernel 命中；否则以负结论和 fallback 收束"
        else:
            cp = jsonl(out/"capacity/cpu_tensor_storage.jsonl")
            gpu_rows = jsonl(out/"perf/packed_microbench.jsonl") if (out/"perf/packed_microbench.jsonl").exists() else []
            rep = jsonl(out/"correctness/representation.jsonl")
            criteria = [
                ("真实cache schema、RoPE点和K/V语义冻结", False, "model config/historical FP16 only; synthetic packed schema"),
                ("FP16与低bit容量公式均和实测闭环", False, "synthetic tensor storage reconciled; no low-bit model allocator"),
                ("表示、attention与自回归三级正确性通过", False, "synthetic representation and attention only; generation absent"),
                ("实际模型cache读写低bit，无FP16 shadow", False, "packed tensor micro only; model runtime missing"),
                ("observed quant/dequant/attention kernel可审计", bool(gpu_rows), "micro profiler only"),
                ("context与generation长度矩阵完整", False, "context micro sweep complete; generation NOT_EXECUTED"),
                ("final/长上下文质量通过", False, "NOT_EXECUTED"),
                ("metadata、碎片、最大context、TPOT、energy完整", False, "metadata micro measured; model metrics NOT_EXECUTED"),
                ("新进程与跨请求reset通过", False, "NOT_EXECUTED"),
                ("适用域和fallback明确", True, "no low-bit model claim; keep FP16 runtime")]
            summary = {"capacity_contract_cases": len(cp), "capacity_contract_pass": sum(r["within_tolerance"] for r in cp),
                       "representation_cases": len(rep), "pack_unpack_pass": sum(r["pack_unpack_exact"] for r in rep),
                       "gpu_micro_cases": len(gpu_rows),
                       "gpu_tensor_storage_pass": sum(r["tensor_capacity"]["within_tolerance"] for r in gpu_rows),
                       "historical_fp16_model_points": len(historical),
                       "historical_fp16_storage_matches": sum(r["storage_matches_formula"] for r in historical),
                       "known_limitations": ["residual capacity helper defaults float32; explicit dtype_bytes=2 required for FP16",
                                             "dense micro format headers/alignment are experimental allocations, not S07 model page allocator",
                                             "attention uses full dequant temporary buffer; not fused quantized attention"]}
            expected = "验证长上下文容量收益是否抵消质量和 TPOT 成本"
            standard = "容量公式与实测相符；长生成误差受控；不支持硬件明确降级"
        verdict = {"schema": f"hqsb.{exp.lower().replace('-', '_')}.verdict/v1", "experiment_id": exp,
                   "overall": "BLOCKED", "scientific_execution_verdict": "NOT_EXECUTED",
                   "micro_execution_verdict": "COMPLETED" if (out/"environment_gpu.json").exists() else "NOT_EXECUTED",
                   "formal_pass_allowed": False, "deployment_claim_allowed": False,
                   "expected_effect": {"name": expected, "passed": False, "reason": "model evidence unavailable"},
                   "single_item_standard": {"name": standard, "passed": False, "reason": "micro/contract evidence is insufficient for model gate"},
                   "detail_pass_criteria": [{"id": i+1, "name": name, "passed": passed, "evidence_scope": evidence}
                                            for i, (name, passed, evidence) in enumerate(criteria)],
                   "blocking_ids": blocked, "final_evaluation_consumed": False,
                   "claim_boundary": "No W8A8 or low-bit KV model quality/speed/capacity/energy claim; no PASS_NEGATIVE for an incomplete experiment.",
                   "fallback": "existing FP16 cache/FP16 model; E05-02 W8/W4 quality failures remain upstream failures",
                   "format_deviation": "JSON/JSONL instead of parquet for dependency-free evidence browsing", "verified_at_utc": now()}
        summary.update({"experiment_id": exp, "overall": "BLOCKED", "scope": "partial P1 micro/contract evidence; formal model experiment NOT_EXECUTED"})
        write(out/"summary.json", summary)
        write(out/"verdict.json", verdict)
        write(out/"EVIDENCE_MANIFEST.json", {"experiment_id": exp, "generated_at_utc": now(),
                                            "files": [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": digest(p)}
                                                      for p in sorted(out.rglob("*")) if p.is_file() and p.name != "EVIDENCE_MANIFEST.json"]})
    print(json.dumps({"phase": "finalize", "completed": True}), flush=True)


def verify():
    checked = 0
    for exp in ("E05-07", "E05-08"):
        out = BASE/exp/"raw"
        manifest = read(out/"EVIDENCE_MANIFEST.json")
        for record in manifest["files"]:
            path = out/record["path"]
            assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], path
            if path.suffix == ".json":
                read(path)
            elif path.suffix == ".jsonl":
                jsonl(path)
            checked += 1
        verdict = read(out/"verdict.json")
        assert verdict["overall"] == "BLOCKED" and not verdict["deployment_claim_allowed"]
    print(json.dumps({"phase": "verify", "checked_files": checked, "passed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cpu", "gpu", "profiles", "finalize", "verify"))
    args = parser.parse_args()
    globals()[args.phase]()
