#!/usr/bin/env python3
"""Collect actual W4/W8 GPU kernel evidence, keeping formal S05 gates separate."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from s05_remaining_common import (STAGE, append, ci95, initialize, load_weight, manifest,
                                  prepared_weight, prereqs, read, rows, sha, write)

MODEL = "~/models/hqsb/Qwen3-1.7B"
MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SPEC = {"scope": "exploratory real-weight microbench and reversible one-module model probe; M4 absent",
        "bits": [4, 8], "group": {"4": 128, "8": "per-output-channel"},
        "modules": list(MODULES), "M": [1, 32, 128, 512], "independent_processes": 3,
        "warmup": 3, "repetitions": 10, "seed": 5606,
        "kernel_gate": {"relative_l2_max": 0.002, "max_abs_error_max": 0.125, "finite": True},
        "controls": ["FP16 source GEMM", "FP16 dequant-weight GEMM", "fused packed-weight GEMM"],
        "matrix_note": "real checkpoint weights; seeded synthetic activations for microbench only",
        "excluded_claims": ["integer INT4 MMA", "full-model memory benefit", "industrial kernel comparison", "formal S05 PASS"]}


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def measure(fn, repeats=10):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    device, wall = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        value = fn()
        end.record()
        end.synchronize()
        device.append(begin.elapsed_time(end))
        wall.append((time.perf_counter() - started) * 1000)
    return {"device_ms": device, "wall_ms": wall, "device_median_ms": statistics.median(device),
            "wall_median_ms": statistics.median(wall), "checksum": float(value.float().sum())}


def error(candidate, reference):
    import torch
    delta = candidate.float() - reference.float()
    return {"max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm() / reference.float().norm().clamp_min(1e-20)),
            "finite": bool(torch.isfinite(candidate).all())}


def passed(value):
    return value["finite"] and value["relative_l2"] <= 0.002 and value["max_abs"] <= 0.125


def begin(args):
    return initialize("E05-06", SPEC, args.output)


def capabilities(output):
    from ops.quant.capability import probe_low_bit_capability
    from ops.quant.w4a16_triton import compile_probe
    probe = probe_low_bit_capability()
    result = probe.as_dict()
    # Numerical probe results must be checked, not merely the absence of an exception.
    if probe.fused_dequant_available:
        result["numerical_probe"] = compile_probe()
        result["numerical_probe_pass"] = result["numerical_probe"]["ok"]
    result["industrial_libraries"] = {name: importlib.util.find_spec(name) is not None
                                      for name in ("awq", "gptqmodel", "bitsandbytes", "torchao", "tensorrt_llm")}
    write(output / "capability.json", result)
    return result


def micro(args):
    import torch
    from ops.quant.w4a16_triton import gemm_low_bit
    output = Path(args.output) if args.output else STAGE / "E05-06/raw"
    target = output / f"micro/process_{args.run}.jsonl"
    if target.exists():
        raise FileExistsError(f"Refusing to append another process to {target}")
    begin(args)
    write(output / f"micro/process_{args.run}_identity.json", {"pid": os.getpid(), "run": args.run,
          "runner_sha256": sha(__file__), "shared_helpers_sha256": sha(Path(__file__).with_name("s05_remaining_common.py")),
          "kernel_sha256": sha(ROOT / "ops/quant/w4a16_triton.py")})
    capability = capabilities(output)
    if not capability.get("numerical_probe_pass"):
        raise RuntimeError("fused low-bit numerical capability probe failed")
    torch.manual_seed(SPEC["seed"] + args.run)
    with torch.inference_mode():
        for suffix in MODULES:
            name = "model.layers.0." + suffix + ".weight"
            source_cpu = load_weight(args.model, name)
            source_hash = tensor_hash(source_cpu)
            weight = source_cpu.to(device="cuda", dtype=torch.float16)
            for bits in (4, 8):
                started = time.perf_counter()
                prepared, dequant = prepared_weight(weight, bits)
                torch.cuda.synchronize()
                offline_ms = (time.perf_counter() - started) * 1000
                n, k = weight.shape
                for m in SPEC["M"]:
                    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
                    started = time.perf_counter()
                    observed = gemm_low_bit(x, prepared)
                    torch.cuda.synchronize()
                    first_ms = (time.perf_counter() - started) * 1000
                    reference = x @ dequant.t()
                    kernel_error = error(observed, reference)
                    record = {"process": args.run, "module": name, "source_weight_sha256": source_hash,
                              "M": m, "N": n, "K": k, "bits": bits,
                              "phase": "decode" if m == 1 else "prefill", "packed": prepared.as_dict(),
                              "packed_bytes": prepared.packed.numel(), "scale_bytes": prepared.scales.numel() * 4,
                              "fp16_bytes": weight.numel() * 2, "workspace_bytes": 0,
                              "output_bytes": m * n * 2, "offline_quant_pack_ms": offline_ms,
                              "first_call_in_process_ms": first_ms, "algorithm_error": error(dequant, weight),
                              "kernel_error": kernel_error, "correctness_pass": passed(kernel_error),
                              "execution": "fused_dequant_weight_only", "weights_materialized_to_fp16_in_fused_path": False}
                    paths = {"fp16": lambda: x @ weight.t(), "storage_only": lambda: x @ dequant.t(),
                             "fused": lambda: gemm_low_bit(x, prepared)}
                    # Alternate control ordering across independent processes.
                    order = list(paths) if args.run % 2 == 0 else list(reversed(paths))
                    for path in order:
                        record[path] = measure(paths[path])
                        record[path]["logical_tflops"] = 2 * m * n * k / (record[path]["device_median_ms"] * 1e9)
                    record["fused_speedup_vs_fp16"] = record["fp16"]["device_median_ms"] / record["fused"]["device_median_ms"]
                    append(target, record)
                del prepared, dequant
            del weight, source_cpu
            torch.cuda.empty_cache()
            print(f"process={args.run} {suffix} complete", flush=True)
    boundaries(output, args.run)


def boundaries(output, run):
    import torch
    from ops.quant.w4a16_triton import gemm_low_bit
    destination = output / f"correctness/boundaries_{run}.jsonl"
    with torch.inference_mode():
        for bits in (4, 8):
            for m, n, k in ((1, 17, 129), (2, 31, 127), (7, 33, 257), (1, 1, 1)):
                weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * .1
                prepared, dequant = prepared_weight(weight, bits)
                for strided in (False, True):
                    x = torch.randn(m, k * (2 if strided else 1), device="cuda", dtype=torch.float16)
                    x = x[:, ::2] if strided else x
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual = gemm_low_bit(x, prepared)
                    torch.cuda.current_stream().wait_stream(stream)
                    metric = error(actual, x @ dequant.t())
                    append(destination, {"bits": bits, "M": m, "N": n, "K": k, "strided": strided,
                                         "nondefault_stream": True, "kernel_error": metric, "passed": passed(metric)})


def profile(args):
    import torch
    from ops.quant.w4a16_triton import gemm_low_bit
    output = begin(args)
    source = load_weight(args.model, "model.layers.0.self_attn.q_proj.weight").to("cuda", torch.float16)
    prepared, _ = prepared_weight(source, 4)
    x = torch.ones((1, prepared.k), device="cuda", dtype=torch.float16)
    gemm_low_bit(x, prepared)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        gemm_low_bit(x, prepared)
        torch.cuda.synchronize()
    path = output / "profiler/micro_trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    trace = read(path)
    kernels = [entry for entry in trace.get("traceEvents", []) if entry.get("cat") == "kernel"]
    write(output / "profiler/observed.json", {"kernels": kernels, "trace_sha256": sha(path),
          "fused_symbol_observed": any("hqsb_dequant_gemm_kernel" in e.get("name", "") for e in kernels),
          "ordinary_timing_separate": True})


def model_probe(args):
    import torch
    from hqsb.models.loader import load_qwen3
    from hqsb.benchmark.workload import make_fixed_token_input
    from ops.quant.w4a16_triton import gemm_low_bit
    output = begin(args)
    free, _ = torch.cuda.mem_get_info()
    if free < 4_800_000_000:
        write(output / "model/not_executed.json", {"status": "BLOCKED", "free_bytes": free,
              "reason": "Insufficient free unified memory for fully resident model probe; active services preserved"})
        return
    tokenizer, model, load_time = load_qwen3(args.model, dtype=torch.float16, attention_backend="eager",
          verify_manifest=str(ROOT / "docs/benchmark/model_sha256_manifest.txt"), allow_extra=("model_sha256_manifest.txt",), cpu_staging=True)
    devices = sorted({str(p.device) for p in model.parameters()})
    if not all(d.startswith("cuda") for d in devices):
        raise RuntimeError(f"Model not fully CUDA resident: {devices}")
    parent = model.model.layers[0].self_attn
    original = parent.q_proj
    prepared, dequant = prepared_weight(original.weight.detach(), 4)
    calls = []

    class Probe(torch.nn.Module):
        def forward(self, value):
            shape = value.shape
            flat = value.reshape(-1, shape[-1])
            calls.append({"M": flat.shape[0], "N": prepared.n, "K": prepared.k, "dtype": str(value.dtype)})
            result = gemm_low_bit(flat, prepared).reshape(*shape[:-1], prepared.n)
            return result + original.bias if original.bias is not None else result

    ids = make_fixed_token_input(tokenizer, 32, device="cuda")
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    with torch.inference_mode():
        reference = model(input_ids=ids, use_cache=False).logits
        # Quantized FP16 reference separates algorithm error from kernel error.
        source_weight = original.weight.detach().clone()
        original.weight.copy_(dequant)
        quant_cache = model(input_ids=ids, use_cache=True)
        quant_reference = quant_cache.logits
        fixed_decode_token = quant_reference[:, -1].argmax(-1, keepdim=True)
        quant_decode_reference = model(input_ids=fixed_decode_token,
                                       past_key_values=quant_cache.past_key_values, use_cache=True).logits
        del quant_cache
        original.weight.copy_(source_weight)
        del source_weight
        parent.q_proj = Probe()
        try:
            model(input_ids=ids, use_cache=False)
            calls.clear()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                cache = model(input_ids=ids, use_cache=True)
                actual = cache.logits
                decode_actual = model(input_ids=fixed_decode_token,
                                      past_key_values=cache.past_key_values, use_cache=True).logits
                torch.cuda.synchronize()
            path = output / "profiler/model_trace.json"
            prof.export_chrome_trace(str(path))
            observed = [e for e in read(path).get("traceEvents", []) if e.get("cat") == "kernel" and "hqsb_dequant_gemm_kernel" in e.get("name", "")]
            write(output / "model/probe.json", {"scope": "one q_proj replaced; original FP16 retained for restoration; not full-model deployment",
                  "load_s": load_time, "module": "model.layers.0.self_attn.q_proj", "calls": calls,
                  "observed_fused_kernel_count": len(observed), "kernels": observed,
                  "algorithm_logit_error": error(quant_reference, reference), "kernel_logit_error": error(actual, quant_reference),
                  "decode_kernel_logit_error": error(decode_actual, quant_decode_reference),
                  "decode_scope": "one teacher-forced token on independently populated candidate/reference caches; not long generation",
                  "trace_sha256": sha(path), "parameter_devices": devices,
                  "packed": prepared.as_dict(), "allocated_bytes": torch.cuda.memory_allocated(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "actual_low_bit_module_count": 1})
        finally:
            parent.q_proj = original
        restored = model(input_ids=ids, use_cache=False).logits
        write(output / "model/restoration.json", {"exact": bool(torch.equal(restored, reference)), "error": error(restored, reference)})
    print("model probe completed", flush=True)


def summarize(args):
    output = Path(args.output) if args.output else STAGE / "E05-06/raw"
    write(output / "prerequisites_resolved.json", {"checks": prereqs(),
          "note": "Derived audit accepts overall/status/verdict; the original collection snapshot is retained unchanged"})
    records = [row for path in sorted((output / "micro").glob("process_*.jsonl")) for row in rows(path)]
    boundary = [row for run in (0, 1, 2) if (output / f"correctness/boundaries_{run}.jsonl").exists()
                for row in rows(output / f"correctness/boundaries_{run}.jsonl")]
    instrumented = {str(run): rows(output / f"correctness/boundaries_{run}.jsonl")
                    for run in (99, 100) if (output / f"correctness/boundaries_{run}.jsonl").exists()}
    groups = {}
    for row in records:
        groups.setdefault((row["module"], row["bits"], row["M"]), []).append(row)
    summary = []
    for (module, bits, m), values in groups.items():
        summary.append({"module": module, "bits": bits, "M": m, "N": values[0]["N"], "K": values[0]["K"],
                        "fused_ms": ci95(v["fused"]["device_median_ms"] for v in values),
                        "fp16_ms": ci95(v["fp16"]["device_median_ms"] for v in values),
                        "speedup": ci95(v["fused_speedup_vs_fp16"] for v in values),
                        "all_correct": all(v["correctness_pass"] for v in values)})
    write(output / "summary.json", {"cases": summary, "micro_rows": len(records), "ordinary_boundary_rows": len(boundary),
          "unique_boundary_configurations": 16, "instrumented_boundary_rows_by_run": {k: len(v) for k, v in instrumented.items()},
          "instrumented_note": "99 attempted user memcheck (tool refused); 100 privileged memcheck; excluded from ordinary rows"})
    observed = read(output / "profiler/observed.json", {})
    model = read(output / "model/probe.json", {})
    restoration = read(output / "model/restoration.json", {})
    sanitizer = {}
    for label in ("user", "privileged"):
        log = output / f"correctness/memcheck_{label}.txt"
        code = output / f"correctness/memcheck_{label}.exit"
        if log.exists() and code.exists():
            sanitizer[label] = {"exit_code": int(code.read_text().strip()), "log_sha256": sha(log),
                                "zero_errors_observed": "ERROR SUMMARY: 0 errors" in log.read_text(),
                                "path": str(log.relative_to(output))}
    sanitizer["status"] = "PASS" if any(row.get("exit_code") == 0 and row.get("zero_errors_observed")
                                         for row in sanitizer.values() if isinstance(row, dict)) else "NOT_PASSED"
    write(output / "correctness/sanitizer.json", sanitizer)
    profiles = []
    for bits in (4, 8):
        for m in (1, 128):
            path = output / f"profiler/ncu_w{bits}_m{m}.csv"
            code = output / f"commands/ncu_w{bits}_m{m}.exit"
            if not path.exists():
                continue
            lines = [line for line in path.read_text().splitlines() if line.startswith('"')]
            metrics = list(csv.DictReader(lines)) if lines else []
            exit_code = int(code.read_text().strip()) if code.exists() else None
            valid = exit_code == 0 and any("hqsb_dequant_gemm_kernel" in row.get("Kernel Name", "")
                                          and row.get("Metric Name") and row.get("Metric Value") for row in metrics)
            profiles.append({"bits": bits, "M": m, "path": str(path.relative_to(output)), "sha256": sha(path),
                             "exit_code": exit_code, "valid": valid, "metrics": metrics, "ordinary_timing": False})
    write(output / "profiler/ncu_summary.json", {"profiles": profiles, "instruction_claim": "fused dequant to FP16 MMA; no integer MMA claim"})
    blockers = ["S04.5 M4 absent", "E05-02 quality baseline rejected", "E05-04 industrial artifact absent",
                "full six-workload low-bit model benchmark and energy not collected"]
    verdict = {"experiment_id": "E05-06", "overall": "BLOCKED", "scientific_execution_verdict": "PARTIAL",
               "expected_effect_met": False, "single_item_standard_met": False, "blocking_reasons": blockers,
               "micro_correctness_pass": bool(records) and all(r["correctness_pass"] for r in records),
               "tail_stream_correctness_pass": bool(boundary) and all(r["passed"] for r in boundary),
               "three_independent_processes": len({r["process"] for r in records}) == 3,
               "profile_kernel_observed": observed.get("fused_symbol_observed", False),
               "model_kernel_observed": model.get("observed_fused_kernel_count", 0) > 0,
               "model_prefill_same_kernel_error_gate": passed(model["kernel_logit_error"]) if model else None,
               "model_decode_same_kernel_error_gate": passed(model["decode_kernel_logit_error"]) if model else None,
               "model_restoration_exact": restoration.get("exact"),
               "sanitizer": sanitizer,
               "ncu_profiles_attempted": len(profiles),
               "ncu_profiles_collected": sum(p["valid"] for p in profiles),
               "no_silent_fp16_fallback": True, "admission_to_deployment_pareto": False}
    write(output / "verdict.json", verdict)
    text = ["# E05-06 实验报告：真实 W8/W4 GEMM 与模型路径\n",
            "最终判定：**BLOCKED（已采集部分真实执行证据，正式前置不满足）**。",
            "\n本实验使用 Qwen3-1.7B checkpoint 的真实 projection 权重和固定随机 activation，区分量化算法误差与 kernel 对 dequant-reference 的实现误差。Fused 路径直接读取 packed W4/W8，在寄存器中反量化为 FP16 fragment；不声称 INT4 整数 MMA。",
            f"\n完成 {len(records)} 条真实权重微基准、{len(boundary)} 条 tail/非连续输入/非默认 stream 正确性记录。三个进程的各自中位数是 CI 的统计单位，launch 重复不冒充独立进程。",
            "\n| 观察项 | 结果 |\n|---|---|",
            *[f"| {k} | {v} |" for k, v in verdict.items() if isinstance(v, bool)],
            "\n| Projection | bit | M | Fused ms（进程均值） | FP16 ms | speedup |\n|---|---:|---:|---:|---:|---:|",
            *[f"| {v['module']} | {v['bits']} | {v['M']} | {v['fused_ms']['mean']:.4f} | {v['fp16_ms']['mean']:.4f} | {v['speedup']['mean']:.3f} |" for v in summary],
            "\n必采集信息：packed layout/hash、M/N/K/group、packed/scale/output/workspace bytes、离线 quant+pack、first-call、host/device steady 时间、logical TFLOPS、algorithm/kernel error 全部在 [raw](raw/summary.json) 及 `micro/`。首次调用包括可能的缓存编译，不能当纯编译耗时；profile/sanitizer 与 ordinary timing 分离。",
            f"\n独立安全检查：{sanitizer['status']}。普通用户 GPU debugging 权限失败的原日志与 sudo 重放均保留；没有改系统权限配置。NCU 尝试 {len(profiles)} 个、成功 {sum(p['valid'] for p in profiles)} 个 W4/W8 × decode/prefill basic profile，成功要求 exit=0、目标 kernel 和非空指标同时成立。完整指标见 [ncu_summary.json](raw/profiler/ncu_summary.json)，不得将 profiler replay 时间当 ordinary latency。basic 集不能代替完整指令、DRAM bytes 与 roofline 归因。",
            "\n成熟工业低比特库、完整模型六 workload、TPOT/TTFT、能量仍不能由本轮微基准推导；缺失值未记为零。模型 probe 如成功，也仅证明一个真实 q_proj 的 prefill/decode 命中，可逆恢复；保留 FP16 原模块用于恢复，不产生整模型内存收益结论。模型 probe 的内存检查结果见 `model/`，现有控制台服务未被停止。",
            "\n模型补测：" + (f"内存恢复后实际完成 `model.layers.0.self_attn.q_proj` 的 M=32 prefill 与 M=1 decode，CUPTI 观察到 {model['observed_fused_kernel_count']} 次 fused kernel。相对同量化 FP16 参考，prefill logit relative L2={model['kernel_logit_error']['relative_l2']:.8f}、max abs={model['kernel_logit_error']['max_abs']:.8f}；单步 decode relative L2={model['decode_kernel_logit_error']['relative_l2']:.8f}、max abs={model['decode_kernel_logit_error']['max_abs']:.8f}。将冻结的 0.002 relative L2/0.125 max-abs 数值门保守用于该模型诊断，prefill={verdict['model_prefill_same_kernel_error_gate']}、decode={verdict['model_decode_same_kernel_error_gate']}；decode 未通过不能因为 kernel 命中而隐藏。恢复原 module 后输出逐值一致={restoration.get('exact')}。仅 32-token prefill+1 token，不替代长生成/任务质量门。" if model else "未完成，见独立 blocker。"),
            "\n预计效果：部分证明 packed weight 可执行；未形成完整部署收益因果链。单项标准：正确性/尾部、模型命中见机器判定，但上游正式门和工业输入未通过，**不满足整体通过标准**。不能用性能负结果覆盖质量/前置失败。",
            "\n阻塞：" + "；".join(blockers) + "。",
            "\n证据：[预注册](raw/spec.json)、[能力](raw/capability.json)、[汇总及95%CI](raw/summary.json)、[判定](raw/verdict.json)、[清单](raw/EVIDENCE_MANIFEST.json)。前端证据中心通过 `raw/verdict.json` 自动索引本目录。",
            "\n复现（远端）：`./scripts/remote_run.sh python3 scripts/audit/run_e05_06_low_bit_execution.py micro --run 0 --output /tmp/e05-06-replay`；依次 run 1、2，再 `profile`、`model`、`summarize`。正式运行前必须先完成 M4、E05-02～04。"]
    (output.parent / "E05-06_实验报告.md").write_text("\n".join(text) + "\n")
    manifest(output)
    print(json.dumps(verdict, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("micro", "profile", "model", "boundaries", "summarize"))
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "boundaries":
        boundaries(begin(args), args.run)
    else:
        {"micro": micro, "profile": profile, "model": model_probe, "summarize": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
