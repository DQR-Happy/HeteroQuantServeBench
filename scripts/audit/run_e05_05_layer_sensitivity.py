#!/usr/bin/env python3
"""Real-weight sensitivity census; model interventions stay prerequisite-gated."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from s05_remaining_common import (STAGE, append, initialize, load_weight, manifest,
                                  quant_weight, read, rows, sha, write)

SPEC = {"scope": "checkpoint weight census and inherited activation-stat audit; no final-evaluation access",
        "model": "Qwen/Qwen3-1.7B", "bits": [4, 8], "group": {"4": 128, "8": "per-channel"},
        "selected_units": "all transformer attention q/k/v/o and MLP gate/up/down Linear weights",
        "excluded": ["embedding", "norm", "lm_head"],
        "excluded_reason": "preserve E05-02 scope; no unmeasured tied-weight or normalization intervention",
        "selection_rule": "no final policy without validated model-level single/LOO/cumulative/pairwise and independent quality gate",
        "seed": 5505, "outlier_definition": "abs(weight) > 6 * weight RMS",
        "cosine_accumulation": "chunked FP64 dot and squared norms; no clipping",
        "timing_boundary": "CPU RTN quantization/reconstruction timed separately from diagnostics",
        "model_interventions": ["single_quant", "leave_one_out", "cumulative", "pairwise"],
        "formal_prerequisites": ["S04.5 M4", "E05-02 quality baseline", "E05-03 frozen calibration", "E05-04 industrial artifacts"]}


def stable_cosine(left, right):
    """Avoid float32 reduction error on multi-million-element projections."""
    left, right = left.reshape(-1), right.reshape(-1)
    dot = left_sq = right_sq = 0.0
    for offset in range(0, left.numel(), 262144):
        a = left[offset:offset + 262144].double()
        b = right[offset:offset + 262144].double()
        dot += float((a * b).sum())
        left_sq += float(a.square().sum())
        right_sq += float(b.square().sum())
    return dot / math.sqrt(left_sq * right_sq) if left_sq and right_sq else (1.0 if not left_sq and not right_sq else 0.0)


def collect(args):
    import torch
    torch.set_num_threads(2)
    output = Path(args.output) if args.output else STAGE / "E05-05/raw"
    target = output / "stats/weight.jsonl"
    if target.exists():
        raise FileExistsError("Use a fresh output directory for another census")
    initialize("E05-05", SPEC, output)
    model = Path(args.model).expanduser()
    index_path = model / "model.safetensors.index.json"
    index = read(index_path)["weight_map"]
    names = sorted(name for name in index if name.startswith("model.layers.") and name.endswith("_proj.weight"))
    source_manifest = model / "model_sha256_manifest.txt"
    write(output / "source.json", {"model_path": str(model), "index_sha256": sha(index_path),
          "model_manifest_sha256": sha(source_manifest), "scope": names,
          "scope_count": len(names), "device": "cpu", "CUDA_inference": False})
    started = time.perf_counter()
    with torch.inference_mode():
        for i, name in enumerate(names):
            weight = load_weight(model, name).to(torch.float16)
            source = weight.float()
            rms = float(source.square().mean().sqrt())
            amax = float(source.abs().max())
            for bits in (4, 8):
                t0 = time.perf_counter()
                q, scales, reconstructed = quant_weight(weight, bits)
                quant_ms = (time.perf_counter() - t0) * 1000
                delta = reconstructed.float() - source
                n, k = weight.shape
                row = {"module": name.removesuffix(".weight"), "layer": int(name.split(".")[2]),
                       "unit": name.split(".", 3)[3].removesuffix(".weight"), "shape": [n, k],
                       "bits": bits, "group": 128 if bits == 4 else None, "symmetric": True,
                       "rmse": float(delta.square().mean().sqrt()),
                       "relative_l2": float(delta.norm() / source.norm().clamp_min(1e-20)),
                       "cosine": stable_cosine(source, reconstructed),
                       "max_abs_error": float(delta.abs().max()), "source_rms": rms, "source_absmax": amax,
                       "outlier_rate": float((source.abs() > 6 * rms).float().mean()),
                       "scale_min": float(scales.min()), "scale_max": float(scales.max()),
                       "scale_mean": float(scales.mean()), "zero_point": 0,
                       "saturation_rate": float((q.abs() == (2 ** (bits - 1) - 1)).float().mean()),
                       "source_bytes": weight.numel() * 2,
                       "payload_bytes": n * (k if bits == 8 else math.ceil(k / 2)),
                       "scale_bytes": scales.numel() * 4, "offline_cpu_quant_ms": quant_ms,
                       "offline_cpu_quant_and_stats_ms": (time.perf_counter() - t0) * 1000,
                       "evidence_level": "weight-reconstruction; not model-sensitivity"}
                append(target, row)
                del q, scales, reconstructed, delta
            del weight, source
            if (i + 1) % 7 == 0:
                print(f"weight census {i+1}/{len(names)}", flush=True)
    write(output / "offline_cost.json", {"wall_s": time.perf_counter() - started, "device": "cpu",
          "scope": "weight load, RTN reconstruction and statistics; not model inference latency"})
    inherited = []
    stats_dir = STAGE / "E05-03/raw/stats/activation"
    for path in sorted(stats_dir.glob("*.json*")):
        inherited.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size})
    if not inherited:
        # Preserve references to sample statistics without mislabelling them as new observations.
        for path in sorted(stats_dir.glob("per_sample/*.json")):
            inherited.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size})
    write(output / "stats/inherited_activation.json", {"provenance": "E05-03; selected modules only",
          "references": inherited, "full_196_module_activation_census": False,
          "reason": "Archived selected-module statistics do not establish all-layer causal sensitivity"})
    summarize(args)


def summarize(args):
    output = Path(args.output) if args.output else STAGE / "E05-05/raw"
    values = rows(output / "stats/weight.jsonl")
    by_layer = []
    for layer in range(28):
        selected = [r for r in values if r["layer"] == layer and r["bits"] == 4]
        by_layer.append({"layer": layer, "units": len(selected),
                         "mean_weight_relative_l2": sum(r["relative_l2"] for r in selected) / len(selected),
                         "max_weight_relative_l2": max(r["relative_l2"] for r in selected),
                         "w4_bytes": sum(r["payload_bytes"] + r["scale_bytes"] for r in selected),
                         "fp16_bytes": sum(r["source_bytes"] for r in selected)})
    write(output / "stats/layers.json", by_layer)
    missing = {
        "single_quant": "real model logits/hidden KL not measured; weight error cannot substitute",
        "leave_one_out": "all-W4 baseline failed model gate; recovery not measured",
        "cumulative": "error propagation model trajectories not measured",
        "pairwise": "non-additive layer effects not measured",
        "policy_search": "M4 and frozen calibration handoff absent; no policy may be frozen",
        "final_evaluation": "NOT_OPENED; independent split preserved",
        "six_workload_latency_energy": "no accepted mixed-precision candidate to benchmark"}
    for name, reason in missing.items():
        write(output / f"interventions/{name}.json", {"status": "NOT_EXECUTED", "reason": reason})
    write(output / "policies/selection.json", {"status": "BLOCKED", "final_policy": None,
          "deployable": False, "reason": "Local weight error is not a model-level policy objective",
          "final_evaluation_opened": False})
    verdict = {"experiment_id": "E05-05", "overall": "BLOCKED", "scientific_execution_verdict": "PARTIAL",
               "expected_effect_met": False, "single_item_standard_met": False,
               "weight_census_complete": len(values) == 392, "quantized_module_count": len(values) // 2,
               "layer_count": len(by_layer), "mixed_policy_selected": False,
               "final_evaluation_opened": False, "blocking_reasons": list(missing.values()),
               "formal_prerequisites": read(output / "prerequisites.json")}
    write(output / "verdict.json", verdict)
    rank = sorted(by_layer, key=lambda r: r["mean_weight_relative_l2"], reverse=True)
    text = ["# E05-05 实验报告：逐层/模块量化误差与混合精度前置审计\n",
            "最终判定：**BLOCKED（真实权重统计已完成，模型因果干预及最终策略未完成）**。",
            f"\n对 Qwen3-1.7B 的 {len(by_layer)} 个 block、{len(values)//2} 个真实 Linear，采集 W4(group=128) 与 W8(per-channel) 共 {len(values)} 条 weight 重构记录。模型来源、scope、文件 hash 在 [source.json](raw/source.json)，数学、outlier 定义和策略禁止条件在 [预注册](raw/spec.json)。所有采集在 Jetson 运行。",
            "\n必采集数据完成部分：每个 module 的 shape、bit/group、scale range/mean、zero-point、RMSE/relative-L2/cosine、max error、outlier/saturation、payload/scale/FP16 bytes 和离线 CPU 量化成本。E05-03 已保存的 activation 统计仅作为带 hash 的输入引用，不能代表全模型逐层 activation 已采集。",
            "\n下表按 **weight reconstruction error** 排序；它不是模型敏感层排名，不能据此直接跳层或选混合精度。",
            "\n| block | W4 mean relative L2 | W4 bytes | FP16 bytes |\n|---:|---:|---:|---:|",
            *[f"| {r['layer']} | {r['mean_weight_relative_l2']:.6f} | {r['w4_bytes']} | {r['fp16_bytes']} |" for r in rank],
            "\n尚缺数据及原因：",
            *[f"\n- `{name}`：{reason}。" for name, reason in missing.items()],
            "\n与预计效果比较：已经定位权重表示误差与资源成本差异，但没有局部干预→logit/任务质量→恢复成本的证据，**未达到质量恢复与误差传播归因目标**。",
            "\n与单项通过标准比较：没有可解释且经独立验证的 mixed-precision policy；`final_policy=null`，**不通过**。没有消费 final-evaluation，也没有以全 FP16 回退冒充混合精度成功。",
            "\n前置链：E05-02 模型质量门失败、S04.5 M4 缺失、E05-03 无合法 calibration handoff、E05-04 无工业 artifact。解阻后应在独立 policy-validation 上完成 single/LOO/cumulative/pairwise、模型 KL/activation、质量 CI、三新进程与六 workload，冻结候选后才可打开 final。",
            "\n证据：[逐模块 JSONL](raw/stats/weight.jsonl)、[逐层统计](raw/stats/layers.json)、[判定](raw/verdict.json)、[manifest](raw/EVIDENCE_MANIFEST.json)。前端按 E05-05 verdict 自动展示，所有未执行项具有独立机器可读原因。",
            "\n复现：`./scripts/remote_run.sh python3 scripts/audit/run_e05_05_layer_sensitivity.py collect --output /tmp/e05-05-replay`。"]
    (output.parent / "E05-05_实验报告.md").write_text("\n".join(text) + "\n")
    manifest(output)
    print(f"E05-05 {len(values)} weight records; BLOCKED model intervention", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "summarize"))
    parser.add_argument("--model", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument("--output")
    args = parser.parse_args()
    {"collect": collect, "summarize": summarize}[args.mode](args)
