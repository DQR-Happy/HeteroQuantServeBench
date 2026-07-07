#!/usr/bin/env python3
"""Generate and check ``configs/experimental/*.yaml`` (the frozen S14 vocabularies).

The documents are *generated from code* so there is exactly one source of truth
for each vocabulary: the extra→feature mapping, the reason codes, the statuses,
the four candidate frontier branches and the statistics plan are all declared in
``hqsb.experimental.records`` and materialised here as YAML.

Two rules make the generated files safe to commit:

* **no measured number.**  Any field whose name suggests a measurement
  (``latency``, ``acceptance``, ``bytes``, ``memory`` …) carries the placeholder
  ``UNMEASURED``, because a config file is not allowed to become the hiding place
  for a fabricated result (任务硬规则 2).  ``hqsb.experimental.specs`` enforces it;
* **no machine-specific absolute path** (AGENTS.md §6).

Usage:
    scripts/experimental/gen_experimental_specs.py --write
    scripts/experimental/gen_experimental_specs.py --check
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml  # noqa: E402

from hqsb.experimental import frontier as frontier_mod  # noqa: E402
from hqsb.experimental import records as rec  # noqa: E402
from hqsb.experimental.specs import UNMEASURED  # noqa: E402

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "experimental")

#: Sources cited by every document, so the audit can check the ``sources`` list.
_PROTOCOL = "docs/stage_experiments/details/S14"
_MANUAL = "docs/stage_experiments/README.md"


def _sources(*names: str) -> List[str]:
    return [_MANUAL, *[f"{_PROTOCOL}/{name}" for name in names]]


def dependency_policy() -> Dict[str, Any]:
    """``E14-01`` steps 2–3: the layer each dependency belongs to."""
    layers: List[Dict[str, Any]] = [
        {"name": "build", "core": True, "purpose": "构建后端与构建期依赖"},
        {"name": "core", "core": True, "purpose": "稳定契约与配置；CPU-minimal 可导入"},
        {"name": "test", "core": True, "purpose": "测试与静态检查"},
        {"name": "doc", "core": True, "purpose": "文档与链接检查"},
        {"name": "train", "core": False, "purpose": "分布式训练与 checkpoint"},
        {"name": "rl", "core": False, "purpose": "rollout / reward / 后训练数据流"},
        {"name": "frontier", "core": False, "purpose": "speculative / MoE / 长上下文 / 稀疏"},
        {"name": "multimodal", "core": False, "purpose": "VLM / Diffusion / Audio（可选）"},
        {"name": "agent", "core": False, "purpose": "tool / memory / workflow trace（可选）"},
        {"name": "edge", "core": False, "purpose": "端侧 runtime adapter（可选）"},
    ]
    rules = [
        {
            "name": "pydantic",
            "layer": "core",
            "owner": "hqsb.core",
            "purpose": "C1–C7 版本化契约",
            "license": "MIT",
            "platforms": ["any"],
            "allowed_import_layer": "module_level",
            "version_constraint": ">=2.0",
        },
        {
            "name": "PyYAML",
            "layer": "core",
            "owner": "hqsb.core",
            "purpose": "配置与冻结词汇表加载",
            "license": "MIT",
            "platforms": ["any"],
            "allowed_import_layer": "module_level",
            "version_constraint": ">=5.4",
        },
        {
            "name": "torch",
            "layer": "train",
            "owner": "hqsb.experimental.training",
            "purpose": "训练后端（函数内惰性探测）",
            "license": "BSD-3-Clause",
            "platforms": ["linux", "aarch64"],
            "allowed_import_layer": "function_level",
            "extra": "train",
            "version_constraint": f"{UNMEASURED}-pinned-by-lock",
        },
        {
            "name": "torch.distributed",
            "layer": "train",
            "owner": "hqsb.experimental.training",
            "purpose": "DDP/FSDP/ZeRO 进程组",
            "license": "BSD-3-Clause",
            "platforms": ["linux"],
            "allowed_import_layer": "probe_only",
            "extra": "train",
        },
        {
            "name": "transformers",
            "layer": "rl",
            "owner": "hqsb.experimental.posttraining",
            "purpose": "tokenizer / 模型定义（惰性）",
            "license": "Apache-2.0",
            "platforms": ["any"],
            "allowed_import_layer": "function_level",
            "extra": "rl",
        },
        {
            "name": "ray",
            "layer": "rl",
            "owner": "hqsb.experimental.posttraining",
            "purpose": "rollout worker 资源放置",
            "license": "Apache-2.0",
            "platforms": ["linux"],
            "allowed_import_layer": "probe_only",
            "extra": "rl",
        },
        {
            "name": "vllm",
            "layer": "rl",
            "owner": "hqsb.experimental.posttraining",
            "purpose": "rollout engine（实验性）",
            "license": "Apache-2.0",
            "platforms": ["linux"],
            "allowed_import_layer": "probe_only",
            "extra": "rl",
        },
        {
            "name": "cusparselt",
            "layer": "frontier",
            "owner": "hqsb.experimental.sparsity",
            "purpose": "结构化稀疏 GEMM 库探测",
            "license": "NVIDIA Software License",
            "platforms": ["linux"],
            "allowed_import_layer": "probe_only",
            "extra": "frontier",
        },
        {
            "name": "torchvision",
            "layer": "multimodal",
            "owner": "hqsb.experimental.multimodal",
            "purpose": "图像预处理（可选）",
            "license": "BSD-3-Clause",
            "platforms": ["any"],
            "allowed_import_layer": "function_level",
            "extra": "multimodal",
        },
        {
            "name": "httpx",
            "layer": "agent",
            "owner": "hqsb.experimental.agent",
            "purpose": "受控工具沙箱的 HTTP 客户端",
            "license": "BSD-3-Clause",
            "platforms": ["any"],
            "allowed_import_layer": "function_level",
            "extra": "agent",
        },
    ]
    return {
        "kind": "dependency-policy",
        "version": "v1",
        "sources": _sources("E14-01_optional_dependencies_feature_isolation.md"),
        "note": "冻结词汇表：不含任何测量值；实际版本由各 extra 的 lock 记录",
        "layers": layers,
        "rules": rules,
    }


def extras() -> Dict[str, Any]:
    """``E14-01`` step 3: one row per extra, mapped to its feature flags."""
    rows = [
        {
            "name": extra,
            "feature_flags": list(capabilities),
            "install_profile": extra,
            "registry_capability": f"extra:{extra}",
            "cli_subcommand": f"run_e14.py --smoke --experiment {_experiment_for(extra)}",
            "test_job": f"tests/unit/experimental (profile={extra})",
        }
        for extra, capabilities in sorted(rec.EXTRA_FEATURE_MAPPING.items())
    ]
    return {
        "kind": "extras",
        "version": "v1",
        "sources": _sources("E14-01_optional_dependencies_feature_isolation.md"),
        "note": "extra↔feature flag↔capability↔CLI↔测试 job 的一一映射；代码侧为 records.EXTRA_FEATURE_MAPPING",
        "extras": rows,
    }


def _experiment_for(extra: str) -> str:
    return {
        "train": "E14-02",
        "rl": "E14-04",
        "frontier": "E14-05",
        "multimodal": "E14-06",
        "agent": "E14-07",
        "edge": "E14-08",
    }[extra]


def failure_semantics() -> Dict[str, Any]:
    """``E14-01`` step 4 + ``details/S14/README.md`` §21."""
    codes = [
        {"code": code, "meaning": _reason_meaning(code), "fallback_is_silent_degradation": True}
        for code in rec.REASON_CODES
    ]
    return {
        "kind": "failure-semantics",
        "version": "v1",
        "sources": _sources("E14-01_optional_dependencies_feature_isolation.md", "README.md"),
        "note": "缺依赖与实现错误不得混为一类；任何 fallback 必须记录 requested/actual/reason",
        "reason_codes": codes,
        "statuses": list(rec.S14_FAILURE_STATUSES),
        "capability_states": list(rec.CAPABILITY_STATES),
        "blocking_capability_states": list(rec.BLOCKING_CAPABILITY_STATES),
        "status_semantics": {
            "PASS_NEGATIVE": "仅用于性能/策略假设被严谨否定；安全/兼容/正确性/质量门不允许用负结果替代",
            "N/A_BY_ADR": "只有阶段范围经 ADR 正式缩减时才可记，且必须同步缩小能力声明",
            "BLOCKED": "前置未满足、硬件缺失、依赖不可用一律记 BLOCKED，不得写成 PASS",
        },
    }


def _reason_meaning(code: str) -> str:
    return {
        "DISABLED_BY_CONFIG": "feature flag 关闭；不得加载实现",
        "DEPENDENCY_ABSENT": "可选依赖缺失；结构化不可用，不是 ImportError 泄漏",
        "VERSION_UNSUPPORTED": "版本不在预注册范围内",
        "ABI_MISMATCH": "ABI/构建二进制不兼容；必须与 dependency absent 区分",
        "DEVICE_UNAVAILABLE": "设备缺失或数量不足",
        "SHAPE_UNSUPPORTED": "shape 不受支持；需显式拒绝或 reasoned fallback",
        "DTYPE_UNSUPPORTED": "dtype 不受支持",
        "PATTERN_UNSUPPORTED": "稀疏 pattern 不满足硬件契约",
        "CAPACITY_EXCEEDED": "容量/内存边界超限",
        "QUALITY_GATE_FAILED": "质量门未过；性能资格被取消",
        "IMPLEMENTATION_ERROR": "实现缺陷；不得计入 capability 不可用",
        "NOT_IMPLEMENTED": "尚未实现",
        "UNKNOWN_FEATURE_FLAG": "未知 flag；必须严格拒绝",
        "CONFLICTING_FEATURE_FLAGS": "互斥 flag 同时开启；必须拒绝",
    }[code]


def environment_matrix() -> Dict[str, Any]:
    """``E14-01`` step 5: the preregistered environment matrix."""
    profiles = [
        {"profile": "core", "installs": ["core"], "expects": "import/CLI/schema/CPU reference 全部可用"},
        {"profile": "train", "installs": ["core", "train"], "expects": "训练能力 AVAILABLE 或结构化 BLOCKED"},
        {"profile": "rl", "installs": ["core", "rl"], "expects": "rollout 能力 AVAILABLE 或结构化 BLOCKED"},
        {"profile": "frontier", "installs": ["core", "frontier"], "expects": "四分支 capability 粒度可分辨"},
        {"profile": "multimodal", "installs": ["core", "multimodal"], "expects": "不污染 core"},
        {"profile": "agent", "installs": ["core", "agent"], "expects": "不产生全局 monkey patch"},
        {"profile": "edge", "installs": ["core", "edge"], "expects": "x86 core 不被移动端 wheel 阻断"},
        {"profile": "all", "installs": sorted(rec.EXTRA_FEATURE_MAPPING), "expects": "组合可解析，feature 全关时等价 core"},
    ]
    environments = [
        {"environment": "cpu_minimal", "device": "none", "note": "清洁 CPU，用于 core 安装与导入追踪"},
        {"environment": "cuda_dev", "device": "gpu", "note": "开发机；结论标注 development"},
        {"environment": "target_device", "device": "gpu_or_npu", "note": "正式实验设备；raw 数据独立保留"},
    ]
    negatives = [
        {"case": "N1-direct-import-leak", "injection": "core 模块直接 import experimental 包"},
        {"case": "N2-indirect-reexport-leak", "injection": "包 __init__ 重导出 experimental 实现"},
        {"case": "N3-unknown-flag", "injection": "拼写错误的 feature flag"},
        {"case": "N4-conflicting-flags", "injection": "两个前沿机制 flag 同时开启"},
        {"case": "N5-missing-dependency", "injection": "隔离环境卸除一个关键包"},
        {"case": "N6-abi-mismatch", "injection": "预注册的不兼容版本组合"},
        {"case": "N7-metadata-leak", "injection": "wheel 把 experimental 依赖写成无条件 Requires-Dist"},
        {"case": "N8-import-side-effect", "injection": "import 时创建 device context / 启动 worker"},
        {"case": "N9-cli-full-import", "injection": "CLI 顶层 import 全部实现"},
        {"case": "N10-false-skip", "injection": "缺 GPU 的真实 import bug 被标 skip"},
    ]
    return {
        "kind": "environment-matrix",
        "version": "v1",
        "sources": _sources("E14-01_optional_dependencies_feature_isolation.md"),
        "note": "矩阵是预注册输入；各组实测结果必须落在 artifacts/S14 下，本文件不存测量值",
        "profiles": profiles,
        "environments": environments,
        "negative_cases": negatives,
        "import_purity_budget": {
            "device_context_created": False,
            "child_process_count": 0,
            "network_access": False,
            "cache_written": False,
        },
    }


def frontier_branches() -> Dict[str, Any]:
    """``E14-05`` steps 4–16: the four candidates with their gates."""
    details = {
        "E14-F1": {
            "name": "Speculative/MTP",
            "gain": "减少 target 串行 decode 次数（acceptance 只是中介变量）",
            "estimand": "在固定 target/quality/arrival 下，对 SLO-qualified output-token goodput 的因果变化",
            "quality_oracle": "greedy 逐 token parity，或声明的采样分布检验",
            "minimal_change": "proposer + accept/residual 路径，其余 runtime 不变",
            "rejection": ["quality fail", "acceptance 高但 break-even 未过", "额外设备/内存未计入"],
        },
        "E14-F2": {
            "name": "MoE",
            "gain": "少量 active expert 获得大参数容量",
            "estimand": "在固定 MoE artifact/topology 下，对 layer/request 尾延迟与 SLO goodput 的因果变化",
            "quality_oracle": "router/expert 输出 parity + 任务质量（drop/reroute 零容忍或预算内）",
            "minimal_change": "一个 placement/cache/quant/load-balance 策略，路由与权重不变",
            "rejection": ["active parameters 代替总显存", "token 丢/重/乱序", "热点尾延迟无保护"],
        },
        "E14-F3": {
            "name": "Long context",
            "gain": "降低 KV/attention 资源并扩展可服务容量",
            "estimand": "在固定模型/输入下，对真实 max context×concurrency 与 TTFT/TPOT 的因果变化",
            "quality_oracle": "长距离 needle/长文 QA 按长度×深度分层，保留短上下文回归",
            "minimal_change": "KV quant/compress/evict 或 chunked prefill 之一",
            "rejection": ["截断/滑窗冒充完整上下文", "短 perplexity 代替长文质量", "长请求饿死短请求"],
        },
        "E14-F4": {
            "name": "Sparsity",
            "gain": "减少有效计算/IO（理论 FLOPs 不是执行稀疏度）",
            "estimand": "在同质量 dense baseline 下，对 kernel→model 延迟与显存的影响",
            "quality_oracle": "路线 A 用 logits/token/任务质量；路线 B 增加长距离检索/QA",
            "minimal_change": "一个 sparse transform + 一个受支持内核路径",
            "rejection": ["pattern 不合规", "actual sparse kernel 未命中", "metadata/转换成本漏计"],
        },
    }
    branches = []
    for branch, risk in sorted(rec.FRONTIER_PRIMARY_RISKS.items()):
        detail = details[branch]
        branches.append(
            {
                "branch": branch,
                "name": detail["name"],
                "primary_gain_hypothesis": detail["gain"],
                "primary_risk": risk,
                "required_layers": list(rec.FRONTIER_REQUIRED_LAYERS[branch]),
                "hard_prerequisites": [
                    "model_licence",
                    "data_licence",
                    "device_capability",
                    "quality_oracle",
                    "source_access",
                    "profile_tooling",
                    "isolation",
                ],
                "quality_oracle": detail["quality_oracle"],
                "minimal_independent_change": detail["minimal_change"],
                "candidate_estimand": detail["estimand"],
                "budget": {dimension: UNMEASURED for dimension in sorted(rec.EXPERIMENT_DEPENDENCIES)},
                "stop_rules": [
                    "oom",
                    "quality_fail",
                    "slo_error_budget",
                    "thermal",
                    "data_leak",
                    "numerical_anomaly",
                    "resource_cap",
                ],
                "rejection_conditions": detail["rejection"],
            }
        )
    return {
        "kind": "frontier-branches",
        "version": "v1",
        "sources": _sources("E14-05_frontier_adr_hypothesis_preregistration.md", "README.md"),
        "selection_rules": [
            "只允许一个主分支；其余标 N/A_BY_ADR 并记录重开条件",
            "硬前置任一不明 ⇒ 该分支 BLOCKED，不因沉没成本继续",
            "论文/官方数字只用于构造假设，不作为 HQSB baseline",
            "exploration 预算与 holdout 必须在使用前冻结",
        ],
        "branches": branches,
    }


def adoption_rules() -> Dict[str, Any]:
    return {
        "kind": "adoption-rules",
        "version": "v1",
        "sources": _sources("E14-05_frontier_adr_hypothesis_preregistration.md", "README.md"),
        "note": "采用是多目标决策；不允许事后把某个指标变好当作成功",
        "rules": [
            {"decision": rec.ADOPT_CORE, "requires": "通过 S13/S15 回归，maturity=ADOPTED_CORE"},
            {"decision": rec.ADOPT_EXPERIMENTAL, "requires": "质量门通过且收益限定于预注册 workload"},
            {"decision": rec.RESEARCH_ONLY, "requires": "机制可解释但工程代价或适用域过窄"},
            {"decision": rec.REJECT_NO_BENEFIT, "requires": "门禁完整且收益无法覆盖成本（需 raw 证据）"},
            {"decision": rec.REJECT_QUALITY, "requires": "质量门失败（优先于任何性能结论）"},
            {"decision": rec.REJECT_COMPLEXITY, "requires": "维护/依赖/升级成本超出采用预算"},
            {"decision": rec.REJECT_PORTABILITY, "requires": "仅特定硬件/shape 可用且不可迁移"},
            {"decision": rec.BLOCKED_EVIDENCE, "requires": "证据不足，不得写成负结论"},
            {"decision": rec.N_A_BY_ADR, "requires": "阶段范围经 ADR 正式缩减，并同步缩小能力声明"},
        ],
        "outcomes": {
            "quality_fail": rec.REJECT_QUALITY,
            "no_benefit": rec.REJECT_NO_BENEFIT,
            "benefit_specific_workload_only": rec.ADOPT_EXPERIMENTAL,
            "not_portable": rec.REJECT_PORTABILITY,
            "engineering_cost_too_high": rec.REJECT_COMPLEXITY,
            "benefit_confirmed": rec.ADOPT_EXPERIMENTAL,
        },
    }


def statistics_plan() -> Dict[str, Any]:
    return {
        "kind": "statistics-plan",
        "version": "v1",
        "sources": [_MANUAL, f"{_PROTOCOL}/README.md"],
        "note": "单位与重复在结果前冻结；token/step 不是独立重复",
        "units": [{"unit": unit, "definition": definition} for unit, definition in sorted(rec.EXPERIMENT_UNITS.items())],
        "tests": [
            {"name": "paired_bootstrap", "use": "latency/quality paired 比较的置信区间"},
            {"name": "median_and_percentiles", "use": "P50/P95/P99 与离散度"},
            {"name": "holdout_confirmation", "use": "冻结配置后在未参与调参的数据上一次性确认"},
        ],
        "exclusions": [{"category": category, "rule": "记录而不是删除"} for category in frontier_mod.INVALIDITY_CATEGORIES],
        "guard_band_policy": "guard band 至少覆盖基线噪声；不得事后改阈值",
        "repeats": {
            "training": UNMEASURED,
            "conversion": UNMEASURED,
            "rollout": UNMEASURED,
            "runtime_service": UNMEASURED,
            "distributed": UNMEASURED,
            "edge_energy": UNMEASURED,
        },
    }


def profiling_contract() -> Dict[str, Any]:
    return {
        "kind": "profiling-contract",
        "version": "v1",
        "sources": ["docs/stage_experiments/details/S14/README.md"],
        "note": "至少两层；上层现象必须能追到下层原因",
        "layers": [
            {"layer": layer, "required_fields": list(rec.PROFILE_LAYER_FIELDS[layer])}
            for layer in rec.PROFILE_LAYERS
        ],
        "actual_path_fields": [
            "requested_implementation",
            "actual_implementation",
            "actual_backend",
            "actual_kernel",
            "graph_or_eager",
            "dtype_layout_shape",
            "rank_device_topology",
            "compile_cache_hit",
            "feature_flags",
            "capability_reason",
        ],
        "resource_ledger_keys": [
            "device_memory_allocated_bytes",
            "device_memory_reserved_bytes",
            "device_memory_peak_bytes",
            "host_memory_bytes",
            "pinned_memory_bytes",
            "kv_or_cache_bytes",
            "collective_bytes",
            "power_w",
            "energy_j",
            "device_count",
        ],
        "timeline_stages": [
            "training_step", "rollout_request", "trajectory", "weight_publish", "weight_load",
            "weight_active", "runtime_iteration", "kernel", "collective", "service_request",
            "service_stream",
        ],
    }


#: kind → generator.  The order fixes the file names.
DOCUMENTS: Tuple[Tuple[str, Any], ...] = (
    ("dependency-policy", dependency_policy),
    ("extras", extras),
    ("failure-semantics", failure_semantics),
    ("environment-matrix", environment_matrix),
    ("frontier-branches", frontier_branches),
    ("adoption-rules", adoption_rules),
    ("statistics-plan", statistics_plan),
    ("profiling-contract", profiling_contract),
)


def _render(payload: Dict[str, Any]) -> str:
    header = (
        "# GENERATED by scripts/experimental/gen_experimental_specs.py — do not edit by hand.\n"
        "# 冻结词汇表：不含任何测量值；实测结果一律落在 artifacts/S14/ 下。\n"
    )
    return header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)


def rendered_documents() -> Dict[str, str]:
    return {f"{kind}.yaml": _render(builder()) for kind, builder in DOCUMENTS}


def write_documents(target: str = SPEC_DIR) -> List[str]:
    os.makedirs(target, exist_ok=True)
    written: List[str] = []
    for name, text in sorted(rendered_documents().items()):
        path = os.path.join(target, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        written.append(path)
    return written


def check_documents(target: str = SPEC_DIR) -> Tuple[bool, List[str]]:
    """Regenerate in memory and compare; drift is reported, never auto-fixed silently."""
    drifted: List[str] = []
    for name, text in sorted(rendered_documents().items()):
        path = os.path.join(target, name)
        if not os.path.exists(path):
            drifted.append(f"missing: {name}")
            continue
        with open(path, encoding="utf-8") as handle:
            if handle.read() != text:
                drifted.append(f"drifted: {name}")
    return (not drifted), drifted


def main() -> int:
    parser = argparse.ArgumentParser(description="generate or check configs/experimental")
    parser.add_argument("--write", action="store_true", help="write the documents")
    parser.add_argument("--check", action="store_true", help="check for drift (default)")
    parser.add_argument("--spec-dir", default=SPEC_DIR)
    args = parser.parse_args()
    if args.write:
        for path in write_documents(args.spec_dir):
            print(f"wrote {path}")
        return 0
    ok, drifted = check_documents(args.spec_dir)
    print(f"spec_dir: {args.spec_dir}")
    print(f"ok: {ok}")
    for item in drifted:
        print(f"  {item}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
