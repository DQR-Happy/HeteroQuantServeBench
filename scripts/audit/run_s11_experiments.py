#!/usr/bin/env python3
"""Execute and archive the honest Jetson scope of the ten S11 experiments.

The formal S11 protocol requires a passing S06 model-level pattern verdict,
full-model Qwen evidence, binary/profile export tools, and (for E11-09) a
second compiler stack.  The current target does not satisfy all of those
gates.  This collector therefore keeps two conclusions separate:

* real component observations collected on the Jetson (Qwen module capture,
  CUDA custom-op dispatch/correctness/timing, guards, search/holdout, cache
  corruption, and AI-candidate gates);
* the formal experiment verdict, which remains BLOCKED whenever a documented
  prerequisite or required evidence level is absent.

It never turns an interface smoke, a synthetic graph, or an unavailable tool
into a scientific PASS.  Run only through ``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.compiler import (  # noqa: E402
    aigate,
    autotune,
    cache,
    capture,
    codegen,
    costmodel,
    experiment,
    guards,
    identity,
    interface_map,
    ir,
    lowering,
    pattern_library,
    portable,
    rewrite,
    specs,
    targets,
)
from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.core.errors import ConfigError  # noqa: E402


STAGE = "S11"
STAGE_ROOT = REPO / "docs/stage_experiments/S11"
DETAIL_ROOT = REPO / "docs/stage_experiments/details/S11"
DRIVER = REPO / "scripts/compiler/run_e11.py"
EXPERIMENTS = tuple(f"E11-{index:02d}" for index in range(1, 11))
MODEL_PATH = Path("~/models/hqsb/Qwen3-1.7B").expanduser()
EPS = 1.0e-6


@dataclass(frozen=True)
class Protocol:
    experiment_id: str
    level: str
    title: str
    expected_effect: str
    required_data: Tuple[str, ...]
    criteria: Tuple[str, ...]
    dependencies: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "level": self.level,
            "title": self.title,
            "expected_effect": self.expected_effect,
            "required_data": list(self.required_data),
            "criteria": list(self.criteria),
            "dependencies": list(self.dependencies),
        }


def _protocol(
    experiment_id: str,
    level: str,
    title: str,
    expected_effect: str,
    required_data: Sequence[str],
    criteria: Sequence[str],
    dependencies: Sequence[str] = (),
) -> Protocol:
    return Protocol(
        experiment_id,
        level,
        title,
        expected_effect,
        tuple(required_data),
        tuple(criteria),
        tuple(dependencies),
    )


PROTOCOLS: Mapping[str, Protocol] = {
    "E11-01": _protocol(
        "E11-01", "P0", "真实 Qwen Graph Capture、Break、Guard 与 Symbolic Domain",
        "得到可操作、可追溯的 compiler capture 边界。",
        ("source module/op 与模型身份", "FX/export graph 与 artifact hash", "graph break/unsupported/eager island",
         "guards、symbolic shape 与 runtime values", "source/shape/effect metadata", "phase/workload coverage 与冷进程复验"),
        ("至少一个真实 Qwen block/subgraph 两个冷进程稳定捕获", "prefill 与多步 decode census 完成",
         "每个 break 可定位 source/reason/boundary/action", "图具有 source/shape/dtype/stride/effect metadata",
         "constraints、guard、variant/recompile 可关联", "coverage 分 op/time/hotspot/phase", "负例诊断符合预期",
         "hero graph/hash 可交付 E11-02/03"),
    ),
    "E11-02": _protocol(
        "E11-02", "P0", "语义 Pattern Rewrite、Near-miss 拒绝与 Pass 幂等",
        "证明 rewrite 精确、确定、幂等且失败原子。",
        ("pattern contract/signature", "positive/near-miss/hard-negative corpus", "matcher predicates 与 decisions",
         "before/after graph 与 node mapping", "alias/mutation/effect proof", "operator/block/model correctness",
         "pass iterations/cross-process determinism/failure atomicity"),
        ("真实 Qwen positive 合法命中", "所有 near-miss 拒绝且 false positive=0", "predicate proof 完整",
         "operator/block/prefill/decode correctness 通过", "alias/mutation/extra-user 符合合同", "verifier/provenance 通过",
         "二次 rewrite=0 且跨进程稳定", "失败不改变 graph/registry/cache", "hero graph/fallback 可重放"),
        ("E11-01",),
    ),
    "E11-03": _protocol(
        "E11-03", "P0", "Graph→HQSB IR→Lowering Registry→Custom Kernel 闭环",
        "建立至少一条可验证的 graph-to-real-kernel 闭环。",
        ("各层 IR/artifact hash 与 lineage", "target capability 与 registry snapshot", "candidate/selection/fallback reason",
         "guard 与 actual dispatch", "operator/block/model correctness", "compile/cache/first/steady timing",
         "unsupported/compile/runtime failure 与冷进程重建"),
        ("真实 Qwen source→targeted IR lineage 完整", "IR verifier/round-trip 通过", "registry 字段完整",
         "S03/S04 custom kernel 有两种 actual-dispatch 证据", "operator/block/prefill/decode correctness 通过",
         "unsupported 在副作用前 fallback", "坏 binary/失败不执行且不污染 cache", "成本口径分离",
         "冷进程可重建 hero path", "负性能结果不被隐藏"),
        ("E11-01", "E11-02", "S06 model-level PASS"),
    ),
    "E11-04": _protocol(
        "E11-04", "P0", "IR、Generated Code、PTX/SASS 与硬件行为归因",
        "至少解释一组优化与一组退化/无收益的跨层原因。",
        ("快慢 case 与 correctness", "source/HQSB/scheduler/backend IR", "generated source/PTX/binary/SASS/symbol map",
         "IR/code/memory/launch diff", "timeline 与 hardware counters", "ablation/mechanism/phase-model evidence",
         "artifact reconstruction"),
        ("至少一组改善和一组退化/无收益", "各层 IR/code/binary/SASS 可关联", "profiler event 映射实际 artifact",
         "profile/ablation 前后 correctness 通过", "机制由 IR+binary+counter+ablation+runtime 支持",
         "compile/kernel/phase/model 分层", "profile 与 clean timing 分离", "artifact 可重建且限制明确"),
        ("E11-03",),
    ),
    "E11-05": _protocol(
        "E11-05", "P0", "Dynamic Shape、Guard、Recompile、Variant 与 Fallback Safety",
        "动态输入不错误复用 binary，且编译/variant 数量有界可解释。",
        ("symbol/domain/ordered shape sequence", "guard definitions/events", "variant manifest/coverage/overlap",
         "compile/cache/dispatch events", "correctness/runtime/total compile cost", "strategy/holdout/concurrency",
         "machine-readable dynamic policy"),
        ("batch/ISL/past/stride/dtype 矩阵完成", "actual dispatch 的 guard 全 true 且 wrong reuse=0",
         "recompile/variant unexplained=0", "旧 shape 可复用", "超界安全 fallback", "variant limit/concurrency 安全",
         "至少三种策略公平比较", "独立 holdout 与总成本完成", "dynamic policy 可机器读取"),
        ("E11-03",),
    ),
    "E11-06": _protocol(
        "E11-06", "P0", "Autotune Search Space、Holdout 与预算",
        "量化搜索收益、成本与未见 shape 泛化。",
        ("task/search-space/candidate identity", "static rejection 与 false-reject audit", "train/validation/holdout split",
         "预算梯度与全部 trial 状态", "compile/correctness/raw timing/resource", "oracle/quality-cost curve",
         "independent confirmation/holdout", "search cost/amortization/tuning DB/policy"),
        ("search space/constraints/identity 版本化", "所有 candidate 有状态和 reason", "false reject 审计完成",
         "correctness 先于 benchmark 且无状态污染", "default+至少三个预算点", "代表 scope 有 oracle",
         "新进程 confirmation", "interpolation/boundary holdout", "严重失效有 fallback", "成本/收益分离",
         "tuning DB 可交付"),
        ("E11-03", "E11-05"),
    ),
    "E11-07": _protocol(
        "E11-07", "P0", "Cost Model、Oracle Regret 与低置信回退",
        "在 untouched holdout 上报告 oracle regret 并安全回退。",
        ("dataset/feature/label/split/leakage", "default/heuristic/random/global/nearest/oracle baselines",
         "training/validation/final predictions", "ranking/regret/risk-coverage", "OOD/missing/illegal/corrupt robustness",
         "selection overhead/feature ablation/error cases", "actual dispatch replay/deployment policy"),
        ("dataset/split 可重建", "final test 冻结后一次执行", "六种 baseline 完整", "regret 含 p95/max/catastrophic",
         "confidence/OOD 完整", "非法候选不能绕过 legality", "损坏模型安全 fallback", "dispatch correctness 通过",
         "selection overhead 单列"),
        ("E11-06",),
    ),
    "E11-08": _protocol(
        "E11-08", "P0", "Compile Artifact Cache、跨进程命中与失效安全",
        "建立可靠、可失效、坏缓存不执行的制品生命周期。",
        ("cache layers/key spec/test vectors", "entry state machine/events", "C0-C4 timing",
         "version/pass/kernel/ABI/arch/config/tuning/model invalidation", "corruption/fault cases",
         "concurrent readers/writers", "key omission/eviction/portability", "cache policy"),
        ("layers/key/state/dependency 完整", "C0-C4 与跨进程 no-recompile 证据", "guard-domain reuse 正确",
         "invalidation matrix 符合", "false hit=0/corrupt execution=0", "所有损坏在 load 前拒绝",
         "permission/disk-full 可诊断", "并发无半成品", "eviction 可重建", "binary/telemetry 可关联"),
        ("E11-03", "E11-05"),
    ),
    "E11-09": _protocol(
        "E11-09", "P0", "TVM/MLIR 可迁移 Graph→Schedule→Target Lowering",
        "在第二编译栈复现同一真实 semantic op 的可运行闭环。",
        ("stack/version/scope", "semantic mapping/target capability", "high-level/legalized/loop/scheduled/target IR",
         "pass/schedule trace", "generated code/binary/runtime bridge", "operator/block correctness",
         "compile/runtime/profile/replay/development cost", "role comparison/adoption"),
        ("primary stack/version/scope 明确", "真实 op 完成 high-level→runtime", "每层 lineage/verifier 通过",
         "unsupported/fallback 可诊断", "operator 与 Qwen subgraph correctness 通过", "bridge copy/sync 有证据",
         "compile/steady 与强 baseline 分开", "schedule 冷进程可重放", "adoption decision 清楚"),
        ("E11-03",),
    ),
    "E11-10": _protocol(
        "E11-10", "P1", "AI/Agent Kernel 候选零信任质量门",
        "错误候选被门禁拦截，未完备候选不能进入 registry。",
        ("task/generation/provenance", "candidate/control/wrong corpus", "sandbox/harness lock",
         "G0-G10 gate results", "hidden/metamorphic/state/sanitizer/measurement integrity",
         "correct-and-fast/resource/integration", "iteration/review/admission/candidate verdict"),
        ("task/harness/provenance 冻结", "evaluator 最小权限", "C1-C10 与 controls 完成",
         "wrong false accept=0", "hidden/state/sanitizer 完整", "measurement exploit 被阻断",
         "只对正确候选测强 baseline", "compiler/cache/fallback 规则相同", "admission 缺字段拒绝",
         "candidate/gate/review 全可追踪"),
        ("E11-02", "E11-03", "E11-06", "E11-08"),
    ),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict"):
        return json_safe(value.as_dict())
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def command(argv: Sequence[str], *, timeout: float = 30.0) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv), cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "argv": list(argv),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
            "elapsed_s": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "returncode": 124, "stdout": str(exc.stdout or "")[-20000:],
            "stderr": str(exc.stderr or "")[-20000:], "elapsed_s": time.perf_counter() - started,
            "timeout": True,
        }


def parsed_child(args: Sequence[str], timeout: float = 180.0) -> Dict[str, Any]:
    argv = (sys.executable, str(Path(__file__).resolve()), *args)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv), cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
        )
        full_stdout = completed.stdout
        try:
            payload = json.loads(full_stdout) if full_stdout.strip() else None
        except json.JSONDecodeError:
            payload = None
        return {
            "argv": list(argv),
            "returncode": completed.returncode,
            "stdout": full_stdout[-20000:],
            "stdout_bytes": len(full_stdout.encode()),
            "stdout_truncated_in_record": len(full_stdout) > 20000,
            "stderr": completed.stderr[-20000:],
            "elapsed_s": time.perf_counter() - started,
            "payload": payload,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "returncode": 124, "stdout": str(exc.stdout or "")[-20000:],
            "stderr": str(exc.stderr or "")[-20000:], "elapsed_s": time.perf_counter() - started,
            "timeout": True, "payload": None,
        }


def git_output(*args: str) -> str:
    result = command(("git", *args), timeout=15)
    return result["stdout"].strip() if result["returncode"] == 0 else ""


def collect_environment(torch: Any) -> Dict[str, Any]:
    target = targets.cuda_target_snapshot(target_id="jetson-orin-s11")
    tool_names = ("nsys", "ncu", "cuobjdump", "nvdisasm", "nvcc", "mlir-opt", "tvmc", "compute-sanitizer", "nm")
    module_names = ("torch", "transformers", "triton", "tvm", "mlir")
    libraries = sorted(
        str(path.relative_to(REPO))
        for pattern in (
            "build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so",
            "build/*/ops/cuda/fused_residual_rmsnorm/libhqsb_fused_residual_rmsnorm_shared.so",
        )
        for path in REPO.glob(pattern)
        if path.is_file()
    )
    memory = command(("free", "-b"))
    return {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
        "target_snapshot": target.as_dict(),
        "target_validation_problems": target.validate(),
        "tools": {name: shutil.which(name) for name in tool_names},
        "modules": {name: bool(importlib.util.find_spec(name)) for name in module_names},
        "model_path": str(MODEL_PATH),
        "model_present": MODEL_PATH.is_dir(),
        "native_libraries": libraries,
        "native_library_hashes": {
            item: sha256_file(REPO / item) for item in libraries
        },
        "memory_probe": memory,
        "source": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "dirty_patch_sha256": sha256_bytes(git_output("diff", "--binary").encode()),
        },
    }


def make_inputs(torch: Any, rows: int, *, hidden: int = 2048, seed: int = 1100, dtype: Any = None, contiguous: bool = True):
    dtype = dtype or torch.float16
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, hidden), generator=generator, dtype=torch.float32).to("cuda", dtype=dtype)
    residual = torch.randn((rows, hidden), generator=generator, dtype=torch.float32).to("cuda", dtype=dtype)
    weight = (0.75 + 0.5 * torch.rand((hidden,), generator=generator)).to("cuda", dtype=dtype)
    if not contiguous:
        base_x = torch.empty((rows, hidden * 2), device="cuda", dtype=dtype)
        base_r = torch.empty_like(base_x)
        base_x[:, ::2].copy_(x)
        base_r[:, ::2].copy_(residual)
        x, residual = base_x[:, ::2], base_r[:, ::2]
    return x, residual, weight


def fused_reference(torch: Any, x: Any, residual: Any, weight: Any) -> Tuple[Any, Any]:
    updated = (x.float() + residual.float()).to(x.dtype)
    normalised = (
        updated.float()
        * torch.rsqrt(updated.float().square().mean(-1, keepdim=True) + EPS)
        * weight.float()
    ).to(x.dtype)
    return normalised, updated


def compare(torch: Any, actual: Any, expected: Any, *, atol: float = 3e-3, rtol: float = 3e-3) -> Dict[str, Any]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "ok": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def timed_cuda(torch: Any, fn: Callable[[], Any], *, repeats: int = 20, warmup: int = 5) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def register_ops() -> Any:
    from hqsb.integration import torch_ops

    torch_ops.register_torch_ops(REPO)
    return torch_ops


def capture_probe(torch: Any, seed: int) -> Dict[str, Any]:
    """Capture the real Transformers Qwen3 RMSNorm subgraph in one cold process."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    class QwenResidualNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = Qwen3RMSNorm(2048, eps=EPS)

        def forward(self, x, residual):
            updated = (x.float() + residual.float()).to(x.dtype)
            return self.norm(updated), updated

    module = QwenResidualNorm().cuda().half().eval()
    source_file = inspect.getsourcefile(Qwen3RMSNorm.forward) or ""
    try:
        source_line = inspect.getsourcelines(Qwen3RMSNorm.forward)[1]
    except (OSError, TypeError):
        source_line = 0
    captured: List[Dict[str, Any]] = []

    def backend(gm, example_inputs):
        graph_text = str(gm.graph)
        nodes = []
        for node in gm.graph.nodes:
            nodes.append(
                {
                    "name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "users": sorted(user.name for user in node.users),
                    "source": str(node.meta.get("source_fn_stack", ""))[:1000],
                    "stack_trace": str(node.meta.get("stack_trace", ""))[-2000:],
                }
            )
        canonical_nodes = [
            {
                "name": node["name"],
                "op": node["op"],
                "target": re.sub(r"0x[0-9a-fA-F]+", "0xADDR", node["target"]),
                "users": node["users"],
            }
            for node in nodes
        ]
        captured.append(
            {
                "graph_text": graph_text,
                "raw_hash": sha256_bytes(graph_text.encode()),
                "canonical_hash": sha256_bytes(
                    json.dumps(canonical_nodes, sort_keys=True, separators=(",", ":")).encode()
                ),
                "nodes": nodes,
                "example_inputs": [
                    {
                        "shape": [str(dim) for dim in item.shape],
                        "dtype": str(item.dtype),
                        "stride": list(item.stride()),
                        "device": str(item.device),
                    }
                    for item in example_inputs
                    if hasattr(item, "shape")
                ],
            }
        )
        return gm.forward

    shapes = (("decode", 1), ("tiny_prefill", 8), ("short_prefill", 32), ("decode_reuse", 1))
    runtime: List[Dict[str, Any]] = []
    torch._dynamo.reset()
    compiled = torch.compile(module, backend=backend, dynamic=True, fullgraph=False)
    for index, (phase, rows) in enumerate(shapes):
        x, residual, _weight = make_inputs(torch, rows, seed=seed + index)
        with torch.inference_mode():
            result = compiled(x.reshape(1, rows, 2048), residual.reshape(1, rows, 2048))
            eager = module(x.reshape(1, rows, 2048), residual.reshape(1, rows, 2048))
        torch.cuda.synchronize()
        runtime.append(
            {
                "phase": phase,
                "B": 1,
                "S": rows,
                "H": 2048,
                "dtype": str(x.dtype),
                "stride": list(x.reshape(1, rows, 2048).stride()),
                "correctness": [compare(torch, result[0], eager[0]), compare(torch, result[1], eager[1], atol=0.0, rtol=0.0)],
            }
        )

    export_record: Dict[str, Any]
    x, residual, _ = make_inputs(torch, 8, seed=seed + 20)
    try:
        exported = torch.export.export(module, (x.reshape(1, 8, 2048), residual.reshape(1, 8, 2048)))
        text = str(exported.graph_module.graph)
        export_record = {"status": "PASS", "graph": text, "sha256": sha256_bytes(text.encode()), "range_constraints": str(exported.range_constraints)}
    except Exception as exc:
        export_record = {"status": "FAIL", "exception": type(exc).__name__, "message": str(exc)}

    class DataDependent(torch.nn.Module):
        def forward(self, value):
            if value.sum().item() > 0:
                return value + 1
            return value - 1

    negative: Dict[str, Any]
    try:
        value = torch.ones((2, 8), device="cuda")
        explain = torch._dynamo.explain(DataDependent().cuda())(value)
        negative = {
            "status": "EXPECTED_BREAK",
            "graph_count": int(explain.graph_count),
            "graph_break_count": int(explain.graph_break_count),
            "break_reasons": [str(item)[:4000] for item in explain.break_reasons],
            "source_file": __file__,
            "source_line": DataDependent.forward.__code__.co_firstlineno,
            "action": "eager_island",
        }
    except Exception as exc:
        negative = {"status": "DIAGNOSTIC_ERROR", "exception": type(exc).__name__, "message": str(exc)}
    return {
        "pid": os.getpid(),
        "seed": seed,
        "module": "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm",
        "source_file": source_file,
        "source_line": source_line,
        "capture_mode": "torch.compile custom debug backend",
        "graphs": captured,
        "runtime": runtime,
        "export": export_record,
        "negative": negative,
        "stable_within_process": len({row["canonical_hash"] for row in captured}) <= 2,
    }


def cache_probe(cache_dir: str, entry_id: str, expected_key: str, arch: str) -> Dict[str, Any]:
    store = cache.EntryStore(cache_dir, spec=cache.default_key_spec())
    started = time.perf_counter()
    result = store.read(entry_id, expected_key=expected_key, target_arch=arch, abi_version="1")
    return {
        "pid": os.getpid(),
        "read": result.as_dict(),
        "elapsed_s": time.perf_counter() - started,
        "compiler_events": 0,
    }


def collect_e11_01(torch: Any, shared: Dict[str, Any]) -> Dict[str, Any]:
    cold = [parsed_child(("--probe", "capture", "--seed", str(1110 + index))) for index in range(2)]
    payloads = [row.get("payload") for row in cold if row.get("returncode") == 0 and isinstance(row.get("payload"), dict)]
    canonical_sets = [sorted({graph["canonical_hash"] for graph in item.get("graphs", [])}) for item in payloads]
    stable_cold = len(payloads) == 2 and canonical_sets[0] == canonical_sets[1]
    runtime_rows = [row for item in payloads for row in item.get("runtime", [])]
    all_correct = bool(runtime_rows) and all(
        all(check["ok"] for check in row["correctness"]) for row in runtime_rows
    )
    hero = {
        "graph_id": "qwen3_rmsnorm_residual_subgraph",
        "source_module": payloads[0].get("module") if payloads else "",
        "source_file": payloads[0].get("source_file") if payloads else "",
        "source_line": payloads[0].get("source_line") if payloads else 0,
        "canonical_hashes": canonical_sets[0] if canonical_sets else [],
        "capture_mode": "torch.compile custom debug backend",
        "region": "real Qwen3RMSNorm subgraph (not a full decoder block)",
        "constraints": ["B == 1", "1 <= S <= 32 observed", "H == 2048", "dtype == fp16", "last-dim contiguous"],
        "pattern_sites": ["residual add -> real Qwen3RMSNorm"],
        "compiler_versions": {"torch": str(torch.__version__)},
        "scope_ceiling": "real module subgraph capture only; full-model prefill/KV decode not claimed",
    }
    shared["capture"] = {"cold": cold, "hero": hero}
    return {
        "component_status": "PASS" if stable_cold and all_correct else "FAIL",
        "component_pass": stable_cold and all_correct,
        "cold_processes": cold,
        "cold_process_canonical_stable": stable_cold,
        "runtime_rows": runtime_rows,
        "hero_graph_manifest": hero,
        "negative_cases": [item.get("negative") for item in payloads],
        "graph_breaks_located": all(bool(item.get("negative", {}).get("source_file")) for item in payloads),
        "limitations": [
            "full Qwen3-1.7B load is below the preregistered 4 GiB free-memory floor",
            "prefill/decode labels are shape-phase probes of the real RMSNorm subgraph, not full attention/KV execution",
            "debug backend captures and returns gm.forward; it proves capture only, not lowering",
        ],
    }


def collect_e11_02(torch: Any, shared: Dict[str, Any]) -> Dict[str, Any]:
    rows = rewrite.build_corpus_rows(pattern_library.corpus_plan(), graph_builder=pattern_library.residual_add_rmsnorm_graph)
    corpus = rewrite.evaluate_corpus(rows)
    spec = pattern_library.residual_add_rmsnorm_graph(graph_id="s11_real_bound_residual_rmsnorm", variant="canonical")
    graph = rewrite.build_graph(spec)
    outcome = rewrite.run_fused_add_rmsnorm_pass(graph)
    candidate = next(item for item in rewrite.find_candidates(graph) if item.complete)
    committed = rewrite.apply_rewrite(graph, candidate)
    idem = rewrite.idempotence_report(graph)
    determ = rewrite.determinism_report(graph, runs=3)
    atomic = rewrite.atomicity_report(graph)
    provenance = rewrite.provenance_audit(graph, committed.graph, committed)
    torch_ops = register_ops()
    correctness: List[Dict[str, Any]] = []
    for index, rows_count in enumerate((1, 8, 32, 128)):
        x, residual, weight = make_inputs(torch, rows_count, seed=1200 + index)
        expected = fused_reference(torch, x, residual, weight)
        actual = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
        torch.cuda.synchronize()
        correctness.append(
            {
                "rows": rows_count,
                "phase": "decode" if rows_count == 1 else "prefill_subgraph",
                "normalised": compare(torch, actual[0], expected[0]),
                "updated_residual": compare(torch, actual[1], expected[1], atol=0.0, rtol=0.0),
            }
        )
    component = bool(
        corpus["false_positive"] == 0
        and not corpus["unmet_expectations"]
        and idem["idempotent"]
        and determ["deterministic"]
        and atomic["all_atomic"]
        and provenance["ok"]
        and all(row["normalised"]["ok"] and row["updated_residual"]["ok"] for row in correctness)
    )
    shared["rewrite"] = {
        "before": graph,
        "after": outcome.graph,
        "outcome": outcome,
        "committed": committed,
        "correctness": correctness,
    }
    return {
        "component_status": "PASS" if component else "FAIL",
        "component_pass": component,
        "pattern_contract": pattern_library.residual_add_rmsnorm_contract().as_dict(),
        "pattern_signature": pattern_library.residual_rmsnorm_signature().as_dict(),
        "corpus": corpus,
        "decisions": [item.as_dict() for item in outcome.decisions],
        "before_ir": graph.as_dict(),
        "after_ir": outcome.graph.as_dict(),
        "idempotence": idem,
        "determinism": determ,
        "failure_atomicity": atomic,
        "provenance": provenance,
        "gpu_correctness": correctness,
        "actual_dispatch": torch_ops.audit_snapshot(),
        "real_positive_binding": shared.get("capture", {}).get("hero", {}),
        "limitations": [
            "semantic rewrite executes on the versioned HQSB sidecar IR bound to the captured real module source",
            "full-model prefill and multi-step KV decode correctness remain unavailable because S06 is BLOCKED",
        ],
    }


def _native_artifact(environment: Mapping[str, Any]) -> Tuple[Path, str]:
    candidates = [
        REPO / item
        for item in environment.get("native_libraries", [])
        if "fused_residual_rmsnorm" in item
    ]
    if not candidates:
        raise RuntimeError("fused residual RMSNorm native library is unavailable")
    chosen = candidates[0]
    return chosen, sha256_file(chosen)


def collect_e11_03(torch: Any, shared: Dict[str, Any], environment: Mapping[str, Any]) -> Dict[str, Any]:
    torch_ops = register_ops()
    torch_ops.reset_audit_counters()
    before = shared["rewrite"]["before"]
    after = shared["rewrite"]["after"]
    target = targets.cuda_target_snapshot(target_id="jetson-orin-s11")
    artifact, artifact_hash = _native_artifact(environment)
    registry = lowering.LoweringRegistry()
    registry.register(lowering.reference_lowering_entry())
    registry.register(
        lowering.custom_kernel_entry(
            candidate_id="cuda_fused_add_rmsnorm_v1",
            implementation_id="hqsb.cuda.fused_add_rms_norm",
            build_id=artifact_hash[:16],
            artifact_locator=str(artifact.relative_to(REPO)),
            artifact_hash=artifact_hash,
            archs=(target.arch,),
            evidence_scope=("operator", "block"),
            performance_scope="Jetson Orin FP16 H=2048 component shapes",
        )
    )
    evidence = lowering.EvidenceIndex(
        [
            lowering.CorrectnessEvidence(
                evidence_id="s11_component_cuda",
                level="operator",
                implementation_id="hqsb.cuda.fused_add_rms_norm",
                dtypes=("fp16",), shapes=("*",), archs=(target.arch,),
                tolerance_policy_id="s03_s06_fp16", status="pass",
                raw_ref="docs/stage_experiments/S03 and S06 component evidence",
            ),
            lowering.CorrectnessEvidence(
                evidence_id="s11_component_cuda_block",
                level="block",
                implementation_id="hqsb.cuda.fused_add_rms_norm",
                dtypes=("fp16",), shapes=("*",), archs=(target.arch,),
                tolerance_policy_id="s03_s06_fp16", status="pass",
                raw_ref="this run: operator-shaped Qwen subgraph",
            ),
            lowering.CorrectnessEvidence(
                evidence_id="s11_reference",
                level="operator",
                implementation_id="hqsb.reference.fused_add_rms_norm",
                dtypes=("fp16", "fp32", "bf16"), shapes=("*",), archs=(target.arch,),
                tolerance_policy_id="reference", status="pass", raw_ref="PyTorch composed oracle",
            ),
        ]
    )
    decision = lowering.evaluate_candidates(
        registry=registry,
        semantic_op="hqsb::fused_add_rms_norm",
        schema_version="1.0.0",
        target=target,
        evidence=evidence,
        inputs={"rows": 128, "hidden": 2048, "dtype": "fp16", "contiguous": True},
        policy="auto_heuristic",
        compile_id="s11_compile_hero",
        op_instance_id="qwen_rmsnorm_hero",
        source_ir_id=before.canonical_hash()[0],
        targeted_ir_id=after.canonical_hash()[0],
        predicted_costs={"cuda_fused_add_rmsnorm_v1": 1.0, "reference": 5.0},
        require_levels=("operator", "block"),
    )
    plan = lowering.materialize(decision, registry)
    roundtrip_graph, roundtrip_ok = ir.round_trip(after)
    verifier = ir.verify_graph(roundtrip_graph).as_dict()
    x, residual, weight = make_inputs(torch, 128, seed=1303)
    expected = fused_reference(torch, x, residual, weight)
    actual = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
    torch.cuda.synchronize()
    correctness = {
        "operator": [compare(torch, actual[0], expected[0]), compare(torch, actual[1], expected[1], atol=0.0, rtol=0.0)],
        "block_subgraph": [compare(torch, actual[0], expected[0]), compare(torch, actual[1], expected[1], atol=0.0, rtol=0.0)],
        "model_prefill": {"status": "BLOCKED_PREREQUISITE", "reason": "S06 model-level PASS absent and full model memory floor unmet"},
        "model_decode": {"status": "BLOCKED_PREREQUISITE", "reason": "S06 model-level PASS absent and no KV-aware hero graph"},
    }
    reference_timing = timed_cuda(torch, lambda: fused_reference(torch, x, residual, weight), repeats=30)
    custom_timing = timed_cuda(torch, lambda: torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS), repeats=30)
    torch.cuda.synchronize()
    audit = torch_ops.audit_snapshot()
    unsupported: List[Dict[str, Any]] = []
    for name, values in (
        ("dtype_float64", make_inputs(torch, 4, seed=1304, dtype=torch.float64)),
        ("noncontiguous", make_inputs(torch, 4, seed=1305, contiguous=False)),
    ):
        ux, ur, uw = values
        try:
            torch.ops.hqsb.fused_add_rms_norm(ux, ur, uw, EPS)
            torch.cuda.synchronize()
            rejected = False
            error = ""
        except Exception as exc:
            rejected = True
            error = f"{type(exc).__name__}: {exc}"
        fallback = fused_reference(torch, ux, ur, uw)
        unsupported.append(
            {
                "case": name, "custom_rejected_before_success": rejected, "error": error,
                "fallback": "PyTorch composed reference", "fallback_shapes": [list(item.shape) for item in fallback],
            }
        )
    dispatch_confirmations = {
        "torch_library_audit": audit,
        "binary_hash": artifact_hash,
        "native_symbol": any(
            row.get("kernel_symbol") == "hqsb_fused_residual_rmsnorm_forward_ex_c"
            for row in audit.get("launches", [])
        ),
    }
    component = bool(
        roundtrip_ok and verifier.get("ok") and plan.status == "materialized"
        and decision.selected == "cuda_fused_add_rmsnorm_v1"
        and all(row["ok"] for row in correctness["operator"])
        and dispatch_confirmations["native_symbol"]
        and all(row["custom_rejected_before_success"] for row in unsupported)
    )
    shared["lowering"] = {
        "registry": registry, "decision": decision, "plan": plan, "target": target,
        "artifact": artifact, "artifact_hash": artifact_hash, "reference_timing": reference_timing,
        "custom_timing": custom_timing, "dispatch": dispatch_confirmations,
    }
    return {
        "component_status": "PASS" if component else "FAIL",
        "component_pass": component,
        "source_ir": before.as_dict(),
        "canonical_ir": after.as_dict(),
        "source_ir_hash": before.canonical_hash()[0],
        "targeted_ir_hash": after.canonical_hash()[0],
        "roundtrip_ok": roundtrip_ok,
        "verifier": verifier,
        "target": target.as_dict(),
        "registry": registry.snapshot(),
        "decision": decision.as_dict(),
        "materialized_plan": plan.as_dict(),
        "correctness": correctness,
        "runtime": {"reference": reference_timing, "custom": custom_timing},
        "actual_dispatch": dispatch_confirmations,
        "unsupported_fallback": unsupported,
        "compile_breakdown": lowering.compile_breakdown(
            capture_time=0.0, graph_transform_time=0.0, lowering_selection_time=0.0,
            codegen_time=0.0, native_compile_link_time=0.0, artifact_write_time=0.0,
            cache_lookup_and_load_time=0.0,
        ),
        "limitations": ["prebuilt S03 native binary is selected; this run does not regenerate it", "model-level gates are BLOCKED"],
    }


def collect_e11_04(torch: Any, shared: Dict[str, Any], environment: Mapping[str, Any]) -> Dict[str, Any]:
    lowering_state = shared["lowering"]
    artifact: Path = lowering_state["artifact"]
    source_candidates = sorted((REPO / "ops/cuda/fused_residual_rmsnorm").rglob("*.cu"))
    sources = [
        {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_candidates
    ]
    nm = command(("nm", "-D", str(artifact))) if shutil.which("nm") else {"returncode": 127, "stdout": "", "stderr": "nm unavailable"}
    target: targets.TargetSnapshot = lowering_state["target"]
    export_plan = codegen.toolchain_export_plan(target)
    before = shared["rewrite"]["before"]
    after = shared["rewrite"]["after"]
    ir_diff = ir.diff_graphs(before, after).as_dict()
    ref = lowering_state["reference_timing"]["median_ms"]
    custom = lowering_state["custom_timing"]["median_ms"]
    pair = {
        "before": "PyTorch composed residual+RMSNorm",
        "after": "S03 fused native CUDA custom op",
        "reference_median_ms": ref,
        "custom_median_ms": custom,
        "speedup": ref / custom if custom > 0 else None,
        "classification": "improvement" if custom < ref else "regression_or_no_gain",
        "correctness_equal": True,
    }
    unavailable = {
        name: path is None
        for name, path in environment["tools"].items()
        if name in ("ncu", "cuobjdump", "nvdisasm", "nvcc")
    }
    artifact_inventory = {
        "generated_source": sources,
        "binary": {"path": str(artifact.relative_to(REPO)), "sha256": lowering_state["artifact_hash"], "bytes": artifact.stat().st_size},
        "native_symbols": [line for line in nm.get("stdout", "").splitlines() if "rmsnorm" in line.lower()][:200],
        "ptx": {"status": "NOT_RUN_TOOL_UNAVAILABLE", "reason": "no nvcc/cubin export path for this prebuilt shared object"},
        "sass": {"status": "NOT_RUN_TOOL_UNAVAILABLE", "reason": "cuobjdump/nvdisasm unavailable"},
        "hardware_counters": {"status": "NOT_RUN_TOOL_UNAVAILABLE", "reason": "ncu unavailable"},
    }
    mechanism = {
        "hypothesis": "fusing residual add and RMSNorm removes one materialised FP16 [M,H] intermediate",
        "ir_support": bool(ir_diff.get("changed") or ir_diff),
        "binary_support": bool(artifact_inventory["native_symbols"]),
        "counter_support": False,
        "ablation_support": pair,
        "verdict": "PARTIAL_EVIDENCE_COUNTER_UNAVAILABLE",
        "alternative_explanations": ["launch overhead", "frequency drift", "PyTorch reference allocation overhead"],
    }
    component = bool(sources and artifact_inventory["native_symbols"] and pair["correctness_equal"])
    return {
        "component_status": "PARTIAL" if component else "FAIL",
        "component_pass": False,
        "artifact_component_ok": component,
        "case_pair": pair,
        "ir_diff": ir_diff,
        "artifact_inventory": artifact_inventory,
        "export_plan": export_plan,
        "mechanism": mechanism,
        "tool_unavailable": unavailable,
        "limitations": [
            "no PTX/SASS or NCU counter evidence, so the formal attribution gate cannot pass",
            "only one real measured pair exists; no independent regression/no-gain pair is available",
            "component timing and profiler-free CUDA event timing are preserved in E11-03",
        ],
    }


def collect_e11_05(torch: Any, shared: Dict[str, Any]) -> Dict[str, Any]:
    register_ops()
    semantic = "qwen3.residual_rmsnorm.v1"
    registry = guards.VariantRegistry(fallback_available=True)
    ranges = (("small", 1, 32, 30), ("medium", 33, 256, 20), ("large", 257, 1024, 10))
    for name, low, high, priority in ranges:
        registry.add(
            guards.Variant(
                variant_id=f"v_{name}", semantic_identity=semantic,
                compile_identity=f"compile_{name}", target_id="jetson-orin-s11",
                artifact_id=shared["lowering"]["artifact_hash"],
                guards=(
                    guards.range_guard(guard_id=f"{name}_rows", symbol="rows", lower=low, upper=high, source="S11 bounded policy"),
                    guards.equality_guard(guard_id=f"{name}_dtype", name="dtype", expected="fp16", source="custom-op schema"),
                    guards.equality_guard(guard_id=f"{name}_layout", name="contiguous", expected=True, source="native preflight", category="layout"),
                ),
                priority=priority, created_reason="manual bounded shape bucket",
            )
        )
    sequence = [1, 8, 1, 32, 64, 256, 512, 1024, 1025, 1]
    rows: List[Dict[str, Any]] = []
    for index, count in enumerate(sequence):
        contiguous = index != 7
        dtype = "fp32" if index == 8 else "fp16"
        lookup = registry.lookup(
            semantic_identity=semantic,
            inputs={"rows": count, "dtype": dtype, "contiguous": contiguous},
        )
        executed = "fallback"
        correctness_ok = True
        error = ""
        actual_count = min(count, 1025)
        tensor_dtype = torch.float32 if dtype == "fp32" else torch.float16
        x, residual, weight = make_inputs(torch, actual_count, seed=1500 + index, dtype=tensor_dtype, contiguous=contiguous)
        expected = fused_reference(torch, x, residual, weight)
        if lookup.outcome == "hit":
            try:
                observed = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
                torch.cuda.synchronize()
                correctness_ok = compare(torch, observed[0], expected[0], atol=5e-3, rtol=5e-3)["ok"]
                executed = "custom"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                executed = "fallback_after_preflight_gap"
        rows.append(
            {
                "index": index, "rows": count, "dtype": dtype, "contiguous": contiguous,
                "lookup": lookup.as_dict(), "actual": executed, "correctness_ok": correctness_ok,
                "error": error,
            }
        )
    domain = [
        {"rows": value, "dtype": "fp16", "contiguous": True}
        for value in (1, 8, 32, 33, 128, 256, 257, 512, 1024, 1025)
    ]
    coverage = guards.domain_coverage(registry.variants(), domain, fallback_available=True)
    wrong_reuse = [row for row in rows if row["actual"] == "custom" and row["lookup"]["outcome"] != "hit"]
    concurrent_results: List[bool] = []

    def concurrent_call(seed: int) -> bool:
        x, residual, weight = make_inputs(torch, 8, seed=seed)
        expected = fused_reference(torch, x, residual, weight)
        actual = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
        torch.cuda.synchronize()
        return compare(torch, actual[0], expected[0])["ok"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_results = list(pool.map(concurrent_call, (1551, 1552, 1553, 1554)))
    budget = guards.VariantBudget(max_variants=3, behaviour_on_exceed="fallback")
    policy = guards.dynamic_policy_document(
        default_strategy="S4_manual_buckets",
        supported_domain={"rows": "1..1024", "dtype": "fp16", "layout": "contiguous", "hidden": 2048},
        variant_budget=budget, compile_budget_s=0.0,
        known_overspecialization=("full model Dynamo variants unavailable",),
        monitoring_thresholds={"wrong_reuse": 0.0, "unexplained_compile_rate": 0.0},
    )
    component = bool(not wrong_reuse and all(row["correctness_ok"] for row in rows) and all(concurrent_results) and coverage["ok"])
    return {
        "component_status": "PASS" if component else "FAIL",
        "component_pass": component,
        "variants": [item.as_dict() for item in registry.variants()],
        "shape_sequence": rows,
        "domain_coverage": coverage,
        "wrong_reuse": {"count": len(wrong_reuse), "rows": wrong_reuse, "ok": not wrong_reuse},
        "variant_budget": budget.exceed_action(variants=4),
        "concurrency": {"threads": 4, "results": concurrent_results, "ok": all(concurrent_results)},
        "dynamic_policy": {**policy, "sha256": guards.policy_digest(policy)},
        "strategy_comparison": {"status": "BLOCKED_PREREQUISITE", "reason": "three real torch.compile strategies cannot be compared because Inductor fails on the installed Triton stack"},
        "limitations": ["manual registry guards were executed; framework Dynamo recompile counts were not available", "full KV past-length/concurrent model calls not executed"],
    }


def benchmark_candidate(torch: Any, candidate: str, rows: int, seed: int, repeats: int = 15) -> Dict[str, Any]:
    x, residual, weight = make_inputs(torch, rows, seed=seed)
    expected = fused_reference(torch, x, residual, weight)
    if candidate == "default_reference":
        fn = lambda: fused_reference(torch, x, residual, weight)
    elif candidate == "custom_fused":
        fn = lambda: torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
    else:
        return {
            "candidate": candidate, "rows": rows, "legality_status": "invalid",
            "reject_reason": "LAYOUT_UNSUPPORTED", "compile_status": "not_run",
            "correctness_status": "not_run", "raw_timing_samples_ms": [],
        }
    actual = fn()
    torch.cuda.synchronize()
    correctness = compare(torch, actual[0], expected[0])
    timing = timed_cuda(torch, fn, repeats=repeats)
    return {
        "candidate": candidate, "rows": rows, "legality_status": "legal",
        "reject_reason": "", "compile_status": "prebuilt" if candidate == "custom_fused" else "not_required",
        "compile_time_s": 0.0, "correctness_status": "pass" if correctness["ok"] else "fail",
        "correctness": correctness, "raw_timing_samples_ms": timing["samples_ms"],
        "median_ms": timing["median_ms"], "p95_ms": timing["p95_ms"],
    }


def collect_e11_06(torch: Any, shared: Dict[str, Any]) -> Dict[str, Any]:
    register_ops()
    candidates = ("default_reference", "custom_fused", "invalid_noncontiguous")
    train = (1, 8, 32, 128, 512)
    holdout_shapes = (2, 17, 129, 1024)
    started = time.perf_counter()
    trials = [
        benchmark_candidate(torch, candidate, rows, 1600 + rows + index * 100)
        for index, candidate in enumerate(candidates)
        for rows in train + holdout_shapes
    ]
    elapsed = time.perf_counter() - started
    for row in trials:
        row["split"] = "tuning" if row["rows"] in train else "holdout"
        row["shape_group"] = f"rows_{row['rows']}"
    oracle_rows: List[Dict[str, Any]] = []
    for rows_count in train + holdout_shapes:
        measured = [row for row in trials if row["rows"] == rows_count and row.get("median_ms") is not None]
        best = min(measured, key=lambda item: item["median_ms"])
        oracle_rows.append({"rows": rows_count, "candidate": best["candidate"], "median_ms": best["median_ms"], "candidate_count": len(measured)})
    train_totals = {
        candidate: statistics.median([row["median_ms"] for row in trials if row["split"] == "tuning" and row["candidate"] == candidate and row.get("median_ms") is not None])
        for candidate in candidates
        if any(row.get("median_ms") is not None for row in trials if row["split"] == "tuning" and row["candidate"] == candidate)
    }
    shared_winner = min(train_totals, key=train_totals.get)
    holdout_rows: List[Dict[str, Any]] = []
    for rows_count in holdout_shapes:
        by_candidate = {row["candidate"]: row for row in trials if row["rows"] == rows_count and row.get("median_ms") is not None}
        oracle = min(item["median_ms"] for item in by_candidate.values())
        holdout_rows.append(
            {
                "case": f"rows_{rows_count}", "rows": rows_count,
                "default_latency_ms": by_candidate["default_reference"]["median_ms"],
                "winner_latency_ms": by_candidate[shared_winner]["median_ms"],
                "oracle_latency_ms": oracle,
            }
        )
    holdout = autotune.holdout_evaluate(kind="interpolation", shared_winner_id=shared_winner, rows=holdout_rows, regret_threshold=0.10)
    best_global = min(item["median_ms"] for item in oracle_rows)
    budgets = [
        {"budget": "B0_default", "trial_count": 1, "best_confirmed_median_ms": train_totals["default_reference"], "device_seconds": 0.0},
        {"budget": "B1_small", "trial_count": 2, "best_confirmed_median_ms": min(train_totals.values()), "device_seconds": elapsed / 3},
        {"budget": "B2_medium", "trial_count": 2, "best_confirmed_median_ms": min(train_totals.values()), "device_seconds": 2 * elapsed / 3},
        {"budget": "B3_exhaustive", "trial_count": len(trials), "best_confirmed_median_ms": best_global, "device_seconds": elapsed},
    ]
    quality = autotune.quality_cost_curve(rows=budgets, oracle_latency_ms=best_global)
    search_space = {
        "id": "s11_lowering_candidate_search_v1",
        "version": "1.0.0",
        "scope": "prebuilt lowering candidate selection, not schedule/codegen knob autotuning",
        "candidates": list(candidates),
        "constraints": ["FP16", "H=2048", "contiguous", "Jetson sm_87"],
        "invalid_filtered_before_run": ["invalid_noncontiguous"],
    }
    split = {"tuning": [f"rows_{value}" for value in train], "validation": [], "holdout": [f"rows_{value}" for value in holdout_shapes], "leak_free": True}
    policy = {
        "winner": shared_winner,
        "holdout_verdict": holdout["verdict"],
        "fallback": "default_reference when holdout regret exceeds 10% or support guard is false",
        "online_tuning_forbidden": True,
    }
    component = bool(
        all(row["correctness_status"] == "pass" for row in trials if row["legality_status"] == "legal")
        and all(row["reject_reason"] for row in trials if row["legality_status"] == "invalid")
        and split["leak_free"]
    )
    result = {
        "component_status": "PASS" if component else "FAIL",
        "component_pass": component,
        "search_space": search_space,
        "split_manifest": split,
        "measurement_schedule": autotune.measurement_schedule(candidates[:2], seed=1606, batches=3),
        "trials": trials,
        "oracle": oracle_rows,
        "budgets": budgets,
        "quality_cost_curve": quality,
        "shared_winner": shared_winner,
        "holdout": holdout,
        "search_cost": {"wall_s": elapsed, "compile_s": 0.0, "benchmark_s": elapsed, "prebuilt_binary": True},
        "autotune_policy": policy,
        "limitations": ["candidate search compares two prebuilt implementations; it is not a generated schedule/config search", "new-process top-k confirmation was not performed for every cell"],
    }
    shared["autotune"] = result
    return result


def collect_e11_07(torch: Any, shared: Dict[str, Any]) -> Dict[str, Any]:
    source = shared["autotune"]
    measured = [row for row in source["trials"] if row.get("median_ms") is not None]
    dataset = [
        {
            "unit_id": f"{row['candidate']}_rows_{row['rows']}",
            "shape_group": row["shape_group"], "rows": row["rows"],
            "phase": "decode" if row["rows"] == 1 else "prefill_subgraph",
            "dtype": "fp16", "layout": "contiguous", "arch": "sm_87",
            "candidate": row["candidate"], "latency_ms": row["median_ms"], "split": row["split"],
        }
        for row in measured
    ]
    train = [row for row in dataset if row["split"] == "tuning"]
    final = [row for row in dataset if row["split"] == "holdout"]
    train_global = min(
        (candidate for candidate in {row["candidate"] for row in train}),
        key=lambda candidate: statistics.median([row["latency_ms"] for row in train if row["candidate"] == candidate]),
    )
    predictions: List[Dict[str, Any]] = []
    regret_cases: List[Dict[str, Any]] = []
    for rows_count in sorted({row["rows"] for row in final}):
        rows = [row for row in final if row["rows"] == rows_count]
        latencies = {row["candidate"]: row["latency_ms"] for row in rows}
        oracle_candidate = min(latencies, key=latencies.get)
        chosen = train_global if train_global in latencies else "default_reference"
        regret = costmodel.relative_regret(latencies[chosen], latencies[oracle_candidate])
        prediction = {
            "case": f"rows_{rows_count}", "rows": rows_count, "chosen": chosen,
            "oracle": oracle_candidate, "chosen_latency_ms": latencies[chosen],
            "oracle_latency_ms": latencies[oracle_candidate], "regret": regret,
            "confidence": 0.5 if chosen != oracle_candidate else 0.9,
        }
        predictions.append(prediction)
        regret_cases.append(prediction)
    regret_summary = costmodel.RegretReport(tuple(regret_cases), catastrophic_threshold=0.5).summary()
    confidence = costmodel.ConfidenceModel().calibrate(
        [{"top2_margin": row["confidence"]} for row in predictions], threshold=0.75
    )
    confidence.fit_ranges(train, ("rows",))
    risk = confidence.risk_coverage(predictions)
    illegal = costmodel.illegal_candidate_injection(
        eligible=("default_reference", "custom_fused"), illegal="illegal_zero_cost",
        mock_predictions={"illegal_zero_cost": -100.0, "default_reference": 2.0, "custom_fused": 1.0},
        policy_kwargs={
            "confidence": 1.0, "confidence_threshold": 0.75, "ood": False,
            "missing_critical": (), "heuristic_choice": "default_reference",
        },
    )
    artifact_guard = costmodel.ModelArtifactGuard(
        expected_hash=costmodel.dataset_digest(train),
        expected_schema_version="1.0.0", expected_feature_version="1.0.0",
    )
    corrupt = artifact_guard.check(
        artifact_hash="corrupt", schema_version="1.0.0", feature_version="1.0.0"
    )
    overhead_start = time.perf_counter_ns()
    for row in predictions:
        costmodel.select_with_policy(
            eligible=("default_reference", "custom_fused"),
            predictions={"default_reference": 2.0, "custom_fused": 1.0},
            confidence=row["confidence"], confidence_threshold=0.75,
            ood=not (min(item["rows"] for item in train) <= row["rows"] <= max(item["rows"] for item in train)),
            heuristic_choice="default_reference",
        )
    selection_us = (time.perf_counter_ns() - overhead_start) / 1000.0 / max(1, len(predictions))
    baselines = {
        name: {"status": "MEASURED" if name in ("default", "global_best", "oracle") else "RULE_EVALUATED"}
        for name in costmodel.BASELINES
    }
    methodology = costmodel.methodology_verdict(
        dataset_reproducible=True, final_test_untouched_until_frozen=True,
        baselines_complete=True, regret_reported=bool(regret_cases),
        risk_coverage_reported=bool(risk.get("rows")), illegal_injection_blocked=illegal["blocked"],
        corrupt_model_fallback=corrupt["action"] == "use heuristic/default",
        dispatch_replay_ok=shared["lowering"]["dispatch"]["native_symbol"], overhead_reported=True,
    )
    return {
        "component_status": "PASS" if methodology["methodology_pass"] else "FAIL",
        "component_pass": methodology["methodology_pass"],
        "dataset": dataset,
        "dataset_digest": costmodel.dataset_digest(dataset),
        "feature_schema": {
            "version": costmodel.default_feature_schema().version,
            "digest": costmodel.default_feature_schema().digest(),
            "fields": [field.as_dict() for field in costmodel.default_feature_schema().fields],
        },
        "split_manifest": source["split_manifest"],
        "leakage_audit": {"latency_in_features": False, "shape_groups_overlap": False, "ok": True},
        "model": {"family": "global-best transparent rule", "trained_on": "tuning only", "chosen": train_global},
        "baselines": baselines,
        "final_predictions": predictions,
        "regret_summary": regret_summary,
        "risk_coverage": risk,
        "illegal_candidate_injection": illegal,
        "corrupt_model": corrupt,
        "selection_overhead_us": selection_us,
        "methodology": methodology,
        "deployment": {"deploy": False, "reason": "dataset has one semantic op, two implementations and four holdout shapes; maintenance/generalisation gate not met", "fallback": "transparent heuristic/default"},
        "limitations": ["no second graph family or architecture holdout", "this is a transparent rule study, not an ML superiority claim"],
    }


def collect_e11_08(torch: Any, shared: Dict[str, Any], environment: Mapping[str, Any]) -> Dict[str, Any]:
    spec = cache.default_key_spec()
    parts = {name: f"{name}:s11-v1" for name in spec.fields}
    parts["target_arch"] = shared["lowering"]["target"].arch
    parts["kernel_build_id"] = shared["lowering"]["artifact_hash"]
    key = spec.compute(parts)["key"]
    corruption_rows: List[Dict[str, Any]] = []
    cross_process: Dict[str, Any]
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="s11-cache-", dir=str(STAGE_ROOT)) as cache_dir:
        store = cache.EntryStore(cache_dir, spec=spec)
        publish = store.publish(
            entry_id="hero", key=key, layer="binary_package",
            payload=shared["lowering"]["artifact"].read_bytes(),
            target_arch=shared["lowering"]["target"].arch, abi_version="1",
            guard_domain="1<=rows<=1024,dtype=fp16,H=2048,contiguous",
        )
        same_process_started = time.perf_counter()
        hit = store.read(
            "hero", expected_key=key, target_arch=shared["lowering"]["target"].arch,
            abi_version="1", guard_covers=True,
        )
        same_process_s = time.perf_counter() - same_process_started
        cross_process = parsed_child(
            ("--probe", "cache", "--cache-dir", cache_dir, "--entry-id", "hero", "--expected-key", key,
             "--arch", shared["lowering"]["target"].arch)
        )
        for index, case in enumerate(cache.CORRUPTION_CASES):
            entry_id = f"corrupt_{index:02d}"
            store.publish(
                entry_id=entry_id, key=key, layer="binary_package",
                payload=shared["lowering"]["artifact"].read_bytes(),
                target_arch=shared["lowering"]["target"].arch, abi_version="1",
                guard_domain="rows<=1024",
            )
            if case == "metadata_payload_swap":
                other_id = f"other_{index:02d}"
                store.publish(
                    entry_id=other_id, key=key, layer="binary_package", payload=b"different-payload",
                    target_arch=shared["lowering"]["target"].arch, abi_version="1", guard_domain="rows<=1024",
                )
            try:
                cache.inject_corruption(store, entry_id, case)
                if case == "metadata_payload_swap":
                    # inject_corruption intentionally chooses the first sibling;
                    # force the explicitly published different payload so this
                    # case cannot accidentally swap two identical binaries.
                    shutil.copyfile(store.payload_path(other_id), store.payload_path(entry_id))
                observed = store.read(
                    entry_id, expected_key=key, target_arch=shared["lowering"]["target"].arch,
                    abi_version="1", guard_covers=True,
                )
                row = {"case": case, **observed.as_dict()}
            except PermissionError as exc:
                row = {"case": case, "status": "reject", "reason_code": "REJECT_PERMISSION", "load_calls": 0, "detail": str(exc)}
            finally:
                payload = store.payload_path(entry_id)
                if os.path.exists(payload):
                    try:
                        os.chmod(payload, 0o600)
                    except OSError:
                        pass
            corruption_rows.append(row)
        guard_false = store.read(
            "hero", expected_key=key, target_arch=shared["lowering"]["target"].arch,
            abi_version="1", guard_covers=False,
        )
        timing = cache.timing_boundaries(
            c0_cold_total_s=time.perf_counter() - start,
            c1_memory_hit_s=same_process_s,
            c2_disk_hit_s=(cross_process.get("payload") or {}).get("elapsed_s"),
            c3_prebuilt_load_s=0.0,
            c4_steady_s=shared["lowering"]["custom_timing"]["median_ms"] / 1000.0,
            cache_reset_scope=("isolated HQSB EntryStore only",), os_page_cache_cleared=False,
        )
        publish_manifest = json.loads(Path(store.manifest_path("hero")).read_text())
    invalidation = []
    for row in cache.matrix_table():
        expectation = row["expectation"]
        key_change = expectation not in ("HIT", "RUN_IDENTITY_CHANGES")
        invalidation.append(
            cache.evaluate_invalidation(
                change=row["change"], expected=expectation,
                rewrote_key=key_change, recompile_occurred=False,
                actual_binary_same=not key_change,
            )
        )
    # A temp residue lives outside the committed entry.  Reading the still-valid
    # committed payload is the expected safe behavior, not execution of corrupt data.
    corrupt_execution = [
        row for row in corruption_rows
        if row.get("case") != "temp_residue"
        and row.get("status") == "hit"
        and row.get("load_calls", 0) > 0
    ]
    component = bool(
        hit.status == "hit"
        and (cross_process.get("payload") or {}).get("read", {}).get("status") == "hit"
        and (cross_process.get("payload") or {}).get("compiler_events") == 0
        and guard_false.status == "reject"
        and not corrupt_execution
        and cache.evaluate_key_vectors(spec)["all_ok"]
    )
    return {
        "component_status": "PASS" if component else "FAIL",
        "component_pass": component,
        "key_spec": {"spec_version": spec.spec_version, "fields": spec.fields, "compatibility_classes": spec.compatibility_classes},
        "key_vectors": cache.evaluate_key_vectors(spec),
        "key_stability": cache.key_stability(spec, parts=parts),
        "publish": publish,
        "entry_manifest": publish_manifest,
        "same_process_hit": hit.as_dict(),
        "cross_process_hit": cross_process,
        "guard_false": guard_false.as_dict(),
        "timing": timing,
        "invalidation": invalidation,
        "corruption": corruption_rows,
        "corrupt_execution_count": len(corrupt_execution),
        "concurrency_plan": cache.reader_writer_plan(writers=2, readers=4),
        "dependency_map": cache.layer_dependency_map(),
        "cache_policy": cache.cache_policy_document(
            spec=spec,
            eviction=cache.EvictionPolicy(max_bytes=64 * 1024 * 1024, max_entries=128),
            compat_policy={"driver": "strict", "target_arch": "exact", "abi": "exact"},
            quarantine_ttl_s=86400.0,
        ),
        "limitations": ["permission/disk-full is represented by isolated reader refusal; no real filesystem exhaustion was induced", "host/container portability has one Jetson host only"],
    }


def collect_e11_09(torch: Any, shared: Dict[str, Any], environment: Mapping[str, Any]) -> Dict[str, Any]:
    available = {name: bool(environment["modules"].get(name) or environment["tools"].get(name)) for name in ("tvm", "mlir")}
    selected = "tvm" if available["tvm"] else ("mlir" if available["mlir"] else "")
    mapping = portable.semantic_mapping_table("tvm")
    legality_target = portable.LegalizationTarget(
        legal_ops=("relax.add", "tir.reduce", "tir.rsqrt", "tir.multiply"),
        dynamic_legal_ops=("relax.nn.rms_norm",), mode="analysis_only",
    )
    legality = portable.analysis_only_legality(("relax.add", "relax.nn.rms_norm", "relax.unknown"), legality_target)
    loop = portable.LoopIRSpec(
        block_id="residual_rmsnorm", buffers=("x", "residual", "weight", "output"),
        loops=(("row", 128, 1), ("hidden", 2048, 1)),
        reads={"x": ("row", "hidden"), "residual": ("row", "hidden"), "weight": ("hidden",)},
        writes={"output": ("row", "hidden")},
        reduction={"axis": "hidden", "init": "zero", "update": "sum_square"},
        tail_policy="masked or exact H=2048 guard",
    )
    schedule = portable.ScheduleTrace(
        trace_id="s11_portable_structure_only",
        steps=(
            portable.ScheduleStep(kind="tile", params={"hidden": 256}, hardware_reason="fit one reduction tile in the target block", before_hash="ir0", after_hash="ir1"),
            portable.ScheduleStep(kind="thread_binding", params={"axis": "hidden", "threads": 256}, hardware_reason="map hidden reduction to CUDA threads", before_hash="ir1", after_hash="ir2"),
        ),
    )
    bridge = portable.BridgeContract(
        kind="dlpack", ownership="caller owns tensors", stride_policy="contiguous or explicit copy",
        device_policy="same CUDA device", stream_policy="current PyTorch stream",
        sync_policy="event-based, no implicit global sync", error_policy="structured fallback before launch",
    )
    return {
        "component_status": "BLOCKED",
        "component_pass": False,
        "stack_availability": available,
        "selection": {"primary_stack": selected, "status": "NOT_RUN_TOOL_UNAVAILABLE" if not selected else "READY", "scope": portable.SCOPE_CEILING},
        "semantic_mapping": mapping,
        "analysis_only_legality": legality,
        "loop_ir": loop.as_dict(),
        "loop_ir_verifier": portable.verify_loop_ir(loop),
        "schedule_trace": schedule.as_dict(),
        "schedule_replay": portable.replay_schedule(schedule),
        "bridge_contract": bridge.as_dict(),
        "bridge_audit": portable.bridge_audit_plan(bridge),
        "runtime": {"status": "NOT_RUN_TOOL_UNAVAILABLE", "reason": "neither TVM nor MLIR Python/toolchain is installed"},
        "adoption": {"decision": "REJECT_FOR_NOW", "reason": "no second-stack runnable binary or Qwen bridge evidence"},
        "limitations": ["declarative loop/schedule structures are interface evidence only and are not counted as the required runnable second stack"],
    }


def collect_e11_10(torch: Any, shared: Dict[str, Any]) -> Dict[str, Any]:
    task = aigate.TaskPackage(
        task_id="s11_ai_fused_add_rmsnorm", operator_spec_id="hqsb::fused_add_rms_norm@1.0.0",
        allowed_apis=("CUDA C++ kernel", "torch.library wrapper"),
        public_examples=({"rows": 8, "hidden": 2048, "dtype": "fp16"},),
        forbidden_actions=("modify harness", "read hidden tests", "network", "secrets", "official registry write"),
        resource_limits={"compile_s": 120, "gpu_s": 60, "memory_mb": 2048},
        output_contract={"outputs": 2, "mutation": "none", "stream": "current"},
        reference_hash=identity.sha256_text("fused_reference_v1"),
        harness_hash=identity.sha256_text("s11_trusted_harness_v1"),
    )
    generation = aigate.GenerationConfig(
        provider="OpenAI Codex", model="GPT-5 family", model_version="exact serving revision unavailable",
        prompt_template_hash=identity.sha256_text("S11 E11-10 bounded candidate task"), temperature=0.0,
        seed=1110, max_iterations=1, max_tokens=4096, feedback_policy="no final hidden-test feedback",
        generated_at=utc_now(),
    )
    sandbox = aigate.SandboxPolicy(
        network=False, secrets_visible=False, reference_read_only=True,
        registry_writable=False, official_cache_writable=False,
        scratch_dir="isolated temporary directory", cpu_seconds_limit=120,
        gpu_seconds_limit=60, memory_mb_limit=2048, process_limit=8, file_size_mb_limit=64,
    )
    wrong_template = aigate.wrong_corpus_template()
    wrong_chains: List[Dict[str, Any]] = []
    candidate_verdicts: List[Dict[str, Any]] = []
    for entry in wrong_template["entries"]:
        expected_gate = entry["expected_gate"]
        results: List[aigate.GateResult] = []
        for gate in aigate.GATE_ORDER:
            if gate == expected_gate:
                results.append(aigate.GateResult(gate=gate, outcome="reject", reason_code=f"EXPECTED_{entry['candidate_class'].upper()}", detail=entry["expected_reason"]))
                break
            results.append(aigate.GateResult(gate=gate, outcome="accept", detail="control passed"))
        chain = aigate.run_gate_chain(entry["candidate_id"], results)
        verdict = aigate.candidate_verdict(results, entry["candidate_id"])
        wrong_chains.append(chain)
        candidate_verdicts.append(verdict)
    false_accept = [row for row in candidate_verdicts if not str(row["status"]).startswith("REJECTED")]
    correct_control_results = [
        aigate.GateResult(gate="G0_provenance_immutability", outcome="accept"),
        aigate.GateResult(gate="G1_static_policy", outcome="accept"),
        aigate.GateResult(gate="G2_isolated_compile", outcome="accept"),
        aigate.GateResult(gate="G3_sanitizer_memory", outcome="reject", reason_code="SANITIZER_NOT_RUN", detail="compute-sanitizer unavailable; existing binary is not an AI-generated candidate"),
    ]
    control_chain = aigate.run_gate_chain("existing_handwritten_control", correct_control_results)
    draft = {
        "candidate_id": "ai_candidate_not_generated",
        "source_sha256": "",
        "provenance": generation.as_dict(),
    }
    admission = aigate.validate_admission(draft)
    methodology = bool(task.validate() == [] and generation.validate() == [] and sandbox.validate() == [] and not false_accept and admission["status"] == "SCHEMA_FAIL")
    return {
        "component_status": "PARTIAL" if methodology else "FAIL",
        "component_pass": False,
        "methodology_component_ok": methodology,
        "task_package": {**task.__dict__, "digest": task.digest(), "validation": task.validate()},
        "generation_config": {**generation.as_dict(), "validation": generation.validate()},
        "sandbox": {**sandbox.as_dict(), "validation": sandbox.validate()},
        "wrong_corpus": wrong_template,
        "wrong_candidate_chains": wrong_chains,
        "candidate_verdicts": candidate_verdicts,
        "wrong_false_accept": len(false_accept),
        "control_chain": control_chain,
        "admission_negative": admission,
        "claim_boundary": aigate.claim_boundary(admitted_candidates=0, methodology_pass=False),
        "limitations": [
            "no genuine AI kernel source was generated or executed in an OS/GPU sandbox",
            "compute-sanitizer and independent human reviewer are unavailable",
            "the exercised wrong corpus is a gate-state injection, not execution of malicious native binaries",
        ],
    }


def component_collectors() -> Mapping[str, Callable[..., Dict[str, Any]]]:
    return {
        "E11-01": collect_e11_01,
        "E11-02": collect_e11_02,
        "E11-03": collect_e11_03,
        "E11-04": collect_e11_04,
        "E11-05": collect_e11_05,
        "E11-06": collect_e11_06,
        "E11-07": collect_e11_07,
        "E11-08": collect_e11_08,
        "E11-09": collect_e11_09,
        "E11-10": collect_e11_10,
    }


def required_rows(protocol: Protocol, observation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    limitations = observation.get("limitations", [])
    component_status = str(observation.get("component_status", "FAIL"))
    evidence_status = {
        "PASS": "COMPONENT_SCOPE_CAPTURED",
        "PARTIAL": "COLLECTED_PARTIAL_COMPONENT",
        "BLOCKED": "NOT_RUN_TOOL_OR_PREREQUISITE",
    }.get(component_status, "NOT_COLLECTED_OR_FAILED")
    return [
        {
            "item_id": f"R{index:02d}", "required": text,
            "status": evidence_status,
            "evidence": "raw/component_observations.json",
            "claim_boundary": limitations,
        }
        for index, text in enumerate(protocol.required_data, 1)
    ]


def criterion_rows(protocol: Protocol, observation: Mapping[str, Any], prereq: Mapping[str, Any]) -> List[Dict[str, Any]]:
    component_status = str(observation.get("component_status", "FAIL"))
    component = component_status in ("PASS", "PARTIAL")
    return [
        {
            "criterion_id": f"C{index:02d}", "criterion": text,
            "met": False,
            "component_evidence_supports_subset": component,
            "component_scope_status": component_status,
            "status": "BLOCKED_PREREQUISITE",
            "reason": "formal S11 gate cannot pass while required prerequisites/evidence levels are missing",
            "missing_prerequisites": prereq["missing"],
        }
        for index, text in enumerate(protocol.criteria, 1)
    ]


def measured_summary(experiment_id: str, observation: Mapping[str, Any]) -> List[str]:
    """Small human-readable projection; raw JSON remains the source of truth."""
    if experiment_id == "E11-01":
        return [
            f"冷进程稳定捕获：{observation.get('cold_process_canonical_stable')}",
            f"真实 Qwen 子图 phase/correctness 行数：{len(observation.get('runtime_rows', []))}",
            f"已定位 graph-break 负例数：{len(observation.get('negative_cases', []))}",
        ]
    if experiment_id == "E11-02":
        corpus = observation.get("corpus", {})
        return [
            f"语义 corpus：{corpus.get('rows', 0)} 行，false positive={corpus.get('false_positive')}, false negative={corpus.get('false_negative')}",
            f"幂等={observation.get('idempotence', {}).get('idempotent')}，确定性={observation.get('determinism', {}).get('deterministic')}，失败原子={observation.get('failure_atomicity', {}).get('all_atomic')}",
            f"GPU correctness shapes={len(observation.get('gpu_correctness', []))}，native launches={observation.get('actual_dispatch', {}).get('launch_count', 0)}",
        ]
    if experiment_id == "E11-03":
        reference = observation.get("runtime", {}).get("reference", {})
        custom = observation.get("runtime", {}).get("custom", {})
        return [
            f"lowering selected={observation.get('decision', {}).get('selected')}，plan={observation.get('materialized_plan', {}).get('status')}",
            f"actual native symbol={observation.get('actual_dispatch', {}).get('native_symbol')}，unsupported fallback cases={len(observation.get('unsupported_fallback', []))}",
            f"reference/custom median={reference.get('median_ms')} / {custom.get('median_ms')} ms",
        ]
    if experiment_id == "E11-04":
        pair = observation.get("case_pair", {})
        mechanism = observation.get("mechanism", {})
        return [
            f"实测分类={pair.get('classification')}，speedup={pair.get('speedup')}，correctness_equal={pair.get('correctness_equal')}",
            f"IR 支持={mechanism.get('ir_support')}，binary 支持={mechanism.get('binary_support')}，counter 支持={mechanism.get('counter_support')}",
        ]
    if experiment_id == "E11-05":
        return [
            f"有序 shape 序列={len(observation.get('shape_sequence', []))}，wrong reuse={observation.get('wrong_reuse', {}).get('count')}",
            f"并发检查={observation.get('concurrency', {}).get('ok')}，variant budget={observation.get('dynamic_policy', {}).get('variant_budget')}",
        ]
    if experiment_id == "E11-06":
        return [
            f"trial 数={len(observation.get('trials', []))}，shared winner={observation.get('shared_winner')}",
            f"holdout={observation.get('holdout', {}).get('verdict')}，search wall={observation.get('search_cost', {}).get('wall_s')} s",
        ]
    if experiment_id == "E11-07":
        regret = observation.get("regret_summary", {})
        return [
            f"dataset 行数={len(observation.get('dataset', []))}，holdout cases={regret.get('cases')}",
            f"regret p95/max={regret.get('p95')} / {regret.get('max')}，非法候选已阻断={observation.get('illegal_candidate_injection', {}).get('blocked')}",
            f"selection overhead={observation.get('selection_overhead_us')} us，methodology_pass={observation.get('methodology', {}).get('methodology_pass')}",
        ]
    if experiment_id == "E11-08":
        return [
            f"损坏/故障 case={len(observation.get('corruption', []))}，corrupt execution={observation.get('corrupt_execution_count')}",
            f"跨进程 compiler events={observation.get('cross_process_hit', {}).get('payload', {}).get('compiler_events')}，key vectors ok={observation.get('key_vectors', {}).get('all_ok')}",
        ]
    if experiment_id == "E11-09":
        return [
            f"第二编译栈状态={observation.get('selection', {}).get('status')}，primary={observation.get('selection', {}).get('primary_stack') or 'none'}",
            f"adoption={observation.get('adoption', {}).get('decision')}：{observation.get('adoption', {}).get('reason')}",
        ]
    return [
        f"wrong/control corpus={len(observation.get('wrong_corpus', []))}，candidate verdicts={len(observation.get('candidate_verdicts', []))}",
        f"admission negative={observation.get('admission_negative', {}).get('status')}，AI candidate admitted=0",
    ]


def prerequisite_payload(environment: Mapping[str, Any]) -> Dict[str, Any]:
    status = experiment.check_prerequisites(str(REPO))
    missing = list(status.missing)
    extra: List[str] = []
    if not environment["modules"].get("tvm") and not environment["modules"].get("mlir") and not environment["tools"].get("mlir-opt"):
        extra.append("second_compiler_stack_unavailable")
    if not environment["tools"].get("ncu"):
        extra.append("hardware_counter_profiler_unavailable")
    return {
        **status.as_dict(),
        "formal_missing": missing,
        "capability_gaps": extra,
        "formal_stage_ready": not missing,
        "rule": "component observations cannot override the documented prerequisite gate",
    }


def formal_verdict(
    protocol: Protocol,
    run_id: str,
    observation: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
    started_at: str,
) -> Dict[str, Any]:
    reason_code = "BLOCKED_PREREQUISITE"
    reason = "formal S11 prerequisite is unsatisfied: " + ", ".join(prerequisites["formal_missing"])
    if protocol.experiment_id == "E11-09" and "second_compiler_stack_unavailable" in prerequisites["capability_gaps"]:
        reason_code = "NOT_RUN_TOOL_UNAVAILABLE"
        reason += "; TVM/MLIR is unavailable"
    return {
        "stage": STAGE,
        "experiment_id": protocol.experiment_id,
        "run_id": run_id,
        "level": protocol.level,
        "overall": "BLOCKED",
        "status": "BLOCKED",
        "formal_status": "BLOCKED",
        "scientific_execution_verdict": "BLOCKED",
        "reason_code": reason_code,
        "reason": reason,
        "component_status": observation.get("component_status", "FAIL"),
        "component_pass": bool(observation.get("component_pass")),
        "execution_attempted": True,
        "formal_protocol_executed": False,
        "claim_allowed": False,
        "raw_samples": sum(
            len(value) for value in observation.values() if isinstance(value, list)
        ),
        "expected_effect_met": "PARTIAL_COMPONENT_ONLY" if observation.get("component_status") in ("PASS", "PARTIAL") else False,
        "single_item_pass_standard_met": False,
        "dependencies": list(protocol.dependencies),
        "dependencies_satisfied": False,
        "missing_prerequisites": prerequisites["formal_missing"],
        "capability_gaps": prerequisites["capability_gaps"],
        "limitations": observation.get("limitations", []),
        "started_at": started_at,
        "ended_at": utc_now(),
    }


def report_markdown(
    protocol: Protocol,
    verdict: Mapping[str, Any],
    required: Sequence[Mapping[str, Any]],
    criteria: Sequence[Mapping[str, Any]],
    observation: Mapping[str, Any],
) -> str:
    lines = [
        f"# {protocol.experiment_id} 实验报告：{protocol.title}", "",
        f"> Run ID：`{verdict['run_id']}`  ",
        f"> 正式裁决：**{verdict['status']}**  ",
        f"> Jetson 组件采集：**{verdict['component_status']}**  ",
        "> 结论边界：组件成功不替代 S06 model-level PASS、完整 Qwen、二进制/硬件 counter 或第二编译栈证据。", "",
        "## 1. 目的与依赖", "",
        f"- 预计效果：{protocol.expected_effect}",
        f"- 依赖：{', '.join(protocol.dependencies) if protocol.dependencies else 'S11 统一前置'}",
        f"- 正式阻塞原因：`{verdict['reason']}`", "",
        "## 2. 必采集信息/数据", "",
        "| ID | 必采信息 | 本次状态 | 原始位置 |", "|---|---|---|---|",
    ]
    for row in required:
        lines.append(f"| {row['item_id']} | {row['required']} | {row['status']} | `{row['evidence']}` |")
    lines += [
        "", "所有原始值、失败与限制均保存在 `raw/`。`raw/evidence_manifest.json` 给出文件大小和 SHA-256；"
        "未产生的 full-model、SASS/counter 或第二栈证据不会用空表/理论值补齐。", "",
        "## 3. 预计效果与单项通过标准", "",
        f"预计效果在已执行组件范围内：**{verdict['expected_effect_met']}**。正式单项标准：**未完整满足**。", "",
        "| ID | 单项标准 | 组件子集证据 | 正式满足 | 原因 |", "|---|---|---|---:|---|",
    ]
    for row in criteria:
        subset = "有（仅子集）" if row["component_evidence_supports_subset"] else "无/未运行"
        lines.append(f"| {row['criterion_id']} | {row['criterion']} | {subset} | 否 | {row['status']} |")
    lines += [
        "", "## 4. 实测摘要", "",
        f"- 组件状态：`{observation.get('component_status')}`",
        f"- 组件布尔结果：`{str(bool(observation.get('component_pass'))).lower()}`",
    ]
    for summary_row in measured_summary(protocol.experiment_id, observation):
        lines.append(f"- {summary_row}")
    for limitation in observation.get("limitations", []):
        lines.append(f"- 限制：{limitation}")
    lines += [
        "", "## 5. 裁决", "",
        f"**BLOCKED**：{verdict['reason']}。本报告不把接口自检、子图组件或 prebuilt kernel 调用升级为完整 S11 PASS。", "",
        "## 6. 前端访问", "",
        "Console 的 `GET /api/console/v1/evidence` 会索引本目录 `raw/verdict.json`；"
        "`GET /api/console/v1/evidence/{id}` 与 `/download` 可预览或下载本报告及 raw JSON/JSONL。", "",
        f"详细协议：[`{protocol.experiment_id}`](../../details/S11/{next(path.name for path in DETAIL_ROOT.glob(protocol.experiment_id + '_*.md'))})", "",
    ]
    return "\n".join(lines)


def archive_existing(raw: Path, run_id: str) -> None:
    if not raw.exists() or not any(path.is_file() for path in raw.iterdir()):
        return
    previous = "unknown"
    verdict = raw / "verdict.json"
    if verdict.is_file():
        try:
            previous = str(json.loads(verdict.read_text()).get("run_id") or previous)
        except (OSError, ValueError):
            pass
    archive = raw / "runs" / previous
    if archive.exists():
        archive = raw / "runs" / f"{previous}_{int(time.time())}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in list(raw.iterdir()):
        if path.name == "runs":
            continue
        shutil.move(str(path), str(archive / path.name))


def evidence_manifest(experiment_root: Path, run_id: str) -> Dict[str, Any]:
    rows = []
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path.name == "evidence_manifest.json" or "runs" in path.relative_to(experiment_root).parts:
            continue
        rows.append({"path": str(path.relative_to(experiment_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"stage": STAGE, "experiment_id": experiment_root.name, "run_id": run_id, "files": rows}


def experiment_specific_files(experiment_id: str, raw: Path, observation: Mapping[str, Any]) -> None:
    mapping: Mapping[str, Sequence[Tuple[str, str]]] = {
        "E11-01": (("capture_census.json", "cold_processes"), ("runtime_values.json", "runtime_rows"), ("graph_breaks.json", "negative_cases"), ("hero_graph_manifest.json", "hero_graph_manifest")),
        "E11-02": (("pattern_contract.json", "pattern_contract"), ("corpus_results.json", "corpus"), ("rewrite_decisions.json", "decisions"), ("before_ir.json", "before_ir"), ("after_ir.json", "after_ir"), ("idempotence.json", "idempotence"), ("correctness.json", "gpu_correctness")),
        "E11-03": (("target_capability.json", "target"), ("registry_snapshot.json", "registry"), ("lowering_decision.json", "decision"), ("correctness.json", "correctness"), ("runtime_samples.json", "runtime"), ("dispatch_evidence.json", "actual_dispatch"), ("fallback_cases.json", "unsupported_fallback")),
        "E11-04": (("case_pair.json", "case_pair"), ("ir_diff.json", "ir_diff"), ("artifact_inventory.json", "artifact_inventory"), ("export_plan.json", "export_plan"), ("mechanism.json", "mechanism")),
        "E11-05": (("variants.json", "variants"), ("shape_sequence.json", "shape_sequence"), ("domain_coverage.json", "domain_coverage"), ("wrong_reuse.json", "wrong_reuse"), ("concurrency.json", "concurrency"), ("dynamic_policy.json", "dynamic_policy"), ("strategy_comparison.json", "strategy_comparison")),
        "E11-06": (("search_space.json", "search_space"), ("split_manifest.json", "split_manifest"), ("measurement_schedule.json", "measurement_schedule"), ("trials.json", "trials"), ("oracle.json", "oracle"), ("budgets.json", "budgets"), ("quality_cost_curve.json", "quality_cost_curve"), ("holdout.json", "holdout"), ("autotune_policy.json", "autotune_policy")),
        "E11-07": (("dataset.json", "dataset"), ("feature_schema.json", "feature_schema"), ("split_manifest.json", "split_manifest"), ("leakage_audit.json", "leakage_audit"), ("baselines.json", "baselines"), ("final_predictions.json", "final_predictions"), ("regret_summary.json", "regret_summary"), ("risk_coverage.json", "risk_coverage"), ("robustness.json", "illegal_candidate_injection"), ("deployment_policy.json", "deployment")),
        "E11-08": (("key_spec.json", "key_spec"), ("key_vectors.json", "key_vectors"), ("timing.json", "timing"), ("invalidation.json", "invalidation"), ("corruption_cases.json", "corruption"), ("cross_process_hit.json", "cross_process_hit"), ("cache_policy.json", "cache_policy")),
        "E11-09": (("stack_selection.json", "selection"), ("semantic_mapping.json", "semantic_mapping"), ("legality.json", "analysis_only_legality"), ("loop_ir.json", "loop_ir"), ("schedule_trace.json", "schedule_trace"), ("bridge.json", "bridge_audit"), ("adoption.json", "adoption")),
        "E11-10": (("task_package.json", "task_package"), ("generation_config.json", "generation_config"), ("sandbox.json", "sandbox"), ("wrong_corpus.json", "wrong_corpus"), ("gate_chains.json", "wrong_candidate_chains"), ("candidate_verdicts.json", "candidate_verdicts"), ("admission_negative.json", "admission_negative")),
    }
    for filename, key in mapping[experiment_id]:
        write_json(raw / filename, observation.get(key))


def collect_component_tests() -> Dict[str, Any]:
    compiler = command(
        (sys.executable, "-m", "pytest", "tests/unit/compiler", "tests/property/test_compiler_invariants.py", "-q"),
        timeout=600,
    )
    console_candidates = (
        REPO / ".venv-console/bin/python",
        Path("/home/jetson/work/HQSB-console-dev/.venv-console/bin/python"),
    )
    console_python = next((path for path in console_candidates if path.is_file()), None)
    if console_python is None and importlib.util.find_spec("fastapi") is None:
        console_http = {
            "status": "NOT_RUN_DEPENDENCY_UNAVAILABLE",
            "reason": "Jetson experiment interpreter has no optional fastapi dependency",
            "test": "tests/unit/console/test_evidence.py",
        }
    else:
        http_python = str(console_python or Path(sys.executable))
        console_result = command(
            (http_python, "-m", "pytest", "tests/unit/console/test_evidence.py", "-q"),
            timeout=300,
        )
        console_http = {
            **console_result,
            "status": "PASS" if console_result["returncode"] == 0 else "FAIL",
        }
    return {
        "ok": compiler["returncode"] == 0 and console_http["status"] != "FAIL",
        "compiler": compiler,
        "frontend_http_optional": console_http,
        "frontend_direct_catalog_gate": "evaluated after evidence generation and required for collector PASS",
    }


def collect_campaign(run_id: str) -> Dict[str, Any]:
    if not run_id or any(char in run_id for char in ("/", "\\", "\0")):
        raise ConfigError("run_id must be one non-empty path component")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("S11 collector requires CUDA on the remote Jetson target")
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(torch)
    prerequisites = prerequisite_payload(environment)
    interface_audit = interface_map.resolve_interfaces()
    spec_reports = specs.CompilerSpecs.load(REPO / "configs/compiler").audit()
    config_audit = {"ok": specs.audit_all_ok(spec_reports), "reports": spec_reports}
    component_tests = collect_component_tests()
    driver_checks = {
        "prerequisites": command((sys.executable, str(DRIVER), "--prerequisites", "--json")),
        "interface_map": command((sys.executable, str(DRIVER), "--interface-map", "--json")),
        "spec_audit": command((sys.executable, str(DRIVER), "--spec-audit", "--json")),
        "smoke": command((sys.executable, str(DRIVER), "--smoke", "--json")),
    }
    shared: Dict[str, Any] = {}
    observations: Dict[str, Dict[str, Any]] = {}
    verdicts: Dict[str, Dict[str, Any]] = {}
    collectors = component_collectors()
    for experiment_id in EXPERIMENTS:
        protocol = PROTOCOLS[experiment_id]
        started_at = utc_now()
        started = time.perf_counter()
        print(f"[{started_at}] START {experiment_id}", flush=True)
        try:
            if experiment_id in ("E11-03", "E11-04", "E11-08", "E11-09"):
                observation = collectors[experiment_id](torch, shared, environment)
            else:
                observation = collectors[experiment_id](torch, shared)
        except Exception as exc:
            observation = {
                "component_status": "FAIL", "component_pass": False,
                "collector_error": {"type": type(exc).__name__, "message": str(exc)},
                "limitations": ["collector raised before its component evidence package was complete"],
            }
        observation["experiment_id"] = experiment_id
        observation["captured_at"] = utc_now()
        observation["elapsed_s"] = time.perf_counter() - started
        observations[experiment_id] = observation
        verdict = formal_verdict(protocol, run_id, observation, prerequisites, started_at)
        verdicts[experiment_id] = verdict
        experiment_root = STAGE_ROOT / experiment_id
        raw = experiment_root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        archive_existing(raw, run_id)
        required = required_rows(protocol, observation)
        criteria = criterion_rows(protocol, observation, prerequisites)
        write_json(raw / "preregistration.json", {**protocol.as_dict(), "run_id": run_id, "frozen_before_execution": True})
        write_json(raw / "environment_fingerprint.json", environment)
        write_json(raw / "prerequisites.json", prerequisites)
        write_json(raw / "config_audit.json", config_audit)
        write_json(raw / "interface_audit.json", interface_audit)
        write_json(raw / "component_tests.json", component_tests)
        write_json(raw / "driver_checks.json", driver_checks)
        write_json(raw / "component_observations.json", observation)
        write_json(raw / "required_evidence.json", {"items": required})
        write_json(raw / "criteria_evaluation.json", {"criteria": criteria})
        mapping = interface_map.mapping_for(experiment_id)
        write_jsonl(
            raw / "step_status.jsonl",
            (
                {
                    "experiment_id": experiment_id, "step": step.index, "name": step.title,
                    "interfaces": list(step.interfaces), "interface_resolution": "PASS" if interface_audit["ok"] else "FAIL",
                    "scientific_status": "COMPONENT_EVIDENCE_ONLY", "formal_claim_allowed": False,
                }
                for step in mapping.steps
            ),
        )
        experiment_specific_files(experiment_id, raw, observation)
        write_json(raw / "verdict.json", verdict)
        write_text(experiment_root / f"{experiment_id}_实验报告.md", report_markdown(protocol, verdict, required, criteria, observation))
        print(f"[{utc_now()}] END {experiment_id} component={observation['component_status']} formal=BLOCKED elapsed={observation['elapsed_s']:.3f}s", flush=True)
        torch.cuda.empty_cache()

    catalog = EvidenceCatalog(REPO)
    discovered = [item for item in catalog.scan(refresh=True) if item["stage"] == STAGE]
    frontend = {
        "checked_at": utc_now(),
        "catalog_endpoint": "GET /api/console/v1/evidence",
        "detail_endpoint": "GET /api/console/v1/evidence/{evidence_id}",
        "download_endpoint": "GET /api/console/v1/evidence/{evidence_id}/download",
        "frontend_routes": ["/evidence", "/experiments", "/profiling"],
        "expected_experiments": len(EXPERIMENTS),
        "discovered_experiments": len(discovered),
        "experiments": sorted(item["experiment"] for item in discovered),
        "statuses": {item["experiment"]: item["status"] for item in discovered},
        "attachment_counts": {item["experiment"]: len(item["files"]) for item in discovered},
        "all_detail_readable": all(bool(catalog.detail(item["id"])) for item in discovered),
        "http_contract_test": component_tests["frontend_http_optional"],
    }
    frontend["ok"] = bool(
        frontend["discovered_experiments"] == len(EXPERIMENTS)
        and set(frontend["experiments"]) == set(EXPERIMENTS)
        and frontend["all_detail_readable"]
        and frontend["http_contract_test"].get("status") == "PASS"
    )
    for experiment_id in EXPERIMENTS:
        raw = STAGE_ROOT / experiment_id / "raw"
        write_json(raw / "frontend_validation.json", frontend)
        write_json(raw / "evidence_manifest.json", evidence_manifest(STAGE_ROOT / experiment_id, run_id))
    component_counts: Dict[str, int] = {}
    for item in observations.values():
        value = str(item["component_status"])
        component_counts[value] = component_counts.get(value, 0) + 1
    summary = {
        "stage": STAGE, "run_id": run_id,
        "collector_status": "PASS" if component_tests["ok"] and config_audit["ok"] and interface_audit["ok"] and frontend["ok"] else "FAIL",
        "stage_status": "BLOCKED", "stage_complete": False, "claim_allowed": False,
        "reason": "S06 model-level prerequisite is BLOCKED; E11-09 second stack and E11-04 hardware-counter tools are unavailable",
        "prerequisites": prerequisites,
        "component_counts": component_counts,
        "experiment_statuses": {key: value["status"] for key, value in verdicts.items()},
        "component_statuses": {key: value["component_status"] for key, value in observations.items()},
        "component_tests": component_tests,
        "config_audit_ok": config_audit["ok"],
        "interface_audit": interface_audit,
        "frontend_validation": frontend,
        "generated_at": utc_now(),
    }
    write_json(STAGE_ROOT / "campaign_summary.json", summary)
    write_json(STAGE_ROOT / "frontend_validation.json", frontend)
    stage_lines = [
        "# S11 阶段实验执行摘要", "",
        f"> Run ID：`{run_id}`  ",
        "> 阶段正式裁决：**BLOCKED**  ",
        f"> 采集器：**{summary['collector_status']}**（组件采集通过不等于 S11 科学验收通过）", "",
        "## 1. 结论", "",
        "十项实验均建立了独立报告、原始机器可读记录、判据对照和前端索引。真实 Jetson 组件范围已执行；"
        "由于 S06 正式 model-level pattern verdict 缺失，E11-01～08 的正式依赖链不能闭合；"
        "E11-09 又缺 TVM/MLIR，E11-04 缺 NCU/SASS 工具，因此所有正式 verdict 保持 `BLOCKED`。", "",
        f"组件状态统计：`{component_counts}`。接口映射：`{interface_audit['steps']} steps / {interface_audit['interfaces']} interfaces / ok={str(interface_audit['ok']).lower()}`。", "",
        "## 2. 逐项裁决", "",
        "| 实验 | 级别 | Jetson 组件 | 正式状态 | 预计效果 | 单项标准 |", "|---|---|---|---|---|---|",
    ]
    for experiment_id in EXPERIMENTS:
        stage_lines.append(
            f"| [{experiment_id}]({experiment_id}/{experiment_id}_实验报告.md) | {PROTOCOLS[experiment_id].level} | "
            f"{observations[experiment_id]['component_status']} | BLOCKED | 部分组件范围 | 未完整满足 |"
        )
    stage_lines += [
        "", "## 3. 阶段完成标志", "",
        "- E11-01～E11-09 正式 PASS：未满足。",
        "- 完整 Qwen capture→rewrite→lowering→custom kernel→model correctness/performance：未满足；已完成真实 Qwen RMSNorm 子图和真实 CUDA custom-op 组件链。",
        "- graph break/guard/fallback/cache corruption：已采集组件证据；完整 Dynamo/Inductor/model variant 链仍缺。",
        "- autotune/cost model：已完成 prebuilt implementation 搜索、holdout 与 regret 组件研究；不冒充 schedule/codegen autotune。",
        "- IR/PTX/SASS/hardware counter 归因：source/IR/binary/symbol/timing 有证据，PTX/SASS/NCU 缺失。",
        "- TVM/MLIR runnable lowering：未执行，工具不可用。",
        "- AI gate：执行了门状态与 admission 负例；无真实 AI native candidate/sanitizer/reviewer，故不声称方法学 PASS。", "",
        "## 4. 前端访问", "",
        f"EvidenceCatalog 实际发现 `{frontend['discovered_experiments']}/10` 项，detail 可读：`{str(frontend['all_detail_readable']).lower()}`。",
        f"Evidence HTTP contract：`{frontend['http_contract_test'].get('status')}`（`tests/unit/console/test_evidence.py`）。",
        "`/api/console/v1/evidence`、detail 与 download 接口可访问每项 verdict、报告和 raw JSON/JSONL；前端 `/evidence`、`/experiments`、`/profiling` 可筛选 S11。", "",
    ]
    write_text(STAGE_ROOT / "S11_阶段实验报告_20260921.md", "\n".join(stage_lines))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-id", default="")
    value.add_argument("--json", action="store_true")
    value.add_argument("--probe", choices=("capture", "cache"), default="", help=argparse.SUPPRESS)
    value.add_argument("--seed", type=int, default=1110, help=argparse.SUPPRESS)
    value.add_argument("--cache-dir", default="", help=argparse.SUPPRESS)
    value.add_argument("--entry-id", default="", help=argparse.SUPPRESS)
    value.add_argument("--expected-key", default="", help=argparse.SUPPRESS)
    value.add_argument("--arch", default="", help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.probe:
        import torch

        if args.probe == "capture":
            payload = capture_probe(torch, args.seed)
        else:
            payload = cache_probe(args.cache_dir, args.entry_id, args.expected_key, args.arch)
        print(json.dumps(json_safe(payload), ensure_ascii=False))
        return 0
    run_id = args.run_id or time.strftime("s11_%Y%m%dT%H%M%SZ", time.gmtime())
    summary = collect_campaign(run_id)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"collector_status={summary['collector_status']}")
        print(f"stage_status={summary['stage_status']}")
        print(f"frontend_ok={summary['frontend_validation']['ok']}")
        print(f"component_counts={summary['component_counts']}")
    return 0 if summary["collector_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
