"""S09 campaign orchestration and evidence policy.

The S09 protocol is hardware gated.  This module deliberately separates a
successful *collector* run from a successful *scientific* experiment:

* read-only preflight can run on any host and records why Ascend execution is
  available or unavailable;
* a missing Ascend device/CANN stack produces ``BLOCKED`` evidence, never a
  fabricated performance row or a downgraded PASS;
* E09-06 is ``N/A_BY_ADR`` while the project makes no Ascend INT8/INT4 claim;
* every experiment records the expected effect, required data and every PASS
  criterion so the browser can expose both the result and the missing proof.

Formal evidence is written only by :func:`collect_campaign`, which the CLI
requires the caller to request explicitly.  Importing this module is read-only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from hqsb.ascend.capability import CAPABILITY_DOMAINS, UNKNOWN
from hqsb.ascend.compatibility import CompatibilityManifest, UNAVAILABLE
from hqsb.ascend.operators import spec_for
from hqsb.ascend.probes import MATCHING_CHAIN, SubprocessExecutor, run_stack_probes
from hqsb.core.errors import ConfigError

STAGE = "S09"
STATUS_PASS = "PASS"
STATUS_PASS_NEGATIVE = "PASS_NEGATIVE"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_N_A_BY_ADR = "N/A_BY_ADR"

PROTOCOL_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)
CONCLUSION_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
)
EXPERIMENTS: Tuple[str, ...] = tuple(f"E09-{index:02d}" for index in range(1, 11))


@dataclass(frozen=True)
class ExperimentProtocol:
    experiment_id: str
    level: str
    title: str
    question: str
    hypothesis: str
    expected_effect: str
    required_data: Tuple[str, ...]
    pass_criteria: Tuple[str, ...]
    dependencies: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "level": self.level,
            "title": self.title,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "expected_effect": self.expected_effect,
            "required_data": list(self.required_data),
            "pass_criteria": list(self.pass_criteria),
            "dependencies": list(self.dependencies),
        }


def _protocol(
    experiment_id: str,
    level: str,
    title: str,
    question: str,
    hypothesis: str,
    expected_effect: str,
    required_data: Sequence[str],
    pass_criteria: Sequence[str],
    dependencies: Sequence[str] = (),
) -> ExperimentProtocol:
    return ExperimentProtocol(
        experiment_id=experiment_id,
        level=level,
        title=title,
        question=question,
        hypothesis=hypothesis,
        expected_effect=expected_effect,
        required_data=tuple(required_data),
        pass_criteria=tuple(pass_criteria),
        dependencies=tuple(dependencies),
    )


PROTOCOLS: Mapping[str, ExperimentProtocol] = {
    "E09-01": _protocol(
        "E09-01", "P0", "Ascend 环境兼容矩阵、Capability 与可复现工具链",
        "目标机的软件/硬件组合是否支持并可复现 S09？",
        "只有芯片、固件、驱动、CANN、框架和 profiler 的匹配链均被本机探针验证，后续实验才可执行。",
        "先固定可复现工具链和限制。",
        ("完整 compatibility manifest 与 canonical hash", "官方兼容依据", "device query 与健康/内存", "compile/load/async-run/framework/profile probes", "capability 四态表", "错版本/错 target/能力缺失错误", "字段级 drift 与脱敏审计"),
        ("manifest schema 与 hash 可复算", "精确版本有官方兼容依据", "device/memory/compile/load/run/framework/profile 探针按声明通过", "所有目标 capability 有证据", "安全错配注入明确失败", "无静默 fallback、错算、hang 或泄漏", "独立新进程复现且证据同源"),
    ),
    "E09-02": _protocol(
        "E09-02", "P0", "Add/Reduction Reference→Ascend C 工具链与边界验证",
        "简单算子能否贯通编译、Tiling、加载、异步调用和 profile？",
        "共同 spec/oracle/vector 下，尾块、多 tile、多核和错误输入均可被正确处理。",
        "用简单算子验证完整工具链。",
        ("Add/Reduction specs 与 oracle", "共享 test vectors/guard", "编译命令/日志/制品 hash", "Tiling 输入输出/block_dim/GM/UB/workspace", "correctness rows", "裸 runtime 与非默认 stream calls", "benchmark/profile", "负向输入与独立复现"),
        ("两算子 spec/oracle/vector/source/制品完整", "aligned/tail/multi-tile/multi-core/dtype/错误 shape 覆盖", "correctness/guard/输入完整性通过", "Tiling 检查溢出/UB/workspace/block_dim/schema", "裸 runtime/非默认 stream 正确且无隐式全局同步", "compile/load/run/profile 同 run 绑定", "负向输入可诊断且新进程可复现"),
        ("E09-01",),
    ),
    "E09-03": _protocol(
        "E09-03", "P0", "RMSNorm 跨后端正确性与性能基线",
        "同一 RMSNorm OperatorSpec 能否在 PyTorch/CUDA/Triton/Ascend 上共同验收？",
        "Ascend reference 在不放宽容差的前提下对齐，并显式计入格式转换。",
        "建立跨后端语义与性能基线。",
        ("唯一 OperatorSpec/hash", "case split 与 test vectors", "实现身份", "完整误差指标与中间量", "format conversion", "device/event 与 E2E latency/GB/s", "actual backend/kernel/fallback trace", "confirmation verdict"),
        ("所有实现共用 C3 spec/vector/comparator", "eps/累加/gamma/layout/NaN/Inf/错误语义完整", "Ascend reference 覆盖关键 shape 并通过共同门", "actual path 与转换可证明", "计时/字节模型/重复统计完整", "confirmation 未被调参污染", "全部产物由 run_id 绑定"),
        ("E09-01", "E09-02"),
    ),
    "E09-04": _protocol(
        "E09-04", "P0", "Tiling、流水与 msprof 优化闭环",
        "Tiling/多核/双缓冲优化为何有效或退化？",
        "至少一个 profile 支撑的优化假设能在独立确认集上成立，并保留退化边界。",
        "得到 NPU 优化闭环和架构解释。",
        ("预注册假设与指标", "V0/V1..实现身份", "shape census 与成本模型", "全部候选/拒绝原因", "correctness/sweep/ablation raw", "fast/slow msprof", "Roofline", "confirmation/adversarial", "shape-aware policy"),
        ("V0 与优化版共同通过正确性/guard", "候选/过滤/UB/workspace/ABI/raw 可审计", "profile+ablation 支持至少一个因果假设", "confirmation/adversarial 给出适用与退化域", "策略按 shape/capability 且有 fallback", "host/runtime/kernel/字节/扰动分开", "独立复现且身份绑定"),
        ("E09-01", "E09-03"),
    ),
    "E09-05": _protocol(
        "E09-05", "P0", "框架、Current Stream、Workspace 与 Fallback 集成",
        "自定义算子能否安全进入框架和模型执行路径？",
        "current stream、两段式 workspace、显式 fallback 和生命周期均可观测且无隐式同步。",
        "证明算子能安全进入框架/模型。",
        ("集成 ADR 与 ABI/schema", "capability predicates", "package identity", "tensor/stream/workspace/fallback/concurrency cases", "dispatcher/stream/allocator traces", "correctness 与 kernel-vs-E2E", "资源时序/async faults", "model smoke"),
        ("集成路径/ABI/capability/fallback 有 ADR", "默认与非默认 stream 正确且无不必要全局同步", "workspace 生命周期/OOM 安全可观测", "metadata/layout/format 支持/转换/拒绝明确", "actual/requested/fallback 有 trace", "并发/异步错误可恢复且无泄漏", "C4/C6/C7 不分叉且模型 smoke 通过"),
        ("E09-01", "E09-03", "E09-04"),
    ),
    "E09-06": _protocol(
        "E09-06", "P1", "低精度、QuantArtifact 与 Layout/Format Conversion",
        "Ascend INT8/INT4 是否真实命中、质量合格且端到端值得使用？",
        "若激活低比特声明，actual kernel 证据将区分 storage/transport/compute precision。",
        "判断 NPU 原生低精度和转换开销。",
        ("scope activation ADR", "C5 schema/manifest/compatibility", "golden packed bytes 与 roundtrip", "offline/load/runtime conversion", "INT8/INT4 correctness", "kernel-only/conversion-inclusive benchmark", "hidden/logits/token/quality", "memory/energy", "artifact mismatch/capability matrix"),
        ("C5 lineage/packing/scale/layout 可复算", "pack/unpack/dequant 与 op correctness 通过", "三层 precision 与 actual kernel 分开", "不兼容制品 launch 前字段级拒绝", "所有 conversion 时间/字节完整", "质量先过门再报告性能", "支持范围与验证层级进入 capability"),
        ("E09-01", "E09-05"),
    ),
    "E09-07": _protocol(
        "E09-07", "P0", "Qwen Model-core 正确性与性能闭环",
        "算子能力能否转化为同 token/workload 的模型级收益？",
        "只替换 custom RMSNorm 的 A/B 在模型 gate 通过后，prefill/decode 收益可被归因。",
        "达到模型级而非单算子演示。",
        ("tiny/representative ModelArtifact", "WorkloadSpec/token artifacts", "A/B identity", "cold load", "hidden/logits/tokens/quality", "actual-op coverage/fallback", "prefill/decode raw/summary", "resources/traces/Amdahl", "confirmation"),
        ("模型/tokenizer/workload 可复算", "A/B 唯一变量与 actual path 可证明", "tensor/logits/token/质量门通过", "prefill/decode 指标边界明确", "转换/workspace/runtime/memory/Amdahl 可解释", "fallback/长上下文/稳定性边界完整", "独立 confirmation 且 C1-C7 不分叉"),
        ("E09-01", "E09-03", "E09-05"),
    ),
    "E09-08": _protocol(
        "E09-08", "P0", "Model-core msprof、跨层 Trace 与 Roofline",
        "模型瓶颈在哪里，micro speedup 为何转化或不转化？",
        "module→op→task→kernel 关联和经验上界能解释 prefill/decode 热点。",
        "得到 NPU 端瓶颈和 Roofline 解释。",
        ("profiling questions/matrix", "metric manifest 与 overhead", "raw profiler/timeline", "module-op-kernel mapping", "prefill/decode census/hotspots", "RMSNorm/MatMul/format-copy deep dives", "byte/FLOP/ceilings/Roofline", "Amdahl reconciliation/taxonomy/recommendations", "confirmation/S10-S12 export"),
        ("metric/mode/overhead/version/raw 完整", "跨层多对多映射可审计", "prefill/decode 分开且 shape-aware", "RMSNorm 与主要热点有 counter/模型/上界", "micro→model Amdahl 差额量化", "分类/建议有对照或消融确认", "公共摘要可被下游消费且未知项不伪装"),
        ("E09-01", "E09-07"),
    ),
    "E09-09": _protocol(
        "E09-09", "P0", "CUDA↔Ascend 同协议公平比较",
        "两种硬件如何在同语义、同任务和 best-valid 三层下公平比较？",
        "pairability 先隔离身份不一致项，再把差异分解为架构/kernel/runtime/capacity/maturity。",
        "建立公平硬件映射和适用边界。",
        ("claims preregistration", "CUDA/Ascend manifests", "pairability rules/pairs/non-comparable", "empirical ceilings", "共同 correctness", "Tier A/B/C rows", "energy/profile taxonomy/maturity", "run-level effects/claim audit", "S12 export"),
        ("Tier A/B/C 与 non-claim 明确", "pairability 字段级隔离", "共同 correctness/quality 先通过且 actual path 可见", "kernel/conversion/model phase/capacity 分开", "计时/上界/内存/能耗/统计透明", "差异分解含 residual", "结论绑定身份并可导出 S12"),
        ("E09-03", "E09-07", "E09-08", "second_backend"),
    ),
    "E09-10": _protocol(
        "E09-10", "P0", "故障兼容性、恢复与 Runbook",
        "版本/能力/制品/编译/Tiling/workspace/runtime/OOM/profiler 故障能否安全失败并恢复？",
        "fail-fast、watchdog、关联 ID、cleanup 和健康探针可阻止挂死/错算/泄漏。",
        "形成迁移 runbook 和稳健降级边界。",
        ("安全 scope/abort", "健康 baseline", "error taxonomy/fault matrix", "structured logs/watchdog", "resource before/after", "各层 fault rows", "retry/fallback", "post-fault probes/MTTR", "runbook 与 blind drill"),
        ("fault matrix 覆盖九类层次", "每个 fault 安全可重复且有 watchdog", "可预检错误在 launch 前 fail-fast", "无错算/hang/越界/partial-success/递归 fallback/泄漏", "异步错误可关联且 retry/fallback 受分类约束", "逐 fault 资源审计与健康恢复", "blind drill 在时限内给正确下一步"),
        ("E09-01", "E09-02", "E09-05", "E09-07"),
    ),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_tree_hash(root: Path) -> str:
    """Hash every S09 source/config file, including untracked working-tree files."""
    digest = hashlib.sha256()
    roots = (
        root / "hqsb/ascend",
        root / "ops/ascend",
        root / "scripts/ascend",
        root / "configs/ascend",
    )
    files = []
    for source_root in roots:
        if source_root.is_file():
            files.append(source_root)
        elif source_root.is_dir():
            files.extend(
                path
                for path in source_root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    for path in sorted(files):
        relative = str(path.relative_to(root)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _run(argv: Sequence[str], *, cwd: Path, timeout_s: float = 30.0) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv), cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
        return {
            "argv": list(argv),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "timed_out": False,
        }
    except FileNotFoundError as exc:
        return {
            "argv": list(argv), "returncode": 127, "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "returncode": -1,
            "stdout": exc.stdout or "", "stderr": exc.stderr or "",
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "timed_out": True,
        }


def git_state(root: Path) -> Dict[str, Any]:
    commit = _run(("git", "rev-parse", "HEAD"), cwd=root)
    status = _run(("git", "status", "--porcelain"), cwd=root)
    diff = _run(("git", "diff", "--binary", "--", "hqsb/ascend", "ops/ascend", "scripts/ascend", "configs/ascend"), cwd=root)
    return {
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else UNAVAILABLE,
        "dirty": bool(status["stdout"].strip()) if status["returncode"] == 0 else True,
        "tracked_diff_sha256": sha256_bytes(diff["stdout"].encode("utf-8")),
        "source_tree_sha256": source_tree_hash(root),
        "git_probe_returncodes": {"commit": commit["returncode"], "status": status["returncode"], "diff": diff["returncode"]},
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return ""


@dataclass(frozen=True)
class CapabilitySnapshot:
    collected_at: str
    host: Mapping[str, Any]
    commands: Tuple[Mapping[str, Any], ...]
    tool_paths: Mapping[str, str]
    device_nodes: Mapping[str, bool]
    python_modules: Mapping[str, bool]
    stack_probe_summary: Mapping[str, Any]
    compatibility_manifest: Mapping[str, Any]
    compatibility_manifest_sha256: str
    compatibility_schema: Mapping[str, Any]
    capabilities: Tuple[Mapping[str, Any], ...]
    ascend_ready: bool
    blocker_codes: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "collected_at": self.collected_at,
            "host": dict(self.host),
            "commands": [dict(row) for row in self.commands],
            "tool_paths": dict(self.tool_paths),
            "device_nodes": dict(self.device_nodes),
            "python_modules": dict(self.python_modules),
            "stack_probe_summary": dict(self.stack_probe_summary),
            "compatibility_manifest": dict(self.compatibility_manifest),
            "compatibility_manifest_sha256": self.compatibility_manifest_sha256,
            "compatibility_schema": dict(self.compatibility_schema),
            "capabilities": [dict(row) for row in self.capabilities],
            "ascend_ready": self.ascend_ready,
            "blocker_codes": list(self.blocker_codes),
        }


def collect_capability_snapshot(root: str | Path) -> CapabilitySnapshot:
    """Collect read-only S09 preflight evidence from the current execution host."""
    repo = Path(root).resolve()
    collected_at = utc_now()
    tools = {
        name: shutil.which(name) or UNAVAILABLE
        for name in ("npu-smi", "ccec", "bisheng", "msprof")
    }
    device_nodes = {
        path: Path(path).exists()
        for path in ("/dev/davinci0", "/dev/davinci_manager", "/dev/npu0")
    }
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "torch_npu")
    }
    commands = tuple(
        _run(argv, cwd=repo)
        for argv in (
            ("uname", "-a"),
            ("npu-smi", "info"),
            ("ccec", "--version"),
            ("bisheng", "--version"),
            ("msprof", "--version"),
        )
    )
    stack = run_stack_probes(
        SubprocessExecutor(),
        cann_root="/usr/local/Ascend" if Path("/usr/local/Ascend").is_dir() else "",
    ).as_dict()
    git = git_state(repo)
    model = _read_text(Path("/proc/device-tree/model")) or UNAVAILABLE
    os_release = _read_text(Path("/etc/os-release")) or UNAVAILABLE
    host = {
        "hostname": platform.node() or UNAVAILABLE,
        "machine": platform.machine() or UNAVAILABLE,
        "platform": platform.platform() or UNAVAILABLE,
        "kernel": platform.release() or UNAVAILABLE,
        "board_model": model,
        "python": sys.version,
    }
    absent_device = not any(device_nodes.values())
    compiler_available = any(tools[name] != UNAVAILABLE for name in ("ccec", "bisheng"))
    blockers = []
    if tools["npu-smi"] == UNAVAILABLE or absent_device:
        blockers.append("ASCEND_DEVICE_NOT_PRESENT")
    if not Path("/usr/local/Ascend").is_dir():
        blockers.append("CANN_ROOT_NOT_PRESENT")
    if not compiler_available:
        blockers.append("ASCEND_COMPILER_NOT_PRESENT")
    if tools["msprof"] == UNAVAILABLE:
        blockers.append("MSPROF_NOT_PRESENT")
    if not modules["torch_npu"]:
        blockers.append("TORCH_NPU_NOT_PRESENT")

    unavailable_reason = "target is not an Ascend/CANN host: " + ", ".join(blockers)
    manifest = CompatibilityManifest(
        hardware={
            "host_id": platform.node() or UNAVAILABLE,
            "board_id": model,
            "chip_sku": UNAVAILABLE,
            "soc_version": UNAVAILABLE,
            "device_count": 0,
            "logical_to_physical": {},
            "visible_device_policy": UNAVAILABLE,
            "cpu_arch": platform.machine() or UNAVAILABLE,
            "os": os_release,
            "kernel": platform.release() or UNAVAILABLE,
            "container_runtime": UNAVAILABLE,
            "image_digest": UNAVAILABLE,
            "health": {"status": UNAVAILABLE, "reason": unavailable_reason},
        },
        ascend_stack={
            "firmware_version": UNAVAILABLE,
            "firmware_components": {},
            "driver_version": UNAVAILABLE,
            "driver_install_mode": UNAVAILABLE,
            "driver_loaded_modules": [],
            "cann_toolkit_version": UNAVAILABLE,
            "cann_runtime_version": UNAVAILABLE,
            "cann_compiler_version": UNAVAILABLE,
            "cann_ops_package_version": UNAVAILABLE,
            "cann_kernel_package_version": UNAVAILABLE,
            "set_env_source": UNAVAILABLE,
            "library_resolution": {},
            "compiler_path": tools["ccec"] if tools["ccec"] != UNAVAILABLE else tools["bisheng"],
            "msprof_version": UNAVAILABLE,
            "msprof_metric_sets": [],
            "install_roots": [],
        },
        framework={
            "python_version": platform.python_version(),
            "pytorch_version": "installed-version-not-imported" if modules["torch"] else UNAVAILABLE,
            "torch_npu_version": "installed-version-not-imported" if modules["torch_npu"] else UNAVAILABLE,
            "mindspore_version": UNAVAILABLE,
            "atb_version": UNAVAILABLE,
            "mindie_version": UNAVAILABLE,
            "package_hashes": {},
            "cpp_abi": UNAVAILABLE,
            "compile_flags": [],
            "selected_integration_path": UNAVAILABLE,
        },
        project={
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "source_patch_hash": git["source_tree_sha256"],
            "source_tree_hash": git["source_tree_sha256"],
            "tracked_diff_hash": git["tracked_diff_sha256"],
            "model_artifact_hash": UNAVAILABLE,
            "operator_spec_hash": spec_for("hqsb.rms_norm").spec_hash,
            "quant_artifact_hash": UNAVAILABLE,
        },
        official_sources=[],
        evidence={
            "collector": "scripts/ascend/run_e09.py",
            "command_count": len(commands),
            "blocker_codes": blockers,
            "note": "No target Ascend SKU/version exists on this host, so no compatibility page is selected or guessed.",
        },
        collected_at=collected_at,
        device_id=UNAVAILABLE,
        requested_backend="ascend",
    )
    validation = manifest.validate().as_dict()
    capabilities = tuple(
        {
            "key": f"{domain}.target_host",
            "domain": domain,
            "state": UNKNOWN,
            "scope": "IN_SCOPE",
            "usable": False,
            "reason": unavailable_reason,
            "probe_status": "UNAVAILABLE",
            "evidence": ["raw/capability_snapshot.json"],
        }
        for domain in CAPABILITY_DOMAINS
    )
    return CapabilitySnapshot(
        collected_at=collected_at,
        host=host,
        commands=commands,
        tool_paths=tools,
        device_nodes=device_nodes,
        python_modules=modules,
        stack_probe_summary=stack,
        compatibility_manifest=manifest.redacted().as_dict(),
        compatibility_manifest_sha256=manifest.sha256,
        compatibility_schema=validation,
        capabilities=capabilities,
        ascend_ready=not blockers,
        blocker_codes=tuple(blockers),
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )


def _archive_existing_raw(raw: Path, new_run_id: str) -> None:
    """Preserve a previous formal snapshot before refreshing the browser head."""
    verdict_path = raw / "verdict.json"
    if not verdict_path.is_file():
        return
    try:
        previous = json.loads(verdict_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
    previous_run_id = str(previous.get("run_id", "unknown_previous_run"))
    if previous_run_id == new_run_id:
        return
    if (
        not previous_run_id
        or previous_run_id in (".", "..")
        or any(char in previous_run_id for char in ("/", "\\", "\0"))
    ):
        previous_run_id = "unknown_previous_run"
    archive = raw / "runs" / previous_run_id
    if archive.exists():
        raise ConfigError(
            f"refusing to overwrite archived S09 evidence {archive}"
        )
    archive.mkdir(parents=True)
    for source in sorted(raw.iterdir()):
        if source.name == "runs":
            continue
        target = archive / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.is_file():
            shutil.copy2(source, target)


def _criterion_rows(protocol: ExperimentProtocol, *, satisfied: bool, reason: str) -> list[Dict[str, Any]]:
    return [
        {
            "criterion_id": f"C{index:02d}",
            "criterion": criterion,
            "satisfied": satisfied,
            "status": "PASS" if satisfied else "NOT_EVALUABLE",
            "reason": reason,
        }
        for index, criterion in enumerate(protocol.pass_criteria, 1)
    ]


_STEP_HEADING = re.compile(r"^### 步骤\s+(\d+)[：:]\s*(.+?)\s*$")


def protocol_steps(root: str | Path, experiment_id: str) -> list[Dict[str, Any]]:
    """Read the 28 canonical step titles from the detailed S09 protocol."""
    if experiment_id not in EXPERIMENTS:
        raise ConfigError(f"unknown S09 experiment {experiment_id!r}")
    detail_root = Path(root).resolve() / "docs" / "stage_experiments" / "details" / "S09"
    matches = sorted(detail_root.glob(f"{experiment_id}_*.md"))
    if len(matches) != 1:
        raise ConfigError(
            f"expected exactly one detail protocol for {experiment_id}, found {len(matches)}"
        )
    rows = []
    for line in matches[0].read_text(encoding="utf-8").splitlines():
        match = _STEP_HEADING.match(line)
        if match:
            rows.append({"step": int(match.group(1)), "title": match.group(2)})
    expected = list(range(1, 29))
    if [row["step"] for row in rows] != expected:
        raise ConfigError(
            f"{experiment_id} detail protocol must contain steps 1..28 in order"
        )
    return rows


def _step_status_rows(
    root: Path,
    protocol: ExperimentProtocol,
    verdict: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    status = "N/A_BY_ADR" if verdict["status"] == STATUS_N_A_BY_ADR else "NOT_RUN"
    reason = (
        "experiment deactivated by the recorded scope ADR"
        if status == "N/A_BY_ADR"
        else "E09-01 Ascend device/CANN hard gate failed before this step could execute"
    )
    rows = []
    for item in protocol_steps(root, protocol.experiment_id):
        collected = protocol.experiment_id == "E09-01" and item["step"] in {
            2, 3, 4, 5, 6, 7, 8, 10, 11, 26, 28
        }
        rows.append(
            {
                **item,
                "status": "COLLECTED_PREFLIGHT_ONLY" if collected else status,
                "scientific_execution": False,
                "claim_allowed": False,
                "reason": (
                    "read-only host preflight recorded; the matching execution chain is still incomplete"
                    if collected
                    else reason
                ),
            }
        )
    return rows


def _required_rows(protocol: ExperimentProtocol, *, experiment_id: str) -> list[Dict[str, Any]]:
    rows = []
    for index, item in enumerate(protocol.required_data, 1):
        preflight = experiment_id == "E09-01" and index in (1, 3, 4, 5)
        rows.append(
            {
                "item_id": f"D{index:02d}",
                "required": item,
                "availability": "COLLECTED_PREFLIGHT_ONLY" if preflight else "NOT_COLLECTED",
                "scientific_sample": False,
                "reason": (
                    "read-only preflight recorded; no matching Ascend execution chain"
                    if preflight
                    else "blocked before allocation/compile/launch by E09-01 capability gate"
                ),
            }
        )
    return rows


def _verdict_for(protocol: ExperimentProtocol, snapshot: CapabilitySnapshot) -> Dict[str, Any]:
    if protocol.experiment_id == "E09-06":
        return {
            "status": STATUS_N_A_BY_ADR,
            "reason_code": "ASCEND_LOW_PRECISION_CLAIM_DISABLED",
            "reason": "The project does not claim Ascend INT8/INT4 execution; P1 is deactivated by the recorded ADR.",
            "dependencies_satisfied": False,
        }
    if not snapshot.ascend_ready:
        code = "NOT_RUN_ON_SECOND_BACKEND" if protocol.experiment_id == "E09-09" else "BLOCKED_BY_ASCEND_CAPABILITY"
        return {
            "status": STATUS_BLOCKED,
            "reason_code": code,
            "reason": "Ascend/CANN execution prerequisites are unavailable: " + ", ".join(snapshot.blocker_codes),
            "dependencies_satisfied": False,
        }
    return {
        "status": STATUS_BLOCKED,
        "reason_code": "EXECUTION_NOT_IMPLEMENTED_BY_COLLECTOR",
        "reason": "Preflight passed, but the collector is not allowed to infer an experimental conclusion without raw samples.",
        "dependencies_satisfied": False,
    }


def smoke_self_check() -> Dict[str, Any]:
    """Exercise host-side S09 contracts; the result is never hardware evidence."""
    from hqsb.ascend.operators import oracle_add, oracle_rmsnorm, oracle_row_reduce_sum
    from hqsb.ascend.tiling import TilingLimits, TilingRequest, UbBudget, compute_tiling

    budget = UbBudget(
        nominal_bytes=64 * 1024,
        framework_reserved_bytes=4096,
        method="synthetic CPU smoke fixture; not a device capability",
    )
    limits = TilingLimits(
        max_block_dim=8,
        max_workspace_bytes=1 << 20,
        alignment_elems=8,
        ub_budget=budget,
        source="synthetic CPU smoke fixture",
    )
    requests = {
        "add_tail": TilingRequest(rows=3, hidden=17, dtype="fp16", block_dim=2, tile_elems=8, rowwise=False, ub_model="elementwise"),
        "reduce_tail": TilingRequest(rows=3, hidden=17, dtype="fp16", block_dim=2, tile_elems=8, ub_model="reduce"),
        "rmsnorm_tail": TilingRequest(rows=3, hidden=17, dtype="fp16", block_dim=2, tile_elems=8, ub_model="rmsnorm"),
    }
    tiling = {}
    for name, request in requests.items():
        decision, rejection = compute_tiling(request, limits)
        tiling[name] = decision.as_dict() if decision else rejection.as_dict()
    x = [[1.0, -2.0], [3.0, 4.0]]
    rms, means, rstd = oracle_rmsnorm(x, [1.0, 0.5], 1e-6, return_intermediates=True)
    return {
        "status": "SMOKE_PASS",
        "simulated": True,
        "claim_allowed": False,
        "note": "CPU host-contract self-check only; not Ascend execution, latency, profile, or correctness evidence.",
        "tiling": tiling,
        "oracles": {
            "add": oracle_add([1.0, 2.0], [3.0, -1.0]),
            "row_reduce_sum": oracle_row_reduce_sum(x),
            "rmsnorm": rms,
            "rmsnorm_mean_square": means,
            "rmsnorm_rstd": rstd,
        },
    }


def _report_markdown(
    protocol: ExperimentProtocol,
    verdict: Mapping[str, Any],
    required_rows: Sequence[Mapping[str, Any]],
    criteria: Sequence[Mapping[str, Any]],
    snapshot: CapabilitySnapshot,
    run_id: str,
) -> str:
    lines = [
        f"# {protocol.experiment_id} 实验报告：{protocol.title}",
        "",
        f"> Run ID：`{run_id}`  ",
        f"> 科学裁决：**{verdict['status']}**  ",
        f"> 原因码：`{verdict['reason_code']}`  ",
        "> 采集器成功退出只代表证据已写盘，不代表实验通过。",
        "",
        "## 1. 问题、假设与预计效果",
        "",
        f"- 问题：{protocol.question}",
        f"- 预注册假设：{protocol.hypothesis}",
        f"- 预计达到的效果：{protocol.expected_effect}",
        f"- 依赖：{', '.join(protocol.dependencies) if protocol.dependencies else 'E09-01 自身环境门禁'}",
        "",
        "## 2. 实际环境与执行边界",
        "",
        f"执行主机为 `{snapshot.host.get('board_model', UNAVAILABLE)}` / `{snapshot.host.get('machine', UNAVAILABLE)}`。",
        f"Ascend readiness=`{str(snapshot.ascend_ready).lower()}`；阻塞项：`{', '.join(snapshot.blocker_codes)}`。",
        "本次没有分配 Ascend 内存、编译/加载 Ascend binary、启动 kernel、运行模型或采集 msprof；raw sample 数为 0。",
        "",
        "## 3. 必采集信息/数据",
        "",
        "| ID | 必采集项 | 当前可用性 | 是否科学样本 | 说明 |",
        "|---|---|---|---:|---|",
    ]
    for row in required_rows:
        lines.append(
            f"| {row['item_id']} | {row['required']} | {row['availability']} | {'是' if row['scientific_sample'] else '否'} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 单项通过标准对照",
            "",
            "| ID | 单项标准 | 是否满足 | 状态/原因 |",
            "|---|---|---:|---|",
        ]
    )
    for row in criteria:
        lines.append(
            f"| {row['criterion_id']} | {row['criterion']} | {'是' if row['satisfied'] else '否'} | {row['status']}：{row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 裁决",
            "",
            f"**{verdict['status']}**：{verdict['reason']}",
            "",
            "这不符合预计效果，也不满足单项通过标准。CPU smoke 仅验证 host-side 契约可运行，`claim_allowed=false`，不得替代 Ascend 实测。",
            "",
            "## 6. 证据与前端访问",
            "",
            "前端通过 `GET /api/console/v1/evidence` 发现本实验的 `raw/verdict.json`；其余 JSON/JSONL/TXT 和本报告可通过 evidence detail/download 接口读取。",
            "`raw/evidence_manifest.json` 保存每个文本证据的相对路径、字节数和 SHA-256。",
            "",
        ]
    )
    return "\n".join(lines)


def collect_campaign(
    root: str | Path,
    *,
    output_root: str | Path = "docs/stage_experiments/S09",
    experiment_ids: Sequence[str] = EXPERIMENTS,
    run_id: str = "",
) -> Dict[str, Any]:
    """Run the read-only preflight and persist honest S09 evidence/report files."""
    repo = Path(root).resolve()
    selected = tuple(experiment_ids)
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise ConfigError(f"unknown S09 experiment(s): {unknown}")
    resolved_output = Path(output_root)
    if not resolved_output.is_absolute():
        resolved_output = repo / resolved_output
    allowed = (repo / "docs" / "stage_experiments" / "S09").resolve()
    if resolved_output.resolve() != allowed:
        raise ConfigError("S09 formal evidence must be written to docs/stage_experiments/S09")
    campaign_run_id = run_id or time.strftime("s09_%Y%m%dT%H%M%SZ", time.gmtime())
    if any(char in campaign_run_id for char in ("/", "\\", "\0")) or not campaign_run_id:
        raise ConfigError("run_id must be one non-empty path component")

    snapshot = collect_capability_snapshot(repo)
    smoke = smoke_self_check()
    outputs = []
    statuses: Dict[str, str] = {}

    for experiment_id in selected:
        protocol = PROTOCOLS[experiment_id]
        experiment_root = allowed / experiment_id
        raw = experiment_root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        _archive_existing_raw(raw, campaign_run_id)
        verdict_base = _verdict_for(protocol, snapshot)
        required_rows = _required_rows(protocol, experiment_id=experiment_id)
        criterion_reason = (
            "not activated by scope ADR"
            if verdict_base["status"] == STATUS_N_A_BY_ADR
            else "not evaluable because the E09-01 hardware/toolchain gate is blocked"
        )
        criteria = _criterion_rows(protocol, satisfied=False, reason=criterion_reason)
        verdict = {
            "stage": STAGE,
            "experiment_id": experiment_id,
            "run_id": campaign_run_id,
            "level": protocol.level,
            "overall": verdict_base["status"],
            "status": verdict_base["status"],
            "scientific_execution_verdict": verdict_base["status"],
            "collector_status": "PASS",
            "reason_code": verdict_base["reason_code"],
            "reason": verdict_base["reason"],
            "execution_attempted": experiment_id == "E09-01",
            "ascend_kernel_launched": False,
            "raw_samples": 0,
            "simulated_samples": 0,
            "claim_allowed": False,
            "dependencies": list(protocol.dependencies),
            "dependencies_satisfied": verdict_base["dependencies_satisfied"],
            "compatibility_manifest_sha256": snapshot.compatibility_manifest_sha256,
            "git_commit": snapshot.compatibility_manifest["project"]["git_commit"],
            "git_dirty": snapshot.compatibility_manifest["project"]["git_dirty"],
            "started_at": snapshot.collected_at,
            "ended_at": utc_now(),
            "limitations": [
                "no Ascend device on the configured execution target",
                "no CANN compiler/runtime or msprof",
                "no torch_npu framework execution path",
                "CPU smoke is interface-only and cannot support an Ascend claim",
            ],
        }
        preregistration = {
            **protocol.as_dict(),
            "run_id": campaign_run_id,
            "frozen_before_device_execution": True,
            "device_execution_started": False,
        }
        _write_json(raw / "preregistration.json", preregistration)
        _write_json(raw / "capability_snapshot.json", snapshot.as_dict())
        _write_json(raw / "required_evidence.json", {"items": required_rows})
        _write_json(raw / "criteria_evaluation.json", {"criteria": criteria})
        _write_jsonl(raw / "step_status.jsonl", _step_status_rows(repo, protocol, verdict))
        _write_json(raw / "smoke.json", smoke)
        _write_json(raw / "verdict.json", verdict)
        _write_jsonl(raw / "commands.jsonl", snapshot.commands)

        if experiment_id == "E09-01":
            _write_json(raw / "compatibility_manifest.json", snapshot.compatibility_manifest)
            _atomic_text(raw / "compatibility_manifest.canonical.json", CompatibilityManifest.from_document(snapshot.compatibility_manifest).canonical() + "\n")
            _atomic_text(raw / "compatibility_manifest.sha256", snapshot.compatibility_manifest_sha256 + "\n")
            _write_json(raw / "schema_validation.json", snapshot.compatibility_schema)
            _write_json(raw / "probe_summary.json", snapshot.stack_probe_summary)
            _write_json(raw / "capabilities.json", {"entries": list(snapshot.capabilities)})
            npu = next(row for row in snapshot.commands if row["argv"][:1] == ["npu-smi"])
            _atomic_text(raw / "device_query.txt", (npu["stdout"] + npu["stderr"]).strip() + "\n")
            _write_json(
                raw / "matching_chain.json",
                {
                    "required": list(MATCHING_CHAIN),
                    "complete": False,
                    "statuses": {
                        probe: snapshot.stack_probe_summary.get("statuses", {}).get(probe, "NOT_RUN")
                        for probe in MATCHING_CHAIN
                    },
                    "first_non_pass": "device_query",
                },
            )
        elif experiment_id == "E09-06":
            _write_json(
                raw / "scope_activation.json",
                {
                    "activated": False,
                    "status": STATUS_N_A_BY_ADR,
                    "claim": "No Ascend INT8/INT4 execution, quality, memory, energy or speed claim is made.",
                    "reactivation_condition": "A target Ascend SKU/CANN stack and explicit low-precision claim are approved.",
                },
            )
        elif experiment_id == "E09-09":
            _write_json(
                raw / "pairability_preflight.json",
                {
                    "pairable": False,
                    "status": "NOT_RUN_ON_SECOND_BACKEND",
                    "missing_side": "ascend",
                    "cuda_or_jetson_history_substituted": False,
                    "vendor_numbers_substituted": False,
                },
            )

        report_path = experiment_root / f"{experiment_id}_实验报告.md"
        _atomic_text(
            report_path,
            _report_markdown(protocol, verdict, required_rows, criteria, snapshot, campaign_run_id),
        )
        manifest_rows = []
        for path in sorted(raw.rglob("*")):
            relative_to_raw = path.relative_to(raw)
            if (
                path.is_file()
                and path.name != "evidence_manifest.json"
                and "runs" not in relative_to_raw.parts
            ):
                manifest_rows.append(
                    {
                        "path": str(path.relative_to(experiment_root)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest_rows.append(
            {
                "path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            }
        )
        _write_json(
            raw / "evidence_manifest.json",
            {
                "stage": STAGE,
                "experiment_id": experiment_id,
                "run_id": campaign_run_id,
                "compatibility_manifest_sha256": snapshot.compatibility_manifest_sha256,
                "files": manifest_rows,
            },
        )
        statuses[experiment_id] = verdict["status"]
        outputs.append(str(experiment_root.relative_to(repo)))

    summary = {
        "stage": STAGE,
        "run_id": campaign_run_id,
        "collector_status": "PASS",
        "stage_status": STATUS_BLOCKED,
        "stage_complete": False,
        "ascend_ready": snapshot.ascend_ready,
        "blocker_codes": list(snapshot.blocker_codes),
        "experiment_statuses": statuses,
        "experiment_roots": outputs,
        "frontend_contract": {
            "catalog": "GET /api/console/v1/evidence",
            "detail": "GET /api/console/v1/evidence/{evidence_id}",
            "download": "GET /api/console/v1/evidence/{evidence_id}/download",
            "index_pattern": "docs/stage_experiments/*/*/raw/verdict.json",
        },
        "generated_at": utc_now(),
    }
    _write_json(allowed / "campaign_summary.json", summary)
    stage_lines = [
        "# S09 阶段实验执行摘要",
        "",
        f"> Run ID：`{campaign_run_id}`  ",
        "> 阶段裁决：**BLOCKED**  ",
        "> 采集器：**PASS**（仅表示证据完整写盘，不等于科学实验通过）",
        "",
        "## 1. 能力门禁",
        "",
        f"- Ascend ready：`{str(snapshot.ascend_ready).lower()}`",
        f"- 阻塞原因：`{', '.join(snapshot.blocker_codes)}`",
        f"- Compatibility manifest：`{snapshot.compatibility_manifest_sha256}`",
        "- CPU smoke：`claim_allowed=false`，未计入实验样本。",
        "",
        "## 2. 实验裁决",
        "",
        "| 实验 | 级别 | 状态 | 预计效果是否达到 | 单项标准是否满足 |",
        "|---|---|---|---:|---:|",
    ]
    for experiment_id in selected:
        protocol = PROTOCOLS[experiment_id]
        status = statuses[experiment_id]
        stage_lines.append(
            f"| {experiment_id} | {protocol.level} | {status} | 否 | 否 |"
        )
    stage_lines.extend(
        [
            "",
            "## 3. 结论边界",
            "",
            "当前 Jetson 目标不能执行 Ascend/CANN 实验。E09-01～05、E09-07～10 不得标为 PASS；E09-06 依据 `configs/ascend/scope_adr.yaml` 为 `N/A_BY_ADR`，且不作任何 Ascend 低比特声明。",
            "",
            "每项实验的 `raw/required_evidence.json`、`raw/criteria_evaluation.json`、28 行 `raw/step_status.jsonl` 和实验报告明确记录已采/未采内容。前端由 `raw/verdict.json` 自动发现。",
            "",
        ]
    )
    _atomic_text(allowed / "S09_阶段执行摘要.md", "\n".join(stage_lines))
    return summary


def protocol_catalog() -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "experiments": [PROTOCOLS[experiment_id].as_dict() for experiment_id in EXPERIMENTS],
        "count": len(EXPERIMENTS),
    }


__all__ = [
    "CONCLUSION_STATUSES",
    "EXPERIMENTS",
    "ExperimentProtocol",
    "PROTOCOLS",
    "PROTOCOL_STATUSES",
    "STAGE",
    "STATUS_BLOCKED",
    "STATUS_FAIL",
    "STATUS_N_A_BY_ADR",
    "STATUS_PASS",
    "STATUS_PASS_NEGATIVE",
    "CapabilitySnapshot",
    "collect_campaign",
    "collect_capability_snapshot",
    "protocol_catalog",
    "protocol_steps",
    "source_tree_hash",
    "smoke_self_check",
]
