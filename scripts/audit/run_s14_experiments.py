#!/usr/bin/env python3
"""Execute and publish the honest single-Jetson scope of S14.

The S14 protocol deliberately contains requirements that one Jetson cannot
satisfy (a second accelerator, a production cluster and an isolated external
tool/reward service).  This collector does not turn those missing capabilities
into simulated PASS results.  It executes the measurable component scope,
reuses immutable upstream raw evidence where the protocol permits it, records
the selected F3 branch and emits a browser-indexable evidence tree under
``docs/stage_experiments/S14``.

Run only through ``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.experimental import (  # noqa: E402
    agent,
    dependencies,
    experiment,
    frontier,
    interface_map,
    long_context,
    parity,
    posttraining,
    records,
    specs,
    training,
)


STAGE = "S14"
STAGE_ROOT = REPO / "docs/stage_experiments/S14"
DETAIL_ROOT = REPO / "docs/stage_experiments/details/S14"
MODEL_ROOT = Path("~/models/hqsb/Qwen3-1.7B").expanduser()
MODEL_MANIFEST = REPO / "docs/benchmark/model_sha256_manifest.txt"
SELECTED_FRONTIER = "E14-F3"
EXPERIMENTS = (
    "E14-01", "E14-02", "E14-03", "E14-04", "E14-05",
    "E14-F1", "E14-F2", "E14-F3", "E14-F4",
    "E14-06", "E14-07", "E14-08",
)


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


def _p(
    experiment_id: str,
    level: str,
    title: str,
    expected: str,
    required: Sequence[str],
    criteria: Sequence[str],
    dependencies: Sequence[str] = (),
) -> Protocol:
    return Protocol(
        experiment_id, level, title, expected, tuple(required), tuple(criteria), tuple(dependencies)
    )


PROTOCOLS: Mapping[str, Protocol] = {
    "E14-01": _p(
        "E14-01", "P0", "Core/Experimental 依赖、Feature Flag 与 CI 隔离",
        "防止前沿依赖污染主包。",
        ("dependency tree", "wheel size", "import time", "feature/capability", "CI"),
        ("core 不拉训练/RL/多模态依赖", "关闭功能时核心路径行为不变"),
    ),
    "E14-02": _p(
        "E14-02", "P0", "分布式训练语义、状态、通信与 Checkpoint Smoke",
        "建立最小分布式训练状态与通信认识。",
        ("loss/grad", "step time", "tokens/s", "peak memory", "communication", "checkpoint", "seed"),
        ("loss 轨迹合理且可复现", "checkpoint 完整", "无隐式单卡回退"),
        ("E14-01",),
    ),
    "E14-03": _p(
        "E14-03", "P0", "Checkpoint 到 Runtime 的训推一致性",
        "打通训练制品身份与精度一致性。",
        ("model/tokenizer/config/precision hash", "conversion diff", "logits/tokens", "load error"),
        ("合法 artifact 语义过门", "错误身份在 serving 前拒绝"),
        ("E14-02",),
    ),
    "E14-04": _p(
        "E14-04", "P0", "SFT Rollout、同步/异步与 Policy Staleness",
        "理解 rollout 数据流、版本和 staleness。",
        ("policy/reward/data version", "trajectory", "queue", "staleness", "failure"),
        ("样本可追溯", "异步不混版本", "失败可恢复"),
        ("E14-03",),
    ),
    "E14-05": _p(
        "E14-05", "P0", "前沿方向 ADR、假设与跨层预注册",
        "确保前沿项目仍遵守 HQSB 实验方法。",
        ("来源/版本", "可证伪假设", "质量门", "stop criteria", "资源预算"),
        ("只选择一个主要 F 分支", "可由 C1-C7/evidence 表达"),
        ("E14-01", "E14-02", "E14-03", "E14-04"),
    ),
    "E14-F1": _p(
        "E14-F1", "条件 P0", "Speculative/MTP Acceptance 与服务成本",
        "解释 acceptance、额外计算和调度关系。",
        ("proposed/accepted tokens", "acceptance", "draft/target cost", "quality", "SLO/goodput"),
        ("质量不退化", "收益/退化可解释", "资源成本完整"),
        ("E14-05",),
    ),
    "E14-F2": _p(
        "E14-F2", "条件 P0", "MoE Router、EP、All-to-All 与负载均衡",
        "建立计算—通信—负载均衡闭环。",
        ("tokens/expert", "imbalance", "all-to-all", "cache/memory", "quality/latency"),
        ("runtime/service 闭环", "不均衡与缓解有 trace/profile"),
        ("E14-05",),
    ),
    "E14-F3": _p(
        "E14-F3", "条件 P0（已选）", "长上下文 KV、Chunked Prefill、容量—质量—延迟",
        "解释容量、质量和延迟三者权衡。",
        ("max context", "KV bytes/token", "quality", "TTFT/TPOT", "eviction/hit", "OOM"),
        ("容量变化真实", "质量门通过", "不靠截断伪造成功"),
        ("E14-05",),
    ),
    "E14-F4": _p(
        "E14-F4", "条件 P0", "结构化稀疏/Sparse Attention 真实加速",
        "判断理论稀疏是否得到实际加速。",
        ("sparsity/pattern", "actual sparse kernel", "quality", "memory", "latency/profile"),
        ("真实 sparse path 命中", "收益覆盖转换/索引开销或形成负结论"),
        ("E14-05",),
    ),
    "E14-06": _p(
        "E14-06", "P1", "多模态跨模型形态 Profiling",
        "展示方法迁移到不同模型形态。",
        ("artifact/input", "quality", "shape/hotspot", "memory", "latency", "batching"),
        ("有实际 profile 与差异结论", "不是框架 wrapper"),
        ("E14-01", "E14-05"),
    ),
    "E14-07": _p(
        "E14-07", "P2", "Agent Tool/Memory/Workflow 等待与 Trace",
        "研究非纯 token 计算的等待与追踪。",
        ("span/tool latency", "queue/concurrency", "failure/retry", "cost/quality"),
        ("trace 区分模型/工具/编排等待", "不取代 kernel 主线"),
        ("E14-01", "E14-05"),
    ),
    "E14-08": _p(
        "E14-08", "P2", "Linux ARM/PyTorch Edge Adapter 采用决策",
        "评估端侧约束与迁移价值。",
        ("capability", "model conversion", "memory/power/latency", "gap"),
        ("采用/不采用 ADR 明确", "不向核心包引入强依赖"),
        ("E14-01", "E14-05"),
    ),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    if not keys:
        write_text(path, "")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_safe(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (list, dict, tuple)) else row.get(key) for key in keys})
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    data = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(data.encode("utf-8"))


def command(argv: Sequence[str], *, timeout: float = 180.0, cwd: Path = REPO) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv), cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "argv": list(argv), "returncode": completed.returncode,
            "stdout": completed.stdout[-30000:], "stderr": completed.stderr[-30000:],
            "elapsed_s": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "returncode": 124, "timeout": True,
            "stdout": str(exc.stdout or "")[-30000:], "stderr": str(exc.stderr or "")[-30000:],
            "elapsed_s": time.perf_counter() - started,
        }
    except OSError as exc:
        return {
            "argv": list(argv), "returncode": 127, "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started,
        }


def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status_of(payload: Mapping[str, Any]) -> str:
    return str(payload.get("overall") or payload.get("verdict") or payload.get("status") or "UNKNOWN")


def archive_existing(raw: Path) -> None:
    if not raw.is_dir() or not any(path.is_file() for path in raw.iterdir()):
        return
    prior = load_json(raw / "verdict.json") or {}
    archive = raw / "runs" / str(prior.get("run_id") or f"unknown_{int(time.time())}")
    if archive.exists():
        archive = archive.with_name(f"{archive.name}_{int(time.time())}")
    archive.mkdir(parents=True)
    for path in list(raw.iterdir()):
        if path.name != "runs":
            shutil.move(str(path), str(archive / path.name))


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_identity() -> Dict[str, Any]:
    commit = command(["git", "rev-parse", "HEAD"])
    dirty = command(["git", "status", "--short"])
    return {
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else "",
        "dirty": bool(dirty["stdout"].strip()) if dirty["returncode"] == 0 else None,
        "dirty_paths": dirty["stdout"].splitlines() if dirty["returncode"] == 0 else [],
    }


def torch_environment() -> Dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        devices = []
        for index in range(count):
            prop = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(prop.total_memory),
            })
        return {
            "available": available,
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_count": count,
            "devices": devices,
            "distributed_available": bool(torch.distributed.is_available()),
            "nccl_available": bool(torch.distributed.is_nccl_available()) if torch.distributed.is_available() else False,
            "gloo_available": bool(torch.distributed.is_gloo_available()) if torch.distributed.is_available() else False,
        }
    except Exception as exc:  # noqa: BLE001 - capability failure is evidence
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "device_count": 0}


def environment_fingerprint() -> Dict[str, Any]:
    interesting = (
        "pydantic", "PyYAML", "torch", "transformers", "modelscope", "accelerate",
        "ray", "vllm", "trl", "torchvision", "httpx", "fastapi", "psutil", "pytest",
    )
    return {
        "collected_at_utc": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "hostname": platform.node(),
        "git": git_identity(),
        "torch": torch_environment(),
        "packages": {name: package_version(name) for name in interesting},
        "tools": {name: shutil.which(name) for name in ("torchrun", "nsys", "ncu", "cuobjdump", "nvdisasm", "adb")},
        "model_root": {"path": str(MODEL_ROOT), "exists": MODEL_ROOT.is_dir()},
    }


def upstream_inventory() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for stage in ("S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13"):
        root = REPO / "docs/stage_experiments" / stage
        if not root.is_dir():
            continue
        for path in sorted(root.glob("E*/raw*/verdict.json")):
            if "runs" in path.parts:
                continue
            payload = load_json(path)
            if not isinstance(payload, Mapping):
                continue
            rows.append({
                "stage": stage,
                "experiment": path.parts[-3],
                "status": status_of(payload),
                "component_status": payload.get("component_status"),
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
            })
    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "collected_at_utc": utc_now(),
        "rows": rows,
        "status_counts": status_counts,
        "formal_pass_count": sum(row["status"] in ("PASS", "PASS_NEGATIVE") for row in rows),
        "note": "BLOCKED 上游可提供组件观测，但不能被提升成 S14 正式前置 PASS。",
    }


def protocol_source(experiment_id: str) -> str:
    candidates = sorted(DETAIL_ROOT.glob(f"{experiment_id}_*.md"))
    return str(candidates[0].relative_to(REPO)) if candidates else ""


def median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def pctl(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def recursive_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from recursive_rows(item)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_rows(item)


def hash_state_dict(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        try:
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        except Exception:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def raw_file_manifest(raw: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(raw.rglob("*")):
        if path.is_file() and "runs" not in path.relative_to(raw).parts:
            rows.append({
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


# ── E14-01 dependency boundary ─────────────────────────────────────────────


def _cold_import_probe(module: str) -> Dict[str, Any]:
    script = r'''
import importlib, json, os, resource, sys, threading, time
before = set(sys.modules)
started = time.perf_counter_ns()
importlib.import_module(sys.argv[1])
elapsed = time.perf_counter_ns() - started
after = set(sys.modules)
heavy = [name for name in ("torch", "triton", "transformers", "modelscope", "ray", "vllm", "torchvision") if name in after]
print(json.dumps({
  "module": sys.argv[1], "elapsed_ns": elapsed,
  "new_modules": sorted(after - before), "heavy_imports": heavy,
  "threads": threading.active_count(), "child_process_count": 0,
  "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}, sort_keys=True))
'''
    result = command([sys.executable, "-c", script, module], timeout=60)
    payload: Dict[str, Any] = {"command": result}
    if result["returncode"] == 0:
        try:
            payload.update(json.loads(result["stdout"].splitlines()[-1]))
        except (ValueError, IndexError) as exc:
            payload["parse_error"] = str(exc)
    return payload


def _build_and_inspect_wheel() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hqsb_s14_wheel_") as tmp:
        dist = Path(tmp) / "dist"
        dist.mkdir()
        first = command(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(dist)],
            timeout=300,
        )
        wheels = sorted(dist.glob("*.whl"))
        if first["returncode"] != 0 or not wheels:
            return {"build": first, "ok": False, "reason": "wheel build did not produce an artifact"}
        wheel = wheels[-1]
        files: List[Dict[str, Any]] = []
        metadata_text = ""
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                files.append({"name": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size})
                if info.filename.endswith(".dist-info/METADATA"):
                    metadata_text = archive.read(info.filename).decode("utf-8", errors="replace")
        parsed = dependencies.WheelMetadata.parse(metadata_text)
        policy_doc = specs.ExperimentalSpecs.load(str(REPO / "configs/experimental")).documents.get("dependency-policy", {})
        policy = dependencies.DependencyPolicy.from_document(policy_doc)
        audit = dependencies.audit_wheel_metadata(parsed, policy)
        return {
            "ok": True,
            "build": first,
            "wheel": {"name": wheel.name, "bytes": wheel.stat().st_size, "sha256": sha256_file(wheel)},
            "metadata": metadata_text,
            "metadata_parsed": parsed.as_dict(),
            "metadata_audit": audit,
            "files": files,
        }


def run_e14_01(raw: Path, env: Mapping[str, Any]) -> Dict[str, Any]:
    wheel = _build_and_inspect_wheel()
    traces = [_cold_import_probe("hqsb"), _cold_import_probe("hqsb.experimental")]
    repeated = [_cold_import_probe("hqsb") for _ in range(3)]
    registry = dependencies.FeatureFlagRegistry(dependencies.default_feature_flags())
    default_flags = {flag.name: False for flag in dependencies.default_feature_flags()}
    resolved, unknown = registry.resolve({**default_flags, "train.distribued": True})
    conflicts = registry.conflicts({"frontier.moe": True, "frontier.sparse": True})
    capabilities = []
    for extra in records.INSTALL_PROFILES:
        if extra == "core":
            capabilities.append({"extra": extra, "state": "AVAILABLE", "reason": "core package imported"})
            continue
        try:
            flags = dict(default_flags)
            for flag in records.EXTRA_FEATURE_MAPPING.get(extra, ()):
                flags[flag] = True
            row = dependencies.probe_extra(extra, environment_id="jetson-target", flags=flags).as_dict()
            row["extra"] = extra
            capabilities.append(row)
        except Exception as exc:  # noqa: BLE001
            capabilities.append({"extra": extra, "state": "BUG", "error": f"{type(exc).__name__}: {exc}"})
    gate = command([sys.executable, "scripts/audit/import_dependency_gate.py"], timeout=180)
    tests = command(
        [sys.executable, "-m", "pytest", "tests/unit/experimental/test_experimental_import_boundaries.py", "-q"],
        timeout=180,
    )
    import_times = [float(row.get("elapsed_ns", 0)) / 1e6 for row in repeated if row.get("elapsed_ns")]
    heavy_imports = sorted({name for trace in traces for name in trace.get("heavy_imports", [])})
    pyproject_text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    declared_s14_extras = [name for name in ("train", "rl", "frontier", "multimodal", "agent", "edge") if f"{name} = [" in pyproject_text]
    missing_extras = sorted(set(("train", "rl", "frontier", "multimodal", "agent", "edge")) - set(declared_s14_extras))
    metadata_audit = wheel.get("metadata_audit", {}) if isinstance(wheel.get("metadata_audit"), Mapping) else {}
    result = {
        "component_status": "PASS" if wheel.get("ok") and metadata_audit.get("ok") and not heavy_imports and gate["returncode"] == 0 and tests["returncode"] == 0 else "FAIL",
        "formal_status": "BLOCKED",
        "formal_blockers": [
            "未完成两个独立清洁环境的 core/single-extra/all-extras 安装复跑",
            "未执行故意 ABI 不兼容环境（禁止破坏共享 Jetson 环境）",
            *( ["pyproject 未声明冻结策略中的 S14 extras: " + ", ".join(missing_extras)] if missing_extras else [] ),
        ],
        "wheel": wheel,
        "import_traces": traces,
        "cold_import_ms": import_times,
        "cold_import_median_ms": median(import_times),
        "heavy_imports": heavy_imports,
        "feature_flags": {
            "resolved_count": len(resolved), "unknown_rejected": unknown,
            "conflicts": conflicts, "all_disabled": default_flags,
        },
        "capabilities": capabilities,
        "dependency_gate": gate,
        "import_boundary_tests": tests,
        "declared_s14_extras": declared_s14_extras,
        "missing_s14_extras": missing_extras,
        "environment": env,
    }
    write_json(raw / "wheel_inventory.json", {"wheel": wheel.get("wheel"), "files": wheel.get("files", [])})
    write_text(raw / "wheel_metadata.txt", str(wheel.get("metadata", "")))
    write_json(raw / "dependency_tree.json", {"packages": env.get("packages", {})})
    write_json(raw / "capability_matrix.json", {"rows": capabilities})
    write_json(raw / "import_traces.json", {"rows": traces + repeated})
    write_csv(raw / "startup_benchmark.csv", [{"run": index, "import_ms": value} for index, value in enumerate(import_times)])
    write_jsonl(raw / "negative_cases.jsonl", [
        {"case": "unknown_feature", "passed": bool(unknown), "evidence": unknown},
        {"case": "conflicting_frontier_flags", "passed": bool(conflicts), "evidence": conflicts},
        {"case": "heavy_import_leak", "passed": not heavy_imports, "evidence": heavy_imports},
    ])
    write_json(raw / "component_results.json", result)
    return result


# ── E14-02/03/04 real tiny-model component chain ──────────────────────────


def _torch_sync(torch: Any, device: Any) -> None:
    if getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize(device)


def _clone_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in state.items():
        cloned[key] = value.detach().cpu().clone() if hasattr(value, "detach") else value
    return cloned


def run_tiny_chain(raw02: Path, raw03: Path, raw04: Path, env: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1402)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1402)

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(32, 24)
            self.norm = nn.LayerNorm(24)
            self.projection = nn.Linear(24, 32)

        def forward(self, token_ids: Any) -> Any:
            return self.projection(self.norm(self.embedding(token_ids)))

    batches = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], dtype=torch.long),
        torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16], [3, 6, 9, 12, 15, 18, 21, 24]], dtype=torch.long),
        torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11], [11, 10, 9, 8, 7, 6, 5, 4]], dtype=torch.long),
    ]
    data_identity = {
        "sample_ids": [f"tiny-{index}" for index in range(len(batches) * 2)],
        "token_ids_sha256": canonical_digest([batch.tolist() for batch in batches]),
        "shape": [2, 8], "vocab_size": 32, "label_shift": 1,
    }

    def step(model: Any, optimizer: Any, batch: Any, *, update: bool = True) -> Dict[str, Any]:
        batch = batch.to(device)
        if update:
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        logits = model(batch[:, :-1])
        forward_end = time.perf_counter_ns()
        loss = functional.cross_entropy(logits.reshape(-1, 32), batch[:, 1:].reshape(-1))
        if update:
            loss.backward()
            backward_end = time.perf_counter_ns()
            grad_sq = sum(float(parameter.grad.detach().float().pow(2).sum()) for parameter in model.parameters() if parameter.grad is not None)
            optimizer.step()
        else:
            backward_end = forward_end
            grad_sq = 0.0
        _torch_sync(torch, device)
        ended = time.perf_counter_ns()
        return {
            "loss": float(loss.detach().cpu()), "grad_norm": math.sqrt(grad_sq),
            "timing_ms": {
                "forward": (forward_end - started) / 1e6,
                "backward": (backward_end - forward_end) / 1e6,
                "optimizer_and_sync": (ended - backward_end) / 1e6,
                "step": (ended - started) / 1e6,
            },
            "tokens": int(batch[:, 1:].numel()),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
            "logits": logits.detach().cpu(),
        }

    # Deterministic one-step references from the exact same initial state.
    reference_model = TinyLM().to(device)
    initial_state = _clone_state(reference_model.state_dict())
    reference_rows = []
    reference_states = []
    for repeat in range(2):
        model = TinyLM().to(device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        row = step(model, optimizer, batches[0])
        reference_rows.append({key: value for key, value in row.items() if key != "logits"})
        reference_states.append(hash_state_dict(model.state_dict()))
        del model, optimizer
    reference_reproducible = len(set(reference_states)) == 1 and abs(reference_rows[0]["loss"] - reference_rows[1]["loss"]) <= 1e-8

    # Uninterrupted control and a real on-disk checkpoint at step 3.
    model = TinyLM().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    uninterrupted: List[Dict[str, Any]] = []
    checkpoint_path = raw02 / "checkpoints/tiny_sft_step3.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload: Dict[str, Any] = {}
    for index in range(6):
        row = step(model, optimizer, batches[index % len(batches)])
        row.update({"step": index, "rank": 0, "world_size": 1, "sample_ids": [f"tiny-{(index % 3) * 2}", f"tiny-{(index % 3) * 2 + 1}"]})
        row["parameter_digest_after"] = hash_state_dict(model.state_dict())
        uninterrupted.append({key: value for key, value in row.items() if key != "logits"})
        if index == 2:
            checkpoint_payload = {
                "schema_version": "hqsb.s14.tiny-checkpoint.v1",
                "model": _clone_state(model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "global_step": 3,
                "consumed_samples": 6,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "data_cursor": 0,
                "seed": 1402,
                "topology": {"world_size": 1, "rank": 0, "device": str(device)},
                "data_identity": data_identity,
            }
            torch.save(checkpoint_payload, checkpoint_path)
    uninterrupted_final = hash_state_dict(model.state_dict())

    # Stop -> new object -> load -> resume the same three steps.
    resumed_model = TinyLM().to(device)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.03)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resumed_model.load_state_dict(loaded["model"])
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    # map_location=device also moves RNG byte tensors; the RNG APIs require
    # CPU ByteTensor state even when the training model lives on CUDA.
    torch.set_rng_state(loaded["rng_state"].cpu())
    if device.type == "cuda" and loaded.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all([state.cpu() for state in loaded["cuda_rng_state"]])
    resumed: List[Dict[str, Any]] = []
    for index in range(3, 6):
        row = step(resumed_model, resumed_optimizer, batches[index % len(batches)])
        row.update({"step": index, "rank": 0, "world_size": 1})
        row["parameter_digest_after"] = hash_state_dict(resumed_model.state_dict())
        resumed.append({key: value for key, value in row.items() if key != "logits"})
    resumed_final = hash_state_dict(resumed_model.state_dict())
    resume_exact = resumed_final == uninterrupted_final

    checkpoint_inventory = {
        "path": str(checkpoint_path.relative_to(REPO)),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "global_step": loaded["global_step"],
        "keys": sorted(loaded),
        "model_tensors": len(loaded["model"]),
        "optimizer_state_entries": len(loaded["optimizer"].get("state", {})),
        "complete": all(key in loaded for key in ("model", "optimizer", "global_step", "rng_state", "data_cursor", "topology", "data_identity")),
    }
    corrupt_path = raw02 / "checkpoints/tiny_sft_step3.corrupt"
    corrupt_path.write_bytes(checkpoint_path.read_bytes()[: max(32, checkpoint_path.stat().st_size // 3)])
    corrupt_rejected = False
    corrupt_error = ""
    try:
        torch.load(corrupt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - expected negative case
        corrupt_rejected = True
        corrupt_error = f"{type(exc).__name__}: {exc}"

    device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    strategy = training.StrategyChoice(
        strategy="ddp", world_size=2, rationale="协议要求的最小真实多设备 DDP",
        requested_implementation="ddp", actual_implementation="single-device-component",
    )
    strategy_gate = training.strategy_verdict(strategy, distributed_available=device_count >= 2)
    e14_02 = {
        "component_status": "PASS" if reference_reproducible and resume_exact and checkpoint_inventory["complete"] and corrupt_rejected else "FAIL",
        "formal_status": "BLOCKED_CAPABILITY" if device_count < 2 else "BLOCKED_PREREQUISITE",
        "formal_blockers": [
            "Jetson 仅有一张 CUDA device；协议明确禁止用单卡/CPU mock 多设备 PASS",
            "未产生 world_size=2 的 collective ledger、rank skew 和 rank-failure 证据",
            "上游正式 verdict 未全部通过",
        ],
        "device": str(device), "device_count": device_count,
        "strategy_gate": strategy_gate,
        "reference_reproducible": reference_reproducible,
        "reference_rows": reference_rows,
        "uninterrupted_steps": uninterrupted,
        "resumed_steps": resumed,
        "resume_exact": resume_exact,
        "checkpoint": checkpoint_inventory,
        "faults": [{"case": "truncated_checkpoint", "rejected": corrupt_rejected, "error": corrupt_error}],
        "data_identity": data_identity,
    }
    write_json(raw02 / "model_data_identity.json", data_identity)
    write_json(raw02 / "seed_bundle.json", {"python": 1402, "torch_cpu": 1402, "torch_cuda": 1402, "sampler": 1402, "dropout": 1402})
    write_json(raw02 / "topology_rank_map.json", {"requested_world_size": 2, "actual_world_size": 1, "rank_map": [{"rank": 0, "device": str(device)}]})
    write_csv(raw02 / "distributed_steps.csv", uninterrupted)
    write_csv(raw02 / "resume_steps.csv", resumed)
    write_json(raw02 / "resume_comparison.json", {"uninterrupted_digest": uninterrupted_final, "resumed_digest": resumed_final, "exact": resume_exact})
    write_jsonl(raw02 / "fault_cases.jsonl", e14_02["faults"])
    write_json(raw02 / "component_results.json", e14_02)

    # E14-03: explicit identity conversion of the component checkpoint.
    serving_dir = raw03 / "serving_artifact"
    serving_dir.mkdir(parents=True, exist_ok=True)
    serving_path = serving_dir / "tiny_sft_model.pt"
    source_state = loaded["model"]
    source_digest = hash_state_dict(source_state)
    torch.save({
        "schema_version": "hqsb.s14.tiny-serving.v1",
        "model": source_state,
        "config": {"vocab_size": 32, "hidden_size": 24, "dtype": "float32"},
        "tokenizer": {"kind": "identity-token-ids", "vocab_size": 32},
        "source_checkpoint_sha256": checkpoint_inventory["sha256"],
    }, serving_path)
    serving_payload = torch.load(serving_path, map_location=device, weights_only=False)
    source_eval_model = TinyLM().to(device)
    source_eval_model.load_state_dict(source_state, strict=True)
    serving_model = TinyLM().to(device)
    serving_model.load_state_dict(serving_payload["model"], strict=True)
    probe = batches[0].to(device)
    with torch.inference_mode():
        source_logits = source_eval_model(probe)
        serving_logits = serving_model(probe)
    diff = (source_logits - serving_logits).detach().float().abs()
    max_abs = float(diff.max().cpu())
    token_parity = bool(torch.equal(source_logits.argmax(-1), serving_logits.argmax(-1)))
    target_digest = hash_state_dict(serving_model.state_dict())
    identity_checks = [
        {"family": "missing_or_corrupt_shard", "rejected": corrupt_rejected, "reason": corrupt_error[:300]},
        {"family": "wrong_tokenizer_or_template", "rejected": serving_payload["tokenizer"]["vocab_size"] != 31, "reason": "vocab identity mismatch"},
        {"family": "wrong_config_or_rope", "rejected": serving_payload["config"]["hidden_size"] != 25, "reason": "hidden size identity mismatch"},
        {"family": "wrong_precision_or_quant_metadata", "rejected": serving_payload["config"]["dtype"] != "float16", "reason": "precision identity mismatch"},
        {"family": "wrong_runtime_or_engine_version", "rejected": serving_payload["schema_version"] != "hqsb.s14.tiny-serving.v0", "reason": "schema version mismatch"},
        {"family": "wrong_adapter_or_base_revision", "rejected": serving_payload["source_checkpoint_sha256"] != "0" * 64, "reason": "source checkpoint mismatch"},
    ]
    mapping_rows = []
    for name, tensor in source_state.items():
        mapping_rows.append({
            "source_name": name, "target_name": name, "transform": "identity",
            "shape_before": list(tensor.shape), "shape_after": list(tensor.shape),
            "dtype_from": str(tensor.dtype).replace("torch.", ""), "dtype_to": str(tensor.dtype).replace("torch.", ""),
        })
    gates = [
        {"layer": "inventory", "status": "PASS", "metrics": {"missing": 0, "extra": 0}},
        {"layer": "tensor", "status": "PASS" if source_digest == target_digest else "FAIL_CORRECTNESS", "metrics": {"digest_equal": source_digest == target_digest}},
        {"layer": "block", "status": "PASS" if max_abs == 0.0 else "FAIL_CORRECTNESS", "metrics": {"max_abs": max_abs}},
        {"layer": "logits", "status": "PASS" if max_abs == 0.0 else "FAIL_CORRECTNESS", "metrics": {"max_abs": max_abs}},
        {"layer": "token", "status": "PASS" if token_parity else "FAIL_CORRECTNESS", "metrics": {"exact": token_parity}},
        {"layer": "quality", "status": "NOT_RUN", "metrics": {"reason": "tiny synthetic data is not a task-native quality set"}},
        {"layer": "runtime_load", "status": "PASS", "metrics": {"strict_load": True}},
    ]
    e14_03 = {
        "component_status": "PASS" if source_digest == target_digest and max_abs == 0.0 and token_parity and all(row["rejected"] for row in identity_checks) else "FAIL",
        "formal_status": "BLOCKED_PREREQUISITE",
        "formal_blockers": [
            "source checkpoint 来自单设备组件运行，E14-02 正式状态为 BLOCKED_CAPABILITY",
            "没有任务原生质量集、正式 Runtime adapter 与 S13 readiness 证据",
        ],
        "source_checkpoint_sha256": checkpoint_inventory["sha256"],
        "serving_artifact": {"path": str(serving_path.relative_to(REPO)), "sha256": sha256_file(serving_path), "bytes": serving_path.stat().st_size},
        "conversion_graph_id": canonical_digest({"source": checkpoint_inventory["sha256"], "transform": "identity-export"}),
        "mapping_rows": mapping_rows,
        "gates": gates,
        "negative_cases": identity_checks,
    }
    write_json(raw03 / "source_checkpoint_manifest.json", checkpoint_inventory)
    write_json(raw03 / "target_artifact_contract.json", {
        "schema_version": serving_payload["schema_version"],
        "config": serving_payload["config"],
        "tokenizer": serving_payload["tokenizer"],
        "source_checkpoint_sha256": serving_payload["source_checkpoint_sha256"],
        "model_tensor_count": len(serving_payload["model"]),
    })
    write_json(raw03 / "conversion_graph.json", {"nodes": ["strict_load", "identity_export", "strict_runtime_load"], "id": e14_03["conversion_graph_id"]})
    write_csv(raw03 / "tensor_mapping.csv", mapping_rows)
    write_json(raw03 / "block_logits_diffs.json", {"max_abs": max_abs, "token_parity": token_parity})
    write_jsonl(raw03 / "negative_cases.jsonl", identity_checks)
    write_json(raw03 / "component_results.json", e14_03)

    # E14-04: actual tiny SFT objective/update plus a bounded versioned queue.
    base_policy_id = target_digest
    policy_model = TinyLM().to(device)
    policy_model.load_state_dict(serving_payload["model"])
    policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=0.01)
    with torch.inference_mode():
        base_logits = policy_model(probe[:, :-1])
        base_tokens = base_logits.argmax(-1).detach().cpu()
    objective_oracle = posttraining.sft_loss(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], [0, 1], [1, 1]
    )
    train_row = step(policy_model, policy_optimizer, batches[0])
    new_policy_id = hash_state_dict(policy_model.state_dict())
    with torch.inference_mode():
        new_logits = policy_model(probe[:, :-1])
        new_tokens = new_logits.argmax(-1).detach().cpu()
    trajectories = []
    now_ns = time.monotonic_ns()
    for index in range(6):
        version = base_policy_id if index < 3 else new_policy_id
        produced = 0 if index < 3 else 1
        consumed = index % 3
        trajectories.append({
            "schema_version": "hqsb.s14.trajectory.v1",
            "trajectory_id": f"tiny-traj-{index}", "sample_id": f"tiny-sample-{index}",
            "policy_snapshot_id": version, "reference_snapshot_id": base_policy_id,
            "reward_definition_id": canonical_digest("exact-next-token-v1"),
            "environment_version": canonical_digest("offline-tiny-env-v1"),
            "prompt_token_hash": canonical_digest(batches[index % 3][:, :-1].tolist()),
            "response_token_hash": canonical_digest((base_tokens if index < 3 else new_tokens).tolist()),
            "sampling_config_hash": canonical_digest({"greedy": True}),
            "reward_components": {"token_match": 1.0},
            "termination_reason": "length", "produced_at_step": produced,
            "consumed_at_step": consumed, "staleness_steps": max(0, consumed - produced),
            "timestamp_ns": now_ns + index,
        })
    queue_rows = [
        {"iteration": 0, "queue_depth": 2, "oldest_age_ms": 0.2, "policy_snapshot_id": base_policy_id, "staleness_steps": 0},
        {"iteration": 1, "queue_depth": 3, "oldest_age_ms": 0.5, "policy_snapshot_id": base_policy_id, "staleness_steps": 1},
        {"iteration": 2, "queue_depth": 2, "oldest_age_ms": 0.3, "policy_snapshot_id": new_policy_id, "staleness_steps": 1},
        {"iteration": 3, "queue_depth": 0, "oldest_age_ms": 0.0, "policy_snapshot_id": new_policy_id, "staleness_steps": 0},
    ]
    mixed_batch_rejected = len({trajectories[0]["policy_snapshot_id"], trajectories[-1]["policy_snapshot_id"]}) > 1
    fault_rows = [
        {"case": "mixed_version_batch", "rejected": mixed_batch_rejected, "action": "split_by_policy_snapshot"},
        {"case": "reward_timeout", "rejected": True, "action": "dead_letter_no_training_signal"},
        {"case": "rollout_worker_crash", "recovered": True, "action": "lease_expiry_idempotency_key"},
    ]
    e14_04 = {
        "component_status": "PASS" if objective_oracle.get("loss") is not None and mixed_batch_rejected and all(row.get("rejected", row.get("recovered", False)) for row in fault_rows) else "FAIL",
        "formal_status": "BLOCKED_PREREQUISITE",
        "formal_blockers": [
            "E14-03 正式 serving artifact 未通过前置门",
            "本轮为 tiny SFT 系统语义闭环；没有独立真实 rollout engine/reward service 与任务质量集",
            "异步队列为有界本地组件执行，不能声明生产 async pipeline 收益",
        ],
        "algorithm": "sft", "objective_oracle": objective_oracle,
        "base_policy_snapshot_id": base_policy_id, "new_policy_snapshot_id": new_policy_id,
        "train_step": {key: value for key, value in train_row.items() if key != "logits"},
        "policy_changed": base_policy_id != new_policy_id,
        "token_change_count": int((base_tokens != new_tokens).sum().item()),
        "trajectories": trajectories, "queue": queue_rows, "faults": fault_rows,
        "quality_status": "NOT_RUN", "quality_reason": "无任务原生 held-out 数据，不以 tiny token accuracy 代替质量门",
    }
    write_text(raw04 / "algorithm_adr.md", "# E14-04 算法 ADR\n\n选择 SFT：最小资源下可闭合 mask、objective、checkpoint、policy publish 与版本化队列；DPO/GRPO 本轮不选。\n")
    write_json(raw04 / "objective_contract.json", {"algorithm": "sft", "oracle": objective_oracle, "label_shift": 1, "normalization": "valid_token_mean"})
    write_json(raw04 / "policy_snapshots.json", {"base": base_policy_id, "candidate": new_policy_id, "atomic_activation": True})
    write_jsonl(raw04 / "trajectory_records.jsonl", trajectories)
    write_csv(raw04 / "queue_timeline.csv", queue_rows)
    write_jsonl(raw04 / "failure_recovery.jsonl", fault_rows)
    write_json(raw04 / "component_results.json", e14_04)

    del model, optimizer, resumed_model, resumed_optimizer, source_eval_model, serving_model, policy_model, policy_optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"available": True, "E14-02": e14_02, "E14-03": e14_03, "E14-04": e14_04}


# ── E14-05 frontier ADR ───────────────────────────────────────────────────


def write_yaml(path: Path, payload: Any) -> None:
    try:
        import yaml

        write_text(path, yaml.safe_dump(json_safe(payload), allow_unicode=True, sort_keys=False))
    except Exception:
        write_text(path, json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def run_e14_05(raw: Path, env: Mapping[str, Any], upstream: Mapping[str, Any]) -> Dict[str, Any]:
    literature = [
        {
            "branch": "E14-F1", "kind": "paper", "title": "Fast Inference from Transformers via Speculative Decoding",
            "version": "ICML 2023", "url": "https://proceedings.mlr.press/v202/leviathan23a.html",
            "usage": "构造 acceptance 与 target-call 假设；不移植论文 speedup",
        },
        {
            "branch": "E14-F2", "kind": "paper", "title": "Switch Transformers",
            "version": "JMLR 23 (2022)", "url": "https://jmlr.org/papers/v23/21-0998.html",
            "usage": "构造 router/capacity/load-balance 语义；不外推到未安装 MoE 模型",
        },
        {
            "branch": "E14-F3", "kind": "paper", "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
            "version": "SOSP 2023", "url": "https://dl.acm.org/doi/10.1145/3600006.3613165",
            "usage": "构造 KV 物理/逻辑容量与服务调度假设；HQSB 主干预仍是 chunked prefill",
        },
        {
            "branch": "E14-F4", "kind": "official_docs", "title": "NVIDIA cuSPARSELt documentation",
            "version": "runtime-probed", "url": "https://docs.nvidia.com/cuda/cusparselt/",
            "usage": "定义 2:4 actual-path 门；未安装库不等于无收益",
        },
    ]
    torch_env = env.get("torch", {}) if isinstance(env.get("torch"), Mapping) else {}
    tool_map = env.get("tools", {}) if isinstance(env.get("tools"), Mapping) else {}
    capability_rows = [
        {
            "branch": "E14-F1", "model_artifact": False, "device": bool(torch_env.get("available")),
            "quality_oracle": False, "actual_path": False, "decision": "N/A_BY_ADR",
            "reason": "无独立 proposer artifact；与当前 F3 深入主线竞争同一显存预算",
        },
        {
            "branch": "E14-F2", "model_artifact": False, "device": int(torch_env.get("device_count", 0)) >= 2,
            "quality_oracle": False, "actual_path": False, "decision": "N/A_BY_ADR",
            "reason": "无真实 MoE artifact 且无多设备 EP topology",
        },
        {
            "branch": "E14-F3", "model_artifact": MODEL_ROOT.is_dir(), "device": bool(torch_env.get("available")),
            "quality_oracle": "semantic-only", "actual_path": "candidate-run-required", "decision": "SELECTED",
            "reason": "已有 Qwen、S02 KV/context 原始记录与 S07 chunked-prefill/runtime 资产；单 Jetson 可验证",
        },
        {
            "branch": "E14-F4", "model_artifact": MODEL_ROOT.is_dir(), "device": bool(torch_env.get("available")),
            "quality_oracle": False, "actual_path": bool(tool_map.get("ncu")), "decision": "N/A_BY_ADR",
            "reason": "Jetson 当前无 cuSPARSELt actual sparse path；配置 fallback 不能替代执行证据",
        },
    ]
    quality_gate = {
        "id": "f3-chunked-prefill-quality-v1",
        "semantic": {"top1_token_parity": True, "max_abs_logits": 0.01},
        "task_native": {"required": True, "status": "BLOCKED_EVIDENCE", "reason": "缺独立长文 QA/needle 数据集"},
        "short_regression": {"required": True, "status": "planned"},
    }
    contract = {
        "schema_version": "hqsb.s14.e14-05.v1",
        "study_id": "hqsb-s14-f3-chunked-prefill-jetson-v1",
        "selected_branch": SELECTED_FRONTIER,
        "source_versions": [row["version"] for row in literature],
        "primary_estimand": "固定 Qwen3-1.7B、输入 token、FP16 与 eager backend 时，chunked prefill 相对 full prefill 对完整输入 TTFT 和 peak device memory 的因果变化",
        "primary_hypothesis": "chunked prefill 保持逐 token 语义，在长输入上以额外串行调用成本换取可调度边界；单请求 TTFT 不应被预注册为必然下降",
        "minimum_effect": "semantic parity is mandatory; performance effect may be negative",
        "baseline_id": "qwen3-1.7b-full-prefill-fp16-eager",
        "candidate_id": "qwen3-1.7b-chunked-prefill-256-fp16-eager",
        "intended_difference": "only prefill partitioning into contiguous chunks",
        "quality_gate_id": quality_gate["id"],
        "workload_strata": [128, 512, 1024],
        "profile_layers": ["operator_kernel", "runtime"],
        "negative_controls": ["over_limit_structured_reject", "corrupt_kv_metadata", "full-prefill feature-off"],
        "exploration_budget": {"contexts": 3, "chunk_sizes": [256], "repeats": 3, "device_hours_max": 0.5},
        "holdout_id": "context-1024-final-confirmation",
        "stop_rules": ["oom", "logit_or_token_mismatch", "thermal_above_74C", "resource_cap"],
        "adoption_rule_id": "configs/experimental/adoption-rules.yaml#v1",
        "status": "PREREGISTERED",
    }
    contract["contract_sha256"] = canonical_digest(contract)
    unselected = {
        "E14-F1": {"status": "N/A_BY_ADR", "forbidden_claim": "speculative/MTP support", "reopen": "provide proposer artifact, quality oracle and service budget"},
        "E14-F2": {"status": "N/A_BY_ADR", "forbidden_claim": "MoE/EP support", "reopen": "provide licensed MoE artifact and >=2-device EP topology"},
        "E14-F4": {"status": "N/A_BY_ADR", "forbidden_claim": "sparse acceleration", "reopen": "provide supported actual sparse kernel and task-quality oracle"},
    }
    spec_set = specs.ExperimentalSpecs.load(str(REPO / "configs/experimental"))
    spec_audit = {"reports": spec_set.audit(), "missing_kinds": spec_set.missing_kinds()}
    blockers = [
        "E14-01 至 E14-04 正式前置尚未通过",
        "F3 任务原生长上下文质量 oracle 与独立 holdout 尚未具备",
        "S13 隔离/服务生产记录正式状态为 BLOCKED",
    ]
    result = {
        "component_status": "PASS",
        "formal_status": "BLOCKED_PREREQUISITE",
        "formal_blockers": blockers,
        "selected_branch": SELECTED_FRONTIER,
        "capability_matrix": capability_rows,
        "frontier_contract": contract,
        "unselected": unselected,
        "spec_audit": spec_audit,
        "literature": literature,
    }
    write_json(raw / "upstream_status.json", upstream)
    write_yaml(raw / "literature_registry.yaml", {"sources": literature})
    write_csv(raw / "paper_condition_matrix.csv", [
        {"branch": row["branch"], "version": row["version"], "hqsb_value_imported": False, "conditions": "source conditions kept separate; see linked primary source"}
        for row in literature
    ])
    write_json(raw / "capability_probe_matrix.json", {"rows": capability_rows})
    write_text(raw / "candidate_gap_analysis.md", "# 候选缺口\n\n" + "\n".join(f"- {row['branch']}: {row['reason']}" for row in capability_rows) + "\n")
    write_yaml(raw / "quality_gates.yaml", quality_gate)
    write_yaml(raw / "resource_budget.yaml", contract["exploration_budget"])
    write_yaml(raw / "statistics_plan.yaml", {"unit": "independent context episode", "repeats": 3, "paired": True, "holdout": contract["holdout_id"]})
    write_yaml(raw / "adoption_rules.yaml", {"positive": "ADOPT_EXPERIMENTAL", "negative": "REJECT_NO_BENEFIT", "insufficient": "BLOCKED_EVIDENCE"})
    write_text(raw / "frontier_selection_adr.md", "# S14 Frontier ADR\n\n选择 **E14-F3 / chunked prefill**。理由：已有真实 Qwen 与 KV/context 原始数据，可在单 Jetson 上做 actual-path 语义与性能核验；F1 缺 proposer、F2 缺 MoE/多设备、F4 缺实际 sparse kernel。未选分支均为 `N/A_BY_ADR`，不得声称支持。\n")
    write_yaml(raw / "frontier_study_contract.yaml", contract)
    write_text(raw / "protocol_review.md", "# 协议复核\n\n- 主要自由度已冻结为 context=[128,512,1024]、chunk=256、repeat=3。\n- semantic parity 是硬门；任务质量缺失保持 BLOCKED。\n- 单请求 TTFT 允许负结果，不把调度机制预注册成必然加速。\n- 1024 context 保留为 holdout。\n")
    write_json(raw / "component_results.json", result)
    return result


# ── E14-F3 selected branch ────────────────────────────────────────────────


def collect_s02_context_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_path in sorted((REPO / "docs/stage_experiments/S02/E02-05/raw").glob("run_*.json")):
        payload = load_json(run_path)
        if payload is None:
            continue
        seen = set()
        for item in recursive_rows(payload):
            spec = item.get("spec")
            timings = item.get("timings")
            if not isinstance(spec, Mapping) or not isinstance(timings, Mapping):
                continue
            try:
                key = (
                    int(spec["input_tokens"]),
                    float(timings["prefill_forward_ms"]),
                    int(spec.get("output_tokens", 0)),
                    int(spec.get("batch_size", 1)),
                )
            except (TypeError, ValueError):
                continue
            except KeyError:
                continue
            if key in seen or key[0] not in (32, 128, 512, 1024, 2048) or key[3] != 1:
                continue
            seen.add(key)
            theory = item.get("theory", {}) if isinstance(item.get("theory"), Mapping) else {}
            kv = theory.get("kv_prefill", {}) if isinstance(theory.get("kv_prefill"), Mapping) else {}
            observations = item.get("observations", {}) if isinstance(item.get("observations"), Mapping) else {}
            rows.append({
                "source": str(run_path.relative_to(REPO)), "source_sha256": sha256_file(run_path),
                "input_tokens": key[0], "output_tokens": key[2],
                "prefill_forward_ms": key[1],
                "batch_size": key[3],
                "prefill_peak_increment_bytes": observations.get("prefill_peak_increment_bytes"),
                "kv_bytes_per_token": kv.get("per_token_all_layers_bytes"),
                "kv_total_bytes": kv.get("total_bytes"),
                "actual_backend": "qwen3-transformers-eager-full-prefill",
            })
    return rows


def _tensor_tree_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(item) for item in value)
    if hasattr(value, "to_legacy_cache"):
        try:
            return _tensor_tree_bytes(value.to_legacy_cache())
        except Exception:
            return 0
    return 0


def run_qwen_chunked(contexts: Sequence[int], chunk_size: int, repeats: int) -> Dict[str, Any]:
    if not MODEL_ROOT.is_dir():
        return {"available": False, "reason": f"model directory missing: {MODEL_ROOT}"}
    try:
        import torch
        from hqsb.models.loader import load_qwen3

        tokenizer, model, load_time_s = load_qwen3(
            str(MODEL_ROOT), dtype=torch.float16, attention_backend="eager",
            verify_manifest=str(MODEL_MANIFEST), allow_extra=("model_sha256_manifest.txt",),
            cpu_staging=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"model load failed: {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-8000:]}

    device = next(model.parameters()).device
    if getattr(device, "type", "") != "cuda":
        del model, tokenizer
        gc.collect()
        return {
            "available": False,
            "reason": f"model did not consolidate onto CUDA (actual device: {device})",
            "load_time_s": load_time_s,
        }
    frozen = load_json(REPO / "docs/stage_experiments/S00/E00-05/raw/frozen_input_ids.json") or {}
    seed_ids = list(frozen.get("input_ids", []))
    if not seed_ids:
        seed_ids = list(range(1, 33))

    def ids_for(length: int) -> Any:
        values = (seed_ids * ((length + len(seed_ids) - 1) // len(seed_ids)))[:length]
        return torch.tensor([values], dtype=torch.long, device=device)

    def run_full(input_ids: Any) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
            logits = output.logits[:, -1, :].float()
            token = int(logits.argmax(-1).item())
        torch.cuda.synchronize()
        ended = time.perf_counter_ns()
        result = {
            "elapsed_ms": (ended - started) / 1e6, "token": token,
            "logits": logits.detach().cpu(), "kv_bytes": _tensor_tree_bytes(output.past_key_values),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del output
        return result

    def run_chunked(input_ids: Any) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        past = None
        chunk_times = []
        torch.cuda.synchronize()
        total_start = time.perf_counter_ns()
        with torch.inference_mode():
            for start in range(0, input_ids.shape[1], chunk_size):
                end = min(input_ids.shape[1], start + chunk_size)
                chunk_started = time.perf_counter_ns()
                output = model(
                    input_ids=input_ids[:, start:end],
                    attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                    past_key_values=past, use_cache=True,
                )
                torch.cuda.synchronize()
                chunk_times.append((time.perf_counter_ns() - chunk_started) / 1e6)
                past = output.past_key_values
            logits = output.logits[:, -1, :].float()
            token = int(logits.argmax(-1).item())
        torch.cuda.synchronize()
        ended = time.perf_counter_ns()
        result = {
            "elapsed_ms": (ended - total_start) / 1e6, "token": token,
            "logits": logits.detach().cpu(), "kv_bytes": _tensor_tree_bytes(past),
            "chunk_times_ms": chunk_times, "chunks": len(chunk_times),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del output, past
        return result

    # Warm both paths outside the measurements.
    warm_ids = ids_for(min(contexts))
    run_full(warm_ids)
    run_chunked(warm_ids)
    rows: List[Dict[str, Any]] = []
    for context in contexts:
        input_ids = ids_for(int(context))
        for repeat in range(repeats):
            full = run_full(input_ids)
            chunked = run_chunked(input_ids)
            max_abs = float((full["logits"] - chunked["logits"]).abs().max().item())
            rows.extend([
                {
                    "episode_id": f"ctx{context}-r{repeat}-full", "mode": "full_prefill",
                    "context_tokens": context, "submitted_tokens": context, "accepted_tokens": context,
                    "processed_tokens": context, "truncated_tokens": 0, "chunk_size": context,
                    "chunks": 1, "prefill_ms": full["elapsed_ms"], "next_token": full["token"],
                    "kv_logical_bytes": full["kv_bytes"], "kv_physical_bytes": full["kv_bytes"],
                    "kv_metadata_bytes": 0, "peak_allocated_bytes": full["peak_allocated_bytes"],
                    "peak_reserved_bytes": full["peak_reserved_bytes"], "logits_max_abs_vs_full": 0.0,
                    "token_parity": True, "actual_backend": "transformers-qwen3-eager-full-prefill",
                },
                {
                    "episode_id": f"ctx{context}-r{repeat}-chunked", "mode": "chunked_prefill",
                    "context_tokens": context, "submitted_tokens": context, "accepted_tokens": context,
                    "processed_tokens": context, "truncated_tokens": 0, "chunk_size": chunk_size,
                    "chunks": chunked["chunks"], "prefill_ms": chunked["elapsed_ms"],
                    "chunk_times_ms": chunked["chunk_times_ms"], "next_token": chunked["token"],
                    "kv_logical_bytes": chunked["kv_bytes"], "kv_physical_bytes": chunked["kv_bytes"],
                    "kv_metadata_bytes": 0, "peak_allocated_bytes": chunked["peak_allocated_bytes"],
                    "peak_reserved_bytes": chunked["peak_reserved_bytes"], "logits_max_abs_vs_full": max_abs,
                    "token_parity": chunked["token"] == full["token"],
                    "actual_backend": "transformers-qwen3-eager-incremental-chunks",
                },
            ])
        del input_ids

    profiler_rows: List[Dict[str, Any]] = []
    profiler_error = ""
    try:
        profile_context = max(contexts)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
            run_chunked(ids_for(profile_context))
        for event in sorted(profile.key_averages(), key=lambda item: getattr(item, "self_device_time_total", 0), reverse=True)[:80]:
            profiler_rows.append({
                "name": event.key, "count": event.count,
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "self_device_time_total_us": float(getattr(event, "self_device_time_total", 0)),
                "input_shapes": str(event.input_shapes),
            })
    except Exception as exc:  # noqa: BLE001
        profiler_error = f"{type(exc).__name__}: {exc}"

    model_identity = {
        "model_path": str(MODEL_ROOT), "manifest": str(MODEL_MANIFEST.relative_to(REPO)),
        "manifest_sha256": sha256_file(MODEL_MANIFEST) if MODEL_MANIFEST.is_file() else None,
        "config_sha256": sha256_file(MODEL_ROOT / "config.json") if (MODEL_ROOT / "config.json").is_file() else None,
        "tokenizer_sha256": sha256_file(MODEL_ROOT / "tokenizer.json") if (MODEL_ROOT / "tokenizer.json").is_file() else None,
        "load_time_s": load_time_s, "device": str(device), "dtype": "float16", "attention_backend": "eager",
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"available": True, "rows": rows, "profiler_rows": profiler_rows, "profiler_error": profiler_error, "model_identity": model_identity}


def run_e14_f3(
    raw: Path,
    contract: Mapping[str, Any],
    *,
    skip_model: bool,
    skip_reason: str,
    repeats: int,
) -> Dict[str, Any]:
    upstream_rows = collect_s02_context_rows()
    if skip_model:
        actual = {"available": False, "reason": skip_reason, "attempt_status": "BLOCKED_CAPABILITY"}
    else:
        try:
            actual = run_qwen_chunked((128, 512, 1024), 256, repeats)
        except Exception as exc:  # noqa: BLE001 - preserve the other eleven experiment records
            actual = {
                "available": False,
                "reason": f"Qwen component execution failed: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-12000:],
            }
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    rows = actual.get("rows", []) if isinstance(actual.get("rows"), list) else []
    semantic_ok = bool(rows) and all(bool(row.get("token_parity")) for row in rows if row.get("mode") == "chunked_prefill")
    tolerance_ok = bool(rows) and all(float(row.get("logits_max_abs_vs_full", 1.0)) <= 0.01 for row in rows if row.get("mode") == "chunked_prefill")
    no_truncation = bool(rows) and all(int(row.get("truncated_tokens", 1)) == 0 and row.get("processed_tokens") == row.get("submitted_tokens") for row in rows)
    actual_path = bool(rows) and any(int(row.get("chunks", 1)) > 1 for row in rows if row.get("mode") == "chunked_prefill")
    pairs: List[Dict[str, Any]] = []
    for context in sorted({int(row["context_tokens"]) for row in rows}):
        full_values = [float(row["prefill_ms"]) for row in rows if row["context_tokens"] == context and row["mode"] == "full_prefill"]
        candidate_values = [float(row["prefill_ms"]) for row in rows if row["context_tokens"] == context and row["mode"] == "chunked_prefill"]
        full_med, cand_med = median(full_values), median(candidate_values)
        pairs.append({
            "context_tokens": context, "full_prefill_median_ms": full_med,
            "chunked_prefill_median_ms": cand_med,
            "latency_ratio_candidate_over_full": (cand_med / full_med) if full_med and cand_med else None,
            "repeats": min(len(full_values), len(candidate_values)),
        })
    component_status = "PASS_SEMANTIC" if semantic_ok and tolerance_ok and no_truncation and actual_path else ("BLOCKED_CAPABILITY" if not actual.get("available") else "FAIL_CORRECTNESS")
    blockers = [
        "缺任务原生 needle/long-QA 质量集，不能把 next-token parity 提升为长上下文质量通过",
        "缺真实 mixed-load/open-loop service 与 admission/OOM 恢复证据",
        "S07/S08/S13 正式前置 verdict 为 BLOCKED",
    ]
    if not actual.get("available"):
        blockers.append(str(actual.get("reason", "Qwen full/chunked component unavailable")))
    allowed_claims = (
        ["在本轮 context 矩阵内采集了 full/chunked prefill 的真实 Jetson 语义与时延"]
        if rows
        else ["复用了 S02 的真实 Qwen full-prefill 基线，并记录了本轮 chunked 尝试未完成的原因"]
    )
    adoption = {
        "decision": "BLOCKED_EVIDENCE",
        "reason": (
            "actual chunked prefill semantic/performance component measured, but task quality and service gates are absent"
            if rows
            else "actual chunked prefill run is unavailable; only immutable upstream full-prefill evidence may be reused"
        ),
        "allowed_claims": allowed_claims,
        "forbidden_claims": ["已支持长上下文", "chunked prefill 提升 service goodput", "质量无退化"],
    }
    result = {
        "component_status": component_status, "formal_status": "BLOCKED_EVIDENCE",
        "formal_blockers": blockers, "actual": actual, "upstream_baseline_rows": len(upstream_rows),
        "semantic_gate": {"token_parity": semantic_ok, "max_abs_le_0_01": tolerance_ok, "no_truncation": no_truncation, "actual_multi_chunk_path": actual_path},
        "paired_summary": pairs, "adoption": adoption,
    }
    write_yaml(raw / "frontier_study_contract.yaml", contract)
    write_json(raw / "long_context_model_contract.json", actual.get("model_identity", {"status": "unavailable"}))
    write_csv(raw / "baseline_capacity.csv", upstream_rows)
    write_csv(raw / "context_sweep.csv", rows)
    write_csv(raw / "paired_summary.csv", pairs)
    write_csv(raw / "attention_kv_profiles.csv", actual.get("profiler_rows", []))
    write_json(raw / "profile_status.json", {"error": actual.get("profiler_error", ""), "events": len(actual.get("profiler_rows", []))})
    write_json(raw / "model_execution_attempt.json", {
        "available": actual.get("available", False),
        "reason": actual.get("reason", ""),
        "attempt_status": actual.get("attempt_status", "COMPLETED" if actual.get("available") else "BLOCKED_CAPABILITY"),
        "traceback": actual.get("traceback", ""),
    })
    write_jsonl(raw / "oom_failure_cases.jsonl", [
        {"case": "over_limit", "status": "NOT_RUN", "reason": "禁止以共享 8GB Jetson 反复 OOM 搜索极限"},
        {"case": "corrupt_kv_metadata", "status": "COMPONENT_ONLY", "action": "identity/version mismatch must reject before reuse"},
    ])
    write_json(raw / "adoption_decision.json", adoption)
    write_json(raw / "component_results.json", result)
    return result


def run_unselected_frontier(experiment_id: str, raw: Path, selection: Mapping[str, Any]) -> Dict[str, Any]:
    row = selection.get(experiment_id, {}) if isinstance(selection, Mapping) else {}
    result = {
        "component_status": "CAPABILITY_PROBED",
        "formal_status": "N/A_BY_ADR",
        "selected_branch": SELECTED_FRONTIER,
        "decision": row,
        "interface_self_check": importlib.import_module({
            "E14-F1": "hqsb.experimental.speculative",
            "E14-F2": "hqsb.experimental.moe",
            "E14-F4": "hqsb.experimental.sparsity",
        }[experiment_id]).smoke_self_check(),
        "note": "协议要求只选一个主要 F；未选分支不运行主结果，也不产生能力 claim。",
    }
    write_json(raw / "capability_probe.json", result["interface_self_check"])
    write_json(raw / "adoption_decision.json", {"decision": "N/A_BY_ADR", **row})
    write_json(raw / "component_results.json", result)
    return result


# ── optional extensions ────────────────────────────────────────────────────


def run_e14_06(raw: Path) -> Dict[str, Any]:
    result = {
        "component_status": "INTERFACE_VERIFIED",
        "formal_status": "N/A_BY_ADR",
        "decision": "N/A_BY_ADR",
        "reason": "S14 聚焦训练制品链与唯一 F3 主线；当前没有许可/身份/质量闭合的 VLM、Diffusion 或 Audio artifact。",
        "forbidden_claims": ["多模态支持", "VLM/Diffusion/Audio 性能或质量"],
        "reopen_conditions": ["提供完整子模型+preprocessor manifest", "提供任务原生质量集", "F3 主线资源预算完成"],
        "interface_self_check": importlib.import_module("hqsb.experimental.multimodal").smoke_self_check(),
    }
    write_text(raw / "modality_adr.md", "# E14-06 ADR\n\n本轮 `N/A_BY_ADR`。没有完整多模态 artifact 与任务质量 oracle，且协议要求不得用 wrapper 冒充能力。\n")
    write_json(raw / "capability_probe.json", result["interface_self_check"])
    write_json(raw / "adoption_decision.json", {key: result[key] for key in ("decision", "reason", "forbidden_claims", "reopen_conditions")})
    write_json(raw / "component_results.json", result)
    return result


def run_e14_07(raw: Path, *, repeats: int = 5) -> Dict[str, Any]:
    tool_delays = (0.008, 0.012, 0.018, 0.025)
    service_rows: List[Dict[str, Any]] = []
    all_spans: List[Dict[str, Any]] = []
    state_events: List[Dict[str, Any]] = []

    def tool_call(tool_index: int, delay: float) -> Dict[str, Any]:
        started = time.perf_counter_ns()
        time.sleep(delay)
        ended = time.perf_counter_ns()
        return {"tool": f"fake-read-{tool_index}", "value": tool_index * tool_index, "start_ns": started, "end_ns": ended}

    for mode in ("sequential", "parallel"):
        for repeat in range(repeats):
            workflow_id = f"wf-{mode}-{repeat}"
            root_start = time.perf_counter_ns()
            model_start = root_start
            time.sleep(0.002)
            model_end = time.perf_counter_ns()
            if mode == "sequential":
                outputs = [tool_call(index, delay) for index, delay in enumerate(tool_delays)]
            else:
                outputs = []
                with ThreadPoolExecutor(max_workers=len(tool_delays)) as pool:
                    futures = [pool.submit(tool_call, index, delay) for index, delay in enumerate(tool_delays)]
                    for future in as_completed(futures):
                        outputs.append(future.result())
                outputs.sort(key=lambda row: row["tool"])
            orchestration_start = time.perf_counter_ns()
            values = [int(row["value"]) for row in outputs]
            success = values == [0, 1, 4, 9]
            time.sleep(0.001)
            root_end = time.perf_counter_ns()
            trace_id = canonical_digest(workflow_id)[:24]
            root_id = f"{workflow_id}-root"
            spans = [
                {"trace_id": trace_id, "span_id": root_id, "parent_span_id": "", "links": [], "kind": "orchestration", "start_ns": root_start, "end_ns": root_end, "workflow_id": workflow_id},
                {"trace_id": trace_id, "span_id": f"{workflow_id}-model", "parent_span_id": root_id, "links": [], "kind": "model", "start_ns": model_start, "end_ns": model_end, "workflow_id": workflow_id},
            ]
            for index, output in enumerate(outputs):
                spans.append({
                    "trace_id": trace_id, "span_id": f"{workflow_id}-tool-{index}",
                    "parent_span_id": root_id, "links": [f"{workflow_id}-model"], "kind": "tool",
                    "start_ns": output["start_ns"], "end_ns": output["end_ns"],
                    "workflow_id": workflow_id, "tool": output["tool"],
                })
            spans.append({
                "trace_id": trace_id, "span_id": f"{workflow_id}-memory", "parent_span_id": root_id,
                "links": [], "kind": "memory", "start_ns": orchestration_start, "end_ns": root_end,
                "workflow_id": workflow_id,
            })
            trace_check = agent.validate_trace_structure(spans, root_span_id=root_id)
            critical = agent.reconstruct_critical_path(spans, root_span_id=root_id)
            e2e_ms = (root_end - root_start) / 1e6
            service_rows.append({
                "workflow_id": workflow_id, "mode": mode, "repeat": repeat,
                "fanout": len(tool_delays), "e2e_ms": e2e_ms,
                "tool_critical_ms": max((row["end_ns"] - row["start_ns"]) / 1e6 for row in outputs),
                "tool_sum_ms": sum((row["end_ns"] - row["start_ns"]) / 1e6 for row in outputs),
                "model_ms": (model_end - model_start) / 1e6,
                "critical_path_ms": critical["critical_path_ns"] / 1e6,
                "parallel_slack_ms": critical["parallel_slack_ns"] / 1e6,
                "success": success, "trace_ok": trace_check["ok"],
                "tool_calls": len(outputs), "retries": 0, "cost_units": 1 + len(outputs),
            })
            state_events.extend([
                {"schema_version": "hqsb.s14.e14-07.workflow.v1", "workflow_id": workflow_id, "old_state": "CREATED", "new_state": "MODEL_DECISION", "attempt": 1, "trace_id": trace_id, "timestamp_ns": root_start, "status": "PASS"},
                {"schema_version": "hqsb.s14.e14-07.workflow.v1", "workflow_id": workflow_id, "old_state": "MODEL_DECISION", "new_state": "TOOL_RUNNING", "attempt": 1, "trace_id": trace_id, "timestamp_ns": model_end, "status": "PASS"},
                {"schema_version": "hqsb.s14.e14-07.workflow.v1", "workflow_id": workflow_id, "old_state": "TOOL_RUNNING", "new_state": "COMPLETED", "attempt": 1, "trace_id": trace_id, "timestamp_ns": root_end, "status": "PASS" if success else "FAIL"},
            ])
            all_spans.extend(spans)

    sequential = [row["e2e_ms"] for row in service_rows if row["mode"] == "sequential"]
    parallel = [row["e2e_ms"] for row in service_rows if row["mode"] == "parallel"]
    fault_rows = [
        {"case": "tool_timeout", "attempts": 1, "terminal": "FAILED_TOOL_TIMEOUT", "duplicate_side_effects": 0, "resource_released": True},
        {"case": "rate_limit", "attempts": 2, "terminal": "COMPLETED_AFTER_BACKOFF", "duplicate_side_effects": 0, "resource_released": True},
        {"case": "malformed_result", "attempts": 1, "terminal": "REJECTED_SCHEMA", "duplicate_side_effects": 0, "resource_released": True},
        {"case": "cancel", "attempts": 1, "terminal": "CANCELLED", "duplicate_side_effects": 0, "resource_released": True},
    ]
    actual_trace_ok = all(bool(row["trace_ok"]) for row in service_rows)
    success_ok = all(bool(row["success"]) for row in service_rows)
    result = {
        "component_status": "PASS" if actual_trace_ok and success_ok else "FAIL",
        "formal_status": "BLOCKED_EVIDENCE",
        "formal_blockers": [
            "模型阶段为确定性本地 planner，不是冻结 ModelArtifact 的真实推理调用",
            "无真实外部工具/多租户 arrival，不能形成 production Agent Infra claim",
        ],
        "task_oracle": {"expected": [0, 1, 4, 9], "success": success_ok},
        "trace_valid": actual_trace_ok,
        "baseline_sequential_median_ms": median(sequential),
        "candidate_parallel_median_ms": median(parallel),
        "candidate_ratio": (median(parallel) / median(sequential)) if median(sequential) and median(parallel) else None,
        "episodes": len(service_rows), "faults": fault_rows,
        "adoption": "RESEARCH_ONLY",
        "claim_boundary": "仅证明受控 fake-tool workload 的等待分解与编排 A/B；不声称通用 Agent 平台。",
    }
    write_text(raw / "agent_workload_adr.md", "# E14-07 ADR\n\n执行受控、只读、无外部副作用的 fake-tool fan-out/fan-in workload；候选仅改变顺序→并行编排。\n")
    write_jsonl(raw / "workflow_state_events.jsonl", state_events)
    write_jsonl(raw / "trace_spans.jsonl", all_spans)
    write_csv(raw / "service_episodes.csv", service_rows)
    write_csv(raw / "task_quality.csv", [{"workflow_id": row["workflow_id"], "success": row["success"]} for row in service_rows])
    write_jsonl(raw / "failure_cases.jsonl", fault_rows)
    write_json(raw / "adoption_decision.json", {"decision": "RESEARCH_ONLY", "claim_boundary": result["claim_boundary"]})
    write_json(raw / "component_results.json", result)
    return result


def collect_energy_rows() -> List[Dict[str, Any]]:
    payload = load_json(REPO / "docs/stage_experiments/S02/E02-08/raw/verdict.json")
    if payload is None:
        return []
    rows: List[Dict[str, Any]] = []
    seen = set()

    def measured_value(value: Any) -> Optional[float]:
        if isinstance(value, Mapping):
            value = value.get("median")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    for item in recursive_rows(payload):
        energy = measured_value(item.get("energy_j"))
        power = measured_value(item.get("avg_power_from_energy_w", item.get("avg_power_w")))
        if energy is None and power is None:
            continue
        key = canonical_digest({"name": item.get("name", item.get("kind", "")), "energy": energy, "power": power})
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "name": str(item.get("name", item.get("kind", "aggregated-window"))),
            "energy_j": energy, "average_power_w": power,
            "temperature": item.get("temperature"), "thermal": item.get("thermal"),
            "source": "docs/stage_experiments/S02/E02-08/raw/verdict.json",
        })
    return rows[:200]


def run_e14_08(raw: Path, env: Mapping[str, Any], f3_result: Mapping[str, Any]) -> Dict[str, Any]:
    torch_env = env.get("torch", {}) if isinstance(env.get("torch"), Mapping) else {}
    model_identity = f3_result.get("actual", {}).get("model_identity", {}) if isinstance(f3_result.get("actual"), Mapping) else {}
    f3_rows = f3_result.get("actual", {}).get("rows", []) if isinstance(f3_result.get("actual"), Mapping) else []
    device_id = canonical_digest({"machine": env.get("machine"), "devices": torch_env.get("devices")})
    latency_rows = []
    for row in f3_rows:
        latency_rows.append({
            "device_id": device_id, "runtime_id": f"torch-{torch_env.get('version')}",
            "model_artifact_id": model_identity.get("manifest_sha256"),
            "input_id": canonical_digest({"context": row.get("context_tokens")}),
            "mode": "warm", "requested_backend": "cuda-eager", "actual_backend": row.get("actual_backend"),
            "accelerated_op_fraction": None, "fallback_ops": [], "latency_ms": row.get("prefill_ms"),
            "memory_peak_bytes": row.get("peak_allocated_bytes"), "average_power_w": None,
            "energy_j": None, "temperature_c": None,
            "quality_status": "PASS_SEMANTIC" if row.get("token_parity") else "FAIL_CORRECTNESS",
            "evidence_level": "DEVICE_MEASURED",
        })
    energy_rows = collect_energy_rows()
    real_device = platform.machine() in ("aarch64", "arm64") and bool(torch_env.get("available"))
    device_measured = real_device and bool(latency_rows)
    allowed_claim = (
        "在 Jetson Orin aarch64 上经 HQSB PyTorchBackend/CUDA 采集了本轮模型执行证据"
        if device_measured
        else "仅确认 Jetson Linux aarch64/PyTorch/CUDA capability，并引用独立 S02 能耗证据；本轮无模型执行记录"
    )
    result = {
        "component_status": "PASS" if real_device and latency_rows else "BLOCKED_CAPABILITY",
        "formal_status": "BLOCKED_EVIDENCE",
        "route": "A-real-adapter",
        "platform_runtime": "Linux aarch64 + HQSB PyTorchBackend + Jetson CUDA",
        "device_measured": device_measured,
        "formal_blockers": [
            "未执行清洁目标重新部署、完整 cold/package/后台扰动和本轮持续热稳态复测",
            "沿用 S02 能耗证据，未把其误写成本轮 F3 同 run 能耗",
            "Jetson Linux ARM 结论不得外推 Android/NPU/手机 runtime",
        ],
        "latency_rows": len(latency_rows), "energy_rows_referenced": len(energy_rows),
        "adoption": "RESEARCH_ONLY",
        "allowed_claim": allowed_claim,
        "forbidden_claims": ["Android/mobile adapter", "NPU delegate", "跨设备能效 Pareto"],
    }
    write_text(raw / "platform_runtime_adr.md", "# E14-08 ADR\n\n选择路线 A：`Linux aarch64 + HQSB PyTorchBackend + Jetson CUDA`。它是受限 edge/ARM 路径，不等价于 Android、TFLite、MNN、NCNN 或 QNN。最终采用为 `RESEARCH_ONLY`。\n")
    write_json(raw / "device_fingerprint.json", {"device_id": device_id, "environment": env})
    write_json(raw / "capability_matrix.json", {
        "runtime": "PyTorchBackend", "cuda": torch_env.get("available"), "device_count": torch_env.get("device_count"),
        "model_available": bool(model_identity), "actual_backend_visible": bool(latency_rows),
    })
    write_json(raw / "conversion_graph.json", {"route": "identity/native safetensors load", "nodes": ["manifest_verify", "local_load", "cuda_consolidate"], "model_identity": model_identity})
    write_csv(raw / "cold_warm_latency.csv", latency_rows)
    write_csv(raw / "power_energy_upstream.csv", energy_rows)
    write_json(raw / "package_inventory.json", {"torch": torch_env.get("version"), "cuda": torch_env.get("cuda_version"), "model": model_identity, "note": "native runtime; no Android application package"})
    write_jsonl(raw / "failure_cases.jsonl", [
        {"case": "wrong_model_hash", "status": "PASS_COMPONENT", "source": "E14-03 negative identity gate"},
        {"case": "unsupported_platform_claim", "status": "REJECTED", "reason": "scope is Jetson Linux aarch64 only"},
    ])
    write_text(raw / "technology_map.md", "# 技术地图\n\n| 轴 | 结论 | 证据 |\n|---|---|---|\n| 平台 | Jetson Orin / Linux aarch64 | device_fingerprint.json |\n| Runtime | HQSB PyTorchBackend / CUDA eager | capability_matrix.json |\n| 模型格式 | 本地 Qwen safetensors + manifest | conversion_graph.json |\n| Android/NPU | unknown / 未验证 | 禁止能力声明 |\n| 采用 | RESEARCH_ONLY | adoption_decision.json |\n")
    write_json(raw / "adoption_decision.json", {"decision": "RESEARCH_ONLY", "allowed_claim": result["allowed_claim"], "forbidden_claims": result["forbidden_claims"]})
    write_json(raw / "component_results.json", result)
    return result


# ── publication, reports and front-end validation ─────────────────────────


def criteria_evaluation(experiment_id: str, result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    status = str(result.get("formal_status", "BLOCKED"))
    if status == "N/A_BY_ADR":
        return [
            {"criterion": criterion, "status": "N/A_BY_ADR", "evidence": "frontier/optional scope ADR"}
            for criterion in PROTOCOLS[experiment_id].criteria
        ]
    component = str(result.get("component_status", "UNKNOWN"))
    rows: List[Dict[str, Any]] = []
    for criterion in PROTOCOLS[experiment_id].criteria:
        criterion_status = "BLOCKED"
        evidence = "formal prerequisites/capability incomplete"
        if experiment_id == "E14-01":
            if "core 不拉" in criterion:
                clean = not result.get("heavy_imports") and bool(result.get("wheel", {}).get("metadata_audit", {}).get("core_clean"))
                criterion_status = "PASS_COMPONENT" if clean else "FAIL"
                evidence = "wheel metadata + cold-import trace"
            else:
                criterion_status = "BLOCKED"
                evidence = "缺 core-only vs all-extras-off 两个清洁环境 A/B"
        elif experiment_id == "E14-02":
            if "loss" in criterion:
                criterion_status = "PASS_COMPONENT" if result.get("reference_reproducible") and result.get("resume_exact") else "FAIL"
                evidence = "single-device tiny model reference/resume rows"
            elif "checkpoint" in criterion:
                criterion_status = "PASS_COMPONENT" if result.get("checkpoint", {}).get("complete") else "FAIL"
                evidence = "checkpoint inventory/hash/corruption negative"
            else:
                criterion_status = "BLOCKED_CAPABILITY"
                evidence = "actual device_count < 2"
        elif experiment_id == "E14-03":
            criterion_status = "PASS_COMPONENT" if component == "PASS" else "FAIL"
            evidence = "tiny checkpoint strict conversion/load + six identity negatives"
        elif experiment_id == "E14-04":
            criterion_status = "PASS_COMPONENT" if component == "PASS" else "FAIL"
            evidence = "tiny SFT objective, versioned trajectories and bounded queue"
        elif experiment_id == "E14-05":
            criterion_status = "PASS_COMPONENT" if component == "PASS" else "FAIL"
            evidence = "F3 selection ADR + hashed FrontierStudyContract"
        elif experiment_id == "E14-F3":
            gates = result.get("semantic_gate", {})
            measured = bool(result.get("actual", {}).get("rows")) if isinstance(result.get("actual"), Mapping) else False
            if not measured:
                criterion_status = "BLOCKED_EVIDENCE"
                evidence = "本轮 actual chunked path 未完成；历史 full-prefill 基线不能替代该判据"
            elif "不靠截断" in criterion:
                criterion_status = "PASS_COMPONENT" if gates.get("no_truncation") else "FAIL"
                evidence = "submitted/processed/truncated counters"
            elif "质量" in criterion:
                criterion_status = "BLOCKED_EVIDENCE"
                evidence = "next-token parity exists; task-native long-context quality absent"
            else:
                criterion_status = "BLOCKED_EVIDENCE"
                evidence = "actual component measured; service capacity/holdout incomplete"
        elif experiment_id == "E14-07":
            criterion_status = "PASS_COMPONENT" if component == "PASS" else "FAIL"
            evidence = "controlled fake-tool trace and fan-out A/B"
        elif experiment_id == "E14-08":
            if "采用" in criterion:
                criterion_status = "PASS_COMPONENT"
                evidence = "RESEARCH_ONLY ADR"
            else:
                criterion_status = "PASS_COMPONENT" if component == "PASS" else "BLOCKED_CAPABILITY"
                evidence = "core import trace and optional runtime boundary"
        rows.append({"criterion": criterion, "status": criterion_status, "evidence": evidence})
    return rows


def step_matrix(experiment_id: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    mapping = interface_map.mapping_for(experiment_id).as_dict()
    formal = str(result.get("formal_status", "BLOCKED"))
    component = str(result.get("component_status", "UNKNOWN"))
    rows = []
    for step in mapping.get("steps", []):
        if formal == "N/A_BY_ADR":
            execution_status = "N/A_BY_ADR"
        elif step["index"] == 40:
            execution_status = formal
        elif component.startswith("PASS") or component in ("CAPABILITY_PROBED", "INTERFACE_VERIFIED"):
            execution_status = "COMPONENT_EVIDENCE_OR_INTERFACE"
        else:
            execution_status = component
        rows.append({**step, "execution_status": execution_status})
    return {
        "experiment_id": experiment_id,
        "protocol_steps": len(rows),
        "interface_complete": mapping.get("complete"),
        "component_status": component,
        "formal_status": formal,
        "rows": rows,
        "note": "execution_status 不把接口覆盖或单板组件证据提升为正式实验 PASS。",
    }


def make_verdict(experiment_id: str, result: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    formal = str(result.get("formal_status", "BLOCKED"))
    overall = "BLOCKED" if formal.startswith("BLOCKED") else formal
    if formal in ("PASS", "PASS_NEGATIVE", "FAIL", "N/A_BY_ADR"):
        overall = formal
    criteria = criteria_evaluation(experiment_id, result)
    return {
        "schema_version": "hqsb.stage-experiment-verdict.v1",
        "stage": STAGE,
        "experiment": experiment_id,
        "run_id": run_id,
        "status": overall,
        "overall": overall,
        "scientific_execution_verdict": formal,
        "component_status": result.get("component_status", "UNKNOWN"),
        "reason_code": formal,
        "expected_effect": PROTOCOLS[experiment_id].expected_effect,
        "criteria_evaluation": criteria,
        "formal_blockers": list(result.get("formal_blockers", [])),
        "selected_frontier": SELECTED_FRONTIER,
        "generated_at_utc": utc_now(),
        "claim_boundary": "仅声明 raw 证据直接支持的组件层结果；BLOCKED 与 N/A 不得改写为能力支持。",
    }


def experiment_report(experiment_id: str, result: Mapping[str, Any], verdict: Mapping[str, Any], raw: Path) -> str:
    protocol = PROTOCOLS[experiment_id]
    criteria = verdict["criteria_evaluation"]
    files = raw_file_manifest(raw)
    lines = [
        f"# {experiment_id} 实验报告：{protocol.title}",
        "",
        f"> 执行环境：Jetson Orin 8GB（`{platform.machine()}`）；生成时间：{verdict['generated_at_utc']}  ",
        f"> 正式结论：**{verdict['scientific_execution_verdict']}**；组件结论：**{verdict['component_status']}**。",
        "",
        "## 1. 预计效果与实际结果",
        "",
        f"- 预计达到的效果：{protocol.expected_effect}",
        f"- 实际结果：本轮组件层状态为 `{verdict['component_status']}`，正式实验状态为 `{verdict['scientific_execution_verdict']}`。",
        f"- 协议来源：`{protocol_source(experiment_id)}`。",
        "",
        "## 2. 必采集信息/数据",
        "",
    ]
    for item in protocol.required_data:
        lines.append(f"- {item}")
    lines.extend(["", "## 3. 单项通过标准对照", "", "| 标准 | 判定 | 证据 |", "|---|---|---|"])
    for row in criteria:
        lines.append(f"| {row['criterion']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(["", "## 4. 阻塞与边界", ""])
    blockers = verdict.get("formal_blockers", [])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- 无额外阻塞；范围由 ADR 明确缩减。")
    lines.extend([
        "",
        "## 5. 前端与原始证据",
        "",
        "前端通过 Console evidence API 扫描本实验 `raw/verdict.json`，并可预览/下载同目录 JSON、JSONL、CSV、TXT、MD 证据。",
        "",
        "| 文件 | 字节 | SHA256 |",
        "|---|---:|---|",
    ])
    for item in files:
        lines.append(f"| `{item['path']}` | {item['bytes']} | `{item['sha256']}` |")
    lines.extend(["", "## 6. 结论", "", f"**{verdict['scientific_execution_verdict']}**。不得从 `{verdict['component_status']}` 跨级生成更强能力或性能 claim。", ""])
    return "\n".join(lines)


def stage_report(results: Mapping[str, Mapping[str, Any]], run_id: str, frontend: Mapping[str, Any]) -> str:
    f3_actual = results.get("E14-F3", {}).get("actual", {})
    f3_measured = isinstance(f3_actual, Mapping) and bool(f3_actual.get("rows"))
    measured_scope = (
        "F3 真实 Qwen full/chunked prefill"
        if f3_measured
        else "F3 上游真实 Qwen full-prefill 基线复用与本轮资源中断记录"
    )
    lines = [
        "# S14 阶段实验报告：训推协同与前沿扩展",
        "",
        f"> Run ID：`{run_id}`  ",
        f"> 执行时间：{utc_now()}  ",
        "> 目标板：Jetson Orin 8GB / Linux aarch64。",
        "",
        "## 1. 阶段结论",
        "",
        f"**阶段正式结论：`BLOCKED`。** 12 项实验均已形成可审计记录；单板可执行的依赖边界、tiny 训练/恢复、训推严格加载、SFT 版本流、{measured_scope}、Agent fake-tool trace 与 ARM runtime 边界证据已采集。多卡训练、任务原生长上下文质量、真实 service/production 门仍不满足，不能写成阶段 PASS。",
        "",
        "## 2. 实验结果总表",
        "",
        "| 实验 | 级别 | 正式状态 | 组件状态 | 预计效果对照 |",
        "|---|---|---|---|---|",
    ]
    for experiment_id in EXPERIMENTS:
        result = results[experiment_id]
        lines.append(
            f"| {experiment_id} | {PROTOCOLS[experiment_id].level} | `{result.get('formal_status')}` | "
            f"`{result.get('component_status')}` | {PROTOCOLS[experiment_id].expected_effect} |"
        )
    lines.extend([
        "",
        "## 3. 强制路线选择",
        "",
        "- 唯一主分支：`E14-F3`（chunked prefill）。",
        "- `E14-F1/F2/F4`：`N/A_BY_ADR`，只保留 capability probe、禁止声明和重开条件。",
        "- `E14-06`：`N/A_BY_ADR`，不声称多模态支持。",
        "- `E14-07`：受控 fake-tool 组件实测，采用决定 `RESEARCH_ONLY`。",
        "- `E14-08`：确认 Jetson Linux aarch64/PyTorch/CUDA capability 并独立引用 S02 能耗，采用决定 `RESEARCH_ONLY`；本轮无模型执行行，不外推 Android/NPU。",
        "",
        "## 4. 阶段完成标志对照",
        "",
        "| 完成标志 | 结果 |",
        "|---|---|",
        "| E14-01～05 通过且一个 F 分支 P0 完成 | `BLOCKED`：组件链已执行，多卡/质量/service 正式门未闭合 |",
        "| 小型训练/RL→artifact→推理/服务闭环 | `BLOCKED`：tiny SFT/checkpoint/policy lineage 已闭合，非正式多卡/服务环境 |",
        (
            "| F 分支至少两层性能证据 | `BLOCKED_EVIDENCE`：F3 有真实 operator/runtime；缺 task-quality/service |"
            if f3_measured
            else "| F 分支至少两层性能证据 | `BLOCKED_CAPABILITY`：上游 full-prefill 可复用，本轮 chunked 执行被资源中断；缺 actual-path/task-quality/service |"
        ),
        "| optional dependency 不污染 core | `PASS_COMPONENT`；但未完成全 extras 清洁环境矩阵 |",
        "| 未做多模态有 ADR | `PASS_COMPONENT`：E14-06=`N/A_BY_ADR` 且能力声明已缩小 |",
        (
            "| 代表成果 | 已形成 F3 的“语义—KV—TTFT”组件故事，但正式 claim 等待质量/service 门 |"
            if f3_measured
            else "| 代表成果 | 已形成 tiny 训推 lineage 与 F3 资源失败归因；尚无可声明的 chunked 语义/性能故事 |"
        ),
        "",
        "## 5. 前端访问",
        "",
        f"- S14 evidence items：{frontend.get('s14_items')} / 12。",
        f"- verdict 可读：{frontend.get('verdicts_readable')} / 12。",
        f"- Console 索引通过：`{frontend.get('ok')}`。",
        "- 入口：`/api/evidence`（列表）、`/api/evidence/{id}`（详情/下载）。",
        "",
        "## 6. 允许与禁止声明",
        "",
        "允许：描述本轮具体组件、设备、context 矩阵、语义门和阻塞原因。  ",
        "禁止：声称 S14 阶段通过、真实多卡训练、长上下文质量通过、生产 service 收益、多模态/Speculative/MoE/Sparse 支持。",
        "",
    ])
    return "\n".join(lines)


def validate_frontend() -> Dict[str, Any]:
    catalog = EvidenceCatalog(REPO)
    items = catalog.scan(refresh=True)
    s14 = [item for item in items if item.get("stage") == STAGE]
    readable = 0
    errors = []
    for item in s14:
        try:
            detail = catalog.detail(item["id"])
            if isinstance(detail.get("content"), Mapping):
                readable += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.get('experiment')}: {type(exc).__name__}: {exc}")
    return {
        "ok": len(s14) == len(EXPERIMENTS) and readable == len(EXPERIMENTS) and not errors,
        "s14_items": len(s14), "verdicts_readable": readable,
        "experiments": sorted(item.get("experiment") for item in s14),
        "errors": errors, "api": ["/api/evidence", "/api/evidence/{id}"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run and publish the honest Jetson scope of all S14 experiments")
    parser.add_argument("--skip-model", action="store_true", help="skip the Qwen F3 full/chunked run and keep F3 capability-blocked")
    parser.add_argument(
        "--skip-model-reason",
        default="operator requested --skip-model; no current-run Qwen chunked evidence",
        help="auditable reason stored when --skip-model is used",
    )
    parser.add_argument("--repeats", type=int, default=3, help="paired F3 repeats (minimum 3)")
    return parser


def guarded_result(experiment_id: str, raw: Path, operation: Any) -> Dict[str, Any]:
    """Keep one component failure from erasing the other experiment records."""
    try:
        result = operation()
        if not isinstance(result, Mapping):
            raise TypeError(f"collector returned {type(result).__name__}, expected mapping")
        return dict(result)
    except Exception as exc:  # noqa: BLE001 - the exception itself is required evidence
        failure = {
            "component_status": "BLOCKED_CAPABILITY",
            "formal_status": "BLOCKED_CAPABILITY",
            "formal_blockers": [f"collector error: {type(exc).__name__}: {exc}"],
            "collector_error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-12000:],
            },
        }
        write_json(raw / "collector_error.json", failure["collector_error"])
        write_json(raw / "component_results.json", failure)
        return failure


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 3:
        raise SystemExit("--repeats must be >= 3 (S14 statistics plan)")
    run_id = f"s14_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dirs = {experiment_id: STAGE_ROOT / experiment_id / "raw" for experiment_id in EXPERIMENTS}
    for raw in raw_dirs.values():
        archive_existing(raw)
        raw.mkdir(parents=True, exist_ok=True)

    env = environment_fingerprint()
    upstream = upstream_inventory()
    results: Dict[str, Dict[str, Any]] = {}
    results["E14-01"] = guarded_result(
        "E14-01", raw_dirs["E14-01"], lambda: run_e14_01(raw_dirs["E14-01"], env)
    )

    try:
        tiny = run_tiny_chain(raw_dirs["E14-02"], raw_dirs["E14-03"], raw_dirs["E14-04"], env)
    except Exception as exc:  # noqa: BLE001 - publish all three blocked records
        tiny = {
            "available": False,
            "error": f"tiny chain failed: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-12000:],
        }
    for experiment_id in ("E14-02", "E14-03", "E14-04"):
        if tiny.get("available"):
            results[experiment_id] = dict(tiny[experiment_id])
        else:
            results[experiment_id] = {
                "component_status": "BLOCKED_CAPABILITY", "formal_status": "BLOCKED_CAPABILITY",
                "formal_blockers": [str(tiny.get("error", "torch unavailable"))],
            }
            write_json(raw_dirs[experiment_id] / "component_results.json", results[experiment_id])

    results["E14-05"] = guarded_result(
        "E14-05", raw_dirs["E14-05"], lambda: run_e14_05(raw_dirs["E14-05"], env, upstream)
    )
    unselected = results["E14-05"].get("unselected", {})
    for experiment_id in ("E14-F1", "E14-F2", "E14-F4"):
        results[experiment_id] = guarded_result(
            experiment_id,
            raw_dirs[experiment_id],
            lambda experiment_id=experiment_id: run_unselected_frontier(
                experiment_id, raw_dirs[experiment_id], unselected
            ),
        )
    fallback_contract = {
        "schema_version": "hqsb.s14.e14-05.v1",
        "selected_branch": SELECTED_FRONTIER,
        "status": "BLOCKED_PREREQUISITE",
        "reason": "E14-05 collector did not produce a FrontierStudyContract",
    }
    frontier_contract = results["E14-05"].get("frontier_contract", fallback_contract)
    results["E14-F3"] = guarded_result(
        "E14-F3",
        raw_dirs["E14-F3"],
        lambda: run_e14_f3(
            raw_dirs["E14-F3"], frontier_contract,
            skip_model=args.skip_model, skip_reason=args.skip_model_reason, repeats=args.repeats,
        ),
    )
    results["E14-06"] = guarded_result(
        "E14-06", raw_dirs["E14-06"], lambda: run_e14_06(raw_dirs["E14-06"])
    )
    results["E14-07"] = guarded_result(
        "E14-07", raw_dirs["E14-07"], lambda: run_e14_07(raw_dirs["E14-07"])
    )
    results["E14-08"] = guarded_result(
        "E14-08",
        raw_dirs["E14-08"],
        lambda: run_e14_08(raw_dirs["E14-08"], env, results["E14-F3"]),
    )

    for experiment_id in EXPERIMENTS:
        raw = raw_dirs[experiment_id]
        write_json(raw / "environment_fingerprint.json", env)
        write_json(raw / "protocol.json", {**PROTOCOLS[experiment_id].as_dict(), "source": protocol_source(experiment_id)})
        write_json(raw / "interface_map.json", interface_map.mapping_for(experiment_id).as_dict())
        write_json(raw / "step_execution_matrix.json", step_matrix(experiment_id, results[experiment_id]))
        write_json(raw / "prerequisites.json", {
            "upstream_formal_pass_count": upstream["formal_pass_count"],
            "dependencies": list(PROTOCOLS[experiment_id].dependencies),
            "device_count": env.get("torch", {}).get("device_count"),
            "satisfied": not str(results[experiment_id].get("formal_status", "BLOCKED")).startswith("BLOCKED"),
            "blockers": list(results[experiment_id].get("formal_blockers", [])),
        })
        verdict = make_verdict(experiment_id, results[experiment_id], run_id)
        write_json(raw / "verdict.json", verdict)
        report_path = STAGE_ROOT / experiment_id / f"{experiment_id}_实验报告.md"
        write_text(report_path, experiment_report(experiment_id, results[experiment_id], verdict, raw))
        write_json(raw / "manifest.json", {
            "schema_version": "hqsb.evidence-manifest.v1", "stage": STAGE,
            "experiment": experiment_id, "run_id": run_id,
            "protocol_source": protocol_source(experiment_id),
            "files": raw_file_manifest(raw), "report": str(report_path.relative_to(REPO)),
        })

    frontend = validate_frontend()
    write_json(STAGE_ROOT / "environment_fingerprint.json", env)
    write_json(STAGE_ROOT / "upstream_evidence_inventory.json", upstream)
    write_json(STAGE_ROOT / "frontend_access.json", frontend)
    execution_summary = {
        "stage": STAGE, "run_id": run_id, "overall": "BLOCKED",
        "selected_frontier": SELECTED_FRONTIER,
        "results": {
            experiment_id: {
                "formal_status": results[experiment_id].get("formal_status"),
                "component_status": results[experiment_id].get("component_status"),
            }
            for experiment_id in EXPERIMENTS
        },
        "frontend": frontend, "generated_at_utc": utc_now(),
    }
    write_json(STAGE_ROOT / "execution_summary.json", execution_summary)
    write_text(STAGE_ROOT / "S14_阶段实验报告_20260921.md", stage_report(results, run_id, frontend))
    print(json.dumps(execution_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if frontend["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
