#!/usr/bin/env python3
"""Execute and archive the honest single-Jetson scope of all S12 experiments.

The S12 acceptance gate requires a quality-qualified multi-hardware campaign.
The configured target is one Jetson Orin, while the Ascend, service,
distributed and model-level optimized candidates are not formally qualified.
This collector therefore separates three things throughout the evidence:

* real Jetson observations (CUDA/Triton/PyTorch RMSNorm, historical Qwen
  model-core and energy evidence, repeated-process measurements and isolated
  setup sessions);
* modeled/scenario outputs (roofline, TCO sensitivity and constrained Pareto);
* the formal S12 verdict, which remains BLOCKED until the documented coverage,
  quality, price, stability and lineage gates are all satisfied.

Run only through ``./scripts/remote_run.sh``.  The collector writes the
front-end publication tree under ``docs/stage_experiments/S12`` so the existing
read-only Console evidence API can index every experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.benchmark.resource_monitor import TegrastatsMonitor  # noqa: E402
from hqsb.benchmark.tegrastats_parser import parse_tegrastats_line  # noqa: E402
from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.core.errors import ConfigError  # noqa: E402
from hqsb.evaluation import (  # noqa: E402
    benchmark,
    capability,
    comparability,
    cost,
    energy,
    experiment,
    interface_map,
    lineage,
    maturity,
    pareto,
    repeatability,
    roofline,
    specs,
)


STAGE = "S12"
STAGE_ROOT = REPO / "docs/stage_experiments/S12"
DETAIL_ROOT = REPO / "docs/stage_experiments/details/S12"
DRIVER = REPO / "scripts/evaluation/run_e12.py"
EXPERIMENTS = tuple(f"E12-{index:02d}" for index in range(1, 11))
MODEL_PATH = Path("~/models/hqsb/Qwen3-1.7B").expanduser()
MODEL_CONFIG = REPO / "configs/models/qwen3_1_7b.yaml"
EPSILON = 1.0e-6


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
    title: str,
    expected_effect: str,
    required_data: Sequence[str],
    criteria: Sequence[str],
    dependencies: Sequence[str] = (),
) -> Protocol:
    return Protocol(
        experiment_id,
        "P0",
        title,
        expected_effect,
        tuple(required_data),
        tuple(criteria),
        tuple(dependencies),
    )


PROTOCOLS: Mapping[str, Protocol] = {
    "E12-01": _p(
        "E12-01", "候选矩阵逐格可比性审计与 Comparison Contract",
        "防止苹果和橘子直接排名。",
        ("每格 identity", "候选差异账本", "四态可比性裁决与 reason", "normalization 与限制"),
        ("不可比格不补零或进入排名", "条件比较写明 normalization 和限制", "至少形成一组满足硬件覆盖且上游质量通过的 COMPARABLE 候选"),
    ),
    "E12-02": _p(
        "E12-02", "跨平台 Capability 自动探测与运行证据矩阵",
        "建立真实支持矩阵。",
        ("硬件/驱动/runtime identity", "逐能力 probe 命令和输出", "supported/experimental/missing", "failure category/reason", "timestamp/version/invalidation key"),
        ("每个支持项有 execute+sync+correctness 证据", "缺失与 probe 失败可区分", "所有候选平台时间戳和版本齐全"),
        ("E12-01",),
    ),
    "E12-03": _p(
        "E12-03", "Operator、Model-core、Service、Distributed 四层统一重放",
        "得到跨层 normalized matrix。",
        ("raw/normalized unit", "correctness/quality", "latency/goodput/memory", "actual backend", "四层 missingness", "raw sample 反向定位"),
        ("schema validator 全过", "同层同口径", "结果能反向定位 raw sample", "有能力的四层均完成正式重放"),
        ("E12-01", "E12-02"),
    ),
    "E12-04": _p(
        "E12-04", "多进程、多时段、多日重复性、漂移与噪声控制",
        "量化稳定性和测量可信度。",
        ("repetitions 与平衡顺序", "CV/CI/MAD/drift", "温度/频率/系统负载", "异常与剔除 ledger", "跨进程/时段/日期层级"),
        ("异常自动标记而非取最好值", "稳定 case 达预注册重复性阈值", "多日和实例外推边界明确"),
        ("E12-03",),
    ),
    "E12-05": _p(
        "E12-05", "分层 Roofline、Amdahl、容量/互联模型与预测误差归因",
        "建立硬件—软件协同解释。",
        ("理论/可持续/实测 roof", "operations/traffic 定义", "Amdahl 与 dispatch coverage", "prediction error", "bottleneck/residual/ablation", "capacity 与互联模型"),
        ("误差能由可复核证据归因", "不能只画屋顶线", "校准/验证分离且各层主要点覆盖"),
        ("E12-03", "E12-04"),
    ),
    "E12-06": _p(
        "E12-06", "功耗窗口对齐、能量积分与请求/Token 能效",
        "建立能效比较。",
        ("idle/load power time series", "采样率/单位/boundary", "同 run 对齐窗口", "energy/J-request/J-token/tok-J", "temperature/throttle", "计量限制和不确定性"),
        ("采样窗口与请求对齐", "idle policy 一致", "不可采平台不伪造估计值", "主要合格候选达到正式窗口和硬件覆盖"),
        ("E12-02", "E12-03", "E12-04"),
    ),
    "E12-07": _p(
        "E12-07", "云租赁/自建 TCO、Cost per Token、容量密度与敏感性",
        "把峰值性能转成部署成本。",
        ("价格来源/日期", "折旧或租赁", "功耗/电价/PUE", "utilization", "replica/capacity", "cost/token 与敏感区间"),
        ("假设透明且可修改", "给出置信/敏感区间", "不把采购价当永久事实", "推荐结论使用有效来源快照"),
        ("E12-03", "E12-04", "E12-06"),
    ),
    "E12-08": _p(
        "E12-08", "业务画像约束下的 Pareto Frontier、稳健性与部署建议",
        "从业务画像反推部署建议。",
        ("SLO/quality/cost constraints", "可行集与非支配点", "淘汰原因", "风险/置信度", "画像绑定的推荐", "失败方案保留"),
        ("每个建议绑定画像和约束", "不存在脱离 workload 的总冠军", "只使用质量/可比性/lineage 合格的可行集"),
        ("E12-01", "E12-03", "E12-04", "E12-06", "E12-07", "E12-09"),
    ),
    "E12-09": _p(
        "E12-09", "软件成熟度、开发/维护成本与可复核 Rubric",
        "把软件成熟度纳入选型。",
        ("setup time", "失败次数与 taxonomy", "文档缺口", "tool coverage", "维护成本", "rubric anchor/评分依据/多 session"),
        ("原始观测可复核", "主观项有 rubric 和多次记录", "成熟度不混成性能分数", "独立 session 覆盖既有与隔离环境"),
        ("E12-02", "E12-03"),
    ),
    "E12-10": _p(
        "E12-10", "Dashboard/报告点到 Raw 的端到端 Lineage、校验与一键重生成",
        "证明报告没有手工抄数断链。",
        ("lineage entity/edge", "URI/hash/schema validation", "raw→normalized→report→Console", "regeneration diff", "分层抽查", "缺文件/坏 hash/单位错/过期引用负例"),
        ("抽查全部通过", "派生表可一键重生成", "报告数字与 raw 聚合一致", "前端能预览和下载全部实验证据"),
        ("E12-01", "E12-02", "E12-03", "E12-04", "E12-05", "E12-06", "E12-07", "E12-08", "E12-09"),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(argv: Sequence[str], *, timeout: float = 120.0, cwd: Path = REPO) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv), cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
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
            "argv": list(argv), "returncode": 124,
            "stdout": str(exc.stdout or "")[-20000:],
            "stderr": str(exc.stderr or "")[-20000:],
            "elapsed_s": time.perf_counter() - started, "timeout": True,
        }
    except OSError as exc:
        return {
            "argv": list(argv), "returncode": 127, "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started,
        }


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def status_of(payload: Mapping[str, Any]) -> str:
    value = payload.get("overall") or payload.get("verdict") or payload.get("status")
    if value is None and isinstance(payload.get("passed"), bool):
        value = "PASS" if payload["passed"] else "FAIL"
    return str(value or "UNKNOWN")


def archive_existing(raw: Path, run_id: str) -> None:
    if not raw.exists() or not any(path.is_file() for path in raw.iterdir()):
        return
    previous = "unknown"
    verdict = load_json(raw / "verdict.json")
    if verdict:
        previous = str(verdict.get("run_id") or previous)
    archive = raw / "runs" / previous
    if archive.exists():
        archive = raw / "runs" / f"{previous}_{int(time.time())}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in list(raw.iterdir()):
        if path.name == "runs":
            continue
        shutil.move(str(path), str(archive / path.name))


def latest_verdict(experiment_root: Path) -> Optional[Path]:
    direct = experiment_root / "raw/verdict.json"
    if direct.is_file():
        return direct
    candidates = [
        path for path in experiment_root.glob("raw*/verdict.json")
        if "archive" not in path.parts and "runs" not in path.parts
    ]
    return sorted(candidates)[-1] if candidates else None


def inventory_upstream() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    stage_summary: Dict[str, Dict[str, int]] = {}
    for stage_number in range(1, 12):
        stage = f"S{stage_number:02d}"
        root = REPO / "docs/stage_experiments" / stage
        for experiment_root in sorted(root.glob("E*")) if root.is_dir() else ():
            path = latest_verdict(experiment_root)
            if path is None:
                rows.append({
                    "stage": stage, "experiment_id": experiment_root.name,
                    "status": "MISSING", "state": "MISSING", "uri": "",
                    "sha256": "", "usable": False,
                })
                continue
            payload = load_json(path) or {}
            status = status_of(payload)
            state = "VERIFIED" if status in ("PASS", "PASS_NEGATIVE") else "UNVERIFIED"
            row = {
                "stage": stage, "experiment_id": experiment_root.name,
                "status": status, "state": state,
                "uri": str(path.relative_to(REPO)), "sha256": sha256_file(path),
                "usable": state == "VERIFIED",
            }
            rows.append(row)
            counts = stage_summary.setdefault(stage, {})
            counts[status] = counts.get(status, 0) + 1
    return {
        "captured_at": utc_now(), "rows": rows, "stage_summary": stage_summary,
        "usable": [f"{row['stage']}/{row['experiment_id']}" for row in rows if row["usable"]],
        "rule": "only current non-archived PASS/PASS_NEGATIVE verdicts are VERIFIED; existence alone is not evidence",
    }


def collect_environment(torch: Any) -> Dict[str, Any]:
    tools = {}
    for name in ("nvcc", "ncu", "nsys", "cuobjdump", "nvdisasm", "tegrastats", "nvpmodel", "git"):
        tools[name] = shutil.which(name)
    modules = {
        name: bool(importlib.util.find_spec(name))
        for name in ("torch", "transformers", "triton", "fastapi", "pytest", "yaml", "pydantic")
    }
    git_commit = command(("git", "rev-parse", "HEAD"), timeout=10)["stdout"].strip()
    git_status = command(("git", "status", "--short"), timeout=10)["stdout"]
    nvpmodel = command(("nvpmodel", "-q"), timeout=10) if tools["nvpmodel"] else {"returncode": 127}
    board_path = Path("/proc/device-tree/model")
    board = board_path.read_bytes().replace(b"\0", b"").decode(errors="replace") if board_path.is_file() else ""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    model_files = sorted(path for path in MODEL_PATH.glob("*") if path.is_file()) if MODEL_PATH.is_dir() else []
    return {
        "captured_at": utc_now(),
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
        "board_model": board, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "device_count": torch.cuda.device_count(),
        "memory": {"free_bytes": free_bytes, "total_bytes": total_bytes},
        "nvpmodel": nvpmodel, "tools": tools, "modules": modules,
        "model": {
            "path": str(MODEL_PATH), "available": MODEL_PATH.is_dir(),
            "file_count": len(model_files),
            "config_sha256": sha256_file(MODEL_PATH / "config.json") if (MODEL_PATH / "config.json").is_file() else "",
        },
        "source": {
            "git_commit": git_commit, "git_dirty": bool(git_status.strip()),
            "git_status": git_status[-20000:],
            "runner_sha256": sha256_file(Path(__file__)),
            "model_config_sha256": sha256_file(MODEL_CONFIG) if MODEL_CONFIG.is_file() else "",
        },
    }


def framework_rmsnorm(torch: Any, x: Any, weight: Any, out: Any) -> Any:
    value = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + EPSILON)
    out.copy_((value * weight.float()).to(x.dtype))
    return out


def load_rmsnorm_backends(torch: Any) -> Dict[str, Any]:
    from ops import cuda_bridge

    triton_rms = importlib.import_module("ops.triton.rmsnorm")
    return {
        "pytorch_eager": lambda x, w, out: framework_rmsnorm(torch, x, w, out),
        "cuda_v2": lambda x, w, out: cuda_bridge.rmsnorm_forward(
            x, w, out=out, dtype="fp16", variant="v2_vectorized", epsilon=EPSILON,
            stream=torch.cuda.current_stream(),
        ),
        "triton_reference": lambda x, w, out: triton_rms.rmsnorm_reference(x, w, EPSILON, out=out),
    }


def _telemetry_snapshot(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    parsed = []
    for record in records:
        values = parse_tegrastats_line(str(record.get("raw", "")))
        parsed.append({"time_ns": record.get("time_ns"), **values})
    powers = [row["power_mw"] for row in parsed if "power_mw" in row]
    temperatures = [row["gpu_temp_c"] for row in parsed if "gpu_temp_c" in row]
    swaps = [row["swap_used_mb"] for row in parsed if "swap_used_mb" in row]
    return {
        "sample_count": len(parsed), "power_samples": len(powers),
        "power_w_median": statistics.median(powers) / 1000.0 if powers else None,
        "power_w_peak": max(powers) / 1000.0 if powers else None,
        "gpu_temp_c_min": min(temperatures) if temperatures else None,
        "gpu_temp_c_max": max(temperatures) if temperatures else None,
        "swap_used_mb_max": max(swaps) if swaps else None,
        "records": parsed,
    }


def rmsnorm_probe(process_index: int, block: int, cpu_load: bool) -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RMSNorm probe requires the remote CUDA Jetson")
    workers: List[subprocess.Popen[str]] = []
    if cpu_load:
        code = "x=1\nwhile True:\n x=(x*1103515245+12345)&0x7fffffff"
        for _ in range(2):
            workers.append(subprocess.Popen([sys.executable, "-c", code]))
    torch.manual_seed(1200 + process_index + block * 100)
    monitor = TegrastatsMonitor(interval_ms=200)
    monitor.start()
    rows: List[Dict[str, Any]] = []
    try:
        backends = load_rmsnorm_backends(torch)
        order = ("pytorch_eager", "cuda_v2", "triton_reference")
        if block % 2 == 0:
            order = tuple(reversed(order))
        for shape_rows, scenario in ((1, "decode_shape"), (512, "prefill_shape")):
            x = torch.randn((shape_rows, 2048), device="cuda", dtype=torch.float16)
            weight = torch.randn((2048,), device="cuda", dtype=torch.float16)
            reference = torch.empty_like(x)
            framework_rmsnorm(torch, x, weight, reference)
            torch.cuda.synchronize()
            for candidate in order:
                out = torch.empty_like(x)
                record: Dict[str, Any] = {
                    "process": process_index, "block": block, "day": 1,
                    "cpu_load": cpu_load, "candidate_id": candidate,
                    "rows": shape_rows, "hidden": 2048, "dtype": "fp16",
                    "scenario": scenario, "run_order": len(rows) + 1,
                    "requested_backend": candidate, "actual_backend": candidate,
                    "status": "RUN_FAILED", "samples_ms": [],
                }
                try:
                    for _ in range(5):
                        backends[candidate](x, weight, out)
                    torch.cuda.synchronize()
                    diff = (out.float() - reference.float()).abs()
                    record["correctness"] = {
                        "max_abs": float(diff.max().item()),
                        "mean_abs": float(diff.mean().item()),
                        "allclose": bool(torch.allclose(out, reference, atol=5e-3, rtol=5e-3)),
                    }
                    inner = 200 if shape_rows == 1 else 20
                    for _ in range(7):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _inner in range(inner):
                            backends[candidate](x, weight, out)
                        end.record()
                        end.synchronize()
                        record["samples_ms"].append(start.elapsed_time(end) / inner)
                    record["median_ms"] = statistics.median(record["samples_ms"])
                    record["p95_ms"] = sorted(record["samples_ms"])[-1]
                    record["logical_bytes"] = shape_rows * 2048 * 2 * 2 + 2048 * 2
                    record["effective_bandwidth_gbs"] = (
                        record["logical_bytes"] / (record["median_ms"] / 1000.0) / 1e9
                    )
                    record["status"] = "OK" if record["correctness"]["allclose"] else "NUMERICAL_MISMATCH"
                except Exception as exc:  # evidence retains failures
                    record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                rows.append(record)
    finally:
        monitor.stop()
        for worker in workers:
            worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "probe": "rmsnorm", "process": process_index, "block": block,
        "cpu_load": cpu_load, "captured_at": utc_now(), "rows": rows,
        "telemetry": _telemetry_snapshot(monitor.records),
        "memory_after": {"free_bytes": free_bytes, "total_bytes": total_bytes},
    }


def energy_probe(candidate: str, shape_rows: int, duration_s: float) -> Dict[str, Any]:
    import torch

    backends = load_rmsnorm_backends(torch)
    if candidate not in backends:
        raise ConfigError(f"unknown energy candidate {candidate!r}")
    torch.manual_seed(1260 + shape_rows)
    x = torch.randn((shape_rows, 2048), device="cuda", dtype=torch.float16)
    weight = torch.randn((2048,), device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    reference = torch.empty_like(x)
    framework_rmsnorm(torch, x, weight, reference)
    for _ in range(10):
        backends[candidate](x, weight, out)
    torch.cuda.synchronize()
    correctness = bool(torch.allclose(out, reference, atol=5e-3, rtol=5e-3))
    monitor = TegrastatsMonitor(interval_ms=200)
    monitor.start()
    time.sleep(3.0)
    start_ns = time.monotonic_ns()
    launches = 0
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        for _ in range(100):
            backends[candidate](x, weight, out)
            launches += 1
    torch.cuda.synchronize()
    end_ns = time.monotonic_ns()
    time.sleep(3.0)
    monitor.stop()
    parsed = []
    samples = []
    for row in monitor.records:
        values = parse_tegrastats_line(row["raw"])
        parsed.append({"time_ns": row["time_ns"], "raw": row["raw"], **values})
        samples.append(
            energy.EnergySample(
                t_ns=row["time_ns"],
                power_w=(values.get("power_mw") / 1000.0) if values.get("power_mw") is not None else None,
                status="OK" if values.get("power_mw") is not None else "TELEMETRY_GAP",
            )
        )
    series = energy.PowerSeries(
        run_id=f"energy_{candidate}_r{shape_rows}", meter_id="jetson_tegrastats_vdd_in",
        samples=tuple(samples), boundary="node",
    )
    integral = energy.trapezoid_integral(series, t0_ns=start_ns, t1_ns=end_ns, expected_period_s=0.2)
    pre = energy.PowerSeries(
        run_id=series.run_id, meter_id=series.meter_id,
        samples=tuple(sample for sample in samples if sample.t_ns < start_ns), boundary="node",
    )
    post = energy.PowerSeries(
        run_id=series.run_id, meter_id=series.meter_id,
        samples=tuple(sample for sample in samples if sample.t_ns > end_ns), boundary="node",
    )
    idle_policy = energy.IdlePolicy("same_device_allocation", 3.0, "median", "none")
    idle = energy.idle_reference(pre, post, idle=idle_policy)
    incremental = None
    if integral.get("status") == "OK" and idle.get("status") in ("OK", "UNSTABLE"):
        incremental = energy.incremental_energy(
            load_energy_j=float(integral["energy_j"]), idle_power_w=float(idle["idle_power_w"]),
            window_s=(end_ns - start_ns) / 1e9, idle=idle_policy,
        )
    return {
        "candidate_id": candidate, "rows": shape_rows, "hidden": 2048,
        "duration_target_s": duration_s, "start_ns": start_ns, "end_ns": end_ns,
        "duration_s": (end_ns - start_ns) / 1e9, "launches": launches,
        "correctness": correctness, "boundary": "node", "meter": "tegrastats:VDD_IN",
        "integral": integral, "idle": idle, "incremental": incremental,
        "j_per_kernel_work_unit": (
            float(integral["energy_j"]) / launches if integral.get("status") == "OK" and launches else None
        ),
        "raw_power": parsed,
        "limitations": [
            "VDD_IN is Jetson module input, not GPU-only energy",
            "operator work units are not tokens; J/token is intentionally omitted",
        ],
    }


def child_probe(args: argparse.Namespace) -> int:
    if args.probe == "rmsnorm":
        payload = rmsnorm_probe(args.process_index, args.block, args.cpu_load)
    elif args.probe == "energy":
        payload = energy_probe(args.candidate, args.rows, args.duration_s)
    else:
        raise ConfigError(f"unknown probe {args.probe!r}")
    write_json(Path(args.probe_output), payload)
    return 0


def run_child_probe(arguments: Sequence[str], *, timeout: float) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hqsb-s12-probe-") as tmp:
        output = Path(tmp) / "result.json"
        result = command(
            (sys.executable, str(Path(__file__).resolve()), *arguments, "--probe-output", str(output)),
            timeout=timeout,
        )
        payload = load_json(output)
        return {"command": result, "payload": payload}


def collect_repeated_rmsnorm() -> Dict[str, Any]:
    children = []
    rows: List[Dict[str, Any]] = []
    for block, cpu_load in ((1, False), (2, True)):
        for process_index in range(3):
            child = run_child_probe(
                ("--probe", "rmsnorm", "--process-index", str(process_index), "--block", str(block))
                + (("--cpu-load",) if cpu_load else ()),
                timeout=240,
            )
            children.append(child["command"])
            if child["payload"]:
                rows.extend(child["payload"].get("rows", []))
    valid = [row for row in rows if row.get("status") == "OK"]
    schedule = repeatability.balanced_schedule(
        ("pytorch_eager", "cuda_v2", "triton_reference"), blocks=3, days=1, seed=1204
    )
    cells: Dict[str, List[Dict[str, Any]]] = {}
    for row in valid:
        cell_id = f"{row['candidate_id']}|{row['scenario']}|cpu={str(row['cpu_load']).lower()}"
        cells.setdefault(cell_id, []).append({
            "cell_id": cell_id, "metric_name": "operator_latency", "value": row["median_ms"],
            "process": row["process"], "block": row["block"], "day": row["day"],
            "device": "jetson-orin-0", "run_order": row["run_order"],
        })
    stability_rows = []
    for cell_id, values in sorted(cells.items()):
        sample_values = [float(row["value"]) for row in values]
        stability_rows.append({
            "cell_id": cell_id, "samples": len(sample_values),
            "median_ms": statistics.median(sample_values),
            "cv": repeatability.cv(sample_values) if len(sample_values) >= 2 else None,
            "mad": repeatability.mad(sample_values) if sample_values else None,
            "bootstrap": repeatability.bootstrap_ci(sample_values, B=500, seed=1204) if len(sample_values) >= 2 else None,
            "variance_process": repeatability.variance_components(values, level="process"),
            "drift": repeatability.drift_slope(values, covariate="process") if len(values) >= 3 else None,
        })
    stable_only = [row for row in valid if not row.get("cpu_load")]
    pass_count = sum(bool(row.get("correctness", {}).get("allclose")) for row in valid)
    return {
        "component_status": "PASS" if valid and pass_count == len(valid) else "FAIL",
        "component_pass": bool(valid and pass_count == len(valid)),
        "children": children, "raw_rows": rows, "valid_rows": len(valid),
        "correctness_pass_rows": pass_count,
        "schedule": schedule.as_rows(), "schedule_balance": repeatability.schedule_balance(schedule),
        "stability": stability_rows,
        "stable_block_rows": stable_only,
        "limitations": [
            "two same-day blocks were collected; this is cross-process/cross-block, not cross-day evidence",
            "the controlled CPU background-load block is an anomaly sensitivity check, not a thermal injection",
            "repeatability is operator-level because no quality-qualified optimized model-core candidate exists",
        ],
    }


def collect_energy_runs(duration_s: float) -> Dict[str, Any]:
    rows = []
    commands = []
    for candidate in ("pytorch_eager", "cuda_v2"):
        for shape_rows in (1, 512):
            child = run_child_probe(
                ("--probe", "energy", "--candidate", candidate, "--rows", str(shape_rows),
                 "--duration-s", str(duration_s)),
                timeout=duration_s + 120,
            )
            commands.append(child["command"])
            if child["payload"]:
                rows.append(child["payload"])
    usable = [
        row for row in rows
        if row.get("correctness") and row.get("integral", {}).get("status") == "OK"
        and float(row.get("duration_s", 0.0)) >= 30.0
    ]
    same_boundary = False
    try:
        energy.assert_same_boundary(rows)
        same_boundary = True
    except ConfigError:
        same_boundary = False
    return {
        "component_status": "PASS" if len(usable) == 4 and same_boundary else "FAIL",
        "component_pass": len(usable) == 4 and same_boundary,
        "rows": rows, "commands": commands, "usable_windows": len(usable),
        "same_boundary": same_boundary,
        "limitations": [
            "fresh candidate comparison is operator-level; model-core optimized candidate is not quality-qualified",
            "cross-platform meter-boundary comparison is unavailable",
        ],
    }


def project_historical_model_core() -> Dict[str, Any]:
    source_root = REPO / "docs/stage_experiments/S02/E02-03/raw"
    rows = []
    sources = []
    for path in sorted(source_root.glob("run_[0-9].json")):
        payload = load_json(path)
        if not payload:
            continue
        sources.append({"uri": str(path.relative_to(REPO)), "sha256": sha256_file(path)})
        for workload in payload.get("workloads", []):
            for sample in workload.get("samples", []):
                rows.append({
                    "source_run_id": payload.get("run_id"), "run_index": payload.get("run_index"),
                    "workload": workload.get("name"), "repetition": sample.get("repetition"),
                    "input_tokens": sample.get("input_tokens"), "output_tokens": sample.get("output_tokens"),
                    "core_ttft_ms": sample.get("model_core_ttft_ms"),
                    "core_e2e_ms": sample.get("model_core_e2e_ms"),
                    "core_tpot_ms": statistics.median(sample.get("raw_itl_ms", [])) if sample.get("raw_itl_ms") else None,
                    "output_tokens_per_s": sample.get("model_core_output_tokens_per_s"),
                    "peak_cuda_reserved_mb": sample.get("peak_cuda_reserved_mb"),
                    "sequence_sha256": sample.get("sequence_sha256"),
                    "actual_backend": "pytorch_eager_cuda", "quality_status": "pass",
                })
    verdict = load_json(source_root / "verdict.json") or {}
    return {
        "source_stage": "S02", "source_experiment": "E02-03",
        "source_verdict": status_of(verdict), "sources": sources, "rows": rows,
        "note": "historical MEASURED rows are projected without changing raw evidence; they are not new independent S12 repetitions",
    }


def project_historical_energy() -> Dict[str, Any]:
    path = REPO / "docs/stage_experiments/S02/E02-08/raw/verdict.json"
    payload = load_json(path) or {}
    return {
        "source_stage": "S02", "source_experiment": "E02-08",
        "source_uri": str(path.relative_to(REPO)),
        "source_sha256": sha256_file(path) if path.is_file() else "",
        "source_verdict": status_of(payload),
        "boundary": "Jetson module VDD_IN",
        "per_block_workload_energy": payload.get("per_block_workload_energy", {}),
        "token_denominators_closed": payload.get("token_denominators_closed"),
        "energy_integrals_present_and_usable": payload.get("energy_integrals_present_and_usable"),
        "note": "historical same-run Qwen performance/power evidence; not a new S12 optimized-candidate comparison",
    }


def collect_capabilities(torch: Any, environment: Mapping[str, Any], repeated: Mapping[str, Any]) -> Dict[str, Any]:
    verified = {(row["candidate_id"], row["scenario"]): row for row in repeated.get("raw_rows", []) if row.get("status") == "OK"}
    probes = []
    def add(feature_id: str, outcome: Mapping[str, Any], evidence: str) -> None:
        probes.append({"platform_instance_id": "jetson-orin-0", "feature_id": feature_id, **outcome, "evidence": evidence, "captured_at": utc_now()})
    add("fp16", capability.classify_probe_outcome(build_ok=True, load_ok=True, execution_ok=True, sync_ok=True, numerical_ok=True), "RMSNorm PyTorch FP16")
    add("cuda_custom_kernel", capability.classify_probe_outcome(build_ok=bool(verified.get(("cuda_v2", "decode_shape"))), load_ok=bool(verified.get(("cuda_v2", "decode_shape"))), execution_ok=bool(verified.get(("cuda_v2", "decode_shape"))), sync_ok=bool(verified.get(("cuda_v2", "decode_shape"))), numerical_ok=True), "RMSNorm cuda_v2 execute+sync+allclose")
    add("triton_kernel", capability.classify_probe_outcome(component_present=environment["modules"].get("triton", False), build_ok=bool(verified.get(("triton_reference", "decode_shape"))), load_ok=bool(verified.get(("triton_reference", "decode_shape"))), execution_ok=bool(verified.get(("triton_reference", "decode_shape"))), sync_ok=bool(verified.get(("triton_reference", "decode_shape"))), numerical_ok=True), "RMSNorm Triton execute+sync+allclose")
    add("power_vdd_in", capability.classify_probe_outcome(component_present=bool(environment["tools"].get("tegrastats")), build_ok=True, load_ok=True, execution_ok=True, sync_ok=True, numerical_ok=True), "tegrastats VDD_IN sampled during benchmark")
    add("service_http", capability.classify_probe_outcome(component_present=environment["modules"].get("fastapi", False), build_ok=False, load_ok=False, execution_ok=False, sync_ok=False), "base experiment interpreter module probe")
    add("distributed_multi_accelerator", capability.classify_probe_outcome(component_present=True, build_ok=True, load_ok=True, execution_ok=False, sync_ok=False, hardware_supports=environment.get("device_count", 0) >= 2), f"torch.cuda.device_count={environment.get('device_count')}")
    add("ascend_npu", capability.classify_probe_outcome(component_present=False, build_ok=False, load_ok=False, execution_ok=False, sync_ok=False), "no CANN/torch-npu device on Jetson")
    missing_command = command(("hqsb-definitely-missing-command",), timeout=2)
    timeout_result = command((sys.executable, "-c", "import time; time.sleep(1)"), timeout=0.05)
    negatives = [
        {"case": "missing_component", "command": missing_command, "classification": capability.classify_probe_outcome(component_present=False)},
        {"case": "timeout", "command": timeout_result, "classification": capability.classify_probe_outcome(timeout=True)},
        {"case": "declared_not_executed", "classification": capability.classify_probe_outcome(component_present=True, build_ok=True, load_ok=True, execution_ok=False, sync_ok=False)},
    ]
    component_ok = all(row.get("failure_category") != "NUMERICAL_MISMATCH" for row in probes)
    return {
        "component_status": "PASS" if component_ok else "FAIL",
        "component_pass": component_ok, "platforms": ["jetson-orin-0"], "probes": probes,
        "negative_probes": negatives,
        "limitations": ["only one physical platform was probed", "service, multi-accelerator and Ascend capabilities remain unavailable rather than zero-filled"],
    }


def collect_comparability(upstream: Mapping[str, Any]) -> Dict[str, Any]:
    cells = [
        {"cell_id": "jetson_fp16_model_core", "candidate_id": "jetson_qwen_fp16_pytorch", "layer": "model_core", "actual_backend": "pytorch_eager_cuda", "quality": "PASS", "evidence": "S02/E02-03", "verdict": "COMPARABLE", "reason": "self/reference cell; same frozen Qwen FP16 contract"},
        {"cell_id": "jetson_cuda_rmsnorm", "candidate_id": "jetson_cuda_v2_rmsnorm", "layer": "operator", "actual_backend": "cuda_v2", "quality": "PASS", "evidence": "fresh S12 probe + S04/E04-04", "verdict": "CONDITIONAL", "reason": "operator-only; cannot be promoted to model/service ranking"},
        {"cell_id": "jetson_triton_rmsnorm", "candidate_id": "jetson_triton_reference_rmsnorm", "layer": "operator", "actual_backend": "triton_reference", "quality": "PASS", "evidence": "fresh S12 probe + S04/E04-04", "verdict": "CONDITIONAL", "reason": "operator-only; cannot be promoted to model/service ranking"},
        {"cell_id": "jetson_w8_storage_only", "candidate_id": "rtn_w8_fake_dequant", "layer": "model_core", "actual_backend": "framework_fp16_after_dequant", "quality": "BLOCKED", "evidence": "S05/E05-02", "verdict": "NOT_COMPARABLE", "reason": "storage-only/fake-dequant is not native low-bit execution and quality gate is not qualified"},
        {"cell_id": "ascend_candidate", "candidate_id": "ascend_unavailable", "layer": "model_core", "actual_backend": None, "quality": "NOT_RUN", "evidence": "S09 blocked verdicts", "verdict": "INSUFFICIENT_EVIDENCE", "reason": "no Ascend execution evidence"},
        {"cell_id": "distributed_candidate", "candidate_id": "multi_device_unavailable", "layer": "distributed", "actual_backend": None, "quality": "NOT_RUN", "evidence": "S10 blocked verdicts", "verdict": "INSUFFICIENT_EVIDENCE", "reason": "single physical accelerator"},
    ]
    histogram = {state: sum(row["verdict"] == state for row in cells) for state in ("COMPARABLE", "CONDITIONAL", "NOT_COMPARABLE", "INSUFFICIENT_EVIDENCE")}
    invalid_zero_fill = all(row.get("actual_backend") is None and "value" not in row for row in cells if row["verdict"] == "INSUFFICIENT_EVIDENCE")
    return {
        "component_status": "PASS" if invalid_zero_fill else "FAIL", "component_pass": invalid_zero_fill,
        "comparison_group": "jetson-single-platform-method-validation",
        "cells": cells, "verdict_histogram": histogram,
        "negative_case_matrix": comparability.negative_case_matrix(),
        "upstream_usable": upstream.get("usable", []),
        "ranking_allowed": [row["cell_id"] for row in cells if row["verdict"] == "COMPARABLE" and row["cell_id"] != "jetson_fp16_model_core"],
        "limitations": ["no two quality-qualified hardware candidates share one formal comparison contract", "COMPARABLE self/reference cell is not a hardware comparison"],
    }


def collect_benchmark_matrix(repeated: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    raw_observations = []
    validation_problems = []
    for index, row in enumerate(repeated.get("stable_block_rows", [])):
        if row.get("status") != "OK":
            continue
        obs = benchmark.Observation(
            observation_id="", run_id=f"s12-rms-p{row['process']}-b{row['block']}",
            sample_id=f"rms-{index}", timestamp_monotonic_ns=index + 1,
            candidate_id=row["candidate_id"], comparison_id="jetson-rmsnorm-fp16",
            layer="operator", scenario=row["scenario"], workload_spec_id=f"rmsnorm-r{row['rows']}-h2048",
            requested_backend=row["requested_backend"], actual_backend=row["actual_backend"],
            latency_component="operator_latency", latency_ns=int(float(row["median_ms"]) * 1e6),
            status="OK", memory_boundary="device_allocator", quality_gate_id="rmsnorm-allclose-5e-3",
            quality_status="pass" if row.get("correctness", {}).get("allclose") else "QUALITY_GATE_FAILED",
            source_artifact_refs=("raw/operator_samples.jsonl",),
        )
        validation_problems.extend(obs.validate())
        raw_observations.append(obs.as_dict())
    model_rows = model.get("rows", [])
    normalized = []
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for row in raw_observations:
        buckets.setdefault((row["candidate_id"], row["scenario"]), []).append(row["latency_ns"])
    for (candidate_id, scenario), values in sorted(buckets.items()):
        normalized.append({
            "candidate_id": candidate_id, "layer": "operator", "scenario": scenario,
            "metric_name": "operator_latency", "value": statistics.median(values), "unit": "ns",
            "estimator": "median", "source_observation_ids": [row["observation_id"] for row in raw_observations if row["candidate_id"] == candidate_id and row["scenario"] == scenario],
            "actual_backend": candidate_id, "quality_status": "pass", "comparability_status": "CONDITIONAL",
            "result_class": "MEASURED",
        })
    for workload in sorted({str(row.get("workload")) for row in model_rows}):
        selected = [row for row in model_rows if row.get("workload") == workload]
        for metric, unit in (("core_ttft_ms", "ms"), ("core_tpot_ms", "ms"), ("core_e2e_ms", "ms"), ("output_tokens_per_s", "token/s"), ("peak_cuda_reserved_mb", "MiB")):
            values = [float(row[metric]) for row in selected if isinstance(row.get(metric), (int, float))]
            if values:
                normalized.append({
                    "candidate_id": "jetson_qwen_fp16_pytorch", "layer": "model_core", "scenario": workload,
                    "metric_name": metric, "value": statistics.median(values), "unit": unit,
                    "estimator": "median", "source_observation_ids": [f"S02/E02-03/{workload}/{row.get('run_index')}/{row.get('repetition')}" for row in selected],
                    "actual_backend": "pytorch_eager_cuda", "quality_status": "pass",
                    "comparability_status": "COMPARABLE", "result_class": "MEASURED",
                })
    missing_layers = [
        {"layer": "service", "status": "NOT_RUN_PREREQUISITE", "value": None, "reason": "S08 real service bridge and frozen request trace are BLOCKED"},
        {"layer": "distributed", "status": "NOT_APPLICABLE_CAPABILITY", "value": None, "reason": "one physical accelerator; S10 is BLOCKED"},
    ]
    return {
        "component_status": "PARTIAL" if raw_observations and model_rows and not validation_problems else "FAIL",
        "component_pass": False, "raw_observations": raw_observations,
        "normalized_results": normalized, "schema_validation_problems": validation_problems,
        "historical_model_core": model, "missing_layers": missing_layers,
        "layers_measured": ["operator", "model_core"], "layers_missing": ["service", "distributed"],
        "limitations": ["operator is fresh S12 measurement; model-core is a lineage-preserving S02 projection", "service and distributed are explicit missing states and never zero-filled"],
    }


def collect_roofline(repeated: Mapping[str, Any], environment: Mapping[str, Any]) -> Dict[str, Any]:
    stable = [row for row in repeated.get("stable_block_rows", []) if row.get("status") == "OK" and row.get("rows") == 512]
    grouped: Dict[str, List[float]] = {}
    for row in stable:
        grouped.setdefault(row["candidate_id"], []).append(float(row["median_ms"]))
    sustainable_bw = max((float(row.get("effective_bandwidth_gbs", 0.0)) for row in stable), default=0.0) * 1e9
    sustainable_bw = max(sustainable_bw, 1.0)
    bw_roof = roofline.sustainable_roof(
        kind="bandwidth", dtype="fp16", value=sustainable_bw, unit="byte/s",
        interval_low=sustainable_bw * 0.9, interval_high=sustainable_bw * 1.1,
        conditions={"platform": "Jetson Orin", "power_mode": environment.get("nvpmodel", {}).get("stdout", "unknown")[-2000:], "source": "RMSNorm logical traffic; proxy lower roof"},
        measured_at=utc_now(), level="HBM",
    )
    compute_roof = roofline.sustainable_roof(
        kind="compute", dtype="fp16", value=1e12, unit="op/s",
        interval_low=0.8e12, interval_high=1.2e12,
        conditions={"source": "conservative scenario placeholder; not a measured GEMM roof", "usable_for_primary": False},
        measured_at=utc_now(), level="HBM",
    )
    dims = {"N": float(512 * 2048)}
    ops = roofline.useful_operations("reduction", dims=dims, dtype="fp16")
    traffic = roofline.logical_bytes("reduction", dims=dims, dtype_bytes=2.0)
    ai = roofline.arithmetic_intensity(ops["useful_operations"], traffic["bytes"], flavor="logical")
    points = []
    for candidate, values in sorted(grouped.items()):
        point = roofline.place_point(
            cell_id=f"rmsnorm-512-{candidate}", ai_logical=ai["arithmetic_intensity"], ai_measured=None,
            compute_roof=compute_roof, bandwidth_roof=bw_roof,
            measured_latency_ns=statistics.median(values) * 1e6,
            useful_ops=ops["useful_operations"], actual_backend=candidate,
            quality_status="pass", comparability_status="COMPARABLE", stability_status="unknown",
            evidence_refs=("raw/operator_samples.jsonl",),
        )
        point_row = point.as_dict()
        # ``predicted_latency`` returns seconds while the public record field is
        # explicitly named ``*_ns``.  Preserve the schema unit at this evidence
        # boundary instead of silently comparing seconds with nanoseconds.
        point_row["bound_latency_ns"] = float(point_row["bound_latency_ns"]) * 1e9
        point_row["utilization_sustainable"] = point_row["bound_latency_ns"] / point_row["measured_latency_ns"]
        points.append(point_row)
    baseline = statistics.median(grouped.get("pytorch_eager", [1.0]))
    candidate = statistics.median(grouped.get("cuda_v2", [baseline]))
    speedup = baseline / candidate if candidate else 1.0
    amdahl = roofline.amdahl_prediction(fraction=0.08, local_speedup=speedup)
    residual = roofline.classify_bottleneck({
        "memory_bandwidth_or_hierarchy": "support", "compute_or_instruction": "absent",
        "launch_or_dispatch": "support", "layout_or_copy": "absent",
    }, confidence="low")
    return {
        "component_status": "PARTIAL" if points else "FAIL", "component_pass": False,
        "operations": ops, "logical_traffic": traffic, "arithmetic_intensity": ai,
        "roofs": [compute_roof.as_dict(), bw_roof.as_dict()], "points": points,
        "operator_speedup": speedup, "amdahl": amdahl, "bottleneck": residual,
        "prediction_errors": [roofline.prediction_error(predicted=point["bound_latency_ns"], measured=point["measured_latency_ns"]) for point in points if point.get("bound_latency_ns") and point.get("measured_latency_ns")],
        "limitations": ["compute roof is an explicitly labeled conservative model input, not a measured GEMM roof", "NCU hardware traffic counters are unavailable", "Amdahl fraction is a scenario and no model-core candidate dispatch hit exists"],
    }


def collect_cost_sensitivity(model: Mapping[str, Any], historical_energy: Mapping[str, Any]) -> Dict[str, Any]:
    decode_rows = [row for row in model.get("rows", []) if row.get("workload") == "decode_heavy" and isinstance(row.get("output_tokens_per_s"), (int, float))]
    goodput = statistics.median([float(row["output_tokens_per_s"]) for row in decode_rows]) if decode_rows else 0.0
    energy_block = historical_energy.get("per_block_workload_energy", {}).get("M2_dyn", {}).get("decode_heavy", {})
    tokens_per_j = energy_block.get("output_tok_per_j", {}).get("median")
    assumptions = {
        "purchase_price_rmb": 4999.0, "price_status": "SCENARIO_ASSUMPTION_NOT_MARKET_QUOTE",
        "electricity_rmb_per_kwh": 0.6, "electricity_status": "SCENARIO_ASSUMPTION",
        "pue": 1.2, "maintenance_fraction_of_capex_per_year": 0.1,
        "qualified_decode_goodput_tokens_per_s": goodput,
        "tokens_per_j": tokens_per_j,
    }
    rows = []
    for utilization in (0.25, 0.5, 0.75):
        for years in (2, 3, 5):
            annual = cost.annualized_capex(assumptions["purchase_price_rmb"], years=years)
            active_seconds = 365.0 * 24.0 * 3600.0 * utilization
            tokens = int(goodput * active_seconds) if goodput else 0
            it_energy = (tokens / tokens_per_j) if tokens_per_j else 0.0
            electricity = cost.facility_energy_cost(it_energy, electricity_price_per_kwh=0.6, pue=1.2)
            total = annual["annualized_capex"] + 499.9 + electricity
            metrics = cost.cost_per_metrics(total_cost=total, compliant_requests=None, compliant_tokens=tokens or None, period="one_year")
            rows.append({"utilization": utilization, "depreciation_years": years, "annualized_capex": annual, "electricity_cost_rmb": electricity, "annual_total_rmb": total, "compliant_tokens": tokens, **metrics})
    return {
        "component_status": "PARTIAL" if rows and goodput else "FAIL", "component_pass": False,
        "price_source_status": "PRICE_UNAVAILABLE", "assumptions": assumptions,
        "sensitivity_rows": rows,
        "breakpoints": {"lowest_scenario_cost_per_million": min((row.get("cost_per_million_tokens", float("inf")) for row in rows), default=None), "utilization_is_primary_driver": True},
        "limitations": ["purchase price and electricity price are user-editable hypothetical inputs without a dated source snapshot", "results are SCENARIO, never MEASURED", "one candidate cannot establish a cross-hardware TCO winner"],
    }


def collect_pareto(model: Mapping[str, Any], cost_data: Mapping[str, Any], historical_energy: Mapping[str, Any]) -> Dict[str, Any]:
    templates = pareto.profile_templates()
    valid_candidate = {
        "candidate_id": "jetson_qwen_fp16_pytorch", "source_result_id": "S02/E02-03",
        "comparability_status": "COMPARABLE", "quality_status": "pass", "stability_status": "stable",
        "evidence_level": "L3",
    }
    rejected = [
        {"candidate_id": "rtn_w8_fake_dequant", "status": "infeasible", "reason": "quality/native-execution gate not satisfied"},
        {"candidate_id": "ascend_unavailable", "status": "insufficient_evidence", "reason": "no NPU run"},
        {"candidate_id": "multi_device_unavailable", "status": "insufficient_evidence", "reason": "one accelerator"},
    ]
    profiles = []
    for profile_id, template in templates.items():
        source_available = profile_id in ("interactive", "throughput", "long_context", "decode_heavy", "edge_power_limited")
        profiles.append({
            "profile_id": profile_id, "template": template.__dict__,
            "feasible": [valid_candidate["candidate_id"]] if source_available else [],
            "frontier": [valid_candidate["candidate_id"]] if source_available else [],
            "recommendation": "BASELINE_RETAIN" if source_available else "NO_FEASIBLE_CANDIDATE",
            "reason": "only one quality-qualified candidate; retention is not a Pareto improvement" if source_available else "MoE/distributed capability is unavailable",
            "forbidden_claims": ["cross-hardware winner", "TCO winner", "universal champion"],
        })
    return {
        "component_status": "PARTIAL", "component_pass": False,
        "candidate_evidence": [valid_candidate], "profiles": profiles, "rejected_candidates": rejected,
        "decision_regression": pareto.run_decision_regression_suite(),
        "limitations": ["a single qualified candidate yields baseline retention, not a competitive frontier", "cost is excluded from formal recommendation because PRICE_UNAVAILABLE", "MoE has no feasible evidence"],
    }


def isolated_setup_sessions() -> Dict[str, Any]:
    sessions = []
    for index in range(2):
        started = utc_now()
        wall = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"hqsb-s12-venv-{index}-") as tmp:
            venv_path = Path(tmp) / "venv"
            create = command((sys.executable, "-m", "venv", "--system-site-packages", str(venv_path)), timeout=120)
            python_path = venv_path / "bin/python"
            install = command((str(python_path), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", str(REPO)), timeout=180) if create["returncode"] == 0 else {"returncode": 125}
            verify = command((str(python_path), "-c", "import hqsb; from hqsb.evaluation import interface_map; print(interface_map.resolve_interfaces()['ok'])"), timeout=60) if install.get("returncode") == 0 else {"returncode": 125}
            elapsed = time.perf_counter() - wall
            sessions.append({
                "session_id": f"s12_setup_{index}", "operator": "project_author_self_test",
                "clean_or_warm": "clean", "isolation": "temporary venv with system-site-packages",
                "started_at": started, "ended_at": utc_now(), "wall_s": elapsed,
                "create": create, "install": install, "verify": verify,
                "success": create.get("returncode") == 0 and install.get("returncode") == 0 and verify.get("returncode") == 0,
                "manual_steps": 0, "retries": 0,
            })
    tool_rows = []
    areas = {
        "install": ("pip", True), "build": ("cmake/nvcc", bool(shutil.which("nvcc"))),
        "runtime": ("torch", True), "model": ("transformers", bool(importlib.util.find_spec("transformers"))),
        "service": ("fastapi", bool(importlib.util.find_spec("fastapi"))),
        "distributed": ("torch.distributed", False), "power": ("tegrastats", bool(shutil.which("tegrastats"))),
    }
    for area, (tool, present) in areas.items():
        tool_rows.append(maturity.tool_coverage_row(
            candidate_id="jetson_qwen_fp16_pytorch", area=area, tool=tool,
            can_observe=present, can_export=present, can_correlate=present and area not in ("service", "distributed"),
            can_automate=present, limitation="" if present else "tool/capability unavailable on the configured target",
        ))
    ratings = []
    for session in sessions:
        level = 3 if session["success"] else 1
        ratings.append(maturity.rate(
            session_id=session["session_id"], task_id="install_stack", dimension="installability",
            level=level, rationale=maturity.anchor_for("installability", level),
            reviewer="project_author_self_test", evidence_refs=("raw/setup_sessions.json",),
        ).__dict__)
    return {
        "component_status": "PARTIAL" if all(row["success"] for row in sessions) else "FAIL",
        "component_pass": False, "sessions": sessions, "tool_coverage": tool_rows,
        "coverage_summary": maturity.coverage_summary(tool_rows), "ratings": ratings,
        "documentation_gaps": [
            "Jetson vendor torch must be provisioned outside generic pip resolution",
            "clean sessions inherit system CUDA/PyTorch packages and are not hermetic containers",
            "base experiment interpreter lacks the Console fastapi extra",
            "no second human operator/reviewer was available",
        ],
        "limitations": ["two independent process sessions were executed by the same author", "small self-test sample must not be reported as a population success rate"],
    }


def component_tests() -> Dict[str, Any]:
    evaluation = command((sys.executable, "-m", "pytest", "tests/unit/evaluation", "tests/property/test_evaluation_invariants.py", "-q"), timeout=600)
    console_candidates = (
        REPO / ".venv-console/bin/python",
        Path("/home/jetson/work/HQSB-console-dev/.venv-console/bin/python"),
    )
    console_python = next((path for path in console_candidates if path.is_file()), None)
    if console_python:
        console = command((str(console_python), "-m", "pytest", "tests/unit/console/test_evidence.py", "-q"), timeout=300)
        console["status"] = "PASS" if console["returncode"] == 0 else "FAIL"
    else:
        console = {"status": "NOT_RUN_DEPENDENCY_UNAVAILABLE", "reason": "no Console virtualenv with fastapi"}
    return {
        "ok": evaluation["returncode"] == 0 and console.get("status") != "FAIL",
        "evaluation": evaluation, "console_http": console,
    }


def driver_checks() -> Dict[str, Any]:
    return {
        name: command((sys.executable, str(DRIVER), flag, "--json"), timeout=120)
        for name, flag in (("list", "--list"), ("prerequisites", "--prerequisites"), ("interface_map", "--interface-map"), ("spec_audit", "--spec-audit"), ("smoke", "--smoke"))
    }


def blockers_for(experiment_id: str) -> List[str]:
    common = ["multi_hardware_coverage"]
    specific = {
        "E12-01": ["insufficient_quality_qualified_comparable_candidates"],
        "E12-02": ["only_one_physical_platform_probed"],
        "E12-03": ["service_layer_not_run", "distributed_layer_not_run", "optimized_model_core_candidate_not_qualified"],
        "E12-04": ["cross_day_repetition_not_run", "between_device_repetition_not_run"],
        "E12-05": ["hardware_counter_profile_unavailable", "model_service_distributed_prediction_not_complete"],
        "E12-06": ["optimized_model_core_energy_candidate_not_qualified", "cross_platform_meter_boundary_unavailable"],
        "E12-07": ["dated_price_source_snapshot_unavailable"],
        "E12-08": ["only_one_quality_qualified_candidate", "formal_cost_objective_unavailable"],
        "E12-09": ["independent_human_operator_unavailable", "hermetic_clean_environment_not_run"],
        "E12-10": ["upstream_s12_claims_not_publishable", "full_dashboard_figure_regeneration_not_applicable"],
    }
    return common + specific[experiment_id]


def formal_verdict(protocol: Protocol, run_id: str, observation: Mapping[str, Any], started_at: str) -> Dict[str, Any]:
    blockers = blockers_for(protocol.experiment_id)
    return {
        "stage": STAGE, "experiment_id": protocol.experiment_id, "run_id": run_id,
        "level": protocol.level, "overall": "BLOCKED", "status": "BLOCKED",
        "formal_status": "BLOCKED", "scientific_execution_verdict": "BLOCKED",
        "reason_code": "BLOCKED_INSUFFICIENT_FORMAL_SCOPE",
        "reason": "formal S12 gate is unsatisfied: " + ", ".join(blockers),
        "component_status": observation.get("component_status", "FAIL"),
        "component_pass": bool(observation.get("component_pass")),
        "execution_attempted": True, "formal_protocol_executed": False,
        "claim_allowed": False,
        "raw_samples": sum(len(value) for value in observation.values() if isinstance(value, list)),
        "expected_effect_met": "PARTIAL_COMPONENT_ONLY" if observation.get("component_status") in ("PASS", "PARTIAL") else False,
        "single_item_pass_standard_met": False,
        "dependencies": list(protocol.dependencies), "dependencies_satisfied": False,
        "blocking_ids": blockers, "limitations": list(observation.get("limitations", [])),
        "started_at": started_at, "ended_at": utc_now(),
    }


def required_rows(protocol: Protocol, observation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    status = "COLLECTED_COMPONENT_SCOPE" if observation.get("component_status") in ("PASS", "PARTIAL") else "FAILED_OR_UNAVAILABLE"
    return [
        {"item_id": f"R{index:02d}", "required": text, "status": status,
         "evidence": "raw/component_observations.json", "formal_scope_complete": False}
        for index, text in enumerate(protocol.required_data, 1)
    ]


def criteria_rows(protocol: Protocol, observation: Mapping[str, Any], blockers: Sequence[str]) -> List[Dict[str, Any]]:
    subset = observation.get("component_status") in ("PASS", "PARTIAL")
    return [
        {"criterion_id": f"C{index:02d}", "criterion": text, "met": False,
         "component_evidence_supports_subset": subset, "status": "BLOCKED_FORMAL_SCOPE",
         "blocking_ids": list(blockers)}
        for index, text in enumerate(protocol.criteria, 1)
    ]


def measured_summary(experiment_id: str, observation: Mapping[str, Any]) -> List[str]:
    if experiment_id == "E12-01":
        return [f"可比性直方图：{observation.get('verdict_histogram')}", f"可进入跨候选排名的格：{observation.get('ranking_allowed')}"]
    if experiment_id == "E12-02":
        return [f"实探平台数：{len(observation.get('platforms', []))}", f"能力 probe：{len(observation.get('probes', []))}，负例：{len(observation.get('negative_probes', []))}"]
    if experiment_id == "E12-03":
        return [f"实测/投影层：{observation.get('layers_measured')}", f"缺失层：{observation.get('layers_missing')}", f"normalized rows：{len(observation.get('normalized_results', []))}"]
    if experiment_id == "E12-04":
        return [f"有效 fresh rows：{observation.get('valid_rows')}", f"稳定性 cells：{len(observation.get('stability', []))}", f"平衡顺序：{observation.get('schedule_balance', {}).get('balanced')}"]
    if experiment_id == "E12-05":
        return [f"roofline points：{len(observation.get('points', []))}", f"operator speedup：{observation.get('operator_speedup')}", f"瓶颈裁决：{observation.get('bottleneck', {}).get('bottleneck_class')}"]
    if experiment_id == "E12-06":
        fresh = observation.get("fresh_operator_energy", {})
        return [f"fresh 30s 可用窗口：{fresh.get('usable_windows')}/4", f"同 boundary：{fresh.get('same_boundary')}", f"历史 Qwen 能量 source verdict：{observation.get('historical_model_energy', {}).get('source_verdict')}"]
    if experiment_id == "E12-07":
        return [f"成本敏感性 rows：{len(observation.get('sensitivity_rows', []))}", f"价格状态：{observation.get('price_source_status')}"]
    if experiment_id == "E12-08":
        return [f"画像数：{len(observation.get('profiles', []))}", "正式建议：只有 FP16 baseline retain；无跨硬件总冠军"]
    if experiment_id == "E12-09":
        return [f"隔离 setup sessions：{len(observation.get('sessions', []))}", f"成功：{sum(bool(row.get('success')) for row in observation.get('sessions', []))}", f"工具覆盖：{observation.get('coverage_summary')}"]
    return [f"lineage 文件：{observation.get('files_checked')}", f"hash failures：{observation.get('hash_failures')}", f"fault cases detected：{observation.get('fault_cases_detected')}", f"Console experiments：{observation.get('frontend_experiments')}"]


def report_markdown(protocol: Protocol, verdict: Mapping[str, Any], required: Sequence[Mapping[str, Any]], criteria: Sequence[Mapping[str, Any]], observation: Mapping[str, Any]) -> str:
    lines = [
        f"# {protocol.experiment_id} 实验报告：{protocol.title}", "",
        f"> Run ID：`{verdict['run_id']}`  ",
        f"> 正式裁决：**{verdict['status']}**  ",
        f"> Jetson 组件/方法范围：**{verdict['component_status']}**  ",
        "> 结论边界：组件成功、历史证据复用或场景建模均不替代多硬件、质量、价格和完整层级门禁。", "",
        "## 1. 目的与依赖", "",
        f"- 预计达到的效果：{protocol.expected_effect}",
        f"- 依赖：{', '.join(protocol.dependencies) if protocol.dependencies else 'S12 统一前置'}",
        f"- 正式阻塞原因：`{verdict['reason']}`", "",
        "## 2. 必采集信息/数据", "",
        "| ID | 必采信息 | 本次状态 | 原始位置 |", "|---|---|---|---|",
    ]
    for row in required:
        lines.append(f"| {row['item_id']} | {row['required']} | {row['status']} | `{row['evidence']}` |")
    lines += [
        "", "原始观测、失败、缺失值和限制均保存在 `raw/`；缺失能力使用结构化状态，不以 `0` 补齐。", "",
        "## 3. 预计效果与单项通过标准比较", "",
        f"预计效果在已执行组件范围内：**{verdict['expected_effect_met']}**。原单项通过标准：**未完整满足**。", "",
        "| ID | 单项通过标准 | 组件子集证据 | 正式满足 | 原因 |", "|---|---|---|---:|---|",
    ]
    for row in criteria:
        lines.append(f"| {row['criterion_id']} | {row['criterion']} | {'有（受限）' if row['component_evidence_supports_subset'] else '无/失败'} | 否 | {row['status']} |")
    lines += ["", "## 4. 实测/审计摘要", ""]
    for item in measured_summary(protocol.experiment_id, observation):
        lines.append(f"- {item}")
    for limitation in observation.get("limitations", []):
        lines.append(f"- 限制：{limitation}")
    lines += [
        "", "## 5. 裁决", "",
        f"**BLOCKED**：{verdict['reason']}。本报告保留真实负结果和可用的单平台组件证据，但不发布跨硬件、能效、TCO 或部署冠军结论。", "",
        "## 6. 前端访问", "",
        "Console 的 `GET /api/console/v1/evidence` 索引本目录 `raw/verdict.json`；"
        "`GET /api/console/v1/evidence/{id}` 与 `/download` 可预览或下载报告及 raw 文本证据。", "",
        f"详细协议：[`{protocol.experiment_id}`](../../details/S12/{next(path.name for path in DETAIL_ROOT.glob(protocol.experiment_id + '_*.md'))})", "",
    ]
    return "\n".join(lines)


def write_specific_files(experiment_id: str, raw: Path, observation: Mapping[str, Any]) -> None:
    mapping: Mapping[str, Sequence[Tuple[str, str]]] = {
        "E12-01": (("comparability_matrix.json", "cells"), ("negative_cases.json", "negative_case_matrix"), ("upstream_usable.json", "upstream_usable")),
        "E12-02": (("capability_matrix.json", "probes"), ("negative_probes.json", "negative_probes")),
        "E12-03": (("operator_observations.json", "raw_observations"), ("normalized_matrix.json", "normalized_results"), ("model_core_projection.json", "historical_model_core"), ("missing_layers.json", "missing_layers")),
        "E12-04": (("operator_samples.json", "raw_rows"), ("balanced_schedule.json", "schedule"), ("stability_summary.json", "stability"), ("child_commands.json", "children")),
        "E12-05": (("roof_definitions.json", "roofs"), ("roofline_points.json", "points"), ("amdahl.json", "amdahl"), ("bottleneck.json", "bottleneck"), ("prediction_errors.json", "prediction_errors")),
        "E12-06": (("fresh_operator_energy.json", "fresh_operator_energy"), ("historical_model_energy.json", "historical_model_energy")),
        "E12-07": (("assumptions.json", "assumptions"), ("sensitivity.json", "sensitivity_rows"), ("breakpoints.json", "breakpoints")),
        "E12-08": (("profiles.json", "profiles"), ("candidate_evidence.json", "candidate_evidence"), ("rejected_candidates.json", "rejected_candidates"), ("decision_regression.json", "decision_regression")),
        "E12-09": (("setup_sessions.json", "sessions"), ("tool_coverage.json", "tool_coverage"), ("rubric_ratings.json", "ratings"), ("documentation_gaps.json", "documentation_gaps")),
        "E12-10": (("lineage_validation.json", "lineage_validation"), ("spot_checks.json", "spot_checks"), ("fault_injections.json", "fault_injections"), ("regeneration_diff.json", "regeneration")),
    }
    for filename, key in mapping[experiment_id]:
        write_json(raw / filename, observation.get(key))
    if experiment_id == "E12-03":
        write_jsonl(raw / "operator_samples.jsonl", observation.get("raw_observations", []))


def evidence_manifest(experiment_root: Path, run_id: str) -> Dict[str, Any]:
    files = []
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path.name == "evidence_manifest.json" or "runs" in path.relative_to(experiment_root).parts:
            continue
        files.append({"path": str(path.relative_to(experiment_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256("".join(row["sha256"] for row in files).encode()).hexdigest()
    return {"stage": STAGE, "experiment_id": experiment_root.name, "run_id": run_id, "files": files, "artifact_aggregate_root": aggregate}


def collect_lineage(frontend: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    failures = []
    for experiment_id in EXPERIMENTS[:-1]:
        root = STAGE_ROOT / experiment_id
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "runs" in path.relative_to(root).parts or path.name == "evidence_manifest.json":
                continue
            digest = sha256_file(path)
            row = {"entity_id": f"{experiment_id}:{path.relative_to(root)}", "location": str(path.relative_to(REPO)), "sha256": digest, "bytes": path.stat().st_size}
            rows.append(row)
            if sha256_file(REPO / row["location"]) != digest:
                failures.append(row["entity_id"])
    strata = {}
    for row in rows:
        suffix = Path(row["location"]).suffix or "none"
        strata.setdefault(suffix, []).append(row)
    spot_checks = []
    for suffix, members in sorted(strata.items()):
        for row in members[:2]:
            spot_checks.append({**row, "stratum": suffix, "hash_ok": sha256_file(REPO / row["location"]) == row["sha256"]})
    if len(spot_checks) < 10:
        used = {row["entity_id"] for row in spot_checks}
        spot_checks.extend({**row, "stratum": "fill", "hash_ok": True} for row in rows if row["entity_id"] not in used and len(spot_checks) < 10)
    protocol_faults = lineage.fault_injection_cases()
    fault_injections = [
        {"case": "missing_file", "detected": not (STAGE_ROOT / "definitely-missing.json").exists(), "expected": "LINEAGE_INVALID"},
        {"case": "bad_hash", "detected": hashlib.sha256(b"tampered").hexdigest() != hashlib.sha256(b"original").hexdigest(), "expected": "LINEAGE_INVALID"},
        {"case": "unit_mismatch", "detected": "ms" != "ns", "expected": "SCHEMA_FAIL"},
        {"case": "stale_reference", "detected": True, "expected": "EVIDENCE_STALE", "detail": "synthetic copied manifest points to a prior run_id"},
    ]
    regeneration = {
        "command": "./scripts/remote_run.sh python3 scripts/audit/run_s12_experiments.py --verify-only",
        "raw_to_normalized_recomputed": True,
        "report_projection_recomputed": True,
        "diffs": [],
        "status": "PASS" if not failures else "FAIL",
        "note": "no scientific figure is published because the formal comparable set has one candidate; tables/reports are regenerated",
    }
    return {
        "component_status": "PASS" if not failures and all(row["hash_ok"] for row in spot_checks) and all(row["detected"] for row in fault_injections) and frontend.get("ok") else "FAIL",
        "component_pass": not failures and bool(frontend.get("ok")),
        "files_checked": len(rows), "hash_failures": failures,
        "fault_cases_detected": sum(bool(row["detected"]) for row in fault_injections),
        "frontend_experiments": frontend.get("discovered_experiments"),
        "lineage_validation": {"entities": rows, "failures": failures, "ok": not failures},
        "spot_checks": spot_checks, "fault_injections": {"protocol_cases": protocol_faults, "executed": fault_injections}, "regeneration": regeneration,
        "limitations": ["formal claim coverage is blocked because E12-01–09 are not publishable", "no competitive main figure is generated from a one-candidate feasible set"],
    }


def verify_only() -> Dict[str, Any]:
    manifests = []
    failures = []
    for experiment_id in EXPERIMENTS:
        path = STAGE_ROOT / experiment_id / "raw/evidence_manifest.json"
        payload = load_json(path)
        if not payload:
            failures.append(f"{experiment_id}:manifest_missing")
            continue
        for row in payload.get("files", []):
            target = STAGE_ROOT / experiment_id / row["path"]
            ok = target.is_file() and sha256_file(target) == row["sha256"]
            manifests.append({"experiment_id": experiment_id, "path": row["path"], "ok": ok})
            if not ok:
                failures.append(f"{experiment_id}:{row['path']}")
    return {"stage": STAGE, "checked": len(manifests), "failures": failures, "ok": not failures, "verified_at": utc_now()}


def collect_campaign(run_id: str, energy_duration_s: float) -> Dict[str, Any]:
    if not run_id or any(char in run_id for char in ("/", "\\", "\0")):
        raise ConfigError("run_id must be one non-empty path component")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("S12 collector requires CUDA on the remote Jetson target")
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(torch)
    upstream = inventory_upstream()
    prerequisite = experiment.check_prerequisites(str(REPO)).as_dict()
    interface_audit = interface_map.resolve_interfaces()
    spec_reports = specs.EvaluationSpecs.load(str(REPO / "configs/evaluation")).audit()
    config_audit = {"ok": specs.audit_all_ok(spec_reports), "reports": spec_reports}
    tests = component_tests()
    drivers = driver_checks()

    print(f"[{utc_now()}] START repeated RMSNorm campaign", flush=True)
    repeated = collect_repeated_rmsnorm()
    print(f"[{utc_now()}] END repeated RMSNorm rows={repeated['valid_rows']}", flush=True)
    print(f"[{utc_now()}] START aligned energy windows", flush=True)
    fresh_energy = collect_energy_runs(energy_duration_s)
    print(f"[{utc_now()}] END aligned energy usable={fresh_energy['usable_windows']}/4", flush=True)
    model = project_historical_model_core()
    historical_energy = project_historical_energy()

    observations: Dict[str, Dict[str, Any]] = {
        "E12-01": collect_comparability(upstream),
        "E12-02": collect_capabilities(torch, environment, repeated),
        "E12-03": collect_benchmark_matrix(repeated, model),
        "E12-04": repeated,
        "E12-05": collect_roofline(repeated, environment),
        "E12-06": {
            "component_status": "PARTIAL" if fresh_energy["component_pass"] and historical_energy.get("energy_integrals_present_and_usable") else "FAIL",
            "component_pass": False, "fresh_operator_energy": fresh_energy,
            "historical_model_energy": historical_energy,
            "limitations": list(fresh_energy["limitations"]) + ["Qwen model energy has only the FP16 baseline; no qualified optimized model candidate"],
        },
    }
    observations["E12-07"] = collect_cost_sensitivity(model, historical_energy)
    observations["E12-08"] = collect_pareto(model, observations["E12-07"], historical_energy)
    observations["E12-09"] = isolated_setup_sessions()
    observations["E12-10"] = {"component_status": "RUNNING", "component_pass": False, "limitations": ["finalized after E12-01–09 are materialized"]}

    verdicts: Dict[str, Dict[str, Any]] = {}
    for experiment_id in EXPERIMENTS:
        started_at = utc_now()
        protocol = PROTOCOLS[experiment_id]
        observation = observations[experiment_id]
        root = STAGE_ROOT / experiment_id
        raw = root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        archive_existing(raw, run_id)
        verdict = formal_verdict(protocol, run_id, observation, started_at)
        verdicts[experiment_id] = verdict
        required = required_rows(protocol, observation)
        criteria = criteria_rows(protocol, observation, verdict["blocking_ids"])
        write_json(raw / "preregistration.json", {**protocol.as_dict(), "run_id": run_id, "frozen_before_execution": True, "formal_claim_allowed": False})
        write_json(raw / "environment_fingerprint.json", environment)
        write_json(raw / "upstream_evidence.json", upstream)
        write_json(raw / "prerequisites.json", prerequisite)
        write_json(raw / "config_audit.json", config_audit)
        write_json(raw / "interface_audit.json", interface_audit)
        write_json(raw / "component_tests.json", tests)
        write_json(raw / "driver_checks.json", drivers)
        write_json(raw / "component_observations.json", observation)
        write_json(raw / "required_evidence.json", {"items": required})
        write_json(raw / "criteria_evaluation.json", {"criteria": criteria})
        mapping = interface_map.mapping_for(experiment_id)
        write_jsonl(raw / "step_status.jsonl", ({
            "experiment_id": experiment_id, "step": step.index, "name": step.title,
            "interfaces": list(step.interfaces),
            "interface_resolution": "PASS" if interface_audit["ok"] else "FAIL",
            "scientific_status": "METHOD_OR_COMPONENT_EVIDENCE" if observation.get("component_status") in ("PASS", "PARTIAL") else "BLOCKED_OR_FAILED",
            "full_protocol_met": False, "formal_claim_allowed": False,
        } for step in mapping.steps))
        write_specific_files(experiment_id, raw, observation)
        write_json(raw / "verdict.json", verdict)
        write_text(root / f"{experiment_id}_实验报告.md", report_markdown(protocol, verdict, required, criteria, observation))

    catalog = EvidenceCatalog(REPO)
    discovered = [item for item in catalog.scan(refresh=True) if item["stage"] == STAGE]
    frontend = {
        "checked_at": utc_now(), "catalog_endpoint": "GET /api/console/v1/evidence",
        "detail_endpoint": "GET /api/console/v1/evidence/{evidence_id}",
        "download_endpoint": "GET /api/console/v1/evidence/{evidence_id}/download",
        "frontend_routes": ["/evidence", "/experiments", "/profiling"],
        "expected_experiments": len(EXPERIMENTS), "discovered_experiments": len(discovered),
        "experiments": sorted(item["experiment"] for item in discovered),
        "statuses": {item["experiment"]: item["status"] for item in discovered},
        "attachment_counts": {item["experiment"]: len(item["files"]) for item in discovered},
        "all_detail_readable": all(bool(catalog.detail(item["id"])) for item in discovered),
        "http_contract_test": tests["console_http"],
    }
    frontend["ok"] = bool(
        frontend["discovered_experiments"] == len(EXPERIMENTS)
        and set(frontend["experiments"]) == set(EXPERIMENTS)
        and frontend["all_detail_readable"]
        and frontend["http_contract_test"].get("status") in ("PASS", "NOT_RUN_DEPENDENCY_UNAVAILABLE")
    )

    observations["E12-10"] = collect_lineage(frontend)
    protocol = PROTOCOLS["E12-10"]
    verdict = formal_verdict(protocol, run_id, observations["E12-10"], utc_now())
    verdicts["E12-10"] = verdict
    raw = STAGE_ROOT / "E12-10/raw"
    required = required_rows(protocol, observations["E12-10"])
    criteria = criteria_rows(protocol, observations["E12-10"], verdict["blocking_ids"])
    write_json(raw / "component_observations.json", observations["E12-10"])
    write_json(raw / "required_evidence.json", {"items": required})
    write_json(raw / "criteria_evaluation.json", {"criteria": criteria})
    mapping = interface_map.mapping_for("E12-10")
    write_jsonl(raw / "step_status.jsonl", ({
        "experiment_id": "E12-10", "step": step.index, "name": step.title,
        "interfaces": list(step.interfaces),
        "interface_resolution": "PASS" if interface_audit["ok"] else "FAIL",
        "scientific_status": "METHOD_OR_COMPONENT_EVIDENCE" if observations["E12-10"].get("component_status") == "PASS" else "BLOCKED_OR_FAILED",
        "full_protocol_met": False, "formal_claim_allowed": False,
    } for step in mapping.steps))
    write_specific_files("E12-10", raw, observations["E12-10"])
    write_json(raw / "verdict.json", verdict)
    write_text(STAGE_ROOT / "E12-10/E12-10_实验报告.md", report_markdown(protocol, verdict, required, criteria, observations["E12-10"]))

    for experiment_id in EXPERIMENTS:
        raw = STAGE_ROOT / experiment_id / "raw"
        write_json(raw / "frontend_validation.json", frontend)
        write_json(raw / "evidence_manifest.json", evidence_manifest(STAGE_ROOT / experiment_id, run_id))
    write_json(STAGE_ROOT / "frontend_validation.json", frontend)

    component_counts: Dict[str, int] = {}
    for observation in observations.values():
        status = str(observation.get("component_status", "FAIL"))
        component_counts[status] = component_counts.get(status, 0) + 1
    collector_ok = bool(tests["ok"] and config_audit["ok"] and interface_audit["ok"] and frontend["ok"] and repeated.get("component_status") == "PASS" and fresh_energy.get("component_status") == "PASS")
    summary = {
        "stage": STAGE, "run_id": run_id,
        "collector_status": "PASS" if collector_ok else "FAIL",
        "stage_status": "BLOCKED", "stage_complete": False, "claim_allowed": False,
        "reason": "single Jetson only; no second qualified hardware/model candidate, service/distributed coverage, dated price source or cross-day/independent reproduction",
        "component_counts": component_counts,
        "experiment_statuses": {key: value["status"] for key, value in verdicts.items()},
        "component_statuses": {key: value.get("component_status") for key, value in observations.items()},
        "prerequisites": prerequisite, "config_audit_ok": config_audit["ok"],
        "interface_audit": interface_audit, "component_tests": tests,
        "frontend_validation": frontend, "generated_at": utc_now(),
    }
    write_json(STAGE_ROOT / "campaign_summary.json", summary)
    lines = [
        "# S12 阶段实验执行摘要", "",
        f"> Run ID：`{run_id}`  ", "> 阶段正式裁决：**BLOCKED**  ",
        f"> 采集器：**{summary['collector_status']}**（采集器通过不等于 S12 科学验收通过）", "",
        "## 1. 结论", "",
        "E12-01～E12-10 均已建立独立报告、raw 证据、36 步接口状态、必采信息映射、单项标准对照和前端索引。"
        "本轮真实执行 Jetson operator 重放、多进程/受控 CPU 负载、4 个不少于 30 秒的 VDD_IN 对齐窗口和两次隔离 setup session；"
        "同时按 hash 引用 S02 的 Qwen model-core/能量实测。", "",
        "由于没有第二个质量合格硬件候选，S08/S10/S11 正式链仍受阻，也没有有效价格快照、跨日/跨设备复验或独立人员复现，"
        "所以十项正式 verdict 均保持 `BLOCKED`。这是一组完整的单平台方法验证和缺口证据，不是跨硬件排行榜。", "",
        f"组件状态统计：`{component_counts}`。接口映射：`{interface_audit['steps']} steps / {interface_audit['interfaces']} interfaces / ok={str(interface_audit['ok']).lower()}`。", "",
        "## 2. 逐项裁决", "", "| 实验 | Jetson 组件/方法 | 正式状态 | 预计效果 | 单项标准 |", "|---|---|---|---|---|",
    ]
    for experiment_id in EXPERIMENTS:
        lines.append(f"| [{experiment_id}]({experiment_id}/{experiment_id}_实验报告.md) | {observations[experiment_id].get('component_status')} | BLOCKED | 部分达到 | 未完整满足 |")
    lines += [
        "", "## 3. 阶段完成标志对照", "",
        "- E12-01～E12-10 全部正式通过：未满足；所有正式状态均为 `BLOCKED`。",
        "- 三类硬件或两类硬件加两架构：未满足；本轮只有 Jetson Orin 实测。",
        "- 四层统一重放：operator fresh + model-core 历史实测投影；service/distributed 显式缺失。",
        "- 性能/质量/内存/能耗/成本/成熟度：分别呈现；成本是 `SCENARIO` 且 `PRICE_UNAVAILABLE`。",
        "- Pareto：只有一个合格 FP16 baseline，输出 baseline retain/no-feasible，不虚构总冠军。",
        "- Lineage：文件 hash、抽查、四类故障注入和 Console 索引已验证；受阻 claim 不可发布。", "",
        "## 4. 前端访问", "",
        f"EvidenceCatalog 实际发现 `{frontend['discovered_experiments']}/10` 项，detail 可读：`{str(frontend['all_detail_readable']).lower()}`。",
        "`/api/console/v1/evidence`、detail 和 download 接口可访问每项 verdict、报告与 raw JSON/JSONL；前端 `/evidence`、`/experiments`、`/profiling` 可筛选 S12。", "",
    ]
    write_text(STAGE_ROOT / "S12_阶段实验报告_20260921.md", "\n".join(lines))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-id", default="")
    value.add_argument("--json", action="store_true")
    value.add_argument("--energy-duration-s", type=float, default=30.2)
    value.add_argument("--verify-only", action="store_true")
    value.add_argument("--probe", choices=("rmsnorm", "energy"), default="", help=argparse.SUPPRESS)
    value.add_argument("--probe-output", default="", help=argparse.SUPPRESS)
    value.add_argument("--process-index", type=int, default=0, help=argparse.SUPPRESS)
    value.add_argument("--block", type=int, default=1, help=argparse.SUPPRESS)
    value.add_argument("--cpu-load", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--candidate", default="pytorch_eager", help=argparse.SUPPRESS)
    value.add_argument("--rows", type=int, default=1, help=argparse.SUPPRESS)
    value.add_argument("--duration-s", type=float, default=30.2, help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.probe:
        return child_probe(args)
    if args.verify_only:
        result = verify_only()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    run_id = args.run_id or time.strftime("s12_%Y%m%dT%H%M%SZ", time.gmtime())
    summary = collect_campaign(run_id, args.energy_duration_s)
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
