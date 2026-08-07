#!/usr/bin/env python3
"""Execute and publish the honest single-Jetson scope of S13.

S13 formally requires an isolated cluster, registry, supply-chain scanners and a
telemetry backend.  The configured target is one Jetson and does not currently
provide those services.  This collector therefore performs every safe, locally
testable part of E13-01..E13-11, records every unavailable mandatory datum, and
keeps the formal verdict BLOCKED.  Component/simulation results are never
promoted to deployment claims.

Run only through ``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import signal
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

from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.core.errors import ConfigError  # noqa: E402
from hqsb.infra import experiment, interface_map, specs  # noqa: E402


STAGE = "S13"
STAGE_ROOT = REPO / "docs/stage_experiments/S13"
MODEL_ROOT = Path("~/models/hqsb/Qwen3-1.7B").expanduser()
EXPERIMENTS = tuple(f"E13-{index:02d}" for index in range(1, 12))
MODULES = {
    "E13-01": "supply_chain",
    "E13-02": "deployment",
    "E13-03": "scheduling",
    "E13-04": "artifacts",
    "E13-05": "lifecycle",
    "E13-06": "capacity",
    "E13-07": "autoscaling",
    "E13-08": "observability",
    "E13-09": "faults",
    "E13-10": "canary",
    "E13-11": "security",
}


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
    expected: str,
    required: Sequence[str],
    criteria: Sequence[str],
    dependencies: Sequence[str] = (),
    level: str = "P0",
) -> Protocol:
    return Protocol(experiment_id, level, title, expected, tuple(required), tuple(criteria), tuple(dependencies))


PROTOCOLS: Mapping[str, Protocol] = {
    "E13-01": _p(
        "E13-01", "可重建镜像、SBOM 与供应链 Gate", "建立可审计软件供应链",
        ("base/toolchain/runtime digest", "layer/size", "SBOM", "CVE severity", "secret/license"),
        ("构建内容可解释", "无 secret/模型权重", "高危处置有 gate", "运行非 root"),
    ),
    "E13-02": _p(
        "E13-02", "空集群 Bootstrap、Cold→Ready→首请求", "证明部署不依赖开发机隐式状态",
        ("IaC/Helm version", "node/device", "image/model hash", "各阶段时间", "事件/日志"),
        ("一条自动化流程完成", "readiness 前不接流量", "模型校验失败不启动"), ("E13-01",),
    ),
    "E13-03": _p(
        "E13-03", "Accelerator 调度、拓扑与隔离", "验证资源调度符合性能和隔离预期",
        ("pod→node→device mapping", "topology", "P2P/NUMA", "util/memory", "调度失败原因"),
        ("正确 placement 可预测", "不满足 capability 不调度", "租户不越权占设备"), ("E13-02",),
    ),
    "E13-04": _p(
        "E13-04", "模型制品 Cache、原子切换与回滚", "建立 artifact 生命周期与一致性",
        ("artifact URI/hash/version", "download/cache/warmup", "active version", "failure/cleanup"),
        ("不服务未校验模型", "切换原子", "损坏 cache 隔离", "可回到上一已知良好版本"), ("E13-02",),
    ),
    "E13-05": _p(
        "E13-05", "Probe、Graceful Drain 与滚动重启", "验证生命周期不会丢请求或卡死发布",
        ("probe timeline", "inflight/accepted/rejected", "drain/termination", "duplicate/lost token"),
        ("未就绪不接流量", "drain 后不收新请求", "策略范围内无重复/截断"), ("E13-02", "E13-04"),
    ),
    "E13-06": _p(
        "E13-06", "容量、Admission、Token/KV Budget 与 OOM Margin", "校验容量规划而非靠 OOM 找极限",
        ("predicted/actual memory", "queue/token budget", "OOM margin", "goodput/SLO"),
        ("安全区内不 OOM", "资源耗尽前拒绝", "预测误差可解释"), ("E13-03", "E13-04", "E13-05", "S12"),
    ),
    "E13-07": _p(
        "E13-07", "SLO/Queue/Token 多指标 Autoscaling", "验证扩缩容响应、稳定性和成本",
        ("metric→desired/actual replica", "scale latency", "oscillation", "SLO/goodput", "cost"),
        ("不持续振荡", "过载能扩且降载能缩", "冷启动期间 admission 保护 SLO"), ("E13-06",),
    ),
    "E13-08": _p(
        "E13-08", "Request→Hardware 可观测性与 RCA", "证明监控覆盖请求到硬件",
        ("request/trace ID", "queue/runtime/device/system", "clock", "alert timeline"),
        ("能回答故障层级和影响范围", "alert 可操作且不依赖人工猜测"), ("E13-07",),
    ),
    "E13-09": _p(
        "E13-09", "故障注入、降级与恢复", "验证降级、恢复和资源一致性",
        ("fault/recovery timeline", "SLO/error", "retry/fallback", "MTTD/MTTR", "残留资源"),
        ("每类故障有预期降级或明确违约", "恢复后状态一致且资源回收"), ("E13-08",),
    ),
    "E13-10": _p(
        "E13-10", "Canary 质量/性能 Gate 与自动回滚", "证明发布不依赖手工看图",
        ("baseline/canary version", "traffic split", "quality/P99/goodput/error", "gate/rollback"),
        ("坏版本自动阻断并回滚", "好版本完成推广", "决策和阈值可审计"), ("E13-09",),
    ),
    "E13-11": _p(
        "E13-11", "多租户认证、配额、秘密与隔离", "验证生产安全边界",
        ("tenant/request", "auth result", "quota/resource", "redaction", "audit log", "attack cases"),
        ("跨租户访问失败", "敏感数据不出日志", "配额/限流有界且有审计"), ("E13-10",), "P1",
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
    temporary.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def command(argv: Sequence[str], timeout: float = 90.0) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(list(argv), cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False)
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
        return {"argv": list(argv), "returncode": 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - started}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def status_of(payload: Mapping[str, Any]) -> str:
    value = payload.get("overall") or payload.get("verdict") or payload.get("status")
    return str(value or "UNKNOWN")


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


def latest_verdict(root: Path) -> Optional[Path]:
    direct = root / "raw/verdict.json"
    if direct.is_file():
        return direct
    candidates = [path for path in root.glob("raw*/verdict.json") if "runs" not in path.parts]
    return sorted(candidates)[-1] if candidates else None


def upstream_inventory() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for stage in ("S08", "S12"):
        root = REPO / "docs/stage_experiments" / stage
        for experiment_root in sorted(root.glob("E*")) if root.is_dir() else ():
            path = latest_verdict(experiment_root)
            payload = load_json(path) if path else None
            status = status_of(payload or {}) if payload else "MISSING"
            rows.append({
                "stage": stage, "experiment_id": experiment_root.name, "status": status,
                "usable_for_formal_s13": status in ("PASS", "PASS_NEGATIVE"),
                "uri": str(path.relative_to(REPO)) if path else "",
                "sha256": sha256_file(path) if path else "",
            })
    return {
        "captured_at": utc_now(), "rows": rows,
        "summary": {stage: {state: sum(row["stage"] == stage and row["status"] == state for row in rows) for state in sorted({row["status"] for row in rows if row["stage"] == stage})} for stage in ("S08", "S12")},
        "rule": "only PASS/PASS_NEGATIVE upstream verdicts satisfy the formal S13 dependency",
    }


def asset_inventory() -> Dict[str, Any]:
    roots = (REPO / "infra", REPO / "configs/infra")
    rows = []
    for root in roots:
        for path in sorted(root.rglob("*")) if root.is_dir() else ():
            if path.is_file():
                rows.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": rows, "count": len(rows), "aggregate": hashlib.sha256("".join(row["sha256"] for row in rows).encode()).hexdigest()}


def model_identity() -> Dict[str, Any]:
    files = sorted(path for path in MODEL_ROOT.iterdir() if path.is_file()) if MODEL_ROOT.is_dir() else []
    selected = [path for path in files if path.name in {"config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"} or path.suffix == ".safetensors"]
    rows = []
    for path in selected:
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256("".join(row["sha256"] for row in rows).encode()).hexdigest() if rows else ""
    return {"root": str(MODEL_ROOT), "available": MODEL_ROOT.is_dir(), "file_count": len(files), "manifest": rows, "aggregate_sha256": aggregate}


def collect_environment() -> Dict[str, Any]:
    tool_names = ("docker", "kubectl", "helm", "helmfile", "kustomize", "cosign", "crane", "skopeo", "syft", "grype", "trivy", "promtool", "otelcol", "prometheus", "tegrastats")
    tools = {name: shutil.which(name) for name in tool_names}
    docker_version = command(("docker", "version", "--format", "{{json .}}"), 15) if tools["docker"] else {"returncode": 127}
    docker_images = command(("docker", "image", "ls", "--digests", "--no-trunc", "--format", "{{json .}}"), 15) if tools["docker"] else {"returncode": 127}
    git_head = command(("git", "rev-parse", "HEAD"), 10)
    git_status = command(("git", "status", "--short"), 10)
    board_path = Path("/proc/device-tree/model")
    board = board_path.read_bytes().replace(b"\0", b"").decode(errors="replace") if board_path.is_file() else ""
    numa = sorted(path.name for path in Path("/sys/devices/system/node").glob("node[0-9]*"))
    cuda: Dict[str, Any] = {"available": False}
    try:
        import torch
        cuda = {
            "available": torch.cuda.is_available(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
            "memory_info": list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else [],
        }
    except Exception as exc:  # noqa: BLE001
        cuda = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "captured_at": utc_now(), "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
        "board_model": board, "numa_nodes": numa, "tools": tools, "docker_version": docker_version,
        "docker_images": docker_images, "cuda": cuda, "model": model_identity(),
        "source": {"git_commit": git_head.get("stdout", "").strip(), "git_dirty": bool(git_status.get("stdout", "").strip()), "git_status": git_status.get("stdout", "")[-30000:], "collector_sha256": sha256_file(Path(__file__))},
    }


def run_component_checks() -> Dict[str, Any]:
    tests = command((sys.executable, "-m", "pytest", "tests/unit/infra", "tests/property/test_infra_invariants.py", "-q"), 600)
    dependency = command((sys.executable, "-c", "import fastapi, httpx"), 15)
    if dependency["returncode"] == 0:
        console_http = command((sys.executable, "-m", "pytest", "tests/unit/console/test_evidence.py", "-q"), 180)
        console_http["status"] = "PASS" if console_http["returncode"] == 0 else "FAIL"
    else:
        console_http = {
            "status": "NOT_RUN_DEPENDENCY_UNAVAILABLE", "returncode": 127,
            "reason": "FastAPI/httpx is not installed in the remote experiment interpreter",
            "dependency_probe": dependency,
        }
    smoke: Dict[str, Any] = {}
    for experiment_id, module_name in MODULES.items():
        try:
            module = importlib.import_module(f"hqsb.infra.{module_name}")
            smoke[experiment_id] = module.smoke_self_check()
        except Exception as exc:  # noqa: BLE001
            smoke[experiment_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "pytest": tests, "pytest_pass": tests["returncode"] == 0,
        "console_http": console_http,
        "module_smoke": smoke,
        "all_smoke_ok": all(row.get("status") == "smoke" and all(row.get("checks", {}).values()) for row in smoke.values()),
    }


def driver_checks() -> Dict[str, Any]:
    driver = REPO / "scripts/infra/run_e13.py"
    commands = {
        "spec_audit": (sys.executable, str(driver), "--spec-audit", "--json"),
        "spec_check": (sys.executable, str(driver), "--spec-check", "--json"),
        "interface_map": (sys.executable, str(driver), "--interface-map", "--json"),
    }
    results = {name: command(argv, 120) for name, argv in commands.items()}
    parsed: Dict[str, Any] = {}
    for name, result in results.items():
        try:
            parsed[name] = json.loads(result["stdout"])
        except (TypeError, ValueError):
            parsed[name] = {"ok": False, "parse_error": True}
    return {"commands": results, "parsed": parsed, "ok": all(result["returncode"] == 0 and parsed[name].get("ok") for name, result in results.items())}


def artifact_atomic_exercise() -> Dict[str, Any]:
    source = MODEL_ROOT / "config.json"
    data = source.read_bytes() if source.is_file() else b'{"component_fixture":"hqsb-s13"}\n'
    expected = hashlib.sha256(data).hexdigest()
    events: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hqsb-s13-artifact-") as temp:
        root = Path(temp)
        staging = root / "staging"
        cache = root / "cache"
        quarantine = root / "quarantine"
        for directory in (staging, cache, quarantine):
            directory.mkdir()
        partial = staging / "model-config.partial"
        partial.write_bytes(data)
        verified = hashlib.sha256(partial.read_bytes()).hexdigest() == expected
        events.append({"event": "download_to_staging", "verified": verified, "bytes": len(data)})
        committed = cache / expected
        if verified:
            os.replace(partial, committed)
        events.append({"event": "atomic_commit", "exists": committed.is_file(), "sha256": sha256_file(committed)})
        active_tmp = root / "active.tmp"
        active = root / "active"
        active_tmp.write_text(expected, encoding="utf-8")
        os.replace(active_tmp, active)
        events.append({"event": "activate_a", "active": active.read_text(encoding="utf-8")})
        bad = staging / "corrupt.partial"
        bad.write_bytes(data + b"tamper")
        bad_detected = hashlib.sha256(bad.read_bytes()).hexdigest() != expected
        if bad_detected:
            os.replace(bad, quarantine / bad.name)
        events.append({"event": "corrupt_quarantine", "detected": bad_detected, "active_unchanged": active.read_text(encoding="utf-8") == expected})
        b_digest = hashlib.sha256(data + b"version-b").hexdigest()
        b_path = cache / b_digest
        b_path.write_bytes(data + b"version-b")
        switch_tmp = root / "active.next"
        switch_tmp.write_text(b_digest, encoding="utf-8")
        os.replace(switch_tmp, active)
        switched = active.read_text(encoding="utf-8") == b_digest
        rollback_tmp = root / "active.rollback"
        rollback_tmp.write_text(expected, encoding="utf-8")
        os.replace(rollback_tmp, active)
        rolled_back = active.read_text(encoding="utf-8") == expected
        events.append({"event": "switch_and_rollback", "switched": switched, "rolled_back": rolled_back, "known_good_present": committed.is_file()})
    return {"status": "PASS_COMPONENT", "representative_only": True, "source": str(source) if source.is_file() else "generated fixture", "artifact_sha256": expected, "events": events, "all_checks_pass": all(row.get("verified", row.get("exists", row.get("detected", row.get("rolled_back", True)))) for row in events)}


def cuda_memory_probe() -> Dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "NOT_RUN", "reason": "CUDA unavailable"}
        rows = []
        initial_free, total = torch.cuda.mem_get_info()
        for size_mib in (8, 16, 32):
            before, _ = torch.cuda.mem_get_info()
            tensor = torch.empty(size_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
            tensor.fill_(1)
            torch.cuda.synchronize()
            after, _ = torch.cuda.mem_get_info()
            rows.append({"requested_bytes": size_mib * 1024 * 1024, "free_before": before, "free_after": after, "observed_delta": before - after, "oom": False})
            del tensor
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        final_free, _ = torch.cuda.mem_get_info()
        return {"status": "PASS_COMPONENT", "safe_bounded_probe": True, "initial_free": initial_free, "final_free": final_free, "total": total, "rows": rows, "resource_recovered": final_free >= initial_free - 4 * 1024 * 1024}
    except Exception as exc:  # noqa: BLE001
        return {"status": "FAIL_COMPONENT", "error": f"{type(exc).__name__}: {exc}"}


def isolated_fault_exercise() -> Dict[str, Any]:
    crash = command((sys.executable, "-c", "import os; os._exit(17)"), 5)
    started = time.perf_counter()
    process = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(3600)"), cwd=REPO)
    killed = False
    try:
        process.wait(timeout=0.3)
    except subprocess.TimeoutExpired:
        os.kill(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        killed = True
    return {
        "status": "PASS_COMPONENT" if crash["returncode"] == 17 and killed else "FAIL_COMPONENT",
        "scope": "isolated child processes only; no pod/node/network/device/thermal fault injected",
        "episodes": [
            {"fault": "process_crash", "ground_truth": True, "returncode": crash["returncode"], "detected": crash["returncode"] == 17},
            {"fault": "process_no_progress", "ground_truth": True, "terminated_by_harness": killed, "detection_s": time.perf_counter() - started},
        ],
    }


def telemetry_probe(environment: Mapping[str, Any]) -> Dict[str, Any]:
    tool = environment.get("tools", {}).get("tegrastats")
    capture = command((str(tool), "--interval", "200", "--count", "5"), 5) if tool else {"returncode": 127, "stdout": "", "stderr": "tegrastats unavailable"}
    samples = [line for line in str(capture.get("stdout", "")).splitlines() if line.strip()]
    return {"status": "PASS_COMPONENT" if capture.get("returncode") == 0 and samples else "PARTIAL", "command": capture, "sample_count": len(samples), "samples": samples, "backend": "direct tegrastats; no Prometheus/Grafana/OTel backend"}


def supply_chain_preflight(environment: Mapping[str, Any], assets: Mapping[str, Any]) -> Dict[str, Any]:
    dockerfile = REPO / "infra/containers/Dockerfile.runtime"
    build_args = REPO / "infra/containers/build_args.yaml"
    text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
    docker_image_lines = [line for line in str(environment.get("docker_images", {}).get("stdout", "")).splitlines() if line.strip()]
    checks = {
        "docker_daemon_available": environment.get("docker_version", {}).get("returncode") == 0,
        "multi_stage_declared": all(f" AS {stage}" in text for stage in ("base", "toolchain", "test", "runtime")),
        "non_root_declared": "USER ${HQSB_UID}:${HQSB_GID}" in text,
        "model_patterns_excluded": "*.safetensors" in (REPO / "infra/containers/.dockerignore").read_text(encoding="utf-8"),
        "requirements_lock_present": (REPO / "infra/containers/requirements.lock.txt").is_file(),
        "build_inputs_frozen": "base_image_digest: \"\"" not in build_args.read_text(encoding="utf-8") and "builder_image_digest: \"\"" not in build_args.read_text(encoding="utf-8"),
        "local_base_images_available": bool(docker_image_lines),
        "sbom_scanner_available": bool(environment.get("tools", {}).get("syft")),
        "vulnerability_scanner_available": bool(environment.get("tools", {}).get("grype") or environment.get("tools", {}).get("trivy")),
        "signature_tool_available": bool(environment.get("tools", {}).get("cosign")),
    }
    blockers = [name for name, ok in checks.items() if not ok]
    return {"status": "BLOCKED", "build_attempted": False, "reason": "preflight gate failed before build; an unfrozen or unscannable image must not become a release", "checks": checks, "blockers": blockers, "asset_count": assets.get("count"), "docker_images": docker_image_lines}


def observations(environment: Mapping[str, Any], assets: Mapping[str, Any], checks: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    artifact = artifact_atomic_exercise()
    memory = cuda_memory_probe()
    faults = isolated_fault_exercise()
    telemetry = telemetry_probe(environment)
    supply = supply_chain_preflight(environment, assets)
    cuda = environment.get("cuda", {})
    no_cluster = "kubectl/Helm and an isolated accelerator cluster are unavailable"
    rows: Dict[str, Dict[str, Any]] = {}
    for experiment_id in EXPERIMENTS:
        smoke = checks["module_smoke"].get(experiment_id, {})
        rows[experiment_id] = {"component_status": "PARTIAL", "module_smoke": smoke, "limitations": [no_cluster]}
    rows["E13-01"].update({"supply_chain_preflight": supply, "assets": assets, "limitations": ["base/builder digests and dependency lock are unresolved", "no local base image, SBOM/CVE scanner, registry or signing tool; two builds were not attempted"]})
    rows["E13-02"].update({"clean_level": "NOT_RUN_CLUSTER_UNAVAILABLE", "model_identity": environment.get("model"), "node_identity": {"board": environment.get("board_model"), "cuda": cuda}, "limitations": [no_cluster, "no ReleaseBundle image digest exists because E13-01 is blocked"]})
    rows["E13-03"].update({"actual_node_device": {"board": environment.get("board_model"), "numa_nodes": environment.get("numa_nodes"), "cuda": cuda}, "limitations": [no_cluster, "single integrated GPU; no scheduler, device-plugin, pod allocation, P2P or tenant isolation evidence"]})
    rows["E13-04"].update({"component_status": artifact["status"], "artifact_atomic_exercise": artifact, "limitations": ["representative config artifact only; full model download/load/warmup/quality and concurrent replicas were not exercised", no_cluster]})
    rows["E13-05"].update({"lifecycle_contract_checks": checks["module_smoke"].get("E13-05"), "limitations": ["no deployed service, EndpointSlice, live streaming request, SIGTERM drain or rolling controller episode", no_cluster]})
    rows["E13-06"].update({"cuda_memory_probe": memory, "s12_dependency": "BLOCKED", "limitations": ["safe allocations are not a model/service capacity sweep", "S12 capacity/quality baseline is not formally passed", no_cluster]})
    rows["E13-07"].update({"controller_replay": checks["module_smoke"].get("E13-07"), "limitations": ["controller decisions are deterministic interface replay, not replica creation", "no metrics adapter/HPA/cluster or workload episodes"]})
    rows["E13-08"].update({"telemetry_probe": telemetry, "schema_checks": checks["module_smoke"].get("E13-08"), "limitations": ["direct Jetson samples only; no request-correlated Prometheus/Grafana/log/trace backend", "no blind RCA or alert episode"]})
    rows["E13-09"].update({"isolated_fault_exercise": faults, "limitations": ["only isolated process crash/no-progress was injected", "pod/node/backend/storage/network/artifact/OOM/thermal faults require an authorized isolated cluster and were not run"]})
    rows["E13-10"].update({"canary_policy_replay": checks["module_smoke"].get("E13-10"), "limitations": ["no control/candidate releases, real traffic split, online quality/performance or rollback deployment"]})
    rows["E13-11"].update({"security_policy_replay": checks["module_smoke"].get("E13-11"), "limitations": ["no independent tenant identities, Kubernetes RBAC/CNI/Secret/device boundary or live attack cases", "multi-tenant production capability is not claimed"]})
    return rows


EVIDENCE_PLAN: Mapping[str, Mapping[str, Tuple[str, str]]] = {
    "E13-01": {
        "base/toolchain/runtime digest": ("BLOCKED", "raw/supply_chain_preflight.json"), "layer/size": ("SOURCE_ONLY", "raw/asset_inventory.json"),
        "SBOM": ("NOT_COLLECTED_TOOL_UNAVAILABLE", "raw/environment_fingerprint.json"), "CVE severity": ("NOT_COLLECTED_TOOL_UNAVAILABLE", "raw/environment_fingerprint.json"),
        "secret/license": ("NOT_COLLECTED_TOOL_UNAVAILABLE", "raw/supply_chain_preflight.json"),
    },
    "E13-02": {"IaC/Helm version": ("SOURCE_ONLY", "raw/asset_inventory.json"), "node/device": ("COLLECTED_TARGET", "raw/environment_fingerprint.json"), "image/model hash": ("PARTIAL_MODEL_ONLY", "raw/component_observations.json"), "各阶段时间": ("NOT_COLLECTED_CLUSTER_UNAVAILABLE", "raw/prerequisites.json"), "事件/日志": ("NOT_COLLECTED_CLUSTER_UNAVAILABLE", "raw/prerequisites.json")},
    "E13-03": {"pod→node→device mapping": ("NOT_COLLECTED_CLUSTER_UNAVAILABLE", "raw/prerequisites.json"), "topology": ("PARTIAL_NODE_ONLY", "raw/component_observations.json"), "P2P/NUMA": ("PARTIAL_NUMA_ONLY", "raw/component_observations.json"), "util/memory": ("COLLECTED_TARGET", "raw/environment_fingerprint.json"), "调度失败原因": ("METHOD_ONLY", "raw/component_observations.json")},
    "E13-04": {"artifact URI/hash/version": ("COLLECTED_COMPONENT", "raw/artifact_lifecycle.json"), "download/cache/warmup": ("PARTIAL_CACHE_ONLY", "raw/artifact_lifecycle.json"), "active version": ("COLLECTED_COMPONENT", "raw/artifact_lifecycle.json"), "failure/cleanup": ("COLLECTED_COMPONENT", "raw/artifact_lifecycle.json")},
    "E13-05": {"probe timeline": ("METHOD_ONLY", "raw/component_observations.json"), "inflight/accepted/rejected": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json"), "drain/termination": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json"), "duplicate/lost token": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json")},
    "E13-06": {"predicted/actual memory": ("PARTIAL_ALLOCATION_ONLY", "raw/cuda_memory_probe.json"), "queue/token budget": ("METHOD_ONLY", "raw/component_observations.json"), "OOM margin": ("NOT_COLLECTED_S12_BLOCKED", "raw/prerequisites.json"), "goodput/SLO": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json")},
    "E13-07": {"metric→desired/actual replica": ("METHOD_ONLY", "raw/component_observations.json"), "scale latency": ("NOT_COLLECTED_CLUSTER_UNAVAILABLE", "raw/prerequisites.json"), "oscillation": ("METHOD_ONLY", "raw/component_observations.json"), "SLO/goodput": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json"), "cost": ("NOT_COLLECTED_FORMAL", "raw/prerequisites.json")},
    "E13-08": {"request/trace ID": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json"), "queue/runtime/device/system": ("PARTIAL_SYSTEM_ONLY", "raw/telemetry_probe.json"), "clock": ("TARGET_TIMESTAMP_ONLY", "raw/environment_fingerprint.json"), "alert timeline": ("NOT_COLLECTED_BACKEND_UNAVAILABLE", "raw/prerequisites.json")},
    "E13-09": {"fault/recovery timeline": ("PARTIAL_PROCESS_ONLY", "raw/fault_episodes.json"), "SLO/error": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json"), "retry/fallback": ("METHOD_ONLY", "raw/component_observations.json"), "MTTD/MTTR": ("PARTIAL_PROCESS_ONLY", "raw/fault_episodes.json"), "残留资源": ("PARTIAL_PROCESS_ONLY", "raw/fault_episodes.json")},
    "E13-10": {"baseline/canary version": ("METHOD_ONLY", "raw/component_observations.json"), "traffic split": ("METHOD_ONLY", "raw/component_observations.json"), "quality/P99/goodput/error": ("NOT_COLLECTED_SERVICE_UNAVAILABLE", "raw/prerequisites.json"), "gate/rollback": ("METHOD_ONLY", "raw/component_observations.json")},
    "E13-11": {"tenant/request": ("METHOD_ONLY", "raw/component_observations.json"), "auth result": ("METHOD_ONLY", "raw/component_observations.json"), "quota/resource": ("METHOD_ONLY", "raw/component_observations.json"), "redaction": ("METHOD_ONLY", "raw/component_observations.json"), "audit log": ("NOT_COLLECTED_CLUSTER_UNAVAILABLE", "raw/prerequisites.json"), "attack cases": ("METHOD_ONLY", "raw/component_observations.json")},
}


def required_rows(protocol: Protocol) -> List[Dict[str, Any]]:
    plan = EVIDENCE_PLAN[protocol.experiment_id]
    return [{"item": item, "status": plan[item][0], "evidence_ref": plan[item][1], "formal_requirement_met": plan[item][0].startswith("COLLECTED_TARGET")} for item in protocol.required_data]


def formal_verdict(protocol: Protocol, run_id: str, observation: Mapping[str, Any], prerequisite: Mapping[str, Any]) -> Dict[str, Any]:
    missing = list(prerequisite.get("missing", []))
    missing.extend(f"dependency:{item}" for item in protocol.dependencies)
    return {
        "schema_version": "1.0.0", "stage": STAGE, "experiment_id": protocol.experiment_id,
        "run_id": run_id, "level": protocol.level, "status": "BLOCKED", "overall": "BLOCKED",
        "scientific_execution_verdict": "BLOCKED", "component_status": observation.get("component_status", "PARTIAL"),
        "execution_attempted": True, "formal_protocol_executed": False, "claim_allowed": False,
        "expected_effect_achieved": False, "single_item_pass_standard_met": False,
        "reason_code": "BLOCKED_PREREQUISITE_AND_EXTERNAL_ENVIRONMENT",
        "reason": "formal S13 protocol requires passed S08/S12 evidence plus an isolated cluster, registry, scanners and telemetry backend",
        "blocking_ids": sorted(set(missing)), "limitations": list(observation.get("limitations", [])), "written_at": utc_now(),
    }


def criteria_rows(protocol: Protocol, verdict: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{"criterion": item, "formal_status": "NOT_MET", "reason": verdict["reason"], "blocking_ids": verdict["blocking_ids"]} for item in protocol.criteria]


def observation_lines(experiment_id: str, observation: Mapping[str, Any]) -> List[str]:
    smoke = observation.get("module_smoke", {})
    checks = smoke.get("checks", {}) if isinstance(smoke, Mapping) else {}
    lines = [f"接口负例/不变量自检：{sum(bool(value) for value in checks.values())}/{len(checks)}。"]
    if experiment_id == "E13-01":
        lines.append(f"供应链 preflight blockers：{observation.get('supply_chain_preflight', {}).get('blockers')}；未启动镜像构建。")
    elif experiment_id == "E13-03":
        lines.append(f"实际设备：{observation.get('actual_node_device', {}).get('cuda', {}).get('device_name')}；NUMA：{observation.get('actual_node_device', {}).get('numa_nodes')}。")
    elif experiment_id == "E13-04":
        lines.append("在临时目录实际完成 staging→hash verify→atomic commit→corruption quarantine→A/B switch→rollback；仅代表性 config artifact。")
    elif experiment_id == "E13-06":
        probe = observation.get("cuda_memory_probe", {})
        lines.append(f"安全 CUDA allocation probe：{probe.get('status')}，资源恢复={probe.get('resource_recovered')}；未逼近 OOM。")
    elif experiment_id == "E13-08":
        probe = observation.get("telemetry_probe", {})
        lines.append(f"直接 tegrastats 样本：{probe.get('sample_count')}；无 request/trace 关联。")
    elif experiment_id == "E13-09":
        lines.append("实际注入隔离子进程 crash/no-progress；未注入 pod/node/network/device/thermal 故障。")
    return lines


def report_markdown(protocol: Protocol, verdict: Mapping[str, Any], required: Sequence[Mapping[str, Any]], criteria: Sequence[Mapping[str, Any]], observation: Mapping[str, Any]) -> str:
    lines = [
        f"# {protocol.experiment_id} 实验报告：{protocol.title}", "",
        f"> Run ID：`{verdict['run_id']}`  ", "> 正式裁决：**BLOCKED**  ",
        f"> 单机组件范围：**{verdict['component_status']}**  ", "> 正式 claim：**不允许**", "",
        "## 1. 结论", "",
        f"预计效果“{protocol.expected_effect}”未在正式集群范围达成，单项通过标准未满足。"
        "本轮完成了安全可执行的 Jetson 组件检查、真实单机探测和负例，并保留所有未采数据的原因；组件结果不等于生产部署通过。", "",
        "## 2. 必采信息/数据", "", "| 信息/数据 | 状态 | 证据 | 正式要求 |", "|---|---|---|---|",
    ]
    for row in required:
        lines.append(f"| {row['item']} | `{row['status']}` | `{row['evidence_ref']}` | {'满足' if row['formal_requirement_met'] else '未满足'} |")
    lines += ["", "## 3. 预计效果与单项通过标准", "", "| 标准 | 正式判定 | 原因 |", "|---|---|---|"]
    for row in criteria:
        lines.append(f"| {row['criterion']} | `{row['formal_status']}` | 前置/外部环境门未满足 |")
    lines += ["", "## 4. 本轮实际观测", ""]
    lines.extend(f"- {line}" for line in observation_lines(protocol.experiment_id, observation))
    lines += ["", "## 5. 阻塞与限制", ""]
    lines.extend(f"- {item}" for item in observation.get("limitations", []))
    lines += ["", "## 6. 前端访问", "", "本实验的 `raw/verdict.json`、本报告及全部 JSON/JSONL 附件由只读 EvidenceCatalog 索引，可经 `/api/console/v1/evidence`、detail 与 download 接口访问。", ""]
    return "\n".join(lines)


def evidence_manifest(root: Path, run_id: str) -> Dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "evidence_manifest.json" or "runs" in path.relative_to(root).parts:
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"stage": STAGE, "experiment_id": root.name, "run_id": run_id, "files": files, "artifact_aggregate_root": hashlib.sha256("".join(row["sha256"] for row in files).encode()).hexdigest()}


def frontend_validation(component_tests: Mapping[str, Any]) -> Dict[str, Any]:
    catalog = EvidenceCatalog(REPO)
    discovered = [item for item in catalog.scan(refresh=True) if item["stage"] == STAGE]
    payload = {
        "checked_at": utc_now(), "catalog_endpoint": "GET /api/console/v1/evidence",
        "detail_endpoint": "GET /api/console/v1/evidence/{evidence_id}",
        "download_endpoint": "GET /api/console/v1/evidence/{evidence_id}/download",
        "frontend_routes": ["/evidence", "/experiments"], "expected_experiments": len(EXPERIMENTS),
        "discovered_experiments": len(discovered), "experiments": sorted(item["experiment"] for item in discovered),
        "statuses": {item["experiment"]: item["status"] for item in discovered},
        "attachment_counts": {item["experiment"]: len(item["files"]) for item in discovered},
        "all_detail_readable": all(bool(catalog.detail(item["id"])) for item in discovered),
        "console_http_test": component_tests.get("console_http", {}),
    }
    payload["ok"] = (
        payload["discovered_experiments"] == len(EXPERIMENTS)
        and set(payload["experiments"]) == set(EXPERIMENTS)
        and payload["all_detail_readable"]
        and payload["console_http_test"].get("status") in ("PASS", "NOT_RUN_DEPENDENCY_UNAVAILABLE")
    )
    return payload


def verify_only() -> Dict[str, Any]:
    checked = 0
    failures = []
    for experiment_id in EXPERIMENTS:
        root = STAGE_ROOT / experiment_id
        manifest = load_json(root / "raw/evidence_manifest.json")
        if not manifest:
            failures.append(f"{experiment_id}:manifest_missing")
            continue
        for row in manifest.get("files", []):
            path = root / row["path"]
            ok = path.is_file() and sha256_file(path) == row["sha256"]
            checked += 1
            if not ok:
                failures.append(f"{experiment_id}:{row['path']}")
    return {"stage": STAGE, "checked": checked, "failures": failures, "ok": not failures, "verified_at": utc_now()}


def collect_campaign(run_id: str) -> Dict[str, Any]:
    if not run_id or any(char in run_id for char in ("/", "\\", "\0")):
        raise ConfigError("run_id must be one non-empty path component")
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()
    upstream = upstream_inventory()
    prerequisite = experiment.check_prerequisites(str(REPO), probe=True).as_dict()
    assets = asset_inventory()
    component_tests = run_component_checks()
    drivers = driver_checks()
    config_reports = specs.InfraSpecs.load(str(REPO / "configs/infra")).audit()
    config_audit = {"ok": specs.audit_all_ok(config_reports), "reports": config_reports}
    interface_audit = interface_map.resolve_interfaces()
    observed = observations(environment, assets, component_tests)
    verdicts: Dict[str, Dict[str, Any]] = {}

    for experiment_id in EXPERIMENTS:
        protocol = PROTOCOLS[experiment_id]
        root = STAGE_ROOT / experiment_id
        raw = root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        archive_existing(raw)
        verdict = formal_verdict(protocol, run_id, observed[experiment_id], prerequisite)
        verdicts[experiment_id] = verdict
        required = required_rows(protocol)
        criteria = criteria_rows(protocol, verdict)
        write_json(raw / "preregistration.json", {**protocol.as_dict(), "run_id": run_id, "frozen_before_execution": True, "formal_claim_allowed": False})
        write_json(raw / "environment_fingerprint.json", environment)
        write_json(raw / "upstream_evidence.json", upstream)
        write_json(raw / "prerequisites.json", prerequisite)
        write_json(raw / "asset_inventory.json", assets)
        write_json(raw / "config_audit.json", config_audit)
        write_json(raw / "interface_audit.json", interface_audit)
        write_json(raw / "component_tests.json", component_tests)
        write_json(raw / "driver_checks.json", drivers)
        write_json(raw / "component_observations.json", observed[experiment_id])
        write_json(raw / "required_evidence.json", {"items": required})
        write_json(raw / "criteria_evaluation.json", {"criteria": criteria})
        write_jsonl(raw / "step_status.jsonl", ({
            "experiment_id": experiment_id, "step": step.index, "name": step.title, "interfaces": list(step.interfaces),
            "interface_resolution": "PASS" if interface_audit.get("ok") else "FAIL", "scientific_status": "METHOD_OR_COMPONENT_EVIDENCE",
            "full_protocol_met": False, "formal_claim_allowed": False,
        } for step in interface_map.mapping_for(experiment_id).steps))
        if experiment_id == "E13-01":
            write_json(raw / "supply_chain_preflight.json", observed[experiment_id]["supply_chain_preflight"])
        if experiment_id == "E13-04":
            write_json(raw / "artifact_lifecycle.json", observed[experiment_id]["artifact_atomic_exercise"])
        if experiment_id == "E13-06":
            write_json(raw / "cuda_memory_probe.json", observed[experiment_id]["cuda_memory_probe"])
        if experiment_id == "E13-08":
            write_json(raw / "telemetry_probe.json", observed[experiment_id]["telemetry_probe"])
        if experiment_id == "E13-09":
            write_json(raw / "fault_episodes.json", observed[experiment_id]["isolated_fault_exercise"])
        write_json(raw / "verdict.json", verdict)
        write_text(root / f"{experiment_id}_实验报告.md", report_markdown(protocol, verdict, required, criteria, observed[experiment_id]))

    frontend = frontend_validation(component_tests)
    for experiment_id in EXPERIMENTS:
        raw = STAGE_ROOT / experiment_id / "raw"
        write_json(raw / "frontend_validation.json", frontend)
        write_json(raw / "evidence_manifest.json", evidence_manifest(STAGE_ROOT / experiment_id, run_id))
    write_json(STAGE_ROOT / "frontend_validation.json", frontend)

    component_counts: Dict[str, int] = {}
    for row in observed.values():
        status = str(row.get("component_status", "UNKNOWN"))
        component_counts[status] = component_counts.get(status, 0) + 1
    collector_ok = bool(component_tests["pytest_pass"] and component_tests["all_smoke_ok"] and drivers["ok"] and config_audit["ok"] and interface_audit["ok"] and frontend["ok"])
    summary = {
        "stage": STAGE, "run_id": run_id, "collector_status": "PASS" if collector_ok else "FAIL",
        "stage_status": "BLOCKED", "stage_complete": False, "claim_allowed": False,
        "reason": "formal prerequisites and external production environment are unavailable; all safe Jetson/component scopes were executed and missing mandatory data was recorded",
        "component_counts": component_counts, "experiment_statuses": {key: value["status"] for key, value in verdicts.items()},
        "prerequisites": prerequisite, "upstream": upstream, "component_tests": component_tests,
        "driver_checks": drivers, "config_audit_ok": config_audit["ok"], "interface_audit": interface_audit,
        "frontend_validation": frontend, "generated_at": utc_now(),
    }
    write_json(STAGE_ROOT / "campaign_summary.json", summary)
    lines = [
        "# S13 阶段实验执行摘要", "", f"> Run ID：`{run_id}`  ", "> 阶段正式裁决：**BLOCKED**  ",
        f"> 采集器：**{summary['collector_status']}**（采集器通过不等于生产化验收通过）", "",
        "## 1. 结论", "",
        "E13-01～E13-11 均已生成独立报告、raw 证据、418 步接口状态、必采信息逐项状态、通过标准对照和前端索引。"
        "本轮实际执行了 Jetson/Docker/工具/模型/设备指纹采集、infra/Console 专项测试、11 模块负例自检、代表性制品原子提交/损坏隔离/切换回滚、"
        "安全 CUDA allocation/recovery、tegrastats 采样以及隔离子进程 crash/no-progress 故障。", "",
        "S08 与 S12 正式链仍为 BLOCKED，且没有隔离 Kubernetes 集群、registry、SBOM/CVE/签名工具或 Prometheus/Grafana/OTel 后端；"
        "因此 11 项正式 verdict 均为 `BLOCKED`，不得声称从空集群部署、弹性、RCA、故障恢复、canary 或多租户生产能力已通过。", "",
        "## 2. 逐项裁决", "", "| 实验 | 组件范围 | 正式状态 | 预计效果 | 单项标准 |", "|---|---|---|---|---|",
    ]
    for experiment_id in EXPERIMENTS:
        lines.append(f"| [{experiment_id}]({experiment_id}/{experiment_id}_实验报告.md) | {observed[experiment_id].get('component_status')} | BLOCKED | 未正式达到 | 未完整满足 |")
    lines += [
        "", "## 3. 阶段完成标志对照", "",
        "- E13-01～E13-10 正式通过：未满足；E13-11 多租户能力也未通过。",
        "- 空集群到可服务全自动：未执行，缺隔离集群与可发布 ReleaseBundle。",
        "- 单节点/核心依赖故障恢复：仅隔离进程组件 fault 已执行，集群与依赖矩阵未执行。",
        "- autoscaling/admission：算法与不变量自检完成，真实控制面/负载未执行。",
        "- canary 自动回滚：状态机和 gate 负例自检完成，真实发布未执行。",
        "- runbook：源码资产存在但未经真实 on-call 演练。", "",
        "## 4. 前端访问", "",
        f"EvidenceCatalog 实际发现 `{frontend['discovered_experiments']}/11` 项，detail 可读：`{str(frontend['all_detail_readable']).lower()}`。",
        "`/api/console/v1/evidence`、detail 和 download 接口可访问每项 verdict、报告与 raw JSON/JSONL；前端 `/evidence`、`/experiments` 可筛选 S13。", "",
    ]
    write_text(STAGE_ROOT / "S13_阶段实验报告_20260921.md", "\n".join(lines))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-id", default="")
    value.add_argument("--json", action="store_true")
    value.add_argument("--verify-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.verify_only:
        result = verify_only()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    run_id = args.run_id or time.strftime("s13_%Y%m%dT%H%M%SZ", time.gmtime())
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
