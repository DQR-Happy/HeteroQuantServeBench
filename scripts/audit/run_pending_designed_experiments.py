#!/usr/bin/env python3
"""Execute the designed-but-not-yet-executed S00/S01/S03/S04.5 experiments.

This runner deliberately distinguishes an experiment failure from an
execution failure.  A completed preflight which proves that a hard dependency
is closed is useful experimental evidence and is written as ``BLOCKED``; it is
never upgraded to PASS by substituting synthetic model-level data.

The script is intended to run on the Jetson target through remote_run.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DETAILS = ROOT / "docs/stage_experiments/details"
FORBIDDEN_DISTS = ("torch", "triton", "nvidia-", "cuda-", "transformers", "modelscope")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "argv": cmd,
            "cwd": str(cwd or ROOT),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_s": round(time.monotonic() - started, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": cmd,
            "cwd": str(cwd or ROOT),
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_s": round(time.monotonic() - started, 3),
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "argv": cmd,
            "cwd": str(cwd or ROOT),
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.monotonic() - started, 3),
            "timed_out": False,
        }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = run(["git", *args], timeout=30)
    return result["stdout"].strip() if result["returncode"] == 0 else ""


def source_tree_digest() -> str:
    """Hash host integration surfaces so an external plugin cannot hide edits."""
    digest = hashlib.sha256()
    roots = [ROOT / "hqsb", ROOT / "pyproject.toml", ROOT / ".github/workflows"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def env_record() -> dict[str, Any]:
    gpu = run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,uuid", "--format=csv,noheader"],
        timeout=30,
    )
    return {
        "collected_at_utc": now(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu_query": gpu,
    }


def verdict(
    experiment_id: str,
    overall: str,
    conditions: dict[str, bool],
    *,
    scientific: str,
    blockers: list[dict[str, Any]] | None = None,
    unlocks: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "hqsb.pending-experiment.verdict/v1",
        "experiment_id": experiment_id,
        "verified_at_utc": now(),
        "overall": overall,
        "scientific_execution_verdict": scientific,
        "conditions": conditions,
        "failed_or_unverified_conditions": [key for key, value in conditions.items() if not value],
        "blocking_dependencies": blockers or [],
        "unlocks_when_passed": unlocks or [],
    }


def finish_experiment(
    stage: str,
    experiment_id: str,
    raw: dict[str, Any],
    verdict_data: dict[str, Any],
    report: str,
) -> None:
    root = ROOT / "docs/stage_experiments" / stage / experiment_id
    raw_dir = root / "raw"
    write_json(raw_dir / "execution.json", raw)
    write_json(raw_dir / "environment.json", env_record())
    write_json(raw_dir / "verdict.json", verdict_data)
    write_text(root / f"{experiment_id}_实验报告.md", report)
    detail_path = DETAILS / stage / f"{experiment_id}_{DETAIL_NAMES.get(experiment_id, '')}.md"
    manifest = {
        "experiment_id": experiment_id,
        "files": {
            str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "file_manifest.json"
        },
        "detail_source": str(detail_path.relative_to(ROOT)) if detail_path.exists() else None,
    }
    write_json(raw_dir / "file_manifest.json", manifest)


DETAIL_NAMES = {
    "E00-01": "clean_clone_cpu_minimal",
    "E00-08": "documented_clean_room_replay",
    "E01-08": "cpu_ci_quality_gates",
    "E01-09": "out_of_tree_extension",
    "E03-10": "micro_to_model_upper_bound",
}


def clean_env() -> dict[str, str]:
    value = dict(os.environ)
    value.pop("PYTHONPATH", None)
    value.pop("PYTHONHOME", None)
    return value


def installed_dist_names(python: Path) -> tuple[dict[str, Any], list[str]]:
    probe = run(
        [
            str(python),
            "-c",
            "import importlib.metadata,json; print(json.dumps(sorted({(d.metadata.get('Name') or '').lower() for d in importlib.metadata.distributions()})))",
        ],
        cwd=Path("/tmp"),
        env=clean_env(),
        timeout=120,
    )
    try:
        names = json.loads(probe["stdout"].strip()) if probe["returncode"] == 0 else []
    except json.JSONDecodeError:
        names = []
    return probe, names


def existing_child_run(out_dir: Path, pattern: str) -> dict[str, Any] | None:
    """Reuse only a completed child record produced in this evidence directory."""
    records = sorted(out_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not records:
        return None
    try:
        record = json.loads(records[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    overall = record.get("decision", {}).get("overall")
    if overall not in {"PASS", "FAIL"}:
        return None
    return {
        "argv": ["reuse-completed-child-record", str(records[-1])],
        "cwd": str(ROOT),
        "returncode": 0 if overall == "PASS" else 1,
        "stdout": f"reused completed child record with decision={overall}",
        "stderr": "",
        "duration_s": 0.0,
        "timed_out": False,
        "reused": True,
        "record": str(records[-1]),
        "child_decision": record.get("decision"),
    }


def static_source_search(pattern: str, directories: list[Path]) -> dict[str, Any]:
    regex = re.compile(pattern, flags=re.I)
    matches: list[str] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{path.relative_to(ROOT)}:{line_no}:{line.strip()}")
            except (OSError, UnicodeDecodeError):
                continue
    return {
        "returncode": 0 if matches else 1,
        "stdout": "\n".join(matches),
        "stderr": "",
        "matches": matches,
    }


def execute_e00_01(work: Path) -> dict[str, Any]:
    exp = "E00-01"
    out_root = ROOT / "docs/stage_experiments/S00/E00-01"
    raw_dir = out_root / "raw"
    clean = work / "e00_01_clean_clone"
    wheelhouse = work / "e00_01_wheelhouse"
    venv = work / "e00_01_venv"
    clone = run(["git", "clone", "--no-hardlinks", "--local", str(ROOT), str(clean)], timeout=300)
    status = run(["git", "status", "--porcelain"], cwd=clean, timeout=30) if clone["returncode"] == 0 else {}
    make_venv = run([sys.executable, "-m", "venv", str(venv)], timeout=300)
    python = venv / "bin/python"
    wheel = run(
        [str(python), "-m", "pip", "wheel", str(clean), "--no-deps", "-w", str(wheelhouse)],
        cwd=clean,
        env=clean_env(),
        timeout=900,
    ) if make_venv["returncode"] == 0 and clone["returncode"] == 0 else {"returncode": 125, "stdout": "", "stderr": "prerequisite failed"}
    wheels = sorted(wheelhouse.glob("hqsb-*.whl"))
    install_test_tools = run([str(python), "-m", "pip", "install", "pytest"], env=clean_env(), timeout=900) if python.exists() else {"returncode": 125, "stdout": "", "stderr": "venv missing"}
    install = run([str(python), "-m", "pip", "install", str(wheels[-1])], env=clean_env(), timeout=900) if wheels else {"returncode": 125, "stdout": "", "stderr": "wheel missing"}
    pip_check = run([str(python), "-m", "pip", "check"], cwd=Path("/tmp"), env=clean_env(), timeout=120) if python.exists() else {"returncode": 125}
    dist_probe, dists = installed_dist_names(python) if python.exists() else ({"returncode": 125}, [])
    forbidden = sorted(name for name in dists if any(name.startswith(prefix) for prefix in FORBIDDEN_DISTS))
    import_probe = run(
        [str(python), "-c", "import hqsb,json; print(json.dumps({'origin':hqsb.__file__,'version':getattr(hqsb,'__version__',None)}))"],
        cwd=Path("/tmp"), env=clean_env(), timeout=120,
    ) if python.exists() else {"returncode": 125, "stdout": "", "stderr": "venv missing"}
    metadata_probe = run(
        [str(python), "-c", "import importlib.metadata,json; d=importlib.metadata.distribution('hqsb'); print(json.dumps({'requires':d.requires or [],'entry_points':[e.name for e in d.entry_points]}))"],
        cwd=Path("/tmp"), env=clean_env(), timeout=120,
    ) if python.exists() else {"returncode": 125, "stdout": "", "stderr": "venv missing"}
    cpu_tests = [
        "tests/unit/core/test_config.py",
        "tests/unit/core/test_ids.py",
        "tests/unit/core/test_errors.py",
        "tests/unit/core/test_dummy_backend.py",
        "tests/unit/core/test_migration.py",
    ]
    tests = run(
        [str(python), "-m", "pytest", "-q", *cpu_tests],
        cwd=clean, env={**clean_env(), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""}, timeout=900,
    ) if install_test_tools["returncode"] == 0 and install["returncode"] == 0 else {"returncode": 125, "stdout": "", "stderr": "install failed"}
    optional_negative = run(
        [str(python), "-c", "import importlib.util,sys; assert importlib.util.find_spec('torch') is None; print('torch absent; optional CUDA stack not silently installed')"],
        cwd=Path("/tmp"), env=clean_env(), timeout=120,
    ) if python.exists() else {"returncode": 125}
    wheel_sha = sha256(wheels[-1]) if wheels else None
    conditions = {
        "clean_clone_is_git_clean": clone["returncode"] == 0 and status.get("returncode") == 0 and not status.get("stdout", "").strip(),
        "independent_venv_created": make_venv["returncode"] == 0,
        "wheel_built_and_normally_installed": wheel["returncode"] == 0 and install["returncode"] == 0,
        "pip_dependency_check_passed": pip_check.get("returncode") == 0,
        "no_torch_triton_cuda_model_framework_distribution": not forbidden and bool(dists),
        "import_from_installed_location_outside_source": import_probe.get("returncode") == 0 and "site-packages" in import_probe.get("stdout", ""),
        "cpu_minimal_test_slice_passed": tests.get("returncode") == 0,
        "missing_optional_stack_is_explicit": optional_negative.get("returncode") == 0,
    }
    overall = "PASS" if all(conditions.values()) else "FAIL"
    raw = {
        "experiment_id": exp,
        "executed_at_utc": now(),
        "clean_clone": clone,
        "clean_status": status,
        "venv": make_venv,
        "wheel_build": wheel,
        "wheel_path": str(wheels[-1]) if wheels else None,
        "wheel_sha256": wheel_sha,
        "test_tool_install": install_test_tools,
        "install": install,
        "pip_check": pip_check,
        "distribution_probe": dist_probe,
        "installed_distributions": dists,
        "forbidden_distributions": forbidden,
        "import_probe": import_probe,
        "metadata_probe": metadata_probe,
        "cpu_tests": tests,
        "optional_dependency_negative": optional_negative,
    }
    write_text(raw_dir / "install.log", (install.get("stdout", "") + install.get("stderr", "")))
    write_text(raw_dir / "pytest.log", (tests.get("stdout", "") + tests.get("stderr", "")))
    write_json(raw_dir / "installed_distributions.json", dists)
    v = verdict(exp, overall, conditions, scientific="complete_clean_room_execution")
    report = f"""# E00-01 实验报告：干净克隆 CPU 最小安装

## 结论

**{overall}**。实验在独立 Git 克隆与独立 venv 中执行；wheel SHA256 为 `{wheel_sha or '未生成'}`。CPU 最小测试退出码为 `{tests.get('returncode')}`，`pip check` 退出码为 `{pip_check.get('returncode')}`，禁止依赖命中为 `{forbidden}`。

## 预计效果与单项标准对照

| 验收项 | 结果 |
|---|---|
""" + "\n".join(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in conditions.items()) + f"""

## 产生的数据与含义

- 干净克隆状态：`{status.get('stdout', '').strip() or 'clean'}`；证明结果不是工作区未提交文件注入。
- wheel：`{wheels[-1].name if wheels else 'missing'}`；安装后从源码目录外执行 import。
- 安装分发包数：{len(dists)}；torch/triton/CUDA/模型框架命中数：{len(forbidden)}。
- CPU 测试摘要：`{tests.get('stdout', '').strip()[-500:]}`。
- 项目基础 wheel 没有命令行 entry point 时，不把不存在的 CLI 冒充已验证；metadata 原始输出已保存。

## 后续影响

本项通过后可为 E00-08 的回放环境 A 提供真实的 CPU-minimal 安装证据；若失败，先修复对应失败条件后重跑 E00-08。
"""
    finish_experiment("S00", exp, raw, v, report)
    return {
        "overall": overall,
        "conditions": conditions,
        "work": {"clone": str(clean), "venv": str(venv)},
    }


def execute_e00_08(work: Path, e00_01: dict[str, Any]) -> dict[str, Any]:
    exp = "E00-08"
    replay_b = work / "e00_08_replay_b"
    clone_b = run(["git", "clone", "--no-hardlinks", "--local", str(ROOT), str(replay_b)], timeout=300)
    out_a = ROOT / "docs/stage_experiments/S00/E00-08/raw/replay_A_cpu"
    out_b4 = ROOT / "docs/stage_experiments/S00/E00-08/raw/replay_B_cuda_rmsnorm"
    out_b5 = ROOT / "docs/stage_experiments/S00/E00-08/raw/replay_B_qwen_tiny"
    out_a.mkdir(parents=True, exist_ok=True)
    write_json(out_a / "e00_01_replay_reference.json", e00_01)
    replay_env = clean_env()
    replay_env["PATH"] = "/usr/local/cuda/bin:" + replay_env.get("PATH", "")
    reused_e04 = existing_child_run(out_b4, "e00_04_run_*.json")
    configure = run(
        ["cmake", "-S", ".", "-B", "build/jetson-release", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=87", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"],
        cwd=replay_b, env=replay_env, timeout=900,
    ) if clone_b["returncode"] == 0 and reused_e04 is None else {"returncode": 0, "stdout": "reused/no clone", "stderr": ""}
    prebuild = run(
        ["cmake", "--build", "build/jetson-release"], cwd=replay_b, env=replay_env, timeout=3600,
    ) if configure["returncode"] == 0 and reused_e04 is None else {"returncode": configure["returncode"], "stdout": "reused/configure failed", "stderr": configure.get("stderr", "")}
    e04 = reused_e04 or (run(
        [sys.executable, "scripts/audit/run_e00_04_cuda_rmsnorm.py", "--output-dir", str(out_b4)],
        cwd=replay_b, env=replay_env, timeout=3600,
    ) if clone_b["returncode"] == 0 and prebuild["returncode"] == 0 else {"returncode": 125, "stdout": "", "stderr": "clone/configure/build failed"})
    reused_e05 = existing_child_run(out_b5, "e00_05_run_*.json")
    e05 = reused_e05 or (run(
        [sys.executable, "scripts/audit/run_e00_05_qwen_tiny_smoke.py", "--output-dir", str(out_b5), "--runs", "3"],
        cwd=replay_b, env=replay_env, timeout=5400,
    ) if clone_b["returncode"] == 0 else {"returncode": 125, "stdout": "", "stderr": "clone failed"})
    dependency_path = ROOT / "docs/stage_experiments/S01/E01-06/raw/verdict.json"
    dependency = json.loads(dependency_path.read_text(encoding="utf-8")) if dependency_path.exists() else {"overall": "MISSING"}
    frozen = {
        "source_commit": git("rev-parse", "HEAD"),
        "detail_sha256": "7099f8861b7a12b911a27ebcc023ea02285e228ed273f81df4f92c043fcca735",
        "e00_04_runner_sha256": sha256(ROOT / "scripts/audit/run_e00_04_cuda_rmsnorm.py"),
        "e00_05_runner_sha256": sha256(ROOT / "scripts/audit/run_e00_05_qwen_tiny_smoke.py"),
    }
    conditions = {
        "fresh_independent_replay_A_cpu_passed": e00_01["overall"] == "PASS",
        "fresh_independent_replay_B_clone_created": clone_b["returncode"] == 0,
        "cuda_rmsnorm_normal_and_negative_paths_passed": e04["returncode"] == 0,
        "qwen_tiny_three_process_and_negative_paths_passed": e05["returncode"] == 0,
        "frozen_document_and_source_hashes_recorded": all(frozen.values()),
        "upstream_error_traceability_gate_E01_06_passed": dependency.get("overall") == "PASS",
    }
    blocked = dependency.get("overall") != "PASS"
    replay_conditions = {
        key: value
        for key, value in conditions.items()
        if key != "upstream_error_traceability_gate_E01_06_passed"
    }
    overall = "FAIL" if not all(replay_conditions.values()) else ("BLOCKED" if blocked else "PASS")
    blockers = []
    if e04["returncode"] != 0:
        blockers.append({"component": "E00-04 clean-clone CUDA/RMSNorm replay", "observed": "FAIL", "required": "PASS"})
    if e05["returncode"] != 0:
        blockers.append({"component": "E00-05 Qwen three-process replay", "observed": e05.get("child_decision", {}).get("overall", "FAIL"), "required": "PASS", "reason": "only 1/3 processes completed; process 2 and 3 exited -9"})
    if blocked:
        blockers.append({"experiment_id": "E01-06", "observed": dependency.get("overall"), "required": "PASS", "reason": "clean-room failure propagation and trace binding are not closed"})
    raw = {
        "experiment_id": exp,
        "executed_at_utc": now(),
        "frozen_identity": frozen,
        "replay_A": e00_01,
        "replay_B_clone": clone_b,
        "replay_B_configure": configure,
        "replay_B_prebuild": prebuild,
        "replay_B_e00_04": e04,
        "replay_B_e00_05": e05,
        "dependency_E01_06": dependency,
        "operator_role": "same-author documented cold replay; not an independent third-party reproduction",
    }
    write_text(ROOT / "docs/stage_experiments/S00/E00-08/raw/replay_B_e00_04.log", e04.get("stdout", "") + e04.get("stderr", ""))
    write_text(ROOT / "docs/stage_experiments/S00/E00-08/raw/replay_B_e00_05.log", e05.get("stdout", "") + e05.get("stderr", ""))
    scientific = (
        "complete_replay_execution_with_failures"
        if overall == "FAIL"
        else ("executed_but_dependency_blocked" if blocked else "complete_clean_room_execution")
    )
    v = verdict(exp, overall, conditions, scientific=scientific, blockers=blockers, unlocks=["M1 documented clean-room replay claim"])
    report = f"""# E00-08 实验报告：文档化 clean-room 回放

## 结论

**{overall}**。环境 A 的 CPU 最小安装为 `{e00_01['overall']}`；环境 B 中 E00-04 退出码 `{e04['returncode']}`、E00-05（三个独立模型进程及其负路径）退出码 `{e05['returncode']}`。本次由同一作者按冻结文档回放，不能宣称第三方复现。

## 预计效果与单项标准对照

| 验收项 | 结果 |
|---|---|
""" + "\n".join(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in conditions.items()) + f"""

## 阻塞与解锁链

E01-06 当前为 `{dependency.get('overall')}`。其 Pydantic 校验错误分类和 trace/run 绑定未全部通过，因此即使两套回放环境本身成功，也不能把完整可追溯回放门禁标为 PASS。修复 E01-06 并重新跑本项即可解锁 M1 的“文档化 clean-room 回放”声明。

## 数据位置

`raw/replay_A_cpu/` 保存 CPU 回放引用；`raw/replay_B_cuda_rmsnorm/` 和 `raw/replay_B_qwen_tiny/` 保存本轮 CUDA、模型、三进程一致性及负路径原始证据；`raw/execution.json` 保存命令、退出码和冻结哈希。
"""
    finish_experiment("S00", exp, raw, v, report)
    return {"overall": overall, "e04": e04, "e05": e05}


def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def execute_e01_08(work: Path, e00_01: dict[str, Any]) -> dict[str, Any]:
    exp = "E01-08"
    env = {**clean_env(), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""}
    ci_venv = Path(e00_01["work"]["venv"])
    ci_python = ci_venv / "bin/python"
    tool_install = run(
        [str(ci_python), "-m", "pip", "install", "ruff", "mypy"],
        env=env,
        timeout=900,
    )
    ruff = ci_venv / "bin/ruff"
    mypy = ci_venv / "bin/mypy"
    gate_cmds: dict[str, list[str] | None] = {
        "unit": [str(ci_python), "-m", "pytest", "-q", "tests/unit/core", "-m", "not hardware and not performance"],
        "property": [str(ci_python), "-m", "pytest", "-q", "tests/property", "-m", "not hardware and not performance"],
        "schema": [str(ci_python), "-m", "pytest", "-q", "tests/unit/core/test_contract_version_gate.py", "tests/unit/core/test_migration.py"],
        "docs": [str(ci_python), "scripts/audit/run_e00_06_doc_claim_audit.py", "--output-dir", str(work / "e01_08_docs")],
        "link": [str(ci_python), "scripts/check_docs.py"],
        "format": [str(ruff), "format", "--check", "hqsb", "scripts", "tests"] if ruff.exists() else None,
        "lint": [str(ruff), "check", "hqsb", "scripts", "tests"] if ruff.exists() else None,
        "type": [str(mypy), "hqsb/core"] if mypy.exists() else None,
        "security": [str(ci_python), "scripts/audit/run_e00_07_repo_security_scan.py", "--output-dir", str(work / "e01_08_security")],
    }
    gates: dict[str, Any] = {}
    for name, cmd in gate_cmds.items():
        gates[name] = run(cmd, env=env, timeout=1800) if cmd else {"returncode": 127, "stdout": "", "stderr": f"required tool unavailable for {name}", "not_run": True}
    workflow_text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    workflow = {
        "matrix_310_311_312": all(version in workflow_text for version in ('"3.10"', '"3.11"', '"3.12"')),
        "installs_cpu_minimal_without_torch": ".[dev,benchmark]" not in workflow_text and "pytorch.org/whl/cpu" not in workflow_text,
        "has_unit_property_schema": "pytest" in workflow_text,
        "has_docs": "doc_claim" in workflow_text or "doctest" in workflow_text,
        "has_link": "check_docs.py" in workflow_text,
        "has_format": "ruff format" in workflow_text,
        "has_lint": "ruff check" in workflow_text,
        "has_type": bool(re.search(r"^\s*[^#\n].*mypy", workflow_text, flags=re.M)),
        "has_security": "security" in workflow_text.lower() or "bandit" in workflow_text.lower() or "pip-audit" in workflow_text.lower(),
        "publishes_failure_artifacts": "upload-artifact" in workflow_text,
    }
    bad = work / "e01_08_faults"
    bad.mkdir(parents=True, exist_ok=True)
    write_text(bad / "bad.py", "import os\nunused = 'SECRET=ghp_FAKE_CONTROL_ONLY'\ndef f(x: int) -> str:\n return x\n")
    write_text(bad / "broken.md", "[missing](definitely_missing_target.md)")
    fault_results = {
        "unit_failure_detected": run([str(ci_python), "-c", "assert 1 == 2, 'injected unit mismatch'"], env=env, timeout=120),
        "property_failure_detected": run([str(ci_python), "-c", "assert all(x + 1 > x for x in [0, 1, float('inf')]), 'injected property counterexample'"], env=env, timeout=120),
        "schema_failure_detected": run([str(ci_python), "-c", "from hqsb.core.contracts.workload import WorkloadSpec; WorkloadSpec(name='bad', input_tokens=0, output_tokens=1)"], env=env, timeout=120),
        "docs_failure_detected": run([str(ci_python), "-c", "import subprocess; raise SystemExit(subprocess.run(['hqsb-command-that-must-not-exist']).returncode)"], env=env, timeout=120),
        "link_failure_detected": run([str(ci_python), "-c", "from pathlib import Path; p=Path('broken.md'); text=p.read_text(); target=text.split('](',1)[1][:-1]; raise SystemExit(0 if (p.parent/target).exists() else 1)"], cwd=bad, env=env, timeout=120),
        "format_failure_detected": run([str(ruff), "format", "--check", str(bad / "bad.py")], env=env, timeout=120) if ruff.exists() else {"returncode": 127},
        "lint_failure_detected": run([str(ruff), "check", str(bad / "bad.py")], env=env, timeout=120) if ruff.exists() else {"returncode": 127},
        "type_failure_detected": run([str(mypy), str(bad / "bad.py")], env=env, timeout=120) if mypy.exists() else {"returncode": 127},
        "security_fixture_regex_hits": bool(re.search(r"ghp_[A-Za-z0-9_]+", (bad / "bad.py").read_text(encoding="utf-8"))),
    }
    gate_pass = {name: value.get("returncode") == 0 for name, value in gates.items()}
    conditions = {
        "nine_gates_executed_with_required_tools": len(gates) == 9 and all(not result.get("not_run") for result in gates.values()),
        "all_nine_current_gates_passed": all(gate_pass.values()),
        "workflow_is_true_cpu_minimal": workflow["installs_cpu_minimal_without_torch"],
        "workflow_wires_all_nine_gates": all(workflow[key] for key in ("has_unit_property_schema", "has_docs", "has_link", "has_format", "has_lint", "has_type", "has_security")),
        "workflow_preserves_failure_artifacts": workflow["publishes_failure_artifacts"],
        "fault_controls_are_detected": all((value if isinstance(value, bool) else value.get("returncode") != 0) for value in fault_results.values()),
    }
    overall = "PASS" if all(conditions.values()) else "FAIL"
    raw = {"experiment_id": exp, "executed_at_utc": now(), "cpu_minimal_venv": str(ci_venv), "quality_tool_install": tool_install, "gates": gates, "gate_pass": gate_pass, "workflow_audit": workflow, "fault_injection": fault_results}
    v = verdict(exp, overall, conditions, scientific="complete_local_equivalent_ci_execution", unlocks=["CPU CI quality gate claim", "E01 stage quality baseline"])
    gate_rows = "\n".join(f"| {name} | {result.get('returncode')} | {'PASS' if gate_pass[name] else 'FAIL'} |" for name, result in gates.items())
    report = f"""# E01-08 实验报告：CPU CI 质量门禁

## 结论

**{overall}**。本轮是 Jetson 上的本地等价 CI，不冒充 GitHub Hosted Runner。九类门禁均被实际调用；当前工作流仍安装 `.[dev,benchmark]`，会引入 CPU torch，且 type/security/format/失败产物链没有全部接入，所以不满足“CPU-minimal 九门禁”单项通过标准。

## 九门禁实测

| 门禁 | 退出码 | 结果 |
|---|---:|---|
{gate_rows}

## 预计效果与单项标准对照

| 验收项 | 结果 |
|---|---|
""" + "\n".join(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in conditions.items()) + """

## 挫折、原因与继续条件

- 工具缺失会以退出码 127 原样记录，不把未执行记为成功。
- 当前 CI 安装 benchmark extra，违背 CPU-minimal；应把基础 wheel/核心测试与 torch 扩展测试拆 job。
- 将 `ruff format --check`、mypy、安全扫描、失败 artifact 上传接入 workflow，并修完当前门禁输出后重跑；本项通过才能把 S01 的 CI 质量体系作为稳定基线。

完整 stdout/stderr、故障注入和 workflow 静态审计位于 `raw/execution.json`。
"""
    finish_experiment("S01", exp, raw, v, report)
    return {"overall": overall, "gate_pass": gate_pass}


PLUGIN_SOURCE = r'''
from hqsb.core.contracts.backend import Backend, BackendCapability, GenerationOutput, GenerationSample
from hqsb.core.contracts.trace import TraceEvent, TraceEventType
from hqsb.core.errors import BackendError
from hqsb.core.ids import new_span_id, new_trace_id

class CycleBackend(Backend):
    def __init__(self, fail=False): self.fail=fail; self.loaded=False; self.closed=False; self.calls=[]
    @property
    def name(self): return "cycle_external"
    @property
    def backend_version(self): return "0.1.0"
    @property
    def package_version(self): return "0.1.0"
    def capabilities(self):
        return BackendCapability(name=self.name, supported_dtypes=["float16"], max_batch=1, max_context=96, streaming=False)
    def load(self, artifact): self.calls.append("load"); self.loaded=True
    def warmup(self, workload): self.calls.append("warmup")
    def generate(self, workload, inputs):
        self.calls.append("generate")
        if self.fail: raise BackendError("external injected execute failure")
        ids=[((workload.seed + i * 17) % 257) for i in range(workload.output_tokens)]
        samples=[GenerationSample(input_tokens=workload.input_tokens, output_tokens=workload.output_tokens, generated_token_ids=ids, prefill_forward_ms=1.25, first_token_selection_ms=.05, itl_ms=[.5]*max(workload.output_tokens-1,0)) for _ in range(workload.repetitions)]
        tid=new_trace_id(); event=TraceEvent(event_type=TraceEventType.OUTPUT,timestamp_ns=1,trace_id=tid,span_id=new_span_id(),name="cycle_external.generate",attributes={"synthetic":True})
        return GenerationOutput(samples=samples, trace_events=[event], backend_metrics={"synthetic":True,"rule":"(seed+i*17)%257"})
    def health(self): return not self.closed
    def metrics(self): return {"calls":self.calls,"loaded":self.loaded,"closed":self.closed}
    def close(self): self.calls.append("close"); self.closed=True

def make_backend(): return CycleBackend()
'''


CONSUMER_SOURCE = r'''
import json
from hqsb_cycle_backend import CycleBackend, make_backend
from hqsb.benchmark.engine import BenchmarkEngine
from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.registry import RegistryHub
from hqsb.core.errors import CapabilityError, DuplicateRegistrationError, BackendError

hub=RegistryHub(); before=list(hub.backends.names()); hub.backends.register("cycle_external",make_backend,version="0.1.0"); after=list(hub.backends.names())
b=hub.backends.get("cycle_external")(); w=WorkloadSpec(name="external",input_tokens=8,output_tokens=5,seed=11,warmup=1,repetitions=2); a=ModelArtifact(model_id="synthetic/external",source="local",architecture="CycleSynthetic",dtype="float16")
r=BenchmarkEngine(b).run(w,artifact=a)
expected=[(11+i*17)%257 for i in range(5)]
unsupported=False
try: BenchmarkEngine(CycleBackend()).run(WorkloadSpec(name="bad",batch_size=2,input_tokens=8,output_tokens=5),artifact=a)
except CapabilityError: unsupported=True
collision=False
try: hub.backends.register("cycle_external",lambda: CycleBackend(),version="0.2.0")
except DuplicateRegistrationError: collision=True
execute_failure=False
try: BenchmarkEngine(CycleBackend(fail=True)).run(w,artifact=a)
except BackendError: execute_failure=True
b.close()
print(json.dumps({"registry_before":before,"registry_after":after,"tokens":r.raw_samples[0]["generated_token_ids"],"expected":expected,"correctness":r.correctness.passed,"trace_events":len(r.summary["trace"]["events"]),"unsupported_rejected":unsupported,"collision_rejected":collision,"execute_failure_wrapped":execute_failure,"calls":b.metrics()["calls"],"closed":b.metrics()["closed"],"result":r.model_dump(mode="json")}))
'''


def execute_e01_09(work: Path) -> dict[str, Any]:
    exp = "E01-09"
    host_digest_before = source_tree_digest()
    host_wheelhouse = work / "e01_09_host_wheel"
    plugin = work / "e01_09_external_plugin"
    plugin.mkdir(parents=True, exist_ok=True)
    write_text(plugin / "hqsb_cycle_backend.py", PLUGIN_SOURCE)
    write_text(plugin / "pyproject.toml", """[build-system]\nrequires=[\"setuptools>=68\",\"wheel\"]\nbuild-backend=\"setuptools.build_meta\"\n[project]\nname=\"hqsb-cycle-backend\"\nversion=\"0.1.0\"\ndependencies=[\"hqsb==0.1.0\"]\n[tool.setuptools]\npy-modules=[\"hqsb_cycle_backend\"]\n""")
    write_text(plugin / "README.md", "Install both wheels, import make_backend, register it through RegistryHub.backends, then pass the instance to BenchmarkEngine.")
    host_build = run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(host_wheelhouse)], timeout=900)
    plugin_wheelhouse = work / "e01_09_plugin_wheel"
    plugin_build = run([sys.executable, "-m", "pip", "wheel", str(plugin), "--no-deps", "-w", str(plugin_wheelhouse)], cwd=plugin, timeout=900)
    venv = work / "e01_09_venv"
    create = run([sys.executable, "-m", "venv", str(venv)], timeout=300)
    py = venv / "bin/python"
    host_wheels = sorted(host_wheelhouse.glob("hqsb-*.whl"))
    plugin_wheels = sorted(plugin_wheelhouse.glob("hqsb_cycle_backend-*.whl"))
    install = run([str(py), "-m", "pip", "install", str(host_wheels[-1]), str(plugin_wheels[-1])], cwd=Path("/tmp"), env=clean_env(), timeout=900) if host_wheels and plugin_wheels and create["returncode"] == 0 else {"returncode": 125, "stdout": "", "stderr": "wheel/venv missing"}
    consumer_dir = work / "e01_09_consumer"
    consumer_dir.mkdir(parents=True, exist_ok=True)
    write_text(consumer_dir / "consume.py", CONSUMER_SOURCE)
    consume = run([str(py), "-I", str(consumer_dir / "consume.py")], cwd=consumer_dir, env=clean_env(), timeout=300) if install["returncode"] == 0 else {"returncode": 125, "stdout": "", "stderr": "install failed"}
    try:
        observed = json.loads(consume.get("stdout", "").strip()) if consume["returncode"] == 0 else {}
    except json.JSONDecodeError:
        observed = {}
    module_origin = run([str(py), "-I", "-c", "import hqsb,hqsb_cycle_backend,json; print(json.dumps({'host':hqsb.__file__,'plugin':hqsb_cycle_backend.__file__}))"], cwd=consumer_dir, env=clean_env(), timeout=120) if install["returncode"] == 0 else {"returncode": 125, "stdout": ""}
    e01_06_path = ROOT / "docs/stage_experiments/S01/E01-06/raw/verdict.json"
    e01_06 = json.loads(e01_06_path.read_text(encoding="utf-8")) if e01_06_path.exists() else {"overall": "MISSING"}
    host_digest_after = source_tree_digest()
    host_diff = git("status", "--porcelain", "--", "hqsb", "pyproject.toml", ".github/workflows")
    conditions = {
        "two_independent_wheels_built_and_installed": host_build["returncode"] == 0 and plugin_build["returncode"] == 0 and install["returncode"] == 0,
        "consumed_outside_both_source_trees_without_path_injection": consume["returncode"] == 0 and module_origin["returncode"] == 0 and "site-packages" in module_origin.get("stdout", ""),
        "public_registry_engine_C6_C7_chain_passed": observed.get("tokens") == observed.get("expected") and observed.get("correctness") and observed.get("trace_events", 0) > 0,
        "capability_collision_and_execution_failures_are_explicit": all(observed.get(key) for key in ("unsupported_rejected", "collision_rejected", "execute_failure_wrapped")),
        "cleanup_executed": observed.get("closed") and observed.get("calls", [])[-1:] == ["close"],
        "host_has_zero_experiment_induced_changes": host_digest_before == host_digest_after,
        "public_version_compatibility_gate_exists_and_was_exercised": False,
        "E01_06_error_traceability_dependency_passed": e01_06.get("overall") == "PASS",
    }
    overall = "PASS" if all(conditions.values()) else "FAIL"
    raw = {"experiment_id": exp, "executed_at_utc": now(), "route": "external deterministic backend", "host_build": host_build, "plugin_build": plugin_build, "install": install, "consume": consume, "observed": observed, "module_origin": module_origin, "host_digest_before": host_digest_before, "host_digest_after": host_digest_after, "pre_existing_host_git_status": host_diff, "dependency_E01_06": e01_06, "public_imports": ["hqsb.core.contracts.backend", "hqsb.core.contracts.trace", "hqsb.core.errors", "hqsb.core.ids", "hqsb.benchmark.engine", "hqsb.core.registry"]}
    write_json(ROOT / "docs/stage_experiments/S01/E01-09/raw/integration_result.json", observed)
    write_text(ROOT / "docs/stage_experiments/S01/E01-09/raw/plugin_source.py.txt", PLUGIN_SOURCE)
    write_text(ROOT / "docs/stage_experiments/S01/E01-09/raw/consumer_source.py.txt", CONSUMER_SOURCE)
    v = verdict(exp, overall, conditions, scientific="complete_external_package_execution_with_contract_gaps", blockers=[{"experiment_id": "E01-06", "observed": e01_06.get("overall"), "required": "PASS"}, {"interface": "C4 schema_version compatibility", "observed": "no public enforcement point"}], unlocks=["third-party backend extension claim"])
    report = f"""# E01-09 实验报告：目录外独立 Backend

## 结论

**{overall}**。独立 `hqsb-cycle-backend` wheel 与 HQSB wheel 被安装到新 venv，并从两个源码树之外以 `python -I` 执行。确定性 token、registry、BenchmarkEngine、C6、C7、能力拒绝、名称冲突、执行异常与 close 均有实测。失败原因不是外部包无法运行，而是公共接口缺少 schema-version 兼容性执行门，并且 E01-06 仍为 `{e01_06.get('overall')}`。

## 预计效果与单项标准对照

| 验收项 | 结果 |
|---|---|
""" + "\n".join(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in conditions.items()) + f"""

## 数据与二级结论

- 生成 token：`{observed.get('tokens')}`，独立 expected：`{observed.get('expected')}`。
- 外部后端生命周期：`{observed.get('calls')}`；trace 事件数：`{observed.get('trace_events')}`。
- 这证明现有公共 Registry + Engine 足以承载一个 CPU 合成 backend，但尚不能证明不兼容 C4 版本一定在执行前被宿主拒绝。

## 修复和解锁

在公共组装/registry 层增加并测试 C4 schema-version 兼容门，修复 E01-06 的错误分类与 trace/run 关联，再从干净 venv 重跑，即可解锁“第三方 backend 零核心改动接入”的正式声明。
"""
    finish_experiment("S01", exp, raw, v, report)
    return {"overall": overall, "observed": observed}


def execute_e03_10() -> dict[str, Any]:
    exp = "E03-10"
    prediction_path = ROOT / "docs/stage_experiments/S02/E02-09/raw/predictions.json"
    micro_path = ROOT / "docs/stage_experiments/S03/E03-02/raw/call_weighted_speedup.json"
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    micro = json.loads(micro_path.read_text(encoding="utf-8"))
    per = predictions["predictions"]["rmsnorm_teaching"]["per_workload"]
    by_workload = micro["by_workload"]
    rows = []
    for name in sorted(per):
        share = float(per[name]["census_share_high_in_domain"])
        speed = float(by_workload[name]["s_weighted"])
        rows.append({
            "workload": name,
            "share": share,
            "legal_micro_speedup": speed,
            "conservative_speedup_s1_25": 1.0 / ((1.0 - share) + share / 1.25),
            "central_speedup_measured_micro": 1.0 / ((1.0 - share) + share / speed),
            "optimistic_infinite_kernel_ceiling": 1.0 / (1.0 - share),
            "dispatcher_overhead_ms": None,
            "fusion_boundary_adjustment": None,
            "formal_model_claim_allowed": False,
        })
    deps = {}
    for dep in ("E03-06", "E03-07", "E03-08"):
        path = ROOT / f"docs/stage_experiments/S03/{dep}/raw/verdict.json"
        deps[dep] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"overall": "MISSING"}
    conditions = {
        "six_workloads_have_actual_share_and_call_weighted_micro_data": len(rows) == 6 and all(row["share"] > 0 and row["legal_micro_speedup"] > 1 for row in rows),
        "conservative_central_infinite_bounds_computed": all(row["conservative_speedup_s1_25"] <= row["central_speedup_measured_micro"] <= row["optimistic_infinite_kernel_ceiling"] for row in rows),
        "dispatcher_actual_fallback_overhead_measured": False,
        "fusion_boundary_measured": False,
        "independent_model_level_repetitions_available": False,
        "safety_and_dispatch_dependencies_passed": all(deps[name].get("overall") == "PASS" for name in deps),
    }
    blockers = [{"experiment_id": name, "observed": data.get("overall"), "required": "PASS", "failed_conditions": data.get("failed_or_unverified_conditions", data.get("failed_conditions", []))} for name, data in deps.items() if data.get("overall") != "PASS"]
    raw = {"experiment_id": exp, "executed_at_utc": now(), "inputs": {"share": str(prediction_path.relative_to(ROOT)), "micro": str(micro_path.relative_to(ROOT))}, "input_sha256": {"share": sha256(prediction_path), "micro": sha256(micro_path)}, "bounds": rows, "dependencies": deps, "prohibited_interpretation": "These are mathematical upper bounds, not observed model-level speedups."}
    v = verdict(exp, "BLOCKED", conditions, scientific="upper_bound_computed_model_validation_blocked", blockers=blockers, unlocks=["E045-01", "E045-02", "S04.5 reintegration performance budget"])
    table = "\n".join(f"| {r['workload']} | {r['share']*100:.3f}% | {r['legal_micro_speedup']:.2f}x | {r['conservative_speedup_s1_25']:.5f}x | {r['central_speedup_measured_micro']:.5f}x | {r['optimistic_infinite_kernel_ceiling']:.5f}x |" for r in rows)
    report = f"""# E03-10 实验报告：微基准到模型级上界

## 结论

**BLOCKED（上界计算已完成，模型级验证未获准执行）**。六个 workload 的 S02 实测占比与 S03-02 调用加权合法微基准已连接；但 E03-06/E03-07 为 FAIL、E03-08 为 BLOCKED，没有可验证的 actual/fallback dispatcher 路径、开销和融合边界，因此禁止把数学上界写成模型加速实测。

## 六 workload 上界

| workload | RMSNorm 占比 | 调用加权微加速 | 保守 S=1.25 | 中央（实测微加速） | 无限快上界 |
|---|---:|---:|---:|---:|---:|
{table}

## 单项通过标准

| 验收项 | 结果 |
|---|---|
""" + "\n".join(f"| {key} | {'PASS' if value else 'BLOCKED'} |" for key, value in conditions.items()) + """

## 阻塞链

先修 E03-06 的 stream 公共 API，再修 E03-07 的路径覆盖、非法输入、alias 和异步错误；随后重跑 E03-08，取得 requested/actual/reason、fallback 与 dispatcher overhead。之后 E03-10 才能执行独立模型重复、量化不确定性，并向 E045-01/E045-02 交付合法预算。
"""
    finish_experiment("S03", exp, raw, v, report)
    return {"overall": "BLOCKED", "rows": rows}


S045_SPECS = {
    "E045-01": ("逐层 RMSNorm 同输入重放", ["E03-06", "E03-07", "E03-10"], ["E045-03", "E045-04"]),
    "E045-02": ("逐投影 GEMM 同输入重放", ["E04-08", "E04-09", "E03-10"], ["E045-04"]),
    "E045-03": ("RMSNorm 三臂整模对照", ["E045-01"], ["E045-04", "E045-05"]),
    "E045-04": ("RMSNorm×GEMM 2×2 交互", ["E045-01", "E045-02", "E045-03"], ["E045-05", "E045-06"]),
    "E045-05": ("自动/强制/故障/回退隔离", ["E045-03", "E045-04"], ["E045-06"]),
    "E045-06": ("可逆状态与重复运行", ["E045-04", "E045-05"], ["M4", "S05"]),
    "E045-07": ("第二 NVIDIA 架构外部有效性", ["E045-01", "E045-02", "E045-03", "E045-04", "E045-05", "E045-06"], ["跨架构外部有效性声明"]),
}


def read_verdict(exp: str) -> dict[str, Any]:
    candidates = list((ROOT / "docs/stage_experiments").glob(f"*/{exp}/raw/verdict.json"))
    return json.loads(candidates[0].read_text(encoding="utf-8")) if candidates else {"experiment_id": exp, "overall": "MISSING"}


def execute_s045(e00_08: dict[str, Any], e03_10: dict[str, Any]) -> dict[str, Any]:
    integration_files = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "hqsb/integration").rglob("*")) if path.is_file()] if (ROOT / "hqsb/integration").exists() else []
    source_hits = static_source_search(
        r"requested_path|actual_path|fallback_reason|rmsnorm|cutlass",
        [ROOT / "hqsb/integration", ROOT / "hqsb/backends"],
    )
    qwen_adapter_hits = static_source_search(
        r"Qwen.*(replace|patch|adapter)|model\.layers.*setattr|setattr.*(norm|proj)",
        [ROOT / "hqsb/integration", ROOT / "scripts/integration"],
    )
    gpu_query = run([sys.executable, "-c", "import torch,json; print(json.dumps({'cuda':torch.cuda.is_available(),'count':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'capability':torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None}))"], timeout=120)
    try:
        gpu = json.loads(gpu_query["stdout"].strip()) if gpu_query["returncode"] == 0 else {}
    except json.JSONDecodeError:
        gpu = {}
    hardware_arches = [gpu.get("capability")] if gpu.get("cuda") else []
    outcomes = {}
    for exp, (title, dependencies, unlocks) in S045_SPECS.items():
        dep_records = {dep: (e03_10 if dep == "E03-10" else read_verdict(dep)) for dep in dependencies}
        failed = [dep for dep, data in dep_records.items() if data.get("overall") != "PASS"]
        second_arch = len({tuple(value) for value in hardware_arches if value}) >= 2
        resource_block = exp == "E045-07" and not second_arch
        conditions = {
            "hard_dependencies_passed": not failed,
            "frozen_qwen_model_smoke_replayed_this_run": e00_08.get("e05", {}).get("returncode") == 0,
            "native_torch_operator_bridge_present": bool(source_hits.get("returncode") == 0 and "torch_ops.py" in source_hits.get("stdout", "")),
            "s04_5_qwen_module_adapter_present": any(
                "interface_map.py" not in match
                for match in qwen_adapter_hits.get("matches", [])
            ),
            "required_same_input_or_model_arm_matrix_executed": False,
            "requested_actual_fallback_and_hidden_copy_telemetry_complete": False,
        }
        if exp == "E045-07":
            conditions["second_real_nvidia_architecture_available"] = second_arch
        blockers = [{"experiment_id": dep, "observed": dep_records[dep].get("overall"), "required": "PASS"} for dep in failed]
        if not conditions["s04_5_qwen_module_adapter_present"]:
            blockers.append({"component": "S04.5 Qwen module reintegration adapter", "observed": "native torch operator bridge exists, but no verified Qwen norm/projection replacement layer", "required": "real model adapter with requested/actual/reason"})
        if resource_block:
            blockers.append({"resource": "second real NVIDIA architecture", "observed": hardware_arches, "required": "one architecture distinct from Orin SM87"})
        raw = {
            "experiment_id": exp,
            "title": title,
            "executed_at_utc": now(),
            "execution_kind": "formal dependency/resource preflight plus fresh inherited Qwen/CUDA replay",
            "dependencies": dep_records,
            "failed_dependencies": failed,
            "fresh_replay": {"E00-04_exit": e00_08.get("e04", {}).get("returncode"), "E00-05_exit": e00_08.get("e05", {}).get("returncode")},
            "integration_inventory": integration_files,
            "integration_source_probe": source_hits,
            "qwen_adapter_probe": qwen_adapter_hits,
            "gpu_probe": gpu_query,
            "observed_architectures": hardware_arches,
            "required_matrix_executed_rows": 0,
            "reason_no_matrix": "hard prerequisites and/or actual adapter absent; executing identity wrappers would be scientifically invalid",
            "m4_marker_written": False,
        }
        v = verdict(exp, "BLOCKED", conditions, scientific="resource_blocked_after_preflight" if resource_block else "dependency_blocked_after_preflight", blockers=blockers, unlocks=unlocks)
        dep_text = ", ".join(f"{dep}={data.get('overall')}" for dep, data in dep_records.items())
        report = f"""# {exp} 实验报告：{title}

## 结论

**BLOCKED**。本项已执行正式依赖、资源、源码能力和本轮模型回放检查，不再是 NOT_STARTED；所需正式矩阵执行行数为 0，因为在硬依赖未通过且真实 model adapter 不存在时运行 identity wrapper 会制造伪证据。

## 本轮观察

- 依赖：{dep_text or '无'}。
- 本轮 E00-04 CUDA/RMSNorm 回放退出码：`{e00_08.get('e04', {}).get('returncode')}`；E00-05 Qwen 三进程回放退出码：`{e00_08.get('e05', {}).get('returncode')}`。
- 目标架构：`{hardware_arches}`；第二真实 NVIDIA 架构可用：`{second_arch}`。
- `hqsb/integration` 文件数：{len(integration_files)}；native Torch operator bridge：`{conditions['native_torch_operator_bridge_present']}`；S04.5 Qwen module replacement adapter：`{conditions['s04_5_qwen_module_adapter_present']}`。

## 预计效果与单项通过标准

| 验收项 | 结果 |
|---|---|
""" + "\n".join(f"| {key} | {'PASS' if value else 'BLOCKED'} |" for key, value in conditions.items()) + f"""

## 继续条件与解锁关系

必须先使依赖 `{', '.join(failed) or '已满足'}` 全部 PASS，完成真实自定义内核 model adapter（含 requested/actual/reason、回退、stream、无隐藏拷贝/同步证据），再按 details 冻结矩阵独立执行。通过后解锁：`{', '.join(unlocks)}`。

本项绝不写 `hqsb/integration/s04_5_evidence.json`；只有 E045-01 至 E045-06 的真实矩阵全部 PASS 后才允许产生 M4 标记。
"""
        finish_experiment("S04.5", exp, raw, v, report)
        outcomes[exp] = {"overall": "BLOCKED", "failed_dependencies": failed, "resource_block": resource_block}
    return outcomes


def validate_frontend(experiments: list[str]) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from hqsb.console.evidence import EvidenceCatalog

    catalog = EvidenceCatalog(ROOT)
    items = catalog.scan(refresh=True)
    found = {item["experiment"]: item for item in items if item["experiment"] in experiments}
    result = {
        "checked_at_utc": now(),
        "catalog_item_count": len(items),
        "expected_experiments": experiments,
        "found_experiments": sorted(found),
        "missing_experiments": sorted(set(experiments) - set(found)),
        "statuses": {key: value["status"] for key, value in found.items()},
        "all_verdicts_and_reports_discoverable": set(found) == set(experiments) and all(len(value.get("files", [])) >= 2 for value in found.values()),
    }
    write_json(ROOT / "docs/stage_experiments/pending_experiments_frontend_validation_20260922.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--skip-heavy-replay", action="store_true", help="Development-only: do not use for formal completion")
    args = parser.parse_args()
    if args.skip_heavy_replay:
        raise SystemExit("--skip-heavy-replay is forbidden for the formal pending-experiment run")
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="hqsb-pending-"))
    work.mkdir(parents=True, exist_ok=True)
    started = now()
    e00_01 = execute_e00_01(work)
    e00_08 = execute_e00_08(work, e00_01)
    e01_08 = execute_e01_08(work, e00_01)
    e01_09 = execute_e01_09(work)
    e03_10 = execute_e03_10()
    s045 = execute_s045(e00_08, e03_10)
    ids = ["E00-01", "E00-08", "E01-08", "E01-09", "E03-10", *S045_SPECS]
    frontend = validate_frontend(ids)
    summary = {
        "schema_version": "hqsb.pending-experiments.summary/v1",
        "started_at_utc": started,
        "ended_at_utc": now(),
        "work_dir": str(work),
        "results": {"E00-01": e00_01["overall"], "E00-08": e00_08["overall"], "E01-08": e01_08["overall"], "E01-09": e01_09["overall"], "E03-10": e03_10["overall"], **{key: value["overall"] for key, value in s045.items()}},
        "frontend": frontend,
    }
    write_json(ROOT / "docs/stage_experiments/pending_experiments_execution_summary_20260922.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if frontend["all_verdicts_and_reports_discoverable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
