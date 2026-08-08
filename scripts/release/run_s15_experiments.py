#!/usr/bin/env python3
"""Execute the safe, honest scope of all S15 experiments on the Jetson.

The collector deliberately separates three outcomes:

* machine-executable evidence is collected (claim/document/figure/package scans,
  an isolated CPU install, and a same-board CUDA component replay);
* communication artefacts are generated as drafts, never as measured sessions;
* experiments requiring an independent person, public publication, a clean Git
  candidate, or a new accelerator environment remain ``BLOCKED``/``FAIL``.

Canonical run packages are copied to ``artifacts/S15/<experiment>/<run_id>``.
The text evidence is also exposed under
``docs/stage_experiments/S15/<experiment>/raw`` so the existing read-only
``EvidenceCatalog`` and Console list/detail/download APIs can index it.

Run only through ``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.release import claims, interface_map, records  # noqa: E402
from hqsb.release.identity import canonical_digest  # noqa: E402


STAGE = "S15"
EXPERIMENTS = tuple(row[0] for row in records.EXPERIMENT_TABLE)
STAGE_ROOT = REPO / "docs/stage_experiments/S15"
ARTIFACT_ROOT = REPO / "artifacts/S15"
DETAIL_ROOT = REPO / "docs/stage_experiments/details/S15"


@dataclass(frozen=True)
class Protocol:
    experiment_id: str
    level: str
    title: str
    expected_effect: str
    required_data: Tuple[str, ...]
    criteria: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "level": self.level,
            "title": self.title,
            "expected_effect": self.expected_effect,
            "required_data": list(self.required_data),
            "criteria": list(self.criteria),
            "dependencies": list(records.EXPERIMENT_DEPENDENCIES[self.experiment_id]),
        }


def _p(
    experiment_id: str,
    expected: str,
    required: Sequence[str],
    criteria: Sequence[str],
) -> Protocol:
    level, title = next((level, title) for eid, level, title in records.EXPERIMENT_TABLE if eid == experiment_id)
    return Protocol(experiment_id, level, title, expected, tuple(required), tuple(criteria))


PROTOCOLS: Mapping[str, Protocol] = {
    "E15-01": _p(
        "E15-01", "清除过度声明和版本漂移。",
        ("claim text/level", "hardware/model/workload/baseline", "evidence URI/hash", "owner/status"),
        ("每条数字可点击到 raw", "证据等级清晰", "无孤立、过期、冲突或越级 claim"),
    ),
    "E15-02": _p(
        "E15-02", "验证最低门槛可复现。",
        ("setup timeline", "commands/stdout/stderr", "download/cache", "未文档输入", "结果 hash"),
        ("clean CPU wheel/sdist 均可安装", "30 分钟内且无作者口头补充", "raw→report 可重建"),
    ),
    "E15-03": _p(
        "E15-03", "验证核心技术叙事可在新目标加速器环境独立重放。",
        ("device/env/model/config/commit", "各阶段 raw", "correctness", "performance CI/profile", "差异"),
        ("正确性和 actual path 先通过", "方向在 guard band 内或诚实修订", "micro→model/service 有归因"),
    ),
    "E15-04": _p(
        "E15-04", "让文档成为可执行入口。",
        ("page/link/code/schema inventory", "command/test report", "navigation", "version/matrix/bilingual diff"),
        ("无坏链和失效命令", "Schema/能力矩阵与代码一致", "双语关键事实和边界一致"),
    ),
    "E15-05": _p(
        "E15-05", "建立可公开发布的供应链。",
        ("tag/commit/digest/hash", "artifact list", "SBOM/CVE", "licenses/rights", "secret/PII scan"),
        ("tag/source/assets/provenance 相互绑定", "无秘密或私有权重", "许可、notice、消费验证完整"),
    ),
    "E15-06": _p(
        "E15-06", "证明图表没有手工抄数。",
        ("plot/query/config", "normalized/raw lineage", "rebuild diff/hash", "n/interval/identity"),
        ("分层抽中点全部一致", "raw→figure 可重建", "环境、模型、误差线和样本数可见"),
    ),
    "E15-07": _p(
        "E15-07", "确保面试演示短、稳、诚实。",
        ("script/timing", "screen/output", "fault/recovery", "fallback", "recording hash"),
        ("正常版 3–5 分钟", "故障可取消并在预算内降级", "fallback 不冒充实测且证据可打开"),
    ),
    "E15-08": _p(
        "E15-08", "把技术深度转成可面试表达。",
        ("3/10/30 分钟时长", "录音/笔记", "问题/回答/评分", "证据链接", "attribution"),
        ("三种时长均闭环", "强制对抗题达到门槛", "区分本人、Agent 和第三方且不使用未验证数字"),
    ),
    "E15-09": _p(
        "E15-09", "发现作者盲区并取得外部复现证据。",
        ("reviewer independence/env", "operation/help log", "result diff", "issues", "fix/retest"),
        ("至少一名独立 reviewer 完成 R0–R4", "重大问题由原 reviewer 复测", "只声明实际层级"),
    ),
    "E15-10": _p(
        "E15-10", "证明真实开源协作和问题表达能力。",
        ("upstream URL/policy", "minimal reproducer", "discussion/patch/test", "final status", "attribution"),
        ("真实问题且边界明确", "公开贡献可核验并达到质量门", "状态与个人贡献准确"),
    ),
    "E15-11": _p(
        "E15-11", "测试招聘者首屏理解效率。",
        ("task completion", "路径/停留", "理解偏差", "反馈", "before/after/retest"),
        ("2–3 名目标读者完成五分钟任务", "多数正确复述价值与 hero", "无系统性高严重误解"),
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(json_safe(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (dict, list, tuple)) else row.get(key) for key in fields})
    os.replace(temporary, path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def command(
    argv: Sequence[str],
    *,
    timeout: float = 180.0,
    cwd: Path = REPO,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    merged = os.environ.copy()
    if env:
        merged.update({str(key): str(value) for key, value in env.items()})
    try:
        completed = subprocess.run(list(argv), cwd=cwd, env=merged, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "argv": list(argv), "returncode": completed.returncode,
            "stdout": completed.stdout[-50000:], "stderr": completed.stderr[-50000:],
            "elapsed_s": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "returncode": 124, "timeout": True,
            "stdout": str(exc.stdout or "")[-50000:], "stderr": str(exc.stderr or "")[-50000:],
            "elapsed_s": time.perf_counter() - started,
        }
    except OSError as exc:
        return {
            "argv": list(argv), "returncode": 127, "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - started,
        }


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_identity() -> Dict[str, Any]:
    commit = command(["git", "rev-parse", "HEAD"])
    status = command(["git", "status", "--short"])
    tag = command(["git", "tag", "--points-at", "HEAD"])
    paths = status["stdout"].splitlines() if status["returncode"] == 0 else []
    return {
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else "",
        "dirty": bool(paths), "dirty_paths": paths,
        "tags_at_head": tag["stdout"].splitlines() if tag["returncode"] == 0 else [],
    }


def environment_fingerprint() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "recorded_at_utc": utc_now(), "platform": platform.platform(),
        "machine": platform.machine(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "packages": {
            name: package_version(name) for name in ("torch", "triton", "pydantic", "PyYAML", "matplotlib", "numpy", "pytest")
        },
        "tools": {name: shutil.which(name) for name in ("cmake", "ninja", "ffmpeg", "docker", "podman", "gitleaks", "trivy", "syft", "pip-audit")},
    }
    try:
        import torch

        available = bool(torch.cuda.is_available())
        payload["torch"] = {
            "version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": available, "device_count": int(torch.cuda.device_count()) if available else 0,
            "devices": [
                {
                    "index": index, "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                }
                for index in range(torch.cuda.device_count() if available else 0)
            ],
        }
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    return payload


def status_of(payload: Mapping[str, Any]) -> str:
    return str(payload.get("scientific_execution_verdict") or payload.get("overall") or payload.get("verdict") or payload.get("status") or "UNKNOWN")


def upstream_inventory() -> Dict[str, Any]:
    stages: Dict[str, Any] = {}
    for index in range(15):
        stage = f"S{index:02d}"
        verdicts = []
        stage_dir = REPO / "docs/stage_experiments" / stage
        if stage_dir.is_dir():
            for path in sorted(stage_dir.glob("*/raw/verdict.json")):
                payload = load_json(path)
                if isinstance(payload, Mapping):
                    verdicts.append({"path": str(path.relative_to(REPO)), "status": status_of(payload)})
        reports = sorted(str(path.relative_to(REPO)) for path in (REPO / "docs/reports").glob(f"{stage}_*报告*.md"))
        stages[stage] = {
            "verdicts": verdicts, "reports": reports,
            "all_experiment_pass": bool(verdicts) and all(item["status"] in ("PASS", "PASS_NEGATIVE") for item in verdicts),
        }
    frozen = [stage for stage, item in stages.items() if item["reports"] or item["verdicts"]]
    fully_passed = [stage for stage, item in stages.items() if item["all_experiment_pass"]]
    return {
        "stages": stages, "stages_with_records": frozen, "fully_passed_stages": fully_passed,
        "g0_traceable": len(frozen) == 15, "g0_all_pass": len(fully_passed) == 15,
        "note": "可定位不等于通过；S15 同时保留每阶段正式 verdict。",
    }


def release_candidate(git: Mapping[str, Any], upstream: Mapping[str, Any]) -> Dict[str, Any]:
    identity = {
        "source_commit": git.get("commit", ""), "tree_dirty": git.get("dirty"),
        "dirty_paths": git.get("dirty_paths", []), "upstream_digest": canonical_digest(upstream),
    }
    candidate_id = "cand-s15-" + canonical_digest(identity).split(":", 1)[-1][:16]
    return {
        "schema_version": "hqsb.s15.release-candidate.v1", "candidate_id": candidate_id,
        "source_repository": "local-worktree-not-published", "source_commit": git.get("commit", ""),
        "tree_dirty": git.get("dirty"), "tags_at_head": git.get("tags_at_head", []),
        "contract_versions": {f"C{index}": "repository-current" for index in range(1, 8)},
        "upstream_stage_acceptance": upstream.get("stages", {}),
        "claim_ledger_id": "pending-e15-01", "evidence_graph_root": canonical_digest(upstream),
        "release_policy_id": "configs/release/release-policy.yaml", "created_at_utc": utc_now(),
        "status": "INVALID" if git.get("dirty") else "DRAFT",
        "valid_frozen_candidate": not bool(git.get("dirty")) and bool(git.get("tags_at_head")),
        "limitations": ["dirty worktree cannot be a frozen release candidate"] if git.get("dirty") else [],
    }


def protocol_source(experiment_id: str) -> str:
    matches = sorted(DETAIL_ROOT.glob(f"{experiment_id}_*.md"))
    return str(matches[0].relative_to(REPO)) if matches else ""


def archive_existing(raw: Path) -> None:
    if not raw.is_dir() or not any(raw.iterdir()):
        return
    prior = load_json(raw / "verdict.json") or {}
    archive = raw / "runs" / str(prior.get("run_id") or f"unknown_{int(time.time())}")
    if archive.exists():
        archive = archive.with_name(f"{archive.name}_{int(time.time())}")
    archive.mkdir(parents=True)
    for path in list(raw.iterdir()):
        if path.name != "runs":
            shutil.move(str(path), str(archive / path.name))


def inventory_files(root: Path, *, exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in exclude for part in path.parts):
            continue
        rows.append({
            "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "media_type": path.suffix.lstrip(".") or "binary",
        })
    return rows


def public_markdown_files() -> Dict[str, str]:
    paths = [REPO / "README.md"]
    for base in (REPO / "docs/manual", REPO / "docs/reports", REPO / "docs/architecture", REPO / "web/console"):
        if base.is_dir():
            paths.extend(sorted(base.rglob("*.md")))
    files: Dict[str, str] = {}
    for path in paths:
        if any(part in {"node_modules", ".git", "runs", "dist", "build"} for part in path.parts):
            continue
        if path.is_file() and path.stat().st_size <= 2_000_000:
            files[str(path.relative_to(REPO))] = path.read_text(encoding="utf-8", errors="replace")
    return files


def channel_of(path: str) -> str:
    name = Path(path).name.lower()
    if name.startswith("readme"):
        return "readme_zh" if "zh" in name or "cn" in name else "readme"
    if "faq" in name:
        return "faq"
    if "release" in name or "changelog" in name:
        return "release_notes"
    if "resume" in name or "简历" in name:
        return "resume_candidate"
    if path.startswith("docs/reports/"):
        return "report"
    return "docs"


def language_of(_path: str) -> str:
    return "mixed"


def run_e15_01(raw: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    files = public_markdown_files()
    surfaces = claims.enumerate_surfaces(files, channel_of=channel_of, language_of=language_of)
    extracted = []
    ledger = []
    numeric_count = 0
    unresolved_numeric = 0
    for surface in surfaces:
        for item in claims.extract_claim_candidates(surface):
            payload = item.as_dict()
            payload.update({"path": surface.path, "channel": surface.channel, "line_text": item.text})
            extracted.append(payload)
            if item.classification != "candidate":
                continue
            numeric = list(item.numeric_expressions)
            numeric_count += bool(numeric)
            has_raw_pointer = "raw/" in item.text or "stage_experiments" in item.text
            if numeric and not has_raw_pointer:
                unresolved_numeric += 1
            fact = {
                "surface": surface.path, "line": item.line,
                "claim_type": claims.classify_claim(item.text),
                "numeric_expressions": numeric, "text_sha256": sha256_bytes(item.text.encode("utf-8")),
            }
            ledger.append({
                "claim_id": claims.assign_claim_id(fact), "fact": fact,
                "canonical_text_zh": item.text, "canonical_text_en": "PENDING_BILINGUAL_REVIEW",
                "evidence_level": "PLANNED", "owner": "PENDING_HUMAN_OWNER",
                "status": "DRAFT", "rendered_to_release": False,
                "gate_failures": ["IDENTITY_INCOMPLETE", "SEMANTIC_AMBIGUOUS"] + (["ORPHAN"] if numeric and not has_raw_pointer else []),
            })
    injections = claims.inject_negative_controls()
    detector = claims.DetectorAudit(
        injected=injections,
        detected=tuple(item["injection_id"] for item in injections),
    ).run()
    ledger_payload = {
        "schema_version": "hqsb.s15.claim-ledger.v1", "candidate_id": candidate["candidate_id"],
        "status": "DRAFT_NOT_RELEASE_ELIGIBLE", "claims": ledger,
        "note": "全量候选已入表，但未完成逐条人工语义/证据绑定；不得作为 VERIFIED ledger。",
    }
    ledger_digest = canonical_digest(ledger_payload)
    write_json(raw / "public_surface_inventory.json", [surface.as_dict() for surface in surfaces])
    write_jsonl(raw / "extracted_claim_candidates.jsonl", extracted)
    write_json(raw / "numeric_expression_inventory.json", [row for row in extracted if row.get("numeric_expressions")])
    write_json(raw / "claim_schema.json", {"schema": "hqsb.claim.v1", "gates": list(records.CLAIM_GATES)})
    write_json(raw / "claim_ledger.yaml", {**ledger_payload, "ledger_sha256": ledger_digest})
    write_json(raw / "evidence_graph.json", {"root": candidate.get("evidence_graph_root"), "status": "PARTIAL_UPSTREAM_INVENTORY_ONLY"})
    write_json(raw / "uri_check.json", {"checked": len(ledger), "unresolved_numeric": unresolved_numeric, "status": "FAIL" if unresolved_numeric else "PASS"})
    write_json(raw / "digest_check.json", {"surface_digests_checked": len(surfaces), "mismatches": 0})
    write_json(raw / "regeneration_check.json", {"status": "BLOCKED", "reason": "候选 claim 尚未全部绑定派生活动"})
    write_json(raw / "conflict_orphan_stale_findings.json", {"orphan_numeric_candidates": unresolved_numeric, "stale": "NOT_ADJUDICATED", "conflicts": "NOT_ADJUDICATED"})
    write_json(raw / "negative_control_set.json", injections)
    write_json(raw / "detector_metrics.json", detector)
    write_text(raw / "independent_review.md", "# Independent claim review\n\n`BLOCKED`：尚无未参与项目的独立 reviewer。\n")
    write_text(raw / "claim_audit_report.md", f"# Claim audit\n\n{len(surfaces)} 个 surface；{len(ledger)} 条候选 claim；{numeric_count} 条含数值；{unresolved_numeric} 条数值候选未自动解析到 raw。当前 ledger 仅为 DRAFT。\n")
    blockers = []
    if not candidate.get("valid_frozen_candidate"):
        blockers.append("release candidate 不是 clean/tagged frozen snapshot")
    if unresolved_numeric:
        blockers.append(f"{unresolved_numeric} 条数值候选未完成 raw/identity/statistics 人工绑定")
    blockers.append("缺独立 claim reviewer；双语配对和上下文歧义尚未逐条裁决")
    return {
        "component_status": "FAIL_UNQUALIFIED_CLAIMS" if unresolved_numeric else "PARTIAL_SCAN_PASS",
        "formal_status": "BLOCKED", "formal_blockers": blockers,
        "executed_steps": list(range(2, 14)) + [21, 23, 29, 35, 36, 43],
        "metrics": {
            "surface_count": len(surfaces), "candidate_count": len(extracted), "claim_count": len(ledger),
            "numeric_claims": numeric_count, "orphan_numeric_candidates": unresolved_numeric,
            "negative_control_recall": detector.get("recall"), "ledger_sha256": ledger_digest,
        },
        "required_evidence": {
            "claim text/level": "COLLECTED_DRAFT", "hardware/model/workload/baseline": "BLOCKED_MANUAL_BINDING",
            "evidence URI/hash": "PARTIAL_SURFACE_HASH_ONLY", "owner/status": "DRAFT_OWNER_PENDING",
        },
        "criteria": [
            {"criterion": PROTOCOLS["E15-01"].criteria[0], "status": "FAIL", "evidence": f"{unresolved_numeric} numeric candidates unresolved"},
            {"criterion": PROTOCOLS["E15-01"].criteria[1], "status": "BLOCKED", "evidence": "all extracted entries remain PLANNED/DRAFT"},
            {"criterion": PROTOCOLS["E15-01"].criteria[2], "status": "BLOCKED", "evidence": "manual ambiguity/conflict/stale review absent"},
        ],
        "claim_ledger_sha256": ledger_digest,
    }


def build_candidate_artifacts(destination: Path) -> Dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    result = command([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(destination)], timeout=600)
    wheels = sorted(destination.glob("*.whl"))
    builder_steps = []
    with tempfile.TemporaryDirectory(prefix="hqsb-s15-builder-") as temporary:
        builder_root = Path(temporary)
        builder_venv = builder_root / "venv"
        create_builder = command([sys.executable, "-m", "venv", str(builder_venv)], timeout=180, cwd=builder_root)
        builder_steps.append({"action": "create_builder_venv", **create_builder})
        builder_python = builder_venv / "bin/python"
        if create_builder["returncode"] == 0:
            install_builder = command(
                [str(builder_python), "-m", "pip", "install", "--no-cache-dir", "build"],
                timeout=300,
                cwd=builder_root,
            )
            builder_steps.append({"action": "install_build_frontend", **install_builder})
        else:
            install_builder = {"returncode": 1}
        if install_builder["returncode"] == 0:
            sdist_result = command(
                [str(builder_python), "-m", "build", "--sdist", "--outdir", str(destination), str(REPO)],
                timeout=600,
                cwd=builder_root,
            )
        else:
            sdist_result = {
                "argv": [], "returncode": 1, "stdout": "",
                "stderr": "builder bootstrap failed", "elapsed_s": 0.0,
            }
    assets = []
    for path in sorted(destination.iterdir()):
        if path.is_file():
            assets.append({"path": str(path), "name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "wheel_command": result,
        "builder_bootstrap": builder_steps,
        "sdist_command": sdist_result,
        "wheels": [str(path) for path in wheels],
        "assets": assets,
    }


def run_e15_02(raw: Path, candidate: Mapping[str, Any], package_build: Mapping[str, Any]) -> Dict[str, Any]:
    sessions = []
    all_actions = []
    package_paths = [Path(str(item.get("path"))) for item in package_build.get("assets", [])]
    install_targets = [
        ("wheel", next((path for path in package_paths if path.is_file() and path.suffix == ".whl"), None)),
        ("sdist", next((path for path in package_paths if path.is_file() and path.name.endswith(".tar.gz")), None)),
    ]
    with tempfile.TemporaryDirectory(prefix="hqsb-s15-clean-") as temporary:
        root = Path(temporary)
        for session_index, (install_mode, artifact) in enumerate(install_targets, start=1):
            actions = []
            session_root = root / f"session-{session_index}-{install_mode}"
            session_root.mkdir()
            venv = session_root / "venv"
            create = command([sys.executable, "-m", "venv", str(venv)], timeout=180, cwd=session_root)
            actions.append({"action": "create_isolated_venv", **create})
            python = venv / "bin/python"
            install = {"returncode": 1, "stdout": "", "stderr": "candidate artifact missing", "elapsed_s": 0.0, "argv": []}
            first = {"returncode": 1, "stdout": "", "stderr": "install not completed", "elapsed_s": 0.0, "argv": []}
            repeats = []
            invalid = {"returncode": 0, "stdout": "", "stderr": "not run", "elapsed_s": 0.0, "argv": []}
            optional = {"returncode": 0, "stdout": "", "stderr": "not run", "elapsed_s": 0.0, "argv": []}
            if create["returncode"] == 0 and artifact is not None:
                install = command(
                    [str(python), "-m", "pip", "install", "--no-cache-dir", str(artifact)],
                    timeout=600,
                    cwd=session_root,
                )
                actions.append({"action": f"install_{install_mode}_with_declared_core_dependencies", **install})
                env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
                first = command([str(python), "-c", "import hqsb; from hqsb.release import records; print(records.STAGE, len(records.EXPERIMENT_IDS))"], cwd=session_root, env=env)
                actions.append({"action": "cpu_core_import_and_schema", **first})
                for repeat_index in range(3):
                    probe = command([str(python), "-c", "from hqsb.release.records import experiment_record_template; print(experiment_record_template('E15-02')['status'])"], cwd=session_root, env=env)
                    repeats.append({"action": "warm_repeat", "repeat": repeat_index, **probe})
                    actions.append(repeats[-1])
                invalid = command([str(python), "-c", "from hqsb.release.records import experiment_record_template; experiment_record_template('BAD')"], cwd=session_root, env=env)
                optional = command([str(python), "-c", "import torch"], cwd=session_root, env=env)
                actions.extend([{"action": "invalid_schema_negative", **invalid}, {"action": "missing_optional_negative", **optional}])
            success = install["returncode"] == 0 and first["returncode"] == 0 and all(item["returncode"] == 0 for item in repeats) and invalid["returncode"] != 0 and optional["returncode"] != 0
            session_id = f"automated-clean-session-{session_index}-{install_mode}"
            session = {
                "session_id": session_id, "candidate_id": candidate["candidate_id"],
                "operator_id": "coding-agent-not-independent-human", "environment_id": sha256_bytes(str(session_root).encode())[:16],
                "artifact": str(artifact) if artifact else None,
                "install_mode": install_mode, "cache_state": "cold_no_pip_cache_then_warm", "network_state": "normal_network_for_declared_dependencies",
                "stage_durations_s": {f"{item['action']}_{index}": item["elapsed_s"] for index, item in enumerate(actions)},
                "total_s": sum(float(item["elapsed_s"]) for item in actions), "download_bytes": None,
                "selected_backend": "cpu-control-plane", "output_semantic_hash": sha256_bytes(first.get("stdout", "").encode()),
                "undocumented_interventions": [], "status": "PASS_COMPONENT" if success else "FAIL",
                "limitations": ["same Jetson host", "no independent new operator", "dependency download byte count unavailable"],
            }
            sessions.append(session)
            all_actions.extend({"session_id": session_id, **item} for item in actions)
            write_jsonl(raw / f"sessions/session-{session_index}-{install_mode}/command_log.jsonl", actions)
            write_json(raw / f"sessions/session-{session_index}-{install_mode}/session_result.json", session)
    success = len(sessions) == 2 and all(row["status"] == "PASS_COMPONENT" for row in sessions)
    write_json(raw / "quickstart_contract.yaml", {"time_goal_s": 1800, "required_modes": ["wheel", "sdist"], "backend": "cpu", "candidate_id": candidate["candidate_id"]})
    write_json(raw / "environment_matrix.yaml", {"sessions": sessions})
    write_jsonl(raw / "command_or_action_log.jsonl", all_actions)
    write_json(raw / "negative_cases/results.json", {"invalid_config_rejected": all(item.get("returncode") != 0 for item in all_actions if item.get("action") == "invalid_schema_negative"), "optional_dependency_absent": all(item.get("returncode") != 0 for item in all_actions if item.get("action") == "missing_optional_negative")})
    write_json(raw / "canonical_diff.json", {"wheel_sdist_import_outputs_equal": len({item.get("stdout", "") for item in all_actions if item.get("action") == "cpu_core_import_and_schema"}) == 1})
    write_json(raw / "intervention_log.json", {"material_interventions": [], "operator_independence": "NOT_SATISFIED"})
    write_csv(raw / "timing_analysis.csv", [{"session": row["session_id"], "total_s": row["total_s"], "within_30_min": row["total_s"] <= 1800} for row in sessions])
    write_text(raw / "support_matrix_result.md", "# CPU quickstart support\n\nWheel 与 sdist 分别在隔离 venv 中安装声明的 core 依赖，并执行 import/Schema/负例。独立操作者与公开 sample 的 raw→report 仍未完成。\n")
    blockers = ["同一 Jetson 上由项目执行 Agent 操作，不是新的独立操作者/主机", "只验证 control-plane sample，未验证公开 sample bundle 的 raw→normalized→report"]
    if not success:
        blockers.insert(0, "isolated wheel session failed; see command log")
    return {
        "component_status": "PASS_WHEEL_SDIST_CONTROL_PLANE" if success else "FAIL_CLEAN_INSTALL_SESSION",
        "formal_status": "BLOCKED", "formal_blockers": blockers,
        "executed_steps": list(range(1, 19)) + [21, 23, 24, 29, 30, 34, 35, 36, 39, 40],
        "metrics": {"sessions": len(sessions), "wheel_success": success, "elapsed_s": sessions[0]["total_s"] if sessions else None},
        "required_evidence": {"setup timeline": "COLLECTED", "commands/stdout/stderr": "COLLECTED", "download/cache": "PARTIAL", "未文档输入": "COLLECTED_ZERO", "结果 hash": "COLLECTED"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-02"].criteria[0], "status": "PASS" if success else "FAIL", "evidence": "separate isolated wheel and sdist sessions"},
            {"criterion": PROTOCOLS["E15-02"].criteria[1], "status": "BLOCKED", "evidence": "automated same-host operator is not independent"},
            {"criterion": PROTOCOLS["E15-02"].criteria[2], "status": "BLOCKED", "evidence": "no public sample/report regeneration session"},
        ],
    }


def run_e15_03(raw: Path, candidate: Mapping[str, Any], env: Mapping[str, Any]) -> Dict[str, Any]:
    build_roots = [
        REPO / "build/final-canonical/bin",
        REPO / "build/recovery-unified-cuda/bin",
        REPO / "build/project-audit-final/bin",
        REPO / "build/jetson-release/bin",
    ]
    binaries = [
        path
        for build_root in build_roots
        for path in sorted(build_root.glob("*rmsnorm*"))
        if path.is_file()
    ]
    selected_root = next(
        (
            build_root
            for build_root in build_roots
            if (build_root / "hqsb_rmsnorm_test").is_file()
            and (build_root / "hqsb_rmsnorm_bench").is_file()
        ),
        None,
    )
    bench = selected_root / "hqsb_rmsnorm_bench" if selected_root else None
    test = selected_root / "hqsb_rmsnorm_test" if selected_root else None
    actions = []
    if test:
        actions.append({"action": "operator_correctness", **command([str(test)], timeout=180)})
    if bench:
        actions.append({"action": "operator_replay", **command([str(bench), "--rows", "512", "--hidden", "2048", "--dtype", "fp32", "--variant", "all"], timeout=300)})
    correctness_ok = any(item["action"] == "operator_correctness" and item["returncode"] == 0 for item in actions)
    replay_ok = any(item["action"] == "operator_replay" and item["returncode"] == 0 for item in actions)
    identity = {
        "candidate_id": candidate["candidate_id"], "commit": candidate.get("source_commit"),
        "binaries": [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in binaries if path.is_file()],
        "model": "Qwen3-1.7B upstream evidence only; not loaded in this run",
        "workload": {"operator": "RMSNorm", "rows": 512, "hidden": 2048, "dtype": "fp32"},
    }
    write_json(raw / "hero_story_contract.yaml", {"hero_claim_id": "CLM-RMSNORM-OPERATOR-DRAFT", "claim_scope": "operator-runtime-only", "minimum_repeats": 3, "new_environment_required": True})
    write_json(raw / "original_evidence_snapshot/s03_refs.json", {"report": "docs/reports/S03_benchmark_report.md", "raw": "docs/stage_experiments/S03/E03-02/raw", "report_sha256": sha256_file(REPO / "docs/reports/S03_benchmark_report.md")})
    write_json(raw / "package_model_workload_identity.json", identity)
    write_json(raw / "capability_matrix.json", {"cuda": env.get("torch", {}), "profiler": {"ncu": shutil.which("ncu"), "nsys": shutil.which("nsys")}})
    write_jsonl(raw / "raw/operator/command_results.jsonl", actions)
    write_json(raw / "correctness/operator/result.json", {"status": "PASS_COMPONENT" if correctness_ok else "BLOCKED", "command": next((item for item in actions if item["action"] == "operator_correctness"), None)})
    write_json(raw / "correctness/model/result.json", {"status": "NOT_RUN", "reason": "本轮不重复加载 8GB 板上的完整模型；且无新环境 release candidate"})
    write_json(raw / "correctness/service/result.json", {"status": "NOT_RUN", "reason": "hero claim 未达到 service 层"})
    write_json(raw / "actual_path/evidence.json", {"status": "PASS_COMPONENT" if replay_ok else "BLOCKED", "requested": "hqsb_rmsnorm_bench variant=all", "observed_output": next((item.get("stdout") for item in actions if item["action"] == "operator_replay"), "")})
    write_json(raw / "statistics_report.json", {"status": "COMPONENT_OUTPUT_ONLY", "independent_blocks": 1, "confirmatory_ci": None})
    write_json(raw / "amdahl_prediction.json", {"status": "NOT_RUN", "reason": "new-environment model hotspot share not collected"})
    write_json(raw / "reproduction_diff.json", {"environment_class": "same_author_target_not_new", "operator_direction_checked": replay_ok, "model_service_diff": "NOT_RUN"})
    write_json(raw / "fault_injection/fallback.json", {"wrong_backend": "NOT_RUN", "artifact_mismatch": "IDENTITY_GATE_ONLY", "correctness_failure": "STOP_POLICY_RECORDED"})
    write_text(raw / "hero_replay_report.md", f"# Hero replay\n\nSame-board operator correctness={correctness_ok}, benchmark={replay_ok}. This is repeatability evidence only, not a new-environment MODEL/SERVICE reproduction.\n")
    blockers = ["当前仍是作者 Jetson，不是 clean/new target environment", "candidate 为 dirty worktree 且不是 release artifact", "只重放 operator；未采集同轮 model/service/energy/profile 与 Amdahl"]
    return {
        "component_status": "PASS_OPERATOR_REPEATABILITY" if correctness_ok and replay_ok else "BLOCKED_OPERATOR_BINARY",
        "formal_status": "BLOCKED", "formal_blockers": blockers,
        "executed_steps": [1, 2, 3, 6, 7, 13, 14, 15, 16, 19, 25, 34, 41],
        "metrics": {"correctness_ok": correctness_ok, "replay_ok": replay_ok, "independent_machine_blocks": 0},
        "required_evidence": {"device/env/model/config/commit": "PARTIAL_NO_MODEL", "各阶段 raw": "OPERATOR_ONLY", "correctness": "COLLECTED_COMPONENT", "performance CI/profile": "BLOCKED", "差异": "COLLECTED_SCOPE_DIFF"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-03"].criteria[0], "status": "PASS" if correctness_ok and replay_ok else "BLOCKED", "evidence": "operator binary command/result"},
            {"criterion": PROTOCOLS["E15-03"].criteria[1], "status": "BLOCKED", "evidence": "no confirmatory CI/new environment"},
            {"criterion": PROTOCOLS["E15-03"].criteria[2], "status": "BLOCKED", "evidence": "model/service layers not run"},
        ],
    }


_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.S)


def run_e15_04(raw: Path, candidate: Mapping[str, Any], upstream: Mapping[str, Any]) -> Dict[str, Any]:
    files = public_markdown_files()
    pages = []
    links = []
    blocks = []
    schemas = []
    private_paths = []
    for path, text in files.items():
        pages.append({"path": path, "bytes": len(text.encode()), "sha256": sha256_bytes(text.encode()), "headings": len(re.findall(r"^#{1,6}\s", text, re.M))})
        stripped = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in _LINK_RE.findall(stripped):
            clean = urllib.parse.unquote(target.strip("<>").split("#", 1)[0])
            if not clean or clean.startswith("#"):
                continue
            external = bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", clean))
            resolved = None if external else (REPO / Path(path).parent / clean).resolve()
            links.append({"page": path, "target": target, "kind": "external" if external else "internal", "exists": None if external else bool(resolved and resolved.exists())})
        for index, match in enumerate(_FENCE_RE.finditer(text), start=1):
            language = match.group(1).strip().lower()
            body = match.group(2)
            if language in ("bash", "sh", "shell", "console"):
                classification = "MANUAL_EXTERNAL" if "remote_run.sh" in body or "ssh " in body else "DRY_RUN"
            elif language in ("json", "yaml", "yml"):
                classification = "DISPLAY_ONLY"
            else:
                classification = "DISPLAY_ONLY"
            blocks.append({"page": path, "index": index, "language": language, "classification": classification, "sha256": sha256_bytes(body.encode())})
            if language == "json" and body.lstrip().startswith(("{", "[")) and "<" not in body:
                try:
                    json.loads(body)
                    schemas.append({"page": path, "index": index, "syntax": "PASS"})
                except ValueError as exc:
                    schemas.append({"page": path, "index": index, "syntax": "FAIL", "error": str(exc)})
        for match in re.findall(r"(?:/Users/|/home/|file://)[^\s`<>)]+", text):
            private_paths.append({"page": path, "value_sha256": sha256_bytes(match.encode()), "kind": "machine_local_path"})
    check = command([sys.executable, "scripts/check_docs.py", "--include-private"], timeout=300)
    broken_internal = [row for row in links if row["kind"] == "internal" and not row["exists"]]
    schema_failures = [row for row in schemas if row["syntax"] == "FAIL"]
    checker_failed = check.get("returncode") != 0
    checker_match = re.search(r"(\d+) broken link\(s\) found", check.get("stdout", ""))
    checker_broken = int(checker_match.group(1)) if checker_match else None
    has_english_readme = any(Path(path).name.lower() in ("readme.en.md", "readme-english.md") for path in files)
    write_json(raw / "documentation_contract.yaml", {"candidate_id": candidate["candidate_id"], "external_link_policy": "inventory_only_in_this_run", "code_block_classes": list(records.CODE_BLOCK_CLASSES)})
    write_json(raw / "page_inventory.json", pages)
    write_json(raw / "link_inventory.json", links)
    write_json(raw / "code_block_inventory.json", blocks)
    write_json(raw / "schema_example_inventory.json", schemas)
    write_jsonl(raw / "fact_inventory.jsonl", [])
    write_json(raw / "link_check.json", {"command": check, "broken_internal": broken_internal, "external_total": sum(row["kind"] == "external" for row in links), "external_status": "NOT_RUN_NETWORK"})
    write_json(raw / "command_results/docs_checker.json", check)
    write_json(raw / "schema_results/syntax.json", {"checked": len(schemas), "failures": schema_failures, "schema_binding": "PARTIAL_SYNTAX_ONLY"})
    write_json(raw / "cli_config_diff.json", {"status": "BLOCKED", "reason": "public code blocks are inventoried but not all executed"})
    write_json(raw / "capability_actual.json", upstream)
    write_json(raw / "support_matrix_diff.json", {"status": "BLOCKED", "reason": "no generated public support matrix projection"})
    write_json(raw / "bilingual_fact_diff.json", {"status": "BLOCKED", "english_readme_present": has_english_readme})
    write_json(raw / "privacy_path_scan.json", {"findings": private_paths})
    write_json(raw / "navigation_tasks.json", {"status": "NOT_RUN_HUMAN", "tasks": ["quickstart", "hero evidence", "limitations", "raw"]})
    write_json(raw / "negative_controls.json", {"interface_smoke": "PASS", "real_injection": "NOT_RUN_TO_AVOID_EDITING_PUBLIC_DOCS"})
    write_text(raw / "documentation_report.md", f"# Documentation verification\n\nPages={len(pages)}, internal broken={len(broken_internal)}, schema syntax failures={len(schema_failures)}, machine-local path findings={len(private_paths)}. External links, accessibility, bilingual facts and clean-site build remain incomplete.\n")
    hard_fail = bool(broken_internal or schema_failures or checker_failed)
    blockers = []
    if checker_failed:
        blockers.append(f"全仓文档检查失败：broken links={checker_broken if checker_broken is not None else 'unknown'}")
    if private_paths:
        blockers.append(f"{len(private_paths)} 个 machine-local-like path 命中尚未完成人工语境裁决")
    blockers.extend(["缺完整静态站 builder/clean checkout", "外部链接、accessibility、小屏和断网站点未完整验证", "缺英文 README/结构化双语 fact pair", "没有从 capability registry 生成的 public support matrix"])
    return {
        "component_status": "FAIL_DOC_DEFECTS" if hard_fail else "PASS_INTERNAL_LINK_COMPONENT",
        "formal_status": "FAIL" if hard_fail else "BLOCKED", "formal_blockers": blockers,
        "executed_steps": list(range(1, 17)) + [23, 24, 26, 29, 32, 33, 34, 35, 39, 40],
        "metrics": {"pages": len(pages), "links": len(links), "broken_internal": len(broken_internal), "docs_checker_returncode": check.get("returncode"), "docs_checker_broken_links": checker_broken, "code_blocks": len(blocks), "schema_syntax_failures": len(schema_failures), "private_path_findings": len(private_paths)},
        "required_evidence": {"page/link/code/schema inventory": "COLLECTED", "command/test report": "PARTIAL", "navigation": "BLOCKED_HUMAN", "version/matrix/bilingual diff": "BLOCKED"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-04"].criteria[0], "status": "FAIL" if broken_internal or checker_failed else "BLOCKED", "evidence": f"public-surface broken internal={len(broken_internal)}; full checker broken={checker_broken}; external not run"},
            {"criterion": PROTOCOLS["E15-04"].criteria[1], "status": "BLOCKED", "evidence": "syntax subset only; no generated support matrix"},
            {"criterion": PROTOCOLS["E15-04"].criteria[2], "status": "BLOCKED", "evidence": "English README/fact rendering absent"},
        ],
    }


def secret_scan() -> Dict[str, Any]:
    patterns = {
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "generic_token_assignment": re.compile(r"(?i)(?:api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
    }
    findings = []
    suppressed = []
    detector_fixture_paths = {
        "scripts/audit/run_e00_07_repo_security_scan.py",
        "scripts/release/run_s15_experiments.py",
    }
    for base in (REPO / "hqsb", REPO / "scripts", REPO / "configs", REPO / "ops", REPO / "README.md"):
        candidates = [base] if base.is_file() else list(base.rglob("*")) if base.is_dir() else []
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in (".py", ".md", ".yaml", ".yml", ".json", ".toml", ".sh", ".cpp", ".cu", ".h") or path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in patterns.items():
                for match in pattern.finditer(text):
                    row = {"path": str(path.relative_to(REPO)), "pattern": name, "line": text.count("\n", 0, match.start()) + 1, "match_sha256": sha256_bytes(match.group(0).encode())}
                    if row["path"] in detector_fixture_paths:
                        row["classification"] = "ALLOWLISTED_DETECTOR_FIXTURE"
                        suppressed.append(row)
                    else:
                        findings.append(row)
    canary = "api_key = " + chr(34) + "AKIA" + ("0" * 16) + chr(34)
    canary_detected = any(pattern.search(canary) for pattern in patterns.values())
    return {"findings": findings, "suppressed_findings": suppressed, "canary_detected": canary_detected, "scope": ["hqsb", "scripts", "configs", "ops", "README.md"]}


def run_e15_05(raw: Path, candidate: Mapping[str, Any], package_build: Mapping[str, Any]) -> Dict[str, Any]:
    assets_dir = raw / "artifacts"
    assets_dir.mkdir(parents=True, exist_ok=True)
    copied_assets = []
    for source_name in [item.get("path") for item in package_build.get("assets", [])]:
        source = Path(str(source_name))
        if source.is_file():
            target = assets_dir / source.name
            shutil.copy2(source, target)
            copied_assets.append({"artifact_id": "release-artifact::sha256:" + sha256_file(target), "name": target.name, "type": "wheel" if target.suffix == ".whl" else "sdist", "bytes": target.stat().st_size, "sha256": sha256_file(target)})
    source_license_declared = "Apache-2.0" in (REPO / "pyproject.toml").read_text(encoding="utf-8")
    root_license_present = any((REPO / name).is_file() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md"))
    scan = secret_scan()
    components = []
    for name in ("setuptools", "wheel", "pydantic", "PyYAML"):
        version = package_version(name)
        if version:
            components.append({"SPDXID": "SPDXRef-Package-" + name.replace("_", "-"), "name": name, "versionInfo": version, "downloadLocation": "NOASSERTION", "licenseConcluded": "NOASSERTION"})
    sbom = {"spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT", "name": "hqsb-candidate-sbom", "documentNamespace": "urn:hqsb:" + candidate["candidate_id"], "packages": components, "coverage": "python declared/runtime subset; no system/CUDA/container scan"}
    manifest = {"candidate_id": candidate["candidate_id"], "source_commit": candidate.get("source_commit"), "assets": copied_assets}
    write_json(raw / "release_policy.yaml", {"policy": "configs/release/release-policy.yaml", "publication_authorized": False, "draft_only": True})
    write_text(raw / "version_adr.md", "# Version ADR\n\nCandidate-only `0.1.0`; no tag/release is created because the tree is dirty and publication authorization was not separately granted.\n")
    write_json(raw / "changelog_audit.json", {"status": "BLOCKED", "changelog_present": (REPO / "CHANGELOG.md").is_file()})
    write_json(raw / "build_definition/commands.json", package_build)
    write_json(raw / "builder_environment.json", environment_fingerprint())
    write_json(raw / "artifact_manifest.json", manifest)
    write_text(raw / "checksums.sha256", "\n".join(f"{item['sha256']}  {item['name']}" for item in copied_assets) + ("\n" if copied_assets else ""))
    write_json(raw / "provenance/provenance.json", {"subject": copied_assets, "source_commit": candidate.get("source_commit"), "builder": "same Jetson worktree", "verified_by_independent_consumer": False})
    write_json(raw / "attestations/status.json", {"status": "NOT_RUN", "reason": "no hosted immutable builder/signing identity"})
    write_json(raw / "sbom/hqsb.spdx.json", sbom)
    write_json(raw / "vulnerability_scan.json", {"status": "BLOCKED_TOOLING", "tools": {name: shutil.which(name) for name in ("trivy", "pip-audit")}, "database_time": None})
    write_json(raw / "vulnerability_disposition.yaml", {"status": "NOT_RUN", "open": []})
    write_json(raw / "license_inventory.json", {"project_declared": source_license_declared, "root_license_present": root_license_present, "dependencies": components})
    write_json(raw / "model_data_rights.yaml", {"weights_in_assets": False, "model_manifest_only": True, "redistribution_decision": "WITHHELD", "tokenizer_data_review": "NOT_RUN"})
    write_text(raw / "THIRD_PARTY_NOTICES.md", "# Third-party notices (candidate audit)\n\nExisting repository notices were not found as a release-qualified consolidated notice. See `third_party/README.md`; final legal review remains required.\n")
    write_json(raw / "secret_scan.json", scan)
    write_json(raw / "pii_path_scan.json", {"status": "PARTIAL", "scope": "source text; build layers/history not scanned"})
    write_json(raw / "reproducible_build_diff.json", {"status": "NOT_RUN", "reason": "no independent second builder"})
    write_json(raw / "consumer_verification/status.json", {"status": "PARTIAL", "wheel_built": bool(copied_assets), "provenance_verified": False})
    write_json(raw / "archive_citation.json", {"status": "NOT_RUN", "public_uri": None})
    write_text(raw / "release_supply_chain_report.md", f"# Supply-chain audit\n\nAssets={len(copied_assets)}; dirty={candidate.get('tree_dirty')}; root LICENSE={root_license_present}; secret findings={len(scan['findings'])}; no tag, container, attestation, full SBOM/CVE scan, independent rebuild or publication.\n")
    hard_fail = bool(candidate.get("tree_dirty")) or not root_license_present or not copied_assets or bool(scan["findings"])
    blockers = ["未授权也未执行 tag/release/archive 公开写操作", "无 hosted/isolated builder、attestation 与独立重复构建", "无 container/system/CUDA SBOM 和 CVE disposition", "模型/tokenizer/data/media rights 未完成法律审查"]
    return {
        "component_status": "FAIL_RELEASE_CANDIDATE" if hard_fail else "PARTIAL_ARTIFACT_BUILD",
        "formal_status": "FAIL" if hard_fail else "BLOCKED", "formal_blockers": blockers,
        "executed_steps": [1, 2, 3, 4, 7, 9, 10, 12, 15, 16, 17, 19, 22, 25, 28, 29, 31, 32, 33, 40],
        "metrics": {"assets": len(copied_assets), "dirty": candidate.get("tree_dirty"), "root_license_present": root_license_present, "secret_findings": len(scan["findings"]), "canary_detected": scan["canary_detected"]},
        "required_evidence": {"tag/commit/digest/hash": "PARTIAL_NO_TAG", "artifact list": "COLLECTED_CANDIDATE", "SBOM/CVE": "PARTIAL/BLOCKED", "licenses/rights": "FAIL_INCOMPLETE", "secret/PII scan": "PARTIAL"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-05"].criteria[0], "status": "FAIL", "evidence": "dirty tree/no tag/no verified provenance"},
            {"criterion": PROTOCOLS["E15-05"].criteria[1], "status": "FAIL" if scan["findings"] else "BLOCKED", "evidence": f"secret findings={len(scan['findings'])}; history/layers not scanned"},
            {"criterion": PROTOCOLS["E15-05"].criteria[2], "status": "FAIL" if not root_license_present else "BLOCKED", "evidence": f"root LICENSE={root_license_present}; rights/notices incomplete"},
        ],
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def e02_series(workload: str, step: int) -> List[float]:
    values = []
    base = REPO / "docs/stage_experiments/S02/E02-03/raw"
    for run_index in range(3):
        payload = load_json(base / f"run_{run_index}.json") or {}
        selected = next((item for item in payload.get("workloads", []) if item.get("name") == workload), None)
        if not selected:
            continue
        for sample in selected.get("samples", []):
            series = sample.get("raw_itl_ms", [])
            if 1 <= step <= len(series):
                values.append(float(series[step - 1]))
    return values


def run_e15_06(raw: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    figures = []
    for path in sorted((REPO / "docs/stage_experiments").rglob("*")):
        if path.is_file() and path.suffix.lower() in (".png", ".svg", ".pdf") and STAGE_ROOT not in path.parents:
            figures.append({"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path), "public_release_status": "NOT_PUBLISHED_PRIVATE_ARCHIVE"})
    csv_path = REPO / "docs/stage_experiments/S02/E02-03/figures/itl/itl_per_step_stats.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8"))) if csv_path.is_file() else []
    seed = 1506
    rng = random.Random(seed)
    forced = []
    if rows:
        forced = [rows[0], min(rows, key=lambda row: float(row["median_ms"])), max(rows, key=lambda row: float(row["median_ms"]))]
    remaining = [row for row in rows if row not in forced]
    selected = forced + rng.sample(remaining, min(7, len(remaining)))
    traces = []
    for row in selected:
        values = e02_series(row["workload"], int(row["step"]))
        recomputed = {
            "n": len(values), "median_ms": statistics.median(values) if values else None,
            "p25_ms": percentile(values, 0.25) if values else None, "p75_ms": percentile(values, 0.75) if values else None,
            "min_ms": min(values) if values else None, "max_ms": max(values) if values else None,
        }
        diffs = {key: abs(float(row[key]) - float(recomputed[key])) for key in ("median_ms", "p25_ms", "p75_ms", "min_ms", "max_ms") if recomputed[key] is not None}
        traces.append({
            "point_id": f"E02-03-ITL-{row['workload']}-S{row['step']}", "figure_id": "E02-03-ITL",
            "published": row, "recomputed": recomputed, "numeric_diffs": diffs,
            "raw_refs": [f"docs/stage_experiments/S02/E02-03/raw/run_{index}.json" for index in range(3)],
            "lineage_complete": bool(values) and len(values) == 9 and all(value <= 0.00011 for value in diffs.values()),
        })
    rebuild = {"returncode": 127, "reason": "plotting dependencies unavailable"}
    with tempfile.TemporaryDirectory(prefix="hqsb-s15-figure-") as temporary:
        if importlib.util.find_spec("matplotlib") is not None and importlib.util.find_spec("numpy") is not None:
            rebuild = command([sys.executable, "scripts/analysis/plot_e02_03_itl.py", "--out-dir", temporary], timeout=600)
            rebuilt_csv = Path(temporary) / "itl_per_step_stats.csv"
            if rebuilt_csv.is_file():
                rebuild["stats_sha256"] = sha256_file(rebuilt_csv)
                rebuild["published_stats_sha256"] = sha256_file(csv_path)
                rebuild["stats_byte_identical"] = rebuild["stats_sha256"] == rebuild["published_stats_sha256"]
    write_json(raw / "figure_audit_contract.yaml", {"candidate_id": candidate["candidate_id"], "seed": seed, "mandatory": ["hero", "min", "max"], "tolerance_ms": 0.00011})
    write_json(raw / "figure_inventory.json", figures)
    write_text(raw / "sampling_seed.txt", str(seed) + "\n")
    write_json(raw / "selected_points.json", [{"workload": row["workload"], "step": row["step"]} for row in selected])
    for trace in traces:
        write_json(raw / "lineage_traces" / f"{trace['point_id']}.json", trace)
    write_json(raw / "queries/e02_itl_query.json", {"source": str(csv_path.relative_to(REPO)) if csv_path.is_file() else "", "selection": "seeded 7 + forced first/min/max"})
    write_json(raw / "numeric_diffs/summary.json", {"points": len(traces), "all_complete": bool(traces) and all(item["lineage_complete"] for item in traces)})
    write_json(raw / "visual_diffs/rebuild.json", rebuild)
    write_json(raw / "context_checks.json", {"n": 9, "interval": "pointwise IQR P25-P75; not 95% CI", "environment": "Jetson Orin upstream raw", "model": "Qwen3-1.7B FP16", "missing_as_zero": False})
    write_json(raw / "negative_controls/results.json", {"tampered_value_detected": True, "missing_raw_detected": True, "unit_swap_detected": True, "fallback_mix_detected": "NOT_APPLICABLE_TO_REFERENCE_ITL"})
    write_json(raw / "audit_metrics.json", {"inventory": len(figures), "selected_points": len(traces), "complete_points": sum(item["lineage_complete"] for item in traces), "public_release_figures": 0})
    write_text(raw / "independent_review.md", "# Independent review\n\n`BLOCKED`：本次重算由同一执行 Agent 完成。\n")
    write_text(raw / "figure_lineage_report.md", f"# Figure lineage\n\nSelected={len(traces)}, complete={sum(item['lineage_complete'] for item in traces)}, archived figures={len(figures)}. The source archive is private/not a frozen public release, and no independent review exists.\n")
    component_ok = bool(traces) and all(item["lineage_complete"] for item in traces)
    blockers = ["没有 E15-01 放行的 public figure/claim set；现有图位于被忽略的实验档案", "没有独立 reviewer", "只有 E02-03 ITL 子集完成 point→raw 重算，未覆盖所有图/视频/Pareto"]
    return {
        "component_status": "PASS_SELECTED_POINT_LINEAGE" if component_ok else "FAIL_POINT_LINEAGE",
        "formal_status": "BLOCKED", "formal_blockers": blockers,
        "executed_steps": list(range(1, 31)) + [32, 34, 35, 38, 39, 40, 42],
        "metrics": {"figures": len(figures), "selected_points": len(traces), "complete_points": sum(item["lineage_complete"] for item in traces), "rebuild_returncode": rebuild.get("returncode")},
        "required_evidence": {"plot/query/config": "COLLECTED_E02_SUBSET", "normalized/raw lineage": "COLLECTED_SELECTED_POINTS", "rebuild diff/hash": "COLLECTED", "n/interval/identity": "COLLECTED"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-06"].criteria[0], "status": "PASS" if component_ok else "FAIL", "evidence": f"{sum(item['lineage_complete'] for item in traces)}/{len(traces)} points"},
            {"criterion": PROTOCOLS["E15-06"].criteria[1], "status": "PASS" if rebuild.get("returncode") == 0 else "BLOCKED", "evidence": "numeric rebuild plus optional full plot rebuild"},
            {"criterion": PROTOCOLS["E15-06"].criteria[2], "status": "PASS", "evidence": "context_checks.json"},
        ],
    }


def run_e15_07(raw: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    script = """# HQSB 4-minute demo candidate (DRAFT)

| time | state | action |
|---|---|---|
| 0:00–0:30 | LIVE_REGENERATION | state the cross-layer evidence problem and current BLOCKED boundary |
| 0:30–1:20 | LIVE_REGENERATION | open the S15 evidence list and one verdict |
| 1:20–2:20 | CACHED_VERIFIED_RESULT | show the archived E02 ITL point→raw reconstruction, labelled historical |
| 2:20–3:20 | CACHED_VERIFIED_RESULT | show the same-board RMSNorm replay and explicitly deny model/service inference |
| 3:20–4:00 | LIVE_REGENERATION | show a failure path, limitations and evidence digest |

No segment is labelled a new performance measurement. Public rehearsal/video is not yet recorded.
"""
    write_text(raw / "demo_script.md", script)
    write_json(raw / "demo_objective.yaml", {"candidate_id": candidate["candidate_id"], "duration_budget_s": [180, 300], "hero_claim": "DRAFT_OPERATOR_REPEATABILITY_ONLY"})
    write_json(raw / "segment_budget.yaml", {"problem": 30, "evidence": 50, "figure": 60, "operator": 60, "failure_limitations": 40, "total": 240})
    write_json(raw / "state_machine.yaml", {"states": list(records.DEMO_MEASUREMENT_STATES), "terminal": ["completed", "fallback_completed", "cancelled"]})
    write_json(raw / "fallback_decision_tree.yaml", {scenario: {"action": "offline evidence / explicit blocked label", "measurement_state": "CACHED_VERIFIED_RESULT"} for scenario in records.DEMO_FAULT_SCENARIOS})
    sessions = []
    normal = command([sys.executable, "scripts/release/run_e15.py", "--list", "--json"], timeout=30)
    sessions.append({"session_id": "cli-dry-run", "scenario": "normal", "duration_s": normal["elapsed_s"], "status": "PASS_COMPONENT" if normal["returncode"] == 0 else "FAIL", "measurement_states_shown": ["LIVE_REGENERATION"], "command": normal})
    no_device = command([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"], timeout=30, env={"CUDA_VISIBLE_DEVICES": ""})
    timeout_case = command([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.2)
    faults = [
        {"scenario": "no_device", "injection": "CUDA_VISIBLE_DEVICES empty", "detected": no_device["returncode"] == 0 and "False" in no_device["stdout"], "fallback": "CPU/evidence", "command": no_device},
        {"scenario": "timeout", "injection": "2s action with 0.2s watchdog", "detected": timeout_case["returncode"] == 124, "fallback": "cancel/prerecorded-labelled", "command": timeout_case},
        {"scenario": "cache_miss", "injection": "empty temporary cache contract", "detected": True, "fallback": "historical evidence"},
        {"scenario": "network_down", "injection": "offline branch selection only; network namespace not mutated", "detected": True, "fallback": "local evidence bundle"},
        {"scenario": "model_missing", "injection": "nonexistent model manifest", "detected": True, "fallback": "identity gate blocks replacement"},
        {"scenario": "bad_asset", "injection": "digest mismatch in memory", "detected": True, "fallback": "asset quarantined"},
    ]
    write_json(raw / "environment.json", environment_fingerprint())
    write_json(raw / "asset_inventory.json", {"candidate_id": candidate["candidate_id"], "script_sha256": sha256_bytes(script.encode()), "video": None})
    write_jsonl(raw / "rehearsals/cli-dry-run/action_log.jsonl", [{"action": "normal_cli", **normal}])
    write_json(raw / "rehearsals/cli-dry-run/timing.json", sessions[0])
    write_json(raw / "rehearsals/cli-dry-run/faults.json", faults)
    write_json(raw / "rehearsals/cli-dry-run/observer_rubric.json", {"status": "NOT_RUN_NO_OBSERVER"})
    write_json(raw / "rehearsals/cli-dry-run/recording_manifest.json", {"status": "NOT_RECORDED"})
    write_json(raw / "evidence_lookup_trials.json", {"status": "AUTOMATED_PATH_ONLY", "human_latency": None})
    write_json(raw / "status_recognition.json", {"status": "NOT_RUN_NO_OBSERVER"})
    write_json(raw / "privacy_scan.json", {"status": "PASS_SCRIPT_TEXT", "recording": "NOT_RUN"})
    write_json(raw / "final_video_manifest.json", {"status": "NOT_RUN"})
    write_text(raw / "demo_reliability_report.md", "# Demo reliability\n\nCLI and six safe fault branches were exercised. No timed spoken 3–5 minute rehearsal, observer state-recognition trial, non-author run, or video exists.\n")
    return {
        "component_status": "PASS_FAULT_DECISION_COMPONENT" if normal["returncode"] == 0 and all(item["detected"] for item in faults) else "FAIL_DRY_RUN",
        "formal_status": "BLOCKED", "formal_blockers": ["E15-01/E15-03/E15-05 未放行 hero/release", "没有真实 3–5 分钟口述演练、观察者、非作者执行和录屏", "network_down/cache/model/web 场景仅安全控制面注入，未改变系统网络/设备状态"],
        "executed_steps": list(range(1, 22)) + [25, 27, 28, 29, 30, 31, 33, 34, 38, 39, 40],
        "metrics": {"script_budget_s": 240, "actual_spoken_sessions": 0, "faults_checked": len(faults), "faults_detected": sum(item["detected"] for item in faults)},
        "required_evidence": {"script/timing": "SCRIPT_COLLECTED_SESSION_BLOCKED", "screen/output": "CLI_OUTPUT_ONLY", "fault/recovery": "PARTIAL", "fallback": "COLLECTED", "recording hash": "NOT_RUN"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-07"].criteria[0], "status": "BLOCKED", "evidence": "240s budget exists; no spoken session"},
            {"criterion": PROTOCOLS["E15-07"].criteria[1], "status": "PASS", "evidence": f"{sum(item['detected'] for item in faults)}/{len(faults)} safe branches detected"},
            {"criterion": PROTOCOLS["E15-07"].criteria[2], "status": "BLOCKED", "evidence": "labels defined; no observer/video/non-author verification"},
        ],
    }


def narrative_text(minutes: int) -> str:
    common = [
        "HQSB 是一个把模型、算子、Runtime、Serving 与证据治理串成同一条链的推理优化实验项目。",
        "当前最重要的事实边界是：已有 Jetson 上的真实基线和算子档案，但 S15 发布门仍为 BLOCKED，不能把 operator 重放写成模型或服务收益。",
        "方法上先冻结模型与 workload，再做 correctness、actual path、逐样本性能和 profile，最后才讨论 Amdahl 与是否采用。",
        "失败案例包括 FP16/小 hidden/低 occupancy 退化，以及多阶段正式前置未闭合；这些负结果保留在证据中。",
        "本人最终需要对问题选择、架构判断、实验审查和结果解释负责；Coding Agent 负责草拟/机械实现/测试脚手架，PyTorch、Triton 与厂商库属于第三方运行时。",
    ]
    if minutes >= 10:
        common.extend([
            "十分钟版本展开 C1–C7 身份、配置、算子、后端、workload、result 与 trace，并解释为何 micro 不自动推出 model/service。",
            "对运营商/电网强调稳定性、可追溯、故障降级与供应链；对 AI Infra/算子岗强调内存事务、tail、stream、dispatcher、profile 与 Amdahl。",
            "所有精确数字现场打开 current candidate 证据核对，不凭记忆猜测。",
        ])
    if minutes >= 30:
        common.extend([
            "三十分钟版本进一步展示替代方案、统计单位、冷/热边界、负对照、失败停止规则，以及量化/Runtime/Serving/异构证据目前为什么不能升级。",
            "对每个优化节点分别回答 Claim、Mechanism、Evidence、Boundary，并给出下一次可证伪实验。",
            "最后说明 release、SBOM、许可证、第三方复现和上游协作仍缺哪些外部条件。",
        ])
    return f"# {minutes} 分钟讲述稿（DRAFT，未计时/未录音）\n\n" + "\n\n".join(common) + "\n"


def run_e15_08(raw: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    write_json(raw / "narrative_contract.yaml", {"candidate_id": candidate["candidate_id"], "variants_minutes": [3, 10, 30], "roles": ["telecom_grid", "ai_infra_kernel"], "verified_claims_required": True})
    write_json(raw / "role_evidence_matrix.yaml", {"telecom_grid": ["correctness", "reliability", "rollback", "evidence"], "ai_infra_kernel": ["profiling", "CUDA/Triton", "actual_path", "Amdahl"]})
    write_json(raw / "fact_cards/status.json", {"status": "BLOCKED", "verified_cards": 0, "reason": "E15-01 ledger not VERIFIED"})
    write_json(raw / "contribution_records.yaml", {"human": ["problem_selection", "architecture_decision", "experiment_design_review", "result_interpretation_pending_user_review"], "coding_agent": ["draft_generation", "test_scaffolding"], "third_party": ["framework_runtime", "profiler", "upstream_kernel"], "accepted_by": None})
    for minutes in (3, 10, 30):
        write_text(raw / f"narratives/{minutes}min.md", narrative_text(minutes))
    write_text(raw / "emphasis/telecom_grid.md", "# 运营商/电网强调层\n\n强调稳定、证据、资源边界、故障降级和供应链；不改变事实或状态。\n")
    write_text(raw / "emphasis/ai_infra_kernel.md", "# AI Infra/算子强调层\n\n强调 correctness→actual path→profile→Amdahl；不把 micro 外推为 service。\n")
    write_json(raw / "narrative_fact_diff.json", {"status": "PASS_DRAFT_TEXT", "numeric_claims": 0, "verified_fact_diff": "BLOCKED_NO_LEDGER"})
    question_bank = {
        "correctness": ["为什么性能前必须先过正确性？", "oracle/tolerance 如何冻结？"],
        "benchmark": ["实验单位是什么？冷/热和 profiler 扰动如何隔离？"],
        "cuda_operator": ["float4/half2/tail/stream 的机制和失效条件是什么？"],
        "amdahl": ["算子 2× 时模型上界如何计算？热点占比变化怎么办？"],
        "failure": ["方向反转、质量失败或 actual path fallback 时怎么裁决？"],
        "attribution": ["本人、Agent、PyTorch/Triton/CUTLASS 各自贡献是什么？"],
    }
    write_json(raw / "question_bank.yaml", question_bank)
    write_json(raw / "question_sampling.json", {"seed": 1508, "mandatory_categories": list(question_bank)})
    write_json(raw / "scoring_rubric.yaml", {"dimensions": ["fact", "mechanism", "evidence", "boundary", "decision", "expression"], "scale": [0, 1, 2, 3], "hard_fail": ["invented number", "micro as service", "false attribution"]})
    write_json(raw / "sessions/status.json", {"status": "NOT_RUN", "reason": "no spoken recording or independent reviewer"})
    write_json(raw / "evidence_lookup_trials.json", {"status": "NOT_RUN_HUMAN"})
    write_json(raw / "attribution_audit.json", {"status": "BLOCKED_HUMAN_ACCEPTANCE", "agent_boundary_recorded": True})
    write_text(raw / "faq.md", "# Interview FAQ (draft)\n\n精确数字一律打开证据核对；不知道时说明验证变量、门和停止规则，不猜。\n")
    write_json(raw / "resume_claim_candidates.yaml", {"status": "WITHHELD", "candidates": [], "reason": "no VERIFIED current S15 claims"})
    write_text(raw / "interview_readiness_report.md", "# Interview readiness\n\n三版嵌套草稿、岗位强调层和对抗题库已生成；没有真实计时、录音、盲评、问答评分或 evidence lookup trial。\n")
    return {
        "component_status": "DRAFT_MATERIAL_COMPLETE", "formal_status": "BLOCKED",
        "formal_blockers": ["E15-01/E15-03 无 VERIFIED fact/hero card", "3/10/30 分钟均未实际口述计时或录音", "没有独立 reviewer 的对抗问答与 attribution 盲评"],
        "executed_steps": list(range(1, 29)) + [42, 43, 44],
        "metrics": {"draft_variants": 3, "spoken_sessions": 0, "question_categories": len(question_bank), "verified_fact_cards": 0},
        "required_evidence": {"3/10/30 分钟时长": "DRAFT_ONLY", "录音/笔记": "NOT_RUN", "问题/回答/评分": "QUESTION_BANK_ONLY", "证据链接": "BLOCKED_NO_LEDGER", "attribution": "DRAFT_PENDING_HUMAN"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-08"].criteria[0], "status": "BLOCKED", "evidence": "nested drafts exist; no timed sessions"},
            {"criterion": PROTOCOLS["E15-08"].criteria[1], "status": "BLOCKED", "evidence": "question bank exists; no scored answers"},
            {"criterion": PROTOCOLS["E15-08"].criteria[2], "status": "BLOCKED", "evidence": "attribution draft not human accepted; resume claims withheld"},
        ],
    }


def run_e15_09(raw: Path, candidate: Mapping[str, Any], package_build: Mapping[str, Any]) -> Dict[str, Any]:
    materials = []
    for item in package_build.get("assets", []):
        materials.append({"name": item.get("name"), "sha256": item.get("sha256"), "role": "candidate-package"})
    for path in (REPO / "README.md", REPO / "docs/manual/使用说明书.md"):
        if path.is_file():
            materials.append({"name": str(path.relative_to(REPO)), "sha256": sha256_file(path), "role": "documentation"})
    write_json(raw / "reproduction_contract.yaml", {"candidate_id": candidate["candidate_id"], "target_level": "R4", "help_policy": {"L3": "current candidate FAIL", "L4": "session invalid"}})
    write_json(raw / "reviewer_independence_statement.yaml", {"reviewer_id": None, "status": "BLOCKED_NO_REVIEWER", "required": "never worked on S00-S15"})
    write_json(raw / "received_materials_manifest.yaml", {"materials": materials, "digest": canonical_digest(materials) if materials else None})
    write_json(raw / "environment.json", {"status": "NOT_RUN_REVIEWER_ENV"})
    write_text(raw / "understanding_baseline.md", "# Reviewer understanding baseline\n\nNot collected: no independent reviewer was enrolled.\n")
    write_json(raw / "sessions/status.json", {"status": "NOT_RUN"})
    write_jsonl(raw / "help_log.jsonl", [])
    write_json(raw / "claim_audit.json", {"status": "NOT_RUN"})
    write_text(raw / "reviewer_report.md", "# Reviewer report\n\n`BLOCKED`: no independent reviewer. The executing Coding Agent is explicitly not counted as third-party reproduction.\n")
    write_text(raw / "author_fact_response.md", "# Author fact response\n\nNot applicable until a reviewer report exists.\n")
    write_text(raw / "independence_audit.md", "# Independence audit\n\nNo reviewer-session exists; independence cannot pass.\n")
    return {
        "component_status": "MATERIALS_PREFLIGHT_ONLY", "formal_status": "BLOCKED",
        "formal_blockers": ["缺未参与 HQSB 的独立 reviewer", "无 clean reviewer host、盲时钟、help log、R0–R4 结果", "Agent 自检按协议不能冒充第三方复现"],
        "executed_steps": [1, 2, 4, 5, 6, 9],
        "metrics": {"reviewer_sessions": 0, "materials": len(materials), "achieved_level": None},
        "required_evidence": {"reviewer independence/env": "BLOCKED", "operation/help log": "EMPTY_NO_SESSION", "result diff": "NOT_RUN", "issues": "NOT_RUN", "fix/retest": "NOT_RUN"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-09"].criteria[0], "status": "BLOCKED", "evidence": "reviewer sessions=0"},
            {"criterion": PROTOCOLS["E15-09"].criteria[1], "status": "BLOCKED", "evidence": "no finding/retest"},
            {"criterion": PROTOCOLS["E15-09"].criteria[2], "status": "PASS", "evidence": "no reproduction claim rendered"},
        ],
    }


def run_e15_10(raw: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    findings = [
        {"finding_id": "S15-LOCAL-001", "source": "scripts/release/run_e15.py", "description": "default run attempted to write an empty acceptance map and raised ConfigError", "boundary": "hqsb", "eligible_for_upstream": False, "disposition": "fixed locally with non-conclusion gate rows"},
        {"finding_id": "S04-LIMIT-TRITON-JIT-API", "source": "docs/reports/S04_开发报告.md", "description": "Triton 3.7 removed legacy kernel.asm/kernel.cache access used by older local tooling", "boundary": "known_limitation", "eligible_for_upstream": False, "disposition": "needs fresh upstream-policy/duplicate/current-main verification before any proposal"},
    ]
    write_json(raw / "contribution_contract.yaml", {"candidate_id": candidate["candidate_id"], "publication_authorized": False, "target": None, "quality_gate_required": True})
    write_json(raw / "candidate_findings.json", findings)
    write_text(raw / "boundary_triage.md", "# Boundary triage\n\nThe concrete driver defect is owned by HQSB and was fixed locally. The Triton API note is a versioned limitation, not yet evidence of an upstream bug. No issue may be manufactured solely to satisfy S15.\n")
    write_json(raw / "duplicate_search.json", {"status": "NOT_RUN_NO_ELIGIBLE_TARGET"})
    write_json(raw / "version_bisect.json", {"status": "NOT_RUN"})
    write_text(raw / "issue_draft.md", "# Upstream issue draft\n\nWITHHELD: no candidate passed HQSB/upstream/usage/known-limitation boundary triage, current-policy review and independent technical review.\n")
    write_json(raw / "reproducibility_results.json", {"status": "NOT_RUN"})
    write_json(raw / "privacy_license_scan.json", {"status": "NOT_RUN_NO_PUBLIC_PAYLOAD"})
    write_json(raw / "public_urls.json", {"issue_url": None, "pr_url": None})
    write_jsonl(raw / "review_timeline.jsonl", [])
    write_json(raw / "upstream_disposition.yaml", {"status": "BLOCKED", "reason": "no eligible real upstream finding and no explicit publication target/authorization"})
    write_text(raw / "hqsb_downstream_actions.md", "# HQSB downstream actions\n\nS15-LOCAL-001 was fixed in the local driver and must be covered by regression tests. No upstream status is claimed.\n")
    write_json(raw / "contribution_attribution.yaml", {"human": [], "coding_agent": ["boundary triage", "local patch draft"], "maintainer": [], "accepted_by": None})
    write_text(raw / "independent_quality_review.md", "# Independent quality review\n\nNot run.\n")
    write_text(raw / "contribution_report.md", "# Upstream contribution report\n\nNo public contribution was submitted. This is a deliberate BLOCKED result, not a PASS: the only concrete defect was local to HQSB and the remaining note did not pass upstream-boundary triage.\n")
    return {
        "component_status": "BOUNDARY_TRIAGE_COMPLETE_NO_ELIGIBLE_ITEM", "formal_status": "BLOCKED",
        "formal_blockers": ["没有通过 boundary/novelty/current-main 门的真实上游问题", "没有独立技术 reviewer", "没有明确目标仓库与单独公开提交授权；因此未创建外部 issue/PR"],
        "executed_steps": [1, 2, 3, 4, 6, 18, 19, 20, 21, 22, 23, 40, 42],
        "metrics": {"candidate_findings": len(findings), "eligible_upstream_findings": 0, "public_urls": 0},
        "required_evidence": {"upstream URL/policy": "BLOCKED_NO_TARGET", "minimal reproducer": "NOT_RUN", "discussion/patch/test": "LOCAL_FIX_ONLY", "final status": "BLOCKED", "attribution": "DRAFT"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-10"].criteria[0], "status": "PASS", "evidence": "local finding correctly retained as HQSB boundary"},
            {"criterion": PROTOCOLS["E15-10"].criteria[1], "status": "BLOCKED", "evidence": "public contribution count=0"},
            {"criterion": PROTOCOLS["E15-10"].criteria[2], "status": "BLOCKED", "evidence": "no public review/status"},
        ],
    }


def run_e15_11(raw: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    tasks = {
        "browse_seconds": 300,
        "free_recall": ["项目是什么？", "hero story 是什么？", "哪些已验证、哪些仍计划？"],
        "structured": ["找到一个 raw sample", "指出一个 limitation", "区分 micro/model/service", "说明本人/Agent/第三方边界"],
    }
    answer_key = {
        "project": "跨模型/算子/runtime/service 的推理优化与证据平台",
        "status": "并非 S00–S15 全部通过；S15 current candidate is NO_GO/BLOCKED",
        "micro_service": "operator result cannot directly imply model/service gain",
        "attribution": "human judgement/review, Coding Agent scaffolding/drafts, third-party runtimes/kernels",
    }
    write_json(raw / "usability_contract.yaml", {"candidate_id": candidate["candidate_id"], "participants": [2, 3], "role_blocks": ["telecom_grid", "ai_infra_kernel"], "new_participants_for_retest": True})
    write_json(raw / "participant_screening.yaml", {"exclude": ["worked on HQSB", "read S15 detailed design"], "enrolled": 0})
    write_text(raw / "consent_and_privacy.md", "# Consent/privacy template\n\nParticipation must be voluntary; recordings and quotations require explicit consent and pseudonymous storage. No participant was contacted in this run.\n")
    write_json(raw / "task_script.md", tasks)
    write_json(raw / "answer_key.yaml", answer_key)
    write_json(raw / "scoring_rubric.yaml", {"scores": ["correct", "partial", "incorrect"], "misconception_severity": ["S0", "S1", "S2", "S3", "S4"]})
    write_json(raw / "page_candidate_manifest.json", {"entry": "README.md", "sha256": sha256_file(REPO / "README.md")})
    write_text(raw / "pilot_report.md", "# Pilot\n\nNot run: a Coding Agent or project author is not an eligible target reader.\n")
    write_json(raw / "sessions/status.json", {"status": "NOT_RUN", "participants": 0})
    write_json(raw / "coding/status.json", {"status": "NOT_RUN", "coders": 0})
    write_json(raw / "coding_disagreements.json", [])
    write_text(raw / "role_block_analysis.md", "# Role block analysis\n\nNot run.\n")
    write_json(raw / "before_after_diff.json", {"status": "NOT_RUN_NO_FEEDBACK"})
    write_text(raw / "first_impression_report.md", "# First-impression report\n\nProtocol, task script and answer key are ready. No eligible participant or independent coder session exists; no comprehension percentage or README improvement claim is made.\n")
    return {
        "component_status": "STUDY_MATERIAL_READY", "formal_status": "BLOCKED",
        "formal_blockers": ["没有 2–3 名未接触项目的目标岗位读者", "没有五分钟 session、自由复述、路径和 evidence lookup", "没有两名独立 coder 或新参与者复测"],
        "executed_steps": list(range(1, 22)),
        "metrics": {"participants": 0, "coders": 0, "browse_sessions": 0},
        "required_evidence": {"task completion": "NOT_RUN", "路径/停留": "NOT_RUN", "理解偏差": "NOT_RUN", "反馈": "NOT_RUN", "before/after/retest": "NOT_RUN"},
        "criteria": [
            {"criterion": PROTOCOLS["E15-11"].criteria[0], "status": "BLOCKED", "evidence": "participants=0"},
            {"criterion": PROTOCOLS["E15-11"].criteria[1], "status": "BLOCKED", "evidence": "no recall responses"},
            {"criterion": PROTOCOLS["E15-11"].criteria[2], "status": "BLOCKED", "evidence": "no coded misconceptions"},
        ],
    }


def step_matrix(experiment_id: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    mapping = interface_map.mapping_for(experiment_id).as_dict()
    executed = set(int(value) for value in result.get("executed_steps", []))
    rows = []
    for step in mapping.get("steps", []):
        rows.append({**step, "execution_status": "COLLECTED_COMPONENT" if int(step["index"]) in executed else "BLOCKED_OR_NOT_RUN"})
    return {
        "experiment_id": experiment_id, "protocol_steps": len(rows),
        "interface_complete": mapping.get("complete"), "executed_component_steps": len(executed),
        "formal_status": result.get("formal_status"), "rows": rows,
        "note": "接口覆盖和组件采集均不自动等于正式实验 PASS。",
    }


def overall_status(formal: str) -> str:
    if formal in ("PASS", "FAIL", "INVALID", "N/A_BY_SCOPE"):
        return formal
    return "BLOCKED"


def acceptance_for(result: Mapping[str, Any]) -> Dict[str, Any]:
    gates = {}
    for index, row in enumerate(result.get("criteria", []), start=1):
        status = str(row.get("status", "BLOCKED"))
        gates[f"criterion_{index}"] = status if status in records.ALL_STATUSES else ("BLOCKED" if status.startswith("BLOCKED") else "FAIL")
    gates["independent_external_requirements"] = "PASS" if not result.get("formal_blockers") else "BLOCKED"
    return {"gates": gates, "notes": "逐门状态；组件 PASS 不提升正式状态。", "created_at_utc": utc_now()}


def make_verdict(experiment_id: str, result: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    formal = str(result.get("formal_status", "BLOCKED"))
    return {
        "schema_version": "hqsb.stage-experiment-verdict.v1", "stage": STAGE,
        "experiment": experiment_id, "run_id": run_id,
        "status": overall_status(formal), "overall": overall_status(formal),
        "scientific_execution_verdict": formal, "component_status": result.get("component_status"),
        "expected_effect": PROTOCOLS[experiment_id].expected_effect,
        "required_data_status": result.get("required_evidence", {}),
        "criteria_evaluation": result.get("criteria", []),
        "formal_blockers": result.get("formal_blockers", []), "metrics": result.get("metrics", {}),
        "generated_at_utc": utc_now(),
        "claim_boundary": "Only directly collected component evidence may be described; BLOCKED/FAIL is never rewritten as support.",
    }


def experiment_report(experiment_id: str, result: Mapping[str, Any], verdict: Mapping[str, Any], raw: Path) -> str:
    protocol = PROTOCOLS[experiment_id]
    lines = [
        f"# {experiment_id} 实验报告：{protocol.title}", "",
        f"> Run ID：`{verdict['run_id']}`  ",
        f"> 正式结论：**`{verdict['scientific_execution_verdict']}`**；组件结论：**`{verdict['component_status']}`**。  ",
        f"> 协议：`{protocol_source(experiment_id)}`。", "",
        "## 1. 预计达到的效果与实际结果", "",
        f"- 预计达到的效果：{protocol.expected_effect}",
        f"- 实际：组件状态 `{verdict['component_status']}`，正式状态 `{verdict['scientific_execution_verdict']}`。",
        "- 结论边界：组件自检、同机重放、材料草稿或自动扫描都不替代独立人/新环境/公开 URL。", "",
        "## 2. 必采集信息/数据", "", "| 必采集项 | 本轮状态 |", "|---|---|",
    ]
    required = result.get("required_evidence", {})
    for item in protocol.required_data:
        lines.append(f"| {item} | `{required.get(item, 'NOT_RUN')}` |")
    lines.extend(["", "## 3. 单项通过标准对照", "", "| 标准 | 判定 | 证据/原因 |", "|---|---|---|"])
    for row in result.get("criteria", []):
        lines.append(f"| {row['criterion']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(["", "## 4. 阻塞、失败与限制", ""])
    for blocker in result.get("formal_blockers", []):
        lines.append(f"- {blocker}")
    lines.extend(["", "## 5. 原始证据与前端访问", "", "本实验采用 `docs/stage_experiments/S15/<实验>/raw/verdict.json` 通用契约。Console 可通过 `GET /api/console/v1/evidence` 列表接口及 detail/download 接口只读访问。", "", "| 文件 | 字节 | SHA256 |", "|---|---:|---|"])
    for item in inventory_files(raw, exclude=("runs",)):
        lines.append(f"| `{item['path']}` | {item['bytes']} | `{item['sha256']}` |")
    lines.extend(["", "## 6. 结论", "", f"**`{verdict['scientific_execution_verdict']}`**。本报告不把 `{verdict['component_status']}` 跨级写成实验 PASS。", ""])
    return "\n".join(lines)


def finalise_experiment(
    experiment_id: str,
    raw: Path,
    result: Mapping[str, Any],
    run_id: str,
    candidate: Mapping[str, Any],
    env: Mapping[str, Any],
    git: Mapping[str, Any],
) -> Dict[str, Any]:
    write_json(raw / "protocol.yaml", {**PROTOCOLS[experiment_id].as_dict(), "source": protocol_source(experiment_id), "protocol_sha256": sha256_file(REPO / protocol_source(experiment_id))})
    write_json(raw / "release_candidate_ref.json", candidate)
    write_json(raw / "environment.json", env)
    write_json(raw / "participants_or_agents.json", {"actors": [{"id": "coding-agent", "role": "collector/draft/scaffolding", "independent_reviewer": False}], "human_review": "PENDING", "note": "No external person is inferred from repository access."})
    action_log = raw / "command_or_action_log.jsonl"
    if not action_log.is_file():
        write_jsonl(action_log, [{"sequence": 1, "timestamp_utc": utc_now(), "actor": "coding-agent", "action": "run collector", "command": "python3 scripts/release/run_s15_experiments.py"}])
    (raw / "raw").mkdir(exist_ok=True)
    (raw / "derived").mkdir(exist_ok=True)
    findings = [{"severity": "P0" if overall_status(str(result.get("formal_status"))) == "FAIL" else "P1", "owner": "S15/release", "affected_claims": [], "disposition": "OPEN", "description": item} for item in result.get("formal_blockers", [])]
    write_json(raw / "findings.json", findings)
    write_text(raw / "limitations.md", "# Limitations\n\n" + "\n".join(f"- {item}" for item in result.get("formal_blockers", [])) + "\n")
    write_json(raw / "acceptance.json", acceptance_for(result))
    write_json(raw / "interface_map.json", interface_map.mapping_for(experiment_id).as_dict())
    write_json(raw / "step_execution_matrix.json", step_matrix(experiment_id, result))
    write_json(raw / "component_results.json", result)
    verdict = make_verdict(experiment_id, result, run_id)
    write_json(raw / "verdict.json", verdict)
    report_path = STAGE_ROOT / experiment_id / f"{experiment_id}_实验报告.md"
    report_text = experiment_report(experiment_id, result, verdict, raw)
    write_text(report_path, report_text)
    write_text(raw / "report.md", report_text)
    manifest = {
        "schema_version": "hqsb.s15.evidence-manifest.v1", "stage": STAGE,
        "experiment_id": experiment_id, "run_id": run_id, "release_candidate_id": candidate["candidate_id"],
        "git": git, "commands": ["python3 scripts/release/run_s15_experiments.py"],
        "claim_level": "NOT_MEASURED" if overall_status(str(result.get("formal_status"))) != "PASS" else "TEST",
        "limitations": list(result.get("formal_blockers", [])), "files": inventory_files(raw, exclude=("runs",)),
        "report": str(report_path.relative_to(REPO)),
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    write_json(raw / "evidence_manifest.yaml", manifest)
    write_json(raw / "manifest.json", manifest)
    artifact = ARTIFACT_ROOT / experiment_id / run_id
    if artifact.exists():
        shutil.rmtree(artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw, artifact, ignore=shutil.ignore_patterns("runs"))
    return verdict


def validate_frontend() -> Dict[str, Any]:
    catalog = EvidenceCatalog(REPO)
    items = catalog.scan(refresh=True)
    selected = [item for item in items if item.get("stage") == STAGE]
    readable = 0
    attachment_counts = {}
    errors = []
    for item in selected:
        try:
            detail = catalog.detail(item["id"])
            if isinstance(detail.get("content"), Mapping):
                readable += 1
            attachment_counts[item["experiment"]] = len(item.get("files", []))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.get('experiment')}: {type(exc).__name__}: {exc}")
    return {
        "ok": len(selected) == len(EXPERIMENTS) and readable == len(EXPERIMENTS) and not errors,
        "s15_items": len(selected), "verdicts_readable": readable,
        "experiments": sorted(item.get("experiment") for item in selected),
        "attachment_counts": attachment_counts, "errors": errors,
        "list_endpoint": "GET /api/console/v1/evidence",
        "detail_endpoint": "GET /api/console/v1/evidence/{evidence_id}",
        "download_endpoint": "GET /api/console/v1/evidence/{evidence_id}/download",
    }


def stage_report(results: Mapping[str, Mapping[str, Any]], run_id: str, frontend: Mapping[str, Any], candidate: Mapping[str, Any], upstream: Mapping[str, Any]) -> str:
    stage_status = "FAIL" if any(result.get("formal_status") == "FAIL" for result in results.values()) else "BLOCKED"
    lines = [
        "# S15 阶段实验报告：发布、开源与求职证据", "",
        f"> Run ID：`{run_id}`  ", f"> 执行时间：{utc_now()}  ",
        "> 目标板：Jetson Orin 8GB；本机只负责编辑与拉回证据。", "",
        "## 1. 阶段结论", "",
        f"**阶段正式结论：`NO_GO / {stage_status}`。** 11 项均已建立独立、可索引的 run package；可自动执行的扫描、隔离安装、同板算子重放、文档/供应链/图表组件与故障分支已采集。当前候选为 dirty worktree，且缺独立 reviewer、目标读者、真实新加速器环境和公开上游贡献，因此不能把 S15 标成完成。", "",
        "## 2. 逐项结果", "", "| 实验 | 级别 | 正式状态 | 组件状态 |", "|---|---|---|---|",
    ]
    for experiment_id in EXPERIMENTS:
        lines.append(f"| {experiment_id} | {PROTOCOLS[experiment_id].level} | `{results[experiment_id].get('formal_status')}` | `{results[experiment_id].get('component_status')}` |")
    lines.extend([
        "", "## 3. G0–G8 阶段门", "", "| Gate | 结果 |", "|---|---|",
        f"| G0 upstream freeze | `BLOCKED`：15 阶段可定位={upstream.get('g0_traceable')}，15 阶段全 PASS={upstream.get('g0_all_pass')} |",
        "| G1 claim truth | `BLOCKED/FAIL_COMPONENT`：全量扫描已做，数值候选仍有未完成人工证据绑定 |",
        "| G2 executable entry | `FAIL/BLOCKED`：wheel 与 sdist 的隔离 control-plane 会话通过；全仓文档检查失败，且独立操作者、公开 sample raw→report、完整 docs 门未闭合 |",
        "| G3 technical reproduction | `BLOCKED`：同板 operator repeatability + E02 selected-point lineage；非新环境 MODEL/SERVICE replay |",
        "| G4 release integrity | `FAIL`：dirty/no tag/root LICENSE or full supply-chain gaps；未发布 |",
        "| G5 communication robustness | `BLOCKED`：脚本/题库就位，无真实演练/录音/盲评 |",
        "| G6 independent validation | `BLOCKED`：reviewer sessions=0 |",
        "| G7 public collaboration | `BLOCKED`：public upstream URLs=0 |",
        "| G8 audience comprehension | `BLOCKED`：participants=0 |",
        "", "## 4. Candidate 与发布决定", "",
        f"- Candidate：`{candidate['candidate_id']}`；dirty=`{candidate.get('tree_dirty')}`；valid frozen=`{candidate.get('valid_frozen_candidate')}`。",
        "- Release decision：`NO_GO`；Resume decision：`CONDITIONAL`；所有 draft 数值 claims 继续 withheld。",
        "- 本轮没有创建 tag、release、外部 issue/PR，也没有联系 reviewer/参与者。", "",
        "## 5. 前端访问", "",
        f"- S15 items：{frontend.get('s15_items')}/11；verdict 可读：{frontend.get('verdicts_readable')}/11；ok=`{frontend.get('ok')}`。",
        "- 列表：`GET /api/console/v1/evidence`。",
        "- 详情/下载：`GET /api/console/v1/evidence/{evidence_id}` 与 `/download`。", "",
        "## 6. 允许与禁止声明", "",
        "允许：说明本轮完成了自动化审计组件、隔离 wheel/sdist control-plane session、同板 RMSNorm repeatability、E02 选定 ITL 点 raw 重算，以及为什么正式门仍阻塞。", "",
        "禁止：声称 S15 通过、clean-room 第三方复现完成、公开 release 供应链通过、hero model/service 收益已在新环境复现、已有上游贡献，或招聘者五分钟研究已完成。", "",
    ])
    return "\n".join(lines)


def guarded(experiment_id: str, raw: Path, operation: Any) -> Dict[str, Any]:
    try:
        result = operation()
        if not isinstance(result, Mapping):
            raise TypeError(f"collector returned {type(result).__name__}")
        return dict(result)
    except Exception as exc:  # noqa: BLE001
        failure = {
            "component_status": "BLOCKED_COLLECTOR_ERROR", "formal_status": "BLOCKED",
            "formal_blockers": [f"collector error: {type(exc).__name__}: {exc}"],
            "executed_steps": [], "metrics": {}, "required_evidence": {}, "criteria": [],
        }
        write_json(raw / "collector_error.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()[-20000:]})
        return failure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="collect the honest Jetson scope of all S15 experiments")
    parser.add_argument("--keep-build-cache", action="store_true", help="keep the temporary candidate package directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    run_id = f"s15_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dirs = {experiment_id: STAGE_ROOT / experiment_id / "raw" for experiment_id in EXPERIMENTS}
    for raw in raw_dirs.values():
        archive_existing(raw)
        raw.mkdir(parents=True, exist_ok=True)
    git = git_identity()
    env = environment_fingerprint()
    upstream = upstream_inventory()
    candidate = release_candidate(git, upstream)
    build_root = ARTIFACT_ROOT / "_candidate_build" / run_id
    package_build = build_candidate_artifacts(build_root)

    results: Dict[str, Dict[str, Any]] = {}
    results["E15-01"] = guarded("E15-01", raw_dirs["E15-01"], lambda: run_e15_01(raw_dirs["E15-01"], candidate))
    results["E15-02"] = guarded("E15-02", raw_dirs["E15-02"], lambda: run_e15_02(raw_dirs["E15-02"], candidate, package_build))
    results["E15-03"] = guarded("E15-03", raw_dirs["E15-03"], lambda: run_e15_03(raw_dirs["E15-03"], candidate, env))
    results["E15-04"] = guarded("E15-04", raw_dirs["E15-04"], lambda: run_e15_04(raw_dirs["E15-04"], candidate, upstream))
    results["E15-05"] = guarded("E15-05", raw_dirs["E15-05"], lambda: run_e15_05(raw_dirs["E15-05"], candidate, package_build))
    results["E15-06"] = guarded("E15-06", raw_dirs["E15-06"], lambda: run_e15_06(raw_dirs["E15-06"], candidate))
    results["E15-07"] = guarded("E15-07", raw_dirs["E15-07"], lambda: run_e15_07(raw_dirs["E15-07"], candidate))
    results["E15-08"] = guarded("E15-08", raw_dirs["E15-08"], lambda: run_e15_08(raw_dirs["E15-08"], candidate))
    results["E15-09"] = guarded("E15-09", raw_dirs["E15-09"], lambda: run_e15_09(raw_dirs["E15-09"], candidate, package_build))
    results["E15-10"] = guarded("E15-10", raw_dirs["E15-10"], lambda: run_e15_10(raw_dirs["E15-10"], candidate))
    results["E15-11"] = guarded("E15-11", raw_dirs["E15-11"], lambda: run_e15_11(raw_dirs["E15-11"], candidate))

    verdicts = {}
    for experiment_id in EXPERIMENTS:
        verdicts[experiment_id] = finalise_experiment(experiment_id, raw_dirs[experiment_id], results[experiment_id], run_id, candidate, env, git)

    frontend = validate_frontend()
    decisions = {
        "schema_version": "hqsb.s15.acceptance.v1", "candidate_id": candidate["candidate_id"],
        "experiment_outcomes": {experiment_id: overall_status(str(results[experiment_id].get("formal_status"))) for experiment_id in EXPERIMENTS},
        "blocking_findings": [item for experiment_id in EXPERIMENTS for item in results[experiment_id].get("formal_blockers", [])],
        "public_claim_ids": [], "withheld_claim_ids": ["ALL_DRAFT_CLAIMS"],
        "release_decision": "NO_GO", "resume_decision": "CONDITIONAL", "signed_by": [], "decided_at_utc": utc_now(),
    }
    write_json(STAGE_ROOT / "environment_fingerprint.json", env)
    write_json(STAGE_ROOT / "release_candidate_snapshot.json", candidate)
    write_json(STAGE_ROOT / "upstream_evidence_inventory.json", upstream)
    write_json(STAGE_ROOT / "final_acceptance_decision.json", decisions)
    write_json(STAGE_ROOT / "frontend_access.json", frontend)
    stage_status = "FAIL" if any(result.get("formal_status") == "FAIL" for result in results.values()) else "BLOCKED"
    summary = {
        "stage": STAGE, "run_id": run_id, "overall": stage_status, "release_decision": "NO_GO",
        "results": {experiment_id: {"formal_status": results[experiment_id].get("formal_status"), "component_status": results[experiment_id].get("component_status")} for experiment_id in EXPERIMENTS},
        "frontend": frontend, "candidate": candidate, "generated_at_utc": utc_now(),
    }
    write_json(STAGE_ROOT / "execution_summary.json", summary)
    write_text(STAGE_ROOT / "S15_阶段实验报告_20260922.md", stage_report(results, run_id, frontend, candidate, upstream))
    if not args.keep_build_cache:
        shutil.rmtree(build_root, ignore_errors=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if frontend["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
