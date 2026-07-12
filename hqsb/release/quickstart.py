"""E15-02 — clean CPU quickstart and the minimum-threshold reproduction.

Protocol: ``docs/stage_experiments/details/S15/E15-02_clean_cpu_quickstart_reproduction.md``
(45 steps).

The experiment is a *time-to-event* study of one honest question: can someone who
has never seen HQSB, on a fresh CPU-only environment, follow only the public
quickstart and reach a verified report — without the author's machine state,
caches, GPUs, private models or verbal help?

The module supplies the machinery the protocol needs, not the session itself:

* :class:`QuickstartContract` — the frozen definition of "ran through" (step 1);
* :class:`SupportMatrix` — supported OS/Python/install/cache/network cells (2/43);
* :class:`SessionIdentity` — what makes a session *independent* (step 3), and the
  explicit rule that a repeat inside one environment is not a second sample;
* :data:`SCENARIO_MATRIX` — the §9 expectation table (legal outcome *and* the
  forbidden shortcut for every scenario);
* :class:`TotalClock` — the 8-stage decomposition that starts when the operator
  opens the page, not when the install finishes (steps 9, 39, 40);
* :class:`InterventionLog` — the L0–L4-style scale of §9.2, where a material
  intervention (author supplies a command) fails the current session;
* :func:`initial_environment_evidence`, :func:`prove_no_accelerator`,
  :func:`prove_no_hqsb_residue` — the three "this really was clean" proofs
  (steps 5–7);
* :func:`compare_metadata`, :func:`check_optional_isolation`,
  :func:`check_capability_probe`, :func:`compare_canonical`,
  :func:`check_byte_stable`, :func:`scan_private_state`, :func:`judge_time_goal`
  — the checks of steps 14–29 and 37–40;
* :func:`negative_case_plan` — the corruption/permission/offline/missing-extra
  cases with their legal and forbidden outcomes (steps 32–36);
* :func:`adjudicate_gates` — per-gate verdicts, so one happy path cannot hide a
  broken error path (step 44).

Nothing here installs a package, downloads a sample or runs a session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, is_digest

#: The eight timing stages of the total clock (§3.2).
TIMING_STAGES: Tuple[str, ...] = rec.QuickstartSessionResult.TIMING_STAGES

#: Default target of the protocol; a real session reports its own measurement.
TARGET_TOTAL_MINUTES = 30.0

#: Installation modes a session can use.
INSTALL_MODES: Tuple[str, ...] = ("wheel", "sdist")

#: Cache and network states a session can run under.
CACHE_STATES: Tuple[str, ...] = ("cold", "warm")
NETWORK_STATES: Tuple[str, ...] = ("online", "offline")

#: Allowed prerequisites a quickstart may name (§9.1 "explicitly requested").
ALLOWED_PREREQUISITES: Tuple[str, ...] = (
    "python>=3.10",
    "pip",
    "a POSIX shell",
    "roughly 1 GB disk for the core install",
)

#: The successful event the clock stops at (step 1).
SUCCESS_EVENT = "first_verified_report"

#: The §9 scenario matrix: what must be observed, what is legal, what is forbidden.
SCENARIO_MATRIX: Tuple[Mapping[str, str], ...] = (
    {
        "scenario": "clean online/cold",
        "must_observe": "完整下载、安装、首次报告",
        "legal": "在时限内成功或如实超时 FAIL",
        "forbidden": "借用宿主 cache",
    },
    {
        "scenario": "clean online/warm",
        "must_observe": "明确 cache hit 与加速来源",
        "legal": "输出语义等价",
        "forbidden": "将 warm 时长当首次体验",
    },
    {
        "scenario": "offline + bundled sample",
        "must_observe": "无网络解析 sample",
        "legal": "本地重建成功",
        "forbidden": "暗中访问模型源",
    },
    {
        "scenario": "offline + required remote model",
        "must_observe": "preflight 列出缺失 artifact",
        "legal": "早失败并给获取步骤",
        "forbidden": "深层 traceback/自动换模型",
    },
    {
        "scenario": "no accelerator libs",
        "must_observe": "core import/schema/report",
        "legal": "正常运行",
        "forbidden": "顶层 import CUDA/CANN 失败",
    },
    {
        "scenario": "request accelerator feature",
        "must_observe": "feature-scoped capability error",
        "legal": "明确 extra/device/reason",
        "forbidden": "静默 CPU fallback",
    },
    {
        "scenario": "corrupted cache",
        "must_observe": "digest mismatch",
        "legal": "拒绝、隔离、重取建议",
        "forbidden": "使用坏文件继续报告",
    },
    {
        "scenario": "unwritable cache",
        "must_observe": "原子写与替代路径",
        "legal": "可诊断失败/指定新目录",
        "forbidden": "留下半写 cache",
    },
    {
        "scenario": "invalid config/schema",
        "must_observe": "字段路径和版本",
        "legal": "稳定非零 exit",
        "forbidden": "忽略字段用默认值",
    },
    {
        "scenario": "wrong artifact version",
        "must_observe": "package/sample/Schema identity",
        "legal": "在执行前拒绝",
        "forbidden": "混版本后生成报告",
    },
)

#: The §9.2 intervention scale: what still counts as an independent session.
INTERVENTION_LEVELS: Tuple[Mapping[str, str], ...] = (
    {"level": "NONE", "example": "完全按公开文档完成", "affects_pass": "no"},
    {"level": "DISCOVERED_NOT_DOCUMENTED", "example": "用 --help 发现未文档参数", "affects_pass": "records a docs finding"},
    {"level": "CLARIFICATION", "example": "作者解释概念但不提供新动作", "affects_pass": "records a docs finding"},
    {
        "level": "MATERIAL_INTERVENTION",
        "example": "作者提供命令/环境变量/文件，或操作者自行补装系统包",
        "affects_pass": "current session cannot pass",
    },
)

#: Non-semantic fields a canonical comparison may ignore (frozen *before* the run).
DEFAULT_IGNORED_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "generated_at",
    "absolute_path",
    "host",
    "user",
    "duration_s",
)

#: Fields that must never be ignored (they are the experiment's facts).
PROTECTED_FIELDS: Tuple[str, ...] = (
    "status",
    "unit",
    "metric",
    "backend",
    "schema_version",
    "claim_refs",
    "result_id",
)

#: Private-state patterns the outputs must not contain (step 37).
PRIVACY_PATTERNS: Mapping[str, str] = {
    "absolute_home": r"(?:/root/|/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/)",
    "token_like": r"(?i)(?:token|api[-_]?key|secret)\s*[:=]\s*[A-Za-z0-9_\-]{12,}",
    "private_host": r"(?:localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)",
    "internal_url": r"(?i)https?://[^\s]*(?:internal|corp|intranet)[^\s]*",
    "machine_id": r"(?i)machine[-_ ]?id\s*[:=]\s*\S+",
    "private_model_path": r"(?i)(?:/data/|/mnt/)[^\s]*\.safetensors",
}


@dataclass
class QuickstartContract:
    """The frozen contract of step 1: what counts as "ran through"."""

    candidate_id: str
    public_page: str
    target_user: str
    target_total_minutes: float = TARGET_TOTAL_MINUTES
    success_event: str = SUCCESS_EVENT
    allowed_prerequisites: Tuple[str, ...] = ALLOWED_PREREQUISITES
    allowed_author_help: str = "none"
    environment_matrix: Tuple[str, ...] = ()
    install_modes: Tuple[str, ...] = INSTALL_MODES

    schema_version = f"{SCHEMA_PREFIX}.quickstart-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.candidate_id:
            findings.append("QuickstartContract: candidate_id is required")
        if not self.public_page or not self.public_page.startswith(("docs/", "README")):
            findings.append("QuickstartContract: the public quickstart page must be a repository path")
        if not self.target_user:
            findings.append("QuickstartContract: the target user must be described")
        if self.target_total_minutes <= 0:
            findings.append("QuickstartContract: a non-positive time target is not a target")
        if self.success_event != SUCCESS_EVENT:
            findings.append(f"QuickstartContract: success event must be {SUCCESS_EVENT!r}")
        if self.allowed_author_help != "none":
            findings.append(
                "QuickstartContract: author help must be 'none'; a session that needed help is recorded, "
                "not relabelled (E15-02 §9.2)"
            )
        if not self.environment_matrix:
            findings.append("QuickstartContract: the support matrix must be declared before running")
        for mode in self.install_modes:
            if mode not in INSTALL_MODES:
                findings.append(f"QuickstartContract: unknown install mode {mode!r}")
        return findings


@dataclass
class SupportCell:
    """One support-matrix cell (OS × Python × install × cache × network)."""

    os_id: str
    python_version: str
    install_mode: str
    cache_state: str
    network_state: str
    result: str = rec.STATUS_NOT_RUN
    session_id: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.install_mode not in ("wheel", "sdist", "both") and self.install_mode:
            findings.append(f"SupportCell: unknown install mode {self.install_mode!r}")
        if self.cache_state and self.cache_state not in CACHE_STATES:
            findings.append(f"SupportCell: unknown cache state {self.cache_state!r}")
        if self.network_state and self.network_state not in NETWORK_STATES:
            findings.append(f"SupportCell: unknown network state {self.network_state!r}")
        if self.result not in rec.ALL_STATUSES:
            findings.append(f"SupportCell: unknown result {self.result!r}")
        if self.result == rec.STATUS_PASS and not self.session_id:
            findings.append("SupportCell: a PASS needs the session that produced it")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "os_id": self.os_id,
            "python_version": self.python_version,
            "install_mode": self.install_mode,
            "cache_state": self.cache_state,
            "network_state": self.network_state,
            "result": self.result,
            "session_id": self.session_id,
        }


class SupportMatrix:
    """The declared support cells plus the rule that unverified cells are not "supported"."""

    def __init__(self, cells: Sequence[SupportCell]) -> None:
        self.cells = list(cells)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.cells:
            findings.append("SupportMatrix: an empty matrix declares nothing")
        for cell in self.cells:
            findings.extend(cell.problems())
        keys = [(cell.os_id, cell.python_version, cell.install_mode, cell.cache_state, cell.network_state) for cell in self.cells]
        if len(set(keys)) != len(keys):
            findings.append("SupportMatrix: duplicate support cells")
        return findings

    def unverified(self) -> List[Dict[str, Any]]:
        return [cell.as_dict() for cell in self.cells if cell.result != rec.STATUS_PASS]

    def coverage(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cell in self.cells:
            counts[cell.result] = counts.get(cell.result, 0) + 1
        return {key: counts[key] for key in sorted(counts)}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.support-matrix.v1",
            "cells": [cell.as_dict() for cell in self.cells],
            "coverage": self.coverage(),
            "unverified": self.unverified(),
        }


@dataclass
class SessionIdentity:
    """What makes one quickstart session independent (step 3)."""

    session_id: str
    host_kind: str  # fresh_vm | fresh_container | clean_user | reused (→ repeat, not independent)
    operator_id: str
    environment_id: str
    is_repeat: bool = False
    parent_session: str = ""

    HOST_KINDS: Tuple[str, ...] = ("fresh_vm", "fresh_container", "clean_user", "reused")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.session_id or not self.operator_id or not self.environment_id:
            findings.append("SessionIdentity: session_id, operator_id and environment_id are required")
        if self.host_kind not in self.HOST_KINDS:
            findings.append(f"SessionIdentity: unknown host_kind {self.host_kind!r}")
        if self.host_kind == "reused" and not self.is_repeat:
            findings.append(
                "SessionIdentity: a session in a reused environment is a repeat, not a second independent "
                "environment (§5 step 3)"
            )
        if self.is_repeat and not self.parent_session:
            findings.append("SessionIdentity: a repeat must name the session it repeats")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "host_kind": self.host_kind,
            "operator_id": self.operator_id,
            "environment_id": self.environment_id,
            "is_repeat": self.is_repeat,
            "parent_session": self.parent_session,
        }


def initial_environment_evidence(facts: Mapping[str, Any]) -> Dict[str, Any]:
    """Record the initial environment, redacting sensitive values (step 5)."""
    required = (
        "os",
        "arch",
        "cpu",
        "memory_bytes",
        "disk_free_bytes",
        "python_version",
        "pip_version",
        "locale",
        "proxy",
        "container",
        "recorded_at_utc",
    )
    missing = [name for name in required if name not in facts]
    if missing:
        raise ConfigError(f"initial environment evidence is missing {', '.join(missing)}")
    redacted: Dict[str, Any] = {}
    for key, value in sorted(facts.items()):
        if key in ("proxy", "registry_credentials", "token"):
            redacted[key] = "<redacted>"
            continue
        redacted[key] = value
    return {"schema_version": f"{SCHEMA_PREFIX}.initial-environment.v1", "facts": redacted}


def prove_no_accelerator(probe: Mapping[str, Any]) -> List[str]:
    """The "no accelerator was available" proof of step 6."""
    findings: List[str] = []
    for name in ("devices", "drivers", "env_vars", "capability_probe"):
        if name not in probe:
            findings.append(f"accelerator proof is missing {name!r}")
    devices = list(probe.get("devices", []))
    if devices:
        findings.append(f"the session had accelerator devices attached: {devices!r} (not a clean CPU session)")
    env_vars = list(probe.get("env_vars", []))
    leaked = [name for name in env_vars if name.startswith(("CUDA_", "ASCEND_", "NPU_"))]
    if leaked:
        findings.append(f"accelerator environment variables leaked into the session: {leaked!r}")
    capability = probe.get("capability_probe")
    if not isinstance(capability, Mapping) or not capability.get("reason"):
        findings.append(
            "the capability probe must report GPU/NPU unavailable with a structured reason, not a traceback"
        )
    return findings


def prove_no_hqsb_residue(residue: Mapping[str, Any]) -> List[str]:
    """The "no HQSB state survived" proof of step 7."""
    findings: List[str] = []
    for name in ("installed_package", "editable_path", "project_cache", "model_dir", "old_config"):
        if name not in residue:
            findings.append(f"residue check is missing {name!r} (unknown is not clean)")
    if residue.get("installed_package"):
        findings.append("hqsb is already installed in the 'clean' environment")
    if residue.get("editable_path"):
        findings.append("an editable install path is present in the 'clean' environment")
    if residue.get("project_cache"):
        findings.append("a project cache is present; the session would consume host state")
    if residue.get("model_dir"):
        findings.append("a model directory is present; the CPU quickstart must not need one")
    if residue.get("old_config"):
        findings.append("an old HQSB config is present")
    return findings


class TotalClock:
    """The 8-stage total clock; it starts when the *reader* starts (steps 9, 39)."""

    def __init__(self, contract: QuickstartContract) -> None:
        self.contract = contract
        self.started_at_utc = ""
        self.first_verified_report_at_utc = ""
        self.stages: Dict[str, float] = {}

    def start(self, started_at_utc: str) -> None:
        if self.started_at_utc:
            raise ConfigError("the total clock already started; a second start would hide the reading time")
        self.started_at_utc = started_at_utc

    def record_stage(self, stage: str, duration_s: float) -> None:
        if stage not in TIMING_STAGES:
            raise ConfigError(f"unknown timing stage {stage!r}; known: {', '.join(TIMING_STAGES)}")
        if duration_s < 0:
            raise ConfigError("a stage duration may not be negative")
        self.stages[stage] = self.stages.get(stage, 0.0) + duration_s

    def finish(self, first_verified_report_at_utc: str) -> Dict[str, Any]:
        if not self.started_at_utc:
            raise ConfigError("the clock never started; the reading time would be missing from the total")
        if not first_verified_report_at_utc:
            raise ConfigError("the clock can only finish at the first verified report")
        self.first_verified_report_at_utc = first_verified_report_at_utc
        return {
            "started_at_utc": self.started_at_utc,
            "first_verified_report_at_utc": first_verified_report_at_utc,
            "stage_durations_s": {key: self.stages[key] for key in sorted(self.stages)},
            "measured_stages": len(self.stages),
            "missing_stages": [stage for stage in TIMING_STAGES if stage not in self.stages],
        }

    def total_s(self) -> float:
        return float(sum(self.stages.values()))


class InterventionLog:
    """Undocumented input tracking (§9.2); a material intervention fails the session."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, str]] = []

    def record(self, level: str, description: str, actor: str = "operator") -> Dict[str, str]:
        known = {entry["level"] for entry in INTERVENTION_LEVELS}
        if level not in known:
            raise ConfigError(f"unknown intervention level {level!r}; known: {', '.join(sorted(known))}")
        if not description:
            raise ConfigError("an intervention needs a description")
        entry = {"level": level, "description": description, "actor": actor}
        self.entries.append(entry)
        return entry

    def material(self) -> List[Dict[str, str]]:
        return [entry for entry in self.entries if entry["level"] == "MATERIAL_INTERVENTION"]

    def blocks_pass(self) -> bool:
        return bool(self.material())

    def as_dict(self) -> Dict[str, Any]:
        return {"entries": list(self.entries), "material": self.material(), "blocks_pass": self.blocks_pass()}


def check_document_prerequisites(documented: Sequence[str], discovered: Sequence[str]) -> List[Dict[str, str]]:
    """Every system step the document did not list is a finding (step 10)."""
    findings: List[Dict[str, str]] = []
    for step in discovered:
        if step not in documented:
            findings.append(
                {
                    "kind": "DISCOVERED_NOT_DOCUMENTED",
                    "step": step,
                    "note": "operator had to supply this step from experience; it belongs in the document",
                }
            )
    return findings


def verify_download_urls(downloads: Sequence[Mapping[str, Any]], *, expected: Mapping[str, str]) -> List[str]:
    """Check redirects, status, size and digest of each download (step 11)."""
    findings: List[str] = []
    for item in downloads:
        name = str(item.get("name", ""))
        status = int(item.get("status", 0))
        if status != 200:
            findings.append(f"download {name!r} returned HTTP {status}")
        if item.get("redirected_to") and not item.get("final_digest"):
            findings.append(f"download {name!r} redirected but no final digest was recorded")
        digest = str(item.get("final_digest", ""))
        if not is_digest(digest):
            findings.append(f"download {name!r} has no sha256 digest")
        elif name in expected and expected[name] != digest:
            findings.append(f"download {name!r} digest {digest} != expected {expected[name]}")
        if item.get("content_type", "").startswith("text/html") and name.endswith((".whl", ".tar.gz")):
            findings.append(f"download {name!r} returned HTML (an error page with status 200)")
    return findings


def compare_metadata(wheel: Mapping[str, Any], sdist: Mapping[str, Any]) -> List[str]:
    """Wheel/sdist identity comparison of step 14."""
    findings: List[str] = []
    for field_name in ("name", "version", "requires_python", "dependencies", "extras", "license", "entry_points", "source_commit"):
        left = wheel.get(field_name)
        right = sdist.get(field_name)
        if isinstance(left, list):
            left = sorted(map(str, left))
        if isinstance(right, list):
            right = sorted(map(str, right))
        if left != right:
            findings.append(f"wheel/sdist metadata disagreement on {field_name!r}: {left!r} != {right!r}")
    return findings


def check_core_import(*, working_directory_outside_repo: bool, import_time_s: float, loaded_modules: Sequence[str]) -> List[str]:
    """The import purity check of steps 15–16."""
    findings: List[str] = []
    if not working_directory_outside_repo:
        findings.append("the import was executed inside the source tree; a relative-path accident cannot be excluded")
    if import_time_s < 0:
        findings.append("negative import time")
    heavy = [
        name
        for name in loaded_modules
        if name.split(".")[0] in ("torch", "triton", "numpy", "cann", "acl", "vllm", "transformers")
    ]
    if heavy:
        findings.append(f"core import pulled heavy optional stacks: {sorted(set(heavy))}")
    return findings


def check_optional_isolation(*, requested_feature: str, outcome: Mapping[str, Any]) -> List[str]:
    """A requested extra must fail inside its own scope, not at core import (steps 16, 35)."""
    findings: List[str] = []
    if requested_feature == "core":
        return findings
    if outcome.get("core_import_failed"):
        findings.append(
            "requesting an optional feature broke the core import; feature-scoped errors are required (E15-02 step 35)"
        )
    extra = outcome.get("required_extra", "")
    reason = outcome.get("reason", "")
    if not extra or not reason:
        findings.append("the feature-scoped error must name the required extra and a reason")
    if outcome.get("silent_fallback"):
        findings.append("a silently substituted implementation is not a feature-scoped error (§3.4)")
    return findings


def check_capability_probe(report: Mapping[str, Any]) -> List[str]:
    """The probe must be structured, honest and traceback-free (steps 18, 35)."""
    findings: List[str] = []
    if "cpu_available" not in report:
        findings.append("capability probe must report CPU availability")
    if "accelerators" not in report or not isinstance(report.get("accelerators"), list):
        findings.append("capability probe must list accelerators (possibly empty) as data")
    else:
        for item in report["accelerators"]:
            if not isinstance(item, Mapping) or not item.get("reason"):
                findings.append("each unavailable accelerator needs a structured reason, not a traceback")
    if report.get("traceback"):
        findings.append("the capability probe raised instead of reporting state")
    if report.get("silent_backend_change"):
        findings.append("the capability probe changed the backend silently (§3.4)")
    return findings


def verify_sample_bundle(manifest: Mapping[str, Any], *, files: Mapping[str, str]) -> List[str]:
    """Sample provenance, license, size and per-file hashes (steps 19–20)."""
    findings: List[str] = []
    for name in ("source", "version", "license", "size_bytes", "files"):
        if name not in manifest:
            findings.append(f"sample manifest is missing {name!r}")
    declared = manifest.get("files")
    if isinstance(declared, Mapping):
        missing = sorted(set(declared) - set(files))
        unexpected = sorted(set(files) - set(declared))
        for path in missing:
            findings.append(f"sample manifest lists {path!r} but the bundle does not contain it")
        for path in unexpected:
            findings.append(f"sample bundle contains undeclared file {path!r}")
        for path, digest in sorted(declared.items()):
            if path in files and digest != files[path]:
                findings.append(f"sample file {path!r} digest mismatch")
    if manifest.get("contains_model_weights"):
        findings.append("the CPU sample must not bundle model weights")
    return findings


def compare_canonical(
    published: Mapping[str, Any],
    regenerated: Mapping[str, Any],
    *,
    ignored_fields: Sequence[str] = DEFAULT_IGNORED_FIELDS,
) -> List[str]:
    """Compare canonical outputs under the *pre-registered* ignore set (step 27).

    Widening ``ignored_fields`` after seeing a diff is the failure the protocol
    forbids, so the protected fields are refused as ignored values outright.
    """
    findings: List[str] = []
    for field_name in ignored_fields:
        if field_name in PROTECTED_FIELDS:
            findings.append(
                f"field {field_name!r} may not be ignored: the ignore set must be frozen before the run "
                "(E15-02 step 27/§11)"
            )
    if findings:
        return findings
    for key in sorted(set(published) | set(regenerated)):
        if key in ignored_fields:
            continue
        left = published.get(key)
        right = regenerated.get(key)
        if isinstance(left, list):
            left = sorted(map(str, left))
        if isinstance(right, list):
            right = sorted(map(str, right))
        if left != right:
            findings.append(f"canonical diff on {key!r}: {left!r} != {right!r}")
    return findings


def check_byte_stable(*, paths: Sequence[str], digest_of: Callable[[str], str], recorded: Mapping[str, str]) -> List[str]:
    """Byte-stable objects (config/manifest/normalized JSON) must hash equal (step 28)."""
    findings: List[str] = []
    for path in paths:
        expected = recorded.get(path)
        if not expected:
            findings.append(f"byte-stable object {path!r} has no recorded digest")
            continue
        actual = digest_of(path)
        if actual != expected:
            findings.append(f"byte-stable object {path!r} changed ({actual} != {expected})")
    return findings


def scan_private_state(texts: Mapping[str, str]) -> List[str]:
    """Scan outputs for private state (step 37)."""
    findings: List[str] = []
    for name, text in sorted(texts.items()):
        for kind, pattern in PRIVACY_PATTERNS.items():
            if re.search(pattern, text):
                findings.append(f"{name}: leaked {kind} (pattern {pattern!r})")
    return findings


def analyze_critical_path(
    *, cold: Mapping[str, float], warm: Optional[Mapping[str, float]] = None
) -> Dict[str, Any]:
    """Decompose the critical path and compare cold/warm (step 39)."""
    missing = [stage for stage in TIMING_STAGES if stage not in cold]
    analysis: Dict[str, Any] = {
        "cold_total_s": float(sum(cold.values())),
        "cold_share": {key: cold[key] for key in sorted(cold)},
        "missing_stages": missing,
    }
    if warm is not None:
        analysis["warm_total_s"] = float(sum(warm.values()))
        analysis["delta_s"] = analysis["cold_total_s"] - analysis["warm_total_s"]
        analysis["note"] = "a warm run may not be quoted as the first-time experience (§11)"
    return analysis


def judge_time_goal(contract: QuickstartContract, *, measured_total_s: float, stage_totals: Mapping[str, float]) -> Dict[str, Any]:
    """Judge the time target honestly (step 40)."""
    target_s = contract.target_total_minutes * 60.0
    missing = [stage for stage in TIMING_STAGES if stage not in stage_totals]
    return {
        "target_s": target_s,
        "measured_total_s": measured_total_s,
        "within_target": measured_total_s <= target_s and not missing,
        "excluded_from_measurement": missing,
        "verdict": rec.STATUS_PASS if measured_total_s <= target_s and not missing else rec.STATUS_FAIL,
        "note": "超时不得删除下载或排障时间；不完整的分段计时不能判 PASS",
    }


def map_gaps_to_owners(findings: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Map each gap to the owning component (step 41)."""
    owners = {
        "packaging": "packaging",
        "cli": "cli",
        "schema": "schema",
        "report": "report",
        "docs": "docs",
    }
    result: Dict[str, List[str]] = {owner: [] for owner in owners.values()}
    for finding in findings:
        kind = str(finding.get("kind", "docs"))
        owner = owners.get(kind, "docs")
        result[owner].append(str(finding.get("step", finding.get("description", "?"))))
    return {owner: sorted(items) for owner, items in result.items()}


def negative_case_plan() -> Tuple[Mapping[str, Any], ...]:
    """The injected negative cases of steps 32–36, as data."""
    matrix = {row["scenario"]: row for row in SCENARIO_MATRIX}
    return tuple(
        {
            "case_id": case,
            "description": matrix[scenario]["must_observe"],
            "legal_outcome": matrix[scenario]["legal"],
            "forbidden_outcome": matrix[scenario]["forbidden"],
        }
        for case, scenario in (
            ("cache_corruption", "corrupted cache"),
            ("unwritable_cache", "unwritable cache"),
            ("missing_optional_feature", "request accelerator feature"),
            ("invalid_config", "invalid config/schema"),
            ("offline_required_model", "offline + required remote model"),
            ("wrong_artifact_version", "wrong artifact version"),
        )
    )


def adjudicate_gates(
    *,
    install: str,
    import_core: str,
    cli: str,
    sample: str,
    schema: str,
    regeneration: str,
    negative_cases: str,
    time_goal: str,
    interventions: str,
    leakage: str,
) -> Dict[str, str]:
    """Per-gate verdicts (step 44): a happy path cannot hide a failed error path."""
    gates = {
        "install": install,
        "import": import_core,
        "cli": cli,
        "sample": sample,
        "schema": schema,
        "regeneration": regeneration,
        "negative_cases": negative_cases,
        "time": time_goal,
        "interventions": interventions,
        "leakage": leakage,
    }
    invalid = {gate: verdict for gate, verdict in gates.items() if verdict not in rec.ALL_STATUSES}
    if invalid:
        raise ConfigError(f"adjudicate_gates received non-status verdicts: {invalid}")
    return gates


def freeze_session(
    *,
    session: SessionIdentity,
    contract: QuickstartContract,
    clock: TotalClock,
    interventions: InterventionLog,
    result: rec.QuickstartSessionResult,
) -> Dict[str, Any]:
    """Freeze the session record, refusing a PASS the session did not earn (step 45)."""
    problems = list(session.problems()) + list(contract.problems()) + list(result.validate())
    if clock.started_at_utc and not result.started_at_utc:
        result.started_at_utc = clock.started_at_utc
    if result.status == rec.STATUS_PASS and interventions.blocks_pass():
        problems.append(
            "the session had a material intervention, so it cannot pass as an independent reproduction (§9.2)"
        )
    if result.status == rec.STATUS_PASS and not result.first_verified_report_at_utc:
        problems.append("a PASS needs the first-verified-report timestamp")
    if problems:
        raise ConfigError("refusing to freeze the session: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.quickstart-session.v1",
        "session": session.as_dict(),
        "contract": contract.schema_version,
        "clock": clock.finish(result.first_verified_report_at_utc or "unfinished"),
        "interventions": interventions.as_dict(),
        "record": result.as_dict(),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 QuickstartContract", ("quickstart.QuickstartContract",)),
    (2, "选择支持矩阵", ("quickstart.SupportMatrix", "quickstart.SupportCell")),
    (3, "定义独立 session", ("quickstart.SessionIdentity",)),
    (4, "冻结网络与缓存场景", ("quickstart.CACHE_STATES", "quickstart.NETWORK_STATES", "quickstart.SCENARIO_MATRIX")),
    (5, "建立初始环境证据", ("quickstart.initial_environment_evidence",)),
    (6, "证明环境中没有加速器依赖", ("quickstart.prove_no_accelerator",)),
    (7, "证明没有 HQSB 残留", ("quickstart.prove_no_hqsb_residue",)),
    (8, "创建动作记录器", ("telemetry.ActionLog", "telemetry.SegmentTiming")),
    (9, "开始总计时", ("quickstart.TotalClock.start", "quickstart.TIMING_STAGES")),
    (10, "执行文档前置检查", ("quickstart.check_document_prerequisites",)),
    (11, "验证下载地址和 TLS", ("quickstart.verify_download_urls",)),
    (12, "从 wheel 安装最小包", ("quickstart.QuickstartContract.install_modes", "telemetry.ActionLog.append")),
    (13, "从 sdist 构建安装", ("quickstart.QuickstartContract.install_modes",)),
    (14, "核对 wheel 与 sdist 元数据", ("quickstart.compare_metadata",)),
    (15, "执行 core import 测试", ("quickstart.check_core_import",)),
    (16, "验证 optional dependency 隔离", ("quickstart.check_optional_isolation", "quickstart.check_core_import")),
    (17, "检查 CLI 可发现性", ("quickstart.check_core_import", "telemetry.ActionLog")),
    (18, "运行环境 capability probe", ("quickstart.check_capability_probe",)),
    (19, "下载或读取公开 sample bundle", ("quickstart.verify_sample_bundle",)),
    (20, "校验 sample Evidence Manifest", ("quickstart.verify_sample_bundle", "identity.content_address_aggregate")),
    (21, "运行最小配置解析", ("quickstart.compare_canonical", "records.RUN_MANIFEST_FIELDS")),
    (22, "运行 sample reference 路径", ("quickstart.freeze_session", "records.QuickstartSessionResult")),
    (23, "验证 C1–C7 Schema", ("quickstart.adjudicate_gates", "quickstart.compare_canonical")),
    (24, "运行负 Schema 用例", ("quickstart.negative_case_plan",)),
    (25, "从 raw 重建 normalized result", ("quickstart.check_byte_stable", "quickstart.compare_canonical")),
    (26, "从 normalized 重建报告", ("quickstart.check_byte_stable", "quickstart.compare_canonical")),
    (27, "比较 canonical 输出", ("quickstart.compare_canonical", "quickstart.DEFAULT_IGNORED_FIELDS")),
    (28, "检查 byte-stable 对象", ("quickstart.check_byte_stable",)),
    (29, "记录资源峰值", ("records.QuickstartSessionResult", "telemetry.SegmentTiming")),
    (30, "执行 warm-cache repeat", ("quickstart.SessionIdentity", "quickstart.analyze_critical_path")),
    (31, "执行 cold-cache repeat", ("quickstart.SessionIdentity", "quickstart.SupportMatrix")),
    (32, "执行离线行为测试", ("quickstart.negative_case_plan", "quickstart.SCENARIO_MATRIX")),
    (33, "注入 cache corruption", ("quickstart.negative_case_plan", "campaign.REQUIRED_ISOLATION")),
    (34, "注入不可写 cache 目录", ("quickstart.negative_case_plan",)),
    (35, "注入缺失 optional feature", ("quickstart.check_optional_isolation", "quickstart.check_capability_probe")),
    (36, "注入无效配置", ("quickstart.check_optional_isolation", "quickstart.negative_case_plan")),
    (37, "扫描私有状态泄漏", ("quickstart.scan_private_state", "quickstart.PRIVACY_PATTERNS")),
    (38, "盘点未文档输入", ("quickstart.InterventionLog", "quickstart.INTERVENTION_LEVELS")),
    (39, "分析关键路径时间", ("quickstart.analyze_critical_path",)),
    (40, "判定 30 分钟目标", ("quickstart.judge_time_goal",)),
    (41, "修订文档与工具", ("quickstart.map_gaps_to_owners",)),
    (42, "由新操作者复测", ("quickstart.SessionIdentity",)),
    (43, "比较支持矩阵结果", ("quickstart.SupportMatrix.coverage", "quickstart.SupportMatrix.unverified")),
    (44, "逐门裁决", ("quickstart.adjudicate_gates",)),
    (45, "冻结 QuickstartReproductionRecord", ("quickstart.freeze_session", "records.QuickstartSessionResult")),
)

TITLE = "Clean CPU Quickstart 与最低门槛复现"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-02 只证明最低访问门槛真实、独立且可审计；不证明 CPU 推理性能或任何 kernel 正确性"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the quickstart interfaces (labelled smoke)."""
    problems: List[str] = []
    contract = QuickstartContract(
        candidate_id="cand-1",
        public_page="docs/tutorials/cpu_quickstart.md",
        target_user="a developer with no HQSB history",
        environment_matrix=("linux-x86_64/py3.10",),
    )
    problems.extend(contract.problems())
    clock = TotalClock(contract)
    clock.start("2026-01-01T00:00:00Z")
    for stage in TIMING_STAGES:
        clock.record_stage(stage, 1.0)
    finished = clock.finish("2026-01-01T00:30:00Z")
    if len(finished["stage_durations_s"]) != len(TIMING_STAGES):
        problems.append("the clock did not record every stage")
    unknown_stage_refused = False
    try:
        clock.record_stage("made_up_stage", 1.0)
    except ConfigError:
        unknown_stage_refused = True
    if not unknown_stage_refused:
        problems.append("an unknown timing stage was accepted")
    interventions = InterventionLog()
    interventions.record("MATERIAL_INTERVENTION", "author supplied an env var")
    if not interventions.blocks_pass():
        problems.append("a material intervention did not block the pass")
    privacy = scan_private_state({"stdout": "path=/root/artifacts/x.json"})
    if not privacy:
        problems.append("the privacy scan missed an absolute home path")
    canonical = compare_canonical({"status": "ok"}, {"status": "ok"}, ignored_fields=("status",))
    if not canonical:
        problems.append("ignoring a protected field was not refused")
    goal_ok = judge_time_goal(contract, measured_total_s=60.0, stage_totals={stage: 1.0 for stage in TIMING_STAGES})
    goal_incomplete = judge_time_goal(contract, measured_total_s=60.0, stage_totals={"read": 1.0})
    if goal_ok["verdict"] != rec.STATUS_PASS or goal_incomplete["verdict"] != rec.STATUS_FAIL:
        problems.append("the time-goal judge did not distinguish complete from partial timing")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "scenarios": len(SCENARIO_MATRIX),
        "negative_cases": len(negative_case_plan()),
        "privacy_findings": len(privacy),
        "problems": problems,
    }
