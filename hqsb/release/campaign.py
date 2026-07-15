"""S15 campaign layout and execution-safety policy.

Two jobs:

* **layout** — S15 writes *run output* under ``artifacts/S15/<experiment>/<run>``
  with the §22 uniform data package, while ``docs/stage_experiments/S15/**`` is
  the *protocol* and is read-only.  Mixing the two is how a protocol file gets
  overwritten by a result, so :func:`run_layout` and :func:`assert_writable` make
  the boundary executable;
* **safety** — S15 is the stage that can touch the outside world (public release,
  upstream repositories, demo recordings, reviewers, resume material), so the
  protocol spends several sections on what a run may *not* do
  (``E15-05`` step 41/42, ``E15-07`` step 38, ``E15-09`` §11, ``E15-10`` §11,
  manual §21).  Those prohibitions are data here, checked by
  :func:`check_execution_safety`, so an unsafe run is refused before it starts
  rather than reviewed afterwards.

Nothing in this module starts anything, publishes anything or contacts anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import assert_safe_segment, canonical_digest

#: Run output root (``details/S15/README.md`` §22 + the S14 precedent).
RUN_ROOT = "artifacts/S15"

#: Protocol root: read-only for every S15 artifact (hard rule: 不修改实验协议).
PROTOCOL_ROOT = "docs/stage_experiments"

#: S15 protocol directory inside the protocol root (also read-only).
STAGE_PROTOCOL_DIR = "docs/stage_experiments/details/S15"

#: Layout entries that are directories (the rest are files).
LAYOUT_DIRECTORIES: Tuple[str, ...] = rec.LAYOUT_DIRECTORIES

#: Large / vendor / secret artefacts that must never be committed
#: (task hard rule + AGENTS.md §6).  S15 adds the release-artifact types it
#: audits, because a "release bundle" is exactly where a wheel or a model weight
#: would otherwise leak into git.
FORBIDDEN_COMMITTED_SUFFIXES: Tuple[str, ...] = (
    ".ncu-rep",
    ".nsys-rep",
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".onnx",
    ".engine",
    ".plan",
    ".so",
    ".whl",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".iso",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".wav",
)

FORBIDDEN_COMMITTED_NAMES: Tuple[str, ...] = (
    ".env",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "kubeconfig",
    "terraform.tfstate",
    ".npmrc",
    ".pypirc",
)

#: Prohibited actions, each with the protocol clause that forbids it.
PROHIBITED_ACTIONS: Mapping[str, str] = {
    "write_protocol_tree": "任务硬规则: 不得修改 docs/stage_experiments/**",
    "overwrite_raw_data": "任务硬规则 5: 不得删除或改写任何 raw 数据",
    "fabricate_measurement": "任务硬规则 2: 不得伪造任何数字或产物",
    "conclusion_without_execute": "任务第五节: 不执行正式实验、不产出实验结论",
    "publish_release_without_authorization": "E15-05 step 41/42: release 需要人工 GO/NO-GO，不得自动发布",
    "replace_published_asset_in_place": "E15-05 §8 invariant 10: 已发布资产不得原地替换",
    "upstream_submission_without_authorization": "E15-10 step 25: 上游提交需人工授权并遵守当前贡献指南",
    "upstream_pressure_ping": "E15-10 §9.2: 不为获得回复频繁 ping maintainer",
    "resume_claim_without_ledger": "E15-08 step 44: 简历数字只能由 verified claim 填充",
    "reviewer_local_mirror": "E15-09 step 5/6: reviewer 只能接收公开冻结材料，不得用作者本地镜像/缓存",
    "author_operates_reviewer_env": "E15-09 §3.2: L3/L4 帮助使当前 candidate 不可独立复现",
    "demo_external_writes": "E15-07 step 38: Demo 不得创建云资源、发布 release 或提交 issue",
    "destructive_fault_on_shared_resource": "E15-07 step 26/30: 故障注入不得抢占/清理他人进程或共享设备",
    "overwrite_acceptance_history": "任务硬规则 5 / E15-09 step 32: 原失败证据不可覆盖",
    "hide_negative_result": "details/S15/README.md §21.8: 失败、撤回、无收益结果必须保留",
    "declare_certification": "details/S15/README.md §21.9: 不得把第三方标准符合性写成认证",
    "commit_model_weights": "AGENTS.md §6: 不提交模型权重/编译产物/大体积报告/secret",
    "auto_push": "AGENTS.md §6: 不自动 push",
    "rewrite_git_history": "AGENTS.md §6: 不重写 Git 历史",
    "write_machine_local_path_in_config": "AGENTS.md §6: 不把本机绝对路径写入受版本控制的配置",
}

#: Isolation requirements: where each risky operation may run at all.
REQUIRED_ISOLATION: Mapping[str, str] = {
    "artifact_corruption_fixture": "E15-02 step 33 / E15-06 step 38: 只对复制品注入损坏，不改原制品",
    "cache_permission_probe": "E15-02 step 34: 受控目录权限实验，不破坏宿主缓存",
    "fault_injection_rehearsal": "E15-07 §5 step 25–32: 在隔离演示环境注入，不抢占共享资源",
    "reviewer_material_handover": "E15-09 step 5/6: 只通过公开冻结材料交付，隔离作者内部记录",
    "upstream_reproducer_runs": "E15-10 step 12/13: 最小复现脱离 HQSB 私有状态与凭据",
    "privacy_scan_scope": "E15-05 step 31/32: 扫描 history/build layers/bundle，允许受控 canary",
    "demo_recording": "E15-07 step 15/43: 录屏前关闭通知，脱敏 prompt/host/path",
    "public_disclosure_of_issues": "E15-10 step 8: 安全问题走私密渠道，不先公开 PoC",
}

#: Default publication policy: a release is a human decision, never a script.
DEFAULT_RELEASE_POLICY: Mapping[str, Any] = {
    "decision_maker": "human",
    "automated_publish": False,
    "immutable_assets": True,
    "retraction_requires_new_version": True,
    "model_weights_distributed": False,
}


@dataclass
class CampaignBudget:
    """A frozen S15 campaign budget: S15 spends *people's* time, not GPU hours."""

    values: Mapping[str, Any] = field(default_factory=dict)
    stop_rules: Tuple[str, ...] = ()

    #: Budget dimensions S15 must declare (unknown is not zero).
    DIMENSIONS: Tuple[str, ...] = (
        "reviewer_sessions",
        "participants",
        "rehearsals",
        "audit_hours",
        "storage_gb",
        "external_contacts",
    )

    def validate(self) -> List[str]:
        problems: List[str] = []
        for dimension in self.DIMENSIONS:
            if dimension not in self.values:
                problems.append(f"budget is missing dimension {dimension!r} (unknown is not zero)")
                continue
            value = self.values[dimension]
            if value is None:
                problems.append(f"budget dimension {dimension!r} is None; an explicit value or 'unmeasured' is required")
            elif isinstance(value, (int, float)) and value < 0:
                problems.append(f"budget dimension {dimension!r} is negative ({value})")
        if not self.stop_rules:
            problems.append("a budget without stop rules is unbounded (manual §5.1)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "hqsb.s15.budget.v1",
            "values": {key: self.values[key] for key in sorted(self.values)},
            "stop_rules": list(self.stop_rules),
        }


@dataclass
class CampaignManifest:
    """One S15 campaign: which experiments, on what candidate, under which policy."""

    campaign_id: str
    stage: str = "S15"
    experiments: Tuple[str, ...] = ()
    release_candidate_id: str = ""
    claim_ledger_id: str = ""
    public_channels: Tuple[str, ...] = ()
    budget: Optional[CampaignBudget] = None
    prohibited_actions: Tuple[str, ...] = ()
    required_isolation: Mapping[str, str] = field(default_factory=dict)
    release_policy: Mapping[str, Any] = field(default_factory=dict)
    upstream_evidence: Tuple[str, ...] = ()

    schema_version = "hqsb.s15.campaign.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.campaign_id:
            problems.append("CampaignManifest: campaign_id is required")
        if self.stage != "S15":
            problems.append(f"CampaignManifest: stage must be 'S15', got {self.stage!r}")
        if not self.experiments:
            problems.append("CampaignManifest: at least one experiment is required")
        unknown = [item for item in self.experiments if item not in rec.EXPERIMENT_IDS]
        if unknown:
            problems.append(f"CampaignManifest: unknown experiments {', '.join(unknown)}")
        if not self.release_candidate_id:
            problems.append("CampaignManifest: release_candidate_id is required (every result references a candidate)")
        if not self.claim_ledger_id:
            problems.append("CampaignManifest: claim_ledger_id is required (rendering reads the ledger)")
        if not self.public_channels:
            problems.append("CampaignManifest: public_channels must be declared, not left implicit")
        if self.budget is None:
            problems.append("CampaignManifest: a campaign budget is required")
        else:
            problems.extend(self.budget.validate())
        if not self.prohibited_actions:
            problems.append("CampaignManifest: prohibited_actions must be declared, not left implicit")
        else:
            problems.extend(check_execution_safety(self.prohibited_actions, self.required_isolation))
        policy = dict(DEFAULT_RELEASE_POLICY)
        policy.update(self.release_policy)
        if policy.get("automated_publish"):
            problems.append("CampaignManifest: automated publishing is forbidden (E15-05 step 41)")
        if policy.get("model_weights_distributed"):
            problems.append("CampaignManifest: model weights must not be distributed (E15-05 §16)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "stage": self.stage,
            "experiments": list(self.experiments),
            "release_candidate_id": self.release_candidate_id,
            "claim_ledger_id": self.claim_ledger_id,
            "public_channels": list(self.public_channels),
            "budget": self.budget.as_dict() if self.budget else None,
            "prohibited_actions": list(self.prohibited_actions),
            "required_isolation": {key: self.required_isolation[key] for key in sorted(self.required_isolation)},
            "release_policy": {key: self.release_policy[key] for key in sorted(self.release_policy)},
            "upstream_evidence": list(self.upstream_evidence),
        }
        payload["manifest_digest"] = canonical_digest(payload)
        return payload


def default_prohibited_actions() -> Tuple[str, ...]:
    """Every prohibition, so a campaign cannot *forget* one by omission."""
    return tuple(sorted(PROHIBITED_ACTIONS))


# ── layout ─────────────────────────────────────────────────────────────────


def run_layout(experiment_id: str, run_id: str, *, root: str = "") -> Dict[str, str]:
    """Paths of one run directory (the §22 data package plus experiment extras)."""
    if experiment_id not in rec.EXPERIMENT_IDS:
        raise ConfigError(f"unknown S15 experiment {experiment_id!r}")
    assert_safe_segment(run_id, what="run_id")
    base = os.path.join(root, RUN_ROOT, experiment_id, run_id) if root else f"{RUN_ROOT}/{experiment_id}/{run_id}"
    layout: Dict[str, str] = {}
    for entry in rec.RUN_LAYOUT:
        layout[entry] = os.path.join(base, entry) if root else f"{base}/{entry}"
    for extra in rec.EXPERIMENT_LAYOUT_EXTRA.get(experiment_id, ()):
        layout[extra] = os.path.join(base, extra) if root else f"{base}/{extra}"
    layout["_base"] = base
    return layout


def assert_writable(path: str, *, root: str = "") -> None:
    """Refuse any write that would land inside the protocol tree.

    An executable boundary is better than a convention: the protocol files under
    ``docs/stage_experiments/**`` describe what *should* happen and must never be
    rewritten by a run — not even the S15 details directory, and not even to
    "record" a result inside the protocol document.
    """
    absolute = os.path.abspath(os.path.join(root, path) if root else path)
    protocol = os.path.abspath(os.path.join(root, PROTOCOL_ROOT) if root else PROTOCOL_ROOT)
    if absolute == protocol or absolute.startswith(protocol + os.sep):
        raise ConfigError(
            f"refusing to write inside the protocol tree ({PROTOCOL_ROOT}); "
            f"offending path: {path!r} (S15 run output belongs under {RUN_ROOT}/)"
        )


def check_committable(paths: Sequence[str]) -> List[str]:
    """Report paths that must not be committed (weights, wheels, recordings, secrets)."""
    problems: List[str] = []
    for path in paths:
        name = os.path.basename(path)
        if name in FORBIDDEN_COMMITTED_NAMES:
            problems.append(f"{path}: secret/credential file must not be committed")
            continue
        if name.endswith(FORBIDDEN_COMMITTED_SUFFIXES):
            problems.append(f"{path}: large binary / release / recording artefact must not be committed")
    return problems


# ── safety ─────────────────────────────────────────────────────────────────


def check_execution_safety(actions: Sequence[str], required_isolation: Mapping[str, str]) -> List[str]:
    """Check a campaign's declared prohibitions and isolation requirements.

    The protocol forbids a specific set of actions; a campaign that does not
    declare the prohibition is not thereby allowed to perform it.  Unknown action
    names are refused (a typo must not silently drop a prohibition), and the
    prohibitions S15 cannot omit are reported when missing.
    """
    problems: List[str] = []
    for action in actions:
        if action not in PROHIBITED_ACTIONS:
            problems.append(f"unknown prohibited action {action!r}; see PROHIBITED_ACTIONS")
    for required in (
        "conclusion_without_execute",
        "publish_release_without_authorization",
        "upstream_submission_without_authorization",
        "reviewer_local_mirror",
        "fabricate_measurement",
        "hide_negative_result",
    ):
        if required not in actions:
            problems.append(f"campaign must explicitly prohibit {required!r} ({PROHIBITED_ACTIONS[required]})")
    for operation, clause in required_isolation.items():
        if operation not in REQUIRED_ISOLATION:
            problems.append(f"unknown isolation-sensitive operation {operation!r}")
        if not clause or len(clause) < 10:
            problems.append(f"isolation for {operation!r} must cite its protocol clause, got {clause!r}")
    missing = [name for name in REQUIRED_ISOLATION if name not in required_isolation]
    if missing:
        problems.append(
            "campaign must record the isolation required for every risky operation; missing: "
            + ", ".join(sorted(missing))
        )
    return problems


def isolation_clause(operation: str) -> str:
    if operation not in REQUIRED_ISOLATION:
        raise ConfigError(f"unknown isolation-sensitive operation {operation!r}")
    return REQUIRED_ISOLATION[operation]


def release_policy(**overrides: Any) -> Dict[str, Any]:
    """The default release policy with explicit overrides."""
    policy = dict(DEFAULT_RELEASE_POLICY)
    policy.update(overrides)
    return {key: policy[key] for key in sorted(policy)}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the campaign layer (labelled smoke, not an experiment)."""
    layout = run_layout("E15-01", "interface_only")
    problems: List[str] = []
    for entry in ("raw", "derived", "acceptance.json", "evidence_manifest.yaml"):
        if entry not in layout:
            problems.append(f"§22 layout entry {entry!r} is missing")
    try:
        assert_writable(layout["raw"])
    except ConfigError as exc:  # pragma: no cover - the run root must be writable
        problems.append(str(exc))
    protocol_refused = False
    try:
        assert_writable(f"{STAGE_PROTOCOL_DIR}/README.md")
    except ConfigError:
        protocol_refused = True
    if not protocol_refused:
        problems.append("writes into the protocol tree were not refused")
    budget = CampaignBudget(
        values={dimension: 0 for dimension in CampaignBudget.DIMENSIONS},
        stop_rules=("quality gate fail", "claim ledger unresolved"),
    )
    safety = check_execution_safety(default_prohibited_actions(), dict(REQUIRED_ISOLATION))
    return {
        "status": "smoke",
        "claim_allowed": False,
        "layout_entries": len(rec.RUN_LAYOUT) + len(rec.EXPERIMENT_LAYOUT_EXTRA.get("E15-01", ())),
        "protocol_tree_write_refused": protocol_refused,
        "budget_problems": budget.validate(),
        "safety_problems": safety,
        "prohibited_actions": len(PROHIBITED_ACTIONS),
        "isolation_operations": len(REQUIRED_ISOLATION),
        "problems": problems,
    }
