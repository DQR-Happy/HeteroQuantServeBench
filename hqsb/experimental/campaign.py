"""S14 campaign layout and the execution-safety policy.

Two jobs:

* **layout** — S14 has two distinct directory families and mixing them is how a
  protocol file gets overwritten by a result:
  ``docs/stage_experiments/S14/**`` is the *protocol* (read-only, and owned by the
  documents, not by this package) while
  ``artifacts/S14/<experiment_id>/<run_id>/**`` is the *run output*
  (``details/S14/README.md`` §22).  :func:`run_layout` and :func:`assert_writable`
  make that boundary executable;
* **safety** — S14 is the first stage whose experiments can touch real trainers,
  real rollout loops, real devices and real external tools, so the protocol
  spends several sections on what a run may *not* do (``E14-01`` step 31,
  ``E14-02`` step 37, ``E14-03`` step 33, ``E14-07`` §2, ``E14-08`` §7 step 8).
  Those prohibitions are data here, checked by :func:`check_execution_safety`,
  so an unsafe run is refused before it starts rather than reviewed afterwards.

Nothing in this module starts anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.identity import SCHEMA_PREFIX, canonical_digest

#: Run output root (``details/S14/README.md`` §22).
RUN_ROOT = "artifacts/S14"

#: Protocol root: read-only for every S14 artifact (hard rule: 不修改实验协议).
PROTOCOL_ROOT = "docs/stage_experiments"

#: Fixed run-directory layout of §22 (files and directories together).
RUN_LAYOUT: Tuple[str, ...] = rec.RUN_DIRECTORY_LAYOUT

#: Layout entries that are directories (the rest are files).
LAYOUT_DIRECTORIES: Tuple[str, ...] = (
    "dependency_lock",
    "model_and_checkpoint",
    "configs",
    "raw",
    "profiles",
    "traces",
    "quality",
    "failures",
    "normalized",
    "plots",
    "decisions",
)

#: Large binary / vendor / secret artefacts that must never be committed
#: (task hard rule + AGENTS.md §6).
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
    ".tflite",
    ".mnn",
    ".nb",
    ".qnn",
)

FORBIDDEN_COMMITTED_NAMES: Tuple[str, ...] = (
    ".env",
    "credentials.json",
    "id_rsa",
    "kubeconfig",
    "terraform.tfstate",
)

#: Resource dimensions a campaign must budget before a frontier run
#: (``E14-05`` step 26, ``E14-04`` step 11).
BUDGET_DIMENSIONS: Tuple[str, ...] = (
    "device_count",
    "gpu_hours",
    "generated_tokens",
    "training_steps",
    "rollout_episodes",
    "wall_clock_hours",
    "storage_gb",
    "cost_units",
)

#: Prohibited actions, each with the protocol clause that forbids it.
PROHIBITED_ACTIONS: Mapping[str, str] = {
    "destroy_shared_dependencies": "E14-01 step 31: 禁止在主环境故意破坏共享依赖",
    "mutate_origin_artifact": "E14-03 step 33 / F4 step 34: 只能在复制品上注入损坏",
    "destructive_fault_on_shared_device": "E14-02 step 37 / F2 step 34: 故障注入必须在隔离环境",
    "unbounded_generation": "E14-04 step 11: 必须限制 generated tokens 与失败率",
    "external_tool_side_effect": "E14-07 §2: 不向真实第三方发送消息、不执行不可逆动作",
    "arbitrary_shell_tool": "E14-07 step 3: 不提供任意 shell/邮件/生产 API 工具",
    "firmware_modification": "E14-08 step 8: 不修改不可逆 firmware",
    "personal_device_profiling": "E14-08 step 8: 不为 profile 破坏个人/生产设备",
    "overwrite_raw_data": "任务硬规则 5: 不得删除或改写任何 raw 数据",
    "write_protocol_tree": "任务硬规则: 不得修改 docs/stage_experiments/**",
    "commit_model_weights": "AGENTS.md §6: 不提交模型权重/编译产物/大体积报告/secret",
    "auto_push": "AGENTS.md §6: 不自动 push",
    "fabricate_measurement": "任务硬规则 2: 不得伪造任何数字或产物",
    "conclusion_without_execute": "任务第五节: 不执行正式实验、不产出实验结论",
}

#: Isolation requirements: where each risky operation may run at all.
REQUIRED_ISOLATION: Mapping[str, str] = {
    "dependency_uninstall_probe": "E14-01 step 38: 只在允许卸载的隔离环境",
    "abi_mismatch_probe": "E14-01 step 31: 使用隔离环境安装预注册的不兼容组合",
    "corrupt_checkpoint_fixture": "E14-02 step 36: 复制 checkpoint 后篡改副本",
    "rank_failure_injection": "E14-02 step 37: 隔离环境 + 可控 hook",
    "oom_boundary_probe": "E14-03 step 39 / F3 step 34: 隔离环境并检查恢复",
    "mixed_version_negative": "E14-04 step 29: 受控 fixture，不污染真实 batch",
    "stale_policy_negative": "E14-04 step 30: 通过 publish/consume 速度控制，不篡改权重",
    "low_acceptance_negative": "E14-F1 step 33: 受控 domain shift/旧版 draft",
    "sparse_corruption_fixture": "E14-F4 step 34: 复制的测试 artifact",
    "device_low_memory_probe": "E14-08 step 35: 受控压力，检查释放与后续请求",
    "external_tool_timeout": "E14-07 step 27: sandbox/proxy 注入",
}


@dataclass
class RunBudget:
    """A frozen resource/stop budget (``E14-05`` step 29, ``E14-04`` step 11)."""

    values: Mapping[str, Any] = field(default_factory=dict)
    stop_rules: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for dimension in BUDGET_DIMENSIONS:
            if dimension not in self.values:
                problems.append(f"budget is missing dimension {dimension!r} (unknown is not zero)")
                continue
            value = self.values[dimension]
            if value is None:
                problems.append(f"budget dimension {dimension!r} is None; an explicit value or 'unmeasured' is required")
            elif isinstance(value, (int, float)) and value < 0:
                problems.append(f"budget dimension {dimension!r} is negative ({value})")
        if not self.stop_rules:
            problems.append("a budget without stop rules is unbounded (E14-05 step 31)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.budget.v1",
            "values": {key: self.values[key] for key in sorted(self.values)},
            "stop_rules": list(self.stop_rules),
        }


@dataclass
class CampaignManifest:
    """One S14 campaign: which experiments, on what, under which policy."""

    campaign_id: str
    stage: str = "S14"
    experiments: Tuple[str, ...] = ()
    hardware_and_power_mode: str = ""
    environment_uri: str = ""
    budget: Optional[RunBudget] = None
    prohibited_actions: Tuple[str, ...] = ()
    required_isolation: Mapping[str, str] = field(default_factory=dict)
    registry_credentials_included: bool = False
    upstream_evidence: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.campaign.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.campaign_id:
            problems.append("CampaignManifest: campaign_id is required")
        if self.stage != "S14":
            problems.append(f"CampaignManifest: stage must be 'S14', got {self.stage!r}")
        if not self.experiments:
            problems.append("CampaignManifest: at least one experiment is required")
        unknown = [item for item in self.experiments if item not in {row[0] for row in rec.EXPERIMENT_TABLE}]
        if unknown:
            problems.append(f"CampaignManifest: unknown experiments {', '.join(unknown)}")
        if not self.hardware_and_power_mode:
            problems.append(
                "CampaignManifest: hardware_and_power_mode is required "
                "(结果必须在能定位到机器/功耗模式时才可比较)"
            )
        if self.budget is None:
            problems.append("CampaignManifest: a resource budget is required")
        else:
            problems.extend(self.budget.validate())
        if not self.prohibited_actions:
            problems.append("CampaignManifest: prohibited_actions must be declared, not left implicit")
        else:
            problems.extend(check_execution_safety(self.prohibited_actions, self.required_isolation))
        if self.registry_credentials_included:
            problems.append(
                "CampaignManifest: registry/cluster credentials must not be part of the campaign "
                "(AGENTS.md §6: 不提交 secret)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "stage": self.stage,
            "experiments": list(self.experiments),
            "hardware_and_power_mode": self.hardware_and_power_mode,
            "environment_uri": self.environment_uri,
            "budget": self.budget.as_dict() if self.budget else None,
            "prohibited_actions": list(self.prohibited_actions),
            "required_isolation": {key: self.required_isolation[key] for key in sorted(self.required_isolation)},
            "registry_credentials_included": self.registry_credentials_included,
            "upstream_evidence": list(self.upstream_evidence),
        }
        payload["manifest_digest"] = canonical_digest(payload)
        return payload


def default_prohibited_actions() -> Tuple[str, ...]:
    """Every prohibition, so a campaign cannot *forget* one by omission."""
    return tuple(sorted(PROHIBITED_ACTIONS))


# ── layout ─────────────────────────────────────────────────────────────────


def run_layout(experiment_id: str, run_id: str, *, root: str = "") -> Dict[str, str]:
    """Absolute-or-root-relative paths of one run directory (``§22`` layout)."""
    if experiment_id not in {row[0] for row in rec.EXPERIMENT_TABLE}:
        raise ConfigError(f"unknown S14 experiment {experiment_id!r}")
    if not run_id or "/" in run_id or run_id in (".", ".."):
        raise ConfigError(f"run_id must be a single safe path segment, got {run_id!r}")
    base = os.path.join(root, RUN_ROOT, experiment_id, run_id) if root else f"{RUN_ROOT}/{experiment_id}/{run_id}"
    layout: Dict[str, str] = {}
    for entry in RUN_LAYOUT:
        layout[entry] = os.path.join(base, entry) if root else f"{base}/{entry}"
    layout["_base"] = base
    return layout


def assert_writable(path: str, *, root: str = "") -> None:
    """Refuse any write that would land inside the protocol tree.

    An executable boundary is better than a convention: the protocol files under
    ``docs/stage_experiments/**`` describe what *should* happen and must never be
    rewritten by a run, even accidentally through a relative path.
    """
    absolute = os.path.abspath(os.path.join(root, path) if root else path)
    protocol = os.path.abspath(os.path.join(root, PROTOCOL_ROOT) if root else PROTOCOL_ROOT)
    if absolute == protocol or absolute.startswith(protocol + os.sep):
        raise ConfigError(
            f"refusing to write inside the protocol tree ({PROTOCOL_ROOT}); "
            f"offending path: {path!r} (S14 run output belongs under {RUN_ROOT}/)"
        )


def check_committable(paths: Sequence[str]) -> List[str]:
    """Report paths that must not be committed (weights, engines, secrets)."""
    problems: List[str] = []
    for path in paths:
        name = os.path.basename(path)
        if name in FORBIDDEN_COMMITTED_NAMES:
            problems.append(f"{path}: secret/credential file must not be committed")
            continue
        if name.endswith(FORBIDDEN_COMMITTED_SUFFIXES):
            problems.append(f"{path}: large binary / compiled artifact must not be committed")
    return problems


# ── safety ─────────────────────────────────────────────────────────────────


def check_execution_safety(
    actions: Sequence[str], required_isolation: Mapping[str, str]
) -> List[str]:
    """Check a campaign's declared prohibitions and isolation requirements.

    The protocol forbids a specific set of actions; a campaign that does not
    declare the prohibition is not thereby allowed to perform it. Unknown
    action names are refused (a typo must not silently drop a prohibition).
    """
    problems: List[str] = []
    for action in actions:
        if action not in PROHIBITED_ACTIONS:
            problems.append(f"unknown prohibited action {action!r}; see PROHIBITED_ACTIONS")
    if "destructive_fault_on_shared_device" not in actions:
        problems.append("campaign must explicitly prohibit destructive faults on shared devices")
    if "external_tool_side_effect" not in actions:
        problems.append("campaign must explicitly prohibit external tool side effects (E14-07 §2)")
    if "conclusion_without_execute" not in actions:
        problems.append("campaign must explicitly prohibit conclusions without --execute")
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


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the campaign layer (labelled smoke, not an experiment)."""
    layout = run_layout("E14-01", "interface_only")
    problems: List[str] = []
    try:
        assert_writable(layout["raw"])
    except ConfigError as exc:  # pragma: no cover - the layout must be writable
        problems.append(str(exc))
    protocol_refused = False
    try:
        assert_writable(f"{PROTOCOL_ROOT}/S14_实验清单.md")
    except ConfigError:
        protocol_refused = True
    budget = RunBudget(
        values={dimension: 0 for dimension in BUDGET_DIMENSIONS}, stop_rules=("quality gate fail",)
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "layout_entries": len(RUN_LAYOUT),
        "protocol_tree_write_refused": protocol_refused,
        "budget_problems": budget.validate(),
        "prohibited_actions": len(PROHIBITED_ACTIONS),
        "isolation_operations": len(REQUIRED_ISOLATION),
        "problems": problems,
    }
