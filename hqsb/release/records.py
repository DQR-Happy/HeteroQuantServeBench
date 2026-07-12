"""S15 vocabularies, record schemas and the run-directory contract.

This module is the frozen vocabulary layer of the release / open-source /
job-evidence stage.  It encodes, as data:

* the eleven experiments of ``docs/stage_experiments/S15_实验清单.md`` with their
  levels and titles (``EXPERIMENT_TABLE``);
* the six S15 statuses of ``details/S15/README.md`` §13.5 (``NOT_RUN``, ``PASS``,
  ``FAIL``, ``BLOCKED``, ``INVALID``, ``N/A_BY_SCOPE``) — ``BLOCKED`` may never be
  rewritten as ``0``/``PASS``;
* the ten-or-so secondary state machines the protocol keeps separate on purpose:
  claim lifecycle (§8.2), supply-chain finding disposition (``E15-05`` §9.1),
  reviewer help levels and reproduction levels (``E15-09`` §9), demo measurement
  states (``E15-07`` §3.2), documentation command verdicts (``E15-04`` §9.1),
  figure rebuild classes (``E15-06`` §9.1), upstream dispositions
  (``E15-10`` §9.1), and the finding severity ladder of ``E15-01`` §10;
* the **minimum record fields** of every experiment (one dataclass each, exactly
  the fields the protocol lists) together with a ``validate()`` that refuses a
  conclusion without evidence;
* the §22 *uniform run data package* and the §12 wave order, so the driver can
  create a complete, empty, auditable run directory before anything runs.

Nothing in this module executes an experiment or produces a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release.identity import (
    CHANNEL_MINIMUM_LEVEL,
    CHANNEL_MIRRORS_CLAIM,
    CLAIM_STATES,
    EVIDENCE_LEVELS,
    PUBLIC_CHANNELS,
    SCHEMA_PREFIX,
    is_digest,
)

STAGE = "S15"

#: (experiment id, level, title) — ``S15_实验清单.md`` §必做实验.
EXPERIMENT_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("E15-01", "P0", "全局 Claim Ledger、声明扫描与证据门禁"),
    ("E15-02", "P0", "Clean CPU Quickstart 与最低门槛复现"),
    ("E15-03", "P0", "目标 GPU/NPU Hero Story 独立重放"),
    ("E15-04", "P0", "可执行文档、双语一致性与能力矩阵对账"),
    ("E15-05", "P0", "Tag→Release 制品、Provenance、SBOM 与许可证"),
    ("E15-06", "P0", "Dashboard/图表→Raw 数据反向追溯与重生成"),
    ("E15-07", "P0", "3–5 分钟 Demo、故障注入与诚实降级"),
    ("E15-08", "P0", "3/10/30 分钟讲述与对抗技术问答"),
    ("E15-09", "P0", "第三方 Clean-room Reproduction 与修复复测"),
    ("E15-10", "P0", "真实上游 Issue/PR/文档贡献"),
    ("E15-11", "P1", "目标岗位读者 5 分钟首屏理解实验"),
)

EXPERIMENT_IDS: Tuple[str, ...] = tuple(experiment_id for experiment_id, _level, _title in EXPERIMENT_TABLE)

#: Step count of every S15 protocol file (facets verified against the documents).
STEPS_PER_EXPERIMENT = 45

# ── primary statuses (§13.5) ────────────────────────────────────────────────

STATUS_NOT_RUN = "NOT_RUN"
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_INVALID = "INVALID"
STATUS_NA_BY_SCOPE = "N/A_BY_SCOPE"

ALL_STATUSES: Tuple[str, ...] = (
    STATUS_NOT_RUN,
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_INVALID,
    STATUS_NA_BY_SCOPE,
)

#: Statuses that are *conclusions* and may only be written through the triple gate.
CONCLUSION_STATUSES: Tuple[str, ...] = (STATUS_PASS, STATUS_FAIL)

#: Statuses that may be written without a conclusion.
NON_CONCLUSION_STATUSES: Tuple[str, ...] = (STATUS_NOT_RUN, STATUS_BLOCKED, STATUS_INVALID, STATUS_NA_BY_SCOPE)

# ── secondary state machines ─────────────────────────────────────────────────

#: Supply-chain finding disposition (``E15-05`` §9.1).
FINDING_DISPOSITIONS: Tuple[str, ...] = (
    "OPEN",
    "FIXED_IN_NEW_CANDIDATE",
    "NOT_AFFECTED_WITH_EVIDENCE",
    "ACCEPTED_RISK_WITH_OWNER_AND_EXPIRY",
    "ASSET_WITHHELD",
    "RELEASE_BLOCKED",
)

#: Severity ladder (``E15-01`` §10).
SEVERITIES: Tuple[str, ...] = ("P0", "P1", "P2", "INFO")

#: Reviewer help levels (``E15-09`` §3.2).  L3/L4 fail the current candidate.
HELP_LEVELS: Tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")

#: Reproduction levels R0–R5 (``E15-09`` §3.3, §9).
REPRODUCTION_LEVELS: Tuple[str, ...] = ("R0", "R1", "R2", "R3", "R4", "R5")

#: Environment comparability classes (``E15-03`` step 5, ``E15-09`` §3.4).
COMPARABILITY_CLASSES: Tuple[str, ...] = ("exact", "compatible", "conditional", "incomparable")

#: Demo measurement states (``E15-07`` §3.2).
DEMO_MEASUREMENT_STATES: Tuple[str, ...] = (
    "LIVE_MEASUREMENT",
    "LIVE_REGENERATION",
    "CACHED_VERIFIED_RESULT",
    "PRERECORDED_VERIFIED_RUN",
    "SIMULATED_UI",
)

#: Demo fault scenarios (``E15-07`` §9 matrix left column).
DEMO_FAULT_SCENARIOS: Tuple[str, ...] = (
    "normal",
    "no_device",
    "busy_device",
    "cache_miss",
    "network_down",
    "model_missing",
    "timeout",
    "bad_asset",
    "web_unavailable",
    "evidence_link_down",
    "correctness_fail",
)

#: Documentation code-block execution verdicts (``E15-04`` §9.1).
COMMAND_VERDICTS: Tuple[str, ...] = (
    "PASS",
    "FAIL",
    "BLOCKED_DEVICE",
    "DRY_RUN_PASS",
    "MANUAL_VERIFIED",
    "DISPLAY_ONLY_VALID",
    "INVALID_UNCLASSIFIED",
)

#: Code-block execution classes (``E15-04`` §3.1).
CODE_BLOCK_CLASSES: Tuple[str, ...] = (
    "EXECUTE_SAFE",
    "EXECUTE_ACCELERATOR",
    "DRY_RUN",
    "MANUAL_EXTERNAL",
    "DISPLAY_ONLY",
)

#: Figure rebuild difference levels (``E15-06`` §9.1); D3–D5 fail.
REBUILD_CLASSES: Tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4", "D5")

#: Upstream contribution dispositions (``E15-10`` §9.1).
UPSTREAM_STATUSES: Tuple[str, ...] = (
    "OPEN",
    "UNDER_REVIEW",
    "ACCEPTED",
    "MERGED",
    "RELEASED",
    "DUPLICATE",
    "REJECTED",
    "WONTFIX",
    "STALLED",
)

#: Boundary triage of an upstream candidate (``E15-10`` step 4/6).
BOUNDARY_TRIAGE: Tuple[str, ...] = ("upstream", "hqsb", "usage", "known_limitation")

#: Hero replay verdicts (``E15-03`` §10 consistency matrix).
REPLAY_VERDICTS: Tuple[str, ...] = (
    "CONFIRMED",
    "CONDITIONAL_CONFIRMED",
    "INCONCLUSIVE",
    "CONTRADICTED_OR_SCOPE_CHANGE",
    "CONTRADICTED_UNATTRIBUTED",
    "INVALID_PATH",
    "QUALITY_DISQUALIFIED",
    "MECHANISM_NOT_REPRODUCED",
    "BLOCKED",
)

#: Claim qualification gates (``E15-01`` §10).
CLAIM_GATES: Tuple[str, ...] = (
    "Identity",
    "Semantics",
    "Correctness",
    "Measurement",
    "ActualPath",
    "Evidence",
    "Integrity",
    "Freshness",
    "Attribution",
    "Rendering",
)

#: Failure codes a claim gate produces (``E15-01`` §10 third column).
CLAIM_GATE_FAILURES: Tuple[str, ...] = (
    "IDENTITY_INCOMPLETE",
    "SEMANTIC_AMBIGUOUS",
    "QUALITY_DISQUALIFIED",
    "MEASUREMENT_INVALID",
    "FALLBACK_MIXED",
    "ORPHAN",
    "CORRUPTED",
    "STALE",
    "ATTRIBUTION_ERROR",
    "CHANNEL_CONFLICT",
)

#: Claim types (``E15-01`` step 3 + §8.1).
CLAIM_TYPES: Tuple[str, ...] = (
    "capability",
    "correctness",
    "performance",
    "quality",
    "memory",
    "energy_or_cost",
    "reliability",
    "portability",
    "attribution",
)

#: Typed edges of the claim→evidence DAG (``E15-01`` step 21, PROV-style).
EVIDENCE_EDGE_TYPES: Tuple[str, ...] = (
    "wasDerivedFrom",
    "wasGeneratedBy",
    "used",
    "wasAssociatedWith",
    "wasAttributedTo",
    "invalidatedBy",
)

#: MISSION factors of ``details/S15/README.md`` §1 (credibility is a product).
CREDIBILITY_FACTORS: Tuple[str, ...] = (
    "claim_truthfulness",
    "artifact_identity_and_provenance",
    "executable_documentation",
    "clean_environment_repeatability",
    "independent_reproducibility",
    "figure_to_raw_lineage",
    "release_security_and_licensing",
    "failure_transparent_demonstration",
    "technically_defensible_communication",
    "public_collaboration_evidence",
)

#: G0–G8 hierarchical gates of ``details/S15/README.md`` §11.
STAGE_GATES: Tuple[Tuple[str, str], ...] = (
    ("G0", "upstream freeze: S00–S14 acceptance/limitations 可定位"),
    ("G1", "claim truth: 无 orphan/stale/overclaim"),
    ("G2", "executable entry: clean CPU quickstart + docs checks"),
    ("G3", "technical reproduction: accelerator hero replay + plot regeneration"),
    ("G4", "release integrity: tag/artifact/digest/provenance/SBOM/license/security"),
    ("G5", "communication robustness: demo + narratives + evidence drill-down"),
    ("G6", "independent validation: clean-room reviewer + fix/retest"),
    ("G7", "public collaboration: high-quality upstream contribution"),
    ("G8", "audience comprehension: 5-minute target-reader study"),
)

#: Wave order of ``details/S15/README.md`` §12.
EXECUTION_WAVES: Mapping[str, Tuple[str, ...]] = {
    "A": ("E15-01",),
    "B": ("E15-02", "E15-04", "E15-06"),
    "C": ("E15-03",),
    "D": ("E15-05",),
    "E": ("E15-07", "E15-08"),
    "F": ("E15-09",),
    "G": ("E15-10",),
    "H": ("E15-11",),
}

#: Upstream experiments each S15 experiment consumes (§4 dependency graph).
EXPERIMENT_DEPENDENCIES: Mapping[str, Tuple[str, ...]] = {
    "E15-01": (),
    "E15-02": ("E15-01",),
    "E15-03": ("E15-01", "E15-02", "E15-04"),
    "E15-04": ("E15-01",),
    "E15-05": ("E15-01", "E15-02", "E15-03", "E15-04"),
    "E15-06": ("E15-01", "E15-03"),
    "E15-07": ("E15-01", "E15-03", "E15-04", "E15-05", "E15-06"),
    "E15-08": ("E15-01", "E15-03", "E15-05", "E15-06", "E15-07"),
    "E15-09": ("E15-01", "E15-02", "E15-03", "E15-04", "E15-05", "E15-06"),
    "E15-10": ("E15-01", "E15-03", "E15-04", "E15-05", "E15-09"),
    "E15-11": ("E15-01", "E15-04", "E15-07", "E15-08", "E15-09"),
}

#: Every S15 experiment consumes the frozen S00–S14 inputs (G0).
UPSTREAM_STAGES: Tuple[str, ...] = tuple(f"S{index:02d}" for index in range(15))

# ── §22 uniform run data package ─────────────────────────────────────────────

#: The minimum data package every E15 experiment must generate (§22).
RUN_LAYOUT: Tuple[str, ...] = (
    "protocol.yaml",
    "release_candidate_ref.json",
    "environment.json",
    "participants_or_agents.json",
    "command_or_action_log.jsonl",
    "raw",
    "derived",
    "findings.json",
    "limitations.md",
    "acceptance.json",
    "evidence_manifest.yaml",
)

#: Entries of ``RUN_LAYOUT`` that are directories (the rest are files).
LAYOUT_DIRECTORIES: Tuple[str, ...] = ("raw", "derived")

#: Experiment-specific top-level entries from each protocol file's §强制数据产出.
EXPERIMENT_LAYOUT_EXTRA: Mapping[str, Tuple[str, ...]] = {
    "E15-01": ("channel_renderings",),
    "E15-02": ("sessions", "negative_cases"),
    "E15-03": ("original_evidence_snapshot", "correctness", "actual_path", "profiles", "fault_injection"),
    "E15-04": ("build", "command_results", "schema_results"),
    "E15-05": ("build_definition", "artifacts", "provenance", "attestations", "sbom", "consumer_verification"),
    "E15-06": (
        "figure_specs",
        "point_sidecars",
        "lineage_traces",
        "queries",
        "rebuilt",
        "numeric_diffs",
        "visual_diffs",
        "negative_controls",
    ),
    "E15-07": ("rehearsals", "fallback_assets"),
    "E15-08": ("fact_cards", "narratives", "emphasis", "sessions"),
    "E15-09": ("sessions", "issues", "original_failure_snapshot", "retest"),
    "E15-10": ("minimal_reproducer", "negative_controls", "patch", "tests", "benchmark"),
    "E15-11": ("sessions", "coding", "retest"),
}

#: The manual §4 uniform record fields, pre-filled at ``NOT_STARTED``.
RUN_MANIFEST_FIELDS: Tuple[str, ...] = (
    "stage",
    "experiment_id",
    "status",
    "question",
    "hypothesis",
    "run_id",
    "git_commit",
    "git_dirty",
    "model_manifest_sha256",
    "config_sha256",
    "operator_or_binary_sha256",
    "environment_uri",
    "hardware_and_power_mode",
    "requested_implementation",
    "actual_implementation",
    "controls",
    "independent_variables",
    "correctness_metrics",
    "performance_samples_uri",
    "profile_artifacts_uri",
    "started_at",
    "ended_at",
    "decision",
    "limitations",
)


def experiment_record_template(experiment_id: str) -> Dict[str, Any]:
    """The manual §4 record, at ``NOT_STARTED``, with S15 identity fields added."""
    if experiment_id not in EXPERIMENT_IDS:
        raise ConfigError(f"unknown S15 experiment {experiment_id!r}")
    template: Dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.experiment-record.v1",
        "stage": STAGE,
        "experiment_id": experiment_id,
        "status": STATUS_NOT_RUN,
        "question": "",
        "hypothesis": "",
        "run_id": "",
        "git_commit": "",
        "git_dirty": None,
        "model_manifest_sha256": "",
        "config_sha256": "",
        "operator_or_binary_sha256": "",
        "environment_uri": "",
        "hardware_and_power_mode": "",
        "requested_implementation": "",
        "actual_implementation": "",
        "controls": {},
        "independent_variables": {},
        "correctness_metrics": {},
        "performance_samples_uri": "",
        "profile_artifacts_uri": "",
        "started_at": "",
        "ended_at": "",
        "decision": "",
        "limitations": [],
        "release_candidate_id": "",
        "claim_ids": [],
        "public_channels": [],
    }
    unknown = [name for name in RUN_MANIFEST_FIELDS if name not in template]
    if unknown:  # pragma: no cover - a template edit that drops a field is a bug
        raise ConfigError(f"experiment record template is missing {', '.join(unknown)}")
    return template


def validate_state_machines() -> List[str]:
    """Cross-check the vocabularies against each other (no contradictions).

    Cheap structural checks that would otherwise only surface mid-experiment:
    every claim state is reachable, every channel has a minimum level or mirrors
    the claim, the conclusion statuses are a subset of all statuses, and the
    reproduction levels are contiguous from ``R0``.
    """
    problems: List[str] = []
    if set(CONCLUSION_STATUSES) - set(ALL_STATUSES):
        problems.append("conclusion statuses must be a subset of ALL_STATUSES")
    if set(NON_CONCLUSION_STATUSES) | set(CONCLUSION_STATUSES) != set(ALL_STATUSES):
        problems.append("NON_CONCLUSION_STATUSES + CONCLUSION_STATUSES must partition ALL_STATUSES")
    for state in ("DRAFT", "VERIFIED", "STALE", "REJECTED", "RETRACTED"):
        if state not in CLAIM_STATES:
            problems.append(f"claim state {state} is missing")
    for channel in PUBLIC_CHANNELS:
        if channel in CHANNEL_MIRRORS_CLAIM:
            continue
        if channel not in CHANNEL_MINIMUM_LEVEL:
            problems.append(f"channel {channel} has no minimum evidence level")
        elif CHANNEL_MINIMUM_LEVEL[channel] not in EVIDENCE_LEVELS:
            problems.append(f"channel {channel} minimum level is not an evidence level")
    expected_levels = tuple(f"R{index}" for index in range(6))
    if REPRODUCTION_LEVELS != expected_levels:
        problems.append("reproduction levels must be R0..R5 in order")
    for experiment_id, upstream in EXPERIMENT_DEPENDENCIES.items():
        if experiment_id not in EXPERIMENT_IDS:
            problems.append(f"dependency table lists unknown experiment {experiment_id}")
        unknown = [name for name in upstream if name not in EXPERIMENT_IDS]
        if unknown:
            problems.append(f"experiment {experiment_id} depends on unknown {', '.join(unknown)}")
    for wave, members in EXECUTION_WAVES.items():
        unknown = [name for name in members if name not in EXPERIMENT_IDS]
        if unknown:
            problems.append(f"wave {wave} lists unknown experiment {', '.join(unknown)}")
    covered = sorted(name for members in EXECUTION_WAVES.values() for name in members)
    if covered != sorted(EXPERIMENT_IDS):
        problems.append("execution waves must cover every experiment exactly once")
    return problems


# ── per-experiment minimum records ───────────────────────────────────────────
#
# One dataclass per protocol file, carrying exactly the "最小字段" table of that
# file.  ``validate()`` encodes the standing rule of the manual: a conclusion
# without evidence is refused (拿不出证据必须降级).


def require_conclusion_evidence(status: str, raw_refs: Sequence[Any]) -> List[str]:
    """The standing rule: a conclusion without evidence must be downgraded."""
    problems: List[str] = []
    if status in CONCLUSION_STATUSES and not raw_refs:
        problems.append(
            f"status {status} is a conclusion and requires at least one raw/evidence reference "
            "(拿不出证据必须降级)"
        )
    return problems


@dataclass
class ClaimAuditResult:
    """``E15-01`` §8 ``ClaimAuditResult`` 最小字段."""

    audit_id: str
    candidate_id: str = ""
    scanner_version: str = ""
    surface_count: int = 0
    claim_candidate_count: int = 0
    claim_count: int = 0
    status_counts: Mapping[str, int] = field(default_factory=dict)
    orphan_count: int = 0
    stale_count: int = 0
    conflict_count: int = 0
    overclaim_count: int = 0
    numeric_claims_with_raw: int = 0
    negative_control_metrics: Mapping[str, Any] = field(default_factory=dict)
    ledger_sha256: str = ""
    evidence_graph_root: str = ""
    blocking_findings: Tuple[str, ...] = ()
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-01.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.audit_id:
            problems.append("ClaimAuditResult: audit_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"ClaimAuditResult: unknown status {self.status!r}")
        for name in ("surface_count", "claim_candidate_count", "claim_count", "orphan_count", "stale_count"):
            if getattr(self, name) < 0:
                problems.append(f"ClaimAuditResult: {name} may not be negative")
        if self.status == STATUS_PASS:
            if self.orphan_count or self.stale_count or self.conflict_count or self.overclaim_count:
                problems.append("ClaimAuditResult: PASS with orphan/stale/conflict/overclaim counts is contradictory")
            if not self.ledger_sha256 or not self.evidence_graph_root:
                problems.append("ClaimAuditResult: PASS requires a frozen ledger and evidence-graph digest")
            if not self.negative_control_metrics:
                problems.append(
                    "ClaimAuditResult: PASS requires negative-control metrics (a scanner with no detection "
                    "power cannot pass — E15-01 step 35/36)"
                )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "schema_version"},
        }
        payload["status_counts"] = {key: int(value) for key, value in sorted(self.status_counts.items())}
        payload["negative_control_metrics"] = dict(sorted(self.negative_control_metrics.items()))
        payload["blocking_findings"] = list(self.blocking_findings)
        return payload


@dataclass
class QuickstartSessionResult:
    """``E15-02`` §7 ``QuickstartSessionResult`` 最小字段."""

    session_id: str
    candidate_id: str = ""
    operator_id: str = ""
    environment_id: str = ""
    install_mode: str = ""
    cache_state: str = ""
    network_state: str = ""
    started_at_utc: str = ""
    first_verified_report_at_utc: Optional[str] = None
    stage_durations_s: Mapping[str, float] = field(default_factory=dict)
    download_bytes: int = 0
    peak_rss_bytes: int = 0
    disk_delta_bytes: int = 0
    undocumented_interventions: Tuple[str, ...] = ()
    selected_backend: str = ""
    output_semantic_hash: str = ""
    negative_case_results: Tuple[Mapping[str, Any], ...] = ()
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-02.v1"

    #: Stages the total clock must be decomposed into (``E15-02`` §3.2).
    TIMING_STAGES: Tuple[str, ...] = (
        "read",
        "provision",
        "download",
        "install",
        "execute",
        "validate",
        "regenerate",
        "debug",
    )

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.session_id:
            problems.append("QuickstartSessionResult: session_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"QuickstartSessionResult: unknown status {self.status!r}")
        if self.install_mode not in ("", "wheel", "sdist"):
            problems.append(f"QuickstartSessionResult: install_mode {self.install_mode!r} unknown")
        if self.cache_state not in ("", "cold", "warm"):
            problems.append(f"QuickstartSessionResult: cache_state {self.cache_state!r} unknown")
        if self.network_state not in ("", "online", "offline"):
            problems.append(f"QuickstartSessionResult: network_state {self.network_state!r} unknown")
        for name, value in self.stage_durations_s.items():
            if name not in self.TIMING_STAGES:
                problems.append(f"QuickstartSessionResult: unknown timing stage {name!r}")
            if value < 0:
                problems.append(f"QuickstartSessionResult: negative duration for stage {name!r}")
        if self.status == STATUS_PASS:
            if self.undocumented_interventions:
                problems.append(
                    "QuickstartSessionResult: PASS with material interventions is not independent "
                    "(E15-02 §9.2)"
                )
            if not self.first_verified_report_at_utc:
                problems.append("QuickstartSessionResult: PASS requires the first-verified-report timestamp")
            if not self.output_semantic_hash:
                problems.append("QuickstartSessionResult: PASS requires the regenerated output hash")
            if not self.selected_backend:
                problems.append("QuickstartSessionResult: PASS requires the selected backend to be explicit")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "operator_id": self.operator_id,
            "environment_id": self.environment_id,
            "install_mode": self.install_mode,
            "cache_state": self.cache_state,
            "network_state": self.network_state,
            "started_at_utc": self.started_at_utc,
            "first_verified_report_at_utc": self.first_verified_report_at_utc,
            "stage_durations_s": {key: self.stage_durations_s[key] for key in sorted(self.stage_durations_s)},
            "download_bytes": self.download_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "disk_delta_bytes": self.disk_delta_bytes,
            "undocumented_interventions": list(self.undocumented_interventions),
            "selected_backend": self.selected_backend,
            "output_semantic_hash": self.output_semantic_hash,
            "negative_case_results": [dict(item) for item in self.negative_case_results],
            "status": self.status,
        }
        return payload


@dataclass
class HeroReplayResult:
    """``E15-03`` §8 ``HeroReplayResult`` 最小字段."""

    replay_id: str
    candidate_id: str = ""
    hero_claim_id: str = ""
    original_campaign_id: str = ""
    environment_id: str = ""
    comparability: str = ""
    model_artifact_id: str = ""
    workload_ids: Tuple[str, ...] = ()
    baseline_id: str = ""
    candidate_impl_id: str = ""
    correctness_status: str = STATUS_NOT_RUN
    quality_status: str = STATUS_NOT_RUN
    actual_path_status: str = STATUS_NOT_RUN
    primary_effect: Mapping[str, Any] = field(default_factory=dict)
    guard_band_result: str = STATUS_NOT_RUN
    amdahl_prediction: Mapping[str, Any] = field(default_factory=dict)
    mechanism_consistency: str = STATUS_NOT_RUN
    reproduction_class: str = STATUS_NOT_RUN
    claim_actions: Tuple[str, ...] = ()
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-03.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.replay_id:
            problems.append("HeroReplayResult: replay_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"HeroReplayResult: unknown status {self.status!r}")
        if self.comparability and self.comparability not in COMPARABILITY_CLASSES:
            problems.append(f"HeroReplayResult: unknown comparability {self.comparability!r}")
        if self.status == STATUS_PASS:
            if self.correctness_status != STATUS_PASS:
                problems.append("HeroReplayResult: performance may not be claimed before correctness passes")
            if self.actual_path_status != STATUS_PASS:
                problems.append("HeroReplayResult: PASS requires a proven actual path (config names are not evidence)")
            if not self.guard_band_result:
                problems.append("HeroReplayResult: PASS requires a guard-band verdict")
            if self.reproduction_class not in REPLAY_VERDICTS:
                problems.append(
                    f"HeroReplayResult: PASS requires a verdict from {', '.join(REPLAY_VERDICTS)}; "
                    f"got {self.reproduction_class!r}"
                )
            if not self.model_artifact_id or not self.baseline_id or not self.candidate_impl_id:
                problems.append("HeroReplayResult: PASS requires model/baseline/candidate identity")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "replay_id": self.replay_id,
            "candidate_id": self.candidate_id,
            "hero_claim_id": self.hero_claim_id,
            "original_campaign_id": self.original_campaign_id,
            "environment_id": self.environment_id,
            "comparability": self.comparability,
            "model_artifact_id": self.model_artifact_id,
            "workload_ids": list(self.workload_ids),
            "baseline_id": self.baseline_id,
            "candidate_impl_id": self.candidate_impl_id,
            "correctness_status": self.correctness_status,
            "quality_status": self.quality_status,
            "actual_path_status": self.actual_path_status,
            "primary_effect": dict(sorted(self.primary_effect.items())),
            "guard_band_result": self.guard_band_result,
            "amdahl_prediction": dict(sorted(self.amdahl_prediction.items())),
            "mechanism_consistency": self.mechanism_consistency,
            "reproduction_class": self.reproduction_class,
            "claim_actions": list(self.claim_actions),
            "status": self.status,
        }


@dataclass
class DocumentationVerificationRecord:
    """``E15-04`` §7 ``DocumentationVerificationRecord`` 最小字段."""

    verification_id: str
    candidate_id: str = ""
    builder_identity: str = ""
    page_count: int = 0
    build_errors: int = 0
    blocking_warnings: int = 0
    broken_internal_links: int = 0
    broken_external_links: int = 0
    command_counts: Mapping[str, int] = field(default_factory=dict)
    schema_example_failures: int = 0
    cli_diff_count: int = 0
    matrix_mismatch_count: int = 0
    bilingual_critical_diff_count: int = 0
    privacy_findings: Tuple[str, ...] = ()
    negative_control_metrics: Mapping[str, Any] = field(default_factory=dict)
    site_digest: str = ""
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-04.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.verification_id:
            problems.append("DocumentationVerificationRecord: verification_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"DocumentationVerificationRecord: unknown status {self.status!r}")
        unknown = [name for name in self.command_counts if name not in COMMAND_VERDICTS]
        if unknown:
            problems.append(f"DocumentationVerificationRecord: unknown command verdicts {', '.join(unknown)}")
        if "SKIP" in self.command_counts:
            problems.append(
                "DocumentationVerificationRecord: SKIP is not a final command verdict (E15-04 §9.1) — "
                "use BLOCKED_DEVICE with an owner/reason"
            )
        if self.status == STATUS_PASS:
            for name in (
                "build_errors",
                "blocking_warnings",
                "broken_internal_links",
                "schema_example_failures",
                "matrix_mismatch_count",
                "bilingual_critical_diff_count",
            ):
                if getattr(self, name):
                    problems.append(f"DocumentationVerificationRecord: PASS with {name}={getattr(self, name)}")
            if self.privacy_findings:
                problems.append("DocumentationVerificationRecord: PASS with privacy findings")
            if not self.site_digest:
                problems.append("DocumentationVerificationRecord: PASS requires a generated-site digest")
            if not self.negative_control_metrics:
                problems.append("DocumentationVerificationRecord: PASS requires negative-control metrics")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verification_id": self.verification_id,
            "candidate_id": self.candidate_id,
            "builder_identity": self.builder_identity,
            "page_count": self.page_count,
            "build_errors": self.build_errors,
            "blocking_warnings": self.blocking_warnings,
            "broken_internal_links": self.broken_internal_links,
            "broken_external_links": self.broken_external_links,
            "command_counts": {key: int(value) for key, value in sorted(self.command_counts.items())},
            "schema_example_failures": self.schema_example_failures,
            "cli_diff_count": self.cli_diff_count,
            "matrix_mismatch_count": self.matrix_mismatch_count,
            "bilingual_critical_diff_count": self.bilingual_critical_diff_count,
            "privacy_findings": list(self.privacy_findings),
            "negative_control_metrics": dict(sorted(self.negative_control_metrics.items())),
            "site_digest": self.site_digest,
            "status": self.status,
        }


@dataclass
class ReleaseArtifactRecord:
    """``E15-05`` §7 ``ReleaseArtifactRecord`` 最小字段."""

    release_audit_id: str
    release_version: str = ""
    tag: str = ""
    source_commit: str = ""
    artifact_id: str = ""
    artifact_type: str = ""
    platform: str = ""
    size_bytes: int = 0
    sha256: str = ""
    builder_id: str = ""
    build_definition_digest: str = ""
    provenance_ref: str = ""
    attestation_ref: Optional[str] = None
    sbom_ref: str = ""
    license_decision: str = ""
    vulnerability_decision: str = ""
    secret_scan_status: str = STATUS_NOT_RUN
    consumer_verification_status: str = STATUS_NOT_RUN
    reproducible_build_status: str = STATUS_NOT_RUN
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-05.v1"

    ARTIFACT_TYPES: Tuple[str, ...] = ("wheel", "sdist", "oci", "sample", "evidence", "docs")

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.release_audit_id:
            problems.append("ReleaseArtifactRecord: release_audit_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"ReleaseArtifactRecord: unknown status {self.status!r}")
        if self.artifact_type and self.artifact_type not in self.ARTIFACT_TYPES:
            problems.append(f"ReleaseArtifactRecord: unknown artifact_type {self.artifact_type!r}")
        if self.license_decision and self.license_decision not in ("ALLOW", "DENY", "CONDITIONAL"):
            problems.append(f"ReleaseArtifactRecord: unknown license_decision {self.license_decision!r}")
        if self.vulnerability_decision and self.vulnerability_decision not in ("PASS", "BLOCK", "ACCEPTED_RISK"):
            problems.append(f"ReleaseArtifactRecord: unknown vulnerability_decision {self.vulnerability_decision!r}")
        if self.status == STATUS_PASS:
            for name in ("release_version", "tag", "source_commit", "sha256", "builder_id", "provenance_ref", "sbom_ref"):
                if not getattr(self, name):
                    problems.append(f"ReleaseArtifactRecord: PASS requires {name}")
            if self.sha256 and not is_digest(self.sha256):
                problems.append("ReleaseArtifactRecord: sha256 must be sha256:<hex>")
            if self.license_decision != "ALLOW":
                problems.append(f"ReleaseArtifactRecord: PASS requires license_decision=ALLOW, got {self.license_decision!r}")
            if self.vulnerability_decision not in ("PASS", "ACCEPTED_RISK"):
                problems.append("ReleaseArtifactRecord: PASS requires a disposed vulnerability decision")
            if self.consumer_verification_status == STATUS_NOT_RUN:
                problems.append("ReleaseArtifactRecord: PASS requires consumer verification to have run")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_audit_id": self.release_audit_id,
            "release_version": self.release_version,
            "tag": self.tag,
            "source_commit": self.source_commit,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "platform": self.platform,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "builder_id": self.builder_id,
            "build_definition_digest": self.build_definition_digest,
            "provenance_ref": self.provenance_ref,
            "attestation_ref": self.attestation_ref,
            "sbom_ref": self.sbom_ref,
            "license_decision": self.license_decision,
            "vulnerability_decision": self.vulnerability_decision,
            "secret_scan_status": self.secret_scan_status,
            "consumer_verification_status": self.consumer_verification_status,
            "reproducible_build_status": self.reproducible_build_status,
            "status": self.status,
        }


@dataclass
class PointLineageRecord:
    """``E15-06`` §7 ``PointLineageRecord`` 最小字段."""

    audit_id: str
    candidate_id: str = ""
    figure_id: str = ""
    figure_version: str = ""
    point_id: str = ""
    claim_ids: Tuple[str, ...] = ()
    published_value: Optional[float] = None
    published_unit: str = ""
    query_digest: str = ""
    aggregate_activity_id: str = ""
    normalized_row_ids: Tuple[str, ...] = ()
    raw_sample_ids: Tuple[str, ...] = ()
    experiment_unit: str = ""
    n: int = 0
    interval_definition: str = ""
    identity_refs: Tuple[str, ...] = ()
    lineage_complete: bool = False
    numeric_diff: Optional[float] = None
    visual_diff_class: str = ""
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-06.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.audit_id or not self.figure_id or not self.point_id:
            problems.append("PointLineageRecord: audit_id, figure_id and point_id are required")
        if self.status not in ALL_STATUSES:
            problems.append(f"PointLineageRecord: unknown status {self.status!r}")
        if self.visual_diff_class and self.visual_diff_class not in REBUILD_CLASSES:
            problems.append(f"PointLineageRecord: unknown visual_diff_class {self.visual_diff_class!r}")
        if self.n < 0:
            problems.append("PointLineageRecord: n may not be negative")
        if self.status == STATUS_PASS:
            if not self.lineage_complete:
                problems.append("PointLineageRecord: PASS requires a complete lineage chain")
            if self.visual_diff_class in ("D3", "D4", "D5"):
                problems.append(
                    f"PointLineageRecord: PASS with rebuild class {self.visual_diff_class} "
                    "(interval/n/label/query/member-set changes are failures — E15-06 §9.1)"
                )
            if not self.query_digest or not self.aggregate_activity_id:
                problems.append("PointLineageRecord: PASS requires a versioned query and aggregate activity")
            if not self.interval_definition:
                problems.append("PointLineageRecord: PASS requires an explicit error-bar definition")
            if self.n <= 0:
                problems.append("PointLineageRecord: PASS requires an experiment unit count > 0")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "candidate_id": self.candidate_id,
            "figure_id": self.figure_id,
            "figure_version": self.figure_version,
            "point_id": self.point_id,
            "claim_ids": list(self.claim_ids),
            "published_value": self.published_value,
            "published_unit": self.published_unit,
            "query_digest": self.query_digest,
            "aggregate_activity_id": self.aggregate_activity_id,
            "normalized_row_ids": list(self.normalized_row_ids),
            "raw_sample_ids": list(self.raw_sample_ids),
            "experiment_unit": self.experiment_unit,
            "n": self.n,
            "interval_definition": self.interval_definition,
            "identity_refs": list(self.identity_refs),
            "lineage_complete": self.lineage_complete,
            "numeric_diff": self.numeric_diff,
            "visual_diff_class": self.visual_diff_class,
            "status": self.status,
        }


@dataclass
class DemoSessionResult:
    """``E15-07`` §7 ``DemoSessionResult`` 最小字段."""

    session_id: str
    candidate_id: str = ""
    operator_id: str = ""
    environment_id: str = ""
    scenario: str = ""
    started_at_utc: str = ""
    completed_at_utc: Optional[str] = None
    duration_s: float = 0.0
    segments: Tuple[Mapping[str, Any], ...] = ()
    measurement_states_shown: Tuple[str, ...] = ()
    fault_detect_time_s: Optional[float] = None
    fallback_selected: Optional[str] = None
    fallback_time_s: Optional[float] = None
    undocumented_actions: Tuple[str, ...] = ()
    evidence_lookup_time_s: Optional[float] = None
    observer_state_accuracy: Optional[float] = None
    privacy_findings: Tuple[str, ...] = ()
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-07.v1"

    #: Demo time budget (``E15-07`` §9.1): the normal script must fit 3–5 minutes.
    MIN_DURATION_S = 180.0
    MAX_DURATION_S = 300.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.session_id:
            problems.append("DemoSessionResult: session_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"DemoSessionResult: unknown status {self.status!r}")
        if self.scenario and self.scenario not in DEMO_FAULT_SCENARIOS:
            problems.append(f"DemoSessionResult: unknown scenario {self.scenario!r}")
        unknown = [state for state in self.measurement_states_shown if state not in DEMO_MEASUREMENT_STATES]
        if unknown:
            problems.append(f"DemoSessionResult: unknown measurement states {', '.join(unknown)}")
        if self.duration_s < 0:
            problems.append("DemoSessionResult: duration_s may not be negative")
        if self.observer_state_accuracy is not None and not 0.0 <= self.observer_state_accuracy <= 1.0:
            problems.append("DemoSessionResult: observer_state_accuracy must be in [0, 1]")
        if self.status == STATUS_PASS:
            if self.scenario == "normal" and not (self.MIN_DURATION_S <= self.duration_s <= self.MAX_DURATION_S):
                problems.append(
                    f"DemoSessionResult: normal-path PASS requires 3–5 minutes, got {self.duration_s}s"
                )
            if self.scenario != "normal" and self.fallback_selected is None:
                problems.append("DemoSessionResult: a fault scenario PASS requires a selected fallback")
            if not self.measurement_states_shown:
                problems.append("DemoSessionResult: PASS requires the live/cached/prerecorded states to be explicit")
            if self.privacy_findings:
                problems.append("DemoSessionResult: PASS with privacy findings")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "operator_id": self.operator_id,
            "environment_id": self.environment_id,
            "scenario": self.scenario,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "duration_s": self.duration_s,
            "segments": [dict(item) for item in self.segments],
            "measurement_states_shown": list(self.measurement_states_shown),
            "fault_detect_time_s": self.fault_detect_time_s,
            "fallback_selected": self.fallback_selected,
            "fallback_time_s": self.fallback_time_s,
            "undocumented_actions": list(self.undocumented_actions),
            "evidence_lookup_time_s": self.evidence_lookup_time_s,
            "observer_state_accuracy": self.observer_state_accuracy,
            "privacy_findings": list(self.privacy_findings),
            "status": self.status,
        }


@dataclass
class InterviewSessionResult:
    """``E15-08`` §7 ``InterviewSessionResult`` 最小字段."""

    session_id: str
    candidate_id: str = ""
    duration_variant: str = ""
    role_variant: str = ""
    actual_duration_s: float = 0.0
    required_nodes_covered: Tuple[str, ...] = ()
    fact_errors: Tuple[str, ...] = ()
    overclaims: Tuple[str, ...] = ()
    evidence_refs_used: Tuple[str, ...] = ()
    question_categories: Tuple[str, ...] = ()
    answer_scores: Mapping[str, float] = field(default_factory=dict)
    unknown_handling_scores: Tuple[float, ...] = ()
    evidence_lookup_times_s: Tuple[float, ...] = ()
    attribution_mismatches: Tuple[str, ...] = ()
    reviewer_ids: Tuple[str, ...] = ()
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-08.v1"

    #: The three nested durations and their time gates (``E15-08`` §9.1).
    DURATION_VARIANTS: Tuple[str, ...] = ("3m", "10m", "30m")
    ROLE_VARIANTS: Tuple[str, ...] = ("telecom_grid", "ai_infra_kernel")
    #: Seconds allowed for each variant; a longer session overran its gate.
    DURATION_LIMITS_S: ClassVar[Mapping[str, float]] = {"3m": 180.0, "10m": 600.0, "30m": 1800.0}

    #: Content nodes every variant must cover (``E15-08`` §9.1 rows).
    REQUIRED_NODES: Tuple[str, ...] = (
        "Problem",
        "Architecture",
        "Hero",
        "Correctness",
        "Performance",
        "Engineering",
        "Contribution",
        "Limitation",
    )

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.session_id:
            problems.append("InterviewSessionResult: session_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"InterviewSessionResult: unknown status {self.status!r}")
        if self.duration_variant and self.duration_variant not in self.DURATION_VARIANTS:
            problems.append(f"InterviewSessionResult: unknown duration_variant {self.duration_variant!r}")
        if self.role_variant and self.role_variant not in self.ROLE_VARIANTS:
            problems.append(f"InterviewSessionResult: unknown role_variant {self.role_variant!r}")
        unknown = [node for node in self.required_nodes_covered if node not in self.REQUIRED_NODES]
        if unknown:
            problems.append(f"InterviewSessionResult: unknown content nodes {', '.join(unknown)}")
        if self.status == STATUS_PASS:
            if self.fact_errors or self.overclaims or self.attribution_mismatches:
                problems.append(
                    "InterviewSessionResult: PASS with fact errors / overclaims / attribution mismatches "
                    "is contradictory (E15-08 §9: 硬失败)"
                )
            if not self.reviewer_ids:
                problems.append("InterviewSessionResult: PASS requires at least one reviewer")
            if self.actual_duration_s <= 0:
                problems.append("InterviewSessionResult: PASS requires an actual duration")
            limit = self.DURATION_LIMITS_S.get(self.duration_variant)
            if limit is not None and self.actual_duration_s > limit:
                problems.append(
                    f"InterviewSessionResult: {self.duration_variant} overran its {limit:.0f}s gate"
                )
            if len(self.required_nodes_covered) < len(self.REQUIRED_NODES):
                problems.append("InterviewSessionResult: PASS requires every content node")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "duration_variant": self.duration_variant,
            "role_variant": self.role_variant,
            "actual_duration_s": self.actual_duration_s,
            "required_nodes_covered": list(self.required_nodes_covered),
            "fact_errors": list(self.fact_errors),
            "overclaims": list(self.overclaims),
            "evidence_refs_used": list(self.evidence_refs_used),
            "question_categories": list(self.question_categories),
            "answer_scores": {key: self.answer_scores[key] for key in sorted(self.answer_scores)},
            "unknown_handling_scores": list(self.unknown_handling_scores),
            "evidence_lookup_times_s": list(self.evidence_lookup_times_s),
            "attribution_mismatches": list(self.attribution_mismatches),
            "reviewer_ids": list(self.reviewer_ids),
            "status": self.status,
        }


@dataclass
class CleanRoomReproductionRecord:
    """``E15-09`` §7 ``CleanRoomReproductionRecord`` 最小字段."""

    campaign_id: str
    candidate_ids: Tuple[str, ...] = ()
    reviewer_id: str = ""
    independence_status: str = STATUS_NOT_RUN
    received_materials_digest: str = ""
    environment_id: str = ""
    comparability: str = ""
    target_level: str = ""
    achieved_level: Optional[str] = None
    stage_times_s: Mapping[str, float] = field(default_factory=dict)
    help_events_by_level: Mapping[str, int] = field(default_factory=dict)
    blocking_findings: Tuple[str, ...] = ()
    correctness_status: str = STATUS_NOT_RUN
    actual_path_status: str = STATUS_NOT_RUN
    effect_consistency: str = STATUS_NOT_RUN
    mechanism_consistency: str = STATUS_NOT_RUN
    issues: Tuple[str, ...] = ()
    fix_retest_status: str = STATUS_NOT_RUN
    report_uri: str = ""
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-09.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.campaign_id:
            problems.append("CleanRoomReproductionRecord: campaign_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"CleanRoomReproductionRecord: unknown status {self.status!r}")
        if self.target_level and self.target_level not in REPRODUCTION_LEVELS:
            problems.append(f"CleanRoomReproductionRecord: unknown target_level {self.target_level!r}")
        if self.achieved_level and self.achieved_level not in REPRODUCTION_LEVELS:
            problems.append(f"CleanRoomReproductionRecord: unknown achieved_level {self.achieved_level!r}")
        if self.comparability and self.comparability not in COMPARABILITY_CLASSES:
            problems.append(f"CleanRoomReproductionRecord: unknown comparability {self.comparability!r}")
        unknown = [level for level in self.help_events_by_level if level not in HELP_LEVELS]
        if unknown:
            problems.append(f"CleanRoomReproductionRecord: unknown help levels {', '.join(unknown)}")
        if self.status == STATUS_PASS:
            if self.help_events_by_level.get("L3") or self.help_events_by_level.get("L4"):
                problems.append(
                    "CleanRoomReproductionRecord: L3/L4 help invalidates the current candidate "
                    "(E15-09 §9.1)"
                )
            if not self.reviewer_id:
                problems.append("CleanRoomReproductionRecord: PASS requires an (at least pseudonymous) reviewer")
            if not self.independence_status or self.independence_status == STATUS_NOT_RUN:
                problems.append("CleanRoomReproductionRecord: PASS requires an independence verdict")
            if not self.received_materials_digest:
                problems.append("CleanRoomReproductionRecord: PASS requires a received-materials digest")
            if self.correctness_status != STATUS_PASS or self.actual_path_status != STATUS_PASS:
                problems.append("CleanRoomReproductionRecord: PASS requires correctness and actual path first")
            if not self.report_uri:
                problems.append("CleanRoomReproductionRecord: PASS requires the reviewer's report")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "candidate_ids": list(self.candidate_ids),
            "reviewer_id": self.reviewer_id,
            "independence_status": self.independence_status,
            "received_materials_digest": self.received_materials_digest,
            "environment_id": self.environment_id,
            "comparability": self.comparability,
            "target_level": self.target_level,
            "achieved_level": self.achieved_level,
            "stage_times_s": {key: self.stage_times_s[key] for key in sorted(self.stage_times_s)},
            "help_events_by_level": {key: int(self.help_events_by_level[key]) for key in sorted(self.help_events_by_level)},
            "blocking_findings": list(self.blocking_findings),
            "correctness_status": self.correctness_status,
            "actual_path_status": self.actual_path_status,
            "effect_consistency": self.effect_consistency,
            "mechanism_consistency": self.mechanism_consistency,
            "issues": list(self.issues),
            "fix_retest_status": self.fix_retest_status,
            "report_uri": self.report_uri,
            "status": self.status,
        }


@dataclass
class UpstreamContributionRecord:
    """``E15-10`` §7 ``UpstreamContributionRecord`` 最小字段."""

    contribution_id: str
    candidate_id: str = ""
    upstream_repository: str = ""
    upstream_commit_or_release: str = ""
    contribution_type: str = ""
    source_finding_ids: Tuple[str, ...] = ()
    boundary_triage: str = ""
    duplicate_status: str = STATUS_NOT_RUN
    reproducer_uri: str = ""
    reproduction_rate: Optional[float] = None
    negative_controls: Tuple[str, ...] = ()
    issue_url: Optional[str] = None
    pr_url: Optional[str] = None
    test_status: str = STATUS_NOT_RUN
    benchmark_status: str = STATUS_NOT_RUN
    review_rounds: int = 0
    upstream_status: str = STATUS_NOT_RUN
    personal_contributions: Tuple[str, ...] = ()
    agent_or_third_party_contributions: Tuple[str, ...] = ()
    hqsb_downstream_actions: Tuple[str, ...] = ()
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-10.v1"

    CONTRIBUTION_TYPES: Tuple[str, ...] = ("issue", "rfc", "docs", "test", "code", "performance")

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.contribution_id:
            problems.append("UpstreamContributionRecord: contribution_id is required")
        if self.status not in ALL_STATUSES:
            problems.append(f"UpstreamContributionRecord: unknown status {self.status!r}")
        if self.contribution_type and self.contribution_type not in self.CONTRIBUTION_TYPES:
            problems.append(f"UpstreamContributionRecord: unknown contribution_type {self.contribution_type!r}")
        if self.boundary_triage and self.boundary_triage not in BOUNDARY_TRIAGE:
            problems.append(f"UpstreamContributionRecord: unknown boundary_triage {self.boundary_triage!r}")
        if self.upstream_status != STATUS_NOT_RUN and self.upstream_status not in UPSTREAM_STATUSES:
            problems.append(f"UpstreamContributionRecord: unknown upstream_status {self.upstream_status!r}")
        if self.reproduction_rate is not None and not 0.0 <= self.reproduction_rate <= 1.0:
            problems.append("UpstreamContributionRecord: reproduction_rate must be in [0, 1]")
        if self.status == STATUS_PASS:
            if self.boundary_triage != "upstream":
                problems.append(
                    "UpstreamContributionRecord: PASS requires boundary_triage=upstream "
                    "(a local bug pushed upstream is a failure — E15-10 §1)"
                )
            if not self.source_finding_ids:
                problems.append("UpstreamContributionRecord: PASS requires the originating HQSB finding ids")
            if not self.reproducer_uri:
                problems.append("UpstreamContributionRecord: PASS requires a minimal reproducer")
            if not self.negative_controls:
                problems.append("UpstreamContributionRecord: PASS requires negative controls")
            if not (self.issue_url or self.pr_url):
                problems.append("UpstreamContributionRecord: PASS requires a public URL")
            if self.duplicate_status == STATUS_NOT_RUN:
                problems.append("UpstreamContributionRecord: PASS requires a duplicate search verdict")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contribution_id": self.contribution_id,
            "candidate_id": self.candidate_id,
            "upstream_repository": self.upstream_repository,
            "upstream_commit_or_release": self.upstream_commit_or_release,
            "contribution_type": self.contribution_type,
            "source_finding_ids": list(self.source_finding_ids),
            "boundary_triage": self.boundary_triage,
            "duplicate_status": self.duplicate_status,
            "reproducer_uri": self.reproducer_uri,
            "reproduction_rate": self.reproduction_rate,
            "negative_controls": list(self.negative_controls),
            "issue_url": self.issue_url,
            "pr_url": self.pr_url,
            "test_status": self.test_status,
            "benchmark_status": self.benchmark_status,
            "review_rounds": self.review_rounds,
            "upstream_status": self.upstream_status,
            "personal_contributions": list(self.personal_contributions),
            "agent_or_third_party_contributions": list(self.agent_or_third_party_contributions),
            "hqsb_downstream_actions": list(self.hqsb_downstream_actions),
            "status": self.status,
        }


@dataclass
class FirstImpressionSessionResult:
    """``E15-11`` §7 ``FirstImpressionSessionResult`` 最小字段."""

    study_id: str
    candidate_id: str = ""
    participant_id: str = ""
    role_block: str = ""
    prior_hqsb_exposure: bool = False
    session_mode: str = ""
    browse_duration_s: float = 0.0
    first_click: str = ""
    path: Tuple[str, ...] = ()
    free_recall_scores: Mapping[str, str] = field(default_factory=dict)
    structured_task_scores: Mapping[str, str] = field(default_factory=dict)
    evidence_lookup_time_s: Optional[float] = None
    evidence_lookup_clicks: int = 0
    status_label_understanding: Mapping[str, str] = field(default_factory=dict)
    attribution_understanding: Mapping[str, str] = field(default_factory=dict)
    limitation_found: bool = False
    misconceptions: Tuple[Mapping[str, Any], ...] = ()
    confidence: Mapping[str, Any] = field(default_factory=dict)
    status: str = STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e15-11.v1"

    ROLE_BLOCKS: Tuple[str, ...] = ("telecom_grid", "ai_infra_kernel", "other")
    SESSION_MODES: Tuple[str, ...] = ("natural", "think_aloud")
    #: Formation-style ceiling of the protocol (2–3 participants, not a survey).
    MAX_PARTICIPANTS = 3
    BROWSE_MINUTES = 5.0
    #: Score codes used by the answer key (``E15-11`` step 13).
    SCORE_CODES: Tuple[str, ...] = ("correct", "partial", "incorrect")

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.study_id or not self.participant_id:
            problems.append("FirstImpressionSessionResult: study_id and participant_id are required")
        if self.status not in ALL_STATUSES:
            problems.append(f"FirstImpressionSessionResult: unknown status {self.status!r}")
        if self.role_block and self.role_block not in self.ROLE_BLOCKS:
            problems.append(f"FirstImpressionSessionResult: unknown role_block {self.role_block!r}")
        if self.session_mode and self.session_mode not in self.SESSION_MODES:
            problems.append(f"FirstImpressionSessionResult: unknown session_mode {self.session_mode!r}")
        for scores in (self.free_recall_scores, self.structured_task_scores):
            unknown = [code for code in scores.values() if code not in self.SCORE_CODES]
            if unknown:
                problems.append(f"FirstImpressionSessionResult: unknown score codes {', '.join(sorted(set(unknown)))}")
        if self.status == STATUS_PASS:
            if self.prior_hqsb_exposure:
                problems.append("FirstImpressionSessionResult: PASS requires a participant without prior exposure")
            if not self.path:
                problems.append("FirstImpressionSessionResult: PASS requires the browse path")
            if not self.free_recall_scores or not self.structured_task_scores:
                problems.append("FirstImpressionSessionResult: PASS requires recall and task scores")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "candidate_id": self.candidate_id,
            "participant_id": self.participant_id,
            "role_block": self.role_block,
            "prior_hqsb_exposure": self.prior_hqsb_exposure,
            "session_mode": self.session_mode,
            "browse_duration_s": self.browse_duration_s,
            "first_click": self.first_click,
            "path": list(self.path),
            "free_recall_scores": dict(sorted(self.free_recall_scores.items())),
            "structured_task_scores": dict(sorted(self.structured_task_scores.items())),
            "evidence_lookup_time_s": self.evidence_lookup_time_s,
            "evidence_lookup_clicks": self.evidence_lookup_clicks,
            "status_label_understanding": dict(sorted(self.status_label_understanding.items())),
            "attribution_understanding": dict(sorted(self.attribution_understanding.items())),
            "limitation_found": self.limitation_found,
            "misconceptions": [dict(item) for item in self.misconceptions],
            "confidence": dict(sorted(self.confidence.items())),
            "status": self.status,
        }


#: Record class of each experiment (used by the driver and the tests).
RECORD_CLASSES: Mapping[str, str] = {
    "E15-01": "ClaimAuditResult",
    "E15-02": "QuickstartSessionResult",
    "E15-03": "HeroReplayResult",
    "E15-04": "DocumentationVerificationRecord",
    "E15-05": "ReleaseArtifactRecord",
    "E15-06": "PointLineageRecord",
    "E15-07": "DemoSessionResult",
    "E15-08": "InterviewSessionResult",
    "E15-09": "CleanRoomReproductionRecord",
    "E15-10": "UpstreamContributionRecord",
    "E15-11": "FirstImpressionSessionResult",
}


def record_class(experiment_id: str) -> type:
    """Resolve the record class of an experiment without a wildcard import."""
    return _RECORD_REGISTRY[experiment_id]


_RECORD_REGISTRY: Mapping[str, type] = {
    "E15-01": ClaimAuditResult,
    "E15-02": QuickstartSessionResult,
    "E15-03": HeroReplayResult,
    "E15-04": DocumentationVerificationRecord,
    "E15-05": ReleaseArtifactRecord,
    "E15-06": PointLineageRecord,
    "E15-07": DemoSessionResult,
    "E15-08": InterviewSessionResult,
    "E15-09": CleanRoomReproductionRecord,
    "E15-10": UpstreamContributionRecord,
    "E15-11": FirstImpressionSessionResult,
}


@dataclass
class Finding:
    """A structured finding (§22 ``findings.json``): severity, owner, disposition."""

    finding_id: str
    severity: str
    owner: str
    affected_claims: Tuple[str, ...] = ()
    disposition: str = "OPEN"
    summary: str = ""
    evidence_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.finding_id:
            problems.append("Finding: finding_id is required")
        if self.severity not in SEVERITIES:
            problems.append(f"Finding: unknown severity {self.severity!r}")
        if not self.owner:
            problems.append("Finding: an owner is required (no owner means no fix path)")
        if self.disposition not in FINDING_DISPOSITIONS:
            problems.append(f"Finding: unknown disposition {self.disposition!r}")
        if self.disposition == "ACCEPTED_RISK_WITH_OWNER_AND_EXPIRY" and not self.owner:
            problems.append("Finding: accepted risk requires an owner")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "owner": self.owner,
            "affected_claims": list(self.affected_claims),
            "disposition": self.disposition,
            "summary": self.summary,
            "evidence_refs": list(self.evidence_refs),
        }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the vocabulary layer (labelled smoke, not an experiment)."""
    problems = validate_state_machines()
    # Negative controls: a conclusion without evidence must be refused.
    negative_controls: Dict[str, List[str]] = {}
    negative_controls["claim_audit_pass_without_ledger"] = ClaimAuditResult(
        audit_id="audit-1", status=STATUS_PASS
    ).validate()
    negative_controls["demo_pass_without_interval"] = PointLineageRecord(
        audit_id="a", figure_id="f", point_id="p", status=STATUS_PASS
    ).validate()
    negative_controls["reproduction_pass_with_L3_help"] = CleanRoomReproductionRecord(
        campaign_id="c",
        reviewer_id="r",
        independence_status=STATUS_PASS,
        received_materials_digest="sha256:" + "0" * 64,
        correctness_status=STATUS_PASS,
        actual_path_status=STATUS_PASS,
        report_uri="reports/x.md",
        help_events_by_level={"L3": 1},
        status=STATUS_PASS,
    ).validate()
    if not all(negative_controls.values()):
        problems.append("the negative controls did not reject the invalid records")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiments": len(EXPERIMENT_TABLE),
        "steps_expected": len(EXPERIMENT_TABLE) * STEPS_PER_EXPERIMENT,
        "statuses": list(ALL_STATUSES),
        "record_classes": len(_RECORD_REGISTRY),
        "negative_controls_rejected": sum(1 for problems_ in negative_controls.values() if problems_),
        "problems": problems,
    }
