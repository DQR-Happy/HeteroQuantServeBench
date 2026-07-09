"""E14-05 — frontier branch ADR, falsifiable hypothesis and cross-layer preregistration.

This is a *methodology gate*, not an algorithm experiment.  ``E14-05`` makes
"choosing the question and freezing the protocol" itself a P0 experiment, because
the failure mode of frontier work is research drift: run many configurations,
pick the best number, then reverse-engineer a story.

What the forty steps need:

* :class:`LiteratureEntry` / :class:`LiteratureRegistry` — original/official
  sources with versions (steps 4–5); a second-hand blog is not a source;
* :class:`PaperConditionMatrix` — the paper's model/hardware/batch/baseline/
  quality/software/repeats, kept separate from HQSB's, so a reported speedup is
  never transplanted (step 5);
* :func:`hqsb_mapping_matrix` — map the paper object onto C1–C7 (step 6);
* :class:`CandidateEstimand` — one per branch, with the mediator variables named
  (steps 11–14); "performance is better" is refused as an estimand (§3.1);
* :class:`HardPrerequisiteGate`, :class:`QualityGate`, :class:`ActualPathGate`
  (steps 15–17) — a missing prerequisite is ``BLOCKED``, never a mock pass;
* :class:`PrimaryHypothesis`, :func:`check_single_primary` (steps 18–19) —
  exactly one primary, with a minimum meaningful effect;
* :class:`BaselinePair`, :class:`NegativeControls`, :class:`AblationMatrix`
  (steps 20–22);
* :class:`WorkloadStrata`, :class:`ProfilePlan`, :class:`TimingBoundaries`,
  :class:`CostDenominator`, :class:`StatisticsPlan`, :class:`StopRules`,
  :class:`InvalidityRules`, :class:`PredictionModel`, :class:`AttributionRule`,
  :class:`ComplexityScore` (steps 23–35);
* :func:`preregister_adoption_rules`, :func:`select_single_branch`,
  :func:`lock_unselected_branches` (steps 36–38);
* :func:`protocol_review`, :func:`issue_contract` (steps 39–40).

Nothing here runs a configuration; :func:`issue_contract` produces the frozen
``FrontierStudyContract`` of ``details/S14/README.md`` §7.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import (
    AdoptionDecision,
    FrontierStudyContract,
    check_negative_control_coverage,
    check_profile_layering,
)
from hqsb.experimental.identity import SCHEMA_PREFIX, canonical_digest

EXPERIMENT_ID = "E14-05"
TITLE = "前沿方向 ADR、可证伪假设与跨层预注册"
LEVEL = "P0"

CLAIM_BOUNDARY = (
    "只证明问题值得问且实验能给出可信答案；不生成任何算法 speedup，也不代表选中分支会有正收益"
    "（E14-05 §11）。"
)

#: The four candidate branches and their protocol risk text (§4).
CANDIDATE_BRANCHES: Tuple[str, ...] = tuple(rec.FRONTIER_BRANCH_NAMES)

#: Selection criteria: value, job relevance, upstream evidence, hardware
#: feasibility, quality oracle, minimal change, profile layers, budget (steps 2–3).
SELECTION_CRITERIA: Tuple[str, ...] = (
    "problem_value",
    "job_relevance",
    "upstream_evidence",
    "hardware_feasibility",
    "quality_oracle",
    "minimal_independent_change",
    "profile_layers",
    "resource_budget",
)

#: Estimated/observed/unknown — three states, because ``unknown`` is not zero.
EVIDENCE_STATES: Tuple[str, ...] = ("VERIFIED", "ESTIMATED", "UNKNOWN")

#: The HQSB contract fields a paper object may map onto (§6 of the step list).
HQSB_CONTRACT_FIELDS: Tuple[str, ...] = ("C1_model", "C2_workload", "C3_operator", "C4_backend",
                                         "C5_quant", "C6_result", "C7_trace")

#: Required paper conditions (step 5) — the prior/context, never the baseline.
PAPER_CONDITION_FIELDS: Tuple[str, ...] = (
    "model",
    "model_size",
    "hardware",
    "device_count",
    "batch_or_concurrency",
    "context_length",
    "baseline",
    "reported_metric",
    "reported_value",
    "quality_metric",
    "software_versions",
    "repeats",
    "limitations",
)

#: Timing boundaries that must be named separately (step 25).
TIMING_BOUNDARIES: Tuple[str, ...] = (
    "cold_start",
    "warm_start",
    "compile",
    "conversion",
    "prefill",
    "decode",
    "queue",
    "end_to_end",
    "amortisation",
)

#: Cost denominators (step 26) — "per token" is not enough on its own.
COST_DENOMINATORS: Tuple[str, ...] = (
    "per_request",
    "per_output_token",
    "per_quality_goodput_unit",
    "per_device_hour",
)

#: Invalidity categories that must stay distinguishable (step 32).
INVALIDITY_CATEGORIES: Tuple[str, ...] = (
    "unsupported",
    "not_run",
    "tool_failure",
    "fallback",
    "outlier",
    "quality_fail",
    "resource_abort",
)


@dataclass(frozen=True)
class LiteratureEntry:
    """One source with a version (``E14-05`` step 4).

    ``implementation`` and ``version`` are required: a claim whose implementation
    cannot be named is a claim about the paper, not about the system.
    """

    key: str
    kind: str  # paper | official_docs | official_impl
    title: str
    venue: str
    year: int
    version: str
    claim: str
    implementation: str = ""
    limitations: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("key", "title", "venue", "version", "claim"):
            if not getattr(self, name):
                findings.append(f"literature entry {self.key or '<unnamed>'}: {name!r} is required")
        if self.kind not in ("paper", "official_docs", "official_impl"):
            findings.append(
                f"literature entry {self.key}: kind {self.kind!r} must be paper/official_docs/official_impl "
                "(二手博客不能作为来源)"
            )
        if self.year < 2000 or self.year > 2100:
            findings.append(f"literature entry {self.key}: implausible year {self.year}")
        if self.kind == "official_impl" and not self.implementation:
            findings.append(f"literature entry {self.key}: an implementation source must name the implementation")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "title": self.title,
            "venue": self.venue,
            "year": self.year,
            "version": self.version,
            "claim": self.claim,
            "implementation": self.implementation,
            "limitations": list(self.limitations),
        }


@dataclass
class LiteratureRegistry:
    """The candidate literature (``E14-05`` steps 4–5)."""

    entries: Tuple[LiteratureEntry, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        keys: List[str] = []
        for entry in self.entries:
            findings.extend(entry.problems())
            keys.append(entry.key)
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            findings.append(f"duplicate literature keys: {', '.join(duplicates)}")
        for branch in CANDIDATE_BRANCHES:
            if not any(entry.key.startswith(branch) for entry in self.entries):
                findings.append(f"no source recorded for candidate {branch}（每个 F 至少记录原始/官方来源）")
        if not self.entries:
            findings.append("the literature registry is empty")
        return findings

    def digest(self) -> str:
        return canonical_digest([entry.as_dict() for entry in self.entries])

    def as_dict(self) -> Dict[str, Any]:
        return {"entries": [entry.as_dict() for entry in self.entries], "digest": self.digest()}


@dataclass
class PaperConditionMatrix:
    """The paper's conditions, kept apart from HQSB's baseline (step 5)."""

    rows: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.rows:
            findings.append("the paper-condition matrix is empty")
        for key, row in sorted(self.rows.items()):
            missing = [name for name in PAPER_CONDITION_FIELDS if name not in row]
            if missing:
                findings.append(f"{key}: paper conditions missing {', '.join(missing)}")
            if "reported_value" in row and not row.get("reported_metric"):
                findings.append(f"{key}: a reported value without its metric is not comparable")
            for name in ("model", "hardware", "baseline"):
                value = str(row.get(name, "")).lower()
                if value and not any(token in value for token in ("qwen", "llama", "phi", "mistral", "mixtral",
                                                                  "gpt", "opt", "gemma", "tiny", "toy", "n/a")):
                    # Not a hard error: the matrix may legitimately describe an
                    # unfamiliar model.  Recorded so the ADR cannot silently
                    # assume the paper's setup looks like HQSB's.
                    findings.append(
                        f"{key}: {name}={row.get(name)!r} — confirm it is not being used as an HQSB baseline"
                    )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows": {key: dict(self.rows[key]) for key in sorted(self.rows)},
            "digest": canonical_digest({key: dict(self.rows[key]) for key in sorted(self.rows)}),
        }


def hqsb_mapping_matrix(branch: str, mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 6: map the paper object onto C1–C7 instead of inventing a new scale.

    A branch that maps onto fewer than four contracts is emitting a separate
    measurement vocabulary (``E14-05`` §9 热词驱动而非问题驱动).
    """
    if branch not in CANDIDATE_BRANCHES:
        raise ConfigError(f"unknown candidate branch {branch!r}")
    problems: List[str] = []
    for field_name in HQSB_CONTRACT_FIELDS:
        if not mapping.get(field_name):
            problems.append(f"{branch}: no HQSB mapping for {field_name}")
    unmapped = [field_name for field_name in HQSB_CONTRACT_FIELDS if not mapping.get(field_name)]
    return {
        "branch": branch,
        "mapped": {key: mapping[key] for key in sorted(mapping)},
        "coverage": len(HQSB_CONTRACT_FIELDS) - len(unmapped),
        "total": len(HQSB_CONTRACT_FIELDS),
        "problems": problems,
        "ok": not problems,
    }


#: Mediator variables each branch must name (steps 11–14).
BRANCH_MEDIATORS: Mapping[str, Tuple[str, ...]] = {
    "E14-F1": ("acceptance", "draft_time", "verify_time", "kv_bytes", "target_calls", "scheduler_wait"),
    "E14-F2": ("routing_skew", "alltoall_bytes", "expert_gemm_shape", "slowest_rank", "expert_cache_hit"),
    "E14-F3": ("kv_bytes_per_token", "chunk_boundary", "attention_backend", "queue_wait", "oom_margin"),
    "E14-F4": ("pattern_compliance", "actual_kernel", "metadata_bytes", "conversion_cost", "hotspot_coverage"),
}

#: What the branch is *not* allowed to conclude (steps 11–14, §9).
BRANCH_FORBIDDEN_CLAIMS: Mapping[str, Tuple[str, ...]] = {
    "E14-F1": ("用 acceptance 代替 speedup", "把 prefix cache 收益算作 speculative 收益"),
    "E14-F2": ("用 active parameters 代替总显存", "用理论通信字节代替 actual All-to-All"),
    "E14-F3": ("把配置最大长度写成真实支持", "用字符数代替 token 数"),
    "E14-F4": ("把零元素比例等于硬件稀疏", "用 micro speedup 外推模型收益"),
}


@dataclass(frozen=True)
class CandidateEstimand:
    """One candidate's estimand (``E14-05`` steps 11–14)."""

    branch: str
    text: str
    baseline_id: str
    primary_metric: str
    mediators: Tuple[str, ...] = ()
    unit: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.branch not in CANDIDATE_BRANCHES:
            findings.append(f"unknown branch {self.branch!r}")
        if not self.text:
            findings.append(f"{self.branch}: the estimand is empty")
        lowered = self.text.lower()
        if lowered and not any(
            token in lowered
            for token in ("goodput", "slo", "quality", "capacity", "latency", "bytes", "tokens", "cost", "energy")
        ):
            findings.append(
                f"{self.branch}: the estimand names no layer/denominator — '性能更好' 不是 estimand"
            )
        if not self.baseline_id:
            findings.append(f"{self.branch}: an estimand without a baseline is a wish")
        if not self.unit:
            findings.append(f"{self.branch}: the estimand has no unit (token/request/episode)")
        expected = BRANCH_MEDIATORS.get(self.branch, ())
        missing = [name for name in expected if name not in self.mediators]
        if missing:
            findings.append(f"{self.branch}: mediators not named: {', '.join(missing)}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "text": self.text,
            "baseline_id": self.baseline_id,
            "primary_metric": self.primary_metric,
            "mediators": list(self.mediators),
            "unit": self.unit,
        }


@dataclass
class HardPrerequisiteGate:
    """Step 15: model/data licence, device, quality oracle, source access, tools."""

    items: Mapping[str, str] = field(default_factory=dict)

    REQUIRED: Tuple[str, ...] = (
        "model_licence",
        "data_licence",
        "device_capability",
        "quality_oracle",
        "source_access",
        "profile_tooling",
        "isolation",
    )

    def evaluate(self, branch: str) -> Dict[str, Any]:
        blocking: List[str] = []
        unknown: List[str] = []
        for name in self.REQUIRED:
            value = str(self.items.get(name, "")).upper()
            if value == "BLOCKED":
                blocking.append(name)
            elif value != "OK":
                unknown.append(name)
        status = rec.STATUS_BLOCKED_PREREQUISITE if (blocking or unknown) else rec.STATUS_NOT_RUN
        return {
            "branch": branch,
            "blocking": blocking,
            "unknown": unknown,
            "status": status,
            "note": "缺前置即 BLOCKED，不 mock 成通过（step 15）",
            "eligible": not blocking and not unknown,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {"items": {key: self.items[key] for key in sorted(self.items)}}


@dataclass(frozen=True)
class QualityGate:
    """Step 16: a task-native oracle with a paired evaluation and guard band."""

    gate_id: str
    task: str
    metric: str
    paired: bool = True
    guard_band: float = 0.0
    invalid_policy: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("gate_id", "task", "metric", "invalid_policy"):
            if not getattr(self, name):
                findings.append(f"QualityGate: {name!r} is required")
        if not self.paired:
            findings.append(
                f"QualityGate {self.gate_id}: an unpaired evaluation cannot separate a candidate "
                "from workload noise"
            )
        if self.guard_band < 0:
            findings.append(f"QualityGate {self.gate_id}: a negative guard band relaxes the gate")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "task": self.task,
            "metric": self.metric,
            "paired": self.paired,
            "guard_band": self.guard_band,
            "invalid_policy": self.invalid_policy,
        }


@dataclass(frozen=True)
class ActualPathGate:
    """Step 17: the kernel/collective/cache/scheduler evidence that must be seen."""

    gate_id: str
    must_observe: Tuple[str, ...] = ()
    fallback_evidence: str = ""
    instrumented: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.gate_id:
            findings.append("ActualPathGate: gate_id is required")
        if not self.must_observe:
            findings.append(
                f"ActualPathGate {self.gate_id}: must_observe is empty — 配置开启但实现 fallback 就抓不到"
            )
        if not self.instrumented:
            findings.append(
                f"ActualPathGate {self.gate_id}: instrumentation is not declared (profiler/trace/库日志)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "must_observe": list(self.must_observe),
            "fallback_evidence": self.fallback_evidence,
            "instrumented": self.instrumented,
        }


# ── hypotheses (steps 18–19) ───────────────────────────────────────────────


@dataclass(frozen=True)
class PrimaryHypothesis:
    """The single falsifiable claim (step 18)."""

    statement: str
    direction: str
    minimum_effect: str
    applicability: str
    refuted_if: str

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("statement", "direction", "minimum_effect", "applicability", "refuted_if"):
            if not getattr(self, name):
                findings.append(f"PrimaryHypothesis: {name!r} is required")
        if self.direction not in ("increase", "decrease", "non_inferior"):
            findings.append(
                f"PrimaryHypothesis: direction {self.direction!r} must be increase/decrease/non_inferior"
            )
        if not self.refuted_if:
            findings.append("PrimaryHypothesis: a hypothesis without a refutation condition is not falsifiable")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "direction": self.direction,
            "minimum_effect": self.minimum_effect,
            "applicability": self.applicability,
            "refuted_if": self.refuted_if,
        }


def check_single_primary(
    primary: Sequence[PrimaryHypothesis], secondary: Sequence[Mapping[str, Any]]
) -> List[str]:
    """Step 18/19: exactly one primary; secondary stays exploratory.

    Promoting a secondary to primary after seeing the data is the manoeuvre the
    step forbids, so a secondary that is *marked* primary is refused here.
    """
    problems: List[str] = []
    if len(primary) != 1:
        problems.append(f"exactly one primary hypothesis is required, got {len(primary)}")
    for hypothesis in primary:
        problems.extend(hypothesis.problems())
    for index, item in enumerate(secondary):
        if item.get("primary") is True:
            problems.append(
                f"secondary hypothesis {index} is marked primary "
                "(事后把 secondary 升为 primary = FAIL)"
            )
        if not item.get("exploratory") is True:
            problems.append(f"secondary hypothesis {index} must declare exploratory=True")
    return problems


# ── baseline, controls, ablation (steps 20–22) ─────────────────────────────


@dataclass(frozen=True)
class BaselinePair:
    """Step 20: identical everywhere except one intended difference."""

    baseline_id: str
    candidate_id: str
    intended_difference: str
    shared_fields: Mapping[str, Any] = field(default_factory=dict)
    conditional_fields: Tuple[str, ...] = ()
    conditional_reason: str = ""

    #: Fields that must be identical, or explicitly declared conditional.
    FROZEN: Tuple[str, ...] = (
        "model_artifact",
        "weights",
        "quality_suite",
        "hardware",
        "runtime",
        "workload",
        "device_count",
        "precision",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("baseline_id", "candidate_id", "intended_difference"):
            if not getattr(self, name):
                findings.append(f"BaselinePair: {name!r} is required")
        if self.baseline_id and self.baseline_id == self.candidate_id:
            findings.append("BaselinePair: baseline and candidate must differ by one intended change")
        for field_name in self.FROZEN:
            if field_name not in self.shared_fields and field_name not in self.conditional_fields:
                findings.append(
                    f"BaselinePair: {field_name!r} is neither frozen in shared_fields nor declared conditional"
                )
        if self.conditional_fields and not self.conditional_reason:
            findings.append(
                "BaselinePair: a conditional comparison must state why it cannot be equal "
                "(否则 candidate 可能用更低精度或更短输出偷赢)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "intended_difference": self.intended_difference,
            "shared_fields": dict(sorted(self.shared_fields.items())),
            "conditional_fields": list(self.conditional_fields),
            "conditional_reason": self.conditional_reason,
        }


@dataclass
class NegativeControls:
    """Step 21: the controls that prove the instrumentation has detection power."""

    controls: Tuple[str, ...] = ()
    note: str = ""

    def problems(self) -> List[str]:
        return list(check_negative_control_coverage(self.controls))

    def as_dict(self) -> Dict[str, Any]:
        return {"controls": list(self.controls), "note": self.note}


@dataclass(frozen=True)
class AblationMatrix:
    """Step 22: only the parameters that answer a mechanism question."""

    factors: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    max_combinations: int = 24
    mechanism_question: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        combinations = 1
        for factor, levels in self.factors.items():
            if not levels:
                findings.append(f"AblationMatrix: factor {factor!r} has no levels")
                continue
            combinations *= len(levels)
        if combinations > self.max_combinations:
            findings.append(
                f"AblationMatrix: {combinations} combinations exceeds the budget {self.max_combinations} "
                "(笛卡尔爆炸式扫参)"
            )
        if not self.mechanism_question:
            findings.append("AblationMatrix: a mechanism question is required (否则只是调参)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "factors": {key: list(value) for key, value in sorted(self.factors.items())},
            "combinations": _combination_count(self.factors),
            "max_combinations": self.max_combinations,
            "mechanism_question": self.mechanism_question,
        }


def _combination_count(factors: Mapping[str, Sequence[Any]]) -> int:
    total = 1
    for levels in factors.values():
        total *= max(len(levels), 1)
    return total


# ── workload, profile, timing, cost, statistics (steps 23–28) ──────────────


@dataclass(frozen=True)
class WorkloadStrata:
    """Step 23: the strata plus a holdout that was never used for tuning."""

    strata: Tuple[str, ...] = ()
    holdout_id: str = ""
    dims: Tuple[str, ...] = ()

    REQUIRED_DIMS: Tuple[str, ...] = ("length", "concurrency", "shared_prefix", "arrival", "domain")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.strata:
            findings.append("WorkloadStrata: at least one stratum is required")
        missing = [name for name in self.REQUIRED_DIMS if name not in self.dims]
        if missing:
            findings.append(f"WorkloadStrata: stratification dims not covered: {', '.join(missing)}")
        if not self.holdout_id:
            findings.append("WorkloadStrata: a holdout is required and must be isolated before exploration")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {"strata": list(self.strata), "holdout_id": self.holdout_id, "dims": list(self.dims)}


@dataclass(frozen=True)
class ProfilePlan:
    """Step 24: at least two layers, with the join keys that make them joinable."""

    layers: Tuple[str, ...] = ()
    join_keys: Tuple[str, ...] = ()

    REQUIRED_JOIN_KEYS: Tuple[str, ...] = ("request_id", "batch_id", "op_id")

    def problems(self) -> List[str]:
        findings: List[str] = list(check_profile_layering(self.layers))
        missing = [name for name in self.REQUIRED_JOIN_KEYS if name not in self.join_keys]
        if missing:
            findings.append(
                f"ProfilePlan: join keys not declared: {', '.join(missing)}（多层数据无法 join 就证明不了因果链）"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {"layers": list(self.layers), "join_keys": list(self.join_keys)}


@dataclass(frozen=True)
class TimingBoundaries:
    """Step 25: which boundaries are included, for baseline and candidate alike."""

    included: Tuple[str, ...] = ()
    excluded: Tuple[str, ...] = ()
    exclusion_reason: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        unknown = [name for name in (*self.included, *self.excluded) if name not in TIMING_BOUNDARIES]
        if unknown:
            findings.append(f"TimingBoundaries: unknown boundaries {', '.join(unknown)}")
        if not self.included:
            findings.append("TimingBoundaries: no boundary included")
        if self.excluded and not self.exclusion_reason:
            findings.append(
                "TimingBoundaries: an exclusion must be justified on both paths "
                "(把预处理排除只对 candidate 有利)"
            )
        if self.excluded and self.included == ():
            findings.append("TimingBoundaries: nothing is measured")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "included": list(self.included),
            "excluded": list(self.excluded),
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class CostDenominator:
    """Step 26: the denominator, and the resources that must be counted with it."""

    denominators: Tuple[str, ...] = ()
    resources: Mapping[str, Any] = field(default_factory=dict)

    REQUIRED_RESOURCES: Tuple[str, ...] = (
        "device_count",
        "device_memory",
        "energy_j",
        "cpu_time_s",
        "network_bytes",
        "storage_bytes",
        "engineering_dependencies",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.denominators:
            findings.append("CostDenominator: at least one denominator is required")
        unknown = [name for name in self.denominators if name not in COST_DENOMINATORS]
        if unknown:
            findings.append(f"CostDenominator: unknown denominators {', '.join(unknown)}")
        missing = [name for name in self.REQUIRED_RESOURCES if name not in self.resources]
        if missing:
            findings.append(
                f"CostDenominator: resources not declared: {', '.join(missing)}"
                "（以更多设备换吞吐后称算法加速）"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "denominators": list(self.denominators),
            "resources": {key: self.resources[key] for key in sorted(self.resources)},
        }


@dataclass(frozen=True)
class StatisticsPlan:
    """Steps 27–28: unit, repeats, analysis and blocking, frozen before the run."""

    unit: str
    minimum_repeats: int
    analysis: str
    randomisation: str
    blocking: Tuple[str, ...] = ()

    UNITS: Tuple[str, ...] = tuple(rec.EXPERIMENT_UNITS)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.unit not in self.UNITS:
            findings.append(f"StatisticsPlan: unit {self.unit!r} must be one of {', '.join(self.UNITS)}")
        if self.minimum_repeats < 1:
            findings.append("StatisticsPlan: at least one independent repetition is required")
        if self.unit in ("runtime_service", "edge_energy", "training") and self.minimum_repeats < 3:
            findings.append(
                f"StatisticsPlan: unit {self.unit!r} needs >= 3 independent repeats "
                "(manual §5.4: 至少 3 次独立进程运行)"
            )
        if not self.analysis:
            findings.append("StatisticsPlan: the analysis method is required (bootstrap/paired)")
        if not self.randomisation:
            findings.append(
                "StatisticsPlan: randomisation is required (candidate 不应总在更冷/更空闲的环境运行)"
            )
        if not self.blocking:
            findings.append("StatisticsPlan: blocking dimensions are required (节点/时段/温度/cache)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "minimum_repeats": self.minimum_repeats,
            "analysis": self.analysis,
            "randomisation": self.randomisation,
            "blocking": list(self.blocking),
        }


@dataclass(frozen=True)
class StopRules:
    """Step 31: the conditions that abort the run instead of widening the blast radius."""

    rules: Tuple[str, ...] = ()

    REQUIRED: Tuple[str, ...] = (
        "oom",
        "quality_fail",
        "slo_error_budget",
        "thermal",
        "data_leak",
        "numerical_anomaly",
        "resource_cap",
    )

    def problems(self) -> List[str]:
        present = {rule.split(":", 1)[0].strip().lower() for rule in self.rules}
        missing = [name for name in self.REQUIRED if name not in present]
        findings: List[str] = []
        if missing:
            findings.append(f"StopRules: missing stop conditions: {', '.join(missing)}")
        if not self.rules:
            findings.append("StopRules: no stop rule declared")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {"rules": list(self.rules)}


@dataclass(frozen=True)
class InvalidityRules:
    """Step 32: missing is not zero; each category keeps its own meaning."""

    categories: Mapping[str, str] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        missing = [name for name in INVALIDITY_CATEGORIES if name not in self.categories]
        if missing:
            findings.append(f"InvalidityRules: categories not defined: {', '.join(missing)}")
        for name, action in sorted(self.categories.items()):
            if name not in INVALIDITY_CATEGORIES:
                findings.append(f"InvalidityRules: unknown category {name!r}")
            if not action:
                findings.append(f"InvalidityRules: category {name!r} has no action")
            if name in ("tool_failure", "fallback") and action.lower().startswith(("drop", "delete", "remove")):
                findings.append(
                    f"InvalidityRules: {name!r} must be recorded rather than deleted "
                    "(缺失填零或只删除慢失败)"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {"categories": {key: self.categories[key] for key in sorted(self.categories)}}


@dataclass(frozen=True)
class PredictionModel:
    """Step 33: the theory written *before* the run, so explanation is not post hoc."""

    memory_prediction: str
    flops_prediction: str
    bytes_prediction: str
    communication_prediction: str
    serial_steps_prediction: str

    REQUIRED: Tuple[str, ...] = (
        "memory_prediction",
        "flops_prediction",
        "bytes_prediction",
        "communication_prediction",
        "serial_steps_prediction",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in self.REQUIRED:
            value = getattr(self, name)
            if not value:
                findings.append(f"PredictionModel: {name!r} is empty")
            elif not any(char.isdigit() for char in value):
                findings.append(
                    f"PredictionModel: {name!r} contains no quantity; a prediction without a number "
                    "cannot produce a residual"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.REQUIRED}


def check_attribution(
    *, primary_effect: Mapping[str, Any], mediator_evidence: Sequence[str]
) -> Dict[str, Any]:
    """Step 34: a primary effect without a mediator profile is only an association.

    ``E14-05`` §10: 只有端到端数字无机制 is a downgrade, not a failure — the result
    is still reported, but its claim level is reduced.
    """
    has_effect = any(value for value in primary_effect.values())
    if not has_effect:
        return {"status": rec.STATUS_NOT_RUN, "reason": "no primary effect recorded", "claim_level": "design"}
    if not mediator_evidence:
        return {
            "status": rec.STATUS_NOT_RUN,
            "claim_level": "association_only",
            "problem": "primary effect has no mediator profile; the conclusion is downgraded to association",
        }
    return {"status": rec.STATUS_NOT_RUN, "claim_level": "mechanism_supported", "mediators": list(mediator_evidence)}


@dataclass(frozen=True)
class ComplexityScore:
    """Step 35: what the candidate costs to maintain, not only what it buys."""

    added_loc: int = 0
    new_dependencies: Tuple[str, ...] = ()
    compile_time_s: float = 0.0
    binary_bytes: int = 0
    platform_branches: int = 0
    new_tests: int = 0
    failure_surface: Tuple[str, ...] = ()
    upgrade_cost: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.added_loc < 0 or self.binary_bytes < 0 or self.platform_branches < 0:
            findings.append("ComplexityScore: negative counts are not meaningful")
        if not self.upgrade_cost:
            findings.append("ComplexityScore: upgrade cost must be stated (升级成本也是采用代价)")
        if self.platform_branches and not self.new_tests:
            findings.append("ComplexityScore: platform branches without tests increase the regression surface")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "added_loc": self.added_loc,
            "new_dependencies": list(self.new_dependencies),
            "compile_time_s": self.compile_time_s,
            "binary_bytes": self.binary_bytes,
            "platform_branches": self.platform_branches,
            "new_tests": self.new_tests,
            "failure_surface": list(self.failure_surface),
            "upgrade_cost": self.upgrade_cost,
        }


# ── adoption, selection, review, issuance (steps 36–40) ────────────────────


@dataclass(frozen=True)
class AdoptionRules:
    """Step 36: the decision is *predefined* for every outcome shape."""

    rule_id: str
    rules: Mapping[str, str] = field(default_factory=dict)

    #: Outcome shapes that must each map to a decision before the run.
    REQUIRED_OUTCOMES: Tuple[str, ...] = (
        "quality_fail",
        "no_benefit",
        "benefit_specific_workload_only",
        "not_portable",
        "engineering_cost_too_high",
        "benefit_confirmed",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.rule_id:
            findings.append("AdoptionRules: rule_id is required")
        missing = [name for name in self.REQUIRED_OUTCOMES if name not in self.rules]
        if missing:
            findings.append(f"AdoptionRules: outcomes without a predefined decision: {', '.join(missing)}")
        for outcome, decision in sorted(self.rules.items()):
            if decision not in rec.ADOPTION_DECISIONS:
                findings.append(
                    f"AdoptionRules: outcome {outcome!r} maps to {decision!r}, which is not an adoption decision"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {"rule_id": self.rule_id, "rules": {key: self.rules[key] for key in sorted(self.rules)}}


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's assessment against the eight selection criteria."""

    branch: str
    states: Mapping[str, str] = field(default_factory=dict)
    notes: Mapping[str, str] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.branch not in CANDIDATE_BRANCHES:
            findings.append(f"CandidateScore: unknown branch {self.branch!r}")
        for criterion in SELECTION_CRITERIA:
            state = str(self.states.get(criterion, "")).upper()
            if not state:
                findings.append(f"CandidateScore {self.branch}: criterion {criterion!r} not assessed")
            elif state not in EVIDENCE_STATES:
                findings.append(
                    f"CandidateScore {self.branch}: criterion {criterion!r} state {state!r} must be "
                    f"{'/'.join(EVIDENCE_STATES)}"
                )
        return findings

    def blockers(self) -> List[str]:
        return [name for name, state in self.states.items() if str(state).upper() == "UNKNOWN"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "states": {key: self.states[key] for key in sorted(self.states)},
            "notes": {key: self.notes[key] for key in sorted(self.notes)},
            "unknown_criteria": self.blockers(),
        }


def select_single_branch(
    scores: Sequence[CandidateScore], *, reviewer: str, evidence: Sequence[str], rationale: str
) -> Dict[str, Any]:
    """Step 37: exactly one primary branch, with the ADR's reviewer and evidence.

    A branch whose hard prerequisite is ``UNKNOWN`` cannot be selected: an
    unverified prerequisite is not a lower priority, it is a missing gate
    (``E14-05`` step 15 / ``details/S14/README.md`` §21).
    """
    problems: List[str] = []
    for score in scores:
        problems.extend(score.problems())
    blockers: Dict[str, List[str]] = {score.branch: score.blockers() for score in scores}
    eligible = [score.branch for score in scores if not blockers[score.branch]]
    if not rationale:
        problems.append("the ADR needs a rationale; a preference is not a decision")
    if not reviewer:
        problems.append("the ADR must name its reviewer")
    if not evidence:
        problems.append("the ADR must cite the evidence it was based on")
    return {
        "eligible": eligible,
        "ineligible": {branch: names for branch, names in sorted(blockers.items()) if names},
        "selectable": len(eligible) == 1 or len(eligible) > 1,
        "ambiguous_selection": len(eligible) > 1,
        "problems": problems,
        "scores": [score.as_dict() for score in scores],
        "note": "只允许一个主分支；eligible>1 时由 ADR 记录否决其他分支的理由",
    }


def lock_unselected_branches(
    selected: str, candidates: Sequence[str], *, reopen_conditions: Mapping[str, str], forbiddens: Mapping[str, Any]
) -> Tuple[Dict[str, Any], ...]:
    """Step 38: the unselected branches become ``N/A_BY_ADR`` with reopen conditions.

    Locking them is what stops a design document from being read as an
    implemented capability (``details/S14/README.md`` §3 forbidden practice 1).
    """
    if selected not in candidates:
        raise ConfigError(f"selected branch {selected!r} is not among the candidates")
    locked: List[Dict[str, Any]] = []
    for branch in candidates:
        if branch == selected:
            continue
        decision = AdoptionDecision(
            decision_id=f"lock::{branch}",
            experiment_id=branch,
            decision=rec.N_A_BY_ADR,
            forbidden_claims=tuple(forbiddens.get(branch, BRANCH_FORBIDDEN_CLAIMS.get(branch, ()))),
            maturity=rec.MATURITY_DESIGN_ONLY,
            reopened_if=(reopen_conditions.get(branch, "未声明重开条件"),),
        )
        locked.append({"branch": branch, "decision": decision.as_dict(), "problems": decision.validate()})
    return tuple(locked)


def protocol_review(*, reviewer: str, ambiguities: Sequence[str], manipulative_metrics: Sequence[str],
                    missing_failure_paths: Sequence[str], revisions: Sequence[str]) -> Dict[str, Any]:
    """Step 39: an independent read of the protocol *before* it is frozen.

    The review is only useful if it can be shown to have found something, so the
    three categories are recorded separately — an empty review is reported as
    such rather than treated as a clean bill of health.
    """
    problems: List[str] = []
    if not reviewer:
        problems.append("protocol review must name an independent reviewer")
    if not revisions and (ambiguities or manipulative_metrics or missing_failure_paths):
        problems.append("the review found issues but produced no revision")
    return {
        "reviewer": reviewer,
        "ambiguities": list(ambiguities),
        "manipulative_metrics": list(manipulative_metrics),
        "missing_failure_paths": list(missing_failure_paths),
        "revisions": list(revisions),
        "found_issues": bool(ambiguities or manipulative_metrics or missing_failure_paths),
        "problems": problems,
    }


def issue_contract(contract: FrontierStudyContract) -> Dict[str, Any]:
    """Step 40: hash the frozen protocol; the returned hash is what a run cites.

    The hash is computed over the contract *without* the hash field, so it is a
    stable function of the content and a later edit cannot preserve it.
    """
    problems = contract.validate()
    payload = contract.as_dict(include_hash=False)
    return {
        "contract": payload,
        "contract_sha256": canonical_digest(payload),
        "frozen": not problems,
        "problems": problems,
        "status": rec.STATUS_BLOCKED_PREREQUISITE if problems else rec.STATUS_NOT_RUN,
    }


@dataclass
class FrontierSelectionVerdict:
    """The E14-05 ruling: methodology only, never a speedup."""

    verdict_id: str
    selected_branch: str
    status: str = rec.STATUS_NOT_RUN
    contract_sha256: str = ""
    review: Mapping[str, Any] = field(default_factory=dict)
    locked_branches: Tuple[Mapping[str, Any], ...] = ()
    problems: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.e14-05.verdict.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        if self.selected_branch not in CANDIDATE_BRANCHES:
            findings.append(f"FrontierSelectionVerdict: selected_branch {self.selected_branch!r} is invalid")
        if self.status == rec.STATUS_PASS:
            if not self.contract_sha256:
                findings.append("FrontierSelectionVerdict: PASS requires the contract hash")
            if not self.review.get("reviewer"):
                findings.append("FrontierSelectionVerdict: PASS requires a completed protocol review")
            if not self.locked_branches:
                findings.append(
                    "FrontierSelectionVerdict: PASS requires the unselected branches to be locked "
                    "（未选方向不能产生已支持 claim）"
                )
            if self.problems:
                findings.append("FrontierSelectionVerdict: PASS with unresolved problems")
        if "speedup" in str(self.as_dict()).lower():
            findings.append(
                "FrontierSelectionVerdict: this gate generates no algorithm speedup; remove any such field"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict_id": self.verdict_id,
            "selected_branch": self.selected_branch,
            "status": self.status,
            "contract_sha256": self.contract_sha256,
            "review": dict(self.review),
            "locked_branches": [dict(item) for item in self.locked_branches],
            "problems": list(self.problems),
        }


# ── result accessors ───────────────────────────────────────────────────────

def contract_hash(issued: Mapping[str, Any]) -> str:
    """Step 40: the hash a run must cite (a modified protocol cannot keep it)."""
    return str(issued.get("contract_sha256", ""))


def gate_blockers(result: Mapping[str, Any]) -> Tuple[str, ...]:
    """Step 15: the prerequisites that make a branch ``BLOCKED`` rather than lower priority."""
    return tuple(result.get("blocking", ())) + tuple(result.get("unknown", ()))


def uninstrumented_gates(gates: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Step 17: gates whose actual path is not observed cannot prove a mechanism."""
    return tuple(str(gate.get("gate_id", "")) for gate in gates if not gate.get("instrumented"))


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-05 interfaces (labelled smoke, not an experiment)."""
    registry = LiteratureRegistry(
        entries=tuple(
            LiteratureEntry(
                key=branch, kind="paper", title=f"source for {branch}", venue="venue", year=2024,
                version="v1", claim="records an effect under its own conditions",
            )
            for branch in CANDIDATE_BRANCHES
        )
    )
    estimand = CandidateEstimand(
        branch="E14-F3",
        text="在固定 target model/quality gate/arrival trace 下，对 SLO-qualified output-token goodput 的因果变化",
        baseline_id="full-kv",
        primary_metric="goodput",
        mediators=BRANCH_MEDIATORS["E14-F3"],
        unit="request_episode",
    )
    gate = HardPrerequisiteGate(items={"device_capability": "BLOCKED", "quality_oracle": "OK"})
    pair = BaselinePair(
        baseline_id="full-kv", candidate_id="kv-quant", intended_difference="KV 精度",
        shared_fields={name: "frozen" for name in BaselinePair.FROZEN},
    )
    rules = AdoptionRules(
        rule_id="r1",
        rules={
            "quality_fail": rec.REJECT_QUALITY,
            "no_benefit": rec.REJECT_NO_BENEFIT,
            "benefit_specific_workload_only": rec.ADOPT_EXPERIMENTAL,
            "not_portable": rec.REJECT_PORTABILITY,
            "engineering_cost_too_high": rec.REJECT_COMPLEXITY,
            "benefit_confirmed": rec.RESEARCH_ONLY,
        },
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "candidates": len(CANDIDATE_BRANCHES),
        "registry_problems": registry.problems(),
        "estimand_problems": estimand.problems(),
        "blocked_prerequisite_status": gate.evaluate("E14-F3")["status"],
        "baseline_pair_problems": pair.problems(),
        "adoption_rule_problems": rules.problems(),
        "criteria": len(SELECTION_CRITERIA),
        "timing_boundaries": len(TIMING_BOUNDARIES),
    }


# ── protocol step table (40 steps of details/S14/E14-05) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 S14 公共前置状态", ("records:EXPERIMENT_DEPENDENCIES", "records:EXPERIMENT_TABLE")),
    (2, "冻结目标岗位与代表能力", ("frontier:SELECTION_CRITERIA", "frontier:CandidateScore")),
    (3, "定义选择约束", ("campaign:RunBudget", "frontier:CandidateScore.states")),
    (4, "建立候选文献清单", ("frontier:LiteratureEntry", "frontier:LiteratureRegistry")),
    (5, "提取论文实验条件", ("frontier:PaperConditionMatrix", "frontier:PAPER_CONDITION_FIELDS")),
    (6, "建立 HQSB 对应矩阵", ("frontier:hqsb_mapping_matrix", "frontier:HQSB_CONTRACT_FIELDS")),
    (7, "盘点已有实现资产", ("records:TABLE_SCHEMAS", "telemetry:PROJECTORS")),
    (8, "运行候选 capability probes", ("dependencies:probe_extra", "records:CAPABILITY_STATES")),
    (9, "定义每个候选的最小独立改动", ("frontier:CandidateScore.notes", "frontier:ComplexityScore")),
    (10, "定义每个候选的科学缺口", ("frontier:CandidateEstimand", "frontier:BRANCH_MEDIATORS")),
    (11, "为 F1 写候选 estimand", ("frontier:CandidateEstimand", "frontier:BRANCH_MEDIATORS")),
    (12, "为 F2 写候选 estimand", ("frontier:CandidateEstimand", "frontier:BRANCH_FORBIDDEN_CLAIMS")),
    (13, "为 F3 写候选 estimand", ("frontier:CandidateEstimand", "frontier:CandidateEstimand.problems")),
    (14, "为 F4 写候选 estimand", ("frontier:CandidateEstimand.mediators", "frontier:BRANCH_MEDIATORS")),
    (15, "定义硬前置门", ("frontier:HardPrerequisiteGate", "frontier:gate_blockers",
                          "records:STATUS_BLOCKED_PREREQUISITE")),
    (16, "定义质量门", ("frontier:QualityGate", "contracts:check_quality_before_performance")),
    (17, "定义 actual-path 门", ("frontier:ActualPathGate", "frontier:uninstrumented_gates")),
    (18, "定义 primary hypothesis", ("frontier:PrimaryHypothesis", "frontier:check_single_primary")),
    (19, "定义 secondary hypotheses", ("frontier:check_single_primary", "records:STATUS_PASS_NEGATIVE")),
    (20, "定义 baseline 和唯一差异", ("frontier:BaselinePair", "frontier:BaselinePair.FROZEN")),
    (21, "定义负对照", ("frontier:NegativeControls", "contracts:check_negative_control_coverage")),
    (22, "定义消融矩阵", ("frontier:AblationMatrix", "frontier:AblationMatrix.max_combinations")),
    (23, "定义 workload strata", ("frontier:WorkloadStrata", "frontier:WorkloadStrata.REQUIRED_DIMS")),
    (24, "定义 profile 层和关联键", ("frontier:ProfilePlan", "contracts:check_profile_layering")),
    (25, "定义计时边界", ("frontier:TimingBoundaries", "frontier:TIMING_BOUNDARIES")),
    (26, "定义资源和成本分母", ("frontier:CostDenominator", "frontier:COST_DENOMINATORS")),
    (27, "定义实验单位和重复", ("frontier:StatisticsPlan", "records:EXPERIMENT_UNITS")),
    (28, "定义随机化与 blocking", ("frontier:StatisticsPlan.randomisation", "frontier:StatisticsPlan.blocking")),
    (29, "定义探索预算", ("campaign:RunBudget.stop_rules", "frontier:AblationMatrix.max_combinations")),
    (30, "定义 holdout confirmation", ("frontier:WorkloadStrata.holdout_id", "frontier:BaselinePair")),
    (31, "定义停止/中止规则", ("frontier:StopRules", "frontier:StopRules.REQUIRED")),
    (32, "定义缺失与 invalid 规则", ("frontier:InvalidityRules", "frontier:INVALIDITY_CATEGORIES")),
    (33, "定义预测模型", ("frontier:PredictionModel", "frontier:PredictionModel.REQUIRED")),
    (34, "定义结果归因要求", ("frontier:check_attribution", "contracts:check_actual_path_recorded")),
    (35, "定义复杂性与维护评分", ("frontier:ComplexityScore", "contracts:AdoptionDecision.limitations")),
    (36, "预注册 AdoptionDecision", ("frontier:AdoptionRules", "records:ADOPTION_DECISIONS")),
    (37, "选择唯一主 F 分支", ("frontier:select_single_branch", "frontier:CandidateScore")),
    (38, "锁定未选分支状态", ("frontier:lock_unselected_branches", "records:N_A_BY_ADR")),
    (39, "执行盲化协议复核", ("frontier:protocol_review", "frontier:FrontierSelectionVerdict")),
    (40, "签发 FrontierStudyContract", ("frontier:issue_contract", "frontier:contract_hash",
                                         "contracts:supersede_contract")),
)
