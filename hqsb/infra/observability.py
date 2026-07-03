"""E13-08: request→queue→runtime→accelerator→system observability, alerts and RCA.

Implements ``details/S13/E13-08_*.md`` as data:

* the layer taxonomy and the versioned semantic conventions (name/unit/attributes,
  required/optional/forbidden) so no component invents its own request id;
* the signal split of §2 (metrics = aggregation; traces = per-request causality;
  logs/events = discrete state; profiles = on-demand and expensive);
* the cardinality policy of §6 — request/tenant/prompt identifiers never become
  Prometheus labels, and the check is enforced by :func:`validate_metric_labels`;
* the SLI/diagnostic split of §5 and the client/server/core boundary naming rule;
* histogram bucket validation (a P99 that all lands in ``+Inf`` is not a P99);
* metric↔raw reconciliation, clock alignment, instrumentation overhead A/B,
  sampling coverage, telemetry-missing detection ("missing is not 0");
* alert rules with user impact/runbook/owner and a threshold source that
  distinguishes a *policy default* from a measured value;
* :class:`RCARecord` for blind root-cause analysis with confidence and
  alternatives, plus the redaction scan.

Nothing here emits, scrapes or queries telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import contracts as ct
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-08"
TITLE = "Request→Queue→Runtime→Accelerator→System 可观测性、告警与 RCA"
CLAIM_BOUNDARY = (
    "本实验通过证明系统可被观测与定位；不证明所有故障已恢复或发布策略正确"
    "（由 E13-09/E13-10 验证行动闭环）。"
)

SCHEMA_VERSION = "1.0.0"

#: Layers a symptom must be attributable to (§9 step 1).
LAYERS: Tuple[str, ...] = rec.OBSERVABILITY_LAYERS
SIGNAL_CLASSES: Tuple[str, ...] = rec.SIGNAL_CLASSES

#: Boundary naming rule: client/server/core latencies get different metric names.
LATENCY_BOUNDARIES: Tuple[str, ...] = ("client", "server", "core")

#: Attribute requirement levels for semantic conventions.
ATTRIBUTE_LEVELS: Tuple[str, ...] = ("required", "optional", "forbidden")

#: Sampling policies (§9 step 4).
SAMPLING_KINDS: Tuple[str, ...] = ("head", "tail", "error_biased", "slo_biased", "rare_event", "none")

#: Cases that must always be kept regardless of sampling rate.
ALWAYS_KEEP_CASES: Tuple[str, ...] = (
    "error",
    "forced_termination",
    "canary_candidate",
    "fault_injected",
    "slo_violation",
)

#: Telemetry components whose failure must be detectable (§9 step 32).
TELEMETRY_COMPONENTS: Tuple[str, ...] = ("exporter", "collector", "backend", "scrape", "log_agent", "trace_agent")

#: Base units (Prometheus practice): seconds, bytes, joules, ratio, counts.
BASE_UNITS: Tuple[str, ...] = (
    "seconds",
    "bytes",
    "joules",
    "watts",
    "ratio",
    "replicas",
    "requests",
    "tokens",
    "requests/second",
    "tokens/second",
    "celsius",
    "hertz",
    "count",
)

#: RCA investigation steps that must be recorded (§11).
RCA_REQUIRED_FIELDS: Tuple[str, ...] = (
    "rca_id", "case_id", "investigator_id", "blind_status", "symptom", "started_at", "ended_at",
    "identified_layer", "root_cause", "scope", "confidence", "ground_truth_match",
)


@dataclass
class SemanticConvention:
    """One versioned signal definition (§9 step 2)."""

    name: str
    signal_class: str = "metrics"
    unit: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)
    required_attributes: Tuple[str, ...] = ()
    version: str = ""
    boundary: str = ""
    description: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.name:
            problems.append("a convention needs a name")
        if self.signal_class not in SIGNAL_CLASSES:
            problems.append(f"unknown signal class {self.signal_class!r}")
        if not self.version:
            problems.append(f"{self.name}: conventions are versioned, the version is required")
        if self.signal_class == "metrics":
            if not self.unit:
                problems.append(f"{self.name}: a metric needs a unit (seconds/bytes/joules/ratio)")
            elif self.unit not in BASE_UNITS:
                problems.append(f"{self.name}: unit {self.unit!r} is not a base unit")
            for label, level in self.attributes.items():
                # An attribute declared ``forbidden`` is the *documented* prohibition
                # (e.g. ``request_id: forbidden``); it is a required or optional
                # attribute that would actually index a metric that is the problem.
                if level == "forbidden":
                    continue
                if label in rec.FORBIDDEN_METRIC_LABELS:
                    problems.append(
                        f"{self.name}: label {label!r} is unbounded and must not index a metric "
                        f"(declare it 'forbidden' or move it to a trace/log attribute)"
                    )
            if self.boundary and self.boundary not in LATENCY_BOUNDARIES:
                problems.append(f"{self.name}: unknown latency boundary {self.boundary!r}")
        for label, level in self.attributes.items():
            if level not in ATTRIBUTE_LEVELS:
                problems.append(f"{self.name}: attribute {label!r} has unknown level {level!r}")
        for attribute in self.required_attributes:
            if attribute not in self.attributes:
                problems.append(f"{self.name}: required attribute {attribute!r} is not declared")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signal_class": self.signal_class,
            "unit": self.unit,
            "attributes": dict(sorted(self.attributes.items())),
            "required": self.required_attributes,
            "version": self.version,
        }


def validate_convention_set(conventions: Sequence[SemanticConvention]) -> Dict[str, Any]:
    """Steps 1–3: one versioned vocabulary, no per-component request id."""
    problems: List[str] = []
    names: List[str] = []
    for convention in conventions:
        problems.extend(convention.validate())
        names.append(convention.name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate convention names: {duplicates}")
    latency_boundaries = {
        convention.boundary for convention in conventions if convention.boundary
    }
    for boundary in LATENCY_BOUNDARIES:
        if boundary not in latency_boundaries:
            problems.append(
                f"no latency convention declares the {boundary} boundary: mixing boundaries in one "
                "latency metric hides the difference between client and core service time"
            )
    return {
        "conventions": len(conventions),
        "signal_classes": sorted({convention.signal_class for convention in conventions}),
        "problems": problems,
        "ok": not problems,
    }


def load_semantic_conventions(path: str) -> Dict[str, Any]:
    """Parse ``infra/observability/semantic_conventions.yaml`` into the vocabulary.

    The assets under ``infra/`` are templates; loading them here keeps the *asset*
    and the *validator* in one place so a hand-edited YAML cannot silently disagree
    with the code (a forbidden label or a missing boundary fails the load).
    """
    import os as _os

    import yaml as _yaml

    if not _os.path.isfile(path):
        raise ConfigError(f"semantic convention file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        document = _yaml.safe_load(handle)
    if not isinstance(document, dict) or "conventions" not in document:
        raise ConfigError(f"{path}: document must be a mapping with a 'conventions' list")
    conventions: List[SemanticConvention] = []
    problems: List[str] = []
    for index, entry in enumerate(document["conventions"]):
        if not isinstance(entry, dict):
            problems.append(f"conventions[{index}]: must be a mapping")
            continue
        convention = SemanticConvention(
            name=str(entry.get("name", "")),
            signal_class=str(entry.get("signal_class", "metrics")),
            unit=str(entry.get("unit", "")),
            attributes={str(key): str(value) for key, value in (entry.get("attributes") or {}).items()},
            required_attributes=tuple(entry.get("required_attributes") or ()),
            version=str(document.get("version", "")),
            boundary=str(entry.get("boundary", "")),
            description=str(entry.get("description", "")),
        )
        conventions.append(convention)
    set_report = validate_convention_set(conventions)
    return {
        "path": path,
        "version": str(document.get("version", "")),
        "status": str(document.get("status", "")),
        "conventions": conventions,
        "problems": list(set_report["problems"]) + problems,
        "ok": set_report["ok"] and not problems,
    }


def load_alert_rules(path: str) -> Dict[str, Any]:
    """Parse ``infra/observability/alerts.yaml`` and validate every rule.

    A rule that drops the user impact, the query, the owner, the runbook or the
    threshold provenance is rejected: an alert without a runbook is a page nobody
    can act on, and an unlabelled threshold looks like a measured value.
    """
    import os as _os

    import yaml as _yaml

    if not _os.path.isfile(path):
        raise ConfigError(f"alert rules file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        document = _yaml.safe_load(handle)
    if not isinstance(document, dict) or "rules" not in document:
        raise ConfigError(f"{path}: document must be a mapping with a 'rules' list")
    rules: List[AlertRule] = []
    problems: List[str] = []
    for index, entry in enumerate(document["rules"]):
        if not isinstance(entry, dict):
            problems.append(f"rules[{index}]: must be a mapping")
            continue
        rule = AlertRule(
            alert_id=str(entry.get("alert_id", "")),
            name=str(entry.get("name", "")),
            user_impact=str(entry.get("user_impact", "")),
            query=str(entry.get("query", "")),
            for_duration=str(entry.get("for_duration", "")),
            severity=str(entry.get("severity", "warning")),
            owner=str(entry.get("owner", "")),
            runbook_id=str(entry.get("runbook_id", "")),
            threshold_source=str(entry.get("threshold_source", "")),
            threshold_value=float(entry.get("threshold_value", 0.0)),
            suppression=tuple(entry.get("suppression") or ()),
            auto_action=str(entry.get("auto_action", "")),
        )
        rules.append(rule)
        problems.extend(rule.validate())
    marker = str(document.get("policy_default_marker", ""))
    if marker != rec.POLICY_DEFAULT_MARKER:
        problems.append(
            f"policy_default_marker must be {rec.POLICY_DEFAULT_MARKER!r} so a template threshold can "
            "never be mistaken for a measurement"
        )
    return {
        "path": path,
        "version": str(document.get("version", "")),
        "status": str(document.get("status", "")),
        "rules": rules,
        "problems": problems,
        "ok": not problems,
    }


def validate_metric_labels(labels: Mapping[str, Any]) -> Dict[str, Any]:
    """§6/§23 V12: enforce the cardinality policy on one label set."""
    result = ct.validate_metric_labels(labels)
    return {"labels": sorted(labels), "ok": result.ok, "reason_codes": result.reason_codes,
            "findings": [finding.as_dict() for finding in result.findings]}


def classify_cardinality(
    *, metric_name: str, series_count: int, growth_per_hour: float, policy_limit: int
) -> Dict[str, Any]:
    """Step 18: cardinality is measured, and the policy action is explicit."""
    over = series_count > policy_limit
    return {
        "metric_name": metric_name,
        "series_count": series_count,
        "growth_rate_per_hour": growth_per_hour,
        "policy_limit": policy_limit,
        "policy_action": "DROP_LABEL_OR_ROUTE" if over else "KEEP",
        "ok": not over,
        "reason": (
            ""
            if not over
            else f"{metric_name}: {series_count} series exceed the policy limit {policy_limit} "
            "(each label set creates its own time series)"
        ),
    }


@dataclass
class SamplingPolicy:
    """Step 4: what is sampled, what is always kept, and the storage budget."""

    policy_id: str
    head_rate: float = 0.0
    tail_enabled: bool = False
    per_token_aggregation: bool = False
    keep_cases: Tuple[str, ...] = ALWAYS_KEEP_CASES
    budget_events_per_s: float = 0.0
    trace_loss_budget: float = 0.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.policy_id:
            problems.append("sampling policy needs an id")
        missing = sorted(set(ALWAYS_KEEP_CASES) - set(self.keep_cases))
        if missing:
            problems.append(
                f"sampling would drop {missing}: errors, forced terminations, canary/fault cases and SLO "
                "violations must always be retained"
            )
        if not self.tail_enabled and self.head_rate < 1.0:
            problems.append(
                "head-only sampling below 100% loses exactly the tail that RCA needs; enable tail sampling"
            )
        if self.budget_events_per_s <= 0:
            problems.append("an event budget is required (telemetry is not free)")
        if self.trace_loss_budget < 0:
            problems.append("trace loss budget must not be negative")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "head_rate": self.head_rate,
            "tail_enabled": self.tail_enabled,
            "per_token_aggregation": self.per_token_aggregation,
            "keep_cases": list(self.keep_cases),
            "budget_events_per_s": self.budget_events_per_s,
            "trace_loss_budget": self.trace_loss_budget,
        }


def sampling_coverage(
    *, injected: Sequence[Mapping[str, Any]], policy: SamplingPolicy
) -> Dict[str, Any]:
    """Step 20: injected errors/tail cases must survive the sampler."""
    problems: List[str] = []
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for case in injected:
        kind = str(case.get("kind", ""))
        retained = bool(case.get("retained", False))
        entry = {"case_id": case.get("case_id", ""), "kind": kind, "retained": retained,
                 "parent_complete": bool(case.get("parent_complete", True))}
        (kept if retained else dropped).append(entry)
        if kind in policy.keep_cases and not retained:
            problems.append(f"{kind} case {entry['case_id']} was sampled away although it must always be kept")
        if retained and not entry["parent_complete"]:
            problems.append(f"{entry['case_id']}: retained span has an incomplete parent chain")
    return {"kept": kept, "dropped": dropped, "problems": problems, "ok": not problems}


# ── SLI / diagnostics ────────────────────────────────────────────────────


@dataclass
class SLIDefinition:
    """§5: a user-facing SLI (never replaced by a system diagnostic)."""

    sli_id: str
    metric_name: str
    unit: str = ""
    boundary: str = "server"
    estimator: str = "histogram"
    slo_target: float = 0.0
    direction: str = ""
    goodput_definition: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("sli_id", "metric_name", "direction", "goodput_definition"):
            if not getattr(self, name):
                problems.append(f"SLI requires {name!r}")
        if self.unit and self.unit not in BASE_UNITS:
            problems.append(f"SLI unit {self.unit!r} is not a base unit")
        if self.boundary not in LATENCY_BOUNDARIES:
            problems.append(f"unknown boundary {self.boundary!r}")
        if self.direction not in ("higher_is_better", "lower_is_better"):
            problems.append(f"unknown direction {self.direction!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sli_id": self.sli_id,
            "metric_name": self.metric_name,
            "unit": self.unit,
            "boundary": self.boundary,
            "estimator": self.estimator,
            "slo_target": self.slo_target,
            "direction": self.direction,
        }


def validate_histogram_buckets(
    *, metric_name: str, buckets: Sequence[float], observed_p99: Optional[float] = None
) -> Dict[str, Any]:
    """Step 17: buckets must cover the SLO range, or the P99 is meaningless."""
    problems: List[str] = []
    if len(buckets) < 5:
        problems.append(f"{metric_name}: {len(buckets)} buckets are too few for a tail estimate")
    if any(b <= a for a, b in zip(buckets, buckets[1:])):
        problems.append(f"{metric_name}: bucket bounds must be strictly increasing")
    if buckets and buckets[-1] != float("inf"):
        problems.append(f"{metric_name}: the last bucket must be +Inf (otherwise quantiles are undefined)")
    finite = [bound for bound in buckets if bound != float("inf")]
    if observed_p99 is not None and finite and observed_p99 > finite[-1]:
        problems.append(
            f"{metric_name}: the observed P99 ({observed_p99}) exceeds the largest finite bucket "
            f"({finite[-1]}): the quantile would collapse into +Inf"
        )
    defaultish = buckets and set(buckets) <= {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                                             float("inf")}
    if defaultish:
        problems.append(
            f"{metric_name}: bucket set looks like an unmodified default; buckets must be designed around "
            "the SLO and the observed range"
        )
    return {"metric_name": metric_name, "buckets": list(buckets), "problems": problems, "ok": not problems}


def reconcile_metric_with_raw(
    *, metric_name: str, metric_value: float, raw_rows: Sequence[Mapping[str, Any]], tolerance: float = 0.0
) -> Dict[str, Any]:
    """Step 16: counters/histograms must be recomputable from the raw events."""
    recomputed = float(sum(float(row.get("value", 0.0)) for row in raw_rows))
    delta = metric_value - recomputed
    reset_detected = any(row.get("counter_reset") for row in raw_rows)
    missing_rows = [row for row in raw_rows if row.get("value") in (None, "")]
    problems: List[str] = []
    if abs(delta) > tolerance:
        problems.append(
            f"{metric_name}: exported {metric_value} vs recomputed {recomputed} (delta {delta:g})"
        )
    if reset_detected:
        problems.append(f"{metric_name}: a counter reset was observed and must be handled, not summed blindly")
    if missing_rows:
        problems.append(f"{metric_name}: {len(missing_rows)} raw rows are missing a value (not zero)")
    return {"metric_name": metric_name, "metric_value": metric_value, "recomputed": recomputed,
            "delta": delta, "problems": problems, "ok": not problems}


@dataclass
class ClockAlignment:
    """Step 6: cross-layer timestamps need a measured offset, not an assumption."""

    pair_id: str
    source: str
    target: str
    wall_offset_s: float = 0.0
    monotonic_offset_s: float = 0.0
    error_bound_s: float = 0.0
    correlation_interval_s: float = 0.0
    sync_method: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("pair_id", "source", "target", "sync_method"):
            if not getattr(self, name):
                problems.append(f"clock alignment requires {name!r}")
        if self.error_bound_s < 0:
            problems.append("clock error bound must not be negative")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source": self.source,
            "target": self.target,
            "wall_offset_s": self.wall_offset_s,
            "monotonic_offset_s": self.monotonic_offset_s,
            "error_bound_s": self.error_bound_s,
            "sync_method": self.sync_method,
        }


def clock_verdict(
    *, alignments: Sequence[ClockAlignment], stage_differences_s: Mapping[str, float]
) -> Dict[str, Any]:
    """Steps 6/36/§13: an offset larger than the stage difference forbids a critical-path claim."""
    problems: List[str] = []
    for alignment in alignments:
        problems.extend(alignment.validate())
        for stage, difference in stage_differences_s.items():
            if alignment.error_bound_s >= abs(difference):
                problems.append(
                    f"{alignment.source}->{alignment.target}: clock error {alignment.error_bound_s}s >= "
                    f"stage difference {stage}={difference}s: only coarse correlation is allowed"
                )
    return {"alignments": len(alignments), "problems": problems, "ok": not problems}


# ── alerts and RCA ───────────────────────────────────────────────────────


@dataclass
class AlertRule:
    """Step 22: every alert names user impact, query, owner and runbook."""

    alert_id: str
    name: str = ""
    user_impact: str = ""
    query: str = ""
    for_duration: str = ""
    severity: str = "warning"
    owner: str = ""
    runbook_id: str = ""
    threshold_source: str = ""
    threshold_value: float = 0.0
    suppression: Tuple[str, ...] = ()
    auto_action: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("alert_id", "name", "user_impact", "query", "for_duration", "owner", "runbook_id"):
            if not getattr(self, name):
                problems.append(f"alert requires {name!r}")
        if self.severity not in rec.ALERT_SEVERITIES:
            problems.append(f"unknown severity {self.severity!r}")
        if not self.query or "{" not in self.query and "(" not in self.query:
            problems.append(f"{self.alert_id}: the alert must carry the query it evaluates, not a description")
        if rec.POLICY_DEFAULT_MARKER not in self.threshold_source and not self.threshold_source.startswith(
            "MEASURED"
        ):
            problems.append(
                f"{self.alert_id}: threshold_source must be '{rec.POLICY_DEFAULT_MARKER}' or 'MEASURED:…' "
                "(an unlabelled threshold looks like a measured value)"
            )
        if "utilization" in self.query and "slo" not in self.query.lower() and "user_impact" not in self.query.lower():
            problems.append(
                f"{self.alert_id}: a bare utilization alert pages on a symptom that may be normal saturation"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "user_impact": self.user_impact,
            "query": self.query,
            "for_duration": self.for_duration,
            "severity": self.severity,
            "owner": self.owner,
            "runbook_id": self.runbook_id,
            "threshold_source": self.threshold_source,
        }


def build_alert_rule(
    *,
    alert_id: str,
    name: str,
    user_impact: str,
    query: str,
    for_duration: str,
    severity: str,
    owner: str,
    runbook_id: str,
    threshold_value: float,
    threshold_source: str = rec.POLICY_DEFAULT_MARKER,
    suppression: Sequence[str] = (),
    auto_action: str = "",
) -> AlertRule:
    """Construct + validate an alert rule (fails fast on an unlabelled threshold)."""
    rule = AlertRule(
        alert_id=alert_id, name=name, user_impact=user_impact, query=query, for_duration=for_duration,
        severity=severity, owner=owner, runbook_id=runbook_id, threshold_source=threshold_source,
        threshold_value=threshold_value, suppression=tuple(suppression), auto_action=auto_action,
    )
    problems = rule.validate()
    if problems:
        raise ConfigError("invalid alert rule: " + "; ".join(problems))
    return rule


def alert_effectiveness(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 31: precision/recall/MTTD of the alerts, including planned-rollout suppression."""
    problems: List[str] = []
    true_positive = false_positive = false_negative = suppressed_ok = 0
    mttds: List[float] = []
    for row in rows:
        if row.get("fault_injected") and row.get("detected"):
            true_positive += 1
            if row.get("mttd_s") is not None:
                mttds.append(float(row["mttd_s"]))
        elif row.get("fault_injected") and not row.get("detected"):
            false_negative += 1
            problems.append(f"{row.get('case_id')}: injected fault was not detected")
        elif not row.get("fault_injected") and row.get("alerted") and not row.get("planned_change"):
            false_positive += 1
        elif row.get("planned_change") and row.get("suppressed"):
            suppressed_ok += 1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "suppressed_planned_change": suppressed_ok,
        "precision": precision,
        "recall": recall,
        "mttd_s": (sum(mttds) / len(mttds)) if mttds else 0.0,
        "problems": problems,
        "ok": not problems,
    }


@dataclass
class RCARecord:
    """§11 ``RCARecord``: a blind investigation with confidence and alternatives."""

    rca_id: str
    case_id: str
    investigator_id: str = ""
    blind_status: str = ""
    symptom: str = ""
    request_trace_id: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    queried_signal_artifact_ids: Tuple[str, ...] = ()
    hypotheses_timeline: Tuple[str, ...] = ()
    identified_layer: str = ""
    root_cause: str = ""
    scope: str = ""
    release_id: str = ""
    model_artifact_id: str = ""
    pod_ids: Tuple[str, ...] = ()
    node_ids: Tuple[str, ...] = ()
    device_ids: Tuple[str, ...] = ()
    evidence_refs: Tuple[str, ...] = ()
    confidence: str = ""
    alternatives: Tuple[str, ...] = ()
    recommended_action: str = ""
    runbook_id: str = ""
    ground_truth: str = ""
    correctness: str = ""
    gaps: Tuple[str, ...] = ()

    def mtti_s(self) -> float:
        return max(self.ended_at - self.started_at, 0.0)

    def validate(self, *, ground_truth_hidden: bool = True) -> List[str]:
        problems: List[str] = []
        for name in ("rca_id", "case_id", "investigator_id", "blind_status", "symptom"):
            if not getattr(self, name):
                problems.append(f"RCA record requires {name!r}")
        if self.blind_status not in ("BLIND", "SEMI_BLIND", "OPEN"):
            problems.append(f"unknown blind status {self.blind_status!r}")
        if self.identified_layer and self.identified_layer not in LAYERS:
            problems.append(f"identified layer {self.identified_layer!r} is outside the taxonomy")
        if not self.evidence_refs:
            problems.append("an RCA conclusion must reference the artifacts it used")
        if not self.confidence:
            problems.append("an RCA conclusion must carry a confidence level")
        if not self.alternatives:
            problems.append(
                "an RCA must list the alternative explanations it rejected (a correlation is not a root cause)"
            )
        if ground_truth_hidden and self.blind_status == "BLIND" and self.ground_truth:
            problems.append("ground truth must not be visible to a blind investigator")
        if self.correctness not in ("CORRECT", "PARTIAL", "INCORRECT", "INCONCLUSIVE", ""):
            problems.append(f"unknown RCA correctness {self.correctness!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rca_id": self.rca_id,
            "case_id": self.case_id,
            "investigator_id": self.investigator_id,
            "blind_status": self.blind_status,
            "symptom": self.symptom,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "identified_layer": self.identified_layer,
            "root_cause": self.root_cause,
            "scope": self.scope,
            "confidence": self.confidence,
            "ground_truth_match": self.correctness,
        }


def rca_verdict(
    *, records: Sequence[RCARecord], ground_truth: Mapping[str, Mapping[str, Any]], time_budget_s: float
) -> Dict[str, Any]:
    """Step 36/§15: reveal the ground truth and score layer/scope/cause."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for record in records:
        problems.extend(record.validate(ground_truth_hidden=True))
        truth = ground_truth.get(record.case_id, {})
        layer_ok = bool(truth) and record.identified_layer == truth.get("layer")
        cause_ok = bool(truth) and _cause_matches(record.root_cause, str(truth.get("root_cause", "")))
        verdict = "CORRECT" if layer_ok and cause_ok else "PARTIAL" if layer_ok else "INCORRECT"
        if record.mtti_s() > time_budget_s:
            problems.append(
                f"{record.case_id}: MTTI {record.mtti_s():.1f}s exceeds the pre-registered budget {time_budget_s}s"
            )
        if not layer_ok:
            problems.append(
                f"{record.case_id}: identified layer {record.identified_layer!r} != ground truth "
                f"{truth.get('layer', '<unknown>')!r}"
            )
        rows.append(
            {
                "rca_id": record.rca_id,
                "case_id": record.case_id,
                "investigator_id": record.investigator_id,
                "blind_status": record.blind_status,
                "symptom": record.symptom,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "identified_layer": record.identified_layer,
                "root_cause": record.root_cause,
                "scope": record.scope,
                "confidence": record.confidence,
                "ground_truth_match": verdict,
            }
        )
    return {"rows": rows, "cases": len(rows), "problems": problems, "ok": not problems}


def _cause_matches(claimed: str, truth: str) -> bool:
    if not truth:
        return False
    tokens = [token for token in truth.lower().replace("_", " ").split() if len(token) > 3]
    return any(token in claimed.lower() for token in tokens)


def telemetry_failure_detection(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 32: a stopped exporter/collector must raise an absence alert, not look like zero."""
    problems: List[str] = []
    for row in rows:
        component = str(row.get("component", ""))
        if component not in TELEMETRY_COMPONENTS:
            raise ConfigError(f"unknown telemetry component {component!r}")
        detected = bool(row.get("detected", False))
        silently_zero = bool(row.get("silently_zero", False))
        if not detected:
            problems.append(f"{component}: telemetry loss was not detected")
        if silently_zero:
            problems.append(
                f"{component}: missing data was rendered as healthy zero (an absence alert is required)"
            )
    return {"rows": list(rows), "problems": problems, "ok": not problems}


def instrumentation_overhead(
    *, rows: Sequence[Mapping[str, Any]], gate_pct: float
) -> Dict[str, Any]:
    """Step 19: overhead is measured A/B, per level, and must respect a pre-registered gate."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    levels: List[str] = []
    for row in rows:
        level = str(row.get("instrumentation_level", ""))
        levels.append(level)
        overhead = float(row.get("overhead_pct", 0.0))
        observations.append(
            {
                "case_id": row.get("case_id", ""),
                "instrumentation_level": level,
                "overhead_pct": overhead,
                "tail_impact": row.get("tail_impact", ""),
            }
        )
        if overhead > gate_pct:
            problems.append(f"{level}: overhead {overhead}% exceeds the gate {gate_pct}%")
    if "full" not in levels or "off" not in levels:
        problems.append("the A/B needs an 'off' and a 'full' instrumentation level")
    return {
        "rows": observations,
        "levels": sorted(set(levels)),
        "gate_pct": gate_pct,
        "problems": problems,
        "ok": not problems,
    }


def redaction_scan(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 33: canary values must not appear in any telemetry surface."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for row in rows:
        signal_class = str(row.get("signal_class", ""))
        if signal_class not in SIGNAL_CLASSES + ("artifacts",):
            raise ConfigError(f"unknown signal class {signal_class!r}")
        leaked = bool(row.get("leaked", False))
        observations.append(
            {
                "case_id": row.get("case_id", ""),
                "signal_class": signal_class,
                "canary_id": row.get("canary_id", ""),
                "subject": row.get("subject", ""),
                "leaked": leaked,
                "location": row.get("location", ""),
            }
        )
        if leaked:
            problems.append(
                f"{signal_class} exposed canary {row.get('canary_id')} at {row.get('location')}: "
                "sensitive content must be redacted on every path (including error paths)"
            )
    return {"rows": observations, "problems": problems, "ok": not problems}


def metric_kind_separation(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """§5/§14.2: user SLIs and system diagnostics are different objects."""
    problems: List[str] = []
    slis: List[str] = []
    diagnostics: List[str] = []
    for entry in entries:
        kind = str(entry.get("sli_or_diagnostic", ""))
        name = str(entry.get("metric_name", ""))
        if kind == "sli":
            slis.append(name)
        elif kind == "diagnostic":
            diagnostics.append(name)
        else:
            problems.append(f"{name}: must be declared as sli or diagnostic")
    if not slis:
        problems.append("no user-facing SLI is defined (system metrics do not replace user SLIs)")
    if not diagnostics:
        problems.append("no diagnostic metric is defined (an SLI alone cannot explain a cause)")
    return {
        "sli": sorted(slis),
        "diagnostics": sorted(diagnostics),
        "problems": problems,
        "ok": not problems,
        "note": "diagnostics explain an SLI; they never replace it in an SLO statement",
    }


def observability_verdict(
    *,
    conventions: Mapping[str, Any],
    coverage: Mapping[str, Any],
    cardinality: Mapping[str, Any],
    overhead: Mapping[str, Any],
    alerts: Mapping[str, Any],
    rca: Mapping[str, Any],
    telemetry_failures: Mapping[str, Any],
    redaction: Mapping[str, Any],
    kinds: Mapping[str, Any],
) -> Dict[str, Any]:
    """Step 38: coverage/overhead/cardinality limits and the RCA/alerts verdict."""
    problems: List[str] = []
    for name, axis in (
        ("convention_set", conventions),
        ("sampling_coverage", coverage),
        ("cardinality", cardinality),
        ("overhead", overhead),
        ("alerts", alerts),
        ("rca", rca),
        ("telemetry_failure_detection", telemetry_failures),
        ("redaction", redaction),
        ("metric_kind_separation", kinds),
    ):
        if not axis.get("ok"):
            problems.append(f"{name}: " + "; ".join(axis.get("problems", []) or ["(no detail)"]))
    return {
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "note": "RCA accuracy, alert precision and overhead numbers come from the executed experiment",
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结观测问题/责任层", ("observability:LAYERS", "observability:SemanticConvention")),
    (2, "冻结 semantic schema", ("observability:SemanticConvention.validate", "observability:ATTRIBUTE_LEVELS")),
    (3, "冻结隐私/基数 policy", ("observability:validate_metric_labels", "records:FORBIDDEN_METRIC_LABELS")),
    (4, "冻结 sampling policy", ("observability:SamplingPolicy", "observability:ALWAYS_KEEP_CASES")),
    (5, "冻结 SLI/SLO/alert 规则", ("observability:SLIDefinition", "observability:AlertRule")),
    (6, "建立 clock/time 基础", ("observability:ClockAlignment", "observability:clock_verdict")),
    (7, "接入 client/gateway spans", ("observability:SemanticConvention",)),
    (8, "接入 auth/admission/routing spans", ("observability:SemanticConvention", "security:SecurityCase")),
    (9, "接入 queue/scheduler/batch/KV", ("observability:SemanticConvention", "capacity:token_work")),
    (10, "接入 runtime/model/backend", ("observability:SemanticConvention", "artifacts:CacheKey")),
    (11, "接入 operator/device profile 关联", ("observability:profile_index",)),
    (12, "接入 system/storage/network", ("observability:SemanticConvention",)),
    (13, "接入 accelerator telemetry", ("observability:SemanticConvention", "scheduling:DeviceRecord")),
    (14, "接入 deployment/control events", ("telemetry:project_release_identity", "observability:SemanticConvention")),
    (15, "验证 context propagation", ("observability:trace_coverage",)),
    (16, "验证 metric 与 raw 对账", ("observability:reconcile_metric_with_raw",)),
    (17, "验证直方图 bucket", ("observability:validate_histogram_buckets",)),
    (18, "测 cardinality", ("observability:classify_cardinality",)),
    (19, "测 instrumentation overhead", ("observability:instrumentation_overhead",)),
    (20, "测采样 coverage", ("observability:sampling_coverage",)),
    (21, "建立正常 baseline dashboard", ("observability:dashboard_definition",)),
    (22, "建立 alerts 与 runbooks", ("observability:build_alert_rule",)),
    (23, "注入 queue/scheduler 慢请求", ("faults:FaultSpec", "observability:rca_verdict")),
    (24, "注入 runtime/kernel 退化", ("faults:FaultSpec", "observability:rca_verdict")),
    (25, "注入 device clock/thermal/power 异常", ("faults:FaultSpec", "observability:rca_verdict")),
    (26, "注入 CPU/memory/swap/IO 瓶颈", ("faults:FaultSpec", "observability:rca_verdict")),
    (27, "注入 network/storage 故障", ("faults:FaultSpec", "observability:rca_verdict")),
    (28, "注入 distributed straggler/link 问题", ("faults:FaultSpec", "observability:rca_verdict")),
    (29, "执行盲化慢请求 RCA", ("observability:RCARecord",)),
    (30, "执行盲化故障 RCA", ("observability:RCARecord", "observability:rca_verdict")),
    (31, "验证 alert 检测/误告", ("observability:alert_effectiveness",)),
    (32, "验证 telemetry 缺失检测", ("observability:telemetry_failure_detection",)),
    (33, "验证日志/trace 脱敏", ("observability:redaction_scan", "contracts:redact_payload")),
    (34, "执行 profile-on-demand", ("observability:profile_index", "faults:FaultSpec")),
    (35, "重复不同负载/时段", ("observability:rca_verdict", "observability:alert_effectiveness")),
    (36, "复核 root-cause 证据", ("observability:rca_verdict",)),
    (37, "修订并独立确认", ("observability:confirmation_case",)),
    (38, "形成 observability verdict", ("observability:observability_verdict",)),
)


def trace_coverage(*, traces: Sequence[Mapping[str, Any]], required_hops: Sequence[str]) -> Dict[str, Any]:
    """Step 15: the trace closure must contain release/model/backend/pod/device."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for trace in traces:
        observed = set(trace.get("hops", ()) or ())
        missing = sorted(set(required_hops) - observed)
        unknown = sorted(hop for hop in observed if hop in ("unknown", ""))
        rows.append(
            {
                "trace_id": trace.get("trace_id", ""),
                "request_id": trace.get("request_id", ""),
                "hops_expected": len(required_hops),
                "hops_observed": len(observed),
                "unknown_hops": unknown,
                "release_id": trace.get("release_id", ""),
                "device_id": trace.get("device_id", ""),
            }
        )
        if missing:
            problems.append(f"trace {trace.get('trace_id')}: missing hops {missing}")
        if unknown:
            problems.append(f"trace {trace.get('trace_id')}: unknown identity hops {unknown}")
    return {"rows": rows, "problems": problems, "ok": not problems}


def dashboard_definition(*, panels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 21: every panel carries query/unit/source — no hand-copied numbers."""
    problems: List[str] = []
    for panel in panels:
        for field_name in ("panel_id", "query", "unit", "source_metric"):
            if not panel.get(field_name):
                problems.append(f"panel {panel.get('panel_id', '?')}: missing {field_name!r}")
        if panel.get("hand_copied"):
            problems.append(f"panel {panel.get('panel_id')}: numbers must not be hand-copied into a dashboard")
    return {
        "panels": len(panels),
        "problems": problems,
        "ok": not problems,
        "drilldown_order": ["user_sli", "queue_runtime", "device_system"],
    }


def profile_index(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 34: profiling is on demand, bounded and linked back to a trace."""
    problems: List[str] = []
    for row in rows:
        for field_name in ("profile_id", "target", "tool", "started_at", "ended_at", "trace_id"):
            if not row.get(field_name):
                problems.append(f"profile {row.get('profile_id', '?')}: missing {field_name!r}")
        if row.get("always_on"):
            problems.append(f"profile {row.get('profile_id')}: always-on profiling distorts the measurement")
        if row.get("duration_s") is not None and float(row["duration_s"]) > float(row.get("budget_s", 30.0)):
            problems.append(f"profile {row.get('profile_id')}: exceeded the profiling time budget")
    return {"rows": list(rows), "problems": problems, "ok": not problems}


def confirmation_case(*, case: Mapping[str, Any], revised_schema: bool, same_case_as_before: bool) -> Dict[str, Any]:
    """Step 37: a fix is confirmed on a *new* case, never on the failing one."""
    problems: List[str] = []
    if not revised_schema:
        problems.append("the schema/dashboard/alert/runbook revision must be recorded before confirmation")
    if same_case_as_before:
        problems.append("confirmation must use a new case (self-confirmation is not confirmation)")
    if case.get("detected") is False:
        problems.append("the revised telemetry did not detect the confirmation case")
    return {"case_id": case.get("case_id", ""), "problems": problems, "ok": not problems}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the observability contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    conventions = [
        SemanticConvention(name="hqsb_client_ttft_seconds", signal_class="metrics", unit="seconds",
                           attributes={"boundary": "required", "workload_bucket": "required"},
                           required_attributes=("boundary",), version="1.0.0", boundary="client"),
        SemanticConvention(name="hqsb_server_ttft_seconds", signal_class="metrics", unit="seconds",
                           attributes={"boundary": "required"}, required_attributes=("boundary",),
                           version="1.0.0", boundary="server"),
        SemanticConvention(name="hqsb_core_tpot_seconds", signal_class="metrics", unit="seconds",
                           attributes={"boundary": "required"}, required_attributes=("boundary",),
                           version="1.0.0", boundary="core"),
        SemanticConvention(name="hqsb_request_span", signal_class="traces", version="1.0.0",
                           attributes={"request_id": "required", "release_id": "required"},
                           required_attributes=("request_id", "release_id")),
    ]
    checks["convention_set_ok"] = validate_convention_set(conventions)["ok"] is True

    bad_convention = SemanticConvention(name="hqsb_requests_total", signal_class="metrics", unit="count",
                                        attributes={"request_id": "required"}, version="1.0.0")
    checks["cardinality_label_rejected"] = any(
        "unbounded" in problem for problem in bad_convention.validate()
    )

    weak_sampling = SamplingPolicy(policy_id="s1", head_rate=0.01, tail_enabled=False,
                                   keep_cases=("error",), budget_events_per_s=1000.0)
    checks["sampling_policy_enforced"] = len(weak_sampling.validate()) >= 2

    buckets = validate_histogram_buckets(
        metric_name="hqsb_server_ttft_seconds",
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")],
        observed_p99=0.4,
    )
    checks["default_bucket_set_flagged"] = buckets["ok"] is False

    reconciliation = reconcile_metric_with_raw(
        metric_name="hqsb_requests_total", metric_value=10.0,
        raw_rows=[{"value": 4.0}, {"value": 4.0}],
    )
    checks["metric_raw_mismatch_detected"] = reconciliation["ok"] is False

    alert = build_alert_rule(
        alert_id="a1", name="TTFT burn", user_impact="users see slow first token",
        query='histogram_quantile(0.99, hqsb_server_ttft_seconds_bucket) > 2',
        for_duration="10m", severity="critical", owner="sre", runbook_id="rb-ttft",
        threshold_value=2.0,
    )
    checks["alert_rule_valid"] = alert.validate() == []

    failure = telemetry_failure_detection(
        [{"component": "exporter", "detected": False, "silently_zero": True}]
    )
    checks["telemetry_loss_detected"] = failure["ok"] is False

    leak = redaction_scan(
        [{"case_id": "c1", "signal_class": "logs", "canary_id": "canary-1", "leaked": True,
          "location": "exception stack"}]
    )
    checks["redaction_leak_detected"] = leak["ok"] is False
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未采集、未查询任何遥测后端",
    }