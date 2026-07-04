"""Cross-cutting validators for S13 (``details/S13/README.md`` §23).

The fourteen bullets of §23 are the *semantic* rules that keep the eleven
experiments from producing individually plausible but jointly meaningless
artifacts.  Each rule is implemented as a pure validator that returns
``{"ok": bool, "reason_codes": [...], "findings": [...]}`` so a report can cite a
stable reason code instead of prose.

Two rules deserve naming because they are the classic "looks fine" traps:

* a *tag* is never a deployment identity, and an incomplete release identity may
  not become ``ready``;
* ``missing`` telemetry is never ``0`` and a stale metric may not be treated as
  the latest one.

Nothing here executes an experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import identity as idn
from hqsb.infra import records as rec

SCHEMA_VERSION = "1.0.0"

# ── stable reason codes ───────────────────────────────────────────────────

RC_RELEASE_IDENTITY_INCOMPLETE = "RELEASE_IDENTITY_INCOMPLETE"
RC_TAG_USED_AS_IDENTITY = "TAG_USED_AS_IDENTITY"
RC_SECRET_IN_IMAGE_CONTEXT = "SECRET_IN_IMAGE_CONTEXT"
RC_MODEL_WEIGHTS_IN_GENERIC_IMAGE = "MODEL_WEIGHTS_IN_GENERIC_IMAGE"
RC_READINESS_WITHOUT_MODEL_STATE = "READINESS_WITHOUT_MODEL_STATE"
RC_LIVENESS_OVERLOAD_COUPLING = "LIVENESS_OVERLOAD_COUPLING"
RC_DRAINING_ACCEPTED_NEW_WORK = "DRAINING_ACCEPTED_NEW_WORK"
RC_ADMISSION_MISSING_BUDGET = "ADMISSION_MISSING_BUDGET"
RC_METRIC_WITHOUT_TIMESTAMP = "METRIC_WITHOUT_TIMESTAMP"
RC_METRIC_STALE_USED_AS_FRESH = "METRIC_STALE_USED_AS_FRESH"
RC_UNDECLARED_CANARY_DIFFERENCE = "UNDECLARED_CANARY_DIFFERENCE"
RC_FAULT_TARGET_OUT_OF_SCOPE = "FAULT_TARGET_OUT_OF_SCOPE"
RC_RECOVERY_ASSERTED_FROM_POD_STATE = "RECOVERY_ASSERTED_FROM_POD_STATE"
RC_HIGH_CARDINALITY_METRIC_LABEL = "HIGH_CARDINALITY_METRIC_LABEL"
RC_SENSITIVE_DATA_IN_PUBLIC_ARTIFACT = "SENSITIVE_DATA_IN_PUBLIC_ARTIFACT"
RC_REPORT_POINT_WITHOUT_EVIDENCE_REF = "REPORT_POINT_WITHOUT_EVIDENCE_REF"
RC_MISSING_VALUE_AS_ZERO = "MISSING_VALUE_AS_ZERO"
RC_QUALITY_GATE_FAILED_FOR_MEASURED_RESULT = "QUALITY_GATE_FAILED_FOR_MEASURED_RESULT"

ALL_REASON_CODES: Tuple[str, ...] = (
    RC_RELEASE_IDENTITY_INCOMPLETE,
    RC_TAG_USED_AS_IDENTITY,
    RC_SECRET_IN_IMAGE_CONTEXT,
    RC_MODEL_WEIGHTS_IN_GENERIC_IMAGE,
    RC_READINESS_WITHOUT_MODEL_STATE,
    RC_LIVENESS_OVERLOAD_COUPLING,
    RC_DRAINING_ACCEPTED_NEW_WORK,
    RC_ADMISSION_MISSING_BUDGET,
    RC_METRIC_WITHOUT_TIMESTAMP,
    RC_METRIC_STALE_USED_AS_FRESH,
    RC_UNDECLARED_CANARY_DIFFERENCE,
    RC_FAULT_TARGET_OUT_OF_SCOPE,
    RC_RECOVERY_ASSERTED_FROM_POD_STATE,
    RC_HIGH_CARDINALITY_METRIC_LABEL,
    RC_SENSITIVE_DATA_IN_PUBLIC_ARTIFACT,
    RC_REPORT_POINT_WITHOUT_EVIDENCE_REF,
    RC_MISSING_VALUE_AS_ZERO,
    RC_QUALITY_GATE_FAILED_FOR_MEASURED_RESULT,
)

#: Severities: ``fail`` blocks the artifact, ``warn`` needs an explicit note.
SEVERITIES: Tuple[str, ...] = ("fail", "warn")


@dataclass(frozen=True)
class ValidatorConstraint:
    """One §23 rule, citable by id in an acceptance report."""

    constraint_id: str
    name: str
    source: str
    description: str
    severity: str = "fail"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "severity": self.severity,
        }


VALIDATOR_CONSTRAINTS: Tuple[ValidatorConstraint, ...] = (
    ValidatorConstraint(
        "V01",
        "release_identity_complete",
        "details/S13/README.md §23 第 1 条",
        "image/model/engine/kernel/config 身份不完整时 release 不得 ready",
    ),
    ValidatorConstraint(
        "V02",
        "digest_not_tag",
        "details/S13/README.md §23 第 2 条",
        "tag 不能代替 digest 作为部署身份",
    ),
    ValidatorConstraint(
        "V03",
        "no_secret_or_weights_in_image",
        "details/S13/README.md §23 第 3 条",
        "secret/模型权重命中时通用镜像 gate 必须 fail",
    ),
    ValidatorConstraint(
        "V04",
        "readiness_references_semantic_state",
        "details/S13/README.md §23 第 4 条",
        "readiness=true 必须引用 active model/quality/warmup/capacity state",
    ),
    ValidatorConstraint(
        "V05",
        "liveness_not_overload_coupled",
        "details/S13/README.md §23 第 5 条",
        "liveness 不得因正常过载/queue 高直接失败",
    ),
    ValidatorConstraint(
        "V06",
        "draining_rejects_new_work",
        "details/S13/README.md §23 第 6 条",
        "draining 后新请求必须拒绝/重路由",
    ),
    ValidatorConstraint(
        "V07",
        "admission_decision_is_explainable",
        "details/S13/README.md §23 第 7 条",
        "admission decision 必须有 policy/reason/current budget",
    ),
    ValidatorConstraint(
        "V08",
        "autoscaling_metric_has_age",
        "details/S13/README.md §23 第 8 条",
        "autoscaling metric 必须有 timestamp/age，stale metric 不得当最新",
    ),
    ValidatorConstraint(
        "V09",
        "canary_scope_preregistered",
        "details/S13/README.md §23 第 9 条",
        "canary candidate/control 只允许预注册差异",
    ),
    ValidatorConstraint(
        "V10",
        "fault_target_authorized",
        "details/S13/README.md §23 第 10 条",
        "fault target 必须在授权 selector/blast radius 内",
    ),
    ValidatorConstraint(
        "V11",
        "recovery_verified_beyond_pod",
        "details/S13/README.md §23 第 11 条",
        "recovery 完成必须验证状态、资源和服务，不以 Pod Running 代替",
    ),
    ValidatorConstraint(
        "V12",
        "no_high_cardinality_labels",
        "details/S13/README.md §23 第 12 条",
        "tenant/user/request ID 禁止进入高基数 metrics labels",
    ),
    ValidatorConstraint(
        "V13",
        "no_sensitive_data_in_public_artifacts",
        "details/S13/README.md §23 第 13 条",
        "secret/token/prompt/模型私有内容禁止出现在公开 log/artifact",
    ),
    ValidatorConstraint(
        "V14",
        "report_points_reference_evidence",
        "details/S13/README.md §23 第 14 条",
        "report/dashboard point 必须引用 raw/normalized IDs",
    ),
)

_CONSTRAINT_BY_ID: Mapping[str, ValidatorConstraint] = {
    constraint.constraint_id: constraint for constraint in VALIDATOR_CONSTRAINTS
}


def constraint(constraint_id: str) -> ValidatorConstraint:
    try:
        return _CONSTRAINT_BY_ID[constraint_id]
    except KeyError as exc:
        raise ConfigError(f"unknown validator constraint {constraint_id!r}") from exc


# ── finding helpers ───────────────────────────────────────────────────────


@dataclass
class Finding:
    """One rule violation with the artifact it was found on."""

    constraint_id: str
    reason_code: str
    subject: str
    detail: str = ""
    severity: str = "fail"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "reason_code": self.reason_code,
            "subject": self.subject,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """A single validator's verdict (stable, machine-readable)."""

    constraint_id: str
    ok: bool
    findings: List[Finding] = field(default_factory=list)

    @property
    def reason_codes(self) -> List[str]:
        return sorted({finding.reason_code for finding in self.findings})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "ok": self.ok,
            "reason_codes": self.reason_codes,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def _result(constraint_id: str, findings: Sequence[Finding]) -> ValidationResult:
    return ValidationResult(constraint_id=constraint_id, ok=not findings, findings=list(findings))


# ── V01 / V02: release identity and digest pinning ────────────────────────


def validate_release_identity(bundle: Any) -> ValidationResult:
    """V01: an incomplete identity may not become a deployable/ready release."""
    findings: List[Finding] = []
    problems = bundle.validate()
    for problem in problems:
        code = RC_MODEL_WEIGHTS_IN_GENERIC_IMAGE if "model weights" in problem else RC_RELEASE_IDENTITY_INCOMPLETE
        findings.append(
            Finding("V01", code, getattr(bundle, "release_id", "<unnamed>"), problem)
        )
    if bundle.status in ("DEPLOYABLE", "SUPPLY_CHAIN_PASSED") and not bundle.identity_complete():
        findings.append(
            Finding(
                "V01",
                RC_RELEASE_IDENTITY_INCOMPLETE,
                bundle.release_id,
                "identity is incomplete yet the release is marked deployable",
            )
        )
    return _result("V01", findings)


def validate_digest_pinning(deployments: Iterable[Mapping[str, Any]]) -> ValidationResult:
    """V02: every deployment reference must be a digest, never a tag."""
    findings: List[Finding] = []
    for index, deployment in enumerate(deployments):
        reference = str(deployment.get("image", deployment.get("image_reference", "")))
        subject = str(deployment.get("workload_id") or f"deployment[{index}]")
        if not reference:
            findings.append(Finding("V02", RC_TAG_USED_AS_IDENTITY, subject, "deployment has no image reference"))
            continue
        try:
            idn.require_digest_reference(reference, field_name="image")
        except ConfigError:
            findings.append(
                Finding("V02", RC_TAG_USED_AS_IDENTITY, subject, f"tag reference {reference!r} cannot identify a release")
            )
    return _result("V02", findings)


# ── V03: image content gate ───────────────────────────────────────────────


def validate_image_content_gate(
    *,
    secret_findings: Sequence[Mapping[str, Any]],
    model_findings: Sequence[Mapping[str, Any]],
    release_id: str = "",
) -> ValidationResult:
    """V03: any secret or model-weight hit fails the generic image gate."""
    findings: List[Finding] = []
    for finding in secret_findings:
        findings.append(
            Finding(
                "V03",
                RC_SECRET_IN_IMAGE_CONTEXT,
                release_id or str(finding.get("finding_id", "")),
                f"secret pattern {finding.get('pattern', '')} on surface {finding.get('surface', '')}",
            )
        )
    for finding in model_findings:
        findings.append(
            Finding(
                "V03",
                RC_MODEL_WEIGHTS_IN_GENERIC_IMAGE,
                release_id or str(finding.get("finding_id", "")),
                f"model artifact pattern {finding.get('pattern', '')} inside the generic runtime image",
            )
        )
    return _result("V03", findings)


# ── V04 / V05 / V06: readiness, liveness, drain ───────────────────────────


def validate_readiness_claim(claim: Mapping[str, Any]) -> ValidationResult:
    """V04: ``ready=true`` must cite the semantic state, not just a live process."""
    findings: List[Finding] = []
    if not claim.get("ready"):
        return _result("V04", findings)
    for condition in rec.READINESS_CONDITIONS:
        if not claim.get(condition, False):
            findings.append(
                Finding("V04", RC_READINESS_WITHOUT_MODEL_STATE, condition, f"readiness claim lacks {condition}=true")
            )
    for identity_field in ("release_id", "model_artifact_id"):
        if not claim.get(identity_field):
            findings.append(
                Finding(
                    "V04",
                    RC_READINESS_WITHOUT_MODEL_STATE,
                    identity_field,
                    "readiness must name the active release/model identity",
                )
            )
    if claim.get("healthz_status") == 200 and claim.get("model_state") in ("", "ABSENT", None):
        findings.append(
            Finding(
                "V04",
                RC_READINESS_WITHOUT_MODEL_STATE,
                "healthz",
                "a 200 from /healthz is not evidence that the model is ready",
            )
        )
    return _result("V04", findings)


def validate_liveness_policy(policy: Mapping[str, Any]) -> ValidationResult:
    """V05: liveness must not be coupled to load/queue state."""
    findings: List[Finding] = []
    for field_name in ("probe_signals", "failure_conditions", "signals"):
        signals = policy.get(field_name, ())
        if isinstance(signals, str):
            signals = (signals,)
        for signal in signals:
            lowered = str(signal).lower()
            if any(token in lowered for token in ("queue_depth", "queue", "load", "utilization", "saturation")):
                findings.append(
                    Finding(
                        "V05",
                        RC_LIVENESS_OVERLOAD_COUPLING,
                        str(signal),
                        "liveness must detect an unrecoverable process state, not high load "
                        "(a load-coupled liveness probe manufactures restart cascades)",
                    )
                )
    if policy.get("liveness_endpoint") and policy.get("liveness_endpoint") == policy.get("readiness_endpoint"):
        findings.append(
            Finding(
                "V05",
                RC_LIVENESS_OVERLOAD_COUPLING,
                str(policy.get("liveness_endpoint")),
                "startup/readiness/liveness must not all point at the same 'process is alive' endpoint",
            )
        )
    return _result("V05", findings)


def validate_drain_semantics(
    *, draining: bool, accepted_after_drain: int, state: str = ""
) -> ValidationResult:
    """V06: after the drain linearization point no new work may start here."""
    findings: List[Finding] = []
    if draining and int(accepted_after_drain) > 0:
        findings.append(
            Finding(
                "V06",
                RC_DRAINING_ACCEPTED_NEW_WORK,
                state or "draining",
                f"{accepted_after_drain} new executions started after the drain point",
            )
        )
    return _result("V06", findings)


# ── V07 / V08: admission and autoscaling ─────────────────────────────────


def validate_admission_decision(decision: Mapping[str, Any]) -> ValidationResult:
    """V07: an admission decision must be explainable from current budgets."""
    findings: List[Finding] = []
    required = (
        "policy_version",
        "reason_code",
        "usable_memory_bytes",
        "safety_margin_bytes",
        "predicted_incremental_kv_bytes",
    )
    for field_name in required:
        if decision.get(field_name) in (None, ""):
            findings.append(
                Finding("V07", RC_ADMISSION_MISSING_BUDGET, field_name, f"admission decision lacks {field_name}")
            )
    if decision.get("usable_memory_bytes") is not None and decision.get("predicted_incremental_kv_bytes") is not None:
        try:
            usable = rec.numeric_value(decision["usable_memory_bytes"])
            margin = rec.numeric_value(decision["safety_margin_bytes"])
            predicted = rec.numeric_value(decision["predicted_incremental_kv_bytes"])
        except ConfigError as exc:
            findings.append(Finding("V07", RC_MISSING_VALUE_AS_ZERO, "admission", str(exc)))
            return _result("V07", findings)
        if decision.get("decision") == "ADMIT_NOW" and predicted > max(usable - margin, 0.0):
            findings.append(
                Finding(
                    "V07",
                    RC_ADMISSION_MISSING_BUDGET,
                    "ADMIT_NOW",
                    "admitted although the predicted increment exceeds usable memory minus margin",
                )
            )
    return _result("V07", findings)


def validate_autoscaling_metric(
    sample: Mapping[str, Any], *, now: Optional[float] = None, max_age_s: float = 60.0
) -> ValidationResult:
    """V08: a metric without a timestamp/age may not drive a decision."""
    findings: List[Finding] = []
    timestamp = sample.get("event_ts", sample.get("timestamp"))
    if timestamp in (None, ""):
        findings.append(
            Finding("V08", RC_METRIC_WITHOUT_TIMESTAMP, str(sample.get("metric_name", "")), "metric has no timestamp")
        )
        return _result("V08", findings)
    age = sample.get("age_s")
    if age in (None, ""):
        if now is not None:
            age = float(now) - float(timestamp)
        else:
            findings.append(
                Finding(
                    "V08",
                    RC_METRIC_WITHOUT_TIMESTAMP,
                    str(sample.get("metric_name", "")),
                    "metric age must be recorded (event→export→scrape→query delays matter)",
                )
            )
            return _result("V08", findings)
    age_value = rec.numeric_value(age)
    if age_value > max_age_s and sample.get("used_as_current", True):
        findings.append(
            Finding(
                "V08",
                RC_METRIC_STALE_USED_AS_FRESH,
                str(sample.get("metric_name", "")),
                f"metric age {age_value}s exceeds the staleness policy {max_age_s}s",
            )
        )
    return _result("V08", findings)


# ── V09 / V10 / V11: canary scope, fault scope, recovery ─────────────────


def validate_canary_scope(
    control: Any, candidate: Any, *, declared_changes: Sequence[str]
) -> ValidationResult:
    """V09: only pre-registered release differences may exist."""
    scope = idn.validate_release_change_scope(control, candidate, declared_changes=declared_changes)
    findings: List[Finding] = []
    for row in scope["undeclared_differences"]:
        findings.append(
            Finding(
                "V09",
                RC_UNDECLARED_CANARY_DIFFERENCE,
                row["field"],
                f"control={row['control']!r} candidate={row['candidate']!r} was not declared as a canary change",
            )
        )
    return _result("V09", findings)


def validate_fault_target(
    fault_spec: Mapping[str, Any], authorized: Mapping[str, Any]
) -> ValidationResult:
    """V10: a fault may only hit resolved, authorized targets inside the blast radius."""
    findings: List[Finding] = []
    selector = str(fault_spec.get("target_selector", ""))
    if any(char in selector for char in ("*", "?", "[")):
        findings.append(
            Finding("V10", RC_FAULT_TARGET_OUT_OF_SCOPE, selector, "wildcard selectors are not allowed for faults")
        )
    resolved = [str(item) for item in fault_spec.get("resolved_targets", ()) or ()]
    if not resolved:
        findings.append(
            Finding("V10", RC_FAULT_TARGET_OUT_OF_SCOPE, selector, "fault has no resolved targets (no wildcard execution)")
        )
    allowed_namespaces = {str(item) for item in authorized.get("namespaces", ()) or ()}
    namespace = str(fault_spec.get("namespace", authorized.get("namespace", "")))
    if allowed_namespaces and namespace not in allowed_namespaces:
        findings.append(
            Finding("V10", RC_FAULT_TARGET_OUT_OF_SCOPE, namespace, "namespace is outside the authorized set")
        )
    max_targets = int(authorized.get("max_targets", 0) or 0)
    if max_targets and len(resolved) > max_targets:
        findings.append(
            Finding(
                "V10",
                RC_FAULT_TARGET_OUT_OF_SCOPE,
                selector,
                f"{len(resolved)} targets exceed the authorized blast radius {max_targets}",
            )
        )
    if not fault_spec.get("abort_threshold"):
        findings.append(
            Finding("V10", RC_FAULT_TARGET_OUT_OF_SCOPE, selector, "fault has no error-budget abort threshold")
        )
    if not fault_spec.get("kill_switch"):
        findings.append(Finding("V10", RC_FAULT_TARGET_OUT_OF_SCOPE, selector, "fault has no kill switch"))
    return _result("V10", findings)


def validate_recovery_completion(evidence: Mapping[str, Any]) -> ValidationResult:
    """V11: recovery needs service + resource + state evidence, not ``Pod Running``."""
    findings: List[Finding] = []
    if not evidence.get("pod_running", True):
        findings.append(Finding("V11", RC_RECOVERY_ASSERTED_FROM_POD_STATE, "pod", "pod is not running"))
    required_axes = ("service", "resource", "state")
    for axis in required_axes:
        entry = evidence.get(axis)
        if not entry:
            findings.append(
                Finding(
                    "V11",
                    RC_RECOVERY_ASSERTED_FROM_POD_STATE,
                    axis,
                    f"recovery claim has no {axis} verification "
                    "(Pod Running is not service recovery)",
                )
            )
            continue
        if isinstance(entry, Mapping) and not entry.get("verified", False):
            findings.append(
                Finding("V11", RC_RECOVERY_ASSERTED_FROM_POD_STATE, axis, f"{axis} verification is not marked verified")
            )
    if evidence.get("correctness_requeried") is False or evidence.get("correctness_rerun") is False:
        findings.append(
            Finding(
                "V11",
                RC_RECOVERY_ASSERTED_FROM_POD_STATE,
                "correctness",
                "recovery must re-run the correctness/quality probe",
            )
        )
    return _result("V11", findings)


# ── V12 / V13 / V14: cardinality, leakage, evidence references ───────────


def validate_metric_labels(labels: Mapping[str, Any]) -> ValidationResult:
    """V12: unbounded identifiers belong in traces/logs, not metric labels."""
    findings: List[Finding] = []
    for name in labels:
        if name in rec.FORBIDDEN_METRIC_LABELS:
            findings.append(
                Finding(
                    "V12",
                    RC_HIGH_CARDINALITY_METRIC_LABEL,
                    name,
                    "unbounded identity in a Prometheus label creates one time series per value",
                )
            )
    return _result("V12", findings)


#: Patterns that must never appear in a public log/artifact (redaction scan).
SENSITIVE_PATTERNS: Tuple[str, ...] = (
    "AKIA[0-9A-Z]{16}",
    "-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "(?i)bearer\\s+[A-Za-z0-9._-]{20,}",
    "(?i)password\\s*=\\s*\\S+",
    "(?i)api[_-]?key\\s*[=:]\\s*\\S+",
    "(?i)hf_[A-Za-z0-9]{20,}",
)

_SENSITIVE_REGEXES: Tuple[re.Pattern, ...] = tuple(re.compile(pattern) for pattern in SENSITIVE_PATTERNS)


def validate_public_payload(payload: str, *, subject: str = "artifact") -> ValidationResult:
    """V13: a public artifact must not carry credentials or private content."""
    findings: List[Finding] = []
    for regex in _SENSITIVE_REGEXES:
        for match in regex.finditer(payload or ""):
            findings.append(
                Finding(
                    "V13",
                    RC_SENSITIVE_DATA_IN_PUBLIC_ARTIFACT,
                    subject,
                    f"matched pattern {regex.pattern!r} at offset {match.start()} "
                    "(value deliberately not echoed)",
                )
            )
    return _result("V13", findings)


def redact_payload(payload: str) -> str:
    """Irreversibly mask sensitive matches (the original value is never returned)."""
    redacted = payload or ""
    for regex in _SENSITIVE_REGEXES:
        redacted = regex.sub("[REDACTED]", redacted)
    return redacted


def validate_report_point(point: Mapping[str, Any]) -> ValidationResult:
    """V14: a report/dashboard point must trace to raw or normalized ids."""
    findings: List[Finding] = []
    references = list(point.get("evidence_refs", ()) or ())
    if not references:
        findings.append(
            Finding(
                "V14",
                RC_REPORT_POINT_WITHOUT_EVIDENCE_REF,
                str(point.get("point_id", point.get("metric_name", ""))),
                "the point references no raw/normalized artifact",
            )
        )
    for reference in references:
        text = str(reference)
        if "#" not in text and "/" not in text:
            findings.append(
                Finding(
                    "V14",
                    RC_REPORT_POINT_WITHOUT_EVIDENCE_REF,
                    text,
                    "evidence reference must be a path/id locator, not a free-text claim",
                )
            )
    if point.get("hand_copied") or point.get("manual_value"):
        findings.append(
            Finding(
                "V14",
                RC_REPORT_POINT_WITHOUT_EVIDENCE_REF,
                str(point.get("point_id", "")),
                "hand-copied numbers may not enter a report point",
            )
        )
    return _result("V14", findings)


# ── aggregate ─────────────────────────────────────────────────────────────


def validate_measured_result(result: Mapping[str, Any]) -> ValidationResult:
    """Quality precedes performance: a failing correctness gate blocks a measurement."""
    findings: List[Finding] = []
    if result.get("result_class") == "MEASURED" and result.get("quality_status") not in ("pass", "not_run"):
        findings.append(
            Finding(
                "V01",
                RC_QUALITY_GATE_FAILED_FOR_MEASURED_RESULT,
                str(result.get("result_id", "")),
                "a MEASURED result may not carry a failed quality/correctness gate",
            )
        )
    return _result("V01", findings)


def validate_all_artifacts(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    """Run every applicable validator over a bundle of artifacts.

    ``artifacts`` is a mapping whose values may be absent; an absent axis is
    reported as *not checked* rather than silently passing.
    """
    results: List[ValidationResult] = []
    not_checked: List[str] = []

    if "release" in artifacts and artifacts["release"] is not None:
        results.append(validate_release_identity(artifacts["release"]))
    else:
        not_checked.append("V01")
    if artifacts.get("deployments"):
        results.append(validate_digest_pinning(artifacts["deployments"]))
    else:
        not_checked.append("V02")
    if artifacts.get("secret_findings") is not None or artifacts.get("model_findings") is not None:
        results.append(
            validate_image_content_gate(
                secret_findings=artifacts.get("secret_findings", ()),
                model_findings=artifacts.get("model_findings", ()),
                release_id=str(getattr(artifacts.get("release"), "release_id", "")),
            )
        )
    else:
        not_checked.append("V03")
    if artifacts.get("readiness"):
        results.append(validate_readiness_claim(artifacts["readiness"]))
    else:
        not_checked.append("V04")
    if artifacts.get("liveness_policy"):
        results.append(validate_liveness_policy(artifacts["liveness_policy"]))
    else:
        not_checked.append("V05")
    if artifacts.get("drain"):
        drain = artifacts["drain"]
        results.append(
            validate_drain_semantics(
                draining=bool(drain.get("draining", True)),
                accepted_after_drain=int(drain.get("accepted_after_drain", 0)),
                state=str(drain.get("state", "")),
            )
        )
    else:
        not_checked.append("V06")
    if artifacts.get("admission_decisions"):
        for decision in artifacts["admission_decisions"]:
            results.append(validate_admission_decision(decision))
    else:
        not_checked.append("V07")
    if artifacts.get("metrics"):
        for sample in artifacts["metrics"]:
            results.append(validate_autoscaling_metric(sample))
    else:
        not_checked.append("V08")
    if artifacts.get("canary_scope"):
        scope = artifacts["canary_scope"]
        results.append(
            validate_canary_scope(scope["control"], scope["candidate"], declared_changes=scope.get("declared_changes", ()))
        )
    else:
        not_checked.append("V09")
    if artifacts.get("fault_specs"):
        for spec in artifacts["fault_specs"]:
            results.append(validate_fault_target(spec, artifacts.get("authorized_scope", {})))
    else:
        not_checked.append("V10")
    if artifacts.get("recovery"):
        for evidence in artifacts["recovery"]:
            results.append(validate_recovery_completion(evidence))
    else:
        not_checked.append("V11")
    if artifacts.get("metric_labels"):
        for labels in artifacts["metric_labels"]:
            results.append(validate_metric_labels(labels))
    else:
        not_checked.append("V12")
    if artifacts.get("public_payloads"):
        for index, payload in enumerate(artifacts["public_payloads"]):
            results.append(validate_public_payload(payload, subject=f"payload[{index}]"))
    else:
        not_checked.append("V13")
    if artifacts.get("report_points"):
        for point in artifacts["report_points"]:
            results.append(validate_report_point(point))
    else:
        not_checked.append("V14")

    failed = [result for result in results if not result.ok]
    return {
        "constraints": len(VALIDATOR_CONSTRAINTS),
        "checked": sorted({result.constraint_id for result in results}),
        "not_checked": sorted(not_checked),
        "failed": [result.constraint_id for result in failed],
        "ok": not failed,
        "results": [result.as_dict() for result in results],
    }
