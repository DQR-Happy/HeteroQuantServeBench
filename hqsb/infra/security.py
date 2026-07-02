"""E13-11: multi-tenant identity, authorization, quota, redaction and isolation.

Implements ``details/S13/E13-11_*.md`` as data:

* the threat model of §5 (tenant definition, attacker capability, trusted
  components, protected assets, boundaries, explicit non-goals, fail-open /
  fail-closed policy) — a security claim without a threat model is not a claim;
* the twelve invariants ``MT-I01``–``MT-I12`` with per-invariant verdicts;
* the RBAC permission-graph audit (wildcards, secrets, ``pods/exec``, workload
  creation, RBAC modification — the classic indirect privilege escalations);
* positive *and* negative cases: a denied cross-tenant call is a PASS, while a
  successful attack call is a FAIL (so "everything is broken" cannot look like
  isolation);
* credential lifecycle cases (invalid/expired/forged/replayed/revoked, gateway
  bypass) and the rotation window;
* quota accounting with an atomic reservation ledger and an oversell check for
  concurrent races (``max_oversell``, not just "429 was returned");
* abuse trials (giant tokens, slow client, retry storm, malformed input) bounded
  before they consume device resources;
* noisy-neighbour interference measured on the *victim's* SLI, with the
  normalized ratio of §8;
* telemetry redaction, audit coverage, resource recovery and the honest
  conclusion wording ("not observed within the given sample and threat model").

Nothing here authenticates, authorizes or scans anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-11"
TITLE = "多租户认证、授权、配额、Secrets、脱敏、网络隔离与滥用防护"
CLAIM_BOUNDARY = (
    "本实验通过只证明声明威胁模型内的控制有效；不等于抵抗所有侧信道、节点管理员攻击或获得安全认证。"
)

SCHEMA_VERSION = "1.0.0"

INVARIANTS: Tuple[str, ...] = rec.TENANT_INVARIANTS
CASE_KINDS: Tuple[str, ...] = rec.SECURITY_CASE_KINDS
VERDICTS: Tuple[str, ...] = rec.SECURITY_VERDICTS
SUBJECTS: Tuple[str, ...] = rec.TENANT_SUBJECTS

#: Effective requests the matrix uses (step 16).
K8S_VERBS: Tuple[str, ...] = ("get", "list", "watch", "create", "patch", "delete", "exec", "port_forward")

#: RBAC risks that must be flagged (step 17).
RBAC_RISKS: Tuple[str, ...] = (
    "WILDCARD_VERB",
    "WILDCARD_RESOURCE",
    "SECRET_READ",
    "PODS_EXEC",
    "POLE_CREATE_WORKLOAD",
    "RBAC_MODIFY",
    "NODE_READ",
    "CSR_APPROVE",
    "CLUSTER_SCOPED_BINDING",
    "TOKEN_REQUEST",
)

#: Resource kinds that participate in the static quota (step 6).
STATIC_QUOTA_RESOURCES: Tuple[str, ...] = ("cpu", "memory", "pods", "storage", "accelerators")

#: Dynamic (application-level) budgets which Kubernetes does not understand.
DYNAMIC_BUDGETS: Tuple[str, ...] = (
    "requests",
    "input_tokens",
    "output_tokens",
    "concurrency",
    "kv_bytes",
    "queue_depth",
    "model_resident_slots",
    "scale_cost_units",
)

#: Abuse cases (steps 30–32).
ABUSE_KINDS: Tuple[str, ...] = (
    "GIANT_PROMPT",
    "GIANT_MAX_TOKENS",
    "MALFORMED_JSON",
    "INVALID_ENCODING",
    "DUPLICATE_STOP",
    "SLOW_CLIENT",
    "CONNECTION_HOARD",
    "ABRUPT_DISCONNECT",
    "RETRY_STORM",
    "ERROR_AMPLIFICATION",
    "TENANT_HEADER_FORGERY",
    "DIRECT_BACKEND_BYPASS",
)

#: Audit event kinds that must exist (step 36).
AUDIT_EVENT_KINDS: Tuple[str, ...] = (
    "authn_failure",
    "authz_deny",
    "secret_access",
    "rbac_change",
    "policy_change",
    "quota_change",
    "model_switch",
    "canary_decision",
    "rollback",
    "break_glass",
    "quota_rejection",
)

#: Cases whose expected result is a *deny* (a successful call is an incident).
DENY_CASE_KINDS: Tuple[str, ...] = (
    "AUTHN",
    "AUTHZ",
    "NETWORK",
    "SECRET_EXPOSURE",
    "ARTIFACT_CACHE",
    "DEVICE_ISOLATION",
    "TELEMETRY_REDACTION",
)

#: Zero-leak wording: a negative result is bounded by sample and threat model.
NEGATIVE_RESULT_WORDING = (
    "在给定样本与威胁模型内未观察到越权/泄露；不得表述为数学意义上的绝对安全"
)


@dataclass
class ThreatModel:
    """Step 1/§5: what is in scope, what is explicitly not."""

    scope_id: str
    tenant_definition: str = ""
    attacker_capability: str = ""
    trusted_components: Tuple[str, ...] = ()
    protected_assets: Tuple[str, ...] = ()
    boundaries: Tuple[str, ...] = ()
    non_goals: Tuple[str, ...] = ()
    failure_policy: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("scope_id", "tenant_definition", "attacker_capability", "failure_policy"):
            if not getattr(self, name):
                problems.append(f"threat model requires {name!r}")
        if self.failure_policy not in ("fail_closed", "fail_open_declared"):
            problems.append(
                f"failure_policy {self.failure_policy!r} must be 'fail_closed' or an explicitly declared "
                "'fail_open_declared' with a compensating control"
            )
        if not self.protected_assets:
            problems.append("the protected assets must be enumerated")
        if not self.non_goals:
            problems.append(
                "the non-goals must be stated (e.g. host root, physical/micro-architectural side channels): "
                "otherwise the claim silently includes them"
            )
        if not self.boundaries:
            problems.append("the trust boundaries must be listed")
        if "node_admin" in self.attacker_capability and "namespace_isolation" in self.boundaries:
            problems.append(
                "a node-admin attacker is outside plain namespace isolation: the scope must say so "
                "instead of claiming isolation against it"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "tenant_definition": self.tenant_definition,
            "attacker_capability": self.attacker_capability,
            "trusted_components": list(self.trusted_components),
            "protected_assets": list(self.protected_assets),
            "boundaries": list(self.boundaries),
            "non_goals": list(self.non_goals),
            "failure_policy": self.failure_policy,
        }


@dataclass
class TenantIdentityMap:
    """Step 13: how a tenant maps to namespace/identity/model/quota."""

    tenant_id: str
    namespace: str = ""
    service_account: str = ""
    api_key_id: str = ""
    model_ids: Tuple[str, ...] = ()
    quota_profile: str = ""
    billing_subject: str = ""
    slo_profile: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("tenant_id", "namespace", "service_account", "api_key_id", "quota_profile"):
            if not getattr(self, name):
                problems.append(f"tenant mapping requires {name!r}")
        if not self.model_ids:
            problems.append(f"tenant {self.tenant_id}: the authorized model set must be explicit")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "namespace": self.namespace,
            "service_account": self.service_account,
            "api_key_id": self.api_key_id,
            "model_ids": list(self.model_ids),
            "quota_profile": self.quota_profile,
        }


def validate_tenant_map(
    tenants: Sequence[TenantIdentityMap], *, subjects: Sequence[str] = SUBJECTS
) -> Dict[str, Any]:
    """Step 5/13: deduplicated namespaces, dedicated SAs, one mapping per subject."""
    problems: List[str] = []
    for tenant in tenants:
        problems.extend(tenant.validate())
    namespaces = [tenant.namespace for tenant in tenants]
    duplicates = sorted({name for name in namespaces if namespaces.count(name) > 1})
    if duplicates:
        problems.append(f"tenants share a namespace: {duplicates} (a namespace is not a tenant by itself)")
    shared_sa = sorted({tenant.service_account for tenant in tenants})
    if len(shared_sa) < len(tenants):
        problems.append("every tenant needs its own service account (a shared identity is shared authority)")
    missing_subjects = sorted(set(subjects) - {"tenant-a-user", "tenant-b-user"} - set(
        f"tenant-{tenant.tenant_id.split('-')[-1]}-runtime" for tenant in tenants
    )) if subjects else []
    if not tenants:
        problems.append("no tenant mapping provided")
    return {
        "tenants": len(tenants),
        "namespaces": sorted(set(namespaces)),
        "problems": problems,
        "missing_subjects": missing_subjects,
        "ok": not problems,
    }


@dataclass
class RBACPermission:
    """One subject→verbs→resources→scope row of the permission graph (step 4)."""

    subject: str
    verbs: Tuple[str, ...] = ()
    resources: Tuple[str, ...] = ()
    scope: str = ""
    risks: Tuple[str, ...] = ()
    risk_reason: str = ""

    def inferred_risks(self) -> List[str]:
        risks: List[str] = []
        if "*" in self.verbs:
            risks.append("WILDCARD_VERB")
        if "*" in self.resources:
            risks.append("WILDCARD_RESOURCE")
        if "secrets" in self.resources and any(verb in self.verbs for verb in ("get", "list", "watch", "*")):
            risks.append("SECRET_READ")
        if any("pods/exec" in resource for resource in self.resources):
            risks.append("PODS_EXEC")
        if "pods" in self.resources and "create" in self.verbs:
            risks.append("POLE_CREATE_WORKLOAD")
        if any(resource.startswith("role") or resource.startswith("clusterrole") for resource in self.resources):
            if any(verb in self.verbs for verb in ("create", "patch", "update", "*")):
                risks.append("RBAC_MODIFY")
        if "nodes" in self.resources:
            risks.append("NODE_READ")
        if any("certificatesigningrequests" in resource for resource in self.resources):
            risks.append("CSR_APPROVE")
        if self.scope == "cluster":
            risks.append("CLUSTER_SCOPED_BINDING")
        if any("serviceaccounts/token" in resource for resource in self.resources):
            risks.append("TOKEN_REQUEST")
        return risks

    def validate(self, *, allow: Sequence[str] = (), tenant_subjects: Sequence[str] = ()) -> List[str]:
        problems: List[str] = []
        if not self.subject or not self.scope:
            problems.append("an RBAC row needs a subject and a scope (namespace/cluster)")
        if not self.verbs or not self.resources:
            problems.append(f"{self.subject}: verbs and resources must both be recorded")
        for risk in self.risks:
            if risk not in RBAC_RISKS:
                problems.append(f"{self.subject}: unknown risk {risk!r}")
        risky = [risk for risk in self.inferred_risks() if risk not in set(allow)]
        if risky and not self.risk_reason:
            problems.append(
                f"{self.subject}: {risky} without a documented justification/compensating control"
            )
        if tenant_subjects and self.subject in tenant_subjects:
            escalation = [risk for risk in risky if risk in ("SECRET_READ", "PODS_EXEC", "POLE_CREATE_WORKLOAD",
                                                             "RBAC_MODIFY", "CLUSTER_SCOPED_BINDING")]
            if escalation:
                problems.append(
                    f"a tenant subject holds indirect-escalation permissions: {escalation} "
                    "(workload creation/exec/secret read can escalate beyond the namespace)"
                )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "role": ",".join(self.verbs),
            "verbs": list(self.verbs),
            "resources": list(self.resources),
            "scope": self.scope,
            "risk": ",".join(self.inferred_risks()),
            "risk_reason": self.risk_reason,
        }


def rbac_graph_audit(
    rows: Sequence[RBACPermission], *, tenant_subjects: Sequence[str], allow: Sequence[str] = ()
) -> Dict[str, Any]:
    """Steps 4/17: least privilege, including the indirect escalation paths."""
    problems: List[str] = []
    graph: List[Dict[str, Any]] = []
    for row in rows:
        problems.extend(row.validate(allow=allow, tenant_subjects=tenant_subjects))
        graph.append(row.as_dict())
    risky = sorted({risk for row in rows for risk in row.inferred_risks()})
    return {
        "subjects": len(rows),
        "risks": risky,
        "graph": graph,
        "problems": problems,
        "ok": not problems,
        "note": "a role name does not describe authority: the verb/resource/scope triple does",
    }


@dataclass
class SecurityCase:
    """§9.1: one case with expected/observed result, decision point and audit link."""

    case_id: str
    kind: str
    subject: str
    action: str = ""
    resource: str = ""
    path: str = ""
    authenticated_tenant: str = ""
    claimed_tenant: str = ""
    expected: str = "deny"
    observed_status: str = ""
    decision_point: str = ""
    audit_event_id: str = ""
    latency_ms: float = 0.0
    evidence_refs: Tuple[str, ...] = ()
    verdict: str = ""

    def evaluate(self) -> str:
        """Deny-cases must be denied; allow-cases must succeed (isolation ≠ outage)."""
        if self.kind in DENY_CASE_KINDS:
            ok = self.observed_status in ("deny", "403", "401", "404", "REJECTED", "fail_closed")
        else:
            ok = self.observed_status in ("allow", "200", "429", "OK", "queued", "rejected_bounded")
        return "PASS" if ok else "FAIL"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.kind not in CASE_KINDS:
            problems.append(f"unknown security case kind {self.kind!r}")
        if self.subject not in SUBJECTS:
            problems.append(f"unknown subject {self.subject!r}")
        if not self.path:
            problems.append(f"{self.case_id}: the request path (gateway→backend) must be recorded")
        if not self.decision_point:
            problems.append(f"{self.case_id}: the authorization decision point must be recorded")
        if self.expected not in rec.DENY_EXPECTATIONS:
            problems.append(f"{self.case_id}: unknown expectation {self.expected!r}")
        if self.claimed_tenant and not self.authenticated_tenant:
            problems.append(
                f"{self.case_id}: a claimed tenant without an authenticated tenant is exactly the forgery case"
            )
        if not self.audit_event_id:
            problems.append(f"{self.case_id}: every decision must be correlatable to an audit event")
        if self.verdict and self.verdict not in VERDICTS:
            problems.append(f"{self.case_id}: unknown verdict {self.verdict!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "subject": self.subject,
            "authenticated_tenant": self.authenticated_tenant,
            "claimed_tenant": self.claimed_tenant,
            "action": self.action,
            "resource": self.resource,
            "path": self.path,
            "expected": self.expected,
            "observed_status": self.observed_status,
            "decision_point": self.decision_point,
            "audit_event_id": self.audit_event_id,
            "verdict": self.verdict or self.evaluate(),
        }


def evaluate_security_cases(
    cases: Sequence[SecurityCase], *, repetitions: int = 3
) -> Dict[str, Any]:
    """Step 38/§10: deterministic cases repeat; any unexpected allow is a failure."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    per_kind: Dict[str, Dict[str, int]] = {}
    for case in cases:
        problems.extend(case.validate())
        verdict = case.verdict or case.evaluate()
        row = case.as_dict()
        row["verdict"] = verdict
        rows.append(row)
        bucket = per_kind.setdefault(case.kind, {"pass": 0, "fail": 0})
        bucket["pass" if verdict == "PASS" else "fail"] += 1
        if verdict == "FAIL":
            problems.append(
                f"{case.case_id} ({case.kind}): expected {case.expected}, observed {case.observed_status}"
            )
    if repetitions < 3 and any(case.kind in DENY_CASE_KINDS for case in cases):
        problems.append("deterministic permission/network cases must be repeated at least 3 times")
    return {
        "rows": rows,
        "cases": len(rows),
        "per_kind": {kind: bucket for kind, bucket in sorted(per_kind.items())},
        "problems": problems,
        "ok": not problems,
        "wording": NEGATIVE_RESULT_WORDING,
    }


def positive_baseline(cases: Sequence[SecurityCase]) -> Dict[str, Any]:
    """Step 14: the legitimate paths must work, otherwise "isolation" hides an outage."""
    problems: List[str] = []
    legitimate = [
        case for case in cases if case.subject in ("tenant-a-user", "tenant-b-user", "platform-readonly")
    ]
    for case in legitimate:
        if case.evaluate() != "PASS":
            problems.append(f"{case.case_id}: the legitimate baseline failed ({case.observed_status})")
    if not legitimate:
        problems.append("no legitimate baseline case was executed")
    return {"legitimate_cases": len(legitimate), "problems": problems, "ok": not problems}


# ── quota ledger and races ───────────────────────────────────────────────


@dataclass
class QuotaProfile:
    """Step 6: the static Kubernetes quota *and* the dynamic application budget."""

    profile_id: str
    static_limits: Mapping[str, float] = field(default_factory=dict)
    dynamic_limits: Mapping[str, float] = field(default_factory=dict)
    reject_status: str = ""
    retry_semantics: str = ""
    max_replicas: int = 0
    cost_cap_units: float = 0.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.profile_id:
            problems.append("quota profile needs an id")
        missing_static = sorted(set(STATIC_QUOTA_RESOURCES) - set(self.static_limits))
        if missing_static:
            problems.append(f"static quota misses {missing_static}")
        missing_dynamic = sorted(set(DYNAMIC_BUDGETS) - set(self.dynamic_limits))
        if missing_dynamic:
            problems.append(
                f"dynamic budget misses {missing_dynamic}: Kubernetes quotas do not understand token/KV cost"
            )
        if not self.reject_status or not self.retry_semantics:
            problems.append("the reject status and retry semantics must be declared")
        if self.max_replicas <= 0:
            problems.append("a per-tenant replica ceiling is required (one tenant may not drive cluster scale)")
        if self.cost_cap_units <= 0:
            problems.append("a cost cap is required")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "static_limits": dict(sorted(self.static_limits.items())),
            "dynamic_limits": dict(sorted(self.dynamic_limits.items())),
            "reject_status": self.reject_status,
            "max_replicas": self.max_replicas,
            "cost_cap_units": self.cost_cap_units,
        }


@dataclass
class QuotaLedger:
    """Steps 28/29: reserve → commit → release with an oversell check."""

    tenant_id: str
    resource: str
    limit: float = 0.0
    reserved: float = 0.0
    committed: float = 0.0
    released: float = 0.0
    settled_actual: float = 0.0
    refunded: float = 0.0

    def balance(self) -> float:
        return self.limit - self.committed

    def oversell(self) -> float:
        return max(self.committed - self.limit, 0.0)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.resource not in STATIC_QUOTA_RESOURCES + DYNAMIC_BUDGETS:
            problems.append(f"unknown quota resource {self.resource!r}")
        if self.limit <= 0:
            problems.append(f"{self.tenant_id}/{self.resource}: the limit must be positive")
        if self.oversell() > 0:
            problems.append(
                f"{self.tenant_id}/{self.resource}: committed {self.committed} exceeds the limit "
                f"{self.limit} (oversell {self.oversell()})"
            )
        if self.reserved < 0 or self.released < 0:
            problems.append(f"{self.tenant_id}/{self.resource}: negative reservation/release")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "resource": self.resource,
            "limit": self.limit,
            "reserved": self.reserved,
            "committed": self.committed,
            "released": self.released,
            "balance": self.balance(),
            "oversell": self.oversell(),
        }


def quota_race_trials(
    trials: Sequence[Mapping[str, Any]], *, reject_status_ok: bool = True
) -> Dict[str, Any]:
    """Step 29: run the concurrent barrier test and report the *maximum* oversell."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for trial in trials:
        accepted_cost = float(trial.get("accepted_cost", 0.0))
        limit = float(trial.get("limit", 0.0))
        max_oversell = max(accepted_cost - limit, 0.0)
        rows.append(
            {
                "trial_id": trial.get("trial_id", ""),
                "concurrency": int(trial.get("concurrency", 0)),
                "gateway_replicas": int(trial.get("gateway_replicas", 1)),
                "accepted": int(trial.get("accepted", 0)),
                "rejected": int(trial.get("rejected", 0)),
                "max_oversell": max_oversell,
                "final_balance": float(trial.get("final_balance", 0.0)),
            }
        )
        if max_oversell > 0:
            problems.append(
                f"trial {trial.get('trial_id')}: oversell {max_oversell} (TOCTOU between check and reserve)"
            )
        if float(trial.get("final_balance", 0.0)) < 0:
            problems.append(f"trial {trial.get('trial_id')}: the ledger went negative")
        if int(trial.get("rejected", 0)) == 0 and accepted_cost > limit:
            problems.append(f"trial {trial.get('trial_id')}: nothing was rejected although demand exceeded the limit")
    if not reject_status_ok:
        problems.append("the rejection status code does not match the declared policy")
    return {
        "rows": rows,
        "trials": len(rows),
        "max_oversell": max((row["max_oversell"] for row in rows), default=0.0),
        "problems": problems,
        "ok": not problems,
        "note": "only 'a 429 was returned' is not evidence: the accepted cost is compared with the limit",
    }


def quota_settlement(ledgers: Sequence[QuotaLedger]) -> Dict[str, Any]:
    """Step 28: failed/cancelled requests must be refunded, not permanently charged."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for ledger in ledgers:
        problems.extend(ledger.validate())
        if ledger.settled_actual > ledger.committed:
            problems.append(
                f"{ledger.tenant_id}/{ledger.resource}: settled {ledger.settled_actual} > committed "
                f"{ledger.committed} (a cancelled flow must not be charged for output it never produced)"
            )
        rows.append(ledger.as_dict())
    return {"rows": rows, "problems": problems, "ok": not problems}


# ── abuse, noisy neighbour, telemetry ───────────────────────────────────


def abuse_trials(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Steps 30–32: reject early and bounded, before device resources are consumed."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("kind", ""))
        if kind not in ABUSE_KINDS:
            raise ConfigError(f"unknown abuse kind {kind!r}")
        bounded = bool(row.get("bounded", False))
        rejected_early = bool(row.get("rejected_before_resource", False))
        neighbor_impact = row.get("neighbor_impact", "")
        observations.append(
            {
                "trial_id": row.get("trial_id", ""),
                "kind": kind,
                "tenant_id": row.get("tenant_id", ""),
                "bounded": bounded,
                "rejected_before_resource": rejected_early,
                "neighbor_impact": neighbor_impact,
            }
        )
        if not bounded:
            problems.append(f"{kind}: the abuse case was not bounded")
        if not rejected_early:
            problems.append(
                f"{kind}: rejected only after allocating device/KV resources (a late reject is an OOM path)"
            )
        if kind in ("TENANT_HEADER_FORGERY", "DIRECT_BACKEND_BYPASS") and row.get("observed") != "deny":
            problems.append(f"{kind}: the forged/bypass attempt was not denied")
    return {"rows": observations, "trials": len(observations), "problems": problems, "ok": not problems}


def noisy_neighbor(
    *, trials: Sequence[Mapping[str, Any]], victim_metric: str, isolation_budget: float
) -> Dict[str, Any]:
    """Step 33: measure the interference on the *victim's* SLI, with percentiles."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for trial in trials:
        alone = float(trial.get("victim_alone", 0.0))
        with_attacker = float(trial.get("victim_with_attacker", 0.0))
        ratio = (with_attacker / alone) if alone else float("inf")
        percentile = str(trial.get("percentile", "p99"))
        rows.append(
            {
                "trial_id": trial.get("trial_id", ""),
                "victim_tenant": trial.get("victim_tenant", ""),
                "attacker_tenant": trial.get("attacker_tenant", ""),
                "metric": victim_metric,
                "victim_alone": alone,
                "victim_with_attacker": with_attacker,
                "interference_ratio": ratio,
                "percentile": percentile,
            }
        )
        if ratio - 1.0 > isolation_budget:
            problems.append(
                f"trial {trial.get('trial_id')}: victim {victim_metric} degraded by "
                f"{(ratio - 1.0) * 100:.1f}% > the isolation budget {isolation_budget * 100:.1f}% "
                "(a global throughput average would hide this)"
            )
    if not trials:
        problems.append("no noisy-neighbour trial was executed")
    return {
        "rows": rows,
        "trials": len(rows),
        "max_interference_ratio": max((row["interference_ratio"] for row in rows), default=0.0),
        "problems": problems,
        "ok": not problems,
    }


def telemetry_redaction_cases(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 35: canary values must not appear in any telemetry surface for any subject."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for row in rows:
        signal_class = str(row.get("signal_class", ""))
        if signal_class not in rec.SIGNAL_CLASSES + ("artifacts", "error_response", "diagnostic_bundle"):
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
                f"{signal_class}: canary {row.get('canary_id')} visible to {row.get('subject')} at "
                f"{row.get('location')}"
            )
        if row.get("error_path") and not row.get("checked_error_path"):
            problems.append(
                f"{signal_class}: the error/rejection path was not checked (stacks leak more than successes)"
            )
    return {"rows": observations, "problems": problems, "ok": not problems}


@dataclass
class AuditCoverage:
    """Step 36: each security-relevant event must exist, be complete and correlatable."""

    event_kind: str
    emitted: bool = False
    fields_complete: bool = False
    lag_s: float = 0.0
    correlatable: bool = False
    actor_recorded: bool = False
    tenant_recorded: bool = False

    def validate(self, *, max_lag_s: float) -> List[str]:
        problems: List[str] = []
        if self.event_kind not in AUDIT_EVENT_KINDS:
            problems.append(f"unknown audit event kind {self.event_kind!r}")
        if not self.emitted:
            problems.append(f"{self.event_kind}: no audit event was emitted")
        if not self.fields_complete:
            problems.append(f"{self.event_kind}: the event misses subject/tenant/action/object/result fields")
        if not self.correlatable:
            problems.append(f"{self.event_kind}: the event cannot be correlated to the request/decision")
        if not self.actor_recorded or not self.tenant_recorded:
            problems.append(f"{self.event_kind}: actor and tenant must both be recorded for attribution")
        if self.lag_s > max_lag_s:
            problems.append(f"{self.event_kind}: audit lag {self.lag_s}s exceeds the policy window {max_lag_s}s")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_kind": self.event_kind,
            "emitted": self.emitted,
            "fields_complete": self.fields_complete,
            "lag_s": self.lag_s,
            "correlatable": self.correlatable,
        }


def audit_coverage_report(rows: Sequence[AuditCoverage], *, max_lag_s: float) -> Dict[str, Any]:
    problems: List[str] = []
    for row in rows:
        problems.extend(row.validate(max_lag_s=max_lag_s))
    missing = sorted(set(AUDIT_EVENT_KINDS) - {row.event_kind for row in rows})
    if missing:
        problems.append(f"audit kinds without coverage: {missing}")
    return {"rows": [row.as_dict() for row in rows], "missing_kinds": missing, "problems": problems,
            "ok": not problems}


def rotation_recovery(
    *, rows: Sequence[Mapping[str, Any]], policy_window_s: float
) -> Dict[str, Any]:
    """Steps 23/37: old credentials must die inside the window; test state must recover."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("kind", ""))
        if kind not in ("revocation", "rotation", "resource_recovery", "cleanup"):
            raise ConfigError(f"unknown rotation/recovery case {kind!r}")
        observed_window = row.get("observed_window_s")
        observations.append(
            {
                "case_id": row.get("case_id", ""),
                "kind": kind,
                "pre_value": row.get("pre_value", ""),
                "post_value": row.get("post_value", ""),
                "observed_window_s": observed_window,
                "consistent": bool(row.get("consistent", False)),
            }
        )
        if kind in ("revocation", "rotation"):
            if observed_window is None:
                problems.append(f"{row.get('case_id')}: the revocation window was not measured")
            elif float(observed_window) > policy_window_s:
                problems.append(
                    f"{row.get('case_id')}: revocation took {observed_window}s > the policy window "
                    f"{policy_window_s}s (a revoked credential must fail inside the window)"
                )
        if not row.get("consistent", False):
            problems.append(f"{row.get('case_id')}: post-test state/ledger is not consistent with the pre-state")
    return {"rows": observations, "problems": problems, "ok": not problems}


def invariant_verdicts(
    *, evidence: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 38: a verdict per invariant; a missing invariant is ``NOT_RUN``, never PASS."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for invariant in INVARIANTS:
        entry = evidence.get(invariant)
        if entry is None:
            rows.append({"invariant": invariant, "verdict": "NOT_RUN", "reason": "no evidence"})
            problems.append(f"{invariant}: no evidence (NOT_RUN, not PASS)")
            continue
        verdict = str(entry.get("verdict", "NOT_RUN"))
        if verdict not in VERDICTS:
            raise ConfigError(f"{invariant}: unknown verdict {verdict!r}")
        rows.append({"invariant": invariant, "verdict": verdict, "reason": entry.get("reason", "")})
        if verdict != "PASS":
            problems.append(f"{invariant}: {verdict} — {entry.get('reason', '')}")
    return {
        "rows": rows,
        "invariants": len(rows),
        "passed": [row["invariant"] for row in rows if row["verdict"] == "PASS"],
        "problems": problems,
        "ok": not problems,
        "wording": NEGATIVE_RESULT_WORDING,
    }


def multitenant_verdict(
    *,
    threat_model: ThreatModel,
    tenant_map: Mapping[str, Any],
    rbac: Mapping[str, Any],
    cases: Mapping[str, Any],
    baseline: Mapping[str, Any],
    quota: Mapping[str, Any],
    races: Mapping[str, Any],
    abuse: Mapping[str, Any],
    neighbor: Mapping[str, Any],
    redaction: Mapping[str, Any],
    audit: Mapping[str, Any],
    rotation: Mapping[str, Any],
    invariants: Mapping[str, Any],
    clean_redeploy_consistent: bool,
) -> Dict[str, Any]:
    """Steps 38/§11: the conditional capability statement the stage may publish."""
    problems: List[str] = []
    for name, axis in (
        ("tenant_map", tenant_map),
        ("rbac_graph", rbac),
        ("security_cases", cases),
        ("positive_baseline", baseline),
        ("quota", quota),
        ("quota_race", races),
        ("abuse", abuse),
        ("noisy_neighbor", neighbor),
        ("redaction", redaction),
        ("audit", audit),
        ("rotation_recovery", rotation),
        ("invariants", invariants),
    ):
        if not axis.get("ok"):
            problems.append(f"{name}: " + "; ".join(axis.get("problems", []) or ["(no detail)"]))
    problems.extend(threat_model.validate())
    if not clean_redeploy_consistent:
        problems.append(
            "the second clean deployment did not reach the same verdict: a security result must be "
            "reproducible, not incidental"
        )
    return {
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "scope": threat_model.as_dict(),
        "wording": NEGATIVE_RESULT_WORDING,
        "unsupported_claims": [
            "absolute security",
            "resistance to node-admin or hard multi-tenant adversaries",
            "host/GPU micro-architectural side-channel isolation",
            "any certification or compliance statement",
        ],
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结租户定义、范围和不支持项", ("security:ThreatModel",)),
    (2, "绘制资产、信任边界和数据流", ("security:ThreatModel", "observability:SemanticConvention")),
    (3, "冻结身份签发、传播、过期与撤销策略", ("security:CredentialPolicy",)),
    (4, "导出 RBAC 权限图并做最小权限审查", ("security:RBACPermission", "security:rbac_graph_audit")),
    (5, "检查 Namespace、ServiceAccount 与自动挂载策略", ("security:validate_tenant_map",)),
    (6, "冻结静态配额和动态业务预算", ("security:QuotaProfile", "security:STATIC_QUOTA_RESOURCES")),
    (7, "建立默认拒绝网络策略和最小允许清单", ("security:NetworkPolicySpec",)),
    (8, "冻结 Pod 运行时安全基线", ("security:PodSecurityBaseline", "supply_chain:RuntimeSecurityContext")),
    (9, "冻结 Secret 生命周期和泄露面", ("security:SecretLifecycle",)),
    (10, "冻结模型、适配器、缓存和存储命名空间", ("artifacts:CacheKey", "security:ArtifactNamespace")),
    (11, "冻结设备、分区和共享策略", ("scheduling:PlacementPlan", "scheduling:SHARING_MODES")),
    (12, "定义遥测访问、脱敏和审计字段", ("observability:SemanticConvention", "security:AuditCoverage")),
    (13, "从清洁环境创建 A、B、只读运维和攻击者主体", ("security:TenantIdentityMap", "deployment:NamespacePolicy")),
    (14, "运行每个主体的合法正向基线", ("security:positive_baseline",)),
    (15, "执行无效、过期、伪造、重放和撤销凭据测试", ("security:CredentialCase", "security:rotation_recovery")),
    (16, "穷举关键 Kubernetes API 越权动作", ("security:SecurityCase", "security:K8S_VERBS")),
    (17, "执行 ServiceAccount 与间接提权测试", ("security:rbac_graph_audit", "security:RBAC_RISKS")),
    (18, "尝试篡改 Namespace、Pod 与路由标签", ("security:SecurityCase", "scheduling:validate_label_provenance")),
    (19, "验证默认拒绝的真实网络效果", ("security:NetworkCase",)),
    (20, "验证允许网络最小且业务仍可运行", ("security:NetworkPolicySpec",)),
    (21, "验证网络插件、DNS 和服务网格的实际执行路径", ("security:NetworkPolicySpec", "security:NetworkCase")),
    (22, "搜索 Secret 在进程和文件系统中的暴露", ("security:secret_exposure_scan",)),
    (23, "执行 Secret 轮换、撤销和故障恢复", ("security:rotation_recovery",)),
    (24, "执行模型仓库、缓存和临时文件跨租户访问测试", ("security:SecurityCase", "artifacts:CacheKey")),
    (25, "验证模型身份、别名与路由不会串租户", ("security:SecurityCase", "supply_chain:verify_attestation_closure")),
    (26, "验证容器内设备可见性与跨租户设备访问", ("scheduling:unauthorized_device_access",)),
    (27, "测量共享加速器下的性能干扰并限定安全声明", ("security:noisy_neighbor",)),
    (28, "验证正常负载下的配额计量与拒绝语义", ("security:QuotaLedger", "security:quota_settlement")),
    (29, "执行并发配额竞态和超卖测试", ("security:quota_race_trials",)),
    (30, "执行超大输入、输出和非法参数滥用测试", ("security:abuse_trials", "capacity:admit")),
    (31, "执行慢客户端、流式连接囤积和主动断连测试", ("security:abuse_trials", "lifecycle:slow_client_policy_check")),
    (32, "执行重试风暴和错误放大测试", ("security:abuse_trials", "faults:retry_storm_check")),
    (33, "执行 noisy-neighbor、公平调度和优先级测试", ("security:noisy_neighbor", "serving:slo.SLOSpec")),
    (34, "验证自动扩缩容的租户归因和成本上限", ("security:QuotaProfile", "autoscaling:cost_accounting")),
    (35, "验证日志、指标、Trace、Profile 与错误响应的隔离和脱敏", ("security:telemetry_redaction_cases",)),
    (36, "验证审计事件的完整性、时序和可追责性", ("security:audit_coverage_report", "security:AUDIT_EVENT_KINDS")),
    (37, "恢复、清理并验证状态一致性", ("security:rotation_recovery", "faults:validate_state_recovery")),
    (38, "重复、统计、裁决并限定声明", ("security:invariant_verdicts", "security:multitenant_verdict")),
)


@dataclass
class CredentialPolicy:
    """Step 3: credential type, issuer, audience, TTL, revocation window."""

    policy_id: str
    credential_type: str = ""
    issuer: str = ""
    audience: str = ""
    tenant_claim: str = ""
    ttl_s: int = 0
    clock_skew_s: int = 0
    revocation_window_s: int = 0
    key_rotation_s: int = 0
    service_identity: str = ""
    gateway_bypass_allowed: bool = False
    clock_skew_cases: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("policy_id", "credential_type", "issuer", "audience", "tenant_claim", "service_identity"):
            if not getattr(self, name):
                problems.append(f"credential policy requires {name!r}")
        if self.ttl_s <= 0:
            problems.append("a credential TTL is required (static long-lived keys must be an explicit exception)")
        if self.revocation_window_s <= 0:
            problems.append("a revocation window must be declared and measurable")
        if self.gateway_bypass_allowed:
            problems.append(
                "bypassing the gateway must not be allowed: the backend would trust client-supplied tenant context"
            )
        for case in ("expired", "not_yet_valid", "forged_claim"):
            if case not in self.clock_skew_cases:
                problems.append(f"the credential test set must include the {case!r} case")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "credential_type": self.credential_type,
            "issuer": self.issuer,
            "audience": self.audience,
            "tenant_claim": self.tenant_claim,
            "ttl_s": self.ttl_s,
            "clock_skew_s": self.clock_skew_s,
            "revocation_window_s": self.revocation_window_s,
            "gateway_bypass_allowed": self.gateway_bypass_allowed,
        }


@dataclass
class CredentialCase:
    """Step 15: one credential lifecycle case."""

    case_id: str
    kind: str
    expected: str = "deny"
    observed: str = ""
    decision_point: str = ""
    latency_ms: float = 0.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        allowed = ("missing", "bad_signature", "wrong_audience", "expired", "not_yet_valid", "replayed",
                   "revoked", "forged_tenant_claim", "valid")
        if self.kind not in allowed:
            problems.append(f"unknown credential case kind {self.kind!r}")
        expected_verdict = "deny" if self.kind != "valid" else "allow"
        if self.expected != expected_verdict:
            problems.append(
                f"{self.case_id}: {self.kind} must expect {expected_verdict!r} (a valid credential must work "
                "so that 'everything is denied' cannot look like security)"
            )
        if not self.decision_point:
            problems.append(f"{self.case_id}: the enforcement point must be recorded")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "expected": self.expected,
            "observed": self.observed,
            "decision_point": self.decision_point,
        }


@dataclass
class NetworkPolicySpec:
    """Steps 7/19–21: default-deny plus the minimal allow list."""

    policy_id: str
    cni: str = ""
    enforcement_points: Tuple[str, ...] = ()
    default_ingress_deny: bool = False
    default_egress_deny: bool = False
    allow_rules: Tuple[str, ...] = ()
    cidr_exceptions: Tuple[str, ...] = ()
    metadata_service_blocked: bool = False
    tested_paths: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("policy_id", "cni"):
            if not getattr(self, name):
                problems.append(f"network policy requires {name!r}")
        if not (self.default_ingress_deny and self.default_egress_deny):
            problems.append("strict isolation starts from default-deny for both directions")
        if not self.metadata_service_blocked:
            problems.append("the cloud metadata endpoint must be blocked (credential theft path)")
        for exception in self.cidr_exceptions:
            if exception.startswith("0.0.0.0") or exception.endswith("/0"):
                problems.append(f"a 0.0.0.0/0 exception defeats the policy: {exception}")
        for path in ("direct_pod_ip", "service_dns", "node_port", "management_port", "cross_node"):
            if path not in self.tested_paths:
                problems.append(
                    f"the {path} path must be tested: a policy that only blocks DNS is not a network isolation"
                )
        if not self.enforcement_points:
            problems.append("the CNI/mesh enforcement points must be recorded")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "cni": self.cni,
            "default_ingress_deny": self.default_ingress_deny,
            "default_egress_deny": self.default_egress_deny,
            "allow_rules": list(self.allow_rules),
            "tested_paths": list(self.tested_paths),
        }


@dataclass
class NetworkCase:
    """Steps 19–21: the packet-level result, not the YAML review."""

    case_id: str
    from_tenant: str
    target: str
    path: str
    expected: str = "deny"
    observed: str = ""
    enforced_by: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.expected not in rec.DENY_EXPECTATIONS:
            problems.append(f"{self.case_id}: unknown expectation {self.expected!r}")
        if not self.enforced_by:
            problems.append(f"{self.case_id}: the enforcing component must be recorded")
        if self.observed == self.expected and self.path not in (
            "direct_pod_ip", "service_dns", "node_port", "management_port", "cross_node", "external_egress",
        ):
            problems.append(f"{self.case_id}: unknown network path {self.path!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "from_tenant": self.from_tenant,
            "target": self.target,
            "path": self.path,
            "expected": self.expected,
            "observed": self.observed,
        }


def secret_exposure_scan(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 22: mounted files, env, /proc, temp dirs, logs, events, dumps, bundles."""
    surfaces = (
        "mounted_file", "file_mode", "environment", "process_args", "proc_fs", "temp_dir",
        "logs", "events", "crash_dump", "support_bundle",
    )
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for row in rows:
        surface = str(row.get("surface", ""))
        if surface not in surfaces:
            raise ConfigError(f"unknown secret exposure surface {surface!r}")
        leaked = bool(row.get("leaked", False))
        unauthorized = bool(row.get("unauthorized_subject", False))
        observations.append(
            {
                "surface": surface,
                "canary_id": row.get("canary_id", ""),
                "leaked": leaked,
                "unauthorized_subject": unauthorized,
                "detail": row.get("detail", ""),
            }
        )
        if leaked and unauthorized:
            problems.append(f"{surface}: the canary secret is reachable by an unauthorized subject")
    missing = sorted(set(surfaces) - {row["surface"] for row in observations})
    if missing:
        problems.append(f"exposure surfaces not scanned: {missing}")
    return {
        "rows": observations,
        "problems": problems,
        "ok": not problems,
        "wording": "canary secrets only; real credentials must never be used in a scan",
    }


@dataclass
class PodSecurityBaseline:
    """Step 8: non-root, read-only rootfs, dropped capabilities, no host namespaces."""

    baseline_id: str
    run_as_non_root: bool = False
    read_only_rootfs: bool = False
    capabilities_dropped: Tuple[str, ...] = ()
    seccomp_profile: str = ""
    allow_privilege_escalation: bool = True
    host_path_volumes: Tuple[str, ...] = ()
    host_network: bool = False
    host_pid: bool = False
    host_ipc: bool = False
    device_exceptions: Tuple[str, ...] = ()
    exception_justification: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.baseline_id:
            problems.append("pod security baseline needs an id")
        if not self.run_as_non_root:
            problems.append("the workload must run as non-root")
        if not self.read_only_rootfs:
            problems.append("a read-only root filesystem is the baseline")
        if self.allow_privilege_escalation:
            problems.append("allowPrivilegeEscalation must be false")
        if "ALL" not in self.capabilities_dropped:
            problems.append("capabilities must be dropped with drop: [ALL] and re-added item by item")
        for name, enabled in (("host_network", self.host_network), ("host_pid", self.host_pid),
                              ("host_ipc", self.host_ipc)):
            if enabled:
                problems.append(f"{name} must be off outside a documented exception")
        if self.host_path_volumes:
            problems.append(f"hostPath volumes: {list(self.host_path_volumes)} (not a tenant boundary)")
        if self.device_exceptions and not self.exception_justification:
            problems.append("device/capability exceptions need a documented justification (never 'privileged')")
        if not self.seccomp_profile:
            problems.append("a seccomp profile is part of the runtime baseline")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "run_as_non_root": self.run_as_non_root,
            "read_only_rootfs": self.read_only_rootfs,
            "capabilities_dropped": list(self.capabilities_dropped),
            "seccomp_profile": self.seccomp_profile,
            "allow_privilege_escalation": self.allow_privilege_escalation,
            "device_exceptions": list(self.device_exceptions),
        }


@dataclass
class SecretLifecycle:
    """Step 9: creation, encryption at rest, mounts, rotation, revocation, audit."""

    lifecycle_id: str
    secret_kind: str = ""
    created_by: str = ""
    encrypted_at_rest: bool = False
    mounted_as: str = ""
    rotation_days: int = 0
    revocation_window_s: int = 0
    audit_enabled: bool = False
    destroyed_after_use: bool = False
    in_image: bool = False
    in_command_line: bool = False
    in_environment_dump: bool = False
    in_logs: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("lifecycle_id", "secret_kind", "created_by", "mounted_as"):
            if not getattr(self, name):
                problems.append(f"secret lifecycle requires {name!r}")
        if not self.encrypted_at_rest:
            problems.append("encryption at rest must be verified (base64 is not encryption)")
        if self.rotation_days <= 0:
            problems.append("a rotation period is required")
        if not self.audit_enabled:
            problems.append("secret access must be audited")
        for name in ("in_image", "in_command_line", "in_environment_dump", "in_logs"):
            if getattr(self, name):
                problems.append(f"the secret must not appear in the {name[3:]}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lifecycle_id": self.lifecycle_id,
            "secret_kind": self.secret_kind,
            "encrypted_at_rest": self.encrypted_at_rest,
            "mounted_as": self.mounted_as,
            "rotation_days": self.rotation_days,
            "revocation_window_s": self.revocation_window_s,
            "audit_enabled": self.audit_enabled,
        }


@dataclass
class ArtifactNamespace:
    """Step 10: tenant-scoped object-store prefixes, cache keys and ownership."""

    tenant_id: str
    object_store_prefix: str = ""
    cache_key_prefix: str = ""
    directory_uid: int = 0
    directory_gid: int = 0
    signing_policy: str = ""
    model_alias_map: Mapping[str, str] = field(default_factory=dict)
    gc_owner: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("tenant_id", "object_store_prefix", "cache_key_prefix", "signing_policy", "gc_owner"):
            if not getattr(self, name):
                problems.append(f"artifact namespace requires {name!r}")
        if self.object_store_prefix.strip("/") in ("", "*"):
            problems.append("a tenant prefix may not be empty or a wildcard")
        if self.tenant_id not in self.object_store_prefix:
            problems.append(
                f"the object-store prefix {self.object_store_prefix!r} does not contain the tenant id: "
                "a shared prefix plus a user-controlled name is not isolation"
            )
        for alias, target in self.model_alias_map.items():
            if self.tenant_id not in target and "shared" not in target:
                problems.append(
                    f"alias {alias!r} resolves to {target!r} outside the tenant namespace "
                    "(a global model_name→path map can serve the wrong tenant)"
                )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "object_store_prefix": self.object_store_prefix,
            "cache_key_prefix": self.cache_key_prefix,
            "model_alias_map": dict(sorted(self.model_alias_map.items())),
            "gc_owner": self.gc_owner,
        }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the multi-tenant contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    model = ThreatModel(
        scope_id="scope-1", tenant_definition="namespace + API key + billing subject",
        attacker_capability="authenticated tenant with API access and pod-creation rights inside its namespace",
        trusted_components=("api-server", "identity-provider", "gateway"),
        protected_assets=("model weights", "prompt/completion", "secrets", "quota", "audit log"),
        boundaries=("namespace_isolation", "data_plane_network", "telemetry"),
        non_goals=("host root", "GPU micro-architectural side channels"),
        failure_policy="fail_closed",
    )
    checks["threat_model_valid"] = model.validate() == []

    naive = ThreatModel(scope_id="s2", tenant_definition="namespace", attacker_capability="node_admin",
                        failure_policy="fail_open", boundaries=("namespace_isolation",))
    checks["naive_scope_rejected"] = len(naive.validate()) >= 3

    tenants = [
        TenantIdentityMap(tenant_id="tenant-a", namespace="hqsb-tenant-a", service_account="sa-a",
                          api_key_id="key-a", model_ids=("qwen3-1.7b",), quota_profile="standard"),
        TenantIdentityMap(tenant_id="tenant-b", namespace="hqsb-tenant-b", service_account="sa-b",
                          api_key_id="key-b", model_ids=("qwen3-1.7b",), quota_profile="standard"),
    ]
    checks["tenant_map_valid"] = validate_tenant_map(tenants)["ok"] is True

    shared = [
        TenantIdentityMap(tenant_id="tenant-a", namespace="shared", service_account="sa", api_key_id="k1",
                          model_ids=("m",), quota_profile="p"),
        TenantIdentityMap(tenant_id="tenant-b", namespace="shared", service_account="sa", api_key_id="k2",
                          model_ids=("m",), quota_profile="p"),
    ]
    checks["shared_namespace_rejected"] = validate_tenant_map(shared)["ok"] is False

    rbac = rbac_graph_audit(
        rows=[
            RBACPermission(subject="tenant-a-runtime", verbs=("create",), resources=("pods",),
                           scope="namespace"),
            RBACPermission(subject="tenant-a-runtime", verbs=("get",), resources=("secrets",),
                           scope="namespace", risk_reason=""),
        ],
        tenant_subjects=("tenant-a-runtime",),
    )
    checks["escalation_detected"] = rbac["ok"] is False

    denied = SecurityCase(
        case_id="MT-AUTHZ-001", kind="AUTHZ", subject="tenant-a-user", action="get",
        resource="tenant-b/model/finance-adapter", path="gateway-to-backend",
        authenticated_tenant="tenant-a", claimed_tenant="tenant-b", expected="deny",
        observed_status="403", decision_point="application-authorizer", audit_event_id="audit-1",
    )
    case_report = evaluate_security_cases([denied])
    checks["deny_case_passes"] = case_report["ok"] is True

    leaked_case = SecurityCase(
        case_id="MT-AUTHZ-002", kind="AUTHZ", subject="tenant-a-user", action="get",
        resource="tenant-b/model", path="gateway-to-backend", authenticated_tenant="tenant-a",
        expected="deny", observed_status="200", decision_point="application-authorizer",
        audit_event_id="audit-2",
    )
    checks["successful_attack_is_failure"] = evaluate_security_cases([leaked_case])["ok"] is False

    ledger = QuotaLedger(tenant_id="tenant-a", resource="output_tokens", limit=1000.0,
                         reserved=1000.0, committed=1200.0)
    checks["oversell_detected"] = ledger.oversell() == 200.0

    race = quota_race_trials(
        [{"trial_id": "t1", "concurrency": 32, "gateway_replicas": 3, "accepted": 12,
          "rejected": 20, "accepted_cost": 1200.0, "limit": 1000.0, "final_balance": -200.0}]
    )
    checks["race_oversell_reported"] = race["max_oversell"] == 200.0 and race["ok"] is False

    neighbor = noisy_neighbor(
        trials=[{"trial_id": "n1", "victim_tenant": "tenant-b", "attacker_tenant": "tenant-a",
                 "victim_alone": 1.0, "victim_with_attacker": 1.8, "percentile": "p99"}],
        victim_metric="ttft_seconds", isolation_budget=0.25,
    )
    checks["noisy_neighbor_flagged"] = neighbor["ok"] is False

    invariants = invariant_verdicts(
        evidence={"MT-I01": {"verdict": "PASS"}, "MT-I02": {"verdict": "FAIL", "reason": "read succeeded"}}
    )
    checks["invariant_verdicts_incomplete"] = invariants["ok"] is False and "MT-I12" not in invariants["passed"]
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未扫描任何集群、未使用任何真实凭据",
    }