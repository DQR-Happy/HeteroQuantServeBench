"""Backend registry, hard feasibility filters and scoring (E08-06).

Feasibility precedes optimisation.  A candidate must pass the ordered hard
filters — model/revision/tokenizer/template, precision/quality, context,
protocol features, device/parallel, readiness/model epoch, health, and
deadline/SLO feasibility — *before* any score is computed.  Mixing the two into
one weighted sum is how a "high scoring" but incompatible Backend gets selected.

Every decision stores the candidate set, each exclusion reason, the telemetry
age, the feature values and their normalisation, the score components, the tie
break, the selected ``(instance_id, model_epoch, route_epoch)`` and the actual
execution the request ended up on: selected-vs-actual is checked, not assumed.
Missing telemetry is never treated as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.serving.circuit import CircuitBreaker, HealthState

#: The ordered hard filters (details README §18 / E08-06 §5).
HARD_FILTER_ORDER: Tuple[str, ...] = (
    "model_alias_resolve",
    "artifact_revision_tokenizer_template",
    "precision_quality_tenant_policy",
    "context_input_output_feasibility",
    "protocol_feature_capability",
    "device_parallel_constraints",
    "readiness_model_epoch",
    "circuit_health",
    "deadline_slo_envelope",
)

#: Fallback strength ladder (§10): stronger degradation needs explicit permission.
FALLBACK_LADDER: Tuple[str, ...] = (
    "same_instance_retry",
    "same_artifact_same_quality_other_instance",
    "approved_alternate_precision",
    "alternate_model_degraded_quality",
    "fail_closed_reject",
)

#: Fallback levels that require the request/SLO to allow the degradation.
PERMISSION_REQUIRED_LEVELS: Tuple[str, ...] = (
    "approved_alternate_precision",
    "alternate_model_degraded_quality",
)


@dataclass(frozen=True)
class Telemetry:
    """One Backend's telemetry sample with its age (never a bare number)."""

    monotonic_ns: int
    queue_depth: float = 0.0
    inflight_requests: float = 0.0
    inflight_tokens: float = 0.0
    kv_bytes: float = 0.0
    cache_saved_prefill_ms: float = 0.0
    error_rate: float = 0.0

    def age_ms(self, now_ns: int) -> float:
        return max(0.0, (now_ns - self.monotonic_ns) / 1e6)

    def as_dict(self, now_ns: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "monotonic_ns": self.monotonic_ns,
            "queue_depth": self.queue_depth,
            "inflight_requests": self.inflight_requests,
            "inflight_tokens": self.inflight_tokens,
            "kv_bytes": self.kv_bytes,
            "cache_saved_prefill_ms": self.cache_saved_prefill_ms,
            "error_rate": self.error_rate,
        }
        if now_ns is not None:
            payload["age_ms"] = self.age_ms(now_ns)
        return payload


@dataclass
class BackendRecord:
    """One registered Backend instance (everything routing may look at)."""

    instance_id: str
    adapter_version: str
    runtime_commit: str
    hardware: str
    device: str
    parallel_degree: int
    model_identity: Mapping[str, Any]
    precision: str
    quality_class: str
    model_aliases: Tuple[str, ...] = ()
    quant_artifact_hash: str = ""
    adapter_hash: str = ""
    tokenizer_id: str = ""
    chat_template_hash: str = ""
    capability: Mapping[str, str] = field(default_factory=dict)
    features: Mapping[str, bool] = field(default_factory=dict)
    max_context_tokens: int = 32768
    model_epoch: str = ""
    config_generation: int = 0  # 0 = "not tied to a generation" (accepts any update)
    ready: bool = False
    healthy: bool = True
    health: Optional[HealthState] = None
    capacity_envelope_ms: Mapping[str, float] = field(default_factory=dict)
    cost_units: float = 0.0
    energy_units: float = 0.0
    telemetry: Optional[Telemetry] = None
    breaker: Optional[CircuitBreaker] = None
    route_epoch: str = ""

    def model_id(self) -> str:
        return str(self.model_identity.get("model_id", ""))

    def revision(self) -> str:
        return str(self.model_identity.get("revision", ""))

    def effective_health(self) -> HealthState:
        if self.health is not None:
            return self.health
        return HealthState(
            process_alive=True,
            adapter_ready=self.ready,
            model_epoch_warm=self.ready,
            capability_probe_fresh=self.telemetry is not None,
            transient_overload=not self.healthy,
            circuit_state=self.breaker.state if self.breaker else "CLOSED",
            quarantined=bool(self.breaker and self.breaker.quarantined),
            model_epoch=self.model_epoch,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "adapter_version": self.adapter_version,
            "runtime_commit": self.runtime_commit,
            "hardware": self.hardware,
            "device": self.device,
            "parallel_degree": self.parallel_degree,
            "model_identity": dict(self.model_identity),
            "model_aliases": list(self.model_aliases),
            "precision": self.precision,
            "quality_class": self.quality_class,
            "quant_artifact_hash": self.quant_artifact_hash,
            "tokenizer_id": self.tokenizer_id,
            "chat_template_hash": self.chat_template_hash,
            "capability": dict(self.capability),
            "features": dict(self.features),
            "max_context_tokens": self.max_context_tokens,
            "model_epoch": self.model_epoch,
            "config_generation": self.config_generation,
            "ready": self.ready,
            "healthy": self.healthy,
            "health": self.effective_health().as_dict(),
            "cost_units": self.cost_units,
            "energy_units": self.energy_units,
            "telemetry": self.telemetry.as_dict() if self.telemetry else None,
            "route_epoch": self.route_epoch,
        }


@dataclass
class RegistrySnapshot:
    """An immutable view of the registry at one generation."""

    generation: int
    records: Mapping[str, BackendRecord]
    created_ns: int = 0

    def ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.records))

    def get(self, instance_id: str) -> BackendRecord:
        if instance_id not in self.records:
            raise ConfigError(f"unknown instance id {instance_id!r} in this snapshot")
        return self.records[instance_id]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "created_ns": self.created_ns,
            "instances": {name: record.as_dict() for name, record in sorted(self.records.items())},
        }


class BackendRegistry:
    """Atomic generation updates: old and new model fields never mix."""

    def __init__(self) -> None:
        self._generation = 0
        self._records: Dict[str, BackendRecord] = {}

    @property
    def generation(self) -> int:
        return self._generation

    def register(self, record: BackendRecord) -> RegistrySnapshot:
        if record.instance_id in self._records:
            raise ConfigError(
                f"duplicate instance id {record.instance_id!r}; two names pointing at one "
                "handle would fake a second Backend (E08-06 §14)"
            )
        if not record.model_epoch:
            raise ConfigError("a registered instance needs a model epoch")
        if not record.ready:
            # registering an unready instance is allowed (it exists, it is just
            # not routable yet) but the reason must be visible
            record.health = record.effective_health()
        self._records[record.instance_id] = record
        self._generation += 1
        return self.snapshot()

    def replace_generation(
        self, records: Sequence[BackendRecord], *, generation: int
    ) -> RegistrySnapshot:
        """All-or-nothing update; a mixed generation is refused."""
        if generation <= self._generation:
            raise ConfigError(
                f"registry generation must increase: got {generation}, current {self._generation}"
            )
        seen: Dict[str, BackendRecord] = {}
        for record in records:
            if record.config_generation not in (generation, 0):
                raise ConfigError(
                    f"{record.instance_id}: record generation {record.config_generation} "
                    f"does not match the update generation {generation}; a partially "
                    "applied update would mix model fields (E08-06 §4)"
                )
            if record.instance_id in seen:
                raise ConfigError(f"duplicate instance id {record.instance_id!r} in the update")
            seen[record.instance_id] = record
        self._records = seen
        self._generation = generation
        return self.snapshot()

    def snapshot(self) -> RegistrySnapshot:
        return RegistrySnapshot(generation=self._generation, records=dict(self._records))

    def get(self, instance_id: str) -> BackendRecord:
        if instance_id not in self._records:
            raise ConfigError(f"unknown instance id {instance_id!r}")
        return self._records[instance_id]


# ── hard filters ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteRequest:
    """What the router knows about the request (no future information)."""

    request_id: str
    model_alias: str
    model_identity: Mapping[str, Any]
    precision: str
    quality_class: str
    prompt_tokens: int
    reserved_output_tokens: int
    stream: bool
    cancel: bool = True
    logprobs: bool = False
    prefix_cache: bool = False
    tenant: str = "default"
    allow_alternate_precision: bool = False
    allow_alternate_model: bool = False
    deadline_ns: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_alias": self.model_alias,
            "precision": self.precision,
            "quality_class": self.quality_class,
            "prompt_tokens": self.prompt_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "stream": self.stream,
            "cancel": self.cancel,
            "logprobs": self.logprobs,
            "prefix_cache": self.prefix_cache,
            "tenant": self.tenant,
            "allow_alternate_precision": self.allow_alternate_precision,
            "allow_alternate_model": self.allow_alternate_model,
            "deadline_ns": self.deadline_ns,
        }


@dataclass(frozen=True)
class FilterOutcome:
    candidate_id: str
    feasible: bool
    reasons: Tuple[str, ...]
    failed_stage: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "feasible": self.feasible,
            "reasons": list(self.reasons),
            "failed_stage": self.failed_stage,
        }


def _check_record(
    record: BackendRecord,
    request: RouteRequest,
    *,
    now_ns: int,
    telemetry_ttl_ms: float,
    hard_filter_order: Sequence[str],
) -> FilterOutcome:
    reasons: List[str] = []
    stage = ""
    health = record.effective_health()
    checks: Mapping[str, Tuple[bool, str]] = {
        "model_alias_resolve": (
            request.model_alias in record.model_aliases,
            f"alias {request.model_alias!r} is not served by {record.instance_id}",
        ),
        "artifact_revision_tokenizer_template": (
            str(record.model_identity.get("model_manifest_sha256", ""))
            == str(request.model_identity.get("model_manifest_sha256", "")),
            "model manifest/revision/tokenizer/template mismatch",
        ),
        "precision_quality_tenant_policy": (
            record.precision == request.precision and record.quality_class == request.quality_class,
            f"precision/quality mismatch (has {record.precision}/{record.quality_class})",
        ),
        "context_input_output_feasibility": (
            request.prompt_tokens + request.reserved_output_tokens <= record.max_context_tokens,
            "input plus reserved output exceeds the backend context",
        ),
        "protocol_feature_capability": (
            (not request.stream or record.features.get("stream", True))
            and (not request.cancel or record.features.get("cancel", True))
            and (not request.logprobs or record.features.get("logprobs", False))
            and (not request.prefix_cache or record.features.get("prefix", False)),
            "a requested protocol feature is unsupported",
        ),
        "device_parallel_constraints": (record.parallel_degree >= 1, "invalid parallel degree"),
        "readiness_model_epoch": (
            health.routable and bool(record.model_epoch),
            f"not routable ({health.classification()})",
        ),
        "circuit_health": (
            (record.breaker.allow_request(monotonic_ns=now_ns)["allowed"] if record.breaker else True),
            "circuit/health refuses new work",
        ),
    }
    for name in hard_filter_order:
        if name == "deadline_slo_envelope":
            continue  # evaluated after scoring inputs are known
        ok, message = checks.get(name, (True, ""))
        if not ok:
            stage = name
            reasons.append(f"{name}: {message}")
    if record.telemetry is None and "prefix" in str(request.prefix_cache):
        reasons.append("prefix routing requested but the instance reports no cache telemetry")
    return FilterOutcome(
        candidate_id=record.instance_id,
        feasible=not reasons,
        reasons=tuple(reasons),
        failed_stage=stage,
    )


def hard_filter(
    request: RouteRequest,
    snapshot: RegistrySnapshot,
    *,
    now_ns: int,
    telemetry_ttl_ms: float,
    order: Sequence[str] = HARD_FILTER_ORDER,
) -> Tuple[List[BackendRecord], List[FilterOutcome]]:
    """Filter first, score later; every exclusion keeps its reason."""
    feasible: List[BackendRecord] = []
    rejected: List[FilterOutcome] = []
    for instance_id in snapshot.ids():
        record = snapshot.records[instance_id]
        outcome = _check_record(
            record,
            request,
            now_ns=now_ns,
            telemetry_ttl_ms=telemetry_ttl_ms,
            hard_filter_order=order,
        )
        if outcome.feasible:
            feasible.append(record)
        else:
            rejected.append(outcome)
    return feasible, rejected


# ── scoring ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoreFeature:
    name: str
    unit: str
    normalization: str
    weight: float
    missing_policy: str = "treat_as_unknown_and_apply_conservative_rule"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "normalization": self.normalization,
            "weight": self.weight,
            "missing_policy": self.missing_policy,
        }


@dataclass(frozen=True)
class ScoreSpec:
    """Frozen scoring configuration (weights, units, tie break, TTL)."""

    features: Tuple[ScoreFeature, ...]
    tie_break: str
    telemetry_ttl_ms: float
    max_telemetry_age_ms: float
    missing_telemetry_never_zero: bool
    score_sense: str
    slo_feasibility_terms: Tuple[str, ...]

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "ScoreSpec":
        features = tuple(
            ScoreFeature(
                name=str(item["name"]),
                unit=str(item["unit"]),
                normalization=str(item["normalization"]),
                weight=float(item["weight"]),
                missing_policy=str(item["missing_policy"]),
            )
            for item in payload["score_features"]
        )
        registry = dict(payload["registry"])
        return cls(
            features=features,
            tie_break=str(payload["tie_break"]),
            telemetry_ttl_ms=float(registry["telemetry_ttl_ms"]),
            max_telemetry_age_ms=float(registry["max_telemetry_age_ms"]),
            missing_telemetry_never_zero=bool(payload["missing_telemetry_never_zero"]),
            score_sense=str(payload["score_sense"]),
            slo_feasibility_terms=tuple(
                str(item) for item in dict(payload["slo_feasibility"])["predicted_completion_terms"]
            ),
        )


@dataclass(frozen=True)
class ScoredCandidate:
    instance_id: str
    raw: Mapping[str, Optional[float]]
    normalized: Mapping[str, float]
    missing: Tuple[str, ...]
    components: Mapping[str, float]
    score: float
    telemetry_age_ms: Optional[float]
    stale: bool
    slo_feasible: Optional[bool]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "raw": {name: value for name, value in self.raw.items()},
            "normalized": dict(self.normalized),
            "missing": list(self.missing),
            "components": dict(self.components),
            "score": self.score,
            "telemetry_age_ms": self.telemetry_age_ms,
            "stale": self.stale,
            "slo_feasible": self.slo_feasible,
        }


def _feature_value(record: BackendRecord, name: str) -> Optional[float]:
    telemetry = record.telemetry
    if name == "predicted_queue_ms":
        if telemetry is None:
            return None
        return telemetry.queue_depth * 10.0  # a queue-depth-to-ms proxy, unit declared
    if name == "predicted_ttft_ms":
        if telemetry is None:
            return None
        return float(record.capacity_envelope_ms.get("ttft_ms", 0.0)) + telemetry.inflight_requests * 5.0
    if name == "cache_saved_prefill_ms":
        if telemetry is None:
            return None
        return telemetry.cache_saved_prefill_ms
    if name == "monetary_cost_units":
        return record.cost_units
    if name == "routing_skew_ratio":
        if telemetry is None:
            return None
        return telemetry.inflight_requests
    return None


def score_candidates(
    feasible: Sequence[BackendRecord],
    *,
    spec: ScoreSpec,
    request: RouteRequest,
    now_ns: int,
) -> List[ScoredCandidate]:
    """Min-max normalise over the feasible set; missing telemetry is penalised."""
    values: Dict[str, Dict[str, Optional[float]]] = {}
    for record in feasible:
        values[record.instance_id] = {
            feature.name: _feature_value(record, feature.name) for feature in spec.features
        }
    scored: List[ScoredCandidate] = []
    for record in feasible:
        raw = values[record.instance_id]
        normalized: Dict[str, float] = {}
        missing: List[str] = []
        for feature in spec.features:
            observed = [
                value
                for value in (
                    values[other.instance_id][feature.name] for other in feasible
                )
                if value is not None
            ]
            value = raw[feature.name]
            if value is None:
                missing.append(feature.name)
                # conservative: an unknown feature is treated as the worst observed
                # value plus a penalty, never as zero
                worst = max(observed) if observed else 1.0
                normalized[feature.name] = worst * 1.25
                continue
            if not observed or max(observed) == min(observed):
                normalized[feature.name] = 0.0
                continue
            low, high = min(observed), max(observed)
            normalized[feature.name] = (value - low) / (high - low)
        components = {
            feature.name: normalized[feature.name] * feature.weight for feature in spec.features
        }
        telemetry_age = (
            record.telemetry.age_ms(now_ns) if record.telemetry is not None else None
        )
        stale = telemetry_age is None or telemetry_age > spec.telemetry_ttl_ms
        slo_feasible: Optional[bool] = None
        if request.deadline_ns is not None:
            # the per-Backend capacity envelope comes from E08-02; a missing term
            # makes the prediction *unknown* rather than zero
            terms = [record.capacity_envelope_ms.get(name) for name in spec.slo_feasibility_terms]
            if all(term is not None for term in terms):
                predicted_ms = sum(float(term) for term in terms if term is not None)
                remaining_ms = (request.deadline_ns - now_ns) / 1e6
                slo_feasible = predicted_ms <= remaining_ms
            else:
                missing.append("slo_feasibility_envelope")
        scored.append(
            ScoredCandidate(
                instance_id=record.instance_id,
                raw=raw,
                normalized=normalized,
                missing=tuple(missing),
                components=components,
                score=sum(components.values()),
                telemetry_age_ms=telemetry_age,
                stale=stale,
                slo_feasible=slo_feasible,
            )
        )
    scored.sort(key=lambda item: (item.score, item.instance_id))
    return scored


# ── decision record ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteDecision:
    request_id: str
    policy_version: str
    candidate_ids: Tuple[str, ...]
    feasible_ids: Tuple[str, ...]
    rejected: Tuple[Mapping[str, Any], ...]
    scored: Tuple[Mapping[str, Any], ...]
    tie_break: str
    selected_instance_id: str
    selected_model_epoch: str
    selected_route_epoch: str
    reason: str
    fallback_level: str = "same_instance_retry"
    no_feasible: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "policy_version": self.policy_version,
            "candidate_ids": list(self.candidate_ids),
            "feasible_ids": list(self.feasible_ids),
            "rejected": [dict(item) for item in self.rejected],
            "scored": [dict(item) for item in self.scored],
            "tie_break": self.tie_break,
            "selected_instance_id": self.selected_instance_id,
            "selected_model_epoch": self.selected_model_epoch,
            "selected_route_epoch": self.selected_route_epoch,
            "reason": self.reason,
            "fallback_level": self.fallback_level,
            "no_feasible": self.no_feasible,
        }


POLICY_VERSION = "s08.routing.v1"


def route(
    request: RouteRequest,
    registry: BackendRegistry,
    *,
    spec: ScoreSpec,
    now_ns: int,
    policy_version: str = POLICY_VERSION,
) -> RouteDecision:
    """The full routing path: filter → score → select (or reject)."""
    snapshot = registry.snapshot()
    feasible, rejected = hard_filter(
        request,
        snapshot,
        now_ns=now_ns,
        telemetry_ttl_ms=spec.telemetry_ttl_ms,
    )
    if not feasible:
        return RouteDecision(
            request_id=request.request_id,
            policy_version=policy_version,
            candidate_ids=snapshot.ids(),
            feasible_ids=(),
            rejected=tuple(outcome.as_dict() for outcome in rejected),
            scored=(),
            tie_break=spec.tie_break,
            selected_instance_id="",
            selected_model_epoch="",
            selected_route_epoch="",
            reason="no candidate passed the hard capability/identity/health filters",
            fallback_level="fail_closed_reject",
            no_feasible=True,
        )
    scored = score_candidates(feasible, spec=spec, request=request, now_ns=now_ns)
    eligible = [item for item in scored if item.slo_feasible is not False]
    if not eligible:
        return RouteDecision(
            request_id=request.request_id,
            policy_version=policy_version,
            candidate_ids=snapshot.ids(),
            feasible_ids=tuple(record.instance_id for record in feasible),
            rejected=tuple(outcome.as_dict() for outcome in rejected),
            scored=tuple(item.as_dict() for item in scored),
            tie_break=spec.tie_break,
            selected_instance_id="",
            selected_model_epoch="",
            selected_route_epoch="",
            reason="every feasible candidate is predicted infeasible for the deadline",
            fallback_level="fail_closed_reject",
            no_feasible=True,
        )
    chosen = eligible[0]
    record = snapshot.get(chosen.instance_id)
    return RouteDecision(
        request_id=request.request_id,
        policy_version=policy_version,
        candidate_ids=snapshot.ids(),
        feasible_ids=tuple(item.instance_id for item in feasible),
        rejected=tuple(outcome.as_dict() for outcome in rejected),
        scored=tuple(item.as_dict() for item in eligible),
        tie_break=spec.tie_break,
        selected_instance_id=chosen.instance_id,
        selected_model_epoch=record.model_epoch,
        selected_route_epoch=record.route_epoch,
        reason=f"lowest score {chosen.score:.4f} among {len(eligible)} feasible candidates",
    )


def route_vs_actual(
    decisions: Sequence[RouteDecision],
    actuals: Mapping[str, Tuple[str, str]],
) -> Dict[str, Any]:
    """``request_id → (actual instance, actual model epoch)`` must match."""
    problems: List[str] = []
    checked = 0
    for decision in decisions:
        if decision.no_feasible:
            continue
        actual = actuals.get(decision.request_id)
        if actual is None:
            problems.append(f"{decision.request_id}: no actual execution recorded")
            continue
        checked += 1
        if actual[0] != decision.selected_instance_id:
            problems.append(
                f"{decision.request_id}: selected {decision.selected_instance_id} but "
                f"executed on {actual[0]}"
            )
        if actual[1] != decision.selected_model_epoch:
            problems.append(
                f"{decision.request_id}: selected epoch {decision.selected_model_epoch} but "
                f"executed epoch {actual[1]}"
            )
    return {
        "ok": not problems,
        "problems": problems,
        "checked": checked,
        "note": "a route that did not happen is an evidence failure, not a rounding issue",
    }


def fallback_plan(
    *,
    request: RouteRequest,
    level: str,
    reason: str,
) -> Dict[str, Any]:
    """A fallback must be explicitly allowed before it may degrade anything."""
    if level not in FALLBACK_LADDER:
        raise ConfigError(f"unknown fallback level {level!r}; expected one of {list(FALLBACK_LADDER)}")
    permitted = True
    problem = ""
    if level == "approved_alternate_precision" and not request.allow_alternate_precision:
        permitted = False
        problem = "the request/SLO does not allow an alternate precision"
    if level == "alternate_model_degraded_quality" and not request.allow_alternate_model:
        permitted = False
        problem = "the request/SLO does not allow an alternate model"
    return {
        "level": level,
        "permitted": permitted,
        "problem": problem,
        "reason": reason,
        "must_revalidate_identity": True,
        "note": (
            "a busy Backend is not a reason to silently switch to a smaller model "
            "(E08-06 §10)"
        ),
    }


def telemetry_freshness_report(
    snapshot: RegistrySnapshot, *, now_ns: int, spec: ScoreSpec
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    stale = 0
    for instance_id in snapshot.ids():
        record = snapshot.records[instance_id]
        age = record.telemetry.age_ms(now_ns) if record.telemetry else None
        is_stale = age is None or age > spec.telemetry_ttl_ms
        if is_stale:
            stale += 1
        rows.append(
            {
                "instance_id": instance_id,
                "age_ms": age,
                "ttl_ms": spec.telemetry_ttl_ms,
                "stale": is_stale,
                "policy": "conservative" if is_stale else "score",
            }
        )
    return {
        "rows": rows,
        "stale_instances": stale,
        "note": (
            "stale telemetry triggers the conservative rule; it is never read as a zero "
            "load (E08-06 §6)"
        ),
    }


__all__ = [
    "BackendRecord",
    "BackendRegistry",
    "FALLBACK_LADDER",
    "FilterOutcome",
    "HARD_FILTER_ORDER",
    "PERMISSION_REQUIRED_LEVELS",
    "POLICY_VERSION",
    "RegistrySnapshot",
    "RouteDecision",
    "RouteRequest",
    "ScoreFeature",
    "ScoreSpec",
    "ScoredCandidate",
    "Telemetry",
    "fallback_plan",
    "hard_filter",
    "route",
    "route_vs_actual",
    "score_candidates",
    "telemetry_freshness_report",
]
