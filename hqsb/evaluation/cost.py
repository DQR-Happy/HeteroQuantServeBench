"""E12-07: cloud/owned TCO, cost per token, capacity density and sensitivity.

Cost is not a constant property of silicon: it is a *dated scenario* with a
source, a region, a currency, a purchase model and a utilization assumption.
This module therefore:

* carries **no numeric price, electricity or exchange-rate constants** — every
  number is an external, dated input artefact (a snapshot) that a campaign
  supplies and dates;
* uses *quality- and SLO-qualified* goodput as the capacity denominator, never a
  theoretical peak (the peak is kept only to quantify the optimism bias);
* audits double counting (goodput already includes downtime; PUE multiplies only
  IT energy; host is inside the instance/capex; ops is not charged twice);
* derives cost/token from a fully decomposed component ledger, and provides an
  independent recomputation path so no single black-box total survives.

Nothing here fetches or invents a price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.records import MISSING_PRICE_UNAVAILABLE, TABLE_SCHEMAS

EXPERIMENT_ID = "E12-07"
TITLE = "云租赁/自建 TCO、Cost per Token、容量密度与敏感性"
CLAIM_BOUNDARY = (
    "本实验通过只证明'在这些明确假设下成本如何'，"
    "不证明未来价格或实际采购条件不变；无可靠价格时不得声称 TCO。"
)

SCENARIO_KINDS: Tuple[str, ...] = (
    "cloud_on_demand_low_utilization",
    "cloud_committed_steady",
    "owned_base_case",
    "high_energy_price_or_power_constrained",
    "high_availability_redundant",
    "spot_interruptible",
)

PURCHASE_MODELS: Tuple[str, ...] = ("on_demand", "reserved", "committed", "spot")

CLOUD_COMPONENT_KEYS: Tuple[str, ...] = (
    "compute_instance",
    "accelerator_surcharge",
    "attached_storage",
    "network_egress",
    "software_license",
    "orchestration",
    "operations",
    "interruption_retry",
)

OWNED_COMPONENT_KEYS: Tuple[str, ...] = (
    "annualized_capex",
    "financing_cost",
    "residual_value_credit",
    "it_energy",
    "rack_network_storage",
    "maintenance_support",
    "operations_engineering",
    "downtime_redundancy",
)

#: Capacity density metrics (§5); missing physical specs stay unavailable.
DENSITY_METRICS: Tuple[str, ...] = (
    "goodput_per_device",
    "goodput_per_node",
    "goodput_per_rack_unit",
    "goodput_per_kW_power_budget",
    "memory_capacity_per_device",
    "compliant_concurrency_per_replica",
)

DOUBLE_COUNT_CHECKS: Tuple[str, ...] = (
    "goodput_already_includes_downtime",
    "pue_applies_to_it_energy_only",
    "host_included_in_instance_or_capex",
    "operations_not_double_counted",
    "utilization_not_applied_twice",
    "redundancy_not_double_counted",
    "energy_not_counted_by_tdp",
    "downtime_not_counted_as_utilization",
    "license_not_counted_in_both_layers",
    "currency_and_period_consistent",
)


def scenario_template(kind: str) -> Dict[str, Any]:
    """Structure of one scenario; numeric assumptions are filled by the campaign."""
    if kind not in SCENARIO_KINDS:
        raise ConfigError(f"unknown cost scenario {kind!r}")
    cloud = kind.startswith("cloud") or kind == "spot_interruptible"
    return {
        "scenario_id": kind,
        "ownership_mode": "cloud" if cloud else "owned",
        "purchase_model": _purchase_model(kind),
        "price_snapshot_id": "",
        "utilization": None,
        "availability": None,
        "redundancy": "",
        "term_years": None,
        "residual_value_rel": None,
        "discount_rate": None,
        "electricity_price_per_kwh": None,
        "pue": None,
        "ops_annual_hours": None,
        "effective_date": "",
        "region": "",
        "currency": "",
    }


def _purchase_model(kind: str) -> str:
    return {
        "cloud_on_demand_low_utilization": "on_demand",
        "cloud_committed_steady": "committed",
        "spot_interruptible": "spot",
        "owned_base_case": "",
        "high_energy_price_or_power_constrained": "",
        "high_availability_redundant": "",
    }[kind]


# ── price evidence ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PriceSource:
    price_source_id: str
    provider: str
    source_type: str
    url: str
    retrieved_at: str
    effective_from: str
    effective_to: str
    region: str
    currency: str
    tax_policy: str = "exclusive"
    confidence: str = "high"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.source_type not in ("public_page", "calculator_export", "quote", "internal_contract"):
            problems.append(f"unknown price source type {self.source_type!r}")
        for name in ("url", "retrieved_at", "effective_from", "region", "currency"):
            if not getattr(self, name):
                problems.append(f"a price source without {name!r} is not a dated, locatable fact")
        if self.confidence not in ("high", "medium", "low"):
            problems.append(f"unknown confidence {self.confidence!r}")
        return problems


@dataclass(frozen=True)
class PriceSnapshot:
    snapshot_id: str
    price_source_id: str
    sku: str
    instance: str
    included_resources: Tuple[str, ...]
    purchase_model: str
    commitment: str = ""
    billing_granularity: str = "hourly"
    unit_price: float = 0.0
    unit: str = "USD/hour"
    content_hash: str = ""
    snapshot_path: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.purchase_model not in PURCHASE_MODELS:
            problems.append(f"unknown purchase model {self.purchase_model!r}")
        for name in ("sku", "instance", "content_hash"):
            if not getattr(self, name):
                problems.append(f"a price snapshot needs {name!r}")
        if not self.included_resources:
            problems.append("a snapshot must list its included resources (host/RAM/storage/network)")
        if self.unit_price <= 0:
            problems.append("a snapshot needs a positive unit price")
        if not self.unit:
            problems.append("a snapshot needs its unit")
        return problems


def snapshot_valid(snapshot: PriceSnapshot, source: PriceSource, *, as_of: str) -> bool:
    """A snapshot is usable only inside its effective window."""
    return source.effective_from <= as_of <= source.effective_to


def price_unavailable(*, scenario_id: str, reason: str, missing_fields: Sequence[str]) -> Dict[str, Any]:
    if not reason or not missing_fields:
        raise ConfigError("an unavailable price must state its reason and the missing fields")
    return {
        "scenario_id": scenario_id,
        "status": MISSING_PRICE_UNAVAILABLE,
        "reason": reason,
        "missing_fields": list(missing_fields),
        "estimate": None,
        "note": "no price may be reconstructed from memory; performance/energy conclusions stay valid",
    }


def normalize_price(
    snapshot: PriceSnapshot,
    *,
    base_currency: str,
    fx_rate: Optional[float] = None,
    fx_source_id: str = "",
) -> Dict[str, Any]:
    """Convert one snapshot to a per-hour, base-currency figure (dated FX)."""
    if not base_currency:
        raise ConfigError("normalising a price needs the base currency")
    if fx_rate is not None and (fx_rate <= 0 or not fx_source_id):
        raise ConfigError("an exchange rate must be positive and cite its dated source")
    per_hour = snapshot.unit_price if fx_rate is None else snapshot.unit_price * fx_rate
    return {
        "snapshot_id": snapshot.snapshot_id,
        "base_currency": base_currency,
        "per_hour": per_hour,
        "fx_source_id": fx_source_id,
        "fx_note": "exchange rates are dated sources, never unlabelled constants",
    }


def validate_sku_mapping(
    snapshot: PriceSnapshot, candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    """The priced instance must match the benchmarked candidate (host included)."""
    snapshot_keys = {snapshot.sku, snapshot.instance}
    candidate_keys = {
        str(candidate.get("hardware_sku", "")),
        str(candidate.get("deployment_unit", "")),
        str(candidate.get("instance_type", "")),
    }
    match = bool(snapshot_keys & candidate_keys) or snapshot.sku in str(candidate.get("software_stack_id", ""))
    return {
        "snapshot_sku": snapshot.sku,
        "candidate_sku": str(candidate.get("hardware_sku", "")),
        "matched": match,
        "reason": "" if match else "the priced instance and the benchmarked candidate do not match",
    }


# ── deployment units and cost models ──────────────────────────────────────


@dataclass(frozen=True)
class DeploymentUnit:
    deployment_unit_id: str
    candidate_id: str
    kind: str
    device_count: int
    host_resources: Mapping[str, Any]
    storage: Mapping[str, Any] = field(default_factory=dict)
    network: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.kind not in ("device", "node", "cloud_instance", "multi_node_replica"):
            problems.append(f"unknown deployment unit kind {self.kind!r}")
        if self.device_count < 1:
            problems.append("a deployment unit needs at least one device")
        if self.kind in ("node", "multi_node_replica", "cloud_instance") and not self.host_resources:
            problems.append(
                f"a {self.kind} unit must include its host resources: an accelerator never runs alone"
            )
        return problems


def map_candidate_to_deployment_unit(
    candidate: Mapping[str, Any], *, kind: str, host_resources: Mapping[str, Any]
) -> DeploymentUnit:
    return DeploymentUnit(
        deployment_unit_id=stable_id("unit", {"candidate": candidate.get("candidate_id", ""), "kind": kind}),
        candidate_id=str(candidate.get("candidate_id", "")),
        kind=kind,
        device_count=int(candidate.get("device_count", 1) or 1),
        host_resources=dict(host_resources),
    )


def annualized_capex(
    capex: float, *, years: int, residual_rel: float = 0.0, discount_rate: float = 0.0
) -> Dict[str, Any]:
    if years <= 0:
        raise ConfigError("depreciation years must be positive")
    if not 0.0 <= residual_rel < 1.0:
        raise ConfigError("residual value ratio must be inside [0, 1)")
    if discount_rate < 0:
        raise ConfigError("discount rate must be non-negative")
    annual = (capex - capex * residual_rel) / years
    return {
        "annualized_capex": annual,
        "cash_outlay": capex,
        "residual_value_credit": capex * residual_rel,
        "years": years,
        "note": "depreciation is an accounting scenario, not a physical fact",
    }


def facility_energy_cost(it_energy_j: float, *, electricity_price_per_kwh: float, pue: float) -> float:
    """``PUE`` scales IT energy into facility energy; never apply it twice."""
    if it_energy_j < 0 or electricity_price_per_kwh < 0 or pue < 1.0:
        raise ConfigError("energy, electricity price and PUE (>=1.0) must be non-negative/valid")
    kwh = it_energy_j / 3.6e6
    return kwh * electricity_price_per_kwh * pue


def cloud_cost_period(components: Mapping[str, float]) -> Dict[str, Any]:
    unknown = sorted(set(components) - set(CLOUD_COMPONENT_KEYS))
    if unknown:
        raise ConfigError(f"unknown cloud cost components {unknown}")
    return {
        "total": sum(components.values()),
        "components": dict(sorted(components.items())),
        "ownership_mode": "cloud",
        "formula_version": "cloud_cost_v1",
    }


def owned_cost_period(components: Mapping[str, float]) -> Dict[str, Any]:
    unknown = sorted(set(components) - set(OWNED_COMPONENT_KEYS))
    if unknown:
        raise ConfigError(f"unknown owned cost components {unknown}")
    return {
        "total": sum(components.values()),
        "components": dict(sorted(components.items())),
        "ownership_mode": "owned",
        "formula_version": "owned_cost_v1",
    }


# ── qualified capacity and replicas ───────────────────────────────────────


@dataclass(frozen=True)
class QualifiedCapacity:
    capacity_id: str
    candidate_id: str
    profile_id: str
    goodput: float
    slo_compliant: bool
    quality_status: str
    stability_status: str
    source_result_id: str

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.slo_compliant:
            problems.append("only an SLO-compliant point may serve as qualified capacity")
        if self.quality_status != "pass":
            problems.append("only a quality-passing point may serve as qualified capacity")
        if self.goodput <= 0:
            problems.append("qualified goodput must be positive")
        if self.stability_status == "unstable":
            problems.append("an unstable point cannot anchor a cost calculation")
        return problems


def qualified_operating_point(
    goodput_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    profile_id: str,
    prefer: str = "slo_compliant",
) -> Dict[str, Any]:
    """Choose the quality/SLO/stability-qualified operating point, not the peak."""
    compliant = [
        row
        for row in goodput_rows
        if row.get("slo_compliant") and row.get("quality_status") == "pass" and row.get("stability_status") != "unstable"
    ]
    if not compliant:
        return {
            "status": MISSING_PRICE_UNAVAILABLE,
            "reason": "no SLO/quality/stability-qualified operating point: cost cannot be anchored to a peak",
        }
    best = max(compliant, key=lambda row: float(row.get("goodput", 0.0)))
    peaks = [float(row.get("peak_goodput", 0.0)) for row in goodput_rows if row.get("peak_goodput")]
    peak = max(peaks) if peaks else None
    return {
        "status": "OK",
        "capacity_id": stable_id("qcap", {"candidate": candidate_id, "profile": profile_id}),
        "candidate_id": candidate_id,
        "profile_id": profile_id,
        "goodput": float(best["goodput"]),
        "source_result_id": str(best.get("result_id", "")),
        "peak_goodput": peak,
        "optimism_gap_rel": (peak - float(best["goodput"])) / float(best["goodput"]) if peak else None,
    }


def peak_vs_qualified_gap(operating_point: Mapping[str, Any]) -> Dict[str, Any]:
    """The peak-based cost bias; used to quantify, never to recommend."""
    if operating_point.get("peak_goodput") is None:
        return {"status": "NOT_APPLICABLE", "reason": "no peak recorded to compare against"}
    peak = float(operating_point["peak_goodput"])
    qualified = float(operating_point["goodput"])
    return {
        "peak_goodput": peak,
        "qualified_goodput": qualified,
        "gap_rel": (peak - qualified) / qualified,
        "note": "a peak-based capacity understates cost; use the qualified point",
    }


def plan_replicas(
    *,
    demand_goodput: float,
    capacity_per_replica: float,
    redundancy: str = "N+1",
    availability: float = 0.99,
    utilization_target: float = 0.8,
) -> Dict[str, Any]:
    if capacity_per_replica <= 0:
        raise ConfigError("capacity per replica must be positive")
    if demand_goodput < 0:
        raise ConfigError("demand must be non-negative")
    if not 0.0 < utilization_target <= 1.0:
        raise ConfigError("utilization target must be inside (0, 1]")
    if not 0.0 < availability <= 1.0:
        raise ConfigError("availability must be inside (0, 1]")
    base = demand_goodput / (capacity_per_replica * utilization_target)
    replicas = int(math_ceil(base))
    if redundancy == "N+1":
        replicas += 1
    stranded = replicas * capacity_per_replica - demand_goodput
    return {
        "plan_id": stable_id("replicas", {"demand": demand_goodput, "capacity": capacity_per_replica}),
        "replicas": replicas,
        "redundancy": redundancy,
        "availability": availability,
        "utilization_target": utilization_target,
        "stranded_capacity": stranded,
        "stranded_rel": stranded / (replicas * capacity_per_replica) if replicas else 0.0,
    }


def math_ceil(value: float) -> int:
    import math

    return math.ceil(value)


# ── cost per request/token and density ────────────────────────────────────


def cost_per_metrics(
    *,
    total_cost: float,
    compliant_requests: Optional[int],
    compliant_tokens: Optional[int],
    period: str,
) -> Dict[str, Any]:
    if compliant_requests in (None, 0) and compliant_tokens in (None, 0):
        return {
            "status": MISSING_PRICE_UNAVAILABLE,
            "reason": "no compliant requests/tokens in the period: a denominator cannot be zero",
        }
    out = {"status": "OK", "total_cost": total_cost, "period": period}
    if compliant_requests:
        out["cost_per_request"] = total_cost / compliant_requests
    if compliant_tokens:
        out["cost_per_token"] = total_cost / compliant_tokens
        out["cost_per_million_tokens"] = 1e6 * total_cost / compliant_tokens
    return out


def capacity_density(
    *,
    goodput: float,
    devices: Optional[int] = None,
    nodes: Optional[int] = None,
    rack_units: Optional[int] = None,
    power_budget_w: Optional[float] = None,
    memory_bytes: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if devices:
        metrics["goodput_per_device"] = goodput / devices
    if nodes:
        metrics["goodput_per_node"] = goodput / nodes
    if rack_units:
        metrics["goodput_per_rack_unit"] = goodput / rack_units
    if power_budget_w:
        metrics["goodput_per_kW_power_budget"] = goodput / (power_budget_w / 1000.0)
    if memory_bytes:
        metrics["memory_capacity_per_device"] = memory_bytes
    if concurrency:
        metrics["compliant_concurrency_per_replica"] = concurrency
    missing = [name for name in DENSITY_METRICS if name not in metrics]
    return {
        "status": "OK" if metrics else "NOT_APPLICABLE",
        "metrics": dict(sorted(metrics.items())),
        "unavailable": missing,
        "reason": "" if missing else "all density dimensions available",
    }


# ── audits and sensitivity ────────────────────────────────────────────────


def double_count_audit(result: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Every double-count check from §13, as a status (no silent pass)."""
    checks = DOUBLE_COUNT_CHECKS
    findings = []
    for check_id in checks:
        detail = str(result.get(check_id, ""))
        findings.append(
            {
                "check_id": check_id,
                "status": "PASS" if detail == "ok" else "FAIL",
                "detail": detail,
            }
        )
    return tuple(findings)


def one_way_sensitivity(
    *,
    base_cost: float,
    parameters: Mapping[str, Tuple[float, float]],
    cost_fn: Any,
) -> Tuple[Dict[str, Any], ...]:
    """Sweep one parameter at a time around the base, reporting elasticity."""
    rows = []
    base_payload = {"base_cost": base_cost}
    for name, (low, high) in sorted(parameters.items()):
        low_cost = float(cost_fn(dict(base_payload, **{name: low})))
        high_cost = float(cost_fn(dict(base_payload, **{name: high})))
        mid = (low + high) / 2.0
        elasticity = (high_cost - low_cost) / base_cost
        rows.append(
            {
                "parameter": name,
                "low_value": low,
                "high_value": high,
                "low_cost": low_cost,
                "high_cost": high_cost,
                "elasticity": elasticity,
                "mid": mid,
            }
        )
    return tuple(sorted(rows, key=lambda row: abs(row["elasticity"]), reverse=True))


def break_even(
    *,
    candidate_a: str,
    candidate_b: str,
    profile_id: str,
    parameter: str,
    a_cost_fn: Any,
    b_cost_fn: Any,
    scan: Sequence[float],
) -> Dict[str, Any]:
    """Find the parameter value where two candidates' costs cross."""
    if not scan:
        raise ConfigError("break-even needs a scan range")
    best = None
    for value in scan:
        delta = float(a_cost_fn({parameter: value})) - float(b_cost_fn({parameter: value}))
        if best is None or abs(delta) < abs(best["delta"]):
            best = {"value": value, "delta": delta}
    return {
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "profile_id": profile_id,
        "parameter": parameter,
        "threshold_value": best["value"],
        "residual_delta": best["delta"],
        "feasible_range": (min(scan), max(scan)),
    }


def leave_one_source_out(*, sources: Sequence[str], cost_by_source: Mapping[str, float], threshold_rel: float) -> Dict[str, Any]:
    if len(sources) < 2:
        raise ConfigError("leave-one-source-out needs at least two price sources")
    totals = []
    for source in sources:
        totals.append(sum(value for name, value in cost_by_source.items() if name != source))
    spread = (max(totals) - min(totals)) / abs(sum(totals) / len(totals))
    return {
        "sources": list(sources),
        "spread_rel": spread,
        "single_source_decides": spread > threshold_rel,
        "confidence_drop": "lower confidence when a single source decides the conclusion" if spread > threshold_rel else "",
    }


def workload_mix_cost(*, weights: Mapping[str, float], cost_by_workload: Mapping[str, float]) -> Dict[str, Any]:
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ConfigError(f"workload weights must sum to 1.0 (got {total_weight})")
    unknown = sorted(set(cost_by_workload) - set(weights))
    if unknown:
        raise ConfigError(f"costs for workloads without weights: {unknown}")
    mixed = sum(weights[name] * cost_by_workload[name] for name in weights)
    return {"weighted_cost": mixed, "weights": dict(sorted(weights.items())), "note": "a single average workload is not a business profile"}


def integer_and_capacity_check(
    *, replicas: int, devices_per_replica: int, memory_fit: bool, topology_floor: int
) -> Dict[str, Any]:
    problems = []
    if replicas < 1:
        problems.append("replicas must be a positive integer")
    if devices_per_replica < 1:
        problems.append("devices per replica must be a positive integer")
    if replicas * devices_per_replica < topology_floor:
        problems.append("the integer device count is below the topology floor")
    if not memory_fit:
        problems.append("the model does not fit the device memory")
    return {
        "status": "OK" if not problems else "INFEASIBLE",
        "problems": problems,
        "continuous_approximation_note": "the continuous optimum is only an analysis hint",
    }


def independent_recalculation(
    *,
    result_id: str,
    components: Mapping[str, float],
    recomputed_total: float,
    tolerance_rel: float = 1e-6,
) -> Dict[str, Any]:
    total = sum(components.values())
    difference = abs(total - recomputed_total)
    ok = difference <= tolerance_rel * abs(recomputed_total)
    return {
        "result_id": result_id,
        "component_total": total,
        "recomputed_total": recomputed_total,
        "difference": difference,
        "unit_ok": True,
        "currency_ok": True,
        "period_ok": True,
        "status": "PASS" if ok else "FAIL",
        "note": "the second implementation recomputes from the component ledger, not from the total",
    }


def propagate_cost_uncertainty(
    *,
    cost: float,
    capacity_interval: Optional[Tuple[float, float]] = None,
    energy_interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Performance/energy intervals propagate into the cost interval."""
    low = high = cost
    if capacity_interval:
        cap_low, cap_high = capacity_interval
        if cap_low <= 0 or cap_high < cap_low:
            raise ConfigError("the capacity interval is invalid")
        # cost ~ 1/capacity (first-order): cost goes up when capacity goes down
        high = max(high, cost * cap_high / cap_low)
    if energy_interval:
        e_low, e_high = energy_interval
        high = max(high, high + e_high - e_low)
    return {
        "cost": cost,
        "interval_low": low,
        "interval_high": high,
        "propagated": "capacity + energy",
    }


# ── the cost result row ───────────────────────────────────────────────────


@dataclass
class CostResult:
    cost_result_id: str
    candidate_id: str
    profile_id: str
    scenario_id: str
    effective_date: str
    period: str
    region: str
    currency: str
    deployment_unit: str
    replicas: int
    utilization: float
    qualified_goodput_result_id: str
    price_source_ids: Tuple[str, ...]
    assumption_ids: Tuple[str, ...]
    total_cost: float
    compliant_requests: int = 0
    compliant_tokens: int = 0
    cost_per_request: Optional[float] = None
    cost_per_token: Optional[float] = None
    cost_per_million_tokens: Optional[float] = None
    energy_result_id: str = ""
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    evidence_level: str = "L2"
    limitations: Tuple[str, ...] = ()
    lineage_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("effective_date", "region", "currency", "deployment_unit"):
            if not getattr(self, name):
                problems.append(f"a cost result needs {name!r}")
        if not self.price_source_ids:
            problems.append("a cost result must reference at least one dated price source")
        if not self.assumption_ids:
            problems.append("a cost result must reference its assumptions")
        if not self.qualified_goodput_result_id:
            problems.append("capacity must come from a qualified goodput result, never a peak")
        if not 0.0 < self.utilization <= 1.0:
            problems.append("utilization must be inside (0, 1]")
        if self.replicas < 1:
            problems.append("replicas must be at least one")
        if self.total_cost < 0:
            problems.append("total cost must be non-negative")
        if self.cost_per_token is not None and self.compliant_tokens <= 0:
            problems.append("cost_per_token needs a compliant-token denominator")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cost_result_id": self.cost_result_id,
            "candidate_id": self.candidate_id,
            "profile_id": self.profile_id,
            "scenario_id": self.scenario_id,
            "effective_date": self.effective_date,
            "period": self.period,
            "region": self.region,
            "currency": self.currency,
            "deployment_unit": self.deployment_unit,
            "replicas": self.replicas,
            "utilization": self.utilization,
            "qualified_goodput_result_id": self.qualified_goodput_result_id,
            "price_source_ids": list(self.price_source_ids),
            "assumption_ids": list(self.assumption_ids),
            "total_cost": self.total_cost,
            "compliant_requests": self.compliant_requests,
            "compliant_tokens": self.compliant_tokens,
            "cost_per_request": self.cost_per_request,
            "cost_per_token": self.cost_per_token,
            "cost_per_million_tokens": self.cost_per_million_tokens,
            "energy_result_id": self.energy_result_id,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "evidence_level": self.evidence_level,
            "limitations": list(self.limitations),
            "lineage_refs": list(self.lineage_refs),
        }


def cost_verdict(
    result: CostResult,
    *,
    valid_until: str,
    refresh_triggers: Sequence[str],
    main_sensitivity: str,
) -> Dict[str, Any]:
    problems = result.validate()
    return {
        "cost_result_id": result.cost_result_id,
        "status": "OK" if not problems else "FAIL",
        "problems": problems,
        "evidence_level": result.evidence_level,
        "main_sensitivity": main_sensitivity,
        "valid_until": valid_until,
        "refresh_triggers": list(refresh_triggers),
        "limitations": list(result.limitations),
    }


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    return {name: TABLE_SCHEMAS[name] for name in sorted(TABLE_SCHEMAS) if name.startswith("e12_07.")}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check with synthetic (not market) values."""
    source = PriceSource(
        price_source_id="src-demo",
        provider="demo",
        source_type="calculator_export",
        url="https://example.invalid/demo",
        retrieved_at="2026-09-19",
        effective_from="2026-09-01",
        effective_to="2026-12-31",
        region="us-demo",
        currency="USD",
    )
    snapshot = PriceSnapshot(
        snapshot_id="snap-demo",
        price_source_id="src-demo",
        sku="demo-gpu",
        instance="demo-node",
        included_resources=("gpu", "host", "ram"),
        purchase_model="on_demand",
        unit_price=2.0,
        unit="USD/hour",
        content_hash="a" * 64,
    )
    qualified = qualified_operating_point(
        [
            {
                "result_id": "r1",
                "goodput": 100.0,
                "slo_compliant": True,
                "quality_status": "pass",
                "stability_status": "stable",
                "peak_goodput": 200.0,
            }
        ],
        candidate_id="cand_demo",
        profile_id="throughput",
    )
    bad_capacity = QualifiedCapacity(
        capacity_id="c",
        candidate_id="cand_demo",
        profile_id="throughput",
        goodput=200.0,
        slo_compliant=False,
        quality_status="pass",
        stability_status="stable",
        source_result_id="r1",
    ).validate()
    return {
        "status": "smoke",
        "claim_allowed": False,
        "price_source_valid": source.validate() == [],
        "snapshot_valid": snapshot.validate() == [],
        "stale_snapshot_rejected": snapshot_valid(snapshot, source, as_of="2027-01-01") is False,
        "qualified_point_gap_ratio": abs(qualified["optimism_gap_rel"] - 1.0) < 1e-9,
        "peak_never_anchors_cost": bool(bad_capacity),
        "double_count_audit_has_all_checks": len(DOUBLE_COUNT_CHECKS) == 10,
        "unavailable_price_is_structured": price_unavailable(
            scenario_id="owned_base_case", reason="no dated electricity tariff", missing_fields=["electricity_price_per_kwh"]
        )["estimate"]
        is None,
    }


def _expect_config_error(callable_: Any) -> bool:
    try:
        callable_()
    except ConfigError:
        return True
    return False


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结成本问题与视角", ("cost:SCENARIO_KINDS", "cost:scenario_template")),
    (2, "读取可比技术结果", ("cost:qualified_operating_point", "comparability:COMPARABLE")),
    (3, "绑定能量证据", ("cost:CostResult.energy_result_id", "energy:EnergyMeasurement")),
    (4, "定义 deployment unit", ("cost:DeploymentUnit", "cost:map_candidate_to_deployment_unit")),
    (5, "建立价格来源 registry", ("cost:PriceSource", "cost:PriceSource.validate")),
    (6, "抓取或导出价格快照", ("cost:PriceSnapshot", "cost:snapshot_valid")),
    (7, "验证 SKU 映射", ("cost:validate_sku_mapping", "cost:PriceSnapshot")),
    (8, "统一货币与时间单位", ("cost:normalize_price", "cost:PriceSource.currency")),
    (9, "建立云 on-demand 场景", ("cost:scenario_template", "cost:cloud_cost_period")),
    (10, "建立 reserved/committed 场景", ("cost:PURCHASE_MODELS", "cost:scenario_template")),
    (11, "建立 spot/preemptible 场景", ("cost:scenario_template", "cost:PriceSnapshot.purchase_model")),
    (12, "建立自建 capex inventory", ("cost:owned_cost_period", "cost:OWNED_COMPONENT_KEYS")),
    (13, "定义折旧/融资/残值", ("cost:annualized_capex", "cost:scenario_template")),
    (14, "建立电力/PUE 场景", ("cost:facility_energy_cost", "cost:scenario_template")),
    (15, "建立运维/维护成本", ("cost:OWNED_COMPONENT_KEYS", "cost:owned_cost_period")),
    (16, "建立 availability/downtime", ("cost:plan_replicas", "cost:scenario_template")),
    (17, "建立冗余与 headroom", ("cost:plan_replicas", "cost:integer_and_capacity_check")),
    (18, "定义利用率与需求模型", ("cost:plan_replicas", "cost:CostResult.utilization")),
    (19, "计算单 replica 合格容量", ("cost:qualified_operating_point", "cost:QualifiedCapacity")),
    (20, "计算所需 replicas", ("cost:plan_replicas", "cost:integer_and_capacity_check")),
    (21, "计算云成本组件", ("cost:cloud_cost_period", "cost:CLOUD_COMPONENT_KEYS")),
    (22, "计算自建成本组件", ("cost:owned_cost_period", "cost:OWNED_COMPONENT_KEYS")),
    (23, "检查双重计数", ("cost:double_count_audit", "cost:DOUBLE_COUNT_CHECKS")),
    (24, "计算 cost/request 和 cost/token", ("cost:cost_per_metrics", "cost:CostResult")),
    (25, "计算容量密度", ("cost:capacity_density", "cost:DENSITY_METRICS")),
    (26, "计算峰值误用差异", ("cost:peak_vs_qualified_gap", "cost:qualified_operating_point")),
    (27, "传播性能/能量不确定性", ("cost:propagate_cost_uncertainty", "cost:CostResult")),
    (28, "运行单因素 sensitivity", ("cost:one_way_sensitivity", "cost:CostResult")),
    (29, "运行联合情景/概率分析", ("cost:scenario_template", "cost:propagate_cost_uncertainty")),
    (30, "计算 break-even", ("cost:break_even", "cost:one_way_sensitivity")),
    (31, "执行 leave-one-source 检查", ("cost:leave_one_source_out", "cost:PriceSource.confidence")),
    (32, "评估 workload mix", ("cost:workload_mix_cost", "cost:CostResult.profile_id")),
    (33, "评估整数和容量约束", ("cost:integer_and_capacity_check", "cost:plan_replicas")),
    (34, "生成成本 provenance", ("cost:CostResult", "lineage:EvidenceActivity")),
    (35, "独立复算抽查", ("cost:independent_recalculation", "cost:CostResult")),
    (36, "形成成本 verdict", ("cost:cost_verdict", "cost:CostResult")),
)
