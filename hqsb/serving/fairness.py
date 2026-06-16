"""Service-cost fairness: cost model, Jain index, lag, starvation, HOL evidence.

Request-count fairness is wrong for LLM serving: one long request can consume
the resources of many short ones.  The accounting unit here is a *versioned
linear service cost* (``α × uncached input + β × committed output +
γ × recomputed positions``), frozen before the run; if the runtime evidence
shows the cost is clearly non-linear, the spec must switch to the segmented
function instead — a policy may not silently re-fit the weights (E08-04 §2).

Fairness is only meaningful on a **backlogged interval** for tenants that are
actually waiting: an idle tenant must never enter the Jain denominator just to
make the index look healthy (§14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.serving.policies import QueueEntry


@dataclass(frozen=True)
class CostModel:
    """The frozen linear service-cost model (``s08.cost.v1`` by default)."""

    alpha: float
    beta: float
    gamma: float
    version: str
    weights_source: str = ""
    requires_reporting: Tuple[str, ...] = ("logical_tokens", "computed_positions", "device_time")

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ConfigError("cost weights must not be negative")
        if not self.version:
            raise ConfigError("the cost model needs a version; an unversioned charge is untraceable")

    def cost(
        self,
        *,
        uncached_input_tokens: int,
        committed_output_tokens: int,
        recomputed_positions: int = 0,
    ) -> float:
        return (
            self.alpha * max(0, uncached_input_tokens)
            + self.beta * max(0, committed_output_tokens)
            + self.gamma * max(0, recomputed_positions)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alpha_uncached_input_token": self.alpha,
            "beta_committed_output_token": self.beta,
            "gamma_recomputed_position": self.gamma,
            "version": self.version,
            "weights_source": self.weights_source,
            "must_report_alongside": list(self.requires_reporting),
        }


#: Fairness guardrails that a policy must satisfy (E08-04 §8).
DEFAULT_GUARDRAILS: Mapping[str, float] = {
    "jain_min": 0.9,
    "max_service_lag_ratio": 0.25,
    "max_continuous_wait_ms": 5000.0,
    "starvation_count_max": 0.0,
    "low_priority_min_service_share": 0.05,
}


def jain_index(values: Sequence[float]) -> float:
    """``(Σ y)² / (n × Σ y²)`` over *backlogged* tenants only."""
    positives = [float(value) for value in values]
    if not positives:
        raise ConfigError("the Jain index needs at least one tenant")
    if any(value < 0 for value in positives):
        raise ConfigError("normalized service must not be negative")
    numerator = sum(positives) ** 2
    denominator = len(positives) * sum(value * value for value in positives)
    if denominator == 0:
        return 1.0
    return numerator / denominator


def normalized_service(served_cost: float, entitlement: float) -> float:
    if entitlement <= 0:
        raise ConfigError("entitlement must be positive")
    return served_cost / entitlement


def ideal_service_lag(
    served: Mapping[str, float],
    entitlements: Mapping[str, float],
    *,
    total_served: float,
) -> Dict[str, float]:
    """How far each tenant is behind its ideal weighted share (can be negative)."""
    if total_served <= 0:
        return {tenant: 0.0 for tenant in served}
    total_weight = sum(entitlements.get(tenant, 1.0) for tenant in served)
    lag: Dict[str, float] = {}
    for tenant, cost in served.items():
        share = entitlements.get(tenant, 1.0) / total_weight if total_weight else 0.0
        lag[tenant] = cost - share * total_served
    return lag


def max_continuous_wait(
    waits_ns: Sequence[int], *, threshold_ns: Optional[int] = None
) -> Dict[str, Any]:
    """Longest unbroken wait inside the backlogged interval."""
    if not waits_ns:
        return {"max_wait_ms": 0.0, "exceeds_threshold": False}
    longest = 0
    current = 0
    for wait in waits_ns:
        if threshold_ns is not None and wait >= threshold_ns:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return {
        "max_consecutive_samples": longest,
        "longest_single_wait_ms": max(waits_ns) / 1e6,
        "exceeds_threshold": bool(threshold_ns is not None and max(waits_ns) > threshold_ns),
    }


def starvation_events(served_at_ns: Sequence[Optional[int]], *, window_ns: int) -> Dict[str, Any]:
    """A tenant that got nothing inside the window is starving (§16)."""
    if not served_at_ns:
        return {"starvation_count": 0, "max_gap_ms": 0.0}
    gaps: List[int] = []
    previous: Optional[int] = None
    for timestamp in served_at_ns:
        if timestamp is None:
            continue
        if previous is not None:
            gaps.append(timestamp - previous)
        previous = timestamp
    return {
        "starvation_count": sum(1 for gap in gaps if gap > window_ns),
        "max_gap_ms": (max(gaps) / 1e6) if gaps else 0.0,
        "window_ms": window_ns / 1e6,
    }


def slowdown(shared_ms: Optional[float], isolated_ms: Optional[float]) -> Optional[float]:
    """Shared vs. matched isolated latency; needs the isolated baseline."""
    if shared_ms is None or isolated_ms is None:
        return None
    if isolated_ms <= 0:
        raise ConfigError("the isolated baseline must be positive")
    return shared_ms / isolated_ms


@dataclass(frozen=True)
class TenantOutcome:
    """One tenant/class result row (per-request aggregation happens upstream)."""

    tenant: str
    entitlement: float
    served_cost: float
    offered: int
    completed: int
    good: int
    priority: str = "normal"
    waits_ns: Tuple[int, ...] = ()
    served_at_ns: Tuple[Optional[int], ...] = ()
    shared_e2e_ms: Optional[float] = None
    isolated_e2e_ms: Optional[float] = None
    rejected: int = 0

    def __post_init__(self) -> None:
        if self.priority not in ("high", "normal", "low"):
            raise ConfigError(f"unknown priority {self.priority!r}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tenant": self.tenant,
            "entitlement": self.entitlement,
            "priority": self.priority,
            "served_cost": self.served_cost,
            "offered": self.offered,
            "completed": self.completed,
            "good": self.good,
            "rejected": self.rejected,
            "shared_e2e_ms": self.shared_e2e_ms,
            "isolated_e2e_ms": self.isolated_e2e_ms,
            "slowdown": slowdown(self.shared_e2e_ms, self.isolated_e2e_ms),
        }


def fairness_report(
    outcomes: Sequence[TenantOutcome],
    *,
    guardrails: Optional[Mapping[str, float]] = None,
    starvation_window_ns: int = 5_000_000_000,
    backlogged_only: bool = True,
) -> Dict[str, Any]:
    """Jain + lag + max wait + starvation + SLO violations, all together."""
    rules = dict(DEFAULT_GUARDRAILS)
    if guardrails:
        rules.update(dict(guardrails))
    considered = [
        item for item in outcomes if (item.offered > 0 if backlogged_only else True)
    ]
    if not considered:
        return {"ok": False, "problems": ["no backlogged tenant to judge"], "tenants": []}
    normalized = [
        normalized_service(item.served_cost, item.entitlement) for item in considered
    ]
    jain = jain_index(normalized)
    total_served = sum(item.served_cost for item in considered)
    lag = ideal_service_lag(
        {item.tenant: item.served_cost for item in considered},
        {item.tenant: item.entitlement for item in considered},
        total_served=total_served,
    )
    max_lag_ratio = max(
        (abs(value) / total_served if total_served else 0.0) for value in lag.values()
    )
    rows: List[Dict[str, Any]] = []
    starvation_total = 0
    for item in considered:
        wait = max_continuous_wait(
            item.waits_ns, threshold_ns=int(rules["max_continuous_wait_ms"] * 1e6)
        )
        starve = starvation_events(item.served_at_ns, window_ns=starvation_window_ns)
        starvation_total += starve["starvation_count"]
        rows.append(
            {
                **item.as_dict(),
                "normalized_service": normalized_service(item.served_cost, item.entitlement),
                "service_lag": lag.get(item.tenant, 0.0),
                "max_continuous_wait": wait,
                "starvation": starve,
            }
        )
    problems: List[str] = []
    if jain < rules["jain_min"]:
        problems.append(f"Jain index {jain:.3f} below the frozen minimum {rules['jain_min']}")
    if max_lag_ratio > rules["max_service_lag_ratio"]:
        problems.append(
            f"max service lag ratio {max_lag_ratio:.3f} exceeds {rules['max_service_lag_ratio']}"
        )
    if starvation_total > rules["starvation_count_max"]:
        problems.append(f"{starvation_total} starvation window(s) observed")
    low_priority = [item for item in considered if item.priority == "low"]
    low_share = None
    if total_served > 0 and low_priority:
        low_share = sum(item.served_cost for item in low_priority) / total_served
        if low_share < rules["low_priority_min_service_share"]:
            problems.append(
                f"low-priority share {low_share:.3f} is below the frozen minimum "
                f"{rules['low_priority_min_service_share']}"
            )
    return {
        "ok": not problems,
        "problems": problems,
        "jain": jain,
        "max_service_lag_ratio": max_lag_ratio,
        "low_priority_share": low_share,
        "total_served_cost": total_served,
        "guardrails": dict(rules),
        "tenants": rows,
        "note": (
            "a high Jain index does not excuse a short starvation window, and a faster "
            "high-priority class alone is not evidence of fairness"
        ),
    }


def priority_inversion_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """A low-priority class that beats a high-priority one must be visible."""
    inversions: List[Dict[str, Any]] = []
    by_priority: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_priority.setdefault(str(row.get("priority", "normal")), []).append(row)
    high = by_priority.get("high", [])
    low = by_priority.get("low", [])
    if high and low:
        high_p99 = max(float(row.get("p99_client_ttft_ms") or 0.0) for row in high)
        low_p99 = max(float(row.get("p99_client_ttft_ms") or 0.0) for row in low)
        if high_p99 > low_p99:
            inversions.append(
                {
                    "high_p99_client_ttft_ms": high_p99,
                    "low_p99_client_ttft_ms": low_p99,
                }
            )
    return {"inversions": inversions, "count": len(inversions)}


# ── HOL trace constructs (E08-04 §7) ───────────────────────────────────────

HOL_CONSTRUCTS: Tuple[str, ...] = (
    "long_first_then_short_burst",
    "short_steady_stream_with_long_insert",
    "cached_long_vs_uncached_short",
    "long_decode_holding_kv",
    "high_priority_long_vs_low_priority_short",
    "tenant_a_long_vs_tenant_b_short",
)


def hol_trace(construct: str, *, base_ns: int = 0) -> List[QueueEntry]:
    """Build one paired HOL trace (structure only, no measured latency)."""
    if construct not in HOL_CONSTRUCTS:
        raise ConfigError(
            f"unknown HOL construct {construct!r}; expected one of {list(HOL_CONSTRUCTS)}"
        )
    entries: List[QueueEntry] = []
    if construct == "long_first_then_short_burst":
        entries.append(
            QueueEntry(
                request_id="long-0",
                tenant="tenant_a",
                request_class="long_prefill",
                priority="normal",
                enqueued_ns=base_ns,
                estimated_cost=2048,
                prompt_tokens=2048,
                reserved_output_tokens=32,
            )
        )
        entries.extend(
            QueueEntry(
                request_id=f"short-{index}",
                tenant="tenant_a",
                request_class="short_chat",
                priority="normal",
                enqueued_ns=base_ns + 10_000_000 * (index + 1),
                estimated_cost=192,
                prompt_tokens=128,
                reserved_output_tokens=64,
            )
            for index in range(4)
        )
    elif construct == "short_steady_stream_with_long_insert":
        entries.extend(
            QueueEntry(
                request_id=f"short-{index}",
                tenant="tenant_a",
                request_class="short_chat",
                priority="normal",
                enqueued_ns=base_ns + 5_000_000 * index,
                estimated_cost=192,
                prompt_tokens=128,
                reserved_output_tokens=64,
            )
            for index in range(5)
        )
        entries.insert(
            2,
            QueueEntry(
                request_id="long-insert",
                tenant="tenant_a",
                request_class="long_prefill",
                priority="normal",
                enqueued_ns=base_ns + 12_000_000,
                estimated_cost=2048,
                prompt_tokens=2048,
                reserved_output_tokens=32,
            ),
        )
    elif construct == "cached_long_vs_uncached_short":
        entries.append(
            QueueEntry(
                request_id="cached-long",
                tenant="tenant_a",
                request_class="shared_prefix_high",
                priority="normal",
                enqueued_ns=base_ns,
                estimated_cost=256,
                prompt_tokens=1024,
                prefix_hit_tokens=768,
                reserved_output_tokens=64,
            )
        )
        entries.extend(
            QueueEntry(
                request_id=f"uncached-short-{index}",
                tenant="tenant_b",
                request_class="short_chat",
                priority="normal",
                enqueued_ns=base_ns + 8_000_000 * (index + 1),
                estimated_cost=192,
                prompt_tokens=128,
                reserved_output_tokens=64,
            )
            for index in range(3)
        )
    elif construct == "long_decode_holding_kv":
        entries.append(
            QueueEntry(
                request_id="long-decode",
                tenant="tenant_a",
                request_class="short_prefill_long_decode",
                priority="normal",
                enqueued_ns=base_ns,
                estimated_cost=640,
                prompt_tokens=128,
                reserved_output_tokens=512,
            )
        )
        entries.extend(
            QueueEntry(
                request_id=f"later-short-{index}",
                tenant="tenant_b",
                request_class="short_chat",
                priority="normal",
                enqueued_ns=base_ns + 20_000_000 * (index + 1),
                estimated_cost=192,
                prompt_tokens=128,
                reserved_output_tokens=64,
            )
            for index in range(3)
        )
    elif construct == "high_priority_long_vs_low_priority_short":
        entries.append(
            QueueEntry(
                request_id="high-long",
                tenant="tenant_a",
                request_class="long_prefill",
                priority="high",
                enqueued_ns=base_ns,
                estimated_cost=2048,
                prompt_tokens=2048,
                reserved_output_tokens=32,
            )
        )
        entries.extend(
            QueueEntry(
                request_id=f"low-short-{index}",
                tenant="tenant_c",
                request_class="short_chat",
                priority="low",
                enqueued_ns=base_ns + 2_000_000 * (index + 1),
                estimated_cost=192,
                prompt_tokens=128,
                reserved_output_tokens=64,
                entitlement=1.0,
            )
            for index in range(4)
        )
    else:  # tenant_a_long_vs_tenant_b_short
        entries.append(
            QueueEntry(
                request_id="a-long",
                tenant="tenant_a",
                request_class="long_prefill",
                priority="normal",
                enqueued_ns=base_ns,
                estimated_cost=2048,
                entitlement=1.0,
                prompt_tokens=2048,
                reserved_output_tokens=32,
            )
        )
        entries.extend(
            QueueEntry(
                request_id=f"b-short-{index}",
                tenant="tenant_b",
                request_class="short_chat",
                priority="normal",
                enqueued_ns=base_ns + 1_000_000 * (index + 1),
                estimated_cost=192,
                entitlement=2.0,
                prompt_tokens=128,
                reserved_output_tokens=64,
            )
            for index in range(4)
        )
    return entries


def isolated_baseline_check(
    *, isolated_available: Mapping[str, bool], homogeneous_control: bool
) -> Dict[str, Any]:
    """Slowdown needs a matched isolated baseline and a homogeneous control."""
    missing = [name for name, present in isolated_available.items() if not present]
    problems: List[str] = []
    if missing:
        problems.append(f"missing isolated baselines: {missing}")
    if not homogeneous_control:
        problems.append("the homogeneous control (equal-cost requests) is missing")
    return {
        "ok": not problems,
        "problems": problems,
        "note": "a slowdown computed without an isolated baseline is not a slowdown",
    }


def work_conserving_utilization(
    *, busy_ns: int, window_ns: int, runnable_requests_ever: bool
) -> Dict[str, Any]:
    if window_ns <= 0:
        raise ConfigError("the observation window must be positive")
    utilization = busy_ns / window_ns
    problems: List[str] = []
    if runnable_requests_ever and utilization < 0.5:
        problems.append(
            "requests were runnable but the service was idle for most of the window: "
            "fairness bookkeeping must not stop the service from working"
        )
    return {"ok": not problems, "problems": problems, "utilization": utilization}


def cache_fairness_split(
    *,
    logical_tokens: Mapping[str, int],
    computed_positions: Mapping[str, int],
    weighted_charge: Mapping[str, float],
    saved_compute: Mapping[str, float],
) -> Dict[str, Any]:
    """Logical vs. computed vs. charged service must stay separated (§10)."""
    tenants = sorted(set(logical_tokens) | set(computed_positions) | set(weighted_charge))
    rows = [
        {
            "tenant": tenant,
            "logical_tokens": logical_tokens.get(tenant, 0),
            "computed_positions": computed_positions.get(tenant, 0),
            "weighted_charge": weighted_charge.get(tenant, 0.0),
            "cache_saved_compute": saved_compute.get(tenant, 0.0),
        }
        for tenant in tenants
    ]
    return {
        "rows": rows,
        "note": (
            "whether a cache hit is charged as logical work or as real GPU work is a "
            "policy choice; the three numbers must all be reported"
        ),
    }


__all__ = [
    "CostModel",
    "DEFAULT_GUARDRAILS",
    "HOL_CONSTRUCTS",
    "TenantOutcome",
    "cache_fairness_split",
    "fairness_report",
    "hol_trace",
    "ideal_service_lag",
    "isolated_baseline_check",
    "jain_index",
    "max_continuous_wait",
    "normalized_service",
    "priority_inversion_report",
    "slowdown",
    "starvation_events",
    "work_conserving_utilization",
]
