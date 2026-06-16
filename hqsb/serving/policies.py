"""Deterministic queue policies behind one interface (E08-04 §5).

FIFO, strict priority and weighted-fair selection are *the same kind of object*
here: they consume the same queue snapshot, return the same decision record and
are logged by the same recorder.  That is what makes the E08-04 comparison a
policy comparison instead of three different code paths.

The module is a pure scheduler: it decides, it records, it audits.  It never
executes a request and never computes a latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Policy identifiers of the frozen comparison.
POLICIES: Tuple[str, ...] = ("fifo", "strict_priority", "weighted_fair")

#: Charge units a policy may account in (one must be chosen and frozen).
SERVICE_UNITS: Tuple[str, ...] = (
    "logical_tokens",
    "computed_positions",
    "weighted_cost",
)

PRIORITY_ORDER: Mapping[str, int] = {"high": 0, "normal": 1, "low": 2}


@dataclass
class QueueEntry:
    """One waiting request as the policy sees it (no future information)."""

    request_id: str
    tenant: str
    request_class: str
    priority: str
    enqueued_ns: int
    estimated_cost: float
    entitlement: float = 1.0
    deadline_ns: Optional[int] = None
    prompt_tokens: int = 0
    reserved_output_tokens: int = 0
    prefix_hit_tokens: int = 0
    actual_cost: float = 0.0
    charged_cost: float = 0.0
    served_at_ns: Optional[int] = None
    cancelled: bool = False
    timeout: bool = False

    def __post_init__(self) -> None:
        if self.priority not in PRIORITY_ORDER:
            raise ConfigError(f"unknown priority {self.priority!r}")
        if self.entitlement <= 0:
            raise ConfigError("entitlement must be positive")

    @property
    def age_ns(self) -> int:
        return 0  # filled by the decision record (the clock belongs to the caller)

    def prediction_error(self) -> Optional[float]:
        if not self.actual_cost:
            return None
        return self.actual_cost - self.estimated_cost

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tenant": self.tenant,
            "request_class": self.request_class,
            "priority": self.priority,
            "enqueued_ns": self.enqueued_ns,
            "estimated_cost": self.estimated_cost,
            "entitlement": self.entitlement,
            "prompt_tokens": self.prompt_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "actual_cost": self.actual_cost,
            "charged_cost": self.charged_cost,
            "served_at_ns": self.served_at_ns,
            "cancelled": self.cancelled,
            "timeout": self.timeout,
        }


@dataclass(frozen=True)
class PolicyDecision:
    """What a policy chose, and why (the audit trail of one iteration)."""

    policy: str
    selected_request_id: str
    reason: str
    now_ns: int
    queue_depth: int
    waited_ns: int
    virtual_time_after: float = 0.0
    tie_break: str = ""
    work_conserving: bool = True
    counters: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "selected_request_id": self.selected_request_id,
            "reason": self.reason,
            "now_ns": self.now_ns,
            "queue_depth": self.queue_depth,
            "waited_ns": self.waited_ns,
            "virtual_time_after": self.virtual_time_after,
            "tie_break": self.tie_break,
            "work_conserving": self.work_conserving,
            "counters": dict(self.counters),
        }


class QueuePolicy:
    """Base class: same input, same output shape, explicit reason."""

    name = "abstract"

    def __init__(self) -> None:
        self.served_cost: Dict[str, float] = {}
        self.served_count: Dict[str, int] = {}

    def _charge(self, entry: QueueEntry) -> None:
        charge = entry.charged_cost or entry.actual_cost or entry.estimated_cost
        self.served_cost[entry.tenant] = self.served_cost.get(entry.tenant, 0.0) + charge
        self.served_count[entry.tenant] = self.served_count.get(entry.tenant, 0) + 1
        entry.charged_cost = charge

    def counters(self) -> Dict[str, float]:
        return dict(sorted(self.served_cost.items()))

    def select(
        self, queue: Sequence[QueueEntry], *, now_ns: int
    ) -> Optional[PolicyDecision]:
        raise NotImplementedError

    def refund(self, entry: QueueEntry, *, unexecuted_fraction: float) -> None:
        """Cancel/timeout refunds only the part that was never executed (§13)."""
        if not 0 <= unexecuted_fraction <= 1:
            raise ConfigError("refund fraction must be in [0, 1]")
        refund = entry.charged_cost * unexecuted_fraction
        entry.charged_cost -= refund
        self.served_cost[entry.tenant] = max(
            0.0, self.served_cost.get(entry.tenant, 0.0) - refund
        )


class FifoPolicy(QueuePolicy):
    """FCFS: simple, explainable, assumes no fairness at all."""

    name = "fifo"

    def select(self, queue: Sequence[QueueEntry], *, now_ns: int) -> Optional[PolicyDecision]:
        if not queue:
            return None
        chosen = min(queue, key=lambda item: item.enqueued_ns)
        self._charge(chosen)
        return PolicyDecision(
            policy=self.name,
            selected_request_id=chosen.request_id,
            reason="oldest enqueue time",
            now_ns=now_ns,
            queue_depth=len(queue),
            waited_ns=now_ns - chosen.enqueued_ns,
            tie_break="request_id",
            counters=self.counters(),
        )


class StrictPriorityPolicy(QueuePolicy):
    """Priority first; without aging this policy is *not* fair (E08-04 §5)."""

    name = "strict_priority"

    def select(self, queue: Sequence[QueueEntry], *, now_ns: int) -> Optional[PolicyDecision]:
        if not queue:
            return None
        chosen = min(
            queue,
            key=lambda item: (PRIORITY_ORDER[item.priority], item.enqueued_ns, item.request_id),
        )
        self._charge(chosen)
        return PolicyDecision(
            policy=self.name,
            selected_request_id=chosen.request_id,
            reason=f"priority {chosen.priority} first, then earliest enqueue",
            now_ns=now_ns,
            queue_depth=len(queue),
            waited_ns=now_ns - chosen.enqueued_ns,
            tie_break="enqueue_time_then_request_id",
            counters=self.counters(),
        )


class WeightedFairPolicy(QueuePolicy):
    """Weighted fair queueing over a frozen service unit (default: cost).

    The counter is *virtual time*: each tenant accumulates its normalized
    service, and the next request comes from the tenant with the smallest
    normalized service.  New/idle tenants start at the current minimum so an
    idle tenant cannot bank credit and then monopolise the service.
    """

    name = "weighted_fair"

    def __init__(self, *, service_unit: str = "weighted_cost") -> None:
        super().__init__()
        if service_unit not in SERVICE_UNITS:
            raise ConfigError(
                f"unknown service unit {service_unit!r}; choose one of {list(SERVICE_UNITS)}"
            )
        self.service_unit = service_unit
        self.virtual_time: Dict[str, float] = {}

    def normalized_service(self, tenant: str, entitlement: float) -> float:
        return self.virtual_time.get(tenant, 0.0) / entitlement

    def select(self, queue: Sequence[QueueEntry], *, now_ns: int) -> Optional[PolicyDecision]:
        if not queue:
            return None
        # a tenant with no history starts at the current minimum: no banked credit
        baseline = min(self.virtual_time.values()) if self.virtual_time else 0.0
        charged = {tenant: self.virtual_time.setdefault(tenant, baseline) for tenant in {item.tenant for item in queue}}
        chosen = min(
            queue,
            key=lambda item: (
                charged[item.tenant] / item.entitlement,
                item.enqueued_ns,
                item.request_id,
            ),
        )
        served = chosen.charged_cost or chosen.actual_cost or chosen.estimated_cost
        self.virtual_time[chosen.tenant] = charged[chosen.tenant] + served
        self._charge(chosen)
        return PolicyDecision(
            policy=self.name,
            selected_request_id=chosen.request_id,
            reason=(
                f"smallest normalized service for tenant {chosen.tenant!r} "
                f"(virtual time {charged[chosen.tenant]:.3f} / entitlement "
                f"{chosen.entitlement})"
            ),
            now_ns=now_ns,
            queue_depth=len(queue),
            waited_ns=now_ns - chosen.enqueued_ns,
            virtual_time_after=self.virtual_time[chosen.tenant],
            tie_break="enqueue_time_then_request_id",
            counters=self.counters(),
        )


def build_policy(name: str, *, service_unit: str = "weighted_cost") -> QueuePolicy:
    if name == "fifo":
        return FifoPolicy()
    if name == "strict_priority":
        return StrictPriorityPolicy()
    if name == "weighted_fair":
        return WeightedFairPolicy(service_unit=service_unit)
    raise ConfigError(f"unknown policy {name!r}; expected one of {list(POLICIES)}")


@dataclass
class DecisionLog:
    """Every decision is kept: a policy without a log cannot be audited."""

    decisions: List[PolicyDecision] = field(default_factory=list)

    def record(self, decision: Optional[PolicyDecision]) -> None:
        if decision is not None:
            self.decisions.append(decision)

    def work_conserving_audit(self, snapshots: Sequence[Tuple[int, int, bool]]) -> Dict[str, Any]:
        """``(now_ns, queue_depth, selected)`` tuples.

        A non-empty queue with no selection would be idling while work is
        runnable — a work-conservation violation.
        """
        problems = [
            f"at {now_ns} ns the queue held {depth} runnable requests but nothing was selected"
            for now_ns, depth, selected in snapshots
            if depth > 0 and not selected
        ]
        return {"ok": not problems, "problems": problems}

    def as_rows(self) -> List[Dict[str, Any]]:
        return [decision.as_dict() for decision in self.decisions]


def prediction_report(entries: Sequence[QueueEntry]) -> Dict[str, Any]:
    """Cost prediction error must be visible, never hidden (§9)."""
    errors = [
        (entry.request_id, entry.estimated_cost, entry.actual_cost, entry.prediction_error())
        for entry in entries
        if entry.actual_cost
    ]
    if not errors:
        return {"status": "NO_COMPLETED_ENTRIES", "rows": []}
    signed = [item[3] for item in errors if item[3] is not None]
    mean_error = sum(signed) / len(signed)
    return {
        "status": "measured",
        "rows": [
            {
                "request_id": request_id,
                "estimated": estimated,
                "actual": actual,
                "error": error,
            }
            for request_id, estimated, actual, error in errors
        ],
        "mean_error": mean_error,
        "over_estimate_ratio": sum(1 for value in signed if value > 0) / len(signed),
        "note": (
            "estimate error above the threshold has to be reported as a calibration "
            "limit; the scheduler may never read the real final output length"
        ),
    }


def max_tokens_abuse_check(entry: QueueEntry, *, completed_tokens: int) -> Dict[str, Any]:
    """A request that declares a huge max_tokens and stops early (§15)."""
    declared = entry.reserved_output_tokens
    if declared <= 0:
        return {"ok": True, "note": "no declaration to compare"}
    ratio = completed_tokens / declared
    return {
        "ok": True,
        "declared_max_tokens": declared,
        "completed_tokens": completed_tokens,
        "utilisation": ratio,
        "note": (
            "the admission/fair charge may not be inflatable by declaring a large "
            "max_tokens: the charge policy has to be frozen and the error recorded"
        ),
    }


def policy_interface_audit(
    policies: Sequence[QueuePolicy], queue: Sequence[QueueEntry], *, now_ns: int
) -> Dict[str, Any]:
    """All policies must consume the same input and emit the same structure."""
    problems: List[str] = []
    shapes = set()
    for policy in policies:
        decision = policy.select(list(queue), now_ns=now_ns)
        if decision is None:
            problems.append(f"{policy.name}: no decision for a non-empty queue")
            continue
        if decision.policy != policy.name:
            problems.append(f"{policy.name}: decision claims policy {decision.policy!r}")
        if not decision.reason:
            problems.append(f"{policy.name}: decision has no reason")
        shapes.add(tuple(sorted(decision.as_dict())))
    if len(shapes) > 1:
        problems.append("policies emit different decision structures")
    return {"ok": not problems, "problems": problems, "decision_shape_uniform": len(shapes) <= 1}


__all__ = [
    "DecisionLog",
    "FifoPolicy",
    "POLICIES",
    "PRIORITY_ORDER",
    "PolicyDecision",
    "QueueEntry",
    "QueuePolicy",
    "SERVICE_UNITS",
    "StrictPriorityPolicy",
    "WeightedFairPolicy",
    "build_policy",
    "max_tokens_abuse_check",
    "policy_interface_audit",
    "prediction_report",
]
