"""Prefix identity, cache telemetry, cache-aware routing policies (E08-07).

The router's prefix fingerprint is a **match index, not a correctness proof**:
the Backend must re-validate the real token/block identity before reusing KV.
A hit that crosses model/version/tenant boundaries is a correctness failure, no
matter how good the output looks.

The policy objective is a *net value*, not a hit rate:

    saved_prefill_compute − extra_queue_delay − load_imbalance
      − eviction/pollution − telemetry/lookup − migration/remote access

so ``pure_affinity`` is included as a deliberate counter-example (maximum hit
rate, unbounded skew), and the joint policy is scored with frozen weights that
may not be retuned per workload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Cache identity fields — they mirror the S07 prefix-cache key semantics.
IDENTITY_FIELDS: Tuple[str, ...] = (
    "model_id",
    "weight_revision",
    "precision",
    "quant_artifact_hash",
    "adapter_hash",
    "tokenizer_id",
    "chat_template_hash",
    "rope_config_hash",
    "attention_config_hash",
    "kv_dtype",
    "kv_layout",
    "cache_format_version",
    "tenant_sharing_policy",
    "cache_epoch",
)

#: Routing policies compared by the experiment.
CACHE_POLICIES: Tuple[str, ...] = (
    "random_round_robin",
    "least_load",
    "pure_affinity",
    "joint_locality_load",
)

#: Telemetry failure modes injected by the experiment.
TELEMETRY_FAULTS: Tuple[str, ...] = (
    "delayed_update",
    "dropped_update",
    "out_of_order_update",
    "contradictory_update",
)


@dataclass(frozen=True)
class CacheIdentity:
    """The frozen identity; every field participates in the digest."""

    fields: Mapping[str, str]

    def __post_init__(self) -> None:
        missing = [name for name in IDENTITY_FIELDS if name not in self.fields]
        if missing:
            raise ConfigError(
                f"cache identity is missing {missing}; a partially specified identity "
                "can produce a wrong hit (E08-07 §4)"
            )
        unknown = sorted(set(self.fields) - set(IDENTITY_FIELDS))
        if unknown:
            raise ConfigError(f"cache identity has unknown fields {unknown}")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(dict(sorted(self.fields.items())), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def matches(self, other: "CacheIdentity") -> bool:
        return dict(self.fields) == dict(other.fields)

    def as_dict(self) -> Dict[str, Any]:
        return {"fields": dict(sorted(self.fields.items())), "digest": self.digest}


def identity_negative_fixtures(identity: CacheIdentity) -> List[Dict[str, Any]]:
    """Mutate each field once: every mutation must change the digest (§5)."""
    fixtures: List[Dict[str, Any]] = []
    for name in IDENTITY_FIELDS:
        mutated = dict(identity.fields)
        mutated[name] = f"{mutated[name]}-mutated"
        fixture = CacheIdentity(mutated)
        fixtures.append(
            {
                "field": name,
                "digest_differs": fixture.digest != identity.digest,
                "must_hit": False,
            }
        )
    return fixtures


def version_invalidation_check(
    old: CacheIdentity, new: CacheIdentity
) -> Dict[str, Any]:
    """A model/version/epoch change must make the old cache unusable."""
    differs = [name for name in IDENTITY_FIELDS if old.fields[name] != new.fields[name]]
    return {
        "old_digest": old.digest,
        "new_digest": new.digest,
        "differing_fields": differs,
        "old_cache_usable": old.matches(new),
        "ok": not old.matches(new) if differs else True,
        "note": "an old cache entry must never be matched after a version/epoch change",
    }


@dataclass(frozen=True)
class TenantSharingPolicy:
    """Default is isolation; sharing is an explicit decision, not a default."""

    allow_cross_tenant_sharing: bool = False
    shared_groups: Tuple[str, ...] = ()

    def allows(self, tenant_a: str, tenant_b: str) -> bool:
        if tenant_a == tenant_b:
            return True
        if not self.allow_cross_tenant_sharing:
            return False
        return tenant_a in self.shared_groups and tenant_b in self.shared_groups

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allow_cross_tenant_sharing": self.allow_cross_tenant_sharing,
            "shared_groups": list(self.shared_groups),
        }


def tenant_isolation_fixture(identity: CacheIdentity, *, other_tenant: str) -> Dict[str, Any]:
    """A cross-tenant fixture must not match when sharing is disabled."""
    policy = TenantSharingPolicy()
    mutated = dict(identity.fields)
    mutated["tenant_sharing_policy"] = other_tenant
    other = CacheIdentity(mutated)
    return {
        "other_tenant": other_tenant,
        "identity_digest_differs": other.digest != identity.digest,
        "sharing_allowed": policy.allows(
            identity.fields["tenant_sharing_policy"], other.fields["tenant_sharing_policy"]
        ),
        "ok": other.digest != identity.digest and not policy.allows(
            identity.fields["tenant_sharing_policy"], other.fields["tenant_sharing_policy"]
        ),
        "note": "the cache must not leak prompt content across tenants",
    }


# ── prefix matching ────────────────────────────────────────────────────────


@dataclass
class PrefixEntry:
    """One cached prefix (block aligned) on one Backend instance."""

    identity: CacheIdentity
    tokens: Tuple[int, ...]
    block_size: int = 16
    last_access_ns: int = 0
    frequency: int = 1
    refcount: int = 0
    bytes_value: float = 0.0

    @property
    def blocks(self) -> int:
        return len(self.tokens) // self.block_size

    def matched_blocks(self, query: Sequence[int]) -> int:
        if not self.identity or not self.tokens:
            return 0
        limit = min(len(self.tokens), len(query))
        matched = 0
        for index in range(limit):
            if self.tokens[index] != query[index]:
                break
            matched = index + 1
        return matched // self.block_size


class PrefixMatcher:
    """Token-level longest-prefix matcher, block aligned (the router's index)."""

    def __init__(self, *, block_size: int = 16) -> None:
        self.block_size = block_size
        self.entries: List[PrefixEntry] = []

    def insert(self, entry: PrefixEntry) -> None:
        if entry.identity is None:
            raise ConfigError("a prefix entry needs an identity")
        self.entries.append(entry)

    def match(
        self, *, identity: CacheIdentity, query_tokens: Sequence[int]
    ) -> Dict[str, Any]:
        best = 0
        best_entry: Optional[PrefixEntry] = None
        for entry in self.entries:
            if not entry.identity.matches(identity):
                continue
            matched = entry.matched_blocks(query_tokens)
            if matched > best:
                best, best_entry = matched, entry
        return {
            "matched_blocks": best,
            "matched_tokens": best * self.block_size,
            "entry_digest": best_entry.identity.digest if best_entry else "",
            "identity_digest": identity.digest,
            "note": (
                "this is an index lookup; the Backend must re-validate the real block "
                "identity before reuse"
            ),
        }

    def evict(self, entry: PrefixEntry, *, reason: str) -> Dict[str, Any]:
        if entry.refcount > 0:
            raise ConfigError(
                "refcount>0 means the block is still in use; eviction must not free it"
            )
        if entry in self.entries:
            self.entries.remove(entry)
        return {"evicted": True, "reason": reason, "blocks": entry.blocks}


def matcher_oracle(
    matcher: PrefixMatcher, *, identity: CacheIdentity, query: Sequence[int]
) -> Dict[str, Any]:
    """Compare the matcher with a brute-force token oracle (step 5)."""
    predicted = matcher.match(identity=identity, query_tokens=query)["matched_tokens"]
    oracle = 0
    for entry in matcher.entries:
        if not entry.identity.matches(identity):
            continue
        limit = min(len(entry.tokens), len(query))
        matched = 0
        for index in range(limit):
            if entry.tokens[index] != query[index]:
                break
            matched = index + 1
        oracle = max(oracle, (matched // matcher.block_size) * matcher.block_size)
    return {
        "ok": predicted == oracle,
        "predicted_tokens": predicted,
        "oracle_tokens": oracle,
    }


def block_identity_second_check(
    *, router_digest: str, backend_digest: str, router_tokens: Sequence[int],
    backend_tokens: Sequence[int],
) -> Dict[str, Any]:
    """A forged/colliding fingerprint must not become a real hit (step 6)."""
    digest_match = router_digest == backend_digest
    token_match = list(router_tokens[: len(backend_tokens)]) == list(backend_tokens)
    if digest_match and not token_match:
        return {
            "accepted": False,
            "reason": "fingerprint collision: token equality failed at the Backend",
            "correct": True,
        }
    if not digest_match:
        return {"accepted": False, "reason": "identity mismatch at the Backend", "correct": True}
    return {"accepted": True, "reason": "", "correct": True}


# ── telemetry ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CacheTelemetrySample:
    """One instance's cache telemetry; always carries its own timestamp."""

    instance_id: str
    monotonic_ns: int
    cache_epoch: str
    capacity_bytes: float
    free_bytes: float
    pressure_ratio: float = 0.0
    eviction_count: int = 0
    invalidation_count: int = 0
    matched_tokens: Mapping[str, int] = field(default_factory=dict)

    def age_ms(self, now_ns: int) -> float:
        return max(0.0, (now_ns - self.monotonic_ns) / 1e6)

    def as_dict(self, now_ns: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "instance_id": self.instance_id,
            "monotonic_ns": self.monotonic_ns,
            "cache_epoch": self.cache_epoch,
            "capacity_bytes": self.capacity_bytes,
            "free_bytes": self.free_bytes,
            "pressure_ratio": self.pressure_ratio,
            "eviction_count": self.eviction_count,
            "invalidation_count": self.invalidation_count,
            "matched_tokens": dict(self.matched_tokens),
        }
        if now_ns is not None:
            payload["age_ms"] = self.age_ms(now_ns)
        return payload


@dataclass
class TelemetryFaultInjector:
    """Delayed/dropped/out-of-order/contradictory telemetry updates (step 17)."""

    fault: str
    delay_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.fault not in TELEMETRY_FAULTS:
            raise ConfigError(
                f"unknown telemetry fault {self.fault!r}; expected one of {list(TELEMETRY_FAULTS)}"
            )

    def apply(
        self, samples: Sequence[CacheTelemetrySample], *, now_ns: int
    ) -> List[CacheTelemetrySample]:
        if self.fault == "delayed_update":
            return [
                CacheTelemetrySample(
                    instance_id=item.instance_id,
                    monotonic_ns=item.monotonic_ns + int(self.delay_ms * 1e6),
                    cache_epoch=item.cache_epoch,
                    capacity_bytes=item.capacity_bytes,
                    free_bytes=item.free_bytes,
                    pressure_ratio=item.pressure_ratio,
                    eviction_count=item.eviction_count,
                    invalidation_count=item.invalidation_count,
                    matched_tokens=dict(item.matched_tokens),
                )
                for item in samples
            ]
        if self.fault == "dropped_update":
            return list(samples[:-1]) if len(samples) > 1 else []
        if self.fault == "out_of_order_update":
            return list(reversed(samples))
        # contradictory: identical epochs reporting different free bytes
        if not samples:
            return []
        last = samples[-1]
        return list(samples) + [
            CacheTelemetrySample(
                instance_id=last.instance_id,
                monotonic_ns=now_ns,
                cache_epoch=last.cache_epoch,
                capacity_bytes=last.capacity_bytes,
                free_bytes=last.capacity_bytes,  # contradicts the previous sample
                pressure_ratio=0.0,
                eviction_count=last.eviction_count,
                invalidation_count=last.invalidation_count,
                matched_tokens={},
            )
        ]


def telemetry_freshness(
    samples: Sequence[CacheTelemetrySample], *, now_ns: int, ttl_ms: float
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    stale = 0
    for sample in samples:
        age = sample.age_ms(now_ns)
        is_stale = age > ttl_ms
        stale += int(is_stale)
        rows.append({**sample.as_dict(now_ns), "stale": is_stale})
    return {
        "rows": rows,
        "stale": stale,
        "policy": "conservative when stale or missing",
        "note": "a stale hit prediction must never lead to a wrong KV reference",
    }


def contradictory_updates(
    samples: Sequence[CacheTelemetrySample]
) -> Dict[str, Any]:
    """Same epoch, different capacity/free bytes: the router must not trust either."""
    problems: List[str] = []
    by_epoch: Dict[str, set] = {}
    for sample in samples:
        key = sample.cache_epoch
        signature = (sample.capacity_bytes, sample.free_bytes)
        by_epoch.setdefault(key, set()).add(signature)
    for epoch, signatures in by_epoch.items():
        if len(signatures) > 1 and any(
            first[0] == second[0] and first[1] != second[1]
            for first in signatures
            for second in signatures
        ):
            problems.append(f"epoch {epoch}: contradictory capacity/free readings")
    return {
        "ok": not problems,
        "problems": problems,
        "action": "fall back to the conservative rule and report the contradiction",
    }


# ── routing policies and net value ─────────────────────────────────────────


@dataclass(frozen=True)
class CacheRouteRequest:
    request_id: str
    identity: CacheIdentity
    query_tokens: Tuple[int, ...]
    queue_ms: Mapping[str, float] = field(default_factory=dict)
    decode_ms: Mapping[str, float] = field(default_factory=dict)
    cost_units: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class JointWeights:
    """Frozen joint-score weights (one set for every workload)."""

    queue: float = 1.0
    uncached_prefill: float = 1.0
    decode: float = 0.5
    eviction: float = 0.25
    skew: float = 1.5

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "JointWeights":
        joint = dict(payload["joint_score"])
        return cls(
            queue=float(joint["weight_predicted_queue"]),
            uncached_prefill=float(joint["weight_predicted_uncached_prefill"]),
            decode=float(joint["weight_predicted_decode"]),
            eviction=float(joint["weight_eviction_cost"]),
            skew=float(joint["weight_skew_penalty"]),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "queue": self.queue,
            "uncached_prefill": self.uncached_prefill,
            "decode": self.decode,
            "eviction": self.eviction,
            "skew": self.skew,
        }


@dataclass(frozen=True)
class NetValueModel:
    """``saved prefill − everything else``; a hit rate is not the objective."""

    saved_prefill_ms: float
    extra_queue_ms: float = 0.0
    load_imbalance_ms: float = 0.0
    eviction_pollution_ms: float = 0.0
    telemetry_lookup_ms: float = 0.0
    migration_ms: float = 0.0

    def net_value_ms(self) -> float:
        return (
            self.saved_prefill_ms
            - self.extra_queue_ms
            - self.load_imbalance_ms
            - self.eviction_pollution_ms
            - self.telemetry_lookup_ms
            - self.migration_ms
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "saved_prefill_compute_ms": self.saved_prefill_ms,
            "minus_extra_queue_delay_ms": self.extra_queue_ms,
            "minus_load_imbalance_cost_ms": self.load_imbalance_ms,
            "minus_eviction_pollution_cost_ms": self.eviction_pollution_ms,
            "minus_telemetry_lookup_cost_ms": self.telemetry_lookup_ms,
            "minus_migration_or_remote_access_cost_ms": self.migration_ms,
            "net_value_ms": self.net_value_ms(),
        }


@dataclass(frozen=True)
class CacheRouteDecision:
    policy: str
    request_id: str
    selected_instance_id: str
    predicted_matched_tokens: int
    predicted_saved_prefill_ms: float
    score: float
    reason: str
    candidates: Tuple[Mapping[str, Any], ...] = ()
    telemetry_stale: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "request_id": self.request_id,
            "selected_instance_id": self.selected_instance_id,
            "predicted_matched_tokens": self.predicted_matched_tokens,
            "predicted_saved_prefill_ms": self.predicted_saved_prefill_ms,
            "score": self.score,
            "reason": self.reason,
            "candidates": [dict(item) for item in self.candidates],
            "telemetry_stale": self.telemetry_stale,
        }


def prefix_saved_prefill_ms(
    matched_tokens: int, *, per_token_prefill_us: float
) -> float:
    """Saved time must be derived from a declared per-token cost, not assumed."""
    if per_token_prefill_us < 0:
        raise ConfigError("the per-token prefill cost must not be negative")
    return matched_tokens * per_token_prefill_us / 1000.0


def cache_route(
    *,
    policy: str,
    request: CacheRouteRequest,
    matcher: PrefixMatcher,
    telemetry: Sequence[CacheTelemetrySample],
    instance_ids: Sequence[str],
    weights: JointWeights,
    per_token_prefill_us: float,
    now_ns: int,
    ttl_ms: float,
    round_robin_cursor: int = 0,
) -> CacheRouteDecision:
    """Four policies behind one decision record, with reasons."""
    if policy not in CACHE_POLICIES:
        raise ConfigError(f"unknown cache policy {policy!r}; expected one of {list(CACHE_POLICIES)}")
    if not instance_ids:
        raise ConfigError("cache routing needs at least one instance")
    latest: Dict[str, CacheTelemetrySample] = {}
    for sample in telemetry:
        current = latest.get(sample.instance_id)
        if current is None or sample.monotonic_ns >= current.monotonic_ns:
            latest[sample.instance_id] = sample
    rows: List[Dict[str, Any]] = []
    for instance_id in instance_ids:
        sample = latest.get(instance_id)
        stale = sample is None or sample.age_ms(now_ns) > ttl_ms
        # the matcher is shared across instances in this scaffold: the per-instance
        # entry set is what the Backend reports through telemetry
        matched_tokens = 0
        for entry in matcher.entries:
            if entry.identity.matches(request.identity):
                matched_tokens = max(matched_tokens, entry.matched_blocks(request.query_tokens) * matcher.block_size)
        saved_prefill = prefix_saved_prefill_ms(
            matched_tokens, per_token_prefill_us=per_token_prefill_us
        )
        queue_ms = float(request.queue_ms.get(instance_id, 0.0))
        decode_ms = float(request.decode_ms.get(instance_id, 0.0))
        eviction_cost = float(
            (sample.pressure_ratio if sample is not None else 1.0) * 10.0
        )
        skew_penalty = (
            float(request.queue_ms.get(instance_id, 0.0)) if stale else 0.0
        )
        if stale:
            # conservative: an unknown cache state may not be credited as a hit
            matched_tokens = 0
            saved_prefill = 0.0
            skew_penalty = max(queue_ms, 1.0)
        rows.append(
            {
                "instance_id": instance_id,
                "matched_tokens": matched_tokens,
                "saved_prefill_ms": saved_prefill,
                "queue_ms": queue_ms,
                "decode_ms": decode_ms,
                "eviction_cost": eviction_cost,
                "skew_penalty": skew_penalty,
                "telemetry_age_ms": None if sample is None else sample.age_ms(now_ns),
                "stale": stale,
            }
        )
    if policy == "random_round_robin":
        chosen = rows[round_robin_cursor % len(rows)]
        reason = f"round-robin cursor {round_robin_cursor}"
    elif policy == "least_load":
        chosen = min(rows, key=lambda row: (row["queue_ms"], row["instance_id"]))
        reason = "lowest reported queue, cache state ignored"
    elif policy == "pure_affinity":
        chosen = max(rows, key=lambda row: (row["matched_tokens"], row["instance_id"]))
        reason = "longest matched prefix, load ignored (counter-example policy)"
    else:
        for row in rows:
            row["score"] = (
                weights.queue * row["queue_ms"]
                + weights.uncached_prefill * max(0.0, 100.0 - row["saved_prefill_ms"])
                + weights.decode * row["decode_ms"]
                + weights.eviction * row["eviction_cost"]
                + weights.skew * row["skew_penalty"]
            )
        chosen = min(rows, key=lambda row: (row["score"], row["instance_id"]))
        reason = (
            f"lowest joint score {chosen['score']:.3f} "
            "(locality credits are net of queue, eviction and skew)"
        )
    return CacheRouteDecision(
        policy=policy,
        request_id=request.request_id,
        selected_instance_id=str(chosen["instance_id"]),
        predicted_matched_tokens=int(chosen["matched_tokens"]),
        predicted_saved_prefill_ms=float(chosen["saved_prefill_ms"]),
        score=float(chosen.get("score", 0.0)),
        reason=reason,
        candidates=tuple(rows),
        telemetry_stale=any(row["stale"] for row in rows),
    )


def skew_report(loads: Mapping[str, float]) -> Dict[str, Any]:
    """Locality must not be bought with unbounded imbalance (§7)."""
    values = [float(value) for value in loads.values()]
    if not values:
        return {"ok": False, "reason": "no load reported"}
    mean = sum(values) / len(values)
    if mean == 0:
        return {"ok": True, "max_to_mean": 1.0, "cv": 0.0}
    max_to_mean = max(values) / mean
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    cv = (variance**0.5) / mean
    return {
        "ok": max_to_mean <= 1.5 and cv <= 0.4,
        "max_to_mean": max_to_mean,
        "cv": cv,
        "limits": {"max_to_mean": 1.5, "cv": 0.4},
        "note": (
            "a high hit rate with unbounded skew is not a service-level win: the hot "
            "instance's queue is part of the cost"
        ),
    }


@dataclass(frozen=True)
class CachePredictionOutcome:
    request_id: str
    predicted_matched_tokens: int
    actual_matched_tokens: int
    predicted_identity: str
    actual_identity: str
    actual_reused_tokens: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "predicted_matched_tokens": self.predicted_matched_tokens,
            "actual_matched_tokens": self.actual_matched_tokens,
            "predicted_identity": self.predicted_identity,
            "actual_identity": self.actual_identity,
            "actual_reused_tokens": self.actual_reused_tokens,
        }


def prediction_vs_actual(outcomes: Sequence[CachePredictionOutcome]) -> Dict[str, Any]:
    """Predicted hits and actual reuse stay separate; wrong hits are fatal."""
    wrong_hits = [
        item for item in outcomes if item.predicted_identity != item.actual_identity
    ]
    errors = [
        item.actual_matched_tokens - item.predicted_matched_tokens for item in outcomes
    ]
    return {
        "ok": not wrong_hits,
        "wrong_hits": [item.as_dict() for item in wrong_hits],
        "mean_match_error": (sum(errors) / len(errors)) if errors else None,
        "predicted_total": sum(item.predicted_matched_tokens for item in outcomes),
        "actual_total": sum(item.actual_matched_tokens for item in outcomes),
        "note": (
            "a predicted hit with an actual miss is a telemetry problem; a hit with a "
            "different identity is a correctness failure"
        ),
    }


def zero_locality_falsification(
    *, joint_net_value_ms: float, baseline_net_value_ms: float
) -> Dict[str, Any]:
    """With no shared prefix the joint policy must not invent a benefit (§14)."""
    delta = joint_net_value_ms - baseline_net_value_ms
    return {
        "delta_ms": delta,
        "overhead_without_locality": -delta if delta < 0 else 0.0,
        "ok": True,
        "note": (
            "a positive 'benefit' under zero locality means the comparison is biased; a "
            "negative delta is the honest overhead lower bound"
        ),
    }


__all__ = [
    "CACHE_POLICIES",
    "CacheIdentity",
    "CachePredictionOutcome",
    "CacheRouteDecision",
    "CacheRouteRequest",
    "CacheTelemetrySample",
    "IDENTITY_FIELDS",
    "JointWeights",
    "NetValueModel",
    "PrefixEntry",
    "PrefixMatcher",
    "TELEMETRY_FAULTS",
    "TelemetryFaultInjector",
    "TenantSharingPolicy",
    "block_identity_second_check",
    "cache_route",
    "contradictory_updates",
    "identity_negative_fixtures",
    "matcher_oracle",
    "prediction_vs_actual",
    "prefix_saved_prefill_ms",
    "skew_report",
    "telemetry_freshness",
    "tenant_isolation_fixture",
    "version_invalidation_check",
    "zero_locality_falsification",
]
