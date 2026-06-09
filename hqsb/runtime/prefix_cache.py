"""Prefix-cache key semantics, hit accounting and eviction (E07-05).

Three things are enforced here because they are the difference between a useful
prefix cache and a silent correctness bug:

1. **The key binds every KV-affecting identity** (:data:`IDENTITY_FIELDS`):
   model/weights revision, precision/quantisation, tokenizer + chat template,
   RoPE/attention semantics, cache layout version, block group, tenant domain and
   multimodal identity — plus, per candidate entry, the **parent chain digest**
   (tokens before the span) and the **block token sequence hash** (the span's own
   tokens), which together form the entry digest.  Raw text is never a key.
2. **A digest is not a hit by itself.**  Even with a SHA-256 chain the cache
   verifies token equality under the frozen default
   (``digest_plus_token_equality``); the forced-collision fixture shows a
   same-digest / different-content record being rejected instead of served, and
   why ``strong_digest`` alone is unsafe.
3. **Saved tokens are not saved time.**  :class:`NetSavingModel` keeps the
   measured prefill, the lookup/hash cost, the eviction/recompute side effects
   and the impact on other requests apart, so no report can multiply
   ``hit_tokens × constant``.

Scope note (honest boundary): an entry covers a block chain that starts at token
offset 0 of the prompt.  Chains starting at a non-zero offset (a radix tree whose
internal nodes carry their own absolute start) are **not** implemented; extending
:class:`PrefixKey` with an explicit start offset is the documented next step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Fields that must be part of the cache key (details README §12 / E07-05 §3).
CACHE_KEY_FIELDS: Tuple[str, ...] = (
    "model_id",
    "weight_revision",
    "precision",
    "quant_artifact_hash",
    "adapter_hash",
    "tokenizer_id",
    "chat_template_hash",
    "rope_config_hash",
    "attention_config_hash",
    "cache_layout_version",
    "block_group",
    "tenant_domain",
    "multimodal_identity",
)

#: Span-specific fields of an entry; together with :data:`CACHE_KEY_FIELDS` they
#: form the complete key (and therefore the digest).
SPAN_KEY_FIELDS: Tuple[str, ...] = (
    "parent_chain_digest",
    "block_token_sequence_hash",
)

ALL_KEY_FIELDS: Tuple[str, ...] = CACHE_KEY_FIELDS + SPAN_KEY_FIELDS

#: Alias kept for the protocol wording ("key bound to every KV-affecting field").
IDENTITY_FIELDS: Tuple[str, ...] = CACHE_KEY_FIELDS

SECURITY_DOMAINS: Tuple[str, ...] = ("shared", "tenant", "request")

COLLISION_POLICIES: Tuple[str, ...] = ("strong_digest", "digest_plus_token_equality")


def token_digest(tokens: Sequence[int]) -> str:
    """Stable hash of a token-ID sequence (the comparison unit of S07)."""
    payload = ",".join(str(int(token)) for token in tokens)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PrefixKey:
    """Prefix identity plus the *content* it claims to describe."""

    fields: Mapping[str, str]
    tokens: Tuple[int, ...] = ()
    digest: str = ""

    def __post_init__(self) -> None:
        missing = [name for name in ALL_KEY_FIELDS if name not in self.fields]
        if missing:
            raise ConfigError(
                "prefix key is incomplete: missing "
                + ", ".join(missing)
                + ". A key that omits KV-affecting identity produces wrong hits "
                "that cannot be explained afterwards",
                details={"fields": missing},
            )

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def content_digest(self) -> str:
        return token_digest(self.tokens)

    def canonical(self) -> str:
        """Everything the digest covers: identity, span identity and content."""
        payload = {name: self.fields[name] for name in ALL_KEY_FIELDS}
        payload["content_digest"] = self.content_digest
        return json.dumps(payload, sort_keys=True)

    def compute_digest(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    @property
    def effective_digest(self) -> str:
        return self.digest or self.compute_digest()

    @property
    def parent_chain_digest(self) -> str:
        return self.fields["parent_chain_digest"]

    @property
    def block_token_sequence_hash(self) -> str:
        return self.fields["block_token_sequence_hash"]

    def identity_matches(self, other: "PrefixKey") -> bool:
        """Model/precision/tokenizer/layout/domain part of the key."""
        return all(
            self.fields[name] == other.fields[name] for name in IDENTITY_FIELDS
        )

    def span_matches(self, other: "PrefixKey") -> bool:
        """Parent chain + block token sequence part of the key."""
        return all(
            self.fields[name] == other.fields[name] for name in SPAN_KEY_FIELDS
        )

    def content_matches(self, other: "PrefixKey") -> bool:
        return self.tokens == other.tokens

    def matches(self, other: "PrefixKey") -> bool:
        """Full identity + span + content match."""
        return (
            self.identity_matches(other)
            and self.span_matches(other)
            and self.content_matches(other)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fields": dict(self.fields),
            "length": self.length,
            "content_digest": self.content_digest,
            "digest": self.effective_digest,
        }


def build_prefix_key(
    *,
    identity: Mapping[str, str],
    tokens: Sequence[int],
    token_span: Tuple[int, int],
    parent_chain_digest: str = "",
    block_group: str = "full_attention",
    tenant_domain: str = "shared",
    cache_layout_version: str = "1.0.0",
    multimodal_identity: str = "",
    block_size: int = 0,
) -> PrefixKey:
    """Build a key from *final token IDs* plus the model-side identity.

    ``block_size`` > 0 truncates the span to complete blocks, because a block
    cache can only reuse whole blocks (details README §2).
    """
    start, end = token_span
    if start < 0 or end > len(tokens):
        raise ConfigError(
            "prefix span falls outside the token sequence",
            details={"field": "token_span"},
        )
    if block_size:
        end = start + ((end - start) // block_size) * block_size
    span = list(tokens[start:end])
    fields = {
        "model_id": identity.get("model_id", ""),
        "weight_revision": identity.get("weight_revision", ""),
        "precision": identity.get("precision", ""),
        "quant_artifact_hash": identity.get("quant_artifact_hash", ""),
        "adapter_hash": identity.get("adapter_hash", ""),
        "tokenizer_id": identity.get("tokenizer_id", ""),
        "chat_template_hash": identity.get("chat_template_hash", ""),
        "rope_config_hash": identity.get("rope_config_hash", ""),
        "attention_config_hash": identity.get("attention_config_hash", ""),
        "cache_layout_version": cache_layout_version,
        "block_group": block_group,
        "tenant_domain": tenant_domain,
        "multimodal_identity": multimodal_identity,
        "parent_chain_digest": parent_chain_digest or token_digest(tokens[:start]),
        "block_token_sequence_hash": token_digest(span),
    }
    return PrefixKey(fields=fields, tokens=tuple(span))


@dataclass(frozen=True)
class LookupResult:
    """What a lookup found, expressed in tokens — not in a boost factor."""

    query_tokens: int
    full_block_hits: int
    partial_tail_tokens: int
    cached_tokens: int
    computed_tokens: int
    first_miss_index: int
    resumed_position: int
    refcounts: Tuple[int, ...] = ()
    block_group_hits: Tuple[str, ...] = ()
    collision_detected: bool = False
    rejected_reason: str = ""

    def __post_init__(self) -> None:
        if self.cached_tokens + self.computed_tokens != self.query_tokens:
            raise ConfigError(
                "a lookup must account for every query token as either cached or "
                f"computed ({self.cached_tokens} + {self.computed_tokens} != "
                f"{self.query_tokens})",
                details={"field": "cached_tokens"},
            )
        if self.partial_tail_tokens > self.query_tokens - self.cached_tokens:
            raise ConfigError(
                "the partial tail cannot exceed the tokens that were not reused",
                details={"field": "partial_tail_tokens"},
            )

    @property
    def saved_model_tokens(self) -> int:
        """Tokens the model does not need to recompute (cached prefix only)."""
        return self.cached_tokens

    @property
    def hit_fraction(self) -> float:
        if self.query_tokens == 0:
            return 0.0
        return self.cached_tokens / self.query_tokens

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query_tokens": self.query_tokens,
            "full_block_hits": self.full_block_hits,
            "partial_tail_tokens": self.partial_tail_tokens,
            "cached_tokens": self.cached_tokens,
            "computed_tokens": self.computed_tokens,
            "first_miss_index": self.first_miss_index,
            "resumed_position": self.resumed_position,
            "saved_model_tokens": self.saved_model_tokens,
            "hit_fraction": self.hit_fraction,
            "refcounts": list(self.refcounts),
            "block_group_hits": list(self.block_group_hits),
            "collision_detected": self.collision_detected,
            "rejected_reason": self.rejected_reason,
        }


@dataclass
class CacheEntryRecord:
    """One cached block chain with owner, refcount and reuse statistics."""

    entry_id: str
    key: PrefixKey
    token_count: int
    block_ids: Tuple[int, ...]
    bytes_value: float
    refcount: int = 0
    reuse_count: int = 0
    last_use_iteration: int = 0
    readers: Tuple[str, ...] = ()
    pinned: bool = False

    @property
    def block_group(self) -> str:
        return self.key.fields["block_group"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "token_start": 0,
            "token_count": self.token_count,
            "block_ids": list(self.block_ids),
            "block_group": self.block_group,
            "key_digest": self.key.effective_digest,
            "refcount": self.refcount,
            "reuse_count": self.reuse_count,
            "last_use_iteration": self.last_use_iteration,
            "readers": list(self.readers),
            "pinned": self.pinned,
            "bytes_value": self.bytes_value,
        }


@dataclass(frozen=True)
class PrefixCacheSpec:
    """Frozen cache policy (``configs/runtime/prefix_spec.yaml``)."""

    block_size: int
    collision_policy: str = "digest_plus_token_equality"
    eviction_policy: str = "lru"
    max_cache_bytes: float = 0.0
    allow_cancelled_retain: bool = False
    verify_token_equality: bool = True
    cache_layout_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ConfigError("block_size must be positive")
        if self.collision_policy not in COLLISION_POLICIES:
            raise ConfigError(
                f"unknown collision policy {self.collision_policy!r}",
                details={"field": "collision_policy"},
            )
        if self.eviction_policy not in ("lru", "lru_reuse_aware"):
            raise ConfigError(
                f"unknown eviction policy {self.eviction_policy!r}",
                details={"field": "eviction_policy"},
            )
        if self.max_cache_bytes <= 0:
            raise ConfigError(
                "a prefix cache must declare a byte cap; an unbounded cache turns "
                "a capacity problem into an OOM",
                details={"field": "max_cache_bytes"},
            )
        if (
            self.collision_policy == "digest_plus_token_equality"
            and not self.verify_token_equality
        ):
            raise ConfigError(
                "collision_policy=digest_plus_token_equality requires "
                "verify_token_equality=true; otherwise the policy name promises a "
                "check the cache does not perform",
                details={"field": "verify_token_equality"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_size": self.block_size,
            "collision_policy": self.collision_policy,
            "eviction_policy": self.eviction_policy,
            "max_cache_bytes": self.max_cache_bytes,
            "allow_cancelled_retain": self.allow_cancelled_retain,
            "verify_token_equality": self.verify_token_equality,
            "cache_layout_version": self.cache_layout_version,
        }


@dataclass(frozen=True)
class _Match:
    """Internal per-entry match decision."""

    length: int
    accepted: bool
    collision: bool
    reason: str = ""


class PrefixCache:
    """A deterministic block-level prefix cache with refcounts and eviction."""

    #: Lookup/eviction audit trails, declared here so the experiment interface map
    #: can address them; ``__init__`` assigns fresh per-instance lists.
    lookups: List[Dict[str, Any]]
    evictions: List[Dict[str, Any]]
    entries: Dict[str, "CacheEntryRecord"]

    def __init__(self, spec: PrefixCacheSpec) -> None:
        self.spec = spec
        self.entries = {}
        self.lookups = []
        self.evictions = []
        self._clock = 0

    # ── insert / lookup ───────────────────────────────────────────────────

    def insert(
        self,
        *,
        key: PrefixKey,
        block_ids: Sequence[int],
        bytes_value: float,
        request_id: str,
    ) -> CacheEntryRecord:
        """Retain a *complete-block* prefix after the request finished legally."""
        if key.length == 0 or key.length % self.spec.block_size:
            raise ConfigError(
                "only complete blocks are cacheable; the tail is recomputed or "
                "handled by the runtime policy (E07-05 §2)",
                details={"field": "token_count"},
            )
        if len(block_ids) != key.length // self.spec.block_size:
            raise ConfigError(
                "block id count does not match the cached token length",
                details={"field": "block_ids"},
            )
        entry = CacheEntryRecord(
            entry_id=f"entry-{len(self.entries):04d}",
            key=key,
            token_count=key.length,
            block_ids=tuple(block_ids),
            bytes_value=float(bytes_value),
            refcount=1,
            readers=(request_id,),
        )
        self.entries[entry.entry_id] = entry
        self._clock += 1
        entry.last_use_iteration = self._clock
        return entry

    def _match_entry(
        self, entry: CacheEntryRecord, query: PrefixKey, query_tokens: int
    ) -> _Match:
        """Decide how much of ``entry`` the query can reuse (never a fraction).

        The two collision policies differ exactly here: ``strong_digest`` trusts a
        stored digest, while ``digest_plus_token_equality`` additionally compares
        the raw tokens and therefore detects a forged/colliding record.
        """
        if not entry.key.identity_matches(query):
            return _Match(0, False, False, "identity_mismatch")
        if entry.refcount <= 0:
            return _Match(0, False, False, "entry_not_shared")
        length = min(entry.token_count, query_tokens)
        length -= length % self.spec.block_size
        if length == 0:
            return _Match(0, False, False, "no_complete_block")
        overlap = tuple(query.tokens[:length])
        entry_overlap = tuple(entry.key.tokens[:length])
        tokens_ok = overlap == entry_overlap
        stored_digest = entry.key.digest
        if stored_digest:
            if stored_digest != query.effective_digest:
                return _Match(0, False, False, "digest_mismatch")
        elif token_digest(entry_overlap) != token_digest(overlap):
            return _Match(0, False, False, "digest_mismatch")
        if not tokens_ok:
            if self.spec.collision_policy == "digest_plus_token_equality":
                return _Match(0, False, True, "forced_collision")
            return _Match(length, True, False, "")
        return _Match(length, True, False, "")

    def lookup(
        self,
        *,
        key: PrefixKey,
        query_tokens: int,
        block_groups: Sequence[str],
        request_id: str,
    ) -> LookupResult:
        """Find the longest prefix reusable across **all** KV groups.

        ``key`` is the query's key (its ``tokens`` hold the query's prompt), so a
        candidate entry can be verified against the actual prompt tokens rather
        than against a stored digest alone.
        """
        if query_tokens <= 0:
            raise ConfigError("query_tokens must be positive")
        if query_tokens > key.length:
            raise ConfigError(
                "query_tokens exceeds the tokens carried by the query key; build "
                "the query key over the full prompt",
                details={"field": "query_tokens"},
            )
        per_group: Dict[str, int] = {}
        touching: List[CacheEntryRecord] = []
        collisions = 0
        rejected = ""
        for entry in list(self.entries.values()):
            if entry.block_group not in block_groups:
                continue
            decision = self._match_entry(entry, key, query_tokens)
            if decision.collision:
                collisions += 1
            if not decision.accepted:
                rejected = rejected or decision.reason
                continue
            touching.append(entry)
            per_group[entry.block_group] = max(
                per_group.get(entry.block_group, 0), decision.length
            )
        # Every requested KV group must be able to serve the prefix: a group with
        # no hit means the whole model cannot reuse that boundary (E07-05 §4).
        cached = min(
            (per_group.get(group, 0) for group in block_groups), default=0
        ) if block_groups else 0
        block_hits = cached // self.spec.block_size
        cached = block_hits * self.spec.block_size
        self._clock += 1
        for entry in touching:
            entry.reuse_count += 1
            entry.last_use_iteration = self._clock
            if request_id not in entry.readers:
                entry.readers = tuple(list(entry.readers) + [request_id])
                entry.refcount += 1
        result = LookupResult(
            query_tokens=query_tokens,
            full_block_hits=block_hits,
            partial_tail_tokens=query_tokens - cached,
            cached_tokens=cached,
            computed_tokens=query_tokens - cached,
            first_miss_index=cached,
            resumed_position=cached,
            refcounts=tuple(entry.refcount for entry in touching),
            block_group_hits=tuple(sorted(per_group)),
            collision_detected=bool(collisions),
            rejected_reason=rejected if not touching else "",
        )
        self.lookups.append(
            {"request_id": request_id, "key": key.as_dict(), **result.as_dict()}
        )
        return result

    # ── lifecycle ─────────────────────────────────────────────────────────

    def release(self, request_id: str, *, retain: bool = True) -> List[str]:
        """Release a request's readers; optionally retain entries as cache."""
        released: List[str] = []
        for entry in list(self.entries.values()):
            if request_id not in entry.readers:
                continue
            entry.readers = tuple(reader for reader in entry.readers if reader != request_id)
            entry.refcount = max(entry.refcount - 1, 0)
            released.append(entry.entry_id)
            if entry.refcount == 0 and not retain:
                self._drop(entry.entry_id, reason="release-no-retain")
        return released

    def _drop(self, entry_id: str, *, reason: str) -> None:
        entry = self.entries.get(entry_id)
        if entry is None:
            return
        if entry.refcount > 0:
            raise ConfigError(
                f"refusing to drop {entry_id}: refcount={entry.refcount} "
                "(an active or shared entry must never be evicted, E07-05 §7)",
                details={"entry_id": entry_id},
            )
        del self.entries[entry_id]
        self.evictions.append(
            {"entry_id": entry_id, "reason": reason, "bytes": entry.bytes_value}
        )

    def evict(self, *, target_bytes: float) -> List[Dict[str, Any]]:
        """Evict least-recently-used entries until the cache fits the cap."""
        evicted: List[Dict[str, Any]] = []
        limit = min(float(target_bytes), self.spec.max_cache_bytes)
        while self.total_bytes() > limit:
            candidates = [
                entry
                for entry in self.entries.values()
                if entry.refcount == 0 and not entry.pinned
            ]
            if not candidates:
                break
            if self.spec.eviction_policy == "lru":
                victim = min(
                    candidates, key=lambda item: (item.last_use_iteration, item.entry_id)
                )
            else:
                victim = min(
                    candidates,
                    key=lambda item: (
                        item.reuse_count,
                        item.last_use_iteration,
                        item.entry_id,
                    ),
                )
            evicted.append(
                {
                    "entry_id": victim.entry_id,
                    "reason": "capacity",
                    "reuse_count": victim.reuse_count,
                    "bytes": victim.bytes_value,
                }
            )
            self._drop(victim.entry_id, reason="capacity")
        return evicted

    def reset(self, *, reason: str) -> Dict[str, Any]:
        """Invalidate the whole cache (model/layout change, E07-05 step 18)."""
        dropped = len(self.entries)
        self.entries.clear()
        return {"dropped_entries": dropped, "reason": reason}

    def total_bytes(self) -> float:
        return sum(entry.bytes_value for entry in self.entries.values())

    def active_entries(self) -> int:
        return sum(1 for entry in self.entries.values() if entry.refcount > 0)

    def invariant_report(self) -> Dict[str, Any]:
        problems: List[str] = []
        for entry in self.entries.values():
            if entry.refcount != len(entry.readers):
                problems.append(
                    f"{entry.entry_id}: refcount {entry.refcount} != readers "
                    f"{len(entry.readers)}"
                )
            if entry.token_count % self.spec.block_size:
                problems.append(
                    f"{entry.entry_id}: token_count {entry.token_count} is not a "
                    "whole number of blocks"
                )
            if len(entry.block_ids) != entry.token_count // self.spec.block_size:
                problems.append(f"{entry.entry_id}: block count/length mismatch")
        return {
            "ok": not problems,
            "problems": problems,
            "entries": len(self.entries),
            "active": self.active_entries(),
            "bytes": self.total_bytes(),
            "within_cap": self.total_bytes() <= self.spec.max_cache_bytes,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "entries": [entry.as_dict() for entry in self.entries.values()],
            "lookups": list(self.lookups),
            "evictions": list(self.evictions),
            "invariants": self.invariant_report(),
        }


# ── correctness comparison and net saving ──────────────────────────────────


@dataclass(frozen=True)
class CacheCorrectnessComparison:
    """cache-off vs cache-on for the *same* request (E07-05 §6)."""

    request_id: str
    token_sequence_equal: bool
    next_token_logits_max_abs_diff: float
    kv_boundary_equal: bool
    resumed_position_equal: bool
    finish_reason_equal: bool
    long_generation_tokens: int
    quality_gate_passed: bool
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.quality_gate_passed:
            raise ConfigError(
                "a cache-on/off comparison that fails its quality gate cannot be "
                "used for a benefit claim (E07-05 §11 / §17)",
                details={"field": "quality_gate_passed"},
            )

    @property
    def semantically_equivalent(self) -> bool:
        return (
            self.token_sequence_equal
            and self.kv_boundary_equal
            and self.resumed_position_equal
            and self.finish_reason_equal
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_sequence_equal": self.token_sequence_equal,
            "next_token_logits_max_abs_diff": self.next_token_logits_max_abs_diff,
            "kv_boundary_equal": self.kv_boundary_equal,
            "resumed_position_equal": self.resumed_position_equal,
            "finish_reason_equal": self.finish_reason_equal,
            "long_generation_tokens": self.long_generation_tokens,
            "quality_gate_passed": self.quality_gate_passed,
            "semantically_equivalent": self.semantically_equivalent,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class NetSavingModel:
    """Saved tokens are not saved time (E07-05 §8)."""

    baseline_prefill_ms: float
    cached_request_prefill_ms: float
    lookup_hash_ms: float
    eviction_recompute_ms: float
    other_requests_impact_ms: float
    saved_tokens: int

    def net_saved_time_ms(self) -> float:
        return (
            self.baseline_prefill_ms
            - self.cached_request_prefill_ms
            - self.lookup_hash_ms
            - self.eviction_recompute_ms
            - self.other_requests_impact_ms
        )

    def naive_saving_ms(self, ms_per_token: float) -> float:
        """The linear extrapolation kept *only* to show the difference."""
        if ms_per_token <= 0:
            raise ConfigError("ms_per_token must be positive")
        return self.saved_tokens * ms_per_token

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline_prefill_ms": self.baseline_prefill_ms,
            "cached_request_prefill_ms": self.cached_request_prefill_ms,
            "lookup_hash_ms": self.lookup_hash_ms,
            "eviction_recompute_ms": self.eviction_recompute_ms,
            "other_requests_impact_ms": self.other_requests_impact_ms,
            "saved_tokens": self.saved_tokens,
            "net_saved_time_ms": self.net_saved_time_ms(),
        }


# ── negative fixtures (E07-05 §6 / steps 8–13) ─────────────────────────────


def _fixture_row(
    case: str, base: PrefixKey, other: PrefixKey, *, must_hit: bool, note: str = ""
) -> Dict[str, Any]:
    return {
        "case": case,
        "must_hit": must_hit,
        "identity_matches": base.identity_matches(other),
        "span_matches": base.span_matches(other),
        "content_matches": base.content_matches(other),
        "digest_differs": base.effective_digest != other.effective_digest,
        "note": note,
    }


def identity_negative_fixtures(
    identity: Mapping[str, str],
    *,
    tokens: Sequence[int] = tuple(range(64)),
    span: Tuple[int, int] = (0, 32),
) -> List[Dict[str, Any]]:
    """Cases that must never hit, plus the one legitimate partial hit.

    Every key field is mutated **on the built key** rather than on the identity
    mapping: several fields (cache layout, block group, tenant domain, parent
    chain, block token hash) are not read from the identity dict, so mutating the
    dict alone would silently produce a "fixture" that changes nothing.
    """
    base = build_prefix_key(identity=identity, tokens=tokens, token_span=span)
    fixtures: List[Dict[str, Any]] = []
    for field_name in ALL_KEY_FIELDS:
        mutated_fields = dict(base.fields)
        mutated_fields[field_name] = str(base.fields.get(field_name, "")) + "-changed"
        other = PrefixKey(fields=mutated_fields, tokens=base.tokens)
        fixtures.append(
            _fixture_row(
                f"different_{field_name}",
                base,
                other,
                must_hit=False,
                note="a mutated key field must not be served as a hit",
            )
        )
    content_changed = list(tokens)
    content_changed[span[0]] = int(content_changed[span[0]]) + 1
    other_content = build_prefix_key(
        identity=identity, tokens=content_changed, token_span=span
    )
    fixtures.append(
        _fixture_row(
            "same_span_different_token_content",
            base,
            other_content,
            must_hit=False,
            note="same identity, different tokens at the same span",
        )
    )
    shorter = build_prefix_key(
        identity=identity, tokens=tokens, token_span=(span[0], span[1] - 1)
    )
    fixtures.append(
        {
            "case": "prefix_shorter_by_one_token",
            "must_hit": True,
            "max_reusable_tokens": shorter.length,
            "must_not_claim": base.length,
            "note": "a shorter cached prefix is a legitimate partial hit; it must "
            "not be reported as the longer prefix",
        }
    )
    return fixtures


def forced_collision_case(
    identity: Mapping[str, str],
    *,
    tokens: Sequence[int] = tuple(range(64)),
    span: Tuple[int, int] = (0, 32),
) -> Dict[str, Any]:
    """A forced digest collision must be rejected under the frozen policy.

    The fixture rewrites only the *content digest* of a forged record, so the
    stored digest matches while the tokens do not — exactly the case a
    digest-only cache would serve wrongly.
    """
    base = build_prefix_key(identity=identity, tokens=tokens, token_span=span)
    forged_tokens = tuple([int(tokens[0]) + 7] + list(tokens[1 : span[1]]))
    forged = PrefixKey(
        fields=dict(base.fields), tokens=forged_tokens, digest=base.effective_digest
    )
    return {
        "case": "forced_collision",
        "digest_equal_by_construction": base.effective_digest == forged.effective_digest,
        "token_equality_holds": base.content_matches(forged),
        "identity_matches": base.identity_matches(forged),
        "span_matches": base.span_matches(forged),
        "frozen_policy": "digest_plus_token_equality",
        "expected": "REJECT",
        "strong_digest_would_serve_wrongly": True,
        "note": (
            "under digest_plus_token_equality the raw token comparison rejects the "
            "forged record; strong_digest alone cannot detect it"
        ),
    }


def collision_is_detected(
    spec: PrefixCacheSpec,
    identity: Mapping[str, str],
    *,
    tokens: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Run the forced-collision fixture through a real cache instance.

    The forged record carries the query's digest but different tokens, so a
    digest-only cache serves the wrong KV while ``digest_plus_token_equality``
    rejects it and counts the collision.
    """
    prompt = list(tokens if tokens is not None else range(64))
    base = build_prefix_key(identity=identity, tokens=prompt, token_span=(0, 32))
    forged_tokens = tuple([int(prompt[0]) + 7] + list(prompt[1:32]))
    forged = PrefixKey(
        fields=dict(base.fields), tokens=forged_tokens, digest=base.effective_digest
    )
    cache = PrefixCache(spec)
    cache.insert(
        key=forged,
        block_ids=tuple(range(32 // spec.block_size)),
        bytes_value=1024.0,
        request_id="r0",
    )
    result = cache.lookup(
        key=base,
        query_tokens=32,
        block_groups=("full_attention",),
        request_id="r1",
    )
    return {
        "policy": spec.collision_policy,
        "fixture": "forced_collision",
        "served_tokens": result.cached_tokens,
        "collision_detected": result.collision_detected,
        "rejected_reason": result.rejected_reason,
        "correct": result.cached_tokens == 0,
    }


__all__ = [
    "ALL_KEY_FIELDS",
    "CACHE_KEY_FIELDS",
    "COLLISION_POLICIES",
    "CacheCorrectnessComparison",
    "CacheEntryRecord",
    "IDENTITY_FIELDS",
    "LookupResult",
    "NetSavingModel",
    "PrefixCache",
    "PrefixCacheSpec",
    "PrefixKey",
    "SECURITY_DOMAINS",
    "SPAN_KEY_FIELDS",
    "build_prefix_key",
    "collision_is_detected",
    "forced_collision_case",
    "identity_negative_fixtures",
    "token_digest",
]
