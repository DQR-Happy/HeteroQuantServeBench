"""Four-state Ascend capability model (E09-01 §4.4, steps 1/19/20/24/28).

The protocol is explicit that a capability is **not a boolean**::

    SUPPORTED_VERIFIED     official evidence exists AND the local probe passed
    SUPPORTED_UNVERIFIED   official evidence exists but no probe has run yet
    UNSUPPORTED            official evidence or a probe says no
    UNKNOWN                evidence/probe insufficient — must NOT be treated as yes

Scope is a separate axis: E09-01 step 1 requires capabilities outside the
stage's declared goal to be marked ``NON_GOAL`` so the install matrix cannot
grow without bound.  ``NON_GOAL`` is therefore *not* a fifth state; it is
recorded in :attr:`CapabilityEntry.scope`.

Two rules drive the whole design:

* ``UNKNOWN`` never authorises execution (:func:`CapabilityTable.decide` returns
  ``allowed=False`` with a reason naming the missing probe);
* a decision always carries ``requested``/``actual``/``reason`` so a fallback
  can never be silent (control-plane §3.5 item 4, handbook §5.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import CapabilityError, ConfigError

SUPPORTED_VERIFIED = "SUPPORTED_VERIFIED"
SUPPORTED_UNVERIFIED = "SUPPORTED_UNVERIFIED"
UNSUPPORTED = "UNSUPPORTED"
UNKNOWN = "UNKNOWN"

#: The four states of E09-01 §4.4.  ``NON_GOAL`` is a scope, not a state.
CAPABILITY_STATES: Tuple[str, ...] = (
    SUPPORTED_VERIFIED,
    SUPPORTED_UNVERIFIED,
    UNSUPPORTED,
    UNKNOWN,
)

IN_SCOPE = "IN_SCOPE"
NON_GOAL = "NON_GOAL"

CAPABILITY_SCOPES: Tuple[str, ...] = (IN_SCOPE, NON_GOAL)

#: Capability domains E09-01 §4.4 requires at minimum.
CAPABILITY_DOMAINS: Tuple[str, ...] = (
    "dtype",
    "layout_format",
    "dynamic_shape",
    "custom_ascend_c",
    "rmsnorm",
    "quant_matmul",
    "non_default_stream",
    "workspace_api",
    "profiler_metric",
    "graph_mode",
    "fallback",
    "power_sampling",
)

#: Reason codes returned by :func:`CapabilityTable.decide`.  Stable strings so a
#: report can group refusals without parsing prose.
REASON_VERIFIED = "capability_verified"
REASON_UNVERIFIED_STRICT = "official_evidence_but_probe_not_run"
REASON_UNSUPPORTED = "explicitly_unsupported"
REASON_UNKNOWN = "insufficient_evidence_probe_required"
REASON_NOT_DECLARED = "capability_not_declared"
REASON_NON_GOAL = "out_of_declared_scope"
REASON_PROBE_CONTRADICTS_DOC = "official_support_but_local_probe_failed"

#: A probe that failed while the documentation claims support is *not* the same
#: thing as "unsupported": E09-01 §9 says to treat it as
#: ``SUPPORTED_UNVERIFIED``/FAIL and look at install/permission/target first.
CONTRADICTION_STATES: Tuple[str, ...] = (SUPPORTED_UNVERIFIED, UNKNOWN)

#: Probe status strings, duplicated from :mod:`hqsb.ascend.probes` on purpose:
#: ``capability`` is the lower-level module and must not import it.
PROBE_PASS = "pass"
PROBE_FAIL = "fail"
PROBE_UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class CapabilityEntry:
    """One declared capability with its evidence and probe identity."""

    key: str
    domain: str
    state: str
    scope: str = IN_SCOPE
    official_source: str = ""
    probe_id: str = ""
    probe_status: str = ""
    reason: str = ""
    evidence: Tuple[str, ...] = ()
    required_by: Tuple[str, ...] = ()
    #: Fields the entry is parameterised by (e.g. ``{"dtype": "fp16"}``).
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ConfigError("a capability entry needs a non-empty key")
        if self.state not in CAPABILITY_STATES:
            raise ConfigError(
                f"unknown capability state {self.state!r} for {self.key!r}; "
                f"E09-01 §4.4 allows {list(CAPABILITY_STATES)}",
                details={"key": self.key, "state": self.state},
            )
        if self.scope not in CAPABILITY_SCOPES:
            raise ConfigError(
                f"unknown scope {self.scope!r} for {self.key!r}; allowed "
                f"{list(CAPABILITY_SCOPES)}",
                details={"key": self.key, "scope": self.scope},
            )
        if self.domain and self.domain not in CAPABILITY_DOMAINS:
            raise ConfigError(
                f"unknown capability domain {self.domain!r} for {self.key!r}; "
                f"E09-01 §4.4 requires one of {list(CAPABILITY_DOMAINS)}",
                details={"key": self.key, "domain": self.domain},
            )
        if self.state == SUPPORTED_VERIFIED and not (self.probe_id and self.official_source):
            raise ConfigError(
                f"{self.key!r} claims SUPPORTED_VERIFIED without both an official source "
                "and a passing local probe; E09-01 §4.4 requires the intersection",
                details={"key": self.key, "probe_id": self.probe_id},
            )
        if self.state == UNKNOWN and self.scope == IN_SCOPE and not self.reason:
            raise ConfigError(
                f"{self.key!r} is UNKNOWN and in scope, so it must say which probe is "
                "missing; an unexplained UNKNOWN is indistinguishable from a guess",
                details={"key": self.key},
            )

    @property
    def usable(self) -> bool:
        """Only a verified, in-scope capability may authorise execution."""
        return self.state == SUPPORTED_VERIFIED and self.scope == IN_SCOPE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "domain": self.domain,
            "state": self.state,
            "scope": self.scope,
            "official_source": self.official_source,
            "probe_id": self.probe_id,
            "probe_status": self.probe_status,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "required_by": list(self.required_by),
            "attributes": dict(self.attributes),
            "usable": self.usable,
        }


@dataclass(frozen=True)
class CapabilityDecision:
    """requested / actual / reason — the anti-silent-fallback triple."""

    key: str
    requested: str
    actual: str
    allowed: bool
    reason: str
    state: str = UNKNOWN
    scope: str = IN_SCOPE
    evidence: Tuple[str, ...] = ()
    fallback_suggested: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "requested": self.requested,
            "actual": self.actual,
            "allowed": self.allowed,
            "reason": self.reason,
            "state": self.state,
            "scope": self.scope,
            "evidence": list(self.evidence),
            "fallback_suggested": self.fallback_suggested,
        }

    def require(self) -> "CapabilityDecision":
        """Raise :class:`CapabilityError` unless the capability is usable."""
        if not self.allowed:
            raise CapabilityError(
                f"capability {self.key!r} is not usable: {self.reason}",
                details=self.as_dict(),
            )
        return self


class CapabilityTable:
    """The declared capability set for one locked chip/CANN/framework tuple.

    ``strict`` controls whether ``SUPPORTED_UNVERIFIED`` may execute.  E09-05
    step 3 and E09-10 step 9 run in strict mode (an unverified capability must
    fail before allocation/launch); an environment survey may run relaxed to
    list what still needs a probe.
    """

    def __init__(self, entries: Iterable[CapabilityEntry], *, strict: bool = True) -> None:
        self.strict = strict
        self._entries: Dict[str, CapabilityEntry] = {}
        for entry in entries:
            if entry.key in self._entries:
                raise ConfigError(
                    f"duplicate capability key {entry.key!r}; a capability table with two "
                    "entries for one key cannot answer 'is this supported'",
                    details={"key": entry.key},
                )
            self._entries[entry.key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    @property
    def entries(self) -> Tuple[CapabilityEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def get(self, key: str) -> Optional[CapabilityEntry]:
        return self._entries.get(key)

    def by_domain(self, domain: str) -> Tuple[CapabilityEntry, ...]:
        return tuple(entry for entry in self.entries if entry.domain == domain)

    def decide(self, key: str, *, fallback: str = "") -> CapabilityDecision:
        """Return the requested/actual/reason triple for ``key``."""
        entry = self._entries.get(key)
        if entry is None:
            return CapabilityDecision(
                key=key,
                requested=key,
                actual="not_declared",
                allowed=False,
                reason=REASON_NOT_DECLARED,
                fallback_suggested=fallback,
            )
        if entry.scope == NON_GOAL:
            return CapabilityDecision(
                key=key,
                requested=key,
                actual="non_goal",
                allowed=False,
                reason=REASON_NON_GOAL,
                state=entry.state,
                scope=entry.scope,
                evidence=entry.evidence,
                fallback_suggested=fallback,
            )
        if entry.state == SUPPORTED_VERIFIED:
            return CapabilityDecision(
                key=key,
                requested=key,
                actual=key,
                allowed=True,
                reason=REASON_VERIFIED,
                state=entry.state,
                scope=entry.scope,
                evidence=entry.evidence,
            )
        if entry.state == SUPPORTED_UNVERIFIED:
            reason = (
                REASON_PROBE_CONTRADICTS_DOC
                if entry.probe_status and entry.probe_status != PROBE_PASS
                else REASON_UNVERIFIED_STRICT
            )
            return CapabilityDecision(
                key=key,
                requested=key,
                actual=fallback or "none",
                allowed=not self.strict,
                reason=reason,
                state=entry.state,
                scope=entry.scope,
                evidence=entry.evidence,
                fallback_suggested=fallback,
            )
        if entry.state == UNSUPPORTED:
            return CapabilityDecision(
                key=key,
                requested=key,
                actual=fallback or "none",
                allowed=False,
                reason=REASON_UNSUPPORTED,
                state=entry.state,
                scope=entry.scope,
                evidence=entry.evidence,
                fallback_suggested=fallback,
            )
        return CapabilityDecision(
            key=key,
            requested=key,
            actual=fallback or "none",
            allowed=False,
            reason=entry.reason or REASON_UNKNOWN,
            state=entry.state,
            scope=entry.scope,
            evidence=entry.evidence,
            fallback_suggested=fallback,
        )

    def require(self, key: str, *, fallback: str = "") -> CapabilityDecision:
        """Decide and raise unless usable — the fail-fast path before launch."""
        return self.decide(key, fallback=fallback).require()

    def missing_probes(self) -> List[str]:
        """In-scope capabilities that are not yet ``SUPPORTED_VERIFIED``."""
        return [
            entry.key
            for entry in self.entries
            if entry.scope == IN_SCOPE and entry.state != SUPPORTED_VERIFIED
        ]

    def coverage(self) -> Dict[str, Any]:
        """Per-domain state census — the E09-01 step 28 verdict input."""
        domains: Dict[str, Dict[str, int]] = {}
        for entry in self.entries:
            bucket = domains.setdefault(
                entry.domain or "undeclared", {state: 0 for state in CAPABILITY_STATES}
            )
            bucket[entry.state] += 1
        declared_domains = {entry.domain for entry in self.entries if entry.domain}
        return {
            "total": len(self._entries),
            "by_state": {
                state: sum(1 for entry in self.entries if entry.state == state)
                for state in CAPABILITY_STATES
            },
            "by_scope": {
                scope: sum(1 for entry in self.entries if entry.scope == scope)
                for scope in CAPABILITY_SCOPES
            },
            "by_domain": domains,
            "undocumented_domains": sorted(
                set(CAPABILITY_DOMAINS) - declared_domains
            ),
            "missing_probes": self.missing_probes(),
            "strict": self.strict,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strict": self.strict,
            "coverage": self.coverage(),
            "entries": [entry.as_dict() for entry in self.entries],
        }


def domain_coverage_gaps(table: CapabilityTable) -> List[str]:
    """Domains E09-01 §4.4 requires but the table does not declare at all."""
    return list(table.coverage()["undocumented_domains"])


def entries_from_documents(
    documents: Sequence[Mapping[str, Any]],
) -> List[CapabilityEntry]:
    """Build entries from a ``capabilities.json`` style document list."""
    entries: List[CapabilityEntry] = []
    for document in documents:
        entries.append(
            CapabilityEntry(
                key=str(document["key"]),
                domain=str(document.get("domain", "")),
                state=str(document.get("state", UNKNOWN)),
                scope=str(document.get("scope", IN_SCOPE)),
                official_source=str(document.get("official_source", "")),
                probe_id=str(document.get("probe_id", "")),
                probe_status=str(document.get("probe_status", "")),
                reason=str(document.get("reason", "")),
                evidence=tuple(document.get("evidence", ()) or ()),
                required_by=tuple(document.get("required_by", ()) or ()),
                attributes=dict(document.get("attributes", {}) or {}),
            )
        )
    return entries


def contradiction_audit(table: CapabilityTable) -> List[Dict[str, Any]]:
    """E09-01 §9: official support + failing probe must not read as supported."""
    rows: List[Dict[str, Any]] = []
    for entry in table.entries:
        if entry.state in CONTRADICTION_STATES and entry.probe_status and entry.probe_status != PROBE_PASS:
            rows.append(
                {
                    "key": entry.key,
                    "state": entry.state,
                    "probe_id": entry.probe_id,
                    "probe_status": entry.probe_status,
                    "official_source": entry.official_source,
                    "verdict": "treat_as_unverified_check_install_permission_target",
                }
            )
    return rows


def merge_probe_updates(
    declarations: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    official_by_key: Optional[Mapping[str, str]] = None,
    *,
    required_by: Optional[Mapping[str, Sequence[str]]] = None,
    scope_by_key: Optional[Mapping[str, str]] = None,
    domain_by_key: Optional[Mapping[str, str]] = None,
) -> Tuple[CapabilityEntry, ...]:
    """Intersect declared scope, official citations and probe results.

    This is E09-01 §3 as code: ``usable = official ∩ probe``.  A passing probe
    with no citation stays ``SUPPORTED_UNVERIFIED`` (the vendor may not support
    it next release); a citation with no probe stays ``SUPPORTED_UNVERIFIED``
    too (it has not been shown to work *here*).  Only both together produce
    ``SUPPORTED_VERIFIED`` — and a probe that *failed* while the documentation
    claims support is recorded as a contradiction, never as support.
    """
    citations = dict(official_by_key or {})
    requirements = {key: tuple(value) for key, value in (required_by or {}).items()}
    scopes = dict(scope_by_key or {})
    domains = dict(domain_by_key or {})

    merged: Dict[str, Dict[str, Any]] = {}
    for declaration in declarations:
        key = str(declaration["key"])
        merged[key] = {
            "key": key,
            "domain": str(declaration.get("domain", domains.get(key, ""))),
            "scope": str(declaration.get("scope", scopes.get(key, IN_SCOPE))),
            "official_source": str(declaration.get("official_source", citations.get(key, ""))),
            "probe_id": "",
            "probe_status": "",
            "state": UNKNOWN,
            "reason": str(declaration.get("reason", "")),
            "evidence": tuple(declaration.get("evidence", ()) or ()),
            "required_by": tuple(declaration.get("required_by", requirements.get(key, ()))),
            "attributes": dict(declaration.get("attributes", {}) or {}),
        }

    for update in updates:
        key = str(update["key"])
        row = merged.setdefault(
            key,
            {
                "key": key,
                "domain": domains.get(key, ""),
                "scope": scopes.get(key, IN_SCOPE),
                "official_source": citations.get(key, ""),
                "probe_id": "",
                "probe_status": "",
                "state": UNKNOWN,
                "reason": "",
                "evidence": (),
                "required_by": requirements.get(key, ()),
                "attributes": {},
            },
        )
        if not row["domain"]:
            row["domain"] = domains.get(key, "")
        if not row["official_source"]:
            row["official_source"] = citations.get(key, "")
        if not row["required_by"]:
            row["required_by"] = requirements.get(key, ())
        row["probe_id"] = str(update.get("probe_id", row["probe_id"]))
        row["probe_status"] = str(update.get("probe_status", row["probe_status"]))
        suggestion = str(update.get("state_suggestion", ""))
        if suggestion:
            row["_suggestion"] = suggestion

    entries: List[CapabilityEntry] = []
    for key in sorted(merged):
        row = dict(merged[key])
        suggestion = str(row.pop("_suggestion", ""))
        probe_status = row["probe_status"]
        official = row["official_source"]
        if probe_status == PROBE_PASS:
            state = SUPPORTED_VERIFIED if official else SUPPORTED_UNVERIFIED
            reason = (
                ""
                if official
                else "local probe passed but no official citation for this exact version/SKU"
            )
        elif probe_status == PROBE_FAIL:
            state = UNSUPPORTED
            reason = "local probe failed"
        elif probe_status == PROBE_UNAVAILABLE or not probe_status:
            state = suggestion if suggestion in (UNSUPPORTED,) else (UNKNOWN if not official else SUPPORTED_UNVERIFIED)
            reason = row["reason"] or f"probe unavailable or not run ({probe_status or 'not_run'})"
        else:
            state = UNKNOWN
            reason = row["reason"] or f"unrecognised probe status {probe_status!r}"
        if row["scope"] == NON_GOAL:
            # Out of scope keeps whatever evidence exists but can never authorise
            # execution, so it does not need the UNKNOWN explanation.
            reason = reason or "declared NON_GOAL for this stage"
        entries.append(
            CapabilityEntry(
                key=key,
                domain=row["domain"],
                state=state,
                scope=row["scope"],
                official_source=official,
                probe_id=row["probe_id"],
                probe_status=probe_status,
                reason=reason,
                evidence=tuple(row["evidence"]),
                required_by=tuple(row["required_by"]),
                attributes=dict(row["attributes"]),
            )
        )
    return tuple(entries)


__all__ = [
    "CAPABILITY_DOMAINS",
    "CAPABILITY_SCOPES",
    "CAPABILITY_STATES",
    "CONTRADICTION_STATES",
    "CapabilityDecision",
    "CapabilityEntry",
    "CapabilityTable",
    "IN_SCOPE",
    "NON_GOAL",
    "PROBE_FAIL",
    "PROBE_PASS",
    "PROBE_UNAVAILABLE",
    "REASON_NON_GOAL",
    "REASON_NOT_DECLARED",
    "REASON_PROBE_CONTRADICTS_DOC",
    "REASON_UNKNOWN",
    "REASON_UNSUPPORTED",
    "REASON_UNVERIFIED_STRICT",
    "REASON_VERIFIED",
    "SUPPORTED_UNVERIFIED",
    "SUPPORTED_VERIFIED",
    "UNKNOWN",
    "UNSUPPORTED",
    "contradiction_audit",
    "domain_coverage_gaps",
    "entries_from_documents",
    "merge_probe_updates",
]
