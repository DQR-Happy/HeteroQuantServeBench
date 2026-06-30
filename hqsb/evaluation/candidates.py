"""Candidate identity, estimands and upstream evidence inventory (E12-01 §3, §9).

A *candidate* in S12 is not a chip name: it is the full System Under Test
(hardware SKU / device count / topology + host + firmware/driver/runtime +
model/tokenizer/precision + kernel/compiler/runtime/service policy).  Two
consequences drive this module:

* the identity is derived from canonical fields (``candidate_id``), never from a
  display name — renaming a column must not silently merge two candidates;
* every consumed upstream artefact (S02–S11) gets an explicit admissibility
  state (``VERIFIED``/``VERIFIED_WITH_LIMITS``/``UNVERIFIED``/``MISSING``/
  ``INCOMPATIBLE``); a missing artefact may not be replaced by a theoretical
  value.

Nothing here runs an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.layers import LAYERS, SCENARIOS, estimand_names
from hqsb.evaluation.records import (
    UPSTREAM_COMPARABLE_STATES,
    UPSTREAM_EVIDENCE_STATES,
    UPSTREAM_MISSING,
)

UNIVERSE_POLICY_VERSION = "s12_universe_1.0.0"

#: Canonical fields that define a comparison cell's candidate identity (§4).
CANDIDATE_IDENTITY_FIELDS: Tuple[str, ...] = (
    "hardware_sku",
    "device_count",
    "topology_id",
    "host_id",
    "software_stack_id",
    "backend_id",
    "compiler_artifact_id",
    "model_artifact_id",
    "tokenizer_id",
    "precision_contract_id",
    "quant_artifact_id",
    "kv_contract_id",
    "parallel_plan_id",
    "service_policy_id",
    "measurement_contract_id",
)

#: Fields that are *not* identity: they are display/report metadata.
NON_IDENTITY_FIELDS: Tuple[str, ...] = ("display_name", "owner", "notes", "region_label")


@dataclass
class CandidateIdentity:
    """One candidate configuration, addressable by a stable id."""

    identity: Mapping[str, Any]
    display_name: str = ""
    raw_manifest_hash: str = ""
    candidate_id: str = ""
    platform_id: str = ""
    policy_version: str = UNIVERSE_POLICY_VERSION
    limitations: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in (
            "hardware_sku",
            "device_count",
            "software_stack_id",
            "model_artifact_id",
            "tokenizer_id",
            "precision_contract_id",
        ):
            if self.identity.get(name) in (None, "", 0):
                problems.append(f"candidate identity is missing {name!r}")
        unknown = sorted(set(self.identity) - set(CANDIDATE_IDENTITY_FIELDS))
        if unknown:
            problems.append(f"unknown identity fields {unknown} (they are not join keys)")
        if not self.raw_manifest_hash:
            problems.append(
                "raw_manifest_hash is required: the canonical identity alone cannot prove "
                "which manifest produced it"
            )
        if self.display_name and self.display_name in self.identity:
            problems.append("a display name must not appear in the canonical identity")
        return problems

    def compute_id(self) -> str:
        """Deterministic ``candidate_id`` from the canonical fields only."""
        canonical = {name: self.identity.get(name, "") for name in CANDIDATE_IDENTITY_FIELDS}
        return stable_id("cand", canonical)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            try:
                self.candidate_id = self.compute_id()
            except ConfigError:
                self.candidate_id = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "platform_id": self.platform_id,
            "display_name": self.display_name,
            "raw_manifest_hash": self.raw_manifest_hash,
            "policy_version": self.policy_version,
            "limitations": list(self.limitations),
            **{name: self.identity.get(name, "") for name in CANDIDATE_IDENTITY_FIELDS},
        }


class CandidateMatrix:
    """The frozen candidate universe; duplicates and conflicts are rejected."""

    def __init__(self, *, policy_version: str = UNIVERSE_POLICY_VERSION) -> None:
        self.policy_version = policy_version
        self._rows: Dict[str, CandidateIdentity] = {}

    def add(self, candidate: CandidateIdentity) -> str:
        problems = candidate.validate()
        if problems:
            raise ConfigError("invalid candidate: " + "; ".join(problems))
        existing = self._rows.get(candidate.candidate_id)
        if existing is not None:
            if existing.as_dict() != candidate.as_dict():
                raise ConfigError(
                    f"candidate id collision for {candidate.candidate_id!r}: two different "
                    "configurations map to the same identity fields"
                )
            return candidate.candidate_id
        self._rows[candidate.candidate_id] = candidate
        return candidate.candidate_id

    def get(self, candidate_id: str) -> CandidateIdentity:
        try:
            return self._rows[candidate_id]
        except KeyError as exc:
            raise ConfigError(f"unknown candidate {candidate_id!r}") from exc

    def ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._rows))

    def by_platform(self, platform_id: str) -> Tuple[CandidateIdentity, ...]:
        return tuple(
            row for _, row in sorted(self._rows.items()) if row.platform_id == platform_id
        )

    def summary(self) -> Dict[str, Any]:
        platforms = sorted({row.platform_id for row in self._rows.values()})
        skus = sorted({str(row.identity.get("hardware_sku", "")) for row in self._rows.values()})
        return {
            "candidates": len(self._rows),
            "platforms": platforms,
            "hardware_skus": skus,
            "policy_version": self.policy_version,
        }

    def freeze(self) -> Dict[str, Any]:
        """Return the frozen universe (id list + content hash)."""
        payload = {
            "policy_version": self.policy_version,
            "candidates": [self._rows[key].as_dict() for key in sorted(self._rows)],
        }
        from hqsb.evaluation.identity import canonical_hash

        return {
            "policy_version": self.policy_version,
            "candidate_ids": list(sorted(self._rows)),
            "universe_hash": canonical_hash(payload),
            "payload": payload,
        }

    def as_rows(self) -> List[Dict[str, Any]]:
        return [self._rows[key].as_dict() for key in sorted(self._rows)]


def universe_invalidated(universe: Mapping[str, Any], new_candidate_ids: Sequence[str]) -> bool:
    """Adding a candidate must produce a new version of the matrix."""
    known = set(universe.get("candidate_ids", ()))
    return bool(set(new_candidate_ids) - known)


# ── estimands ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Estimand:
    """A quantitative question with an object, boundary, population and unit."""

    estimand_id: str
    layer: str
    scenario: str
    object_description: str
    boundary: str
    population: str
    estimator: str
    unit: str

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.layer not in LAYERS:
            problems.append(f"unknown layer {self.layer!r}")
        elif self.scenario not in SCENARIOS[self.layer]:
            problems.append(f"scenario {self.scenario!r} does not belong to {self.layer!r}")
        for name in ("object_description", "boundary", "population", "estimator", "unit"):
            if not getattr(self, name):
                problems.append(f"estimand {self.estimand_id} is missing {name!r}")
        if self.estimator == "总体性能":
            problems.append("'总体性能' is not an executable estimator")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "estimand_id": self.estimand_id,
            "layer": self.layer,
            "scenario": self.scenario,
            "object": self.object_description,
            "boundary": self.boundary,
            "population": self.population,
            "estimator": self.estimator,
            "unit": self.unit,
        }


_DEFAULT_ESTIMANDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("operator", "fixed_shape", "kernel latency under a frozen OperatorSpec", "device_event", "median_of_repetitions"),
    ("operator", "shape_sweep", "latency across the shape census", "device_event", "per_shape_median"),
    ("model_core", "prefill", "core TTFT for fixed token ids", "model_core_forward", "median_of_runs"),
    ("model_core", "decode", "core TPOT for fixed token ids", "model_core_forward", "median_of_runs"),
    ("model_core", "prefill_decode_e2e", "core E2E and accepted tokens", "model_core_forward", "median_of_runs"),
    ("service", "target_zone", "SLO-qualified goodput at the target rate", "service_client_visible", "run_level_goodput"),
    ("service", "overload", "error/reject behaviour under overload", "service_client_visible", "run_level_rate"),
    ("distributed", "strong_scaling", "strong scaling speedup at fixed global workload", "cluster_wall_clock", "ratio_of_medians"),
    ("distributed", "capacity_scaling", "capacity per device at constant latency", "cluster_wall_clock", "capacity_point"),
)


def estimand_registry(layer: Optional[str] = None) -> Tuple[Estimand, ...]:
    """Executable estimands per layer (no vague "overall performance")."""
    rows: List[Estimand] = []
    for row_layer, scenario, description, boundary, estimator in _DEFAULT_ESTIMANDS:
        if layer is not None and row_layer != layer:
            continue
        unit = _unit_for(row_layer, scenario)
        rows.append(
            Estimand(
                estimand_id=f"{row_layer}.{scenario}",
                layer=row_layer,
                scenario=scenario,
                object_description=description,
                boundary=boundary,
                population="frozen workload_spec + frozen candidate",
                estimator=estimator,
                unit=unit,
            )
        )
    return tuple(rows)


def _unit_for(layer: str, scenario: str) -> str:
    if layer == "distributed" and scenario in ("strong_scaling", "weak_scaling"):
        return "ratio"
    if layer == "service" and scenario in ("target_zone",):
        return "request/s"
    if layer == "service":
        return "1/s"
    if layer == "distributed":
        return "token/s"
    return "ns"


def validate_estimands(rows: Sequence[Estimand]) -> List[str]:
    problems: List[str] = []
    seen: set = set()
    for row in rows:
        problems.extend(row.validate())
        if row.estimand_id in seen:
            problems.append(f"duplicate estimand_id {row.estimand_id!r}")
        seen.add(row.estimand_id)
    covered = {row.layer for row in rows}
    for layer in LAYERS:
        if layer not in covered:
            problems.append(f"layer {layer!r} has no estimand")
        else:
            names = estimand_names(layer)
            if not names:
                problems.append(f"layer {layer!r} exposes no engine estimand")
    return problems


# ── capability requirements handed to E12-02 ──────────────────────────────


_BASE_FEATURES: Mapping[str, Tuple[str, ...]] = {
    "operator": ("dtype_storage", "dtype_elementwise", "dtype_reduction_accum", "dtype_gemm_native", "kernel_toolchain", "profiler_counters"),
    "model_core": ("model_load", "attention_kv_path", "graph_capture", "runtime_engine", "dtype_model_core"),
    "service": ("service_start", "service_streaming", "service_open_loop", "service_metrics", "service_cancellation"),
    "distributed": ("collective_all_reduce", "collective_all_gather", "topology_discovery", "parallel_plan_support", "p2p_access"),
}


def capability_requirements(
    candidate: CandidateIdentity,
    layer: str,
    *,
    precision: str = "bf16",
) -> Tuple[str, ...]:
    """Feature ids the candidate must verify before it may run this layer.

    The ids are the E12-02 registry's vocabulary; they are strings so that the two
    experiments stay decoupled (no cross-module import).
    """
    if layer not in LAYERS:
        raise ConfigError(f"unknown layer {layer!r}")
    features = [f"{precision}_{name}" for name in _BASE_FEATURES[layer]]
    if layer == "distributed":
        count = int(candidate.identity.get("device_count", 1) or 1)
        if count < 2:
            raise ConfigError(
                f"candidate {candidate.candidate_id} declares {count} device(s): the distributed "
                "layer is NOT_APPLICABLE_CAPABILITY for a single-device candidate"
            )
    if layer == "service" and not candidate.identity.get("service_policy_id"):
        raise ConfigError(
            f"candidate {candidate.candidate_id} has no service_policy_id: the service layer "
            "cannot be claimed"
        )
    return tuple(sorted(set(features)))


def requirements_by_candidate(
    candidates: Sequence[CandidateIdentity],
    *,
    layers: Sequence[str],
    precision: str = "bf16",
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """``{candidate_id: {layer: (feature_id, …)}}`` with explicit N/A entries."""
    out: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for candidate in candidates:
        per_layer: Dict[str, Tuple[str, ...]] = {}
        for layer in layers:
            try:
                per_layer[layer] = capability_requirements(candidate, layer, precision=precision)
            except ConfigError:
                per_layer[layer] = ()
        out[candidate.candidate_id] = per_layer
    return out


# ── upstream evidence inventory ───────────────────────────────────────────


@dataclass
class UpstreamEvidenceRow:
    """One consumed upstream artefact (five admissible states, §2)."""

    candidate_id: str
    upstream_stage: str
    experiment_id: str
    artifact_uri: str = ""
    state: str = UPSTREAM_MISSING
    reason: str = ""
    schema_ok: bool = False
    hash_ok: bool = False
    verified_at: str = ""
    limits: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.state not in UPSTREAM_EVIDENCE_STATES:
            problems.append(f"unknown upstream state {self.state!r}")
        if not self.upstream_stage:
            problems.append("upstream_stage is required")
        if self.state in UPSTREAM_COMPARABLE_STATES:
            if not self.artifact_uri or not self.hash_ok:
                problems.append("a usable artefact needs a URI and a verified hash")
            if self.state == "VERIFIED_WITH_LIMITS" and not self.limits:
                problems.append("VERIFIED_WITH_LIMITS must state its limits")
        elif not self.reason:
            problems.append(f"state {self.state} requires a reason (no silent gaps)")
        return problems

    @property
    def usable(self) -> bool:
        return self.state in UPSTREAM_COMPARABLE_STATES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "upstream_stage": self.upstream_stage,
            "experiment_id": self.experiment_id,
            "artifact_uri": self.artifact_uri,
            "state": self.state,
            "reason": self.reason,
            "schema_ok": self.schema_ok,
            "hash_ok": self.hash_ok,
            "verified_at": self.verified_at,
            "limits": list(self.limits),
        }


def inventory_for_candidates(
    candidate_ids: Sequence[str],
    evidence_rows: Sequence[UpstreamEvidenceRow],
    *,
    required_stages: Sequence[str] = ("S02", "S03", "S05", "S07", "S08"),
) -> Dict[str, Any]:
    """Per-candidate evidence inventory plus the unresolved dependency list."""
    rows: List[UpstreamEvidenceRow] = []
    problems: List[str] = []
    known_stages = set(required_stages)
    for row in evidence_rows:
        problems.extend(f"{row.candidate_id}/{row.experiment_id}: {item}" for item in row.validate())
        known_stages.add(row.upstream_stage)
        rows.append(row)
    unresolved: List[Dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for stage in sorted(known_stages):
            matches = [
                row
                for row in rows
                if row.candidate_id == candidate_id and row.upstream_stage == stage
            ]
            if not matches:
                unresolved.append(
                    {
                        "candidate_id": candidate_id,
                        "upstream_stage": stage,
                        "state": UPSTREAM_MISSING,
                        "reason": "no upstream artefact registered for this stage",
                    }
                )
                continue
            if not any(row.usable for row in matches):
                best = matches[0]
                unresolved.append(
                    {
                        "candidate_id": candidate_id,
                        "upstream_stage": stage,
                        "state": best.state,
                        "reason": best.reason,
                    }
                )
    return {
        "rows": [row.as_dict() for row in rows],
        "unresolved": unresolved,
        "unresolved_count": len(unresolved),
        "usable_count": sum(1 for row in rows if row.usable),
        "problems": problems,
        "ok": not problems,
    }


def require_usable(states: Mapping[str, str], stage: str, *, candidate_id: str = "") -> None:
    """Fail closed: an unusable upstream stage may not enter a comparison."""
    state = states.get(stage, UPSTREAM_MISSING)
    if state not in UPSTREAM_COMPARABLE_STATES:
        raise ConfigError(
            f"upstream stage {stage} is {state}"
            + (f" for candidate {candidate_id}" if candidate_id else "")
            + ": it may not enter a formal comparison (a theoretical value is not a substitute)",
            details={"stage": stage, "state": state, "candidate_id": candidate_id},
        )


def state_counts(rows: Iterable[UpstreamEvidenceRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.state] = counts.get(row.state, 0) + 1
    return dict(sorted(counts.items()))
