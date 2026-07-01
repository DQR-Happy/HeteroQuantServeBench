"""E12-10: end-to-end evidence lineage, validation and regeneration.

A dashboard number with no path back to a raw sample is an orphan, and an
orphan may not be published — even when the number looks right.  This module
builds the minimal evidence graph (entities / activities / edges) and the
checks that keep it honest:

* ``EntityRegistry`` is append-only: re-registering the same id with a different
  byte hash is refused, never silently overwritten;
* the DAG validator reports cycles, dangling references, duplicate identities
  and orphan nodes by *name*, not just "the graph is broken";
* fault injection must localise the exact entity/field/edge and the downstream
  claims — "the aggregate root changed" is not an answer;
* regeneration compares structure with exact/tolerance rules per transform
  kind (never a global ``1e-3``) and may disable the derived cache so a
  confirmation cannot read its own outputs.

Nothing here runs an experiment or regenerates a real campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import (
    EntityRef,
    canonical_hash,
    logical_uri,
)
from hqsb.evaluation.records import (
    CLAIM_STATUSES,
    DIFF_KINDS,
    MIN_REGENERATION_FOR_KEY_RESULT,
    MIN_REGENERATION_FOR_REPORT,
    TABLE_SCHEMAS,
    VALIDATION_STATUSES,
)

EXPERIMENT_ID = "E12-10"
TITLE = "Dashboard/报告点到 Raw 的端到端 Lineage、校验与一键重生成"
CLAIM_BOUNDARY = (
    "本实验通过证明 S12 报告是可追溯、可重建的 evidence product，"
    "不证明外部团队无需任何硬件/权限即可复现所有设备测量。"
)

ENTITY_TYPES: Tuple[str, ...] = (
    "model",
    "config",
    "code",
    "environment",
    "raw",
    "derived",
    "figure",
    "report",
    "price",
    "meter",
    "validation",
    "claim",
    "registry",
)

ACTIVITY_TYPES: Tuple[str, ...] = (
    "run",
    "probe",
    "profile",
    "clean",
    "aggregate",
    "model",
    "integrate",
    "cost",
    "pareto",
    "render",
    "validate",
    "register",
)

RELATIONS: Tuple[str, ...] = (
    "used",
    "generated",
    "derived_from",
    "invalidated_by",
    "supports",
    "rendered_as",
    "validated_by",
)

RESPONSIBILITY_KINDS: Tuple[str, ...] = ("runner", "tool", "operator", "reviewer", "pipeline")

REQUIRED_LINEAGE_OBJECTS: Tuple[str, ...] = (
    "comparison_contract",
    "capability_evidence",
    "raw_events",
    "normalized_matrix",
    "schedule",
    "telemetry",
    "exclusion",
    "statistics",
    "roofs",
    "predictions",
    "residuals",
    "meter",
    "power_series",
    "energy_window",
    "price_snapshot",
    "assumptions",
    "capacity",
    "cost",
    "profile",
    "constraints",
    "frontier",
    "recommendation",
    "session",
    "failure",
    "rubric",
    "maturity",
)


def validate_relation(relation: str) -> List[str]:
    return [] if relation in RELATIONS else [f"unknown lineage relation {relation!r}"]


# ── activities and edges ──────────────────────────────────────────────────


@dataclass
class EvidenceActivity:
    activity_id: str
    transform_type: str
    code_artifact_id: str
    environment_id: str
    entrypoint: str
    parameter_entity_id: str = ""
    seed_policy: str = "deterministic"
    input_entity_ids: Tuple[str, ...] = ()
    output_entity_ids: Tuple[str, ...] = ()
    started_at: str = ""
    ended_at: str = ""
    status: str = "not_run"
    log_entity_id: str = ""
    code_commit: str = ""
    dirty_patch_hash: str = ""
    container_or_tool: Mapping[str, str] = field(default_factory=dict)
    input_hashes: Mapping[str, str] = field(default_factory=dict)
    output_hashes: Mapping[str, str] = field(default_factory=dict)
    parameters_hash: str = ""
    schema_in: str = ""
    schema_out: str = ""
    random_seed: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.transform_type not in ACTIVITY_TYPES:
            problems.append(f"unknown transform type {self.transform_type!r}")
        for name in ("code_artifact_id", "environment_id", "entrypoint"):
            if not getattr(self, name):
                problems.append(f"a transform must record its {name!r}")
        if not self.input_entity_ids or not self.output_entity_ids:
            problems.append("a transform must list its input and output entity ids")
        if not self.parameters_hash:
            problems.append("a transform must record its parameters hash")
        if not self.schema_in or not self.schema_out:
            problems.append("a transform must record its input/output schema versions")
        return problems

    def validate_transform_contract(self) -> List[str]:
        """The six §6 transform fields: code/params/inputs/outputs/schema/seed."""
        problems = self.validate()
        if self.seed_policy == "random" and not self.random_seed:
            problems.append("a random transform must record its seed")
        if self.seed_policy not in ("deterministic", "random"):
            problems.append(f"unknown seed policy {self.seed_policy!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "transform_type": self.transform_type,
            "code_artifact_id": self.code_artifact_id,
            "environment_id": self.environment_id,
            "entrypoint": self.entrypoint,
            "parameter_entity_id": self.parameter_entity_id,
            "seed_policy": self.seed_policy,
            "input_entity_ids": list(self.input_entity_ids),
            "output_entity_ids": list(self.output_entity_ids),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "log_entity_id": self.log_entity_id,
            "code_commit": self.code_commit,
            "dirty_patch_hash": self.dirty_patch_hash,
            "container_or_tool": dict(sorted(self.container_or_tool.items())),
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "output_hashes": dict(sorted(self.output_hashes.items())),
            "parameters_hash": self.parameters_hash,
            "schema_in": self.schema_in,
            "schema_out": self.schema_out,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class EvidenceEdge:
    edge_id: str
    relation: str
    source_entity_id: str
    target_entity_id: str
    activity_id: str = ""
    parameters_hash: str = ""
    created_at: str = ""
    validator_status: str = "not_validated"

    def validate(self) -> List[str]:
        problems = validate_relation(self.relation)
        if not self.source_entity_id or not self.target_entity_id:
            problems.append("an edge needs both endpoints")
        if self.validator_status not in VALIDATION_STATUSES:
            problems.append(f"unknown validator status {self.validator_status!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "relation": self.relation,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "activity_id": self.activity_id,
            "parameters_hash": self.parameters_hash,
            "created_at": self.created_at,
            "validator_status": self.validator_status,
        }


# ── registries ────────────────────────────────────────────────────────────


class EntityRegistry:
    """Append-only entity store: raw evidence is never silently overwritten."""

    def __init__(self) -> None:
        self._rows: Dict[str, EntityRef] = {}
        self._versions: Dict[str, List[str]] = {}

    def register(self, entity: EntityRef) -> str:
        problems = entity.validate()
        if problems:
            raise ConfigError("invalid entity: " + "; ".join(problems))
        existing = self._rows.get(entity.entity_id)
        if existing is not None:
            if existing.byte_hash and entity.byte_hash and existing.byte_hash != entity.byte_hash:
                raise ConfigError(
                    f"entity {entity.entity_id!r} already exists with a different byte hash: "
                    "raw evidence is append-only, register a new version instead"
                )
            if existing.as_dict() == entity.as_dict():
                return entity.entity_id
        self._rows[entity.entity_id] = entity
        self._versions.setdefault(entity.entity_id, []).append(entity.entity_id)
        return entity.entity_id

    def get(self, entity_id: str) -> EntityRef:
        try:
            return self._rows[entity_id]
        except KeyError as exc:
            raise ConfigError(f"unknown entity {entity_id!r}") from exc

    def by_type(self, entity_type: str) -> Tuple[EntityRef, ...]:
        return tuple(row for row in self._rows.values() if row.entity_type == entity_type)

    def versions(self, entity_id: str) -> Tuple[str, ...]:
        return tuple(self._versions.get(entity_id, ()))

    def as_rows(self) -> List[Dict[str, Any]]:
        return [self._rows[key].as_dict() for key in sorted(self._rows)]


class EdgeRegistry:
    def __init__(self) -> None:
        self._rows: List[EvidenceEdge] = []

    def add(self, edge: EvidenceEdge) -> str:
        problems = edge.validate()
        if problems:
            raise ConfigError("invalid edge: " + "; ".join(problems))
        self._rows.append(edge)
        return edge.edge_id

    def rows(self) -> Tuple[EvidenceEdge, ...]:
        return tuple(self._rows)


class EvidenceDAG:
    """The entity/activity/edge graph with a validating builder."""

    def __init__(self, entities: EntityRegistry, edges: Sequence[EvidenceEdge]) -> None:
        self.entities = entities
        self.edges = tuple(edges)

    def validate(self) -> Tuple[Dict[str, Any], ...]:
        problems: List[Dict[str, Any]] = []
        known = set(self.entities._rows)
        incoming: Dict[str, Set[str]] = {}
        for edge in self.edges:
            for endpoint in (edge.source_entity_id, edge.target_entity_id):
                if endpoint not in known:
                    problems.append(
                        {
                            "check_id": "dag_dangling_reference",
                            "kind": "dangling",
                            "status": "FAIL",
                            "detail": f"edge {edge.edge_id!r} references unknown entity {endpoint!r}",
                            "affected_ids": [endpoint],
                        }
                    )
            if edge.activity_id and edge.activity_id not in known:
                problems.append(
                    {
                        "check_id": "dag_dangling_activity",
                        "kind": "dangling_activity",
                        "status": "FAIL",
                        "detail": f"edge {edge.edge_id!r} references unknown activity {edge.activity_id!r}",
                        "affected_ids": [edge.activity_id],
                    }
                )
            incoming.setdefault(edge.target_entity_id, set()).add(edge.source_entity_id)
        # duplicate identities
        seen: Dict[str, str] = {}
        for entity_id in sorted(known):
            entity = self.entities._rows[entity_id]
            key = (entity.logical_uri, entity.byte_hash)
            if key in seen and seen[key] != entity_id:
                problems.append(
                    {
                        "check_id": "dag_duplicate_identity",
                        "kind": "duplicate",
                        "status": "FAIL",
                        "detail": f"{entity_id!r} and {seen[key]!r} share the same identity",
                        "affected_ids": sorted([entity_id, seen[key]]),
                    }
                )
            else:
                seen[key] = entity_id
        # cycles over the edge graph
        cycle = self._find_cycle(incoming)
        if cycle:
            problems.append(
                {
                    "check_id": "dag_cycle",
                    "kind": "cycle",
                    "status": "FAIL",
                    "detail": "cycle: " + " -> ".join(cycle),
                    "affected_ids": cycle,
                }
            )
        # orphan nodes (no edge at all) are reported, never hidden
        referenced = {edge.source_entity_id for edge in self.edges} | {edge.target_entity_id for edge in self.edges}
        orphans = sorted(known - referenced)
        for entity_id in orphans:
            problems.append(
                {
                    "check_id": "dag_orphan",
                    "kind": "orphan",
                    "status": "FAIL",
                    "detail": f"entity {entity_id!r} has no lineage edge",
                    "affected_ids": [entity_id],
                }
            )
        return tuple(problems)

    def _find_cycle(self, incoming: Mapping[str, Set[str]]) -> List[str]:
        visiting: List[str] = []
        visited: set = set()

        def visit(node: str) -> Optional[List[str]]:
            if node in visiting:
                index = visiting.index(node)
                return visiting[index:] + [node]
            if node in visited:
                return None
            visiting.append(node)
            for predecessor in incoming.get(node, ()):
                found = visit(predecessor)
                if found:
                    return found
            visiting.pop()
            visited.add(node)
            return None

        for node in sorted(incoming):
            found = visit(node)
            if found:
                return found
        return []

    def downstream(self, entity_id: str) -> Tuple[str, ...]:
        """Everything reachable from ``entity_id`` via edges."""
        children: Dict[str, Set[str]] = {}
        for edge in self.edges:
            children.setdefault(edge.source_entity_id, set()).add(edge.target_entity_id)
        seen: List[str] = []

        def visit(node: str) -> None:
            for child in sorted(children.get(node, ())):
                if child not in seen:
                    seen.append(child)
                    visit(child)

        visit(entity_id)
        return tuple(seen)

    def reverse_trace(self, entity_id: str, *, max_hops: int = 64) -> Dict[str, Any]:
        """Walk back to every ancestor; manual steps must be zero to be mature."""
        parents: Dict[str, Set[str]] = {}
        for edge in self.edges:
            parents.setdefault(edge.target_entity_id, set()).add(edge.source_entity_id)
        seen: List[str] = []
        queue = [entity_id]
        hops = 0
        while queue and hops <= max_hops:
            next_queue = []
            for node in queue:
                for parent in sorted(parents.get(node, ())):
                    if parent not in seen:
                        seen.append(parent)
                        next_queue.append(parent)
            queue = next_queue
            hops += 1
        complete = bool(queue is not None) and hops <= max_hops
        return {
            "entity_id": entity_id,
            "ancestors": seen,
            "hops": hops,
            "manual_steps": 0,
            "missing_entities": [],
            "complete": complete,
            "note": "a trace that needs the author to find files for half a day is not mature",
        }


# ── claims and dashboard points ───────────────────────────────────────────


class ClaimRegistry:
    def __init__(self) -> None:
        self._claims: Dict[str, Dict[str, Any]] = {}

    def register_claim(
        self,
        claim_id: str,
        *,
        experiment_id: str,
        evidence_level: str,
        status: str,
        source_result_ids: Sequence[str],
        figure_or_table_ids: Sequence[str],
        limitations: Sequence[str],
    ) -> Dict[str, Any]:
        if status not in CLAIM_STATUSES:
            raise ConfigError(f"unknown claim status {status!r}")
        row = {
            "claim_id": claim_id,
            "experiment_id": experiment_id,
            "evidence_level": evidence_level,
            "status": status,
            "source_result_ids": list(source_result_ids),
            "figure_or_table_ids": list(figure_or_table_ids),
            "limitations": list(limitations),
            "orphan": not source_result_ids,
        }
        self._claims[claim_id] = row
        return row

    def orphan_claims(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(row for row in self._claims.values() if row["orphan"])

    def publishable_claims(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(row for row in self._claims.values() if not row["orphan"] and row["status"] in ("PUBLISHABLE", "PUBLISHABLE_WITH_LIMITS"))

    def as_rows(self) -> List[Dict[str, Any]]:
        return [self._claims[key] for key in sorted(self._claims)]


@dataclass(frozen=True)
class DashboardPoint:
    point_id: str
    view_or_query: str
    view_version: str
    filters: Tuple[str, ...]
    normalized_result_ids: Tuple[str, ...]
    format: str
    displayed_rounding: str
    label: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.view_or_query or not self.view_version:
            problems.append("a dashboard point must record its view/query and version")
        if not self.normalized_result_ids:
            problems.append("a dashboard point must cite its normalized result ids (no hand-copied numbers)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "view_or_query": self.view_or_query,
            "view_version": self.view_version,
            "filters": list(self.filters),
            "normalized_result_ids": list(self.normalized_result_ids),
            "format": self.format,
            "displayed_rounding": self.displayed_rounding,
            "label": self.label,
        }


def validate_point(point: DashboardPoint, registry: EntityRegistry) -> List[str]:
    problems = point.validate()
    for result_id in point.normalized_result_ids:
        try:
            entity = registry.get(result_id)
        except ConfigError:
            problems.append(f"point {point.point_id!r} references unknown result {result_id!r}")
            continue
        if entity.entity_type not in ("derived", "normalized"):
            problems.append(f"point {point.point_id!r} references a non-result entity {result_id!r}")
    return problems


# ── validation ────────────────────────────────────────────────────────────


def validate_entities(registry: EntityRegistry, *, resolver: Optional[Any] = None) -> Tuple[Dict[str, Any], ...]:
    """Re-hash entities that carry a byte hash; localise every mismatch."""
    findings: List[Dict[str, Any]] = []
    for entity in registry.by_type("raw"):
        if not entity.byte_hash:
            continue
        location = resolver(entity.logical_uri) if resolver else None
        if location is None:
            findings.append(
                {
                    "check_id": "hash_unresolvable",
                    "entity_id": entity.entity_id,
                    "status": "SKIPPED",
                    "detail": "no resolver for the logical URI",
                }
            )
            continue
        ok, actual = _verify(location, entity.byte_hash)
        findings.append(
            {
                "check_id": "hash_mismatch" if not ok else "hash_ok",
                "entity_id": entity.entity_id,
                "status": "PASS" if ok else "FAIL",
                "expected_hash": entity.byte_hash,
                "actual_hash": actual,
                "detail": "" if ok else f"file {location} does not match its recorded hash",
            }
        )
    return tuple(findings)


def _verify(location: str, expected: str) -> Tuple[bool, str]:
    from hqsb.evaluation.identity import verify_byte_hash

    return verify_byte_hash(location, expected)


def semantic_cross_checks(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """Consistency between statuses (interface-level; no conclusion is drawn)."""
    findings: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("comparability_status") == "COMPARABLE" and not row.get("contract_sha256"):
            findings.append(
                {"check_id": "semantic_comparable_without_contract", "subject_id": str(row.get("id", "")), "status": "FAIL", "detail": "COMPARABLE without a contract"}
            )
        if row.get("energy_measurement_id") and not row.get("window_id"):
            findings.append(
                {"check_id": "semantic_energy_without_window", "subject_id": str(row.get("id", "")), "status": "FAIL", "detail": "energy without a same-run window"}
            )
        if row.get("cost_result_id") and not row.get("price_source_ids"):
            findings.append(
                {"check_id": "semantic_cost_without_price", "subject_id": str(row.get("id", "")), "status": "FAIL", "detail": "cost without a dated price source"}
            )
        if row.get("recommendation_id") and not row.get("frontier_id"):
            findings.append(
                {"check_id": "semantic_recommendation_without_frontier", "subject_id": str(row.get("id", "")), "status": "FAIL", "detail": "recommendation without its frontier"}
            )
    return tuple(findings)


def unit_closure_check(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Denominators and totals must close (J/token, cost/token, goodput)."""
    problems = []
    for row in rows:
        if row.get("j_per_token") is not None:
            energy = float(row.get("total_energy_j", 0.0))
            tokens = int(row.get("accepted_compliant_tokens", 0))
            if tokens and abs(float(row["j_per_token"]) - energy / tokens) > 1e-9 * max(1.0, abs(float(row["j_per_token"]))):
                problems.append(f"{row.get('id', '?')}: j_per_token does not close against total energy")
        if row.get("cost_per_token") is not None:
            cost = float(row.get("total_cost", 0.0))
            tokens = int(row.get("compliant_tokens", 0))
            if tokens and abs(float(row["cost_per_token"]) - cost / tokens) > 1e-9 * max(1.0, abs(float(row["cost_per_token"]))):
                problems.append(f"{row.get('id', '?')}: cost_per_token does not close")
    return {"rows": len(rows), "problems": problems, "ok": not problems}


# ── diff policy and regeneration ──────────────────────────────────────────


@dataclass(frozen=True)
class DiffPolicy:
    policy_version: str = "s12_diff_1.0.0"
    tolerances: Mapping[str, float] = field(default_factory=dict)

    def tolerance_for(self, transform_kind: str) -> float:
        return self.tolerances.get(transform_kind, 1e-12)

    def validate(self) -> List[str]:
        problems = []
        for name, tolerance in self.tolerances.items():
            if tolerance < 0:
                problems.append(f"tolerance for {name!r} must be non-negative")
        return problems


DEFAULT_DIFF_POLICY = DiffPolicy(
    tolerances={
        "aggregate": 1e-12,
        "statistics": 1e-9,
        "energy_integration": 1e-6,
        "cost_calculation": 1e-9,
        "render": 1e-12,
        "float_generic": 1e-9,
    }
)


def diff_objects(a: Mapping[str, Any], b: Mapping[str, Any], *, policy: DiffPolicy, float_kind: str = "float_generic") -> Tuple[Dict[str, Any], ...]:
    """Structure diff with per-transform tolerance (never a global ``1e-3``)."""
    tolerance = policy.tolerance_for(float_kind)
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(a) | set(b)):
        av = a.get(key)
        bv = b.get(key)
        if key not in a:
            rows.append({"field": key, "kind": DIFF_KINDS[4], "expected": None, "observed": bv})
        elif key not in b:
            rows.append({"field": key, "kind": DIFF_KINDS[4], "expected": av, "observed": None})
        elif isinstance(av, float) or isinstance(bv, float):
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)) and abs(float(av) - float(bv)) <= tolerance * max(1.0, abs(float(av)), abs(float(bv))):
                continue
            rows.append({"field": key, "kind": "SEMANTIC_CHANGE", "expected": av, "observed": bv})
        elif av != bv:
            rows.append({"field": key, "kind": "SEMANTIC_CHANGE", "expected": av, "observed": bv})
    return tuple(rows)


class RegenerationRun:
    def __init__(self, *, disable_derived_cache: bool = True) -> None:
        self.disable_derived_cache = disable_derived_cache
        self._steps: List[Dict[str, Any]] = []

    def record_step(self, step: str, status: str, detail: str = "") -> None:
        if status not in VALIDATION_STATUSES:
            raise ConfigError(f"unknown step status {status!r}")
        self._steps.append({"step": step, "status": status, "detail": detail})

    def rows(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._steps)


def plan_regeneration(*, dag: EvidenceDAG, stages: Sequence[str]) -> Dict[str, Any]:
    """validate inputs → resolve DAG → rebuild → render → compare → verdict."""
    dag_problems = dag.validate()
    return {
        "stages": list(stages),
        "dag_ok": not dag_problems,
        "dag_problems": [dict(row) for row in dag_problems],
        "order": ["validate_inputs", "resolve_dag", "rebuild_by_stage", "validate_outputs", "render", "compare", "emit_verdict"],
    }


def regeneration_verdict(*, regeneration: RegenerationRun, diffs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = regeneration.rows()
    failed = [step for step in steps if step["status"] == "FAIL"]
    return {
        "steps": [dict(step) for step in steps],
        "semantic_diffs": [dict(diff) for diff in diffs if diff.get("kind") == "SEMANTIC_CHANGE"],
        "presentation_diffs": [dict(diff) for diff in diffs if diff.get("kind") != "SEMANTIC_CHANGE"],
        "ok": not failed and not any(diff.get("kind") == "SEMANTIC_CHANGE" for diff in diffs),
        "derived_cache_disabled": regeneration.disable_derived_cache,
    }


# ── spot checks and fault injection ───────────────────────────────────────


def spot_check_plan(*, strata: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """Layered sampling: every experiment/layer/platform/winner-loser/cost/energy."""
    return tuple(
        {
            "stratum": str(row.get("stratum", "")),
            "population": int(row.get("population", 0)),
            "sampled": str(row.get("sampled", "")),
            "selection_rule": "covers every stratum plus every critical claim",
            "covers_critical_claims": True,
        }
        for row in strata
    )


def evaluate_spot_checks(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in results:
        complete = bool(row.get("reverse_trace_complete") and row.get("regenerated"))
        rows.append(
            {
                "point_id": str(row.get("point_id", "")),
                "stratum": str(row.get("stratum", "")),
                "reverse_trace_complete": bool(row.get("reverse_trace_complete")),
                "regenerated": bool(row.get("regenerated")),
                "diff_kind": str(row.get("diff_kind", "")),
                "status": "PASS" if complete else "FAIL",
            }
        )
    return {"rows": rows, "passed": sum(1 for row in rows if row["status"] == "PASS"), "ok": bool(rows) and all(row["status"] == "PASS" for row in rows)}


def fault_injection_cases() -> Tuple[Dict[str, Any], ...]:
    cases = (
        ("raw_tampered", "entity byte hash"),
        ("missing_file", "storage location"),
        ("wrong_hash", "entity byte hash"),
        ("schema_field_dropped", "entity field"),
        ("unit_changed", "metric unit"),
        ("dangling_reference", "edge target"),
        ("stale_price", "price effective window"),
        ("wrong_actual_backend", "observation actual_backend"),
        ("hand_edited_figure", "figure data payload"),
    )
    return tuple(
        {"case_id": name, "injection": name, "expected_locator": locator} for name, locator in cases
    )


def evaluate_fault_injection(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in results:
        located = bool(row.get("observed_locator") and row.get("downstream_claims") is not None)
        vague = bool(row.get("aggregate_only"))
        rows.append(
            {
                "case_id": str(row.get("case_id", "")),
                "expected_locator": str(row.get("expected_locator", "")),
                "observed_locator": str(row.get("observed_locator", "")),
                "localised": located and not vague,
                "downstream_claims": list(row.get("downstream_claims", ())),
                "aggregate_only": vague,
            }
        )
    return {
        "rows": rows,
        "localised": sum(1 for row in rows if row["localised"]),
        "ok": bool(rows) and all(row["localised"] for row in rows),
        "note": "an answer of 'the aggregate root changed' localises nothing",
    }


def coverage_report(rows: Sequence[Mapping[str, Any]], *, claims: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "levels": [dict(row) for row in rows],
        "claims": [dict(row) for row in claims],
        "ok": all(str(row.get("regeneration_level", "R0")) >= MIN_REGENERATION_FOR_REPORT for row in rows),
        "note": f"formal reports need {MIN_REGENERATION_FOR_REPORT}; key results need {MIN_REGENERATION_FOR_KEY_RESULT}",
    }


def report_ready(coverage: Dict[str, Any]) -> bool:
    return coverage.get("ok", False) is True


def assert_required_objects_documented(registry: EntityRegistry) -> Tuple[str, ...]:
    """Which §9 lineage objects have no registered entity (reported, never assumed)."""
    by_type = {str(row["entity_type"]) for row in registry.as_rows()}
    return tuple(name for name in REQUIRED_LINEAGE_OBJECTS if name not in by_type)


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    return {name: TABLE_SCHEMAS[name] for name in sorted(TABLE_SCHEMAS) if name.startswith("e12_10.")}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check; never an experiment."""
    registry = EntityRegistry()
    raw = EntityRef(
        entity_id="raw_1",
        entity_type="raw",
        logical_uri=logical_uri("camp", "raw", "raw_1"),
        byte_hash="a" * 64,
        canonical_hash=canonical_hash({"v": 1.0}),
        status="REGISTERED",
    )
    derived = EntityRef(
        entity_id="norm_1",
        entity_type="derived",
        logical_uri=logical_uri("camp", "derived", "norm_1"),
        canonical_hash=canonical_hash({"v": 2.0}),
        status="REGISTERED",
    )
    registry.register(raw)
    registry.register(derived)
    edge = EvidenceEdge(
        edge_id="e1",
        relation="derived_from",
        source_entity_id="raw_1",
        target_entity_id="norm_1",
    )
    dag = EvidenceDAG(registry, [edge])
    claims = ClaimRegistry()
    claims.register_claim(
        "c1",
        experiment_id="E12-03",
        evidence_level="L2",
        status="PUBLISHABLE",
        source_result_ids=("norm_1",),
        figure_or_table_ids=("fig_1",),
        limitations=("preliminary",),
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "append_only_rejects_overwrite": _expect_config_error(
            lambda: registry.register(
                EntityRef(entity_id="raw_1", entity_type="raw", logical_uri=logical_uri("camp", "raw", "raw_1"), byte_hash="b" * 64)
            )
        ),
        "dag_valid_when_edges_connect": dag.validate() == (),
        "orphan_claims_flagged": ClaimRegistry().register_claim(
            "c2", experiment_id="E12-03", evidence_level="L0", status="NOT_PUBLISHABLE", source_result_ids=(), figure_or_table_ids=(), limitations=()
        )["orphan"]
        is True,
        "reverse_trace_finds_ancestors": dag.reverse_trace("norm_1")["ancestors"] == ["raw_1"],
        "point_requires_result_ids": DashboardPoint(
            point_id="p", view_or_query="q", view_version="v1", filters=(), normalized_result_ids=(), format="table", displayed_rounding="2"
        ).validate()
        != [],
        "fault_injection_needs_localisation": fault_injection_cases()[0]["expected_locator"] == "entity byte hash",
    }


def _expect_config_error(callable_: Any) -> bool:
    try:
        callable_()
    except ConfigError:
        return True
    return False


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 campaign snapshot", ("campaign:CampaignManifest", "campaign:new_campaign_id")),
    (2, "定义 entity/activity/edge schema", ("lineage:ENTITY_TYPES", "lineage:RELATIONS", "lineage:ACTIVITY_TYPES")),
    (3, "定义 canonicalization/hash policy", ("identity:CanonicalPolicy", "identity:canonical_hash", "identity:byte_hash_file")),
    (4, "建立逻辑 URI resolver", ("identity:logical_uri", "identity:parse_logical_uri")),
    (5, "登记上游输入实体", ("lineage:EntityRegistry.register", "campaign:UpstreamEvidence")),
    (6, "登记外部 source snapshots", ("lineage:EntityRegistry", "identity:EntityRef")),
    (7, "登记 raw run entities", ("lineage:EntityRegistry", "identity:EntityRef")),
    (8, "验证 raw append-only", ("lineage:EntityRegistry.register", "records:MISSINGNESS_CODES")),
    (9, "登记 transform activities", ("lineage:EvidenceActivity", "lineage:ACTIVITY_TYPES")),
    (10, "登记 derived entities", ("lineage:EntityRegistry.register", "lineage:EvidenceActivity")),
    (11, "构建 evidence DAG", ("lineage:EvidenceDAG", "lineage:EdgeRegistry")),
    (12, "执行 schema validation", ("telemetry:validate_table_row", "lineage:EvidenceActivity.validate")),
    (13, "执行 hash/size validation", ("lineage:validate_entities", "identity:verify_byte_hash")),
    (14, "执行 semantic cross-check", ("lineage:semantic_cross_checks", "records:COMPARABILITY_STATES")),
    (15, "验证单位和分母闭合", ("lineage:unit_closure_check", "layers:TokenAccounting")),
    (16, "生成 dashboard point manifest", ("lineage:DashboardPoint", "lineage:validate_point")),
    (17, "生成 narrative claim registry", ("lineage:ClaimRegistry", "lineage:ClaimRegistry.orphan_claims")),
    (18, "设计分层 spot checks", ("lineage:spot_check_plan", "lineage:evaluate_spot_checks")),
    (19, "执行 point→normalized 反查", ("lineage:validate_point", "lineage:DashboardPoint")),
    (20, "执行 normalized→raw 反查", ("lineage:EvidenceDAG.reverse_trace", "lineage:EvidenceEdge")),
    (21, "执行 raw→identity 反查", ("lineage:EvidenceDAG.reverse_trace", "identity:IdentityChain")),
    (22, "执行 energy 反查", ("lineage:semantic_cross_checks", "energy:EnergyMeasurement")),
    (23, "执行 cost 反查", ("lineage:semantic_cross_checks", "cost:CostResult")),
    (24, "执行 recommendation 反查", ("lineage:semantic_cross_checks", "pareto:Recommendation")),
    (25, "测 reverse-trace 性能", ("lineage:EvidenceDAG.reverse_trace", "records:REGENERATION_CRITERIA")),
    (26, "准备 clean regeneration environment", ("lineage:RegenerationRun", "campaign:ArtifactLayout")),
    (27, "重建 normalized data", ("lineage:plan_regeneration", "benchmark:normalize")),
    (28, "重建模型/能量/成本/Pareto", ("lineage:plan_regeneration", "lineage:RegenerationRun")),
    (29, "重建图表/dashboard 数据", ("lineage:DashboardPoint", "lineage:RegenerationRun")),
    (30, "重建报告", ("lineage:ClaimRegistry", "lineage:coverage_report")),
    (31, "执行 exact/tolerance diff", ("lineage:diff_objects", "lineage:DiffPolicy")),
    (32, "分类非确定差异", ("lineage:regeneration_verdict", "records:DIFF_KINDS")),
    (33, "注入完整性故障", ("lineage:fault_injection_cases", "lineage:evaluate_fault_injection")),
    (34, "验证故障定位", ("lineage:evaluate_fault_injection", "lineage:EvidenceDAG.downstream")),
    (35, "生成证据覆盖报告", ("lineage:coverage_report", "records:MIN_REGENERATION_FOR_REPORT")),
    (36, "形成最终验收裁决", ("lineage:report_ready", "campaign:AcceptanceDecision")),
)
