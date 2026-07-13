"""E15-06 — dashboard/figure → raw lineage and regeneration.

Protocol: ``docs/stage_experiments/details/S15/E15-06_plot_raw_lineage_regeneration.md``
(45 steps).

A figure is a *derived entity*: ``raw → normalized → aggregate → point → figure``.
The module implements the chain, the sampling discipline and the diff classes:

* :class:`FigureAuditContract` — frozen before any tracing (step 1), because a
  contract written after seeing which point was easy to trace is not an audit;
* :class:`FigureSpec` + :func:`point_id` — the figure's semantics and the stable
  point identity that survives renaming (steps 3–5);
* :data:`SAMPLING_STRATA` and :data:`MANDATORY_SAMPLE_KINDS` — the stratified
  sampling frame with the forced high-risk objects (steps 6–7), and
  :func:`draw_sample` which freezes the seed *and* the full-frame hash (step 8);
* :class:`LineageChain` — the six-layer trace with per-layer digests
  (steps 10–29), culminating in :func:`rebuild_chain` (step 30);
* :func:`compare_numeric_table` and :data:`REBUILD_CLASSES` — numeric diff first,
  pixels second (steps 32–33, §9.1), with D3–D5 failing and D1/D2 requiring the
  frozen policy to explain the difference;
* :func:`check_min_context`, :func:`check_missing_expression`,
  :func:`check_axis_transform`, :func:`recompute_pareto` — the "does the picture
  still say the truth" checks (steps 34–37);
* :func:`inject_lineage_negative_controls` — raw deletion, query tampering, unit
  swap, fallback mixture (steps 38–41);
* :func:`compute_audit_metrics`, :func:`impact_analysis`,
  :func:`independent_review_task`, :func:`check_access_friction`,
  :func:`freeze_audit` — steps 42–45.

Nothing here renders a figure, downloads a raw file or reads a dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest, is_digest

#: Rebuild difference classes (§9.1).  D3–D5 fail; D0–D2 may pass under policy.
REBUILD_CLASSES: Tuple[str, ...] = rec.REBUILD_CLASSES
FAILING_REBUILD_CLASSES: Tuple[str, ...] = ("D3", "D4", "D5")

#: Sampling strata of step 6.
SAMPLING_STRATA: Tuple[str, ...] = (
    "stage",
    "evidence_level",
    "layer_micro_model_service",
    "hardware",
    "figure_type",
    "positive_negative_result",
    "error_bars",
    "public_channel",
)

#: High-risk objects that must be in the sample regardless of the draw (step 7).
MANDATORY_SAMPLE_KINDS: Tuple[str, ...] = (
    "hero_primary_claim",
    "maximum_value",
    "minimum_value",
    "direction_reversal",
    "cross_hardware_comparison",
    "pareto_point",
    "failure_or_missing_marker",
    "table_cell",
    "negative_result",
)

#: The minimum context a published figure must carry (step 34).
FIGURE_CONTEXT_FIELDS: Tuple[str, ...] = (
    "model",
    "hardware",
    "workload_or_metric",
    "baseline",
    "n_or_error",
    "evidence_link",
)

#: Distinct markers for non-measurements (step 35).
MISSING_MARKERS: Tuple[str, ...] = ("missing", "oom", "quality_fail", "unsupported", "not_run", "excluded")

#: Axis/normalisation features to verify (step 36).
AXIS_FEATURES: Tuple[str, ...] = (
    "log_scale",
    "truncated_axis",
    "zero_baseline",
    "speedup_denominator",
    "per_device_vs_aggregate",
    "percentage_arithmetic",
)

#: Lineage access budget defaults (§9.2): beyond these, the point is "technically
#: reachable" but not usable.
DEFAULT_ACCESS_BUDGET: Mapping[str, float] = {
    "max_parse_seconds": 30.0,
    "max_clicks": 4.0,
    "max_download_mb": 50.0,
}


@dataclass
class FigureAuditContract:
    """The frozen audit contract of step 1."""

    candidate_id: str
    figure_inventory: Tuple[str, ...] = ()
    sampling_frame_id: str = ""
    sampling_seed: int = -1
    mandatory_samples: Tuple[str, ...] = MANDATORY_SAMPLE_KINDS
    rebuild_levels: Tuple[str, ...] = ("numeric", "semantic", "visual")
    numeric_tolerance: Mapping[str, float] = field(default_factory=dict)
    visual_diff_policy: str = ""
    access_budget: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_ACCESS_BUDGET))

    schema_version = f"{SCHEMA_PREFIX}.figure-audit-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.candidate_id:
            findings.append("FigureAuditContract: candidate_id is required")
        if not self.figure_inventory:
            findings.append("FigureAuditContract: the figure inventory must be frozen before tracing")
        if self.sampling_seed < 0:
            findings.append("FigureAuditContract: the sampling seed must be frozen before seeing lineage status")
        missing = [kind for kind in MANDATORY_SAMPLE_KINDS if kind not in self.mandatory_samples]
        if missing:
            findings.append(f"FigureAuditContract: mandatory sample kinds missing: {', '.join(missing)}")
        if not self.numeric_tolerance:
            findings.append("FigureAuditContract: numeric tolerances must be pre-registered, not chosen per figure")
        if not self.visual_diff_policy:
            findings.append("FigureAuditContract: the visual-diff policy (which metadata diffs are allowed) is required")
        return findings


@dataclass
class FigureSpec:
    """The figure's semantics, not its pixels (step 4)."""

    figure_id: str
    version: str
    title: str
    claim_ids: Tuple[str, ...] = ()
    query_id: str = ""
    x_field: str = ""
    y_field: str = ""
    group_field: str = ""
    facet_field: str = ""
    unit: str = ""
    scale: str = "linear"
    aggregation: str = ""
    interval: str = ""
    filters: Tuple[str, ...] = ()
    ordering: str = ""
    source_entities: Tuple[str, ...] = ()
    rendering_version: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("figure_id", "version", "title"):
            if not getattr(self, name):
                findings.append(f"FigureSpec: {name} is required")
        if not self.claim_ids:
            findings.append(f"FigureSpec({self.figure_id}): a figure must bind to claim ids")
        if not self.query_id:
            findings.append(f"FigureSpec({self.figure_id}): the query is part of the protocol and must be versioned")
        if not self.interval:
            findings.append(f"FigureSpec({self.figure_id}): the interval definition must be explicit")
        if self.scale not in ("linear", "log"):
            findings.append(f"FigureSpec({self.figure_id}): unknown scale {self.scale!r}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "version": self.version,
            "title": self.title,
            "claim_ids": list(self.claim_ids),
            "query_id": self.query_id,
            "x_field": self.x_field,
            "y_field": self.y_field,
            "group_field": self.group_field,
            "facet_field": self.facet_field,
            "unit": self.unit,
            "scale": self.scale,
            "aggregation": self.aggregation,
            "interval": self.interval,
            "filters": list(self.filters),
            "ordering": self.ordering,
            "source_entities": list(self.source_entities),
            "rendering_version": self.rendering_version,
        }


def point_id(*, figure_id: str, figure_version: str, series: str, coordinates: Mapping[str, Any], result_ids: Sequence[str]) -> str:
    """Stable point identity: figure version + series + coordinates + result ids (step 5).

    Coordinates and result ids are part of the identity so a point can be located
    without guessing pixel positions, and the *version* is part of it so a repaired
    figure does not silently inherit the old point's lineage.
    """
    if not figure_id or not figure_version or not series:
        raise ConfigError("point_id: figure_id, figure_version and series are required")
    payload = {
        "figure_id": figure_id,
        "figure_version": figure_version,
        "series": series,
        "coordinates": {key: coordinates[key] for key in sorted(coordinates)},
        "result_ids": sorted(result_ids),
    }
    return "PT-" + canonical_digest(payload)[len("sha256:") :][:16]


# ── sampling (steps 6–8) ────────────────────────────────────────────────────


@dataclass
class SamplingFrameRow:
    """One candidate point in the sampling frame."""

    point_id: str
    figure_id: str
    strata: Mapping[str, str] = field(default_factory=dict)
    kinds: Tuple[str, ...] = ()
    lineage_state: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "figure_id": self.figure_id,
            "strata": {key: self.strata[key] for key in sorted(self.strata)},
            "kinds": list(self.kinds),
            "lineage_state": self.lineage_state,
        }


def draw_sample(
    frame: Sequence[SamplingFrameRow], *, seed: int, per_cell: int = 1
) -> Dict[str, Any]:
    """Draw the stratified sample with the frozen seed (step 8).

    Deterministic by construction: the order is derived from a digest of
    ``(seed, stratum, point_id)``, so the draw can be repeated by anyone holding
    the frame and the seed — and the *full-frame hash* is recorded so a late
    substitution of an inconvenient point is visible.
    """
    if seed < 0:
        raise ConfigError("draw_sample: the seed must be frozen (non-negative) before sampling")
    if per_cell < 1:
        raise ConfigError("draw_sample: per_cell must be at least 1")
    cells: Dict[Tuple[Tuple[str, str], ...], List[SamplingFrameRow]] = {}
    for row in frame:
        key = tuple(sorted((name, str(value)) for name, value in row.strata.items()))
        cells.setdefault(key, []).append(row)
    selected: List[str] = []
    for key in sorted(cells):
        rows = sorted(cells[key], key=lambda item: canonical_digest({"seed": seed, "point": item.point_id}))
        selected.extend(row.point_id for row in rows[:per_cell])
    mandatory = [row.point_id for row in frame if row.kinds]
    return {
        "seed": seed,
        "frame_size": len(frame),
        "frame_sha256": canonical_digest([row.as_dict() for row in frame]),
        "selected": sorted(set(selected) | set(mandatory)),
        "mandatory": sorted(set(mandatory)),
        "selection_probability": per_cell / max(len(frame), 1),
        "note": "抽样前冻结 seed 与全集 hash；事后替换追溯失败点等于伪造审计",
    }


def missing_mandatory_kinds(frame: Sequence[SamplingFrameRow]) -> List[str]:
    """Which mandatory object classes the frame cannot even offer (step 7)."""
    present: set = set()
    for row in frame:
        present.update(row.kinds)
    return [kind for kind in MANDATORY_SAMPLE_KINDS if kind not in present]


# ── lineage chain (steps 10–31) ─────────────────────────────────────────────

#: The six layers of the trace (§3.1).
LINEAGE_LAYERS: Tuple[str, ...] = ("figure", "point", "aggregate", "normalized", "raw", "identity")


@dataclass
class LineageLayer:
    """One layer of the trace with its identity and digests."""

    layer: str
    refs: Tuple[str, ...] = ()
    digest: str = ""
    contract_version: str = ""
    notes: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.layer not in LINEAGE_LAYERS:
            findings.append(f"LineageLayer: unknown layer {self.layer!r}")
        if not self.refs:
            findings.append(f"LineageLayer({self.layer}): no entity references")
        if self.layer in ("aggregate", "normalized", "raw") and not self.digest:
            findings.append(f"LineageLayer({self.layer}): a derived layer must carry a digest")
        if self.digest and not is_digest(self.digest):
            findings.append(f"LineageLayer({self.layer}): digest must be sha256:<hex>")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "refs": list(self.refs),
            "digest": self.digest,
            "contract_version": self.contract_version,
            "notes": self.notes,
        }


class LineageChain:
    """The ``figure → point → aggregate → normalized → raw → identity`` trace."""

    def __init__(self, point: str) -> None:
        self.point = point
        self.layers: Dict[str, LineageLayer] = {}

    def add_layer(self, layer: str, *, refs: Sequence[str], digest: str = "", contract_version: str = "", notes: str = "") -> LineageLayer:
        if layer in self.layers:
            raise ConfigError(f"layer {layer!r} is already recorded; a trace is a chain, not a pile")
        item = LineageLayer(layer=layer, refs=tuple(refs), digest=digest, contract_version=contract_version, notes=notes)
        self.layers[layer] = item
        return item

    def problems(self) -> List[str]:
        findings: List[str] = []
        for layer in LINEAGE_LAYERS:
            if layer not in self.layers:
                findings.append(f"lineage chain is missing layer {layer!r}")
            else:
                findings.extend(self.layers[layer].problems())
        return findings

    @property
    def complete(self) -> bool:
        return not self.problems()

    @property
    def digest(self) -> str:
        return canonical_digest({layer: self.layers[layer].as_dict() for layer in sorted(self.layers)})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.lineage-trace.v1",
            "point_id": self.point,
            "layers": {layer: self.layers[layer].as_dict() for layer in sorted(self.layers)},
            "complete": self.complete,
            "chain_digest": self.digest,
        }


def check_filter_legality(query: Mapping[str, Any], *, protocol: Mapping[str, Any]) -> List[str]:
    """Every filter must be justified by the pre-registered protocol (step 13).

    The dangerous filters are the ones that *remove inconvenient data*: status,
    quality, fallback and outlier filters must match the protocol, and a filter
    the protocol does not mention is a silent cherry-pick.
    """
    findings: List[str] = []
    allowed_status = set(protocol.get("status_filter", []))
    for name in ("status", "quality", "fallback", "outlier", "warm_cold", "missing"):
        value = query.get(name)
        if value is None:
            continue
        if name == "status" and allowed_status and set(value if isinstance(value, (list, tuple)) else [value]) - allowed_status:
            findings.append(f"filter {name!r} removes statuses the protocol keeps: {value!r}")
        if name == "quality" and protocol.get("quality_filter") != value:
            findings.append(f"filter {name!r}={value!r} is not the pre-registered quality filter")
        if name == "fallback" and value in ("exclude", False) and not protocol.get("fallback_filter_documented"):
            findings.append("excluding fallback runs without a documented rule hides mixed actual paths")
        if name == "outlier" and not protocol.get("outlier_policy"):
            findings.append("an outlier filter without a pre-registered policy is arbitrary")
    return findings


def recompute_aggregate(
    rows: Sequence[Mapping[str, Any]], *, value_field: str, estimator: str, interval: str
) -> Dict[str, Any]:
    """Recompute a point, its interval and its n from member rows (step 15).

    The estimator is applied literally: "mean of already-averaged rows" and "mean
    of raw samples" are different numbers, and mixing them is the classic silent
    aggregation bug.
    """
    if not rows:
        raise ConfigError("recompute_aggregate: no member rows (an aggregate of nothing is not zero)")
    values = [float(row[value_field]) for row in rows]
    if estimator == "mean":
        point = sum(values) / len(values)
    elif estimator == "median":
        ordered = sorted(values)
        middle = len(ordered) // 2
        point = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    elif estimator == "sum":
        point = float(sum(values))
    elif estimator == "max":
        point = float(max(values))
    else:
        raise ConfigError(f"recompute_aggregate: unknown estimator {estimator!r}")
    if interval == "none":
        low = high = point
    elif interval in ("min_max", "range"):
        low, high = float(min(values)), float(max(values))
    elif interval == "sd":
        mean = point
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        spread = variance**0.5
        low, high = mean - spread, mean + spread
    elif interval == "se":
        mean = point
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        spread = (variance**0.5) / (len(values) ** 0.5)
        low, high = mean - spread, mean + spread
    else:
        raise ConfigError(f"recompute_aggregate: unknown interval {interval!r}")
    return {
        "point": point,
        "interval": (low, high),
        "n": len(values),
        "estimator": estimator,
        "interval_definition": interval,
        "members_sha256": canonical_digest(sorted(str(row.get("result_id", index)) for index, row in enumerate(rows))),
    }


def validate_experiment_unit(*, unit: str, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """The n must count real independent units (step 16)."""
    findings: List[str] = []
    if unit not in ("run", "request", "seed", "device_block", "process", "session"):
        findings.append(
            f"experiment unit {unit!r} is not an independent unit; kernel launches/tokens are observations, "
            "not samples (E15-06 §11)"
        )
    ids = [str(row.get("run_id", row.get("request_id", ""))) for row in rows]
    if ids and len(set(ids)) != len(ids):
        findings.append("the member set contains repeated run/request ids counted twice")
    return findings


def validate_interval_semantics(*, definition: str, confidence: Optional[float], paired: bool) -> List[str]:
    """The error bar must say what it is (step 17)."""
    findings: List[str] = []
    known = ("sd", "std", "se", "sem", "percentile", "bootstrap", "min_max", "range", "none")
    if not any(token in definition.lower() for token in known):
        findings.append(f"interval definition {definition!r} does not name a statistic; a legend saying 'error' is not enough")
    if ("bootstrap" in definition.lower() or "percentile" in definition.lower()) and confidence is None:
        findings.append("a bootstrap/percentile interval must state its confidence level")
    if paired and "paired" not in definition.lower() and "blocked" not in definition.lower():
        findings.append("a paired/blocked comparison must say so in the interval definition")
    return findings


def rebuild_chain(
    *,
    raw_inventory: Mapping[str, str],
    transform: Callable[[Mapping[str, str]], Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> List[str]:
    """Run ``raw → normalized → aggregate → point`` and compare with what was published (step 30)."""
    if not raw_inventory:
        return ["the chain was asked to rebuild from an empty raw inventory"]
    produced = dict(transform(raw_inventory) or {})
    findings: List[str] = []
    for key in ("point", "n", "interval"):
        if key not in produced:
            findings.append(f"rebuild output is missing {key!r}")
            continue
        left = produced.get(key)
        right = expected.get(key)
        if isinstance(left, tuple):
            left = list(left)
        if isinstance(right, tuple):
            right = list(right)
        if left != right:
            findings.append(f"rebuild mismatch on {key!r}: {left!r} != published {right!r}")
    return findings


def compare_numeric_table(
    published: Mapping[str, Any], rebuilt: Mapping[str, Any], *, tolerance: float
) -> Dict[str, Any]:
    """Structured numeric/semantic diff of one point (step 32)."""
    findings: List[str] = []
    published_value = published.get("value")
    rebuilt_value = rebuilt.get("value")
    numeric_diff: Optional[float] = None
    if isinstance(published_value, (int, float)) and isinstance(rebuilt_value, (int, float)):
        numeric_diff = abs(float(rebuilt_value) - float(published_value))
        if numeric_diff > tolerance:
            findings.append(f"numeric diff {numeric_diff} exceeds tolerance {tolerance}")
    elif published_value != rebuilt_value:
        findings.append(f"value changed type/content: {published_value!r} != {rebuilt_value!r}")
    for field_name in ("interval", "n", "labels", "units", "sort", "missing_markers"):
        left = published.get(field_name)
        right = rebuilt.get(field_name)
        if isinstance(left, (list, tuple)):
            left = list(left)
        if isinstance(right, (list, tuple)):
            right = list(right)
        if left != right:
            findings.append(f"{field_name} differs: {left!r} != {right!r}")
    return {"numeric_diff": numeric_diff, "findings": findings, "ok": not findings}


def classify_visual_diff(*, pixel_similarity: float, semantic_changes: Sequence[str], policy: str) -> str:
    """Map a visual diff onto D0–D5 (§9.1); metadata-only diffs may not hide data diffs."""
    if semantic_changes:
        lowered = " ".join(semantic_changes).lower()
        if "query" in lowered or "member" in lowered or "filter" in lowered:
            return "D4"
        if "raw" in lowered or "identity" in lowered or "actual_path" in lowered:
            return "D5"
        if "interval" in lowered or "label" in lowered or "axis" in lowered:
            return "D3"
        return "D3"
    if pixel_similarity >= 0.999:
        return "D0"
    if policy and pixel_similarity >= 0.98:
        return "D1"
    return "D1"


def check_min_context(spec: FigureSpec, *, context: Mapping[str, Any]) -> List[str]:
    """The figure keeps its scope even when someone screenshots it (step 34)."""
    findings: List[str] = []
    for field_name in FIGURE_CONTEXT_FIELDS:
        if not context.get(field_name):
            findings.append(f"figure {spec.figure_id}: missing context field {field_name!r}")
    return findings


def check_missing_expression(markers: Sequence[str]) -> List[str]:
    """Missing/unsupported states must be distinguished, not drawn as zero (step 35)."""
    findings: List[str] = []
    if not markers:
        return findings
    unknown = [marker for marker in markers if marker not in MISSING_MARKERS]
    if unknown:
        findings.append(f"unknown missing markers: {', '.join(unknown)}")
    if "missing" in markers and len(set(markers)) == 1:
        findings.append(
            "all non-measurements collapsed into one marker; OOM/quality-fail/unsupported/not-run "
            "need distinct states (E15-06 step 35)"
        )
    return findings


def check_axis_transform(*, features: Mapping[str, Any], policy: Mapping[str, Any]) -> List[str]:
    """Axis transforms, zero baselines and denominators must match the policy (step 36)."""
    findings: List[str] = []
    for name in AXIS_FEATURES:
        if name not in features:
            findings.append(f"axis feature {name!r} is not declared (unknown is not default)")
            continue
        if features[name] and not policy.get(name):
            findings.append(f"axis feature {name!r} is enabled but the frozen policy does not allow it")
    if features.get("truncated_axis") and not features.get("zero_baseline") and not policy.get("truncation_declared"):
        findings.append("a truncated axis must declare the truncation on the figure")
    if features.get("speedup_denominator") in (None, ""):
        findings.append("a speedup figure must name its denominator (baseline id and direction)")
    return findings


def recompute_pareto(
    points: Sequence[Mapping[str, Any]], *, objectives: Mapping[str, str], constraints: Sequence[str] = ()
) -> Dict[str, Any]:
    """Recompute the frontier and keep the dominated points (step 37)."""
    if not points:
        raise ConfigError("recompute_pareto: no points")
    for objective, direction in objectives.items():
        if direction not in ("min", "max"):
            raise ConfigError(f"objective {objective!r} direction must be min/max")
    frontier: List[str] = []
    dominated: List[str] = []
    for candidate in points:
        if any(not candidate.get(name, True) for name in constraints):
            dominated.append(str(candidate.get("point_id", "")))
            continue
        is_dominated = False
        for other in points:
            if other is candidate or any(not other.get(name, True) for name in constraints):
                continue
            better_or_equal = all(
                (other[objective] <= candidate[objective] if direction == "min" else other[objective] >= candidate[objective])
                for objective, direction in objectives.items()
            )
            strictly_better = any(
                (other[objective] < candidate[objective] if direction == "min" else other[objective] > candidate[objective])
                for objective, direction in objectives.items()
            )
            if better_or_equal and strictly_better:
                is_dominated = True
                break
        (dominated if is_dominated else frontier).append(str(candidate.get("point_id", "")))
    return {
        "frontier": sorted(frontier),
        "dominated": sorted(dominated),
        "objectives": {key: objectives[key] for key in sorted(objectives)},
        "constraints": list(constraints),
        "note": "被支配点必须保留；手选 Pareto 候选属于伪造",
    }


# ── negative controls / metrics (steps 38–45) ───────────────────────────────


def inject_lineage_negative_controls() -> Tuple[Mapping[str, str], ...]:
    """Injected lineage defects (steps 38–41)."""
    return (
        {"injection_id": "inj-raw-missing", "severity": "P0", "check": "LineageChain.problems"},
        {"injection_id": "inj-raw-tampered", "severity": "P0", "check": "identity.content_address_aggregate"},
        {"injection_id": "inj-query-tampered", "severity": "P0", "check": "check_filter_legality"},
        {"injection_id": "inj-unit-swap", "severity": "P0", "check": "compare_numeric_table"},
        {"injection_id": "inj-series-swap", "severity": "P0", "check": "compare_numeric_table"},
        {"injection_id": "inj-fallback-mixed", "severity": "P0", "check": "LineageChain.problems"},
        {"injection_id": "inj-missing-as-zero", "severity": "P1", "check": "check_missing_expression"},
        {"injection_id": "inj-truncated-axis", "severity": "P1", "check": "check_axis_transform"},
    )


def compute_audit_metrics(
    *,
    frame: Sequence[SamplingFrameRow],
    selected: Sequence[str],
    traces: Sequence[LineageChain],
    rebuild_results: Sequence[Mapping[str, Any]],
    negative_control_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Per-stratum audit metrics (step 42)."""
    complete = [chain for chain in traces if chain.complete]
    return {
        "frame_size": len(frame),
        "selected": len(selected),
        "traced": len(traces),
        "lineage_complete": len(complete),
        "lineage_completion": (len(complete) / len(traces)) if traces else 0.0,
        "rebuild_failures": [item.get("point_id") for item in rebuild_results if not item.get("ok", False)],
        "negative_control_metrics": dict(sorted(negative_control_metrics.items())),
        "note": "一个平均值不能掩盖 hero 图失败；指标必须按 strata 展开",
    }


def impact_analysis(*, changed_figure: str, dependents: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """Which figures and channels a fix affects (step 43)."""
    affected = sorted({name for name, items in dependents.items() if changed_figure in items})
    return {
        "changed_figure": changed_figure,
        "affected_figures": affected,
        "note": "修一张图必须分析所有受影响 figure 与渠道，不能只修被抽中的那张",
    }


def independent_review_task(*, point_id_value: str, chain: LineageChain) -> Dict[str, Any]:
    """The reviewer's task: rebuild and explain one point without the author (step 44)."""
    return {
        "task": "rebuild-and-explain-one-point",
        "point_id": point_id_value,
        "expected_layers": list(LINEAGE_LAYERS),
        "chain_complete": chain.complete,
        "author_interventions_allowed": 0,
        "note": "原作者熟悉 notebook 隐式状态，复核必须由非作者完成",
    }


def check_access_friction(
    *, measured: Mapping[str, float], budget: Mapping[str, float] = DEFAULT_ACCESS_BUDGET
) -> List[str]:
    """A point that takes too long to open is *accessible* but not *usable* (§9.2)."""
    findings: List[str] = []
    for name, limit in sorted(budget.items()):
        value = measured.get(name)
        if value is None:
            findings.append(f"ACCESS_FRICTION: {name} was not measured")
        elif value > limit:
            findings.append(f"ACCESS_FRICTION: {name}={value} exceeds the pre-registered budget {limit}")
    return findings


def freeze_audit(
    *, audit_id: str, candidate_id: str, contract: FigureAuditContract, record: rec.PointLineageRecord
) -> Dict[str, Any]:
    """Freeze the audit record (step 45)."""
    problems = list(contract.problems()) + list(record.validate())
    if problems:
        raise ConfigError("refusing to freeze the audit: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.figure-lineage-audit.v1",
        "audit_id": audit_id,
        "candidate_id": candidate_id,
        "sampling_seed": contract.sampling_seed,
        "record": record.as_dict(),
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 FigureAuditContract", ("figures.FigureAuditContract",)),
    (2, "盘点公开视觉对象", ("figures.FigureSpec", "docs_gate.PageInventory")),
    (3, "给每个对象分配稳定 Figure ID", ("figures.FigureSpec.figure_id",)),
    (4, "建立 FigureSpec", ("figures.FigureSpec",)),
    (5, "建立 point-level ID 规则", ("figures.point_id",)),
    (6, "定义抽样 strata", ("figures.SAMPLING_STRATA", "figures.SamplingFrameRow")),
    (7, "冻结强制样本", ("figures.MANDATORY_SAMPLE_KINDS", "figures.missing_mandatory_kinds")),
    (8, "执行随机抽样", ("figures.draw_sample",)),
    (9, "建立 clean analysis 环境", ("campaign.REQUIRED_ISOLATION", "quickstart.SessionIdentity")),
    (10, "验证 figure artifact identity", ("figures.FigureSpec.problems", "figures.LineageLayer")),
    (11, "从图点定位 point record", ("figures.point_id", "telemetry.EvidenceLookupTrial")),
    (12, "定位冻结 query", ("figures.FigureSpec.query_id",)),
    (13, "检查过滤合法性", ("figures.check_filter_legality",)),
    (14, "定位 aggregate entity", ("figures.LineageChain.add_layer",)),
    (15, "重算聚合值", ("figures.recompute_aggregate",)),
    (16, "验证实验单位", ("figures.validate_experiment_unit",)),
    (17, "验证误差线语义", ("figures.validate_interval_semantics",)),
    (18, "定位 normalized rows", ("figures.LineageChain.add_layer",)),
    (19, "验证 normalized Schema", ("figures.LineageLayer.problems",)),
    (20, "定位 normalization activity", ("figures.LineageChain.add_layer",)),
    (21, "重建 normalized rows", ("figures.rebuild_chain",)),
    (22, "定位 raw sample entities", ("figures.LineageLayer", "identity.EvidenceRef")),
    (23, "校验 raw digest 和大小", ("claims.verify_digests", "identity.content_address_aggregate")),
    (24, "验证 raw Schema 与完整性", ("figures.LineageLayer.problems",)),
    (25, "绑定 protocol/config", ("figures.FigureSpec", "claims.ClaimBindings")),
    (26, "绑定 model/hardware/environment", ("claims.ClaimBindings", "experiment.environment_fingerprint")),
    (27, "绑定 source/binary/actual path", ("hero_replay.ActualPathEvidence", "figures.LineageLayer")),
    (28, "绑定 correctness/quality gate", ("contracts.check_quality_before_performance",)),
    (29, "绑定 ClaimRecord", ("contracts.ClaimRecord", "figures.FigureSpec.claim_ids")),
    (30, "从 raw 全链重建抽中点", ("figures.rebuild_chain", "figures.LineageChain.digest")),
    (31, "重建完整抽中 figure", ("figures.FigureSpec.as_dict", "figures.classify_visual_diff")),
    (32, "比较数值表", ("figures.compare_numeric_table",)),
    (33, "执行 visual diff", ("figures.classify_visual_diff", "figures.REBUILD_CLASSES")),
    (34, "验证图中最小上下文", ("figures.check_min_context", "figures.FIGURE_CONTEXT_FIELDS")),
    (35, "验证 missing/unsupported 表达", ("figures.check_missing_expression", "figures.MISSING_MARKERS")),
    (36, "验证对数轴/截断轴/归一化", ("figures.check_axis_transform", "figures.AXIS_FEATURES")),
    (37, "验证 Pareto/frontier 计算", ("figures.recompute_pareto",)),
    (38, "注入 raw 缺失/篡改", ("figures.inject_lineage_negative_controls",)),
    (39, "注入 query/filter 篡改", ("figures.check_filter_legality", "figures.inject_lineage_negative_controls")),
    (40, "注入单位和 series 错位", ("figures.compare_numeric_table", "figures.inject_lineage_negative_controls")),
    (41, "注入 fallback 混入", ("figures.inject_lineage_negative_controls", "hero_replay.ActualPathEvidence")),
    (42, "计算审计指标", ("figures.compute_audit_metrics", "telemetry.DetectorMetrics")),
    (43, "修复并全量影响分析", ("figures.impact_analysis", "contracts.new_candidate")),
    (44, "独立复核抽样与重建", ("figures.independent_review_task", "figures.check_access_friction")),
    (45, "冻结 FigureLineageAuditRecord", ("figures.freeze_audit", "records.PointLineageRecord")),
)

TITLE = "Dashboard/图表→Raw 数据反向追溯与重生成"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-06 证明抽查与结构门下图表可追溯、可重建；不证明未抽数据的科学结论自动正确"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the figure-lineage interfaces (labelled smoke)."""
    problems: List[str] = []
    contract = FigureAuditContract(
        candidate_id="cand-1",
        figure_inventory=("FIG-hero",),
        sampling_seed=7,
        numeric_tolerance={"default": 0.01},
        visual_diff_policy="layout/metadata diffs allowed; data diffs are D3+",
    )
    problems.extend(contract.problems())
    frame = (
        SamplingFrameRow(point_id="PT-1", figure_id="FIG-hero", strata={"stage": "S03"}, kinds=("hero_primary_claim",)),
        SamplingFrameRow(point_id="PT-2", figure_id="FIG-hero", strata={"stage": "S05"}),
        SamplingFrameRow(point_id="PT-3", figure_id="FIG-hero", strata={"stage": "S05"}, kinds=("negative_result",)),
    )
    draw = draw_sample(frame, seed=7)
    if "PT-1" not in draw["selected"] or "PT-3" not in draw["selected"]:
        problems.append("mandatory samples were not forced into the draw")
    again = draw_sample(frame, seed=7)
    if again["selected"] != draw["selected"]:
        problems.append("the draw is not deterministic under a frozen seed")
    chain = LineageChain("PT-1")
    chain.add_layer("figure", refs=("FIG-hero:v1",))
    chain.add_layer("point", refs=("PT-1",))
    if not chain.problems():
        problems.append("an incomplete chain was reported complete")
    aggregate = recompute_aggregate(
        [{"value": 1.0, "result_id": "r1"}, {"value": 3.0, "result_id": "r2"}],
        value_field="value",
        estimator="mean",
        interval="min_max",
    )
    if aggregate["point"] != 2.0 or aggregate["n"] != 2:
        problems.append("aggregate recomputation is wrong")
    illegal = check_filter_legality({"fallback": "exclude"}, protocol={})
    if not illegal:
        problems.append("an undocumented fallback exclusion was not flagged")
    diff = compare_numeric_table({"value": 2.0, "n": 2}, {"value": 2.5, "n": 2}, tolerance=0.01)
    if diff["ok"]:
        problems.append("a numeric diff beyond tolerance was accepted")
    pareto = recompute_pareto([{"point_id": "a", "x": 1, "y": 1}, {"point_id": "b", "x": 2, "y": 2}], objectives={"x": "min", "y": "min"})
    if pareto["frontier"] != ["a"] or pareto["dominated"] != ["b"]:
        problems.append("Pareto recomputation is wrong")
    friction = check_access_friction(measured={"max_parse_seconds": 60.0, "max_clicks": 1.0, "max_download_mb": 1.0})
    if not friction:
        problems.append("an over-budget drill-down was not reported as ACCESS_FRICTION")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "strata": len(SAMPLING_STRATA),
        "mandatory_kinds": len(MANDATORY_SAMPLE_KINDS),
        "injections": len(inject_lineage_negative_controls()),
        "problems": problems,
    }
