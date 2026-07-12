"""E15-01 — global claim ledger, declaration scan and the evidence gate.

Protocol: ``docs/stage_experiments/details/S15/E15-01_claim_ledger_evidence_gate.md``
(45 steps) and ``details/S15/README.md`` §8/§9/§10.

The module implements the *capability* the protocol describes, as seven groups:

1. **surface inventory** (:class:`PublicSurface`, :func:`enumerate_surfaces`) —
   every public channel (README, bilingual docs, reports, figures, release notes,
   demo captions, FAQ, resume candidates, upstream contribution texts), each
   with a text hash and no rewriting of the original (step 2, step 7);
2. **detectors** (:data:`DETECTABLE_PATTERNS`, :func:`extract_claim_candidates`,
   :func:`extract_numeric_expressions`, :func:`extract_implicit_comparisons`,
   :func:`extract_capability_claims`) — numbers, percentages, multiples,
   superlatives, support/compatibility, stability, scale, cross-hardware and
   "production-ready" phrasings in both languages, plus the non-claim exclusion
   rules (steps 5, 6, 8, 9, 10);
3. **canonicalisation** (:func:`canonicalise_candidates`,
   :func:`pair_bilingual`, :func:`assign_claim_id`) — one canonical claim per
   fact, renderings kept, stable ids from the fact signature (steps 11–13);
4. **binding** (:func:`bind_fact_fields`) — C1 model identity, C2/C3/C4/C5
   scope, hardware/environment, baseline, correctness/quality gate and the
   statistics object (steps 14–20);
5. **evidence graph and integrity** (:class:`EvidenceGraph`,
   :func:`verify_uris`, :func:`verify_digests`, :func:`verify_derivable`) —
   typed PROV-style edges, consumer-side URI resolution, per-file digests and
   regeneration checks (steps 21–25);
6. **audit** (:func:`detect_orphans`, :func:`detect_stale`,
   :func:`detect_conflicts`, :func:`check_units_and_magnitudes`,
   :func:`check_extrapolation_boundaries`, :func:`adjudicate_release_eligibility`,
   :func:`coverage_and_risk_report`) — the failure classifications of step 29–33,
   43–44 plus the severity ladder of §10;
7. **detection power and freeze** (:func:`inject_negative_controls`,
   :func:`run_detector_audit`, :func:`freeze_ledger`) — a scanner with no
   negative controls may not declare a clean result (steps 35, 36, 45).

Nothing here executes the audit or publishes a claim: the module returns
structures and verdicts, and ``smoke_self_check`` proves the negative paths are
refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.contracts import (
    FAILURE_SEVERITY,
    ClaimRecord,
    check_bilingual_fact_consistency,
)
from hqsb.release.identity import (
    EVIDENCE_LEVELS,
    PUBLIC_CHANNELS,
    SCHEMA_PREFIX,
    canonical_digest,
    digest_text,
    is_digest,
    level_index,
    stable_claim_id,
)

#: Parser version recorded with every extracted candidate (step 7).
PARSER_VERSION = "hqsb.s15.claims.extract.v1"

#: Public channels a surface may belong to (subset of PUBLIC_CHANNELS with the
#: document-level channels the scan reads from the repository).
SURFACE_CHANNELS: Tuple[str, ...] = (
    "readme",
    "readme_zh",
    "docs",
    "report",
    "figure_caption",
    "release_notes",
    "demo_caption",
    "faq",
    "resume_candidate",
    "upstream_text",
    "changelog",
)

#: Languages a surface can be written in.
SURFACE_LANGUAGES: Tuple[str, ...] = ("zh", "en", "mixed", "code")

#: Detectable language patterns (step 5).  Each entry is data, not logic, so the
#: frozen copy in ``configs/release/detector-patterns.yaml`` and this table can be
#: compared (the spec audit does that by key).
DETECTABLE_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "numeric": (
        r"\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s*[x×]\b",
        r"\b\d+(?:\.\d+)?\s*(?:ms|us|µs|ns|s)\b",
        r"\b\d+(?:\.\d+)?\s*(?:GB|MB|MiB|GiB|KB)\b",
        r"\b\d+(?:\.\d+)?\s*(?:tokens?/s|tok/s|GB/s|TFLOPS|TFLOPs|FLOPs)\b",
        r"\b\d+(?:\.\d+)?\s*(?:W|J/token|J)\b",
        r"提升\s*\d+",
        r"降低\s*\d+",
        r"提升\s*(?:了|至)\s*\d+(?:\.\d+)?\s*(?:%|倍)",
    ),
    "superlative": (
        r"最快",
        r"最优",
        r"最高",
        r"首(?:个|次)",
        r"fastest",
        r"best[- ]in[- ]class",
        r"state[- ]of[- ]the[- ]art",
        r"SOTA",
    ),
    "capability": (
        r"支持\s*(?:CUDA|Ascend|NPU|vLLM|SGLang|Triton|CUTLASS|TensorRT|ONNX)",
        r"兼容\s*\S+",
        r"production[- ]ready",
        r"生产级",
        r"tail[- ]safe",
        r"deterministic",
        r"确定性",
        r"可扩展",
        r"scalable",
        r"跨硬件",
        r"cross[- ]hardware",
        r"端到端",
        r"end[- ]to[- ]end",
    ),
    "comparison": (
        r"更快",
        r"更慢",
        r"优于",
        r"劣于",
        r"接近",
        r"不退化",
        r"无回退",
        r"持平",
        r"faster",
        r"slower",
        r"better than",
        r"worse than",
        r"no regression",
        r"on par with",
    ),
}

#: Non-claim exclusion rules (step 6).  A candidate matching one of these is
#: classified ``excluded`` with the reason, so the audit can *see* what it
#: dropped instead of quietly reducing the denominator.
NON_CLAIM_EXCLUSIONS: Mapping[str, Tuple[str, ...]] = {
    "code_block": (r"^```", r"^\s{4,}\S", r"^\s*\$\s"),
    "formula": (r"^[A-Za-z_]+\s*=\s*", r"\\frac\{", r"\$.*\$"),
    "citation": (r"^\s*>", r"https?://", r"\(\d{4}\)", r"et al\."),
    "plan": (r"计划", r"将在", r"TODO", r"roadmap", r"planned", r"will be"),
    "problem_statement": (r"问题", r"挑战", r"question:", r"hypothesis"),
    "issue_quote": (r"^\s*[-*]\s*\[ \]", r"^### "),
}

#: Channel → the public-channel constant used by the claim record.
SURFACE_TO_CHANNEL: Mapping[str, str] = {
    "readme": "readme_results",
    "readme_zh": "readme_results",
    "docs": "design_document",
    "report": "readme_results",
    "figure_caption": "readme_results",
    "release_notes": "roadmap",
    "demo_caption": "demo",
    "faq": "design_document",
    "resume_candidate": "resume_bullet",
    "upstream_text": "design_document",
    "changelog": "roadmap",
}

#: Stage suffixes that a claim's scope can name (step 33 extrapolation checks).
EXTRAPOLATION_KINDS: Tuple[str, ...] = (
    "single_shape_to_model",
    "single_model_to_family",
    "single_gpu_to_cross_hardware",
    "micro_to_service",
    "single_run_to_distribution",
)

#: Units the magnitude checker understands, with their canonical family.
UNIT_FAMILIES: Mapping[str, str] = {
    "ms": "time",
    "s": "time",
    "us": "time",
    "µs": "time",
    "ns": "time",
    "mb": "bytes",
    "mib": "bytes",
    "gb": "bytes",
    "gib": "bytes",
    "kb": "bytes",
    "tokens/s": "throughput",
    "tok/s": "throughput",
    "gb/s": "bandwidth",
    "w": "power",
    "j": "energy",
    "j/token": "energy_intensity",
    "%": "ratio",
    "x": "speedup",
}

#: Unit pairs that must never be compared directly (step 32).
UNIT_MISMATCH_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("time", "bytes"),
    ("bytes", "time"),
    ("ratio", "speedup"),
    ("speedup", "ratio"),
    ("power", "energy"),
    ("throughput", "bandwidth"),
)


@dataclass
class PublicSurface:
    """One public document/channel with its text frozen (step 2, step 7)."""

    surface_id: str
    path: str
    channel: str
    language: str
    text: str
    lines: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.channel not in SURFACE_CHANNELS:
            raise ConfigError(f"unknown surface channel {self.channel!r}; known: {', '.join(SURFACE_CHANNELS)}")
        if self.language not in SURFACE_LANGUAGES:
            raise ConfigError(f"unknown surface language {self.language!r}")
        if not self.lines:
            self.lines = tuple(self.text.splitlines())

    @property
    def text_sha256(self) -> str:
        return digest_text(self.text)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "path": self.path,
            "channel": self.channel,
            "language": self.language,
            "lines": len(self.lines),
            "text_sha256": self.text_sha256,
        }


@dataclass
class ClaimCandidate:
    """One extracted candidate statement, never a modified document (step 7)."""

    candidate_id: str
    surface_id: str
    line: int
    text: str
    language: str
    matched_patterns: Tuple[str, ...] = ()
    classification: str = "candidate"  # candidate | excluded
    exclusion_reason: str = ""
    numeric_expressions: Tuple[str, ...] = ()
    parser_version: str = PARSER_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "surface_id": self.surface_id,
            "line": self.line,
            "text_sha256": digest_text(self.text),
            "language": self.language,
            "matched_patterns": list(self.matched_patterns),
            "classification": self.classification,
            "exclusion_reason": self.exclusion_reason,
            "numeric_expressions": list(self.numeric_expressions),
            "parser_version": self.parser_version,
        }


def enumerate_surfaces(
    files: Mapping[str, str],
    *,
    channel_of: Callable[[str], str],
    language_of: Callable[[str], str],
) -> List[PublicSurface]:
    """Build the full public-surface inventory from ``{path: text}`` (step 2).

    The inventory covers *all* channels, not only ``README`` — the protocol notes
    that resume candidates and demo captions are the easiest places to overclaim.
    """
    surfaces: List[PublicSurface] = []
    for index, path in enumerate(sorted(files)):
        text = files[path]
        surfaces.append(
            PublicSurface(
                surface_id=f"SURF-{index + 1:04d}",
                path=path,
                channel=channel_of(path),
                language=language_of(path),
                text=text,
            )
        )
    return surfaces


def extract_claim_candidates(surface: PublicSurface) -> List[ClaimCandidate]:
    """Extract candidate statements and numeric expressions from one surface.

    Every candidate keeps file/line/context/hash; nothing in the surface is
    modified.  Lines matching a non-claim exclusion rule are returned as
    ``classification='excluded'`` with the reason, so the audit can report what
    it dropped (steps 6, 7, 8).
    """
    candidates: List[ClaimCandidate] = []
    for number, line in enumerate(surface.lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        matched: List[str] = []
        for name, patterns in DETECTABLE_PATTERNS.items():
            if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns):
                matched.append(name)
        exclusion = ""
        for reason, patterns in NON_CLAIM_EXCLUSIONS.items():
            if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns):
                exclusion = reason
                break
        if not matched and not exclusion:
            continue
        numeric = extract_numeric_expressions(stripped)
        candidates.append(
            ClaimCandidate(
                candidate_id=f"{surface.surface_id}-L{number:04d}",
                surface_id=surface.surface_id,
                line=number,
                text=stripped,
                language=surface.language,
                matched_patterns=tuple(matched),
                classification="excluded" if exclusion else "candidate",
                exclusion_reason=exclusion,
                numeric_expressions=tuple(numeric),
            )
        )
    return candidates


#: Numeric expression classes of step 8, with their unit family.
NUMERIC_CLASSES: Tuple[str, ...] = (
    "latency",
    "throughput",
    "memory",
    "power",
    "energy",
    "cost",
    "error",
    "quality",
    "confidence_interval",
    "percentage",
    "speedup",
)

_NUMERIC_UNIT_MAP: Mapping[str, str] = {
    "ms": "latency",
    "us": "latency",
    "µs": "latency",
    "ns": "latency",
    "s": "latency",
    "tokens/s": "throughput",
    "tok/s": "throughput",
    "gb/s": "throughput",
    "mib": "memory",
    "gib": "memory",
    "mb": "memory",
    "gb": "memory",
    "kb": "memory",
    "w": "power",
    "j/token": "energy",
    "j": "energy",
    "%": "percentage",
    "x": "speedup",
}


def extract_numeric_expressions(text: str) -> List[str]:
    """All numeric expressions with a unit, as ``value unit`` strings (step 8)."""
    found: List[str] = []
    pattern = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|us|µs|ns|s|tokens?/s|tok/s|GB/s|MiB|GiB|MB|GB|KB|W|J/token|J|x|×|%)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        unit = match.group("unit").lower().replace("×", "x")
        found.append(f"{match.group('value')} {unit}")
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:-|–|~)\s*\d+(?:\.\d+)?", text):
        found.append("interval")
    if re.search(r"\b(?:CI|置信区间|confidence interval)\b", text, re.IGNORECASE):
        found.append("confidence_interval")
    return found


def extract_implicit_comparisons(text: str) -> List[str]:
    """Comparisons without an explicit number (step 9)."""
    return [
        pattern
        for pattern in DETECTABLE_PATTERNS["comparison"]
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def extract_capability_claims(text: str) -> List[str]:
    """Absolute capability statements (step 10)."""
    return [
        pattern
        for pattern in DETECTABLE_PATTERNS["capability"]
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def classify_claim(text: str) -> str:
    """Assign the claim type of step 3 from the text (no NLP dependency)."""
    lowered = text.lower()
    if any(token in text for token in ("支持", "兼容")) or any(
        token in lowered for token in ("support", "compatible", "production-ready", "tail-safe")
    ):
        return "capability"
    if any(token in text for token in ("正确", "误差", "一致", "tolerance")) or any(
        token in lowered for token in ("correct", "error", "accuracy", "parity")
    ):
        return "correctness"
    if any(token in text for token in ("内存", "显存", "RSS")) or any(token in lowered for token in ("memory", "rss")):
        return "memory"
    if any(token in text for token in ("功耗", "能耗", "J/token", "成本")) or any(
        token in lowered for token in ("power", "energy", "cost", "tco")
    ):
        return "energy_or_cost"
    if any(token in text for token in ("稳定", "可靠", "SLO", "可用性")) or any(
        token in lowered for token in ("reliab", "slo", "availability")
    ):
        return "reliability"
    if any(token in text for token in ("跨硬件", "可移植", "多平台")) or any(
        token in lowered for token in ("portable", "cross-hardware")
    ):
        return "portability"
    if any(token in lowered for token in ("faster", "speedup", "tokens/s", "latency", "吞吐", "时延", "加速")):
        return "performance"
    if any(token in text for token in ("质量", "得分", "准确")) or any(token in lowered for token in ("quality", "score")):
        return "quality"
    return "capability"


@dataclass
class CanonicalClaim:
    """One canonical claim with all its renderings (step 11, step 12)."""

    fact: Mapping[str, Any]
    renderings: List[Mapping[str, Any]] = field(default_factory=list)

    @property
    def claim_id(self) -> str:
        return stable_claim_id(self.fact)

    @property
    def channels(self) -> Tuple[str, ...]:
        return tuple(sorted({str(item.get("channel", "")) for item in self.renderings}))

    def add_rendering(self, surface_id: str, channel: str, language: str, text: str) -> Dict[str, Any]:
        rendering = {
            "surface_id": surface_id,
            "channel": channel,
            "language": language,
            "text_sha256": digest_text(text),
        }
        self.renderings.append(rendering)
        return rendering

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "fact_signature": {key: self.fact.get(key) for key in sorted(self.fact)},
            "renderings": [dict(item) for item in self.renderings],
            "channels": list(self.channels),
        }


def canonicalise_candidates(
    candidates: Sequence[ClaimCandidate], *, fact_of: Callable[[ClaimCandidate], Mapping[str, Any]]
) -> List[CanonicalClaim]:
    """Merge candidates that state the same fact into one canonical claim (step 11).

    ``fact_of`` supplies the fact signature (scope/baseline/estimand/direction);
    the merge key is the stable claim id derived from it, so two channels that
    say the same thing collapse into one record while a differently-scoped
    sentence becomes a different claim.
    """
    merged: Dict[str, CanonicalClaim] = {}
    for candidate in candidates:
        if candidate.classification != "candidate":
            continue
        claim = CanonicalClaim(fact=fact_of(candidate))
        key = claim.claim_id
        if key not in merged:
            merged[key] = claim
        merged[key].add_rendering(
            surface_id=candidate.surface_id,
            channel=str(claim.fact.get("channel", "docs")),
            language=candidate.language,
            text=candidate.text,
        )
    return [merged[key] for key in sorted(merged)]


def pair_bilingual(
    zh: Mapping[str, Any], en: Mapping[str, Any], *, claim_id: str
) -> List[str]:
    """Compare the fact fields of a bilingual pair (step 12)."""
    if zh.get("claim_id") != claim_id or en.get("claim_id") != claim_id:
        raise ConfigError("pair_bilingual: both renderings must carry the canonical claim id")
    return check_bilingual_fact_consistency(zh, en)


def assign_claim_id(fact: Mapping[str, Any]) -> str:
    """Stable id from the fact signature (step 13)."""
    return stable_claim_id(fact)


# ── binding (steps 14–20) ────────────────────────────────────────────────────


@dataclass
class ClaimBindings:
    """The C1–C5 / hardware / baseline / quality / statistics bindings of a claim."""

    model_artifact_id: Optional[str] = None
    workload_spec_id: Optional[str] = None
    operator_spec_id: Optional[str] = None
    backend_id: Optional[str] = None
    quant_artifact_id: Optional[str] = None
    hardware_ids: Tuple[str, ...] = ()
    device_count: int = 0
    power_mode: str = ""
    driver_runtime: str = ""
    environment_fingerprint: str = ""
    baseline_id: Optional[str] = None
    baseline_config: str = ""
    intended_difference: str = ""
    correctness_gate: str = ""
    quality_gate: str = ""
    correctness_status: str = "not_run"
    statistics_plan: Mapping[str, Any] = field(default_factory=dict)

    #: Fields without which a claim cannot be qualified (step 44).
    REQUIRED_FOR_VERIFIED: Tuple[str, ...] = (
        "environment_fingerprint",
        "baseline_id",
        "intended_difference",
        "correctness_gate",
    )

    def problems(self, claim_type: str) -> List[str]:
        findings: List[str] = []
        if claim_type in ("performance", "correctness", "quality"):
            if not self.model_artifact_id and not self.operator_spec_id:
                findings.append("binding: a model/operator identity is required (step 15/16)")
        if claim_type == "performance":
            if not self.baseline_id:
                findings.append("binding: a performance claim needs a baseline identity (step 18)")
            if not self.intended_difference:
                findings.append("binding: the unique intended difference must be written down (step 18)")
            if self.correctness_status != "pass":
                findings.append("binding: correctness must pass before a performance claim (step 19)")
            if not self.statistics_plan.get("unit"):
                findings.append("binding: the statistics unit is required (step 20)")
            if int(self.statistics_plan.get("n", 0)) < 1:
                findings.append("binding: at least one independent repetition must be planned (step 20)")
        if not self.environment_fingerprint:
            findings.append("binding: the environment fingerprint is required (step 17)")
        if not self.hardware_ids:
            findings.append("binding: at least one hardware id is required (step 17)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "REQUIRED_FOR_VERIFIED"}
        payload["hardware_ids"] = list(self.hardware_ids)
        payload["statistics_plan"] = dict(sorted(self.statistics_plan.items()))
        return payload


def bind_fact_fields(claim: CanonicalClaim, bindings: ClaimBindings) -> List[str]:
    """Attach bindings to a canonical claim and report what is still missing."""
    claim.fact = dict(claim.fact)  # copy-on-write; the caller's dict is not mutated
    claim.fact.update(
        {
            "model_artifact_id": bindings.model_artifact_id,
            "workload_spec_id": bindings.workload_spec_id,
            "operator_spec_id": bindings.operator_spec_id,
            "backend_id": bindings.backend_id,
            "hardware_ids": list(bindings.hardware_ids),
        }
    )
    return bindings.problems(str(claim.fact.get("claim_type", "")))


# ── evidence graph (step 21–25) ──────────────────────────────────────────────


@dataclass
class EvidenceEdge:
    """One typed PROV-style edge (step 21)."""

    edge_type: str
    source: str
    target: str
    activity: str = ""

    def __post_init__(self) -> None:
        if self.edge_type not in rec.EVIDENCE_EDGE_TYPES:
            raise ConfigError(
                f"unknown evidence edge type {self.edge_type!r}; known: {', '.join(rec.EVIDENCE_EDGE_TYPES)}"
            )
        if not self.source or not self.target:
            raise ConfigError("an evidence edge needs both endpoints")

    def as_dict(self) -> Dict[str, Any]:
        return {"edge_type": self.edge_type, "source": self.source, "target": self.target, "activity": self.activity}


class EvidenceGraph:
    """Claim→report→normalized→raw→identity DAG (step 21, step 29)."""

    def __init__(self) -> None:
        self.edges: List[EvidenceEdge] = []
        self.entities: Dict[str, Mapping[str, Any]] = {}

    def add_entity(self, entity_id: str, *, kind: str = "manifest", uri: str = "", digest: str = "") -> None:
        if entity_id in self.entities:
            raise ConfigError(f"entity {entity_id!r} is already declared (no silent overwrite)")
        if digest and not is_digest(digest):
            raise ConfigError(f"entity {entity_id!r} digest must be sha256:<hex>")
        self.entities[entity_id] = {"kind": kind, "uri": uri, "digest": digest}

    def add_edge(self, edge_type: str, source: str, target: str, *, activity: str = "") -> EvidenceEdge:
        edge = EvidenceEdge(edge_type=edge_type, source=source, target=target, activity=activity)
        self.edges.append(edge)
        return edge

    def producers(self, target: str) -> List[EvidenceEdge]:
        return [edge for edge in self.edges if edge.target == target]

    def ancestors(self, entity_id: str) -> List[str]:
        seen: set = set()
        frontier = [entity_id]
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge.target == current and edge.source not in seen:
                    seen.add(edge.source)
                    frontier.append(edge.source)
        return sorted(seen)

    def reachable(self, source: str) -> List[str]:
        seen: set = set()
        frontier = [source]
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge.source == current and edge.target not in seen:
                    seen.add(edge.target)
                    frontier.append(edge.target)
        return sorted(seen)

    def validate(self) -> List[str]:
        problems: List[str] = []
        produced: Dict[str, int] = {}
        for edge in self.edges:
            produced[edge.target] = produced.get(edge.target, 0) + 1
        for target, count in sorted(produced.items()):
            if count > 1:
                problems.append(f"entity {target!r} has {count} producers (the derivation is not a function)")
        if self._has_cycle():
            problems.append("evidence graph contains a cycle")
        return problems

    def _has_cycle(self) -> bool:
        adjacency: Dict[str, List[str]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
        state: Dict[str, int] = {}

        def visit(node: str) -> bool:
            if state.get(node) == 1:
                return True
            if state.get(node) == 2:
                return False
            state[node] = 1
            for nxt in adjacency.get(node, ()):
                if visit(nxt):
                    return True
            state[node] = 2
            return False

        return any(visit(node) for node in list(adjacency))

    @property
    def root(self) -> str:
        """The graph root digest (step 45)."""
        return canonical_digest(
            {
                "entities": {key: dict(value) for key, value in sorted(self.entities.items())},
                "edges": [edge.as_dict() for edge in self.edges],
            }
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.evidence-graph.v1",
            "entities": {key: dict(value) for key, value in sorted(self.entities.items())},
            "edges": [edge.as_dict() for edge in self.edges],
            "root": self.root,
        }


@dataclass
class UriCheck:
    """One URI resolution result in a consumer environment (step 22)."""

    uri: str
    resolved: bool = False
    status: str = ""
    redirected_to: str = ""
    size_bytes: Optional[int] = None
    used_local_mount: bool = False
    reason: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.resolved:
            findings.append(f"URI {self.uri!r} did not resolve ({self.reason or 'no reason recorded'})")
        if self.used_local_mount:
            findings.append(
                f"URI {self.uri!r} resolved through the author's local mount; it is not consumer-resolvable "
                "(E15-01 step 22)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "resolved": self.resolved,
            "status": self.status,
            "redirected_to": self.redirected_to,
            "size_bytes": self.size_bytes,
            "used_local_mount": self.used_local_mount,
            "reason": self.reason,
        }


def verify_uris(refs: Sequence[Mapping[str, str]], *, resolver: Callable[[str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve every evidence URI and report per-URI problems (step 22).

    ``resolver`` performs the actual fetch/head in the consumer environment; the
    audit refuses a resolver result that silently used a local mount.
    """
    results: List[Dict[str, Any]] = []
    for ref in refs:
        uri = str(ref.get("uri", ""))
        if not uri:
            results.append(UriCheck(uri="", resolved=False, reason="missing URI").as_dict())
            continue
        outcome = dict(resolver(uri) or {})
        check = UriCheck(
            uri=uri,
            resolved=bool(outcome.get("resolved", False)),
            status=str(outcome.get("status", "")),
            redirected_to=str(outcome.get("redirected_to", "")),
            size_bytes=outcome.get("size_bytes"),
            used_local_mount=bool(outcome.get("used_local_mount", False)),
            reason=str(outcome.get("reason", "")),
        )
        results.append({**check.as_dict(), "findings": check.problems()})
    return results


def verify_digests(inventory: Mapping[str, str], *, rehash: Callable[[str], str]) -> List[str]:
    """Recompute per-file digests and the aggregate root (step 23)."""
    problems: List[str] = []
    if not inventory:
        problems.append("digest verification was given an empty inventory (an empty inventory is not complete)")
        return problems
    for path, digest in sorted(inventory.items()):
        if not is_digest(digest):
            problems.append(f"{path}: recorded digest is not sha256:<hex>")
            continue
        actual = rehash(path)
        if actual != digest:
            problems.append(f"{path}: digest mismatch (recorded {digest}, actual {actual})")
    return problems


def verify_derivable(
    *,
    raw_inventory: Mapping[str, str],
    declared_output: str,
    transform: Callable[[Mapping[str, str]], str],
) -> List[str]:
    """Run the declared transformation and compare its output digest (step 25).

    A file that exists but cannot be regenerated is "available but not
    reproducible" — the distinction ``E15-06`` §10 draws explicitly.
    """
    if not raw_inventory:
        return ["regeneration was asked to run from an empty raw inventory"]
    produced = transform(raw_inventory)
    if produced != declared_output:
        return [
            "derivation mismatch: the declared transformation produced "
            f"{produced} but {declared_output} was published (a manual copy would look exactly like this)"
        ]
    return []


# ── failure classification (steps 27–33) ─────────────────────────────────────

#: Orphan categories (step 29).
ORPHAN_KINDS: Tuple[str, ...] = ("no_ledger_entry", "no_evidence", "no_raw", "no_owner", "no_scope")


@dataclass
class LedgerEntryView:
    """The fields the audit needs from a ledger entry (a narrow view of ClaimRecord)."""

    claim_id: str
    status: str = "DRAFT"
    owner: str = ""
    scope_present: bool = False
    evidence_refs: Tuple[str, ...] = ()
    raw_refs: Tuple[str, ...] = ()
    fact_signature: Mapping[str, Any] = field(default_factory=dict)
    invalidation_keys: Mapping[str, str] = field(default_factory=dict)
    channels: Tuple[str, ...] = ()
    rendered_numbers: Mapping[str, str] = field(default_factory=dict)
    evidence_level: str = "PLANNED"
    direction: str = ""


def detect_orphans(entries: Sequence[LedgerEntryView], *, unledgered_surfaces: Sequence[str] = ()) -> Dict[str, List[str]]:
    """Classify orphans per category (step 29).

    A bare count is not actionable; the protocol wants each orphan class
    separately so the fix owner is obvious.
    """
    findings: Dict[str, List[str]] = {kind: [] for kind in ORPHAN_KINDS}
    findings["no_ledger_entry"] = sorted(unledgered_surfaces)
    for entry in entries:
        if not entry.evidence_refs:
            findings["no_evidence"].append(entry.claim_id)
        elif not entry.raw_refs:
            findings["no_raw"].append(entry.claim_id)
        if not entry.owner:
            findings["no_owner"].append(entry.claim_id)
        if not entry.scope_present:
            findings["no_scope"].append(entry.claim_id)
    return {kind: sorted(set(items)) for kind, items in findings.items()}


def detect_stale(
    entries: Sequence[LedgerEntryView], *, current_candidate: Mapping[str, str]
) -> List[str]:
    """Compare invalidation keys against the current candidate (step 30).

    A historical PASS is not inherited: if any key the claim declared as
    invalidating differs, the claim is stale rather than verified.
    """
    stale: List[str] = []
    for entry in entries:
        for key, declared in sorted(entry.invalidation_keys.items()):
            current = current_candidate.get(key)
            if current is not None and current != declared:
                stale.append(f"{entry.claim_id}:{key}")
    return sorted(stale)


def detect_conflicts(entries: Sequence[LedgerEntryView]) -> List[str]:
    """Same-fact conflicts across channels and near-duplicate ids (step 31)."""
    conflicts: List[str] = []
    seen_numeric: Dict[Tuple[str, str], str] = {}
    for entry in entries:
        for channel, value in sorted(entry.rendered_numbers.items()):
            key = (entry.claim_id, channel)
            previous = seen_numeric.get((entry.claim_id, "any"))
            if previous is not None and previous != value:
                conflicts.append(f"{entry.claim_id}: channel {channel!r} says {value!r} but another channel says {previous!r}")
            seen_numeric[(entry.claim_id, "any")] = value
            seen_numeric[key] = value
    seen_sigs: Dict[str, str] = {}
    for entry in entries:
        signature = canonical_digest(entry.fact_signature) if entry.fact_signature else ""
        if not signature:
            continue
        if signature in seen_sigs and seen_sigs[signature] != entry.claim_id:
            conflicts.append(
                f"{entry.claim_id} and {seen_sigs[signature]} carry the same fact signature under two ids "
                "(duplicate ledger entries)"
            )
        seen_sigs[signature] = entry.claim_id
    return sorted(set(conflicts))


def unit_family(unit: str) -> str:
    """The canonical measurement family of a unit string."""
    return UNIT_FAMILIES.get(unit.strip().lower(), "unknown")


def check_units_and_magnitudes(
    *, left: Tuple[float, str], right: Tuple[float, str], comparison: str = "ratio"
) -> List[str]:
    """Unit and magnitude checks of step 32.

    Refuses silent unit conversions (``ms`` vs ``s``), cross-family comparisons
    (bytes vs time), the percent / percentage-point confusion, and per-device vs
    aggregate framings.  ``comparison='speedup'`` additionally refuses a zero
    denominator; the audit itself records the denominator and direction.
    """
    problems: List[str] = []
    left_value, left_unit = left
    right_value, right_unit = right
    if comparison not in ("ratio", "speedup", "change", "percentage_point"):
        problems.append(f"unknown comparison kind {comparison!r}")
        return problems
    if "per_device" in left_unit.lower() or "per_device" in right_unit.lower():
        left_unit = left_unit.replace("per_device", "")
        right_unit = right_unit.replace("per_device", "")
    if "aggregate" in left_unit.lower() or "aggregate" in right_unit.lower():
        problems.append("per-device and aggregate framings must not be mixed")
    left_family = unit_family(left_unit)
    right_family = unit_family(right_unit)
    if left_family == "unknown" or right_family == "unknown":
        problems.append(f"unknown unit in comparison ({left_unit!r} vs {right_unit!r})")
        return problems
    if left_family != right_family:
        problems.append(
            f"comparing different unit families ({left_unit} → {left_family} vs {right_unit} → {right_family})"
        )
        return problems
    if left_unit.strip().lower() != right_unit.strip().lower():
        problems.append(
            f"units {left_unit!r} and {right_unit!r} share a family but are not the same unit; "
            "an explicit conversion must be recorded"
        )
    if comparison == "speedup" and right_value == 0:
        problems.append("speedup denominator is zero")
    if comparison == "percentage_point" and left_family == "ratio":
        problems.append(
            "百分比与百分点混淆：a percentage-point change must not be rendered as a percentage change"
        )
    return problems


def check_extrapolation_boundaries(
    *, claim_scope: Mapping[str, Any], wording: str
) -> List[str]:
    """Flag quantifier widening (step 33).

    The rule is linguistic because the failure is linguistic: the raw is true, the
    quantifier is wider than the raw supports.
    """
    problems: List[str] = []
    has_single_shape = bool(claim_scope.get("operator_spec_id")) and not claim_scope.get("workload_spec_id")
    if has_single_shape and re.search(r"(全模型|end-to-end|模型整体|model[- ]level)", wording):
        problems.append("EXTRAPOLATION single_shape_to_model")
    model_count = len(claim_scope.get("model_artifact_ids", []) or ([claim_scope["model_artifact_id"]] if claim_scope.get("model_artifact_id") else []))
    if model_count == 1 and re.search(r"(模型族|所有模型|model family|all models)", wording):
        problems.append("EXTRAPOLATION single_model_to_family")
    hardware = list(claim_scope.get("hardware_ids", []))
    if len(hardware) == 1 and re.search(r"(跨硬件|cross[- ]hardware|任意设备|任意 GPU|any device)", wording):
        problems.append("EXTRAPOLATION single_gpu_to_cross_hardware")
    level = str(claim_scope.get("evidence_level", "PLANNED"))
    if level in ("SOURCE", "TEST", "RUNTIME") and re.search(r"(服务|service|goodput|SLO|P99)", wording):
        problems.append("EXTRAPOLATION micro_to_service")
    if int(claim_scope.get("run_count", 1)) == 1 and re.search(r"(稳定|reliably|consistently|总是)", wording):
        problems.append("EXTRAPOLATION single_run_to_distribution")
    return problems


# ── adjudication (steps 34, 37, 38, 43, 44) ─────────────────────────────────


@dataclass
class ContextAdjudication:
    """A human decision about an automatic finding (step 34)."""

    finding_ref: str
    verdict: str  # false_positive | true_positive | ambiguous
    reviewer: str
    reason: str
    rule_change: str = ""

    VERDICTS: Tuple[str, ...] = ("false_positive", "true_positive", "ambiguous")

    def __post_init__(self) -> None:
        if self.verdict not in self.VERDICTS:
            raise ConfigError(f"unknown adjudication verdict {self.verdict!r}")
        if not self.reviewer:
            raise ConfigError("an adjudication needs a reviewer; the scanner is not its own truth source")
        if len(self.reason) < 10:
            raise ConfigError("an adjudication needs a reason that a later reader can follow")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_ref": self.finding_ref,
            "verdict": self.verdict,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "rule_change": self.rule_change,
        }


@dataclass
class ClaimRevision:
    """A revision of a claim that keeps history (steps 37, 38)."""

    claim_id: str
    revision: int
    previous_revision: Optional[int]
    change_kind: str  # scope_narrowed | evidence_added | level_lowered | rejected | retracted
    reason: str
    diff: Mapping[str, Any] = field(default_factory=dict)
    reopen_condition: str = ""

    CHANGE_KINDS: Tuple[str, ...] = ("scope_narrowed", "evidence_added", "level_lowered", "rejected", "retracted")

    def __post_init__(self) -> None:
        if self.change_kind not in self.CHANGE_KINDS:
            raise ConfigError(f"unknown claim change kind {self.change_kind!r}")
        if self.change_kind in ("rejected", "retracted") and not self.reopen_condition:
            raise ConfigError(
                "a rejected/retracted claim must record the condition under which it may be reopened "
                "(E15-01 step 38)"
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "revision": self.revision,
            "previous_revision": self.previous_revision,
            "change_kind": self.change_kind,
            "reason": self.reason,
            "diff": dict(sorted(self.diff.items())),
            "reopen_condition": self.reopen_condition,
        }


def revise_claim(
    record: ClaimRecord,
    *,
    change_kind: str,
    reason: str,
    diff: Mapping[str, Any],
    revision: int,
    previous_revision: Optional[int] = None,
    reopen_condition: str = "",
) -> ClaimRevision:
    """Record a revision and apply the state change (steps 37, 38).

    The claim's history is *appended to*, never rewritten: the caller keeps the
    returned revision in the ledger's revision list, and must pass the revision
    number explicitly so a re-numbered history cannot silently collapse two
    changes into one.
    """
    if revision < 1:
        raise ConfigError("a claim revision number starts at 1")
    revision_record = ClaimRevision(
        claim_id=record.claim_id,
        revision=revision,
        previous_revision=previous_revision,
        change_kind=change_kind,
        reason=reason,
        diff=diff,
        reopen_condition=reopen_condition,
    )
    if change_kind == "rejected":
        record.transition("REJECTED")
    elif change_kind == "retracted":
        record.transition("RETRACTED")
    elif change_kind == "level_lowered":
        index = level_index(record.evidence_level)
        if index == 0:
            raise ConfigError("cannot lower the evidence level below PLANNED")
        record.evidence_level = EVIDENCE_LEVELS[index - 1]
    return revision_record


@dataclass
class ReleaseEligibility:
    """Per-claim release eligibility (step 44)."""

    claim_id: str
    gate_results: Mapping[str, str] = field(default_factory=dict)
    final_status: str = "DRAFT"
    allowed_channels: Tuple[str, ...] = ()
    forbidden_wording: Tuple[str, ...] = ()
    blocking_findings: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "gate_results": {key: self.gate_results[key] for key in sorted(self.gate_results)},
            "final_status": self.final_status,
            "allowed_channels": list(self.allowed_channels),
            "forbidden_wording": list(self.forbidden_wording),
            "blocking_findings": list(self.blocking_findings),
        }


def adjudicate_release_eligibility(record: ClaimRecord) -> ReleaseEligibility:
    """Compute the per-claim gate table (step 44).

    No overall PASS hides a single failed gate: every gate of §10 keeps its own
    result, and the allowed channels are derived from the *level actually earned*.
    """
    from hqsb.release.identity import CHANNEL_MINIMUM_LEVEL, channel_level_ok

    failures = record.gate_failures()
    severity = {code: FAILURE_SEVERITY.get(code, "P2") for code in failures}
    gate_results = {
        gate: ("fail" if any(code in _GATE_OF_FAILURE.get(gate, ()) for code in failures) else "pass")
        for gate in rec.CLAIM_GATES
    }
    blocking = sorted(code for code in failures if severity.get(code) == "P0")
    allowed: Tuple[str, ...] = ()
    if record.status == "VERIFIED" and "CHANNEL_CONFLICT" not in failures:
        allowed = tuple(
            channel
            for channel in PUBLIC_CHANNELS
            if channel in ("demo", "resume_bullet")
            or channel_level_ok(channel, record.evidence_level, record.evidence_level)
            and channel in CHANNEL_MINIMUM_LEVEL
        )
    forbidden: List[str] = []
    if record.evidence_level in ("PLANNED", "SOURCE"):
        forbidden.extend(("实测提升", "measured speedup", "已复现", "reproduced"))
    if "QUALITY_DISQUALIFIED" in failures:
        forbidden.extend(("更快", "faster", "提升", "improved"))
    if "FALLBACK_MIXED" in failures:
        forbidden.extend(("命中自定义算子", "custom kernel hit"))
    return ReleaseEligibility(
        claim_id=record.claim_id,
        gate_results=gate_results,
        final_status=record.status,
        allowed_channels=allowed,
        forbidden_wording=tuple(sorted(set(forbidden))),
        blocking_findings=tuple(blocking),
    )


_GATE_OF_FAILURE: Mapping[str, Tuple[str, ...]] = {
    "Identity": ("IDENTITY_INCOMPLETE",),
    "Semantics": ("SEMANTIC_AMBIGUOUS",),
    "Correctness": ("QUALITY_DISQUALIFIED",),
    "Measurement": ("MEASUREMENT_INVALID",),
    "ActualPath": ("FALLBACK_MIXED",),
    "Evidence": ("ORPHAN",),
    "Integrity": ("CORRUPTED",),
    "Freshness": ("STALE",),
    "Attribution": ("ATTRIBUTION_ERROR",),
    "Rendering": ("CHANNEL_CONFLICT",),
}


def coverage_and_risk_report(entries: Sequence[LedgerEntryView], *, surfaces: Sequence[PublicSurface]) -> Dict[str, Any]:
    """Coverage and risk by channel/type/level/stage (step 43).

    The free function exists so a single 100% headline cannot hide a completely
    uncovered channel: the report is per channel, not only overall.
    """
    by_channel: Dict[str, int] = {}
    for surface in surfaces:
        by_channel[surface.channel] = by_channel.get(surface.channel, 0) + 1
    status_counts: Dict[str, int] = {}
    for entry in entries:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
    return {
        "surfaces": len(surfaces),
        "surfaces_by_channel": {key: by_channel[key] for key in sorted(by_channel)},
        "claims": len(entries),
        "claims_by_status": {key: status_counts[key] for key in sorted(status_counts)},
        "uncovered_channels": [channel for channel in SURFACE_CHANNELS if channel not in by_channel],
    }


# ── negative controls and detector audit (steps 35, 36) ─────────────────────

#: Injected defect classes: each is a deliberately broken claim.
INJECTION_CLASSES: Tuple[Tuple[str, str, str], ...] = (
    ("inj-orphan", "P0", "a numeric claim with no raw reference"),
    ("inj-wrong-unit", "P0", "a claim whose rendered unit disagrees with the ledger"),
    ("inj-corrupted-hash", "P0", "an evidence digest that does not match the file"),
    ("inj-stale-commit", "P1", "a claim whose valid_from_commit predates the candidate"),
    ("inj-level-jump", "P0", "a SOURCE implementation advertised as RUNTIME evidence"),
    ("inj-bilingual-drift", "P0", "the English rendering carries a different number"),
    ("inj-attribution", "P0", "third-party/framework work attributed to the author"),
    ("inj-missing-owner", "P1", "a claim with no owner"),
    ("inj-channel-conflict", "P0", "the same claim rendered with two different values"),
    ("inj-extrapolation", "P1", "a single-shape result phrased as a model-level result"),
)


@dataclass
class DetectorAudit:
    """The result of running the audit's detectors over the injected set (step 36)."""

    injected: Tuple[Mapping[str, Any], ...] = ()
    detected: Tuple[str, ...] = ()
    false_positives: Tuple[str, ...] = ()
    detector: str = "hqsb.s15.claims"

    #: The checks each injection class must be caught by (at least one).
    EXPECTED_DETECTOR: ClassVar[Mapping[str, str]] = {
        "inj-orphan": "detect_orphans",
        "inj-wrong-unit": "check_units_and_magnitudes",
        "inj-corrupted-hash": "verify_digests",
        "inj-stale-commit": "detect_stale",
        "inj-level-jump": "check_level_upgrade",
        "inj-bilingual-drift": "check_bilingual_fact_consistency",
        "inj-attribution": "ContributionRecord",
        "inj-missing-owner": "detect_orphans",
        "inj-channel-conflict": "detect_conflicts",
        "inj-extrapolation": "check_extrapolation_boundaries",
    }

    def run(self) -> Dict[str, Any]:
        from hqsb.release.telemetry import DetectorMetrics

        metrics = DetectorMetrics(
            detector=self.detector, injected=self.injected, detected=self.detected, false_positives=self.false_positives
        )
        report = metrics.metrics()
        report["expected_detector"] = dict(sorted(self.EXPECTED_DETECTOR.items()))
        return report


def inject_negative_controls() -> Tuple[Dict[str, Any], ...]:
    """Construct the injected defect set as *data* (step 35).

    The injections are described, not executed against real documents: the audit
    applies the corresponding check and records whether it fired.  Every severe
    injection must be detected before a clean result counts.
    """
    return tuple(
        {
            "injection_id": injection_id,
            "severity": severity,
            "description": description,
            "expected_detector": DetectorAudit.EXPECTED_DETECTOR.get(injection_id, ""),
        }
        for injection_id, severity, description in INJECTION_CLASSES
    )


def check_level_upgrade(*, claimed_level: str, cited_levels: Sequence[str]) -> List[str]:
    """Adapter so the injection table can call the level check by name."""
    from hqsb.release.contracts import check_evidence_level_upgrade

    return check_evidence_level_upgrade(claimed_level=claimed_level, cited_levels=cited_levels)


# ── freeze (step 45) ────────────────────────────────────────────────────────


@dataclass
class ClaimLedger:
    """The frozen ledger: rendering reads from here, never from the document (step 45)."""

    audit_id: str
    candidate_id: str = ""
    claims: List[ClaimRecord] = field(default_factory=list)
    revisions: List[ClaimRevision] = field(default_factory=list)
    evidence_graph: Optional[EvidenceGraph] = None
    renderings: Tuple[Mapping[str, Any], ...] = ()
    frozen: bool = False

    def add(self, record: ClaimRecord) -> ClaimRecord:
        if self.frozen:
            raise ConfigError("the ledger is frozen; a change requires a new audit run (step 45)")
        problems = record.problems()
        if problems:
            raise ConfigError("refusing to add an invalid claim: " + "; ".join(problems))
        self.claims.append(record)
        return record

    def claim(self, claim_id: str) -> ClaimRecord:
        for record in self.claims:
            if record.claim_id == claim_id:
                return record
        raise ConfigError(f"unknown claim {claim_id!r}")

    @property
    def ledger_digest(self) -> str:
        return canonical_digest([record.as_dict() for record in sorted(self.claims, key=lambda item: item.claim_id)])

    @property
    def rendering_bundle_digest(self) -> str:
        return canonical_digest(sorted((dict(item) for item in self.renderings), key=lambda item: str(item.get("claim_id", ""))))

    def freeze(self) -> Dict[str, Any]:
        if not self.claims:
            raise ConfigError(
                "refusing to freeze an empty ledger: a frozen ledger with no claims cannot gate a release"
            )
        problems: List[str] = []
        for record in self.claims:
            problems.extend(f"{record.claim_id}: {problem}" for problem in record.problems())
        if self.evidence_graph is not None:
            problems.extend(self.evidence_graph.validate())
        if problems:
            raise ConfigError("refusing to freeze a ledger with problems: " + "; ".join(problems[:5]))
        self.frozen = True
        return {
            "schema_version": f"{SCHEMA_PREFIX}.claim-ledger.v1",
            "audit_id": self.audit_id,
            "candidate_id": self.candidate_id,
            "claims": len(self.claims),
            "ledger_sha256": self.ledger_digest,
            "evidence_graph_root": self.evidence_graph.root if self.evidence_graph else "",
            "rendering_bundle_sha256": self.rendering_bundle_digest,
            "frozen": True,
        }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

#: ``E15-01`` 45 protocol steps → interfaces of this module (and its neighbours).
PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 ReleaseCandidateSnapshot", ("contracts.ReleaseCandidateSnapshot", "contracts.new_candidate")),
    (2, "建立公开渠道全量清单", ("claims.enumerate_surfaces", "claims.PublicSurface", "claims.SURFACE_CHANNELS")),
    (3, "定义 claim 分类法", ("claims.classify_claim", "records.CLAIM_TYPES")),
    (4, "冻结证据等级规则", ("identity.required_support", "identity.EVIDENCE_LEVELS", "specs.EXPECTED_KINDS")),
    (5, "定义可检测语言模式", ("claims.DETECTABLE_PATTERNS", "claims.extract_claim_candidates")),
    (6, "定义非 claim 排除规则", ("claims.NON_CLAIM_EXCLUSIONS", "claims.ClaimCandidate")),
    (7, "全量抽取候选语句", ("claims.extract_claim_candidates", "claims.ClaimCandidate.as_dict")),
    (8, "单独抽取全部数字与单位", ("claims.extract_numeric_expressions", "claims.NUMERIC_CLASSES")),
    (9, "抽取隐式比较", ("claims.extract_implicit_comparisons",)),
    (10, "抽取绝对能力声明", ("claims.extract_capability_claims",)),
    (11, "规范化语义相同的候选", ("claims.canonicalise_candidates", "claims.CanonicalClaim")),
    (12, "建立中英文配对", ("claims.pair_bilingual", "contracts.check_bilingual_fact_consistency")),
    (13, "给每条 claim 分配稳定 ID", ("claims.assign_claim_id", "identity.stable_claim_id")),
    (14, "填写 claim 类型与 estimand", ("claims.classify_claim", "contracts.ClaimRecord.problems")),
    (15, "绑定 C1 模型身份", ("claims.ClaimBindings", "claims.bind_fact_fields")),
    (16, "绑定 C2/C3/C4/C5 范围", ("claims.ClaimBindings", "claims.bind_fact_fields")),
    (17, "绑定硬件和环境范围", ("claims.ClaimBindings", "experiment.environment_fingerprint")),
    (18, "绑定 baseline 身份", ("claims.ClaimBindings", "contracts.ClaimRecord.problems")),
    (19, "绑定正确性与质量资格", ("contracts.check_quality_before_performance", "claims.ClaimBindings")),
    (20, "绑定统计对象", ("claims.ClaimBindings", "contracts.ClaimEffect")),
    (21, "构造 Claim→Evidence DAG", ("claims.EvidenceGraph", "claims.EvidenceEdge", "records.EVIDENCE_EDGE_TYPES")),
    (22, "验证全部 Evidence URI", ("claims.verify_uris", "identity.assert_public_uri")),
    (23, "校验逐文件 digest 与聚合根", ("claims.verify_digests", "identity.content_address_aggregate")),
    (24, "验证 raw 可解析性", ("claims.verify_uris", "identity.EvidenceRef")),
    (25, "验证派生可重建性", ("claims.verify_derivable",)),
    (26, "记录贡献与责任归属", ("contracts.ContributionRecord",)),
    (27, "定义时间有效区间", ("identity.FrozenInputs", "contracts.ClaimRecord.problems")),
    (28, "构建失效依赖图", ("claims.detect_stale", "claims.LedgerEntryView.invalidation_keys")),
    (29, "执行 orphan 检测", ("claims.detect_orphans", "claims.ORPHAN_KINDS")),
    (30, "执行 stale 检测", ("claims.detect_stale",)),
    (31, "执行冲突与重复检测", ("claims.detect_conflicts",)),
    (32, "执行单位和数量级检查", ("claims.check_units_and_magnitudes", "claims.UNIT_FAMILIES")),
    (33, "执行外推边界检查", ("claims.check_extrapolation_boundaries", "claims.EXTRAPOLATION_KINDS")),
    (34, "人工裁决上下文歧义", ("claims.ContextAdjudication",)),
    (35, "注入负对照 Claim", ("claims.inject_negative_controls", "claims.INJECTION_CLASSES")),
    (36, "计算检测性能", ("claims.DetectorAudit", "telemetry.DetectorMetrics")),
    (37, "修订有证据但表述过强的 claim", ("claims.revise_claim", "claims.ClaimRevision")),
    (38, "撤回无法证明的 claim", ("claims.revise_claim", "contracts.ClaimRecord.transition")),
    (39, "从 Ledger 生成渠道文案", ("contracts.ClaimRecord.render", "identity.CHANNEL_MINIMUM_LEVEL")),
    (40, "建立 CI Claim Gate", ("claims.LedgerEntryView", "claims.detect_stale", "records.CLAIM_GATES")),
    (41, "抽取代表 claim 做端到端追溯", ("claims.EvidenceGraph.ancestors", "telemetry.EvidenceLookupTrial")),
    (42, "进行独立 claim review", ("claims.ReleaseEligibility", "claims.coverage_and_risk_report")),
    (43, "生成覆盖与风险报告", ("claims.coverage_and_risk_report",)),
    (44, "逐条裁决发布资格", ("claims.adjudicate_release_eligibility", "claims.ReleaseEligibility")),
    (45, "冻结 Final Claim Ledger", ("claims.ClaimLedger.freeze", "claims.ClaimLedger.ledger_digest")),
)

TITLE = "全局 Claim Ledger、声明扫描与证据门禁"
LEVEL = "P0"
CLAIM_BOUNDARY = (
    "E15-01 只证明“当前候选公开陈述有证据且无已知越界”；不证明证据已被独立第三方复现（E15-09）"
)


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the claim-audit interfaces (labelled smoke)."""
    problems: List[str] = []
    surface = PublicSurface(
        surface_id="S1",
        path="README.md",
        channel="readme",
        language="zh",
        text="我们在 Qwen3 上实现 RMSNorm kernel，端到端提升 18%，支持 CUDA。\n计划支持 Ascend。",
    )
    candidates = extract_claim_candidates(surface)
    if not candidates:
        problems.append("the extractor found nothing in a sentence full of claims")
    excluded = [candidate for candidate in candidates if candidate.classification == "excluded"]
    if not excluded:
        problems.append("the plan sentence ('计划支持 Ascend') was not excluded")
    graph = EvidenceGraph()
    graph.add_entity("claim:CLM-1", kind="claim")
    graph.add_entity("report:r1", kind="manifest", digest="sha256:" + "2" * 64)
    graph.add_entity("raw:s1", kind="manifest", digest="sha256:" + "3" * 64)
    graph.add_edge("wasDerivedFrom", "raw:s1", "report:r1")
    graph.add_edge("wasDerivedFrom", "report:r1", "claim:CLM-1")
    if graph.validate():
        problems.append(f"a well-formed graph was rejected: {graph.validate()}")
    orphan_report = detect_orphans(
        [LedgerEntryView(claim_id="CLM-1", status="VERIFIED", owner="", scope_present=False)]
    )
    if orphan_report["no_owner"] != ["CLM-1"] or orphan_report["no_scope"] != ["CLM-1"]:
        problems.append("orphan classification missed the owner/scope cases")
    units = check_units_and_magnitudes(left=(120.0, "ms"), right=(100.0, "s"))
    if not units:
        problems.append("ms-vs-s comparison was not flagged")
    ledger = ClaimLedger(audit_id="audit-smoke", candidate_id="cand-1")
    freezes = 0
    try:
        ledger.freeze()
        freezes = 1
    except ConfigError:
        pass
    if freezes:
        problems.append("an empty ledger was frozen without a candidate")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "candidates_extracted": len(candidates),
        "excluded": len(excluded),
        "graph_root": graph.root,
        "injections": len(inject_negative_controls()),
        "problems": problems,
    }
