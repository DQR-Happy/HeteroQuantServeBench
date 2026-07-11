"""Audit of the frozen S15 vocabularies under ``configs/release/``.

S15 needs several *policy* documents frozen before any audit runs: the claim
taxonomy, evidence-level rules and channel limits (``E15-01`` §11), the detector
patterns and non-claim exclusions (§5/step 5–6), the §22 evidence package, the
documentation code-block classes and defect severity matrix (``E15-04``), the
release policy and supply-chain finding dispositions (``E15-05``), the figure
sampling strata and rebuild classes (``E15-06``), the demo/narrative vocabularies
(``E15-07``/``E15-08``/``E15-11``), and the reproduction/upstream ladders
(``E15-09``/``E15-10``).

Those documents live in ``configs/release/*.yaml`` and are **vocabularies, not
results**.  The audit enforces exactly that:

* a document must be valid YAML, declare ``kind`` and cite its protocol
  ``sources``;
* it may not contain a measured number (a value under a field whose name
  suggests a measurement must be a *placeholder string* such as ``UNMEASURED``),
  because a config file is not allowed to become the hiding place for a
  fabricated result (任务硬规则 2);
* it may not contain a machine-specific absolute path (AGENTS.md §6:
  不把本机绝对路径写入受跟踪的配置);
* unknown top-level keys are reported, so a typo cannot silently drop a
  constraint;
* the eight expected kinds must all be present (a missing vocabulary would let
  the driver invent its own, un-audited rules at run time).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release.identity import EVIDENCE_LEVELS, PUBLIC_CHANNELS, canonical_digest

#: Placeholder that stands in for "this number does not exist yet".
UNMEASURED = "UNMEASURED"

#: Field-name fragments that mark a *measurement*; such fields must not carry a
#: bare number in a frozen vocabulary.
MEASUREMENT_HINTS: Tuple[str, ...] = (
    "latency",
    "throughput",
    "speedup",
    "acceptance_rate",
    "hit_rate",
    "occupancy",
    "bytes",
    "tokens_per",
    "memory",
    "power",
    "energy",
    "score",
    "accuracy",
    "quality_value",
    "elapsed",
    "cost_per",
    "duration_s",
    "download_bytes",
    "peak_rss",
)

#: Path prefixes that indicate a machine-specific absolute path.
ABSOLUTE_PATH_PREFIXES: Tuple[str, ...] = ("/root/", "/home/", "/Users/", "/mnt/", "/data/", "C:\\")

#: Allowed top-level keys of a frozen vocabulary document.
ALLOWED_TOP_KEYS: Tuple[str, ...] = (
    "kind",
    "version",
    "sources",
    "note",
    "rules",
    "patterns",
    "exclusions",
    "levels",
    "channels",
    "severities",
    "classes",
    "matrix",
    "states",
    "ladders",
    "policy",
    "sections",
    "tasks",
    "anchors",
    "quotas",
    # per-document vocabularies (each is a frozen list/map, not a measured value)
    "command_verdicts",
    "uniform_package",
    "strata",
    "mandatory_samples",
    "rebuild_classes",
    "access_budget",
    "asset_kinds",
    "finding_states",
    "help_levels",
    "quality_dimensions",
    "upstream_dispositions",
    "duration_gates",
    "fault_scenarios",
    "slo_fields",
    "hard_failures",
    "first_impression_participants",
    "first_impression_minutes",
)

#: The vocabularies S15 must freeze before a run.
EXPECTED_KINDS: Tuple[str, ...] = (
    "claim-taxonomy",
    "detector-patterns",
    "evidence-package",
    "documentation-policy",
    "release-policy",
    "figure-audit",
    "demo-narrative",
    "reproduction-upstream",
)

#: Protocol sources every document must be able to cite (existence is checked by
#: the generator, not here — this audit only verifies the strings look like paths).
SOURCE_PREFIXES: Tuple[str, ...] = ("docs/stage_experiments/", "docs/architecture/", "docs/stages/")


@dataclass
class SpecReport:
    """One audited document."""

    path: str
    kind: str = ""
    version: str = ""
    problems: List[str] = field(default_factory=list)
    unknown_keys: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "version": self.version,
            "ok": self.ok,
            "problems": list(self.problems),
            "unknown_keys": list(self.unknown_keys),
            "warnings": list(self.warnings),
        }


def _walk(value: Any, path: str = "$") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = [(path, value)]
    if isinstance(value, Mapping):
        for key, item in value.items():
            items.extend(_walk(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            items.extend(_walk(item, f"{path}[{index}]"))
    return items


def _field_name(path: str) -> str:
    tail = path.rsplit(".", 1)[-1]
    return tail.split("[", 1)[0].lower()


def audit_document(path: str, payload: Any) -> SpecReport:
    """Audit one loaded YAML document against the vocabulary rules."""
    report = SpecReport(path=path)
    if not isinstance(payload, Mapping):
        report.problems.append("document is not a mapping")
        return report
    report.kind = str(payload.get("kind", ""))
    report.version = str(payload.get("version", ""))
    if not report.kind:
        report.problems.append("document does not declare 'kind'")
    if not report.version:
        report.problems.append("document does not declare 'version'")
    sources = payload.get("sources")
    if not sources or not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        report.problems.append("document must cite its protocol 'sources' (a vocabulary without a source is an opinion)")
    else:
        for source in sources:
            if not isinstance(source, str) or not source.startswith(SOURCE_PREFIXES):
                report.problems.append(f"source {source!r} does not look like a repository document path")
    report.unknown_keys = sorted(str(key) for key in payload if key not in ALLOWED_TOP_KEYS)
    for key in report.unknown_keys:
        report.problems.append(f"unknown top-level key {key!r}; a typo must not silently drop a constraint")
    for sub_path, value in _walk(payload):
        if isinstance(value, str):
            for prefix in ABSOLUTE_PATH_PREFIXES:
                if value.startswith(prefix) or f" {prefix}" in value:
                    report.problems.append(f"{sub_path}: machine-specific path in a tracked config")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            name = _field_name(sub_path)
            if any(hint in name for hint in MEASUREMENT_HINTS):
                report.problems.append(
                    f"{sub_path}: a frozen vocabulary may not carry a measured number "
                    f"(use {UNMEASURED!r}); got {value!r}"
                )
    return report


class ReleaseSpecs:
    """All frozen documents under one directory."""

    def __init__(self, documents: Mapping[str, Any]) -> None:
        self.documents: Dict[str, Any] = dict(documents)

    @classmethod
    def load(cls, directory: str) -> "ReleaseSpecs":
        if not os.path.isdir(directory):
            raise ConfigError(f"release spec directory does not exist: {directory!r}")
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - core dependency per pyproject.toml
            raise ConfigError(f"PyYAML is required to audit release specs: {exc}") from exc
        documents: Dict[str, Any] = {}
        for name in sorted(os.listdir(directory)):
            if not name.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, name)
            with open(path, encoding="utf-8") as handle:
                documents[path] = yaml.safe_load(handle)
        if not documents:
            raise ConfigError(
                f"no YAML documents under {directory!r}; an empty spec directory cannot freeze a vocabulary"
            )
        return cls(documents)

    def audit(self) -> List[Dict[str, Any]]:
        return [audit_document(path, payload).as_dict() for path, payload in sorted(self.documents.items())]

    def missing_kinds(self) -> List[str]:
        present = {str(payload.get("kind", "")) for payload in self.documents.values() if isinstance(payload, Mapping)}
        return [kind for kind in EXPECTED_KINDS if kind not in present]

    def duplicate_kinds(self) -> List[str]:
        seen: Dict[str, int] = {}
        for payload in self.documents.values():
            if isinstance(payload, Mapping):
                kind = str(payload.get("kind", ""))
                seen[kind] = seen.get(kind, 0) + 1
        return sorted(kind for kind, count in seen.items() if count > 1)

    def digest(self) -> str:
        return canonical_digest({path: payload for path, payload in sorted(self.documents.items())})


def check_cross_document(documents: Mapping[str, Any]) -> List[str]:
    """Cross-document checks: the vocabularies must agree with each other.

    A claim taxonomy that lists a channel the evidence-package document does not
    know, or a demo document that references a measurement state outside
    ``records``, is a contradiction that would otherwise surface mid-experiment.
    """
    problems: List[str] = []
    claim_taxonomy: Mapping[str, Any] = {}
    for payload in documents.values():
        if isinstance(payload, Mapping) and payload.get("kind") == "claim-taxonomy":
            claim_taxonomy = payload
    if claim_taxonomy:
        channels = claim_taxonomy.get("channels")
        if isinstance(channels, Mapping):
            unknown = sorted(str(name) for name in channels if str(name) not in PUBLIC_CHANNELS)
            if unknown:
                problems.append(f"claim-taxonomy lists unknown public channels: {', '.join(unknown)}")
        levels = claim_taxonomy.get("levels")
        if isinstance(levels, Mapping):
            unknown = sorted(str(name) for name in levels if str(name) not in EVIDENCE_LEVELS)
            if unknown:
                problems.append(f"claim-taxonomy lists unknown evidence levels: {', '.join(unknown)}")
    return problems


def audit_all_ok(reports: Sequence[Mapping[str, Any]]) -> bool:
    return all(bool(report.get("ok")) for report in reports)


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the spec layer (labelled smoke, not an experiment)."""
    problems: List[str] = []
    good = {
        "kind": "release-policy",
        "version": "v1",
        "sources": ["docs/stage_experiments/details/S15/E15-05_release_supply_chain_provenance.md"],
        "policy": {"decision_maker": "human", "download_bytes": UNMEASURED},
    }
    clean = audit_document("good.yaml", good)
    if not clean.ok:
        problems.append(f"a valid document was rejected: {clean.problems}")
    negative_controls = {
        "measured_number": audit_document(
            "bad.yaml",
            {
                "kind": "release-policy",
                "version": "v1",
                "sources": ["docs/stage_experiments/details/S15/README.md"],
                "policy": {"speedup_latency_ms": 12.5},
            },
        ).problems,
        "local_path": audit_document(
            "bad2.yaml",
            {
                "kind": "release-policy",
                "version": "v1",
                "sources": ["docs/stage_experiments/details/S15/README.md"],
                "note": "see /root/artifacts/x.json",
            },
        ).problems,
        "no_sources": audit_document("bad3.yaml", {"kind": "x", "version": "v1"}).problems,
        "unknown_key": audit_document(
            "bad4.yaml",
            {
                "kind": "x",
                "version": "v1",
                "sources": ["docs/stage_experiments/README.md"],
                "typo_key": "x",
            },
        ).problems,
    }
    for name, findings in negative_controls.items():
        if not findings:
            problems.append(f"negative control {name!r} was not rejected")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "expected_kinds": list(EXPECTED_KINDS),
        "negative_controls_rejected": sum(1 for findings in negative_controls.values() if findings),
        "problems": problems,
    }
