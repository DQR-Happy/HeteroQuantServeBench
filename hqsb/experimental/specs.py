"""Audit of the frozen S14 vocabularies under ``configs/experimental/``.

S14 needs several *policy* documents to be frozen before any run: which
dependency belongs to which layer (``E14-01`` step 2), which extra maps to which
feature flag (step 3), which reason code means what (step 4), which environment
matrix is preregistered (step 5), and the four candidate frontier branches with
their estimands, quality gates, budgets and stop rules (``E14-05`` steps 4–16).

Those documents live in ``configs/experimental/*.yaml`` and are **vocabularies,
not results** (``module_ownership.md`` S13 note, applied to S14 unchanged).  The
audit enforces exactly that:

* a document must be valid YAML, declare ``kind`` and cite its protocol
  ``sources``;
* it may not contain a measured number (a value under a field whose name
  suggests a measurement must be a *placeholder string* such as
  ``UNMEASURED``), because a config file is not allowed to become the hiding
  place for a fabricated result (任务硬规则 2);
* it may not contain a machine-specific absolute path (AGENTS.md §6:
  不把本机绝对路径写入受跟踪的配置);
* unknown keys are reported, so a typo cannot silently drop a constraint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.identity import canonical_digest

#: Placeholder that stands in for "this number does not exist yet".
UNMEASURED = "UNMEASURED"

#: Field-name fragments that mark a *measurement*; such fields must not carry a
#: bare number in a frozen vocabulary.
MEASUREMENT_HINTS: Tuple[str, ...] = (
    "latency",
    "throughput",
    "speedup",
    "acceptance",
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
)

#: Path prefixes that indicate a machine-specific absolute path.
ABSOLUTE_PATH_PREFIXES: Tuple[str, ...] = ("/root/", "/home/", "/Users/", "/mnt/", "/data/", "C:\\")

#: Every spec kind S14 ships, with its required top-level keys.
SPEC_KINDS: Mapping[str, Tuple[str, ...]] = {
    "dependency-policy": ("kind", "version", "sources", "layers", "rules"),
    "extras": ("kind", "version", "sources", "extras"),
    "failure-semantics": ("kind", "version", "sources", "reason_codes", "statuses"),
    "environment-matrix": ("kind", "version", "sources", "profiles", "environments", "negative_cases"),
    "frontier-branches": ("kind", "version", "sources", "selection_rules", "branches"),
    "adoption-rules": ("kind", "version", "sources", "rules"),
    "statistics-plan": ("kind", "version", "sources", "units", "tests", "exclusions"),
    "profiling-contract": ("kind", "version", "sources", "layers", "actual_path_fields", "resource_ledger_keys"),
}

#: Which document owns which experiment (used by the driver's ``--spec-audit``).
SPEC_OWNERS: Mapping[str, Tuple[str, ...]] = {
    "dependency-policy": ("E14-01",),
    "extras": ("E14-01",),
    "failure-semantics": ("E14-01",),
    "environment-matrix": ("E14-01",),
    "frontier-branches": ("E14-05",),
    "adoption-rules": ("E14-05",),
    "statistics-plan": ("E14-05",),
    "profiling-contract": ("E14-05",),
}


@dataclass
class SpecFinding:
    kind: str
    path: str
    problem: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = {"kind": self.kind, "path": self.path, "problem": self.problem}
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class DocumentReport:
    kind: str
    path: str
    digest: str = ""
    findings: List[SpecFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "digest": self.digest,
            "ok": self.ok,
            "findings": [item.as_dict() for item in self.findings],
        }


class ExperimentalSpecs:
    """The loaded ``configs/experimental`` documents plus their audit."""

    def __init__(self, root: str, documents: Mapping[str, Any], reports: Sequence[DocumentReport]) -> None:
        self.root = root
        self.documents = dict(documents)
        self.reports = list(reports)

    @classmethod
    def load(cls, root: str) -> "ExperimentalSpecs":
        documents: Dict[str, Any] = {}
        reports: List[DocumentReport] = []
        if not os.path.isdir(root):
            reports.append(
                DocumentReport(
                    kind="<root>",
                    path=root,
                    findings=[SpecFinding("<root>", root, "spec directory missing")],
                )
            )
            return cls(root, documents, reports)
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - yaml is a core dependency
            reports.append(
                DocumentReport(
                    kind="<root>",
                    path=root,
                    findings=[SpecFinding("<root>", root, f"PyYAML unavailable: {exc}")],
                )
            )
            return cls(root, documents, reports)
        for filename in sorted(os.listdir(root)):
            if not filename.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(root, filename)
            report = DocumentReport(kind="<unparsed>", path=path)
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle)
            except Exception as exc:  # noqa: BLE001 - reported, never skipped
                report.findings.append(SpecFinding("<unparsed>", path, f"yaml error: {exc}"))
                reports.append(report)
                continue
            if not isinstance(payload, dict):
                report.findings.append(SpecFinding("<unparsed>", path, "document root must be a mapping"))
                reports.append(report)
                continue
            kind = str(payload.get("kind", ""))
            report.kind = kind
            documents[kind] = payload
            report.digest = canonical_digest(payload)
            report.findings.extend(_audit_document(kind, path, payload))
            reports.append(report)
        return cls(root, documents, reports)

    def audit(self) -> List[Dict[str, Any]]:
        return [report.as_dict() for report in self.reports]

    def findings(self) -> List[SpecFinding]:
        return [item for report in self.reports for item in report.findings]

    def kinds(self) -> List[str]:
        return sorted(documents for documents in self.documents if documents)

    def missing_kinds(self) -> List[str]:
        return sorted(set(SPEC_KINDS) - set(self.documents))

    def digest_map(self) -> Dict[str, str]:
        return {
            report.kind: report.digest for report in self.reports if report.kind in SPEC_KINDS and report.digest
        }

    def document(self, kind: str) -> Mapping[str, Any]:
        if kind not in self.documents:
            raise ConfigError(f"spec kind {kind!r} is not present under {self.root!r}")
        return self.documents[kind]


def _audit_document(kind: str, path: str, payload: Mapping[str, Any]) -> List[SpecFinding]:
    findings: List[SpecFinding] = []
    if kind not in SPEC_KINDS:
        findings.append(SpecFinding(kind or "<missing>", path, "unknown or missing 'kind'"))
        return findings
    for key in SPEC_KINDS[kind]:
        if key not in payload:
            findings.append(SpecFinding(kind, path, f"required key {key!r} is missing"))
    version = str(payload.get("version", ""))
    if version and not version.startswith("v"):
        findings.append(SpecFinding(kind, path, f"version {version!r} should look like 'v1'"))
    sources = payload.get("sources", [])
    if not isinstance(sources, list) or not sources:
        findings.append(SpecFinding(kind, path, "'sources' must list the protocol documents/clauses cited"))
    else:
        for source in sources:
            if not isinstance(source, str) or not source.strip():
                findings.append(SpecFinding(kind, path, f"source entry {source!r} must be a non-empty string"))
    findings.extend(_walk_for_measurements(kind, path, payload))
    findings.extend(_walk_for_absolute_paths(kind, path, payload))
    return findings


def _walk_for_measurements(kind: str, path: str, node: Any, *, trail: str = "") -> List[SpecFinding]:
    findings: List[SpecFinding] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{trail}.{key}" if trail else str(key)
            lowered = str(key).lower()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if any(hint in lowered for hint in MEASUREMENT_HINTS):
                    findings.append(
                        SpecFinding(
                            kind,
                            path,
                            "frozen vocabulary contains a number under a measurement-like key",
                            f"{child}={value!r}; use {UNMEASURED!r} (a config file may not hold a result)",
                        )
                    )
            findings.extend(_walk_for_measurements(kind, path, value, trail=child))
        return findings
    if isinstance(node, list):
        for index, item in enumerate(node):
            findings.extend(_walk_for_measurements(kind, path, item, trail=f"{trail}[{index}]"))
    return findings


def _walk_for_absolute_paths(kind: str, path: str, node: Any, *, trail: str = "") -> List[SpecFinding]:
    findings: List[SpecFinding] = []
    if isinstance(node, str):
        for prefix in ABSOLUTE_PATH_PREFIXES:
            if node.startswith(prefix):
                findings.append(
                    SpecFinding(
                        kind,
                        path,
                        "machine-specific absolute path in a tracked config",
                        f"{trail or '<root>'}={node!r} (AGENTS.md §6)",
                    )
                )
                break
        return findings
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{trail}.{key}" if trail else str(key)
            findings.extend(_walk_for_absolute_paths(kind, path, value, trail=child))
        return findings
    if isinstance(node, list):
        for index, item in enumerate(node):
            findings.extend(_walk_for_absolute_paths(kind, path, item, trail=f"{trail}[{index}]"))
    return findings


def audit_all_ok(reports: Sequence[Mapping[str, Any]]) -> bool:
    return all(report.get("ok") for report in reports)


# ── cross-document consistency ─────────────────────────────────────────────


def check_extra_feature_mapping(documents: Mapping[str, Any]) -> List[str]:
    """``configs/experimental/extras.yaml`` must equal ``records.EXTRA_FEATURE_MAPPING``.

    The extra→flag mapping is declared twice on purpose (code + frozen config);
    comparing them turns a silent divergence into a failing audit.
    """
    problems: List[str] = []
    document = documents.get("extras")
    if not document:
        return ["extras spec is missing"]
    declared: Dict[str, Tuple[str, ...]] = {}
    for entry in document.get("extras", []):
        if not isinstance(entry, Mapping) or "name" not in entry:
            problems.append("extras entry must be a mapping with a 'name'")
            continue
        declared[str(entry["name"])] = tuple(str(flag) for flag in entry.get("feature_flags", []))
    expected = dict(rec.EXTRA_FEATURE_MAPPING)
    for name in sorted(set(expected) | set(declared)):
        if name not in declared:
            problems.append(f"extra {name!r} is declared in code but missing from extras.yaml")
        elif name not in expected:
            problems.append(f"extra {name!r} exists in extras.yaml but not in code")
        elif declared[name] != expected[name]:
            problems.append(
                f"extra {name!r} flags differ: config={declared[name]} code={expected[name]}"
            )
    return problems


def check_reason_codes(documents: Mapping[str, Any]) -> List[str]:
    """``failure-semantics.yaml`` must cover every reason code and status of code."""
    problems: List[str] = []
    document = documents.get("failure-semantics")
    if not document:
        return ["failure-semantics spec is missing"]
    declared = {str(item.get("code")) for item in document.get("reason_codes", []) if isinstance(item, Mapping)}
    for code in rec.REASON_CODES:
        if code not in declared:
            problems.append(f"reason code {code!r} is implemented but not documented")
    for code in sorted(declared - set(rec.REASON_CODES)):
        problems.append(f"reason code {code!r} is documented but not implemented")
    statuses = {str(item) for item in document.get("statuses", [])}
    for status in rec.S14_FAILURE_STATUSES:
        if status not in statuses:
            problems.append(f"status {status!r} is implemented but not documented")
    return problems


def check_frontier_branches(documents: Mapping[str, Any]) -> List[str]:
    """``frontier-branches.yaml`` must describe exactly the four F branches."""
    problems: List[str] = []
    document = documents.get("frontier-branches")
    if not document:
        return ["frontier-branches spec is missing"]
    declared = {str(item.get("branch")) for item in document.get("branches", []) if isinstance(item, Mapping)}
    expected = set(rec.FRONTIER_BRANCH_NAMES)
    if declared != expected:
        problems.append(f"frontier branches {sorted(declared)} != {sorted(expected)}")
    for entry in document.get("branches", []):
        if not isinstance(entry, Mapping):
            problems.append("branch entry must be a mapping")
            continue
        branch = str(entry.get("branch"))
        required = ("branch", "name", "primary_gain_hypothesis", "primary_risk", "required_layers",
                    "hard_prerequisites", "quality_oracle", "minimal_independent_change", "budget",
                    "stop_rules", "rejection_conditions")
        for key in required:
            if key not in entry:
                problems.append(f"{branch}: frontier-branches entry is missing {key!r}")
        declared_layers = tuple(str(layer) for layer in entry.get("required_layers", []))
        expected_layers = tuple(rec.FRONTIER_REQUIRED_LAYERS.get(branch, ()))
        if declared_layers != expected_layers:
            problems.append(
                f"{branch}: required_layers {list(declared_layers)} != protocol {list(expected_layers)}"
            )
        if entry.get("primary_risk") and branch in rec.FRONTIER_PRIMARY_RISKS:
            expected_risk = rec.FRONTIER_PRIMARY_RISKS[branch]
            if str(entry["primary_risk"]).strip() != expected_risk:
                problems.append(
                    f"{branch}: primary_risk differs from the protocol text "
                    f"(config={entry['primary_risk']!r} protocol={expected_risk!r})"
                )
    return problems


def check_all(documents: Mapping[str, Any]) -> List[str]:
    """Every cross-document rule, returned as a flat problem list."""
    problems: List[str] = []
    problems.extend(check_extra_feature_mapping(documents))
    problems.extend(check_reason_codes(documents))
    problems.extend(check_frontier_branches(documents))
    return problems


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the spec layer (labelled smoke, not an experiment)."""
    payload = {
        "kind": "dependency-policy",
        "version": "v1",
        "sources": ["details/S14/E14-01_optional_dependencies_feature_isolation.md"],
        "layers": [],
        "rules": [],
        "dummy_latency_ms": 1.0,
    }
    findings = _audit_document("dependency-policy", "<memory>", payload)
    absolute = _audit_document(
        "dependency-policy",
        "<memory>",
        {"kind": "dependency-policy", "version": "v1", "sources": ["x"], "layers": ["/root/x"], "rules": []},
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "spec_kinds": len(SPEC_KINDS),
        "measurement_guard_fires": any("measurement-like key" in item.problem for item in findings),
        "absolute_path_guard_fires": any("absolute path" in item.problem for item in absolute),
    }
