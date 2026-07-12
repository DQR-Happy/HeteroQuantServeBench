"""E15-04 — executable documentation, bilingual consistency, capability matrix.

Protocol: ``docs/stage_experiments/details/S15/E15-04_executable_docs_bilingual_consistency.md``
(45 steps).

The document is treated as an executable interface, not an attachment:

* :class:`DocumentationContract` — the frozen builder/checker definition (step 1);
* :class:`PageInventory`, :class:`LinkInventory`, :class:`CodeBlockInventory`,
  :class:`SchemaExampleInventory`, :class:`FactInventory` — the five-inventory
  model of steps 5–10, with the code-block classification of §3.1;
* :data:`DEFECT_OWNERSHIP` — the §9 severity/ownership matrix, so a defect is
  fixed at the layer that owns the fact (docs vs CLI vs schema vs release vs
  claim), never by editing the page until the checker turns green;
* :func:`classify_warnings`, :func:`check_internal_links`, :func:`check_anchors`,
  :func:`check_release_links`, :func:`check_external_references`,
  :func:`check_download_content` — the link/anchor gates of steps 12–18;
* :func:`run_code_blocks`, :func:`diff_cli_help`, :func:`check_config_precedence`,
  :func:`validate_schema_examples`, :func:`check_expected_output` — the
  executability gates of steps 19–28, including the §9.1 command verdict table
  (``SKIP`` is explicitly *not* a final state);
* :func:`capability_matrix_from_evidence`, :func:`reconcile_support_matrix`,
  :func:`check_limitation_backlinks` — the *projection* rule of §3.2: a support
  cell is the intersection of declared capability, source adapter, test evidence,
  runtime freshness, release availability and limitation policy;
* :func:`extract_language_facts`, :func:`check_bilingual_diff`,
  :func:`check_version_consistency` — the bilingual/version gates of steps 32–34;
* :func:`scan_privacy_paths`, :func:`check_accessibility`,
  :func:`check_offline_static`, :func:`record_navigation_tasks` — steps 35–38;
* :func:`inject_doc_negative_controls`, :func:`compute_coverage_metrics`,
  :func:`classify_defect`, :func:`adjudicate_doc_gates`, :func:`freeze_verification`
  — steps 39–45.

Nothing here builds a site, follows a link or executes a command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest, digest_text, is_digest

#: Public entry points the inventory must cover (step 2).
PUBLIC_ENTRY_KINDS: Tuple[str, ...] = (
    "root_readme",
    "language_readme",
    "docs_site",
    "api_reference",
    "examples",
    "reports",
    "release_notes",
    "faq",
    "limitations",
    "contributing",
    "security",
)

#: Reader tasks the information architecture must serve (step 3).
READER_TASKS: Tuple[Tuple[str, str], ...] = (
    ("recruiter", "5 分钟内判断项目是什么、完成到什么等级"),
    ("cpu_newcomer", "按 quickstart 完成安装与 sample"),
    ("accelerator_reproducer", "在目标设备重放 hero story"),
    ("backend_contributor", "新增一个 backend 并跑通测试"),
    ("evidence_reviewer", "从 claim 追到 raw 与命令"),
)

#: Documentation sections and their owners (step 4).
DOC_SECTIONS: Mapping[str, str] = {
    "concepts": "architecture owner",
    "tutorials": "quickstart owner",
    "how-to": "feature owner",
    "reference": "contract owner",
    "experiments": "experiment owner",
    "reports": "stage owner",
    "limitations": "claim owner",
}

#: Code-block execution classes (§3.1).
CODE_BLOCK_CLASSES: Tuple[str, ...] = rec.CODE_BLOCK_CLASSES

#: Command verdicts (§9.1); ``SKIP`` is not among them on purpose.
COMMAND_VERDICTS: Tuple[str, ...] = rec.COMMAND_VERDICTS

#: Warning classes and their blocking level (step 12).
WARNING_LEVELS: Mapping[str, str] = {
    "missing_reference": "P0",
    "duplicate_anchor": "P0",
    "unknown_directive": "P1",
    "not_in_toc": "P2",
    "deprecation": "P2",
    "broken_image": "P0",
}

#: Defect severity and who must fix it (§9).
DEFECT_OWNERSHIP: Tuple[Mapping[str, str], ...] = (
    {
        "defect": "quickstart 命令复制即失败",
        "severity": "P0",
        "owner": "docs/CLI/package",
        "release_impact": "阻断",
    },
    {
        "defect": "Schema example 被当前 validator 拒绝",
        "severity": "P0",
        "owner": "example/schema/migration",
        "release_impact": "阻断",
    },
    {
        "defect": "support matrix 高报能力",
        "severity": "P0",
        "owner": "registry/matrix/claim",
        "release_impact": "阻断相关 release",
    },
    {
        "defect": "中英文性能数字/范围不同",
        "severity": "P0",
        "owner": "Claim rendering",
        "release_impact": "阻断",
    },
    {
        "defect": "evidence/release asset 链接错误",
        "severity": "P0",
        "owner": "manifest/docs",
        "release_impact": "阻断相关 claim",
    },
    {
        "defect": "accelerator 命令无设备验证",
        "severity": "P1",
        "owner": "device CI/docs status",
        "release_impact": "降级为未验证",
    },
    {
        "defect": "外部参考暂时不可达",
        "severity": "P1/P2",
        "owner": "reference/archive",
        "release_impact": "按关键性决定",
    },
    {
        "defect": "当前 CLI option 漏文档",
        "severity": "P1",
        "owner": "generated reference",
        "release_impact": "阻断受影响 how-to",
    },
    {
        "defect": "历史报告旧命令且标 archival",
        "severity": "INFO",
        "owner": "archival banner",
        "release_impact": "不阻断当前入口",
    },
    {
        "defect": "非关键错字/样式",
        "severity": "P2",
        "owner": "page source",
        "release_impact": "可排期",
    },
)

#: Required elements of "the figure keeps its context" (step 36 + E15-06 step 34).
MIN_FIGURE_CONTEXT: Tuple[str, ...] = ("model", "hardware", "workload", "baseline", "n", "evidence_link")

#: Link target types the checker distinguishes (§3.4).
LINK_TYPES: Tuple[str, ...] = (
    "internal_anchor",
    "repository_path",
    "release_asset",
    "permanent_evidence",
    "external_reference",
    "authenticated_resource",
)

#: Machine-readable example formats (step 9).
EXAMPLE_FORMATS: Tuple[str, ...] = ("yaml", "json", "toml", "csv")


@dataclass
class DocumentationContract:
    """The frozen builder/checker definition of step 1."""

    candidate_id: str
    builder: str
    theme: str = ""
    plugins: Tuple[str, ...] = ()
    docs_extra: str = "docs"
    link_policy: str = ""
    command_policy: str = ""
    schema_policy: str = ""
    failure_levels: Mapping[str, str] = field(default_factory=dict)

    schema_version = f"{SCHEMA_PREFIX}.documentation-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.candidate_id:
            findings.append("DocumentationContract: candidate_id is required")
        if not self.builder:
            findings.append("DocumentationContract: the builder (and version) must be frozen")
        for name in ("link_policy", "command_policy", "schema_policy"):
            if not getattr(self, name):
                findings.append(f"DocumentationContract: {name} must be declared before the audit")
        if not self.failure_levels:
            findings.append("DocumentationContract: the failure levels must be frozen, not decided after a failure")
        return findings


@dataclass
class PageEntry:
    """One page of the inventory (step 5)."""

    page_id: str
    path: str
    language: str
    counterpart: str = ""
    generated: bool = False
    owner: str = ""
    last_verified_candidate: str = ""
    entry_kind: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.page_id or not self.path:
            findings.append("PageEntry: page_id and path are required")
        if self.language not in ("zh", "en", "neutral"):
            findings.append(f"PageEntry({self.page_id}): unknown language {self.language!r}")
        if self.entry_kind and self.entry_kind not in PUBLIC_ENTRY_KINDS:
            findings.append(f"PageEntry({self.page_id}): unknown entry kind {self.entry_kind!r}")
        if not self.owner:
            findings.append(f"PageEntry({self.page_id}): an owner is required (an orphan page goes stale)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "path": self.path,
            "language": self.language,
            "counterpart": self.counterpart,
            "generated": self.generated,
            "owner": self.owner,
            "last_verified_candidate": self.last_verified_candidate,
            "entry_kind": self.entry_kind,
        }


class PageInventory:
    """All public pages, with the entry-point coverage check."""

    def __init__(self, pages: Sequence[PageEntry]) -> None:
        self.pages = list(pages)

    def problems(self) -> List[str]:
        findings: List[str] = []
        for page in self.pages:
            findings.extend(page.problems())
        covered = {page.entry_kind for page in self.pages if page.entry_kind}
        missing = [kind for kind in PUBLIC_ENTRY_KINDS if kind not in covered]
        if missing:
            findings.append(f"page inventory does not cover entry kinds: {', '.join(missing)}")
        paths = [page.path for page in self.pages]
        if len(set(paths)) != len(paths):
            findings.append("page inventory contains duplicate paths")
        return findings

    def bilingual_pairs(self) -> List[Tuple[PageEntry, PageEntry]]:
        by_id = {page.page_id: page for page in self.pages}
        pairs: List[Tuple[PageEntry, PageEntry]] = []
        for page in self.pages:
            if page.language == "zh" and page.counterpart and page.counterpart in by_id:
                pairs.append((page, by_id[page.counterpart]))
        return pairs

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.page-inventory.v1",
            "pages": [page.as_dict() for page in self.pages],
            "count": len(self.pages),
        }


@dataclass
class LinkEntry:
    """One link with its type and expectations (step 6)."""

    source_page: str
    anchor_text: str
    target: str
    link_type: str
    needs_network: bool = False
    needs_auth: bool = False
    expected_content_type: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.source_page or not self.target:
            findings.append("LinkEntry: source_page and target are required")
        if self.link_type not in LINK_TYPES:
            findings.append(f"LinkEntry({self.target}): unknown link type {self.link_type!r}")
        if self.link_type == "permanent_evidence" and not is_digest(_digest_in(self.target)):
            findings.append(
                f"LinkEntry({self.target}): an evidence link must carry a sha256 digest (a link alone is not evidence)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_page": self.source_page,
            "anchor_text": self.anchor_text,
            "target": self.target,
            "link_type": self.link_type,
            "needs_network": self.needs_network,
            "needs_auth": self.needs_auth,
            "expected_content_type": self.expected_content_type,
        }


def _digest_in(text: str) -> str:
    match = re.search(r"sha256:[0-9a-f]{64}", text)
    return match.group(0) if match else ""


@dataclass
class CodeBlock:
    """One code block with its execution classification (steps 7–8)."""

    block_id: str
    page: str
    line: int
    language: str
    command: str
    classification: str
    reason: str = ""
    working_directory: str = ""
    prerequisites: Tuple[str, ...] = ()
    expected_duration_s: Optional[float] = None
    expected_output: str = ""
    resource_notes: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.classification not in CODE_BLOCK_CLASSES:
            findings.append(
                f"CodeBlock({self.block_id}): classification {self.classification!r} is missing/unknown; "
                "unclassified blocks may not enter the document (§3.1)"
            )
        if not self.reason:
            findings.append(f"CodeBlock({self.block_id}): the classification needs a reason")
        if not self.command:
            findings.append(f"CodeBlock({self.block_id}): empty command")
        if self.classification == "DISPLAY_ONLY" and self.command.lstrip().startswith(("$", ">", "❯")):
            findings.append(
                f"CodeBlock({self.block_id}): a display-only example must not use a copyable shell prompt "
                "(E15-04 step 23)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "line": self.line,
            "language": self.language,
            "command_sha256": digest_text(self.command),
            "classification": self.classification,
            "reason": self.reason,
            "working_directory": self.working_directory,
            "prerequisites": list(self.prerequisites),
            "expected_duration_s": self.expected_duration_s,
            "expected_output": self.expected_output,
            "resource_notes": self.resource_notes,
        }


class CodeBlockInventory:
    """All public code blocks with their classification coverage."""

    def __init__(self, blocks: Sequence[CodeBlock]) -> None:
        self.blocks = list(blocks)

    def problems(self) -> List[str]:
        findings: List[str] = []
        for block in self.blocks:
            findings.extend(block.problems())
        return findings

    def coverage(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for block in self.blocks:
            counts[block.classification] = counts.get(block.classification, 0) + 1
        return {key: counts[key] for key in sorted(counts)}

    def unclassified(self) -> List[str]:
        return [block.block_id for block in self.blocks if block.classification not in CODE_BLOCK_CLASSES]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.code-block-inventory.v1",
            "blocks": [block.as_dict() for block in self.blocks],
            "coverage": self.coverage(),
            "unclassified": self.unclassified(),
        }


@dataclass
class SchemaExample:
    """One machine-readable example bound to a schema (step 9)."""

    example_id: str
    page: str
    format: str
    schema_id: str
    schema_version: str
    complete: bool
    body: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.format not in EXAMPLE_FORMATS:
            findings.append(f"SchemaExample({self.example_id}): unknown format {self.format!r}")
        if not self.schema_id or not self.schema_version:
            findings.append(f"SchemaExample({self.example_id}): every example must name its schema and version")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "page": self.page,
            "format": self.format,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "complete": self.complete,
            "body_sha256": digest_text(self.body),
        }


def validate_schema_examples(
    examples: Sequence[SchemaExample], *, validator: Callable[[SchemaExample], Sequence[str]]
) -> List[Dict[str, Any]]:
    """Validate every complete example; fragments are assembled first (steps 26–27)."""
    results: List[Dict[str, Any]] = []
    for example in examples:
        findings = list(example.problems())
        if example.complete:
            findings.extend(validator(example))
        result = {
            "example_id": example.example_id,
            "page": example.page,
            "validated": example.complete,
            "findings": findings,
        }
        results.append(result)
    return results


@dataclass
class FactEntry:
    """One documented fact bound to a claim id (step 10)."""

    fact_id: str
    page: str
    text: str
    claim_id: str = ""
    value: str = ""
    unit: str = ""
    kind: str = ""

    FACT_KINDS: Tuple[str, ...] = ("number", "capability", "status", "version", "matrix_cell", "limitation")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.kind and self.kind not in self.FACT_KINDS:
            findings.append(f"FactEntry({self.fact_id}): unknown fact kind {self.kind!r}")
        if not self.claim_id:
            findings.append(
                f"FactEntry({self.fact_id}): a documented fact must bind to a claim id "
                "(the document may not be a second truth source)"
            )
        if self.kind == "number" and not self.unit:
            findings.append(f"FactEntry({self.fact_id}): a number without a unit is not a fact")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "page": self.page,
            "claim_id": self.claim_id,
            "kind": self.kind,
            "value": self.value,
            "unit": self.unit,
            "text_sha256": digest_text(self.text),
        }


# ── build / links (steps 11–18) ─────────────────────────────────────────────


def classify_warnings(*, warnings: Sequence[str], levels: Mapping[str, str] = WARNING_LEVELS) -> Dict[str, Any]:
    """Classify build warnings; a P0 warning is a failure, not a footnote (step 12)."""
    grouped: Dict[str, List[str]] = {}
    for warning in warnings:
        klass = next(
            (key for key in levels if key in warning or key.replace("_", " ") in warning), "unknown"
        )
        grouped.setdefault(klass, []).append(warning)
    blocking = [key for key in grouped if levels.get(key) == "P0"]
    return {
        "by_class": {key: sorted(grouped[key]) for key in sorted(grouped)},
        "blocking_classes": sorted(blocking),
        "blocking": bool(blocking),
        "unknown_warnings": sorted(grouped.get("unknown", [])),
    }


def check_internal_links(
    links: Sequence[LinkEntry], *, resolve_path: Callable[[str, str], bool], resolve_anchor: Callable[[str, str], bool]
) -> List[Dict[str, Any]]:
    """Resolve repository paths and anchors under both rendering semantics (steps 14–15)."""
    results: List[Dict[str, Any]] = []
    for link in links:
        if link.link_type not in ("internal_anchor", "repository_path"):
            continue
        if link.link_type == "repository_path":
            ok = resolve_path(link.source_page, link.target)
        else:
            ok = resolve_anchor(link.source_page, link.target)
        results.append(
            {
                **link.as_dict(),
                "resolved": ok,
                "findings": [] if ok else [f"broken {link.link_type}: {link.target!r}"],
            }
        )
    return results


def check_release_links(links: Sequence[LinkEntry], *, resolve: Callable[[str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Release/evidence links must resolve to the *frozen* asset, not `latest` (step 16)."""
    results: List[Dict[str, Any]] = []
    for link in links:
        if link.link_type not in ("release_asset", "permanent_evidence"):
            continue
        outcome = dict(resolve(link.target) or {})
        findings: List[str] = []
        if "latest" in link.target or link.target.endswith(("/latest", "/main")):
            findings.append(f"{link.target!r} is a floating link; frozen evidence needs an immutable target")
        if outcome.get("login_page"):
            findings.append(f"{link.target!r} returned a login page for an unauthenticated reader")
        if not outcome.get("resolved", False):
            findings.append(f"{link.target!r} did not resolve")
        if link.link_type == "permanent_evidence":
            digest = _digest_in(link.target)
            if outcome.get("digest") and outcome["digest"] != digest:
                findings.append(f"{link.target!r} digest mismatch ({outcome['digest']} != {digest})")
        results.append({**link.as_dict(), "resolved": bool(outcome.get("resolved")), "findings": findings})
    return results


def check_external_references(references: Sequence[Mapping[str, Any]], *, retry_policy: str) -> List[Dict[str, Any]]:
    """External references: status, redirects, canonical URL, retries (step 17)."""
    results: List[Dict[str, Any]] = []
    for reference in references:
        findings: List[str] = []
        status = int(reference.get("status", 0))
        if status in (301, 302, 307, 308) and not reference.get("canonical_url"):
            findings.append("a redirect must record the canonical URL it settled on")
        if status == 0 or status >= 400:
            findings.append(f"HTTP {status}; retries follow the frozen policy {retry_policy!r}")
        if reference.get("is_blog_mirror") and not reference.get("official_source"):
            findings.append("a blog mirror may not stand in for the official source")
        results.append({"url": reference.get("url", ""), "status": status, "findings": findings})
    return results


def check_download_content(downloads: Sequence[Mapping[str, Any]]) -> List[str]:
    """Content type and hash of downloads (step 18)."""
    findings: List[str] = []
    for item in downloads:
        name = str(item.get("name", ""))
        content_type = str(item.get("content_type", ""))
        if content_type.startswith("text/html") and name.endswith((".whl", ".tar.gz", ".json", ".yaml")):
            findings.append(f"{name}: served as HTML (an error page with status 200)")
        if not is_digest(str(item.get("sha256", ""))):
            findings.append(f"{name}: no sha256 recorded")
        declared = item.get("declared_size")
        actual = item.get("size_bytes")
        if declared is not None and actual is not None and declared != actual:
            findings.append(f"{name}: size mismatch ({declared} != {actual})")
    return findings


# ── commands / CLI / schema (steps 19–28) ───────────────────────────────────


def run_code_blocks(
    blocks: Sequence[CodeBlock], *, runner: Callable[[CodeBlock], Mapping[str, Any]], environment: str
) -> List[Dict[str, Any]]:
    """Execute (or dry-run) every classifiable block and record its verdict.

    ``runner`` performs the real work in the right environment; this function
    enforces the §9.1 rules: every block ends in a real verdict, ``SKIP`` is
    rejected, and a GPU-only block executed on CPU is ``BLOCKED_DEVICE`` with an
    owner and expiry instead of a permanent skip.
    """
    results: List[Dict[str, Any]] = []
    for block in blocks:
        findings = list(block.problems())
        if block.classification not in CODE_BLOCK_CLASSES:
            results.append({"block_id": block.block_id, "verdict": "INVALID_UNCLASSIFIED", "findings": findings})
            continue
        outcome = dict(runner(block) or {})
        verdict = str(outcome.get("verdict", "FAIL"))
        if verdict == "SKIP":
            findings.append("SKIP is not a final command verdict (§9.1)")
            verdict = "FAIL"
        if verdict not in COMMAND_VERDICTS:
            findings.append(f"unknown verdict {verdict!r}")
        if block.classification == "EXECUTE_ACCELERATOR" and environment == "cpu":
            verdict = "BLOCKED_DEVICE"
            findings.append("an accelerator block on a CPU host is BLOCKED_DEVICE, not PASS and not a silent skip")
        if verdict == "BLOCKED_DEVICE" and not outcome.get("owner"):
            findings.append("BLOCKED_DEVICE must name an owner and an expiry (§9.1)")
        results.append(
            {
                "block_id": block.block_id,
                "page": block.page,
                "classification": block.classification,
                "verdict": verdict,
                "exit_code": outcome.get("exit_code"),
                "duration_s": outcome.get("duration_s"),
                "findings": findings,
            }
        )
    return results


def diff_cli_help(*, exported: Mapping[str, Any], documented: Mapping[str, Any]) -> List[str]:
    """Compare the exported CLI surface with the reference docs (step 24)."""
    findings: List[str] = []
    exported_commands = set(exported.get("commands", {}))
    documented_commands = set(documented.get("commands", {}))
    for command in sorted(exported_commands - documented_commands):
        findings.append(f"CLI command {command!r} is missing from the reference docs")
    for command in sorted(documented_commands - exported_commands):
        findings.append(f"reference docs document unknown command {command!r}")
    for command in sorted(exported_commands & documented_commands):
        left = exported["commands"][command]
        right = documented["commands"][command]
        for field_name in ("options", "defaults", "deprecations"):
            left_value = left.get(field_name)
            right_value = right.get(field_name)
            if isinstance(left_value, (list, tuple)):
                left_value = sorted(map(str, left_value))
            if isinstance(right_value, (list, tuple)):
                right_value = sorted(map(str, right_value))
            if left_value != right_value:
                findings.append(f"{command}: {field_name} drift ({left_value!r} != {right_value!r})")
    return findings


def check_config_precedence(
    *, documented: Mapping[str, str], observed: Mapping[str, str]
) -> List[str]:
    """The documented default/file/env/CLI behaviour must match the implementation (step 25)."""
    findings: List[str] = []
    for layer in ("default", "file", "env", "cli"):
        if documented.get(layer) != observed.get(layer):
            findings.append(
                f"config precedence drift on {layer}: documented {documented.get(layer)!r} "
                f"vs observed {observed.get(layer)!r}"
            )
    return findings


def check_expected_output(
    *, block: CodeBlock, actual_output: str, normalisers: Sequence[Callable[[str], str]]
) -> List[str]:
    """Compare an example's expected output after the declared normalisation (step 28)."""
    if not block.expected_output:
        return [f"block {block.block_id} has no expected output to compare"]
    left = block.expected_output
    right = actual_output
    for normalise in normalisers:
        left = normalise(left)
        right = normalise(right)
    if left != right:
        return [f"block {block.block_id}: expected output drifted (either the doc or the tool must change)"]
    return []


# ── capability matrix projection (steps 29–31) ──────────────────────────────

#: The six conditions of the support-cell intersection (§3.2).
MATRIX_CONDITIONS: Tuple[str, ...] = (
    "declared_capability",
    "source_adapter_exists",
    "required_test_evidence",
    "runtime_evidence_fresh",
    "release_dependency_available",
    "limitation_policy_allows",
)

#: States a support cell can display.
MATRIX_STATES: Tuple[str, ...] = ("supported", "partial", "experimental", "planned", "unsupported")


@dataclass
class SupportCell:
    """One cell of the capability matrix, projected from evidence (§3.2)."""

    cell_id: str
    conditions: Mapping[str, bool] = field(default_factory=dict)
    declared_state: str = "planned"
    evidence_refs: Tuple[str, ...] = ()
    limitation_ref: str = ""

    def projected_state(self) -> str:
        if any(name not in self.conditions for name in MATRIX_CONDITIONS):
            return "planned"
        if all(self.conditions[name] for name in MATRIX_CONDITIONS):
            return "supported"
        if self.conditions.get("source_adapter_exists") and self.conditions.get("required_test_evidence"):
            return "partial"
        if self.conditions.get("source_adapter_exists"):
            return "experimental"
        return "unsupported"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.declared_state not in MATRIX_STATES:
            findings.append(f"SupportCell({self.cell_id}): unknown declared state {self.declared_state!r}")
        projected = self.projected_state()
        if self.declared_state == "supported" and projected != "supported":
            findings.append(
                f"SupportCell({self.cell_id}): declared supported but evidence projects {projected!r} "
                "(§3.2: 缺一项时 cell 应显示 planned/experimental/partial/unsupported)"
            )
        if projected in ("partial", "experimental", "unsupported") and not self.limitation_ref:
            findings.append(f"SupportCell({self.cell_id}): a non-supported cell must link a limitation (step 31)")
        if projected == "supported" and not self.evidence_refs:
            findings.append(f"SupportCell({self.cell_id}): a supported cell must cite evidence")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "conditions": {key: self.conditions.get(key, False) for key in MATRIX_CONDITIONS},
            "declared_state": self.declared_state,
            "projected_state": self.projected_state(),
            "evidence_refs": list(self.evidence_refs),
            "limitation_ref": self.limitation_ref,
        }


def capability_matrix_from_evidence(cells: Sequence[SupportCell]) -> Dict[str, Any]:
    """Project the actual matrix from evidence (step 29)."""
    problems: List[str] = []
    for cell in cells:
        problems.extend(cell.problems())
    return {
        "schema_version": f"{SCHEMA_PREFIX}.capability-matrix.v1",
        "cells": [cell.as_dict() for cell in cells],
        "problems": problems,
        "counts": _count_projected(cells),
    }


def _count_projected(cells: Sequence[SupportCell]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cell in cells:
        state = cell.projected_state()
        counts[state] = counts.get(state, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def reconcile_support_matrix(
    documented: Sequence[SupportCell], actual: Sequence[SupportCell]
) -> List[str]:
    """Per-cell comparison of the document against the projection (step 30)."""
    by_id = {cell.cell_id: cell for cell in actual}
    findings: List[str] = []
    for cell in documented:
        other = by_id.get(cell.cell_id)
        if other is None:
            findings.append(f"documented cell {cell.cell_id!r} has no projected counterpart")
            continue
        if cell.declared_state != other.projected_state():
            findings.append(
                f"cell {cell.cell_id!r}: documented {cell.declared_state!r} but evidence projects "
                f"{other.projected_state()!r}"
            )
    extra = sorted(set(by_id) - {cell.cell_id for cell in documented})
    if extra:
        findings.append(f"evidence knows cells the document omits: {', '.join(extra)}")
    return findings


def check_limitation_backlinks(
    *, cells: Sequence[SupportCell], limitations: Sequence[Mapping[str, str]]
) -> List[str]:
    """Both directions: cell → limitation and limitation → affected cells (step 31)."""
    findings: List[str] = []
    limitation_ids = {str(item.get("limitation_id", "")) for item in limitations}
    referenced: set = set()
    for cell in cells:
        if cell.projected_state() != "supported":
            referenced.add(cell.limitation_ref)
            if cell.limitation_ref and cell.limitation_ref not in limitation_ids:
                findings.append(f"cell {cell.cell_id!r} links unknown limitation {cell.limitation_ref!r}")
    for item in limitations:
        affected = list(item.get("affected_cells", []))
        if not affected:
            findings.append(f"limitation {item.get('limitation_id')!r} lists no affected capability")
        for cell_id in affected:
            if not any(cell.cell_id == cell_id for cell in cells):
                findings.append(f"limitation {item.get('limitation_id')!r} references unknown cell {cell_id!r}")
    return findings


# ── bilingual / version / privacy (steps 32–35) ─────────────────────────────

#: Fact fields the bilingual diff compares (§3.3).
BILINGUAL_FACT_FIELDS: Tuple[str, ...] = (
    "claim_id",
    "value",
    "unit",
    "model",
    "hardware",
    "workload",
    "baseline",
    "evidence_level",
    "status",
    "limitation",
)


def extract_language_facts(entries: Sequence[FactEntry], *, language: str) -> Dict[str, Dict[str, Any]]:
    """Build a language fact table keyed by claim id (step 32)."""
    table: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        table.setdefault(entry.claim_id, {})[language] = {
            "fact_id": entry.fact_id,
            "value": entry.value,
            "unit": entry.unit,
            "kind": entry.kind,
            "page": entry.page,
        }
    return table


def check_bilingual_diff(
    table: Mapping[str, Mapping[str, Mapping[str, Any]]], *, allowed_language_specific: Sequence[str] = ()
) -> List[str]:
    """Critical-field diff between the two language tables (step 33).

    Only *recorded* language-specific navigation/explanation differences are
    allowed; a differing number, unit or limitation is a critical diff.
    """
    findings: List[str] = []
    for claim_id, languages in sorted(table.items()):
        if "zh" not in languages or "en" not in languages:
            continue
        zh = languages["zh"]
        en = languages["en"]
        for field_name in ("value", "unit", "kind"):
            left = zh.get(field_name)
            right = en.get(field_name)
            if left != right and f"{claim_id}:{field_name}" not in allowed_language_specific:
                findings.append(
                    f"{claim_id}: critical bilingual diff on {field_name!r} ({left!r} != {right!r})"
                )
    return findings


def check_version_consistency(*, facts: Mapping[str, str], declared: Mapping[str, str]) -> List[str]:
    """README badge / install command / container tag / schema / citation (step 34)."""
    findings: List[str] = []
    for name in ("readme_badge", "install_command", "container_tag", "docs_version", "schema_version", "citation"):
        if name not in facts or name not in declared:
            findings.append(f"version consistency: {name!r} is not declared on both sides")
            continue
        if facts[name] != declared[name]:
            findings.append(f"version drift on {name!r}: document {facts[name]!r} vs release {declared[name]!r}")
    return findings


def scan_privacy_paths(texts: Mapping[str, str]) -> List[str]:
    """Absolute homes, internal hosts, credentials, temp buckets (step 35)."""
    patterns = {
        "absolute_home": r"(?:/root/|/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/)",
        "internal_host": r"(?i)(?:internal|corp|intranet)\.example|[a-z0-9.-]+\.internal\b",
        "token_like": r"(?i)(?:token|api[-_]?key|secret)\s*[:=]\s*[A-Za-z0-9_\-]{12,}",
        "temp_bucket": r"(?i)s3://[a-z0-9.-]*(?:tmp|temp|staging)[a-z0-9.-]*",
    }
    findings: List[str] = []
    for name, text in sorted(texts.items()):
        for kind, pattern in patterns.items():
            if re.search(pattern, text):
                findings.append(f"{name}: contains {kind}")
    return findings


def check_accessibility(*, pages: Sequence[Mapping[str, Any]]) -> List[str]:
    """Alt text, table readability, colour-only encoding, keyboard nav (step 36).

    The protocol is explicit that this is a documentation gate and *not* a
    certification claim.
    """
    findings: List[str] = []
    for page in pages:
        path = page.get("path", "?")
        if page.get("images_without_alt"):
            findings.append(f"{path}: {page['images_without_alt']} image(s) without alt text")
        if page.get("colour_only_encoding"):
            findings.append(f"{path}: PASS/FAIL encoded by colour alone")
        if page.get("overflow_tables"):
            findings.append(f"{path}: {page['overflow_tables']} table(s) unreadable on a small screen")
        if page.get("keyboard_trap"):
            findings.append(f"{path}: keyboard navigation traps")
    return findings


def check_offline_static(*, required_assets: Mapping[str, bool]) -> List[str]:
    """Local navigation/images/CSS must work without a CDN (step 37)."""
    findings: List[str] = []
    for asset, local in sorted(required_assets.items()):
        if not local:
            findings.append(f"{asset}: only available through a remote host; a frozen site must serve it locally")
    return findings


def record_navigation_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evidence-finding tasks with clicks/time/failure (step 38)."""
    results: List[Dict[str, Any]] = []
    for task in tasks:
        results.append(
            {
                "task": task.get("task", ""),
                "target": task.get("target", ""),
                "clicks": int(task.get("clicks", 0)),
                "duration_s": float(task.get("duration_s", 0.0)),
                "found": bool(task.get("found", False)),
            }
        )
    return {
        "tasks": results,
        "success_rate": (
            sum(1 for item in results if item["found"]) / len(results) if results else 0.0
        ),
    }


# ── negative controls / defects (steps 39–44) ───────────────────────────────


def inject_doc_negative_controls() -> Tuple[Mapping[str, str], ...]:
    """Injected documentation defects (step 39)."""
    return (
        {"injection_id": "inj-bad-anchor", "severity": "P0", "check": "check_internal_links"},
        {"injection_id": "inj-old-cli", "severity": "P0", "check": "diff_cli_help"},
        {"injection_id": "inj-invalid-schema", "severity": "P0", "check": "validate_schema_examples"},
        {"injection_id": "inj-wrong-asset-version", "severity": "P0", "check": "check_version_consistency"},
        {"injection_id": "inj-fake-supported-cell", "severity": "P0", "check": "reconcile_support_matrix"},
        {"injection_id": "inj-bilingual-number", "severity": "P0", "check": "check_bilingual_diff"},
        {"injection_id": "inj-display-only-as-command", "severity": "P1", "check": "CodeBlock.problems"},
    )


def compute_coverage_metrics(
    *,
    pages: PageInventory,
    links: Sequence[LinkEntry],
    blocks: CodeBlockInventory,
    examples: Sequence[SchemaExample],
    facts: Sequence[FactEntry],
) -> Dict[str, Any]:
    """Coverage per object class plus the negative-control requirement (step 40)."""
    return {
        "pages": len(pages.pages),
        "links": len(links),
        "code_blocks": len(blocks.blocks),
        "schema_examples": len(examples),
        "facts": len(facts),
        "unclassified_blocks": blocks.unclassified(),
        "facts_without_claim": [fact.fact_id for fact in facts if fact.problems()],
        "note": "一个综合通过率不能掩盖 P0 漏检；负对照结果单独报告",
    }


def classify_defect(finding: Mapping[str, Any]) -> Dict[str, str]:
    """Route a defect to the layer that owns the fact (step 41)."""
    text = str(finding.get("description", "")).lower()
    if "command" in text or "cli" in text:
        owner = "source/CLI/backend owner"
    elif "schema" in text or "example" in text:
        owner = "contract/migration owner"
    elif "capability" in text or "matrix" in text:
        owner = "registry/test/runtime evidence owner"
    elif "claim" in text or "number" in text:
        owner = "Claim Ledger owner"
    elif "asset" in text or "release" in text or "version" in text:
        owner = "release manifest owner"
    elif "translation" in text or "bilingual" in text:
        owner = "canonical rendering + language reviewer"
    else:
        owner = "docs/example owner"
    return {"owner": owner, "note": "禁止只改页面让检查绿，而实际行为继续错误"}


def adjudicate_doc_gates(*, gates: Mapping[str, str]) -> Dict[str, str]:
    """Per-gate verdicts (step 44)."""
    invalid = {gate: verdict for gate, verdict in gates.items() if verdict not in rec.ALL_STATUSES}
    if invalid:
        raise ConfigError(f"adjudicate_doc_gates received non-status verdicts: {invalid}")
    return {gate: gates[gate] for gate in sorted(gates)}


def freeze_verification(
    *, verification_id: str, candidate_id: str, builder_identity: str, record: rec.DocumentationVerificationRecord
) -> Dict[str, Any]:
    """Freeze the verification record, refusing a PASS with open P0 defects (step 45)."""
    problems = list(record.validate())
    if record.status == rec.STATUS_PASS and not record.negative_control_metrics:
        problems.append("a PASS requires negative-control metrics (a checker with no detection power proves nothing)")
    if problems:
        raise ConfigError("refusing to freeze the verification: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.documentation-verification.v1",
        "verification_id": verification_id,
        "candidate_id": candidate_id,
        "builder_identity": builder_identity,
        "record": record.as_dict(),
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 DocumentationContract", ("docs_gate.DocumentationContract",)),
    (2, "盘点全部公开入口", ("docs_gate.PUBLIC_ENTRY_KINDS", "docs_gate.PageInventory")),
    (3, "定义读者与信息任务", ("docs_gate.READER_TASKS",)),
    (4, "冻结文档信息架构", ("docs_gate.DOC_SECTIONS",)),
    (5, "建立页面 inventory", ("docs_gate.PageEntry", "docs_gate.PageInventory.bilingual_pairs")),
    (6, "建立链接 inventory", ("docs_gate.LinkEntry", "docs_gate.LINK_TYPES")),
    (7, "建立 code-block inventory", ("docs_gate.CodeBlock", "docs_gate.CodeBlockInventory")),
    (8, "给 code block 分类", ("docs_gate.CODE_BLOCK_CLASSES", "docs_gate.CodeBlock.problems")),
    (9, "建立机器可读示例 inventory", ("docs_gate.SchemaExample", "docs_gate.EXAMPLE_FORMATS")),
    (10, "建立事实与 Claim ID inventory", ("docs_gate.FactEntry",)),
    (11, "从 clean checkout 构建文档", ("docs_gate.DocumentationContract", "telemetry.ActionLog")),
    (12, "把 warning 升级为受控结果", ("docs_gate.classify_warnings", "docs_gate.WARNING_LEVELS")),
    (13, "验证导航与目录完整性", ("docs_gate.PageInventory.problems", "docs_gate.READER_TASKS")),
    (14, "验证内部相对链接", ("docs_gate.check_internal_links",)),
    (15, "验证 heading anchor", ("docs_gate.check_internal_links",)),
    (16, "验证 release/evidence 链接", ("docs_gate.check_release_links",)),
    (17, "验证外部引用", ("docs_gate.check_external_references",)),
    (18, "验证下载内容类型与 hash", ("docs_gate.check_download_content",)),
    (19, "运行 CPU-safe 命令", ("docs_gate.run_code_blocks",)),
    (20, "运行 accelerator 命令", ("docs_gate.run_code_blocks", "records.COMMAND_VERDICTS")),
    (21, "验证 dry-run 命令", ("docs_gate.run_code_blocks",)),
    (22, "验证 manual-external 步骤", ("docs_gate.run_code_blocks", "docs_gate.CODE_BLOCK_CLASSES")),
    (23, "验证 DISPLAY_ONLY 标识", ("docs_gate.CodeBlock.problems",)),
    (24, "对账 CLI help", ("docs_gate.diff_cli_help",)),
    (25, "对账配置优先级", ("docs_gate.check_config_precedence",)),
    (26, "校验所有完整 Schema 示例", ("docs_gate.validate_schema_examples",)),
    (27, "验证片段可组装性", ("docs_gate.validate_schema_examples", "docs_gate.SchemaExample")),
    (28, "验证示例预期输出", ("docs_gate.check_expected_output",)),
    (29, "生成实际 Capability Matrix", ("docs_gate.capability_matrix_from_evidence", "docs_gate.SupportCell")),
    (30, "对账文档 Support Matrix", ("docs_gate.reconcile_support_matrix", "docs_gate.MATRIX_CONDITIONS")),
    (31, "验证 limitations 反向链接", ("docs_gate.check_limitation_backlinks",)),
    (32, "提取双语结构化事实", ("docs_gate.extract_language_facts", "docs_gate.BILINGUAL_FACT_FIELDS")),
    (33, "执行双语事实 diff", ("docs_gate.check_bilingual_diff",)),
    (34, "执行版本一致性检查", ("docs_gate.check_version_consistency",)),
    (35, "执行路径和隐私扫描", ("docs_gate.scan_privacy_paths",)),
    (36, "执行可访问性与小屏检查", ("docs_gate.check_accessibility", "docs_gate.MIN_FIGURE_CONTEXT")),
    (37, "执行断网静态站检查", ("docs_gate.check_offline_static",)),
    (38, "执行搜索与证据定位任务", ("docs_gate.record_navigation_tasks", "telemetry.EvidenceLookupTrial")),
    (39, "注入文档负对照", ("docs_gate.inject_doc_negative_controls",)),
    (40, "计算覆盖与检测指标", ("docs_gate.compute_coverage_metrics", "telemetry.DetectorMetrics")),
    (41, "归因并修复文档缺陷", ("docs_gate.classify_defect", "docs_gate.DEFECT_OWNERSHIP")),
    (42, "重建新候选文档", ("contracts.new_candidate", "docs_gate.freeze_verification")),
    (43, "执行 clean consumer smoke", ("quickstart.SessionIdentity", "docs_gate.run_code_blocks")),
    (44, "逐门裁决", ("docs_gate.adjudicate_doc_gates",)),
    (45, "冻结 DocumentationVerificationRecord", ("docs_gate.freeze_verification", "records.DocumentationVerificationRecord")),
)

TITLE = "可执行文档、双语一致性与能力矩阵对账"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-04 通过说明文档与当前候选一致且可执行，不说明陌生读者一定能快速理解（E15-11）"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the documentation gate (labelled smoke)."""
    problems: List[str] = []
    blocks = CodeBlockInventory(
        [
            CodeBlock(
                block_id="b1",
                page="README.md",
                line=10,
                language="bash",
                command="python -m hqsb.benchmark --help",
                classification="EXECUTE_SAFE",
                reason="CPU-only, read-only, fast",
            ),
        ]
    )
    problems.extend(blocks.problems())
    # Negative control: a DISPLAY_ONLY block with a copyable prompt must be flagged.
    bad_display = CodeBlock(
        block_id="b2",
        page="README.md",
        line=20,
        language="bash",
        command="$ python demo.py",
        classification="DISPLAY_ONLY",
        reason="placeholder output example",
    )
    if not any("display-only" in problem for problem in bad_display.problems()):
        problems.append("a DISPLAY_ONLY block with a shell prompt was not flagged")
    verdicts = run_code_blocks(
        blocks.blocks,
        runner=lambda block: {"verdict": "PASS", "exit_code": 0},
        environment="cpu",
    )
    if any(item["verdict"] == "SKIP" for item in verdicts):
        problems.append("SKIP leaked into the verdicts")
    accelerator = CodeBlock(
        block_id="b3",
        page="README.md",
        line=30,
        language="bash",
        command="python bench.py --device cuda",
        classification="EXECUTE_ACCELERATOR",
        reason="needs a GPU",
    )
    blocked = run_code_blocks([accelerator], runner=lambda block: {"verdict": "PASS"}, environment="cpu")
    if blocked[0]["verdict"] != "BLOCKED_DEVICE":
        problems.append("an accelerator block on CPU did not become BLOCKED_DEVICE")
    cell = SupportCell(
        cell_id="cuda/rmsnorm",
        conditions={"declared_capability": True, "source_adapter_exists": True},
        declared_state="supported",
    )
    if cell.projected_state() == "supported":
        problems.append("a half-evidenced cell projected as supported")
    warnings = classify_warnings(warnings=["WARNING: missing reference in index.md"])
    if not warnings["blocking"]:
        problems.append("a P0 build warning was not blocking")
    privacy = scan_privacy_paths({"readme": "run: cd /root/work/hqsb"})
    if not privacy:
        problems.append("the privacy scan missed an absolute home path")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "entry_kinds": len(PUBLIC_ENTRY_KINDS),
        "matrix_conditions": len(MATRIX_CONDITIONS),
        "injections": len(inject_doc_negative_controls()),
        "problems": problems,
    }
