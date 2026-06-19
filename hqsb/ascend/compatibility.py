"""CompatibilityManifest: the S09 environment identity card (E09-01 §4, steps 10/11/25/26).

Every later S09 experiment must cite this manifest's canonical hash *and* keep
the readable fields, because "two hashes differ" is not diagnosable — the
field-level diff is (E09-01 §1).

Four properties are enforced here rather than left to convention:

1. **A missing field is not an empty string.**  Anything a tool could not read
   is the literal :data:`UNAVAILABLE` plus a reason, so a report cannot imply
   a value was collected.
2. **Canonicalization is the project's**, reused from
   :mod:`hqsb.core.fingerprint` — one key order, one encoding, one null rule.
3. **Fields that may legitimately differ between two runs of the same locked
   environment** (serial numbers, timestamps, health readings) are declared in
   :data:`VOLATILE_FIELDS`; the diff marks them ``VOLATILE`` instead of
   pretending the environments are bit-identical (E09-01 §9).
4. **Secrets never reach the artifact.**  :func:`redact_text` and
   :func:`redaction_audit` run over every string before hashing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.core.fingerprint import canonical_json, sha256_hex

SCHEMA_VERSION = "1.0.0"

#: The literal used when a tool could not provide a value (E09-01 step 5).
UNAVAILABLE = "UNAVAILABLE"

#: A framework that is installed but not the chosen integration path must be
#: labelled, not silently omitted (E09-01 step 7).
NOT_SELECTED = "NOT_SELECTED"

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_MAJOR = "MAJOR"
SEVERITY_MINOR = "MINOR"
SEVERITY_VOLATILE = "VOLATILE"

DIFF_SEVERITIES: Tuple[str, ...] = (
    SEVERITY_CRITICAL,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
    SEVERITY_VOLATILE,
)

#: Top-level sections of the manifest (E09-01 §4.1-§4.3 plus project identity).
SECTIONS: Tuple[str, ...] = (
    "hardware",
    "ascend_stack",
    "framework",
    "project",
    "official_sources",
    "evidence",
)

#: Required keys per section.  A manifest missing one is rejected, not patched.
REQUIRED_KEYS: Mapping[str, Tuple[str, ...]] = {
    "hardware": (
        "host_id",
        "board_id",
        "chip_sku",
        "soc_version",
        "device_count",
        "logical_to_physical",
        "visible_device_policy",
        "cpu_arch",
        "os",
        "kernel",
        "container_runtime",
        "image_digest",
        "health",
    ),
    "ascend_stack": (
        "firmware_version",
        "firmware_components",
        "driver_version",
        "driver_install_mode",
        "driver_loaded_modules",
        "cann_toolkit_version",
        "cann_runtime_version",
        "cann_compiler_version",
        "cann_ops_package_version",
        "cann_kernel_package_version",
        "set_env_source",
        "library_resolution",
        "compiler_path",
        "msprof_version",
        "msprof_metric_sets",
        "install_roots",
    ),
    "framework": (
        "python_version",
        "pytorch_version",
        "torch_npu_version",
        "mindspore_version",
        "atb_version",
        "mindie_version",
        "package_hashes",
        "cpp_abi",
        "compile_flags",
        "selected_integration_path",
    ),
    "project": (
        "git_commit",
        "git_dirty",
        "source_patch_hash",
        "model_artifact_hash",
        "operator_spec_hash",
        "quant_artifact_hash",
    ),
    "official_sources": (),
    "evidence": (),
}

#: Fields that may differ between two runs of one locked environment.  The diff
#: reports them as ``VOLATILE`` — reproducible, but *not* bit-identical.
VOLATILE_FIELDS: Tuple[str, ...] = (
    "/hardware/host_id",
    "/hardware/board_id",
    "/hardware/health",
    "/ascend_stack/library_resolution",
    "/ascend_stack/install_roots",
    "/framework/package_hashes",
    "/collected_at",
)

#: Fields whose change means the environment is a *different* environment: the
#: ABI identity of the vertical stack (E09-01 §3).
CRITICAL_FIELDS: Tuple[str, ...] = (
    "/hardware/chip_sku",
    "/hardware/soc_version",
    "/hardware/cpu_arch",
    "/ascend_stack/firmware_version",
    "/ascend_stack/driver_version",
    "/ascend_stack/cann_toolkit_version",
    "/ascend_stack/cann_runtime_version",
    "/ascend_stack/cann_compiler_version",
    "/ascend_stack/cann_ops_package_version",
    "/ascend_stack/cann_kernel_package_version",
    "/framework/pytorch_version",
    "/framework/torch_npu_version",
    "/framework/cpp_abi",
    "/framework/selected_integration_path",
    "/project/git_commit",
    "/project/operator_spec_hash",
    "/project/model_artifact_hash",
)

#: Integration paths E09-05 §1 allows as the single main path.
INTEGRATION_PATHS: Tuple[str, ...] = (
    "torch_npu_custom_op",
    "acl_aclnn",
    "atb",
    "mindie",
    UNAVAILABLE,
    NOT_SELECTED,
)

#: Patterns that keep the key name and replace only the value.
_KV_SECRET_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("bearer", re.compile(r"(?i)\b(bearer|token|authorization)(\s*[:=]\s*)(\S+)")),
    ("password", re.compile(r"(?i)\b(password|passwd|pwd|secret|api[_-]?key)(\s*[:=]\s*)(\S+)")),
)

#: Patterns whose whole match is a credential (user:pass inside a URL).
_WHOLE_SECRET_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("private_registry", re.compile(r"//[^/\s]+:[^@\s]+@")),
    ("ssh_url", re.compile(r"(?i)\bgit\+ssh://[^/\s]+:[^@\s]+@")),
)

_SECRET_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    _KV_SECRET_PATTERNS + _WHOLE_SECRET_PATTERNS
)

#: ``/home/<user>``, ``/Users/<user>`` are personal paths: they are normalised so
#: a manifest cannot leak a username (E09-01 step 10).
_HOME_PATTERN = re.compile(r"/(?:home|Users)/[^/\s]+")


def _require(condition: bool, message: str, **details: Any) -> None:
    if not condition:
        raise ConfigError(message, details=details or None)


def redact_text(text: str) -> str:
    """Strip credentials and personal home paths from one string."""
    redacted = text
    for _name, pattern in _KV_SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}<REDACTED>", redacted)
    for _name, pattern in _WHOLE_SECRET_PATTERNS:
        redacted = pattern.sub("//<REDACTED>@", redacted)
    return _HOME_PATTERN.sub("/home/<user>", redacted)


def redaction_audit(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Walk a manifest and report which fields *would* have leaked.

    The audit runs on the pre-redaction document so a report can prove the
    redactor had work to do (or that it did not).
    """
    hits: List[Dict[str, str]] = []

    def walk(node: Any, pointer: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{pointer}/{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{pointer}/{index}")
        elif isinstance(node, str):
            for name, pattern in _SECRET_PATTERNS:
                if pattern.search(node):
                    hits.append({"pointer": pointer, "pattern": name})
            if _HOME_PATTERN.search(node):
                hits.append({"pointer": pointer, "pattern": "personal_home_path"})

    walk(payload, "")
    return {"clean": not hits, "hits": hits, "hit_count": len(hits)}


def redact_document(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-copy ``payload`` with every string passed through :func:`redact_text`."""

    def walk(node: Any) -> Any:
        if isinstance(node, Mapping):
            return {str(key): walk(value) for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(value) for value in node]
        if isinstance(node, str):
            return redact_text(node)
        return node

    return dict(walk(payload))


@dataclass(frozen=True)
class OfficialSource:
    """One official compatibility citation (E09-01 §5, step 9).

    The page version and access time are mandatory: a CANN 8.5 API page cannot
    prove a 9.x SKU, and five pages from five versions are not a matrix.
    """

    product: str
    cann_version: str
    page_uri: str
    page_version: str
    accessed_at: str
    entries: Tuple[str, ...] = ()
    covers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("product", "cann_version", "page_uri", "page_version", "accessed_at"):
            _require(bool(getattr(self, name)), f"an official source needs a non-empty {name!r}")
        _require(bool(self.entries), "an official source with no cited entry proves nothing")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "product": self.product,
            "cann_version": self.cann_version,
            "page_uri": self.page_uri,
            "page_version": self.page_version,
            "accessed_at": self.accessed_at,
            "entries": list(self.entries),
            "covers": list(self.covers),
        }


@dataclass
class CompatibilityManifest:
    """The frozen chip/firmware/driver/CANN/framework identity for one run set."""

    schema_version: str = SCHEMA_VERSION
    hardware: Dict[str, Any] = field(default_factory=dict)
    ascend_stack: Dict[str, Any] = field(default_factory=dict)
    framework: Dict[str, Any] = field(default_factory=dict)
    project: Dict[str, Any] = field(default_factory=dict)
    official_sources: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    collected_at: str = ""
    device_id: str = ""
    requested_backend: str = "ascend"

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "CompatibilityManifest":
        unknown = set(document) - set(SECTIONS) - {
            "schema_version",
            "collected_at",
            "device_id",
            "requested_backend",
        }
        if unknown:
            raise SchemaError(
                f"unknown top-level manifest keys {sorted(unknown)}; the schema is closed "
                "so a typo cannot silently create a new field",
                details={"unknown_keys": sorted(unknown)},
            )
        return cls(
            schema_version=str(document.get("schema_version", SCHEMA_VERSION)),
            hardware=dict(document.get("hardware", {}) or {}),
            ascend_stack=dict(document.get("ascend_stack", {}) or {}),
            framework=dict(document.get("framework", {}) or {}),
            project=dict(document.get("project", {}) or {}),
            official_sources=[dict(item) for item in document.get("official_sources", []) or []],
            evidence=dict(document.get("evidence", {}) or {}),
            collected_at=str(document.get("collected_at", "")),
            device_id=str(document.get("device_id", "")),
            requested_backend=str(document.get("requested_backend", "ascend")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collected_at": self.collected_at,
            "device_id": self.device_id,
            "requested_backend": self.requested_backend,
            "hardware": dict(self.hardware),
            "ascend_stack": dict(self.ascend_stack),
            "framework": dict(self.framework),
            "project": dict(self.project),
            "official_sources": [dict(item) for item in self.official_sources],
            "evidence": dict(self.evidence),
        }

    # ── canonicalization / hash ─────────────────────────────────────────────

    def redacted(self) -> "CompatibilityManifest":
        """A copy with credentials and personal paths removed."""
        return CompatibilityManifest.from_document(redact_document(self.as_dict()))

    def canonical(self) -> str:
        """Canonical JSON of the **redacted** manifest (E09-01 step 11)."""
        return canonical_json(self.redacted().as_dict())

    @property
    def sha256(self) -> str:
        return sha256_hex(self.canonical())

    # ── validation ──────────────────────────────────────────────────────────

    def validate(self) -> "SchemaValidation":
        """Field/type/enum/required validation (E09-01 step 26)."""
        problems: List[Dict[str, str]] = []

        if self.schema_version != SCHEMA_VERSION:
            problems.append(
                {
                    "pointer": "/schema_version",
                    "problem": "unsupported_schema_version",
                    "detail": f"expected {SCHEMA_VERSION!r}, got {self.schema_version!r}",
                }
            )

        for section in SECTIONS:
            payload = getattr(self, section)
            if section in ("official_sources",):
                if not isinstance(payload, list):
                    problems.append(
                        {"pointer": f"/{section}", "problem": "wrong_type", "detail": "expected a list"}
                    )
                continue
            if not isinstance(payload, Mapping):
                problems.append(
                    {"pointer": f"/{section}", "problem": "wrong_type", "detail": "expected a mapping"}
                )
                continue
            for key in REQUIRED_KEYS.get(section, ()):
                if key not in payload:
                    problems.append(
                        {
                            "pointer": f"/{section}/{key}",
                            "problem": "missing_required_field",
                            "detail": "required by E09-01 §4",
                        }
                    )
                elif payload[key] in ("", None):
                    problems.append(
                        {
                            "pointer": f"/{section}/{key}",
                            "problem": "empty_required_field",
                            "detail": f"write {UNAVAILABLE!r} plus a reason instead of leaving it blank",
                        }
                    )

        path = self.framework.get("selected_integration_path", "")
        if path and path not in INTEGRATION_PATHS:
            problems.append(
                {
                    "pointer": "/framework/selected_integration_path",
                    "problem": "unknown_enum",
                    "detail": f"expected one of {list(INTEGRATION_PATHS)}, got {path!r}",
                }
            )

        device_count = self.hardware.get("device_count")
        if device_count is not None and not isinstance(device_count, int):
            problems.append(
                {
                    "pointer": "/hardware/device_count",
                    "problem": "wrong_type",
                    "detail": f"expected an int, got {type(device_count).__name__}",
                }
            )
        dirty = self.project.get("git_dirty")
        if dirty is not None and not isinstance(dirty, bool):
            problems.append(
                {
                    "pointer": "/project/git_dirty",
                    "problem": "wrong_type",
                    "detail": f"expected a bool, got {type(dirty).__name__}",
                }
            )

        for index, source in enumerate(self.official_sources):
            try:
                OfficialSource(
                    product=str(source.get("product", "")),
                    cann_version=str(source.get("cann_version", "")),
                    page_uri=str(source.get("page_uri", "")),
                    page_version=str(source.get("page_version", "")),
                    accessed_at=str(source.get("accessed_at", "")),
                    entries=tuple(source.get("entries", ()) or ()),
                    covers=tuple(source.get("covers", ()) or ()),
                )
            except ConfigError as exc:
                problems.append(
                    {
                        "pointer": f"/official_sources/{index}",
                        "problem": "invalid_official_source",
                        "detail": str(exc),
                    }
                )

        redaction = redaction_audit(self.as_dict())
        for hit in redaction["hits"]:
            problems.append(
                {
                    "pointer": hit["pointer"],
                    "problem": "sensitive_field",
                    "detail": f"pattern {hit['pattern']!r} must be redacted before the manifest is stored",
                }
            )

        return SchemaValidation(ok=not problems, problems=problems, redaction=redaction)


@dataclass(frozen=True)
class SchemaValidation:
    """Result of :meth:`CompatibilityManifest.validate`."""

    ok: bool
    problems: Tuple[Dict[str, str], ...] = ()
    redaction: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "problem_count": len(self.problems),
            "problems": [dict(item) for item in self.problems],
            "redaction": dict(self.redaction),
        }

    def require(self) -> "SchemaValidation":
        if not self.ok:
            raise SchemaError(
                f"compatibility manifest failed schema validation with {len(self.problems)} problem(s): "
                + "; ".join(f"{item['pointer']}={item['problem']}" for item in self.problems[:5]),
                details=self.as_dict(),
            )
        return self


@dataclass(frozen=True)
class FieldDiff:
    """One field-level difference between two manifests (E09-01 step 25)."""

    pointer: str
    old: Any
    new: Any
    severity: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pointer": self.pointer,
            "old": self.old,
            "new": self.new,
            "severity": self.severity,
        }


def _severity_for(pointer: str) -> str:
    if pointer in VOLATILE_FIELDS or any(
        pointer == volatile or pointer.startswith(volatile + "/") for volatile in VOLATILE_FIELDS
    ):
        return SEVERITY_VOLATILE
    if pointer in CRITICAL_FIELDS or any(
        pointer.startswith(critical + "/") for critical in CRITICAL_FIELDS
    ):
        return SEVERITY_CRITICAL
    if pointer.startswith("/hardware") or pointer.startswith("/ascend_stack"):
        return SEVERITY_MAJOR
    return SEVERITY_MINOR


def _flatten(node: Any, pointer: str, out: Dict[str, Any]) -> None:
    if isinstance(node, Mapping):
        if not node:
            out[pointer] = {}
        for key, value in node.items():
            _flatten(value, f"{pointer}/{key}", out)
    elif isinstance(node, (list, tuple)):
        if not node:
            out[pointer] = []
        for index, value in enumerate(node):
            _flatten(value, f"{pointer}/{index}", out)
    else:
        out[pointer] = node


def diff_manifests(
    old: CompatibilityManifest, new: CompatibilityManifest
) -> Dict[str, Any]:
    """Field-level diff with JSON Pointer, old/new value and severity.

    The hash answers "did anything change"; this answers "what changed and does
    it invalidate the run".  Both are kept because they do different jobs
    (E09-01 step 11).
    """
    left: Dict[str, Any] = {}
    right: Dict[str, Any] = {}
    _flatten(old.redacted().as_dict(), "", left)
    _flatten(new.redacted().as_dict(), "", right)

    diffs: List[FieldDiff] = []
    for pointer in sorted(set(left) | set(right)):
        before = left.get(pointer, "<absent>")
        after = right.get(pointer, "<absent>")
        if before == after:
            continue
        diffs.append(FieldDiff(pointer=pointer, old=before, new=after, severity=_severity_for(pointer)))

    by_severity = {
        severity: [item.as_dict() for item in diffs if item.severity == severity]
        for severity in DIFF_SEVERITIES
    }
    return {
        "old_sha256": old.sha256,
        "new_sha256": new.sha256,
        "identical": old.sha256 == new.sha256,
        "diff_count": len(diffs),
        "by_severity": by_severity,
        "diffs": [item.as_dict() for item in diffs],
        # E09-01 §9: a diff limited to volatile fields means "reproducible", not
        # "bit-identical environment".  Saying so is the point of the field.
        "reproducible_ignoring_volatile": not any(
            item.severity in (SEVERITY_CRITICAL, SEVERITY_MAJOR, SEVERITY_MINOR) for item in diffs
        ),
    }


def unavailable(reason: str) -> Dict[str, Any]:
    """The canonical 'we could not read this' value (E09-01 step 5)."""
    _require(bool(reason), "UNAVAILABLE needs a reason; a bare marker hides why")
    return {"value": UNAVAILABLE, "reason": reason}


def require_official_coverage(
    manifest: CompatibilityManifest, required: Iterable[str]
) -> Dict[str, Any]:
    """Check that every required five-tuple member has a citation (step 9)."""
    covered: Dict[str, List[str]] = {}
    for index, source in enumerate(manifest.official_sources):
        for item in source.get("covers", []) or []:
            covered.setdefault(str(item), []).append(f"/official_sources/{index}")
    missing = sorted({name for name in required if name not in covered})
    return {
        "ok": not missing,
        "required": sorted(required),
        "covered": covered,
        "missing": missing,
        "note": (
            ""
            if not missing
            else "an item without a citation stays UNKNOWN; it is not upgraded by experience"
        ),
    }


def compatibility_verdict(
    manifest: CompatibilityManifest,
    validation: SchemaValidation,
    probe_summary: Mapping[str, Any],
    capability_coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    """E09-01 step 28: the gate verdict assembled from evidence, never guessed."""
    hash_ok = bool(probe_summary.get("hash_recomputable", False))
    probes = dict(probe_summary.get("probes", {}) or {})
    required_probes: Sequence[str] = tuple(probe_summary.get("required_probes", ()) or ())
    probe_rows = [
        {"probe": name, "status": probes.get(name, "NOT_RUN")} for name in required_probes
    ]
    probe_ok = all(row["status"] == "pass" for row in probe_rows) and bool(probe_rows)
    capability_ok = not capability_coverage.get("missing_probes")
    official_ok = bool(probe_summary.get("official_coverage_ok", False))

    if not validation.ok:
        status, reason = "FAIL", "manifest schema validation failed"
    elif not hash_ok:
        status, reason = "FAIL", "canonical hash is not recomputable"
    elif not official_ok:
        status, reason = "BLOCKED", "official compatibility citations incomplete"
    elif not probe_ok:
        status, reason = "BLOCKED", "one or more required probes did not pass"
    elif not capability_ok:
        status, reason = "BLOCKED", "in-scope capabilities still lack a passing probe"
    else:
        status, reason = "PASS", "matching chain verified; negative injections reported separately"

    return {
        "status": status,
        "reason": reason,
        "manifest_sha256": manifest.sha256,
        "schema_ok": validation.ok,
        "hash_recomputable": hash_ok,
        "official_coverage_ok": official_ok,
        "probes": probe_rows,
        "capability_coverage": dict(capability_coverage),
        "note": (
            "profiler failures are reported separately: a passing compute chain with an "
            "unavailable profiler is BLOCKED_BY_PROFILER_CAPABILITY for E09-04/E09-08, "
            "not a lowered standard"
        ),
    }


__all__ = [
    "CRITICAL_FIELDS",
    "CompatibilityManifest",
    "DIFF_SEVERITIES",
    "FieldDiff",
    "INTEGRATION_PATHS",
    "NOT_SELECTED",
    "OfficialSource",
    "REQUIRED_KEYS",
    "SCHEMA_VERSION",
    "SECTIONS",
    "SEVERITY_CRITICAL",
    "SEVERITY_MAJOR",
    "SEVERITY_MINOR",
    "SEVERITY_VOLATILE",
    "SchemaValidation",
    "UNAVAILABLE",
    "VOLATILE_FIELDS",
    "compatibility_verdict",
    "diff_manifests",
    "redact_document",
    "redact_text",
    "redaction_audit",
    "require_official_coverage",
    "unavailable",
]
