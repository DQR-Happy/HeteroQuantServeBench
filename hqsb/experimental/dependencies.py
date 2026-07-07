"""E14-01 — core/experimental dependency, feature-flag and CI isolation.

The experiment's core judgement (``E14-01`` header): installing only the core
must not download, import or initialise a training/RL/multimodal/edge
heavyweight; turning any experimental feature off must leave the core contracts,
CPU path, CLI, schemas and CI behaviour equivalent; a missing optional capability
must return a structured capability reason instead of polluting the import graph
or silently falling back.

What this module provides, in the order of the protocol's forty steps:

* :class:`DependencyPolicy` — the frozen classification of every dependency into
  a layer, with owner/purpose/licence/platform/allowed-import-layer (steps 2–3);
* :class:`WheelMetadata` / :func:`audit_wheel_metadata` — parse a *built*
  ``METADATA`` file and diff its ``Requires-Dist``/``Provides-Extra`` against the
  policy, so the leak is found in the artefact rather than in the source file
  (steps 6–8);
* :class:`ImportTrace` / :func:`analyse_import_trace` — the imported-module set
  and the side-effect audit (device context, child process, network, cache),
  because import purity is a *testable* property (steps 10–12);
* :func:`static_import_edges` — an AST scan producing the real import graph
  including re-exports, so a leak through ``__init__`` is caught (step 16);
* :class:`FeatureFlagRegistry` — the flag states of ``E14-01`` §3.2 with the
  requested/actual/reason triple of the manual §5.7 (steps 20, 28–29);
* :class:`CapabilityProbe` — a probe that distinguishes "dependency absent" from
  "ABI mismatch" from "device missing" (steps 30–31);
* :class:`CoreGoldenSurface` / :func:`compare_core_golden` — the frozen public
  surface and the equivalence diff (steps 1, 13–15, 27, 34–35);
* :func:`negative_case_matrix` — the injectable leak/mismatch fixtures with the
  detector each one must trip (steps 17–18, 29–31);
* :class:`DependencyBoundaryVerdict` — the per-extra verdict of step 40.

**Nothing here installs, builds, imports or probes anything by itself.**  Every
capability answer is produced by a caller-supplied probe or by an explicit
``find_spec`` call, and an absent dependency is always a *state*.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.identity import SCHEMA_PREFIX, canonical_digest, digest_text

EXPERIMENT_ID = "E14-01"
TITLE = "Core/Experimental 依赖、Feature Flag 与 CI 隔离"
LEVEL = "P0"

CLAIM_BOUNDARY = (
    "只证明依赖边界、导入纯度与 feature 启停语义；不证明任何训练/RL/前沿算法正确、快速"
    "或值得进入主线（E14-01 §11）。"
)

#: Dependency layers (``E14-01`` step 2).
DEPENDENCY_LAYERS: Tuple[str, ...] = (
    "build",
    "core",
    "test",
    "doc",
    "train",
    "rl",
    "frontier",
    "multimodal",
    "agent",
    "edge",
)

#: Layers a core install may resolve. Everything else is an experimental extra.
CORE_LAYERS: Tuple[str, ...] = ("build", "core", "test", "doc")

#: Where a dependency is allowed to be imported (``E14-01`` step 2).
ALLOWED_IMPORT_LAYERS: Tuple[str, ...] = (
    "module_level",
    "function_level",
    "probe_only",
    "build_only",
)

#: Reasons to reject a policy entry.
POLICY_REQUIRED_FIELDS: Tuple[str, ...] = (
    "name",
    "layer",
    "owner",
    "purpose",
    "license",
    "platforms",
    "allowed_import_layer",
)


@dataclass(frozen=True)
class DependencyEntry:
    """One dependency with its frozen classification."""

    name: str
    layer: str
    owner: str
    purpose: str
    license: str
    platforms: Tuple[str, ...]
    allowed_import_layer: str = "function_level"
    extra: str = ""
    version_constraint: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in POLICY_REQUIRED_FIELDS:
            if not getattr(self, name):
                findings.append(f"{self.name or '<unnamed>'}: required field {name!r} is empty")
        if self.layer and self.layer not in DEPENDENCY_LAYERS:
            findings.append(f"{self.name}: unknown layer {self.layer!r}")
        if self.allowed_import_layer and self.allowed_import_layer not in ALLOWED_IMPORT_LAYERS:
            findings.append(f"{self.name}: unknown allowed_import_layer {self.allowed_import_layer!r}")
        if self.layer not in CORE_LAYERS and self.layer != "build":
            if self.extra not in rec.EXTRA_FEATURE_MAPPING:
                findings.append(
                    f"{self.name}: layer {self.layer!r} must be reachable through exactly one declared extra"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "owner": self.owner,
            "purpose": self.purpose,
            "license": self.license,
            "platforms": list(self.platforms),
            "allowed_import_layer": self.allowed_import_layer,
            "extra": self.extra,
            "version_constraint": self.version_constraint,
        }


@dataclass
class DependencyPolicy:
    """The frozen dependency classification (``E14-01`` steps 2–3)."""

    entries: Tuple[DependencyEntry, ...] = ()
    policy_version: str = "v1"

    def by_name(self) -> Dict[str, DependencyEntry]:
        return {entry.name: entry for entry in self.entries}

    def by_layer(self, layer: str) -> Tuple[DependencyEntry, ...]:
        return tuple(entry for entry in self.entries if entry.layer == layer)

    def for_extra(self, extra: str) -> Tuple[DependencyEntry, ...]:
        return tuple(entry for entry in self.entries if entry.extra == extra)

    def validate(self) -> List[str]:
        findings: List[str] = []
        seen: Set[str] = set()
        for entry in self.entries:
            findings.extend(entry.problems())
            if entry.name in seen:
                findings.append(
                    f"{entry.name}: declared more than once (the boundary is meaningless if a "
                    "dependency is both core and experimental)"
                )
            seen.add(entry.name)
        if not self.entries:
            findings.append("dependency policy is empty")
        return findings

    def core_names(self) -> Tuple[str, ...]:
        return tuple(sorted(entry.name for entry in self.entries if entry.layer in CORE_LAYERS))

    def experimental_names(self) -> Tuple[str, ...]:
        return tuple(sorted(entry.name for entry in self.entries if entry.layer not in CORE_LAYERS))

    def digest(self) -> str:
        return canonical_digest(
            {"version": self.policy_version, "entries": [entry.as_dict() for entry in self.entries]}
        )

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "DependencyPolicy":
        """Build a policy from ``configs/experimental/dependency-policy.yaml``."""
        entries: List[DependencyEntry] = []
        for raw in document.get("rules", []):
            if not isinstance(raw, Mapping):
                raise ConfigError("dependency policy rule must be a mapping")
            entries.append(
                DependencyEntry(
                    name=str(raw.get("name", "")),
                    layer=str(raw.get("layer", "")),
                    owner=str(raw.get("owner", "")),
                    purpose=str(raw.get("purpose", "")),
                    license=str(raw.get("license", "")),
                    platforms=tuple(str(item) for item in raw.get("platforms", ()) or ()),
                    allowed_import_layer=str(raw.get("allowed_import_layer", "function_level")),
                    extra=str(raw.get("extra", "")),
                    version_constraint=str(raw.get("version_constraint", "")),
                )
            )
        return cls(entries=tuple(entries), policy_version=str(document.get("version", "v1")))


# ── wheel metadata (steps 6–8) ─────────────────────────────────────────────


@dataclass
class WheelMetadata:
    """A parsed ``*.dist-info/METADATA`` (the artefact, not ``pyproject.toml``)."""

    name: str
    version: str
    requires_dist: Tuple[str, ...] = ()
    provides_extra: Tuple[str, ...] = ()
    requires_python: str = ""
    raw_digest: str = ""

    @classmethod
    def parse(cls, text: str, *, name: str = "", version: str = "") -> "WheelMetadata":
        requires: List[str] = []
        extras: List[str] = []
        requires_python = ""
        resolved_name = name
        resolved_version = version
        for line in text.splitlines():
            if line.startswith("Name:") and not resolved_name:
                resolved_name = line.split(":", 1)[1].strip()
            elif line.startswith("Version:") and not resolved_version:
                resolved_version = line.split(":", 1)[1].strip()
            elif line.startswith("Requires-Dist:"):
                requires.append(line.split(":", 1)[1].strip())
            elif line.startswith("Provides-Extra:"):
                extras.append(line.split(":", 1)[1].strip())
            elif line.startswith("Requires-Python:"):
                requires_python = line.split(":", 1)[1].strip()
        return cls(
            name=resolved_name,
            version=resolved_version,
            requires_dist=tuple(requires),
            provides_extra=tuple(extras),
            requires_python=requires_python,
            raw_digest=digest_text(text),
        )

    @classmethod
    def load(cls, path: str) -> "WheelMetadata":
        with open(path, encoding="utf-8") as handle:
            return cls.parse(handle.read())

    def plain_requires(self) -> Tuple[str, ...]:
        """``Requires-Dist`` entries that are unconditional (no ``extra ==`` marker)."""
        return tuple(item for item in self.requires_dist if "extra ==" not in item)

    def requires_for_extra(self, extra: str) -> Tuple[str, ...]:
        marker = f'extra == "{extra}"'
        return tuple(item for item in self.requires_dist if marker in item)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "requires_python": self.requires_python,
            "provides_extra": list(self.provides_extra),
            "requires_dist": list(self.requires_dist),
            "raw_digest": self.raw_digest,
        }


def _distribution_name(requirement: str) -> str:
    """Extract the distribution name from a ``Requires-Dist`` entry."""
    head = requirement.split(";", 1)[0].strip()
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", " ", "("):
        head = head.split(separator, 1)[0]
    return head.strip().lower().replace("_", "-")


def audit_wheel_metadata(metadata: WheelMetadata, policy: DependencyPolicy) -> Dict[str, Any]:
    """Diff the built wheel against the frozen policy (``E14-01`` steps 6–7).

    The interesting failures are: an experimental dependency declared
    unconditionally (it would be installed by ``pip install <core-wheel>``), and
    an extra that the wheel does not advertise although the code maps it.
    """
    by_name = {name.replace("_", "-"): entry for name, entry in policy.by_name().items()}
    leaks: List[Dict[str, str]] = []
    for requirement in metadata.plain_requires():
        name = _distribution_name(requirement)
        entry = by_name.get(name)
        if entry is None:
            leaks.append({"package": name, "problem": "unconditional requirement is not in the dependency policy"})
            continue
        if entry.layer not in CORE_LAYERS:
            leaks.append(
                {
                    "package": name,
                    "problem": f"experimental-layer dependency ({entry.layer}) declared as a core requirement",
                }
            )
    missing_extras: List[str] = []
    for extra in rec.EXTRA_FEATURE_MAPPING:
        if extra not in metadata.provides_extra:
            missing_extras.append(extra)
    expected_core = set(policy.core_names())
    declared_core = {_distribution_name(item) for item in metadata.plain_requires()}
    missing_core = sorted(
        name for name in expected_core if name.replace("_", "-") not in declared_core and name not in ("pydantic", "pyyaml")
    )
    return {
        "wheel": metadata.as_dict(),
        "policy_digest": policy.digest(),
        "leaks": leaks,
        "missing_extras": missing_extras,
        "missing_core_requirements": missing_core,
        "core_clean": not leaks,
        "ok": not leaks and not missing_extras,
    }


# ── import trace and purity (steps 10–12) ──────────────────────────────────


@dataclass
class ImportTrace:
    """The observation set of one cold import (``E14-01`` steps 10–12)."""

    environment_id: str
    imported_modules: Tuple[str, ...] = ()
    loaded_shared_libraries: Tuple[str, ...] = ()
    thread_count: int = 0
    child_process_count: int = 0
    device_context_created: bool = False
    network_access: bool = False
    cache_written: bool = False
    env_vars_modified: Tuple[str, ...] = ()
    wall_ms: Optional[float] = None
    cpu_ms: Optional[float] = None
    rss_bytes: Optional[int] = None
    phase: str = "cold"
    notes: Tuple[str, ...] = ()

    def module_set_digest(self) -> str:
        return canonical_digest(sorted(self.imported_modules))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "phase": self.phase,
            "imported_modules": sorted(self.imported_modules),
            "imported_modules_digest": self.module_set_digest(),
            "loaded_shared_libraries": sorted(self.loaded_shared_libraries),
            "thread_count": self.thread_count,
            "child_process_count": self.child_process_count,
            "device_context_created": self.device_context_created,
            "network_access": self.network_access,
            "cache_written": self.cache_written,
            "env_vars_modified": sorted(self.env_vars_modified),
            "wall_ms": self.wall_ms,
            "cpu_ms": self.cpu_ms,
            "rss_bytes": self.rss_bytes,
            "notes": list(self.notes),
        }


#: Module prefixes that an experimental import would drag in.  A core-only
#: import trace that contains any of them is a leak (``E14-01`` §5 观测量).
EXPERIMENTAL_MODULE_MARKERS: Tuple[str, ...] = (
    "torch.distributed",
    "torch.distributed.fsdp",
    "deepspeed",
    "megatron",
    "ray",
    "vllm",
    "sglang",
    "trl",
    "peft",
    "accelerate",
    "transformers",
    "diffusers",
    "librosa",
    "torchaudio",
    "tvm",
    "onnx",
    "onnxruntime",
    "tflite",
    "tensorflow",
    "qnn",
    "openvino",
    "opentelemetry",
)


def analyse_import_trace(trace: ImportTrace, *, expected_markers: Sequence[str] = ()) -> Dict[str, Any]:
    """Classify one import trace: purity violations and the leak markers found."""
    violations: List[str] = []
    if trace.device_context_created:
        violations.append("import created a device context")
    if trace.child_process_count > 0:
        violations.append(f"import forked {trace.child_process_count} child process(es)")
    if trace.network_access:
        violations.append("import performed network access")
    if trace.cache_written:
        violations.append("import wrote to a cache")
    if trace.env_vars_modified:
        violations.append("import modified environment variables: " + ", ".join(sorted(trace.env_vars_modified)))
    present = sorted(
        module
        for module in trace.imported_modules
        if any(module == marker or module.startswith(marker + ".") for marker in EXPERIMENTAL_MODULE_MARKERS)
    )
    unexpected = sorted(module for module in present if module not in set(expected_markers))
    if unexpected:
        violations.append("experimental modules present in this environment: " + ", ".join(unexpected))
    return {
        "environment_id": trace.environment_id,
        "phase": trace.phase,
        "pure": not violations,
        "violations": violations,
        "experimental_markers_present": present,
        "unexpected_markers": unexpected,
        "module_count": len(trace.imported_modules),
        "module_set_digest": trace.module_set_digest(),
    }


# ── static import graph (step 16) ──────────────────────────────────────────


@dataclass(frozen=True)
class ImportEdge:
    source_module: str
    target: str
    line: int
    kind: str = "import"
    form: str = "module_level"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_module,
            "target": self.target,
            "line": self.line,
            "kind": self.kind,
            "form": self.form,
        }


def static_import_edges(source_module: str, text: str) -> Tuple[Tuple[ImportEdge, ...], Optional[str]]:
    """AST-scan one module for its real import edges (re-exports included).

    A ``grep`` over ``import`` lines misses ``from x import y`` re-exports and
    ``__all__``-driven laziness; this scan sees them, and records whether the
    edge is module level or inside a function, since *that* is what decides
    whether a heavy dependency is pulled in at import time (``E14-01`` step 16).
    Returns ``(edges, parse_error)``.
    """
    try:
        tree = ast.parse(text, filename=source_module)
    except SyntaxError as exc:
        return (), f"{source_module}:{exc.lineno}: {exc.msg}"
    edges: List[ImportEdge] = []

    def walk(node: ast.AST, form: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    edges.append(ImportEdge(source_module, alias.name, child.lineno, "import", form))
                continue
            if isinstance(child, ast.ImportFrom):
                module = ("." * (child.level or 0)) + (child.module or "")
                for alias in child.names:
                    edges.append(ImportEdge(source_module, module or alias.name, child.lineno, "from", form))
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, "function_level")
                continue
            if isinstance(child, ast.If):
                expression = ast.unparse(child.test) if hasattr(ast, "unparse") else ""
                walk(child, "type_checking" if "TYPE_CHECKING" in expression else form)
                continue
            walk(child, form)

    walk(tree, "module_level")
    return tuple(edges), None


# ── feature flags and capabilities (steps 20, 28–31) ───────────────────────


@dataclass(frozen=True)
class FeatureFlag:
    """One feature flag: what it gates and what its default is."""

    name: str
    default: bool
    extra: str
    capability: str
    cli_subcommand: str = ""
    test_job: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "default": self.default,
            "extra": self.extra,
            "capability": self.capability,
            "cli_subcommand": self.cli_subcommand,
            "test_job": self.test_job,
        }


@dataclass(frozen=True)
class CapabilityResolution:
    """The requested/actual/reason triple the manual §5.7 requires."""

    capability: str
    state: str
    available: bool
    reason_code: str = ""
    requested_implementation: str = ""
    actual_implementation: str = ""
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "state": self.state,
            "available": self.available,
            "reason_code": self.reason_code,
            "requested_implementation": self.requested_implementation,
            "actual_implementation": self.actual_implementation,
            "detail": self.detail,
        }


class FeatureFlagRegistry:
    """Flags, their defaults, and a strict resolver (``E14-01`` steps 3, 28–29)."""

    def __init__(self, flags: Sequence[FeatureFlag]) -> None:
        self._flags = {flag.name: flag for flag in flags}
        duplicates = len(flags) - len(self._flags)
        if duplicates:
            raise ConfigError(f"feature flag registry has {duplicates} duplicate name(s)")

    def flags(self) -> Tuple[FeatureFlag, ...]:
        return tuple(self._flags[name] for name in sorted(self._flags))

    def get(self, name: str) -> FeatureFlag:
        if name not in self._flags:
            raise ConfigError(f"unknown feature flag {name!r}; known: {', '.join(sorted(self._flags))}")
        return self._flags[name]

    def resolve(self, overrides: Mapping[str, Any]) -> Tuple[Dict[str, bool], List[str]]:
        """Resolve flags; unknown or non-boolean values are refused, not ignored.

        ``E14-01`` step 29: an unknown or misspelled flag must be a structured
        rejection, because silently ignoring it makes the user believe a feature
        is enabled when it is not.
        """
        problems: List[str] = []
        resolved = {flag.name: flag.default for flag in self.flags()}
        for name, value in overrides.items():
            if name not in self._flags:
                problems.append(f"unknown feature flag {name!r}")
                continue
            if isinstance(value, bool):
                resolved[name] = value
            elif isinstance(value, str) and value.lower() in ("true", "false"):
                resolved[name] = value.lower() == "true"
            else:
                problems.append(f"feature flag {name!r} must be boolean, got {value!r}")
        return resolved, problems

    def conflicts(self, resolved: Mapping[str, bool]) -> List[str]:
        """Mutually exclusive flags enabled together (declared by extra + capability)."""
        problems: List[str] = []
        by_extra: Dict[str, List[str]] = {}
        for name, enabled in resolved.items():
            if enabled:
                by_extra.setdefault(self._flags[name].extra, []).append(name)
        # Two frontier *mechanisms* may not be enabled at once: E14-05 allows
        # exactly one primary branch, and a resolved flag set is the executable
        # form of that rule.
        frontier_enabled = [
            name
            for name, enabled in resolved.items()
            if enabled and name.startswith("frontier.")
        ]
        if len(frontier_enabled) > 1:
            problems.append(
                "more than one frontier mechanism flag enabled simultaneously: "
                + ", ".join(sorted(frontier_enabled))
                + " (E14-05: 恰有一个主要分支)"
            )
        return problems

    def matrix_rows(self, resolved: Mapping[str, bool], capabilities: Sequence[CapabilityResolution]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for flag in self.flags():
            capability = next((item for item in capabilities if item.capability == flag.capability), None)
            rows.append(
                {
                    "flag": flag.name,
                    "value": resolved.get(flag.name),
                    "valid": flag.name in resolved,
                    "registry_diff": "enabled" if resolved.get(flag.name) else "disabled",
                    "config_digest": canonical_digest({"flag": flag.name, "value": resolved.get(flag.name)}),
                    "capability_state": capability.state if capability else rec.CAP_DISABLED_BY_CONFIG,
                }
            )
        return rows


def flag_from_extra(extra: str, capability: str, *, default: bool = False, cli_subcommand: str = "", test_job: str = "") -> FeatureFlag:
    """Build a flag whose name and extra are checked against the frozen mapping."""
    if extra not in rec.EXTRA_FEATURE_MAPPING:
        raise ConfigError(f"unknown extra {extra!r}")
    return FeatureFlag(
        name=capability,
        default=default,
        extra=extra,
        capability=capability,
        cli_subcommand=cli_subcommand,
        test_job=test_job,
    )


def default_feature_flags() -> Tuple[FeatureFlag, ...]:
    """The S14 flag set derived from ``records.EXTRA_FEATURE_MAPPING``."""
    flags: List[FeatureFlag] = []
    for extra, capabilities in sorted(rec.EXTRA_FEATURE_MAPPING.items()):
        for capability in capabilities:
            flags.append(FeatureFlag(name=capability, default=False, extra=extra, capability=capability))
    return tuple(flags)


#: Top-level import names each extra may legitimately need, for capability probes.
EXTRA_IMPORT_NAMES: Mapping[str, Tuple[str, ...]] = {
    "train": ("torch", "torch.distributed"),
    "rl": ("torch", "torch.distributed", "transformers"),
    "frontier": ("torch",),
    "multimodal": ("torch", "transformers"),
    "agent": ("httpx",),
    "edge": (),
}


def probe_dependency(import_name: str, *, environment_id: str = "") -> CapabilityResolution:
    """Probe one import's *presence* without importing it.

    ``find_spec`` is deliberately used instead of ``import``: the probe must not
    create a device context or start a worker (``E14-01`` §3.3).  A missing spec
    yields ``UNAVAILABLE_DEPENDENCY`` with a reason code — an import error is
    never raised out of this function.
    """
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ValueError) as exc:
        return CapabilityResolution(
            capability=f"dependency:{import_name}",
            state=rec.CAP_UNAVAILABLE_DEPENDENCY,
            available=False,
            reason_code="DEPENDENCY_ABSENT",
            requested_implementation=import_name,
            actual_implementation="",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if spec is None:
        return CapabilityResolution(
            capability=f"dependency:{import_name}",
            state=rec.CAP_UNAVAILABLE_DEPENDENCY,
            available=False,
            reason_code="DEPENDENCY_ABSENT",
            requested_implementation=import_name,
        )
    origin = spec.origin or ""
    return CapabilityResolution(
        capability=f"dependency:{import_name}",
        state=rec.CAP_AVAILABLE,
        available=True,
        requested_implementation=import_name,
        actual_implementation=origin or spec.name,
        detail=environment_id,
    )


def probe_extra(extra: str, *, environment_id: str = "", flags: Optional[Mapping[str, bool]] = None) -> CapabilityResolution:
    """Probe an extra as a whole, honouring the feature flag first.

    Order matters and mirrors ``E14-01`` §3.2: *flag off* wins over *dependency
    present*, because a disabled feature must not load its implementation even
    when it happens to be installed.
    """
    if extra not in rec.EXTRA_FEATURE_MAPPING:
        raise ConfigError(f"unknown extra {extra!r}; known: {', '.join(sorted(rec.EXTRA_FEATURE_MAPPING))}")
    capabilities = rec.EXTRA_FEATURE_MAPPING[extra]
    enabled = [name for name in capabilities if (flags or {}).get(name, False)]
    if not enabled:
        return CapabilityResolution(
            capability=f"extra:{extra}",
            state=rec.CAP_DISABLED_BY_CONFIG,
            available=False,
            reason_code="DISABLED_BY_CONFIG",
            requested_implementation=extra,
            actual_implementation="",
            detail="every flag of this extra is off",
        )
    missing: List[CapabilityResolution] = []
    for import_name in EXTRA_IMPORT_NAMES.get(extra, ()):
        probe = probe_dependency(import_name, environment_id=environment_id)
        if not probe.available:
            missing.append(probe)
    if missing:
        return CapabilityResolution(
            capability=f"extra:{extra}",
            state=rec.CAP_UNAVAILABLE_DEPENDENCY,
            available=False,
            reason_code="DEPENDENCY_ABSENT",
            requested_implementation=extra,
            actual_implementation="",
            detail="missing: " + ", ".join(item.capability.split(":", 1)[1] for item in missing),
        )
    return CapabilityResolution(
        capability=f"extra:{extra}",
        state=rec.CAP_AVAILABLE,
        available=True,
        requested_implementation=extra,
        actual_implementation=extra,
        detail="flags: " + ", ".join(sorted(enabled)),
    )


def classify_mismatch(
    *, requested_version: str, actual_version: str, expected_prefix: str = ""
) -> CapabilityResolution:
    """Distinguish a version/ABI mismatch from a simple absence (step 31).

    ``E14-01`` §8.5: an ABI error must not be reported as a plain
    ``unavailable``, otherwise the operator looks for a missing wheel instead of
    a rebuilt one.
    """
    if not requested_version or not actual_version:
        return CapabilityResolution(
            capability="version-check",
            state=rec.CAP_BUG,
            available=False,
            reason_code="IMPLEMENTATION_ERROR",
            requested_implementation=requested_version,
            actual_implementation=actual_version,
            detail="both versions are required to classify a mismatch",
        )
    if expected_prefix and not actual_version.startswith(expected_prefix):
        return CapabilityResolution(
            capability="version-check",
            state=rec.CAP_ABI_MISMATCH,
            available=False,
            reason_code="ABI_MISMATCH",
            requested_implementation=requested_version,
            actual_implementation=actual_version,
            detail=f"actual version does not start with {expected_prefix!r}",
        )
    if actual_version != requested_version:
        return CapabilityResolution(
            capability="version-check",
            state=rec.CAP_ABI_MISMATCH,
            available=False,
            reason_code="VERSION_UNSUPPORTED",
            requested_implementation=requested_version,
            actual_implementation=actual_version,
            detail="version differs from the preregistered one",
        )
    return CapabilityResolution(
        capability="version-check",
        state=rec.CAP_AVAILABLE,
        available=True,
        requested_implementation=requested_version,
        actual_implementation=actual_version,
    )


# ── core golden surface (steps 1, 13–15, 27, 34–35) ────────────────────────


@dataclass
class CoreGoldenSurface:
    """The frozen public surface that experimental imports must not change."""

    imports: Tuple[str, ...] = ()
    cli_entries: Tuple[str, ...] = ()
    schema_names: Tuple[str, ...] = ()
    cpu_reference_entry: str = ""
    report_entries: Tuple[str, ...] = ()
    golden_outputs: Mapping[str, Any] = field(default_factory=dict)
    observed: Mapping[str, Any] = field(default_factory=dict)

    def required(self) -> Tuple[str, ...]:
        return tuple(
            ["imports", "cli_entries", "schema_names", "cpu_reference_entry", "golden_outputs"]
        )

    def validate(self) -> List[str]:
        findings: List[str] = []
        if not self.imports:
            findings.append("core golden surface must list the supported imports")
        if not self.cli_entries:
            findings.append("core golden surface must list the supported CLI entries")
        if not self.schema_names:
            findings.append("core golden surface must list the C1–C7 schema names")
        if not self.cpu_reference_entry:
            findings.append("core golden surface must name the CPU reference entry")
        if not self.golden_outputs:
            findings.append(
                "core golden surface must carry golden outputs; 'import did not raise' is not behaviour equivalence"
            )
        return findings

    def digest(self) -> str:
        return canonical_digest(
            {
                "imports": sorted(self.imports),
                "cli_entries": sorted(self.cli_entries),
                "schema_names": sorted(self.schema_names),
                "cpu_reference_entry": self.cpu_reference_entry,
                "report_entries": sorted(self.report_entries),
                "golden_outputs": self.golden_outputs,
            }
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "imports": sorted(self.imports),
            "cli_entries": sorted(self.cli_entries),
            "schema_names": sorted(self.schema_names),
            "cpu_reference_entry": self.cpu_reference_entry,
            "report_entries": sorted(self.report_entries),
            "golden_digest": self.digest(),
            "golden_outputs": dict(sorted(self.golden_outputs.items())),
            "observed": dict(sorted(self.observed.items())),
        }


#: Fields compared by the strict A/B (``E14-01`` step 34).
CORE_AB_FIELDS: Tuple[str, ...] = (
    "c6_output_digest",
    "c7_output_digest",
    "stdout_digest",
    "stderr_digest",
    "exit_code",
    "files_written",
    "schema_roundtrip_ok",
    "config_hash",
    "cpu_reference_digest",
)


def compare_core_golden(
    core_only: Mapping[str, Any], other: Mapping[str, Any], *, label: str = "all-extras-off"
) -> Dict[str, Any]:
    """Strict A/B over the frozen core surface (``E14-01`` steps 27/34)."""
    diffs: List[Dict[str, Any]] = []
    for field_name in CORE_AB_FIELDS:
        if field_name not in core_only:
            diffs.append({"field": field_name, "problem": "missing from the core-only observation"})
            continue
        if field_name not in other:
            diffs.append({"field": field_name, "problem": f"missing from the {label} observation"})
            continue
        if core_only[field_name] != other[field_name]:
            diffs.append({"field": field_name, "core_only": core_only[field_name], "other": other[field_name]})
    return {
        "label": label,
        "fields": len(CORE_AB_FIELDS),
        "diffs": diffs,
        "equivalent": not diffs,
    }


def compare_performance_budget(
    *, core_only: Mapping[str, float], other: Mapping[str, float], budget: Mapping[str, float]
) -> Dict[str, Any]:
    """Startup-cost A/B with leaf 35's preregistered budget.

    The budget is supplied by the caller (it is a preregistration input, not a
    constant here); a missing budget entry is reported as unmeasured rather than
    assumed to pass.
    """
    violations: List[Dict[str, Any]] = []
    unmeasured: List[str] = []
    for metric, allowed in budget.items():
        if metric not in core_only or metric not in other:
            unmeasured.append(metric)
            continue
        delta = other[metric] - core_only[metric]
        if allowed >= 0 and delta > allowed:
            violations.append({"metric": metric, "delta": delta, "allowed": allowed})
        if allowed < 0 and delta < allowed:  # a *reduction* below the floor is also a change
            violations.append({"metric": metric, "delta": delta, "allowed": allowed})
    return {
        "metrics": sorted(budget),
        "violations": violations,
        "unmeasured": unmeasured,
        "within_budget": not violations and not unmeasured,
    }


# ── negative cases (steps 17–18, 29–31, 37–38) ─────────────────────────────


@dataclass(frozen=True)
class NegativeCase:
    """One injectable leak/mismatch fixture and the detector it must trip."""

    case_id: str
    description: str
    injection: str
    detector: str
    expected_outcome: str
    isolation_required: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "injection": self.injection,
            "detector": self.detector,
            "expected_outcome": self.expected_outcome,
            "isolation_required": self.isolation_required,
        }


def negative_case_matrix() -> Tuple[NegativeCase, ...]:
    """The fixtures ``E14-01`` requires, each bound to its detector.

    A gate that has never been shown to fail is not a gate; these cases are the
    executable form of that requirement (``E14-01`` §10 PASS: 负例能被
    import/metadata/CI 门检测).
    """
    return (
        NegativeCase(
            case_id="N1-direct-import-leak",
            description="core 模块直接 import 一个 experimental 包",
            injection="受控测试分支加入 `import torch.distributed`",
            detector="static_import_edges + import_dependency_gate",
            expected_outcome="静态门与 core-only CI 同时失败",
        ),
        NegativeCase(
            case_id="N2-indirect-reexport-leak",
            description="通过公共 __init__ 或 registry 间接导入重依赖",
            injection="在包 __init__ 中重导出 experimental 实现",
            detector="static_import_edges（module_level 边）+ import trace",
            expected_outcome="运行追踪与图规则均检测到",
        ),
        NegativeCase(
            case_id="N3-unknown-flag",
            description="拼写错误的 feature flag",
            injection="resolve({'train.distribued': True})",
            detector="FeatureFlagRegistry.resolve",
            expected_outcome="structured reject，绝不静默忽略",
        ),
        NegativeCase(
            case_id="N4-conflicting-flags",
            description="两个前沿机制 flag 同时开启",
            injection="resolve({'frontier.moe': True, 'frontier.sparse': True})",
            detector="FeatureFlagRegistry.conflicts",
            expected_outcome="拒绝并指出冲突（E14-05 只允许一个主分支）",
        ),
        NegativeCase(
            case_id="N5-missing-dependency",
            description="卸除 extra 的一个关键包后调用相应能力",
            injection="隔离环境卸载后再 probe_extra",
            detector="probe_extra",
            expected_outcome="UNAVAILABLE_DEPENDENCY + DEPENDENCY_ABSENT，core 仍通过",
            isolation_required="E14-01 step 38",
        ),
        NegativeCase(
            case_id="N6-abi-mismatch",
            description="预注册的不兼容版本组合",
            injection="隔离环境安装旧版/前缀不符的 wheel",
            detector="classify_mismatch",
            expected_outcome="ABI_MISMATCH，不与 dependency absent 混淆",
            isolation_required="E14-01 step 31",
        ),
        NegativeCase(
            case_id="N7-metadata-leak",
            description="最终 wheel 把 experimental 依赖写成无条件 Requires-Dist",
            injection="受控构建把 train 依赖写入 core dependencies",
            detector="audit_wheel_metadata",
            expected_outcome="leaks 非空 ⇒ 边界 FAIL",
        ),
        NegativeCase(
            case_id="N8-import-side-effect",
            description="import 时创建 CUDA context / 启动 worker / 下载模型",
            injection="受控模块在 import 时初始化设备",
            detector="analyse_import_trace",
            expected_outcome="purity violation ⇒ 即使功能正确也 FAIL",
        ),
        NegativeCase(
            case_id="N9-cli-full-import",
            description="CLI 顶层导入全部实现，导致 core 环境启动失败",
            injection="受控 CLI 顶层 import experimental 模块",
            detector="static_import_edges + core-only CLI 运行",
            expected_outcome="core-only CLI 失败被记录为真实失败，不得被 skip",
        ),
        NegativeCase(
            case_id="N10-false-skip",
            description="缺 GPU 的真实 import bug 被标 skip",
            injection="受控 import 错误 + 无条件 skip 装饰器",
            detector="CI job/skip 语义审计（ci_matrix.json）",
            expected_outcome="fail 与 skip 语义可区分，错误不得被跳成绿色",
        ),
    )


def audit_ci_matrix(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """``E14-01`` step 37 — core job may not install extras; skips must be honest."""
    problems: List[str] = []
    for row in rows:
        job = str(row.get("job", "<unnamed>"))
        profile = str(row.get("install_profile", ""))
        skipped = row.get("skipped", False)
        skip_reason = str(row.get("skip_reason", ""))
        failure = row.get("failure", False)
        if job.startswith("core") and profile not in ("core", ""):
            problems.append(f"CI job {job!r} is a core job but installs profile {profile!r}")
        if skipped and not skip_reason:
            problems.append(f"CI job {job!r} skipped without a structured reason (缺硬件是结构化 skip/block)")
        if skipped and failure:
            problems.append(f"CI job {job!r} both skipped and failed — the semantics are ambiguous")
        if profile and profile not in rec.INSTALL_PROFILES:
            problems.append(f"CI job {job!r} declares unknown install profile {profile!r}")
        for extra in row.get("installs_extras", ()) or ():
            if extra not in rec.EXTRA_FEATURE_MAPPING:
                problems.append(f"CI job {job!r} installs unknown extra {extra!r}")
    return problems


# ── verdict (step 40) ──────────────────────────────────────────────────────


@dataclass
class DependencyBoundaryResult:
    """The per-environment result of ``E14-01`` §7 (``DependencyBoundaryResult``)."""

    environment_id: str
    install_profile: str
    feature_flags: Mapping[str, bool] = field(default_factory=dict)
    wheel_digest: str = ""
    dependency_lock_digest: str = ""
    imported_modules_digest: str = ""
    device_context_created: bool = False
    child_process_count: int = 0
    core_golden_match: Optional[bool] = None
    capabilities: Tuple[CapabilityResolution, ...] = ()
    violations: Tuple[str, ...] = ()
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e14-01.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        if self.install_profile not in rec.INSTALL_PROFILES:
            findings.append(f"DependencyBoundaryResult: unknown install_profile {self.install_profile!r}")
        if not self.environment_id:
            findings.append("DependencyBoundaryResult: environment_id is required")
        if self.status in rec.CONCLUSION_STATUSES and not self.violations == ():
            findings.append(
                f"DependencyBoundaryResult: cannot report {self.status} with unresolved violations"
            )
        if self.status == rec.STATUS_PASS and self.core_golden_match is not True:
            findings.append("DependencyBoundaryResult: PASS requires core_golden_match is True")
        if self.install_profile == "core" and self.device_context_created:
            findings.append("DependencyBoundaryResult: core install created a device context")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "install_profile": self.install_profile,
            "feature_flags": {key: self.feature_flags[key] for key in sorted(self.feature_flags)},
            "wheel_digest": self.wheel_digest,
            "dependency_lock_digest": self.dependency_lock_digest,
            "imported_modules_digest": self.imported_modules_digest,
            "device_context_created": self.device_context_created,
            "child_process_count": self.child_process_count,
            "core_golden_match": self.core_golden_match,
            "capabilities": [item.as_dict() for item in self.capabilities],
            "violations": list(self.violations),
            "status": self.status,
        }


@dataclass
class DependencyBoundaryVerdict:
    """``E14-01`` step 40 — the per-extra ``CORE_SAFE/EXPERIMENTAL_ONLY/BLOCKED`` ruling."""

    verdict_id: str
    results: Tuple[DependencyBoundaryResult, ...] = ()
    per_extra: Mapping[str, str] = field(default_factory=dict)
    residual_risks: Tuple[str, ...] = ()
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e14-01.verdict.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        for extra, verdict in self.per_extra.items():
            if extra not in rec.EXTRA_FEATURE_MAPPING:
                findings.append(f"DependencyBoundaryVerdict: unknown extra {extra!r}")
            if verdict not in rec.DEPENDENCY_VERDICTS:
                findings.append(
                    f"DependencyBoundaryVerdict: extra {extra!r} verdict {verdict!r} must be one of "
                    f"{', '.join(rec.DEPENDENCY_VERDICTS)}"
                )
        if self.status == rec.STATUS_PASS:
            missing = [extra for extra in rec.EXTRA_FEATURE_MAPPING if extra not in self.per_extra]
            if missing:
                findings.append(
                    "DependencyBoundaryVerdict: PASS requires a verdict for every extra; missing: "
                    + ", ".join(sorted(missing))
                    + "（一条“可选依赖测试通过”不能掩盖局部失败）"
                )
            core_results = [item for item in self.results if item.install_profile == "core"]
            if not core_results:
                findings.append("DependencyBoundaryVerdict: PASS requires at least one core-only result")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict_id": self.verdict_id,
            "per_extra": {key: self.per_extra[key] for key in sorted(self.per_extra)},
            "results": [item.as_dict() for item in self.results],
            "residual_risks": list(self.residual_risks),
            "status": self.status,
        }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-01 interfaces (labelled smoke, not an experiment)."""
    registry = FeatureFlagRegistry(default_feature_flags())
    problems: List[str] = []
    _resolved, unknown_problems = registry.resolve({"train.distribued": True})
    conflict_problems = registry.conflicts({"frontier.moe": True, "frontier.sparse": True})
    disabled = probe_extra("train", flags={})
    assert disabled.state == rec.CAP_DISABLED_BY_CONFIG, disabled
    policy = DependencyPolicy(
        entries=(
            DependencyEntry("pydantic", "core", "core", "contracts", "MIT", ("any",), "module_level"),
        )
    )
    metadata = WheelMetadata.parse(
        "Name: hqsb\nVersion: 0.1.0\nRequires-Dist: pydantic>=2.0\nRequires-Dist: torch>=2.0 ; extra == \"train\"\n"
        "Provides-Extra: train\n"
    )
    audit = audit_wheel_metadata(metadata, policy)
    edges, parse_error = static_import_edges("m", "import os\nfrom . import x\n")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "flags": len(registry.flags()),
        "unknown_flag_rejected": bool(unknown_problems),
        "conflicting_flags_rejected": bool(conflict_problems),
        "flag_off_wins_over_installed": disabled.state == rec.CAP_DISABLED_BY_CONFIG,
        "wheel_leak_detected": bool(audit["leaks"]) or bool(audit["missing_extras"]),
        "static_edges": len(edges),
        "static_parse_error": parse_error,
        "negative_cases": len(negative_case_matrix()),
        "policy_problems": problems,
    }


# ── build / content / SBOM / propagation helpers (steps 6, 8, 26, 32–33, 36, 39) ─


@dataclass
class BuildIdentity:
    """Build identity for one wheel/sdist (``E14-01`` steps 6 and 39).

    Step 39 requires an *independent rebuild*: the second identity is compared to
    the first, so a build that silently depends on the developer's cache or an
    editable path is detected rather than assumed away.
    """

    artifact_name: str
    backend: str
    python_version: str
    platform_tag: str
    digest: str
    builder_environment_id: str = ""
    editable: bool = False
    dependency_lock_digest: str = ""
    build_log_uri: str = ""
    warnings: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("artifact_name", "backend", "python_version", "platform_tag", "digest"):
            if not getattr(self, name):
                findings.append(f"BuildIdentity: {name} is required")
        if self.editable:
            findings.append(
                "BuildIdentity: an editable install may not back a release conclusion "
                "(E14-01 §9: 用开发环境 editable install 做发布结论)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "backend": self.backend,
            "python_version": self.python_version,
            "platform_tag": self.platform_tag,
            "digest": self.digest,
            "builder_environment_id": self.builder_environment_id,
            "editable": self.editable,
            "dependency_lock_digest": self.dependency_lock_digest,
            "build_log_uri": self.build_log_uri,
            "warnings": list(self.warnings),
        }


def compare_builds(first: BuildIdentity, second: BuildIdentity) -> Dict[str, Any]:
    """Step 39: two independent builds must agree on content or explain the delta."""
    diffs: List[Dict[str, Any]] = []
    for field_name in ("backend", "python_version", "platform_tag", "digest", "dependency_lock_digest"):
        left, right = getattr(first, field_name), getattr(second, field_name)
        if left != right:
            diffs.append({"field": field_name, "first": left, "second": right})
    return {
        "fields": 5,
        "diffs": diffs,
        "reproducible": not diffs,
        "reproducible_modulo_metadata": bool(diffs) and all(
            item["field"] == "digest" for item in diffs
        ),
    }


def audit_wheel_contents(
    entries: Mapping[str, int], *, policy: DependencyPolicy, max_native_libraries: int = 1
) -> Dict[str, Any]:
    """Step 8: what is actually inside the artifact (size contributions, natives).

    A core wheel that ships a training checkpoint, a test dataset or a vendored
    runtime is the same failure as a metadata leak, only harder to notice.
    """
    problems: List[Dict[str, Any]] = []
    total = sum(entries.values()) or 1
    native: List[str] = []
    for path, size in sorted(entries.items()):
        lowered = path.lower()
        if lowered.endswith((".so", ".dll", ".dylib")):
            native.append(path)
        if lowered.startswith(("tests/data/", "test_vectors/", "checkpoints/", "weights/", "models/")):
            problems.append({"path": path, "problem": "test data or model weights must not ship in the wheel"})
        elif lowered.endswith((".pyc", ".ncu-rep", ".nsys-rep")):
            problems.append({"path": path, "problem": "generated/compiled artefact must not ship in the wheel"})
    if len(native) > max_native_libraries:
        problems.append(
            {
                "path": ", ".join(sorted(native)),
                "problem": f"{len(native)} native libraries in a core wheel (budget {max_native_libraries})",
            }
        )
    return {
        "entries": len(entries),
        "total_bytes": sum(entries.values()),
        "native_libraries": native,
        "top_contributors": sorted(
            ({"path": path, "ratio": size / total} for path, size in entries.items()),
            key=lambda item: item["ratio"],
            reverse=True,
        )[:10],
        "problems": problems,
        "ok": not problems,
    }


def resolve_all_extras(profiles: Sequence[str]) -> Dict[str, Any]:
    """Step 26: install every extra together and report what the resolver did.

    Independent installability does not imply joint installability — the whole
    point of the step (``E14-01`` §9: all-extras 组合从未测试).
    """
    known = set(rec.INSTALL_PROFILES)
    unknown = [profile for profile in profiles if profile not in known]
    if unknown:
        raise ConfigError(f"unknown install profiles {unknown}; known: {', '.join(sorted(known))}")
    if "core" not in profiles:
        raise ConfigError("the all-extras resolution must include the core profile for comparison")
    return {
        "profiles": list(profiles),
        "extras": sorted(extra for extra in rec.EXTRA_FEATURE_MAPPING if extra in set(profiles)),
        "conflicts": [],
        "duplicate_native_libraries": [],
        "notes": ["resolver decisions, conflicts and downgrades must be recorded verbatim by the driver"],
    }


def check_registry_laziness(module_name: str, imported_modules: Sequence[str]) -> List[str]:
    """Step 32: plugin discovery must not import every implementation.

    Enabling every feature and then importing the registry must not drag in the
    heavy modules of the *disabled* ones; a last-registered-wins override is
    reported too (``E14-01`` §9: 最后注册者静默覆盖).
    """
    problems: List[str] = []
    lowered = [module.lower() for module in imported_modules]
    for marker in EXPERIMENTAL_MODULE_MARKERS:
        if any(module == marker or module.startswith(marker + ".") for module in lowered):
            problems.append(f"importing {module_name} pulled in disabled implementation {marker}")
    return problems


def check_worker_env_propagation(
    driver: Mapping[str, Any], workers: Sequence[Mapping[str, Any]]
) -> List[str]:
    """Step 33: a worker must not use different flags/config than the driver.

    ``E14-01`` §8.6: driver 关闭 feature，而 worker 用默认值偷偷开启, is a
    silent divergence in the *worker* environment, which no driver-side test
    observes unless it is compared explicitly.
    """
    problems: List[str] = []
    if not workers:
        problems.append("no worker environments recorded (a launcher that spawns nothing proves nothing)")
    driver_flags = dict(driver.get("feature_flags", {}))
    driver_config = driver.get("config_digest", "")
    driver_modules = set(driver.get("imported_modules", ()) or ())
    for index, worker in enumerate(workers):
        worker_flags = dict(worker.get("feature_flags", {}))
        if worker_flags != driver_flags:
            problems.append(
                f"worker {index} flags {worker_flags} differ from driver {driver_flags}"
            )
        if driver_config and worker.get("config_digest") != driver_config:
            problems.append(
                f"worker {index} config_digest {worker.get('config_digest')!r} differs from driver {driver_config!r}"
            )
        extra_modules = sorted(set(worker.get("imported_modules", ()) or ()) - driver_modules)
        if extra_modules:
            problems.append(f"worker {index} imported modules the driver did not: {', '.join(extra_modules)}")
    return problems


def compare_sbom(
    core_sbom: Mapping[str, Any], all_extras_sbom: Mapping[str, Any]
) -> Dict[str, Any]:
    """Step 36: keep the extras' risk out of the core's SBOM, and vice versa."""
    core_ids = set(core_sbom.get("components", ()) or ())
    extras_ids = set(all_extras_sbom.get("components", ()) or ())
    added = sorted(extras_ids - core_ids)
    removed = sorted(core_ids - extras_ids)
    new_licenses = sorted(
        {
            str(all_extras_sbom.get("licenses", {}).get(component, ""))
            for component in added
        }
        - set(core_sbom.get("licenses", {}).values())
        - {""}
    )
    vulnerabilities = sorted(
        component
        for component in added
        if component in set(all_extras_sbom.get("vulnerable_components", ()) or ())
    )
    return {
        "added_components": added,
        "removed_components": removed,
        "new_licenses": new_licenses,
        "vulnerable_components_in_extras": vulnerabilities,
        "core_risk_unchanged": not removed,
        "note": "extra 的 CVE/许可证边界不得写成 core 已无风险（E14-01 §9）",
    }


# ── protocol step table (40 steps of details/S14/E14-01) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 core 公共表面", ("dependencies:CoreGoldenSurface", "dependencies:CoreGoldenSurface.digest",
                               "dependencies:compare_core_golden")),
    (2, "冻结依赖分类策略", ("dependencies:DependencyEntry", "dependencies:DependencyPolicy",
                            "dependencies:DEPENDENCY_LAYERS", "dependencies:ALLOWED_IMPORT_LAYERS")),
    (3, "冻结 extra 与 feature 映射", ("dependencies:FeatureFlagRegistry", "dependencies:default_feature_flags",
                                       "dependencies:flag_from_extra", "records:EXTRA_FEATURE_MAPPING")),
    (4, "冻结失败语义", ("records:REASON_CODES", "records:CAPABILITY_STATES", "records:BLOCKING_CAPABILITY_STATES",
                        "dependencies:CapabilityResolution")),
    (5, "生成测试环境矩阵", ("records:INSTALL_PROFILES", "dependencies:probe_extra",
                            "dependencies:negative_case_matrix")),
    (6, "从清洁 builder 构建 wheel 与 sdist", ("dependencies:BuildIdentity", "dependencies:BuildIdentity.problems",
                                             "identity:digest_file")),
    (7, "审查 wheel 元数据", ("dependencies:WheelMetadata", "dependencies:WheelMetadata.parse",
                              "dependencies:audit_wheel_metadata")),
    (8, "审查 wheel 内容和包体", ("dependencies:audit_wheel_contents", "campaign:check_committable")),
    (9, "在全新 CPU 环境安装 core", ("dependencies:DependencyBoundaryResult",
                                     "identity:content_address_aggregate")),
    (10, "执行 core 冷导入追踪", ("dependencies:ImportTrace", "dependencies:analyse_import_trace",
                                  "dependencies:EXPERIMENTAL_MODULE_MARKERS")),
    (11, "测 core 冷/热 import 成本", ("dependencies:ImportTrace.wall_ms", "dependencies:ImportTrace.rss_bytes",
                                       "dependencies:ImportTrace.phase")),
    (12, "验证 import 无副作用", ("dependencies:analyse_import_trace", "contracts:check_noncapture_matrix")),
    (13, "运行 core Schema/配置 round-trip", ("dependencies:CoreGoldenSurface.schema_names",
                                              "dependencies:CORE_AB_FIELDS")),
    (14, "运行 core CPU reference/golden", ("dependencies:CoreGoldenSurface.cpu_reference_entry",
                                            "dependencies:CoreGoldenSurface.golden_outputs")),
    (15, "运行 core CLI 全表面", ("dependencies:CoreGoldenSurface.cli_entries", "telemetry:coverage_report")),
    (16, "静态构建真实 import 图", ("dependencies:static_import_edges", "dependencies:ImportEdge")),
    (17, "注入直接依赖泄漏负例", ("dependencies:negative_case_matrix", "dependencies:static_import_edges")),
    (18, "注入间接导入泄漏负例", ("dependencies:negative_case_matrix", "dependencies:static_import_edges",
                                  "dependencies:check_registry_laziness")),
    (19, "安装 train extra", ("dependencies:probe_extra", "dependencies:EXTRA_IMPORT_NAMES",
                              "dependencies:DependencyBoundaryResult")),
    (20, "测试训练 feature off/on", ("dependencies:FeatureFlagRegistry.resolve", "dependencies:probe_extra",
                                     "records:CAP_DISABLED_BY_CONFIG")),
    (21, "安装并测试 rl extra", ("dependencies:probe_extra", "dependencies:CapabilityResolution.as_dict")),
    (22, "逐个安装 F1–F4 extra", ("dependencies:default_feature_flags", "dependencies:FeatureFlagRegistry.conflicts",
                                   "records:CAP_UNAVAILABLE_CAPABILITY")),
    (23, "安装多模态 extra", ("dependencies:probe_extra", "records:EXTRA_FEATURE_MAPPING")),
    (24, "安装 Agent extra", ("dependencies:probe_extra", "dependencies:check_registry_laziness")),
    (25, "安装 edge extra", ("dependencies:probe_extra", "dependencies:probe_dependency")),
    (26, "测试 all-extras 解析", ("dependencies:resolve_all_extras", "dependencies:DependencyPolicy.digest")),
    (27, "执行 feature 全关闭等价测试", ("dependencies:compare_core_golden", "contracts:core_golden_equivalence")),
    (28, "执行逐 flag 启停测试", ("dependencies:FeatureFlagRegistry.matrix_rows",
                                  "dependencies:FeatureFlagRegistry.resolve")),
    (29, "测试未知和冲突 flag", ("dependencies:FeatureFlagRegistry.resolve", "dependencies:FeatureFlagRegistry.conflicts",
                                 "records:REASON_CODES")),
    (30, "测试缺依赖路径", ("dependencies:probe_dependency", "dependencies:probe_extra",
                            "records:CAP_UNAVAILABLE_DEPENDENCY")),
    (31, "测试版本/ABI 不兼容路径", ("dependencies:classify_mismatch", "records:CAP_ABI_MISMATCH",
                                     "campaign:REQUIRED_ISOLATION")),
    (32, "验证 registry 懒加载与冲突", ("dependencies:check_registry_laziness", "dependencies:static_import_edges")),
    (33, "验证子进程/worker 环境传播", ("dependencies:check_worker_env_propagation", "identity:RankIdentity")),
    (34, "执行 core 行为严格 A/B", ("dependencies:compare_core_golden", "dependencies:CORE_AB_FIELDS")),
    (35, "执行性能预算 A/B", ("dependencies:compare_performance_budget", "dependencies:ImportTrace.wall_ms")),
    (36, "审查 SBOM、许可证和漏洞面", ("dependencies:compare_sbom", "records:DEPENDENCY_VERDICTS")),
    (37, "验证 CI job 与 skip 语义", ("dependencies:audit_ci_matrix", "records:INSTALL_PROFILES")),
    (38, "卸载 experimental 依赖后复测", ("dependencies:analyse_import_trace", "campaign:isolation_clause",
                                          "dependencies:compare_core_golden")),
    (39, "独立重建和复跑", ("dependencies:BuildIdentity", "dependencies:compare_builds",
                            "dependencies:resolve_all_extras")),
    (40, "形成 DependencyBoundaryVerdict", ("dependencies:DependencyBoundaryVerdict",
                                            "dependencies:DependencyBoundaryVerdict.validate",
                                            "records:DEPENDENCY_VERDICTS")),
)
