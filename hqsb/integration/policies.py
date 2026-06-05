"""Frozen integration specs with strict YAML loaders (E06 pre-registration).

Every experiment in S06 starts by freezing a spec ("步骤1: 冻结 …Spec").  This
module is where those documents live: compile/dynamic policy, cache spec, CUDA
Graph spec, resource spec, reuse spec, pattern declarations and the operator
contract fields C3 does not carry.

Loading rules follow the repository's existing convention
(:mod:`hqsb.quant.config_io`): a document declares its ``kind``, unknown keys are
refused rather than ignored, and every returned object is validated.  A typo
must never keep a default silently — that is the same failure mode as a silent
fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.integration import cache as cache_mod
from hqsb.integration import cuda_graph as graph_mod
from hqsb.integration import guards as guards_mod
from hqsb.integration import lifecycle as lifecycle_mod
from hqsb.integration import adapter as adapter_mod

KIND_COMPILE_POLICY = "hqsb.integration.compile_policy"
KIND_CACHE_SPEC = "hqsb.integration.cache_spec"
KIND_GRAPH_SPEC = "hqsb.integration.graph_spec"
KIND_RESOURCE_SPEC = "hqsb.integration.resource_spec"
KIND_REUSE_SPEC = "hqsb.integration.reuse_spec"
KIND_PATTERN_SPECS = "hqsb.integration.pattern_specs"
KIND_OPERATOR_CONTRACTS = "hqsb.integration.operator_contracts"

KINDS = (
    KIND_COMPILE_POLICY,
    KIND_CACHE_SPEC,
    KIND_GRAPH_SPEC,
    KIND_RESOURCE_SPEC,
    KIND_REUSE_SPEC,
    KIND_PATTERN_SPECS,
    KIND_OPERATOR_CONTRACTS,
)


def load_yaml_document(path: str) -> Dict[str, Any]:
    """Load a YAML mapping and refuse an unknown/missing ``kind``."""
    import yaml

    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: YAML root must be a mapping")
    kind = payload.get("kind")
    if kind not in KINDS:
        raise ConfigError(
            f"{path}: kind {kind!r} is not one of {list(KINDS)}"
        )
    return payload


def _reject_unknown(path: str, payload: Mapping[str, Any], allowed: Sequence[str]) -> None:
    unknown = set(payload) - set(allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")


# ── compile policy ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompilePolicyDocument:
    """Dynamic-shape policy, buckets, storm thresholds and fallback settings."""

    name: str
    policy: str
    dimensions: Tuple[guards_mod.DynamicDimensionSpec, ...]
    storm: guards_mod.StormThresholds
    fallback_enabled: bool = True
    strict: bool = False
    recompile_limit: Optional[int] = None
    capture_mode: str = "dynamo"
    notes: str = ""

    @property
    def compiler_config(self) -> guards_mod.CompilerConfig:
        return guards_mod.resolve_compiler_config(
            self.policy,
            self.dimensions,
            recompile_limit=self.recompile_limit,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "policy": self.policy,
            "dimensions": [item.as_dict() for item in self.dimensions],
            "storm": self.storm.as_dict(),
            "fallback_enabled": self.fallback_enabled,
            "strict": self.strict,
            "recompile_limit": self.recompile_limit,
            "capture_mode": self.capture_mode,
            "notes": self.notes,
            "resolved": self.compiler_config.as_dict(),
        }


def load_compile_policy(path: str) -> CompilePolicyDocument:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_COMPILE_POLICY:
        raise ConfigError(
            f"{path}: expected kind {KIND_COMPILE_POLICY!r}, got {payload['kind']!r}"
        )
    _reject_unknown(
        path,
        payload,
        (
            "kind",
            "description",
            "name",
            "policy",
            "capture_mode",
            "dimensions",
            "storm",
            "fallback_enabled",
            "strict",
            "recompile_limit",
            "notes",
        ),
    )
    dimensions = tuple(
        guards_mod.DynamicDimensionSpec(
            name=str(item["name"]),
            symbol=str(item["symbol"]),
            bounds=tuple(item["bounds"]),
            buckets=tuple(item.get("buckets") or ()),
        )
        for item in payload.get("dimensions") or ()
    )
    storm_payload = dict(payload.get("storm") or {})
    storm = guards_mod.StormThresholds(**storm_payload)
    document = CompilePolicyDocument(
        name=str(payload.get("name", os.path.basename(path))),
        policy=str(payload["policy"]),
        dimensions=dimensions,
        storm=storm,
        fallback_enabled=bool(payload.get("fallback_enabled", True)),
        strict=bool(payload.get("strict", False)),
        recompile_limit=payload.get("recompile_limit"),
        capture_mode=str(payload.get("capture_mode", "dynamo")),
        notes=str(payload.get("notes", "")),
    )
    if document.strict and document.fallback_enabled:
        raise ConfigError(
            f"{path}: strict mode forbids fallback, but fallback_enabled is true; "
            "one of the two must change",
            details={"field": "strict"},
        )
    return document


# ── cache spec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CacheSpecDocument:
    """Cache layers, states and pre-registered invalidation expectations."""

    name: str
    cache_root: str
    states: Tuple[str, ...]
    layers: Tuple[str, ...]
    cap_entries: int = 128
    cap_bytes: int = 512 * 1024 * 1024
    invalidation: Tuple[cache_mod.InvalidationFactor, ...] = ()
    imported_remote_claimed: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        for state in self.states:
            if state not in cache_mod.CacheState.ALL:
                raise ConfigError(
                    f"{self.name}: unknown cache state {state!r}",
                    details={"field": "states", "allowed": list(cache_mod.CacheState.ALL)},
                )
        for layer in self.layers:
            if layer not in cache_mod.CacheLayer.ALL:
                raise ConfigError(
                    f"{self.name}: unknown cache layer {layer!r}",
                    details={"field": "layers", "allowed": list(cache_mod.CacheLayer.ALL)},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cache_root": self.cache_root,
            "states": list(self.states),
            "layers": list(self.layers),
            "cap_entries": self.cap_entries,
            "cap_bytes": self.cap_bytes,
            "invalidation": [item.as_dict() for item in self.invalidation],
            "imported_remote_claimed": self.imported_remote_claimed,
            "notes": self.notes,
        }


def load_cache_spec(path: str) -> CacheSpecDocument:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_CACHE_SPEC:
        raise ConfigError(
            f"{path}: expected kind {KIND_CACHE_SPEC!r}, got {payload['kind']!r}"
        )
    _reject_unknown(
        path,
        payload,
        (
            "kind",
            "description",
            "name",
            "cache_root",
            "states",
            "layers",
            "cap_entries",
            "cap_bytes",
            "invalidation",
            "imported_remote_claimed",
            "notes",
        ),
    )
    invalidation = (
        tuple(
            cache_mod.InvalidationFactor(
                name=str(item["factor"]),
                expected_action=str(item["expected_action"]),
                semantic=bool(item.get("semantic", True)),
                note=str(item.get("note", "")),
            )
            for item in payload.get("invalidation") or ()
        )
        or cache_mod.default_invalidation_matrix()
    )
    document = CacheSpecDocument(
        name=str(payload.get("name", os.path.basename(path))),
        cache_root=str(payload.get("cache_root", "experiment_results/cache")),
        states=tuple(str(item) for item in payload.get("states") or cache_mod.CacheState.ALL),
        layers=tuple(str(item) for item in payload.get("layers") or cache_mod.CacheLayer.ALL),
        cap_entries=int(payload.get("cap_entries", 128)),
        cap_bytes=int(payload.get("cap_bytes", 512 * 1024 * 1024)),
        invalidation=invalidation,
        imported_remote_claimed=bool(payload.get("imported_remote_claimed", False)),
        notes=str(payload.get("notes", "")),
    )
    if document.imported_remote_claimed and cache_mod.CacheState.IMPORTED_REMOTE not in document.states:
        raise ConfigError(
            f"{path}: imported/remote cache is claimed but its state is not declared",
            details={"field": "states"},
        )
    return document


# ── graph spec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphSpecDocument:
    """Capture scope, buckets, output contract and failure expectations."""

    name: str
    spec: graph_mod.GraphSpec
    claim_cuda_graph: bool = False
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec.as_dict(),
            "claim_cuda_graph": self.claim_cuda_graph,
            "notes": self.notes,
        }


def load_graph_spec(path: str) -> GraphSpecDocument:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_GRAPH_SPEC:
        raise ConfigError(
            f"{path}: expected kind {KIND_GRAPH_SPEC!r}, got {payload['kind']!r}"
        )
    _reject_unknown(
        path,
        payload,
        (
            "kind",
            "description",
            "name",
            "capture_scope",
            "buckets",
            "output_contract",
            "out_of_bucket_policy",
            "max_graphs",
            "stream",
            "includes_input_copy",
            "includes_output_copy",
            "allow_recapture",
            "claim_cuda_graph",
            "notes",
        ),
    )
    buckets = tuple(
        graph_mod.BucketSpec(
            name=str(item["name"]),
            dims=dict(item["dims"]),
            padding_cost=dict(item.get("padding_cost") or {}),
        )
        for item in payload.get("buckets") or ()
    )
    spec = graph_mod.GraphSpec(
        capture_scope=str(payload["capture_scope"]),
        buckets=buckets,
        output_contract=str(payload.get("output_contract", graph_mod.OutputContract.REUSED_BUFFER)),
        out_of_bucket_policy=str(
            payload.get("out_of_bucket_policy", graph_mod.OutOfBucketPolicy.FALLBACK_NON_GRAPH)
        ),
        max_graphs=int(payload.get("max_graphs", 4)),
        stream=str(payload.get("stream", "capture_stream")),
        includes_input_copy=bool(payload.get("includes_input_copy", True)),
        includes_output_copy=bool(payload.get("includes_output_copy", True)),
        allow_recapture=bool(payload.get("allow_recapture", True)),
    )
    return GraphSpecDocument(
        name=str(payload.get("name", os.path.basename(path))),
        spec=spec,
        claim_cuda_graph=bool(payload.get("claim_cuda_graph", False)),
        notes=str(payload.get("notes", "")),
    )


# ── resource spec ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResourceSpecDocument:
    """Tracked resources with owner, caps and leak thresholds."""

    name: str
    resources: Tuple[lifecycle_mod.ResourceSpec, ...]
    warmup_cycles: int = 1
    steady_cycles: int = 0
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "warmup_cycles": self.warmup_cycles,
            "steady_cycles": self.steady_cycles,
            "resources": [item.as_dict() for item in self.resources],
            "notes": self.notes,
        }


def load_resource_spec(path: str) -> ResourceSpecDocument:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_RESOURCE_SPEC:
        raise ConfigError(
            f"{path}: expected kind {KIND_RESOURCE_SPEC!r}, got {payload['kind']!r}"
        )
    _reject_unknown(
        path,
        payload,
        (
            "kind",
            "description",
            "name",
            "resources",
            "warmup_cycles",
            "steady_cycles",
            "notes",
        ),
    )
    entries = payload.get("resources")
    if not entries:
        raise ConfigError(
            f"{path}: resource spec must list at least one resource",
            details={"field": "resources"},
        )
    resources = tuple(
        lifecycle_mod.ResourceSpec(
            name=str(item["name"]),
            resource_class=str(item["resource_class"]),
            owner=str(item["owner"]),
            create_span=str(item.get("create_span", "create")),
            destroy_span=str(item.get("destroy_span", "destroy")),
            cache_cap_entries=item.get("cache_cap_entries"),
            reclaim=str(item.get("reclaim", "released")),
            growth_threshold_per_cycle=float(item.get("growth_threshold_per_cycle", 0.0)),
        )
        for item in entries
    )
    return ResourceSpecDocument(
        name=str(payload.get("name", os.path.basename(path))),
        resources=resources,
        warmup_cycles=int(payload.get("warmup_cycles", 1)),
        steady_cycles=int(payload.get("steady_cycles", 0)),
        notes=str(payload.get("notes", "")),
    )


# ── reuse spec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReuseSpecDocument:
    """Second-model / dummy-backend route and the allowed adapter boundary."""

    name: str
    targets: Tuple[adapter_mod.AdapterRegistration, ...]
    route: str = ""
    pass_conditions: Tuple[str, ...] = ()
    core_modification_policy: str = (
        "generic bug fixes and contract extensions are allowed and must be "
        "classified; model/backend-specific leaks inside core fail the experiment"
    )
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "targets": [item.as_dict() for item in self.targets],
            "route": self.route,
            "pass_conditions": list(self.pass_conditions),
            "core_modification_policy": self.core_modification_policy,
            "notes": self.notes,
        }


def load_reuse_spec(path: str) -> ReuseSpecDocument:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_REUSE_SPEC:
        raise ConfigError(
            f"{path}: expected kind {KIND_REUSE_SPEC!r}, got {payload['kind']!r}"
        )
    _reject_unknown(
        path,
        payload,
        (
            "kind",
            "description",
            "name",
            "targets",
            "route",
            "pass_conditions",
            "core_modification_policy",
            "notes",
        ),
    )
    entries = payload.get("targets")
    if not entries:
        raise ConfigError(
            f"{path}: reuse spec must declare at least one target "
            "(second model or dummy backend)",
            details={"field": "targets"},
        )
    targets = tuple(
        adapter_mod.AdapterRegistration(
            target=str(item["target"]),
            kind=str(item["kind"]),
            adapter_path=str(item["adapter_path"]),
            config_path=str(item.get("config_path", "")),
            version=str(item.get("version", "1.0.0")),
            changes_global_default=bool(item.get("changes_global_default", False)),
        )
        for item in entries
    )
    return ReuseSpecDocument(
        name=str(payload.get("name", os.path.basename(path))),
        targets=targets,
        route=str(payload.get("route", "")),
        pass_conditions=tuple(str(item) for item in payload.get("pass_conditions") or ()),
        core_modification_policy=str(
            payload.get(
                "core_modification_policy",
                "generic bug fixes and contract extensions are allowed and must be "
                "classified; model/backend-specific leaks inside core fail the experiment",
            )
        ),
        notes=str(payload.get("notes", "")),
    )


# ── pattern declarations and operator contracts ───────────────────────────


@dataclass(frozen=True)
class PatternDeclaration:
    pattern_id: str
    version: str
    ir_level: str
    lowering_targets: Tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "version": self.version,
            "ir_level": self.ir_level,
            "lowering_targets": list(self.lowering_targets),
            "note": self.note,
        }


@dataclass(frozen=True)
class PatternSpecsDocument:
    name: str
    patterns: Tuple[PatternDeclaration, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "patterns": [item.as_dict() for item in self.patterns]}


def load_pattern_specs(path: str) -> PatternSpecsDocument:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_PATTERN_SPECS:
        raise ConfigError(
            f"{path}: expected kind {KIND_PATTERN_SPECS!r}, got {payload['kind']!r}"
        )
    _reject_unknown(path, payload, ("kind", "description", "name", "patterns"))
    entries = payload.get("patterns")
    if not entries:
        raise ConfigError(
            f"{path}: pattern spec must declare at least one pattern",
            details={"field": "patterns"},
        )
    return PatternSpecsDocument(
        name=str(payload.get("name", os.path.basename(path))),
        patterns=tuple(
            PatternDeclaration(
                pattern_id=str(item["pattern_id"]),
                version=str(item["version"]),
                ir_level=str(item["ir_level"]),
                lowering_targets=tuple(str(entry) for entry in item.get("lowering_targets") or ()),
                note=str(item.get("note", "")),
            )
            for item in entries
        ),
    )


def audit_pattern_declarations(document: PatternSpecsDocument) -> Dict[str, Any]:
    """The YAML declaration and the code's frozen patterns must agree."""
    from hqsb.integration import patterns as patterns_mod

    code = {(spec.pattern_id, spec.version): spec for spec in patterns_mod.frozen_patterns()}
    declared = {(item.pattern_id, item.version): item for item in document.patterns}
    missing_in_code = sorted(set(declared) - set(code))
    missing_in_config = sorted(set(code) - set(declared))
    mismatched = [
        {
            "pattern": key,
            "config_ir_level": declared[key].ir_level,
            "code_ir_level": code[key].ir_level,
        }
        for key in sorted(set(code) & set(declared))
        if declared[key].ir_level != code[key].ir_level
    ]
    return {
        "ok": not (missing_in_code or missing_in_config or mismatched),
        "missing_in_code": [list(item) for item in missing_in_code],
        "missing_in_config": [list(item) for item in missing_in_config],
        "ir_level_mismatch": mismatched,
    }


@dataclass(frozen=True)
class OperatorContractsDocument:
    """The contract fields C3 does not express, per operator (E06-01 §3)."""

    name: str
    contracts: Mapping[str, Mapping[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "contracts": {key: dict(value) for key, value in self.contracts.items()},
        }


def load_operator_contracts(path: str) -> OperatorContractsDocument:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_OPERATOR_CONTRACTS:
        raise ConfigError(
            f"{path}: expected kind {KIND_OPERATOR_CONTRACTS!r}, got {payload['kind']!r}"
        )
    _reject_unknown(path, payload, ("kind", "description", "name", "contracts"))
    contracts = payload.get("contracts")
    if not contracts:
        raise ConfigError(
            f"{path}: at least one operator contract is required",
            details={"field": "contracts"},
        )
    for key, value in contracts.items():
        if not isinstance(value, dict) or "spec" not in value or "contract" not in value:
            raise ConfigError(
                f"{path}: contract {key!r} must contain both 'spec' (C3 fields) and "
                "'contract' (mutation/alias/policy)",
                details={"field": "contracts"},
            )
    return OperatorContractsDocument(
        name=str(payload.get("name", os.path.basename(path))),
        contracts={str(key): dict(value) for key, value in contracts.items()},
    )


def contract_audit(document: OperatorContractsDocument) -> Dict[str, Any]:
    """Compare the config's contract fields with the code's frozen schemas.

    The C3 *projection* in the YAML is not required to reproduce the code schema
    byte-for-byte (the authoritative argument list lives in code, and duplicating
    it in YAML would only create a second source of drift).  What **must** agree
    is the part the config exists to freeze: mutation, alias, autograd/autocast
    policy, inference-only flag, and the tensor argument names.
    """
    from hqsb.integration import specs as specs_mod

    built = {}
    for qualified_name, entry in document.contracts.items():
        schema = specs_mod.schema_from_operator_spec(entry["spec"], entry["contract"])
        if schema.name != qualified_name:
            raise ConfigError(
                f"contract key {qualified_name!r} does not match the spec name "
                f"{schema.name!r}",
                details={"field": "contracts"},
            )
        schema.validate()
        built[schema.name] = schema
    frozen = {item.name: item for item in specs_mod.frozen_schemas()}
    problems: List[Dict[str, Any]] = []
    for name, schema in sorted(built.items()):
        code = frozen.get(name)
        if code is None:
            continue
        checks = (
            ("mutation", code.mutation_signature(), schema.mutation_signature()),
            ("alias", code.alias_signature(), schema.alias_signature()),
            ("autograd_policy", code.autograd_policy, schema.autograd_policy),
            ("autocast_policy", code.autocast_policy, schema.autocast_policy),
            ("inference_only", code.inference_only, schema.inference_only),
            (
                "tensor_args",
                tuple(arg.name for arg in code.args if arg.kind.startswith("Tensor")),
                tuple(arg.name for arg in schema.args if arg.kind.startswith("Tensor")),
            ),
        )
        for field_name, expected, actual in checks:
            if expected != actual:
                problems.append(
                    {
                        "operator": name,
                        "field": field_name,
                        "code": expected,
                        "config": actual,
                    }
                )
    missing = sorted(set(frozen) - set(built))
    return {
        "ok": not problems and not missing,
        "operators": sorted(built),
        "problems": problems,
        "missing_in_config": missing,
    }


def build_schemas(document: OperatorContractsDocument) -> Tuple[Any, ...]:
    """Build and audit schemas from a contracts document (fails on drift)."""
    from hqsb.integration import specs as specs_mod

    audit = contract_audit(document)
    if not audit["ok"]:
        raise ConfigError(
            "operator contract config drifted from the frozen code schema: "
            + "; ".join(
                f"{item['operator']}.{item['field']}" for item in audit["problems"]
            )
            + (f"; missing {audit['missing_in_config']}" if audit["missing_in_config"] else ""),
            details=audit,
        )
    return tuple(
        specs_mod.schema_from_operator_spec(entry["spec"], entry["contract"])
        for entry in document.contracts.values()
    )


# ── aggregate loading ─────────────────────────────────────────────────────


def load_any(path: str) -> Dict[str, Any]:
    """Load by declared kind and return the validated object."""
    payload = load_yaml_document(path)
    kind = payload["kind"]
    loaders = {
        KIND_COMPILE_POLICY: load_compile_policy,
        KIND_CACHE_SPEC: load_cache_spec,
        KIND_GRAPH_SPEC: load_graph_spec,
        KIND_RESOURCE_SPEC: load_resource_spec,
        KIND_REUSE_SPEC: load_reuse_spec,
        KIND_PATTERN_SPECS: load_pattern_specs,
        KIND_OPERATOR_CONTRACTS: load_operator_contracts,
    }
    return {"path": path, "kind": kind, "object": loaders[kind](path)}


def load_directory(directory: str) -> List[Dict[str, Any]]:
    """Load every ``*.yaml`` in a directory (used by the driver's self-check)."""
    if not os.path.isdir(directory):
        raise ConfigError(f"config directory not found: {directory}")
    documents: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith((".yaml", ".yml")):
            documents.append(load_any(os.path.join(directory, name)))
    if not documents:
        raise ConfigError(f"no YAML documents found in {directory}")
    return documents


def default_config_dir(root: str) -> str:
    return os.path.join(root, "configs", "integration")


__all__ = [
    "KINDS",
    "KIND_CACHE_SPEC",
    "KIND_COMPILE_POLICY",
    "KIND_GRAPH_SPEC",
    "KIND_OPERATOR_CONTRACTS",
    "KIND_PATTERN_SPECS",
    "KIND_RESOURCE_SPEC",
    "KIND_REUSE_SPEC",
    "CacheSpecDocument",
    "CompilePolicyDocument",
    "GraphSpecDocument",
    "OperatorContractsDocument",
    "PatternDeclaration",
    "PatternSpecsDocument",
    "ResourceSpecDocument",
    "ReuseSpecDocument",
    "audit_pattern_declarations",
    "build_schemas",
    "default_config_dir",
    "load_any",
    "load_cache_spec",
    "load_compile_policy",
    "load_directory",
    "load_graph_spec",
    "load_operator_contracts",
    "load_pattern_specs",
    "load_resource_spec",
    "load_reuse_spec",
    "load_yaml_document",
]
