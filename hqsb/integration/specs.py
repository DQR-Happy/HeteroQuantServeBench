"""Operator schema as a compile contract (E06-01 §3, §4, §10).

The schema is not documentation: the graph system derives value dependencies,
aliasing and write effects from it.  A kernel that writes ``residual`` in place
while the schema claims functional behaviour lets Dynamo/Inductor reorder or
delete reads and produce silently wrong results (E06-01 §7).

This module therefore freezes, per operator:

* namespace/name/overload and the argument/return signature;
* mutation and alias annotations (and refuses inconsistent combinations);
* inference-only / autograd / autocast policy;
* the dispatch requirements the implementation must satisfy.

It also provides the *single schema owner* audit: exactly one definition site
may own a qualified operator name (Python ``torch.library`` **or** C++
``TORCH_LIBRARY``), everything else must be an implementation/fragment.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, SchemaError

SCHEMA_VERSION = "1.0.0"

#: Operator namespace reserved for HQSB (E06-01 §10).
NAMESPACE = "hqsb"

OP_RMS_NORM = f"{NAMESPACE}::rms_norm"
OP_FUSED_ADD_RMS_NORM = f"{NAMESPACE}::fused_add_rms_norm"
OP_DEQUANT_LINEAR = f"{NAMESPACE}::dequant_linear"

#: The P0 registration set (E06-01 §2): two operators plus one quant/second
#: hotspot boundary.
P0_OPERATORS: Tuple[str, ...] = (
    OP_RMS_NORM,
    OP_FUSED_ADD_RMS_NORM,
    OP_DEQUANT_LINEAR,
)

# ── argument / return / mutation / alias specs ────────────────────────────

ARG_KINDS = (
    "Tensor",
    "Tensor?",
    "int",
    "int?",
    "float",
    "float?",
    "bool",
    "str",
    "str?",
)


@dataclass(frozen=True)
class ArgSpec:
    """One schema argument."""

    name: str
    kind: str
    default: Optional[Any] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ARG_KINDS:
            raise SchemaError(
                f"argument {self.name!r}: unknown kind {self.kind!r}; "
                f"allowed={list(ARG_KINDS)}",
                details={"field": "kind", "expected": list(ARG_KINDS), "actual": self.kind},
            )
        if self.kind.endswith("?") and self.default is None:
            # Optional arguments carry an explicit default in the schema text so
            # the graph system knows whether calling with fewer args is legal.
            object.__setattr__(self, "default", _default_for(self.kind))

    @property
    def optional(self) -> bool:
        return self.kind.endswith("?")

    def render(self) -> str:
        if self.optional and self.default is not None:
            return f"{self.kind.rstrip('?')} {self.name}={self.default!r}"
        return f"{self.kind} {self.name}"


def _default_for(kind: str) -> Any:
    return {
        "Tensor?": None,
        "int?": None,
        "float?": None,
        "str?": None,
        "int": 0,
        "float": 0.0,
        "bool": False,
        "str": "",
    }.get(kind)


@dataclass(frozen=True)
class MutationSpec:
    """Declared write effect (schema honesty, E06-01 §7)."""

    input_name: str
    kind: str  # "in_place" | "out" | "none"
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("in_place", "out", "none"):
            raise SchemaError(
                f"mutation {self.input_name!r}: unknown kind {self.kind!r}",
                details={"field": "mutation.kind", "actual": self.kind},
            )


@dataclass(frozen=True)
class AliasSpec:
    """Declared alias relation between an output and an input."""

    output_index: int
    input_name: Optional[str]  # None == fresh storage
    kind: str  # "view" | "reuse_input" | "no_alias"
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("view", "reuse_input", "no_alias"):
            raise SchemaError(
                f"alias for output {self.output_index}: unknown kind {self.kind!r}",
                details={"field": "alias.kind", "actual": self.kind},
            )
        if self.kind != "no_alias" and self.input_name is None:
            raise SchemaError(
                f"alias for output {self.output_index}: kind {self.kind!r} "
                "requires an input name",
                details={"field": "alias.input", "actual": self.input_name},
            )
        if self.kind == "no_alias" and self.input_name is not None:
            raise SchemaError(
                f"alias for output {self.output_index}: 'no_alias' must not name "
                f"an input (got {self.input_name!r})",
                details={"field": "alias.input", "actual": self.input_name},
            )


@dataclass(frozen=True)
class ReturnSpec:
    """One named return value."""

    name: str
    kind: str = "Tensor"
    description: str = ""

    def render(self) -> str:
        return f"{self.kind} {self.name}"


# ── the schema ────────────────────────────────────────────────────────────

AUTOGRAD_POLICIES = (
    "INFERENCE_ONLY_ERROR",  # explicit error before the kernel runs
    "REGISTERED_BACKWARD",  # a backward exists and is tested
    "FALLTHROUGH",  # autograd falls through to the base implementation
)

AUTOCAST_POLICIES = (
    "DISABLED",  # inputs are cast per the caller's policy; op does not re-cast
    "CAST_INPUTS_TO",  # op casts inputs to a frozen dtype
    "FOLLOW_INPUT",  # no cast at all
)


@dataclass(frozen=True)
class OpSchema:
    """A versioned, implementation-independent operator schema (C3 successor)."""

    name: str
    overload: str
    args: Tuple[ArgSpec, ...]
    returns: Tuple[ReturnSpec, ...]
    mutation: Tuple[MutationSpec, ...] = ()
    aliases: Tuple[AliasSpec, ...] = ()
    inference_only: bool = True
    autograd_policy: str = "INFERENCE_ONLY_ERROR"
    autocast_policy: str = "FOLLOW_INPUT"
    autocast_cast_dtype: Optional[str] = None
    semantic_version: str = "1.0.0"
    composite_owner: str = "python"
    note: str = ""

    def __post_init__(self) -> None:
        if self.autograd_policy not in AUTOGRAD_POLICIES:
            raise SchemaError(
                f"{self.name}: unknown autograd policy {self.autograd_policy!r}",
                details={"field": "autograd_policy", "actual": self.autograd_policy},
            )
        if self.autocast_policy not in AUTOCAST_POLICIES:
            raise SchemaError(
                f"{self.name}: unknown autocast policy {self.autocast_policy!r}",
                details={"field": "autocast_policy", "actual": self.autocast_policy},
            )
        if self.autocast_policy == "CAST_INPUTS_TO" and not self.autocast_cast_dtype:
            raise SchemaError(
                f"{self.name}: autocast policy CAST_INPUTS_TO needs a dtype",
                details={"field": "autocast_cast_dtype"},
            )
        if self.composite_owner not in ("python", "cpp"):
            raise SchemaError(
                f"{self.name}: composite_owner must be 'python' or 'cpp'",
                details={"field": "composite_owner", "actual": self.composite_owner},
            )
        self._validate_names()

    # ── validation ────────────────────────────────────────────────────

    def _validate_names(self) -> None:
        if "::" not in self.name:
            raise SchemaError(
                f"operator name {self.name!r} must be namespace::name",
                details={"field": "name"},
            )
        seen = set()
        for arg in self.args:
            if arg.name in seen:
                raise SchemaError(
                    f"{self.name}: duplicate argument {arg.name!r}",
                    details={"field": "args", "actual": arg.name},
                )
            seen.add(arg.name)
        for spec in self.mutation:
            if spec.input_name not in seen:
                raise SchemaError(
                    f"{self.name}: mutation names unknown input {spec.input_name!r}",
                    details={"field": "mutation.input", "actual": spec.input_name},
                )
        for alias in self.aliases:
            if alias.input_name is not None and alias.input_name not in seen:
                raise SchemaError(
                    f"{self.name}: alias names unknown input {alias.input_name!r}",
                    details={"field": "alias.input", "actual": alias.input_name},
                )
            if not 0 <= alias.output_index < len(self.returns):
                raise SchemaError(
                    f"{self.name}: alias output index {alias.output_index} out of "
                    f"range (returns={len(self.returns)})",
                    details={"field": "alias.output_index"},
                )

    def validate(self) -> None:
        """Refuse the schema/implementation mismatches E06-01 §7 warns about.

        * an ``in_place`` mutation must be paired with an alias whose kind is
          ``reuse_input`` for the same input (the returned object *is* the input);
        * a functional schema (no mutation) must not declare ``reuse_input``;
        * every return must have exactly one alias entry (explicit "no_alias" is
          required so a missing entry cannot be mistaken for freshness).
        """
        mutated = {spec.input_name for spec in self.mutation if spec.kind == "in_place"}
        alias_by_output = {alias.output_index: alias for alias in self.aliases}
        problems: List[Dict[str, Any]] = []
        for output_index in range(len(self.returns)):
            alias = alias_by_output.get(output_index)
            if alias is None:
                problems.append(
                    {
                        "field": f"aliases[{output_index}]",
                        "reason": "MISSING_ALIAS_DECLARATION",
                        "detail": "every output must declare an alias relation",
                    }
                )
                continue
            if alias.kind == "reuse_input" and alias.input_name not in mutated:
                problems.append(
                    {
                        "field": f"aliases[{output_index}]",
                        "reason": "ALIAS_WITHOUT_MUTATION",
                        "detail": (
                            f"output {output_index} reuses {alias.input_name!r} but the "
                            "schema declares no in_place mutation"
                        ),
                    }
                )
        for input_name in mutated:
            if not any(
                alias.kind == "reuse_input" and alias.input_name == input_name
                for alias in self.aliases
            ):
                problems.append(
                    {
                        "field": f"mutation[{input_name}]",
                        "reason": "MUTATION_WITHOUT_ALIAS",
                        "detail": (
                            f"{input_name!r} is written in place but no output declares "
                            "reuse_input"
                        ),
                    }
                )
        if problems:
            raise SchemaError(
                f"{self.name}: schema is inconsistent with its mutation/alias contract",
                details={"problems": problems},
            )

    # ── identity ──────────────────────────────────────────────────────

    def signature(self) -> str:
        """Canonical textual signature used as the schema identity input."""
        args = ", ".join(arg.render() for arg in self.args)
        returns = ", ".join(ret.render() for ret in self.returns)
        flags = [
            f"mut={self.mutation_signature()}",
            f"alias={self.alias_signature()}",
            f"autograd={self.autograd_policy}",
            f"autocast={self.autocast_policy}",
            f"autocast_dtype={self.autocast_cast_dtype or 'none'}",
            f"inference_only={self.inference_only}",
        ]
        return f"{self.name}.{self.overload}({args}) -> ({returns}) [{'; '.join(flags)}]"

    def mutation_signature(self) -> str:
        entries = sorted(
            f"{spec.input_name}:{spec.kind}" for spec in self.mutation
        )
        return ",".join(entries) if entries else "none"

    def alias_signature(self) -> str:
        entries = []
        for alias in sorted(self.aliases, key=lambda item: item.output_index):
            target = alias.input_name or "fresh"
            entries.append(f"{alias.output_index}:{alias.kind}:{target}")
        return ",".join(entries) if entries else "none"

    @property
    def schema_hash(self) -> str:
        payload = f"{SCHEMA_VERSION}|{self.signature()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "overload": self.overload,
            "signature": self.signature(),
            "schema_hash": self.schema_hash,
            "args": [
                {"name": arg.name, "kind": arg.kind, "default": arg.default}
                for arg in self.args
            ],
            "returns": [ret.render() for ret in self.returns],
            "mutation": [f"{spec.input_name}:{spec.kind}" for spec in self.mutation],
            "alias": [f"{alias.output_index}:{alias.kind}" for alias in self.aliases],
            "inference_only": self.inference_only,
            "autograd_policy": self.autograd_policy,
            "autocast_policy": self.autocast_policy,
            "composite_owner": self.composite_owner,
        }


# ── the frozen P0 schema set ──────────────────────────────────────────────


def rms_norm_schema() -> OpSchema:
    """``hqsb::rms_norm`` — functional, FP32 accumulation, inference only."""
    return OpSchema(
        name=OP_RMS_NORM,
        overload="default",
        args=(
            ArgSpec("x", "Tensor", description="input activations, [..., H]"),
            ArgSpec("weight", "Tensor", description="scale weight, [H]"),
            ArgSpec("eps", "float?", default=1e-6),
        ),
        returns=(ReturnSpec("out"),),
        mutation=(),
        aliases=(AliasSpec(0, None, "no_alias", note="fresh output; never views x"),),
        inference_only=True,
        autograd_policy="INFERENCE_ONLY_ERROR",
        autocast_policy="FOLLOW_INPUT",
        semantic_version="1.0.0",
        note="frozen from C3 OperatorSpec rmsnorm_v2 + S04.5 semantics",
    )


def fused_add_rms_norm_schema() -> OpSchema:
    """``hqsb::fused_add_rms_norm`` — the frozen contract choice for S06.

    Two candidates are legal per E06-01 §7 (functional vs mutable).  The S06
    contract is **functional**: the fused op returns a fresh ``normalized`` and a
    fresh ``updated_residual`` and never writes either input.  The reason is
    auditability (no alias/functionalization/CUDA-Graph write hazard); the cost
    is measured by E06-07 rather than assumed away.  A mutable variant, if ever
    needed, must be a *separate private op* — never a second interpretation of
    this name.
    """
    return OpSchema(
        name=OP_FUSED_ADD_RMS_NORM,
        overload="default",
        args=(
            ArgSpec("x", "Tensor", description="input activations [..., H]"),
            ArgSpec("residual", "Tensor", description="residual [..., H]"),
            ArgSpec("weight", "Tensor", description="RMSNorm weight [H]"),
            ArgSpec("eps", "float?", default=1e-6),
        ),
        returns=(
            ReturnSpec("normalized"),
            ReturnSpec("updated_residual"),
        ),
        mutation=(),
        aliases=(
            AliasSpec(0, None, "no_alias"),
            AliasSpec(1, None, "no_alias"),
        ),
        inference_only=True,
        autograd_policy="INFERENCE_ONLY_ERROR",
        autocast_policy="FOLLOW_INPUT",
        semantic_version="1.0.0",
        note=(
            "functional fusion contract; rounding/accumulation frozen in "
            "hqsb.integration.differential.FROZEN_ADD_RMSNORM_SEMANTICS"
        ),
    )


def dequant_linear_schema() -> OpSchema:
    """``hqsb::dequant_linear`` — quant boundary op (second hotspot, S05 route).

    Semantics: ``y = x @ dequant(weight_q)`` with the S05 QuantArtifact frozen
    scale/zero/group/tail/layout.  The packed payload is an opaque tensor; the
    op must not expose a materialized FP16 weight as its graph output.
    """
    return OpSchema(
        name=OP_DEQUANT_LINEAR,
        overload="default",
        args=(
            ArgSpec("x", "Tensor", description="activations [..., K]"),
            ArgSpec("weight_q", "Tensor", description="packed codes (opaque uint8)"),
            ArgSpec("scales", "Tensor", description="dequant scales"),
            ArgSpec("zeros", "Tensor?", default=None),
            ArgSpec("bias", "Tensor?", default=None),
            ArgSpec("group_size", "int?", default=None),
            ArgSpec("out_features", "int", description="logical N"),
            ArgSpec("layout_id", "str", description="packing layout identity"),
        ),
        returns=(ReturnSpec("out"),),
        mutation=(),
        aliases=(AliasSpec(0, None, "no_alias"),),
        inference_only=True,
        autograd_policy="INFERENCE_ONLY_ERROR",
        autocast_policy="FOLLOW_INPUT",
        semantic_version="1.0.0",
        note="S05 QuantArtifact layout is part of the operator identity",
    )


def frozen_schemas() -> Tuple[OpSchema, ...]:
    """The three P0 schemas, each validated on construction."""
    schemas = (rms_norm_schema(), fused_add_rms_norm_schema(), dequant_linear_schema())
    for schema in schemas:
        schema.validate()
    return schemas


def schema_by_name(name: str) -> OpSchema:
    for schema in frozen_schemas():
        if schema.name == name:
            return schema
    raise SchemaError(
        f"no frozen schema for {name!r}",
        details={"known": [schema.name for schema in frozen_schemas()]},
    )


def schema_table() -> Dict[str, Dict[str, Any]]:
    return {schema.name: schema.as_dict() for schema in frozen_schemas()}


# ── C3 OperatorSpec projection ────────────────────────────────────────────

#: Fields C3 does not carry; they must be supplied explicitly by the S04.5/S06
#: contract, never defaulted silently (E06-01 §3).
REQUIRED_CONTRACT_FIELDS = (
    "mutation",
    "alias",
    "autograd_policy",
    "autocast_policy",
    "inference_only",
)


def schema_from_operator_spec(
    spec: Mapping[str, Any],
    contract: Optional[Mapping[str, Any]] = None,
) -> OpSchema:
    """Derive an :class:`OpSchema` from a C3 OperatorSpec document.

    ``contract`` must supply the mutation/alias/policy fields C3 does not
    express.  A missing field is a hard error: defaulting it would be exactly
    the "schema lies about mutation" failure mode this experiment targets.
    """
    name = spec.get("name")
    if not name:
        raise SchemaError("C3 spec is missing 'name'", details={"field": "name"})
    qualified = name if "::" in name else f"{NAMESPACE}::{name}"
    contract = dict(contract or {})
    missing = [field for field in REQUIRED_CONTRACT_FIELDS if field not in contract]
    if missing:
        raise SchemaError(
            f"{qualified}: contract fields {missing} are required; C3 OperatorSpec "
            "does not express mutation/alias/policy, and guessing them would let the "
            "graph system make unsound assumptions",
            details={"field": "contract", "missing": missing},
        )
    inputs = spec.get("inputs") or ()
    args: List[ArgSpec] = []
    for item in inputs:
        args.append(
            ArgSpec(
                name=str(item.get("name", f"arg{len(args)}")),
                kind="Tensor",
                description=str(item.get("dtype", "")),
            )
        )
    outputs = spec.get("outputs") or ()
    returns = tuple(
        ReturnSpec(name=str(item.get("name", f"out{index}")))
        for index, item in enumerate(outputs)
    )
    if not returns:
        raise SchemaError(
            f"{qualified}: C3 spec declares no outputs",
            details={"field": "outputs"},
        )
    return OpSchema(
        name=qualified,
        overload=str(contract.get("overload", "default")),
        args=tuple(args),
        returns=returns,
        mutation=tuple(
            MutationSpec(**entry) if isinstance(entry, Mapping) else MutationSpec(*entry)
            for entry in contract["mutation"]
        ),
        aliases=tuple(
            AliasSpec(**entry) if isinstance(entry, Mapping) else AliasSpec(*entry)
            for entry in contract["alias"]
        ),
        inference_only=bool(contract["inference_only"]),
        autograd_policy=str(contract["autograd_policy"]),
        autocast_policy=str(contract["autocast_policy"]),
        autocast_cast_dtype=contract.get("autocast_cast_dtype"),
        semantic_version=str(spec.get("semantic_version", "1.0.0")),
        note=str(contract.get("note", "")),
    )


# ── single schema owner audit (E06-01 §2, §10) ────────────────────────────

#: Definition sites.  ``torch.library.define`` (Python) and ``m.def`` (C++)
#: are the only legal schema *owners*; ``Library(...)``/``TORCH_LIBRARY[_FRAGMENT]``
#: register implementations and are recorded as fragments (E06-01 §2).
PY_DEFINE_RE = re.compile(
    r"""torch\.library\.define\(\s*['"](?P<define>[^'"]+)['"]"""
    r"""|torch\.library\.Library\(\s*['"](?P<library>[^'"]+)['"]"""
    r"""|\.def\(\s*['"](?P<defname>[^'"]+)['"]""",
    re.VERBOSE,
)
CPP_DEFINE_RE = re.compile(
    r"""TORCH_LIBRARY(?P<fragment>_FRAGMENT)?\(\s*(?P<ns>\w+)"""
    r"""|\.def\(\s*"(?P<defname>[^"]+)\"""",
    re.VERBOSE,
)
_SOURCE_SUFFIXES = (".py", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")
_SCHEMA_NAME_RE = re.compile(r"^(?P<name>[A-Za-z_][\w:.]*)\s*\(")


@dataclass(frozen=True)
class OwnerSite:
    """One definition site discovered by the audit."""

    qualified_name: str
    owner_kind: str  # "python" | "cpp" | "fragment"
    file: str
    line: int


@dataclass
class OwnerAudit:
    """Result of the single-owner scan."""

    sites: Tuple[OwnerSite, ...] = ()
    parse_errors: Tuple[str, ...] = ()

    @property
    def owners(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for site in self.sites:
            if site.owner_kind == "fragment":
                continue
            grouped.setdefault(site.qualified_name, []).append(site.file)
        return grouped

    @property
    def conflicts(self) -> Dict[str, List[str]]:
        return {
            name: files
            for name, files in self.owners.items()
            if len(set(files)) > 1
        }

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.parse_errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "owners": {name: sorted(set(files)) for name, files in self.owners.items()},
            "conflicts": self.conflicts,
            "parse_errors": list(self.parse_errors),
            "sites": [
                {
                    "name": site.qualified_name,
                    "owner_kind": site.owner_kind,
                    "file": site.file,
                    "line": site.line,
                }
                for site in self.sites
            ],
        }


def _iter_sources(paths: Sequence[str]) -> Iterable[str]:
    for path in paths:
        if os.path.isfile(path):
            if path.endswith(_SOURCE_SUFFIXES):
                yield path
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name != "__pycache__"]
            for filename in filenames:
                if filename.endswith(_SOURCE_SUFFIXES):
                    yield os.path.join(dirpath, filename)


def audit_schema_owners(paths: Sequence[str], namespace: str = NAMESPACE) -> OwnerAudit:
    """Scan ``paths`` for schema definition sites and report the owners.

    A qualified name owned from two files is a conflict: whichever loads last
    would silently win (E06-01 §10).  The function never imports the scanned
    code; it reads text and records file/line so the audit can be re-run.
    """
    sites: List[OwnerSite] = []
    parse_errors: List[str] = []
    for file_path in sorted(_iter_sources(paths)):
        try:
            with open(file_path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:  # pragma: no cover - unreadable file
            parse_errors.append(f"{file_path}: {exc}")
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "torch.library" in line:
                match = PY_DEFINE_RE.search(line)
                if match:
                    if match.group("define") or match.group("defname"):
                        raw = match.group("define") or match.group("defname")
                        sites.append(
                            OwnerSite(
                                qualified_name=_qualify(_strip_schema_args(raw), namespace),
                                owner_kind="python",
                                file=file_path,
                                line=line_number,
                            )
                        )
                    elif match.group("library"):
                        # ``torch.library.Library(ns, kind)`` only registers
                        # implementations; it is not a schema owner.
                        sites.append(
                            OwnerSite(
                                qualified_name=f"{match.group('library')}::<library>",
                                owner_kind="fragment",
                                file=file_path,
                                line=line_number,
                            )
                        )
            if "TORCH_LIBRARY" in line or ".def(" in line:
                match = CPP_DEFINE_RE.search(line)
                if match:
                    if match.group("defname"):
                        sites.append(
                            OwnerSite(
                                qualified_name=_qualify(
                                    _strip_schema_args(match.group("defname")), namespace
                                ),
                                owner_kind="cpp",
                                file=file_path,
                                line=line_number,
                            )
                        )
                    elif match.group("ns"):
                        sites.append(
                            OwnerSite(
                                qualified_name=f"{match.group('ns')}::<library>",
                                owner_kind="fragment",
                                file=file_path,
                                line=line_number,
                            )
                        )
    return OwnerAudit(sites=tuple(sites), parse_errors=tuple(parse_errors))


def _strip_schema_args(raw: str) -> str:
    """``"rms_norm(Tensor x) -> Tensor"`` → ``"rms_norm"``."""
    match = _SCHEMA_NAME_RE.match(raw.strip())
    return match.group("name") if match else raw.strip()


def _qualify(name: str, namespace: str) -> str:
    if not name or name.startswith("<"):
        return name
    if "::" in name:
        return name
    return f"{namespace}::{name}"


def assert_single_owner(schemas: Sequence[OpSchema], audit: OwnerAudit) -> None:
    """Raise when a schema's qualified name has no owner or several owners."""
    problems: List[Dict[str, Any]] = []
    for schema in schemas:
        owners = sorted(set(audit.owners.get(schema.name, ())))
        if not owners:
            problems.append(
                {
                    "operator": schema.name,
                    "reason": "NO_SCHEMA_OWNER",
                    "detail": "no torch.library/TORCH_LIBRARY definition found",
                }
            )
        elif len(owners) > 1:
            problems.append(
                {
                    "operator": schema.name,
                    "reason": "MULTIPLE_SCHEMA_OWNERS",
                    "detail": owners,
                }
            )
    if problems:
        raise ConfigError(
            "schema ownership audit failed: " + "; ".join(
                f"{item['operator']} {item['reason']}" for item in problems
            ),
            details={"problems": problems},
        )


@dataclass(frozen=True)
class SchemaRegistrySnapshot:
    """Hashable snapshot of the frozen schema set (feeds graph/compile identity)."""

    schemas: Tuple[Tuple[str, str], ...]  # (qualified name, schema_hash)

    @property
    def digest(self) -> str:
        payload = "|".join(f"{name}@{digest}" for name, digest in self.schemas)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, str]:
        return {name: digest for name, digest in self.schemas}


def schema_registry_snapshot(schemas: Optional[Sequence[OpSchema]] = None) -> SchemaRegistrySnapshot:
    selected = tuple(schemas) if schemas is not None else frozen_schemas()
    return SchemaRegistrySnapshot(
        schemas=tuple(sorted((schema.name, schema.schema_hash) for schema in selected))
    )


__all__ = [
    "ARG_KINDS",
    "AUTOCAST_POLICIES",
    "AUTOGRAD_POLICIES",
    "AliasSpec",
    "ArgSpec",
    "MutationSpec",
    "NAMESPACE",
    "OP_DEQUANT_LINEAR",
    "OP_FUSED_ADD_RMS_NORM",
    "OP_RMS_NORM",
    "OpSchema",
    "OwnerAudit",
    "OwnerSite",
    "P0_OPERATORS",
    "REQUIRED_CONTRACT_FIELDS",
    "ReturnSpec",
    "SCHEMA_VERSION",
    "SchemaRegistrySnapshot",
    "assert_single_owner",
    "audit_schema_owners",
    "dequant_linear_schema",
    "frozen_schemas",
    "fused_add_rms_norm_schema",
    "rms_norm_schema",
    "schema_by_name",
    "schema_from_operator_spec",
    "schema_registry_snapshot",
    "schema_table",
]
