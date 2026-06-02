"""Registration, dispatch and redispatch boundaries (E06-01 §4–§6, §9, §18).

"``torch.ops.hqsb.rms_norm`` is callable" proves nothing: the call may have
landed on a composite, on an older extension, or on a CPU reference after a
silent device copy (E06-01 §15).  This module therefore models what the
dispatcher *actually* did, as a first-class record:

* registration matrix with conflict detection (duplicate schema, same name
  different schema, same key different implementation, third-party namespace);
* dispatch-table snapshots with requested / selected / internal route /
  observed kernel / fallback, plus a structural hash and a diff;
* redispatch bookkeeping that refuses recursion and key-set loss instead of
  hiding it behind a global recursion counter;
* deterministic fallback policy (requested → actual → reason), with an
  explicit strict mode that must fail loudly.

Nothing here calls into torch: the records describe the boundary, and the
(future) runner fills them from real registrations.  That keeps the contract
testable on CPU and keeps the module free of an unconditional torch import.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, RegistryError

# ── dispatch keys ─────────────────────────────────────────────────────────

#: Dispatch keys S06 must place implementations under (E06-01 §4).  ``CUDA``
#: covers CUDA/Triton/quant *routes*: those are an internal backend route, not
#: an independent PyTorch dispatch key, and must be recorded as such.
class DispatchKey:
    COMPOSITE_EXPLICIT_AUTOGRAD = "CompositeExplicitAutograd"
    CPU = "CPU"
    CUDA = "CUDA"
    META = "Meta"
    AUTOGRAD = "Autograd"
    AUTOCAST_CUDA = "AutocastCUDA"
    FUNCTIONALIZE = "Functionalize"

    ALL = (
        COMPOSITE_EXPLICIT_AUTOGRAD,
        CPU,
        CUDA,
        META,
        AUTOGRAD,
        AUTOCAST_CUDA,
        FUNCTIONALIZE,
    )

    #: Keys whose implementation is a *math reference* rather than a kernel.
    REFERENCE_KEYS = (COMPOSITE_EXPLICIT_AUTOGRAD, CPU, META)

    #: Layer order used when computing the redispatch successor key set.
    ORDER = (
        COMPOSITE_EXPLICIT_AUTOGRAD,
        CPU,
        CUDA,
        META,
        AUTOCAST_CUDA,
        FUNCTIONALIZE,
        AUTOGRAD,
    )


PROVIDER_KINDS = (
    "python",
    "cpp",
    "cuda_shared_lib",
    "triton",
    "composite",
    "dummy",
)

#: Namespaces HQSB claims; a provider from outside this set registering into a
#: claimed namespace is a namespace collision (E06-01 §10).
CLAIMED_NAMESPACES = ("hqsb", "hqsb_preview")


@dataclass(frozen=True)
class RegistrationRecord:
    """One registered implementation of one operator under one dispatch key."""

    qualified_name: str
    key: str
    implementation: str
    provider: str
    library: str
    build_hash: str = ""
    order: int = 0
    schema_hash: str = ""
    #: Who owns this record.  ``hqsb`` is the project; anything else trying to
    #: register into a claimed namespace is a collision (E06-01 §10).
    owner: str = "hqsb"
    note: str = ""

    def __post_init__(self) -> None:
        if self.key not in DispatchKey.ALL:
            raise RegistryError(
                f"{self.qualified_name}: unknown dispatch key {self.key!r}",
                details={"field": "key", "allowed": list(DispatchKey.ALL)},
            )
        if self.provider not in PROVIDER_KINDS:
            raise RegistryError(
                f"{self.qualified_name}: unknown provider {self.provider!r}",
                details={"field": "provider", "allowed": list(PROVIDER_KINDS)},
            )

    @property
    def namespace(self) -> str:
        return self.qualified_name.split("::", 1)[0]

    @property
    def identity(self) -> Tuple[str, str]:
        return (self.qualified_name, self.key)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "key": self.key,
            "implementation": self.implementation,
            "provider": self.provider,
            "library": self.library,
            "build_hash": self.build_hash,
            "order": self.order,
            "owner": self.owner,
        }


class RegistrationAction:
    """Outcome of :meth:`RegistrationMatrix.register`."""

    REGISTERED = "REGISTERED"
    IDEMPOTENT = "IDEMPOTENT"
    REJECTED_DUPLICATE_CONFLICT = "REJECTED_DUPLICATE_CONFLICT"
    REJECTED_SCHEMA_MISMATCH = "REJECTED_SCHEMA_MISMATCH"
    REJECTED_NAMESPACE_COLLISION = "REJECTED_NAMESPACE_COLLISION"
    REJECTED_UNKNOWN_SCHEMA = "REJECTED_UNKNOWN_SCHEMA"

    ALL = (
        REGISTERED,
        IDEMPOTENT,
        REJECTED_DUPLICATE_CONFLICT,
        REJECTED_SCHEMA_MISMATCH,
        REJECTED_NAMESPACE_COLLISION,
        REJECTED_UNKNOWN_SCHEMA,
    )


@dataclass(frozen=True)
class RegistrationOutcome:
    """What happened, with a stable reason code — never a silent overwrite."""

    action: str
    qualified_name: str
    key: str
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.action in (RegistrationAction.REGISTERED, RegistrationAction.IDEMPOTENT)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "qualified_name": self.qualified_name,
            "key": self.key,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


@dataclass
class RegistrationMatrix:
    """The declared implementation matrix, with conflict-safe registration."""

    schemas: Dict[str, str] = field(default_factory=dict)  # name -> schema_hash
    records: List[RegistrationRecord] = field(default_factory=list)
    outcomes: List[RegistrationOutcome] = field(default_factory=list)
    _order: int = 0

    # ── schema ownership ──────────────────────────────────────────────

    def declare_schema(self, qualified_name: str, schema_hash: str) -> RegistrationOutcome:
        if not qualified_name or "::" not in qualified_name:
            raise ConfigError(
                f"qualified operator name must be namespace::name, got {qualified_name!r}",
                details={"field": "qualified_name"},
            )
        existing = self.schemas.get(qualified_name)
        if existing is None:
            self.schemas[qualified_name] = schema_hash
            outcome = RegistrationOutcome(
                action=RegistrationAction.REGISTERED,
                qualified_name=qualified_name,
                key="<schema>",
                reason="schema owner declared",
            )
        elif existing == schema_hash:
            outcome = RegistrationOutcome(
                action=RegistrationAction.IDEMPOTENT,
                qualified_name=qualified_name,
                key="<schema>",
                reason="identical schema re-declared",
            )
        else:
            outcome = RegistrationOutcome(
                action=RegistrationAction.REJECTED_SCHEMA_MISMATCH,
                qualified_name=qualified_name,
                key="<schema>",
                reason=(
                    "the same qualified name was declared with a different schema; "
                    "bump the overload or the operator name instead of reinterpreting it"
                ),
                detail={"existing": existing, "requested": schema_hash},
            )
        self.outcomes.append(outcome)
        return outcome

    # ── implementations ───────────────────────────────────────────────

    def register(self, record: RegistrationRecord) -> RegistrationOutcome:
        """Register one implementation; conflicts are rejected, never resolved."""
        self._order += 1
        record = _with_order(record, self._order)

        if record.namespace in CLAIMED_NAMESPACES and record.owner != "hqsb":
            outcome = RegistrationOutcome(
                action=RegistrationAction.REJECTED_NAMESPACE_COLLISION,
                qualified_name=record.qualified_name,
                key=record.key,
                reason=(
                    f"owner {record.owner!r} must not register into claimed namespace "
                    f"{record.namespace!r}; use its own namespace"
                ),
                detail={"owner": record.owner, "namespace": record.namespace},
            )
            self.outcomes.append(outcome)
            return outcome

        declared = self.schemas.get(record.qualified_name)
        if record.schema_hash and declared and record.schema_hash != declared:
            outcome = RegistrationOutcome(
                action=RegistrationAction.REJECTED_SCHEMA_MISMATCH,
                qualified_name=record.qualified_name,
                key=record.key,
                reason="implementation built against a different schema hash",
                detail={"declared": declared, "implementation": record.schema_hash},
            )
            self.outcomes.append(outcome)
            return outcome

        existing = [item for item in self.records if item.identity == record.identity]
        if existing:
            current = existing[-1]
            if (
                current.implementation == record.implementation
                and current.build_hash == record.build_hash
            ):
                outcome = RegistrationOutcome(
                    action=RegistrationAction.IDEMPOTENT,
                    qualified_name=record.qualified_name,
                    key=record.key,
                    reason="identical implementation already registered (repeat load)",
                )
            else:
                outcome = RegistrationOutcome(
                    action=RegistrationAction.REJECTED_DUPLICATE_CONFLICT,
                    qualified_name=record.qualified_name,
                    key=record.key,
                    reason=(
                        "a different implementation is already registered for this "
                        "operator/key; last-load-wins is forbidden"
                    ),
                    detail={
                        "existing": current.as_dict(),
                        "requested": record.as_dict(),
                    },
                )
            self.outcomes.append(outcome)
            return outcome

        self.records.append(record)
        outcome = RegistrationOutcome(
            action=RegistrationAction.REGISTERED,
            qualified_name=record.qualified_name,
            key=record.key,
            reason="registered",
        )
        self.outcomes.append(outcome)
        return outcome

    # ── inspection ────────────────────────────────────────────────────

    def keys_for(self, qualified_name: str) -> List[str]:
        return sorted({record.key for record in self.records if record.qualified_name == qualified_name})

    def implementations(self, qualified_name: str, key: str) -> List[RegistrationRecord]:
        return [
            record
            for record in self.records
            if record.qualified_name == qualified_name and record.key == key
        ]

    @property
    def conflicts(self) -> List[RegistrationOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.accepted]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schemas": dict(self.schemas),
            "records": [record.as_dict() for record in self.records],
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
            "conflicts": [outcome.as_dict() for outcome in self.conflicts],
        }

    def matrix_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _with_order(record: RegistrationRecord, order: int) -> RegistrationRecord:
    return replace(record, order=order)


def frozen_p0_matrix() -> RegistrationMatrix:
    """The declared matrix for the three P0 operators (E06-01 §4).

    Layout of responsibilities:

    * CPU/Composite — math reference, also the fallback target;
    * CUDA — HQSB CUDA/CUTLASS kernels (Triton/quant are an *internal route*
      selected by backend capability, not a separate dispatch key);
    * Meta — metadata only, used by capture/export;
    * Autocast — explicit policy;
    * Autograd — explicit policy (inference-only ⇒ explicit error).
    """
    from hqsb.integration import specs

    matrix = RegistrationMatrix()
    for schema in specs.frozen_schemas():
        matrix.declare_schema(schema.name, schema.schema_hash)
    plan: Sequence[Tuple[str, str, str, str]] = (
        (specs.OP_RMS_NORM, DispatchKey.COMPOSITE_EXPLICIT_AUTOGRAD, "composite_cpu_reference", "composite"),
        (specs.OP_RMS_NORM, DispatchKey.CPU, "cpu_reference", "cpp"),
        (specs.OP_RMS_NORM, DispatchKey.CUDA, "hqsb_cuda_rms_norm_dispatch", "cuda_shared_lib"),
        (specs.OP_RMS_NORM, DispatchKey.META, "meta_rms_norm", "python"),
        (specs.OP_RMS_NORM, DispatchKey.AUTOCAST_CUDA, "autocast_policy_rms_norm", "python"),
        (specs.OP_RMS_NORM, DispatchKey.AUTOGRAD, "inference_only_error_rms_norm", "python"),
        (specs.OP_FUSED_ADD_RMS_NORM, DispatchKey.COMPOSITE_EXPLICIT_AUTOGRAD, "composite_cpu_reference", "composite"),
        (specs.OP_FUSED_ADD_RMS_NORM, DispatchKey.CPU, "cpu_reference", "cpp"),
        (specs.OP_FUSED_ADD_RMS_NORM, DispatchKey.CUDA, "hqsb_cuda_fused_add_rms_norm", "cuda_shared_lib"),
        (specs.OP_FUSED_ADD_RMS_NORM, DispatchKey.META, "meta_fused_add_rms_norm", "python"),
        (specs.OP_FUSED_ADD_RMS_NORM, DispatchKey.AUTOCAST_CUDA, "autocast_policy_fused", "python"),
        (specs.OP_FUSED_ADD_RMS_NORM, DispatchKey.AUTOGRAD, "inference_only_error_fused", "python"),
        (specs.OP_DEQUANT_LINEAR, DispatchKey.COMPOSITE_EXPLICIT_AUTOGRAD, "composite_dequant_reference", "composite"),
        (specs.OP_DEQUANT_LINEAR, DispatchKey.CUDA, "hqsb_triton_dequant_linear", "triton"),
        (specs.OP_DEQUANT_LINEAR, DispatchKey.META, "meta_dequant_linear", "python"),
        (specs.OP_DEQUANT_LINEAR, DispatchKey.AUTOGRAD, "inference_only_error_dequant", "python"),
    )
    for qualified_name, key, implementation, provider in plan:
        matrix.register(
            RegistrationRecord(
                qualified_name=qualified_name,
                key=key,
                implementation=implementation,
                provider=provider,
                library="libhqsb_ops" if provider in ("cpp", "cuda_shared_lib") else "hqsb.python",
                build_hash="unbuilt",
                schema_hash=matrix.schemas.get(qualified_name, ""),
            )
        )
    return matrix


# ── capability-driven selection ───────────────────────────────────────────


@dataclass(frozen=True)
class CapabilityRequest:
    """What the caller needs, independent of which backend provides it."""

    op: str
    dtype: str = "float16"
    layout: str = "contiguous"
    shape: Tuple[int, ...] = ()
    rank: int = 0
    alignment: int = 1
    device: str = "cuda"
    arch: str = ""
    quant_group: Optional[int] = None
    compile_mode: str = "eager"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityDecision:
    """``supports`` verdict with field-level failures (E06-02 §10, E06-11 §7)."""

    ok: bool
    failures: Tuple[Dict[str, Any], ...] = ()

    def reasons(self) -> Tuple[str, ...]:
        return tuple(str(item["reason"]) for item in self.failures)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "failures": [dict(item) for item in self.failures]}


@dataclass(frozen=True)
class OperatorCapability:
    """Capability of one operator implementation (not a hardware model claim)."""

    op: str
    provider: str
    dtypes: Tuple[str, ...] = ("float16", "bfloat16", "float32")
    layouts: Tuple[str, ...] = ("contiguous",)
    min_rank: int = 1
    max_rank: int = 4
    alignment: int = 1
    group_sizes: Tuple[Optional[int], ...] = (None,)
    arch: Tuple[str, ...] = ()
    streams: Tuple[str, ...] = ("default", "any")
    compile_safe: bool = True
    max_workspace_bytes: int = 0
    kernel_symbol: str = ""

    def supports(self, request: CapabilityRequest) -> CapabilityDecision:
        failures: List[Dict[str, Any]] = []

        def fail(field_name: str, expected: Any, actual: Any, reason: str) -> None:
            failures.append(
                {
                    "field": field_name,
                    "expected": expected,
                    "actual": actual,
                    "reason": reason,
                }
            )

        if request.op != self.op:
            fail("op", self.op, request.op, "OP_MISMATCH")
        if request.dtype not in self.dtypes:
            fail("dtype", list(self.dtypes), request.dtype, "DTYPE")
        if request.layout not in self.layouts:
            fail("layout", list(self.layouts), request.layout, "STRIDE_LAYOUT")
        rank = request.rank or len(request.shape)
        if not self.min_rank <= rank <= self.max_rank:
            fail("rank", [self.min_rank, self.max_rank], rank, "SHAPE")
        if self.alignment > 1 and request.alignment % self.alignment:
            fail("alignment", self.alignment, request.alignment, "STRIDE_LAYOUT")
        if request.quant_group is not None:
            allowed_groups = [size for size in self.group_sizes if size is not None]
            if request.quant_group not in allowed_groups:
                fail("quant_group", allowed_groups, request.quant_group, "QUANT_POLICY")
        if self.arch and request.arch and request.arch not in self.arch:
            fail("arch", list(self.arch), request.arch, "BACKEND_CAPABILITY")
        if request.compile_mode != "eager" and not self.compile_safe:
            fail("compile_safe", True, False, "BACKEND_CAPABILITY")
        if request.extra.get("workspace_bytes") and (
            int(request.extra["workspace_bytes"]) > self.max_workspace_bytes
        ):
            fail(
                "workspace_bytes",
                self.max_workspace_bytes,
                int(request.extra["workspace_bytes"]),
                "SHAPE",
            )
        return CapabilityDecision(ok=not failures, failures=tuple(failures))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "provider": self.provider,
            "dtypes": list(self.dtypes),
            "layouts": list(self.layouts),
            "rank": [self.min_rank, self.max_rank],
            "alignment": self.alignment,
            "group_sizes": list(self.group_sizes),
            "arch": list(self.arch),
            "compile_safe": self.compile_safe,
            "kernel_symbol": self.kernel_symbol,
        }


# ── selection ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SelectionResult:
    """Which registered implementation would serve this request, and why."""

    requested: str
    selected: str
    internal_route: str
    reason: str
    key: str
    fallback_used: bool = False
    capability: Optional[CapabilityDecision] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "internal_route": self.internal_route,
            "reason": self.reason,
            "key": self.key,
            "fallback_used": self.fallback_used,
            "capability": self.capability.as_dict() if self.capability else None,
        }


def choose_key(request: CapabilityRequest) -> str:
    """Map a request onto the dispatch key that must serve it.

    Device/dtype drive the key; Triton/quant do **not** become keys
    (E06-01 §4): they stay an internal route under ``CUDA``.
    """
    if request.device == "meta":
        return DispatchKey.META
    if request.device == "cpu":
        return DispatchKey.CPU
    return DispatchKey.CUDA


def select_implementation(
    matrix: RegistrationMatrix,
    capability: OperatorCapability,
    request: CapabilityRequest,
) -> SelectionResult:
    """Capability-driven selection with an explicit reason (never silent)."""
    decision = capability.supports(request)
    key = choose_key(request)
    records = [
        record
        for record in matrix.implementations(request.op, key)
        if record.provider == capability.provider
    ]
    requested = f"{request.op}[{key}]"
    if not decision.ok:
        reference = [
            record
            for record in matrix.implementations(
                request.op, DispatchKey.COMPOSITE_EXPLICIT_AUTOGRAD
            )
        ]
        actual = reference[-1].implementation if reference else "<none>"
        return SelectionResult(
            requested=requested,
            selected=actual,
            internal_route="composite_reference",
            reason="; ".join(decision.reasons()),
            key=DispatchKey.COMPOSITE_EXPLICIT_AUTOGRAD,
            fallback_used=True,
            capability=decision,
        )
    if not records:
        raise RegistryError(
            f"{request.op}: no {capability.provider!r} implementation registered for key {key}",
            details={"request": requested, "keys": matrix.keys_for(request.op)},
        )
    selected = records[-1]
    return SelectionResult(
        requested=requested,
        selected=selected.implementation,
        internal_route=capability.provider,
        reason="capability satisfied",
        key=key,
        capability=decision,
    )


# ── dispatch table snapshots ──────────────────────────────────────────────


@dataclass(frozen=True)
class DispatchTableEntry:
    """One operator's observed dispatch state at a point in time."""

    qualified_name: str
    keyset: FrozenSet[str]
    selected: str
    internal_route: str
    observed_kernel: str
    fallback: str = ""
    requested: str = ""
    schema_hash: str = ""
    library_version: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "keyset": sorted(self.keyset),
            "selected": self.selected,
            "internal_route": self.internal_route,
            "observed_kernel": self.observed_kernel,
            "fallback": self.fallback,
            "requested": self.requested,
            "schema_hash": self.schema_hash,
            "library_version": self.library_version,
        }


@dataclass
class DispatchSnapshot:
    """A hashable dispatch-table snapshot (before/after each registration)."""

    stage: str
    entries: Tuple[DispatchTableEntry, ...] = ()

    def add(self, entry: DispatchTableEntry) -> None:
        self.entries = tuple(sorted((*self.entries, entry), key=lambda e: e.qualified_name))

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [entry.as_dict() for entry in self.entries], sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def diff(self, other: "DispatchSnapshot") -> Dict[str, Any]:
        """Structural diff between two snapshots (never a text diff of dumps)."""
        before = {entry.qualified_name: entry for entry in self.entries}
        after = {entry.qualified_name: entry for entry in other.entries}
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed: Dict[str, Dict[str, Any]] = {}
        for name in sorted(set(before) & set(after)):
            left, right = before[name], after[name]
            fields: Dict[str, Any] = {}
            for field_name in (
                "keyset",
                "selected",
                "internal_route",
                "observed_kernel",
                "fallback",
            ):
                left_value = getattr(left, field_name)
                right_value = getattr(right, field_name)
                if field_name == "keyset":
                    left_value, right_value = sorted(left_value), sorted(right_value)
                if left_value != right_value:
                    fields[field_name] = {"before": left_value, "after": right_value}
            if fields:
                changed[name] = fields
        return {
            "before": self.digest,
            "after": other.digest,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "digest": self.digest,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def snapshot_from_matrix(stage: str, matrix: RegistrationMatrix) -> DispatchSnapshot:
    """Build a snapshot from the declared matrix (audit table generated from raw)."""
    entries = []
    by_name: Dict[str, List[RegistrationRecord]] = {}
    for record in matrix.records:
        by_name.setdefault(record.qualified_name, []).append(record)
    for name, records in sorted(by_name.items()):
        keyset = frozenset(record.key for record in records)
        selected = records[-1].implementation
        entries.append(
            DispatchTableEntry(
                qualified_name=name,
                keyset=keyset,
                selected=selected,
                internal_route=records[-1].provider,
                observed_kernel=records[-1].implementation,
                requested=name,
                schema_hash=matrix.schemas.get(name, ""),
                library_version=records[-1].build_hash,
            )
        )
    return DispatchSnapshot(stage=stage, entries=tuple(entries))


# ── redispatch ────────────────────────────────────────────────────────────


class RedispatchError(RegistryError):
    """Redispatch went wrong (recursion or an exhausted key set)."""


def redispatch_keyset(
    keyset: FrozenSet[str], handled: FrozenSet[str]
) -> FrozenSet[str]:
    """Return the key set a redispatch must target.

    The handled keys are removed; an empty result means the implementation
    would re-enter the same dispatcher with nothing left to do — a design error
    (E06-01 §6), not something to paper over.
    """
    remaining = frozenset(key for key in keyset if key not in handled)
    if not remaining:
        raise RedispatchError(
            "redispatch exhausted: every key in the set was already handled; "
            "the wrapper would recurse into itself",
            details={"keyset": sorted(keyset), "handled": sorted(handled)},
        )
    return remaining


class RedispatchGuard:
    """Recursion *detection* with per-call key sets (not a global counter)."""

    def __init__(self, max_depth: int = 4) -> None:
        self.max_depth = max_depth
        self._stack: List[Tuple[str, FrozenSet[str]]] = []

    @contextmanager
    def enter(
        self, qualified_name: str, keyset: FrozenSet[str]
    ) -> Iterator[FrozenSet[str]]:
        frame = (qualified_name, frozenset(keyset))
        if frame in self._stack:
            raise RedispatchError(
                f"redispatch recursion detected for {qualified_name} "
                f"with key set {sorted(keyset)}",
                details={
                    "stack": [
                        {"op": name, "keyset": sorted(keys)}
                        for name, keys in self._stack
                    ]
                },
            )
        if len(self._stack) >= self.max_depth:
            raise RedispatchError(
                f"redispatch depth {len(self._stack)} exceeds max_depth={self.max_depth}",
                details={"op": qualified_name},
            )
        self._stack.append(frame)
        try:
            yield frozenset(keyset)
        finally:
            self._stack.pop()

    @property
    def depth(self) -> int:
        return len(self._stack)

    def path(self) -> Tuple[Tuple[str, FrozenSet[str]], ...]:
        return tuple(self._stack)


# ── fallback policy ───────────────────────────────────────────────────────

#: Fixed priority (E06-09 §7).  Evaluated top-down; a disabled capability is
#: skipped *with a reason*, never silently.
FALLBACK_PRIORITY: Tuple[str, ...] = (
    "hqsb.compiled.fused",
    "hqsb.eager.custom",
    "torch.compile.reference",
    "torch.eager.reference",
    "explicit_error",
)


@dataclass(frozen=True)
class RouteDecision:
    """requested → actual → reason, the triple every result must carry."""

    requested: str
    actual: str
    reason: str
    fallback_used: bool
    stage: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "actual": self.actual,
            "reason": self.reason,
            "fallback_used": self.fallback_used,
            "stage": self.stage,
        }


@dataclass
class FallbackPolicy:
    """Deterministic, pinnable routing (no dependence on last failure)."""

    enabled: bool = True
    strict: bool = False
    priority: Tuple[str, ...] = FALLBACK_PRIORITY
    disabled_capabilities: FrozenSet[str] = frozenset()
    reasons: Mapping[str, str] = field(default_factory=dict)

    def resolve(self, requested: str, available: Sequence[str]) -> RouteDecision:
        if self.strict:
            return RouteDecision(
                requested=requested,
                actual=requested,
                reason="STRICT_MODE_NO_FALLBACK",
                fallback_used=False,
                stage="dispatch_capability",
            )
        if not self.enabled:
            return RouteDecision(
                requested=requested,
                actual="explicit_error",
                reason="FALLBACK_DISABLED",
                fallback_used=False,
                stage="dispatch_capability",
            )
        if requested in available and requested not in self.disabled_capabilities:
            return RouteDecision(
                requested=requested,
                actual=requested,
                reason="REQUESTED_AVAILABLE",
                fallback_used=False,
            )
        for candidate in self.priority:
            if candidate == requested:
                continue
            if candidate in self.disabled_capabilities:
                continue
            if candidate in available:
                reason = self.reasons.get(candidate, f"FALLBACK_TO:{candidate}")
                return RouteDecision(
                    requested=requested,
                    actual=candidate,
                    reason=reason,
                    fallback_used=True,
                    stage="dispatch_capability",
                )
        return RouteDecision(
            requested=requested,
            actual="explicit_error",
            reason="NO_AVAILABLE_IMPLEMENTATION",
            fallback_used=False,
            stage="dispatch_capability",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strict": self.strict,
            "priority": list(self.priority),
            "disabled_capabilities": sorted(self.disabled_capabilities),
            "reasons": dict(self.reasons),
        }


@dataclass(frozen=True)
class OpcheckItem:
    """One ``torch.library.opcheck`` subtest to run for a schema (E06-01 §13).

    The item is a *plan*: the locked PyTorch version executes it, and each
    subtest is stored separately.  A single aggregate "opcheck passed" is not an
    acceptable substitute for the per-subtest record.
    """

    subtest: str
    purpose: str
    requires: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"subtest": self.subtest, "purpose": self.purpose, "requires": self.requires}


def opcheck_plan(schemas: Optional[Sequence[Any]] = None) -> Tuple[OpcheckItem, ...]:
    """The opcheck project list derived from the frozen schema set."""
    from hqsb.integration import specs as specs_mod

    selected = tuple(schemas) if schemas is not None else specs_mod.frozen_schemas()
    items: List[OpcheckItem] = []
    for schema in selected:
        prefix = schema.name
        items.extend(
            (
                OpcheckItem("schema_check", f"{prefix}: schema is well-formed", "locked torch"),
                OpcheckItem("test_schema", f"{prefix}: schema matches behaviour", "locked torch"),
                OpcheckItem(
                    "test_autograd_registration",
                    f"{prefix}: inference-only policy is explicit",
                    "locked torch",
                ),
                OpcheckItem(
                    "test_faketensor",
                    f"{prefix}: fake implementation agrees with Meta",
                    "locked torch",
                ),
                OpcheckItem(
                    "test_aot_dispatch_dynamic",
                    f"{prefix}: composes with AOTAutograd/dynamic shapes",
                    "locked torch",
                ),
            )
        )
        if schema.mutation:
            items.append(
                OpcheckItem(
                    "test_mutation",
                    f"{prefix}: declared mutation matches the kernel",
                    "locked torch",
                )
            )
    return tuple(items)


@dataclass(frozen=True)
class KernelObservation:
    """Evidence that a specific kernel symbol actually ran (E06-01 §5)."""

    run_id: str
    operator: str
    requested: str
    actual: str
    observed_kernel: str
    launch_count: int = 0
    source: str = "profiler"
    verification: str = "unverified"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "operator": self.operator,
            "requested": self.requested,
            "actual": self.actual,
            "observed_kernel": self.observed_kernel,
            "launch_count": self.launch_count,
            "source": self.source,
            "verification": self.verification,
        }


__all__ = [
    "CLAIMED_NAMESPACES",
    "CapabilityDecision",
    "CapabilityRequest",
    "DispatchKey",
    "DispatchSnapshot",
    "DispatchTableEntry",
    "FALLBACK_PRIORITY",
    "FallbackPolicy",
    "KernelObservation",
    "OpcheckItem",
    "OperatorCapability",
    "PROVIDER_KINDS",
    "RedispatchError",
    "RedispatchGuard",
    "RegistrationAction",
    "RegistrationMatrix",
    "RegistrationOutcome",
    "RegistrationRecord",
    "RouteDecision",
    "SelectionResult",
    "choose_key",
    "frozen_p0_matrix",
    "opcheck_plan",
    "redispatch_keyset",
    "select_implementation",
    "snapshot_from_matrix",
]
