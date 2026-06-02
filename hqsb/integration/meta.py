"""Meta / FakeTensor metadata contracts and symbolic shape propagation (E06-02).

The graph system must understand an operator **without allocating device memory
and without running a kernel**.  A wrong fake implementation is dangerous
precisely because capture and compilation still succeed: the generated code then
rests on a wrong shape/stride/dtype/alias assumption and fails later, or worse,
guards on an example-specific constant (E06-02 §1).

Contents:

* a tiny symbolic integer algebra (:class:`SymInt`) used to express relations
  like ``Y = [*prefix, N]`` instead of baking in example values;
* :class:`TensorMeta` — the metadata a fake tensor carries, including logical
  device and alias/mutation relations;
* :class:`MetadataContract` — per-operator input rules, output rules,
  constraints and structured rejection;
* :func:`compare_metadata` — the real-vs-fake oracle (field by field, with the
  worst mismatch retained);
* :class:`FakeCallSpy` — records what the fake path did, so "no real
  allocation / no kernel / no payload read" is evidence rather than a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from hqsb.core.errors import SchemaError

# ── symbolic integers ─────────────────────────────────────────────────────


class SymInt:
    """A symbolic integer expression (concrete values use plain ``int``)."""

    __slots__ = ("op", "args")

    def __init__(self, op: str, args: Tuple[Union[int, "SymInt"], ...]) -> None:
        self.op = op
        self.args = args

    # -- constructors --------------------------------------------------
    @staticmethod
    def symbol(name: str) -> "SymInt":
        if not name or not name.replace("_", "").isalnum():
            raise SchemaError(
                f"symbol name must be alphanumeric/underscore, got {name!r}",
                details={"field": "symbol"},
            )
        return SymInt("symbol", (name,))  # type: ignore[arg-type]

    @staticmethod
    def coerce(value: Union[int, "SymInt"]) -> Union[int, "SymInt"]:
        return value

    # -- algebra -------------------------------------------------------
    def __add__(self, other: Union[int, "SymInt"]) -> "SymInt":
        return SymInt("add", (self, other))

    def __radd__(self, other: Union[int, "SymInt"]) -> "SymInt":
        return SymInt("add", (other, self))

    def __mul__(self, other: Union[int, "SymInt"]) -> "SymInt":
        return SymInt("mul", (self, other))

    def __rmul__(self, other: Union[int, "SymInt"]) -> "SymInt":
        return SymInt("mul", (other, self))

    def __sub__(self, other: Union[int, "SymInt"]) -> "SymInt":
        return SymInt("sub", (self, other))

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, SymInt):
            return self.canonical() == other.canonical()
        if isinstance(other, int):
            return self.is_concrete() and self.concrete_value() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.canonical())

    # -- inspection ----------------------------------------------------
    def is_concrete(self) -> bool:
        return not self.symbols()

    def symbols(self) -> Tuple[str, ...]:
        if self.op == "symbol":
            return (str(self.args[0]),)
        collected: List[str] = []
        for arg in self.args:
            if isinstance(arg, SymInt):
                collected.extend(arg.symbols())
        return tuple(sorted(set(collected)))

    def canonical(self) -> str:
        if self.op == "symbol":
            return str(self.args[0])
        if self.op in ("add", "mul"):
            parts = [str(arg) if not isinstance(arg, SymInt) else arg.canonical() for arg in self.args]
            return f"({self.op} {' '.join(sorted(parts))})"
        rendered = []
        for arg in self.args:
            rendered.append(str(arg) if not isinstance(arg, SymInt) else arg.canonical())
        return f"({self.op} {' '.join(rendered)})"

    def evaluate(self, bindings: Mapping[str, int]) -> int:
        if self.op == "symbol":
            name = str(self.args[0])
            if name not in bindings:
                raise SchemaError(
                    f"symbol {name!r} is not bound",
                    details={"field": "bindings", "missing": name},
                )
            return int(bindings[name])
        values = [
            int(arg) if not isinstance(arg, SymInt) else arg.evaluate(bindings)
            for arg in self.args
        ]
        if self.op == "add":
            return sum(values)
        if self.op == "mul":
            product = 1
            for value in values:
                product *= value
            return product
        if self.op == "sub":
            return values[0] - values[1]
        raise SchemaError(f"unknown symbolic op {self.op!r}")  # pragma: no cover

    def concrete_value(self) -> int:
        if not self.is_concrete():
            raise SchemaError(
                f"expression {self.canonical()} is symbolic",
                details={"field": "symbols", "actual": list(self.symbols())},
            )
        return self.evaluate({})

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"SymInt({self.canonical()})"


Dim = Union[int, SymInt]


def dims_repr(dims: Sequence[Dim]) -> str:
    return "[" + ", ".join(d.canonical() if isinstance(d, SymInt) else str(d) for d in dims) + "]"


def dims_evaluate(dims: Sequence[Dim], bindings: Mapping[str, int]) -> Tuple[int, ...]:
    return tuple(
        int(dim) if isinstance(dim, int) else dim.evaluate(bindings) for dim in dims
    )


# ── tensor metadata ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TensorMeta:
    """The metadata a fake/meta tensor carries (E06-02 §2)."""

    shape: Tuple[Dim, ...]
    dtype: str
    device: str
    stride: Tuple[int, ...] = ()
    layout: str = "strided"
    storage_offset: int = 0
    requires_grad: bool = False
    alias_of_input: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stride and len(self.stride) != len(self.shape):
            raise SchemaError(
                "stride rank must match shape rank",
                details={"field": "stride", "shape": dims_repr(self.shape)},
            )

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_contiguous(self) -> bool:
        if not self.stride:
            return True
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.stride)):
            concrete = size if isinstance(size, int) else 1
            if concrete != 1 and stride != expected:
                return False
            expected *= concrete
        return True

    def symbols(self) -> Tuple[str, ...]:
        collected: List[str] = []
        for dim in self.shape:
            if isinstance(dim, SymInt):
                collected.extend(dim.symbols())
        return tuple(sorted(set(collected)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shape": dims_repr(self.shape),
            "shape_expressions": [dim if isinstance(dim, int) else dim.canonical() for dim in self.shape],
            "rank": self.rank,
            "dtype": self.dtype,
            "device": self.device,
            "stride": list(self.stride),
            "layout": self.layout,
            "storage_offset": self.storage_offset,
            "requires_grad": self.requires_grad,
            "alias_of_input": self.alias_of_input,
        }


# ── errors ────────────────────────────────────────────────────────────────


class MetadataError(SchemaError):
    """Metadata rejection with op/field/reason and the stage that produced it."""

    def __init__(
        self,
        message: str,
        *,
        op: str,
        field_name: str,
        reason: str,
        stage: str = "fake",
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        super().__init__(
            message,
            details={
                "op": op,
                "field": field_name,
                "reason": reason,
                "stage": stage,
                "expected": expected,
                "actual": actual,
                "pre_allocation": True,
            },
        )
        self.op = op
        self.field_name = field_name
        self.reason = reason
        self.stage = stage


# ── metadata contract ─────────────────────────────────────────────────────

STRIDE_POLICIES = ("preserve", "contiguous", "kernel_contract", "reject")


@dataclass(frozen=True)
class OutputRule:
    """How one output's metadata is derived from the inputs."""

    name: str
    shape_rule: str  # "same_as_input" | "prefix_plus" | "reduce_axis" | "explicit"
    source_input: int = 0
    axis: int = -1
    replacement: Optional[Dim] = None
    dtype_rule: str = "follow_input"  # "follow_input" | "explicit"
    dtype: Optional[str] = None
    device_rule: str = "follow_input"
    stride_policy: str = "preserve"
    alias_of_input: Optional[int] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.stride_policy not in STRIDE_POLICIES:
            raise SchemaError(
                f"output {self.name!r}: unknown stride policy {self.stride_policy!r}",
                details={"field": "stride_policy", "actual": self.stride_policy},
            )


@dataclass(frozen=True)
class Constraint:
    """A semantic constraint checked on metadata only.

    Performance-only specialisations (tile divisibility, alignment) must stay
    out of the operator contract: making them semantic guards would refuse
    shapes that could simply fall back (E06-02 §4).
    """

    name: str
    kind: str  # "semantic" | "capability"
    description: str
    field_name: str
    reason: str

    def check(self, **facts: Any) -> Optional[str]:
        """Return a failure detail or ``None``; subclasses/instances override the rule."""
        rule = CONSTRAINTS.get(self.name)
        if rule is None:
            return None
        ok = rule(facts)
        return None if ok else f"{self.field_name}: expected {self.name}"


CONSTRAINTS: Dict[str, Any] = {
    "matching_last_dim": lambda f: f.get("a_last") == f.get("b_last"),
    "rank_at_least_1": lambda f: int(f.get("rank", 0)) >= 1,
    "weight_rank_1": lambda f: int(f.get("weight_rank", 1)) == 1,
    "weight_shape_matches_hidden": lambda f: f.get("weight_len") == f.get("hidden"),
    "weight_dtype_matches_compute": lambda f: f.get("weight_dtype") in ("float16", "bfloat16", "float32"),
    "group_size_divides_or_tail_allowed": lambda f: bool(f.get("group_size")) is False
    or int(f.get("k", 0)) % int(f["group_size"]) == 0
    or bool(f.get("tail_policy_allowed", False)),
}


@dataclass
class MetadataContract:
    """Per-operator metadata contract and its fake implementation."""

    op: str
    input_rule: str
    outputs: Tuple[OutputRule, ...]
    constraints: Tuple[Constraint, ...] = ()
    dtype_policy: str = "follow_input"
    device_policy: str = "follow_input"
    allow_zero_size_dims: bool = True

    # ── input validation (metadata only, pre-allocation) ───────────────

    def validate_inputs(self, inputs: Sequence[TensorMeta]) -> None:
        for index, meta in enumerate(inputs):
            if meta.rank < 1:
                raise MetadataError(
                    f"{self.op}: input {index} has rank {meta.rank}",
                    op=self.op,
                    field_name=f"inputs[{index}].rank",
                    reason="RANK",
                    expected=">=1",
                    actual=meta.rank,
                )
            if not self.allow_zero_size_dims:
                for axis, dim in enumerate(meta.shape):
                    if isinstance(dim, int) and dim == 0:
                        raise MetadataError(
                            f"{self.op}: input {index} dim {axis} is zero-sized",
                            op=self.op,
                            field_name=f"inputs[{index}].shape[{axis}]",
                            reason="SHAPE",
                            expected=">0",
                            actual=0,
                        )
            if meta.dtype not in ("float16", "bfloat16", "float32", "float64", "uint8", "int8", "int32"):
                raise MetadataError(
                    f"{self.op}: input {index} dtype {meta.dtype!r} unsupported by the contract",
                    op=self.op,
                    field_name=f"inputs[{index}].dtype",
                    reason="DTYPE",
                    expected=["float16", "bfloat16", "float32"],
                    actual=meta.dtype,
                )

    # ── fake implementation ───────────────────────────────────────────

    def infer_outputs(
        self, inputs: Sequence[TensorMeta], **facts: Any
    ) -> Tuple[TensorMeta, ...]:
        """Derive output metadata without touching data or devices."""
        self.validate_inputs(inputs)
        for constraint in self.constraints:
            failure = constraint.check(**facts)
            if failure:
                raise MetadataError(
                    f"{self.op}: {failure}",
                    op=self.op,
                    field_name=constraint.field_name,
                    reason=constraint.reason,
                    expected=constraint.description,
                    actual=facts.get(constraint.field_name, "<unknown>"),
                )
        outputs: List[TensorMeta] = []
        for rule in self.outputs:
            source = inputs[rule.source_input]
            shape = self._shape_for(rule, source)
            dtype = source.dtype if rule.dtype_rule == "follow_input" else str(rule.dtype)
            device = source.device if rule.device_rule == "follow_input" else self.device_policy
            stride = self._stride_for(rule, source, shape)
            alias_of = None
            if rule.alias_of_input is not None:
                alias_of = rule.alias_of_input
            outputs.append(
                TensorMeta(
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    stride=stride,
                    alias_of_input=alias_of,
                )
            )
        return tuple(outputs)

    def _shape_for(self, rule: OutputRule, source: TensorMeta) -> Tuple[Dim, ...]:
        if rule.shape_rule == "same_as_input":
            return source.shape
        if rule.shape_rule == "prefix_plus":
            if rule.replacement is None:
                raise SchemaError(
                    f"{self.op}: output {rule.name!r} needs a replacement dim",
                    details={"field": "replacement"},
                )
            prefix = (
                source.shape[: rule.axis]
                if rule.axis >= 0
                else source.shape[: rule.axis]
            )
            return (*prefix, rule.replacement)
        if rule.shape_rule == "reduce_axis":
            return tuple(
                dim for index, dim in enumerate(source.shape) if index != (rule.axis % max(source.rank, 1))
            )
        raise SchemaError(
            f"{self.op}: output {rule.name!r} has unsupported shape_rule {rule.shape_rule!r}",
            details={"field": "shape_rule", "actual": rule.shape_rule},
        )

    def _stride_for(
        self, rule: OutputRule, source: TensorMeta, shape: Tuple[Dim, ...]
    ) -> Tuple[int, ...]:
        if rule.stride_policy == "preserve" and source.stride and len(source.stride) == len(shape):
            return source.stride
        if rule.stride_policy == "reject":
            if source.stride and not source.is_contiguous:
                raise MetadataError(
                    f"{self.op}: output {rule.name!r} rejects non-contiguous input",
                    op=self.op,
                    field_name=f"outputs.{rule.name}.stride",
                    reason="STRIDE_LAYOUT",
                    expected="contiguous",
                    actual=[s for s in source.stride],
                )
        return _contiguous_stride(shape)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "input_rule": self.input_rule,
            "outputs": [
                {
                    "name": rule.name,
                    "shape_rule": rule.shape_rule,
                    "stride_policy": rule.stride_policy,
                    "alias_of_input": rule.alias_of_input,
                }
                for rule in self.outputs
            ],
            "constraints": [
                {
                    "name": constraint.name,
                    "kind": constraint.kind,
                    "field": constraint.field_name,
                }
                for constraint in self.constraints
            ],
        }


def _contiguous_stride(shape: Sequence[Dim]) -> Tuple[int, ...]:
    stride: List[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        concrete = dim if isinstance(dim, int) else 1
        running *= concrete
    return tuple(reversed(stride))


# ── real-vs-fake oracle ───────────────────────────────────────────────────


@dataclass(frozen=True)
class MetadataDiff:
    """Field-level comparison of a real and a fake/meta result."""

    op: str
    mismatches: Tuple[Dict[str, Any], ...] = ()
    fields_compared: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "ok": self.ok,
            "fields_compared": self.fields_compared,
            "mismatches": [dict(item) for item in self.mismatches],
        }


_COMPARED_FIELDS = (
    "shape",
    "rank",
    "dtype",
    "device",
    "stride",
    "layout",
    "storage_offset",
    "alias_of_input",
)


def compare_metadata(op: str, real: TensorMeta, fake: TensorMeta) -> MetadataDiff:
    """Compare metadata field by field — shape equality alone is not enough.

    A real kernel that always returns contiguous output while the fake uses
    ``empty_like`` (preserving a transposed stride) is still a bug: a later
    layout optimisation or guard would rest on a false assumption (E06-02 §14).
    """
    mismatches: List[Dict[str, Any]] = []
    compared = 0
    for field_name in _COMPARED_FIELDS:
        left = getattr(real, field_name)
        right = getattr(fake, field_name)
        if field_name == "shape":
            left = dims_repr(left)
            right = dims_repr(right)
        compared += 1
        if left != right:
            mismatches.append(
                {
                    "field": field_name,
                    "real": left,
                    "fake": right,
                    "reason": f"METADATA_{field_name.upper()}_MISMATCH",
                }
            )
    return MetadataDiff(op=op, mismatches=tuple(mismatches), fields_compared=compared)


# ── guards / constraints recorded from capture ────────────────────────────


@dataclass(frozen=True)
class MetadataGuard:
    """A guard/constraint produced while capturing with fake metadata."""

    kind: str  # "shape_bound" | "equality" | "dtype" | "device" | "layout" | "data_dependent"
    expression: str
    source: str
    necessary: bool = True
    performance_only: bool = False
    introduced_by: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "expression": self.expression,
            "source": self.source,
            "necessary": self.necessary,
            "performance_only": self.performance_only,
            "introduced_by": self.introduced_by,
        }


@dataclass
class GuardSet:
    """Guards collected from a capture, with redundancy accounting."""

    guards: List[MetadataGuard] = field(default_factory=list)

    def add(self, guard: MetadataGuard) -> None:
        """Append every guard, duplicates included.

        Redundancy is an observation to report (E06-02 §11 "冗余 guard 数量"),
        not something to deduplicate away before the analysis sees it.
        """
        self.guards.append(guard)

    @property
    def data_dependent(self) -> Tuple[MetadataGuard, ...]:
        return tuple(g for g in self.guards if g.kind == "data_dependent")

    @property
    def redundant(self) -> Tuple[MetadataGuard, ...]:
        seen: Dict[Tuple[str, str], int] = {}
        for guard in self.guards:
            key = (guard.kind, guard.expression)
            seen[key] = seen.get(key, 0) + 1
        return tuple(
            guard
            for guard in self.guards
            if seen[(guard.kind, guard.expression)] > 1
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self.guards),
            "data_dependent": len(self.data_dependent),
            "redundant": len(self.redundant),
            "guards": [guard.as_dict() for guard in self.guards],
        }


# ── no-allocation evidence ────────────────────────────────────────────────


@dataclass
class FakeCallSpy:
    """Instrument a fake path and prove it did not allocate or launch.

    The spy is intentionally explicit about what it *cannot* observe
    (``device_api_calls`` is a counter the caller must feed); a zero allocation
    delta alone does not prove that no kernel ran (E06-02 §7).
    """

    op: str = ""
    real_kernel_calls: int = 0
    cuda_api_calls: int = 0
    allocations_bytes: int = 0
    payload_bytes_read: int = 0
    cpu_buffers_bytes: int = 0

    def record_real_kernel(self, name: str = "") -> None:
        self.real_kernel_calls += 1
        self._last = name

    def record_allocation(self, nbytes: int) -> None:
        self.allocations_bytes += int(nbytes)

    def record_device_api(self, name: str = "") -> None:
        self.cuda_api_calls += 1

    def record_payload_read(self, nbytes: int) -> None:
        self.payload_bytes_read += int(nbytes)

    def record_cpu_buffer(self, nbytes: int) -> None:
        self.cpu_buffers_bytes += int(nbytes)

    @property
    def clean(self) -> bool:
        return (
            self.real_kernel_calls == 0
            and self.allocations_bytes == 0
            and self.payload_bytes_read == 0
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "real_kernel_calls": self.real_kernel_calls,
            "cuda_api_calls": self.cuda_api_calls,
            "allocations_bytes": self.allocations_bytes,
            "payload_bytes_read": self.payload_bytes_read,
            "cpu_buffers_bytes": self.cpu_buffers_bytes,
            "clean": self.clean,
        }


# ── the frozen contracts ──────────────────────────────────────────────────


def rms_norm_contract() -> MetadataContract:
    return MetadataContract(
        op="hqsb::rms_norm",
        input_rule="x [..., H], weight [H]; weight broadcast over leading dims",
        outputs=(
            OutputRule(
                name="out",
                shape_rule="same_as_input",
                source_input=0,
                stride_policy="preserve",
                alias_of_input=None,
                note="kernel contract: output follows x's stride policy",
            ),
        ),
        constraints=(
            Constraint(
                name="matching_last_dim",
                kind="semantic",
                description="x's last dim equals weight length",
                field_name="weight.shape[0]",
                reason="SHAPE",
            ),
            Constraint(
                name="weight_rank_1",
                kind="semantic",
                description="weight is 1-D",
                field_name="weight.rank",
                reason="SHAPE",
            ),
            Constraint(
                name="rank_at_least_1",
                kind="semantic",
                description="x has rank >= 1",
                field_name="x.rank",
                reason="RANK",
            ),
        ),
    )


def fused_add_rms_norm_contract() -> MetadataContract:
    return MetadataContract(
        op="hqsb::fused_add_rms_norm",
        input_rule="x/residual [..., H] with equal shape, weight [H]",
        outputs=(
            OutputRule(
                name="normalized",
                shape_rule="same_as_input",
                source_input=0,
                stride_policy="preserve",
                alias_of_input=None,
            ),
            OutputRule(
                name="updated_residual",
                shape_rule="same_as_input",
                source_input=1,
                stride_policy="preserve",
                alias_of_input=None,
                note="functional contract: fresh tensor, never the input",
            ),
        ),
        constraints=(
            Constraint(
                name="matching_last_dim",
                kind="semantic",
                description="x's last dim equals residual's last dim",
                field_name="residual.shape[-1]",
                reason="SHAPE",
            ),
            Constraint(
                name="weight_shape_matches_hidden",
                kind="semantic",
                description="weight length equals hidden size",
                field_name="weight.shape[0]",
                reason="SHAPE",
            ),
        ),
    )


def dequant_linear_contract() -> MetadataContract:
    return MetadataContract(
        op="hqsb::dequant_linear",
        input_rule="x [..., K]; packed codes stay opaque; logical N carried by out_features",
        outputs=(
            OutputRule(
                name="out",
                shape_rule="prefix_plus",
                source_input=0,
                axis=-1,
                replacement=SymInt.symbol("N"),
                stride_policy="contiguous",
                alias_of_input=None,
                note="logical N replaces the last dim; packed bytes never leak in",
            ),
        ),
        constraints=(
            Constraint(
                name="matching_last_dim",
                kind="semantic",
                description="x last dim equals logical K of the artifact",
                field_name="weight.logical_K",
                reason="SHAPE",
            ),
            Constraint(
                name="group_size_divides_or_tail_allowed",
                kind="capability",
                description="K is divisible by group size, or the tail policy allows it",
                field_name="quant.group_size",
                reason="QUANT_POLICY",
            ),
        ),
    )


def contract_by_name(op: str) -> MetadataContract:
    contracts = {
        "hqsb::rms_norm": rms_norm_contract,
        "hqsb::fused_add_rms_norm": fused_add_rms_norm_contract,
        "hqsb::dequant_linear": dequant_linear_contract,
    }
    if op not in contracts:
        raise SchemaError(
            f"no metadata contract for {op!r}",
            details={"known": sorted(contracts)},
        )
    return contracts[op]()


def frozen_contracts() -> Tuple[MetadataContract, ...]:
    return (
        rms_norm_contract(),
        fused_add_rms_norm_contract(),
        dequant_linear_contract(),
    )


def contract_table() -> Dict[str, Dict[str, Any]]:
    return {contract.op: contract.as_dict() for contract in frozen_contracts()}


__all__ = [
    "CONSTRAINTS",
    "Constraint",
    "Dim",
    "FakeCallSpy",
    "GuardSet",
    "MetadataContract",
    "MetadataDiff",
    "MetadataError",
    "MetadataGuard",
    "OutputRule",
    "STRIDE_POLICIES",
    "SymInt",
    "TensorMeta",
    "compare_metadata",
    "contract_by_name",
    "contract_table",
    "dequant_linear_contract",
    "dims_evaluate",
    "dims_repr",
    "frozen_contracts",
    "fused_add_rms_norm_contract",
    "rms_norm_contract",
]
