"""Canonical request semantics and capability negotiation (E07-01 §2–§4).

Every runtime adapter must express the *same* request through this canonical
:class:`RequestSpec`, and must expose which fields it can honour through a
:class:`CapabilityReport`.  Two rules are enforced in code rather than prose:

* an unsupported field is either **rejected** (``UNSUPPORTED_REJECT``) or
  resolved by an explicitly declared, measured fallback (``UNSUPPORTED_FALLBACK``
  / ``EMULATED``) — a silent default is impossible because
  :class:`ResolvedConfig` refuses a requested/actual mismatch without a reason;
* ``UNKNOWN`` never counts as supported: by default it blocks the claim
  (``require_supported`` raises), which is the details-README §4 rule
  "UNKNOWN 默认拒绝相关 claim".

Comparisons across backends use **token IDs** (``input_token_hash``), never raw
text: each backend tokenizing its own string is the first mistake E07-01 §10
lists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import CapabilityError, ConfigError

# ── capability states (E07-01 §4) ──────────────────────────────────────────

SUPPORTED_EXACT = "SUPPORTED_EXACT"
SUPPORTED_WITH_CONSTRAINT = "SUPPORTED_WITH_CONSTRAINT"
EMULATED = "EMULATED"
UNSUPPORTED_REJECT = "UNSUPPORTED_REJECT"
UNSUPPORTED_FALLBACK = "UNSUPPORTED_FALLBACK"
UNKNOWN = "UNKNOWN"

CAPABILITY_STATES: Tuple[str, ...] = (
    SUPPORTED_EXACT,
    SUPPORTED_WITH_CONSTRAINT,
    EMULATED,
    UNSUPPORTED_REJECT,
    UNSUPPORTED_FALLBACK,
    UNKNOWN,
)

#: States that permit a *measured* claim; ``EMULATED``/``SUPPORTED_WITH_CONSTRAINT``
#: carry a constraint string that a report must quote.
CLAIMABLE_STATES = (SUPPORTED_EXACT, SUPPORTED_WITH_CONSTRAINT, EMULATED)

#: Fields whose state must be reported for every backend (E07-01 §2).
REQUEST_FIELDS: Tuple[str, ...] = (
    "model_artifact",
    "tokenizer",
    "chat_template",
    "input_token_ids",
    "max_new_tokens",
    "min_new_tokens",
    "sampling_mode",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "repetition_penalty",
    "logprobs",
    "stop_token_ids",
    "eos_handling",
    "stop_string_layer",
    "streaming",
    "timeout",
    "cancel",
    "precision",
    "quant_artifact",
    "prefix_cache",
    "backend_extensions",
)


@dataclass(frozen=True)
class SamplingSpec:
    """Sampling semantics, frozen before any comparison (E07-01 §2)."""

    mode: str = "greedy"
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None
    generator: str = "runtime_default"
    repetition_penalty: float = 1.0
    logprobs: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("greedy", "sampling"):
            raise ConfigError(
                f"unknown sampling mode {self.mode!r}; E07-01 compares either a "
                "greedy path or a distribution path, never a mixture",
                details={"field": "mode"},
            )
        if self.mode == "greedy" and self.temperature not in (0.0,):
            raise ConfigError(
                "greedy mode must declare temperature=0; otherwise the same "
                "RequestSpec could mean two different decoding rules",
                details={"field": "temperature"},
            )
        if self.mode == "sampling" and self.temperature <= 0:
            raise ConfigError(
                "sampling mode needs temperature > 0",
                details={"field": "temperature"},
            )
        if self.top_k < 0 or not 0.0 < self.top_p <= 1.0:
            raise ConfigError("invalid top_k/top_p", details={"field": "top_p"})

    @property
    def comparable_by_tokens(self) -> bool:
        """Only greedy decoding can be compared token-for-token (E07-01 §5)."""
        return self.mode == "greedy"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "generator": self.generator,
            "repetition_penalty": self.repetition_penalty,
            "logprobs": self.logprobs,
        }


@dataclass(frozen=True)
class StopSpec:
    """Termination rules and the layer that evaluates them."""

    max_new_tokens: int
    min_new_tokens: int = 0
    eos_token_id: Optional[int] = None
    stop_token_ids: Tuple[int, ...] = ()
    stop_string_layer: str = "none"
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ConfigError("max_new_tokens must be positive")
        if self.min_new_tokens < 0 or self.min_new_tokens > self.max_new_tokens:
            raise ConfigError("min_new_tokens must be within [0, max_new_tokens]")
        if self.stop_string_layer not in ("none", "tokenizer", "runtime", "gateway"):
            raise ConfigError(
                "stop_string_layer must be explicit; 'the text looked finished' is "
                "not a comparable stop rule (E07-01 §10)",
                details={"field": "stop_string_layer"},
            )
        if self.stop_string_layer == "gateway":
            raise ConfigError(
                "string-level stopping in a gateway is out of the S07 model-core "
                "boundary (details README §5); compare it inside the runtime",
                details={"field": "stop_string_layer"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "eos_token_id": self.eos_token_id,
            "stop_token_ids": list(self.stop_token_ids),
            "stop_string_layer": self.stop_string_layer,
            "ignore_eos": self.ignore_eos,
        }


def token_hash(token_ids: Sequence[int]) -> str:
    """Stable hash of a token-ID sequence (the comparison unit of E07-01)."""
    payload = ",".join(str(int(token)) for token in token_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    """Model/tokenizer/precision identity that must match across backends."""

    model_id: str
    model_manifest_sha256: str
    revision: str
    tokenizer_id: str
    chat_template_hash: str
    precision: str
    quant_artifact_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("model_id", "model_manifest_sha256", "revision", "tokenizer_id"):
            if not getattr(self, name):
                raise ConfigError(
                    f"model identity requires {name!r}; a backend that auto-selects "
                    "a different revision must be refused, not silently compared "
                    "(E07-01 §3 load contract)",
                    details={"field": name},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_manifest_sha256": self.model_manifest_sha256,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "chat_template_hash": self.chat_template_hash,
            "precision": self.precision,
            "quant_artifact_hash": self.quant_artifact_hash,
        }


@dataclass(frozen=True)
class RequestSpec:
    """The canonical request every S07 backend must execute identically."""

    request_id: str
    identity: ModelIdentity
    input_token_ids: Tuple[int, ...]
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    stop: StopSpec = field(default_factory=lambda: StopSpec(max_new_tokens=1))
    streaming: bool = False
    timeout_s: Optional[float] = None
    prefix_cache_enabled: bool = False
    shared_prefix_group: str = ""
    priority: int = 0
    backend_extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ConfigError("request_id must not be empty")
        if not self.input_token_ids:
            raise ConfigError(
                "a request must carry token IDs: S07 compares tokenized requests so "
                "that each backend cannot apply its own text preprocessing "
                "(E07-01 §2)",
                details={"field": "input_token_ids"},
            )
        if any(int(token) < 0 for token in self.input_token_ids):
            raise ConfigError("token IDs must be non-negative")

    @property
    def input_token_hash(self) -> str:
        return token_hash(self.input_token_ids)

    @property
    def input_tokens(self) -> int:
        return len(self.input_token_ids)

    @property
    def request_hash(self) -> str:
        """Identity of the whole request (E07-10 §3 freezes it in the trace)."""
        payload = {
            "request_id": self.request_id,
            "identity": self.identity.as_dict(),
            "input_token_hash": self.input_token_hash,
            "input_tokens": self.input_tokens,
            "sampling": self.sampling.as_dict(),
            "stop": self.stop.as_dict(),
            "streaming": self.streaming,
            "timeout_s": self.timeout_s,
            "prefix_cache_enabled": self.prefix_cache_enabled,
            "shared_prefix_group": self.shared_prefix_group,
            "priority": self.priority,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "identity": self.identity.as_dict(),
            "input_tokens": self.input_tokens,
            "input_token_hash": self.input_token_hash,
            "sampling": self.sampling.as_dict(),
            "stop": self.stop.as_dict(),
            "streaming": self.streaming,
            "timeout_s": self.timeout_s,
            "prefix_cache_enabled": self.prefix_cache_enabled,
            "shared_prefix_group": self.shared_prefix_group,
            "priority": self.priority,
            "request_hash": self.request_hash,
        }


# ── capability negotiation ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CapabilityField:
    """One field's negotiated state on one backend."""

    name: str
    state: str
    constraint: str = ""
    reason: str = ""
    emulated_cost_note: str = ""

    def __post_init__(self) -> None:
        if self.state not in CAPABILITY_STATES:
            raise ConfigError(
                f"unknown capability state {self.state!r}",
                details={"field": "state", "allowed": list(CAPABILITY_STATES)},
            )
        if self.state in (UNSUPPORTED_REJECT, UNSUPPORTED_FALLBACK, EMULATED, UNKNOWN):
            if not self.reason:
                raise ConfigError(
                    f"capability field {self.name!r} in state {self.state} needs a "
                    "reason; an unexplained degradation is exactly what the project "
                    "forbids",
                    details={"field": "reason"},
                )
        if self.state == SUPPORTED_WITH_CONSTRAINT and not self.constraint:
            raise ConfigError(
                f"{self.name!r} is SUPPORTED_WITH_CONSTRAINT but carries no "
                "constraint string",
                details={"field": "constraint"},
            )
        if self.state == EMULATED and not self.emulated_cost_note:
            raise ConfigError(
                f"{self.name!r} is EMULATED: the emulation cost must be measured "
                "and recorded (E07-01 §4)",
                details={"field": "emulated_cost_note"},
            )

    @property
    def claimable(self) -> bool:
        return self.state in CLAIMABLE_STATES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "constraint": self.constraint,
            "reason": self.reason,
            "emulated_cost_note": self.emulated_cost_note,
        }


@dataclass
class CapabilityReport:
    """The probe result of one backend, one entry per :data:`REQUEST_FIELDS`."""

    backend_id: str
    fields: Dict[str, CapabilityField] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend_id:
            raise ConfigError("capability report needs a backend_id")

    def declare(self, field_name: str, state: str, **kwargs: Any) -> CapabilityField:
        if field_name not in REQUEST_FIELDS:
            raise ConfigError(
                f"{field_name!r} is not a declared request field; adding fields "
                "silently would let a backend hide an unsupported parameter",
                details={"field": field_name},
            )
        entry = CapabilityField(name=field_name, state=state, **kwargs)
        self.fields[field_name] = entry
        return entry

    def state_of(self, field_name: str) -> str:
        entry = self.fields.get(field_name)
        return entry.state if entry else UNKNOWN

    def missing_fields(self) -> List[str]:
        """Fields with no probe entry: UNKNOWN by default, never 'supported'."""
        return [name for name in REQUEST_FIELDS if name not in self.fields]

    def require_supported(self, field_name: str) -> CapabilityField:
        """Raise unless the field can be honoured with a measured claim."""
        entry = self.fields.get(field_name)
        if entry is None:
            raise CapabilityError(
                f"{self.backend_id}: no capability probe result for {field_name!r}; "
                "an unprobed field is UNKNOWN and therefore not supported "
                "(capability must come from a probe, not from documentation)",
                details={"backend": self.backend_id, "field": field_name},
            )
        if not entry.claimable:
            raise CapabilityError(
                f"{self.backend_id}: {field_name!r} is {entry.state}: {entry.reason}",
                details={
                    "backend": self.backend_id,
                    "field": field_name,
                    "state": entry.state,
                },
            )
        return entry

    def matrix(self) -> List[Dict[str, Any]]:
        return [
            {
                "backend": self.backend_id,
                **self.fields[name].as_dict(),
            }
            for name in REQUEST_FIELDS
            if name in self.fields
        ]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "fields": {name: entry.as_dict() for name, entry in self.fields.items()},
            "missing_fields": self.missing_fields(),
        }


@dataclass(frozen=True)
class ResolvedParameter:
    """One requested → actual parameter resolution with its reason."""

    name: str
    requested: Any
    actual: Any
    reason: str = ""

    @property
    def changed(self) -> bool:
        return self.requested != self.actual

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "requested": self.requested,
            "actual": self.actual,
            "reason": self.reason,
            "changed": self.changed,
        }


@dataclass
class ResolvedConfig:
    """Requested/actual parameter table; silent changes are refused."""

    backend_id: str
    parameters: Dict[str, ResolvedParameter] = field(default_factory=dict)

    def record(self, name: str, requested: Any, actual: Any, reason: str = "") -> None:
        if requested != actual and not reason:
            raise ConfigError(
                f"{self.backend_id}: parameter {name!r} requested={requested!r} but "
                f"actual={actual!r} without a reason; silent degradation is not "
                "allowed (AGENTS.md / control plane §3.6)",
                details={"backend": self.backend_id, "field": name},
            )
        self.parameters[name] = ResolvedParameter(
            name=name, requested=requested, actual=actual, reason=reason
        )

    def changed_parameters(self) -> List[ResolvedParameter]:
        return [entry for entry in self.parameters.values() if entry.changed]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "parameters": {
                name: entry.as_dict() for name, entry in self.parameters.items()
            },
            "changed": [entry.name for entry in self.changed_parameters()],
        }


def resolve_request(
    request: RequestSpec, capability: CapabilityReport
) -> ResolvedConfig:
    """Map a request onto one backend, refusing silent parameter changes.

    Both halves of E07-01 §4 are enforced: claim-relevant fields must be
    claimable, and every field whose actual value differs from the request must
    carry a reason.
    """
    resolved = ResolvedConfig(backend_id=capability.backend_id)
    resolved.record("model_artifact", request.identity.model_id, request.identity.model_id)
    resolved.record("precision", request.identity.precision, request.identity.precision)
    resolved.record("input_token_ids", request.input_token_hash, request.input_token_hash)
    resolved.record("max_new_tokens", request.stop.max_new_tokens, request.stop.max_new_tokens)
    resolved.record("streaming", request.streaming, request.streaming)

    for field_name in ("model_artifact", "input_token_ids", "max_new_tokens", "precision"):
        capability.require_supported(field_name)

    # Sampling: a backend that cannot do the requested mode must fall back
    # explicitly (greedy under sampling is a *different* experiment).
    sampling_entry = capability.fields.get("sampling_mode")
    if sampling_entry is not None and sampling_entry.state == UNSUPPORTED_FALLBACK:
        resolved.record(
            "sampling_mode",
            request.sampling.mode,
            "greedy",
            reason=sampling_entry.reason or "backend does not support sampling",
        )
    elif sampling_entry is not None and sampling_entry.state == UNSUPPORTED_REJECT:
        raise CapabilityError(
            f"{capability.backend_id}: sampling mode {request.sampling.mode!r} is "
            f"rejected: {sampling_entry.reason}",
            details={"field": "sampling_mode"},
        )
    else:
        resolved.record("sampling_mode", request.sampling.mode, request.sampling.mode)

    if request.streaming:
        capability.require_supported("streaming")
    if request.prefix_cache_enabled:
        capability.require_supported("prefix_cache")
    if request.identity.quant_artifact_hash:
        capability.require_supported("quant_artifact")
    if request.timeout_s is not None:
        capability.require_supported("timeout")
    return resolved


@dataclass(frozen=True)
class BackendSpec:
    """One participating backend and the role it plays (E07-01 §8 step 1)."""

    backend_id: str
    role: str
    version: str
    commit: str
    source_identity: str
    adapter_module: str
    hardware: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.role not in (
            "reference",
            "edge",
            "cloud_primary",
            "cloud_secondary",
        ):
            raise ConfigError(
                f"unknown backend role {self.role!r}; S07 picks exactly one "
                "cloud_primary for source-level depth (details README §6)",
                details={"field": "role"},
            )
        for name in ("backend_id", "version", "commit", "adapter_module"):
            if not getattr(self, name):
                raise ConfigError(
                    f"backend spec requires {name!r}: the runtime version/commit and "
                    "adapter are part of the run identity",
                    details={"field": name},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "role": self.role,
            "version": self.version,
            "commit": self.commit,
            "source_identity": self.source_identity,
            "adapter_module": self.adapter_module,
            "hardware": self.hardware,
            "notes": self.notes,
        }


def assert_single_primary(specs: Sequence[BackendSpec]) -> str:
    """Exactly one source-level primary runtime (details README §6)."""
    primaries = [spec.backend_id for spec in specs if spec.role == "cloud_primary"]
    if len(primaries) != 1:
        raise ConfigError(
            "S07 requires exactly one cloud_primary runtime to instrument at source "
            f"level and at most one cloud_secondary as a fair baseline; got "
            f"{primaries}",
            details={"field": "role"},
        )
    return primaries[0]


__all__ = [
    "CAPABILITY_STATES",
    "CLAIMABLE_STATES",
    "CapabilityField",
    "CapabilityReport",
    "EMULATED",
    "BackendSpec",
    "ModelIdentity",
    "REQUEST_FIELDS",
    "RequestSpec",
    "ResolvedConfig",
    "ResolvedParameter",
    "SUPPORTED_EXACT",
    "SUPPORTED_WITH_CONSTRAINT",
    "SamplingSpec",
    "StopSpec",
    "UNKNOWN",
    "UNSUPPORTED_FALLBACK",
    "UNSUPPORTED_REJECT",
    "assert_single_primary",
    "resolve_request",
    "token_hash",
]
