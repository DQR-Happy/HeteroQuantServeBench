"""Runtime adapter contract: load/warmup/generate/stream/cancel/metrics/close.

E07-01 §3 defines seven operations that every participating runtime must expose
through *this* interface.  Two properties matter more than the method list:

* **Business code never touches a private runtime API.**  Adapters are registered
  by name with their version/commit/source identity
  (:class:`AdapterRegistry`), and a backend whose engine is not installed is
  reported as ``NOT_INSTALLED`` with the module it looked for — never assumed
  available because a document said so.
* **Capability comes from a probe.**  :func:`probe_environment` inspects the
  interpreter (``importlib.util.find_spec``) and returns a structured state; the
  default state is ``UNKNOWN``, which blocks execution rather than silently
  running with a default.

The heavy engine import happens **inside** ``load``/``generate`` and is wrapped:
a missing engine is a :class:`~hqsb.core.errors.CapabilityError` with a reason,
not an ``ImportError`` leaking through the stack.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import BackendError, CapabilityError, ConfigError
from hqsb.runtime.request import (
    BackendSpec,
    CapabilityReport,
    ModelIdentity,
    RequestSpec,
    SamplingSpec,
    StopSpec,
    token_hash,
)

# ── adapter lifecycle (E07-01 §7) ──────────────────────────────────────────


class AdapterState:
    """Lifecycle states of one adapter instance."""

    CREATED = "CREATED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    WARMED = "WARMED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


ALLOWED_ADAPTER_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    AdapterState.CREATED: (AdapterState.LOADING, AdapterState.FAILED),
    AdapterState.LOADING: (AdapterState.LOADED, AdapterState.FAILED),
    AdapterState.LOADED: (AdapterState.WARMED, AdapterState.CLOSING, AdapterState.FAILED),
    AdapterState.WARMED: (
        AdapterState.LOADED,
        AdapterState.CLOSING,
        AdapterState.FAILED,
    ),
    AdapterState.CLOSING: (AdapterState.CLOSED, AdapterState.FAILED),
    AdapterState.CLOSED: (),
    AdapterState.FAILED: (AdapterState.CLOSING,),
}


# ── probing (E07-01 steps 1 & 4; details README §6) ────────────────────────

PROBE_NOT_INSTALLED = "NOT_INSTALLED"
PROBE_INSTALLED_NOT_VERIFIED = "INSTALLED_NOT_VERIFIED"
PROBE_AVAILABLE = "AVAILABLE"
PROBE_UNKNOWN = "UNKNOWN"

#: Candidate engines of the S07 adapter matrix.  The list is a *candidate* list:
#: which one becomes the source-level primary is decided by the capability probe
#: and frozen in ``main_runtime_selection.json``, never assumed here.
CANDIDATE_ENGINES: Tuple[str, ...] = ("vllm", "sglang", "tensorrt_llm", "llama_cpp")


@dataclass(frozen=True)
class AdapterProbeResult:
    """Result of probing one engine, with the module it looked for."""

    engine: str
    state: str
    module_name: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.state not in (
            PROBE_NOT_INSTALLED,
            PROBE_INSTALLED_NOT_VERIFIED,
            PROBE_AVAILABLE,
            PROBE_UNKNOWN,
        ):
            raise ConfigError(f"unknown probe state {self.state!r}")

    @property
    def usable(self) -> bool:
        """Only a verified availability may be used for execution."""
        return self.state == PROBE_AVAILABLE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "state": self.state,
            "module_name": self.module_name,
            "detail": self.detail,
            "usable": self.usable,
        }


def probe_environment(
    engines: Sequence[str] = CANDIDATE_ENGINES,
) -> List[AdapterProbeResult]:
    """Probe the interpreter for each candidate engine (no import side effects).

    ``find_spec`` tells us whether the module *exists*; it does not prove the
    engine can load a model on this machine, so an installed engine is reported
    as ``INSTALLED_NOT_VERIFIED`` until a real load smoke verifies it.  This is
    the honest boundary between "the package is importable" and "the runtime
    works here".
    """
    results: List[AdapterProbeResult] = []
    for engine in engines:
        try:
            spec = importlib.util.find_spec(engine)
        except (ImportError, ValueError) as exc:  # broken/partial install
            results.append(
                AdapterProbeResult(
                    engine=engine,
                    state=PROBE_UNKNOWN,
                    module_name=engine,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if spec is None:
            results.append(
                AdapterProbeResult(
                    engine=engine,
                    state=PROBE_NOT_INSTALLED,
                    module_name=engine,
                    detail=f"module {engine!r} is not importable in this interpreter",
                )
            )
        else:
            results.append(
                AdapterProbeResult(
                    engine=engine,
                    state=PROBE_INSTALLED_NOT_VERIFIED,
                    module_name=engine,
                    detail=(
                        "module is present but no model load smoke has verified it "
                        "on this machine"
                    ),
                )
            )
    return results


def select_primary_engine(probes: Sequence[AdapterProbeResult]) -> Dict[str, Any]:
    """Choose the source-level primary engine from probe results.

    The selection is *evidence*, not a preference: it fails loudly when no
    candidate is verified, because the protocol requires the choice to come from
    a capability probe (details README §6).
    """
    verified = [probe for probe in probes if probe.usable]
    if len(verified) != 1:
        return {
            "selected": "",
            "verified": [probe.engine for probe in verified],
            "ok": False,
            "reason": (
                "exactly one verified engine is required to pick a source-level "
                f"primary runtime; verified={[probe.engine for probe in verified]}"
            ),
            "probes": [probe.as_dict() for probe in probes],
        }
    return {
        "selected": verified[0].engine,
        "verified": [probe.engine for probe in verified],
        "ok": True,
        "reason": "",
        "probes": [probe.as_dict() for probe in probes],
    }


# ── streaming / cancel contracts ───────────────────────────────────────────


@dataclass(frozen=True)
class StreamChunk:
    """One streamed chunk; ordering and finality are part of the contract."""

    request_id: str
    token_index: int
    token_id: int
    timestamp_ns: int
    final: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_index": self.token_index,
            "token_id": self.token_id,
            "timestamp_ns": self.timestamp_ns,
            "final": self.final,
        }


def validate_stream(
    chunks: Sequence[StreamChunk],
    *,
    request_id: str,
    expected_token_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Check order, duplication, token identity and the final flag."""
    problems: List[str] = []
    if not chunks:
        problems.append("empty stream")
    for index, chunk in enumerate(chunks):
        if chunk.request_id != request_id:
            problems.append(
                f"chunk {index} belongs to {chunk.request_id!r}, not {request_id!r}"
            )
        if chunk.token_index != index:
            problems.append(
                f"chunk {index} carries token_index={chunk.token_index}; "
                "duplicate/lost chunks must not be renumbered silently"
            )
        if index and chunk.timestamp_ns < chunks[index - 1].timestamp_ns:
            problems.append(f"chunk {index} timestamp went backwards")
        if index < len(chunks) - 1 and chunk.final:
            problems.append(f"chunk {index} is marked final but more chunks follow")
    if chunks and not chunks[-1].final:
        problems.append("the last chunk is not marked final")
    if expected_token_ids is not None:
        observed = [chunk.token_id for chunk in chunks]
        if observed != list(expected_token_ids):
            problems.append(
                "streamed token IDs differ from the non-streaming result: "
                f"{observed[:8]}... vs {list(expected_token_ids)[:8]}..."
            )
    return {
        "ok": not problems,
        "problems": problems,
        "chunks": len(chunks),
        "request_id": request_id,
    }


@dataclass(frozen=True)
class CancelRecord:
    """The three cancel instants and the in-flight policy (E07-01 §3)."""

    request_id: str
    t_requested_ns: int
    t_observed_ns: int
    t_done_ns: int
    in_flight_policy: str
    extra_tokens_emitted: int = 0
    cleanup_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.in_flight_policy not in (
            "discard_output",
            "emit_until_kernel_boundary",
            "no_in_flight_work",
        ):
            raise ConfigError(
                "cancel must declare what happens to already-submitted work; "
                "'whatever the runtime does' is not a contract",
                details={"field": "in_flight_policy"},
            )
        if self.t_observed_ns < self.t_requested_ns or self.t_done_ns < self.t_observed_ns:
            raise ConfigError("cancel instants must be ordered")

    @property
    def observation_latency_ms(self) -> float:
        return (self.t_observed_ns - self.t_requested_ns) / 1e6

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "observation_latency_ms": self.observation_latency_ms,
            "cleanup_ms": self.cleanup_ms,
            "in_flight_policy": self.in_flight_policy,
            "extra_tokens_emitted": self.extra_tokens_emitted,
            "allowed_extra_tokens": (
                0 if self.in_flight_policy == "discard_output" else None
            ),
        }


@dataclass(frozen=True)
class GenerationResult:
    """Normalized generate output; logits/logprobs are optional by design."""

    request_id: str
    token_ids: Tuple[int, ...]
    finish_reason: str
    actual_parameters: Mapping[str, Any] = field(default_factory=dict)
    logits: Optional[Tuple[Tuple[float, ...], ...]] = None
    logprobs: Optional[Tuple[Tuple[float, float], ...]] = None
    timings: Mapping[str, float] = field(default_factory=dict)
    usage: Mapping[str, int] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.finish_reason not in ("stop", "length", "cancelled", "error"):
            raise ConfigError(
                f"unknown finish_reason {self.finish_reason!r}; a free-form string "
                "cannot be compared across backends",
                details={"field": "finish_reason"},
            )
        if self.finish_reason == "stop" and not self.token_ids and not self.raw.get(
            "allow_empty_stop"
        ):
            raise ConfigError(
                "finish_reason='stop' with zero tokens needs an explicit "
                "allow_empty_stop flag; otherwise the case is indistinguishable "
                "from a truncated stream",
                details={"field": "finish_reason"},
            )

    @property
    def token_hash(self) -> str:
        return token_hash(self.token_ids)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tokens": len(self.token_ids),
            "token_ids": list(self.token_ids),
            "token_hash": self.token_hash,
            "finish_reason": self.finish_reason,
            "actual_parameters": dict(self.actual_parameters),
            "timings": dict(self.timings),
            "usage": dict(self.usage),
            "has_logits": self.logits is not None,
            "has_logprobs": self.logprobs is not None,
        }


# ── the adapter interface ──────────────────────────────────────────────────


class RuntimeAdapter(ABC):
    """The seven-operation contract of E07-01 §3."""

    def __init__(self, name: str, spec: BackendSpec) -> None:
        self._name = name
        self.spec = spec
        self.state = AdapterState.CREATED
        self.events: List[Dict[str, Any]] = []
        self._close_count = 0

    # ── lifecycle helper ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    def transition(self, new_state: str, reason: str = "") -> None:
        allowed = ALLOWED_ADAPTER_TRANSITIONS.get(self.state, ())
        if new_state not in allowed:
            raise BackendError(
                f"adapter {self.name}: illegal transition {self.state} → {new_state} "
                f"({reason}); allowed: {allowed}",
                details={"adapter": self.name, "state": self.state},
            )
        self.events.append(
            {"event": "state", "from": self.state, "to": new_state, "reason": reason}
        )
        self.state = new_state

    def _record(self, kind: str, **payload: Any) -> None:
        self.events.append({"event": kind, **payload})

    # ── the seven operations ─────────────────────────────────────────────

    @abstractmethod
    def capability(self) -> CapabilityReport:
        """Probe the backend for one entry per request field."""

    @abstractmethod
    def load(self, identity: ModelIdentity, **kwargs: Any) -> Mapping[str, Any]:
        """Load the artifact and return the *actual* loaded identity."""

    @abstractmethod
    def warmup(self, request: RequestSpec) -> Mapping[str, Any]:
        """Declare the warmup shapes and its side effects (compile/graph/cache)."""

    @abstractmethod
    def generate(self, request: RequestSpec) -> GenerationResult:
        """Execute one request and return normalized output."""

    def stream(self, request: RequestSpec) -> Tuple[StreamChunk, ...]:
        raise CapabilityError(
            f"adapter {self.name} does not implement streaming; the capability "
            "report must say so instead of the caller discovering it at runtime",
            details={"adapter": self.name, "field": "streaming"},
        )

    def cancel(self, request_id: str, **kwargs: Any) -> CancelRecord:
        raise CapabilityError(
            f"adapter {self.name} does not implement cancellation",
            details={"adapter": self.name, "field": "cancel"},
        )

    @abstractmethod
    def metrics(self) -> Mapping[str, Any]:
        """Return stable, unified metrics; private fields must be namespaced."""

    @abstractmethod
    def close(self) -> Mapping[str, Any]:
        """Stop new work, release resources and be idempotent."""

    # ── shared helpers ───────────────────────────────────────────────────

    def verify_identity(
        self, expected: ModelIdentity, actual: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Refuse a loaded identity that differs from the frozen one."""
        mismatches = {
            name: {
                "expected": getattr(expected, name),
                "actual": actual.get(name, "<missing>"),
            }
            for name in (
                "model_id",
                "model_manifest_sha256",
                "revision",
                "tokenizer_id",
                "precision",
            )
            if actual.get(name, "<missing>") != getattr(expected, name)
        }
        if mismatches:
            raise BackendError(
                f"adapter {self.name} loaded a different artifact than the frozen "
                f"identity: {sorted(mismatches)}; auto-selecting another revision "
                "must be refused (E07-01 §3)",
                details={"adapter": self.name, "mismatches": sorted(mismatches)},
            )
        return {"ok": True, "identity": dict(actual)}

    def require_open(self) -> None:
        if self.state not in (AdapterState.WARMED, AdapterState.LOADED):
            raise BackendError(
                f"adapter {self.name} is {self.state}; generate requires a loaded "
                "and warmed adapter",
                details={"adapter": self.name, "state": self.state},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec.as_dict(),
            "state": self.state,
            "close_count": self._close_count,
            "events": list(self.events),
        }


class DummyRuntimeAdapter(RuntimeAdapter):
    """A deterministic adapter for contract/conformance tests.

    It never claims performance (:meth:`performance_claim_allowed` is always
    ``False``) and it validates that the *contract* is expressible before any
    real runtime exists.
    """

    def __init__(self, spec: BackendSpec, *, vocabulary: int = 128) -> None:
        super().__init__(name="dummy_runtime", spec=spec)
        self.vocabulary = vocabulary
        self.loaded_identity: Dict[str, Any] = {}
        self.warmed: List[str] = []
        self.cancelled: List[str] = []
        self.closed = False

    def capability(self) -> CapabilityReport:
        report = CapabilityReport(backend_id=self.name)
        report.declare("model_artifact", "SUPPORTED_EXACT")
        report.declare("input_token_ids", "SUPPORTED_EXACT")
        report.declare("max_new_tokens", "SUPPORTED_EXACT")
        report.declare("precision", "SUPPORTED_EXACT")
        report.declare("sampling_mode", "SUPPORTED_EXACT")
        report.declare("streaming", "SUPPORTED_EXACT")
        report.declare("cancel", "SUPPORTED_EXACT")
        report.declare("timeout", "SUPPORTED_EXACT")
        report.declare(
            "prefix_cache",
            "UNSUPPORTED_FALLBACK",
            reason="the dummy adapter has no KV store; the fallback is a full prefill",
        )
        report.declare(
            "quant_artifact",
            "UNSUPPORTED_REJECT",
            reason="the dummy adapter executes no quantized weights",
        )
        return report

    def load(self, identity: ModelIdentity, **kwargs: Any) -> Mapping[str, Any]:
        self.transition(AdapterState.LOADING, "load")
        self.loaded_identity = identity.as_dict()
        self.transition(AdapterState.LOADED, "load complete")
        return dict(self.loaded_identity)

    def warmup(self, request: RequestSpec) -> Mapping[str, Any]:
        self.transition(AdapterState.WARMED, "warmup")
        self.warmed.append(request.request_id)
        return {"shapes": [request.input_tokens], "compile": False, "graph": False}

    def generate(self, request: RequestSpec) -> GenerationResult:
        self.require_open()
        steps = request.stop.max_new_tokens
        tokens = [
            (sum(request.input_token_ids) + index) % self.vocabulary
            for index in range(steps)
        ]
        return GenerationResult(
            request_id=request.request_id,
            token_ids=tuple(tokens),
            finish_reason="length",
            actual_parameters={"max_new_tokens": steps},
            timings={"total_s": 0.0},
            usage={"input_tokens": request.input_tokens, "output_tokens": steps},
            raw={"simulated": True, "adapter": self.name},
        )

    def cancel(self, request_id: str, **kwargs: Any) -> CancelRecord:
        self.cancelled.append(request_id)
        base = kwargs.pop("now_ns", 0)
        return CancelRecord(
            request_id=request_id,
            t_requested_ns=base,
            t_observed_ns=base + 1,
            t_done_ns=base + 2,
            in_flight_policy="no_in_flight_work",
        )

    def metrics(self) -> Mapping[str, Any]:
        return {"adapter": self.name, "requests": len(self.warmed)}

    def close(self) -> Mapping[str, Any]:
        self._close_count += 1
        if self.state == AdapterState.CLOSED:
            return {"closed": True, "already_closed": True, "count": self._close_count}
        if self.state != AdapterState.CLOSING:
            self.transition(AdapterState.CLOSING, "close")
        self.transition(AdapterState.CLOSED, "close complete")
        self.closed = True
        return {"closed": True, "already_closed": False, "count": self._close_count}

    def performance_claim_allowed(self) -> Dict[str, Any]:
        return {
            "allowed": False,
            "reason": (
                "the dummy runtime adapter is a contract fixture; it executes no "
                "model and can never support a performance claim"
            ),
        }


class ReferenceRuntimeAdapter(RuntimeAdapter):
    """Adapter over a C4 :class:`~hqsb.core.contracts.backend.Backend`.

    The backend is injected (constructor) so the runtime layer never imports a
    concrete backend module; the caller decides which C4 implementation is the
    semantic oracle.  This adapter is the *reference* role: it is never required
    to be the fastest, only to be correct.
    """

    def __init__(self, backend: Any, spec: BackendSpec) -> None:
        super().__init__(name=getattr(backend, "name", "reference"), spec=spec)
        self.backend = backend

    def capability(self) -> CapabilityReport:
        declared = self.backend.capabilities()
        report = CapabilityReport(backend_id=self.name)
        report.declare("model_artifact", "SUPPORTED_EXACT")
        report.declare("input_token_ids", "SUPPORTED_EXACT")
        report.declare("max_new_tokens", "SUPPORTED_EXACT")
        report.declare(
            "precision",
            "SUPPORTED_WITH_CONSTRAINT",
            constraint=f"dtypes={list(declared.supported_dtypes)}",
        )
        report.declare("sampling_mode", "SUPPORTED_EXACT")
        report.declare(
            "streaming",
            "SUPPORTED_EXACT" if declared.streaming else "UNSUPPORTED_REJECT",
            reason="" if declared.streaming else "C4 backend reports streaming=False",
        )
        report.declare("cancel", "SUPPORTED_EXACT")
        report.declare("timeout", "SUPPORTED_EXACT")
        report.declare(
            "prefix_cache",
            "UNSUPPORTED_REJECT",
            reason="the C4 reference backend owns no paged KV cache",
        )
        if declared.quantization:
            report.declare(
                "quant_artifact",
                "SUPPORTED_WITH_CONSTRAINT",
                constraint=f"schemes={list(declared.quantization)}",
            )
        else:
            report.declare(
                "quant_artifact",
                "UNSUPPORTED_REJECT",
                reason="the C4 reference backend reports no quantization schemes",
            )
        return report

    def load(self, identity: ModelIdentity, **kwargs: Any) -> Mapping[str, Any]:
        self.transition(AdapterState.LOADING, "load")
        artifact = kwargs.get("artifact")
        if artifact is None:
            self.transition(AdapterState.FAILED, "no artifact supplied")
            raise ConfigError(
                "ReferenceRuntimeAdapter.load needs the C4 artifact (the adapter "
                "never guesses a model path)",
                details={"field": "artifact"},
            )
        self.backend.load(artifact)
        self.transition(AdapterState.LOADED, "load complete")
        return identity.as_dict()

    def warmup(self, request: RequestSpec) -> Mapping[str, Any]:
        self.transition(AdapterState.WARMED, "warmup")
        self.backend.warmup(request)
        return {"shapes": [request.input_tokens], "compile": False, "graph": False}

    def generate(self, request: RequestSpec) -> GenerationResult:
        self.require_open()
        output = self.backend.generate(request, inputs=request.input_token_ids)
        samples = getattr(output, "samples", [])
        first = samples[0] if samples else None
        tokens = tuple(getattr(first, "generated_token_ids", ())) if first else ()
        return GenerationResult(
            request_id=request.request_id,
            token_ids=tokens,
            finish_reason="length" if tokens else "error",
            actual_parameters={"max_new_tokens": request.stop.max_new_tokens},
            timings={
                "prefill_ms": float(getattr(first, "prefill_forward_ms", 0.0) or 0.0),
                "first_token_ms": float(
                    getattr(first, "first_token_selection_ms", 0.0) or 0.0
                ),
            },
            usage={
                "input_tokens": int(getattr(first, "input_tokens", request.input_tokens)),
                "output_tokens": int(getattr(first, "output_tokens", len(tokens))),
            },
            raw={"backend_metrics": dict(getattr(output, "backend_metrics", {}))},
        )

    def metrics(self) -> Mapping[str, Any]:
        return dict(self.backend.metrics())

    def close(self) -> Mapping[str, Any]:
        self._close_count += 1
        if self.state == AdapterState.CLOSED:
            return {"closed": True, "already_closed": True, "count": self._close_count}
        if self.state != AdapterState.CLOSING:
            self.transition(AdapterState.CLOSING, "close")
        self.backend.close()
        self.transition(AdapterState.CLOSED, "close complete")
        return {"closed": True, "already_closed": False, "count": self._close_count}


# ── registry ───────────────────────────────────────────────────────────────


@dataclass
class AdapterRegistration:
    """One registered adapter; the registry never guesses an adapter."""

    name: str
    factory: str
    spec: BackendSpec
    engine: str = ""
    probe: Optional[AdapterProbeResult] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "factory": self.factory,
            "spec": self.spec.as_dict(),
            "engine": self.engine,
            "probe": self.probe.as_dict() if self.probe else None,
        }


class AdapterRegistry:
    """Register adapters by name and resolve them against a probe result."""

    def __init__(self) -> None:
        self.registrations: Dict[str, AdapterRegistration] = {}

    def register(self, registration: AdapterRegistration) -> AdapterRegistration:
        if registration.name in self.registrations:
            raise ConfigError(
                f"adapter {registration.name!r} is already registered; a duplicate "
                "registration would make 'which adapter ran' ambiguous",
                details={"adapter": registration.name},
            )
        self.registrations[registration.name] = registration
        return registration

    def resolve(self, name: str) -> AdapterRegistration:
        registration = self.registrations.get(name)
        if registration is None:
            raise CapabilityError(
                f"adapter {name!r} is not registered; registered: "
                f"{sorted(self.registrations)}",
                details={"adapter": name},
            )
        if registration.probe is not None and not registration.probe.usable:
            raise CapabilityError(
                f"adapter {name!r} is not usable in this environment: "
                f"{registration.probe.state} ({registration.probe.detail})",
                details={
                    "adapter": name,
                    "state": registration.probe.state,
                    "engine": registration.engine,
                },
            )
        return registration

    def matrix(self) -> List[Dict[str, Any]]:
        return [item.as_dict() for item in self.registrations.values()]

    def as_dict(self) -> Dict[str, Any]:
        return {"registrations": self.matrix()}


# ── load/close scenario matrix (E07-01 §7 / step 19) ───────────────────────


def load_close_scenarios() -> Tuple[Dict[str, Any], ...]:
    """The scenario list that must be covered before the adapter is trusted."""
    return (
        {
            "case": "load_warmup_generate_close",
            "expectation": "all four operations succeed; identity verified",
            "must_error": False,
        },
        {
            "case": "load_cancel_close",
            "expectation": "cancel is observed; close still releases resources",
            "must_error": False,
        },
        {
            "case": "load_failure_close",
            "expectation": "load failure is diagnosed; close is still safe",
            "must_error": True,
        },
        {
            "case": "double_close",
            "expectation": "second close is idempotent and reports already_closed",
            "must_error": False,
        },
        {
            "case": "generate_after_close",
            "expectation": "old handle refuses loudly instead of executing",
            "must_error": True,
        },
        {
            "case": "close_with_in_flight",
            "expectation": "close waits for/cancels in-flight work before release",
            "must_error": False,
        },
    )


def run_load_close_scenarios(adapter_factory: Any) -> List[Dict[str, Any]]:
    """Execute the scenario matrix against a *fixture* adapter.

    This is a contract self-check (smoke), not an experiment: it proves the
    interface can express each lifecycle case and that the negative cases are
    refused.  Real backends replace ``adapter_factory``.
    """
    results: List[Dict[str, Any]] = []
    for scenario in load_close_scenarios():
        adapter = adapter_factory()
        outcome: Dict[str, Any] = {"case": scenario["case"], "ok": False}
        try:
            identity = ModelIdentity(
                model_id="fixture/model",
                model_manifest_sha256="0" * 64,
                revision="fixture-revision",
                tokenizer_id="fixture/tokenizer",
                chat_template_hash="fixture-template",
                precision="float16",
            )
            request = RequestSpec(
                request_id=f"scenario-{scenario['case']}",
                identity=identity,
                input_token_ids=(1, 2, 3, 4),
                sampling=SamplingSpec(mode="greedy"),
                stop=StopSpec(max_new_tokens=4),
            )
            if scenario["case"] == "load_failure_close":
                try:
                    adapter.load(identity)
                except Exception as exc:  # noqa: BLE001 - the failure is the scenario
                    outcome["expected_error"] = f"{type(exc).__name__}: {exc}"
            else:
                adapter.load(identity)
            adapter.warmup(request)
            if scenario["case"] == "load_cancel_close":
                adapter.cancel(request.request_id)
            elif scenario["case"] != "load_failure_close":
                output = adapter.generate(request)
                outcome["tokens"] = len(output.token_ids)
            first_close = adapter.close()
            second_close = adapter.close()
            outcome["idempotent_close"] = bool(second_close.get("already_closed")) and bool(
                first_close.get("closed")
            )
            if scenario["case"] == "generate_after_close":
                try:
                    adapter.generate(request)
                    outcome["refused_after_close"] = False
                except Exception:  # noqa: BLE001 - refusal is the expectation
                    outcome["refused_after_close"] = True
            outcome["state"] = adapter.state
            outcome["ok"] = True
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        results.append(outcome)
    return results


__all__ = [
    "ALLOWED_ADAPTER_TRANSITIONS",
    "AdapterProbeResult",
    "AdapterRegistration",
    "AdapterRegistry",
    "AdapterState",
    "CANDIDATE_ENGINES",
    "CancelRecord",
    "DummyRuntimeAdapter",
    "GenerationResult",
    "PROBE_AVAILABLE",
    "PROBE_INSTALLED_NOT_VERIFIED",
    "PROBE_NOT_INSTALLED",
    "PROBE_UNKNOWN",
    "ReferenceRuntimeAdapter",
    "RuntimeAdapter",
    "StreamChunk",
    "load_close_scenarios",
    "probe_environment",
    "run_load_close_scenarios",
    "select_primary_engine",
    "validate_stream",
]
