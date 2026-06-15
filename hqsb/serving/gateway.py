"""Gateway plane: request state machine, SSE delivery, cancel and drain.

The gateway sits between the protocol plane and the policy plane.  It owns:

* the service request state machine of details README §7 (every transition
  records reason, monotonic time, identity, deadline, route, queue, HTTP/SSE
  status and the token counters);
* streaming: SSE frames are serialised here and pushed through a
  :class:`~hqsb.serving.transport.TransportWriter`, with a terminal marker that
  is written exactly once and never after a committed error;
* cancel / disconnect / deadline propagation to the Backend, and the delivery
  ledger that keeps ``generated ≥ committed ≥ emitted ≥ flushed ≥ received``;
* the service lifecycle: ``SERVING → DRAINING → FORCE_CANCELLING → CLOSED``
  (E08-08 §8) — in-process drain only; Kubernetes rolling updates are S13.

Collaborators (admission, routing, telemetry) are injected as protocols so the
service layer never imports a private SDK, and a *missing* collaborator is
recorded as an explicit reason rather than silently defaulting to "allow".
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from hqsb.core.errors import ConfigError, HqsbError, SchemaError

from hqsb.serving import protocol as protocol_mod
from hqsb.serving import sse, timing
from hqsb.serving.pipeline import DeliveryLedger
from hqsb.serving.protocol import (
    CanonicalBackendRequest,
    ErrorCatalog,
    ProtocolProfile,
    RequestRejected,
)
from hqsb.serving.transport import TransportRequest, TransportWriter

#: Service lifecycle states (E08-08 §8).
SERVICE_STATES: Tuple[str, ...] = ("SERVING", "DRAINING", "FORCE_CANCELLING", "CLOSED")

#: Request states of details README §7.
REQUEST_STATES: Tuple[str, ...] = (
    "CLIENT_SCHEDULED",
    "CONNECTING",
    "RECEIVED",
    "VALIDATED",
    "ADMISSION_PENDING",
    "QUEUED",
    "ROUTED",
    "BACKEND_SUBMITTED",
    "FIRST_TOKEN_READY",
    "STREAMING",
    "TERMINAL_FRAME_SENT",
    "CLIENT_COMPLETED",
    "CLEANED",
    "REJECTED_INVALID",
    "REJECTED_OVERLOAD",
    "DISCONNECTED",
    "CANCEL_REQUESTED",
    "CANCEL_PROPAGATED",
    "CANCELLED",
    "TIMED_OUT",
    "BACKEND_FAILED",
    "RETRYING_BEFORE_COMMIT",
    "FALLBACK_SELECTED",
    "DRAINING",
)

#: Legal transitions.  Anything else raises: a request that jumps states would
#: hide where a token was lost.
ALLOWED_REQUEST_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    "CLIENT_SCHEDULED": ("CONNECTING",),
    "CONNECTING": ("RECEIVED",),
    "RECEIVED": ("VALIDATED", "REJECTED_INVALID", "REJECTED_OVERLOAD"),
    "VALIDATED": ("ADMISSION_PENDING", "REJECTED_INVALID"),
    "ADMISSION_PENDING": ("QUEUED", "REJECTED_OVERLOAD", "BACKEND_FAILED"),
    "QUEUED": (
        "ROUTED",
        "CANCEL_REQUESTED",
        "REJECTED_OVERLOAD",
        "DISCONNECTED",
        "BACKEND_FAILED",
    ),
    "ROUTED": (
        "BACKEND_SUBMITTED",
        "REJECTED_OVERLOAD",
        "DISCONNECTED",
        "CANCEL_REQUESTED",
        "BACKEND_FAILED",
    ),
    "BACKEND_SUBMITTED": (
        "FIRST_TOKEN_READY",
        "BACKEND_FAILED",
        "CANCEL_REQUESTED",
        "DISCONNECTED",
        "TIMED_OUT",
        "RETRYING_BEFORE_COMMIT",
    ),
    "RETRYING_BEFORE_COMMIT": ("BACKEND_SUBMITTED", "BACKEND_FAILED", "FALLBACK_SELECTED"),
    "FALLBACK_SELECTED": ("BACKEND_SUBMITTED",),
    # a non-stream answer has no intermediate frames: its whole body is written
    # once, so TERMINAL_FRAME_SENT follows FIRST_TOKEN_READY directly.
    "FIRST_TOKEN_READY": (
        "STREAMING",
        "TERMINAL_FRAME_SENT",
        "CANCEL_REQUESTED",
        "DISCONNECTED",
        "TIMED_OUT",
    ),
    "STREAMING": (
        "TERMINAL_FRAME_SENT",
        "CANCEL_REQUESTED",
        "DISCONNECTED",
        "TIMED_OUT",
        "BACKEND_FAILED",
    ),
    "TERMINAL_FRAME_SENT": ("CLIENT_COMPLETED", "DISCONNECTED"),
    "CLIENT_COMPLETED": ("CLEANED",),
    "CANCEL_REQUESTED": ("CANCEL_PROPAGATED",),
    "CANCEL_PROPAGATED": ("CANCELLED",),
    "CANCELLED": ("CLEANED",),
    # a disconnect or an expired deadline cancels the Backend directly: the
    # intermediate CANCEL_REQUESTED models a client-side cancel API call.
    "DISCONNECTED": ("CANCEL_REQUESTED", "CANCEL_PROPAGATED", "CLEANED"),
    "TIMED_OUT": ("CANCEL_REQUESTED", "CANCEL_PROPAGATED", "CLEANED"),
    "BACKEND_FAILED": ("CLEANED", "RETRYING_BEFORE_COMMIT"),
    "REJECTED_INVALID": ("CLEANED",),
    "REJECTED_OVERLOAD": ("CLEANED",),
    "CLEANED": (),
}

#: Terminal states; a request must end in exactly one of them.
TERMINAL_REQUEST_STATES: Tuple[str, ...] = (
    "CLEANED",
    "CANCELLED",
    "REJECTED_INVALID",
    "REJECTED_OVERLOAD",
)

#: Branch states that are entered directly by the gateway.
BRANCH_STATES: Tuple[str, ...] = tuple(
    state for state in REQUEST_STATES if state in ("REJECTED_INVALID", "REJECTED_OVERLOAD")
)


@dataclass
class TransitionRecord:
    """One state transition with everything the protocol asks to be saved."""

    request_id: str
    trace_id: str
    tenant: str
    request_class: str
    old_state: str
    new_state: str
    reason: str
    monotonic_ns: int
    model_alias: str = ""
    model_identity: Mapping[str, Any] = field(default_factory=dict)
    deadline_remaining_ms: Optional[float] = None
    selected_backend: str = ""
    route_epoch: str = ""
    route_reason: str = ""
    queue_class: str = ""
    queue_depth: int = 0
    queue_position: int = 0
    http_status: int = 0
    sse_sequence: int = -1
    finish_reason: str = ""
    generated_tokens: int = 0
    committed_tokens: int = 0
    emitted_tokens: int = 0
    flushed_tokens: int = 0
    received_tokens: int = 0
    retry_attempt: int = 0
    idempotency_state: str = ""
    cleanup_result: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "tenant": self.tenant,
            "request_class": self.request_class,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "reason": self.reason,
            "monotonic_ns": self.monotonic_ns,
            "model_alias": self.model_alias,
            "model_identity": dict(self.model_identity),
            "deadline_remaining_ms": self.deadline_remaining_ms,
            "selected_backend": self.selected_backend,
            "route_epoch": self.route_epoch,
            "route_reason": self.route_reason,
            "queue_class": self.queue_class,
            "queue_depth": self.queue_depth,
            "queue_position": self.queue_position,
            "http_status": self.http_status,
            "sse_sequence": self.sse_sequence,
            "finish_reason": self.finish_reason,
            "generated_tokens": self.generated_tokens,
            "committed_tokens": self.committed_tokens,
            "emitted_tokens": self.emitted_tokens,
            "flushed_tokens": self.flushed_tokens,
            "received_tokens": self.received_tokens,
            "retry_attempt": self.retry_attempt,
            "idempotency_state": self.idempotency_state,
            "cleanup_result": self.cleanup_result,
        }


class RequestStateMachine:
    """Transition guard for one request (illegal jumps are refused)."""

    def __init__(self, *, request_id: str, trace_id: str, tenant: str = "", request_class: str = "") -> None:
        self.request_id = request_id
        self.trace_id = trace_id
        self.tenant = tenant
        self.request_class = request_class
        self.state = "CLIENT_SCHEDULED"
        self.records: List[TransitionRecord] = []

    def transition(self, new_state: str, *, reason: str, **fields: Any) -> TransitionRecord:
        if new_state not in REQUEST_STATES:
            raise SchemaError(f"unknown request state {new_state!r}")
        allowed = ALLOWED_REQUEST_TRANSITIONS.get(self.state, ())
        if new_state not in allowed:
            raise SchemaError(
                f"illegal request transition {self.state} -> {new_state} for "
                f"{self.request_id}; allowed: {list(allowed)}",
                details={"from": self.state, "to": new_state, "reason": reason},
            )
        record = TransitionRecord(
            request_id=self.request_id,
            trace_id=self.trace_id,
            tenant=self.tenant,
            request_class=self.request_class,
            old_state=self.state,
            new_state=new_state,
            reason=reason,
            monotonic_ns=int(fields.pop("monotonic_ns", 0)),
            **fields,
        )
        self.records.append(record)
        self.state = new_state
        return record

    @property
    def terminated(self) -> bool:
        return self.state in ("CLEANED",)

    def terminal_branch(self) -> str:
        for record in self.records:
            if record.new_state in ("CANCELLED", "REJECTED_INVALID", "REJECTED_OVERLOAD"):
                return record.new_state
            if record.new_state in ("TIMED_OUT", "BACKEND_FAILED"):
                return record.new_state
        return ""


# ── backend contract (S07 surface only) ────────────────────────────────────


@dataclass(frozen=True)
class BackendChunk:
    """One token as the Backend hands it over (no HTTP knowledge)."""

    token_id: int
    text: str
    committed: bool = True
    finish_reason: str = ""
    usage: Optional[Mapping[str, Any]] = None
    model_epoch: str = ""


class BackendSession(Protocol):
    """One in-flight request on a Backend (the S07 stable surface)."""

    backend_request_id: str
    model_epoch: str

    def model_identity(self) -> Mapping[str, Any]: ...

    def next_chunk(self) -> Optional[BackendChunk]: ...

    def cancel(self, *, reason: str) -> None: ...

    def cleanup(self) -> Mapping[str, Any]: ...


class ServingBackend(Protocol):
    """A registered Backend instance the gateway can submit to."""

    instance_id: str

    def accepts(self, request: CanonicalBackendRequest) -> Mapping[str, Any]: ...

    def open(self, request: CanonicalBackendRequest, *, deadline_ns: int) -> BackendSession: ...

    def health(self) -> Mapping[str, Any]: ...


class AdmissionDecider(Protocol):
    """Policy-plane admission hook."""

    def decide(self, *, context: Mapping[str, Any], snapshot: Mapping[str, Any]) -> Mapping[str, Any]: ...


class RouteSelector(Protocol):
    """Policy-plane routing hook."""

    def select(
        self, request: CanonicalBackendRequest, candidates: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class TelemetrySink(Protocol):
    """Evidence-plane hook (counters/spans); observation must not alter policy."""

    def record_outcome(self, outcome: "RequestOutcome") -> None: ...

    def record_transition(self, record: TransitionRecord) -> None: ...


@dataclass
class GatewayConfig:
    """Frozen gateway behaviour knobs."""

    service_id: str = "hqsb-serve"
    default_deadline_ms: float = 30_000.0
    max_deadline_ms: float = 600_000.0
    deadline_header: str = "x-hqsb-deadline-ms"
    tenant_header: str = "x-hqsb-tenant"
    class_header: str = "x-hqsb-class"
    emit_usage_frame: bool = False
    drain_deadline_ms: float = 30_000.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "default_deadline_ms": self.default_deadline_ms,
            "max_deadline_ms": self.max_deadline_ms,
            "deadline_header": self.deadline_header,
            "tenant_header": self.tenant_header,
            "class_header": self.class_header,
            "emit_usage_frame": self.emit_usage_frame,
            "drain_deadline_ms": self.drain_deadline_ms,
        }


@dataclass
class RequestOutcome:
    """The evidence a run keeps for one request."""

    request_id: str
    trace_id: str
    status: int
    code: str
    stream: bool
    wrote_head: bool
    frames: int
    body_bytes: int
    ledger: DeliveryLedger
    timestamps: timing.TimestampLedger
    transitions: List[TransitionRecord]
    terminal_state: str
    backend_id: str = ""
    model_epoch: str = ""
    router_reason: str = ""
    admission_reason: str = ""
    cancel_propagated: bool = False
    deadline_exceeded: bool = False
    detached_reason: str = ""
    raw_sse: bytes = b""

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.detached_reason

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "status": self.status,
            "code": self.code,
            "stream": self.stream,
            "wrote_head": self.wrote_head,
            "frames": self.frames,
            "body_bytes": self.body_bytes,
            "ledger": self.ledger.as_dict(),
            "ledger_audit": self.ledger.audit(),
            "timestamps": self.timestamps.as_dict(),
            "transitions": [record.as_dict() for record in self.transitions],
            "terminal_state": self.terminal_state,
            "backend_id": self.backend_id,
            "model_epoch": self.model_epoch,
            "router_reason": self.router_reason,
            "admission_reason": self.admission_reason,
            "cancel_propagated": self.cancel_propagated,
            "deadline_exceeded": self.deadline_exceeded,
            "detached_reason": self.detached_reason,
        }


class ServingGateway:
    """The HTTP-facing service core (transport-agnostic)."""

    def __init__(
        self,
        *,
        profile: ProtocolProfile,
        catalog: ErrorCatalog,
        model_registry: Mapping[str, Mapping[str, Any]],
        tokenizer: Callable[[str], Sequence[int]],
        chat_template: Callable[[Sequence[Mapping[str, Any]]], str],
        backends: Mapping[str, Any],
        admission: Optional[AdmissionDecider] = None,
        router: Optional[RouteSelector] = None,
        telemetry: Optional[TelemetrySink] = None,
        config: Optional[GatewayConfig] = None,
    ) -> None:
        if not backends:
            raise ConfigError(
                "the gateway needs at least one registered Backend; a service with no "
                "backend would answer 503 for every request"
            )
        self.profile = profile
        self.catalog = catalog
        self.model_registry = model_registry
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.backends = dict(backends)
        self.admission = admission
        self.router = router
        self.telemetry = telemetry
        self.config = config or GatewayConfig()
        self._state = "SERVING"
        self._lock = threading.Lock()
        self._inflight = 0
        self._counters: Dict[str, int] = {}
        self._drain_started_ns: Optional[int] = None
        self._drain_deadline_ns: Optional[int] = None
        self._cancelled: Dict[str, bool] = {}

    # ── lifecycle ──────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    def health(self) -> Dict[str, Any]:
        return {
            "service_id": self.config.service_id,
            "state": self._state,
            "inflight": self._inflight,
            "backends": {
                name: dict(backend.health()) for name, backend in self.backends.items()
            },
        }

    def readiness(self) -> Dict[str, Any]:
        ready = self._state == "SERVING"
        return {
            "ready": ready,
            "reason": "" if ready else f"service state is {self._state}",
        }

    def begin_drain(self, *, monotonic_ns: int, reason: str = "SIGTERM") -> Dict[str, Any]:
        with self._lock:
            if self._state == "SERVING":
                self._state = "DRAINING"
                self._drain_started_ns = monotonic_ns
                self._drain_deadline_ns = monotonic_ns + int(self.config.drain_deadline_ms * 1e6)
            elif self._state != "DRAINING":
                raise SchemaError(
                    f"drain requested while service state is {self._state}; shutdown "
                    "transitions are one-way (E08-08 §8)"
                )
        return {
            "state": self._state,
            "reason": reason,
            "readiness": self.readiness(),
            "drain_deadline_ns": self._drain_deadline_ns,
            "inflight_at_start": self._inflight,
        }

    def force_cancel(self, *, monotonic_ns: int) -> Dict[str, Any]:
        with self._lock:
            if self._state not in ("DRAINING", "FORCE_CANCELLING"):
                raise SchemaError("force_cancel requires a draining service")
            self._state = "FORCE_CANCELLING"
            for request_id in list(self._cancelled):
                self._cancelled[request_id] = True
        return {"state": self._state, "monotonic_ns": monotonic_ns}

    def close(self) -> Dict[str, Any]:
        with self._lock:
            if self._state == "CLOSED":
                return {"state": self._state, "idempotent": True}
            if self._state not in ("DRAINING", "FORCE_CANCELLING"):
                raise SchemaError(
                    "close without drain would drop in-flight work; call begin_drain first"
                )
            self._state = "CLOSED"
        return {"state": self._state, "idempotent": False}

    def request_cancel(self, request_id: str, *, reason: str) -> None:
        self._cancelled[request_id] = True

    # ── metrics ────────────────────────────────────────────────────────
    def _count(self, name: str, delta: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + delta

    def counters(self) -> Dict[str, int]:
        return dict(sorted(self._counters.items()))

    def metrics_exposition(self) -> str:
        lines = [f"hqsb_service_state {SERVICE_STATES.index(self._state)}"]
        for name, value in sorted(self._counters.items()):
            lines.append(f"hqsb_{name} {value}")
        return "\n".join(lines) + "\n"

    # ── request handling ───────────────────────────────────────────────
    def handle(self, request: TransportRequest, writer: TransportWriter) -> RequestOutcome:
        request_id = request.header("x-request-id") or f"req-{request.received_ns}"
        trace_id = request.header("traceparent") or request_id
        tenant = request.header(self.config.tenant_header, "default")
        request_class = request.header(self.config.class_header, "interactive")
        machine = RequestStateMachine(
            request_id=request_id, trace_id=trace_id, tenant=tenant, request_class=request_class
        )
        ledger = DeliveryLedger(request_id=request_id)
        stamps = timing.TimestampLedger(request_id=request_id)
        stamps.record("t_gateway_recv", request.received_ns)
        self._count("requests_received")

        # transport-level routing (health/readiness/metrics are never queued)
        if request.path in ("/healthz", "/readyz", "/metrics"):
            return self._handle_control(request, writer, machine, ledger, stamps)
        if request.path not in self.profile.endpoints:
            return self._reject(
                writer, machine, ledger, stamps, code="not_found", reason=request.path
            )
        if request.method != "POST":
            return self._reject(
                writer, machine, ledger, stamps, code="method_not_allowed", reason=request.method
            )
        if self._state != "SERVING":
            return self._reject(
                writer,
                machine,
                ledger,
                stamps,
                code="service_not_serving",
                reason=f"service state is {self._state}",
            )

        deadline_ms, deadline_problem = self._deadline_ms(request)
        machine.transition("CONNECTING", reason="connection accepted", monotonic_ns=request.received_ns)
        machine.transition("RECEIVED", reason="body received", monotonic_ns=request.received_ns)

        # protocol pipeline: validation, identity, template, budget
        try:
            canonical = protocol_mod.request_pipeline(
                request.body,
                endpoint=request.path,
                request_id=request_id,
                profile=self.profile,
                catalog=self.catalog,
                model_registry=self.model_registry,
                tokenizer=self.tokenizer,
                chat_template=self.chat_template,
            )
        except RequestRejected as rejected:
            return self._reject(
                writer,
                machine,
                ledger,
                stamps,
                code=rejected.issue.code,
                reason=f"{rejected.issue.path}: {rejected.issue.message}",
                stage="VALIDATED",
            )
        stamps.record("t_validate_done", self._now(writer))
        if deadline_problem:
            return self._reject(
                writer, machine, ledger, stamps, code="deadline_infeasible", reason=deadline_problem
            )
        machine.transition(
            "VALIDATED",
            reason="schema/semantic/identity/budget checks passed",
            monotonic_ns=stamps.value("t_validate_done") or request.received_ns,
            model_alias=canonical.model_alias,
            model_identity=canonical.model_identity,
        )

        # admission
        machine.transition(
            "ADMISSION_PENDING",
            reason="asking the admission policy",
            monotonic_ns=stamps.value("t_validate_done") or request.received_ns,
        )
        decision = self._admission_decision(canonical, deadline_ms)
        if not decision.get("admitted", True):
            code = str(decision.get("code", "service_overloaded"))
            if code not in self.catalog.entries:
                raise ConfigError(
                    f"admission returned unknown error code {code!r}; reject reasons must "
                    "come from the frozen catalog"
                )
            return self._reject(
                writer,
                machine,
                ledger,
                stamps,
                code=code,
                reason=str(decision.get("reason", "")),
                stage="ADMISSION_PENDING",
                retry_after_ms=int(decision.get("retry_after_ms", 0) or 0),
                queue=decision,
            )
        stamps.record("t_enqueue", self._now(writer))
        machine.transition(
            "QUEUED",
            reason="admitted",
            monotonic_ns=stamps.value("t_enqueue") or request.received_ns,
            queue_class=str(decision.get("queue_class", "")),
            queue_depth=int(decision.get("queue_depth", 0) or 0),
            queue_position=int(decision.get("queue_position", 0) or 0),
        )

        # routing
        selection = self._route(canonical)
        if selection is None:
            return self._reject(
                writer,
                machine,
                ledger,
                stamps,
                code="no_feasible_backend",
                reason="hard capability/identity/health filters removed every candidate",
                stage="QUEUED",
            )
        backend = selection["backend"]
        stamps.record("t_dequeue", self._now(writer))
        machine.transition(
            "ROUTED",
            reason=str(selection.get("reason", "")),
            monotonic_ns=stamps.value("t_dequeue") or request.received_ns,
            selected_backend=str(selection.get("instance_id", "")),
            route_epoch=str(selection.get("route_epoch", "")),
            route_reason=str(selection.get("reason", "")),
        )
        self._count("requests_admitted")
        return self._serve(
            request=request,
            writer=writer,
            machine=machine,
            ledger=ledger,
            stamps=stamps,
            canonical=canonical,
            backend=backend,
            selection=selection,
            deadline_ms=deadline_ms,
        )

    # ── internals ──────────────────────────────────────────────────────
    def _now(self, writer: TransportWriter) -> int:
        try:
            return int(writer.now_ns())
        except Exception:  # noqa: BLE001 - a writer without a clock is still usable
            import time

            return time.monotonic_ns()

    def _deadline_ms(self, request: TransportRequest) -> Tuple[float, str]:
        raw = request.header(self.config.deadline_header, "")
        if not raw:
            return self.config.default_deadline_ms, ""
        try:
            value = float(raw)
        except ValueError:
            return self.config.default_deadline_ms, f"invalid deadline header {raw!r}"
        if value <= 0:
            return self.config.default_deadline_ms, "deadline must be positive"
        if value > self.config.max_deadline_ms:
            return self.config.max_deadline_ms, "deadline above the service maximum"
        return value, ""

    def _admission_decision(
        self, canonical: CanonicalBackendRequest, deadline_ms: float
    ) -> Mapping[str, Any]:
        if self.admission is None:
            return {"admitted": True, "reason": "no admission policy configured"}
        snapshot = {
            "inflight": self._inflight,
            "deadline_ms": deadline_ms,
            "reserved_output_tokens": canonical.reserved_output_tokens,
            "prompt_tokens": canonical.prompt_tokens,
        }
        return self.admission.decide(
            context={"request_id": canonical.request_id, "tenant": "", "class": ""},
            snapshot=snapshot,
        )

    def _route(self, canonical: CanonicalBackendRequest) -> Optional[Mapping[str, Any]]:
        if self.router is None:
            instance_id = next(iter(self.backends))
            return {
                "instance_id": instance_id,
                "backend": self.backends[instance_id],
                "reason": "single registered backend (no router configured)",
                "route_epoch": "",
                "fallback_level": 0,
            }
        result = dict(self.router.select(canonical, self.backends))
        instance_id = str(result.get("instance_id", ""))
        if not instance_id or instance_id not in self.backends:
            return None
        result["backend"] = self.backends[instance_id]
        return result

    def _handle_control(
        self,
        request: TransportRequest,
        writer: TransportWriter,
        machine: RequestStateMachine,
        ledger: DeliveryLedger,
        stamps: timing.TimestampLedger,
    ) -> RequestOutcome:
        payload = (
            self.health() if request.path == "/healthz" else self.readiness()
        )
        # readiness must not lie during drain (E08-05 §11)
        body = json.dumps(payload).encode()
        writer.write_head(
            200,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        writer.write_body(body)
        writer.close()
        machine.transition("CONNECTING", reason="control request", monotonic_ns=request.received_ns)
        machine.transition("RECEIVED", reason="control request", monotonic_ns=request.received_ns)
        machine.transition("VALIDATED", reason="control endpoint", monotonic_ns=request.received_ns)
        machine.transition("ADMISSION_PENDING", reason="control endpoint", monotonic_ns=request.received_ns)
        machine.transition("QUEUED", reason="control endpoint bypasses the business queue", monotonic_ns=request.received_ns)
        machine.transition("ROUTED", reason="control endpoint", monotonic_ns=request.received_ns)
        machine.transition("BACKEND_SUBMITTED", reason="no backend involved", monotonic_ns=request.received_ns)
        machine.transition("FIRST_TOKEN_READY", reason="body ready", monotonic_ns=request.received_ns)
        machine.transition("STREAMING", reason="body written", monotonic_ns=request.received_ns)
        machine.transition("TERMINAL_FRAME_SENT", reason="closed", monotonic_ns=request.received_ns)
        machine.transition("CLIENT_COMPLETED", reason="control body delivered", monotonic_ns=request.received_ns)
        machine.transition("CLEANED", reason="control request finished", monotonic_ns=request.received_ns)
        return RequestOutcome(
            request_id=machine.request_id,
            trace_id=machine.trace_id,
            status=200,
            code="",
            stream=False,
            wrote_head=True,
            frames=0,
            body_bytes=len(body),
            ledger=ledger,
            timestamps=stamps,
            transitions=list(machine.records),
            terminal_state=machine.state,
        )

    def _reject(
        self,
        writer: TransportWriter,
        machine: RequestStateMachine,
        ledger: DeliveryLedger,
        stamps: timing.TimestampLedger,
        *,
        code: str,
        reason: str,
        stage: str = "RECEIVED",
        retry_after_ms: int = 0,
        queue: Optional[Mapping[str, Any]] = None,
    ) -> RequestOutcome:
        entry = self.catalog.entry(code)
        body = json.dumps(self.catalog.wire_body(code, message=entry.message)).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        if retry_after_ms > 0 or entry.category == "overload":
            suggested = retry_after_ms or 1000
            headers["Retry-After"] = str(max(1, int(suggested / 1000)))
        writer.write_head(entry.http_status, headers)
        writer.write_body(body)
        writer.close()

        target = (
            "REJECTED_OVERLOAD"
            if entry.category == "overload" or code in ("service_not_serving", "no_feasible_backend")
            else "REJECTED_INVALID"
        )
        if machine.state == "CLIENT_SCHEDULED":
            # a transport-level rejection still has to be recorded end to end
            machine.transition("CONNECTING", reason="connection accepted", monotonic_ns=0)
            machine.transition("RECEIVED", reason="rejection raised", monotonic_ns=0)
        if machine.state == "VALIDATED" and target == "REJECTED_OVERLOAD":
            machine.transition(
                "ADMISSION_PENDING",
                reason="overload rejection raised after validation",
                monotonic_ns=0,
            )
        if machine.state in ("RECEIVED", "VALIDATED", "ADMISSION_PENDING", "QUEUED", "ROUTED"):
            machine.transition(target, reason=reason, monotonic_ns=0)
            machine.transition("CLEANED", reason="rejection recorded", monotonic_ns=0)
        del stage
        self._count(f"rejected_{code}")
        outcome = RequestOutcome(
            request_id=machine.request_id,
            trace_id=machine.trace_id,
            status=entry.http_status,
            code=code,
            stream=False,
            wrote_head=True,
            frames=1,
            body_bytes=len(body),
            ledger=ledger,
            timestamps=stamps,
            transitions=list(machine.records),
            terminal_state=machine.state,
            admission_reason=reason if entry.stage == "admission" else "",
            router_reason=reason if entry.stage == "routing" else "",
        )
        outcome.ledger.terminal_clean = False
        if self.telemetry is not None:
            self.telemetry.record_outcome(outcome)
            for record in machine.records:
                self.telemetry.record_transition(record)
        return outcome

    def _serve(
        self,
        *,
        request: TransportRequest,
        writer: TransportWriter,
        machine: RequestStateMachine,
        ledger: DeliveryLedger,
        stamps: timing.TimestampLedger,
        canonical: CanonicalBackendRequest,
        backend: Any,
        selection: Mapping[str, Any],
        deadline_ms: float,
    ) -> RequestOutcome:
        self._inflight += 1
        deadline_ns = request.received_ns + int(deadline_ms * 1e6)
        session: Optional[BackendSession] = None
        code = ""
        status = 200
        finish_reason = ""
        frames = 0
        raw_sse = bytearray()
        detached_reason = ""
        cancel_propagated = False
        deadline_exceeded = False
        response_body = b""
        wrote_head = False
        usage: Optional[Mapping[str, Any]] = None
        write_refused = False

        try:
            session = backend.open(canonical, deadline_ns=deadline_ns)
            verify = getattr(session, "model_identity", lambda: {})()
            if verify and canonical.model_identity and dict(verify) != dict(canonical.model_identity):
                raise SchemaError(
                    "backend answered with a different model identity than the routed "
                    "one; returning this response would be an identity failure",
                    details={"requested": dict(canonical.model_identity), "actual": dict(verify)},
                )
            stamps.record("t_backend_submit", self._now(writer))
            machine.transition(
                "BACKEND_SUBMITTED",
                reason="backend accepted the request",
                monotonic_ns=stamps.value("t_backend_submit") or 0,
                selected_backend=str(selection.get("instance_id", "")),
                route_epoch=str(selection.get("route_epoch", "")),
            )
            first_token = True
            content_frames: List[Mapping[str, Any]] = []
            while True:
                if self._cancelled.get(canonical.request_id):
                    machine.transition(
                        "CANCEL_REQUESTED", reason="client cancel observed", monotonic_ns=self._now(writer)
                    )
                    session.cancel(reason="client_cancelled")
                    cancel_propagated = True
                    machine.transition(
                        "CANCEL_PROPAGATED",
                        reason="cancel reached the backend",
                        monotonic_ns=self._now(writer),
                    )
                    machine.transition("CANCELLED", reason="request cancelled", monotonic_ns=self._now(writer))
                    ledger.cancelled = True
                    code = "client_cancelled"
                    status = self.catalog.entry(code).http_status
                    break
                if not writer.client_connected():
                    machine.transition(
                        "DISCONNECTED",
                        reason="client closed the connection",
                        monotonic_ns=self._now(writer),
                    )
                    session.cancel(reason="client_disconnected")
                    cancel_propagated = True
                    machine.transition(
                        "CANCEL_PROPAGATED",
                        reason="cancel reached the backend after disconnect",
                        monotonic_ns=self._now(writer),
                    )
                    machine.transition("CANCELLED", reason="disconnect", monotonic_ns=self._now(writer))
                    ledger.disconnected = True
                    code = "client_disconnected"
                    status = self.catalog.entry(code).http_status
                    detached_reason = "client_disconnected"
                    break
                if self._now(writer) > deadline_ns:
                    machine.transition(
                        "TIMED_OUT", reason="deadline exceeded", monotonic_ns=self._now(writer)
                    )
                    session.cancel(reason="deadline_exceeded")
                    cancel_propagated = True
                    deadline_exceeded = True
                    code = "deadline_exceeded"
                    status = self.catalog.entry(code).http_status
                    break
                chunk = session.next_chunk()
                if chunk is None:
                    break
                ledger.generated += 1
                if chunk.committed:
                    ledger.committed += 1
                if first_token:
                    stamps.record("t_runtime_first", self._now(writer))
                    machine.transition(
                        "FIRST_TOKEN_READY",
                        reason="first token committed",
                        monotonic_ns=stamps.value("t_runtime_first") or 0,
                    )
                    first_token = False
                if canonical.stream and not wrote_head:
                    writer.write_head(
                        200,
                        {"Content-Type": self.profile.sse.content_type, "Cache-Control": "no-cache"},
                    )
                    wrote_head = True
                    stamps.record("t_first_frame_write", self._now(writer))
                    machine.transition(
                        "STREAMING", reason="first frame written", monotonic_ns=self._now(writer)
                    )
                if canonical.stream:
                    payload = {
                        "id": canonical.request_id,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": canonical.model_alias,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.text, "token_ids": [chunk.token_id]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    frame = sse.encode_data_frame(payload, data_prefix=self.profile.sse.data_prefix)
                    accepted = writer.write_body(frame)
                    raw_sse.extend(frame)
                    if accepted:
                        frames += 1
                        ledger.emitted += 1
                        ledger.flushed += 1
                        if frames == 1:
                            stamps.record("t_first_byte_client", self._now(writer))
                    else:
                        # the socket refused the write: the client is gone, and the
                        # Backend must be cancelled rather than kept generating
                        write_refused = True
                        break
                else:
                    content_frames.append(
                        {
                            "index": 0,
                            "text": chunk.text,
                            "token_ids": [chunk.token_id],
                            "delta": {"content": chunk.text, "token_ids": [chunk.token_id]},
                        }
                    )
                    ledger.emitted += 1
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.usage:
                    usage = dict(chunk.usage)
            if write_refused and not code:
                code = "client_disconnected"
                status = self.catalog.entry(code).http_status
                ledger.disconnected = True
                detached_reason = "socket_write_refused"
                session.cancel(reason="client_disconnected")
                cancel_propagated = True
                machine.transition(
                    "DISCONNECTED",
                    reason="socket write refused: the client is gone",
                    monotonic_ns=self._now(writer),
                )
                machine.transition(
                    "CANCEL_PROPAGATED",
                    reason="cancel reached the backend after the refused write",
                    monotonic_ns=self._now(writer),
                )
                machine.transition(
                    "CANCELLED", reason="disconnect", monotonic_ns=self._now(writer)
                )
            if not code and not finish_reason:
                # the frozen profile requires a finish reason on every successful
                # answer; inventing "stop" would hide a broken backend
                code = "protocol_failure"
            # ── terminal ────────────────────────────────────────────────
            usage_payload = usage
            if not code and canonical.stream:
                if finish_reason:
                    payload = {
                        "id": canonical.request_id,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": canonical.model_alias,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        ],
                    }
                    frame = sse.encode_data_frame(
                        payload, data_prefix=self.profile.sse.data_prefix
                    )
                    writer.write_body(frame)
                    raw_sse.extend(frame)
                    frames += 1
                if usage_payload and (
                    self.config.emit_usage_frame
                    or bool((canonical.normalized.get("stream_options") or {}).get("include_usage"))
                ):
                    frame = sse.encode_usage_frame(
                        usage_payload,
                        model=canonical.model_alias,
                        frame_id=canonical.request_id,
                        data_prefix=self.profile.sse.data_prefix,
                    )
                    writer.write_body(frame)
                    raw_sse.extend(frame)
                    frames += 1
                terminal = sse.encode_terminal(data_prefix=self.profile.sse.data_prefix)
                writer.write_body(terminal)
                raw_sse.extend(terminal)
                frames += 1
                stamps.record("t_terminal_write", self._now(writer))
                machine.transition(
                    "TERMINAL_FRAME_SENT",
                    reason="terminal marker written",
                    monotonic_ns=stamps.value("t_terminal_write") or 0,
                    sse_sequence=frames - 1,
                    finish_reason=finish_reason,
                )
                ledger.terminal_clean = True
            elif not code:
                stamps.record("t_runtime_last", self._now(writer))
                payload = {
                    "id": canonical.request_id,
                    "object": "text_completion" if canonical.endpoint.endswith("completions") else "chat.completion",
                    "created": 0,
                    "model": canonical.model_alias,
                    "choices": [
                        {
                            "index": 0,
                            "text": "".join(item["text"] for item in content_frames),
                            "token_ids": [
                                token
                                for item in content_frames
                                for token in item["token_ids"]
                            ],
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage_payload
                    or {
                        "prompt_tokens": canonical.prompt_tokens,
                        "completion_tokens": ledger.committed,
                        "total_tokens": canonical.prompt_tokens + ledger.committed,
                    },
                }
                if canonical.endpoint.endswith("chat/completions"):
                    payload["choices"][0]["message"] = {
                        "role": "assistant",
                        "content": payload["choices"][0]["text"],
                    }
                response_body = json.dumps(payload).encode()
                response_body = json.dumps(
                    self._normalize_response(json.loads(response_body.decode()), canonical)
                ).encode()
                writer.write_head(
                    200,
                    {"Content-Type": "application/json", "Content-Length": str(len(response_body))},
                )
                wrote_head = True
                writer.write_body(response_body)
                stamps.record("t_terminal_write", self._now(writer))
                ledger.terminal_clean = True
                finish_reason = finish_reason or "stop"
                machine.transition(
                    "TERMINAL_FRAME_SENT",
                    reason="response body written",
                    monotonic_ns=stamps.value("t_terminal_write") or 0,
                    http_status=200,
                    finish_reason=finish_reason,
                )
            if code:
                # a failure after the head is committed must be reported inside the
                # stream: the HTTP status can no longer change (E08-01 §13)
                if wrote_head and canonical.stream:
                    payload = {
                        "id": canonical.request_id,
                        "object": "chat.completion.chunk",
                        "model": canonical.model_alias,
                        "error": self.catalog.wire_body(code)["error"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    }
                    frame = sse.encode_data_frame(
                        payload, data_prefix=self.profile.sse.data_prefix
                    )
                    writer.write_body(frame)
                    raw_sse.extend(frame)
                    frames += 1
                elif not wrote_head:
                    entry = self.catalog.entry(code)
                    body = json.dumps(self.catalog.wire_body(code)).encode()
                    writer.write_head(entry.http_status, {"Content-Type": "application/json"})
                    writer.write_body(body)
                    response_body = body
                    wrote_head = True
            if stamps.value("t_runtime_last") is None:
                stamps.record("t_runtime_last", self._now(writer))
            writer.close()
            stamps.record("t_client_done", self._now(writer))
            stamps.record("t_cleanup_done", self._now(writer))
            if machine.state in ("TERMINAL_FRAME_SENT", "CANCELLED", "TIMED_OUT", "DISCONNECTED"):
                if machine.state == "TERMINAL_FRAME_SENT":
                    machine.transition(
                        "CLIENT_COMPLETED",
                        reason="response delivered",
                        monotonic_ns=stamps.value("t_client_done") or 0,
                        http_status=status,
                        finish_reason=finish_reason,
                    )
                machine.transition(
                    "CLEANED",
                    reason="resources released",
                    monotonic_ns=stamps.value("t_cleanup_done") or 0,
                    cleanup_result="released",
                )
        except HqsbError as exc:
            if not code:
                details = getattr(exc, "details", {}) or {}
                if details.get("fault") == "oom":
                    code = "backend_oom"
                elif isinstance(exc, SchemaError):
                    # identity mismatch or a malformed answer is a correctness
                    # problem, not a transient availability problem
                    code = "backend_identity_mismatch"
                else:
                    code = "backend_unavailable"
            status = self.catalog.entry(code).http_status
            detached_reason = f"{type(exc).__name__}: {exc}"
            try:
                if machine.state in ("BACKEND_SUBMITTED", "FIRST_TOKEN_READY", "STREAMING"):
                    machine.transition(
                        "BACKEND_FAILED", reason=str(exc), monotonic_ns=self._now(writer)
                    )
                    machine.transition("CLEANED", reason="backend failure", monotonic_ns=self._now(writer))
                elif machine.state in ("ROUTED", "QUEUED", "ADMISSION_PENDING"):
                    machine.transition("BACKEND_FAILED", reason=str(exc), monotonic_ns=0)
                    machine.transition("CLEANED", reason="backend failure", monotonic_ns=0)
                else:
                    machine.transition("CLEANED", reason=str(exc), monotonic_ns=0)
            except SchemaError:
                pass
            if not wrote_head:
                entry = self.catalog.entry(code)
                body = json.dumps(self.catalog.wire_body(code)).encode()
                writer.write_head(entry.http_status, {"Content-Type": "application/json"})
                writer.write_body(body)
                response_body = body
                wrote_head = True
        finally:
            self._inflight = max(0, self._inflight - 1)
            if session is not None:
                try:
                    cleanup = session.cleanup()
                except Exception as exc:  # noqa: BLE001 - cleanup must never crash the gateway
                    cleanup = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self._count("cleanup_ok" if cleanup.get("ok", True) else "cleanup_failed")
            self._count("requests_completed" if code == "" else f"failed_{code}")

        outcome = RequestOutcome(
            request_id=canonical.request_id,
            trace_id=machine.trace_id,
            status=status,
            code=code,
            stream=canonical.stream,
            wrote_head=wrote_head,
            frames=frames,
            body_bytes=len(response_body) if not canonical.stream else len(raw_sse),
            ledger=ledger,
            timestamps=stamps,
            transitions=list(machine.records),
            terminal_state=machine.state,
            backend_id=str(selection.get("instance_id", "")),
            model_epoch=str(getattr(session, "model_epoch", "") if session else ""),
            router_reason=str(selection.get("reason", "")),
            cancel_propagated=cancel_propagated,
            deadline_exceeded=deadline_exceeded,
            detached_reason=detached_reason,
            raw_sse=bytes(raw_sse),
        )
        # the delivery ledger must audit clean; a violation is an interface error
        audit = ledger.audit()
        if not audit["ok"]:
            raise SchemaError(
                "delivery ledger violates its ordering invariant: "
                + "; ".join(audit["problems"])
            )
        if self.telemetry is not None:
            self.telemetry.record_outcome(outcome)
            for record in machine.records:
                self.telemetry.record_transition(record)
        return outcome

    def _normalize_response(
        self, payload: Mapping[str, Any], canonical: CanonicalBackendRequest
    ) -> Dict[str, Any]:
        """Shape the non-stream body to the frozen response fields."""
        normalized = dict(payload)
        normalized.setdefault("id", canonical.request_id)
        normalized.setdefault(
            "object",
            "chat.completion"
            if canonical.endpoint.endswith("chat/completions")
            else "text_completion",
        )
        normalized.setdefault("created", 0)
        normalized.setdefault("model", canonical.model_alias)
        for choice in normalized.get("choices", []):
            choice.setdefault("logprobs", None)
        return normalized


__all__ = [
    "ALLOWED_REQUEST_TRANSITIONS",
    "AdmissionDecider",
    "BackendChunk",
    "BackendSession",
    "BRANCH_STATES",
    "GatewayConfig",
    "REQUEST_STATES",
    "RequestOutcome",
    "RequestStateMachine",
    "RouteSelector",
    "SERVICE_STATES",
    "ServingBackend",
    "ServingGateway",
    "TERMINAL_REQUEST_STATES",
    "TelemetrySink",
    "TransitionRecord",
]
