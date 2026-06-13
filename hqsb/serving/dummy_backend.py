"""Deterministic Backend fixture for protocol work (E08-01 step 3).

E08-01 requires a *model-free* fixture: protocol, SSE framing and error mapping
must be provable without a model in the loop.  This module is that fixture, and
it is also the target for fault-injection precision checks (E08-09 step 2):
every fault is declared ahead of time, has an exact trigger, and the injector's
own records are returned so the run can prove the fault really happened.

It is deliberately a *fixture*, not a runtime: :meth:`DummyServingBackend.claim_allowed`
always says ``False`` so no report can quote it as a real Backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import BackendError, ConfigError

from hqsb.serving.gateway import BackendChunk, ServingBackend  # noqa: F401 - protocol ref
from hqsb.serving.protocol import CanonicalBackendRequest

#: Fault kinds the fixture can inject (frozen vocabulary).
FAULT_KINDS: Tuple[str, ...] = (
    "unavailable",
    "oom",
    "internal",
    "hang",
    "identity_mismatch",
    "cache_unavailable",
)


@dataclass(frozen=True)
class DummyFault:
    """A declared fault: exact kind, exact trigger position."""

    kind: str
    at_token: int = 0
    after_commit: bool = False
    duration_ms: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ConfigError(
                f"unknown dummy fault {self.kind!r}; expected one of {list(FAULT_KINDS)}"
            )
        if self.kind == "hang" and self.duration_ms <= 0:
            raise ConfigError("a hang fault needs an explicit duration")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "at_token": self.at_token,
            "after_commit": self.after_commit,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
        }


@dataclass
class DummyScript:
    """What the fixture generates (tokens, text, finish reason, timings)."""

    token_ids: Tuple[int, ...] = (11, 12, 13)
    texts: Tuple[str, ...] = ()
    finish_reason: str = "stop"
    first_token_delay_ms: float = 0.0
    per_token_delay_ms: float = 0.0
    usage: Optional[Mapping[str, int]] = None
    fault: Optional[DummyFault] = None
    model_epoch: str = "epoch-1"

    def text_for(self, index: int) -> str:
        if self.texts:
            return self.texts[index % len(self.texts)]
        return f"tok{self.token_ids[index]} "


@dataclass
class DummyBackendSession:
    """One in-flight fixture request; deterministic clock, explicit cancel."""

    request: CanonicalBackendRequest
    backend: "DummyServingBackend"
    script: DummyScript
    model_epoch: str
    backend_request_id: str
    ticks: List[int] = field(default_factory=list)
    cancelled: bool = False
    cancelled_at_token: int = -1
    cleaned: bool = False
    chunks_served: int = 0
    #: identity captured at open time (a fault may report a different one)
    reported_identity: Mapping[str, Any] = field(default_factory=dict)

    def model_identity(self) -> Mapping[str, Any]:
        return dict(self.reported_identity or self.backend.identity)

    def next_chunk(self) -> Optional[BackendChunk]:
        fault = self.script.fault
        if self.cancelled:
            return None
        if fault is not None:
            if fault.kind == "unavailable" and self.chunks_served == fault.at_token:
                raise BackendError("dummy backend is unavailable", details={"fault": fault.kind})
            if fault.kind == "oom" and self.chunks_served == fault.at_token:
                raise BackendError("dummy backend ran out of memory", details={"fault": fault.kind})
            if fault.kind == "internal" and self.chunks_served == fault.at_token:
                raise BackendError("dummy backend internal error", details={"fault": fault.kind})
            if fault.kind == "cache_unavailable" and self.chunks_served == fault.at_token:
                self.backend.cache_available = False
                self.backend.fault_records.append(
                    {"fault": fault.kind, "at_token": self.chunks_served, "applied": True}
                )
            if fault.kind == "hang" and self.chunks_served == fault.at_token:
                self.backend.fault_records.append(
                    {"fault": fault.kind, "at_token": self.chunks_served, "applied": True}
                )
                # a hang returns no chunk; the gateway's deadline/watchdog is what
                # must break the wait, never an infinite loop inside the fixture.
                return None
        if self.chunks_served >= len(self.script.token_ids):
            return None
        index = self.chunks_served
        self.chunks_served += 1
        self.ticks.append(self.backend.tick())
        text = self.script.text_for(index)
        usage = None
        finish = ""
        if self.chunks_served == len(self.script.token_ids):
            finish = self.script.finish_reason
            usage = dict(
                self.script.usage
                or {
                    "prompt_tokens": self.request.prompt_tokens,
                    "completion_tokens": len(self.script.token_ids),
                    "total_tokens": self.request.prompt_tokens + len(self.script.token_ids),
                }
            )
        return BackendChunk(
            token_id=self.script.token_ids[index],
            text=text,
            committed=True,
            finish_reason=finish,
            usage=usage,
            model_epoch=self.model_epoch,
        )

    def cancel(self, *, reason: str) -> None:
        self.cancelled = True
        self.cancelled_at_token = self.chunks_served
        self.backend.cancel_records.append(
            {
                "backend_request_id": self.backend_request_id,
                "reason": reason,
                "at_token": self.chunks_served,
            }
        )

    def cleanup(self) -> Mapping[str, Any]:
        self.cleaned = True
        return {"ok": True, "backend_request_id": self.backend_request_id}


class DummyServingBackend:
    """A registered Backend instance with declared capabilities (fixture)."""

    def __init__(
        self,
        *,
        instance_id: str = "dummy-0",
        identity: Mapping[str, Any],
        model_aliases: Sequence[str] = ("dummy-model",),
        precision: str = "float16",
        capability: Optional[Mapping[str, str]] = None,
        script: Optional[DummyScript] = None,
        tick_ns: int = 1_000_000,
        ready: bool = True,
        healthy: bool = True,
        model_epoch: str = "epoch-1",
    ) -> None:
        self.instance_id = instance_id
        self.identity = dict(identity)
        self.model_aliases = tuple(model_aliases)
        self.precision = precision
        self.capability = dict(
            capability
            or {
                "model_artifact": "SUPPORTED_EXACT",
                "precision": "SUPPORTED_EXACT",
                "streaming": "SUPPORTED_EXACT",
                "cancel": "SUPPORTED_EXACT",
                "max_context_tokens": "SUPPORTED_EXACT",
            }
        )
        self.script = script or DummyScript()
        self.ready = ready
        self.healthy = healthy
        self.model_epoch = model_epoch
        self.cache_available = True
        self._tick = 0
        self.tick_ns = tick_ns
        self.fault_records: List[Dict[str, Any]] = []
        self.cancel_records: List[Dict[str, Any]] = []
        self.open_count = 0

    # -- fixture policy --------------------------------------------------
    def claim_allowed(self) -> Dict[str, Any]:
        return {
            "allowed": False,
            "reason": "dummy fixture: it proves protocol behaviour, never performance",
        }

    def tick(self) -> int:
        self._tick += 1
        return self._tick * self.tick_ns

    def health(self) -> Mapping[str, Any]:
        return {
            "instance_id": self.instance_id,
            "ready": self.ready,
            "healthy": self.healthy,
            "model_epoch": self.model_epoch,
            "cache_available": self.cache_available,
        }

    def accepts(self, request: CanonicalBackendRequest) -> Mapping[str, Any]:
        reasons: List[str] = []
        if not self.ready:
            reasons.append("not_ready")
        if not self.healthy:
            reasons.append("unhealthy")
        if request.model_alias not in self.model_aliases:
            reasons.append("model_alias_not_served")
        if request.prompt_tokens + request.reserved_output_tokens > int(
            self.capability.get("max_context_tokens_value", 32768)
        ):
            reasons.append("context_exceeded")
        return {"accepted": not reasons, "reasons": reasons}

    def open(self, request: CanonicalBackendRequest, *, deadline_ns: int) -> DummyBackendSession:
        accepted = self.accepts(request)
        if not accepted["accepted"]:
            raise BackendError(
                f"backend {self.instance_id} refuses the request: {accepted['reasons']}",
                details=dict(accepted),
            )
        self.open_count += 1
        reported: Dict[str, Any] = dict(self.identity)
        if self.script.fault is not None and self.script.fault.kind == "identity_mismatch":
            self.fault_records.append(
                {"fault": "identity_mismatch", "at_token": 0, "applied": True}
            )
            reported["precision"] = "float32"
            reported["model_manifest_sha256"] = "f" * 64
        del deadline_ns
        return DummyBackendSession(
            request=request,
            backend=self,
            script=self.script,
            model_epoch=self.model_epoch,
            backend_request_id=f"{self.instance_id}-{self.open_count}",
            reported_identity=reported,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "model_aliases": list(self.model_aliases),
            "precision": self.precision,
            "capability": dict(self.capability),
            "ready": self.ready,
            "healthy": self.healthy,
            "model_epoch": self.model_epoch,
            "cache_available": self.cache_available,
            "faults": [dict(item) for item in self.fault_records],
            "cancels": [dict(item) for item in self.cancel_records],
        }


def frozen_identity(
    *, model_id: str = "Qwen/Qwen3-1.7B", precision: str = "float16", revision: str = "rev-1"
) -> Dict[str, Any]:
    """A frozen ModelArtifact identity for fixtures (hashes are placeholders)."""
    return {
        "model_id": model_id,
        "revision": revision,
        "precision": precision,
        "model_manifest_sha256": "0" * 64,
        "tokenizer_id": model_id,
        "chat_template_hash": "template-hash",
    }


def tokenizer_for_vocab(vocab: int = 256):
    """A deterministic character tokenizer for protocol fixtures."""

    def tokenize(text: str) -> Tuple[int, ...]:
        return tuple(ord(char) % vocab for char in text)

    return tokenize


def template_identity(messages: Sequence[Mapping[str, Any]]) -> str:
    """A deterministic chat template: roles and content, nothing fancy."""
    return "\n".join(f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages)


def detect_fault_applied(backend: DummyServingBackend, kind: str) -> bool:
    """Injector precision (E08-09 step 2): the fault must be recorded once."""
    matches = [item for item in backend.fault_records if item.get("fault") == kind]
    return bool(matches) and all(item.get("applied") for item in matches)


__all__ = [
    "FAULT_KINDS",
    "DummyBackendSession",
    "DummyFault",
    "DummyScript",
    "DummyServingBackend",
    "detect_fault_applied",
    "frozen_identity",
    "template_identity",
    "tokenizer_for_vocab",
]
