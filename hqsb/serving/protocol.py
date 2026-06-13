"""Protocol plane: the frozen OpenAI-compatible subset and its error catalog.

This module owns three things and nothing else:

* :class:`ProtocolProfile` — what HQSB actually implements (wire + schema +
  semantic + lifecycle compatibility are four different claims, details/S08
  E08-01 §2).  Everything outside the subset is an explicit
  ``UNSUPPORTED_REJECT``; nothing is silently dropped;
* :class:`ErrorCatalog` — one frozen entry per failure: HTTP status, stable
  code, retryability, pipeline stage and whether a Backend request may exist.
  Permanent errors must not be disguised as 500 and overload must not be
  disguised as an input error (E08-01 §8);
* the canonical request path — raw bytes → JSON → schema → semantic → alias →
  template/tokenize → budget → :class:`CanonicalBackendRequest`, keeping the
  ``requested`` / ``normalized`` / ``actual`` views separable (E08-01 §5).

The plane is pure Python: no HTTP framework, no engine SDK.  Transport lives in
:mod:`hqsb.serving.transport`, orchestration in :mod:`hqsb.serving.gateway`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Wire surface of the frozen subset.
DEFAULT_PROFILE_VERSION = "s08.1"

#: Field statuses inside the frozen subset.
FIELD_REQUIRED = "required"
FIELD_OPTIONAL = "optional"
FIELD_UNSUPPORTED = "unsupported"

#: Views kept separable for auditability (E08-01 §5).
REQUESTED = "requested"
NORMALIZED = "normalized"
ACTUAL = "actual"

#: Retryability values in the error catalog.
RETRY_YES = "yes"
RETRY_NO = "no"
RETRY_CONDITIONAL = "conditional"

STREAM_TRUE = "stream_true"
STREAM_FALSE = "stream_false"

#: Value ranges enforced by semantic validation.
SAMPLING_RANGES: Mapping[str, Tuple[float, float]] = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
}


@dataclass(frozen=True)
class SseProfile:
    """SSE wire constants of the frozen subset."""

    content_type: str
    data_prefix: str
    event_separator: str
    terminal_marker: str
    usage_frame: str
    heartbeat_comment: str

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "SseProfile":
        return cls(
            content_type=str(payload["content_type"]),
            data_prefix=str(payload["data_field"]),
            event_separator=str(payload["event_separator"]),
            terminal_marker=str(payload["terminal_marker"]),
            usage_frame=str(payload["usage_frame"]),
            heartbeat_comment=str(payload["heartbeat_comment"]),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type,
            "data_prefix": self.data_prefix,
            "event_separator": self.event_separator,
            "terminal_marker": self.terminal_marker,
            "usage_frame": self.usage_frame,
            "heartbeat_comment": self.heartbeat_comment,
        }


@dataclass(frozen=True)
class ProtocolProfile:
    """The declared compatible subset (a claim, not a measurement)."""

    profile_version: str
    schema_version: str
    endpoints: Tuple[str, ...]
    stream_modes: Tuple[str, ...]
    completion_request_fields: Mapping[str, str]
    chat_request_fields: Mapping[str, str]
    message_roles: Tuple[str, ...]
    response_fields: Tuple[str, ...]
    choice_fields: Tuple[str, ...]
    finish_reasons: Tuple[str, ...]
    sse: SseProfile
    limits: Mapping[str, int]
    policies: Mapping[str, str]
    trace_headers: Tuple[str, ...]
    extension_namespace: str
    unsupported_features: Tuple[str, ...]
    error_envelope_fields: Tuple[str, ...]
    upstream_reference: str = ""

    # ── construction ───────────────────────────────────────────────────
    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "ProtocolProfile":
        try:
            return cls(
                profile_version=str(payload["profile_version"]),
                schema_version=str(payload["schema_version"]),
                endpoints=tuple(str(item) for item in payload["endpoints"]),
                stream_modes=tuple(str(item) for item in payload["stream_modes"]),
                completion_request_fields={
                    str(k): str(v)
                    for k, v in dict(payload["completion_request_fields"]).items()
                },
                chat_request_fields={
                    str(k): str(v) for k, v in dict(payload["chat_request_fields"]).items()
                },
                message_roles=tuple(str(item) for item in payload["message_roles"]),
                response_fields=tuple(str(item) for item in payload["response_fields"]),
                choice_fields=tuple(str(item) for item in payload["choice_fields"]),
                finish_reasons=tuple(str(item) for item in payload["finish_reasons"]),
                sse=SseProfile.from_document(dict(payload["sse"])),
                limits={str(k): int(v) for k, v in dict(payload["limits"]).items()},
                policies={str(k): str(v) for k, v in dict(payload["policies"]).items()},
                trace_headers=tuple(str(item) for item in payload["trace_headers"]),
                extension_namespace=str(payload["extension_namespace"]),
                unsupported_features=tuple(
                    str(item) for item in payload["unsupported_features"]
                ),
                error_envelope_fields=tuple(
                    str(item) for item in payload["error_envelope_fields"]
                ),
                upstream_reference=str(payload.get("upstream_reference", "")),
            )
        except KeyError as exc:  # pragma: no cover - guarded by the YAML audit
            raise ConfigError(
                f"protocol profile is missing key {exc.args[0]!r}",
                details={"field": str(exc.args[0])},
            ) from exc

    # ── queries ────────────────────────────────────────────────────────
    def request_fields(self, endpoint: str) -> Mapping[str, str]:
        if endpoint == "/v1/completions":
            return self.completion_request_fields
        if endpoint == "/v1/chat/completions":
            return self.chat_request_fields
        raise ConfigError(
            f"unknown endpoint {endpoint!r}; the frozen subset declares "
            f"{list(self.endpoints)}",
            details={"endpoint": endpoint},
        )

    def field_status(self, endpoint: str, name: str) -> str:
        return self.request_fields(endpoint).get(name, FIELD_UNSUPPORTED)

    def require_implemented(self, feature: str) -> None:
        """Refuse a capability that is not part of the frozen subset."""
        if feature in self.unsupported_features:
            raise ConfigError(
                f"feature {feature!r} is explicitly outside the frozen compatible "
                "subset; claiming it would be a wire/semantic overstatement",
                details={"feature": feature},
            )

    @property
    def schema_hash(self) -> str:
        """Deterministic hash of the declared subset (identity for reports)."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "schema_version": self.schema_version,
            "endpoints": list(self.endpoints),
            "stream_modes": list(self.stream_modes),
            "completion_request_fields": dict(self.completion_request_fields),
            "chat_request_fields": dict(self.chat_request_fields),
            "message_roles": list(self.message_roles),
            "response_fields": list(self.response_fields),
            "choice_fields": list(self.choice_fields),
            "finish_reasons": list(self.finish_reasons),
            "sse": self.sse.as_dict(),
            "limits": dict(self.limits),
            "policies": dict(self.policies),
            "trace_headers": list(self.trace_headers),
            "extension_namespace": self.extension_namespace,
            "unsupported_features": list(self.unsupported_features),
            "error_envelope_fields": list(self.error_envelope_fields),
            "upstream_reference": self.upstream_reference,
        }


# ── error catalog ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ErrorEntry:
    """One frozen failure: status, code, retryability, stage."""

    code: str
    http_status: int
    stage: str
    category: str
    retryable: str
    backend_request_created: bool
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "http_status": self.http_status,
            "stage": self.stage,
            "category": self.category,
            "retryable": self.retryable,
            "backend_request_created": self.backend_request_created,
            "message": self.message,
        }


@dataclass(frozen=True)
class ErrorCatalog:
    """The frozen error matrix plus its consistency rules."""

    entries: Mapping[str, ErrorEntry]
    stages: Tuple[str, ...]
    retryable_values: Tuple[str, ...]
    envelope_fields: Tuple[str, ...]
    schema_version: str = "1.0.0"

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "ErrorCatalog":
        entries: Dict[str, ErrorEntry] = {}
        for raw in payload["entries"]:
            item = dict(raw)
            code = str(item["code"])
            if code in entries:
                raise ConfigError(
                    f"duplicate error code {code!r} in the catalog; a duplicated entry "
                    "would make retryability ambiguous",
                    details={"code": code},
                )
            entries[code] = ErrorEntry(
                code=code,
                http_status=int(item["http_status"]),
                stage=str(item["stage"]),
                category=str(item["category"]),
                retryable=str(item["retryable"]),
                backend_request_created=bool(item["backend_request_created"]),
                message=str(item["message"]),
            )
        return cls(
            entries=entries,
            stages=tuple(str(item) for item in payload["stages"]),
            retryable_values=tuple(str(item) for item in payload["retryable_values"]),
            envelope_fields=tuple(str(item) for item in payload["envelope_fields"]),
            schema_version=str(payload.get("schema_version", "1.0.0")),
        )

    def entry(self, code: str) -> ErrorEntry:
        if code not in self.entries:
            raise ConfigError(
                f"error code {code!r} is not in the frozen catalog",
                details={"code": code},
            )
        return self.entries[code]

    def validate(self) -> Dict[str, Any]:
        """The catalog's own consistency rules (E08-01 §8 / E08-05 §7)."""
        problems: List[str] = []
        for code, entry in sorted(self.entries.items()):
            if entry.stage not in self.stages:
                problems.append(f"{code}: unknown stage {entry.stage!r}")
            if entry.retryable not in self.retryable_values:
                problems.append(f"{code}: unknown retryability {entry.retryable!r}")
            if entry.category == "invalid" and entry.http_status >= 500:
                problems.append(
                    f"{code}: an invalid request must not be answered with "
                    f"{entry.http_status} (permanent errors must not disguise as 500)"
                )
            if entry.category == "overload" and entry.http_status not in (429, 503):
                problems.append(
                    f"{code}: overload must use 429/503, got {entry.http_status}"
                )
            if entry.category == "invalid" and entry.backend_request_created:
                problems.append(
                    f"{code}: an invalid request must be refused before a Backend "
                    "request exists"
                )
            if entry.code == "backend_internal" and entry.http_status != 500:
                problems.append("backend_internal must map to 500")
        if "invalid_json" not in self.entries:
            problems.append("invalid_json is part of the frozen matrix and is missing")
        if "client_cancelled" not in self.entries:
            problems.append("client_cancelled is part of the frozen matrix and is missing")
        return {"ok": not problems, "problems": problems, "entries": len(self.entries)}

    def wire_body(self, code: str, *, message: str = "", param: str = "") -> Dict[str, Any]:
        entry = self.entry(code)
        return {
            "error": {
                "message": message or entry.message,
                "type": entry.category,
                "code": entry.code,
                "param": param,
            }
        }

    def overload_codes(self) -> Tuple[str, ...]:
        return tuple(
            code
            for code, entry in sorted(self.entries.items())
            if entry.category == "overload"
        )

    def is_retryable(self, code: str, *, committed: bool = False) -> bool:
        """Retryability, honouring the streaming commit boundary."""
        entry = self.entry(code)
        if entry.retryable == RETRY_NO:
            return False
        if entry.retryable == RETRY_YES:
            return True
        return not committed


# ── validation ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ValidationIssue:
    """One located validation failure (field-level localisation)."""

    path: str
    code: str
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "code": self.code, "message": self.message}


class RequestRejected(Exception):
    """Internal control-flow carrier for a structured rejection."""

    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(f"{issue.code} at {issue.path}: {issue.message}")
        self.issue = issue


@dataclass(frozen=True)
class NormalizedRequest:
    """The default-expanded request, plus the two other views."""

    endpoint: str
    requested: Mapping[str, Any]
    normalized: Mapping[str, Any]
    expanded_defaults: Mapping[str, Any]
    stream: bool
    model_alias: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "requested": dict(self.requested),
            "normalized": dict(self.normalized),
            "expanded_defaults": dict(self.expanded_defaults),
            "stream": self.stream,
            "model_alias": self.model_alias,
        }


@dataclass(frozen=True)
class CanonicalBackendRequest:
    """What the Backend actually receives (the ``actual`` view)."""

    request_id: str
    endpoint: str
    model_alias: str
    model_identity: Mapping[str, Any]
    input_token_ids: Tuple[int, ...]
    prompt_tokens: int
    reserved_output_tokens: int
    sampling: Mapping[str, Any]
    stop_sequences: Tuple[str, ...]
    stream: bool
    context_limit_tokens: int
    requested: Mapping[str, Any]
    normalized: Mapping[str, Any]
    actual: Mapping[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "model_alias": self.model_alias,
            "model_identity": dict(self.model_identity),
            "input_token_ids": list(self.input_token_ids),
            "prompt_tokens": self.prompt_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "sampling": dict(self.sampling),
            "stop_sequences": list(self.stop_sequences),
            "stream": self.stream,
            "context_limit_tokens": self.context_limit_tokens,
            "requested": dict(self.requested),
            "normalized": dict(self.normalized),
            "actual": dict(self.actual),
        }


def parse_json_body(raw: bytes) -> Dict[str, Any]:
    """Transport + parse stage: refuse non-UTF8 and malformed JSON structurally."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestRejected(
            ValidationIssue("$", "invalid_json", f"body is not valid UTF-8: {exc}")
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RequestRejected(
            ValidationIssue("$", "invalid_json", f"malformed JSON: {exc.msg}")
        ) from exc
    if not isinstance(payload, dict):
        raise RequestRejected(
            ValidationIssue("$", "wrong_type", "the request body must be a JSON object")
        )
    return payload


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


_FIELD_TYPES: Mapping[str, str] = {
    "model": "string",
    "prompt": "string_or_array",
    "messages": "array",
    "max_tokens": "integer",
    "temperature": "number",
    "top_p": "number",
    "seed": "integer",
    "stop": "string_or_array",
    "stream": "boolean",
    "stream_options": "object",
    "logprobs": "boolean",
    "n": "integer",
    "tools": "array",
    "response_format": "object",
}

_FIELD_DEFAULTS: Mapping[str, Callable[[Mapping[str, int]], Any]] = {
    "max_tokens": lambda limits: int(limits["default_completion_tokens"]),
    "temperature": lambda _limits: 1.0,
    "top_p": lambda _limits: 1.0,
    "seed": lambda _limits: None,
    "stop": lambda _limits: (),
    "stream": lambda _limits: False,
    "stream_options": lambda _limits: {"include_usage": False},
}


def validate_request(
    payload: Mapping[str, Any],
    *,
    endpoint: str,
    profile: ProtocolProfile,
    catalog: ErrorCatalog,
    body_bytes: int = 0,
) -> NormalizedRequest:
    """Schema + semantic validation with field-level localisation."""
    del catalog  # the catalog is passed for symmetry with the transport layer
    if endpoint not in profile.endpoints:
        raise RequestRejected(
            ValidationIssue("$path", "unknown_field", f"endpoint {endpoint!r} is not served")
        )
    if body_bytes and body_bytes > int(profile.limits["max_body_bytes"]):
        raise RequestRejected(
            ValidationIssue("$body", "body_too_large", "request body exceeds the limit")
        )
    fields = profile.request_fields(endpoint)

    # unknown field: refused, never dropped (E08-01 §14)
    for name in payload:
        status = fields.get(name)
        if status is None:
            if name.startswith(profile.extension_namespace + "."):
                continue  # namespaced extension: allowed only if declared
            raise RequestRejected(
                ValidationIssue(f"$.{name}", "unknown_field", "unknown field refused")
            )
        if status == FIELD_UNSUPPORTED:
            raise RequestRejected(
                ValidationIssue(
                    f"$.{name}",
                    "unsupported_parameter",
                    f"{name!r} is outside the frozen compatible subset",
                )
            )

    # required fields
    for name, status in fields.items():
        if status == FIELD_REQUIRED and name not in payload:
            raise RequestRejected(
                ValidationIssue(f"$.{name}", "missing_required_field", "field is required")
            )

    # types
    for name, expected in _FIELD_TYPES.items():
        if name not in payload:
            continue
        value = payload[name]
        if expected == "string_or_array":
            if not isinstance(value, (str, list)):
                raise RequestRejected(
                    ValidationIssue(f"$.{name}", "wrong_type", "expected string or array")
                )
            if isinstance(value, list) and not value:
                raise RequestRejected(
                    ValidationIssue(f"$.{name}", "empty_messages", "must not be empty")
                )
            if isinstance(value, list) and any(not isinstance(item, str) for item in value):
                raise RequestRejected(
                    ValidationIssue(f"$.{name}", "wrong_type", "array items must be strings")
                )
        elif not _type_ok(value, expected):
            raise RequestRejected(
                ValidationIssue(f"$.{name}", "wrong_type", f"expected {expected}")
            )

    # semantic: prompt / messages
    if endpoint == "/v1/chat/completions":
        messages = list(payload["messages"])
        if not messages:
            raise RequestRejected(
                ValidationIssue("$.messages", "empty_messages", "messages must not be empty")
            )
        if len(messages) > int(profile.limits["max_messages"]):
            raise RequestRejected(
                ValidationIssue("$.messages", "context_length_exceeded", "too many messages")
            )
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise RequestRejected(
                    ValidationIssue(
                        f"$.messages[{index}]",
                        "wrong_type",
                        "each message needs role and content",
                    )
                )
            if message["role"] not in profile.message_roles:
                raise RequestRejected(
                    ValidationIssue(
                        f"$.messages[{index}].role",
                        "invalid_role",
                        f"role {message['role']!r} is not supported",
                    )
                )
            if not isinstance(message["content"], str):
                raise RequestRejected(
                    ValidationIssue(
                        f"$.messages[{index}].content",
                        "wrong_type",
                        "content must be a string in the frozen subset",
                    )
                )
    else:
        prompt = payload["prompt"]
        if isinstance(prompt, str) and not prompt:
            raise RequestRejected(
                ValidationIssue("$.prompt", "empty_messages", "prompt must not be empty")
            )

    # semantic: sampling ranges
    for name, (low, high) in SAMPLING_RANGES.items():
        if name in payload and payload[name] is not None:
            value = float(payload[name])
            if name == "top_p":
                if not (low < value <= high):
                    raise RequestRejected(
                        ValidationIssue(f"$.{name}", "invalid_sampling", f"{name} out of range")
                    )
            elif not (low <= value <= high):
                raise RequestRejected(
                    ValidationIssue(f"$.{name}", "invalid_sampling", f"{name} out of range")
                )
    if "max_tokens" in payload and payload["max_tokens"] is not None:
        if int(payload["max_tokens"]) <= 0:
            raise RequestRejected(
                ValidationIssue("$.max_tokens", "invalid_sampling", "max_tokens must be > 0")
            )
        if int(payload["max_tokens"]) > int(profile.limits["max_completion_tokens"]):
            raise RequestRejected(
                ValidationIssue(
                    "$.max_tokens",
                    "max_tokens_exceeded",
                    "max_tokens exceeds the frozen completion limit",
                )
            )

    # semantic: stop sequences
    stop_raw = payload.get("stop", ())
    stop_sequences: Tuple[str, ...]
    if isinstance(stop_raw, str):
        stop_sequences = (stop_raw,)
    else:
        stop_sequences = tuple(str(item) for item in stop_raw or ())
    if len(stop_sequences) > int(profile.limits["max_stop_sequences"]):
        raise RequestRejected(
            ValidationIssue("$.stop", "invalid_stop", "too many stop sequences")
        )
    if any(not item for item in stop_sequences):
        raise RequestRejected(
            ValidationIssue("$.stop", "invalid_stop", "empty stop sequence")
        )

    # semantic: unsupported combinations
    stream = bool(payload.get("stream", False))
    stream_options = payload.get("stream_options") or {}
    if stream_options and not stream:
        raise RequestRejected(
            ValidationIssue(
                "$.stream_options",
                "unsupported_combo",
                "stream_options requires stream=true",
            )
        )
    if "include_usage" in stream_options and not isinstance(
        stream_options["include_usage"], bool
    ):
        raise RequestRejected(
            ValidationIssue(
                "$.stream_options.include_usage", "wrong_type", "expected boolean"
            )
        )

    # defaults are expanded *now* and enter the config hash (E08-01 §5)
    normalized: Dict[str, Any] = dict(payload)
    expanded: Dict[str, Any] = {}
    for name, factory in _FIELD_DEFAULTS.items():
        if name not in normalized:
            value = factory(profile.limits)
            normalized[name] = list(value) if isinstance(value, tuple) else value
            expanded[name] = normalized[name]
        elif name == "stop":
            normalized[name] = list(stop_sequences)
        elif name == "stream_options":
            merged = {"include_usage": False}
            merged.update(dict(normalized[name]))
            normalized[name] = merged
    return NormalizedRequest(
        endpoint=endpoint,
        requested=dict(payload),
        normalized=normalized,
        expanded_defaults=expanded,
        stream=stream,
        model_alias=str(payload["model"]),
    )


@dataclass(frozen=True)
class ModelResolution:
    """Alias → frozen ModelArtifact identity."""

    alias: str
    identity: Mapping[str, Any]


def resolve_model_alias(
    alias: str, registry: Mapping[str, Mapping[str, Any]]
) -> ModelResolution:
    if alias not in registry:
        raise RequestRejected(
            ValidationIssue("$.model", "unknown_model", f"unknown model alias {alias!r}")
        )
    return ModelResolution(alias=alias, identity=dict(registry[alias]))


def canonicalize_request(
    normalized: NormalizedRequest,
    *,
    request_id: str,
    resolution: ModelResolution,
    tokenizer: Callable[[str], Sequence[int]],
    chat_template: Callable[[Sequence[Mapping[str, Any]]], str],
    profile: ProtocolProfile,
) -> CanonicalBackendRequest:
    """Template/tokenize + budget validation → the canonical Backend request."""
    if normalized.endpoint == "/v1/chat/completions":
        text = chat_template(list(normalized.normalized["messages"]))
    else:
        prompt = normalized.normalized["prompt"]
        text = prompt if isinstance(prompt, str) else "\n".join(str(item) for item in prompt)
    token_ids = tuple(int(item) for item in tokenizer(text))
    prompt_tokens = len(token_ids)
    reserved = int(normalized.normalized.get("max_tokens") or 0)

    if prompt_tokens > int(profile.limits["max_prompt_tokens"]):
        raise RequestRejected(
            ValidationIssue(
                "$.prompt",
                "context_length_exceeded",
                f"prompt has {prompt_tokens} tokens > limit "
                f"{profile.limits['max_prompt_tokens']}",
            )
        )
    if prompt_tokens + reserved > int(profile.limits["max_context_tokens"]):
        raise RequestRejected(
            ValidationIssue(
                "$.max_tokens",
                "context_length_exceeded",
                "prompt plus reserved output exceeds the context limit",
            )
        )

    sampling = {
        "mode": "sampling" if float(normalized.normalized["temperature"]) > 0 else "greedy",
        "temperature": float(normalized.normalized["temperature"]),
        "top_p": float(normalized.normalized["top_p"]),
        "seed": normalized.normalized["seed"],
    }
    actual = {
        "model_identity": dict(resolution.identity),
        "input_token_ids": list(token_ids),
        "prompt_tokens": prompt_tokens,
        "reserved_output_tokens": reserved,
        "sampling": dict(sampling),
        "stream": normalized.stream,
        "context_limit_tokens": int(profile.limits["max_context_tokens"]),
        "template_applied": normalized.endpoint == "/v1/chat/completions",
        "truncation_applied": False,
    }
    return CanonicalBackendRequest(
        request_id=request_id,
        endpoint=normalized.endpoint,
        model_alias=resolution.alias,
        model_identity=dict(resolution.identity),
        input_token_ids=token_ids,
        prompt_tokens=prompt_tokens,
        reserved_output_tokens=reserved,
        sampling=sampling,
        stop_sequences=tuple(normalized.normalized.get("stop") or ()),
        stream=normalized.stream,
        context_limit_tokens=int(profile.limits["max_context_tokens"]),
        requested=dict(normalized.requested),
        normalized=dict(normalized.normalized),
        actual=actual,
    )


def request_pipeline(
    raw: bytes,
    *,
    endpoint: str,
    request_id: str,
    profile: ProtocolProfile,
    catalog: ErrorCatalog,
    model_registry: Mapping[str, Mapping[str, Any]],
    tokenizer: Callable[[str], Sequence[int]],
    chat_template: Callable[[Sequence[Mapping[str, Any]]], str],
) -> CanonicalBackendRequest:
    """The whole canonical path: bytes → validated, tokenized Backend request.

    Each stage stops at its own error class; nothing is repaired downstream
    (E08-01 §5).
    """
    payload = parse_json_body(raw)
    normalized = validate_request(
        payload, endpoint=endpoint, profile=profile, catalog=catalog, body_bytes=len(raw)
    )
    resolution = resolve_model_alias(normalized.model_alias, model_registry)
    return canonicalize_request(
        normalized,
        request_id=request_id,
        resolution=resolution,
        tokenizer=tokenizer,
        chat_template=chat_template,
        profile=profile,
    )


# ── response oracle + conformance matrix ───────────────────────────────────


def validate_response_body(body: Mapping[str, Any], *, profile: ProtocolProfile) -> List[str]:
    """Non-stream response oracle: the problems a caller must never see."""
    problems: List[str] = []
    for name in profile.response_fields:
        if name not in body:
            problems.append(f"missing response field {name!r}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        problems.append("choices must be a non-empty array")
        return problems
    for index, choice in enumerate(choices):
        for name in profile.choice_fields:
            if name not in choice:
                problems.append(f"choices[{index}] missing {name!r}")
        if choice.get("finish_reason") not in profile.finish_reasons:
            problems.append(f"choices[{index}] has an unknown finish_reason")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        problems.append("usage must be an object")
    else:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if name not in usage:
                problems.append(f"usage missing {name!r}")
        if (
            isinstance(usage.get("prompt_tokens"), int)
            and isinstance(usage.get("completion_tokens"), int)
            and usage.get("total_tokens") != usage["prompt_tokens"] + usage["completion_tokens"]
        ):
            problems.append("usage.total_tokens != prompt_tokens + completion_tokens")
    leaked = [name for name in ("traceback", "stack", "secret", "api_key") if name in body]
    if leaked:
        problems.append(f"response leaks internal detail: {leaked}")
    return problems


@dataclass(frozen=True)
class ConformanceRow:
    """One row of the conformance matrix (E08-01 step 24)."""

    case: str
    endpoint: str
    stream_mode: str
    backend: str
    verdict: str  # PASS / UNSUPPORTED / FAIL / NOT_RUN
    raw_uri: str = ""
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "endpoint": self.endpoint,
            "stream_mode": self.stream_mode,
            "backend": self.backend,
            "verdict": self.verdict,
            "raw_uri": self.raw_uri,
            "reason": self.reason,
        }


CONFORMANCE_VERDICTS: Tuple[str, ...] = ("PASS", "UNSUPPORTED", "FAIL", "NOT_RUN")


def conformance_matrix(rows: Sequence[ConformanceRow]) -> Dict[str, Any]:
    """Refuse a verdict that has no raw evidence behind it."""
    problems: List[str] = []
    for row in rows:
        if row.verdict not in CONFORMANCE_VERDICTS:
            problems.append(f"{row.case}: unknown verdict {row.verdict!r}")
        if row.verdict in ("PASS", "FAIL") and not row.raw_uri:
            problems.append(f"{row.case}: {row.verdict} without a raw URI")
        if row.verdict == "UNSUPPORTED" and not row.reason:
            problems.append(f"{row.case}: UNSUPPORTED needs a reason")
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    return {
        "rows": [row.as_dict() for row in rows],
        "counts": counts,
        "ok": not problems,
        "problems": problems,
    }


# ── table-driven negative corpus (E08-01 step 13) ──────────────────────────


@dataclass(frozen=True)
class NegativeCase:
    name: str
    endpoint: str
    body: bytes
    expected_code: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "expected_code": self.expected_code,
        }


_CHAT = "/v1/chat/completions"


def negative_corpus() -> Tuple[NegativeCase, ...]:
    """Malformed / wrong-type / out-of-range cases with their expected code.

    The corpus is a *fixture*: it proves that the pipeline refuses each case at
    the right stage.  It is not an experiment result.
    """
    good_messages = [{"role": "user", "content": "hi"}]
    cases = [
        NegativeCase("truncated_json", _CHAT, b'{"model": "m", "messages": [', "invalid_json"),
        NegativeCase("not_utf8", _CHAT, b"\xff\xfe\x00", "invalid_json"),
        NegativeCase("top_level_array", _CHAT, b"[]", "wrong_type"),
        NegativeCase("missing_model", _CHAT, b'{"messages": []}', "missing_required_field"),
        NegativeCase(
            "unknown_field",
            _CHAT,
            json.dumps({"model": "m", "messages": good_messages, "foo": 1}).encode(),
            "unknown_field",
        ),
        NegativeCase(
            "unsupported_tools",
            _CHAT,
            json.dumps({"model": "m", "messages": good_messages, "tools": []}).encode(),
            "unsupported_parameter",
        ),
        NegativeCase(
            "wrong_type_messages",
            _CHAT,
            json.dumps({"model": "m", "messages": "hi"}).encode(),
            "wrong_type",
        ),
        NegativeCase(
            "null_content",
            _CHAT,
            json.dumps({"model": "m", "messages": [{"role": "user", "content": None}]}).encode(),
            "wrong_type",
        ),
        NegativeCase(
            "empty_messages",
            _CHAT,
            json.dumps({"model": "m", "messages": []}).encode(),
            "empty_messages",
        ),
        NegativeCase(
            "invalid_role",
            _CHAT,
            json.dumps({"model": "m", "messages": [{"role": "root", "content": "x"}]}).encode(),
            "invalid_role",
        ),
        NegativeCase(
            "temperature_out_of_range",
            _CHAT,
            json.dumps({"model": "m", "messages": good_messages, "temperature": 9}).encode(),
            "invalid_sampling",
        ),
        NegativeCase(
            "top_p_zero",
            _CHAT,
            json.dumps({"model": "m", "messages": good_messages, "top_p": 0}).encode(),
            "invalid_sampling",
        ),
        NegativeCase(
            "negative_max_tokens",
            _CHAT,
            json.dumps({"model": "m", "messages": good_messages, "max_tokens": -1}).encode(),
            "invalid_sampling",
        ),
        NegativeCase(
            "max_tokens_over_limit",
            _CHAT,
            json.dumps({"model": "m", "messages": good_messages, "max_tokens": 10**7}).encode(),
            "max_tokens_exceeded",
        ),
        NegativeCase(
            "too_many_stop",
            "/v1/completions",
            json.dumps({"model": "m", "prompt": "x", "stop": ["a", "b", "c", "d", "e"]}).encode(),
            "invalid_stop",
        ),
        NegativeCase(
            "stream_options_without_stream",
            _CHAT,
            json.dumps(
                {"model": "m", "messages": good_messages, "stream_options": {"include_usage": True}}
            ).encode(),
            "unsupported_combo",
        ),
        NegativeCase(
            "empty_prompt",
            "/v1/completions",
            json.dumps({"model": "m", "prompt": ""}).encode(),
            "empty_messages",
        ),
        NegativeCase(
            "unknown_model",
            _CHAT,
            json.dumps({"model": "not-registered", "messages": good_messages}).encode(),
            "unknown_model",
        ),
    ]
    return tuple(cases)


def evaluate_negative_case(
    case: NegativeCase,
    *,
    profile: ProtocolProfile,
    catalog: ErrorCatalog,
    model_registry: Mapping[str, Mapping[str, Any]],
) -> ValidationIssue:
    """Run one corpus case through the staged pipeline and return its issue.

    Each stage must stop a case at *its* own level; the returned issue is the
    evidence that the refusal happened where the catalog says it should.
    """
    payload = parse_json_body(case.body)
    normalized = validate_request(
        payload,
        endpoint=case.endpoint,
        profile=profile,
        catalog=catalog,
        body_bytes=len(case.body),
    )
    resolve_model_alias(normalized.model_alias, model_registry)
    raise AssertionError(
        f"negative case {case.name!r} was accepted; it claims to be refused with "
        f"{case.expected_code!r}"
    )


__all__ = [
    "ACTUAL",
    "CanonicalBackendRequest",
    "ConformanceRow",
    "DEFAULT_PROFILE_VERSION",
    "ErrorCatalog",
    "ErrorEntry",
    "FIELD_OPTIONAL",
    "FIELD_REQUIRED",
    "FIELD_UNSUPPORTED",
    "NegativeCase",
    "NORMALIZED",
    "NormalizedRequest",
    "ProtocolProfile",
    "REQUESTED",
    "RETRY_CONDITIONAL",
    "RETRY_NO",
    "RETRY_YES",
    "RequestRejected",
    "ModelResolution",
    "SAMPLING_RANGES",
    "STREAM_FALSE",
    "STREAM_TRUE",
    "SseProfile",
    "ValidationIssue",
    "canonicalize_request",
    "conformance_matrix",
    "evaluate_negative_case",
    "negative_corpus",
    "parse_json_body",
    "request_pipeline",
    "resolve_model_alias",
    "validate_request",
    "validate_response_body",
]
