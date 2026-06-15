"""Gateway plane tests: protocol path, streaming, cancel, faults, drain.

Everything here runs against the deterministic in-process transport with the
model-free fixture backend, so the assertions are about *semantics* (framing,
state machine, ledger, cancel propagation), never about performance.
"""

from __future__ import annotations

import json
import os

import pytest
import yaml

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.serving import protocol as P
from hqsb.serving import sse
from hqsb.serving.clients import ClientScript, behavior_matrix
from hqsb.serving.dummy_backend import (
    DummyFault,
    DummyScript,
    DummyServingBackend,
    frozen_identity,
    template_identity,
    tokenizer_for_vocab,
)
from hqsb.serving.gateway import (
    ALLOWED_REQUEST_TRANSITIONS,
    GatewayConfig,
    RequestStateMachine,
    ServingGateway,
)
from hqsb.serving.transport import InProcessTransport, TransportRequest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "serving")

CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"


def load_profile():
    with open(os.path.join(_CONFIG_DIR, "protocol_profile.yaml"), encoding="utf-8") as handle:
        return P.ProtocolProfile.from_document(yaml.safe_load(handle))


def load_catalog():
    with open(os.path.join(_CONFIG_DIR, "error_catalog.yaml"), encoding="utf-8") as handle:
        return P.ErrorCatalog.from_document(yaml.safe_load(handle))


def build_gateway(*, backend=None, backends=None, **kwargs):
    profile = load_profile()
    registry = {"dummy-model": frozen_identity()}
    instance = backend or DummyServingBackend(identity=frozen_identity())
    resolved = backends if backends is not None else {instance.instance_id: instance}
    gateway = ServingGateway(
        profile=profile,
        catalog=load_catalog(),
        model_registry=registry,
        tokenizer=tokenizer_for_vocab(),
        chat_template=template_identity,
        backends=resolved,
        config=GatewayConfig(),
        **kwargs,
    )
    return gateway, instance


def chat_body(**overrides):
    payload = {"model": "dummy-model", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return json.dumps(payload).encode()


class Exchange:
    """Both sides of one request: the client view and the service evidence."""

    def __init__(self, client, service):
        self.client = client
        self.service = service
        self.body = client.body
        self.status = service.status
        self.code = service.code
        self.transitions = service.transitions
        self.ledger = service.ledger
        self.frames = service.frames
        self.detached_reason = service.detached_reason
        self.terminal_state = service.terminal_state
        self.timestamps = service.timestamps


def run(
    gateway,
    body: bytes,
    *,
    path: str = CHAT,
    method: str = "POST",
    behavior: str = "normal_fast_reader",
    **headers,
) -> Exchange:
    script = ClientScript(connection_id="c0", behavior=behavior_matrix()[behavior])
    transport = InProcessTransport(script=script)
    request = TransportRequest(
        method=method,
        path=path,
        headers=headers,
        body=body,
        received_ns=1_000_000,
        connection_id="c0",
    )
    client = transport.run(request, gateway.handle)
    return Exchange(client, transport.last_handler_result)


@pytest.mark.unit
class TestRequestStateMachine:
    def test_happy_path_reaches_cleaned(self):
        machine = RequestStateMachine(request_id="r0", trace_id="t0")
        for state in (
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
        ):
            machine.transition(state, reason="test")
        assert machine.terminated
        assert all(record.reason for record in machine.records)

    def test_illegal_transition_is_refused(self):
        machine = RequestStateMachine(request_id="r0", trace_id="t0")
        with pytest.raises(SchemaError):
            machine.transition("STREAMING", reason="jump")

    def test_every_transition_target_is_a_known_state(self):
        from hqsb.serving.gateway import REQUEST_STATES

        for targets in ALLOWED_REQUEST_TRANSITIONS.values():
            for target in targets:
                assert target in REQUEST_STATES


@pytest.mark.unit
class TestNonStreamPath:
    def test_chat_non_stream_success(self):
        gateway, backend = build_gateway()
        outcome = run(gateway, chat_body())
        assert outcome.status == 200, outcome.detached_reason
        assert outcome.code == ""
        body = json.loads(outcome.body)
        assert P.validate_response_body(body, profile=gateway.profile) == []
        assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 3
        assert outcome.ledger.audit()["ok"]
        assert outcome.ledger.committed == 3
        assert outcome.terminal_state == "CLEANED"
        assert backend.open_count == 1
        assert outcome.timestamps.value("t_backend_submit") is not None

    def test_completion_endpoint_object_type(self):
        gateway, _ = build_gateway()
        outcome = run(
            gateway,
            json.dumps({"model": "dummy-model", "prompt": "hi"}).encode(),
            path=COMPLETIONS,
        )
        body = json.loads(outcome.body)
        assert body["object"] == "text_completion"
        assert outcome.status == 200

    def test_stream_success_framing(self):
        gateway, _ = build_gateway()
        outcome = run(gateway, chat_body(stream=True))
        assert outcome.status == 200
        parsed = sse.parse_stream(outcome.body)
        assert parsed["complete"]
        audit = sse.validate_stream(parsed["parsed"])
        assert audit["ok"], audit["problems"]
        rebuilt = sse.reconstruct(parsed["parsed"])
        assert rebuilt["token_ids"] == [11, 12, 13]
        assert rebuilt["finish_reason"] == "stop"
        assert outcome.ledger.terminal_clean
        assert outcome.frames >= 4  # three deltas + finish + terminal
        # the gateway records the client-visible boundaries; the loadgen fills
        # t_sched/t_send from its own clock, so client_TTFT is completed there
        assert outcome.timestamps.value("t_first_byte_client") is not None
        assert outcome.timestamps.ttft_report()["server_TTFT_ms"] is not None

    def test_stream_and_non_stream_agree(self):
        gateway, _ = build_gateway()
        stream_outcome = run(gateway, chat_body(stream=True))
        plain_outcome = run(gateway, chat_body())
        frames = sse.parse_stream(stream_outcome.body)["parsed"]
        body = json.loads(plain_outcome.body)
        body["choices"][0]["text"] = body["choices"][0]["message"]["content"]
        report = sse.stream_vs_nonstream(frames, body)
        assert report["ok"], report["problems"]

    def test_stream_never_drops_a_token(self):
        gateway, _ = build_gateway()
        outcome = run(gateway, chat_body(stream=True))
        frames = sse.parse_stream(outcome.body)["parsed"]
        sse.require_no_silent_drop(frames, 3)


@pytest.mark.unit
class TestRejections:
    def test_invalid_json_is_a_400_without_backend_state(self):
        gateway, backend = build_gateway()
        outcome = run(gateway, b"{not json")
        assert outcome.status == 400
        assert outcome.code == "invalid_json"
        assert backend.open_count == 0
        assert outcome.terminal_state == "CLEANED"

    def test_unknown_model_is_404_before_any_backend_call(self):
        gateway, backend = build_gateway()
        outcome = run(gateway, chat_body(model="ghost"))
        assert outcome.status == 404
        assert outcome.code == "unknown_model"
        assert backend.open_count == 0

    def test_unsupported_feature_is_refused(self):
        gateway, _ = build_gateway()
        outcome = run(gateway, chat_body(tools=[]))
        assert outcome.status == 400
        assert outcome.code == "unsupported_parameter"

    def test_unknown_path_and_method(self):
        gateway, _ = build_gateway()
        outcome = run(gateway, b"{}", path="/v1/embeddings")
        assert outcome.code == "not_found"
        outcome = run(gateway, chat_body(), method="GET")
        assert outcome.code == "method_not_allowed"

    def test_control_endpoints_report_readiness_without_a_backend_request(self):
        gateway, backend = build_gateway()
        outcome = run(gateway, b"", path="/readyz", method="GET")
        assert outcome.status == 200
        assert backend.open_count == 0
        assert json.loads(outcome.body)["ready"] is True

    def test_rejection_carries_retry_after_for_overload(self):
        class Shedding:
            def decide(self, *, context, snapshot):  # noqa: ARG002
                return {
                    "admitted": False,
                    "code": "service_overloaded",
                    "reason": "pressure state SHEDDING",
                    "retry_after_ms": 2000,
                }

        gateway, _ = build_gateway(admission=Shedding())
        outcome = run(gateway, chat_body())
        assert outcome.status == 503
        assert outcome.code == "service_overloaded"
        assert outcome.client.headers.get("Retry-After") == "2"

    def test_admission_may_not_invent_an_error_code(self):
        class Broken:
            def decide(self, *, context, snapshot):  # noqa: ARG002
                return {"admitted": False, "code": "invented_code", "reason": "?"}

        gateway, _ = build_gateway(admission=Broken())
        with pytest.raises(ConfigError):
            run(gateway, chat_body())


@pytest.mark.unit
class TestFaultsAndCancel:
    def test_backend_unavailable_maps_to_503(self):
        backend = DummyServingBackend(
            identity=frozen_identity(),
            script=DummyScript(fault=DummyFault("unavailable", at_token=0)),
        )
        gateway, _ = build_gateway(backend=backend)
        outcome = run(gateway, chat_body())
        assert outcome.status == 503
        assert outcome.code == "backend_unavailable"
        assert outcome.terminal_state == "CLEANED"

    def test_identity_mismatch_never_returns_a_success(self):
        backend = DummyServingBackend(
            identity=frozen_identity(),
            script=DummyScript(fault=DummyFault("identity_mismatch")),
        )
        gateway, _ = build_gateway(backend=backend)
        outcome = run(gateway, chat_body())
        assert outcome.status != 200
        assert outcome.code == "backend_identity_mismatch"
        assert outcome.status == 502
        # the wrong-model answer must never reach the client as a 200 body
        assert b"tok11" not in outcome.body

    def test_client_abort_propagates_cancel_to_the_backend(self):
        gateway, backend = build_gateway()
        outcome = run(gateway, chat_body(stream=True), behavior="abrupt_reset")
        assert backend.cancel_records, "disconnect must reach the Backend"
        assert outcome.ledger.disconnected or outcome.ledger.cancelled
        assert outcome.ledger.audit()["ok"]
        states = [record.new_state for record in outcome.transitions]
        assert "CANCEL_PROPAGATED" in states
        assert states[-1] == "CLEANED"

    def test_never_read_after_headers_is_bounded(self):
        gateway, backend = build_gateway()
        outcome = run(gateway, chat_body(stream=True), behavior="never_read_after_headers")
        assert backend.open_count == 1
        assert outcome.ledger.generated <= 3  # the fixture only has three tokens
        assert outcome.ledger.audit()["ok"]

    def test_explicit_cancel_is_propagated(self):
        gateway, backend = build_gateway()
        original_open = backend.open

        def open_and_cancel(request, *, deadline_ns):
            session = original_open(request, deadline_ns=deadline_ns)
            gateway.request_cancel(request.request_id, reason="client_cancelled")
            return session

        backend.open = open_and_cancel  # type: ignore[assignment]
        outcome = run(gateway, chat_body(stream=True))
        assert backend.cancel_records
        assert outcome.code == "client_cancelled"
        assert outcome.service.cancel_propagated is True


@pytest.mark.unit
class TestLifecycle:
    def test_drain_marks_readiness_false_and_rejects_new_business(self):
        gateway, _ = build_gateway()
        record = gateway.begin_drain(monotonic_ns=1000)
        assert record["state"] == "DRAINING"
        assert gateway.readiness()["ready"] is False
        outcome = run(gateway, chat_body())
        assert outcome.status == 503
        assert outcome.code == "service_not_serving"
        assert gateway.close()["idempotent"] is False
        assert gateway.close()["idempotent"] is True

    def test_health_still_answers_while_draining(self):
        gateway, _ = build_gateway()
        gateway.begin_drain(monotonic_ns=1000)
        outcome = run(gateway, b"", path="/healthz", method="GET")
        assert outcome.status == 200
        assert json.loads(outcome.body)["state"] == "DRAINING"

    def test_close_without_drain_is_refused(self):
        gateway, _ = build_gateway()
        with pytest.raises(SchemaError):
            gateway.close()

    def test_force_cancel_requires_draining(self):
        gateway, _ = build_gateway()
        with pytest.raises(SchemaError):
            gateway.force_cancel(monotonic_ns=1)

    def test_gateway_requires_a_backend(self):
        with pytest.raises(ConfigError):
            build_gateway(backends={})

    def test_counters_and_metrics_exposition(self):
        gateway, _ = build_gateway()
        run(gateway, chat_body())
        run(gateway, b"{not json")
        counters = gateway.counters()
        assert counters["requests_received"] == 2
        assert counters.get("rejected_invalid_json") == 1
        exposition = gateway.metrics_exposition()
        assert "hqsb_service_state" in exposition


@pytest.mark.unit
class TestLedgerAndTiming:
    def test_delivery_ledger_ordering_is_enforced(self):
        from hqsb.serving.pipeline import DeliveryLedger

        ledger = DeliveryLedger(request_id="r0", generated=3, committed=5)
        assert not ledger.audit()["ok"]
        ledger = DeliveryLedger(request_id="r0", generated=5, committed=4, emitted=4, flushed=4)
        ledger.client_received = 4
        ledger.client_received_observed = True
        ledger.terminal_clean = True
        assert ledger.audit()["ok"], ledger.audit()["problems"]

    def test_disconnect_shortfall_must_be_attributed(self):
        from hqsb.serving.pipeline import DeliveryLedger

        ledger = DeliveryLedger(
            request_id="r0", generated=5, committed=4, emitted=2, flushed=2
        )
        ledger.client_received = 1
        ledger.client_received_observed = True
        ledger.disconnected = True
        ledger.discarded_after_disconnect = 3
        assert ledger.audit()["ok"], ledger.audit()["problems"]
        ledger.discarded_after_disconnect = 0
        assert not ledger.audit()["ok"]

    def test_timestamp_ledger_ttfts_are_prefixed(self):
        from hqsb.serving.timing import TimestampLedger, require_prefixed_ttft

        ledger = TimestampLedger(request_id="r0")
        ledger.record("t_send", 0)
        ledger.record("t_gateway_recv", 1_000_000)
        ledger.record("t_enqueue", 2_000_000)
        ledger.record("t_dequeue", 3_000_000)
        ledger.record("t_backend_submit", 4_000_000)
        ledger.record("t_runtime_first", 9_000_000)
        ledger.record("t_first_frame_write", 10_000_000)
        ledger.record("t_first_byte_client", 11_000_000)
        report = ledger.ttft_report()
        assert report["runtime_TTFT_ms"] == pytest.approx(5.0)
        assert report["server_TTFT_ms"] == pytest.approx(9.0)
        assert report["client_TTFT_ms"] == pytest.approx(11.0)
        assert ledger.monotonicity_problems() == []
        with pytest.raises(ConfigError):
            require_prefixed_ttft("ttft")

    def test_duplicate_timestamp_is_refused(self):
        from hqsb.serving.timing import TimestampLedger

        ledger = TimestampLedger(request_id="r0")
        ledger.record("t_send", 1)
        with pytest.raises(ConfigError):
            ledger.record("t_send", 2)

    def test_cross_clock_subtraction_is_refused_without_calibration(self):
        from hqsb.serving.timing import TimestampLedger

        ledger = TimestampLedger(request_id="r0")
        ledger.record("t_send", 1, clock="loadgen_monotonic")
        ledger.record("t_gateway_recv", 2, clock="gateway_monotonic")
        with pytest.raises(ConfigError):
            ledger.duration_ns("ingress")
