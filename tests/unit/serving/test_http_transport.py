"""Real stdlib HTTP/SSE binding smoke test (labelled smoke, not an experiment)."""

from __future__ import annotations

import json
import os
import urllib.request

import pytest
import yaml

from hqsb.serving import protocol as protocol_mod
from hqsb.serving.dummy_backend import (
    DummyServingBackend,
    frozen_identity,
    template_identity,
    tokenizer_for_vocab,
)
from hqsb.serving.gateway import GatewayConfig, ServingGateway
from hqsb.serving.transport_http import StdlibHttpTransport

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG = os.path.join(_REPO_ROOT, "configs", "serving")


def _gateway():
    with open(os.path.join(_CONFIG, "protocol_profile.yaml"), encoding="utf-8") as handle:
        profile = protocol_mod.ProtocolProfile.from_document(yaml.safe_load(handle))
    with open(os.path.join(_CONFIG, "error_catalog.yaml"), encoding="utf-8") as handle:
        catalog = protocol_mod.ErrorCatalog.from_document(yaml.safe_load(handle))
    backend = DummyServingBackend(identity=frozen_identity())
    return ServingGateway(
        profile=profile,
        catalog=catalog,
        model_registry={"dummy-model": frozen_identity()},
        tokenizer=tokenizer_for_vocab(),
        chat_template=template_identity,
        backends={backend.instance_id: backend},
        config=GatewayConfig(),
    ), backend


@pytest.mark.unit
class TestHttpTransport:
    def test_end_to_end_non_stream_and_health(self):
        gateway, backend = _gateway()
        transport = StdlibHttpTransport(gateway.handle)
        base = transport.start()
        try:
            with urllib.request.urlopen(base + "/healthz", timeout=5) as response:
                health = json.loads(response.read())
            assert response.status == 200 and health["state"] == "SERVING"

            payload = json.dumps(
                {"model": "dummy-model", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read())
            assert response.status == 200
            assert body["choices"][0]["finish_reason"] == "stop"
            assert backend.open_count == 1
        finally:
            transport.stop()

    def test_invalid_request_maps_to_400(self):
        gateway, _ = _gateway()
        transport = StdlibHttpTransport(gateway.handle)
        base = transport.start()
        try:
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=b"{not json",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(request, timeout=5)
            assert excinfo.value.code == 400
            payload = json.loads(excinfo.value.read())
            assert payload["error"]["code"] == "invalid_json"
        finally:
            transport.stop()
