"""Contract checks for the remote provider; no claim about a physical remote GPU."""

import threading
from contextlib import contextmanager

import httpx
import pytest

from hqsb.backends.interactive import OpenAIProvider


@pytest.fixture
def provider():
    return OpenAIProvider(
        {"model": "declared-model", "base_url": "http://configured.invalid/v1"}
    )


def mock_stream(monkeypatch, content):
    @contextmanager
    def stream(*args, **kwargs):
        assert args[1] == "http://configured.invalid/v1/chat/completions"
        assert kwargs["json"]["temperature"] == 0
        yield httpx.Response(200, text=content, request=httpx.Request("POST", args[1]))

    monkeypatch.setattr(httpx, "stream", stream)


def request():
    return {
        "messages": [{"role": "user", "content": "test"}],
        "max_output_tokens": 8,
        "remaining_ms": 5000,
    }


def test_remote_usage_is_not_number_of_network_chunks(provider, monkeypatch):
    mock_stream(
        monkeypatch,
        'data: {"choices":[{"delta":{"content":"你好，世界"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":4,"prompt_tokens":9}}\n\ndata: [DONE]\n\n',
    )
    rows = list(provider.generate(request(), threading.Event()))
    assert rows[0]["output_tokens"] is None
    assert rows[-1]["metrics"]["output_tokens"] == 4
    assert rows[-1]["metrics"]["runtime_first_token_ms"] is None
    assert rows[-1]["metrics"]["remote_cleanup"] == "not_observable"


def test_upstream_eof_without_terminal_is_failure(provider, monkeypatch):
    mock_stream(monkeypatch, 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
    with pytest.raises(RuntimeError, match="terminal"):
        list(provider.generate(request(), threading.Event()))


def test_cancel_closes_stream_and_reports_unknown_usage(provider, monkeypatch):
    mock_stream(monkeypatch, 'data: {"choices":[{"delta":{"content":"late"}}]}\n\n')
    event = threading.Event()
    event.set()
    rows = list(provider.generate(request(), event))
    assert rows[-1]["finish_reason"] == "cancelled"
    assert rows[-1]["metrics"]["output_tokens"] is None
