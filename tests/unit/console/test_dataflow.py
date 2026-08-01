"""Evidence projection tests; synthetic trace rows are not hardware results."""

from copy import deepcopy
import json

import pytest

from hqsb.console.dataflow import MAX_COPY_DETAILS, build_dataflow


RUN = {
    "id": "run_" + "a" * 32,
    "metrics": {
        "observation": {
            "profile": {
                "activities": ["CPU", "CUDA"],
                "coverage": {
                    "prefill": True,
                    "decode_steps": 8,
                    "total_output_tokens": 40,
                    "scope": "first_prefill_and_bounded_decode_window",
                },
            }
        }
    },
}
TRACE = {
    "status": "available",
    "excluded_or_truncated_events": 0,
    "sha256": "test-fixture-hash-not-hardware-evidence",
    "clock_domain": "profiler_trace_relative",
}


def activity(name="Memcpy HtoD (Pageable -> Device)", **overrides):
    return {
        "id": 1,
        "name": name,
        "category": "gpu_memcpy",
        "start_ms": 2.0,
        "duration_ms": 0.4,
        "args": {"bytes": 4096, "correlation": 41, "stream": 7, "device": 0},
        **overrides,
    }


def test_copy_direction_volume_and_duration_are_direct_evidence():
    run = deepcopy(RUN)
    rows = [
        activity(),
        activity("Memcpy HtoD", duration_ms=0.6, args={"bytes": 1024}),
        activity("Memcpy DtoH", args={"bytes": 8}),
        activity("Memcpy DtoD", args={"bytes": 512}),
        activity("Memcpy PtoP", args={"bytes": 128}),
        activity("attention_kernel", category="kernel"),
    ]
    original = deepcopy(rows)
    result = build_dataflow(run, iter(rows), TRACE)
    edges = {edge["direction"]: edge for edge in result["edges"]}
    assert edges["host_to_device"]["bytes"] == 5120
    assert edges["host_to_device"]["duration_ms_sum"] == pytest.approx(1.0)
    assert edges["host_to_device"]["count"] == 2
    assert edges["device_to_host"]["bytes"] == 8
    assert edges["device_to_device"]["source"] == edges["device_to_device"]["target"]
    assert edges["peer_to_peer"]["target"] == "peer_device"
    assert result["totals"]["copy_events"] == 5
    assert result["totals"]["kernel_events"] == 1
    assert result["totals"]["bytes"] == 5768
    assert result["copy_events"][0]["correlation"] == 41
    assert result["coverage"]["profile_window"]["total_output_tokens"] == 40
    assert result["coverage"]["retained_events"] == 6
    assert result["coverage"]["trace_sha256"] == TRACE["sha256"]
    assert result["status"] == "available"
    assert run == RUN and rows == original


def test_cpu_api_and_framework_copy_are_not_double_counted():
    rows = [
        activity(),
        activity("cudaMemcpyAsync", category="cuda_runtime"),
        activity("cuMemcpyHtoDAsync", category="cuda_driver"),
        activity("aten::copy_", category="cpu_op"),
        activity("Memcpy HtoD", category="user_annotation"),
        activity("Memcpy HtoD", category="python_function"),
        activity("Memcpy copy_kernel", category="kernel"),
    ]
    result = build_dataflow(RUN, rows, TRACE)
    assert result["totals"]["copy_events"] == 1
    assert result["totals"]["kernel_events"] == 1
    assert result["totals"]["bytes"] == 4096


def test_missing_bytes_remain_unknown_while_preserving_known_subset():
    result = build_dataflow(
        RUN,
        [activity(), activity(args={"device": 0}), activity(args={"bytes": 0})],
        TRACE,
    )
    edge = result["edges"][0]
    assert edge["bytes"] is None
    assert edge["known_bytes"] == 4096
    assert edge["bytes_known_events"] == 2
    assert edge["count"] == 3
    assert result["totals"]["bytes_complete"] is False
    assert result["status"] == "partial"
    assert result["copy_events"][1]["bytes"] is None
    assert result["copy_events"][2]["bytes"] == 0


@pytest.mark.parametrize(
    "value", [True, -1, 2.5, "4096", float("nan"), float("inf"), 2**60]
)
def test_invalid_or_unrepresentable_bytes_are_not_counted(value):
    result = build_dataflow(RUN, [activity(args={"bytes": value})], TRACE)
    assert result["totals"]["bytes"] is None
    assert result["totals"]["known_bytes"] is None
    assert result["totals"]["bytes_known_events"] == 0
    json.dumps(result, allow_nan=False)


def test_ambiguous_direction_is_not_inferred_from_device_or_stream():
    result = build_dataflow(
        RUN,
        [activity("Memcpy Unknown", args={"bytes": 32, "device": 0, "stream": 7})],
        TRACE,
    )
    assert result["edges"][0]["direction"] == "unknown"
    assert result["edges"][0]["source"] == "unknown"
    assert result["edges"][0]["target"] == "unknown"
    assert result["totals"]["bytes"] == 32


def test_alternate_names_and_missing_categories_are_supported_conservatively():
    rows = [
        activity("Memcpy Host to Device", category=""),
        activity("memcpy DEVICE -> HOST", category="cuda_memcpy"),
        activity("Memcpy HtoH", category="memcpy"),
        activity("cudaMemcpyAsync", category=""),
    ]
    result = build_dataflow(RUN, rows, TRACE)
    assert {edge["direction"] for edge in result["edges"]} == {
        "host_to_device",
        "device_to_host",
        "host_to_host",
    }
    assert result["totals"]["copy_events"] == 3


def test_detail_cap_does_not_truncate_aggregate_or_consume_iterable_twice():
    count = MAX_COPY_DETAILS + 17
    result = build_dataflow(
        RUN, (activity(id=index, args={"bytes": 8}) for index in range(count)), TRACE
    )
    assert len(result["copy_events"]) == MAX_COPY_DETAILS
    assert result["copy_events_truncated"] == 17
    assert result["totals"]["copy_events"] == count
    assert result["totals"]["bytes"] == count * 8


@pytest.mark.parametrize(
    "status", ["not_collected", "invalid", "download_only", "partial"]
)
def test_trace_unavailability_and_truncation_are_not_disguised(status):
    result = build_dataflow({}, [], {"status": status})
    assert result["status"] == status
    assert result["totals"]["bytes"] is None
    assert result["totals"]["known_bytes"] is None
    assert result["totals"]["duration_ms_sum"] is None
    assert result["coverage"]["bytes_complete"] is False
    assert result["coverage"]["excluded_or_truncated_events"] is None
    assert result["coverage"]["profile_window"] == {}


def test_no_copy_events_does_not_claim_zero_physical_memory_traffic():
    result = build_dataflow(RUN, [activity("gemm", category="kernel")], TRACE)
    assert result["status"] == "no_copy_events"
    assert result["totals"]["copy_events"] == 0
    assert result["totals"]["kernel_events"] == 1
    assert result["totals"]["bytes"] is None
    assert result["edges"] == []
    assert "未观测到不代表未发生" in " ".join(result["limitations"])


def test_nonfinite_duration_and_unneeded_physical_addresses_are_not_exposed():
    result = build_dataflow(
        RUN,
        [
            activity(
                duration_ms=float("inf"),
                start_ms=float("nan"),
                args={"bytes": 16, "src_address": "0x1234", "dst_address": "0x5678"},
            ),
            activity(duration_ms=-1),
        ],
        TRACE,
    )
    assert result["totals"]["duration_ms_sum"] is None
    assert result["totals"]["duration_known_events"] == 0
    assert result["copy_events"][0]["start_ms"] is None
    encoded = json.dumps(result, allow_nan=False)
    assert "0x1234" not in encoded and "0x5678" not in encoded


def test_overflowing_aggregate_never_claims_exact_json_byte_count():
    rows = [activity(args={"bytes": 2**52}) for _ in range(2)]
    result = build_dataflow(RUN, rows, TRACE)
    assert result["totals"]["bytes_known_events"] == 2
    assert result["totals"]["bytes"] is None
    assert result["totals"]["known_bytes"] is None
    assert result["totals"]["bytes_complete"] is False
    assert result["status"] == "partial"
