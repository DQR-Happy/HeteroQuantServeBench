"""Unit tests for the E02-07 multi-level profiling correlation logic.

Every synthetic trace is built by hand so the expected phase, operator and
kernel for each event is known before the code under test sees it. The tests
target exactly the failure modes the E02-07 protocol lists as anti-examples:

* a kernel attributed to the wrong phase because attribution went through the
  launch API instead of the owning operator;
* a same-named kernel invoked with several shapes averaged into one row;
* kernel durations summed as if streams never overlapped;
* an unattributed kernel silently pushed into a real phase;
* an NCU metric that is ``n/a`` on this platform reported as ``0``;
* theoretical FLOPs, modeled bytes and measured L2 bytes conflated.
"""

from __future__ import annotations

import json

import pytest

from hqsb.benchmark import multilevel_profiling as mp


# ───────────────────────────── synthetic trace ───────────────────────────


def _event(name, cat, ts, dur, *, pid=1, tid=1, **args):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def _synthetic_trace():
    """A two-phase trace with a nested module role and one unattributed op.

    Layout (microseconds, same clock for CPU and device rows)::

        e02_07_run            [0, 1000]
          e02_07_prefill      [10, 100]
            role.q_proj       [20, 60]
          e02_07_decode_early [200, 700]
            role.q_proj       [220, 250]
              (a second invocation at [400, 430])
        (a stray kernel at [900, 950] belonging to no phase)
    """
    events = [
        _event(mp.RUN_RANGE, "user_annotation", 0, 1000),
        _event(mp.PREFILL_RANGE, "user_annotation", 10, 90),
        _event(f"{mp.MODULE_ROLE_PREFIX}q_proj", "user_annotation", 20, 40),
        _event(mp.DECODE_EARLY_RANGE, "user_annotation", 200, 500),
        _event(f"{mp.MODULE_ROLE_PREFIX}q_proj", "user_annotation", 220, 30),
        _event(f"{mp.MODULE_ROLE_PREFIX}q_proj", "user_annotation", 400, 30),
        # Host operators: the owning ATen op, the launch API it used, and the
        # record-function id that links them to the device kernel.
        _event("aten::mm", "cpu_op", 21, 38, **{"External id": 1001,
                                                "Input Dims": [[1, 2048], [2048, 2048]]}),
        _event("cudaLaunchKernel", "cuda_runtime", 30, 2,
               **{"External id": 1001, "correlation": 9001}),
        _event("aten::mm", "cpu_op", 221, 28, **{"External id": 1002,
                                                 "Input Dims": [[1, 2048], [2048, 6144]]}),
        _event("cudaLaunchKernel", "cuda_runtime", 230, 2,
               **{"External id": 1002, "correlation": 9002}),
        _event("aten::mm", "cpu_op", 401, 28, **{"External id": 1003,
                                                 "Input Dims": [[1, 2048], [2048, 6144]]}),
        _event("cudaLaunchKernel", "cuda_runtime", 410, 2,
               **{"External id": 1003, "correlation": 9003}),
        _event("aten::copy_", "cpu_op", 890, 40, **{"External id": 1004,
                                                    "Input Dims": [[1, 1, 128], [1, 1, 128]]}),
        _event("cudaLaunchKernel", "cuda_runtime", 895, 2,
               **{"External id": 1004, "correlation": 9004}),
        # Device kernels.
        _event("gemm_prefill_kernel", "kernel", 40, 50, stream=7,
               **{"correlation": 9001, "External id": 1001,
                  "grid": [8, 1, 1], "block": [128, 1, 1],
                  "registers per thread": 200, "shared memory": 49152}),
        _event("sliced_gemm_kernel", "kernel", 240, 20, stream=7,
               **{"correlation": 9002, "External id": 1002,
                  "grid": [96, 1, 1], "block": [128, 1, 1]}),
        _event("sliced_gemm_kernel", "kernel", 420, 30, stream=7,
               **{"correlation": 9003, "External id": 1003,
                  "grid": [24, 1, 1], "block": [128, 1, 1]}),
        _event("stray_kernel", "kernel", 900, 50, stream=9,
               **{"correlation": 9004, "External id": 1004}),
    ]
    return {"traceEvents": events}


# ───────────────────────────── phase ledger ──────────────────────────────


def test_phase_ledger_partitions_every_decode_step_once():
    ledger = mp.phase_ledger(input_len=128, output_tokens=32, early_steps=4, late_steps=4)
    assert ledger["decode_early"]["steps"] == [1, 2, 3, 4]
    assert ledger["decode_late"]["steps"] == [28, 29, 30, 31]
    assert ledger["decode_middle"]["steps"] == list(range(5, 28))
    assert ledger["phases_partition_decode"] is True
    assert ledger["model_forward_passes_total"] == 1 + 4 + 23 + 4


def test_phase_ledger_context_lengths_start_after_prefill():
    ledger = mp.phase_ledger(input_len=128, output_tokens=8, early_steps=2, late_steps=2)
    # 7 decode steps, so "late" is steps 6 and 7 -> contexts 134 and 135.
    assert ledger["decode_early"]["context_len"] == [129, 130]
    assert ledger["decode_late"]["context_len"] == [134, 135]


def test_phase_ledger_rejects_windows_larger_than_the_generation():
    with pytest.raises(ValueError, match="exceeds"):
        mp.phase_ledger(input_len=128, output_tokens=4, early_steps=2, late_steps=2)


def test_phase_ledger_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        mp.phase_ledger(input_len=0, output_tokens=4, early_steps=0, late_steps=0)


# ──────────────────────── attribution through the chain ──────────────────


def test_phase_spans_are_found_and_merged_per_name():
    spans = mp.phase_span_index(_synthetic_trace())
    assert spans[mp.PREFILL_RANGE]["dur"] == 90
    assert spans[mp.PREFILL_RANGE]["instances"] == 1
    assert spans[mp.DECODE_EARLY_RANGE]["dur"] == 500


def test_kernels_are_attributed_to_the_owning_operator_not_the_launch_api():
    attribution = mp.attribute_events_to_phases(_synthetic_trace())
    prefill = attribution["phases"][mp.PREFILL_RANGE]
    decode = attribution["phases"][mp.DECODE_EARLY_RANGE]

    assert prefill["count"] == 1
    assert decode["count"] == 2
    assert prefill["kernels"][0]["op"] == "aten::mm"
    # Both decode rows share the kernel name but have different shapes, so they
    # must stay distinct observations rather than being merged by name.
    dims = sorted(kernel["op_dims"] for kernel in decode["kernels"])
    assert dims == ["[1,2048] x [2048,6144]", "[1,2048] x [2048,6144]"]
    assert prefill["kernels"][0]["op_dims"] == "[1,2048] x [2048,2048]"


def test_unattributed_work_is_kept_in_its_own_bucket():
    attribution = mp.attribute_events_to_phases(_synthetic_trace())
    assert attribution["unattributed"]["count"] == 1
    assert attribution["unattributed"]["names"] == ["stray_kernel"]
    assert attribution["coverage"]["device_events_attributed"] == 3
    assert attribution["coverage"]["device_events_total"] == 4


def test_operator_resolution_reports_the_external_id_channel():
    attribution = mp.attribute_events_to_phases(_synthetic_trace())
    resolution = attribution["coverage"]["operator_resolution"]
    assert resolution["resolved"] == 4
    assert resolution["unresolved"] == 0
    assert resolution["external_ids_indexed"] == 4


def test_host_op_table_counts_operator_invocations_and_device_time():
    table = mp.host_op_table_by_phase(_synthetic_trace())
    prefill_rows = {row["name"]: row for row in table[mp.PREFILL_RANGE]}
    assert prefill_rows["aten::mm"]["kernel_count"] == 1
    assert prefill_rows["aten::mm"]["op_invocations"] == 1
    assert prefill_rows["aten::mm"]["device_us"] == 50
    decode_rows = {row["name"]: row for row in table[mp.DECODE_EARLY_RANGE]}
    assert decode_rows["aten::mm"]["kernel_count"] == 2
    assert decode_rows["aten::mm"]["device_us"] == 50


def test_module_role_table_maps_role_to_op_and_kernel():
    roles = mp.module_role_table(_synthetic_trace())["roles"]
    q_proj = roles[f"{mp.MODULE_ROLE_PREFIX}q_proj"]
    # One prefill invocation plus two decode invocations.
    assert q_proj["op_invocations"] == 3
    assert q_proj["ops"] == {"aten::mm": 3}
    assert q_proj["kernel_count"] == 3
    assert set(q_proj["kernels"]) == {"gemm_prefill_kernel", "sliced_gemm_kernel"}
    assert q_proj["device_us"] == 100


def test_module_role_table_returns_empty_without_annotations():
    trace = {"traceEvents": [_event(mp.PREFILL_RANGE, "user_annotation", 0, 10)]}
    assert mp.module_role_table(trace) == {"roles": {}, "span_count": 0}


# ───────────────────────────── rankings ──────────────────────────────────


def test_aggregate_kernels_separates_shape_signatures():
    kernels = [
        {"name": "k", "dur": 10.0, "grid": [1, 1, 1], "block": [128, 1, 1],
         "op": "aten::mm", "op_dims": "[1,2048] x [2048,2048]"},
        {"name": "k", "dur": 30.0, "grid": [96, 1, 1], "block": [128, 1, 1],
         "op": "aten::mm", "op_dims": "[1,2048] x [2048,6144]"},
    ]
    rows = mp.aggregate_kernels(kernels)
    assert len(rows) == 1
    row = rows[0]
    assert row["count"] == 2
    assert row["total_us"] == 40.0
    assert row["mean_us"] == 20.0
    assert row["min_us"] == 10.0
    assert row["max_us"] == 30.0
    # Two distinct grids and two distinct shapes -> the row is NOT one shape.
    assert row["shape_signature_count"] == 2
    assert row["dims"] == ["[1,2048] x [2048,2048]", "[1,2048] x [2048,6144]"]


def test_attach_shares_and_cumulative_coverage():
    rows = [{"name": "a", "total_us": 50.0}, {"name": "b", "total_us": 30.0},
            {"name": "c", "total_us": 20.0}]
    ranked = mp.attach_shares(rows)
    assert [round(r["time_share"], 4) for r in ranked] == [0.5, 0.3, 0.2]
    assert ranked[-1]["cumulative_share"] == pytest.approx(1.0)
    coverage = mp.cumulative_coverage(ranked, top=2)
    assert coverage["covered_share"] == pytest.approx(0.8)
    assert coverage["residual_share"] == pytest.approx(0.2)


def test_attach_shares_accepts_a_wall_clock_denominator():
    rows = [{"name": "a", "total_us": 50.0}]
    ranked = mp.attach_shares(rows, total_us=200.0)
    assert ranked[0]["time_share"] == 0.25


def test_kernel_buckets_and_summary():
    assert mp.kernel_bucket("ampere_fp16_s16816gemm_fp16_64x64") == "gemm"
    assert mp.kernel_bucket("void cutlass::Kernel2<...>") == "gemm"
    assert mp.kernel_bucket("CatArrayBatchedCopy_contig") == "memory_kv"
    assert mp.kernel_bucket("void at::native::reduce_kernel<...>") == "reduction"
    summary = mp.bucket_summary(
        [
            {"name": "ampere_fp16_s16816gemm_fp16_64x64", "count": 2, "total_us": 60.0},
            {"name": "at::native::silu_kernel", "count": 1, "total_us": 40.0},
        ]
    )
    assert summary[0]["bucket"] == "gemm"
    assert summary[0]["share_of_top_table"] == pytest.approx(0.6)


def test_candidate_selection_prefers_frequent_large_work():
    rows = [
        {"name": "sliced1x2_gemm", "count": 255, "total_us": 277.0, "mean_us": 1.09},
        {"name": "elementwise_kernel", "count": 10, "total_us": 5.0, "mean_us": 0.5},
    ]
    ranked = mp.attach_shares(rows, field="total_us")
    selected = mp.candidate_selection(
        phase=mp.DECODE_EARLY_RANGE, rows=ranked, span_us=400.0, max_candidates=1
    )
    assert len(selected) == 1
    assert selected[0]["name"] == "sliced1x2_gemm"
    assert selected[0]["bucket"] == "gemm"
    assert selected[0]["wall_share_bound"] == pytest.approx(277.0 / 400.0)


# ───────────────────────────── timeline ──────────────────────────────────


def test_merge_intervals_joins_overlapping_ranges():
    assert mp.merge_intervals([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]


def test_gpu_activity_window_reports_overlap_and_idle():
    kernels = [
        {"ts": 0, "dur": 100, "stream": 7},
        {"ts": 50, "dur": 100, "stream": 8},
        {"ts": 300, "dur": 100, "stream": 7},
    ]
    window = mp.gpu_activity_window(kernels)
    assert window["span_us"] == 400
    # The first two kernels overlap, so the busy union is [0,150]+[300,400].
    assert window["busy_us"] == 250
    assert window["idle_us"] == 150
    assert window["idle_ratio"] == pytest.approx(0.375)
    # Durations sum to 300 over a 400 span, but 50 us of it is overlapped work.
    assert window["sum_of_durations_us"] == 300
    assert window["overlap_factor"] == pytest.approx(0.75)


def test_gpu_activity_window_handles_no_kernels():
    window = mp.gpu_activity_window([])
    assert window["kernel_count"] == 0
    assert window["idle_ratio"] == 0.0


def test_timeline_gaps_stay_on_one_stream():
    kernels = [
        {"ts": 0, "dur": 10, "stream": 7},
        {"ts": 100, "dur": 10, "stream": 7},
        {"ts": 20, "dur": 10, "stream": 8},
    ]
    gaps = mp.timeline_gaps(kernels, stream=7)
    assert gaps["gap_count"] == 1
    assert gaps["total_gap_us"] == 90
    assert gaps["max_gap_us"] == 90
    # Stream 8 is the second busiest and must not contribute to stream 7's gap.
    assert mp.timeline_gaps(kernels, stream=8)["gap_count"] == 0


def test_stream_breakdown_sorts_by_device_time():
    breakdown = mp.stream_breakdown(
        [{"ts": 0, "dur": 10, "stream": 7}, {"ts": 0, "dur": 90, "stream": 9}]
    )
    assert breakdown[0]["stream"] == 9
    assert breakdown[0]["total_us"] == 90


def test_cpu_to_kernel_latency_uses_the_launch_row():
    latency = mp.cpu_to_kernel_latency(_synthetic_trace())
    # launch end -> kernel start: 32->40 = 8, 232->240 = 8, 412->420 = 8,
    # 897->900 = 3 microseconds.
    assert latency["samples"] == 4
    assert latency["min_us"] == 3
    assert latency["max_us"] == 8
    assert latency["p50_us"] == pytest.approx(8.0)


def test_critical_path_note_flags_overlap():
    note = mp.critical_path_share_note(kernel_work_us=1500.0, span_us=1000.0)
    assert note["overlaps"] is True
    assert note["overlap_factor"] == pytest.approx(1.5)
    assert note["wall_clock_share"] == 1.0
    non_overlapping = mp.critical_path_share_note(kernel_work_us=400.0, span_us=1000.0)
    assert non_overlapping["overlaps"] is False
    assert non_overlapping["wall_clock_share"] == pytest.approx(0.4)


# ───────────────────────────── NCU parsing ───────────────────────────────


_NCU_SECTION_CSV = "\n".join(
    [
        "==WARNING== Note: Running with unmodified GPU clocks.",
        '"ID","Process ID","Kernel Name","Grid Size","Block Size","Context",'
        '"Stream","Section Name","Metric Name","Metric Unit","Metric Value"',
        '"0","1","gemm_kernel","(96, 1, 1)","(128, 1, 1)","1","7",'
        '"GPU Speed Of Light Throughput","Duration","ns","853,984"',
        '"0","1","gemm_kernel","(96, 1, 1)","(128, 1, 1)","1","7",'
        '"GPU Speed Of Light Throughput","Memory Throughput","%","87.54"',
        '"0","1","gemm_kernel","(96, 1, 1)","(128, 1, 1)","1","7",'
        '"GPU Speed Of Light Throughput","Compute (SM) Throughput","%","37.35"',
        '"0","1","gemm_kernel","(96, 1, 1)","(128, 1, 1)","1","7",'
        '"Occupancy","Achieved Occupancy","%","16.33"',
        '"0","1","gemm_kernel","(96, 1, 1)","(128, 1, 1)","1","7",'
        '"Launch Statistics","Registers Per Thread","register/thr","128"',
        '"0","1","gemm_kernel","(96, 1, 1)","(128, 1, 1)","1","7",'
        '"Memory Workload Analysis","Mem Busy","%","n/a"',
    ]
)

_NCU_LAUNCH_CSV = "\n".join(
    [
        '"Process ID","Kernel Name","Block Size","Grid Size","Invocations",'
        '"Section Name","Metric Name","Metric Unit","Minimum","Maximum","Average"',
        '"1","gemm_kernel","(128, 1, 1)","(96, 1, 1)","3",'
        '"Command line profiler metrics","sm__throughput.avg.pct_of_peak_sustained_elapsed",'
        '"%","10.00","20.00","15.00"',
        '"1","gemm_kernel","(128, 1, 1)","(96, 1, 1)","3",'
        '"Command line profiler metrics","lts__t_bytes.sum","byte","1,000.00","2,000.00","1,500.00"',
    ]
)


def test_parse_ncu_csv_skips_preamble_and_keeps_the_grid_fingerprint():
    parsed = mp.parse_ncu_csv(_NCU_SECTION_CSV)
    assert len(parsed["kernels"]) == 1
    kernel = parsed["kernels"][0]
    assert kernel["name"] == "gemm_kernel"
    assert kernel["grid"] == "(96, 1, 1)"
    assert kernel["block"] == "(128, 1, 1)"
    assert kernel["invocations"] == 1
    # NCU prints Duration in nanoseconds; reading it as microseconds would
    # understate every derived rate by 1000x.
    assert kernel["panel"]["duration_ns"] == 853984
    assert kernel["panel"]["duration_ms"] == pytest.approx(0.853984)
    assert kernel["panel"]["memory_throughput_pct"] == pytest.approx(87.54)
    assert kernel["panel"]["compute_sm_throughput_pct"] == pytest.approx(37.35)
    assert kernel["panel"]["achieved_occupancy_pct"] == pytest.approx(16.33)
    assert kernel["panel"]["registers_per_thread"] == pytest.approx(128)


def test_parse_ncu_csv_reports_unavailable_metrics_rather_than_zero():
    parsed = mp.parse_ncu_csv(_NCU_SECTION_CSV)
    kernel = parsed["kernels"][0]
    # Mem Busy is n/a on this platform: it must stay None, not become 0.0.
    assert kernel["panel"]["mem_busy_pct"] is None
    assert "Mem Busy" in parsed["unavailable_metrics"]


def test_parse_ncu_csv_accepts_the_per_launch_layout():
    parsed = mp.parse_ncu_csv(_NCU_LAUNCH_CSV)
    kernel = parsed["kernels"][0]
    assert kernel["invocations"] == 3
    assert kernel["metrics"]["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["value"] == 15.0
    assert kernel["metrics"]["lts__t_bytes.sum"]["value"] == 1500.0


def test_parse_ncu_csv_without_a_header_returns_an_error():
    assert mp.parse_ncu_csv("no table here\n").get("error")


def test_ncu_stall_metrics_scans_by_name_prefix():
    metrics = {
        "Stall Long Scoreboard": {"value": 5.0, "unit": "cycle", "section": "WarpStateStats"},
        "Stall Barrier": {"value": 3.0, "unit": "cycle", "section": "WarpStateStats"},
        "Duration": {"value": 100.0, "unit": "ns", "section": "SpeedOfLight"},
    }
    stalls = mp.ncu_stall_metrics(metrics)
    assert stalls == {"Stall Long Scoreboard": 5.0, "Stall Barrier": 3.0}


# ───────────────────────────── roofline ──────────────────────────────────


def test_kernel_shape_flops_matches_the_hand_computed_gemm():
    info = mp.kernel_shape_flops(m=1, n=6144, k=2048)
    assert info["flops"] == 2 * 1 * 6144 * 2048
    assert info["min_bytes"] == 2 * (1 * 2048 + 2048 * 6144 + 1 * 6144)


def test_roofline_consistency_keeps_the_three_quantities_separate():
    result = mp.roofline_consistency(
        useful_flops=2 * 1 * 6144 * 2048,
        modeled_dram_bytes=2 * (1 * 2048 + 2048 * 6144),
        measured_l2_bytes=209_792_928.0,
        duration_us=853.984,
        peak_flops=67e12,
        peak_bandwidth=68e9,
    )
    assert result["useful_flops"] == 2 * 1 * 6144 * 2048
    assert result["measured_l2_bytes"] == 209_792_928.0
    # The decode GEMM is tiny and bandwidth-bound with a nominal envelope.
    assert result["classification"] == "bandwidth_bound"
    assert result["ceiling_source"] == "nominal_datasheet"
    assert result["caveats"]


def test_roofline_consistency_rejects_a_zero_duration():
    assert "error" in mp.roofline_consistency(
        useful_flops=1.0,
        modeled_dram_bytes=1.0,
        measured_l2_bytes=None,
        duration_us=0.0,
        peak_flops=1.0,
        peak_bandwidth=1.0,
    )


# ───────────────────────────── tool compatibility ────────────────────────


def test_tool_compatibility_compares_identity_not_durations():
    result = mp.tool_compatibility(
        profiler_kernels=[{"name": "gemm"}, {"name": "silu"}],
        nsys_kernels=[{"name": "gemm"}, {"name": "silu"}, {"name": "memcpy"}],
        ncu_kernel_names=["gemm"],
        token_hashes={"p": "abc", "n": "abc"},
    )
    assert result["profiler_missing_in_nsys"] == []
    assert result["nsys_missing_in_profiler"] == ["memcpy"]
    assert result["ncu_names_not_in_profiler"] == []
    assert result["token_hashes_identical"] is True
    # Durations are explicitly not comparable across tools.
    assert result["durations_comparable"] is False


def test_tool_compatibility_detects_divergent_tokens():
    result = mp.tool_compatibility(
        profiler_kernels=[],
        nsys_kernels=[],
        ncu_kernel_names=[],
        token_hashes={"a": "one", "b": "two"},
    )
    assert result["token_hashes_identical"] is False


# ───────────────────────────── trace loading ─────────────────────────────


def test_load_chrome_trace_accepts_a_bare_event_list(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps([_event("x", "kernel", 0, 1)]), encoding="utf-8")
    trace = mp.load_chrome_trace(str(path))
    assert len(mp.trace_events(trace)) == 1


def test_load_chrome_trace_rejects_a_payload_without_events(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        mp.load_chrome_trace(str(path))


def test_trace_events_ignores_metadata_rows():
    trace = {"traceEvents": [{"ph": "M", "name": "process_name"}, _event("k", "kernel", 0, 1)]}
    assert [e["name"] for e in mp.trace_events(trace)] == ["k"]


def test_external_id_index_ignores_launch_rows():
    index = mp.external_id_index(_synthetic_trace())
    assert index[1001]["name"] == "aten::mm"
    assert index[1001]["input_dims"] == [[1, 2048], [2048, 2048]]
    assert set(index) == {1001, 1002, 1003, 1004}


def test_dims_signature_handles_nested_cat_dims():
    signature = mp._dims_signature([[1, 8, 128, 128], [1, 8, 1, 128]])
    assert signature == "[1,8,128,128] x [1,8,1,128]"
    nested = mp._dims_signature([[[1, 8, 128, 128], [1, 8, 1, 128]]])
    assert nested == "{[1,8,128,128] x [1,8,1,128]}"
    assert mp._dims_signature(None) is None
    assert mp._dims_signature([]) is None
