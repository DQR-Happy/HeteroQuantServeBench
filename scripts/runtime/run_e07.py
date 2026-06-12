#!/usr/bin/env python3
"""S07 experiment driver / interface entry point.

Modes
-----
``status`` (default)
    Print the prerequisite report and each experiment's protocol status.
    Writes nothing unless ``--output-dir`` is given, in which case a
    ``status.json`` (never a verdict) is recorded.
``preregister``
    Write ``preregistration.json`` from the shipped template with a frozen hash,
    plus the empty run layout.
``interface-map``
    Print (or write) the experiment-step → interface table and verify that every
    referenced symbol imports.
``self-check``
    Run the *smoke-level* interface self-checks, labelled as smoke.  These are
    not experiment results: they only prove that the interfaces are callable and
    that the negative paths are refused.
``execute``
    Refused unless ``--confirm-execute`` is passed **and** the prerequisite chain
    is satisfied.  This repository state does not satisfy it, so the mode exits
    non-zero with the missing evidence listed — by design.

Exit codes follow :class:`hqsb.core.errors.ExitCode`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.core.errors import ExitCode, UsageError, exit_code_for  # noqa: E402
from hqsb.runtime import adapter, comparison, experiment, failure, graph_route  # noqa: E402
from hqsb.runtime import interface_map as imap  # noqa: E402
from hqsb.runtime import kv, metrics, parity, policy_ab, prefix_cache  # noqa: E402
from hqsb.runtime import request as request_mod  # noqa: E402
from hqsb.runtime import scheduler, spec_decode, specs, telemetry, trace  # noqa: E402

CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "runtime")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "S07 runtime experiment driver. The default mode only inspects "
            "prerequisites and interfaces; no conclusion can be produced without "
            "an explicit, prerequisite-gated execution."
        )
    )
    parser.add_argument(
        "--experiment",
        default="all",
        help="E07-01 .. E07-10, or 'all' (default).",
    )
    parser.add_argument(
        "--mode",
        default="status",
        choices=["status", "preregister", "interface-map", "self-check", "execute"],
        help="Driver mode (default: status).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Run directory root; empty means 'do not write anything'.",
    )
    parser.add_argument("--run-id", default="", help="Explicit run id (default: UTC timestamp).")
    parser.add_argument(
        "--confirm-execute",
        action="store_true",
        help="Required for --mode execute; an extra guard against accidental runs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--template",
        default="",
        help="Optional JSON file with preregistration fields for --mode preregister.",
    )
    return parser


def _selected_experiments(value: str) -> List[str]:
    if value == "all":
        return list(experiment.EXPERIMENTS)
    if value not in experiment.EXPERIMENTS:
        raise UsageError(
            f"unknown experiment {value!r}; expected one of {list(experiment.EXPERIMENTS)}"
        )
    return [value]


def _preregistration(experiment_id: str, template_path: str) -> experiment.Preregistration:
    payload: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "question": "see docs/stage_experiments/details/S07/<file>",
        "hypothesis": "see docs/stage_experiments/details/S07/<file>",
        "backend_roles": {},
        "runtime_versions": {},
        "model_artifact_hash": "",
        "precision": "",
        "request_trace_hash": "",
        "scheduler_policy": "",
        "kv_policy": "",
        "prefix_policy": "",
        "graph_policy": "",
        "quality_gate": "",
        "performance_metrics": (),
        "repeats": 0,
        "independent_processes": 3,
        "exclusions": (),
        "stop_conditions": (),
        "allowed_claims": (),
        "claim_boundary": (
            "template only: the owner must state what this experiment may NOT be "
            "used to claim before any execution"
        ),
        "hardware": "",
        "notes": (
            "template only; the experiment owner must fill question/hypothesis/"
            "runtime versions/policies/metrics from the protocol document before "
            "any execution"
        ),
    }
    if template_path:
        with open(template_path, encoding="utf-8") as handle:
            payload.update(json.load(handle))
    payload["experiment_id"] = experiment_id
    return experiment.Preregistration(**payload)


def mode_status(
    experiments: Sequence[str], output_dir: str, run_id: str, as_json: bool
) -> Dict[str, Any]:
    prerequisites = experiment.check_prerequisites(REPO_ROOT)
    results: List[Dict[str, Any]] = []
    for experiment_id in experiments:
        entry: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": (
                experiment.STATUS_NOT_STARTED
                if prerequisites.satisfied
                else experiment.STATUS_BLOCKED
            ),
            "prerequisites_satisfied": prerequisites.satisfied,
            "missing": prerequisites.missing,
        }
        if output_dir:
            record = experiment.interface_only_run(REPO_ROOT, experiment_id, run_id or None)
            entry["run_dir"] = record["run_dir"]
            entry["status"] = record["status"]
        results.append(entry)
    report = {
        "stage": experiment.STAGE,
        "mode": "status",
        "prerequisites": prerequisites.as_dict(),
        "experiments": results,
        "note": (
            "no experiment is executed by this mode; a conclusion requires "
            "--mode execute with satisfied prerequisites"
        ),
    }
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"[S07] prerequisite chain satisfied: {prerequisites.satisfied}")
        for check in prerequisites.checks:
            marker = "OK  " if check.satisfied else "MISS"
            print(f"  {marker} {check.name}: {check.reason or check.evidence}")
        for entry in results:
            print(f"  {entry['experiment_id']}: {entry['status']}")
    return report


def mode_preregister(
    experiments: Sequence[str], output_dir: str, run_id: str, template: str, as_json: bool
) -> Dict[str, Any]:
    if not output_dir:
        raise UsageError("--mode preregister requires --output-dir")
    prerequisites = experiment.check_prerequisites(REPO_ROOT)
    written: List[Dict[str, Any]] = []
    for experiment_id in experiments:
        run = experiment.RunDirectory(output_dir, experiment_id, run_id or "prereg")
        run.create()
        prereg = _preregistration(experiment_id, template)
        run.write_preregistration(prereg)
        # The handbook §4 unified record is written as a template: it can never
        # carry a conclusion because the executor must supply the raw evidence.
        record_status = (
            experiment.STATUS_BLOCKED
            if not prerequisites.satisfied
            else experiment.STATUS_NOT_STARTED
        )
        record = experiment.template_experiment_record(
            experiment_id,
            status=record_status,
            reason=(
                "no experiment was executed; the interface layer cannot produce "
                "measurements"
            ),
        )
        run.write_experiment_record(record)
        run.write_json("prerequisites.json", prerequisites.as_dict())
        run.write_json("environment_fingerprint.json", experiment.environment_fingerprint())
        run.write_report_skeleton(experiment_id, imap.mapping_for(experiment_id).title)
        run.write_status(
            record_status,
            "preregistration only; no execution attempted",
            prerequisites,
        )
        written.append(
            {
                "experiment_id": experiment_id,
                "run_dir": run.path,
                "prereg_hash": prereg.prereg_hash,
            }
        )
    report = {"mode": "preregister", "written": written}
    print(
        json.dumps(report, indent=2)
        if as_json
        else "\n".join(
            f"{item['experiment_id']}: prereg_hash={item['prereg_hash'][:16]} "
            f"dir={item['run_dir']}"
            for item in written
        )
    )
    return report


def mode_interface_map(experiments: Sequence[str], output_dir: str, as_json: bool) -> Dict[str, Any]:
    resolved = imap.resolve_interfaces()
    if not as_json:
        print(
            f"[S07] interface map: {resolved['steps']} steps, "
            f"{resolved['interfaces']} interfaces, ok={resolved['ok']}"
        )
        for failure_item in resolved["failures"]:
            print(f"  FAIL {failure_item['symbol']}: {failure_item['error']}")
        if len(experiments) == len(experiment.EXPERIMENTS):
            print(imap.mapping_table_markdown())
    else:
        print(json.dumps(resolved, indent=2))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "s07_interface_map.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(imap.mapping_table_markdown())
        print(f"table written to {path}", file=sys.stderr)
    return resolved


def mode_self_check(as_json: bool) -> Dict[str, Any]:  # noqa: C901 - a flat audit is clearer
    """Smoke-level checks: interfaces callable, negative paths refused.

    Everything here is labelled ``smoke``; none of it is an experiment result and
    none of it may be quoted as one (AGENTS.md result-labelling rules).
    """
    checks: Dict[str, Any] = {"label": "smoke", "stage": experiment.STAGE}
    identity = request_mod.ModelIdentity(
        model_id="Qwen/Qwen3-1.7B",
        model_manifest_sha256="0" * 64,
        revision="smoke-revision",
        tokenizer_id="Qwen/Qwen3-1.7B",
        chat_template_hash="smoke-template",
        precision="float16",
    )
    request = request_mod.RequestSpec(
        request_id="smoke-request",
        identity=identity,
        input_token_ids=tuple(range(32)),
        sampling=request_mod.SamplingSpec(mode="greedy"),
        stop=request_mod.StopSpec(max_new_tokens=4, eos_token_id=2),
    )

    # ── E07-01: capability, silent degradation, adapter lifecycle ──────
    capability = request_mod.CapabilityReport(backend_id="smoke")
    capability.declare("model_artifact", "SUPPORTED_EXACT")
    capability.declare("input_token_ids", "SUPPORTED_EXACT")
    capability.declare("max_new_tokens", "SUPPORTED_EXACT")
    capability.declare("precision", "SUPPORTED_EXACT")
    capability.declare("sampling_mode", "SUPPORTED_EXACT")
    capability.declare("streaming", "SUPPORTED_EXACT")
    capability.declare("cancel", "SUPPORTED_EXACT")
    capability.declare("timeout", "SUPPORTED_EXACT")
    capability.declare(
        "prefix_cache",
        "UNSUPPORTED_REJECT",
        reason="smoke backend has no KV store",
    )
    capability.declare(
        "quant_artifact",
        "UNSUPPORTED_REJECT",
        reason="smoke backend executes no quantized weights",
    )
    resolved = request_mod.resolve_request(request, capability)
    refused_prefix_cache = False
    try:
        request_mod.resolve_request(
            request_mod.RequestSpec(
                request_id="smoke-prefix",
                identity=identity,
                input_token_ids=(1, 2, 3),
                prefix_cache_enabled=True,
            ),
            capability,
        )
    except Exception:  # noqa: BLE001 - the refusal is the expected behaviour
        refused_prefix_cache = True
    silent_change_refused = False
    try:
        resolved.record("streaming", True, False)
    except Exception:  # noqa: BLE001 - expected refusal
        silent_change_refused = True
    backend_spec = request_mod.BackendSpec(
        backend_id="smoke",
        role="reference",
        version="0.0.0",
        commit="smoke-commit",
        source_identity="smoke-source",
        adapter_module="hqsb.runtime.adapter",
    )
    dummy = adapter.DummyRuntimeAdapter(backend_spec)
    scenario_results = adapter.run_load_close_scenarios(
        lambda: adapter.DummyRuntimeAdapter(backend_spec)
    )
    checks["E07-01_capability_adapter"] = {
        "capability_fields": len(capability.fields),
        "missing_fields": capability.missing_fields(),
        "changed_parameters": [item.name for item in resolved.changed_parameters()],
        "prefix_cache_refused": refused_prefix_cache,
        "silent_change_refused": silent_change_refused,
        "engine_probe": [probe.as_dict() for probe in adapter.probe_environment()],
        "primary_selection_ok": adapter.select_primary_engine(
            adapter.probe_environment()
        )["ok"],
        "load_close_scenarios_completed": all(item.get("ok") for item in scenario_results),
        "dummy_performance_claim": dummy.performance_claim_allowed()["allowed"],
    }

    # ── E07-01: parity oracles ─────────────────────────────────────────
    reference = adapter.GenerationResult(
        request_id=request.request_id,
        token_ids=(11, 12, 13),
        finish_reason="length",
    )
    candidate = adapter.GenerationResult(
        request_id=request.request_id,
        token_ids=(11, 12, 13),
        finish_reason="length",
    )
    divergent = adapter.GenerationResult(
        request_id=request.request_id,
        token_ids=(11, 12, 99),
        finish_reason="length",
    )
    chunks = tuple(
        adapter.StreamChunk(
            request_id=request.request_id,
            token_index=index,
            token_id=token,
            timestamp_ns=index + 1,
            final=index == len(reference.token_ids) - 1,
        )
        for index, token in enumerate(reference.token_ids)
    )
    unsupported_matrix = parity.unsupported_negative_matrix(
        capability,
        [
            parity.UnsupportedObservation(
                field="prefix_cache",
                declared_state=capability.state_of("prefix_cache"),
                observed_behaviour="REJECTED",
                reason_exposed="no KV store",
            ),
            parity.UnsupportedObservation(
                field="quant_artifact",
                declared_state=capability.state_of("quant_artifact"),
                observed_behaviour="REJECTED",
                reason_exposed="no quantized weights",
            ),
        ],
    )
    checks["E07-01_parity"] = {
        "greedy_parity_ok": parity.compare_greedy(reference, candidate).ok,
        "first_divergence_index": parity.compare_greedy(
            reference, divergent
        ).first_divergence_index,
        "streaming_ok": parity.compare_streaming(reference, chunks)["ok"],
        "boundary_cases": len(parity.boundary_cases()),
        "unsupported_matrix_ok": unsupported_matrix["ok"],
        "single_seed_refused": _refuses(
            lambda: parity.compare_sampling_distribution(
                request_id="smoke",
                left_samples=[0.1],
                right_samples=[0.1],
                pre_registered_bound=0.01,
            )
        ),
    }

    # ── E07-02: state machine, spans, conservation, overhead ───────────
    machine = trace.RequestStateMachine("smoke-request")
    for state in (
        trace.RequestState.VALIDATED,
        trace.RequestState.WAITING,
        trace.RequestState.ADMITTED,
        trace.RequestState.PREFILLING,
        trace.RequestState.DECODING,
        trace.RequestState.FINISHED,
        trace.RequestState.CLEANED,
    ):
        machine.transition(state)
    illegal_refused = _refuses(
        lambda: trace.RequestStateMachine("illegal").transition(
            trace.RequestState.DECODING
        )
    )
    collector = trace.SpanCollector(run_id="smoke-run")
    parent = collector.emit(
        "scheduler_iteration",
        request_id="r0",
        source_symbol="runtime.scheduler.step",
        end_ns=5,
    )
    collector.emit(
        "model_runner",
        request_id="r0",
        parent_span_id=parent.span_id,
        source_symbol="runtime.runner.forward",
        iteration=0,
        start_ns=2,
        end_ns=4,
    )
    ledger_entry = trace.IterationLedgerEntry(
        iteration=0,
        scheduled_tokens={"r0": 4},
        token_budget=8,
        previous_computed_positions=0,
        new_computed_positions=4,
    )
    overhead = trace.instrumentation_overhead(
        [
            trace.OverheadObservation(level="off", cpu_ms=10.0, gpu_ms=5.0, ttft_ms=20.0, tpot_ms=8.0),
            trace.OverheadObservation(level="minimal", cpu_ms=10.2, gpu_ms=5.0, ttft_ms=20.2, tpot_ms=8.05),
            trace.OverheadObservation(level="full", cpu_ms=25.0, gpu_ms=5.2, ttft_ms=30.0, tpot_ms=12.0),
        ],
        thresholds=trace.OverheadThresholds(),
    )
    clock = trace.ClockCalibration(
        host_reference_ns=0, device_reference_ns=0, skew_ns=0, method="smoke"
    )
    checks["E07-02_trace"] = {
        "state_machine_terminal": machine.state,
        "illegal_transition_refused": illegal_refused,
        "join_audit_ok": collector.join_audit()["ok"],
        "cross_request_parent_detected": _detects_cross_request(collector),
        "conservation_ok": ledger_entry.conservation_residual() == 0,
        "timing_level": overhead["timing_level"],
        "full_trace_not_timing": not next(
            row for row in overhead["rows"] if row["level"] == "full"
        )["usable_for_timing"],
        "device_span_ns": clock.device_span_ns(10, 30),
        "redaction": trace.redact_attributes({"prompt": "secret text"})["prompt"],
    }

    # ── E07-03: KV geometry, block pool, faults ────────────────────────
    geometry = kv.KVGeometry(
        num_layers=28, num_kv_heads=8, head_dim=128, element_bytes=2
    )
    pool = kv.BlockPool(total_blocks=8, block_size=4)
    allocated = pool.allocate("r0", token_start=0, token_count=9)
    pool.mark_cached(allocated[-1], "r0")
    pool.release("r0")
    double_free = pool.inject_double_free("r0")
    pool.evict(allocated[0])
    stale = pool.inject_stale_id_access(allocated[0])
    capacity = kv.KVCapacityModel(
        geometry=geometry,
        block_size=16,
        blocks_total=64,
        non_kv_resident_bytes=1.0,
        watermark=0.9,
    )
    reconciliation = kv.MemoryReconciliation(
        geometry=geometry,
        block_size=16,
        active_tokens=64,
        tolerance_bytes=1024.0,
    )
    reconciliation.declare("block_table_metadata", 512.0)
    reconciliation.measured_framework_reserved = reconciliation.predicted_total_bytes + 128
    checks["E07-03_kv"] = {
        "bytes_per_token": geometry.bytes_per_token,
        "blocks_for_9_tokens": kv.blocks_for(9, 4),
        "boundary_points": list(kv.block_boundary_points(4, blocks=1)),
        "pool_invariants_ok": pool.invariant_report()["ok"],
        "double_free": double_free,
        "stale_id": stale,
        "capacity_safe_tokens": capacity.safe_admission_tokens(),
        "reconciliation_explained": reconciliation.explained(),
        "oom_sequence_leak": failure.oom_sequence("actual_leak"),
        "oom_sequence_exec": failure.oom_sequence("execution_oom"),
    }

    # ── E07-04: scheduler structure and conservation ───────────────────
    spec = scheduler.SchedulerSpec(
        mode=scheduler.CHUNKED_PREFILL,
        max_batched_tokens=64,
        max_sequences=4,
        block_size=16,
        kv_blocks=64,
        chunk_size=16,
    )
    trace_requests = scheduler.mixed_trace(
        "smoke-mixed",
        short_count=2,
        long_count=2,
        short_prompt=32,
        long_prompt=128,
        max_new_tokens=8,
    )
    simulated = scheduler.simulate(spec, trace_requests, max_iterations=128)
    chunk_audit = scheduler.chunk_coverage_audit(64, 16, [(0, 16), (16, 16), (32, 16), (48, 16)])
    checks["E07-04_scheduler"] = {
        "makespan_iterations": simulated.makespan_iterations,
        "conservation_ok": simulated.conservation()["ok"],
        "padding_slots": simulated.padding_slots,
        "kv_peak_blocks": simulated.kv_peak_blocks,
        "fairness": simulated.fairness(),
        "congestion": scheduler.detect_congestion(simulated).as_dict(),
        "chunk_coverage_ok": chunk_audit["ok"],
        "jain_uniform": scheduler.fairness_jain([1.0, 1.0, 1.0]),
    }

    # ── E07-05: prefix cache key, negatives, eviction protection ───────
    prefix_identity = {
        "model_id": identity.model_id,
        "weight_revision": identity.revision,
        "precision": identity.precision,
        "quant_artifact_hash": "",
        "adapter_hash": "",
        "tokenizer_id": identity.tokenizer_id,
        "chat_template_hash": identity.chat_template_hash,
        "rope_config_hash": "rope",
        "attention_config_hash": "attn",
    }
    fixture_tokens = list(range(64))
    key = prefix_cache.build_prefix_key(
        identity=prefix_identity, tokens=fixture_tokens, token_span=(0, 32)
    )
    cache_spec = prefix_cache.PrefixCacheSpec(block_size=16, max_cache_bytes=1 << 20)
    cache = prefix_cache.PrefixCache(cache_spec)
    entry = cache.insert(
        key=key, block_ids=(0, 1), bytes_value=1024.0, request_id="r0"
    )
    lookup = cache.lookup(
        key=key, query_tokens=32, block_groups=("full_attention",), request_id="r1"
    )
    evict_refused = _refuses(lambda: cache._drop(entry.entry_id, reason="smoke"))
    collision = prefix_cache.collision_is_detected(cache_spec, prefix_identity)
    digest_only = prefix_cache.collision_is_detected(
        prefix_cache.PrefixCacheSpec(
            block_size=16,
            collision_policy="strong_digest",
            verify_token_equality=False,
            max_cache_bytes=1 << 20,
        ),
        prefix_identity,
    )
    negatives = prefix_cache.identity_negative_fixtures(
        prefix_identity, tokens=fixture_tokens
    )
    checks["E07-05_prefix"] = {
        "key_fields": len(prefix_cache.IDENTITY_FIELDS),
        "digest_len": len(key.effective_digest),
        "lookup_cached_tokens": lookup.cached_tokens,
        "lookup_computed_tokens": lookup.computed_tokens,
        "active_entry_evict_refused": evict_refused,
        "collision_rejected_by_strict_policy": collision["correct"],
        "collision_flagged": collision["collision_detected"],
        "digit_only_policy_serves_forged_record": not digest_only["correct"],
        "negative_fixtures": len(negatives),
        # A fixture that must not hit has to be refusable: either its identity
        # differs or its digest does. Anything else would be a fixture that
        # changes nothing.
        "identity_negatives_rejected": all(
            item["must_hit"]
            or (not item["identity_matches"])
            or bool(item.get("digest_differs"))
            for item in negatives
        ),
        "net_saving_kept_apart": prefix_cache.NetSavingModel(
            baseline_prefill_ms=10.0,
            cached_request_prefill_ms=6.0,
            lookup_hash_ms=0.5,
            eviction_recompute_ms=0.0,
            other_requests_impact_ms=0.2,
            saved_tokens=16,
        ).net_saved_time_ms(),
    }

    # ── E07-06: buckets, replay distinctness, claim gate ──────────────
    graph_spec = graph_route.GraphSpec(
        buckets=(
            graph_route.GraphBucketSpec(name="decode_b1_s1", max_sequences=1, max_tokens=1),
            graph_route.GraphBucketSpec(name="prefill_b1_s512", max_sequences=1, max_tokens=512),
        ),
        claim_cuda_graph=False,
    )
    hit = graph_spec.resolve(sequences=1, tokens=1, phase="decode")
    missed = graph_spec.resolve(sequences=1, tokens=2048, phase="prefill")
    replays = (
        graph_route.ReplayRecord(bucket="decode_b1_s1", inputs_hash="a", replay_index=0),
        graph_route.ReplayRecord(bucket="decode_b1_s1", inputs_hash="b", replay_index=1),
    )
    stale_replays = (
        graph_route.ReplayRecord(bucket="decode_b1_s1", inputs_hash="a", replay_index=0),
        graph_route.ReplayRecord(bucket="decode_b1_s1", inputs_hash="a", replay_index=1),
    )
    attention_candidate = graph_route.AttentionCandidate(
        name="smoke_default",
        provider="runtime",
        version="0.0.0",
        dtypes=("float16",),
        kv_dtypes=("float16",),
        head_dim=128,
        supports_gqa=True,
        mask="causal",
        phases=("prefill", "decode"),
        paged_layout=True,
        max_context=4096,
    )
    supported = graph_route.check_attention_support(
        attention_candidate,
        graph_route.AttentionRequest(
            dtype="float16",
            kv_dtype="float16",
            head_dim=128,
            query_heads=16,
            kv_heads=8,
            context=1024,
            phase="decode",
            paged=True,
        ),
    )
    unsupported_attention = graph_route.check_attention_support(
        attention_candidate,
        graph_route.AttentionRequest(
            dtype="float16",
            kv_dtype="float16",
            head_dim=128,
            query_heads=16,
            kv_heads=8,
            context=8192,
            phase="decode",
            paged=True,
        ),
    )
    checks["E07-06_graph"] = {
        "bucket_hit": (hit.bucket, hit.status, hit.actual_mode),
        "out_of_bucket": (missed.status, missed.fallback, missed.reason != ""),
        "replay_distinctness_ok": graph_route.replay_distinctness(replays)["ok"],
        "stale_replay_detected": not graph_route.replay_distinctness(stale_replays)["ok"],
        "attention_supported": supported.supported,
        "attention_rejected_reason": unsupported_attention.reason != "",
        "graph_claim": graph_route.claim_status(
            graph_route.GraphClaimEvidence(), executed=False, spec=graph_spec
        )["status"],
        "e06_08_inheritance_available": graph_route.inherit_from_e06_08()["available"],
    }

    # ── E07-08: exact acceptance, rollback, claim gate ────────────────
    from fractions import Fraction

    residual = spec_decode.residual_distribution(
        (Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 4), Fraction(1, 4))
    )
    cycles = (
        spec_decode.CycleRecord(
            cycle_index=0,
            proposed=3,
            accepted=2,
            correction_token_committed=True,
            advanced_tokens=3,
            draft_ms=1.0,
            verify_ms=4.0,
            rollback_ms=0.5,
        ),
        spec_decode.CycleRecord(
            cycle_index=1,
            proposed=3,
            accepted=1,
            correction_token_committed=False,
            advanced_tokens=1,
            draft_ms=1.0,
            verify_ms=4.0,
        ),
    )
    checks["E07-08_spec_decode"] = {
        "residual_normalized": spec_decode.residual_is_normalized(residual),
        "acceptance_probability": str(
            spec_decode.acceptance_probability(Fraction(1, 2), Fraction(1, 4))
        ),
        "clamped_acceptance": str(
            spec_decode.acceptance_probability(Fraction(1, 1), Fraction(1, 4))
        ),
        "greedy_exactness": spec_decode.verify_greedy_exactness(
            [1, 2, 3], [1, 2, 3], max_new_tokens=3
        )["ok"],
        "rollback_audit_ok": spec_decode.kv_commit_rollback_audit(
            proposed_positions=3,
            accepted_positions=2,
            committed_positions=3,
            rolled_back_positions=1,
        )["ok"],
        "benefit_effective_tpot": spec_decode.BenefitModel(
            cycles=cycles, vanilla_tpot_ms=10.0
        ).effective_tpot_ms,
        "claim_without_execution": spec_decode.claim_status(
            spec_decode.SpeculativeEvidence(), executed=False, algorithm=spec_decode.GREEDY
        )["status"],
        "mtp_borrow_refused": _refuses(
            lambda: spec_decode.assert_mtp_does_not_borrow(
                spec_decode.MtpContract(
                    heads=1,
                    quality_contract="none",
                    borrows_strict_sampling_guarantee=True,
                )
            )
        ),
    }

    # ── E07-09: failure matrix, context abuse, long-run verdicts ──────
    matrix = failure.frozen_matrix()
    context_check = failure.ContextAbuseCheck(
        requested_tokens=100000,
        rejected_at_layer="model_max",
        tokens_allocated_before_reject=0,
        kv_capacity_tokens=16000,
    )
    slopes = failure.resource_slope_report(
        {"kv_blocks": [10.0, 10.5, 11.0, 11.2, 11.3, 11.35]},
        warmup_cycles=2,
        tolerance=1024.0,
    )
    growing = failure.resource_slope_report(
        {"kv_blocks": [10.0, 12.0, 20.0, 40.0, 80.0, 160.0]},
        warmup_cycles=1,
        tolerance=1.0,
    )
    checks["E07-09_failure"] = {
        "matrix_cases": len(matrix),
        "matrix_categories": sorted({case.category for case in matrix}),
        "common_invariants": len(failure.COMMON_INVARIANTS),
        "context_abuse_ok": context_check.ok,
        "bounded_steady": slopes.ok,
        "growing_blocks_pass": failure.leak_blocks_pass(growing)["pass_allowed"],
        "run_separation": sorted(failure.run_separation_plan().keys())[:3],
        "load_close_cases": len(failure.load_close_scenarios()),
        "concurrency_cases": len(failure.concurrency_release_scenarios()),
        "healthy_probe_single_round_not_enough": failure.healthy_request_probe(
            lambda: True, rounds=3
        )["ok"],
    }

    # ── E07-10 / E07-07: comparison and A/B gates ─────────────────────
    comparison_spec = comparison.ComparisonSpec(
        model_id=identity.model_id,
        model_manifest_sha256=identity.model_manifest_sha256,
        precision="float16",
        hardware="smoke-host",
        request_trace_hash=trace_requests.trace_hash,
    )
    backend_identity = comparison.BackendIdentity(
        backend_id="smoke",
        role="reference",
        version="0.0.0",
        commit="smoke-commit",
        model_id=identity.model_id,
        precision="float16",
        hardware="smoke-host",
    )
    row = comparison.ComparisonRow(
        backend=backend_identity,
        workload="smoke",
        tier=backend_identity.tier(comparison_spec),
        mode="common_denominator",
        metrics={"ttft_ms": 10.0},
    )
    recomputed = comparison.recompute_metrics(
        requests=[
            {
                "logical_input_tokens": 32,
                "committed_output_tokens": 4,
                "ttft_ms": 10.0,
                "tpot_ms": 2.0,
            }
        ],
        iterations=[{"model_runner_ms": 20.0, "scheduler_cpu_ms": 1.0, "sample_ms": 0.5}],
    )
    denominator_audit = comparison.token_denominator_audit(
        raw=recomputed, reported={"output_tps": recomputed["output_tps"]}
    )
    gate = policy_ab.SelectionGate(
        baseline_bottleneck_quantified=True,
        mechanism_explainable=True,
        metric_preregistered=True,
        not_duplicating_upstream=True,
        risk_and_fallback_controllable=True,
        counterexample_constructible=True,
        scope_consistent_with_s08_s11=True,
        evidence_refs=("E07-02:kv_lookup",),
    )
    ineligible_refused = _refuses(
        lambda: policy_ab.SelectionGate(
            baseline_bottleneck_quantified=False,
            mechanism_explainable=True,
            metric_preregistered=True,
            not_duplicating_upstream=True,
            risk_and_fallback_controllable=True,
            counterexample_constructible=True,
            scope_consistent_with_s08_s11=True,
            evidence_refs=("E07-02:kv_lookup",),
        ).require_ok()
    )
    checks["E07-10_comparison"] = {
        "tier": row.tier,
        "recomputed_output_tps": recomputed["output_tps"],
        "token_ledger": recomputed["token_ledger"],
        "denominator_audit_ok": denominator_audit["ok"],
        "pareto_fronts": len(
            comparison.pareto_front(
                [
                    comparison.ParetoPoint(
                        backend_id="smoke",
                        workload="smoke",
                        hardware="smoke-host",
                        objectives={"ttft_ms": 10.0, "throughput": 100.0},
                    )
                ]
            )["fronts"]
        ),
        "s08_surface_keys": sorted(comparison.s08_interface_surface()),
        "selection_gate_ok": gate.ok,
        "ineligible_change_refused": ineligible_refused,
        "schedule_balance_ok": policy_ab.schedule_balance(
            policy_ab.block_schedule(blocks=3, scheme="ABBA")
        )["ok"],
        "identity_diff_detected": not policy_ab.identity_equal(
            _smoke_identity("p1"), _smoke_identity("p1", model="other")
        )["ok"],
    }

    # ── metrics / configs / telemetry ─────────────────────────────────
    ledger = metrics.TokenLedger(
        logical_input_tokens=64,
        committed_output_tokens=16,
        cached_prefix_tokens=32,
        speculative_verified_positions=2,
        cancelled_or_failed_positions=4,
    )
    checks["metrics"] = {
        "useful_committed": ledger.useful_committed_tokens,
        "model_computed_positions": ledger.model_computed_positions,
        "conservation_ok": ledger.audit().ok,
        "paired_effect_crosses_zero": metrics.paired_effect(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]
        ).crosses_zero,
        "equivalence_requires_bound": _refuses(
            lambda: metrics.EquivalenceCheck(
                bound=0.0, effect=metrics.paired_effect([1.0], [1.0])
            )
        ),
    }

    config_report: Dict[str, Any] = {}
    try:
        loaded = specs.RuntimeSpecs.load(CONFIG_DIR)
        config_report = {
            "documents": len(loaded.documents),
            "kinds": sorted(loaded.documents),
            "audits_ok": loaded.ok,
            "failing": [
                audit["kind"] for audit in loaded.audits if not audit["ok"]
            ],
        }
    except Exception as exc:  # noqa: BLE001 - report, never fake a pass
        config_report = {"error": f"{type(exc).__name__}: {exc}"}
    checks["configs"] = config_report

    coverage = telemetry.c6_c7_summary()
    collector = telemetry.TraceCollector(run_id="smoke-run")
    chain_ok = True
    parent_ids: Dict[str, str] = {}
    previous = ""
    for kind in telemetry.SPAN_CHAIN:
        parent_ids[kind] = previous
        record = collector.emit(kind, request_id="r0", parent_span_id=previous, iteration=0)
        previous = record.span_id
    events = telemetry.to_trace_events(collector.records, run_id="smoke-run", trace_id="t0")
    projected = telemetry.project_c6(
        "smoke-run",
        telemetry.S07ResultFields(
            backend_id="smoke",
            runtime_version="0.0.0",
            runtime_commit="smoke-commit",
            requested_capability={"prefix_cache": "SUPPORTED_EXACT"},
            actual_capability={"prefix_cache": "UNSUPPORTED_REJECT"},
            capability_reasons={"prefix_cache": "smoke backend has no KV store"},
            request_trace_hash=trace_requests.trace_hash,
            scheduler_policy="chunked_prefill",
            kv_policy="block-16",
            prefix_policy="lru",
            graph_policy="decode_b1_s1",
            per_request_metrics_uri="artifact://requests",
            iteration_ledger_uri="artifact://iterations",
            actual_precision="float16",
            observed_kernel="smoke_kernel",
            observed_attention_backend="runtime_default",
            quality_status="not_run",
            raw_artifacts={"requests": "artifact://requests"},
        ),
    )
    checks["telemetry"] = {
        "c6_ok": coverage["c6"]["ok"],
        "c7_ok": coverage["c7"]["ok"],
        "c2_ok": coverage["c2"]["ok"],
        "chain_ok": telemetry.chain_coverage(collector.records)["ok"],
        "join_ok": telemetry.trace_join_check(events)["ok"],
        "c6_summary_keys": sorted(projected.summary.get("s07", {})),
        "chain_ok_flag": chain_ok,
    }

    # ── handbook §4 unified record + C2 alignment ──────────────────────
    record = experiment.template_experiment_record(
        "E07-01", status=experiment.STATUS_BLOCKED, reason="self-check template"
    )
    conclusion_without_evidence_refused = _refuses(
        lambda: experiment.ExperimentRecord(
            experiment_id="E07-01",
            question="q",
            hypothesis="h",
            status=experiment.STATUS_PASS,
            decision="looks good",
        )
    )
    checks["experiment_record"] = {
        "fields": len(experiment.EXPERIMENT_RECORD_FIELDS),
        "template_status": record.status,
        "template_decision_empty": record.decision == "",
        "conclusion_without_evidence_refused": conclusion_without_evidence_refused,
    }

    if as_json:
        print(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    else:
        print("[S07] self-check (smoke; NOT an experiment result)")
        for name, payload in checks.items():
            if name in ("label", "stage"):
                continue
            print(f"  {name}: {json.dumps(payload, ensure_ascii=False, default=str)}")
    return checks


def _smoke_identity(patch: str, model: str = "Qwen/Qwen3-1.7B") -> policy_ab.AbIdentity:
    return policy_ab.AbIdentity(
        model_id=model,
        tokenizer_id=model,
        precision="float16",
        runtime_base_commit="smoke-base",
        hardware="smoke-host",
        request_trace_hash="smoke-trace",
        scheduler_config_hash="smoke-scheduler",
        kv_graph_attention_config_hash="smoke-kv",
        warmup_policy="smoke-warmup",
        measurement_policy="smoke-measure",
        seed=0,
        patch_hash=patch,
        build_hash="smoke-build",
    )


def _refuses(callable_: Any) -> bool:
    """Return True when the call raises — the negative path is the expectation."""
    try:
        callable_()
    except Exception:  # noqa: BLE001 - refusal is the expected behaviour
        return True
    return False


def _detects_cross_request(collector: trace.SpanCollector) -> bool:
    """A child whose request differs from its parent must be reported."""
    probe = trace.SpanCollector(run_id="cross-request")
    parent = probe.emit("scheduler_iteration", request_id="r0", source_symbol="s")
    probe.emit(
        "model_runner",
        request_id="r1",
        parent_span_id=parent.span_id,
        source_symbol="s",
    )
    return bool(probe.join_audit()["cross_request_parent"])


def mode_execute(
    experiments: Sequence[str], output_dir: str, run_id: str, confirm: bool, as_json: bool
) -> Dict[str, Any]:
    """Refuse to execute unless the whole gate chain explicitly allows it."""
    prerequisites = experiment.check_prerequisites(REPO_ROOT)
    if not confirm:
        raise UsageError("--mode execute requires --confirm-execute; nothing was run")
    if not prerequisites.satisfied:
        report = {
            "mode": "execute",
            "executed": False,
            "status": experiment.STATUS_BLOCKED,
            "missing": prerequisites.missing,
            "reason": (
                "S07 cannot execute: the upstream evidence chain (S04.5 M4 marker, "
                "S05 quality/kernel verdicts, S06 stable capability verdicts, frozen "
                "request fixtures, runtime capability probe, main-runtime selection) "
                "is unmet. The interface code is ready and its self-check is "
                "available via --mode self-check."
            ),
        }
        print(json.dumps(report, indent=2) if as_json else report["reason"])
        return report
    # Prerequisites satisfied: the environment-specific execution is still
    # deliberately not implemented here — the driver must be extended with the
    # concrete runner for the target machine instead of silently "succeeding".
    raise UsageError(
        "prerequisites are satisfied but no real runner is wired in this tree; "
        "extend scripts/runtime/run_e07.py with the hardware-specific runner "
        "instead of faking a result"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        experiments = _selected_experiments(args.experiment)
        if args.mode == "status":
            mode_status(experiments, args.output_dir, args.run_id, args.json)
        elif args.mode == "preregister":
            mode_preregister(experiments, args.output_dir, args.run_id, args.template, args.json)
        elif args.mode == "interface-map":
            mode_interface_map(experiments, args.output_dir, args.json)
        elif args.mode == "self-check":
            mode_self_check(args.json)
        elif args.mode == "execute":
            report = mode_execute(
                experiments, args.output_dir, args.run_id, args.confirm_execute, args.json
            )
            if not report.get("executed"):
                return ExitCode.CAPABILITY
        return ExitCode.SUCCESS
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - map to a stable exit code
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return exit_code_for(exc)


if __name__ == "__main__":
    raise SystemExit(main())
