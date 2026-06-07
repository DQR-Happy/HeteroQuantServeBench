#!/usr/bin/env python3
"""S06 experiment driver / interface entry point.

Modes
-----
``status`` (default)
    Print the prerequisite report and each experiment's protocol status.
    Writes nothing unless ``--output-dir`` is given, in which case a
    ``status.json`` (never a verdict) is recorded.
``preregister``
    Write ``preregistration.json`` from the shipped template with a frozen
    hash, plus the empty run layout.
``interface-map``
    Print (or write) the experiment-step → interface table and verify that
    every referenced symbol imports.
``self-check``
    Run the *smoke-level* interface self-checks, labelled as smoke.  These are
    not experiment results: they only prove that the interfaces are callable and
    that the negative paths are refused.
``execute``
    Refused unless ``--confirm-execute`` is passed **and** the prerequisite
    chain is satisfied.  This repository state does not satisfy it, so the mode
    exits non-zero with the missing evidence listed — by design.

Exit codes follow :class:`hqsb.core.errors.ExitCode`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.core.errors import ExitCode, UsageError, exit_code_for  # noqa: E402
from hqsb.integration import adapter, dispatch, experiment, graph, guards, lowering  # noqa: E402
from hqsb.integration import interface_map as imap  # noqa: E402
from hqsb.integration import meta, patterns, policies, specs, taxonomy, telemetry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "S06 experiment driver. The default mode only inspects prerequisites "
            "and interfaces; no conclusion can be produced without an explicit, "
            "prerequisite-gated execution."
        )
    )
    parser.add_argument(
        "--experiment",
        default="all",
        help="E06-01 .. E06-11, or 'all' (default).",
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
        "question": "see docs/stage_experiments/details/S06/<file>",
        "hypothesis": "see docs/stage_experiments/details/S06/<file>",
        "capture_mode": "",
        "pattern": "",
        "dynamic_policy": "",
        "shape_sequence": (),
        "cache_state": "",
        "quality_tolerance": {},
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
            "pattern/dynamic policy/tolerances/metrics from the protocol document "
            "before any execution"
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
        print(f"[S06] prerequisite chain satisfied: {prerequisites.satisfied}")
        for check in prerequisites.checks:
            marker = "OK " if check.satisfied else "MISS"
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
        run.write_json("prerequisites.json", prerequisites.as_dict())
        run.write_json("environment_fingerprint.json", experiment.environment_fingerprint())
        run.write_report_skeleton(experiment_id, imap.mapping_for(experiment_id).title)
        run.write_status(
            experiment.STATUS_BLOCKED if not prerequisites.satisfied else experiment.STATUS_NOT_STARTED,
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
            f"{item['experiment_id']}: prereg_hash={item['prereg_hash'][:16]} dir={item['run_dir']}"
            for item in written
        )
    )
    return report


def mode_interface_map(experiments: Sequence[str], output_dir: str, as_json: bool) -> Dict[str, Any]:
    resolved = imap.resolve_interfaces()
    if not as_json:
        print(
            f"[S06] interface map: {resolved['steps']} steps, "
            f"{resolved['interfaces']} interfaces, ok={resolved['ok']}"
        )
        for failure in resolved["failures"]:
            print(f"  FAIL {failure['symbol']}: {failure['error']}")
        if len(experiments) == len(experiment.EXPERIMENTS):
            print(imap.mapping_table_markdown())
    else:
        print(json.dumps(resolved, indent=2))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "s06_interface_map.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(imap.mapping_table_markdown())
        print(f"table written to {path}", file=sys.stderr)
    return resolved


def mode_self_check(as_json: bool) -> Dict[str, Any]:
    """Smoke-level checks: interfaces callable, negative paths refused.

    Everything here is labelled ``smoke``; none of it is an experiment result
    and none of it may be quoted as one (AGENTS.md result-labelling rules).
    """
    checks: Dict[str, Any] = {"label": "smoke", "stage": experiment.STAGE}

    # ── E06-01: schema, dispatch, redispatch, conflicts ────────────────
    matrix = dispatch.frozen_p0_matrix()
    duplicate = matrix.register(
        dispatch.RegistrationRecord(
            qualified_name=specs.OP_RMS_NORM,
            key=dispatch.DispatchKey.CUDA,
            implementation="different_kernel",
            provider="cuda_shared_lib",
            library="libhqsb_ops",
            schema_hash=matrix.schemas[specs.OP_RMS_NORM],
        )
    )
    collision = matrix.register(
        dispatch.RegistrationRecord(
            qualified_name="hqsb::squatter",
            key=dispatch.DispatchKey.CUDA,
            implementation="third_party",
            provider="python",
            library="somewhere",
            owner="third_party",
        )
    )
    try:
        dispatch.redispatch_keyset(
            frozenset({dispatch.DispatchKey.CUDA}),
            frozenset({dispatch.DispatchKey.CUDA}),
        )
        redispatch_refused = False
    except dispatch.RedispatchError:
        redispatch_refused = True
    checks["dispatch"] = {
        "ops": len(matrix.schemas),
        "records": len(matrix.records),
        "duplicate_action": duplicate.action,
        "namespace_collision_action": collision.action,
        "redispatch_exhaustion_refused": redispatch_refused,
    }

    # ── E06-02: metadata contract + no-allocation evidence ─────────────
    contract = meta.rms_norm_contract()
    symbols = meta.SymInt.symbol("S")
    x = meta.TensorMeta(shape=(symbols, 64), dtype="float16", device="cuda", stride=(64, 1))
    weight = meta.TensorMeta(shape=(64,), dtype="float16", device="cuda", stride=(1,))
    spy = meta.FakeCallSpy(op=contract.op)
    outputs = contract.infer_outputs([x, weight], a_last=64, b_last=64, rank=2, weight_rank=1)
    invalid_refused = False
    try:
        contract.infer_outputs([x, weight], a_last=64, b_last=32, rank=2, weight_rank=1)
    except meta.MetadataError:
        invalid_refused = True
    checks["meta"] = {
        "output_shape": meta.dims_repr(outputs[0].shape),
        "preserves_symbol": outputs[0].symbols() == ("S",),
        "no_allocation_evidence": spy.clean,
        "invalid_shape_refused": invalid_refused,
        "real_vs_fake_ok": meta.compare_metadata(contract.op, outputs[0], outputs[0]).ok,
    }

    # ── E06-04: positive hit, negative mutations, copy-on-rewrite ──────
    fixture = graph.from_node_sequence(
        [
            {"name": "x", "op": "graph_input"},
            {"name": "residual", "op": "graph_input"},
            {"name": "add", "op": "aten.add.Tensor", "args": ["x", "residual"]},
            {"name": "norm", "op": "hqsb.rms_norm", "args": ["add", "weight", 1e-6], "output": True},
        ],
        capture_mode=graph.CaptureMode.DYNAMO,
        ir_level=graph.IRLevel.DYNAMO_FX,
    )
    spec = patterns.residual_add_rmsnorm_pattern()
    context = patterns.PatternContext(
        graph=fixture,
        declared_eps=1e-6,
        pattern_eps=1e-6,
        norm_axis=-1,
        hidden_size=64,
        weight_shapes={"weight": (64,)},
        weight_dtype="float16",
        supported_versions=("1.0.0",),
        version="1.0.0",
    )
    decisions = patterns.scan_graph(spec, fixture, lambda _graph, _match: context)
    mutations = patterns.mutation_report(
        spec, decisions[0].candidate, context, patterns.standard_mutations()
    ) if decisions else ()
    before_hash = fixture.structural_hash()
    if decisions:
        rewrite = patterns.apply_rewrite(
            fixture,
            decisions[0],
            context,
            lambda decision, _ctx: graph.GraphNode(
                name="fused_norm",
                op="hqsb.fused_add_rms_norm",
                args=(*fixture.node("add").args, *fixture.node("norm").args[1:2]),
            ),
        )
        rewrite_diff = rewrite.diff.as_dict()
    else:
        rewrite_diff = {}
    checks["patterns"] = {
        "candidates": len(decisions),
        "hit_reason": decisions[0].reason if decisions else "STRUCTURE_MISMATCH",
        "mutations_rejected": sum(1 for item in mutations if item.ok),
        "mutations_total": len(mutations),
        "original_graph_unchanged": fixture.structural_hash() == before_hash,
        "rewrite_node_delta": rewrite_diff.get("node_delta", 0),
    }

    # ── E06-05/06: five-way separation, identity sensitivity, cache ────
    ledger = guards.CompileLedger()
    ledger.record(guards.EventKind.GUARD_FAIL, 0, reason="shape")
    ledger.record(guards.EventKind.RECOMPILE, 0, reason="shape")
    ledger.record(guards.EventKind.GRAPH_BREAK, 0, reason="data-dependent")
    ledger.record(guards.EventKind.FALLBACK, 1, reason="stride")
    five_way = ledger.five_way().as_dict()
    storm = guards.evaluate_storm(
        guards.StormThresholds(),
        guards.StormObservation(
            requests=100,
            recompiles=1,
            compile_wall_ms=10.0,
            total_wall_ms=1000.0,
            unique_shapes=1,
            graph_variants=1,
            fallbacks=1,
            baseline_p95_ms=10.0,
            observed_p95_ms=10.5,
            cache_entries_start=0,
            cache_entries_end=1,
        ),
    )
    with tempfile.TemporaryDirectory(prefix="hqsb-e06-selfcheck-") as tmp:
        entry_dir = os.path.join(tmp, "golden")
        os.makedirs(entry_dir, exist_ok=True)
        payload_path = os.path.join(entry_dir, "kernel.bin")
        with open(payload_path, "wb") as handle:
            handle.write(b"hqsb-kernel-placeholder")
        from hqsb.integration import cache as cache_mod

        entry = cache_mod.CacheEntry(
            key_digest="k",
            layer=cache_mod.CacheLayer.INDUCTOR_FX,
            entry_hash=cache_mod.sha256_file(payload_path),
            payload_path="kernel.bin",
            size_bytes=os.path.getsize(payload_path),
            target_arch="sm_86",
            abi_version="1",
            schema_hash="s",
        )
        clean = cache_mod.validate_entry(entry, entry_dir, expected_arch="sm_86")
        fixture_dir = cache_mod.CorruptionFixture(
            source_dir=entry_dir, scratch_root=os.path.join(tmp, "scratch")
        )
        work = fixture_dir.prepare()
        fixture_dir.bit_flip("kernel.bin", 0)
        corrupt = cache_mod.validate_entry(entry, work, expected_arch="sm_86")
        foreign = cache_mod.validate_entry(entry, entry_dir, expected_arch="sm_87")
    identity_a = cache_mod.graph_identity("graph-a", model_id="m", rewrite_spec_id="r")
    identity_b = cache_mod.graph_identity("graph-a", model_id="m2", rewrite_spec_id="r")
    checks["cache"] = {
        "five_way": five_way,
        "storm_within_thresholds": storm.within_thresholds,
        "clean_entry_ok": clean.ok,
        "bitflip_detected": (not corrupt.ok, corrupt.failure),
        "foreign_arch_detected": (not foreign.ok, foreign.failure),
        "identity_sensitive": identity_a.digest != identity_b.digest,
        "break_even": cache_mod.compute_break_even(1000.0, 10.0, 8.0).requests,
        "no_amortisation": cache_mod.compute_break_even(1000.0, 8.0, 9.0).requests,
    }

    # ── E06-07: lowering selection + allocation accounting ─────────────
    registry = lowering.frozen_registry()
    request = lowering.LoweringRequest(
        pattern_id="hqsb.pattern.residual_add_rms_norm",
        node="fused_norm",
        op="hqsb::fused_add_rms_norm",
        dtype="float16",
        layout="contiguous",
        rank=2,
        m=1,
        arch="sm_86",
    )
    decision = registry.select(request)
    unsupported = registry.select(
        lowering.LoweringRequest(
            pattern_id="hqsb.pattern.dequant_linear",
            op="hqsb::dequant_linear",
            dtype="int8",
            rank=2,
            m=1,
            group_size=7,
        )
    )
    account = lowering.AllocationAccount(
        intermediate_bytes_before=lowering.tensor_bytes((1, 64), "float16") * 2,
        intermediate_bytes_after=lowering.tensor_bytes((1, 64), "float16"),
    )
    checks["lowering"] = {
        "selected": decision.selected_name,
        "rule_reason": decision.rule_reason,
        "unsupported_falls_back": unsupported.fallback,
        "unsupported_reasons": [item.reason for item in unsupported.rejected][:4],
        "theoretical_saving_bytes": account.theoretical_saving_bytes,
        "selection_audit_ok": registry.selection_audit()["ok"],
    }

    # ── E06-08/09/10/11: claim gates, taxonomy, lifecycle, adapter ─────
    from hqsb.integration import abi as abi_mod
    from hqsb.integration import cuda_graph as graph_mod
    from hqsb.integration import lifecycle as lifecycle_mod

    eligibility = graph_mod.evaluate_eligibility(graph_mod.EligibilityObservation())
    claim = graph_mod.claim_status(graph_mod.ClaimEvidence(), executed=False)
    machine = lifecycle_mod.LifecycleMachine(name="compiled_callable")
    machine.transition(lifecycle_mod.LifecycleState.REGISTERED)
    machine.transition(lifecycle_mod.LifecycleState.MODEL_ATTACHED)
    illegal_refused = False
    try:
        machine.transition(lifecycle_mod.LifecycleState.CLOSED)
    except Exception:  # noqa: BLE001 - the refusal is the expected behaviour
        illegal_refused = True
    identity = abi_mod.simulate_identity(torch_version="2.5.0", fatbin_targets=("sm_86",))
    good_report = abi_mod.CompatibilityMatrix().check(identity)
    bad_report = abi_mod.CompatibilityMatrix().check(
        abi_mod.simulate_identity(gpu_arch="sm_90", fatbin_targets=("sm_86",))
    )
    dummy = adapter.DummyBackendAdapter()
    dummy.capability()
    dummy.route({"dtype": "float64"})
    with tempfile.TemporaryDirectory(prefix="hqsb-e06-hardcode-") as tmp:
        offending = os.path.join(tmp, "core_leak.py")
        with open(offending, "w", encoding="utf-8") as handle:
            handle.write(
                "from transformers import Qwen3ForCausalLM\n"
                "num_hidden_layers = 28\n"
                "if backend == 'cuda':\n"
                "    pass\n"
            )
        findings = adapter.hardcode_scan([offending], adapter_markers=())
        hardcode = adapter.hardcode_report(findings)
    checks["guards"] = {
        "eligibility_failures": list(eligibility.failures),
        "cuda_graph_claim": claim["status"],
        "illegal_transition_refused": illegal_refused,
        "abi_ok": good_report.ok,
        "abi_mismatch_codes": list(bad_report.codes()),
        "dummy_backend_claim": dummy.performance_claim_allowed()["allowed"],
        "spy_events": len(dummy.spy),
        "hardcode_leaks": hardcode["leaks"],
        "taxonomy_uncovered_stages": list(taxonomy.frozen_taxonomy().uncovered_stages()),
    }

    # ── configs + C6/C7 projection ────────────────────────────────────
    config_report: Dict[str, Any] = {}
    try:
        documents = policies.load_directory(policies.default_config_dir(REPO_ROOT))
        config_report = {
            "documents": len(documents),
            "kinds": sorted({item["kind"] for item in documents}),
        }
        for document in documents:
            if document["kind"] == policies.KIND_PATTERN_SPECS:
                config_report["pattern_audit"] = policies.audit_pattern_declarations(
                    document["object"]
                )
    except Exception as exc:  # noqa: BLE001 - report, never fake a pass
        config_report = {"error": f"{type(exc).__name__}: {exc}"}
    checks["configs"] = config_report

    coverage = telemetry.c6_c7_summary()
    projected = telemetry.project_c6(
        "self-check",
        telemetry.S06ResultFields(
            compile_mode="dynamo",
            graph_identity="g",
            compile_identity="c",
            cache_layer="dynamo_code",
            cache_hit=True,
            requested_lowering="hqsb.cuda.fused_add_rms_norm",
            actual_lowering="hqsb.cuda.fused_add_rms_norm",
            observed_kernel="hqsb_fused_add_rms_norm_v1",
            correctness_status="not_run",
            raw_artifacts={"graph": "artifact://graph"},
        ),
    )
    checks["telemetry"] = {
        "c6_ok": coverage["c6"]["ok"],
        "c7_ok": coverage["c7"]["ok"],
        "c6_summary_keys": sorted(projected.summary.get("s06", {})),
    }

    if as_json:
        print(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    else:
        print("[S06] self-check (smoke; NOT an experiment result)")
        for name, payload in checks.items():
            if name in ("label", "stage"):
                continue
            print(f"  {name}: {json.dumps(payload, ensure_ascii=False, default=str)}")
    return checks


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
                "S06 cannot execute: the upstream evidence chain (S04.5 M4, S05 P0, "
                "frozen six-workload baseline, recorded environment fingerprint) is "
                "unmet. The interface code is ready and its self-check is available "
                "via --mode self-check."
            ),
        }
        print(json.dumps(report, indent=2) if as_json else report["reason"])
        return report
    # Prerequisites satisfied: the environment-specific execution is still
    # deliberately not implemented here — the driver must be extended with the
    # concrete runner for the target machine rather than silently "succeeding".
    raise UsageError(
        "prerequisites are satisfied but no real runner is wired in this tree; "
        "extend scripts/integration/run_e06.py with the hardware-specific runner "
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
