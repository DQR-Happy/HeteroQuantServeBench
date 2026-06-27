#!/usr/bin/env python3
"""S10 experiment driver — interface entry point, not an experiment result.

This driver implements the full "preregistration → collect → write → decide"
structure for every S10 experiment, but it is **forbidden by default from
producing a conclusion**.  Concretely:

* without ``--execute`` any verdict stays ``BLOCKED`` (the prerequisites are
  unmet in this repository state: no S07 P0 verdict, no S08 trace reference,
  fewer than two accelerators, no sealed topology manifest, no frozen backend
  identity, no frozen model/workload pair);
* with ``--execute`` it still refuses without satisfied prerequisites *and* raw
  samples — and this host has a single accelerator, so no collective/TP/scaling
  run can be executed at all;
* ``--smoke`` runs a **CPU-only self-check** of the interfaces (loopback
  collectives, oracles, preflight fixtures, plan derivation, MoE permutation,
  interval algebra).  It is explicitly labelled ``smoke``, carries
  ``claim_allowed=False`` and is *not* an experiment.

Usage:
    scripts/distributed/run_e10.py --list
    scripts/distributed/run_e10.py --prerequisites
    scripts/distributed/run_e10.py --experiment E10-02 --prerequisites
    scripts/distributed/run_e10.py --interface-map
    scripts/distributed/run_e10.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.distributed import experiment as exp  # noqa: E402
from hqsb.distributed import interface_map as imap  # noqa: E402

EXPERIMENT_TITLES: Dict[str, str] = {
    mapping.experiment_id: mapping.title for mapping in imap.EXPERIMENTS
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S10 experiment driver (interface entry, no conclusions by default)"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_TITLES), help="experiment id")
    parser.add_argument("--list", action="store_true", help="list all experiments and steps")
    parser.add_argument("--prerequisites", action="store_true", help="print the prerequisite status")
    parser.add_argument("--interface-map", action="store_true", help="print the step→interface table")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the CPU-only interface self-check (labelled smoke, not an experiment)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "explicitly allow experiment execution; still refuses a conclusion without "
            "satisfied prerequisites and raw samples"
        ),
    )
    parser.add_argument("--run-id", default="", help="override the run id (default: timestamp)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def _emit(payload: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))
    else:
        for key, value in payload.items():
            if isinstance(value, (list, dict)):
                print(f"{key}:")
                print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))
            else:
                print(f"{key}: {value}")


def cmd_list(as_json: bool) -> None:
    rows = [
        {
            "experiment_id": mapping.experiment_id,
            "title": mapping.title,
            "steps": len(mapping.steps),
            "driver": mapping.driver,
        }
        for mapping in imap.EXPERIMENTS
    ]
    _emit(
        {
            "experiments": rows,
            "total_steps": sum(len(mapping.steps) for mapping in imap.EXPERIMENTS),
            "note": (
                "all experiments are interface-complete; none has been executed and none "
                "carries a conclusion in this repository state"
            ),
        },
        as_json,
    )


def cmd_prerequisites(as_json: bool) -> None:
    _emit(exp.check_prerequisites(REPO_ROOT).as_dict(), as_json)


def cmd_interface_map(experiment_id: Optional[str], as_json: bool) -> None:
    if experiment_id:
        _emit(imap.mapping_for(experiment_id).as_dict(), as_json)
        return
    _emit(imap.resolve_interfaces(), as_json)


def _smoke_self_check() -> Dict[str, Any]:
    """CPU-only interface self-check; never an experiment, never a performance number."""
    from hqsb.distributed import collectives as coll
    from hqsb.distributed import faults as fault_mod
    from hqsb.distributed import moe as moe_mod
    from hqsb.distributed import overlap as overlap_mod
    from hqsb.distributed import parallel_plan as plan_mod
    from hqsb.distributed import placement as placement_mod
    from hqsb.distributed import probes as probes_mod
    from hqsb.distributed import scaling as scaling_mod
    from hqsb.distributed import topology as topology_mod
    from hqsb.distributed.ranks import RankIdentity

    results: Dict[str, Any] = {"status": "smoke", "claim_allowed": False}

    # 1) topology manifest + placement plan (synthetic two-device fixture)
    scope = topology_mod.ObservationScope(branch="cuda_nccl", node_scope="single_node")
    host = topology_mod.HostIdentity(
        node_id="smoke-node",
        host_alias="fixture-host",
        cpu_arch="x86_64",
        os_release="fixture",
        kernel="fixture",
    )
    numa = topology_mod.NumaTopology(
        nodes=(topology_mod.NumaNode(0, 0, (0, 1), 1 << 34, 1 << 33),),
        core_to_numa={0: 0, 1: 0},
    )
    accelerators = tuple(
        topology_mod.AcceleratorRecord(
            device_id=index,
            sku="fixture-accelerator",
            uuid=f"GPU-SMOKE-{index}",
            pci_bdf=f"0000:0{index}:00.0",
            memory_bytes=24 << 30,
            health="OK",
        )
        for index in range(2)
    )
    nodes = tuple(
        topology_mod.TopologyNode(node_id=f"dev{index}", kind="accelerator")
        for index in range(2)
    )
    edges = (
        topology_mod.TopologyEdge(
            src="dev0",
            dst="dev1",
            edge_type="nvlink",
            status="UP",
            confidence="MEASURED",
            source_tool="fixture",
            measured_latency_us=1.5,
            measured_bandwidth_gbps=400.0,
        ),
    )
    manifest = topology_mod.TopologyManifest(
        scope=scope,
        host=host,
        numa=numa,
        accelerators=accelerators,
        nodes=nodes,
        edges=edges,
        collected_at="fixture",
    )
    results["manifest_errors"] = manifest.validate()
    results["manifest_sha256_prefix"] = manifest.sha256[:12]

    plan = placement_mod.PlacementPlan(
        plan_id="smoke-plan",
        world_size=2,
        node_count=1,
        entries=tuple(
            placement_mod.RankPlacement(
                global_rank=index,
                local_rank=index,
                node_rank=0,
                coordinate=placement_mod.ParallelCoordinate(tp=index),
                planned_device_uuid=f"GPU-SMOKE-{index}",
            )
            for index in range(2)
        ),
        source_topology_hash=manifest.sha256,
    )
    results["plan_errors"] = plan.validate()
    results["gate"] = topology_mod.gate_for_collective_probe(
        manifest, placement_ok=True, data_path_verified=True
    ).as_dict()

    # 2) preflight negative fixture: duplicate device must be rejected before init
    duplicate = probes_mod.make_duplicate_device_identities(plan)
    preflight = probes_mod.preflight_rank_mapping(
        plan, duplicate, known_device_uuids=[f"GPU-SMOKE-{i}" for i in range(2)]
    )
    results["duplicate_device_rejected"] = not preflight.ok
    results["duplicate_device_codes"] = [
        item.code for item in preflight.rejections
    ]

    # 3) loopback collectives vs the CPU oracles
    executor = coll.LoopbackCollectiveExecutor(world_size=2)
    inputs = {rank: coll.vector_for_rank(rank, 8) for rank in range(2)}
    loopback: Dict[str, Any] = {}
    for op in ("all_reduce", "all_gather", "reduce_scatter", "broadcast"):
        spec = coll.spec_for(op, dtype="fp16", root=0 if op == "broadcast" else None)
        outcome = executor.execute(spec, inputs=inputs)
        expected: Any
        if op == "all_reduce":
            expected = coll.oracle_all_reduce([inputs[0], inputs[1]])
        elif op == "all_gather":
            expected = coll.oracle_all_gather([inputs[0], inputs[1]])
        elif op == "reduce_scatter":
            expected = coll.oracle_reduce_scatter([inputs[0], inputs[1]], 0)
        else:
            expected = coll.oracle_broadcast([inputs[0], inputs[1]], 0)
        actual = outcome.outputs_by_rank.get(0, [])
        verdict = coll.compare_vectors([float(v) for v in expected], [float(v) for v in actual], tolerance=0.0)
        loopback[op] = {"match": verdict.ok, "simulated": outcome.simulated}
    results["loopback_collectives"] = loopback
    results["loopback_claim_allowed"] = coll.LoopbackResult(op="all_reduce", outputs_by_rank={}).claim_allowed()

    # 4) fault oracle: a correct failure passes, a cheating one fails
    oracle = fault_mod.oracle_for_fault(
        "op_mismatch", maximum_detection_time_s=5.0, maximum_global_abort_time_s=10.0
    )
    good = fault_mod.FailureObservation(
        per_rank_terminal_state={0: "ABORTED", 1: "ABORTED"},
        first_error_rank=0,
        detected_at_layer="init_timeout_or_preflight",
        normalized_error="hqsb.collective.mismatch",
        recovery_level_used="COMMUNICATOR_RECREATE",
    )
    bad = fault_mod.FailureObservation(
        per_rank_terminal_state={0: "ABORTED", 1: "ABORTED"},
        first_error_rank=0,
        detected_at_layer="init_timeout_or_preflight",
        normalized_error="hqsb.collective.mismatch",
        recovery_level_used="COMMUNICATOR_RECREATE",
        communicator_reused_after_error=True,
    )
    times = fault_mod.TimeMetrics(
        fault_injected_s=0.0,
        first_local_detection_s=0.5,
        global_failure_decision_s=1.0,
        all_ranks_abort_started_s=1.2,
        last_rank_exited_or_clean_s=1.5,
        resources_reclaimed_s=2.0,
        restart_started_s=2.5,
        ready_s=4.0,
        first_healthy_request_done_s=5.0,
    )
    results["fault_oracle_ok"] = fault_mod.evaluate_fault(oracle, good, times, world_size=2).ok
    cheating = fault_mod.evaluate_fault(oracle, bad, times, world_size=2)
    results["fault_oracle_rejects_reuse"] = (not cheating.ok) and bool(cheating.failures)

    # 5) TP plan derivation on a tiny synthetic census
    census = plan_mod.census_from_mapping(
        {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "vocab_size": 128,
        }
    )
    tp_plan = plan_mod.derive_plan(
        census, plan_id="smoke-tp", model_manifest_sha256="fixture", degree=2, ordered_ranks=[0, 1]
    )
    results["tp_plan_errors"] = tp_plan.validate(census)
    results["tp_plan_sha256_prefix"] = tp_plan.sha256[:12]

    # 6) MoE dispatch/combine permutation conservation
    route = moe_mod.generate_route_artifact(
        artifact_id="smoke-route",
        token_count=8,
        world_size=2,
        experts_per_rank=2,
        top_k=2,
        profile="severe_zipf",
        seed=7,
    )
    placement = moe_mod.round_robin_placement(num_experts=4, world_size=2)
    matrix = moe_mod.count_matrix(route, expert_to_rank=placement.expert_to_rank, world_size=2)
    dispatch = moe_mod.dispatch_oracle(route)
    outputs = {
        position: moe_mod.deterministic_stub_expert(
            position=position,
            expert_id=dispatch.expert_order[position],
            hidden_size=census.hidden_size,
        )
        for position in range(dispatch.packed_count)
    }
    combined = moe_mod.combine_oracle(dispatch, expert_outputs=outputs, hidden_size=census.hidden_size)
    results["moe_conservation"] = matrix.audit()["ok"]
    results["moe_token_conservation"] = bool(combined["token_conservation"])

    # 7) overlap interval algebra on a synthetic timeline
    compute = [overlap_mod.Interval(0, 1_000_000, kind="compute")]
    comm = [overlap_mod.Interval(500_000, 1_500_000, kind="comm")]
    metrics = overlap_mod.overlap_metrics_from_intervals(
        compute, comm, no_overlap_total_ms=1.5, overlap_total_ms=1.0
    )
    results["overlap_ms"] = metrics.overlap_ms
    results["overlap_conclusive"] = metrics.conclusive

    # 8) scaling refuses a fake T1
    baseline = scaling_mod.baseline_calibration(t1_latency_ms=None, reason="model does not fit")
    speedup = scaling_mod.strong_speedup(baseline, degree=2, latency_ms=100.0)
    results["scaling_metric_name"] = speedup.metric_name
    results["scaling_refused_fake_t1"] = speedup.metric_name == "speedup_from_p0" and speedup.speedup is None

    # 9) rank identity invariant on the fixture
    identities = [
        RankIdentity(
            global_rank=index,
            local_rank=index,
            node_rank=0,
            node_id="smoke-node",
            pid=1000 + index,
            device_uuid=f"GPU-SMOKE-{index}",
        )
        for index in range(2)
    ]
    results["rank_identity_ok"] = __import__(
        "hqsb.distributed.ranks", fromlist=["validate_rank_identities"]
    ).validate_rank_identities(identities, expected_world_size=2)["ok"]

    results["note"] = (
        "CPU-only self-check of the interfaces; it is NOT an experiment and carries no "
        "collective/TP/scaling/overlap/quality conclusion"
    )
    return results


def cmd_smoke(as_json: bool) -> None:
    _emit(_smoke_self_check(), as_json)


def cmd_experiment(
    experiment_id: str,
    *,
    prerequisites_only: bool,
    execute: bool,
    run_id: str,
    as_json: bool,
) -> None:
    status = exp.check_prerequisites(REPO_ROOT)
    if prerequisites_only:
        _emit(
            {
                "experiment_id": experiment_id,
                "prerequisites_satisfied": status.satisfied,
                "prerequisites_missing": status.missing,
                "checks": [check.as_dict() for check in status.checks],
            },
            as_json,
        )
        return
    run = exp.RunDirectory(REPO_ROOT, experiment_id, run_id or exp.STATUS_NOT_STARTED)
    # Never write into docs/stage_experiments: only experiment_results/S10/<E>/<run>/.
    run.create()
    run.write_json("environment_fingerprint.json", exp.environment_fingerprint())
    run.write_json("prerequisites.json", status.as_dict())
    run.write_json("interface_map.json", imap.mapping_for(experiment_id).as_dict())
    run.write_report_skeleton(experiment_id, EXPERIMENT_TITLES[experiment_id])
    if not status.satisfied:
        verdict = run.write_verdict(
            status=exp.STATUS_BLOCKED,
            reason="prerequisites unsatisfied: " + ", ".join(status.missing),
            prerequisites=status,
            executed=False,
            allow_execute=execute,
            raw_samples=0,
        )
    else:
        verdict = run.write_status(
            exp.STATUS_NOT_STARTED,
            "prerequisites satisfied but no experiment has been executed; see the "
            "development report for the interface layer",
            status,
        )
    _emit(
        {
            "experiment_id": experiment_id,
            "run_dir": run.path,
            "status": verdict["status"],
            "reason": verdict.get("reason", ""),
            "prerequisites_satisfied": status.satisfied,
            "prerequisites_missing": status.missing,
            "execution_allowed": execute,
        },
        as_json,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.list:
        cmd_list(args.json)
        return 0
    if args.interface_map:
        cmd_interface_map(args.experiment, args.json)
        return 0
    if args.experiment is None:
        if args.smoke:
            cmd_smoke(args.json)
            return 0
        _emit(
            {
                "status": exp.STATUS_BLOCKED,
                "reason": (
                    "no experiment selected; the S10 prerequisites are unmet, so no conclusion "
                    "may be produced. Use --list, --smoke, --prerequisites or "
                    "--experiment E10-xx --prerequisites"
                ),
            },
            args.json,
        )
        return 0
    if args.smoke:
        cmd_smoke(args.json)
        return 0
    if args.prerequisites:
        cmd_experiment(
            args.experiment,
            prerequisites_only=True,
            execute=args.execute,
            run_id=args.run_id,
            as_json=args.json,
        )
        return 0
    cmd_experiment(
        args.experiment,
        prerequisites_only=False,
        execute=args.execute,
        run_id=args.run_id,
        as_json=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
