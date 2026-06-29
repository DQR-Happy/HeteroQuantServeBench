#!/usr/bin/env python3
"""S11 experiment driver — interface entry point, not an experiment result.

This driver implements the full "preregister → collect → write → decide"
structure for every S11 experiment, but it is **forbidden by default from
producing a conclusion**.  Concretely:

* without ``--execute`` any verdict stays ``BLOCKED`` (the S11 prerequisites
  are unmet in this repository state: no S03/S04/S06 PASS verdicts, no frozen
  compiler-environment fingerprint);
* with ``--execute`` it still refuses without satisfied prerequisites *and*
  raw samples;
* ``--smoke`` runs a **CPU-only self-check** of the interfaces (pattern corpus
  with near-miss rejection and idempotence, guard/variant lookup, cache
  transaction + corruption rejection, reference lowering, admission schema).
  It is explicitly labelled ``smoke``, carries ``claim_allowed=False`` and is
  *not* an experiment.

Usage:
    scripts/compiler/run_e11.py --list
    scripts/compiler/run_e11.py --prerequisites
    scripts/compiler/run_e11.py --experiment E11-02 --prerequisites
    scripts/compiler/run_e11.py --interface-map
    scripts/compiler/run_e11.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.compiler import experiment as exp  # noqa: E402
from hqsb.compiler import interface_map as imap  # noqa: E402
from hqsb.compiler import specs as spec_mod  # noqa: E402

EXPERIMENT_TITLES: Dict[str, str] = {
    mapping.experiment_id: mapping.title for mapping in imap.EXPERIMENTS
}

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "compiler")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S11 experiment driver (interface entry, no conclusions by default)"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_TITLES), help="experiment id")
    parser.add_argument("--list", action="store_true", help="list all experiments and steps")
    parser.add_argument("--prerequisites", action="store_true", help="print the prerequisite status")
    parser.add_argument("--interface-map", action="store_true", help="print the step→interface table")
    parser.add_argument("--spec-audit", action="store_true", help="audit configs/compiler documents")
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
            "level": mapping.level,
            "steps": len(mapping.steps),
            "driver": mapping.driver,
            "claim_boundary": mapping.claim_boundary,
        }
        for mapping in imap.EXPERIMENTS
    ]
    _emit(
        {
            "experiments": rows,
            "total_steps": imap.total_steps(),
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


def cmd_spec_audit(as_json: bool) -> None:
    documents = spec_mod.CompilerSpecs.load(SPEC_DIR)
    reports = documents.audit()
    _emit(
        {
            "spec_dir": SPEC_DIR,
            "kinds": len(documents.documents),
            "ok": spec_mod.audit_all_ok(reports),
            "reports": reports,
        },
        as_json,
    )


def _smoke_self_check() -> Dict[str, Any]:
    """CPU-only interface self-check; never an experiment, never a performance number."""
    from hqsb.compiler import aigate as ag
    from hqsb.compiler import cache as ch
    from hqsb.compiler import guards as gd
    from hqsb.compiler import lowering as lw
    from hqsb.compiler import pattern_library as pl
    from hqsb.compiler import rewrite as rw
    from hqsb.compiler import targets as tg

    results: Dict[str, Any] = {"status": "smoke", "claim_allowed": False}

    # 1) pattern corpus: positives match, near-misses are rejected, FP must be 0
    rows = rw.build_corpus_rows(pl.corpus_plan(), graph_builder=pl.residual_add_rmsnorm_graph)
    corpus = rw.evaluate_corpus(rows)
    results["corpus"] = {
        "rows": corpus["rows"],
        "true_positive": corpus["true_positive"],
        "false_positive": corpus["false_positive"],
        "true_negative": corpus["true_negative"],
        "false_negative": corpus["false_negative"],
        "false_positive_zero": corpus["false_positive_zero"],
        "unmet_expectations": corpus["unmet_expectations"],
    }

    # 2) idempotence + atomicity on the canonical fixture
    spec = pl.residual_add_rmsnorm_graph(graph_id="smoke_idem", variant="canonical")
    graph = rw.build_graph(spec)
    idem = rw.idempotence_report(graph)
    results["idempotence"] = {
        "second_pass_rewrites": idem["second_pass_rewrites"],
        "hash_stable": idem["hash_stable"],
        "idempotent": idem["idempotent"],
    }
    results["atomicity_all"] = rw.atomicity_report(graph)["all_atomic"]

    # 3) guard/variant lookup: hit inside the domain, miss outside, no wrong reuse
    guards = (
        gd.range_guard(guard_id="smoke_batch", symbol="B", lower=1, upper=8, source="smoke"),
        gd.equality_guard(guard_id="smoke_dtype", name="dtype", expected="fp16", source="smoke"),
    )
    variant = gd.Variant(
        variant_id="smoke_v1",
        semantic_identity="smoke",
        compile_identity="smoke_cid",
        target_id="cpu",
        artifact_id="smoke_artifact",
        guards=guards,
        priority=1,
        created_reason="smoke fixture",
    )
    registry = gd.VariantRegistry()
    registry.add(variant)
    inside = registry.lookup(semantic_identity="smoke", inputs={"B": 4, "dtype": "fp16"})
    outside = registry.lookup(semantic_identity="smoke", inputs={"B": 9, "dtype": "fp16"})
    results["variant_lookup"] = {
        "inside": inside.outcome,
        "outside": outside.outcome,
        "outside_failed_guards": list(outside.failed_guards),
        "guard_false_is_not_a_hit": outside.outcome != "hit",
    }

    # 4) reference lowering selection + materialize + debug-backend refusal
    store = lw.LoweringRegistry()
    store.register(lw.reference_lowering_entry(fallback_id=""))
    evidence = lw.EvidenceIndex(
        [
            lw.CorrectnessEvidence(
                evidence_id="smoke_ev",
                level="operator",
                implementation_id="hqsb.reference.fused_add_rms_norm",
                dtypes=("fp16", "fp32", "bf16"),
                tolerance_policy_id="common_s06",
                status="pass",
                raw_ref="smoke://raw",
            )
        ]
    )
    decision = lw.evaluate_candidates(
        registry=store,
        semantic_op="hqsb::fused_add_rms_norm",
        schema_version="1.0.0",
        target=tg.cpu_target_snapshot(),
        evidence=evidence,
        inputs={},
        policy="reference",
    )
    plan = lw.materialize(decision, store)
    results["lowering"] = {
        "selected": decision.selected,
        "materialize_status": plan.status,
        "fallback": plan.fallback_implementation,
        "decision_problems": decision.validate(),
    }

    # 5) cache transaction + corruption rejection on an isolated temp store
    key_spec = ch.default_key_spec()
    parts = {name: f"{name}:v1" for name in key_spec.fields}
    with tempfile.TemporaryDirectory() as root:
        entry_store = ch.EntryStore(root, spec=key_spec)
        publish = entry_store.publish(
            entry_id="smoke_entry",
            key=key_spec.compute(parts)["key"],
            layer="pass_ir",
            payload=b"smoke-payload",
            target_arch="cpu",
            abi_version="1",
            guard_domain="B<=8",
        )
        hit = entry_store.read(
            "smoke_entry",
            expected_key=key_spec.compute(parts)["key"],
            target_arch="cpu",
            abi_version="1",
        )
        ch.inject_corruption(entry_store, "smoke_entry", "payload_bit_flip")
        corrupt = entry_store.read(
            "smoke_entry",
            expected_key=key_spec.compute(parts)["key"],
            target_arch="cpu",
            abi_version="1",
        )
    results["cache"] = {
        "publish_state": publish["state"],
        "stages": publish["stages"],
        "hit_status": hit.status,
        "corrupt_status": corrupt.status,
        "corrupt_reject_reason": corrupt.reason_code,
        "corrupt_load_calls": corrupt.load_calls,
        "corrupt_not_executed": corrupt.status == "reject" and corrupt.load_calls == 0,
        "key_vectors_ok": ch.evaluate_key_vectors(key_spec)["all_ok"],
        "key_stability": ch.key_stability(key_spec, parts=parts)["stable"],
    }

    # 6) AI-gate: wrong-corpus template complete, admission schema rejects drafts
    results["aigate"] = {
        "wrong_corpus_complete": ag.wrong_corpus_template()["complete"],
        "admission_draft_status": ag.validate_admission({"candidate_id": "smoke"})["status"],
        "fast_p_requires_correctness": ag.fast_p(
            [{"correct": False, "speedup": 99.0}], p=1.2
        )["fast_p"],
        "claim_boundary": ag.claim_boundary(admitted_candidates=0, methodology_pass=True)["text"],
    }

    # 7) interface map + spec audit
    interface_result = imap.resolve_interfaces()
    spec_reports = spec_mod.CompilerSpecs.load(SPEC_DIR).audit()
    results["interface_map"] = {
        "steps": interface_result["steps"],
        "interfaces": interface_result["interfaces"],
        "references": interface_result["references"],
        "ok": interface_result["ok"],
        "failures": interface_result["failures"],
    }
    results["spec_audit_ok"] = spec_mod.audit_all_ok(spec_reports)

    results["note"] = (
        "CPU-only self-check of the interfaces; it is NOT an experiment and carries no "
        "capture/lowering/autotune/cache/performance conclusion"
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
                "prerequisites_advisory": status.advisory,
                "checks": [check.as_dict() for check in status.checks],
            },
            as_json,
        )
        return
    run = exp.RunDirectory(REPO_ROOT, experiment_id, run_id or exp.STATUS_NOT_STARTED)
    # Never write into docs/stage_experiments: only experiment_results/S11/<E>/<run>/.
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
    if args.spec_audit:
        cmd_spec_audit(args.json)
        return 0
    if args.experiment is None:
        if args.smoke:
            cmd_smoke(args.json)
            return 0
        if args.prerequisites:
            cmd_prerequisites(args.json)
            return 0
        _emit(
            {
                "status": exp.STATUS_BLOCKED,
                "reason": (
                    "no experiment selected; the S11 prerequisites are unmet, so no conclusion "
                    "may be produced. Use --list, --smoke, --prerequisites or "
                    "--experiment E11-xx --prerequisites"
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
    try:
        raise SystemExit(main())
    except BrokenPipeError:  # a downstream `| head` closed the pipe: not an error
        try:
            sys.stdout.close()
        except OSError:
            pass
        raise SystemExit(0) from None
