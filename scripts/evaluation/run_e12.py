#!/usr/bin/env python3
"""S12 experiment driver — interface entry point, not an experiment result.

This driver implements the full "preregister → collect → write → decide"
structure for every S12 experiment, but it is **forbidden by default from
producing a conclusion**.  Concretely:

* without ``--execute`` any verdict stays ``BLOCKED`` (the S12 prerequisites
  are unmet in this repository state: no S03–S11 PASS verdicts for the upstream
  evidence chain, no multi-hardware coverage, no frozen evaluation environment);
* with ``--execute`` it still refuses without satisfied prerequisites *and*
  raw samples;
* ``--smoke`` runs a **CPU-only self-check** of the interfaces (comparability
  verdicts, capability upgrade/invalidation, forbidden normalizations, balanced
  schedules, roofline roofs, energy integration, cost double-count, Pareto
  regression, rubric anchors, lineage append-only).  It is explicitly labelled
  ``smoke``, carries ``claim_allowed=False`` and is *not* an experiment.

Usage:
    scripts/evaluation/run_e12.py --list
    scripts/evaluation/run_e12.py --prerequisites
    scripts/evaluation/run_e12.py --experiment E12-01 --prerequisites
    scripts/evaluation/run_e12.py --interface-map
    scripts/evaluation/run_e12.py --smoke
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

from hqsb.evaluation import experiment as exp  # noqa: E402
from hqsb.evaluation import interface_map as imap  # noqa: E402
from hqsb.evaluation import specs as spec_mod  # noqa: E402

EXPERIMENT_TITLES: Dict[str, str] = {
    mapping.experiment_id: mapping.title for mapping in imap.EXPERIMENTS
}

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "evaluation")

#: The ten experiment modules' smoke self-checks are aggregated; a broken module
#: is reported, never silently skipped.
SMOKE_MODULES: Dict[str, str] = {
    "E12-01": "hqsb.evaluation.comparability",
    "E12-02": "hqsb.evaluation.capability",
    "E12-03": "hqsb.evaluation.benchmark",
    "E12-04": "hqsb.evaluation.repeatability",
    "E12-05": "hqsb.evaluation.roofline",
    "E12-06": "hqsb.evaluation.energy",
    "E12-07": "hqsb.evaluation.cost",
    "E12-08": "hqsb.evaluation.pareto",
    "E12-09": "hqsb.evaluation.maturity",
    "E12-10": "hqsb.evaluation.lineage",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S12 experiment driver (interface entry, no conclusions by default)"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_TITLES), help="experiment id")
    parser.add_argument("--list", action="store_true", help="list all experiments and steps")
    parser.add_argument("--prerequisites", action="store_true", help="print the prerequisite status")
    parser.add_argument("--interface-map", action="store_true", help="print the step→interface table")
    parser.add_argument("--spec-audit", action="store_true", help="audit configs/evaluation documents")
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
    documents = spec_mod.EvaluationSpecs.load(SPEC_DIR)
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


def cmd_smoke(as_json: bool) -> None:
    results: Dict[str, Any] = {"status": "smoke", "claim_allowed": False, "modules": {}}
    import importlib

    for experiment_id, module_name in SMOKE_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            check = getattr(module, "smoke_self_check")()
            results["modules"][experiment_id] = check
        except Exception as exc:  # noqa: BLE001 - reported, never skipped
            results["modules"][experiment_id] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    interface_result = imap.resolve_interfaces()
    spec_reports = spec_mod.EvaluationSpecs.load(SPEC_DIR).audit()
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
        "comparability/capability/performance/energy/cost/maturity/lineage conclusion"
    )
    _emit(results, as_json)


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
    run = exp.RunDirectory(REPO_ROOT, experiment_id, run_id or "interface_only")
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
            "prerequisites satisfied but no experiment has been executed; see the development "
            "report for the interface layer",
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
                    "no experiment selected; the S12 prerequisites are unmet, so no conclusion "
                    "may be produced. Use --list, --smoke, --prerequisites or "
                    "--experiment E12-xx --prerequisites"
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
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except OSError:
            pass
        raise SystemExit(0) from None
