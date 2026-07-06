#!/usr/bin/env python3
"""S13 experiment driver — interface entry point, not an experiment result.

This driver implements the full "preregister → collect → write → decide" structure
for every S13 experiment, but it is **forbidden by default from producing a
conclusion**.  Concretely:

* without ``--execute`` any verdict stays ``BLOCKED``;
* with ``--execute`` it still refuses without satisfied prerequisites *and* raw
  samples;
* ``--smoke`` runs a **CPU-only self-check** of the interfaces (release identity,
  OCI DAG compare, supply-chain gate, deployment state machine, readiness claim,
  placement filter, artifact commit/GC, drain timeline, admission, autoscaling
  decisions, semantic conventions, fault contract, canary gates, tenant cases).
  It is explicitly labelled ``smoke``, carries ``claim_allowed=False`` and is *not*
  an experiment;
* nothing is ever written into ``docs/stage_experiments/**`` (the protocol tree is
  read-only).

Usage:
    scripts/infra/run_e13.py --list
    scripts/infra/run_e13.py --prerequisites
    scripts/infra/run_e13.py --interface-map
    scripts/infra/run_e13.py --spec-audit
    scripts/infra/run_e13.py --smoke
    scripts/infra/run_e13.py --experiment E13-02 --prerequisites
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.infra import experiment as exp  # noqa: E402
from hqsb.infra import interface_map as imap  # noqa: E402
from hqsb.infra import records as rec  # noqa: E402
from hqsb.infra import specs as spec_mod  # noqa: E402

EXPERIMENT_TITLES: Dict[str, str] = {mapping.experiment_id: mapping.title for mapping in imap.EXPERIMENTS}

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "infra")

#: Experiment module of each experiment, for the smoke self-check (a broken module
#: is reported, never silently skipped).
SMOKE_MODULES: Dict[str, str] = {
    "E13-01": "hqsb.infra.supply_chain",
    "E13-02": "hqsb.infra.deployment",
    "E13-03": "hqsb.infra.scheduling",
    "E13-04": "hqsb.infra.artifacts",
    "E13-05": "hqsb.infra.lifecycle",
    "E13-06": "hqsb.infra.capacity",
    "E13-07": "hqsb.infra.autoscaling",
    "E13-08": "hqsb.infra.observability",
    "E13-09": "hqsb.infra.faults",
    "E13-10": "hqsb.infra.canary",
    "E13-11": "hqsb.infra.security",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S13 experiment driver (interface entry, no conclusions by default)"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_TITLES), help="experiment id")
    parser.add_argument("--list", action="store_true", help="list all experiments and steps")
    parser.add_argument("--prerequisites", action="store_true", help="print the prerequisite status")
    parser.add_argument("--interface-map", action="store_true", help="print the step→interface table")
    parser.add_argument("--spec-audit", action="store_true", help="audit configs/infra documents")
    parser.add_argument("--spec-check", action="store_true", help="regenerate check for configs/infra")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the CPU-only interface self-check (labelled smoke, not an experiment)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="also probe the machine for cluster/scan/telemetry tooling when checking prerequisites",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "explicitly allow experiment execution; still refuses a conclusion without satisfied "
            "prerequisites and raw samples"
        ),
    )
    parser.add_argument("--run-id", default="", help="override the run id (default: interface_only)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def _emit(payload: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))
        return
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
                "all experiments are interface-complete; none has been executed and none carries a "
                "conclusion in this repository state"
            ),
        },
        as_json,
    )


def cmd_prerequisites(experiment_id: Optional[str], as_json: bool, probe: bool) -> None:
    status = exp.check_prerequisites(REPO_ROOT, probe=probe)
    payload = status.as_dict()
    if experiment_id:
        payload["experiment_id"] = experiment_id
    _emit(payload, as_json)


def cmd_interface_map(experiment_id: Optional[str], as_json: bool) -> None:
    if experiment_id:
        _emit(imap.mapping_for(experiment_id).as_dict(), as_json)
        return
    result = imap.resolve_interfaces()
    result["steps_without_interfaces"] = imap.steps_without_interfaces()
    result["ok"] = result["ok"] and not result["steps_without_interfaces"]
    _emit(result, as_json)


def cmd_spec_audit(as_json: bool) -> None:
    documents = spec_mod.InfraSpecs.load(SPEC_DIR)
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


def cmd_spec_check(as_json: bool) -> None:
    from scripts.infra.gen_infra_specs import check_documents  # type: ignore[import-not-found]

    ok, drifted = check_documents(SPEC_DIR)
    _emit({"spec_dir": SPEC_DIR, "ok": ok, "drifted": drifted}, as_json)


def cmd_smoke(as_json: bool) -> None:
    results: Dict[str, Any] = {"status": "smoke", "claim_allowed": False, "modules": {}}
    for experiment_id, module_name in SMOKE_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            results["modules"][experiment_id] = getattr(module, "smoke_self_check")()
        except Exception as exc:  # noqa: BLE001 - reported, never skipped
            results["modules"][experiment_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    interface_result = imap.resolve_interfaces()
    spec_reports = spec_mod.InfraSpecs.load(SPEC_DIR).audit()
    foundation = {
        "identity": "hqsb.infra.identity",
        "records": "hqsb.infra.records",
        "contracts": "hqsb.infra.contracts",
        "campaign": "hqsb.infra.campaign",
        "telemetry": "hqsb.infra.telemetry",
    }
    for name, module_name in foundation.items():
        try:
            module = importlib.import_module(module_name)
            checks = {}
            if hasattr(module, "smoke_self_check"):
                checks = getattr(module, "smoke_self_check")()
            elif name == "records":
                checks = {"state_machines": rec.validate_state_machines() == [],
                          "tables": len(rec.TABLE_SCHEMAS)}
            elif name == "identity":
                checks = {"digest_pattern": True}
            results["modules"][f"foundation:{name}"] = checks
        except Exception as exc:  # noqa: BLE001
            results["modules"][f"foundation:{name}"] = {"status": "error",
                                                        "error": f"{type(exc).__name__}: {exc}"}
    results["interface_map"] = {
        "steps": interface_result["steps"],
        "interfaces": interface_result["interfaces"],
        "references": interface_result["references"],
        "ok": interface_result["ok"],
        "failures": interface_result["failures"][:5],
    }
    results["spec_audit_ok"] = spec_mod.audit_all_ok(spec_reports)
    results["state_machines_ok"] = rec.validate_state_machines() == []
    results["note"] = (
        "CPU-only self-check of the interfaces; it is NOT an experiment and carries no deployment/"
        "supply-chain/capacity/observability/fault/canary/security conclusion"
    )
    _emit(results, as_json)


def cmd_experiment(experiment_id: str, *, prerequisites_only: bool, execute: bool,
                   run_id: str, as_json: bool, probe: bool) -> None:
    status = exp.check_prerequisites(REPO_ROOT, probe=probe)
    if prerequisites_only:
        _emit(
            {
                "experiment_id": experiment_id,
                "prerequisites_satisfied": status.satisfied,
                "prerequisites_missing": status.missing,
                "prerequisites_advisory": status.advisory,
                "states": status.states(),
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
            status=rec.STATUS_BLOCKED,
            reason="prerequisites unsatisfied: " + ", ".join(status.missing),
            prerequisites=status,
            executed=False,
            allow_execute=execute,
            raw_samples=0,
        )
    else:
        verdict = run.write_status(
            rec.STATUS_NOT_STARTED,
            "prerequisites satisfied but no experiment has been executed; see the development report "
            "for the interface layer",
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
    if args.spec_check:
        cmd_spec_check(args.json)
        return 0
    if args.smoke:
        cmd_smoke(args.json)
        return 0
    if args.experiment is None:
        if args.prerequisites:
            cmd_prerequisites(None, args.json, args.probe)
            return 0
        _emit(
            {
                "status": rec.STATUS_BLOCKED,
                "reason": (
                    "no experiment selected; S13 experiments may not be executed without a cluster, so no "
                    "conclusion may be produced. Use --list, --prerequisites, --interface-map, "
                    "--spec-audit or --smoke"
                ),
            },
            args.json,
        )
        return 0
    if args.prerequisites:
        cmd_experiment(
            args.experiment, prerequisites_only=True, execute=args.execute, run_id=args.run_id,
            as_json=args.json, probe=args.probe,
        )
        return 0
    cmd_experiment(
        args.experiment, prerequisites_only=False, execute=args.execute, run_id=args.run_id,
        as_json=args.json, probe=args.probe,
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
