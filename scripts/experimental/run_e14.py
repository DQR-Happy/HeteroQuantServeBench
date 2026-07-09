#!/usr/bin/env python3
"""S14 experiment driver — interface entry point, not an experiment result.

This driver implements the full "preregister → collect → write → decide" structure
for every S14 experiment, but it is **forbidden by default from producing a
conclusion**.  Concretely:

* without ``--execute`` any verdict stays ``BLOCKED``;
* with ``--execute`` it still refuses without satisfied prerequisites *and* raw
  samples;
* ``--smoke`` runs a **CPU-only self-check** of the interfaces (dependency
  boundary, distributed training state, conversion parity, post-training
  oracles, frontier preregistration, the four F branches, the three optional
  transfers).  It is explicitly labelled ``smoke``, carries
  ``claim_allowed=False`` and is *not* an experiment;
* nothing is ever written into ``docs/stage_experiments/**`` (the protocol tree is
  read-only — ``hqsb.experimental.campaign.assert_writable`` enforces it).

Usage:
    scripts/experimental/run_e14.py --list
    scripts/experimental/run_e14.py --prerequisites
    scripts/experimental/run_e14.py --interface-map
    scripts/experimental/run_e14.py --objects
    scripts/experimental/run_e14.py --spec-audit
    scripts/experimental/run_e14.py --smoke
    scripts/experimental/run_e14.py --experiment E14-03 --prerequisites
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

from hqsb.experimental import contracts as ct  # noqa: E402
from hqsb.experimental import experiment as exp  # noqa: E402
from hqsb.experimental import interface_map as imap  # noqa: E402
from hqsb.experimental import records as rec  # noqa: E402
from hqsb.experimental import specs as spec_mod  # noqa: E402

EXPERIMENT_TITLES: Dict[str, str] = {mapping.experiment_id: mapping.title for mapping in imap.EXPERIMENTS}

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "experimental")

#: Experiment module of each experiment for the smoke self-check (a broken module
#: is reported, never silently skipped).
SMOKE_MODULES: Dict[str, str] = {
    experiment_id: f"hqsb.experimental.{name}"
    for name, experiment_id, _level in imap.EXPERIMENT_MODULES
}

#: Foundations checked alongside the experiments.
FOUNDATION_MODULES: Dict[str, str] = {
    "identity": "hqsb.experimental.identity",
    "records": "hqsb.experimental.records",
    "contracts": "hqsb.experimental.contracts",
    "campaign": "hqsb.experimental.campaign",
    "telemetry": "hqsb.experimental.telemetry",
    "specs": "hqsb.experimental.specs",
    "experiment": "hqsb.experimental.experiment",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S14 experiment driver (interface entry, no conclusions by default)"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_TITLES), help="experiment id")
    parser.add_argument("--list", action="store_true", help="list all experiments and steps")
    parser.add_argument("--prerequisites", action="store_true", help="print the prerequisite status")
    parser.add_argument("--interface-map", action="store_true", help="print the step→interface table")
    parser.add_argument("--objects", action="store_true", help="list the seven unified evidence objects")
    parser.add_argument("--spec-audit", action="store_true", help="audit configs/experimental documents")
    parser.add_argument("--spec-check", action="store_true", help="regenerate check for configs/experimental")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the CPU-only interface self-check (labelled smoke, not an experiment)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="also probe the machine for launcher/device/profiler tooling when checking prerequisites",
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
            "expected_steps": imap.EXPERIMENTS and len(imap.EXPERIMENTS) * imap.EXPECTED_STEPS_PER_EXPERIMENT,
            "note": (
                "all twelve experiments are interface-complete; none has been executed and none carries "
                "a conclusion in this repository state"
            ),
        },
        as_json,
    )


def cmd_prerequisites(experiment_id: Optional[str], as_json: bool, probe: bool) -> None:
    status = exp.check_prerequisites(REPO_ROOT, probe=probe)
    payload = status.as_dict()
    if experiment_id:
        payload["experiment_id"] = experiment_id
        payload["upstream_chain"] = list(rec.EXPERIMENT_DEPENDENCIES.get(experiment_id, ()))
    _emit(payload, as_json)


def cmd_interface_map(experiment_id: Optional[str], as_json: bool) -> None:
    if experiment_id:
        _emit(imap.mapping_for(experiment_id).as_dict(), as_json)
        return
    result = imap.resolve_interfaces()
    result["steps_without_interfaces"] = imap.steps_without_interfaces()
    result["coverage"] = imap.coverage_summary()
    result["ok"] = result["ok"] and not result["steps_without_interfaces"]
    _emit(result, as_json)


def cmd_objects(as_json: bool) -> None:
    _emit(
        {
            "objects": ct.EVIDENCE_OBJECTS,
            "count": len(ct.EVIDENCE_OBJECTS),
            "note": "§7 的七个统一证据对象；契约只校验结构，不产生测量值",
        },
        as_json,
    )


def cmd_spec_audit(as_json: bool) -> None:
    documents = spec_mod.ExperimentalSpecs.load(SPEC_DIR)
    reports = documents.audit()
    cross = spec_mod.check_all(documents.documents)
    _emit(
        {
            "spec_dir": SPEC_DIR,
            "kinds": len(documents.documents),
            "missing_kinds": documents.missing_kinds(),
            "ok": spec_mod.audit_all_ok(reports) and not cross,
            "reports": reports,
            "cross_document_problems": cross,
        },
        as_json,
    )


def cmd_spec_check(as_json: bool) -> None:
    from scripts.experimental.gen_experimental_specs import check_documents  # type: ignore[import-not-found]

    ok, drifted = check_documents(SPEC_DIR)
    _emit({"spec_dir": SPEC_DIR, "ok": ok, "drifted": drifted}, as_json)


def cmd_smoke(as_json: bool) -> None:
    results: Dict[str, Any] = {"status": "smoke", "claim_allowed": False, "modules": {}}
    for experiment_id, module_name in sorted(SMOKE_MODULES.items()):
        try:
            module = importlib.import_module(module_name)
            results["modules"][experiment_id] = getattr(module, "smoke_self_check")()
        except Exception as exc:  # noqa: BLE001 - reported, never skipped
            results["modules"][experiment_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    for name, module_name in sorted(FOUNDATION_MODULES.items()):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "smoke_self_check"):
                results["modules"][f"foundation:{name}"] = getattr(module, "smoke_self_check")()
            elif name == "records":
                results["modules"][f"foundation:{name}"] = {
                    "state_machines": rec.validate_state_machines(),
                    "tables": len(rec.TABLE_SCHEMAS),
                    "statuses": len(rec.ALL_STATUSES),
                    "maturity_levels": len(rec.MATURITY_LEVELS),
                }
        except Exception as exc:  # noqa: BLE001
            results["modules"][f"foundation:{name}"] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}"
            }
    interface_result = imap.resolve_interfaces()
    spec_reports = spec_mod.ExperimentalSpecs.load(SPEC_DIR).audit()
    results["interface_map"] = {
        "experiments": interface_result["experiments"],
        "steps": interface_result["steps"],
        "expected_steps": interface_result["expected_steps"],
        "interfaces": interface_result["interfaces"],
        "references": interface_result["references"],
        "ok": interface_result["ok"],
        "failures": interface_result["failures"][:5],
    }
    results["spec_audit_ok"] = spec_mod.audit_all_ok(spec_reports)
    results["state_machines_ok"] = rec.validate_state_machines() == []
    results["note"] = (
        "CPU-only self-check of the interfaces; it is NOT an experiment and carries no training/"
        "conversion/rollout/frontier/transfer conclusion"
    )
    _emit(results, as_json)


def cmd_experiment(
    experiment_id: str, *, prerequisites_only: bool, execute: bool, run_id: str, as_json: bool, probe: bool
) -> None:
    status = exp.check_prerequisites(REPO_ROOT, probe=probe)
    if prerequisites_only:
        _emit(
            {
                "experiment_id": experiment_id,
                "prerequisites_satisfied": status.satisfied,
                "prerequisites_missing": status.missing,
                "prerequisites_advisory": status.advisory,
                "states": status.states(),
                "depends_on": list(rec.EXPERIMENT_DEPENDENCIES.get(experiment_id, ())),
                "checks": [check.as_dict() for check in status.checks],
            },
            as_json,
        )
        return
    run = exp.RunDirectory(REPO_ROOT, experiment_id, run_id or "interface_only")
    run.create()
    run.write_json("environment_fingerprint.json", exp.environment_fingerprint(REPO_ROOT))
    run.write_json("prerequisites.json", status.as_dict())
    run.write_json("interface_map.json", imap.mapping_for(experiment_id).as_dict())
    run.write_json("source_identity.json", exp.git_identity(REPO_ROOT))
    run.write_json(
        "experiment_record.json", exp.build_experiment_record(experiment_id, git=exp.git_identity(REPO_ROOT))
    )
    run.write_report_skeleton(EXPERIMENT_TITLES[experiment_id])
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
    if args.objects:
        cmd_objects(args.json)
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
                    "no experiment selected; S14 experiments may not be executed without experimental "
                    "prerequisites, so no conclusion may be produced. Use --list, --prerequisites, "
                    "--interface-map, --objects, --spec-audit or --smoke"
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
