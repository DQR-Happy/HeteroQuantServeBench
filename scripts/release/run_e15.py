#!/usr/bin/env python3
"""S15 experiment driver — interface entry point, not an experiment result.

This driver implements the full "preregister → collect → write → decide" structure
for every S15 experiment, but it is **forbidden by default from producing a
conclusion**.  Concretely:

* without ``--execute`` any verdict stays ``BLOCKED``;
* with ``--execute`` it still refuses without satisfied prerequisites, raw
  samples *and* a release candidate + claim ledger digest (the S15 quadruple
  gate);
* ``--smoke`` runs a **CPU-only self-check** of the interfaces (claim scan,
  quickstart contract, hero replay, docs gate, supply chain, figure lineage,
  demo, narrative, clean room, upstream, first impression).  It is labelled
  ``smoke``, carries ``claim_allowed=False`` and is *not* an experiment;
* nothing is ever written into ``docs/stage_experiments/**`` (the protocol tree is
  read-only — ``hqsb.release.campaign.assert_writable`` enforces it).

Usage:
    scripts/release/run_e15.py --list
    scripts/release/run_e15.py --prerequisites
    scripts/release/run_e15.py --interface-map
    scripts/release/run_e15.py --objects
    scripts/release/run_e15.py --spec-audit
    scripts/release/run_e15.py --smoke
    scripts/release/run_e15.py --experiment E15-01 --prerequisites
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

from hqsb.release import experiment as exp  # noqa: E402
from hqsb.release import interface_map as imap  # noqa: E402
from hqsb.release import records as rec  # noqa: E402
from hqsb.release import specs as spec_mod  # noqa: E402

EXPERIMENT_TITLES: Dict[str, str] = {mapping.experiment_id: mapping.title for mapping in imap.EXPERIMENTS}

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "release")

#: Experiment module of each experiment for the smoke self-check (a broken module
#: is reported, never silently skipped).
SMOKE_MODULES: Dict[str, str] = {
    experiment_id: f"hqsb.release.{name}"
    for name, experiment_id, _level in imap.EXPERIMENT_MODULES
}

#: Foundations checked alongside the experiments.
FOUNDATION_MODULES: Dict[str, str] = {
    "identity": "hqsb.release.identity",
    "records": "hqsb.release.records",
    "contracts": "hqsb.release.contracts",
    "campaign": "hqsb.release.campaign",
    "telemetry": "hqsb.release.telemetry",
    "specs": "hqsb.release.specs",
    "experiment": "hqsb.release.experiment",
    "interface_map": "hqsb.release.interface_map",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S15 experiment driver (interface entry, no conclusions by default)"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_TITLES), help="experiment id")
    parser.add_argument("--list", action="store_true", help="list all experiments and steps")
    parser.add_argument("--prerequisites", action="store_true", help="print the prerequisite status")
    parser.add_argument("--interface-map", action="store_true", help="print the step→interface table")
    parser.add_argument("--objects", action="store_true", help="list the S15 frozen objects")
    parser.add_argument("--spec-audit", action="store_true", help="audit configs/release documents")
    parser.add_argument("--spec-check", action="store_true", help="regenerate check for configs/release")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the CPU-only interface self-check (labelled smoke, not an experiment)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="also probe the machine for tooling when checking prerequisites",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "explicitly allow experiment execution; still refuses a conclusion without satisfied "
            "prerequisites, raw samples and a release candidate + claim ledger digest"
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
                "all eleven experiments are interface-complete; none has been executed and none carries "
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
        scoped = status.for_experiment(experiment_id)
        payload["scoped"] = scoped.as_dict()
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
            "objects": [
                "ReleaseCandidateSnapshot (§7.1)",
                "PublicEvidenceBundle (§7.2)",
                "FinalAcceptanceDecision (§7.3)",
                "ClaimRecord (§8.1)",
                "ContributionRecord (§19)",
            ],
            "note": "契约只校验结构，不产生测量值",
        },
        as_json,
    )


def cmd_spec_audit(as_json: bool) -> None:
    documents = spec_mod.ReleaseSpecs.load(SPEC_DIR)
    reports = documents.audit()
    cross = spec_mod.check_cross_document(documents.documents)
    _emit(
        {
            "spec_dir": SPEC_DIR,
            "kinds": len(documents.documents),
            "missing_kinds": documents.missing_kinds(),
            "duplicate_kinds": documents.duplicate_kinds(),
            "ok": spec_mod.audit_all_ok(reports) and not cross,
            "reports": reports,
            "cross_document_problems": cross,
        },
        as_json,
    )


def cmd_spec_check(as_json: bool) -> None:
    from scripts.release.gen_release_specs import check_documents  # type: ignore[import-not-found]

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
                    "experiments": len(rec.EXPERIMENT_TABLE),
                    "statuses": len(rec.ALL_STATUSES),
                    "record_classes": len(rec.RECORD_CLASSES),
                }
        except Exception as exc:  # noqa: BLE001
            results["modules"][f"foundation:{name}"] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}"
            }
    interface_result = imap.resolve_interfaces()
    spec_reports = spec_mod.ReleaseSpecs.load(SPEC_DIR).audit()
    results["interface_map"] = {
        "experiments": interface_result["experiments"],
        "steps": interface_result["steps"],
        "expected_steps": interface_result["expected_steps"],
        "references": interface_result["references"],
        "ok": interface_result["ok"],
        "failures": interface_result["failures"][:5],
    }
    results["spec_audit_ok"] = spec_mod.audit_all_ok(spec_reports)
    results["state_machines_ok"] = rec.validate_state_machines() == []
    results["note"] = (
        "CPU-only self-check of the interfaces; it is NOT an experiment and carries no claim/quickstart/"
        "replay/docs/release/figure/demo/narrative/reproduction/upstream/usability conclusion"
    )
    _emit(results, as_json)


def cmd_experiment(
    experiment_id: str, *, prerequisites_only: bool, execute: bool, run_id: str, as_json: bool, probe: bool
) -> None:
    status = exp.check_prerequisites(REPO_ROOT, probe=probe)
    scoped = status.for_experiment(experiment_id)
    if prerequisites_only:
        _emit(
            {
                "experiment_id": experiment_id,
                "prerequisites_satisfied": scoped.satisfied,
                "prerequisites_missing": scoped.missing,
                "prerequisites_advisory": scoped.advisory,
                "states": scoped.states(),
                "depends_on": list(rec.EXPERIMENT_DEPENDENCIES.get(experiment_id, ())),
                "checks": [check.as_dict() for check in scoped.checks],
            },
            as_json,
        )
        return
    run = exp.RunDirectory(REPO_ROOT, experiment_id, run_id or "interface_only")
    run.create()
    run.write_json("environment_fingerprint.json", exp.environment_fingerprint(REPO_ROOT))
    run.write_json("prerequisites.json", scoped.as_dict())
    run.write_json("interface_map.json", imap.mapping_for(experiment_id).as_dict())
    run.write_json("source_identity.json", exp.git_identity(REPO_ROOT))
    run.write_json(
        "experiment_record.json", exp.build_experiment_record(experiment_id, git=exp.git_identity(REPO_ROOT))
    )
    run.write_report_skeleton(EXPERIMENT_TITLES[experiment_id])
    run.write_json("participants_or_agents.json", {"actors": [], "note": "尚无参与者/审阅者记录"})
    run.write_text("limitations.md", "（待填：实验执行前的适用边界与限制）")
    run.write_acceptance({}, notes="尚无门禁结果；本 run 未执行任何实验")
    if not scoped.satisfied:
        verdict = run.write_status(
            status=rec.STATUS_BLOCKED,
            reason="prerequisites unsatisfied: " + ", ".join(scoped.missing),
            prerequisites=scoped,
        )
    else:
        verdict = run.write_status(
            rec.STATUS_NOT_STARTED,
            "prerequisites satisfied but no experiment has been executed; see the development report "
            "for the interface layer",
            scoped,
        )
    _emit(
        {
            "experiment_id": experiment_id,
            "run_dir": run.path,
            "status": verdict["status"],
            "reason": verdict.get("reason", ""),
            "prerequisites_satisfied": scoped.satisfied,
            "prerequisites_missing": scoped.missing,
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
                    "no experiment selected; S15 experiments may not be executed without release "
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
