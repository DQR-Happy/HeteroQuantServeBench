#!/usr/bin/env python3
"""S05 experiment driver / interface entry point.

Modes
-----
``status`` (default)
    Print the S04.5-M4 prerequisite report and the experiment's protocol
    status. Writes nothing unless ``--output-dir`` is given, in which case a
    ``status.json`` (never a verdict) is recorded.
``preregister``
    Write ``preregistration.json`` from the shipped template with a frozen
    hash, plus the empty run layout.
``interface-map``
    Print (or write) the experiment-step → interface table and verify that
    every referenced symbol imports.
``self-check``
    Run the *smoke-level* interface self-checks and print them labelled as
    smoke. These are not experiment results: they only prove that the
    interfaces are callable and that the negative paths are refused.
``execute``
    Refused unless ``--confirm-execute`` is passed **and** the prerequisite
    chain is satisfied. This repository state does not satisfy it, so the
    mode exits non-zero with the missing evidence listed — by design.

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
from hqsb.quant import experiment as exp  # noqa: E402
from hqsb.quant import interface_map as imap  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "S05 experiment driver. The default mode only inspects "
            "prerequisites and interfaces; no conclusion can be produced "
            "without an explicit, prerequisite-gated execution."
        )
    )
    parser.add_argument(
        "--experiment",
        default="all",
        help="E05-01 .. E05-10, or 'all' (default).",
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
    parser.add_argument(
        "--run-id",
        default="",
        help="Explicit run id (default: UTC timestamp).",
    )
    parser.add_argument(
        "--confirm-execute",
        action="store_true",
        help="Required for --mode execute; an extra guard against accidental runs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human summary.",
    )
    parser.add_argument(
        "--template",
        default="",
        help="Optional JSON file with preregistration fields for --mode preregister.",
    )
    return parser


def _selected_experiments(value: str) -> List[str]:
    if value == "all":
        return list(exp.EXPERIMENTS)
    if value not in exp.EXPERIMENTS:
        raise UsageError(
            f"unknown experiment {value!r}; expected one of {list(exp.EXPERIMENTS)}"
        )
    return [value]


def _preregistration(experiment_id: str, template_path: str) -> exp.Preregistration:
    payload: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "question": "see docs/stage_experiments/details/S05/<file>",
        "hypothesis": "see docs/stage_experiments/details/S05/<file>",
        "controls": {},
        "independent_variables": {},
        "metrics": {},
        "gates": {},
        "guard_band": {},
        "seeds": (),
        "repeats": 0,
        "independent_processes": 3,
        "exclusions": (),
        "stop_conditions": (),
        "allowed_claims": (),
        "hardware": "",
        "backend": "",
        "notes": (
            "template only; the experiment owner must fill question/hypothesis/"
            "metrics/gates from the protocol document before any execution"
        ),
    }
    if template_path:
        with open(template_path, encoding="utf-8") as handle:
            payload.update(json.load(handle))
    payload["experiment_id"] = experiment_id
    return exp.Preregistration(**payload)


def mode_status(experiments: Sequence[str], output_dir: str, run_id: str, as_json: bool) -> Dict[str, Any]:
    prerequisites = exp.check_prerequisites(REPO_ROOT)
    results: List[Dict[str, Any]] = []
    for experiment_id in experiments:
        entry: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": exp.STATUS_NOT_STARTED if prerequisites.satisfied else exp.STATUS_BLOCKED,
            "prerequisites_satisfied": prerequisites.satisfied,
            "missing": prerequisites.missing,
        }
        if output_dir:
            record = exp.interface_only_run(REPO_ROOT, experiment_id, run_id or None)
            entry["run_dir"] = record["run_dir"]
            entry["status"] = record["status"]
        results.append(entry)
    report = {
        "stage": exp.STAGE,
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
        print(f"[S05] prerequisite chain satisfied: {prerequisites.satisfied}")
        for check in prerequisites.checks:
            marker = "OK " if check.satisfied else "MISS"
            print(f"  {marker} {check.name}: {check.reason or check.evidence}")
        for entry in results:
            print(f"  {entry['experiment_id']}: {entry['status']}")
    return report


def mode_preregister(experiments: Sequence[str], output_dir: str, run_id: str, template: str, as_json: bool) -> Dict[str, Any]:
    if not output_dir:
        raise UsageError("--mode preregister requires --output-dir")
    prerequisites = exp.check_prerequisites(REPO_ROOT)
    written: List[Dict[str, Any]] = []
    for experiment_id in experiments:
        run = exp.RunDirectory(output_dir, experiment_id, run_id or "prereg")
        run.create()
        prereg = _preregistration(experiment_id, template)
        run.write_preregistration(prereg)
        run.write_json("prerequisites.json", prerequisites.as_dict())
        run.write_status(
            exp.STATUS_BLOCKED if not prerequisites.satisfied else exp.STATUS_NOT_STARTED,
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
    print(json.dumps(report, indent=2) if as_json else "\n".join(
        f"{item['experiment_id']}: prereg_hash={item['prereg_hash'][:16]} dir={item['run_dir']}"
        for item in written
    ))
    return report


def mode_interface_map(experiments: Sequence[str], output_dir: str, as_json: bool) -> Dict[str, Any]:
    resolved = imap.resolve_interfaces()
    if not as_json:
        print(
            f"[S05] interface map: {resolved['steps']} steps, "
            f"{resolved['interfaces']} interfaces, ok={resolved['ok']}"
        )
        for failure in resolved["failures"]:
            print(f"  FAIL {failure['symbol']}: {failure['error']}")
        if len(experiments) == len(exp.EXPERIMENTS):
            print(imap.mapping_table_markdown())
    else:
        print(json.dumps(resolved, indent=2))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "s05_interface_map.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(imap.mapping_table_markdown())
        print(f"table written to {path}", file=sys.stderr)
    return resolved


def mode_self_check(as_json: bool) -> Dict[str, Any]:
    """Smoke-level checks: interfaces callable, negative paths refused.

    Everything here is labelled ``smoke``; none of it is an experiment result
    and none of it may be quoted as one (AGENTS.md result-labelling rules).
    """
    from hqsb.quant import compat, faults
    from hqsb.quant.fixtures import save_golden_artifacts
    from hqsb.quant.golden import cross_check_with_reference
    from hqsb.quant.packing import LAYOUT_W4A16_ROWMAJOR_NK_V1

    import tempfile

    checks: Dict[str, Any] = {"label": "smoke", "stage": exp.STAGE}
    checks["golden_cross_check"] = cross_check_with_reference()
    with tempfile.TemporaryDirectory(prefix="hqsb-e05-selfcheck-") as tmp:
        paths = save_golden_artifacts(os.path.join(tmp, "golden"))
        capability = compat.KernelCapability(
            kernel_id="hqsb.w4a16.triton",
            provider="triton",
            layouts=(LAYOUT_W4A16_ROWMAJOR_NK_V1,),
            bits=(4,),
            group_sizes=(128, None),
            target_arch="sm_86",
            abi_version="1",
        )
        results = faults.run_fault_matrix(
            paths["w4"], os.path.join(tmp, "scratch"), capability
        )
        checks["fault_matrix"] = faults.summarize_fault_matrix(results)
    try:
        from ops.quant.capability import probe_low_bit_capability

        capability_report = probe_low_bit_capability(compile_probe=False).as_dict()
        checks["low_bit_capability"] = {
            "torch_available": capability_report["torch_available"],
            "cuda_available": capability_report["cuda_available"],
            "triton_available": capability_report["triton_available"],
            "device": capability_report["device_name"],
            "device_capability": capability_report["device_capability"],
            "reasons": capability_report["reasons"],
        }
    except Exception as exc:  # noqa: BLE001 - capability probing never blocks
        checks["low_bit_capability"] = {"error": f"{type(exc).__name__}: {exc}"}
    if as_json:
        print(json.dumps(checks, indent=2, ensure_ascii=False))
    else:
        print("[S05] self-check (smoke; NOT an experiment result)")
        print(f"  golden cross-check: {checks['golden_cross_check']}")
        print(f"  fault matrix: {checks['fault_matrix']}")
        print(f"  low-bit capability: {json.dumps(checks['low_bit_capability'])}")
    return checks


def mode_execute(experiments: Sequence[str], output_dir: str, run_id: str, confirm: bool, as_json: bool) -> Dict[str, Any]:
    """Refuse to execute unless the whole gate chain explicitly allows it."""
    prerequisites = exp.check_prerequisites(REPO_ROOT)
    if not confirm:
        raise UsageError(
            "--mode execute requires --confirm-execute; nothing was run"
        )
    if not prerequisites.satisfied:
        report = {
            "mode": "execute",
            "executed": False,
            "status": exp.STATUS_BLOCKED,
            "missing": prerequisites.missing,
            "reason": (
                "S05 cannot execute: the S04.5 M4 prerequisite chain is unmet. "
                "Run S04.5 first; the interface code is ready and its self-check "
                "is available via --mode self-check."
            ),
        }
        print(json.dumps(report, indent=2) if as_json else report["reason"])
        return report
    # Prerequisites satisfied: the environment-specific execution is still
    # deliberately not implemented here — the driver must be extended with the
    # concrete runner for the target machine rather than silently "succeeding".
    raise UsageError(
        "prerequisites are satisfied but no real runner is wired in this tree; "
        "extend scripts/quant/run_e05.py with the hardware-specific runner "
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
            report = mode_execute(experiments, args.output_dir, args.run_id, args.confirm_execute, args.json)
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
