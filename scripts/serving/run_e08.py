#!/usr/bin/env python3
"""S08 experiment driver — interface entry point, not an experiment result.

This driver implements the full "preregistration → collect → write → decide"
structure for every S08 experiment, but it is **forbidden by default from
producing a conclusion**.  Concretely:

* without ``--execute``, any verdict stays ``BLOCKED`` (the prerequisites are
  unmet in this repository state: no S07 P0 verdict, no two-Backend registry,
  no frozen SLO/request fixtures/topology/loadgen calibration);
* with ``--execute`` it will only emit a conclusion if the prerequisites pass
  *and* raw samples exist — on this development host there is no GPU/real
  Backend, so it still refuses;
* ``--smoke`` runs the model-free fixture through the gateway and validates the
  SSE bytes — it is explicitly labelled ``smoke`` and is not an experiment.

Usage:
    scripts/serving/run_e08.py --list
    scripts/serving/run_e08.py --experiment E08-01 --prerequisites
    scripts/serving/run_e08.py --experiment E08-01 --smoke
    scripts/serving/run_e08.py --interface-map
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

from hqsb.serving import experiment as exp  # noqa: E402
from hqsb.serving import interface_map as imap  # noqa: E402

EXPERIMENT_TITLES: Dict[str, str] = {
    mapping.experiment_id: mapping.title for mapping in imap.EXPERIMENTS
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S08 experiment driver (interface entry, no conclusions by default)"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_TITLES), help="experiment id")
    parser.add_argument("--list", action="store_true", help="list all experiments and steps")
    parser.add_argument("--prerequisites", action="store_true", help="print the prerequisite status")
    parser.add_argument("--interface-map", action="store_true", help="print the step→interface table")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the model-free fixture through the gateway (labelled smoke, not an experiment)",
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
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON instead of text"
    )
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
    status = exp.check_prerequisites(REPO_ROOT)
    _emit(status.as_dict(), as_json)


def cmd_interface_map(experiment_id: Optional[str], as_json: bool) -> None:
    if experiment_id:
        mapping = imap.mapping_for(experiment_id)
        _emit(mapping.as_dict(), as_json)
        return
    _emit(imap.resolve_interfaces(), as_json)


def _smoke_gateway() -> Dict[str, Any]:
    """Drive the model-free fixture; prove the interface works, not performance."""
    from hqsb.serving import protocol as protocol_mod
    from hqsb.serving import sse
    from hqsb.serving.clients import ClientScript, behavior_matrix
    from hqsb.serving.dummy_backend import (
        DummyServingBackend,
        frozen_identity,
        template_identity,
        tokenizer_for_vocab,
    )
    from hqsb.serving.gateway import GatewayConfig, ServingGateway
    from hqsb.serving.transport import InProcessTransport, TransportRequest

    config_dir = os.path.join(REPO_ROOT, "configs", "serving")
    import yaml

    with open(os.path.join(config_dir, "protocol_profile.yaml"), encoding="utf-8") as handle:
        profile = protocol_mod.ProtocolProfile.from_document(yaml.safe_load(handle))
    with open(os.path.join(config_dir, "error_catalog.yaml"), encoding="utf-8") as handle:
        catalog = protocol_mod.ErrorCatalog.from_document(yaml.safe_load(handle))
    backend = DummyServingBackend(identity=frozen_identity())
    gateway = ServingGateway(
        profile=profile,
        catalog=catalog,
        model_registry={"dummy-model": frozen_identity()},
        tokenizer=tokenizer_for_vocab(),
        chat_template=template_identity,
        backends={backend.instance_id: backend},
        config=GatewayConfig(),
    )
    body = json.dumps(
        {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    ).encode()
    script = ClientScript(connection_id="smoke-0", behavior=behavior_matrix()["normal_fast_reader"])
    transport = InProcessTransport(script=script)
    request = TransportRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        body=body,
        received_ns=1_000_000,
        connection_id="smoke-0",
    )
    client = transport.run(request, gateway.handle)
    outcome = transport.last_handler_result
    parsed = sse.parse_stream(client.body)
    audit = sse.validate_stream(parsed["parsed"])
    return {
        "status": "smoke",
        "service_status": outcome.status,
        "code": outcome.code,
        "frames": outcome.frames,
        "ledger_ok": outcome.ledger.audit()["ok"],
        "sse_complete": parsed["complete"],
        "sse_ok": audit["ok"],
        "reconstructed_tokens": sse.reconstruct(parsed["parsed"])["token_ids"],
        "terminal_state": outcome.terminal_state,
        "backend_open_count": backend.open_count,
        "note": (
            "this is a smoke self-check of the interfaces; it is NOT an experiment and "
            "carries no performance/capacity/quality conclusion"
        ),
    }


def cmd_smoke(as_json: bool) -> None:
    _emit(_smoke_gateway(), as_json)


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
        _emit(status.as_dict(), as_json)
        return
    run = exp.RunDirectory(REPO_ROOT, experiment_id, run_id or exp.STATUS_NOT_STARTED)
    # Never write into docs/stage_experiments: only experiment_results/S08/<E>/<run>/.
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
                    "no experiment selected; the S08 prerequisites are unmet, so no "
                    "conclusion may be produced. Use --list, --smoke, --prerequisites or "
                    "--experiment E08-xx --prerequisites"
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
