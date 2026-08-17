#!/usr/bin/env python3
"""Real model/API acceptance, isolated store, no mock actors or measured speedup."""

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from hqsb.console.app import PREFIX, create_app
from hqsb.console.config import load_settings
from hqsb.console.store import TERMINAL
from hqsb.observation.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".console/observation-validation")
    parser.add_argument(
        "--output", default="reports/observation/console-validation.json"
    )
    parser.add_argument("--privileged-worker", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    settings.data_dir = Path(args.data_dir).resolve()
    settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (settings.data_dir / "console.sqlite3").exists():
        parser.error("Use a new store: retained evidence must never be overwritten")
    settings.privileged_worker = args.privileged_worker
    settings.deployments[0].manifest = str(
        ROOT / "docs/benchmark/model_sha256_manifest.txt"
    )
    token = secrets.token_urlsafe(32)
    token_path = settings.data_dir / "access-token"
    token_path.write_text(token)
    token_path.chmod(0o600)
    report = {
        "status": "FAIL",
        "scope": "real_model_console_api",
        "checks": [],
        "runs": [],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    def check(name, condition, details=None):
        report["checks"].append(
            {"name": name, "passed": bool(condition), "details": details}
        )
        print(json.dumps({"check": name, "passed": bool(condition)}), flush=True)
        if not condition:
            raise AssertionError(name)

    app = create_app(settings, token, monitor=False)
    try:
        with TestClient(app, headers={"Authorization": "Bearer " + token}) as client:

            def get(path):
                response = client.get(PREFIX + path)
                response.raise_for_status()
                return response.json()

            def post(path, payload=None):
                response = client.post(
                    PREFIX + path,
                    json=payload or {},
                    headers={"Idempotency-Key": uuid.uuid4().hex},
                )
                response.raise_for_status()
                return response.json()

            def wait(run):
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    current = get("/runs/" + run["id"])
                    if current["state"] in TERMINAL:
                        report["runs"].append(current)
                        return current
                    time.sleep(0.1)
                post("/requests/" + run["id"] + "/cancel")
                raise TimeoutError(run["id"])

            deployment_id = settings.deployments[0].id
            loaded = wait(post(f"/deployments/{deployment_id}/load"))
            check(
                "real_model_loaded", loaded["state"] == "completed", loaded.get("error")
            )
            deployment = next(
                d for d in get("/deployments")["items"] if d["id"] == deployment_id
            )
            body = {
                "deployment_id": deployment_id,
                "expected_epoch": deployment["epoch"],
                "messages": [
                    {"role": "user", "content": "请简要解释推理时 KV Cache 的作用。"}
                ],
                "max_output_tokens": 9,
                "deadline_ms": 120000,
                "save_input": False,
            }
            ids, baseline = {}, None
            for mode in ("off", "basic", "memory", "pages", "operators", "basic"):
                run = wait(post("/requests", {**body, "observation_mode": mode}))
                check(
                    f"{mode}_completed", run["state"] == "completed", run.get("error")
                )
                tokens = run["metrics"]["generated_token_ids"]
                baseline = tokens if baseline is None else baseline
                check(f"{mode}_identical_generated_tokens", tokens == baseline)
                summary = get(f"/runs/{run['id']}/observations")
                if mode == "off":
                    check(
                        "off_not_collected", summary["availability"] == "not_collected"
                    )
                    continue
                check(
                    f"{mode}_sealed",
                    summary["capture_status"] == "complete"
                    and summary["publication"] == "supervisor_acknowledged",
                    summary.get("source_problems"),
                )
                check(f"{mode}_kv_observed", bool(summary["kv"]))
                origin = summary["context"].get("host_time_origin_ns")
                check(
                    f"{mode}_host_clock_origin",
                    isinstance(origin, str) and origin.isdigit(),
                )
                os_row = get(
                    f"/runs/{run['id']}/observations/records?kind=os_sample&limit=1"
                )["items"][0]["payload"]
                check(
                    f"{mode}_os_clock_compatible",
                    int(os_row["read_start_ns"]) >= int(origin)
                    and os_row["clock_domain"]
                    == summary["context"]["host_clock_domain"],
                )
                workload = get(f"/runs/{run['id']}/observations/records?kind=workload")[
                    "items"
                ][0]
                check(
                    f"{mode}_input_privacy",
                    workload["payload"]["input_token_ids"] is None,
                )
                manifest = get(f"/runs/{run['id']}/observations/manifest")
                raw = client.get(PREFIX + f"/runs/{run['id']}/observations/artifacts/0")
                check(
                    f"{mode}_download_hash",
                    raw.status_code == 200
                    and hashlib.sha256(raw.content).hexdigest()
                    == manifest["artifacts"][0]["sha256"],
                )
                if mode == "memory":
                    check(
                        "allocator_lifetimes",
                        summary["allocation_lifetimes"]["total_items"] > 0,
                    )
                    check(
                        "tensor_allocation_attribution",
                        any(
                            a["confidence"] == "snapshot_correlated"
                            for a in summary["allocation_attribution"]
                        ),
                    )
                elif mode == "pages":
                    check(
                        "page_samples",
                        any(p["sampled_pages"] > 0 for p in summary["page_samples"]),
                    )
                elif mode == "operators":
                    profile = run["metrics"]["observation"]["profile"]
                    check(
                        "cuda_operators",
                        profile["status"] == "complete",
                        profile.get("limitations"),
                    )
                    check(
                        "all_decoder_roots",
                        len(profile["coverage"]["module_names"]) > 1,
                    )
                    report["links"] = get(f"/runs/{run['id']}/observations/links")
                    check(
                        "sealed_framework_trace",
                        profile.get("trace_sha256")
                        == report["links"]["trace"].get("sha256")
                        and profile.get("trace_sha256") is not None,
                    )
                if mode in ids:
                    comparison = post(
                        "/observation-comparisons", {"run_ids": [ids[mode], run["id"]]}
                    )
                    check(
                        "repeat_mechanism_comparable",
                        comparison["comparable"],
                        comparison["reasons"],
                    )
                    check(
                        "no_quality_or_speedup_claim", comparison["decision"] == "hold"
                    )
                    report["comparison"] = comparison
                ids[mode] = run["id"]
            cancel = post(
                "/requests",
                {**body, "observation_mode": "basic", "max_output_tokens": 512},
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                current = get("/runs/" + cancel["id"])
                if (current["metrics"].get("output_tokens") or 0) > 0 or current[
                    "state"
                ] in TERMINAL:
                    break
                time.sleep(0.05)
            post(f"/requests/{cancel['id']}/cancel")
            cancelled = wait(cancel)
            check(
                "cancel_cleanup",
                cancelled["state"] == "cancelled"
                and cancelled["cleanup"] == "succeeded",
                {"state": cancelled["state"], "cleanup": cancelled["cleanup"]},
            )
            cancel_summary = get(f"/runs/{cancel['id']}/observations")
            check(
                "cancel_evidence_retained",
                cancel_summary.get("publication") == "supervisor_acknowledged",
            )
            unloaded = wait(post(f"/deployments/{deployment_id}/unload"))
            check("unload_cleanup", unloaded["state"] == "completed")
            check(
                "archive_survives_unload",
                get(f"/runs/{ids['memory']}/observations")["allocator"] != [],
            )
            report.update(
                status="PASS", selected_runs=ids, data_dir=str(settings.data_dir)
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        app.state.service.close()
        atomic_json(output, report)


if __name__ == "__main__":
    main()
