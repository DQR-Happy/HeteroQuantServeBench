#!/usr/bin/env python3
"""Real lifecycle + kernel receipt API check, in a private temporary Console.

Optional privileged child uses the existing bounded protocol for GPU profiling.
No API listener or access token is published by this script.
"""
import argparse
import json
from pathlib import Path
import secrets
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from fastapi.testclient import TestClient
from hqsb.console.app import PREFIX, create_app
from hqsb.console.config import load_settings
from hqsb.console.store import TERMINAL
from hqsb.observation.storage import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--kernel-manifest",required=True)
    parser.add_argument("--data-dir",default=".console/l2-api")
    parser.add_argument("--output",default="reports/l2/api-validation.json")
    parser.add_argument("--privileged-worker",action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    settings.data_dir = Path(args.data_dir).resolve()
    settings.data_dir.mkdir(parents=True,exist_ok=False)
    settings.privileged_worker = args.privileged_worker
    settings.deployments[0].manifest = str(Path(args.manifest).resolve())
    settings.deployments[0].kernel_manifest = str(Path(args.kernel_manifest).resolve())
    token = secrets.token_urlsafe(32)
    app = create_app(settings,token,monitor=False)
    report = {"status":"FAIL","checks":[],"runs":[],"captures":[]}

    def check(name,value):
        report["checks"].append({"name":name,"passed":bool(value)})
        print(json.dumps(report["checks"][-1]),flush=True)
        if not value:
            raise AssertionError(name)

    try:
        with TestClient(app,headers={"Authorization":"Bearer "+token}) as client:
            def get(path):
                r = client.get(PREFIX+path)
                r.raise_for_status()
                return r.json()

            def post(path,body=None):
                r = client.post(PREFIX+path,json=body or {},headers={"Idempotency-Key":uuid.uuid4().hex})
                r.raise_for_status()
                return r.json()

            def wait(run):
                deadline = time.monotonic()+600
                while time.monotonic()<deadline:
                    current = get("/runs/"+run["id"])
                    if current["state"] in TERMINAL:
                        report["runs"].append({"id":current["id"],"state":current["state"],"error":current.get("error")})
                        return current
                    time.sleep(.1)
                raise TimeoutError(run["id"])

            name = settings.deployments[0].id
            selected = "qwen3-local-fusions-v1"
            plan = post("/experiments/plans",{"spec":{
                "candidate":{"changes":{"kernel_profile":selected}},"declared_changes":["kernel_profile"],
                "performance_pairs":1,"input_tokens":16,"output_tokens":3,
                "quality_plan":{"samples":[{"text":"The GPU reads model weights."}],
                                "generation_tokens":2,"forward_mode":"full_prefill"},
                "observation_plan":{"mode":"operators" if args.privileged_worker else "memory"}}})
            experiment_id = plan["experiment_id"]

            def job(arm,role,epoch):
                run = wait(post(f"/experiments/{experiment_id}/jobs",{"deployment_id":name,
                    "expected_epoch":epoch,"arm":arm,"role":role,"pair":0}))
                check(arm+"_"+role,run["state"]=="completed")
                return run
            check("profile_catalogue",any(p["profile_id"]==selected for p in get("/kernels/profiles")["profiles"]))
            check("load_profile",wait(post(f"/deployments/{name}/load",{"kernel_profile":selected}))["state"]=="completed")
            first = get("/deployments")["items"][0]["epoch"]
            body = {"deployment_id":name,"expected_epoch":first,"messages":[{"role":"user","content":"Explain KV cache."}],
                    "max_output_tokens":3,"observation_mode":"operators" if args.privileged_worker else "memory",
                    "optimization":{"changes":{"kernel_profile":selected}}}
            run = job("candidate","diagnostic",first)
            check("real_generation",run["state"]=="completed")
            receipt = run["metrics"]["optimization_receipt"]
            check("profile_invoked",receipt["status"]=="applied" and receipt["kernel_receipt"]["status"]=="invoked")
            bundle = get("/runs/"+run["id"]+"/experiment")
            check("kernel_artifacts_sealed",{"kernel-plan.json","kernel-bindings.json","kernel-summary.json"}.issubset(
                {a["path"] for a in bundle["manifest"]["artifacts"]}))
            report["captures"].append(str(settings.data_dir/"captures"/run["id"]))
            report["profile"] = run["metrics"].get("observation",{}).get("profile")
            job("candidate","quality",first)
            job("candidate","performance",first)
            invalid = client.post(PREFIX+"/requests",json={**body,"optimization":{"changes":{"kernel_profile":"framework-native"}}},
                                  headers={"Idempotency-Key":uuid.uuid4().hex})
            check("reload_guard",invalid.status_code==422)
            cancelling = post("/requests",{**body,"max_output_tokens":128,"observation_mode":"off"})
            post("/requests/"+cancelling["id"]+"/cancel")
            check("cancel_terminal",wait(cancelling)["state"] in {"cancelled","completed"})
            check("unload",wait(post(f"/deployments/{name}/unload"))["state"]=="completed")
            check("reload_reference",wait(post(f"/deployments/{name}/load"))["state"]=="completed")
            second = get("/deployments")["items"][0]["epoch"]
            check("new_epoch",first!=second)
            stale = client.post(PREFIX+"/requests",json={**body,"optimization":None},headers={"Idempotency-Key":uuid.uuid4().hex})
            check("stale_epoch",stale.status_code==409)
            reference = wait(post("/requests",{**body,"expected_epoch":second,"optimization":None,"observation_mode":"off"}))
            check("legacy_generation",reference["state"]=="completed" and reference["metrics"]["optimization_receipt"]["runtime"]["kernel"]=="framework-native")
            for role in ("diagnostic","quality","performance"):
                job("baseline",role,second)
            comparison = get(f"/experiments/{experiment_id}/report")
            report["comparison"] = comparison
            check("kernel_proof_comparable",comparison["comparable"])
            check("insufficient_pairs_hold",comparison["decision"]=="hold")
            check("kernel_mechanism_differences",any(d["metric_id"]=="kernel.logical_site_calls" for m in comparison["mechanisms"] for d in m["differences"]))
            check("final_unload",wait(post(f"/deployments/{name}/unload"))["state"]=="completed")
            report["status"] = "PASS"
    finally:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(destination,report)


if __name__ == "__main__":
    main()
