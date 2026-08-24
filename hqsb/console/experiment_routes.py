"""Authenticated experiment plans, device jobs and verified reports.

Runs use the existing serial worker and its cancellation/epoch protocol. No
browser-provided paths, executables, imports or shell commands are accepted.
"""

import re
import threading
import time
import uuid

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from hqsb.core.optimization import catalogue, fingerprint
from hqsb.console.captures import safe_directory
from hqsb.console.store import TERMINAL
from hqsb.experiments.artifacts import load_bundle, read_json
from hqsb.experiments.comparison import compare_campaign
from hqsb.experiments.plans import role_request, validate_spec
from hqsb.observation.storage import atomic_json

EXPERIMENT_ID = re.compile(r"exp_[0-9a-f]{32}\Z")


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: dict


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str
    arm: Literal["baseline", "candidate"]
    role: Literal["performance", "diagnostic", "quality"]
    pair: int = Field(0, ge=0, le=19)


def experiment_router(settings, service):
    router = APIRouter(tags=["experiments"])
    lock = threading.RLock()
    root = settings.data_dir / "experiments"

    def directory(experiment_id):
        if not EXPERIMENT_ID.fullmatch(experiment_id):
            raise HTTPException(422, "Invalid experiment ID")
        path = root / experiment_id
        if root.is_symlink() or path.is_symlink():
            raise HTTPException(409, "Invalid experiment directory")
        return path

    @router.get("/optimization-catalog")
    def optimization_catalog():
        return catalogue()

    @router.post("/experiments/plans", status_code=201)
    def create_plan(body: PlanRequest):
        try:
            plan = validate_spec(body.spec)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        experiment_id = "exp_" + uuid.uuid4().hex
        path = directory(experiment_id)
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        plan.update(experiment_id=experiment_id, registered_at=time.time(), evaluation_scope="explicit_fixture_dataset")
        atomic_json(path / "plan.json", plan)
        atomic_json(path / "ledger.json", {"jobs": []})
        return plan

    @router.get("/experiments/{experiment_id}/plan")
    def get_plan(experiment_id: str):
        try:
            plan = read_json(directory(experiment_id) / "plan.json")
            if fingerprint(plan["spec"]) != plan["spec_sha256"] or plan["experiment_id"] != experiment_id:
                raise ValueError("EXPERIMENT_PLAN_INTEGRITY")
            return plan
        except FileNotFoundError as exc:
            raise HTTPException(404, "Unknown experiment") from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, "Experiment plan integrity failed") from exc

    @router.post("/experiments/{experiment_id}/jobs", status_code=202)
    def job(experiment_id: str, body: JobRequest, idempotency_key: str = Header(min_length=1, max_length=128)):
        with lock:
            plan = get_plan(experiment_id)
            if body.role == "performance" and body.pair >= plan["spec"]["performance_pairs"]:
                raise HTTPException(422, "Pair exceeds pre-registered sample count")
            if body.role != "performance" and body.pair != 0:
                raise HTTPException(422, "Quality and diagnostic have one pair per plan")
            path = directory(experiment_id)
            ledger = read_json(path / "ledger.json")
            existing = [item for item in ledger["jobs"] if (item["arm"], item["role"], item["pair"]) == (body.arm, body.role, body.pair)]
            if existing:
                if existing[0].get("submission_sha256") != fingerprint(body.model_dump()):
                    raise HTTPException(409, "Experiment slot already belongs to another submission")
                return service.store.get(existing[0]["run_id"])
            payload = {"expected_epoch": body.expected_epoch, "deadline_ms": settings.request_deadline_ms,
                       "messages": [{"role": "user", "content": "Explain model inference and memory caching."}],
                       "save_input": False, **role_request(plan, body.arm, body.role),
                       "experiment_links": {"campaign_id": experiment_id, "spec_sha256": plan["spec_sha256"],
                                            "pair_id": str(body.pair), "arm": body.arm, "role": body.role}}
            if body.role != "quality":
                payload.update(experiment_input_tokens=plan["spec"]["input_tokens"], fixed_output_length=True)
            run = service.submit("generate", body.deployment_id, payload, idempotency_key)
            ledger["jobs"].append({"run_id": run["id"], "arm": body.arm, "role": body.role, "pair": body.pair,
                                   "submission_sha256": fingerprint(body.model_dump())})
            atomic_json(path / "ledger.json", ledger)
            return run

    @router.get("/runs/{run_id}/experiment")
    def run_experiment(run_id: str):
        run = service.store.get(run_id)
        if run["state"] not in TERMINAL:
            raise HTTPException(409, "Await worker cleanup")
        try:
            return load_bundle(safe_directory(settings.data_dir / "captures", run_id))
        except (ValueError, OSError) as exc:
            raise HTTPException(409, f"Experimental evidence unavailable: {exc}") from exc

    @router.get("/runs/{run_id}/experiment/artifacts/{index}")
    def artifact(run_id: str, index: int):
        bundle = run_experiment(run_id)
        files = bundle["manifest"]["artifacts"]
        if not 0 <= index < len(files):
            raise HTTPException(404, "Unknown artifact")
        item = files[index]
        return FileResponse(safe_directory(settings.data_dir / "captures", run_id) / item["path"],
                            filename=f"{run_id}-{item['path']}", headers={"X-Content-SHA256": item["sha256"]})

    @router.get("/experiments/{experiment_id}/report")
    def report(experiment_id: str):
        plan = get_plan(experiment_id)
        ledger = read_json(directory(experiment_id) / "ledger.json")
        pairs, missing, failures = [], [], []
        for role, count in (("performance", plan["spec"]["performance_pairs"]), ("diagnostic", 1), ("quality", 1)):
            for pair_id in range(count):
                entries = {item["arm"]: item for item in ledger["jobs"] if item["role"] == role and item["pair"] == pair_id}
                if set(entries) != {"baseline", "candidate"}:
                    missing.append({"role": role, "pair": pair_id})
                    continue
                a, b = (service.store.get(entries[arm]["run_id"]) for arm in ("baseline", "candidate"))
                if a["state"] != "completed" or b["state"] != "completed":
                    failures.append({"role": role, "pair": pair_id, "states": [a["state"], b["state"]]})
                    continue
                pairs.append({"role": role,
                              "baseline_dir": str(safe_directory(settings.data_dir / "captures", a["id"])),
                              "candidate_dir": str(safe_directory(settings.data_dir / "captures", b["id"])),
                              "baseline_metrics": a["metrics"], "candidate_metrics": b["metrics"]})
        try:
            result = compare_campaign(plan, pairs)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(409, f"Experiment integrity failed: {exc}") from exc
        result.update(experiment_id=experiment_id, missing_pairs=missing, failed_pairs=failures,
                      completeness="complete" if not missing and not failures and not any(reason in result["reasons"] for reason in ("INCOMPLETE_EXECUTION", "DIAGNOSTIC_COVERAGE_MISMATCH")) else "partial")
        if missing or failures:
            result["decision"] = "hold"
        return result

    return router
