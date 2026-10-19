"""L9 API and sealed same-workload comparison."""

from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict
from hqsb.core.parallel_execution import ParallelPlan, catalogue
from hqsb.console.schemas import RunRecord
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json


class ParallelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    plan: ParallelPlan


class ParallelComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str
    candidate_run_id: str


class PeerRequest(ParallelRequest):
    peer_machine_id: str
    peer_deployment_id: str


def compare(a, b):
    reasons = []
    if any(d["status"] != "applied" or d["cleanup"] != "succeeded" for d in [a, b]):
        reasons.append("completed_applied_clean_runs_required")
    for key in ["model_identity", "tokenizer_identity", "source_digest"]:
        if not a.get(key) or a[key] != b.get(key):
            reasons.append(key + "_mismatch")
    for key in [
        "device_type",
        "precision",
        "prompts",
        "max_output_tokens",
        "fixed_output_length",
        "trace_collectives",
    ]:
        if a["plan"][key] != b["plan"][key]:
            reasons.append(key + "_mismatch")
    if (
        a["ranks"]
        and b["ranks"]
        and a["ranks"][0]["versions"] != b["ranks"][0]["versions"]
    ):
        reasons.append("runtime_versions_mismatch")
    actual = lambda d: [
        (r["index"], r["input_token_hash"], r["token_ids"]) for r in d["requests"]
    ]
    exact = bool(a["requests"]) and actual(a) == actual(b)
    different_workload_clock = (a["plan"]["strategy"] == "dp_hosts") != (
        b["plan"]["strategy"] == "dp_hosts"
    )
    if not exact:
        reasons.append("actual_token_or_input_mismatch")
    return {
        "comparable": not reasons,
        "reasons": reasons,
        "greedy_tokens_exact": exact,
        "performance_comparable": not different_workload_clock,
        "performance_scope": {
            "output_tokens_per_s": "clock_boundary_mismatch"
            if different_workload_clock
            else "matched_workload_boundary",
            "workload_ms": "clock_boundary_mismatch"
            if different_workload_clock
            else "matched_workload_boundary",
            "job_total_ms": "both_include_cold_load_and_cleanup",
        },
        "performance": {
            k: [a.get(k), b.get(k)]
            for k in ["output_tokens_per_s", "workload_ms", "job_total_ms"]
        },
        "claim": "descriptive_pair_only_no_general_speedup_or_quality_claim",
    }


def parallel_router(settings, service, jobs, peers):
    router = APIRouter(tags=["parallel-execution"])

    @router.get("/parallel/catalog")
    def catalog():
        return catalogue()

    @router.get("/parallel/hardware")
    def hardware():
        return jobs.hardware()

    @router.post("/parallel/jobs", response_model=RunRecord, status_code=202)
    def create(
        body: ParallelRequest,
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        return jobs.start(body.deployment_id, body.plan, idempotency_key)

    @router.get("/parallel/jobs")
    def listing():
        return service.store.list(60, kind="distributed")

    @router.get("/parallel/peers")
    def peer_list():
        with peers.registry.lock:
            return {
                "items": [
                    {"id": c.id, "name": c.name} for c in peers.registry.items.values()
                ]
            }

    @router.post("/parallel/federated-jobs", response_model=RunRecord, status_code=202)
    def federated(
        body: PeerRequest, idempotency_key: str = Header(min_length=1, max_length=128)
    ):
        return peers.start(
            body.deployment_id,
            body.peer_machine_id,
            body.peer_deployment_id,
            body.plan,
            idempotency_key,
        )

    @router.get("/runs/{identity}/parallel")
    def receipt(identity: str):
        doc, manifest = jobs.read(identity)
        return {**doc, "publication": manifest["publication"]}

    @router.post("/parallel/comparisons")
    def comparison(body: ParallelComparison):
        a, ma = jobs.read(body.baseline_run_id)
        b, mb = jobs.read(body.candidate_run_id)
        result = {
            **compare(a, b),
            "run_ids": [body.baseline_run_id, body.candidate_run_id],
            "sealed_manifests": [fingerprint(ma), fingerprint(mb)],
        }
        root = settings.data_dir / "l9-comparisons"
        root.mkdir(exist_ok=True)
        atomic_json(root / (fingerprint(result) + ".json"), result)
        return result

    return router
