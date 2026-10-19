"""L7 metadata, explicit auxiliary-model release and same-workload evidence pairs."""

from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict
from hqsb.console.schemas import RunRecord
from hqsb.console.service import ServiceError
from hqsb.console.captures import safe_directory
from hqsb.core.decoding_plan import catalogue
from hqsb.core.optimization import fingerprint
from hqsb.experiments.artifacts import load_bundle
from hqsb.experiments.quality_contract import source_digest
from hqsb.observation.storage import atomic_json


class DraftRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str


class DecodingComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str
    candidate_run_id: str


def compare(a, b, left, right):
    reasons = []
    receipts = [left.get("receipt") or {}, right.get("receipt") or {}]
    x, y = receipts
    if any(
        r.get("status") != "applied" or r.get("cleanup_status") != "succeeded"
        for r in receipts
    ):
        reasons.append("applied_execution_receipts_required")
    dx = x.get("decoding_receipt") or {}
    dy = y.get("decoding_receipt") or {}
    for run, bundle, receipt in ((a, left, dx), (b, right, dy)):
        if (
            run["state"] != "completed"
            or run["cleanup"] != "succeeded"
            or bundle["manifest"]["status"] != "complete"
            or receipt.get("status") != "invoked"
        ):
            reasons.append("completed_applied_l7_run_required")
    for key in ("model_identity", "tokenizer_identity", "runtime"):
        if not x.get(key) or x[key] != y.get(key):
            reasons.append(key + "_mismatch")
    if not source_digest(x) or source_digest(x) != source_digest(y):
        reasons.append("source_digest_mismatch")
    if x.get("precision_receipt", {}).get("plan_sha256") != y.get(
        "precision_receipt", {}
    ).get("plan_sha256"):
        reasons.append("precision_plan_mismatch")
    for key in ("attention_backend", "cleanup_policy"):
        if x.get("effective", {}).get(key) != y.get("effective", {}).get(key):
            reasons.append(key + "_mismatch")
    for key in ("input_token_sha256", "requested_token_limit"):
        if not x.get("proof", {}).get(key) or x["proof"][key] != y.get("proof", {}).get(
            key
        ):
            reasons.append(key + "_mismatch")
    for key in ("observation_mode", "fixed_output_length"):
        if a["config"].get(key) != b["config"].get(key):
            reasons.append(key + "_mismatch")
    device = lambda r: r["config"].get("actual_runtime", {}).get("device_name")
    if not device(a) or device(a) != device(b):
        reasons.append("device_mismatch")
    plans = [r.get("plan", {}) for r in (dx, dy)]
    semantic_keys = (
        "sampling",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "min_new_tokens",
        "seed",
        "choice_sha256",
    )
    if any(plans[0].get(k) != plans[1].get(k) for k in semantic_keys):
        reasons.append("different_sampling_or_constraint_semantics")
    ga, gb = (
        a["metrics"].get("generated_token_ids"),
        b["metrics"].get("generated_token_ids"),
    )
    if any(
        not isinstance(tokens, list)
        or fingerprint(tokens) != r.get("output_token_hash")
        for tokens, r in ((ga, x), (gb, y))
    ):
        reasons.append("sealed_output_token_identity_mismatch")
    exact = ga is not None and gb is not None and ga == gb
    greedy = plans[0].get("sampling") == plans[1].get("sampling") == "greedy"
    if greedy and not exact:
        reasons.append("greedy_token_exactness_failed")
    if a["metrics"].get("output_tokens") != b["metrics"].get("output_tokens"):
        reasons.append("actual_output_length_mismatch")
    quality = {
        "metric": "generated_token_agreement",
        "exact_tokens": exact,
        "status": "greedy_exact_on_this_input"
        if greedy and exact and not reasons
        else "not_conformant"
        if greedy
        else "sampling_distribution_not_tested",
        "first_mismatch": next(
            (i for i, (u, v) in enumerate(zip(ga or [], gb or [])) if u != v), None
        ),
        "constraint_satisfied": [
            dx.get("constraint_satisfied"),
            dy.get("constraint_satisfied"),
        ],
        "ppl_not_applicable": "Teacher-forcing PPL does not exercise decoding or establish a sampling-policy quality score.",
    }
    return {
        "schema_version": "l7-comparison-v1",
        "baseline_run_id": a["id"],
        "candidate_run_id": b["id"],
        "comparable": not reasons,
        "reasons": sorted(set(reasons)),
        "quality": quality,
        "decoding": [dx, dy],
        "performance": {
            "status": "descriptive_pair_no_speedup_claim",
            **{
                key: [r["metrics"].get(key) for r in (a, b)]
                for key in (
                    "output_tokens_per_s",
                    "runtime_first_token_ms",
                    "runtime_e2e_ms",
                    "worker_request_total_ms",
                    "memory",
                )
            },
        },
        "limitations": [
            "A greedy match on one input is not a general numerical-equivalence or accuracy proof.",
            "Sampling policies require task-level multi-seed evaluation; identical seeds need not match across constraint implementations.",
            "Draft load cost and retained weights are separate from generation-loop Token/s; include request-total time for cold requests.",
        ],
    }


def decoding_router(settings, service):
    router = APIRouter(tags=["decoding"])

    @router.get("/decoding/catalog")
    def catalog():
        return {
            **catalogue(),
            "draft_candidates": [
                c.public()
                for c in service.configs.values()
                if c.provider == "pytorch" and c.manifest
            ],
        }

    @router.get("/decoding/status")
    def status(deployment_id: str):
        with service.lock:
            if deployment_id not in service.configs:
                raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
            state = service.states[deployment_id]
            if state["state"] != "ready":
                return {
                    "status": "target_not_loaded",
                    "deployment_id": None,
                    "parameter_bytes": 0,
                }
            if (
                service.configs[deployment_id].provider != "pytorch"
                or settings.privileged_worker
            ):
                raise ServiceError(
                    "L7_LOCAL_WORKER_REQUIRED", "需要本机非特权 PyTorch worker。", 422
                )
            if service.pending or service.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "请等待当前任务清理完成。")
            service.pending += 1
            future = service.executors[deployment_id].submit(
                service.actors[deployment_id].execute,
                "decoding_status",
                {},
                lambda _: None,
                lambda: service.closed,
                10,
            )
        try:
            return future.result(timeout=15)
        finally:
            with service.lock:
                service.pending -= 1

    @router.post("/decoding/draft/unload", response_model=RunRecord, status_code=202)
    def unload(
        body: DraftRelease, idempotency_key: str = Header(min_length=1, max_length=128)
    ):
        return service.submit(
            "draft_unload",
            body.deployment_id,
            {"expected_epoch": body.expected_epoch},
            idempotency_key,
        )

    @router.get("/decoding/runs")
    def runs(deployment_id: str):
        if deployment_id not in service.configs:
            raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
        with service.store.lock:
            rows = service.store.db.execute(
                "SELECT id FROM runs WHERE json_extract(body,'$.config.deployment.id')=? AND json_extract(body,'$.config.decoding_plan') IS NOT NULL ORDER BY created DESC LIMIT 60",
                (deployment_id,),
            ).fetchall()
        return {"items": [service.store.get(row[0]) for row in rows]}

    @router.post("/decoding/comparisons")
    def comparison(body: DecodingComparison):
        a, b = [
            service.store.get(i) for i in (body.baseline_run_id, body.candidate_run_id)
        ]
        try:
            left, right = [
                load_bundle(safe_directory(settings.data_dir / "captures", r["id"]))
                for r in (a, b)
            ]
        except (OSError, ValueError) as exc:
            raise ServiceError("L7_SEALED_RECEIPT_REQUIRED", str(exc), 409) from exc
        result = compare(a, b, left, right)
        folder = settings.data_dir / "l7-comparisons"
        folder.mkdir(exist_ok=True)
        identity = fingerprint(
            [body.model_dump(), left["manifest_sha256"], right["manifest_sha256"]]
        )
        if not (folder / (identity + ".json")).exists():
            atomic_json(folder / (identity + ".json"), result)
        return result

    return router
