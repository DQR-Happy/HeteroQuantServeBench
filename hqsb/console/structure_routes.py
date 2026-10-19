"""L5 CPU jobs, explicit artifacts, real resident-model quality/performance pairs."""

import copy
from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict
from hqsb.core.structure_plan import StructurePlan, catalogue
from hqsb.console.schemas import RunRecord
from hqsb.console.service import ServiceError
from hqsb.console.captures import safe_directory
from hqsb.experiments.artifacts import load_bundle
from hqsb.experiments.quality_contract import (
    compare_scores,
    is_floating_reference,
    source_digest,
)
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json


class CompressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str
    plan: StructurePlan


class StructureComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str
    candidate_run_id: str


def structure_router(settings, service, jobs):
    router = APIRouter(tags=["model-structure"])

    @router.get("/structure/catalog")
    def catalog():
        return {**catalogue(), "max_deadline_ms": settings.request_deadline_ms}

    @router.get("/structure/capabilities")
    def capabilities(deployment_id: str):
        with service.lock:
            if deployment_id not in service.configs:
                raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
            if service.states[deployment_id]["state"] != "ready":
                raise ServiceError("DEPLOYMENT_NOT_READY", "请在设备页加载原模型。")
            if (
                service.configs[deployment_id].provider != "pytorch"
                or settings.privileged_worker
            ):
                raise ServiceError(
                    "L5_LOCAL_NATIVE_WORKER_REQUIRED",
                    "需要本机非特权 PyTorch 模型。",
                    422,
                )
            if service.pending or service.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "请先等待当前任务结束。")
            service.pending += 1
            future = service.executors[deployment_id].submit(
                service.actors[deployment_id].execute,
                "structure_capabilities",
                {},
                lambda _: None,
                lambda: service.closed,
                20,
            )
        try:
            return future.result(timeout=25)
        finally:
            with service.lock:
                service.pending -= 1

    @router.post("/structure/jobs", response_model=RunRecord, status_code=202)
    def create(
        body: CompressionRequest,
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        return jobs.start(
            body.deployment_id, body.expected_epoch, body.plan, idempotency_key
        )

    @router.get("/structure/jobs")
    def listing():
        return service.store.list(60, kind="compress")

    @router.get("/structure/artifacts")
    def artifacts(deployment_id: str):
        if deployment_id not in service.configs:
            raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
        with service.store.lock:
            rows = service.store.db.execute(
                "SELECT id FROM runs WHERE json_extract(body,'$.kind')='compress' AND json_extract(body,'$.state')='completed' AND json_extract(body,'$.config.deployment.id')=? ORDER BY created DESC LIMIT 64",
                (deployment_id,),
            ).fetchall()
        result = []
        for row in rows:
            run = service.store.get(row[0])
            doc = jobs.manifest(run["metrics"]["structure_artifact"]["artifact_id"])
            result.append(
                {
                    k: doc[k]
                    for k in (
                        "artifact_id",
                        "creation_run_id",
                        "recipe",
                        "statistics",
                        "transforms",
                        "disk_bytes",
                        "creation_seconds",
                        "publication",
                        "source_model_identity",
                    )
                }
            )
        return {"items": result}

    @router.get("/structure/artifacts/{identity}")
    def artifact(identity: str):
        try:
            return jobs.manifest(identity)
        except (ValueError, KeyError, OSError) as exc:
            raise ServiceError("L5_ARTIFACT_UNAVAILABLE", str(exc), 409) from exc

    @router.get("/structure/runs")
    def runs(deployment_id: str):
        with service.store.lock:
            rows = service.store.db.execute(
                "SELECT id FROM runs WHERE json_extract(body,'$.config.deployment.id')=? AND json_extract(body,'$.kind')='generate' AND (json_extract(body,'$.metrics.structure_receipt') IS NOT NULL OR json_extract(body,'$.metrics.optimization_receipt.structure_receipt') IS NOT NULL) ORDER BY created DESC LIMIT 60",
                (deployment_id,),
            ).fetchall()
        return {"items": [service.store.get(row[0]) for row in rows]}

    @router.post("/structure/comparisons")
    def comparison(body: StructureComparison):
        bundles = []
        for identity in (body.baseline_run_id, body.candidate_run_id):
            run = service.store.get(identity)
            if run["state"] != "completed" or run["cleanup"] != "succeeded":
                raise ServiceError(
                    "L5_CLEAN_COMPLETED_RUN_REQUIRED", "请选择已完成且清理成功的运行。"
                )
            try:
                bundle = load_bundle(
                    safe_directory(settings.data_dir / "captures", identity)
                )
            except (ValueError, OSError) as exc:
                raise ServiceError("L5_SEALED_RECEIPT_REQUIRED", str(exc)) from exc
            if (
                bundle["manifest"]["status"] != "complete"
                or bundle["receipt"]["status"] != "applied"
            ):
                raise ServiceError(
                    "L5_APPLIED_RECEIPT_REQUIRED", "运行尚未完整生效或封存。"
                )
            bundles.append((run, bundle))
        (a, left), (b, right) = bundles
        ra, rb = left["receipt"], right["receipt"]
        structure = rb.get("structure_receipt") or {}
        reasons = []
        if (
            (ra.get("structure_receipt") or {}).get("artifact_id")
            or not structure.get("artifact_id")
            or structure.get("status") != "invoked"
        ):
            reasons.append("original_baseline_and_invoked_structure_candidate_required")
        if structure.get("source_model_identity") != ra.get("model_identity") or rb.get(
            "model_identity", {}
        ).get("hash") != structure.get("artifact_manifest_sha256"):
            reasons.append("source_lineage_mismatch")
        for key in ("tokenizer_identity", "runtime"):
            if not ra.get(key) or ra.get(key) != rb.get(key):
                reasons.append(key + "_mismatch")
        if not source_digest(ra) or source_digest(ra) != source_digest(rb):
            reasons.append("runtime_source_mismatch")
        if ra.get("precision_receipt", {}).get("plan_sha256") != rb.get(
            "precision_receipt", {}
        ).get("plan_sha256"):
            reasons.append("precision_mismatch")
        if ra.get("effective", {}).get("attention_backend") != rb.get(
            "effective", {}
        ).get("attention_backend"):
            reasons.append("attention_mismatch")
        if a["config"].get("actual_runtime", {}).get("device_name") != b["config"].get(
            "actual_runtime", {}
        ).get("device_name"):
            reasons.append("device_mismatch")
        quality = left.get("quality") is not None and right.get("quality") is not None
        result = {
            "schema_version": "l5-comparison-v1",
            **body.model_dump(),
            "comparable": False,
            "reasons": reasons,
            "quality": None,
            "performance": None,
            "structure": structure,
            "limitations": [
                "One pair is descriptive, not a speedup claim. Lossy structure transforms may reduce quality."
            ],
        }
        if quality and not reasons:
            if not is_floating_reference(left):
                reasons.append("unmodified_float_baseline_required")
            evaluation = {
                s.get(k)
                for s in right["quality"]["samples"]
                for k in ("text_sha256", "scored_text_sha256")
                if s.get(k)
            }
            if evaluation & set(structure.get("calibration_text_sha256", [])):
                reasons.append("calibration_evaluation_leakage")
            candidate = copy.deepcopy(right)
            candidate["receipt"]["model_identity"] = ra["model_identity"]
            if not reasons:
                try:
                    result["quality"] = compare_scores(left, candidate)
                except ValueError as exc:
                    reasons.append(str(exc))
        elif not quality and not reasons:
            if left.get("quality") or right.get("quality"):
                reasons.append("measurement_role_mismatch")
            for key in ("input_token_sha256", "requested_token_limit"):
                if not ra.get("proof", {}).get(key) or ra["proof"][key] != rb.get(
                    "proof", {}
                ).get(key):
                    reasons.append(key + "_mismatch")
            for key in (
                "observation_mode",
                "observation_plan",
                "fixed_output_length",
                "runtime_plan",
            ):
                if a["config"].get(key) != b["config"].get(key):
                    reasons.append(key + "_mismatch")
            if a["metrics"].get("output_tokens") != b["metrics"].get("output_tokens"):
                reasons.append("output_length_mismatch")
            if ra.get("effective", {}).get("cleanup_policy") != rb.get(
                "effective", {}
            ).get("cleanup_policy"):
                reasons.append("cleanup_policy_mismatch")
            if not reasons:
                result["performance"] = {
                    "status": "descriptive_pair_no_speedup_claim",
                    **{
                        k: [r["metrics"].get(k) for r in (a, b)]
                        for k in (
                            "output_tokens_per_s",
                            "runtime_first_token_ms",
                            "runtime_e2e_ms",
                            "worker_request_total_ms",
                            "memory",
                        )
                    },
                    "parameter_bytes": [
                        r["config"].get("actual_runtime", {}).get("parameter_bytes")
                        for r in (a, b)
                    ],
                    "exact_generated_tokens": a["metrics"].get("generated_token_ids")
                    == b["metrics"].get("generated_token_ids"),
                }
        result["comparable"] = not reasons
        comparison_id = fingerprint(
            [body.model_dump(), left["manifest_sha256"], right["manifest_sha256"]]
        )
        path = settings.data_dir / "l5-comparisons" / f"{comparison_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            atomic_json(path, result)
        return result

    return router
