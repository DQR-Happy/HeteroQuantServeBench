"""L3 metadata reads and comparisons. No tracing or GPU execution on GET."""
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from hqsb.console.captures import safe_directory
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.core.runtime_plan import catalogue
from hqsb.experiments.artifacts import read_manifest, read_json


class RuntimeComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str
    candidate_run_id: str


def runtime_router(settings, service):
    router = APIRouter(tags=["compiler-runtime"])

    @router.get("/runtime/catalog")
    def catalog():
        from hqsb.core.graph_views import FORMATS
        return {**catalogue(), "graph_formats": FORMATS, "limits": {"max_deadline_ms": settings.request_deadline_ms}}

    @router.get("/runtime/runs")
    def runs(deployment_id: str):
        if deployment_id not in service.configs:
            raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
        with service.store.lock:
            rows = service.store.db.execute(
                "SELECT id FROM runs WHERE json_extract(body, '$.config.deployment.id')=? "
                "AND json_extract(body, '$.config.runtime_plan') IS NOT NULL ORDER BY created DESC LIMIT 30",
                (deployment_id,)).fetchall()
        result = []
        for row in rows:
            run = service.store.get(row[0])
            result.append({"run_id": run["id"], "state": run["state"], "epoch": run["config"].get("epoch"),
                "plan": run["config"]["runtime_plan"], "error": run.get("error"),
                "role": run["config"].get("experiment_role", "generation"),
                "summary": run["metrics"].get("optimization_receipt", {}).get("runtime_receipt"),
                "output_tokens_per_s": run["metrics"].get("output_tokens_per_s"),
                "quality": run["metrics"].get("quality_evaluation")})
        return {"items": result}

    def read(run_id):
        try:
            run = service.store.get(run_id)
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", "Unknown run", 404) from exc
        if run["state"] not in TERMINAL:
            raise ServiceError("L3_RUN_NOT_TERMINAL", "Wait for cleanup and publication", 409)
        try:
            directory = safe_directory(settings.data_dir / "captures", run_id)
            manifest = read_manifest(directory)
            if not any(row["path"] == "runtime-graph.json" for row in manifest["artifacts"]):
                raise ValueError("L3_ARTIFACT_MISSING")
            doc = read_json(directory / "runtime-graph.json")
            if doc.get("run_id") != run_id or doc.get("epoch") != run["config"].get("epoch"):
                raise ValueError("L3_ARTIFACT_IDENTITY")
            return run, doc, manifest
        except (ValueError, OSError) as exc:
            raise ServiceError("L3_ARTIFACT_UNAVAILABLE", str(exc), 409) from exc

    @router.get("/runs/{run_id}/runtime-graph")
    def graph(run_id: str):
        run, doc, manifest = read(run_id)
        return {**doc, "publication": manifest["publication"], "run_state": run["state"],
                "deployment_id": run["config"]["deployment"]["id"]}

    @router.get("/runs/{run_id}/graph-views")
    def graph_views(run_id: str):
        run, doc, manifest = read(run_id)
        names = {a["path"] for a in manifest["artifacts"]}
        if "graph-views.json" not in names:
            return {"run_id": run_id, "epoch": doc["epoch"], "views": [], "status": "not_collected"}
        extra = read_json(safe_directory(settings.data_dir / "captures", run_id) / "graph-views.json")
        if extra.get("run_id") != run_id or extra.get("epoch") != run["config"].get("epoch") or extra.get("source_digest") != doc.get("source_digest"):
            raise ServiceError("GRAPH_VIEW_IDENTITY", "Graph view identity does not match run", 409)
        for view in extra["views"]:
            if view.get("status") == "available" and view.get("artifact") not in names:
                raise ServiceError("GRAPH_VIEW_ARTIFACT", "Unsealed graph artifact", 409)
            if view.get("artifact") in names:
                view["artifact_index"] = next(i for i, a in enumerate(manifest["artifacts"]) if a["path"] == view["artifact"])
        return {**extra, "publication": manifest["publication"]}

    @router.post("/runtime/comparisons")
    def compare(body: RuntimeComparison):
        a, x, ma = read(body.baseline_run_id)
        b, y, mb = read(body.candidate_run_id)
        reasons = []
        if a["config"].get("memory_plan") != b["config"].get("memory_plan"):
            reasons.append("memory_plan_mismatch")
        if a["config"].get("graph_capture") or b["config"].get("graph_capture"):
            reasons.append("diagnostic_capture_not_performance_sample")
        for k in ("model_identity", "tokenizer_identity", "source_digest"):
            if not x.get(k) or x[k] != y.get(k):
                reasons.append(k + "_mismatch")
        for name, run, manifest in (("baseline", a, ma), ("candidate", b, mb)):
            if run["state"] != "completed" or manifest["status"] != "complete":
                reasons.append(name + "_incomplete")
        receipts = [read_json(safe_directory(settings.data_dir / "captures", r["id"]) / "execution-receipt.json") for r in (a, b)]
        if any(receipt.get("status") != "applied" for receipt in receipts):
            reasons.append("execution_not_applied")
        for k in ("precision_receipt",):
            # Invocation counters vary per run; compare immutable recipe identity.
            if receipts[0].get(k, {}).get("plan_sha256") != receipts[1].get(k, {}).get("plan_sha256"):
                reasons.append("precision_plan_mismatch")
        for key in ("kernel", "dtype", "input_mode", "chat_template_sha256"):
            if receipts[0].get("runtime", {}).get(key) != receipts[1].get("runtime", {}).get(key):
                reasons.append(key + "_mismatch")
        if receipts[0].get("effective", {}).get("attention_backend") != receipts[1].get("effective", {}).get("attention_backend"):
            reasons.append("attention_mismatch")
        if a["config"].get("experiment_role") != b["config"].get("experiment_role"):
            reasons.append("measurement_role_mismatch")
        if a["config"].get("experiment_role") == "quality":
            from hqsb.experiments.artifacts import load_bundle
            from hqsb.experiments.quality_contract import compare_scores
            try:
                quality = compare_scores(*[load_bundle(safe_directory(settings.data_dir / "captures", r["id"])) for r in (a, b)])
            except ValueError as exc:
                reasons.append(str(exc))
                quality = None
        else:
            quality = None
            inputs = [r.get("proof", {}).get("input_token_sha256") for r in receipts]
            if not inputs[0] or inputs[0] != inputs[1]:
                reasons.append("input_tokens_mismatch")
            if a["metrics"].get("output_tokens") != b["metrics"].get("output_tokens"):
                reasons.append("generated_length_mismatch")
            if receipts[0].get("proof", {}).get("requested_token_limit") != receipts[1].get("proof", {}).get("requested_token_limit"):
                reasons.append("output_budget_mismatch")
            if receipts[0].get("effective", {}).get("cleanup_policy") != receipts[1].get("effective", {}).get("cleanup_policy"):
                reasons.append("cleanup_policy_mismatch")
        return {"schema_version": "l3-v1", "baseline_run_id": a["id"], "candidate_run_id": b["id"],
                "comparable": not reasons, "reasons": reasons,
                "quality": quality if not reasons else None,
                "runtime": [{k: d.get(k) for k in ("plan", "status", "new_graphs", "session_graphs", "graph_invocations", "backend_compile_ms", "sample_class")} for d in (x, y)],
                "graph_analysis": [d.get("analysis") for d in (x, y)],
                "performance": {"status": "descriptive_pair_only_no_speedup_claim",
                                "output_tokens_per_s": [r["metrics"].get("output_tokens_per_s") for r in (a, b)] if not reasons else None,
                                "runtime_e2e_ms": [r["metrics"].get("runtime_e2e_ms") for r in (a, b)] if not reasons else None},
                "limitations": ["Use repeated independent same-workload performance runs; compilation and graph metadata collection have overhead.",
                                "FX node counts do not establish physical fusion or memory/cache behavior."]}

    return router
