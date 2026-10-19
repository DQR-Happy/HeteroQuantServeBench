"""L4 quality jobs and sealed, source-aligned comparisons; device work stays in actors."""

import json
import re
import threading
import uuid
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from hqsb.core.runtime_plan import RuntimePlan
from hqsb.console.captures import safe_directory
from hqsb.console.schemas import RunRecord
from hqsb.core.optimization import validate_quality_plan, fingerprint
from hqsb.experiments.artifacts import load_bundle
from hqsb.experiments.comparison import _quality
from hqsb.experiments.quality_contract import check_executor
from hqsb.observation.storage import atomic_json


class QualityLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_logit_abs_error: float = Field(ge=0, le=100)
    max_kl: float = Field(ge=0, le=100)
    max_nll_increase: float = Field(ge=0, le=100)
    max_accuracy_drop: float | None = Field(None, ge=0, le=1)


class PerplexityLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_mean_nll_increase: float = Field(ge=0, le=20)


class PrecisionQualityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str
    quality_plan: dict
    runtime_plan: RuntimePlan | None = None
    quality_limits: QualityLimits | None = None
    ppl_limits: PerplexityLimits | None = None


class PrecisionComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str
    candidate_run_ids: list[str] = Field(min_length=1, max_length=8)


class BaselineRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str


def precision_router(settings, service):
    router = APIRouter(tags=["precision"])
    root = settings.data_dir / "precision-comparisons"
    from hqsb.console.quality_baselines import QualityBaselines

    baselines = None
    baseline_lock = threading.Lock()

    def baseline_store():
        # Read-only evidence apps may expose a partial service. Require the
        # run store only when these endpoints are actually used.
        nonlocal baselines
        with baseline_lock:
            if baselines is None:
                baselines = QualityBaselines(settings.data_dir, service.store)
        return baselines

    @router.get("/runs/{run_id}/quality-baseline")
    def quality_baseline(run_id: str):
        try:
            return baseline_store().for_run(run_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown quality run") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/precision/quality-baselines", status_code=201)
    def register_baseline(body: BaselineRegistration):
        try:
            return baseline_store().register(body.run_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown quality run") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/precision/quality-benchmarks")
    def quality_benchmarks():
        from hqsb.core.corpus_benchmark import catalogue

        return {"items": catalogue()}

    @router.post("/precision/quality-jobs", status_code=202, response_model=RunRecord)
    def quality_job(
        body: PrecisionQualityRequest,
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        try:
            plan = validate_quality_plan(body.quality_plan)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        corpus = plan.get("score_mode") == "corpus_nll"
        if (corpus and body.quality_limits is not None) or (
            not corpus and body.ppl_limits is not None
        ):
            raise HTTPException(422, "QUALITY_LIMITS_DO_NOT_MATCH_SCORING_PROTOCOL")
        return service.submit(
            "generate",
            body.deployment_id,
            {
                "expected_epoch": body.expected_epoch,
                "prompt": plan["samples"][0]["text"],
                "max_output_tokens": plan["generation_tokens"],
                "deadline_ms": settings.request_deadline_ms,
                "experiment_role": "quality",
                "observation_mode": "off",
                "quality_plan": plan,
                "quality_limits": body.quality_limits.model_dump()
                if body.quality_limits
                else None,
                "ppl_limits": body.ppl_limits.model_dump() if body.ppl_limits else None,
                "save_input": False,
                **({"runtime_plan": body.runtime_plan.model_dump()} if body.runtime_plan else {}),
            },
            idempotency_key,
        )

    @router.post("/precision/comparisons", status_code=201)
    def compare(body: PrecisionComparison):
        run_ids = [body.baseline_run_id] + body.candidate_run_ids
        if len(set(run_ids)) != len(run_ids):
            raise HTTPException(422, "L4_COMPARISON_REQUIRES_DISTINCT_RUNS")
        bundles = []
        for identity in run_ids:
            if not re.fullmatch(r"run_[0-9a-f]{32}", identity):
                raise HTTPException(422, "Invalid run ID")
            run = service.store.get(identity)
            if run["state"] != "completed" or run["cleanup"] != "succeeded":
                raise HTTPException(409, "L4_COMPARISON_REQUIRES_COMPLETED_CLEAN_RUNS")
            path = safe_directory(settings.data_dir / "captures", identity)
            try:
                bundle = load_bundle(path)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(
                    409, "L4_SEALED_EVIDENCE_REQUIRED:" + str(exc)
                ) from exc
            if bundle["manifest"]["status"] != "complete":
                raise HTTPException(409, "L4_COMPLETE_SEALED_EVIDENCE_REQUIRED")
            bundles.append((path, run, bundle))
        left_dir, left_run, baseline = bundles[0]
        candidates = []
        for path, run, candidate in bundles[1:]:
            a, b = baseline["receipt"], candidate["receipt"]
            try:
                check_executor(a, b)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            limits = left_run["config"].get("quality_limits")
            if run["config"].get("quality_limits") != limits:
                raise HTTPException(409, "L4_PREDECLARED_QUALITY_LIMITS_MISMATCH")
            if not baseline["quality"] or not candidate["quality"]:
                raise HTTPException(409, "L4_QUALITY_JOB_REQUIRED")
            evaluation_hashes = {
                s.get(key)
                for s in candidate["quality"]["samples"]
                for key in ("text_sha256", "scored_text_sha256")
                if s.get(key)
            }
            calibration_hashes = set(
                b.get("precision_receipt", {}).get("calibration_text_sha256", [])
            )
            if calibration_hashes & evaluation_hashes:
                raise HTTPException(409, "L4_EXACT_CALIBRATION_EVALUATION_LEAKAGE")
            if any(
                q.get("score_mode") == "corpus_nll"
                for q in (baseline["quality"], candidate["quality"])
            ):
                from hqsb.core.corpus_benchmark import compare as compare_corpus

                ppl_limits = left_run["config"].get("ppl_limits")
                if ppl_limits != run["config"].get("ppl_limits"):
                    raise HTTPException(409, "L4_PREDECLARED_PPL_LIMITS_MISMATCH")
                try:
                    measured = compare_corpus(
                        baseline["quality"],
                        candidate["quality"],
                        ppl_limits["max_mean_nll_increase"] if ppl_limits else None,
                    )
                except ValueError as exc:
                    raise HTTPException(409, str(exc)) from exc
                candidates.append(
                    {
                        "run_id": run["id"],
                        "precision": b.get("precision_receipt"),
                        "quality": measured,
                    }
                )
                continue
            measured = _quality(
                left_dir,
                path,
                baseline["quality"],
                candidate["quality"],
                (
                    {
                        **limits,
                        "max_accuracy_drop": limits.get("max_accuracy_drop")
                        if limits.get("max_accuracy_drop") is not None
                        else 1.0,
                    }
                )
                if limits
                else dict(
                    max_logit_abs_error=100.0,
                    max_kl=100.0,
                    max_nll_increase=100.0,
                    max_accuracy_drop=1.0,
                ),
            )
            for row in measured.get("positions", []):
                # The shared historical evaluator names this KL field after
                # FP16; L4 also accepts FP32/BF16 baselines. Keep its old field
                # for compatibility and expose the actual comparison semantics.
                row["kl_baseline_candidate"] = row["kl_fp16_quant"]
            if limits is not None and measured["status"] in {"passed", "failed"}:
                measured["limits"] = limits
                if (
                    limits.get("max_accuracy_drop") is not None
                    and measured["accuracy_drop"] is None
                ):
                    measured["status"] = (
                        "failed" if measured["status"] == "failed" else "inconclusive"
                    )
                    measured["reason"] = (
                        "TASK_ACCURACY_LABELS_REQUIRED_FOR_DECLARED_LIMIT"
                    )
            if limits is None and measured["status"] in {"passed", "failed"}:
                measured.update(status="measured_no_acceptance_limits", limits=None)
                for row in measured["positions"]:
                    row.pop("passed", None)
            candidates.append(
                {
                    "run_id": run["id"],
                    "precision": b.get("precision_receipt"),
                    "quality": measured,
                }
            )
        identity = "l4cmp_" + uuid.uuid4().hex
        result = {
            "schema_version": "l4-comparison-v1",
            "id": identity,
            "baseline_run_id": body.baseline_run_id,
            "baseline_precision": baseline["receipt"].get("precision_receipt"),
            "candidates": candidates,
            "performance": "separate_performance_runs_required_quality_jobs_are_not_timed_as_benchmarks",
            "leakage_scope": "exact_normalized_text_only_not_near_duplicate_detection",
            "request_sha256": fingerprint(body.model_dump()),
        }
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / (identity + ".json"), result)
        return result

    @router.get("/precision/comparisons/{identity}")
    def comparison(identity: str):
        if not re.fullmatch(r"l4cmp_[0-9a-f]{32}", identity):
            raise HTTPException(422, "Invalid L4 comparison ID")
        try:
            return json.loads((root / (identity + ".json")).read_text())
        except FileNotFoundError as exc:
            raise HTTPException(404, "Unknown L4 comparison") from exc

    return router
