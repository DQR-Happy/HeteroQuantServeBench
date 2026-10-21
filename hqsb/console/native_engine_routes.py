"""Register native variants of verified local artifacts and submit owned jobs."""
import json
import subprocess
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from hqsb.console.config import DeploymentConfig
from hqsb.core.native_engine import VERSIONS, catalogue, resolve_options
from hqsb.observation.storage import atomic_json
from hqsb.serving.native_benchmark import NativeBenchmark


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_deployment_id: str
    engine: Literal["vllm", "sglang"]


class BenchmarkRequest(NativeBenchmark):
    deployment_id: str


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str
    candidate_run_id: str
    declared_changes: list[str] = Field(default_factory=list, max_length=32)


def native_engine_router(settings, service):
    router = APIRouter(tags=["native-engines"])
    registry = settings.data_dir / "native-deployments.json"
    inventories = {}
    if registry.exists():
        for row in json.loads(registry.read_text())["items"]:
            # Executable paths are always resolved from current operator settings.
            row["engine_python"] = settings.native_engine_pythons[row["provider"]]
            service.register_deployment(DeploymentConfig.model_validate(row))

    @router.get("/native-engines/catalog")
    def engines():
        items = []
        for engine, version in VERSIONS.items():
            cached_at, inventory = inventories.get(engine, (0, None))
            if time.monotonic() - cached_at > 30 or inventory is None:
                executable = settings.native_engine_pythons.get(engine, "")
                inventory = {"installed": False, "environment_status": "python_entrypoint_missing"}
                if Path(executable).is_file():
                    try:
                        probe = subprocess.run([executable, "-c", f"import importlib.metadata as m; print(m.version('{engine}'))"],
                                               capture_output=True, text=True, check=True, timeout=5)
                        actual = probe.stdout.strip()
                        inventory = {"installed": actual == version, "installed_version": actual,
                                     "environment_status": "package_version_verified" if actual == version else "version_mismatch"}
                    except (OSError, subprocess.SubprocessError) as exc:
                        inventory = {"installed": False, "environment_status": "package_probe_failed", "error": str(exc)[:300]}
                inventories[engine] = (time.monotonic(), inventory)
            items.append({**catalogue(engine), **inventory})
        return {"items": items}

    @router.post("/native-engines/deployments", status_code=201)
    def register(body: Registration):
        with service.lock:
            source = service.configs.get(body.source_deployment_id)
            if source is None or source.provider != "pytorch" or not source.manifest:
                raise HTTPException(422, "请选择带有 SHA256 清单的本地 PyTorch 模型部署")
            identity = f"{source.id[:55]}-{body.engine}"
            if identity in service.configs:
                existing = service.configs[identity]
                if existing.provider != body.engine or any(getattr(existing, name) != getattr(source, name)
                                                          for name in ("model_path", "manifest", "model", "input_mode")):
                    raise HTTPException(409, "部署身份已被其他制品使用，请先移除旧部署引用")
                return service.states[identity]
            metadata = Path(source.model_path).expanduser() / "config.json"
            dtype = source.precision
            if metadata.is_file() and metadata.stat().st_size <= 1024**2:
                try:
                    model_config = json.loads(metadata.read_text())
                    declared = model_config.get("dtype") or model_config.get("torch_dtype")
                    if isinstance(declared, str) and declared in {"float16", "bfloat16", "float32"}:
                        dtype = declared
                except (OSError, ValueError) as exc:
                    raise HTTPException(422, "请先修复制品 config.json：" + str(exc)[:200]) from exc
            config = DeploymentConfig.model_validate({**source.model_dump(), "id": identity,
                       "name": source.name + " · " + body.engine, "provider": body.engine,
                       "kernel_profile": "framework-native", "kernel_manifest": None,
                       "engine_python": settings.native_engine_pythons[body.engine],
                       "precision": dtype, "engine_options": resolve_options(body.engine, {"dtype": dtype})})
            service.register_deployment(config)
            atomic_json(registry, {"items": [c.model_dump(exclude={"engine_python"}) for c in service.configs.values()
                                             if c.provider in VERSIONS and c.id not in {s.id for s in settings.deployments}]})
            return service.states[identity]

    @router.get("/native-engines/deployments/{deployment_id}/receipt")
    def receipt(deployment_id: str):
        with service.lock:
            state = service.states.get(deployment_id)
            if state is None:
                raise HTTPException(404, "Unknown deployment")
            return {"state": state["state"], "epoch": state["epoch"],
                    "receipt": state["detail"].get("native_engine_receipt")}

    @router.post("/native-engines/deployments/{deployment_id}/retire")
    def retire(deployment_id: str):
        with service.lock:
            config = service.configs.get(deployment_id)
            if config is None or config.provider not in VERSIONS:
                raise HTTPException(404, "请选择一个已注册的原生部署")
            service.retire_deployments([deployment_id])
            atomic_json(registry, {"items": [c.model_dump(exclude={"engine_python"}) for c in service.configs.values()
                                             if c.provider in VERSIONS and c.id not in {s.id for s in settings.deployments}]})
            return {"id": deployment_id, "state": "retired", "historical_runs": "preserved"}

    @router.post("/native-engines/benchmarks", status_code=202)
    def benchmark(body: BenchmarkRequest, idempotency_key: str = Header(min_length=1, max_length=128)):
        return service.submit("native_benchmark", body.deployment_id,
                              body.model_dump(exclude={"deployment_id"}), idempotency_key)

    @router.get("/native-engines/benchmarks")
    def benchmarks():
        return service.store.list(100, kind="native_benchmark")

    @router.post("/native-engines/comparisons")
    def compare(body: Comparison):
        if body.baseline_run_id == body.candidate_run_id:
            raise HTTPException(422, "请选择两个不同的运行")
        from hqsb.core.native_engine import FIELDS
        if set(body.declared_changes) - set(FIELDS):
            raise HTTPException(422, "声明的变化应选择原生引擎参数目录中的字段")
        try:
            baseline, candidate = [service.store.get(identity) for identity in (body.baseline_run_id, body.candidate_run_id)]
            a, b = [r["metrics"]["native_benchmark"] for r in (baseline, candidate)]
        except KeyError as exc:
            raise HTTPException(422, "请选择已记录公共负载证据的两个运行") from exc
        mismatches = []
        for key in ("workload_sha256", "measurement_profile"):
            if a[key] != b[key]:
                mismatches.append(key)
        if a["engine"]["model_manifest_sha256"] != b["engine"]["model_manifest_sha256"]:
            mismatches.append("model_manifest_sha256")
        hardware = [r["engine"].get("hardware_inventory", {}) for r in (a, b)]
        if any(r.get("availability") != "measured" for r in hardware) or hardware[0].get("csv") != hardware[1].get("csv"):
            mismatches.append("hardware_inventory_unverified_or_different")
        for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
            if a["engine"].get("launch_environment", {}).get(name) != b["engine"].get("launch_environment", {}).get(name):
                mismatches.append("environment:" + name)
        changes = {key: {"baseline": a["engine"]["requested"][key], "candidate": b["engine"]["requested"][key]}
                   for key in FIELDS if a["engine"]["requested"][key] != b["engine"]["requested"][key]}
        fixed = set(FIELDS) if a["engine"].get("engine") == b["engine"].get("engine") else {"dtype", "quantization", "kv_cache_dtype", "prefix_cache", "seed", "enable_thinking"}
        for key in fixed:
            if key in changes and key not in body.declared_changes:
                mismatches.append(key)
        input_modes = [r["config"]["deployment"].get("input_mode", "chat") for r in (baseline, candidate)]
        if input_modes[0] != input_modes[1]:
            mismatches.append("input_mode")
        for run, receipt in ((baseline, a), (candidate, b)):
            summary = receipt["summary"]
            if run["state"] != "completed" or summary["failed_or_cancelled"] or not summary["arrival_valid"] or not summary["usage_complete"]:
                mismatches.append(run["id"] + ":incomplete_or_invalid_load")
        input_counts_a = [row["metrics"].get("input_tokens") for row in a["requests"]]
        input_counts_b = [row["metrics"].get("input_tokens") for row in b["requests"]]
        if input_counts_a != input_counts_b or any(n is None for n in input_counts_a):
            mismatches.append("actual_prompt_token_counts")
        qa, qb = [r["summary"].get("quality_exact_match_rate") for r in (a, b)]
        quality_gate = "not_evaluated" if qa is None or qb is None else "passed" if qb >= qa else "failed"
        return {"baseline_run_id": baseline["id"], "candidate_run_id": candidate["id"],
                "comparable": not mismatches, "mismatches": mismatches,
                "declared_changes": body.declared_changes, "configuration_changes": changes,
                "quality_gate": {"status": quality_gate, "baseline_score": qa, "candidate_score": qb, "max_accuracy_drop": 0},
                "eligible_for_adoption": not mismatches and quality_gate == "passed",
                "output_tokens_per_s_ratio": b["summary"]["output_tokens_per_s"] / a["summary"]["output_tokens_per_s"]
                    if not mismatches and a["summary"]["output_tokens_per_s"] else None,
                "baseline": a["summary"], "candidate": b["summary"],
                "quality_scope": "exact_match_fixture" if a["plan"]["expected_outputs"] else "outputs_require_task_quality_evaluation",
                "interpretation": "single_run_whole_engine_comparison_repeat_trials_for_uncertainty"}

    return router
