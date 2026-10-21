"""Freeze a saved, verified PyTorch inference as a DMAT replay workload."""
from copy import deepcopy
from pathlib import Path

from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import sha256


def pytorch_reference(source, deployment):
    config = source["config"]
    if source["kind"] != "generate" or source["state"] != "completed" or deployment.provider != "pytorch":
        raise ValueError("DMAT_SOURCE_REQUIRED: 请选择已完成并保存输入的 PyTorch 推理记录。")
    if not config.get("messages") and not config.get("prompt"):
        raise ValueError("DMAT_SAVED_INPUT_REQUIRED: 请在推理页启用保存输入，完成一次推理后再选择记录。")
    for key in ("runtime_plan", "memory_plan", "decoding_plan", "capture_architecture", "graph_capture"):
        if config.get(key):
            raise ValueError("DMAT_REPLAY_ADAPTER_REQUIRED:" + key + "；请先选择标准贪心推理记录。")
    receipt = source["metrics"]["optimization_receipt"]
    model_hash = receipt["model_identity"]["hash"]
    if not deployment.manifest or sha256(Path(deployment.manifest).expanduser()) != model_hash:
        raise ValueError("DMAT_SOURCE_MANIFEST_CHANGED: 请使用当前制品重新推理。")
    actual = config["actual_runtime"]
    implementation = actual.get("kernel_plan", {}).get("triton_implementation")
    if implementation:
        current = Path(__file__).resolve().parents[1] / "backends/model_kernels/triton_attention.py"
        if sha256(current) != implementation["source_sha256"]:
            raise ValueError("DMAT_SOURCE_KERNEL_CHANGED: 算子实现已更新，请重新加载方案并保存新的推理记录。")
    precision = actual.get("parameter_dtypes", [receipt.get("runtime", {}).get("dtype", deployment.precision)])
    if len(precision) != 1 or precision[0] not in {"float16", "float32", "bfloat16"}:
        raise ValueError("DMAT_FLOAT_REPLAY_REQUIRED: 请先选择浮点权重推理记录。")
    load = {"kernel_profile": actual["kernel_plan"]["profile_id"], "precision": precision[0]}
    if actual.get("structure_receipt", {}).get("status", "original") not in {"original", "reference"}:
        raise ValueError("DMAT_STRUCTURE_REPLAY_ADAPTER_REQUIRED")
    request = {k: deepcopy(config[k]) for k in ("messages", "prompt", "max_output_tokens", "fixed_output_length", "optimization", "cleanup_policy") if k in config}
    request.update(input_mode="completion" if request.get("prompt") else "chat", observation_mode="off")
    workload = {"request": request, "concurrency": 1}
    environment = actual["source_environment"]
    engine = {"engine": "pytorch", "model_manifest_sha256": model_hash,
              "kernel_plan_sha256": fingerprint(actual["kernel_plan"]),
              "requested": {"load_options": load, "attention": config["deployment"]["attention"]},
              "versions": environment["packages"], "hardware_inventory": {"device_name": actual["device_name"], "machine": environment["machine"]},
              "environment_freeze_sha256": None}
    return {"plan": workload, "engine": engine, "workload_sha256": fingerprint(workload),
            "requests": [{"text": source["output"]}], "source_output_token_ids": source["metrics"].get("generated_token_ids")}
