"""Executor receipts: resolved configuration alone is never application proof."""

import time
from pathlib import Path

from hqsb.core.optimization import fingerprint, resolve_plan
from hqsb.observation.storage import atomic_json


def resolve_request(config, request):
    kernel_profile = config.get("kernel_profile", "framework-native")
    extra = {"kernel_profile": kernel_profile} if kernel_profile != "framework-native" else {}
    plan = resolve_plan(request.get("optimization"), {
        "cleanup_policy": request.get("cleanup_policy", "empty_cache"),
        "attention_backend": config["attention"],
        "output_budget": request["max_output_tokens"],
        **extra,
    }, deployment_attention=config["attention"], deployment_kernel_profile=kernel_profile)
    request = {**request, "cleanup_policy": plan["resolved"]["cleanup_policy"],
               "max_output_tokens": plan["resolved"]["output_budget"]}
    return request, plan


def receipt(provider, request, plan, *, role, output_hash, status, cleanup, proof):
    attention = getattr(provider.model.config, "_attn_implementation", None)
    applied = status == "completed" and cleanup == "succeeded" and attention == plan["resolved"]["attention_backend"]
    kernels = getattr(provider, "kernel_session", None)
    kernel_summary = kernels.summary() if kernels else None
    if kernels:
        kernels.save(request.get("capture_dir"))
        applied = applied and kernel_summary["status"] in {"invoked", "reference"}
    kernel_id = kernel_summary["profile_id"] if kernel_summary else "framework-native"
    from hqsb.backends.precision_runtime import receipt as precision_receipt
    document = {**plan, "status": "applied" if applied else "not_fully_applied",
                "run_id": request.get("capture_id"), "measurement_role": role,
                "request_id": request.get("capture_id"),
                "model_identity": provider.source_identity,
                "source": provider.source_environment,
                "tokenizer_identity": provider.tokenizer_identity,
                "runtime": {"dtype": (getattr(provider, "config", None) or {}).get("precision", "float16"), "kernel": kernel_id, "executor": "InteractivePyTorch",
                            "input_mode": request.get("input_mode", "chat"),
                            "chat_template_sha256": (getattr(provider, "chat_template_policy", None) or {}).get("template_sha256")
                            if request.get("input_mode", "chat") == "chat" else None},
                "kernel_receipt": kernel_summary,
                "precision_receipt": precision_receipt(provider),
                "experiment_links": request.get("experiment_links", {}),
                "effective": {**plan["resolved"], "attention_backend": attention},
                "output_token_hash": output_hash, "execution_status": status,
                "cleanup_status": cleanup, "host_monotonic_ns": str(time.monotonic_ns()),
                "proof": proof, "proof_scope": "executor_control_flow_and_model_config_not_instruction_trace"}
    if request.get("capture_dir"):
        atomic_json(Path(request["capture_dir"]) / "execution-receipt.json", document)
    return {key: value for key, value in document.items() if key != "source"} | {
        "source_digest": document["source"]["source_digest"], "receipt_sha256": fingerprint(document)}
