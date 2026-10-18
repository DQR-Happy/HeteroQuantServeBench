"""Model-load L4 composition: policy, official PTQ, compact execution and proof."""

import gc
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from hqsb.core.precision import PrecisionPlan, catalogue, resolve_exclusions
from hqsb.core.optimization import fingerprint


def probe():
    import torch

    result = catalogue()
    sm = torch.cuda.get_device_capability()
    result.update(device=torch.cuda.get_device_name(), sm=list(sm), versions={})
    for package in (
        "torch",
        "transformers",
        "bitsandbytes",
        "triton",
        "optimum-quanto",
    ):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][package] = None
    calibration = (
        Path(__file__).resolve().parents[2] / ".venv-l4-calibration/bin/python"
    )
    for method in result["methods"]:
        reasons = []
        if sm[0] * 10 + sm[1] < method["minimum_cuda_sm"]:
            reasons.append("DEVICE_SM_BELOW_REQUIRED")
        package = (
            "bitsandbytes"
            if method["id"].startswith("bnb_")
            else "triton"
            if method["execution"] == "hqsb_fused_dequant"
            else None
        )
        if package and not result["versions"].get(package):
            reasons.append("DEPENDENCY_MISSING:" + package)
        method.update(
            calibration_available=not method["requires_calibration"]
            or calibration.is_file(),
            availability="available_for_target_validation"
            if not reasons
            else "unsupported",
            reasons=reasons,
        )
    return result


def _replace(model, plan):
    import torch
    from hqsb.backends.precision_linear import PackedLinear, FP8Linear, encode_weight

    excluded = resolve_exclusions(
        [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)],
        plan.exclude_modules,
    )
    selected = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n not in excluded
    ]
    if not selected:
        raise ValueError("L4_NO_LINEAR_TARGETS")
    for name, module in selected:
        if plan.method == "fp8":
            replacement = FP8Linear(module.weight.detach(), module.bias)
        else:
            packed, scale, zero, group = encode_weight(
                module.weight.detach(), plan.bits, plan.group_size
            )
            replacement = PackedLinear(
                packed, scale, zero, module.in_features, group, plan.bits, module.bias
            )
        model.set_submodule(name, replacement)
    return model


def _native_records(model):
    import torch
    from hqsb.backends.precision_linear import PackedLinear, Int8Linear

    records, payload = [], {}
    for name, module in model.named_modules():
        if not isinstance(module, (PackedLinear, Int8Linear)):
            continue
        i = len(records)
        bits = module.bits if isinstance(module, PackedLinear) else 8
        payload[f"q{i}"] = (
            module.packed
            if isinstance(module, PackedLinear)
            else module.codes[: module.out_features, : module.in_features].view(
                torch.uint8
            )
        )
        payload[f"s{i}"] = module.scales
        if isinstance(module, PackedLinear):
            payload[f"z{i}"] = module.zeros
        records.append(
            {
                "name": name,
                "k": module.in_features,
                "n": module.out_features,
                "bits": bits,
                "group_size": module.group_size
                if isinstance(module, PackedLinear)
                else module.in_features,
                "codes": f"q{i}",
                "scales": f"s{i}",
                "zeros": f"z{i}",
                "execution": "hqsb_fused_dequant"
                if isinstance(module, PackedLinear)
                else "cuda_int8_mm",
            }
        )
    return records, payload


def load(config, plan, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from hqsb.models.loader import load_causal_lm
    from hqsb.models.manifest import verify_or_raise
    from hqsb.backends.precision_artifacts import (
        verify,
        sha256,
        export_native,
        publish,
        load_native,
        ensure_capacity,
    )

    plan = PrecisionPlan.model_validate(plan)
    method, started = plan.method, time.monotonic()
    capability = next(m for m in probe()["methods"] if m["id"] == method)
    if capability["reasons"]:
        raise ValueError("L4_UNSUPPORTED:" + ",".join(capability["reasons"]))
    if method in {"rtn", "gptq", "awq"} and dtype != torch.float16:
        raise ValueError("L4_FUSED_DEQUANT_REQUIRES_FLOAT16")
    if method != "none" and dtype == torch.float32:
        raise ValueError("L4_QUANTIZED_EXECUTION_REQUIRES_FP16_OR_BF16")
    root = Path(config["model_path"]).expanduser().resolve()
    if config.get("manifest"):
        verify_or_raise(
            str(root),
            config["manifest"],
            strict_extra=True,
            allow_extra=tuple(config.get("manifest_allow_extra", [])),
        )
        source = sha256(Path(config["manifest"]).expanduser())
    else:
        raise ValueError("L4_REQUIRES_VERIFIED_SOURCE_MANIFEST")
    package_versions = {
        p: importlib.metadata.version(p) for p in ("torch", "transformers")
    }
    if method.startswith("bnb_"):
        package_versions["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    identity = {
        "source_model_hash": source,
        "plan": plan.identity(),
        "compute_dtype": str(dtype),
        "runtime_versions": package_versions,
    }
    identity["codec_sources"] = {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "precision_linear.py",
            "precision_artifacts.py",
            "precision_calibration.py",
        )
    }
    artifact_id = plan.artifact_id or fingerprint(identity)
    artifact_root = Path(config["l4_artifact_root"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    destination = artifact_root / artifact_id
    manifest = None
    if destination.exists():
        if destination.is_symlink():
            raise ValueError("L4_ARTIFACT_PATH_ESCAPE")
        manifest = verify(destination)
        if (
            manifest["source_model_hash"] != source
            or manifest["method"] != method
            or manifest["compute_dtype"] != str(dtype)
        ):
            raise ValueError("L4_ARTIFACT_SOURCE_OR_PRECISION_MISMATCH")
        if not plan.artifact_id and manifest["identity_sha256"] != fingerprint(
            identity
        ):
            raise ValueError("L4_ARTIFACT_PLAN_MISMATCH")
        if plan.artifact_id:
            stored = {
                **manifest["recipe"],
                "calibration": [],
                "artifact_id": artifact_id,
                "allow_tf32": plan.allow_tf32,
                "kv_bits": plan.kv_bits,
                "kv_group_size": plan.kv_group_size,
                "kv_residual_length": plan.kv_residual_length,
            }
            plan = PrecisionPlan.model_validate(stored)
    elif plan.artifact_id:
        raise ValueError("L4_ARTIFACT_NOT_FOUND")
    remaining_budget = config.get("l4_artifact_budget_bytes", 8 * 1024**3)
    if manifest is None and method != "none":
        used = sum(f.stat().st_size for f in artifact_root.rglob("*") if f.is_file())
        if shutil.disk_usage(artifact_root).free < 512 * 1024**2:
            raise ValueError("L4_INSUFFICIENT_DISK_FOR_ARTIFACT")
        remaining_budget = (
            config.get("l4_artifact_budget_bytes", 8 * 1024**3) - used - 4 * 1024**2
        )
        if remaining_budget <= 0:
            raise ValueError("L4_ARTIFACT_STORE_BUDGET_EXCEEDED")
    metadata = {
        "artifact_id": artifact_id,
        "source_model": config.get("model"),
        "method": method,
        "source_model_hash": source,
        "compute_dtype": str(dtype),
        "identity_sha256": fingerprint(identity),
        "recipe": plan.identity(),
        "calibration_text_sha256": [
            fingerprint(" ".join(s.split())) for s in plan.calibration
        ],
        "runtime_versions": package_versions,
        "max_artifact_bytes": remaining_budget
        if manifest is None
        else manifest.get("max_artifact_bytes", 8 * 1024**3),
    }
    if method in {"gptq", "awq", "smoothquant"} and manifest is None:
        calibration = (
            Path(__file__).resolve().parents[2] / ".venv-l4-calibration/bin/python"
        )
        if not calibration.is_file():
            raise ValueError(
                "L4_CALIBRATION_ENV_MISSING: install the isolated PTQ environment or load a saved artifact"
            )
        job = artifact_root / ("job-" + uuid.uuid4().hex)
        job.mkdir()
        try:
            (job / "request.json").write_text(
                json.dumps(
                    {
                        "model_path": str(root),
                        "plan": plan.model_dump(),
                        "destination": str(destination),
                        "metadata": metadata,
                    }
                )
            )
            with (job / "calibration.log").open("w") as log:
                outcome = subprocess.run(
                    [
                        str(calibration),
                        "-m",
                        "hqsb.backends.precision_calibration",
                        str(job / "request.json"),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                    },
                    timeout=240,
                )
            if outcome.returncode:
                raise RuntimeError(
                    "L4_CALIBRATION_FAILED ["
                    + job.name
                    + "]: "
                    + (job / "calibration.log").read_text()[-600:]
                )
            manifest = verify(destination)
        finally:
            (job / "request.json").unlink(
                missing_ok=True
            )  # do not retain plaintext calibration prompts in logs
    if manifest and manifest["format"] == "hqsb-native":
        tokenizer, model, manifest = load_native(
            destination, dtype, config["attention"]
        )
    elif method.startswith("bnb_"):
        kwargs = (
            dict(load_in_8bit=True, llm_int8_threshold=plan.outlier_threshold)
            if method == "bnb_int8"
            else dict(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4" if method == "bnb_nf4" else "fp4",
                bnb_4bit_use_double_quant=plan.double_quant,
                bnb_4bit_compute_dtype=dtype,
            )
        )
        options = BitsAndBytesConfig(
            **kwargs, llm_int8_skip_modules=plan.exclude_modules
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(destination if manifest else root),
            dtype=dtype,
            device_map={"": 0},
            **({} if manifest else {"quantization_config": options}),
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=config["attention"],
        ).eval()
        excluded = resolve_exclusions(
            [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)],
            plan.exclude_modules,
        )
        if any(
            type(model.get_submodule(n)).__module__.startswith("bitsandbytes")
            for n in excluded
        ):
            raise ValueError("L4_LIBRARY_DID_NOT_HONOR_EXCLUSIONS")
        if manifest is None:
            partial = destination.with_name(
                destination.name + ".partial-" + uuid.uuid4().hex
            )
            try:
                state = model.state_dict()
                storages = {
                    (
                        str(t.device),
                        t.untyped_storage().data_ptr(),
                    ): t.untyped_storage().nbytes()
                    for t in state.values()
                }
                ensure_capacity(partial, sum(storages.values()), metadata)
                del state
                model.save_pretrained(partial, safe_serialization=True)
                tokenizer.save_pretrained(partial)
                manifest = publish(
                    partial,
                    destination,
                    {**metadata, "format": "transformers-bitsandbytes"},
                )
            finally:
                if partial.exists():
                    shutil.rmtree(partial)
    else:
        tokenizer, model, _ = load_causal_lm(
            str(root),
            dtype=dtype,
            attention_backend=config["attention"],
            cpu_staging=True,
            verify_manifest=config.get("manifest"),
            allow_extra=tuple(config.get("manifest_allow_extra", [])),
        )
        if method in {"rtn", "fp8"}:
            _replace(model, plan)
        if method == "rtn":
            records, payload = _native_records(model)
            manifest = export_native(
                model, tokenizer, destination, records, payload, metadata
            )
    gc.collect()
    torch.cuda.synchronize()
    model._hqsb_l4_plan = plan
    effective_metadata = {
        **metadata,
        **{
            k: v
            for k, v in (manifest or {}).items()
            if k
            in {"identity_sha256", "recipe", "source_model_hash", "runtime_versions"}
        },
    }
    return (
        tokenizer,
        model,
        time.monotonic() - started,
        {
            **effective_metadata,
            "manifest": manifest,
            "artifact_id": artifact_id if manifest else None,
            "requested": plan.model_dump(exclude={"calibration"}),
            "calibration_dataset_sha256": fingerprint(plan.calibration)
            if plan.calibration
            else (manifest or {}).get("recipe", {}).get("calibration"),
            "calibration_text_sha256": (manifest or metadata).get(
                "calibration_text_sha256", []
            ),
            "actual_dtype": str(model.dtype),
            "capability": capability,
        },
    )


def receipt(provider):
    if not hasattr(provider, "precision_plan"):
        return {"schema_version": "l4-receipt-v1", "status": "not_observed"}
    import torch
    from hqsb.backends.precision_linear import PackedLinear, Int8Linear, FP8Linear
    from hqsb.backends.tensor_memory import quantization_auxiliaries

    rows, tensors = [], {}
    for name, module in provider.model.named_modules():
        is_bnb = type(module).__module__.startswith("bitsandbytes.nn") and type(
            module
        ).__name__ in {"Linear4bit", "Linear8bitLt"}
        if not isinstance(module, (PackedLinear, Int8Linear, FP8Linear)) and not is_bnb:
            continue
        storage = []
        for key, t in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)
        ):
            pointer = (str(t.device), t.untyped_storage().data_ptr())
            tensors[pointer] = t.untyped_storage().nbytes()
            storage.append(
                {
                    "role": key,
                    "dtype": str(t.dtype),
                    "shape": list(t.shape),
                    "bytes": t.untyped_storage().nbytes(),
                }
            )
        if is_bnb:
            for key, t in quantization_auxiliaries(module):
                tensors[(str(t.device), t.untyped_storage().data_ptr())] = (
                    t.untyped_storage().nbytes()
                )
                storage.append(
                    {
                        "role": key,
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "bytes": t.untyped_storage().nbytes(),
                    }
                )
        rows.append(
            {
                "name": name,
                "implementation": type(module).__module__ + "." + type(module).__name__,
                "calls": getattr(module, "calls", None),
                "logical_weight_shape": [module.out_features, module.in_features],
                "bits": getattr(
                    module, "bits", 4 if type(module).__name__ == "Linear4bit" else 8
                ),
                "group_size": getattr(
                    module,
                    "group_size",
                    getattr(
                        getattr(module.weight, "quant_state", None), "blocksize", None
                    ),
                )
                if is_bnb
                else getattr(module, "group_size", None),
                "storage": storage,
                "proof_scope": "module_binding_and_calls_not_instruction_trace",
            }
        )
    plan = getattr(provider, "precision_plan", PrecisionPlan())
    library = getattr(provider.model.config, "quantization_config", None)
    if hasattr(library, "to_dict"):
        library = library.to_dict()
    actual_recipe = dict(
        (getattr(provider, "precision_load", {}).get("manifest") or {}).get(
            "recipe", plan.identity()
        )
    )
    for field in ("kv_bits", "kv_group_size", "kv_residual_length", "allow_tf32"):
        actual_recipe[field] = getattr(plan, field)
    return {
        "schema_version": "l4-receipt-v1",
        "method": plan.method,
        "compute_dtype": provider.config["precision"],
        "plan_sha256": fingerprint(actual_recipe),
        "resolved_recipe": actual_recipe,
        "artifact_id": getattr(provider, "precision_load", {}).get("artifact_id"),
        "calibration_text_sha256": getattr(provider, "precision_load", {}).get(
            "calibration_text_sha256", []
        ),
        "quantized_module_count": len(rows),
        "modules": rows,
        "quantized_storage_bytes": sum(tensors.values()),
        "effective_library_config": library,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "remaining_float_parameter_bytes": sum(
            {
                (
                    str(p.device),
                    p.untyped_storage().data_ptr(),
                ): p.untyped_storage().nbytes()
                for p in provider.model.parameters()
                if p.is_floating_point()
                and (str(p.device), p.untyped_storage().data_ptr()) not in tensors
            }.values()
        ),
        "storage_scope": "deduplicated_selected_parameters_buffers_and_known_bnb_quant_state_not_allocator_total",
        "execution": "floating_reference"
        if plan.method == "none"
        else "fused_dequant_weight_only"
        if plan.method in {"rtn", "gptq", "awq"}
        else "native_int8_w8a8"
        if plan.method == "smoothquant"
        else "native_fp8_w8a8"
        if plan.method == "fp8"
        else "bitsandbytes_quantized_linear",
        "kv": {
            "bits": plan.kv_bits,
            "backend": "quanto" if plan.kv_bits else "model_default",
            "residual_length": plan.kv_residual_length if plan.kv_bits else None,
        },
        "fallbacks": [],
        "quality": "requires_independent_evaluation",
    }


def reset_calls(provider):
    if hasattr(provider, "precision_plan"):
        for module in provider.model.modules():
            if hasattr(module, "calls"):
                module.calls = 0


def new_cache(provider):
    plan = getattr(provider, "precision_plan", None)
    if plan is None or plan.kv_bits is None:
        return None
    from transformers import QuantizedCache

    config = provider.model.config.get_text_config(decoder=True)
    if getattr(config, "sliding_window", None) or any(
        x != "full_attention" for x in (getattr(config, "layer_types", None) or [])
    ):
        raise ValueError("L4_KV_QUANTIZATION_REQUIRES_FULL_ATTENTION")
    return QuantizedCache(
        backend="quanto",
        config=config,
        nbits=plan.kv_bits,
        q_group_size=plan.kv_group_size,
        residual_length=plan.kv_residual_length,
    )


def cache_receipt(cache):
    """Walk Quanto tensor wrappers down to deduplicated physical tensor leaves."""
    import torch

    seen, rows = set(), []

    def walk(value, role):
        flatten = getattr(value, "__tensor_flatten__", None)
        if flatten is not None:
            names, _ = flatten()
            for name in names:
                walk(getattr(value, name), role + "." + name)
        elif isinstance(value, torch.Tensor):
            key = (str(value.device), value.untyped_storage().data_ptr())
            if key not in seen:
                seen.add(key)
                rows.append(
                    {
                        "role": role,
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "bytes": value.untyped_storage().nbytes(),
                    }
                )

    for index, layer in enumerate(getattr(cache, "layers", [])):
        for name in ("keys", "values", "_quantized_keys", "_quantized_values"):
            walk(getattr(layer, name, None), f"layer{index}.{name}")
    return {
        "scope": "resident_cache_storage_leaves_not_attention_workspace",
        "tensors": rows,
        "bytes": sum(x["bytes"] for x in rows),
        "quantized_tensor_leaves": sum("_quantized_" in x["role"] for x in rows),
    }
