"""Isolated CPU transform process. Never modifies a live model or source files."""

import json
import os
from pathlib import Path
import sys
import time


def execute(payload):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hqsb.models.manifest import verify_or_raise
    from hqsb.core.structure_plan import StructurePlan
    from hqsb.backends.structure_transforms import select, activation_stats, apply
    from hqsb.backends.structure_artifacts import export
    from hqsb.observation.storage import atomic_json, sha256

    torch.set_num_threads(2)
    plan = StructurePlan.model_validate(payload["plan"])
    config = payload["config"]
    source = Path(os.path.expandvars(config["model_path"])).expanduser()
    verify_or_raise(
        str(source),
        config["manifest"],
        strict_extra=True,
        allow_extra=tuple(config.get("manifest_allow_extra", [])),
    )
    source_identity = {
        "hash": sha256(Path(config["manifest"]).expanduser()),
        "scope": "verified_manifest_at_load",
        "revision": "local_manifest_verified",
    }
    origin = time.monotonic()
    progress_file = Path(payload["progress"])

    def check():
        if time.monotonic() > payload["deadline"]:
            raise TimeoutError("L5_TRANSFORM_DEADLINE")

    def progress(done, total, name):
        atomic_json(
            progress_file,
            {
                "kind": "progress",
                "phase": "transform",
                "completed": done,
                "total": total,
                "module": name,
            },
        )

    atomic_json(
        progress_file, {"kind": "progress", "phase": "cpu_source_load", "completed": 0}
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            dtype=getattr(torch, payload["precision"]),
            device_map="cpu",
            attn_implementation=config["attention"],
        )
        .eval()
        .float()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    check()
    names = select(model, plan)
    params_before = sum(p.numel() for p in model.parameters())
    stats, hashes = (
        activation_stats(model, tokenizer, names, plan, check, config["context_limit"])
        if plan.method == "wanda"
        else ({}, [])
    )
    torch.manual_seed(plan.seed)
    records = apply(model, names, plan, check, progress, stats)
    check()
    dtype = getattr(torch, payload["precision"])
    model.to(dtype=dtype)
    params_after = sum(p.numel() for p in model.parameters())
    from hqsb.observation.identity import environment

    doc = export(
        model,
        tokenizer,
        payload["stage"],
        {
            "artifact_id": payload["artifact_id"],
            "source_model_identity": source_identity,
            "source_digest": environment(Path.cwd())["source_digest"],
            "recipe": plan.model_dump(exclude={"calibration"}),
            "calibration_text_sha256": hashes,
            "calibration_count": len(plan.calibration),
            "transforms": records,
            "creation_seconds": time.monotonic() - origin,
            "creation_device": "cpu",
            "statistics": {
                "parameters_before": params_before,
                "parameters_after": params_after,
                "parameter_reduction": 1 - params_after / params_before,
                "same_dtype_parameter_bytes_before": params_before * dtype.itemsize,
                "same_dtype_parameter_bytes_after": params_after * dtype.itemsize,
                "export_dtype": payload["precision"],
                "sparse_kernel": False,
            },
            "packages": {"torch": str(torch.__version__)},
            "lineage": "one_transform_from_verified_original_no_training",
            "limitations": [
                "Dense masked weights do not imply storage savings or sparse acceleration.",
                "No distillation, recovery training, attention/head/expert conversion was performed.",
                "Transform Frobenius diagnostics are computed in FP32 before export; model quality is measured independently.",
            ],
        },
    )
    doc["creation_seconds"] = time.monotonic() - origin
    atomic_json(Path(payload["stage"]) / "l5-manifest.json", doc)
    atomic_json(
        progress_file,
        {
            "kind": "progress",
            "phase": "worker_complete",
            "disk_bytes": doc["disk_bytes"],
        },
    )


def main():
    if sys.platform == "linux":
        import ctypes, signal

        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() == 1:
            raise SystemExit(1)
    payload = json.loads(Path(sys.argv[1]).read_text())
    execute(payload)


if __name__ == "__main__":
    main()
