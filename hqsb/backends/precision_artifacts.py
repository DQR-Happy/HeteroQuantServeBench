"""Atomic, hash-verified standalone L4 bundles, separate from old RTN artifacts."""

import hashlib
import json
import shutil
import uuid
from pathlib import Path


def ensure_capacity(destination, tensor_bytes, metadata):
    """Bound the compact payload before writing it; source FP16 size is irrelevant.

    Reserve 8 MiB for safetensors headers and small config/tokenizer files. This
    is a conservative export reserve, not an assertion about measured savings.
    """
    destination = Path(destination)
    required = (
        tensor_bytes
        + 8 * 1024**2
        + sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    )
    if required > min(8 * 1024**3, metadata.get("max_artifact_bytes", 8 * 1024**3)):
        raise ValueError("L4_ARTIFACT_STORE_BUDGET_EXCEEDED")
    if shutil.disk_usage(destination.parent).free < tensor_bytes + 512 * 1024**2:
        raise ValueError("L4_INSUFFICIENT_DISK_FOR_COMPACT_ARTIFACT")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest_file = root / "l4-manifest.json"
    if manifest_file.is_symlink() or manifest_file.stat().st_size > 4 * 1024**2:
        raise ValueError("L4_ARTIFACT_MANIFEST_INVALID")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema_version") != "l4-bundle-v1":
        raise ValueError("L4_ARTIFACT_SCHEMA_INVALID")
    if not isinstance(manifest.get("files"), list) or len(manifest["files"]) > 4096:
        raise ValueError("L4_ARTIFACT_FILE_BUDGET")
    expected = {"l4-manifest.json"}
    for item in manifest["files"]:
        file = root / item["path"]
        if (
            file.is_symlink()
            or not file.resolve().is_relative_to(root)
            or file.stat().st_size != item["bytes"]
            or sha256(file) != item["sha256"]
        ):
            raise ValueError("L4_ARTIFACT_INTEGRITY_FAILURE:" + item["path"])
        expected.add(item["path"])
    if {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    } != expected:
        raise ValueError("L4_ARTIFACT_UNDECLARED_FILE")
    return manifest


def publish(partial, destination, metadata):
    partial, destination = Path(partial), Path(destination)
    files = [
        {
            "path": f.relative_to(partial).as_posix(),
            "bytes": f.stat().st_size,
            "sha256": sha256(f),
        }
        for f in sorted(partial.rglob("*"))
        if f.is_file()
    ]
    total = sum(f["bytes"] for f in files)
    if total > min(8 * 1024**3, metadata.get("max_artifact_bytes", 8 * 1024**3)):
        raise ValueError("L4_ARTIFACT_STORE_BUDGET_EXCEEDED")
    manifest = {
        "schema_version": "l4-bundle-v1",
        **metadata,
        "files": files,
        "disk_bytes": total,
    }
    (partial / "l4-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2)
    )
    verify(partial)
    partial.rename(destination)
    return manifest


def export_native(
    model, tokenizer, destination, records, payload, metadata, original_keys=None
):
    """Selected Linear weights are stored ONLY in packed form, rest stays float."""
    from safetensors.torch import save_file
    from hqsb.backends.precision_linear import PackedLinear, Int8Linear, FP8Linear

    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    partial.mkdir(parents=True)
    try:
        selected = {r["name"] for r in records}
        state = model.state_dict()
        rest = {}
        for key, tensor in state.items():
            if original_keys is not None and key not in original_keys:
                continue
            if any(key.startswith(name + ".") for name in selected):
                continue
            identifier = "rest_" + str(len(rest))
            rest[key] = identifier
            payload[identifier] = tensor.detach().cpu().contiguous().clone()
        # Native runtime export may receive replacement modules instead of a
        # calibration model. Their bias is kept in each linear's own record.
        for record in records:
            module = model.get_submodule(record["name"])
            if (
                isinstance(module, (PackedLinear, Int8Linear, FP8Linear))
                and module.bias is not None
            ):
                key = record["name"] + ".bias"
                identifier = "rest_" + str(len(rest))
                rest[key] = identifier
                payload[identifier] = module.bias.detach().cpu().contiguous().clone()
            elif getattr(module, "bias", None) is not None:
                key = record["name"] + ".bias"
                identifier = "rest_" + str(len(rest))
                rest[key] = identifier
                payload[identifier] = module.bias.detach().cpu().contiguous().clone()
        config = model.config.to_dict()
        config.pop("quantization_config", None)
        (partial / "config.json").write_text(json.dumps(config))
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(partial)
        tokenizer.save_pretrained(partial)
        ensure_capacity(
            partial,
            sum(t.numel() * t.element_size() for t in payload.values()),
            metadata,
        )
        save_file(
            {k: v.detach().cpu().contiguous().clone() for k, v in payload.items()},
            str(partial / "payload.safetensors"),
        )
        return publish(
            partial,
            destination,
            {**metadata, "format": "hqsb-native", "linears": records, "rest": rest},
        )
    finally:
        if partial.exists():
            shutil.rmtree(partial)


def load_native(root, dtype, attention):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file
    from hqsb.backends.precision_linear import PackedLinear, Int8Linear

    root = Path(root)
    manifest = verify(root)
    tensors = load_file(str(root / "payload.safetensors"), device="cpu")
    config = AutoConfig.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    # Construct on CPU; only compact buffers and remaining float parameters ever
    # move onto the GPU. This also preserves the model's original RoPE buffers.
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=dtype, attn_implementation=attention
    )
    for record in manifest["linears"]:
        name, k, n = record["name"], record["k"], record["n"]
        bias_key = manifest["rest"].get(name + ".bias")
        bias = tensors[bias_key].to(dtype) if bias_key else None
        if record["execution"] == "cuda_int8_mm":
            module = Int8Linear(
                tensors[record["codes"]], tensors[record["scales"]], bias
            )
        else:
            module = PackedLinear(
                tensors[record["codes"]],
                tensors[record["scales"]],
                tensors[record["zeros"]],
                k,
                record["group_size"],
                record["bits"],
                bias,
            )
        assert module.out_features == n
        model.set_submodule(name, module)
    # load_state_dict copies into each destination's own dtype. Casting every
    # floating tensor to the compute dtype first would lose precision in a
    # persistent FP32 buffer (for example model-specific positional state).
    rest = {k: tensors[v] for k, v in manifest["rest"].items()}
    actual = model.state_dict()
    for key in list(rest):
        if key not in actual:
            raise ValueError("L4_REST_WEIGHT_NOT_IN_MODEL:" + key)
    result = model.load_state_dict(rest, strict=False)
    allowed = {
        r["name"] + "." + k
        for r in manifest["linears"]
        for k in ("packed", "scales", "zeros", "codes")
    }
    if result.unexpected_keys or set(result.missing_keys) - allowed:
        raise ValueError("L4_BUNDLE_WEIGHT_COVERAGE_MISMATCH:" + str(result))
    model.to("cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    return tokenizer, model, manifest
