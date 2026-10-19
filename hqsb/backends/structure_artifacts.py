"""Strict L5 bundles: compact state, original config, explicit module recipe."""

import json
import re
from pathlib import Path
from hqsb.observation.storage import sha256, atomic_json


def resolve(root, identity):
    if not re.fullmatch(r"l5_[0-9a-f]{32}", identity):
        raise ValueError("L5_ARTIFACT_ID_INVALID")
    root = Path(root).resolve()
    result = root / identity
    if result.is_symlink() or not result.resolve().is_relative_to(root):
        raise ValueError("L5_ARTIFACT_PATH")
    return result


def verify(root, *, published=True):
    root = Path(root).resolve()
    file = root / "l5-manifest.json"
    if file.is_symlink() or file.stat().st_size > 4 * 1024**2:
        raise ValueError("L5_MANIFEST_INVALID")
    doc = json.loads(file.read_text())
    if doc.get("schema_version") != "hqsb-l5-v1" or (
        published and doc.get("publication") != "supervisor_acknowledged"
    ):
        raise ValueError("L5_PUBLICATION_INVALID")
    files = doc.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 256:
        raise ValueError("L5_FILES_INVALID")
    names = set()
    for item in files:
        name = item["path"]
        path = root / name
        if (
            name != Path(name).name
            or name in names
            or name == "l5-manifest.json"
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]
        ):
            raise ValueError("L5_ARTIFACT_INTEGRITY")
        names.add(name)
    if {p.name for p in root.iterdir()} != names | {"l5-manifest.json"}:
        raise ValueError("L5_UNDECLARED_ARTIFACT_FILE")
    if (
        not doc.get("source_model_identity", {}).get("hash")
        or not 1 <= len(doc.get("transforms", [])) <= 32
    ):
        raise ValueError("L5_RECIPE_IDENTITY")
    return doc


def export(model, tokenizer, root, metadata):
    from safetensors.torch import save_file

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    keys, payload, aliases = {}, {}, {}
    for name, tensor in model.state_dict().items():
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            str(tensor.dtype),
        )
        if key not in aliases:
            identifier = f"t{len(payload)}"
            payload[identifier] = tensor.detach().cpu().contiguous().clone()
            aliases[key] = identifier
        keys[name] = aliases[key]
    save_file(payload, str(root / "weights.safetensors"))
    model.config.save_pretrained(root)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(root)
    tokenizer.save_pretrained(root)
    files = [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.iterdir())
        if p.is_file()
    ]
    doc = {
        "schema_version": "hqsb-l5-v1",
        **metadata,
        "state_keys": keys,
        "files": files,
        "disk_bytes": sum(p["bytes"] for p in files),
        "publication": "worker_complete_pending_supervisor",
        "format": "strict_state_with_explicit_module_shapes",
    }
    atomic_json(root / "l5-manifest.json", doc)
    verify(root, published=False)
    return doc


def load(root, dtype, attention):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from hqsb.backends.structure_modules import restore_shapes

    root = Path(root)
    doc = verify(root)
    config = AutoConfig.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_config(
        config, dtype=dtype, attn_implementation=attention
    )
    restore_shapes(model, doc["transforms"])
    tensors = load_file(str(root / "weights.safetensors"), device="cpu")
    if set(tensors) != set(doc["state_keys"].values()):
        raise ValueError("L5_PAYLOAD_KEYS_INVALID")
    state = {name: tensors[key] for name, key in doc["state_keys"].items()}
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    return tokenizer, model, doc
