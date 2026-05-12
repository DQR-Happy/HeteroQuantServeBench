"""E00-05 shared smoke helpers: one frozen-input Qwen3 greedy run.

This module is the single source of truth for the "run a Qwen3-1.7B
tiny-workload model-core greedy smoke once" procedure used by the
E00-05 driver (``scripts/audit/run_e00_05_qwen_tiny_smoke.py``) and its
cross-process repetition harness.

Everything here is intentionally small: a frozen input is already provided,
so tokenization happens only to *prove tokenizer identity* (re-tokenize the
registered seed text and compare) and never counts towards model-core time.
The generation loop mirrors ``benchmarks/scripts/generate_golden.py``:
prefill -> first-token logits -> argmax -> KV-cache decode for exactly
``output_tokens`` tokens with no early stop on EOS (pre-registered stop
rule: produce exactly OSL tokens; an EOS *inside* the window is recorded,
not used to stop).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    """Lowercase hex SHA256 of raw bytes (identity helper)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    """Lowercase hex SHA256 of a file's content."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def model_family_config(model) -> Dict[str, Any]:
    """Model-level identity fields (architecture, hidden size, dtype, device)."""
    cfg = model.config
    devices = {str(p.device) for p in model.parameters()}
    on_gpu = {d for d in devices if d.startswith("cuda")}
    dtype_counts: Dict[str, int] = {}
    for p in model.parameters():
        dtype_counts[str(p.dtype)] = dtype_counts.get(str(p.dtype), 0) + 1
    return {
        "model_type": str(getattr(cfg, "model_type", "")),
        "architectures": list(getattr(cfg, "architectures", []) or []),
        "hidden_size": int(getattr(cfg, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1)),
        "num_attention_heads": int(getattr(cfg, "num_attention_heads", -1)),
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", -1)),
        "intermediate_size": int(getattr(cfg, "intermediate_size", -1)),
        "vocab_size": int(getattr(cfg, "vocab_size", -1)),
        "max_position_embeddings": int(getattr(cfg, "max_position_embeddings", -1)),
        "param_devices": sorted(devices),
        "fully_on_cuda": bool(devices and devices == on_gpu),
        "param_dtype_counts": dtype_counts,
    }


def tokenizer_identity_files(model_path: str) -> Dict[str, Any]:
    """Tokenizer identity = sha256 of tokenizer.json + tokenizer_config.json.

    Files are declared by ``docs/benchmark/model_sha256_manifest.txt`` and
    are deliberately treated as *files* (raw bytes), not as an executable
    program, so the identity is stable and does not depend on tokenizer
    library version at verification time.
    """
    names = ["tokenizer.json", "tokenizer_config.json"]
    files: Dict[str, str] = {}
    for name in names:
        path = os.path.join(model_path, name)
        files[name] = sha256_file(path)
    composite = sha256_bytes(
        "|".join(f"{k}:{v}" for k, v in files.items()).encode("utf-8")
    )
    return {"files": files, "tokenizer_sha256": composite, "include_special": "false"}


def config_identity(model_path: str) -> str:
    """sha256 of config.json bytes (model-level config identity)."""
    return sha256_file(os.path.join(model_path, "config.json"))


def generation_policy_hash(
    do_sample: bool,
    max_new_tokens: int,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    eos_rule: str,
) -> str:
    payload = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "eos_rule": eos_rule,
        "cache": "past_key_values",
        "attention_mask": "ones",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256_bytes(raw)


def _logits_summary(logits: torch.Tensor, top_k: int = 32) -> Dict[str, Any]:
    """First-token logits summary: top-K ids/logits, L2 norm, full-tensor hash.

    The fp16 logits tensor is hashed bit-exactly (raw bytes) so identical
    models + inputs give an identical hash across processes.
    """
    flat = logits.reshape(-1).float()
    finite = bool(torch.isfinite(flat).all().item())
    topk_val, topk_idx = torch.topk(flat, k=min(top_k, flat.numel()))
    return {
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_device": str(logits.device),
        "finite": finite,
        "nan_count": int(torch.isnan(flat).sum().item()),
        "inf_count": int(torch.isinf(flat).sum().item()),
        "l2_norm": round(float(torch.norm(flat, p=2).item()), 8),
        "logits_sha256": sha256_bytes(flat.detach().cpu().numpy().tobytes()),
        "top_k": top_k,
        "top_token_id": int(topk_idx[0].item()),
        "top_k_token_ids": [int(i) for i in topk_idx.tolist()],
        "top_k_logits": [round(float(v), 8) for v in topk_val.tolist()],
    }


@torch.inference_mode()
def run_greedy_smoke_once(
    model,
    frozen_input_ids: List[int],
    *,
    requested_output_tokens: int,
    top_k: int = 32,
    eos_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one frozen-input prefill + greedy decode; return full evidence.

    Pre-registered stop rule: generate *exactly* ``requested_output_tokens``
    tokens. EOS appearing before the window end is *recorded* (``eos_hit``)
    and decode continues, matching the golden generator. This avoids
    post-hoc interpretation of the output length.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [frozen_input_ids], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    # ── Prefill ────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - t0

    logits = outputs.logits[0, -1, :]  # (vocab,)
    first_logits = _logits_summary(logits, top_k=top_k)

    # ── Greedy decode (exactly requested_output_tokens, no early EOS stop) ─
    next_token = torch.tensor(
        [[first_logits["top_token_id"]]], dtype=torch.long, device=device
    )
    past_key_values = outputs.past_key_values
    generated: List[int] = [first_logits["top_token_id"]]
    current_length = input_ids.shape[1]

    step_seconds: List[float] = []
    for _ in range(1, requested_output_tokens):
        current_length += 1
        decode_mask = torch.ones((1, current_length), dtype=torch.long, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - t0)
        tok = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(tok)
        next_token = torch.tensor([[tok]], dtype=torch.long, device=device)
        past_key_values = outputs.past_key_values

    generated_ids_sha = sha256_bytes(json.dumps(generated).encode("utf-8"))

    # ── Memory peak ────────────────────────────────────────────────────────
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    return {
        "input_ids_used_sha256": sha256_bytes(
            json.dumps(frozen_input_ids).encode("utf-8")
        ),
        "actual_output_tokens": len(generated),
        "requested_output_tokens": requested_output_tokens,
        "output_token_ids": generated,
        "output_token_ids_sha256": generated_ids_sha,
        "output_text": "",
        "eos_token_id": eos_token_id,
        "eos_hit_in_window": bool(
            eos_token_id is not None and eos_token_id in generated
        ),
        "prefill_seconds": round(prefill_seconds, 4),
        "decode_steps": len(step_seconds),
        "mean_decode_seconds": (
            round(sum(step_seconds) / len(step_seconds), 4) if step_seconds else None
        ),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_cuda_allocated_mb": round(peak_allocated / (2**20), 1),
        "peak_cuda_reserved_mb": round(peak_reserved / (2**20), 1),
        "first_logits": first_logits,
    }


def run_once_entry(
    *,
    model_path: str,
    manifest_path: Optional[str],
    allow_extra: Tuple[str, ...],
    frozen_input_ids: List[int],
    requested_output_tokens: int,
    seed_text: str,
    requested_dtype: str,
    requested_backend: str,
    top_k: int = 32,
    no_verify: bool = False,
    run_tag: str = "",
    extra_environment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load Qwen3 via the real loader and execute one smoke; return evidence.

    Raises on any failure so the caller (driver / subprocess) records a
    non-zero exit and the raw stderr.
    """
    load_start = time.perf_counter()

    loader_kwargs: Dict[str, Any] = {}
    if (not no_verify) and manifest_path:
        loader_kwargs = {
            "verify_manifest": manifest_path,
            "strict_extra": True,
            "allow_extra": tuple(allow_extra),
        }

    tokenizer, model, load_time_s = _load_qwen3(model_path, loader_kwargs)

    model_identity = model_family_config(model)
    requested_arch = "auto-from-config"
    actual_arch = model_identity.get("architectures", [])

    # ── Tokenizer identity proof (frozen input must reproduce exactly) ────
    # The frozen input was produced by hqsb/benchmark/workload.py: repeat the
    # seed-token sequence until >= ISL, then truncate to exactly ISL. Rebuild
    # that sequence today with the *same* rules (frozen token IDs never drive
    # the rebuild) and require an exact match. This proves the tokenizer that
    # produced the frozen input is the tokenizer loaded right now.
    seed_tokens = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    reconstructed: List[int] = []
    while len(reconstructed) < len(frozen_input_ids):
        reconstructed.extend(seed_tokens)
    reconstructed = reconstructed[: len(frozen_input_ids)]
    frozen_ok = bool(reconstructed == frozen_input_ids)
    if not frozen_ok:
        # Tokenizer that produced the frozen IDs would *not* reproduce them
        # today. Record loudly; the generation below still uses the frozen
        # IDs so the model-core claim stays independent.
        logger.warning(
            "Frozen-input/tokenizer mismatch: rebuilt head=%s vs frozen head=%s",
            reconstructed[:8],
            frozen_input_ids[:8],
        )

    tok_files = tokenizer_identity_files(model_path)
    config_hash = config_identity(model_path)

    eos_id = int(getattr(tokenizer, "eos_token_id", -1) or -1)
    do_sample = False
    policy_hash = generation_policy_hash(
        do_sample=False,
        max_new_tokens=requested_output_tokens,
        top_k=None,
        top_p=None,
        temperature=None,
        eos_rule="no_early_stop_exact_osl",
    )

    rec = run_greedy_smoke_once(
        model,
        frozen_input_ids,
        requested_output_tokens=requested_output_tokens,
        top_k=top_k,
        eos_token_id=eos_id,
    )

    requested_dtype_actual = str(dict(model.named_parameters()).popitem()[1].dtype)
    # Actual dtype/device of the causal LM head logits after prefill is not
    # stored on the model; use first parameter dtype + module device for the
    # record.

    load_seconds = round(time.perf_counter() - load_start, 3)

    return {
        "model_identity": model_identity,
        "actual_model_load_seconds": load_seconds,
        "load_time_seconds": round(float(load_time_s), 3),
        "requested": {
            "device": "cuda",
            "dtype": requested_dtype,
            "attention_backend": requested_backend,
            "manifest_verification": (not no_verify) and bool(manifest_path),
        },
        "config_hash": config_hash,
        "tokenizer_identity": tok_files,
        "frozen_input_reproducible_by_tokenizer": frozen_ok,
        "requested_architecture": requested_arch,
        "actual_architecture": actual_arch,
        "eos_token_id": eos_id,
        "generation_policy_hash": policy_hash,
        "result": rec,
        "run_tag": run_tag,
        "extra_environment": extra_environment or {},
    }


def _load_qwen3(model_path: str, loader_kwargs: Dict[str, Any]):
    """Thin wrapper so the real loader is used unmodified."""
    from hqsb.models.loader import load_qwen3

    return load_qwen3(
        model_path,
        dtype=torch.float16,
        attention_backend="eager",
        **loader_kwargs,
    )
