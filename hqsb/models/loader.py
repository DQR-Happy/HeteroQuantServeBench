"""Model loader for HQSB benchmarks.

Provides a unified interface for loading causal language models from
ModelScope local directories, with support for dtype configuration,
attention backend selection, and timing instrumentation.

All loading is performed with ``local_files_only=True`` to ensure
offline reproducibility and artifact integrity.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hqsb.benchmark.memory import (
    gpu_memory_budget_bytes,
    host_memory_budget_bytes,
    memory_budget_bytes,
    model_weight_bytes,
)
from hqsb.hardware import device as device_api
from hqsb.models.manifest import verify_or_raise

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> str:
    """Resolve a filesystem path, expanding ``~`` and environment variables.

    Args:
        path: Raw path string (may contain ``~`` or ``$VAR``).

    Returns:
        Absolute path with all expansions applied.
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _validate_model_directory(model_path: str) -> None:
    """Validate that a model directory exists and contains expected files.

    Args:
        model_path: Absolute path to the model directory.

    Raises:
        FileNotFoundError: If the directory or critical files are missing.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please download the model first using:\n"
            f"  python scripts/models/download_qwen3_modelscope.py"
        )

    required_files = ["config.json", "tokenizer.json"]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(model_path, f))]
    if missing:
        raise FileNotFoundError(
            f"Required model files missing in {model_path}: {missing}"
        )


def _consolidate_to_device(model: Any, *, target: Any, force: bool = False) -> bool:
    """Move every parameter of ``model`` onto ``target``, but only if it provably fits.

    ``device_map="auto"`` may leave part of the model on CPU or on disk, which
    silently degrades inference: every forward pass pays host<->device copies
    for the offloaded layers, while saving no device memory in practice.

    The migration is *pre-checked* against the derived budget instead of being
    attempted and rolled back: ``Module.to()`` is not exception-safe, so a
    mid-migration OOM would leave some parameters on CUDA and others on the
    host — an inconsistent, silently-wrong model. Skipping up-front keeps the
    model in the coherent ``device_map="auto"`` state.

    Args:
        model: The model to move.
        force: Skip the up-front budget check and always attempt the move. This
            is correct for the host-staging path (``load_qwen3(cpu_staging=True)``),
            where the weights are ordinary swappable CPU tensors: ``Module.to()``
            then migrates tensor-by-tensor, so the transient peak is one tensor,
            not the whole weight set. The budget check would otherwise be
            misled by the free memory already consumed by the CPU copy.

    Returns:
        True if the model is fully on CUDA afterwards, False otherwise.
    """
    devices = {
        str(param.device)
        for param in list(model.parameters()) + list(model.buffers())
    }
    target_kind = device_api.accelerator_kind(target)
    if devices and all(device_api.accelerator_kind(item) == target_kind for item in devices):
        return True

    weight_bytes = model_weight_bytes(model)
    memory_info = device_api.mem_get_info(target)
    budget_bytes: Optional[int] = None
    if memory_info is not None:
        free_bytes, _total = memory_info
        budget_bytes = memory_budget_bytes(free_bytes)

    if not force and budget_bytes is not None and weight_bytes > budget_bytes:
        logger.warning(
            "Weights (%.2f GiB) exceed the safe device budget (%.2f GiB, from "
            "%.2f GiB free); keeping the device_map='auto' split. "
            "Free system memory to load the model entirely on the device.",
            weight_bytes / (1024**3),
            budget_bytes / (1024**3),
            free_bytes / (1024**3),
        )
        return False

    try:
        model.to(target)
    except (device_api.out_of_memory_error(target), RuntimeError) as exc:
        # Only reachable if the estimate was off (e.g. allocator
        # fragmentation). The model stays split; callers must not assume it
        # is fully on the device -- check the return value.
        logger.warning(
            "Device consolidation failed despite a sufficient budget (%s); "
            "keeping the device_map='auto' split",
            exc,
        )
        device_api.empty_cache(target)
        return False

    logger.info(
        "Consolidated %.2f GiB of weights onto %s (budget %s)",
        weight_bytes / (1024**3),
        target,
        f"{budget_bytes / (1024**3):.2f} GiB" if budget_bytes is not None else "unknown",
    )
    if force:
        # Host staging frees the CPU weight copies inside ``Model.to``; return
        # those pages to the OS so inference has real headroom (glibc otherwise
        # keeps them in its arena, which shows up as RSS on a unified-memory
        # device and can trigger the OOM killer during long prefills).
        try:
            import ctypes
            import gc

            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
    return True


def load_causal_lm(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    attention_backend: str = "eager",
    device_map: dict | str | None = None,
    offload_folder: str | None = None,
    verify_manifest: str | None = None,
    strict_extra: bool = True,
    allow_extra: tuple = (),
    cpu_staging: bool = False,
    device: str | None = None,
) -> Tuple:
    """Load a built-in Transformers causal model from a local ModelScope snapshot.

    This is the primary entry point for loading models in HQSB benchmarks.
    All model files are loaded from the local filesystem only
    (``local_files_only=True``), ensuring deterministic and offline-safe
    benchmark runs.

    On memory-constrained Jetson platforms (8 GB unified memory), the
    loader uses ``device_map="auto"`` with CPU offloading and
    ``low_cpu_mem_usage=True`` to minimize peak memory. The GPU/CPU
    budgets are derived from *measured* free memory rather than from
    hardcoded constants, because on unified-memory devices only a
    fraction of the reported device total is actually usable. After
    loading, the model is consolidated onto the GPU when the whole
    weight set fits; otherwise the auto split is kept and a warning is
    logged.

    Args:
        model_path: Path to the local model directory. ``~`` and environment
            variables are expanded automatically.
        dtype: PyTorch data type for model weights. Default: ``torch.float16``.
        attention_backend: Attention implementation to use. One of:
            ``"eager"``, ``"sdpa"``, ``"flash_attention_2"``.
            Default: ``"eager"``.
        device_map: Device mapping strategy. If ``None``, defaults to
            ``"auto"`` for memory-constrained platforms, then moves
            all layers to GPU after loading.
        offload_folder: Directory for CPU offloading of weights during
            loading. If ``None``, uses a temp directory under
            ``/tmp/hqsb_offload``.
        verify_manifest: Optional path to a SHA256 manifest. When provided,
            every file listed in the manifest is verified against
            ``model_path`` before loading. Any integrity fault (missing,
            mismatched, or undeclared file; unsafe manifest path) raises
            :class:`~hqsb.core.errors.ArtifactError` with a diagnostic
            ``details`` mapping. When ``None``, no digest verification is
            performed (fast path).
        strict_extra: When true (default), a file present under
            ``model_path`` but not declared by the manifest fails the gate.
        allow_extra: Relative paths or ``fnmatch`` globs permitted to exist
            without being declared by the manifest.
        device: Accelerator to load onto (``"cuda"`` / ``"npu"`` / ``"cpu"``).
            ``None`` honours ``HQSB_DEVICE`` and otherwise prefers CUDA, then
            Ascend NPU, then CPU -- see :mod:`hqsb.hardware.device`.

    Returns:
        A tuple of ``(tokenizer, model, load_time_s)`` where:
            - ``tokenizer``: The loaded tokenizer instance.
            - ``model``: The loaded model instance (in eval mode, on GPU).
            - ``load_time_s``: Wall-clock time spent loading in seconds.

    Raises:
        FileNotFoundError: If the model directory does not exist or required
            files are missing.
        ArtifactError: If the snapshot fails the manifest integrity gate
            (exit code 8).
        RuntimeError: If model loading fails (e.g., insufficient memory).
    """
    model_path = _resolve_path(model_path)
    _validate_model_directory(model_path)

    # Optional artifact integrity gate: verify the model snapshot against
    # a SHA256 manifest before loading any weights. This turns a corrupted
    # or partially-downloaded model into a diagnostic failure early, rather
    # than a silent numerical regression later. verify_or_raise collapses
    # every fault class into one ArtifactError, so no bad snapshot reaches
    # the weight loader.
    if verify_manifest is not None:
        verification = verify_or_raise(
            model_path,
            verify_manifest,
            strict_extra=strict_extra,
            allow_extra=allow_extra,
        )
        logger.info(
            "Artifact verification passed: %s",
            verification.describe(),
        )

    # On unified-memory devices (Jetson) the CUDA "total" is the whole system
    # DRAM, and only a fraction of it is actually free for weights. Budgets are
    # therefore derived from measured free memory instead of hardcoded values,
    # which otherwise overshoot the real headroom (e.g. a literal "6GB" on a
    # 7.6 GiB device whose free memory is ~4 GiB).
    target = device_api.resolve_device(device or "")
    target_kind = device_api.accelerator_kind(target)
    target_is_accelerator = device_api.is_accelerator(target)

    # Ascend has no accelerate device_map integration here, so it always stages
    # through host memory and is consolidated afterwards -- the same path the
    # Jetson CPU-staging mode uses, and the reason the two accelerate-based
    # branches below are CUDA-only.
    if target_kind == device_api.NPU:
        cpu_staging = True

    gpu_budget_bytes = gpu_memory_budget_bytes() if target_is_accelerator else 0
    cpu_budget_bytes = host_memory_budget_bytes()

    if device_map is None:
        if cpu_staging and target_is_accelerator:
            # Host staging: load all weights as ordinary (swappable) CPU
            # tensors, then let ``_consolidate_to_device`` move them to the
            # accelerator. ``device_map="auto"`` stages through pinned host
            # memory, whose peak (~2x the weight bytes, non-swappable) can OOM
            # an 8 GiB unified-memory device even though the final 3.2 GiB fits.
            device_map = "cpu"
        else:
            device_map = "auto" if target_is_accelerator else "cpu"

    if offload_folder is None:
        offload_folder = "/tmp/hqsb_offload"
    os.makedirs(offload_folder, exist_ok=True)

    # ``max_memory`` is honored only for the string device_map strategies;
    # it is harmless (ignored) when ``device_map`` is an explicit dict.
    max_memory = (
        {0: int(gpu_budget_bytes), "cpu": int(cpu_budget_bytes)}
        if device_map == "auto"
        else None
    )

    logger.info("Loading model from: %s", model_path)
    logger.info("  dtype: %s", dtype)
    logger.info("  attention_backend: %s", attention_backend)
    logger.info("  device_map: %s", device_map)

    start = time.perf_counter()

    # Load tokenizer first (lightweight, validates directory structure)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )

    # Load model with memory-efficient strategy.
    # ``device_map="auto"`` dispatches layers across GPU/CPU/disk according to
    # the measured budget, so loading never OOMs. We then *consolidate* the
    # model onto the GPU when the whole weight set fits (see below), because a
    # split map silently degrades inference with per-layer host<->device
    # copies. ``model.eval()`` alone does NOT move anything.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=attention_backend,
            offload_folder=offload_folder,
            max_memory=max_memory,
        )
    except (device_api.out_of_memory_error(target), RuntimeError) as e:
        if "out of memory" in str(e).lower() and device_map != "auto":
            logger.warning(
                "Direct GPU load OOM, retrying with device_map='auto' + offloading"
            )
            device_map = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation=attention_backend,
                offload_folder=offload_folder,
                max_memory={
                    0: int(gpu_budget_bytes),
                    "cpu": int(cpu_budget_bytes),
                },
            )
        else:
            raise

    model.eval()

    # Consolidate onto the GPU when the full weight set fits. Without this the
    # model keeps whatever split ``device_map="auto"`` chose, and every forward
    # pass pays host<->device copies for the offloaded layers — slower than a
    # pure-GPU model while saving no device memory in practice.
    if target_is_accelerator:
        _consolidate_to_device(model, target=target, force=cpu_staging)
        # Ensure every queued load/copy is complete before the caller times it.
        device_api.synchronize(target)

    load_time_s = time.perf_counter() - start

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded in %.2f s (%d parameters, %.2f GB FP16)",
        load_time_s,
        param_count,
        param_count * dtype.itemsize / (1024**3),
    )

    # Report memory usage. `reserved` is CUDA-specific; on Ascend it is reported
    # as UNAVAILABLE instead of being filled with the allocated number.
    if target_is_accelerator:
        allocated = device_api.memory_allocated(target)
        reserved = device_api.max_memory_reserved(target)
        logger.info(
            "%s memory: allocated=%s, reserved=%s",
            target_kind.upper(),
            f"{allocated / (1024**3):.2f} GB" if allocated is not None else "UNAVAILABLE",
            f"{reserved / (1024**3):.2f} GB" if reserved is not None else "UNAVAILABLE",
        )

    return tokenizer, model, load_time_s


# Frozen benchmark callers retain their public entry point and semantics.
load_qwen3 = load_causal_lm
