"""Model loader for HQSB benchmarks.

Provides a unified interface for loading Qwen3-family models from
ModelScope local directories, with support for dtype configuration,
attention backend selection, and timing instrumentation.

All loading is performed with ``local_files_only=True`` to ensure
offline reproducibility and artifact integrity.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Tuple

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

from hqsb.models.manifest import verify_model_files

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


def load_qwen3(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    attention_backend: str = "eager",
    device_map: dict | str | None = None,
    offload_folder: str | None = None,
    verify_manifest: str | None = None,
) -> Tuple:
    """Load a Qwen3-family model and tokenizer from a local ModelScope directory.

    This is the primary entry point for loading models in HQSB benchmarks.
    All model files are loaded from the local filesystem only
    (``local_files_only=True``), ensuring deterministic and offline-safe
    benchmark runs.

    On memory-constrained Jetson platforms (8 GB unified memory), the
    loader uses ``device_map="auto"`` with CPU offloading and
    ``low_cpu_mem_usage=True`` to minimize peak memory. After loading,
    the model is moved entirely to GPU.

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
            ``model_path`` before loading. A missing or mismatched file
            raises :class:`RuntimeError` with a diagnostic summary. When
            ``None``, no digest verification is performed (fast path).

    Returns:
        A tuple of ``(tokenizer, model, load_time_s)`` where:
            - ``tokenizer``: The loaded tokenizer instance.
            - ``model``: The loaded model instance (in eval mode, on GPU).
            - ``load_time_s``: Wall-clock time spent loading in seconds.

    Raises:
        FileNotFoundError: If the model directory does not exist or required
            files are missing.
        RuntimeError: If model loading fails (e.g., insufficient memory).
    """
    model_path = _resolve_path(model_path)
    _validate_model_directory(model_path)

    # Optional artifact integrity gate: verify the model snapshot against
    # a SHA256 manifest before loading any weights. This turns a corrupted
    # or partially-downloaded model into a diagnostic failure early, rather
    # than a silent numerical regression later.
    if verify_manifest is not None:
        verification = verify_model_files(model_path, verify_manifest)
        if not verification.ok:
            details: list[str] = []
            if verification.missing_files:
                details.append(
                    "missing files: "
                    + ", ".join(verification.missing_files)
                )
            if verification.mismatched_files:
                details.append(
                    "mismatched files: "
                    + ", ".join(path for path, _, _ in verification.mismatched_files)
                )
            raise RuntimeError(
                f"Model artifact verification failed "
                f"({verification.describe()}): " + "; ".join(details)
            )
        logger.info(
            "Artifact verification passed: %s",
            verification.describe(),
        )

    # On Jetson, use "auto" device_map with offloading to survive
    # the 8 GB unified memory constraint, then move fully to GPU.
    if device_map is None:
        if torch.cuda.is_available():
            # Try direct GPU load first; fall back to auto if OOM
            device_map = {"": 0}
        else:
            device_map = "cpu"

    if offload_folder is None:
        offload_folder = "/tmp/hqsb_offload"
    os.makedirs(offload_folder, exist_ok=True)

    logger.info("Loading model from: %s", model_path)
    logger.info("  dtype: %s", dtype)
    logger.info("  attention_backend: %s", attention_backend)
    logger.info("  device_map: %s", device_map)

    start = time.perf_counter()

    # Load tokenizer first (lightweight, validates directory structure)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    # Load model with memory-efficient strategy.
    # On Jetson (8 GB unified memory), the 3.4 GB model + KV cache
    # can exceed available RAM. We use offload_folder to spill to disk
    # during loading, then move fully to GPU.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=attention_backend,
            offload_folder=offload_folder,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() and device_map != "auto":
            logger.warning(
                "Direct GPU load OOM, retrying with device_map='auto' + offloading"
            )
            # Retry with auto device map (may split across CPU/GPU)
            device_map = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation=attention_backend,
                offload_folder=offload_folder,
                max_memory={0: "6GB", "cpu": "8GB"},
            )
        else:
            raise

    model.eval()

    # Ensure all CUDA operations from loading are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_time_s = time.perf_counter() - start

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded in %.2f s (%d parameters, %.2f GB FP16)",
        load_time_s,
        param_count,
        param_count * dtype.itemsize / (1024**3),
    )

    # Report memory usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        logger.info(
            "CUDA memory: allocated=%.2f GB, reserved=%.2f GB",
            allocated_gb,
            reserved_gb,
        )

    return tokenizer, model, load_time_s
