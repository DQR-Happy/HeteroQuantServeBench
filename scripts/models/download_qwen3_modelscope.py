#!/usr/bin/env python3
"""Download Qwen3-1.7B from ModelScope to a local directory.

Downloads the complete model (weights, tokenizer, config) from
ModelScope using ``snapshot_download`` and stores it under
``$HQSB_MODEL_ROOT/Qwen3-1.7B`` (default: ``~/models/hqsb/Qwen3-1.7B``).

All subsequent benchmarks load the model exclusively from this
local directory with ``local_files_only=True``, ensuring offline
reproducibility and artifact integrity.

Usage:
    export HQSB_MODEL_ROOT="$HOME/models/hqsb"
    python scripts/models/download_qwen3_modelscope.py
"""

from __future__ import annotations

import logging
import os
import sys

from modelscope import snapshot_download

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-1.7B"

DEFAULT_MODEL_ROOT = "~/models/hqsb"


def main() -> None:
    """Download the model and report progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_root = os.path.expanduser(
        os.environ.get("HQSB_MODEL_ROOT", DEFAULT_MODEL_ROOT)
    )
    local_dir = os.path.join(model_root, "Qwen3-1.7B")

    print(f"ModelScope model : {MODEL_ID}")
    print(f"Model root       : {model_root}")
    print(f"Local directory  : {local_dir}")
    print()

    os.makedirs(model_root, exist_ok=True)

    logger.info("Starting download from ModelScope...")
    logger.info("This may take several minutes depending on network speed.")

    try:
        model_dir = snapshot_download(
            MODEL_ID,
            local_dir=local_dir,
        )
    except Exception as e:
        logger.error("Download failed: %s", e)
        sys.exit(1)

    print()
    print("Download complete.")
    print(f"Model directory: {model_dir}")

    # List downloaded files
    if os.path.isdir(model_dir):
        files = sorted(
            f for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f))
        )
        print(f"\nFiles ({len(files)}):")
        for fname in files:
            fpath = os.path.join(model_dir, fname)
            size_mb = os.path.getsize(fpath) / (1024**2)
            print(f"  {fname:<40s} {size_mb:>10.1f} MB")

        # Total size
        total_bytes = sum(
            os.path.getsize(os.path.join(model_dir, f))
            for f in files
        )
        print(f"\nTotal: {total_bytes / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
