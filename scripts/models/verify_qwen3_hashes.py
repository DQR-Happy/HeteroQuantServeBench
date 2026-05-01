#!/usr/bin/env python3
"""Verify the Qwen3-1.7B local snapshot against its SHA256 manifest.

Checks every file listed in the manifest exists under the model directory
and that its SHA256 digest matches. This is the artifact integrity gate
required before a benchmark result may be considered reproducible.

Exit codes:
    0  all manifest files present and digests match;
    1  operational error (missing model directory or manifest file);
    2  verification failure (missing or mismatched model files).

Usage:
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python scripts/models/verify_qwen3_hashes.py \\
        --model-path ~/models/hqsb/Qwen3-1.7B \\
        --manifest docs/benchmark/model_sha256_manifest.txt
"""

from __future__ import annotations

import argparse
import os
import sys

from hqsb.models.manifest import verify_model_files

EXIT_OK = 0
EXIT_OPERATIONAL_ERROR = 1
EXIT_VERIFICATION_FAILURE = 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the hash verification CLI."""
    parser = argparse.ArgumentParser(
        description="Verify a local model snapshot against a SHA256 manifest.",
    )
    parser.add_argument(
        "--model-path",
        default="~/models/hqsb/Qwen3-1.7B",
        help="Local model directory (default: ~/models/hqsb/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--manifest",
        default="docs/benchmark/model_sha256_manifest.txt",
        help="SHA256 manifest path (default: docs/benchmark/model_sha256_manifest.txt)",
    )
    return parser


def main() -> int:
    """Run verification and return a process exit code."""
    args = build_parser().parse_args()

    try:
        result = verify_model_files(args.model_path, args.manifest)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "If the model is missing, download it first:\n"
            "  python scripts/models/download_qwen3_modelscope.py",
            file=sys.stderr,
        )
        return EXIT_OPERATIONAL_ERROR
    except ValueError as e:
        print(f"ERROR: invalid manifest: {e}", file=sys.stderr)
        return EXIT_OPERATIONAL_ERROR

    print(f"Model path : {result.model_path}")
    print(f"Manifest   : {result.manifest_path}")
    print(f"Result     : {result.describe()}")

    for relative in result.missing_files:
        print(f"  [MISSING] {relative}")
    for relative, expected, actual in result.mismatched_files:
        print(f"  [MISMATCH] {relative}")
        print(f"      expected: {expected}")
        print(f"      actual  : {actual}")

    if result.ok:
        print("VERIFICATION PASSED - artifact snapshot is intact.")
        return EXIT_OK

    print("VERIFICATION FAILED - do not trust benchmark results from this snapshot.",
          file=sys.stderr)
    return EXIT_VERIFICATION_FAILURE


if __name__ == "__main__":
    sys.exit(main())
