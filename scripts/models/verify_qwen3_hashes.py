#!/usr/bin/env python3
"""Verify the Qwen3-1.7B local snapshot against its SHA256 manifest.

Checks that every file listed in the manifest exists under the model
directory, that its SHA256 digest matches, and that the snapshot contains
no file the manifest does not declare (E00-02). This is the artifact
integrity gate required before a benchmark result may be considered
reproducible.

By default an *undeclared* file fails the gate: an extra weight,
tokenizer, or leftover download-temp file silently changes what a
benchmark measures. Files that are legitimately shipped inside the
snapshot (e.g. a copy of the manifest itself) must be allow-listed
explicitly with ``--allow-extra``, which keeps the exception visible in
the command line instead of hiding it in code.

Exit codes (CLI contract, unchanged):
    0  all manifest files present, digests match, no undeclared files;
    1  operational error (missing model directory/manifest, malformed
       manifest, unsafe manifest path);
    2  verification failure (missing, mismatched, or undeclared files).

The equivalent HQSB process code for the Python gate is
:data:`hqsb.core.errors.ExitCode.ARTIFACT` (8), raised by
:func:`hqsb.models.manifest.verify_or_raise`; the CLI prints it as
``hqsb_exit_code`` on failure for traceability.

Usage:
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python scripts/models/verify_qwen3_hashes.py \\
        --model-path ~/models/hqsb/Qwen3-1.7B \\
        --manifest docs/benchmark/model_sha256_manifest.txt \\
        --allow-extra model_sha256_manifest.txt
"""

from __future__ import annotations

import argparse
import os
import sys

from hqsb.core.errors import ExitCode
from hqsb.models.manifest import ManifestError, verify_model_files

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
    parser.add_argument(
        "--allow-extra",
        action="append",
        default=[],
        metavar="PATH_OR_GLOB",
        help=(
            "Permit an undeclared file (relative path or fnmatch glob). "
            "Repeatable. Use for files intentionally shipped inside the "
            "snapshot, so the exception is visible in the command line."
        ),
    )
    parser.add_argument(
        "--no-strict-extra",
        action="store_true",
        help=(
            "Downgrade undeclared files from a failure to a warning "
            "(legacy pre-E00-02 behavior). Reported files are still listed."
        ),
    )
    return parser


def main() -> int:
    """Run verification and return a process exit code."""
    args = build_parser().parse_args()

    try:
        result = verify_model_files(
            args.model_path,
            args.manifest,
            strict_extra=not args.no_strict_extra,
            allow_extra=tuple(args.allow_extra),
        )
    except ManifestError as e:
        print(f"ERROR: invalid manifest: {e}", file=sys.stderr)
        for issue in e.issues:
            print(
                f"  [{issue.reason}] line {issue.line_number}: "
                f"{issue.path!r} - {issue.message}",
                file=sys.stderr,
            )
        print(f"hqsb_exit_code: {ExitCode.ARTIFACT}", file=sys.stderr)
        return EXIT_OPERATIONAL_ERROR
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "If the model is missing, download it first:\n"
            "  python scripts/models/download_qwen3_modelscope.py",
            file=sys.stderr,
        )
        return EXIT_OPERATIONAL_ERROR

    print(f"Model path   : {result.model_path}")
    print(f"Manifest     : {result.manifest_path}")
    print(f"Manifest sha : {result.manifest_sha256}")
    print(f"Result       : {result.describe()}")
    print(f"Strict extra : {result.strict_extra}")

    for relative in result.missing_files:
        print(f"  [MISSING] {relative}")
    for relative, expected, actual in result.mismatched_files:
        print(f"  [MISMATCH] {relative}")
        print(f"      expected: {expected}")
        print(f"      actual  : {actual}")
    for relative in result.extra_files:
        print(f"  [EXTRA] {relative}")
    for relative in result.allowed_extra_files:
        print(f"  [EXTRA-ALLOWED] {relative}")

    if result.ok:
        print("VERIFICATION PASSED - artifact snapshot is intact.")
        return EXIT_OK

    print(f"First bad file: {result.first_bad_file}")
    print(f"Reason codes  : {', '.join(result.reason_codes) or 'none'}")
    print(
        "VERIFICATION FAILED - do not trust benchmark results from this snapshot.",
        file=sys.stderr,
    )
    print(f"hqsb_exit_code: {ExitCode.ARTIFACT}", file=sys.stderr)
    return EXIT_VERIFICATION_FAILURE


if __name__ == "__main__":
    sys.exit(main())
