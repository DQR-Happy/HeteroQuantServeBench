#!/usr/bin/env python3
"""Download Qwen3-1.7B from ModelScope into a *manifest-reproducible* snapshot.

Downloads the model from ModelScope using ``snapshot_download`` and then
verifies the result against ``docs/benchmark/model_sha256_manifest.txt``
before reporting success.

Why this script is manifest-driven (S04 补齐)
--------------------------------------------
A plain ``snapshot_download(model_id, local_dir=...)`` does **not** reproduce
the recorded artifact:

* it also fetches repository files the manifest does not declare
  (``.gitattributes``), which fails the ``strict_extra`` gate;
* the ModelScope client writes a local cache index ``.msc``, which is
  documented as client-owned metadata in
  :data:`hqsb.models.manifest.CLIENT_CACHE_METADATA` and is excluded from the
  digest comparison.

Passing the manifest's declared paths as ``allow_patterns`` makes the
download fetch exactly the declared artifact, so the snapshot and the
manifest can no longer drift apart. The download is verified at the end and
the process exits non-zero if the snapshot does not match.

Client version pinning
----------------------
``.mv``/``.msc`` are produced by the ModelScope client, so the recorded
snapshot is tied to a client version. ``1.29.0`` is the version recorded in
``configs/environment/jetson_python_lock.txt`` (the snapshot's provenance);
a different version triggers a warning rather than a hard failure, because
the model content itself is version-independent.

Usage:
    export HQSB_MODEL_ROOT="$HOME/models/hqsb"
    python scripts/models/download_qwen3_modelscope.py [--model-path DIR]
        [--manifest PATH] [--skip-verify]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.models.manifest import (  # noqa: E402
    CLIENT_CACHE_METADATA,
    load_manifest,
    verify_model_files,
)

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_MODEL_ROOT = "~/models/hqsb"
DEFAULT_MANIFEST = "docs/benchmark/model_sha256_manifest.txt"

#: ModelScope client version that produced the recorded snapshot (the value
#: pinned in ``configs/environment/jetson_python_lock.txt``).
EXPECTED_MODELSCOPE_VERSION = "1.29.0"


def _declared_allow_patterns(manifest_path: str) -> list[str]:
    """Declared artifact paths, minus the metadata the client writes itself."""
    entries = load_manifest(manifest_path)
    return [
        entry.normalized_path
        for entry in entries
        if entry.normalized_path not in CLIENT_CACHE_METADATA
    ]


def _check_client_version() -> None:
    try:
        import modelscope

        version = getattr(modelscope, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("cannot determine the ModelScope version: %s", exc)
        return

    if version != EXPECTED_MODELSCOPE_VERSION:
        logger.warning(
            "ModelScope %s is installed but the recorded snapshot was produced "
            "by %s. The model content is version-independent, but the "
            "client-owned '.mv'/'.msc' files may differ; re-check the "
            "verification result below.",
            version,
            EXPECTED_MODELSCOPE_VERSION,
        )
    else:
        logger.info("ModelScope client %s matches the recorded snapshot.", version)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Qwen3-1.7B from ModelScope (manifest-driven)."
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(
            os.path.expanduser(os.environ.get("HQSB_MODEL_ROOT", DEFAULT_MODEL_ROOT)),
            "Qwen3-1.7B",
        ),
        help="Local destination directory (default: $HQSB_MODEL_ROOT/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / DEFAULT_MANIFEST),
        help="SHA256 manifest that defines the artifact to reproduce.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Download only; do not run the artifact verification at the end.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_parser().parse_args()

    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    manifest_path = os.path.abspath(os.path.expanduser(args.manifest))

    allow_patterns = _declared_allow_patterns(manifest_path)

    print(f"ModelScope model : {MODEL_ID}")
    print(f"Local directory  : {model_path}")
    print(f"Manifest         : {manifest_path}")
    print(f"Declared files   : {len(allow_patterns)}")
    print()
    _check_client_version()

    from modelscope import snapshot_download

    logger.info("Starting download from ModelScope...")
    try:
        snapshot_download(
            MODEL_ID,
            local_dir=model_path,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return 1

    print()
    print("Download complete.")

    if args.skip_verify:
        print("(--skip-verify: artifact verification skipped)")
        return 0

    result = verify_model_files(model_path, manifest_path, strict_extra=True)
    print()
    print("=== artifact verification ===")
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
    for relative in result.ignored_client_metadata:
        print(f"  [CLIENT-METADATA-EXCLUDED] {relative}")

    if not result.ok:
        print()
        print(
            "VERIFICATION FAILED - the snapshot does not match the manifest.",
            file=sys.stderr,
        )
        return 2

    print()
    print("VERIFICATION PASSED - snapshot matches the recorded artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
