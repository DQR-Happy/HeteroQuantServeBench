#!/usr/bin/env python3
"""Download curated models exclusively from ModelScope; seal publisher hashes.

Run on an authorized Linux target through scripts/remote_run.sh. Metadata is
stored outside model directories so strict artifact verification stays enabled.
The SDK resolves a branch, not an immutable snapshot commit: the pre-download
file revisions/hashes are recorded and verified; a moving branch cannot silently
change accepted weights. Offline runtime never contacts either model hub.
"""
from __future__ import annotations

import argparse
import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hqsb.models.catalog import MODELS, runtime_files
from hqsb.models.manifest import CLIENT_CACHE_METADATA, verify_or_raise


def download(key, root, *, reserve_bytes):
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    model_id, label, model_type = MODELS[key]
    directory = root / key
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = manifests / (key + ".sha256")
    provenance_path = manifests / (key + ".json")
    # Existing sealed snapshots are reused without silently updating master.
    if manifest.exists() and provenance_path.exists():
        verify_or_raise(directory, manifest, allow_extra=CLIENT_CACHE_METADATA)
        print(f"REUSED {key}: verified sealed snapshot", flush=True)
        return
    api = HubApi()
    revision = api.get_valid_revision_detail(model_id, revision="master")
    entries = runtime_files(api.get_model_files(model_id, revision="master", recursive=True))
    required = sum(item["Size"] for item in entries)
    # Allow resumed partial files, but conservatively budget a complete download.
    if shutil.disk_usage(root).free < required + reserve_bytes:
        raise RuntimeError(f"MODEL_DISK_BUDGET:{key}:need {required + reserve_bytes} free bytes")
    text = "".join(f"{item['Sha256'].lower()}  {item['Path']}\n" for item in entries)
    pending = manifests / (key + ".pending.sha256")
    pending.write_text(text)
    print(f"DOWNLOADING {key}: {required} bytes from {model_id}", flush=True)
    snapshot_download(model_id, revision="master", local_dir=str(directory),
                      allow_patterns=[item["Path"] for item in entries], max_workers=2)
    verify_or_raise(directory, pending, allow_extra=CLIENT_CACHE_METADATA)
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != model_type:
        raise ValueError("MODEL_ARCHITECTURE_CHANGED:" + key)
    if config.get("auto_map"):
        raise ValueError("REMOTE_MODEL_CODE_NOT_SUPPORTED:" + key)
    provenance = {"schema_version": 1, "source": "modelscope", "model_id": model_id,
                  "label": label, "model_type": model_type, "branch": revision,
                  "revision_scope": "publisher_per_file_revisions_and_sha256_not_snapshot_commit",
                  "excluded_client_metadata": list(CLIENT_CACHE_METADATA),
                  "client": {"modelscope_version": version("modelscope"), "file_workers": 2,
                             "download_parallels": int(os.environ.get("MODELSCOPE_DOWNLOAD_PARALLELS", "1"))},
                  "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "files": [{field: item.get(field) for field in ("Path", "Size", "Sha256", "Revision")}
                            for item in entries]}
    os.replace(pending, manifest)
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(f"VERIFIED {key}: {manifest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=sorted(MODELS))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=int, default=8)
    parser.add_argument("--download-parallels", type=int, choices=range(1, 17), default=16)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run on an authorized Linux target")
    if args.reserve_gib < 1:
        parser.error("reserve-gib must be positive")
    # ModelScope 1.29 reads this at import. Use its verified range downloader,
    # bounded to two files at once; no separate download stack or hub fallback.
    os.environ["MODELSCOPE_DOWNLOAD_PARALLELS"] = str(args.download_parallels)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    locks = root / ".locks"
    locks.mkdir(exist_ok=True)
    for key in args.models:
        with (locks / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            download(key, root, reserve_bytes=args.reserve_gib * 1024**3)


if __name__ == "__main__":
    main()
