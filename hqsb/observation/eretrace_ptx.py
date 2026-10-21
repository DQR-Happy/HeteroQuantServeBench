"""Locate retained Triton PTX by exact CUBIN identity, rather than kernel name."""
import hashlib
import json
import os
from pathlib import Path


def cached_ptx(binary_path):
    path = Path(binary_path)
    cache = os.environ.get("HQSB_ERE_TRITON_CACHE")
    if not cache or path.stat().st_size > 16 * 1024**2:
        return None
    image = path.read_bytes()
    if not image.startswith(b"\x7fELF"):
        return None
    identity = hashlib.sha256(image).hexdigest()
    for index, cubin in enumerate(Path(cache).glob("*/*.cubin")):
        if index >= 4096:
            raise ValueError("ERE_TRITON_CACHE_SCAN_LIMIT")
        ptx = cubin.with_suffix(".ptx")
        if not ptx.is_file() or cubin.stat().st_size != len(image) or ptx.stat().st_size > 16 * 1024**2:
            continue
        if hashlib.sha256(cubin.read_bytes()).hexdigest() == identity:
            evidence = {"source": "triton_cache_matching_cubin", "cubin_sha256": identity,
                        "ptx_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest(), "ptx_path": str(ptx)}
            (path.parent / "ptx-origin.json").write_text(json.dumps(evidence) + "\n")
            return ptx
    return None
