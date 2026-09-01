#!/usr/bin/env python3
"""Build only the new model library and pin its exact artifact. Target-only CLI."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build/l2")
    parser.add_argument("--cuda-compiler", default="/usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    build = (ROOT / args.build_dir).resolve()
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CUDA_ARCHITECTURES=87", "-DCMAKE_CUDA_COMPILER=" + args.cuda_compiler], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "hqsb_model_kernels", "-j2"], check=True)
    library = build / "ops/cuda/model_kernels/libhqsb_model_kernels.so"
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    binary_hash = digest(library)
    artifacts = build / "artifacts"
    artifacts.mkdir(exist_ok=True)
    pinned_library = artifacts / (binary_hash + ".so")
    if not pinned_library.exists():
        shutil.copyfile(library, pinned_library)
    if digest(pinned_library) != binary_hash:
        raise ValueError("CONTENT_ADDRESSED_ARTIFACT_CORRUPTED")
    import torch
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "ops/cuda/model_kernels").glob("*")) if p.is_file()}
    sources["CMakeLists.txt"] = digest(ROOT / "CMakeLists.txt")
    manifest = {"schema_version": "l2-native-v1", "abi": 1, "library": str(pinned_library),
                "library_sha256": binary_hash, "sources": sources, "sm": [87],
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "compiler": subprocess.check_output([args.cuda_compiler, "--version"], text=True).strip(),
                "flags": ["Release", "sm_87", "--fmad=false", "-lineinfo"],
                "binding": "C-ABI-v1-no-libtorch", "workspace_max_bytes": 0}
    destination = build / "model-kernels.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    pinned_manifest = artifacts / (digest(destination) + ".json")
    if not pinned_manifest.exists():
        shutil.copyfile(destination, pinned_manifest)
    if pinned_manifest.read_bytes() != destination.read_bytes():
        raise ValueError("CONTENT_ADDRESSED_MANIFEST_CORRUPTED")
    print(json.dumps({"manifest": str(destination), "library_sha256": manifest["library_sha256"]}))


if __name__ == "__main__":
    main()
