#!/usr/bin/env python3
"""Build only the new model library and pin its exact artifact. Target-only CLI."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def architectures(value, capability=None):
    """Explicit cross-build targets or the executing device; never assume Orin."""
    if value is None:
        if capability is None:
            raise ValueError("KERNEL_BUILD_REQUIRES_CUDA_DEVICE_OR_EXPLICIT_ARCHITECTURES")
        return [int(capability[0]) * 10 + int(capability[1])]
    if not re.fullmatch(r"[1-9][0-9]{1,2}(;[1-9][0-9]{1,2})*", value):
        raise ValueError("KERNEL_BUILD_ARCHITECTURES_INVALID")
    return sorted(set(int(item) for item in value.split(";")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build/l2")
    parser.add_argument("--cuda-compiler", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--cuda-architectures", help="CMake SM list, e.g. '87;89'; default: executing CUDA device")
    args = parser.parse_args()
    import torch

    targets = architectures(args.cuda_architectures,
                            torch.cuda.get_device_capability(0) if args.cuda_architectures is None and torch.cuda.is_available() else None)
    compiler = subprocess.check_output([args.cuda_compiler, "--version"], text=True).strip()
    release = re.search(r"release ([0-9]+\.[0-9]+)", compiler)
    if not release or release.group(1) != torch.version.cuda:
        raise ValueError("KERNEL_CUDA_TOOLKIT_TORCH_MISMATCH")
    build = (ROOT / args.build_dir).resolve()
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(map(str, targets)), "-DCMAKE_CUDA_COMPILER=" + args.cuda_compiler], check=True)
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
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "ops/cuda/model_kernels").glob("*")) if p.is_file()}
    sources["CMakeLists.txt"] = digest(ROOT / "CMakeLists.txt")
    manifest = {"schema_version": "l2-native-v1", "abi": 1, "library": str(pinned_library),
                "library_sha256": binary_hash, "sources": sources, "sm": targets,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "compiler": compiler,
                "flags": ["Release", *[f"sm_{sm}" for sm in targets], "--fmad=false", "-lineinfo"],
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
