"""Build an isolated ERETrace runtime from an explicitly supplied private source.

Run on the CUDA target through remote_run.sh. The BITS checkout stays read-only;
HQSB adapter changes apply to the installed copy and enter its build receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"ERE_SOURCE_LAYOUT_CHANGED:{path.name}:{old[:60]}")
    path.write_text(text.replace(old, new))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda"))
    p.add_argument("--arch", choices=["86", "87", "90"], default="86")
    a = p.parse_args()
    source, dest = a.source.resolve(strict=True), a.dest.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        p.error("Source and runtime directories need independent locations")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        p.error("Choose an empty runtime directory; existing builds retain their identity")
    required = ["neutrino", "benchmarks/capture_pack.cu", "benchmarks/capture_compact.cuh", "scripts/build.py", "LICENSE", "THIRD_PARTY_NOTICES.md"]
    origins = {}
    for name in required:
        path = source / name
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for f in files:
            if f.is_file() and not any(x in f.parts for x in ("build", "__pycache__")) and f.suffix not in {".so", ".pyc", ".cubin"}:
                relative = f.relative_to(source)
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
                origins[str(relative)] = digest(f)
    native = dest / "neutrino/src"
    shutil.copyfile(ROOT / "native/eretrace/hqsb_capture.h", native / "hqsb_capture.h")
    replace(native / "cuda.c", 'CUresult cuLaunchKernel(CUfunction f,', 'static CUresult hqsb_capture_launch(CUfunction f,')
    replace(native / "cuda.c", '#include "ablation.h"', '#include "ablation.h"\n#include "hqsb_capture.h"')
    replace(native / "cuda.c", 'static void* get_hooked_driver_symbol(const char* symbol) {',
            'static void* get_hooked_driver_symbol(const char* symbol) {\n'
            '    if (strcmp(symbol, "cuProfilerStart") == 0) return (void*)cuProfilerStart;\n'
            '    if (strcmp(symbol, "cuProfilerStop") == 0) return (void*)cuProfilerStop;\n'
            )
    replace(native / "parse.py", '"cu": ["cuLaunchKernelEx_ptsz",', '"cu": ["cuProfilerStart", "cuProfilerStop", "cuLaunchKernelEx_ptsz",')
    replace(dest / "neutrino/probe/cuda.py", '    bin_path = os.path.join(workdir, name) + suffix',
            '    bin_path = os.path.join(workdir, name) + suffix\n'
            '    from hqsb.observation.eretrace_ptx import cached_ptx\n'
            '    cached = cached_ptx(bin_path)\n'
            '    if cached is not None:\n'
            '        shutil.copyfile(cached, os.path.join(workdir, name) + ".ptx")\n'
            '        return open(cached).read()')
    (dest / "neutrino/build").mkdir(exist_ok=True)
    python_deps = dest / "python"
    subprocess.run([sys.executable, "-m", "pip", "install", "--index-url", "https://pypi.org/simple", "--no-deps", "--target", str(python_deps), "toml==0.10.2"], check=True)
    env = {**os.environ, "PYTHONPATH": str(dest) + os.pathsep + str(python_deps), "PATH": str(a.cuda_home / "bin") + os.pathsep + os.environ["PATH"]}
    subprocess.run([sys.executable, str(dest / "scripts/build.py"), "--cuda-home", str(a.cuda_home), "--arch", a.arch], env=env, cwd=dest, check=True)
    shutil.copyfile(ROOT / "native/eretrace/hqsb_preload.c", native / "hqsb_preload.c")
    subprocess.run(["cc", str(native / "hqsb_preload.c"), "-fPIC", "-shared", "-O3", "-o", str(dest / "neutrino/build/preload.so"), "-ldl", "-lpthread"], check=True)
    subprocess.run([sys.executable, "-c", 'from pathlib import Path; import toml; from neutrino.language.compiler import compile; Path("build/block_sched.toml").write_text(toml.dumps(compile("CUDA", Path("neutrino/tools/block_sched.py").read_text())))'], env=env, cwd=dest, check=True)
    build = json.loads((dest / "build/build-manifest.json").read_text())
    build.update(schema_version="hqsb-eretrace-1", source_files=origins,
                 adapter_sha256=digest(ROOT / "native/eretrace/hqsb_capture.h"),
                 loader_sha256=digest(ROOT / "native/eretrace/hqsb_preload.c"),
                 installer_sha256=digest(Path(__file__)), cuda_home=str(a.cuda_home),
                 scope="selected_ordinary_launches_one_producer_one_context")
    # Discovery verifies the executable and probe implementation, including patches.
    build["runtime_files"] = {str(f.relative_to(dest)): digest(f) for f in dest.rglob("*")
                              if f.is_file() and f.suffix in {".py", ".h", ".c", ".cuh", ".cubin", ".toml"} and "python" not in f.relative_to(dest).parts}
    build["runtime_files"].update({"neutrino/build/libcuda.so.1": digest(dest / "neutrino/build/libcuda.so.1"),
                                   "neutrino/build/preload.so": digest(dest / "neutrino/build/preload.so")})
    (dest / "hqsb-runtime.json").write_text(json.dumps(build, indent=2) + "\n")
    print("ERETrace runtime:", dest, flush=True)


if __name__ == "__main__":
    main()
