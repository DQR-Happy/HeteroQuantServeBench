"""Redacted reproducibility manifest; run only through the authorized proxy."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
from pathlib import Path


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return {"exit_code": result.returncode, "stdout": result.stdout[:4000], "stderr": result.stderr[:1000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perf_capability():
    number = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
    if number is None:
        return {"available": None, "reason": "no syscall ABI mapping"}
    attr = ctypes.create_string_buffer(128)
    # Disabled CPU cycles event, userspace only. Opening probes permissions only.
    struct.pack_into("<IIQQQQQ", attr, 0, 0, 128, 0, 0, 0, 0, 1 | (1 << 5) | (1 << 6))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(number), ctypes.byref(attr), ctypes.c_int(0),
                      ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_ulong(8))
    if fd < 0:
        code = ctypes.get_errno()
        return {"available": False, "errno": code, "reason": os.strerror(code)}
    os.close(fd)
    return {"available": True, "reason": "permission probe only; no counters collected in timing runs"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.work_root.resolve()
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), "unavailable")
    limits = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        limits[name] = path.read_text().strip() if path.exists() else None
    repos = {}
    for path in sorted((root / "upstream").iterdir()):
        if (path / ".git").exists():
            # Transferred source can retain its original UID; trust only this exact
            # path for this invocation, without changing the user's Git config.
            git = ["git", "-c", f"safe.directory={path}", "-C", str(path)]
            repos[path.name] = {"commit": command([*git, "rev-parse", "HEAD"]),
                                "dirty": command([*git, "status", "--short"])}
    binaries = {}
    for path in sorted((root / "build").glob("*")):
        if path.is_file():
            binaries[path.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    otap = root / "upstream/otel-arrow/rust/otap-dataflow/target/release/df_engine"
    if otap.exists():
        binaries["df_engine"] = {"bytes": otap.stat().st_size, "sha256": sha(otap)}
    result = {"schema": "hqsb.capture.environment/1", "architecture": platform.machine(),
              "kernel": platform.release(), "python": platform.python_version(), "cpu_model": model,
              "effective_allowed_cpus": sorted(os.sched_getaffinity(0)), "cgroup_limits": limits,
              "compiler": command(["g++", "--version"]), "protoc": command(["protoc", "--version"]),
              "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
              "upstreams": repos, "binaries": binaries,
              "perf_event_open": perf_capability(),
              "storage": {str(path): command(["df", "-hT", str(path)]) for path in (root, Path("/root/autodl-tmp"))},
              "limitations": ["shared cloud host, frequency and other tenants not controlled", "no GPU measurements", "CPU affinity is enforced; no claim of exclusive physical cores", "1GiB address-space guard is not exact collector-owned or cgroup-resident bytes"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
