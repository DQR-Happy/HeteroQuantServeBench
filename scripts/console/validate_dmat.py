"""On-device address oracle for the complete DMAT collection/decoding path."""
import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def child():
    global tl
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def hqsb_dmat_add(X, Y, Z, N: tl.constexpr):
        i = tl.program_id(0) * 256 + tl.arange(0, 256)
        x = tl.load(X + i, i < N, other=0)
        y = tl.load(Y + i, i < N, other=0)
        tl.store(Z + i, x + y, i < N)

    n = 1025
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.ones_like(x)
    z = torch.empty_like(x)
    hqsb_dmat_add[(triton.cdiv(n, 256),)](x, y, z, n)
    torch.cuda.synchronize()
    hook = ctypes.CDLL(os.environ["NEUTRINO_HOOK_DRIVER"])
    for name in ("hqsb_eretrace_start", "hqsb_eretrace_stop"):
        getattr(hook, name).argtypes = []
        getattr(hook, name).restype = ctypes.c_int
    assert hook.hqsb_eretrace_start() == 0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        hqsb_dmat_add[(triton.cdiv(n, 256),)](x, y, z, n)
    stream.synchronize()
    assert hook.hqsb_eretrace_stop() == 0
    torch.testing.assert_close(z, x + y, rtol=0, atol=0)
    Path(os.environ["HQSB_DMAT_ORACLE"]).write_text(json.dumps({
        "n": n, "x": str(x.data_ptr()), "y": str(y.data_ptr()), "z": str(z.data_ptr()), "output_equal": True,
    }) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=32)
    args = parser.parse_args()
    from hqsb.core.diagnostics import DiagnosticPlan
    from hqsb.observation.eretrace_tools import launch_environment, discover
    from hqsb.observation.dmat_reports import read_frame, summarize_memory
    from hqsb.observation.lease import DeviceLease
    from hqsb.observation.storage import atomic_json
    assert "memory" in discover(args.runtime).get("capabilities", [])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = DiagnosticPlan(source_run_id="run_" + "0" * 32, tool="eretrace", ere_mode="memory", ere_codec="raw",
                          ere_kernel="hqsb_dmat_add", memory_capacity=args.capacity, launch_count=1)
    env = launch_environment(args.runtime, root, plan.model_dump(), root / "triton")
    env.update(TRITON_CACHE_DIR=str(root / "triton"), HQSB_DMAT_ORACLE=str(root / "oracle.json"))
    lease = DeviceLease(Path.home() / ".cache/hqsb/device0.lock")
    lease.acquire()
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env={**os.environ, **env}, check=True, timeout=180)
    finally:
        lease.release()
    panel = summarize_memory(root, page_bytes=4096)
    oracle = json.loads((root / "oracle.json").read_text())
    assert panel["counts"]["launches"] == 1, panel
    capture = panel["memory_launches"][0]
    frame = read_frame(root / capture["trace"])
    observed = {"read": Counter(), "write": Counter()}
    for i, count in enumerate(frame["counts"]):
        for row in frame["records"][i, :min(int(count), frame["capacity"])]:
            operation = capture["plot"]["sites"][int(row["site"])]["operation"]
            for offset in range(0, int(row["width"]), 4):
                observed[operation][int(row["address"]) + offset] += 1
    expected = {"read": Counter(int(oracle[p]) + offset * 4 for p in ("x", "y") for offset in range(oracle["n"])),
                "write": Counter(int(oracle["z"]) + offset * 4 for offset in range(oracle["n"]))}
    if args.capacity >= 3:
        assert observed == expected, "Actual per-address read/write multiset differs from kernel oracle"
        assert panel["availability"] == "observed", panel
    else:
        assert panel["availability"] == "partial" and panel["counts"]["dropped_events"] > 0
    assert sum(c["events"] for c in capture["plot"]["cells"]) == panel["counts"]["events"]
    atomic_json(root / "panel.json", panel)
    atomic_json(root / "acceptance.json", {"passed": True, "device": "remote_CUDA", "capacity": args.capacity,
                "address_multisets_equal": observed == expected, "numerical_output_equal": oracle["output_equal"],
                "counts": panel["counts"], "availability": panel["availability"]})
    print((root / "acceptance.json").read_text(), flush=True)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
