"""Real CUDA bridge/codec acceptance, including untouched cuBLAS and Triton."""
import argparse
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
    def ere_vector_add(X, Y, Z, N: tl.constexpr):
        offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
        x = tl.load(X + offsets, offsets < N, other=0)
        y = tl.load(Y + offsets, offsets < N, other=0)
        tl.store(Z + offsets, x + y, offsets < N)

    # cuBLAS and ordinary torch execution keep the installed CUDA driver.
    torch.testing.assert_close(torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda"),
                               torch.full((32, 32), 32., device="cuda"), rtol=0, atol=0)
    n = 1 << 20
    x, y = torch.ones(n, device="cuda"), torch.full((n,), 2., device="cuda")
    z = torch.empty_like(x)
    ere_vector_add[(triton.cdiv(n, 256),)](x, y, z, n)
    torch.cuda.synchronize()
    hook = ctypes.CDLL(os.environ["NEUTRINO_HOOK_DRIVER"])
    for name in ("hqsb_eretrace_start", "hqsb_eretrace_stop"):
        getattr(hook, name).argtypes = []
        getattr(hook, name).restype = ctypes.c_int
    assert hook.hqsb_eretrace_start() == 0
    for _ in range(2):
        ere_vector_add[(triton.cdiv(n, 256),)](x, y, z, n)
    torch.cuda.synchronize()
    assert hook.hqsb_eretrace_stop() == 0
    torch.testing.assert_close(z, x + y, rtol=0, atol=0)
    print("cuBLAS + Triton numerical checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--codec", choices=["reference", "raw"], default="reference")
    parser.add_argument("--python", default=".venv-sglang/bin/python")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from hqsb.core.diagnostics import DiagnosticPlan
    from hqsb.observation.eretrace_tools import launch_environment, discover
    from hqsb.observation.eretrace_reports import summarize
    from hqsb.observation.lease import DeviceLease
    from hqsb.observation.storage import atomic_json
    assert discover(args.runtime)["installed"]
    args.output.mkdir(parents=True, exist_ok=False)
    plan = DiagnosticPlan(source_run_id="run_" + "0" * 32, tool="eretrace", ere_kernel="ere_vector_add", ere_codec=args.codec, ere_verify=True, launch_count=2)
    env = launch_environment(args.runtime, args.output, plan.model_dump(), args.output / "triton")
    env.update(TRITON_CACHE_DIR=str(args.output.resolve() / "triton"))
    lease = DeviceLease(Path.home() / ".cache/hqsb/device0.lock")
    lease.acquire()
    try:
        subprocess.run([args.python, str(Path(__file__).resolve()), "--child"], env={**os.environ, **env}, check=True, timeout=180)
        panel = summarize(args.output)
        assert panel["availability"] == "observed" and panel["counts"]["launches"] == 2, panel
        if args.codec == "reference":
            assert panel["encoding_counts"].get("reference", 0) > 0
        atomic_json(args.output / "panel.json", panel)
        print(json.dumps({"counts": panel["counts"], "encodings": panel["encoding_counts"], "ratio": panel["logical_to_stored_ratio"]}), flush=True)
    finally:
        lease.release()


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
