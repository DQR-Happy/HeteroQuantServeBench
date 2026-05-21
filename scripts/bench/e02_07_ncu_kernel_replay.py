#!/usr/bin/env python3
"""E02-07 shape-exact kernel replay harness for Nsight Compute.

Why this exists
---------------
In-model Nsight Compute capture is **not possible** on this board:

* ``--replay-mode kernel`` makes the process exceed the 7.6 GiB unified memory
  and the kernel OOM killer terminates it (``Out of memory: Killed process
  ... pt_main_thread``, the same ``nvmap`` signature E02-06 section 4.13
  documents for the caching allocator).
* ``--replay-mode application`` does survive, but it relaunches the whole
  process once per collection pass. With a ~40 s model load per pass, a
  full-section run takes ~23 minutes for a *tiny* workload and would take
  far longer for ``decode_heavy`` — for a single candidate.

So the hardware counters come from a **shape-exact isolated replay**: the same
operator, the same dtype, the same tensor shapes and the same call path
(``aten::mm`` for the projections, ``aten::cat`` for the KV concat), placed
inside an NVTX range named after the candidate and the phase/context it comes
from. Isolation means the counters describe the kernel, not the model around
it, and the report never treats the replay duration as a model latency.

Faithfulness is checked, not assumed: the replay's grid/block/shared-memory
fingerprint is compared against the grid/block/shared-memory recorded for that
kernel ``inside`` the model run by the PyTorch profiler trace. A mismatch is a
failed equivalence check, not a footnote.

One in-model anchor (the top decode GEMM, application replay, SpeedOfLight
only) is captured separately by the orchestrator so the isolated measurement
has an in-model reference point.

Usage (on the Jetson, via ./scripts/remote_run.sh):

    scripts/bench/e02_07_ncu_kernel_replay.py \\
        --kind linear --m 1 --k 2048 --n 6144 \\
        --label D1.decode.T136 --repeat 4 --output out.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from hqsb.benchmark.multilevel_profiling import RUN_RANGE

logger = logging.getLogger("e02_07.replay")

REPLAY_RANGE_PREFIX = "e02_07_replay"


def _nvtx_push(name: str) -> None:
    try:
        torch.cuda.nvtx.range_push(name)
    except Exception:
        pass


def _nvtx_pop() -> None:
    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _build_linear(m: int, k: int, n: int, device: torch.device, seed: int):
    """Reproduce a model ``nn.Linear`` as exactly as the shapes allow.

    The weight is ``[out_features, in_features] = [n, k]`` and the call is
    ``F.linear``, *not* ``x @ w`` with a ``[k, n]`` weight. Qwen3 stores its
    projections as ``nn.Linear`` weights, so the model's GEMM is ``C = A * Bᵀ``
    and cuBLASLt selects the transpose-B (``..._tn``) kernel family. Building
    ``[k, n]`` instead would ask for ``..._nn`` and silently pick a *different*
    kernel with a different tile — which is precisely the faithfulness failure
    this harness has to avoid, and which an earlier revision did produce.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, k, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    weight = torch.randn(n, k, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    return (
        lambda: torch.nn.functional.linear(x, weight),
        {
            "inputs": {"x": [m, k], "weight": [n, k]},
            "dtype": "float16",
            "call": "F.linear(x, weight) with weight [out_features, in_features]",
        },
    )


def _build_softmax(shape: List[int], dim: int, device: torch.device, seed: int):
    """Reproduce ``aten::_softmax`` as the eager attention path calls it.

    Qwen3 eager attention applies the mask with ``add`` and then a float32
    ``_softmax`` over the last dimension of a ``[1, heads, T, T]`` score
    tensor. Replaying that exact shape is what decides which
    ``cunn_SoftMaxForwardSmem`` template instantiation (and therefore which
    grid/block) is launched.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn(*shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    return (
        lambda: torch.nn.functional.softmax(scores, dim=dim),
        {
            "inputs": {"scores": shape, "dim": dim},
            "dtype": "float16",
            "call": "F.softmax(scores, dim=-1)",
        },
    )


def _build_bmm(
    a_shape: List[int],
    b_source_shape: List[int],
    device: torch.device,
    seed: int,
):
    """Reproduce the eager attention ``Q @ Kᵀ`` batched matmul.

    ``--b-shape`` is the shape of the *contiguous source* tensor (``K``), and
    the replay passes ``K.transpose(-1, -2)`` to ``bmm``. That matters: the
    model feeds cuBLAS a transposed *view* of ``K``, so the selected kernel is
    the ``..._tn`` variant. Handing ``bmm`` a contiguous ``[heads, d, T]``
    tensor instead would ask for ``..._nn`` and yield a different kernel with
    the same grid — the same silent faithfulness failure the ``F.linear``
    weight layout avoids for the projections.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(*a_shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    b_source = torch.randn(*b_source_shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    return (
        lambda: torch.bmm(a, b_source.transpose(-1, -2)),
        {
            "inputs": {"a": a_shape, "b_source": b_source_shape},
            "dtype": "float16",
            "call": "torch.bmm(a, b_source.transpose(-1, -2))",
        },
    )


def _build_cat(shapes: List[List[int]], dim: int, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = [
        torch.randn(*shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.float16
        )
        for shape in shapes
    ]
    return (
        lambda: torch.cat(tensors, dim=dim),
        {"inputs": {"tensors": shapes, "dim": dim}, "dtype": "float16"},
    )


def _build_rmsnorm(rows: int, hidden: int, device: torch.device, seed: int):
    """Reference RMSNorm inside-out as transformers computes it in eager mode.

    Kept for completeness: the S03 teaching line. It is *not* one of the
    pre-registered E02-07 candidates, so it is only used when explicitly
    requested on the command line.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, hidden, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    weight = torch.ones(hidden, dtype=torch.float16, device=device)

    def _run():
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + 1e-6)
        return weight * hidden_states.to(torch.float16)

    return _run, {"inputs": {"x": [rows, hidden]}, "dtype": "float16"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="E02-07 shape-exact NCU replay harness")
    parser.add_argument(
        "--kind",
        choices=("linear", "cat", "softmax", "bmm", "rmsnorm"),
        default="linear",
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument(
        "--shape",
        default="",
        help="comma-separated tensor shape for kind=softmax, e.g. '1,16,2048,2048'",
    )
    parser.add_argument(
        "--a-shape", default="", help="comma-separated first shape for kind=bmm"
    )
    parser.add_argument(
        "--b-shape", default="", help="comma-separated second shape for kind=bmm"
    )
    parser.add_argument(
        "--cat-shapes",
        default="",
        help="semicolon-separated shapes, e.g. '1,8,1,128;1,8,136,128'",
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--label", default="replay")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is required for the replay harness")
        return 2
    device = torch.device("cuda")

    def _parse_shape(text: str) -> List[int]:
        return [int(v) for v in text.split(",") if v.strip()]

    if args.kind == "linear":
        run, spec = _build_linear(args.m, args.k, args.n, device, args.seed)
    elif args.kind == "cat":
        shapes = [
            _parse_shape(chunk) for chunk in args.cat_shapes.split(";") if chunk.strip()
        ]
        if not shapes:
            logger.error("--cat-shapes is required for kind=cat")
            return 2
        run, spec = _build_cat(shapes, args.dim, device, args.seed)
    elif args.kind == "softmax":
        shape = _parse_shape(args.shape)
        if not shape:
            logger.error("--shape is required for kind=softmax")
            return 2
        run, spec = _build_softmax(shape, args.dim, device, args.seed)
    elif args.kind == "bmm":
        a_shape, b_shape = _parse_shape(args.a_shape), _parse_shape(args.b_shape)
        if not a_shape or not b_shape:
            logger.error("--a-shape and --b-shape are required for kind=bmm")
            return 2
        run, spec = _build_bmm(a_shape, b_shape, device, args.seed)
    else:
        run, spec = _build_rmsnorm(args.rows, args.hidden, device, args.seed)

    spec.update(
        {
            "label": args.label,
            "kind": args.kind,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "seed": args.seed,
        }
    )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    latencies: List[float] = []
    _nvtx_push(RUN_RANGE)
    _nvtx_push(f"{REPLAY_RANGE_PREFIX}.{args.label}")
    for _ in range(args.repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    _nvtx_pop()
    _nvtx_pop()

    payload: Dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "spec": spec,
        "nvtx_ranges": [RUN_RANGE, f"{REPLAY_RANGE_PREFIX}.{args.label}"],
        "latencies_ms": latencies,
        "median_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "note": (
            "Isolated shape-exact replay. Durations are NOT model latencies: "
            "NCU serialises and replays launches, flushes caches, and no other "
            "kernel competes for the device."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"label": args.label, "spec": spec, "median_ms": payload["median_ms"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
