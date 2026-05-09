"""TileLang runtime probe (kept free of ``from __future__ import annotations``).

TileLang/TVM inspects the *actual* type annotations of a ``T.prim_func`` to
build the IR; ``from __future__ import annotations`` turns those annotations
into strings, which TVM rejects with "Forward references must evaluate to
types". This module therefore deliberately omits that future import, and is
imported lazily by :mod:`ops.capability` so the rest of the package keeps its
string-annotation ergonomics.
"""

import torch

import tilelang
import tilelang.language as T


@T.prim_func
def _probe(
    A: T.Tensor((2,), "float32"),
    B: T.Tensor((2,), "float32"),
    C: T.Tensor((2,), "float32"),
):
    with T.Kernel(2, threads=32) as bx:
        C[bx] = A[bx] + B[bx]


def run_probe() -> bool:
    """Compile + run a 2-element add kernel; return True on success."""
    kernel = tilelang.compile(_probe, out_idx=[2])
    a = torch.ones(2, device="cuda")
    b = torch.ones(2, device="cuda")
    c = kernel(a, b)
    torch.cuda.synchronize()
    return bool(torch.allclose(c, torch.full_like(c, 2.0)))


__all__ = ["run_probe"]
