#!/usr/bin/env python3
"""One real Qwen projection for external NCU / compute-sanitizer collection."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from s05_remaining_common import load_weight, prepared_weight


def main():
    import torch
    import math
    from ops.quant.w4a16_triton import gemm_low_bit
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--model", default="~/models/hqsb/Qwen3-1.7B")
    args = parser.parse_args()
    with torch.inference_mode():
        weight = load_weight(args.model, "model.layers.0.self_attn.q_proj.weight").to("cuda", torch.float16)
        prepared, reference_weight = prepared_weight(weight, args.bits)
        torch.manual_seed(5606)
        x = torch.randn(args.m, prepared.k, device="cuda", dtype=torch.float16)
        for _ in range(2):
            actual = gemm_low_bit(x, prepared)
        torch.cuda.synchronize()
        reference = x @ reference_weight.t()
        error = float((actual.float() - reference.float()).norm() / reference.float().norm())
        max_abs = float((actual.float() - reference.float()).abs().max())
        if (not math.isfinite(error) or error > .002 or max_abs > .125
                or not torch.isfinite(actual).all() or not torch.isfinite(reference).all()):
            raise AssertionError(f"kernel relative error {error}")
        print(f"W{args.bits} M={args.m} N={prepared.n} K={prepared.k} rel_l2={error} max_abs={max_abs}; packed_hash={prepared.variant_hash}")


if __name__ == "__main__":
    main()
