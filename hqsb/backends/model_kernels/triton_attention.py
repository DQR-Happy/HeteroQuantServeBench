"""Inspectable online-softmax Attention, with a fixed K/V tile comparison.

FP16, inference, B=1, causal unpadded self-attention (including cached decode).
GQA maps query heads directly onto K/V heads. FP32 softmax/accumulation.
The two variants differ only in BLOCK_N (32 or 64); both retain Q on chip.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def hqsb_dmat_attention(Q, K, V, O,
    qh: tl.constexpr, qs: tl.constexpr, qd: tl.constexpr,
    kh: tl.constexpr, ks: tl.constexpr, kd: tl.constexpr,
    vh: tl.constexpr, vs: tl.constexpr, vd: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr,
    D: tl.constexpr, SCALE: tl.constexpr, CAUSAL: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    m = block * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    q = tl.load(Q + head * qh + m[:, None] * qs + d[None, :] * qd,
                (m[:, None] < M) & (d[None, :] < D), other=0)
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    end = tl.minimum(N, (block + 1) * BM + N - M) if CAUSAL else N
    for start in range(0, end, BN):
        n = start + tl.arange(0, BN)
        k = tl.load(K + (head // GROUP) * kh + n[None, :] * ks + d[:, None] * kd,
                    (n[None, :] < N) & (d[:, None] < D), other=0)
        logits = tl.dot(q, k) * SCALE
        valid = n[None, :] < N
        if CAUSAL:
            valid = valid & (n[None, :] <= m[:, None] + N - M)
        logits = tl.where(valid, logits, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        p = tl.exp(logits - next_max[:, None])
        alpha = tl.exp(maximum - next_max)
        v = tl.load(V + (head // GROUP) * vh + n[:, None] * vs + d[None, :] * vd,
                    (n[:, None] < N) & (d[None, :] < D), other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        denom = denom * alpha + tl.sum(p, 1)
        maximum = next_max
    result = acc / denom[:, None]
    tl.store(O + (head * M + m[:, None]) * D + d[None, :], result,
             (m[:, None] < M) & (d[None, :] < D))


def run(q, k, v, scale, *, block_n=64, causal=True, layer=None, phase=None):
    if torch.is_grad_enabled():
        raise ValueError('DMAT_ATTENTION_INFERENCE_ONLY')
    if block_n not in (32, 64) or q.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1 or k.shape != v.shape:
        raise ValueError('DMAT_ATTENTION_SHAPE')
    if any(t.device != q.device or t.dtype != torch.float16 or t.device.type != 'cuda' or any(s < 0 for s in t.stride()) for t in (q, k, v)):
        raise ValueError('DMAT_ATTENTION_DEVICE_DTYPE_STRIDE')
    _, h, m, d = q.shape
    _, hk, n, dk = k.shape
    if min(m, n, hk) < 1 or n < m or h % hk or d != dk or not 16 <= d <= 256:
        raise ValueError('DMAT_ATTENTION_DIMENSIONS')
    out = torch.empty((1, h, m, d), dtype=q.dtype, device=q.device)
    from hqsb.observation.dmat_context import memory_context
    with memory_context({'Q': q, 'K': k, 'V': v, 'O': out}, layer=layer, phase=phase,
                        operation='scaled_dot_product_attention', causal=causal,
                        implementation='hqsb-online-softmax-triton-v1', block_m=32, block_n=block_n,
                        semantic_shape={'q': list(q.shape), 'k': list(k.shape), 'v': list(v.shape)}, scale=scale):
        hqsb_dmat_attention[(triton.cdiv(m, 32), h)](
            q, k, v, out, *q.stride()[1:], *k.stride()[1:], *v.stride()[1:],
            m, n, h, h // hk, d, scale, causal, 32, block_n, triton.next_power_of_2(d),
            num_warps=4, num_stages=1)
    return out.transpose(1, 2)
