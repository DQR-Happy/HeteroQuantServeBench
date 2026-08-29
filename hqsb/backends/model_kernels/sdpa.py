"""Vendor-wheel-compatible SDPA: explicit GQA repeat, pinned backend, no fallback.

The supplied additive/boolean mask is authoritative. Without one, build a
bottom-right causal mask, including the cached-prefix offset. Never use the
upper-left is_causal shortcut for decode. K/V cache storage is only borrowed.
"""


def run(q, k, v, mask, scale, variant, *, unpadded_causal=False):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if torch.is_grad_enabled() or q.device.type != "cuda" or q.dtype != torch.float16:
        raise ValueError("KERNEL_SDPA_INFERENCE_CUDA_FP16_ONLY")
    b,h,t,d = q.shape
    if k.shape != v.shape or k.shape[0] != b or k.shape[-1] != d or h % k.shape[1] or t > k.shape[2]:
        raise ValueError("KERNEL_SDPA_GQA_SHAPE")
    length = k.shape[2]
    flash = variant == "torch-sdpa-flash-gqa-repeat-v1"
    causal = False
    if flash:
        if not unpadded_causal:
            raise ValueError("KERNEL_FLASH_UNPADDED_CAUSAL_PROOF_REQUIRED")
        if t not in (1, length):
            raise ValueError("KERNEL_FLASH_CACHED_MULTI_QUERY_UNSUPPORTED")
        if mask is not None and (mask.ndim != 4 or mask.shape[-2:] != (t,length)):
            raise ValueError("KERNEL_FLASH_CAUSAL_MASK_SHAPE")
        # Vendor 2.5 flash cannot consume arbitrary additive masks. The model
        # boundary proved an unpadded sequence; square prefill is causal, a
        # single cached query can attend to every key. No upper-left decode.
        mask, causal = None, t == length
    groups = h // k.shape[1]
    if groups != 1:
        k = k.repeat_interleave(groups,dim=1)
        v = v.repeat_interleave(groups,dim=1)
    if mask is None and not flash:
        query = torch.arange(t,device=q.device) + length - t
        mask = torch.arange(length,device=q.device)[None,:] <= query[:,None]
    elif mask is not None:
        if mask.device != q.device or mask.dtype not in (torch.bool, q.dtype, torch.float32) or mask.ndim != 4:
            raise ValueError("KERNEL_SDPA_MASK_UNSUPPORTED")
        if mask.shape[-2:] != (t,length) or mask.shape[0] not in (1,b) or mask.shape[1] not in (1,h):
            raise ValueError("KERNEL_SDPA_MASK_SHAPE")
    backend = SDPBackend.FLASH_ATTENTION if flash else SDPBackend.MATH
    with sdpa_kernel(backend):
        out = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,
                                                                is_causal=causal,scale=scale)
    # This NVIDIA 2.5 wheel returns NaN on fully hidden rows in the math
    # backend. Define SDPA zero-row semantics without a host/device sync.
    if mask is not None:
        visible = mask.any(-1,keepdim=True) if mask.dtype == torch.bool else (~torch.isneginf(mask)).any(-1,keepdim=True)
        out = out.masked_fill(~visible, 0)
    return out.transpose(1,2).contiguous()
