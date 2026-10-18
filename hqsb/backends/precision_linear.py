"""Native L4 linear storage and execution. No full-weight dequantized cache.

Weights/biases and packed buffers belong to the replacement module. Transient
activation quantization and INT8 padding belong to each synchronized forward.
"""

import torch
from torch import nn


def encode_weight(weight, bits, group_size, scale=None, zero=None):
    n, k = weight.shape
    group_size = k if bits == 8 else group_size
    groups = (k + group_size - 1) // group_size
    padded = torch.nn.functional.pad(
        weight.float(), (0, groups * group_size - k)
    ).reshape(n, groups, group_size)
    maximum = 127 if bits == 8 else 7
    if scale is None:
        scale = padded.abs().amax(-1).clamp_min(1e-12) / maximum
    scale = scale.float().reshape(n, groups)
    zero = (
        torch.zeros_like(scale, dtype=torch.int32)
        if zero is None
        else zero.int().reshape(n, groups)
    )
    codes = (
        (padded / scale[..., None] + zero[..., None])
        .round()
        .clamp(-maximum - 1, maximum)
        .to(torch.int8)
        .reshape(n, -1)[:, :k]
    )
    return encode_codes(codes, bits), scale, zero, group_size


def encode_codes(codes, bits):
    if bits == 8:
        return codes.to(torch.int8).view(torch.uint8).contiguous()
    if codes.shape[1] % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    return (
        ((codes[:, ::2].int() & 15) | ((codes[:, 1::2].int() & 15) << 4))
        .to(torch.uint8)
        .contiguous()
    )


class PackedLinear(nn.Module):
    def __init__(self, packed, scales, zeros, k, group, bits, bias=None):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales.float().contiguous())
        self.register_buffer("zeros", zeros.int().contiguous())
        self.register_buffer(
            "bias", bias.detach().clone() if bias is not None else None
        )
        self.in_features, self.out_features = k, packed.shape[0]
        self.group_size, self.bits, self.calls = group, bits, 0

    def forward(self, x):
        from ops.quant.w4a16_triton import PreparedWeights, gemm_low_bit

        self.calls += 1
        prepared = PreparedWeights(
            self.packed,
            self.scales,
            self.zeros,
            self.out_features,
            self.in_features,
            self.packed.shape[1],
            self.scales.shape[1],
            self.group_size,
            self.bits,
            "hqsb-canonical-signed-row-major",
            "l4-runtime",
        )
        result = gemm_low_bit(x.reshape(-1, self.in_features), prepared)
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*x.shape[:-1], self.out_features)


class Int8Linear(nn.Module):
    """SmoothQuant dynamic per-token activations, per-output-channel weights.

    Rescaling is folded by the official SmoothQuant calibration. Integer GEMM
    uses torch._int_mm (INT32 accumulation), with explicit zero-padded dimensions.
    """

    def __init__(self, codes, scales, bias=None):
        super().__init__()
        self.in_features, self.out_features = codes.shape[1], codes.shape[0]
        self.register_buffer(
            "codes",
            torch.nn.functional.pad(
                codes.to(torch.int8),
                (0, -self.in_features % 32, 0, -self.out_features % 8),
            ).contiguous(),
        )
        self.register_buffer("scales", scales.float().reshape(-1))
        self.register_buffer(
            "bias", bias.detach().clone() if bias is not None else None
        )
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        flat = x.reshape(-1, self.in_features)
        scale = flat.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        activation = (flat.float() / scale).round().clamp(-127, 127).to(torch.int8)
        m, k, n = activation.shape[0], self.in_features, self.out_features
        # CUDA's int_mm contract requires M > 16 even for a single decode row.
        pm, pk = ((max(m, 17) + 15) // 16) * 16 - m, (-k) % 32
        activation = torch.nn.functional.pad(activation, (0, pk, 0, pm))
        result = torch._int_mm(activation.contiguous(), self.codes.t())[:m, :n].float()
        result = (result * scale * self.scales[None]).to(x.dtype)
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*x.shape[:-1], n)


class FP8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        scale = weight.float().abs().amax().clamp_min(1e-12) / 448
        self.register_buffer("scales", scale.reshape(1))
        self.register_buffer(
            "codes", (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        self.register_buffer(
            "bias", bias.detach().clone() if bias is not None else None
        )
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        flat = x.reshape(-1, self.in_features)
        scale = (flat.float().abs().amax().clamp_min(1e-12) / 448).reshape(1)
        a = (flat.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        m, k, n = len(flat), self.in_features, self.out_features
        a = torch.nn.functional.pad(a, (0, -k % 16, 0, -m % 16))
        w = torch.nn.functional.pad(self.codes, (0, -k % 16, 0, -n % 16)).t()
        result = torch._scaled_mm(
            a,
            w,
            scale_a=scale,
            scale_b=self.scales,
            out_dtype=x.dtype,
            use_fast_accum=False,
        )[:m, :n]
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*x.shape[:-1], n)
