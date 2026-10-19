"""Lossy algorithms over a private CPU model. Checks do not alter thresholds."""

import math
import hashlib
import torch
from torch import nn
from hqsb.backends.structure_modules import FactorizedLinear, inspect, replace


def select(model, plan):
    candidates = inspect(model)[
        "groups" if plan.method == "mlp_channels" else "linears"
    ]
    available = {row["name"]: row for row in candidates}
    names = plan.targets or list(available)[: plan.max_modules]
    if not names or any(n not in available for n in names):
        raise ValueError("L5_TARGET_NOT_AN_INSPECTED_COMPATIBLE_MODULE")
    # Bound SVD workspace/work before starting expensive matrix factorization.
    if plan.method == "svd_low_rank" and any(
        max(available[n]["in_features"], available[n]["out_features"]) > 8192
        or available[n]["in_features"] * available[n]["out_features"] > 16 * 1024**2
        for n in names
    ):
        raise ValueError("L5_SVD_MATRIX_BUDGET")
    return names


def activation_stats(model, tokenizer, names, plan, check, context_limit):
    sums, counts, handles = {}, {}, []

    def hook(name):
        def collect(_owner, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            if not torch.isfinite(x).all():
                raise ValueError("L5_NONFINITE_CALIBRATION_ACTIVATION")
            value = x.square().sum(0)
            sums[name] = sums.get(name, 0) + value
            counts[name] = counts.get(name, 0) + x.shape[0]

        return collect

    hashes = []
    try:
        for name in names:
            handles.append(
                model.get_submodule(name).register_forward_pre_hook(hook(name))
            )
        with torch.inference_mode():
            for text in plan.calibration:
                check()
                ids = tokenizer.encode(text, add_special_tokens=False)[
                    : min(context_limit, plan.calibration_max_tokens)
                ]
                if not ids:
                    raise ValueError("L5_EMPTY_CALIBRATION_TOKENS")
                inputs = torch.tensor([ids], dtype=torch.long)
                model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    use_cache=False,
                )
                hashes.extend(
                    [
                        hashlib.sha256(text.encode()).hexdigest(),
                        hashlib.sha256(
                            tokenizer.decode(ids, skip_special_tokens=False).encode()
                        ).hexdigest(),
                    ]
                )
    finally:
        for handle in handles:
            handle.remove()
    if set(counts) != set(names):
        raise ValueError("L5_CALIBRATION_TARGET_NOT_EXECUTED")
    return {name: (sums[name] / counts[name]).sqrt() for name in names}, hashes


def prune_weight(weight, sparsity, pattern, scales=None):
    score = weight.float().abs()
    if scales is not None:
        score = score * scales[None, :]
    mask = torch.ones_like(weight, dtype=torch.bool)
    if pattern == "2:4":
        if weight.shape[1] % 4:
            raise ValueError("L5_2_4_INPUT_WIDTH_NOT_DIVISIBLE_BY_4")
        grouped = score.reshape(score.shape[0], -1, 4)
        indices = grouped.argsort(dim=-1, stable=True)[..., :2]
        mask.reshape(grouped.shape).scatter_(-1, indices, False)
    else:
        count = math.floor(weight.shape[1] * sparsity)
        mask.scatter_(-1, score.argsort(dim=-1, stable=True)[:, :count], False)
    return weight * mask, mask


def apply(model, names, plan, check, progress, scales=None):
    records = []
    with torch.no_grad():
        for index, name in enumerate(names):
            check()
            module = model.get_submodule(name)
            if plan.method == "mlp_channels":
                gate, up, down = module.gate_proj, module.up_proj, module.down_proj
                width = gate.out_features
                kept = max(1, math.floor(width * plan.keep_ratio))
                score = (
                    gate.weight.square().sum(1)
                    + up.weight.square().sum(1)
                    + down.weight.square().sum(0)
                )
                channels = (
                    score.argsort(descending=True, stable=True)[:kept].sort().values
                )
                for suffix in ("gate_proj", "up_proj", "down_proj"):
                    old = getattr(module, suffix)
                    weight = (
                        old.weight[channels, :]
                        if suffix != "down_proj"
                        else old.weight[:, channels]
                    )
                    new = nn.Linear(
                        weight.shape[1],
                        weight.shape[0],
                        bias=old.bias is not None,
                        dtype=weight.dtype,
                    )
                    new.weight.copy_(weight)
                    if old.bias is not None:
                        new.bias.copy_(
                            old.bias[channels] if suffix != "down_proj" else old.bias
                        )
                    setattr(module, suffix, new)
                if hasattr(module, "intermediate_size"):
                    module.intermediate_size = kept
                row = {
                    "kind": "mlp_channels",
                    "name": name,
                    "original_width": width,
                    "width": kept,
                    "kept_channels": channels.tolist(),
                    "metric": "joint_gate_up_rows_and_down_columns_squared_magnitude",
                }
            elif plan.method == "svd_low_rank":
                weight = module.weight.float()
                m, n = weight.shape
                rank = max(1, min(m, n, math.floor(plan.keep_ratio * m * n / (m + n))))
                u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                compact = FactorizedLinear(
                    n, m, rank, module.bias is not None, dtype=module.weight.dtype
                )
                compact.left.weight.copy_(vh[:rank, :])
                compact.right.weight.copy_(u[:, :rank] * s[:rank][None, :])
                if module.bias is not None:
                    compact.right.bias.copy_(module.bias)
                approximation = (
                    compact.right.weight.float() @ compact.left.weight.float()
                )
                row = {
                    "kind": "factorized",
                    "name": name,
                    "original_shape": [n, m],
                    "rank": rank,
                    "relative_frobenius_error": float(
                        (weight - approximation).norm() / weight.norm().clamp_min(1e-30)
                    ),
                    "original_parameters": m * n,
                    "factor_parameters": rank * (m + n),
                    "solver": "torch.linalg.svd_full_matrices_false",
                }
                replace(model, name, compact)
            else:
                original = module.weight.detach().clone()
                pruned, mask = prune_weight(
                    original, plan.sparsity, plan.pattern, (scales or {}).get(name)
                )
                module.weight.copy_(pruned)
                row = {
                    "kind": "masked_dense",
                    "name": name,
                    "shape": list(original.shape),
                    "masked_fraction": float((~mask).float().mean()),
                    "actual_zero_fraction": float((pruned == 0).float().mean()),
                    "execution": "dense_linear",
                    "pattern": plan.pattern,
                    "sparse_kernel": False,
                    "relative_frobenius_error": float(
                        (pruned - original).norm() / original.norm().clamp_min(1e-30)
                    ),
                }
            records.append(row)
            progress(index + 1, len(names), name)
    return records
