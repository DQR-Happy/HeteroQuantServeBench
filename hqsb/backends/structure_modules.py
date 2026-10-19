"""Real compact modules and geometry validation; no HTTP or artifact ownership."""

from torch import nn


class FactorizedLinear(nn.Module):
    """W ~= right @ left; two real GEMMs, no dense reconstruction in forward."""

    def __init__(self, in_features, out_features, rank, bias=False, *, dtype=None):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.left = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.right = nn.Linear(rank, out_features, bias=bias, dtype=dtype)

    def forward(self, inputs):
        return self.right(self.left(inputs))


def inspect(model):
    groups, linears = [], []
    for name, module in model.named_modules():
        if not name:
            continue
        gate, up, down = (
            getattr(module, key, None) for key in ("gate_proj", "up_proj", "down_proj")
        )
        if all(type(x) is nn.Linear for x in (gate, up, down)):
            if (
                gate.in_features == up.in_features == down.out_features
                and gate.out_features == up.out_features == down.in_features
            ):
                groups.append(
                    {
                        "name": name,
                        "width": gate.out_features,
                        "hidden": gate.in_features,
                        "kind": "gated_mlp",
                    }
                )
        if (
            type(module) is nn.Linear
            and ".mlp." in name
            and name.rsplit(".", 1)[-1] in {"gate_proj", "up_proj", "down_proj"}
        ):
            linears.append(
                {
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                }
            )
    return {"groups": groups, "linears": linears}


def replace(model, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, module)


def restore_shapes(model, records):
    for row in records:
        if row["kind"] == "factorized":
            old = model.get_submodule(row["name"])
            if (
                type(old) is not nn.Linear
                or [old.in_features, old.out_features] != row["original_shape"]
            ):
                raise ValueError("L5_FACTOR_SHAPE_IDENTITY")
            replace(
                model,
                row["name"],
                FactorizedLinear(
                    old.in_features,
                    old.out_features,
                    row["rank"],
                    old.bias is not None,
                    dtype=old.weight.dtype,
                ),
            )
        elif row["kind"] == "mlp_channels":
            group = model.get_submodule(row["name"])
            if not any(
                g["name"] == row["name"] and g["width"] == row["original_width"]
                for g in inspect(model)["groups"]
            ):
                raise ValueError("L5_MLP_SHAPE_IDENTITY")
            for suffix in ("gate_proj", "up_proj", "down_proj"):
                old = getattr(group, suffix)
                inp = old.in_features if suffix != "down_proj" else row["width"]
                out = old.out_features if suffix == "down_proj" else row["width"]
                setattr(
                    group,
                    suffix,
                    nn.Linear(
                        inp, out, bias=old.bias is not None, dtype=old.weight.dtype
                    ),
                )
            if hasattr(group, "intermediate_size"):
                group.intermediate_size = row["width"]
        elif row["kind"] != "masked_dense":
            raise ValueError("L5_UNKNOWN_MODULE_RECORD")
