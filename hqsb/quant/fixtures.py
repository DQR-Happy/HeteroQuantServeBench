"""Synthetic artifacts used for self-checks (never for experiment results).

The QuantArtifact self-checks (round-trip, compatibility, fault injection,
kernel oracles) need a *complete* artifact with known values, but they must
not need a model download or a GPU. This module builds a tiny deterministic
weight-like tensor and packages it as a real :class:`QuantArtifactDocument`
with canonical values, a packed variant, compatibility records and known
limitations.

The fixtures are labelled ``synthetic`` in ``method``/``known_limitations``;
they can never be mistaken for a quantized Qwen weight, and no experiment
driver may report them as model evidence.
"""

from __future__ import annotations

import os
from typing import Optional

from hqsb.quant.artifact import (
    CalibrationProvenance,
    CompatibilityRecord,
    ModelIdentity,
    QuantArtifactDocument,
)
from hqsb.quant.packing import (
    LAYOUT_W4A16_HIFIRST_NK_V1,
    LAYOUT_W4A16_ROWMAJOR_NK_V1,
    LAYOUT_W8A16_ROWMAJOR_NK_V1,
    pack_kernel_variant,
)
from hqsb.quant.rtn import quantize
from hqsb.quant.spec import (
    GRANULARITY_PER_GROUP,
    MAIN_W4_GROUP,
    QuantScheme,
)

#: Deterministic pseudo-values: no RNG, no device, no model.
def synthetic_weight_values(rows: int, cols: int, *, offset: int = 0) -> list:
    """Deterministic, thin-tailed values in roughly ``[-3, 3]``."""
    values = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            numerator = ((index * 7 + offset) % 61) - 30
            values.append(numerator / 10.0)
    return values


def build_tiny_artifact(
    *,
    name: str = "synthetic.q_proj.weight",
    bits: int = 4,
    group_size: Optional[int] = 128,
    rows: int = 4,
    cols: int = 256,
    target_arch: str = "sm_86",
    model: Optional[ModelIdentity] = None,
    seed_offset: int = 0,
    layout_variant: str = "rowmajor",
) -> QuantArtifactDocument:
    """Build one synthetic ``QuantArtifact`` (W4 or W8) with one variant."""
    if bits == 4:
        scheme = MAIN_W4_GROUP.with_overrides(
            group_size=group_size, label=f"synthetic_w4_g{group_size}"
        )
        layout_id = (
            LAYOUT_W4A16_HIFIRST_NK_V1
            if layout_variant == "hifirst"
            else LAYOUT_W4A16_ROWMAJOR_NK_V1
        )
    elif bits == 8:
        scheme = QuantScheme(
            bits=8,
            granularity=GRANULARITY_PER_GROUP,
            group_size=group_size,
            label=f"synthetic_w8_g{group_size}",
        )
        layout_id = LAYOUT_W8A16_ROWMAJOR_NK_V1
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported fixture bits: {bits}")

    values = synthetic_weight_values(rows, cols, offset=seed_offset)
    qt = quantize(values, scheme, shape=(rows, cols))
    model_identity = model or ModelIdentity(
        model_id="hqsb/synthetic-fixture",
        revision="synthetic-v1",
        architecture="synthetic",
        config_hash="0" * 64,
        tokenizer_hash="0" * 64,
        source_weight_sha256="0" * 64,
        model_root_sha256="0" * 64,
    )
    document = QuantArtifactDocument.from_quantized(
        qt,
        tensor_name=name,
        source_dtype="float16",
        source_sha256="0" * 64,
        model=model_identity,
        method="rtn",
        method_config_hash=scheme.scheme_hash(),
        calibration=CalibrationProvenance(kind="NONE", notes="RTN uses weight statistics only"),
        known_limitations=[
            "synthetic fixture for capability self-checks; not model evidence",
        ],
    )
    document.compatibility = [
        CompatibilityRecord(
            kernel_id=("hqsb.w4a16.triton" if bits == 4 else "hqsb.w8a16.triton"),
            provider="triton",
            layout_id=layout_id,
            target_arch=target_arch,
            abi_version="1",
            supported_bits=(bits,),
            supported_groups=(group_size,) if group_size else (),
            dtype="float16",
            workspace_bytes=0,
            fallback_policy="fail_closed",
            status="declared",
        )
    ]
    packed = pack_kernel_variant(
        qt.q,
        qt.scales,
        qt.zeros,
        qt.scheme,
        rows,
        cols,
        layout_id=layout_id,
        parent_canonical_hash=document.canonical_hash(),
    )
    document.add_variant(packed, target_arch=target_arch)
    return document


def save_golden_artifacts(root: str, *, target_arch: str = "sm_86") -> dict:
    """Write the W4/W8 synthetic golden artifacts; returns ``{name: path}``."""
    paths = {}
    for label, bits in (("w4", 4), ("w8", 8)):
        document = build_tiny_artifact(bits=bits, target_arch=target_arch)
        path = os.path.join(root, f"synthetic_rtn_{label}")
        document.save(path)
        paths[label] = path
    return paths


__all__ = [
    "build_tiny_artifact",
    "save_golden_artifacts",
    "synthetic_weight_values",
]
