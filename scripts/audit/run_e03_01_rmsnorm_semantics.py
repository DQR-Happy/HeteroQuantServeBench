#!/usr/bin/env python3
"""E03-01 — RMSNorm 数学语义、独立 Reference 与正确性边界.

Implements the ten protocol steps of
``docs/stage_experiments/details/S03/E03-01_rmsnorm_semantics_and_correctness.md``
and records every required field of §9 (seed/input hash, eps, shape/dtype,
max/mean/RMSE/cosine/L2rel, first mismatch, classification, pollution,
repeatability, negative cases, coverage matrix).

Design constraints honoured here
-------------------------------
* The candidate is only ever reached through the *forced-variant* C ABI
  (``ops.cuda_bridge``); no test may fall back to a default variant without
  recording it (protocol §5 step 5 / case G).
* The two oracles never call V0/V1/V2, the auto dispatcher, or any fused
  helper: the FP64 oracle is numpy-on-CPU, the FP32 oracle is a plain
  ``torch`` expression evaluated on the device.
* A third, analytic classification oracle predicts the IEEE class of each
  output element from the input classes alone (protocol §7).
* Tolerance is resolved from a pre-registered table (protocol §6); nothing is
  relaxed after seeing a result.

Subcommands::

    python3 scripts/audit/run_e03_01_rmsnorm_semantics.py collect   --output-dir <dir>
    python3 scripts/audit/run_e03_01_rmsnorm_semantics.py verify    --output-dir <dir>
    python3 scripts/audit/run_e03_01_rmsnorm_semantics.py summarize --output-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hqsb.benchmark import rmsnorm_correctness as rc  # noqa: E402
from ops import cuda_bridge  # noqa: E402
from ops.capability import detect_capabilities  # noqa: E402

EXPERIMENT_ID = "E03-01"
STAGE = "S03"
DEFAULT_OUTPUT_DIR = "docs/stage_experiments/S03/E03-01/raw"

#: Model artifact frozen by S00/S02 (fallback only; verified by path+sha256).
DEFAULT_MODEL_PATH = "/home/jetson/models/hqsb/Qwen3-1.7B"
#: S02 real-shape census produced by E02-02.
DEFAULT_S02_CENSUS = "docs/stage_experiments/S02/E02-02/raw_v2"

#: Sentinel written into the output buffer before every forced launch, so a
#: partial write or a "kernel never ran" case is visible.
OUTPUT_SENTINEL = 1234.0

#: Variant codes of the C ABI (see ops/cuda/rmsnorm/src/rmsnorm_c_api.cu).
VARIANT_NAME = {
    0: "auto",
    1: "reference",
    2: "v0_shared",
    3: "v1_warp_shuffle",
    4: "v2_vectorized",
}
VARIANT_CODE = {v: k for k, v in VARIANT_NAME.items()}
#: Frozen claim: which variants support which dtype (rmsnorm_dispatcher.cu).
SUPPORTED_VARIANTS = {
    "fp32": ("v0_shared", "v1_warp_shuffle", "v2_vectorized"),
    "fp16": ("v2_vectorized",),
}
UNSUPPORTED_VARIANTS = {
    "fp32": (),
    "fp16": ("v0_shared", "v1_warp_shuffle"),
}

#: Extra epsilon values probed by the epsilon-sensitivity sub-matrix (the model
#: epsilon is always added, so at least three distinct values are covered).
EXTRA_EPSILONS = (1.0e-5, 1.0e-2)

REPEAT_RUNS = 5
PASS2_SHUFFLE_SEED = 20260918


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _to_cuda(array: np.ndarray, dtype: str) -> "torch.Tensor":
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.to(device="cuda", dtype=torch_dtype)


# ══════════════════════════════════════════════════════════════════════
# step 1 — audit the handoff evidence
# ══════════════════════════════════════════════════════════════════════


def collect_environment() -> Dict[str, Any]:
    caps = detect_capabilities()
    info: Dict[str, Any] = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uname": " ".join(platform.uname()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
        "capabilities": caps.as_dict(),
        "cwd": str(_REPO_ROOT),
        "started_at_utc": _utc_now(),
    }
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        info["device_memory_free_bytes"] = int(free_b)
        info["device_memory_total_bytes"] = int(total_b)
    return info


def read_model_config() -> Dict[str, Any]:
    """Step 1: read the S00/S02 model revision and freeze epsilon from it."""
    path = Path(os.environ.get("HQSB_MODEL_PATH", DEFAULT_MODEL_PATH))
    config_path = path / "config.json"
    result: Dict[str, Any] = {
        "model_path": str(path),
        "config_path": str(config_path),
        "config_exists": config_path.is_file(),
        "config_sha256": _sha256_file(config_path),
    }
    if not config_path.is_file():
        result["epsilon_source"] = "fallback_default"
        result["rms_norm_eps"] = 1.0e-6
        result["epsilon_source_verified"] = False
        return result
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result.update(
        {
            "epsilon_source": "model config (config.json::rms_norm_eps)",
            "epsilon_source_verified": "rms_norm_eps" in config,
            "rms_norm_eps": float(config.get("rms_norm_eps", 1.0e-6)),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "torch_dtype": config.get("torch_dtype"),
        }
    )
    head_dim = result.get("head_dim") or (
        int(result["hidden_size"]) // int(result["num_attention_heads"])
        if result.get("hidden_size") and result.get("num_attention_heads")
        else None
    )
    result["head_dim_resolved"] = head_dim
    return result


def import_s02_shape_census(census_dir: Path) -> Dict[str, Any]:
    """Step 1/5.1-A: import the real RMSNorm shapes measured by E02-02."""
    out: Dict[str, Any] = {
        "census_dir": str(census_dir),
        "exists": census_dir.is_dir(),
        "runs": [],
        "rmsnorm_calls": [],
        "unique_signatures": [],
    }
    if not census_dir.is_dir():
        return out

    signatures: Dict[str, Dict[str, Any]] = {}
    for run_dir in sorted(census_dir.glob("run_*")):
        for census in sorted(run_dir.glob("census_*.json")):
            doc = json.loads(census.read_text(encoding="utf-8"))
            out["runs"].append(
                {
                    "file": str(census.relative_to(_REPO_ROOT)),
                    "workload": doc.get("workload"),
                    "token_parity": doc.get("instrumented", {}).get("token_parity"),
                }
            )
            for module in doc.get("modules", []):
                if module.get("module_type") != "Qwen3RMSNorm":
                    continue
                for shape in module.get("input_shapes", []):
                    entry = {
                        "module": module.get("module"),
                        "phase": module.get("phase"),
                        "call_count": module.get("call_count"),
                        "input_shape": shape,
                        "dtype": (module.get("input_dtypes") or [None])[0],
                        "stride": (module.get("input_strides") or [None])[0],
                        "layout": (module.get("input_layouts") or [None])[0],
                        "contiguous": module.get("input_contiguous"),
                        "workload": doc.get("workload"),
                        "run": run_dir.name,
                    }
                    out["rmsnorm_calls"].append(entry)
                    key = f"{module.get('module')}|{entry['phase']}|{shape}"
                    signatures[key] = entry
    out["unique_signatures"] = sorted(
        signatures.values(), key=lambda e: (str(e["module"]), str(e["phase"]))
    )
    return out


def freeze_operator_source() -> Dict[str, Any]:
    """Step 1: source/binary identity of the implementation under test."""
    sources = [
        "ops/cuda/rmsnorm/include/hqsb/rmsnorm.h",
        "ops/cuda/rmsnorm/src/rmsnorm_launchers.h",
        "ops/cuda/rmsnorm/src/rmsnorm_v0.cu",
        "ops/cuda/rmsnorm/src/rmsnorm_v1.cu",
        "ops/cuda/rmsnorm/src/rmsnorm_v2.cu",
        "ops/cuda/rmsnorm/src/rmsnorm_dispatcher.cu",
        "ops/cuda/rmsnorm/src/rmsnorm_reference.cu",
        "ops/cuda/rmsnorm/src/rmsnorm_c_api.cu",
        "ops/cuda_bridge.py",
        "ops/dispatcher.py",
        "hqsb/benchmark/rmsnorm_correctness.py",
        "configs/operators/rmsnorm_v0.json",
        "configs/operators/rmsnorm_v1.json",
        "configs/operators/rmsnorm_v2.json",
    ]
    hashes = {}
    for rel in sources:
        hashes[rel] = _sha256_file(_REPO_ROOT / rel)
    caps = detect_capabilities()
    lib_hash = _sha256_file(Path(caps.cuda_rmsnorm_lib)) if caps.cuda_rmsnorm_lib else None
    return {
        "source_sha256": hashes,
        "cuda_rmsnorm_lib": caps.cuda_rmsnorm_lib,
        "cuda_rmsnorm_lib_sha256": lib_hash,
        "note": (
            "the ELF sha256 is a per-run binary fingerprint; the linker build-id "
            "is not bit-reproducible across clean rebuilds (E00-04 §10), so the "
            "source sha256 set is the content-level identity"
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# step 2 — resolved OperatorSpec
# ══════════════════════════════════════════════════════════════════════


def build_resolved_spec(
    model_config: Mapping[str, Any],
    epsilon: float,
    environment: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve inheritance/defaults/CLI overrides into one canonical spec."""
    spec: Dict[str, Any] = {
        "schema": "hqsb.operator.resolved/v1",
        "c3_contract": {
            "name": "OperatorSpec",
            "schema_version": "1.0.0",
            "module": "hqsb.core.contracts.operator",
            "c3_expressible": [
                "input-level dtype (TensorSpec.dtype)",
                "logical shape (TensorSpec.shape)",
                "layout/stride map (TensorSpec.layout)",
                "stream (OperatorSpec.stream)",
                "fallback (OperatorSpec.fallback)",
                "deterministic flag",
                "single scalar tolerance (OperatorSpec.tolerance)",
            ],
            "c3_gap": [
                "alias policy (in-place / partial overlap) is not a C3 field",
                "accumulation dtype is not a C3 field",
                "epsilon (value/source/legal domain) is not a C3 field",
                "NaN/Inf/FastMath policy is not a C3 field",
                "per-dtype/per-shape-class tolerance table is not a C3 field "
                "(only one scalar tolerance exists)",
            ],
            "c3_gap_handling": (
                "E03-01 does NOT mutate the frozen C3 model (that would change "
                "the C3 schema digest already published in S01/E01-01 and break "
                "its schema_version gate). Instead the gaps above are resolved "
                "here into a canonical, hashed superset document, and are "
                "registered as an explicit handoff item for S04. The C3-visible "
                "subset is validated against OperatorSpec itself."
            ),
        },
        "name": "rmsnorm",
        "semantic_version": rc.SEMANTIC_VERSION,
        "equation": rc.EQUATION,
        "reduction_axis": rc.REDUCTION_AXIS,
        "epsilon": {
            "value": float(epsilon),
            "source": model_config.get("epsilon_source"),
            "source_verified": bool(model_config.get("epsilon_source_verified")),
            "position": rc.EPSILON_POSITION,
            "legal_domain": "finite and strictly > 0; NaN/Inf/non-positive rejected before launch",
        },
        "dtype": {
            "supported_input": ["float32", "float16"],
            "unsupported_input": {
                "bfloat16": "not compiled, not validated, and not claimed: the "
                            "C ABI rejects any dtype code other than 0/1",
                "float64": "not a device path of this operator",
                "int*": "not a device path of this operator",
            },
            "accumulation": rc.ACCUM_DTYPE,
            "output": {"fp32": "float32", "fp16": "float16"},
            "cast_policy": "no implicit cast: a dtype outside the declared set is rejected",
        },
        "shape": {
            "layout": "row-major, contiguous, 2-D (rows, hidden)",
            "stride": "contiguous only; a strided view is rejected instead of silently mis-read",
            "rows": {"min": 1, "note": "leading dimensions are flattened into rows"},
            "hidden": {
                "min": 1,
                "max": 2147483647,
                "note": "hidden must fit int32 because the device kernels index with int",
            },
            "out_shape": rc.OUT_SHAPE,
            "empty_tensor": (
                "H=0 or rows=0 is rejected before launch with cudaErrorInvalidValue "
                "(no grid=0 launch, no divide-by-zero)"
            ),
        },
        "weight": {
            "shape": "(hidden,)",
            "broadcast": rc.WEIGHT_BROADCAST,
            "length_validation": (
                "checked by the Python bridge (weight.numel() == hidden). The raw "
                "C ABI takes an untyped device pointer and therefore cannot see a "
                "length; that boundary is registered for E03-03/E03-07."
            ),
        },
        "alias": {
            "in_place_allowed": True,
            "rationale": (
                "rmsnorm.h documents `output ... (may alias input)`. The second "
                "pass is element-wise (out[i] = in[i]*inv*w[i]) and the reduction "
                "phase only reads, so aliasing is sound; verified explicitly in "
                "the in_place sub-matrix."
            ),
            "partial_overlap": "not supported and not claimed",
        },
        "alignment": {
            "vectorized_load": {
                "fp32": "float4 requires hidden %% 4 == 0 and a 16-B aligned row start",
                "fp16": "half2 requires hidden %% 2 == 0 and a 4-B aligned row start",
            },
            "tail": "non-multiple widths take a scalar tail path inside V2; the "
                    "variant identity is preserved (no variant switch)",
        },
        "stream": {
            "abi": "hqsb_rmsnorm_forward_ex_c(input, weight, output, rows, hidden, "
                   "epsilon, dtype, variant, cudaStream_t stream)",
            "legacy_abi": "hqsb_rmsnorm_forward_c(...) launches on the default stream",
            "implicit_global_sync": False,
            "note": "concurrency/serialisation evidence is owned by E03-06; E03-01 "
                    "only proves that an explicit stream can be named and is correct",
        },
        "workspace_bytes": 0,
        "supported_architecture": {
            "declared": ["sm_87"],
            "device_capability": environment.get("device_capability"),
            "cuda_rmsnorm_lib": source_identity.get("cuda_rmsnorm_lib"),
        },
        "deterministic": True,
        "tolerance_table": {
            "fp16": dict(rc.TOLERANCE_TABLE["fp16"]),
            "fp32_small_h": dict(rc.TOLERANCE_TABLE["fp32"]["small_h"]),
            "fp32_large_h": dict(rc.TOLERANCE_TABLE["fp32"]["large_h"]),
            "fp32_small_h_max": rc.TOLERANCE_TABLE["fp32"]["small_h_max"],
            "derivation": {
                "fp16": rc._tolerance_derivation()["fp16"],
                "fp32": rc._tolerance_derivation()["fp32"],
                "reference_budget": rc._tolerance_derivation()["reference_budget_factor"],
            },
            "l2_floor_abs": rc.L2_FLOOR_ABS,
            "metric_set": ["max_abs", "mean_abs", "RMSE", "cosine", "L2rel"],
        },
        "comparison_rule": (
            "abs_error <= atol(dtype, shape_class) OR "
            "abs_error <= rtol(dtype, shape_class) * abs(reference)"
        ),
        "special_value_policy": {
            "nan": "a NaN anywhere in a row propagates through the reduction and "
                   "makes every element of that row NaN",
            "posinf_neginf": "±Inf makes the square-sum +Inf, inv_rms = 0, the "
                             "finite lanes of that row become exactly 0 and the "
                             "±Inf lanes become Inf*0 = NaN",
            "fp32_square_overflow": "a finite x whose FP32 square-sum overflows "
                                    "gives inv_rms = 0 and an exactly-0 output; "
                                    "this is a declared divergence from the FP64 "
                                    "oracle, not a failure",
            "comparison": "per-element IEEE class is compared first; equal_nan is "
                          "never used to hide a mismatch",
            "fast_math": "not enabled; FTZ/denormal policy is the CUDA default "
                         "(no -use_fast_math in the CMake flags)",
        },
        "forced_variant": {
            "policy": "a forced variant either executes or is rejected with a "
                      "stable error; it never silently becomes another variant",
            "auto": "variant 0 resolves through rmsnorm_select_variant; the "
                    "resolved variant is not observable through the current C ABI "
                    "(handed to E03-08)",
        },
        "error_taxonomy": {
            "c_abi": {
                "0": "cudaSuccess",
                "1": "cudaErrorInvalidValue (null pointer / rows<1 / hidden<1 / "
                     "hidden>INT_MAX / invalid dtype code / invalid variant code / "
                     "kReference / unsupported dtype+variant / epsilon not finite "
                     "or <= 0) — always returned before any kernel launch",
            },
            "python_bridge_reason_codes": list(cuda_bridge.RMSNORM_REASON_CODES),
        },
    }
    spec["canonical_sha256"] = rc.sha256_hex(spec)
    return spec


def validate_c3_subset(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the C3-expressible subset against the real OperatorSpec model."""
    from hqsb.core.contracts import OperatorSpec, TensorSpec

    try:
        obj = OperatorSpec(
            name=spec["name"],
            semantic_version=spec["semantic_version"],
            inputs=[
                TensorSpec(name="x", dtype="float32|float16", shape=[-1, -1],
                           layout="contiguous"),
                TensorSpec(name="weight", dtype="float32|float16", shape=[-1],
                           layout="contiguous"),
            ],
            outputs=[
                TensorSpec(name="y", dtype="float32|float16", shape=[-1, -1],
                           layout="contiguous"),
            ],
            device="cuda",
            stream="stream-aware",
            workspace_bytes=0,
            deterministic=bool(spec["deterministic"]),
            tolerance=rc.TOLERANCE_TABLE["fp32"]["small_h"]["atol"],
            implementation="v2_vectorized",
            fallback="v1_warp_shuffle",
        )
        return {
            "valid": True,
            "schema_version": obj.schema_version,
            "serialized_sha256": rc.sha256_hex(json.loads(obj.model_dump_json())),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"valid": False, "schema_version": None, "error": repr(exc)}


# ══════════════════════════════════════════════════════════════════════
# step 5.1 — pre-registered shape plan
# ══════════════════════════════════════════════════════════════════════


def shape_plan() -> Dict[str, List[Dict[str, Any]]]:
    real = [
        {
            "shape_class": "real_hidden_prefill",
            "rows": 512,
            "hidden": 2048,
            "shape_provenance": "S02 E02-02: layers.*.input_layernorm / "
                                "post_attention_layernorm prefill [1,512,2048] fp16 "
                                "contiguous, call_count=1",
        },
        {
            "shape_class": "real_hidden_decode",
            "rows": 1,
            "hidden": 2048,
            "shape_provenance": "S02 E02-02: same modules, decode [1,1,2048] fp16 "
                                "contiguous, call_count=127",
        },
        {
            "shape_class": "real_q_norm_prefill",
            "rows": 8192,
            "hidden": 128,
            "shape_provenance": "S02 E02-02: layers.*.self_attn.q_norm prefill "
                                "[1,512,16,128] -> rows=512*16=8192, H=128",
        },
        {
            "shape_class": "real_k_norm_decode",
            "rows": 8,
            "hidden": 128,
            "shape_provenance": "S02 E02-02: layers.*.self_attn.k_norm decode "
                                "[1,1,8,128] -> rows=8, H=128",
        },
    ]
    design = [
        {"shape_class": "syn_h100", "rows": 1, "hidden": 100,
         "shape_provenance": "S03 list: hidden=100 (design-specified)"},
        {"shape_class": "syn_h128", "rows": 1, "hidden": 128,
         "shape_provenance": "S03 list: hidden=128 (design-specified); coincides "
                             "with the real head-dim norm width but is a synthetic "
                             "row count"},
        {"shape_class": "syn_h512", "rows": 1, "hidden": 512,
         "shape_provenance": "S03 list: hidden=512 (design-specified)"},
        {"shape_class": "syn_h2048", "rows": 1, "hidden": 2048,
         "shape_provenance": "S03 list: hidden=2048 (design-specified)"},
        {"shape_class": "syn_h6144", "rows": 1, "hidden": 6144,
         "shape_provenance": "SYNTHETIC_BOUNDARY: 6144 is the MLP intermediate "
                             "width of Qwen3-1.7B, NOT an RMSNorm width of this "
                             "model; recorded as a synthetic boundary, never as a "
                             "runtime shape"},
        {"shape_class": "syn_h8192", "rows": 1, "hidden": 8192,
         "shape_provenance": "SYNTHETIC_BOUNDARY: hidden=8192 from the S03 list; "
                             "not an RMSNorm width of Qwen3-1.7B"},
        {"shape_class": "syn_h100_r4", "rows": 4, "hidden": 100,
         "shape_provenance": "small batch x odd hidden"},
        {"shape_class": "syn_h512_r4", "rows": 4, "hidden": 512,
         "shape_provenance": "small batch"},
        {"shape_class": "syn_h2048_r4", "rows": 4, "hidden": 2048,
         "shape_provenance": "small batch"},
        {"shape_class": "syn_h2048_r16_prefill_flatten", "rows": 16, "hidden": 2048,
         "shape_provenance": "prefill flattened B*I; 16 is the S02 E02-04 max safe "
                             "batch for decode_heavy"},
        {"shape_class": "syn_h8192_r64_index_edge", "rows": 64, "hidden": 8192,
         "shape_provenance": "rows*hidden = 524288 element index arithmetic near "
                             "the check boundary without OOM"},
    ]
    boundary = [
        (1, 1), (1, 2), (3, 3), (2, 31), (2, 32), (2, 33), (2, 63), (2, 64),
        (2, 65), (1, 99), (1, 100), (1, 101), (1, 127), (1, 128), (1, 129),
        (1, 2047), (1, 2048), (1, 2049), (2, 101), (3, 129), (4, 2049),
    ]
    return {
        "real": real,
        "design": design,
        "boundary": [
            {
                "shape_class": f"bnd_h{hidden}_r{rows}",
                "rows": rows,
                "hidden": hidden,
                "shape_provenance": "S03 boundary shape",
            }
            for rows, hidden in boundary
        ],
    }


#: Full input-mode matrix (protocol §5.3) for the key shapes.
CORE_MODES = rc.INPUT_MODES
#: Reduced mode set for the boundary shapes (documented, not silently reduced).
BOUNDARY_MODES = ("random_normal", "zeros", "tiny")


def build_case_plan(
    plan: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the pre-registered case plan (one entry per forced execution)."""
    cases: List[Dict[str, Any]] = []
    for entry in plan["real"] + plan["design"]:
        for mode in CORE_MODES:
            for dtype in ("fp32", "fp16"):
                for variant in SUPPORTED_VARIANTS[dtype]:
                    cases.append(
                        _case(
                            entry, mode, dtype, variant,
                            group="full_mode",
                            shape_group="real" if entry in plan["real"] else "design",
                        )
                    )
    for entry in plan["boundary"]:
        for mode in BOUNDARY_MODES:
            for dtype in ("fp32", "fp16"):
                for variant in SUPPORTED_VARIANTS[dtype]:
                    cases.append(
                        _case(entry, mode, dtype, variant,
                              group="boundary", shape_group="boundary")
                    )
    return cases, {
        "full_mode_shapes": [
            dict(e, mode_set=list(CORE_MODES)) for e in plan["real"] + plan["design"]
        ],
        "boundary_shapes": [
            dict(e, mode_set=list(BOUNDARY_MODES)) for e in plan["boundary"]
        ],
        "mode_set_full": list(CORE_MODES),
        "mode_set_boundary": list(BOUNDARY_MODES),
        "variants_per_dtype": {k: list(v) for k, v in SUPPORTED_VARIANTS.items()},
    }


def _case(
    entry: Mapping[str, Any],
    mode: str,
    dtype: str,
    variant: str,
    *,
    group: str,
    shape_group: str,
    epsilon: Optional[float] = None,
    stream: str = "explicit",
    in_place: bool = False,
) -> Dict[str, Any]:
    rows = int(entry["rows"])
    hidden = int(entry["hidden"])
    eps_tag = "" if epsilon is None else f"eps{epsilon:g}"
    input_key = (
        f"{entry['shape_class']}|r{rows}|h{hidden}|{dtype}|{mode}|{eps_tag}"
    )
    suffix = f"{variant}|{stream}" + ("|inplace" if in_place else "")
    case = {
        "case_id": f"{input_key}|{suffix}",
        "input_key": input_key,
        "group": group,
        "shape_group": shape_group,
        "shape_class": entry["shape_class"],
        "shape_provenance": entry.get("shape_provenance"),
        "rows": rows,
        "hidden": hidden,
        "dtype": dtype,
        "mode": mode,
        "variant_requested": variant,
        "stream": stream,
        "in_place": in_place,
    }
    if epsilon is not None:
        case["epsilon_override"] = epsilon
    return case


# ══════════════════════════════════════════════════════════════════════
# oracle
# ══════════════════════════════════════════════════════════════════════


def framework_fp32_oracle(
    x: "torch.Tensor", w: "torch.Tensor", epsilon: float, rows: int, hidden: int
) -> np.ndarray:
    """Independent FP32 framework oracle built from plain torch tensor ops.

    Deliberately never calls V0/V1/V2, the CUDA bridge, the dispatcher or any
    fused helper: it re-expresses the same equation with ``to``/``*``/``mean``/
    ``rsqrt`` so a candidate bug cannot be shared.

    Operation order mirrors the kernels (``(x * inv) * w``) so that IEEE
    ``Inf * 0 -> NaN`` behaviour is comparable element by element.
    """
    dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
    xf = x.to(torch.float32)
    wf = w.to(torch.float32)
    square_mean = (xf * xf).mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(square_mean + float(epsilon))
    out = (xf * inverse_rms * wf).to(dtype)
    torch.cuda.synchronize()
    return out.detach().cpu().numpy().reshape(rows, hidden)


# ══════════════════════════════════════════════════════════════════════
# step 3–7 — execute one case
# ══════════════════════════════════════════════════════════════════════


def execute_case(
    case: Mapping[str, Any],
    *,
    epsilon: float,
    stream_obj: Optional["torch.cuda.Stream"],
    repeat: int = 1,
    light: bool = False,
) -> Dict[str, Any]:
    rows = int(case["rows"])
    hidden = int(case["hidden"])
    dtype = str(case["dtype"])
    mode = str(case["mode"])
    variant_name = str(case["variant_requested"])
    variant_code = VARIANT_CODE[variant_name]
    if mode not in rc.INPUT_MODES:
        raise ValueError(f"unknown input mode {mode!r} in case {case['case_id']!r}")

    seed = rc.derive_seed(case["input_key"])
    generated = rc.generate_case_arrays(mode, rows, hidden, dtype, seed)
    x_np = generated["x"]
    w_np = generated["w"]

    input_hash = rc.sha256_bytes(x_np.tobytes())
    weight_hash = rc.sha256_bytes(w_np.tobytes())

    x = _to_cuda(x_np, dtype)
    w = _to_cuda(w_np, dtype)

    numeric_dtype = np.float16 if dtype == "fp16" else np.float32
    finite_input = np.isfinite(x_np.astype(np.float64))
    record: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "case_id": case["case_id"],
        "input_key": case["input_key"],
        "group": case["group"],
        "shape_group": case["shape_group"],
        "shape_class": case["shape_class"],
        "shape_provenance": case.get("shape_provenance"),
        "rows": rows,
        "hidden": hidden,
        "dtype_in": dtype,
        "dtype_weight": dtype,
        "dtype_out": dtype,
        "accumulation": rc.ACCUM_DTYPE[dtype],
        "mode": mode,
        "epsilon": float(epsilon),
        "seed": seed,
        "input_hash": input_hash,
        "weight_hash": weight_hash,
        "input_special_values": rc.class_counts(rc.classify(x_np)),
        "injections": generated["injections"],
        "cast": generated["cast"],
        "finite_input_count": int(finite_input.sum()),
        "requested_variant": variant_name,
        "requested_variant_code": variant_code,
        # coverage-matrix aliases (kept explicit so aggregation cannot guess)
        "dtype": dtype,
        "variant": variant_name,
        "stream": case["stream"],
        "in_place": bool(case.get("in_place")),
    }

    # ── oracles (independent of each other and of the candidate) ──────
    divergence = rc.DECLARED_DIVERGENCE.get((dtype, mode))
    ref64 = None if light else rc.fp64_oracle(x_np, w_np, rows, hidden, epsilon)
    fw32 = None if light else (
        framework_fp32_oracle(x, w, epsilon, rows, hidden).reshape(-1).astype(np.float64)
    )

    # ── forced execution with a sentinel-filled output buffer ─────────
    out = torch.full_like(x, OUTPUT_SENTINEL)
    use_explicit_stream = case["stream"] == "explicit" and stream_obj is not None
    if use_explicit_stream:
        torch.cuda.synchronize()  # make the H2D copies visible to the new stream
    kernel_args: Dict[str, Any] = {
        "dtype": dtype,
        "variant": variant_code,
        "epsilon": float(epsilon),
        "out": out,
    }
    if use_explicit_stream:
        kernel_args["stream"] = stream_obj

    immediate_status: Optional[int] = None
    completion_status: Optional[int] = None
    api_reason: Optional[str] = None
    api_reason_code: Optional[str] = None
    launched = False
    try:
        if case.get("in_place"):
            # out aliases x on purpose: the API documents aliasing as legal.
            kernel_args["out"] = x
            out = x
        cuda_bridge.rmsnorm_forward(x, w, **kernel_args)
        launched = True
        immediate_status = 0
    except cuda_bridge.RmsNormContractError as exc:
        api_reason = str(exc)
        api_reason_code = exc.reason_code
    except RuntimeError as exc:
        api_reason = str(exc)
        api_reason_code = "CUDA_LAUNCH_ERROR"
    except Exception as exc:  # pragma: no cover - defensive
        api_reason = repr(exc)
        api_reason_code = "UNEXPECTED_ERROR"

    if launched:
        try:
            torch.cuda.synchronize()
            completion_status = 0
        except RuntimeError as exc:
            completion_status = 1
            api_reason = (api_reason or "") + f" | async: {exc}"

    repeat_hashes: List[str] = []
    if launched and repeat > 1:
        for _ in range(repeat - 1):
            scratch = torch.full_like(x, OUTPUT_SENTINEL)
            extra_args = dict(kernel_args)
            extra_args["out"] = scratch
            cuda_bridge.rmsnorm_forward(x, w, **extra_args)
            torch.cuda.synchronize()
            repeat_hashes.append(
                rc.sha256_bytes(
                    np.ascontiguousarray(scratch.detach().cpu().numpy()).tobytes()
                )
            )

    is_auto = variant_name == "auto"
    if not launched:
        actual_variant: Optional[str] = None
    elif is_auto:
        # The C ABI resolves variant 0 internally (rmsnorm_select_variant) and
        # does not report back which kernel ran: the resolved variant is NOT
        # observable from Python. Recording a guess would be a false claim, so
        # it is declared unobservable and handed to E03-08.
        actual_variant = None
    else:
        actual_variant = variant_name

    record.update(
        {
            "actual_variant": actual_variant,
            "actual_variant_observed": (
                "variant 0 (auto) resolved inside the C ABI; the resolved "
                "variant is not exposed by the current ABI (handoff E03-08)"
                if (launched and is_auto)
                else (
                    f"forced via C ABI variant code {variant_code}"
                    if launched
                    else "not launched"
                )
            ),
            "immediate_cuda_status": immediate_status,
            "completion_cuda_status": completion_status,
            "api_reason": api_reason,
            "api_reason_code": api_reason_code,
            "launched": launched,
        }
    )

    if not launched:
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        untouched = bool(
            torch.equal(
                out.detach().cpu(),
                torch.full_like(x, OUTPUT_SENTINEL).detach().cpu(),
            )
        )
        record.update(
            {
                "output_untouched_after_rejection": None if case.get("in_place") else untouched,
                "output_hash": None,
                "input_after_hash": rc.sha256_bytes(
                    x.detach().cpu().numpy().tobytes()
                ),
                "weight_after_hash": rc.sha256_bytes(
                    w.detach().cpu().numpy().tobytes()
                ),
                "checks": {
                    "launched": False,
                    "rejected_before_launch": True,
                },
                "status": "REJECTED",
                "failed_checks": [],
            }
        )
        return record

    # ── post-execution hashes (pollution check) ───────────────────────
    cand_np = out.detach().cpu().numpy().reshape(rows, hidden)
    output_hash = rc.sha256_bytes(np.ascontiguousarray(cand_np).tobytes())
    input_after_hash = rc.sha256_bytes(x.detach().cpu().numpy().tobytes())
    weight_after_hash = rc.sha256_bytes(w.detach().cpu().numpy().tobytes())

    if light:
        record.update(
            {
                "output_hash": output_hash,
                "input_after_hash": input_after_hash,
                "weight_after_hash": weight_after_hash,
                "checks": {"launched": True},
                "status": "PASS",
                "failed_checks": [],
            }
        )
        return record

    cand_cls = rc.classify(cand_np)
    exp_cls = rc.expected_classes(x_np, rows, hidden)
    class_mismatch = int(np.count_nonzero(cand_cls != exp_cls))

    finite_mask = np.isfinite(ref64) & np.isfinite(fw32) & np.isfinite(
        cand_np.astype(np.float64).reshape(-1)
    )

    tolerance = rc.tolerance_for(dtype, hidden)
    ref_budget = rc.reference_budget(dtype, hidden)
    reference_used = "fp32_framework" if divergence else "fp64_cpu_oracle"
    reference_array = fw32 if divergence else ref64

    metrics = rc.error_metrics(cand_np, reference_array, finite_mask)
    violations = rc.elementwise_violations(cand_np, reference_array, tolerance, finite_mask)
    agreement_metrics = rc.error_metrics(fw32, ref64, finite_mask)
    agreement_violations = rc.elementwise_violations(fw32, ref64, ref_budget, finite_mask)
    mismatch = rc.first_mismatch(
        candidate=cand_np,
        reference=reference_array,
        candidate_classes=cand_cls,
        expected_class_array=exp_cls,
        tolerance=tolerance,
        hidden=hidden,
    )

    zero_mask = rc.inf_row_finite_lanes_exact_zero(x_np, rows, hidden)
    zero_violations = int(
        np.count_nonzero(cand_np.astype(np.float64).reshape(-1)[zero_mask] != 0.0)
    )
    cosine_ok: Optional[bool] = None
    if metrics["cosine_applicable"]:
        cosine_ok = bool(metrics["cosine"] >= tolerance["cosine_min"])

    overflow_rows = None
    if mode == "overflow_probe":
        overflow_rows = int(rc.fp32_square_sum_overflow(x_np, rows, hidden).sum())

    shape_ok = tuple(cand_np.shape) == (rows, hidden)
    dtype_ok = bool(cand_np.dtype == numeric_dtype)
    device_ok = bool(out.is_cuda)
    input_unchanged = input_after_hash == input_hash
    weight_unchanged = weight_after_hash == weight_hash

    na_reasons: Dict[str, str] = {}
    checks: Dict[str, Optional[bool]] = {
        "launched": True,
        "immediate_cuda_status_ok": immediate_status == 0,
        "completion_cuda_status_ok": completion_status == 0,
        "output_shape_ok": shape_ok,
        "output_dtype_ok": dtype_ok,
        "output_device_ok": device_ok,
        "classification_matches_analytic_oracle": class_mismatch == 0,
        "numeric_within_tolerance": bool(violations["passed"]),
        "inf_row_finite_lanes_exactly_zero": zero_violations == 0,
        "cosine_within_min": cosine_ok,
        # in-place is a *declared legal* mode: out aliases x, so x must change.
        # It is therefore "not applicable", never a failed check.
        "input_unchanged": None if case.get("in_place") else input_unchanged,
        "weight_unchanged": weight_unchanged,
        "actual_variant_equals_requested": (
            None if is_auto else bool(actual_variant == variant_name)
        ),
        "oracle_agreement_within_reference_budget": (
            None if divergence else bool(agreement_violations["passed"])
        ),
        "l2rel_within_tolerance": (
            bool(metrics["l2rel"] <= tolerance["l2rel"])
            if metrics["applicable"]
            else None
        ),
    }
    if not metrics["applicable"]:
        na_reasons["numeric_within_tolerance"] = "no finite element pair"
        na_reasons["l2rel_within_tolerance"] = "no finite element pair"
    if cosine_ok is None:
        na_reasons["cosine_within_min"] = "cosine denominator is zero (NOT_APPLICABLE)"
    if divergence:
        na_reasons["oracle_agreement_within_reference_budget"] = divergence
        if overflow_rows is not None and overflow_rows != rows:
            # The divergence must be *confirmed* by the measurement, otherwise
            # the exemption would be hiding an unexplained disagreement.
            checks["declared_divergence_confirmed_by_fp32_overflow"] = (
                overflow_rows == rows
            )
    if is_auto:
        na_reasons["actual_variant_equals_requested"] = (
            "variant 0 (auto) is resolved inside the C ABI and the resolved "
            "variant is not reported back; E03-08 owns dispatcher observability"
        )
    if case.get("in_place"):
        na_reasons["input_unchanged"] = (
            "in-place contract: out aliases x, so x is intentionally overwritten"
        )

    verdict = rc.judge_case(checks, na_reasons)
    record.update(
        {
            "output_hash": output_hash,
            "input_after_hash": input_after_hash,
            "weight_after_hash": weight_after_hash,
            "class_counts_candidate": rc.class_counts(cand_cls),
            "class_counts_expected": rc.class_counts(exp_cls),
            "classification_mismatch_count": class_mismatch,
            "exact_zero_required_count": int(zero_mask.sum()),
            "exact_zero_violation_count": zero_violations,
            "reference_used": reference_used,
            "declared_divergence": divergence,
            "fp32_square_sum_overflow_rows": overflow_rows,
            "metrics": metrics,
            "numeric_violations": violations,
            "tolerance": tolerance,
            "oracle_agreement": {
                "reference_pair": "fp32_framework_vs_fp64_oracle",
                "tolerance": ref_budget,
                "metrics": None if divergence else agreement_metrics,
                "violations": None if divergence else agreement_violations,
                "applicable": divergence is None,
                "na_reason": divergence,
            },
            "first_mismatch": mismatch,
            "repeat_runs": repeat,
            "repeat_output_hashes": repeat_hashes,
            "repeat_bitwise_stable": (
                None if not repeat_hashes else all(h == output_hash for h in repeat_hashes)
            ),
            "checks": checks,
            "na_reasons": na_reasons,
            "status": verdict["status"],
            "failed_checks": verdict["failed_checks"],
        }
    )
    return record


# ══════════════════════════════════════════════════════════════════════
# step 8 — negative cases
# ══════════════════════════════════════════════════════════════════════


def _c_abi_probe(lib, *, kwargs_overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Call ``hqsb_rmsnorm_forward_ex_c`` directly with raw ABI arguments."""
    import ctypes

    x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    w = torch.randn(64, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, OUTPUT_SENTINEL)
    args = {
        "input": ctypes.c_void_p(x.data_ptr()),
        "weight": ctypes.c_void_p(w.data_ptr()),
        "output": ctypes.c_void_p(out.data_ptr()),
        "rows": 4,
        "hidden": 64,
        "epsilon": 1.0e-5,
        "dtype": 0,
        "variant": 4,
        "stream": ctypes.c_void_p(0),
    }
    args.update(kwargs_overrides)
    status = lib.hqsb_rmsnorm_forward_ex_c(
        args["input"], args["weight"], args["output"], args["rows"], args["hidden"],
        ctypes.c_float(args["epsilon"]), args["dtype"], args["variant"], args["stream"],
    )
    torch.cuda.synchronize()
    return {
        "returned_status": int(status),
        "output_untouched": bool(
            torch.equal(out.detach().cpu(), torch.full_like(x, OUTPUT_SENTINEL).detach().cpu())
        ),
    }


def run_negative_case(
    lib,
    negative: Mapping[str, Any],
    epsilon: float,
) -> Dict[str, Any]:
    """Execute one negative case through the bridge or the raw C ABI."""
    record: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "case_id": f"neg::{negative['case_id']}",
        "group": "negative",
        "contract_clause": negative["contract_clause"],
        "expected_outcome": negative["expected"],
        "how": negative["how"],
    }
    if negative["method"] == "bridge":
        try:
            outcome = negative["builder"](epsilon)
        except cuda_bridge.RmsNormContractError as exc:
            record.update(
                {
                    "observed_outcome": "rejected",
                    "reason_code": exc.reason_code,
                    "message": str(exc),
                    "launched": False,
                }
            )
        except RuntimeError as exc:
            record.update(
                {
                    "observed_outcome": "rejected",
                    "reason_code": "CUDA_LAUNCH_ERROR",
                    "message": str(exc),
                    "launched": False,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            record.update(
                {
                    "observed_outcome": "unexpected",
                    "reason_code": type(exc).__name__,
                    "message": repr(exc),
                    "launched": None,
                }
            )
        else:
            record.update(
                {
                    "observed_outcome": "accepted",
                    "reason_code": None,
                    "message": f"call returned {outcome!r}",
                    "launched": True,
                }
            )
    else:
        probe = _c_abi_probe(lib, kwargs_overrides=negative["abi_overrides"])
        record.update(
            {
                "observed_outcome": (
                    "rejected" if probe["returned_status"] != 0 else "accepted"
                ),
                "reason_code": f"cudaError={probe['returned_status']}",
                "message": negative["expected"],
                "launched": probe["returned_status"] == 0,
                "output_untouched": probe["output_untouched"],
            }
        )

    record["as_expected"] = record["observed_outcome"] == "rejected"
    record["status"] = "PASS" if record["as_expected"] else "FAIL"
    return record


def negative_case_plan() -> List[Dict[str, Any]]:
    """The 30+ invalid-input cases required by protocol §8 step 8."""
    return _negative_case_plan_impl()


def _negative_case_plan_impl() -> List[Dict[str, Any]]:
    def bridge(desc):
        def builder(epsilon):  # pragma: no cover - executed on device only
            import torch

            x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
            w = torch.randn(64, device="cuda", dtype=torch.float32)
            return desc(x, w, epsilon)
        return builder

    def make(case_id, clause, expected, builder):
        return {
            "case_id": case_id,
            "contract_clause": clause,
            "expected": expected,
            "how": "ops.cuda_bridge",
            "method": "bridge",
            "method_builder": "bridge",
            "builder": builder,
        }

    def abi(case_id, clause, expected, overrides):
        return {
            "case_id": case_id,
            "contract_clause": clause,
            "expected": expected,
            "how": "raw C ABI (hqsb_rmsnorm_forward_ex_c)",
            "method": "c_abi",
            "method_builder": "n/a",
            "abi_overrides": overrides,
        }

    plan: List[Dict[str, Any]] = []

    plan.append(make(
        "rows_zero", "rows >= 1", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            torch.empty(0, 64, device="cuda", dtype=torch.float32), w, epsilon=e)),
    ))
    plan.append(make(
        "hidden_zero", "hidden >= 1", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            torch.empty(4, 0, device="cuda", dtype=torch.float32),
            torch.empty(0, device="cuda", dtype=torch.float32), epsilon=e)),
    ))
    plan.append(make(
        "weight_length_mismatch", "weight length == hidden", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            x, torch.randn(56, device="cuda", dtype=torch.float32), epsilon=e)),
    ))
    plan.append(make(
        "dtype_string_unknown", "dtype in {fp32, fp16} (no silent cast)",
        "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, dtype="bfloat16",
                                                          epsilon=e)),
    ))
    plan.append(make(
        "dtype_tensor_mismatch", "declared dtype must match tensor dtype",
        "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, dtype="fp16",
                                                          epsilon=e)),
    ))
    plan.append(make(
        "x_not_2d", "x must be 2-D (rows, hidden)", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            torch.randn(64, device="cuda", dtype=torch.float32), w, epsilon=e)),
    ))
    plan.append(make(
        "weight_not_1d", "weight must be 1-D", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            x, torch.randn(1, 64, device="cuda", dtype=torch.float32), epsilon=e)),
    ))
    plan.append(make(
        "x_non_contiguous", "contiguous row-major storage", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            torch.randn(64, 8, device="cuda", dtype=torch.float32).t(), w, epsilon=e)),
    ))
    plan.append(make(
        "weight_non_contiguous", "contiguous row-major storage", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            x, torch.randn(128, device="cuda", dtype=torch.float32)[::2], epsilon=e)),
    ))
    plan.append(make(
        "x_on_cpu", "device must be CUDA", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            torch.randn(4, 64), w, epsilon=e)),
    ))
    plan.append(make(
        "epsilon_zero", "epsilon finite and > 0", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, epsilon=0.0)),
    ))
    plan.append(make(
        "epsilon_negative", "epsilon finite and > 0", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, epsilon=-1e-5)),
    ))
    plan.append(make(
        "epsilon_nan", "epsilon finite and > 0", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, epsilon=float("nan"))),
    ))
    plan.append(make(
        "epsilon_posinf", "epsilon finite and > 0", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, epsilon=float("inf"))),
    ))
    plan.append(make(
        "out_shape_mismatch", "out must match x", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            x, w, epsilon=e,
            out=torch.empty(4, 32, device="cuda", dtype=torch.float32))),
    ))
    plan.append(make(
        "variant_unknown_name", "variant in the frozen set", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, variant="v9_magic",
                                                          epsilon=e)),
    ))
    plan.append(make(
        "variant_unknown_code", "variant in the frozen set", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, variant=9, epsilon=e)),
    ))
    plan.append(make(
        "variant_reference_forced", "kReference is not a device implementation",
        "rejected before launch (no silent switch to a GPU variant)",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(x, w, variant="reference",
                                                          epsilon=e)),
    ))
    plan.append(make(
        "fp16_with_v0", "V0 is FP32-only", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            x.to(torch.float16), w.to(torch.float16), dtype="fp16", variant="v0_shared",
            epsilon=e)),
    ))
    plan.append(make(
        "fp16_with_v1", "V1 is FP32-only", "rejected before launch",
        bridge(lambda x, w, e: cuda_bridge.rmsnorm_forward(
            x.to(torch.float16), w.to(torch.float16), dtype="fp16",
            variant="v1_warp_shuffle", epsilon=e)),
    ))

    plan.append(abi("abi_null_input", "input must be non-null",
                    "cudaErrorInvalidValue (1)",
                    {"input": __import__("ctypes").c_void_p(None)}))
    plan.append(abi("abi_null_weight", "weight must be non-null",
                    "cudaErrorInvalidValue (1)",
                    {"weight": __import__("ctypes").c_void_p(None)}))
    plan.append(abi("abi_null_output", "output must be non-null",
                    "cudaErrorInvalidValue (1)",
                    {"output": __import__("ctypes").c_void_p(None)}))
    plan.append(abi("abi_dtype_unknown", "dtype code must be 0 or 1",
                    "cudaErrorInvalidValue (1)", {"dtype": 7}))
    plan.append(abi("abi_dtype_bfloat16", "bfloat16 is not a claimed device dtype",
                    "cudaErrorInvalidValue (1)", {"dtype": 2}))
    plan.append(abi("abi_variant_unknown", "variant code must be in {0,2,3,4}",
                    "cudaErrorInvalidValue (1)", {"variant": 9}))
    plan.append(abi("abi_variant_reference", "variant 1 (reference) is CPU-only",
                    "cudaErrorInvalidValue (1)", {"variant": 1}))
    plan.append(abi("abi_epsilon_zero", "epsilon must be > 0",
                    "cudaErrorInvalidValue (1)", {"epsilon": 0.0}))
    plan.append(abi("abi_epsilon_nan", "epsilon must be finite",
                    "cudaErrorInvalidValue (1)", {"epsilon": float("nan")}))
    plan.append(abi("abi_epsilon_inf", "epsilon must be finite",
                    "cudaErrorInvalidValue (1)", {"epsilon": float("inf")}))
    plan.append(abi("abi_rows_zero", "rows >= 1",
                    "cudaErrorInvalidValue (1)", {"rows": 0}))
    plan.append(abi("abi_hidden_zero", "hidden >= 1",
                    "cudaErrorInvalidValue (1)", {"hidden": 0}))
    plan.append(abi("abi_fp16_with_v0", "V0 cannot consume fp16",
                    "cudaErrorInvalidValue (1)", {"dtype": 1, "variant": 2}))
    plan.append(abi("abi_hidden_exceeds_int_max", "hidden must fit int32",
                    "cudaErrorInvalidValue (1)", {"hidden": 2 ** 31}))
    return plan


NOT_EXPRESSIBLE_BOUNDARIES = [
    {
        "item": "weight length mismatch at the raw C ABI",
        "why_not_testable": (
            "the C ABI receives an untyped device pointer (void*), so a length "
            "cannot be observed there; the check exists only in the Python bridge"
        ),
        "covered_at": "ops.cuda_bridge (WEIGHT_SHAPE_MISMATCH) and neg::weight_length_mismatch",
        "handoff": "E03-03 / E03-07 (API validation gaps)",
    },
    {
        "item": "workspace missing or too small",
        "why_not_testable": (
            "the resolved spec declares workspace_bytes = 0: no workspace is "
            "requested, so there is nothing to under-allocate"
        ),
        "covered_at": "resolved_spec.workspace_bytes == 0",
        "handoff": "E03-07 (lifetime/allocation) if a workspace is ever added",
    },
]


# ══════════════════════════════════════════════════════════════════════
# collect
# ══════════════════════════════════════════════════════════════════════


def _supported_case_group(cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(c) for c in cases]


def main_collect(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"run_{int(time.time() * 1000)}"
    environment = collect_environment()
    model_config = read_model_config()
    census = import_s02_shape_census(
        Path(args.s02_census) if Path(args.s02_census).is_absolute()
        else _REPO_ROOT / args.s02_census
    )
    source_identity = freeze_operator_source()
    epsilon = float(model_config["rms_norm_eps"])

    spec = build_resolved_spec(model_config, epsilon, environment, source_identity)
    c3_check = validate_c3_subset(spec)

    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "environment": environment,
        "model_config": model_config,
        "s02_shape_census": {
            "census_dir": census["census_dir"],
            "exists": census["exists"],
            "runs_imported": len(census["runs"]),
            "unique_rmsnorm_signatures": census["unique_signatures"],
        },
        "operator_source": source_identity,
        "resolved_spec_sha256": spec["canonical_sha256"],
        "c3_subset_validation": c3_check,
        "epsilon_used": epsilon,
    }
    _write_json(out_dir / "provenance.json", provenance)
    _write_json(out_dir / "resolved_spec.json", spec)

    plan = shape_plan()
    cases, matrix_plan = build_case_plan(plan)
    _write_json(out_dir / "case_plan.json", matrix_plan)

    main_cases = _supported_case_group(cases)
    # second pass: same matrix in a shuffled order (order-dependence probe)
    pass2_order = list(main_cases)
    random.Random(PASS2_SHUFFLE_SEED).shuffle(pass2_order)

    repeat_pick = sorted({c["case_id"]: c for c in main_cases}.values(),
                         key=lambda c: c["case_id"])
    step = max(1, len(repeat_pick) // 24)
    repeat_cases = repeat_pick[::step][:24]

    eps_cases: List[Dict[str, Any]] = []
    eps_values = tuple(sorted({epsilon, *EXTRA_EPSILONS}))
    for entry in [plan["real"][1], plan["design"][5], plan["design"][2], plan["design"][7]]:
        for mode in ("random_normal", "zeros"):
            for dtype in ("fp32", "fp16"):
                for eps in eps_values:
                    for variant in SUPPORTED_VARIANTS[dtype]:
                        eps_cases.append(
                            _case(entry, mode, dtype, variant, group="epsilon_sensitivity",
                                  shape_group="design", epsilon=eps)
                        )
    in_place_cases: List[Dict[str, Any]] = []
    for entry in [plan["design"][2], plan["design"][7], plan["real"][0]]:
        for dtype in ("fp32", "fp16"):
            for variant in SUPPORTED_VARIANTS[dtype]:
                in_place_cases.append(
                    _case(entry, "random_normal", dtype, variant, group="in_place",
                          shape_group="design", in_place=True)
                )
    auto_cases: List[Dict[str, Any]] = []
    for entry in (plan["design"][2], plan["design"][3]):
        for mode in ("random_normal", "zeros"):
            for dtype in ("fp32", "fp16"):
                auto_cases.append(
                    _case(entry, mode, dtype, "auto", group="auto_routing",
                          shape_group="design")
                )

    unsupported_records: List[Dict[str, Any]] = []
    for entry in (plan["design"][2], plan["design"][1]):
        for dtype, variants in UNSUPPORTED_VARIANTS.items():
            for variant in variants:
                case = _case(entry, "random_normal", dtype, variant,
                             group="unsupported", shape_group="design")
                unsupported_records.append(
                    run_unsupported_case(case, epsilon=epsilon)
                )

    lib = cuda_bridge._bridge._ensure_loaded()  # noqa: SLF001 - forced ABI probes
    import ctypes

    lib.hqsb_rmsnorm_forward_ex_c.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.hqsb_rmsnorm_forward_ex_c.restype = ctypes.c_int

    stream_obj = torch.cuda.Stream()

    print(f"[{EXPERIMENT_ID}] collect run_id={run_id}")
    print(f"[{EXPERIMENT_ID}] lib={source_identity['cuda_rmsnorm_lib']}")
    print(f"[{EXPERIMENT_ID}] epsilon={epsilon} source={model_config['epsilon_source']}")
    print(f"[{EXPERIMENT_ID}] main cases={len(main_cases)} "
          f"eps={len(eps_cases)} in_place={len(in_place_cases)} auto={len(auto_cases)}")

    _write_json(out_dir / "expected_case_ids.json", {
        "group": "full_mode+boundary (claimed domain)",
        "count": len(main_cases),
        "case_ids": [c["case_id"] for c in main_cases],
    })

    main_records = _execute_many(main_cases, epsilon=epsilon, stream_obj=stream_obj,
                                label="main")
    _write_jsonl(out_dir / "cases.jsonl", main_records)

    pass2_records = _execute_many(
        pass2_order, epsilon=epsilon, stream_obj=stream_obj, label="pass2",
        hash_only=True, light=True,
    )
    pass2_by_id = {r["case_id"]: r for r in pass2_records}
    pass2_compare = []
    for rec in main_records:
        other = pass2_by_id.get(rec["case_id"])
        pass2_compare.append(
            {
                "case_id": rec["case_id"],
                "main_output_hash": rec.get("output_hash"),
                "pass2_output_hash": (other or {}).get("output_hash"),
                "match": bool(other and rec.get("output_hash") == other.get("output_hash")),
            }
        )
    _write_json(out_dir / "order_independence.json", {
        "shuffle_seed": PASS2_SHUFFLE_SEED,
        "compared": len(pass2_compare),
        "mismatches": [r for r in pass2_compare if not r["match"]],
        "all_match": all(r["match"] for r in pass2_compare),
    })

    repeat_records = _execute_many(repeat_cases, epsilon=epsilon,
                                   stream_obj=stream_obj, label="repeat",
                                   repeat=REPEAT_RUNS)
    _write_jsonl(out_dir / "repeatability.jsonl", repeat_records)

    eps_records = _execute_many(eps_cases, epsilon=epsilon, stream_obj=stream_obj,
                                label="epsilon", use_case_epsilon=True)
    _write_jsonl(out_dir / "epsilon_sensitivity.jsonl", eps_records)

    in_place_records = _execute_many(in_place_cases, epsilon=epsilon,
                                     stream_obj=stream_obj, label="in_place")
    _write_jsonl(out_dir / "in_place.jsonl", in_place_records)

    auto_records = _execute_many(auto_cases, epsilon=epsilon, stream_obj=stream_obj,
                                 label="auto")
    _write_jsonl(out_dir / "auto_routing.jsonl", auto_records)

    _write_jsonl(out_dir / "unsupported_combinations.jsonl", unsupported_records)

    negative_records = [
        run_negative_case(lib, neg, epsilon) for neg in negative_case_plan()
    ]
    _write_jsonl(out_dir / "negative_cases.jsonl", negative_records)

    _write_json(out_dir / "not_expressible_boundaries.json", NOT_EXPRESSIBLE_BOUNDARIES)

    historical = run_historical_regression(epsilon=epsilon, stream_obj=stream_obj)
    _write_json(out_dir / "historical_regression.json", historical)

    coverage = rc.coverage_matrix(main_records)
    coverage["epsilon_sensitivity"] = rc.coverage_matrix(eps_records)
    coverage["in_place"] = rc.coverage_matrix(in_place_records)
    coverage["auto_routing"] = rc.coverage_matrix(auto_records)
    _write_json(out_dir / "coverage_matrix.json", coverage)

    summary = {
        "run_id": run_id,
        "main_cases": len(main_records),
        "main_pass": sum(1 for r in main_records if r["status"] == "PASS"),
        "main_fail": sum(1 for r in main_records if r["status"] == "FAIL"),
        "repeat_cases": len(repeat_records),
        "orders_compared": len(pass2_compare),
        "negative_cases": len(negative_records),
        "negative_pass": sum(1 for r in negative_records if r["status"] == "PASS"),
        "ended_at_utc": _utc_now(),
    }
    _write_json(out_dir / "collect_summary.json", summary)
    print(f"[{EXPERIMENT_ID}] main {summary['main_pass']}/{summary['main_cases']} PASS")
    print(f"[{EXPERIMENT_ID}] negative {summary['negative_pass']}/{summary['negative_cases']} PASS")
    return 0


def _execute_many(
    cases: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
    stream_obj: "torch.cuda.Stream",
    label: str,
    repeat: int = 1,
    hash_only: bool = False,
    light: bool = False,
    use_case_epsilon: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_epsilon = float(case.get("epsilon_override", epsilon)) if use_case_epsilon \
            else epsilon
        rec = execute_case(case, epsilon=case_epsilon, stream_obj=stream_obj,
                          repeat=repeat, light=light)
        if hash_only:
            rec = {
                "case_id": rec["case_id"],
                "output_hash": rec.get("output_hash"),
                "status": rec["status"],
            }
        records.append(rec)
        if (index + 1) % 200 == 0:
            print(f"[{EXPERIMENT_ID}]   {label}: {index + 1}/{len(cases)}")
    return records


def run_unsupported_case(case: Mapping[str, Any], *, epsilon: float) -> Dict[str, Any]:
    """A (dtype, variant) combination the contract does not claim must be rejected."""
    rows, hidden = int(case["rows"]), int(case["hidden"])
    dtype = str(case["dtype"])
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    x = torch.randn(rows, hidden, device="cuda", dtype=torch_dtype)
    w = torch.randn(hidden, device="cuda", dtype=torch_dtype)
    out = torch.full_like(x, OUTPUT_SENTINEL)
    record: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "case_id": f"unsupported::{case['case_id']}",
        "group": "unsupported",
        "dtype": dtype,
        "variant_requested": case["variant_requested"],
        "hidden": hidden,
        "rows": rows,
        "contract_clause": "only the claimed (dtype, variant) pairs may execute",
        "expected_outcome": "rejected before launch, no silent variant switch",
    }
    try:
        cuda_bridge.rmsnorm_forward(
            x, w, dtype=dtype, variant=VARIANT_CODE[case["variant_requested"]],
            epsilon=epsilon, out=out,
        )
        torch.cuda.synchronize()
        record.update(
            {
                "observed_outcome": "accepted",
                "reason_code": None,
                "status": "FAIL",
                "output_untouched": None,
            }
        )
    except (cuda_bridge.RmsNormContractError, RuntimeError) as exc:
        torch.cuda.synchronize()
        untouched = bool(
            torch.equal(out.detach().cpu(),
                        torch.full_like(x, OUTPUT_SENTINEL).detach().cpu())
        )
        record.update(
            {
                "observed_outcome": "rejected",
                "reason_code": getattr(exc, "reason_code", "CUDA_LAUNCH_ERROR"),
                "message": str(exc),
                "status": "PASS",
                "output_untouched": untouched,
            }
        )
    return record


def run_historical_regression(
    *, epsilon: float, stream_obj: "torch.cuda.Stream"
) -> Dict[str, Any]:
    """Step 9: re-run the E00-04 smoke shapes against the current binary/C3.

    Only 'same order of magnitude / different' is reported; the historical run
    is never mixed into the S03 statistics.
    """
    historical = [
        (1, 1, "fp32", "random_normal"),
        (16, 2048, "fp32", "random_normal"),
        (8, 101, "fp32", "random_normal"),
        (512, 1024, "fp32", "random_normal"),
        (32, 2048, "fp16", "random_normal"),
        (8, 3, "fp16", "random_normal"),
        (128, 256, "fp32", "zeros"),
        (128, 256, "fp32", "tiny"),
        (128, 256, "fp32", "large_safe"),
        (32, 256, "fp16", "zeros"),
    ]
    rows_out = []
    for rows, hidden, dtype, mode in historical:
        entry = {
            "shape_class": f"e00_04_h{hidden}_r{rows}",
            "rows": rows,
            "hidden": hidden,
            "shape_provenance": "E00-04 historical smoke case (re-bound to the "
                                "current binary, environment and C3)",
        }
        for variant in SUPPORTED_VARIANTS[dtype]:
            case = _case(entry, mode, dtype, variant,
                         group="historical_regression", shape_group="historical")
            rec = execute_case(case, epsilon=epsilon, stream_obj=stream_obj)
            rows_out.append(
                {
                    "case_id": rec["case_id"],
                    "dtype": dtype,
                    "rows": rows,
                    "hidden": hidden,
                    "mode": mode,
                    "variant": variant,
                    "max_abs": (rec.get("metrics") or {}).get("max_abs"),
                    "rmse": (rec.get("metrics") or {}).get("rmse"),
                    "status": rec["status"],
                }
            )
    return {
        "note": "E00-04 covered fp32 5e-4 / fp16 2e-2 with max_abs <= 9.537e-07 "
                "(fp32) and 9.766e-04 (fp16); these numbers are a pilot hint for "
                "the order of magnitude only and are NOT copied into the S03 "
                "tolerance table",
        "cases": rows_out,
    }


# ══════════════════════════════════════════════════════════════════════
# verify
# ══════════════════════════════════════════════════════════════════════


def main_verify(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir

    provenance = _load(out_dir / "provenance.json")
    spec = _load(out_dir / "resolved_spec.json")
    plan = _load(out_dir / "case_plan.json")
    coverage = _load(out_dir / "coverage_matrix.json")
    order = _load(out_dir / "order_independence.json")
    cases = _read_jsonl(out_dir / "cases.jsonl")
    repeats = _read_jsonl(out_dir / "repeatability.jsonl")
    negatives = _read_jsonl(out_dir / "negative_cases.jsonl")
    unsupported = _read_jsonl(out_dir / "unsupported_combinations.jsonl")
    in_place = _read_jsonl(out_dir / "in_place.jsonl")
    eps_cases = _read_jsonl(out_dir / "epsilon_sensitivity.jsonl")
    auto_cases = _read_jsonl(out_dir / "auto_routing.jsonl")

    census = provenance.get("s02_shape_census", {})
    conditions: Dict[str, Any] = {}

    conditions["s02_runtime_shapes_available"] = bool(census.get("exists")) and bool(
        census.get("unique_rmsnorm_signatures")
    )
    conditions["epsilon_from_model_config"] = bool(
        provenance.get("model_config", {}).get("epsilon_source_verified")
    )
    conditions["resolved_spec_complete_and_hashed"] = bool(
        spec.get("canonical_sha256")
    ) and bool(provenance.get("c3_subset_validation", {}).get("valid"))
    conditions["binary_and_source_identity_bound"] = bool(
        provenance.get("operator_source", {}).get("cuda_rmsnorm_lib_sha256")
    ) and all(
        v for v in provenance.get("operator_source", {}).get("source_sha256", {}).values()
    )

    def _all_pass(records: Iterable[Mapping[str, Any]]) -> bool:
        recs = list(records)
        return bool(recs) and all(r.get("status") == "PASS" for r in recs)

    main_failures = [r for r in cases if r.get("status") == "FAIL"]
    conditions["all_planned_cases_pass"] = len(main_failures) == 0 and len(cases) > 0

    expected = _load(out_dir / "expected_case_ids.json")
    expected_ids = set(expected.get("case_ids", []))
    actual_ids = [r["case_id"] for r in cases]
    conditions["no_not_run_in_claimed_domain"] = (
        bool(expected_ids)
        and set(actual_ids) == expected_ids
        and len(actual_ids) == len(set(actual_ids))
    )
    conditions["every_case_launched_forced_variant"] = all(
        r.get("launched") and r.get("actual_variant") == r.get("requested_variant")
        for r in cases
    )
    conditions["no_extra_nan_or_inf"] = all(
        r.get("classification_mismatch_count") == 0 for r in cases
    )
    conditions["no_input_or_weight_mutation"] = all(
        (r.get("input_after_hash") == r.get("input_hash"))
        and (r.get("weight_after_hash") == r.get("weight_hash"))
        for r in cases
    )
    conditions["in_place_contract_verified"] = _all_pass(in_place) and len(in_place) > 0
    def _oracle_agreement_ok(rec: Mapping[str, Any]) -> bool:
        agreement = rec.get("oracle_agreement") or {}
        if agreement.get("applicable") is False:
            # declared-divergence cell: pre-registered as not comparable, and
            # its own numeric gate runs against the FP32 framework reference
            return True
        violations = agreement.get("violations") or {}
        return violations.get("passed") is True

    conditions["oracles_agree_within_reference_budget"] = all(
        _oracle_agreement_ok(r) for r in cases
    )
    conditions["first_mismatch_recorded_for_every_failure"] = all(
        r.get("first_mismatch") is not None for r in main_failures
    )
    conditions["repeatability_bitwise_stable"] = bool(repeats) and all(
        r.get("repeat_bitwise_stable") is True for r in repeats
    )
    conditions["order_independent"] = bool(order.get("all_match"))

    conditions["all_modes_covered"] = set(
        coverage.get("per_axis", {}).get("mode", {}).keys()
    ) >= set(rc.INPUT_MODES)
    conditions["all_boundary_shapes_covered"] = set(
        s["shape_class"] for s in plan.get("boundary_shapes", [])
    ) <= set(coverage.get("per_axis", {}).get("shape_class", {}).keys())
    conditions["all_real_shapes_covered"] = set(
        s["shape_class"] for s in plan.get("full_mode_shapes", [])
    ) <= set(coverage.get("per_axis", {}).get("shape_class", {}).keys())
    conditions["epsilon_sensitivity_covered"] = (
        len(eps_cases) > 0 and _all_pass(eps_cases)
        and len({r.get("epsilon") for r in eps_cases}) >= 3
    )
    conditions["negative_cases_rejected_before_launch"] = (
        len(negatives) >= 30 and _all_pass(negatives)
    )
    conditions["unsupported_combinations_rejected"] = _all_pass(unsupported)
    conditions["auto_routing_correct_and_actual_variant_unknown"] = (
        _all_pass(auto_cases)
    )

    overall = "PASS" if all(conditions.values()) else "FAIL"
    if not conditions["s02_runtime_shapes_available"] or not conditions[
        "epsilon_from_model_config"
    ]:
        overall = "BLOCKED"

    verdict = {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "run_id": provenance.get("run_id"),
        "git_commit": provenance.get("environment", {}).get("git_commit"),
        "git_dirty": provenance.get("environment", {}).get("git_dirty"),
        "resolved_spec_sha256": spec.get("canonical_sha256"),
        "conditions": conditions,
        "failed_conditions": sorted(k for k, v in conditions.items() if not v),
        "main_failures": [r["case_id"] for r in main_failures][:20],
        "overall": overall,
        "verified_at_utc": _utc_now(),
    }
    _write_json(out_dir / "verdict.json", verdict)
    print(f"[{EXPERIMENT_ID}] verify: {sum(conditions.values())}/{len(conditions)} "
          f"conditions pass -> {overall}")
    return 0 if overall == "PASS" else 1


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════
# summarize
# ══════════════════════════════════════════════════════════════════════


def main_summarize(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir

    cases = _read_jsonl(out_dir / "cases.jsonl")
    negatives = _read_jsonl(out_dir / "negative_cases.jsonl")
    unsupported = _read_jsonl(out_dir / "unsupported_combinations.jsonl")
    in_place = _read_jsonl(out_dir / "in_place.jsonl")
    eps_cases = _read_jsonl(out_dir / "epsilon_sensitivity.jsonl")
    auto_cases = _read_jsonl(out_dir / "auto_routing.jsonl")
    repeats = _read_jsonl(out_dir / "repeatability.jsonl")
    spec = _load(out_dir / "resolved_spec.json")
    verdict = _load(out_dir / "verdict.json")
    coverage = _load(out_dir / "coverage_matrix.json")
    order = _load(out_dir / "order_independence.json")
    historical = _load(out_dir / "historical_regression.json")
    provenance = _load(out_dir / "provenance.json")

    def _status_counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
        recs = list(records)
        return {
            "total": len(recs),
            "pass": sum(1 for r in recs if r.get("status") == "PASS"),
            "fail": sum(1 for r in recs if r.get("status") == "FAIL"),
        }

    def _worst(records: Iterable[Mapping[str, Any]], dtype: str) -> Dict[str, Any]:
        rows = [
            r for r in records
            if r.get("dtype_in") == dtype and (r.get("metrics") or {}).get("applicable")
        ]
        if not rows:
            return {}
        return {
            "cases": len(rows),
            "max_abs_max": max(r["metrics"]["max_abs"] for r in rows),
            "mean_abs_max": max(r["metrics"]["mean_abs"] for r in rows),
            "rmse_max": max(r["metrics"]["rmse"] for r in rows),
            "l2rel_max": max(r["metrics"]["l2rel"] for r in rows),
            "cosine_min": min(
                (r["metrics"]["cosine"] for r in rows
                 if r["metrics"]["cosine"] is not None),
                default=None,
            ),
            "tolerance": rc.tolerance_for(dtype, 2048),
        }

    def _headroom(records: Iterable[Mapping[str, Any]], dtype: str) -> Dict[str, Any]:
        """Worst ``|error| / allowed`` seen for one dtype (gate closeness)."""
        scored = []
        for rec in records:
            if rec.get("dtype_in") != dtype:
                continue
            ratio = (rec.get("numeric_violations") or {}).get("max_allowed_ratio")
            if ratio is not None:
                scored.append((ratio, rec))
        if not scored:
            return {}
        ratio, rec = max(scored, key=lambda item: item[0])
        index = (rec.get("numeric_violations") or {}).get("max_allowed_ratio_index")
        row = col = None
        if index is not None:
            row, col = divmod(int(index), int(rec["hidden"]))
        return {
            "cases_scored": len(scored),
            "worst_ratio": ratio,
            "worst_case_id": rec["case_id"],
            "worst_element": {"flat_index": index, "row": row, "col": col},
            "tolerance": rec.get("tolerance"),
            "metrics": rec.get("metrics"),
        }

    per_variant: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANT_NAME.values():
        rows = [r for r in cases if r.get("requested_variant") == variant]
        if not rows:
            continue
        per_variant[variant] = {
            "cases": len(rows),
            "pass": sum(1 for r in rows if r["status"] == "PASS"),
            "fail": sum(1 for r in rows if r["status"] == "FAIL"),
            "max_abs_max": max(
                ((r.get("metrics") or {}).get("max_abs") or 0.0) for r in rows
            ),
        }

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": verdict.get("run_id"),
        "overall": verdict.get("overall"),
        "tolerance_table": spec.get("tolerance_table"),
        "main": {
            "total": len(cases),
            "pass": sum(1 for r in cases if r["status"] == "PASS"),
            "fail": sum(1 for r in cases if r["status"] == "FAIL"),
            "worst_fp32": _worst(cases, "fp32"),
            "worst_fp16": _worst(cases, "fp16"),
        },
        "per_variant": per_variant,
        "tolerance_headroom": {
            dtype: _headroom(cases, dtype) for dtype in ("fp32", "fp16")
        },
        "per_shape_class": {
            name: {
                "cases": rec["total"],
                "pass": rec["pass"],
                "fail": rec["fail"],
            }
            for name, rec in sorted(
                coverage.get("per_axis", {}).get("shape_class", {}).items()
            )
        },
        "negative_cases": {
            "total": len(negatives),
            "pass": sum(1 for r in negatives if r["status"] == "PASS"),
            "reason_codes": sorted(
                {r.get("reason_code") for r in negatives if r.get("reason_code")}
            ),
        },
        "per_mode": {
            name: rec
            for name, rec in sorted(
                coverage.get("per_axis", {}).get("mode", {}).items()
            )
        },
        "sub_matrices": {
            "in_place": _status_counts(in_place),
            "auto_routing": _status_counts(auto_cases),
            "epsilon_sensitivity": _status_counts(eps_cases),
            "unsupported_combinations": _status_counts(unsupported),
            "repeatability": _status_counts(repeats),
            "historical_regression": _status_counts(historical.get("cases", [])),
        },
        "repeatability": {
            "runs_per_case": REPEAT_RUNS,
            "cases": len(repeats),
            "bitwise_stable": sum(
                1 for r in repeats if r.get("repeat_bitwise_stable") is True
            ),
        },
        "order_independence": order,
        "epsilon_values_covered": sorted({r.get("epsilon") for r in eps_cases}),
        "unsupported_reason_codes": sorted(
            {r.get("reason_code") for r in unsupported if r.get("reason_code")}
        ),
        "declared_divergences": list(rc.DECLARED_DIVERGENCE.values()),
        "historical_regression_pilot_reference": historical.get("note"),
        "source_identity": {
            "cuda_rmsnorm_lib": provenance.get("operator_source", {}).get(
                "cuda_rmsnorm_lib"
            ),
            "cuda_rmsnorm_lib_sha256": provenance.get("operator_source", {}).get(
                "cuda_rmsnorm_lib_sha256"
            ),
            "git_commit": provenance.get("environment", {}).get("git_commit"),
        },
        "verdict_conditions": verdict.get("conditions"),
        "generated_at_utc": _utc_now(),
    }
    _write_json(out_dir / "summary.json", summary)

    manifest = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            manifest[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _write_json(out_dir / "EVIDENCE_MANIFEST.json", {
        "experiment_id": EXPERIMENT_ID,
        "run_id": verdict.get("run_id"),
        "files": manifest,
        "generated_at_utc": _utc_now(),
    })
    print(f"[{EXPERIMENT_ID}] summarize: {len(manifest)} artifacts, "
          f"overall={verdict.get('overall')}")
    return 0


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E03-01 RMSNorm semantics, dual oracle and correctness boundary."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "verify", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
        p.add_argument("--run-id", default=None)
        p.add_argument("--s02-census", default=DEFAULT_S02_CENSUS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        return main_collect(args)
    if args.command == "verify":
        return main_verify(args)
    return main_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
