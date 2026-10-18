"""L4 request contracts and method registry; no torch or device imports."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from hqsb.core.optimization import fingerprint

METHODS = {
    "none": (
        "Floating point",
        None,
        "framework",
        "https://pytorch.org/docs/stable/tensor_attributes.html",
    ),
    "rtn": (
        "Round-to-nearest W4/W8",
        None,
        "hqsb_fused_dequant",
        "https://github.com/IST-DASLab/gptq",
    ),
    "gptq": (
        "GPTQ",
        "llmcompressor",
        "hqsb_fused_dequant",
        "https://github.com/IST-DASLab/gptq",
    ),
    "awq": (
        "Activation-aware Weight Quantization",
        "llmcompressor",
        "hqsb_fused_dequant",
        "https://github.com/mit-han-lab/llm-awq",
    ),
    "smoothquant": (
        "SmoothQuant W8A8",
        "llmcompressor",
        "cuda_int8_mm",
        "https://github.com/mit-han-lab/smoothquant",
    ),
    "bnb_int8": (
        "LLM.int8()",
        "bitsandbytes",
        "bitsandbytes",
        "https://github.com/bitsandbytes-foundation/bitsandbytes",
    ),
    "bnb_nf4": (
        "NormalFloat4 / double quant",
        "bitsandbytes",
        "bitsandbytes",
        "https://arxiv.org/abs/2305.14314",
    ),
    "bnb_fp4": (
        "bitsandbytes FP4",
        "bitsandbytes",
        "bitsandbytes",
        "https://github.com/bitsandbytes-foundation/bitsandbytes",
    ),
    "fp8": (
        "Dynamic E4M3 W8A8",
        None,
        "cuda_scaled_mm",
        "https://docs.pytorch.org/ao/stable/quantization_overview.html",
    ),
}


def resolve_exclusions(linear_names, suffixes):
    """Resolve user Linear names/suffixes once, before any model mutation.

    Calibration libraries differ in their short-name matching semantics. Pass
    exact names to them and fail on typos instead of silently quantizing a
    module the user asked to preserve.
    """
    resolved = set()
    for suffix in suffixes:
        matches = {
            name
            for name in linear_names
            if name == suffix or name.endswith("." + suffix)
        }
        if not matches:
            raise ValueError("L4_EXCLUSION_NOT_MATCHED:" + suffix)
        resolved.update(matches)
    return sorted(resolved)


class PrecisionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["l4-v1"] = "l4-v1"
    method: Literal[
        "none",
        "rtn",
        "gptq",
        "awq",
        "smoothquant",
        "bnb_int8",
        "bnb_nf4",
        "bnb_fp4",
        "fp8",
    ] = "none"
    allow_tf32: bool = False
    bits: Literal[4, 8] = 4
    group_size: Literal[32, 64, 128] = 128
    exclude_modules: list[str] = Field(
        default_factory=lambda: ["lm_head"], max_length=64
    )
    calibration: list[str] = Field(default_factory=list, max_length=128)
    calibration_tokens: int = Field(128, ge=16, le=512)
    smooth_alpha: float = Field(0.5, ge=0, le=1)
    damping_fraction: float = Field(0.01, gt=0, le=0.2)
    awq_grid: int = Field(20, ge=2, le=64)
    double_quant: bool = True
    outlier_threshold: float = Field(6.0, ge=0, le=64)
    kv_bits: Literal[2, 4] | None = None
    kv_group_size: Literal[32, 64, 128] = 64
    kv_residual_length: int = Field(128, ge=1, le=4096)
    artifact_id: str | None = Field(None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def semantics(self):
        if self.artifact_id and self.method in {"none", "fp8"}:
            raise ValueError("L4_METHOD_HAS_NO_PERSISTED_ARTIFACT")
        if self.method in {"smoothquant", "bnb_int8", "fp8"}:
            if "bits" in self.model_fields_set and self.bits != 8:
                raise ValueError("L4_METHOD_REQUIRES_8_BITS")
            self.bits = 8
        if self.method in {"bnb_nf4", "bnb_fp4"} and self.bits != 4:
            raise ValueError("L4_BNB_4BIT_FORMAT_REQUIRES_4_BITS")
        if (
            self.method in {"gptq", "awq", "smoothquant"}
            and not self.artifact_id
            and not self.calibration
        ):
            raise ValueError(
                "L4_CALIBRATION_REQUIRED: supply representative calibration texts, independently of evaluation"
            )
        if self.calibration and (
            self.method not in {"gptq", "awq", "smoothquant"} or self.artifact_id
        ):
            raise ValueError("L4_CALIBRATION_NOT_APPLICABLE")
        if any(
            not isinstance(s, str) or not s.strip() or len(s) > 4096
            for s in self.calibration
        ):
            raise ValueError("L4_CALIBRATION_TEXT_INVALID")
        if any(
            not name or len(name) > 256 or any(c in name for c in "*[]/")
            for name in self.exclude_modules
        ):
            raise ValueError("L4_EXCLUSIONS_MUST_BE_MODULE_NAMES_OR_SUFFIXES")
        if len(set(self.exclude_modules)) != len(self.exclude_modules):
            raise ValueError("L4_DUPLICATE_EXCLUSION")
        if self.method == "awq" and self.bits != 4:
            raise ValueError("L4_AWQ_REQUIRES_W4")
        return self

    def identity(self):
        body = self.model_dump(exclude={"artifact_id"})
        body["calibration"] = fingerprint(body["calibration"])
        if self.method not in {"rtn", "gptq", "awq"}:
            body.pop("group_size")
        return body


def catalogue():
    return {
        "schema_version": "l4-v1",
        "compute_dtypes": ["float32", "float16", "bfloat16"],
        "methods": [
            {
                "id": k,
                "name": v[0],
                "calibration_backend": v[1],
                "execution": v[2],
                "source": v[3],
                "requires_calibration": k in {"gptq", "awq", "smoothquant"},
                "minimum_cuda_sm": 89 if k == "fp8" else 75 if k == "bnb_int8" else 70,
                "availability": "requires_target_probe",
            }
            for k, v in METHODS.items()
        ],
        "kv": {
            "backend": "optimum-quanto",
            "bits": [2, 4],
            "execution": "packed_storage_with_dequantized_attention",
            "limits": [
                "single-request full attention only",
                "residual and dequantization workspace are measured separately",
            ],
        },
        "calibration_runtime": "isolated llmcompressor 0.9.0 / Transformers 4.57.3; serving runtime unchanged",
        "quality": "teacher-forcing logits, perplexity and task evaluation; no automatic quality pass from storage size",
    }
