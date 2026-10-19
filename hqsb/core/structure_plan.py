"""L5 compression contracts: lossy transforms, never kernel speedup claims."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StructurePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["magnitude", "wanda", "mlp_channels", "svd_low_rank"]
    targets: list[str] = Field(default_factory=list, max_length=32)
    max_modules: int = Field(4, ge=1, le=32)
    sparsity: float = Field(0.5, ge=0, le=0.9)
    pattern: Literal["unstructured", "2:4"] = "unstructured"
    keep_ratio: float = Field(0.5, ge=0.1, le=0.95)
    calibration: list[str] = Field(default_factory=list, max_length=16)
    calibration_max_tokens: int = Field(64, ge=8, le=256)
    seed: int = Field(42, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def valid(self):
        if len(set(self.targets)) != len(self.targets) or any(
            not x or len(x) > 256 for x in self.targets
        ):
            raise ValueError("L5_INVALID_OR_DUPLICATE_TARGET")
        if self.targets and len(self.targets) > self.max_modules:
            raise ValueError("L5_TARGET_COUNT_EXCEEDS_MODULE_BUDGET")
        if self.method == "wanda" and not self.calibration:
            raise ValueError("L5_WANDA_REQUIRES_SEPARATE_CALIBRATION_TEXT")
        if self.method != "wanda" and self.calibration:
            raise ValueError("L5_CALIBRATION_ONLY_USED_BY_WANDA")
        if any(not text.strip() or len(text) > 8192 for text in self.calibration):
            raise ValueError("L5_CALIBRATION_TEXT_BOUND")
        if self.pattern == "2:4" and (
            self.method not in {"magnitude", "wanda"} or self.sparsity != 0.5
        ):
            raise ValueError("L5_2_4_REQUIRES_MAGNITUDE_OR_WANDA_AND_50_PERCENT")
        return self


def catalogue():
    return {
        "schema_version": "l5-v1",
        "methods": [
            {
                "id": "magnitude",
                "label": "幅值剪枝（稠密执行）",
                "changes_shape": False,
                "reference": "https://pytorch.org/tutorials/intermediate/pruning_tutorial.html",
            },
            {
                "id": "wanda",
                "label": "Wanda 评分剪枝 / 单次校准（稠密执行）",
                "changes_shape": False,
                "reference": "https://github.com/locuslab/wanda",
            },
            {
                "id": "mlp_channels",
                "label": "成组 MLP 通道裁剪（真实缩维）",
                "changes_shape": True,
                "reference": None,
            },
            {
                "id": "svd_low_rank",
                "label": "截断 SVD（双 Linear 紧凑执行）",
                "changes_shape": True,
                "reference": "https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html",
            },
        ],
        "not_implemented": [
            {
                "id": "distillation",
                "reason": "需要教师/学生训练、数据与训练后验收；当前不是训练平台。",
            },
            {
                "id": "gqa_mqa_mla_conversion",
                "reason": "当前可使用原生此类架构的模型；任意既有权重转换需要模型专属映射与训练/质量验收。",
            },
            {
                "id": "moe_structure_rewrite",
                "reason": "当前保留原模型路由与专家；未实现训练和专家结构改写。",
            },
            {
                "id": "sparse_local_linear_attention",
                "reason": "需要独立 Attention 语义、长距离质量及 KV 适配；当前不伪装为普通权重剪枝。",
            },
        ],
        "limits": {
            "max_targets": 32,
            "artifact_format": "hqsb-l5-v1",
            "creation_device": "cpu_isolated_process",
        },
        "limitations": [
            "Lossy transforms require independent quality evaluation; no quality preservation is implied.",
            "Zero masks and 2:4 patterns execute dense kernels here; disk/VRAM savings and sparse acceleration are not implied.",
            "Wanda uses one frozen activation-statistics pass, not the official sequential pruning/calibration recipe.",
            "Gated MLP pruning and SVD are limited to inspected native Linear modules; L4/compiled/custom-kernel composition is not verified.",
        ],
    }
