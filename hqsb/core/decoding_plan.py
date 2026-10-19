"""L7 request semantics; no accelerator imports in the control process."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from hqsb.core.optimization import fingerprint


class DecodingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["autoregressive", "prompt_lookup", "assistant"] = "autoregressive"
    sampling: Literal["greedy", "multinomial"] = "greedy"
    temperature: float = Field(1.0, ge=0.05, le=5.0)
    top_k: int = Field(0, ge=0, le=1024)
    top_p: float = Field(1.0, gt=0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    min_new_tokens: int = Field(0, ge=0, le=256)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    draft_tokens: int = Field(4, ge=1, le=16)
    lookup_min_ngram: int = Field(2, ge=1, le=16)
    lookup_max_ngram: int = Field(6, ge=1, le=16)
    lookup_window: int = Field(4096, ge=16, le=8192)
    draft_deployment_id: str | None = Field(None, min_length=1, max_length=128)
    choices: list[str] = Field(default_factory=list, max_length=16)
    constraint_backend: Literal["mask", "gather"] = "gather"

    @model_validator(mode="after")
    def compatible(self):
        if self.sampling == "greedy" and (
            self.temperature != 1 or self.top_k or self.top_p != 1
        ):
            raise ValueError("L7_GREEDY_DOES_NOT_USE_TEMPERATURE_TOP_K_TOP_P")
        if self.method != "autoregressive" and (
            self.sampling != "greedy"
            or self.repetition_penalty != 1
            or self.min_new_tokens
            or self.choices
        ):
            raise ValueError(
                "L7_SPECULATION_REQUIRES_UNCONSTRAINED_GREEDY_NO_LOGIT_PROCESSORS"
            )
        if (self.method == "assistant") != bool(self.draft_deployment_id):
            raise ValueError("L7_ASSISTANT_REQUIRES_A_REGISTERED_DRAFT_DEPLOYMENT_ONLY")
        if self.lookup_min_ngram > self.lookup_max_ngram:
            raise ValueError("L7_LOOKUP_NGRAM_RANGE")
        if self.method != "prompt_lookup" and (
            self.lookup_min_ngram != 2
            or self.lookup_max_ngram != 6
            or self.lookup_window != 4096
        ):
            raise ValueError("L7_LOOKUP_OPTIONS_REQUIRE_PROMPT_LOOKUP")
        if self.method == "autoregressive" and self.draft_tokens != 4:
            raise ValueError("L7_DRAFT_BUDGET_REQUIRES_SPECULATION")
        if self.constraint_backend != "gather" and not self.choices:
            raise ValueError("L7_CONSTRAINT_BACKEND_REQUIRES_CHOICES")
        if len(set(self.choices)) != len(self.choices) or any(
            not c.strip() or len(c) > 2048 for c in self.choices
        ):
            raise ValueError("L7_CHOICES_MUST_BE_UNIQUE_NONEMPTY_BOUNDED_TEXT")
        return self


def public_plan(plan):
    import hashlib

    plan = DecodingPlan.model_validate(plan)
    return {
        **plan.model_dump(exclude={"choices"}),
        "choice_sha256": [hashlib.sha256(c.encode()).hexdigest() for c in plan.choices],
    }


def validate_decoding(plan, *, native, modified, request):
    p = DecodingPlan.model_validate(plan)
    if not native or modified:
        raise ValueError(
            "L7_REQUIRES_LOCAL_NATIVE_FLOAT_MODEL_DEFAULT_KV_RESTORE_ON_PRECISION_OR_STRUCTURE_PAGE"
        )
    if (
        request.get("memory_plan") is not None
        or request.get("runtime_plan") is not None
        or request.get("capture_architecture")
        or request.get("graph_capture")
        or request.get("experiment_role")
    ):
        raise ValueError(
            "L7_MEMORY_COMPILE_GRAPH_TEACHER_FORCING_COMPOSITION_NOT_VERIFIED"
        )
    if request.get("observation_mode", "basic") not in {"off", "basic"} or request.get(
        "observation_plan"
    ):
        raise ValueError("L7_FIRST_RELEASE_SUPPORTS_OFF_OR_BASIC_OBSERVATION_ONLY")
    if (
        request["max_output_tokens"] > 512
        or p.min_new_tokens >= request["max_output_tokens"]
    ):
        raise ValueError("L7_OUTPUT_BUDGET_1_TO_512_AND_MIN_NEW_TOKENS_BELOW_LIMIT")
    if p.choices and (
        request.get("fixed_output_length") or request.get("prompt") is None
    ):
        raise ValueError("L7_FINITE_CHOICES_REQUIRE_RAW_PROMPT_AND_EOS_STOPPING")
    return p.model_dump()


def catalogue():
    return {
        "schema_version": "l7-v1",
        "default": DecodingPlan().model_dump(),
        "methods": [
            {"id": "autoregressive", "label": "逐 Token / 贪心或采样"},
            {"id": "prompt_lookup", "label": "提示词查找推测 / 贪心验证"},
            {"id": "assistant", "label": "独立草稿模型推测 / 贪心验证"},
        ],
        "max_output_tokens": 512,
        "not_implemented": [
            "随机推测的拒绝采样与残差分布校正",
            "不同 Tokenizer 的通用辅助解码",
            "自推测早退、Medusa/EAGLE/MTP 训练头与树形验证",
            "通用 JSON Schema / 正则语法编译",
            "自定义融合采样 CUDA Kernel",
        ],
        "limitations": [
            "Finite token-choice tries are not general JSON grammar support.",
            "Greedy block verification can differ numerically from single-token forwards near ties; compare actual tokens.",
            "Teacher-forcing PPL is not a quality measurement of a decoding policy.",
        ],
    }


def plan_hash(plan):
    return fingerprint(public_plan(plan))
