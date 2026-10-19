"""Request-scoped L3 execution contracts; safe to import on the control plane."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool


class RuntimePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["eager", "dynamo_eager", "inductor"] = "eager"
    shape_policy: Literal["auto", "static", "dynamic"] = "auto"
    fullgraph: StrictBool = False


def validate_runtime(plan, *, provider="pytorch", kernel="framework-native",
                     method="none", kv_bits=None, privileged=False, capture=False,
                     observation="off"):
    resolved = RuntimePlan.model_validate(plan or {}).model_dump()
    if provider != "pytorch":
        raise ValueError("L3_LOCAL_PYTORCH_REQUIRED")
    if resolved["mode"] != "eager":
        if privileged:
            raise ValueError("L3_PRIVILEGED_WORKER_NOT_VERIFIED")
        if kernel != "framework-native" or method != "none" or kv_bits:
            raise ValueError("L3_REQUIRES_NATIVE_FLOAT_MODEL_AND_DEFAULT_KV")
        if capture or observation != "off":
            raise ValueError("L3_COMPILE_REQUIRES_OBSERVATION_OFF:ATen hooks change capture semantics")
    return resolved


def catalogue():
    return {"schema_version": "l3-v1", "scope": "request", "default": RuntimePlan().model_dump(),
            "modes": [
                {"id": "eager", "label": "Eager 原生执行", "lowering": False},
                {"id": "dynamo_eager", "label": "Dynamo 捕获 / FX 原样执行", "lowering": False},
                {"id": "inductor", "label": "Inductor 编译优化", "lowering": True}],
            "shape_policies": ["auto", "static", "dynamic"],
            "cuda_graphs": "disabled_pending_static_KV_address_lifetime_validation",
            "limitations": ["Compilation is lazy; first forwards include compilation costs.",
                            "FX graphs are compiler input, not physical kernel/fusion evidence.",
                            "Quantized/custom-kernel and L8 composition are not verified."]}
