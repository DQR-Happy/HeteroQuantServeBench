"""Graph format contracts; no compiler, exporter or device imports here."""
from typing import Literal
from pydantic import BaseModel, ConfigDict


class GraphCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["onnx", "inductor"]


# A format owns its adapter and evidence semantics. Future engines can add a
# format without changing inference selection or the model-family registry.
FORMATS = [
    {"id": "static", "label": "静态语义结构", "source": "config + Safetensors + reviewed implementation"},
    {"id": "aten", "label": "实际 ATen 依赖", "source": "TorchDispatch / storage dependencies"},
    {"id": "fx", "label": "Dynamo FX 编译输入", "source": "actual Dynamo backend GraphModule"},
    {"id": "onnx", "label": "ONNX 导出图", "source": "torch.onnx / torch.export; static prefill, no KV"},
    {"id": "inductor_pre", "label": "Inductor 融合前 IR", "source": "actual scheduler ir_pre_fusion.txt"},
    {"id": "inductor_post", "label": "Inductor 融合后 IR", "source": "actual scheduler ir_post_fusion.txt"},
    {"id": "generated_code", "label": "Inductor 生成代码", "source": "actual output_code.py; text, not a graph"},
]


def validate_capture(value, *, runtime, native, privileged, observation, quality=False, tokens=4):
    capture = GraphCapture.model_validate(value).model_dump()
    if not native or privileged or observation != "off" or quality or tokens > 4:
        raise ValueError("GRAPH_CAPTURE_REQUIRES_NATIVE_FLOAT_DEFAULT_KV_OFF_DIAGNOSTIC_MAX_4_TOKENS")
    expected = "eager" if capture["kind"] == "onnx" else "inductor"
    if not runtime or runtime["mode"] != expected:
        raise ValueError("GRAPH_CAPTURE_REQUIRES_RUNTIME_" + expected.upper())
    return capture
