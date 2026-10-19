"""Explicit worker-only, weight-free ONNX structure export for one static prefill.

No replacement of the inference backend. Export errors are diagnostic failures,
recorded independently from generation. The exported inputs include weights;
this artifact alone is deliberately not a deployable model checkpoint.
"""
from pathlib import Path
import time

from hqsb.backends.graph_formats import onnx_graph


def capture(model, original_forward, input_ids, attention_mask, directory):
    start = time.monotonic()
    entry = {"kind": "onnx", "id": "onnx-prefill", "phase": "prefill",
             "scope": "static batch=1 no-KV forward; parameters are graph inputs, no weights saved",
             "input_shape": list(input_ids.shape), "status": "failed"}
    if input_ids.shape[0] != 1 or input_ids.shape[1] > 128:
        entry.update(status="unsupported_input", reason="ONNX diagnostic supports batch=1 and at most 128 input tokens")
        return entry
    try:
        import torch
        import onnx

        class Prefill(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, ids, mask):
                return original_forward(input_ids=ids, attention_mask=mask,
                                        use_cache=False, return_dict=True).logits

        # Avoid model.forward's diagnostic wrapper while exporting: the original
        # bound forward still references the exact resident parameters.
        program = torch.onnx.export(Prefill().eval(), (input_ids, attention_mask),
            dynamo=True, optimize=False, fallback=False, input_names=["input_ids", "attention_mask"],
            output_names=["logits"], opset_version=18)
        path = Path(directory) / "graph-export.onnx"
        program.save(path, include_initializers=False, keep_initializers_as_inputs=True, external_data=False)
        if path.stat().st_size > 32 * 1024**2:
            path.unlink()
            raise ValueError("ONNX_STRUCTURE_EXCEEDS_32_MIB")
        proto = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(proto)
        entry.update(onnx_graph(proto), status="available", artifact=path.name,
                     exporter="torch.onnx(dynamo=True, optimize=False)", checker="passed",
                     numerical_equivalence="not_checked; structural checker is not numerical validation")
    except ImportError as exc:
        entry.update(status="dependency_missing", error=str(exc)[:1000])
    except Exception as exc:
        entry.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:2000]}")
        path = Path(directory) / "graph-export.onnx"
        path.unlink(missing_ok=True)
    entry["capture_host_ms"] = (time.monotonic() - start) * 1000
    return entry
