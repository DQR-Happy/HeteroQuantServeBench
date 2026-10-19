"""Local safetensors element budget; independent of model/accelerator imports."""

import json
import struct


def weight_budget(root, size):
    elements = 0
    for path in root.glob("*.safetensors"):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 8 * 1024**2:
                raise ValueError("L9_WEIGHT_HEADER_BUDGET")
            header = json.loads(f.read(n))
        for k, v in header.items():
            if k != "__metadata__":
                count = 1
                for dim in v["shape"]:
                    count *= dim
                elements += count
    if not elements:
        raise ValueError("L9_LOCAL_SAFETENSORS_REQUIRED")
    return elements * size
