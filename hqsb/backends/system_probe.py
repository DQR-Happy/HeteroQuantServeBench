"""Read-only isolated L1 inventory; respects CUDA visibility via torch UUID."""

import json
import importlib.metadata
from hqsb.hardware.system import inventory, Nvidia, cuda_uuid, read


def probe():
    import torch

    result = inventory()
    result.update(torch=torch.__version__, cuda=torch.version.cuda, devices=[])
    nvml, reason = None, ""
    try:
        nvml = Nvidia()
        result["nvml_binding_version"] = importlib.metadata.version("nvidia-ml-py")
        result["nvidia_driver"] = nvml.api.nvmlSystemGetDriverVersion()
    except (ImportError, OSError, RuntimeError) as exc:
        reason = str(exc)
    except Exception as exc:
        # NVML runtime errors are capability facts, never successful settings.
        reason = str(exc)
    try:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            uuid = cuda_uuid(p.uuid)
            d = {
                "id": i,
                "uuid": uuid,
                "name": p.name,
                "total_bytes": p.total_memory,
                "compute_capability": [p.major, p.minor],
                "power_constraints_w": None,
                "power_reason": reason,
                "power_permission": "checked_when_applying_not_inferred_from_uid",
            }
            if nvml:
                try:
                    h = nvml.handle(uuid)
                    bus = nvml.api.nvmlDeviceGetPciInfo(h).busId
                    bus = bus.decode() if isinstance(bus, bytes) else bus
                    d.update(
                        power_constraints_w=[
                            x / 1000
                            for x in nvml.api.nvmlDeviceGetPowerManagementLimitConstraints(
                                h
                            )
                        ],
                        power_limit_w=nvml.power(uuid),
                        sample=nvml.sample(uuid),
                        pci=bus,
                        pci_numa_node=read(
                            f"/sys/bus/pci/devices/{bus.lower().replace('00000000:', '0000:')}/numa_node"
                        ),
                    )
                except nvml.api.NVMLError as exc:
                    d["power_reason"] = str(exc)
            result["devices"].append(d)
    finally:
        if nvml:
            nvml.close()
    return result


if __name__ == "__main__":
    print(json.dumps(probe()))
