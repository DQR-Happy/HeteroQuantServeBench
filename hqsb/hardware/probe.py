"""Device / CUDA probe for the environment fingerprint.

Supplies the torch-dependent device facts that ``hqsb.core.fingerprint``
(which must stay dependency-free) cannot collect itself: CUDA availability,
device names, compute capabilities, and the CUDA runtime version.

Importing torch is deferred and guarded so that CPU-only or torch-less
environments still produce a usable (empty) probe.
"""

from __future__ import annotations

from typing import Any, Dict


def cuda_device_probe() -> Dict[str, Any]:
    """Return the device facts for ``collect_device_basic``.

    Never raises: on any import/query failure the corresponding values are
    left as their empty defaults.
    """
    probe: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "compute_capabilities": [],
        "cuda_runtime_version": "",
    }
    try:
        import torch  # noqa: PLC0415  (lazy, optional dependency)
    except Exception:
        return probe

    try:
        probe["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        return probe

    if not probe["cuda_available"]:
        return probe

    try:
        count = torch.cuda.device_count()
        probe["device_count"] = int(count)
        names = []
        capabilities = []
        for index in range(count):
            names.append(torch.cuda.get_device_name(index))
            capability = torch.cuda.get_device_capability(index)
            capabilities.append([int(capability[0]), int(capability[1])])
        probe["device_names"] = names
        probe["compute_capabilities"] = capabilities
    except Exception:
        pass

    try:
        probe["cuda_runtime_version"] = torch.version.cuda or ""
    except Exception:
        pass

    return probe


__all__ = ["cuda_device_probe"]
