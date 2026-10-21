"""vLLM 0.11 V1 diagnostic worker; its model execution remains upstream.

The public worker-class extension replaces only the profile control method.
The upstream HTTP /start_profile route needs its profiler environment switch;
temporarily suppress that switch during Worker construction to avoid creating
a competing torch/CUPTI profiler. This module is loaded only in diagnostic jobs.
"""
import os

from vllm.v1.worker.gpu_worker import Worker


class ProfileWorker(Worker):
    def __init__(self, *args, **kwargs):
        marker = os.environ.pop("VLLM_TORCH_PROFILER_DIR", None)
        try:
            super().__init__(*args, **kwargs)
        finally:
            if marker is not None:
                os.environ["VLLM_TORCH_PROFILER_DIR"] = marker

    def profile(self, is_start=True):
        import torch
        torch.cuda.synchronize()
        if os.environ.get("HQSB_ERE_RUNTIME"):
            import ctypes
            hook = ctypes.CDLL(os.environ["NEUTRINO_HOOK_DRIVER"])
            control = hook.hqsb_eretrace_start if is_start else hook.hqsb_eretrace_stop
            control.argtypes, control.restype = [], ctypes.c_int
            if control() != 0:
                raise RuntimeError("ERE_CAPTURE_CONTROL_FAILED")
            return
        runtime = torch.cuda.cudart()
        result = runtime.cudaProfilerStart() if is_start else runtime.cudaProfilerStop()
        if int(result) != 0:
            raise RuntimeError(f"CUDA_PROFILER_CONTROL_FAILED:{result}")
