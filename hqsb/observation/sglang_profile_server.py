"""SGLang 0.5.5 diagnostic entry: upstream server plus worker profile controls.

The spawn main module installs a narrowly targeted import hook in every child.
Only the profiler mixin's public start/stop methods receive ERE control calls;
model execution, CUDA runtime objects and installed packages keep their APIs.
"""
import importlib.abc
import importlib.machinery
import os
import sys
from functools import wraps

TARGET = "sglang.srt.managers.scheduler_profiler_mixin"


def control(start):
    import ctypes
    import torch
    torch.cuda.synchronize()
    hook = ctypes.CDLL(os.environ["NEUTRINO_HOOK_DRIVER"])
    call = hook.hqsb_eretrace_start if start else hook.hqsb_eretrace_stop
    call.argtypes, call.restype = [], ctypes.c_int
    if call() != 0:
        raise RuntimeError("ERE_SGLANG_PROFILE_CONTROL_FAILED")


def adapt(module):
    cls = module.SchedulerProfilerMixin
    if getattr(cls, "_hqsb_eretrace", False):
        return
    original_start, original_stop = cls.start_profile, cls.stop_profile

    @wraps(original_start)
    def start(self, *args, **kwargs):
        result = original_start(self, *args, **kwargs)
        if "CUDA_PROFILER" in self.profiler_activities and getattr(result, "success", True):
            control(True)
        return result

    @wraps(original_stop)
    def stop(self, *args, **kwargs):
        if self.profile_in_progress and "CUDA_PROFILER" in self.profiler_activities:
            control(False)
        return original_stop(self, *args, **kwargs)

    cls.start_profile, cls.stop_profile = start, stop
    cls._hqsb_eretrace = True


class ProfileLoader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        return self.delegate.create_module(spec)

    def exec_module(self, module):
        self.delegate.exec_module(module)
        adapt(module)


class ProfileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError("ERE_SGLANG_PROFILER_MODULE_REQUIRED")
        spec.loader = ProfileLoader(spec.loader)
        return spec


if os.environ.get("HQSB_ERE_RUNTIME"):
    sys.meta_path.insert(0, ProfileFinder())


def main():
    from sglang.launch_server import prepare_server_args, run_server, kill_process_tree
    args = prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
