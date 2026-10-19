"""Linux CPU/NUMA facts and NVML adapter. No torch or hardware work on import."""

import ctypes
import ctypes.util
import os
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cuda_uuid(value):
    value = str(value)
    return value if value.startswith(("GPU-", "MIG-")) else "GPU-" + value


def numa_policy(node=None):
    """Read current policy; bind only future allocations in this calling thread.

    Executed before importing torch in a fresh child, so subsequently created
    threads inherit the policy. No migration of resident weights is claimed.
    """
    name = ctypes.util.find_library("numa")
    if not name:
        raise RuntimeError("libnuma is not installed")
    lib = ctypes.CDLL(name, use_errno=True)
    mask = (ctypes.c_ulong * 1)()
    mode = ctypes.c_int()
    if lib.get_mempolicy(ctypes.byref(mode), mask, 64, None, 0) != 0:
        raise OSError(
            ctypes.get_errno(), "get_mempolicy unavailable (container/seccomp policy)"
        )
    if node is not None:
        mask[0] = 1 << node
        if lib.set_mempolicy(2, mask, 64) != 0:
            raise OSError(ctypes.get_errno(), "set_mempolicy MPOL_BIND denied")
        return numa_policy()
    return {"mode": mode.value, "node_mask": int(mask[0])}


class Nvidia:
    def __init__(self):
        import pynvml

        self.api = pynvml
        pynvml.nvmlInit()

    def close(self):
        self.api.nvmlShutdown()

    def handle(self, uuid):
        return self.api.nvmlDeviceGetHandleByUUID(uuid)

    def power(self, uuid):
        return self.api.nvmlDeviceGetPowerManagementLimit(self.handle(uuid)) / 1000

    def set_power(self, uuid, watts):
        if self.power(uuid) == watts:
            return watts
        self.api.nvmlDeviceSetPowerManagementLimit(self.handle(uuid), int(watts * 1000))
        actual = self.power(uuid)
        if actual != watts:
            raise RuntimeError(f"L1_POWER_READBACK_MISMATCH: {actual} != {watts}")
        return actual

    def processes(self, uuid):
        # Fail closed if driver cannot enumerate either compute or graphics.
        h, n = self.handle(uuid), self.api
        return sorted(
            {p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)}
            | {p.pid for p in n.nvmlDeviceGetGraphicsRunningProcesses(h)}
        )

    def sample(self, uuid):
        n, h = self.api, self.handle(uuid)
        result, unavailable = {}, {}
        calls = {
            "power_w": lambda: n.nvmlDeviceGetPowerUsage(h) / 1000,
            "temperature_c": lambda: n.nvmlDeviceGetTemperature(
                h, n.NVML_TEMPERATURE_GPU
            ),
            "sm_clock_mhz": lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
            "memory_clock_mhz": lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
            "gpu_util_percent": lambda: n.nvmlDeviceGetUtilizationRates(h).gpu,
            "memory_used_bytes": lambda: n.nvmlDeviceGetMemoryInfo(h).used,
            "throttle_reasons": lambda: n.nvmlDeviceGetCurrentClocksThrottleReasons(h),
            "total_energy_mj": lambda: n.nvmlDeviceGetTotalEnergyConsumption(h),
        }
        for key, call in calls.items():
            try:
                result[key] = call()
            except n.NVMLError as exc:
                result[key], unavailable[key] = None, str(exc)
        return {**result, "unavailable": unavailable}


def inventory():
    cpus = sorted(os.sched_getaffinity(0))
    nodes = []
    for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        nodes.append(
            {
                "id": int(p.name[4:]),
                "cpulist": read(p / "cpulist"),
                "distance": read(p / "distance"),
            }
        )
    try:
        policy = {"available": True, "current": numa_policy(), "reason": ""}
    except (OSError, RuntimeError, AttributeError) as exc:
        policy = {"available": False, "reason": str(exc)}
    return {
        "allowed_cpus": cpus,
        "cpu_quota": read("/sys/fs/cgroup/cpu.max"),
        "kernel": os.uname().release,
        "machine": os.uname().machine,
        "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
        "cpu_topology": [
            {
                "cpu": i,
                "package": read(
                    f"/sys/devices/system/cpu/cpu{i}/topology/physical_package_id"
                ),
                "core": read(f"/sys/devices/system/cpu/cpu{i}/topology/core_id"),
                "thread_siblings": read(
                    f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list"
                ),
            }
            for i in cpus
        ],
        "cpu_caches": [
            {
                k: read(p / k)
                for k in (
                    "level",
                    "type",
                    "size",
                    "coherency_line_size",
                    "shared_cpu_list",
                )
            }
            for p in sorted(
                Path(f"/sys/devices/system/cpu/cpu{cpus[0]}/cache").glob("index*")
            )
        ]
        if cpus
        else [],
        "cpu_identity": next(
            (
                line
                for line in (read("/proc/cpuinfo") or "").splitlines()
                if line.startswith(("model name", "Hardware"))
            ),
            None,
        ),
        "cpuset_effective": read("/sys/fs/cgroup/cpuset.cpus.effective"),
        "mems_effective": read("/sys/fs/cgroup/cpuset.mems.effective"),
        "memory_max": read("/sys/fs/cgroup/memory.max"),
        "cpu_governors": {
            str(i): read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor")
            for i in cpus
        },
        "numa_nodes": nodes,
        "numa_policy": policy,
        "limitations": [
            "Hardware selection uses registered nodes; physical hardware cannot be replaced by a software setting.",
            "GPU power is board-wide; set requires exclusive lease, readable original limit, no external GPU process and driver permission.",
            "CPU governor/clock controls, GPU/EMC clock locking and Jetson nvpmodel are not writable here: no recoverable portable prior-lock-state API.",
            "No NPU execution adapter, CPU/GPU heterogeneous partitioning, or interconnect reconfiguration in L1.",
            "CPU page faults and NUMA pages do not describe GPU L2 hit rate or physical GPU pages.",
        ],
    }
