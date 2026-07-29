"""Read-only resource accounting, independent of GPU libraries and HTTP."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from hqsb.console import __version__


def read_proc_bytes(path: Path) -> dict[str, int]:
    values = {}
    try:
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parts = value.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * (1024 if parts[-1] == "kB" else 1)
    except OSError:
        pass
    return values


def host_memory(label: str = "current", proc: Path = Path("/proc")) -> dict:
    memory = read_proc_bytes(proc / "meminfo")
    return {
        "label": label,
        "sampled_at": time.time(),
        "total_bytes": memory.get("MemTotal"),
        "available_bytes": memory.get("MemAvailable"),
        "free_bytes": memory.get("MemFree"),
        "cached_bytes": memory.get("Cached"),
        "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"]
        if {"SwapTotal", "SwapFree"} <= memory.keys()
        else None,
        "memory_scope": "control_host",
        "source": "procfs",
        "availability": "measured" if memory else "unavailable",
    }


def process_memory(pid: int, role: str, proc: Path = Path("/proc")) -> dict:
    status = read_proc_bytes(proc / str(pid) / "status")
    rollup = read_proc_bytes(proc / str(pid) / "smaps_rollup")
    return {
        "pid": pid,
        "role": role,
        "sampled_at": time.time(),
        "rss_bytes": status.get("VmRSS"),
        "pss_bytes": rollup.get("Pss"),
        "swap_bytes": status.get("VmSwap"),
        "availability": "measured" if "VmRSS" in status else "unavailable",
    }


def resource_snapshot(service) -> dict:
    processes = [process_memory(os.getpid(), "console_api")]
    deployments = service.deployments()
    for deployment in deployments:
        actor = service.actors.get(deployment["id"])
        process = getattr(actor, "process", None)
        if process is not None and process.pid:
            processes.append(process_memory(process.pid, "worker:" + deployment["id"]))
    return {
        "schema_version": 1,
        "sampled_at": time.time(),
        "host": host_memory(),
        "processes": processes,
        "deployments": [
            {
                "id": row["id"],
                "state": row["state"],
                "load_observation": {
                    "sampled_at": row.get("loaded_at"),
                    "memory": row.get("detail", {}).get("memory", {}),
                    "source": "load_time_snapshot",
                },
                "latest_observation": row.get("latest_observation"),
                "lifecycle": row.get("memory_lifecycle", []),
            }
            for row in deployments
        ],
        "limitations": [
            "MemAvailable 是模型和其他进程占用后的系统可用量估计，不是保证可分配的 GPU 内存。",
            "RSS/PSS、allocator allocated/reserved 存在不同或重叠范围，不能相加；reserved 包含 allocated。",
            "加载快照与最近一次推理快照不是实时 allocator 采样；未采集项保留为空。",
            "本页主机数据属于 Console 控制节点，不能代表外部 API 提供商设备。",
        ],
    }


def observation_capabilities(service) -> dict:
    local = any(cfg.provider == "pytorch" for cfg in service.configs.values())
    ncu = shutil.which("ncu") or next(
        (
            str(p)
            for p in Path("/opt/nvidia/nsight-compute").glob("*/ncu")
            if p.is_file()
        ),
        None,
    )
    definitions = [
        (
            "request_phases",
            "请求阶段与逐 token 观测",
            local,
            "本地 PyTorch worker；CPU 调用包络与 GPU 时间分开。",
        ),
        (
            "torch_profiler",
            "限定窗口算子采集",
            local,
            "worker 运行时检查；CUDA 活动依赖 CUPTI 权限，缺失明确标注。",
        ),
        (
            "memory_accounting",
            "内存口径与生命周期",
            True,
            "procfs 和 worker 历史快照；不同范围不可相加。",
        ),
        (
            "rtn_artifact",
            "自制 RTN-W4/W8 制品",
            local,
            "需要 ready 模型、内存/磁盘预算；仅制品，不自动改变部署。",
        ),
        (
            "native_low_bit",
            "原生低比特交互部署",
            False,
            "尚无通过质量和模型回接验收的本地低比特部署。",
        ),
        (
            "nsys_live",
            "Nsight Systems 实时采集",
            False,
            "工具可发现" if shutil.which("nsys") else "工具未发现",
        ),
        (
            "ncu_live",
            "Nsight Compute 实时计数器",
            False,
            "工具可发现；当前支持历史证据，实时任务尚未接入。"
            if ncu
            else "工具未发现。",
        ),
        (
            "byte_register_trace",
            "全量每字节/物理寄存器录像",
            False,
            "无完整零扰动的观测接口；不生成推测轨迹。",
        ),
    ]
    return {
        "schema_version": 1,
        "version": __version__,
        "worker_mode": "privileged_stdio"
        if service.settings.privileged_worker
        else "standard_process",
        "items": [
            {
                "id": key,
                "name": name,
                "status": "available" if supported else "unavailable",
                "reason": reason,
                "scope": "control_host",
            }
            for key, name, supported, reason in definitions
        ],
    }
