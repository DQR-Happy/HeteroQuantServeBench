"""Bounded, read-only host telemetry; missing sensors stay missing."""

from __future__ import annotations

import collections
import csv
import math
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

from hqsb.benchmark.tegrastats_parser import parse_tegrastats_line


def parse_nvidia_sample(text):
    """Physical GPU 0 board sensors; N/A is missing, never zero or SoC power."""
    rows = list(csv.reader(text.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise ValueError("NVIDIA_TELEMETRY_SHAPE")
    uuid, name, *values = (field.strip() for field in rows[0])
    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) and result >= 0 else None
        except ValueError:
            return None
    util, temperature, watts, used, total = map(number, values)
    return {"gpu_uuid": uuid, "gpu_name": name, "gpu_util_pct": util,
            "gpu_temp_c": temperature, "power_w": watts,
            "gpu_memory_used_bytes": int(used * 1024**2) if used is not None else None,
            "gpu_memory_total_bytes": int(total * 1024**2) if total is not None else None,
            "power_scope": "GPU board", "sensor_device": "physical_gpu_0",
            "source": "nvidia-smi + procfs"}


def nvidia_sample():
    try:
        result = subprocess.run(["nvidia-smi", "--id=0",
            "--query-gpu=uuid,name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=True)
        return parse_nvidia_sample(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


class Telemetry:
    def __init__(self):
        self.samples = collections.deque(maxlen=120)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.process = None
        self.threads = []
        self.gpu = {}
        self.gpu_time = None
        self.nvidia = False

    def start(self):
        if shutil.which("tegrastats"):
            self.process = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            thread = threading.Thread(target=self._tegra, daemon=True)
            thread.start()
            self.threads.append(thread)
        else:
            self.nvidia = bool(shutil.which("nvidia-smi"))
        thread = threading.Thread(target=self._sample, daemon=True)
        thread.start()
        self.threads.append(thread)

    def _tegra(self):
        for line in self.process.stdout:
            with self.lock:
                self.gpu = parse_tegrastats_line(line)
                self.gpu_time = time.time()

    def _sample(self):
        while not self.stop_event.is_set():
            board = nvidia_sample() if self.nvidia else {}
            memory = {}
            try:
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, value = line.split(":", 1)
                    memory[key] = int(value.strip().split()[0]) * 1024
            except (OSError, ValueError):
                pass
            with self.lock:
                fresh = self.gpu_time is not None and time.time() - self.gpu_time < 5
                gpu = self.gpu if fresh else {}
                self.samples.append(
                    {
                        "time": time.time(),
                        "host_total_bytes": memory.get("MemTotal"),
                        "host_available_bytes": memory.get("MemAvailable"),
                        "load_1m": os.getloadavg()[0],
                        "gpu_util_pct": gpu.get("gpu_util_pct", board.get("gpu_util_pct")),
                        "gpu_temp_c": gpu.get("gpu_temp_c", board.get("gpu_temp_c")),
                        "power_w": gpu["power_mw"] / 1000
                        if "power_mw" in gpu
                        else board.get("power_w"),
                        "source": "tegrastats + procfs" if fresh else board.get("source", "procfs"),
                        "gpu_availability": "measured" if fresh or board else "not_collected",
                        "power_scope": "SoC VDD_IN" if fresh else board.get("power_scope", "unavailable"),
                        **{key: board[key] for key in ("gpu_uuid", "gpu_name", "sensor_device", "gpu_memory_used_bytes", "gpu_memory_total_bytes") if key in board},
                        "memory_scope": "control_host",
                    }
                )
            self.stop_event.wait(1)

    def snapshot(self):
        with self.lock:
            return {
                "host": platform.node(),
                "architecture": platform.machine(),
                "samples": list(self.samples),
                "sample_interval_s": 1,
                "scope": "control_host",
                "not_remote_provider_telemetry": True,
            }

    def close(self):
        self.stop_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for thread in self.threads:
            thread.join(timeout=3)
