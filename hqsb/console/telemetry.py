"""Bounded, read-only host telemetry; missing sensors stay missing."""

from __future__ import annotations

import collections
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

from hqsb.benchmark.tegrastats_parser import parse_tegrastats_line


class Telemetry:
    def __init__(self):
        self.samples = collections.deque(maxlen=120)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.process = None
        self.threads = []
        self.gpu = {}
        self.gpu_time = None

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
                        "gpu_util_pct": gpu.get("gpu_util_pct"),
                        "gpu_temp_c": gpu.get("gpu_temp_c"),
                        "power_w": gpu["power_mw"] / 1000
                        if "power_mw" in gpu
                        else None,
                        "source": "tegrastats + procfs" if fresh else "procfs",
                        "gpu_availability": "measured" if fresh else "not_collected",
                        "power_scope": "SoC VDD_IN",
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
