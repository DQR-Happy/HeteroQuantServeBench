"""Jetson tegrastats resource monitor.

Provides real-time GPU/CPU/memory/power/temperature monitoring via
tegrastats on NVIDIA Jetson platforms. The ``TegrastatsMonitor`` class
spawns a background thread to continuously collect system metrics
during benchmark runs.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TegrastatsMonitor:
    """Background tegrastats collector for Jetson resource monitoring.

    Launches ``tegrastats`` as a subprocess and continuously reads
    its output in a daemon thread. Each output line is timestamped
    with ``time.monotonic_ns()`` for precise time alignment with
    benchmark events.

    Typical usage::

        monitor = TegrastatsMonitor(interval_ms=100)
        monitor.start()
        # ... run benchmark ...
        monitor.stop()
        for record in monitor.records:
            print(record["time_ns"], record["raw"])

    Attributes:
        interval_ms: Sampling interval in milliseconds.
        records: List of collected ``{"time_ns": int, "raw": str}``
            dictionaries, populated during monitoring.
    """

    def __init__(self, interval_ms: int = 100) -> None:
        """Initialize the tegrastats monitor.

        Args:
            interval_ms: Sampling interval passed to ``tegrastats --interval``.
                Default: 100 ms.
        """
        self.interval_ms = interval_ms
        self._process: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.records: List[Dict[str, Any]] = []

    @property
    def is_running(self) -> bool:
        """Check if the monitor is currently collecting data."""
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Start tegrastats monitoring in a background thread.

        Raises:
            RuntimeError: If tegrastats is not available or already running.
        """
        if self.is_running:
            raise RuntimeError("TegrastatsMonitor is already running")

        try:
            self._process = subprocess.Popen(
                [
                    "tegrastats",
                    "--interval",
                    str(self.interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "tegrastats not found. This module requires a Jetson platform "
                "with tegrastats installed (part of nvidia-l4t-tools)."
            )

        self._thread = threading.Thread(
            target=self._reader,
            daemon=True,
            name="tegrastats-monitor",
        )
        self._thread.start()
        logger.info("Tegrastats monitor started (interval=%d ms)", self.interval_ms)

    def _reader(self) -> None:
        """Internal thread target: reads tegrastats stdout line by line."""
        assert self._process is not None
        assert self._process.stdout is not None

        try:
            for line in self._process.stdout:
                record: Dict[str, Any] = {
                    "time_ns": time.monotonic_ns(),
                    "raw": line.strip(),
                }
                with self._lock:
                    self.records.append(record)
        except Exception:
            logger.exception("Tegrastats reader thread encountered an error")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop tegrastats monitoring and wait for the reader thread.

        Args:
            timeout: Maximum time in seconds to wait for graceful
                termination. Default: 5.0.
        """
        if self._process is None:
            return

        # Send SIGTERM
        self._process.terminate()

        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("tegrastats did not exit gracefully, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=2)

        if self._thread is not None:
            self._thread.join(timeout=1)

        self._process = None
        self._thread = None

        logger.info(
            "Tegrastats monitor stopped (%d records collected)",
            len(self.records),
        )

    def __enter__(self) -> "TegrastatsMonitor":
        """Context manager entry: starts monitoring."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit: stops monitoring."""
        self.stop()
