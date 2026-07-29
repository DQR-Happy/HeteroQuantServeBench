"""Opt-in stdio actor for Tegra CUPTI's privileged device-worker requirement.

The API stays unprivileged. The executable and module are fixed here; browser
requests cannot select commands, environment variables, or filesystem roots.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from hqsb.console.worker_stdio import (
    COMMANDS,
    ProtocolError,
    encode_message,
    read_message,
)


class ProcessHandle:
    """Expose the actual worker PID while lifecycle follows our sudo process."""

    def __init__(self, process):
        self.child = process
        self.pid = process.pid

    def is_alive(self):
        return self.child.poll() is None


class PrivilegedActor:
    def __init__(self, config, *, data_dir=None):
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir or self.project_root / ".console").resolve()
        self.config = dict(config)
        if self.config.get("provider") != "pytorch":
            raise ValueError(
                "Privileged worker only supports local PyTorch deployments"
            )
        for key in ("model_path", "manifest"):
            if self.config.get(key):
                self.config[key] = str(
                    Path(os.path.expandvars(self.config[key])).expanduser().resolve()
                )
        self.process = None
        self._messages = None
        self._reader = None
        self._reader_stop = None
        self._write_lock = threading.Lock()
        self._execute_lock = threading.Lock()
        self._poisoned = None

    def _send(self, message):
        raw = encode_message(message)
        with self._write_lock:
            if self.process is None or self.process.child.stdin is None:
                raise RuntimeError("Device worker is not connected")
            self.process.child.stdin.write(raw)
            self.process.child.stdin.flush()

    def _start(self):
        self.close()
        process = subprocess.Popen(
            ["sudo", "-n", "--", sys.executable, "-m", "hqsb.console.worker_stdio"],
            cwd=self.project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            close_fds=True,
        )
        self.process = ProcessHandle(process)
        messages, stopping = queue.Queue(maxsize=1), threading.Event()
        self._messages, self._reader_stop = messages, stopping

        def read_output():
            try:
                while not stopping.is_set():
                    message = read_message(process.stdout)
                    if message is None:
                        raise RuntimeError(
                            "Privileged worker disconnected; inspect server stderr"
                        )
                    while not stopping.is_set():
                        try:
                            messages.put(message, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                while not stopping.is_set():
                    try:
                        messages.put(
                            {"kind": "error", "message": str(exc)[:800]}, timeout=0.1
                        )
                        break
                    except queue.Full:
                        continue

        self._reader = threading.Thread(
            target=read_output, name="hqsb-privileged-output", daemon=True
        )
        self._reader.start()
        package_paths = list(
            dict.fromkeys(
                str(Path(value).absolute())
                for value in sys.path
                if value
                and Path(value).is_dir()
                and Path(value).name in {"site-packages", "dist-packages"}
            )
        )
        try:
            self._send(
                {
                    "kind": "init",
                    "config": self.config,
                    "runtime_paths": package_paths,
                    "artifact_roots": {
                        name: str(self.data_dir / name)
                        for name in ("captures", "artifacts")
                    },
                }
            )
            ready = messages.get(timeout=30)
            if ready.get("kind") != "ready" or ready.get("uid") != 0:
                raise RuntimeError(
                    ready.get("message", "Privileged worker initialization failed")
                )
            if not isinstance(ready.get("pid"), int) or ready["pid"] <= 1:
                raise ProtocolError("Invalid worker PID in handshake")
            self.process.pid = ready["pid"]
        except Exception:
            self.close()
            raise

    def execute(self, kind, payload, on_output, is_cancelled, timeout_s):
        if self._poisoned is not None:
            raise RuntimeError(self._poisoned)
        if kind not in COMMANDS - {"cancel", "close"}:
            raise ValueError("Unsupported worker operation")
        if not self._execute_lock.acquire(blocking=False):
            raise RuntimeError("Device actor operations must be serialized")
        try:
            if self.process is None or not self.process.is_alive():
                self._start()
            self._send({"kind": kind, "payload": payload})
            start, cancelled_at = time.monotonic(), None
            while True:
                timed_out = time.monotonic() - start >= timeout_s
                if (is_cancelled() or timed_out) and cancelled_at is None:
                    self._send({"kind": "cancel"})
                    cancelled_at = time.monotonic()
                if cancelled_at is not None and time.monotonic() - cancelled_at > 10:
                    self.close()
                    return {
                        "finish_reason": "timed_out" if timed_out else "cancelled",
                        "worker_terminated": True,
                        "metrics": {},
                    }
                try:
                    event = self._messages.get(timeout=0.1)
                except queue.Empty:
                    if not self.process.is_alive():
                        raise RuntimeError(
                            "Privileged device worker exited unexpectedly"
                        )
                    continue
                event_kind = event.get("kind")
                if event_kind == "done":
                    result = event["result"]
                    if timed_out and result.get("finish_reason") == "cancelled":
                        result["finish_reason"] = "timed_out"
                    if kind == "unload":
                        self.close()
                    return result
                if event_kind == "error":
                    raise RuntimeError(event.get("message", "Device worker failed"))
                if event_kind not in {"output", "progress"}:
                    raise ProtocolError("Unexpected device event")
                on_output(event)
        finally:
            self._execute_lock.release()

    def close(self):
        handle = self.process
        if handle is None:
            return
        process = handle.child
        signal_errors = []

        def wait(seconds):
            try:
                process.wait(timeout=seconds)
                return True
            except subprocess.TimeoutExpired:
                return False

        try:
            if handle.is_alive():
                try:
                    self._send({"kind": "close"})
                except (OSError, ValueError, RuntimeError):
                    pass
                if not wait(5):
                    # EOF makes the child terminate itself, even where an
                    # unprivileged API cannot signal a root-owned sudo monitor.
                    try:
                        process.stdin.close()
                    except (OSError, ValueError):
                        pass
                    if not wait(2):
                        for action in (process.terminate, process.kill):
                            try:
                                action()
                            except ProcessLookupError:
                                pass
                            except PermissionError as exc:
                                signal_errors.append(str(exc))
                            if wait(5):
                                break
        finally:
            dead = process.poll() is not None
            try:
                if self._reader_stop is not None:
                    self._reader_stop.set()
                for stream in (process.stdin, process.stdout):
                    if stream is None:
                        continue
                    try:
                        if dead:
                            stream.close()
                        else:
                            # Do not acquire a BufferedReader lock held by a
                            # thread waiting on a child we could not terminate.
                            os.close(stream.fileno())
                    except (OSError, ValueError):
                        pass
                if self._reader is not None:
                    self._reader.join(timeout=1)
            finally:
                self._messages = self._reader = self._reader_stop = None
                self.process = None if dead else handle
                self._poisoned = (
                    None
                    if dead
                    else (
                        f"Cannot confirm privileged worker PID {handle.pid} stopped "
                        f"(sudo PID {process.pid}); refusing further device work. "
                        + "; ".join(signal_errors)
                    )
                )
        if self._poisoned is not None:
            raise RuntimeError(self._poisoned)
