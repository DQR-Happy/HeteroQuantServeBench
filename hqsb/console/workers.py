"""One isolated, serial device worker per deployment; API never imports torch."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time


def worker_main(connection, cancel, config):
    # On Linux an abruptly stopped API must not leave its private GPU worker alive.
    if os.name == "posix":
        try:
            import ctypes

            ctypes.CDLL(None).prctl(1, signal.SIGTERM)
            if os.getppid() == 1:
                return
        except (AttributeError, OSError):
            pass
    from hqsb.backends.interactive import InteractivePyTorch, OpenAIProvider

    provider = (
        InteractivePyTorch(config)
        if config["provider"] == "pytorch"
        else OpenAIProvider(config)
    )
    try:
        while True:
            command = connection.recv()
            try:
                if command["kind"] == "load":
                    result = provider.load()
                elif command["kind"] in {"generate", "quantize"}:
                    result = {}
                    operation = getattr(provider, command["kind"])
                    for event in operation(command["payload"], cancel):
                        if event["kind"] == "result":
                            result = event
                        else:
                            connection.send(event)
                elif command["kind"] == "unload":
                    provider.close()
                    connection.send({"kind": "done", "result": {}})
                    return
                else:
                    raise ValueError("Unknown worker operation")
                # generate() has exited its finally block before cleanup is acknowledged.
                connection.send({"kind": "done", "result": result})
            except Exception as exc:
                connection.send(
                    {
                        "kind": "error",
                        "message": f"{type(exc).__name__}: {str(exc)[:800]}",
                    }
                )
    except (EOFError, BrokenPipeError):
        pass
    finally:
        provider.close()
        connection.close()


class Actor:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.pipe = None
        self.cancel = None

    def execute(self, kind, payload, on_output, is_cancelled, timeout_s):
        if self.process is None or not self.process.is_alive():
            self.close()
            ctx = mp.get_context("spawn")
            self.pipe, child = ctx.Pipe()
            self.cancel = ctx.Event()
            self.process = ctx.Process(
                target=worker_main, args=(child, self.cancel, self.config), daemon=True
            )
            self.process.start()
            child.close()
        self.cancel.clear()
        self.pipe.send({"kind": kind, "payload": payload})
        start, cancelled_at = time.monotonic(), None
        while True:
            timed_out = time.monotonic() - start >= timeout_s
            if is_cancelled() or timed_out:
                self.cancel.set()
                cancelled_at = cancelled_at or time.monotonic()
            if cancelled_at and time.monotonic() - cancelled_at > 10:
                self.close()
                return {
                    "finish_reason": "timed_out" if timed_out else "cancelled",
                    "worker_terminated": True,
                    "metrics": {},
                }
            if self.pipe.poll(0.1):
                event = self.pipe.recv()
                if event["kind"] == "done":
                    result = event["result"]
                    if timed_out and result.get("finish_reason") == "cancelled":
                        result["finish_reason"] = "timed_out"
                    if kind == "unload":
                        self.close()
                    return result
                if event["kind"] == "error":
                    raise RuntimeError(event["message"])
                on_output(event)
            if not self.process.is_alive():
                raise RuntimeError("Device worker exited unexpectedly")

    def close(self):
        if self.process is not None:
            if self.cancel:
                self.cancel.set()
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=5)
            else:
                self.process.join(timeout=1)
            self.process = None
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None
