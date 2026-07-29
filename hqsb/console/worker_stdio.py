"""Opt-in privileged device child: bounded JSON over inherited pipes only.

No network listener, shell, user-selected executable, or general command runner
is exposed. Configuration comes from the administrator-owned API process.
"""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import stat
import sys
import threading
from pathlib import Path

MAX_MESSAGE_BYTES = 8 * 1024 * 1024
OPERATIONS = frozenset({"load", "generate", "quantize", "unload"})
COMMANDS = OPERATIONS | {"cancel", "close"}
RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")


class ProtocolError(ValueError):
    pass


class PipeReader:
    """Bounded lines without a BufferedReader lock during daemon shutdown."""

    def __init__(self, fd):
        self.fd = fd
        self.pending = bytearray()

    def readline(self, limit):
        while True:
            newline = self.pending.find(b"\n", 0, limit)
            if newline >= 0 or len(self.pending) >= limit:
                size = newline + 1 if newline >= 0 else limit
                result = bytes(self.pending[:size])
                del self.pending[:size]
                return result
            chunk = os.read(self.fd, min(65536, limit - len(self.pending)))
            if not chunk:
                result = bytes(self.pending)
                self.pending.clear()
                return result
            self.pending.extend(chunk)


def encode_message(message):
    raw = (
        json.dumps(
            message, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Worker message exceeds the 8 MiB protocol budget")
    return raw


def read_message(stream):
    raw = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not raw:
        return None
    if len(raw) > MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
        raise ProtocolError("Worker message is oversized or unterminated")
    try:
        message = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("Worker message is not valid JSON") from exc
    if not isinstance(message, dict) or not isinstance(message.get("kind"), str):
        raise ProtocolError("Worker message must be an object with a kind")
    return message


class ProtocolWriter:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()

    def send(self, message):
        raw = encode_message(message)
        with self.lock:
            self.stream.write(raw)
            self.stream.flush()


def controlled_directory(path, root):
    """Only one server-created run directory directly under a pinned root."""
    candidate, parent = Path(path), Path(root)
    if not candidate.is_absolute() or not parent.is_absolute():
        raise ProtocolError("Artifact paths must be absolute")
    if candidate.parent != parent or not RUN_ID.fullmatch(candidate.name):
        raise ProtocolError("Artifact path is outside the pinned run directory")
    # Reject symlinks in every ancestor before accepting a privileged write.
    for part in (candidate, *candidate.parents):
        if part.is_symlink():
            raise ProtocolError("Symlink in privileged artifact path")
    if candidate.resolve() != candidate:
        raise ProtocolError("Artifact path must be normalized")
    return candidate


def operation_directories(command, roots):
    payload = command.get("payload") or {}
    if not isinstance(payload, dict):
        raise ProtocolError("Operation payload must be an object")
    directories = []
    for key, kind, root in (
        ("capture_dir", "generate", "captures"),
        ("artifact_dir", "quantize", "artifacts"),
    ):
        if key in payload:
            if command["kind"] != kind:
                raise ProtocolError("Unexpected artifact path for this operation")
            directories.append(controlled_directory(payload[key], roots[root]))
    if command["kind"] == "quantize" and not directories:
        raise ProtocolError("Quantization requires its controlled artifact path")
    return directories


def restore_ownership(path, uid, gid):
    """Use open descriptors: never follow symlinks or modify linked files."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for part in Path(path).parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child

        def visit(directory_fd):
            for name in os.listdir(directory_fd):
                entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(entry.st_mode):
                    raise ProtocolError(
                        "Refusing symlink during artifact ownership restoration"
                    )
                if not stat.S_ISDIR(entry.st_mode) and not stat.S_ISREG(entry.st_mode):
                    raise ProtocolError("Refusing special artifact file")
                if stat.S_ISREG(entry.st_mode) and entry.st_nlink != 1:
                    raise ProtocolError("Refusing hard-linked artifact file")
                file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if stat.S_ISDIR(entry.st_mode):
                    file_flags |= os.O_DIRECTORY
                child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                try:
                    actual = os.fstat(child_fd)
                    if (actual.st_dev, actual.st_ino) != (entry.st_dev, entry.st_ino):
                        raise ProtocolError(
                            "Artifact changed during ownership restoration"
                        )
                    if not stat.S_ISDIR(actual.st_mode) and not stat.S_ISREG(
                        actual.st_mode
                    ):
                        raise ProtocolError("Refusing special artifact file")
                    if stat.S_ISREG(actual.st_mode) and actual.st_nlink != 1:
                        raise ProtocolError("Refusing hard-linked artifact file")
                    if stat.S_ISDIR(actual.st_mode):
                        visit(child_fd)
                    else:
                        os.fchown(child_fd, uid, gid)
                finally:
                    os.close(child_fd)
            os.fchown(directory_fd, uid, gid)

        visit(descriptor)
    except FileNotFoundError:
        # Cancelled jobs remove their private .partial directories themselves.
        if Path(path).exists():
            raise
    finally:
        os.close(descriptor)


def serve(provider, source, writer, roots, *, owner=None, disconnected=None):
    """Serial provider execution with an independent cancellation reader."""
    pending = queue.Queue(maxsize=1)
    cancelled, stopping, busy = threading.Event(), threading.Event(), threading.Event()
    protocol_errors = []

    def receive():
        try:
            # Keep reading after a graceful close request. If the provider hangs
            # while draining, the parent's pipe EOF remains a force-stop channel.
            while True:
                command = read_message(source)
                if command is None:
                    cancelled.set()
                    stopping.set()
                    if disconnected is not None:
                        disconnected()
                    return
                kind = command["kind"]
                if kind not in COMMANDS:
                    raise ProtocolError("Unsupported worker command")
                if kind in {"cancel", "close"}:
                    cancelled.set()
                    if kind == "close":
                        stopping.set()
                    continue
                if stopping.is_set():
                    raise ProtocolError("Worker is already closing")
                if busy.is_set():
                    raise ProtocolError("Worker accepts only one serial operation")
                operation_directories(command, roots)
                cancelled.clear()
                busy.set()
                pending.put_nowait(command)
        except Exception as exc:
            protocol_errors.append(exc)
            cancelled.set()
            stopping.set()
            if disconnected is not None:
                # No command reader means EOF can no longer force a hung native
                # call to stop. Terminate before attempting any potentially
                # blocked protocol write; this is a lost-control condition.
                disconnected()

    reader = threading.Thread(target=receive, name="hqsb-worker-commands", daemon=True)
    reader.start()
    try:
        while not stopping.is_set():
            try:
                command = pending.get(timeout=0.1)
            except queue.Empty:
                continue
            directories = operation_directories(command, roots)
            failure = None
            result = {}
            try:
                kind = command["kind"]
                if kind == "load":
                    result = provider.load()
                elif kind == "unload":
                    provider.close()
                else:
                    for event in getattr(provider, kind)(
                        command.get("payload") or {}, cancelled
                    ):
                        if event["kind"] == "result":
                            result = event
                        else:
                            writer.send(event)
            except Exception as exc:
                failure = exc
            finally:
                if owner is not None:
                    try:
                        for directory in directories:
                            restore_ownership(directory, *owner)
                    except Exception as exc:
                        failure = failure or exc
                busy.clear()
            if failure is not None:
                writer.send(
                    {
                        "kind": "error",
                        "message": f"{type(failure).__name__}: {failure}"[:800],
                    }
                )
            else:
                writer.send({"kind": "done", "result": result})
            if command["kind"] == "unload":
                break
        if protocol_errors:
            raise protocol_errors[0]
    finally:
        stopping.set()
        cancelled.set()
        provider.close()


def main():
    # Preserve only the protocol descriptor, redirect even C/C++ stdout to stderr.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    writer = ProtocolWriter(os.fdopen(protocol_fd, "wb"))
    source = PipeReader(sys.stdin.fileno())
    try:
        if not sys.platform.startswith("linux") or os.geteuid() != 0:
            raise ProtocolError("Privileged worker requires explicit sudo on Linux")
        import ctypes

        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() == 1:
            raise ProtocolError("Cannot establish worker parent lifetime guard")
        init = read_message(source)
        if init is None or init["kind"] != "init":
            raise ProtocolError("Worker requires administrator initialization")
        for value in init.get("runtime_paths", []):
            path = Path(value)
            if (
                not path.is_absolute()
                or path.name not in {"site-packages", "dist-packages"}
                or not path.is_dir()
            ):
                raise ProtocolError("Invalid administrator runtime package path")
            if str(path) not in sys.path:
                sys.path.append(str(path))
        from hqsb.console.config import DeploymentConfig
        from hqsb.backends.interactive import InteractivePyTorch

        config = DeploymentConfig.model_validate(init["config"])
        if config.provider != "pytorch" or not Path(config.model_path).is_absolute():
            raise ProtocolError(
                "Privileged worker only supports an absolute local PyTorch model"
            )
        roots = init["artifact_roots"]
        if set(roots) != {"captures", "artifacts"}:
            raise ProtocolError("Expected fixed capture and artifact roots")
        owner = (int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
        if owner[0] <= 0 or owner[1] < 0:
            raise ProtocolError("Console API must run as a non-root user")
        writer.send({"kind": "ready", "pid": os.getpid(), "uid": os.geteuid()})
        serve(
            InteractivePyTorch(config.model_dump()),
            source,
            writer,
            roots,
            owner=owner,
            disconnected=lambda: os.kill(os.getpid(), signal.SIGTERM),
        )
    except Exception as exc:
        try:
            writer.send(
                {"kind": "error", "message": f"{type(exc).__name__}: {exc}"[:800]}
            )
        except (BrokenPipeError, OSError):
            pass
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
