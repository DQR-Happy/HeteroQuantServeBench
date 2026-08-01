"""Privilege bridge tests use pipes/fake providers; never sudo or GPU execution."""

import io
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hqsb.console.privileged_worker import PrivilegedActor, ProcessHandle
from hqsb.console.worker_stdio import (
    PipeReader,
    ProtocolError,
    ProtocolWriter,
    controlled_directory,
    encode_message,
    operation_directories,
    read_message,
    restore_ownership,
    serve,
)


RUN = "run_" + "a" * 32


@pytest.mark.parametrize("raw", [b"[]\n", b"{\n", b"{}\n", b'{"kind":"load"}'])
def test_protocol_rejects_invalid_messages(raw):
    with pytest.raises(ProtocolError):
        read_message(io.BytesIO(raw))


def test_message_size_bound_and_nonfinite_numbers(monkeypatch):
    monkeypatch.setattr("hqsb.console.worker_stdio.MAX_MESSAGE_BYTES", 32)
    with pytest.raises(ProtocolError, match="8 MiB"):
        encode_message({"kind": "output", "text": "x" * 40})
    with pytest.raises(ProtocolError, match="oversized"):
        read_message(io.BytesIO(b"x" * 33 + b"\n"))
    with pytest.raises(ValueError):
        encode_message({"kind": "output", "value": float("nan")})
    assert read_message(io.BytesIO()) is None


def test_unicode_protocol_roundtrip():
    message = {"kind": "output", "text": "访存分析"}
    assert read_message(io.BytesIO(encode_message(message))) == message


def test_pipe_reader_preserves_multiple_lines_and_partial_eof():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b'{"kind":"load"}\n{"kind":"cancel"}\npartial')
        os.close(write_fd)
        write_fd = None
        reader = PipeReader(read_fd)
        assert read_message(reader) == {"kind": "load"}
        assert read_message(reader) == {"kind": "cancel"}
        with pytest.raises(ProtocolError, match="unterminated"):
            read_message(reader)
        assert read_message(reader) is None
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def test_artifact_roots_and_symlinks_are_rejected(tmp_path):
    root = tmp_path / "captures"
    root.mkdir()
    assert controlled_directory(root / RUN, root) == root / RUN
    with pytest.raises(ProtocolError):
        controlled_directory(tmp_path / RUN, root)
    with pytest.raises(ProtocolError):
        controlled_directory(root / "../../etc", root)
    with pytest.raises(ProtocolError):
        controlled_directory(root / "arbitrary-name", root)
    (root / RUN).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ProtocolError, match="Symlink"):
        controlled_directory(root / RUN, root)


def test_command_cannot_smuggle_an_artifact_path(tmp_path):
    roots = {name: str(tmp_path / name) for name in ("captures", "artifacts")}
    with pytest.raises(ProtocolError, match="Unexpected"):
        operation_directories(
            {
                "kind": "load",
                "payload": {"capture_dir": str(tmp_path / "captures" / RUN)},
            },
            roots,
        )
    with pytest.raises(ProtocolError, match="requires"):
        operation_directories({"kind": "quantize", "payload": {}}, roots)


def test_ownership_walk_never_follows_symlinks_or_hardlinks(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: calls.append((uid, gid)))
    safe = tmp_path / RUN
    safe.mkdir()
    (safe / "trace.json").write_text("{}")
    restore_ownership(safe, 1000, 1000)
    assert len(calls) == 2
    (safe / "escape").symlink_to(tmp_path)
    with pytest.raises(ProtocolError, match="symlink"):
        restore_ownership(safe, 1000, 1000)
    (safe / "escape").unlink()
    os.link(safe / "trace.json", safe / "linked")
    with pytest.raises(ProtocolError, match="hard-linked"):
        restore_ownership(safe, 1000, 1000)


@pytest.mark.parametrize("replacement", ["fifo", "hardlink"])
def test_ownership_revalidates_open_file_without_fifo_blocking(
    tmp_path, monkeypatch, replacement
):
    safe = tmp_path / RUN
    safe.mkdir()
    artifact = safe / "trace.json"
    artifact.write_text("{}")
    original_open = os.open

    def raced_open(path, flags, **kwargs):
        if path == "trace.json":
            assert flags & os.O_NONBLOCK
            if replacement == "fifo":
                artifact.unlink()
                os.mkfifo(artifact)
            else:
                os.link(artifact, safe / "second-link")
        return original_open(path, flags, **kwargs)

    monkeypatch.setattr(os, "open", raced_open)
    monkeypatch.setattr(
        os, "fchown", lambda *_: pytest.fail("changed artifact must not be chowned")
    )
    with pytest.raises(ProtocolError):
        restore_ownership(safe, 1000, 1000)


class FakeProvider:
    def __init__(self):
        self.started = threading.Event()
        self.closed = False

    def load(self):
        return {"ready": True}

    def generate(self, payload, cancelled):
        self.started.set()
        yield {"kind": "output", "text": "真实协议，模拟提供者"}
        assert cancelled.wait(2), "cancel message was not read while generation ran"
        yield {"kind": "result", "finish_reason": "cancelled", "metrics": {}}

    def close(self):
        self.closed = True


class QueueWriter:
    def __init__(self):
        self.events = queue.Queue()

    def send(self, event):
        self.events.put(event)


def test_reader_delivers_cancel_without_waiting_for_model(tmp_path):
    read_fd, write_fd = os.pipe()
    source, sink = os.fdopen(read_fd, "rb"), os.fdopen(write_fd, "wb")
    provider, writer, failures = FakeProvider(), QueueWriter(), []

    def run():
        try:
            serve(
                provider,
                source,
                writer,
                {"captures": str(tmp_path), "artifacts": str(tmp_path)},
            )
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    protocol = ProtocolWriter(sink)
    try:
        protocol.send({"kind": "generate", "payload": {}})
        assert provider.started.wait(2)
        assert writer.events.get(timeout=2)["kind"] == "output"
        protocol.send({"kind": "cancel"})
        result = writer.events.get(timeout=2)
        assert result["kind"] == "done"
        assert result["result"]["finish_reason"] == "cancelled"
        protocol.send({"kind": "close"})
        thread.join(timeout=2)
        assert not thread.is_alive() and provider.closed and not failures
    finally:
        sink.close()
        thread.join(timeout=3)
        source.close()


def test_unknown_command_is_not_executed(tmp_path):
    provider = FakeProvider()
    with pytest.raises(ProtocolError, match="Unsupported"):
        serve(
            provider,
            io.BytesIO(encode_message({"kind": "shell", "payload": {"cmd": "id"}})),
            QueueWriter(),
            {},
        )
    assert not provider.started.is_set()
    assert provider.closed


def test_disconnected_control_pipe_notifies_guard():
    provider, disconnected = FakeProvider(), threading.Event()
    serve(provider, io.BytesIO(), QueueWriter(), {}, disconnected=disconnected.set)
    assert disconnected.is_set() and provider.closed


def test_malformed_control_pipe_also_notifies_guard():
    provider, disconnected = FakeProvider(), threading.Event()
    with pytest.raises(ProtocolError):
        serve(
            provider,
            io.BytesIO(b"invalid\n"),
            QueueWriter(),
            {},
            disconnected=disconnected.set,
        )
    assert disconnected.is_set() and provider.closed


class FakePopen:
    def __init__(self):
        self.pid = 200
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(
            encode_message({"kind": "ready", "pid": 201, "uid": 0})
        )
        self.exited = False
        self.calls = []

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout):
        self.exited = True
        return 0


def test_fixed_spawn_command_paths_and_actual_worker_pid(tmp_path, monkeypatch):
    fake, seen = FakePopen(), {}

    def popen(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return fake

    monkeypatch.setattr(subprocess, "Popen", popen)
    config = {
        "provider": "pytorch",
        "model_path": "~/models/model",
        "manifest": "~/manifest.txt",
    }
    actor = PrivilegedActor(config, data_dir=tmp_path)
    actor._start()
    try:
        assert seen["command"] == [
            "sudo",
            "-n",
            "--",
            sys.executable,
            "-m",
            "hqsb.console.worker_stdio",
        ]
        assert seen["kwargs"]["cwd"] == Path(__file__).resolve().parents[3]
        assert "shell" not in seen["kwargs"]
        assert actor.process.pid == 201 and actor.process.is_alive()
        init = json.loads(fake.stdin.getvalue().splitlines()[0])
        assert Path(init["config"]["model_path"]).is_absolute()
        assert "~" not in init["config"]["manifest"]
        assert init["artifact_roots"]["captures"] == str(tmp_path / "captures")
        assert config["model_path"].startswith("~")
    finally:
        actor.close()
    assert fake.exited and actor.process is None


def test_privilege_bridge_rejects_remote_provider():
    with pytest.raises(ValueError, match="local PyTorch"):
        PrivilegedActor({"provider": "openai"})


def test_unresponsive_worker_close_terminates_then_kills():
    fake = FakePopen()

    def wait(timeout):
        fake.calls.append("wait")
        if fake.calls.count("wait") < 4:
            raise subprocess.TimeoutExpired("sudo", timeout)
        fake.exited = True

    fake.wait = wait
    fake.terminate = lambda: fake.calls.append("terminate")
    fake.kill = lambda: fake.calls.append("kill")
    actor = PrivilegedActor({"provider": "pytorch", "model_path": "/model"})
    actor.process = ProcessHandle(fake)
    actor.close()
    assert fake.calls == ["wait", "wait", "terminate", "wait", "kill", "wait"]


def test_actor_timeout_sends_one_cancel_and_preserves_cleanup_result(monkeypatch):
    actor = PrivilegedActor({"provider": "pytorch", "model_path": "/model"})
    actor.process = ProcessHandle(FakePopen())
    actor._messages = queue.Queue()
    actor._messages.put({"kind": "output", "text": "part"})
    actor._messages.put(
        {"kind": "done", "result": {"finish_reason": "cancelled", "metrics": {}}}
    )
    commands, outputs = [], []
    monkeypatch.setattr(actor, "_send", commands.append)
    result = actor.execute("generate", {}, outputs.append, lambda: False, timeout_s=0)
    assert [row["kind"] for row in commands] == ["generate", "cancel"]
    assert result["finish_reason"] == "timed_out"
    assert len(outputs) == 1
    assert "worker_terminated" not in result
    actor.close()


def test_broken_pipe_during_close_does_not_skip_state_reset():
    class BrokenPipe(io.BytesIO):
        def close(self):
            super().close()
            raise BrokenPipeError("peer exited")

    fake = FakePopen()
    fake.stdin = BrokenPipe()
    actor = PrivilegedActor({"provider": "pytorch", "model_path": "/model"})
    actor.process = ProcessHandle(fake)
    actor.close()
    assert actor.process is None and fake.stdout.closed


def test_permission_denied_poison_prevents_second_worker():
    fake = FakePopen()

    def wait(timeout):
        raise subprocess.TimeoutExpired("sudo", timeout)

    def denied():
        raise PermissionError("signal denied")

    fake.wait, fake.terminate, fake.kill = wait, denied, denied
    actor = PrivilegedActor({"provider": "pytorch", "model_path": "/model"})
    actor.process = ProcessHandle(fake)
    with pytest.raises(RuntimeError, match="Cannot confirm privileged worker PID"):
        actor.close()
    assert actor.process is not None
    with pytest.raises(RuntimeError, match="refusing further device work"):
        actor.execute("load", {}, lambda _: None, lambda: False, 1)
    fake.exited = True
    actor.close()
    assert actor.process is None and actor._poisoned is None
