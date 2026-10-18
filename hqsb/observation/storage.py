"""Bounded asynchronous evidence streams; supervisor publication is separate."""

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path

from hqsb.observation.contracts import SCHEMA_VERSION


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, body):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class EvidenceWriter:
    def __init__(
        self,
        directory,
        context,
        *,
        max_records=100_000,
        max_bytes=128 * 1024 * 1024,
        chunk_bytes=8 * 1024 * 1024,
        queue_records=256,
    ):
        self.directory = Path(directory)
        if not self.directory.is_absolute() or self.directory.is_symlink():
            raise ValueError("EVIDENCE_DIRECTORY_INVALID")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.raw = self.directory / "observations"
        self.raw.mkdir(mode=0o700, exist_ok=False)
        self.context = context
        self.max_records, self.max_bytes, self.chunk_bytes = (
            max_records,
            max_bytes,
            chunk_bytes,
        )
        if type(queue_records) is not int or not 1 <= queue_records <= 4096:
            raise ValueError("EVIDENCE_QUEUE_BUDGET_INVALID")
        self.queue = queue.Queue(maxsize=queue_records)
        self.queue_high_watermark = 0
        self.lock = threading.Lock()
        self.sequence = self.accepted = self.written = self.dropped = self.bytes = 0
        self.artifacts, self.error, self.closed = [], None, False
        self._report = None
        self.thread = threading.Thread(
            target=self._drain, daemon=True, name="hqsb-evidence"
        )
        self.thread.start()

    def offer(self, kind, payload):
        with self.lock:
            if self.closed:
                return "closed"
            self.sequence += 1
            if self.accepted >= self.max_records or self.error:
                self.dropped += 1
                return "budget_exceeded" if not self.error else "io_failed"
            try:
                self.queue.put_nowait(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "seq": self.sequence,
                        "kind": kind,
                        "payload": payload,
                    }
                )
            except queue.Full:
                self.dropped += 1
                return "queue_full"
            self.accepted += 1
            self.queue_high_watermark = max(self.queue_high_watermark, self.queue.qsize())
            return "accepted"

    def _drain(self):
        stream, path, count = None, None, 0
        try:
            while True:
                record = self.queue.get()
                if record is None:
                    break
                if self.error:
                    continue
                encoded = (
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                ).encode()
                if len(encoded) > 32 * 1024 * 1024:
                    self.error = "EVIDENCE_RECORD_BUDGET"
                    continue
                if self.bytes + len(encoded) > self.max_bytes:
                    self.error = "EVIDENCE_BYTE_BUDGET"
                    continue
                if stream is None or stream.tell() + len(encoded) > self.chunk_bytes:
                    if stream:
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                        self._register(path)
                    count += 1
                    path = self.raw / f"records-{count:06d}.jsonl"
                    stream = path.open("xb")
                stream.write(encoded)
                self.bytes += len(encoded)
                self.written += 1
        except (OSError, ValueError, TypeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            if stream and not stream.closed:
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                    self._register(path)
                except OSError as exc:
                    self.error = str(exc)[:400]

    def _register(self, path):
        self.artifacts.append(
            {
                "path": str(path.relative_to(self.directory)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    def close(self, *, execution_status, cleanup_status, source_problems=None):
        with self.lock:
            if self._report is not None:
                return self._report
            self.closed = True
        # Called outside the model step. A dead writer must not deadlock shutdown.
        deadline = time.monotonic() + 30
        while self.thread.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            raise RuntimeError("EVIDENCE_WRITER_CLEANUP_UNKNOWN")
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": self.context,
            "execution_status": execution_status,
            "cleanup_status": cleanup_status,
            "capture_status": "partial"
            if self.error
            or self.dropped
            or source_problems
            or self.written != self.accepted
            or cleanup_status != "succeeded"
            else "complete",
            "counts": {
                "attempted": self.sequence,
                "accepted": self.accepted,
                "written": self.written,
                "dropped": self.dropped,
                "accepted_not_written": self.accepted - self.written,
            },
            "error": self.error,
            "queue": {"capacity_records": self.queue.maxsize, "high_watermark_records": self.queue_high_watermark},
            "artifacts": self.artifacts,
            "source_problems": source_problems or [],
            "publication": "awaiting_supervisor",
        }
        atomic_json(self.raw / "capture-manifest.json", report)
        self._report = report
        return report


class EvidenceReader:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.raw = self.directory / "observations"
        if self.raw.is_symlink():
            raise ValueError("EVIDENCE_PATH_ESCAPE")

    def manifest(self, *, pending=False):
        path = self.raw / ("capture-manifest.json" if pending else "manifest.json")
        if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("EVIDENCE_MANIFEST_INVALID")
        body = json.loads(path.read_text())
        if not isinstance(body, dict) or body.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("EVIDENCE_SCHEMA_UNSUPPORTED")
        if (
            not isinstance(body.get("context"), dict)
            or not isinstance(body.get("counts"), dict)
            or not isinstance(body.get("artifacts"), list)
        ):
            raise ValueError("EVIDENCE_MANIFEST_INVALID")
        written = body["counts"].get("written")
        if (
            type(written) is not int
            or not 0 <= written <= 100_000
            or len(body["artifacts"]) > 100_000
        ):
            raise ValueError("EVIDENCE_MANIFEST_INVALID")
        for artifact in body["artifacts"]:
            if (
                not isinstance(artifact, dict)
                or not isinstance(artifact.get("path"), str)
                or type(artifact.get("bytes")) is not int
                or not 0 <= artifact["bytes"] <= 512 * 1024**2
                or not isinstance(artifact.get("sha256"), str)
                or len(artifact["sha256"]) != 64
            ):
                raise ValueError("EVIDENCE_MANIFEST_INVALID")
        return body

    def records(self, *, pending=False):
        manifest = self.manifest(pending=pending)
        total, sequence, count = 0, 0, 0
        for artifact in manifest["artifacts"]:
            path = self.directory / artifact["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(
                self.raw.resolve()
            ):
                raise ValueError("EVIDENCE_PATH_ESCAPE")
            total += artifact["bytes"]
            if (
                total > 512 * 1024 * 1024
                or path.stat().st_size != artifact["bytes"]
                or sha256(path) != artifact["sha256"]
            ):
                raise ValueError("EVIDENCE_HASH_MISMATCH")
            with path.open() as stream:
                for line in stream:
                    if len(line) > 32 * 1024 * 1024:
                        raise ValueError("EVIDENCE_RECORD_BUDGET")
                    row = json.loads(line)
                    if (
                        not isinstance(row, dict)
                        or row.get("schema_version") != SCHEMA_VERSION
                        or type(row.get("seq")) is not int
                        or row["seq"] <= sequence
                        or not isinstance(row.get("kind"), str)
                        or not isinstance(row.get("payload"), dict)
                    ):
                        raise ValueError("EVIDENCE_RECORD_INVALID")
                    sequence, count = row["seq"], count + 1
                    if count > manifest["counts"]["written"]:
                        raise ValueError("EVIDENCE_COUNT_MISMATCH")
                    yield row
        if count != manifest["counts"]["written"]:
            raise ValueError("EVIDENCE_COUNT_MISMATCH")


def publish(directory, *, expected_run_id=None, supervisor_failure=None):
    """Only the caller with a completed worker acknowledgement publishes."""
    reader = EvidenceReader(directory)
    if (reader.raw / "manifest.json").exists():
        body = reader.manifest()
    else:
        body = reader.manifest(pending=True)
        # Verify every retained stream before publishing the durable terminal.
        count = sum(1 for _ in reader.records(pending=True))
        if count != body["counts"]["written"]:
            raise ValueError("EVIDENCE_COUNT_MISMATCH")
        if expected_run_id and body["context"].get("run_id") != expected_run_id:
            raise ValueError("EVIDENCE_RUN_MISMATCH")
        if supervisor_failure:
            body.update(capture_status="partial", supervisor_failure=supervisor_failure)
        body["publication"] = "supervisor_acknowledged"
        atomic_json(reader.raw / "manifest.json", body)
    if expected_run_id and body["context"].get("run_id") != expected_run_id:
        raise ValueError("EVIDENCE_RUN_MISMATCH")
    return body
