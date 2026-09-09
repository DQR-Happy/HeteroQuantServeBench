"""HQSB lifecycle, ownership, policy and process-crash probes on real storage.

Fault instrumentation is confined to fresh spawned processes. Never patches a
running HQSB service. These are contract probes, not production timing results.
"""

import argparse
import json
import multiprocessing
import os
import signal
import threading
import time
import uuid
from pathlib import Path


def crash_child(directory, stage, connection):
    from hqsb.observation.storage import EvidenceWriter

    original_fsync = os.fsync

    def observe(fd):
        name = os.readlink(f"/proc/self/fd/{fd}")
        if "records-" in name:
            if stage == "written_not_fsynced":
                connection.send({"stage": stage, "artifact": name, "fsync_confirmed": False})
                time.sleep(60)
            original_fsync(fd)
            if stage == "fsynced_not_manifested":
                connection.send({"stage": stage, "artifact": name, "fsync_confirmed": True})
                time.sleep(60)
        else:
            original_fsync(fd)

    os.fsync = observe
    gate = threading.Event()

    class GatedWriter(EvidenceWriter):
        def _drain(self):
            gate.wait()
            super()._drain()

    cls = GatedWriter if stage == "accepted_not_written" else EvidenceWriter
    writer = cls(directory, {"run_id": "run_" + uuid.uuid4().hex}, chunk_bytes=1024)
    accepted = 0
    for i in range(50):
        accepted += writer.offer("crash_probe", {"i": i, "data": "x" * 128}) == "accepted"
    if stage == "accepted_not_written":
        connection.send({"stage": stage, "accepted": accepted, "fsync_confirmed": False})
        time.sleep(60)
    report = writer.close(execution_status="succeeded", cleanup_status="succeeded")
    if stage == "supervisor_before_publish":
        connection.send({"stage": stage, "accepted": accepted, "report": report, "fsync_confirmed": True})
        time.sleep(60)


def crash_probe(directory, stage):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=crash_child, args=(str(directory), stage, child))
    process.start()
    try:
        if not parent.poll(15):
            raise TimeoutError("crash stage not reached")
        observed = parent.recv()
        os.kill(process.pid, signal.SIGKILL)
        process.join(5)
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        child.close()
    valid_lines, broken_lines = 0, 0
    for path in (directory / "observations").glob("records-*.jsonl"):
        for line in path.read_bytes().splitlines():
            try:
                json.loads(line)
                valid_lines += 1
            except (ValueError, UnicodeDecodeError):
                broken_lines += 1
    return {"stage": stage, "observed": observed, "recovered_readable_lines": valid_lines,
            "broken_lines": broken_lines, "published_manifest_exists": (directory / "observations/manifest.json").exists(),
            "scope": "SIGKILL process crash; readable un-fsynced bytes do not prove power-loss durability",
            "checkpoint_acknowledged_loss": None,
            "ack_reason": "stock writer has no per-record checkpoint acknowledgement; probe observes internal fsync boundary"}


def basic_probes(root):
    from hqsb.observation.storage import EvidenceReader, EvidenceWriter, publish

    rows = []

    def writer(label, **kwargs):
        path = root / label
        return EvidenceWriter(path, {"run_id": "run_" + uuid.uuid4().hex}, **kwargs), path

    for label, limit, byte_budget in (("record-limit", 10, 1024**2), ("byte-limit", 1000, 1024)):
        w, path = writer(label, max_records=limit, max_bytes=byte_budget)
        results = [w.offer("probe", {"i": i, "data": "x" * 128}) for i in range(50)]
        report = w.close(execution_status="succeeded", cleanup_status="succeeded")
        counts = report["counts"]
        assert counts["accepted"] == counts["written"] + counts["accepted_not_written"]
        assert report["capture_status"] == "partial"
        rows.append({"workload": "W22", "probe": label, "status": "pass", "results": results, "report": report})

    for cycle in range(100):
        w, path = writer(f"lifecycle-{cycle}")
        assert w.offer("probe", {"cycle": cycle}) == "accepted"
        first = w.close(execution_status="succeeded", cleanup_status="succeeded")
        second = w.close(execution_status="succeeded", cleanup_status="succeeded")
        assert first == second
        assert w.offer("late", {}) == "closed"
        assert not w.thread.is_alive()
        assert len(list(EvidenceReader(path).records(pending=True))) == 1
    rows.append({"workload": "W17/W18", "probe": "100-lifecycle-repeated-close-late-offer", "status": "pass"})

    gate = threading.Event()

    class GatedWriter(EvidenceWriter):
        def _drain(self):
            gate.wait()
            super()._drain()

    path = root / "ownership"
    w = GatedWriter(path, {"run_id": "run_" + uuid.uuid4().hex})
    event = {"nested": [1]}
    assert w.offer("probe", event) == "accepted"
    event["nested"][0] = 999
    gate.set()
    w.close(execution_status="succeeded", cleanup_status="succeeded")
    actual = next(EvidenceReader(path).records(pending=True))["payload"]
    rows.append({"workload": "W21", "probe": "mutation-after-offer", "observed": actual,
                 "status": "observed_borrowed" if actual["nested"] == [999] else "observed_frozen",
                 "interpretation": "existing API borrows payload; not a retroactive promise of deep-copy semantics"})

    w, path = writer("integer-utf8")
    event = {"addresses": [2**32-1, 2**32+1, 2**53-1, 2**53+1, 2**63-1, 2**63+1], "name": "推理层µ"}
    w.offer("probe", event)
    w.close(execution_status="succeeded", cleanup_status="succeeded")
    assert next(EvidenceReader(path).records(pending=True))["payload"] == event
    rows.append({"workload": "W30", "probe": "64bit-and-utf8", "status": "pass"})

    w, path = writer("corrupt-hash")
    w.offer("probe", {"data": "original"})
    report = w.close(execution_status="succeeded", cleanup_status="succeeded")
    artifact = path / report["artifacts"][0]["path"]
    artifact.write_bytes(artifact.read_bytes().replace(b"original", b"modified"))
    try:
        publish(path)
        raise AssertionError("corrupt artifact unexpectedly published")
    except ValueError as exc:
        assert "HASH" in str(exc)
    assert not (path / "observations/manifest.json").exists()
    rows.append({"workload": "W20", "probe": "hash-corruption-publication", "status": "pass"})

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = basic_probes(root)
    for stage in ("accepted_not_written", "written_not_fsynced", "fsynced_not_manifested", "supervisor_before_publish"):
        result = crash_probe(root / stage, stage)
        assert not result["published_manifest_exists"]
        rows.append({"workload": "W19", "probe": stage, "status": "observed", "result": result})
        (root / "all-probes.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"probes": len(rows), "status": "completed", "raw": str(root)}))


if __name__ == "__main__":
    main()
