"""Existing HQSB/OTel implementations; no replacement queues or codecs."""
import json
import os
from pathlib import Path

from benchmarks.capture.adapters import HQSBAdapter, OTelAdapter
from benchmarks.capture.worker import RESULTS

adapter = None
directory = None


def initialize(path):
    global adapter, directory
    directory = Path(path).parent
    case = json.loads((directory / "config.json").read_text())
    profile = int(os.environ["HQSB_FAIR_PROFILE"])
    case["chunk_bytes"] = (2, 8, 32)[profile] * 1024**2
    case["max_bytes"] = 1024**3
    case["queue_records"] = (256, 4096, 65536)[profile]
    case["batch_events"] = (64, 512, 4096)[profile]
    cls = HQSBAdapter if case["adapter"] == "hqsb" else OTelAdapter
    adapter = cls(directory, case, None)


def offer(run, producer, sequence, timestamp, body):
    return RESULTS[adapter.offer(dict(run_id=run, producer_id=producer, source_seq=sequence,
                                     source_timestamp=timestamp, body=body))]


def close():
    report = adapter.close()
    (directory / "python-terminal.json").write_text(json.dumps(report, indent=2))
