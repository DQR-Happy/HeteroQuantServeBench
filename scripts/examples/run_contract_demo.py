#!/usr/bin/env python3
"""Minimal C1/C2 → registry/C4 → C6/C7 walkthrough using synthetic data.

No model or hardware is used. Timings come from DummyBackend constants and
must never be reported as performance measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hqsb.backends.dummy import DummyBackend  # noqa: E402
from hqsb.benchmark.engine import BenchmarkEngine  # noqa: E402
from hqsb.core.contracts import ModelArtifact, WorkloadSpec  # noqa: E402
from hqsb.core.registry import RegistryHub  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional new C6 JSON file; otherwise print")
    args = parser.parse_args()
    model = ModelArtifact(
        model_id="synthetic-demo", source="local", architecture="dummy", dtype="float32"
    )
    workload = WorkloadSpec(
        name="synthetic-demo",
        input_tokens=3,
        token_ids=[1, 2, 3],
        output_tokens=4,
        repetitions=2,
    )
    registry = RegistryHub()
    registry.backends.register("dummy", DummyBackend)
    backend = registry.backends.get("dummy")()
    try:
        result = BenchmarkEngine(backend).run(workload, artifact=model)
        result.summary.update(claim_level="SMOKE", hardware_measurement=False)
        text = result.model_dump_json(indent=2) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
            print(path)
        else:
            print(text, end="")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
