"""Bounded, auditable Exa discovery. Execute only on the authorized CPU host."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    wrapper = root / "tools/exa-search/scripts/exa_search.py"
    queries = []
    for venue in ("OSDI", "MICRO microarchitecture", "ICLR", "ICML", "NeurIPS NIPS"):
        for year in (2024, 2025, 2026):
            queries.append({"query": f"{venue} {year} low overhead tracing profiling telemetry event collection logging buffers",
                            "category": "publication"})
    queries += [{"query": q, "category": "publication"} for q in (
        "2024 2025 2026 SOSP NSDI ASPLOS EuroSys low overhead tracing telemetry logging event collection",
        "2024 2025 2026 IEEE TPDS ACM TOCS TACO low overhead event tracing telemetry collection",
        "2024 2025 2026 distributed trace lossless compression collection Mint TracePicker profiling buffering",
    )]
    queries += [{"query": q} for q in (
        "2024 2025 2026 high throughput low overhead tracing event collection open source tools",
        "Perfetto SDK streaming trace buffer discard flush write_into_file benchmark",
        "LTTng UST userspace tracer ring buffer per cpu discard blocking benchmark",
        "Quill C++ logging bounded queue options blocking dropping binary format benchmark",
    )]
    if len(sys.argv) > 2 and sys.argv[2] == "--prepare-http":
        directory = root / "research/requests"
        directory.mkdir(exist_ok=True)
        for i, item in enumerate(queries):
            body = dict(item, numResults=8, contents={"highlights": True})
            if item.get("category"):
                body.update(startPublishedDate="2024-01-01T00:00:00Z", endPublishedDate="2026-10-17T23:59:59Z")
            (directory / f"exa-{i:02d}.json").write_text(json.dumps(body))
        (root / "research/query-plan.json").write_text(json.dumps(queries, indent=2))
        print(len(queries))
        return
    os.environ["EXA_API_KEY"] = sys.stdin.read().strip()
    if not os.environ["EXA_API_KEY"]:
        raise RuntimeError("missing Exa credential on stdin")
    index = []
    for i, item in enumerate(queries):
        name = f"exa-{i:02d}"
        output = root / "research/raw" / (name + ".json")
        argv = [sys.executable, str(wrapper), item["query"], "--num-results", "8", "--highlights",
                "--start-published-date", "2024-01-01T00:00:00Z",
                "--end-published-date", "2026-10-17T23:59:59Z", "-o", str(output)]
        if item.get("category"):
            argv += ["--category", item["category"]]
        with (root / "logs" / (name + ".log")).open("w") as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False)
        index.append(dict(item, file=str(output), exit_code=process.returncode))
        (root / "research/search-index.json").write_text(json.dumps(index, indent=2))
        print(name, process.returncode, item["query"], flush=True)


if __name__ == "__main__":
    main()
