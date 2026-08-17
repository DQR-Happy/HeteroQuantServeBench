#!/usr/bin/env python3
"""Target-only known CPU page stimulus; never implies GPU mapping visibility."""

import ctypes
import json
import mmap
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hqsb.observation.procfs import page_sample, process_key, sample_process
from hqsb.observation.storage import atomic_json


def main():
    page_size, pages = os.sysconf("SC_PAGE_SIZE"), 1024
    key = process_key()
    region = mmap.mmap(
        -1,
        pages * page_size,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    region.madvise(mmap.MADV_NOHUGEPAGE)
    address = ctypes.addressof(ctypes.c_char.from_buffer(region))
    vma = {
        "start": hex(address),
        "end": hex(address + pages * page_size),
        "permissions": "rw-p",
        "path": "",
        "flags": [],
    }
    before_pages = page_sample(key, [vma], max_pages=pages)
    before = sample_process(key)
    for index in range(pages):
        region[index * page_size] = 1
    after = sample_process(key)
    after_pages = page_sample(key, [vma], max_pages=pages)
    region.close()
    minor_delta = after["faults"]["minor"] - before["faults"]["minor"]
    present_before = sum(row["present"] for row in before_pages["pages"])
    present_after = sum(row["present"] for row in after_pages["pages"])
    report = {
        "scope": "known_private_anonymous_cpu_pages_only",
        "process": key,
        "page_size": page_size,
        "touched_pages": pages,
        "minor_fault_delta": minor_delta,
        "present_before": present_before,
        "present_after": present_after,
        "pfn_exported": False,
        "before": before,
        "after": after,
        "status": "PASS"
        if present_before == 0 and present_after == pages and minor_delta >= pages
        else "FAIL",
        "limitations": [
            "Additional Python/procfs activity can cause extra minor faults.",
            "No GPU page, physical address, huge page or swap calibration claim.",
        ],
    }
    destination = Path("reports/observation/procfs-calibration.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "page_size",
                    "touched_pages",
                    "minor_fault_delta",
                    "present_before",
                    "present_after",
                )
            }
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
