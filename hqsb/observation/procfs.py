"""Read-only process/VMA/page observations; physical GPU mappings are opaque."""

import os
import struct
import time
from pathlib import Path


def fields(path):
    result = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        words = raw.split()
        if words and words[0].isdigit():
            result[key] = int(words[0]) * (1024 if words[-1] == "kB" else 1)
    return result


def process_key(pid=None, proc=Path("/proc")):
    pid = os.getpid() if pid is None else pid
    text = (proc / str(pid) / "stat").read_text()
    tail = text[text.rfind(")") + 1 :].split()
    if len(tail) < 20:
        raise ValueError("PROC_STAT_INVALID")
    return {
        "pid": pid,
        "start_ticks": int(tail[19]),
        "boot_id": (proc / "sys/kernel/random/boot_id").read_text().strip(),
    }


def _read(fn):
    try:
        return {"availability": "observed", "values": fn()}
    except (OSError, ValueError, IndexError) as exc:
        return {
            "availability": "permission_denied"
            if isinstance(exc, PermissionError)
            else "not_collected",
            "values": {},
            "reason": f"{type(exc).__name__}: {exc}"[:300],
        }


def sample_process(key, proc=Path("/proc"), *, with_rollup=True):
    start = time.monotonic_ns()
    if process_key(key["pid"], proc) != key:
        raise ValueError("PROCESS_REPLACED")
    root = proc / str(key["pid"])
    stat = (root / "stat").read_text()
    tail = stat[stat.rfind(")") + 1 :].split()

    def vmstat():
        return {
            k: int(v)
            for k, v in (
                line.split() for line in (proc / "vmstat").read_text().splitlines()
            )
        }

    result = {
        "process": key,
        "read_start_ns": str(start),
        "clock_domain": "host_monotonic",
        "scope": "process_and_system_separate",
        "status": _read(lambda: fields(root / "status")),
        "smaps_rollup": _read(lambda: fields(root / "smaps_rollup"))
        if with_rollup
        else {
            "availability": "not_collected",
            "values": {},
            "reason": "periodic_budget",
        },
        "faults": {"minor": int(tail[7]), "major": int(tail[9])},
        "system_memory": _read(lambda: fields(proc / "meminfo")),
        "system_vmstat": _read(vmstat),
        "psi": _read(lambda: {"raw": (proc / "pressure/memory").read_text()}),
    }
    if process_key(key["pid"], proc) != key:
        raise ValueError("PROCESS_REPLACED")
    result["read_end_ns"] = str(time.monotonic_ns())
    return result


def vma_snapshot(key, *, proc=Path("/proc"), max_vmas=4096):
    start = time.monotonic_ns()
    if process_key(key["pid"], proc) != key:
        raise ValueError("PROCESS_REPLACED")
    rows, row = [], None
    with (proc / str(key["pid"]) / "smaps").open() as stream:
        for line in stream:
            words = line.split(maxsplit=5)
            if words and "-" in words[0] and len(words) >= 5:
                try:
                    left, right = [int(x, 16) for x in words[0].split("-")]
                except ValueError:
                    continue
                if len(rows) >= max_vmas:
                    break
                row = {
                    "start": hex(left),
                    "end": hex(right),
                    "permissions": words[1],
                    "offset": words[2],
                    "device": words[3],
                    "inode": words[4],
                    "path": words[5].strip() if len(words) == 6 else "",
                    "fields": {},
                }
                rows.append(row)
            elif row is not None and ":" in line:
                name, raw = line.split(":", 1)
                items = raw.split()
                if items and items[0].isdigit():
                    row["fields"][name] = int(items[0]) * (
                        1024 if items[-1] == "kB" else 1
                    )
                elif name == "VmFlags":
                    row["flags"] = items
    if process_key(key["pid"], proc) != key:
        raise ValueError("PROCESS_REPLACED")
    return {
        "process": key,
        "read_start_ns": str(start),
        "read_end_ns": str(time.monotonic_ns()),
        "vmas": rows,
        "availability": "partial" if len(rows) >= max_vmas else "observed",
        "scope": "cpu_virtual_address_space",
        "max_vmas": max_vmas,
    }


def page_sample(key, vmas, *, max_pages=4096, proc=Path("/proc")):
    """Sample an anonymous CPU range, without clearing flags or exposing PFNs."""
    start = time.monotonic_ns()
    size = os.sysconf("SC_PAGE_SIZE")
    candidates = [
        v
        for v in vmas
        if v["permissions"].startswith("rw")
        and v["path"] in ("", "[heap]")
        and "io" not in v.get("flags", [])
        and "pf" not in v.get("flags", [])
    ][:2]
    if process_key(key["pid"], proc) != key:
        raise ValueError("PROCESS_REPLACED")
    rows = []
    with (proc / str(key["pid"]) / "pagemap").open("rb", buffering=0) as handle:
        for vma in candidates:
            for address in range(int(vma["start"], 16), int(vma["end"], 16), size):
                if len(rows) >= max_pages:
                    break
                raw = os.pread(handle.fileno(), 8, (address // size) * 8)
                if len(raw) != 8:
                    raise ValueError("PAGEMAP_SHORT_READ")
                value = struct.unpack("=Q", raw)[0]
                rows.append(
                    {
                        "address": hex(address),
                        "vma_start": vma["start"],
                        "present": bool(value & (1 << 63)),
                        "swapped": bool(value & (1 << 62)),
                        "file_or_shared": bool(value & (1 << 61)),
                        "soft_dirty": bool(value & (1 << 55)),
                        "pfn": None,
                    }
                )
    if process_key(key["pid"], proc) != key:
        raise ValueError("PROCESS_REPLACED")
    return {
        "process": key,
        "page_size": size,
        "pages": rows,
        "read_start_ns": str(start),
        "read_end_ns": str(time.monotonic_ns()),
        "clock_domain": "host_monotonic",
        "availability": "observed" if rows else "not_collected",
        "selection": "first_bounded_pages_of_first_two_anonymous_rw_vmas",
        "scope": "sampled_cpu_pages_not_gpu_mappings",
        "pfn_status": "not_collected",
        "limitations": [
            "Snapshot only; intervening page events are unknown.",
            "PFNs are deliberately not exported; no physical GPU mapping inference.",
        ],
    }
