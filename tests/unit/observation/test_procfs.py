import struct
from pathlib import Path

import pytest

from hqsb.observation.procfs import (
    process_key,
    sample_process,
    vma_snapshot,
    page_sample,
)


def proc_tree(tmp_path):
    proc = tmp_path / "proc"
    (proc / "11").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot")
    tail = ["S"] + ["0"] * 21
    tail[7], tail[9], tail[19] = "5", "2", "101"
    (proc / "11/stat").write_text("11 (model (worker)) " + " ".join(tail))
    (proc / "11/status").write_text("VmRSS: 2 kB\nVmSwap: 0 kB\n")
    (proc / "meminfo").write_text("MemAvailable: 100 kB\nCached: 3 kB\n")
    (proc / "vmstat").write_text("pgfault 9\npswpin 0\n")
    return proc


def test_stat_parentheses_units_and_missing_rollup(tmp_path):
    proc = proc_tree(tmp_path)
    key = process_key(11, proc)
    assert key["start_ticks"] == 101
    sample = sample_process(key, proc)
    assert sample["faults"] == {"minor": 5, "major": 2}
    assert sample["status"]["values"]["VmRSS"] == 2048
    assert sample["status"]["values"]["VmSwap"] == 0
    assert sample["smaps_rollup"]["availability"] == "not_collected"
    with pytest.raises(ValueError, match="PROCESS_REPLACED"):
        sample_process({**key, "start_ticks": 100}, proc)


def test_page_flags_not_physical_frame_inference(tmp_path):
    proc = proc_tree(tmp_path)
    (proc / "11/smaps").write_text(
        "00001000-00003000 rw-p 00000000 00:00 0\nRss: 4 kB\nVmFlags: rd wr\n"
    )
    key = process_key(11, proc)
    vmas = vma_snapshot(key, proc=proc)
    assert vmas["vmas"][0]["fields"]["Rss"] == 4096
    with (proc / "11/pagemap").open("wb") as stream:
        stream.write(struct.pack("=Q", 0))
        stream.write(struct.pack("=Q", (1 << 63) | (1 << 55)))
        stream.write(struct.pack("=Q", 1 << 62))
    result = page_sample(key, vmas["vmas"], max_pages=2, proc=proc)
    assert result["pages"][0]["present"] and result["pages"][0]["soft_dirty"]
    assert result["pages"][1]["swapped"]
    assert all(row["pfn"] is None for row in result["pages"])


def test_real_self_scope_is_read_only():
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux procfs required")
    key = process_key()
    row = sample_process(key)
    assert row["process"] == key
    assert int(row["read_end_ns"]) >= int(row["read_start_ns"])
