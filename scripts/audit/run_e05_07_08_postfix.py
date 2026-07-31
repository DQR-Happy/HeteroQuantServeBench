#!/usr/bin/env python3
"""Append CPU numerical regression evidence without modifying E05-07/08 raw."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hqsb.quant.activation import ActivationQuantSpec, quantize_activation, w8a8_reference
from hqsb.quant.kv import KvQuantSpec, capacity_ratio, kv_capacity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_raw(root):
    manifest = read(root / "EVIDENCE_MANIFEST.json")
    failures = []
    for item in manifest["files"]:
        relative = item.get("path", item.get("relative_path"))
        path = root / relative
        if not path.is_file() or sha(path) != item["sha256"]:
            failures.append(relative)
    if failures:
        raise ValueError(f"pre-fix evidence hash mismatches: {failures}")
    return {"files_verified": len(manifest["files"]), "manifest_sha256": sha(root / "EVIDENCE_MANIFEST.json"), "unchanged": True}


def main():
    stage = ROOT / "docs/stage_experiments/S05"
    raw7, raw8 = stage / "E05-07/raw", stage / "E05-08/raw"
    out7, out8 = stage / "E05-07/post_fix", stage / "E05-08/post_fix"
    for output in (out7, out8):
        output.mkdir(exist_ok=True)
        if (output / "correctness.json").exists():
            raise FileExistsError(f"refusing to replace completed post-fix evidence: {output}")
    untouched = {"E05-07": verify_raw(raw7), "E05-08": verify_raw(raw8)}
    metadata = {"run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "host": platform.node(), "python": sys.version, "scope": "CPU helper numerical fixes only; no new model/GPU performance claim", "original_raw": untouched, "sources": [{"path": name, "sha256": sha(ROOT / name)} for name in ("hqsb/quant/activation.py", "hqsb/quant/kv.py", "tests/unit/quant/test_activation_kv_regressions.py", "scripts/audit/run_e05_07_08_postfix.py")]}
    before = read(raw7 / "correctness/reference_broadcast_audit.json")
    actual = w8a8_reference([1, 2, 3, 4], [2, 1, 1, 2], x_scales=[0.5, 2.0], w_scales=[0.25, 3.0], m=2, n=2, k=2)
    expected = [0.5, 7.5, 5.0, 66.0]
    result = quantize_activation([-127, 127, -1, 0, 1, -64, 64] * 2, ActivationQuantSpec(granularity="per_tensor"), shape=(2, 7))
    activation = {"metadata": metadata, "broadcast": {"before": before, "after": actual, "independent_expected": expected, "max_abs": max(abs(x - y) for x, y in zip(actual, expected)), "passed": actual == expected}, "scale_utilization": {"codes": result.q, "before_fraction": 4 / 14, "after_fraction": result.scale_utilization, "expected_half_range_fraction": 8 / 14, "boundary_occupancy": result.saturation_rate, "passed": result.scale_utilization == 8 / 14 and result.saturation_rate == 4 / 14}, "group_scope": "more than one K group is now explicitly rejected; no silent first-group scaling"}
    write(out7 / "correctness.json", activation)
    dims = dict(layers=28, kv_heads=8, head_dim=128, tokens=128)
    spec = KvQuantSpec(k_bits=4, v_bits=4, residual_window=32)
    default, fp16, fp32 = (kv_capacity(spec, **dims), kv_capacity(spec, **dims, dtype_bytes=2), kv_capacity(spec, **dims, dtype_bytes=4))
    plain = kv_capacity(KvQuantSpec(k_bits=16, v_bits=16), **dims)
    mixed = kv_capacity(KvQuantSpec(k_bits=16, v_bits=8), **dims)
    all_residual = kv_capacity(KvQuantSpec(k_granularity="per_tensor", v_granularity="per_channel", residual_window=128), **dims)
    capacity = {"metadata": metadata, "before_residual_dtype": read(raw8 / "capacity/residual_dtype_audit.json"), "after_residual_dtype": {"default_total_bytes": default.total_bytes, "explicit_fp16_total_bytes": fp16.total_bytes, "explicit_fp32_total_bytes": fp32.total_bytes, "passed": default.total_bytes == fp16.total_bytes}, "fp16_kv": plain.as_dict(), "fp16_kv_metadata_passed": plain.scale_bytes == plain.zero_bytes == 0, "mixed_fp16_int8": mixed.as_dict(), "mixed_metadata_passed": mixed.scale_bytes == 28 * 8 * 128 * 4, "all_residual": all_residual.as_dict(), "all_residual_metadata_passed": all_residual.scale_bytes == all_residual.zero_bytes == all_residual.payload_bytes == 0, "ratio_explicit_fp32": capacity_ratio(spec, **dims, dtype_bytes=4), "scientific_raw_impact": "E05-08 runner explicitly supplied dtype_bytes=2 and used INT4/INT8 per-token/head; its original measured capacity rows are unchanged"}
    write(out8 / "correctness.json", capacity)
    tests_before, tests_after = (out7 / "pytest_before.txt"), (out7 / "pytest_after.txt")
    tests = {"before": {"path": "../E05-07/post_fix/pytest_before.txt", "sha256": sha(tests_before), "result": "20 newly added numerical regressions failed on original code"}, "after": {"path": "../E05-07/post_fix/pytest_after.txt", "sha256": sha(tests_after), "result": "47 passed (27 existing + 20 new), CPU Jetson"}}
    if "20 failed" not in tests_before.read_text() or "47 passed" not in tests_after.read_text():
        raise AssertionError("stored pytest outcome differs from this frozen regression record")
    for output in (out7, out8):
        write(output / "regression_tests.json", tests)
        (output / "collector_snapshot.py").write_bytes(Path(__file__).read_bytes())
        write(output / "EVIDENCE_MANIFEST.json", {"schema": "hqsb.e05.post_fix_evidence/v1", "original_raw_unchanged": untouched, "files": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(output.iterdir()) if path.is_file() and path.name != "EVIDENCE_MANIFEST.json"]})
    assert activation["broadcast"]["passed"] and activation["scale_utilization"]["passed"]
    assert all(capacity[key] for key in ("fp16_kv_metadata_passed", "mixed_metadata_passed", "all_residual_metadata_passed"))
    print(json.dumps({"passed": True, "cpu_regressions": 47, "original_raw_unchanged": untouched}))


if __name__ == "__main__":
    main()
