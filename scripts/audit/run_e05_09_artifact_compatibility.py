#!/usr/bin/env python3
"""Collect E05-09 on the remote target, including real RTN weight slices.

The default collection is CPU-only and can coexist with another GPU experiment.
``--gpu`` adds an explicit CPU -> CUDA packed-operator replay. Neither slice
replay nor synthetic boundary fixtures count as whole-model/industrial evidence.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hqsb.quant import artifact, compat, faults, packing
from hqsb.quant.artifact import CalibrationProvenance, CompatibilityRecord, ModelIdentity, QuantArtifactDocument
from hqsb.quant.fixtures import build_tiny_artifact
from hqsb.quant.rtn import QuantStats, QuantizedTensor, dequantize_flat
from hqsb.quant.spec import QuantScheme, tail_length

DEFAULT = ROOT / "docs/stage_experiments/S05/E05-09/raw"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def finite_json(value):
    """Keep deliberately injected NaN diagnostics without invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_diagnostic": str(value)}
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(finite_json(x), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for x in rows))


def read(path):
    return json.loads(Path(path).read_text())


def capability(bits=4, group=128, arch="sm_87"):
    layout = packing.LAYOUT_W4A16_ROWMAJOR_NK_V1 if bits == 4 else packing.LAYOUT_W8A16_ROWMAJOR_NK_V1
    return compat.KernelCapability(kernel_id=f"hqsb.w{bits}a16.triton", provider="triton", layouts=(layout,), bits=(bits,), group_sizes=(group,), target_arch=arch, abi_version="1")


def source_slice(model_dir, tensor, rows, cols):
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
            if tensor not in header:
                continue
            record = header[tensor]
            if record["dtype"] not in ("F16", "BF16") or record["shape"][1] != cols:
                raise ValueError(f"unexpected safetensors tensor: {record}")
            f.seek(8 + size + record["data_offsets"][0])
            payload = f.read(rows * cols * 2)
        return payload, {"source_file": str(path), "tensor": tensor, "full_shape": record["shape"], "slice_rows": [0, rows], "source_dtype": "float16" if record["dtype"] == "F16" else "bfloat16", "source_slice_sha256": digest(payload)}
    raise FileNotFoundError(f"source tensor {tensor} not found in {model_dir}")


def convert_real_slice(bits, out):
    source_dir = ROOT / f"docs/stage_experiments/S05/E05-02/raw/artifacts/rtn_w{bits}"
    manifest = read(source_dir / "manifest.json")
    spec = read(ROOT / "docs/stage_experiments/S05/E05-02/raw/spec.json")
    index, entry = next((i, x) for i, x in enumerate(manifest["tensors"]) if x["name"].endswith("k_proj"))
    stem = f"{index:04d}_{digest(entry['name'].encode())[:12]}"
    folder = source_dir / "tensors" / stem
    for record in entry["files"].values():
        path = folder / record["path"]
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"source payload failed integrity: {path}")
    rows, cols = min(4, entry["source_shape"][0]), entry["source_shape"][1]
    group = manifest["quantization"]["group_size"]
    units = math.ceil(cols / group) if group else 1
    with (folder / entry["files"]["qvalues"]["path"]).open("rb") as f:
        payload = f.read(rows * (math.ceil(cols / 2) if bits == 4 else cols))
    if bits == 4:
        values = []
        stride = math.ceil(cols / 2)
        for row in range(rows):
            unpacked = [v for b in payload[row * stride:(row + 1) * stride] for v in (b & 15, b >> 4)][:cols]
            values.extend(v if v < 8 else v - 16 for v in unpacked)
    else:
        values = list(struct.unpack(f"<{len(payload)}b", payload))
    with (folder / entry["files"]["scales"]["path"]).open("rb") as f:
        scales = list(struct.unpack(f"<{rows * units}f", f.read(rows * units * 4)))
    original, provenance = source_slice(Path(spec["model"]["local_path"]).expanduser(), entry["name"] + ".weight", rows, cols)
    source_hash = digest(original)
    scheme = QuantScheme(bits=bits, granularity="per_group" if group else "per_channel", group_size=group)
    dequant = dequantize_flat(values, scales, [], (rows, cols), units, scheme, group_size=group, axis=1)
    qt = QuantizedTensor(scheme=scheme, shape=(rows, cols), axis=1, q=values, scales=scales, zeros=[], values_dequant=dequant, units_per_row=units, group_size=group, tail=tail_length(cols, group), stats=QuantStats(len(values), len(scales), 0, 0.0, 0, 0, 0), unit_stats=[])
    model = spec["model"]
    identity = ModelIdentity(model_id=model["id"], revision=model["revision"], architecture="Qwen3ForCausalLM", config_hash=model["config_sha256"], tokenizer_hash=model["tokenizer_sha256"], source_weight_sha256=source_hash, model_root_sha256=model["model_manifest_sha256"])
    doc = QuantArtifactDocument.from_quantized(qt, tensor_name=entry["name"] + ".weight.__rows_0_4", source_dtype=provenance["source_dtype"], source_sha256=source_hash, model=identity, method="rtn", method_config_hash=scheme.scheme_hash(), calibration=CalibrationProvenance(kind="NONE", notes="RTN source is E05-02; no calibration"), known_limitations=["Real Qwen RTN weight slice: first four rows only; not full tensor/model execution", "Source model root is the historical E05-02 manifest; the selected checkpoint slice is independently read and hashed in this run; E05-02 quantized its FP16 runtime cast"])
    cap = capability(bits, group)
    doc.compatibility = [CompatibilityRecord(kernel_id=cap.kernel_id, provider=cap.provider, layout_id=cap.layouts[0], target_arch=cap.target_arch, abi_version="1", supported_bits=(bits,), supported_groups=(group,) if group else (), dtype="float16", status="declared")]
    packed = packing.pack_kernel_variant(values, scales, [], scheme, rows, cols, layout_id=cap.layouts[0], parent_canonical_hash=doc.canonical_hash())
    doc.add_variant(packed, target_arch="sm_87")
    doc.save(str(out))
    provenance.update({"scope": "real_model_weight_slice", "bits": bits, "parent_artifact_id": manifest["artifact_id"], "parent_manifest_sha256": file_hash(source_dir / "manifest.json"), "source_payloads": entry["files"], "shape": [rows, cols], "source_checkpoint_slice_bytes": len(original), "quantization_input_dtype": "float16 (E05-02 runtime cast)", "conversion": "decode existing qvalues/scales without requantization", "canonical_hash": doc.canonical_hash(), "artifact_id": doc.artifact_id(), "logical_values_unchanged": artifact.load_document(str(out)).values == values})
    return provenance


def replay(path):
    doc = artifact.load_document(str(path))
    cap = capability(doc.scheme.bits, doc.scheme.group_size)
    decision = compat.check_compatibility(doc, cap, expected_model=doc.model, tensor_name=doc.tensor.name, tensor_shape=doc.tensor.shape)
    weights = dequantize_flat(doc.values, doc.scales, doc.zeros, doc.tensor.shape, doc.units_per_row, doc.scheme, group_size=doc.scheme.group_size, axis=doc.scheme.axis)
    rows, cols = doc.tensor.shape
    x = [((i * 7) % 31 - 15) / 16.0 for i in range(cols)]
    y = [math.fsum(x[k] * weights[n * cols + k] for k in range(cols)) for n in range(rows)]
    return {"pid": os.getpid(), "artifact_id": doc.artifact_id(), "identity_hash": doc.identity_hash(), "canonical_hash": doc.canonical_hash(), "dispatch": decision.as_dict(), "operator": "CPU float64 canonical dequant GEMV; no low-bit kernel claim", "input_hash": digest(struct.pack(f"<{len(x)}d", *x)), "output": y, "output_hash": digest(struct.pack(f"<{len(y)}d", *y)), "low_bit_executed": False}


def child_atomic(source, target, stage, abrupt):
    doc = artifact.load_document(source)
    doc.variant_payloads = artifact.load_variant_payloads(doc)
    original_write, original_rename = artifact._write_file, artifact.os.rename
    calls = 0
    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == stage:
            if abrupt:
                os._exit(73)
            raise OSError("E05-09 injected interruption")
    def wrapped_write(path, payload):
        original_write(path, payload)
        checkpoint()
    def wrapped_rename(src, dst):
        checkpoint()
        return original_rename(src, dst)
    artifact._write_file, artifact.os.rename = wrapped_write, wrapped_rename
    try:
        doc.save(target)
        print(json.dumps({"saved": True, "calls": calls}))
    except Exception as exc:
        print(json.dumps({"saved": False, "calls": calls, "error": str(exc), "type": type(exc).__name__}))
    finally:
        artifact._write_file, artifact.os.rename = original_write, original_rename


def subcommand(*args):
    return [sys.executable, str(Path(__file__).resolve()), *map(str, args)]


def collect_faults(golden, scratch):
    doc = artifact.load_document(str(golden))
    cap = capability(doc.scheme.bits, doc.scheme.group_size)
    cases = [x for x in faults.build_fault_matrix() if not x.fixture_variant]
    results = faults.run_fault_matrix(str(golden), str(scratch), cap, cases=cases)
    rows = [x.as_row(doc.artifact_id()) for x in results]
    # Preserve the loader's native diagnostic details for field/tensor audit.
    for row, case in zip(rows, cases):
        tmp = scratch / (case.case_id + "_diagnostic")
        shutil.copytree(golden, tmp)
        try:
            case.apply(str(tmp))
            validation = artifact.validate_artifact_dir(str(tmp))
            if not validation["accepted"]:
                row["diagnostic"] = validation
            else:
                effective = copy.deepcopy(cap)
                for key, value in (case.capability_variant or {}).items():
                    setattr(effective, key, value)
                row["diagnostic"] = compat.check_compatibility(artifact.load_document(str(tmp)), effective, expected_model=doc.model, tensor_name=doc.tensor.name, tensor_shape=doc.tensor.shape, workspace_available_bytes=1 << 30).as_dict()
        finally:
            shutil.rmtree(tmp)
    return rows


def collect_atomic(source, scratch):
    rows = []
    for abrupt in (False, True):
        for stage in range(1, 6):
            folder = scratch / f"{'kill' if abrupt else 'exception'}_{stage}"
            folder.mkdir(parents=True)
            target = folder / "artifact"
            p = subprocess.run(subcommand("--child", "atomic", "--source", source, "--target", target, "--stage", stage, *( ["--abrupt"] if abrupt else [])), capture_output=True, text=True)
            orphans = list(folder.glob(".hqsb-quant-*"))
            visible = target.exists()
            valid = artifact.validate_artifact_dir(str(target))["accepted"] if visible else False
            injected = p.returncode == 73 if abrupt else "injected interruption" in p.stdout
            for tmp in orphans:
                shutil.rmtree(tmp)
            rows.append({"case_id": folder.name, "stage": stage, "abrupt_exit": abrupt, "exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "interruption_reached": injected, "target_visible": visible, "target_valid": valid, "orphan_temporary_directories": len(orphans), "recovery_cleanup_ok": not list(folder.glob(".hqsb-quant-*")), "passed": injected and not visible and not list(folder.glob(".hqsb-quant-*"))})
    # Existing artifacts are immutable: a rejected overwrite must leave old data.
    folder = scratch / "existing"
    folder.mkdir()
    target = folder / "artifact"
    shutil.copytree(source, target)
    before = file_hash(target / "manifest.json")
    p = subprocess.run(subcommand("--child", "atomic", "--source", source, "--target", target, "--stage", 0), capture_output=True, text=True)
    rows.append({"case_id": "existing_artifact_immutable", "stdout": p.stdout, "passed": before == file_hash(target / "manifest.json") and "refusing to overwrite" in p.stdout and artifact.validate_artifact_dir(str(target))["accepted"] and not list(folder.glob(".hqsb-quant-*"))})
    folder = scratch / "concurrent"
    folder.mkdir()
    target = folder / "artifact"
    processes = [subprocess.Popen(subcommand("--child", "atomic", "--source", source, "--target", target, "--stage", 0), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outputs = [p.communicate() for p in processes]
    records = [json.loads(x[0]) for x in outputs]
    rows.append({"case_id": "concurrent_immutable_writers", "records": records, "stderr": [x[1] for x in outputs], "passed": sum(x["saved"] for x in records) == 1 and artifact.validate_artifact_dir(str(target))["accepted"] and not list(folder.glob(".hqsb-quant-*"))})
    return rows


def collect_gpu(source):
    import torch
    from ops.quant.w4a16_triton import prepare_weights, gemm_low_bit, KERNEL_SYMBOL
    doc = artifact.load_document(str(source))
    prepared = prepare_weights(doc, device="cuda")
    x = torch.arange(doc.tensor.shape[1], device="cuda", dtype=torch.float32).remainder(31).sub(15).div(16).half().reshape(1, -1)
    reference_weights = dequantize_flat(doc.values, doc.scales, doc.zeros, doc.tensor.shape, doc.units_per_row, doc.scheme, group_size=doc.scheme.group_size, axis=1)
    reference = x @ torch.tensor(reference_weights, device="cuda", dtype=torch.float16).reshape(doc.tensor.shape).T
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        actual = gemm_low_bit(x, prepared)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(Path(source).parents[1] / "cross_device" / "packed_slice_trace.json"))
    names = sorted({event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA})
    error = float((actual - reference).abs().max())
    return {"scope": "real_model_weight_slice_operator", "effective_uid": os.geteuid(), "profiler_privilege": "sudo -n for CUPTI only; no system settings changed" if os.geteuid() == 0 else "normal user", "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "max_abs": error, "reference": "canonical dequant -> FP16 matmul", "reference_values": reference.cpu().tolist(), "actual_values": actual.cpu().tolist(), "observed_kernel_names": names, "low_bit_executed": any(KERNEL_SYMBOL in x for x in names), "passed": torch.allclose(actual, reference, atol=0.02, rtol=0.02) and any(KERNEL_SYMBOL in x for x in names), "model_executed": False}


def collect(out, use_gpu):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "verdict.json").exists():
        raise FileExistsError("use a fresh output directory; evidence is immutable")
    (out / "collector_snapshot.py").write_bytes(Path(__file__).read_bytes())
    spec = {"schema": "hqsb.e05_09.spec/v1", "experiment_id": "E05-09", "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "scope": "real E05-02 RTN weight slices plus explicitly labelled synthetic contracts", "independent_processes": 3, "source_slice_rows": 4, "atomic_interrupt_checkpoints": [1, 2, 3, 4, 5], "gpu_requested": use_gpu, "faults": [x.as_dict() for x in faults.build_fault_matrix()], "quality_claim": False, "formal_pass_requires": "RTN and industrial full-model reload; actual device transfer; every detail criterion"}
    write(out / "spec.json", spec)
    write(out / "environment.json", {"host": platform.node(), "platform": platform.platform(), "python": sys.version, "executable": sys.executable, "pid": os.getpid(), "collector_sha256": file_hash(__file__), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "execution": "CPU-only unless gpu_requested=true"})
    sources = {}
    for bits in (4, 8):
        path = out / "golden" / f"real_rtn_w{bits}_slice"
        sources[bits] = path
        provenance = convert_real_slice(bits, path)
        write(out / "golden" / f"real_rtn_w{bits}_provenance.json", provenance)
    w4 = artifact.load_document(str(sources[4]))
    cap = capability()
    with tempfile.TemporaryDirectory(prefix="hqsb-e05-09-") as tmp:
        scratch = Path(tmp)
        rows = collect_faults(sources[4], scratch / "faults")
        jsonl(out / "validation" / "results.jsonl", rows)
        write(out / "validation" / "summary.json", {"total": len(rows), "caught": sum(x["caught"] for x in rows), "cleanup": all(x["cleanup"] for x in rows), "source": "real RTN W4 model-weight slice", "all_passed": all(x["caught"] and x["prelaunch"] and x["cleanup"] for x in rows)})
        replay_rows = []
        for bits, path in sources.items():
            for repeat in range(3):
                p = subprocess.run(subcommand("--child", "replay", "--source", path), capture_output=True, text=True, check=True)
                replay_rows.append({"bits": bits, "repeat": repeat, **json.loads(p.stdout)})
        jsonl(out / "cross_process" / "results.jsonl", replay_rows)
        replay_ok = all(len({x[field] for x in replay_rows if x["bits"] == bits}) == 1 for bits in (4, 8) for field in ("canonical_hash", "identity_hash", "output_hash", "input_hash")) and len({x["pid"] for x in replay_rows}) == 6
        write(out / "cross_process" / "summary.json", {"passed": replay_ok, "processes": len({x["pid"] for x in replay_rows}), "scope": "CPU canonical operator; no full model or observed CUDA kernel"})
        version_rows = []
        for version in ("0.9.0", "1.0.0", "1.0.1", "1.1.0", "2.0.0"):
            path = scratch / ("schema_" + version)
            shutil.copytree(sources[4], path)
            manifest = read(path / "manifest.json")
            manifest["schema_version"] = version
            write(path / "manifest.json", manifest)
            validation = artifact.validate_artifact_dir(str(path))
            version_rows.append({"declared": version, "reader": artifact.ARTIFACT_SCHEMA_VERSION, "validation": validation, "plan": compat.plan_migration(version, artifact.ARTIFACT_SCHEMA_VERSION), "passed": validation["accepted"] == (version == artifact.ARTIFACT_SCHEMA_VERSION)})
        jsonl(out / "migration" / "version_matrix.jsonl", version_rows)
        repacked, provenance = compat.repack(w4, packing.LAYOUT_W4A16_HIFIRST_NK_V1, target_arch="sm_87")
        # Loaded documents keep payloads on disk. Retained variants must be
        # materialized before writing the new immutable multi-variant artifact.
        repacked.variant_payloads.update(artifact.load_variant_payloads(w4))
        target = out / "repack" / "real_w4_hifirst"
        repacked.save(str(target))
        loaded = artifact.load_document(str(target))
        invariant = {"qvalues": loaded.values == w4.values, "scales": loaded.scales == w4.scales, "zeros": loaded.zeros == w4.zeros, "canonical_hash": loaded.canonical_hash() == w4.canonical_hash(), "model_identity": loaded.model.as_dict() == w4.model.as_dict(), "multiple_variants": len(loaded.variants) == 2}
        write(out / "repack" / "provenance.json", {**provenance.as_dict(), "verified_after_disk_reload": invariant, "passed": all(invariant.values()), "payload_materialization": "load_variant_payloads preserves previously loaded packed variants", "kernel_oracle": "NOT_EXECUTED for hifirst: current HQSB kernel supports rowmajor only"})
        hifirst = build_tiny_artifact(layout_variant="hifirst", target_arch="sm_87")
        hifirst_path = out / "fixtures" / "synthetic_hifirst"
        hifirst.save(str(hifirst_path))
        decision = compat.check_compatibility(artifact.load_document(str(hifirst_path)), cap)
        write(out / "repack" / "unsupported_layout_dispatch.json", {"scope": "synthetic boundary contract", **decision.as_dict(), "passed": decision.status == compat.REPACK_REQUIRED})
        fallback_rows = []
        for mode in compat.FALLBACK_MODES:
            unavailable = copy.deepcopy(cap)
            unavailable.available = False
            unavailable.unavailable_reason = "E05-09 controlled provider-unavailable injection"
            result = compat.check_compatibility(w4, unavailable, mode=mode)
            expect_fallback = mode in (compat.MODE_EXPLICIT_FALLBACK, compat.MODE_DEBUG)
            fallback_rows.append({"mode": mode, **result.as_dict(), "low_bit_claim_allowed": result.allows_low_bit_claim, "passed": result.status == (compat.EXPLICIT_FALLBACK if expect_fallback else compat.REJECT) and not result.allows_low_bit_claim})
        jsonl(out / "fallback" / "results.jsonl", fallback_rows)
        quant_rows = []
        for changed in ({"bits": (8,)}, {"group_sizes": (32,)}):
            incompatible = copy.deepcopy(cap)
            for key, value in changed.items():
                setattr(incompatible, key, value)
            result = compat.check_compatibility(w4, incompatible)
            quant_rows.append({"change": changed, **result.as_dict(), "passed": result.status == compat.REQUANTIZE_REQUIRED})
        jsonl(out / "repack" / "requantize_refusal.jsonl", quant_rows)
        atomic_rows = collect_atomic(sources[4], scratch / "atomic")
        jsonl(out / "atomicity" / "results.jsonl", atomic_rows)
    gpu = {"status": "NOT_EXECUTED", "reason": "CPU-only collection; GPU reserved for E05-06", "passed": False, "full_model_reload": False}
    if use_gpu:
        try:
            gpu = collect_gpu(sources[4])
        except Exception as exc:
            gpu = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "passed": False}
    write(out / "cross_device" / "cpu_to_cuda.json", gpu)
    criteria = [
        (1, "canonical / packed 分层可审计", True),
        (2, "model / quant / pack / kernel 四层身份完整", True),
        (3, "golden artifact 在新进程重放", replay_ok),
        (4, "支持的跨 device 路径通过，不支持路径正确拒绝", bool(gpu.get("passed"))),
        (5, "version / migration 有机器规则", all(x["passed"] for x in version_rows)),
        (6, "故障注入在预期阶段和 launch 前识别", all(x["caught"] for x in rows)),
        (7, "字段 / tensor 级定位", all(bool(x["diagnostic"].get("field_path")) for x in rows)),
        (8, "lossless repack 不变量和新 provenance", all(invariant.values())),
        (9, "repack 与 requantize 严格区分", all(x["passed"] for x in quant_rows)),
        (10, "fallback 显式可关闭且不污染低 bit claim", all(x["passed"] for x in fallback_rows)),
        (11, "原子保存与中断恢复不暴露半成品", all(x["passed"] for x in atomic_rows)),
        (12, "资源失败后无脏 cache / 泄漏", False),
        (13, "兼容矩阵由 raw 自动产生", True),
        (14, "RTN 和工业 artifact 完成端到端模型重载", False),
    ]
    jsonl(out / "compatibility_matrix.jsonl", [{"kind": "fault", **x} for x in rows] + [{"kind": "fallback", **x} for x in fallback_rows] + [{"kind": "requantize", **x} for x in quant_rows])
    verdict = {"schema": "hqsb.e05_09.verdict/v1", "experiment_id": "E05-09", "overall": "BLOCKED", "scientific_execution_verdict": "PARTIAL", "formal_pass_allowed": False, "scope": "real RTN weight slices and actual loader contracts; no industrial or full-model replay", "detail_pass_criteria": [{"id": i, "name": name, "passed": passed} for i, name, passed in criteria], "passed_criteria": sum(x[2] for x in criteria), "total_criteria": len(criteria), "expected_effect": {"name": "防止静默使用错误量化制品", "passed": False, "partial_scope_passed": all(x["caught"] for x in rows) and all(x["passed"] for x in fallback_rows)}, "single_item_standard": {"name": "所有不兼容执行前识别；repack 可追溯；禁止静默 FP16 回退后称低 bit", "passed": False, "reason": "字段定位、设备资源恢复与工业 / 全模型重载证据尚不齐全"}, "blocking_reasons": ["E05-04 has no industrial artifact", "No whole-model RTN / industrial reload in E05-09", "Resource workspace refusal is tested but real device OOM/cache recovery is not"], "wall_time_s": time.perf_counter() - started, "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "format_deviation": "JSONL tables instead of Parquet; bytes are not relabelled"}
    write(out / "verdict.json", verdict)
    report = f"""# E05-09 实验报告：量化制品兼容性与恢复

最终判定：**BLOCKED / PARTIAL**；细则 {verdict['passed_criteria']}/14。机器判定见 [verdict.json](raw/verdict.json)。

本轮在 Jetson 上实际执行了读取、转换、故障注入、独立进程和持久化实验。RTN W4/W8 输入来自 E05-02 已落盘的 Qwen3-1.7B k_proj 前 4 行；原始 qvalue/scale 全文件先校验 hash，源 checkpoint slice 从 safetensors 直接读取并 hash（checkpoint 为 BF16，E05-02 在 FP16 runtime cast 后量化）。转换不重新量化。它是实际模型权重 slice，不能当作整 tensor 或模型推理。额外的 hifirst fixture 明确为 synthetic contract。

| 必采集项 | 实测结果 / 证据 |
|---|---|
| manifest / model / quant / layout 身份 | [golden](raw/golden/)；两种 RTN、源 manifest 和 checkpoint slice hash |
| 不匹配矩阵与 prelaunch / reason | [results.jsonl](raw/validation/results.jsonl)：{sum(x['caught'] for x in rows)}/{len(rows)} 按预期识别 |
| 跨进程 output / artifact / dispatch | [cross_process](raw/cross_process/results.jsonl)：6 个全新进程，稳定={replay_ok} |
| lossless repack | [provenance.json](raw/repack/provenance.json)：保存、重载后全部不变量={all(invariant.values())} |
| 版本与迁移 | [version_matrix.jsonl](raw/migration/version_matrix.jsonl)：5 版本；只接受现有 1.0.0，其他显式拒绝 |
| bit/group 改动 | [requantize_refusal.jsonl](raw/repack/requantize_refusal.jsonl)：归类 REQUANTIZE_REQUIRED |
| fallback | [results.jsonl](raw/fallback/results.jsonl)：4 模式，strict/repack-only 拒绝；允许模式标 FP16 fallback |
| 原子保存与恢复 | [results.jsonl](raw/atomicity/results.jsonl)：{sum(x['passed'] for x in atomic_rows)}/{len(atomic_rows)}；异常、强制进程退出、不可覆盖、并发写 |
| CPU→CUDA / observed kernel | [cpu_to_cuda.json](raw/cross_device/cpu_to_cuda.json) |

强制退出会留下不对 reader 可见的临时 sibling 目录；本轮明确记录并清理这些目录。异常回滚与进程崩溃后的清理是不同机制，没有把残留临时目录宣称为自动清理。save 接口采用不可覆盖的 immutable artifact 语义，不支持把旧 artifact 原地覆盖成新版。

预计效果“防止静默使用错误量化制品”在本轮已执行的 slice / loader 范围得到部分证据；总体**未达到**。单项通过标准总体**未通过**：并非每个故障都返回 field_path；实际设备 OOM/cache 恢复未测，E05-04 未提供工业 artifact，也没有完成 RTN 与工业制品的端到端模型重载。版本测试只有当前 reader 与不同 artifact version；未伪造不存在的历史 reader 或迁移实现。CPU operator 一致性不能支持 CUDA、低 bit 性能或模型质量 claim。

全部表由本 runner 输出，未覆盖 E05-01..04 原始记录。JSONL 为真实格式，未伪装 Parquet。
"""
    (out.parent / "E05-09_实验报告.md").write_text(report)
    evidence_manifest(out)
    print(json.dumps({"overall": verdict["overall"], "passed_criteria": verdict["passed_criteria"], "faults_caught": sum(x["caught"] for x in rows), "faults_total": len(rows), "output": str(out)}, ensure_ascii=False))


def evidence_manifest(out):
    manifest = {"schema": "hqsb.evidence_manifest/v1", "experiment_id": "E05-09", "files": [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "EVIDENCE_MANIFEST.json"], "report": {"path": "../E05-09_实验报告.md", "sha256": file_hash(out.parent / "E05-09_实验报告.md")}, "collector": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": file_hash(__file__)}}
    write(out / "EVIDENCE_MANIFEST.json", manifest)


def supplement_gpu(out):
    """Append one predeclared device check without rerunning CPU evidence."""
    verdict = read(out / "verdict.json")
    path = out / "cross_device" / "cpu_to_cuda.json"
    previous = read(path)
    if previous.get("status") != "NOT_EXECUTED":
        raise FileExistsError("GPU evidence exists; use an explicit new run instead of overwriting it")
    write(out / "cross_device" / "before_device_supplement.json", previous)
    spec = {"schema": "hqsb.e05_09.device_supplement/v1", "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source": "golden/real_rtn_w4_slice", "scope": "CPU validated canonical -> CUDA packed operator; not model replay", "max_abs_tolerance": 0.02, "relative_tolerance": 0.02, "required_observed_symbol": "hqsb_dequant_gemm_kernel", "collector_sha256": file_hash(__file__)}
    write(out / "cross_device" / "supplement_spec.json", spec)
    (out / "cross_device" / "collector_snapshot.py").write_bytes(Path(__file__).read_bytes())
    try:
        result = collect_gpu(out / "golden" / "real_rtn_w4_slice")
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "passed": False}
    write(path, result)
    old_count = verdict["passed_criteria"]
    for row in verdict["detail_pass_criteria"]:
        if row["id"] == 4:
            row["passed"] = bool(result.get("passed"))
    verdict["passed_criteria"] = sum(x["passed"] for x in verdict["detail_pass_criteria"])
    verdict["device_supplement"] = {"passed": bool(result.get("passed")), "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "evidence": "cross_device/cpu_to_cuda.json"}
    write(out / "verdict.json", verdict)
    report_path = out.parent / "E05-09_实验报告.md"
    report = report_path.read_text().replace(f"细则 {old_count}/14", f"细则 {verdict['passed_criteria']}/14")
    report += f"\n设备补测：CPU→CUDA packed slice operator passed={bool(result.get('passed'))}，max_abs={result.get('max_abs')}，observed kernel 命中={result.get('low_bit_executed', False)}。补测预注册与原始 kernel 名称见 [cross_device](raw/cross_device/cpu_to_cuda.json)。这不补足工业和端到端模型重载门。\n"
    report_path.write_text(report)
    evidence_manifest(out)
    print(json.dumps({"overall": verdict["overall"], "passed_criteria": verdict["passed_criteria"], "gpu": result}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--gpu-only", action="store_true", help="append a device check to existing CPU evidence")
    parser.add_argument("--child", choices=("replay", "atomic"))
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--abrupt", action="store_true")
    args = parser.parse_args()
    if args.child == "replay":
        print(json.dumps(replay(args.source)))
    elif args.child == "atomic":
        child_atomic(args.source, args.target, args.stage, args.abrupt)
    elif args.gpu_only:
        supplement_gpu(args.output_dir)
    else:
        collect(args.output_dir, args.gpu)


if __name__ == "__main__":
    main()
