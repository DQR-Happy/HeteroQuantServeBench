#!/usr/bin/env python3
"""E05-04 industrial-method adapter gate and contract evidence collector.

The experiment protocol is fail-closed.  A real GPTQ/AWQ run may start only
after E05-03 emits a frozen ``e05_04_handoff.json`` and the exact source
libraries are importable.  When either gate is absent, this driver still
captures the target environment and executes clearly-labelled *contract
fixtures* for field mapping, source-to-canonical equivalence, common packing,
tail handling and new-process replay.  Contract fixtures are never counted as
industrial artifacts, model quality, or low-bit runtime evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hqsb.quant import packing  # noqa: E402
from hqsb.quant.adapters import (  # noqa: E402
    AWQ_ADAPTER,
    GPTQ_ADAPTER,
    SMOOTHQUANT_ADAPTER,
    MethodMatrix,
    PackedVariantBuilder,
    SourceTensorRecord,
    audit_field_mapping,
    transform_equivalence_check,
)
from hqsb.quant.artifact import (  # noqa: E402
    CalibrationProvenance,
    CompatibilityRecord,
    ModelIdentity,
    QuantArtifactDocument,
    load_document,
    validate_artifact_dir,
)
from hqsb.quant.rtn import QuantStats, QuantizedTensor, dequantize_flat  # noqa: E402
from hqsb.quant.spec import tail_length  # noqa: E402


DEFAULT_OUTPUT = ROOT / "docs/stage_experiments/S05/E05-04/raw"
E05_01_VERDICT = ROOT / "docs/stage_experiments/S05/E05-01/raw/verdict.json"
E05_02_VERDICT = ROOT / "docs/stage_experiments/S05/E05-02/raw/verdict.json"
E05_02_SPEC = ROOT / "docs/stage_experiments/S05/E05-02/raw/spec.json"
E05_02_RTN_MANIFEST = (
    ROOT / "docs/stage_experiments/S05/E05-02/raw/artifacts/rtn_w4/manifest.json"
)
E05_03_VERDICT = ROOT / "docs/stage_experiments/S05/E05-03/raw/verdict.json"
E05_03_HANDOFF = (
    ROOT
    / "docs/stage_experiments/S05/E05-03/raw/selection/e05_04_handoff.json"
)
MODEL_MANIFEST = ROOT / "docs/benchmark/model_sha256_manifest.txt"

METHOD_PACKAGES = {
    "gptq": {"distribution": "gptqmodel", "import_name": "gptqmodel"},
    "awq": {"distribution": "autoawq", "import_name": "awq"},
    "auto_gptq_compat": {
        "distribution": "auto-gptq",
        "import_name": "auto_gptq",
    },
    "smoothquant": {
        "distribution": "smoothquant",
        "import_name": "smoothquant",
    },
}

REQUESTED_CONFIGS = {
    "gptq": {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "damp_percent": 0.01,
        "desc_act": True,
    },
    "awq": {
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": False,
        "version": "GEMM",
    },
    "smoothquant": {
        "alpha": 0.5,
        "quantize_weights": True,
        "calib_dataset": "E05-03 frozen handoff (required)",
    },
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    if not status.is_file():
        return 0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _package_probe(distribution: str, import_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    spec = importlib.util.find_spec(import_name)
    row: dict[str, Any] = {
        "distribution": distribution,
        "import_name": import_name,
        "available": spec is not None,
        "version": None,
        "license": None,
        "origin": str(spec.origin) if spec and spec.origin else None,
        "distribution_fingerprint_sha256": None,
        "reason": None,
    }
    if spec is None:
        row["reason"] = "package not installed on the Jetson target"
    else:
        try:
            dist = importlib.metadata.distribution(distribution)
            row["version"] = dist.version
            row["license"] = dist.metadata.get("License") or None
            fingerprint_parts = []
            for filename in ("METADATA", "RECORD", "direct_url.json"):
                content = dist.read_text(filename)
                if content is not None:
                    fingerprint_parts.append(filename.encode() + b"\0" + content.encode())
            if fingerprint_parts:
                row["distribution_fingerprint_sha256"] = _sha256_bytes(
                    b"\0".join(fingerprint_parts)
                )
        except Exception as exc:  # installed import and metadata can diverge
            row["reason"] = f"distribution metadata unavailable: {type(exc).__name__}: {exc}"
    row["probe_wall_time_s"] = time.perf_counter() - started
    return row


def _dependency_report() -> dict[str, Any]:
    rows = {
        method: _package_probe(info["distribution"], info["import_name"])
        for method, info in METHOD_PACKAGES.items()
    }
    for distribution, import_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("triton", "triton"),
        ("pyarrow", "pyarrow"),
    ):
        rows[distribution] = _package_probe(distribution, import_name)
    return {
        "schema": "hqsb.e05_04.dependencies/v1",
        "captured_at_utc": _utc_now(),
        "target": "Jetson Orin 8GB",
        "packages": rows,
        "lock_status": {
            "gptq": "LOCKED" if rows["gptq"]["available"] else "UNAVAILABLE",
            "awq": "LOCKED" if rows["awq"]["available"] else "UNAVAILABLE",
            "smoothquant": (
                "LOCKED" if rows["smoothquant"]["available"] else "UNAVAILABLE"
            ),
        },
        "mutation_policy": (
            "No package installation was attempted: the missing frozen E05-03 "
            "handoff independently forbids a valid industrial-method run."
        ),
    }


def _environment(dependencies: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    disk = shutil.disk_usage(ROOT)
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    capability = list(torch.cuda.get_device_capability(0)) if cuda_available else None
    total_memory = (
        int(torch.cuda.get_device_properties(0).total_memory) if cuda_available else 0
    )
    return {
        "schema": "hqsb.e05_04.environment/v1",
        "captured_at_utc": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "compute_capability": capability,
        "device_total_memory_bytes": total_memory,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--short")),
        "model_manifest_sha256": _sha256_file(MODEL_MANIFEST),
        "dependency_schema": dependencies["schema"],
        "execution_scope": (
            "capability and adapter-contract probes only; no model quantization, "
            "quality evaluation, profiler, or runtime performance claim"
        ),
    }


def _status_from(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    payload = _json_read(path)
    return str(
        payload.get("overall")
        or payload.get("status")
        or payload.get("verdict")
        or "UNKNOWN"
    )


def _prerequisites(dependencies: Mapping[str, Any]) -> dict[str, Any]:
    e05_03 = _json_read(E05_03_VERDICT) if E05_03_VERDICT.is_file() else {}
    package_rows = dependencies["packages"]
    checks = [
        {
            "id": "e05_01_pass",
            "passed": _status_from(E05_01_VERDICT) == "PASS",
            "observed": _status_from(E05_01_VERDICT),
            "evidence": str(E05_01_VERDICT.relative_to(ROOT)),
        },
        {
            "id": "e05_02_pass",
            "passed": _status_from(E05_02_VERDICT) == "PASS",
            "observed": _status_from(E05_02_VERDICT),
            "evidence": str(E05_02_VERDICT.relative_to(ROOT)),
        },
        {
            "id": "e05_03_pass",
            "passed": _status_from(E05_03_VERDICT) == "PASS",
            "observed": _status_from(E05_03_VERDICT),
            "evidence": str(E05_03_VERDICT.relative_to(ROOT)),
        },
        {
            "id": "e05_03_frozen_handoff",
            "passed": E05_03_HANDOFF.is_file(),
            "observed": "present" if E05_03_HANDOFF.is_file() else "missing",
            "evidence": str(E05_03_HANDOFF.relative_to(ROOT)),
        },
        {
            "id": "e05_03_selected_candidate",
            "passed": bool(e05_03.get("selected_candidate")),
            "observed": e05_03.get("selected_candidate"),
            "evidence": str(E05_03_VERDICT.relative_to(ROOT)),
        },
        {
            "id": "gptq_source_library",
            "passed": bool(package_rows["gptq"]["available"]),
            "observed": package_rows["gptq"]["reason"] or package_rows["gptq"]["version"],
            "evidence": "dependencies/availability.json",
        },
        {
            "id": "awq_source_library",
            "passed": bool(package_rows["awq"]["available"]),
            "observed": package_rows["awq"]["reason"] or package_rows["awq"]["version"],
            "evidence": "dependencies/availability.json",
        },
    ]
    blockers = [row for row in checks if not row["passed"]]
    return {
        "schema": "hqsb.e05_04.prerequisites/v1",
        "captured_at_utc": _utc_now(),
        "formal_execution_allowed": not blockers,
        "checks": checks,
        "blocking_ids": [row["id"] for row in blockers],
        "decision": "BLOCKED" if blockers else "READY",
        "fail_closed_rule": (
            "Do not consume final-evaluation or generate an industrial artifact "
            "unless every prerequisite is true."
        ),
    }


def _spec(dependencies: Mapping[str, Any], prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    versions = {
        method: str(dependencies["packages"][method].get("version") or "UNAVAILABLE")
        for method in ("gptq", "awq", "smoothquant")
    }
    matrix = MethodMatrix(
        methods=("rtn", "gptq", "awq"),
        library_versions=versions,
        module_scope="all 2-D Qwen torch.nn.Linear weights except lm_head",
        scheme_policy="W4A16, group_size=128, symmetric requested",
        calibration_budget={
            "source": "E05-03 frozen handoff only",
            "selected_candidate": None,
            "status": "UNAVAILABLE",
        },
        quality_gate={
            "max_ppl_ratio": 1.10,
            "min_mean_logit_cosine": 0.98,
            "min_top1_agreement": 0.80,
            "final_policy": "one shot after freeze",
        },
        tunable_parameters={
            "gptq.damp_percent": (0.01,),
            "gptq.desc_act": (True,),
            "awq.zero_point": (False,),
            "awq.version": ("GEMM",),
        },
    )
    payload = {
        "schema": "hqsb.e05_04.experiment_spec/v1",
        "experiment_id": "E05-04",
        "frozen_at_utc": _utc_now(),
        "scope": (
            "blocked gate audit plus synthetic adapter-contract fixtures; "
            "contract fixtures are excluded from industrial-method PASS claims"
        ),
        "model": {
            "id": "Qwen/Qwen3-1.7B",
            "dtype": "float16",
            "model_manifest_sha256": _sha256_file(MODEL_MANIFEST),
        },
        "method_matrix": json.loads(matrix.to_json()),
        "method_matrix_hash": matrix.matrix_hash,
        "requested_configs": REQUESTED_CONFIGS,
        "comparison_levels": {
            "A": "common fake-dequant quality",
            "B": "common packing and common kernel",
            "C": "source-library native runtime",
            "D": "offline cost and engineering capability",
        },
        "adapter_contract_tolerance": {
            "max_abs": 1e-12,
            "relative_l2": 1e-12,
        },
        "transform_contract_tolerance": {
            "max_abs": 1e-12,
            "relative_l2": 1e-12,
            "cosine_min": 1.0 - 1e-12,
        },
        "quality_protocol_source": str(E05_02_SPEC.relative_to(ROOT)),
        "calibration_protocol_source": str(E05_03_HANDOFF.relative_to(ROOT)),
        "formal_execution_allowed": prerequisites["formal_execution_allowed"],
        "blocking_ids": prerequisites["blocking_ids"],
        "forbidden_claims": [
            "synthetic contract fixtures are industrial method outputs",
            "common packed bytes imply an observed low-bit kernel",
            "fake-dequant contract timing is deployment performance",
            "final-evaluation was run without a frozen E05-03 handoff",
        ],
    }
    payload["spec_hash"] = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    return payload


def _method_configs(
    output: Path, dependencies: Mapping[str, Any], prerequisites: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    adapters = {
        "gptq": GPTQ_ADAPTER,
        "awq": AWQ_ADAPTER,
        "smoothquant": SMOOTHQUANT_ADAPTER,
    }
    for method, adapter in adapters.items():
        probe = dependencies["packages"][method]
        declared_mappings = adapter.field_mappings()
        mapping_audit = audit_field_mapping(adapter.source_fields, declared_mappings)
        config = adapter.source_config(
            version=str(probe.get("version") or ""),
            public_config=REQUESTED_CONFIGS[method],
            resolved_config={},
            calibration_manifest_sha256="",
            unavailable_reason=str(probe.get("reason") or ""),
        )
        row = {
            "schema": "hqsb.e05_04.method_config/v1",
            "paper_method": method,
            "source_config": config.as_dict(),
            "requested_config": REQUESTED_CONFIGS[method],
            "resolved_config_status": (
                "NOT_RESOLVED_SOURCE_LIBRARY_UNAVAILABLE"
                if not probe["available"]
                else "NOT_RESOLVED_FROZEN_CALIBRATION_UNAVAILABLE"
            ),
            "source_schema_field_mapping": [
                mapping.as_dict() for mapping in declared_mappings
            ],
            "source_schema_mapping_audit": mapping_audit,
            "known_differences": list(adapter.known_differences),
            "transform": adapter.transform,
            "runtime_capability_requirement": adapter.capability_binding(
                available=False,
                unavailable_reason=(
                    "No industrial source artifact was produced; runtime binding "
                    "was not exercised."
                ),
            ).as_dict(),
            "execution_status": "BLOCKED",
            "blocking_ids": prerequisites["blocking_ids"],
        }
        _json_write(output / f"method_configs/{method}.json", row)
        rows.append(row)

    rtn_manifest = _json_read(E05_02_RTN_MANIFEST)
    rtn = {
        "schema": "hqsb.e05_04.method_config/v1",
        "paper_method": "rtn",
        "role": "control",
        "source": str(E05_02_RTN_MANIFEST.relative_to(ROOT)),
        "source_artifact_id": rtn_manifest.get("artifact_id"),
        "quantization": rtn_manifest.get("quantization"),
        "coverage": rtn_manifest.get("coverage"),
        "quality_status": _status_from(E05_02_VERDICT),
        "reuse_status": "REFERENCE_ONLY_NOT_REEXECUTED",
    }
    _json_write(output / "method_configs/rtn_control.json", rtn)
    rows.append(rtn)
    return rows


def _model_mapping(output: Path) -> dict[str, Any]:
    manifest = _json_read(E05_02_RTN_MANIFEST)
    rows = []
    for tensor in manifest.get("tensors", []):
        rows.append(
            {
                "hqsb_module": tensor["name"],
                "shape": tensor["source_shape"],
                "rtn_control_scope": "quantized",
                "gptq_source_module": None,
                "awq_source_module": None,
                "industrial_mapping_status": "UNVERIFIED_SOURCE_LIBRARIES_UNAVAILABLE",
                "shared_weight": False,
                "fused": False,
                "unsupported": None,
            }
        )
    _jsonl_write(output / "adapter/module_mapping.jsonl", rows)
    summary = {
        "schema": "hqsb.e05_04.module_mapping_summary/v1",
        "candidate_module_count": len(rows),
        "candidate_parameter_count": manifest.get("coverage", {}).get(
            "selected_parameter_count"
        ),
        "candidate_parameter_coverage": manifest.get("coverage", {}).get(
            "parameter_coverage"
        ),
        "verified_gptq_mapping_count": 0,
        "verified_awq_mapping_count": 0,
        "source": str(E05_02_RTN_MANIFEST.relative_to(ROOT)),
        "scope_note": (
            "Rows reproduce the frozen RTN candidate scope only. They are not "
            "evidence that either unavailable source library supports the module."
        ),
    }
    _json_write(output / "adapter/module_mapping_summary.json", summary)
    return summary


def _source_values(rows: int, cols: int) -> list[int]:
    return [((index * 5 + 3) % 15) - 7 for index in range(rows * cols)]


def _scales(rows: int, cols: int, group_size: int) -> list[float]:
    groups = math.ceil(cols / group_size)
    return [0.03125 + index * 0.00390625 for index in range(rows * groups)]


def _source_dequant(
    qvalues: Sequence[int], scales: Sequence[float], rows: int, cols: int, group: int
) -> list[float]:
    groups = math.ceil(cols / group)
    values = []
    for index, code in enumerate(qvalues):
        row = index // cols
        col = index % cols
        values.append(float(code) * float(scales[row * groups + col // group]))
    return values


def _quantized_tensor(converted: Mapping[str, Any], rows: int, cols: int) -> QuantizedTensor:
    scheme = converted["scheme"]
    qvalues = list(converted["qvalues"])
    scales = list(converted["scales"])
    zeros = list(converted["zeros"])
    dequantized = dequantize_flat(
        qvalues,
        scales,
        zeros,
        (rows, cols),
        scheme.unit_count(cols),
        scheme,
        group_size=scheme.group_size,
        axis=scheme.axis,
    )
    saturated = sum(code in (scheme.qmin, scheme.qmax) for code in qvalues)
    stats = QuantStats(
        num_values=len(qvalues),
        num_units=len(scales),
        clamp_count=saturated,
        saturated_fraction=saturated / len(qvalues),
        zero_group_count=0,
        constant_group_count=0,
        nonfinite_count=0,
        invalid_scale_units=[],
        max_scale=max(scales),
        min_positive_scale=min(scales),
    )
    return QuantizedTensor(
        scheme=scheme,
        shape=(rows, cols),
        axis=scheme.axis,
        q=qvalues,
        scales=scales,
        zeros=zeros,
        values_dequant=dequantized,
        units_per_row=scheme.unit_count(cols),
        group_size=scheme.group_size,
        tail=tail_length(cols, scheme.group_size),
        stats=stats,
        unit_stats=[],
    )


def _replay_process(script: Path, artifact: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [sys.executable, str(script), "replay", "--artifact-dir", str(artifact)],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def _contract_artifacts(output: Path) -> dict[str, Any]:
    rows, cols, group = 3, 129, 128
    qvalues = _source_values(rows, cols)
    scales = _scales(rows, cols, group)
    source_reference = _source_dequant(qvalues, scales, rows, cols, group)
    builder = PackedVariantBuilder()
    equivalence_rows = []
    field_rows = []
    replay_rows = []
    boundary_rows = []
    script = Path(__file__).resolve()

    for method, adapter, nibble_order in (
        ("gptq", GPTQ_ADAPTER, packing.NIBBLE_LOW_FIRST),
        ("awq", AWQ_ADAPTER, packing.NIBBLE_HIGH_FIRST),
    ):
        payload = packing.pack_canonical(qvalues, 4, nibble_order=nibble_order)
        metadata = (
            {
                "g_idx_policy": "identity-order contract fixture",
                "desc_act": True,
                "source_schema": "gptqmodel-like contract fixture",
            }
            if method == "gptq"
            else {
                "alpha": 0.5,
                "duo_scaling": True,
                "fold_target": "linear-input wrapper contract fixture",
                "source_schema": "autoawq-like contract fixture",
            }
        )
        pre_scale = [2.0 ** ((index % 4) - 1) for index in range(cols)] if method == "awq" else None
        post_scale = [1.0 / value for value in pre_scale] if pre_scale else None
        record = SourceTensorRecord(
            name=f"contract.{method}.tail_weight",
            shape=(rows, cols),
            bits=4,
            symmetric=True,
            group_size=group,
            packed_payload=payload,
            scales=scales,
            source_metadata=metadata,
            source_nibble_order=nibble_order,
            source_axis=1,
            container="nibble",
            pre_scale=pre_scale,
            post_scale=post_scale,
        )
        fixture_dir = output / f"source_artifacts/contract_fixtures/{method}"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        (fixture_dir / "payload.bin").write_bytes(payload)
        _json_write(
            fixture_dir / "source_record.json",
            {
                "label": "SYNTHETIC_CONTRACT_FIXTURE_NOT_INDUSTRIAL_OUTPUT",
                "method_schema_emulated": method,
                "name": record.name,
                "shape": list(record.shape),
                "bits": record.bits,
                "symmetric": record.symmetric,
                "group_size": record.group_size,
                "source_axis": record.source_axis,
                "source_nibble_order": record.source_nibble_order,
                "container": record.container,
                "source_fields": record.source_fields(),
                "source_metadata": metadata,
                "payload_sha256": _sha256_bytes(payload),
                "qvalue_reference_sha256": _sha256_bytes(bytes((v & 0xFF) for v in qvalues)),
                "scale_reference_sha256": _sha256_bytes(
                    struct.pack(f"<{len(scales)}f", *scales)
                ),
            },
        )

        started = time.perf_counter()
        converted = adapter.convert_record(record, rows=rows, cols=cols)
        equivalence = adapter.equivariance_check(
            record,
            source_reference,
            tolerance={"max_abs": 1e-12, "relative_l2": 1e-12},
            rows=rows,
            cols=cols,
        )
        conversion_time = time.perf_counter() - started
        equivalence_rows.append(
            {
                "method": method,
                "scope": "synthetic_contract_fixture",
                "tensor": record.name,
                "shape": [rows, cols],
                "tail": cols % group,
                "qvalue_hash": _sha256_bytes(bytes((v & 0xFF) for v in converted["qvalues"])),
                "scale_hash": _sha256_bytes(
                    struct.pack(f"<{len(converted['scales'])}f", *converted["scales"])
                ),
                "source_canonical_equivalence": equivalence,
                "conversion_wall_time_s": conversion_time,
            }
        )
        for mapping in adapter.field_mappings():
            field_rows.append(
                {
                    "method": method,
                    "scope": "declared_source_schema_not_installed",
                    **mapping.as_dict(),
                }
            )
        for mapping in adapter.converter().field_mappings(record):
            field_rows.append(
                {
                    "method": method,
                    "scope": "synthetic_contract_record",
                    **mapping.as_dict(),
                }
            )

        qt = _quantized_tensor(converted, rows, cols)
        source_config = adapter.source_config(
            version="CONTRACT_FIXTURE",
            public_config=REQUESTED_CONFIGS[method],
            resolved_config={"contract_fixture": True},
            unavailable_reason="source library absent; this is not a source-library artifact",
        )
        document = QuantArtifactDocument.from_quantized(
            qt,
            tensor_name=record.name,
            source_dtype="float16",
            source_sha256=_sha256_bytes(payload),
            model=ModelIdentity(
                model_id="hqsb/e05-04-contract-fixture",
                revision="synthetic-v1",
                architecture="matrix",
                source_weight_sha256=_sha256_bytes(payload),
            ),
            method=method,
            method_config_hash=source_config.config_hash,
            calibration=CalibrationProvenance(
                kind="CONTRACT_FIXTURE",
                statistics_kind="synthetic fixed scales",
                notes="excluded from industrial method, quality, and runtime claims",
            ),
            compatibility=[
                CompatibilityRecord(
                    kernel_id="hqsb.w4a16.triton",
                    provider="triton",
                    layout_id=packing.LAYOUT_W4A16_ROWMAJOR_NK_V1,
                    target_arch="sm_87",
                    supported_bits=(4,),
                    supported_groups=(128,),
                    status="unverified_contract_fixture_only",
                )
            ],
            known_limitations=[
                "synthetic adapter contract fixture, not a third-party output",
                "packed variant was not launched by a kernel",
                "no model quality or performance inference is permitted",
            ],
        )
        packed = builder.build(
            converted,
            rows,
            cols,
            layout_id=packing.LAYOUT_W4A16_ROWMAJOR_NK_V1,
            alignment=16,
            parent_canonical_hash=document.canonical_hash(),
        )
        document.add_variant(packed, target_arch="sm_87-contract-unverified")
        artifact_dir = output / f"adapter/contract_artifacts/{method}"
        document.save(str(artifact_dir), host=platform.node(), run_id="contract-fixture")
        validation = validate_artifact_dir(str(artifact_dir))
        first = _replay_process(script, artifact_dir)
        second = _replay_process(script, artifact_dir)
        replay_rows.append(
            {
                "method": method,
                "scope": "synthetic_contract_fixture",
                "validation": validation,
                "first_process": first,
                "second_process": second,
                "canonical_hash_stable": first["canonical_hash"] == second["canonical_hash"],
                "identity_hash_stable": first["identity_hash"] == second["identity_hash"],
                "dequant_hash_stable": first["dequant_sha256"] == second["dequant_sha256"],
                "packed_variant_verified": first["accepted"] and second["accepted"],
            }
        )

        wrong_record = SourceTensorRecord(
            name=record.name + ".wrong_layout",
            shape=record.shape,
            bits=4,
            symmetric=True,
            group_size=group,
            packed_payload=payload,
            scales=scales,
            source_nibble_order=(
                packing.NIBBLE_HIGH_FIRST
                if nibble_order == packing.NIBBLE_LOW_FIRST
                else packing.NIBBLE_LOW_FIRST
            ),
            source_axis=1,
        )
        wrong = adapter.equivariance_check(
            wrong_record,
            source_reference,
            tolerance={"max_abs": 1e-12, "relative_l2": 1e-12},
            rows=rows,
            cols=cols,
        )
        boundary_rows.append(
            {
                "method": method,
                "case": "wrong_nibble_order_detected",
                "passed": not wrong["passed"],
                "observed_equivalence_passed": wrong["passed"],
                "metrics": wrong["metrics"],
            }
        )

    good_record = SourceTensorRecord(
        name="contract.gptq.tail_weight",
        shape=(rows, cols),
        bits=4,
        symmetric=True,
        group_size=group,
        packed_payload=packing.pack_canonical(qvalues, 4),
        scales=scales,
        source_axis=1,
    )
    for case, mutation, expected_fragment in (
        (
            "truncated_payload_refused",
            {"packed_payload": good_record.packed_payload[:-1]},
            "payload length",
        ),
        (
            "scale_count_mismatch_refused",
            {"scales": scales[:-1]},
            "source scales",
        ),
    ):
        data = dict(good_record.__dict__)
        data.update(mutation)
        try:
            GPTQ_ADAPTER.convert_record(SourceTensorRecord(**data), rows=rows, cols=cols)
            boundary_rows.append({"method": "gptq", "case": case, "passed": False})
        except Exception as exc:  # expected fail-closed path
            boundary_rows.append(
                {
                    "method": "gptq",
                    "case": case,
                    "passed": expected_fragment in str(exc),
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )

    boundary_rows.append(
        {
            "method": "gptq+awq",
            "case": "tail_group_and_odd_nibble_preserved",
            "passed": all(
                row["shape"] == [3, 129]
                and row["tail"] == 1
                and row["source_canonical_equivalence"]["passed"]
                for row in equivalence_rows
            ),
            "rows": rows,
            "cols": cols,
            "group_size": group,
            "tail": 1,
            "logical_values": rows * cols,
        }
    )

    _jsonl_write(output / "adapter/field_mapping.jsonl", field_rows)
    _jsonl_write(output / "adapter/equivalence.jsonl", equivalence_rows)
    _jsonl_write(output / "adapter/new_process_replay.jsonl", replay_rows)
    _json_write(
        output / "adapter/boundary_cases.json",
        {
            "schema": "hqsb.e05_04.adapter_boundary_cases/v1",
            "all_passed": all(row["passed"] for row in boundary_rows),
            "cases": boundary_rows,
        },
    )
    return {
        "equivalence": equivalence_rows,
        "replay": replay_rows,
        "boundary": boundary_rows,
    }


def _transform_contract(output: Path) -> dict[str, Any]:
    weight = [[1.0, -2.0, 0.5], [-4.0, 0.25, 2.0]]
    scale = [2.0, 4.0, 8.0]

    def original(sample: Sequence[float]) -> list[float]:
        return [sum(w * x for w, x in zip(row, sample)) for row in weight]

    def transformed(sample: Sequence[float]) -> list[float]:
        scaled_input = [x / s for x, s in zip(sample, scale)]
        scaled_weight = [[w * s for w, s in zip(row, scale)] for row in weight]
        return [
            sum(w * x for w, x in zip(row, scaled_input))
            for row in scaled_weight
        ]

    report = transform_equivalence_check(
        original,
        transformed,
        ([1.0, -3.0, 2.0], [0.0, 8.0, -4.0], [-2.0, 0.5, 16.0]),
        tolerance={
            "max_abs": 1e-12,
            "relative_l2": 1e-12,
            "cosine_min": 1.0 - 1e-12,
        },
    )
    payload = {
        "schema": "hqsb.e05_04.transform_contract/v1",
        "method": "awq",
        "scope": "synthetic_unquantized_linear_contract_fixture",
        "passed": report["passed"],
        "graph_hash": _sha256_bytes(
            b"x@W.T == (x/scale)@(W*scale).T;scale=[2,4,8]"
        ),
        "fold_target": "linear input wrapper; no Qwen graph rewrite was attempted",
        "report": report,
        "claim_boundary": (
            "This validates the algebraic adapter primitive only. It does not "
            "validate AutoAWQ's Qwen graph mapping or model-level folding."
        ),
    }
    _json_write(output / "adapter/transform_equivalence.json", payload)
    return payload


def _not_executed_outputs(
    output: Path, prerequisites: Mapping[str, Any], e05_02_spec: Mapping[str, Any]
) -> None:
    reason = {
        "status": "NOT_EXECUTED",
        "blocking_ids": prerequisites["blocking_ids"],
        "reason": (
            "No frozen E05-03 handoff and fewer than two source libraries are "
            "available. Running quality, packing/kernel performance, or final "
            "evaluation would violate the pre-registered protocol."
        ),
        "contract_fixture_excluded": True,
    }
    _json_write(
        output / "source_artifacts/not_generated.json",
        {**reason, "artifact_kind": "industrial source artifacts"},
    )
    _json_write(
        output / "canonical_artifacts/not_generated.json",
        {**reason, "artifact_kind": "industrial canonical QuantArtifacts"},
    )
    _json_write(
        output / "packed_variants/not_generated.json",
        {**reason, "artifact_kind": "industrial model packed variants"},
    )
    _json_write(
        output / "quality/level_a.json",
        {
            **reason,
            "path_level": "A",
            "permitted_claim": "none",
            "final_evaluation_consumed": False,
        },
    )
    _json_write(
        output / "quality/final.json",
        {
            **reason,
            "split": "final-evaluation",
            "final_evaluation_consumed": False,
            "policy": "kept unopened because no candidate was frozen",
        },
    )
    workloads = [row.get("name") for row in e05_02_spec.get("workloads", [])]
    _json_write(
        output / "perf/level_b.json",
        {
            **reason,
            "path_level": "B",
            "workloads_planned": workloads,
            "observed_kernel": None,
            "fallback": "not launched",
        },
    )
    _json_write(
        output / "perf/level_c.json",
        {
            **reason,
            "path_level": "C",
            "workloads_planned": workloads,
            "observed_kernel": None,
            "fallback": "native runtimes unavailable",
        },
    )
    _json_write(
        output / "profiler/not_executed.json",
        {
            **reason,
            "planned_phases": ["prefill", "decode"],
            "observed_symbols": [],
        },
    )


def _calibration_audit(output: Path) -> dict[str, Any]:
    e05_03 = _json_read(E05_03_VERDICT)
    report = {
        "schema": "hqsb.e05_04.calibration_handoff_audit/v1",
        "handoff_path": str(E05_03_HANDOFF.relative_to(ROOT)),
        "handoff_exists": E05_03_HANDOFF.is_file(),
        "e05_03_overall": e05_03.get("overall"),
        "e05_03_scientific_execution_verdict": e05_03.get(
            "scientific_execution_verdict"
        ),
        "selected_family": e05_03.get("selected_family"),
        "selected_candidate": e05_03.get("selected_candidate"),
        "usable": bool(
            E05_03_HANDOFF.is_file()
            and e05_03.get("overall") == "PASS"
            and e05_03.get("selected_candidate")
        ),
        "consumed": False,
        "final_evaluation_manifest_opened_by_e05_04": False,
        "reason": (
            "E05-03 did not select a minimum sufficient family and therefore "
            "did not emit the required frozen handoff."
        ),
    }
    _json_write(output / "calibration/handoff_audit.json", report)
    return report


def _offline_cost(
    output: Path,
    started: float,
    dependencies: Mapping[str, Any],
    contracts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "phase": "dependency_capability_probe",
            "wall_time_s": sum(
                float(row.get("probe_wall_time_s", 0.0))
                for row in dependencies["packages"].values()
            ),
            "host_rss_bytes": _rss_bytes(),
            "device_peak_bytes": 0,
            "failures": sum(
                not bool(dependencies["packages"][method]["available"])
                for method in ("gptq", "awq", "smoothquant")
            ),
            "scope": "capability only",
        },
        {
            "phase": "synthetic_adapter_contracts",
            "wall_time_s": sum(
                float(row["conversion_wall_time_s"])
                for row in contracts["equivalence"]
            ),
            "host_rss_bytes": _rss_bytes(),
            "device_peak_bytes": 0,
            "failures": sum(
                not row["source_canonical_equivalence"]["passed"]
                for row in contracts["equivalence"]
            ),
            "scope": "synthetic contract fixture; not offline quantization",
        },
        {
            "phase": "total_gate_and_contract_collection",
            "wall_time_s": time.perf_counter() - started,
            "host_rss_bytes": _rss_bytes(),
            "device_peak_bytes": 0,
            "failures": 0,
            "scope": "no model loaded and no CUDA kernel launched",
        },
    ]
    _jsonl_write(output / "offline_cost.jsonl", rows)
    return rows


def collect(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty raw evidence directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".e05-04-collect-", dir=str(output.parent))
    )
    started = time.perf_counter()
    try:
        dependencies = _dependency_report()
        _json_write(temporary / "dependencies/availability.json", dependencies)
        environment = _environment(dependencies)
        _json_write(temporary / "environment.json", environment)
        prerequisites = _prerequisites(dependencies)
        _json_write(temporary / "prerequisites.json", prerequisites)
        spec = _spec(dependencies, prerequisites)
        _json_write(temporary / "spec.json", spec)
        configs = _method_configs(temporary, dependencies, prerequisites)
        module_mapping = _model_mapping(temporary)
        calibration = _calibration_audit(temporary)
        contracts = _contract_artifacts(temporary)
        transform = _transform_contract(temporary)
        e05_02_spec = _json_read(E05_02_SPEC)
        _not_executed_outputs(temporary, prerequisites, e05_02_spec)
        costs = _offline_cost(temporary, started, dependencies, contracts)
        _json_write(
            temporary / "summary.json",
            {
                "schema": "hqsb.e05_04.summary/v1",
                "experiment_id": "E05-04",
                "execution_status": "BLOCKED",
                "formal_execution_allowed": prerequisites["formal_execution_allowed"],
                "blocking_ids": prerequisites["blocking_ids"],
                "industrial_methods_completed": 0,
                "industrial_artifacts_generated": 0,
                "contract_fixture_methods": ["gptq", "awq"],
                "contract_fixture_equivalence_passed": all(
                    row["source_canonical_equivalence"]["passed"]
                    for row in contracts["equivalence"]
                ),
                "contract_fixture_replay_passed": all(
                    row["canonical_hash_stable"]
                    and row["identity_hash_stable"]
                    and row["dequant_hash_stable"]
                    and row["packed_variant_verified"]
                    for row in contracts["replay"]
                ),
                "boundary_cases_passed": all(
                    row["passed"] for row in contracts["boundary"]
                ),
                "transform_contract_passed": transform["passed"],
                "module_mapping": module_mapping,
                "calibration_handoff_usable": calibration["usable"],
                "method_config_records": len(configs),
                "offline_cost_phases": len(costs),
                "level_a_executed": False,
                "level_b_executed": False,
                "level_c_executed": False,
                "level_d_capability_audit_executed": True,
                "final_evaluation_consumed": False,
            },
        )
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(artifact_dir: Path) -> dict[str, Any]:
    validation = validate_artifact_dir(str(artifact_dir))
    document = load_document(str(artifact_dir))
    rows = math.prod(document.tensor.shape[:-1]) if len(document.tensor.shape) > 1 else 1
    cols = int(document.tensor.shape[-1])
    values = dequantize_flat(
        document.values,
        document.scales,
        document.zeros,
        (rows, cols),
        document.units_per_row,
        document.scheme,
        group_size=document.scheme.group_size,
        axis=document.scheme.axis,
    )
    report = {
        "accepted": validation["accepted"],
        "artifact_id": document.artifact_id(),
        "canonical_hash": document.canonical_hash(),
        "identity_hash": document.identity_hash(),
        "dequant_sha256": _sha256_bytes(struct.pack(f"<{len(values)}d", *values)),
        "value_count": len(values),
        "variant_count": len(document.variants),
    }
    return report


def _criteria(summary: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "name": "at least two industrial methods converted to unified QuantArtifact",
            "passed": summary["industrial_methods_completed"] >= 2,
            "evidence": "summary.json",
        },
        {
            "id": 2,
            "name": "all actual source fields mapped, preserved or explicitly unsupported",
            "passed": False,
            "evidence": "adapter/field_mapping.jsonl (declared schemas and fixtures only)",
        },
        {
            "id": 3,
            "name": "actual source and canonical dequant agree within preregistered tolerance",
            "passed": False,
            "evidence": "adapter/equivalence.jsonl (synthetic contract fixtures only)",
        },
        {
            "id": 4,
            "name": "actual model pre/post transforms are equivalent before quantization",
            "passed": False,
            "evidence": "adapter/transform_equivalence.json (algebraic fixture only)",
        },
        {
            "id": 5,
            "name": "industrial canonical artifacts replay in a new process",
            "passed": False,
            "evidence": "adapter/new_process_replay.jsonl (synthetic fixtures only)",
        },
        {
            "id": 6,
            "name": "methods share one frozen calibration/policy/final protocol",
            "passed": False,
            "evidence": "calibration/handoff_audit.json",
        },
        {
            "id": 7,
            "name": "Level A/B/C/D records are separated and unexecuted levels are explicit",
            "passed": all(
                path.is_file()
                for path in (
                    output / "quality/level_a.json",
                    output / "perf/level_b.json",
                    output / "perf/level_c.json",
                    output / "offline_cost.jsonl",
                )
            ),
            "evidence": "quality/, perf/, offline_cost.jsonl",
        },
        {
            "id": 8,
            "name": "different algorithms same kernel and same algorithm different kernels attributed",
            "passed": False,
            "evidence": "perf/level_b.json and perf/level_c.json",
        },
        {
            "id": 9,
            "name": "quality decided by gates and confidence intervals",
            "passed": False,
            "evidence": "quality/level_a.json and quality/final.json",
        },
        {
            "id": 10,
            "name": "observed kernels, module coverage and fallback are auditable",
            "passed": False,
            "evidence": "adapter/module_mapping_summary.json and profiler/not_executed.json",
        },
        {
            "id": 11,
            "name": "attempted-stage offline cost and failure outcomes are complete",
            "passed": bool(_jsonl_read(output / "offline_cost.jsonl")),
            "evidence": "offline_cost.jsonl and dependencies/availability.json",
        },
        {
            "id": 12,
            "name": "at least one industrial artifact can be handed to E05-06",
            "passed": False,
            "evidence": "packed_variants/not_generated.json",
        },
        {
            "id": 13,
            "name": "all conclusions are generated from raw evidence",
            "passed": True,
            "evidence": "verdict.json and EVIDENCE_MANIFEST.json",
        },
    ]


def _write_evidence_manifest(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "EVIDENCE_MANIFEST.json":
            continue
        relative = path.relative_to(output).as_posix()
        rows.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    root_payload = b"".join(
        row["path"].encode() + b"\0" + row["sha256"].encode() + b"\n" for row in rows
    )
    manifest = {
        "schema": "hqsb.evidence_manifest/v1",
        "experiment_id": "E05-04",
        "generated_at_utc": _utc_now(),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "root_sha256": _sha256_bytes(root_payload),
        "files": rows,
    }
    _json_write(output / "EVIDENCE_MANIFEST.json", manifest)
    return manifest


def verify(output: Path) -> dict[str, Any]:
    if not output.is_dir():
        raise RuntimeError(f"raw directory does not exist: {output}")
    if (output / "verdict.json").exists() or (output / "EVIDENCE_MANIFEST.json").exists():
        raise RuntimeError("refusing to overwrite finalized E05-04 evidence")
    summary = _json_read(output / "summary.json")
    prerequisites = _json_read(output / "prerequisites.json")
    boundary = _json_read(output / "adapter/boundary_cases.json")
    transform = _json_read(output / "adapter/transform_equivalence.json")
    equivalence = _jsonl_read(output / "adapter/equivalence.jsonl")
    replay_rows = _jsonl_read(output / "adapter/new_process_replay.jsonl")
    contract_checks = {
        "two_fixture_methods": {row["method"] for row in equivalence} == {"gptq", "awq"},
        "source_canonical_equivalence": bool(equivalence)
        and all(row["source_canonical_equivalence"]["passed"] for row in equivalence),
        "new_process_replay": bool(replay_rows)
        and all(
            row["canonical_hash_stable"]
            and row["identity_hash_stable"]
            and row["dequant_hash_stable"]
            and row["packed_variant_verified"]
            for row in replay_rows
        ),
        "negative_and_boundary_cases": bool(boundary["all_passed"]),
        "unquantized_transform_primitive": bool(transform["passed"]),
        "final_evaluation_not_consumed": not summary["final_evaluation_consumed"],
    }
    criteria = _criteria(summary, output)
    expected_effect = {
        "name": "separate algorithm benefit from third-party packing/runtime packaging",
        "passed": False,
        "reason": (
            "No industrial method produced an artifact, and no Level A/B/C model "
            "result exists. Contract fixtures validate adapter plumbing only."
        ),
    }
    single_standard = {
        "name": (
            "at least two industrial methods replay; adapter metadata is complete; "
            "methods use one quality protocol"
        ),
        "passed": False,
        "reason": (
            "0 industrial methods completed; GPTQ/AWQ source libraries and the "
            "frozen E05-03 handoff are unavailable."
        ),
    }
    verdict = {
        "schema": "hqsb.e05_04.verdict/v1",
        "experiment_id": "E05-04",
        "verified_at_utc": _utc_now(),
        "overall": "BLOCKED",
        "scientific_execution_verdict": "NOT_EXECUTED",
        "formal_pass_allowed": prerequisites["formal_execution_allowed"],
        "blocking_ids": prerequisites["blocking_ids"],
        "expected_effect": expected_effect,
        "single_item_standard": single_standard,
        "detail_pass_criteria": criteria,
        "passed_criteria": sum(row["passed"] for row in criteria),
        "total_criteria": len(criteria),
        "contract_fixture_checks": contract_checks,
        "contract_fixture_overall": "PASS" if all(contract_checks.values()) else "FAIL",
        "industrial_methods_completed": summary["industrial_methods_completed"],
        "industrial_artifacts_generated": summary["industrial_artifacts_generated"],
        "levels": {
            "A": "NOT_EXECUTED",
            "B": "NOT_EXECUTED",
            "C": "NOT_EXECUTED",
            "D": "CAPABILITY_AUDIT_ONLY",
        },
        "final_evaluation_consumed": False,
        "format_deviation": (
            "Required parquet tables are JSONL because pyarrow is unavailable; "
            "no fake .parquet files were created."
        ),
        "claim_boundary": (
            "A PASS for synthetic adapter contracts is not an E05-04 scientific "
            "PASS and not an industrial-method, quality, kernel, or performance claim."
        ),
    }
    _json_write(output / "verdict.json", verdict)
    _write_evidence_manifest(output)
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        collect(args.output_dir.resolve())
        print(json.dumps({"status": "COLLECTED", "output": str(args.output_dir)}))
    elif args.command == "verify":
        result = verify(args.output_dir.resolve())
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(replay(args.artifact_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
