"""Research evidence must preserve measurement scopes and reject unsafe input."""

import hashlib
import json

import pytest

from hqsb.console import research
from hqsb.console.research import PROFILE_PATH, QUANT_PATH, ResearchCatalog


def write_json(root, relative, document):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def catalog(tmp_path):
    write_json(
        tmp_path,
        PROFILE_PATH / "summary.json",
        {
            "run": "run_0",
            "sections": {
                "P_early": {
                    "prefill": {
                        "span_ms": 10,
                        "kernel_work_ms": 8,
                        "kernel_count": 2,
                        "idle_ratio": 0.2,
                        "top": [
                            {
                                "name": "gemm",
                                "total_us": 8000,
                                "mean_us": 4000,
                                "count": 2,
                                "time_share": 1,
                                "ops": ["aten::mm"],
                                "dims": ["[1,2] x [2,3]"],
                                "streams": [7],
                            }
                        ],
                    }
                }
            },
            "ncu": {
                "P1": [
                    {
                        "name": "gemm",
                        "panel": {
                            "registers_per_thread": 32,
                            "l2_hit_rate_pct": None,
                        },
                    }
                ]
            },
            "roofline": {"P1": {"modeled_dram_bytes": 128, "measured_l2_bytes": None}},
        },
    )
    write_json(tmp_path, PROFILE_PATH / "verdict.json", {"passed": True})
    write_json(
        tmp_path,
        PROFILE_PATH / "run_0/ncu/ncu_summary.json",
        {
            "P1": {
                "mode": "isolated_shape_exact_replay",
                "csv": "P1.csv",
                "candidate": {
                    "sample": "P",
                    "range": "prefill",
                    "replay": {"m": 1, "k": 2, "n": 3},
                },
            },
        },
    )
    csv_path = tmp_path / PROFILE_PATH / "run_0/ncu/P1.csv"
    csv_path.write_text('"Metric Name","Metric Value"\n"Registers Per Thread","32"\n')
    method = {
        "bits": 4,
        "group_size": 128,
        "native_low_bit_kernel": False,
        "execution": "validate + whole-weight FP16 dequant, then FP16 eager GEMM",
    }
    write_json(tmp_path, QUANT_PATH / "spec.json", {"methods": {"rtn_w4": method}})
    write_json(
        tmp_path,
        QUANT_PATH / "artifact_summary.json",
        {
            "source_load": {"parameter_bytes": 1000},
            "methods": {
                "rtn_w4": {
                    "coverage": {"selected_source_bytes": 800},
                    "disk": {
                        "qvalues": 200,
                        "scales": 10,
                        "manifest": 10,
                        "total": 220,
                    },
                }
            },
        },
    )
    write_json(
        tmp_path,
        QUANT_PATH / "verdict.json",
        {
            "status": "BLOCKED",
            "quality": {
                "rtn_w4": {
                    "passed": False,
                    "checks": {"logit_cosine": False},
                    "observed": {"minimum_logit_cosine": 0.9},
                    "thresholds": {"min_mean_logit_cosine": 0.98},
                }
            },
        },
    )
    write_json(
        tmp_path,
        QUANT_PATH / "runs/rtn_w4_run_0.json",
        {
            "run_id": "rtn_w4_run_0",
            "execution_truth": method,
            "steady_memory_before_workloads": {
                "cuda": {"allocated": 1050, "reserved": 1100, "peak_allocated": 1070},
            },
        },
    )
    return ResearchCatalog(tmp_path)


def test_absent_evidence_is_unavailable_and_never_fabricated(tmp_path):
    result = ResearchCatalog(tmp_path).overview()
    assert result["historical"] is True
    assert result["profiling"]["status"] == "unavailable"
    assert result["profiling"]["phases"] == []
    assert result["quantization"]["status"] == "unavailable"
    assert result["quantization"]["methods"] == []
    assert result["artifacts"] == []


def test_phase_units_replay_and_missing_counters_are_preserved(catalog):
    profile = catalog.overview()["profiling"]
    assert profile["status"] == "available"
    assert profile["verdict"] == "PASS"
    phase = profile["phases"][0]
    assert phase["span_ms"] == 10
    assert phase["kernel_work_ms"] == 8
    assert phase["hotspots"][0]["total_ms"] == 8
    assert phase["hotspots"][0]["time_share"] == 1
    assert phase["idle_ratio"] == 0.2
    candidate = profile["kernels"][0]
    assert candidate["mode"] == "isolated_shape_exact_replay"
    assert candidate["observations"][0]["metrics"]["l2_hit_rate_pct"] is None
    assert candidate["roofline"]["measured_l2_bytes"] is None
    assert any("allocator" in item for item in profile["limitations"])


def test_quantized_files_do_not_become_runtime_savings_or_quality_pass(catalog):
    quant = catalog.overview()["quantization"]
    assert quant["verdict"] == "BLOCKED"
    method = quant["methods"][0]
    assert method["execution_label"] == "storage_only"
    assert method["native_low_bit_kernel"] is False
    assert method["quality"]["passed"] is False
    sizes = method["storage_sizes"]
    assert sizes["fp16_source_bytes"] == 1000
    assert sizes["quantized_bytes"] == 220
    assert sizes["retained_fp16_bytes"] == 200
    assert sizes["whole_model_equivalent_bytes"] == 420
    assert sizes["compression_ratio"] == pytest.approx(1000 / 420)
    assert method["runtime_memory"][0]["allocated_bytes"] == 1050
    assert method["runtime_memory"][0]["reserved_bytes"] == 1100


def test_execution_without_run_evidence_is_unknown(catalog):
    (catalog.root / QUANT_PATH / "runs/rtn_w4_run_0.json").unlink()
    result = catalog.overview()
    assert result["quantization"]["methods"][0]["execution_label"] == "unknown"
    assert result["quantization"]["methods"][0]["native_low_bit_kernel"] is None
    assert result["quantization"]["methods"][0]["execution_description"] == "unknown"


def test_artifact_download_hash_and_changed_file_rejection(catalog):
    record = catalog.overview()["profiling"]["kernels"][0]["artifacts"][0]
    path, payload, digest = catalog.artifact(record["id"])
    assert digest == record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert path.suffix == ".csv"
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        catalog.artifact(record["id"])
    with pytest.raises(KeyError):
        catalog.artifact("../../etc/passwd")


def test_symlinks_and_oversized_artifacts_are_not_indexed(tmp_path, monkeypatch):
    raw = tmp_path / PROFILE_PATH
    raw.mkdir(parents=True)
    outside = tmp_path / "private.json"
    outside.write_text('{"sections":{"secret":{}}}')
    (raw / "summary.json").symlink_to(outside)
    monkeypatch.setattr(research, "MAX_ARTIFACT_BYTES", 100)
    (raw / "verdict.json").write_text(" " * 101)
    result = ResearchCatalog(tmp_path).overview()
    assert result["profiling"]["status"] == "unavailable"
    assert result["artifacts"] == []
    assert any("escaped" in item["reason"] for item in result["issues"])
    assert any("budget" in item["reason"] for item in result["issues"])


def test_nonfinite_or_invalid_json_does_not_enter_projection(tmp_path):
    raw = tmp_path / PROFILE_PATH
    raw.mkdir(parents=True)
    (raw / "summary.json").write_text('{"sections":NaN}')
    (raw / "verdict.json").write_text("[1,2]")
    result = ResearchCatalog(tmp_path).overview()
    assert result["profiling"]["status"] == "unavailable"
    assert result["profiling"]["verdict"] == "UNKNOWN"
    json.dumps(result, allow_nan=False)


def test_cached_response_cannot_be_mutated_by_callers(catalog):
    first = catalog.overview()
    first["profiling"]["phases"].clear()
    assert len(catalog.overview()["profiling"]["phases"]) == 1
