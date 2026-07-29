"""Bounded, read-only projections of historical profiling and quantization evidence.

This control-plane module deliberately has no torch/CUDA dependencies. Evidence is
selected by an explicit experiment adapter, never by a browser-supplied path. Raw
traces, executable formats and model tensors are not parsed by the API process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hqsb.console.research/v1"
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_SCAN_BYTES = 64 * 1024 * 1024
PROFILE_PATH = Path("docs/stage_experiments/S02/E02-07/raw")
QUANT_PATH = Path("docs/stage_experiments/S05/E05-02/raw")


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _scale(value: Any, divisor: float) -> float | None:
    value = _number(value)
    return value / divisor if value is not None else None


def _list(value: Any, limit: int = 64) -> list:
    return value[:limit] if isinstance(value, list) else []


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _verdict(document: dict) -> str:
    for key in ("status", "overall", "verdict"):
        if isinstance(document.get(key), str):
            return document[key]
    if isinstance(document.get("passed"), bool):
        return "PASS" if document["passed"] else "FAIL"
    return "UNKNOWN"


class ResearchCatalog:
    """Expose versioned evidence projections and allowlisted, hashed attachments."""

    def __init__(self, root: Path, *, cache_seconds: float = 30.0):
        self.root = root.resolve()
        self.cache_seconds = cache_seconds
        self._lock = threading.RLock()
        self._cached: dict | None = None
        self._scanned_at = 0.0
        self._files: dict[str, tuple[Path, dict]] = {}
        self._issues: list[dict] = []
        self._read_bytes = 0

    def _path(self, relative: Path) -> Path:
        path = self.root / relative
        # Check the lexical root as well as the symlink-resolved location.
        if not any(
            relative.is_relative_to(base) for base in (PROFILE_PATH, QUANT_PATH)
        ):
            raise ValueError("Outside the research evidence allowlist")
        resolved = path.resolve()
        if not any(
            resolved.is_relative_to(self.root / base)
            for base in (PROFILE_PATH, QUANT_PATH)
        ):
            raise ValueError("Evidence symlink escaped its experiment")
        if resolved.suffix.lower() not in {".json", ".csv"}:
            raise ValueError("Unsupported research artifact format")
        return resolved

    def _read(self, relative: Path) -> tuple[bytes, dict] | None:
        try:
            path = self._path(relative)
            size = path.stat().st_size
            if size > MAX_ARTIFACT_BYTES or self._read_bytes + size > MAX_SCAN_BYTES:
                raise ValueError("Research artifact exceeds the parsing budget")
            with path.open("rb") as stream:
                data = stream.read(MAX_ARTIFACT_BYTES + 1)
            if len(data) > MAX_ARTIFACT_BYTES:
                raise ValueError("Research artifact exceeds the parsing budget")
            self._read_bytes += len(data)
            relative_name = str(path.relative_to(self.root))
            digest = hashlib.sha256(data).hexdigest()
            key = hashlib.sha256(relative_name.encode()).hexdigest()[:24]
            record = {
                "id": key,
                "name": path.name,
                "relative_path": relative_name,
                "bytes": len(data),
                "sha256": digest,
                "format": path.suffix.lstrip("."),
            }
            self._files[key] = (path, record)
            return data, record
        except (OSError, ValueError, RuntimeError) as exc:
            self._issues.append({"path": str(relative), "reason": str(exc)})
            return None

    def _json(self, relative: Path) -> tuple[dict, dict | None]:
        result = self._read(relative)
        if result is None:
            return {}, None
        data, record = result
        try:
            document = json.loads(
                data, parse_constant=_reject_constant, parse_float=_finite_float
            )
            if not isinstance(document, dict):
                raise ValueError("Expected a JSON object")
            return document, record
        except (ValueError, UnicodeError, RecursionError) as exc:
            self._issues.append({"path": str(relative), "reason": str(exc)})
            return {}, record

    def _attachments(self, relatives: list[Path]) -> list[dict]:
        records = []
        for relative in relatives:
            result = self._read(relative)
            if result is not None:
                records.append(result[1])
        return records

    def _profiling(self) -> dict:
        summary, source = self._json(PROFILE_PATH / "summary.json")
        verdict, verdict_ref = self._json(PROFILE_PATH / "verdict.json")
        result = {
            "status": "available"
            if _mapping(summary.get("sections"))
            else "unavailable",
            "source": "E02-07",
            "verdict": _verdict(verdict),
            "limitations": [
                "历史实验，不是当前请求的在线采集；不同工具、窗口与重放数据分别标注。",
                "kernel work 为设备累计工作时间，不能与阶段墙钟相加，不能直接作为端到端加速比例。",
                "NCU 独立 shape 重放与应用重放不是原次推理；缓存刷新、频率和采集扰动影响比较。",
                "E02-07 使用禁用 CUDA 缓存分配器的历史口径；cudaMalloc/cudaFree 墙钟不能与当前默认 allocator 直接比较。",
                "Roofline 的 modeled DRAM bytes 不是实测总线字节；缺失计数器保持 null。",
                "本接口只解析有界摘要和 CSV；完整 trace 不在 API 进程中加载。",
            ],
            "phases": [],
            "kernels": [],
            "perturbation": _mapping(summary.get("perturbation")),
            "module_roles": _mapping(summary.get("module_roles")),
            "nsys": _mapping(summary.get("nsys")),
            "artifacts": [ref for ref in (source, verdict_ref) if ref],
        }
        for sample, phases in list(_mapping(summary.get("sections")).items())[:8]:
            for phase, details in list(_mapping(phases).items())[:16]:
                details = _mapping(details)
                hotspots = []
                for row in _list(details.get("top"), 30):
                    row = _mapping(row)
                    if not isinstance(row.get("name"), str):
                        continue
                    hotspots.append(
                        {
                            "name": row["name"],
                            "count": _number(row.get("count")),
                            "total_ms": _scale(row.get("total_us"), 1000),
                            "mean_us": _number(row.get("mean_us")),
                            "time_share": _number(row.get("time_share")),
                            "bucket": row.get("bucket"),
                            "ops": _list(row.get("ops")),
                            "dims": _list(row.get("dims")),
                            "streams": _list(row.get("streams")),
                            "grids": _list(row.get("grids")),
                            "blocks": _list(row.get("blocks")),
                        }
                    )
                result["phases"].append(
                    {
                        "id": f"{sample}:{phase}",
                        "sample": sample,
                        "phase": phase,
                        "span_ms": _number(details.get("span_ms")),
                        "kernel_work_ms": _number(details.get("kernel_work_ms")),
                        "kernel_count": _number(details.get("kernel_count")),
                        "idle_ratio": _number(details.get("idle_ratio")),
                        "hotspots": hotspots,
                    }
                )

        run = summary.get("run")
        if (
            not isinstance(run, str)
            or not run.startswith("run_")
            or not run[4:].isdigit()
        ):
            return result
        run_path = PROFILE_PATH / run
        ncu, ncu_ref = self._json(run_path / "ncu/ncu_summary.json")
        if ncu_ref:
            result["artifacts"].append(ncu_ref)
        result["replay_policy"] = _mapping(ncu.get("replay_mode"))
        panels = _mapping(summary.get("ncu"))
        rooflines = _mapping(summary.get("roofline"))
        for key, entry in list(ncu.items())[:24]:
            entry = _mapping(entry)
            candidate = _mapping(entry.get("candidate"))
            if not candidate:
                continue
            observations = []
            for row in _list(panels.get(key), 8):
                row = _mapping(row)
                observations.append(
                    {
                        "name": row.get("name"),
                        "grid": row.get("grid"),
                        "block": row.get("block"),
                        "metrics": {
                            name: _number(value)
                            for name, value in _mapping(row.get("panel")).items()
                        },
                        "stalls": _mapping(row.get("stalls")),
                    }
                )
            csv_name = entry.get("csv")
            refs = []
            if (
                isinstance(csv_name, str)
                and Path(csv_name).name == csv_name
                and csv_name.endswith(".csv")
            ):
                refs = self._attachments([run_path / "ncu" / csv_name])
            result["kernels"].append(
                {
                    "id": key,
                    "mode": entry.get("mode", "unknown"),
                    "sample": candidate.get("sample"),
                    "phase": candidate.get("range"),
                    "role": candidate.get("role"),
                    "kernel_regex": candidate.get("kernel_regex"),
                    "shape": _mapping(candidate.get("replay")),
                    "observations": observations,
                    "roofline": _mapping(rooflines[key]) if key in rooflines else None,
                    "artifacts": refs,
                }
            )
        nsys, nsys_ref = self._json(run_path / "nsys/nsys_summary.json")
        result["nsys_capture"] = nsys
        if nsys_ref:
            result["artifacts"].append(nsys_ref)
        result["artifacts"].extend(
            self._attachments(
                [
                    run_path / "nsys" / f"{sample}_{report}.csv"
                    for sample in ("P", "D")
                    for report in (
                        "cuda_gpu_kern_sum",
                        "cuda_api_sum",
                        "cuda_gpu_mem_time_sum",
                        "nvtx_gpu_proj_sum",
                    )
                ]
            )
        )
        return result

    def _quantization(self) -> dict:
        spec, spec_ref = self._json(QUANT_PATH / "spec.json")
        artifact_summary, artifact_ref = self._json(
            QUANT_PATH / "artifact_summary.json"
        )
        verdict, verdict_ref = self._json(QUANT_PATH / "verdict.json")
        summary, summary_ref = self._json(QUANT_PATH / "summary.json")
        methods = _mapping(spec.get("methods"))
        refs = [
            ref for ref in (spec_ref, artifact_ref, verdict_ref, summary_ref) if ref
        ]
        result = {
            "status": "available" if methods and verdict else "unavailable",
            "source": "E05-02",
            "verdict": _verdict(verdict),
            "limitations": [
                "历史质量门与运行快照，不能代表当前交互部署已支持 W8/W4。",
                "压缩制品字节、理论整模型组合、实际 allocated/reserved 分别展示，不能相加。",
                "whole_model_equivalent_bytes 是压缩制品加未量化参数的估算，不是运行内存。",
                "每个 runtime_memory 条目属于独立历史进程；allocator reserved 受缓存历史影响。",
                "storage_only 全量恢复 FP16，不构成原生低比特加速证据。质量门失败保持失败。",
            ],
            "methods": [],
            "artifacts": refs,
        }
        method_artifacts = _mapping(artifact_summary.get("methods"))
        source_load = _mapping(artifact_summary.get("source_load"))
        source_bytes = _number(source_load.get("parameter_bytes"))
        quality = _mapping(verdict.get("quality"))
        for key, method in list(methods.items())[:16]:
            if not isinstance(key, str) or not key.replace("_", "").isalnum():
                continue
            method = _mapping(method)
            artifact = _mapping(method_artifacts.get(key))
            disk = _mapping(artifact.get("disk"))
            coverage = _mapping(artifact.get("coverage"))
            selected_bytes = _number(coverage.get("selected_source_bytes"))
            quantized_bytes = _number(disk.get("total"))
            retained = (
                source_bytes - selected_bytes
                if source_bytes is not None and selected_bytes is not None
                else None
            )
            combined = (
                quantized_bytes + retained
                if quantized_bytes is not None and retained is not None
                else None
            )
            method_refs = []
            memories = []
            executions = []
            for index in range(3):
                run, ref = self._json(QUANT_PATH / "runs" / f"{key}_run_{index}.json")
                if ref:
                    method_refs.append(ref)
                if not run:
                    continue
                memory = _mapping(
                    _mapping(run.get("steady_memory_before_workloads")).get("cuda")
                )
                memories.append(
                    {
                        "run_id": run.get("run_id", f"{key}_run_{index}"),
                        "allocated_bytes": _number(memory.get("allocated")),
                        "reserved_bytes": _number(memory.get("reserved")),
                        "peak_allocated_bytes": _number(memory.get("peak_allocated")),
                    }
                )
                executions.append(_mapping(run.get("execution_truth")))
            bits = _number(method.get("bits"))
            native_values = [item.get("native_low_bit_kernel") for item in executions]
            native = (
                native_values[0]
                if native_values
                and all(
                    isinstance(value, bool) and value == native_values[0]
                    for value in native_values
                )
                else None
            )
            descriptions = [item.get("execution") for item in executions]
            description = (
                descriptions[0]
                if descriptions
                and all(
                    isinstance(value, str) and value == descriptions[0]
                    for value in descriptions
                )
                else "unknown"
            )
            # These are normalized semantics of the recorded execution, not an
            # inference from the file extension or advertised weight precision.
            execution_label = "unknown"
            if bits == 16 and executions:
                execution_label = "fp16_reference"
            elif executions and all(
                "whole-weight FP16 dequant" in str(item.get("execution"))
                and item.get("native_low_bit_kernel") is False
                for item in executions
            ):
                execution_label = "storage_only"
            profiler, profiler_ref = self._json(
                QUANT_PATH / "profiler" / key / "summary.json"
            )
            if profiler_ref:
                method_refs.append(profiler_ref)
            result["methods"].append(
                {
                    "id": key,
                    "bits": bits,
                    "group_size": _number(method.get("group_size")),
                    "execution_label": execution_label,
                    "execution_description": description,
                    "declared_execution_description": str(
                        method.get("execution", "unknown")
                    ),
                    "native_low_bit_kernel": native,
                    "storage_sizes": {
                        "fp16_source_bytes": source_bytes,
                        "quantized_bytes": quantized_bytes,
                        "retained_fp16_bytes": retained,
                        "whole_model_equivalent_bytes": combined,
                        "compression_ratio": source_bytes / combined
                        if source_bytes is not None and combined
                        else None,
                        "qvalues_bytes": _number(disk.get("qvalues")),
                        "scales_bytes": _number(disk.get("scales")),
                        "manifest_bytes": _number(disk.get("manifest")),
                    },
                    "coverage": coverage,
                    "offline_seconds": _number(
                        _mapping(artifact.get("offline")).get("wall_time_s")
                    ),
                    "quality": _mapping(quality.get(key)),
                    "runtime_memory": memories,
                    "performance": _mapping(
                        _mapping(summary.get("performance")).get(key)
                    ),
                    "profiler": {
                        name: profiler.get(name)
                        for name in (
                            "available",
                            "expected_kernel",
                            "observed_execution",
                            "native_low_bit_kernel",
                            "fallback_reason",
                            "trace_sha256",
                        )
                    },
                    "artifacts": method_refs,
                }
            )
        return result

    def overview(self, *, refresh: bool = False) -> dict:
        with self._lock:
            if (
                not refresh
                and self._cached is not None
                and time.monotonic() - self._scanned_at < self.cache_seconds
            ):
                return copy.deepcopy(self._cached)
            self._files, self._issues, self._read_bytes = {}, [], 0
            profiling = self._profiling()
            quantization = self._quantization()
            self._cached = {
                "schema_version": SCHEMA_VERSION,
                "historical": True,
                "profiling": profiling,
                "quantization": quantization,
                "capabilities": [
                    {
                        "id": "historical_kernel_evidence",
                        "status": profiling["status"],
                        "detail": "E02-07 阶段、kernel、NCU/NSys 证据；不触发硬件采集",
                    },
                    {
                        "id": "historical_quantization_evidence",
                        "status": quantization["status"],
                        "detail": "E05-02 自行量化、质量门、制品与实际执行内存",
                    },
                    {
                        "id": "live_hardware_counters",
                        "status": "unavailable",
                        "detail": "历史导入器不提供当前请求的 NCU 计数器采集",
                    },
                    {
                        "id": "byte_register_reconstruction",
                        "status": "unsupported",
                        "detail": "聚合计数器不能重建每字节或物理寄存器的完整轨迹",
                    },
                ],
                "artifacts": [record for _, record in self._files.values()],
                "issues": self._issues,
                "limits": {
                    "max_artifact_bytes": MAX_ARTIFACT_BYTES,
                    "max_scan_bytes": MAX_SCAN_BYTES,
                    "raw_traces_parsed": False,
                },
            }
            self._scanned_at = time.monotonic()
            return copy.deepcopy(self._cached)

    def artifact(self, artifact_id: str) -> tuple[Path, bytes, str]:
        """Read an indexed attachment; never accept a path or deserialize pickle."""
        with self._lock:
            self.overview()
            record = self._files.get(artifact_id)
            if record is None:
                raise KeyError(artifact_id)
            path, metadata = record
            path = self._path(Path(metadata["relative_path"]))
            if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
                raise ValueError("Research artifact is missing or too large")
            with path.open("rb") as stream:
                data = stream.read(MAX_ARTIFACT_BYTES + 1)
            digest = hashlib.sha256(data).hexdigest()
            if len(data) > MAX_ARTIFACT_BYTES or digest != metadata["sha256"]:
                raise ValueError(
                    "Research artifact changed; refresh the evidence index"
                )
            return path, data, digest
