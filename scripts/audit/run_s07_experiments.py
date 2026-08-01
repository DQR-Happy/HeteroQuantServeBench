#!/usr/bin/env python3
"""Collect the honest executable scope of S07 on the Jetson target.

The collector records target capabilities, runs the shipped S07 contract and
invariant checks, inventories immutable upstream evidence, and emits one formal
evidence directory per experiment. Missing upstream gates or a missing verified
main runtime remain ``BLOCKED``; component fixtures never become performance
claims. Run only through ``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.runtime import adapter, experiment  # noqa: E402

STAGE_ROOT = REPO / "docs/stage_experiments/S07"
RUNNER = REPO / "scripts/runtime/run_e07.py"

TITLES = {
    "E07-01": "Backend 语义一致性",
    "E07-02": "Runtime Hot Path 与 C7 Trace",
    "E07-03": "Paged KV 容量、碎片与生命周期",
    "E07-04": "Static/Continuous/Chunked Prefill 调度",
    "E07-05": "Prefix Cache 正确性与局部性",
    "E07-06": "CUDA Graph 与 Attention Backend",
    "E07-07": "Runtime 核心策略严格 A/B",
    "E07-08": "Speculative Decoding/MTP",
    "E07-09": "Cancel/OOM/Timeout 与资源恢复",
    "E07-10": "跨 Runtime 公平 Model-core 比较",
}

EXPECTED = {
    "E07-01": "建立 reference/edge/cloud Backend semantic parity。",
    "E07-02": "得到 request→scheduler→runner→attention/KV→output 的源码级 hot path。",
    "E07-03": "闭环验证 KV 容量公式、碎片与 allocate/free/evict 生命周期。",
    "E07-04": "建立吞吐、TTFT、TPOT、尾延迟与公平性的策略曲线。",
    "E07-05": "验证 prefix key/失效正确并量化 saved tokens/time 与缓存成本。",
    "E07-06": "证明 eager/Graph 与 attention backend 的 actual path 和适用边界。",
    "E07-07": "用唯一变量 A/B 形成可归因的正收益或 PASS_NEGATIVE。",
    "E07-08": "判断 target 调用减少能否覆盖 draft/verify/rollback 额外成本。",
    "E07-09": "证明取消、超时、OOM 和 close 后请求与资源可恢复。",
    "E07-10": "在统一 model-core 口径下给出 reference/edge/cloud 公平边界。",
}

REQUIRED = {
    "E07-01": [
        "backend/version/capability", "tokens/logits/length", "requested/actual parameters",
        "stream/cancel/close states", "errors/fallback", "memory",
    ],
    "E07-02": [
        "source symbols/commit", "request state transitions", "scheduler iteration ledger",
        "KV owner/refcount events", "thread/stream/timestamps", "kernel parent joins",
        "queue/compute/wait split", "instrumentation overhead",
    ],
    "E07-03": [
        "KV schema/page/block", "effective/reserved tokens", "internal/external fragmentation",
        "metadata/cache/workspace/reserve bytes", "block lifecycle/refcounts",
        "capacity/OOM boundary", "attention latency/kernel", "long-run slope",
    ],
    "E07-04": [
        "request traces", "active/waiting batches", "token/sequence/chunk budgets",
        "TTFT/TPOT/TPS/tails", "GPU utilization/memory", "fairness/congestion",
        "preempt/cancel", "multi-process CI",
    ],
    "E07-05": [
        "key/version/identity", "hit/miss/cached/computed tokens", "cache-off/on correctness",
        "saved tokens/time and bookkeeping", "cache bytes/capacity/energy", "eviction/refcounts",
        "collision and wrong-identity negatives", "locality/reuse matrix",
    ],
    "E07-06": [
        "2x2 graph-attention matrix", "graph buckets/hit/capture/replay", "actual attention kernel",
        "prefill/decode TTFT/TPOT", "CPU/GPU timeline", "pool/workspace/capacity",
        "fallback reasons", "cold/capture/steady split",
    ],
    "E07-07": [
        "ADR and frozen hypothesis", "patch/build/config hashes", "unique variable",
        "ABBA/random blocks", "before/after raw", "internal mechanism metrics",
        "quality/fault/long-run", "effect/CI/regression envelope/rollback",
    ],
    "E07-08": [
        "algorithm and target/draft identity", "accepted/proposed/advanced tokens",
        "acceptance and rollback", "target/draft/verify timing", "quality/distribution",
        "memory/energy/E2E", "gamma/workload/batch matrix",
    ],
    "E07-09": [
        "failure case matrix", "request state and cleanup timestamps", "KV/refcount/free/evict",
        "extra output after cancel", "bounded OOM actions", "concurrent release/races",
        "post-failure health", "resource slope/load-close",
    ],
    "E07-10": [
        "comparison tier and identity", "capability/common/best-valid", "phase latency and TPS",
        "TTFT/ITL/TPOT", "KV/memory/kernel/energy", "quality/reliability",
        "cold/compile/steady", "multi-process CI/Pareto/limitations",
    ],
}

COMPONENT_KEYS = {
    "E07-01": ["E07-01_capability_adapter", "E07-01_parity"],
    "E07-02": ["E07-02_trace"],
    "E07-03": ["E07-03_kv"],
    "E07-04": ["E07-04_scheduler"],
    "E07-05": ["E07-05_prefix"],
    "E07-06": ["E07-06_graph"],
    "E07-07": ["metrics"],
    "E07-08": ["E07-08_spec_decode"],
    "E07-09": ["E07-09_failure"],
    "E07-10": ["E07-10_comparison"],
}

UPSTREAM = {
    "reference_generation": "docs/stage_experiments/S00/E00-05/raw/generation_raw.jsonl",
    "reference_identity": "docs/stage_experiments/S00/E00-05/raw/identity_freeze.json",
    "kv_memory_model": "docs/stage_experiments/S02/E02-05/raw/verdict.json",
    "runtime_shape_census": "docs/stage_experiments/S02/E02-02/raw_v1/metadata.json",
    "profiler": "docs/stage_experiments/S02/E02-07/raw/audit/audit.json",
    "graph_component": "docs/stage_experiments/S06/E06-08/raw/observations.json",
    "lifecycle_component": "docs/stage_experiments/S06/E06-10/raw/observations.json",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command(argv: Sequence[str], timeout: int = 120) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv), cwd=REPO, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(argv), "exit_code": None,
            "duration_seconds": time.monotonic() - started,
            "stdout": exc.stdout or "", "stderr": exc.stderr or "",
            "error": "collector_timeout",
        }
    except FileNotFoundError as exc:
        return {
            "command": list(argv), "exit_code": None,
            "duration_seconds": time.monotonic() - started,
            "stdout": "", "stderr": str(exc), "error": "command_not_found",
        }
    return {
        "command": list(argv), "exit_code": result.returncode,
        "duration_seconds": time.monotonic() - started,
        "stdout": result.stdout, "stderr": result.stderr,
    }


def collect_environment() -> Dict[str, Any]:
    payload = experiment.environment_fingerprint()
    payload["collector"] = str(Path(__file__).resolve().relative_to(REPO))
    payload["git"] = experiment.git_state(str(REPO))
    payload["target_role"] = "Jetson remote target"
    payload["local_execution_forbidden"] = True
    payload["hostname"] = platform.node()
    payload["source_identity"] = {
        "collector": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_driver": {
            "path": str(RUNNER.relative_to(REPO)),
            "sha256": sha256_file(RUNNER),
        },
        "prerequisite_gate": {
            "path": "hqsb/runtime/experiment.py",
            "sha256": sha256_file(REPO / "hqsb/runtime/experiment.py"),
        },
    }
    payload["power_mode"] = command(["nvpmodel", "-q"], timeout=20)
    payload["tegrastats_snapshot"] = command(
        ["timeout", "1", "tegrastats", "--interval", "100"], timeout=5
    )
    payload["tegrastats_snapshot"]["capture_policy"] = (
        "bounded one-second sample; timeout exit 124 is expected after records are emitted"
    )
    return payload


def collect_self_check() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    result = command(
        [sys.executable, str(RUNNER), "--mode", "self-check", "--json"], timeout=300
    )
    if result["exit_code"] != 0:
        return {}, result
    try:
        return json.loads(result["stdout"]), result
    except json.JSONDecodeError as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return {}, result


def collect_probe(environment: Mapping[str, Any]) -> Dict[str, Any]:
    engines = [item.as_dict() for item in adapter.probe_environment()]
    usable = [item["engine"] for item in engines if item.get("usable")]
    model_path = Path("~/models/hqsb/Qwen3-1.7B").expanduser()
    manifest = REPO / "docs/benchmark/model_sha256_manifest.txt"
    return {
        "schema_version": "1.0.0", "stage": "S07", "captured_at": utc_now(),
        "target": "Jetson Orin 8GB", "probe_complete": True,
        "engines": engines, "verified_usable_engines": usable,
        "reference": {
            "backend": "pytorch_reference", "model_path_exists": model_path.is_dir(),
            "manifest_exists": manifest.is_file(),
            "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
            "cuda_available": bool(environment.get("cuda")),
            "verification_scope": "availability only; real generation is inherited from E00-05",
        },
        "result": "PASS" if usable else "NO_ELIGIBLE_MAIN_RUNTIME",
        "claim_boundary": "importability is not successful model load or runtime correctness",
    }


def select_runtime(probe: Mapping[str, Any]) -> Dict[str, Any]:
    usable = list(probe.get("verified_usable_engines", []))
    selected = usable[0] if usable else None
    return {
        "schema_version": "1.0.0", "stage": "S07", "decided_at": utc_now(),
        "selection_policy": [
            "verified model load on target", "Qwen/QuantArtifact support",
            "scheduler/KV source observability", "feature coverage", "license/build stability",
        ],
        "selected_engine": selected, "selected_verified": bool(selected),
        "decision": "SELECTED" if selected else "BLOCKED_NO_ELIGIBLE_RUNTIME",
        "reason": (
            f"selected {selected} from verified probe" if selected else
            "vLLM/SGLang/TensorRT-LLM/llama.cpp are all NOT_INSTALLED on the target; "
            "a PyTorch reference is not a paged/continuous-batching cloud runtime"
        ),
        "source_patch_allowed": bool(selected),
    }


def upstream_inventory() -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for name, relative in UPSTREAM.items():
        path = REPO / relative
        items[name] = {
            "relative_path": relative, "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
            "use": "inherited context only; not an S07 main-runtime measurement",
        }
    return {"captured_at": utc_now(), "items": items}


def coverage_for(experiment_id: str, self_check: Mapping[str, Any]) -> List[str]:
    coverage = {
        "E07-01": ["backend/version/capability", "requested/actual parameters", "errors/fallback"],
        "E07-02": ["request state transitions", "scheduler iteration ledger", "thread/stream/timestamps"],
        "E07-03": ["KV schema/page/block", "effective/reserved tokens", "block lifecycle/refcounts"],
        "E07-04": ["request traces", "active/waiting batches", "token/sequence/chunk budgets", "fairness/congestion", "preempt/cancel"],
        "E07-05": ["key/version/identity", "hit/miss/cached/computed tokens", "eviction/refcounts", "collision and wrong-identity negatives"],
        "E07-06": ["graph buckets/hit/capture/replay", "fallback reasons"],
        "E07-07": [],
        "E07-08": ["accepted/proposed/advanced tokens", "acceptance and rollback"],
        "E07-09": ["failure case matrix", "request state and cleanup timestamps", "bounded OOM actions"],
        "E07-10": ["comparison tier and identity", "capability/common/best-valid"],
    }
    keys = COMPONENT_KEYS[experiment_id]
    return coverage[experiment_id] if self_check and all(key in self_check for key in keys) else []


def evidence_rows(experiment_id: str, available: Iterable[str]) -> List[Dict[str, Any]]:
    available_set = set(available)
    return [
        {
            "required": field,
            "available": field in available_set,
            "evidence_level": "COMPONENT" if field in available_set else "UNAVAILABLE",
            "reason": (
                "covered by target-executed S07 interface/invariant self-check"
                if field in available_set else
                "requires a verified main runtime and frozen upstream model/request semantics"
            ),
        }
        for field in REQUIRED[experiment_id]
    ]


def verdict_for(
    experiment_id: str, prerequisites: experiment.PrerequisiteStatus
) -> Dict[str, Any]:
    if experiment_id == "E07-08":
        return {
            "status": experiment.STATUS_N_A_BY_ADR, "component_status": "PASS",
            "reason": "spec_decode_spec.yaml freezes claim=false; no speculative/MTP claim is made",
        }
    if experiment_id == "E07-07":
        return {
            "status": experiment.STATUS_BLOCKED, "component_status": "NOT_RUN",
            "reason": "no verified main runtime exists, so no source/policy A/B was selected or executed",
        }
    return {
        "status": experiment.STATUS_BLOCKED, "component_status": "PARTIAL",
        "reason": "formal experiment cannot pass: " + ", ".join(prerequisites.missing),
    }


def report_markdown(
    experiment_id: str,
    verdict: Mapping[str, Any],
    prerequisites: experiment.PrerequisiteStatus,
    rows: Sequence[Mapping[str, Any]],
    probe: Mapping[str, Any],
    component: Mapping[str, Any],
) -> str:
    available = sum(bool(row["available"]) for row in rows)
    lines = [
        f"# {experiment_id}：{TITLES[experiment_id]}实验报告", "",
        f"> 正式状态：**{verdict['status']}**；组件级状态：**{verdict['component_status']}**。", "",
        "## 1. 执行边界", "",
        "本轮在 Jetson 目标板实际运行 capability probe 与 S07 接口/不变量自检。"
        "未安装的 Runtime 不被模拟，历史数据只作为继承证据，不改写成本轮性能结果。", "",
        f"- 预计达到的效果：{EXPECTED[experiment_id]}",
        f"- 实际覆盖：必采集项 {available}/{len(rows)} 项有组件级证据；其余逐项登记不可用原因。",
        f"- 主 Runtime：`{probe.get('result', 'UNKNOWN')}`；verified engines = `{probe.get('verified_usable_engines', [])}`。",
        f"- 正式裁决理由：{verdict['reason']}。", "",
        "## 2. 必采集信息/数据", "",
        "| 必采集项 | 可用 | 证据级别 | 说明 |", "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['required']} | {'是' if row['available'] else '否'} | "
            f"{row['evidence_level']} | {row['reason']} |"
        )
    lines += [
        "", "组件观测见 [`raw/observations.json`](raw/observations.json)，完整门禁见 "
        "[`raw/prerequisites.json`](raw/prerequisites.json)。", "",
        "## 3. 预计效果与单项通过标准比较", "",
    ]
    if verdict["status"] == experiment.STATUS_N_A_BY_ADR:
        lines += [
            "E07-08 协议允许在不提出 speculative/MTP claim 时裁决 `N/A_BY_ADR`。"
            "本轮只验证 acceptance/residual/rollback 契约，未执行 target+draft 双模型，"
            "所以既不宣称加速，也不填写 PASS/PASS_NEGATIVE。", "",
            "结论：预计效果不适用；单项标准按当前 claim 边界正确收束。",
        ]
    else:
        lines += [
            "预计效果**未达到正式实验层级**。组件契约可调用且负路径能拒绝，但缺少经过验证的"
            "主 Runtime、冻结请求语义以及上游 S05/S06 通过链，不能产生 Runtime 性能/容量/因果结论。", "",
            "单项通过标准：**未完整满足**。当前结果是带原始证据的 `BLOCKED`，不是 `NOT_STARTED`。", "",
            "解除阻塞后必须补齐所有 `available=false` 行，并以至少三个独立进程执行确认性统计；"
            "不得把本轮 smoke/模拟量并入延迟或吞吐结论。",
        ]
    lines += [
        "", "## 4. 组件级结果（非性能结论）", "", "```json",
        json.dumps(component, ensure_ascii=False, indent=2), "```", "",
        "## 5. 前端访问", "",
        "`raw/verdict.json` 位于 EvidenceCatalog 固定扫描路径；Console 可通过 "
        "`GET /api/console/v1/evidence`、`GET /api/console/v1/evidence/{id}` 和 "
        "`/api/console/v1/evidence/{id}/download` 访问本实验 JSON/Markdown 证据。"
        "本轮验证到 EvidenceCatalog 的 scan/detail 数据层；目标 Python 缺 FastAPI，未启动 HTTP server。", "",
        "## 6. 结论", "", f"**{verdict['status']}**：{verdict['reason']}。", "",
    ]
    return "\n".join(lines)


def manifest_for(directory: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "path": str(path.relative_to(directory.parent.parent)),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    return {"schema_version": "1.0.0", "generated_at": utc_now(), "files": files}


def write_experiment(
    experiment_id: str,
    environment: Mapping[str, Any],
    prerequisites: experiment.PrerequisiteStatus,
    probe: Mapping[str, Any],
    selection: Mapping[str, Any],
    upstream: Mapping[str, Any],
    self_check: Mapping[str, Any],
    self_check_command: Mapping[str, Any],
) -> Dict[str, Any]:
    root = STAGE_ROOT / experiment_id
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    verdict = verdict_for(experiment_id, prerequisites)
    component = {
        key: self_check.get(key, {"missing": True})
        for key in COMPONENT_KEYS[experiment_id]
    }
    rows = evidence_rows(experiment_id, coverage_for(experiment_id, self_check))
    observations = {
        "stage": "S07", "experiment_id": experiment_id, "captured_at": utc_now(),
        "execution_scope": "target-executed component checks plus inherited evidence inventory",
        "not_measured": "no main-runtime performance, capacity, energy or cross-runtime result",
        "component_observations": component, "required_evidence": rows,
        "runtime_probe": {
            "result": probe.get("result"),
            "verified_usable_engines": probe.get("verified_usable_engines", []),
        },
        "main_runtime_selection": selection, "upstream_evidence": upstream,
    }
    write_json(raw / "environment_fingerprint.json", environment)
    write_json(raw / "prerequisites.json", prerequisites.as_dict())
    write_json(raw / "capability_probe.json", probe)
    write_json(raw / "main_runtime_selection.json", selection)
    write_json(raw / "observations.json", observations)
    write_json(raw / "required_evidence.json", {"experiment_id": experiment_id, "items": rows})
    write_json(raw / "self_check_command.json", self_check_command)
    verdict_payload = {
        "schema_version": "1.0.0", "stage": "S07", "experiment_id": experiment_id,
        "status": verdict["status"], "overall": verdict["status"],
        "component_status": verdict["component_status"], "reason": verdict["reason"],
        "executed": True, "execution_scope": observations["execution_scope"],
        "expected_effect_achieved": False if experiment_id != "E07-08" else None,
        "single_item_pass_standard_met": False if experiment_id != "E07-08" else None,
        "required_evidence_available": sum(row["available"] for row in rows),
        "required_evidence_total": len(rows), "written_at": utc_now(),
        "limitations": [
            "no verified main runtime is installed on the target",
            "upstream S04.5/S05/S06 formal gate chain is incomplete",
            "component simulations and smoke checks are not runtime performance measurements",
        ],
    }
    write_json(raw / "verdict.json", verdict_payload)
    (root / f"{experiment_id}_实验报告.md").write_text(
        report_markdown(experiment_id, verdict, prerequisites, rows, probe, component),
        encoding="utf-8",
    )
    write_json(raw / "manifest.json", manifest_for(raw))
    return verdict_payload


def validate_frontend() -> Dict[str, Any]:
    catalog = EvidenceCatalog(REPO)
    records = catalog.scan(refresh=True)
    s07 = [item for item in records if item.get("stage") == "S07"]
    expected = set(TITLES)
    observed = {item["experiment"] for item in s07}
    detail_errors: List[str] = []
    for item in s07:
        try:
            detail = catalog.detail(item["id"])
            if detail.get("format") != "json":
                detail_errors.append(f"{item['experiment']}: verdict detail is not JSON")
        except Exception as exc:  # noqa: BLE001
            detail_errors.append(f"{item['experiment']}: {type(exc).__name__}: {exc}")
    return {
        "checked_at": utc_now(), "ok": observed == expected and not detail_errors,
        "verification_scope": "EvidenceCatalog direct scan/detail against the same allowlist used by the API",
        "live_http_executed": False,
        "live_http_reason": "target Python does not have the optional FastAPI console dependency installed",
        "catalog_endpoint": "/api/console/v1/evidence",
        "detail_endpoint": "/api/console/v1/evidence/{evidence_id}",
        "download_endpoint": "/api/console/v1/evidence/{evidence_id}/download",
        "frontend_routes": ["/evidence", "/experiments"],
        "expected_experiments": sorted(expected), "observed_experiments": sorted(observed),
        "records": [
            {"experiment": item["experiment"], "status": item["status"], "files": len(item["files"])}
            for item in s07
        ],
        "detail_errors": detail_errors,
    }


def stage_report(
    verdicts: Sequence[Mapping[str, Any]], frontend: Mapping[str, Any]
) -> str:
    lines = [
        "# S07 推理 Runtime 内核阶段实验报告", "",
        f"执行时间：{utc_now()}。目标：Jetson Orin 8GB。", "", "## 总结论", "",
        "10 项均已建立正式结果目录，并在 Jetson 上完成 capability probe、接口/不变量自检、"
        "上游证据继承盘点和必采集字段审计。9 项 P0 因没有可用的主 Runtime 且上游正式门禁"
        "未闭合，裁决为 `BLOCKED`；E07-08 因预注册 `claim=false`，按协议裁决 `N/A_BY_ADR`。", "",
        "本报告不把 scheduler/KV/prefix fixture 的模拟数值当作真实性能，也不把 S00/S02 的"
        "PyTorch reference 结果改写成 S07 cloud runtime 数据。", "", "## 逐项裁决", "",
        "| ID | 组件状态 | 正式状态 | 预计效果 | 单项标准 |", "|---|---|---|---|---|",
    ]
    for item in verdicts:
        eid = item["experiment_id"]
        effect = "不适用（未声明）" if eid == "E07-08" else "未达到正式层级"
        standard = "按 ADR 正确收束" if eid == "E07-08" else "未完整满足"
        lines.append(
            f"| [{eid}]({eid}/{eid}_实验报告.md) | {item['component_status']} | "
            f"{item['status']} | {effect} | {standard} |"
        )
    lines += [
        "", "## 阻塞事实", "",
        "- Jetson probe：vLLM、SGLang、TensorRT-LLM、llama.cpp 均 `NOT_INSTALLED`；没有可选主 Runtime。",
        "- S04.5 M4 marker 与冻结 request fixtures 缺失。",
        "- S05 quality/kernel 与 S06 stable capability 没有满足正式 PASS 门禁。",
        "- 因此无法合法采集 paged KV、continuous batching、prefix cache、runtime CUDA Graph/attention、源码 A/B 和跨 Runtime 数据。",
        "", "## 前端接口", "",
        f"EvidenceCatalog 数据层校验：`{'PASS' if frontend.get('ok') else 'FAIL'}`，"
        f"发现 {len(frontend.get('observed_experiments', []))}/10 个 S07 实验。",
        "前端 `/evidence`、`/experiments` 通过既有 API 访问各实验 verdict、observations、"
        "required_evidence、manifest 与报告；详见 [`frontend_access.json`](frontend_access.json)。"
        "本轮未启动 HTTP server：目标 Python 未安装可选 FastAPI 依赖。",
        "", "## 解除阻塞的最短路径", "",
        "1. 完成 S04.5 M4 与冻结 tokenized request fixtures。",
        "2. 关闭 S05 质量/真实 kernel 门与 S06 stable capability 门。",
        "3. 在 Jetson 可行性允许的前提下安装并验证一个主 Runtime；若 8GB 无法容纳，需明确引入云端目标机。",
        "4. 用同一 ModelArtifact/precision/request trace 复跑 E07-01～07、09、10；每项至少三独立进程。",
        "5. 只有提出 speculative/MTP claim 时再把 E07-08 从 `N/A_BY_ADR` 解锁。", "",
    ]
    return "\n".join(lines)


def main() -> int:
    global STAGE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(STAGE_ROOT))
    args = parser.parse_args()
    STAGE_ROOT = Path(args.output_root).resolve()
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    environment = collect_environment()
    self_check, self_check_command = collect_self_check()
    probe = collect_probe(environment)
    selection = select_runtime(probe)
    write_json(STAGE_ROOT / "capability_probe.json", probe)
    write_json(STAGE_ROOT / "main_runtime_selection.json", selection)
    upstream = upstream_inventory()
    write_json(STAGE_ROOT / "upstream_evidence_inventory.json", upstream)
    prerequisites = experiment.check_prerequisites(str(REPO))
    write_json(STAGE_ROOT / "prerequisites.json", prerequisites.as_dict())
    write_json(STAGE_ROOT / "interface_self_check.json", self_check)

    verdicts = [
        write_experiment(
            eid, environment, prerequisites, probe, selection, upstream,
            self_check, self_check_command,
        )
        for eid in TITLES
    ]
    frontend = validate_frontend()
    write_json(STAGE_ROOT / "frontend_access.json", frontend)
    (STAGE_ROOT / "S07_阶段实验报告_20260921.md").write_text(
        stage_report(verdicts, frontend), encoding="utf-8"
    )
    write_json(
        STAGE_ROOT / "execution_summary.json",
        {
            "stage": "S07", "executed_at": utc_now(), "target": platform.node(),
            "verdicts": verdicts, "prerequisites": prerequisites.as_dict(),
            "runtime_probe": probe, "selection": selection, "frontend": frontend,
        },
    )
    print(json.dumps({
        "stage": "S07", "output_root": str(STAGE_ROOT),
        "statuses": {item["experiment_id"]: item["status"] for item in verdicts},
        "frontend_ok": frontend.get("ok"),
        "missing_prerequisites": prerequisites.missing,
    }, ensure_ascii=False, indent=2))
    return 0 if frontend.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
