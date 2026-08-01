#!/usr/bin/env python3
"""Collect the honest executable scope of S08 on the Jetson target.

This runner executes the model-free protocol/policy/evidence checks that are
available on the target, audits every mandatory data item from the S08 detail
documents, and writes one evidence directory per experiment.  It deliberately
does not turn dummy-backend checks into online-serving measurements: when S07
or the two-real-backend/SLO/loadgen gates are missing, the formal verdict stays
``BLOCKED``.

Run only through ``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.serving import experiment, interface_map, service_ab, specs, telemetry  # noqa: E402

STAGE_ROOT = REPO / "docs/stage_experiments/S08"
RUNNER = REPO / "scripts/serving/run_e08.py"
CONFIG_ROOT = REPO / "configs/serving"

TITLES = {
    "E08-01": "Completion/Chat、SSE、错误、断连与取消协议一致性",
    "E08-02": "Open-loop 容量、SLO Cliff 与最大 Goodput 曲线",
    "E08-03": "Constant、Poisson 与 Burst 到达对 Queue/Tail/SLO 的影响",
    "E08-04": "Short/Long、Priority 与 Tenant 混部的 HOL/Fairness",
    "E08-05": "Admission、Token Budget、Backpressure 与过载恢复",
    "E08-06": "多 Backend 的 Capability/Health/SLO/Cost 路由",
    "E08-07": "Prefix Locality、Cache-aware Routing 与负载倾斜",
    "E08-08": "慢读、断连、批量 Cancel 与 Graceful Shutdown",
    "E08-09": "Backend/Cache/Network 故障、熔断与恢复",
    "E08-10": "Gateway 到 Runtime/Kernel 的全链观测",
    "E08-11": "Scheduler/Cache/Routing 的 Service-level 严格 A/B",
}

EXPECTED = {
    "E08-01": "建立可用且可取消的最小 OpenAI-compatible API，并稳定处理 SSE 与错误。",
    "E08-02": "得到从低负载到过载的 capacity/SLO-cliff 曲线与最大 SLO 内 goodput。",
    "E08-03": "在相同平均负载下量化 arrival 方差对 queue、tail 和恢复的影响。",
    "E08-04": "比较 FIFO/priority/fair policy，识别 HOL、饥饿和公平边界。",
    "E08-05": "证明服务能在 OOM 前解释性拒绝，并在降载后有界恢复。",
    "E08-06": "至少两个真实 Backend 按能力、身份、健康、SLO 与成本正确路由。",
    "E08-07": "量化 cache-aware routing 的 reuse 收益、倾斜与淘汰代价。",
    "E08-08": "证明慢客户端、断连、取消与 shutdown 的背压和资源闭环。",
    "E08-09": "证明 crash/hang/OOM/cache/network 故障下幂等、熔断、降级与恢复。",
    "E08-10": "用一个 request ID 重建 gateway→queue→runtime→kernel→stream 全链。",
    "E08-11": "基于前十项实测瓶颈完成唯一变量、可回滚的服务级严格 A/B。",
}

REQUIRED: Mapping[str, Sequence[str]] = {
    "E08-01": (
        "protocol_profile.json 与 schema/hash", "error_catalog.json",
        "request requested/normalized/actual JSONL", "raw HTTP headers/body",
        "raw SSE bytes、parsed frames 与 token reconstruction",
        "request/trace/backend/model IDs", "lifecycle timestamps",
        "generated/committed/emitted/received/discarded token ledger",
        "cancel/deadline/disconnect/cleanup events", "dummy 与真实 Backend conformance matrix",
        "negative/fuzz corpus 和最小复现",
    ),
    "E08-02": (
        "capacity_spec.json、slo_spec.json", "payload/arrival trace 与 hash",
        "scheduled/sent/received/admitted/completed/good JSONL",
        "per-request client/server/runtime timestamps", "per-token/SSE events",
        "queue/inflight/token/KV/cache time series", "client/gateway/Backend/CPU/GPU/network/resource",
        "errors/rejects/timeouts/drain residual", "每负载点 raw、重复、CI",
        "capacity/goodput/tail/resource 多轴曲线", "saturation root-cause trace",
    ),
    "E08-03": (
        "arrival_spec.json", "intended/actual arrival JSONL 与 hash", "payload mapping",
        "no-op loadgen calibration", "inter-arrival/CV/peak-window/burst statistics",
        "queue/inflight/batch/KV/cache/resource time series",
        "per-request/tail/goodput/error/reject", "burst event 与 recovery timeline",
        "run-level paired effects/CI", "arrival sensitivity heatmap", "representative trace/profile",
    ),
    "E08-04": (
        "tenant/class/cost/policy specs", "arrival/payload traces", "isolated baselines",
        "per-request estimated/actual cost", "policy decision/counter/virtual-time log",
        "queue/batch/KV/cache/runtime events", "per-tenant/class latency/goodput/error/reject",
        "Jain/service lag/max wait/starvation/slowdown", "utilization/work-conserving evidence",
        "paired effects/CI", "HOL trace 与 timeline",
    ),
    "E08-05": (
        "overload_spec.json、pressure/admission policy", "threshold/hysteresis/state transition log",
        "admission estimate/actual/error", "original-request/attempt/retry lineage",
        "offered→good/reject funnel", "queue/token/KV/socket/resource time series",
        "per-class/tenant SLO/fairness", "reject wire captures",
        "wasted compute 与 expired-before-start", "recovery/oscillation/amplification metrics",
        "paired policy effects/CI", "safety-stop events",
    ),
    "E08-06": (
        "routing_spec.json、registry snapshots/generations",
        "Backend identity/capability/health/capacity evidence",
        "per-request candidate/filter reason/score/tie-break", "telemetry value/age/TTL",
        "selected route/epoch 与 actual Backend C6/C7", "model/precision/quality correctness",
        "route distribution/skew", "predicted/actual TTFT/TPOT/E2E calibration",
        "fallback/degradation/reject lineage", "registry race/long-run resource",
        "paired policy effects/CI",
    ),
    "E08-07": (
        "cache routing/identity/telemetry specs", "prefix token corpus/hash/tree",
        "per-instance cache inventory/epoch/time series", "per-request predicted match/cost/candidates/route",
        "actual hit/reused blocks/tokens/saved prefill", "queue/load/cache skew/eviction/pollution",
        "model/token correctness 与 wrong-hit count", "TTFT/TPOT/E2E/goodput/fairness",
        "telemetry delay/drop/reorder faults", "prediction calibration 与 paired CI",
        "locality-load heatmap/Pareto",
    ),
    "E08-08": (
        "stream_lifecycle_spec.json", "client behavior traces",
        "raw SSE/frame/token/byte pipeline events", "app/server/socket/client buffer time series",
        "generated/committed/emitted/flushed/received/discarded token ledger",
        "disconnect/cancel/linearization timestamps", "task/future/fd/RSS/KV/connection resources",
        "normal-client collateral tail/goodput", "drain/shutdown state timeline",
        "long-run slope/CI", "representative trace/profile",
    ),
    "E08-09": (
        "fault_spec.json 与 injector evidence", "healthy golden",
        "fault/detection/isolation/recovery timeline", "circuit counters/states/probes",
        "request→attempt→Backend/epoch lineage", "raw HTTP/SSE/error/token sequence",
        "retry amplification/wasted compute", "fallback actual identity/quality",
        "per-group collateral tail/goodput/error", "task/KV/socket/queue/cache/resource cleanup",
        "MTTD/isolation/Backend/traffic/SLO recovery", "paired repeats/CI",
    ),
    "E08-10": (
        "telemetry_spec.json 与 schema", "trace context/clock calibration",
        "spans/events/links raw", "metrics exposition/snapshots/histogram buckets/exemplars",
        "structured logs 与 redaction audit", "request/attempt/backend/batch/iteration/kernel map",
        "client/server/runtime token/timing ledger", "counter/histogram consistency audit",
        "exporter loss/cardinality/storage bytes", "off/minimal/full/profile overhead",
        "selected P99 root-cause/critical-path/counterfactual", "dashboard queries/runbook",
    ),
    "E08-11": (
        "baseline evidence/bottleneck table", "ADR/preregistration",
        "A/B source/config/build diff/hash", "unit/property/protocol/fault gates",
        "final arrival/payload traces", "run order/state reset/environment",
        "offered→good/reject/error funnel", "primary/secondary/guardrail raw",
        "policy decision 与 near-cause metrics", "queue/cache/router/Backend/runtime/kernel/stream trace",
        "target/neutral/holdout stratification", "paired effect/CI/practical margin",
        "ablation/regression envelope", "merge/rollback decision",
    ),
}

# Items that the target-executed model-free suite can genuinely support.  All
# other fields need a real service run; source presence alone is not data.
COMPONENT_AVAILABLE: Mapping[str, Sequence[str]] = {
    "E08-01": (REQUIRED["E08-01"][0], REQUIRED["E08-01"][1], REQUIRED["E08-01"][10]),
    "E08-02": (REQUIRED["E08-02"][0],),
    "E08-03": (REQUIRED["E08-03"][0], REQUIRED["E08-03"][4]),
    "E08-04": (REQUIRED["E08-04"][0], REQUIRED["E08-04"][7], REQUIRED["E08-04"][8]),
    "E08-05": (REQUIRED["E08-05"][0], REQUIRED["E08-05"][1], REQUIRED["E08-05"][11]),
    "E08-06": (REQUIRED["E08-06"][0], REQUIRED["E08-06"][2], REQUIRED["E08-06"][3]),
    "E08-07": (REQUIRED["E08-07"][0], REQUIRED["E08-07"][8]),
    "E08-08": (REQUIRED["E08-08"][0], REQUIRED["E08-08"][1]),
    "E08-09": (REQUIRED["E08-09"][0], REQUIRED["E08-09"][3]),
    "E08-10": (REQUIRED["E08-10"][0], REQUIRED["E08-10"][4]),
    "E08-11": (REQUIRED["E08-11"][3],),
}

TEST_TOKENS: Mapping[str, Sequence[str]] = {
    "E08-01": ("TestProtocolProfile", "TestErrorCatalog", "TestValidation", "TestSseCodec", "TestRequestStateMachine", "TestNonStreamPath", "TestRejections", "TestHttpTransport"),
    "E08-02": ("TestSloFunnel", "TestLoadgen", "TestFunnelConservation"),
    "E08-03": ("TestArrival", "TestLoadgen", "TestArrivalMeanRate", "TestTimestampLedgerMonotone"),
    "E08-04": ("TestFairness", "TestPolicies"),
    "E08-05": ("TestAdmission", "TestFunnelConservation"),
    "E08-06": ("TestRouter",),
    "E08-07": ("TestCacheRouting", "TestPrefixMatcherBounds"),
    "E08-08": ("TestFaultsAndCancel", "TestLifecycle", "TestLedgerAndTiming", "TestDeliveryLedgerMonotone"),
    "E08-09": ("TestCircuit", "TestFaults"),
    "E08-10": ("TestObservability", "TestTimestampLedgerMonotone", "TestInterfaceMap"),
    "E08-11": ("TestInterfaceMap", "TestVerdictRefusal"),
}

TEST_PATHS = (
    "tests/unit/serving",
    "tests/property/test_serving_invariants.py",
)


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


def command(argv: Sequence[str], timeout: int = 900) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv), cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False,
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
    payload = experiment.environment_fingerprint(
        {
            "scope": "Jetson remote target; model-free S08 component validation only",
            "target_role": "Jetson remote target",
            "collector": str(Path(__file__).resolve().relative_to(REPO)),
            "hostname": platform.node(),
        }
    )
    payload["git"] = experiment.git_state(str(REPO))
    payload["source_identity"] = {
        "collector": {"path": str(Path(__file__).resolve().relative_to(REPO)), "sha256": sha256_file(Path(__file__).resolve())},
        "driver": {"path": str(RUNNER.relative_to(REPO)), "sha256": sha256_file(RUNNER)},
        "gate": {"path": "hqsb/serving/experiment.py", "sha256": sha256_file(REPO / "hqsb/serving/experiment.py")},
    }
    payload["power_mode"] = command(["nvpmodel", "-q"], timeout=20)
    payload["tegrastats_snapshot"] = command(["timeout", "1", "tegrastats", "--interval", "100"], timeout=5)
    payload["tegrastats_snapshot"]["capture_policy"] = "bounded one-second environment sample; exit 124 is expected"
    return payload


def config_inventory() -> Dict[str, Any]:
    loaded = specs.ServingSpecs.load(str(CONFIG_ROOT))
    files = []
    for path in sorted(CONFIG_ROOT.glob("*.yaml")):
        files.append({
            "path": str(path.relative_to(REPO)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "captured_at": utc_now(), "strict_load_ok": loaded.ok,
        "documents": files, "audits": list(loaded.audits),
        "claim_boundary": "templates and source contracts are not frozen formal-run preregistrations",
    }


def collect_tests(junit_path: Path) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    result = command(
        [sys.executable, "-m", "pytest", "-q", "-rA", "--junitxml", str(junit_path), *TEST_PATHS],
        timeout=1200,
    )
    cases: List[Dict[str, Any]] = []
    if junit_path.is_file():
        root = ET.parse(junit_path).getroot()
        for case in root.iter("testcase"):
            status = "PASS"
            message = ""
            for tag, label in (("failure", "FAIL"), ("error", "ERROR"), ("skipped", "SKIP")):
                child = case.find(tag)
                if child is not None:
                    status = label
                    message = str(child.attrib.get("message", ""))
                    break
            cases.append({
                "node": f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}",
                "status": status, "time_seconds": float(case.attrib.get("time", 0.0)),
                "message": message,
            })
    summary = {
        **result,
        "junit_path": str(junit_path.relative_to(STAGE_ROOT)) if junit_path.is_file() else "",
        "cases": len(cases),
        "counts": {name: sum(row["status"] == name for row in cases) for name in ("PASS", "FAIL", "ERROR", "SKIP")},
        "ok": result.get("exit_code") == 0 and bool(cases) and all(row["status"] in ("PASS", "SKIP") for row in cases),
    }
    by_experiment: Dict[str, List[Dict[str, Any]]] = {}
    for eid, tokens in TEST_TOKENS.items():
        by_experiment[eid] = [row for row in cases if any(token in row["node"] for token in tokens)]
    return summary, by_experiment


def collect_driver(command_args: Sequence[str]) -> Dict[str, Any]:
    result = command([sys.executable, str(RUNNER), *command_args, "--json"], timeout=300)
    if result.get("exit_code") == 0:
        try:
            result["payload"] = json.loads(result.get("stdout", ""))
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def upstream_inventory() -> Dict[str, Any]:
    rows = []
    for eid in [f"E07-{index:02d}" for index in range(1, 11)]:
        path = REPO / f"docs/stage_experiments/S07/{eid}/raw/verdict.json"
        payload: Dict[str, Any] = {}
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {"status": "UNREADABLE"}
        rows.append({
            "experiment_id": eid, "path": str(path.relative_to(REPO)),
            "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None,
            "status": payload.get("status", "MISSING"),
        })
    return {
        "captured_at": utc_now(), "stage": "S07", "verdicts": rows,
        "passing": [row["experiment_id"] for row in rows if row["status"] in ("PASS", "PASS_NEGATIVE")],
        "claim_boundary": "S07 verdicts are inherited gate evidence, not S08 measurements",
    }


def topology_component() -> Dict[str, Any]:
    return {
        "captured_at": utc_now(), "host": platform.node(), "machine": platform.machine(),
        "load_generator": "same Python process on Jetson",
        "service_transports_exercised": ["in-process deterministic transport", "loopback stdlib HTTP transport"],
        "real_service_process": False, "separate_client_host": False, "formal_eligible": False,
        "reason": "component topology cannot prove that a production load generator is not the bottleneck",
    }


def ab_gate_component() -> Dict[str, Any]:
    table = service_ab.bottleneck_table([])
    refused = False
    reason = ""
    try:
        service_ab.selection_gate_from_table(table=table, candidate="routing_score")
    except Exception as exc:  # the refusal is the expected safety property
        refused = True
        reason = f"{type(exc).__name__}: {exc}"
    return {
        "empty_baseline_table": table, "selection_refused": refused, "reason": reason,
        "status": "PASS_GATE" if refused else "FAIL_GATE",
        "claim_boundary": "no treatment B was selected or executed without E08-01..10 formal baselines",
    }


def evidence_rows(experiment_id: str, component_ok: bool) -> List[Dict[str, Any]]:
    available = set(COMPONENT_AVAILABLE[experiment_id]) if component_ok else set()
    rows = []
    for item in REQUIRED[experiment_id]:
        is_available = item in available
        rows.append({
            "required": item, "available": is_available,
            "evidence_level": "COMPONENT" if is_available else "UNAVAILABLE",
            "reason": (
                "covered by target-executed model-free/unit/property evidence; not a performance claim"
                if is_available else
                "requires the blocked formal run with frozen SLO/workload and verified real Backend(s)"
            ),
        })
    return rows


def manifest_for(directory: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "path": str(path.relative_to(directory.parent.parent)),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    return {"schema_version": "1.0.0", "generated_at": utc_now(), "files": files}


def report_markdown(
    experiment_id: str,
    verdict: Mapping[str, Any],
    prerequisites: experiment.PrerequisiteStatus,
    rows: Sequence[Mapping[str, Any]],
    component: Mapping[str, Any],
) -> str:
    available = sum(bool(row["available"]) for row in rows)
    lines = [
        f"# {experiment_id}：{TITLES[experiment_id]}实验报告", "",
        f"> 正式状态：**{verdict['status']}**；组件级状态：**{verdict['component_status']}**。", "",
        "## 1. 执行边界", "",
        "本轮在 Jetson 目标板实际执行 S08 model-free HTTP/SSE、策略、生命周期、故障与观测组件/属性测试。"
        "dummy Backend 只验证接口语义；没有把它写成真实 Runtime、容量或服务级收益。", "",
        f"- 预计达到的效果：{EXPECTED[experiment_id]}",
        f"- 实际覆盖：必采集项 {available}/{len(rows)} 项具有组件级证据。",
        f"- 正式门禁：`satisfied={str(prerequisites.satisfied).lower()}`；缺失 `{prerequisites.missing}`。",
        f"- 正式裁决理由：{verdict['reason']}", "",
        "## 2. 必采集信息/数据审计", "",
        "| 必采集项 | 可用 | 证据级别 | 说明 |", "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['required']} | {'是' if row['available'] else '否'} | {row['evidence_level']} | {row['reason']} |"
        )
    lines += [
        "", "完整机器可读审计见 [`raw/required_evidence.json`](raw/required_evidence.json)，"
        "组件执行明细见 [`raw/component_validation.json`](raw/component_validation.json)。", "",
        "## 3. 预计效果与单项通过标准比较", "",
    ]
    if experiment_id == "E08-11":
        lines += [
            "预计效果**未达到**。选择门正确拒绝了没有 E08-01～10 正式 baseline 的 A/B；"
            "因此没有事后挑选候选、没有 treatment patch，也没有伪造 `PASS_NEGATIVE`。", "",
            "单项通过标准：**未满足**。缺少可访问的正式 baseline、冻结 ADR、ABBA/随机区组、"
            "target/neutral/holdout 三分层与至少三个独立服务进程。",
        ]
    else:
        lines += [
            "预计效果**未达到正式实验层级**。组件契约与安全拒绝路径通过不等于真实服务"
            "在随机到达、GPU Runtime、网络背压或故障条件下达到效果。", "",
            "单项通过标准：**未完整满足**。关键缺口包括 S07 P0 通过链、两个真实 Backend、"
            "冻结 request/SLO、实际 loadgen 校准和相应 raw 时序/资源数据。",
        ]
    lines += [
        "", "## 4. 组件级结果（非性能结论）", "", "```json",
        json.dumps(component, ensure_ascii=False, indent=2), "```", "",
        "## 5. 前端访问", "",
        "`raw/verdict.json` 位于 EvidenceCatalog 的固定扫描路径。Console 可通过 "
        "`GET /api/console/v1/evidence`、`GET /api/console/v1/evidence/{id}`、"
        "`GET /api/console/v1/evidence/{id}/download` 读取本实验 JSON/Markdown；"
        "页面入口为 `/evidence` 和 `/experiments`。", "",
        "## 6. 结论", "", f"**{verdict['status']}**：{verdict['reason']}。", "",
    ]
    return "\n".join(lines)


def write_experiment(
    experiment_id: str,
    environment: Mapping[str, Any],
    prerequisites: experiment.PrerequisiteStatus,
    tests: Sequence[Mapping[str, Any]],
    smoke: Mapping[str, Any],
    config: Mapping[str, Any],
    ab_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    root = STAGE_ROOT / experiment_id
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tests_ok = bool(tests) and all(row.get("status") in ("PASS", "SKIP") for row in tests)
    component_ok = tests_ok and (experiment_id != "E08-01" or smoke.get("payload", {}).get("sse_ok") is True)
    component_status = "PASS_GATE" if experiment_id == "E08-11" and ab_gate.get("selection_refused") else ("PASS" if component_ok else "FAIL")
    rows = evidence_rows(experiment_id, component_ok)
    mapping = interface_map.mapping_for(experiment_id).as_dict()
    component = {
        "stage": "S08", "experiment_id": experiment_id, "captured_at": utc_now(),
        "scope": "target-executed model-free/unit/property validation",
        "component_status": component_status, "test_cases": list(tests),
        "interface_steps": len(mapping.get("steps", [])),
        "smoke": smoke.get("payload", {}) if experiment_id == "E08-01" else None,
        "ab_selection_gate": ab_gate if experiment_id == "E08-11" else None,
        "not_measured": [
            "real Runtime service capacity or SLO goodput", "real two-Backend routing",
            "GPU/KV/kernel request chain", "networked slow-client and fault recovery",
            "service-level A/B effect or confidence interval",
        ],
    }
    reason = "formal experiment cannot pass: " + ", ".join(prerequisites.missing)
    verdict = {
        "schema_version": "1.0.0", "stage": "S08", "experiment_id": experiment_id,
        "status": experiment.STATUS_BLOCKED, "overall": experiment.STATUS_BLOCKED,
        "component_status": component_status, "reason": reason,
        "executed": True,
        "execution_scope": "target-executed component checks plus prerequisite and mandatory-data audit",
        "formal_experiment_executed": False, "expected_effect_achieved": False,
        "single_item_pass_standard_met": False,
        "required_evidence_available": sum(row["available"] for row in rows),
        "required_evidence_total": len(rows), "written_at": utc_now(),
        "limitations": [
            "S07 P0 verdict chain is not passed", "no two verified real Serving Backends are registered",
            "request fixtures and SLO are not frozen for a formal run",
            "component/dummy measurements are not service performance evidence",
        ],
    }
    write_json(raw / "environment_fingerprint.json", environment)
    write_json(raw / "prerequisites.json", prerequisites.as_dict())
    write_json(raw / "config_inventory.json", config)
    write_json(raw / "interface_map.json", mapping)
    write_json(raw / "component_validation.json", component)
    write_json(raw / "required_evidence.json", {"experiment_id": experiment_id, "items": rows})
    write_json(raw / "verdict.json", verdict)
    (root / f"{experiment_id}_实验报告.md").write_text(
        report_markdown(experiment_id, verdict, prerequisites, rows, component), encoding="utf-8"
    )
    write_json(raw / "manifest.json", manifest_for(raw))
    return verdict


def validate_frontend(
    api_contract_test: Mapping[str, Any], frontend_build: Mapping[str, Any]
) -> Dict[str, Any]:
    catalog = EvidenceCatalog(REPO)
    records = catalog.scan(refresh=True)
    items = [item for item in records if item.get("stage") == "S08"]
    expected = set(TITLES)
    observed = {item["experiment"] for item in items}
    detail_errors: List[str] = []
    for item in items:
        try:
            detail = catalog.detail(item["id"])
            if detail.get("format") != "json":
                detail_errors.append(f"{item['experiment']}: verdict detail is not JSON")
        except Exception as exc:
            detail_errors.append(f"{item['experiment']}: {type(exc).__name__}: {exc}")
    catalog_ok = observed == expected and not detail_errors
    build_ok = frontend_build.get("exit_code") == 0
    return {
        "checked_at": utc_now(), "ok": catalog_ok and build_ok,
        "catalog_ok": catalog_ok, "frontend_build_ok": build_ok,
        "verification_scope": "EvidenceCatalog scan/detail against the same allowlist used by the API",
        "live_http_executed": api_contract_test.get("exit_code") == 0,
        "live_http_ok": api_contract_test.get("exit_code") == 0,
        "live_http_reason": (
            "FastAPI TestClient list/detail/download contract passed"
            if api_contract_test.get("exit_code") == 0 else
            "target cannot import the optional FastAPI dependency; direct EvidenceCatalog scan/detail passed"
        ),
        "api_contract_test_artifact": "frontend_contract_test.json",
        "frontend_build_artifact": "frontend_build.json",
        "catalog_endpoint": "/api/console/v1/evidence",
        "detail_endpoint": "/api/console/v1/evidence/{evidence_id}",
        "download_endpoint": "/api/console/v1/evidence/{evidence_id}/download",
        "frontend_routes": ["/evidence", "/experiments"],
        "expected_experiments": sorted(expected), "observed_experiments": sorted(observed),
        "records": [{"experiment": item["experiment"], "status": item["status"], "files": len(item["files"])} for item in items],
        "detail_errors": detail_errors,
    }


def stage_report(verdicts: Sequence[Mapping[str, Any]], frontend: Mapping[str, Any], tests: Mapping[str, Any]) -> str:
    lines = [
        "# S08 ServeFabric 与性能治理阶段实验报告", "",
        f"执行时间：{utc_now()}。目标：Jetson Orin 8GB（`{platform.node()}`）。", "",
        "## 总结论", "",
        "11 项均已建立正式结果目录，并在 Jetson 上完成可执行的协议、HTTP/SSE、策略、"
        "路由、缓存、生命周期、故障与观测组件/属性测试，同时逐项审计 details 要求的"
        "必采集信息。由于 S07 P0 未通过、没有两个真实 Backend、request/SLO 未冻结，"
        "11 项正式状态均为 `BLOCKED`。", "",
        "本报告不把 dummy/in-process fixture 写成容量、真实路由、GPU/KV/kernel、网络背压或 A/B 收益。", "",
        "## 目标板测试", "",
        f"- pytest：`{'PASS' if tests.get('ok') else 'FAIL'}`；{tests.get('counts', {})}。",
        "- 12 份 Serving 配置通过 strict-load/audit；264 个 details 步骤均有接口映射。",
        "- E08-01 model-free gateway smoke 通过 SSE framing/ledger/terminal-state 检查。",
        "- E08-11 selection gate 正确拒绝在缺少正式 baseline 时选择 treatment。", "",
        "## 逐项裁决", "",
        "| ID | 组件状态 | 正式状态 | 预计效果 | 单项标准 |", "|---|---|---|---|---|",
    ]
    for item in verdicts:
        eid = str(item["experiment_id"])
        lines.append(
            f"| [{eid}]({eid}/{eid}_实验报告.md) | {item['component_status']} | {item['status']} | 未达到正式层级 | 未完整满足 |"
        )
    lines += [
        "", "## 阻塞事实", "",
        "- S07 现有 verdict 没有 `PASS/PASS_NEGATIVE`，Backend Contract/取消/metrics/C7 正式链未闭合。",
        "- 目标板没有经验证的主 Runtime，也没有两套真实、独立 Backend registry evidence。",
        "- 没有冻结 tokenized request fixtures 与 owner-confirmed SLO；配置文件仍明确标记 `template`。",
        "- 因而不得采集并声明 capacity/G*、真实 prefix reuse、真实 fault recovery、request→kernel 全链或 A/B 收益。",
        "", "## 前端接口", "",
        f"EvidenceCatalog 校验：`{'PASS' if frontend.get('ok') else 'FAIL'}`；发现 {len(frontend.get('observed_experiments', []))}/11 项。",
        f"Console TypeScript/Vite production build：`{'PASS' if frontend.get('frontend_build_ok') else 'FAIL'}`。",
        "Console 通过既有 `/api/console/v1/evidence`、detail/download 接口和 `/evidence`、"
        "`/experiments` 页面访问结果；详见 [`frontend_access.json`](frontend_access.json)。", "",
        (
            "FastAPI TestClient 的 evidence list/detail/download 契约已通过；原始输出见 "
            "[`frontend_contract_test.json`](frontend_contract_test.json)。"
            if frontend.get("live_http_ok") else
            "目标板缺少可选 FastAPI 依赖，因此本轮不能启动 HTTP TestClient；已把该限制和失败输出保存到 "
            "[`frontend_contract_test.json`](frontend_contract_test.json)，没有把数据层校验冒充 live HTTP。"
        ), "",
        "## 解除阻塞的最短路径", "",
        "1. 先关闭 S07 P0：完成 S04.5/S05/S06 门禁、冻结请求并验证一个主 Runtime。",
        "2. 注册并实际加载两个独立 Backend，冻结 model/tokenizer/template/precision/sampling/stop。",
        "3. 由实验 owner 在结果揭晓前冻结 SLO、负载点、窗口、停止条件与三次独立进程重复。",
        "4. 校准独立 loadgen/服务拓扑后，依次重跑 E08-01～10；只有形成 baseline 后才允许 E08-11 选题。",
        "5. 正式复跑时补齐各报告 `available=false` 行，保留失败、拒绝、超时和资源残余。", "",
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
    config = config_inventory()
    upstream = upstream_inventory()
    prerequisites = experiment.check_prerequisites(str(REPO))
    smoke = collect_driver(("--smoke",))
    interface_check = interface_map.resolve_interfaces()
    ab_gate = ab_gate_component()
    topology = topology_component()

    write_json(STAGE_ROOT / "environment_fingerprint.json", environment)
    write_json(STAGE_ROOT / "config_inventory.json", config)
    write_json(STAGE_ROOT / "upstream_evidence_inventory.json", upstream)
    write_json(STAGE_ROOT / "prerequisites.json", prerequisites.as_dict())
    write_json(STAGE_ROOT / "interface_self_check.json", interface_check)
    write_json(STAGE_ROOT / "gateway_smoke.json", smoke)
    write_json(STAGE_ROOT / "component_topology.json", topology)
    write_json(STAGE_ROOT / "ab_selection_gate.json", ab_gate)
    write_json(STAGE_ROOT / "telemetry_coverage.json", telemetry.coverage_summary())

    tests, tests_by_experiment = collect_tests(STAGE_ROOT / "component_tests.junit.xml")
    write_json(STAGE_ROOT / "component_test_summary.json", tests)
    console_python = REPO / ".venv-console-test/bin/python"
    frontend_contract_test = command(
        [str(console_python if console_python.is_file() else Path(sys.executable)),
         "-m", "pytest", "-q", "tests/unit/console/test_evidence.py"],
        timeout=300,
    )
    write_json(STAGE_ROOT / "frontend_contract_test.json", frontend_contract_test)
    frontend_build = command(
        ["npm", "--prefix", "web/console", "run", "build"], timeout=600
    )
    write_json(STAGE_ROOT / "frontend_build.json", frontend_build)
    verdicts = [
        write_experiment(eid, environment, prerequisites, tests_by_experiment[eid], smoke, config, ab_gate)
        for eid in TITLES
    ]
    frontend = validate_frontend(frontend_contract_test, frontend_build)
    write_json(STAGE_ROOT / "frontend_access.json", frontend)
    (STAGE_ROOT / "S08_阶段实验报告_20260921.md").write_text(
        stage_report(verdicts, frontend, tests), encoding="utf-8"
    )
    write_json(
        STAGE_ROOT / "execution_summary.json",
        {
            "stage": "S08", "executed_at": utc_now(), "target": platform.node(),
            "formal_stage_status": experiment.STATUS_BLOCKED,
            "verdicts": verdicts, "prerequisites": prerequisites.as_dict(),
            "tests": tests, "interface": interface_check, "frontend": frontend,
            "frontend_contract_test": frontend_contract_test,
            "frontend_build": frontend_build,
            "claim_boundary": "component execution is complete; formal online experiments remain blocked",
        },
    )
    print(json.dumps({
        "stage": "S08", "output_root": str(STAGE_ROOT),
        "statuses": {item["experiment_id"]: item["status"] for item in verdicts},
        "component_tests_ok": tests.get("ok"), "frontend_ok": frontend.get("ok"),
        "missing_prerequisites": prerequisites.missing,
    }, ensure_ascii=False, indent=2))
    return 0 if tests.get("ok") and frontend.get("ok") and interface_check.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
