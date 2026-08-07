#!/usr/bin/env python3
"""Collect the honest executable scope of S10 on the Jetson target.

The configured target is a single-GPU Jetson.  This collector therefore does
three things and keeps their evidence levels separate:

* records real, read-only topology/backend/upstream capability evidence;
* executes the S10 interface, contract and invariant checks on the target;
* emits one formal ``BLOCKED`` (or ``N/A_BY_ADR``) evidence package per
  experiment when the documented multi-accelerator gate is unavailable.

It never launches two ranks on one physical accelerator and never promotes the
CPU loopback smoke into collective/TP/scaling/overlap evidence.  Run only via
``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.console.evidence import EvidenceCatalog  # noqa: E402
from hqsb.distributed import experiment, interface_map, probes, specs, topology  # noqa: E402
from hqsb.core.errors import ConfigError  # noqa: E402

STAGE = "S10"
STAGE_ROOT = REPO / "docs/stage_experiments/S10"
CONFIG_ROOT = REPO / "configs/distributed"
DRIVER = REPO / "scripts/distributed/run_e10.py"
EXPERIMENTS = tuple(f"E10-{index:02d}" for index in range(1, 11))


@dataclass(frozen=True)
class Protocol:
    experiment_id: str
    level: str
    title: str
    expected_effect: str
    required_data: Tuple[str, ...]
    pass_criteria: Tuple[str, ...]
    dependencies: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "level": self.level,
            "title": self.title,
            "expected_effect": self.expected_effect,
            "required_data": list(self.required_data),
            "pass_criteria": list(self.pass_criteria),
            "dependencies": list(self.dependencies),
        }


def _p(
    experiment_id: str,
    level: str,
    title: str,
    expected_effect: str,
    required_data: Sequence[str],
    pass_criteria: Sequence[str],
    dependencies: Sequence[str] = (),
) -> Protocol:
    return Protocol(
        experiment_id=experiment_id,
        level=level,
        title=title,
        expected_effect=expected_effect,
        required_data=tuple(required_data),
        pass_criteria=tuple(pass_criteria),
        dependencies=tuple(dependencies),
    )


PROTOCOLS: Mapping[str, Protocol] = {
    "E10-01": _p(
        "E10-01", "P0", "拓扑身份、链路可达性与 Rank Placement",
        "建立拓扑身份和 placement 前提。",
        (
            "TopologyManifest schema、canonical JSON、hash 与原始来源命令",
            "accelerator、CPU/NUMA、PCIe/device fabric、NIC/RDMA 与软件栈 inventory",
            "每条 link 的 width/speed/status/capability/confidence/source",
            "P2P、device copy、host-device NUMA 与 RDMA reachability 探针",
            "实际 collective transport/data-path 证据",
            "PlacementPlan、rank table、planned/actual 绑定",
            "逐 rank identity 与 group membership",
            "独立新 job topology drift 与最终 gate verdict",
        ),
        (
            "TopologyManifest schema/canonical hash/字段级证据完整",
            "accelerator、CPU/NUMA、PCIe/device fabric、NIC/RDMA 和软件栈全部采集",
            "每个关键 edge 有 capability/状态/来源，P2P 与 data path 经探针",
            "PlacementPlan 能生成 rank/group 映射，planned/actual 一致",
            "每 rank 的 host/PID/device/CPU/group 身份完整，无重复或遗漏",
            "degraded/unknown link 有明确 deny/降级政策",
            "独立新 job 可复现 topology class，漂移可定位",
            "所有 raw 与后续可引用 hash 绑定，无敏感信息泄漏",
        ),
    ),
    "E10-02": _p(
        "E10-02", "P0", "Collective Microbenchmark 与通信模型",
        "得到 latency/bandwidth 曲线和算法边界。",
        (
            "collective spec、backend/harness identity 与 capability matrix",
            "op/dtype/message-size/rank/topology/algorithm/protocol case grid",
            "五类必需 collective 的逐 rank input/output hash 与 correctness",
            "completion-timed raw rank/job latency samples 与 quantiles",
            "payload/bus bytes、AlgBW/BusBW、CPU 与错误",
            "alpha-beta segments、algorithm crossover 与 placement effect",
            "profiler raw/metric manifest 与官方工具 crosscheck",
            "模型消息成本投影与独立 confirmation",
        ),
        (
            "五类必需 collective 都有数值正确性和逐 rank hash",
            "size 覆盖 latency/转换/带宽平台及模型真实消息邻域",
            "dtype/rank/topology/algorithm 的可用格有 raw，缺失格不补零",
            "completion 计时、payload、algbw/busbw 公式与单位可复算",
            "alpha-beta 分区、algorithm crossover 和 topology 差异有证据",
            "official tool crosscheck 差异可解释",
            "profiler 与 clean benchmark 分离，异常点保留",
            "独立 job confirmation 支持曲线与 E10-04 成本预测",
        ),
        ("E10-01",),
    ),
    "E10-03": _p(
        "E10-03", "P0", "Collective 顺序、一致性与超时",
        "验证分布式错误不会无限挂起或静默错算。",
        (
            "collective call schema、communicator state machine 与 timeout config",
            "合法 golden sequence 的逐 rank call/input/output hash",
            "op/count/dtype/root/order/group/skipped/delayed fault matrix",
            "逐 rank call records、field diffs 与 normalized errors",
            "watchdog heartbeat/event 与 MTTD/abort 时间",
            "guard buffer 与 partial-output invalidation",
            "cleanup 前后 process/device/fd/socket/rendezvous 资源",
            "新 epoch recovery probe、独立重复与 drill/runbook",
        ),
        (
            "合法 Golden Sequence 全 rank 正确且无误报",
            "op/count/dtype/root/order/group/skipped/delayed faults 覆盖",
            "每 fault 有首个分叉 group/seq/rank/字段证据",
            "所有超时有界，无无限 hang、越界或 partial-success",
            "fatal error 后 communicator 不再复用，全 rank abort 状态完整",
            "buffer、device memory、进程、socket/fd、rendezvous 清理可验证",
            "新 epoch communicator 可重建并通过健康序列",
            "MTTD/abort/cleanup/recovery 有独立重复与 runbook",
        ),
        ("E10-01", "E10-02"),
    ),
    "E10-04": _p(
        "E10-04", "P0", "Tensor Parallel 正确性与通信账本",
        "建立模型级并行正确性和通信账本。",
        (
            "Qwen architecture census 与版本化 ParallelPlan",
            "TP capability、attention/MLP/KV/head/embedding shard mapping",
            "per-rank shard hash、round-trip 与 direct-load 审计",
            "per-rank load/runtime/peak memory 与复制项",
            "column/row/attention/MLP/layer correctness",
            "prefill/decode hidden/logits/token/KV 与单卡 reference 对齐",
            "expected/observed collective event/call/bytes ledger 与 diff",
            "逐 rank trace、unsupported/GQA 边界与独立冷启动 confirmation",
        ),
        (
            "ParallelPlan 覆盖每个权重、activation、head、KV 和 collective",
            "shard round-trip 无遗漏/重叠，direct load 无隐式全模型复制",
            "column/row/attention/MLP/layer/model 分层正确性通过共同门",
            "representative prefill 与多步 decode 的 logits/token/KV 语义对齐",
            "expected/observed collective op/count/dtype/bytes/calls 闭合或差异有证据",
            "per-rank memory 理论/实测与复制项透明",
            "unsupported/uneven/GQA 边界安全",
            "独立冷启动 job 能由 plan+shard artifact 复现",
        ),
        ("E10-02", "E10-03", "S07 single-card reference"),
    ),
    "E10-05": _p(
        "E10-05", "P0", "Strong/Weak/Capacity Scaling",
        "画出扩展效率与容量收益。",
        (
            "available resource grid 与每 degree 完整 identity",
            "strong/weak/capacity workload specs 与 work-unit 定义",
            "每 degree correctness、cold/load/init 与 raw per-rank samples",
            "throughput/latency/TTFT/TPOT/memory/device-seconds",
            "speedup、strong/weak efficiency 与 capacity gain",
            "compute/comm/overlap/wait/idle critical-path decomposition",
            "per-degree ledger、predicted/observed communication 与 scaling fit",
            "随机区组、CI、异常、missing-resource 与 confirmation",
        ),
        (
            "每个可用卡数有真实 raw、正确性和完整 identity",
            "strong/weak/capacity 三种定义、work unit和图表分开",
            "无真实 T1 时不计算强扩展 speedup",
            "prefill/decode、latency/throughput/memory/device-seconds共同报告",
            "per-rank compute/comm/overlap/wait/idle 与消息账本完整",
            "效率下降能由 E10-02、topology、kernel shape或straggler解释",
            "独立 job/随机区组/CI 和异常保留符合预注册",
            "输出明确推荐域、容量收益、退化域和 missing resource",
        ),
        ("E10-04",),
    ),
    "E10-06": _p(
        "E10-06", "P0", "通信—计算重叠",
        "量化通信计算重叠的真实收益。",
        (
            "tensor dependency DAG 与 schedule identities",
            "stream/chunk/bucket candidate grid 与 readiness/completion events",
            "blocking/async-no-overlap/overlap 的 correctness 与 race probes",
            "clean raw/summary latency 与逐 rank actual schedule timeline",
            "compute/comm/overlap/idle interval-union 与 critical path",
            "exposed communication、contention、chunk/event overhead",
            "theoretical bound 与 prefill/decode phase policy",
            "ABBA 独立 confirmation 与正/负因果裁决",
        ),
        (
            "dependency DAG、schedule和合法独立compute窗口明确",
            "blocking/async-no-overlap/overlap 三组唯一变量成立",
            "tensor/logit/token/KV/collective seq 在race/adversarial下正确",
            "无死锁、隐式global sync、资源泄漏和非法buffer生命周期",
            "timeline逐 rank证明真实重叠并用区间集合计算",
            "chunk/event/contention/exposed comm与E2E因果闭合",
            "prefill/decode和正/负适用域分开",
            "ABBA独立confirmation支持正结果或严谨 PASS_NEGATIVE",
        ),
        ("E10-04", "E10-05"),
    ),
    "E10-07": _p(
        "E10-07", "P1", "PP/CP/SP 选择与最小实验",
        "理解 TP 之外的适用边界。",
        (
            "activation ADR、selection rubric 与 candidate scores",
            "PP/CP/SP analytical costs 与唯一具体 ParallelPlan",
            "partition ranges/hash 与 expected/observed communication ledger",
            "tensor/logits/token/KV correctness",
            "microbatch/sequence/degree raw benchmark 与 summary",
            "bubble/stage balance/context communication 与 per-rank timeline",
            "memory/latency/throughput/device-seconds 与 TP comparison",
            "adversarial/unsupported、confirmation 与 adopt/reject verdict",
        ),
        (
            "选择来自capacity/profile/topology证据和预注册rubric",
            "唯一具体PP/CP/SP算法、ParallelPlan和通信账本完整",
            "partition重组、tensor/logits/token/KV通过共同门",
            "microbatch/sequence/degree扫描覆盖主要与对抗边界",
            "bubble/stage或context/head通信由逐rank timeline量化",
            "memory/latency/throughput/device-seconds和TP基线共同报告",
            "failure/unsupported安全且confirmation复现",
            "adopt/reject结论绑定模型、workload、topology和runtime",
        ),
        ("E10-05",),
    ),
    "E10-08": _p(
        "E10-08", "P0", "MoE Expert Parallel、Dispatch/Combine 与 AllToAll",
        "量化负载不均和 all-to-all 瓶颈。",
        (
            "claim level、MoE operator spec 与 route artifacts",
            "uniform/mild/severe/time-varying skew manifest",
            "dispatch/combine oracle、token/gate-weight conservation",
            "真实 L2 AllToAll(V) 或 L3 runtime correctness",
            "send-count matrix 与 expected/observed byte ledger",
            "tokens/expert/rank、imbalance、expert compute/cache",
            "all-to-all time、throughput/latency 与逐 rank profile",
            "placement A/B、holdout、prefill/decode 与 confirmation",
        ),
        (
            "完成层级和允许声明明确",
            "route artifact、dispatch/combine oracle和token守恒完整",
            "至少真实L2 AllToAll(V)闭环正确，或更高L3 runtime闭环",
            "uniform到severe skew、top-k/tokens/hidden/EP degree覆盖",
            "per-expert/per-rank imbalance、communication matrix/bytes/time和expert compute完整",
            "至少一个placement/skew缓解策略有holdout正/负结论",
            "prefill/decode与actual backend/profile/独立confirmation完整",
            "只有L4通过时才允许完整MoE模型质量/性能声明",
        ),
        ("E10-02", "E10-03"),
    ),
    "E10-09": _p(
        "E10-09", "P0", "多 Rank Trace、Straggler 与链路归因",
        "定位 straggler 和同步放大。",
        (
            "profiler metric manifest、overhead 与全 rank raw traces",
            "clock offset/drift/uncertainty calibration",
            "normalized/unmapped events 与 request-rank-collective-kernel mapping",
            "per-rank compute/comm/overlap/wait/idle/unknown breakdown",
            "collective arrival/completion skew 与 communication matrix",
            "baseline variability 与 slow-rank/host/compute/topology/skew injections",
            "root-cause confusion matrix 与 amplification chains",
            "prefill/decode analysis、runbook 与 independent confirmation",
        ),
        (
            "所有相关rank有raw trace、metric和clock manifest",
            "request到phase到layer到group/seq到kernel/transport可关联",
            "时钟offset/drift/uncertainty与profiler overhead量化",
            "compute/comm/overlap/wait/idle/unknown不重复计时",
            "arrival/completion skew与通信矩阵能区分等待者和root cause",
            "至少慢arrival、compute/host和topology/skew多类注入可盲定位",
            "scaling/overlap退化能由逐rank证据解释或明确unknown",
            "独立confirmation和可执行runbook完成",
        ),
        ("E10-05", "E10-06", "E10-08"),
    ),
    "E10-10": _p(
        "E10-10", "P0", "Rank/网络/OOM 故障与恢复",
        "建立故障有界性与 runbook。",
        (
            "safety/blast-radius scope 与 healthy baseline resource snapshot",
            "rank crash/hang/OOM/network/communicator fault matrix 与 injection events",
            "control heartbeat/global decision 与 per-rank last progress/error",
            "fault time、MTTD、propagation、abort、cleanup、restart、MTTR",
            "request token/KV commit-invalidation 与 retry/idempotency",
            "communicator state/epoch/abort/recreate/restart ledger",
            "process/device-memory/socket/fd/shared-state residuals",
            "分层 health probes、runbook、blind drill 与 independent repeats",
        ),
        (
            "rank exit/SIGKILL/hang、OOM、connect/in-flight network和communicator faults按可安全范围覆盖",
            "独立control plane/watchdog能在预注册上界内检测并全组决策",
            "每rank最终状态、root group/seq/error链完整，无无限hang",
            "partial output/token/KV被invalid，retry幂等且有界",
            "fatal communicator不复用，abort/destroy/recreate/restart级别如实证明",
            "process/device memory/socket/fd/shared state无不可解释残留",
            "新epoch从冻结artifacts启动并通过分层健康probe",
            "MTTD/MTTR有独立重复，runbook blind drill有效，mock限制明确",
        ),
        ("E10-03", "E10-09"),
    ),
}

E10_01_PREFLIGHT_STEPS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 27, 29, 30}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(json_safe(dict(row)), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def run_command(argv: Sequence[str], *, timeout: int = 120) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv), cwd=REPO, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
        return {
            "argv": list(argv), "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False,
        }
    except FileNotFoundError as exc:
        return {
            "argv": list(argv), "returncode": 127,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": exc.stdout or "", "stderr": exc.stderr or "",
            "timed_out": True,
        }


def run_json(argv: Sequence[str], *, timeout: int = 120) -> Dict[str, Any]:
    result = run_command(argv, timeout=timeout)
    if result["returncode"] == 0:
        try:
            result["payload"] = json.loads(result["stdout"])
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return default


def read_int(path: Path) -> Optional[int]:
    try:
        return int(read_text(path))
    except ValueError:
        return None


def parse_cpu_list(value: str) -> List[int]:
    cpus: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def collect_numa() -> topology.NumaTopology:
    nodes: List[topology.NumaNode] = []
    mapping: Dict[int, int] = {}
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        node_id = int(path.name[4:])
        cpus = parse_cpu_list(read_text(path / "cpulist"))
        meminfo = read_text(path / "meminfo")
        total_match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
        free_match = re.search(r"MemFree:\s+(\d+)\s+kB", meminfo)
        total = int(total_match.group(1)) * 1024 if total_match else topology.unavailable("node MemTotal unavailable")
        free = int(free_match.group(1)) * 1024 if free_match else topology.unavailable("node MemFree unavailable")
        distance_values = read_text(path / "distance").split()
        distance = {index: int(item) for index, item in enumerate(distance_values) if item.isdigit()}
        nodes.append(topology.NumaNode(node_id, 0, tuple(cpus), total, free, distance))
        mapping.update({cpu: node_id for cpu in cpus})
    if not nodes:
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
        nodes.append(
            topology.NumaNode(
                0, 0, tuple(cpus),
                topology.unavailable("NUMA sysfs not exposed"),
                topology.unavailable("NUMA sysfs not exposed"),
            )
        )
        mapping = {cpu: 0 for cpu in cpus}
    return topology.NumaTopology(nodes=tuple(nodes), core_to_numa=mapping)


def collect_nics() -> Tuple[topology.NicRecord, ...]:
    rows: List[topology.NicRecord] = []
    for path in sorted(Path("/sys/class/net").glob("*")):
        name = path.name
        state = read_text(path / "operstate", "UNKNOWN").upper()
        mtu = read_int(path / "mtu")
        speed_mbps = read_int(path / "speed")
        numa = read_int(path / "device/numa_node")
        device = path / "device"
        try:
            bdf = device.resolve().name if device.exists() else ""
        except OSError:
            bdf = ""
        rows.append(
            topology.NicRecord(
                interface=name,
                pci_bdf=bdf,
                numa_node=numa if numa is not None and numa >= 0 else topology.unavailable("NIC NUMA unavailable"),
                link_layer="loopback" if name == "lo" else "ethernet",
                speed_gbps=(speed_mbps / 1000.0) if speed_mbps is not None and speed_mbps >= 0 else topology.unavailable("link speed unavailable"),
                mtu=mtu if mtu is not None else topology.unavailable("MTU unavailable"),
                state=state,
                reachable="LOCAL_ONLY" if name == "lo" else "UNVERIFIED",
            )
        )
    return tuple(rows)


def parse_nvidia_smi_l(text: str) -> List[Dict[str, str]]:
    rows = []
    pattern = re.compile(r"GPU\s+(\d+):\s+(.+?)\s+\(UUID:\s*([^\)]+)\)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            rows.append({"index": match.group(1), "name": match.group(2), "uuid": match.group(3)})
    return rows


def redact_command_outputs(commands: Dict[str, Dict[str, Any]]) -> None:
    """Remove network identifiers that are not needed for topology claims."""
    result = commands.get("ip_link", {})
    try:
        rows = json.loads(result.get("stdout", ""))
    except (TypeError, json.JSONDecodeError):
        rows = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            for field in ("address", "broadcast", "permaddr"):
                if field in row:
                    row[field] = "REDACTED_MAC"
        result["stdout"] = json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n"
        result["redactions"] = ["address", "broadcast", "permaddr"]


def source_identity() -> Dict[str, Any]:
    commit = run_command(("git", "rev-parse", "HEAD"))
    status = run_command(("git", "status", "--porcelain"))
    digest = hashlib.sha256()
    roots = (REPO / "hqsb/distributed", CONFIG_ROOT, REPO / "scripts/distributed", Path(__file__))
    files: List[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for path in sorted(files):
        relative = str(path.relative_to(REPO)).encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return {
        "git_commit": commit["stdout"].strip() if commit["returncode"] == 0 else None,
        "git_dirty": bool(status["stdout"].strip()),
        "git_status_sha256": sha256_bytes(status["stdout"].encode("utf-8")),
        "s10_source_tree_sha256": digest.hexdigest(),
    }


def torch_capability() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - a failed capability probe is data
        return {
            "importable": False, "error": f"{type(exc).__name__}: {exc}",
            "cuda_available": False, "device_count": 0, "devices": [],
        }
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index, "name": prop.name,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(prop.total_memory),
                    "multiprocessor_count": int(prop.multi_processor_count),
                    "integrated": bool(getattr(prop, "is_integrated", True)),
                }
            )
    nccl_version: Any = topology.unavailable("PyTorch NCCL version query unavailable")
    try:
        version = torch.cuda.nccl.version()
        if version:
            nccl_version = ".".join(str(item) for item in version) if isinstance(version, tuple) else str(version)
    except Exception as exc:  # noqa: BLE001
        nccl_version = topology.unavailable(f"{type(exc).__name__}: {exc}")
    distributed = getattr(torch, "distributed", None)
    return {
        "importable": True,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": devices,
        "distributed_available": bool(distributed and distributed.is_available()),
        "nccl_available": bool(distributed and distributed.is_available() and distributed.is_nccl_available()),
        "gloo_available": bool(distributed and distributed.is_available() and distributed.is_gloo_available()),
        "nccl_version": nccl_version,
    }


def collect_environment() -> Dict[str, Any]:
    commands = {
        "uname": run_command(("uname", "-srvmo")),
        "nvidia_smi_list": run_command(("nvidia-smi", "-L")),
        "nvidia_smi_query": run_command(("nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total", "--format=csv,noheader")),
        "nvidia_smi_topology": run_command(("nvidia-smi", "topo", "-m")),
        "lscpu": run_command(("lscpu", "-J")),
        "lscpu_topology": run_command(("lscpu", "-e=CPU,NODE,SOCKET,CORE")),
        "numactl": run_command(("numactl", "--hardware")),
        "lspci": run_command(("lspci", "-Dnnvv"), timeout=30),
        "ip_link": run_command(("ip", "-j", "link", "show")),
        "rdma_link": run_command(("rdma", "link", "show")),
        "ibv_devinfo": run_command(("ibv_devinfo", "-l")),
        "nvpmodel": run_command(("nvpmodel", "-q")),
        "tegrastats": run_command(("timeout", "1", "tegrastats", "--interval", "100"), timeout=5),
    }
    redact_command_outputs(commands)
    torch_info = torch_capability()
    board_model = read_text(Path("/proc/device-tree/model"), topology.unavailable("board model unavailable"))
    return {
        "captured_at": utc_now(),
        "scope": "Jetson remote target; read-only S10 capability and single-device topology preflight",
        "host_alias": "jetson-target",
        "board_model": board_model,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "torch": torch_info,
        "commands": commands,
        "source": source_identity(),
    }


def build_manifest(environment: Mapping[str, Any]) -> Tuple[topology.TopologyManifest, Dict[str, Any]]:
    torch_info = environment["torch"]
    smi_rows = parse_nvidia_smi_l(environment["commands"]["nvidia_smi_list"]["stdout"])
    accelerators: List[topology.AcceleratorRecord] = []
    for index, item in enumerate(torch_info.get("devices", [])):
        smi = next((row for row in smi_rows if int(row["index"]) == index), {})
        accelerators.append(
            topology.AcceleratorRecord(
                device_id=index,
                sku=str(item.get("name") or smi.get("name") or topology.unavailable("SKU unavailable")),
                uuid=str(smi.get("uuid") or topology.unavailable("UUID unavailable")),
                pci_bdf=topology.unavailable("integrated Jetson GPU has no discrete PCI BDF"),
                memory_bytes=item.get("total_memory_bytes", topology.unavailable("memory unavailable")),
                driver=read_text(Path("/proc/driver/nvidia/version"), topology.unavailable("driver version unavailable")),
                health="OK" if torch_info.get("cuda_available") else "UNKNOWN",
                visible_index=index,
            )
        )
    numa = collect_numa()
    nics = collect_nics()
    nodes: List[topology.TopologyNode] = [
        topology.TopologyNode("host0", "host", {"alias": "jetson-target"})
    ]
    nodes.extend(topology.TopologyNode(f"numa{node.node_id}", "cpu_numa") for node in numa.nodes)
    nodes.extend(topology.TopologyNode(f"accel{item.device_id}", "accelerator", {"uuid": item.uuid}) for item in accelerators)
    nodes.extend(topology.TopologyNode(f"nic:{item.interface}", "nic_port", {"interface": item.interface}) for item in nics)
    edges: List[topology.TopologyEdge] = []
    for node in numa.nodes:
        edges.append(
            topology.TopologyEdge(
                src="host0", dst=f"numa{node.node_id}", edge_type="cpu_interconnect",
                status="UP", source_tool="sysfs", observed_at=environment["captured_at"],
                confidence="QUERY_VERIFIED",
            )
        )
    for nic in nics:
        source = f"numa{nic.numa_node}" if isinstance(nic.numa_node, int) and f"numa{nic.numa_node}" in {node.node_id for node in nodes} else "host0"
        edges.append(
            topology.TopologyEdge(
                src=source, dst=f"nic:{nic.interface}", edge_type="ethernet",
                negotiated_capacity=nic.speed_gbps, status="UP" if nic.state == "UP" else "UNKNOWN",
                source_tool="sysfs", observed_at=environment["captured_at"],
                confidence="QUERY_VERIFIED",
            )
        )
    uuids = tuple(item.uuid for item in accelerators)
    visible = ()
    if uuids:
        visible = (
            topology.VisibleDeviceAudit(
                launcher_env={key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES",) if key in os.environ},
                runtime_visible=uuids, current_device_uuid=uuids[0], local_index=0,
            ),
        )
    backend = topology.BackendRuntimeConfig(
        backend="nccl" if torch_info.get("nccl_available") else "gloo",
        version=str(torch_info.get("nccl_version") if torch_info.get("nccl_available") else "PyTorch " + str(torch_info.get("torch_version", "unavailable"))),
        env_vars=probes.capture_backend_env(
            os.environ,
            backend="nccl" if torch_info.get("nccl_available") else "gloo",
            version=str(torch_info.get("nccl_version") if torch_info.get("nccl_available") else torch_info.get("torch_version", "unavailable")),
        )["env_vars"],
        defaults_explained_by="locked PyTorch/NCCL environment fingerprint",
    )
    manifest = topology.TopologyManifest(
        scope=topology.ObservationScope(
            branch="cuda_nccl", node_scope="single_node", declared_at=environment["captured_at"],
            notes="single-device Jetson preflight; not a multi-rank topology gate",
            missing_resource_branches=("second_accelerator", "multi_node"),
        ),
        host=topology.HostIdentity(
            node_id="node0", host_alias="jetson-target", cpu_arch=platform.machine(),
            os_release=platform.platform(), kernel=platform.release(),
            collected_at=environment["captured_at"], redacted_fields=("hostname", "ip_address"),
        ),
        numa=numa,
        accelerators=tuple(accelerators),
        affinity=tuple(
            topology.AffinityRecord(
                accelerator_uuid=item.uuid, cpu_numa_node=0, memory_numa_node=0,
                distance=topology.unavailable("integrated GPU NUMA distance not separately exposed"),
            )
            for item in accelerators
        ),
        nics=nics,
        rdma=topology.RdmaStackRecord(
            rdma_core_version=topology.unavailable("rdma tool unavailable or no RDMA device"),
            device_direct_available=False,
        ),
        software=(backend,), visible_audits=visible, nodes=tuple(nodes), edges=tuple(edges),
        raw_evidence={name: f"raw/topology_commands.json#{name}" for name in environment["commands"]},
        collected_at=environment["captured_at"],
    )
    multi_accelerator = len(accelerators) >= experiment.MINIMUM_ACCELERATORS
    gate = topology.gate_for_collective_probe(
        manifest, placement_ok=False, data_path_verified=False
    ).as_dict()
    if not multi_accelerator:
        gate["ready"] = False
        gate.setdefault("blockers", []).insert(
            0, f"requires >=2 real accelerators; observed {len(accelerators)}"
        )
        gate["reason_code"] = "NOT_RUN_RESOURCE_UNAVAILABLE"
    validation = {
        "schema_errors": manifest.validate(),
        "schema_valid": not manifest.validate(),
        "accelerator_count": len(accelerators),
        "minimum_accelerators": experiment.MINIMUM_ACCELERATORS,
        "multi_accelerator_available": multi_accelerator,
        "manifest_sha256": manifest.sha256,
        "collective_gate": gate,
        "claim_boundary": "a schema-valid single-device manifest is not a passing rank-placement/data-path gate",
    }
    return manifest, validation


def collect_component_tests(junit_path: Path) -> Dict[str, Any]:
    result = run_command(
        (
            sys.executable, "-m", "pytest", "-q", "-rA", "--junitxml", str(junit_path),
            "tests/unit/distributed", "tests/property/test_distributed_invariants.py",
        ),
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
            cases.append(
                {
                    "node": f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}",
                    "status": status, "time_seconds": float(case.attrib.get("time", 0.0)),
                    "message": message,
                }
            )
    result["cases"] = cases
    result["counts"] = {
        name: sum(row["status"] == name for row in cases)
        for name in ("PASS", "FAIL", "ERROR", "SKIP")
    }
    result["ok"] = result["returncode"] == 0 and bool(cases) and all(
        row["status"] in ("PASS", "SKIP") for row in cases
    )
    result["evidence_level"] = "COMPONENT_ONLY"
    result["claim_allowed"] = False
    return result


def collect_configs() -> Dict[str, Any]:
    loaded = specs.DistributedSpecs.load(str(CONFIG_ROOT))
    return {
        "strict_load_ok": loaded.ok,
        "document_count": len(loaded.documents),
        "audits": list(loaded.audits),
        "documents": [
            {
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(CONFIG_ROOT.glob("*.yaml"))
        ],
        "claim_boundary": "frozen templates and contract audits are not multi-rank samples",
    }


def collect_upstream() -> Dict[str, Any]:
    stages: Dict[str, Any] = {}
    for stage, count in (("S07", 10), ("S08", 11)):
        rows = []
        for index in range(1, count + 1):
            experiment_id = f"E{stage[1:]}-{index:02d}"
            path = REPO / f"docs/stage_experiments/{stage}/{experiment_id}/raw/verdict.json"
            payload: Dict[str, Any] = {}
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = {"status": "UNREADABLE"}
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(path.relative_to(REPO)),
                    "exists": path.is_file(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                    "status": payload.get("status", payload.get("overall", "MISSING")),
                }
            )
        stages[stage] = {
            "verdicts": rows,
            "passing": [row["experiment_id"] for row in rows if row["status"] in ("PASS", "PASS_NEGATIVE")],
        }
    s08_jsonl = sorted((REPO / "docs/stage_experiments/S08").glob("**/*.jsonl"))
    s02_reference = sorted((REPO / "docs/stage_experiments/S02").glob("**/*.json"))
    return {
        "captured_at": utc_now(),
        "stages": stages,
        "s08_jsonl_count": len(s08_jsonl),
        "s08_trace_candidates": [str(path.relative_to(REPO)) for path in s08_jsonl[:20]],
        "s02_reference_json_count": len(s02_reference),
        "s02_reference_examples": [str(path.relative_to(REPO)) for path in s02_reference[:10]],
        "claim_boundary": "upstream files are inherited gates, not S10 measurements",
    }


def prerequisite_summary(
    environment: Mapping[str, Any],
    topology_validation: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> Dict[str, Any]:
    s07_pass = bool(upstream["stages"]["S07"]["passing"])
    s08_trace = upstream["s08_jsonl_count"] > 0
    checks = [
        {
            "name": "s07_p0_verdicts", "satisfied": s07_pass,
            "reason": "S07 has no PASS/PASS_NEGATIVE P0 runtime verdict" if not s07_pass else "",
        },
        {
            "name": "s08_tokenized_trace", "satisfied": s08_trace,
            "reason": "no S08 JSONL request trace is present" if not s08_trace else "",
        },
        {
            "name": "two_accelerators_available",
            "satisfied": topology_validation["multi_accelerator_available"],
            "reason": (
                f"requires >=2 real accelerators; observed {topology_validation['accelerator_count']}"
                if not topology_validation["multi_accelerator_available"] else ""
            ),
        },
        {
            "name": "rank_placement_and_data_path",
            "satisfied": topology_validation["collective_gate"]["ready"],
            "reason": topology_validation["collective_gate"].get("reason", ""),
        },
        {
            "name": "exact_backend_identity",
            "satisfied": bool(environment["torch"].get("nccl_available") and not topology.is_unavailable(environment["torch"].get("nccl_version"))),
            "reason": "PyTorch NCCL backend/version unavailable for a real multi-rank run",
        },
        {
            "name": "single_card_reference",
            "satisfied": upstream["s02_reference_json_count"] > 0,
            "reason": "no S02 JSON reference exists" if upstream["s02_reference_json_count"] == 0 else "",
        },
        {
            "name": "frozen_model_workload_pair",
            "satisfied": False,
            "reason": "no S10 model_workload.json binds a ModelArtifact hash to a tokenized workload hash",
        },
        {
            "name": "c6_c7_schema_available",
            "satisfied": True,
            "reason": "",
        },
    ]
    return {
        "stage": STAGE,
        "satisfied": all(row["satisfied"] for row in checks),
        "missing": [row["name"] for row in checks if not row["satisfied"]],
        "checks": checks,
        "formal_execution_allowed": False,
        "reason_code": "NOT_RUN_RESOURCE_UNAVAILABLE",
    }


def archive_existing_raw(raw: Path, new_run_id: str) -> None:
    verdict_path = raw / "verdict.json"
    if not verdict_path.is_file():
        return
    try:
        previous = json.loads(verdict_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
    previous_run_id = str(previous.get("run_id", "unknown_previous_run"))
    if previous_run_id == new_run_id:
        return
    if not previous_run_id or previous_run_id in (".", "..") or any(char in previous_run_id for char in ("/", "\\", "\0")):
        previous_run_id = "unknown_previous_run"
    archive = raw / "runs" / previous_run_id
    if archive.exists():
        raise ConfigError(f"refusing to overwrite archived S10 evidence {archive}")
    archive.mkdir(parents=True)
    for source in sorted(raw.iterdir()):
        if source.name == "runs":
            continue
        target = archive / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.is_file():
            shutil.copy2(source, target)


def evidence_level(experiment_id: str, item_index: int) -> str:
    preflight = {
        "E10-01": {1, 2, 3, 8},
        "E10-02": {1, 5},
        "E10-03": {1},
        "E10-04": {1, 2},
        "E10-05": {1},
        "E10-06": {1},
        "E10-07": {1, 2},
        "E10-08": {1, 2, 3},
        "E10-09": {1},
        "E10-10": {1},
    }
    if item_index in preflight.get(experiment_id, set()):
        return "COLLECTED_PREFLIGHT_OR_COMPONENT_ONLY"
    return "NOT_COLLECTED_RESOURCE_UNAVAILABLE"


def required_rows(protocol: Protocol) -> List[Dict[str, Any]]:
    rows = []
    for index, item in enumerate(protocol.required_data, 1):
        availability = evidence_level(protocol.experiment_id, index)
        rows.append(
            {
                "item_id": f"D{index:02d}",
                "required": item,
                "availability": availability,
                "scientific_sample": False,
                "reason": (
                    "real read-only target preflight or target-executed component contract evidence exists, but it is not a multi-rank sample"
                    if availability.startswith("COLLECTED")
                    else "requires >=2 physical accelerators and a passing upstream/runtime gate"
                ),
            }
        )
    return rows


def criteria_rows(protocol: Protocol, *, n_a: bool) -> List[Dict[str, Any]]:
    return [
        {
            "criterion_id": f"C{index:02d}",
            "criterion": criterion,
            "satisfied": False,
            "status": "N/A_BY_ADR" if n_a else "NOT_EVALUABLE",
            "reason": (
                "PP/CP/SP claim is deactivated by configs/distributed/boundary_spec.yaml"
                if n_a else
                "not evaluable without two physical accelerators, verified rank placement and multi-rank raw evidence"
            ),
        }
        for index, criterion in enumerate(protocol.pass_criteria, 1)
    ]


def step_rows(protocol: Protocol, interface_ok: bool, *, n_a: bool) -> List[Dict[str, Any]]:
    mapping = interface_map.mapping_for(protocol.experiment_id)
    rows = []
    for step in mapping.steps:
        if n_a:
            execution_status = "N/A_BY_ADR"
            reason = "PP/CP/SP claim is not activated"
        elif protocol.experiment_id == "E10-01" and step.step in E10_01_PREFLIGHT_STEPS:
            execution_status = "COLLECTED_PREFLIGHT_ONLY"
            reason = "read-only single-device topology preflight completed; multi-rank condition remains unavailable"
        else:
            execution_status = "NOT_RUN_RESOURCE_UNAVAILABLE"
            reason = "requires a verified multi-accelerator communicator or its downstream evidence"
        rows.append(
            {
                "step": step.step, "title": step.title,
                "interfaces": list(step.interfaces), "declared_maturity": step.maturity,
                "interface_resolved": interface_ok,
                "component_status": "TEST_VERIFIED" if interface_ok else "FAIL",
                "execution_status": execution_status,
                "scientific_execution": False, "claim_allowed": False,
                "reason": reason,
            }
        )
    return rows


def verdict_for(protocol: Protocol, prerequisites: Mapping[str, Any]) -> Dict[str, Any]:
    if protocol.experiment_id == "E10-07":
        return {
            "status": experiment.STATUS_N_A_BY_ADR,
            "reason_code": "PP_CP_SP_CLAIM_DISABLED",
            "reason": "configs/distributed/boundary_spec.yaml freezes claiming_parallelism=false; no PP/CP/SP claim is made.",
        }
    return {
        "status": experiment.STATUS_BLOCKED,
        "reason_code": "NOT_RUN_RESOURCE_UNAVAILABLE",
        "reason": "formal S10 execution is blocked: " + ", ".join(prerequisites["missing"]),
    }


def report_markdown(
    protocol: Protocol,
    verdict: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    criteria: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    topology_validation: Mapping[str, Any],
    tests: Mapping[str, Any],
    run_id: str,
) -> str:
    n_a = verdict["status"] == experiment.STATUS_N_A_BY_ADR
    lines = [
        f"# {protocol.experiment_id} 实验报告：{protocol.title}", "",
        f"> Run ID：`{run_id}`  ",
        f"> 科学裁决：**{verdict['status']}**  ",
        f"> 原因码：`{verdict['reason_code']}`  ",
        "> 采集器与组件测试成功不等于分布式实验通过。", "",
        "## 1. 预计效果与执行边界", "",
        f"- 预计达到的效果：{protocol.expected_effect}",
        f"- 依赖：{', '.join(protocol.dependencies) if protocol.dependencies else 'S10 物理拓扑门禁'}",
        f"- 实测设备：`{environment['torch'].get('device_count', 0)}×{environment['torch'].get('devices', [{}])[0].get('name', 'UNAVAILABLE') if environment['torch'].get('devices') else 'UNAVAILABLE'}`；正式最低要求：`{experiment.MINIMUM_ACCELERATORS}`。",
        f"- 多 accelerator 可用：`{str(topology_validation['multi_accelerator_available']).lower()}`；collective gate ready：`{str(topology_validation['collective_gate']['ready']).lower()}`。",
        f"- distributed 组件测试：`{tests.get('counts', {})}`；其证据级别为 `COMPONENT_ONLY`。", "",
        "本轮未把一个物理 GPU 伪装成两个 rank/device，未产生 collective/TP/scaling/overlap/MoE-L2/multi-rank fault 性能样本。", "",
        "## 2. 必采集信息/数据", "",
        "| ID | 必采集项 | 当前可用性 | 科学样本 | 说明 |", "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['item_id']} | {row['required']} | {row['availability']} | "
            f"{'是' if row['scientific_sample'] else '否'} | {row['reason']} |"
        )
    lines += ["", "## 3. 预计效果和单项通过标准对照", ""]
    if n_a:
        lines.append("当前未声称 PP/CP/SP，按冻结 ADR 本项不激活；预计效果不适用，不能填写 PASS。")
    else:
        lines.append("预计效果**未达到正式实验层级**；资源与上游门禁在 communicator 创建前即阻止了执行。")
    lines += ["", "| ID | 单项通过标准 | 是否满足 | 状态/原因 |", "|---|---|---:|---|"]
    for row in criteria:
        lines.append(
            f"| {row['criterion_id']} | {row['criterion']} | 否 | {row['status']}：{row['reason']} |"
        )
    lines += [
        "", "## 4. 裁决", "", f"**{verdict['status']}**：{verdict['reason']}", "",
        "接口、配置和 CPU oracle 的通过只说明采集面与拒绝逻辑可用；`claim_allowed=false`，不能替代真实多卡实测。", "",
        "## 5. 证据与前端访问", "",
        "前端通过 `GET /api/console/v1/evidence` 自动发现本实验的 `raw/verdict.json`；",
        "通过 evidence detail/download 接口读取本报告及 `raw/*.json|jsonl|txt`。",
        "`raw/evidence_manifest.json` 给出文件大小与 SHA-256。", "",
    ]
    return "\n".join(lines)


def evidence_manifest(experiment_root: Path, run_id: str) -> Dict[str, Any]:
    rows = []
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path.name == "evidence_manifest.json" or "runs" in path.relative_to(experiment_root).parts:
            continue
        rows.append(
            {
                "path": str(path.relative_to(experiment_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "stage": STAGE, "run_id": run_id,
        "experiment_id": experiment_root.name,
        "files": rows,
    }


def collect_campaign(run_id: str) -> Dict[str, Any]:
    if not run_id or any(char in run_id for char in ("/", "\\", "\0")):
        raise ConfigError("run_id must be one non-empty path component")
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    environment = collect_environment()
    manifest, topology_validation = build_manifest(environment)
    upstream = collect_upstream()
    prerequisites = prerequisite_summary(environment, topology_validation, upstream)
    config_audit = collect_configs()
    interface_audit = interface_map.resolve_interfaces()
    junit_path = STAGE_ROOT / ".component_tests.junit.xml"
    tests = collect_component_tests(junit_path)
    if junit_path.exists():
        junit_path.unlink()
    driver = {
        "list": run_json((sys.executable, str(DRIVER), "--list", "--json")),
        "interface_map": run_json((sys.executable, str(DRIVER), "--interface-map", "--json")),
        "smoke": run_json((sys.executable, str(DRIVER), "--smoke", "--json")),
        "prerequisites": run_json((sys.executable, str(DRIVER), "--prerequisites", "--json")),
    }
    smoke_payload = driver["smoke"].get("payload", {})
    component_ok = bool(
        tests.get("ok") and config_audit["strict_load_ok"] and interface_audit["ok"]
        and smoke_payload.get("status") == "smoke" and smoke_payload.get("claim_allowed") is False
    )
    statuses: Dict[str, str] = {}
    roots: List[str] = []

    for experiment_id in EXPERIMENTS:
        protocol = PROTOCOLS[experiment_id]
        experiment_root = STAGE_ROOT / experiment_id
        raw = experiment_root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        archive_existing_raw(raw, run_id)
        verdict_base = verdict_for(protocol, prerequisites)
        n_a = verdict_base["status"] == experiment.STATUS_N_A_BY_ADR
        required = required_rows(protocol)
        criteria = criteria_rows(protocol, n_a=n_a)
        steps = step_rows(protocol, interface_audit["ok"], n_a=n_a)
        verdict = {
            "stage": STAGE, "experiment_id": experiment_id, "run_id": run_id,
            "level": protocol.level,
            "overall": verdict_base["status"], "status": verdict_base["status"],
            "scientific_execution_verdict": verdict_base["status"],
            "collector_status": "PASS" if component_ok else "FAIL",
            "component_status": "PASS" if component_ok else "FAIL",
            "reason_code": verdict_base["reason_code"], "reason": verdict_base["reason"],
            "execution_attempted": False, "communicator_created": False,
            "multi_rank_kernel_launched": False, "raw_samples": 0,
            "preflight_observations": sum(row["execution_status"] == "COLLECTED_PREFLIGHT_ONLY" for row in steps),
            "simulated_samples": 0, "claim_allowed": False,
            "dependencies": list(protocol.dependencies), "dependencies_satisfied": False,
            "accelerator_count": topology_validation["accelerator_count"],
            "minimum_accelerators": topology_validation["minimum_accelerators"],
            "topology_manifest_sha256": manifest.sha256,
            "source_tree_sha256": environment["source"]["s10_source_tree_sha256"],
            "git_commit": environment["source"]["git_commit"],
            "git_dirty": environment["source"]["git_dirty"],
            "started_at": environment["captured_at"], "ended_at": utc_now(),
            "limitations": [
                "only one physical accelerator is available on the target",
                "S07 has no passing main-runtime P0 verdict",
                "S08 has no frozen tokenized request trace",
                "no S10 ModelArtifact/workload hash pair is frozen",
                "CPU loopback and component tests are not distributed experiment samples",
            ],
        }
        write_json(raw / "preregistration.json", {**protocol.as_dict(), "run_id": run_id, "frozen_before_distributed_execution": True})
        write_json(raw / "environment_fingerprint.json", environment)
        write_json(raw / "upstream_evidence.json", upstream)
        write_json(raw / "prerequisites.json", prerequisites)
        write_json(raw / "config_audit.json", config_audit)
        write_json(raw / "interface_audit.json", interface_audit)
        write_json(raw / "component_tests.json", tests)
        write_json(raw / "driver_checks.json", driver)
        write_json(raw / "required_evidence.json", {"items": required})
        write_json(raw / "criteria_evaluation.json", {"criteria": criteria})
        write_jsonl(raw / "step_status.jsonl", steps)
        write_json(raw / "verdict.json", verdict)

        if experiment_id == "E10-01":
            write_json(raw / "topology_manifest.json", manifest.as_dict())
            # Keep the canonical bytes identical to the bytes hashed by
            # TopologyManifest.sha256; a trailing newline would make a direct
            # ``sha256sum`` of the artifact disagree with manifest.sha256.
            atomic_text(raw / "topology_manifest.canonical.json", manifest.canonical_json())
            atomic_text(raw / "topology_manifest.sha256", manifest.sha256 + "\n")
            write_json(raw / "topology_validation.json", topology_validation)
            write_json(raw / "accelerators.json", {"accelerators": [item.as_dict() for item in manifest.accelerators]})
            write_json(raw / "topology_commands.json", environment["commands"])
            write_json(raw / "rank_placement_gate.json", topology_validation["collective_gate"])
        elif experiment_id == "E10-02":
            write_json(
                raw / "backend_identity.json",
                {
                    "torch": environment["torch"],
                    "formal_eligible": False,
                    "reason": "no second accelerator and no verified rank placement/data path",
                },
            )
        elif experiment_id == "E10-07":
            write_json(
                raw / "scope_activation.json",
                {
                    "claiming_parallelism": False, "status": experiment.STATUS_N_A_BY_ADR,
                    "source": "configs/distributed/boundary_spec.yaml",
                    "claim": "No PP/CP/SP correctness, performance or capacity claim is made.",
                },
            )
        elif experiment_id == "E10-08":
            write_json(
                raw / "claim_level.json",
                {
                    "verified_component_level": "L1",
                    "required_p0_level": "L2",
                    "claim_allowed": False,
                    "reason": "CPU dispatch/combine oracle passed; real AllToAll(V) is unavailable",
                },
            )
        elif experiment_id == "E10-10":
            write_json(
                raw / "safety_scope.json",
                {
                    "read_only_campaign": True, "signals_sent": [], "network_modified": False,
                    "oom_injected": False, "communicator_created": False,
                    "reason": "fault injection gate E10-03/E10-09 and multi-rank isolation are unavailable",
                },
            )

        report = experiment_root / f"{experiment_id}_实验报告.md"
        atomic_text(
            report,
            report_markdown(protocol, verdict, required, criteria, environment, topology_validation, tests, run_id),
        )
        statuses[experiment_id] = verdict["status"]
        roots.append(str(experiment_root.relative_to(REPO)))

    catalog = EvidenceCatalog(REPO)
    discovered = [item for item in catalog.scan(refresh=True) if item["stage"] == STAGE]
    frontend_validation = {
        "checked_at": utc_now(),
        "catalog_endpoint": "GET /api/console/v1/evidence",
        "detail_endpoint": "GET /api/console/v1/evidence/{evidence_id}",
        "download_endpoint": "GET /api/console/v1/evidence/{evidence_id}/download",
        "expected_experiments": len(EXPERIMENTS),
        "discovered_experiments": len(discovered),
        "experiments": sorted(item["experiment"] for item in discovered),
        "statuses": {item["experiment"]: item["status"] for item in discovered},
        "all_detail_readable": all(bool(catalog.detail(item["id"])) for item in discovered),
    }
    frontend_validation["ok"] = bool(
        frontend_validation["discovered_experiments"] == len(EXPERIMENTS)
        and frontend_validation["all_detail_readable"]
        and set(frontend_validation["experiments"]) == set(EXPERIMENTS)
    )
    for experiment_id in EXPERIMENTS:
        experiment_root = STAGE_ROOT / experiment_id
        raw = experiment_root / "raw"
        write_json(raw / "frontend_validation.json", frontend_validation)
        write_json(raw / "evidence_manifest.json", evidence_manifest(experiment_root, run_id))

    summary = {
        "stage": STAGE, "run_id": run_id,
        "collector_status": "PASS" if component_ok else "FAIL",
        "stage_status": experiment.STATUS_BLOCKED, "stage_complete": False,
        "reason_code": "NOT_RUN_RESOURCE_UNAVAILABLE",
        "accelerator_count": topology_validation["accelerator_count"],
        "minimum_accelerators": topology_validation["minimum_accelerators"],
        "multi_accelerator_available": topology_validation["multi_accelerator_available"],
        "topology_manifest_sha256": manifest.sha256,
        "prerequisites": prerequisites,
        "component_tests": {"ok": tests["ok"], "counts": tests["counts"]},
        "config_audit_ok": config_audit["strict_load_ok"],
        "interface_audit": interface_audit,
        "driver_smoke_claim_allowed": smoke_payload.get("claim_allowed"),
        "experiment_statuses": statuses, "experiment_roots": roots,
        "frontend_validation": frontend_validation,
        "generated_at": utc_now(),
    }
    write_json(STAGE_ROOT / "campaign_summary.json", summary)
    write_json(STAGE_ROOT / "frontend_validation.json", frontend_validation)
    stage_lines = [
        "# S10 阶段实验执行摘要", "",
        f"> Run ID：`{run_id}`  ",
        "> 阶段裁决：**BLOCKED**  ",
        f"> 采集器：**{summary['collector_status']}**（组件通过不等于科学实验通过）", "",
        "## 1. 物理资源与门禁", "",
        f"- 目标板：`{environment['board_model']}` / `{environment['machine']}`",
        f"- Accelerator：`{topology_validation['accelerator_count']}`，正式最低要求：`{topology_validation['minimum_accelerators']}`",
        f"- 多 accelerator ready：`{str(topology_validation['multi_accelerator_available']).lower()}`",
        f"- Topology manifest：`{manifest.sha256}`（单设备 preflight；不解锁 collective）",
        f"- 缺失门禁：`{', '.join(prerequisites['missing'])}`",
        f"- 组件测试：`{tests['counts']}`；接口映射：`{interface_audit['steps']} steps / {interface_audit['interfaces']} interfaces / ok={str(interface_audit['ok']).lower()}`。", "",
        "## 2. 实验裁决", "",
        "| 实验 | 级别 | 状态 | 预计效果达到 | 单项标准满足 |", "|---|---|---|---:|---:|",
    ]
    for experiment_id in EXPERIMENTS:
        protocol = PROTOCOLS[experiment_id]
        status = statuses[experiment_id]
        stage_lines.append(
            f"| {experiment_id} | {protocol.level} | {status} | {'不适用' if status == experiment.STATUS_N_A_BY_ADR else '否'} | 否 |"
        )
    stage_lines += [
        "", "## 3. 阶段完成标志对照", "",
        "- E10-01～06、08～10：均未通过，原因是单卡资源及上游 Runtime/trace/model-workload 门禁缺失。",
        "- TP + collective 模型闭环：未执行。",
        "- message-size/scaling/compute-comm-idle 曲线：未执行，不补零、不外推。",
        "- overlap 因果结论：未执行；CPU interval algebra 只作组件自检。",
        "- failure/OOM/network bounded recovery：未执行真实注入；未触碰共享网络或非实验进程。",
        "- MoE：仅 L1 CPU dispatch/combine oracle；未达到 P0 所需的真实 L2 AllToAll(V)。", "",
        "## 4. 前端访问", "",
        f"Console evidence 实际发现 `{frontend_validation['discovered_experiments']}/10` 项，detail 可读：`{str(frontend_validation['all_detail_readable']).lower()}`。",
        "每项 `raw/verdict.json` 由 `/api/console/v1/evidence` 索引，报告及 raw 文本附件可由 detail/download 接口访问。", "",
    ]
    atomic_text(STAGE_ROOT / "S10_阶段执行摘要.md", "\n".join(stage_lines))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="HQSB S10 single-target evidence campaign")
    value.add_argument("--run-id", default="", help="one path component; timestamp when omitted")
    value.add_argument("--json", action="store_true", help="emit the campaign summary as JSON")
    return value


def main() -> int:
    args = parser().parse_args()
    run_id = args.run_id or time.strftime("s10_%Y%m%dT%H%M%SZ", time.gmtime())
    summary = collect_campaign(run_id)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"collector_status={summary['collector_status']}")
        print(f"stage_status={summary['stage_status']}")
        print(f"accelerator_count={summary['accelerator_count']}")
        print(f"frontend_ok={summary['frontend_validation']['ok']}")
    return 0 if summary["collector_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
