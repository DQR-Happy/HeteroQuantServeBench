# S10 并行与通信设计（Design 制品）

> 阶段：S10（分布式推理与通信）
> 性质：**设计制品（Design）/ 接口层交付**；不含任何实验结果或结论数字
> 生成时间：2026-09-19
> 依据：`docs/stage_experiments/S10_实验清单.md`、
> `docs/stage_experiments/details/S10/*`（11 份）、`docs/stages/S10_分布式推理与通信.md`、
> `docs/architecture/HQSB_项目完整剖析.md`（§19、§26）、`docs/architecture/顶层架构.md`、
> `docs/architecture/module_ownership.md`（1.4.0）
> 对上一阶段格式基准：`docs/reports/S06_graph_integration_design.md`

---

## 1. 本阶段要解决的工程问题

S10 的命题由实验清单给出：**模型切到多卡后，理论计算扩展为什么会被通信、拓扑、负载不均和
同步吃掉，以及怎样用 overlap 和 placement 改善；单卡与多卡必须保持相同模型语义。**

把它拆成可设计的工程问题：

| 问题 | 设计回答的落点 |
|---|---|
| rank 实际落在哪个设备/NUMA/NIC，计划与物理拓扑是否一致？ | TopologyManifest + PlacementPlan + preflight 不变量 |
| collective 的延迟—带宽边界是什么？ | CollectiveSpec + CPU oracle + 带宽公式登记表 + α–β 分段拟合 |
| rank 调用不一致时能否有界失败而非挂起？ | Communicator 状态机 + metadata preflight + watchdog + fault oracle |
| Qwen 权重/激活/KV 怎么切，通信量是否与推导一致？ | QwenArchitectureCensus + ParallelPlan + 通信账本 diff |
| 增卡带来多少速度/容量，效率为何下降？ | strong/weak/capacity 三种 work unit + pairability + 时间分解 |
| timeline 里是否真的重叠？ | 依赖 DAG + schedule identity + 区间集合代数 + 因果矩阵 |
| PP/CP/SP 与模型/拓扑是否匹配？ | P1 激活判定 + rubric + 采用/拒绝标准 |
| MoE 的 expert skew 如何形成瓶颈？ | RouteArtifact + dispatch/combine oracle + skew profile + holdout |
| 等待究竟由哪个 rank/kernel/链路引起？ | 统一 DistributedTraceEvent + 时钟校准 + 根因证据矩阵 |
| crash/hang/OOM/断链后是否全 rank 感知并释放？ | 错误分类 + recovery level + fault oracle + 资源快照 |

---

## 2. 分层与依赖方向

`hqsb.distributed` 位于依赖图最顶端（新规则 R7/R8）：

```text
core ← models/benchmark ← backends/integration/quant ← runtime ← serving ← distributed
```

包内分四层，依赖只向下：

```text
L0 基础设施      topology · ranks · placement · probes
L1 通信语义      backend · collectives · faults · sequence
L2 模型与性能     parallel_plan · ledger · scaling · overlap · boundary · moe · traces
L3 证据与脚手架   telemetry · specs · experiment · interface_map
```

设计取舍与理由：

1. **`faults` 在 `sequence` 之下**：E10-03 的 collective 故障与 E10-10 的进程/网络/OOM 故障
   共用同一套 `FaultOracle` / `TimeMetrics` / `ResourceSnapshot`，先落地通用机制可避免两份
   实现漂移；`sequence` 只负责 collective 语义（状态机/序列/预检）。
2. **`ledger` 在 `collectives`/`parallel_plan` 之上**：账本的 expected 侧由 plan+shape 推导，
   observed 侧来自 C7/profiler，diff 需要两者的结构定义。
3. **`telemetry` 不修改冻结契约**：C6/C7 是 S01 的稳定契约，S10 只投影到
   `summary["s10"]` 命名空间与事件 attributes（与 S06/S07/S08 同法）。
4. **`interface_map` 独立于驱动**：300 步对照表是"无缺失能力"的机器可校验证明，
   驱动脚本只调用它，不反向被它依赖。

包初始化采用 PEP 562 惰性映射而**不在模块级（含 `TYPE_CHECKING`）导入子模块**：
包级导入边会形成 `hqsb.distributed → experiment → specs → hqsb.distributed` 的环，
被依赖门禁的环检测拒绝（实测记录见开发报告 §7）。

---

## 3. 关键接口（Data Contracts）

### 3.1 拓扑与 placement（E10-01）

```text
TopologyManifest
  scope(branch, node_scope) · host · numa(nodes, core_to_numa) ·
  accelerators[] · pcie[] · fabric[] · affinity[] · nics[] · rdma · software[] ·
  nodes[] · edges[] · raw_evidence{} · schema_version
  validate(): 悬空 node / 重复不一致 edge / UNAVAILABLE 无理由 / 身份重复 → 拒绝
  canonical_json() + sha256 → 后续 run 引用 hash，diff 用字段级

TopologyEdge
  src, dst, edge_type, direction, nominal/negotiated capacity, hop_class,
  numa_distance, p2p_read/write/atomic, rdma_capable, status, error_counter,
  source_tool, observed_at, confidence(DECLARED/QUERY_VERIFIED/DATA_PATH_VERIFIED/MEASURED/DEGRADED),
  evidence_uri, measured_latency_us, measured_bandwidth_gbps

PlacementPlan
  plan_id · world_size · node_count · entries[] {global/local/node rank, coordinate(tp,pp,ep,cp),
  planned/actual device uuid, cpu_affinity, numa policy, preferred nic, communicator ids,
  launcher env digest} · high_bandwidth_domains · sha256
```

不变式（`check_hard_invariants`，机器校验，非人工 review）：

```text
unique(global_rank) · unique(process_id) · unique(device_uuid) ·
planned_world_size == observed_rank_count · all(group members exist) ·
all(planned fast-path edges are not DOWN/UNKNOWN) ·
actual_device_uuid(rank) == planned_device_uuid(rank)
```

### 3.2 collective 语义与成本（E10-02）

- `COUNT_SEMANTICS` 冻结每个 API 的 payload 定义（ReduceScatter 的 total input count 与
  AllGather 的 per-rank count 不可混淆）；
- `FORMULA_REGISTRY` 用**公式 ID** 固定 algbw / NCCL-tests bus correction / α–β 拟合；
  对 HCCL 原生结果只生成派生列 `normalized_nccltests_bus_correction`（不冒充官方 BusBW 或
  物理链路流量）；
- `TimingCalibration` 结构上拒绝"用 host API 返回时间当 collective latency"；
- `CapabilityDecision` 中 `UNKNOWN` 永不授权执行。

### 3.3 communicator 安全（E10-03）

```text
INIT → READY → ENQUEUED(seq=k) → IN_FLIGHT → COMPLETED
                                   ├→ ERROR → ABORTING → ABORTED → DESTROYED
                                   └→ TIMEOUT → ABORTING → ABORTED
```

- `SequenceAllocator` 的命名空间绑定 `(group_id, rank_epoch)`：重建 communicator 不能复用旧 seq；
- `preflight_check` 只做控制面一致性检查，`handshake_is_not_a_barrier=True` 显式声明它
  **不是**性能路径上的全局 barrier，也不替代 backend 的异步错误处理；
- dtype 比较按语义：`byte_counts_match_but_semantics_differ()` 专门识别 FP16↔FP32 同字节陷阱。

### 3.4 TP plan 与通信账本（E10-04）

```text
QwenArchitectureCensus → parameter_shapes()（每个参数名 → 全局 shape）
TpCapability(degree)   → hidden/intermediate/Q heads/KV heads/vocab 整除 + KV 策略
ParallelPlan           → parameter_shards(ShardSpec[]) + attention/mlp shards +
                         replication policies + embedding + KV ownership +
                         activation layouts + collective events + unsupported predicate
ledger.expected_ledger → 从 plan/shape/phase/token 枚举（绝不从 trace 反推）
ledger.diff_ledger     → 逐事件 op/count/bytes + 原因码（fusion/deferred_gather/padding/…）
```

KV-head 策略是显式枚举（`SPLIT_EVEN`/`REPLICATE`/`UNEVEN`/`REJECT`），
`kv_ownership()` 把"KV memory 不随 TP 下降"的复制成本算进 per-rank 账本；
`unsupported_policy()` 只允许 `PAD`/`UNEVEN`/`REJECT`，`assert_no_truncation()` 结构上禁止截断。

### 3.5 scaling 与时间分解（E10-05）

- `Baseline(status=T1|NO_T1_CAPACITY)`：无真实 T1 时 `strong_speedup` 返回
  `speedup_from_p0` 且 `speedup=None`，note 明写"不得称为 strong scaling efficiency"；
- `ResourceMatrixCell.as_row()` 对缺失格返回 `status=MISSING` + `None` 数值 + 原因（不补 0）；
- `TimeDecomposition` 满足 `compute + exposed_comm + wait + host_gap + sync + unexplained = total`，
  `unexplained_idle_ms` 可为正但被显式报告；overlap 只减一次；
- `pairability()` 用 17 个身份字段判定可配对性，不可配对行进入 `non_pairable` 并保留原因。

### 3.6 overlap（E10-06）

- `legal_overlap_window()` 在没有独立 compute 时返回 `feasible=False`，拒绝"重排依赖制造伪重叠"；
- overlap 由区间集合计算：`union ∩ union`，同侧多个 kernel 先 merge 再求交（不重复计时）；
- `OverlapMetrics.conclusive` 在差值落到时钟不确定度量级时降级；
- `ScheduleIdentity` 强制三组只差一个分量并记录 requested/actual/fallback。

### 3.7 PP/CP/SP（E10-07，P1）

`ActivationDecision` 的未激活态是 `NOT_RUN_NOT_CLAIMED`（不是 PASS/FAIL）；
`validate_candidates` 删除未命名算法的候选（禁止只写 `SP=true`）；
`select_primary` 只选一个主方案，全部不过硬门时给出 `PASS_NEGATIVE` 选择结论。

### 3.8 MoE（E10-08）

`ClaimLevel L0–L4` + `claim_gate()` 决定允许声明；`RouteArtifact` 冻结路由并做守恒审计；
`dispatch_oracle`/`combine_oracle` 处理零 token expert、tail、重复 top-k；
`ComputeOffsets` 用 checked arithmetic；`imbalance_metrics` 报告 max/mean、CV、Gini、熵、
热点持续性、communication cut、critical-rank load（不只报平均）。

### 3.9 trace 与归因（E10-09）

`DistributedTraceEvent` 覆盖 request→iteration→phase→token→layer→module→group/seq→kernel/transport；
`pair_collective_events` 按 group+epoch+seq 配对，缺 rank 进 unmapped 表；
`RootCauseClassifier` 从**最早超过基线置信区间**的事件出发，`UNKNOWN` 是合法答案；
`link_shaping_plan` 无隔离网络时返回 `NOT_RUN`。

### 3.10 故障有界性（E10-10）

`FaultOracle` 预注册 first_detectable_layer / 最大检测与 abort 时间 / 全 rank 终态 /
output_validity=INVALID / communicator_post_state=ABORTED / 允许恢复级别 / 资源差值上限 /
post-recovery probe；`evaluate_fault()` 显式拒绝 §12 列出的 8 类"作弊式通过"（例如错误被
捕获但某 rank 永久 IN_FLIGHT、部分输出被下游消费、communicator 被复用、外部强杀无记录、
资源持续增长仍判通过）。

---

## 4. 数据布局与目录归属

| 类别 | 落点 | 说明 |
|---|---|---|
| 核心接口 | `hqsb/distributed/*.py`（20 文件） | 纯 CPU 可导入，无模块级重依赖 |
| 冻结配置 | `configs/distributed/*.yaml`（12 份） | 严格键校验 + 与代码逐字段审计 |
| 驱动入口 | `scripts/distributed/run_e10.py` | `--list/--prerequisites/--interface-map/--smoke/--execute` |
| 测试 | `tests/unit/distributed/`、`tests/property/test_distributed_invariants.py` | 225 + 15 |
| 运行产物（实验执行时） | `experiment_results/S10/<E>/<run>/` | 由 `RunDirectory` 生成 34 个子目录 |
| 协议证据（只读） | `docs/stage_experiments/S10/**` | 本层代码**从不写入** |

`RunDirectory` 的目录集覆盖 details 各实验的"强制数据产出"：`topology/ placement/ ranks/ groups/
specs/ harness/ cases/ capabilities/ correctness/ golden/ faults/ watchdog/ cleanup/ recovery/
metrics/ ledger/ benchmark/ bandwidth/ models/ scaling/ decomposition/ memory/ traces/ clocks/
network/ injections/ attribution/ analysis/ confirmation/ decision/ commands/ stdout/ stderr`。

---

## 5. 边界（本阶段明确不做）

- **不重做 S07/S08**：不启动 HTTP 服务、不重跑服务容量；S08 trace 只作为上层关联输入。
- **不替代 S11/S12/S13**：不做自动 lowering/autotune，不做跨硬件总矩阵与 TCO，不做
  K8s 弹性与生产 SRE。
- **不执行实验**：所有接口默认拒绝产结论；`--execute` 在本机（单卡、无 S11 前置）仍为
  `BLOCKED`。
- **不修改 `docs/stage_experiments/**`**：协议目录只读。
- **不改上一阶段冻结产物**：`reports/jetson/**`、S01–S09 的契约与报告均未改动。

---

## 6. ADR（关键取舍与已否决方案）

| # | 决策 | 理由 | 已否决的替代方案 |
|---|---|---|---|
| ADR-S10-01 | 新目录 `hqsb/distributed`，不塞进 `runtime`/`serving` | 控制平面 §3.2/§3.3 的模块职责与 R7/R8 规则 | 放进 `hqsb.runtime`：会把"被并行"与"并行策略"绑死，且违反 R3 |
| ADR-S10-02 | CPU loopback/仿真只做 smoke，`claim_allowed()=False` | 禁止把模拟当设备证据 | 用仿真数字充当 collective/TP 结论（被 §20 禁止事项明确禁止） |
| ADR-S10-03 | 前置证据只读 `docs/stage_experiments/**` | 防自我解锁（与 S06/S07/S08 同一陷阱） | 允许脚手架写指纹到 `experiment_results/` 后自证 |
| ADR-S10-04 | 包初始化不导入子模块（连 `TYPE_CHECKING` 也不） | 门禁环检测会把包级边算成环 | 保留 `TYPE_CHECKING` 导入：环 `distributed→experiment→specs→distributed` 实测 FAIL |
| ADR-S10-05 | 带宽公式带 ID 并区分 vendor 原生/派生列 | 防止派生 BusBW 冒充物理链路流量 | 无条件套 NCCL-tests correction 生成 `busbw` |
| ADR-S10-06 | overlap 用区间集合而非 kernel 时长求和 | 并发 kernel 重复计时会夸大重叠 | 累加各 rank/kernel 时长 |
| ADR-S10-07 | 无 T1 时改名为 `speedup_from_p0` 且不给 efficiency | §10/§11 明确禁止伪造 T1 | 用最小可行 p0 线性外推冒充强扩展 |
| ADR-S10-08 | MoE 完成层级显式 L0–L4 并设 `claim_gate` | 防止 L2 micro 冒充完整 EP/MoE | 只报"跑通"不给层级与缺口 |

---

## 7. 回退方式

- 本阶段只新增文件与新增配置：回退＝删除 `hqsb/distributed/`、`configs/distributed/`、
  `scripts/distributed/`、`tests/unit/distributed/`、`tests/property/test_distributed_invariants.py`，
  并在 `scripts/audit/import_dependency_gate.py` 中移除 R7/R8 与 `distributed` 区域
  （`RULES_VERSION` 回到 1.2.0）；
- 未改动任何既有模块的行为，因此回退不影响 S00–S09 的任何已验收结论；
- `module_ownership.md` 的 1.4.0 修订与 README/ledger/status 的 S10 章节需同步回滚（同一提交内）。

---

*本文件是设计与接口说明，不含任何实验结果；实验层状态见 `S10_阶段验收报告.md`。*
