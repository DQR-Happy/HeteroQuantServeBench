# S12 跨硬件评估与统一 Benchmark —— Design 制品

> 阶段：S12（`hqsb/evaluation/` + `configs/evaluation/` + `scripts/evaluation/`）
> 性质：**接口/代码层设计；实验层 BLOCKED**
> 生成时间：2026-09-19
> 依据：`docs/architecture/HQSB_项目完整剖析.md` §21/§26.1、`docs/stage_experiments/README.md`、
> `docs/stage_experiments/S12_实验清单.md`、`docs/stage_experiments/details/S12/*`（11 份，逐份读完）、
> `docs/stages/S12_跨硬件评估与统一Benchmark.md`、`docs/architecture/module_ownership.md`（1.6.0）

---

## 1. 设计目标与边界

S12 要回答的不是「哪张卡 tokens/s 最高」，而是「在给定 workload/质量/SLO 下，哪个
硬件×精度×runtime×并行策略更合适，且结论公平、可审计」。因此本层是**决策基础设施**，
不是又一个 benchmark 脚本：

- 每个比较组必须有冻结合同、字段分类与四态裁决；
- 每项能力必须区分声明/发现/验证/正式 benchmark；
- 每项结论必须能沿 lineage 反查到 raw、配置、模型、环境、价格/功耗来源。

本层**不执行任何正式实验、不内置任何价格/功耗数值**；所有数值都是 campaign 输入。

## 2. 分层与依赖方向

`hqsb/evaluation` 位于所有区域之上，**只依赖 `hqsb.core`**（规则 R11/R12）：

```text
core
 └─ evaluation
     ├─ L0  identity / records / contracts / layers / campaign
     ├─ L1  candidates / comparability / platform / capability
     ├─ L2  benchmark / repeatability
     ├─ L3  roofline / energy / cost
     ├─ L4  pareto / maturity
     ├─ L5  lineage
     └─ L6  telemetry / specs / experiment / interface_map
```

依赖只向下；`interface_map` 聚合各实验模块的 `PROTOCOL_STEPS`（每模块 36 步 × 10 实验 =
360 步），并用 `resolve_interfaces()` 逐符号导入校验，映射不可腐烂。

## 3. 关键设计决策与权衡

| # | 决策 | 理由 | 被否决的替代方案与回退 |
|---|---|---|---|
| 1 | 四态可比性（`COMPARABLE/CONDITIONAL/NOT_COMPARABLE/INSUFFICIENT_EVIDENCE`），不是布尔 | 布尔会迫使「有差异」强判可比；`CONDITIONAL` 必须带公式/适用域/禁止主张 | 布尔 + 事后备注（无法被 validator 拒绝非法 join） |
| 2 | 缺失是**状态**不是零（10 类 `MISSINGNESS_CODES`） | 否则「不支持」会被错误排成最省电/最低成本 | 统一 `null`（丢失证据语义） |
| 3 | 能量 `total` 与 `incremental` 同报、不同 boundary 不同排序 | 只报 incremental 或混排 device/node 会隐藏 idle/系统边界 | 只报好看的那个（被 `publish_metrics` 拒绝） |
| 4 | 成本**无内置价格**，价格快照是 dated 输入 | 成本是场景，不是硅片常数 | 硬编码 `$x/小时`（违反「来源/日期/地区/币种」） |
| 5 | peak 只作乐观偏差量化，**不得**当容量分母 | peak 会低估成本、隐藏闲置 | peak 当 goodput（被 `QualifiedCapacity` 拒绝） |
| 6 | 成熟度三合法路径（硬约束/独立目标/风险标签），永不合成总分 | 总分是不透明黑箱 | 0–100 加权（违反 §16/§17） |
| 7 | lineage raw append-only、fault injection 必须定位到实体与下游 claim | 「aggregate root 变了」无法修复任何东西 | 只报根 hash 不同（被 `evaluate_fault_injection` 拒绝） |
| 8 | 决策回归套件标注 `algorithm_regression`（合成数据，非实验） | 用合成小数据验证支配/过滤方向，又不冒充实验结果 | 混进实验结论（被 `claim_allowed=false` 阻断） |
| 9 | `artifacts/S12/` 逻辑目录落到 `experiment_results/S12/`（仓库既有约定） | 与 S08/S10/S11 一致，且不污染协议树 | 新开顶层 `artifacts/`（破坏仓库目录语义） |

## 4. 数据流（一个候选的完整旅程）

```text
candidate identity（canonical 字段 → candidate_id）
  → E12-01 可比性审计（字段分类 + contract + 四态裁决 + suite manifest）
  → E12-02 capability 探测（四级证据 + 失效规则 + coverage join）
  → E12-03 四层重放（Observation → NormalizedResult，11 条交叉校验）
  → E12-04 重复性（block/process/day + 排除账本 + bootstrap）
  → E12-05 Roofline/Amdahl（预测 → 实测 → 残差 → 验证）
  → E12-06 能量（窗口对齐 + 积分 + compliant 分母）
  → E12-07 成本（dated 价格 + 组件账本 + 敏感性）
  → E12-08 Pareto（证据 → 质量 → 硬约束 → 支配 → 敏感性 → 推荐）
  → E12-09 成熟度（rubric 向量，独立维度）
  → E12-10 lineage（反查 + 重生成 + 故障定位）
```

## 5. 每阶段六类制品映射（控制平面 §26.1）

| 制品 | 本阶段交付 |
|---|---|
| Design | 本文件（目标/边界/分层/9 项决策与权衡/回退） |
| Implementation | `hqsb/evaluation/` 21 模块 + `configs/evaluation/` 12 份 + `scripts/evaluation/run_e12.py` |
| Correctness | 319 单元 + 38 属性：负向路径（非法 join、forbidden 归一化、peak 当分母、NOT_EVALUATED=0、fallback 验证、append-only、TDP 代测） |
| Performance | **未产出**（不执行实验）；成本口径/能效口径/归一化分母接口就位 |
| Acceptance | `docs/reports/S12_阶段验收报告.md`（代码层 / 实验层分列） |
| Evidence Manifest | `hqsb/evaluation/experiment.py::EvidenceManifest`（S12 身份字段 + 重生成等级）；本阶段无运行证据，不写 manifest 实例 |

## 6. 前置与阻塞（实测）

4 条必需前置中 3 条未满足：`s03_s11_upstream_evidence_chain`（协议树仅 S01/S02 有
verdict）、`multi_hardware_coverage`、`frozen_evaluation_environment`；3 项 advisory
（profiler/导出工具、功率计证据、价格快照）未满足。因此 E12-01~E12-10 实验层 `BLOCKED`，
energy/cost 停在接口层，见开发报告 §2/§8。

## 7. 回退方式

本层是**新增**区域，不改写任何上游契约：若后续发现某接口语义与 `details/S12/*` 冲突，
最小回退是修改对应模块与测试（同一目录内），不影响 S01–S11 已冻结交付；
`configs/evaluation/*.yaml` 只冻结词汇表，任何数值化（价格/阈值）改动都在 campaign 输入侧完成，
无需回退代码。

---

*生成时间：2026-09-19；本文件不含任何实验结果或结论数字。*
