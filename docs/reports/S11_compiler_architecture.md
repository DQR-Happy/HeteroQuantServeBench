# S11 编译器架构与 IR/Lowering 设计（Design 制品）

> 阶段：S11（AI 编译器与自动优化）
> 性质：**设计与契约文档，不是实验报告**。本文不含任何测量数字；所有"能做什么"
> 的陈述都可定位到 `hqsb/compiler/` 的代码与测试。
> 生成时间：2026-09-19
> 依据：`docs/architecture/HQSB_项目完整剖析.md`（§20、§26.1、§29）、
> `docs/stage_experiments/details/S11/README.md`（§4–§17）、
> `docs/stage_experiments/details/S11/E11-01…E11-10`、`docs/stages/S11_AI编译器与自动优化.md`

---

## 1. 目的与边界

S11 的目标不是"给 Qwen 套上 `torch.compile`"，也不是再写一组手工 dispatch 规则，而是把
S02–S10 已由人发现和验证的知识固化成一条**可审计的自动优化链**：

```text
冻结的模型/负载/算子语义/target
  → 真实 Qwen 执行捕获带来源与 shape 约束的图
  → 语义等价（而非外形相似）的 pattern 识别
  → 可验证、可版本化、可重放的中间表示
  → 依据 target capability 与 guard 判断 lowering 是否合法
  → 依据 autotune/cost model 判断哪个合法实现更合算
  → lower 到 S03/S04/S05/S09 已通过门禁的真实 kernel
  → 保存 generated code、binary、cache key 与实际 dispatch 证据
  → operator → block → model 正确性与性能重新验证
  → 动态输入/未知 op/低置信/编译失败/坏缓存安全回退
```

**本文档覆盖范围**：分层与模块划分、多级 IR 与 verifier、身份/lineage、pattern 与
predicate 纪律、lowering registry 与选择链、autotune/cost model 协议、cache 生命周期、
第二栈与 AI 门链的接口设计，以及关键 ADR 与回退方式。

**本文档不覆盖**：任何实验结果、任何性能/命中率/收益数字、任何跨硬件结论。
E11-01…E11-10 的实验层状态见 `S11_阶段验收报告.md`。

---

## 2. 分层与模块架构

```text
L0 基础      identity / ir / records
                 │  （身份、IR、统一记录：被所有上层复用）
L1 前端      capture / guards
                 │  （捕获 census、break、metadata、coverage、guard/variant）
L2 重写      pattern_library / rewrite
                 │  （pattern 契约、结构匹配、语义 predicate、原子重写、pass 纪律）
L3 后端      targets / lowering / backend / codegen
                 │  （capability、registry/选择、backend 九步契约、跨层归因）
L4 搜索      autotune / costmodel
                 │  （搜索空间/预算/holdout、特征/regret/安全回退）
L5–L7 制品   cache / portable / aigate
                 │  （cache 生命周期、第二栈、AI 候选门链）
L8 证据      telemetry / specs / experiment / interface_map
```

依赖方向：**全部只依赖 `hqsb.core` 与包内模块**（module ownership R9/R10）。设计理由：

1. 编译层必须能在 CPU-minimal 环境独立导入与测试（否则无法在 CI 中守住它的不变量）；
2. kernel 以 capability/provider 名称 + artifact locator/hash 描述，禁止 import `ops`，
   避免"优化器"与"被优化的实现"互相绑死；
3. 重依赖（torch/triton）只在函数内惰性导入并给结构化 reason（`NOT_INSTALLED`），
   使 FX/Export 相关能力在缺依赖时可诊断而不是 ImportError。

包入口使用 PEP 562 惰性 `_LAZY`，**不在模块级导入子模块**：否则会形成
`hqsb.compiler → experiment → specs → hqsb.compiler` 的环（被依赖门 R1–R10 的环检测拒绝）。

---

## 3. 多级 IR 与 verifier

### 3.1 层级

`identity.IR_LEVELS` 冻结了协议 §5 的层级：`SOURCE / DYNAMO_FX / EXPORT_ATEN /
HQSB_CANONICAL / HQSB_TARGETED / INDUCTOR_LOOP / TIR / MLIR / TTIR / TTGIR /
BACKEND_NATIVE / LLVM_IR / PTX / CUBIN_SASS / NPU_BINARY`。

- `VOLATILE_IR_LEVELS`（DX/Export/Inductor/LLVM/PTX/TTIR…）**强制 canonical hash**：
  raw hash 只证明字节未变，不能定义身份（地址/临时名/时间戳会漂移）。
- HQSB 自己只定义两层：`HQSB_CANONICAL`（与后端无关的语义 op、effect、符号 shape）与
  `HQSB_TARGETED`（target、候选集、guard、layout 变换、workspace、fallback、选择 provenance）。

### 3.2 为什么是"sidecar IR"而不是新编译器

HQSB IR 只保存 PyTorch 图与多后端 lowering 之间**真正缺失的稳定信息**：语义 op 身份、
有序 operands/results、符号维度与范围、effect（alias/mutation/RNG/state/stream）、
QuantArtifact/layout/KV 语义属性、source lineage、合法 target 集合与拒绝原因、
候选与 fallback。是否使用 SSA 语法不重要，能否唯一表达 def-use/effect/合法性才重要。

### 3.3 verifier 纪律

`IRVerifier` 的 14 项检查把"静默错误"变成结构化 issue（code/detail/op_id/severity）：

| 检查 | 失败后果 |
|---|---|
| graph id / level | 未知层级直接拒绝 |
| 唯一 id、def-use、输出可达 | 悬空值/未定义操作数/环 |
| schema version | 版本不一致拒绝 |
| 类型与 shape | 未知 dtype/layout、rank 不匹配 |
| effect 存在性 | UNKNOWN_UNSAFE 必须带 reason |
| source lineage | 缺 module_path/source_nodes |
| 约束声明 | 无符号约束 ⇒ **不可编译**（导出丢约束是不完整制品） |
| targeted 层候选/选择/fallback | 三者缺一即拒绝 |
| guard 引用 | 引用未知 guard 即拒绝 |

`round_trip()` 保证序列化→反序列化后 canonical hash 不变；未知字段在 `from_dict`
处直接拒绝（strict schema），避免"读进来一半"的静默降级。

---

## 4. 身份、lineage 与版本纪律

- 每层制品记录 `ArtifactIdentity`：artifact_id、run/compile/parent ids、ir_level、format、
  schema_version、source commit + dirty patch hash、capture/pass/lowering/compiler 版本、
  target triple/arch/features、符号约束、guards、**raw_hash 与 canonical_hash**、
  created_at、command、environment fingerprint、verification status、consumer。
- `LineageGraph` 校验重复 id/未知父/环，并按实验给出必需 IR 链（`IDENTITY_CHAINS`），
  用于 E11-03 H2"每层表示保持可验证 lineage"。
- 三个身份显式区分（协议 §6）：`graph_identity`（结构+schemas+constants+metadata）、
  `semantic_identity`（canonical 图 + 模型/量化策略 + 数值/状态合同）、
  `compile_identity`（pass 流水线 + registry + kernel build + toolchain + target + guard 域 +
  autotune/cost-model id）。三者都标注为**审计键**，不声称等于任何框架内部 cache key；
  实验需同时记录可观察的内部 key 与 HQSB key。
- 版本一律要求冻结：`latest/main/HEAD/unknown/stable`/空值被 `version_is_frozen` 拒绝；
  driver、target 快照、tuning DB key 共用同一判定。

---

## 5. Pattern 与 rewrite：fail-closed 的合法性

### 5.1 四个部分严格分离

1. `structural matcher`（`find_candidates`）：只做候选发现，按 `ROLE_CHAIN` 前向遍历
   （锚点 → 消费者链），可穿越**已声明**的 decomposition cast；部分命中也要记录
   "最接近层级"（`missing_role`），不把 near-miss 从证据里删掉。
2. `semantic predicates`（8 条，冻结顺序）：structural_match → dtype_cast → shape_reduction
   → epsilon → users_liveness → alias_mutation_effect → layout → return_contract。
   每条输出 expected/actual/outcome/proof_source/reject_code；**UNKNOWN 按拒绝处理**。
3. `replacement builder`：构造一个版本化 semantic op，复制 source/shape/effect metadata，
   并保留 fallback 子图引用。
4. `proof record`：即使结构匹配失败也保留可解释的拒绝理由（`PatternDecisionRecord`）。

### 5.2 三类硬拒绝（不可用 allclose 交换）

- **累积差异**：把 reduction 输入重新舍入到 storage 精度（fp16）后参与归约，与 eager
  语义不同 ⇒ `PATTERN_NEAR_MISS`。CPU oracle 用 `reduction_inputs_differ` 提供确定性证据，
  明确"某个样本 allclose 通过不构成合法性"。
- **mutation/alias**：`aten.add_`/写回 residual、effect 证据为 unknown、layout/view 改变
  alias identity ⇒ `MUTATION_UNSAFE`/`ALIAS_UNSAFE`。
- **users/liveness**：pattern 外消费者必须被 replacement 精确保留；否则拒绝。

### 5.3 pass 纪律与原子性

`apply_rewrite` 从不修改输入图：新图通过 verifier 后才返回，任何注入点
（matcher/replacement/verifier/commit）失败都返回原图 + 结构化错误。
`run_pipeline` 提供固定点收敛与振荡检测；`idempotence_report` 要求二阶 rewrite=0 且
canonical hash 不变；`metadata_diff` 检查 provenance 不增殖；`pass_identity` 把规则与
predicate 顺序计入 compile identity。

---

## 6. Lowering registry 与选择链

### 6.1 registry entry 的字段（协议 §10）

semantic op/schema、backend/implementation/build id、supported dtype/layout/shape/arch/features、
required alignment/workspace/stream 语义、mutation/alias 合同、guard builder、artifact
locator/hash/signature、correctness evidence scope、performance evidence scope、priority、
cost-model feature adapter、fallback target。

**注册无副作用**：import 时不编译、不创建 device context、不下载。

### 6.2 选择链（每条候选都留痕）

```text
semantic legality（E11-02 已证明）
  → target capability（arch/dtype/feature/ABI/资源，结构化 reject reason）
  → artifact compatibility（hash/ABI/version）
  → runtime guard
  → correctness evidence coverage（operator/block/model 分级）
  → policy 选择（reference | forced:<id> | auto_heuristic）
  → 无合法候选 ⇒ reference lowering / 原图
```

`forced:<id>` 在不可用时**明确失败或按配置回退**，绝不悄悄改 actual；`MaterializedPlan`
强制要求 fallback 可达，并拒绝"每调用重做选择"。

### 6.3 actual dispatch 的证明

`ActualDispatchEvidence` 要求 ≥2 类独立证据（compiled code symbol / profiler symbol /
kernel telemetry / 受控 fail build / candidate disable ablation）。`DispatchRow` 要求
actual_candidate_id ≠ selected 时必须有 fallback_reason（禁止静默回退），
`SideEffectGate` 要求所有检查在 launch 之前完成（禁止"先写状态再回退"）。

---

## 7. Autotune 与 cost model 协议

- **搜索空间**是语义对象：knob 必须说明控制的机制；约束必须带类别/理由/**可判定
  predicate**（未知不得静默通过）；候选 identity 覆盖 config + kernel build + target +
  flags；`max_candidates` 在生成前声明，超限直接拒绝。
- **三道门**：static legality/capability → compile/resource → correctness → benchmark。
  invalid 候选不得被计时（`latency=∞` 不是观测），correctness 必须先于 benchmark。
- **预算**：B0–B3（default/small/medium/exhaustive）带 wall/device/compile/disk 上限与
  early stop；不同方法预算不一致被判"不公平比较"。
- **数据切分**：按 semantic graph family + regime + bucket + dtype/layout + arch 分组，
  组不跨 split；final test 只能消费一次（`consume_final_test`），random-row 只作诊断。
- **主指标**是 holdout chosen-vs-oracle regret（mean/p95/max/catastrophic/加权），
  不是 prediction loss；`fast_p`/speedup 的分母包含 correctness 失败，不静默剔除。
- **回退链**（fail-closed）：无合法候选→语义 fallback；OOD/缺关键特征→heuristic；
  低置信→benchmark top-k 或 fallback；模型只能在**已合法**候选集中排序
  （`illegal_candidate_injection` 证明它无权把非法变合法）。
- **裁决分离**：methodology PASS 与 deployment PASS 分开；deployment 门含 regret/tail/
  catastrophic/overhead/confidence 单调性。

---

## 8. Cache 与 artifact 生命周期

cache 的定义是**条件化复用**：

```text
cache_hit = 存在 ∧ metadata 可解析 ∧ schema 受支持 ∧ 重算 key 一致 ∧ 语义/ABI/target 兼容
            ∧ guard 域覆盖当前输入 ∧ metadata hash 一致 ∧ payload hash 一致 ∧ 状态为 COMMITTED
```

- **key 规范**：22 个字段逐项带 inclusion reason；非语义字段（timestamp/pid/output_root/
  absolute_path/log_level/hostname）**禁止**入 key；缺字段即报错（不静默空串）。
  `evaluate_key_vectors`/`key_stability` 给出可复跑的自检。
- **事务发布**：`lock → TEMP → VALIDATING → atomic rename → manifest 提升为 COMMITTED
  → COMMITTED marker 最后`。writer 在任一阶段被杀都不留下可读半成品；同 key 二次发布
  被拒绝（内容寻址去重由上层负责）。
- **安全读取**：parse→schema→key→compat→guard→state（marker 与 manifest 状态必须一致）
  →metadata hash→payload hash→load；任一失败 `load_calls=0`（不 dlopen、不 launch），
  损坏 entry 被 quarantine。
- **损坏与失效**：11 例损坏注入在隔离副本上执行；14 行失效矩阵（噪声必须 HIT，语义/
  compiler/arch/ABI 必须 MISS，guard false 必须 MISS/FALLBACK），观测与矩阵不符即不通过。
- **计时边界**：C0 冷编译 / C1 同进程内存命中 / C2 新进程磁盘命中 / C3 prebuilt 加载 /
  C4 稳态；reset scope 显式记录，不声称"绝对冷机"。

---

## 9. 第二栈（TVM/MLIR）与 AI 候选门链

- **第二栈**：P0 只要求一条 runnable loop（同一真实 semantic op：high-level → legalisation
  → loop IR → schedule → target → runtime），另一条标 `NOT_RUN_SCOPE_LIMITED`。语义映射表
  逐行声明 must-preserve，未映射语义默认 block 或显式 external call；bridge 合同六项
  （ownership/stride/device/stream/sync/error）必填，零拷贝声明与 copy 实现矛盾即拒绝；
  采用裁决只允许 4 类公开结论，绝不因"跑通一个 op"声称通用编译器或性能可移植。
- **AI 候选**：生成与评估分离；候选视为不可信代码（无网络/无 secrets/只读 reference/
  正式 registry 与 cache 不可写/配额齐全）；门链 G0–G10 累积，后门永远不能覆盖前门失败；
  hidden + randomized + adversarial correctness 与 sanitizer 是 admission 的必要条件；
  admission schema 缺任一必需字段即失败；`claim_boundary` 规定对外可用措辞。

---

## 10. ADR 与关键取舍

| ADR | 决策 | 被否决的替代方案 | 回退方式 |
|---|---|---|---|
| ADR-S11-01 | HQSB IR 作为 **sidecar**，不重写 MLIR | 自建完整 MLIR 方言/dialect 栈 | 删除 `ir.py` 只保留 FX metadata（E11-03 会随之降级为 BLOCKED） |
| ADR-S11-02 | 只依赖 `hqsb.core`，不 import `ops`/其他区域 | 直接调用 `ops.dispatcher` 复用选择逻辑 | 改为 capability/provider 字符串 + artifact locator（现状即此） |
| ADR-S11-03 | 合法性由 predicate 决定，性能只决定"选哪个合法实现" | 用 benchmark 结果反推 pattern 是否合法 | 任何削弱 predicate 的改动必须带负向对照（否则 S11-24 语料测试失败） |
| ADR-S11-04 | cost model 只能在合法/证据覆盖的候选集中排序 | 让模型参与 capability/legality 判定 | `illegal_candidate_injection` + `lowering.evaluate_candidates` 双保险 |
| ADR-S11-05 | cache 安全优先：unknown/不兼容/损坏一律 fail closed | "先 load 试试看"、按文件存在判 hit | 回退到重编译或 reference lowering（cache 层可整体关闭） |
| ADR-S11-06 | autotune 必须独立 holdout + 新进程 confirmation | 在 tuning shapes 上报 winner | holdout 失败→per-shape tune/bucket/default（策略文档强制） |
| ADR-S11-07 | 实验脚手架默认拒绝产结论（三重门） | 允许 driver 直接写 PASS | `write_verdict` 三重门 + `--execute` 显式开关 |
| ADR-S11-08 | AI 候选门链固定顺序、后门不覆盖前门 | 由 reviewer 口头豁免某些门 | `run_gate_chain` 记录最早失败；admission schema 强制字段 |

---

## 11. 与上游阶段的边界（不重做）

- **不重做 S06**：S06 回答"自定义 op 能否被 dispatcher/FakeTensor/Dynamo/rewrite/cache
  稳定接入"；S11 消费其真实输入，新增多级 IR/verifier、pass legality、target capability、
  IR→binary→counter 解释链、train/holdout 分离的 autotune、跨进程 cache 生命周期与第二栈对照。
- **不重做 S03/S04/S05/S09**：kernel 是受版本管理的 lowering target；若 kernel 自身没有
  对应 shape/dtype/layout/arch correctness evidence，编译器**不得**通过自动选择把它
  "升级"为可信实现（`EvidenceIndex` 强制 evidence scope 覆盖当前点）。
- **不重做 S07/S08**：S11 可以使用 S07/S08 的真实 shape trace 训练/验证选择器，但不得在
  改动 scheduler/arrival process 后把收益归因给 graph compiler。
- **不替代 S12/S13**：跨硬件公平排名/TCO/能效属 S12；签名、SBOM、容器、canary、回滚属 S13。

---

## 12. 遗留风险与限制（如实登记）

1. **前置缺口**：本机缺少 S03/S04 硬件证据、S06 模型级 pattern correctness 与冻结编译器
   环境指纹 ⇒ 真实 Qwen/Inductor 链未执行；`run_e11.py --prerequisites` 如实报告。
2. **工具缺口**：nvcc/cuobjdump/nvdisasm/nsys/ncu 均不在 PATH ⇒ E11-04 的 PTX/SASS/counter
   层在正式 run 时只能标 `NOT_RUN_TOOL_UNAVAILABLE`。
3. **第二栈与 AI 门链**：TVM/MLIR、AI 候选均为接口层；E11-10 为 P1，未激活。
4. **mypy 非门禁**：CI 中 mypy 步骤为注释状态；本层 mypy 结果与既有模块同量级，
   本阶段不声称 mypy 通过（见验收报告 §5.8）。
