# S13 生产化架构与关键取舍（设计制品）

> 阶段：S13（`hqsb/infra/` + `configs/infra/` + `scripts/infra/` + `infra/`）
> 性质：**设计制品（Design）**；不包含任何实验结果、性能数字或可靠性结论
> 生成时间：2026-09-19
> 依据：`docs/stage_experiments/details/S13/README.md` §5–§23、`docs/stages/S13_生产化云原生与可靠性.md`、
> `docs/architecture/顶层架构.md`、`docs/architecture/HQSB_项目完整剖析.md` §22/§26/§29、
> `docs/architecture/module_ownership.md`（1.7.0）
> 结论用词：本文件中的架构与接口是 **`IMPLEMENTED_UNVERIFIED`**；任何"已生产化/已可靠/已安全"的表述都不成立。

---

## 1. 五个平面的职责边界（`details/S13/README.md` §5）

| 平面 | 由谁承载 | 本阶段交付的代码落点 |
|---|---|---|
| Supply-chain plane | 镜像/依赖/扫描/证明链 | `hqsb/infra/supply_chain.py`、`hqsb/infra/identity.py`、`infra/containers/` |
| Deployment control plane | IaC/namespace/RBAC/设备/调度/探针/发布 | `hqsb/infra/deployment.py`、`scheduling.py`、`infra/deploy/helm/` |
| Artifact plane | 模型 URI→staging→verify→cache→active→GC | `hqsb/infra/artifacts.py` |
| Serving data plane | auth/quota→admission→queue→runtime→stream | `hqsb/infra/capacity.py`、`lifecycle.py`（协议与账本） |
| Observability & reliability plane | SLI/SLO/告警/故障/发布治理 | `hqsb/infra/observability.py`、`faults.py`、`canary.py`、`security.py`、`infra/observability/`、`infra/runbooks/` |

**依赖方向**：`hqsb.infra` 只依赖 `hqsb.core`（依赖门 R13/R14）。因此上面每个平面都以
**identity/record + 校验器**的形式落位，不反向依赖 `serving`/`evaluation`/`ops`：
集群与集群对象由外部工具产生，本层负责"记录 + 判定 + 拒绝"。

---

## 2. 三个状态机必须分开（§9）

| 状态机 | 语义 | 落点 |
|---|---|---|
| Pod/进程生命周期 | PENDING→…→TERMINATED/FORCED_KILL | `records.POD_STATE_MACHINE`、`deployment.DEPLOYMENT_STATE_MACHINE` |
| 模型生命周期 | ABSENT→…→ACTIVE→…→GC_ELIGIBLE | `records.ARTIFACT_STATE_MACHINE`、`artifacts.activate_version` |
| 请求生命周期 | RECEIVED→…→COMPLETED/CANCELLED/…→ACCOUNTED_RELEASED | `records.REQUEST_STATE_MACHINE`、`lifecycle.DrainTimeline` |

不变式：`Pod Ready ≠ 模型 ready`、`模型 loaded ≠ 容量足够`、`客户端断开 ≠ device work 取消`。
因此 `deployment.readiness_verdict` 要求 9 个语义条件同时成立，`capacity.admit` 独立于 Pod 状态。

---

## 3. 关键设计决策与权衡（含被否决方案）

| # | 决策 | 理由（文档依据） | 被否决的替代方案 |
|---|---|---|---|
| D1 | **状态码四态以上，不用布尔**：`PASS / PASS_NEGATIVE / FAIL / BLOCKED / N/A_BY_ADR`，S13 另有 6 个 `NOT_RUN_*` 前置态 | `stage_experiments/README.md` §3 + `details/S13/README.md` §4 | 布尔 `passed`：无法区分"没跑"与"跑了失败"，也无法表达安全边界拒绝执行 |
| D2 | **缺失是状态，不是 0**；`records.numeric_value` 拒绝把 missing 变成数字 | `details/S13/README.md` §19 + S12 既有口径 | `None→0` 或留空：会让"未观测"看起来像"零错误/零延迟" |
| D3 | **gate 用引用/身份 + 解析后的符号**（`interface_map`），而非文档表格 | 控制平面 §29.1 + 反模式"文档很多但没有闭环" | 手写对照表：会随重命名腐化（本项目已有 S12 先例） |
| D4 | **驱动默认拒绝产结论**（`--execute` + 前置 + raw 样本三重门） | `stage_experiments/README.md` §6 + 本任务约束 | 允许 `--experiment X` 直接写 verdict：会把"接口就位"写成"实验通过" |
| D5 | **模板阈值一律标注 `POLICY_DEFAULT_UNVERIFIED`**，由测试断言标记存在 | §20.1 预注册 + 反模式"把计划阈值当测量值" | 在 values/alerts 里写"看起来合理"的数字：会被误读为测量结果 |
| D6 | **active/inflight/canary/rollback 制品必须持有 pin/lease**，GC 只删 eligible | E13-04 §4.4 + §11 门 8 | "按最旧目录删除"：会删掉回滚目标（生产回滚能力直接失去） |
| D7 | **三类 probe 语义分离**，且 liveness 不得读取 queue/load | E13-05 §2 + Kubernetes 官方 probes 语义 | 三个 probe 指向同一 `/healthz`：高负载下制造级联重启 |
| D8 | **admission 决策必须带 policy/reason/budget**；reserve/commit 原子 | §12 + E13-06 §4/§21 | 只按请求数准入：mixed-length 下会给长请求分配不足预算 |
| D9 | **autoscaling 决策必须带 metric age**，stale/missing 一律 fail-safe | E13-07 §7 + §23 V08 | 用"最后一次读数"当最新：stale 时可危险缩容到 min/zero |
| D10 | **canary 的硬门（G0–G4）不可被性能补偿**；序贯规则冻结；`INCONCLUSIVE` 不自动推进 | E13-10 §4/§6 + Google SRE canary | 用"综合评分"混合质量与性能：质量退化会被性能收益掩盖 |
| D11 | **故障合同必须先解析 target、声明 blast radius、kill switch、abort 阈值** | E13-09 §2/§20.2 | "随机杀一个 pod"：无 ground truth、无法复现、可能越界影响他人 |
| D12 | **多租户结论必须带威胁模型与样本边界措辞**（`NEGATIVE_RESULT_WORDING`） | E13-11 §10/§5 | "未攻破即安全"：把有界测试写成绝对安全声明 |
| D13 | **`hqsb.infra` 只依赖 `core`，不 import `serving`/`evaluation`/`ops`** | `module_ownership.md` 依赖方向 + R13/R14 | 直接调用 S08/S12 的对象：会把"评估器/服务器"绑进部署层，且 CPU-minimal 失效 |

---

## 4. 回退方式（每一项都可回退）

| 交付 | 回退动作 | 影响面 |
|---|---|---|
| `hqsb/infra/*` | 删除包目录（无其他区域 import，R13 保证） | 无：没有下游依赖 |
| `configs/infra/*.yaml` | `scripts/infra/gen_infra_specs.py` 可整体重建 | 词汇表与代码同步 |
| `infra/**`（Dockerfile/Helm/告警/runbook） | 删除目录；未接入任何集群或流水线 | 无：`deployment_assets_present` 前置会转 BLOCKED |
| 依赖门 R13/R14 | 从 `REGION_PREFIXES`/`RULES` 移除并下调 `RULES_VERSION` | 门禁变松（需在报告中登记，不得静默） |
| `scripts/infra/*.py` | 删除驱动与生成器 | 接口仍在包内，可由实验所有者自行调用 |
| `docs/reports/S13_interface_map_generated.md` | `scripts/infra/gen_interface_map.py` 重新生成 | 生成物，无手工内容 |

---

## 5. 本设计**不**主张什么

1. 不主张任何镜像已构建、SBOM 已生成、漏洞已扫描（E13-01 未运行）；
2. 不主张任何集群部署、升级、回滚已执行（E13-02/E13-05/E13-10 未运行）；
3. 不主张任何容量数字、OOM margin、autoscaling 参数已被验证（E13-06/E13-07 未运行）；
4. 不主张 alert 已触发过、RCA 已完成（E13-08 未运行）；
5. 不主张任何故障已被注入或恢复（E13-09 未运行）；
6. 不主张多租户安全已成立（E13-11 未运行）；
7. 不主张 SLSA level、Kubernetes conformance、任何安全认证（`details/S13/README.md` §29）。

以上每一项的接口都在 `hqsb/infra/` 中就位，并由单元/属性测试证明其**契约**成立；
实验层判定见 `docs/reports/S13_阶段验收报告.md`（`BLOCKED`）。
