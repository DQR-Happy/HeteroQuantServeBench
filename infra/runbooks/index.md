# HQSB S13 runbooks（模板索引）

> 状态：`DESIGN_ONLY` —— 这些手册**未经故障演练验证**。E13-09/E13-10 要求"发布、恢复、
> 回滚不需要作者临场修改命令"；在这些手册被执行并留下 timeline 证据之前，
> 任何"on-call runbook 已就绪"的表述都不成立。

| runbook | 触发信号（E13-08 告警） | 目标 | 证据落点 |
|---|---|---|---|
| [drain_and_rollback.md](drain_and_rollback.md) | `HQSB-READY-REPLICAS`、`HQSB-ADMISSION-REJECT-RATIO` | 安全缩容/滚动/回滚，不丢请求 | `lifecycle/drain/`、`canary/rollback/` |
| [fault_response.md](fault_response.md) | `HQSB-TTFT-BURN`、`HQSB-GOODPUT-LOW`、`HQSB-QUEUE-AGE`、`HQSB-METRIC-STALE`、`HQSB-TELEMETRY-ABSENT` | 分层定位（queue→runtime→device→system）并按设计降级 | `faults/recoveries/`、`rca/verdicts.parquet` |
| [supply_chain_gate.md](supply_chain_gate.md) | 发布门禁失败、篡改/秘密/高危 CVE 命中 | 阻断或隔离 release，保留可审计原因 | `supply_chain/gates/decision.json` |

## 使用约束（与实验一致）

1. 所有动作必须**可脚本化**：手工 `kubectl edit/exec` 记为该轮 deviation（E13-02 step 36）；
2. 破坏性动作前先解析 target 并核对 blast radius（E13-09 §20.2）；
3. 每个动作后必须验证**服务 + 资源 + 状态**三轴恢复，`Pod Running` 不算恢复（§23 V11）；
4. 手册更新后要用**新 case** 确认，不能用原来的 case 自证（E13-08 step 37）。
