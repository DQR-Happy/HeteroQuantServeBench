# Runbook：drain、缩容与回滚（模板，未演练）

## 触发
`HQSB-READY-REPLICAS`（ready 容量低于下限）或 `HQSB-ADMISSION-REJECT-RATIO`（拒绝率超阈值），
或发布过程中 canary gate 返回 `STOP`/`ROLLBACK`。

## 前置检查
1. 确认目标 release/model 身份（`release_id`、`model_artifact_id`、`image_index_digest`）；
2. 确认回滚目标存在且被 pin/lease 保护（`artifacts:RollbackPolicy`、`gc_safety_check`）；
3. 确认 availability 预算：PDB、maxUnavailable、当前 ready 副本数。

## 步骤
1. **停止接新请求**：置 readiness=false（`/readyz` 返回 not-ready），确认 EndpointSlice 已移除该 Pod；
2. **drain inflight**：等待 ≤ `maxGenerationSeconds`；超时的长流按预注册语义取消
   （`CANCELLED_EXPLICITLY` 或 `PARTIAL_STREAM_NONRETRYABLE`），**不得**把 partial 记为 complete；
3. **释放资源**：KV/queue slot/device context/cache lease/socket，逐项记录 released 布尔值；
4. **回滚**：切换 active generation 到 known-good；记录 decision→routed→baseline restored 时间戳；
5. **验证闭环**：重跑质量探针 + SLO/capacity 检查 + 残留对象清点（endpoints/model slots/leases）。

## 禁止
- 用 `kubectl delete pod --force` 代替 drain（forced kill 必须记为 SLO 违约）；
- 依赖客户端重试掩盖丢 token；
- 为了快速缩容删除 rollback 目标或被 pin 的制品。
