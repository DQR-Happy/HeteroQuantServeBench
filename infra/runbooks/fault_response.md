# Runbook：故障分层定位与有界降级（模板，未演练）

## 触发
`HQSB-TTFT-BURN` / `HQSB-GOODPUT-LOW` / `HQSB-QUEUE-AGE` / `HQSB-METRIC-STALE` /
`HQSB-TELEMETRY-ABSENT` / `HQSB-ACCELERATOR-THROTTLE`。

## 分层定位（按 E13-08 的 taxonomy 逐层排除，禁止跳层断言）
1. **client/gateway**：TTFT 的 client 与 server 边界对比（同名指标混淆会误判）；
2. **admission/queue**：`queue_tokens`、`oldest_age`、拒绝 reason code 分布；
3. **runtime/model**：actual backend、compile/cache hit、prefill/decode 相位；
4. **op/kernel/device**：设备 util/memory/clock/throttle、link/collective；
5. **system/storage/network**：CPU/RAM/swap/IO、对象存储延迟、DNS/重传；
6. **control plane**：rollout/autoscaling/fault 事件时间线。

## 有界降级（按预注册策略，顺序不得随意调整）
`MASKED_REDUNDANCY` → `RETRY_BOUNDED` → `ADMISSION_SHED` → `PAUSE_NOT_READY` → `ROLLBACK` → `FAIL_CLOSED`。
任何改变模型/精度/语义的 fallback 必须**显式声明 + 重过质量门 + 对客户端可见**。

## 必须记录
- ground truth 生效时间（命令返回 ≠ 故障生效）；
- MTTD/MTTM/MTTR（以用户稳态为准，不以 Pod Running 为准）；
- 残留观察窗口内的延迟故障（内存泄漏、stale metric、cache poisoning）；
- 每个动作的 owner 与验证方式（postmortem 要求）。
