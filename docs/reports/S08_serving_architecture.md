# S08 设计制品：HQSB ServeFabric 服务平面架构

> 阶段：S08；制品类型：Design 制品（对应 `docs/reports/S07_runtime_architecture.md` 的角色）
> 生成时间：2026-09-18
> 上游设计依据：`docs/architecture/顶层架构.md`、`docs/architecture/HQSB_项目完整剖析.md`、
> `docs/architecture/module_ownership.md`、`docs/stage_experiments/details/S08/README.md` §6 分层

## 1. 分层与依赖方向

```
                     ┌────────────────────────────────────────────┐
                     │  evidence plane（证据平面）                  │
                     │  observability / telemetry / faults /      │
                     │  service_ab / experiment / specs /         │
                     │  interface_map                             │
                     └──────────────┬─────────────────────────────┘
                                    │ 读取所有下层产出
        ┌───────────────────────────┼───────────────────────────────┐
        │  policy plane（策略平面）   │                               │
        │  slo / arrival / loadgen / │  admission / policies /      │
        │  fairness                  │  fairness / router / circuit │
        │                            │  / cache_routing             │
        └──────────────┬─────────────┴──────────────┬───────────────┘
                       │                            │
        ┌──────────────┴──────────┐      ┌──────────┴───────────────┐
        │  gateway plane（网关平面）│      │  protocol plane（协议平面）│
        │  gateway / transport /  │      │  protocol / sse           │
        │  timing / pipeline      │      │                           │
        └──────────────┬──────────┘      └───────────────────────────┘
                       │
        ┌──────────────┴───────────────────────────────────────────┐
        │  backend plane（后端平面，只消费 S07 稳定契约）              │
        │  BackendSession / ServingBackend（Protocol）+ dummy 夹具   │
        └───────────────────────────────────────────────────────────┘
```

依赖只向下；`hqsb.serving` 不 import `ops`，模块级不 import `torch`/`triton`/
`numpy`；下层（`core`…`runtime`）不得反向依赖 `serving`（gate R5/R6）。

## 2. 关键抽象

### 2.1 三个视角可分离的请求（E08-01 §5）

`requested`（客户端原样）→ `normalized`（默认值展开，进入 config hash）→
`actual`（后端实际收到的 token ids / sampling / template 是否应用），三者永不
合并，任何 fallback 都带 reason。

### 2.2 外部提交点（retry 边界）

`before_backend_start` / `backend_started_no_external_bytes` /
`after_external_commit`。流式一旦写出首个字节，透明重试被禁止，只能
显式 incomplete/error 终帧。

### 2.3 时间边界与五种 TTFT

`t_sched→t_send→t_gateway_recv→t_validate_done→t_enqueue→t_dequeue→
t_backend_submit→t_runtime_first→t_first_frame_write→t_first_byte_client→
t_runtime_last→t_terminal_write→t_client_done→t_cleanup_done`；
TTFT 必带前缀（runtime/server/client），跨进程时间戳需实测 offset 才能相减。

### 2.4 token/字节投递账本

`generated ≥ committed ≥ emitted ≥ flushed ≥ client_received`，五个缓冲层
（application_queue / serializer_buffer / server_buffer / socket_send_buffer /
client_receive_buffer）各有 owner/cap/溢出动作。

### 2.5 能力与身份

Backend 通过 `BackendRegistry`（原子代际）注册，硬过滤先于评分，路由决策记录
candidate 集/排除原因/遥测年龄/评分分量/selected(instance, model_epoch,
route_epoch)，`route_vs_actual` 逐请求核对。

### 2.6 证据与门禁

S08 实验脚手架拒绝在无前置/无 raw samples 时产结论；SLO 模板未冻结时拒绝评估；
`interface_map.resolve_interfaces()` 保证 264 步对照不腐烂。

## 3. 与 S07 的边界

- 复用 S07：`request`（canonical request/RequestSpec）、`adapter`（七类操作）、
  `prefix_cache`（key 绑定/refcount）、`metrics`（口径）、`policy_ab`（A/B 门禁与
  统计）、`telemetry`（C6/C7 投影）、`comparison`（S08 稳定接口面）。
- S08 新增：协议/SSE、网关与请求状态机、传输、SLO/到达/loadgen、公平性、
  准入/熔断/路由/缓存路由、可观测性、服务 A/B、S08 实验脚手架。
- 不侵入 S07 冻结契约；`hqsb.serving` 只消费其公共对象。

## 4. 交付边界

本设计制品只描述接口与不变量；任何「已实现」的声明以
`docs/reports/S08_开发报告.md` §7 与 `docs/evidence_ledger.md` §14 为准。
