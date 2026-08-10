# S09 Ascend Backend Report

> 更新日期：2026-09-21<br>
> 性质：**已实现控制面 + 未实现/未验证数据面**<br>
> 科学状态：**BLOCKED**

## 1. 架构目标

S09 的目标是让 CUDA 阶段形成的 OperatorSpec、test vector、容差、模型 workload 和 C1～C7 证据契约在 Ascend 上保持同一语义。硬件差异只能通过 capability、tiling、workspace、actual kernel、format conversion 和 trace 扩展显式表达，不能通过修改数学定义或静默 fallback 隐藏。

目标依赖方向是：

```text
core ← models/benchmark ← ascend/backend adapter ← runtime ← serving
```

当前仓库只实现了该目标的一部分，不能据此声称已经存在可工作的 Ascend Backend。

## 2. 已实现

| 文件 | 已实现范围 | 证据等级 |
|---|---|---|
| `compatibility.py` | manifest、canonical hash、schema、redaction、field diff | CPU test/control plane |
| `capability.py` | 四态 capability、strict decision、requested/actual/reason | CPU test/control plane |
| `probes.py` | read-only stack probe、fixture executor、结构化 UNAVAILABLE | CPU test/control plane |
| `tiling.py` | 72-byte TilingData ABI、checked arithmetic、UB 模型、候选过滤 | CPU test/control plane |
| `operators.py` | Add/Reduction/RMSNorm 语义、CPU oracle、共同 tolerance | CPU test/control plane |
| `test_vectors.py` | deterministic vectors、case split、guard artifact | CPU test/control plane |
| `sim_device.py` | GM/UB/queue 测试双，始终禁止硬件声明 | simulated/non-claim |
| `experiment.py` | 10 项协议、硬门、采集、报告、身份 hash、旧 run 归档 | M1/M2 orchestration |
| `interface_map.py` | details 中 10×28 步可追踪 | M1 orchestration |
| `scripts/ascend/run_e09.py` | list/probe/map/smoke/collect CLI | Jetson preflight verified |

## 3. 未实现或未验证

下列对象曾被旧版报告错误写成“源码存在/接口就位”，当前工作树并不存在相应模块或真实执行证据：

- C4 `AscendBackend` 数据面；
- torch_npu/aclnn/ATB/MindIE binding；
- current-stream 与 two-phase workspace 实现；
- QuantArtifact/INT8/INT4 actual-kernel 路径；
- Qwen model-core A/B integration；
- msprof timeline/Roofline pipeline；
- CUDA↔Ascend pairability/benchmark executor；
- 真实故障注入、cleanup、恢复和 blind drill；
- C6/C7 Ascend runtime projection。

`ops/ascend/add|reduction|rmsnorm` 中的 `.cpp/.h` 仍是占位源码，未在 CANN 编译器上编译、加载或运行。

## 4. Tiling ABI 的当前含义

现有 `TilingData` 固定为 72 字节、little-endian，包含 schema version、tiling key、dtype tag、block dim、shape、tile、loop/tail、buffer count、行划分、总元素与 workspace。CPU smoke 已验证：

- host serialization/deserialization 一致；
- tail 与 multi-core 行覆盖无 gap/duplicate；
- checked arithmetic、对齐、UB/workspace/block_dim 可结构化拒绝；
- UB 估算把输入、输出、gamma、临时量、队列/对齐开销分项记录。

这些事实只说明 host-side ABI/算法自洽，不说明任一 Ascend C kernel 能解析或执行该 ABI。

## 5. 实际 capability

正式 run `s09_20260921_jetson_preflight_v3` 在 Jetson 上得到：

| Capability domain | 当前状态 |
|---|---|
| dtype/layout/dynamic shape | UNKNOWN |
| custom Ascend C/RMSNorm | UNKNOWN |
| quant matmul | UNKNOWN |
| non-default stream/workspace | UNKNOWN |
| profiler/graph mode | UNKNOWN |
| fallback/power sampling | UNKNOWN |

所有条目 `usable=false`。这不是“该 Ascend 产品明确不支持”，而是当前主机没有 Ascend 设备和目标版本组合，无法产生支持证据。

## 6. 目标数据面契约（待实现）

真实实现至少需要满足：

1. C4 backend 记录 requested/actual backend、actual kernel 和 fallback reason；
2. host tiling 与 device kernel 共享同一 versioned ABI；
3. launch 使用 current device/current stream，无隐藏全局同步；
4. workspace 先 query、后由框架 allocator 分配，并记录生命周期；
5. layout/format conversion 的次数、字节和时间进入 end-to-end 结果；
6. storage/transport/compute precision 分开，低比特声明必须有 actual-kernel 证据；
7. profiler 缺指标时写 `UNAVAILABLE`，不得填 0；
8. 所有故障有 normalized/vendor error、correlation ID、cleanup 和 post-fault probe。

## 7. 当前构建与运行边界

`scripts/ascend/build_ascend_ops.sh` 和 `collect_ascend_env.sh` 会在缺 CANN/`npu-smi` 时明确失败。2026-09-21 的真实探测确认：

```text
npu-smi: not found
/usr/local/Ascend: absent
ccec/bisheng: absent
msprof: absent
torch_npu: absent
Davinci device nodes: absent
```

因此没有执行 Ascend build、kernel、model 或 profiler，所有性能/质量/能耗字段保持未采集。

## 8. 验证与证据

- interface map：10 experiments / 280 steps / 0 failures；
- S09 专项单测：4 passed；
- CPU smoke：`SMOKE_PASS`、`claim_allowed=false`；
- 正式 manifest：`docs/stage_experiments/S09/E09-01/raw/compatibility_manifest.json`；
- 阶段裁决：`docs/reports/S09_阶段验收报告.md`；
- 每项实验报告：`docs/stage_experiments/S09/E09-xx/E09-xx_实验报告.md`。

## 9. 下一步

需要真实 Ascend 节点后，先锁定 SKU/固件/驱动/CANN/framework 版本并完成 E09-01。只有 matching device→compile→load→async run→framework→profile 链通过，才实现和验收 Add/Reduction/RMSNorm、框架回接、模型、profiler、跨硬件比较与故障恢复。
