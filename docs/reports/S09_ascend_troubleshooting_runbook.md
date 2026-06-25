# S09 Ascend Backend Troubleshooting Runbook（故障排查手册）

> 阶段：S09  
> 性质：诊断链与运维指南  
> 生成时间：2026-09-18

---

## 1. 使用说明

本手册面向**一线工程师**，提供从用户错误到根因定位的标准化诊断链。每条故障都有：

1. **错误码**：normalized error class
2. **诊断步骤**：按顺序执行的检查项
3. **预期输出**：每个步骤应看到的结果
4. **升级条件**：何时需要运维介入或重启进程

**禁止**：
- 看到错误先重启机器
- 静默 fallback 到 CPU/PyTorch
- 忽略 run_id 直接查日志

---

## 2. 快速诊断链（5 步法）

```text
Step 1: 读取 run_id、normalized/vendor error、first failure time
Step 2: 核对 E09-01 manifest 和 capability，不先重装
Step 3: 确认 artifact/model/spec/quant hash 和 SoC/ABI
Step 4: 检查 requested/actual backend、fallback、shape/dtype/layout
Step 5: 检查 host TilingData、block_dim、workspace 和 checked validation
Step 6: 对齐 stream/task/kernel 与 async error
Step 7: 查看内存/健康前后变化
Step 8: 运行最小 device/Add/RMSNorm probe
Step 9: 决定 request fail、process restart、node quarantine 或运维升级
Step 10: 保存证据，不用"重启后好了"替代根因
```

---

## 3. 故障分类与处理流程

### 3.1 ENV_MISMATCH（环境错配）

**场景**：driver/firmware/CANN/framework 版本组合不符合官方兼容矩阵

**诊断步骤**：

```bash
# Step 1: 读取 manifest
cat docs/evidence/ascend_env.json | jq '.ascend_stack'

# Expected output:
# {
#   "cann_version": "8.0.RC1",
#   "toolkit_root": "/usr/local/Ascend"
# }

# Step 2: 核对 official_sources
grep -A 5 "official_sources" docs/stage_experiments/S09/E09-01/*.md

# Step 3: 检查当前安装
npu-smi info -t version

# Step 4: 对比差异
# If mismatch found:
echo "ERROR: ENV_MISMATCH detected"
echo "Expected: CANN 8.0.RC1 + driver X.Y.Z"
echo "Actual: CANN A.B.C + driver D.E.F"
```

**处理**：
1. 如果 minor 版本差异（如 8.0.RC1 vs 8.0.RC2），尝试重新加载驱动模块
2. 如果 major 版本差异，必须回滚到兼容版本
3. 记录 field-level diff（哪些字段变了）
4. 拒绝执行直到匹配

**升级条件**：无法回滚 → node quarantine，通知运维升级 CANN 工具链

---

### 3.2 UNSUPPORTED（能力不支持）

**场景**：请求的 dtype/layout/op/shape 不在 capability table 中

**诊断步骤**：

```python
from hqsb.ascend.capability import CapabilityTable

table = CapabilityTable.load_from("configs/ascend/capability_table.json")
decision = table.decide("dtype.int4.compute")

print(decision.allowed)  # False
print(decision.reason)   # "kernel_bits_unsupported"
print(decision.fallback_suggested)  # "fp16" or "none"
```

**处理**：
1. 检查 `reason_code` 是否在预注册的错误目录中
2. 如果是 `UNSUPPORTED` 且允许 fallback，选择 next-best implementation
3. 如果是 strict mode，立即失败（fail-fast）
4. 记录 actual_backend ≠ requested_backend + reason

**升级条件**：频繁 UNSUPPORTED 表明 capability table 过时 → 更新 E09-01 probes

---

### 3.3 ARTIFACT_INCOMPATIBLE（制品不兼容）

**场景**：编译的 kernel artifact 的 SoC target 与实际设备不符

**诊断步骤**：

```bash
# Step 1: 读取 artifact metadata
cat artifacts/rmsnorm_v1.meta | jq '.soc_target'

# Expected: "910b"

# Step 2: 读取设备 SoC
npu-smi info -t device -i 0 | jq '.chip_sku'

# Expected: "Ascend 910B"

# Step 3: 对比
if [ "$artifact_soc" != "$device_sku" ]; then
    echo "ERROR: ARTIFACT_INCOMPATIBLE"
    echo "Artifact compiled for: $artifact_soc"
    echo "Device is: $device_sku"
fi
```

**处理**：
1. 拒绝加载该 artifact
2. 触发重新编译（如果源可用）
3. 记录 source_hash → artifact_hash 映射失效
4. 检查 build cache 是否混用不同 SoC 的制品

**升级条件**：build cache 污染 → 清理缓存并重新构建所有 kernels

---

### 3.4 COMPILE_ERROR（编译错误）

**场景**：Ascend C 源码编译失败

**诊断步骤**：

```bash
# Step 1: 读取编译日志
cat logs/compile_rmsnorm.log | tail -50

# Typical errors:
# - syntax error in kernel source
# - UB budget exceeded (host tiling bug)
# - unsupported instruction for target SoC
# - missing header file

# Step 2: 验证 source hash
git rev-parse HEAD > source_commit.txt
cat artifacts/source_hash.txt

# Step 3: 清缓存重编
rm -rf build/ascend-cache/*
scripts/ascend/build_ascend_ops.sh -B build/ascend-910b --soc=910b --no-cache
```

**处理**：
1. 解析编译器 stderr，提取 normalized error class
2. 如果是语法错误 → 修复源码
3. 如果是 UB 预算不足 → 调整 tiling 参数
4. 如果是不支持指令 → 降级到更低 precision 或 simpler kernel

**升级条件**：多次编译失败 → 检查 CANN 工具链完整性

---

### 3.5 INVALID_ARGUMENT（无效参数）

**场景**：host tiling 检查失败（overflow/block_dim/workspace/tail）

**诊断步骤**：

```python
from hqsb.ascend.tiling import compute_tiling, TilingRequest, UbBudget, TilingLimits

ub = UbBudget(nominal_bytes=192*1024, framework_reserved_bytes=8192, ...)
limits = TilingLimits(max_block_dim=48, max_workspace_bytes=1<<20, alignment_elems=8, ub_budget=ub, ...)

req = TilingRequest(rows=1000000, hidden=4096, dtype='fp16', block_dim=48, tile_elems=32, ub_model='rmsnorm')
decision, rejection = compute_tiling(req, limits)

if rejection:
    print(rejection.reason_code)  # REJECT_OVERFLOW / REJECT_BLOCK_DIM / REJECT_WORKSPACE
    print(rejection.detail)       # "total_elements=4096000000 exceeds 32-bit limit"
```

**处理**：
1. 根据 reason_code 采取对应措施：
   - `REJECT_OVERFLOW`：拆分 batch 或使用 64-bit index path
   - `REJECT_BLOCK_DIM`：降低 block_dim 或拒绝请求
   - `REJECT_WORKSPACE`：减少 buffer_count 或拒绝请求
   - `REJECT_ALIGNMENT`：调整 tile_elems 为 alignment_elems 倍数
2. 记录 requested shape → rejection reason
3. 向调用方返回 structured error（不是 panic）

**升级条件**：频繁 invalid argument → 检查 workload spec 是否超出设备能力

---

### 3.6 WORKSPACE_ERROR / DEVICE_OOM（内存错误）

**场景**：UB/GM 分配失败或 OOM

**诊断步骤**：

```bash
# Step 1: 检查 allocator 状态
npu-smi info -t memory -i 0

# Step 2: 查看前后快照
diff before_oom.json after_oom.json

# Step 3: 分析 leak
python3 scripts/ascend/check_resource_leaks.py --run-id <run_id>

# Expected output:
# "Leaked: 256MB in ub_allocations (tag: rms_x_c0_s1)"
```

**处理**：
1. 如果是 UB overrun → 调整 tiling（减小 tile_elems 或 buffer_count）
2. 如果是 GM OOM → 减少 batch size 或启用 gradient checkpointing
3. 检查是否有 leak（still_live tags 非空）
4. 执行 cleanup：free all allocations, reset context

**升级条件**：OOM 后无法恢复 → process restart；频繁 OOM → node quota adjustment

---

### 3.7 DEVICE_EXECUTION_ERROR（设备执行错误）

**场景**：kernel launch 成功但同步时暴露设备错误

**诊断步骤**：

```python
# Step 1: 捕获 async error
try:
    result = dev.launch(kernel, tiling_payload, inputs=...)
except DeviceMemoryFault as exc:
    print(exc.details)  # {"buffer": "y", "index": 12345, ...}

# Step 2: 关联原 op/stream/task
trace = trace_collector.chain_for(run_id, task_id)
print(trace.span_chain)  # ["gateway", "runtime", "kernel_rmsnorm", "stream_0"]

# Step 3: 检查 stream 状态
aclrtGetLastErrorCode(stream_id)
```

**处理**：
1. 验证 correlation ID（run/request/op/stream/task/error 五元组）
2. 检查是否 async error 被错误归到后续 op
3. 如果 kernel 崩溃 → 重置 context 和 stream
4. 记录 vendor error code + normalized class

**升级条件**：同一 kernel 反复失败 → device quarantine，硬件诊断

---

### 3.8 TIMEOUT / HANG（超时/挂起）

**场景**：watchdog 触发或进程无响应

**诊断步骤**：

```bash
# Step 1: 检查 watchdog 事件
cat logs/watchdog_events.jsonl | grep -A 5 "<run_id>"

# Step 2: 查看 kernel 状态
npu-smi info -t task -i 0 | grep "<run_id>"

# Step 3: 检查 thermal/throttle
npu-smi info -t health -i 0 | grep -E "temperature|frequency"

# Step 4: 手动 kill 并 dump
kill -SIGQUIT <process_id>
gdb -p <pid>  # attach and dump stack
```

**处理**：
1. 区分 true hang（kernel stuck）vs slow execution（within budget but long）
2. 如果是 thermal throttle → 降频或等待冷却
3. 如果是 kernel deadlock → 重置 device
4. 记录 MTTD（Mean Time To Detect）和 MTTR（Mean Time To Recover）

**升级条件**：频繁 timeout → 调整 watchdog threshold 或优化 kernel

---

### 3.9 PROFILER_ERROR（Profiler 错误）

**场景**：msprof 启动失败或指标 unavailable

**诊断步骤**：

```bash
# Step 1: 检查 profiler 可用性
which msprof
msprof --version

# Step 2: 测试采集
msprof --output /tmp/test_profile --aic-metrics PipeUtilization \
       --command "python3 run_small_kernel.py"

# Step 3: 解析结果
msprof --convert /tmp/test_profile.prof /tmp/test_profile.json
jq '.metric_set' /tmp/test_profile.json
```

**处理**：
1. 如果 msprof 不可用 → 标记 profiler_observed = UNAVAILABLE（不是填 0）
2. 如果 metric set unsupported → 降级到基础 metrics
3. 如果文件解析失败 → 保留原始 .prof 文件供人工分析

**升级条件**：完全无法 profiling → 依赖 Nsight-like 外部工具或接受 M3 降级

---

### 3.10 Fallback 相关故障

**场景**：fallback 策略执行错误或静默降级

**诊断步骤**：

```python
from hqsb.ascend.framework import AscendBackend

backend = AscendBackend(...)
result = backend.generate(workload, inputs)

print(result.actual_backend)  # "ascend_rmsnorm_v1" or "pytorch_reference"
print(result.fallback_reason) # "dtype_int4_unsupported" or ""
print(result.coverage)        # 0.85 (15% fell back)
```

**处理**：
1. 检查 fallback 是否合法（同一 OperatorSpec + tolerance）
2. 如果是静默 fallback（actual≠requested 但 reason=""）→ BUG
3. 如果是非法 fallback（quality 下降）→ 改为 fail-fast
4. 记录 fallback count 和 coverage

**升级条件**：fallback coverage > 50% → 重新评估 capability table

---

## 4. 资源审计清单

每次故障后必须审计以下资源：

| 资源类型 | 检查项 | 正常值 | 异常信号 |
|---|---|---|---|
| **GM Allocation** | allocated_bytes ≤ capacity_bytes | True | leaked ≥ 10MB |
| **UB Usage** | high_water_bytes ≤ usable_bytes | True | still_live tags non-empty |
| **Queue Depth** | enque_count == deque_count | True | blocked_enque > 0 |
| **Stream State** | all streams drained | True | stream alive after sync |
| **Context** | device health = green | True | temperature > 80°C |
| **FD Count** | fd count stable | Δfd ≈ 0 | fd leak > 10 |
| **Process Memory** | RSS stable | ΔRSS ≈ 0 | RSS growth > 100MB/hr |

---

## 5. 恢复级别定义

| 级别 | 操作 | MTTR | 适用场景 |
|---|---|---|---|
| **L1: Request Fail** | 拒绝当前请求，不影响其他 | < 1ms | INVALID_ARGUMENT, UNSUPPORTED |
| **L2: Process Restart** | 重启当前进程，保留其他 | ~1s | DEVICE_OOM, DEVICE_EXECUTION_ERROR |
| **L3: Node Quarantine** | 隔离节点，迁移流量 | ~1min | ENV_MISMATCH, frequent timeouts |
| **L4: Hardware Reset** | 重置 device，需运维介入 | ~5min | persistent execution errors |
| **L5: Full Reinstall** | 重装 CANN 工具链 | ~30min | compile errors, artifact incompatibility |

---

## 6. 常见误诊案例

### 案例 1：把 runtime gap 当成 kernel 慢

**症状**：kernel latency 10us，但 end-to-end 50us

**误诊**："kernel 太慢，需要优化"

**真相**：30us 是 format conversion + workspace allocation + dispatcher overhead

**正确诊断**：
```python
accounting = result.accounting
print(f"kernel_time: {accounting['gm']['read_bytes'] + accounting['gm']['write_bytes']}")
print(f"overhead: {50 - 10}")
```

### 案例 2：把 profiler 扰动当成真实 slowdown

**症状**：开启 profiler 后 latency 增加 200%

**误诊**："profiler 指标显示 bottleneck 在 X"

**真相**：profiler-on 本身引入大量额外开销

**正确诊断**：
```python
clean_latency = measure_without_profiler()
profiled_latency = measure_with_profiler()
print(f"overhead: {(profiled_latency - clean_latency) / clean_latency * 100}%")
```

### 案例 3：把偶发抖动当成系统问题

**症状**：P99 latency 偶尔达到 100ms

**误诊**："系统不稳定，需要扩容"

**真相**：thermal throttle 或共享负载干扰

**正确诊断**：
```bash
# Check thermal history
npu-smi info -t health -i 0 --history 1h | grep temperature

# Check other processes
ps aux | grep npu
```

---

## 7. 证据保存规范

每次故障必须保存：

1. **run_id**：唯一标识符
2. **error_code**：normalized + vendor raw
3. **stack_trace**：Python call stack + device kernel trace
4. **resource_snapshot**：before/after GM/UB/queue/state
5. **manifest_hash**：兼容性 manifest SHA256
6. **artifact_hash**：kernel artifact SHA256
7. **command**：实际执行的命令
8. **stdout/stderr**：完整输出
9. **profiler_raw**：原始 .prof 文件（如果可用）
10. **health_probe**：故障前后的 device health

**禁止**：只保存结论，不保存 raw evidence

---

## 8. 升级路径

| 故障类型 | 一线工程师 | 运维团队 | 厂商支持 |
|---|---|---|---|
| ENV_MISMATCH | ✅ 回滚版本 | ⚠️ 协助升级 | ❌ |
| UNSUPPORTED | ✅ 调整请求 | ❌ | ❌ |
| ARTIFACT_INCOMPATIBLE | ✅ 重新编译 | ⚠️ 清理缓存 | ❌ |
| COMPILE_ERROR | ✅ 修复源码 | ⚠️ 工具链检查 | ✅ (if toolchain bug) |
| INVALID_ARGUMENT | ✅ 调整 tiling | ❌ | ❌ |
| WORKSPACE_ERROR | ✅ 调整 batch | ⚠️ 配额调整 | ❌ |
| DEVICE_EXECUTION_ERROR | ✅ 重置 context | ⚠️ 硬件诊断 | ✅ |
| TIMEOUT | ✅ 调整 threshold | ⚠️ 负载调度 | ❌ |
| PROFILER_ERROR | ✅ 降级模式 | ❌ | ✅ (if tool bug) |

---

## 9. 附录：错误码速查表

| Error Class | Vendor Code Example | Reason | Fix |
|---|---|---|---|
| ENV_MISMATCH | ACL_ERROR_VERSION_MISMATCH | driver/CANN version mismatch | Rollback to compatible version |
| UNSUPPORTED | ACL_ERROR_NOT_SUPPORTED | dtype/int4 not supported | Use fp16 or fall back |
| ARTIFACT_INCOMPATIBLE | ACL_ERROR_ARTIFACT_VERSION | SoC target mismatch | Recompile for correct SoC |
| COMPILE_ERROR | ACL_ERROR_COMPILE_FAIL | Syntax/UB error | Fix source or tiling |
| INVALID_ARGUMENT | ACL_ERROR_INVALID_PARAM | overflow/block_dim | Adjust request params |
| WORKSPACE_ERROR | ACL_ERROR_MEM_ALLOC_FAILED | UB/GM OOM | Reduce batch/buffer_count |
| DEVICE_OOM | ACL_ERROR_MEM_OUT_OF_MEMORY | GM exhaustion | Enable checkpointing |
| DEVICE_EXECUTION_ERROR | ACL_ERROR_LAUNCH_FAILED | Kernel crash | Reset context |
| TIMEOUT | ACL_ERROR_TIMEOUT | Watchdog triggered | Optimize kernel or increase timeout |
| PROFILER_ERROR | ACL_ERROR_PROFILER_INIT | msprof unavailable | Use fallback metrics |

---

*生成时间：2026-09-18*  
*版本：draft v0.1*
