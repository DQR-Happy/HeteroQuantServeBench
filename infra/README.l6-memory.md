# L6：真实内存与 KV 管理

入口：**优化 → 内存与 KV 管理**（`/memory-cache/configure`、`/memory-cache/records`）。多请求块池在 **评测 → 多请求调度实验**。保留单请求工作台、L3/L4/L8 及免登录；不传 `memory_plan` 的请求沿用原模型缓存行为。

## 已接通的执行路径

| 策略 | 真正执行的操作 | 边界 |
| --- | --- | --- |
| model_default | 保留 L4 `new_cache`，包括既有 Quanto KV 精度 | 量化 KV 不与其他 L6 布局组合 |
| dynamic | Transformers DynamicCache，按生成增长 | 根据 config 保留滑窗/混合层语义 |
| static | StaticCache，惰性逐层预分配固定容量 | 默认输入 Token + 输出预算 − 1；显式容量不足会正常拒绝，模型仍可用 |
| offloaded_dynamic / offloaded_static | 上游 Cache 的逐层 CPU 卸载与 GPU 异步预取 | 继承上游 stream 顺序；请求释放前同步所有 CUDA streams；搬运可能降低速度 |
| recompute | `use_cache=False`，每步输入完整历史 | 无驻留请求 KV，增加处理位置和计算量 |
| 前缀复用 | GPU 不可变前缀快照、最长匹配、命中后复制到请求私有 KV、LRU 淘汰 | 仅 dynamic + full attention；至少保留一个 prompt Token 重新执行；不缓存 logits；不是零复制共享 |
| L8 block_pool | 实际 GPU slab、请求独立块表、空闲块复用、每步仅写入新增 K/V | Attention 前还原成稠密 KV，保留并统计 gather 成本；**不是 PagedAttention kernel** |
| L8 recompute 抢占 | 保留原按步引擎：完成设备操作后释放 KV，再回放历史 | 调度策略、租约及原 dense 路径不变 |

L6 不与编译、计算图采集、独立 teacher-forcing 质量任务混为一个样本。不支持的组合在控制面拒绝；混合滑窗的前缀复用在真正分配前拒绝，不静默回退。已验证软件版本为目标机 PyTorch 2.8.0+cu128、Transformers 5.8.0；其他版本必须重新验收。

## 模块、输入输出和接口

| 所有者 / 开发语言 | 高内聚职责 | 输入 → 输出 / 接口 |
| --- | --- | --- |
| `hqsb/core/memory_plan.py` / Python | 请求契约、组合约束、目录；不导入 torch | `MemoryPlan` → 规范化计划；`validate_memory()`、`catalogue()` |
| `hqsb/backends/memory_session.py` / Python | 单请求缓存构造、模式语义、stream 完成与实测凭据 | provider + 冻结 request；`prepare(token_ids, limit)` → cache、复用长度；`after_forward()` → 进度；`finish(success)` → receipt |
| `hqsb/backends/prefix_store.py` / Python + PyTorch Tensor | 一个驻留 worker 内的不可变 payload、匹配、容量与 LRU | 模型/配置/分词/精度/源码/namespace identity + 实际 Token；`lookup` → 私有 clone；`publish` → 拷贝字节；`trim/clear` → 释放 |
| `hqsb/backends/block_kv.py` / Python + PyTorch CUDA copy | L8 请求块存储；不决定谁先执行 | `refusal(ids, starts, additions)` → 容量/过期拒绝原因；`pack` → DynamicCache + 复制量；`commit` → 新增写回量；`release/clear/receipt` |
| `step_executor.py` / Python | 原有 mask/position/forward/选择/完成边界，选择 DenseKV 或 BlockKV | `Step[]` → `StepResult`，含逐次 forward 的存储/搬运事实；`prepare` 先检查临时与驻留空间 |
| `LiveEngine` / Python | 原有准入、独立逻辑租约、策略、抢占与取消 | `Sequence`/调度 limits → steps、进度事件；不依赖具体块表或张量布局 |
| `memory_routes.py` / Python | 串行 worker 控制、历史、封存读取、比较；不导入 CUDA | HTTP → 规范化请求、已校验凭据与比较结果 |
| `memory_comparison.py` / Python | 可比性与数值/性能的描述性差异 | 两条运行 + 已封存 receipts → 拒绝原因、同输入生成检查与指标对齐 |
| `web/console/src/features/memory-cache` / TypeScript/React | 配置、存储表和比较视图 | API 返回 → 实际状态/指标/链接；不推导实验 PASS/FAIL |

内存策略和所有权用 Python 表达，实际张量分配、copy、Attention 由现有 PyTorch/Transformers CUDA 执行。当前没有新增 C++/CUDA 内核；只有需要避免稠密还原、直接读块表的 PagedAttention 才需要新增并独立验收设备 kernel。

## 接口

- `GET /api/console/v1/memory/catalog`：静态目录，不执行模型。
- `GET /memory/capabilities?deployment_id=...`：在空闲串行 worker 检查实际驻留模型。忙时返回明确 409；不在前端轮询中反复探测。
- `POST /memory/prefix/clear`：`deployment_id`、`expected_epoch`；设备完成后显式清空前缀；不替用户加载模型。
- `POST /requests`：保留原字段，新增 `memory_plan` 和可选 `fixed_output_length`。
- `GET /memory/runs?deployment_id=...`：最近 60 个 L6 记录。
- `GET /runs/{id}/memory-cache`：只读 SHA 校验与 supervisor 确认后的 `memory-cache.json`。
- `POST /memory/comparisons`：`baseline_run_id`、`candidate_run_id`；只读比较，不启动模型。
- `POST /scheduling-benchmarks` 的 `limits` 增加 `kv_layout: dense|block_pool`、`kv_block_tokens`（1–256）、`kv_pool_max_bytes`（1 MiB–2 GiB）。旧负载默认 dense。

`MemoryPlan`：`mode`、`max_cache_tokens`（仅静态）、`prefix_reuse`、`prefix_budget_bytes`（最多 512 MiB）、`prefix_max_entries`（最多 64）、`prefix_max_tokens`（最多 8192）、`prefix_namespace`。namespace 用于逻辑隔离，不是身份认证边界。前缀属于单个模型 worker；卸载会清空，不跨重载、设备、精度或源码复用。

## 关联、保存与观测口径

- 冻结计划进入 SQLite run config，绑定 deployment、epoch、输入 Token SHA、模型/分词器/源码与精度 identity。
- `memory-cache.json` 与 `execution-receipt.json` 保存相同 receipt；清理确认后写入不可变 `experiment-manifest.json`，带 SHA-256。读取会检查 manifest、run/epoch 与数据库/执行凭据的一致性。旧记录不会用当前状态补写。
- L8 继续使用 EvidenceWriter：每次 `forward` 记录块池、复制与完成证据，`session.result` 汇总峰值、gather/scatter、尾部浪费和回收结果，沿用已有 records 查询。

| 可以直接测 / 记录 | 定义与不能外推的部分 |
| --- | --- |
| Cache 类、use_cache、forward 次数、处理输入位置 | 控制流与实际执行对象；重计算位置数是操作量，不是 FLOPs |
| K/V 与量化 wrapper 的 tensor leaves、shape、dtype、device、storage 字节去重 | 请求释放前快照；量化 code/scale leaves 也计入；不含 Attention 临时工作空间 |
| CUDA allocator allocated/reserved/peak | 整个 worker allocator；前缀发布后的统计覆盖 clone 成本；不是整机显存或物理页分布 |
| 前缀 hit/miss、实际复用 Token、clone 字节、保留字节、淘汰数 | Token 内容比较并有实际 payload；不是 GPU L1/L2 命中率；全长前缀至少重算最后一个 prompt Token |
| D2H/H2D 字节与调用次数 | 上游 Cache API 实际变换设备的 tensor 字节；不是 PCIe 硬件计数或精确带宽 |
| 实际 slab 字节、空闲/活跃块、尾部浪费、回收与复用 | 逻辑块映射 + 真实 GPU 张量；不是 CUDA/UVM/操作系统页表 |
| worker minor/major faults | 请求期间进程累计增量；不做张量或 GPU 页因果归因；现有 pages 诊断仍可独立开启 |
| Token/s、首 Token、ITL、Worker 总时间 | 生成循环不含前缀还原/发布和清理；Worker 总时间包含这些成本。Cache API 传输包含在生成执行内 |
| 生成 Token、一致性、所选 Token logprob 差 | 固定输入正确性检查；不是模型全局质量、PPL 或通用准确率 |

**估算**：L8 几何 `kv_token_bytes` 与临时 workspace/headroom 是保守准入估计；不能作为实测 tensor storage。逻辑租约是预算，不是已分配显存。

**暂时没有**：真正的 paged-attention GPU kernel、跨请求零复制共享页面、混合滑窗前缀、L8 CPU offload/KV 量化组合、通用 SSD 分层迁移、GPU L2 hit rate、CUDA/UVM 逐页迁移和物理地址映射；也不把一次生成对照标成正式速度或质量提升。

## 使用与验收

1. 设备页加载模型；进入 L6 选择模型，检查能力。
2. 选固定输入/输出，先跑 dynamic 基线，再跑 static、offloaded 或 recompute。
3. full attention 模型勾选前缀，连续运行两次；确认 miss → hit、处理位置减少、生成 Token 一致；清空后再测冷缓存。
4. 在运行对照选同 epoch 的两条记录，核对输入、输出预算、精度、源码及观测口径。不同条件不输出直接性能比较。
5. L8 选择块池运行多请求，查看真实 copy、尾部浪费和 cleanup；稠密还原成本必须保留。

验收包含：六模式数值一致性；offload/prefetch 非零真实搬运；前缀私有 ownership、LRU/namespace、取消不发布；静态容量不足正常拒绝且后续请求可继续；块池乱序 pack、仅新增 scatter、块复用和完成后释放；busy/stale epoch 与封存防篡改；旧单请求、L4/L3/L8 回归；前端真实提交、禁用原因与响应式子页。

执行遵守项目代理，测试和构建只在 3090-inf。可重复入口：

```bash
./scripts/remote_run.sh env OMP_NUM_THREADS=2 pytest -q tests/unit/console/test_memory_management.py tests/hardware/test_memory_cache_cuda.py
./scripts/remote_run.sh python -m scripts.console.validate_l6
```

第二条要求服务空闲，先保存当前模型的完整配方，执行后在 finally 恢复；不下载或改写权重。实测结果及浏览器截图见 `reports/console/l6-memory/`。测试数字和具体 run ID 见该目录的 `report.md`，不能把短输入测试当作长上下文或所有模型的证明。
