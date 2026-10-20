# L10 服务与应用优化

## 入口与使用

进入工作台 **服务与应用优化**：`/application/run?deployment=qwen-reference`。
页面按任务划分为 **策略与负载、目标与副本、结果与机制** 三个子页；运行目录中的 application 记录也会跳转到结果子页。保留已有单请求推理、L1–L9 和测试环境免登录。

1. 在设备与部署页加载模型，默认选择“使用已有驻留模型”。加载/卸载/切换链接会携带部署 ID。
2. 使用默认四条典型输入比较 A/B：A 原生逐请求计算，B 开启精确结果缓存与缓存感知路由。也可以载入典型多轮聊天。
3. 分别修改 A/B 策略；请求 JSON 可设置 prompt **或** messages、tenant、arrival_ms、max_output_tokens。应用 JSON 时先调用纯校验接口，非法数据不会覆盖有效页面状态。
4. 点击“运行 L10 服务对照”。根任务持有设备租约，A/B 依次执行，内部根据并发上限向不同空闲节点派发请求。关闭页面不会取消任务。
5. 在结果页比较完成率、服务 E2E、服务交付 Token/s、实际模型调用/计算 Token、输入变化和内存证据；展开每个请求查看真实输出、Token ID、缓存来源及后端子记录。完整 JSON 经制品封存校验后读取。

托管模型生命周期实验：先卸载对应设备上的驻留模型，再选择“实验托管 / 加载和回收”。A 固定预加载托管副本，B 可开启需求加载/空闲回收。模型按该节点部署默认配置加载，不继承其他设备的精度或编译产物；实验只回收自己加载并核对过 epoch/load run ID 的模型。已有副本不会被自动卸载。

多节点：先在 **机器与连接** 登记当前控制服务可访问的 SSH 隧道或 HTTP/HTTPS HQSB 节点，再在“目标与副本”编辑 targets。凭据来自控制服务机器，不读取 Mac 的 SSH 配置。所有参与端都必须安装支持 `l10-v1` 的 HQSB 服务。只接受登记节点，不接受请求附带任意 URL。

## 功能与边界

| 策略 | 实际行为 | 比较时注意 |
| --- | --- | --- |
| round_robin | 在可用目标间轮换 | 模型、Tokenizer、精度、量化、Kernel 与运行时身份须一致 |
| least_pending | 一个目标只允许一条活动推理；优先空闲，按最久未使用打破平局 | 不等同于节点内 continuous batching；L8 保持独立 |
| cache_affinity | 优先本方案中存在精确结果缓存的目标 | 缓存的是 CPU 输出结果，不是 KV；不假装预测 KV 命中 |
| length_model | 按输入字符数阈值选择 small/large 角色 | 需要不同节点的两种角色；这是字符阈值，不是 Token 成本预测；换模型需要独立质量评测 |
| result_cache | 租户/输入/策略/实际模型与 Tokenizer/运行时/epoch 隔离的精确 LRU，限制 TTL、条目和序列化字节 | 各方案冷启动，无跨租户命中，不支持语义答案缓存；命中不会重复调用 GPU |
| prefix_reuse | 调用 L6 经后端验证支持的 KV 前缀复用 | 每方案临时独立 PrefixStore，带租户命名空间；结束清理临时 KV 并恢复用户原有前缀缓存 |
| head_tail_tokens | 使用真实 Tokenizer 保留头尾 Token，重编码确认不超过上限 | 仅 raw prompt；有损，可能改变语义，不是学习式压缩 |
| recent_messages | 保留 system 消息与最近 N 条非 system 消息 | 仅聊天；N 是消息数，不是完整对话轮数；可能删除历史上下文 |
| output_cap | 实际输出上限取请求、策略、部署上限和剩余上下文的最小值 | 记录 requested/effective；更少输出不等于相同质量下更快 |
| autoscale | 在已登记目标中按需求加载托管模型，并按空闲超时回收，遵守 min/max | 不创建 VM、Pod 或物理机器，不是 Kubernetes HPA |

限额：1–4 个目标，每个物理节点至多一个目标；1–32 个请求，并发 1–8；总输入字符不超过 65536。节点别名指向同一物理节点时拒绝实验。单节点仍串行提交推理，网关并发不代表 GPU batch size。多节点路由与身份隔离的逻辑测试不能替代真实多 GPU 性能验收。

本次不是生产 OpenAI 流式网关，不提供 Agent/tool-call 自动改写、语义缓存、云资源申请或完整多租户安全系统；测试环境免登录保持不变。对改写输入、减少输出、更换模型的方案，输出 Token 对齐只是差异证据，应继续使用现有独立质量评测；不把字符串一致率冒充任务准确率。

## 模块、接口与语言

所有新增协调逻辑为 Python；前端为 TypeScript/React。没有新增 CUDA Kernel，实际计算继续使用已选 L2/L4 后端，因此不为网络、JSON 与调度逻辑引入 C++/Rust。

| 模块 | 单一职责 | 输入 → 输出 / 接口 |
| --- | --- | --- |
| `core/application_plan.py` | 纯配置契约与跨字段约束，不导入设备运行时 | JSON → ApplicationPlan/Policy/Input/Target；非法类型与策略组合拒绝 |
| `serving/application_cache.py` | 线程安全、有限容量、独立方案的精确结果 LRU | `result_key(item, policy, snapshot, fixed)`；get/put/contains/summary，不操作 GPU |
| `serving/application_engine.py` | 请求到达、目标占用、路由与副本状态机、成对方案统计 | Plan + Endpoint 映射 + cancel 回调 + event 回调 → variants 与 compare_variants；不认识 HTTP、SSH、Actor 或凭据 |
| `backends/application_input.py` | 驻留真实 Tokenizer 的输入处理与运行身份；L6 私有前缀作用域 | metadata(provider)、prepare(provider,{input,policy})、prefix_scope(provider,owner,begin) → 身份/有效输入 Token/预算/作用域凭据；不执行模型 forward |
| `console/application_transport.py` | 将 Endpoint 协议映射为既有 Console 生命周期或已登记远端 | snapshot/prepare/generate/load/unload/begin_prefix/end_prefix/acquire/release；返回实际子 Run 与清理确认 |
| `console/application_jobs.py` | 根任务准入、设备租约、线程生命周期、配额、取消与制品封存 | start(plan,key) → application Run；read(run_id) → 经 manifest/hash 与 Store 对照核验的 receipt |
| `console/application_routes.py` | HTTP 类型校验与窄参与端协议 | 下表 API；不直接操作模型或接受任意 worker 命令 |
| `features/application/*`、`pages/Application.tsx` | 策略编辑、目标配置、对照证据呈现 | 生成的 OpenAPI Plan 类型 + REST → 子页；复用 ActionButton 的禁用说明 |

Endpoint 边界规定：`snapshot()` 返回 state/epoch/load_run_id/identity；`prepare(input,policy)` 返回 payload、真实输入 Token 和有效输出预算；`generate(payload,key,deadline,cancelled)` 返回完成后的后端 Run；`load(key)` / `unload(epoch,key,load_run_id)` 必须有生命周期清理确认。模型加载/卸载等待不持有路由条件锁，其他已就绪节点可继续服务。

复用组件：现有 MachineRegistry/ConnectionManager、ConsoleService/Actor/Store、Pydantic、L6 PrefixStore、封存制品模块。Store 增加 created 与 kind/created 索引，L10 轮询读取轻量状态投影，避免列表排序与大 JSON 读取阻塞记录线程。

## API

统一前缀 `/api/console/v1`，沿用现有 Console 客户端头与幂等约定。

| 方法与路径 | 用途 |
| --- | --- |
| GET `/application/catalog` | 协议版本、默认计划、策略、限制、租约与隔离状态 |
| GET `/application/targets` | 当前控制端登记的节点及本地部署列表 |
| POST `/application/validate` | `{plan: ...}` → 归一化 Plan；纯校验，无加载和执行 |
| POST `/application/jobs` | `{plan: ...}`，Idempotency-Key → 202 根 Run |
| GET `/application/jobs` | L10 历史投影，不携带整份制品 |
| GET `/runs/{id}/application` | 终态根任务的完整、核验后 receipt |
| POST `/requests/{id}/cancel` | 复用现有取消接口，等待子任务与托管模型清理 |
| POST `/application/leases/acquire`、`release` | 参与节点租约，owner=`l10:run_<32hex>`，target 为声明目标 |
| POST `/application/worker` | 白名单 snapshot/metadata/prepare/prefix_begin/prefix_end；要求该 owner 持有租约 |
| POST `/application/operations` | load/unload/generate，稳定 Idempotency-Key；类型与卸载所有权检查 |
| GET `/application/operations/{id}/status` | 小体积状态与清理确认，终态再取完整子记录 |

最小请求示例（浏览器页面可直接生成，不需要手填全部默认字段）：

```json
{
  "plan": {
    "targets": [{"id":"primary","machine_id":"local","deployment_id":"qwen-reference","lifecycle":"existing"}],
    "requests": [
      {"id":"a","prompt":"Once upon a time","max_output_tokens":16},
      {"id":"b","prompt":"Once upon a time","max_output_tokens":16}
    ],
    "baseline": {"result_cache":false},
    "candidate": {"routing":"cache_affinity","result_cache":true},
    "fixed_output_length":true
  }
}
```

## 观测、关联与保存

根 `application` run ID → 冻结计划 → variant → request ID/tenant/target → backend.submitting 的稳定幂等键 → backend.accepted 的实际子 Run ID → 原有完整推理、加载、卸载及机制证据。缓存命中保存来源子 Run，但 `backend_invoked=false`，后端生成速度为“未调用”，不复制成一次新 GPU 执行。

SQLite 保留根状态、事件、摘要和 receipt；`<console data_dir>/captures/<root-run-id>/` 保留 `application-receipt.json`、`execution-receipt.json` 与封存 manifest。读取时核对 hash、root ID 与数据库内容；失败/取消也封存 partial，不丢弃已完成请求与控制事件。前端运行目录只加载摘要投影，按需取完整记录。

| 可直接记录 | 定义/边界 |
| --- | --- |
| P50/P95/P99 服务 E2E、排队、请求/s | 协调端 monotonic；E2E 含输入处理、网络与后端状态查询等待，不是纯 GPU 时间；小样本分位数只作描述 |
| startup_ms、workload_ms、variant_total_ms、job_total_ms | 分别显示初始加载、负载窗口、含回收的方案总时间、整体实验时间，候选按需冷加载计入实际请求 E2E |
| served/computed output tokens、backend_invocations | 交付输出量、实际调用产生的输出量与调用次数分开；服务交付 Token/s 包含缓存，禁止当 GPU 吞吐量 |
| 输入 Token 前/后、effective output budget、输入 hash、lossy | 真实 Tokenizer 编码，不用字符估算 Token；输出预算同时受上下文限制 |
| result cache hits/misses/evictions/entries/bytes | CPU JSON 序列化缓存大小，不等于 Python RSS 或显存 |
| 后端生成速度、分配器峰值、L6 复用 Token 与机制 | 来自实际子运行；未采集为缺失，不填零；峰值不等于整机或多个节点同时显存总和 |
| 输出 Token ID 对齐、模型/输入身份差异 | 固定负载一致性证据；不能替代任务质量、困惑度或独立 benchmark |

暂不由 L10 直接测量：路由导致的 GPU L2/LLC 硬件 Cache 命中率、跨节点统一时钟的微秒级通信分解、逐页物理页迁移因果关系、云端资源费用、真实用户满意度。可通过已有 L1–L9 支持的采集能力关联子 Run；无硬件/权限/数据的字段保持明确缺失。

## 取消与恢复

任务取消后停止新准入，向已接受子请求发送取消并等待确认；托管副本核对 epoch/load run ID 后卸载，临时 KV 作用域恢复，最后释放节点租约。服务关闭顺序先停止 L10，再关闭连接与 ConsoleService。

远端 POST 超时只用同一幂等键恢复原提交，不自动转发到第二台机器；不能确认执行/取消/释放时隔离该节点与根任务，不报告清理成功。控制端崩溃后远端租约和模型可能保留，目前需要人工恢复：

1. 读取根记录 backend.submitting/backend.accepted，确定 owner、目标、稳定提交键与子 ID。
2. 在参与节点检查该 owner 的子运行，必要时按同一提交键恢复身份，再取消并确认终态与清理。不能取消其他 owner 的请求。
3. 对照加载子 Run、epoch 与所有权处理本实验托管副本，确认 pending=0。不得卸载用户原有模型或删除记录以绕过检查。
4. 确认后调用该节点 `/application/leases/release`，body 为原 owner 与原 target；若控制服务已隔离，在所有参与节点确认清理后重启控制服务。租约不自动过期，不能把进程失联推断成 GPU 已清理。

## 验证

全部通过项目代理在 3090-inf 执行；Mac 仅编辑、同步和读取已保存证据。

```bash
./scripts/remote_run.sh pytest tests/unit/console -q
./scripts/remote_run.sh python -m scripts.console.export_schema
./scripts/remote_run.sh bash -c 'cd web/console && npm run api:generate && npm test && npm run build'
./scripts/remote_run.sh python -m scripts.console.validate_l10 --managed
./scripts/remote_run.sh bash -c 'cd web/console && HQSB_CONSOLE_URL=http://127.0.0.1:8768 npx playwright test tests/application.spec.ts tests/no-auth.spec.ts'
```

实机脚本要求空闲服务与一个已有 ready PyTorch 模型，运行 Qwen/Gemma 固定验收，并在 finally 恢复原驻留配置与普通单请求验证。输出 `reports/console/l10-application/summary.json` 与逐项原始 receipt；具体本次结果见 `docs/reports/2026-10-19_L10_console_acceptance.md`。
