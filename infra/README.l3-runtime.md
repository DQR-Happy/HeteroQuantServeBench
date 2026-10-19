# L3：计算图观测、执行策略与架构联动

多种图类型选择、ONNX 导出、Inductor 融合前后 IR 与生成代码见 [计算图表示与采集](README.graph-views.md)。

入口：环境 → 模型架构 → **计算图与 L3 执行**。运行详情中的同名按钮跳转到对应的历史图。现有制品结构、ATen 动态依赖、单请求、量化与 L8 路径保留；未传 `runtime_plan` 的调用继续使用原执行路径。

## 可执行范围

每个推理/独立质量请求可选：

- `eager`：原模型 forward，L3 基线；ATen 依赖另用原有“采集运行架构”。
- `dynamo_eager`：真实 TorchDynamo 捕获 FX 图，返回原 FX forward；**不进行 Inductor lowering，也不宣称加速**。
- `inductor`：实际调用 TorchInductor，进行上游编译优化；显式关闭 CUDA Graph。
- `shape_policy=auto/static/dynamic` 对应 `dynamic=None/False/True`；`fullgraph=true` 遇不支持的图断裂直接失败。

首批执行范围为原生浮点权重、框架原生 L2、默认 KV。编译与量化 KV、量化模块、自定义 L2 算子、特权 worker 组合明确拒绝。编译请求必须 `observation_mode=off`，不得同时打开 ATen 架构 hooks/profiler，以免探针改变 Dynamo 捕获行为。量化模型仍可用 eager 与原有观测。

L8 不接收该请求参数，仍沿用原始 forward，不会被先前单请求编译策略污染。当前目标为 Torch 2.8.0+cu128 / Transformers 5.8.0 / RTX 3090；其他组合需重新验证。

## 使用

1. 在表示精度页选 Floating point、默认 KV、framework-native 并加载模型。
2. 模型架构页选 Eager，运行固定样例，记录基线；再选 Dynamo 或 Inductor、形状策略，执行相同输入预算。
3. 同一编译方案再执行一次，观察 `new_graphs` 是否归零以及已捕获图是否仍被调用。切换方案清除该 worker 的会话编译缓存；不删除上游磁盘缓存。首次样本包含追踪/编译成本，不能冒充稳态性能。

请求限时读取 `/runtime/catalog` 的节点限制。若较大模型冷编译超过原有请求限时，应由管理员调整该节点 `request_deadline_ms`（最多 600000 ms）；接口不会绕过服务器限制。
4. 点击“独立 PPL 评测所选策略”，执行已有 WikiText-2 mini 固定协议。每种方案单独评分，以 eager 作为质量基线；编译结果不能注册为原始浮点质量基线。
5. 选历史记录、子图、算子，查看源码、shape、dtype、模块路径和关联权重；选择同模型同输入的基线记录进行比较。比较接口拒绝不同源码、权重、tokenizer、精度、attention、输入、输出预算或质量协议。单次运行仅给描述性指标，不生成加速采纳结论。

固定生成输入按聊天/续写模式各有一个，均保留实际输入 token 摘要。PPL 使用独立固定文本，保留逐样本和逐 token NLL；性能和质量任务不混用。

## 模块与接口

后端均为 Python，界面为 TypeScript/React；各模块按状态和数据来源划分。

| 模块 | 输入 | 输出 / 状态所有权 |
| --- | --- | --- |
| `core.runtime_plan` | 请求方案、已加载精度/KV/算子/探针组合 | 有限、禁止额外字段的执行契约；组合验证；无 torch 导入 |
| `backends.runtime_session` | 常驻 nn.Module、方案、请求身份 | 临时 forward 绑定、单个缓存编译会话、实际图调用/失败计数；退出恢复原绑定 |
| `backends.runtime_graph` | 真实 Dynamo GraphModule、已加载 module 路径集合 | 有界 FX 节点/边/元数据、源码、图断裂、模块与算子分布；不读 tensor 值 |
| 既有 `optimization_receipt` / `experiments.artifacts` | 执行结束/清理状态、L3 会话 | `runtime_receipt`、`runtime-graph.json`；沿用 supervisor 封存、SHA256/长度校验 |
| `console.runtime_routes` | 部署/运行 ID、对照双方 ID | 只读历史/图、严格身份对照；GET 不触发推理、不加载 GPU |
| `architecture/RuntimeExplorer` | 既有架构制品与模块、历史图、执行配置 | 提交固定输入/独立 PPL、筛选/邻域图、关联权重、对照和 JSON 查看 |

原有 `hqsb.compiler` 的 S11 研究契约保留；在线适配器调用上游 Dynamo/Inductor，不把实验矩阵或注入式 mock 后端当作真实执行实现。

```text
GET  /runtime/catalog
POST /requests                      + runtime_plan
POST /precision/quality-jobs         + runtime_plan
GET  /runtime/runs?deployment_id=...
GET  /runs/{id}/runtime-graph
POST /runtime/comparisons            {baseline_run_id, candidate_run_id}
```

请求示例：`runtime_plan={"mode":"inductor","shape_policy":"auto","fullgraph":false}`。

## 观测含义与边界

**直接测量**：实际 forward/成功次数、后端回调次数、新捕获图、图 callable 调用次数/图 ID、后端编译 host 时间、forward host 时间、实际输入/mask shape、FX 节点/依赖、FakeTensor shape/stride/dtype、可获得的模块栈/源码、Dynamo 图断裂事件。与原有 token/s、内存 allocator、PPL 同 run ID 关联。

**需谨慎解释**：`new_graphs` 是新编译回调，不是精确重编译数；一个 forward 可包含多个图。0 新图不是磁盘/硬件 Cache 命中证明；图调用大于 0 不代表全模型编译覆盖。动态图和历史特化均可能存在；每图记录 `first_run_id`，`used_graph_ids` 标识当前请求调用的图。模块未映射时明确留空，不按模型族猜测。符号维度保留字符串，不强制具体化。

**未直接测量**：生成 kernel 的融合次数/运行次数、真实设备依赖/重叠、编译内存规划、硬件 Cache/TLB/页命中率、通用精确重编译原因与完整 guards、优化前后等价性形式化证明。FX 是 lowering 前的图，不能据此推导这些硬件事实。CUDA Graph、手工节点编辑/重排、跨 L2/L4/L8 的编译组合暂不开放。

图记录最多 16 个特化、每图 4096 节点/20000 边，最终 JSON 约 3 MiB 预算，截断显式标记且不留悬空边。记录预算不改变实际执行；触发编译器重编译上限会报错，禁止静默降回 eager。`fullgraph=false` 的正常 eager islands 属于显式策略，不是完整编译覆盖。

异常、取消或超时保留失败记录并按既有 worker 生命周期清理；编译本身不可逐 token 抢占，长编译可能导致 deadline 后终止 worker，需要重新加载。GET 历史图只接受终态、封存且完整性通过的文件；重载后会标明历史 epoch。

依据：[PyTorch compile API](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html)、[自定义后端契约](https://docs.pytorch.org/docs/2.8/torch.compiler_custom_backends.html)。版本敏感实现另以目标机安装代码及测试核对。

验收脚本 `scripts/console/validate_l3.py`：空闲设备，备份原方案，实际权重 eager/Dynamo/Inductor 各两次生成与独立 PPL，对照原模型同条件 NLL（预设绝对均值差 ≤0.01），保留失败/原始图，finally 恢复原量化制品和 KV。该阈值仅是本次小基准的回归门，不等价于通用质量无损。
