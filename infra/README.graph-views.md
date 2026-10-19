# 多种计算图表示与真实采集

导出 API 对照 [PyTorch 2.8 ONNX 文档](https://docs.pytorch.org/docs/2.8/onnx.html)，调试产物格式按目标端已安装的 Inductor 实现核对；官方调试示例参见 [Inductor debugging](https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html)。GPU 端文件是否实际产生，以本项目的运行产物为准。

入口：模型架构 → 计算图类型。静态、ATen、FX 原有功能保留；URL `graph=static|aten|fx|onnx|inductor_pre|inductor_post|generated_code` 可直接定位，`runtime_run` 指定历史运行。切换表示只读数据，不加载模型、不提交推理。

模型架构现在按任务分为五个独立子页：`/model-architecture/structure`（模块与权重）、`/semantic`（静态计算结构）、`/execution`（运行算子观测）、`/compiler`（编译与导出）、`/provenance`（来源与凭据）；后四项均在 `/model-architecture` 前缀下。顶部共享导航保留 deployment/runtime_run，非当前工作面不挂载或轮询。旧地址按 graph/runtime_run 自动重定向，无图参数时默认进入模块与权重。

不可用业务操作统一由 `components/ActionButton.tsx` 显示具体原因和前置步骤的链接，支持悬停、点击、键盘聚焦与 Escape 关闭。模型未加载链接到该部署卡片；量化方案不兼容链接到携带 deployment 的精度页；自定义算子链接到相应推理工作台。组件不自动执行前置操作。`npm run check:actions` 检查显式禁用按钮的公共组件归属和解释字段，build/typecheck 一并执行该检查。

## 表示与边界

| 表示 | 实际来源 | 能回答什么 | 不能据此声称什么 |
| --- | --- | --- | --- |
| 静态语义结构 | config、Safetensors、核对过的模型实现 | 模块、参数与数学依赖 | 实际 kernel 或融合 |
| ATen | 前两步实际 TorchDispatch 与 tensor/storage 依赖 | 这次 forward 执行的算子与依赖 | 全输入空间通用图、物理并行 |
| Dynamo FX | 实际 Dynamo backend 的 GraphModule | 编译器输入、模块来源、特化与图断裂 | 已完成物理融合 |
| ONNX | `torch.onnx.export(dynamo=True,optimize=False,fallback=False)` | 固定 batch=1、≤128 输入 tokens、无 KV prefill 的 ONNX 数据流 | 完整动态解码、ONNX Runtime 实测、已部署 ONNX |
| Inductor pre/post IR | 实际编译器 `ir_pre_fusion.txt` / `ir_post_fusion.txt` | 调度单元、显式 buffer 依赖、融合分组 | 设备实际调用次数、Cache 命中、性能收益 |
| 生成代码 | 实际 `output_code.py` | 生成的 Triton/C++/Python 源码 | 代码行等于图节点，或每个定义都执行 |

ONNX 只保存结构，参数保留为图输入；没有复制原始权重。通过 `onnx.checker` 的结构合法性验证，不宣称数值等价。当前不是完整的 ONNX 部署制品。TensorRT 内部图、ONNX Runtime 优化后图尚未接入；不拿 ONNX 输入图冒充其内部优化结果。

## 使用

1. 在精度页加载原生浮点权重与默认 KV。现有量化模型的历史图仍可只读查看。
2. 选择 ONNX 后点击导出诊断按钮；选择任意 Inductor IR/代码视图后点击编译诊断按钮。两类诊断单独执行，固定输出预算为 2 tokens。
3. Inductor 一次记录包含融合前、融合后、生成代码，可选择同一运行切换查看。区分 prefill/decode 特化与首次捕获的 `first_run_id`。
4. 表格选择节点查看局部前驱后继；原始 IR 和代码可展开、下载。大型图会明确标记截断，下载文件自身是否截断另行记录。
5. 性能与质量使用原来的固定样例、独立 PPL 与重复对照；诊断任务禁止当作 `/runtime/comparisons` 的有效性能样本。

## 模块接口与所有权

| 模块 | 输入 → 输出 | 语言 / 所有权 |
| --- | --- | --- |
| `core.graph_views` | 请求与当前能力 → 严格 `GraphCapture` / 格式目录 | Python，纯契约，无 torch |
| `backends.onnx_capture` | 驻留原生模型、原 forward、输入 shape 与目录 → ONNX 结构文件与节点/边 | Python，worker 内显式导出；依赖局部导入 |
| `backends.compiler_evidence` | 本次编译的临时目录 → 有界原始文件与视图条目 | Python，拥有临时文件到封存文件的转换 |
| `backends.graph_formats` | ModelProto / IR 文本 → 节点、边、范围、截断/解析状态 | Python，格式适配，不依赖模型族 |
| `backends.runtime_session` | 请求计划、调用与 graph_capture → 调用凭据、FX 与额外视图 | Python，worker 独占会话、原 forward 恢复、缓存与来源 |
| `console.runtime_routes` | 已终结 run_id → 校验后的只读视图与下载索引 | Python，无 GPU 执行、无 exporter 导入 |
| `CapturedGraph` / `RuntimeExplorer` | 格式与运行选择 → 查询、显式采集、节点与源码展示 | TypeScript/React；沿用统一运行与模型状态 |

扩展新引擎时在格式目录增加表示，独立适配其实际产物，沿用视图条目 `kind/id/status/phase/first_run_id/artifact` 与有界 `nodes/edges/text`。推理后端选择与图表示选择分离；不得通过增加菜单项宣称某后端已执行。

## 保存与资源契约

- 可选请求字段 `graph_capture: {kind: "onnx"|"inductor"}`；必须搭配 `runtime_plan`（分别 eager / inductor）。仅原生浮点、默认 KV、非特权本地 PyTorch worker，`observation_mode=off`，非质量任务，最多 4 输出 tokens。旧请求不受影响。
- `graph-views.json` 关联 run_id、epoch、source_digest、model_identity；复用会话时保留各文件首次运行来源，复制到当前运行自己的目录再封存。
- 新图诊断关闭 Inductor FX 图缓存以取得实际文件；不表示清空所有上游缓存或证明冷启动成本。收集前 4 个后端图，每图最多 4 个编译分区，每种文本文件最多 1 MiB，页面文本最多 48,000 字符；保留截断状态。FX 原有预算不变。
- IR 解析依据 `.snodes[]` 明确的融合组归属，避免把顶格打印的子节点重复计数；读写包括多行 MemoryDep 和 StarDep，未推测 alias/control 依赖。版本不认识时保留文本，标记 `unrecognized_or_empty`。
- ONNX 结构文件最多 32 MiB；视图 JSON 总预算 3 MiB。文件进入既有 experiment manifest，清理确认后发布，记录 SHA256/字节数；读取与下载都验证封存完整性。
- `available`、`not_produced`、`dependency_missing`、`unsupported_input`、`failed` 区分。导出失败不悄悄换 exporter 或替代推理后端；生成是否成功与诊断是否成功分别显示。

## 环境与验证

可选依赖为 `.[graph-export]`，在目标环境安装；torch/CUDA 独立管理，不替换。3090 验证组合：Torch 2.8.0+cu128、Transformers 5.8.0、ONNX 1.19.1、onnxscript 0.3.2、onnx_ir 0.1.12。后两者固定在本次实际验证组合；新版本应重新做导出验收。

远端执行入口：`python -m scripts.console.validate_graph_views --snapshot <原部署状态JSON> --restart`，必须通过项目 remote_run 代理，服务空闲时执行。依次加载 TinyLlama 与 Qwen3-0.6B，检查 ONNX checker、真实 IR 与代码、缓存来源、封存下载索引、采集前后旧单请求输出，并在 finally 恢复原部署及量化制品。证据在 `reports/console/graph-views/`。早期解析问题的记录保留在 `graph-views-before-ir-ownership-fix/`，不能作为修正后解析验收。
