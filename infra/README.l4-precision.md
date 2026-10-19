# L4：表示精度、PTQ 与低比特推理

工作台入口：**优化 → 表示精度与量化**（`/precision`）。历史 `/quantization` 页面和单请求、L8 调度入口继续保留。新入口的方案在模型加载时生效；运行过程中修改表单不会改变已加载模型。

## 方法与执行语义

| 方案 | 校准与权重 | 真实执行路径 | 边界 |
| --- | --- | --- | --- |
| FP32 / FP16 / BF16 | 原始浮点制品 | PyTorch Linear | TF32 默认关闭，独立记录 |
| RTN W4/W8 | 无数据，逐组/逐输出通道对称量化 | 紧凑权重 + Triton 内核内反量化 + FP16 GEMM | 反量化后执行 FP16 GEMM |
| [GPTQ](https://github.com/IST-DASLab/gptq) W4/W8 | 官方 LLM Compressor GPTQModifier，Hessian 校准，阻尼可配 | 同一 canonical packed 执行格式 | act-order 关闭；不导入任意第三方 GPTQ 文件格式 |
| [AWQ](https://github.com/mit-han-lab/llm-awq) W4 | 官方 LLM Compressor AWQModifier，激活感知缩放、网格搜索 | 紧凑 W4A16 Triton | 默认保留 lm_head 浮点 |
| [SmoothQuant](https://github.com/mit-han-lab/smoothquant) W8A8 | 官方 SmoothQuantModifier 平滑，逐输出通道 W8 校准 | 动态逐 token A8，CUDA INT8 GEMM，INT32 累加后缩放 | 单 token decode 需零填充至 GEMM 要求；性能通过相同负载测量 |
| [LLM.int8()](https://huggingface.co/docs/transformers/v5.8.0/quantization/bitsandbytes) | bitsandbytes INT8，离群值阈值可配 | 官方 Linear8bitLt | 离群值混合精度路径由库执行 |
| [NF4 / double quant](https://arxiv.org/abs/2305.14314) | bitsandbytes 非均匀四位存储，scale 二次量化可选 | 官方 Linear4bit | 使用量化权重进行推理 |
| bitsandbytes FP4 | 官方 FP4 编码、可选 double quant | 官方 Linear4bit | 采用 bitsandbytes FP4 编码 |
| FP8 E4M3 | 动态逐张量 scale，紧凑 E4M3 权重 | CUDA `_scaled_mm` | SM89+ 能力门禁；3090 SM86 明确拒绝，尚无 FP8 硬件验收 |
| KV INT2 / INT4 | Transformers QuantizedCache + Quanto | packed KV + 浮点残留，读取时反量化 | 单请求完整 attention；滑动/混合 KV、L8 组合明确拒绝 |

GPTQ、AWQ、SmoothQuant 的算法来源为 [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-algo/)，转换为 HQSB 自有、校验过的部署格式。校准算法和执行内核分别记录，不能把同名算法当成同一种执行性能。当前未实现 QAT、任意现成量化仓库导入、NVFP4/MXFP4、FP8 KV、量化 attention 内核或 CPU/NPU 部署。

混合精度的排除项保留浮点表示。AWQ 0.9 的 `ignore` 同时影响量化与兄弟层匹配，单独排除 Q 投影会使 Q/K/V 映射不完整；HQSB 在隔离校准适配器中仅调整映射解析，让共享输入的浮点兄弟也参与等价缩放补偿，仍由官方算法进行搜索和量化，排除层不会被打包。浮点保留控制存储精度，平滑过程会将等价比例因子折叠进 norm/Linear。

## 使用步骤

1. 先在“模型”页从 ModelScope 下载并注册原始浮点模型；L4 要求原始权重通过既有 manifest 校验。
2. 在 `/precision` 选择模型、计算 dtype、量化算法。RTN/GPTQ/AWQ 目前要求 FP16；量化路径不允许与未验证的自定义 L2 kernel profile 混搭。
3. GPTQ/AWQ/SmoothQuant 填入代表性校准文本，每行一条，与评测文本分开。界面默认文本只用于验证功能，应用校准请使用代表性文本。填写位宽、分组、浮点保留模块等参数。
4. 点击“应用方案并重载模型”，等卸载和加载任务完成。校准在隔离子进程执行；失败会保留错误，可重新加载浮点基线。量化产物不会覆盖源模型。
5. 核对“实际生效与存储”：实际 dtype、被替换模块、实现类、调用数、packed/scale/zero 张量、制品 ID。随后做普通推理，查看 token/s、TTFT 和分配器内存。
6. 分别在浮点基线和候选方案运行“独立质量评测”，保存各自 run ID，再提交质量对照。验收阈值必须在运行前填写，基线与候选一致；不填时只报告测量值，不判定质量通过。
7. 再次加载可选保存的制品 ID，跳过校准。制品绑定源模型、dtype 和方法；哈希/文件集合异常会拒绝加载。权重方案可用于现有单请求和受支持模型的 L8 调度；量化 KV 尚不能用于 L8。

加载、推理或质量任务被接受后，页面会持续轮询到任务终态，期间暂停精度能力探测和手动刷新，保留上次凭据快照；结束后自动读取新凭据。`DEVICE_BUSY` 表示其他任务或性能对照正占用设备，此时显示等待并在空闲后重试，量化制品与质量结果按各自校验状态展示。任务本身是否失败以对应 run 的状态、错误和清理结果为准；其他能力读取错误仍明确展示。加载切换后先刷新部署 epoch，再读取该实例凭据。

## 模块所有权与接口

| 模块 | 输入 → 输出 | 语言/依赖及职责 |
| --- | --- | --- |
| `hqsb/core/precision.py` | JSON → PrecisionPlan / catalogue / 精确排除列表 | Python/Pydantic；纯配置和语义，不初始化 CUDA |
| `hqsb/backends/precision_runtime.py` | 已校验源制品 + plan + dtype → tokenizer/model/receipt | Python；能力门禁、校准子进程所有权、加载、实际绑定和存储证据 |
| `hqsb/backends/precision_calibration.py` | 临时请求文件 → 原子制品或非零退出码 | Python/隔离 LLM Compressor；官方校准配方和转换；进程退出释放 GPU，删除临时明文请求 |
| `hqsb/backends/precision_linear.py` | 激活张量 + packed buffers → 原 shape 输出张量 | Python/PyTorch glue；packed Linear、W8A8、FP8；既有 Triton/CUDA 内核负责设备执行 |
| `ops/quant/w4a16_triton.py` | canonical packed weights + FP16 激活 → GEMM 输出 | Triton；内核内解码，FP32 累加；不保留完整反量化权重缓存 |
| `hqsb/backends/precision_artifacts.py` | 张量/config/tokenizer → hash manifest + safetensors，反向加载 | Python/safetensors；文件完整性、原子发布、路径和容量边界 |
| `hqsb/backends/tensor_memory.py` | 模型张量及已知 vendor 量化状态 → 去重 storage 账本 | Python；统一原有内存图和 L4 凭据的辅助状态枚举，不读张量值或复制数据 |
| `hqsb/console/precision_routes.py` | quality job / run IDs → sealed 对照记录 | Python/FastAPI；复用既有 actor、证据封存及质量计算，不持有 GPU 模型 |
| `web/console/src/pages/Precision.tsx` | 表单与 job 状态 → 加载/推理/评测 API | TypeScript/React；复用模型生命周期切换，不直接访问权重文件 |

模型和紧凑 buffers 归 GPU actor 所有，校准子进程结束后才可发布部署。服务串行化加载与设备任务；制品目录、8 GiB 保存预算及路径由服务端管理，客户端不能指定任意输出目录。不同机器拥有独立环境、制品和证据。

已有 GPTQ/AWQ/SmoothQuant 制品的加载不依赖校准环境；创建新制品时才要求该隔离环境。部署仍需对应的 Triton/CUDA 或 bitsandbytes 执行依赖。L4 当前沿用有界加载任务，校准子进程上限 240 秒、加载上限 300 秒；大模型或大校准集超过此预算会失败，长时离线制品任务尚未扩展。

保存前按紧凑张量的实际字节检查剩余制品预算和磁盘空间，并留文件头/配置与磁盘余量；不能拿原始 FP16 文件大小充当量化文件大小来拒绝部署。当前 RTN、GPTQ、AWQ、SmoothQuant 准备阶段仍要求完整浮点模型及对应工作区能装入目标 GPU；没有实现超出显存的流式离线 PTQ。bitsandbytes 使用其官方加载路径，最终能否部署仍由实际设备和制品校验决定。

## API 与数据关联

### 内置 WikiText-2 Mini 与 PPL

`/precision` 的“质量评测数据集”默认选用 `wikitext2-mini-v1`，旧的自定义短文本 / 全词表 logits / 标准答案评测仍可选择。内置数据来自 WikiText-2 raw 测试集，采用 llama.cpp 使用的 ggml-org 归档镜像，固定归档、来源文本与子集的 SHA256。8 段来自 8 个不同文章区间，在每个区间按固定种子的 SHA256 选择文章与段落；选择不使用模型分数。来源、文章、原始行号和转换规则随数据保存。数据授权与代码的 Apache-2.0 分开，见 `hqsb/core/data/ATTRIBUTION.md`。

每段使用当前 tokenizer 的前 64 个 token，不添加 BOS/EOS 或聊天模板，段间重置上下文，首 token 不计分；不足 64 的样本保留其实际长度。这里的前缀截取是固定评测协议，原始和实际 token 数、输入 token 哈希都存证。当前 Qwen3-1.7B 共计 504 个计分 token。按 `exp(sum(NLL) / sum(scored_tokens))` 计算 token 加权 PPL，不能平均各段的 PPL。此语料使用 NLL / PPL 评分，答案匹配率适用于带标签的任务集。

默认逐 token teacher-forcing，运行实际的已加载权重和 KV 路径，不自由生成回答。另有 API `full_prefill` 模式只用于权重量化检查；已启用量化 KV 时明确拒绝该模式，避免绕过被测缓存。不同模式不能直接对照。计分只保存每位置 NLL、段级统计与身份，不保存全词表 logits，因此本协议不提供 KL / logits 最大误差；旧自定义协议继续提供这些诊断。

主要指标是平均 NLL、PPL、相对基线的平均 NLL 增量和 PPL 倍率。可在基线和候选运行前登记相同的 `ppl_limits.max_mean_nll_increase`；例如 0.1 对应 PPL 倍率不超过约 1.105，此工程阈值可按任务要求选择。未登记时结果为 `measured_no_acceptance_limits`。对照核对模型、tokenizer、源码、Attention、数据哈希、协议、每段实际 token 哈希、计分数与预登记阈值，拒绝混用自定义协议或不同上下文规则。封存和取消沿用现有任务与制品生命周期。

此英文语料子集用于快速量化回归，采用短上下文、跨段重置与固定样本；中文、数学、代码和指令遵循能力请使用相应任务集。PPL 与 lm-eval 的 word perplexity、byte perplexity 不同；不同 tokenizer 的 token PPL 不应直接排名。预训练污染及近似重复校准泄漏尚未检测。

长上下文误差累积、量化 KV residual 缓冲区多次满载后的重压缩行为，可使用独立版本的长上下文协议评测。扩展窗口时同步登记新的 benchmark ID。

职责与依赖：`hqsb/core/corpus_benchmark.py` 是不依赖 torch 的固定数据/协议/计分数学契约；`hqsb/backends/corpus_evaluation.py` 只负责常驻模型、设备执行、取消、局部 KV 生命周期和证据；`precision_routes.py` 负责受控提交与封存比较；页面只选择预设 ID 和显示结果。数据随 Python 包一起安装，日常评测无网络和 datasets/pyarrow 依赖。开发期重建脚本为 `scripts/console/build_wikitext2_mini.py`；目标端验收为 `scripts/console/validate_corpus_benchmark.py`，实际执行仍走远端代理。

```http
GET /api/console/v1/precision/quality-benchmarks
POST /api/console/v1/precision/quality-jobs
{"deployment_id":"your-id","expected_epoch":"current-epoch",
 "quality_plan":{"benchmark_id":"wikitext2-mini-v1"},
 "ppl_limits":{"max_mean_nll_increase":0.1}}
```

语料、窗口和抽样规则由服务器冻结，客户端不能在预设请求中替换样本或修改长度。结果仍通过 run ID 与既有 `/precision/comparisons` 关联；原始统计在 `captures/run_id/quality-result.json`，与执行凭据一起封存。可在页面填写浮点基线和量化候选的 run ID，直接查看 PPL 对照表。

点击独立质量任务的运行记录，默认进入“质量评分与样本”，显示 PPL、平均 NLL、计分数、逐文章评分和逐 Token NLL 曲线；旧记录直接读取已保存的封存数据，不需要重跑。运行列表可单独筛选“独立质量评测”。该视图不套用聊天的 TTFT、Token/s 与文本输出；未采集的 profiler、Cache 命中率和 OS 页级指标会明确说明。执行凭据中的 `after_cleanup_allocator` 只作为清理后 worker 快照显示，其历史累计峰值不能视为本次评测的独立峰值。

### 每模型浮点基线与自动质量变化

独立质量记录、量化页的评测结果和运行列表均自动显示相对浮点基线的变化，不需要手填 baseline / candidate ID。已保存的历史浮点评测可以自动回填；尚无匹配基线的模型必须先用 Floating point、默认 KV 和框架原生算子运行一次相同质量评测，不能凭空补齐分数。

“浮点基线”指未量化权重与默认 KV，明确标注实际 FP16 / BF16 / FP32；基线按实际计算 dtype 标注。FP32 基线须关闭 TF32，且所有基线要求实际零量化模块、无量化制品、生效及清理成功、完整封存评分原件。默认选择最早的兼容有效记录，不按最低 PPL 挑选。基线一旦保存，后续重测不会自动替换；可在合格浮点记录点击“将本次浮点评测设为基线”显式更新，历史选择保留。

基线按模型制品身份、分词器、运行源码、依赖库版本、设备名称 / 架构、Attention、数据版本、计分协议、样本 Token 哈希与计分数分开存储。同一模型的不同代码、数据或环境版本会有各自的基线。元数据位于 `.console/quality-baselines/<context_sha256>.json`，包括主基线 run ID、封存 manifest SHA256、选择方式与替换历史；原始运行和封存文件不会被改写。读取时重新核验原件，损坏时报告不可核验，不静默更换参考。

显示的变化包括：PPL 绝对变化、`(PPL_candidate / PPL_baseline - 1) × 100%`、平均 NLL 增量，以及内置语料的逐段变化。正 PPL 增幅表示退化，其百分比按 PPL 比率计算；有标准答案时单独显示匹配率下降的百分点。没有预登记阈值只报告测量结果；这些是质量基线，性能与显存改善通过独立测量比较。

自动对照及注册接口：`GET /api/console/v1/runs/{run_id}/quality-baseline`、`POST /api/console/v1/precision/quality-baselines`（`{"run_id":"已完成浮点评测的 run ID"}`）。GET 可建立可重建的历史基线索引，不改动模型或原始证据；POST 仅显式选择已核验的浮点评测。原有手动 `/precision/comparisons` 和预登记质量裁决功能继续保留。极端 NLL 时保留 NLL，不可表示的比例返回空值；检测为非有限数后被置空的字段另记入 `numeric_overflow_fields`，不输出非法 Infinity JSON。

以下路径以 `/api/console/v1` 为前缀。变更请求保留现有 `X-HQSB-Client: console` 和 `Idempotency-Key` 契约；免登录开发配置不变。

```http
GET /precision/catalog
GET /deployments/{id}/precision-capabilities
GET /precision/artifacts
POST /deployments/{id}/load
{"precision":"float16","kernel_profile":"framework-native",
 "quantization":{"method":"gptq","bits":4,"group_size":128,
                 "calibration":["your representative calibration text"],
                 "calibration_tokens":128,"exclude_modules":["lm_head"]}}

POST /precision/quality-jobs
{"deployment_id":"your-id","expected_epoch":"current-epoch",
 "quality_plan":{"samples":[{"id":"heldout-1","text":"independent heldout text"}],
                 "max_tokens":64,"generation_tokens":4}}

POST /precision/comparisons
{"baseline_run_id":"run_...","candidate_run_ids":["run_..."]}
GET /precision/comparisons/{l4cmp_id}
```

- `load_run_id → epoch → inference/quality run_id` 绑定实际加载实例；优化 receipt 带模型/tokenizer/源码身份、实际精度、plan hash、量化模块和制品 ID。
- 制品保存于服务 `data_dir/l4-artifacts/{artifact_id}`，manifest 含源模型 hash、配方/校准集指纹、工具版本、文件大小/SHA256。运行配置保存校准集指纹。
- 推理与质量证据继续保存在既有 `captures/run_id`；logits 独立保存并封存。对照记录保存于 `data_dir/precision-comparisons/l4cmp_id.json`。
- 对照拒绝模型/tokenizer/源码/attention 后端、评测协议或预声明阈值不一致，拒绝校准与评测文本的规范化精确重合；目前不做近似重复检测。
- `QualityLimits` 要求填写 `max_logit_abs_error/max_kl/max_nll_increase`；`max_accuracy_drop` 可省略。若声明准确率门槛却没有标签，结果为 inconclusive（已有数值失败仍为 failed），准确率判定等待标签补齐。

## 可测量范围与验收

**直接记录**：实际模块类型/调用数，packed/scale/zero/保留浮点参数的 dtype、shape、去重 storage 字节，源/制品磁盘字节，量化 KV 的 packed 与 residual 张量，CUDA allocator allocated/reserved/peak，既有进程内存、token/s/TTFT，teacher-forcing logits/NLL/PPL、top1/KL及有标签样本准确率。

**指标口径**：分别统计模块 storage、整卡占用和 allocator reserved；位宽比例记录理论压缩率，节省量通过实测对照；调用计数按模块调用统计，Attention 中间量精度按实际 kernel 记录。CPU 页、Cache 命中率与 DRAM 流量由各自采集器提供，并标注采集权限与范围。

验收分开：①真实低比特模块调用；②独立数学参考 CUDA 尾维/bias/INT8 padding 检查；③推理、质量证据封存及可比性门禁；④保存→卸载→加载后相同 greedy token；⑤不支持能力报错后可恢复单请求；⑥支持模型下权重量化与 L8 合批；⑦前端选择/任务/证据可操作。质量达标与性能收益需要代表性数据、预声明阈值、相同负载和重复测量，功能、质量与性能分别记录验收结果。

## 目标环境与验证入口

当前锁定：torch 2.8.0+cu128、Transformers 5.8.0、Triton 3.4.0、bitsandbytes 0.50.2、Quanto 0.2.7。校准环境独立使用 LLM Compressor 0.9.0 / Transformers 4.57.3，共用目标的 torch。安装脚本拒绝其他主版本；换 GPU/Jetson 需独立锁定与验收，不能复制这些二进制或结论。

在 Mac 项目根目录通过远端代理执行：

```bash
./scripts/remote_run.sh bash scripts/setup/install_l4.sh
./scripts/remote_run.sh python -m scripts.console.validate_l4
# 可切换已在本机下载并注册的模型，原始证据另存
./scripts/remote_run.sh python -m scripts.console.validate_l4 \
  --model Qwen/Qwen3-0.6B --methods fp16,rtn4,bnb_nf4 \
  --output reports/console/l4/qwen
```

验证使用独立控制端口/数据目录，先检查主服务空闲，再释放模型，结束恢复原模型。原始 run JSON、环境 freeze 和日志在被 Git 忽略的 `reports/console/l4`。小样本只是明确的功能夹具；不要据其 PPL 或一次 token/s 选择生产方案。
