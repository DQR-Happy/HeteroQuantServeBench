# L4：表示精度、PTQ 与低比特推理

工作台入口：**优化 → 表示精度与量化**（`/precision`）。历史 `/quantization` 页面和单请求、L8 调度入口继续保留。新入口的方案在模型加载时生效；运行过程中修改表单不会改变已加载模型。

## 方法与执行语义

| 方案 | 校准与权重 | 真实执行路径 | 边界 |
| --- | --- | --- | --- |
| FP32 / FP16 / BF16 | 原始浮点制品 | PyTorch Linear | TF32 默认关闭，独立记录 |
| RTN W4/W8 | 无数据，逐组/逐输出通道对称量化 | 紧凑权重 + Triton 内核内反量化 + FP16 GEMM | 不宣称原生 INT4 Tensor Core 乘法 |
| [GPTQ](https://github.com/IST-DASLab/gptq) W4/W8 | 官方 LLM Compressor GPTQModifier，Hessian 校准，阻尼可配 | 同一 canonical packed 执行格式 | act-order 关闭；不导入任意第三方 GPTQ 文件格式 |
| [AWQ](https://github.com/mit-han-lab/llm-awq) W4 | 官方 LLM Compressor AWQModifier，激活感知缩放、网格搜索 | 紧凑 W4A16 Triton | 默认保留 lm_head 浮点 |
| [SmoothQuant](https://github.com/mit-han-lab/smoothquant) W8A8 | 官方 SmoothQuantModifier 平滑，逐输出通道 W8 校准 | 动态逐 token A8，CUDA INT8 GEMM，INT32 累加后缩放 | 单 token decode 需零填充至 GEMM 要求；不保证更快 |
| [LLM.int8()](https://huggingface.co/docs/transformers/v5.8.0/quantization/bitsandbytes) | bitsandbytes INT8，离群值阈值可配 | 官方 Linear8bitLt | 离群值混合精度路径由库执行 |
| [NF4 / double quant](https://arxiv.org/abs/2305.14314) | bitsandbytes 非均匀四位存储，scale 二次量化可选 | 官方 Linear4bit | 只做推理，不等于实现 QLoRA 训练 |
| bitsandbytes FP4 | 官方 FP4 编码、可选 double quant | 官方 Linear4bit | 不等于 NVIDIA NVFP4 |
| FP8 E4M3 | 动态逐张量 scale，紧凑 E4M3 权重 | CUDA `_scaled_mm` | SM89+ 能力门禁；3090 SM86 明确拒绝，尚无 FP8 硬件验收 |
| KV INT2 / INT4 | Transformers QuantizedCache + Quanto | packed KV + 浮点残留，读取时反量化 | 单请求完整 attention；滑动/混合 KV、L8 组合明确拒绝 |

GPTQ、AWQ、SmoothQuant 的算法来源为 [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-algo/)，转换为 HQSB 自有、校验过的部署格式。校准算法和执行内核分别记录，不能把同名算法当成同一种执行性能。当前未实现 QAT、任意现成量化仓库导入、NVFP4/MXFP4、FP8 KV、量化 attention 内核或 CPU/NPU 部署。

混合精度的排除项保留浮点表示。AWQ 0.9 的 `ignore` 同时影响量化与兄弟层匹配，单独排除 Q 投影会使 Q/K/V 映射不完整；HQSB 在隔离校准适配器中仅调整映射解析，让共享输入的浮点兄弟也参与等价缩放补偿，仍由官方算法进行搜索和量化，排除层不会被打包。浮点保留不表示权重字节始终不变：平滑可将等价比例因子折叠进 norm/Linear。

## 使用步骤

1. 先在“模型”页从 ModelScope 下载并注册原始浮点模型；L4 要求原始权重通过既有 manifest 校验。
2. 在 `/precision` 选择模型、计算 dtype、量化算法。RTN/GPTQ/AWQ 目前要求 FP16；量化路径不允许与未验证的自定义 L2 kernel profile 混搭。
3. GPTQ/AWQ/SmoothQuant 填入代表性校准文本，每行一条，与评测文本分开。界面默认文本只用于验证功能，不是推荐生产校准集。填写位宽、分组、浮点保留模块等参数。
4. 点击“应用方案并重载模型”，等卸载和加载任务完成。校准在隔离子进程执行；失败会保留错误，可重新加载浮点基线。量化产物不会覆盖源模型。
5. 核对“实际生效与存储”：实际 dtype、被替换模块、实现类、调用数、packed/scale/zero 张量、制品 ID。随后做普通推理，查看 token/s、TTFT 和分配器内存。
6. 分别在浮点基线和候选方案运行“独立质量评测”，保存各自 run ID，再提交质量对照。验收阈值必须在运行前填写，基线与候选一致；不填时只报告测量值，不判定质量通过。
7. 再次加载可选保存的制品 ID，跳过校准。制品绑定源模型、dtype 和方法；哈希/文件集合异常会拒绝加载。权重方案可用于现有单请求和受支持模型的 L8 调度；量化 KV 尚不能用于 L8。

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
- 制品保存于服务 `data_dir/l4-artifacts/{artifact_id}`，manifest 含源模型 hash、配方/校准集指纹、工具版本、文件大小/SHA256。不把校准原文写进运行配置。
- 推理与质量证据继续保存在既有 `captures/run_id`；logits 独立保存并封存。对照记录保存于 `data_dir/precision-comparisons/l4cmp_id.json`。
- 对照拒绝模型/tokenizer/源码/attention 后端、评测协议或预声明阈值不一致，拒绝校准与评测文本的规范化精确重合；目前不做近似重复检测。
- `QualityLimits` 要求填写 `max_logit_abs_error/max_kl/max_nll_increase`；`max_accuracy_drop` 可省略。若声明准确率门槛却没有标签，结果为 inconclusive（已有数值失败仍为 failed），不伪造准确率通过。

## 可测量范围与验收

**直接记录**：实际模块类型/调用数，packed/scale/zero/保留浮点参数的 dtype、shape、去重 storage 字节，源/制品磁盘字节，量化 KV 的 packed 与 residual 张量，CUDA allocator allocated/reserved/peak，既有进程内存、token/s/TTFT，teacher-forcing logits/NLL/PPL、top1/KL及有标签样本准确率。

**不能混同**：模块 storage 不是整卡占用；allocator reserved 不是有效权重；位宽比例是理论压缩率，不是实测节省；调用计数不是 GPU 指令追踪；KV 压缩不意味着 attention 中间量也以 INT4 计算。CPU 页、cache 命中率与 DRAM 流量仍取决于已有采集器/权限，L4 不据位宽反推出这些指标。

验收分开：①真实低比特模块调用；②独立数学参考 CUDA 尾维/bias/INT8 padding 检查；③推理、质量证据封存及可比性门禁；④保存→卸载→加载后相同 greedy token；⑤不支持能力报错后可恢复单请求；⑥支持模型下权重量化与 L8 合批；⑦前端选择/任务/证据可操作。质量达标与性能收益需要代表性数据、预声明阈值、相同负载和重复测量，功能闭环通过不代表这两项自动通过。

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
