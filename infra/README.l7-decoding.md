# L7 解码与生成算法

入口：优化 → 解码与生成算法。三个可独立寻址的子页共享 deployment/run 身份：

- `/decoding/run`：固定典型输入、原始 prompt、策略参数、真实流式输出和执行凭据。
- `/decoding/draft`：查看独立草稿的来源、参数字节、冷加载时间，显式卸载草稿。
- `/decoding/compare`：选择两条封存运行，核对输入、模型、Token、成本与可比性。

原 `/playground` 保留原有单请求贪心生成。只有显式传入 `decoding_plan` 才进入本层新执行器；其他层原有路径不被替换。原工作台提供 L7 入口，运行详情增加独立的“L7 解码与验证”页签。免登录保持节点原配置。

## 已实现的实际算法

| 方法 | 实际工作 | 可作出的结论 |
| --- | --- | --- |
| 自回归 greedy | 逐 Token 原生 Forward + FP32 logits 选择 | 当前输入可和原交互路径逐 Token 对照 |
| 自回归 multinomial | 请求独立 GPU Generator；重复惩罚 → 最少长度/EOS/有限候选 → 温度 → Top-k → Top-p → torch.multinomial | 同实现/设备/模型下检验 Seed 重现；不能从一次同 Seed 输出证明分布或任务质量 |
| prompt_lookup | 在已提交历史中搜索重复后缀 n-gram，提出后续 Token；目标模型一次 Forward 验证多个候选 | 不加载草稿权重；无匹配时正常执行单 Token 目标验证；收益取决于重复结构和接受率 |
| assistant | 同 Tokenizer 的独立草稿模型逐步提出候选；目标模型批量验证 | 只开放 greedy；必须计入额外模型/KV、首轮加载及长期驻留成本 |
| 有限候选约束 | 将明确的 Token 序列构成 trie，逐步允许合法分支与终点 EOS；可切换全词表 Mask / 候选 Gather | 验证 Token 路径是否命中注册候选；不是通用 JSON Schema、正则或任意语法支持 |

Top-k 使用阈值过滤，并列值可能使有效集合大于 k；Top-p 在排序累计概率超过阈值时保留跨越阈值的第一个 Token。重复惩罚按正负 logits 分别除/乘系数，作用于输入与已生成历史中的去重 Token。约束 Gather 仍需完整 LM Head logits，不声称省掉全词表模型输出计算。

## 推测正确性与 KV 不变量

目标先完成 prefill。每轮验证候选最长匹配前缀，遇拒绝即提交目标的纠正 Token；全接受时可提交目标的 bonus Token。第一轮使用已知的 prefill logits，后续把上一轮尚未送入模型的最后一个已提交 Token 与本轮候选一起验证，避免每轮再增加一次单 Token Forward。

提交一轮后，目标 KV 长度必须等于全部已提交输入/输出历史长度减 1；最后一个已提交 Token 留待下一轮输入。拒绝、EOS、输出尾部截断时先裁掉超前状态，再输出确认 Token。不会把未验证的草稿 Token 发给前端。

草稿拥有自己的 DynamicCache 和已消费 Token 历史。下轮先计算与目标已提交历史的公共前缀，裁掉错误未来，再补齐纠正 Token 并继续提案。目标和草稿的请求 KV 不共享，不跨请求复用。取消和超时停止提案/验证/输出，finally 释放两个请求缓存；未完整完成的记录不标记 applied。

首批要求单序列、原生浮点模型、framework-native、默认动态 KV。滑动窗口缓存只有在输入加预算严格低于窗口长度时允许回滚；不会在已丢弃历史状态后假装支持 crop。无量化 KV、L3 编译/图采集、L5 结构制品、特殊 L6 缓存或 L8 合批组合。跨层组合需要单独验收，不能因各层单独可用便推断组合已验证。

模型的批量验证和逐 Token Forward 在浮点舍入上可能不同，尤其接近并列最大 logits；因此逐 Token 一致性是实际检查，不以理论 greedy 保证掩盖数值分歧。当前只实现贪心前缀验证，没有实现随机推测的拒绝采样/残差分布校正。

## 模块、接口、语言和所有权

| 模块 | 语言 / 所有权 | 输入 → 输出 |
| --- | --- | --- |
| `hqsb/core/decoding_plan.py` | Python/Pydantic，无 torch | 原始计划与当前部署能力 → 严格计划、公开 hash 配方、目录；拒绝未使用或不兼容配置 |
| `decoding_sampling.py` | Python/PyTorch GPU 原语；每请求 Generator/trie | logits、已提交历史与采样计划 → 合法 Token；不加载模型或操作 HTTP |
| `decoding_engine.py` | Python/PyTorch；请求目标/草稿 KV 与验证状态 | 模型、输入 Token、计划、取消回调 → 已确认 Token 组与真实 Forward/接受/回滚统计 |
| `decoding_draft.py` | Python；目标 serial worker 持有一个辅助模型 | 管理员登记配置 → 校验过的驻留草稿；metadata/release 管理权重生命周期，不拥有请求 KV |
| `decoding_execution.py` | Python；一次请求的格式、流式 I/O、时间与清理 | 请求、目标 provider、取消事件 → output/result 事件、观测及执行凭据 |
| `console/decoding_routes.py` | Python/FastAPI；查询、显式释放、封存证据比较 | HTTP → 目录/运行/草稿状态；两条原件 → 严格对照和持久 JSON |
| `features/decoding` 与 `pages/Decoding.tsx` | TypeScript/React；草稿和 URL/查询状态 | 实际 API → 参数、输出、驻留资源与对照，复用 ActionButton/Token 预算/运行轮询 |

本阶段不新增 C++/CUDA Kernel，张量选择和模型计算复用目标机 PyTorch/CUDA 实现。Python 负责采样规则、前缀与 KV 状态；只有发现实际瓶颈且有数值/性能证据时才替换设备实现。

## 草稿模型生命周期与资源

1. `draft_deployment_id` 只能指向管理员已登记、有 manifest 的本地 PyTorch 模型，客户端不能传任意路径。服务端补充私有配置，不把路径复制到请求公开 config。
2. 第一条明确选择 assistant 的请求触发加载；校验完整源 manifest、实际 Tokenizer 编码/ID 映射和词表尺寸。不同 Tokenizer 明确失败，没有隐式翻译或 fallback。
3. 草稿在 CPU 暂存后完整放到同一 GPU，无自动 CPU offload。只接受未量化的 safetensors 源；读取张量头，按目标 dtype 的元素字节计算计划参数量 P，而非直接把磁盘文件量 W 当成 FP32/FP16 加载量。预检 W 不超过 6 GiB、GPU 空闲量至少 P + 2 GiB、有效主机/cgroup 可用量至少 P + max(W,P) + 1 GiB。凭据同时保留计划与实际参数字节；这是保守估算，不是任何形状都不会 OOM 的保证。
4. 草稿权重由目标 worker 持有，可跨请求复用；`draft_loaded_this_request` 区分冷加载/复用。请求 KV 均独立清理。不得把仍然驻留的草稿参数当成“释放失败”或省略额外显存成本。
5. 更换草稿前显式卸载。草稿卸载是可追溯的 `draft_unload` 生命周期任务，必须在目标就绪、epoch 匹配、设备空闲时提交。释放不卸载目标、不改变目标 epoch。目标卸载或 worker 关闭时也释放草稿。
6. 当前节点清理后仅保留 Qwen3 1.7B 与 Gemma3 1B IT。本次同检查点独立副本用于验证正确性和驻留清理；未下载或恢复用户已经清理的模型，未声称获得小草稿模型的加速收益。后续可在 ModelScope 页面接入相同 Tokenizer 的较小草稿，再做对应负载验收。

取消是协作式检查：提案/验证/输出之间和草稿加载的几个边界可响应。若取消落在不能中断的 native 加载内部，监督进程仍沿用既有 worker 超时终止机制，可能连同目标 worker 结束；此时应显示 `worker_terminated` 并重新加载目标，不能宣称目标始终就绪。本轮真实取消覆盖生成阶段，未声称覆盖所有加载中断位置。

## 配置与 API

请求沿用 `POST /api/console/v1/requests`，增加可选 `decoding_plan`：

```json
{"method":"prompt_lookup","draft_tokens":4,"lookup_min_ngram":2,"lookup_max_ngram":6,"lookup_window":4096}
```

普通采样例如：

```json
{"method":"autoregressive","sampling":"multinomial","temperature":0.8,"top_k":40,"top_p":0.95,"repetition_penalty":1.1,"min_new_tokens":2,"seed":42}
```

assistant 需 `method=assistant` 与 `draft_deployment_id`。有限候选需 `choices` 文本数组，`constraint_backend=mask|gather`；仅用于原始 prompt、非固定输出长度，保证最长候选加 EOS 能放入预算。每候选 1–64 Token、最多 16 条，候选不得包含 EOS，最少生成长度不得超过最短候选。公开 config/配方只保存候选文本哈希，原文不混入凭据；生成输出仍属于原有运行输出数据。

- `GET /decoding/catalog`：策略、限制和已登记草稿候选，不推断 Tokenizer 一定匹配。
- `GET /decoding/status?deployment_id=...`：串行 worker 元数据读取，不执行模型 Forward。
- `POST /decoding/draft/unload`：deployment_id + expected_epoch + Idempotency-Key，返回持久任务。
- `GET /decoding/runs?deployment_id=...`：当前模型的最近 L7 记录。
- `POST /decoding/comparisons`：baseline_run_id / candidate_run_id。
- 取消继续沿用 `POST /requests/{run_id}/cancel`，并保留原 SSE/事件与输出快照。

输出预算 1–512、每轮候选 1–16；查找历史最多 8192 Token，逐轮详情最多 256 条并显式标记截断，总计数继续完整累积。不能在 greedy 模式传温度/Top-k/Top-p 非默认值并假装生效。

## 观测与质量边界

直接测量：实际 Token IDs、目标/草稿 Forward 数与输入长度、候选/接受/丢弃数、验证批次、回滚 Token 数、生成循环 Token/s/首 Token/总时间、含冷加载与清理的 worker 请求总时间、allocator allocated/reserved/peak、实际目标/草稿 KV inventory、草稿参数字节及来源、阶段时间与基础 OS 样本。

`decoding-receipt.json` 与主 execution receipt 一并由 supervisor 封存，清理 ACK 之后才 complete。L7 对照核对封存哈希、applied 状态、模型、Tokenizer、运行源码、精度、Attention、设备、输入 Token hash、预算、实际长度、固定长度与观测口径、采样和约束语义。严格贪心对照还要求实际 Token 序列一致；不一致时给出首个分歧并阻止合格比较。结果保存到 `.console/l7-comparisons/`。

同一验证批中的多个 Token 可在相同确认时间到达，ITL=0 只表示批量提交，不能解释成对应 Token 的计算耗时为零。没有采集的 Kernel GPU 时间、Cache 命中和物理页迁移不填成 0。额外草稿权重、独立 KV、Python 查找/采样与选择同步都有成本。

质量需要分清：

- greedy 推测的执行一致性：逐 Token 与原始串行基线比较，仅支持本次样例结论。
- 有限候选：合法终点校验不等于分类准确率。
- 采样：Seed 重现不是分布等价或任务质量；需多 Seed、任务数据和预登记阈值。
- Teacher-forcing PPL：衡量权重模型的语言概率，不经过这里的采样/验证策略，不能冒充 L7 策略质量指标。

没有自推测早退、随机推测拒绝采样、不同 Tokenizer 辅助、Medusa/EAGLE/MTP 训练头或树验证、Beam 多分支 KV、通用语法编译和自定义融合采样 Kernel。单次更快的结果只是描述性观测；需要同负载交错重复、多模型/多提示类型与资源/质量约束才能主张正式加速。

## 验证与来源

所有运行经代理在 3090-inf：

```bash
./scripts/remote_run.sh env OMP_NUM_THREADS=2 pytest -q tests/unit/console/test_decoding.py tests/hardware/test_decoding_cuda.py
./scripts/remote_run.sh python -m scripts.console.validate_l7
./scripts/remote_run.sh bash -c 'cd web/console && HQSB_CONSOLE_URL=http://127.0.0.1:8768 npx playwright test decoding.spec.ts'
```

验证脚本保存当前真实驻留配方并在 finally 恢复；不新增下载或复活已清理模型。具体运行 ID、数值、失败排查和限制见内部 `reports/console/l7-decoding/report.md`。

语义参考：Transformers [generation strategies](https://huggingface.co/docs/transformers/generation_strategies)、[assisted generation](https://huggingface.co/blog/assisted-generation)；缓存裁剪以目标机实际安装的 Transformers 5.8.0 代码为准。这里只实现表中明示的子集，不把整个上游能力计入本项目。
