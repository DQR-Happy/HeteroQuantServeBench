# L5 模型结构与压缩

入口：优化 → 模型结构与压缩。三个独立子页共享 `deployment / artifact / job / run` 身份：

- `/model-structure/build`：检查真实模块、选择目标、冻结变换配方、创建或取消 CPU 任务。
- `/model-structure/artifacts`：查看独立制品、真实参数量、磁盘数据量与模块形状；显式卸载/加载，或恢复原始浮点模型。
- `/model-structure/evaluation`：对当前驻留模型分别运行 WikiText-2 Mini/PPL 与固定原始文本的 16 Token 生成，再选择原始基线与结构候选对照。

本层改变模型表示或结构，通常有损；不能预设质量不变、显存必然下降或服务必然加速。原权重不改写，当前 GPU 模型不被 CPU 构建任务原地修改。现有单请求、L4、L6、L8 和免登录保留。

## 实际实现

| 方法 | 实际变换 / 执行 | 验证边界 |
| --- | --- | --- |
| magnitude | 每输出行按绝对幅值剪去指定比例；可选按输入轴连续 4 个保留 2 个位置 | 权重仍为稠密 Tensor / Linear。零值比例不等于稀疏硬件利用，不声称压缩存储或稀疏加速 |
| wanda | 独立文本校准，记录 Linear 输入特征的平方和与 Token 计数；评分 `abs(W_ij) * sqrt(E[x_j²])`，逐输出行剪枝；支持 2:4 | 本实现冻结一次激活统计，使用 Wanda 评分，**没有复现官方逐层剪枝后重校准协议**。不得把该变体结果标成官方 Wanda 论文复现 |
| mlp_channels | 检查 Gate/Up/Down 的原生 Linear 耦合形状，用三组权重的联合平方幅值选通道；Gate、Up 同时裁输出行，Down 裁对应输入列；保留原顺序与 bias | 真实缩小中间维度与参数存储；支持已检查的 gated MLP。不是“置零但继续全尺寸运算”，也没有修改 Attention 头数或路由 |
| svd_low_rank | CPU FP32 `torch.linalg.svd(full_matrices=False)`，截取 U/S/V；保存两个紧凑矩阵；forward 真正执行两次 Linear | 目标 rank 按矩阵参数保留比例选择：`floor(keep_ratio * m*n/(m+n))`，至少 1，至多 min(m,n)。没有在每次 forward 重建完整 W；两次 GEMM/启动也有成本 |

目标由当前驻留模型的真实模块检查产生。首批候选只选择 `.mlp.` 下 Gate/Up/Down 原生 Linear 或形状一致的 gated MLP；Embedding、LM Head 不在选择器内。未知/不匹配目标失败，不通过字符串名字强行转换。

比例解释：MLP `keep_ratio` 是中间通道保留比例；SVD `keep_ratio` 是选中权重矩阵的目标参数保留比例，rank 取整使实际比例略有变化。全模型减少比例必须用实测参数数计算，不能用所选模块比例代替。幅值/Wanda 的 `sparsity` 是每输出行移除比例；2:4 固定为 0.5，记录保留位置以及实际零值比例，不把原有零值当新增剪枝效果。

## 模块边界、输入输出和语言

| 所有者 | 语言与状态所有权 | 接口 / 输入 → 输出 |
| --- | --- | --- |
| `hqsb/core/structure_plan.py` | Python/Pydantic；无 torch/设备导入 | 原始计划 → 严格 `StructurePlan`；方法、目标、预算、比例、模式、校准上限与支持边界 |
| `structure_modules.py` | Python/PyTorch；真实模块几何、FactorizedLinear | `inspect(model)` → MLP/Linear 候选；`restore_shapes(model, records)` → 具备相同 state-key 结构的模块 |
| `structure_transforms.py` | Python/PyTorch；私有 CPU 模型和瞬时激活统计 | `select / activation_stats / apply` → 可核对模块配方、保留索引、rank、稀疏率和 FP32 Frobenius 诊断 |
| `structure_artifacts.py` | Python/safetensors；格式、去重 state payload、严格验证 | 模型 + tokenizer + 配方 → stage；`verify / load` → 完整性检查和 CPU 紧凑模型；不进行调度 |
| `structure_job.py` | 独立 Python CPU 进程；拥有全部构建对象 | 私有输入文件 → progress、待监督确认制品；禁用 CUDA 可见性，源权重校验后独立加载；归一到所选浮点 dtype 再以 FP32 变换 |
| `structure_jobs.py` | Python；进程生命周期、独占租约、取消、deadline、发布 | 提交契约 → 持久 run；监督 child 退出、校验文件、原子发布、cleanup 及释放 pending/owner |
| `structure_runtime.py` | Python；已加载制品 lineage、模块调用计数 | 真实绑定模块 → 每请求/每调度 variant 的调用凭据；不做指令级稀疏证明 |
| `structure_routes.py` | Python/FastAPI；元数据、封存读取与严格对照 | HTTP → CPU 任务/目录；原始与候选封存证据 → PPL/NLL/性能/存储差异 |
| `features/structure` 与 `ModelStructure.tsx` | TypeScript/React；子页与 URL、查询、草稿 | 实际 API → 变换配置、进度、制品、加载与质量/性能对照；复用已有模型生命周期函数 |

本阶段不新增 C++/CUDA 算子。PyTorch 已提供 SVD、矩阵乘和设备执行；这里的变化轴是模型表示、配方、权重所有权与证据。若要加速 2:4 的真实稀疏执行，需另接稀疏格式和 GPU kernel 并做 L2 验收。

## 资源、并发与制品

1. 构建要求空闲、已就绪、原始非量化、原生算子的本地模型。一次只接受一个 CPU 制品任务，借用现有全局 owner 与 pending 防止期间加载/卸载/推理；模型本身保持 ready，epoch 不变。
2. CPU 子进程从已验证源检查点加载独立副本。`CUDA_VISIBLE_DEVICES=''`，Torch/OMP/MKL 限制 2 线程；不克隆 GPU 模型，也不把驻留模型搬到 CPU。准入复用项目的有效主机内存检查，取主机可用量与容器 cgroup 剩余额度的较小值；CPU 副本/工作空间估算与磁盘制品预算分开，估算不等于实际峰值保证。
3. 任务拥有自己的输入、进度、日志和 stage。校准原文只在私有任务输入文件中短暂存在，finally 删除；run config 与制品只留哈希及计数。校准每条最多 256 Token / 16 条，显式目标最多 32。SVD 有单矩阵维度/元素上界和整体 wall-clock deadline。
4. supervisor 确认子进程成功退出、文件 SHA-256 与预算、取消状态后，原子发布 `l5_<id>/`，绑定 creation run 和 manifest SHA。发布/完成与取消的最后竞态由 service lock 排序。取消/超时终止并等待子进程后清理 stage，原模型不失效。若强制终止后仍无法确认子进程退出，标记 `cleanup=unknown`、保留不确定的 stage、停止接受新 CPU 构建，并释放全局设备队列；不假报清理成功。私有输入文件仍在 finally 中单独尝试删除，需检查运行错误并重启控制服务恢复 CPU 构建。
5. 原始 config、generation config 与 tokenizer 随制品保存；模块覆写配方和紧凑 `weights.safetensors` 单独定义 HQSB 格式。state payload 按完全相同的 storage/shape/offset 去重，保留 tied weights。加载严格匹配全部 state keys，恢复 tie；原始全尺寸模型只在 CPU 构造，替换后才将紧凑模型上 GPU。
6. 此格式需使用 HQSB 加载器；不能直接声称任意 `AutoModel.from_pretrained` 可以读成压缩模型。CPU 构建/加载峰值、保存与验证有成本，不能把驻留参数减少当成整个生命周期峰值减少。
7. 制品目录只读列表不反复扫描 GB 级权重；读取 metadata 会核对创建任务的 manifest SHA，真正加载会再次校验全部文件。中断的 partial 不作为可加载制品；不自动删除其他任务的数据。

## API 和跨层联动

- `GET /api/console/v1/structure/catalog`：方法、来源和未实现方向。
- `GET /structure/capabilities?deployment_id=...`：空闲串行 worker 内检查真实模块，无 forward。
- `POST /structure/jobs`：`deployment_id`、`expected_epoch`、`plan`；通用 run kind 为 `compress`。提交按 Idempotency-Key 去重。
- `GET /structure/jobs`、`GET /structure/artifacts?deployment_id=...`、`GET /structure/artifacts/{id}`：记录与已确认制品。
- 取消沿用 `POST /requests/{run_id}/cancel`，不会通过停止 GPU worker 来取消 CPU 压缩。
- `POST /deployments/{id}/load` 新增 `structure_artifact_id`；省略时恢复原始路径。只允许匹配源模型、dtype 的自有已确认制品；不组合 L4 量化、特权 worker 或自定义 L2 算子。
- 推理与质量沿用 `/requests`、`/precision/quality-jobs`。L5 模型的运行均封存执行凭据；固定生成基线显式登记 optimization 以确保也有封存证据。
- `GET /structure/runs?deployment_id=...` 和 `POST /structure/comparisons`（baseline/candidate run ID）负责本层对照。

原模型 identity 是验证过的源 manifest；候选 identity 是新 L5 manifest SHA。执行凭据增加 `structure_receipt`，保存 parent identity、制品 hash、真实 rank/宽度/索引和调用次数。候选不被自动登记为“原始浮点 baseline”。

L5 对照只在 parent lineage 严格匹配原模型后，为质量协议比较临时归一来源身份；原封存候选身份不改写。仍核对 tokenizer、源码、精度、attention、硬件及评分样本/协议，并检查校准/评测文本精确重合。保存对照到 `l5-comparisons/`。性能对照另核对输入 Token hash、输出预算/长度和采集口径；不要求 epoch 相同，因为模型重载是本层实验的必要步骤。

架构页继续显示源 config/静态语义图，同时标明当前 L5 覆写；模块树、所选模块参数表展示实际运行形状。L6 的模型身份包含 L5 制品 hash，前缀不会跨不同结构串用。L8 variant 增加结构调用凭据。编译与 L5 组合尚未验收，控制面明确拒绝非 eager 方案。

## 指标与解读

直接记录：参数个数与同 dtype 参数字节、actual storage/allocator（现有运行指标）、选中模块形状、保留通道/rank、实际零值比例、调用次数、制品数据文件字节、任务耗时、PPL/NLL 与相对基线变化、固定长度生成 Token/s/首 Token/总时间。

Frobenius 误差在 CPU FP32 变换阶段计算，不是模型质量，也不包含最终导出 dtype 的全部误差。参数字节是参数量口径，不包括 KV/激活/workspace；`disk_bytes` 为 manifest 清单中的数据文件字节，不含 manifest 自身。不报告凭空估算的 GPU L2 命中、物理页、有效稀疏 FLOPs 或正式端到端 speedup。

内置 WikiText-2 Mini 是固定英文子集；不能代表数学、中文、多模态或长距离能力。没有恢复训练的高比例压缩可能明显退化。Wanda 的一次校准变体不保证优于幅值剪枝。两次 Linear 也不保证快于一次大 GEMM；比较要看相同负载的重复测量、真实 allocator 与质量约束。

## 当前没有的能力

没有教师/学生蒸馏训练、剪枝后恢复训练、任意 MHA→GQA/MQA/MLA 转换、MoE 专家/路由结构改写、局部/稀疏/线性 Attention 语义改写，也没有真实 2:4 稀疏 GPU kernel。原生具有这些结构的模型仍可通过原有 ModelScope 模型接入路径使用；这不等于项目已经训练或转换出了这些结构。

## 验证

按 AGENTS，所有测试经 `remote_run.sh` 在 3090-inf：

```bash
./scripts/remote_run.sh env OMP_NUM_THREADS=2 pytest -q tests/unit/console/test_structure_compression.py
./scripts/remote_run.sh env OMP_NUM_THREADS=2 pytest -q tests/hardware/test_structure_cuda.py
./scripts/remote_run.sh python -m scripts.console.validate_l5
```

生产验证脚本只在服务空闲时运行，保存原驻留模型配方，在 finally 恢复，不下载模型或覆盖原权重。具体模型/算法矩阵、质量变化、运行 ID、测试状态和未覆盖内容见 `reports/console/l5-structure/report.md`。

来源：PyTorch 的 [SVD](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)、[剪枝说明](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)；Wanda [作者代码](https://github.com/locuslab/wanda)。实现以目标机安装代码和本次实际验证为准。
