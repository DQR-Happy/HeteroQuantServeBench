# HQSB 项目现状总览

本页更新于 2026-09-20，基于源码、Jetson 软件回归与本机原始实验档案。审查起点为 `de9b117b5385bed2152214ed27279e95764fd705`，工作树已有未提交的 E05 适配器/实验脚本修改；本次审查未提交 Git，也未修改历史 raw/verdict。

**项目尚未全部完成。** 当前最可靠的定位是“带真实 CUDA、Qwen 基线和量化实验的推理优化研究平台”；后期大量代码实现的是契约、策略模型、审计和实验编排，不能据此声称拥有完整的生产推理引擎、集群系统或已验收的跨硬件优化。

现状依据优先级：具体原始 verdict/运行证据 → 当前回归日志 → 实现源码 → 阶段验收报告 → 设计方案。测试通过验证软件行为，不自动升级正式实验状态。旧版累积报告保存在 [历史快照](history/project_status_20260919.md)，不再同时维护多个“当前阶段”。

## 阶段与实际能力

| 阶段 | 当前已有 | 关键缺口/证据边界 |
|---|---|---|
| S00 审计恢复 | 模型制品校验、环境身份、原始证据与审计 runner | 原始档案不在 Git；需要单独归档/恢复 |
| S01 工程契约 | C1–C7、配置、注册表、错误、依赖门、广泛测试 | E01-06 raw 判定 FAIL；本次补上 C6→C7 trace 关联，但未重跑正式实验 |
| S02 基线/profile | Qwen FP16 model-core、6 个 workload、内存/功耗/热点分析 | model-core 指标不包含 HTTP/排队；测试不能替代新的性能基线 |
| S03 CUDA | RMSNorm V0/V1/V2/scalar、fused residual、C API、CTest | E03-06/07 FAIL；E03-08/09 BLOCKED。原生裸指针/stream 边界与完整安全验收仍有缺口 |
| S04 DSL/库 | Triton RMSNorm/GEMM、CUTLASS、TileLang、capability/dispatcher | E04-02/10 BLOCKED，E04-08/09 FAIL；forced 全路径、完整 routing key、版本化 ABI 未闭环 |
| S04.5 模型回接 | 有正式设计与可复用算子 | 缺真实 Qwen 自定义算子回接、逐层/token 对齐与双架构验收，不能标成已完成 |
| S05 量化 | RTN、packing/artifact、质量/校准、低比特 Triton 算子、方法 adapter | E05-01 PASS；E05-02 BLOCKED 且 W8/W4 质量门未过；E05-03 BLOCKED/scientific FAIL；E05-04 BLOCKED/NOT_EXECUTED |
| S06 图集成 | schema/meta、图 IR、rewrite/guard/cache/lowering、差分工具 | 大量对象是契约/图模型；没有设计要求的真实 Qwen 两算子+fusion 验收 |
| S07 Runtime | 请求/身份、KV/调度策略、dummy、C4 reference bridge | 本次修复 C4 对接；未提供完整 vLLM/SGLang/edge 执行 adapter，真实策略 A/B 待实验 |
| S08 Serving | 可执行 stdlib HTTP/SSE、网关、dummy 后端、路由/取消/背压工具 | ServingBackend 与 RuntimeAdapter 是不同接口；缺真实模型服务桥，不能直接当生产服务启动 |
| S09 Ascend | C++ 算子源码、Python capability/tiling/模拟、runbook | 部分交付；缺阶段 driver/config/test 闭环，无本次 NPU 硬件验证 |
| S10 分布式 | 拓扑/collective/并行计划/trace/overlap 模型与 gate | 多设备真实 TP/EP、通信与扩展收益未验收 |
| S11 编译器 | IR、rewrite、target/lowering、缓存/调优/成本模型 | 本次补 C6 导出；真实 Qwen capture→自定义 kernel 与 TVM/MLIR 链未验收 |
| S12 评估 | 可比性、重复性、能耗/成本、Pareto/lineage | 本次补真实 C6/C7 导出；多硬件统一 campaign 尚缺证据 |
| S13 生产化 | 策略对象、Helm/Docker/观测/runbook 模板 | 模板含实际不可构建/启动项，未在集群验证；见审查报告 |
| S14 前沿 | 训练/转换/rollout/MoE/长上下文等契约与审计模型 | 没有完整真实训练/前沿测量闭环；实验层 BLOCKED |
| S15 发布 | claim、证据、复现、供应链与讲述工具 | 正式 release、外部复现、上游协作与读者研究未完成 |

上表的 E03/E04 状态来自 `docs/stage_experiments/S03|S04/<实验>/raw/verdict.json`，并与对应实验报告核对。历史 `docs/reports/S03_阶段验收报告.md`、S04 汇总的“完成”措辞不能覆盖这些更具体的失败记录。

## 本轮修复

- S02→S07：将 RequestSpec 转为 WorkloadSpec，保持显式 token IDs、seed 和固定输出长度；拒绝无法兑现的 timeout/cancel/streaming/sampling 语义。
- PyTorch 后端：按完整 artifact identity 判断复用，避免同名不同 revision 静默命中；执行约定的 warmup 次数，校验请求形状/语义；接通严格 manifest 校验的显式元数据例外与 CPU staging。
- Benchmark：分开 decode-tail TPS 和 output TPS；补 context gate、损坏样本检查、C6/C7 trace 关联。
- S07/S11/S12：partial 不再映射成质量 PASS；将域字段包进冻结的 C6/C7 扩展点。
- CUDA bridge：缺省 stream 使用 PyTorch 当前流；旧库缺 `_ex` 明确拒绝；检查输出设备/别名并登记 allocator stream lifetime。
- 量化：CPU 权重准备不再提前要求 Triton/CUDA；真正 fused 执行仍要求相应能力。
- 八阶段实验存储去重，保留各阶段 verdict 规则，拒绝路径逃逸。
- 清理静态错误/冗余；修复测试依赖私有数据、新增跨阶段/硬件用例；启用 CI lint 与公共文档链接门。
- 同步工具统一远端变量、固定项目根目录、正确同步 `docs/reports`，隔离环境/模型/证据/构建产物。

准确测试计数、逐项场景、预期/实际结果与证据见 [独立功能测试报告](audit/全项目功能测试报告_20260920.md)；问题和修改依据见 [审查报告](audit/全项目审查报告_20260920.md)。

## 使用与后续优先级

先读 [使用说明书](manual/使用说明书.md)，再按 [逐文件/API 索引](manual/generated/README.md) 定位代码。每次修改先跑跨阶段回归，再跑对应硬件 correctness；仅影响代码导航的编辑无需重跑全部 GPU 性能矩阵。

优先完成 E03 安全/stream 约定与 E04 dispatcher/ABI，再做 S04.5 真实模型回接和 S05 质量问题定位。随后选择一个真实 Runtime 接入 Serving，建立可复现的端到端证据。S09/S10/S13/S14 的外部硬件/集群扩展按资源条件推进，避免继续用接口数量代替主链深度。

面向求职：可重点讲清楚真实 profile→算子→数值/流语义→量化失败归因→系统排障这一条链。算子微基准收益、模型收益和服务收益必须分别陈述；对于编译/分布式/生产化模块，说明是设计/模型/脚手架还是已经实测，能比笼统“做完十五阶段”更经得起追问。
