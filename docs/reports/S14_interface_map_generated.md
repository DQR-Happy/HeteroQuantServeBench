# S14 实验步骤 → 代码接口对照表（自动生成）

> 本文件由 `scripts/experimental/gen_interface_map.py` 生成，**请勿手工编辑**。
> 每个接口引用都会被 `hqsb.experimental.interface_map.resolve_interfaces()` 实际导入解析，
> 因此改名/删除会在 `--check` 和单元测试中直接失败，而不是静默指向空。

- 实验数：**12**
- 步骤数：**480 / 480**
- 覆盖完整：**True**

## 总览

| 实验 | 级别 | 步骤数 | 驱动入口 | 覆盖能力（示例） |
|---|---|---|---|---|
| E14-01 | P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-01` | dependencies:CoreGoldenSurface, dependencies:CoreGoldenSurface.digest … |
| E14-02 | P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-02` | training:StrategyChoice, training:STRATEGIES … |
| E14-03 | P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-03` | contracts:CheckpointArtifact, contracts:CheckpointArtifact.aggregate_root … |
| E14-04 | P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-04` | posttraining:PostTrainingAlgorithmChoice, posttraining:ALGORITHMS … |
| E14-05 | P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-05` | records:EXPERIMENT_DEPENDENCIES, records:EXPERIMENT_TABLE … |
| E14-F1 | 条件 P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-F1` | frontier:issue_contract, frontier:contract_hash … |
| E14-F2 | 条件 P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-F2` | frontier:issue_contract, frontier:contract_hash … |
| E14-F3 | 条件 P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-F3` | frontier:issue_contract, frontier:contract_hash … |
| E14-F4 | 条件 P0 | 40 | `scripts/experimental/run_e14.py --experiment E14-F4` | frontier:issue_contract, frontier:contract_hash … |
| E14-06 | P1 | 40 | `scripts/experimental/run_e14.py --experiment E14-06` | multimodal:ModalityChoice, records:MULTIMODAL_MODALITIES … |
| E14-07 | P2 | 40 | `scripts/experimental/run_e14.py --experiment E14-07` | frontier:issue_contract, agent:CLAIM_BOUNDARY … |
| E14-08 | P2 | 40 | `scripts/experimental/run_e14.py --experiment E14-08` | edge:RouteChoice, records:EDGE_ROUTES … |

## 逐实验步骤表

### E14-01 Core/Experimental 依赖、Feature Flag 与 CI 隔离

claim boundary: 只证明依赖边界、导入纯度与 feature 启停语义；不证明任何训练/RL/前沿算法正确、快速或值得进入主线（E14-01 §11）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 core 公共表面 | `dependencies:CoreGoldenSurface`, `dependencies:CoreGoldenSurface.digest`, `dependencies:compare_core_golden` |
| 2 | 冻结依赖分类策略 | `dependencies:DependencyEntry`, `dependencies:DependencyPolicy`, `dependencies:DEPENDENCY_LAYERS` … (+1) |
| 3 | 冻结 extra 与 feature 映射 | `dependencies:FeatureFlagRegistry`, `dependencies:default_feature_flags`, `dependencies:flag_from_extra` … (+1) |
| 4 | 冻结失败语义 | `records:REASON_CODES`, `records:CAPABILITY_STATES`, `records:BLOCKING_CAPABILITY_STATES` … (+1) |
| 5 | 生成测试环境矩阵 | `records:INSTALL_PROFILES`, `dependencies:probe_extra`, `dependencies:negative_case_matrix` |
| 6 | 从清洁 builder 构建 wheel 与 sdist | `dependencies:BuildIdentity`, `dependencies:BuildIdentity.problems`, `identity:digest_file` |
| 7 | 审查 wheel 元数据 | `dependencies:WheelMetadata`, `dependencies:WheelMetadata.parse`, `dependencies:audit_wheel_metadata` |
| 8 | 审查 wheel 内容和包体 | `dependencies:audit_wheel_contents`, `campaign:check_committable` |
| 9 | 在全新 CPU 环境安装 core | `dependencies:DependencyBoundaryResult`, `identity:content_address_aggregate` |
| 10 | 执行 core 冷导入追踪 | `dependencies:ImportTrace`, `dependencies:analyse_import_trace`, `dependencies:EXPERIMENTAL_MODULE_MARKERS` |
| 11 | 测 core 冷/热 import 成本 | `dependencies:ImportTrace.wall_ms`, `dependencies:ImportTrace.rss_bytes`, `dependencies:ImportTrace.phase` |
| 12 | 验证 import 无副作用 | `dependencies:analyse_import_trace`, `contracts:check_noncapture_matrix` |
| 13 | 运行 core Schema/配置 round-trip | `dependencies:CoreGoldenSurface.schema_names`, `dependencies:CORE_AB_FIELDS` |
| 14 | 运行 core CPU reference/golden | `dependencies:CoreGoldenSurface.cpu_reference_entry`, `dependencies:CoreGoldenSurface.golden_outputs` |
| 15 | 运行 core CLI 全表面 | `dependencies:CoreGoldenSurface.cli_entries`, `telemetry:coverage_report` |
| 16 | 静态构建真实 import 图 | `dependencies:static_import_edges`, `dependencies:ImportEdge` |
| 17 | 注入直接依赖泄漏负例 | `dependencies:negative_case_matrix`, `dependencies:static_import_edges` |
| 18 | 注入间接导入泄漏负例 | `dependencies:negative_case_matrix`, `dependencies:static_import_edges`, `dependencies:check_registry_laziness` |
| 19 | 安装 train extra | `dependencies:probe_extra`, `dependencies:EXTRA_IMPORT_NAMES`, `dependencies:DependencyBoundaryResult` |
| 20 | 测试训练 feature off/on | `dependencies:FeatureFlagRegistry.resolve`, `dependencies:probe_extra`, `records:CAP_DISABLED_BY_CONFIG` |
| 21 | 安装并测试 rl extra | `dependencies:probe_extra`, `dependencies:CapabilityResolution.as_dict` |
| 22 | 逐个安装 F1–F4 extra | `dependencies:default_feature_flags`, `dependencies:FeatureFlagRegistry.conflicts`, `records:CAP_UNAVAILABLE_CAPABILITY` |
| 23 | 安装多模态 extra | `dependencies:probe_extra`, `records:EXTRA_FEATURE_MAPPING` |
| 24 | 安装 Agent extra | `dependencies:probe_extra`, `dependencies:check_registry_laziness` |
| 25 | 安装 edge extra | `dependencies:probe_extra`, `dependencies:probe_dependency` |
| 26 | 测试 all-extras 解析 | `dependencies:resolve_all_extras`, `dependencies:DependencyPolicy.digest` |
| 27 | 执行 feature 全关闭等价测试 | `dependencies:compare_core_golden`, `contracts:core_golden_equivalence` |
| 28 | 执行逐 flag 启停测试 | `dependencies:FeatureFlagRegistry.matrix_rows`, `dependencies:FeatureFlagRegistry.resolve` |
| 29 | 测试未知和冲突 flag | `dependencies:FeatureFlagRegistry.resolve`, `dependencies:FeatureFlagRegistry.conflicts`, `records:REASON_CODES` |
| 30 | 测试缺依赖路径 | `dependencies:probe_dependency`, `dependencies:probe_extra`, `records:CAP_UNAVAILABLE_DEPENDENCY` |
| 31 | 测试版本/ABI 不兼容路径 | `dependencies:classify_mismatch`, `records:CAP_ABI_MISMATCH`, `campaign:REQUIRED_ISOLATION` |
| 32 | 验证 registry 懒加载与冲突 | `dependencies:check_registry_laziness`, `dependencies:static_import_edges` |
| 33 | 验证子进程/worker 环境传播 | `dependencies:check_worker_env_propagation`, `identity:RankIdentity` |
| 34 | 执行 core 行为严格 A/B | `dependencies:compare_core_golden`, `dependencies:CORE_AB_FIELDS` |
| 35 | 执行性能预算 A/B | `dependencies:compare_performance_budget`, `dependencies:ImportTrace.wall_ms` |
| 36 | 审查 SBOM、许可证和漏洞面 | `dependencies:compare_sbom`, `records:DEPENDENCY_VERDICTS` |
| 37 | 验证 CI job 与 skip 语义 | `dependencies:audit_ci_matrix`, `records:INSTALL_PROFILES` |
| 38 | 卸载 experimental 依赖后复测 | `dependencies:analyse_import_trace`, `campaign:isolation_clause`, `dependencies:compare_core_golden` |
| 39 | 独立重建和复跑 | `dependencies:BuildIdentity`, `dependencies:compare_builds`, `dependencies:resolve_all_extras` |
| 40 | 形成 DependencyBoundaryVerdict | `dependencies:DependencyBoundaryVerdict`, `dependencies:DependencyBoundaryVerdict.validate`, `records:DEPENDENCY_VERDICTS` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-01`

覆盖模块：`campaign`, `contracts`, `dependencies`, `identity`, `records`, `telemetry`

### E14-02 分布式训练语义、状态、通信与 Checkpoint Smoke

claim boundary: 只证明小模型、给定 topology 的训练能力；不证明模型质量更好，也不证明所选框架优于其他框架（E14-02 §13）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结研究范围和主策略 | `training:StrategyChoice`, `training:STRATEGIES`, `training:StrategyChoice.problems` |
| 2 | 冻结 base ModelArtifact | `contracts:TrainingRunArtifact.base_model_artifact_id`, `identity:artifact_ref` |
| 3 | 冻结数据制品 | `training:GlobalBatchSpec.sample_ids`, `identity:file_inventory` |
| 4 | 冻结 objective 与 reduction | `training:LossReduction`, `training:reconstruct_global_loss` |
| 5 | 冻结 global batch 等价关系 | `training:GlobalBatchSpec`, `training:GlobalBatchSpec.global_batch` |
| 6 | 冻结 optimizer/scheduler/scaler | `contracts:CheckpointArtifact.optimizer_identity`, `training:TrainingStepRecord.scaler` |
| 7 | 冻结 seed bundle | `identity:SeedBundle`, `identity:SEED_ROLES`, `identity:SeedBundle.derive` |
| 8 | 冻结 precision contract | `training:PRECISION_COMPONENTS`, `training:validate_precision_contract` |
| 9 | 冻结 topology 和设备映射 | `identity:RankIdentity`, `identity:topology_digest` |
| 10 | 运行 capability probe | `training:strategy_verdict`, `records:CAP_DEVICE_UNAVAILABLE` |
| 11 | 构建未并行单卡 reference | `training:reconstruct_global_loss`, `training:tolerance_gate` |
| 12 | 验证 reference 的可重复性 | `training:LossReduction`, `training:tolerance_gate` |
| 13 | 启动分布式进程组 | `identity:RankIdentity.process_group`, `training:StrategyChoice.world_size` |
| 14 | 执行 rank/device 一致性握手 | `identity:assert_rank_identities`, `identity:RankIdentity.device_id` |
| 15 | 验证数据分片 | `training:check_data_sharding`, `training:GlobalBatchSpec.problems` |
| 16 | 验证初始参数一致性 | `training:reconcile_tensors`, `training:tolerance_gate` |
| 17 | 运行分布式高精度一步 | `training:reconstruct_global_loss`, `training:TrainingStepRecord` |
| 18 | 对账 loss | `training:reconstruct_global_loss`, `training:naive_rank_mean` |
| 19 | 对账 gradients | `training:reconcile_tensors`, `training:reconciliation_all_within` |
| 20 | 对账参数更新 | `training:reconciliation_first_divergence`, `training:tolerance_gate` |
| 21 | 验证 accumulation no-sync 语义 | `training:accumulation_sync_plan`, `training:accumulation_final_sync` |
| 22 | 运行目标 mixed precision | `training:validate_precision_contract`, `records:REASON_CODES` |
| 23 | 运行短程 loss 轨迹 | `training:TrainingStepRecord`, `training:TrainingStepRecord.validate` |
| 24 | 测每阶段时间 | `training:TIMING_PHASES`, `training:TrainingStepRecord.timing_ms` |
| 25 | 建立 collective 账本 | `training:CollectiveLedgerEntry`, `records:TABLE_SCHEMAS` |
| 26 | 建立每 rank 显存账本 | `training:MemoryLedger`, `training:MEMORY_CATEGORIES` |
| 27 | 归因理论与实测显存差 | `training:attribute_memory_gap`, `training:MemoryLedger.categories` |
| 28 | 检测 straggler 和 rank skew | `training:detect_straggler`, `training:straggler_slowest_rank` |
| 29 | 保存完整 checkpoint | `contracts:CheckpointArtifact`, `training:CheckpointInventory` |
| 30 | 验证 checkpoint inventory/hash | `training:CheckpointInventory.verify`, `identity:content_address_aggregate` |
| 31 | 执行同 topology load-only | `training:CheckpointInventory.topology_digest`, `contracts:CheckpointArtifact.missing_state` |
| 32 | 执行 uninterrupted control | `training:compare_resume_continuity`, `training:RESUME_QUANTITIES` |
| 33 | 执行 stop→resume | `training:compare_resume_continuity`, `training:TrainingStepRecord.step` |
| 34 | 比较恢复连续性 | `training:compare_resume_continuity`, `training:RESUME_QUANTITIES` |
| 35 | 测试 reshard 能力 | `training:reshard_capability`, `records:CAP_UNAVAILABLE_CAPABILITY` |
| 36 | 注入缺 shard/损坏 metadata | `training:inject_checkpoint_fault`, `training:CheckpointInventory.verify`, `campaign:isolation_clause` |
| 37 | 注入 rank failure | `training:inject_checkpoint_fault`, `records:STATUS_FAIL_RECOVERY` |
| 38 | 测 checkpoint save/load 开销 | `training:CheckpointInventory.save_mode`, `training:CheckpointInventory.pause_ms` |
| 39 | 跨独立运行重复 | `training:DistributedTrainingVerdict.limitations`, `records:EXPERIMENT_UNITS` |
| 40 | 形成 DistributedTrainingVerdict | `training:DistributedTrainingVerdict`, `training:DistributedTrainingVerdict.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-02`

覆盖模块：`campaign`, `contracts`, `identity`, `records`, `training`

### E14-03 Checkpoint→转换/Merge→Quant→Runtime 的训推一致性

claim boundary: 只证明转换 DAG、分层语义门与负向拒绝；不证明训练改善质量，也不证明转换后的 runtime 更快（E14-03 §12）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 source CheckpointArtifact | `contracts:CheckpointArtifact`, `contracts:CheckpointArtifact.aggregate_root` |
| 2 | 冻结目标 ServingModelArtifact 契约 | `parity:required_serving_artifact_fields`, `contracts:artifact_hierarchy` |
| 3 | 冻结转换 DAG | `parity:ConversionNodeEvent`, `parity:CONVERSION_NODES`, `identity:LineageChain` |
| 4 | 冻结唯一 intended path | `parity:REFERENCE_PATHS`, `identity:LineageChain.validate` |
| 5 | 冻结 reference workload | `telemetry:project_conversion_node`, `records:TABLE_SCHEMAS` |
| 6 | 冻结精度与容差 | `parity:ParityGate.tolerance`, `records:REASON_CODES` |
| 7 | 冻结 adapter/merge 语义 | `parity:AdapterSemantics`, `parity:AdapterSemantics.problems` |
| 8 | 冻结 quant 前提 | `contracts:ServingModelArtifact.quant_artifact_id`, `records:REASON_CODES` |
| 9 | 盘点 source inventory | `identity:file_inventory`, `identity:content_address_aggregate` |
| 10 | 建立 architecture 参数规范 | `identity:artifact_ref`, `parity:mapping_missing_targets` |
| 11 | 生成显式 mapping table | `parity:TensorMappingRow`, `parity:validate_mapping` |
| 12 | 执行 source 原生 eval baseline | `parity:REFERENCE_PATHS`, `telemetry:project_conversion_node` |
| 13 | 校验 source checkpoint 可完整加载 | `contracts:CheckpointArtifact.missing_state`, `training:CheckpointInventory.verify` |
| 14 | 执行分片 gather/reshard | `parity:TRANSFORMS`, `training:reshard_capability` |
| 15 | 执行 rename/transpose/fusion | `parity:TensorMappingRow.permute`, `parity:validate_mapping` |
| 16 | 执行 adapter unmerged 路径 | `parity:AdapterSemantics.merged`, `parity:REFERENCE_PATHS` |
| 17 | 执行 adapter merge | `parity:AdapterSemantics.merge_count`, `parity:AdapterSemantics.merge_dtype` |
| 18 | 比较 merged 与 unmerged | `parity:compare_merged_unmerged`, `parity:ParityGate.tolerance` |
| 19 | 执行 full-precision 通用格式导出 | `contracts:artifact_hierarchy`, `identity:LineageChain.add` |
| 20 | 验证输出 inventory 完整性 | `parity:validate_mapping`, `contracts:ServingModelArtifact.aggregate_root` |
| 21 | 运行 tensor-level diff | `training:reconcile_tensors`, `parity:ParityGate` |
| 22 | 运行 block-level parity | `parity:PARITY_LAYERS`, `parity:evaluate_gates` |
| 23 | 运行 logits/top-k parity | `parity:gates_first_failure`, `parity:PARITY_LAYERS` |
| 24 | 运行 greedy token parity | `parity:PARITY_LAYERS`, `records:TABLE_SCHEMAS` |
| 25 | 运行 full-precision task quality | `contracts:check_quality_before_performance`, `parity:ParityGate.status` |
| 26 | 执行可选 quant 节点 | `contracts:ServingModelArtifact.quant_artifact_id`, `contracts:ServingModelArtifact.precision_contract_id` |
| 27 | 验证 quant tensor/scale/packing | `parity:TensorMappingRow.dtype_to`, `parity:DTYPES` |
| 28 | 运行 quant correctness/quality gate | `contracts:check_quality_before_performance`, `records:STATUS_FAIL_QUALITY` |
| 29 | 执行 runtime load | `contracts:ServingModelArtifact.runtime_id`, `contracts:check_actual_path_recorded` |
| 30 | 执行可选 engine build | `contracts:ServingModelArtifact.engine_artifact_id`, `contracts:ServingModelArtifact.compatibility_digest` |
| 31 | 验证 runtime/engine 语义 | `parity:evaluate_gates`, `contracts:ServingModelArtifact.status` |
| 32 | 验证 serving readiness gate | `contracts:ServingModelArtifact.quality_evidence_id`, `records:STATUS_INVALID_IDENTITY` |
| 33 | 注入缺失/损坏 shard | `parity:negative_artifact_matrix`, `identity:content_address_aggregate` |
| 34 | 注入错误 tokenizer/chat template | `parity:negative_artifact_matrix`, `contracts:ServingModelArtifact.tokenizer_artifact_id` |
| 35 | 注入错误 config/precision/version | `parity:negative_artifact_matrix`, `contracts:ServingModelArtifact.config_artifact_id` |
| 36 | 注入错误 adapter/base 组合 | `parity:negative_artifact_matrix`, `parity:AdapterSemantics.base_artifact_id` |
| 37 | 验证并发转换和原子发布 | `parity:check_atomic_publish`, `identity:LineageChain.validate` |
| 38 | 验证重复转换与 cache | `parity:check_conversion_cache`, `identity:canonical_digest` |
| 39 | 测转换/加载资源和成本 | `parity:audit_conversion_resources`, `parity:CONVERSION_RESOURCE_KEYS` |
| 40 | 形成 TrainServeParityVerdict | `parity:TrainServeParityVerdict`, `parity:TrainServeParityVerdict.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-03`

覆盖模块：`contracts`, `identity`, `parity`, `records`, `telemetry`, `training`

### E14-04 SFT/DPO/GRPO Rollout、同步/异步与 Policy Staleness

claim boundary: 只证明小模型、数据集与有限训练步下的系统正确性；不等于模型已获得通用对齐、推理或安全能力（E14-04 §12）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 用 ADR 选择唯一主算法 | `posttraining:PostTrainingAlgorithmChoice`, `posttraining:ALGORITHMS` |
| 2 | 冻结算法论文与实现版本 | `posttraining:PostTrainingAlgorithmChoice.framework_identity`, `posttraining:offline_semantics` |
| 3 | 冻结能力声明 | `posttraining:CLAIM_BOUNDARY`, `contracts:AdoptionDecision.forbidden_claims` |
| 4 | 冻结 base PolicySnapshot | `posttraining:snapshot_from_checkpoint`, `contracts:PolicySnapshot` |
| 5 | 冻结 data artifact | `identity:artifact_ref`, `identity:content_address_aggregate` |
| 6 | 冻结 environment/tool contract | `posttraining:build_trajectory`, `records:TABLE_SCHEMAS` |
| 7 | 冻结 reward/reference 定义 | `contracts:TrajectoryRecord.reward_definition_id`, `posttraining:reward_determinism` |
| 8 | 冻结 sampling contract | `posttraining:build_trajectory`, `identity:canonical_digest` |
| 9 | 冻结同步状态机 | `posttraining:SyncStateMachine`, `records:POSTTRAINING_SYNC_TRANSITIONS` |
| 10 | 冻结异步状态机 | `posttraining:AsyncQueuePolicy`, `posttraining:ASYNC_QUEUE_STATES` |
| 11 | 冻结资源和停止预算 | `campaign:RunBudget`, `campaign:BUDGET_DIMENSIONS` |
| 12 | 构造手算最小 objective case | `posttraining:sft_loss`, `posttraining:dpo_loss`, `posttraining:grpo_loss` |
| 13 | 验证 token mask 和长度归一化 | `posttraining:validate_token_mask`, `posttraining:length_normalisation` |
| 14 | 验证 policy/reference logprob | `posttraining:compare_logprobs`, `contracts:PolicySnapshot` |
| 15 | 验证 reward determinism/variance | `posttraining:reward_determinism`, `posttraining:REWARD_FAILURE_MODES` |
| 16 | 验证 rollout engine 语义 | `contracts:PolicySnapshot.rollout_engine_id`, `contracts:check_actual_path_recorded` |
| 17 | 执行合法同步 rollout | `posttraining:build_trajectory`, `contracts:TrajectoryRecord` |
| 18 | 审计 trajectory 完整性 | `posttraining:audit_trajectory_batch`, `contracts:TrajectoryRecord.validate` |
| 19 | 重建 objective 与 batch 统计 | `posttraining:grpo_advantages`, `posttraining:grpo_loss` |
| 20 | 执行单步训练更新 | `posttraining:check_consumption_ledger`, `records:TABLE_SCHEMAS` |
| 21 | 验证更新方向和边界 | `posttraining:dpo_loss`, `posttraining:grpo_clip_fraction` |
| 22 | 转换并发布新 PolicySnapshot | `posttraining:PolicyPublishEvent`, `posttraining:snapshot_from_checkpoint` |
| 23 | 验证权重同步完整性 | `posttraining:PolicyPublishEvent.verified_on_all_ranks`, `contracts:PolicySnapshot.transfer_verified` |
| 24 | 运行短同步闭环 | `posttraining:SyncStateMachine`, `records:TABLE_SCHEMAS` |
| 25 | 执行 held-out 质量与反作弊评价 | `posttraining:check_heldout_quality`, `posttraining:ANTI_CHEAT_INDICATORS` |
| 26 | 部署有界异步队列 | `posttraining:AsyncQueuePolicy`, `posttraining:AsyncQueuePolicy.problems` |
| 27 | 启用版本化异步生成 | `contracts:PolicySnapshot.is_active_at`, `contracts:TrajectoryRecord.produced_at_step` |
| 28 | 验证原子 weight activate | `posttraining:PolicyPublishEvent.atomic_activate`, `contracts:PolicySnapshot.active_from_ns` |
| 29 | 注入 mixed-version batch 负例 | `contracts:check_batch_version_consistency`, `posttraining:audit_trajectory_batch` |
| 30 | 扫描 staleness 梯度 | `posttraining:check_staleness_sweep`, `posttraining:AsyncQueuePolicy.staleness_cap` |
| 31 | 测 batching/queue 与 GPU 利用 | `posttraining:AsyncQueuePolicy.capacity`, `records:TABLE_SCHEMAS` |
| 32 | 验证 KV/prefix reuse 身份 | `posttraining:check_kv_reuse_identity`, `contracts:PolicySnapshot.policy_snapshot_id` |
| 33 | 注入 environment/tool timeout | `posttraining:failure_scenarios`, `posttraining:check_failure_accounting` |
| 34 | 注入 reward failure/slowdown | `posttraining:REWARD_FAILURE_MODES`, `posttraining:check_failure_accounting` |
| 35 | 注入 rollout/trainer worker crash | `posttraining:failure_scenarios`, `posttraining:consumption_semantics` |
| 36 | 执行 checkpoint→resume 闭环 | `training:compare_resume_continuity`, `posttraining:SyncStateMachine` |
| 37 | 做跨层 profiling | `telemetry:project_trajectory_event`, `records:PROFILE_LAYERS` |
| 38 | 跨 seed/run 重复 | `records:EXPERIMENT_UNITS`, `identity:SeedBundle` |
| 39 | 比较 sync/async Pareto | `posttraining:compare_sync_async_pareto`, `contracts:AdoptionDecision` |
| 40 | 形成 PostTrainingVerdict | `posttraining:PostTrainingVerdict`, `posttraining:PostTrainingVerdict.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-04`

覆盖模块：`campaign`, `contracts`, `identity`, `posttraining`, `records`, `telemetry`, `training`

### E14-05 前沿方向 ADR、可证伪假设与跨层预注册

claim boundary: 只证明问题值得问且实验能给出可信答案；不生成任何算法 speedup，也不代表选中分支会有正收益（E14-05 §11）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 S14 公共前置状态 | `records:EXPERIMENT_DEPENDENCIES`, `records:EXPERIMENT_TABLE` |
| 2 | 冻结目标岗位与代表能力 | `frontier:SELECTION_CRITERIA`, `frontier:CandidateScore` |
| 3 | 定义选择约束 | `campaign:RunBudget`, `frontier:CandidateScore.states` |
| 4 | 建立候选文献清单 | `frontier:LiteratureEntry`, `frontier:LiteratureRegistry` |
| 5 | 提取论文实验条件 | `frontier:PaperConditionMatrix`, `frontier:PAPER_CONDITION_FIELDS` |
| 6 | 建立 HQSB 对应矩阵 | `frontier:hqsb_mapping_matrix`, `frontier:HQSB_CONTRACT_FIELDS` |
| 7 | 盘点已有实现资产 | `records:TABLE_SCHEMAS`, `telemetry:PROJECTORS` |
| 8 | 运行候选 capability probes | `dependencies:probe_extra`, `records:CAPABILITY_STATES` |
| 9 | 定义每个候选的最小独立改动 | `frontier:CandidateScore.notes`, `frontier:ComplexityScore` |
| 10 | 定义每个候选的科学缺口 | `frontier:CandidateEstimand`, `frontier:BRANCH_MEDIATORS` |
| 11 | 为 F1 写候选 estimand | `frontier:CandidateEstimand`, `frontier:BRANCH_MEDIATORS` |
| 12 | 为 F2 写候选 estimand | `frontier:CandidateEstimand`, `frontier:BRANCH_FORBIDDEN_CLAIMS` |
| 13 | 为 F3 写候选 estimand | `frontier:CandidateEstimand`, `frontier:CandidateEstimand.problems` |
| 14 | 为 F4 写候选 estimand | `frontier:CandidateEstimand.mediators`, `frontier:BRANCH_MEDIATORS` |
| 15 | 定义硬前置门 | `frontier:HardPrerequisiteGate`, `frontier:gate_blockers`, `records:STATUS_BLOCKED_PREREQUISITE` |
| 16 | 定义质量门 | `frontier:QualityGate`, `contracts:check_quality_before_performance` |
| 17 | 定义 actual-path 门 | `frontier:ActualPathGate`, `frontier:uninstrumented_gates` |
| 18 | 定义 primary hypothesis | `frontier:PrimaryHypothesis`, `frontier:check_single_primary` |
| 19 | 定义 secondary hypotheses | `frontier:check_single_primary`, `records:STATUS_PASS_NEGATIVE` |
| 20 | 定义 baseline 和唯一差异 | `frontier:BaselinePair`, `frontier:BaselinePair.FROZEN` |
| 21 | 定义负对照 | `frontier:NegativeControls`, `contracts:check_negative_control_coverage` |
| 22 | 定义消融矩阵 | `frontier:AblationMatrix`, `frontier:AblationMatrix.max_combinations` |
| 23 | 定义 workload strata | `frontier:WorkloadStrata`, `frontier:WorkloadStrata.REQUIRED_DIMS` |
| 24 | 定义 profile 层和关联键 | `frontier:ProfilePlan`, `contracts:check_profile_layering` |
| 25 | 定义计时边界 | `frontier:TimingBoundaries`, `frontier:TIMING_BOUNDARIES` |
| 26 | 定义资源和成本分母 | `frontier:CostDenominator`, `frontier:COST_DENOMINATORS` |
| 27 | 定义实验单位和重复 | `frontier:StatisticsPlan`, `records:EXPERIMENT_UNITS` |
| 28 | 定义随机化与 blocking | `frontier:StatisticsPlan.randomisation`, `frontier:StatisticsPlan.blocking` |
| 29 | 定义探索预算 | `campaign:RunBudget.stop_rules`, `frontier:AblationMatrix.max_combinations` |
| 30 | 定义 holdout confirmation | `frontier:WorkloadStrata.holdout_id`, `frontier:BaselinePair` |
| 31 | 定义停止/中止规则 | `frontier:StopRules`, `frontier:StopRules.REQUIRED` |
| 32 | 定义缺失与 invalid 规则 | `frontier:InvalidityRules`, `frontier:INVALIDITY_CATEGORIES` |
| 33 | 定义预测模型 | `frontier:PredictionModel`, `frontier:PredictionModel.REQUIRED` |
| 34 | 定义结果归因要求 | `frontier:check_attribution`, `contracts:check_actual_path_recorded` |
| 35 | 定义复杂性与维护评分 | `frontier:ComplexityScore`, `contracts:AdoptionDecision.limitations` |
| 36 | 预注册 AdoptionDecision | `frontier:AdoptionRules`, `records:ADOPTION_DECISIONS` |
| 37 | 选择唯一主 F 分支 | `frontier:select_single_branch`, `frontier:CandidateScore` |
| 38 | 锁定未选分支状态 | `frontier:lock_unselected_branches`, `records:N_A_BY_ADR` |
| 39 | 执行盲化协议复核 | `frontier:protocol_review`, `frontier:FrontierSelectionVerdict` |
| 40 | 签发 FrontierStudyContract | `frontier:issue_contract`, `frontier:contract_hash`, `contracts:supersede_contract` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-05`

覆盖模块：`campaign`, `contracts`, `dependencies`, `frontier`, `records`, `telemetry`

### E14-F1 Speculative/MTP Acceptance、额外计算与服务调度

claim boundary: 只在 F1 为唯一主分支时成立；质量与 actual-path 门通过前，acceptance 数字不构成加速声明。无收益但门禁与归因完整时可形成 PASS_NEGATIVE/REJECT_NO_BENEFIT（E14-F1 §12）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 绑定 E14-05 协议 | `frontier:issue_contract`, `frontier:contract_hash` |
| 2 | 冻结 speculative 机制版本 | `speculative:PROPOSER_KINDS`, `speculative:TargetProposerBinding.proposer_kind` |
| 3 | 冻结 target ModelArtifact | `contracts:ServingModelArtifact`, `contracts:ServingModelArtifact.quality_evidence_id` |
| 4 | 冻结 proposer artifact | `speculative:TargetProposerBinding`, `identity:artifact_ref` |
| 5 | 验证 tokenizer/vocabulary compatibility | `speculative:TargetProposerBinding.problems`, `speculative:TargetProposerBinding.token_id_mapping` |
| 6 | 冻结生成语义 | `speculative:GenerationContract`, `speculative:GENERATION_MODES` |
| 7 | 冻结 workload strata | `speculative:ACCEPTANCE_STRATA`, `frontier:WorkloadStrata` |
| 8 | 冻结 service load | `frontier:BaselinePair`, `records:EXPERIMENT_UNITS` |
| 9 | 预注册理论成本模型 | `speculative:SpeculationCostModel`, `speculative:SpeculationCostModel.predict` |
| 10 | 运行 capability/actual-path probe | `contracts:check_actual_path_recorded`, `records:STATUS_INVALID_IDENTITY` |
| 11 | 建立 target-only correctness baseline | `speculative:greedy_prefix_match`, `parity:REFERENCE_PATHS` |
| 12 | 建立 target-only 性能基线 | `contracts:check_resource_ledger`, `telemetry:coverage_report` |
| 13 | 建立 proposer 独立能力基线 | `speculative:TargetProposerBinding`, `speculative:propose_with_entropy` |
| 14 | 构造手算 acceptance case | `speculative:accept_reject_test`, `speculative:residual_distribution` |
| 15 | 验证 greedy 逐 token parity | `speculative:greedy_prefix_match`, `speculative:SpeculationCycle.committed_token_ids` |
| 16 | 验证 sampling 分布语义 | `speculative:residual_distribution`, `speculative:check_quality_gate` |
| 17 | 插桩 proposal/verification 状态机 | `speculative:SpeculationCycle`, `speculative:acceptance_by_position` |
| 18 | 验证 EOS/stop/长度边界 | `speculative:STOP_REASONS`, `speculative:GenerationContract.eos_token_id` |
| 19 | 验证 KV commit/rollback | `speculative:KV_ACTIONS`, `speculative:SpeculationCycle.kv_discarded_bytes` |
| 20 | 运行主参数 γ/steps 扫描 | `speculative:SpeculationCostModel.gamma`, `frontier:StatisticsPlan` |
| 21 | 运行 tree width/depth 消融 | `frontier:AblationMatrix`, `speculative:PROPOSER_KINDS` |
| 22 | 运行 draft size/precision 消融 | `speculative:TargetProposerBinding.proposer_precision`, `frontier:CostDenominator` |
| 23 | 运行 domain/entropy 分层 | `speculative:stratify_by_domain`, `speculative:propose_with_entropy` |
| 24 | 运行 context/output 长度分层 | `speculative:stratify_by_domain`, `frontier:WorkloadStrata.holdout_id` |
| 25 | 运行单请求与固定 batch | `speculative:effective_tpot`, `records:TABLE_SCHEMAS` |
| 26 | 运行多并发 continuous batching | `speculative:effective_tpot`, `contracts:check_resource_ledger` |
| 27 | 运行 open-loop SLO 曲线 | `records:PROFILE_LAYERS`, `contracts:AdoptionDecision.performance_status` |
| 28 | 验证长短/高低 acceptance 公平性 | `speculative:stratify_by_domain`, `frontier:StopRules` |
| 29 | 验证 prefix cache 交互 | `posttraining:check_kv_reuse_identity`, `speculative:wasted_work` |
| 30 | 采集 kernel/系统 profile | `speculative:cycle_cost_ms`, `records:PROFILE_LAYER_FIELDS` |
| 31 | 计算计算与内存浪费 | `speculative:wasted_work`, `speculative:SpeculationCycle.kv_written_bytes` |
| 32 | 测显存、能耗和设备成本 | `contracts:check_resource_ledger`, `frontier:CostDenominator` |
| 33 | 注入低 acceptance/错误 proposer | `speculative:low_acceptance_guard`, `speculative:proposer_failure_fallback` |
| 34 | 注入 proposer failure/timeout | `speculative:proposer_failure_fallback`, `campaign:REQUIRED_ISOLATION` |
| 35 | 验证 cancel/backpressure/drain | `speculative:check_cancel_release`, `speculative:STOP_REASONS` |
| 36 | 运行任务质量门 | `speculative:check_quality_gate`, `contracts:check_quality_before_performance` |
| 37 | 对账预测与实测 | `speculative:reconcile_prediction`, `speculative:SpeculationCostModel.predict` |
| 38 | 在 holdout workloads 确认 | `frontier:WorkloadStrata.holdout_id`, `frontier:StatisticsPlan` |
| 39 | 跨 run/time block 重复 | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan.minimum_repeats` |
| 40 | 形成 F1 AdoptionDecision | `speculative:f1_adoption`, `contracts:AdoptionDecision.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-F1`

覆盖模块：`campaign`, `contracts`, `frontier`, `identity`, `parity`, `posttraining`, `records`, `speculative`, `telemetry`

### E14-F2 MoE Router、Expert Parallel、All-to-All 与负载均衡

claim boundary: 只在 F2 为唯一主分支时成立；受控 router replay 只能用于系统测试，不能产生质量结论。无收益但能证明通信/小 GEMM/热点是原因时应输出 PASS_NEGATIVE（E14-F2 §12）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 绑定 E14-05 协议 | `frontier:issue_contract`, `frontier:contract_hash` |
| 2 | 冻结真实 MoE ModelArtifact | `moe:MoEModelContract`, `moe:MoEModelContract.total_to_active_ratio` |
| 3 | 冻结 reference 与 candidate | `frontier:BaselinePair`, `moe:select_single_mitigation` |
| 4 | 冻结并行与 topology | `moe:ExpertPlacement`, `moe:validate_placement` |
| 5 | 冻结 router 语义 | `moe:ROUTER_ACTIVATIONS`, `moe:ROUTER_NORMALISATIONS`, `moe:router_topk` |
| 6 | 冻结 workload/domain strata | `moe:stratify_routing`, `frontier:WorkloadStrata` |
| 7 | 构造可控 skew 输入 | `moe:controlled_count_replay`, `moe:stratify_routing` |
| 8 | 冻结质量与容量门 | `moe:check_quality_gate`, `frontier:QualityGate` |
| 9 | 冻结成本模型 | `frontier:PredictionModel`, `moe:load_statistics` |
| 10 | 运行 capability/actual-path probe | `contracts:check_actual_path_recorded`, `moe:ExpertGEMMLedgerEntry.kernel` |
| 11 | 运行单设备 reference correctness | `moe:expert_output`, `moe:router_topk` |
| 12 | 验证 router instrumentation | `moe:RoutingEvent`, `moe:load_statistics` |
| 13 | 运行真实 workload routing census | `moe:load_statistics`, `moe:IMBALANCE_METRICS` |
| 14 | 验证 expert output parity | `moe:expert_output`, `moe:check_quality_gate` |
| 15 | 启动 EP baseline | `moe:ExpertPlacement`, `moe:validate_placement` |
| 16 | 对账 token dispatch/combine | `moe:check_token_dispatch`, `moe:AllToAllLedgerEntry` |
| 17 | 建立 All-to-All ledger | `moe:AllToAllLedgerEntry`, `moe:a2a_balance` |
| 18 | 建立 expert GEMM ledger | `moe:ExpertGEMMLedgerEntry`, `moe:gemm_efficiency` |
| 19 | 建立 per-rank 时间分解 | `moe:rank_time_decomposition`, `moe:LAYER_PHASES` |
| 20 | 建立内存账本 | `moe:MoEMemoryLedger`, `moe:MOE_MEMORY_CATEGORIES` |
| 21 | 运行 batch/context 扫描 | `moe:gemm_efficiency`, `frontier:WorkloadStrata` |
| 22 | 运行真实 domain skew 分层 | `moe:stratify_routing`, `moe:load_statistics` |
| 23 | 运行受控 count replay | `moe:controlled_count_replay`, `moe:load_statistics` |
| 24 | 扫描 expert placement | `moe:placement_scan`, `moe:ExpertPlacement.expert_to_rank` |
| 25 | 验证 placement 身份与迁移 | `moe:ExpertPlacement.version`, `moe:validate_placement` |
| 26 | 评估 expert replication/load-balance | `moe:replication_cost`, `frontier:CostDenominator` |
| 27 | 评估 expert cache/offload | `moe:expert_cache_scan`, `moe:MoEMemoryLedger.resident_bytes` |
| 28 | 评估 expert quant | `moe:expert_quant_gate`, `contracts:check_quality_before_performance` |
| 29 | 选择一个主要缓解策略 | `moe:select_single_mitigation`, `moe:MITIGATIONS` |
| 30 | 运行严格 runtime A/B | `moe:runtime_ab`, `frontier:BaselinePair` |
| 31 | 运行 Service 到达率曲线 | `moe:service_arrival_curve`, `records:PROFILE_LAYERS` |
| 32 | 验证混合租户/优先级公平性 | `moe:check_fairness`, `frontier:StopRules` |
| 33 | 采集跨层关联 profile | `moe:cross_layer_link`, `moe:rank_time_decomposition` |
| 34 | 注入单 expert/rank 慢化 | `moe:straggler_injection`, `campaign:isolation_clause` |
| 35 | 注入 expert/cache/communication failure | `moe:failure_scenarios`, `moe:check_failure_accounting` |
| 36 | 运行质量门 | `moe:check_quality_gate`, `contracts:check_quality_before_performance` |
| 37 | 对账模型预测与实测 | `moe:reconcile_prediction`, `moe:rank_time_decomposition` |
| 38 | 在 holdout domain/load 确认 | `frontier:WorkloadStrata.holdout_id`, `moe:stratify_routing` |
| 39 | 跨 topology/time block 重复 | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan.minimum_repeats` |
| 40 | 形成 F2 AdoptionDecision | `moe:f2_adoption`, `contracts:AdoptionDecision.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-F2`

覆盖模块：`campaign`, `contracts`, `frontier`, `moe`, `records`

### E14-F3 长上下文 KV、Chunked Prefill、容量—质量—延迟

claim boundary: 只在 F3 为唯一主分支时成立；容量提升必须同时给出质量与延迟，截断/滑窗/OOM fallback 不算支持。允许合法的“容量显著改善但性能无收益”结论（E14-F3 §12）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 绑定 E14-05 协议 | `frontier:issue_contract`, `frontier:contract_hash` |
| 2 | 冻结 long-context ModelArtifact | `long_context:kv_bytes_per_token`, `contracts:ServingModelArtifact` |
| 3 | 冻结 full-KV/full-attention baseline | `frontier:BaselinePair`, `long_context:kv_capacity` |
| 4 | 冻结 candidate artifact/config | `frontier:BaselinePair.intended_difference`, `long_context:INTERVENTIONS` |
| 5 | 冻结真实输入 token ledger | `long_context:LongContextEpisode.submitted_tokens`, `identity:file_inventory` |
| 6 | 冻结长上下文质量集 | `long_context:LONG_CONTEXT_TASKS`, `frontier:QualityGate` |
| 7 | 冻结生成与评分语义 | `frontier:TimingBoundaries`, `speculative:GenerationContract` |
| 8 | 冻结长度/并发扫描 | `long_context:LENGTH_BUCKETS`, `frontier:WorkloadStrata` |
| 9 | 冻结容量与 OOM 安全边界 | `long_context:ADMISSION_BUDGETS`, `frontier:StopRules` |
| 10 | 建立理论 KV/attention 模型 | `long_context:kv_bytes_per_token`, `frontier:PredictionModel` |
| 11 | 运行 capability/actual-path probe | `contracts:check_actual_path_recorded`, `long_context:kv_quant_report` |
| 12 | 建立短上下文 correctness baseline | `parity:evaluate_gates`, `long_context:short_context_regression` |
| 13 | 建立 full-KV 长度基线 | `long_context:LongContextEpisode`, `long_context:kv_capacity` |
| 14 | 验证无截断/无滑窗偷换 | `long_context:check_no_truncation`, `long_context:LongContextEpisode.truncated_tokens` |
| 15 | 校准 KV bytes/token | `long_context:calibrate_bytes_per_token`, `long_context:kv_bytes_per_token` |
| 16 | 建立 block/fragmentation 基线 | `long_context:KvAllocatorTimeline`, `long_context:KvAllocatorTimeline.tail_waste` |
| 17 | 执行 candidate 小规模语义测试 | `parity:evaluate_gates`, `long_context:LongContextEpisode.quality` |
| 18 | 验证 position/mask/chunk boundary | `long_context:verify_position_and_mask`, `long_context:chunked_prefill_audit` |
| 19 | 验证 KV quant/dequant（若适用） | `long_context:kv_quant_report`, `long_context:KV_STORAGE_BYTES` |
| 20 | 验证 eviction/recompute（若适用） | `long_context:eviction_audit`, `long_context:INTERVENTIONS` |
| 21 | 验证 chunked prefill（若适用） | `long_context:chunked_prefill_audit`, `long_context:verify_position_and_mask` |
| 22 | 运行 context-length 扫描 | `long_context:LENGTH_BUCKETS`, `long_context:LongContextEpisode` |
| 23 | 运行 output-length 扫描 | `long_context:LongContextEpisode.output_tokens`, `long_context:LongContextEpisode.tpot_ms` |
| 24 | 运行 batch/concurrency 扫描 | `long_context:kv_capacity`, `long_context:admission_budget` |
| 25 | 运行 prefix reuse 分层 | `posttraining:check_kv_reuse_identity`, `long_context:KvAllocatorTimeline` |
| 26 | 运行长上下文质量门 | `long_context:quality_by_length_position`, `frontier:QualityGate.paired` |
| 27 | 运行短上下文回归门 | `long_context:short_context_regression`, `contracts:check_quality_before_performance` |
| 28 | 采集 attention/KV kernel profile | `long_context:kv_actual_kernel`, `long_context:kv_quant_report`, `records:PROFILE_LAYER_FIELDS` |
| 29 | 运行长短混合 closed-loop | `long_context:mixed_load_fairness`, `frontier:StatisticsPlan` |
| 30 | 运行 open-loop 到达率曲线 | `long_context:admission_budget`, `records:PROFILE_LAYERS` |
| 31 | 验证 chunk/admission 调度协同 | `long_context:admission_budget`, `long_context:chunked_prefill_audit` |
| 32 | 测显存、主存、能耗和成本 | `contracts:check_resource_ledger`, `frontier:CostDenominator` |
| 33 | 执行受控 over-limit 负例 | `long_context:over_limit_rejection`, `records:STATUS_FAIL_CORRECTNESS` |
| 34 | 执行受控 OOM/fragmentation 边界 | `long_context:kv_capacity`, `campaign:isolation_clause` |
| 35 | 注入损坏 KV metadata/cache | `long_context:corrupted_kv_metadata_check`, `campaign:isolation_clause` |
| 36 | 验证取消、超时和 eviction 并发 | `long_context:cancel_release`, `long_context:eviction_audit` |
| 37 | 对账预测与实测 | `long_context:reconcile_prediction`, `long_context:calibrate_bytes_per_token` |
| 38 | 在 holdout 长文/负载确认 | `frontier:WorkloadStrata.holdout_id`, `long_context:quality_by_length_position` |
| 39 | 跨 run/time block 重复 | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan.minimum_repeats` |
| 40 | 形成 F3 AdoptionDecision | `long_context:f3_adoption`, `contracts:AdoptionDecision.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-F3`

覆盖模块：`campaign`, `contracts`, `frontier`, `identity`, `long_context`, `parity`, `posttraining`, `records`, `speculative`

### E14-F4 结构化稀疏/Sparse Attention 的真实硬件加速

claim boundary: 只在 F4 为唯一主分支时成立；零元素比例不等于硬件稀疏，micro speedup 不等于模型收益。严格证明“当前硬件/shape 不值得采用”同样是有价值的结果（E14-F4 §12）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 绑定 E14-05 协议 | `frontier:issue_contract`, `frontier:contract_hash` |
| 2 | 冻结 dense ModelArtifact | `contracts:ServingModelArtifact`, `sparsity:SparseArtifactSchema.source_artifact_id` |
| 3 | 冻结 sparse transform | `sparsity:SparseTransform`, `sparsity:ROUTES` |
| 4 | 冻结 sparse artifact schema | `sparsity:SparseArtifactSchema`, `sparsity:SPARSE_DTYPES` |
| 5 | 冻结硬件/runtime 支持域 | `sparsity:SupportDomain`, `sparsity:SupportDomain.supports` |
| 6 | 冻结 dense baseline | `frontier:BaselinePair`, `sparsity:SparseTransform.problems` |
| 7 | 冻结质量门 | `frontier:QualityGate`, `sparsity:DISTANCE_METRICS` |
| 8 | 冻结 shape/workload 矩阵 | `sparsity:shape_scan`, `frontier:WorkloadStrata` |
| 9 | 冻结成本与摊销口径 | `sparsity:conversion_cost`, `sparsity:SPARSE_COST_COMPONENTS` |
| 10 | 建立理论 FLOPs/bytes/Amdahl 模型 | `sparsity:coverage_and_amdahl`, `frontier:PredictionModel` |
| 11 | 运行 capability probe | `sparsity:check_actual_dispatch`, `sparsity:SupportDomain.supports` |
| 12 | 建立 dense correctness/performance baseline | `sparsity:sparse_gemm_reference`, `frontier:BaselinePair` |
| 13 | 执行 sparse transform | `sparsity:SparseTransform.retrained`, `identity:content_address_aggregate` |
| 14 | 验证 sparsity/pattern compliance | `sparsity:pattern_compliance`, `sparsity:WEIGHT_PATTERNS` |
| 15 | 验证 compressed representation | `sparsity:compressed_roundtrip`, `sparsity:SparseArtifactSchema.alignment` |
| 16 | 运行 tensor-level correctness | `sparsity:sparse_gemm_reference`, `parity:evaluate_gates` |
| 17 | 分离算法误差与实现误差 | `sparsity:sparse_gemm_reference`, `sparsity:SparseExecutionRecord` |
| 18 | 运行特殊 shape/tail/非对齐 | `sparsity:shape_scan`, `sparsity:SupportDomain.supports` |
| 19 | 验证 actual sparse kernel | `sparsity:check_actual_dispatch`, `sparsity:DISPATCH_REASONS` |
| 20 | 做 kernel shape 扫描 | `sparsity:shape_scan`, `frontier:StatisticsPlan` |
| 21 | 测 metadata/index/mask 开销 | `sparsity:metadata_cost`, `sparsity:SparseExecutionRecord.metadata_bytes` |
| 22 | 测 conversion/compression 开销 | `sparsity:conversion_cost`, `frontier:CostDenominator` |
| 23 | 运行 layer/block 集成 | `sparsity:coverage_and_amdahl`, `parity:evaluate_gates` |
| 24 | 扫描覆盖层选择 | `sparsity:coverage_and_amdahl`, `sparsity:runtime_dynamic_workload` |
| 25 | 运行 full-model correctness | `parity:evaluate_gates`, `sparsity:SparseExecutionRecord.correctness_status` |
| 26 | 运行任务质量门 | `sparsity:attention_quality_by_distance`, `contracts:check_quality_before_performance` |
| 27 | 运行 model-core 性能 | `sparsity:coverage_and_amdahl`, `records:PROFILE_LAYER_FIELDS` |
| 28 | 运行编译/graph/cache 测试 | `sparsity:compile_cache_check`, `records:REASON_CODES` |
| 29 | 运行 Runtime 动态 workload | `sparsity:runtime_dynamic_workload`, `sparsity:dispatch_fell_back` |
| 30 | 运行 Service 到达率曲线 | `sparsity:service_curve`, `records:PROFILE_LAYERS` |
| 31 | 测内存和容量 | `sparsity:memory_and_capacity`, `contracts:check_resource_ledger` |
| 32 | 测能耗和成本 | `sparsity:energy_and_cost`, `frontier:CostDenominator` |
| 33 | 测试 unsupported capability/shape | `sparsity:unsupported_capability`, `sparsity:DISPATCH_REASONS` |
| 34 | 注入损坏 metadata/index/mask | `sparsity:corrupted_artifact_check`, `campaign:isolation_clause` |
| 35 | 测试版本/engine mismatch | `sparsity:version_mismatch_check`, `records:STATUS_INVALID_IDENTITY` |
| 36 | 执行故障后的 fallback/recovery | `sparsity:fallback_policy`, `records:STATUS_FAIL_RECOVERY` |
| 37 | 对账预测与实测 | `sparsity:reconcile_prediction`, `sparsity:coverage_and_amdahl` |
| 38 | 在 holdout shape/workload 确认 | `frontier:WorkloadStrata.holdout_id`, `sparsity:shape_scan` |
| 39 | 跨 run/设备条件重复 | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan.minimum_repeats` |
| 40 | 形成 F4 AdoptionDecision | `sparsity:f4_adoption`, `contracts:AdoptionDecision.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-F4`

覆盖模块：`campaign`, `contracts`, `frontier`, `identity`, `parity`, `records`, `sparsity`

### E14-06 VLM、Diffusion/DiT 或 Audio 的跨模型形态 Profiling

claim boundary: 只证明一个模型/任务/平台上的方法迁移，不声称通用多模态平台；未执行时必须标 N/A_BY_ADR，并在文档中同步缩小能力声明（E14-06 §11）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 用 ADR 选择一种模型形态 | `multimodal:ModalityChoice`, `records:MULTIMODAL_MODALITIES` |
| 2 | 冻结能力声明 | `multimodal:check_capability_claim`, `multimodal:CLAIM_BOUNDARY` |
| 3 | 冻结 ModelArtifact | `multimodal:ModelComponentManifest`, `multimodal:COMPONENT_ROLES` |
| 4 | 冻结输入制品 | `multimodal:InputArtifactManifest`, `multimodal:MEDIA_PARAMETERS` |
| 5 | 冻结任务质量 oracle | `telemetry:MODALITY_METRICS`, `multimodal:ModalityTraceMapping.c6_metric` |
| 6 | 冻结阶段状态机 | `multimodal:PhaseStateMachine`, `multimodal:PHASES` |
| 7 | 扩展 WorkloadSpec | `multimodal:WorkloadContractExtension`, `multimodal:WORKLOAD_EXTENSION_FIELDS` |
| 8 | 扩展结果与 Trace 映射 | `multimodal:ModalityTraceMapping`, `telemetry:C6_EXTENSION_FIELDS` |
| 9 | 冻结 baseline 与单一候选变化 | `frontier:BaselinePair`, `frontier:BaselinePair.intended_difference` |
| 10 | 冻结 workload 矩阵 | `frontier:WorkloadStrata`, `multimodal:WORKLOAD_EXTENSION_FIELDS` |
| 11 | 运行 E14-01 依赖隔离验证 | `dependencies:probe_extra`, `dependencies:analyse_import_trace` |
| 12 | 运行 capability/actual-path probe | `contracts:check_actual_path_recorded`, `records:STATUS_INVALID_IDENTITY` |
| 13 | 验证原始输入 decode | `multimodal:InputArtifactManifest`, `multimodal:verify_preprocess` |
| 14 | 验证预处理语义 | `multimodal:verify_preprocess`, `multimodal:InputArtifactManifest.parameters` |
| 15 | 建立高精度/reference 输出 | `parity:REFERENCE_PATHS`, `parity:evaluate_gates` |
| 16 | 验证确定性/随机性 | `multimodal:determinism_report`, `identity:SeedBundle` |
| 17 | 运行 correctness/quality baseline | `multimodal:ModalityTraceMapping.c6_metric`, `contracts:check_quality_before_performance` |
| 18 | 采集框架级阶段 profile | `multimodal:phase_timeline`, `multimodal:TIMING_STAGES` |
| 19 | 采集 operator/shape profile | `multimodal:operator_shape_profile`, `records:PROFILE_LAYER_FIELDS` |
| 20 | 采集 kernel/system profile | `multimodal:operator_shape_profile`, `records:PROFILE_LAYERS` |
| 21 | 建立显存生命周期 | `multimodal:memory_lifecycle`, `contracts:check_resource_ledger` |
| 22 | 建立 baseline Roofline/Amdahl | `multimodal:roofline_amdhahl`, `frontier:PredictionModel` |
| 23 | 实现/启用候选路径 | `dependencies:FeatureFlagRegistry.resolve`, `frontier:BaselinePair` |
| 24 | 验证 candidate actual path | `multimodal:roofline_amdhahl`, `contracts:check_actual_path_recorded` |
| 25 | 运行中间 tensor correctness | `parity:evaluate_gates`, `multimodal:verify_preprocess` |
| 26 | 运行 candidate 质量门 | `contracts:check_quality_before_performance`, `multimodal:ModalityTraceMapping` |
| 27 | 运行单样本阶段 benchmark | `multimodal:phase_timeline`, `frontier:TimingBoundaries` |
| 28 | 运行 batch/shape 扫描 | `frontier:WorkloadStrata`, `multimodal:operator_shape_profile` |
| 29 | 运行并发/服务 workload | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan` |
| 30 | 验证流式语义（若适用） | `multimodal:streaming_semantics`, `records:N_A_BY_ADR` |
| 31 | 测冷启动和模型切换 | `multimodal:cold_start`, `multimodal:ModelComponentManifest` |
| 32 | 测内存/能耗/成本 | `multimodal:RESOURCE_KEYS`, `contracts:check_resource_ledger` |
| 33 | 注入坏输入 | `multimodal:bad_input_rejection`, `campaign:isolation_clause` |
| 34 | 注入缺子模型/错 preprocessor | `multimodal:missing_component_check`, `records:STATUS_INVALID_IDENTITY` |
| 35 | 验证取消/超时/资源恢复 | `multimodal:cancel_release`, `records:STATUS_FAIL_RECOVERY` |
| 36 | 执行候选消融 | `multimodal:candidate_ablation`, `frontier:AblationMatrix` |
| 37 | 对账预测与实测 | `multimodal:reconcile_prediction`, `multimodal:roofline_amdhahl` |
| 38 | 在 holdout 输入确认 | `frontier:WorkloadStrata.holdout_id`, `frontier:StatisticsPlan` |
| 39 | 跨 run/time block 重复 | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan.minimum_repeats` |
| 40 | 形成 MultimodalAdoptionDecision | `multimodal:multimodal_adoption`, `multimodal:check_report_completeness`, `contracts:AdoptionDecision.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-06`

覆盖模块：`campaign`, `contracts`, `dependencies`, `frontier`, `identity`, `multimodal`, `parity`, `records`, `telemetry`

### E14-07 Agent Tool/Memory/异步 Workflow 的等待、失败与 Trace

claim boundary: 只作为 Serving/Trace 压测扩展，不取代 kernel/Runtime 主线，也不升级为通用 Agent 平台 claim；不执行时标 N/A_BY_ADR（E14-07 §13）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 用 ADR 决定是否执行 | `frontier:issue_contract`, `agent:CLAIM_BOUNDARY` |
| 2 | 冻结一个最小任务族 | `agent:TaskFamily`, `agent:TaskFamily.deterministic_termination` |
| 3 | 冻结工具清单与安全级别 | `agent:ToolContract`, `agent:TOOL_SAFETY_LEVELS` |
| 4 | 冻结 tool schema | `agent:ToolContract.parameters`, `agent:PARAMETER_TYPES` |
| 5 | 冻结 workflow 状态机 | `agent:WorkflowStateMachine`, `records:WORKFLOW_TRANSITIONS` |
| 6 | 冻结 task success oracle | `agent:SuccessOracle`, `agent:score_task` |
| 7 | 冻结模型与 prompt artifact | `contracts:PolicySnapshot`, `identity:digest_text` |
| 8 | 冻结 memory 语义 | `agent:MemoryContract`, `agent:memory_consistency` |
| 9 | 冻结 retry/idempotency 语义 | `agent:RetryPolicy`, `agent:ERROR_CLASSES` |
| 10 | 冻结并发/负载矩阵 | `frontier:WorkloadStrata`, `agent:concurrency_sweep` |
| 11 | 冻结 trace schema 与隐私 | `agent:TraceSchema`, `agent:FORBIDDEN_LABEL_FIELDS` |
| 12 | 冻结资源与中止预算 | `agent:run_budget`, `agent:BUDGET_DIMENSIONS` |
| 13 | 建立确定性 fake-tool baseline | `agent:ToolContract.safety_level`, `agent:phase_baseline` |
| 14 | 验证合法 tool-call parse/validation | `agent:validate_tool_call`, `agent:ToolContract.required` |
| 15 | 验证非法/越权 tool-call | `agent:validate_tool_call`, `agent:DANGEROUS_ARGUMENT_TOKENS` |
| 16 | 运行单任务正确性 baseline | `agent:score_task`, `agent:WorkflowStateMachine` |
| 17 | 验证 trace parent/link 结构 | `agent:validate_trace_structure`, `agent:SPAN_KINDS` |
| 18 | 验证时间轴和 critical path | `agent:reconstruct_critical_path`, `agent:TraceSchema.span_names` |
| 19 | 测 instrumentation overhead | `agent:instrumentation_overhead`, `agent:TraceSchema.cardinality_bound` |
| 20 | 运行 model/tool 分阶段 baseline | `agent:phase_baseline`, `records:PROFILE_LAYER_FIELDS` |
| 21 | 运行工具延迟扫描 | `agent:latency_sweep`, `agent:tool_share`, `agent:model_share` |
| 22 | 运行 fan-out/fan-in 扫描 | `agent:fanout_sweep`, `agent:SPAN_KINDS` |
| 23 | 运行 Agent 并发扫描 | `agent:concurrency_sweep`, `frontier:StatisticsPlan` |
| 24 | 运行长短 workflow 混合 | `agent:mixed_workflow_fairness`, `frontier:StopRules` |
| 25 | 验证模型 batching 与 tool wait 解耦 | `agent:tool_wait_decoupling`, `posttraining:check_kv_reuse_identity` |
| 26 | 验证 memory 读写一致性 | `agent:memory_consistency`, `agent:MemoryContract.conflict_policy` |
| 27 | 注入 tool timeout | `agent:fault_cases`, `agent:check_fault_accounting` |
| 28 | 注入 rate limit/过载 | `agent:RetryPolicy.circuit_breaker_threshold`, `agent:check_fault_accounting` |
| 29 | 注入畸形/超大 tool result | `agent:validate_tool_result`, `agent:ToolContract.max_payload_bytes` |
| 30 | 注入模型/工具暂时性错误 | `agent:ERROR_CLASSES`, `agent:check_fault_accounting` |
| 31 | 注入 worker crash | `agent:check_fault_accounting`, `agent:WorkflowStateMachine.transition` |
| 32 | 执行 cancel/用户超时 | `agent:cancel_propagation`, `records:WORKFLOW_STATES` |
| 33 | 执行 retry storm 保护 | `agent:retry_storm_protection`, `agent:RetryPolicy.jitter` |
| 34 | 选择一个编排策略候选 | `frontier:BaselinePair`, `frontier:BaselinePair.intended_difference` |
| 35 | 运行 candidate correctness/quality | `agent:score_task`, `contracts:check_quality_before_performance` |
| 36 | 运行 candidate 服务 A/B | `agent:concurrency_sweep`, `frontier:StatisticsPlan` |
| 37 | 对账阶段模型与实测 | `agent:reconcile_phase_model`, `agent:phase_baseline` |
| 38 | 在 holdout workflow/故障确认 | `frontier:WorkloadStrata.holdout_id`, `agent:fault_cases` |
| 39 | 跨 run/time block 重复 | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan.minimum_repeats` |
| 40 | 形成 AgentWorkloadDecision | `agent:agent_workload_decision`, `contracts:AdoptionDecision.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-07`

覆盖模块：`agent`, `contracts`, `frontier`, `identity`, `posttraining`, `records`

### E14-08 Android/ARM/端侧 Runtime Adapter 与采用决策

claim boundary: 路线 A 只在真实设备有实测时成立；路线 B 只产生采用/不采用 ADR，**不得**产生任何设备性能或 adapter 支持 claim。优秀结果可以是证据充分的“不做”（E14-08 §12/§13）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 用 ADR 选择路线 A 或 B | `edge:RouteChoice`, `records:EDGE_ROUTES` |
| 2 | 冻结一个平台—runtime 组合 | `edge:PlatformRuntimeChoice`, `edge:PLATFORM_RUNTIMES` |
| 3 | 冻结端侧任务 | `edge:realtime_feasibility`, `frontier:BaselinePair` |
| 4 | 冻结设备身份 | `edge:DeviceFingerprint`, `edge:DEVICE_FINGERPRINT_FIELDS` |
| 5 | 冻结 ModelArtifact 与许可 | `contracts:ServingModelArtifact`, `edge:ConversionDag` |
| 6 | 冻结输入与质量门 | `edge:device_correctness`, `frontier:QualityGate` |
| 7 | 冻结性能/功耗边界 | `edge:TIMING_MODES`, `edge:cold_warm_sustained` |
| 8 | 冻结安全与系统边界 | `campaign:REQUIRED_ISOLATION`, `edge:RouteChoice.has_permission` |
| 9 | 建立 runtime capability map | `edge:RuntimeCapabilityMap`, `edge:RuntimeCapabilityMap.problems` |
| 10 | 建立 op coverage map | `edge:op_coverage_map`, `edge:OP_PLACEMENT` |
| 11 | 估计内存和包体 | `edge:estimate_memory_and_package`, `edge:PACKAGE_COMPONENTS` |
| 12 | 估计实时和能耗可行性 | `edge:realtime_feasibility`, `frontier:PredictionModel` |
| 13 | 执行 Go/No-Go 决策 | `edge:go_no_go`, `frontier:StopRules` |
| 14 | 建立转换 DAG（路线 A） | `edge:ConversionDag`, `edge:ConversionDag.NODE_KINDS` |
| 15 | 验证转换 inventory/shape | `edge:ConversionDag.dynamic_axes_preserved`, `edge:ConversionDag.state_io_declared` |
| 16 | 验证 host/reference correctness | `edge:ConversionDag.host_correctness_passed`, `parity:evaluate_gates` |
| 17 | 实现最小 C4 adapter（路线 A） | `edge:EdgeAdapterContract`, `edge:EdgeAdapterContract.REQUIRED_OPERATIONS` |
| 18 | 构建目标应用/二进制 | `edge:package_and_startup`, `identity:file_inventory` |
| 19 | 部署到清洁目标设备 | `edge:DeviceFingerprint.evidence_level`, `campaign:isolation_clause` |
| 20 | 运行设备 capability probe | `edge:partition_placement`, `edge:RuntimeCapabilityMap.profiler_available` |
| 21 | 验证逐 op/partition placement | `edge:partition_placement`, `edge:OP_PLACEMENT` |
| 22 | 运行设备 correctness/quality | `edge:device_correctness`, `contracts:check_quality_before_performance` |
| 23 | 建立 CPU baseline | `edge:cpu_baseline_same_device`, `frontier:BaselinePair.shared_fields` |
| 24 | 测 cold load/compile/first inference | `edge:cold_warm_sustained`, `edge:TIMING_MODES` |
| 25 | 测 warm single-stream | `edge:cold_warm_sustained`, `edge:edge_execution_record` |
| 26 | 测 sustained thermal steady-state | `edge:thermal_steady_state`, `edge:cold_warm_sustained` |
| 27 | 测线程/亲和性扫描 | `edge:thread_affinity_scan`, `frontier:AblationMatrix` |
| 28 | 测 shape/context/batch 扫描 | `edge:shape_scan`, `frontier:WorkloadStrata` |
| 29 | 测内存生命周期 | `edge:memory_lifecycle`, `contracts:check_resource_ledger` |
| 30 | 测功率/能耗 | `edge:power_energy`, `records:REASON_CODES` |
| 31 | 测应用包体和启动体验 | `edge:package_and_startup`, `frontier:CostDenominator` |
| 32 | 测后台/系统扰动 | `edge:background_disturbance`, `frontier:StopRules` |
| 33 | 测试 unsupported op/shape | `edge:unsupported_op`, `edge:OP_PLACEMENT` |
| 34 | 测试损坏/错版本模型 | `edge:corrupted_model_check`, `records:STATUS_INVALID_IDENTITY` |
| 35 | 测试低内存和取消 | `edge:low_memory_cancel`, `records:STATUS_FAIL_RECOVERY` |
| 36 | 验证 core 依赖隔离 | `edge:core_isolation_check`, `dependencies:analyse_import_trace` |
| 37 | 形成路线 B 技术地图 | `edge:technology_map`, `edge:MAP_AXES` |
| 38 | 在 holdout input/steady run 确认 | `frontier:WorkloadStrata.holdout_id`, `edge:thermal_steady_state` |
| 39 | 跨安装/运行重复 | `records:EXPERIMENT_UNITS`, `frontier:StatisticsPlan.minimum_repeats` |
| 40 | 形成 EdgeAdoptionDecision | `edge:edge_adoption`, `contracts:AdoptionDecision.validate` |

驱动入口：`scripts/experimental/run_e14.py --experiment E14-08`

覆盖模块：`campaign`, `contracts`, `dependencies`, `edge`, `frontier`, `identity`, `parity`, `records`
