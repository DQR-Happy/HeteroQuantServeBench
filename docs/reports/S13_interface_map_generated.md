# S13 实验步骤 → 代码接口对照表（生成物，勿手工编辑）

> 生成命令：`.venv/bin/python scripts/infra/gen_interface_map.py`
> 校验命令：`.venv/bin/python scripts/infra/run_e13.py --interface-map --json`
> 本文件由 `hqsb.infra.interface_map` 从各实验模块的 `PROTOCOL_STEPS` 生成；
> `resolve_interfaces()` 对每个符号做导入校验，重命名会让审计失败而不是静默指向空。

总计：**11 个实验 / 418 步**；接口引用 638 处、唯一接口 343 个；解析状态 `ok=True`。

| 实验 | 级别 | 步骤数 | 驱动入口 | 覆盖能力（示例） |
|---|---|---|---|---|
| E13-01 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-01` | supply_chain:BuildQuestion, identity:REPRODUCIBILITY_LEVELS … |
| E13-02 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-02` | deployment:CleanScope, deployment:CLEAN_LEVELS … |
| E13-03 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-03` | scheduling:PlacementPlan, scheduling:NEGATIVE_CASES … |
| E13-04 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-04` | artifacts:CacheKey, artifacts:CACHE_KEY_FIELDS … |
| E13-05 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-05` | records:POD_STATE_MACHINE, records:REQUEST_STATE_MACHINE … |
| E13-06 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-06` | capacity:capacity_model_binding, capacity:prediction_residual … |
| E13-07 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-07` | autoscaling:CapacityModel, serving:slo.SLOSpec … |
| E13-08 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-08` | observability:LAYERS, observability:SemanticConvention … |
| E13-09 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-09` | faults:FaultSpec, faults:LAYERS … |
| E13-10 | P0 | 38 | `scripts/infra/run_e13.py --experiment E13-10` | canary:CandidateIdentity, identity:validate_release_change_scope … |
| E13-11 | P1 | 38 | `scripts/infra/run_e13.py --experiment E13-11` | security:ThreatModel … |

### E13-01 可重建容器、SBOM、漏洞/秘密/许可证与供应链 Gate

claim boundary: 本实验通过证明一个 release image 的来源与运行边界可信（digest/SBOM/扫描/非 root）；不证明该 release 已在集群完成模型服务、容量或可靠性验收。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 release build question | `supply_chain:BuildQuestion`, `identity:REPRODUCIBILITY_LEVELS` |
| 2 | 冻结 source identity | `identity:source_identity`, `identity:SourceIdentity` |
| 3 | 冻结构建输入闭包 | `supply_chain:BuildInputs`, `supply_chain:BuildInputs.validate` |
| 4 | 定义构建网络与 cache 策略 | `supply_chain:BuildInputs`, `supply_chain:PIN` |
| 5 | 设计 multi-stage 分层 | `supply_chain:LayerPlan`, `supply_chain:IMAGE_STAGES` |
| 6 | 定义 runtime users/permissions | `supply_chain:RuntimeSecurityContext` |
| 7 | 定义模型独立规则 | `records:MODEL_FILE_PATTERNS`, `supply_chain:ModelScanFinding` |
| 8 | 建立 build A 环境 | `supply_chain:BuildEnvironment` |
| 9 | 执行 build A | `supply_chain:BuildRecord` |
| 10 | 建立独立 build B 环境 | `supply_chain:BuildEnvironment` |
| 11 | 执行 build B | `supply_chain:BuildRecord` |
| 12 | 比较 OCI digest DAG | `identity:compare_oci_dag`, `identity:oci_digest_dag` |
| 13 | 执行 filesystem/package diff | `identity:canonical_filesystem_diff`, `identity:hash_directory` |
| 14 | 归因非确定性 | `identity:NONDETERMINISM_SOURCES`, `supply_chain:attribute_nondeterminism` |
| 15 | 修正并确认 reproducibility | `identity:reproducibility_verdict` |
| 16 | 生成 image SBOM | `supply_chain:Sbom`, `supply_chain:SbomComponent` |
| 17 | 生成 native artifact inventory | `supply_chain:native_artifact_inventory` |
| 18 | 验证 SBOM completeness | `supply_chain:sbom_completeness` |
| 19 | 执行 vulnerability scan | `supply_chain:VulnerabilityScan`, `supply_chain:VulnerabilityFinding` |
| 20 | 做 reachability/context triage | `supply_chain:VULN_STATUSES`, `supply_chain:VulnerabilityFinding.validate` |
| 21 | 建立 exception/expiry policy | `supply_chain:VulnerabilityException`, `supply_chain:scan_status_semantics` |
| 22 | 扫描 build context 秘密 | `supply_chain:scan_build_context`, `records:SECRET_SCAN_SURFACES` |
| 23 | 扫描 image layers/history/final FS | `supply_chain:scan_build_context`, `contracts:validate_public_payload` |
| 24 | 扫描模型/私有制品 | `supply_chain:scan_build_context`, `records:MODEL_FILE_PATTERNS` |
| 25 | 执行 license/notice scan | `supply_chain:license_scan`, `supply_chain:LicenseFinding` |
| 26 | 生成 build provenance | `supply_chain:ProvenanceStatement` |
| 27 | 签署/证明 release identity | `supply_chain:Attestation` |
| 28 | 验证 attestation 闭包 | `supply_chain:verify_attestation_closure` |
| 29 | 运行容器静态安全审计 | `supply_chain:RuntimeSecurityContext`, `supply_chain:DANGEROUS_CAPABILITIES` |
| 30 | 运行 non-root 启动与 smoke | `supply_chain:RuntimeSecurityContext.validate` |
| 31 | 验证 read-only/最小写入 | `supply_chain:validate_write_paths` |
| 32 | 验证最小 device/capability | `supply_chain:validate_device_capability_minimum` |
| 33 | 执行镜像大小/layer审计 | `supply_chain:image_size_inventory` |
| 34 | 注入供应链负例 | `supply_chain:run_negative_cases`, `supply_chain:NEGATIVE_CASES` |
| 35 | 注入篡改 | `supply_chain:run_negative_cases`, `identity:compare_oci_dag` |
| 36 | 执行 release gate | `supply_chain:release_gate`, `supply_chain:SupplyChainGateResult` |
| 37 | 独立验证与重建抽查 | `supply_chain:validate_independent_rebuild` |
| 38 | 生成 ReleaseBundle 供应链记录 | `supply_chain:release_bundle_record`, `identity:ReleaseBundle` |

### E13-02 空集群 Bootstrap、模型校验与 Cold→Ready→首个正确请求

claim boundary: 本实验通过证明自动部署与服务准入正确；不证明 placement 最优、容量安全、生命周期无丢请求、自动扩缩稳定或故障可恢复（由 E13-03…E13-09 验证）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 clean level 与验收边界 | `deployment:CleanScope`, `deployment:CLEAN_LEVELS` |
| 2 | 冻结 IaC/deployment source | `deployment:IaCSource`, `deployment:render_manifests` |
| 3 | 冻结 ReleaseBundle 与模型 | `identity:ReleaseBundle`, `deployment:bind_release` |
| 4 | 冻结 startup/quality/first-request contract | `deployment:StartupContract` |
| 5 | 冻结环境资源与预算 | `deployment:CleanScope`, `campaign:SafetyPolicy` |
| 6 | 建立 cluster baseline snapshot | `deployment:cluster_baseline` |
| 7 | 验证 clean preconditions | `deployment:evaluate_preconditions` |
| 8 | 创建隔离 namespace/project | `deployment:NamespacePolicy` |
| 9 | 安装/验证基础组件 | `deployment:InstalledComponent` |
| 10 | 验证 node/device capability | `scheduling:NodeInventory`, `scheduling:validate_allocatable` |
| 11 | 应用 storage/cache 资源 | `deployment:StorageRequest` |
| 12 | 注入最小 Secret/Config | `deployment:SecretReference` |
| 13 | 部署 observability collector | `observability:SemanticConvention` |
| 14 | 验证 image supply-chain admission | `supply_chain:admit_deployment_digest` |
| 15 | 创建 HQSB workload/service | `deployment:render_manifests`, `deployment:WorkloadSpec` |
| 16 | 记录 scheduling/pull | `deployment:DeploymentStageEvent` |
| 17 | 启动模型 artifact downloader | `artifacts:StagingDownload` |
| 18 | 校验下载结果 | `artifacts:verify_download` |
| 19 | 原子提交 verified cache | `artifacts:atomic_commit` |
| 20 | 启动 service process | `deployment:DeploymentStageEvent` |
| 21 | 加载模型与 runtime | `artifacts:load_and_measure` |
| 22 | 执行 compile/engine/graph 初始化 | `artifacts:resolve_engine` |
| 23 | 执行 warmup | `artifacts:WarmupResult` |
| 24 | 执行 startup correctness/quality probe | `deployment:QualityProbe` |
| 25 | 验证 readiness transition | `deployment:readiness_verdict`, `deployment:ReadinessClaim` |
| 26 | 验证 pre-ready traffic exclusion | `deployment:pre_ready_traffic_exclusion` |
| 27 | 执行首个正式请求 | `deployment:FirstRequestRecord` |
| 28 | 核对首请求正确性 | `deployment:FirstRequestRecord.validate` |
| 29 | 验证资源/身份闭合 | `deployment:identity_closure` |
| 30 | 执行 warm-cache redeploy | `deployment:cold_warm_stage_times` |
| 31 | 执行 cold-cache repeat | `deployment:cold_warm_stage_times` |
| 32 | 注入损坏模型 | `deployment:run_negative_deployment_cases` |
| 33 | 注入下载中断/慢存储 | `deployment:run_negative_deployment_cases`, `artifacts:interrupted_download_state` |
| 34 | 注入不兼容 runtime/device | `deployment:run_negative_deployment_cases`, `deployment:validate_compatibility` |
| 35 | 执行失败 cleanup | `deployment:evaluate_failure_cleanup`, `deployment:RESIDUAL_KINDS` |
| 36 | 重复独立 bootstrap | `deployment:ManualIntervention`, `deployment:automation_verdict` |
| 37 | 构建阶段时延/瓶颈分析 | `deployment:cold_warm_stage_times`, `deployment:deployment_timeline` |
| 38 | 生成 clean deployment verdict | `deployment:deployment_verdict` |

### E13-03 Accelerator 资源调度、Capability/NUMA/Topology 与隔离

claim boundary: 本实验通过证明资源落位与隔离符合声明；不证明模型生命周期、容量控制、autoscaling 或多租户整体安全（由 E13-04/E13-06/E13-07/E13-11 验证）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 placement questions | `scheduling:PlacementPlan`, `scheduling:NEGATIVE_CASES` |
| 2 | 冻结 cluster/device inventory | `scheduling:NodeInventory`, `scheduling:DeviceRecord` |
| 3 | 定义 capability label schema | `scheduling:CapabilityLabel`, `scheduling:LABEL_CLASSES` |
| 4 | 保护可信 labels | `scheduling:validate_label_provenance`, `scheduling:PROTECTED_LABEL_CLASSES` |
| 5 | 冻结 device plugin/DRA 配置 | `scheduling:DevicePluginConfig` |
| 6 | 冻结 kubelet topology/CPU/memory policy | `scheduling:NodeInventory`, `scheduling:TOPOLOGY_POLICIES` |
| 7 | 建立 PlacementPlan | `scheduling:PlacementPlan` |
| 8 | 验证 allocatable inventory | `scheduling:validate_allocatable` |
| 9 | 运行单设备合法 placement | `scheduling:filter_nodes`, `scheduling:SchedulerEvent` |
| 10 | 验证容器内 device 映射 | `scheduling:PlacementEvidence` |
| 11 | 验证 actual backend 使用 | `scheduling:PlacementEvidence.validate` |
| 12 | 验证 CPU/cpuset/NUMA | `scheduling:TopologyRecord` |
| 13 | 运行 locality A/B | `scheduling:locality_ab` |
| 14 | 验证 Topology Manager 行为 | `scheduling:TopologyPolicyResult` |
| 15 | 运行多设备合法 placement | `scheduling:validate_rank_mapping` |
| 16 | 验证 P2P/link topology | `scheduling:P2PResult` |
| 17 | 运行 topology A/B | `scheduling:locality_ab`, `scheduling:LOCALITY_METRICS` |
| 18 | 验证 anti-affinity/spread | `scheduling:anti_affinity_check` |
| 19 | 验证 partition profile | `scheduling:PARTITION_PROFILES`, `scheduling:DeviceRecord` |
| 20 | 验证 exclusive 模式 | `scheduling:isolation_verdict` |
| 21 | 验证 shared/time-slicing 模式 | `scheduling:isolation_verdict`, `scheduling:SHARING_MODES` |
| 22 | 测邻居干扰 | `scheduling:neighbor_interference` |
| 23 | 测试 wrong vendor/arch | `scheduling:run_negative_placements` |
| 24 | 测试 memory/partition不足 | `scheduling:run_negative_placements` |
| 25 | 测试 taint/toleration错误 | `scheduling:run_negative_placements` |
| 26 | 测试缺失/伪造 label | `scheduling:run_negative_placements`, `scheduling:validate_label_provenance` |
| 27 | 测试 unhealthy device | `scheduling:run_negative_placements`, `scheduling:validate_allocatable` |
| 28 | 测试 device plugin restart | `scheduling:plugin_restart_recovery` |
| 29 | 测试 rank/device mismatch | `scheduling:validate_rank_mapping` |
| 30 | 测试跨 namespace 越权 | `scheduling:unauthorized_device_access` |
| 31 | 验证资源释放 | `scheduling:validate_resource_release` |
| 32 | 验证 placement observability | `scheduling:PlacementEvidence`, `observability:SemanticConvention` |
| 33 | 计算 scheduling 指标 | `scheduling:scheduling_metrics` |
| 34 | 计算 locality/topology效应 | `scheduling:locality_ab` |
| 35 | 计算隔离/共享指标 | `scheduling:isolation_verdict`, `scheduling:unauthorized_device_access` |
| 36 | 执行 policy regression | `scheduling:policy_regression` |
| 37 | 独立复验 | `scheduling:validate_resource_release` |
| 38 | 形成 placement/isolation verdict | `scheduling:placement_verdict` |

### E13-04 模型制品下载、Cache、原子激活、版本切换与回滚

claim boundary: 本实验通过证明模型制品生命周期与版本一致性；不证明 rolling 请求无损、canary 判断正确或跨区域灾备（由 E13-05/E13-10 验证）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 artifact identity contract | `artifacts:CacheKey`, `artifacts:CACHE_KEY_FIELDS` |
| 2 | 冻结生命周期状态机 | `records:ARTIFACT_STATE_MACHINE`, `artifacts:ArtifactLifecycleEvent.validate` |
| 3 | 冻结 cache layout | `artifacts:cache_layout_manifest`, `artifacts:CACHE_AREAS` |
| 4 | 冻结 activation semantics | `artifacts:ActivationGeneration` |
| 5 | 冻结 rollback policy | `artifacts:RollbackPolicy` |
| 6 | 冻结 workload/quality contract | `deployment:QualityProbe`, `artifacts:assert_servable` |
| 7 | 盘点存储与原子能力 | `artifacts:StorageCapability` |
| 8 | 清理任务专用 cold cache | `artifacts:cache_pre_inventory` |
| 9 | 执行单副本 cold download | `artifacts:StagingDownload` |
| 10 | 执行逐文件与聚合校验 | `artifacts:verify_download`, `artifacts:FileVerification` |
| 11 | 执行 compatibility 校验 | `artifacts:check_compatibility`, `artifacts:COMPATIBILITY_CHECKS` |
| 12 | 原子提交 verified entry | `artifacts:atomic_commit`, `artifacts:AtomicCommit` |
| 13 | 加载并记录内存 | `artifacts:load_and_measure` |
| 14 | 编译/解析 engine artifact | `artifacts:resolve_engine` |
| 15 | 执行 warmup/quality/capacity probe | `artifacts:WarmupResult`, `deployment:QualityProbe` |
| 16 | 激活 baseline 版本 A | `artifacts:ActivationGeneration`, `artifacts:assert_servable` |
| 17 | 执行 warm cache restart | `artifacts:cache_layout_manifest` |
| 18 | 执行并发副本下载 | `artifacts:single_flight_outcome` |
| 19 | 执行并发不同版本下载 | `artifacts:CacheKey.key`, `artifacts:single_flight_outcome` |
| 20 | 准备版本 B | `artifacts:atomic_commit`, `artifacts:assert_loadable` |
| 21 | 运行切换前对照流量 | `artifacts:ActivationGeneration` |
| 22 | 触发 A→B 原子切换 | `artifacts:ActivationGeneration.validate` |
| 23 | 验证新旧请求版本一致性 | `artifacts:validate_request_version_consistency` |
| 24 | 验证 KV/prefix cache 隔离 | `artifacts:validate_kv_prefix_isolation` |
| 25 | 验证切换质量/性能 | `deployment:QualityProbe`, `capacity:AdmissionDecision` |
| 26 | 触发 B→A 回滚 | `artifacts:RollbackPolicy`, `artifacts:rollback_timeline` |
| 27 | 验证回滚闭环 | `artifacts:rollback_timeline` |
| 28 | 注入缺失/篡改 shard | `artifacts:run_fault_cases`, `artifacts:verify_download` |
| 29 | 注入 tokenizer/config/quant mismatch | `artifacts:run_fault_cases`, `artifacts:check_compatibility` |
| 30 | 注入下载中断/进程崩溃 | `artifacts:interrupted_download_state` |
| 31 | 注入存储超时/陈旧读取 | `artifacts:run_fault_cases` |
| 32 | 注入磁盘满/水位 | `artifacts:run_fault_cases`, `artifacts:gc_plan` |
| 33 | 测试 lock owner死亡 | `artifacts:LeaseRecord` |
| 34 | 测试 GC 与保留 | `artifacts:gc_plan`, `artifacts:gc_safety_check` |
| 35 | 测试并发 GC/load/switch | `artifacts:gc_safety_check`, `artifacts:single_flight_outcome` |
| 36 | 重复跨进程/节点 | `artifacts:CacheKey.key`, `artifacts:single_flight_outcome` |
| 37 | 验证 lineage 与审计 | `artifacts:ArtifactLifecycleEvent`, `telemetry:project_request_lifecycle_event` |
| 38 | 形成 artifact lifecycle verdict | `artifacts:artifact_lifecycle_verdict` |

### E13-05 Startup/Readiness/Liveness、Graceful Drain 与滚动重启

claim boundary: 本实验通过证明生命周期符合声明；不证明容量模型、autoscaler、故障矩阵或 canary 判定本身正确（由 E13-06…E13-10 验证）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 lifecycle state machine | `records:POD_STATE_MACHINE`, `records:REQUEST_STATE_MACHINE` |
| 2 | 冻结 probe semantics | `lifecycle:ProbeConfig`, `lifecycle:PROBE_SEMANTICS` |
| 3 | 冻结 termination/drain policy | `lifecycle:TerminationPolicy`, `lifecycle:TERMINATION_STAGES` |
| 4 | 冻结 retry/idempotency policy | `lifecycle:DECLARED_COMPLETION_SEMANTICS`, `lifecycle:TerminationPolicy` |
| 5 | 冻结 rolling policy | `lifecycle:RollingPolicy` |
| 6 | 部署基线 ReleaseBundle | `deployment:render_manifests`, `deployment:readiness_verdict` |
| 7 | 部署端到端事件采集 | `observability:SemanticConvention`, `telemetry:project_request_lifecycle_event` |
| 8 | 测 probe 自身开销 | `lifecycle:probe_overhead`, `lifecycle:ProbeResult` |
| 9 | 验证正常 startup | `lifecycle:ProbeResult`, `deployment:DeploymentStageEvent` |
| 10 | 注入慢启动 | `lifecycle:run_negative_lifecycle_cases` |
| 11 | 注入启动卡死 | `lifecycle:run_negative_lifecycle_cases` |
| 12 | 验证 readiness 语义 | `lifecycle:READINESS_ONLY_REASONS`, `deployment:readiness_verdict` |
| 13 | 验证 liveness 语义 | `lifecycle:run_negative_lifecycle_cases`, `contracts:validate_liveness_policy` |
| 14 | 测试 readiness flapping | `lifecycle:readiness_flapping` |
| 15 | 建立稳定负载 | `serving:slo.SLOSpec`, `lifecycle:TokenIntegrity` |
| 16 | 触发单 Pod SIGTERM/delete | `lifecycle:DrainTimeline` |
| 17 | 验证先停止接新请求 | `contracts:validate_drain_semantics`, `lifecycle:DrainTimeline.validate` |
| 18 | 验证短 inflight 完成 | `lifecycle:TokenIntegrity` |
| 19 | 验证长 streaming 请求 | `lifecycle:DrainTimeline`, `lifecycle:TerminationPolicy.validate` |
| 20 | 验证慢客户端/backpressure | `lifecycle:slow_client_policy_check` |
| 21 | 验证客户端取消/断连 | `lifecycle:client_cancel_accounting` |
| 22 | 验证 preStop 失败/重复 | `lifecycle:run_negative_lifecycle_cases` |
| 23 | 验证 SIGTERM handler | `lifecycle:DrainTimeline.validate` |
| 24 | 验证 forced SIGKILL | `lifecycle:forced_kill_accounting` |
| 25 | 检查终止后资源 | `lifecycle:resource_release_report`, `lifecycle:RELEASE_TARGETS` |
| 26 | 运行同版本 rolling restart | `lifecycle:rolling_availability` |
| 27 | 运行版本 rolling upgrade | `lifecycle:rolling_availability`, `artifacts:activate_version` |
| 28 | 验证 PDB/maxUnavailable | `lifecycle:RollingPolicy.validate`, `lifecycle:rolling_availability` |
| 29 | 验证 capacity during rollout | `lifecycle:rolling_availability`, `capacity:AdmissionDecision` |
| 30 | 注入新 Pod 启动失败 | `lifecycle:run_negative_lifecycle_cases` |
| 31 | 测试 rollout progress deadline | `lifecycle:validate_progress_deadline` |
| 32 | 测试连续滚动/并发 drain 保护 | `lifecycle:run_negative_lifecycle_cases`, `lifecycle:RollingPolicy` |
| 33 | 计算请求完整性 | `lifecycle:request_integrity_report`, `lifecycle:retry_duplicate_accounting` |
| 34 | 计算生命周期指标 | `lifecycle:lifecycle_metrics` |
| 35 | 执行跨时段重复 | `lifecycle:lifecycle_metrics`, `lifecycle:DIFF` |
| 36 | 执行 probe/timeout 敏感性 | `lifecycle:probe_sensitivity` |
| 37 | 验证 runbook 自动性 | `deployment:ManualIntervention`, `observability:AlertRule` |
| 38 | 形成 lifecycle verdict | `lifecycle:lifecycle_verdict` |

### E13-06 S12 容量模型、Admission、Token/KV Budget 与 OOM Margin 验证

claim boundary: 本实验通过证明容量与准入在声明 workload 内安全；不证明 autoscaler 能及时增加容量，也不证明节点/网络/存储故障恢复（由 E13-07/E13-09 验证）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结容量模型版本 | `capacity:capacity_model_binding`, `capacity:prediction_residual` |
| 2 | 冻结可用内存边界 | `capacity:memory_ledger`, `capacity:MEMORY_COMPONENTS` |
| 3 | 冻结 admission policies | `capacity:AdmissionPolicy`, `capacity:ADMISSION_POLICIES` |
| 4 | 冻结安全扫参计划 | `capacity:SweepPlan`, `capacity:SWEEP_STAGES` |
| 5 | 冻结 workload 与 quality/SLO | `capacity:token_work`, `serving:slo.SLOSpec` |
| 6 | 建立内存观测账本 | `capacity:MemoryComponent` |
| 7 | 测空 Pod/runtime baseline | `capacity:memory_ledger` |
| 8 | 测模型 loaded baseline | `capacity:memory_ledger`, `artifacts:load_and_measure` |
| 9 | 测 warmup 增量 | `capacity:memory_ledger`, `artifacts:WarmupResult` |
| 10 | 校准单请求 KV 增量 | `capacity:KVModel`, `capacity:KVModel.incremental_bytes` |
| 11 | 校准 batch/workspace 增量 | `capacity:MemoryComponent` |
| 12 | 校准 prefill/decode 相位 | `capacity:MemoryComponent`, `capacity:KVModel` |
| 13 | 验证多 rank 内存模型 | `capacity:memory_ledger` |
| 14 | 运行低负载 policy 基线 | `capacity:admit` |
| 15 | 运行 homogeneous 安全扫参 | `capacity:SweepPlan`, `capacity:admit` |
| 16 | 运行 mixed-length 扫参 | `capacity:token_work`, `capacity:admit` |
| 17 | 运行 long-context 边界 | `capacity:KVModel.incremental_bytes` |
| 18 | 运行 decode-heavy 边界 | `capacity:token_work` |
| 19 | 运行 open-loop 过载 | `capacity:admit`, `autoscaling:StaleMetricPolicy` |
| 20 | 比较 admission policies | `capacity:admission_policy_comparison` |
| 21 | 验证 decision linearization | `capacity:reservation_atomicity` |
| 22 | 验证 prefix cache 影响 | `capacity:KVModel.incremental_bytes` |
| 23 | 验证 speculative decoding 影响 | `capacity:BUDGET_FEATURES` |
| 24 | 验证 chunked prefill/batching 影响 | `capacity:BUDGET_FEATURES` |
| 25 | 验证 KV quant/precision 影响 | `capacity:KVModel` |
| 26 | 注入取消/超时 | `capacity:cancel_release` |
| 27 | 注入请求失败/backend 错误 | `capacity:cancel_release` |
| 28 | 构造 allocator fragmentation 历史 | `capacity:fragmentation_history` |
| 29 | 验证模型切换/双驻留 | `capacity:dual_residency_budget` |
| 30 | 执行 over-budget 负例 | `capacity:run_capacity_fault_cases` |
| 31 | 执行受控 OOM 故障 | `capacity:run_capacity_fault_cases` |
| 32 | 计算 prediction residual | `capacity:prediction_residual` |
| 33 | 估计 false admit/reject | `capacity:false_decisions` |
| 34 | 选择/验证 safety margin | `capacity:margin_holdout_validation` |
| 35 | 验证预算恢复/长期稳定性 | `capacity:cancel_release`, `capacity:reservation_atomicity` |
| 36 | 回流 S12 容量模型 | `capacity:s12_feedback` |
| 37 | 生成 S13-07 capacity 接口 | `capacity:autoscaling_capacity_interface` |
| 38 | 形成 capacity/admission verdict | `capacity:capacity_verdict` |

### E13-07 SLO/Queue/Token 多指标 Autoscaling、阶跃/突发/降载与控制稳定性

claim boundary: 本实验通过证明弹性控制在声明 workload 内稳定；不证明底层故障都能恢复，也不证明新 release 一定安全（由 E13-09/E13-10 验证）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 SLO/容量/成本目标 | `autoscaling:CapacityModel`, `serving:slo.SLOSpec` |
| 2 | 冻结策略与调参预算 | `autoscaling:POLICIES`, `autoscaling:EpisodePlan` |
| 3 | 冻结负载 episode | `autoscaling:EpisodePlan`, `records:AUTOSCALING_EPISODES` |
| 4 | 冻结集群/节点条件 | `autoscaling:ClusterConditions` |
| 5 | 部署 autoscaling telemetry | `observability:SemanticConvention`, `autoscaling:MetricSample` |
| 6 | 验证 metric 语义/单位 | `autoscaling:METRIC_UNITS`, `autoscaling:MetricSample.validate` |
| 7 | 测 metric age 与延迟 | `autoscaling:classify_metric`, `autoscaling:StaleMetricPolicy` |
| 8 | 验证 desired replica 公式 | `autoscaling:desired_replicas` |
| 9 | 部署 STATIC baseline | `autoscaling:cost_accounting` |
| 10 | 部署 CPU_ONLY baseline | `autoscaling:POLICIES` |
| 11 | 部署 ACCELERATOR_UTIL 策略 | `autoscaling:POLICIES` |
| 12 | 部署 QUEUE_REQUESTS 策略 | `autoscaling:POLICIES` |
| 13 | 部署 QUEUE_TOKEN_WORK 策略 | `autoscaling:LEADING_SIGNALS` |
| 14 | 部署 MULTI_SIGNAL 策略 | `autoscaling:decide` |
| 15 | 校准 scale-to-ready 分布 | `deployment:cold_warm_stage_times`, `autoscaling:control_loop_delay` |
| 16 | 运行 steady low episode | `autoscaling:decide`, `autoscaling:stabilize` |
| 17 | 运行 steady high episode | `autoscaling:decide` |
| 18 | 运行 step-up episode | `autoscaling:control_metrics` |
| 19 | 运行 short-burst episode | `autoscaling:control_loop_delay` |
| 20 | 运行 ramp-up episode | `autoscaling:enforce_rate_limit` |
| 21 | 运行 periodic/repeated burst | `autoscaling:control_metrics` |
| 22 | 运行 mixed-length episode | `autoscaling:desired_replicas` |
| 23 | 运行 step-down episode | `autoscaling:stabilize`, `lifecycle:DrainTimeline` |
| 24 | 验证 scale-down 候选选择 | `autoscaling:scale_down_candidate_selection` |
| 25 | 验证 E13-06 admission 协同 | `autoscaling:coordination_checks`, `capacity:admit` |
| 26 | 验证 E13-05 lifecycle 协同 | `autoscaling:coordination_checks`, `lifecycle:rolling_availability` |
| 27 | 验证 E13-04 cache 协同 | `autoscaling:coordination_checks`, `artifacts:single_flight_outcome` |
| 28 | 测试 metric missing/stale | `autoscaling:run_failure_cases`, `autoscaling:classify_metric` |
| 29 | 测试 controller restart/leader change | `autoscaling:run_failure_cases` |
| 30 | 测试 unschedulable/device shortage | `autoscaling:run_failure_cases`, `scheduling:filter_nodes` |
| 31 | 测试慢 artifact/model startup | `autoscaling:run_failure_cases`, `autoscaling:control_loop_delay` |
| 32 | 测试 rollout 与 autoscaling 并发 | `autoscaling:coordination_checks`, `lifecycle:RollingPolicy` |
| 33 | 计算控制性能 | `autoscaling:control_metrics` |
| 34 | 计算服务与成本 | `autoscaling:cost_accounting` |
| 35 | 做参数敏感性 | `autoscaling:sensitivity_sweep` |
| 36 | 在 holdout episodes 确认 | `autoscaling:evaluate_holdout` |
| 37 | 验证 runbook/alerts | `observability:build_alert_rule`, `autoscaling:run_failure_cases` |
| 38 | 形成 autoscaling verdict | `autoscaling:autoscaling_verdict` |

### E13-08 Request→Queue→Runtime→Accelerator→System 可观测性、告警与 RCA

claim boundary: 本实验通过证明系统可被观测与定位；不证明所有故障已恢复或发布策略正确（由 E13-09/E13-10 验证行动闭环）。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结观测问题/责任层 | `observability:LAYERS`, `observability:SemanticConvention` |
| 2 | 冻结 semantic schema | `observability:SemanticConvention.validate`, `observability:ATTRIBUTE_LEVELS` |
| 3 | 冻结隐私/基数 policy | `observability:validate_metric_labels`, `records:FORBIDDEN_METRIC_LABELS` |
| 4 | 冻结 sampling policy | `observability:SamplingPolicy`, `observability:ALWAYS_KEEP_CASES` |
| 5 | 冻结 SLI/SLO/alert 规则 | `observability:SLIDefinition`, `observability:AlertRule` |
| 6 | 建立 clock/time 基础 | `observability:ClockAlignment`, `observability:clock_verdict` |
| 7 | 接入 client/gateway spans | `observability:SemanticConvention` |
| 8 | 接入 auth/admission/routing spans | `observability:SemanticConvention`, `security:SecurityCase` |
| 9 | 接入 queue/scheduler/batch/KV | `observability:SemanticConvention`, `capacity:token_work` |
| 10 | 接入 runtime/model/backend | `observability:SemanticConvention`, `artifacts:CacheKey` |
| 11 | 接入 operator/device profile 关联 | `observability:profile_index` |
| 12 | 接入 system/storage/network | `observability:SemanticConvention` |
| 13 | 接入 accelerator telemetry | `observability:SemanticConvention`, `scheduling:DeviceRecord` |
| 14 | 接入 deployment/control events | `telemetry:project_release_identity`, `observability:SemanticConvention` |
| 15 | 验证 context propagation | `observability:trace_coverage` |
| 16 | 验证 metric 与 raw 对账 | `observability:reconcile_metric_with_raw` |
| 17 | 验证直方图 bucket | `observability:validate_histogram_buckets` |
| 18 | 测 cardinality | `observability:classify_cardinality` |
| 19 | 测 instrumentation overhead | `observability:instrumentation_overhead` |
| 20 | 测采样 coverage | `observability:sampling_coverage` |
| 21 | 建立正常 baseline dashboard | `observability:dashboard_definition` |
| 22 | 建立 alerts 与 runbooks | `observability:build_alert_rule` |
| 23 | 注入 queue/scheduler 慢请求 | `faults:FaultSpec`, `observability:rca_verdict` |
| 24 | 注入 runtime/kernel 退化 | `faults:FaultSpec`, `observability:rca_verdict` |
| 25 | 注入 device clock/thermal/power 异常 | `faults:FaultSpec`, `observability:rca_verdict` |
| 26 | 注入 CPU/memory/swap/IO 瓶颈 | `faults:FaultSpec`, `observability:rca_verdict` |
| 27 | 注入 network/storage 故障 | `faults:FaultSpec`, `observability:rca_verdict` |
| 28 | 注入 distributed straggler/link 问题 | `faults:FaultSpec`, `observability:rca_verdict` |
| 29 | 执行盲化慢请求 RCA | `observability:RCARecord` |
| 30 | 执行盲化故障 RCA | `observability:RCARecord`, `observability:rca_verdict` |
| 31 | 验证 alert 检测/误告 | `observability:alert_effectiveness` |
| 32 | 验证 telemetry 缺失检测 | `observability:telemetry_failure_detection` |
| 33 | 验证日志/trace 脱敏 | `observability:redaction_scan`, `contracts:redact_payload` |
| 34 | 执行 profile-on-demand | `observability:profile_index`, `faults:FaultSpec` |
| 35 | 重复不同负载/时段 | `observability:rca_verdict`, `observability:alert_effectiveness` |
| 36 | 复核 root-cause 证据 | `observability:rca_verdict` |
| 37 | 修订并独立确认 | `observability:confirmation_case` |
| 38 | 形成 observability verdict | `observability:observability_verdict` |

### E13-09 进程、Pod、节点、设备、Cache、存储、网络、制品、OOM、Thermal 故障注入与恢复

claim boundary: 本实验通过证明已测故障的真实行为；不证明系统对未建模故障、区域级灾难或恶意攻击自动可靠。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 fault taxonomy/matrix | `faults:FaultSpec`, `faults:LAYERS` |
| 2 | 冻结稳态假设 | `faults:SteadyState` |
| 3 | 冻结预期行为 | `faults:FaultSpec.validate`, `faults:DEGRADATIONS` |
| 4 | 冻结安全边界 | `campaign:SafetyPolicy`, `campaign:authorize_targets` |
| 5 | 冻结实验设计 | `faults:FaultEpisode`, `faults:validate_fault_matrix` |
| 6 | 验证 fault tool/mechanism | `faults:ground_truth` |
| 7 | 验证 observability/runbook 准备 | `faults:safety_gate`, `observability:build_alert_rule` |
| 8 | 建立 fault-free baseline | `faults:SteadyState`, `faults:reliability_metrics` |
| 9 | 注入 process crash | `faults:ground_truth`, `lifecycle:DrainTimeline` |
| 10 | 注入 process deadlock/no-progress | `faults:ground_truth`, `lifecycle:ProbeConfig` |
| 11 | 注入 Pod delete/eviction | `faults:ground_truth`, `lifecycle:RollingPolicy` |
| 12 | 注入 node/kubelet 不可用 | `faults:ground_truth`, `scheduling:validate_resource_release` |
| 13 | 注入 backend/runtime failure | `faults:ground_truth`, `faults:RetryFallback` |
| 14 | 注入 device unhealthy/plugin 故障 | `faults:ground_truth`, `scheduling:plugin_restart_recovery` |
| 15 | 注入 cache corruption | `faults:ground_truth`, `artifacts:verify_download` |
| 16 | 注入 cache lock/lease 故障 | `faults:ground_truth`, `artifacts:LeaseRecord` |
| 17 | 注入 model artifact 错误 | `faults:ground_truth`, `artifacts:check_compatibility` |
| 18 | 注入 object storage latency/error | `faults:ground_truth`, `artifacts:StagingDownload` |
| 19 | 注入 local/PVC disk full | `faults:ground_truth`, `artifacts:gc_plan` |
| 20 | 注入 client/service network 故障 | `faults:ground_truth`, `lifecycle:slow_client_policy_check` |
| 21 | 注入 DNS/service discovery 故障 | `faults:ground_truth` |
| 22 | 注入 distributed link/rank 故障 | `faults:ground_truth`, `scheduling:validate_rank_mapping` |
| 23 | 注入 device OOM | `faults:ground_truth`, `capacity:run_capacity_fault_cases` |
| 24 | 注入 host memory/swap pressure | `faults:ground_truth` |
| 25 | 注入 resource leak episode | `faults:ground_truth`, `capacity:margin_holdout_validation` |
| 26 | 注入 thermal/power/clock 退化 | `faults:ground_truth` |
| 27 | 注入 metrics/autoscaler failure | `faults:ground_truth`, `autoscaling:run_failure_cases` |
| 28 | 测试 retry storm 防护 | `faults:retry_storm_check`, `faults:RetryFallback` |
| 29 | 测试 fallback 语义 | `faults:RetryFallback.validate` |
| 30 | 运行一个 chained fault | `faults:ground_truth`, `faults:validate_fault_matrix` |
| 31 | 对每 case 执行恢复 | `faults:FaultEpisode` |
| 32 | 验证请求/质量恢复 | `faults:validate_state_recovery`, `lifecycle:request_integrity_report` |
| 33 | 验证资源/状态恢复 | `faults:validate_state_recovery`, `faults:RESOURCE_INVARIANTS` |
| 34 | 检查残留/延迟故障 | `faults:residual_watch` |
| 35 | 计算可靠性指标 | `faults:reliability_metrics` |
| 36 | 跨时段重复和敏感性 | `faults:reliability_metrics`, `faults:validate_fault_matrix` |
| 37 | 生成 postmortem 与修复验证 | `faults:postmortem` |
| 38 | 形成 fault/recovery verdict | `faults:fault_verdicts`, `faults:fault_recovery_verdict` |

### E13-10 Correctness/Performance Canary、自动 Gate、渐进发布与回滚

claim boundary: 本实验通过证明已测退化类别可由当前 canary policy 安全治理；不证明未知退化、极低频长尾或区域级发布风险被消除。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 release change taxonomy | `canary:CandidateIdentity`, `identity:validate_release_change_scope` |
| 2 | 冻结 canary 状态机 | `canary:CANARY_STATE_MACHINE`, `canary:PROMOTION_PATH` |
| 3 | 冻结业务风险预算 | `canary:CanaryPolicy`, `canary:exposure_budget` |
| 4 | 冻结 traffic assignment | `canary:AssignmentPolicy`, `canary:ASSIGNMENT_UNITS` |
| 5 | 冻结 metrics 和方向 | `canary:METRIC_DIRECTIONS`, `records:CANARY_GATES` |
| 6 | 冻结统计/序贯规则 | `canary:sequential_decision`, `canary:paired_interval` |
| 7 | 冻结 bad/good candidate 构造 | `canary:CANDIDATE_KINDS`, `canary:CandidateIdentity` |
| 8 | 验证 control 稳态 | `faults:SteadyState`, `observability:SLIDefinition` |
| 9 | 执行 G0 供应链/兼容 gate | `supply_chain:release_gate`, `canary:GateResult` |
| 10 | 执行 G1 离线 correctness | `canary:GateResult`, `deployment:QualityProbe` |
| 11 | 执行 G2 质量 gate | `canary:GateResult`, `deployment:QualityProbe` |
| 12 | 部署 candidate 不接流量 | `artifacts:ActivationGeneration`, `deployment:readiness_verdict` |
| 13 | 执行 shadow/synthetic smoke | `canary:GateResult`, `deployment:FirstRequestRecord` |
| 14 | 启动 Stage 1 小流量 | `canary:AssignmentPolicy.assign`, `canary:exposure_budget` |
| 15 | 验证 assignment 正确性 | `canary:AssignmentPolicy.assign` |
| 16 | 验证 control/canary 可比性 | `canary:case_mix_balance` |
| 17 | 运行 correctness/stream gate | `canary:GateResult`, `lifecycle:request_integrity_report` |
| 18 | 运行 error/availability gate | `canary:GateResult`, `serving:slo.SLOSpec` |
| 19 | 运行 latency/goodput gate | `canary:paired_interval`, `canary:GateResult` |
| 20 | 运行 resource/capacity gate | `canary:GateResult`, `capacity:memory_ledger` |
| 21 | 运行 thermal/power/cost 监测 | `canary:GateResult`, `autoscaling:cost_accounting` |
| 22 | 执行序贯 decision | `canary:sequential_decision`, `canary:CanaryDecisionRow` |
| 23 | 推进 exposure stages | `canary:exposure_budget`, `canary:decide_stage` |
| 24 | 协调 autoscaling | `autoscaling:coordination_checks` |
| 25 | 协调 cache/placement | `canary:case_mix_balance`, `scheduling:PlacementEvidence` |
| 26 | 运行 correctness-bad canary | `canary:evaluate_gates`, `canary:false_rate_accounting` |
| 27 | 运行 performance-bad canary | `canary:paired_interval`, `canary:false_rate_accounting` |
| 28 | 运行 resource-bad canary | `canary:false_rate_accounting`, `capacity:memory_ledger` |
| 29 | 运行 good/equivalent canary | `canary:false_rate_accounting` |
| 30 | 触发自动 rollback | `canary:rollback_closure`, `artifacts:rollback_timeline` |
| 31 | 验证 rollback 请求语义 | `canary:rollback_closure`, `lifecycle:request_integrity_report` |
| 32 | 验证 control 恢复 | `canary:rollback_closure`, `artifacts:rollback_timeline` |
| 33 | 检查 candidate 残留 | `canary:rollback_closure`, `artifacts:gc_safety_check` |
| 34 | 计算 gate 质量 | `canary:false_rate_accounting` |
| 35 | 做 threshold/sample 敏感性 | `canary:sensitivity_replay` |
| 36 | 独立 holdout canary | `canary:holdout_canary` |
| 37 | 验证人工 override/审计 | `canary:OverrideAudit` |
| 38 | 形成 release governance verdict | `canary:release_governance_verdict` |

### E13-11 多租户认证、授权、配额、Secrets、脱敏、网络隔离与滥用防护

claim boundary: 本实验通过只证明声明威胁模型内的控制有效；不等于抵抗所有侧信道、节点管理员攻击或获得安全认证。

| # | 步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结租户定义、范围和不支持项 | `security:ThreatModel` |
| 2 | 绘制资产、信任边界和数据流 | `security:ThreatModel`, `observability:SemanticConvention` |
| 3 | 冻结身份签发、传播、过期与撤销策略 | `security:CredentialPolicy` |
| 4 | 导出 RBAC 权限图并做最小权限审查 | `security:RBACPermission`, `security:rbac_graph_audit` |
| 5 | 检查 Namespace、ServiceAccount 与自动挂载策略 | `security:validate_tenant_map` |
| 6 | 冻结静态配额和动态业务预算 | `security:QuotaProfile`, `security:STATIC_QUOTA_RESOURCES` |
| 7 | 建立默认拒绝网络策略和最小允许清单 | `security:NetworkPolicySpec` |
| 8 | 冻结 Pod 运行时安全基线 | `security:PodSecurityBaseline`, `supply_chain:RuntimeSecurityContext` |
| 9 | 冻结 Secret 生命周期和泄露面 | `security:SecretLifecycle` |
| 10 | 冻结模型、适配器、缓存和存储命名空间 | `artifacts:CacheKey`, `security:ArtifactNamespace` |
| 11 | 冻结设备、分区和共享策略 | `scheduling:PlacementPlan`, `scheduling:SHARING_MODES` |
| 12 | 定义遥测访问、脱敏和审计字段 | `observability:SemanticConvention`, `security:AuditCoverage` |
| 13 | 从清洁环境创建 A、B、只读运维和攻击者主体 | `security:TenantIdentityMap`, `deployment:NamespacePolicy` |
| 14 | 运行每个主体的合法正向基线 | `security:positive_baseline` |
| 15 | 执行无效、过期、伪造、重放和撤销凭据测试 | `security:CredentialCase`, `security:rotation_recovery` |
| 16 | 穷举关键 Kubernetes API 越权动作 | `security:SecurityCase`, `security:K8S_VERBS` |
| 17 | 执行 ServiceAccount 与间接提权测试 | `security:rbac_graph_audit`, `security:RBAC_RISKS` |
| 18 | 尝试篡改 Namespace、Pod 与路由标签 | `security:SecurityCase`, `scheduling:validate_label_provenance` |
| 19 | 验证默认拒绝的真实网络效果 | `security:NetworkCase` |
| 20 | 验证允许网络最小且业务仍可运行 | `security:NetworkPolicySpec` |
| 21 | 验证网络插件、DNS 和服务网格的实际执行路径 | `security:NetworkPolicySpec`, `security:NetworkCase` |
| 22 | 搜索 Secret 在进程和文件系统中的暴露 | `security:secret_exposure_scan` |
| 23 | 执行 Secret 轮换、撤销和故障恢复 | `security:rotation_recovery` |
| 24 | 执行模型仓库、缓存和临时文件跨租户访问测试 | `security:SecurityCase`, `artifacts:CacheKey` |
| 25 | 验证模型身份、别名与路由不会串租户 | `security:SecurityCase`, `supply_chain:verify_attestation_closure` |
| 26 | 验证容器内设备可见性与跨租户设备访问 | `scheduling:unauthorized_device_access` |
| 27 | 测量共享加速器下的性能干扰并限定安全声明 | `security:noisy_neighbor` |
| 28 | 验证正常负载下的配额计量与拒绝语义 | `security:QuotaLedger`, `security:quota_settlement` |
| 29 | 执行并发配额竞态和超卖测试 | `security:quota_race_trials` |
| 30 | 执行超大输入、输出和非法参数滥用测试 | `security:abuse_trials`, `capacity:admit` |
| 31 | 执行慢客户端、流式连接囤积和主动断连测试 | `security:abuse_trials`, `lifecycle:slow_client_policy_check` |
| 32 | 执行重试风暴和错误放大测试 | `security:abuse_trials`, `faults:retry_storm_check` |
| 33 | 执行 noisy-neighbor、公平调度和优先级测试 | `security:noisy_neighbor`, `serving:slo.SLOSpec` |
| 34 | 验证自动扩缩容的租户归因和成本上限 | `security:QuotaProfile`, `autoscaling:cost_accounting` |
| 35 | 验证日志、指标、Trace、Profile 与错误响应的隔离和脱敏 | `security:telemetry_redaction_cases` |
| 36 | 验证审计事件的完整性、时序和可追责性 | `security:audit_coverage_report`, `security:AUDIT_EVENT_KINDS` |
| 37 | 恢复、清理并验证状态一致性 | `security:rotation_recovery`, `faults:validate_state_recovery` |
| 38 | 重复、统计、裁决并限定声明 | `security:invariant_verdicts`, `security:multitenant_verdict` |
