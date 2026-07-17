# S15 实验步骤 → 代码接口对照表（自动生成）

> 由 `scripts/release/gen_interface_map.py` 从 `hqsb.release.interface_map` 生成；
> 本文件是派生产物，不手工编辑。

- 实验数：11
- 步骤数：495 / 495
- 接口引用数：796
- 解析状态：全部通过

## 总览

| 实验 | 级别 | 步骤 | 标题 |
|---|---|---|---|
| E15-01 | P0 | 45 | 全局 Claim Ledger、声明扫描与证据门禁 |
| E15-02 | P0 | 45 | Clean CPU Quickstart 与最低门槛复现 |
| E15-03 | P0 | 45 | 目标 GPU/NPU Hero Story 独立重放 |
| E15-04 | P0 | 45 | 可执行文档、双语一致性与能力矩阵对账 |
| E15-05 | P0 | 45 | Tag→Release 制品、Provenance、SBOM 与许可证 |
| E15-06 | P0 | 45 | Dashboard/图表→Raw 数据反向追溯与重生成 |
| E15-07 | P0 | 45 | 3–5 分钟 Demo、故障注入与诚实降级 |
| E15-08 | P0 | 45 | 3/10/30 分钟讲述与对抗技术问答 |
| E15-09 | P0 | 45 | 第三方 Clean-room Reproduction 与修复复测 |
| E15-10 | P0 | 45 | 真实上游 Issue/PR/文档贡献 |
| E15-11 | P1 | 45 | 目标岗位读者 5 分钟首屏理解实验 |

## E15-01 — 全局 Claim Ledger、声明扫描与证据门禁

> claim boundary: E15-01 只证明“当前候选公开陈述有证据且无已知越界”；不证明证据已被独立第三方复现（E15-09）

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 ReleaseCandidateSnapshot | `contracts.ReleaseCandidateSnapshot`, `contracts.new_candidate` |
| 2 | 建立公开渠道全量清单 | `claims.enumerate_surfaces`, `claims.PublicSurface`, `claims.SURFACE_CHANNELS` |
| 3 | 定义 claim 分类法 | `claims.classify_claim`, `records.CLAIM_TYPES` |
| 4 | 冻结证据等级规则 | `identity.required_support`, `identity.EVIDENCE_LEVELS`, `specs.EXPECTED_KINDS` |
| 5 | 定义可检测语言模式 | `claims.DETECTABLE_PATTERNS`, `claims.extract_claim_candidates` |
| 6 | 定义非 claim 排除规则 | `claims.NON_CLAIM_EXCLUSIONS`, `claims.ClaimCandidate` |
| 7 | 全量抽取候选语句 | `claims.extract_claim_candidates`, `claims.ClaimCandidate.as_dict` |
| 8 | 单独抽取全部数字与单位 | `claims.extract_numeric_expressions`, `claims.NUMERIC_CLASSES` |
| 9 | 抽取隐式比较 | `claims.extract_implicit_comparisons` |
| 10 | 抽取绝对能力声明 | `claims.extract_capability_claims` |
| 11 | 规范化语义相同的候选 | `claims.canonicalise_candidates`, `claims.CanonicalClaim` |
| 12 | 建立中英文配对 | `claims.pair_bilingual`, `contracts.check_bilingual_fact_consistency` |
| 13 | 给每条 claim 分配稳定 ID | `claims.assign_claim_id`, `identity.stable_claim_id` |
| 14 | 填写 claim 类型与 estimand | `claims.classify_claim`, `contracts.ClaimRecord.problems` |
| 15 | 绑定 C1 模型身份 | `claims.ClaimBindings`, `claims.bind_fact_fields` |
| 16 | 绑定 C2/C3/C4/C5 范围 | `claims.ClaimBindings`, `claims.bind_fact_fields` |
| 17 | 绑定硬件和环境范围 | `claims.ClaimBindings`, `experiment.environment_fingerprint` |
| 18 | 绑定 baseline 身份 | `claims.ClaimBindings`, `contracts.ClaimRecord.problems` |
| 19 | 绑定正确性与质量资格 | `contracts.check_quality_before_performance`, `claims.ClaimBindings` |
| 20 | 绑定统计对象 | `claims.ClaimBindings`, `contracts.ClaimEffect` |
| 21 | 构造 Claim→Evidence DAG | `claims.EvidenceGraph`, `claims.EvidenceEdge`, `records.EVIDENCE_EDGE_TYPES` |
| 22 | 验证全部 Evidence URI | `claims.verify_uris`, `identity.assert_public_uri` |
| 23 | 校验逐文件 digest 与聚合根 | `claims.verify_digests`, `identity.content_address_aggregate` |
| 24 | 验证 raw 可解析性 | `claims.verify_uris`, `identity.EvidenceRef` |
| 25 | 验证派生可重建性 | `claims.verify_derivable` |
| 26 | 记录贡献与责任归属 | `contracts.ContributionRecord` |
| 27 | 定义时间有效区间 | `identity.FrozenInputs`, `contracts.ClaimRecord.problems` |
| 28 | 构建失效依赖图 | `claims.detect_stale`, `claims.LedgerEntryView.invalidation_keys` |
| 29 | 执行 orphan 检测 | `claims.detect_orphans`, `claims.ORPHAN_KINDS` |
| 30 | 执行 stale 检测 | `claims.detect_stale` |
| 31 | 执行冲突与重复检测 | `claims.detect_conflicts` |
| 32 | 执行单位和数量级检查 | `claims.check_units_and_magnitudes`, `claims.UNIT_FAMILIES` |
| 33 | 执行外推边界检查 | `claims.check_extrapolation_boundaries`, `claims.EXTRAPOLATION_KINDS` |
| 34 | 人工裁决上下文歧义 | `claims.ContextAdjudication` |
| 35 | 注入负对照 Claim | `claims.inject_negative_controls`, `claims.INJECTION_CLASSES` |
| 36 | 计算检测性能 | `claims.DetectorAudit`, `telemetry.DetectorMetrics` |
| 37 | 修订有证据但表述过强的 claim | `claims.revise_claim`, `claims.ClaimRevision` |
| 38 | 撤回无法证明的 claim | `claims.revise_claim`, `contracts.ClaimRecord.transition` |
| 39 | 从 Ledger 生成渠道文案 | `contracts.ClaimRecord.render`, `identity.CHANNEL_MINIMUM_LEVEL` |
| 40 | 建立 CI Claim Gate | `claims.LedgerEntryView`, `claims.detect_stale`, `records.CLAIM_GATES` |
| 41 | 抽取代表 claim 做端到端追溯 | `claims.EvidenceGraph.ancestors`, `telemetry.EvidenceLookupTrial` |
| 42 | 进行独立 claim review | `claims.ReleaseEligibility`, `claims.coverage_and_risk_report` |
| 43 | 生成覆盖与风险报告 | `claims.coverage_and_risk_report` |
| 44 | 逐条裁决发布资格 | `claims.adjudicate_release_eligibility`, `claims.ReleaseEligibility` |
| 45 | 冻结 Final Claim Ledger | `claims.ClaimLedger.freeze`, `claims.ClaimLedger.ledger_digest` |

## E15-02 — Clean CPU Quickstart 与最低门槛复现

> claim boundary: E15-02 只证明最低访问门槛真实、独立且可审计；不证明 CPU 推理性能或任何 kernel 正确性

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 QuickstartContract | `quickstart.QuickstartContract` |
| 2 | 选择支持矩阵 | `quickstart.SupportMatrix`, `quickstart.SupportCell` |
| 3 | 定义独立 session | `quickstart.SessionIdentity` |
| 4 | 冻结网络与缓存场景 | `quickstart.CACHE_STATES`, `quickstart.NETWORK_STATES`, `quickstart.SCENARIO_MATRIX` |
| 5 | 建立初始环境证据 | `quickstart.initial_environment_evidence` |
| 6 | 证明环境中没有加速器依赖 | `quickstart.prove_no_accelerator` |
| 7 | 证明没有 HQSB 残留 | `quickstart.prove_no_hqsb_residue` |
| 8 | 创建动作记录器 | `telemetry.ActionLog`, `telemetry.SegmentTiming` |
| 9 | 开始总计时 | `quickstart.TotalClock.start`, `quickstart.TIMING_STAGES` |
| 10 | 执行文档前置检查 | `quickstart.check_document_prerequisites` |
| 11 | 验证下载地址和 TLS | `quickstart.verify_download_urls` |
| 12 | 从 wheel 安装最小包 | `quickstart.QuickstartContract.install_modes`, `telemetry.ActionLog.append` |
| 13 | 从 sdist 构建安装 | `quickstart.QuickstartContract.install_modes` |
| 14 | 核对 wheel 与 sdist 元数据 | `quickstart.compare_metadata` |
| 15 | 执行 core import 测试 | `quickstart.check_core_import` |
| 16 | 验证 optional dependency 隔离 | `quickstart.check_optional_isolation`, `quickstart.check_core_import` |
| 17 | 检查 CLI 可发现性 | `quickstart.check_core_import`, `telemetry.ActionLog` |
| 18 | 运行环境 capability probe | `quickstart.check_capability_probe` |
| 19 | 下载或读取公开 sample bundle | `quickstart.verify_sample_bundle` |
| 20 | 校验 sample Evidence Manifest | `quickstart.verify_sample_bundle`, `identity.content_address_aggregate` |
| 21 | 运行最小配置解析 | `quickstart.compare_canonical`, `records.RUN_MANIFEST_FIELDS` |
| 22 | 运行 sample reference 路径 | `quickstart.freeze_session`, `records.QuickstartSessionResult` |
| 23 | 验证 C1–C7 Schema | `quickstart.adjudicate_gates`, `quickstart.compare_canonical` |
| 24 | 运行负 Schema 用例 | `quickstart.negative_case_plan` |
| 25 | 从 raw 重建 normalized result | `quickstart.check_byte_stable`, `quickstart.compare_canonical` |
| 26 | 从 normalized 重建报告 | `quickstart.check_byte_stable`, `quickstart.compare_canonical` |
| 27 | 比较 canonical 输出 | `quickstart.compare_canonical`, `quickstart.DEFAULT_IGNORED_FIELDS` |
| 28 | 检查 byte-stable 对象 | `quickstart.check_byte_stable` |
| 29 | 记录资源峰值 | `records.QuickstartSessionResult`, `telemetry.SegmentTiming` |
| 30 | 执行 warm-cache repeat | `quickstart.SessionIdentity`, `quickstart.analyze_critical_path` |
| 31 | 执行 cold-cache repeat | `quickstart.SessionIdentity`, `quickstart.SupportMatrix` |
| 32 | 执行离线行为测试 | `quickstart.negative_case_plan`, `quickstart.SCENARIO_MATRIX` |
| 33 | 注入 cache corruption | `quickstart.negative_case_plan`, `campaign.REQUIRED_ISOLATION` |
| 34 | 注入不可写 cache 目录 | `quickstart.negative_case_plan` |
| 35 | 注入缺失 optional feature | `quickstart.check_optional_isolation`, `quickstart.check_capability_probe` |
| 36 | 注入无效配置 | `quickstart.check_optional_isolation`, `quickstart.negative_case_plan` |
| 37 | 扫描私有状态泄漏 | `quickstart.scan_private_state`, `quickstart.PRIVACY_PATTERNS` |
| 38 | 盘点未文档输入 | `quickstart.InterventionLog`, `quickstart.INTERVENTION_LEVELS` |
| 39 | 分析关键路径时间 | `quickstart.analyze_critical_path` |
| 40 | 判定 30 分钟目标 | `quickstart.judge_time_goal` |
| 41 | 修订文档与工具 | `quickstart.map_gaps_to_owners` |
| 42 | 由新操作者复测 | `quickstart.SessionIdentity` |
| 43 | 比较支持矩阵结果 | `quickstart.SupportMatrix.coverage`, `quickstart.SupportMatrix.unverified` |
| 44 | 逐门裁决 | `quickstart.adjudicate_gates` |
| 45 | 冻结 QuickstartReproductionRecord | `quickstart.freeze_session`, `records.QuickstartSessionResult` |

## E15-03 — 目标 GPU/NPU Hero Story 独立重放

> claim boundary: E15-03 是作者侧/受控新环境的重放，不等于独立第三方复现（E15-09 才验证独立性）

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 选择唯一主 Hero Claim | `hero_replay.HeroStoryContract`, `claims.adjudicate_release_eligibility` |
| 2 | 冻结 HeroStoryContract | `hero_replay.HeroStoryContract`, `hero_replay.NARRATIVE_NODES` |
| 3 | 冻结原始发布证据 | `identity.content_address_aggregate`, `claims.EvidenceGraph` |
| 4 | 选择新目标环境 | `hero_replay.EnvironmentFingerprint` |
| 5 | 定义环境可比性类别 | `hero_replay.classify_comparability`, `hero_replay.COMPARABILITY_CLASSES` |
| 6 | 采集加速器环境指纹 | `hero_replay.EnvironmentFingerprint.as_dict` |
| 7 | 验证设备健康和隔离 | `hero_replay.DeviceHealth` |
| 8 | 验证 profiler 可用性 | `hero_replay.ProfilerAvailability` |
| 9 | 从候选 release 安装 | `records.ReleaseArtifactRecord`, `identity.EvidenceRef` |
| 10 | 验证可选依赖与二进制兼容 | `hero_replay.check_binary_compatibility` |
| 11 | 获取模型和 tokenizer | `hero_replay.ArtifactIdentity` |
| 12 | 验证 workload tokens | `hero_replay.check_workload_tokens` |
| 13 | 验证 OperatorSpec 和 candidate binary | `hero_replay.check_operator_spec`, `hero_replay.ArtifactIdentity` |
| 14 | 执行 capability preflight | `hero_replay.CapabilityPreflight` |
| 15 | 建立 reference 语义基线 | `hero_replay.evaluate_correctness_matrix` |
| 16 | 运行 operator correctness matrix | `hero_replay.evaluate_correctness_matrix` |
| 17 | 运行 block/model differential correctness | `hero_replay.evaluate_correctness_matrix` |
| 18 | 运行服务语义门 | `hero_replay.evaluate_correctness_matrix`, `records.HeroReplayResult.quality_status` |
| 19 | 确认 actual path | `hero_replay.ActualPathEvidence`, `hero_replay.ACTUAL_PATH_PROOFS` |
| 20 | 运行故意 fallback 负对照 | `hero_replay.ActualPathEvidence.problems` |
| 21 | 冻结 confirmatory workload strata | `hero_replay.ConfirmatoryPlan` |
| 22 | 冻结冷/热边界 | `hero_replay.ConfirmatoryPlan` |
| 23 | 冻结运行 block 与交错顺序 | `hero_replay.ConfirmatoryPlan.INTERLEAVE_ORDERS` |
| 24 | 记录运行前稳定性窗口 | `hero_replay.DeviceHealth`, `hero_replay.DeviceHealth.problems` |
| 25 | 采集 operator micro raw | `telemetry.ActionLog`, `records.HeroReplayResult` |
| 26 | 采集 block/model raw | `records.HeroReplayResult`, `telemetry.SegmentTiming` |
| 27 | 采集 service raw | `records.HeroReplayResult.primary_effect` |
| 28 | 采集资源与能效 | `telemetry.SegmentTiming`, `records.HeroReplayResult` |
| 29 | 采集系统时间线 Profile | `hero_replay.profile_reconciliation_rows`, `figures.LineageLayer` |
| 30 | 采集 kernel counter Profile | `hero_replay.profile_reconciliation_rows`, `hero_replay.PROFILE_LAYERS` |
| 31 | 重算原始热点占比 | `hero_replay.AmdahlModel` |
| 32 | 计算预注册 Amdahl 预测 | `hero_replay.AmdahlModel.predicted`, `hero_replay.AmdahlModel.prediction_interval` |
| 33 | 估计 primary effect | `hero_replay.BlockedEstimate`, `hero_replay.BlockedEstimate.effect` |
| 34 | 比较原报告与重放结果 | `hero_replay.compare_with_release` |
| 35 | 分析 Amdahl 预测误差 | `hero_replay.AmdahlModel.explain_residual`, `hero_replay.AMDAHL_ERROR_COMPONENTS` |
| 36 | 分析环境差异 | `hero_replay.environment_diff` |
| 37 | 注入 artifact mismatch | `hero_replay.ArtifactIdentity.problems`, `identity.EvidenceRef` |
| 38 | 注入错误后端或 silent fallback | `hero_replay.ActualPathEvidence`, `contracts.check_no_silent_degradation` |
| 39 | 注入 correctness failure | `hero_replay.evaluate_correctness_matrix`, `contracts.check_quality_before_performance` |
| 40 | 执行最小跨版本敏感性 | `hero_replay.environment_diff`, `campaign.REQUIRED_ISOLATION` |
| 41 | 裁决结论一致性 | `hero_replay.replay_verdict`, `hero_replay.REPLAY_VERDICTS` |
| 42 | 更新 Claim Ledger | `hero_replay.claim_actions_for`, `claims.revise_claim` |
| 43 | 生成一键 Hero Replay | `hero_replay.one_click_replay_command`, `experiment.RunDirectory.write_verdict` |
| 44 | 独立复核 raw→report | `figures.independent_review_task`, `figures.rebuild_chain` |
| 45 | 冻结 HeroReplayRecord | `hero_replay.freeze_replay`, `records.HeroReplayResult` |

## E15-04 — 可执行文档、双语一致性与能力矩阵对账

> claim boundary: E15-04 通过说明文档与当前候选一致且可执行，不说明陌生读者一定能快速理解（E15-11）

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 DocumentationContract | `docs_gate.DocumentationContract` |
| 2 | 盘点全部公开入口 | `docs_gate.PUBLIC_ENTRY_KINDS`, `docs_gate.PageInventory` |
| 3 | 定义读者与信息任务 | `docs_gate.READER_TASKS` |
| 4 | 冻结文档信息架构 | `docs_gate.DOC_SECTIONS` |
| 5 | 建立页面 inventory | `docs_gate.PageEntry`, `docs_gate.PageInventory.bilingual_pairs` |
| 6 | 建立链接 inventory | `docs_gate.LinkEntry`, `docs_gate.LINK_TYPES` |
| 7 | 建立 code-block inventory | `docs_gate.CodeBlock`, `docs_gate.CodeBlockInventory` |
| 8 | 给 code block 分类 | `docs_gate.CODE_BLOCK_CLASSES`, `docs_gate.CodeBlock.problems` |
| 9 | 建立机器可读示例 inventory | `docs_gate.SchemaExample`, `docs_gate.EXAMPLE_FORMATS` |
| 10 | 建立事实与 Claim ID inventory | `docs_gate.FactEntry` |
| 11 | 从 clean checkout 构建文档 | `docs_gate.DocumentationContract`, `telemetry.ActionLog` |
| 12 | 把 warning 升级为受控结果 | `docs_gate.classify_warnings`, `docs_gate.WARNING_LEVELS` |
| 13 | 验证导航与目录完整性 | `docs_gate.PageInventory.problems`, `docs_gate.READER_TASKS` |
| 14 | 验证内部相对链接 | `docs_gate.check_internal_links` |
| 15 | 验证 heading anchor | `docs_gate.check_internal_links` |
| 16 | 验证 release/evidence 链接 | `docs_gate.check_release_links` |
| 17 | 验证外部引用 | `docs_gate.check_external_references` |
| 18 | 验证下载内容类型与 hash | `docs_gate.check_download_content` |
| 19 | 运行 CPU-safe 命令 | `docs_gate.run_code_blocks` |
| 20 | 运行 accelerator 命令 | `docs_gate.run_code_blocks`, `records.COMMAND_VERDICTS` |
| 21 | 验证 dry-run 命令 | `docs_gate.run_code_blocks` |
| 22 | 验证 manual-external 步骤 | `docs_gate.run_code_blocks`, `docs_gate.CODE_BLOCK_CLASSES` |
| 23 | 验证 DISPLAY_ONLY 标识 | `docs_gate.CodeBlock.problems` |
| 24 | 对账 CLI help | `docs_gate.diff_cli_help` |
| 25 | 对账配置优先级 | `docs_gate.check_config_precedence` |
| 26 | 校验所有完整 Schema 示例 | `docs_gate.validate_schema_examples` |
| 27 | 验证片段可组装性 | `docs_gate.validate_schema_examples`, `docs_gate.SchemaExample` |
| 28 | 验证示例预期输出 | `docs_gate.check_expected_output` |
| 29 | 生成实际 Capability Matrix | `docs_gate.capability_matrix_from_evidence`, `docs_gate.SupportCell` |
| 30 | 对账文档 Support Matrix | `docs_gate.reconcile_support_matrix`, `docs_gate.MATRIX_CONDITIONS` |
| 31 | 验证 limitations 反向链接 | `docs_gate.check_limitation_backlinks` |
| 32 | 提取双语结构化事实 | `docs_gate.extract_language_facts`, `docs_gate.BILINGUAL_FACT_FIELDS` |
| 33 | 执行双语事实 diff | `docs_gate.check_bilingual_diff` |
| 34 | 执行版本一致性检查 | `docs_gate.check_version_consistency` |
| 35 | 执行路径和隐私扫描 | `docs_gate.scan_privacy_paths` |
| 36 | 执行可访问性与小屏检查 | `docs_gate.check_accessibility`, `docs_gate.MIN_FIGURE_CONTEXT` |
| 37 | 执行断网静态站检查 | `docs_gate.check_offline_static` |
| 38 | 执行搜索与证据定位任务 | `docs_gate.record_navigation_tasks`, `telemetry.EvidenceLookupTrial` |
| 39 | 注入文档负对照 | `docs_gate.inject_doc_negative_controls` |
| 40 | 计算覆盖与检测指标 | `docs_gate.compute_coverage_metrics`, `telemetry.DetectorMetrics` |
| 41 | 归因并修复文档缺陷 | `docs_gate.classify_defect`, `docs_gate.DEFECT_OWNERSHIP` |
| 42 | 重建新候选文档 | `contracts.new_candidate`, `docs_gate.freeze_verification` |
| 43 | 执行 clean consumer smoke | `quickstart.SessionIdentity`, `docs_gate.run_code_blocks` |
| 44 | 逐门裁决 | `docs_gate.adjudicate_doc_gates` |
| 45 | 冻结 DocumentationVerificationRecord | `docs_gate.freeze_verification`, `records.DocumentationVerificationRecord` |

## E15-05 — Tag→Release 制品、Provenance、SBOM 与许可证

> claim boundary: E15-05 证明发布资产可绑定、可验证、可合法分发；不证明源码无恶意或无漏洞

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 ReleasePolicy | `supply_chain.ReleasePolicy` |
| 2 | 冻结 ReleaseCandidateSnapshot | `contracts.ReleaseCandidateSnapshot`, `contracts.new_candidate` |
| 3 | 执行 release GO/NO-GO 前置检查 | `supply_chain.supply_chain_go_no_go`, `experiment.check_prerequisites` |
| 4 | 选择版本号并验证语义 | `supply_chain.validate_version` |
| 5 | 生成并审查 changelog | `supply_chain.ReleasePolicy`, `telemetry.ActionLog` |
| 6 | 验证 tag 目标和保护策略 | `supply_chain.validate_version`, `campaign.PROHIBITED_ACTIONS` |
| 7 | 冻结 build definition | `supply_chain.BuildDefinition` |
| 8 | 最小化 builder 权限 | `supply_chain.BuildDefinition.problems` |
| 9 | 验证构建隔离 | `supply_chain.BuildDefinition`, `campaign.REQUIRED_ISOLATION` |
| 10 | 建立 release asset inventory | `supply_chain.AssetRecord`, `supply_chain.EXTRA_ASSETS` |
| 11 | 构建 sdist | `supply_chain.AssetRecord`, `telemetry.ActionLog` |
| 12 | 从 sdist 构建 wheel | `supply_chain.AssetRecord`, `supply_chain.consumer_verify` |
| 13 | 构建 container image | `supply_chain.AssetRecord`, `supply_chain.consumer_verify` |
| 14 | 构建架构/后端变体 | `records.ReleaseArtifactRecord.ARTIFACT_TYPES`, `supply_chain.AssetRecord` |
| 15 | 构建 sample/evidence bundle | `contracts.PublicEvidenceBundle`, `supply_chain.ArtifactManifest` |
| 16 | 生成 canonical artifact manifest | `supply_chain.ArtifactManifest`, `supply_chain.verify_artifact_manifest` |
| 17 | 生成构建 provenance | `supply_chain.Provenance` |
| 18 | 生成 artifact attestation | `supply_chain.Provenance.problems`, `supply_chain.verify_provenance` |
| 19 | 生成软件包 SBOM | `supply_chain.SbomDocument`, `supply_chain.SbomComponent` |
| 20 | 生成 container SBOM | `supply_chain.SbomDocument`, `supply_chain.SbomDocument.COVERAGE_KINDS` |
| 21 | 审计 accelerator binary 依赖 | `supply_chain.SbomComponent`, `hero_replay.check_binary_compatibility` |
| 22 | 验证 SBOM 语法与覆盖 | `supply_chain.verify_sbom_coverage` |
| 23 | 运行漏洞扫描 | `supply_chain.VulnerabilityReport` |
| 24 | 执行漏洞适用性裁决 | `supply_chain.dispose_vulnerabilities`, `records.FINDING_DISPOSITIONS` |
| 25 | 审计源码许可证 | `supply_chain.LicenseInventory` |
| 26 | 执行逐文件许可覆盖 | `supply_chain.LicenseInventory.problems` |
| 27 | 审计第三方源码与 notices | `supply_chain.LicenseInventory`, `supply_chain.AssetRecord` |
| 28 | 审计模型与 tokenizer 权利 | `supply_chain.RightsDecision`, `supply_chain.audit_model_data_rights` |
| 29 | 审计数据与 prompt 权利 | `supply_chain.RightsDecision`, `supply_chain.RIGHTS_KINDS` |
| 30 | 审计图片、字体和录屏素材 | `supply_chain.RightsDecision.KINDS`, `demo.DemoRecordingManifest` |
| 31 | 运行 secret 扫描 | `supply_chain.verify_secret_scan`, `supply_chain.SECRET_SCAN_SCOPES` |
| 32 | 运行 PII/路径/内网扫描 | `supply_chain.verify_pii_scan`, `docs_gate.scan_privacy_paths` |
| 33 | 检查 archive 内容边界 | `supply_chain.check_archive_boundary` |
| 34 | 执行独立重复构建 | `supply_chain.compare_rebuilds` |
| 35 | 归因非确定性 | `supply_chain.attribute_nondeterminism`, `supply_chain.NONDETERMINISM_SOURCES` |
| 36 | 执行 wheel/sdist 消费验证 | `supply_chain.consumer_verify`, `quickstart.check_core_import` |
| 37 | 执行 container 消费验证 | `supply_chain.consumer_verify` |
| 38 | 执行 provenance 消费端验证 | `supply_chain.verify_provenance`, `supply_chain.PROVENANCE_CONSUMER_CHECKS` |
| 39 | 执行 SBOM/notice 消费验证 | `supply_chain.verify_sbom_coverage`, `supply_chain.consumer_verify` |
| 40 | 验证 release notes 与实际资产 | `supply_chain.consumer_verify`, `docs_gate.check_version_consistency` |
| 41 | 验证不可变发布流程 | `supply_chain.check_immutable_publication` |
| 42 | 验证撤回和安全修复路径 | `supply_chain.retraction_plan`, `records.FINDING_DISPOSITIONS` |
| 43 | 归档可引用版本 | `supply_chain.ArchiveCitation` |
| 44 | 执行最终独立供应链审查 | `supply_chain.supply_chain_go_no_go`, `claims.ContextAdjudication` |
| 45 | 签发 ReleaseSupplyChainRecord | `supply_chain.freeze_release`, `records.ReleaseArtifactRecord` |

## E15-06 — Dashboard/图表→Raw 数据反向追溯与重生成

> claim boundary: E15-06 证明抽查与结构门下图表可追溯、可重建；不证明未抽数据的科学结论自动正确

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 FigureAuditContract | `figures.FigureAuditContract` |
| 2 | 盘点公开视觉对象 | `figures.FigureSpec`, `docs_gate.PageInventory` |
| 3 | 给每个对象分配稳定 Figure ID | `figures.FigureSpec.figure_id` |
| 4 | 建立 FigureSpec | `figures.FigureSpec` |
| 5 | 建立 point-level ID 规则 | `figures.point_id` |
| 6 | 定义抽样 strata | `figures.SAMPLING_STRATA`, `figures.SamplingFrameRow` |
| 7 | 冻结强制样本 | `figures.MANDATORY_SAMPLE_KINDS`, `figures.missing_mandatory_kinds` |
| 8 | 执行随机抽样 | `figures.draw_sample` |
| 9 | 建立 clean analysis 环境 | `campaign.REQUIRED_ISOLATION`, `quickstart.SessionIdentity` |
| 10 | 验证 figure artifact identity | `figures.FigureSpec.problems`, `figures.LineageLayer` |
| 11 | 从图点定位 point record | `figures.point_id`, `telemetry.EvidenceLookupTrial` |
| 12 | 定位冻结 query | `figures.FigureSpec.query_id` |
| 13 | 检查过滤合法性 | `figures.check_filter_legality` |
| 14 | 定位 aggregate entity | `figures.LineageChain.add_layer` |
| 15 | 重算聚合值 | `figures.recompute_aggregate` |
| 16 | 验证实验单位 | `figures.validate_experiment_unit` |
| 17 | 验证误差线语义 | `figures.validate_interval_semantics` |
| 18 | 定位 normalized rows | `figures.LineageChain.add_layer` |
| 19 | 验证 normalized Schema | `figures.LineageLayer.problems` |
| 20 | 定位 normalization activity | `figures.LineageChain.add_layer` |
| 21 | 重建 normalized rows | `figures.rebuild_chain` |
| 22 | 定位 raw sample entities | `figures.LineageLayer`, `identity.EvidenceRef` |
| 23 | 校验 raw digest 和大小 | `claims.verify_digests`, `identity.content_address_aggregate` |
| 24 | 验证 raw Schema 与完整性 | `figures.LineageLayer.problems` |
| 25 | 绑定 protocol/config | `figures.FigureSpec`, `claims.ClaimBindings` |
| 26 | 绑定 model/hardware/environment | `claims.ClaimBindings`, `experiment.environment_fingerprint` |
| 27 | 绑定 source/binary/actual path | `hero_replay.ActualPathEvidence`, `figures.LineageLayer` |
| 28 | 绑定 correctness/quality gate | `contracts.check_quality_before_performance` |
| 29 | 绑定 ClaimRecord | `contracts.ClaimRecord`, `figures.FigureSpec.claim_ids` |
| 30 | 从 raw 全链重建抽中点 | `figures.rebuild_chain`, `figures.LineageChain.digest` |
| 31 | 重建完整抽中 figure | `figures.FigureSpec.as_dict`, `figures.classify_visual_diff` |
| 32 | 比较数值表 | `figures.compare_numeric_table` |
| 33 | 执行 visual diff | `figures.classify_visual_diff`, `figures.REBUILD_CLASSES` |
| 34 | 验证图中最小上下文 | `figures.check_min_context`, `figures.FIGURE_CONTEXT_FIELDS` |
| 35 | 验证 missing/unsupported 表达 | `figures.check_missing_expression`, `figures.MISSING_MARKERS` |
| 36 | 验证对数轴/截断轴/归一化 | `figures.check_axis_transform`, `figures.AXIS_FEATURES` |
| 37 | 验证 Pareto/frontier 计算 | `figures.recompute_pareto` |
| 38 | 注入 raw 缺失/篡改 | `figures.inject_lineage_negative_controls` |
| 39 | 注入 query/filter 篡改 | `figures.check_filter_legality`, `figures.inject_lineage_negative_controls` |
| 40 | 注入单位和 series 错位 | `figures.compare_numeric_table`, `figures.inject_lineage_negative_controls` |
| 41 | 注入 fallback 混入 | `figures.inject_lineage_negative_controls`, `hero_replay.ActualPathEvidence` |
| 42 | 计算审计指标 | `figures.compute_audit_metrics`, `telemetry.DetectorMetrics` |
| 43 | 修复并全量影响分析 | `figures.impact_analysis`, `contracts.new_candidate` |
| 44 | 独立复核抽样与重建 | `figures.independent_review_task`, `figures.check_access_friction` |
| 45 | 冻结 FigureLineageAuditRecord | `figures.freeze_audit`, `records.PointLineageRecord` |

## E15-07 — 3–5 分钟 Demo、故障注入与诚实降级

> claim boundary: E15-07 证明演示在有限时间和故障下可靠、诚实；不证明观众已经理解技术（E15-08/E15-11）

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 DemoObjective | `demo.DemoObjective` |
| 2 | 选择单一 Hero Claim | `demo.DemoObjective`, `claims.adjudicate_release_eligibility` |
| 3 | 定义目标观众 | `demo.DemoObjective`, `narrative.role_matrix` |
| 4 | 冻结 3–5 分钟时间预算 | `demo.SegmentBudget`, `demo.NORMAL_PATH_SEGMENTS` |
| 5 | 建立 Demo 状态标签 | `demo.MEASUREMENT_STATES` |
| 6 | 建立场景状态机 | `demo.StateMachine`, `demo.STATES` |
| 7 | 选择正常路径 | `demo.NORMAL_PATH_SEGMENTS`, `demo.DemoScript` |
| 8 | 选择最小 live 工作量 | `demo.SegmentBudget`, `hero_replay.ConfirmatoryPlan` |
| 9 | 准备离线 evidence 路径 | `demo.artifact_integrity_check`, `contracts.PublicEvidenceBundle` |
| 10 | 准备预录路径 | `demo.DemoRecordingManifest` |
| 11 | 准备 CPU fallback | `quickstart.SCENARIO_MATRIX`, `demo.fault_matrix_for` |
| 12 | 冻结演示环境 | `demo.DemoEnvironment` |
| 13 | 执行干净启动检查 | `demo.clean_start_check` |
| 14 | 执行 artifact 完整性检查 | `demo.artifact_integrity_check` |
| 15 | 执行隐私和通知检查 | `demo.privacy_check`, `demo.DemoEnvironment` |
| 16 | 执行设备和网络 preflight | `demo.preflight_report` |
| 17 | 冻结 fallback 决策阈值 | `demo.FallbackThresholds`, `demo.SLO_FIELDS` |
| 18 | 编写逐句 Demo 脚本 | `demo.DemoScript`, `demo.ScriptAction` |
| 19 | 验证所有命令来自 E15-04 | `demo.check_script_provenance`, `docs_gate.run_code_blocks` |
| 20 | 验证所有数字来自 E15-01 | `demo.check_script_provenance`, `contracts.ClaimRecord` |
| 21 | 验证所有图来自 E15-06 | `demo.check_script_provenance`, `figures.FigureSpec` |
| 22 | 执行正常路径冷启动演练 | `demo.RehearsalSession`, `demo.SegmentBudget` |
| 23 | 执行正常路径热启动演练 | `demo.RehearsalSession`, `demo.RehearsalSession.passed` |
| 24 | 重复正常演练 | `demo.analyse_reliability` |
| 25 | 注入无 GPU/NPU | `demo.fault_matrix_for`, `demo.FAULT_ACTION_MATRIX` |
| 26 | 注入设备忙或显存不足 | `demo.fault_matrix_for`, `demo.check_no_external_writes` |
| 27 | 注入 cache miss | `demo.fault_matrix_for` |
| 28 | 注入网络不可用 | `demo.fault_matrix_for` |
| 29 | 注入模型缺失或 hash 错 | `demo.fault_matrix_for`, `hero_replay.ArtifactIdentity` |
| 30 | 注入 live command 超时 | `demo.FallbackThresholds.live_command_exceeded` |
| 31 | 注入坏 release/evidence asset | `demo.artifact_integrity_check` |
| 32 | 注入网页/投屏不可用 | `demo.fault_matrix_for` |
| 33 | 验证 fallback 口头诚实性 | `demo.FAULT_ACTION_MATRIX`, `demo.check_observer_state_accuracy` |
| 34 | 验证状态识别 | `demo.check_observer_state_accuracy` |
| 35 | 执行证据 drill-down | `telemetry.EvidenceLookupTrial`, `figures.check_access_friction` |
| 36 | 展示一个失败或无收益案例 | `demo.DemoObjective.negative_story_claim_id` |
| 37 | 验证现场输出可读性 | `docs_gate.check_accessibility` |
| 38 | 验证演示不改变外部状态 | `demo.check_no_external_writes` |
| 39 | 记录每次动作与偏差 | `telemetry.ActionLog`, `demo.RehearsalSession` |
| 40 | 分析时长和可靠性 | `demo.analyse_reliability`, `telemetry.TimeToEventSummary` |
| 41 | 修订脚本和故障树 | `demo.fault_matrix_for`, `demo.DemoScript.problems` |
| 42 | 录制候选正式视频 | `demo.DemoRecordingManifest` |
| 43 | 执行录屏供应链检查 | `demo.check_recording_supply_chain`, `supply_chain.RightsDecision` |
| 44 | 让非作者按脚本演示 | `demo.RehearsalSession`, `quickstart.InterventionLog` |
| 45 | 冻结 DemoRehearsalRecord | `demo.freeze_rehearsal`, `records.DemoSessionResult` |

## E15-08 — 3/10/30 分钟讲述与对抗技术问答

> claim boundary: E15-08 证明表达可验证且技术上可防守；不代表真实招聘结果，也不替代 E15-09 的独立复现

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 NarrativeContract | `narrative.NarrativeContract` |
| 2 | 建立岗位能力矩阵 | `narrative.role_matrix`, `narrative.ROLE_MATRIX` |
| 3 | 选择唯一 Hero Story | `hero_replay.HeroStoryContract`, `narrative.FactCard` |
| 4 | 选择一个失败故事 | `narrative.FactCard`, `claims.adjudicate_release_eligibility` |
| 5 | 冻结事实卡片 | `narrative.FactCard` |
| 6 | 冻结 ContributionRecord | `contracts.ContributionRecord` |
| 7 | 写一句项目定义 | `narrative.NarrativeContract.allowed_claims` |
| 8 | 写 30 秒价值段 | `narrative.FactCard`, `demo.DemoObjective` |
| 9 | 设计 3 分钟骨架 | `narrative.NarrativeOutline`, `narrative.CONTENT_GATES` |
| 10 | 设计 10 分钟扩展 | `narrative.NarrativeOutline` |
| 11 | 设计 30 分钟深挖 | `narrative.NarrativeOutline` |
| 12 | 设计运营商/电网强调层 | `narrative.ROLE_MATRIX`, `narrative.role_matrix` |
| 13 | 设计 AI Infra/算子强调层 | `narrative.ROLE_MATRIX` |
| 14 | 执行两版本事实 diff | `narrative.check_two_role_fact_diff`, `contracts.check_bilingual_fact_consistency` |
| 15 | 准备最小架构图 | `figures.check_min_context`, `figures.FigureSpec` |
| 16 | 准备 hero 图和反例图 | `figures.FigureSpec`, `figures.check_min_context` |
| 17 | 准备 evidence drill-down 路径 | `telemetry.EvidenceLookupTrial`, `figures.check_access_friction` |
| 18 | 建立问题分类法 | `narrative.QUESTION_CATEGORIES` |
| 19 | 编写 correctness 对抗题 | `narrative.question_bank` |
| 20 | 编写 benchmark 对抗题 | `narrative.question_bank` |
| 21 | 编写 CUDA/算子对抗题 | `narrative.question_bank` |
| 22 | 编写 Amdahl 对抗题 | `narrative.question_bank`, `hero_replay.AmdahlModel` |
| 23 | 编写 Runtime/Serving 对抗题 | `narrative.question_bank` |
| 24 | 编写异构/分布式对抗题 | `narrative.question_bank` |
| 25 | 编写失败/反事实问题 | `narrative.question_bank` |
| 26 | 编写个人贡献追问 | `narrative.question_bank`, `contracts.ContributionRecord` |
| 27 | 建立回答 rubric | `narrative.AnswerRubric`, `narrative.SCORING_DIMENSIONS` |
| 28 | 冻结问题抽样规则 | `narrative.sample_questions`, `narrative.MANDATORY_QUESTION_CATEGORIES` |
| 29 | 录制 3 分钟无打断讲述 | `narrative.InterviewSession`, `records.InterviewSessionResult` |
| 30 | 录制 10 分钟讲述 | `narrative.InterviewSession` |
| 31 | 录制 30 分钟讲述 | `narrative.InterviewSession` |
| 32 | 执行运营商/电网模拟面试 | `narrative.InterviewSession`, `narrative.ROLE_VARIANTS` |
| 33 | 执行 AI Infra/算子模拟面试 | `narrative.InterviewSession` |
| 34 | 执行 correctness 追问链 | `narrative.AnswerRubric` |
| 35 | 执行 benchmark 追问链 | `narrative.AnswerRubric` |
| 36 | 执行 Amdahl 现场题 | `hero_replay.AmdahlModel.ideal_upper_bound`, `narrative.AnswerRubric` |
| 37 | 执行 evidence 随机定位 | `telemetry.EvidenceLookupTrial`, `narrative.NarrativeContract.evidence_lookup_limit_s` |
| 38 | 执行未知问题处理测试 | `narrative.AnswerRubric`, `narrative.HARD_FAILURES` |
| 39 | 执行矛盾证据测试 | `hero_replay.replay_verdict`, `narrative.AnswerRubric` |
| 40 | 执行 attribution 审计 | `contracts.ContributionRecord.problems`, `narrative.InterviewSession` |
| 41 | 汇总盲评和分歧 | `narrative.InterviewSession.reviewer_ids`, `telemetry.TimeToEventSummary` |
| 42 | 修订表达而非事实 | `narrative.check_two_role_fact_diff`, `claims.revise_claim` |
| 43 | 生成 FAQ 和证据索引 | `narrative.generate_faq` |
| 44 | 生成简历 bullet 候选 | `narrative.generate_resume_bullets` |
| 45 | 冻结 InterviewNarrativeRecord | `narrative.freeze_narrative`, `records.InterviewSessionResult` |

## E15-09 — 第三方 Clean-room Reproduction 与修复复测

> claim boundary: E15-09 代表至少一次受控、独立的主要结果复现；不代表所有用户、所有设备或未来版本都能自动复现

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 ReproductionContract | `clean_room.ReproductionContract` |
| 2 | 定义 reviewer 独立性标准 | `clean_room.ReviewerIndependenceStatement` |
| 3 | 选择 reviewer | `clean_room.ReviewerIndependenceStatement`, `clean_room.reviewer_variation` |
| 4 | 签署范围和隐私约定 | `clean_room.ReviewerIndependenceStatement`, `records.CleanRoomReproductionRecord` |
| 5 | 冻结公开材料集合 | `clean_room.ReceivedMaterials`, `clean_room.RECEIVED_MATERIAL_KINDS` |
| 6 | 建立信息隔离墙 | `clean_room.ReceivedMaterials.problems`, `clean_room.FORBIDDEN_MATERIALS` |
| 7 | 选择 clean 环境 | `quickstart.SessionIdentity`, `quickstart.prove_no_hqsb_residue` |
| 8 | 采集 reviewer 环境指纹 | `hero_replay.EnvironmentFingerprint`, `experiment.environment_fingerprint` |
| 9 | 设置独立记录机制 | `telemetry.ActionLog`, `clean_room.HelpLog` |
| 10 | 开始盲时钟 | `quickstart.TotalClock`, `telemetry.TimeToEventSummary` |
| 11 | 让 reviewer 复述项目和 claim | `clean_room.ReceivedMaterials`, `records.CleanRoomReproductionRecord` |
| 12 | 获取并验证 release | `supply_chain.verify_provenance`, `supply_chain.ArtifactManifest` |
| 13 | 执行 CPU quickstart | `quickstart.QuickstartContract`, `quickstart.freeze_session` |
| 14 | 执行 evidence bundle inventory | `contracts.PublicEvidenceBundle`, `identity.content_address_aggregate` |
| 15 | 执行抽定图表重建 | `figures.draw_sample`, `figures.rebuild_chain` |
| 16 | 获取模型和运行依赖 | `hero_replay.ArtifactIdentity`, `supply_chain.RightsDecision` |
| 17 | 执行 accelerator preflight | `hero_replay.CapabilityPreflight`, `hero_replay.DeviceHealth` |
| 18 | 验证 ModelArtifact/WorkloadSpec | `hero_replay.ArtifactIdentity`, `hero_replay.check_workload_tokens` |
| 19 | 运行 reference correctness | `hero_replay.evaluate_correctness_matrix` |
| 20 | 运行 candidate correctness | `hero_replay.evaluate_correctness_matrix` |
| 21 | 验证 actual path | `hero_replay.ActualPathEvidence` |
| 22 | 执行冻结性能 protocol | `hero_replay.ConfirmatoryPlan`, `hero_replay.BlockedEstimate` |
| 23 | 采集同轮 profile | `hero_replay.profile_reconciliation_rows`, `figures.LineageLayer` |
| 24 | 计算独立统计结果 | `hero_replay.BlockedEstimate.effect`, `hero_replay.AmdahlModel` |
| 25 | 比较 reproduction guard band | `hero_replay.compare_with_release`, `hero_replay.replay_verdict` |
| 26 | 审计 Claim Ledger 样本 | `claims.ClaimLedger`, `claims.detect_orphans` |
| 27 | 记录所有帮助请求 | `clean_room.HelpLog`, `clean_room.HELP_LEVELS` |
| 28 | 要求先提交结构化 issue | `clean_room.Finding`, `clean_room.classify_finding` |
| 29 | 作者按公开信息响应 | `clean_room.HelpLog`, `clean_room.HELP_PASS_EFFECT` |
| 30 | 给 finding 分类 | `clean_room.classify_finding`, `clean_room.FINDING_CATEGORIES` |
| 31 | 给 finding 定严重度 | `clean_room.severity_of`, `clean_room.FINDING_SEVERITIES` |
| 32 | 保护原失败证据 | `clean_room.protect_original_failure` |
| 33 | 作者实施最小修复 | `clean_room.minimal_fix_check`, `docs_gate.classify_defect` |
| 34 | 生成新 Release Candidate | `clean_room.new_candidate_required`, `contracts.new_candidate` |
| 35 | 由原 reviewer 复测原问题 | `clean_room.retest_scope` |
| 36 | 验证无回归 | `clean_room.regression_scope` |
| 37 | 重复完整 hero confirmation | `clean_room.regression_scope`, `hero_replay.freeze_replay` |
| 38 | 进行 reviewer 独立结论撰写 | `clean_room.independence_audit`, `records.CleanRoomReproductionRecord.report_uri` |
| 39 | 作者事实核对但不改判断 | `clean_room.author_fact_response` |
| 40 | 计算复现层级和指标 | `clean_room.reproduction_metrics`, `clean_room.LEVEL_CLAIM_MATRIX` |
| 41 | 评估 reviewer 间变异 | `clean_room.reviewer_variation` |
| 42 | 更新文档和 Claim Ledger | `clean_room.LEVEL_CLAIM_MATRIX`, `claims.revise_claim` |
| 43 | 发布复现报告和 issue 链 | `clean_room.Finding`, `docs_gate.check_release_links` |
| 44 | 执行独立审计复核 | `clean_room.independence_audit` |
| 45 | 冻结 CleanRoomReproductionRecord | `clean_room.freeze_reproduction`, `records.CleanRoomReproductionRecord` |

## E15-10 — 真实上游 Issue/PR/文档贡献

> claim boundary: E15-10 证明真实、可核验的上游协作；merged 不是唯一通过标准

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 ContributionStudyContract | `upstream.ContributionStudyContract` |
| 2 | 建立真实问题候选池 | `upstream.CandidateFinding` |
| 3 | 为候选保存原始证据 | `upstream.CandidateFinding.problems`, `identity.EvidenceRef` |
| 4 | 先排查 HQSB 自身错误 | `upstream.boundary_triage`, `upstream.BOUNDARY_TRIAGE` |
| 5 | 复现于上游最小环境 | `upstream.MinimalReproducer` |
| 6 | 选择候选目标上游 | `upstream.ContributionStudyContract.target_upstreams` |
| 7 | 读取当前贡献指南 | `upstream.ContributionStudyContract`, `telemetry.ActionLog` |
| 8 | 确认安全披露边界 | `upstream.boundary_triage`, `supply_chain.retraction_plan` |
| 9 | 搜索重复 issue/PR | `upstream.ContributionStudyContract`, `telemetry.ActionLog` |
| 10 | 检查当前主干和最新 release | `upstream.MinimalReproducer`, `docs_gate.check_version_consistency` |
| 11 | 执行版本二分或边界定位 | `upstream.MinimalReproducer.negative_controls` |
| 12 | 最小化输入 | `upstream.MinimalReproducer` |
| 13 | 最小化依赖 | `upstream.MinimalReproducer.dependencies` |
| 14 | 最小化执行步骤 | `upstream.MinimalReproducer.command_count` |
| 15 | 验证 reproducer 稳定性 | `upstream.MinimalReproducer.failure_rate`, `upstream.MinimalReproducer.problems` |
| 16 | 加入负对照 | `upstream.MinimalReproducer.negative_controls` |
| 17 | 采集必要诊断 | `upstream.IssueDraft.evidence`, `telemetry.ActionLog` |
| 18 | 判断贡献类型 | `upstream.CONTRIBUTION_TYPES`, `records.UpstreamContributionRecord.CONTRIBUTION_TYPES` |
| 19 | 写问题陈述 | `upstream.IssueDraft` |
| 20 | 限制性能措辞 | `upstream.check_performance_wording` |
| 21 | 执行公开前隐私扫描 | `upstream.IssueDraft`, `supply_chain.verify_pii_scan` |
| 22 | 执行许可和可提交性检查 | `supply_chain.LicenseInventory`, `upstream.IssueDraft` |
| 23 | 记录 Agent 使用边界 | `upstream.attribution_audit`, `contracts.ContributionRecord` |
| 24 | 让独立技术 reviewer 预审 | `upstream.quality_score`, `claims.ContextAdjudication` |
| 25 | 提交 issue/RFC | `upstream.IssueDraft`, `upstream.disposition_claim` |
| 26 | 若适合，设计最小 patch | `upstream.PatchDesign` |
| 27 | 补 regression test | `upstream.check_patch` |
| 28 | 运行上游规定测试 | `upstream.check_patch`, `telemetry.ActionLog` |
| 29 | 运行 correctness 扩展矩阵 | `upstream.check_patch`, `hero_replay.evaluate_correctness_matrix` |
| 30 | 运行性能 benchmark | `upstream.check_performance_wording`, `hero_replay.BlockedEstimate` |
| 31 | 审查 API/ABI 和 backward compatibility | `upstream.PatchDesign.compatibility` |
| 32 | 更新文档和 release-note fragment | `upstream.downstream_actions` |
| 33 | 提交 PR 并关联 issue | `upstream.ReviewRound`, `upstream.disposition_claim` |
| 34 | 响应自动 CI 和 reviewer | `upstream.ReviewRound` |
| 35 | 记录每轮设计变化 | `upstream.ReviewRound` |
| 36 | 处理替代方案 | `upstream.PatchDesign.alternatives`, `upstream.ReviewRound` |
| 37 | 处理 duplicate/won't-fix | `upstream.disposition_claim`, `upstream.downstream_actions` |
| 38 | 处理 stalled/no-response | `upstream.disposition_claim`, `upstream.UPSTREAM_DISPOSITIONS` |
| 39 | 验证最终上游状态 | `upstream.disposition_claim`, `records.UpstreamContributionRecord` |
| 40 | 将上游结果回流 HQSB | `upstream.downstream_actions` |
| 41 | 在 HQSB 重放原问题 | `hero_replay.freeze_replay`, `upstream.downstream_actions` |
| 42 | 执行个人贡献审计 | `upstream.attribution_audit`, `contracts.ContributionRecord` |
| 43 | 让第三方评审贡献质量 | `upstream.quality_score`, `upstream.QUALITY_DIMENSIONS` |
| 44 | 生成可面试贡献故事 | `upstream.ReviewRound`, `narrative.generate_faq` |
| 45 | 冻结 UpstreamContributionRecord | `upstream.freeze_contribution`, `records.UpstreamContributionRecord` |

## E15-11 — 目标岗位读者 5 分钟首屏理解实验

> claim boundary: E15-11 只能改变信息架构；不改变成熟度或性能数字，不提供统计代表性

| 步骤 | 协议步骤 | 代码接口 |
|---|---|---|
| 1 | 冻结 UsabilityStudyContract | `first_impression.UsabilityStudyContract` |
| 2 | 定义主要读者画像 | `first_impression.ParticipantProfile`, `first_impression.ROLE_BLOCKS` |
| 3 | 定义排除标准 | `first_impression.ParticipantProfile.excluded` |
| 4 | 确定参与者数量与边界 | `first_impression.MAX_PARTICIPANTS` |
| 5 | 设计统一入口 | `first_impression.UsabilityStudyContract.entry_point` |
| 6 | 冻结页面候选版本 | `docs_gate.PageInventory`, `first_impression.UsabilityStudyContract` |
| 7 | 设计非引导性场景 | `first_impression.TaskSet`, `first_impression.TaskSet.leading` |
| 8 | 定义核心复述问题 | `first_impression.TaskSet`, `first_impression.TASK_GROUPS` |
| 9 | 定义状态边界问题 | `first_impression.TaskSet` |
| 10 | 定义证据定位任务 | `first_impression.TaskSet`, `telemetry.EvidenceLookupTrial` |
| 11 | 定义贡献归属问题 | `first_impression.TaskSet`, `contracts.ContributionRecord` |
| 12 | 定义限制与可信度问题 | `first_impression.TaskSet`, `claims.ClaimRecord.limitations` |
| 13 | 建立答案 key | `first_impression.AnswerKey`, `first_impression.SCORE_CODES` |
| 14 | 建立行为指标 | `first_impression.ParticipantSession.path`, `telemetry.EvidenceLookupTrial` |
| 15 | 选择自然浏览或 think-aloud 模式 | `first_impression.SESSION_MODES` |
| 16 | 设计事后回放访谈 | `first_impression.ParticipantSession.debrief` |
| 17 | 准备隐私与同意 | `first_impression.UsabilityStudyContract.privacy_policy` |
| 18 | 校准记录工具 | `first_impression.UsabilityStudyContract.recording_mode` |
| 19 | 执行主持人标准化训练 | `first_impression.TaskSet.leading` |
| 20 | 运行 pilot | `first_impression.UsabilityStudyContract`, `first_impression.TaskSet` |
| 21 | 冻结修订后的协议 | `first_impression.UsabilityStudyContract.task_set_version` |
| 22 | 采集参与者背景基线 | `first_impression.ParticipantProfile` |
| 23 | 开始五分钟自然浏览 | `first_impression.BROWSE_LIMIT_S`, `first_impression.ParticipantSession` |
| 24 | 记录首屏和 first-click | `first_impression.ParticipantSession.first_click` |
| 25 | 记录完整浏览路径 | `first_impression.ParticipantSession.path` |
| 26 | 五分钟强制停止 | `first_impression.ParticipantSession.forced_stop` |
| 27 | 无页面条件下自由复述 | `first_impression.ParticipantSession.free_recall` |
| 28 | 执行结构化理解问题 | `first_impression.code_answers` |
| 29 | 重新开放页面做证据任务 | `first_impression.ParticipantSession.evidence_lookup_time_s` |
| 30 | 执行 micro/model/service 区分题 | `first_impression.ParticipantSession.task_answers` |
| 31 | 执行状态标签理解题 | `first_impression.ParticipantSession.status_label_understanding` |
| 32 | 执行 attribution 题 | `first_impression.ParticipantSession.attribution_understanding` |
| 33 | 执行限制发现题 | `first_impression.ParticipantSession.limitation_found` |
| 34 | 采集继续阅读意向及理由 | `first_impression.ParticipantSession.continue_intention` |
| 35 | 执行事后路径回放 | `first_impression.ParticipantSession.debrief` |
| 36 | 独立编码答案 | `first_impression.code_answers`, `first_impression.AnswerKey` |
| 37 | 编码误解根因 | `first_impression.encode_misconceptions`, `first_impression.MISCONCEPTION_SEVERITIES` |
| 38 | 分析岗位 block 差异 | `first_impression.role_block_analysis`, `first_impression.ROLE_BLOCKS` |
| 39 | 确定 P0 信息架构修复 | `first_impression.RevisionPlan` |
| 40 | 检查修订不改变事实 | `first_impression.check_no_fact_change`, `first_impression.RevisionPlan` |
| 41 | 生成新文档候选 | `contracts.new_candidate`, `first_impression.RevisionPlan` |
| 42 | 用新参与者复测 | `first_impression.retest_scope` |
| 43 | 检查意外退化 | `first_impression.unexpected_regression` |
| 44 | 裁决多数与关键误解 | `first_impression.role_block_analysis`, `first_impression.RevisionPlan` |
| 45 | 冻结 FirstImpressionStudyRecord | `first_impression.freeze_study`, `records.FirstImpressionSessionResult` |
