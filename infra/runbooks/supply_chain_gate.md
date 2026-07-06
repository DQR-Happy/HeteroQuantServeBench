# Runbook：供应链 gate 失败处置（模板，未演练）

## 触发
`hqsb-release-gate` 工作流返回非 0；或 E13-01 gate decision ∈ {FAIL, BLOCKED_VENDOR_FIX}。

## 判定顺序（不要把"SBOM 存在"当安全）
1. **identity**：release 是否 digest-pinned、image index/platform digest 是否与部署一致；
2. **reproducibility**：两次独立构建的 OCI DAG 差异是否已归因（时间戳/压缩/包索引/编译 JIT）；
3. **SBOM completeness**：filesystem 中未被 SBOM 覆盖的 `.so`/wheel/vendor runtime 是否已补齐；
4. **vulnerability**：severity + reachability + fix availability + exception（owner/期限/补偿控制）；
5. **secret/model**：构建上下文、history、layers、final FS 任一面命中即 fail；
6. **license**：unknown/conflict 必须显式处置，不得静默忽略；
7. **provenance/signature**：subject digest 必须等于部署 digest，过期即失效。

## 处置
- FAIL → release 进入 quarantine，禁止 E13-02 部署；
- BLOCKED_VENDOR_FIX → 记录 vendor 依赖与临时补偿控制，**不得**静默放行；
- 修复后必须重新生成 SBOM/扫描（旧报告不得复用），并重新过 gate。

## 证据
`supply_chain/gates/decision.json`（机器可读）+ `decision/limitations.md`；
任何人工 override 必须留 actor/reason/expiry。
