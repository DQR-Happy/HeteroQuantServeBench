# `infra/` — S13 部署与可观测性资产（模板，未验证）

> 状态：**`IMPLEMENTED_UNVERIFIED` / `DESIGN_ONLY`（模板未构建、未部署）**
> 本目录提供 E13-01…E13-11 的构建、渲染、部署和扫描输入。
> 镜像、Helm 与告警的执行状态记录在 `docs/reports/S13_阶段验收报告.md`，
> 每项判定关联相应 raw 证据。

## 目录与用途

| 路径 | 用途 | 对应实验 |
|---|---|---|
| `containers/Dockerfile.runtime` | 四阶段镜像（base/toolchain/test/runtime），runtime 为非 root、只读根、最小 capability | E13-01 |
| `containers/build_args.yaml` | 构建输入闭包（base/builder digest、lockfile、网络阶段、cache 策略）；**tag 不得作为身份** | E13-01 |
| `containers/.dockerignore` | 构建上下文排除（权重、secret、缓存、`build/`、`.git`） | E13-01 |
| `deploy/helm/hqsb/` | namespace/RBAC/quota/NetworkPolicy/Deployment/Service/HPA/PDB 模板 | E13-02、E13-05、E13-07、E13-11 |
| `observability/` | 语义约定、告警规则、面板定义 | E13-08 |
| `runbooks/` | 部署、drain/回滚、故障处置、供应链 gate 的运行手册模板 | E13-05、E13-09、E13-10 |
| `ci/release_gate.yaml` | 发布门禁工作流（调用驱动 + 依赖门 + 测试），**未接入任何真实集群** | E13-01、E13-10 |

## 模板中的阈值

`values.yaml`、`alerts.yaml`、`dashboards.yaml` 中出现的数值一律是**策略默认值**，
以标记 `POLICY_DEFAULT_UNVERIFIED` 标注，并由
`tests/unit/infra/test_infra_assets.py` 检查该标记。正式运行使用 campaign
输入（`preregistration/*.yaml`）覆盖默认值，并将实际配置冻结到报告中。

## 与实验的关系

这些资产用于环境部署、生命周期管理、发布门禁与回滚。
E13-02/E13-05/E13-10 按 cluster events、probe timeline、rollout 记录与 canary 决策验收（见
`docs/stage_experiments/details/S13/*`）。
