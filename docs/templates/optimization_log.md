# 优化日志：<算子/模块名称> <版本>

- 日期：YYYY-MM-DD
- 算子 / 模块：
- 版本 / 变体：v0 → v1
- 状态：Retained / Reverted / Superseded

## 目标（Goal）

本次优化要解决什么瓶颈（引用 Profile 证据）。

## 假设（Hypothesis）

为什么这个改动预期能带来收益（Amdahl 上限 / Roofline 分析）。

## 实现（Implementation）

改动要点、关键参数（block/tile/vectorization/occupancy）。

## 验证（Validation）

- 正确性：reference/differential/property 测试结果。
- 性能：shape 矩阵、加速比、退化 shape。
- Profiler：带宽 / occupancy / stall 等指标变化。

## 决策（Decision）

保留 / 回滚及理由；失败假设记录。
