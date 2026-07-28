"""Evidence-backed hypotheses and portable reports, never automatic speedup claims."""

from __future__ import annotations

import json


def optimization_report(run: dict, trace: dict) -> dict:
    metrics = run.get("metrics", {})
    observation = metrics.get("observation") or {}
    profile = observation.get("profile") or {}
    findings = []

    def add(key, title, evidence, action, validation, priority="P1"):
        findings.append(
            dict(
                id=key,
                title=title,
                evidence=evidence,
                action=action,
                validation=validation,
                priority=priority,
            )
        )

    if run.get("kind") == "quantize":
        add(
            "artifact_only",
            "制品生成与部署验证分开",
            {"state": run.get("state"), "quality": run.get("quality", "not_evaluated")},
            "固定校准/评估集；验证模型质量，再接入兼容的 packed 权重执行器。",
            "检查实际 dispatch、全量 FP16 materialization、fallback、稳态与峰值内存；不能用文件压缩率代替部署收益。",
            "P0",
        )
    rows = profile.get("operators") or []
    if rows:
        device_rows = [
            r
            for r in rows
            if isinstance(r.get("self_cuda_ms"), (int, float)) and r["self_cuda_ms"] > 0
        ]
        key = "self_cuda_ms" if device_rows else "self_cpu_ms"
        top = sorted(device_rows or rows, key=lambda r: r.get(key) or 0, reverse=True)[
            :5
        ]
        add(
            "observed_hotspots",
            "优先验证采集窗口内的热点",
            {"time_basis": key, "coverage": profile.get("coverage"), "operators": top},
            "按阶段、形状和真实 kernel 身份选择最小复现；用硬件计数器验证访存/算力假设。",
            "独立无详录 baseline、数值正确性、模型回接和重复端到端实验；累计算子时间不等于请求墙钟占比。",
        )
    elif run.get("kind") == "generate":
        add(
            "capture_needed",
            "底层热点尚无证据",
            {"profile_status": profile.get("status", "not_collected")},
            "对新请求选择算子详录，先检查窗口内 CPU/GPU 覆盖；保留当前请求作为独立记录。",
            "新采集产生新 run ID；输入、精度、输出长度与环境受控，不能重建未采的历史行为。",
        )
    tokens = observation.get("tokens") or []
    if tokens:
        measured = {
            key: sum(row.get(key) or 0 for row in tokens)
            for key in ("model_host_ms", "selection_ms", "detokenize_ms")
        }
        add(
            "host_path",
            "分离模型调用与输出路径成本",
            {"token_count": len(tokens), "host_totals_ms": measured},
            "分别检查 token 选择的同步、反分词、IPC、持久化和输出节奏；按一次一个变量开展消融。",
            "这些是 host 包络，包含等待；性能对照保持相同输出及取消/流式语义，不据此直接断言 GPU 瓶颈。",
        )
    memory = metrics.get("memory") or {}
    if memory:
        add(
            "memory_scope",
            "检查权重、临时峰值与保留内存",
            memory,
            "对比加载、生成、结束和卸载快照；检查 KV 增长、重复 materialization 和生命周期。",
            "allocated/reserved/RSS 不相加；同上下文、相同模型驻留条件下比较，多轮稳态后再判断泄漏。",
        )
    add(
        "quality_gate",
        "补齐质量与重复测量后再作推荐",
        {
            "quality": run.get("quality", "not_evaluated"),
            "measurement_profile": metrics.get("measurement_profile"),
        },
        "预先固定数值和任务质量门，独立测量候选/基线/消融并保存失败。",
        "单次交互与 profiler 墙钟都不证明正式 speedup；报告样本量、分布、工具扰动及适用范围。",
        "P0",
    )
    return {
        "schema_version": 1,
        "run_id": run["id"],
        "status": "hypotheses_only",
        "measurement_profile": metrics.get("measurement_profile"),
        "findings": findings,
        "constraints": [
            "仅根据已保存证据生成候选检查项；未自动修改部署或执行优化。",
            "结果不代表质量通过、硬件因果成立或优化已经完成。",
        ],
        "trace": trace,
    }


def markdown_report(report: dict) -> str:
    lines = [
        f"# HQSB 优化检查方案 · {report['run_id']}",
        "",
        "状态：待验证假设。此导出未执行优化，不是加速成果报告。",
        "",
    ]
    for finding in report["findings"]:
        lines += [
            f"## {finding['title']}",
            "",
            f"优先级：{finding['priority']}",
            "",
            "证据：",
            "",
            "```json",
            json.dumps(finding["evidence"], ensure_ascii=False, indent=2).replace(
                "```", "` ` `"
            ),
            "```",
            "",
            f"操作：{finding['action']}",
            "",
            f"验收：{finding['validation']}",
            "",
        ]
    lines += ["## 适用限制", "", *[f"- {text}" for text in report["constraints"]], ""]
    return "\n".join(lines)
