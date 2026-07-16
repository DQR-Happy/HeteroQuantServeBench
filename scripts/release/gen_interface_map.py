#!/usr/bin/env python3
"""Render the S15 495-step interface map to Markdown.

The generated file (``docs/reports/S15_interface_map_generated.md``) is a
derived artefact: every step, its title and its resolved interface references.
It is regenerated from ``hqsb.release.interface_map`` so it cannot drift from the
code, and any unresolved reference fails the render.

Usage:
    scripts/release/gen_interface_map.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.release import interface_map as imap  # noqa: E402

OUT_PATH = os.path.join(REPO_ROOT, "docs", "reports", "S15_interface_map_generated.md")


def render() -> str:
    result = imap.resolve_interfaces()
    if not result["ok"]:
        raise SystemExit("interface map has unresolved references; refusing to render:\n" + "\n".join(result["failures"][:10]))
    lines: list[str] = []
    lines.append("# S15 实验步骤 → 代码接口对照表（自动生成）")
    lines.append("")
    lines.append("> 由 `scripts/release/gen_interface_map.py` 从 `hqsb.release.interface_map` 生成；")
    lines.append("> 本文件是派生产物，不手工编辑。")
    lines.append("")
    lines.append(f"- 实验数：{len(imap.EXPERIMENTS)}")
    lines.append(f"- 步骤数：{result['steps']} / {result['expected_steps']}")
    lines.append(f"- 接口引用数：{result['references']}")
    lines.append(f"- 解析状态：{'全部通过' if result['ok'] else '存在失败'}")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| 实验 | 级别 | 步骤 | 标题 |")
    lines.append("|---|---|---|---|")
    for mapping in imap.EXPERIMENTS:
        lines.append(f"| {mapping.experiment_id} | {mapping.level} | {len(mapping.steps)} | {mapping.title} |")
    lines.append("")
    for mapping in imap.EXPERIMENTS:
        lines.append(f"## {mapping.experiment_id} — {mapping.title}")
        lines.append("")
        lines.append(f"> claim boundary: {mapping.claim_boundary}")
        lines.append("")
        lines.append("| 步骤 | 协议步骤 | 代码接口 |")
        lines.append("|---|---|---|")
        for step in mapping.steps:
            refs = ", ".join(f"`{reference}`" for reference in step.interfaces)
            lines.append(f"| {step.index} | {step.title} | {refs} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    text = render()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
