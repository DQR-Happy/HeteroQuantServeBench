#!/usr/bin/env python3
"""Materialise the S14 step→interface tables into ``docs/reports/S14_interface_map_generated.md``.

The tables are **generated**, never hand-copied: hand-copying is how a report
drifts from the code it describes, and the drift is invisible precisely because
the report still reads plausibly.  The generator refuses to write a table for an
experiment whose mapping is incomplete, so a missing step surfaces as a failing
command rather than as a shorter table.

Usage:
    scripts/experimental/gen_interface_map.py --write
    scripts/experimental/gen_interface_map.py --check
    scripts/experimental/gen_interface_map.py --experiment E14-F3
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.experimental import interface_map as imap  # noqa: E402

TARGET = os.path.join(REPO_ROOT, "docs", "reports", "S14_interface_map_generated.md")


def render_header() -> str:
    coverage = imap.coverage_summary()
    lines = [
        "# S14 实验步骤 → 代码接口对照表（自动生成）",
        "",
        "> 本文件由 `scripts/experimental/gen_interface_map.py` 生成，**请勿手工编辑**。",
        "> 每个接口引用都会被 `hqsb.experimental.interface_map.resolve_interfaces()` 实际导入解析，",
        "> 因此改名/删除会在 `--check` 和单元测试中直接失败，而不是静默指向空。",
        "",
        f"- 实验数：**{coverage['experiments']}**",
        f"- 步骤数：**{coverage['steps']} / {coverage['expected_steps']}**",
        f"- 覆盖完整：**{coverage['ok']}**",
        "",
        "## 总览",
        "",
        imap.mapping_table_markdown(),
        "",
        "## 逐实验步骤表",
        "",
    ]
    return "\n".join(lines)


def render_experiment(experiment_id: str) -> str:
    mapping = imap.mapping_for(experiment_id)
    if not mapping.complete:
        raise SystemExit(
            f"refusing to emit a table for an incomplete mapping: {experiment_id} "
            f"({len(mapping.steps)} steps, load_error={mapping.load_error!r})"
        )
    lines = [
        imap.step_table_markdown(experiment_id, interfaces_per_row=3),
        "",
        f"驱动入口：`{mapping.driver}`",
        "",
        f"覆盖模块：{', '.join('`' + name + '`' for name in imap.covered_prefixes(experiment_id))}",
        "",
    ]
    return "\n".join(lines)


def render(target_experiment: str = "") -> str:
    if target_experiment:
        return render_header() + render_experiment(target_experiment)
    body = [render_header()]
    for mapping in imap.EXPERIMENTS:
        body.append(render_experiment(mapping.experiment_id))
    return "\n".join(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="generate or check the S14 interface map")
    parser.add_argument("--write", action="store_true", help="write the generated report")
    parser.add_argument("--check", action="store_true", help="check the report for drift")
    parser.add_argument("--experiment", default="", help="restrict the output to one experiment")
    parser.add_argument("--target", default=TARGET)
    args = parser.parse_args()

    text = render(args.experiment)
    if args.check:
        if not os.path.exists(args.target):
            print(f"missing: {args.target}")
            return 1
        with open(args.target, encoding="utf-8") as handle:
            drift = handle.read() != text
        print(f"target: {args.target}")
        print(f"ok: {not drift}")
        return 0 if not drift else 1

    problems: List[str] = imap.steps_without_interfaces()
    if problems:
        for problem in problems:
            print(f"incomplete mapping: {problem}", file=sys.stderr)
        return 1
    result = imap.resolve_interfaces()
    if not result["ok"]:
        for failure in result["failures"][:10]:
            print(f"unresolved interface: {failure}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.target), exist_ok=True)
    with open(args.target, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"wrote {args.target}")
    print(f"steps: {result['steps']}/{result['expected_steps']} refs: {result['references']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
