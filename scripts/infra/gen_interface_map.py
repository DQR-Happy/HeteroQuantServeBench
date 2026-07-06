#!/usr/bin/env python3
"""Generate ``docs/reports/S13_interface_map_generated.md`` (418 steps).

The mapping itself lives in ``hqsb.infra.interface_map`` (backed by each
experiment module's ``PROTOCOL_STEPS``); this script only renders it so the report
and the code cannot drift:

    python3 scripts/infra/gen_interface_map.py            # write the table
    python3 scripts/infra/gen_interface_map.py --check    # fail if it would change
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.infra import interface_map as im  # noqa: E402

TARGET = os.path.join(REPO_ROOT, "docs", "reports", "S13_interface_map_generated.md")


def render() -> str:
    result = im.resolve_interfaces()
    lines: List[str] = [
        "# S13 实验步骤 → 代码接口对照表（生成物，勿手工编辑）",
        "",
        "> 生成命令：`.venv/bin/python scripts/infra/gen_interface_map.py`",
        "> 校验命令：`.venv/bin/python scripts/infra/run_e13.py --interface-map --json`",
        "> 本文件由 `hqsb.infra.interface_map` 从各实验模块的 `PROTOCOL_STEPS` 生成；",
        "> `resolve_interfaces()` 对每个符号做导入校验，重命名会让审计失败而不是静默指向空。",
        "",
        f"总计：**{len(im.EXPERIMENTS)} 个实验 / {im.total_steps()} 步**；"
        f"接口引用 {result['references']} 处、唯一接口 {result['interfaces']} 个；"
        f"解析状态 `ok={result['ok']}`。",
        "",
        im.mapping_table_markdown(),
        "",
    ]
    for mapping in im.EXPERIMENTS:
        lines.append(im.step_table_markdown(mapping.experiment_id))
        lines.append("")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the S13 step→interface table.")
    parser.add_argument("--check", action="store_true", help="fail if the file would change")
    parser.add_argument("--out", default=TARGET, help="target markdown file")
    args = parser.parse_args(argv)
    content = render()
    if args.check:
        if not os.path.isfile(args.out):
            print(f"[interface_map] missing {args.out}")
            return 1
        with open(args.out, encoding="utf-8") as handle:
            current = handle.read()
        if current != content:
            print("[interface_map] the generated table is stale; re-run without --check")
            return 1
        print(f"[interface_map] steps={im.total_steps()} status=PASS")
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"[interface_map] wrote {args.out} ({len(content)} bytes, {im.total_steps()} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
