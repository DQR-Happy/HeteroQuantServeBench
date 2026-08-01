#!/usr/bin/env python3
"""Package verified text evidence for an already-running top-level-only index.

Run remotely. Original raw files, verdicts and manifests remain untouched.
Markdown containers preserve each source text verbatim between explicit fences;
the external delivery manifest stores exact UTF-8 byte offsets and SHA256s.
This is a compatibility delivery, not a replacement scientific data format.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path("docs/stage_experiments/S05")
TARGET_BYTES = 900_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    validation_path = root / STAGE / "E05-10/raw/frontend_api_validation.json"
    validation_data = validation_path.read_bytes()
    validation = json.loads(validation_data)
    if not validation.get("archive_and_updated_api_passed"):
        raise ValueError("Only already-validated archived text may be packaged")
    output = root / STAGE / "delivery_20260921/browser_manifest.json"
    manifest = {"schema": "hqsb.s05.browser-packaging/v1", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "scope": "UTF-8 text compatibility containers; original scientific evidence unchanged",
                "validator_snapshot_sha256": digest(validation_data),
                "collector_sha256": digest(Path(__file__).read_bytes()), "bundles": []}
    for experiment in validation["experiments"]:
        identifier = experiment["experiment"]
        if identifier < "E05-05":
            continue
        sources = [row for row in experiment["attachments"]
                   if not row["path"].endswith((f"/{identifier}_实验报告.md", "/raw/verdict.json"))]
        groups, group, group_size = [], [], 0
        for row in sources:
            source = (root / row["path"]).resolve()
            if not source.is_relative_to(root / STAGE / identifier):
                raise ValueError("Source escaped its experiment")
            data = source.read_bytes()
            data.decode("utf-8")
            if digest(data) != row["sha256"] or len(data) != row["bytes"] or not row["passed"]:
                raise ValueError(f"Source changed after validation: {row['path']}")
            if group and group_size + len(data) > TARGET_BYTES:
                groups.append(group); group, group_size = [], 0
            group.append((row, data)); group_size += len(data)
        if group:
            groups.append(group)
        for index, group in enumerate(groups, 1):
            path = root / STAGE / identifier / f"{identifier}_原始证据汇编_{index:02d}.md"
            content = bytearray((f"# {identifier} 原始文本证据汇编 {index:02d}\n\n"
                "这是供现有证据中心读取的兼容容器，不改变实验结论。每项内容原样收录，"
                "来源路径、字节数与 SHA256 可核对；下载的是本汇编 Markdown。"
                "JSONL/CSV/JSON 原文件仍保存在标出的相对路径，新索引加载后可独立下载。"
                "科学判定请同时阅读实验报告和 raw/verdict.json。\n\n").encode())
            entries = []
            for row, data in group:
                text = data.decode("utf-8")
                fence = "`" * max(3, 1 + max((len(s) for s in re.findall(r"`+", text)), default=0))
                suffix = Path(row["path"]).suffix.lstrip(".")
                intro = (f"## {row['path']}\n\n"
                         f"原件字节：{len(data)}；SHA256：`{row['sha256']}`。\n\n{fence}{suffix}\n").encode()
                content.extend(intro)
                start = len(content)
                content.extend(data)
                end = len(content)
                content.extend((("" if data.endswith(b"\n") else "\n") + fence + "\n\n").encode())
                entries.append({"source": row["path"], "bytes": len(data), "sha256": row["sha256"],
                                "start_byte": start, "end_byte": end})
            if len(content) > 8_000_000:
                raise ValueError("Container exceeds browser attachment limit")
            path.write_bytes(content)
            manifest["bundles"].append({"path": str(path.relative_to(root)), "bytes": len(content),
                                         "sha256": digest(content), "sources": entries})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"bundles": len(manifest["bundles"]),
                      "source_files": sum(len(b["sources"]) for b in manifest["bundles"]),
                      "source_bytes": sum(s["bytes"] for b in manifest["bundles"] for s in b["sources"]),
                      "manifest": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
