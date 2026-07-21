#!/usr/bin/env python3
"""Build a source-derived file/API manual without importing project modules.

Run through scripts/remote_run.sh. Git files and visible untracked source are
catalogued individually; ignored experiment evidence is listed separately.
AST references are navigation aids, not proof of runtime integration.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
GROUP_ROLES = {
    "core": "S01 稳定契约、错误、身份、配置与注册表；由所有上层消费。",
    "models": "S00/S02 本地模型制品校验与加载；输出 tokenizer/model 给参考后端。",
    "backends": "S01/S02 C4 执行后端；接收 C1/C2，输出原始 GenerationOutput。",
    "benchmark": "S02 测量与分析；消费原始执行样本，输出指标、资源和热点证据。",
    "hardware": "S02 设备能力与 Jetson 实验环境协议。",
    "quant": "S05 量化语义、制品、校准、质量与执行路径；不等同于工业方法已完成实测。",
    "integration": "S06 算子契约、图模式、缓存与回退；真实模型闭环需另行实验。",
    "runtime": "S07 请求语义、C4 适配、KV/调度模型和证据；外部引擎集成尚不完整。",
    "serving": "S08 协议、网关、HTTP/SSE、策略与故障；默认可测试后端为 dummy。",
    "ascend": "S09 Ascend 探测、兼容性、tiling 与模拟；需要 NPU 才能验证设备执行。",
    "distributed": "S10 拓扑、并行/通信模型与证据；多设备收益需要真实集群。",
    "compiler": "S11 IR、guard、rewrite、lowering 与成本/缓存模型；真实 Qwen lowering 未验收。",
    "evaluation": "S12 跨硬件可比性、统计、能量/成本与 lineage；消费冻结证据。",
    "infra": "S13 部署/供应链/故障/容量策略与审计模型；不代表已经部署集群。",
    "experimental": "S14 隔离的训练/前沿实验契约和证据模型。",
    "release": "S15 声明、发布与复现证据门；不执行自动对外发布。",
}


def git_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return sorted(set(result.stdout.decode().strip("\0").split("\0")))


def module_name(path: str) -> str:
    name = path.removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def group_for(path: str) -> str:
    parts = Path(path).parts
    if parts[0] == "hqsb" and len(parts) > 2:
        return "hqsb-" + parts[1]
    if parts[0] in ("tests", "scripts", "ops", "configs", "docs") and len(parts) > 2:
        return parts[0] + "-" + parts[1]
    return parts[0] if len(parts) > 1 else "root"


def purpose(path: str, source: str, doc: str) -> str:
    if doc:
        return " ".join(doc.split())[:700]
    if path.endswith(".md"):
        heading = re.search(r"^# +(.+)$", source, re.M)
        return heading.group(1) if heading else "说明文档；见正文中的范围、输入与结论。"
    if path.endswith(".gitkeep"):
        return "空目录占位；无运行逻辑。"
    if path.endswith("CMakeLists.txt"):
        return "CMake 构建声明；定义同目录算子/可执行目标、依赖与测试注册。"
    if path.endswith((".yaml", ".yml", ".json")):
        comments = [
            line.lstrip("# ")
            for line in source.splitlines()[:20]
            if line.startswith("#")
        ]
        return (
            "配置/冻结词汇表/数据样例；由同域 loader 或 runner 读取。 "
            + " ".join(comments)[:450]
        )
    comments = [
        line.strip().lstrip("#/* ")
        for line in source.splitlines()[:25]
        if line.strip().startswith(("# ", "//", "/*", "* "))
    ]
    return " ".join(comments)[:600] or "仓库支持文件；根据所在目录与下列声明使用。"


def symbol_record(node: ast.AST, prefix: str = "") -> dict:
    name = prefix + node.name
    if isinstance(node, FUNCTIONS):
        signature = f"{name}({ast.unparse(node.args)})"
        if node.returns:
            signature += " -> " + ast.unparse(node.returns)
    else:
        signature = name + "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
    return {
        "name": name,
        "line": node.lineno,
        "lines": node.end_lineno - node.lineno + 1,
        "signature": signature,
        "doc": " ".join((ast.get_docstring(node) or "").split())[:500],
        "fields": [
            ast.unparse(child)
            for child in getattr(node, "body", [])
            if isinstance(node, ast.ClassDef) and isinstance(child, ast.AnnAssign)
        ],
    }


def inspect_file(path: str) -> dict:
    file = ROOT / path
    payload = file.read_bytes()
    record = {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {
            **record,
            "purpose": "二进制/图像资源；由引用它的报告或工具消费。",
            "lines": 0,
        }
    record["lines"] = len(source.splitlines())
    record["symbols"] = []
    record["imports"] = []
    record["cli"] = []
    record["duplicates"] = []
    doc = ""
    if path.endswith(".py"):
        tree = ast.parse(source, filename=path)
        doc = ast.get_docstring(tree) or ""
        for node in tree.body:
            if isinstance(node, (*FUNCTIONS, ast.ClassDef)):
                record["symbols"].append(symbol_record(node))
                if isinstance(node, ast.ClassDef):
                    record["symbols"].extend(
                        symbol_record(child, node.name + ".")
                        for child in node.body
                        if isinstance(child, FUNCTIONS)
                    )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                record["imports"].extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                record["imports"].append(node.module)
                record["imports"].extend(
                    node.module + "." + alias.name for alias in node.names
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                record["cli"].append(ast.unparse(node))
            if isinstance(node, FUNCTIONS) and node.end_lineno - node.lineno >= 15:
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                canonical = ast.dump(
                    ast.Module(body=body, type_ignores=[]), include_attributes=False
                )
                record["duplicates"].append(
                    {
                        "hash": hashlib.sha256(canonical.encode()).hexdigest(),
                        "name": node.name,
                        "line": node.lineno,
                    }
                )
    elif path.endswith((".cu", ".h", ".cuh", ".cpp")):
        record["declarations"] = [
            line.strip()
            for line in source.splitlines()
            if re.match(
                r"\s*(?:__global__|extern \"C\"|cudaError_t|void |int main|struct |enum )",
                line,
            )
        ][:80]
    record["purpose"] = purpose(path, source, doc)
    if path.endswith((".yaml", ".yml", ".json")):
        record["config_keys"] = re.findall(
            r"^([A-Za-z_][A-Za-z0-9_.-]*):", source, re.M
        )
        if path.endswith(".json"):
            try:
                payload = json.loads(source)
                record["config_keys"] = (
                    list(payload) if isinstance(payload, dict) else ["array"]
                )
            except json.JSONDecodeError:
                record["config_keys"] = ["invalid JSON; inspect source"]
    record["imports"] = sorted(set(record["imports"]))
    record["entry"] = (
        '__name__ == "__main__"' in source or "__name__ == '__main__'" in source
    )
    return record


def usage(record: dict) -> str:
    path = record["path"]
    if path.startswith("tests/"):
        return f"`./scripts/remote_run.sh python3 -m pytest {path} -q`；具体覆盖点见下列 test 符号。"
    if record.get("entry"):
        return f"Jetson 入口：`./scripts/remote_run.sh python3 {path} --help`（先核对下列参数；无 argparse 的脚本应先阅读 main）。"
    if path.startswith(("hqsb/", "ops/")) and path.endswith(".py"):
        return f"库模块：`import {module_name(path)}`；使用下列类/函数，由调用方管理资源和输入。"
    if path.endswith(".sh"):
        return "Shell 编排入口；阅读文件头参数。同步/拉取在 Mac 执行，构建/运行/探测经 remote_run.sh。"
    if path.endswith((".cu", ".h", ".cuh", ".cpp", "CMakeLists.txt")):
        return (
            "由相邻 CMake target 编译/链接；仅在对应远端硬件执行，不单独在 Mac 编译。"
        )
    if path.endswith((".yaml", ".yml", ".json")):
        return "作为配置、schema 或样例传给同域 runner/loader；冻结配置不是测量结果，不直接执行。"
    return "阅读/引用资源；结合所在目录的工作流使用，不作为程序入口。"


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/manual/generated")
    parser.add_argument(
        "--evidence-paths",
        help="newline file inventory from the local client's ignored evidence tree",
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for p in git_paths()
        if (ROOT / p).is_file()
        and not p.startswith(("docs/manual/generated/", "docs/audit/generated/"))
    ]
    records = [inspect_file(path) for path in paths]
    modules = {
        module_name(r["path"]): r["path"] for r in records if r["path"].endswith(".py")
    }
    consumers = defaultdict(set)
    duplicate_groups = defaultdict(list)
    for record in records:
        deps = sorted(
            {
                modules[name]
                for name in record.get("imports", [])
                if name in modules and modules[name] != record["path"]
            }
        )
        record["dependencies"] = deps
        for dep in deps:
            consumers[dep].add(record["path"])
        for item in record.pop("duplicates", []):
            duplicate_groups[item["hash"]].append({"path": record["path"], **item})
    groups = defaultdict(list)
    for record in records:
        record["consumers"] = sorted(consumers[record["path"]])
        groups[group_for(record["path"])].append(record)
    stats = {
        "files": len(records),
        "lines": sum(r["lines"] for r in records),
        "python_files": sum(r["path"].endswith(".py") for r in records),
        "groups": {k: len(v) for k, v in sorted(groups.items())},
        "largest_files": sorted(
            (
                {"path": r["path"], "lines": r["lines"]}
                for r in records
                if r["path"].endswith(".py")
            ),
            key=lambda r: r["lines"],
            reverse=True,
        )[:40],
        "identical_function_bodies": [
            v for v in duplicate_groups.values() if len(v) > 1
        ],
    }
    (output / "inventory.json").write_text(
        json.dumps({"stats": stats, "files": records}, ensure_ascii=False, indent=2)
        + "\n"
    )
    index = [
        "# 逐文件与 API 索引（源码生成）",
        "",
        "本目录由 `scripts/audit/project_inventory.py` 在 Jetson 静态解析生成，不导入业务模块。",
        "作用来自文件/类/函数原始说明，依赖与消费者来自显式 AST import；动态注册、字符串路径和反射不一定被覆盖。",
        "它是全文件导航，不代表逐行人工验证或所有 API 已完成真实硬件集成。设计与操作顺序请先读 [使用说明书](../使用说明书.md)。",
        "",
        f"覆盖 {len(records)} 个源码/配置/文档/测试文件；生成目录自身不递归列入。机器可读明细：`inventory.json`。",
        "",
        "| 目录组 | 文件数 |",
        "|---|---:|",
    ]
    for group, entries in sorted(groups.items()):
        index.append(f"| [{group}]({group}.md) | {len(entries)} |")
        text = [f"# {group} 文件与模块说明", "", "[返回总索引](README.md)", ""]
        if group.startswith("hqsb-"):
            text += [GROUP_ROLES.get(group[5:], "HQSB 库模块。"), ""]
        for record in entries:
            path = record["path"]
            text += [
                f"## `{path}`",
                "",
                f"[查看源文件](../../../{path}) · {record['lines']} 行",
                "",
                record["purpose"],
                "",
                "使用：" + usage(record),
                "",
            ]
            if record.get("dependencies"):
                text += [
                    "上游依赖："
                    + "、".join(f"`{p}`" for p in record["dependencies"])
                    + "。",
                    "",
                ]
            if record.get("consumers"):
                text += [
                    "下游调用/测试："
                    + "、".join(f"`{p}`" for p in record["consumers"])
                    + "。",
                    "",
                ]
            if record.get("cli"):
                text += [
                    "命令参数（从 argparse 提取）：",
                    "",
                    "```python",
                    *record["cli"],
                    "```",
                    "",
                ]
            if record.get("config_keys"):
                text += [
                    "配置入口字段："
                    + "、".join(f"`{key}`" for key in record["config_keys"])
                    + "。",
                    "",
                ]
            if record.get("declarations"):
                text += [
                    "原生接口/入口声明：",
                    "",
                    "```cpp",
                    *record["declarations"],
                    "```",
                    "",
                ]
            if record.get("symbols"):
                text += [
                    "模块接口与设计说明（私有符号为内部实现，勿作为稳定 API）：",
                    "",
                    "| 符号/签名 | 行 | 职责/语义 |",
                    "|---|---:|---|",
                ]
                for symbol in record["symbols"]:
                    doc = symbol["doc"] or "源代码未单独说明；见上级类/模块与关联测试。"
                    text.append(
                        f"| `{escape(symbol['signature'])}` | {symbol['line']} | {escape(doc)} |"
                    )
                text.append("")
                for symbol in record["symbols"]:
                    if symbol.get("fields"):
                        text += [
                            f"`{symbol['name']}` 字段/默认值：",
                            "",
                            "```python",
                            *symbol["fields"],
                            "```",
                            "",
                        ]
        (output / f"{group}.md").write_text("\n".join(text) + "\n")
    if args.evidence_paths:
        evidence = sorted(set(Path(args.evidence_paths).read_text().splitlines()))
        roles = {
            ".md": "实验方案/说明/报告；连接设计与 raw 判定",
            ".json": "结构化输入、元数据或结果；按 schema/字段解释",
            ".jsonl": "逐次/逐样本原始记录；汇总的上游",
            ".csv": "统计表或 profiler 导出；保留单位与边界",
            ".bin": "量化张量/scale/打包制品；按 manifest 的 shape/layout/位宽解释，供校验和 executor 消费",
            ".npy": "保存的数值数组；由采集/分析脚本写入，供离线统计或数值对齐读取",
            ".npz": "保存的数值数组集合；按键名由分析/对齐脚本读取",
            ".ptx": "CUDA 中间指令；用于编译/资源与指令审查，不是性能测量值",
            ".cubin": "编译后的 CUDA 二进制；供设备加载或反汇编分析，需绑定目标架构和源码身份",
            ".svg": "静态矢量图；从 raw/统计表生成",
            ".png": "图像证据/图表",
            ".pdf": "可分享图表/报告",
            ".ncu-rep": "Nsight Compute 原始 profile；在兼容工具中打开",
            ".nsys-rep": "Nsight Systems 原始 timeline；在兼容工具中打开",
            ".sh": "本机私有辅助脚本；先阅读文件头与副作用，不作为公共入口",
        }
        rows = [
            "# 本机实验与辅助文件逐路径索引",
            "",
            f"由 Mac 的文件清单生成，共 {len(evidence)} 项；不复制、不修改 raw 内容。",
            "这些路径属于本机证据归档，通常不在 Git 中；新检出不可假定存在。原始文件→汇总→verdict→报告，不能反向用报告补造 raw。",
            "",
            "| 路径 | 作用与上下游 |",
            "|---|---|",
        ]
        for path in evidence:
            name = Path(path).name
            role = roles.get(
                Path(path).suffix,
                "采集日志/辅助证据；由对应实验 runner 产生并被 verify/report 消费",
            )
            if name == "verdict.json":
                role = "实验验收判定；verify 消费 raw/协议后产出，下游阶段读取；软件测试通过不修改它"
            elif name in (
                "manifest.json",
                "file_manifest.json",
                "environment.json",
                "provenance.json",
            ):
                role = "证据身份与环境绑定；实验复现和 hash 审计输入"
            rows.append(f"| `{escape(path)}` | {role} |")
        (output / "local-evidence.md").write_text("\n".join(rows) + "\n")
        index += [
            "",
            f"本机私有实验与辅助文件：[{len(evidence)} 个路径](local-evidence.md)；独立于源码文件计数。",
        ]
    (output / "README.md").write_text("\n".join(index) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
