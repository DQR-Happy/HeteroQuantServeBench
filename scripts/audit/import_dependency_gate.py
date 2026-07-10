#!/usr/bin/env python3
"""Import dependency architecture gate for HQSB (E01-05).

A standalone, CI-callable static checker that scans the Python module import
graph and enforces architectural ownership rules and cycle-freedom.

What it does
------------
1. Scans every ``.py`` file under the configured namespace/path pairs and
   extracts import edges with ``ast``, recording for each edge:
   - source module, target module, file, line number,
   - import form (``import`` / ``from`` / ``dynamic``),
   - import location (top / function / class / condition / type_checking),
   - relative-import level.
2. Resolves each target module to a dotted name and classifies it as
   *local* (belongs to a scanned namespace) or *external*.
3. Maps every local module to an architectural **region** and applies the
   versioned ownership rules (R1: ``core`` must not depend on any concrete /
   implementation region).
4. Detects cycles over local edges using Tarjan's strongly-connected
   components (SCC) and reports the closed path, not just a count.
5. Reports unresolved dynamic imports and syntax errors instead of silently
   dropping edges.
6. Exits non-zero on violations, cycles, syntax errors, or an empty scan
   (no-tautology / no-silent-pass guards).

Exit codes
----------
0  PASS  (no violations, no cycles, non-empty scan, all files parsed)
1  FAIL  (one or more ownership violations or cycles detected)
2  ERROR (empty scan or at least one file failed to parse)

Usage
-----
    python3 scripts/audit/import_dependency_gate.py                 # scan repo default
    python3 scripts/audit/import_dependency_gate.py \
        --scan hqsb:/path/to/hqsb --scan ops:/path/to/ops \
        --report-path out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

GATE_NAME = "import_dependency_gate"
RULES_VERSION = "1.8.0"

#: Architectural regions. A module belongs to the first region whose dotted
#: prefix matches it. Order matters (most specific first).
REGION_PREFIXES: List[Tuple[str, str]] = [
    ("core", "hqsb.core"),
    ("benchmark", "hqsb.benchmark"),
    ("backends", "hqsb.backends"),
    ("models", "hqsb.models"),
    ("hardware", "hqsb.hardware"),
    ("quant", "hqsb.quant"),
    ("integration", "hqsb.integration"),
    ("runtime", "hqsb.runtime"),
    ("serving", "hqsb.serving"),
    ("distributed", "hqsb.distributed"),
    ("compiler", "hqsb.compiler"),
    ("evaluation", "hqsb.evaluation"),
    ("infra", "hqsb.infra"),
    ("experimental", "hqsb.experimental"),
    ("ops", "ops"),
]

#: Ownership rules (hard gates). ``source_regions`` must not import
#: ``forbidden_target_regions``.
RULES: List[Dict[str, Any]] = [
    {
        "id": "R1",
        "name": "core_no_concrete_dependency",
        "description": (
            "core defines the stable contracts/config/registry/errors surface; "
            "it must not import any concrete backend, model loader, benchmark "
            "executor, hardware adapter, quant or serving implementation."
        ),
        "source_regions": ["core"],
        "forbidden_target_regions": [
            "benchmark",
            "backends",
            "models",
            "hardware",
            "quant",
            "integration",
            "serving",
            "ops",
        ],
    },
    {
        "id": "R2",
        "name": "lower_layers_do_not_depend_on_integration",
        "description": (
            "hqsb.integration is the framework-integration layer (S06); the "
            "stable contracts, model artifacts and measurement kernel must not "
            "depend on it (the dependency arrow points core ← models/benchmark "
            "← backends/integration/quant)."
        ),
        "source_regions": ["core", "models", "benchmark"],
        "forbidden_target_regions": ["integration"],
    },
    {
        "id": "R3",
        "name": "lower_layers_do_not_depend_on_runtime",
        "description": (
            "hqsb.runtime is the request/KV/scheduling layer (S07); it sits above "
            "the contracts, model, benchmark, backend, hardware, quantization and "
            "integration layers, so none of them may import it (the dependency "
            "arrow points core ← models/benchmark ← backends/integration/quant "
            "← runtime ← serving)."
        ),
        "source_regions": [
            "core",
            "models",
            "benchmark",
            "backends",
            "hardware",
            "integration",
            "quant",
        ],
        "forbidden_target_regions": ["runtime"],
    },
    {
        "id": "R4",
        "name": "runtime_does_not_import_kernel_implementations",
        "description": (
            "Kernels are addressed by capability/provider names, never by "
            "importing the ops package (the same rule S06 follows for "
            "integration); a runtime adapter that needs a kernel declares it as a "
            "provider string."
        ),
        "source_regions": ["runtime"],
        "forbidden_target_regions": ["ops"],
    },
    {
        "id": "R5",
        "name": "serving_does_not_import_kernel_implementations",
        "description": (
            "hqsb.serving addresses kernels through the Backend/Runtime capability "
            "names, never by importing the ops package; the service core must run on "
            "the CPU-minimal installation."
        ),
        "source_regions": ["serving"],
        "forbidden_target_regions": ["ops"],
    },
    {
        "id": "R6",
        "name": "lower_layers_do_not_depend_on_serving",
        "description": (
            "hqsb.serving is the service layer (S08); it sits above every other "
            "region, so core/models/benchmark/backends/hardware/integration/quant/"
            "runtime may not import it (the dependency arrow points core ← "
            "models/benchmark ← backends/integration/quant ← runtime ← serving)."
        ),
        "source_regions": [
            "core",
            "models",
            "benchmark",
            "backends",
            "hardware",
            "integration",
            "quant",
            "runtime",
        ],
        "forbidden_target_regions": ["serving"],
    },
    {
        "id": "R7",
        "name": "lower_layers_do_not_depend_on_distributed",
        "description": (
            "hqsb.distributed is the multi-device communication layer (S10); it sits "
            "above every other region, so core/models/benchmark/backends/hardware/"
            "integration/quant/runtime/serving may not import it (the dependency arrow "
            "points core ← models/benchmark ← backends/integration/quant ← runtime ← "
            "serving ← distributed)."
        ),
        "source_regions": [
            "core",
            "models",
            "benchmark",
            "backends",
            "hardware",
            "integration",
            "quant",
            "runtime",
            "serving",
        ],
        "forbidden_target_regions": ["distributed"],
    },
    {
        "id": "R8",
        "name": "distributed_does_not_import_kernel_implementations",
        "description": (
            "hqsb.distributed addresses kernels and collective backends through capability/"
            "provider names, never by importing the ops package; the distributed core must "
            "run on the CPU-minimal installation."
        ),
        "source_regions": ["distributed"],
        "forbidden_target_regions": ["ops"],
    },
    {
        "id": "R9",
        "name": "lower_layers_do_not_depend_on_compiler",
        "description": (
            "hqsb.compiler is the graph→IR→lowering→autotune layer (S11); it consumes the "
            "stable contracts plus the integration/runtime/quant contracts, so core/models/"
            "benchmark/backends/hardware/integration/quant/runtime/serving/distributed may "
            "not import it (a lower layer importing the compiler would bind "
            "'measurable' code to the 'optimiser')."
        ),
        "source_regions": [
            "core",
            "models",
            "benchmark",
            "backends",
            "hardware",
            "integration",
            "quant",
            "runtime",
            "serving",
            "distributed",
        ],
        "forbidden_target_regions": ["compiler"],
    },
    {
        "id": "R10",
        "name": "compiler_does_not_import_kernel_implementations",
        "description": (
            "hqsb.compiler addresses kernels through capability/provider names plus artifact "
            "locators/hashes, never by importing the ops package; the compiler layer must run "
            "on the CPU-minimal installation and its lowering registry must stay "
            "side-effect free at import time."
        ),
        "source_regions": ["compiler"],
        "forbidden_target_regions": ["ops"],
    },
    {
        "id": "R11",
        "name": "lower_layers_do_not_depend_on_evaluation",
        "description": (
            "hqsb.evaluation is the cross-hardware evaluation and unified benchmark layer (S12); "
            "it consumes the evidence of every earlier stage, so core/models/benchmark/backends/"
            "hardware/quant/integration/runtime/serving/distributed/compiler may not import it "
            "(a lower layer importing the evaluator would let the 'measured' code depend on the "
            "'measurer')."
        ),
        "source_regions": [
            "core",
            "models",
            "benchmark",
            "backends",
            "hardware",
            "quant",
            "integration",
            "runtime",
            "serving",
            "distributed",
            "compiler",
        ],
        "forbidden_target_regions": ["evaluation"],
    },
    {
        "id": "R12",
        "name": "evaluation_does_not_import_kernel_implementations",
        "description": (
            "hqsb.evaluation addresses kernels/backends through capability/provider names plus "
            "artifact locators, never by importing the ops package; the evaluation layer must run "
            "on the CPU-minimal installation (no module-level torch/triton/numpy) so that the "
            "comparability/capability/cost/Pareto/lineage contracts are testable without a device."
        ),
        "source_regions": ["evaluation"],
        "forbidden_target_regions": ["ops"],
    },
    {
        "id": "R13",
        "name": "lower_layers_do_not_depend_on_infra",
        "description": (
            "hqsb.infra is the production/cloud-native/reliability layer (S13): it consumes the "
            "service (S08) and evaluation (S12) evidence, so core/models/benchmark/backends/hardware/"
            "quant/integration/runtime/serving/distributed/compiler/evaluation may not import it "
            "(a lower layer importing the deployer would bind 'measurable' code to the 'deployed')."
        ),
        "source_regions": [
            "core",
            "models",
            "benchmark",
            "backends",
            "hardware",
            "quant",
            "integration",
            "runtime",
            "serving",
            "distributed",
            "compiler",
            "evaluation",
        ],
        "forbidden_target_regions": ["infra"],
    },
    {
        "id": "R14",
        "name": "infra_does_not_import_kernel_implementations",
        "description": (
            "hqsb.infra addresses containers, models and kernels through identity/digest records, "
            "never by importing the ops package; the infra layer must run on the CPU-minimal "
            "installation (no module-level torch/triton/numpy) so the release/supply-chain/"
            "lifecycle/capacity/fault/canary/security contracts stay testable without a device."
        ),
        "source_regions": ["infra"],
        "forbidden_target_regions": ["ops"],
    },
    {
        "id": "R15",
        "name": "lower_layers_do_not_depend_on_experimental",
        "description": (
            "hqsb.experimental is the train-serve coordination and frontier-extension layer (S14): "
            "it consumes the contracts plus the S07/S08/S10/S13 evidence, so core/models/benchmark/"
            "backends/hardware/quant/integration/runtime/serving/distributed/compiler/evaluation/infra "
            "may not import it (a lower layer importing the experimental layer would let a "
            "feature-gated research path leak into the mainline install)."
        ),
        "source_regions": [
            "core",
            "models",
            "benchmark",
            "backends",
            "hardware",
            "quant",
            "integration",
            "runtime",
            "serving",
            "distributed",
            "compiler",
            "evaluation",
            "infra",
        ],
        "forbidden_target_regions": ["experimental"],
    },
    {
        "id": "R16",
        "name": "experimental_does_not_import_kernel_implementations",
        "description": (
            "hqsb.experimental addresses trainers, engines, kernels and devices through capability/"
            "provider names plus artifact identities, never by importing the ops package; the layer "
            "must run on the CPU-minimal installation (no module-level torch/triton/numpy/transformers/"
            "ray/vllm) so that the train-serve/rollout/frontier/transfer contracts stay testable "
            "without a device (E14-01: core-only 环境必须不拉实验重依赖)."
        ),
        "source_regions": ["experimental"],
        "forbidden_target_regions": ["ops"],
    },
]


def region_of(module: str) -> Optional[str]:
    """Return the architectural region for a dotted module name."""
    for region, prefix in REGION_PREFIXES:
        if module == prefix or module.startswith(prefix + "."):
            return region
    return None


def _is_dynamic_import_call(node: ast.Call) -> bool:
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    ):
        return True
    if isinstance(func, ast.Name) and func.id == "__import__":
        return True
    return False


def _dynamic_arg(node: ast.Call) -> str:
    if node.args and isinstance(node.args[0], ast.Constant):
        return node.args[0].value if isinstance(node.args[0].value, str) else "<non-str>"
    return "<dynamic>"


def _is_type_checking_test(test: ast.expr) -> bool:
    try:
        src = ast.unparse(test)
    except Exception:
        return False
    return "TYPE_CHECKING" in src


def extract_imports(source_module: str, text: str, filename: str) -> Dict[str, Any]:
    """Extract import edges from one module's source.

    Returns ``{"edges": [...], "dynamic": [...], "parse_error": null|str}``.
    Each edge record carries enough info to be independently re-checked.
    """
    result: Dict[str, Any] = {"edges": [], "dynamic": [], "parse_error": None}
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as exc:
        result["parse_error"] = f"{filename}:{exc.lineno}: {exc.msg}"
        return result

    def walk(node: ast.AST, ctx: str, typechecking: bool):
        if isinstance(node, ast.Import):
            for alias in node.names:
                form = "type_checking" if typechecking else ctx
                result["edges"].append(
                    {
                        "source": source_module,
                        "kind": "import",
                        "module": alias.name,
                        "level": 0,
                        "line": node.lineno,
                        "form": form,
                        "file": filename,
                    }
                )
            return
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                form = "type_checking" if typechecking else ctx
                result["edges"].append(
                    {
                        "source": source_module,
                        "kind": "from",
                        "module": node.module,
                        "level": node.level,
                        "line": node.lineno,
                        "form": form,
                        "file": filename,
                    }
                )
            else:
                # ``from . import x`` / ``from .. import x``
                form = "type_checking" if typechecking else ctx
                for alias in node.names:
                    result["edges"].append(
                        {
                            "source": source_module,
                            "kind": "from",
                            "module": None,
                            "level": node.level,
                            "imported_name": alias.name,
                            "line": node.lineno,
                            "form": form,
                            "file": filename,
                        }
                    )
            return
        if isinstance(node, ast.Call) and _is_dynamic_import_call(node):
            result["dynamic"].append(
                {
                    "source": source_module,
                    "arg": _dynamic_arg(node),
                    "line": getattr(node, "lineno", None),
                    "file": filename,
                }
            )

        new_ctx = ctx
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_ctx = "function"
        elif isinstance(node, ast.ClassDef):
            new_ctx = "class"
        elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With)) and ctx == "top":
            new_ctx = "condition"

        new_tc = typechecking
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            new_tc = True

        for child in ast.iter_child_nodes(node):
            walk(child, new_ctx, new_tc)

    walk(tree, "top", False)
    return result


def _module_from_rel_path(namespace: str, rel_parts: Sequence[str]) -> Optional[str]:
    parts = list(rel_parts)
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
        if not parts:
            return namespace  # the namespace root package ``__init__.py``
    else:
        parts[-1] = parts[-1][:-3]  # strip .py
    return ".".join([namespace] + parts)


def iter_modules(scans: Sequence[Tuple[str, Path]]) -> Iterable[Tuple[str, Path, bool]]:
    """Yield ``(module_name, file_path, is_package)`` for every python file."""
    for namespace, root in scans:
        if not root.is_dir():
            continue
        for pyfile in sorted(root.rglob("*.py")):
            if "__pycache__" in pyfile.parts:
                continue
            try:
                rel = pyfile.relative_to(root)
            except ValueError:
                continue
            is_package = rel.name == "__init__.py"
            module = _module_from_rel_path(namespace, rel.parts)
            if module is None:
                continue
            yield module, pyfile, is_package


def _parent(dotted: str) -> Optional[str]:
    parts = dotted.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def resolve_target(
    source_module: str, edge: Dict[str, Any], package_names: Set[str]
) -> Optional[str]:
    """Resolve an edge to a dotted target module name (or None if external-only).

    For relative imports (``level > 0``) the base package is the module's
    ``__package__`` — the module itself for an ``__init__.py``, otherwise its
    parent — then walked up ``level - 1`` more levels.
    """
    kind = edge["kind"]
    if kind == "import":
        return edge["module"]
    if kind == "from":
        level = edge.get("level", 0)
        module = edge.get("module")
        if level == 0:
            return module
        base = source_module if source_module in package_names else _parent(source_module)
        for _ in range(level - 1):
            base = _parent(base) if base is not None else None
        if module:
            return base + "." + module if base else module
        imported = edge.get("imported_name")
        if imported:
            return base + "." + imported if base else imported
        return base
    return None


def _is_local(target: Optional[str], namespaces: Set[str]) -> bool:
    if not target:
        return False
    for ns in namespaces:
        if target == ns or target.startswith(ns + "."):
            return True
    return False


def _scc(nodes: Sequence[str], adjacency: Dict[str, Set[str]]) -> List[List[str]]:
    """Tarjan SCC over the local dependency graph."""
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {n: False for n in nodes}
    stack: List[str] = []
    counter = [0]
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        index[v] = counter[0]
        lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adjacency.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(sorted(comp))

    for n in nodes:
        if n not in index:
            strongconnect(n)
    return sccs


def find_cycles(
    local_edges: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Return ``(cycles, scc_map)`` where cycles are SCCs of size>1 or self-loops."""
    nodes: Set[str] = set()
    adjacency: Dict[str, Set[str]] = {}
    edge_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in local_edges:
        s = e["source"]
        t = e["target"]
        nodes.add(s)
        nodes.add(t)
        adjacency.setdefault(s, set()).add(t)
        edge_by_pair[(s, t)] = e

    sccs = _scc(sorted(nodes), adjacency)
    cycles: List[Dict[str, Any]] = []
    scc_map: Dict[str, List[str]] = {}
    for comp in sccs:
        is_cycle = False
        if len(comp) > 1:
            is_cycle = True
        elif len(comp) == 1 and comp[0] in adjacency.get(comp[0], ()):
            is_cycle = True
        scc_map.setdefault(str(len(comp) > 1), [])
        if is_cycle:
            # Build a closed path through the component.
            path = list(comp)
            edges_in_cycle = []
            for i, a in enumerate(path):
                b = path[(i + 1) % len(path)]
                if (a, b) in edge_by_pair:
                    edges_in_cycle.append(edge_by_pair[(a, b)])
                elif len(path) == 1 and (a, a) in edge_by_pair:
                    edges_in_cycle.append(edge_by_pair[(a, a)])
            cycles.append(
                {
                    "modules": comp,
                    "size": len(comp),
                    "closed_path": path + [path[0]] if len(path) > 1 else [path[0], path[0]],
                    "edges": edges_in_cycle,
                }
            )
    return cycles, {}


def scan(
    scans: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    """Run the full scan and return a structured report."""
    namespaces = {ns for ns, _ in scans}
    edges: List[Dict[str, Any]] = []
    dynamic: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    modules: List[str] = []
    files: List[str] = []
    package_names: Set[str] = set()

    for module, path, is_package in iter_modules(scans):
        modules.append(module)
        files.append(str(path))
        if is_package:
            package_names.add(module)
        text = path.read_text(encoding="utf-8")
        extracted = extract_imports(module, text, str(path))
        edges.extend(extracted["edges"])
        dynamic.extend(extracted["dynamic"])
        if extracted["parse_error"]:
            parse_errors.append(extracted["parse_error"])

    # Resolve + classify every edge.
    resolved_edges: List[Dict[str, Any]] = []
    for e in edges:
        target = resolve_target(e["source"], e, package_names)
        local = _is_local(target, namespaces)
        resolved_edges.append(
            {
                "source": e["source"],
                "target": target,
                "kind": e["kind"],
                "level": e["level"],
                "line": e["line"],
                "form": e["form"],
                "file": e["file"],
                "local": local,
                "source_region": region_of(e["source"]),
                "target_region": region_of(target) if local else None,
            }
        )

    local_edges = [e for e in resolved_edges if e["local"]]

    # Ownership rule violations.
    violations: List[Dict[str, Any]] = []
    for rule in RULES:
        for e in resolved_edges:
            if (
                e["local"]
                and e["source_region"] in rule["source_regions"]
                and e["target_region"] in rule["forbidden_target_regions"]
            ):
                violations.append(
                    {
                        "rule_id": rule["id"],
                        "rule_name": rule["name"],
                        "source": e["source"],
                        "source_region": e["source_region"],
                        "target": e["target"],
                        "target_region": e["target_region"],
                        "file": e["file"],
                        "line": e["line"],
                        "form": e["form"],
                        "kind": e["kind"],
                    }
                )

    cycles, _ = find_cycles(local_edges)

    ownership: Dict[str, str] = {}
    for m in sorted(set(modules)):
        ownership[m] = region_of(m) or "unknown"

    files_scanned = len(files)
    status = "PASS"
    exit_code = 0
    if files_scanned == 0:
        status = "ERROR"
        exit_code = 2
    elif parse_errors:
        status = "ERROR"
        exit_code = 2
    elif violations or cycles:
        status = "FAIL"
        exit_code = 1

    report = {
        "gate": GATE_NAME,
        "rules_version": RULES_VERSION,
        "rules": RULES,
        "scans": [[ns, str(path)] for ns, path in scans],
        "scan_stats": {
            "files_scanned": files_scanned,
            "modules_scanned": len(modules),
            "total_edges": len(resolved_edges),
            "local_edges": len(local_edges),
            "external_edges": len(resolved_edges) - len(local_edges),
            "dynamic_imports": len(dynamic),
            "parse_errors": len(parse_errors),
        },
        "ownership": ownership,
        "regions": [r for r, _ in REGION_PREFIXES],
        "edges": sorted(
            resolved_edges,
            key=lambda e: (e["source"], e["target"] or "", e["line"]),
        ),
        "local_edges": sorted(
            local_edges,
            key=lambda e: (e["source"], e["target"], e["line"]),
        ),
        "dynamic": sorted(dynamic, key=lambda d: (d["source"], d["line"] or 0)),
        "parse_errors": parse_errors,
        "violations": violations,
        "cycles": cycles,
        "verdict": {
            "status": status,
            "exit_code": exit_code,
            "violation_count": len(violations),
            "cycle_count": len(cycles),
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HQSB import dependency architecture gate.")
    parser.add_argument(
        "--scan",
        action="append",
        default=[],
        metavar="NAMESPACE:PATH",
        help="Scan a namespace/path pair (repeatable). Defaults to hqsb:<repo>/hqsb and ops:<repo>/ops.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional file path to write the JSON report.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in the console summary.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.scan:
        scans: List[Tuple[str, Path]] = []
        for item in args.scan:
            ns, _, p = item.partition(":")
            if not p:
                print(f"error: --scan must be NAMESPACE:PATH, got {item!r}", file=sys.stderr)
                return 2
            scans.append((ns, Path(p)))
    else:
        scans = [
            ("hqsb", _REPO_ROOT / "hqsb"),
            ("ops", _REPO_ROOT / "ops"),
        ]

    report = scan(scans)

    if args.report_path:
        Path(args.report_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    verdict = report["verdict"]
    print(
        f"[{GATE_NAME}] rules={RULES_VERSION} files={report['scan_stats']['files_scanned']} "
        f"modules={report['scan_stats']['modules_scanned']} edges={report['scan_stats']['local_edges']} "
        f"violations={verdict['violation_count']} cycles={verdict['cycle_count']} "
        f"status={verdict['status']}"
    )
    for v in report["violations"]:
        print(
            f"  [VIOLATION {v['rule_id']}] {v['source']} -> {v['target']} "
            f"({v['file']}:{v['line']}, form={v['form']})"
        )
    for c in report["cycles"]:
        print(f"  [CYCLE] {' -> '.join(c['closed_path'])}")
    for pe in report["parse_errors"]:
        print(f"  [PARSE-ERROR] {pe}")

    return verdict["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
