#!/usr/bin/env python3
"""E01-05 — Module import graph, dependency direction & cycle architecture gate.

Question
--------
Can HQSB's module dependency boundaries be enforced by a *static* checker (not
just drawn on an architecture diagram)? Specifically: is the legal production
graph acyclic, does it respect the ownership rules (core must not depend on a
concrete backend / model loader / benchmark executor), and does the checker
reliably catch injected counterexamples — a core→concrete-backend edge and
cyclic dependencies — with precise source/location diagnostics?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  The legal production import graph is acyclic and respects the frozen
    ownership rules; the checker precisely locates (file:line) a direct
    core→concrete-backend violation, indirect/hidden violations (via a
    utility module, via a package ``__init__``, inside a function body, via
    relative import), and 2-node / 3-node cycles; the same checker exits 0
    on the clean graph and non-zero on every injected bad graph; and no rule
    is trivially satisfied by a tautology, empty scan, or result-swallowing.
H0  The legal graph actually contains a cycle or a core→concrete edge; or the
    checker misses a hidden/indirect violation or a cycle; or the gate returns
    0 on a bad graph.

Design (protocol §6 steps 1–10)
-------------------------------
* scan the real production graph (``hqsb`` + ``ops``) with the standalone
  gate ``scripts/audit/import_dependency_gate.py``.
* extraction-coverage fixture: import/from/alias/relative/__init__/function/
  condition/TYPE_CHECKING/dynamic forms.
* legal factory-registration control (core defines registry; plugin imports
  core; assembly entry registers; benchmark calls via interface) → must PASS.
* direct counterexample: core module imports ``hqsb.backends.pytorch``.
* indirect counterexamples: core→utility→backend, core→package.__init__→
  backend, function-body import, relative import.
* cycle counterexamples: 2-node and 3-node; remove one edge → PASS control.
* gate exit propagation: clean graph exit 0, each bad graph exit non-zero.
* no-tautology audit: gate/test source has no ``or True``; empty scan → ERROR.

Pure static AST analysis: no torch, no GPU, no model weights, no import side
effects beyond parsing.

Raw output (under <out>/)
-------------------------
``e01_05_<run_id>.json``          full record (cases + verdict)
``e01_05_<run_id>_env.json``      frozen environment / git identity
``baseline_report.json``          real-graph gate report
``extraction_coverage.json``      import-form coverage fixture result
``legal_registration_report.json`` legal control gate report
``counterexample_direct.json``    direct violation gate report
``counterexample_indirect.json``  indirect violations gate reports
``counterexample_cycles.json``    cycle counterexample gate reports
``no_tautology_audit.json``       tautology/empty-scan audit
``verdict.json``                  pass criteria + overall verdict

Usage
-----
    python3 scripts/audit/run_e01_05_import_dependency_boundaries.py \
        --output-dir docs/stage_experiments/S01/E01-05/raw
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_DIR = _REPO_ROOT / "scripts" / "audit"
if str(_AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(_AUDIT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import import_dependency_gate as gate  # noqa: E402

EXPERIMENT_ID = "E01-05"
STAGE = "S01"

#: Namespaces the gate considers "local". The real gate scans ``hqsb`` + ``ops``.
_REAL_SCANS: List[Tuple[str, Path]] = [
    ("hqsb", _REPO_ROOT / "hqsb"),
    ("ops", _REPO_ROOT / "ops"),
]


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


def collect_environment() -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(_REPO_ROOT),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── Fixture helpers ─────────────────────────────────────────────────────


def _write_files(base: Path, files: Dict[str, str]) -> None:
    """Write ``files`` (relative path → content) under ``base``."""
    for rel, content in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _scan(scans: Sequence[Tuple[str, Path]]) -> Dict[str, Any]:
    return gate.scan(list(scans))


def _run_gate_cli(scans: Sequence[Tuple[str, str]]) -> Tuple[int, str, str]:
    """Run the gate as a subprocess (the same entry CI uses) and return
    ``(returncode, stdout, stderr)``."""
    cmd = [sys.executable, str(_AUDIT_DIR / "import_dependency_gate.py")]
    for ns, path in scans:
        cmd += ["--scan", f"{ns}:{path}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    return proc.returncode, proc.stdout, proc.stderr


def _case(case_id: str, kind: str, expected: str) -> Dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "case_id": case_id,
        "case_kind": kind,
        "expected": expected,
    }


def _finish(rec: Dict[str, Any], status: str, **observed: Any) -> Dict[str, Any]:
    rec["status"] = status
    rec["observed"] = observed
    return rec


# ── Case executors ──────────────────────────────────────────────────────


def case_baseline_legal_graph() -> Dict[str, Any]:
    """Step 1/3: the unmodified production graph must be acyclic and rule-clean."""
    rec = _case(
        "baseline_legal_graph",
        "positive",
        "real graph acyclic + no ownership violation",
    )
    report = _scan(_REAL_SCANS)
    verdict = report["verdict"]
    stats = report["scan_stats"]
    ok = verdict["status"] == "PASS" and verdict["cycle_count"] == 0 and verdict["violation_count"] == 0
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        verdict=verdict,
        scan_stats=stats,
        report=report,
    )


def case_extraction_coverage() -> Dict[str, Any]:
    """Step 2: the checker must resolve ordinary/from/alias/relative/__init__/
    function/condition/TYPE_CHECKING/dynamic import forms."""
    rec = _case(
        "extraction_coverage",
        "positive",
        "all import forms extracted & classified",
    )
    with tempfile.TemporaryDirectory(prefix="hqsb_e01_05_cov_") as tmp:
        base = Path(tmp)
        _write_files(
            base,
            {
                "pkg/__init__.py": "from . import sub\n",
                "pkg/sub.py": "import os\nimport pkg.b as alias_target\n",
                "pkg/a.py": "from pkg.b import thing\n",
                "pkg/b.py": "def f():\n    import pkg.c\n",
                "pkg/c.py": "if True:\n    import pkg.a\n",
                "pkg/d.py": (
                    "from typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n"
                    "    import pkg.b\n"
                ),
                "pkg/e.py": "import importlib\nimportlib.import_module('pkg.b')\n",
                "pkg/rel.py": "from . import b\n",
            },
        )
        report = _scan([("pkg", base / "pkg")])
        edges = report["edges"]
        forms = sorted({e["form"] for e in edges})
        # We expect: top, function, condition, type_checking forms present.
        forms_ok = {"top", "function", "condition", "type_checking"} <= set(forms)
        # dynamic import recorded
        dynamic = report["dynamic"]
        dynamic_ok = any(d["arg"] == "pkg.b" for d in dynamic)
        # relative import resolved to a local target pkg.b (level=1)
        rel_edges = [e for e in edges if e["kind"] == "from" and e["level"] > 0]
        rel_ok = any(e["target"] == "pkg.b" and e["local"] for e in rel_edges)
        # alias still records the real module name (pkg.b), not the alias
        alias_ok = any(
            e["kind"] == "import" and e["target"] == "pkg.b" for e in edges
        )
        # __init__ re-export edge pkg → pkg.sub
        init_ok = any(
            e["source"] == "pkg" and e["target"] == "pkg.sub" for e in edges
        )
        ok = forms_ok and dynamic_ok and rel_ok and alias_ok and init_ok
        return _finish(
            rec,
            "PASS" if ok else "FAIL",
            forms=sorted(forms),
            forms_ok=forms_ok,
            dynamic=dynamic,
            dynamic_ok=dynamic_ok,
            rel_ok=rel_ok,
            alias_ok=alias_ok,
            init_ok=init_ok,
            edges=edges,
            scan_stats=report["scan_stats"],
        )


def case_legal_registration() -> Dict[str, Any]:
    """Step 4: a legal factory-registration graph must NOT be falsely flagged."""
    rec = _case(
        "legal_registration",
        "positive",
        "core/plugin/assembly legal graph passes",
    )
    with tempfile.TemporaryDirectory(prefix="hqsb_e01_05_legal_") as tmp:
        base = Path(tmp)
        _write_files(
            base,
            {
                # core defines the registry + C4 protocol, imports nothing concrete
                "core/__init__.py": "from hqsb.core.registry import Registry\n",
                "core/registry.py": "class Registry:\n    pass\n",
                "core/contract.py": "class Backend:\n    pass\n",
                # plugin imports only public core
                "backends/__init__.py": "from hqsb.backends.plugin import Plugin\n",
                "backends/plugin.py": (
                    "from hqsb.core.contract import Backend\n"
                    "class Plugin(Backend):\n    pass\n"
                ),
                # assembly entry (outside core) wires the plugin factory
                "entry.py": (
                    "from hqsb.backends.plugin import Plugin\n"
                    "from hqsb.core.registry import Registry\n"
                    "def main():\n"
                    "    r = Registry()\n"
                ),
            },
        )
        report = _scan([("hqsb", base)])
        verdict = report["verdict"]
        ok = (
            verdict["status"] == "PASS"
            and verdict["violation_count"] == 0
            and verdict["cycle_count"] == 0
        )
        return _finish(
            rec,
            "PASS" if ok else "FAIL",
            verdict=verdict,
            report=report,
        )


def _counterexample_direct_report() -> Dict[str, Any]:
    """Build + scan the direct counterexample fixture."""
    with tempfile.TemporaryDirectory(prefix="hqsb_e01_05_direct_") as tmp:
        base = Path(tmp)
        _write_files(
            base,
            {
                "core/__init__.py": "",
                "core/leaky.py": (
                    "from hqsb.backends.pytorch import PyTorchBackend\n"
                    "def load():\n"
                    "    return PyTorchBackend()\n"
                ),
                "backends/__init__.py": "",
                "backends/pytorch.py": "class PyTorchBackend:\n    pass\n",
            },
        )
        return _scan([("hqsb", base)])


def case_counterexample_direct() -> Dict[str, Any]:
    """Step 5: core → concrete backend direct import must be caught."""
    rec = _case(
        "counterexample_direct",
        "negative",
        "core→backend violation located (file:line)",
    )
    report = _counterexample_direct_report()
    verdict = report["verdict"]
    viol = report["violations"]
    hits = [
        v
        for v in viol
        if v["rule_id"] == "R1"
        and v["source"] == "hqsb.core.leaky"
        and v["target"] == "hqsb.backends.pytorch"
        and v["file"].endswith("core/leaky.py")
        and v["line"] == 1
    ]
    ok = verdict["status"] == "FAIL" and verdict["exit_code"] == 1 and len(hits) == 1
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        verdict=verdict,
        violations=viol,
        matched=hits,
    )


def case_counterexample_indirect() -> Dict[str, Any]:
    """Step 6: indirect/hidden positions must each be caught."""
    rec = _case(
        "counterexample_indirect",
        "negative",
        "utility/__init__/function-body/relative forms caught",
    )
    results: Dict[str, Any] = {}
    cases = {
        # core → utility module → concrete backend (transitive first hop)
        "via_utility": {
            "core/__init__.py": "from hqsb.core import util\n",
            "core/util.py": "from hqsb.backends.pytorch import PyTorchBackend\n",
            "core/public.py": "from hqsb.core.util import helper\n",
            "backends/__init__.py": "",
            "backends/pytorch.py": "class PyTorchBackend:\n    pass\n",
        },
        # core → package.__init__ → concrete backend (package init re-export)
        "via_package_init": {
            "core/__init__.py": "from hqsb.backends.pytorch import PyTorchBackend\n",
            "core/anything.py": "",
            "backends/__init__.py": "",
            "backends/pytorch.py": "class PyTorchBackend:\n    pass\n",
        },
        # function-body import (deferred, still a violation)
        "function_body": {
            "core/__init__.py": "",
            "core/loader.py": (
                "def run():\n"
                "    from hqsb.backends.pytorch import PyTorchBackend\n"
                "    return PyTorchBackend()\n"
            ),
            "backends/__init__.py": "",
            "backends/pytorch.py": "class PyTorchBackend:\n    pass\n",
        },
        # relative import expressing the same forbidden dependency
        "relative_import": {
            "core/__init__.py": "",
            "core/sneaky.py": "from ..backends import pytorch\n",
            "backends/__init__.py": "from hqsb.backends.pytorch import PyTorchBackend\n",
            "backends/pytorch.py": "class PyTorchBackend:\n    pass\n",
        },
    }
    all_ok = True
    for name, files in cases.items():
        with tempfile.TemporaryDirectory(prefix=f"hqsb_e01_05_ind_{name}_") as tmp:
            base = Path(tmp)
            _write_files(base, files)
            report = _scan([("hqsb", base)])
            verdict = report["verdict"]
            viol = report["violations"]
            # every case must be FAIL and contain a core→backends R1 violation
            hits = [
                v
                for v in viol
                if v["rule_id"] == "R1"
                and v["source_region"] == "core"
                and v["target_region"] == "backends"
            ]
            case_ok = verdict["status"] == "FAIL" and len(hits) >= 1
            results[name] = {
                "ok": case_ok,
                "verdict": verdict,
                "violations": viol,
            }
            all_ok = all_ok and case_ok
    return _finish(
        rec,
        "PASS" if all_ok else "FAIL",
        results=results,
    )


def case_counterexample_cycles() -> Dict[str, Any]:
    """Step 7: 2-node and 3-node cycles caught; edge-removed control passes."""
    rec = _case(
        "counterexample_cycles",
        "negative",
        "2-node + 3-node cycle located; removal control passes",
    )
    results: Dict[str, Any] = {}

    # 2-node cycle
    with tempfile.TemporaryDirectory(prefix="hqsb_e01_05_cyc2_") as tmp:
        base = Path(tmp)
        _write_files(
            base,
            {
                "audit_a.py": "import audit.audit_b\n",
                "audit_b.py": "import audit.audit_a\n",
            },
        )
        report = _scan([("audit", base)])
        cyc = report["cycles"]
        two_node_ok = (
            report["verdict"]["status"] == "FAIL"
            and len(cyc) == 1
            and cyc[0]["size"] == 2
            and set(cyc[0]["modules"]) == {"audit.audit_a", "audit.audit_b"}
        )
        results["two_node"] = {
            "ok": two_node_ok,
            "verdict": report["verdict"],
            "cycles": cyc,
        }

    # 3-node cycle
    with tempfile.TemporaryDirectory(prefix="hqsb_e01_05_cyc3_") as tmp:
        base = Path(tmp)
        _write_files(
            base,
            {
                "audit_a.py": "import audit.audit_b\n",
                "audit_b.py": "import audit.audit_c\n",
                "audit_c.py": "import audit.audit_a\n",
            },
        )
        report = _scan([("audit", base)])
        cyc = report["cycles"]
        three_node_ok = (
            report["verdict"]["status"] == "FAIL"
            and len(cyc) == 1
            and cyc[0]["size"] == 3
            and set(cyc[0]["modules"])
            == {"audit.audit_a", "audit.audit_b", "audit.audit_c"}
        )
        results["three_node"] = {
            "ok": three_node_ok,
            "verdict": report["verdict"],
            "cycles": cyc,
        }

    # remove one edge → control should pass (not name-based false positive)
    with tempfile.TemporaryDirectory(prefix="hqsb_e01_05_cyc_ctl_") as tmp:
        base = Path(tmp)
        _write_files(
            base,
            {
                "audit_a.py": "import audit.audit_b\n",
                "audit_b.py": "import audit.audit_c\n",
                "audit_c.py": "",  # removed audit_c → audit_a
            },
        )
        report = _scan([("audit", base)])
        control_ok = (
            report["verdict"]["status"] == "PASS"
            and report["verdict"]["cycle_count"] == 0
        )
        results["removal_control"] = {
            "ok": control_ok,
            "verdict": report["verdict"],
        }

    ok = two_node_ok and three_node_ok and control_ok
    return _finish(rec, "PASS" if ok else "FAIL", results=results)


def case_gate_exit_propagation() -> Dict[str, Any]:
    """Step 9 (layer 2): the CLI gate must exit non-zero on bad graphs and 0 on
    the clean production graph."""
    rec = _case(
        "gate_exit_propagation",
        "negative",
        "clean exit 0; bad graph exit non-zero",
    )
    results: Dict[str, Any] = {}

    # clean production graph via CLI
    rc_clean, out_clean, _ = _run_gate_cli(
        [(ns, str(path)) for ns, path in _REAL_SCANS]
    )
    clean_ok = rc_clean == 0 and "status=PASS" in out_clean

    # bad graph (direct core→backend) via CLI
    with tempfile.TemporaryDirectory(prefix="hqsb_e01_05_exit_") as tmp:
        base = Path(tmp)
        _write_files(
            base,
            {
                "core/__init__.py": "",
                "core/leaky.py": "from hqsb.backends.pytorch import PyTorchBackend\n",
                "backends/__init__.py": "",
                "backends/pytorch.py": "class PyTorchBackend:\n    pass\n",
            },
        )
        rc_bad, out_bad, _ = _run_gate_cli([("hqsb", str(base))])
    bad_ok = rc_bad == 1 and "VIOLATION R1" in out_bad and "hqsb.core.leaky" in out_bad

    ok = clean_ok and bad_ok
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        results={
            "clean": {"returncode": rc_clean, "ok": clean_ok, "stdout": out_clean[:500]},
            "bad": {"returncode": rc_bad, "ok": bad_ok, "stdout": out_bad[:500]},
        },
    )


def _tautological_asserts(src: str) -> List[Dict[str, Any]]:
    """Return AST-located ``assert`` statements whose test is trivially True.

    This uses the AST (not a substring search) so that a *comment* mentioning
    ``or True`` — e.g. the historical note in ``test_dependency.py`` — is not
    mistaken for a weakened gate. A tautology is either a constant ``True`` or
    an ``or`` expression with a constant ``True`` operand (the ``assert X or
    True`` risk called out in the protocol).
    """
    import ast as _ast

    found: List[Dict[str, Any]] = []
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return [{"error": "syntax error"}]
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Assert):
            continue
        test = node.test

        def is_taut(expr: _ast.AST) -> bool:
            if isinstance(expr, _ast.Constant):
                return bool(expr.value) is True
            if isinstance(expr, _ast.BoolOp) and isinstance(expr.op, _ast.Or):
                return any(is_taut(v) for v in expr.values)
            return False

        if is_taut(test):
            found.append(
                {
                    "line": getattr(node, "lineno", None),
                    "expr": _ast.unparse(test) if hasattr(_ast, "unparse") else "<expr>",
                }
            )
    return found


def case_no_tautology() -> Dict[str, Any]:
    """Step 9 (layer 1): the gate/test must not rely on tautologies or an
    empty-scan pass. Audits the checker + dependency test source with AST and
    verifies that an empty scan yields ERROR (exit 2), not PASS."""
    rec = _case(
        "no_tautology",
        "negative",
        "no tautological assert; empty scan → ERROR",
    )
    gate_src = (_AUDIT_DIR / "import_dependency_gate.py").read_text(encoding="utf-8")
    dep_test_src = (_REPO_ROOT / "tests/unit/core/test_dependency.py").read_text(
        encoding="utf-8"
    )

    # 1) no tautological assert in either gate or dependency test (AST-based).
    gate_taut = _tautological_asserts(gate_src)
    test_taut = _tautological_asserts(dep_test_src)

    # 2) empty scan must not be PASS.
    empty_report = _scan([])
    empty_ok = (
        empty_report["verdict"]["status"] == "ERROR"
        and empty_report["verdict"]["exit_code"] == 2
    )

    ok = (len(gate_taut) == 0) and (len(test_taut) == 0) and empty_ok
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        gate_tautologies=gate_taut,
        test_tautologies=test_taut,
        empty_scan=empty_report["verdict"],
        gate_rules_version=gate.RULES_VERSION,
    )


# ── Drivers ─────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or gate_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env_info = collect_environment()

    cases: List[Dict[str, Any]] = []
    cases.append(case_baseline_legal_graph())
    cases.append(case_extraction_coverage())
    cases.append(case_legal_registration())
    cases.append(case_counterexample_direct())
    cases.append(case_counterexample_indirect())
    cases.append(case_counterexample_cycles())
    cases.append(case_gate_exit_propagation())
    cases.append(case_no_tautology())

    passed = sum(1 for c in cases if c["status"] == "PASS")
    verdict = {
        "total_cases": len(cases),
        "passed_cases": passed,
        "overall": "PASS" if passed == len(cases) else "FAIL",
    }

    record = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "environment": env_info,
        "verdict": verdict,
        "cases": cases,
    }

    (out_dir / f"e01_05_{run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"e01_05_{run_id}_env.json").write_text(
        json.dumps(env_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Persist per-case raw artifacts.
    for c in cases:
        cid = c["case_id"]
        obs = c.get("observed", {})
        if cid == "baseline_legal_graph":
            _write_json(out_dir / "baseline_report.json", obs.get("report", {}))
        if cid == "extraction_coverage":
            _write_json(
                out_dir / "extraction_coverage.json",
                {k: v for k, v in obs.items() if k != "edges"} | {"edges": obs.get("edges", [])},
            )
        if cid == "legal_registration":
            _write_json(out_dir / "legal_registration_report.json", obs.get("report", {}))
        if cid == "counterexample_direct":
            _write_json(
                out_dir / "counterexample_direct.json",
                {"verdict": obs.get("verdict"), "violations": obs.get("violations")},
            )
        if cid == "counterexample_indirect":
            _write_json(out_dir / "counterexample_indirect.json", obs.get("results", {}))
        if cid == "counterexample_cycles":
            _write_json(out_dir / "counterexample_cycles.json", obs.get("results", {}))
        if cid == "gate_exit_propagation":
            _write_json(out_dir / "gate_exit_propagation.json", obs.get("results", {}))
        if cid == "no_tautology":
            _write_json(out_dir / "no_tautology_audit.json", obs)

    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(f"[{EXPERIMENT_ID}] commit={env_info['git_commit_short']} dirty={env_info['git_dirty']}")
    print(f"[{EXPERIMENT_ID}] cases={passed}/{verdict['total_cases']} pass")
    print(f"[{EXPERIMENT_ID}] verdict={verdict['overall']}")
    print()
    for c in cases:
        mark = "PASS" if c["status"] == "PASS" else "FAIL"
        print(f"  [{mark}] {c['case_id']:<28} ({c['case_kind']})")
    return 0 if verdict["overall"] == "PASS" else 1


def gate_run_id() -> str:
    try:
        from hqsb.core.ids import new_run_id

        return new_run_id()
    except Exception:
        import hashlib

        return f"run_{int(time.time() * 1000)}_{hashlib.sha256(os.urandom(8)).hexdigest()[:8]}"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E01-05 import dependency boundaries.")
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S01/E01-05/raw",
        help="Directory for raw JSON artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
