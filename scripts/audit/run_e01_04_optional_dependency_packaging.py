#!/usr/bin/env python3
"""E01-04 — Optional dependency isolation, install matrix & lightweight core.

Question
--------
Does the HQSB *base* wheel install and run its core (contracts, config,
registry, Dummy backend, result handling) on a clean CPU environment without
pulling torch / triton / CUDA / serving / model-framework dependencies — and
does a missing optional component degrade *only* the corresponding
capability instead of breaking the whole import?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  The base wheel's ``Requires-Dist`` declares only ``pydantic`` + ``PyYAML``;
    a cold ``pip install`` in an isolated venv resolves no forbidden
    distribution; cold-start imports of the core public surface load no
    torch/triton/CUDA/serving module; core functionality (config/schema/
    registry/Dummy C6/C7) works without any optional stack; and requesting a
    missing capability raises an error that names the missing component while
    unrelated features stay usable.
H0  The base metadata pulls a forbidden dependency, a package ``__init__``
    eagerly imports an optional backend, or a missing optional capability
    breaks the whole core.

Design (protocol §6 steps 1–9)
-------------------------------
* static: build the wheel, parse METADATA (Requires-Dist / Provides-Extra /
  wheel tags), compute size + content listing, compare vs source declaration.
* B0: fresh venv, normal dependency resolution, forbidden-dist check.
* cold start from outside the repo with PYTHONPATH cleared: module origins,
  forbidden-import check, actual import graph.
* functionality in B0: config parse/hash, schema round-trip, registry,
  Dummy end-to-end C6/C7, result read.
* missing capability in B0: pytorch backend / models / serving.
* negative controls: (1) inject ``torch`` into base deps → metadata checker
  flags it; (2) inject eager ``import torch`` into ``hqsb/__init__.py`` →
  cold-start import fails and the detector names the offending module.

Pure CPU / packaging only: no GPU correctness, no model weights.

Raw output (under <out>/)
-------------------------
``hqsb-0.1.0-py3-none-any.whl``    the delivered artifact
``wheel_metadata.json``             parsed METADATA (Requires-Dist/Provides-Extra/tags)
``wheel_size_content.json``         compressed/expanded size + file listing + largest
``b0_install.log``                  pip install transaction
``b0_installed_distributions.json`` final installed distribution list
``b0_import_check.json``            cold-start origins / forbidden imports / import graph
``b0_functionality.json``           config/schema/registry/Dummy/result smoke
``b0_missing_capability.json``      requested-missing capability behavior
``counterexample_metadata.json``    negative control #1
``counterexample_import.json``      negative control #2
``e01_04_<run_id>.json``            full record (cases + verdict)
``e01_04_<run_id>_env.json``        frozen environment / git identity
``verdict.json``                    pass criteria + overall verdict

Usage
-----
    python3 scripts/audit/run_e01_04_optional_dependency_packaging.py \
        --output-dir docs/stage_experiments/S01/E01-04/raw
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXPERIMENT_ID = "E01-04"
STAGE = "S01"
PKG_NAME = "hqsb"

#: Forbidden *distributions*: a clean CPU base install must not resolve these
#: (torch/Triton/CUDA-vendor wheels/serving/model frameworks).
FORBIDDEN_DIST_PREFIXES = (
    "torch",
    "triton",
    "nvidia-",
    "cuda-",
    "cudnn",
    "transformers",
    "modelscope",
    "fastapi",
    "uvicorn",
    "httpx",
    "psutil",
)

#: Forbidden distributions for the *serving* combo: the serving gateway must
#: NOT pull model frameworks or hardware stacks, but IS expected to pull
#: fastapi/uvicorn/httpx (so those are excluded from this list).
SERVING_FORBIDDEN_PREFIXES = (
    "torch",
    "triton",
    "nvidia-",
    "cuda-",
    "cudnn",
    "transformers",
    "modelscope",
    "psutil",
)

#: Forbidden *imports*: a cold core import must not load these top-levels.
FORBIDDEN_IMPORTS = (
    "torch",
    "triton",
    "transformers",
    "modelscope",
    "fastapi",
    "uvicorn",
    "httpx",
)

#: Base run dependencies as declared in pyproject.toml (frozen matrix input).
BASE_DEPS_EXPECTED = {"pydantic", "pyyaml"}

#: Public core surface exercised by the cold-start import check.
COLD_IMPORTS = [
    "hqsb",
    "hqsb.core",
    "hqsb.core.contracts",
    "hqsb.core.config",
    "hqsb.core.registry",
    "hqsb.core.schema",
    "hqsb.core.ids",
    "hqsb.core.errors",
    "hqsb.benchmark.engine",
    "hqsb.backends",
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


# ── Subprocess helpers ──────────────────────────────────────────────────


def _run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Run a command and capture stdout/stderr + return code."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _clean_env() -> Dict[str, str]:
    """Environment with PYTHONPATH removed (isolated path injection)."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _venv_python(venv_dir: str) -> str:
    return os.path.join(venv_dir, "bin", "python")


def _build_wheel(src_dir: str, out_dir: str) -> Dict[str, Any]:
    """Build a wheel from ``src_dir`` into ``out_dir`` with isolated build."""
    os.makedirs(out_dir, exist_ok=True)
    res = _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            src_dir,
            "-w",
            out_dir,
            "--no-deps",
        ],
        cwd=src_dir,
        env=_clean_env(),
    )
    wheels = sorted(Path(out_dir).glob("*.whl"))
    return {
        **res,
        "wheel": str(wheels[0]) if wheels else None,
    }


def _read_wheel_metadata(wheel_path: str) -> Dict[str, Any]:
    """Parse the wheel's dist-info METADATA into requires/extra/tags."""
    requires_dist: List[str] = []
    provides_extra: List[str] = []
    tags: List[str] = []
    with zipfile.ZipFile(wheel_path) as zf:
        meta_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if not meta_names:
            return {}
        raw = zf.read(meta_names[0]).decode("utf-8", "replace")
        wheel_names = [n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")]
        wheel_tags = []
        if wheel_names:
            wheel_raw = zf.read(wheel_names[0]).decode("utf-8", "replace")
            msg = email.parser.Parser().parsestr(wheel_raw)
            wheel_tags = msg.get_all("Tag", [])
    msg = email.parser.Parser().parsestr(raw)
    requires_dist = msg.get_all("Requires-Dist", []) or []
    provides_extra = msg.get_all("Provides-Extra", []) or []
    name = msg.get("Name")
    version = msg.get("Version")

    base_deps: List[str] = []
    extra_deps: Dict[str, List[str]] = {}
    for req in requires_dist:
        m = re.search(r'extra\s*==\s*"([^"]+)"', req)
        if m:
            extra_deps.setdefault(m.group(1), []).append(req.split(";")[0].strip())
        else:
            base_deps.append(req.strip())

    return {
        "name": name,
        "version": version,
        "requires_dist": requires_dist,
        "provides_extra": provides_extra,
        "wheel_tags": wheel_tags,
        "base_deps": base_deps,
        "extra_deps": extra_deps,
    }


def _wheel_size_content(wheel_path: str) -> Dict[str, Any]:
    """Compute compressed/expanded sizes and the file listing."""
    compressed = os.path.getsize(wheel_path)
    entries: List[Dict[str, Any]] = []
    expanded = 0
    with zipfile.ZipFile(wheel_path) as zf:
        for info in zf.infolist():
            expanded += info.file_size
            entries.append(
                {"name": info.filename, "compressed": info.compress_size, "size": info.file_size}
            )
    entries_sorted = sorted(entries, key=lambda e: e["size"], reverse=True)
    return {
        "wheel": os.path.basename(wheel_path),
        "compressed_bytes": compressed,
        "expanded_bytes": expanded,
        "file_count": len(entries),
        "largest_files": entries_sorted[:10],
        "files": sorted(e["name"] for e in entries),
    }


def _parse_pyproject_deps() -> Dict[str, Any]:
    """Lightweight extraction of base deps + optional-dependency keys."""
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def extract_array(key: str) -> List[str]:
        m = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]", text, re.S | re.M)
        if not m:
            return []
        return re.findall(r'"([^"]+)"', m.group(1))

    base = extract_array("dependencies")
    extras_match = re.search(
        r"\[project\.optional-dependencies\](.*?)(?:\[tool\.|$)", text, re.S
    )
    extras: Dict[str, List[str]] = {}
    if extras_match:
        block = extras_match.group(1)
        for m in re.finditer(r"^(\w+)\s*=\s*\[(.*?)\]", block, re.S | re.M):
            extras[m.group(1)] = re.findall(r'"([^"]+)"', m.group(2))
    return {"base_deps": base, "extras": extras}


# ── Case record helpers ─────────────────────────────────────────────────


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


# ── Embedded in-venv check script ───────────────────────────────────────
# Runs inside the B0 venv with cwd=/tmp and PYTHONPATH cleared. Emits one
# JSON object on stdout describing origins, forbidden imports, import graph,
# functionality, and missing-capability behavior.


_VENV_CHECK_SCRIPT = r'''
import json, sys, importlib

report = {"origins": {}, "forbidden_imports": [], "import_graph": [],
          "functionality": {}, "missing_capability": {}, "unrelated_ok": False}

# 1) Cold-start import of the public core surface.
for mod in ["hqsb", "hqsb.core", "hqsb.core.contracts", "hqsb.core.config",
            "hqsb.core.registry", "hqsb.core.schema", "hqsb.core.ids",
            "hqsb.core.errors", "hqsb.benchmark.engine", "hqsb.backends"]:
    m = importlib.import_module(mod)
    report["origins"][mod] = getattr(m, "__file__", None)

# 2) Forbidden imports present after the cold start.
present = sorted(
    m for m in sys.modules
    if any(m == f or m.startswith(f + ".") for f in
           ("torch", "triton", "transformers", "modelscope",
            "fastapi", "uvicorn", "httpx"))
)
report["forbidden_imports"] = present

# 3) Actual import graph: hqsb submodules actually loaded.
report["import_graph"] = sorted(m for m in sys.modules if m.startswith("hqsb"))

# 4) Core functionality smoke (no optional stack).
from hqsb.core.config import ConfigLoader, BenchmarkConfig, deep_merge, config_hash  # noqa
from hqsb.core.schema import SchemaVersion, migrate_document
from hqsb.core.ids import new_run_id
from hqsb.core.registry import RegistryHub
from hqsb.backends import DummyBackend, make_dummy_backend
from hqsb.benchmark.engine import BenchmarkEngine
from hqsb.core.contracts import ModelArtifact, WorkloadSpec, BenchmarkResult

func = {}

# config parse + hash
merged = deep_merge({"benchmark": {"batch_size": 1}}, {"benchmark": {"dtype": "float16"}})
func["deep_merge"] = merged["benchmark"]
cfg = ConfigLoader(BenchmarkConfig).load(environ={}, defaults={
    "benchmark": {"model": "fixture/synthetic", "model_source": "modelscope",
                  "backend": "dummy", "dtype": "float16"},
    "workloads": [{"name": "short", "input_tokens": 128, "output_tokens": 32,
                   "seed": 42, "warmup": 1, "repetitions": 3}],
})
func["config_model"] = type(cfg).__name__
func["config_hash_len"] = len(ConfigLoader(BenchmarkConfig).load_resolved(environ={}, defaults={
    "benchmark": {"model": "fixture/synthetic", "model_source": "modelscope",
                  "backend": "dummy", "dtype": "float16"},
    "workloads": [{"name": "short", "input_tokens": 128, "output_tokens": 32,
                   "seed": 42, "warmup": 1, "repetitions": 3}],
}).config_hash)

# schema round-trip
v = SchemaVersion.parse("1.0.0")
func["schema_parse"] = str(v)
func["schema_order"] = SchemaVersion.parse("1.0.1") > SchemaVersion.parse("1.0.0")
migrated = migrate_document({"schema_version": "1.0.0", "x": 1}, SchemaVersion.parse("1.0.1"),
                            {SchemaVersion.parse("1.0.0"): lambda d: {**d, "x": d["x"] + 1}})
func["schema_migrate"] = migrated

# registry register/lookup (lazy factory)
hub = RegistryHub()
hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
factory = hub.backends.get("dummy")
func["registry_names"] = list(hub.backends.names())
func["registry_lazy"] = callable(factory) and not isinstance(factory, DummyBackend)

# Dummy end-to-end C6/C7
backend = make_dummy_backend()
artifact = ModelArtifact(model_id="fixture/dummy", source="local",
                         architecture="DummyForCausalLM", dtype="float16")
workload = WorkloadSpec(name="short", input_tokens=128, output_tokens=32,
                        seed=42, warmup=1, repetitions=3)
result = BenchmarkEngine(backend).run(workload, artifact=artifact)
func["dummy_correctness"] = result.correctness.passed
func["dummy_samples"] = len(result.raw_samples)
func["dummy_run_id"] = result.run_id
func["trace_event_types"] = [e.event_type.value for e in backend.trace_events]

# result read: round-trip a BenchmarkResult through JSON
roundtripped = BenchmarkResult.model_validate_json(json.dumps(result.model_dump(mode="json")))
func["result_roundtrip"] = roundtripped.schema_version

report["functionality"] = func

# 5) Missing optional capability requests.
miss = {}
try:
    from hqsb.backends import PyTorchBackend  # noqa
    miss["pytorch"] = {"imported": True}
except Exception as e:
    miss["pytorch"] = {"imported": False, "error": type(e).__name__, "message": str(e)[:200]}

try:
    import hqsb.models  # noqa
    miss["models"] = {"imported": True}
except Exception as e:
    miss["models"] = {"imported": False, "error": type(e).__name__, "message": str(e)[:200]}

try:
    import hqsb.serving  # noqa
    miss["serving"] = {"imported": True}
except Exception as e:
    miss["serving"] = {"imported": False, "error": type(e).__name__, "message": str(e)[:200]}

report["missing_capability"] = miss

# 6) Unrelated features still usable after the failed capability requests.
hub2 = RegistryHub()
hub2.backends.register("dummy", make_dummy_backend, version="1.0.0")
report["unrelated_ok"] = list(hub2.backends.names()) == ["dummy"]

print(json.dumps(report, ensure_ascii=False, sort_keys=True))
'''


# ── Case executors ──────────────────────────────────────────────────────


def _run_venv_check(venv_dir: str) -> Dict[str, Any]:
    """Execute the embedded check script inside the venv, isolated."""
    script_path = os.path.join(venv_dir, "_e01_04_check.py")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(_VENV_CHECK_SCRIPT)
    res = _run(
        [_venv_python(venv_dir), script_path],
        cwd="/tmp",
        env=_clean_env(),
        timeout=300,
    )
    parsed: Dict[str, Any] = {}
    if res["returncode"] == 0:
        try:
            parsed = json.loads(res["stdout"])
        except json.JSONDecodeError:
            parsed = {"_raw": res["stdout"][:5000]}
    return {**res, "report": parsed}


# ── Cases ───────────────────────────────────────────────────────────────


def case_metadata_declaration(wheel_path: str) -> Dict[str, Any]:
    rec = _case("metadata_declaration", "positive", "base deps == pydantic+PyYAML only")
    meta = _read_wheel_metadata(wheel_path)
    src = _parse_pyproject_deps()
    base_names = {re.split(r"[<>=!~\s]", d)[0].lower() for d in meta.get("base_deps", [])}
    forbidden_base = [d for d in meta.get("base_deps", [])
                      if d.lower().startswith(FORBIDDEN_DIST_PREFIXES)]
    extra_names = set(meta.get("provides_extra", []))
    src_extra_names = set(src.get("extras", {}).keys())
    ok = (
        base_names == BASE_DEPS_EXPECTED
        and not forbidden_base
        and extra_names == src_extra_names
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        wheel=os.path.basename(wheel_path),
        metadata=meta,
        source_declaration=src,
        base_dep_names=sorted(base_names),
        forbidden_base_deps=forbidden_base,
        extra_names=sorted(extra_names),
        source_extra_names=sorted(src_extra_names),
    )


def case_package_size_content(wheel_path: str) -> Dict[str, Any]:
    rec = _case("package_size_content", "positive", "no weights/caches/logs; small")
    stats = _wheel_size_content(wheel_path)
    suspicious = [f for f in stats["files"] if re.search(
        r"\.(safetensors|bin|pt|pth|ckpt|onnx|whl|tar|gz)$", f)
        or re.search(r"(__pycache__|\.pyc|\.log|checkpoint|cache)", f)]
    ok = not suspicious and stats["compressed_bytes"] < 5_000_000
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        size_stats=stats,
        suspicious_files=suspicious,
    )


def _make_venv() -> str:
    venv_dir = tempfile.mkdtemp(prefix="hqsb_e01_04_")
    _run([sys.executable, "-m", "venv", venv_dir], cwd="/tmp", env=_clean_env())
    return venv_dir


def _install_wheel(venv_dir: str, wheel_path: str) -> Dict[str, Any]:
    return _run(
        [_venv_python(venv_dir), "-m", "pip", "install", wheel_path],
        cwd="/tmp",
        env=_clean_env(),
        timeout=600,
    )


def _installed_dists(venv_dir: str) -> Dict[str, Any]:
    res = _run(
        [_venv_python(venv_dir), "-m", "pip", "list", "--format=json"],
        cwd="/tmp",
        env=_clean_env(),
    )
    dists: List[Dict[str, str]] = []
    if res["returncode"] == 0:
        try:
            dists = json.loads(res["stdout"])
        except json.JSONDecodeError:
            pass
    names = [d["name"] for d in dists]
    forbidden = sorted(
        n for n in names
        if any(n.lower().startswith(p) for p in FORBIDDEN_DIST_PREFIXES)
    )
    return {"returncode": res["returncode"], "distributions": dists,
            "names": names, "forbidden": forbidden, "install_log": res}


def case_b0_install(wheel_path: str, venv_dir: str) -> Dict[str, Any]:
    rec = _case("b0_install", "positive", "no forbidden dist resolved")
    install = _install_wheel(venv_dir, wheel_path)
    dists = _installed_dists(venv_dir)
    forbidden = dists["forbidden"]
    ok = install["returncode"] == 0 and not forbidden
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        venv_dir=venv_dir,
        install_returncode=install["returncode"],
        install_log=install,
        installed_distributions=dists["distributions"],
        installed_names=dists["names"],
        forbidden_distributions=forbidden,
    )


def case_b0_cold_import(check: Dict[str, Any]) -> Dict[str, Any]:
    rec = _case("b0_cold_import", "positive", "wheel origin + no forbidden import")
    report = check.get("report", {})
    origins = report.get("origins", {})
    forbidden = report.get("forbidden_imports", [])
    graph = report.get("import_graph", [])
    site = "site-packages"
    all_wheel = all(
        (v or "").replace("/", os.sep).__contains__(site) for v in origins.values()
    )
    ok = all_wheel and not forbidden
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        origins=origins,
        forbidden_imports=forbidden,
        import_graph=graph,
        all_from_wheel=all_wheel,
        check_stdout_tail=check.get("stdout", "")[:2000],
    )


def case_b0_functionality(check: Dict[str, Any]) -> Dict[str, Any]:
    rec = _case("b0_functionality", "positive", "config/schema/registry/Dummy/result OK")
    func = check.get("report", {}).get("functionality", {})
    ok = (
        func.get("config_model") == "BenchmarkConfig"
        and func.get("config_hash_len") == 64
        and func.get("schema_order") is True
        and func.get("registry_lazy") is True
        and func.get("dummy_correctness") is True
        and func.get("dummy_samples") == 3
        and func.get("result_roundtrip") == "1.0.0"
    )
    return _finish(rec, "PASS" if ok else "FAIL", functionality=func)


def case_b0_missing_capability(check: Dict[str, Any]) -> Dict[str, Any]:
    rec = _case("b0_missing_capability", "negative", "errors name missing component; unrelated OK")
    miss = check.get("report", {}).get("missing_capability", {})
    unrelated = check.get("report", {}).get("unrelated_ok", False)
    pytorch = miss.get("pytorch", {})
    models = miss.get("models", {})
    serving = miss.get("serving", {})

    pytorch_ok = (
        not pytorch.get("imported")
        and pytorch.get("error") == "ModuleNotFoundError"
        and "torch" in pytorch.get("message", "").lower()
    )
    models_ok = not models.get("imported") and "torch" in models.get("message", "").lower()
    serving_ok = (
        not serving.get("imported")
        and "hqsb.serving" in serving.get("message", "")
    )
    ok = pytorch_ok and models_ok and serving_ok and unrelated
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        missing_capability=miss,
        unrelated_still_works=unrelated,
        pytorch_names_missing_component=pytorch_ok,
        models_names_missing_component=models_ok,
        serving_names_missing_component=serving_ok,
    )


def _install_wheel_extras(venv_dir: str, wheel_path: str, extras: str) -> Dict[str, Any]:
    """Install ``<wheel>[extras]`` in an isolated venv."""
    return _run(
        [_venv_python(venv_dir), "-m", "pip", "install", f"{wheel_path}[{extras}]"],
        cwd="/tmp",
        env=_clean_env(),
        timeout=600,
    )


def case_serving_combo(wheel_path: str) -> Dict[str, Any]:
    """Matrix row S: base + serving extra pulls gateway only, no model stack."""
    rec = _case("serving_combo", "positive", "serving gateway, no torch/model framework")
    venv_dir = _make_venv()
    install = _install_wheel_extras(venv_dir, wheel_path, "serving")
    dists = _installed_dists(venv_dir)
    names = {d["name"] for d in dists["distributions"]}
    gateway = {"fastapi", "uvicorn", "httpx"}
    gateway_present = gateway <= names
    forbidden = sorted(
        n for n in names
        if any(n.lower().startswith(p) for p in SERVING_FORBIDDEN_PREFIXES)
    )
    check = _run_venv_check(venv_dir)
    core_forbidden = check.get("report", {}).get("forbidden_imports", [])
    ok = (
        install["returncode"] == 0
        and gateway_present
        and not forbidden
        and not core_forbidden
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        venv_dir=venv_dir,
        install_returncode=install["returncode"],
        gateway_present=sorted(gateway & names),
        gateway_expected=sorted(gateway),
        forbidden_distributions=forbidden,
        installed_names=sorted(names),
        core_forbidden_imports=core_forbidden,
    )


def _inject_and_rebuild(tmp_base: str, edit: str, wheel_out: str, tag: str = "") -> Dict[str, Any]:
    """Copy the source tree into a throwaway dir, apply ``edit``, rebuild."""
    src = str(_REPO_ROOT)
    copy = os.path.join(tmp_base, f"src_copy_{tag}" if tag else "src_copy")
    shutil.copytree(
        src,
        copy,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", "build", "dist",
            "hqsb.egg-info", "docs", "reports", ".pytest_cache",
            "tests", "benchmarks", "ops", "third_party", ".github",
        ),
    )
    edit(copy)
    return _build_wheel(copy, wheel_out)


def _check_metadata_forbidden(meta: Dict[str, Any]) -> List[str]:
    return [d for d in meta.get("base_deps", [])
            if d.lower().startswith(FORBIDDEN_DIST_PREFIXES)]


def case_counterexample_metadata(tmp_base: str) -> Dict[str, Any]:
    """Negative control #1: add torch to base deps; checker must flag it."""
    rec = _case("counterexample_metadata", "negative_control",
                "checker flags torch in base deps")
    wheel_out = os.path.join(tmp_base, "ce_wheel_meta")
    res = _inject_and_rebuild(
        tmp_base,
        lambda d: _append_base_dep(d),
        wheel_out,
        tag="meta",
    )
    wheels = sorted(Path(wheel_out).glob("*.whl"))
    if not wheels:
        return _finish(rec, "FAIL", build=res)
    meta = _read_wheel_metadata(str(wheels[0]))
    flagged = _check_metadata_forbidden(meta)
    ok = bool(flagged) and any("torch" in f for f in flagged)
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        injected_base_deps=meta.get("base_deps", []),
        checker_flagged=flagged,
        wheel=os.path.basename(str(wheels[0])),
    )


def _append_base_dep(copy_dir: str) -> None:
    """Append ``torch>=2.0`` to the ``[project] dependencies`` array."""
    py = os.path.join(copy_dir, "pyproject.toml")
    text = open(py, encoding="utf-8").read()
    marker = 'dependencies = [\n    "pydantic>=2.0",\n    "PyYAML>=5.4",\n]'
    if marker in text:
        text = text.replace(
            marker,
            'dependencies = [\n    "pydantic>=2.0",\n    "PyYAML>=5.4",\n    "torch>=2.0",\n]',
        )
        open(py, "w", encoding="utf-8").write(text)


def case_counterexample_import(tmp_base: str, venv_dir: str) -> Dict[str, Any]:
    """Negative control #2: eager ``import torch`` in ``hqsb/__init__.py``.

    The cold-start import must fail with a ModuleNotFoundError naming torch,
    proving the detector would catch an eager optional-backend import.
    """
    rec = _case("counterexample_import", "negative_control",
                "eager optional import breaks cold start and is detected")
    wheel_out = os.path.join(tmp_base, "ce_wheel_import")
    res = _inject_and_rebuild(
        tmp_base,
        lambda d: _inject_eager_torch(d),
        wheel_out,
        tag="import",
    )
    wheels = sorted(Path(wheel_out).glob("*.whl"))
    if not wheels:
        return _finish(rec, "FAIL", build=res)

    ce_venv = _make_venv()
    _install_wheel(ce_venv, str(wheels[0]))
    probe = os.path.join(ce_venv, "_probe.py")
    open(probe, "w", encoding="utf-8").write(
        "try:\n    import hqsb\n    print('IMPORTED')\n"
        "except Exception as e:\n    print(type(e).__name__ + ': ' + str(e))\n"
    )
    probe_res = _run([_venv_python(ce_venv), probe], cwd="/tmp", env=_clean_env())
    out = probe_res["stdout"]
    names_torch = "torch" in out.lower()
    ok = "IMPORTED" not in out and names_torch
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        probe_stdout=out,
        names_missing_component=names_torch,
        wheel=os.path.basename(str(wheels[0])),
        venv_dir=venv_dir,
    )


def _inject_eager_torch(copy_dir: str) -> None:
    init = os.path.join(copy_dir, "hqsb", "__init__.py")
    text = open(init, encoding="utf-8").read()
    if "import torch" not in text:
        text = "import torch  # E01-04 counterexample injection\n" + text
        open(init, "w", encoding="utf-8").write(text)


# ── Drivers ─────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or _make_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env_info = collect_environment()
    tmp_base = tempfile.mkdtemp(prefix="hqsb_e01_04_tmp_")

    # Step A: build + static metadata review.
    build_out = os.path.join(tmp_base, "wheel")
    build_res = _build_wheel(str(_REPO_ROOT), build_out)
    wheel_path = build_res.get("wheel")
    if not wheel_path:
        print(f"[{EXPERIMENT_ID}] wheel build failed", file=sys.stderr)
        print(build_res.get("stderr", ""), file=sys.stderr)
        return 2

    # Persist the wheel into the raw dir.
    wheel_name = os.path.basename(wheel_path)
    persisted_wheel = str(out_dir / wheel_name)
    shutil.copyfile(wheel_path, persisted_wheel)

    cases: List[Dict[str, Any]] = []
    cases.append(case_metadata_declaration(persisted_wheel))
    cases.append(case_package_size_content(persisted_wheel))

    # Step C–E: B0 environment.
    b0_venv = _make_venv()
    cases.append(case_b0_install(persisted_wheel, b0_venv))
    check = _run_venv_check(b0_venv)
    cases.append(case_b0_cold_import(check))
    cases.append(case_b0_functionality(check))
    cases.append(case_b0_missing_capability(check))

    # Step G: serving combo (matrix row S).
    cases.append(case_serving_combo(persisted_wheel))

    # Step F: negative controls.
    cases.append(case_counterexample_metadata(tmp_base))
    cases.append(case_counterexample_import(tmp_base, b0_venv))

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

    # Persist raw artifacts.
    (out_dir / f"e01_04_{run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"e01_04_{run_id}_env.json").write_text(
        json.dumps(env_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for c in cases:
        if c["case_id"] == "metadata_declaration":
            _write_json(out_dir / "wheel_metadata.json", c["observed"].get("metadata", {}))
        if c["case_id"] == "package_size_content":
            _write_json(out_dir / "wheel_size_content.json", c["observed"].get("size_stats", {}))
        if c["case_id"] == "b0_install":
            _write_json(
                out_dir / "b0_installed_distributions.json",
                {
                    "venv_dir": c["observed"].get("venv_dir"),
                    "forbidden": c["observed"].get("forbidden_distributions"),
                    "distributions": c["observed"].get("installed_distributions"),
                },
            )
            (out_dir / "b0_install.log").write_text(
                c["observed"].get("install_log", {}).get("stdout", "")
                + "\n--- stderr ---\n"
                + c["observed"].get("install_log", {}).get("stderr", ""),
                encoding="utf-8",
            )
        if c["case_id"] == "b0_cold_import":
            _write_json(
                out_dir / "b0_import_check.json",
                c["observed"],
            )
        if c["case_id"] == "b0_functionality":
            _write_json(out_dir / "b0_functionality.json", c["observed"])
        if c["case_id"] == "b0_missing_capability":
            _write_json(out_dir / "b0_missing_capability.json", c["observed"])
        if c["case_id"] == "serving_combo":
            _write_json(out_dir / "serving_combo.json", c["observed"])
        if c["case_id"] == "counterexample_metadata":
            _write_json(out_dir / "counterexample_metadata.json", c["observed"])
        if c["case_id"] == "counterexample_import":
            _write_json(out_dir / "counterexample_import.json", c["observed"])

    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Console report.
    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(f"[{EXPERIMENT_ID}] commit={env_info['git_commit_short']} dirty={env_info['git_dirty']}")
    print(f"[{EXPERIMENT_ID}] cases={passed}/{verdict['total_cases']} pass")
    print(f"[{EXPERIMENT_ID}] verdict={verdict['overall']}")
    print()
    for c in cases:
        mark = "PASS" if c["status"] == "PASS" else "FAIL"
        print(f"  [{mark}] {c['case_id']:<30} ({c['case_kind']})")
    return 0 if verdict["overall"] == "PASS" else 1


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_run_id() -> str:
    # Reuse hqsb's run id if importable from the source tree (best effort).
    try:
        from hqsb.core.ids import new_run_id

        return new_run_id()
    except Exception:
        return f"run_{int(time.time() * 1000)}_{hashlib.sha256(os.urandom(8)).hexdigest()[:8]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E01-04 optional dependency packaging.")
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S01/E01-04/raw",
        help="Directory for raw JSON/JSONL/wheel artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
