#!/usr/bin/env python3
"""E00-05 — Qwen3-1.7B tiny-workload model-core greedy smoke.

Question
--------
In a legal ModelArtifact, with frozen token IDs and the recorded
environment, is the minimal Qwen3-1.7B model-core chain (ModelArtifact
validation -> load -> prefill/forward -> greedy decode -> result record)
usable and reproducible across processes, and are bad model paths or
tampered hashes rejected before any weight is loaded?

Pre-registered design (per docs/stage_experiments/details/S00/E00-05_*.md)
-------------------------------------------------------------------------
* tiny workload (unique source of truth ``configs/benchmarks/jetson_qwen3_fp16.yaml``):
  ISL=32, OSL=16, batch=1, seed text = the registered fixed-token seed,
  stop rule = produce exactly OSL tokens (EOS inside the window is recorded,
  never stops), sampling off (greedy argmax).
* Model identity = Qwen3-1.7B artifact declared by
  ``docs/benchmark/model_sha256_manifest.txt`` (manifest sha256
  f2d545ef…) -> ModelArtifact identity hash e7af1c75… (E00-02/E00-03).
  Verified against the real snapshot with the documented allow-list for the
  convenience copy ``model_sha256_manifest.txt``.
* Frozen input: the golden 32 token IDs already produced by the registered
  fixed-token workload; the run never re-tokenizes to build the input.
  Tokenizer identity is still *proven* by re-tokenizing the seed text and
  asserting an exact match with the frozen IDs.
* Repetition: 3 independent OS processes run the same frozen configuration;
  cross-process agreement requires identical output token hash + first-logits
  hash + token count.
* Negative path: one fault per run against the *real* loader, each in its
  own subprocess: non-existent model path, manifest hash tamper, tokenizer
  file hash tamper, strict-extra file. All must exit non-zero before weights
  load, never download, and never silently use another model.

Raw output (docs/stage_experiments/S00/E00-05/raw/)
----------------------------------------------------
``identity_freeze.json``, ``frozen_input_ids.json``, ``generation_raw.jsonl``
(one line per OS process), ``negative_cases.jsonl``, ``artifact_gate.json``,
``environment.json``, ``run_records/*.log``, ``command.txt``,
``file_manifest.json`` and the aggregated run record ``e00_05_run_<id>.json``.

Usage
-----
    python3 scripts/audit/run_e00_05_qwen_tiny_smoke.py \
        --output-dir docs/stage_experiments/S00/E00-05/raw \
        [--runs 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.core.fingerprint import (  # noqa: E402
    collect_commit,
    collect_device_basic,
    collect_os,
    collect_packages,
    collect_power,
    collect_python,
    collect_volatile,
)
from hqsb.core.ids import new_run_id  # noqa: E402
from hqsb.hardware.probe import cuda_device_probe  # noqa: E402

EXPERIMENT_ID = "E00-05"
STAGE = "S00"

# Registered workload ``tiny`` (configs/benchmarks/jetson_qwen3_fp16.yaml).
WORKLOAD = {
    "name": "tiny",
    "isl": 32,
    "osl": 16,
    "batch": 1,
    "sampling": "greedy-argmax",
    "stop_rule": "exact_osl_no_early_stop",
}

# Same seed text as hqsb/benchmark/workload.py (fixed-token seed).
_SEED_TEXT = (
    "CUDA GPU inference optimization "
    "memory bandwidth kernel latency "
    "transformer attention cache "
    "performance benchmark. "
)

_DEFAULT_MODEL = "~/models/hqsb/Qwen3-1.7B"
_DEFAULT_MANIFEST = "docs/benchmark/model_sha256_manifest.txt"
_ALLOW_EXTRA = ("model_sha256_manifest.txt",)

# Repo manifest digest (E00-02 / E00-03 recorded value).
_REPO_MANIFEST_SHA256 = "f2d545ef73b16222d398c54fcf525f99132cc92a7be6b003d08fc6d3340a94ea"
# ModelArtifact.identity_hash() of the 14-file artifact (E00-02 record).
_ARTIFACT_IDENTITY_SHA256 = "e7af1c7599fa3d6339cfd205602fdfd160e6df4c52d654e8e7e843b8f735dcc4"


# ── helpers ────────────────────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=str(_REPO_ROOT)
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


def _log_lines(proc: subprocess.CompletedProcess) -> List[str]:
    lines = [f"exit_code = {proc.returncode}", "--- stdout ---"]
    lines.append((proc.stdout or "").rstrip())
    lines.append("--- stderr ---")
    lines.append((proc.stderr or "").rstrip())
    return lines


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_token_ids(ids: List[int]) -> str:
    return sha256_hex_bytes(json.dumps(ids).encode("utf-8"))


# ── environment ────────────────────────────────────────────────────────────


def collect_environment_block() -> Dict[str, Any]:
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
        "started_at_utc": _now_utc(),
    }


def collect_environment_json() -> Dict[str, Any]:
    device = collect_device_basic(cuda_device_probe())
    power = collect_power()
    return {
        "os": collect_os().model_dump(mode="json"),
        "device": device.model_dump(mode="json"),
        "python": collect_python().model_dump(mode="json"),
        "packages": collect_packages(),
        "power": power.model_dump(mode="json"),
        "volatile": collect_volatile().model_dump(mode="json"),
        "commit": collect_commit(str(_REPO_ROOT)).model_dump(mode="json"),
    }


# ── identity freeze (CPU only) ─────────────────────────────────────────────


def compute_identity_freeze(
    model_path: str, manifest_path: str, input_ids: List[int]
) -> Dict[str, Any]:
    from hqsb.models.manifest import compute_sha256, load_manifest

    entries = load_manifest(manifest_path)
    files = {e.normalized_path: e.sha256 for e in entries}
    manifest_sha = compute_sha256(manifest_path)

    # Construction parameters reproduce the E00-02 recorded artifact exactly:
    # revision=None, source=modelscope, architecture=Qwen3ForCausalLM,
    # dtype=float16, file_hashes = the 14 repo-manifest entries.
    from hqsb.core.contracts.model import ModelArtifact

    artifact = ModelArtifact(
        model_id="Qwen/Qwen3-1.7B",
        source="modelscope",
        architecture="Qwen3ForCausalLM",
        dtype="float16",
        file_hashes=files,
    )

    def cfg_hash(name: str) -> str:
        return hashlib.sha256(
            Path(model_path, name).read_bytes()
        ).hexdigest()

    tokenizer_files = {
        "tokenizer.json": files.get("tokenizer.json", ""),
        "tokenizer_config.json": files.get("tokenizer_config.json", ""),
    }
    tokenizer_composite = sha256_hex_bytes(
        "|".join(
            f"{k}:{v}" for k, v in sorted(tokenizer_files.items())
        ).encode("utf-8")
    )

    return {
        "experiment_id": EXPERIMENT_ID,
        "model_id": "Qwen/Qwen3-1.7B",
        "model_source": "modelscope",
        "revision": None,
        "revision_note": "E00-02/E00-03 recorded identity uses revision=None",
        "model_path": os.path.abspath(os.path.expanduser(model_path)),
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_sha256_expected": _REPO_MANIFEST_SHA256,
        "manifest_declares_n_files": len(files),
        "artifact_identity_payload": {
            "model_id": "Qwen/Qwen3-1.7B",
            "source": "modelscope",
            "revision": None,
            "architecture": "Qwen3ForCausalLM",
            "dtype": "float16",
        },
        "artifact_identity_hash": artifact.identity_hash(),
        "artifact_identity_hash_expected": _ARTIFACT_IDENTITY_SHA256,
        "config_hash": cfg_hash("config.json"),
        "generation_config_hash": cfg_hash("generation_config.json"),
        "tokenizer": {
            "tokenizer_json_sha256": files.get("tokenizer.json", ""),
            "tokenizer_config_json_sha256": files.get("tokenizer_config.json", ""),
            "tokenizer_sha256": tokenizer_composite,
            "include_bos": False,
            "include_eos": False,
            "pad_rule": "no_pad",
        },
        "files_declared": {k: v for k, v in sorted(files.items())},
    }


def load_frozen_input(workload_isl: int) -> Dict[str, Any]:
    """Frozen input = the 32 token IDs of the registered fixed-token seed."""
    golden_path = _REPO_ROOT / "benchmarks" / "workloads" / "golden" / "isl32_osl32.json"
    if not golden_path.is_file():
        raise FileNotFoundError(f"golden workload missing: {golden_path}")
    import json as _json

    golden = _json.loads(golden_path.read_text(encoding="utf-8"))
    ids = list(golden["input_token_ids"])
    if len(ids) != workload_isl:
        raise RuntimeError(
            f"golden isl32 has {len(ids)} input tokens; expected {workload_isl}"
        )
    return {
        "source": "benchmarks/workloads/golden/isl32_osl32.json#input_token_ids",
        "shape": [1, len(ids)],
        "dtype": "int64",
        "seed_text": _SEED_TEXT,
        "include_bos": False,
        "include_eos": False,
        "input_ids": ids,
        "token_ids_sha256": _hash_token_ids(ids),
    }


# ── generation harness (one subprocess per run) ─────────────────────────────


def _child_harness_source(no_verify: bool = False) -> str:
    """Python source for one frozen generation run in a fresh process.

    Args are positional on argv: 1=model_path, 2=input_ids_json, 3=run_tag,
    4=manifest_path (only when not ``no_verify``).
    """
    return f'''
import json, os, sys
sys.path.insert(0, {str(_REPO_ROOT)!r})
import torch
from hqsb.benchmark._e00_05_smoke import run_once_entry

model_path = sys.argv[1]
ids_json = sys.argv[2]
run_tag = sys.argv[3]
manifest_path = {"sys.argv[4]" if not no_verify else "None"}
ids = json.loads(ids_json)
rec = run_once_entry(
    model_path=model_path,
    manifest_path=manifest_path,
    allow_extra={list(_ALLOW_EXTRA)!r},
    frozen_input_ids=ids,
    requested_output_tokens={WORKLOAD["osl"]},
    seed_text={_SEED_TEXT!r},
    requested_dtype="float16",
    requested_backend="eager",
    top_k=32,
    run_tag=run_tag,
)
# extra_environment
rec["extra_environment"] = {{
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    "cuda_device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "torch_cuda_version": torch.version.cuda or "",
    "torch_version": torch.__version__,
}}
print("HQSB_E00_05_EVIDENCE_START")
print(json.dumps(rec, ensure_ascii=False))
print("HQSB_E00_05_EVIDENCE_END")
'''


def run_generation_subprocess(
    model_path: str, manifest_path: str, input_ids: List[int], run_tag: str, out_dir: Path
) -> Tuple[Dict[str, Any], str, str, int]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-c", _child_harness_source(no_verify=False)]
    cmd += [os.path.expanduser(model_path), json.dumps(input_ids), run_tag,
            os.path.expanduser(manifest_path)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), env=env, timeout=2400,
    )
    (out_dir / "run_records" / f"{run_tag}.log").write_text(
        "# " + " ".join(cmd) + "\n" + "\n".join(_log_lines(proc)),
        encoding="utf-8",
    )
    ev_start = (proc.stdout or "").find("HQSB_E00_05_EVIDENCE_START")
    ev_end = (proc.stdout or "").find("HQSB_E00_05_EVIDENCE_END")
    evidence: Dict[str, Any] = {"parsed": False}
    if ev_start >= 0 and ev_end > ev_start:
        try:
            evidence = json.loads(proc.stdout[ev_start + len("HQSB_E00_05_EVIDENCE_START") : ev_end].strip())
            evidence["parsed"] = True
        except Exception as exc:  # pragma: no cover - parse guard
            evidence = {"parsed": False, "parse_error": f"{type(exc).__name__}: {exc}"}
    evidence["exit_code"] = int(proc.returncode)
    evidence["run_tag"] = run_tag
    evidence["stdout_path"] = f"run_records/{run_tag}.log"
    evidence["no_silent_fallback"] = "Consolidated" in proc.stdout or "fully_on_cuda" in proc.stdout
    return evidence, proc.stdout or "", proc.stderr or "", int(proc.returncode)


# ── artifact gate (in-process; CPU-only until verification) ────────────────


def run_artifact_gate(model_path: str, manifest_path: str) -> Dict[str, Any]:
    import time as _time

    gate_records = []
    start = _time.perf_counter()
    passed = False
    error = ""
    try:
        from hqsb.models.manifest import verify_or_raise

        res = verify_or_raise(
            os.path.expanduser(model_path),
            manifest_path,
            strict_extra=True,
            allow_extra=list(_ALLOW_EXTRA),
        )
        passed = True
        gate_records.append(res.as_dict())
    except Exception as exc:  # pragma: no cover
        error = f"{type(exc).__name__}: {exc}"
    seconds = round(_time.perf_counter() - start, 3)
    return {
        "gate": "hqsb.models.manifest.verify_or_raise",
        "loader": "hqsb.models.loader.load_qwen3(verify_manifest=...)",
        "manifest_path": manifest_path,
        "verify_passed": passed,
        "full_snapshot_verify_seconds": seconds,
        "verification_detail": gate_records[0] if gate_records else None,
        "error": error,
    }


# ── negative cases ──────────────────────────────────────────────────────────


def run_negative_cases(
    model_path: str, manifest_path: str, input_ids: List[int], out_dir: Path
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # N01: model root does not exist.
    bad_root = str(_REPO_ROOT / "docs" / "stage_experiments" / "S00" / "E00-05" / "does_not_exist")
    n01_code = f"""
import sys; sys.path.insert(0, {str(_REPO_ROOT)!r})
import torch
from hqsb.models.loader import load_qwen3
try:
    load_qwen3({bad_root!r}, dtype=torch.float16, attention_backend="eager",
               verify_manifest={os.path.abspath(manifest_path)!r},
               strict_extra=True, allow_extra={list(_ALLOW_EXTRA)!r})
except Exception as e:
    print("ERRTYPE:", type(e).__name__)
    print("ERRMSG:", str(e).replace("\\n", " | "))
    raise SystemExit(getattr(e, "exit_code", 1) if type(e).__name__ != "FileNotFoundError" else 1)
print("UNEXPECTED_LOAD_OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", n01_code], capture_output=True, text=True,
        cwd=str(_REPO_ROOT), env=env, timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    cases.append({
        "case_id": "N01_model_path_not_exist",
        "fault": "model root points to a non-existent directory",
        "exit_code": int(proc.returncode),
        "non_zero_exit": proc.returncode != 0,
        "loader_not_called": ("UNEXPECTED_LOAD_OK" not in (proc.stdout or "")),
        "error_points_at_path": bad_root in out,
        "rejected_reason": "FileNotFoundError / directory check before load"
        if "UNEXPECTED_LOAD_OK" not in (proc.stdout or "") else "none",
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    })

    # N02: manifest copied + first weight-shard hash tampered (model files
    # untouched). The repo manifest declares sharded weights
    # model-0000N-of-00002.safetensors; tampering the first shard's digest is
    # the "weight file hash wrong" fault that must be rejected before load.
    manifest_text = Path(manifest_path).read_text(encoding="utf-8")
    tampered_lines = []
    for line in manifest_text.splitlines():
        if line.strip().startswith("#") or not line.strip():
            tampered_lines.append(line)
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1].strip("./") == "model-00001-of-00002.safetensors":
            fake = "0" * 64
            tampered_lines.append(f"{fake}  {parts[1]}")
        else:
            tampered_lines.append(line)
    tampered_manifest = out_dir / "neg_N02_tampered_manifest.txt"
    tampered_manifest.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")

    n02_code = f"""
import sys, json; sys.path.insert(0, {str(_REPO_ROOT)!r})
import torch
from hqsb.models.loader import load_qwen3
try:
    load_qwen3({os.path.expanduser(model_path)!r}, dtype=torch.float16,
               attention_backend="eager",
               verify_manifest={str(tampered_manifest)!r},
               strict_extra=True, allow_extra={list(_ALLOW_EXTRA)!r})
except Exception as e:
    print("ERRTYPE:", type(e).__name__)
    print("ERRMSG:", str(e).replace("\\n", " | "))
    print("HAS_EXIT_CODE:", hasattr(e, "exit_code"))
    print("EXIT_CODE:", getattr(e, "exit_code", "none"))
    if hasattr(e, "details") and isinstance(e.details, dict):
        print("DETAILS:", json.dumps(e.details, ensure_ascii=False))
    raise SystemExit(8)
print("UNEXPECTED_LOAD_OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", n02_code], capture_output=True, text=True,
        cwd=str(_REPO_ROOT), env=env, timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    cases.append({
        "case_id": "N02_manifest_hash_tampered",
        "fault": "manifest model-00001-of-00002.safetensors hash replaced "
                 "with all-zeros; model files untouched",
        "exit_code": int(proc.returncode),
        "non_zero_exit": proc.returncode != 0,
        "loader_not_called": "UNEXPECTED_LOAD_OK" not in (proc.stdout or ""),
        "reason_mentions_hash_mismatch": "hash_mismatch" in out,
        "reason_mentions_first_bad_file": (
            "first_bad_file" in out or "model-00001-of-00002.safetensors" in out
        ),
        "rejected_before_weights": (
            "ArtifactError" in out or "EXIT_CODE: 8" in out
        ),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    })

    # N03: tokenizer.json hash tampered (identity mismatch on the real loader gate).
    tampered_lines = []
    for line in manifest_text.splitlines():
        if line.strip().startswith("#") or not line.strip():
            tampered_lines.append(line)
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1].strip("./") == "tokenizer.json":
            tampered_lines.append(f"{'a'*64}  {parts[1]}")
        else:
            tampered_lines.append(line)
    tampered_manifest_tok = out_dir / "neg_N03_tampered_tokenizer_manifest.txt"
    tampered_manifest_tok.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")
    n03_code = n02_code.replace(str(tampered_manifest), str(tampered_manifest_tok))
    proc = subprocess.run(
        [sys.executable, "-c", n03_code], capture_output=True, text=True,
        cwd=str(_REPO_ROOT), env=env, timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    cases.append({
        "case_id": "N03_tokenizer_hash_tampered",
        "fault": "manifest tokenizer.json hash replaced; model files untouched",
        "exit_code": int(proc.returncode),
        "non_zero_exit": proc.returncode != 0,
        "loader_not_called": "UNEXPECTED_LOAD_OK" not in (proc.stdout or ""),
        "reason_mentions_hash_mismatch": "hash_mismatch" in out,
        "reason_mentions_first_bad_file": "first_bad_file" in out or "tokenizer.json" in out,
        "rejected_before_weights": "ArtifactError" in out or "EXIT_CODE: 8" in out,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    })

    # N04: extra undeclared file under a *copy* model dir (strict extra gate).
    import shutil, tempfile

    extra_root = Path(tempfile.mkdtemp(prefix="hqsb_e0005_extra_"))
    (extra_root / "config.json").write_text("{}", encoding="utf-8")
    (extra_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (extra_root / "evil_weight.safetensors").write_text("tampered", encoding="utf-8")
    mini_manifest = out_dir / "neg_N04_extra_mini_manifest.txt"
    mini_manifest.write_text(
        "\n".join([
            hashlib.sha256(b"{}").hexdigest() + "  ./config.json",
            hashlib.sha256(b"{}").hexdigest() + "  ./tokenizer.json",
        ]) + "\n",
        encoding="utf-8",
    )
    n04_code = f"""
import sys, json; sys.path.insert(0, {str(_REPO_ROOT)!r})
import torch
from hqsb.models.loader import load_qwen3
try:
    load_qwen3({str(extra_root)!r}, dtype=torch.float16,
               attention_backend="eager",
               verify_manifest={str(mini_manifest)!r},
               strict_extra=True)
except Exception as e:
    print("ERRTYPE:", type(e).__name__)
    print("ERRMSG:", str(e).replace("\\n", " | "))
    print("EXIT_CODE:", getattr(e, "exit_code", "none"))
    if hasattr(e, "details") and isinstance(e.details, dict):
        print("DETAILS:", json.dumps(e.details, ensure_ascii=False))
    raise SystemExit(8)
print("UNEXPECTED_LOAD_OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", n04_code], capture_output=True, text=True,
        cwd=str(_REPO_ROOT), env=env, timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    cases.append({
        "case_id": "N04_extra_undeclared_file",
        "fault": "undeclared extra weight file in model dir under strict_extra",
        "exit_code": int(proc.returncode),
        "non_zero_exit": proc.returncode != 0,
        "loader_not_called": "UNEXPECTED_LOAD_OK" not in (proc.stdout or ""),
        "reason_mentions_extra": "extra_file" in out,
        "reason_mentions_first_bad_file": "first_bad_file" in out or "evil_weight.safetensors" in out,
        "rejected_before_weights": "ArtifactError" in out or "EXIT_CODE: 8" in out,
        "stdout_tail": (proc.stdout or "")[-1500:],
        "stderr_tail": (proc.stderr or "")[-1500:],
    })
    shutil.rmtree(extra_root, ignore_errors=True)
    return cases


# ── drivers ────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E00-05 Qwen tiny model-core greedy smoke.")
    parser.add_argument(
        "--output-dir", default="docs/stage_experiments/S00/E00-05/raw",
        help="Directory for raw artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--model-path", default=_DEFAULT_MODEL, help="Local Qwen3-1.7B directory."
    )
    parser.add_argument(
        "--manifest", default=_DEFAULT_MANIFEST, help="Repo model SHA256 manifest."
    )
    parser.add_argument("--runs", type=int, default=3, help="Independent OS processes.")
    parser.add_argument(
        "--skip-negative", action="store_true", help="Skip the negative-path block."
    )
    parser.add_argument(
        "--gate-only", action="store_true",
        help="Run only the CPU artifact-gate verification then exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or new_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_records").mkdir(exist_ok=True)

    started_at = _now_utc()
    commands: List[str] = []
    commands.append(f"python3 scripts/audit/run_e00_05_qwen_tiny_smoke.py --output-dir {args.output_dir}")

    # 1) environment
    environment = collect_environment_json()
    environment["environment_block"] = collect_environment_block()
    (out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model_path = os.path.expanduser(args.model_path)
    manifest_path = str((Path(args.manifest) if Path(args.manifest).is_absolute()
                         else _REPO_ROOT / args.manifest))

    # 2) frozen workload
    frozen = load_frozen_input(WORKLOAD["isl"])
    frozen["tokenizer_sha256"] = "see identity_freeze.json"
    (out_dir / "frozen_input_ids.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    commands.append(
        f"python3 scripts/audit/run_e00_05_qwen_tiny_smoke.py --gate-only"
    )

    # 3) CPU artifact gate + identity freeze
    identity = compute_identity_freeze(model_path, manifest_path, frozen["input_ids"])
    identity["model_identity_hash"] = identity["artifact_identity_hash"]
    identity["identity_stable_vs_expected"] = (
        identity["artifact_identity_hash"] == _ARTIFACT_IDENTITY_SHA256
    )
    (out_dir / "identity_freeze.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gate = run_artifact_gate(model_path, manifest_path)
    (out_dir / "artifact_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.gate_only:
        print(f"[{EXPERIMENT_ID}] gate-only: verify_passed={gate['verify_passed']} "
              f"seconds={gate['full_snapshot_verify_seconds']} "
              f"identity_ok={identity['identity_stable_vs_expected']}")
        return 0 if (gate["verify_passed"] and identity["identity_stable_vs_expected"]) else 1

    # 4) independent-process generation runs
    lines: List[str] = []
    for i in range(1, args.runs + 1):
        run_tag = f"process{i:02d}_run_{run_id}"
        ev, so, se, rc = run_generation_subprocess(
            model_path, manifest_path, frozen["input_ids"], run_tag, out_dir
        )
        lines.append(json.dumps(ev, ensure_ascii=False))
        print(f"[{EXPERIMENT_ID}] {run_tag}: exit={rc} parsed={ev.get('parsed')} "
              f"out_hash={ev.get('result', {}).get('output_token_ids_sha256', 'n/a')[:12]}")
    (out_dir / "generation_raw.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 5) negative cases
    negative_cases: List[Dict[str, Any]] = []
    if not args.skip_negative:
        negative_cases = run_negative_cases(model_path, manifest_path, frozen["input_ids"], out_dir)
        (out_dir / "negative_cases.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in negative_cases) + "\n",
            encoding="utf-8",
        )

    # 6) cross-process agreement
    agreement: Dict[str, Any] = {"n_runs": len(lines), "distinct_output_hashes": 0,
                                  "distinct_logits_hashes": 0, "ok": False}
    records = [json.loads(l) for l in lines if l.strip()]
    if records:
        out_hashes = {
            r.get("result", {}).get("output_token_ids_sha256") for r in records
        }
        logits_hashes = {
            r.get("result", {}).get("first_logits", {}).get("logits_sha256")
            for r in records
        }
        token_counts = {
            r.get("result", {}).get("actual_output_tokens") for r in records
        }
        frozen_repro = {
            r.get("frozen_input_reproducible_by_tokenizer") for r in records
        }
        agreement = {
            "n_runs": len(records),
            "distinct_output_hashes": len(out_hashes),
            "distinct_logits_hashes": len(logits_hashes),
            "distinct_actual_token_counts": len(token_counts),
            "frozen_input_reproducible_by_tokenizer_all": frozen_repro == {True},
            "ok": len(out_hashes) == 1 and len(logits_hashes) == 1 and len(token_counts) == 1,
        }
    (out_dir / "cross_process_agreement.json").write_text(
        json.dumps(agreement, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 7) command.txt + run record
    (out_dir / "command.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    ended_at = _now_utc()

    # Aggregate minimal fields of every generation record.
    gen_summary: List[Dict[str, Any]] = []
    for r in records:
        res = r.get("result", {})
        fl = res.get("first_logits", {})
        gen_summary.append({
            "run_tag": r.get("run_tag"),
            "exit_code": r.get("exit_code"),
            "parsed": r.get("parsed"),
            "model_load_seconds": r.get("actual_model_load_seconds"),
            "frozen_input_reproducible": r.get("frozen_input_reproducible_by_tokenizer"),
            "requested_output_tokens": res.get("requested_output_tokens"),
            "actual_output_tokens": res.get("actual_output_tokens"),
            "output_token_ids_sha256": res.get("output_token_ids_sha256"),
            "output_token_ids": res.get("output_token_ids"),
            "eos_hit_in_window": res.get("eos_hit_in_window"),
            "first_token_id": fl.get("top_token_id"),
            "first_logits_l2": fl.get("l2_norm"),
            "first_logits_sha256": fl.get("logits_sha256"),
            "logits_finite": fl.get("finite"),
            "prefill_seconds": res.get("prefill_seconds"),
            "mean_decode_seconds": res.get("mean_decode_seconds"),
            "peak_cuda_allocated_mb": res.get("peak_cuda_allocated_mb"),
            "peak_cuda_reserved_mb": res.get("peak_cuda_reserved_mb"),
            "config_hash": r.get("config_hash"),
            "tokenizer_sha256": r.get("tokenizer_identity", {}).get("tokenizer_sha256"),
            "actual_architecture": r.get("actual_architecture"),
            "model_fully_on_cuda": r.get("model_identity", {}).get("fully_on_cuda"),
            "param_devices": r.get("model_identity", {}).get("param_devices"),
        })

    record: Dict[str, Any] = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "environment": collect_environment_block(),
        "workload": WORKLOAD,
        "identity_freeze": identity,
        "artifact_gate": gate,
        "generation_records": gen_summary,
        "cross_process_agreement": agreement,
        "negative_cases": negative_cases,
        "decision": None,
    }

    all_runs_ok = (
        agreement["ok"]
        and len(records) == args.runs
        and all(r.get("exit_code") == 0 and r.get("parsed") for r in records)
        and all(r.get("result", {}).get("first_logits", {}).get("finite") for r in records)
        and all(r.get("frozen_input_reproducible_by_tokenizer") for r in records)
    )
    gate_pass = bool(gate["verify_passed"]) and bool(identity["identity_stable_vs_expected"])
    negative_pass = True
    if negative_cases:
        negative_pass = all(
            c["non_zero_exit"] and c["loader_not_called"] for c in negative_cases
        ) and not args.skip_negative

    record["decision"] = {
        "normal_path_pass": all_runs_ok,
        "gate_pass": gate_pass,
        "negative_path_pass": negative_pass,
        "overall": "PASS" if (all_runs_ok and gate_pass and negative_pass) else "FAIL",
    }

    json_path = out_dir / f"e00_05_run_{run_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # file_manifest.json (sha256 of every raw artifact in this run).
    file_manifest = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            file_manifest[str(p.relative_to(out_dir))] = _file_sha256(p)
    (out_dir / "file_manifest.json").write_text(
        json.dumps(file_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(f"[{EXPERIMENT_ID}] identity_ok={identity['identity_stable_vs_expected']} "
          f"artifact_hash={identity['artifact_identity_hash'][:12]}…")
    print(f"[{EXPERIMENT_ID}] gate verify passed={gate['verify_passed']} "
          f"({gate['full_snapshot_verify_seconds']} s full snapshot)")
    print(f"[{EXPERIMENT_ID}] generation {len(records)}/{args.runs} ok; "
          f"cross-process agreement ok={agreement['ok']}")
    if records:
        r0 = gen_summary[0]
        print(f"[{EXPERIMENT_ID}] tokens={r0['actual_output_tokens']} "
              f"first={r0['first_token_id']} out_sha={str(r0['output_token_ids_sha256'])[:12]}… "
              f"peak_alloc_mb={r0['peak_cuda_allocated_mb']}")
    print(f"[{EXPERIMENT_ID}] negative: {len(negative_cases)} cases; "
          f"pass={negative_pass}")
    print(f"[{EXPERIMENT_ID}] decision={record['decision']}")
    return 0 if record["decision"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
