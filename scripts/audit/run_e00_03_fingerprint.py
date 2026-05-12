#!/usr/bin/env python3
"""E00-03 — Run / environment fingerprint stability and sensitivity.

Question
--------
Is a run's identity (environment + inputs) stable under identical
conditions, and does it reliably change when exactly one controlled field
changes?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  Two consecutive fingerprints generated on the same machine / same
    configuration are identical at the ``environment_fingerprint`` and
    ``run_fingerprint`` level (across in-process repeats and independent
    processes). Changing exactly one controlled field changes the
    corresponding section digest (and the aggregate root that contains it),
    while leaving the other roots/sections untouched. Volatile observations
    (clock / temperature) are *recorded* through ``volatile_digest`` but
    never change ``run_fingerprint``.
H0  Either two identical runs disagree, or a controlled single-field change
    fails to move the corresponding identity.

Design
------
* **Baseline**: collect the eight identity sections, compute the fingerprint
  ``--repeat`` times in-process (proving in-process stability), and write one
  JSON per process run so multiple process runs can be compared.
* **Sensitivity**: apply one controlled change at a time and verify the
  resulting digest movement is exactly as predicted:
    - environment sections (os/device/python/packages/power) move
      ``environment_fingerprint`` AND ``run_fingerprint``;
    - input sections (config/model/commit) move ``run_fingerprint`` only;
    - volatile observations move ``volatile_digest`` only.

Raw output
----------
``<out>/e00_03_run_<run_id>.json``     full record (baseline + sensitivity)
``<out>/e00_03_run_<run_id>_env.json`` frozen environment block
``<out>/cross_process_agreement.json`` fingerprint agreement across runs

Usage
-----
    python3 scripts/audit/run_e00_03_fingerprint.py \
        --output-dir docs/stage_experiments/S00/E00-03/raw --repeat 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.core.config import config_hash  # noqa: E402
from hqsb.core.contracts.model import ModelArtifact  # noqa: E402
from hqsb.core.fingerprint import (  # noqa: E402
    ConfigSection,
    FingerprintSections,
    ModelSection,
    RunFingerprint,
    VolatileObservations,
    collect_commit,
    collect_device_basic,
    collect_os,
    collect_packages,
    collect_power,
    collect_python,
    collect_volatile,
    compute_run_fingerprint,
    diff_sections,
    sha256_hex,
)
from hqsb.core.ids import new_run_id  # noqa: E402
from hqsb.hardware.probe import cuda_device_probe  # noqa: E402
from hqsb.models.manifest import load_manifest  # noqa: E402

EXPERIMENT_ID = "E00-03"
STAGE = "S00"

_MODEL_SOURCE = "modelscope"
_MODEL_ARCHITECTURE = "Qwen3ForCausalLM"
_MODEL_DTYPE = "float16"


class _ConfigDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark: Dict[str, Any]
    workloads: List[Dict[str, Any]]


# ── Section builders ────────────────────────────────────────────────────


def _relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_config_section(config_path: Path) -> ConfigSection:
    text = config_path.read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    semantic_hash = config_hash(_ConfigDoc.model_validate(document))
    return ConfigSection(
        config_path=_relpath(config_path),
        config_sha256=sha256_hex(text),
        config_hash=semantic_hash,
    )


def collect_model_section(manifest_path: Path, model_id: str) -> ModelSection:
    entries = load_manifest(str(manifest_path))
    file_hashes = {entry.normalized_path: entry.sha256 for entry in entries}
    artifact = ModelArtifact(
        model_id=model_id,
        source=_MODEL_SOURCE,
        architecture=_MODEL_ARCHITECTURE,
        dtype=_MODEL_DTYPE,
        file_hashes=file_hashes,
    )
    return ModelSection(
        model_id=model_id,
        manifest_path=_relpath(manifest_path),
        manifest_sha256=_file_sha256(manifest_path),
        model_hash=artifact.identity_hash(),
    )


def build_sections(
    config_path: Path, manifest_path: Path, model_id: str, repo_path: Path
) -> FingerprintSections:
    return FingerprintSections(
        os=collect_os(),
        device=collect_device_basic(cuda_device_probe()),
        python=collect_python(),
        packages=collect_packages(),
        power=collect_power(),
        config=collect_config_section(config_path),
        model=collect_model_section(manifest_path, model_id),
        commit=collect_commit(str(repo_path)),
    )


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
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _git(*args: str) -> str:
    try:
        import subprocess

        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


# ── Sensitivity harness ─────────────────────────────────────────────────


def _mutate_sections(
    sections: FingerprintSections, section_name: str, field: str, value: Any
) -> FingerprintSections:
    data = sections.model_dump()
    if section_name == "packages":
        data["packages"][field] = value
    else:
        data[section_name][field] = value
    return FingerprintSections.model_validate(data)


def _modified_config(config_path: Path, tmp_dir: Path) -> Path:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["benchmark"]["batch_size"] = document["benchmark"].get("batch_size", 1) + 1
    target = tmp_dir / "config_modified.yaml"
    target.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return target


def _modified_manifest(manifest_path: Path, tmp_dir: Path) -> Path:
    lines = manifest_path.read_text(encoding="utf-8").splitlines(keepends=True)
    target = tmp_dir / "manifest_modified.txt"
    if lines:
        digest, _, rest = lines[0].partition(" ")
        lines[0] = "0" * 64 + "  " + rest
    target.write_text("".join(lines), encoding="utf-8")
    return target


def _arm(
    name: str,
    kind: str,
    changed_section: str,
    baseline: RunFingerprint,
    mutated: RunFingerprint,
) -> Dict[str, Any]:
    changed_digests = diff_sections(baseline, mutated)
    env_changed = baseline.environment_fingerprint != mutated.environment_fingerprint
    run_changed = baseline.run_fingerprint != mutated.run_fingerprint
    volatile_changed = baseline.volatile_digest != mutated.volatile_digest

    if kind == "environment":
        expected = (
            env_changed
            and run_changed
            and not volatile_changed
            and changed_digests == [changed_section]
        )
    elif kind == "input":
        expected = (
            run_changed
            and not env_changed
            and not volatile_changed
            and changed_digests == [changed_section]
        )
    else:  # volatile
        expected = (
            volatile_changed
            and not run_changed
            and not env_changed
            and changed_digests == []
        )

    return {
        "name": name,
        "kind": kind,
        "changed_section": changed_section,
        "environment_fingerprint_changed": env_changed,
        "run_fingerprint_changed": run_changed,
        "volatile_digest_changed": volatile_changed,
        "section_digests_changed": changed_digests,
        "pass": expected,
    }


def run_sensitivity(
    baseline: RunFingerprint,
    baseline_sections: FingerprintSections,
    baseline_volatile: VolatileObservations,
    config_path: Path,
    manifest_path: Path,
    model_id: str,
    repo_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="hqsb_e00_03_"))
    arms: List[Dict[str, Any]] = []

    def _fp(sections: FingerprintSections, volatile: VolatileObservations) -> RunFingerprint:
        return compute_run_fingerprint(sections, volatile)

    # Real input changes (re-collected from modified files).
    modified_config = _modified_config(config_path, tmp_dir)
    sections = build_sections(modified_config, manifest_path, model_id, repo_path)
    arms.append(
        _arm("config_batch_size", "input", "config", baseline, _fp(sections, baseline_volatile))
    )

    modified_manifest = _modified_manifest(manifest_path, tmp_dir)
    sections = build_sections(config_path, modified_manifest, model_id, repo_path)
    arms.append(
        _arm("manifest_digest_flip", "input", "model", baseline, _fp(sections, baseline_volatile))
    )

    # Controlled single-field overrides.
    arms.append(
        _arm(
            "power_mode",
            "environment",
            "power",
            baseline,
            _fp(
                _mutate_sections(baseline_sections, "power", "nvpmodel_mode", 0),
                baseline_volatile,
            ),
        )
    )
    arms.append(
        _arm(
            "python_version",
            "environment",
            "python",
            baseline,
            _fp(
                _mutate_sections(baseline_sections, "python", "version", "3.99.0"),
                baseline_volatile,
            ),
        )
    )
    arms.append(
        _arm(
            "os_release",
            "environment",
            "os",
            baseline,
            _fp(
                _mutate_sections(baseline_sections, "os", "release", "6.6.6-test"),
                baseline_volatile,
            ),
        )
    )
    arms.append(
        _arm(
            "device_runtime",
            "environment",
            "device",
            baseline,
            _fp(
                _mutate_sections(
                    baseline_sections, "device", "cuda_runtime_version", "99.0"
                ),
                baseline_volatile,
            ),
        )
    )
    arms.append(
        _arm(
            "package_torch",
            "environment",
            "packages",
            baseline,
            _fp(
                _mutate_sections(baseline_sections, "packages", "torch", "9.9.9"),
                baseline_volatile,
            ),
        )
    )
    arms.append(
        _arm(
            "commit_sha",
            "input",
            "commit",
            baseline,
            _fp(
                _mutate_sections(
                    baseline_sections, "commit", "commit", "0" * 40
                ),
                baseline_volatile,
            ),
        )
    )

    # Volatile observation change: must be recorded but not affect identity.
    mutated_volatile = baseline_volatile.model_copy(
        update={"max_temperature_c": (baseline_volatile.max_temperature_c or 0.0) + 100.0}
    )
    arms.append(
        _arm(
            "temperature_delta",
            "volatile",
            "volatile",
            baseline,
            _fp(baseline_sections, mutated_volatile),
        )
    )

    summary = {
        "total": len(arms),
        "passed": sum(1 for arm in arms if arm["pass"]),
        "passed_environment": sum(1 for a in arms if a["pass"] and a["kind"] == "environment"),
        "passed_input": sum(1 for a in arms if a["pass"] and a["kind"] == "input"),
        "passed_volatile": sum(1 for a in arms if a["pass"] and a["kind"] == "volatile"),
    }
    return arms, summary


# ── Cross-process agreement ─────────────────────────────────────────────


def compare_runs(out_dir: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("e00_03_run_*.json")):
        if path.name.endswith("_env.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "baseline" not in payload:
            continue
        runs.append(
            {
                "file": path.name,
                "sha256": _file_sha256(path),
                "run_id": payload.get("run_id"),
                "environment_fingerprint": payload["baseline"]["environment_fingerprint_values"][0],
                "run_fingerprint": payload["baseline"]["run_fingerprint_values"][0],
                "started_at_utc": payload.get("environment", {}).get("started_at_utc"),
                "baseline_stable": payload["baseline"]["stable"],
                "sensitivity_passed": payload["sensitivity_summary"]["passed"],
                "sensitivity_total": payload["sensitivity_summary"]["total"],
            }
        )

    env_fps = sorted({r["environment_fingerprint"] for r in runs})
    run_fps = sorted({r["run_fingerprint"] for r in runs})
    agreement = {
        "experiment_id": EXPERIMENT_ID,
        "runs": runs,
        "run_count": len(runs),
        "distinct_environment_fingerprints": env_fps,
        "distinct_run_fingerprints": run_fps,
        "environment_agreement": len(env_fps) <= 1,
        "run_agreement": len(run_fps) <= 1,
        "agreement": len(env_fps) <= 1 and len(run_fps) <= 1,
    }
    (out_dir / "cross_process_agreement.json").write_text(
        json.dumps(agreement, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return agreement


# ── Drivers ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E00-03 run fingerprint.")
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S00/E00-03/raw",
        help="Directory for raw JSON artifacts.",
    )
    parser.add_argument(
        "--config",
        default="configs/benchmarks/jetson_qwen3_fp16.yaml",
        help="Benchmark config file (relative to repo root).",
    )
    parser.add_argument(
        "--manifest",
        default="docs/benchmark/model_sha256_manifest.txt",
        help="Model sha256 manifest file (relative to repo root).",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="In-process repeat count for baseline stability.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or new_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = _REPO_ROOT / args.config
    manifest_path = _REPO_ROOT / args.manifest
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    # Baseline: collect the fingerprint ``repeat`` times in-process.
    baseline_samples: List[Dict[str, Any]] = []
    for index in range(max(1, args.repeat)):
        sections = build_sections(config_path, manifest_path, args.model_id, _REPO_ROOT)
        volatile = collect_volatile()
        fp = compute_run_fingerprint(sections, volatile)
        fp.verify()
        baseline_samples.append(
            {
                "index": index,
                "environment_fingerprint": fp.environment_fingerprint,
                "run_fingerprint": fp.run_fingerprint,
                "volatile_digest": fp.volatile_digest,
                "section_digests": fp.section_digests,
                "volatile": volatile.model_dump(mode="json"),
            }
        )

    env_values = [s["environment_fingerprint"] for s in baseline_samples]
    run_values = [s["run_fingerprint"] for s in baseline_samples]
    volatile_values = [s["volatile_digest"] for s in baseline_samples]

    baseline = {
        "samples": baseline_samples,
        "environment_fingerprint_values": env_values,
        "run_fingerprint_values": run_values,
        "volatile_digest_values": volatile_values,
        "distinct_environment": len(set(env_values)),
        "distinct_run": len(set(run_values)),
        "distinct_volatile": len(set(volatile_values)),
        "stable": len(set(env_values)) <= 1 and len(set(run_values)) <= 1,
    }

    # Sensitivity: one controlled change at a time.
    reference_sections = build_sections(config_path, manifest_path, args.model_id, _REPO_ROOT)
    reference_volatile = collect_volatile()
    reference_fp = compute_run_fingerprint(reference_sections, reference_volatile)
    arms, sensitivity_summary = run_sensitivity(
        reference_fp,
        reference_sections,
        reference_volatile,
        config_path,
        manifest_path,
        args.model_id,
        _REPO_ROOT,
    )

    record: Dict[str, Any] = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "environment": collect_environment_block(),
        "baseline": baseline,
        "reference_fingerprint": {
            "environment_fingerprint": reference_fp.environment_fingerprint,
            "run_fingerprint": reference_fp.run_fingerprint,
            "volatile_digest": reference_fp.volatile_digest,
            "section_digests": reference_fp.section_digests,
            "sections": reference_fp.sections.model_dump(mode="json"),
        },
        "sensitivity": arms,
        "sensitivity_summary": sensitivity_summary,
    }

    json_path = out_dir / f"e00_03_run_{run_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env_path = out_dir / f"e00_03_run_{run_id}_env.json"
    env_path.write_text(
        json.dumps(record["environment"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    agreement = compare_runs(out_dir, record)

    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(
        f"[{EXPERIMENT_ID}] baseline stable={baseline['stable']} "
        f"(env distinct={baseline['distinct_environment']}, "
        f"run distinct={baseline['distinct_run']}, "
        f"volatile distinct={baseline['distinct_volatile']})"
    )
    print(
        f"[{EXPERIMENT_ID}] environment_fingerprint={reference_fp.environment_fingerprint}"
    )
    print(f"[{EXPERIMENT_ID}] run_fingerprint={reference_fp.run_fingerprint}")
    print(
        f"[{EXPERIMENT_ID}] sensitivity {sensitivity_summary['passed']}"
        f"/{sensitivity_summary['total']} pass "
        f"(env={sensitivity_summary['passed_environment']}, "
        f"input={sensitivity_summary['passed_input']}, "
        f"volatile={sensitivity_summary['passed_volatile']})"
    )
    print(
        f"[{EXPERIMENT_ID}] cross_process agreement={agreement['agreement']} "
        f"(runs={agreement['run_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
