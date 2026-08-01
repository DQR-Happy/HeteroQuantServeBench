"""Evidence helpers for the remaining S05 collectors (execute on Jetson only)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/stage_experiments/S05"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prereqs():
    result = {}
    for name in ("E05-01", "E05-02", "E05-03", "E05-04"):
        path = STAGE / name / "raw/verdict.json"
        value = read(path, {})
        result[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path) if path.is_file() else None,
                        "overall": value.get("overall") or value.get("status") or value.get("verdict"),
                        "raw_status_fields": {key: value.get(key) for key in ("overall", "status", "verdict") if key in value},
                        "scientific": value.get("scientific_execution_verdict")}
    m4 = ROOT / "docs/stage_experiments/S04.5"
    result["S04.5"] = {"evidence_files": [str(p.relative_to(ROOT)) for p in m4.rglob("*.json")] if m4.exists() else [],
                        "verified": False, "reason": "No accepted M4 evidence in the supplied evidence chain"}
    handoff = STAGE / "E05-03/raw/selection/e05_04_handoff.json"
    result["calibration_handoff"] = {"exists": handoff.is_file(), "sha256": sha(handoff) if handoff.is_file() else None}
    return result


def environment():
    import torch
    import triton
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "host": platform.node(),
            "platform": platform.platform(), "python": platform.python_version(), "pid": os.getpid(),
            "torch": str(torch.__version__), "triton": str(triton.__version__), "cuda": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            "cuda_memory": list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None,
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout,
            "execution_host_rule": "Jetson via scripts/remote_run.sh"}


def initialize(experiment, spec, output=None):
    output = Path(output) if output else STAGE / experiment / "raw"
    output.mkdir(parents=True, exist_ok=True)
    frozen = {"experiment_id": experiment, **spec}
    previous = read(output / "spec.json")
    if previous is not None and previous != frozen:
        raise RuntimeError("Spec changed: use a fresh output directory")
    write(output / "spec.json", frozen)
    if not (output / "prerequisites.json").exists():
        write(output / "prerequisites.json", prereqs())
    snapshot = environment()
    if not (output / "environment.json").exists():
        write(output / "environment.json", snapshot)
    write(output / f"environments/process_{os.getpid()}.json", snapshot)
    return output


def manifest(output):
    output = Path(output)
    entries = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(output.rglob("*")) if p.is_file() and p.name != "EVIDENCE_MANIFEST.json" and not p.name.endswith(".tmp")]
    value = {"schema": "hqsb.s05.evidence.v1", "files": entries, "file_count": len(entries),
             "root_hash": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}
    write(output / "EVIDENCE_MANIFEST.json", value)
    return value


def ci95(values):
    """Student-t 95% interval over independent process means, never launches."""
    values = list(values)
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95": None, "n": len(values), "reason": "fewer than two independent processes"}
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 2.571)
    half = critical * statistics.stdev(values) / len(values) ** 0.5
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": len(values), "method": "Student-t over process means"}


def load_weight(model_path, name):
    from safetensors import safe_open
    model_path = Path(model_path).expanduser()
    index = read(model_path / "model.safetensors.index.json")
    with safe_open(model_path / index["weight_map"][name], framework="pt", device="cpu") as stream:
        return stream.get_tensor(name)


def quant_weight(weight, bits, group=128):
    """Symmetric RTN used only by the diagnostic collectors; scale is FP32."""
    import torch
    import torch.nn.functional as F
    n, k = weight.shape
    group = k if bits == 8 else group
    padded = ((k + group - 1) // group) * group
    source = F.pad(weight.float(), (0, padded - k)).reshape(n, -1, group)
    qmax = 2 ** (bits - 1) - 1
    amax = source.abs().amax(-1, keepdim=True)
    scales = torch.where(amax == 0, torch.ones_like(amax), amax / qmax)
    q = torch.round(source / scales).clamp(-qmax, qmax).to(torch.int8)
    reconstructed = (q.float() * scales).reshape(n, padded)[:, :k].to(weight.dtype)
    return q.reshape(n, padded)[:, :k], scales.squeeze(-1).contiguous(), reconstructed


def prepared_weight(weight, bits, group=128):
    import torch
    import torch.nn.functional as F
    from ops.quant.w4a16_triton import PreparedWeights
    q, scales, reconstructed = quant_weight(weight, bits, group)
    n, k = q.shape
    if bits == 4:
        q = F.pad(q, (0, k % 2))
        packed = ((q[:, 0::2].to(torch.int16) & 15) | ((q[:, 1::2].to(torch.int16) & 15) << 4)).to(torch.uint8)
    else:
        packed = q.to(torch.uint8)
    packed = packed.contiguous()
    fingerprint = hashlib.sha256(packed.cpu().numpy().tobytes() + scales.cpu().numpy().tobytes()).hexdigest()
    prepared = PreparedWeights(packed, scales, None, n, k, packed.shape[1], scales.shape[1],
                               k if bits == 8 else group, bits, f"hqsb.w{bits}a16.rowmajor.nk.v1", fingerprint)
    return prepared, reconstructed
