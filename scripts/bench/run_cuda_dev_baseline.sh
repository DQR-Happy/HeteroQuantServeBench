#!/usr/bin/env bash
# HQSB unified CUDA development-host acceptance baseline.
#
# Purpose
# -------
# Establish a repeatable *development* environment baseline on a generic
# x86_64 NVIDIA CUDA host (e.g. an RTX 3090 / sm_86) so the machine can be
# trusted as a development box for CUDA/Triton/CUTLASS work.
#
# Scope / boundaries (read before using the output)
# -------------------------------------------------
#   * This is NOT the Jetson edge acceptance flow. It never writes into
#     ``reports/jetson/**`` and never touches ``nvpmodel``/``jetson_clocks``/
#     ``tegrastats``. Results are ``development``/``smoke``/
#     ``cross-architecture validation``/``exploratory benchmark`` only.
#   * It does NOT compute 3090-vs-Jetson speedups and must not be used to
#     overwrite Jetson raw data.
#   * The Jetson build (``build/jetson-release``, sm_87) is never modified:
#     the CUDA arch is a configure-time argument here, and the CMake default
#     stays 87.
#   * No step fabricates a later PASS: every gate is PASS / FAIL / BLOCKED
#     from its own exit status, and a gate whose prerequisite failed is
#     BLOCKED, not skipped-and-assumed-good.
#
# Usage
# -----
#   ./scripts/bench/run_cuda_dev_baseline.sh \
#       --platform rtx3090 \
#       --cuda-arch 86 \
#       --model-path ~/models/hqsb/Qwen3-1.7B
#
# Options:
#   --platform NAME     label written into the report (default: rtx3090)
#   --cuda-arch N       CUDA compute capability digits, e.g. 86 or 87 (required)
#   --model-path PATH   local Qwen3-1.7B snapshot (default ~/models/hqsb/Qwen3-1.7B)
#   --manifest PATH     SHA256 manifest (default docs/benchmark/model_sha256_manifest.txt)
#   --python PATH       interpreter to use (default: ./.venv/bin/python, else python3)
#   --build-dir PATH    CMake build dir (default: build/cuda-sm<arch>-release)
#   --run-id ID         reuse an explicit run id (default: UTC timestamp)
#   --output-root DIR   report root (default: reports/dev/<platform>)
#   --with-gate8        also run the optional six-workload baseline
#   --skip-gate6        skip the model smoke gates
#   -h|--help
#
# Output: <output-root>/<run_id>/ with one directory per gate
# (command.txt, stdout.txt, stderr.txt, exit_code.txt, duration) plus
# environment.json and verdict.json. Existing runs are never overwritten.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PLATFORM="rtx3090"
CUDA_ARCH=""
MODEL_PATH="$HOME/models/hqsb/Qwen3-1.7B"
MANIFEST="docs/benchmark/model_sha256_manifest.txt"
PYTHON=""
BUILD_DIR=""
RUN_ID=""
OUTPUT_ROOT=""
WITH_GATE8=0
SKIP_GATE6=0

while [ $# -gt 0 ]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --with-gate8) WITH_GATE8=1; shift ;;
    --skip-gate6) SKIP_GATE6=1; shift ;;
    -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$CUDA_ARCH" ]; then
  echo "error: --cuda-arch is required (e.g. 86 for RTX 3090, 87 for Jetson Orin)" >&2
  exit 2
fi

# Interpreter: explicit > venv > PATH.
if [ -z "$PYTHON" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi
fi
if [ ! -x "$PYTHON" ]; then
  echo "error: python interpreter not found: $PYTHON" >&2
  exit 2
fi

if [ -z "$BUILD_DIR" ]; then
  BUILD_DIR="build/cuda-sm${CUDA_ARCH}-release"
fi
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="reports/dev/$PLATFORM"
fi

# Never write into the Jetson acceptance tree.
case "$OUTPUT_ROOT" in
  */references/jetson*|reports/jetson*|*/reports/jetson*)
    echo "error: refusing to write into the Jetson report tree ($OUTPUT_ROOT)" >&2
    exit 2 ;;
esac

# Unique run id: never overwrite an existing run.
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
  suffix=0
  while [ -e "$OUTPUT_ROOT/$RUN_ID" ]; do
    suffix=$((suffix + 1))
    RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${suffix}"
  done
fi
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
if [ -e "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  echo "error: run directory already exists and is not empty: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

# ── Toolchain discovery ────────────────────────────────────────────────────
# nvcc/ncu/nsys are commonly installed but absent from PATH on fresh hosts;
# discover them instead of silently failing a build.
for candidate in /usr/local/cuda/bin /usr/local/cuda-12.8/bin /usr/local/cuda-12.6/bin; do
  if [ -x "$candidate/nvcc" ]; then
    export PATH="$candidate:$PATH"
    export CUDA_HOME="${candidate%/bin}"
  fi
done
export PATH="$PATH:$(dirname "$PYTHON")"

CMAKE_GENERATOR="Ninja"
if ! command -v ninja >/dev/null 2>&1; then
  if command -v make >/dev/null 2>&1; then
    echo "warning: ninja not found; falling back to the 'Unix Makefiles' generator" >&2
    CMAKE_GENERATOR="Unix Makefiles"
  fi
fi

JOBS="$( (nproc 2>/dev/null || echo 4) )"

# ── Gate bookkeeping ───────────────────────────────────────────────────────
GATE_NAMES=()
GATE_STATUS=()
GATE_NOTES=()

record_gate() {  # name status note
  GATE_NAMES+=("$1")
  GATE_STATUS+=("$2")
  GATE_NOTES+=("$3")
  printf '  -> %-28s %s%s\n' "$1" "$2" "${3:+  ($3)}"
}

# run_gate <gate_dir> <command...>
# Writes command.txt/stdout.txt/stderr.txt/exit_code.txt and returns the exit
# status so callers can gate on it.
run_gate() {
  local dir="$RUN_DIR/$1"; shift
  mkdir -p "$dir"
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
  } > "$dir/command.txt"
  local start end
  start="$(date +%s)"
  "$@" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
  local rc=$?
  end="$(date +%s)"
  echo "$rc" > "$dir/exit_code.txt"
  echo "$((end - start))" > "$dir/duration_seconds.txt"
  return $rc
}

echo "=============================================================="
echo " HQSB CUDA development baseline"
echo " platform   : $PLATFORM"
echo " cuda arch  : sm_$CUDA_ARCH"
echo " run id     : $RUN_ID"
echo " output     : $RUN_DIR"
echo " python     : $PYTHON"
echo " build dir  : $BUILD_DIR"
echo "=============================================================="

# ── Stage 0: environment identity ──────────────────────────────────────────
echo "[stage 0] environment identity"
mkdir -p "$RUN_DIR/gate0_environment"
"$PYTHON" - "$RUN_DIR/gate0_environment/environment.json" "$PLATFORM" "$CUDA_ARCH" \
    "$BUILD_DIR" <<'PY' > "$RUN_DIR/gate0_environment/stdout.txt" 2>&1
import json, os, platform, subprocess, sys, datetime
out, plat, arch, build_dir = sys.argv[1:5]
def sh(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return (p.stdout or "").strip() or (p.stderr or "").strip()
    except Exception as exc:
        return f"ERR:{exc}"
def lines(cmd):
    return [l for l in sh(cmd).splitlines() if l.strip()]
torch_info = {}
try:
    import torch
    torch_info = {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        torch_info.update({
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_GiB": round(p.total_memory / 1024 ** 3, 2),
            "multi_processor_count": p.multi_processor_count,
            "warp_size": getattr(p, "warp_size", None),
            "is_integrated": getattr(p, "is_integrated", None),
            "shared_memory_per_block_optin": getattr(p, "shared_memory_per_block_optin", None),
        })
except Exception as exc:
    torch_info = {"error": f"{type(exc).__name__}: {exc}"}
pkgs = {}
for name in ["triton", "transformers", "modelscope", "pydantic", "pytest", "tilelang", "numpy"]:
    try:
        mod = __import__(name)
        pkgs[name] = getattr(mod, "__version__", "installed")
    except Exception as exc:
        pkgs[name] = f"MISSING ({type(exc).__name__})"
doc = {
    "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "role": "cuda_development_host",
    "result_class": ["development", "smoke", "cross-architecture_validation", "exploratory_benchmark"],
    "not_valid_for": "Jetson edge acceptance; do not compare absolute latency against Jetson",
    "platform": plat,
    "requested_cuda_arch": arch,
    "build_dir": build_dir,
    "git": {"head": sh("git rev-parse HEAD"), "branch": sh("git branch --show-current"),
            "dirty": bool(sh("git status --porcelain")),
            "status_short": lines("git status --porcelain")},
    "os": {"uname": sh("uname -a"), "platform": platform.platform()},
    "cpu": {"model": sh("lscpu | grep 'Model name' | head -1"), "nproc": sh("nproc")},
    "gpu": {"nvidia_smi_L": lines("nvidia-smi -L"), "topo": sh("nvidia-smi topo -m | head -8")},
    "toolchain": {"nvcc": sh("nvcc --version"), "gcc": sh("gcc --version | head -1"),
                  "cmake": sh("cmake --version | head -1"), "ninja": sh("ninja --version"),
                  "nsys": sh("nsys --version"), "ncu": sh("ncu --version | head -2"),
                  "cuda_home": os.environ.get("CUDA_HOME", ""), "nvcc_path": sh("which nvcc"),
                  "ncu_path": sh("which ncu")},
    "python": {"version": platform.python_version(), "executable": sys.executable},
    "torch": torch_info,
    "python_packages": pkgs,
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, ensure_ascii=False)
print(json.dumps({"device": torch_info.get("device_name"),
                  "capability": torch_info.get("capability"),
                  "git_head": doc["git"]["head"], "git_dirty": doc["git"]["dirty"],
                  "packages": pkgs}, indent=2, ensure_ascii=False))
PY
record_gate "gate0_environment" "PASS" "identity frozen"

# ── Gate 1: Python core regression ─────────────────────────────────────────
echo "[gate 1] python core regression (not hardware / not e2e / not performance)"
if run_gate "gate1_python" "$PYTHON" -m pytest \
      -m "not hardware and not e2e and not performance" -q; then
  summary="$(tail -3 "$RUN_DIR/gate1_python/stdout.txt" | tr '\n' ' ')"
  record_gate "gate1_python" "PASS" "$summary"
  G1=PASS
else
  summary="$(grep -E '^(FAILED|[0-9]+ (failed|passed))' "$RUN_DIR/gate1_python/stdout.txt" | tail -2 | tr '\n' ' ')"
  record_gate "gate1_python" "FAIL" "$summary"
  G1=FAIL
fi

# ── Gate 2: CUDA build for the requested arch ──────────────────────────────
echo "[gate 2] CUDA build (sm_$CUDA_ARCH) into $BUILD_DIR"
if run_gate "gate2_configure" cmake -S . -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"; then
  record_gate "gate2_configure" "PASS" "$BUILD_DIR"
  if run_gate "gate2_build" cmake --build "$BUILD_DIR" --parallel "$JOBS"; then
    record_gate "gate2_build" "PASS" "targets built"
    G2=PASS
  else
    record_gate "gate2_build" "FAIL" "see build stderr"
    G2=FAIL
  fi
else
  record_gate "gate2_configure" "FAIL" "cmake configure failed"
  record_gate "gate2_build" "BLOCKED" "configure failed"
  G2=FAIL
fi

# ── Gate 3: device query + operator correctness (CTest) ────────────────────
if [ "$G2" = PASS ]; then
  echo "[gate 3] device query + ctest"
  devq="$BUILD_DIR/bin/hqsb_device_query"
  csub="$(echo "$CUDA_ARCH" | sed 's/^\(.\)/\1./')"
  if [ ! -x "$devq" ]; then
    record_gate "gate3_device_query" "FAIL" "binary missing: $devq"
    G3=FAIL
  elif run_gate "gate3_device_query" "$devq" \
        && grep -q "Compute capability: $csub" "$RUN_DIR/gate3_device_query/stdout.txt"; then
    record_gate "gate3_device_query" "PASS" "compute capability $csub"
    if run_gate "gate3_ctest" ctest --test-dir "$BUILD_DIR" --output-on-failure; then
      record_gate "gate3_ctest" "PASS" \
        "$(grep -E 'tests passed' "$RUN_DIR/gate3_ctest/stdout.txt" | tail -1)"
      G3=PASS
    else
      record_gate "gate3_ctest" "FAIL" "operator correctness failed"
      G3=FAIL
    fi
  else
    record_gate "gate3_device_query" "FAIL" "unexpected device report"
    record_gate "gate3_ctest" "BLOCKED" "device query failed"
    G3=FAIL
  fi
else
  record_gate "gate3_device_query" "BLOCKED" "build not available"
  record_gate "gate3_ctest" "BLOCKED" "build not available"
  G3=BLOCKED
fi

# ── Gate 4: shared library + capability detection ─────────────────────────
if [ "$G2" = PASS ]; then
  echo "[gate 4] shared library + capability probe"
  LIB_PATH="$REPO_ROOT/$BUILD_DIR/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so"
  export HQSB_CUDA_RMSNORM_LIB="$LIB_PATH"
  if run_gate "gate4_capability" "$PYTHON" - "$CUDA_ARCH" <<'PY'
import json, sys
from ops.capability import detect_capabilities
from ops.dispatcher import OperatorDispatcher
want = tuple(int(c) for c in str(sys.argv[1]))
cap = detect_capabilities()
d = OperatorDispatcher(cap)
report = {"capability": cap.as_dict(),
          "dispatcher": {"rmsnorm_fp32_aligned": d.select_rmsnorm("fp32", 2048).as_dict(),
                         "rmsnorm_fp32_unaligned": d.select_rmsnorm("fp32", 101).as_dict(),
                         "rmsnorm_fp16": d.select_rmsnorm("fp16", 3).as_dict(),
                         "gemm_fp16": d.select_gemm("fp16").as_dict()}}
print(json.dumps(report, indent=2, ensure_ascii=False))
problems = []
if not cap.cuda_available: problems.append("cuda_available=false")
if tuple(cap.device_capability or ()) != want:
    problems.append(f"device_capability={cap.device_capability} != {want}")
if cap.cuda_rmsnorm_build_arch != want:
    problems.append(f"cuda_rmsnorm_build_arch={cap.cuda_rmsnorm_build_arch} != {want}")
if not cap.cuda_rmsnorm_available: problems.append("cuda_rmsnorm_available=false")
if not cap.cublas_available: problems.append("cublas_available=false")
if d.select_rmsnorm("fp32", 2048).backend != "cuda":
    problems.append("dispatcher did not select the hand-written CUDA path")
if problems:
    print("PROBLEMS: " + "; ".join(problems))
    sys.exit(1)
print("capability+dispatcher OK")
PY
  then
    record_gate "gate4_capability" "PASS" "arch-matched CUDA dispatch"
    G4=PASS
  else
    record_gate "gate4_capability" "FAIL" \
      "$(grep -o 'PROBLEMS:.*' "$RUN_DIR/gate4_capability/stdout.txt" | head -1)"
    G4=FAIL
  fi
else
  record_gate "gate4_capability" "BLOCKED" "build not available"
  G4=BLOCKED
fi

# ── Gate 5: Triton / cuBLAS / CUTLASS ─────────────────────────────────────
echo "[gate 5] Triton + cuBLAS + CUTLASS"
mkdir -p "$RUN_DIR/gate5_backends"
cat > "$RUN_DIR/gate5_backends/triton_cublas_probe.py" <<'PY'
import torch, triton, triton.language as tl
@triton.jit
def _scale2(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=mask) * 2.0, mask=mask)
def main() -> int:
    n = 4096
    x = torch.randn(n, device="cuda")
    o = torch.empty_like(x)
    _scale2[(triton.cdiv(n, 1024),)](x, o, n, BLOCK=1024)
    torch.cuda.synchronize()
    err = (o - 2 * x).abs().max().item()
    print(f"triton={triton.__version__} probe_max_err={err}")
    if err != 0.0:
        return 1
    a = torch.randn(512, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    ref = (a.float() @ b.float()).half()
    print(f"cublas(torch.matmul) fp16 max_err={float((a @ b - ref).abs().max()):.4f}")
    return 0
raise SystemExit(main())
PY
if run_gate "gate5_triton_cublas" "$PYTHON" "$RUN_DIR/gate5_backends/triton_cublas_probe.py"; then
  record_gate "gate5_triton_cublas" "PASS" \
    "$(grep -o 'triton=[^ ]* probe_max_err=[^ ]*' "$RUN_DIR/gate5_triton_cublas/stdout.txt" | head -1)"
  G5=PARTIAL
else
  record_gate "gate5_triton_cublas" "FAIL" "Triton/cuBLAS probe failed"
  G5=FAIL
fi

cutlass_bin="$BUILD_DIR/bin/hqsb_cutlass_gemm_bench"
if [ -x "$cutlass_bin" ]; then
  if run_gate "gate5_cutlass_gemm" "$cutlass_bin" --m 512 --n 2048 --k 2048; then
    record_gate "gate5_cutlass_gemm" "PASS" \
      "$(tail -1 "$RUN_DIR/gate5_cutlass_gemm/stdout.txt")"
    [ "$G5" = PARTIAL ] && G5=PASS
  else
    record_gate "gate5_cutlass_gemm" "FAIL" \
      "$(tail -1 "$RUN_DIR/gate5_cutlass_gemm/stderr.txt")"
    G5=FAIL
  fi
else
  record_gate "gate5_cutlass_gemm" "BLOCKED" "target not configured (CUTLASS headers absent)"
  [ "$G5" = PARTIAL ] && G5=BLOCKED
fi

# ── Gate 6: real Qwen model smoke ─────────────────────────────────────────
G6=BLOCKED
if [ "$SKIP_GATE6" = "1" ]; then
  record_gate "gate6_artifact_integrity" "BLOCKED" "--skip-gate6"
  record_gate "gate6_model_smoke" "BLOCKED" "--skip-gate6"
elif [ ! -d "$MODEL_PATH" ]; then
  record_gate "gate6_artifact_integrity" "BLOCKED" "model snapshot missing: $MODEL_PATH"
  record_gate "gate6_model_smoke" "BLOCKED" "model snapshot missing"
else
  echo "[gate 6] real Qwen3 model smoke"
  if run_gate "gate6_artifact_integrity" "$PYTHON" scripts/models/verify_qwen3_hashes.py \
        --model-path "$MODEL_PATH" --manifest "$MANIFEST"; then
    record_gate "gate6_artifact_integrity" "PASS" "manifest verified byte-exactly"
    gate6_ok=1
  else
    record_gate "gate6_artifact_integrity" "FAIL" \
      "$(grep -E '^Result ' "$RUN_DIR/gate6_artifact_integrity/stdout.txt" | head -1)"
    gate6_ok=0
  fi

  # The unmodified E00-05 driver. It will refuse to load when the artifact
  # gate fails, which is recorded rather than worked around.
  if run_gate "gate6_e00_05" "$PYTHON" scripts/audit/run_e00_05_qwen_tiny_smoke.py \
        --model-path "$MODEL_PATH" --manifest "$MANIFEST" --runs 3 --skip-negative \
        --output-dir "$RUN_DIR/gate6_e00_05/raw"; then
    record_gate "gate6_e00_05" "PASS" "E00-05 overall PASS"
    G6=PASS
  else
    record_gate "gate6_e00_05" "FAIL" \
      "$(grep -o "decision=.*" "$RUN_DIR/gate6_e00_05/stdout.txt" | tail -1)"
    G6=FAIL
  fi

  # Supplementary, clearly separate: canonical loader smoke (no manifest gate)
  if run_gate "gate6_loader_smoke" "$PYTHON" scripts/models/smoke_qwen3.py; then
    record_gate "gate6_loader_smoke" "PASS" \
      "$(grep -E 'Tokens/second|Model loaded in' "$RUN_DIR/gate6_loader_smoke/stdout.txt" | tail -2 | tr '\n' ' ')"
  else
    record_gate "gate6_loader_smoke" "FAIL" "$(tail -1 "$RUN_DIR/gate6_loader_smoke/stderr.txt")"
  fi
fi

# ── Gate 7: S04 multi-backend development baseline ────────────────────────
G7=BLOCKED
if [ "$G2" = PASS ]; then
  echo "[gate 7] S04 backend baseline"
  lib="$REPO_ROOT/$BUILD_DIR/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so"
  if [ -f "$lib" ]; then
    export HQSB_CUDA_RMSNORM_LIB="$lib"
    if run_gate "gate7_s04_bench" "$PYTHON" scripts/bench/bench_s04.py \
          --output "$RUN_DIR/gate7_s04_bench/s04_backend_baseline.json"; then
      record_gate "gate7_s04_bench" "PASS" \
        "$(grep -cE '^  ' "$RUN_DIR/gate7_s04_bench/stdout.txt") measurements recorded"
      G7=PASS
    else
      record_gate "gate7_s04_bench" "FAIL" "bench_s04.py failed"
      G7=FAIL
    fi
  else
    record_gate "gate7_s04_bench" "BLOCKED" "shared library missing: $lib"
  fi
else
  record_gate "gate7_s04_bench" "BLOCKED" "build not available"
fi

# ── Gate 8 (optional): six-workload model baseline ────────────────────────
G8=SKIPPED
if [ "$WITH_GATE8" = "1" ]; then
  echo "[gate 8] optional six-workload baseline"
  if run_gate "gate8_s02_baseline" "$PYTHON" scripts/bench/run_s02_baseline.py \
        --config "configs/benchmarks/qwen3_1_7b_fp16.yaml" \
        --model-path "$MODEL_PATH" --manifest "$MANIFEST" --repetitions 3 \
        --output-dir "$RUN_DIR/gate8_s02_baseline/results"; then
    record_gate "gate8_s02_baseline" "PASS" "six workloads + summary.csv"
    G8=PASS
  else
    record_gate "gate8_s02_baseline" "FAIL" "$(tail -1 "$RUN_DIR/gate8_s02_baseline/stderr.txt")"
    G8=FAIL
  fi
else
  record_gate "gate8_s02_baseline" "SKIPPED" "pass --with-gate8 to enable"
fi

# ── Verdict ───────────────────────────────────────────────────────────────
n_gates=${#GATE_NAMES[@]}
"$PYTHON" - "$RUN_DIR/verdict.json" "$PLATFORM" "$CUDA_ARCH" "$RUN_DIR" \
    "$G1" "$G2" "$G3" "$G4" "$G5" "$G6" "$G7" "$G8" <<'PY'
import json, subprocess, sys
out, platform, arch, run_dir, g1, g2, g3, g4, g5, g6, g7, g8 = sys.argv[1:13]

def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""

levels = []
if g1 == "PASS":
    levels.append({"require": "Gate 1", "claim": "Python/HQSB core environment OK"})
if g1 == "PASS" and g2 == "PASS" and g3 == "PASS":
    levels.append({"require": "Gates 1-3", "claim": "RTX-class CUDA development environment OK"})
if all(x == "PASS" for x in (g1, g2, g3, g4, g5, g6)):
    levels.append({"require": "Gates 1-6", "claim": "real-model development environment OK"})
if all(x == "PASS" for x in (g1, g2, g3, g4, g5, g6, g7)):
    levels.append({"require": "Gates 1-7", "claim": "S04/S04.5 development environment ready"})
if all(x == "PASS" for x in (g1, g2, g3, g4, g5, g6, g7, g8)):
    levels.append({"require": "Gates 1-8", "claim": "full RTX development baseline established"})

doc = {
    "run_dir": run_dir,
    "platform": platform,
    "cuda_arch": arch,
    "result_class": ["development", "smoke", "cross-architecture_validation",
                     "exploratory_benchmark"],
    "not_valid_for": ["Jetson edge acceptance", "3090-vs-Jetson speedup claims"],
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
    "gate_status": {"gate1_python": g1, "gate2_cuda_build": g2, "gate3_correctness": g3,
                    "gate4_capability": g4, "gate5_backends": g5, "gate6_model": g6,
                    "gate7_s04_baseline": g7, "gate8_full_baseline": g8},
    "levels_achieved": levels,
    "note": ("BLOCKED means a prerequisite failed and the gate was not run; "
             "it is never reported as PASS."),
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, ensure_ascii=False)
print(json.dumps(doc, indent=2, ensure_ascii=False))
PY

echo
echo "verdict: $RUN_DIR/verdict.json"
