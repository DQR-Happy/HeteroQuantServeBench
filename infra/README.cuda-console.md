# Linux CUDA Console deployment

The console runs on a Linux CUDA target. Keep editing/build execution separate:
from the Mac, invoke commands through `scripts/remote_run.sh`, selecting the target
with `HQSB_REMOTE_HOST` and `HQSB_REMOTE_DIR`. Jetson remains the default target;
`4090D-hqsb` is an additional SSH alias configured on the editing client.

## Prerequisites and initialization

Use Python >=3.10, an already working **CUDA PyTorch**, a matching CUDA toolkit
(`nvcc` at `/usr/local/cuda/bin/nvcc` or pass `--cuda-compiler`), CMake >=3.18,
a C++17 compiler, and Node.js >=20.19 with npm. The verified 4090D image uses
Python 3.12.3, torch 2.8.0+cu128, CUDA 12.8, SM 8.9 and Node 22.16.0.
Do not copy Jetson's aarch64 wheel, environment lock or `.so` to x86_64.

After cloning on the target, run:

```bash
# Enter these commands on the Mac; they execute on the target.
export HQSB_REMOTE_HOST=4090D-hqsb
export HQSB_REMOTE_DIR=/root/HeteroQuantServeBench-4090D
./scripts/remote_run.sh bash scripts/console/bootstrap_cuda.sh
./scripts/remote_run.sh bash scripts/console/serve.sh \
  --config configs/console/cuda.example.yaml --port 8768
```

`bootstrap_cuda.sh` creates a `.venv-console` overlay with
`--system-site-packages` to reuse the target's CUDA torch. It installs the
`console,console-model,dev` extras, verifies the exact Qwen3 adapter source,
downloads the manifest-declared Qwen3-1.7B files, verifies their hashes, builds
native kernels for the actual GPU, and builds the locked frontend dependencies.
It does not install or replace the driver, toolkit or base torch. Set
`HQSB_PYTHON` / `HQSB_CONSOLE_VENV` to select a different interpreter/environment;
activate that environment for later commands. Node must be on PATH.

The model extra pins Transformers 5.8.0 and ModelScope 1.29.0 for the source-pinned
Qwen3 adapter; the broader research `benchmark` extra is intentionally separate.
API-only installations can still install just `[console]` without model packages.
The build script detects the executing GPU's SM instead of assuming 87, checks
that toolkit and torch CUDA versions match, and records the selected SM in the
content-addressed artifact manifest. Explicit cross-build example:

```bash
./scripts/remote_run.sh .venv-console/bin/python \
  scripts/observation/build_model_kernels.py --cuda-architectures '87;89'
```

Start from the checkout root: the generic CUDA configuration uses relative
manifest paths, `~/models/hqsb/Qwen3-1.7B`, no privileged-worker bridge, and the
existing no-login test mode. Keep it bound to loopback and access through SSH:

```bash
ssh -N -L 127.0.0.1:8879:127.0.0.1:8768 4090D-hqsb
# Browser: http://127.0.0.1:8879/benchmarks
```

The initialized 4090D environment is `/root/hqsb-env`, linked as `.venv-console`.
Its Node/npm/npx links refer to the SHA-256-verified official Node installation
under `/root/hqsb-tools/node-v22.16.0-linux-x64`. Activating `.venv-console`
exposes both Python and these Node tools. The service data is `.console/`;
validation logs and reports are in `reports/deployment/4090d/`.

## Network and clean-checkout boundaries

Initial GitHub requests timed out. A command-scoped direct connection and later
fetch succeeded; follow-up inspection found no global or environment proxy, so
the initial timeout cannot be attributed to an inherited proxy. Temporary local
proxy overrides were removed. No global Git configuration was modified. For a
different image that actually has an unusable inherited proxy, a command-scoped
override can isolate that cause; do not disable a proxy your network requires:

```bash
# On the target, if and only if its inherited proxy is unusable:
git -c http.proxy= -c https.proxy= clone \
  https://github.com/DQR-Happy/HeteroQuantServeBench.git
# Inside that checkout, if needed:
git config --local http.proxy ''
git config --local https.proxy ''
```

The initial checkout was transferred as a Git bundle of the client HEAD, then
successfully fetched from origin/main and checked against the same commit.
No developer SSH key or GitHub credential was copied to the target.
Private stage archives are not part of a clean clone. The runtime validator can
explicitly accept their absence while recording `not_provided`; strict archive
validation remains the default:

```bash
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_live.py \
  --url http://127.0.0.1:8768 --data-dir .console --allow-empty-history
```

## Measurement semantics

On Jetson, existing tegrastats sampling retains SoC VDD_IN power semantics.
On discrete NVIDIA targets, bounded `nvidia-smi` queries collect **physical GPU
0** identity, utilization, temperature, board power and framebuffer usage.
These values are host-level samples, not per-request GPU energy or SM occupancy;
CUDA_VISIBLE_DEVICES remapping does not change the declared physical sensor ID.
Unsupported sensor values remain null. Host DRAM, framebuffer usage, PyTorch
allocated/reserved memory and CPU page samples retain separate scopes.
CUDA activity traces do not by themselves provide cache-hit counters, physical
GPU page tables or a complete physical memory access trace.

The trace reader now enforces an explicit 128-container nesting budget per JSON
value. Python 3.12 no longer reliably rejects the old over-deep input via the
same recursion failure as Python 3.10. Quoted brackets do not count as nesting;
the existing file/value/event limits remain in force.

Hardware smoke tests and fixed-input results establish functional compatibility,
not universal numerical equivalence, a performance win or production readiness.

The 2026-10-16 setup passed 3708 CPU unit/property tests (30 hardware/performance
cases deselected), 21 real CUDA kernel tests, 25 frontend unit tests, a full
bootstrap rerun, 12 model-level kernel-profile smoke checks, and the default
three-scheme/four-case fixed suite. Memory and CPU-page captures were complete;
the operator capture included CPU and CUDA activities with a hash-verified trace.
Whole-repository Ruff still reports the same 55 findings as the unchanged HEAD;
modified Python files pass. Raw evidence and source hashes are retained privately
under `reports/deployment/4090d/acceptance.json` and adjacent logs.
