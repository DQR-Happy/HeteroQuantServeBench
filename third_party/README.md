# Third-party dependencies

This directory holds external, header-only/large dependencies that are
**fetched on demand** and excluded from version control (`.gitignore`). The
S04 capability detector (`ops/capability.py`) discovers them at runtime and
gracefully reports `False` when they are absent — the core CUDA/Triton path
never hard-depends on them.

## CUTLASS / CuTe (S04 GEMM comparison)

CUTLASS is a header-only template library (no linking). Fetch a shallow
checkout into `third_party/cutlass`:

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

Then verify the capability probe detects it:

```bash
python3 -c "from ops.capability import detect_capabilities; print(detect_capabilities().cutlass_available)"
# -> True
```

Override locations (CI/cloud machines) via the environment variable:

```bash
export HQSB_CUTLASS_INCLUDE_DIR=/path/to/cutlass/include
```

The CMake target `hqsb_cutlass_gemm_bench` is configured automatically when
the headers are present; otherwise it is skipped (see
`ops/cuda/cutlass_gemm/CMakeLists.txt`).

## CUTLASS version note

The S04 comparison was validated against **CUTLASS 4.7.0** (main, fetched
2026-08-17). The default tensor-op configuration (`OpClassTensorOp`, `Sm80`)
targets the FP16 tensor-core MMA path on Jetson Orin (sm_87).
