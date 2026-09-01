# Model-facing L2 kernels

Qwen3-1.7B FP16 operator versions are selected through `kernel_profile`, fixed
for one worker/model epoch. Default: `framework-native`. Other supported IDs
and exact variants live in `hqsb/core/kernel_profiles.py`.

- Native math and C ABI: `ops/cuda/model_kernels/` (CUDA C++17 / cuBLASLt).
- Tensor guards and exact artifact checks: `hqsb/integration/model_native.py`.
- Single torch schema owner: `hqsb/integration/torch_ops.py`, `hqsb::l2_*` IDs.
- Model semantic binding: `qwen3.py`; no global HF monkeypatch or weight copies.
- Lifetime, restoration and bounded site/phase counters: `../kernel_session.py`.
- API: `GET /api/console/v1/kernels/profiles`; optional load body
  `{"kernel_profile":"qwen3-local-fusions-v1"}`. Unload before changing profile.
- CLI: `scripts/observation/run_model_observation.py --kernel-profile ...
  --kernel-manifest build/l2/model-kernels.json`.

Build on the target using `scripts/observation/build_model_kernels.py` through
`./scripts/remote_run.sh`. Administrator config example:
`configs/console/l2-jetson.example.yaml`. Browsers cannot select a shared library
path. Native artifact, source, ABI, SDK and SM must match the manifest.

Each request saves `kernel-plan.json`, `kernel-bindings.json` and
`kernel-summary.json`, covered by the existing experiment manifest. Counts are
successful **logical site** dispatches, not GPU launches or instruction proof.
GQA-repeat bytes are tensor-shape accounting, not measured DRAM traffic.
cuBLASLt records actual algorithm IDs and caps its plan cache at 256 entries;
zero workspace means caller-provided workspace, not vendor internal storage.

Quality supports separate `full_prefill` and `incremental_decode` protocols.
`validate_model_kernels.py` preserves the existing zero-error comparison gate;
nonzero errors are reported, not silently accepted. No speedup or adoption is
inferred from dispatch success. Formal performance requires registered paired
runs with `run_experiment.py`; example: `configs/kernels/norm-ablation.example.json`.

Initial scope: eval/inference only, dense source-pinned Qwen3, CUDA:0 FP16.
Flash SDPA explicitly validates B=1/unpadded input once per model forward (a
synchronizing guard whose cost is included), supports square prefill and
single-query decode. It never falls back to math. Math SDPA supports explicit
masks and offset causal GQA, and defines fully hidden bool/-inf rows as zero.
Unsupported providers/versions/shapes raise errors. External flash-attn,
CUTLASS model execution, QK-Norm+RoPE fusion, weight packing, quantized Linear,
MoE, training, graph capture, and other model/SDK revisions remain outside
this integration.

Detailed design and validation: `docs/architecture/HQSB_L2_接入与验证_20261015.md`.
Raw runs stay in the target evidence directory; the repository does not embed
model weights or access tokens.
