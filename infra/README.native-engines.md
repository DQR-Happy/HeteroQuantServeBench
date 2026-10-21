# Native vLLM / SGLang baselines

For a step-by-step console walkthrough, see the [中文操作指南](README.native-engines.zh-CN.md).

HQSB owns an unmodified upstream server in a separate environment. Its scheduler,
KV manager, model loader and kernels remain upstream implementations. HQSB owns
the deployment lifecycle, public-argument translation, common HTTP workload and
evidence storage. The existing PyTorch backend remains available.

## Execution environments

Run installation, tests and inference through `scripts/remote_run.sh` on an
authorized execution host. The initial compatibility set is vLLM **0.11.0**,
SGLang **0.5.5**, torch **2.8.0**, Transformers **4.57.1**. These are reproducible
baseline versions, rather than a moving latest-version dependency. The loader
checks the installed engine version before startup.
Shared runtime pins are maintained in
`configs/environment/native-engines.constraints.txt`; each installed environment
also saves its complete package inventory.

```bash
./scripts/remote_run.sh bash scripts/setup/install_native_engines.sh
# For a separate data volume:
./scripts/remote_run.sh env HQSB_NATIVE_ENV_ROOT=/data/hqsb/engines \
  bash scripts/setup/install_native_engines.sh
```

The bootstrap uses separate venvs with system site packages to reuse the target's
torch/CUDA installation. pip installs engine-specific dependencies in each venv;
it can install the required torch there when the inherited version differs.
Check available disk before installation. Each venv has an environment freeze.
No weight download takes place: upstream servers receive a verified local model
directory with offline Hub access. Additional engine/kernel packages may need
their own installation or compilation for particular hardware.

Operator settings `native_engine_pythons` select the executable for each engine;
HTTP callers select an engine name and an existing source deployment.
`native_engine_cache_dir` relocates compilation caches. API authentication keeps
the existing `auth_required` setting; engine servers listen on loopback.

## User workflow

Open **优化 → 原生引擎调优** (`/engines`).

The workspace has three addressable pages: `/engines/configure` for deployment
and parameters, `/engines/benchmark` for workloads, and `/engines/records` for
outputs and comparisons. They share deployment/run identity and draft controls.
Existing `/engines?run=...` links open the corresponding records view.

1. Choose an existing local model and vLLM or SGLang, then register its native
   deployment. Registration reuses the source weight directory and manifest.
2. Unload the resident GPU model, select the native deployment and edit controls.
3. Load with the selected controls. Inspect the receipt and any startup errors.
4. Run the common workload. Each line is one input; rounds repeat the input list.
5. Unload, change controls or engine, reload, then replay the same workload.
6. Choose two benchmark runs to check comparability and inspect differences.
   For a tuning experiment, declare the changed controls (for example,
   `prefix_cache`). Artifact, workload and hardware identity stay fixed.
   The comparison reports exact-match score changes and a zero-drop quality gate.

Native variants are persisted in `.console/native-deployments.json` and restored
on restart. They continue to protect their source directory against deletion.
Use **移除部署引用** to retire an unloaded native variant. This releases its
source-artifact reference and preserves historical runs.

## Public contracts and layer mappings

| Interface | Responsibility |
| --- | --- |
| `GET /native-engines/catalog` | Engine versions, controls and HQSB layer mapping |
| `GET /optimization-catalog?provider=vllm` | Existing optimization catalogue interface with native controls |
| `POST /native-engines/deployments` | Register a variant of a verified source deployment |
| `POST /deployments/{id}/load` | Apply `engine_options` and start the owned upstream server |
| `POST /deployments/{id}/unload` | Drain and stop the owned server process group |
| `POST /requests` | Existing streaming inference task contract |
| `POST /native-engines/benchmarks` | Common concurrent HTTP workload, including warmup and SLOs |
| `GET /native-engines/deployments/{id}/receipt` | Current instance identity and execution evidence |
| `POST /native-engines/comparisons` | Check artifact/workload/precision compatibility and report measurements |

Layer 2 selects upstream Attention backends; layer 3 selects eager or engine
graph/compile execution; layer 4 selects computation/KV precision and compatible
quantized loaders; layer 6 selects prefix caching, page/block size, memory budget
and CPU weight offload; layer 8 selects batching/prefill budgets and scheduler
policy; layer 9 selects tensor and pipeline parallel degrees. Layer 7 currently
exposes seed and greedy generation. The catalogue describes integration, while
model/hardware acceptance is checked during actual startup. Multi-GPU settings
need matching hardware and independent validation.

The initial controls use eager execution with prefix caching disabled for a
conservative execution baseline. For tuned performance comparisons, enable graph
execution and caching where appropriate and retain the selected configuration.
These two configurations answer different experimental questions.

L1 system experiments and L10 application policies currently execute through the
PyTorch backend; native engines expose hardware receipts and streaming serving.
Their system-policy/application-policy adapters remain separate future work.
L5 accepts upstream-compatible model artifacts. HQSB PyTorch-specific patches,
precision artifacts, memory plans and speculative decoding plans have independent
formats/lifetimes; native equivalents require their own argument/adapter mapping.

## Evidence and measurement boundaries

Each load stores a versioned receipt and server log below
`.console/native-engines/{instance_id}`. It records versions, verified model
manifest, requested controls, launch argv, configuration hash and readiness.
SGLang also reports its resolved configuration through `/get_server_info`.
vLLM's initial proof level is owned process arguments, server logs and readiness;
the receipt distinguishes this from runtime kernel-dispatch proof.

Each benchmark is a durable `native_benchmark` run in the existing SQLite store.
Its receipt includes the full workload, workload hash, engine instance, actual
output/usage and errors for each request, HTTP first-content/end-to-end times,
native Prometheus snapshots, and process RSS/PSS/page-fault snapshots for the
upstream process tree. Counter snapshots are instance-wide, with separate host
system fields. They do not provide per-request GPU L2 or GPU physical-page data.

The load generator uses pooled HTTP connections and runs inside the Actor, with
concurrent requests sent directly to upstream. The API's serial task executor
therefore does not serialize the benchmark's individual requests. Warmup and
counter collection are excluded from the throughput window. Its denominator is
the first send through the last response, including drain. This is a finite
workload result; steady-state capacity requires longer load sweeps and repeats.

Closed-loop tests bound concurrency. Open-loop tests schedule deterministic
arrival intervals and record actual send lag; exceeding the configured lag limit
makes the offered-load comparison invalid. Failures and timeouts remain in the
record. Reported throughput uses actual usage, with missing usage producing an
unavailable throughput value. HTTP first-content/chunk timings have client scope.
P99 is omitted below 1,000 completed timed requests.

Optional reference outputs enable exact-match fixture quality and quality-gated
goodput. With no reference, outputs are saved for task evaluation and the receipt
states `outputs_recorded_not_scored`. This supports baseline collection; a short
smoke run establishes execution and measurement plumbing. Representative quality,
performance superiority and numerical accuracy each need their own experiments.

Comparison checks full workload identity, artifact manifest, precision, prefix
configuration, actual prompt-token counts, completion and arrival validity.
It reports a whole-engine result for one trial; repeat trials and alternate order
to estimate uncertainty. Cache state after warmup is recorded; repeat-prefix
and cold-cache experiments need explicitly separate workload protocols.

Target acceptance can be reproduced with:

```bash
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_native_engines.py \
  --source gemma3-1b --prefix-cache off --official-crosscheck
```

This qualifying baseline keeps the same precision and checks exact outputs, then
cross-checks each server using the upstream vLLM HTTP benchmark client (eight
requests, 16 output tokens each). `--prefix-cache both` additionally tests the
cache-enabled candidates with the same quality threshold; failed candidates keep
their outputs and metrics and cause acceptance to fail.

On the RTX 3090 acceptance target, Gemma 3 1B IT with BF16, eager execution and
prefix caching disabled passed the fixed arithmetic fixture in both engines.
The SGLang 0.5.5 candidate using Triton attention, page size 1 and prefix caching
enabled scored 50% on the same fixture (`17 + 25` produced `32`). Its comparison
is retained as a quality regression; this result applies to that tested stack
and workload. The default cache-disabled baseline passed the quality gate.
