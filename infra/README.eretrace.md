# ERETrace diagnostics

HQSB uses the private BITS ERETrace implementation as an optional CUDA diagnostic
runtime. The control server stays in Python; the original C/CUDA collector owns
probe submission, encoding, device/host arenas, asynchronous output and recovery.
HQSB owns bounded selection, immutable workload lineage, decoding, comparison,
and the diagnostic UI.

## Install on an execution node

Transfer `BITS/release/neutrino-artifact` to a private source directory on the
execution node. This source stays outside the HQSB Git tree. Original notices
and a per-file source identity are retained in the installed runtime. Existing
BITS sources and paper evidence are read-only throughout installation.

From the HQSB editing checkout, use the project execution proxy:

```bash
./scripts/remote_run.sh .venv-console/bin/python -m pip install -e '.[eretrace]'
./scripts/remote_run.sh .venv-console/bin/python scripts/setup/install_eretrace.py \
  --source /private/eretrace-source --dest /private/eretrace-runtime \
  --cuda-home /usr/local/cuda --arch 86
```

The destination must be empty. Each rebuilt runtime gets its own directory and
`hqsb-runtime.json`: source SHA-256, HQSB adapter/installer identity, driver,
CUDA compiler, architecture and installed file hashes. The builder installs
`toml==0.10.2` into that runtime's `python/` directory. Model and engine
environments keep their existing packages. Configure the server with:

```yaml
eretrace_runtime: /private/eretrace-runtime
```

Restart an idle console service after configuration changes. Discovery checks
runtime file identities; admission independently checks source workload,
model manifest, device ownership and disk budget. A changed driver requires
a fresh runtime build. The driver hook is enabled only for ERE diagnostic child
processes; normal inference and Nsight jobs retain their existing launch paths.

## Use

1. Under **原生引擎**, load a model with Eager enabled, TP=1 and PP=1.
2. Run a public benchmark with request concurrency=1. Save this baseline and
   unload the resident deployment.
3. In **执行诊断**, choose that source and **ERETrace · warp / SM 观测**.
4. Enter a kernel-name substring, launch skip and maximum attempted launches
   (1–32). `_fwd_kernel` selects compatible Triton attention kernels; the exact
   kernels available depend on the source engine and its attention backend.
5. Choose reference encoding or raw-record control. Enable same-execution
   verification when checking the codec; the extra raw copies are diagnostic
   work and enter capture overhead.
6. Inspect observed launches, SM record counts, per-launch elapsed-cycle
   percentiles, encoding/recovery counts, buffers and output equality. Download
   original `.nts` files and the sealed manifest for independent analysis.

The HTTP plan uses `tool=eretrace`, `ere_kernel`, `ere_codec=reference|raw`,
`ere_verify`, `launch_skip` and `launch_count` on the existing
`POST /api/console/v1/diagnostics/jobs` route. Idempotency and cancellation use
the existing console protocol. Comparisons use the existing diagnostics route.

## Observation contract

- Capture window: measured requests after model load and explicit warmup.
- Frontend: compatible Triton ordinary CUDA launches routed through a selective
  loader bridge. cuBLAS and ordinary PyTorch calls keep the installed driver.
  A single submitting thread and CUDA context own the selected capture.
- PTX: embedded PTX, or retained Triton PTX located through an exact matching
  CUBIN SHA-256. The selected PTX hash and source enter the kernel artifact.
  PTX/compiler failures retain original execution and appear as failed probe
  attempts. Empty captures fail with their evidence preserved.
- Graph and multi-rank sources need their own adaptation; admission guides the
  user to create an Eager, single-rank source without changing saved options.
- Record: lane-0 entry `u64`, elapsed-to-own-exit `u32`, SM ID `u32` in CTA/warp
  order. Zero slots are reported separately. Elapsed is in cycles; each SM has
  its own `clock64` domain. Statistics aggregate durations, not global starts.
- Coverage: routed launches in the active bounded-selection interval, matched launches, attempts, successful
  probes, probe failures and other-thread launches. Closed library launches
  outside the hook are outside this denominator. The recorded scope is
  selective, and request attribution remains the engine capture window.
- Window control: the upstream worker profile routes open collection after
  warmup and drain it after measured requests. vLLM uses its worker-class
  extension; a diagnostic-only SGLang entry adapts its profiler mixin in spawned
  workers. A maximum attempt count bounds captured launches; fewer matches are
  closed normally at the worker stop. Cold probe compilation enters diagnostic
  overhead. The native adapter also supports a file-gated bounded window for
  standalone clients; console model jobs use explicit worker controls.
- Storage: the streaming reader validates magic, shape, tag, expected payload
  size, complete frames and frame-to-kernel lineage. One Map is bounded to
  16 MiB raw; launch count is bounded to 32. Device raw/output arenas are
  32 MiB each, pinned host arena 64 MiB, maximum segment 16 MiB, plus collector
  metadata, recovery staging and original model allocations. The enclosing
  diagnostic artifact quota, timeout and process supervisor remain active.
- Accepted reference payloads and raw recovery reconstruct captured integers
  exactly. Same-execution verification is recorded separately from model output
  equality. Output equality compares actual texts against the source workload.
- Completion includes collector drain and file close. Host filesystem persistence
  follows the existing diagnostic artifact protocol; file close is the trace
  writer's endpoint.

The current capture reports warp/SM behavior. Cache-hit rates come from ncu;
CPU pages come from procfs. Variable-count memory-event tracing and tensor/KV
address attribution are separate extensions. Whole-model optimization decisions
continue to use normal benchmark performance and quality alongside diagnostics.

## Modules and interfaces

| Module | Language | Input → output / responsibility |
| --- | --- | --- |
| `core/diagnostics.py` | Python | Typed plan → bounded diagnostic configuration |
| `observation/eretrace_tools.py` | Python | Runtime + plan → verified inventory and child environment |
| `native/eretrace/hqsb_capture.h` | C | Explicit window + ordinary launch → selected capture and coverage sidecars |
| `native/eretrace/hqsb_preload.c` | C | Triton caller → local hook; runtime/library caller → installed driver |
| Private ERE collector/encoder | C/CUDA | Raw warp Maps → encoded/raw recovery stream and collector counters |
| `observation/eretrace_ptx.py` | Python | CUBIN bytes + private cache → matching PTX and provenance |
| `observation/eretrace_reports.py` | Python/NumPy | Bounded stream + launch sidecars → exact decoded records and aggregates |
| `console/diagnostic_jobs.py` | Python | Source run + plan → owned replay, cancellation and sealed receipt |
| `observation/sglang_profile_server.py` | Python | Upstream worker profiler routes → explicit ERE start/drain; spawn-safe entry |
| `observation/diagnostic_comparison.py` | Python | Two receipts → comparability checks and cycle/record changes |
| `pages/ERETracePanel.tsx` | TypeScript/React | Aggregates → distribution, launch details and raw-record downloads |

There is no Python callback for each warp record on the GPU launch path.

## Verification

```bash
./scripts/remote_run.sh .venv-console/bin/python -m pytest \
  tests/unit/console/test_eretrace.py tests/unit/console/test_diagnostics.py \
  --basetemp=/root/autodl-tmp/hqsb/eretrace-validation/pytest-eretrace -q
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_eretrace.py --verify
```

The acceptance script creates a concurrency=1 baseline, unloads it and submits
an ERE capture through the public API. It checks workload/environment identity,
actual warp records, complete cleanup and source-output equality, and writes
receipts to `reports/console/eretrace/`. A source can be reused with `--source`;
`--codec raw` provides the collection control. Failed receipts remain available.
The example keeps test temporary files on the 3090 data disk; select the
corresponding data-disk path on other nodes. `--launch-skip` and `--launch-count`
exercise shorter selections, and `--cancel` verifies loading-time cleanup.
