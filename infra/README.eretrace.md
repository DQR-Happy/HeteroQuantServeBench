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

The warp capture reports warp/SM behavior; the memory mode below provides DMAT.
Cache-hit rates come from ncu; CPU pages come from procfs. Tensor/KV address
attribution requires allocation-lifecycle evidence. Whole-model optimization decisions
continue to use normal benchmark performance and quality alongside diagnostics.

## Real memory access / DMAT

The current dual-clock `HQDMAT2` implementation, tensor mapping, common-scale
comparison APIs and Attention/model acceptance are documented in
[README.dmat.md](README.dmat.md). New collection requires a runtime advertising
`dmat-dual-clock`; old raw files remain readable.

### Archived HQDMAT1 format

The following describes the initial globaltimer-only implementation and its
saved artifacts. Its frame limits, unsupported async copies and saved-grid zoom
refer specifically to `HQDMAT1`; use the current document for new captures.

In **执行诊断**, select **ERETrace → 真实访存 / DMAT**. Sources can be an
existing single-request Eager native benchmark or a completed PyTorch greedy
inference with saved input and a verified model manifest. Unload resident models
before submitting. The replay reloads the saved model/profile/precision and opens
the capture window after warmup. PyTorch sources with compiler, memory, decoding
or structure transformations need their corresponding replay adapter; admission
explains which source setting needs adaptation.

The runtime must be rebuilt with the current installer. Its identity advertises
`capabilities: [warp, memory]`; existing warp-only installations remain usable for
warp jobs. `preload.so` retains the selective Triton bridge for native engines;
`preload-pytorch.so` is built from the private artifact's original loader and is
used only by the owned PyTorch DMAT child. Ordinary service workers keep their
normal execution environment.

DMAT uses a separate synchronous, single-execution collector and raw format:

```
magic HQDMAT1\n
header <8I: grid[3], block[3], shared_bytes, per_thread_capacity
counts: thread_count * u64 attempted events
events: thread_count * capacity * <QQIIII
        GPU globaltimer ns, virtual_address, site, thread, predicate_mask, width
```

Enabled threads record immediately before supported explicit `ld.global` and
`st.global` instructions. Predicated accesses retain the enabled-lane mask;
vector operations retain their instruction width. Generic addressing, atomics,
async copies/TMA and called PTX functions require separate instrumentation
adapters. Their selected launches retain compiler diagnostics and coverage
failures. Closed library calls outside the selected loader path remain outside
the capture denominator. This mode does not claim whole-model instruction
coverage. It has no counting replay and never re-executes stateful kernels to
size a buffer.

Each launch reserves at most 16 MiB including count buffers, with capacity
1–512 events per thread and 1–32 selected launches. Zeroing, launch and completion
are ordered on the submitting stream, including non-default streams. Overflow
keeps the original inference result and captured events, reports exact dropped
counts, and marks the diagnostic partial/failed. Model output comparison and
source workload/environment lineage are recorded separately. Selected PTX and
instrumented PTX hashes, assembler identity, instruction sites, raw `.dmat`,
`panel.json` and downloadable SVG enter the existing sealed artifact manifest.
The assembler may be the matching Triton-bundled version when the system
assembler cannot accept the retained PTX version; PTX versions are preserved.

The heatmap uses GPU-wide `globaltimer` timestamps, relative to the first observed
access in each launch. Nanoseconds describe the timer unit, not its resolution.
Address labels are exact hexadecimal strings; absolute u64 values never pass
through JavaScript floating point. Color counts enabled-thread instruction
events grouped by their starting virtual address. Requested address bins are
4/16/64 KiB, coarsened explicitly if needed to keep at most 256 observed rows.
Address gaps are collapsed and the actual bin width is displayed. Time is
aggregated into 16–512 intervals, coarsened explicitly when needed to bound each
plot to 8,192 cells. Requested and actual bin counts are saved. Vector instruction bytes and read/write counts
are separate values; instruction bytes are distinct from DRAM bus traffic.
Time/address zoom and read/write selection operate on these saved aggregates.
Raw events remain available for independent analysis at their original precision.

DMAT shows actual software-visible memory access. Physical page placement,
faults/migrations, cache hits and tensor/KV ownership need their respective
driver, counter and allocation-lifecycle evidence. Separate runs can use different
virtual addresses, so address-aligned diffs require object attribution. Profiling
perturbation remains part of the diagnostic interpretation.

Validate on the execution node:

```bash
./scripts/remote_run.sh .venv-console/bin/python -m pytest tests/unit/console/test_dmat.py -q
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_dmat.py \
  --runtime /path/to/runtime --output reports/console/dmat/address-oracle
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_dmat.py \
  --runtime /path/to/runtime --output reports/console/dmat/overflow-oracle --capacity 1
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_eretrace.py \
  --mode memory --source run_<saved-source> --kernel <substring> --launch-count 1
```

The GPU oracle checks the complete address/read/write multiset against the
actual input/output allocation addresses, numerical output equality, masked tail
and non-default-stream execution. The low-capacity control requires partial
coverage with retained real data. It cannot pass by replacing measured events
with inferred tensor positions.

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
