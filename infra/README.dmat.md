# DMAT: measured memory structure and Attention comparisons

## Observation contract

Neutrino §5, Figure 1/11 uses **thread-entry normalized clock64**. Figure 15
uses the **GPU-local globaltimer**. HQSB records both in a single execution:
`HQDMAT2\n`, eight little-endian uint32 launch fields, one attempted-event uint64
per thread, then a fixed-capacity array per thread. Each event is 48 bytes:
`globaltimer:u64, clock64:u64, entry_clock64:u64, VA:u64, site:u32,
thread:u32, active_predicate_mask:u32, accessed_bytes:u32`.

Entry clock is captured before instrumentation buffer indexing. Normalization
subtracts that actual thread entry, including the interval before its first
memory access. Global time subtracts the launch's first observed memory event.
Cycles and nanoseconds remain distinct units; timer units do not establish timer
resolution. The selected view, quantum and window are retained in each grid.
`HQDMAT1` remains readable in global-time mode; it has no entry clock to recover.

Each enabled thread instruction contributes one density event at its starting
virtual address, including vector instructions. Read/write and instruction-width
bytes are separate counts. These are software-visible instruction observations;
DRAM transactions, cache hits and physical/OS page placement have separate data
sources. Address grains 4/16/64 KiB are explicit software bins. Observed gaps are
collapsed. New grids retain the requested grain, with an explicit view-budget
error above 1,024 observed tensor/page rows instead of silently changing grain.

Supported PTX sites follow the original probe selector: `ld.global`,
`st.global`, `cp.async.ca.shared.global`, `cp.async.cg.shared.global`, including
predicates, vectors, cache hints and cp.async source-size/zero fill. Commit/wait
instructions retain their execution. Zero-filled lanes produce no source-read
event. Atomics, generic-address memory, TMA and retained device-call bodies need
separate adapters; failures retain the original reason and coverage. Jetson
acceptance covers ordinary launches, one producer/context and a nondefault
stream. At most 2,048 events/thread and 256 MiB raw buffers per launch are allowed;
attempted counts expose overflow. Job disk/time budgets still apply.

PTX reassembly and record stores perturb execution. Thread-normalized plots can
reveal algorithm structure, but holes and density require workload and probe
coverage context. Performance judgments use separate runs with no probe/preload.

## Tensor ownership and grouping

`memory_context({'Q': q, 'K': k, 'V': v, 'O': out}, ...)` snapshots actual live
pointers, byte envelopes, logical bytes, dtype, shape, strides, layer, phase and
Attention semantics into the selected launch record. No GPU tensor data is
copied for attribution. Context restoration is nested and producer-owned; native
collection synchronizes its own selected stream before consuming buffers.

A containing tensor view owns the event; the narrowest view wins and aliases
remain in the context. Strided views are labelled byte envelopes. Sparse strides
may enclose gaps; arbitrary overlapping views need more precise adapters before
semantic conclusions. Unknown ownership is retained and counted. Q/K/V/O colors
are fixed. Density levels use each tensor's nonzero cell 20/40/60/80 percentiles,
with the original callback's strict-threshold rule; tooltip keeps actual counts.

The integrated Triton Attention path has exact Q/K/V/O contexts, layer identity
and prefill/decode phase. Other kernels can still be collected; their mapping
coverage is explicit. General automatic ownership for every vendor kernel is a
separate integration task.

## APIs and interface

- `POST /diagnostics/jobs`: `ere_mode=memory`, `ere_codec=raw`,
  `memory_clock=thread_entry_clock64|GPU_globaltimer`, capacity/grain/bins.
- `GET /diagnostics/jobs/{run}/memory/{launch}`: same clock choices, `page_bytes`,
  `time_bins`, optional `start`/`end`. It verifies sealed raw artifacts and
  recomputes the grid without rewriting the original trace/manifest.
- `POST /diagnostics/comparisons`: source model, workload, environment and capture
  contracts are checked first. Memory kernel selectors may differ. Tensor shape,
  layer, phase, causality and scale must agree; complete mapped events are
  required. Raw frames are rebinned with a common time window/quantum and pooled
  per-tensor color thresholds. Page positions align by tensor role and relative
  page offset, with original allocation alignment retained in metadata.

On **执行诊断**, choose a saved PyTorch generation → ERETrace → 真实访存 / DMAT.
Unload the resident model using the linked deployment page, then start the owned
replay. Choose time mode/address grain in the result. The embedded overview lets
the mouse wheel scroll the page. **聚焦查看** opens a viewport-sized dialog with
independent time/address zoom and pan controls, draggable content and axis sliders.
The wheel defaults to address panning; its selector changes the target/action.
Shift-wheel pans time. Focus the canvas for arrow-key panning, +/- time zoom,
Home to fit the current grid, and Escape to return with focus restored.
Zooming preserves the measured bins and density scale. In a single-run view,
**按此时间范围重新分箱** explicitly requests finer aggregation from sealed raw
events, and **完整时间范围** restores the source window. Viewport readouts include
the current time range, address rows and bin sizes. Comparison focus views retain
the common bins and colors of the pair. Select another diagnostic record for side-by-side
common-scale grids and per-tensor event/instruction-byte changes.

On the existing model/optimization configuration page choose:

- `qwen3-attention-triton-n32-v1`: online softmax, K/V tile 32;
- `qwen3-attention-triton-n64-v1`: same semantics, K/V tile 64.

Load/reload, save input and generate, then select that source on diagnostics.
These are inspectable HQSB research implementations, FP16 batch-1 unpadded causal
Qwen3-1.7B (source-pinned existing adapter); head grouping is direct GQA, with
FP32 softmax/accumulation. They do not claim to be official FlashAttention-v1/v2
or universally faster. Arbitrary padding/masks, training and other model adapters
retain their existing paths. The profile plan records Triton version/source hash.

## Responsibilities

| Module | Language | Input → output / owner |
| --- | --- | --- |
| `dmat_instrument.py`, `dmat_probe.py` | Python/PTX | selected PTX → dual-clock probes, sites and assembler identity |
| `native/eretrace/hqsb_memory.h` | C/CUDA Driver API | selected launch → bounded single-execution raw buffers |
| `dmat_context.py` | Python/ctypes | live tensor views → producer-local launch context |
| `dmat_reports.py` | Python/NumPy | validated legacy/v2 bytes + lineage → exact events and saved view |
| `dmat_grid.py` | Python/NumPy | exact events + clock/window/tensors → density grid and attribution counts |
| `dmat_compare.py` | Python/NumPy | compatible raw pair → common-scale grids and object changes |
| `triton_attention.py` | Python/Triton | Q/K/V + tile/causal contract → output and exact context |
| `diagnostic_jobs.py`, `diagnostic_routes.py` | Python | source/config + sealed artifacts → lifecycle, API and derived views |
| `DMATPanel.tsx` | TypeScript | diagnostic identity and view settings → raw rebin requests and tensor metadata |
| `DMATChart.tsx`, `dmatViewport.ts` | TypeScript | measured grid → focused viewport, bounded zoom/pan and explicit rebin selection |
| `DMATCanvas.tsx` | TypeScript/ECharts | measured grid and viewport → overview/focused heatmap with identical density semantics |

## Target verification

All commands go through `./scripts/remote_run.sh` on Jetson. The installer uses an
independent runtime directory; the original private BITS source stays unchanged.

1. `validate_dmat.py`: exact address/read/write multiset, masked tail and stream;
   `--capacity 1` requires overflow/partial evidence.
2. `validate_dmat_async.py`: real cp.async, 17 source reads + 32 writes, 15 zero
   fill lanes, matching addresses and numerical output.
3. `validate_dmat_attention.py`: seven numerical cases per variant (causal,
   noncausal, GQA, strided inputs, tails and cached multi-query/decode); FP32
   reference with fixed `rtol=.02, atol=.003`. Real causal B1 H2 S512 D64 pair
   requires all Q/K/V/O events mapped, full coverage and expected instruction
   bytes from an independent causal-tile formula. Unprobed timing is separate.
4. `validate_dmat_paper.py`: export the exact normalized events to Neutrino's
   original thread-map ABI, run its unchanged C++ sparsifier, compare every exact
   `(cycle, 64-KiB page, density)` entry. This checks representation semantics;
   device scheduling and workload-dependent images can vary.
5. `validate_dmat_model.py`: public API load/reload and saved Qwen3 inference,
   both tile profiles and framework baseline; independently replay prefill/decode,
   require source output/token/environment matches, real mapped events, both
   clock views, common-scale mechanism comparison and cleanup.
6. `web/console/tests/dmat.spec.ts`: actual browser controls, clock/grain/zoom,
   sealed artifacts and before/after view.

Raw capture files and receipts remain immutable. Rebinning/plots are derivatives.
Evidence is saved under `reports/console/dmat-v2/` on the execution host and the
service's `.console/diagnostics/run_*/` artifact stores. Keep numerical correctness,
quality parity for the tested prompt, mechanism changes and unprobed timing as
separate claims. A short fixed prompt does not establish broad model quality.
