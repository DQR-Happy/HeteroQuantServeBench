# L1 hardware and system experiments

The console's **优化 → 硬件与系统配置** entry has three independently addressable views:

- `/system/run`: select a verified local floating source, device, fixed prompts and A/B profiles.
- `/system/topology`: effective CPU affinity/cgroup quota, NUMA topology/policy availability, CUDA UUID, PCI proximity, power constraints and telemetry support.
- `/system/results?deployment=…&run=…`: sealed settings readback, per-trial outputs, samples and descriptive paired differences.

Use **设备与部署** to unload resident models before starting. An L1 job owns its private model processes; the ordinary single-request worker and L2–L9 plans remain separate. L1 currently measures native eager floating inference with no quantization or graph/profile substitution, so changing multiple layers cannot silently contaminate the pair. Existing machine selection pins all calls to the chosen SSH/HTTP node. Test installation remains unauthenticated on its loopback listener.

## Contracts and ownership

| Module | Input → output | Responsibility / language |
| --- | --- | --- |
| `core/system_execution.py` | bounded `SystemPlan` + hardware → admission / paired summary | Pure Python/Pydantic: profiles, experiment limits and comparison semantics. No hardware imports. |
| `hardware/system.py` | Linux procfs/sysfs or CUDA UUID → facts / NVML readback | Python OS + optional NVML adapter; no SDK initialization on import. NUMA uses libnuma through ctypes. |
| `backends/process_supervision.py`, `models/source_budget.py` | private process handle / local headers → reaping and byte budget | Shared Python resource primitives reused by L1/L9; backends do not import an L9 console controller. |
| `backends/system_probe.py` | inherited CUDA visibility → inventory JSON | Fresh Python process owns torch/NVML discovery. API does not initialize CUDA. |
| `console/system_jobs.py` | deployment + plan + idempotency key → durable run | Python control plane owns admission, shared service reservation, guardian supervision, cancellation and publication. |
| `backends/system_guard.py` | private job JSON → trial processes + rollback result | Python guardian owns device flock and a write-ahead power journal. Survives API exit, detects parent identity loss, kills worker before rollback. |
| `backends/system_worker.py` | one profile + source identity → effective settings, outputs and measurements | Fresh Python/PyTorch process applies affinity/NUMA before importing torch, then intra/inter-op threads before model load. Native PyTorch/CUDA performs compute. |
| `console/system_routes.py` | HTTP → job/inventory/sealed receipt | Transport only; no arbitrary commands, paths or driver functions from the client. |
| frontend `features/system` + `pages/System.tsx` | typed API → experiment/topology/results | TypeScript/React: independent profile editor and evidence view, shared URL identities and existing ActionButton disabled explanations. |

The existing native local-only manifest-verified loader/generator and process termination utilities are reused. There is no new C++/CUDA kernel: this layer changes launch/environment constraints, not operator math. Other console APIs remain backward compatible; the additive run kind is `system`.

## API

`GET /api/console/v1/system/catalog`, `/system/hardware`, `/system/jobs`.

`POST /system/jobs` with the existing `Idempotency-Key` header:

```json
{
  "deployment_id": "qwen-reference",
  "plan": {
    "device": 0,
    "precision": "float16",
    "baseline": {"intra_threads": 2, "interop_threads": 1},
    "candidate": {"cpu_ids": [0, 1], "intra_threads": 1, "interop_threads": 1},
    "prompts": ["Once upon a time, there was a little fox."],
    "max_output_tokens": 32,
    "rounds": 2,
    "timeout_s": 300
  }
}
```

Each profile may additionally set `numa_node` and `power_limit_w`. Null CPU IDs inherit effective affinity. Null power inherits **the pre-experiment cap**, even after the other profile changes it. NUMA null inherits the caller's memory policy; binding affects subsequent allocations, not migration of old pages. Device index follows inherited CUDA visibility and is verified using UUID inside the worker.

Use existing `GET /runs/{id}`, event stream and `POST /requests/{id}/cancel`. `GET /runs/{id}/system` requires terminal state and verifies the sealed hashes and stored receipt identity before returning evidence. SQLite retains complete runs; listings strip large trial/telemetry payloads before allocating objects. JSON export of a run remains available through the existing `/runs/{id}/export` route.

## Measurement and causal boundaries

- Two rounds per profile default to **A/B/B/A**; one round gives A/B. Up to four rounds, four inputs, 512 input tokens each and 128 output tokens. Fixed-length greedy decoding records all actual token IDs, input hashes and visible text. Source manifest, package versions, dtype, GPU UUID and source digest are recorded.
- Every trial starts a fresh process, verifies the source, loads, performs a two-token warmup, resets allocator peaks, then measures. This isolates write-once inter-op thread state. Cold load/warmup time is separately recorded and excluded from generation Token/s. Inputs are raw continuation text, not chat-templated prompts.
- Generation Token/s is total output tokens divided by summed generation-loop wall time, including first token. Mean first-token latency, per-request times, allocator allocated/reserved peaks, worker RSS, CPU-time/page-fault/context-switch deltas and `/proc/self/numa_maps` are direct framework/OS measurements.
- NVML samples every 250 ms: SM/memory clocks, board power, temperature, utilization, board memory and throttle-reason bitmask. Sampling includes two boundary readings and is bounded to 1,000 periodic samples. Device-wide counters do not isolate this process. A short run may have only boundary samples; telemetry overhead is present in both profiles and is not claimed to be zero.
- `energy_j` uses board energy-counter deltas, when exposed. Its explicit sample window surrounds the measured workload and excludes load/warmup, but includes sampler shutdown/boundary overhead. `energy_estimate_j` is separately labeled trapezoidal power integration. Neither is CPU/socket energy, per-process energy, nor an assertion of model-only energy. Missing readings remain null with a reason.
- Summaries report medians, candidate minus baseline, and percentage changes. Greedy exact agreement is a separate quality result. Differences are preserved rather than rejected as missing performance; this is not a representative task accuracy benchmark. Use L4 independent benchmarks for task/NLL quality.

## Global-state recovery

CPU affinity, thread pools and NUMA state are private to each child; exit releases them. GPU caps are board-wide. Before changing one, the guardian holds the HQSB device lease, requires an original readable cap and checks that NVML reports **no compute or graphics processes**. External applications do not honor HQSB's advisory flock, so the node still needs operator-exclusive use; it is not a multi-tenant GPU control service.

Before each attempted power mutation the guardian atomically fsyncs `.console/l1-power-journal.json` containing run ID, UUID and original cap. Settings are applied through NVML, then read back. Normal completion, timeout, cancellation and API-parent disappearance all terminate the private worker **before** restoring and reading back the original cap. Even a failed setter is followed by readback/rollback; an unchanged original cap can be confirmed without requiring another privileged write. No complete manifest is published before the guardian exits and confirms cleanup.

If restoration or guardian exit is unconfirmed, L1 is quarantined. A journal survives service restart and blocks future L1 jobs. Do not delete it to bypass recovery: inspect the named UUID, original cap and guardian PID on the selected execution node; stop residual private model processes, restore the original cap, confirm readback and guardian exit, then archive the journal and restart the service. PID alone is not evidence of ownership; inspect command line/start identity. SIGKILL of the guardian, driver loss or host failure cannot guarantee immediate rollback and are explicitly outside the successful-cleanup claim. Restarting the API does not automatically replay an experiment or reset a GPU.

## Supported versus unavailable

CPU affinity and both PyTorch pool counts are executable controls. NUMA bind is conditional on libnuma and permitted get/set_mempolicy; the 3090 container exposes topology but denies these syscalls. GPU selection follows available CUDA devices; power-cap support and driver permissions are checked separately. Install optional NVML binding on each node using its own environment (`nvidia-ml-py>=12,<14`, project extra `system`); no CUDA stack replacement is required. Missing NVML leaves CPU controls usable and marks unavailable GPU telemetry/power controls.

GPU/EMC clock locking, CPU governor changes, Jetson nvpmodel changes, NPU inference, heterogeneous graph partitioning, physical hardware substitution and interconnect reconfiguration are **not implemented** by this endpoint. Some frequency/governor facts are observable; this is not evidence that clock configuration was applied. GPU L2 hit rates, physical VRAM pages, bandwidth attribution and individual OS-page migrations cannot be inferred from these samples. Existing deep-observation modules remain separately available.

## Target acceptance

All commands run through `./scripts/remote_run.sh`; the Mac only edits/transfers files.

1. Run L1 contract, admission, output-comparison and rollback tests plus relevant console regressions.
2. Build TypeScript/disabled-action checks and run frontend tests on the target.
3. `python -m scripts.console.validate_l1` requires an idle console. It snapshots the resident recipe, unloads it, restarts the updated loopback service, runs Qwen3/Gemma3 real A/B/B/A trials, checks actual affinity/thread readback, exact greedy IDs, nonempty timing/memory/counter evidence, then tests real power-cap behavior and cancellation. It restores the saved model recipe in `finally` and verifies ordinary single-request inference. For a short integration check use `--models qwen-reference --rounds 1`; this is an A/B smoke test, not repeated performance evidence.
4. Browser acceptance checks the navigation, prerequisite explanations and sealed results, including mobile overflow.
5. Preserve raw evidence under `reports/console/l1-system/`; report driver/NUMA failures as unavailable or failed, not successful interventions. These checks establish the bounded control/observation loop, not a universal speedup.

Primary API reference: [NVIDIA NVML device commands](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceCommands.html). Power settings use milliwatts internally and require driver support/permission; the visible supported range alone does not prove write permission.
