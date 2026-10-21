# Nsight execution diagnostics

HQSB launches native NVIDIA Nsight Systems / Compute against an owned vLLM or
SGLang process tree. Each job replays the frozen workload and engine options of
a completed `native_benchmark` record. The existing PyTorch capture CLI remains
available through `scripts/observation/run_capture.py`.

## Workflow

1. Load a native deployment, run the common benchmark and keep its record.
2. In **原生引擎 → 运行与对照**, open **诊断此方案（nsys / ncu）**.
3. Explicitly unload resident GPU models in the configuration page. Diagnostics
   own a fresh engine instance and the device lease; the source record stays intact.
4. Choose **nsys** for a system timeline, or **ncu** for selected kernel counters.
   The hotspot table's **选择此 Kernel** action prepares an escaped NCU name filter
   and the corresponding source workload; review the selection before starting.
   Set a kernel regex, matching-launch skip/count and a counter section group.
5. Start the job. Loading and the source workload's explicit warmup happen before
   the GPU workers call `cudaProfilerStart`. Collection ends after the measured
   requests drain, through `cudaProfilerStop`.
6. Inspect the timeline/counters and output consistency. Choose another completed
   diagnostic of the same tool for mechanism comparison. Re-run the ordinary
   benchmark after tuning to measure service performance and quality.

The browser route is `/diagnostics?source=<source_run_id>&run=<diagnostic_run_id>`.
The UI renders a bounded GPU timeline, top kernels, activity-interval unions,
copy statistics, CUDA host API aggregates and per-launch native NCU metrics with
their original units. Downloads include native reports and structured exports.

## Modules and ownership

| Module | Input → output | Ownership |
| --- | --- | --- |
| `core/diagnostics.py` | typed plan → bounded profiler argv | selection, budgets, collection policy; no CUDA imports |
| `console/diagnostic_jobs.py` | source record + plan → durable diagnostic run and artifacts | admission, exclusive execution, cancellation, quota, lineage, sealing |
| `observation/diagnostic_target.py` | server-owned input file → per-replay outputs/engine receipts | fresh native instance, warmup, capture window, teardown |
| `observation/vllm_profile_worker.py` | upstream profile RPC → CUDA profiler API | vLLM 0.11 V1 diagnostic-only worker subclass |
| `observation/nsight_reports.py` | SQLite/CSV → timeline/counter panels | schema interpretation, units, metric scope |
| `observation/diagnostic_comparison.py` | two sealed receipts → scope checks and mechanism deltas | comparison semantics |
| `console/diagnostic_routes.py` | typed HTTP request → jobs, comparisons, artifact downloads | API transport |

These modules use Python. NVIDIA's native CLI performs collection; the upstream
engine's CUDA/Triton kernels perform inference. The adapter adds no per-token
Python hooks. It synchronizes the device only at the two diagnostic boundaries.

SGLang 0.5.5 uses its public `CUDA_PROFILER` profiling activity. vLLM 0.11.0 uses
its public worker-class extension and profile endpoints. The subclass preserves
the upstream worker's model execution, creates no torch profiler, and replaces
only profile start/stop. Its class identity, source hashes and launch arguments
are recorded. Normal deployments use the upstream worker directly.

## API

| Endpoint | Behavior |
| --- | --- |
| `GET /diagnostics/catalog` | tools, versions, last actual collection result and prerequisites |
| `POST /diagnostics/jobs` | create with `Idempotency-Key`; `source_run_id`, `tool`, optional selection and budgets |
| `GET /diagnostics/jobs` | bounded history, detailed payloads projected out |
| `GET /runs/{id}` | full durable receipt, progress and errors |
| `POST /requests/{id}/cancel` | existing common cancellation protocol |
| `POST /diagnostics/comparisons` | `baseline_run_id`, `candidate_run_id`; collection scope and parameter changes |
| `GET /diagnostics/jobs/{id}/artifacts/{path}` | owned reports, logs and exports; private launch input stays internal |

Requests provide selections, rather than executable paths or shell commands.
Executable paths come from operator configuration. Artifact paths are resolved
inside the selected run directory, including symlink confinement.

## Configuration and installation

`Settings.diagnostics_dir` chooses report storage; `diagnostic_tools` optionally
specifies absolute CLI paths. Otherwise PATH and standard NVIDIA installations
are searched. Each tool's actual `--version` is recorded. The 3090 configuration
uses data-disk storage and the independently installed NCU 2025.1.1 CLI.

For the pinned x86-64 Linux CLI-only NSYS 2025.3.1 package:

```bash
./scripts/remote_run.sh env HQSB_NSIGHT_ROOT=/data/hqsb/tools \
  bash scripts/setup/install_nsight.sh
```

The script verifies NVIDIA's published package SHA256 and extracts the CLI into
that directory. Configure the resulting `target-linux-x64/nsys` path in the node
YAML. Other architectures use their matching NVIDIA packages and target checks.

## Measurement contract

- The source artifact manifest, workload hash, engine options, upstream versions,
  environment inventory and hardware identity accompany each diagnostic.
- Each NCU application replay starts a fresh native server and runs the same
  bounded workload, including warmup. Every attempt keeps its outputs and receipt.
  Application-managed prefix priming is therefore reproducible when warmup defines it.
- Native counter/procfs snapshots remain instance-wide. HTTP request timestamps,
  NSYS relative nanoseconds and GPU event correlations retain separate time domains.
- NSYS GPU busy intervals use an interval union **per device** across kernels,
  copies and memset. Kernel summed durations can overlap. Gaps cover the captured
  first-to-last GPU activity span and require further evidence for causal attribution.
- NCU uses application replay, `cache-control=none`, `clock-control=none`, a bounded
  matching-launch count, and selected sections. Percentages keep their native units.
  Missing/unavailable metrics remain null with the raw tool value.
- Diagnostic duration and throughput include instrumentation effects. Formal
  service performance comes from the linked common benchmark. Full model quality
  continues through task evaluations; output consistency is recorded separately.
- The initial unified API covers owned native-engine benchmarks on one host.
  Automatic request → batch → forward → kernel semantic attribution, OS physical
  page layouts and native KV block maps require their corresponding event sources.
  Raw NSYS correlation/context/stream fields and full SQLite remain available.
- Multi-rank counter pairing needs rank identities. Initial NCU comparisons pair
  kernel name/grid/block and occurrence only for single-rank sources.

## Permissions, failure and recovery

The first real 3090 container counter probe returned `ERR_NVGPUCTRPERM` with
`RmProfilingAdminOnly=1`. NSYS CUDA tracing works on this node. NCU jobs preserve
the actual permission error, release engine ownership and report counters as
unavailable. Tool installation and counter authorization have separate states.

The host administrator can grant profiling access or launch the container with
the capabilities permitted by the host policy. For the current R570 driver,
follow [NVIDIA's counter permission guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
Changing a modprobe setting inside an already running restricted container does
not grant a host driver capability. The HQSB service does not reload the driver.

Cancellation, timeout and report budget exhaustion stop the owned process tree.
The supervisor's owner pipe also closes if the API exits. Source records and
partial artifacts remain available. An unconfirmed cleanup retains device
exclusion for operator inspection. Report writing errors still finalize the
SQLite task and release a confirmed-clean device.

Target acceptance uses:

```bash
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_diagnostics.py \
  --source <completed-native-run> --tool nsys
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_diagnostics.py \
  --source <completed-native-run> --tool ncu
```

On a known restricted node, `--expect-counter-permission-error` verifies the
error/cleanup path and records hardware counters as unavailable. It does not
count as successful counter collection.
