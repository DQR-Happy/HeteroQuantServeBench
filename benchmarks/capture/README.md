# CPU capture-path benchmark

This package evaluates the actual `EvidenceWriter.offer()` replacement boundary,
not an LLM inference engine. It has no torch/CUDA dependency. Run it on an
authorized Linux target through `scripts/remote_run.sh`; the HQSB Mac remains an
editing client. Upstream versions are fixed in `baselines.lock.json`.

`spec/` preserves the original design. **`cpu-profile.json` describes what is
implemented and what is not claimed.** The CPU pilot is not full acceptance of
all 31 design families or all 50 proposed metrics.

## Boundaries

| Track | Input and measured work | Required interpretation |
|---|---|---|
| PYTHON | Prepared UTF-8 body, Python API/binding, native queue/codec/file work | Includes FFI and copies; excludes preparing the body from timed offer |
| PYTHON_STRUCTURED | dict/list body; adapters encode when their library requires bytes | Required conversion belongs inside timed offer |
| NATIVE_CORE | C++ producer, real NanoLog C++17 runtime and decompressor | Compare with native control; not with Python call ns |
| SERVICE | OTLP HTTP through real OTAP engine to independent decoding/fsync sink | Includes batching/codec/loopback; one sequential sender is not engine saturation |
| MooBench | Upstream recursive Python SuT | Method instrumentation overhead, not event-path throughput |

Every run gets a fresh process and UUID. The input seed, logical content, planned
arrivals and resource placement match across comparable adapters. Native binary
records and decoded semantic records are verified, rather than counting successful
calls as delivered events. NanoLog/Hindsight/OTel have void submit APIs; their
`accepted` field is unavailable. `submitted_unknown` is not an acknowledgement.

CPU quotas, memory policy and retention still matter. Stock buffering and overload
policies differ: queue size is not a byte budget, blocking is not dropping, and a
full-stream Hindsight C sink is not its distributed retroactive sampler. Results
are as-configured tradeoffs, not an overall winner. Public HQSB reader/publish
limits remain 100k records/512MiB; larger CORE files use an independent oracle.

## Setup and execution

Copy this repository's benchmark files and `hqsb/observation` into an isolated
target checkout. Do not reuse a Jetson virtual environment or compiled objects.
From the local project root (replace the remote directory with the prepared one):

```bash
HQSB_REMOTE_HOST=cpu-hqsb HQSB_REMOTE_DIR=/path/to/HQSB \
  ./scripts/remote_run.sh bash benchmarks/capture/bootstrap.sh /path/to/work-root
```

The bootstrap refuses to reset an existing checkout with an unexpected commit.
When GitLab cannot be reached from the target, fetch the exact official commit
through an available Git transport, transfer the source, and verify its HEAD;
do not substitute a similarly named project.

OTAP uses its own isolated Rust toolchain/cache. The required build at the pinned
revision is:

```bash
cargo +1.98.1 build --locked --release --bin df_engine \
  --no-default-features --features otlp,otap,crypto-ring
```

Run that command through the remote proxy from `upstream/otel-arrow/rust/otap-dataflow`.
The bare no-default-features binary lacks OTLP nodes/crypto provider. Validate the
actual pipeline with `df_engine -c config.yaml --validate-and-exit` before timing.
The executed builds use optimization 3 and `CARGO_PROFILE_RELEASE_DEBUG=0`.

The following commands are **remote** commands; invoke them through the proxy
with the isolated checkout as `HQSB_REMOTE_DIR` and its venv Python:

```bash
python -m pytest tests/unit/capture_benchmark -q
python -m benchmarks.capture.run --tier smoke --passes throughput,latency \
  --upstream /path/to/work-root/upstream --output /path/to/new-smoke
python -m benchmarks.capture.run --tier screening --wait-for-lock \
  --upstream /path/to/work-root/upstream --output /path/to/new-screening
python -m benchmarks.capture.native_run --upstream /path/to/work-root/upstream \
  --output /path/to/new-native
python -m benchmarks.capture.service_run --upstream /path/to/work-root/upstream \
  --output /path/to/new-service
python -m benchmarks.capture.reproduce_moobench --upstream /path/to/work-root/upstream \
  --output /path/to/new-moobench
python -m benchmarks.capture.diagnose --output /path/to/new-probes
```

All output directories must be new. Campaigns use a shared measurement lock;
`--wait-for-lock` queues the main runner. Never run an adversarial probe against a
live HQSB directory. Fault probes use fresh spawned processes and their own files.
SIGKILL is a process-crash model, not power failure.

`--cases small-1k,body-4096` selects registered cases. `--case-file cases.json`
accepts a list of explicit cases. For a paired pacer experiment:

```json
[
  {"id":"pacer-10k", "workload":"W01/W16", "rate":10000,
   "duration_s":10, "pacer_variants":["cooperative","hybrid_spin"], "sample_every":1},
  {"id":"structured", "workload":"W09", "rate":10000,
   "duration_s":3, "structured":true, "depth":4, "list_entries":64, "sample_every":1}
]
```

The executed pilot input definitions are in `profiles/cpu-pacer-ab.json` and
`profiles/cpu-extended-screening.json`. The former uses the screening default
three repeats and both passes; the latter was a one-repeat latency-pass coverage
screen, not statistical acceptance. Reproduce them with `--case-file` and those
explicit repeat/pass options. Frozen executed-source archives and per-campaign
fingerprints belong with the evidence; later lint/reporting changes do not
retroactively change which version was measured.

Pacer variants are randomized inside each process-repeat block. A busy Python
pacer can hold the GIL and penalize Python consumers; it is an explicit stress
condition, not a neutral surrogate for GPU event arrival. The cooperative pacer
can instead have OS scheduling lag. Always inspect the null control and dispatch
lag, especially for short bursts. Planned peak rate is not proof of achieved peak.

## Metrics and evidence

- `offered=scheduled/H`, `attempted=actual_calls/H`.
- `goodput=unique_semantically_valid_records/(H+drain)` at the stated endpoint.
- Delivery ratio uses scheduled input too, so unattempted events do not disappear.
- Call latency is conditional on the recorded attempt/result; rejected calls can
  be faster. Keep delivery and rejection reason beside latency.
- Quantile is sorted sample `x[min(n-1,floor(p*n))]`; require `n*(1-p)>=100`.
  Missing tail support is null, not zero. No per-record constant is subtracted.
- CPU/event includes the stated source/adapter/collector scope; PSS is sampled
  process memory, not exact collector-owned bytes. RSS includes shared mappings.
- Close includes real flushing/fsync. HQSB's own hash work remains included;
  other tools do not thereby acquire HQSB supervisor-publication semantics.
- Verification duration includes deterministic truth reconstruction and is not
  a production trace-query-engine performance claim.

Raw output includes config, source fingerprints, binary receipts, process timeline,
original encoded logs, worker terminal report, semantic oracle result and failures.
Native NanoLog pretty-text derivatives are removed only after validation; the
original binary/decompressor version is retained and can regenerate them.

`report.py --campaign <path>` (repeatable) produces CSV, all-run JSON, block
summaries and metric availability. It refuses superseded campaigns. It separates
campaign/filesystem, pacer and pass; three-run intervals are exploratory. The
native and service tests may use a different output filesystem and must not be
pooled with the Python pilot for a disk-sensitive ranking.

`aux_report.py --service <all-runs.json> --moobench <all-runs.json> --output <new-dir>`
summarizes those separate tracks. `plot.py` reads saved aggregate JSON only.
`publication_ms` follows the original design: close call to supervisor publication,
including close. `supervisor_publication_ms` isolates the later publish call and
is unavailable in older captures without its separate start timestamp.

The 1GiB worker address-space guard and 2M-event receipt/oracle bound are explicit
benchmark limits. Rate/duration combinations beyond those or available storage
are resource-limited, not evidence of SUT saturation. Sustainable-capacity and GPU
interference conclusions need their separate acceptance conditions and hardware.
