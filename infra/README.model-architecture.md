# Artifact-backed model architecture

Open **环境 → 模型架构** (`/model-architecture?deployment=<id>`) or the per-model
link in devices. A downloaded model can be inspected after registration, without
loading, warming, unloading or invoking a GPU worker. No new login is required.

Inspection now starts with family-independent evidence: actual weight-path
containment, resident `nn.Module` classes/parameters/buffers after loading, and
an explicitly requested ATen dataflow capture on the existing model owner.
An unknown family never blocks these paths. Admission and execution policies
still belong to the existing model-hub/provider, not to this inspector.

Optional semantic annotations cover 13 groups: Llama, Qwen, Gemma, DeepSeek,
GLM, Mistral/Mixtral, Phi, GPT-OSS, MiniMax, InternLM, Falcon, OLMo and
GPT/OPT/Bloom, including available MoE/MLA/hybrid-layer configuration declarations.
These labels are not official-weight or device certification. Reviewed static
diagrams cover Llama, Qwen2, Qwen3 and Gemma 3 **text** implementations.
The legacy `metadata_only` field describes absence of a reviewed static diagram;
generic structure and explicit runtime capture remain independently available.

| Module | Language | Responsibility / boundary |
| --- | --- | --- |
| `hqsb.models.architecture` | Python standard library | Server-configured artifact directory → bounded config, tensor headers, shapes, storage bytes, shard checks and structure identity; no torch import or tensor data reads |
| `hqsb.models.architecture_structure` | Python standard library | Tensor paths → containment tree; config → optional semantic family and feature declarations; no hardware or executor dependency |
| `hqsb.models.architecture_graphs` | Python | Reviewed model-family implementation fingerprint + config → logical Block/Attention/MLP dependencies, including family-specific norms and layer types |
| `hqsb.backends.architecture_capture` | Python / PyTorch | `module_inventory(model)` → resident containment; `ArchitectureCapture(model).forward(phase)` → actual ATen nodes, Tensor producer/consumer and conservative storage-write edges; `document()` → `runtime-architecture-v1` |
| `hqsb.backends.interactive` | Python / PyTorch | Wrap only the actual forward; retain existing generation/cancellation/cleanup; publish diagnostic file and compact metrics |
| `hqsb.console.service`, `architecture_routes` | Python / FastAPI | Reuse epoch, serial owner, storage preflight and sealed artifacts; deployment/run IDs → checked evidence; external internals cannot be inferred |
| `StructureExplorer`, `ExecutionExplorer`, `ModelArchitecture` | TypeScript / React / SVG | Generic module tree, actual class/weight details, explicit capture, phase/operator search, dependency neighborhood, JSON export, plus existing static diagrams |

`GET /api/console/v1/deployments/{id}/architecture` follows the existing node,
auth and relay boundary. Paths cannot be supplied by the browser. Config reads
are bounded to 2 MiB, index to 8 MiB, header reads to 16 MiB/shard and 32 MiB
total, up to 256 shards. Layer count is bounded to 1024. No untrusted checkpoint
Python or pickle is executed. Metadata parsing is not full safetensors
validation or full-weight manifest verification.

Static adapters were checked against installed Transformers 5.8.0 source files.
GET inspection never constructs a model. Exact source-file fingerprints gate
only static semantic graph generation. Artifact snapshots include config,
header and implementation hashes and a graph-sensitive `snapshot_sha256`.
Browser JSON export records observation time and the separate current epoch,
load run and config match. Snapshot hashes do not certify weight values.

Edges are mathematical dependencies, residual identity paths or KV state
dependencies. Independent Q/K/V and Gate/Up branches express **possible logical
parallelism**, not observed concurrent CUDA streams. Standard decoder blocks
and successive autoregressive tokens have dependencies. Attention's mathematical
QKᵀ/mask/softmax/PV decomposition can be fused in SDPA or custom kernels; no
one-node/one-kernel correspondence, hardware placement, cache hit or Tensor Core
claim is made. Actual runtime dataflow is a separate observation source.

## Explicit runtime capture and persistence

`POST /api/console/v1/requests` adds optional `capture_architecture`, default
false. Ordinary requests retain their legacy idempotency payload. Capture is
restricted to local PyTorch, at most four final output tokens, and cannot mix
with a quality-evaluation role. The UI submits fixed `Hello`, two output tokens,
using chat or raw completion according to deployment. Reading or refreshing the
page never starts inference. Diagnostic overhead invalidates performance use
of this run's Token/s.

Only the first two actual forward calls are captured. Existing module hooks
remain intact; added hooks are removed on success and failure. There is no
forward rewrite, Tensor value copy or added CUDA synchronization. Tensor
producers are weakly referenced; anonymous node IDs replace storage addresses.
Outputs record shape/dtype/device. Shared-storage writes create conservative
edges, not proof of overlapping byte intervals or physical-page placement.

`architecture-capture.json` binds run ID, epoch, input-token digest, model and
tokenizer identity, source digest, resident module inventory, phases and graph.
It joins the existing supervisor-acknowledged experiment manifest with size and
SHA-256. The run list stores only a summary. Evidence survives model unload and
service restart. `GET /deployments/{id}/architecture/execution/{run_id}` checks
deployment ownership, terminal state, publication and integrity before reading.
The architecture GET includes the latest capture ID and whether its epoch still
matches the current instance. Historical evidence is labelled accordingly.

Budgets: at most 4096 resident modules / 2 MiB, 64 direct parameters and 64
buffers per module; 8192 execution nodes / 40000 edges by default; 3 MiB per
runtime graph file. Limits produce explicit partial/truncated evidence and
retain no dangling edges when the serialized graph is clipped.

The observed graph covers this input, visited branches and selected experts
only. Fused/custom/compiled operators may remain opaque boundaries; a module
with no dispatch events is recorded as opaque rather than inventing internals.
No single sample proves all possible dynamic paths. Kernel internals, GPU
overlap, cache hit rates and physical pages require separate profiler/system
collectors. Family recognition cannot create that missing evidence.

Local tensor storage dtype/bytes may differ from FP16 runtime residency. Tied
weights can be omitted or aliased in serialized storage; reported element/byte
counts belong to stored tensors, not a universal architecture parameter count.

Remote tests and evidence: `reports/console/model-architecture/` and
`reports/console/generic-architecture/`. Random small CPU configurations and
real GPU weights are separately recorded. Follow
`scripts/remote_run.sh` for every runtime/build/test operation. Preserve all
existing model artifacts, registrations, history and active workload ownership.
