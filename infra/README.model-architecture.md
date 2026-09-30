# Artifact-backed model architecture

Open **环境 → 模型架构** (`/model-architecture?deployment=<id>`) or the per-model
link in devices. A downloaded model can be inspected after registration, without
loading, warming, unloading or invoking a GPU worker. No new login is required.

The initial adapters cover Qwen2, Qwen3 and Gemma 3 **text** models. Unknown
families, missing/inconsistent weights or changed model implementation files
produce `metadata_only`, not a guessed Transformer diagram. Multimodal, MoE,
remote code and packed quantization need separate reviewed adapters.

| Module | Language | Responsibility / boundary |
| --- | --- | --- |
| `hqsb.models.architecture` | Python standard library | Server-configured artifact directory → bounded config, tensor headers, shapes, storage bytes, shard checks and structure identity; no torch import or tensor data reads |
| `hqsb.models.architecture_graphs` | Python | Reviewed model-family implementation fingerprint + config → logical Block/Attention/MLP dependencies, including family-specific norms and layer types |
| `hqsb.console.architecture_routes` | Python / FastAPI | Deployment ID → read-only selected-node inspection and separately labelled current load identity; external providers rejected |
| `pages/ModelArchitecture`, `features/architecture` | TypeScript / SVG / CSS | Artifact snapshot → serial overview, layer picker, interactive DAG, tensor evidence, JSON export and capability boundaries |

`GET /api/console/v1/deployments/{id}/architecture` follows the existing node,
auth and relay boundary. Paths cannot be supplied by the browser. Config reads
are bounded to 2 MiB, index to 8 MiB, header reads to 16 MiB/shard and 32 MiB
total, up to 256 shards. Layer count is bounded to 1024. No untrusted checkpoint
Python or pickle is executed. Metadata parsing is not full safetensors
validation or full-weight manifest verification.

The adapter was checked against installed Transformers 5.8.0 source files and
CPU-only meta module trees on 3090. Runtime inspection itself never constructs
a model. Exact source-file fingerprints gate logical graph generation; future
versions degrade to metadata until reviewed. Artifact snapshots include config,
header and implementation hashes and a graph-sensitive `snapshot_sha256`.
Browser JSON export records observation time and the separate current epoch,
load run and config match. Snapshot hashes do not certify weight values.

Edges are mathematical dependencies, residual identity paths or KV state
dependencies. Independent Q/K/V and Gate/Up branches express **possible logical
parallelism**, not observed concurrent CUDA streams. Standard decoder blocks
and successive autoregressive tokens have dependencies. Attention's mathematical
QKᵀ/mask/softmax/PV decomposition can be fused in SDPA or custom kernels; no
one-node/one-kernel correspondence, hardware placement, cache hit or Tensor Core
claim is made. Runtime trace is a separate observation source.

Local tensor storage dtype/bytes may differ from FP16 runtime residency. Tied
weights can be omitted or aliased in serialized storage; reported element/byte
counts belong to stored tensors, not a universal architecture parameter count.

Remote tests and evidence: `reports/console/model-architecture/`. Follow
`scripts/remote_run.sh` for every runtime/build/test operation. Preserve all
existing model artifacts, registrations, history and active workload ownership.
