# ModelScope discovery and node-owned model acquisition

Open **Environment → 模型下载与接入** (`/models`); devices and benchmarks also link
there. The existing selected-node API prefix owns every catalogue, inspection,
download and dynamic deployment. SSH/HTTP relay does not run the download on the
Mac or move the result between nodes. No additional login is introduced.

## Responsibilities and contracts

| Owner | Language | Input → output / invariant |
| --- | --- | --- |
| `hqsb.models.modelscope_source` | Python | Registered publisher/repo/revision → live SDK metadata, validated runtime file plan, per-file commit/SHA-256, bounded byte stream; no torch import |
| `hqsb.models.model_admission` | Python | Weight bytes, model shape, node capacities → conservative FP16/batch1/context2048 estimate and explicit blocking reasons; never claims runtime certification |
| `hqsb.console.model_hub` | Python | Frozen inspection digest → durable download jobs, cancellation/resume, verified immutable snapshot, optional deployment registration; owns threads, disk paths and journals |
| `hqsb.console.model_hub_routes` | Python | Strict HTTP requests → above operations; same auth/CSRF and node relay as inference |
| `ConsoleService.register_deployment` | Python | Verified server-built config → unloaded actor/executor; keeps serial GPU owner, normal load/warmup and manifest gates |
| `pages/ModelHub.tsx`, `features/model-hub/` | TypeScript / CSS | Node API snapshots → searchable catalogue, version/check drawer and real byte/file progress; no browser weight transfer |

Metadata calls use installed/pinned ModelScope 1.29 `HubApi.list_models`,
`get_model_branches_and_tags`, `get_model_files`. Byte downloads use that version's
[official file endpoint contract](https://github.com/modelscope/modelscope/blob/v1.29.0/modelscope/hub/file_download.py)
with a **per-file publisher commit**, identity encoding, bounded streaming reads,
and validated HTTP Range when resuming. This does not monkeypatch SDK progress or
parse terminal tqdm output. No Hugging Face download fallback exists.

## Discovery scope and availability

Current source policy registers Qwen, google, deepseek-ai, microsoft, ZhipuAI,
internlm and 01ai. LLM-Research is explicitly labelled a platform mirror, **not**
the original vendor. This list is maintained in `modelscope_source.PUBLISHERS`;
only add a new owner after checking provenance. Do not claim these owners cover
all ModelScope uploaders or every official model worldwide.

Refresh indexes **all public repositories for the selected owner**: it first
requests a bounded single response (up to 10,000 repositories), avoiding losses
at equal-ranked page boundaries. If the server caps that response, it falls back
to 50/page and merges by repository identity. Completion requires the unique
count to equal a consistent upstream total; raw row count alone is insufficient.
Search/pagination/resource filtering operate over the indexed set.
Click the parameter count or estimated runtime memory column to cycle ascending,
descending and default ordering. Numeric sorting runs on the node over the full
filtered set before pagination; unknown values are always last, equal values use
repository IDs as a stable tie-break. Sorting stays in the URL across searches.
Partial results, errors and progress stay visible. A 10,000-repository budget
marks oversized indexes partial instead of silently claiming completeness. The
remote list can change during pagination; the count is an observed snapshot,
not a transactional upstream inventory. All public branches/tags of a selected
repository are read separately. Per-file hashes/commits freeze its actual files.
Refresh keeps the previous usable catalogue visible until replacement completes;
initial discovery exposes received pages with an explicit in-progress message.
Raw scanned rows, duplicate count and unique repository count are separate.
An unresolved upstream total mismatch is reported as partial with a reason, not an empty
success or a claim that every duplicate represents a missing repository.

Summary filtering uses model parameter count or weight size and architecture
metadata; native causal-language models with missing size summaries remain
visible as `needs_inspection`, without claiming a resource pass. This includes
Gemma 3 270M repositories whose upstream list omits both sizes and parameters.
Missing architecture still requires cancelling the filter or inspecting a full
registered-owner repo ID. Publisher and search are retained in the page URL.
Concrete revision checks remain mandatory. Download
admission additionally rejects unsupported architectures, remote Python code,
quantization, unsupported MoE/multimodal configs, missing chat templates, missing
safetensors/tokenizer/config files, incomplete shards, missing hashes and
insufficient memory/disk. Downloads are restricted to safe root runtime files;
alternate `.bin`/GGUF formats and custom Python code are not selected.

GPU capacity is measured without importing CUDA into the API process. Runtime
memory is **estimated** as weight storage + shape-derived FP16 KV (or a reserve
when unknown) + max(2 GiB, 20% weight bytes). Weight bytes may overestimate FP16
residency for FP32 storage. Batch1, context≤2048, one GPU, no offload; shared
Tegra capacity is separately labelled. Host staging has an additional check.
This does not prove supported operators, quality, numerical stability or speed.
After registration, the existing strict loader and finite-logit warmup remain
the runtime gate; inference/observation runs provide actual evidence.

## Jobs and persistence

- `.console/model-hub/catalog-<owner>.json`: cached catalogue/progress.
- `.console/model-hub/download-<uuid>.json`: frozen plan, received bytes, current
  file, verified count, state, attempts and errors. API omits bulky file plans.
- `model_download_dir/partial-<plan_sha>/`: request-owned partial/checksummed files.
- `model_download_dir/<plan_sha>/`: immutable verified snapshot, published by
  rename only after strict manifest verification.
- `.console/model-hub/<plan_sha>.sha256` and `.provenance.json`: publisher identity.
- `.console/model-hub/deployments.json`: server-built dynamic deployment configs.

`model_download_dir` defaults below the private hub directory; configure a data
disk on each execution node. The 3090 config uses
`/root/autodl-tmp/hqsb/downloaded-models`. Default free-disk reserve is 4 GiB;
remaining bytes and reserve are checked again when queued work begins and during
transfer. Other applications can still consume disk; this is admission, not an
OS-level quota. No unrelated file cleanup occurs.

One download transfers at a time, at most eight accepted active/queued jobs;
200 durable job records bound metadata. Identical frozen plans return the same
job; failure/cancellation requires explicit resume. The UI polls real byte state
and separate verified file counts. A 100% byte value can still be verifying or
failed, so success is the sealed terminal state, not the percentage alone.

Network interruption leaves `.part` files. Resume verifies completed files and
requires the server to acknowledge the exact offset via 206/Content-Range;
ignored Range is an error. Hash-failed files are quarantined on the model disk
and never registered. Restart marks unfinished jobs interrupted, preserving
identity. A crash after snapshot rename can recover that exact verified snapshot.
The service owns and joins acquisition threads at shutdown, independently of the
GPU queue. These downloads and metadata hashing consume disk/network/CPU and can
still perturb a concurrent benchmark; run formal performance samples separately.

After success, explicit **登记为推理部署** publishes an unloaded framework-native
model, persists registration and exposes it in playground/benchmarks without
restarting. At most 16 deployments are registered per node. Reaching this limit
does not delete a completed download. Existing configured models/epochs are not
changed. Newly registered Qwen3 1.7B does not inherit the separately verified
custom native-kernel manifest automatically. Licenses and gated access remain
the publisher's; this UI does not accept agreements or bypass authorization.

## API and validation

`/api/console/v1/model-hub`: `GET /config`, `GET /catalog`,
`POST /catalog/refresh?publisher=...`, `POST /revisions`, `POST /inspect`,
`GET/POST /downloads`, `POST /downloads/{id}/{cancel,resume,register}`.
`GET /catalog` accepts `sort_by=default|parameters|runtime_bytes` and
`sort_order=asc|desc`; omitted values preserve the existing candidate-first order.
Inspect returns a short-lived digest (30 minutes); start accepts only that digest,
not browser-supplied file paths, URLs, hashes or deployment config.

Follow the project proxy for all execution:

```bash
./scripts/remote_run.sh python -m pytest tests/unit/console/test_model_hub.py -q
./scripts/remote_run.sh python scripts/audit/import_dependency_gate.py
./scripts/remote_run.sh bash -c 'cd web/console && npm test && npm run build'
```

Unit transport doubles test corruption, resume, cancellation, disk/memory gates,
plan expiry, restart recovery and registration. They are not hardware evidence.
Actual 3090 acceptance and downloaded publisher identities are recorded under
`reports/console/model-hub/`. Preserve all old manifests and benchmark results.

Native architecture discovery reads the installed Transformers literal
`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` table with `ast`, without importing an
inference runtime in the API. This avoids an arbitrary fixed model-size/vendor
allowlist while still refusing unsupported/remote implementations. Format-name
hints (MLX/GGUF/GPTQ/AWQ/FP8/low-bit) are excluded from the default resource filter;
concrete inspection additionally checks both `quantization_config` and MLX's
`quantization`. Shanghai_AI_Laboratory is the registered InternLM publisher.
Container cgroup v2/v1 memory limits constrain host-staging availability; raw
host `/proc/meminfo` is not advertised as the container's allocation budget.
