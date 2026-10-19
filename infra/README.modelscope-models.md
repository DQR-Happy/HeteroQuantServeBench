# 4090D ModelScope model deployments

The 4090D console uses `configs/console/4090d-models.yaml`. Jetson and the generic
single-model CUDA configuration keep their existing defaults. The Mac only edits
and transfers files; run every command below from the checkout root through the
authorized target proxy.

## Artifacts and runtime boundaries

| Deployment | ModelScope repository | Runtime |
| --- | --- | --- |
| qwen-reference | Qwen/Qwen3-1.7B | Existing frozen manifest; native and HQSB Qwen3 1.7B kernel profiles |
| qwen3-06b | Qwen/Qwen3-0.6B | Framework native |
| qwen3-4b | Qwen/Qwen3-4B | Framework native |
| qwen3-8b | Qwen/Qwen3-8B | Framework native |
| gemma3-1b | google/gemma-3-1b-it | Framework native; text-only 1B checkpoint |
| llama32-1b | LLM-Research/Llama-3.2-1B-Instruct | Framework native; ModelScope mirror |
| llama32-3b | LLM-Research/Llama-3.2-3B-Instruct | Framework native; ModelScope mirror |
| deepseek-r1-15b | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | Framework native; distilled Qwen2 architecture |

The catalogue targets the configured 24 GB single-GPU setup. Larger checkpoints, quantized formats, remote Python code, MoE, multimodal
inputs and new custom-kernel shape variants require separate integration and
validation. Each listed deployment caps total context at 2048 tokens and output
at 512, and loads full FP16 weights on CUDA:0 without silently accepting CPU or
disk offload. Only one local deployment may own the GPU. The configured limit of
16 deployments bounds lazy workers; it does not enable 16 resident models.

## Acquire weights from ModelScope

```bash
HQSB_REMOTE_HOST=4090D-hqsb \
HQSB_REMOTE_DIR=/root/HeteroQuantServeBench-4090D \
./scripts/remote_run.sh .venv-console/bin/python \
  scripts/models/download_catalog_modelscope.py \
  --root /root/autodl-tmp/hqsb/models \
  gemma3-1b llama3.2-1b llama3.2-3b qwen3-0.6b qwen3-4b qwen3-8b deepseek-r1-1.5b
```

The curated catalog controls repository IDs. Acquisition uses ModelScope 1.29
only, with at most two files and sixteen range requests per file concurrently
(configurable `--download-parallels`). A per-model file lock serializes duplicate
invocations. Default admission retains 8 GiB of free disk beyond the complete
checkpoint estimate; range merging also consumes transient disk space. Disk admission is
checked alongside other applications sharing the disk. Interrupted SDK temporary files
may remain; they are never accepted as verified weights.

`models/manifests/<key>.sha256` contains publisher SHA256 for every selected
runtime asset, tokenizer and supplied README/license. The adjacent JSON records
ModelScope repository, branch metadata, per-file revisions, sizes, hashes and
download time. ModelScope's SDK resolves the `master` branch. Verification
against per-file hashes captured before downloading detects branch
movement or corrupt content. Sealed snapshots are reused offline rather than
silently updated on subsequent invocations; an intentional update needs a new
snapshot directory and separate validation.

Only the exact SDK cache names `.msc` and `.mv` are allowed outside the manifest;
they are recorded separately as client metadata. All other extra
files still fail strict verification. Custom Python files and alternate `.bin`
weight representations are excluded. Loading uses Transformers with
`local_files_only=True` and `trust_remote_code=False`, so it never falls back to
downloading weights from Hugging Face. Access restrictions and model licenses
remain those of each ModelScope repository.

The existing Qwen3 1.7B snapshot stays at `~/models/hqsb/Qwen3-1.7B`, retaining its
original project manifest. New snapshots live on the data disk under
`/root/autodl-tmp/hqsb/models`. Model artifact content never shares the console
database or private evidence directory.

## Serve and switch

```bash
HQSB_REMOTE_HOST=4090D-hqsb \
HQSB_REMOTE_DIR=/root/HeteroQuantServeBench-4090D \
./scripts/remote_run.sh .venv-console/bin/python -m hqsb.console \
  --config configs/console/4090d-models.yaml --port 8768
```

Keep the existing `HQSB_CONSOLE_DATA` directory when replacing a service. Drain
pending requests, unload its resident model, take a SQLite backup, then stop the
old service before binding the same port. Authentication remains disabled in this
test configuration. Operators can additionally set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` on the service environment; acquisition is a separate
explicit ModelScope command.

Select a deployment in the inference page and click **切换模型并加载方案**. The
client waits for the old model's unload acknowledgment before loading the new
model, and verifies its new epoch/load-run identity. Failed unloads stop the
sequence. Backend GPU ownership and benchmark ownership checks remain enforced;
GPU ownership is coordinated by the backend across browser windows.

The benchmark page resets scheme selections when changing models. Starting a
comparison can drain another resident model through the same lifecycle helper.
Qwen3 1.7B retains its custom-kernel comparisons. Other models default to
`empty_cache` versus `preserve_allocator` to compare allocator cleanup. Model switching does not extend custom kernels
to unvalidated families or sizes. Benchmark reports keep model, tokenizer and
actual token identity checks; measurements across different models cannot be
presented as an isolated operator speedup.

Each model uses its own shipped chat template. Qwen3 receives
`enable_thinking=False`; other models keep their template default. In particular,
the distilled R1 checkpoint is not advertised as having a working thinking-off
switch. The actual template policy is recorded in load details and inference
metrics. A finite-logit warmup rejects obvious dtype/runtime failures at load;
Task datasets provide the subsequent model-quality evaluation.

## Observations and acceptance

The same request protocol records load identity, actual model class/type, device
placement, parameter storage, request KV snapshots, allocator counters, host
pages, CPU/CUDA operator traces and sealed execution receipts. Availability and
scope remain explicit. CPU pages use pagemap, native CUDA traces record device
activities, and hardware counters supply cache-hit measurements. Gemma's
sliding-layer KV storage follows the configured window.

The allocator history has a bounded 100,000-entry ring. In the 2026-10-17 Gemma
check, a request producing 23 tokens overwrote that ring and was correctly marked
`partial` with `ALLOCATOR_HISTORY_RING_OVERWRITTEN`. That raw result is retained.
Four-token memory diagnostics were complete; the longer basic generation check
also succeeded. Complete short captures do not establish complete allocator
history for longer requests, and no capture limit was raised to change the old
result's status.

Acceptance must include every selected model loading with its manifest verified,
full CUDA placement, finite last-token warmup logits, successful templated
generation, populated parameter/KV inventories, acknowledged memory/page
captures, a hash-verified trace with CUDA activities, and a small finite-logit
quality fixture. The fixture checks the quality-evaluation path. Task accuracy
uses its own dataset. Downloads and diagnostics perturb timings; performance comparisons use a
separate fixed-input run after all downloads stop. Existing Qwen3 custom-kernel
and lifecycle tests remain required for compatibility.

Private target evidence for the 2026-10-17 deployment is stored under
`reports/deployment/4090d-models-20261017/`, including disk cleanup, publisher
provenance, test logs and actual run IDs. Do not replace failed raw logs with
successful summaries.
