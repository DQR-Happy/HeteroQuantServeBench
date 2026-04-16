可以。下面只保留**可执行步骤**，不再展开原理，也去掉 Git/Commit 相关内容。模型下载、模型定位和加载全部切换到 **ModelScope 魔搭**。ModelScope 当前有官方 `Qwen/Qwen3-1.7B` 模型页，并支持通过 `snapshot_download` 或 CLI 下载到指定本地目录。([ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B/summary "千问3-1.7B"))

本阶段完成后的目标是：

> **Jetson Orin Nano Super + Qwen3-1.7B FP16 + ModelScope + Model-Core Benchmark + 固定 ISL/OSL + Prefill/Decode/TTFT/TPOT/TPS/显存/系统内存/功耗/温度 + 可重复结果文件。**

先不要做 CUDA 自定义算子、量化、vLLM、SGLang。

---

# 0\. 本阶段最终目录

在你现有仓库下补齐：

```text
HeteroQuantServeBench/
├── configs/
│   ├── environment/
│   ├── models/
│   │   └── qwen3_1_7b.yaml
│   └── benchmarks/
│       └── jetson_qwen3_fp16.yaml
│
├── hqsb/
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── workload.py
│   │   ├── model_core.py
│   │   ├── resource_monitor.py
│   │   └── tegrastats_parser.py
│   │
│   └── models/
│       ├── __init__.py
│       └── loader.py
│
├── benchmarks/
│   ├── scripts/
│   │   ├── run_model_core.py
│   │   ├── run_jetson_baseline.py
│   │   └── summarize_baseline.py
│   ├── schemas/
│   └── workloads/
│
├── scripts/
│   └── models/
│       ├── download_qwen3_modelscope.py
│       ├── verify_qwen3.py
│       └── dump_model_manifest.py
│
├── docs/
│   └── benchmark/
│       ├── methodology.md
│       ├── metric_definitions.md
│       └── qwen3_model_manifest.json
│
└── reports/                 # 整体继续 gitignore
    └── dev/
        └── llm/
```

---

# 1\. 创建目录

在 Jetson：

```bash
cd ~/work/HeteroQuantServeBench

mkdir -p \
  configs/environment \
  configs/models \
  configs/benchmarks \
  hqsb/benchmark \
  hqsb/models \
  benchmarks/scripts \
  benchmarks/schemas \
  benchmarks/workloads \
  scripts/models \
  docs/benchmark \
  reports/dev/llm
```

初始化 Python Package：

```bash
touch hqsb/__init__.py
touch hqsb/benchmark/__init__.py
touch hqsb/models/__init__.py
```

---

# 2\. 检查 Jetson PyTorch，不要重新安装 Torch

执行：

```bash
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print(
        "compute capability:",
        torch.cuda.get_device_capability(0)
    )
PY
```

必须满足：

```text
CUDA available: True
```

并且能看到 Jetson GPU。

**不要执行：**

```bash
pip install -U torch
```

---

# 3\. 创建项目 Python 环境

使用系统中的 Jetson PyTorch：

```bash
cd ~/work/HeteroQuantServeBench

python3 -m venv \
  --system-site-packages \
  .venv

source .venv/bin/activate
```

再次确认：

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

必须仍然：

```text
True
```

---

# 4\. 安装 Benchmark 和 ModelScope 依赖

执行：

```bash
python -m pip install --upgrade pip

python -m pip install \
  modelscope \
  "transformers>=4.51,<5" \
  accelerate \
  safetensors \
  psutil \
  pyyaml \
  numpy
```

这里虽然安装了 `transformers` Python 包，但**不通过 Hugging Face 网站下载任何东西**；模型获取全部由 ModelScope 完成，然后只从本地路径加载。

ModelScope 官方文档和示例均支持 `snapshot_download()` 下载模型到本地，再从本地目录加载。([ModelScope](https://www.modelscope.cn/learn/434591?utm_source=chatgpt.com "Getting started with ModelScope Notebook · Learn"))

验证：

```bash
python - <<'PY'
import torch
import transformers
import modelscope
import accelerate
import psutil

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("modelscope:", modelscope.__version__)
print("accelerate:", accelerate.__version__)
print("CUDA:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
PY
```

---

# 5\. 锁定当前 Python 环境

```bash
python -m pip freeze \
  > configs/environment/jetson_python_lock.txt
```

同时记录关键版本：

```bash
cat > configs/environment/jetson_runtime.txt <<EOF
Timestamp: $(date --iso-8601=seconds)

CUDA:
$(nvcc --version)

Python:
$(python --version)

PyTorch:
$(python -c 'import torch; print(torch.__version__)')

Transformers:
$(python -c 'import transformers; print(transformers.__version__)')

ModelScope:
$(python -c 'import modelscope; print(modelscope.__version__)')

Device:
$(python -c 'import torch; print(torch.cuda.get_device_name(0))')
EOF
```

---

# 6\. 建立外部模型目录

模型不要放项目仓库。

```bash
mkdir -p "$HOME/models/hqsb"

export HQSB_MODEL_ROOT="$HOME/models/hqsb"
```

永久添加：

```bash
grep -q "HQSB_MODEL_ROOT" ~/.bashrc || \
echo 'export HQSB_MODEL_ROOT="$HOME/models/hqsb"' \
>> ~/.bashrc
```

---

# 7\. 使用 ModelScope 下载 Qwen3-1.7B

ModelScope当前提供：

```text
Qwen/Qwen3-1.7B
```

官方模型页可直接访问。([ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B/summary "千问3-1.7B"))

创建：

```bash
cat > scripts/models/download_qwen3_modelscope.py <<'PY'
from __future__ import annotations

import os

from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen3-1.7B"

MODEL_ROOT = os.path.expanduser(
    os.environ.get(
        "HQSB_MODEL_ROOT",
        "~/models/hqsb",
    )
)

LOCAL_DIR = os.path.join(
    MODEL_ROOT,
    "Qwen3-1.7B",
)

print(f"ModelScope model : {MODEL_ID}")
print(f"Local directory  : {LOCAL_DIR}")

os.makedirs(
    MODEL_ROOT,
    exist_ok=True,
)

model_dir = snapshot_download(
    MODEL_ID,
    local_dir=LOCAL_DIR,
)

print()
print("Download complete.")
print("Model directory:")
print(model_dir)
PY
```

执行：

```bash
source .venv/bin/activate

python scripts/models/download_qwen3_modelscope.py
```

ModelScope官方示例即使用：

```python
from modelscope import snapshot_download
snapshot_download(model_id, local_dir=...)
```

这种形式。([ModelScope](https://www.modelscope.cn/learn/434591?utm_source=chatgpt.com "Getting started with ModelScope Notebook · Learn"))

---

# 8\. 检查模型完整性

```bash
du -sh "$HOME/models/hqsb/Qwen3-1.7B"

find "$HOME/models/hqsb/Qwen3-1.7B" \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort
```

至少应看到类似：

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
model*.safetensors
...
```

---

# 9\. 为模型生成本地 SHA256 Manifest

因为以后 ModelScope 上的默认分支可能变化，所以不要只依赖远端名称。

直接把你当前下载到本地的模型文件哈希锁死：

```bash
cd "$HOME/models/hqsb/Qwen3-1.7B"

find . \
  -type f \
  ! -path './.cache/*' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > model_sha256_manifest.txt
```

检查：

```bash
head -n 20 model_sha256_manifest.txt
```

复制一份到项目：

```bash
cp \
  "$HOME/models/hqsb/Qwen3-1.7B/model_sha256_manifest.txt" \
  ~/work/HeteroQuantServeBench/docs/benchmark/
```

以后最终 Benchmark 可以通过：

```bash
cd "$HOME/models/hqsb/Qwen3-1.7B"

sha256sum -c model_sha256_manifest.txt
```

验证模型 Artifact 是否发生变化。

---

# 10\. 创建模型配置

回到项目：

```bash
cd ~/work/HeteroQuantServeBench
```

创建：

```bash
cat > configs/models/qwen3_1_7b.yaml <<'EOF'
model:
  source: modelscope
  id: Qwen/Qwen3-1.7B

  local_path: ~/models/hqsb/Qwen3-1.7B

  dtype: float16

  attention_backend: eager

  trust_remote_code: false

  local_files_only: true

  batch_size: 1
EOF
```

---

# 11\. 验证模型架构

创建：

```bash
cat > scripts/models/verify_qwen3.py <<'PY'
from __future__ import annotations

import os

from modelscope import AutoConfig

MODEL_PATH = os.path.expanduser(
    "~/models/hqsb/Qwen3-1.7B"
)

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("model_type:")
print(config.model_type)

print("hidden_size:")
print(config.hidden_size)

print("intermediate_size:")
print(config.intermediate_size)

print("layers:")
print(config.num_hidden_layers)

print("attention_heads:")
print(config.num_attention_heads)

print("kv_heads:")
print(config.num_key_value_heads)

print("max_position_embeddings:")
print(config.max_position_embeddings)

print("rms_norm_eps:")
print(config.rms_norm_eps)

print("rope_theta:")
print(config.rope_theta)
PY
```

执行：

```bash
python scripts/models/verify_qwen3.py
```

确认没有模型识别异常。

---

# 12\. 生成模型 Manifest

创建：

```bash
cat > scripts/models/dump_model_manifest.py <<'PY'
from __future__ import annotations

import json
import os

from modelscope import AutoConfig

MODEL_PATH = os.path.expanduser(
    "~/models/hqsb/Qwen3-1.7B"
)

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

manifest = {
    "source":
        "modelscope",

    "model_id":
        "Qwen/Qwen3-1.7B",

    "model_type":
        config.model_type,

    "hidden_size":
        config.hidden_size,

    "intermediate_size":
        config.intermediate_size,

    "num_hidden_layers":
        config.num_hidden_layers,

    "num_attention_heads":
        config.num_attention_heads,

    "num_key_value_heads":
        config.num_key_value_heads,

    "max_position_embeddings":
        config.max_position_embeddings,

    "rms_norm_eps":
        config.rms_norm_eps,

    "rope_theta":
        config.rope_theta,
}

print(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
)
PY
```

执行：

```bash
python scripts/models/dump_model_manifest.py \
  > docs/benchmark/qwen3_model_manifest.json
```

查看：

```bash
cat docs/benchmark/qwen3_model_manifest.json
```

---

# 13\. 建立统一 Model Loader

创建：

```bash
cat > hqsb/models/loader.py <<'PY'
from __future__ import annotations

import os
import time

import torch

from modelscope import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def load_qwen3(
    model_path: str,
    dtype=torch.float16,
    attention_backend: str = "eager",
):
    model_path = os.path.expanduser(
        model_path
    )

    start = time.perf_counter()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=
                attention_backend,
        )
    )

    model.eval()

    torch.cuda.synchronize()

    load_time_s = (
        time.perf_counter()
        - start
    )

    return (
        tokenizer,
        model,
        load_time_s,
    )
PY
```

---

# 14\. 先跑模型 Smoke Test

创建：

```bash
cat > scripts/models/smoke_qwen3.py <<'PY'
from __future__ import annotations

import os
import time

import torch

from hqsb.models.loader import load_qwen3

MODEL_PATH = os.path.expanduser(
    "~/models/hqsb/Qwen3-1.7B"
)

print("Loading model...")

tokenizer, model, load_time = (
    load_qwen3(
        MODEL_PATH,
        dtype=torch.float16,
        attention_backend="eager",
    )
)

print(
    f"Model load time: "
    f"{load_time:.2f} s"
)

messages = [
    {
        "role": "user",
        "content":
            "Briefly explain what CUDA is."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to("cuda")

torch.cuda.synchronize()

start = time.perf_counter()

with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )

torch.cuda.synchronize()

elapsed = (
    time.perf_counter()
    - start
)

generated = output[
    0,
    inputs["input_ids"].shape[1]:
]

print()
print(
    tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )
)

print()
print(
    "input tokens:",
    inputs["input_ids"].shape[1],
)

print(
    "output tokens:",
    generated.numel(),
)

print(
    "generation time:",
    elapsed,
)
PY
```

设置 Python Path：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

执行：

```bash
python scripts/models/smoke_qwen3.py
```

这一关要求：

```text
模型成功加载
GPU可见
没有OOM
能生成文本
没有CUDA Runtime错误
```

---

# 15\. 同时观察 Jetson 内存

另一个 SSH 终端：

```bash
tegrastats --interval 1000
```

再执行：

```bash
python scripts/models/smoke_qwen3.py
```

重点先看：

```text
RAM
SWAP
GR3D_FREQ
温度
功耗
```

如果 FP16 稳定运行，继续。

---

# 16\. 建立固定 Token Workload

创建：

```bash
cat > hqsb/benchmark/workload.py <<'PY'
from __future__ import annotations

import torch

def make_fixed_token_input(
    tokenizer,
    input_tokens: int,
    device: str = "cuda",
):
    seed_text = (
        "CUDA GPU inference optimization "
        "memory bandwidth kernel latency "
        "transformer attention cache "
        "performance benchmark. "
    )

    token_ids = tokenizer(
        seed_text,
        add_special_tokens=False,
    )["input_ids"]

    if not token_ids:
        raise RuntimeError(
            "Tokenizer produced no tokens."
        )

    expanded = []

    while len(expanded) < input_tokens:
        expanded.extend(token_ids)

    expanded = expanded[:input_tokens]

    input_ids = torch.tensor(
        [expanded],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
    )

    assert (
        input_ids.shape[1]
        ==
        input_tokens
    )

    return {
        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,
    }
PY
```

---

# 17\. 创建统计模块

```bash
cat > hqsb/benchmark/metrics.py <<'PY'
from __future__ import annotations

import math
import statistics

def percentile(
    values: list[float],
    quantile: float,
) -> float:

    if not values:
        return float("nan")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        len(ordered) - 1
    ) * quantile

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower]
        * (1.0 - weight)
        +
        ordered[upper]
        * weight
    )

def latency_summary(
    values_ms: list[float],
) -> dict:

    if not values_ms:
        return {}

    return {
        "count":
            len(values_ms),

        "mean_ms":
            statistics.mean(
                values_ms
            ),

        "median_ms":
            statistics.median(
                values_ms
            ),

        "stddev_ms":
            statistics.pstdev(
                values_ms
            ),

        "min_ms":
            min(values_ms),

        "max_ms":
            max(values_ms),

        "p50_ms":
            percentile(
                values_ms,
                0.50,
            ),

        "p95_ms":
            percentile(
                values_ms,
                0.95,
            ),

        "p99_ms":
            percentile(
                values_ms,
                0.99,
            ),
    }
PY
```

---

# 18\. 建立 Model-Core Benchmark

创建：

```bash
cat > hqsb/benchmark/model_core.py <<'PY'
from __future__ import annotations

import time

import torch

from hqsb.benchmark.metrics import (
    latency_summary,
)

@torch.inference_mode()
def benchmark_model_core(
    model,
    inputs,
    output_tokens: int,
):

    if output_tokens < 1:
        raise ValueError(
            "output_tokens must be >= 1"
        )

    input_ids = inputs["input_ids"]
    attention_mask = (
        inputs["attention_mask"]
    )

    input_token_count = (
        input_ids.shape[1]
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    #
    # PREFILL
    #

    torch.cuda.synchronize()

    prefill_start = (
        time.perf_counter()
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    torch.cuda.synchronize()

    prefill_forward_ms = (
        time.perf_counter()
        - prefill_start
    ) * 1000.0

    #
    # FIRST TOKEN
    #

    first_token_start = (
        time.perf_counter()
    )

    next_token = (
        outputs.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    past_key_values = (
        outputs.past_key_values
    )

    torch.cuda.synchronize()

    first_token_selection_ms = (
        time.perf_counter()
        - first_token_start
    ) * 1000.0

    model_core_ttft_ms = (
        prefill_forward_ms
        +
        first_token_selection_ms
    )

    generated_tokens = [
        int(next_token.item())
    ]

    #
    # DECODE
    #

    itl_ms = []

    current_length = (
        input_token_count
    )

    for _ in range(
        1,
        output_tokens,
    ):

        current_length += 1

        decode_attention_mask = (
            torch.ones(
                (
                    1,
                    current_length,
                ),
                dtype=torch.long,
                device=input_ids.device,
            )
        )

        torch.cuda.synchronize()

        decode_start = (
            time.perf_counter()
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=
                decode_attention_mask,
            past_key_values=
                past_key_values,
            use_cache=True,
        )

        next_token = (
            outputs.logits[:, -1, :]
            .argmax(
                dim=-1,
                keepdim=True,
            )
        )

        past_key_values = (
            outputs.past_key_values
        )

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter()
            - decode_start
        ) * 1000.0

        itl_ms.append(
            elapsed_ms
        )

        generated_tokens.append(
            int(next_token.item())
        )

    decode_total_ms = sum(
        itl_ms
    )

    model_core_e2e_ms = (
        model_core_ttft_ms
        +
        decode_total_ms
    )

    decode_tokens = max(
        output_tokens - 1,
        0,
    )

    prefill_tps = (
        input_token_count
        /
        (prefill_forward_ms / 1000.0)
    )

    decode_tps = (
        decode_tokens
        /
        (decode_total_ms / 1000.0)
        if decode_total_ms > 0
        else 0.0
    )

    output_tps = (
        output_tokens
        /
        (model_core_e2e_ms / 1000.0)
    )

    return {
        "input_tokens":
            input_token_count,

        "output_tokens":
            output_tokens,

        "prefill_forward_ms":
            prefill_forward_ms,

        "first_token_selection_ms":
            first_token_selection_ms,

        "model_core_ttft_ms":
            model_core_ttft_ms,

        "decode_total_ms":
            decode_total_ms,

        "model_core_e2e_ms":
            model_core_e2e_ms,

        "prefill_tokens_per_s":
            prefill_tps,

        "decode_tokens_per_s":
            decode_tps,

        "model_core_output_tokens_per_s":
            output_tps,

        "itl":
            latency_summary(
                itl_ms
            ),

        "peak_cuda_allocated_mb":
            (
                torch.cuda
                .max_memory_allocated()
                / 1024**2
            ),

        "peak_cuda_reserved_mb":
            (
                torch.cuda
                .max_memory_reserved()
                / 1024**2
            ),

        "generated_token_ids":
            generated_tokens,
    }
PY
```

---

# 19\. 创建单 Case Runner

```bash
cat > benchmarks/scripts/run_model_core.py <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time

import psutil
import torch
import transformers
import modelscope

from hqsb.benchmark.model_core import (
    benchmark_model_core,
)
from hqsb.benchmark.workload import (
    make_fixed_token_input,
)
from hqsb.models.loader import (
    load_qwen3,
)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--model-path",
        default=
            "~/models/hqsb/Qwen3-1.7B",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    model_path = os.path.expanduser(
        args.model_path
    )

    (
        tokenizer,
        model,
        load_time_s,
    ) = load_qwen3(
        model_path,
        dtype=torch.float16,
        attention_backend="eager",
    )

    workload = make_fixed_token_input(
        tokenizer,
        args.input_tokens,
    )

    #
    # Warmup
    #

    print("Warmup...")

    benchmark_model_core(
        model,
        workload,
        min(
            args.output_tokens,
            8,
        ),
    )

    runs = []

    process = psutil.Process()

    for repetition in range(
        args.repetitions
    ):

        print(
            f"Repetition "
            f"{repetition + 1}/"
            f"{args.repetitions}"
        )

        result = benchmark_model_core(
            model,
            workload,
            args.output_tokens,
        )

        result[
            "process_rss_mb"
        ] = (
            process.memory_info().rss
            / 1024**2
        )

        result[
            "system_memory_used_mb"
        ] = (
            psutil.virtual_memory().used
            / 1024**2
        )

        runs.append(result)

    hashes = []

    for run in runs:

        encoded = json.dumps(
            run["generated_token_ids"]
        ).encode()

        hashes.append(
            hashlib.sha256(
                encoded
            ).hexdigest()
        )

    deterministic = (
        len(set(hashes))
        == 1
    )

    output = {
        "schema_version":
            "1.0",

        "timestamp":
            time.time(),

        "hardware": {
            "device":
                torch.cuda
                .get_device_name(0),

            "compute_capability":
                torch.cuda
                .get_device_capability(0),
        },

        "software": {
            "python":
                platform.python_version(),

            "torch":
                torch.__version__,

            "torch_cuda":
                torch.version.cuda,

            "transformers":
                transformers.__version__,

            "modelscope":
                modelscope.__version__,
        },

        "model": {
            "source":
                "modelscope",

            "id":
                "Qwen/Qwen3-1.7B",

            "local_path":
                model_path,

            "dtype":
                "float16",

            "backend":
                "modelscope-transformers",

            "attention_backend":
                "eager",

            "load_time_s":
                load_time_s,
        },

        "workload": {
            "batch_size":
                1,

            "input_tokens":
                args.input_tokens,

            "output_tokens":
                args.output_tokens,
        },

        "deterministic":
            deterministic,

        "generated_token_sha256":
            hashes[0],

        "repetitions":
            runs,
    }

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "deterministic:",
        deterministic,
    )

    print(
        "saved:",
        args.output,
    )

if __name__ == "__main__":
    main()
PY
```

---

# 20\. 测试最小 Benchmark

```bash
cd ~/work/HeteroQuantServeBench

source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

执行：

```bash
python benchmarks/scripts/run_model_core.py \
  --input-tokens 32 \
  --output-tokens 8 \
  --repetitions 1 \
  --output \
  reports/dev/llm/smoke.json
```

查看：

```bash
python -m json.tool \
  reports/dev/llm/smoke.json \
  | less
```

必须至少出现：

```text
prefill_forward_ms

first_token_selection_ms

model_core_ttft_ms

decode_total_ms

model_core_e2e_ms

prefill_tokens_per_s

decode_tokens_per_s

model_core_output_tokens_per_s

itl.mean_ms
itl.p50_ms
itl.p95_ms
itl.p99_ms

peak_cuda_allocated_mb

peak_cuda_reserved_mb

process_rss_mb

system_memory_used_mb
```

---

# 21\. 验证确定性

执行：

```bash
python benchmarks/scripts/run_model_core.py \
  --input-tokens 128 \
  --output-tokens 32 \
  --repetitions 3 \
  --output \
  reports/dev/llm/determinism.json
```

检查：

```bash
python - <<'PY'
import json

with open(
    "reports/dev/llm/determinism.json"
) as f:
    data = json.load(f)

print(
    "deterministic:",
    data["deterministic"],
)
PY
```

目标：

```text
deterministic: True
```

如果不是 True，先停止后续完整 Benchmark。

---

# 22\. 创建正式 Workload 配置

```bash
cat > configs/benchmarks/jetson_qwen3_fp16.yaml <<'EOF'
benchmark:

  model:
    Qwen/Qwen3-1.7B

  model_source:
    modelscope

  backend:
    modelscope-transformers

  dtype:
    float16

  attention_backend:
    eager

  batch_size:
    1

  warmup:
    true

  repetitions:
    3

workloads:

  - name: tiny
    input_tokens: 32
    output_tokens: 16

  - name: short
    input_tokens: 128
    output_tokens: 32

  - name: balanced
    input_tokens: 512
    output_tokens: 128

  - name: long_prefill
    input_tokens: 2048
    output_tokens: 32

  - name: decode_heavy
    input_tokens: 128
    output_tokens: 256

  - name: long_balanced
    input_tokens: 2048
    output_tokens: 128
EOF
```

---

# 23\. 创建完整 Baseline Runner

创建：

```bash
cat > benchmarks/scripts/run_jetson_baseline.py <<'PY'
from __future__ import annotations

import datetime
import os
import subprocess
import sys

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)

os.chdir(ROOT)

run_id = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_dir = os.path.join(
    ROOT,
    "reports",
    "dev",
    "llm",
    run_id,
)

os.makedirs(
    output_dir,
    exist_ok=True,
)

cases = [
    ("tiny", 32, 16),

    ("short", 128, 32),

    ("balanced", 512, 128),

    ("long_prefill", 2048, 32),

    ("decode_heavy", 128, 256),

    ("long_balanced", 2048, 128),
]

print("Run ID:")
print(run_id)

print("Output:")
print(output_dir)

for name, isl, osl in cases:

    print()
    print("=" * 70)

    print(
        f"{name}: "
        f"ISL={isl}, "
        f"OSL={osl}"
    )

    print("=" * 70)

    output_path = os.path.join(
        output_dir,
        f"{name}.json",
    )

    command = [
        sys.executable,

        "benchmarks/scripts/"
        "run_model_core.py",

        "--input-tokens",
        str(isl),

        "--output-tokens",
        str(osl),

        "--repetitions",
        "3",

        "--output",
        output_path,
    ]

    subprocess.run(
        command,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH":
                ROOT
                + ":"
                + os.environ.get(
                    "PYTHONPATH",
                    "",
                ),
        },
    )

print()
print("Baseline finished.")
print(output_dir)
PY
```

---

# 24\. 正式运行前固定 Jetson 状态

保持与你之前 CUDA Baseline **完全相同的功耗模式**。

检查：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

如果之前正式 Baseline 使用了：

```text
MAXN_SUPER
+
jetson_clocks
```

则本次也保持一致。

不要中途修改。

---

# 25\. 启动 Tegrastats

打开第二个 SSH 窗口。

执行：

```bash
cd ~/work/HeteroQuantServeBench

mkdir -p reports/dev/llm/system
```

启动：

```bash
tegrastats \
  --interval 100 \
  --logfile \
  reports/dev/llm/system/tegrastats_baseline.log
```

保持运行。

---

# 26\. 执行第一轮完整 Baseline

第一个 SSH：

```bash
cd ~/work/HeteroQuantServeBench

source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python benchmarks/scripts/run_jetson_baseline.py
```

这一阶段可能持续较长时间。

运行完成以后，在第二个终端：

```text
Ctrl+C
```

停止 `tegrastats`。

---

# 27\. 检查全部结果文件

找最新目录：

```bash
LATEST="$(
  find reports/dev/llm \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name '20*' \
    -printf '%T@ %p\n' \
  | sort -nr \
  | head -1 \
  | cut -d' ' -f2-
)"

echo "$LATEST"
```

查看：

```bash
find "$LATEST" \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort
```

应看到：

```text
tiny.json
short.json
balanced.json
long_prefill.json
decode_heavy.json
long_balanced.json
```

---

# 28\. 检查所有 case 是否确定

```bash
python - <<'PY'
import glob
import json
import os

dirs = sorted(
    glob.glob(
        "reports/dev/llm/20*"
    )
)

latest = dirs[-1]

for path in sorted(
    glob.glob(
        os.path.join(
            latest,
            "*.json",
        )
    )
):

    with open(path) as f:
        data = json.load(f)

    print(
        os.path.basename(path),
        "deterministic=",
        data["deterministic"],
    )
PY
```

目标：

```text
tiny.json True
short.json True
balanced.json True
long_prefill.json True
decode_heavy.json True
long_balanced.json True
```

---

# 29\. 创建结果汇总程序

```bash
cat > benchmarks/scripts/summarize_baseline.py <<'PY'
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-dir",
    required=True,
)

args = parser.parse_args()

rows = []

for path in sorted(
    glob.glob(
        os.path.join(
            args.input_dir,
            "*.json",
        )
    )
):

    with open(
        path,
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    runs = data["repetitions"]

    def avg(key):
        return statistics.mean(
            run[key]
            for run in runs
        )

    rows.append({
        "case":
            os.path.basename(path)
            .replace(".json", ""),

        "input_tokens":
            data["workload"]
            ["input_tokens"],

        "output_tokens":
            data["workload"]
            ["output_tokens"],

        "prefill_ms":
            avg(
                "prefill_forward_ms"
            ),

        "model_core_ttft_ms":
            avg(
                "model_core_ttft_ms"
            ),

        "decode_total_ms":
            avg(
                "decode_total_ms"
            ),

        "e2e_ms":
            avg(
                "model_core_e2e_ms"
            ),

        "prefill_tps":
            avg(
                "prefill_tokens_per_s"
            ),

        "decode_tps":
            avg(
                "decode_tokens_per_s"
            ),

        "output_tps":
            avg(
                "model_core_output_tokens_per_s"
            ),

        "peak_cuda_allocated_mb":
            max(
                run[
                    "peak_cuda_allocated_mb"
                ]
                for run in runs
            ),

        "peak_cuda_reserved_mb":
            max(
                run[
                    "peak_cuda_reserved_mb"
                ]
                for run in runs
            ),

        "process_rss_mb":
            max(
                run[
                    "process_rss_mb"
                ]
                for run in runs
            ),

        "system_memory_used_mb":
            max(
                run[
                    "system_memory_used_mb"
                ]
                for run in runs
            ),

        "deterministic":
            data[
                "deterministic"
            ],
    })

output = os.path.join(
    args.input_dir,
    "summary.csv",
)

with open(
    output,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        rows
    )

print(output)
PY
```

执行：

```bash
python \
  benchmarks/scripts/summarize_baseline.py \
  --input-dir "$LATEST"
```

查看：

```bash
column -s, -t \
  < "$LATEST/summary.csv" \
  | less -S
```

---

# 30\. 第一版你必须确认这些指标已经存在

每个 workload 至少确认：

```text
Prefill
--------
prefill_forward_ms
prefill_tokens_per_s

First Token
-----------
model_core_ttft_ms
first_token_selection_ms

Decode
------
decode_total_ms

ITL mean
ITL median
ITL P50
ITL P95
ITL P99

decode_tokens_per_s

End-to-End
----------
model_core_e2e_ms
model_core_output_tokens_per_s

Memory
------
peak_cuda_allocated_mb
peak_cuda_reserved_mb
process_rss_mb
system_memory_used_mb

Reproducibility
---------------
model ID
ModelScope source
PyTorch version
ModelScope version
Transformers version
CUDA version
GPU
ISL
OSL
dtype
attention backend
```

---

# 31\. 下一步加入逐 Case 功耗监控

当前整个 Baseline 已经用 `tegrastats` 记录系统状态。

下一步把它自动集成到 Runner。

创建：

```bash
cat > hqsb/benchmark/resource_monitor.py <<'PY'
from __future__ import annotations

import subprocess
import threading
import time

class TegrastatsMonitor:

    def __init__(
        self,
        interval_ms: int = 100,
    ):
        self.interval_ms = interval_ms

        self.process = None

        self.records = []

        self.thread = None

    def start(self):

        self.process = subprocess.Popen(
            [
                "tegrastats",
                "--interval",
                str(
                    self.interval_ms
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True,
        )

        self.thread.start()

    def _reader(self):

        assert (
            self.process is not None
        )

        assert (
            self.process.stdout
            is not None
        )

        for line in self.process.stdout:

            self.records.append({
                "time_ns":
                    time.monotonic_ns(),

                "raw":
                    line.strip(),
            })

    def stop(self):

        if self.process is None:
            return

        self.process.terminate()

        try:
            self.process.wait(
                timeout=3
            )

        except subprocess.TimeoutExpired:
            self.process.kill()

        if self.thread:
            self.thread.join(
                timeout=1
            )
PY
```

---

# 32\. 先测试 Monitor

```bash
python - <<'PY'
import time

from hqsb.benchmark.resource_monitor \
    import TegrastatsMonitor

m = TegrastatsMonitor(
    interval_ms=100
)

m.start()

time.sleep(3)

m.stop()

print(
    "samples:",
    len(m.records)
)

for item in m.records[:3]:
    print(item)
PY
```

应该至少采集二十多条记录。

---

# 33\. 把 Tegrastats Monitor 接入单 Case Runner

在：

```text
benchmarks/scripts/run_model_core.py
```

增加：

```python
from hqsb.benchmark.resource_monitor import (
    TegrastatsMonitor,
)
```

在每次正式 repetition 前：

```python
monitor = TegrastatsMonitor(
    interval_ms=100
)

monitor.start()
```

Benchmark完成后：

```python
monitor.stop()

result[
    "tegrastats_samples"
] = monitor.records
```

这样以后每个 workload 的 JSON 自带：

```text
timestamp
+
tegrastats raw line
```

---

# 34\. 创建 Tegrastats Parser

第一阶段只解析最稳定的几个字段：

```bash
cat > hqsb/benchmark/tegrastats_parser.py <<'PY'
from __future__ import annotations

import re

RAM_PATTERN = re.compile(
    r"RAM\s+(\d+)/(\d+)MB"
)

GPU_PATTERN = re.compile(
    r"GR3D_FREQ\s+(\d+)%"
)

TEMP_PATTERNS = [
    re.compile(
        r"gpu@([\d.]+)C",
        re.IGNORECASE,
    ),

    re.compile(
        r"GPU@([\d.]+)C",
        re.IGNORECASE,
    ),
]

POWER_PATTERN = re.compile(
    r"VDD_IN\s+(\d+)mW"
)

def parse_tegrastats_line(
    line: str,
):

    result = {}

    ram = RAM_PATTERN.search(
        line
    )

    if ram:
        result[
            "ram_used_mb"
        ] = int(
            ram.group(1)
        )

    gpu = GPU_PATTERN.search(
        line
    )

    if gpu:
        result[
            "gpu_util_pct"
        ] = int(
            gpu.group(1)
        )

    for pattern in TEMP_PATTERNS:

        match = pattern.search(
            line
        )

        if match:

            result[
                "gpu_temp_c"
            ] = float(
                match.group(1)
            )

            break

    power = POWER_PATTERN.search(
        line
    )

    if power:

        result[
            "power_w"
        ] = (
            int(
                power.group(1)
            )
            / 1000.0
        )

    return result
PY
```

---

# 35\. 先查看你的 Tegrastats 实际格式

因为不同 JetPack 输出字段可能不同。

执行：

```bash
tegrastats --interval 1000
```

复制三五行。

检查是否实际包含：

```text
RAM
GR3D_FREQ
GPU@
VDD_IN
```

如果功耗Rails名称与你实际输出不同，**按你的真实输出修改 `POWER_PATTERN`**。

---

# 36\. 计算功耗与能耗指标

资源解析最终要输出：

```text
avg_power_w
peak_power_w
energy_j
energy_per_output_token_j

avg_gpu_utilization_pct
peak_gpu_utilization_pct

peak_gpu_temperature_c

peak_system_ram_mb
```

能量积分使用：

```text
每条记录自己的 monotonic timestamp
+
相邻采样点功率
```

即梯形积分。

不要直接拿：

```text
平均功率 × 粗略运行秒数
```

作为最终版本。

---

# 37\. 建立第一版 Golden Numerical Baseline

下一步做模型级数值回归。

建立：

```bash
mkdir -p \
  benchmarks/workloads/golden \
  reports/dev/llm/golden
```

先固定几个输入长度：

```text
32
128
512
2048
```

对于每一个至少保存：

```text
input token IDs

generated token IDs

first-token top-32 token IDs

first-token top-32 logits

first-token logits L2 norm
```

暂时不要保存整个 Decode 的全 Vocabulary logits。

---

# 38\. Golden Reference 的目的是以后这样调用

未来替换：

```text
PyTorch RMSNorm
        ↓
CUDA RMSNorm
```

以后立即跑：

```text
Golden FP16
vs
Custom CUDA
```

比较：

```text
top1 token agreement

generated token agreement

top-k overlap

max logit error

mean logit error

cosine similarity
```

只有通过 Model-level Regression 才跑性能测试。

---

# 39\. 第一版 Benchmark 文档

创建：

```bash
cat > docs/benchmark/methodology.md <<'EOF'
# HQSB LLM Benchmark Methodology

## Reference Model

Qwen/Qwen3-1.7B

Source:
ModelScope

## Reference Backend

ModelScope local model loading +
PyTorch execution.

## Reference Precision

FP16

## Attention

Eager

## Batch Size

1

## Performance Workloads

Fixed input token length and
fixed output token length.

## Performance Layers

1. Model Core
2. Operator
3. Online Serving
4. Numerical Regression
5. Quality Evaluation

## Current Stage

The current benchmark measures
model-core inference without:

- HTTP
- request queueing
- network transport
- serving scheduler

## Metrics

- prefill latency
- prefill throughput
- model-core TTFT
- decode ITL
- decode throughput
- end-to-end latency
- CUDA memory
- process memory
- system memory
- Jetson utilization
- power
- temperature
EOF
```

---

# 40\. 指标定义文件

```bash
cat > docs/benchmark/metric_definitions.md <<'EOF'
# Metric Definitions

## prefill_forward_ms

Time spent executing the initial
full-sequence model forward pass.

## model_core_ttft_ms

Prefill execution plus local
first-token selection.

It excludes HTTP, queueing,
network transport and client-side
processing.

## ITL

Inter-token latency after the
first generated token.

## decode_tokens_per_s

(output_tokens - 1) /
decode_seconds

## model_core_e2e_ms

model_core_ttft_ms +
decode_total_ms

## peak_cuda_allocated_mb

Peak memory allocated by the
PyTorch CUDA allocator.

## peak_cuda_reserved_mb

Peak memory reserved by the
PyTorch CUDA allocator.

## system_memory_used_mb

System-wide RAM usage.

This is especially important on
Jetson because CPU and GPU share
system memory.
EOF
```

---

# 41\. 本阶段暂时不要做这些

现在明确不做：

```text
× CUDA RMSNorm
× Triton
× AWQ
× GPTQ
× 自己的量化算法
× llama.cpp
× TensorRT-LLM
× vLLM
× SGLang
× CANN
× Ascend C
× NCCL
× lm-eval
```

全部等 Reference Benchmark 固定以后。

---

# 42\. 本阶段最后一次正式 Benchmark

完成：

```text
逐case tegrastats
+
parser
+
energy
+
golden
```

以后重新执行：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

确认设备状态与此前一致。

然后：

```bash
cd ~/work/HeteroQuantServeBench

source .venv/bin/activate

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python \
  benchmarks/scripts/run_jetson_baseline.py
```

---

# 43\. 最终检查六种 workload

必须全部完成：

```text
tiny
ISL 32
OSL 16

short
ISL 128
OSL 32

balanced
ISL 512
OSL 128

long_prefill
ISL 2048
OSL 32

decode_heavy
ISL 128
OSL 256

long_balanced
ISL 2048
OSL 128
```

全部：

```text
Batch = 1
FP16
Eager Attention
Qwen3-1.7B
ModelScope local artifact
```

---

# 44\. 本阶段最终验收清单

## ModelScope

```text
[ ] Qwen/Qwen3-1.7B从ModelScope下载成功
[ ] 完全使用本地模型路径
[ ] 断网状态下仍能加载模型
[ ] SHA256 Manifest生成
```

测试断网模式最简单的方法不是直接断SSH网络，而是保证所有加载调用都：

```python
local_files_only=True
```

然后正常运行。

## Model

```text
[ ] FP16加载成功
[ ] eager attention成功
[ ] Smoke Generation成功
[ ] 不OOM
```

## Workload

```text
[ ] ISL严格等于指定值
[ ] OSL严格等于指定值
[ ] Batch=1
[ ] 输出不因EOS提前终止
[ ] 重复结果确定
```

## Prefill

```text
[ ] prefill_forward_ms
[ ] prefill_tokens_per_s
```

## First token

```text
[ ] first_token_selection_ms
[ ] model_core_ttft_ms
```

## Decode

```text
[ ] decode_total_ms
[ ] ITL mean
[ ] ITL median
[ ] ITL P50
[ ] ITL P95
[ ] ITL P99
[ ] decode_tokens_per_s
```

## End-to-end

```text
[ ] model_core_e2e_ms
[ ] model_core_output_tokens_per_s
```

## Memory

```text
[ ] peak_cuda_allocated_mb
[ ] peak_cuda_reserved_mb
[ ] process_rss_mb
[ ] system_memory_used_mb
```

## Jetson

```text
[ ] tegrastats自动采样
[ ] avg GPU utilization
[ ] peak GPU utilization
[ ] peak temperature
[ ] avg power
[ ] peak power
[ ] energy J
[ ] energy J/output-token
```

## Numerical

```text
[ ] Golden inputs
[ ] Golden generated tokens
[ ] first-token Top-K logits
[ ] numerical baseline保存成功
```

## Reproducibility

```text
[ ] ModelScope模型ID
[ ] 模型本地SHA256
[ ] torch版本
[ ] CUDA版本
[ ] ModelScope版本
[ ] transformers版本
[ ] Power Mode
[ ] GPU型号
[ ] workload参数
```

---

# 45\. 这个阶段完成后的下一步

下一阶段才正式进入：

```text
Qwen3-1.7B FP16 Reference
            │
            ▼
      RMSNorm识别
            │
            ▼
    PyTorch原始RMSNorm
            │
            ▼
     CUDA RMSNorm V0
            │
            ▼
      Golden Regression
            │
            ▼
      Kernel Benchmark
            │
            ▼
替换进Qwen3完整模型
            │
            ▼
重新运行完全相同的6组Benchmark
            │
            ▼
比较
Kernel latency
Prefill
TTFT
TPOT
TPS
Memory
Power
J/token
Numerical error
```

也就是说，**这轮先把“尺子”做准，下一轮再开始优化被测对象。**

建议你严格按照 **1 → 44** 的顺序执行；其中最关键的检查点是 **第14步模型跑通、第21步确定性通过、第27步六组Baseline完整生成、第36步功耗指标正常、第38步Golden Reference建立**。这五个检查点都通过后，再进入CUDA算子库开发。