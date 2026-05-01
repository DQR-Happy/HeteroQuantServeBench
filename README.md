# HeteroQuantServeBench

Heterogeneous quantization, kernel optimization, serving, and benchmarking
platform for NVIDIA CUDA and Huawei Ascend C/CANN backends.

HQSB 统一管理模型制品、工作负载、算子、量化、运行时、服务、通信、编译与跨硬件
benchmark，使每一次优化都能从 Kernel 追踪到模型、服务和硬件收益。

> **Profile before optimize；correctness before performance；evidence before claim。**

## Current stage

当前阶段为 **S01（核心契约与工程质量体系）**，已完成。`hqsb/core` 提供了稳定的
C1–C7 Contract、注册机制、统一配置/错误/日志体系、版本化 schema 与迁移、Python
打包与测试金字塔。

阶段路线图见 [`docs/architecture/顶层架构.md`](docs/architecture/顶层架构.md) 与
[`docs/stages/`](docs/stages/)。模块边界与依赖规则见
[`docs/architecture/module_ownership.md`](docs/architecture/module_ownership.md)。

## Quick start

目标环境：NVIDIA Jetson Orin Nano Super 8GB（sm_87）、JetPack 6 / L4T R36.4.3、
CUDA 12.6、Python 3.10、PyTorch 2.5（NVIDIA 构建）。

### 0. 安装（可选依赖组）

```bash
pip install -e .              # 仅核心：pydantic + PyYAML（CPU 可跑全部测试）
pip install -e ".[dev]"       # 附加 pytest/ruff/mypy
pip install -e ".[benchmark]" # 附加 torch/transformers/modelscope（S02）
```

### 1. CPU 单元测试（无需 GPU / 模型权重）

```bash
python3 -m pytest -q                 # 全量（含 S00/S01，166 passed）
python3 -m pytest -m unit -q         # 纯单元测试
python3 -m pytest -m property -q     # 属性/不变量测试
```

### 2. CUDA device query 与 RMSNorm baseline

```bash
cmake -S . -B build/jetson-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build/jetson-release --parallel

build/jetson-release/bin/hqsb_device_query
build/jetson-release/bin/hqsb_rmsnorm_baseline --rows 512 --hidden 1024
```

RMSNorm baseline 会在输出末尾打印 `correctness=PASS`（`max_abs_error <= 5e-4`）。

### 3. Qwen3-1.7B model-core smoke

```bash
# 下载模型（与 benchmark 严格分离）
python3 scripts/models/download_qwen3_modelscope.py

# 校验本地快照 SHA256（artifact integrity gate）
python3 scripts/models/verify_qwen3_hashes.py

# 最小加载 + 生成 smoke
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 scripts/models/smoke_qwen3.py
```

完整六 workload model-core baseline：

```bash
python3 benchmarks/scripts/run_jetson_baseline.py
```

## Reproducibility

所有 benchmark 结果绑定：Git commit、模型与 tokenizer 的 SHA256 manifest、配置、
环境（nvpmodel/clock/温度）、PyTorch/CUDA/Transformers/ModelScope 版本与 seed。
原始 sample 保存于 `reports/` 并以 artifact 索引引用，summary 不可反推 raw。

## Status matrix

图例：**Implemented** = 源码存在且被测试或运行验证；**Verified** = 有运行证据；
**Experimental** = 存在但未闭环；**Planned** = 仅有规划。

| 能力 | 状态 | 位置 | 证据 |
|---|---|---|---|
| CUDA device query | Verified | `ops/cuda/device_query/` | `reports/jetson/20260808_010256/device_query.txt` |
| CUDA RMSNorm V0（shared reduction） | Verified | `ops/cuda/rmsnorm/` | `reports/jetson/20260808_010256/rmsnorm_runs.txt` |
| Qwen3-1.7B model loader（local-only） | Verified | `hqsb/models/loader.py` | `reports/dev/llm/` 各 run |
| 模型 SHA256 manifest 校验 | Implemented | `hqsb/models/manifest.py`、`scripts/models/verify_qwen3_hashes.py` | `tests/unit/test_manifest.py` |
| model-core benchmark 引擎 | Verified | `hqsb/benchmark/model_core.py` | `reports/dev/llm/20260812_094129/` |
| tegrastats monitor/parser | Implemented | `hqsb/benchmark/resource_monitor.py`、`tegrastats_parser.py` | `tests/unit/test_tegrastats_parser.py` |
| 六 workload baseline orchestrator | Verified | `benchmarks/scripts/run_jetson_baseline.py` | `reports/dev/llm/20260812_094129/summary.csv` |
| golden 数值回归基线（4 份） | Implemented | `benchmarks/workloads/golden/` | 已迁移到 C6 schema（S01） |
| C1–C7 版本化 Contract | Implemented | `hqsb/core/contracts/` | `tests/unit/core/test_contracts.py` |
| 统一配置加载 + hash | Implemented | `hqsb/core/config/loader.py` | `tests/unit/core/test_config.py` |
| 插件注册表 Registry | Implemented | `hqsb/core/registry/registry.py` | `tests/unit/core/test_registry.py` |
| 错误分类 + exit code | Implemented | `hqsb/core/errors.py` | `tests/unit/core/test_errors.py` |
| Schema 版本化 + legacy 迁移 | Implemented | `hqsb/core/schema/` | `tests/unit/core/test_migration.py` |
| Dummy backend（C4 参考实现） | Implemented | `hqsb/backends/dummy.py` | `tests/unit/core/test_dummy_backend.py` |
| backend-interface benchmark engine | Implemented | `hqsb/benchmark/engine.py` | `tests/unit/core/test_dummy_backend.py` |
| CPU 单元测试 | Implemented | `tests/` | `pytest -q`（166 passed） |
| QuantLab（RTN/GPTQ/AWQ/SmoothQuant） | Planned | `hqsb/quant/` | — |
| KernelLab（Triton/Ascend C） | Planned | `ops/triton/`、`ops/ascend/` | — |
| Runtime adapters（PyTorch/vLLM/TensorRT/llama.cpp） | Planned | `hqsb/backends/`（仅 dummy） | S02/S07 |
| ServeFabric（OpenAI-compatible gateway） | Planned | `hqsb/serving/` | — |
| BenchLab（跨硬件统一 benchmark） | Planned | `benchmarks/` | — |

## Planned components

- **QuantLab**：RTN、GPTQ、AWQ、SmoothQuant 与论文方法复现
- **KernelLab**：CUDA、Triton 与 Ascend C 算子
- **Runtime adapters**：TensorRT、llama.cpp、vLLM 与 Ascend 运行时
- **ServeFabric**：OpenAI-compatible 异构推理网关
- **BenchLab**：latency、throughput、memory、accuracy、energy 报告

## Hardware

- NVIDIA Jetson Orin Nano Super 8GB
- Orange Pi AI Pro 20T
- 按需 NVIDIA datacenter/desktop GPU 实例

## Repository integrity

- 模型权重、`*.safetensors`/`*.gguf`/`*.onnx`/`*.engine` 一律被 `.gitignore` 排除；
- SSH 私钥（`id_rsa*`、`id_ed25519*`）、`.env`、`credentials*`、`secrets/` 禁止入库；
- 默认配置不含机器专属绝对路径（模型路径经 `~` 展开）。
