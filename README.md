# HeteroQuantServeBench

Heterogeneous quantization, kernel optimization, serving, and benchmarking
platform for NVIDIA CUDA and Huawei Ascend C/CANN backends.

HQSB 统一管理模型制品、工作负载、算子、量化、运行时、服务、通信、编译与跨硬件
benchmark，使每一次优化都能从 Kernel 追踪到模型、服务和硬件收益。

> **Profile before optimize；correctness before performance；evidence before claim。**

## Current stage

当前阶段为 **S04（Triton、CUTLASS/CuTe 与 Kernel DSL）**，已完成。实测 Triton
3.7.1 在 Jetson sm_87 可用，实现 Triton RMSNorm/GEMM + 统一 dispatcher/
capability（CUDA/Triton/cuBLAS/CUTLASS 按 arch/dtype/shape/依赖选择，未安装 DSL
走明确 fallback）。CUDA vs Triton 与 cuBLAS vs Triton 的性能权衡见
`docs/reports/S04_comparison_report.md`。

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
python3 -m pytest -q                 # 全量（E00-06 复跑 HEAD e4a031c：340 passed）
python3 -m pytest -m unit -q         # 纯单元测试
python3 -m pytest -m property -q     # 属性/不变量测试
```

### 2. CUDA 算子库（RMSNorm 多版本 + fused + correctness + benchmark）

```bash
cmake -S . -B build/jetson-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build/jetson-release --parallel

# correctness（CTest 集成）
ctest --test-dir build/jetson-release --output-on-failure

# benchmark：RMSNorm V0/V1/V2 扫参
build/jetson-release/bin/hqsb_rmsnorm_bench \
  --rows 512 --hidden 2048 --dtype fp32 --variant all

# benchmark：fused residual+rmsnorm
build/jetson-release/bin/hqsb_fused_residual_rmsnorm_bench \
  --rows 512 --hidden 2048 --dtype fp32 --variant all
```

算子库公共 API 见 `ops/cuda/rmsnorm/include/hqsb/rmsnorm.h`，V0/V1/V2 性能数据
与退化分析见 `docs/reports/S03_benchmark_report.md`。

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
| PyTorchBackend（FP16 Reference Runtime） | Verified | `hqsb/backends/pytorch.py` | `reports/dev/llm/s02_smoke_tiny.json`（decode 9.28 tok/s） |
| Roofline/Amdahl 分析 | Implemented | `hqsb/benchmark/roofline.py` | `tests/unit/core/test_roofline.py` |
| golden/determinism/首错位对比 | Implemented | `hqsb/benchmark/correctness.py` | `tests/unit/core/test_correctness.py` |
| YAML workload 单一事实源 | Implemented | `hqsb/benchmark/workload_config.py` | `tests/unit/core/test_workload_config.py` |
| KV cache/内存核算 | Implemented | `hqsb/benchmark/memory.py` | `tests/unit/core/test_memory.py` |
| PyTorch Profiler 采集 | Verified | `hqsb/benchmark/profiling.py` | `reports/dev/profiler/s02/hotspot_summary.json` |
| Jetson 实验协议 | Implemented | `hqsb/hardware/jetson.py` | `tests/unit/core/test_jetson.py` |
| RMSNorm 算子库（V0/V1/V2 + dispatcher） | Verified | `ops/cuda/rmsnorm/` | `ctest`（33 checks）；V2 +56%~142% |
| Fused residual+rmsnorm 算子 | Verified | `ops/cuda/fused_residual_rmsnorm/` | `ctest`（15 checks） |
| CUDA 测试框架 + 数值指标 | Implemented | `ops/cuda/common/{test_util,test_metrics}.h` | CTest 集成 |
| 后端能力检测（Triton 实测编译 probe） | Implemented | `ops/capability.py` | `tests/unit/ops/test_capability.py` |
| CUDA shared lib ctypes 绑定 | Implemented | `ops/cuda_bridge.py` | `tests/unit/ops/test_cuda_bridge.py` |
| 统一 dispatcher（CUDA/Triton/cuBLAS/CUTLASS fallback） | Implemented | `ops/dispatcher.py` | `tests/unit/ops/test_dispatcher.py` |
| Triton RMSNorm（reference + autotune） | Verified | `ops/triton/rmsnorm.py` | FP16 hidden=1024 反超 CUDA 36% |
| Triton GEMM（reference + autotune） | Verified | `ops/triton/gemm.py` | 窄矩阵反超 cuBLAS |
| CPU 单元测试 | Implemented | `tests/` | `pytest -q`（E00-06 复跑 e4a031c：340 passed） |
| QuantLab（RTN/GPTQ/AWQ/SmoothQuant） | Planned | `hqsb/quant/` | S05 |
| KernelLab（CUTLASS/Ascend C） | Planned | `ops/ascend/`（CUTLASS 待网络恢复） | S05/S09 |
| Runtime adapters（vLLM/TensorRT/llama.cpp） | Planned | `hqsb/backends/`（dummy/pytorch 已有） | S07 |
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
