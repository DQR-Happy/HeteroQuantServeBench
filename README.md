# HeteroQuantServeBench

Heterogeneous quantization, kernel optimization, serving, and benchmarking
platform for NVIDIA CUDA and Huawei Ascend C/CANN backends.

## Current milestone

Milestone 0 focuses on the Jetson Orin Nano Super CUDA baseline:

- reproducible hardware/software manifest;
- CUDA device discovery;
- CMake-based CUDA build;
- CPU reference and CUDA RMSNorm baseline;
- correctness and latency measurements;
- tegrastats/Nsight-ready profiling workflow.

## Planned components

- QuantLab: RTN, GPTQ, AWQ, SmoothQuant and paper-method reproduction
- KernelLab: CUDA, Triton and Ascend C operators
- Runtime adapters: TensorRT, llama.cpp, vLLM and Ascend runtimes
- ServeFabric: OpenAI-compatible heterogeneous inference gateway
- BenchLab: latency, throughput, memory, accuracy and energy reports

## Hardware

- NVIDIA Jetson Orin Nano Super 8GB
- Orange Pi AI Pro 20T
- On-demand NVIDIA datacenter/desktop GPU instances

## Status

Work in progress.
