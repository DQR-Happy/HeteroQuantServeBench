"""Tests for the low-bit execution layer (ops/quant).

CPU-safe tests run always; the fused-dequant Triton correctness test is marked
with ``cuda`` and skipped when the device is absent (the kernel is a real
low-bit path, but its absence is a *capability* state, not a test failure —
E05-06 §6 "本机能力").
"""

from __future__ import annotations

import os
import tempfile

import pytest

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.quant import compat
from hqsb.quant.fixtures import build_tiny_artifact


@pytest.mark.unit
class TestCapabilityProbe:
    def test_probe_returns_structured_report(self):
        from ops.quant.capability import probe_low_bit_capability

        report = probe_low_bit_capability(compile_probe=False).as_dict()
        assert "torch_available" in report
        assert "cuda_available" in report
        assert "triton_available" in report
        # A capability report never raises and always has a reason field.
        assert "reasons" in report

    def test_probe_to_kernel_capability(self):
        from ops.quant.capability import kernel_capability_from_probe, probe_low_bit_capability

        capability = kernel_capability_from_probe(probe_low_bit_capability(compile_probe=False))
        assert isinstance(capability, compat.KernelCapability)
        if not capability.available:
            assert capability.unavailable_reason


@pytest.mark.unit
class TestExecutorRegistry:
    def test_all_executors_resolvable(self):
        from ops.quant.executors import (
            EXECUTOR_FP16,
            EXECUTOR_FUSED,
            EXECUTOR_STORAGE,
            get_executor,
        )

        assert get_executor(EXECUTOR_FP16).name == EXECUTOR_FP16
        assert get_executor(EXECUTOR_STORAGE).name == EXECUTOR_STORAGE
        assert get_executor(EXECUTOR_FUSED).name == EXECUTOR_FUSED

    def test_unknown_executor_refused(self):
        from ops.quant.executors import get_executor

        with pytest.raises(ConfigError):
            get_executor("nope")

    def test_matrix_lists_all(self):
        from ops.quant.executors import EXECUTOR_REGISTRY, executor_matrix

        rows = executor_matrix()
        assert {row["executor"] for row in rows} == set(EXECUTOR_REGISTRY)

    def test_fp16_reference_refuses_packed_input(self):
        from ops.quant.executors import Fp16ReferenceExecutor

        with pytest.raises(CapabilityError):
            Fp16ReferenceExecutor().dequantize_weight(build_tiny_artifact())

    def test_labels_are_distinct(self):
        from ops.quant.executors import (
            Fp16ReferenceExecutor,
            FusedDequantExecutor,
            StorageOnlyExecutor,
        )

        assert Fp16ReferenceExecutor.execution_label == "fp16_reference"
        assert StorageOnlyExecutor.execution_label == "storage_only"
        assert FusedDequantExecutor.execution_label == "fused_dequant_weight_only"
        assert FusedDequantExecutor.low_bit is True
        assert StorageOnlyExecutor.low_bit is False


@pytest.mark.unit
class TestPrepareWeights:
    def test_unsupported_layout_refused(self, tmp_path):
        doc = build_tiny_artifact(bits=4)
        doc.variants[0].layout_id = "hqsb.unknown.layout.v1"
        from ops.quant.w4a16_triton import prepare_weights

        path = os.path.join(str(tmp_path), "artifact")
        doc.save(path)
        from hqsb.quant.artifact import load_document

        loaded = load_document(path, verify_variants=False)
        with pytest.raises(CapabilityError, match="not implemented"):
            prepare_weights(loaded, loaded.variants[0], device="cpu")

    def test_payload_length_mismatch_refused(self, tmp_path):
        doc = build_tiny_artifact(bits=4)
        path = os.path.join(str(tmp_path), "artifact")
        doc.save(path)
        from hqsb.quant.artifact import load_document

        loaded = load_document(path, verify_variants=False)
        # Corrupt the recorded row stride so the payload no longer matches.
        loaded.variants[0].layout_extra = {"layout": {"row_stride_bytes": 999}}
        from ops.quant.w4a16_triton import prepare_weights

        with pytest.raises(ConfigError):
            prepare_weights(loaded, loaded.variants[0], device="cpu")


@pytest.mark.unit
class TestMicrobench:
    def test_benchmark_callable_host_clock(self):
        from ops.quant.microbench import benchmark_callable

        result = benchmark_callable(lambda: sum(range(1000)), name="cpu", warmup=1, repeats=3, use_cuda_events=False)
        assert len(result.samples) == 3
        assert result.summary()["count"] == 3
        assert result.notes  # host clock label is recorded

    def test_dequant_decomposition_plan(self):
        from ops.quant.microbench import dequant_decomposition_plan

        plan = dequant_decomposition_plan()
        assert {p["path"] for p in plan} == {
            "unpack_only",
            "dequant_to_buffer",
            "dequant_to_buffer_plus_fp16_gemm",
            "fused_dequant_gemm",
            "vendor_fused",
            "fp16_gemm",
        }


@pytest.mark.unit
class TestSafety:
    def test_sanitizer_report_never_fabricates(self):
        from ops.quant.safety import sanitizer_command, sanitizer_report

        command = sanitizer_command("python", ["script.py"])
        report = sanitizer_report(command)
        assert report["command"] == command
        # "available" is probed, never assumed true.
        assert isinstance(report["available"], bool)


_HAS_CUDA = False
try:
    import torch

    _HAS_CUDA = bool(torch.cuda.is_available())
except ImportError:
    pass


@pytest.mark.hardware
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA device required for the fused-dequant kernel")
class TestFusedDequantKernel:
    def test_compile_probe(self):
        from ops.quant.w4a16_triton import compile_probe

        probe = compile_probe()
        assert probe["ok"] is True
        assert probe["kernel_symbol"] == "hqsb_dequant_gemm_kernel"

    def test_kernel_matches_oracle(self):
        import torch

        from hqsb.quant import oracle
        from hqsb.quant.packing import pack_kernel_variant
        from hqsb.quant.rtn import quantize
        from hqsb.quant.spec import RangePolicy
        from ops.quant.w4a16_triton import gemm_low_bit, prepare_weights

        for bits, symmetric in ((4, True), (4, False), (8, True)):
            scheme = build_tiny_artifact(bits=bits).scheme
            if bits == 4 and not symmetric:
                scheme = scheme.with_overrides(
                    symmetric=False, range_policy=RangePolicy.TWOS_COMPLEMENT
                )
            weight = torch.randn(8, 64, dtype=torch.float64)
            qt = quantize(
                [float(v) for v in weight.reshape(-1).tolist()],
                scheme,
                shape=(8, 64),
            )
            from hqsb.quant.artifact import ModelIdentity, QuantArtifactDocument

            doc = QuantArtifactDocument.from_quantized(
                qt, tensor_name="w", model=ModelIdentity(model_id="m", revision="r")
            )
            doc.add_variant(pack_kernel_variant(qt.q, qt.scales, qt.zeros, qt.scheme, 8, 64))
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "artifact")
                doc.save(path)
                from hqsb.quant.artifact import load_document

                loaded = load_document(path)
                x = torch.randn(3, 64, dtype=torch.float16, device="cuda")
                prepared = prepare_weights(loaded, loaded.variants[0])
                y = gemm_low_bit(x, prepared)
                reference = oracle.kernel_oracle(x.cpu(), loaded, loaded.variants[0])
                report = oracle.correctness_report(
                    reference,
                    y.cpu(),
                    tolerance={"max_abs": 5e-2, "relative_l2": 1e-2, "cosine_min": 0.999},
                )
                assert report["passed"] is True, report

    def test_wrong_activation_width_refused(self):
        import torch

        from hqsb.quant.artifact import load_document
        from ops.quant.executors import get_executor

        doc = build_tiny_artifact(bits=4, rows=8, cols=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "artifact")
            doc.save(path)
            loaded = load_document(path)
            executor = get_executor("fused_dequant")
            x = torch.randn(2, 32, dtype=torch.float16, device="cuda")
            with pytest.raises(ConfigError):
                executor.gemm_low_bit(x, loaded, loaded.variants[0])

    def test_storage_only_matches_fused(self):
        import torch

        from ops.quant.executors import get_executor

        doc = build_tiny_artifact(bits=4, rows=8, cols=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "artifact")
            doc.save(path)
            from hqsb.quant.artifact import load_document

            loaded = load_document(path)
            x = torch.randn(3, 64, dtype=torch.float16, device="cuda")
            fused = get_executor("fused_dequant").gemm_low_bit(x, loaded, loaded.variants[0])
            storage = get_executor("storage_only").gemm_low_bit(x, loaded, loaded.variants[0])
            assert torch.allclose(fused, storage, atol=5e-2)
