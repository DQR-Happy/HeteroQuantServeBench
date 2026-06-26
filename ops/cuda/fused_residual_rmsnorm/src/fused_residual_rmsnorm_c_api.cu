#include "hqsb/fused_residual_rmsnorm.h"
#include "rmsnorm_launchers.h"

#include <cuda_runtime.h>

namespace {
bool decode_dtype(int value, hqsb::DType* out) {
  if (value == 0) { *out = hqsb::DType::kFloat32; return true; }
  if (value == 1) { *out = hqsb::DType::kFloat16; return true; }
  return false;
}
bool decode_variant(int value, hqsb::FusedResidualVariant* out) {
  if (value < 0 || value > 3) return false;
  *out = static_cast<hqsb::FusedResidualVariant>(value);
  return true;
}
bool decode_semantic(int value, hqsb::FusedResidualSemantic* out) {
  if (value != 1) return false;
  *out = hqsb::FusedResidualSemantic::kStrictRoundedResidual;
  return true;
}
}  // namespace

extern "C" {

int hqsb_fused_residual_rmsnorm_forward_ex_c(
    const void* input, const void* residual, const void* weight,
    void* residual_out, void* output, long long rows, long long hidden,
    float epsilon, int dtype, int variant, int semantic, void* stream) {
  hqsb::DType dt;
  hqsb::FusedResidualVariant v;
  hqsb::FusedResidualSemantic s;
  if (!decode_dtype(dtype, &dt) || !decode_variant(variant, &v) ||
      !decode_semantic(semantic, &s)) return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(hqsb::fused_residual_rmsnorm_forward(
      input, residual, weight, residual_out, output, rows, hidden, epsilon,
      dt, v, s, reinterpret_cast<cudaStream_t>(stream)));
}

int hqsb_separate_residual_rmsnorm_forward_ex_c(
    const void* input, const void* residual, const void* weight,
    void* residual_out, void* output, long long rows, long long hidden,
    float epsilon, int dtype, int semantic, void* stream) {
  hqsb::DType dt;
  hqsb::FusedResidualSemantic s;
  if (!decode_dtype(dtype, &dt) || !decode_semantic(semantic, &s))
    return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(hqsb::separate_residual_rmsnorm_forward(
      input, residual, weight, residual_out, output, rows, hidden, epsilon,
      dt, s, reinterpret_cast<cudaStream_t>(stream)));
}

int hqsb_residual_add_forward_ex_c(const void* input, const void* residual,
                                   void* residual_out, long long rows,
                                   long long hidden, int dtype, void* stream) {
  hqsb::DType dt;
  if (!decode_dtype(dtype, &dt)) return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(hqsb::residual_add_forward(
      input, residual, residual_out, rows, hidden, dt,
      reinterpret_cast<cudaStream_t>(stream)));
}

int hqsb_fused_baseline_rmsnorm_forward_ex_c(
    const void* input, const void* weight, void* output, long long rows,
    long long hidden, float epsilon, int dtype, void* stream) {
  hqsb::DType dt;
  if (!decode_dtype(dtype, &dt) || input == nullptr || weight == nullptr ||
      output == nullptr || rows < 1 || hidden < 1 || !(epsilon > 0.0F)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(hqsb::rmsnorm_v2(
      input, weight, output, rows, hidden, epsilon, dt, 256,
      reinterpret_cast<cudaStream_t>(stream)));
}

long long hqsb_fused_residual_dynamic_shared_bytes_c(long long hidden,
                                                     int dtype, int variant) {
  hqsb::DType dt;
  hqsb::FusedResidualVariant v;
  if (!decode_dtype(dtype, &dt) || !decode_variant(variant, &v)) return -1;
  return static_cast<long long>(
      hqsb::fused_residual_dynamic_shared_bytes(hidden, dt, v));
}

int hqsb_fused_residual_occupancy_c(long long hidden, int dtype, int variant) {
  hqsb::DType dt;
  hqsb::FusedResidualVariant v;
  if (!decode_dtype(dtype, &dt) || !decode_variant(variant, &v)) return 0;
  return hqsb::fused_residual_occupancy(hidden, dt, v);
}

}  // extern "C"
