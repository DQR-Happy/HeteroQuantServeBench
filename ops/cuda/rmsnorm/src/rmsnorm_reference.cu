#include "hqsb/rmsnorm.h"

#include <cmath>

namespace hqsb {

namespace {

// FP64 reference implementation used by both tests and the benchmark. The
// reference is intentionally the *only* copy of the RMSNorm math that does
// not run on the GPU, so it is independent of the kernel code under test.
void reference_fp64(const float* input,
                    const float* weight,
                    float* output,
                    int64_t rows,
                    int64_t hidden,
                    float epsilon) {
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t offset = row * hidden;
    double square_sum = 0.0;
    for (int64_t col = 0; col < hidden; ++col) {
      const double value = static_cast<double>(input[offset + col]);
      square_sum += value * value;
    }
    const double inverse_rms =
        1.0 / std::sqrt(square_sum / static_cast<double>(hidden) +
                        static_cast<double>(epsilon));
    for (int64_t col = 0; col < hidden; ++col) {
      output[offset + col] = static_cast<float>(
          static_cast<double>(input[offset + col]) * inverse_rms *
          static_cast<double>(weight[col]));
    }
  }
}

}  // namespace

void rmsnorm_reference_cpu(const float* input,
                           const float* weight,
                           float* output,
                           int64_t rows,
                           int64_t hidden,
                           float epsilon) {
  reference_fp64(input, weight, output, rows, hidden, epsilon);
}

const char* rmsnorm_variant_name(RmsNormVariant variant) {
  switch (variant) {
    case RmsNormVariant::kAuto:
      return "auto";
    case RmsNormVariant::kReference:
      return "reference";
    case RmsNormVariant::kV0Shared:
      return "v0_shared";
    case RmsNormVariant::kV1WarpShuffle:
      return "v1_warp_shuffle";
    case RmsNormVariant::kV2Vectorized:
      return "v2_vectorized";
  }
  return "unknown";
}

}  // namespace hqsb
