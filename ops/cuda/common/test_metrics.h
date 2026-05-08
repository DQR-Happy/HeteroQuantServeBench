#pragma once
//
// Numerical comparison metrics shared by the operator correctness suites.
//
// Every kernel is compared against the FP64 CPU reference using five
// standard metrics so a regression is characterized precisely, not just by
// a pass/fail flag:
//   max_abs_error, mean_abs_error, rmse, cosine_similarity, l2_relative_error.
//

#include <cmath>
#include <cstddef>

namespace hqsb::test {

struct Metrics {
  double max_abs_error = 0.0;
  double mean_abs_error = 0.0;
  double rmse = 0.0;
  double cosine_similarity = 1.0;
  double l2_relative_error = 0.0;
};

inline Metrics compute_metrics(const float* actual,
                               const float* expected,
                               size_t n) {
  Metrics m;
  if (n == 0) {
    return m;
  }

  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double dot = 0.0;
  double norm_actual = 0.0;
  double norm_expected = 0.0;

  for (size_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(actual[i]);
    const double e = static_cast<double>(expected[i]);
    const double diff = std::fabs(a - e);

    m.max_abs_error = std::max(m.max_abs_error, diff);
    sum_abs += diff;
    sum_sq += diff * diff;
    dot += a * e;
    norm_actual += a * a;
    norm_expected += e * e;
  }

  m.mean_abs_error = sum_abs / static_cast<double>(n);
  m.rmse = std::sqrt(sum_sq / static_cast<double>(n));

  const double denom = std::sqrt(norm_actual) * std::sqrt(norm_expected);
  m.cosine_similarity = (denom > 0.0) ? dot / denom : 1.0;

  const double ref_norm = std::sqrt(norm_expected);
  m.l2_relative_error =
      (ref_norm > 0.0) ? std::sqrt(sum_sq) / ref_norm : 0.0;

  return m;
}

}  // namespace hqsb::test
