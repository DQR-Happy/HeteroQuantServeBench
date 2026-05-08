#pragma once
//
// Minimal, dependency-free test harness for the CUDA operator library.
//
// The suite runs on a Jetson device where heavyweight test frameworks are
// unnecessary; this header provides expect/assert macros that accumulate
// failures and produce a deterministic non-zero exit code, plus CTest-
// friendly output.
//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace hqsb::test {

struct Suite {
  int checks = 0;
  int failures = 0;
};

inline Suite& suite() {
  static Suite s;
  return s;
}

inline void report_failure(const char* file, int line, const std::string& msg) {
  Suite& s = suite();
  ++s.checks;
  ++s.failures;
  std::fprintf(stderr, "  [FAIL] %s:%d  %s\n", file, line, msg.c_str());
}

inline int finish(const char* name) {
  Suite& s = suite();
  if (s.failures == 0) {
    std::printf("PASS  %s  (%d checks)\n", name, s.checks);
    return EXIT_SUCCESS;
  }
  std::fprintf(stderr, "FAIL  %s  (%d/%d checks failed)\n",
               name, s.failures, s.checks);
  return EXIT_FAILURE;
}

}  // namespace hqsb::test

#define CHECK(cond)                                                       \
  do {                                                                    \
    ::hqsb::test::suite().checks++;                                        \
    if (!(cond)) {                                                        \
      ::hqsb::test::suite().failures++;                                    \
      std::fprintf(stderr, "  [FAIL] %s:%d  %s\n", __FILE__, __LINE__,     \
                   #cond);                                                 \
    }                                                                     \
  } while (0)

#define CHECK_NEAR(actual, expected, tol)                                 \
  do {                                                                    \
    ::hqsb::test::suite().checks++;                                        \
    const double a_ = static_cast<double>(actual);                         \
    const double e_ = static_cast<double>(expected);                       \
    if (!(std::fabs(a_ - e_) <= static_cast<double>(tol))) {               \
      ::hqsb::test::suite().failures++;                                    \
      std::fprintf(stderr,                                                \
                   "  [FAIL] %s:%d  %s: %.6g !~= %.6g (tol %.2g)\n",       \
                   __FILE__, __LINE__, #actual, a_, e_,                    \
                   static_cast<double>(tol));                              \
    }                                                                     \
  } while (0)
