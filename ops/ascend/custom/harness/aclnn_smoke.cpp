/**
 * aclnn smoke harness for the HQSB Ascend C operators (S09 E09-02/03).
 *
 * Runs HqsbAdd / HqsbRowReduceSum / HqsbRmsNorm through their generated aclnn
 * entry points on a real device and compares each result against an
 * independent CPU reference computed in double precision.
 *
 * What this proves: the operator package compiles, installs, loads and executes
 * the kernels on Ascend 310B, and the numbers agree with the frozen semantics
 * (including RMSNorm's epsilon position).  What it does NOT prove: performance,
 * bandwidth or energy -- those belong to E09-04 and msprof.
 *
 * The reference is computed here in plain C++ rather than by calling another
 * framework op, so a shared bug cannot cancel itself out.
 */
#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "aclnn/acl_meta.h"
#include "aclnn/opdev/op_errno.h"  // ACLNN_SUCCESS

#include "aclnn_hqsb_add.h"
#include "aclnn_hqsb_rms_norm.h"
#include "aclnn_hqsb_row_reduce_sum.h"

namespace {

constexpr float kTolerance = 1e-4f;

#define HQSB_ACL_CHECK(expr, what)                                                       \
    do {                                                                                 \
        const aclError hqsbStatus = (expr);                                              \
        if (hqsbStatus != ACL_SUCCESS) {                                                 \
            std::printf("FAIL  %-24s | acl error %d\n", (what), static_cast<int>(hqsbStatus)); \
            return false;                                                                \
        }                                                                                \
    } while (false)

aclTensor *MakeTensor(const std::vector<int64_t> &shape, void *deviceData)
{
    return aclCreateTensor(
        shape.data(), shape.size(), ACL_FLOAT,
        nullptr, 0, ACL_FORMAT_ND,
        shape.data(), shape.size(), deviceData);
}

bool Upload(const std::vector<float> &host, void **device, size_t bytes)
{
    HQSB_ACL_CHECK(aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    HQSB_ACL_CHECK(aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    return true;
}

bool Download(void *device, std::vector<float> &host, size_t bytes)
{
    HQSB_ACL_CHECK(aclrtMemcpy(host.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    return true;
}

void Report(const char *label, bool ok, float maxDiff, const char *detail)
{
    std::printf("%s  %-16s | max|diff|=%.3e %s\n", ok ? "PASS" : "FAIL", label, maxDiff, detail);
}

// ── HqsbAdd: y = x1 + x2 ────────────────────────────────────────────────────
bool TestAdd(aclrtStream stream)
{
    constexpr int64_t kTotal = 1024;
    const std::vector<int64_t> shape = {kTotal};
    const size_t bytes = static_cast<size_t>(kTotal) * sizeof(float);

    std::vector<float> hostX1(kTotal), hostX2(kTotal), hostY(kTotal, 0.0f);
    for (int64_t index = 0; index < kTotal; ++index) {
        hostX1[index] = static_cast<float>(index) * 0.5f;
        hostX2[index] = static_cast<float>(index) * 0.25f;
    }

    void *deviceX1 = nullptr;
    void *deviceX2 = nullptr;
    void *deviceY = nullptr;
    if (!Upload(hostX1, &deviceX1, bytes) || !Upload(hostX2, &deviceX2, bytes)) {
        return false;
    }
    HQSB_ACL_CHECK(aclrtMalloc(&deviceY, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "add.alloc.y");

    aclTensor *x1 = MakeTensor(shape, deviceX1);
    aclTensor *x2 = MakeTensor(shape, deviceX2);
    aclTensor *y = MakeTensor(shape, deviceY);
    if (x1 == nullptr || x2 == nullptr || y == nullptr) {
        std::printf("FAIL  add              | aclCreateTensor returned null\n");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnHqsbAddGetWorkspaceSize(x1, x2, y, &workspaceSize, &executor);
    if (status != ACLNN_SUCCESS) {
        std::printf("FAIL  add              | GetWorkspaceSize error %d\n", static_cast<int>(status));
        return false;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        HQSB_ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "add.alloc.ws");
    }
    status = aclnnHqsbAdd(workspace, workspaceSize, executor, stream);
    if (status != ACLNN_SUCCESS) {
        std::printf("FAIL  add              | launch error %d\n", static_cast<int>(status));
        return false;
    }
    HQSB_ACL_CHECK(aclrtSynchronizeStream(stream), "add.sync");
    if (!Download(deviceY, hostY, bytes)) {
        return false;
    }

    float maxDiff = 0.0f;
    for (int64_t index = 0; index < kTotal; ++index) {
        const double expect = static_cast<double>(hostX1[index]) + static_cast<double>(hostX2[index]);
        maxDiff = std::fmax(maxDiff, static_cast<float>(std::fabs(hostY[index] - expect)));
    }
    const bool ok = maxDiff <= kTolerance;
    Report("add", ok, maxDiff, "(x1 + x2)");

    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(y);
    aclrtFree(deviceX1);
    aclrtFree(deviceX2);
    aclrtFree(deviceY);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ok;
}

// ── HqsbRowReduceSum: y_r = sum_j x[r, j] ───────────────────────────────────
bool TestRowReduceSum(aclrtStream stream)
{
    constexpr int64_t kRows = 8;
    constexpr int64_t kHidden = 256;
    const std::vector<int64_t> inShape = {kRows, kHidden};
    const std::vector<int64_t> outShape = {kRows};
    const size_t inBytes = static_cast<size_t>(kRows * kHidden) * sizeof(float);
    const size_t outBytes = static_cast<size_t>(kRows) * sizeof(float);

    std::vector<float> hostX(kRows * kHidden);
    std::vector<float> hostY(kRows, 0.0f);
    for (int64_t index = 0; index < kRows * kHidden; ++index) {
        hostX[index] = static_cast<float>((index % 17) - 8) * 0.125f;
    }

    void *deviceX = nullptr;
    void *deviceY = nullptr;
    if (!Upload(hostX, &deviceX, inBytes)) {
        return false;
    }
    HQSB_ACL_CHECK(aclrtMalloc(&deviceY, outBytes, ACL_MEM_MALLOC_HUGE_FIRST), "reduce.alloc.y");

    aclTensor *x = MakeTensor(inShape, deviceX);
    aclTensor *y = MakeTensor(outShape, deviceY);
    if (x == nullptr || y == nullptr) {
        std::printf("FAIL  row_reduce_sum   | aclCreateTensor returned null\n");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnHqsbRowReduceSumGetWorkspaceSize(x, y, &workspaceSize, &executor);
    if (status != ACLNN_SUCCESS) {
        std::printf("FAIL  row_reduce_sum   | GetWorkspaceSize error %d\n", static_cast<int>(status));
        return false;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        HQSB_ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "reduce.alloc.ws");
    }
    status = aclnnHqsbRowReduceSum(workspace, workspaceSize, executor, stream);
    if (status != ACLNN_SUCCESS) {
        std::printf("FAIL  row_reduce_sum   | launch error %d\n", static_cast<int>(status));
        return false;
    }
    HQSB_ACL_CHECK(aclrtSynchronizeStream(stream), "reduce.sync");
    if (!Download(deviceY, hostY, outBytes)) {
        return false;
    }

    float maxDiff = 0.0f;
    for (int64_t row = 0; row < kRows; ++row) {
        double expect = 0.0;
        for (int64_t column = 0; column < kHidden; ++column) {
            expect += static_cast<double>(hostX[row * kHidden + column]);
        }
        maxDiff = std::fmax(maxDiff, static_cast<float>(std::fabs(hostY[row] - expect)));
    }
    const bool ok = maxDiff <= kTolerance;
    Report("row_reduce_sum", ok, maxDiff, "(sum over last axis)");

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(deviceX);
    aclrtFree(deviceY);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ok;
}

// ── HqsbRmsNorm: y = x * rstd * gamma, rstd = 1 / sqrt(mean(x^2) + eps) ─────
bool TestRmsNorm(aclrtStream stream)
{
    constexpr int64_t kRows = 8;
    constexpr int64_t kHidden = 256;
    constexpr double kEpsilon = 1e-6;
    const std::vector<int64_t> inShape = {kRows, kHidden};
    const std::vector<int64_t> gammaShape = {kHidden};
    const size_t inBytes = static_cast<size_t>(kRows * kHidden) * sizeof(float);
    const size_t gammaBytes = static_cast<size_t>(kHidden) * sizeof(float);

    std::vector<float> hostX(kRows * kHidden);
    std::vector<float> hostGamma(kHidden);
    std::vector<float> hostY(kRows * kHidden, 0.0f);
    for (int64_t index = 0; index < kRows * kHidden; ++index) {
        hostX[index] = static_cast<float>((index % 13) - 6) * 0.25f;
    }
    for (int64_t index = 0; index < kHidden; ++index) {
        hostGamma[index] = 1.0f + static_cast<float>(index % 5) * 0.125f;
    }

    void *deviceX = nullptr;
    void *deviceGamma = nullptr;
    void *deviceY = nullptr;
    if (!Upload(hostX, &deviceX, inBytes) || !Upload(hostGamma, &deviceGamma, gammaBytes)) {
        return false;
    }
    HQSB_ACL_CHECK(aclrtMalloc(&deviceY, inBytes, ACL_MEM_MALLOC_HUGE_FIRST), "rmsnorm.alloc.y");

    aclTensor *x = MakeTensor(inShape, deviceX);
    aclTensor *gamma = MakeTensor(gammaShape, deviceGamma);
    aclTensor *y = MakeTensor(inShape, deviceY);
    if (x == nullptr || gamma == nullptr || y == nullptr) {
        std::printf("FAIL  rms_norm         | aclCreateTensor returned null\n");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnHqsbRmsNormGetWorkspaceSize(x, gamma, kEpsilon, y, &workspaceSize, &executor);
    if (status != ACLNN_SUCCESS) {
        std::printf("FAIL  rms_norm         | GetWorkspaceSize error %d\n", static_cast<int>(status));
        return false;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        HQSB_ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "rmsnorm.alloc.ws");
    }
    status = aclnnHqsbRmsNorm(workspace, workspaceSize, executor, stream);
    if (status != ACLNN_SUCCESS) {
        std::printf("FAIL  rms_norm         | launch error %d\n", static_cast<int>(status));
        return false;
    }
    HQSB_ACL_CHECK(aclrtSynchronizeStream(stream), "rmsnorm.sync");
    if (!Download(deviceY, hostY, inBytes)) {
        return false;
    }

    float maxDiff = 0.0f;
    for (int64_t row = 0; row < kRows; ++row) {
        const int64_t base = row * kHidden;
        double squareSum = 0.0;
        for (int64_t column = 0; column < kHidden; ++column) {
            const double value = static_cast<double>(hostX[base + column]);
            squareSum += value * value;
        }
        // eps after the mean of squares, before the square root.
        const double rstd = 1.0 / std::sqrt(squareSum / static_cast<double>(kHidden) + kEpsilon);
        for (int64_t column = 0; column < kHidden; ++column) {
            const double expect =
                static_cast<double>(hostX[base + column]) * rstd * static_cast<double>(hostGamma[column]);
            maxDiff = std::fmax(maxDiff, static_cast<float>(std::fabs(hostY[base + column] - expect)));
        }
    }
    const bool ok = maxDiff <= kTolerance;
    Report("rms_norm", ok, maxDiff, "(eps after mean, before sqrt)");

    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y);
    aclrtFree(deviceX);
    aclrtFree(deviceGamma);
    aclrtFree(deviceY);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ok;
}

}  // namespace

int main()
{
    aclError status = aclInit(nullptr);
    if (status != ACL_SUCCESS) {
        std::printf("FAIL  aclInit: %d\n", static_cast<int>(status));
        return 1;
    }
    status = aclrtSetDevice(0);
    if (status != ACL_SUCCESS) {
        std::printf("FAIL  aclrtSetDevice: %d\n", static_cast<int>(status));
        return 1;
    }
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);
    if (status != ACL_SUCCESS) {
        std::printf("FAIL  aclrtCreateStream: %d\n", static_cast<int>(status));
        return 1;
    }

    std::printf("=== HQSB Ascend C aclnn smoke (real device) ===\n");
    bool ok = true;
    ok = TestAdd(stream) && ok;
    ok = TestRowReduceSum(stream) && ok;
    ok = TestRmsNorm(stream) && ok;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::printf("----------------------------------------------\n");
    std::printf("%s\n", ok ? "ACLNN SMOKE: ALL PASS" : "ACLNN SMOKE: FAILURES PRESENT");
    return ok ? 0 : 1;
}
