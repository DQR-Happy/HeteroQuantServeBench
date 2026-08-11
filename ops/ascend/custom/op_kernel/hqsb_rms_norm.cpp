/**
 * Device side of HqsbRmsNorm (E09-03/04):
 *   mean_square = (1/H) * sum_j x_j^2
 *   rstd        = 1 / sqrt(mean_square + eps)
 *   y_j         = x_j * rstd * gamma_j
 *
 * eps sits after the mean of squares and before the square root -- moving it
 * inside the sqrt defines a different operator, so the position here is the
 * spec's, not an implementation detail.
 *
 * epsilon arrives in `reserved_0` as an fp32 bit pattern (see the host side):
 * the 72-byte ABI has no dedicated field and a second per-operator struct would
 * be the divergence the protocol forbids.
 *
 * Naive scalar path: two passes per row (square sum, then scale).  One pass
 * with a staged UB tile is E09-04's optimisation.
 */

#include "kernel_operator.h"

#include "../../common/hqsb_tiling_data.h"

using namespace AscendC;

namespace {
__aicore__ inline float BitsToFloat(uint32_t bits)
{
    union BitCast {
        uint32_t asUint;
        float asFloat;
    } caster;
    caster.asUint = bits;
    return caster.asFloat;
}
}  // namespace

extern "C" __global__ __aicore__ void hqsb_rms_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HqsbTilingData);
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t rows = tilingData.rows;
    const uint32_t hidden = tilingData.hidden;
    if (rows == 0U || hidden == 0U) {
        return;
    }

    GlobalTensor<float> xGlobal;
    GlobalTensor<float> gammaGlobal;
    GlobalTensor<float> yGlobal;
    xGlobal.SetGlobalBuffer((__gm__ float *)x, rows * hidden);
    gammaGlobal.SetGlobalBuffer((__gm__ float *)gamma, hidden);
    yGlobal.SetGlobalBuffer((__gm__ float *)y, rows * hidden);

    if (tilingData.tiling_key != HQSB_TILING_KEY_ROW_ALIGNED &&
        tilingData.tiling_key != HQSB_TILING_KEY_ROW_TAIL) {
        return;
    }

    const float epsilon = BitsToFloat(tilingData.reserved_0);

    for (uint32_t row = 0U; row < rows; ++row) {
        const uint32_t base = row * hidden;
        float squareSum = 0.0f;
        for (uint32_t column = 0U; column < hidden; ++column) {
            const float value = xGlobal.GetValue(base + column);
            squareSum += value * value;
        }
        // AI Core rejects a direct uint32 -> float cast ("cast between floating
        // and unsigned integer variable is not allowed in aicore function"), so
        // the count is routed through int32 first.
        const float hiddenCount = static_cast<float>(static_cast<int32_t>(hidden));
        const float meanSquare = squareSum / hiddenCount;
        // `sqrtf` is not declared on the device; `sqrt` is the CCE builtin.
        const float rstd = 1.0f / sqrt(meanSquare + epsilon);
        for (uint32_t column = 0U; column < hidden; ++column) {
            const float value = xGlobal.GetValue(base + column);
            const float scale = gammaGlobal.GetValue(column);
            yGlobal.SetValue(base + column, value * rstd * scale);
        }
    }
}
