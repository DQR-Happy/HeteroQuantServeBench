/**
 * Device side of HqsbRowReduceSum (E09-02): y_r = sum_j x[r, j], last axis only.
 *
 * Scalar per-element accumulation in FP32 (the dtype the frozen spec declares
 * for the accumulator), one output element per row.  Naive on purpose: the
 * block/tile reduction with UB staging is E09-04.
 */

#include "kernel_operator.h"

#include "../../common/hqsb_tiling_data.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void hqsb_row_reduce_sum(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HqsbTilingData);
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t rows = tilingData.rows;
    const uint32_t hidden = tilingData.hidden;
    if (rows == 0U || hidden == 0U) {
        return;
    }

    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    xGlobal.SetGlobalBuffer((__gm__ float *)x, rows * hidden);
    yGlobal.SetGlobalBuffer((__gm__ float *)y, rows);

    if (tilingData.tiling_key != HQSB_TILING_KEY_ROW_ALIGNED &&
        tilingData.tiling_key != HQSB_TILING_KEY_ROW_TAIL) {
        return;
    }

    for (uint32_t row = 0U; row < rows; ++row) {
        const uint32_t base = row * hidden;
        float rowSum = 0.0f;
        for (uint32_t column = 0U; column < hidden; ++column) {
            rowSum += xGlobal.GetValue(base + column);
        }
        // One fixed output schema [R]: never [R, 1].
        yGlobal.SetValue(row, rowSum);
    }
}
