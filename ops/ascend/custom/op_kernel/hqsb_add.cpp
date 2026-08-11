/**
 * Device side of HqsbAdd (E09-02): y = x1 + x2, elementwise.
 *
 * Deliberately the *most basic* kernel that completes the task: a scalar
 * per-element loop over global memory.  It exists to prove
 * compile -> load -> run -> numeric-check works end to end on the real device,
 * not to be fast.  The vector/UB path with tiling and double buffering is
 * E09-04's job, and this file must not be mistaken for it.
 *
 * The tiling struct is the shared 72-byte ABI in
 * ops/ascend/common/hqsb_tiling_data.h -- the same header the host tiling fills,
 * so host and device cannot disagree about field order or width.
 */

#include "kernel_operator.h"

#include "../../common/hqsb_tiling_data.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void hqsb_add(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HqsbTilingData);
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t total = static_cast<uint32_t>(tilingData.total_elements);
    if (total == 0U) {
        return;
    }

    GlobalTensor<float> x1Global;
    GlobalTensor<float> x2Global;
    GlobalTensor<float> yGlobal;
    x1Global.SetGlobalBuffer((__gm__ float *)x1, total);
    x2Global.SetGlobalBuffer((__gm__ float *)x2, total);
    yGlobal.SetGlobalBuffer((__gm__ float *)y, total);

    if (tilingData.tiling_key != HQSB_TILING_KEY_SINGLE_TILE_ALIGNED &&
        tilingData.tiling_key != HQSB_TILING_KEY_MULTI_TILE_ALIGNED &&
        tilingData.tiling_key != HQSB_TILING_KEY_TAIL_MASKED) {
        // An unknown key must fail loudly: falling through to a default path is
        // exactly the silent-downgrade the protocol forbids.
        return;
    }

    for (uint32_t index = 0U; index < total; ++index) {
        yGlobal.SetValue(index, x1Global.GetValue(index) + x2Global.GetValue(index));
    }
}
