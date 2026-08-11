/**
 * Host side of HqsbAdd (E09-02): z = x1 + x2, elementwise, no broadcasting.
 *
 * The tiling writes the shared 72-byte HqsbTilingData ABI
 * (ops/ascend/common/hqsb_tiling_data.h) that the device kernel parses, so host
 * and device cannot drift apart on field order or width.
 *
 * This is the *naive* reference implementation: one AI core, a bounded UB tile,
 * a plain per-tile loop.  It exists to prove the compile -> load -> run ->
 * numeric-check chain end to end, not to be fast (E09-04 optimises later).
 */

#include "register/op_def_registry.h"

#include "../../common/hqsb_tiling_data.h"

namespace optiling {
namespace {
//: Elements per UB pass: 16 KiB of fp32, far below the 310B UB budget, so the
//: naive kernel never has to reason about double buffering.
constexpr uint32_t kTileElems = 4096;
//: DataCopy requires a 32-byte aligned span; fp32 is 4 bytes.
constexpr uint32_t kAlignment = 8;

uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HqsbTilingData* tiling = context->GetTilingData<HqsbTilingData>();
    const gert::StorageShape* x1Shape = context->GetInputShape(0);

    uint64_t total = 1;
    for (size_t index = 0; index < x1Shape->GetStorageShape().GetDimNum(); ++index) {
        total *= static_cast<uint64_t>(x1Shape->GetStorageShape().GetDim(index));
    }

    const uint32_t bounded = static_cast<uint32_t>(total > kTileElems ? kTileElems : total);
    tiling->schema_version = HQSB_TILING_SCHEMA_VERSION;
    tiling->tiling_key = HQSB_TILING_KEY_MULTI_TILE_ALIGNED;
    tiling->dtype_tag = 0;  // fp32; narrow dtype tags belong to E09-06
    tiling->block_dim = 1;  // naive single-core path
    tiling->rows = 1;
    tiling->hidden = static_cast<uint32_t>(total);
    tiling->tile_elems = AlignUp(bounded, kAlignment);
    tiling->loop_count = static_cast<uint32_t>((total + tiling->tile_elems - 1) / tiling->tile_elems);
    tiling->tail_elems = static_cast<uint32_t>(total % tiling->tile_elems);
    tiling->buffer_count = 3;  // x1 + x2 + y
    tiling->base_rows = 1;
    tiling->extra_rows = 0;
    tiling->total_elements = total;
    tiling->workspace_bytes = 0;
    tiling->reserved_0 = 0;
    tiling->reserved_1 = 0;

    context->SetBlockDim(1);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *x1Shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge


namespace ops {
class HqsbAdd : public OpDef {
public:
    explicit HqsbAdd(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(HqsbAdd);
}  // namespace ops
