/**
 * Host side of HqsbRowReduceSum (E09-02): y_r = sum_j x[r, j], last axis only.
 *
 * Output shape is the single schema [R] -- not "[R] or [R, 1] depending on the
 * backend", which the frozen spec forbids.  Accumulation is FP32 (declared in
 * the spec, not discovered later); a naive one-core loop accumulates it.
 */

#include "register/op_def_registry.h"

#include "../../common/hqsb_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t kTileElems = 4096;
constexpr uint32_t kAlignment = 8;

uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HqsbTilingData* tiling = context->GetTilingData<HqsbTilingData>();
    const gert::Shape& xShape = context->GetInputShape(0)->GetStorageShape();

    const uint32_t dimNum = static_cast<uint32_t>(xShape.GetDimNum());
    const uint32_t rows = (dimNum >= 2) ? static_cast<uint32_t>(xShape.GetDim(0)) : 1;
    const uint32_t hidden = (dimNum >= 2) ? static_cast<uint32_t>(xShape.GetDim(1))
                                          : static_cast<uint32_t>(xShape.GetDim(0));

    const uint32_t bounded = (hidden > kTileElems) ? kTileElems : hidden;
    tiling->schema_version = HQSB_TILING_SCHEMA_VERSION;
    tiling->tiling_key = HQSB_TILING_KEY_ROW_ALIGNED;
    tiling->dtype_tag = 0;  // fp32
    tiling->block_dim = 1;  // naive single-core path
    tiling->rows = rows;
    tiling->hidden = hidden;
    tiling->tile_elems = AlignUp(bounded, kAlignment);
    tiling->loop_count = (hidden + tiling->tile_elems - 1) / tiling->tile_elems;
    tiling->tail_elems = hidden % tiling->tile_elems;
    tiling->buffer_count = 2;  // input tile + reduction work buffer
    tiling->base_rows = 1;
    tiling->extra_rows = 0;
    tiling->total_elements = static_cast<uint64_t>(rows) * hidden;
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
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    // Last-axis reduction collapses to [R]; a [R, 1] variant would be a
    // different schema, so it is not produced here.
    yShape->SetDimNum(1);
    yShape->SetDim(0, xShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    // The spec fixes the reduction output at FP32 regardless of the input dtype
    // (FP16/BF16 inputs still accumulate in FP32).
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge


namespace ops {
class HqsbRowReduceSum : public OpDef {
public:
    explicit HqsbRowReduceSum(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(HqsbRowReduceSum);
}  // namespace ops
