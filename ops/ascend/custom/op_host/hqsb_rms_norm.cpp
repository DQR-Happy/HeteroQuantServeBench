/**
 * Host side of HqsbRmsNorm (E09-03/04):
 *   mean_square = (1/H) * sum_j x_j^2
 *   rstd        = 1 / sqrt(mean_square + eps)      <- eps after the mean, before sqrt
 *   y_j         = x_j * rstd * gamma_j
 *
 * LayerNorm (mean subtraction) and sqrt(mean(x^2 + eps)) are different operators
 * and are not produced here.
 *
 * epsilon is carried in `reserved_0` as the fp32 bit pattern.  The 72-byte ABI
 * has no dedicated eps field, and inventing a second, per-operator struct is
 * exactly the divergence the protocol forbids -- so the value rides in the
 * reserved word and both sides decode it explicitly.
 */

#include <cstring>

#include "register/op_def_registry.h"

#include "../../common/hqsb_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t kTileElems = 2048;  // smaller: RMSNorm holds x, gamma, y + work
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

    float epsilon = 1e-6f;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* epsilonPtr = attrs->GetAttrPointer<float>(0);
        if (epsilonPtr != nullptr) {
            epsilon = *epsilonPtr;
        }
    }
    if (!(epsilon > 0.0f)) {
        // The spec rejects eps <= 0: a non-positive epsilon has no defined rstd.
        return ge::GRAPH_FAILED;
    }
    uint32_t epsilonBits = 0;
    static_assert(sizeof(epsilonBits) == sizeof(epsilon), "epsilon must bit-cast into one uint32");
    std::memcpy(&epsilonBits, &epsilon, sizeof(epsilonBits));

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
    tiling->buffer_count = 4;  // x tile + gamma tile + y tile + reduction work
    tiling->base_rows = 1;
    tiling->extra_rows = 0;
    tiling->total_elements = static_cast<uint64_t>(rows) * hidden;
    tiling->workspace_bytes = 0;
    tiling->reserved_0 = epsilonBits;  // fp32 bit pattern of epsilon
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
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge


namespace ops {
class HqsbRmsNorm : public OpDef {
public:
    explicit HqsbRmsNorm(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(HqsbRmsNorm);
}  // namespace ops
