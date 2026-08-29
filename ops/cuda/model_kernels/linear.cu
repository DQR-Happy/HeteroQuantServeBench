#include "model_kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, out = nullptr;
    cublasLtMatmulAlgo_t algo;
    ~Plan() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (out) cublasLtMatrixLayoutDestroy(out);
    }
};
struct Context {
    cublasLtHandle_t handle = nullptr;
    std::map<std::tuple<int,int,int>, Plan> plans;
    std::mutex mutex;
    ~Context() { plans.clear(); if (handle) cublasLtDestroy(handle); }
};
extern "C" int hqsb_l2_lt_create(void** result) {
    if (!result) return -1;
    *result = nullptr;
    try {
        auto c = new Context;
        int status = int(cublasLtCreate(&c->handle));
        if (status) { delete c; return status; }
        *result = c;
        return 0;
    } catch (...) { return -3; }
}
extern "C" void hqsb_l2_lt_destroy(void* ptr) { delete static_cast<Context*>(ptr); }
extern "C" size_t hqsb_l2_lt_version() { return cublasLtGetVersion(); }

static int prepare(Context* c, Plan& p, int m, int n, int k) {
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    size_t workspace = 0;
    #define CHECK(expr) do { int status = int(expr); if (status) return status; } while(0)
    CHECK(cublasLtMatmulDescCreate(&p.op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    CHECK(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_TRANSB,&trans,sizeof(trans)));
    CHECK(cublasLtMatrixLayoutCreate(&p.a,CUDA_R_16F,m,k,k));
    CHECK(cublasLtMatrixLayoutCreate(&p.b,CUDA_R_16F,n,k,k));
    CHECK(cublasLtMatrixLayoutCreate(&p.out,CUDA_R_16F,m,n,n));
    for (auto layout : {p.a,p.b,p.out})
        CHECK(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order)));
    cublasLtMatmulPreference_t preference;
    CHECK(cublasLtMatmulPreferenceCreate(&preference));
    int status = int(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
    cublasLtMatmulHeuristicResult_t result;
    int count = 0;
    if (!status) status = int(cublasLtMatmulAlgoGetHeuristic(c->handle,p.op,p.a,p.b,p.out,p.out,
                                                           preference,1,&result,&count));
    cublasLtMatmulPreferenceDestroy(preference);
    if (status) return status;
    if (count != 1 || result.state != CUBLAS_STATUS_SUCCESS) return -4;
    p.algo = result.algo;
    return 0;
    #undef CHECK
}

extern "C" int hqsb_l2_lt(void* context, const void* x, const void* w, void* y,
                          int m, int n, int k, void* stream) {
    if (!context || !x || !w || !y || m<=0 || n<=0 || k<=0) return -1;
    auto c = static_cast<Context*>(context);
    std::lock_guard<std::mutex> lock(c->mutex);
    try {
        auto key = std::make_tuple(m,n,k);
        auto found = c->plans.find(key);
        if (found == c->plans.end()) {
            if (c->plans.size() >= 256) return -5;
            auto inserted = c->plans.try_emplace(key);
            int status = prepare(c,inserted.first->second,m,n,k);
            if (status) { c->plans.erase(inserted.first); return status; }
            found = inserted.first;
        }
        auto& p = found->second;
        float alpha = 1.f, beta = 0.f;
        return int(cublasLtMatmul(c->handle,p.op,&alpha,x,p.a,w,p.b,&beta,y,p.out,
                                 y,p.out,&p.algo,nullptr,0,static_cast<cudaStream_t>(stream)));
    } catch (...) { return -3; }
}

extern "C" int hqsb_l2_lt_plan_count(void* context) {
    if (!context) return 0;
    auto c = static_cast<Context*>(context);
    std::lock_guard<std::mutex> lock(c->mutex);
    return int(c->plans.size());
}
extern "C" int hqsb_l2_lt_plan_info(void* context, int index, int* m, int* n, int* k, int* algo) {
    if (!context || index < 0 || !m || !n || !k || !algo) return -1;
    auto c = static_cast<Context*>(context);
    std::lock_guard<std::mutex> lock(c->mutex);
    if (size_t(index) >= c->plans.size()) return -1;
    auto entry = c->plans.begin();
    std::advance(entry,index);
    std::tie(*m,*n,*k) = entry->first;
    size_t written = 0;
    return int(cublasLtMatmulAlgoConfigGetAttribute(&entry->second.algo,CUBLASLT_ALGO_CONFIG_ID,
                                                   algo,sizeof(*algo),&written));
}

// Research small-M variant: FP32 reduction, no weight packing or workspace.
__global__ void gemv(const half* x, const half* w, half* y, int n, int k) {
    int col = blockIdx.x, row = blockIdx.y;
    float sum = 0;
    for (int j=threadIdx.x; j<k; j+=128)
        sum += __half2float(x[int64_t(row)*k+j]) * __half2float(w[int64_t(col)*k+j]);
    __shared__ float parts[4];
    for (int d=16;d;d/=2) sum += __shfl_down_sync(0xffffffff,sum,d);
    if ((threadIdx.x & 31)==0) parts[threadIdx.x/32]=sum;
    __syncthreads();
    sum = threadIdx.x<4 ? parts[threadIdx.x] : 0;
    if (threadIdx.x<32)
        for (int d=16;d;d/=2) sum += __shfl_down_sync(0xffffffff,sum,d);
    if (threadIdx.x==0) y[int64_t(row)*n+col]=__float2half_rn(sum);
}
extern "C" int hqsb_l2_gemv(const void* x,const void* w,void* y,int m,int n,int k,void* stream) {
    if (!x || !w || !y || m<1 || m>4 || n<1 || k<1) return -1;
    gemv<<<dim3(n,m),128,0,static_cast<cudaStream_t>(stream)>>>(
        (const half*)x,(const half*)w,(half*)y,n,k);
    return int(cudaGetLastError());
}
