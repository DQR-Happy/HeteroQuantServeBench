#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Trusted C ABI v1. Inputs FP16 CUDA:0, contiguous, outputs disjoint and fresh.
// Shapes and stride units are elements; pointers already include storage offset.
// Caller owns buffers/stream, validates capacity, and waits before releasing them.
// No kernel synchronizes or allocates. Negative status is a contract/provider
// error; CUDA/cuBLASLt status is otherwise returned unchanged. No fallback.
int hqsb_l2_abi(void);
int hqsb_l2_norm(const void* x,const void* residual,const void* weight,void* y,void* updated,
                 int64_t rows,int h,float eps,int variant,void* stream);
int hqsb_l2_silu(const void* gate,const void* up,void* y,int64_t n,void* stream);
int hqsb_l2_rope(const void* x,const void* cos,const void* sin,void* y,
                 int b,int t,int h,int d,void* stream);
int hqsb_l2_lt_create(void** context);
void hqsb_l2_lt_destroy(void* context);
size_t hqsb_l2_lt_version(void);
int hqsb_l2_lt(void* context,const void* x,const void* weight,void* y,int m,int n,int k,void* stream);
int hqsb_l2_lt_plan_count(void* context);
int hqsb_l2_lt_plan_info(void* context,int index,int* m,int* n,int* k,int* algorithm);
int hqsb_l2_gemv(const void* x,const void* weight,void* y,int m,int n,int k,void* stream);
#ifdef __cplusplus
}
#endif
