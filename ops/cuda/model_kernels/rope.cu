#include "model_kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Input is contiguous [B,T,H,D]; output [B,H,T,D]. cos/sin [B,T,D].
// Preserve two FP16 products and an FP16 addition, rather than contracting FMA.
__global__ void rope(const half* x, const half* c, const half* s, half* y,
                     int t, int h, int d, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int col = i % d, head = (i / d) % h, token = (i / (d*h)) % t;
    int batch = i / (int64_t(d)*h*t);
    int other = col < d/2 ? col + d/2 : col - d/2;
    half v = x[i - col + other];
    if (col < d/2) v = __hneg(v);
    int64_t pos = (int64_t(batch)*t + token)*d + col;
    y[((int64_t(batch)*h+head)*t+token)*d+col] =
        __hadd(__hmul(x[i],c[pos]),__hmul(v,s[pos]));
}
extern "C" int hqsb_l2_rope(const void* x, const void* c, const void* s, void* y,
                            int b, int t, int h, int d, void* stream) {
    int64_t n = int64_t(b)*t*h*d;
    if (!x || !c || !s || !y || b<=0 || t<=0 || h<=0 || d<=0 || d%2 ||
        n > int64_t(2147483647)*256) return -1;
    rope<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        (const half*)x,(const half*)c,(const half*)s,(half*)y,t,h,d,n);
    return int(cudaGetLastError());
}
