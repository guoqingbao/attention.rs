#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename T>
__global__ void gptoss_swiglu_kernel(const T *__restrict__ gate,
                                     const T *__restrict__ up,
                                     T *__restrict__ output, const uint32_t N,
                                     const float alpha, const float limit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = (float)gate[idx];
    float u = (float)up[idx];

    float gate_clamped = fminf(g, limit);
    float up_clamped = fmaxf(fminf(u, limit), -limit);
    float glu = gate_clamped * fast_sigmoid(gate_clamped * alpha);
    float result = (up_clamped + 1.0f) * glu;

    output[idx] = (T)result;
}

template <typename T, typename T4>
__global__ void gptoss_swiglu_kernel_vec4(const T4 *__restrict__ gate,
                                          const T4 *__restrict__ up,
                                          T4 *__restrict__ output,
                                          const uint32_t N4, const float alpha,
                                          const float limit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N4) return;

    T4 g4 = gate[idx];
    T4 u4 = up[idx];

    float g0 = (float)((T *)&g4)[0];
    float g1 = (float)((T *)&g4)[1];
    float g2 = (float)((T *)&g4)[2];
    float g3 = (float)((T *)&g4)[3];

    float u0 = (float)((T *)&u4)[0];
    float u1 = (float)((T *)&u4)[1];
    float u2 = (float)((T *)&u4)[2];
    float u3 = (float)((T *)&u4)[3];

#pragma unroll
    for (int i = 0; i < 4; i++) {
        float g = (i == 0) ? g0 : (i == 1) ? g1 : (i == 2) ? g2 : g3;
        float u = (i == 0) ? u0 : (i == 1) ? u1 : (i == 2) ? u2 : u3;

        float gate_clamped = fminf(g, limit);
        float up_clamped = fmaxf(fminf(u, limit), -limit);
        float glu = gate_clamped * fast_sigmoid(gate_clamped * alpha);
        float result = (up_clamped + 1.0f) * glu;

        if (i == 0) ((T *)&g4)[0] = (T)result;
        else if (i == 1) ((T *)&g4)[1] = (T)result;
        else if (i == 2) ((T *)&g4)[2] = (T)result;
        else ((T *)&g4)[3] = (T)result;
    }

    output[idx] = g4;
}

extern "C" void gptoss_swiglu_f16(const __half *gate, const __half *up,
                                  __half *output, uint32_t N, float alpha,
                                  float limit, cudaStream_t stream) {
    if (N % 4 == 0) {
        const int N4 = N / 4;
        const int nthreads = 256;
        const int nblocks = (N4 + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel_vec4<__half, uint64_t>
            <<<nblocks, nthreads, 0, stream>>>(
                (const uint64_t *)gate, (const uint64_t *)up, (uint64_t *)output,
                N4, alpha, limit);
    } else {
        const int nthreads = 256;
        const int nblocks = (N + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(gate, up, output, N,
                                                               alpha, limit);
    }
}

#ifndef NO_BF16_KERNEL
extern "C" void gptoss_swiglu_bf16(const __nv_bfloat16 *gate,
                                   const __nv_bfloat16 *up,
                                   __nv_bfloat16 *output, uint32_t N,
                                   float alpha, float limit,
                                   cudaStream_t stream) {
    if (N % 4 == 0) {
        const int N4 = N / 4;
        const int nthreads = 256;
        const int nblocks = (N4 + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel_vec4<__nv_bfloat16, uint64_t>
            <<<nblocks, nthreads, 0, stream>>>(
                (const uint64_t *)gate, (const uint64_t *)up, (uint64_t *)output,
                N4, alpha, limit);
    } else {
        const int nthreads = 256;
        const int nblocks = (N + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(gate, up, output, N,
                                                               alpha, limit);
    }
}
#else
extern "C" void gptoss_swiglu_bf16(const void *, const void *, void *,
                                   uint32_t, float, float, cudaStream_t) {}
#endif

extern "C" void gptoss_swiglu_f32(const float *gate, const float *up,
                                  float *output, uint32_t N, float alpha,
                                  float limit, cudaStream_t stream) {
    if (N % 4 == 0) {
        const int N4 = N / 4;
        const int nthreads = 256;
        const int nblocks = (N4 + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel_vec4<float, float4><<<nblocks, nthreads, 0, stream>>>(
            (const float4 *)gate, (const float4 *)up, (float4 *)output, N4, alpha,
            limit);
    } else {
        const int nthreads = 256;
        const int nblocks = (N + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(gate, up, output, N,
                                                               alpha, limit);
    }
}
