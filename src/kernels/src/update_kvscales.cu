/**
 * @brief CUDA kernel used for kv scales update when using fp8 kvcache.
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 * This kernel computes k_scale and v_scale and update existing ones during fp8 kvcache computation
 *
 * This CUDA kernel is part of the vllm.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/update_kvscales.cu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// Scale = max_abs / 448.0f (clamped to >= 1e-6).
#define DIV_CONST 448.0f

template<typename T>
__device__ __forceinline__ float to_float_abs(T x);

template<>
__device__ __forceinline__ float to_float_abs<float>(float x) {
    return fabsf(x);
}

template<>
__device__ __forceinline__ float to_float_abs<__half>(__half x) {
    return fabsf(__half2float(x));
}

template<>
__device__ __forceinline__ float to_float_abs<__nv_bfloat16>(__nv_bfloat16 x) {
    return fabsf(__bfloat162float(x));
}

// --- atomicMax for floats using atomicCAS on int bits ---
// Works for non-negative floats (we only use absolute values divided by 240)
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* addr_i = reinterpret_cast<int*>(address);
    int old_i = *addr_i;
    float old_f = __int_as_float(old_i);

    while (old_f < val) {
        int assumed = old_i;
        int new_i = __float_as_int(val);
        int prev = atomicCAS(addr_i, assumed, new_i);
        if (prev == assumed) {
            return __int_as_float(prev); // return previous value
        }
        old_i = prev;
        old_f = __int_as_float(old_i);
    }
    return old_f;
}
// Per-head scale update. Layout is [num_tokens, num_heads, head_dim] contiguous
template<typename T>
__global__ void compute_and_update_scales_per_head_kernel(
    const T* __restrict__ k,
    const T* __restrict__ v,
    long num_tokens,
    int num_heads,
    int head_dim,
    float* __restrict__ k_scales,
    float* __restrict__ v_scales
) {
    extern __shared__ float sdata[];
    float* s_k = sdata;
    float* s_v = sdata + blockDim.x;

    const int head_idx = blockIdx.y;
    const long numel_per_head = num_tokens * (long)head_dim;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)blockDim.x * (long)gridDim.x;

    float local_max_k = 0.0f;
    float local_max_v = 0.0f;

    for (long i = idx; i < numel_per_head; i += stride) {
        const long token = i / head_dim;
        const int d = (int)(i % head_dim);
        const long base = token * (long)(num_heads * head_dim) + (long)head_idx * head_dim + d;
        float avk = to_float_abs<T>(k[base]);
        float avv = to_float_abs<T>(v[base]);
        if (avk > local_max_k) local_max_k = avk;
        if (avv > local_max_v) local_max_v = avv;
    }

    s_k[threadIdx.x] = local_max_k;
    s_v[threadIdx.x] = local_max_v;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float other_k = s_k[threadIdx.x + s];
            float other_v = s_v[threadIdx.x + s];
            if (other_k > s_k[threadIdx.x]) s_k[threadIdx.x] = other_k;
            if (other_v > s_v[threadIdx.x]) s_v[threadIdx.x] = other_v;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float candidate_k_scale = fmaxf(s_k[0] / DIV_CONST, 1e-6f);
        float candidate_v_scale = fmaxf(s_v[0] / DIV_CONST, 1e-6f);
        if (candidate_k_scale > 0.0f) atomicMaxFloat(&k_scales[head_idx], candidate_k_scale);
        if (candidate_v_scale > 0.0f) atomicMaxFloat(&v_scales[head_idx], candidate_v_scale);
    }
}


template<typename T>
void update_scales_per_head_typed(
    T* k,
    T* v,
    long num_tokens,
    int num_heads,
    int head_dim,
    float* k_scales,
    float* v_scales,
    int64_t stream_
) {
    const int threads = 256;
    long numel_per_head = num_tokens * (long)head_dim;
    long blocks = (numel_per_head + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    const cudaStream_t stream = (cudaStream_t)stream_;
    size_t shared_bytes = 2 * threads * sizeof(float);
    dim3 grid((int)blocks, num_heads, 1);
    compute_and_update_scales_per_head_kernel<T><<<grid, threads, shared_bytes, stream>>>(
        k, v, num_tokens, num_heads, head_dim, k_scales, v_scales
    );
}

extern "C" void update_kv_scales_per_head_f32(
    void* k, void* v, const long num_tokens, int num_heads, int head_dim,
    float* k_scales, float* v_scales, int64_t stream_
) {
    update_scales_per_head_typed<float>(reinterpret_cast<float*>(k), reinterpret_cast<float*>(v),
                                        num_tokens, num_heads, head_dim, k_scales, v_scales,
                                        stream_);
}

extern "C" void update_kv_scales_per_head_f16(
    void* k, void* v, const long num_tokens, int num_heads, int head_dim,
    float* k_scales, float* v_scales, int64_t stream_
) {
    update_scales_per_head_typed<__half>(reinterpret_cast<__half*>(k), reinterpret_cast<__half*>(v),
                                         num_tokens, num_heads, head_dim, k_scales, v_scales,
                                         stream_);
}

extern "C" void update_kv_scales_per_head_bf16(
    void* k, void* v, const long num_tokens, int num_heads, int head_dim,
    float* k_scales, float* v_scales, int64_t stream_
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return;
#else
    update_scales_per_head_typed<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(k), reinterpret_cast<__nv_bfloat16*>(v), num_tokens,
        num_heads, head_dim, k_scales, v_scales, stream_);
#endif
}
