/**
 * @brief Metal kernel used for kv scales update when using fp8 kvcache.
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 * This kernel computes k_scale and v_scale and update existing ones during fp8 kvcache computation
 *
 * This Metal kernel is part of the vllm.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/metal-kernels/src/update_kvscales.metal
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
#include "metal_dtype.metal"
#include <metal_stdlib>
using namespace metal;

#define DIV_CONST 240.0f

template <typename T>
float to_float_abs(T x);

// FP32 specialization
template <>
float to_float_abs<float>(float x) {
    return fabs(x);
}

// FP16 specialization
template <>
float to_float_abs<half>(half x) {
    return fabs(float(x));  // Convert half to float for the operation
}

// BF16 specialization
template <>
float to_float_abs<bfloat16_t>(bfloat16_t x) {
    return fabs(float(x));  // Convert bfloat16 to float for the operation
}


#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

template <typename T>
kernel void compute_and_update_scales_per_head_kernel(
    device const T* k [[buffer(0)]],
    device const T* v [[buffer(1)]],
    constant long& num_tokens [[buffer(2)]],
    constant int& num_heads [[buffer(3)]],
    constant int& head_dim [[buffer(4)]],
    device atomic_uint* k_scales [[buffer(5)]],
    device atomic_uint* v_scales [[buffer(6)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    device float* sdata = nullptr;
    device float* s_k = sdata;
    device float* s_v = sdata + blockDim.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int gdim = gridDim.x;
    const int head_idx = blockIdx.y;

    long global_thread_index = (long)blockIdx.x * bdim + tid;
    long stride = (long)bdim * (long)gdim;
    long numel_per_head = num_tokens * (long)head_dim;

    float local_max_k = 0.0f;
    float local_max_v = 0.0f;

    for (long i = global_thread_index; i < numel_per_head; i += stride) {
        long token = i / head_dim;
        int d = (int)(i % head_dim);
        long base = token * (long)(num_heads * head_dim) + (long)head_idx * head_dim + d;
        float avk = to_float_abs<T>(k[base]);
        float avv = to_float_abs<T>(v[base]);
        if (avk > local_max_k) local_max_k = avk;
        if (avv > local_max_v) local_max_v = avv;
    }

    s_k[tid] = local_max_k;
    s_v[tid] = local_max_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_k[tid + s] > s_k[tid]) s_k[tid] = s_k[tid + s];
            if (s_v[tid + s] > s_v[tid]) s_v[tid] = s_v[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float candidate_k_scale = s_k[0] / DIV_CONST;
        float candidate_v_scale = s_v[0] / DIV_CONST;
        float cur_k_scale = (float)atomic_load_explicit(&k_scales[head_idx], memory_order_relaxed);
        float cur_v_scale = (float)atomic_load_explicit(&v_scales[head_idx], memory_order_relaxed);
        if (candidate_k_scale > cur_k_scale) {
            atomic_exchange_explicit(&k_scales[head_idx], candidate_k_scale, memory_order_relaxed);
        }
        if (candidate_v_scale > cur_v_scale) {
            atomic_exchange_explicit(&v_scales[head_idx], candidate_v_scale, memory_order_relaxed);
        }
    }
}

#define instantiate_compute_and_update_scales_per_head(type)        \
  template [[host_name("compute_and_update_scales_per_head_" #type)]]                \
  kernel void compute_and_update_scales_per_head_kernel<type>( \
    device const type* k [[buffer(0)]],                    \
    device const type* v [[buffer(1)]],                    \
    constant long& num_tokens [[buffer(2)]],               \
    constant int& num_heads [[buffer(3)]],                 \
    constant int& head_dim [[buffer(4)]],                  \
    device atomic_uint* k_scales [[buffer(5)]],            \
    device atomic_uint* v_scales [[buffer(6)]],            \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint3 threadIdx [[thread_position_in_threadgroup]], \
    uint3 gridDim [[threadgroups_per_grid]],\
    uint3 blockDim [[threads_per_threadgroup]]);

instantiate_compute_and_update_scales_per_head(float)
instantiate_compute_and_update_scales_per_head(half)
instantiate_compute_and_update_scales_per_head(bfloat16_t)
