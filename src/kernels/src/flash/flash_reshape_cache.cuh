/**
 * @brief Reshape-and-cache kernels for writing K/V into paged NHD-layout cache.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_reshape_cache.cuh
 *
 * @details
 * Writes K/V tensors [num_tokens, num_kv_heads, head_dim] from model output
 * into the paged cache [num_blocks, block_size, num_kv_heads, head_dim].
 * Contains BF16→BF16, BF16→FP8 E4M3 store variants with per-head scales,
 * and an absmax scale computation kernel for dynamic FP8 quantization.
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

#include "flash_sm_compat.cuh"
#include <cuda_fp8.h>

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif

#ifndef HDIM
#define HDIM FLASH_HDIM
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// BF16 → BF16 cache write
extern "C" __global__ void flash_reshape_and_cache(
    const flash_half_t* __restrict__ key,
    const flash_half_t* __restrict__ value,
    flash_half_t* __restrict__ key_cache,
    flash_half_t* __restrict__ value_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;
    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    unsigned int block_idx = (unsigned int)slot / cache_block_size;
    unsigned int block_off = (unsigned int)slot % cache_block_size;

    unsigned long long src_offset = (unsigned long long)token_idx * num_kv_heads * head_dim
                                  + (unsigned long long)head_idx * head_dim;
    unsigned long long dst_offset = (unsigned long long)block_idx * cache_block_size * num_kv_heads * head_dim
                                  + (unsigned long long)block_off * num_kv_heads * head_dim
                                  + (unsigned long long)head_idx * head_dim;

    for (unsigned int d = tid * 8; d < head_dim; d += blockDim.x * 8) {
        if (d + 8 <= head_dim) {
            uint4 kv = *(const uint4*)&key[src_offset + d];
            *(uint4*)&key_cache[dst_offset + d] = kv;
            uint4 vv = *(const uint4*)&value[src_offset + d];
            *(uint4*)&value_cache[dst_offset + d] = vv;
        } else {
            for (unsigned int i = d; i < head_dim; i++) {
                key_cache[dst_offset + i] = key[src_offset + i];
                value_cache[dst_offset + i] = value[src_offset + i];
            }
        }
    }
}

// BF16 → FP8 E4M3 cache write (with per-head GPU scale pointers)
extern "C" __global__ void flash_reshape_and_cache_fp8(
    const flash_half_t* __restrict__ key,
    const flash_half_t* __restrict__ value,
    void* __restrict__ key_cache,
    void* __restrict__ value_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;
    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    float ks = k_scale_ptr[head_idx];
    float vs = v_scale_ptr[head_idx];

    unsigned int block_idx = (unsigned int)slot / cache_block_size;
    unsigned int block_off = (unsigned int)slot % cache_block_size;

    unsigned long long src_offset = (unsigned long long)token_idx * num_kv_heads * head_dim
                                  + (unsigned long long)head_idx * head_dim;
    unsigned long long dst_offset = (unsigned long long)block_idx * cache_block_size * num_kv_heads * head_dim
                                  + (unsigned long long)block_off * num_kv_heads * head_dim
                                  + (unsigned long long)head_idx * head_dim;

    __nv_fp8_storage_t* k_dst = (__nv_fp8_storage_t*)key_cache + dst_offset;
    __nv_fp8_storage_t* v_dst = (__nv_fp8_storage_t*)value_cache + dst_offset;

    float inv_k = (ks > 0.f) ? (1.f / ks) : 1.f;
    float inv_v = (vs > 0.f) ? (1.f / vs) : 1.f;

    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float kf = FLASH_HALF2FLOAT(key[src_offset + d]) * inv_k;
        float vf = FLASH_HALF2FLOAT(value[src_offset + d]) * inv_v;
        k_dst[d] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        v_dst[d] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, __NV_E4M3);
    }
}

// Absmax scale computation for dynamic FP8 quantization.
// Processes a [num_tokens, num_kv_heads, head_dim] BF16 tensor
// and writes a single float scale = max(absmax, 1e-12) / 448.0
extern "C" __global__ void flash_bf16_absmax(
    const flash_half_t* __restrict__ data,
    float* __restrict__ scale_out,
    const unsigned int total_elements
) {
    __shared__ float smem_max[256];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    const unsigned int stride = blockDim.x * gridDim.x;

    float local_max = 0.f;
    for (unsigned int i = gid; i < total_elements; i += stride) {
        float val = fabsf(FLASH_HALF2FLOAT(data[i]));
        local_max = fmaxf(local_max, val);
    }

    smem_max[tid] = local_max;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < (unsigned int)s) smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        float block_max = smem_max[0];
        atomicMax((int*)scale_out, __float_as_int(fmaxf(block_max, 1e-12f) / 448.f));
    }
}
