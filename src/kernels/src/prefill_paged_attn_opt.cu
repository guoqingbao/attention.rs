/**
 * @brief Optimized CUDA kernel for chunked prefill attention with paged KV-cache.
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 * 
 * This is an optimized version of the prefill_paged_attn.cu kernel designed for
 * large KV-cache scenarios (num_blocks > 64). It uses shared memory tiling for
 * cooperative K/V block loading and binary search for O(log N) sequence lookup.
 *
 * This CUDA kernel is part of the vllm.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/prefill_paged_attn_opt.cu
 * 
 * Optimizations:
 *  - Shared Memory Tiling: Cooperative loading of K/V blocks reduces global memory bandwidth.
 *  - Binary Search: O(log N) sequence lookup instead of O(N) linear scan.
 *  - Chunk Processing: Multiple tokens share the same KV blocks via shared memory.
 *
 * Features:
 *  - Support Chunked Prefill (prefilled attention with kvcache)
 *  - Supports paged KV-cache (blocks of tokens stored in memory)
 *  - Handles sliding window attention
 *  - Uses online softmax for numerical stability
 *  - Extended shared memory support for large models (up to 96KB on V100/A100)
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

#include <stdint.h>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"
#include <stdexcept>
#include <algorithm>

#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define TOKEN_CHUNK_SIZE_DEFAULT 256
#define TOKEN_CHUNK_SIZE_LARGE_HEAD 128
using namespace vllm;

namespace vllm_rs {

inline __device__ float fast_tanh_opt(float x) {
  #if defined(__CUDA_ARCH__)
    #if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
      float y;
      asm volatile ( "tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
      return y;
    #else
      return ::tanhf(x);
    #endif
  #else
  return std::tanh(x);
  #endif
}

template <typename T>
__device__ inline T make_zero_opt() {
    T x;
    memset(&x, 0, sizeof(T));
    return x;
}

/**
 * @brief Optimized chunked prefill paged attention kernel with shared memory tiling.
 *
 * This kernel processes a chunk of query tokens that attend to the same paged KV-cache.
 * All threads in a block cooperatively load K/V tiles into shared memory, then each
 * thread computes attention for its assigned token.
 *
 * Key design decisions:
 *  - Lane 0 computes sequence info (via binary search) and broadcasts to all threads
 *  - All threads share the same KV blocks, enabling cooperative shared memory loading
 *  - Two __syncthreads() per block iteration ensure correct data dependencies
 *
 * @tparam scalar_t    Scalar type for Q/K/V/O tensors (half, bfloat16)
 * @tparam cache_t     Cache storage type (same as scalar_t, or uint8_t for FP8)
 * @tparam HEAD_SIZE   Number of elements per head (64, 96, 128, 192, 256)
 * @tparam BLOCK_SIZE  Tokens per KV-cache block (32, 64)
 *
 * @param out           Output tensor [num_query_tokens, num_heads, head_size]
 * @param q             Query tensor [num_query_tokens, num_heads, head_size]
 * @param k_cache       Paged K cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
 * @param v_cache       Paged V cache [num_blocks, num_kv_heads, head_size, block_size]
 * @param k_scales      K scale for FP8 quantization (nullptr if not quantized)
 * @param v_scales      V scale for FP8 quantization (nullptr if not quantized)
 * @param query_start_len  Cumulative sum of query lengths per sequence [num_seqs+1]
 */
template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE, int TOKEN_CHUNK_SIZE>
__global__ void chunked_prefill_paged_attention_kernel_opt(
    scalar_t* __restrict__ out,              
    const scalar_t* __restrict__ q,          
    const cache_t* __restrict__ k_cache,     
    const cache_t* __restrict__ v_cache,     
    const float* __restrict__ k_scales, 
    const float* v_scales,
    int32_t num_kv_heads,
    float sm_scale,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ seq_lens,
    int32_t block_table_stride,
    int32_t num_seqs,
    int32_t num_query_heads,
    int32_t num_query_tokens,
    float softscapping,
    int32_t o_stride_tokens,
    const uint32_t* __restrict__ query_start_len,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ sinks,
    int32_t sliding_window,
    int32_t total_num_blocks,
    int32_t kv_block_stride,
    int32_t kv_head_stride
) {
    const bool is_quantized = !std::is_same<scalar_t, cache_t>::value;
    
    // --- Shared Memory Layout ---
    // First: sequence info broadcast from lane 0 (16 bytes)
    // Then: K cache tile and V cache tile
    extern __shared__ char smem_buffer[];
    
    // Sequence info shared by all threads (computed by lane 0)
    struct SeqInfo {
        int seq_idx;
        int num_blocks;
        int start_block_idx;
        int start_token_idx;
        int seq_query_start;
        int seq_query_len;
        int q_pos_start;
        int boundary_seq_idx;    // second sequence if chunk spans boundary, -1 otherwise
        int boundary_token_pos;  // first token of second sequence within chunk
    };
    SeqInfo* shared_seq_info = reinterpret_cast<SeqInfo*>(smem_buffer);
    
    cache_t* k_smem = reinterpret_cast<cache_t*>(smem_buffer + 64);
    cache_t* v_smem = k_smem + (HEAD_SIZE * BLOCK_SIZE);

    constexpr int THREAD_GROUP_SIZE = 1;
    constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
    constexpr int NUM_VECS  = HEAD_SIZE / VEC_SIZE;
    constexpr int X = 16 / sizeof(cache_t);

    const int tid = threadIdx.x;
    const int lane = tid % TOKEN_CHUNK_SIZE;
    const int block_dim = blockDim.x;

    const int NUM_BLOCK_VECS = BLOCK_SIZE / VEC_SIZE;
    const int qh_base_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int chunk_start = blockIdx.z * TOKEN_CHUNK_SIZE;
    const int token_start = chunk_start + lane;

    const int num_queries_per_kv = num_query_heads / num_kv_heads;
    const bool use_alibi = (alibi_slopes != nullptr);
    const bool use_sinks = (sinks != nullptr);

    const int64_t q_stride_tokens = (int64_t)num_query_heads * (int64_t)HEAD_SIZE;
    const int64_t q_stride_heads = (int64_t)HEAD_SIZE;
    const int64_t o_stride_heads = (int64_t)HEAD_SIZE;

    // Lane 0 resolves the dominant sequence and detects chunk-boundary crossings
    if (lane == 0) {
        int seq_idx = 0;
        if (chunk_start < (int)query_start_len[num_seqs] && chunk_start >= (int)query_start_len[0]) {
            int left = 0, right = num_seqs - 1;
            while (left <= right) {
                int mid = (left + right) / 2;
                if ((int)query_start_len[mid + 1] <= chunk_start) {
                    left = mid + 1;
                } else if ((int)query_start_len[mid] > chunk_start) {
                    right = mid - 1;
                } else {
                    seq_idx = mid;
                    break;
                }
            }
        }
        
        uint32_t seq_len_full = seq_lens[seq_idx];
        int seq_query_start = query_start_len[seq_idx];
        int seq_query_end = query_start_len[seq_idx + 1];
        int seq_query_len = seq_query_end - seq_query_start;
        int q_pos_start = (int)seq_len_full - seq_query_len;
        int first_local_q_pos = chunk_start - seq_query_start;
        if (first_local_q_pos < 0) first_local_q_pos = 0;
        int first_q_abs_pos = q_pos_start + first_local_q_pos;
        int num_blocks_seq = (int)((seq_len_full + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        int start_token_idx = 0;
        int start_block_idx = 0;
        if (sliding_window > 0 && sliding_window <= first_q_abs_pos) {
            start_token_idx = first_q_abs_pos + 1 - sliding_window;
            start_block_idx = start_token_idx / BLOCK_SIZE;
        }
        
        // Detect if this chunk spans a sequence boundary
        int chunk_end = chunk_start + TOKEN_CHUNK_SIZE;
        int boundary_seq = -1;
        int boundary_pos = TOKEN_CHUNK_SIZE;
        if (seq_idx + 1 < num_seqs && seq_query_end < chunk_end && seq_query_end > chunk_start) {
            boundary_seq = seq_idx + 1;
            boundary_pos = seq_query_end - chunk_start;
        }

        shared_seq_info->seq_idx = seq_idx;
        shared_seq_info->num_blocks = num_blocks_seq;
        shared_seq_info->start_block_idx = start_block_idx;
        shared_seq_info->start_token_idx = start_token_idx;
        shared_seq_info->seq_query_start = seq_query_start;
        shared_seq_info->seq_query_len = seq_query_len;
        shared_seq_info->q_pos_start = q_pos_start;
        shared_seq_info->boundary_seq_idx = boundary_seq;
        shared_seq_info->boundary_token_pos = boundary_pos;
    }
    __syncthreads();
    
    // All threads read shared info for the dominant (first) sequence
    const int dom_seq_idx = shared_seq_info->seq_idx;
    const int dom_num_blocks = shared_seq_info->num_blocks;
    const int dom_start_block_idx = shared_seq_info->start_block_idx;
    const int dom_start_token_idx = shared_seq_info->start_token_idx;
    const int dom_seq_query_start = shared_seq_info->seq_query_start;
    const int dom_q_pos_start = shared_seq_info->q_pos_start;
    const int boundary_seq_idx = shared_seq_info->boundary_seq_idx;
    const int boundary_token_pos = shared_seq_info->boundary_token_pos;

    // Per-thread sequence resolution: am I in the dominant or boundary sequence?
    const bool in_boundary_seq = (boundary_seq_idx >= 0 && lane >= boundary_token_pos);
    
    // Each thread resolves its own sequence context
    int my_seq_idx, my_num_blocks, my_start_block_idx, my_start_token_idx;
    int my_seq_query_start, my_q_pos_start;
    uint32_t my_seq_len_full;
    
    if (!in_boundary_seq) {
        my_seq_idx = dom_seq_idx;
        my_num_blocks = dom_num_blocks;
        my_start_block_idx = dom_start_block_idx;
        my_start_token_idx = dom_start_token_idx;
        my_seq_query_start = dom_seq_query_start;
        my_q_pos_start = dom_q_pos_start;
        my_seq_len_full = seq_lens[dom_seq_idx];
    } else {
        my_seq_idx = boundary_seq_idx;
        my_seq_len_full = seq_lens[boundary_seq_idx];
        my_seq_query_start = query_start_len[boundary_seq_idx];
        int my_seq_query_len = query_start_len[boundary_seq_idx + 1] - my_seq_query_start;
        my_q_pos_start = (int)my_seq_len_full - my_seq_query_len;
        my_num_blocks = (int)((my_seq_len_full + BLOCK_SIZE - 1) / BLOCK_SIZE);
        int local_q = token_start - my_seq_query_start;
        if (local_q < 0) local_q = 0;
        int abs_q = my_q_pos_start + local_q;
        my_start_token_idx = 0;
        my_start_block_idx = 0;
        if (sliding_window > 0 && sliding_window <= abs_q) {
            my_start_token_idx = abs_q + 1 - sliding_window;
            my_start_block_idx = my_start_token_idx / BLOCK_SIZE;
        }
    }

    const uint32_t* my_block_table = block_tables + (int64_t)my_seq_idx * (int64_t)block_table_stride;
    const int my_local_q_pos = token_start - my_seq_query_start;
    const int my_q_abs_pos = my_q_pos_start + my_local_q_pos;
    
    // Recompute sliding window for this specific thread
    if (sliding_window > 0 && sliding_window <= my_q_abs_pos && !in_boundary_seq) {
        my_start_token_idx = my_q_abs_pos + 1 - sliding_window;
    }

    using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
    using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
    using Float_vec = typename Vec<float, VEC_SIZE>::Type;
    using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

    Q_vec q_vec[NUM_VECS];
    float qk_block[BLOCK_SIZE];

    const int query_head_idx = kv_head_idx * num_queries_per_kv + qh_base_idx;
    const bool head_active = (qh_base_idx < num_queries_per_kv) && (query_head_idx < num_query_heads);
    const bool lane_active = (token_start < num_query_tokens);

    const int64_t q_off = (int64_t)token_start * q_stride_tokens + (int64_t)query_head_idx * q_stride_heads;
    const int64_t o_off = (int64_t)token_start * (int64_t)o_stride_tokens + (int64_t)query_head_idx * o_stride_heads;
    
    if (head_active && lane_active) {
        #pragma unroll
        for (int k = 0; k < NUM_VECS; k++) {
            int d_base = k * VEC_SIZE;
            q_vec[k] = *reinterpret_cast<const Q_vec*>(&q[q_off + d_base]);
        }
    }

    float acc_vec[HEAD_SIZE] = { 0.f };
    float M = (use_sinks && head_active && lane_active) ? sinks[query_head_idx] : -INFINITY;
    float alibi = (use_alibi && head_active && lane_active) ? alibi_slopes[query_head_idx] : 0.f;
    float L = 1.f;

    const int elems_per_block = HEAD_SIZE * BLOCK_SIZE;

    // --- Boundary-sequence threads: non-tiled path (read from global memory) ---
    // These threads cannot participate in cooperative KV loading for the dominant
    // sequence since they need different KV blocks. They run independently.
    if (in_boundary_seq && head_active && lane_active) {
        for (int blk = my_start_block_idx; blk < my_num_blocks; ++blk) {
            const uint32_t physical_block = my_block_table[blk];
            const bool valid_block = (physical_block != UINT32_MAX) &&
                                     ((uint64_t)physical_block < (uint64_t)total_num_blocks);
            if (!valid_block) continue;

            const int block_in_full = blk * BLOCK_SIZE;
            const int64_t k_base = (int64_t)physical_block * kv_block_stride + (int64_t)kv_head_idx * kv_head_stride;
            const int64_t v_base = k_base;
            bool in_contexts[BLOCK_SIZE];

            for (int b = 0; b < BLOCK_SIZE; ++b) {
                const int token_idx_in_full = block_in_full + b;
                bool in_context = (token_idx_in_full <= my_q_abs_pos);
                bool in_window = (token_idx_in_full >= my_start_token_idx);
                in_contexts[b] = in_context && in_window;

                if (!in_context || !in_window) {
                    qk_block[b] = -INFINITY;
                } else {
                    K_vec k_vec_local[NUM_VECS];
                    #pragma unroll
                    for (int k = 0; k < NUM_VECS; k++) {
                        int d = k * VEC_SIZE;
                        int gy = d / X;
                        int gx = d % X;
                        int64_t k_idx = k_base + b * X + gy * (BLOCK_SIZE * X) + gx;
                        if constexpr (!is_quantized) {
                            k_vec_local[k] = *reinterpret_cast<const K_vec*>(&k_cache[k_idx]);
                        } else {
                            Quant_vec fp8_k = *reinterpret_cast<const Quant_vec*>(&k_cache[k_idx]);
                            k_vec_local[k] = vllm::fp8::scaled_convert<K_vec, Quant_vec>(fp8_k, k_scales[kv_head_idx]);
                        }
                    }
                    float qk = Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vec, k_vec_local) * sm_scale;
                    if (softscapping != 1.0) qk = fast_tanh_opt(qk / softscapping) * softscapping;
                    if (use_alibi) qk += alibi * float(token_idx_in_full - my_q_abs_pos);
                    qk_block[b] = qk;
                }
            }

            float Smax = -INFINITY;
            #pragma unroll
            for (int b = 0; b < BLOCK_SIZE; ++b) Smax = fmaxf(Smax, qk_block[b]);
            const float m_j = fmaxf(M, Smax);
            const float alpha_v = __expf(M - m_j);
            M = m_j;
            L = L * alpha_v;
            #pragma unroll
            for (int i = 0; i < HEAD_SIZE; ++i) acc_vec[i] *= alpha_v;

            Float_vec p_vec[NUM_BLOCK_VECS];
            float acc_lane = 0.f;
            #pragma unroll
            for (int b = 0; b < BLOCK_SIZE; ++b) {
                if (in_contexts[b]) {
                    const float P = __expf(qk_block[b] - M);
                    reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = P;
                    acc_lane += P;
                } else {
                    reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = 0.f;
                }
            }
            L += acc_lane;

            for (int k = 0; k < HEAD_SIZE; ++k) {
                const cache_t* v_row = &v_cache[v_base + (int64_t)k * BLOCK_SIZE];
                for (int bv = 0; bv < NUM_BLOCK_VECS; bv++) {
                    Float_vec v_val;
                    if constexpr (!is_quantized) {
                        v_val = to_float(*reinterpret_cast<const K_vec*>(v_row + bv * VEC_SIZE));
                    } else {
                        Quant_vec fp8_v = *reinterpret_cast<const Quant_vec*>(v_row + bv * VEC_SIZE);
                        v_val = vllm::fp8::scaled_convert<Float_vec, Quant_vec>(fp8_v, v_scales[kv_head_idx]);
                    }
                    acc_vec[k] += dot(p_vec[bv], v_val);
                }
            }
        }

        // Write boundary-thread output
        using O_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
        O_vec o_vec[NUM_VECS];
        #pragma unroll
        for (int k = 0; k < HEAD_SIZE; k++) {
            float outv = acc_vec[k] / (L + 1e-6f);
            from_float(reinterpret_cast<scalar_t*>(&o_vec[k / VEC_SIZE])[k % VEC_SIZE], outv);
        }
        #pragma unroll
        for (int k = 0; k < NUM_VECS; k++) {
            *reinterpret_cast<O_vec*>(out + o_off + k * VEC_SIZE) = o_vec[k];
        }
    }

    // Boundary threads are done; they still participate in __syncthreads below
    // but skip computation for dominant-sequence KV blocks.
    
    // --- Dominant-sequence threads: cooperative shared-memory tiled path ---
    const uint32_t* dom_block_table = block_tables + (int64_t)dom_seq_idx * (int64_t)block_table_stride;

    for (int blk = dom_start_block_idx; blk < dom_num_blocks; ++blk) {
        const uint32_t physical_block = dom_block_table[blk];
        const bool valid_block = (physical_block != UINT32_MAX) && 
                                 ((uint64_t)physical_block < (uint64_t)total_num_blocks);

        // ALL threads cooperatively load KV into shared memory
        if (valid_block) {
            const int64_t k_base = (int64_t)physical_block * kv_block_stride + (int64_t)kv_head_idx * kv_head_stride;
            const cache_t* k_src = k_cache + k_base;
            for (int i = tid; i < elems_per_block; i += block_dim) {
                k_smem[i] = k_src[i];
            }
            const cache_t* v_src = v_cache + k_base;
            for (int i = tid; i < elems_per_block; i += block_dim) {
                v_smem[i] = v_src[i];
            }
        }
        
        __syncthreads();

        // Only dominant-sequence threads compute attention from tiled data
        if (valid_block && !in_boundary_seq) {
            const int block_in_full = blk * BLOCK_SIZE;
            bool in_contexts[BLOCK_SIZE];

            for (int b = 0; b < BLOCK_SIZE; ++b) {
                const int token_idx_in_full = block_in_full + b;
                bool in_context = (token_idx_in_full <= my_q_abs_pos);
                bool in_window = (token_idx_in_full >= my_start_token_idx);
                in_contexts[b] = in_context && in_window;

                if (!in_context || !in_window || !lane_active) {
                    qk_block[b] = -INFINITY;
                } else {
                    K_vec k_vec_local[NUM_VECS];
                    #pragma unroll
                    for (int k = 0; k < NUM_VECS; k++) {
                        int d = k * VEC_SIZE;
                        int gy = d / X;
                        int gx = d % X;
                        int smem_idx = b * X + gy * (BLOCK_SIZE * X) + gx;
                        if constexpr (!is_quantized) {
                            k_vec_local[k] = *reinterpret_cast<const K_vec*>(&k_smem[smem_idx]);
                        } else {
                            Quant_vec fp8_k = *reinterpret_cast<const Quant_vec*>(&k_smem[smem_idx]);
                            k_vec_local[k] = vllm::fp8::scaled_convert<K_vec, Quant_vec>(fp8_k, k_scales[kv_head_idx]);
                        }
                    }
                    float qk = Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vec, k_vec_local) * sm_scale;
                    if (softscapping != 1.0) qk = fast_tanh_opt(qk / softscapping) * softscapping;
                    if (use_alibi) qk += alibi * float(token_idx_in_full - my_q_abs_pos);
                    qk_block[b] = qk;
                }
            }

            if (head_active && lane_active) {
                float Smax = -INFINITY;
                #pragma unroll
                for (int b = 0; b < BLOCK_SIZE; ++b) Smax = fmaxf(Smax, qk_block[b]);

                const float m_j = fmaxf(M, Smax);
                const float alpha_v = __expf(M - m_j);
                M = m_j;
                L = L * alpha_v;
                
                #pragma unroll
                for (int i = 0; i < HEAD_SIZE; ++i) acc_vec[i] *= alpha_v;

                Float_vec p_vec[NUM_BLOCK_VECS];
                float acc_lane = 0.f;
                #pragma unroll
                for (int b = 0; b < BLOCK_SIZE; ++b) {
                    if (in_contexts[b]) {
                        const float P = __expf(qk_block[b] - M);
                        reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = P;
                        acc_lane += P;
                    } else {
                        reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = 0.f;
                    }
                }
                L += acc_lane;

                for (int k = 0; k < HEAD_SIZE; ++k) {
                    const cache_t* v_row_ptr = &v_smem[(int64_t)k * BLOCK_SIZE];
                    for (int b_vec = 0; b_vec < NUM_BLOCK_VECS; b_vec++) {
                        const cache_t* src = v_row_ptr + b_vec * VEC_SIZE;
                        Float_vec v_val_vec;
                        if constexpr (!is_quantized) {
                            v_val_vec = to_float(*reinterpret_cast<const K_vec*>(src));
                        } else {
                            Quant_vec fp8_v = *reinterpret_cast<const Quant_vec*>(src);
                            v_val_vec = vllm::fp8::scaled_convert<Float_vec, Quant_vec>(fp8_v, v_scales[kv_head_idx]);
                        }
                        acc_vec[k] += dot(p_vec[b_vec], v_val_vec);
                    }
                }
            }
        }
        
        __syncthreads();
    }

    // Write dominant-thread output
    if (!in_boundary_seq && head_active && lane_active) {
        using O_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
        O_vec o_vec[NUM_VECS];
        #pragma unroll
        for (int k = 0; k < HEAD_SIZE; k++) {
            float outv = acc_vec[k] / (L + 1e-6f);
            from_float(reinterpret_cast<scalar_t*>(&o_vec[k / VEC_SIZE])[k % VEC_SIZE], outv);
        }
        #pragma unroll
        for (int k = 0; k < NUM_VECS; k++) {
            *reinterpret_cast<O_vec*>(out + o_off + k * VEC_SIZE) = o_vec[k];
        }
    }
}

} // namespace vllm_rs

// --- Launcher Code ---

#define LAUNCH_PAGED_ATTENTION_PREFILL_OPT(HEAD_SIZE, CHUNK_SIZE)                            \
  do {                                                                                     \
    int num_token_chunks_ = (num_query_tokens + (CHUNK_SIZE) - 1) / (CHUNK_SIZE);          \
    dim3 grid_(num_queries_per_kv, num_kv_heads, num_token_chunks_);                       \
    dim3 block_(CHUNK_SIZE);                                                               \
    /* Only request extended shared memory if we need more than 48KB default */            \
    if (smem_size > 48 * 1024) {                                                           \
      cudaFuncSetAttribute(                                                                \
        vllm_rs::chunked_prefill_paged_attention_kernel_opt<T, cache_T, HEAD_SIZE, BLOCK_SIZE, CHUNK_SIZE>,\
        cudaFuncAttributeMaxDynamicSharedMemorySize,                                       \
        smem_size);                                                                        \
      cudaGetLastError();                                                                  \
    }                                                                                      \
    vllm_rs::chunked_prefill_paged_attention_kernel_opt<T, cache_T, HEAD_SIZE, BLOCK_SIZE, CHUNK_SIZE> \
    <<<grid_, block_, smem_size, stream>>>(                                                \
      reinterpret_cast<T*>(out),                                                           \
      reinterpret_cast<T*>(query),                                                         \
      reinterpret_cast<cache_T*>(key_cache),                                               \
      reinterpret_cast<cache_T*>(value_cache),                                             \
      k_scales, v_scales,                                                                  \
      num_kv_heads,                                                                        \
      scale,                                                                               \
      block_tables,                                                                        \
      context_lens,                                                                        \
      max_num_blocks_per_seq,                                                              \
      num_seqs,                                                                            \
      num_query_heads,                                                                     \
      num_query_tokens,                                                                    \
      softscapping,                                                                        \
      o_stride_tokens,                                                                     \
      query_start_len,                                                                     \
      alibi_slopes_ptr,                                                                    \
      sinks,                                                                               \
      sliding_window,                                                                      \
      num_blocks,                                                                          \
      kv_block_stride,                                                                     \
      kv_head_stride);                                                                     \
  } while (0)


template<
  typename T,
  typename cache_T,
  int BLOCK_SIZE
  >
void paged_attention_prefill_opt_launcher(
  void *out,
  void *query,
  void *key_cache,
  void *value_cache,
  float* k_scales,
  float* v_scales,
  int32_t num_kv_heads,
  float scale,
  uint32_t *block_tables,
  uint32_t *context_lens,
  int32_t max_num_blocks_per_seq,
  int32_t num_seqs,
  int32_t num_query_heads,
  int32_t num_query_tokens,
  int32_t head_size,
  float softscapping,
  int32_t o_stride_tokens,      // out.stride(0)
  uint32_t* __restrict__ query_start_len, // [num_seqs+1] or nullptr
  float* __restrict__ sinks,  // [num_query_heads] or nullptr
  int32_t sliding_window,
  int32_t num_blocks,
  int32_t kv_block_stride,   // stride between consecutive physical blocks for k_cache (elements)
  int32_t kv_head_stride,    // stride between consecutive kv heads for k_cache (elements)
  int64_t stream_) {

  const float* alibi_slopes_ptr = nullptr;
  const int num_queries_per_kv = num_query_heads / num_kv_heads;
  const cudaStream_t stream = (cudaStream_t)stream_;
  size_t smem_size = 64 + 2 * head_size * BLOCK_SIZE * sizeof(cache_T);
  
  switch (head_size) {
    case 64:
      LAUNCH_PAGED_ATTENTION_PREFILL_OPT(64, TOKEN_CHUNK_SIZE_DEFAULT);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_PREFILL_OPT(128, TOKEN_CHUNK_SIZE_DEFAULT);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_PREFILL_OPT(256, TOKEN_CHUNK_SIZE_DEFAULT);
      break;
    default:
      break;
  }
}

#define CALL_PREFILL_OPT_LAUNCHER(T, cache_T, BLOCK_SIZE)                             \
  paged_attention_prefill_opt_launcher<T, cache_T, BLOCK_SIZE>(                       \
    out,                                                                \
    query,                                                                \
    key_cache,                                                            \
    value_cache,                                                          \
    reinterpret_cast<float*>(k_scales),                                                             \
    reinterpret_cast<float*>(v_scales),                                                         \
    num_kv_heads,                                                                     \
    scale,                                                                            \
    block_tables,                                                                     \
    context_lens,                                                                     \
    max_num_blocks_per_seq,                                                           \
    num_seqs,\
    num_query_heads,                                                        \
    num_query_tokens,\
    head_size, \
    softscapping,                                                         \
    o_stride_tokens,                                                          \
    query_start_len,                                             \
    sinks,                                                               \
    sliding_window,                                                          \
    num_blocks,\
    kv_block_stride,\
    kv_head_stride,\
    stream);

#define CALL_PREFILL_OPT_LAUNCHER_BLOCK_SIZE(T, cache_T)                               \
  switch (block_size) {                                                   \
    case 32:                                                              \
      CALL_PREFILL_OPT_LAUNCHER(T, cache_T, 32);                                          \
      break;                                                              \
    case 64:                                                              \
      CALL_PREFILL_OPT_LAUNCHER(T, cache_T, 64);                                          \
      break;                                                              \
    default:                                                              \
      break;                                                              \
  }

extern "C" void paged_attention_prefill_opt(
  void *out,             // [num_seqs, num_heads, head_size]
  void *query,           // [num_seqs, num_heads, head_size]
  void *key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  void *value_cache,     // [num_blocks, num_heads, head_size, block_size]
  void * k_scales,
  void * v_scales,
  int32_t num_kv_heads,               // [num_heads]
  float scale,
  uint32_t *block_tables,    // [num_seqs, max_num_blocks_per_seq]
  uint32_t *context_lens,    // [num_seqs]
  int32_t block_size,
  int32_t max_context_len,

  int32_t num_seqs,
  int32_t num_query_heads,
  int32_t num_query_tokens,
  int32_t head_size,
  int32_t max_num_blocks_per_seq,
  int32_t q_stride,
  int32_t num_blocks,
  int32_t kv_block_stride,
  int32_t kv_head_stride,

  uint32_t dtype,      // 0 => f16; 1 => bf16; 2 => f32
  float softscapping,

  int32_t o_stride_tokens,      // out.stride(0)
  uint32_t* query_start_len, // [num_seqs+1] or nullptr
  float* sinks,  // [num_query_heads] or nullptr
  int32_t sliding_window,
  int64_t stream
  ) {

  if (k_scales != nullptr || v_scales != nullptr) {
#ifndef NO_FP8_KVCACHE
    if (dtype == 2) {
      // CALL_PREFILL_OPT_LAUNCHER_BLOCK_SIZE(float);
    } else if (dtype == 0) {
      CALL_PREFILL_OPT_LAUNCHER_BLOCK_SIZE(uint16_t, uint8_t);
    } else if (dtype == 1) {
      #ifndef NO_BF16_KERNEL //cuda_arc < 800 (no bf16 support)
      CALL_PREFILL_OPT_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, uint8_t);
      #endif
    }
#else
    throw std::runtime_error("Error: FP8 KV-cache is disabled (possiblly because flashattn or context-cache enabled).");
#endif
  } else {
    if (dtype == 2) {
      // CALL_PREFILL_OPT_LAUNCHER_BLOCK_SIZE(float);
    } else if (dtype == 0) {
      CALL_PREFILL_OPT_LAUNCHER_BLOCK_SIZE(uint16_t, uint16_t);
    } else if (dtype == 1) {
      #ifndef NO_BF16_KERNEL //cuda_arc < 800 (no bf16 support)
      CALL_PREFILL_OPT_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, __nv_bfloat16);
      #endif
    }
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
