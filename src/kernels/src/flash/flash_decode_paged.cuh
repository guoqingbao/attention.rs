/**
 * @brief Native flash decode attention — paged half KV cache, SM75+ (FP16) / SM80+ (BF16).
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_decode_paged.cuh
 * 
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 *
 * @details
 * Per-Q-head paged decode with online softmax. 8 warps split the KV
 * sequence; each thread covers head_dim/32 BF16 elements. Batched (BC=4)
 * score/V accumulation reduces __expf calls. GQA_RATIO compile-time
 * parameter controls Q heads per CTA (always 1 in current dispatch).
 * Includes main decode, split-K, and reduce kernels.
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

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef GQA_RATIO
#define GQA_RATIO 1
#endif

#define WARP_SIZE 32
#define HDIM FLASH_HDIM
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4

#ifndef FLASH_DECODE_UNPACK_DEFINED
#define FLASH_DECODE_UNPACK_DEFINED
__device__ __forceinline__ void unpack2_bf16_d(unsigned int packed, float& v0, float& v1) {
    const unsigned short lo = (unsigned short)(packed & 0xFFFF);
    const unsigned short hi = (unsigned short)(packed >> 16);
    v0 = FLASH_HALF2FLOAT(*reinterpret_cast<const flash_half_t*>(&lo));
    v1 = FLASH_HALF2FLOAT(*reinterpret_cast<const flash_half_t*>(&hi));
}
#endif

#ifndef LDG_VEC_DEFINED
#define LDG_VEC_DEFINED
#endif

#define LDG_VEC_LOAD(addr, dst, count) \
    do { \
        _Pragma("unroll") \
        for (int _lv_i = 0; _lv_i < (count); _lv_i++) (dst)[_lv_i] = __ldg((addr) + _lv_i); \
    } while(0)

// ============================================================================
// Paged decode — GQA-native
// Grid: GQA_RATIO==1 ? (num_q_heads, num_seqs) : (num_kv_heads, num_seqs)
// Block: (256, 1, 1)
// ============================================================================

extern "C" __global__ void flash_decode_paged(
    const flash_half_t* __restrict__ Q,
    const flash_half_t* __restrict__ K_cache,
    const flash_half_t* __restrict__ V_cache,
    flash_half_t* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int sliding_window,
    const float softcap
) {
    const unsigned int head_idx = blockIdx.x; // q_head when GQA=1, kv_head when GQA>1
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

#if GQA_RATIO == 1
    const unsigned int q_head_start = head_idx;
    const unsigned int kv_head = head_idx / (num_q_heads / num_kv_heads);
    if (head_idx >= num_q_heads) return;
#else
    const unsigned int kv_head = head_idx;
    const unsigned int q_head_start = kv_head * GQA_RATIO;
    if (kv_head >= num_kv_heads) return;
#endif

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int vec_offset = lane_id * VEC_BF16;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    float q_reg[GQA_RATIO][VEC_BF16];
    #pragma unroll
    for (int g = 0; g < GQA_RATIO; g++) {
        const unsigned int* q32 = (const unsigned int*)(Q
            + (unsigned long long)seq_idx * q_stride
            + (unsigned long long)(q_head_start + g) * head_dim + vec_offset);
        unsigned int qp[VEC_U32];
        LDG_VEC_LOAD(q32, qp, VEC_U32);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++)
            unpack2_bf16_d(qp[i], q_reg[g][2*i], q_reg[g][2*i+1]);
    }

    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m_val[GQA_RATIO], l_val[GQA_RATIO];
    float o_reg[GQA_RATIO][VEC_BF16];
    #pragma unroll
    for (int g = 0; g < GQA_RATIO; g++) {
        m_val[g] = -1e30f; l_val[g] = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] = 0.f;
    }

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const flash_half_t* k_block_base = K_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;
        const flash_half_t* v_block_base = V_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;

        unsigned int t = 0;
        unsigned int aligned_bc = (batch_count / BC) * BC;

        for (; t < aligned_bc; t += BC) {
            float scores[GQA_RATIO][BC];
            float vf_batch[BC][VEC_BF16];

            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(t + b) * head_stride_kv + vec_offset);
                unsigned int kp[VEC_U32];
                LDG_VEC_LOAD(k32, kp, VEC_U32);
                float kf[VEC_BF16];
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    unpack2_bf16_d(kp[i], kf[2*i], kf[2*i+1]);

                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(t + b) * head_stride_kv + vec_offset);
                unsigned int vp2[VEC_U32];
                LDG_VEC_LOAD(v32, vp2, VEC_U32);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    unpack2_bf16_d(vp2[i], vf_batch[b][2*i], vf_batch[b][2*i+1]);

                #pragma unroll
                for (int g = 0; g < GQA_RATIO; g++) {
                    float dot = 0.f;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) dot += q_reg[g][i] * kf[i];
                    #pragma unroll
                    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, off);
                    scores[g][b] = dot * inv_sqrt_d;
                    if (softcap > 0.f) scores[g][b] = softcap * tanhf(scores[g][b] / softcap);
                }
            }

            #pragma unroll
            for (int g = 0; g < GQA_RATIO; g++) {
                float m_new = m_val[g];
                #pragma unroll
                for (int b = 0; b < BC; b++) m_new = fmaxf(m_new, scores[g][b]);
                float exp_old = __expf(m_val[g] - m_new);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] *= exp_old;
                l_val[g] *= exp_old;

                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float w = __expf(scores[g][b] - m_new);
                    l_val[g] += w;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] += w * vf_batch[b][i];
                }
                m_val[g] = m_new;
            }
        }

        for (; t < batch_count; t++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)t * head_stride_kv + vec_offset);
            unsigned int kp[VEC_U32];
            LDG_VEC_LOAD(k32, kp, VEC_U32);
            float kf[VEC_BF16];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++)
                unpack2_bf16_d(kp[i], kf[2*i], kf[2*i+1]);

            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)t * head_stride_kv + vec_offset);
            unsigned int vp[VEC_U32];
            LDG_VEC_LOAD(v32, vp, VEC_U32);
            float vf[VEC_BF16];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++)
                unpack2_bf16_d(vp[i], vf[2*i], vf[2*i+1]);

            #pragma unroll
            for (int g = 0; g < GQA_RATIO; g++) {
                float dot = 0.f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) dot += q_reg[g][i] * kf[i];
                #pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                float score = dot * inv_sqrt_d;
                if (softcap > 0.f) score = softcap * tanhf(score / softcap);

                float m_new = fmaxf(m_val[g], score);
                float exp_old = __expf(m_val[g] - m_new);
                float exp_new = __expf(score - m_new);
                l_val[g] = l_val[g] * exp_old + exp_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[g][i] = o_reg[g][i] * exp_old + exp_new * vf[i];
                m_val[g] = m_new;
            }
        }
        pos += batch_count;
    }

    // Inter-warp reduction
    __shared__ float smem_m[GQA_RATIO][NUM_WARPS];
    __shared__ float smem_l[GQA_RATIO][NUM_WARPS];
    __shared__ float smem_o[GQA_RATIO][NUM_WARPS][HDIM];

    #pragma unroll
    for (int g = 0; g < GQA_RATIO; g++) {
        if (lane_id == 0) { smem_m[g][warp_id] = m_val[g]; smem_l[g][warp_id] = l_val[g]; }
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            smem_o[g][warp_id][vec_offset + i] = o_reg[g][i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            #pragma unroll
            for (int g = 0; g < GQA_RATIO; g++) {
                float lw = smem_l[g][other];
                if (lw > 0.f) {
                    float mw = smem_m[g][other], my_m = smem_m[g][warp_id], my_l = smem_l[g][warp_id];
                    float m_new = fmaxf(my_m, mw);
                    float scale_me = __expf(my_m - m_new), scale_w = __expf(mw - m_new);
                    smem_l[g][warp_id] = my_l * scale_me + lw * scale_w;
                    smem_m[g][warp_id] = m_new;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++)
                        smem_o[g][warp_id][vec_offset + i] =
                            smem_o[g][warp_id][vec_offset + i] * scale_me +
                            smem_o[g][other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        #pragma unroll
        for (int g = 0; g < GQA_RATIO; g++) {
            float final_l = smem_l[g][0];
            float inv_l = (final_l > 0.f) ? (1.f / final_l) : 0.f;
            unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                                  + (unsigned long long)(q_head_start + g) * head_dim + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0 = smem_o[g][0][vec_offset + 2*i]     * inv_l;
                float v1 = smem_o[g][0][vec_offset + 2*i + 1] * inv_l;
                unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v0));
                unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v1));
                o32[i] = lo | (hi << 16);
            }
        }
    }
}

// ============================================================================
// Split-K variant
// Grid: GQA_RATIO==1 ? (num_q_heads, num_splits, num_seqs)
//                     : (num_kv_heads, num_splits, num_seqs)
// Block: (256,1,1)
// ============================================================================

extern "C" __global__ void flash_decode_paged_splitk(
    const flash_half_t* __restrict__ Q,
    const flash_half_t* __restrict__ K_cache,
    const flash_half_t* __restrict__ V_cache,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const float softcap,
    const unsigned int sliding_window
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

#if GQA_RATIO == 1
    const unsigned int q_head_start = head_idx;
    const unsigned int kv_head = head_idx / (num_q_heads / num_kv_heads);
    if (head_idx >= num_q_heads) return;
#else
    const unsigned int kv_head = head_idx;
    const unsigned int q_head_start = kv_head * GQA_RATIO;
    if (kv_head >= num_kv_heads) return;
#endif

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;
    const unsigned int attended = seq_len - window_start;
    unsigned int split_size = (attended + num_splits - 1) / num_splits;
    unsigned int kv_start = window_start + split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int vec_offset = lane_id * VEC_BF16;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    float q_reg[GQA_RATIO][VEC_BF16];
    #pragma unroll
    for (int g = 0; g < GQA_RATIO; g++) {
        const unsigned int* q32 = (const unsigned int*)(Q
            + (unsigned long long)seq_idx * q_stride
            + (unsigned long long)(q_head_start + g) * head_dim + vec_offset);
        unsigned int qp[VEC_U32];
        LDG_VEC_LOAD(q32, qp, VEC_U32);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++)
            unpack2_bf16_d(qp[i], q_reg[g][2*i], q_reg[g][2*i+1]);
    }

    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end_pos = my_start + chunk_size;
    if (my_end_pos > kv_end) my_end_pos = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val[GQA_RATIO], l_val[GQA_RATIO];
    float o_reg[GQA_RATIO][VEC_BF16];
    #pragma unroll
    for (int g = 0; g < GQA_RATIO; g++) {
        m_val[g] = -1e30f; l_val[g] = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] = 0.f;
    }

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;

    {
        unsigned int pos = my_start;
        while (pos < my_end_pos) {
            unsigned int logical_block = pos / block_size;
            unsigned int block_offset_s = pos % block_size;
            unsigned int physical_block = (unsigned int)my_block_table[logical_block];
            unsigned int remaining_in_block = block_size - block_offset_s;
            unsigned int remaining_total = my_end_pos - pos;
            unsigned int bc = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

            const flash_half_t* k_base = K_cache + (unsigned long long)physical_block * page_stride
                + (unsigned long long)block_offset_s * head_stride_kv
                + (unsigned long long)kv_head * head_dim;
            const flash_half_t* v_base = V_cache + (unsigned long long)physical_block * page_stride
                + (unsigned long long)block_offset_s * head_stride_kv
                + (unsigned long long)kv_head * head_dim;

            unsigned int t = 0;
            unsigned int aligned_bc = (bc / BC) * BC;

            for (; t < aligned_bc; t += BC) {
                float scores[GQA_RATIO][BC];
                float vf_batch[BC][VEC_BF16];

                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    const unsigned int* k32 = (const unsigned int*)(k_base
                        + (unsigned long long)(t + b) * head_stride_kv + vec_offset);
                    unsigned int kp2[VEC_U32];
                    LDG_VEC_LOAD(k32, kp2, VEC_U32);
                    float kf[VEC_BF16];
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) unpack2_bf16_d(kp2[i], kf[2*i], kf[2*i+1]);

                    const unsigned int* v32 = (const unsigned int*)(v_base
                        + (unsigned long long)(t + b) * head_stride_kv + vec_offset);
                    unsigned int vp2[VEC_U32];
                    LDG_VEC_LOAD(v32, vp2, VEC_U32);
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) unpack2_bf16_d(vp2[i], vf_batch[b][2*i], vf_batch[b][2*i+1]);

                    #pragma unroll
                    for (int g = 0; g < GQA_RATIO; g++) {
                        float dot = 0.f;
                        #pragma unroll
                        for (int i = 0; i < VEC_BF16; i++) dot += q_reg[g][i] * kf[i];
                        #pragma unroll
                        for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                            dot += __shfl_xor_sync(0xffffffff, dot, off);
                        scores[g][b] = dot * inv_sqrt_d;
                        if (softcap > 0.f) scores[g][b] = softcap * tanhf(scores[g][b] / softcap);
                    }
                }

                #pragma unroll
                for (int g = 0; g < GQA_RATIO; g++) {
                    float m_new = m_val[g];
                    #pragma unroll
                    for (int b = 0; b < BC; b++) m_new = fmaxf(m_new, scores[g][b]);
                    float exp_old = __expf(m_val[g] - m_new);
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] *= exp_old;
                    l_val[g] *= exp_old;
                    #pragma unroll
                    for (int b = 0; b < BC; b++) {
                        float w = __expf(scores[g][b] - m_new);
                        l_val[g] += w;
                        #pragma unroll
                        for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] += w * vf_batch[b][i];
                    }
                    m_val[g] = m_new;
                }
            }

            for (; t < bc; t++) {
                const unsigned int* k32 = (const unsigned int*)(k_base
                    + (unsigned long long)t * head_stride_kv + vec_offset);
                unsigned int kp[VEC_U32];
                LDG_VEC_LOAD(k32, kp, VEC_U32);
                float kf[VEC_BF16];
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) unpack2_bf16_d(kp[i], kf[2*i], kf[2*i+1]);

                const unsigned int* v32 = (const unsigned int*)(v_base
                    + (unsigned long long)t * head_stride_kv + vec_offset);
                unsigned int vp[VEC_U32];
                LDG_VEC_LOAD(v32, vp, VEC_U32);
                float vf[VEC_BF16];
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) unpack2_bf16_d(vp[i], vf[2*i], vf[2*i+1]);

                #pragma unroll
                for (int g = 0; g < GQA_RATIO; g++) {
                    float dot = 0.f;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) dot += q_reg[g][i] * kf[i];
                    #pragma unroll
                    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, off);
                    float score = dot * inv_sqrt_d;
                    if (softcap > 0.f) score = softcap * tanhf(score / softcap);

                    float m_new = fmaxf(m_val[g], score);
                    float exp_old = __expf(m_val[g] - m_new), exp_new = __expf(score - m_new);
                    l_val[g] = l_val[g] * exp_old + exp_new;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++)
                        o_reg[g][i] = o_reg[g][i] * exp_old + exp_new * vf[i];
                    m_val[g] = m_new;
                }
            }
            pos += bc;
        }
    }

    __shared__ float smem_m[GQA_RATIO][NUM_WARPS];
    __shared__ float smem_l[GQA_RATIO][NUM_WARPS];
    __shared__ float smem_o[GQA_RATIO][NUM_WARPS][HDIM];

    #pragma unroll
    for (int g = 0; g < GQA_RATIO; g++) {
        if (lane_id == 0) { smem_m[g][warp_id] = m_val[g]; smem_l[g][warp_id] = l_val[g]; }
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            smem_o[g][warp_id][vec_offset + i] = o_reg[g][i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            #pragma unroll
            for (int g = 0; g < GQA_RATIO; g++) {
                float lw = smem_l[g][other];
                if (lw > 0.f) {
                    float mw = smem_m[g][other], my_m = smem_m[g][warp_id], my_l = smem_l[g][warp_id];
                    float m_new = fmaxf(my_m, mw);
                    float scale_me = __expf(my_m - m_new), scale_w = __expf(mw - m_new);
                    smem_l[g][warp_id] = my_l * scale_me + lw * scale_w;
                    smem_m[g][warp_id] = m_new;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++)
                        smem_o[g][warp_id][vec_offset + i] =
                            smem_o[g][warp_id][vec_offset + i] * scale_me +
                            smem_o[g][other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Write workspace: layout [seq][q_head][split][hd+2]
    unsigned int ws_stride = head_dim + 2;
    if (warp_id == 0) {
        #pragma unroll
        for (int g = 0; g < GQA_RATIO; g++) {
            float* ws = workspace + ((unsigned long long)seq_idx * num_q_heads + (q_head_start + g)) * num_splits * ws_stride
                       + split_id * ws_stride;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) ws[vec_offset + i] = smem_o[g][0][vec_offset + i];
            if (lane_id == 0) { ws[head_dim] = smem_m[g][0]; ws[head_dim + 1] = smem_l[g][0]; }
        }
    }
}

// ============================================================================
// Reduce split-K partials (GQA-independent, works for any ratio)
// Grid: (num_q_heads, num_seqs, 1)  Block: (32,1,1)
// ============================================================================

extern "C" __global__ void flash_decode_paged_reduce(
    const float* __restrict__ workspace,
    flash_half_t* __restrict__ O,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;
    const unsigned int vec_off = lane_id * VEC_BF16;

    if (q_head >= num_q_heads) return;

    unsigned int ws_stride = head_dim + 2;
    const float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    float m = ws_base[head_dim];
    float l = ws_base[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = ws_base[vec_off + i];

    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + s * ws_stride;
        float ms = ws[head_dim], ls = ws[head_dim + 1];
        if (ls <= 0.f) continue;
        float m_new = fmaxf(m, ms);
        float scale_me = __expf(m - m_new), scale_s = __expf(ms - m_new);
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;
        l = l * scale_me + ls * scale_s;
        m = m_new;
    }

    float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_off);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i] * inv_l, v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v0));
        unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v1));
        o32[i] = lo | (hi << 16);
    }
}
