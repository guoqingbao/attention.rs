/**
 * @brief TurboQuant KV cache — low-bit variants (turbo4/turbo3), SM80+.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_turboquant_lowbit.cuh
 * 
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 *
 * @details
 * Implements turbo4 (4-bit K + 4-bit V) and turbo3 (3-bit K + 4-bit V)
 * TurboQuant modes. Keys use Walsh-Hadamard Transform rotation with random
 * sign flips before quantization. Values use per-position absmax 4-bit
 * uniform quantization. Includes store, decode (main + split-K), and
 * WHT/sign-flip helper functions. Based on TurboQuant (Zandieh et al.,
 * ICLR 2026).
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
// Keys: WHT rotation → per-head absmax → N-bit uniform quantization
// Values: per-head absmax → 4-bit uniform quantization
// WHT rotation on keys spreads outlier energy across all channels.
//
// Cache layout per token-slot for turbo4 (4-bit keys + 4-bit values):
//   K_absmax: [num_blocks, block_size, num_kv_heads] as float
//   K_quant:  [num_blocks, block_size, num_kv_heads, head_dim/2] as uint8 (packed 4-bit)
//   V_absmax: [num_blocks, block_size, num_kv_heads] as float
//   V_quant:  [num_blocks, block_size, num_kv_heads, head_dim/2] as uint8 (packed 4-bit)
//
// Cache layout for turbo3 (3-bit keys + 4-bit values):
//   K_absmax: [num_blocks, block_size, num_kv_heads] as float
//   K_quant:  [num_blocks, block_size, num_kv_heads, ceil(head_dim*3/8)] as uint8 (packed 3-bit)
//   V_absmax: [num_blocks, block_size, num_kv_heads] as float
//   V_quant:  [num_blocks, block_size, num_kv_heads, head_dim/2] as uint8 (packed 4-bit)

#include "flash_sm_compat.cuh"

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

#define TQ4_VEC (HDIM / WARP_SIZE)

#ifndef TQ_HELPERS_DEFINED
#define TQ_HELPERS_DEFINED
// Already defined in flash_turboquant.cuh, but guard for safety
#endif

// ============================================================================
// turbo4: 4-bit keys + 4-bit values store kernel
// ============================================================================

extern "C" __global__ void flash_tq4_store(
    const flash_half_t* __restrict__ K,
    const flash_half_t* __restrict__ V,
    float* __restrict__ K_absmax,
    unsigned char* __restrict__ K_quant,
    float* __restrict__ V_absmax,
    unsigned char* __restrict__ V_quant,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;
    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    unsigned int block_idx = (unsigned int)(slot / block_size);
    unsigned int block_off = (unsigned int)(slot % block_size);

    unsigned int base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

    // Load K, apply sign flip, WHT
    float k_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        k_reg[i] = FLASH_HALF2FLOAT(K[base + ch]);
        k_reg[i] *= get_sign_flip(head_idx, ch);
    }
    wht_transform(k_reg, lane_id);

    // K absmax reduction
    float k_absmax = 0.f;
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) k_absmax = fmaxf(k_absmax, fabsf(k_reg[i]));
    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        k_absmax = fmaxf(k_absmax, __shfl_xor_sync(0xffffffff, k_absmax, off));

    unsigned long long am_off = (unsigned long long)block_idx * block_size * num_kv_heads
        + (unsigned long long)block_off * num_kv_heads + head_idx;
    if (lane_id == 0) K_absmax[am_off] = k_absmax;
    k_absmax = __shfl_sync(0xffffffff, k_absmax, 0);

    // Quantize K to 4-bit and pack
    float k_inv_absmax = (k_absmax > 0.f) ? (1.f / k_absmax) : 0.f;
    unsigned long long kq_off = (unsigned long long)block_idx * block_size * num_kv_heads * (head_dim / 2)
        + (unsigned long long)block_off * num_kv_heads * (head_dim / 2)
        + (unsigned long long)head_idx * (head_dim / 2);
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i += 2) {
        unsigned char q_lo = quantize_4bit(k_reg[i], k_inv_absmax);
        unsigned char q_hi = quantize_4bit(k_reg[i+1], k_inv_absmax);
        unsigned int byte_idx = (lane_id * TQ4_VEC + i) / 2;
        K_quant[kq_off + byte_idx] = pack_4bit(q_lo, q_hi);
    }

    // Load V
    float v_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        v_reg[i] = FLASH_HALF2FLOAT(V[base + ch]);
    }

    // V absmax
    float v_absmax = 0.f;
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) v_absmax = fmaxf(v_absmax, fabsf(v_reg[i]));
    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        v_absmax = fmaxf(v_absmax, __shfl_xor_sync(0xffffffff, v_absmax, off));

    if (lane_id == 0) V_absmax[am_off] = v_absmax;
    v_absmax = __shfl_sync(0xffffffff, v_absmax, 0);

    // Quantize V to 4-bit
    float v_inv_absmax = (v_absmax > 0.f) ? (1.f / v_absmax) : 0.f;
    unsigned long long vq_off = kq_off; // same layout
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i += 2) {
        unsigned char q_lo = quantize_4bit(v_reg[i], v_inv_absmax);
        unsigned char q_hi = quantize_4bit(v_reg[i+1], v_inv_absmax);
        unsigned int byte_idx = (lane_id * TQ4_VEC + i) / 2;
        V_quant[vq_off + byte_idx] = pack_4bit(q_lo, q_hi);
    }
}

// ============================================================================
// turbo4 decode: 4-bit K + 4-bit V → attention
// Optimized: 8 warps (256 threads), BC=2 batching, block-contiguous iteration
// ============================================================================

#ifndef TQ4_NUM_WARPS
#define TQ4_NUM_WARPS 8
#endif
#define TQ4_VEC_U32 (HDIM / (WARP_SIZE * 2))
#define TQ4_BC 8

#ifndef UNPACK2_BF16_TQ4_DEFINED
#define UNPACK2_BF16_TQ4_DEFINED
__device__ __forceinline__ void unpack2_bf16_tq4(unsigned int packed, float &a, float &b) {
    const unsigned short lo = (unsigned short)(packed & 0xFFFF);
    const unsigned short hi = (unsigned short)(packed >> 16);
    a = FLASH_HALF2FLOAT(*reinterpret_cast<const flash_half_t*>(&lo));
    b = FLASH_HALF2FLOAT(*reinterpret_cast<const flash_half_t*>(&hi));
}
#endif

// Fast inline 4-bit dequant: precompute scale = absmax / 7.5 to avoid per-element division
#define TQ4_DEQUANT_SCALE(absmax) ((absmax) * 0.13333333f)

// Macro: 4-bit K dot product with vectorized uint32 loads and fused dequant.
#define TQ4_K_DOT(q_reg, K_quant, kq_base, ka, lane_id, dot_out) \
    do { \
        float _dot = 0.f; \
        float _ks = TQ4_DEQUANT_SCALE(ka); \
        unsigned int _byte_off = ((lane_id) * TQ4_VEC) / 2; \
        const unsigned char* _kp = (K_quant) + (kq_base) + _byte_off; \
        unsigned int _num_bytes = TQ4_VEC / 2; \
        if (_num_bytes >= 4) { \
            const unsigned int* _kp32 = (const unsigned int*)_kp; \
            _Pragma("unroll") \
            for (unsigned int _g = 0; _g < _num_bytes / 4; _g++) { \
                unsigned int _pk4 = __ldg(_kp32 + _g); \
                _Pragma("unroll") \
                for (int _b = 0; _b < 4; _b++) { \
                    unsigned int _byte = (_pk4 >> (_b * 8)) & 0xFF; \
                    int _vi = _g * 8 + _b * 2; \
                    _dot += (q_reg)[_vi]   * ((float)(_byte & 0xF) - 7.5f) * _ks \
                          + (q_reg)[_vi+1] * ((float)(_byte >> 4)  - 7.5f) * _ks; \
                } \
            } \
        } else if (_num_bytes >= 2) { \
            unsigned short _pk2 = __ldg((const unsigned short*)_kp); \
            unsigned int _b0 = _pk2 & 0xFF, _b1 = (_pk2 >> 8) & 0xFF; \
            _dot += (q_reg)[0] * ((float)(_b0 & 0xF) - 7.5f) * _ks \
                  + (q_reg)[1] * ((float)(_b0 >> 4)  - 7.5f) * _ks; \
            _dot += (q_reg)[2] * ((float)(_b1 & 0xF) - 7.5f) * _ks \
                  + (q_reg)[3] * ((float)(_b1 >> 4)  - 7.5f) * _ks; \
        } else { \
            _Pragma("unroll") \
            for (int _i = 0; _i < TQ4_VEC; _i += 2) { \
                unsigned int _bi = ((lane_id) * TQ4_VEC + _i) / 2; \
                unsigned int _pk = __ldg((K_quant) + (kq_base) + _bi); \
                _dot += (q_reg)[_i]   * ((float)(_pk & 0xF) - 7.5f) * _ks \
                      + (q_reg)[_i+1] * ((float)(_pk >> 4)  - 7.5f) * _ks; \
            } \
        } \
        _Pragma("unroll") \
        for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1) \
            _dot += __shfl_xor_sync(0xffffffff, _dot, _off); \
        (dot_out) = _dot; \
    } while(0)

// Macro: 4-bit V accumulate with vectorized loads and fused scale*weight.
#define TQ4_V_ACCUM(o_reg, V_quant, vq_base, va, weight, lane_id) \
    do { \
        float _vs = TQ4_DEQUANT_SCALE(va) * (weight); \
        unsigned int _byte_off = ((lane_id) * TQ4_VEC) / 2; \
        const unsigned char* _vp = (V_quant) + (vq_base) + _byte_off; \
        unsigned int _num_bytes = TQ4_VEC / 2; \
        if (_num_bytes >= 4) { \
            const unsigned int* _vp32 = (const unsigned int*)_vp; \
            _Pragma("unroll") \
            for (unsigned int _g = 0; _g < _num_bytes / 4; _g++) { \
                unsigned int _pk4 = __ldg(_vp32 + _g); \
                _Pragma("unroll") \
                for (int _b = 0; _b < 4; _b++) { \
                    unsigned int _byte = (_pk4 >> (_b * 8)) & 0xFF; \
                    int _vi = _g * 8 + _b * 2; \
                    (o_reg)[_vi]   += ((float)(_byte & 0xF) - 7.5f) * _vs; \
                    (o_reg)[_vi+1] += ((float)(_byte >> 4)  - 7.5f) * _vs; \
                } \
            } \
        } else if (_num_bytes >= 2) { \
            unsigned short _pk2 = __ldg((const unsigned short*)_vp); \
            unsigned int _b0 = _pk2 & 0xFF, _b1 = (_pk2 >> 8) & 0xFF; \
            (o_reg)[0] += ((float)(_b0 & 0xF) - 7.5f) * _vs; \
            (o_reg)[1] += ((float)(_b0 >> 4)  - 7.5f) * _vs; \
            (o_reg)[2] += ((float)(_b1 & 0xF) - 7.5f) * _vs; \
            (o_reg)[3] += ((float)(_b1 >> 4)  - 7.5f) * _vs; \
        } else { \
            _Pragma("unroll") \
            for (int _i = 0; _i < TQ4_VEC; _i += 2) { \
                unsigned int _bi = ((lane_id) * TQ4_VEC + _i) / 2; \
                unsigned int _pk = __ldg((V_quant) + (vq_base) + _bi); \
                (o_reg)[_i]   += ((float)(_pk & 0xF) - 7.5f) * _vs; \
                (o_reg)[_i+1] += ((float)(_pk >> 4)  - 7.5f) * _vs; \
            } \
        } \
    } while(0)

extern "C" __global__ void flash_tq4_decode(
    const flash_half_t* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
    flash_half_t* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_seqs,
    const unsigned int q_stride,
    const float softcap,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads || seq_idx >= num_seqs) return;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    // Load Q, apply sign flip + WHT
    const unsigned int bf16_vec_off = lane_id * TQ4_VEC;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC / 2; i++) unpack2_bf16_tq4(q32[i], q_reg[2*i], q_reg[2*i+1]);

    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        q_reg[i] *= get_sign_flip(kv_head, ch);
    }
    wht_transform(q_reg, lane_id);

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + TQ4_NUM_WARPS - 1) / TQ4_NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) o_reg[i] = 0.f;

    const unsigned int hd_half = head_dim / 2;
    const unsigned long long am_stride = (unsigned long long)block_size * num_kv_heads;
    const unsigned long long qv_stride = (unsigned long long)block_size * num_kv_heads * hd_half;

    // Process positions in BC=2 batches within contiguous blocks
    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        unsigned long long am_block_base = (unsigned long long)physical_block * am_stride
            + (unsigned long long)kv_head;
        unsigned long long qv_block_base = (unsigned long long)physical_block * qv_stride
            + (unsigned long long)kv_head * hd_half;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / TQ4_BC) * TQ4_BC;

        for (; processed < aligned; processed += TQ4_BC) {
            float scores[TQ4_BC];
            unsigned long long am[TQ4_BC], qv[TQ4_BC];

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                am[b] = am_block_base + (unsigned long long)bo * num_kv_heads;
                qv[b] = qv_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float ka = __ldg(K_absmax + am[b]);
                float _s;
                TQ4_K_DOT(q_reg, K_quant, qv[b], ka, lane_id, _s);
                scores[b] = _s * inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            float exp_factors[TQ4_BC];
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l_val += exp_factors[b];
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float va = __ldg(V_absmax + am[b]);
                TQ4_V_ACCUM(o_reg, V_quant, qv[b], va, exp_factors[b], lane_id);
            }
            m_val = m_new;
        }

        for (; processed < batch_count; processed++) {
            unsigned int bo = block_offset + processed;
            unsigned long long am_s = am_block_base + (unsigned long long)bo * num_kv_heads;
            unsigned long long qv_s = qv_block_base + (unsigned long long)bo * num_kv_heads * hd_half;

            float ka = __ldg(K_absmax + am_s);
            float s;
            TQ4_K_DOT(q_reg, K_quant, qv_s, ka, lane_id, s);
            s *= inv_sqrt_d;
            if (softcap > 0.f) s = softcap * tanhf(s / softcap);

            float m_new = fmaxf(m_val, s);
            float exp_old = __expf(m_val - m_new), e = __expf(s - m_new);
            l_val = l_val * exp_old + e;
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;

            float va = __ldg(V_absmax + am_s);
            TQ4_V_ACCUM(o_reg, V_quant, qv_s, va, e, lane_id);
            m_val = m_new;
        }

        pos += batch_count;
    }

    // Warp reduction across TQ4_NUM_WARPS
    __shared__ float smem_m[TQ4_NUM_WARPS];
    __shared__ float smem_l[TQ4_NUM_WARPS];
    __shared__ float smem_o[TQ4_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = TQ4_NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.f) {
                float mw = smem_m[other], my_m = smem_m[warp_id], my_l = smem_l[warp_id];
                float mn = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - mn), scale_w = __expf(mw - mn);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = mn;
                #pragma unroll
                for (int i = 0; i < TQ4_VEC; i++)
                    smem_o[warp_id][bf16_vec_off + i] =
                        smem_o[warp_id][bf16_vec_off + i] * scale_me +
                        smem_o[other][bf16_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.f) ? (1.f / final_l) : 0.f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + bf16_vec_off);
        #pragma unroll
        for (int i = 0; i < TQ4_VEC_U32; i++) {
            float v0 = smem_o[0][bf16_vec_off + 2*i]     * inv_l;
            float v1 = smem_o[0][bf16_vec_off + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v0));
            unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// Split-K TQ4 decode for long sequences
// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256,1,1)
// ============================================================================

extern "C" __global__ void flash_tq4_decode_splitk(
    const flash_half_t* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
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
    const unsigned int num_seqs,
    const unsigned int q_stride,
    const float softcap,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads || seq_idx >= num_seqs) return;
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

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int bf16_vec_off = lane_id * TQ4_VEC;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC / 2; i++) unpack2_bf16_tq4(q32[i], q_reg[2*i], q_reg[2*i+1]);

    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        q_reg[i] *= get_sign_flip(kv_head, ch);
    }
    wht_transform(q_reg, lane_id);

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + TQ4_NUM_WARPS - 1) / TQ4_NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) o_reg[i] = 0.f;

    const unsigned int hd_half = head_dim / 2;
    const unsigned long long am_stride = (unsigned long long)block_size * num_kv_heads;
    const unsigned long long qv_stride = (unsigned long long)block_size * num_kv_heads * hd_half;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        unsigned long long am_block_base = (unsigned long long)physical_block * am_stride
            + (unsigned long long)kv_head;
        unsigned long long qv_block_base = (unsigned long long)physical_block * qv_stride
            + (unsigned long long)kv_head * hd_half;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / TQ4_BC) * TQ4_BC;

        for (; processed < aligned; processed += TQ4_BC) {
            float scores[TQ4_BC];
            unsigned long long am[TQ4_BC], qv[TQ4_BC];

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                am[b] = am_block_base + (unsigned long long)bo * num_kv_heads;
                qv[b] = qv_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float ka = __ldg(K_absmax + am[b]);
                float _s;
                TQ4_K_DOT(q_reg, K_quant, qv[b], ka, lane_id, _s);
                scores[b] = _s * inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            float exp_factors[TQ4_BC];
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l_val += exp_factors[b];
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float va = __ldg(V_absmax + am[b]);
                TQ4_V_ACCUM(o_reg, V_quant, qv[b], va, exp_factors[b], lane_id);
            }
            m_val = m_new;
        }

        for (; processed < batch_count; processed++) {
            unsigned int bo = block_offset + processed;
            unsigned long long am_s = am_block_base + (unsigned long long)bo * num_kv_heads;
            unsigned long long qv_s = qv_block_base + (unsigned long long)bo * num_kv_heads * hd_half;

            float ka = __ldg(K_absmax + am_s);
            float s;
            TQ4_K_DOT(q_reg, K_quant, qv_s, ka, lane_id, s);
            s *= inv_sqrt_d;
            if (softcap > 0.f) s = softcap * tanhf(s / softcap);

            float m_new = fmaxf(m_val, s);
            float exp_old = __expf(m_val - m_new), e = __expf(s - m_new);
            l_val = l_val * exp_old + e;
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;

            float va = __ldg(V_absmax + am_s);
            TQ4_V_ACCUM(o_reg, V_quant, qv_s, va, e, lane_id);
            m_val = m_new;
        }

        pos += batch_count;
    }

    __shared__ float smem_m[TQ4_NUM_WARPS];
    __shared__ float smem_l[TQ4_NUM_WARPS];
    __shared__ float smem_o[TQ4_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = TQ4_NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.f) {
                float mw = smem_m[other], my_m = smem_m[warp_id], my_l = smem_l[warp_id];
                float mn = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - mn), scale_w = __expf(mw - mn);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = mn;
                #pragma unroll
                for (int i = 0; i < TQ4_VEC; i++)
                    smem_o[warp_id][bf16_vec_off + i] =
                        smem_o[warp_id][bf16_vec_off + i] * scale_me +
                        smem_o[other][bf16_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    unsigned int ws_stride = head_dim + 2;
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < TQ4_VEC; i++) ws_base[bf16_vec_off + i] = smem_o[0][bf16_vec_off + i];
        if (lane_id == 0) { ws_base[head_dim] = smem_m[0]; ws_base[head_dim + 1] = smem_l[0]; }
    }
}

// ============================================================================
// turbo3: 3-bit keys + 4-bit values
// ============================================================================
//
// 3-bit packing: groups of 8 channels → 3 bytes (24 bits = 8×3).
// Byte layout within each 3-byte group:
//   byte0: [ch0_b2 ch0_b1 ch0_b0 ch1_b2 ch1_b1 ch1_b0 ch2_b2 ch2_b1]
//   byte1: [ch2_b0 ch3_b2 ch3_b1 ch3_b0 ch4_b2 ch4_b1 ch4_b0 ch5_b2]
//   byte2: [ch5_b1 ch5_b0 ch6_b2 ch6_b1 ch6_b0 ch7_b2 ch7_b1 ch7_b0]
// K_quant stride per head = head_dim * 3 / 8 bytes
//
// Quantization: signed uniform to [-3, 3] range mapped to [0, 7].
//   q = clamp(round(val / absmax * 3.0) + 3, 0, 6) stored as 3-bit unsigned (0-6, not 7)
//   dequant: val = (q - 3) * absmax / 3.0

#define TQ3_K_BYTES_PER_HEAD (HDIM * 3 / 8)

#ifndef TQ3_QUANT_HELPERS_DEFINED
#define TQ3_QUANT_HELPERS_DEFINED

__device__ __forceinline__ unsigned char quantize_3bit(float val, float inv_absmax) {
    float scaled = val * inv_absmax * 3.f;
    int q = __float2int_rn(scaled) + 3;
    q = max(0, min(6, q));
    return (unsigned char)q;
}

__device__ __forceinline__ float dequantize_3bit(unsigned char q, float absmax) {
    return ((float)q - 3.f) * absmax * (1.f / 3.f);
}

// Pack 8 channels (3-bit each) into 3 bytes
__device__ __forceinline__ void pack_3bit_x8(
    const unsigned char vals[8], unsigned char out[3]
) {
    // vals[i] in [0,6], 3 bits each
    unsigned int bits = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) bits |= ((unsigned int)vals[i]) << (i * 3);
    out[0] = (unsigned char)(bits & 0xFF);
    out[1] = (unsigned char)((bits >> 8) & 0xFF);
    out[2] = (unsigned char)((bits >> 16) & 0xFF);
}

// Unpack 8 channels from 3 bytes
__device__ __forceinline__ void unpack_3bit_x8(
    const unsigned char in[3], unsigned char vals[8]
) {
    unsigned int bits = (unsigned int)in[0] | ((unsigned int)in[1] << 8) | ((unsigned int)in[2] << 16);
    #pragma unroll
    for (int i = 0; i < 8; i++) vals[i] = (unsigned char)((bits >> (i * 3)) & 0x7);
}

#endif // TQ3_QUANT_HELPERS_DEFINED

// turbo3 store: K → WHT → 3-bit, V → 4-bit
extern "C" __global__ void flash_tq3_store(
    const flash_half_t* __restrict__ K,
    const flash_half_t* __restrict__ V,
    float* __restrict__ K_absmax,
    unsigned char* __restrict__ K_quant,
    float* __restrict__ V_absmax,
    unsigned char* __restrict__ V_quant,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;
    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    unsigned int block_idx = (unsigned int)(slot / block_size);
    unsigned int block_off = (unsigned int)(slot % block_size);

    unsigned int base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

    // Load K, apply sign flip, WHT
    float k_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        k_reg[i] = FLASH_HALF2FLOAT(K[base + ch]);
        k_reg[i] *= get_sign_flip(head_idx, ch);
    }
    wht_transform(k_reg, lane_id);

    // K absmax
    float k_absmax = 0.f;
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) k_absmax = fmaxf(k_absmax, fabsf(k_reg[i]));
    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        k_absmax = fmaxf(k_absmax, __shfl_xor_sync(0xffffffff, k_absmax, off));

    unsigned long long am_off = (unsigned long long)block_idx * block_size * num_kv_heads
        + (unsigned long long)block_off * num_kv_heads + head_idx;
    if (lane_id == 0) K_absmax[am_off] = k_absmax;
    k_absmax = __shfl_sync(0xffffffff, k_absmax, 0);

    // Quantize K to 3-bit, pack in groups of 8
    float k_inv_absmax = (k_absmax > 0.f) ? (1.f / k_absmax) : 0.f;
    unsigned long long kq_off = (unsigned long long)block_idx * block_size * num_kv_heads * TQ3_K_BYTES_PER_HEAD
        + (unsigned long long)block_off * num_kv_heads * TQ3_K_BYTES_PER_HEAD
        + (unsigned long long)head_idx * TQ3_K_BYTES_PER_HEAD;

    // Each lane handles TQ4_VEC channels. Pack in groups of 8.
    // If TQ4_VEC >= 8: each lane packs TQ4_VEC/8 groups internally.
    // If TQ4_VEC < 8: collaborate across lanes (TQ4_VEC is always multiple of 4; for HDIM=128 VEC=4).
    #if TQ4_VEC >= 8
    #pragma unroll
    for (int g = 0; g < TQ4_VEC / 8; g++) {
        unsigned char q3[8], packed[3];
        #pragma unroll
        for (int i = 0; i < 8; i++) q3[i] = quantize_3bit(k_reg[g * 8 + i], k_inv_absmax);
        pack_3bit_x8(q3, packed);
        unsigned int group_global = (lane_id * TQ4_VEC) / 8 + g;
        unsigned int byte_base = group_global * 3;
        K_quant[kq_off + byte_base + 0] = packed[0];
        K_quant[kq_off + byte_base + 1] = packed[1];
        K_quant[kq_off + byte_base + 2] = packed[2];
    }
    #else
    // TQ4_VEC=4 (HDIM=128): 2 lanes collaborate on 1 group of 8
    {
        unsigned int pair_lane = lane_id ^ 1; // pair with adjacent lane
        unsigned char q3_local[TQ4_VEC];
        #pragma unroll
        for (int i = 0; i < TQ4_VEC; i++) q3_local[i] = quantize_3bit(k_reg[i], k_inv_absmax);

        // Exchange values with pair lane
        unsigned char q3_remote[TQ4_VEC];
        #pragma unroll
        for (int i = 0; i < TQ4_VEC; i++) {
            unsigned int v = (unsigned int)q3_local[i];
            v = __shfl_sync(0xffffffff, v, pair_lane);
            q3_remote[i] = (unsigned char)v;
        }

        // Even lanes do the packing
        if ((lane_id & 1) == 0) {
            unsigned char q3_8[8], packed[3];
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) q3_8[i] = q3_local[i];
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) q3_8[TQ4_VEC + i] = q3_remote[i];
            pack_3bit_x8(q3_8, packed);
            unsigned int group_global = (lane_id * TQ4_VEC) / 8;
            unsigned int byte_base = group_global * 3;
            K_quant[kq_off + byte_base + 0] = packed[0];
            K_quant[kq_off + byte_base + 1] = packed[1];
            K_quant[kq_off + byte_base + 2] = packed[2];
        }
    }
    #endif

    // Load V, compute absmax, quantize to 4-bit (same as turbo4)
    float v_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        v_reg[i] = FLASH_HALF2FLOAT(V[base + ch]);
    }

    float v_absmax = 0.f;
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) v_absmax = fmaxf(v_absmax, fabsf(v_reg[i]));
    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        v_absmax = fmaxf(v_absmax, __shfl_xor_sync(0xffffffff, v_absmax, off));

    if (lane_id == 0) V_absmax[am_off] = v_absmax;
    v_absmax = __shfl_sync(0xffffffff, v_absmax, 0);

    float v_inv_absmax = (v_absmax > 0.f) ? (1.f / v_absmax) : 0.f;
    unsigned long long vq_off = (unsigned long long)block_idx * block_size * num_kv_heads * (head_dim / 2)
        + (unsigned long long)block_off * num_kv_heads * (head_dim / 2)
        + (unsigned long long)head_idx * (head_dim / 2);
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i += 2) {
        unsigned char q_lo = quantize_4bit(v_reg[i], v_inv_absmax);
        unsigned char q_hi = quantize_4bit(v_reg[i+1], v_inv_absmax);
        unsigned int byte_idx = (lane_id * TQ4_VEC + i) / 2;
        V_quant[vq_off + byte_idx] = pack_4bit(q_lo, q_hi);
    }
}

// ============================================================================
// turbo3 decode: 3-bit K + 4-bit V → attention
// Optimized: 8 warps (256 threads), BC=2 batching, vectorized 3-bit unpack
// ============================================================================

// Macro: 3-bit K dot product for a single position.
// Uses TQ4_VEC which varies per HDIM instantiation.
#define TQ3_DEQUANT_SCALE(absmax) ((absmax) * 0.33333333f)

#define TQ3_K_DOT(q_reg, K_quant, kq_base, ka, lane_id, dot_out) \
    do { \
        const unsigned char* _kq_ptr = (K_quant) + (kq_base); \
        float _dot = 0.f; \
        float _ks = TQ3_DEQUANT_SCALE(ka); \
        _Pragma("unroll") \
        for (int _g = 0; _g < (TQ4_VEC + 7) / 8; _g++) { \
            unsigned int _group_global = ((lane_id) * TQ4_VEC) / 8 + _g; \
            unsigned int _byte_base = _group_global * 3; \
            unsigned int _bits = (unsigned int)__ldg(_kq_ptr + _byte_base) \
                | ((unsigned int)__ldg(_kq_ptr + _byte_base + 1) << 8) \
                | ((unsigned int)__ldg(_kq_ptr + _byte_base + 2) << 16); \
            int _n_ch = (TQ4_VEC >= 8) ? 8 : TQ4_VEC; \
            int _ch_start = (TQ4_VEC >= 8) ? 0 : (((lane_id) & 1) * TQ4_VEC); \
            _Pragma("unroll") \
            for (int _i = 0; _i < _n_ch; _i++) { \
                unsigned int _q3 = (_bits >> ((_ch_start + _i) * 3)) & 0x7; \
                int _ri = (TQ4_VEC >= 8) ? (_g * 8 + _i) : _i; \
                _dot += (q_reg)[_ri] * ((float)_q3 - 3.f) * _ks; \
            } \
        } \
        _Pragma("unroll") \
        for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1) \
            _dot += __shfl_xor_sync(0xffffffff, _dot, _off); \
        (dot_out) = _dot; \
    } while(0)

extern "C" __global__ void flash_tq3_decode(
    const flash_half_t* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
    flash_half_t* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_seqs,
    const unsigned int q_stride,
    const float softcap,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads || seq_idx >= num_seqs) return;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int bf16_vec_off = lane_id * TQ4_VEC;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC / 2; i++) unpack2_bf16_tq4(q32[i], q_reg[2*i], q_reg[2*i+1]);

    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        q_reg[i] *= get_sign_flip(kv_head, ch);
    }
    wht_transform(q_reg, lane_id);

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + TQ4_NUM_WARPS - 1) / TQ4_NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) o_reg[i] = 0.f;

    const unsigned long long am_stride = (unsigned long long)block_size * num_kv_heads;
    const unsigned long long kq_stride = (unsigned long long)block_size * num_kv_heads * TQ3_K_BYTES_PER_HEAD;
    const unsigned int hd_half = head_dim / 2;
    const unsigned long long vq_stride = (unsigned long long)block_size * num_kv_heads * hd_half;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        unsigned long long am_block_base = (unsigned long long)physical_block * am_stride + kv_head;
        unsigned long long kq_block_base = (unsigned long long)physical_block * kq_stride
            + (unsigned long long)kv_head * TQ3_K_BYTES_PER_HEAD;
        unsigned long long vq_block_base = (unsigned long long)physical_block * vq_stride
            + (unsigned long long)kv_head * hd_half;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / TQ4_BC) * TQ4_BC;

        for (; processed < aligned; processed += TQ4_BC) {
            float scores[TQ4_BC];
            unsigned long long am[TQ4_BC], kq[TQ4_BC], vq[TQ4_BC];

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                am[b] = am_block_base + (unsigned long long)bo * num_kv_heads;
                kq[b] = kq_block_base + (unsigned long long)bo * num_kv_heads * TQ3_K_BYTES_PER_HEAD;
                vq[b] = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float ka = __ldg(K_absmax + am[b]);
                float _s;
                TQ3_K_DOT(q_reg, K_quant, kq[b], ka, lane_id, _s);
                scores[b] = _s * inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            float exp_factors[TQ4_BC];
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l_val += exp_factors[b];
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float va = __ldg(V_absmax + am[b]);
                TQ4_V_ACCUM(o_reg, V_quant, vq[b], va, exp_factors[b], lane_id);
            }
            m_val = m_new;
        }

        for (; processed < batch_count; processed++) {
            unsigned int bo = block_offset + processed;
            unsigned long long am_r = am_block_base + (unsigned long long)bo * num_kv_heads;
            unsigned long long kq_r = kq_block_base + (unsigned long long)bo * num_kv_heads * TQ3_K_BYTES_PER_HEAD;
            unsigned long long vq_r = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;

            float ka = __ldg(K_absmax + am_r);
            float s;
            TQ3_K_DOT(q_reg, K_quant, kq_r, ka, lane_id, s);
            s *= inv_sqrt_d;
            if (softcap > 0.f) s = softcap * tanhf(s / softcap);

            float m_new = fmaxf(m_val, s);
            float exp_old = __expf(m_val - m_new), e = __expf(s - m_new);
            l_val = l_val * exp_old + e;
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;

            float va = __ldg(V_absmax + am_r);
            TQ4_V_ACCUM(o_reg, V_quant, vq_r, va, e, lane_id);
            m_val = m_new;
        }

        pos += batch_count;
    }

    __shared__ float smem_m[TQ4_NUM_WARPS];
    __shared__ float smem_l[TQ4_NUM_WARPS];
    __shared__ float smem_o[TQ4_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = TQ4_NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.f) {
                float mw = smem_m[other], my_m = smem_m[warp_id], my_l = smem_l[warp_id];
                float mn = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - mn), scale_w = __expf(mw - mn);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = mn;
                #pragma unroll
                for (int i = 0; i < TQ4_VEC; i++)
                    smem_o[warp_id][bf16_vec_off + i] =
                        smem_o[warp_id][bf16_vec_off + i] * scale_me +
                        smem_o[other][bf16_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.f) ? (1.f / final_l) : 0.f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + bf16_vec_off);
        #pragma unroll
        for (int i = 0; i < TQ4_VEC_U32; i++) {
            float v0 = smem_o[0][bf16_vec_off + 2*i]     * inv_l;
            float v1 = smem_o[0][bf16_vec_off + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v0));
            unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// Split-K TQ3 decode for long sequences
// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256,1,1)
// ============================================================================

extern "C" __global__ void flash_tq3_decode_splitk(
    const flash_half_t* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
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
    const unsigned int num_seqs,
    const unsigned int q_stride,
    const float softcap,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads || seq_idx >= num_seqs) return;
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

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int bf16_vec_off = lane_id * TQ4_VEC;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC / 2; i++) unpack2_bf16_tq4(q32[i], q_reg[2*i], q_reg[2*i+1]);

    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) {
        unsigned int ch = lane_id * TQ4_VEC + i;
        q_reg[i] *= get_sign_flip(kv_head, ch);
    }
    wht_transform(q_reg, lane_id);

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + TQ4_NUM_WARPS - 1) / TQ4_NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[TQ4_VEC];
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) o_reg[i] = 0.f;

    const unsigned long long am_stride = (unsigned long long)block_size * num_kv_heads;
    const unsigned long long kq_stride = (unsigned long long)block_size * num_kv_heads * TQ3_K_BYTES_PER_HEAD;
    const unsigned int hd_half = head_dim / 2;
    const unsigned long long vq_stride = (unsigned long long)block_size * num_kv_heads * hd_half;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        unsigned long long am_block_base = (unsigned long long)physical_block * am_stride + kv_head;
        unsigned long long kq_block_base = (unsigned long long)physical_block * kq_stride
            + (unsigned long long)kv_head * TQ3_K_BYTES_PER_HEAD;
        unsigned long long vq_block_base = (unsigned long long)physical_block * vq_stride
            + (unsigned long long)kv_head * hd_half;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / TQ4_BC) * TQ4_BC;

        for (; processed < aligned; processed += TQ4_BC) {
            float scores[TQ4_BC];
            unsigned long long am[TQ4_BC], kq[TQ4_BC], vq[TQ4_BC];

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                am[b] = am_block_base + (unsigned long long)bo * num_kv_heads;
                kq[b] = kq_block_base + (unsigned long long)bo * num_kv_heads * TQ3_K_BYTES_PER_HEAD;
                vq[b] = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float ka = __ldg(K_absmax + am[b]);
                float _s;
                TQ3_K_DOT(q_reg, K_quant, kq[b], ka, lane_id, _s);
                scores[b] = _s * inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            float exp_factors[TQ4_BC];
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l_val += exp_factors[b];
            }

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                float va = __ldg(V_absmax + am[b]);
                TQ4_V_ACCUM(o_reg, V_quant, vq[b], va, exp_factors[b], lane_id);
            }
            m_val = m_new;
        }

        for (; processed < batch_count; processed++) {
            unsigned int bo = block_offset + processed;
            unsigned long long am_r = am_block_base + (unsigned long long)bo * num_kv_heads;
            unsigned long long kq_r = kq_block_base + (unsigned long long)bo * num_kv_heads * TQ3_K_BYTES_PER_HEAD;
            unsigned long long vq_r = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;

            float ka = __ldg(K_absmax + am_r);
            float s;
            TQ3_K_DOT(q_reg, K_quant, kq_r, ka, lane_id, s);
            s *= inv_sqrt_d;
            if (softcap > 0.f) s = softcap * tanhf(s / softcap);

            float m_new = fmaxf(m_val, s);
            float exp_old = __expf(m_val - m_new), e = __expf(s - m_new);
            l_val = l_val * exp_old + e;
            #pragma unroll
            for (int i = 0; i < TQ4_VEC; i++) o_reg[i] *= exp_old;

            float va = __ldg(V_absmax + am_r);
            TQ4_V_ACCUM(o_reg, V_quant, vq_r, va, e, lane_id);
            m_val = m_new;
        }

        pos += batch_count;
    }

    __shared__ float smem_m[TQ4_NUM_WARPS];
    __shared__ float smem_l[TQ4_NUM_WARPS];
    __shared__ float smem_o[TQ4_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < TQ4_VEC; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = TQ4_NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.f) {
                float mw = smem_m[other], my_m = smem_m[warp_id], my_l = smem_l[warp_id];
                float mn = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - mn), scale_w = __expf(mw - mn);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = mn;
                #pragma unroll
                for (int i = 0; i < TQ4_VEC; i++)
                    smem_o[warp_id][bf16_vec_off + i] =
                        smem_o[warp_id][bf16_vec_off + i] * scale_me +
                        smem_o[other][bf16_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    unsigned int ws_stride = head_dim + 2;
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < TQ4_VEC; i++) ws_base[bf16_vec_off + i] = smem_o[0][bf16_vec_off + i];
        if (lane_id == 0) { ws_base[head_dim] = smem_m[0]; ws_base[head_dim + 1] = smem_l[0]; }
    }
}
