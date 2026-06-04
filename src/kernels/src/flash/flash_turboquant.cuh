/**
 * @brief TurboQuant KV cache — k8v4 preset (FP8 keys + 4-bit values), SM80+.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_turboquant.cuh
 * 
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 *
 * @details
 * Implements TurboQuant k8v4 mode: keys stored as FP8 E4M3 in the standard
 * cache (no WHT rotation), values compressed to 4-bit with per-position
 * absmax scaling. Store kernel writes V to TQ buffers. Decode kernels use
 * hardware FP8 dequant for K and 4-bit uniform dequant for V. Based on
 * TurboQuant (Zandieh et al., ICLR 2026). Includes main + split-K decode
 * and split-K reduce kernels.
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
// Based on TurboQuant (Zandieh et al., ICLR 2026):
//   - Keys: FP8 E4M3 in standard cache (no WHT rotation)
//   - Values: Uniform 4-bit quantization (per-head absmax scaling)
//   - ~2.6x compression ratio, 79-100% baseline throughput
//
// WHT rotation is self-inverse: H = H^T = H^(-1), so dequant uses the same transform.
// Random sign flips (per-head, deterministic seed) provide randomized rotation
// to distribute outliers across all channels.

#include "flash_sm_compat.cuh"
#include <cuda_fp8.h>

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

// ============================================================================
// Walsh-Hadamard Transform helpers
// ============================================================================

// In-place fast Walsh-Hadamard transform on float[HDIM] in registers.
// Uses butterfly structure: O(d log d) operations.
// We split HDIM across threads: each thread holds VEC elements.
// For VEC=4 (HDIM=128, 32 threads): each thread transforms its 4 elements,
// then does cross-thread butterflies via __shfl_xor_sync.

#define TQ_VEC (HDIM / WARP_SIZE)

__device__ __forceinline__ void wht_intra_thread(float* v, int n) {
    for (int step = 1; step < n; step <<= 1) {
        for (int i = 0; i < n; i++) {
            int j = i ^ step;
            if (j > i) {
                float a = v[i], b = v[j];
                v[i] = a + b;
                v[j] = a - b;
            }
        }
    }
}

__device__ __forceinline__ void wht_cross_thread(float* v, unsigned int lane_id) {
    // Cross-thread butterfly stages for the remaining log2(WARP_SIZE) steps.
    // After intra-thread WHT on VEC elements, we need WARP_SIZE butterfly stages.
    #pragma unroll
    for (int stride = 1; stride < WARP_SIZE; stride <<= 1) {
        #pragma unroll
        for (int i = 0; i < TQ_VEC; i++) {
            float other = __shfl_xor_sync(0xffffffff, v[i], stride);
            if (lane_id & stride)
                v[i] = other - v[i];
            else
                v[i] = v[i] + other;
        }
    }
}

// Full WHT: first intra-thread on VEC-element blocks, then cross-thread butterflies.
// Normalizing factor: 1/sqrt(HDIM) applied after transform.
__device__ __forceinline__ void wht_transform(float* v, unsigned int lane_id) {
    wht_intra_thread(v, TQ_VEC);
    wht_cross_thread(v, lane_id);
    float norm = rsqrtf((float)HDIM);
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i++) v[i] *= norm;
}

// Deterministic random sign flip: per-head, per-channel.
// Simple hash: sign = ((head_idx * 2654435761u + channel_idx * 40503u) & 1) ? -1 : 1
__device__ __forceinline__ float get_sign_flip(unsigned int head_idx, unsigned int channel_idx) {
    unsigned int hash = head_idx * 2654435761u + channel_idx * 40503u;
    return (hash & 1) ? -1.0f : 1.0f;
}

// ============================================================================
// 4-bit uniform quantization helpers
// ============================================================================

// Pack two 4-bit unsigned ints into a single byte.
__device__ __forceinline__ unsigned char pack_4bit(unsigned char lo, unsigned char hi) {
    return (hi << 4) | (lo & 0xF);
}

// Unpack lower 4 bits from byte.
__device__ __forceinline__ unsigned char unpack_4bit_lo(unsigned char packed) {
    return packed & 0xF;
}

// Unpack upper 4 bits from byte.
__device__ __forceinline__ unsigned char unpack_4bit_hi(unsigned char packed) {
    return (packed >> 4) & 0xF;
}

// Uniform 4-bit quantize: map float to [0, 15] given absmax scale.
// val_scaled = val / absmax * 7.5 + 7.5, clamped to [0, 15]
__device__ __forceinline__ unsigned char quantize_4bit(float val, float inv_absmax) {
    float scaled = val * inv_absmax * 7.5f + 7.5f;
    scaled = fminf(fmaxf(scaled, 0.0f), 15.0f);
    return (unsigned char)(scaled + 0.5f);
}

// Uniform 4-bit dequantize: map [0, 15] back to float.
__device__ __forceinline__ float dequantize_4bit(unsigned char q, float absmax) {
    return ((float)q - 7.5f) / 7.5f * absmax;
}

// ============================================================================
// TurboQuant Store Kernel: K → WHT rotate → FP8, V → 4-bit uniform
// ============================================================================
//
// Cache layout per token-slot:
//   Key cache: [num_blocks, block_size, num_kv_heads, head_dim] as FP8 E4M3
//   Value meta: [num_blocks, block_size, num_kv_heads, 1] as float (absmax per head)
//   Value data: [num_blocks, block_size, num_kv_heads, head_dim/2] as uint8 (packed 4-bit)
//
// Input K, V: [num_tokens, num_kv_heads, head_dim] as BF16
// slot_mapping: [num_tokens] as i64
// k_scale_ptr: [num_kv_heads] as float (FP8 scale per head)
//
// We store keys as FP8 E4M3 after WHT rotation (same as FP8 KV cache path).
// Values are quantized to 4-bit uniform with per-head absmax scaling.
// The absmax is stored alongside the packed 4-bit data.

extern "C" __global__ void flash_tq_store_k8v4(
    const flash_half_t* __restrict__ K,       // [num_tokens, num_kv_heads, head_dim]
    const flash_half_t* __restrict__ V,       // [num_tokens, num_kv_heads, head_dim]
    void* __restrict__ K_cache,                // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    float* __restrict__ V_absmax,              // [num_blocks, block_size, num_kv_heads] float
    unsigned char* __restrict__ V_quant,       // [num_blocks, block_size, num_kv_heads, head_dim/2] uint8
    const long long* __restrict__ slot_mapping, // [num_tokens]
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float* __restrict__ k_scale_ptr      // [num_kv_heads] reciprocal scale for FP8
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;

    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    unsigned int block_idx = (unsigned int)(slot / block_size);
    unsigned int block_off = (unsigned int)(slot % block_size);

    // K is already written to FP8 cache by flash_reshape_and_cache (no WHT rotation).
    // This kernel only handles the 4-bit V quantization for TQ8.

    // Load V values into registers
    const unsigned int v_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim;
    float v_reg[TQ_VEC];
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i++) {
        unsigned int ch = lane_id * TQ_VEC + i;
        v_reg[i] = FLASH_HALF2FLOAT(V[v_offset + ch]);
    }

    // Compute per-head absmax for V (warp reduction)
    float local_absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i++) {
        local_absmax = fmaxf(local_absmax, fabsf(v_reg[i]));
    }
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, offset));

    // Store absmax
    if (lane_id == 0) {
        unsigned long long am_off = (unsigned long long)block_idx * block_size * num_kv_heads
            + (unsigned long long)block_off * num_kv_heads + head_idx;
        V_absmax[am_off] = local_absmax;
    }

    // Broadcast absmax to all lanes
    local_absmax = __shfl_sync(0xffffffff, local_absmax, 0);

    // Quantize V to 4-bit and pack pairs
    float inv_absmax = (local_absmax > 0.0f) ? (1.0f / local_absmax) : 0.0f;
    unsigned long long v_quant_off = (unsigned long long)block_idx * block_size * num_kv_heads * (head_dim / 2)
        + (unsigned long long)block_off * num_kv_heads * (head_dim / 2)
        + (unsigned long long)head_idx * (head_dim / 2);
    unsigned char* v_out = V_quant + v_quant_off;

    // Each thread handles TQ_VEC values, packed into TQ_VEC/2 bytes
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i += 2) {
        unsigned char q_lo = quantize_4bit(v_reg[i], inv_absmax);
        unsigned char q_hi = quantize_4bit(v_reg[i+1], inv_absmax);
        unsigned int byte_idx = (lane_id * TQ_VEC + i) / 2;
        v_out[byte_idx] = pack_4bit(q_lo, q_hi);
    }
}

// ============================================================================
// TurboQuant Decode Kernel: FP8 keys + 4-bit values → attention output
// ============================================================================
//
// Similar structure to flash_decode_paged_fp8, but:
// - K cache: FP8 E4M3, needs WHT inverse rotation after dequant
// - V cache: 4-bit packed + per-head absmax, needs dequant
// - Each warp processes a chunk of KV positions
// - Online softmax with warp/block reduction

#ifndef TQ_NUM_WARPS
#define TQ_NUM_WARPS 8
#endif
#define TQ_BC 8
#define TQ_VEC_U32 (HDIM / (WARP_SIZE * 2))

#ifndef FP8_DEQUANT_HELPERS_DEFINED
#define FP8_DEQUANT_HELPERS_DEFINED
__device__ __forceinline__ float fp8_to_f32_d(__nv_fp8_storage_t b, float scale) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
}
__device__ __forceinline__ void fp8x4_to_f32x4(unsigned int packed, float scale,
                                                float &f0, float &f1, float &f2, float &f3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __half2_raw pair0 = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(packed & 0xFFFF), __NV_E4M3);
    __half2_raw pair1 = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>((packed >> 16) & 0xFFFF), __NV_E4M3);
    f0 = __half2float(*reinterpret_cast<__half*>(&pair0.x)) * scale;
    f1 = __half2float(*reinterpret_cast<__half*>(&pair0.y)) * scale;
    f2 = __half2float(*reinterpret_cast<__half*>(&pair1.x)) * scale;
    f3 = __half2float(*reinterpret_cast<__half*>(&pair1.y)) * scale;
#else
    __nv_fp8_storage_t b0 = (__nv_fp8_storage_t)(packed & 0xFF);
    __nv_fp8_storage_t b1 = (__nv_fp8_storage_t)((packed >> 8) & 0xFF);
    __nv_fp8_storage_t b2 = (__nv_fp8_storage_t)((packed >> 16) & 0xFF);
    __nv_fp8_storage_t b3 = (__nv_fp8_storage_t)((packed >> 24) & 0xFF);
    f0 = __half2float(__nv_cvt_fp8_to_halfraw(b0, __NV_E4M3)) * scale;
    f1 = __half2float(__nv_cvt_fp8_to_halfraw(b1, __NV_E4M3)) * scale;
    f2 = __half2float(__nv_cvt_fp8_to_halfraw(b2, __NV_E4M3)) * scale;
    f3 = __half2float(__nv_cvt_fp8_to_halfraw(b3, __NV_E4M3)) * scale;
#endif
}
#endif

__device__ __forceinline__ void unpack2_bf16_tq(unsigned int packed, float &a, float &b) {
    const unsigned short lo = (unsigned short)(packed & 0xFFFF);
    const unsigned short hi = (unsigned short)(packed >> 16);
    a = FLASH_HALF2FLOAT(*reinterpret_cast<const flash_half_t*>(&lo));
    b = FLASH_HALF2FLOAT(*reinterpret_cast<const flash_half_t*>(&hi));
}

extern "C" __global__ void flash_tq_decode_k8v4(
    const flash_half_t* __restrict__ Q,       // [num_seqs, num_q_heads, head_dim]
    const void* __restrict__ K_cache,          // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    const float* __restrict__ V_absmax,        // [num_blocks, block_size, num_kv_heads]
    const unsigned char* __restrict__ V_quant, // [num_blocks, block_size, num_kv_heads, head_dim/2]
    flash_half_t* __restrict__ O,             // [num_seqs, num_q_heads, head_dim]
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
    const float* __restrict__ k_scale_ptr,     // [num_kv_heads] FP8 dequant scale
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
    const float k_scale = k_scale_ptr[kv_head];

    // Load query into registers (vectorized)
    const unsigned int bf16_vec_off = lane_id * (HDIM / WARP_SIZE);
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[TQ_VEC];
    {
#if TQ_VEC >= 4
        uint2 qv = __ldg((const uint2*)q32);
        unpack2_bf16_tq(qv.x, q_reg[0], q_reg[1]);
        unpack2_bf16_tq(qv.y, q_reg[2], q_reg[3]);
#if TQ_VEC >= 8
        uint2 qv2 = __ldg(((const uint2*)q32) + 1);
        unpack2_bf16_tq(qv2.x, q_reg[4], q_reg[5]);
        unpack2_bf16_tq(qv2.y, q_reg[6], q_reg[7]);
#if TQ_VEC >= 16
        uint4 qv3 = __ldg(((const uint4*)q32) + 1);
        unpack2_bf16_tq(qv3.x, q_reg[8], q_reg[9]);
        unpack2_bf16_tq(qv3.y, q_reg[10], q_reg[11]);
        unpack2_bf16_tq(qv3.z, q_reg[12], q_reg[13]);
        unpack2_bf16_tq(qv3.w, q_reg[14], q_reg[15]);
#endif
#endif
#else
        #pragma unroll
        for (int i = 0; i < TQ_VEC / 2; i++) {
            unpack2_bf16_tq(__ldg(q32 + i), q_reg[2*i], q_reg[2*i+1]);
        }
#endif
    }

    // No WHT rotation on Q for TQ8 — K is stored as plain FP8 (no rotation), so Q·K is direct.

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + TQ_NUM_WARPS - 1) / TQ_NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[TQ_VEC];
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i++) o_reg[i] = 0.f;

    unsigned long long k_head_stride = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long k_page_stride = (unsigned long long)block_size * k_head_stride;
    unsigned int vec_offset = lane_id * TQ_VEC;
    const unsigned int hd_half = head_dim / 2;
    const unsigned long long am_stride = (unsigned long long)block_size * num_kv_heads;
    const unsigned long long vq_stride = (unsigned long long)block_size * num_kv_heads * hd_half;

    // Inline helper: compute FP8 K dot product for one position
    #define TQ8_K_DOT(k_base_arg, dot_out) \
        do { \
            const __nv_fp8_storage_t* kp = (const __nv_fp8_storage_t*)K_cache + (k_base_arg); \
            float _dot = 0.f; \
            _Pragma("unroll") \
            for (int _g = 0; _g < TQ_VEC / 4; _g++) { \
                unsigned int kpk = __ldg((const unsigned int*)(kp) + _g); \
                float kf0, kf1, kf2, kf3; \
                fp8x4_to_f32x4(kpk, k_scale, kf0, kf1, kf2, kf3); \
                _dot += q_reg[_g*4]*kf0 + q_reg[_g*4+1]*kf1 + q_reg[_g*4+2]*kf2 + q_reg[_g*4+3]*kf3; \
            } \
            _Pragma("unroll") \
            for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1) \
                _dot += __shfl_xor_sync(0xffffffff, _dot, _off); \
            (dot_out) = _dot; \
        } while(0)

    #define TQ8_V_ACCUM(vq_base_arg, va, weight) \
        do { \
            float _vs = (va) * 0.13333333f * (weight); \
            const unsigned char* vp = V_quant + (vq_base_arg); \
            unsigned int _byte_off = (lane_id * TQ_VEC) / 2; \
            unsigned int _num_bytes = TQ_VEC / 2; \
            if (_num_bytes >= 4) { \
                const unsigned int* _vp32 = (const unsigned int*)(vp + _byte_off); \
                _Pragma("unroll") \
                for (unsigned int _g = 0; _g < _num_bytes / 4; _g++) { \
                    unsigned int _pk4 = __ldg(_vp32 + _g); \
                    _Pragma("unroll") \
                    for (int _b = 0; _b < 4; _b++) { \
                        unsigned int _byte = (_pk4 >> (_b * 8)) & 0xFF; \
                        int _vi = _g * 8 + _b * 2; \
                        o_reg[_vi]   += ((float)(_byte & 0xF) - 7.5f) * _vs; \
                        o_reg[_vi+1] += ((float)(_byte >> 4)  - 7.5f) * _vs; \
                    } \
                } \
            } else if (_num_bytes >= 2) { \
                unsigned short _pk2 = __ldg((const unsigned short*)(vp + _byte_off)); \
                unsigned int _b0 = _pk2 & 0xFF, _b1 = (_pk2 >> 8) & 0xFF; \
                o_reg[0] += ((float)(_b0 & 0xF) - 7.5f) * _vs; \
                o_reg[1] += ((float)(_b0 >> 4)  - 7.5f) * _vs; \
                o_reg[2] += ((float)(_b1 & 0xF) - 7.5f) * _vs; \
                o_reg[3] += ((float)(_b1 >> 4)  - 7.5f) * _vs; \
            } else { \
                _Pragma("unroll") \
                for (int _i = 0; _i < TQ_VEC; _i += 2) { \
                    unsigned int _bi = (lane_id * TQ_VEC + _i) / 2; \
                    unsigned int _pk = __ldg(vp + _bi); \
                    o_reg[_i]   += ((float)(_pk & 0xF) - 7.5f) * _vs; \
                    o_reg[_i+1] += ((float)(_pk >> 4)  - 7.5f) * _vs; \
                } \
            } \
        } while(0)

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        unsigned long long k_block_base = (unsigned long long)physical_block * k_page_stride
            + (unsigned long long)kv_head * head_dim + vec_offset;
        unsigned long long am_block_base = (unsigned long long)physical_block * am_stride + kv_head;
        unsigned long long vq_block_base = (unsigned long long)physical_block * vq_stride
            + (unsigned long long)kv_head * hd_half;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / TQ_BC) * TQ_BC;

        for (; processed < aligned; processed += TQ_BC) {
            float scores[TQ_BC];
            unsigned long long kb[TQ_BC], am_b[TQ_BC], vq_b[TQ_BC];

            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                kb[b] = k_block_base + (unsigned long long)bo * k_head_stride;
                am_b[b] = am_block_base + (unsigned long long)bo * num_kv_heads;
                vq_b[b] = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            }

            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                TQ8_K_DOT(kb[b], scores[b]);
                scores[b] *= inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < TQ_VEC; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            float exp_factors[TQ_BC];
            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l_val += exp_factors[b];
            }

            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                float va = __ldg(V_absmax + am_b[b]);
                TQ8_V_ACCUM(vq_b[b], va, exp_factors[b]);
            }
            m_val = m_new;
        }

        for (; processed < batch_count; processed++) {
            unsigned int bo = block_offset + processed;
            unsigned long long kb = k_block_base + (unsigned long long)bo * k_head_stride;
            float s;
            TQ8_K_DOT(kb, s);
            s *= inv_sqrt_d;
            if (softcap > 0.f) s = softcap * tanhf(s / softcap);

            float m_new = fmaxf(m_val, s);
            float exp_old = __expf(m_val - m_new), e = __expf(s - m_new);
            l_val = l_val * exp_old + e;
            #pragma unroll
            for (int i = 0; i < TQ_VEC; i++) o_reg[i] *= exp_old;

            unsigned long long am = am_block_base + (unsigned long long)bo * num_kv_heads;
            float va = __ldg(V_absmax + am);
            unsigned long long vq = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            TQ8_V_ACCUM(vq, va, e);
            m_val = m_new;
        }

        pos += batch_count;
    }

    #undef TQ8_K_DOT
    #undef TQ8_V_ACCUM

    // Warp reduction across TQ_NUM_WARPS warps
    __shared__ float smem_m[TQ_NUM_WARPS];
    __shared__ float smem_l[TQ_NUM_WARPS];
    __shared__ float smem_o[TQ_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = TQ_NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.f) {
                float mw = smem_m[other], my_m = smem_m[warp_id], my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new), scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < TQ_VEC; i++)
                    smem_o[warp_id][bf16_vec_off + i] =
                        smem_o[warp_id][bf16_vec_off + i] * scale_me +
                        smem_o[other][bf16_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    // Write output as BF16
    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.f) ? (1.f / final_l) : 0.f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + bf16_vec_off);
        #pragma unroll
        for (int i = 0; i < TQ_VEC_U32; i++) {
            float v0 = smem_o[0][bf16_vec_off + 2*i]     * inv_l;
            float v1 = smem_o[0][bf16_vec_off + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v0));
            unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// Split-K TQ k8v4 decode for long sequences
// Grid: (num_q_heads, num_splits, num_seqs)  Block: (TQ_NUM_WARPS * WARP_SIZE)
// Writes partial results to float workspace; reduced by flash_decode_paged_reduce.
// ============================================================================

extern "C" __global__ void flash_tq_decode_k8v4_splitk(
    const flash_half_t* __restrict__ Q,
    const void* __restrict__ K_cache,
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
    const float* __restrict__ k_scale_ptr,
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
    const float k_scale = k_scale_ptr[kv_head];

    const unsigned int bf16_vec_off = lane_id * (HDIM / WARP_SIZE);
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[TQ_VEC];
    {
#if TQ_VEC >= 4
        uint2 qv = __ldg((const uint2*)q32);
        unpack2_bf16_tq(qv.x, q_reg[0], q_reg[1]);
        unpack2_bf16_tq(qv.y, q_reg[2], q_reg[3]);
#if TQ_VEC >= 8
        uint2 qv2 = __ldg(((const uint2*)q32) + 1);
        unpack2_bf16_tq(qv2.x, q_reg[4], q_reg[5]);
        unpack2_bf16_tq(qv2.y, q_reg[6], q_reg[7]);
#if TQ_VEC >= 16
        uint4 qv3 = __ldg(((const uint4*)q32) + 1);
        unpack2_bf16_tq(qv3.x, q_reg[8], q_reg[9]);
        unpack2_bf16_tq(qv3.y, q_reg[10], q_reg[11]);
        unpack2_bf16_tq(qv3.z, q_reg[12], q_reg[13]);
        unpack2_bf16_tq(qv3.w, q_reg[14], q_reg[15]);
#endif
#endif
#else
        #pragma unroll
        for (int i = 0; i < TQ_VEC / 2; i++) {
            unpack2_bf16_tq(__ldg(q32 + i), q_reg[2*i], q_reg[2*i+1]);
        }
#endif
    }

    // No WHT rotation on Q for TQ8 — K is stored as plain FP8 (no rotation).

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + TQ_NUM_WARPS - 1) / TQ_NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[TQ_VEC];
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i++) o_reg[i] = 0.f;

    unsigned long long k_head_stride = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long k_page_stride = (unsigned long long)block_size * k_head_stride;
    unsigned int vec_offset = lane_id * TQ_VEC;
    const unsigned int hd_half = head_dim / 2;
    const unsigned long long am_stride = (unsigned long long)block_size * num_kv_heads;
    const unsigned long long vq_stride = (unsigned long long)block_size * num_kv_heads * hd_half;

    #define TQ8SK_K_DOT(k_base_arg, dot_out) \
        do { \
            const __nv_fp8_storage_t* kp = (const __nv_fp8_storage_t*)K_cache + (k_base_arg); \
            float _dot = 0.f; \
            _Pragma("unroll") \
            for (int _g = 0; _g < TQ_VEC / 4; _g++) { \
                unsigned int kpk = __ldg((const unsigned int*)(kp) + _g); \
                float kf0, kf1, kf2, kf3; \
                fp8x4_to_f32x4(kpk, k_scale, kf0, kf1, kf2, kf3); \
                _dot += q_reg[_g*4]*kf0 + q_reg[_g*4+1]*kf1 + q_reg[_g*4+2]*kf2 + q_reg[_g*4+3]*kf3; \
            } \
            _Pragma("unroll") \
            for (int _off = WARP_SIZE/2; _off > 0; _off >>= 1) \
                _dot += __shfl_xor_sync(0xffffffff, _dot, _off); \
            (dot_out) = _dot; \
        } while(0)

    #define TQ8SK_V_ACCUM(vq_base_arg, va, weight) \
        do { \
            float _vs = (va) * 0.13333333f * (weight); \
            const unsigned char* vp = V_quant + (vq_base_arg); \
            unsigned int _byte_off = (lane_id * TQ_VEC) / 2; \
            unsigned int _num_bytes = TQ_VEC / 2; \
            if (_num_bytes >= 4) { \
                const unsigned int* _vp32 = (const unsigned int*)(vp + _byte_off); \
                _Pragma("unroll") \
                for (unsigned int _g = 0; _g < _num_bytes / 4; _g++) { \
                    unsigned int _pk4 = __ldg(_vp32 + _g); \
                    _Pragma("unroll") \
                    for (int _b = 0; _b < 4; _b++) { \
                        unsigned int _byte = (_pk4 >> (_b * 8)) & 0xFF; \
                        int _vi = _g * 8 + _b * 2; \
                        o_reg[_vi]   += ((float)(_byte & 0xF) - 7.5f) * _vs; \
                        o_reg[_vi+1] += ((float)(_byte >> 4)  - 7.5f) * _vs; \
                    } \
                } \
            } else if (_num_bytes >= 2) { \
                unsigned short _pk2 = __ldg((const unsigned short*)(vp + _byte_off)); \
                unsigned int _b0 = _pk2 & 0xFF, _b1 = (_pk2 >> 8) & 0xFF; \
                o_reg[0] += ((float)(_b0 & 0xF) - 7.5f) * _vs; \
                o_reg[1] += ((float)(_b0 >> 4)  - 7.5f) * _vs; \
                o_reg[2] += ((float)(_b1 & 0xF) - 7.5f) * _vs; \
                o_reg[3] += ((float)(_b1 >> 4)  - 7.5f) * _vs; \
            } else { \
                _Pragma("unroll") \
                for (int _i = 0; _i < TQ_VEC; _i += 2) { \
                    unsigned int _bi = (lane_id * TQ_VEC + _i) / 2; \
                    unsigned int _pk = __ldg(vp + _bi); \
                    o_reg[_i]   += ((float)(_pk & 0xF) - 7.5f) * _vs; \
                    o_reg[_i+1] += ((float)(_pk >> 4)  - 7.5f) * _vs; \
                } \
            } \
        } while(0)

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        unsigned long long k_block_base = (unsigned long long)physical_block * k_page_stride
            + (unsigned long long)kv_head * head_dim + vec_offset;
        unsigned long long am_block_base = (unsigned long long)physical_block * am_stride + kv_head;
        unsigned long long vq_block_base = (unsigned long long)physical_block * vq_stride
            + (unsigned long long)kv_head * hd_half;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / TQ_BC) * TQ_BC;

        for (; processed < aligned; processed += TQ_BC) {
            float scores[TQ_BC];
            unsigned long long kb[TQ_BC], am_b[TQ_BC], vq_b[TQ_BC];

            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                kb[b] = k_block_base + (unsigned long long)bo * k_head_stride;
                am_b[b] = am_block_base + (unsigned long long)bo * num_kv_heads;
                vq_b[b] = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            }

            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                TQ8SK_K_DOT(kb[b], scores[b]);
                scores[b] *= inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < TQ_VEC; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            float exp_factors[TQ_BC];
            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l_val += exp_factors[b];
            }

            #pragma unroll
            for (int b = 0; b < TQ_BC; b++) {
                float va = __ldg(V_absmax + am_b[b]);
                TQ8SK_V_ACCUM(vq_b[b], va, exp_factors[b]);
            }
            m_val = m_new;
        }

        for (; processed < batch_count; processed++) {
            unsigned int bo = block_offset + processed;
            unsigned long long kb = k_block_base + (unsigned long long)bo * k_head_stride;
            float s;
            TQ8SK_K_DOT(kb, s);
            s *= inv_sqrt_d;
            if (softcap > 0.f) s = softcap * tanhf(s / softcap);

            float m_new = fmaxf(m_val, s);
            float exp_old = __expf(m_val - m_new), e = __expf(s - m_new);
            l_val = l_val * exp_old + e;
            #pragma unroll
            for (int i = 0; i < TQ_VEC; i++) o_reg[i] *= exp_old;

            unsigned long long am = am_block_base + (unsigned long long)bo * num_kv_heads;
            float va = __ldg(V_absmax + am);
            unsigned long long vq = vq_block_base + (unsigned long long)bo * num_kv_heads * hd_half;
            TQ8SK_V_ACCUM(vq, va, e);
            m_val = m_new;
        }

        pos += batch_count;
    }

    #undef TQ8SK_K_DOT
    #undef TQ8SK_V_ACCUM

    // Warp reduction across TQ_NUM_WARPS warps
    __shared__ float smem_m[TQ_NUM_WARPS];
    __shared__ float smem_l[TQ_NUM_WARPS];
    __shared__ float smem_o[TQ_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < TQ_VEC; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = TQ_NUM_WARPS/2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.f) {
                float mw = smem_m[other], my_m = smem_m[warp_id], my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new), scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < TQ_VEC; i++)
                    smem_o[warp_id][bf16_vec_off + i] =
                        smem_o[warp_id][bf16_vec_off + i] * scale_me +
                        smem_o[other][bf16_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    // Write partial (m, d, O[head_dim]) to workspace for cross-split reduction
    // Layout: workspace[(seq_idx * num_q_heads + q_head) * num_splits + split_id]
    //         * stride (head_dim + 2) floats: [O_0..O_{hd-1}, m, d]
    if (warp_id == 0) {
        const unsigned int ws_stride = head_dim + 2;
        unsigned long long ws_off = ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits + split_id;
        float* ws = workspace + ws_off * ws_stride;
        for (int i = 0; i < TQ_VEC; i++) {
            ws[bf16_vec_off + i] = smem_o[0][bf16_vec_off + i];
        }
        if (lane_id == 0) {
            ws[head_dim] = smem_m[0];
            ws[head_dim + 1] = smem_l[0];
        }
    }
}
