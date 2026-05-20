/**
 * @brief Native Flash Attention v2 — paged prefill with FP8 E4M3 KV cache, SM80+.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_prefill_paged_fp8.cuh
 * 
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 *
 * @details
 * Same tiled prefill algorithm as flash_prefill_paged.cuh but reads K/V
 * from FP8 E4M3 paged cache. Dequantizes FP8→BF16 in shared memory during
 * tile load using per-head scales, then runs standard BF16 matmul tiles.
 * Supports sliding window attention and softcapping.
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
// Same algorithm as flash_prefill_paged.cu but K/V cache stores FP8.
// Dequantizes FP8→BF16 in shared memory during tile load, then runs
// identical tensor-core Flash Attention pipeline.
//
// Q/O: BF16 [q_len, num_q_heads, head_dim]
// KV cache: FP8 E4M3 [num_blocks, block_size, num_kv_heads, head_dim]

#include "flash_sm_compat.cuh"
#include <cuda_fp8.h>

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif

#ifndef BR
#define BR 32
#endif
#ifndef BC
#define BC 32
#endif

#ifndef PAD_KV
#if FLASH_HDIM <= 256
#define PAD_KV 8
#else
#define PAD_KV 0
#endif
#endif

#ifndef HDIM
#define HDIM FLASH_HDIM
#endif
#ifndef HDIM_PAD
#define HDIM_PAD (HDIM + PAD_KV)
#endif
#ifndef PAD_P
#define PAD_P 8
#endif
#ifndef N_TILES_PER_WARP
#define N_TILES_PER_WARP ((HDIM / 8) / 2)
#endif
#ifndef TILE_CHUNKS
#define TILE_CHUNKS (BR * (HDIM / 8))
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

__device__ __forceinline__ flash_half_t fp8_to_bf16_s(__nv_fp8_storage_t b, float scale) {
    return FLASH_FLOAT2HALF(__half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale);
}

__device__ __forceinline__ void fp8x4_to_bf16x4(unsigned int packed, float scale,
                                                 flash_half_t &b0, flash_half_t &b1,
                                                 flash_half_t &b2, flash_half_t &b3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __half2_raw pair0 = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(packed & 0xFFFF), __NV_E4M3);
    __half2_raw pair1 = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>((packed >> 16) & 0xFFFF), __NV_E4M3);
    b0 = FLASH_FLOAT2HALF(__half2float(*reinterpret_cast<__half*>(&pair0.x)) * scale);
    b1 = FLASH_FLOAT2HALF(__half2float(*reinterpret_cast<__half*>(&pair0.y)) * scale);
    b2 = FLASH_FLOAT2HALF(__half2float(*reinterpret_cast<__half*>(&pair1.x)) * scale);
    b3 = FLASH_FLOAT2HALF(__half2float(*reinterpret_cast<__half*>(&pair1.y)) * scale);
#else
    b0 = fp8_to_bf16_s((__nv_fp8_storage_t)(packed & 0xFF), scale);
    b1 = fp8_to_bf16_s((__nv_fp8_storage_t)((packed >> 8) & 0xFF), scale);
    b2 = fp8_to_bf16_s((__nv_fp8_storage_t)((packed >> 16) & 0xFF), scale);
    b3 = fp8_to_bf16_s((__nv_fp8_storage_t)((packed >> 24) & 0xFF), scale);
#endif
}

// FP8 tile loader (raw): load FP8 bytes to smem without dequant (SM89+ path).
// smem layout: [rows][HDIM] bytes (half the BF16 footprint).
#define LOAD_KV_TILE_FP8_RAW(cache, bt, smem_u8, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos >> _bs_shift; \
                unsigned int _bo = _pos & _bs_mask; \
                unsigned int _pb = __ldg(&(bt)[_lb]); \
                const void* _base = (const void*)((const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col); \
                unsigned int _sa = __cvta_generic_to_shared(&(smem_u8)[_row * HDIM + _col]); \
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" \
                             :: "r"(_sa), "l"(_base)); \
            } else { \
                *((unsigned long long*)&(smem_u8)[_row * HDIM + _col]) = 0ULL; \
            } \
        } \
    } while(0)

// FP8 tile loader: vectorized load + dequant + store to BF16 smem
#define LOAD_KV_TILE_FP8(cache, bt, smem, kv_s, kv_l, kvh, t, stride, dq_sc) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos >> _bs_shift; \
                unsigned int _bo = _pos & _bs_mask; \
                unsigned int _pb = __ldg(&(bt)[_lb]); \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                const unsigned int* _base32 = (const unsigned int*)_base; \
                flash_half_t _v[8]; \
                fp8x4_to_bf16x4(_base32[0], (dq_sc), _v[0], _v[1], _v[2], _v[3]); \
                fp8x4_to_bf16x4(_base32[1], (dq_sc), _v[4], _v[5], _v[6], _v[7]); \
                *((uint4*)&(smem)[_row * HDIM_PAD + _col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row * HDIM_PAD + _col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#if HDIM <= 256

#if HDIM > 128
#define PREFILL_FP8_USE_DYNAMIC_SMEM 1
#else
#define PREFILL_FP8_USE_DYNAMIC_SMEM 0
#endif

extern "C" __global__ void
#if PREFILL_FP8_USE_DYNAMIC_SMEM
__launch_bounds__(NUM_THREADS)
#endif
flash_prefill_paged_fp8(
    const flash_half_t* __restrict__ Q,
    const void* __restrict__ K_cache,
    const void* __restrict__ V_cache,
    flash_half_t* __restrict__ O,
    const int* __restrict__ block_tables,
    const unsigned int block_table_stride,
    const unsigned int* __restrict__ cu_seqlens_q,
    const unsigned int* __restrict__ context_lens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal,
    const float inv_sqrt_d,
    const float softcap,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr,
    const unsigned long long fp8_cache_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_seq_start = cu_seqlens_q[seq_idx];
    const unsigned int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
    const unsigned int kv_len = context_lens[seq_idx];
    const unsigned int q_offset = kv_len > q_len ? kv_len - q_len : 0;
    const int* block_table = block_tables + seq_idx * block_table_stride;

    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
    const float k_scale = k_scale_ptr[kv_head];
    const float v_scale = v_scale_ptr[kv_head];

    Q += q_seq_start * q_seq_stride;
    O += q_seq_start * q_seq_stride;

#if PREFILL_FP8_USE_DYNAMIC_SMEM
    extern __shared__ __align__(16) unsigned char prefill_fp8_smem_dyn[];
    flash_half_t* smem_Q = reinterpret_cast<flash_half_t*>(prefill_fp8_smem_dyn);
    flash_half_t* smem_K_flat = smem_Q + BR * HDIM_PAD;
    flash_half_t* smem_V = smem_K_flat + 2 * BC * HDIM_PAD;
    flash_half_t* smem_P = smem_V + BC * HDIM_PAD;
    float* smem_ml = reinterpret_cast<float*>(smem_P + BR * (BC + PAD_P));
    #define SMEM_K_FP8(buf, idx) smem_K_flat[(buf) * BC * HDIM_PAD + (idx)]
#else
    __shared__ flash_half_t smem_Q[BR * HDIM_PAD];
    __shared__ flash_half_t smem_K_arr[2][BC * HDIM_PAD];
    __shared__ flash_half_t smem_V[BC * HDIM_PAD];
    __shared__ flash_half_t smem_P[BR * (BC + PAD_P)];
    __shared__ float smem_ml[BR * 2];
    #define SMEM_K_FP8(buf, idx) smem_K_arr[buf][idx]
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int qk_n_start = (warp_id >> 1) * 2;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;
    const unsigned int p_stride = BC + PAD_P;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f;
        acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ? (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / BC;
    }

    // Load Q (BF16 cp.async) + K[0] (FP8 dequant)
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += NUM_THREADS) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * HDIM_PAD + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
                FLASH_CP_ASYNC(sa, gm);
            } else {
                *((uint4*)&smem_Q[row * HDIM_PAD + col]) = make_uint4(0, 0, 0, 0);
            }
        }
        FLASH_ASYNC_COMMIT();
    }

    if (kv_block_start < num_kv_blocks) {
        LOAD_KV_TILE_FP8(K_cache, block_table, (&SMEM_K_FP8(0, 0)), kv_block_start * BC, kv_len, kv_head, tid, NUM_THREADS, k_scale);
    }
    FLASH_ASYNC_WAIT();
    __syncthreads();

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // V load (synchronous FP8 dequant, overlaps temporally with QK^T on warps 0-1)
        LOAD_KV_TILE_FP8(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, NUM_THREADS, v_scale);

        // QK^T: ALL 4 warps, N-split by warp_id>>1
        float acc_s[2][4];
        {
            #pragma unroll
            for (int i = 0; i < 2; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }

            const unsigned int* sQ32 = (const unsigned int*)smem_Q;
            const unsigned int* sK32 = (const unsigned int*)(&SMEM_K_FP8(buf, 0));
            const unsigned int hdim_pad_u32 = HDIM_PAD / 2;

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int kb_u32 = ks * 8;
                unsigned int ar0 = qk_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int aq_off = tid_in_group + kb_u32;
                unsigned int a0 = sQ32[ar0 * hdim_pad_u32 + aq_off];
                unsigned int a1 = sQ32[ar1 * hdim_pad_u32 + aq_off];
                unsigned int a2 = sQ32[ar0 * hdim_pad_u32 + aq_off + 4];
                unsigned int a3 = sQ32[ar1 * hdim_pad_u32 + aq_off + 4];

                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    unsigned int nc = (qk_n_start + nt) * 8 + group_id;
                    unsigned int bk_off = tid_in_group + kb_u32;
                    unsigned int b0 = sK32[nc * hdim_pad_u32 + bk_off];
                    unsigned int b1 = sK32[nc * hdim_pad_u32 + bk_off + 4];
                    FLASH_MMA_K16(acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3]);
                }
            }
        }

        unsigned int row0 = qk_warp_m + group_id, row1 = row0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
            acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
            if (softcap > 0.f) {
                acc_s[nt][0] = softcap * tanhf(acc_s[nt][0] / softcap);
                acc_s[nt][1] = softcap * tanhf(acc_s[nt][1] / softcap);
                acc_s[nt][2] = softcap * tanhf(acc_s[nt][2] / softcap);
                acc_s[nt][3] = softcap * tanhf(acc_s[nt][3] / softcap);
            }
            unsigned int c0 = (qk_n_start + nt) * 8 + tid_in_group * 2, c1 = c0 + 1;
            unsigned int qr0 = q_offset + q_start + row0, qr1 = q_offset + q_start + row1;
            if (causal) {
                if (kv_start + c0 > qr0) acc_s[nt][0] = -1e30f;
                if (kv_start + c1 > qr0) acc_s[nt][1] = -1e30f;
                if (kv_start + c0 > qr1) acc_s[nt][2] = -1e30f;
                if (kv_start + c1 > qr1) acc_s[nt][3] = -1e30f;
            }
            if (sliding_window > 0) {
                if (qr0 >= kv_start + c0 && qr0 - (kv_start + c0) >= sliding_window) acc_s[nt][0] = -1e30f;
                if (qr0 >= kv_start + c1 && qr0 - (kv_start + c1) >= sliding_window) acc_s[nt][1] = -1e30f;
                if (qr1 >= kv_start + c0 && qr1 - (kv_start + c0) >= sliding_window) acc_s[nt][2] = -1e30f;
                if (qr1 >= kv_start + c1 && qr1 - (kv_start + c1) >= sliding_window) acc_s[nt][3] = -1e30f;
            }
            if (c0 >= kv_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
            if (c1 >= kv_tile_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
            if (row0 >= q_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
            if (row1 >= q_tile_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
        }

        float rmax0 = fmaxf(fmaxf(acc_s[0][0], acc_s[0][1]), fmaxf(acc_s[1][0], acc_s[1][1]));
        float rmax1 = fmaxf(fmaxf(acc_s[0][2], acc_s[0][3]), fmaxf(acc_s[1][2], acc_s[1][3]));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        if (tid_in_group == 0) {
            smem_ml[row0 * 2 + (warp_id >> 1)] = rmax0;
            smem_ml[row1 * 2 + (warp_id >> 1)] = rmax1;
        }
        __syncthreads();
        {
            unsigned int half = warp_id >> 1;
            rmax0 = fmaxf(rmax0, smem_ml[row0 * 2 + (1 - half)]);
            rmax1 = fmaxf(rmax1, smem_ml[row1 * 2 + (1 - half)]);
        }

        float mn0 = fmaxf(m_r0, rmax0);
        if (mn0 != m_r0) {
            float eo0 = __expf(m_r0 - mn0); l_r0 *= eo0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
            m_r0 = mn0;
        }
        float mn1 = fmaxf(m_r1, rmax1);
        if (mn1 != m_r1) {
            float eo1 = __expf(m_r1 - mn1); l_r1 *= eo1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
            m_r1 = mn1;
        }

        float sum0 = 0.f, sum1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
            float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
            sum0 += p00 + p01; sum1 += p10 + p11;
            unsigned int c0 = (qk_n_start + nt) * 8 + tid_in_group * 2;
            smem_P[row0 * p_stride + c0]     = FLASH_FLOAT2HALF(p00);
            smem_P[row0 * p_stride + c0 + 1] = FLASH_FLOAT2HALF(p01);
            smem_P[row1 * p_stride + c0]     = FLASH_FLOAT2HALF(p10);
            smem_P[row1 * p_stride + c0 + 1] = FLASH_FLOAT2HALF(p11);
        }
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);

        if (tid_in_group == 0) {
            smem_ml[row0 * 2 + (warp_id >> 1)] = sum0;
            smem_ml[row1 * 2 + (warp_id >> 1)] = sum1;
        }
        __syncthreads();
        l_r0 += smem_ml[row0 * 2] + smem_ml[row0 * 2 + 1];
        l_r1 += smem_ml[row1 * 2] + smem_ml[row1 * 2 + 1];

        // Preload K[i+1] (FP8 dequant)
        if (kv_block + 1 < num_kv_blocks) {
            LOAD_KV_TILE_FP8(K_cache, block_table, (&SMEM_K_FP8(1 - buf, 0)),
                (kv_block + 1) * BC, kv_len, kv_head, tid, NUM_THREADS, k_scale);
        }

        // PV MMA (all 4 warps)
        {
            const unsigned int* sP32 = (const unsigned int*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_V;
            const unsigned int p_stride_u32 = p_stride / 2;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int pk_off = ks * 8 + tid_in_group;
                unsigned int ar0 = pv_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int a0 = sP32[ar0 * p_stride_u32 + pk_off];
                unsigned int a1 = sP32[ar1 * p_stride_u32 + pk_off];
                unsigned int a2 = sP32[ar0 * p_stride_u32 + pk_off + 4];
                unsigned int a3 = sP32[ar1 * p_stride_u32 + pk_off + 4];
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int nc = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = ks * 16 + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k0 * HDIM_PAD + nc];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k1 * HDIM_PAD + nc];
                    FLASH_MMA_K16(acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3]);
                }
            }
        }

        __syncthreads();
    }

    // Final normalization and store
    {
        unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8;
        float il0 = (l_r0 > 0.f) ? (1.f / l_r0) : 0.f;
        float il1 = (l_r1 > 0.f) ? (1.f / l_r1) : 0.f;

        flash_half_t* ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int c0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + r0, gr1 = q_start + r1;
            if (gr0 < q_len && r0 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][0] * il0));
                unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][1] * il0));
                *(unsigned int*)&ob[gr0 * q_seq_stride + c0] = lo | (hi << 16);
            }
            if (gr1 < q_len && r1 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][2] * il1));
                unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][3] * il1));
                *(unsigned int*)&ob[gr1 * q_seq_stride + c0] = lo | (hi << 16);
            }
        }
    }
}
#undef SMEM_K_FP8
#undef PREFILL_FP8_USE_DYNAMIC_SMEM

#else // HDIM == 512 FP8 variant

// HDIM=512 FP8: same dynamic smem layout as BF16 512,
// but K/V tile loads do FP8 dequant.

#define BR_512 32
#define BC_512 32
#define PAD_P_512 8
#define N_TILES_PER_WARP_512 16
#define TILE_CHUNKS_512 (BR_512 * (512 / 8))
#define NUM_THREADS_512 256

#define LOAD_KV_512_FP8(cache, bt, dst, kv_s, kv_l, kvh, t, stride, dq_sc) \
    do { \
        const unsigned int _cpr = 512 / 8; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS_512; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos >> _bs_shift; \
                unsigned int _bo = _pos & _bs_mask; \
                unsigned int _pb = __ldg(&(bt)[_lb]); \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                flash_half_t _v[8]; \
                for (int _j = 0; _j < 8; _j++) _v[_j] = fp8_to_bf16_s(_base[_j], dq_sc); \
                *((uint4*)&(dst)[_row * 512 + _col]) = *((uint4*)_v); \
            } else { *((uint4*)&(dst)[_row * 512 + _col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

extern "C" __global__ void flash_prefill_paged_fp8(
    const flash_half_t* __restrict__ Q,
    const void* __restrict__ K_cache,
    const void* __restrict__ V_cache,
    flash_half_t* __restrict__ O,
    const int* __restrict__ block_tables,
    const unsigned int block_table_stride,
    const unsigned int* __restrict__ cu_seqlens_q,
    const unsigned int* __restrict__ context_lens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal,
    const float inv_sqrt_d,
    const float softcap,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr,
    const unsigned long long fp8_cache_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_seq_start = cu_seqlens_q[seq_idx];
    const unsigned int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
    const unsigned int kv_len = context_lens[seq_idx];
    const unsigned int q_offset = kv_len > q_len ? kv_len - q_len : 0;
    const int* block_table = block_tables + seq_idx * block_table_stride;

    const unsigned int q_start = q_block * BR_512;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR_512, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
    const float k_scale = k_scale_ptr[kv_head];
    const float v_scale = v_scale_ptr[kv_head];

    Q += q_seq_start * q_seq_stride;
    O += q_seq_start * q_seq_stride;

    extern __shared__ __align__(16) unsigned char smem_dyn[];
    flash_half_t* smem_Q = reinterpret_cast<flash_half_t*>(smem_dyn);
    flash_half_t* smem_K = smem_Q + BR_512 * 512;
    flash_half_t* smem_V = smem_K + BC_512 * 512;
    flash_half_t* smem_P = smem_V + BC_512 * 512;
    float* smem_ml = reinterpret_cast<float*>(smem_P + BR_512 * (BC_512 + PAD_P_512));

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP_512;
    const unsigned int p_stride_512 = BC_512 + PAD_P_512;

    float acc_o[N_TILES_PER_WARP_512][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP_512; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f;
        acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + BC_512 - 1) / BC_512;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / BC_512;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ? (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / BC_512;
    }

    // Load Q (BF16 cp.async) + K[0] (FP8 dequant)
    {
        const unsigned int cpr = 512 / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_512; idx += NUM_THREADS_512) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * 512 + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
                FLASH_CP_ASYNC(sa, gm);
            } else {
                *((uint4*)&smem_Q[row * 512 + col]) = make_uint4(0, 0, 0, 0);
            }
        }
        FLASH_ASYNC_COMMIT();
    }
    if (kv_block_start < num_kv_blocks) {
        LOAD_KV_512_FP8(K_cache, block_table, smem_K, kv_block_start * BC_512, kv_len, kv_head, tid, NUM_THREADS_512, k_scale);
    }
    FLASH_ASYNC_WAIT();
    __syncthreads();

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC_512;
        unsigned int kv_end = min(kv_start + BC_512, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }
            const unsigned int* sQ32 = (const unsigned int*)smem_Q;
            const unsigned int* sK32 = (const unsigned int*)smem_K;

            #pragma unroll
            for (unsigned int ks = 0; ks < (512/16); ks++) {
                unsigned int kb_u32 = ks * 8;
                unsigned int ar0 = qk_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int aq_off = tid_in_group + kb_u32;
                unsigned int a0 = sQ32[ar0 * 256 + aq_off];
                unsigned int a1 = sQ32[ar1 * 256 + aq_off];
                unsigned int a2 = sQ32[ar0 * 256 + aq_off + 4];
                unsigned int a3 = sQ32[ar1 * 256 + aq_off + 4];
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int nc = nt * 8 + group_id;
                    unsigned int bk_off = tid_in_group + kb_u32;
                    unsigned int b0 = sK32[nc * 256 + bk_off];
                    unsigned int b1 = sK32[nc * 256 + bk_off + 4];
                    FLASH_MMA_K16(acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3]);
                }
            }

            unsigned int row0 = qk_warp_m + group_id, row1 = row0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
                if (softcap > 0.f) {
                    acc_s[nt][0] = softcap * tanhf(acc_s[nt][0] / softcap);
                    acc_s[nt][1] = softcap * tanhf(acc_s[nt][1] / softcap);
                    acc_s[nt][2] = softcap * tanhf(acc_s[nt][2] / softcap);
                    acc_s[nt][3] = softcap * tanhf(acc_s[nt][3] / softcap);
                }
                unsigned int c0 = nt * 8 + tid_in_group * 2, c1 = c0 + 1;
                unsigned int qr0 = q_offset + q_start + row0, qr1 = q_offset + q_start + row1;
                if (causal) {
                    if (kv_start + c0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + c1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + c0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + c1 > qr1) acc_s[nt][3] = -1e30f;
                }
                if (sliding_window > 0) {
                    if (qr0 >= kv_start + c0 && qr0 - (kv_start + c0) >= sliding_window) acc_s[nt][0] = -1e30f;
                    if (qr0 >= kv_start + c1 && qr0 - (kv_start + c1) >= sliding_window) acc_s[nt][1] = -1e30f;
                    if (qr1 >= kv_start + c0 && qr1 - (kv_start + c0) >= sliding_window) acc_s[nt][2] = -1e30f;
                    if (qr1 >= kv_start + c1 && qr1 - (kv_start + c1) >= sliding_window) acc_s[nt][3] = -1e30f;
                }
                if (c0 >= kv_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (c1 >= kv_tile_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_tile_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            float mn0 = fmaxf(m_r0, rmax0);
            if (mn0 != m_r0) {
                float eo0 = __expf(m_r0 - mn0); l_r0 *= eo0;
                #pragma unroll
                for (int i = 0; i < N_TILES_PER_WARP_512; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
                m_r0 = mn0;
            }
            float mn1 = fmaxf(m_r1, rmax1);
            if (mn1 != m_r1) {
                float eo1 = __expf(m_r1 - mn1); l_r1 *= eo1;
                #pragma unroll
                for (int i = 0; i < N_TILES_PER_WARP_512; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
                m_r1 = mn1;
            }

            float sum0 = 0.f, sum1 = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01; sum1 += p10 + p11;
                unsigned int c0 = nt * 8 + tid_in_group * 2;
                smem_P[row0 * p_stride_512 + c0]     = FLASH_FLOAT2HALF(p00);
                smem_P[row0 * p_stride_512 + c0 + 1] = FLASH_FLOAT2HALF(p01);
                smem_P[row1 * p_stride_512 + c0]     = FLASH_FLOAT2HALF(p10);
                smem_P[row1 * p_stride_512 + c0 + 1] = FLASH_FLOAT2HALF(p11);
            }
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1); sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1); sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0; l_r1 += sum1;
            if (tid_in_group == 0) {
                smem_ml[row0 * 2] = m_r0; smem_ml[row0 * 2 + 1] = l_r0;
                smem_ml[row1 * 2] = m_r1; smem_ml[row1 * 2 + 1] = l_r1;
            }
            FLASH_ASYNC_COMMIT();
        } else {
            LOAD_KV_512_FP8(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid - 64, 192, v_scale);
            FLASH_ASYNC_COMMIT();
        }

        FLASH_ASYNC_WAIT();
        __syncthreads();

        if (warp_id >= 2) {
            unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8;
            float cm0 = smem_ml[r0 * 2], cm1 = smem_ml[r1 * 2];
            if (cm0 != m_r0) {
                float er0 = __expf(m_r0 - cm0);
                #pragma unroll
                for (int i = 0; i < N_TILES_PER_WARP_512; i++) { acc_o[i][0] *= er0; acc_o[i][1] *= er0; }
                m_r0 = cm0;
            }
            if (cm1 != m_r1) {
                float er1 = __expf(m_r1 - cm1);
                #pragma unroll
                for (int i = 0; i < N_TILES_PER_WARP_512; i++) { acc_o[i][2] *= er1; acc_o[i][3] *= er1; }
                m_r1 = cm1;
            }
        }

        // PV
        {
            const unsigned int* sP32 = (const unsigned int*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_V;
            const unsigned int p_stride_u32_512 = p_stride_512 / 2;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int pk_off = ks * 8 + tid_in_group;
                unsigned int ar0 = pv_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int a0 = sP32[ar0 * p_stride_u32_512 + pk_off];
                unsigned int a1 = sP32[ar1 * p_stride_u32_512 + pk_off];
                unsigned int a2 = sP32[ar0 * p_stride_u32_512 + pk_off + 4];
                unsigned int a3 = sP32[ar1 * p_stride_u32_512 + pk_off + 4];
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++) {
                    unsigned int nc = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = ks * 16 + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * 512 + nc] << 16) | (unsigned int)sV[k0 * 512 + nc];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * 512 + nc] << 16) | (unsigned int)sV[k1 * 512 + nc];
                    FLASH_MMA_K16(acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3]);
                }
            }
        }

        __syncthreads();

        if (kv_block + 1 < num_kv_blocks) {
            LOAD_KV_512_FP8(K_cache, block_table, smem_K, (kv_block + 1) * BC_512, kv_len, kv_head, tid, NUM_THREADS_512, k_scale);
            FLASH_ASYNC_COMMIT();
            FLASH_ASYNC_WAIT();
            __syncthreads();
        }
    }

    // Final
    {
        unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8;
        float il0, il1;
        if (warp_id < 2) {
            il0 = (l_r0 > 0.f) ? (1.f / l_r0) : 0.f;
            il1 = (l_r1 > 0.f) ? (1.f / l_r1) : 0.f;
        } else {
            float lv0 = smem_ml[r0 * 2 + 1], lv1 = smem_ml[r1 * 2 + 1];
            il0 = (lv0 > 0.f) ? (1.f / lv0) : 0.f;
            il1 = (lv1 > 0.f) ? (1.f / lv1) : 0.f;
        }
        flash_half_t* ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++) {
            unsigned int c0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + r0, gr1 = q_start + r1;
            if (gr0 < q_len && r0 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][0] * il0));
                unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][1] * il0));
                *(unsigned int*)&ob[gr0 * q_seq_stride + c0] = lo | (hi << 16);
            }
            if (gr1 < q_len && r1 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][2] * il1));
                unsigned int hi = (unsigned int)FLASH_HALF_AS_USHORT(FLASH_FLOAT2HALF(acc_o[nt][3] * il1));
                *(unsigned int*)&ob[gr1 * q_seq_stride + c0] = lo | (hi << 16);
            }
        }
    }
}

#undef LOAD_KV_512_FP8

#endif // HDIM > 256
