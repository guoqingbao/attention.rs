/**
 * @brief TurboQuant-3bit-K aware Flash Attention v2 paged prefill, SM80+.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_prefill_tq3.cuh
 * 
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 *
 * @details
 * Same tiled prefill algorithm as flash_prefill_tq4.cuh but reads K from
 * 3-bit TurboQuant buffers (V remains 4-bit). K_quant layout uses packed
 * 3-bit groups: every 8 channels → 3 bytes (24 bits for 8×3-bit values).
 * Dequant: val = (q3 - 3) * absmax / 3.0
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

#define TQ3P_BR 32
#define TQ3P_BC 32
#define TQ3P_HDIM FLASH_HDIM

#if TQ3P_HDIM <= 256
#define TQ3P_PAD_KV 8
#else
#define TQ3P_PAD_KV 0
#endif

#define TQ3P_HDIM_PAD (TQ3P_HDIM + TQ3P_PAD_KV)
#define TQ3P_PAD_P 8
#define TQ3P_N_TILES_PER_WARP ((TQ3P_HDIM / 8) / 2)
#define TQ3P_TILE_CHUNKS (TQ3P_BR * (TQ3P_HDIM / 8))
#define TQ3P_NUM_THREADS 128
#define TQ3P_K_BYTES_PER_HEAD (TQ3P_HDIM * 3 / 8)
#define TQ3P_V_BYTES_PER_HEAD (TQ3P_HDIM / 2)

// Load a K tile from 3-bit TQ buffers, dequantize to BF16 in smem.
// 3-bit packing: 8 channels → 3 bytes (24 bits). Dequant: (q3 - 3) * absmax / 3.
// Threads cooperatively load TQ3P_BC rows × TQ3P_HDIM columns.
// Each thread processes one 8-channel chunk per iteration.
#define LOAD_TQ3_K_TILE(absmax_buf, quant_buf, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _k_bph = TQ3P_K_BYTES_PER_HEAD; \
        const unsigned int _cpr = TQ3P_HDIM / 8; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TQ3P_TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr; \
            unsigned int _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos >> _bs_shift; \
                unsigned int _bo = _pos & _bs_mask; \
                unsigned int _pb = __ldg(&(bt)[_lb]); \
                unsigned long long _am_off = (unsigned long long)_pb * cache_block_size * num_kv_heads \
                    + (unsigned long long)_bo * num_kv_heads + (kvh); \
                float _scale = (absmax_buf)[_am_off] * 0.33333333f; \
                unsigned long long _q_base = (unsigned long long)_pb * cache_block_size * num_kv_heads * _k_bph \
                    + (unsigned long long)_bo * num_kv_heads * _k_bph \
                    + (unsigned long long)(kvh) * _k_bph; \
                const unsigned char* _qp = (quant_buf) + _q_base; \
                unsigned int _group = _col / 8; \
                unsigned int _byte_base = _group * 3; \
                unsigned int _bits = (unsigned int)_qp[_byte_base] \
                    | ((unsigned int)_qp[_byte_base + 1] << 8) \
                    | ((unsigned int)_qp[_byte_base + 2] << 16); \
                flash_half_t _tmp[8]; \
                _Pragma("unroll") \
                for (int _c = 0; _c < 8; _c++) { \
                    unsigned int _q3 = (_bits >> (_c * 3)) & 0x7; \
                    _tmp[_c] = FLASH_FLOAT2HALF(((float)_q3 - 3.f) * _scale); \
                } \
                *((uint4*)&(smem)[_row * TQ3P_HDIM_PAD + _col]) = *((uint4*)_tmp); \
            } else { \
                *((uint4*)&(smem)[_row * TQ3P_HDIM_PAD + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

// V is still 4-bit, reuse the same logic as turbo4
#define LOAD_TQ3_V_TILE(absmax_buf, quant_buf, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _hd_half = TQ3P_V_BYTES_PER_HEAD; \
        const unsigned int _cpr = TQ3P_HDIM / 8; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TQ3P_TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr; \
            unsigned int _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos >> _bs_shift; \
                unsigned int _bo = _pos & _bs_mask; \
                unsigned int _pb = __ldg(&(bt)[_lb]); \
                unsigned long long _am_off = (unsigned long long)_pb * cache_block_size * num_kv_heads \
                    + (unsigned long long)_bo * num_kv_heads + (kvh); \
                float _scale = (absmax_buf)[_am_off] * 0.13333333f; \
                unsigned long long _q_base = (unsigned long long)_pb * cache_block_size * num_kv_heads * _hd_half \
                    + (unsigned long long)_bo * num_kv_heads * _hd_half \
                    + (unsigned long long)(kvh) * _hd_half; \
                const unsigned char* _qp = (quant_buf) + _q_base; \
                unsigned int _byte_off = _col / 2; \
                flash_half_t _tmp[8]; \
                const unsigned int* _qp32 = (const unsigned int*)(_qp + _byte_off); \
                unsigned int _pk4 = *_qp32; \
                _Pragma("unroll") \
                for (int _b = 0; _b < 4; _b++) { \
                    unsigned int _packed = (_pk4 >> (_b * 8)) & 0xFF; \
                    _tmp[_b * 2]     = FLASH_FLOAT2HALF(((float)(_packed & 0xF) - 7.5f) * _scale); \
                    _tmp[_b * 2 + 1] = FLASH_FLOAT2HALF(((float)(_packed >> 4) - 7.5f) * _scale); \
                } \
                *((uint4*)&(smem)[_row * TQ3P_HDIM_PAD + _col]) = *((uint4*)_tmp); \
            } else { \
                *((uint4*)&(smem)[_row * TQ3P_HDIM_PAD + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)


#if TQ3P_HDIM <= 256

#if TQ3P_HDIM > 128
#define TQ3P_USE_DYNAMIC_SMEM 1
#else
#define TQ3P_USE_DYNAMIC_SMEM 0
#endif

extern "C" __global__ void
#if TQ3P_USE_DYNAMIC_SMEM
__launch_bounds__(TQ3P_NUM_THREADS)
#endif
flash_tq3_prefill(
    const flash_half_t* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
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
    const float softcap
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

    const unsigned int q_start = q_block * TQ3P_BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + TQ3P_BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    Q += q_seq_start * q_seq_stride;
    O += q_seq_start * q_seq_stride;

#if TQ3P_USE_DYNAMIC_SMEM
    extern __shared__ __align__(16) unsigned char tq3p_smem_dyn[];
    flash_half_t* smem_Q = reinterpret_cast<flash_half_t*>(tq3p_smem_dyn);
    flash_half_t* smem_K_flat = smem_Q + TQ3P_BR * TQ3P_HDIM_PAD;
    flash_half_t* smem_V = smem_K_flat + 2 * TQ3P_BC * TQ3P_HDIM_PAD;
    flash_half_t* smem_P = smem_V + TQ3P_BC * TQ3P_HDIM_PAD;
    float* smem_ml = reinterpret_cast<float*>(smem_P + TQ3P_BR * (TQ3P_BC + TQ3P_PAD_P));
    #define SMEM_K_TQ3P(buf, idx) smem_K_flat[(buf) * TQ3P_BC * TQ3P_HDIM_PAD + (idx)]
#else
    __shared__ flash_half_t smem_Q[TQ3P_BR * TQ3P_HDIM_PAD];
    __shared__ flash_half_t smem_K_arr[2][TQ3P_BC * TQ3P_HDIM_PAD];
    __shared__ flash_half_t smem_V[TQ3P_BC * TQ3P_HDIM_PAD];
    __shared__ flash_half_t smem_P[TQ3P_BR * (TQ3P_BC + TQ3P_PAD_P)];
    __shared__ float smem_ml[TQ3P_BR * 2];
    #define SMEM_K_TQ3P(buf, idx) smem_K_arr[buf][idx]
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int qk_n_start = (warp_id >> 1) * 2;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * TQ3P_N_TILES_PER_WARP;
    const unsigned int p_stride = TQ3P_BC + TQ3P_PAD_P;

    float acc_o[TQ3P_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < TQ3P_N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f;
        acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + TQ3P_BC - 1) / TQ3P_BC;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / TQ3P_BC;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ? (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / TQ3P_BC;
    }

    // Load Q (BF16 from model output, cp.async)
    {
        const unsigned int cpr = TQ3P_HDIM / 8;
        for (unsigned int idx = tid; idx < TQ3P_TILE_CHUNKS; idx += TQ3P_NUM_THREADS) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * TQ3P_HDIM_PAD + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
                FLASH_CP_ASYNC(sa, gm);
            } else {
                *((uint4*)&smem_Q[row * TQ3P_HDIM_PAD + col]) = make_uint4(0, 0, 0, 0);
            }
        }
        FLASH_ASYNC_COMMIT();
        FLASH_ASYNC_WAIT();
    }
    __syncthreads();

    // Apply sign_flip + WHT to Q rows
    {
        const unsigned int tq_vec = TQ3P_HDIM / WARP_SIZE;
        for (unsigned int row = warp_id; row < TQ3P_BR; row += 4) {
            if (q_start + row >= q_len) continue;
            float qr[TQ3P_HDIM / WARP_SIZE];
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                qr[i] = FLASH_HALF2FLOAT(smem_Q[row * TQ3P_HDIM_PAD + ch]);
                qr[i] *= get_sign_flip(kv_head, ch);
            }
            wht_transform(qr, lane_id);
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                smem_Q[row * TQ3P_HDIM_PAD + ch] = FLASH_FLOAT2HALF(qr[i]);
            }
        }
    }
    __syncthreads();

    // Load K[0] from 3-bit TQ buffers
    if (kv_block_start < num_kv_blocks) {
        LOAD_TQ3_K_TILE(K_absmax, K_quant, block_table, (&SMEM_K_TQ3P(0, 0)), kv_block_start * TQ3P_BC, kv_len, kv_head, tid, TQ3P_NUM_THREADS);
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * TQ3P_BC;
        unsigned int kv_end = min(kv_start + TQ3P_BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // Load V from 4-bit TQ (same as turbo4)
        LOAD_TQ3_V_TILE(V_absmax, V_quant, block_table, smem_V, kv_start, kv_len, kv_head, tid, TQ3P_NUM_THREADS);
        __syncthreads();

        // QK^T MMA
        float acc_s[2][4];
        {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                acc_s[i][0] = 0.f; acc_s[i][1] = 0.f;
                acc_s[i][2] = 0.f; acc_s[i][3] = 0.f;
            }

            const unsigned int* sQ32 = (const unsigned int*)smem_Q;
            const unsigned int* sK32 = (const unsigned int*)(&SMEM_K_TQ3P(buf, 0));
            const unsigned int tq3_hdim_pad_u32 = TQ3P_HDIM_PAD / 2;

            #pragma unroll
            for (unsigned int ks = 0; ks < (TQ3P_HDIM / 16); ks++) {
                unsigned int kb_u32 = ks * 8;
                unsigned int ar0 = qk_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int aq_off = tid_in_group + kb_u32;

                unsigned int a0 = sQ32[ar0 * tq3_hdim_pad_u32 + aq_off];
                unsigned int a1 = sQ32[ar1 * tq3_hdim_pad_u32 + aq_off];
                unsigned int a2 = sQ32[ar0 * tq3_hdim_pad_u32 + aq_off + 4];
                unsigned int a3 = sQ32[ar1 * tq3_hdim_pad_u32 + aq_off + 4];

                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    unsigned int nc = (qk_n_start + nt) * 8 + group_id;
                    unsigned int bk_off = tid_in_group + kb_u32;
                    unsigned int b0 = sK32[nc * tq3_hdim_pad_u32 + bk_off];
                    unsigned int b1 = sK32[nc * tq3_hdim_pad_u32 + bk_off + 4];

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
            for (int i = 0; i < TQ3P_N_TILES_PER_WARP; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
            m_r0 = mn0;
        }
        float mn1 = fmaxf(m_r1, rmax1);
        if (mn1 != m_r1) {
            float eo1 = __expf(m_r1 - mn1); l_r1 *= eo1;
            #pragma unroll
            for (int i = 0; i < TQ3P_N_TILES_PER_WARP; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
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

        // Preload K[i+1] from 3-bit TQ
        if (kv_block + 1 < num_kv_blocks) {
            LOAD_TQ3_K_TILE(K_absmax, K_quant, block_table, (&SMEM_K_TQ3P(1 - buf, 0)),
                (kv_block + 1) * TQ3P_BC, kv_len, kv_head, tid, TQ3P_NUM_THREADS);
        }

        // PV MMA (all 4 warps)
        {
            const unsigned int* sP32 = (const unsigned int*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_V;
            const unsigned int tq3_p_stride_u32 = p_stride / 2;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int pk_off = ks * 8 + tid_in_group;
                unsigned int ar0 = pv_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int a0 = sP32[ar0 * tq3_p_stride_u32 + pk_off];
                unsigned int a1 = sP32[ar1 * tq3_p_stride_u32 + pk_off];
                unsigned int a2 = sP32[ar0 * tq3_p_stride_u32 + pk_off + 4];
                unsigned int a3 = sP32[ar1 * tq3_p_stride_u32 + pk_off + 4];
                #pragma unroll
                for (int nt = 0; nt < TQ3P_N_TILES_PER_WARP; nt++) {
                    unsigned int nc = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = ks * 16 + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * TQ3P_HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k0 * TQ3P_HDIM_PAD + nc];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * TQ3P_HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k1 * TQ3P_HDIM_PAD + nc];
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
        for (int nt = 0; nt < TQ3P_N_TILES_PER_WARP; nt++) {
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

#undef SMEM_K_TQ3P
#undef TQ3P_USE_DYNAMIC_SMEM

#else // TQ3P_HDIM == 512

#define TQ3P_BR_512 32
#define TQ3P_BC_512 32
#define TQ3P_PAD_P_512 8
#define TQ3P_N_TILES_PER_WARP_512 16
#define TQ3P_TILE_CHUNKS_512 (TQ3P_BR_512 * (512 / 8))
#define TQ3P_NUM_THREADS_512 256
#define TQ3P_K_BPH_512 (512 * 3 / 8)

#define LOAD_TQ3_K_512(absmax_buf, quant_buf, bt, dst, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _k_bph = TQ3P_K_BPH_512; \
        const unsigned int _cpr = 64; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TQ3P_TILE_CHUNKS_512; _i += (stride)) { \
            unsigned int _row = _i / _cpr; \
            unsigned int _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos >> _bs_shift; \
                unsigned int _bo = _pos & _bs_mask; \
                unsigned int _pb = __ldg(&(bt)[_lb]); \
                unsigned long long _am_off = (unsigned long long)_pb * cache_block_size * num_kv_heads \
                    + (unsigned long long)_bo * num_kv_heads + (kvh); \
                float _scale = (absmax_buf)[_am_off] * 0.33333333f; \
                unsigned long long _q_base = (unsigned long long)_pb * cache_block_size * num_kv_heads * _k_bph \
                    + (unsigned long long)_bo * num_kv_heads * _k_bph \
                    + (unsigned long long)(kvh) * _k_bph; \
                const unsigned char* _qp = (quant_buf) + _q_base; \
                unsigned int _group = _col / 8; \
                unsigned int _byte_base = _group * 3; \
                unsigned int _bits = (unsigned int)_qp[_byte_base] \
                    | ((unsigned int)_qp[_byte_base + 1] << 8) \
                    | ((unsigned int)_qp[_byte_base + 2] << 16); \
                flash_half_t _tmp[8]; \
                _Pragma("unroll") \
                for (int _c = 0; _c < 8; _c++) { \
                    unsigned int _q3 = (_bits >> (_c * 3)) & 0x7; \
                    _tmp[_c] = FLASH_FLOAT2HALF(((float)_q3 - 3.f) * _scale); \
                } \
                *((uint4*)&(dst)[_row * 512 + _col]) = *((uint4*)_tmp); \
            } else { *((uint4*)&(dst)[_row * 512 + _col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define LOAD_TQ3_V_512(absmax_buf, quant_buf, bt, dst, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _hd_half = 256; \
        const unsigned int _cpr = 64; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TQ3P_TILE_CHUNKS_512; _i += (stride)) { \
            unsigned int _row = _i / _cpr; \
            unsigned int _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos >> _bs_shift; \
                unsigned int _bo = _pos & _bs_mask; \
                unsigned int _pb = __ldg(&(bt)[_lb]); \
                unsigned long long _am_off = (unsigned long long)_pb * cache_block_size * num_kv_heads \
                    + (unsigned long long)_bo * num_kv_heads + (kvh); \
                float _scale = (absmax_buf)[_am_off] * 0.13333333f; \
                unsigned long long _q_base = (unsigned long long)_pb * cache_block_size * num_kv_heads * _hd_half \
                    + (unsigned long long)_bo * num_kv_heads * _hd_half \
                    + (unsigned long long)(kvh) * _hd_half; \
                const unsigned char* _qp = (quant_buf) + _q_base; \
                unsigned int _byte_off = _col / 2; \
                flash_half_t _tmp[8]; \
                const unsigned int* _qp32 = (const unsigned int*)(_qp + _byte_off); \
                unsigned int _pk4 = *_qp32; \
                _Pragma("unroll") \
                for (int _b = 0; _b < 4; _b++) { \
                    unsigned int _packed = (_pk4 >> (_b * 8)) & 0xFF; \
                    _tmp[_b * 2]     = FLASH_FLOAT2HALF(((float)(_packed & 0xF) - 7.5f) * _scale); \
                    _tmp[_b * 2 + 1] = FLASH_FLOAT2HALF(((float)(_packed >> 4) - 7.5f) * _scale); \
                } \
                *((uint4*)&(dst)[_row * 512 + _col]) = *((uint4*)_tmp); \
            } else { *((uint4*)&(dst)[_row * 512 + _col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

extern "C" __global__ void flash_tq3_prefill(
    const flash_half_t* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
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
    const float softcap
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

    const unsigned int q_start = q_block * TQ3P_BR_512;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + TQ3P_BR_512, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;

    Q += q_seq_start * q_seq_stride;
    O += q_seq_start * q_seq_stride;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    extern __shared__ __align__(16) unsigned char smem_dyn[];
    flash_half_t* smem_Q = reinterpret_cast<flash_half_t*>(smem_dyn);
    flash_half_t* smem_K = smem_Q + TQ3P_BR_512 * 512;
    flash_half_t* smem_V = smem_K + TQ3P_BC_512 * 512;
    flash_half_t* smem_P = smem_V + TQ3P_BC_512 * 512;
    float* smem_ml = reinterpret_cast<float*>(smem_P + TQ3P_BR_512 * (TQ3P_BC_512 + TQ3P_PAD_P_512));

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * TQ3P_N_TILES_PER_WARP_512;
    const unsigned int p_stride_512 = TQ3P_BC_512 + TQ3P_PAD_P_512;

    float acc_o[TQ3P_N_TILES_PER_WARP_512][4];
    #pragma unroll
    for (int i = 0; i < TQ3P_N_TILES_PER_WARP_512; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f;
        acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + TQ3P_BC_512 - 1) / TQ3P_BC_512;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / TQ3P_BC_512;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ? (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / TQ3P_BC_512;
    }

    // Load Q (cp.async)
    {
        const unsigned int cpr = 512 / 8;
        for (unsigned int idx = tid; idx < TQ3P_TILE_CHUNKS_512; idx += TQ3P_NUM_THREADS_512) {
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
        FLASH_ASYNC_WAIT();
    }
    __syncthreads();

    // Apply sign_flip + WHT to Q rows
    {
        const unsigned int tq_vec = 512 / WARP_SIZE;
        for (unsigned int row = warp_id; row < TQ3P_BR_512; row += 8) {
            if (q_start + row >= q_len) continue;
            float qr[512 / WARP_SIZE];
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                qr[i] = FLASH_HALF2FLOAT(smem_Q[row * 512 + ch]);
                qr[i] *= get_sign_flip(kv_head, ch);
            }
            wht_transform(qr, lane_id);
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                smem_Q[row * 512 + ch] = FLASH_FLOAT2HALF(qr[i]);
            }
        }
    }
    __syncthreads();

    if (kv_block_start < num_kv_blocks) {
        LOAD_TQ3_K_512(K_absmax, K_quant, block_table, smem_K, kv_block_start * TQ3P_BC_512, kv_len, kv_head, tid, TQ3P_NUM_THREADS_512);
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * TQ3P_BC_512;
        unsigned int kv_end = min(kv_start + TQ3P_BC_512, kv_len);
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
                for (int i = 0; i < TQ3P_N_TILES_PER_WARP_512; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
                m_r0 = mn0;
            }
            float mn1 = fmaxf(m_r1, rmax1);
            if (mn1 != m_r1) {
                float eo1 = __expf(m_r1 - mn1); l_r1 *= eo1;
                #pragma unroll
                for (int i = 0; i < TQ3P_N_TILES_PER_WARP_512; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
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
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0; l_r1 += sum1;
            if (tid_in_group == 0) {
                smem_ml[row0 * 2] = m_r0; smem_ml[row0 * 2 + 1] = l_r0;
                smem_ml[row1 * 2] = m_r1; smem_ml[row1 * 2 + 1] = l_r1;
            }
            FLASH_ASYNC_COMMIT();
        } else {
            LOAD_TQ3_V_512(V_absmax, V_quant, block_table, smem_V, kv_start, kv_len, kv_head, tid - 64, 192);
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
                for (int i = 0; i < TQ3P_N_TILES_PER_WARP_512; i++) { acc_o[i][0] *= er0; acc_o[i][1] *= er0; }
                m_r0 = cm0;
            }
            if (cm1 != m_r1) {
                float er1 = __expf(m_r1 - cm1);
                #pragma unroll
                for (int i = 0; i < TQ3P_N_TILES_PER_WARP_512; i++) { acc_o[i][2] *= er1; acc_o[i][3] *= er1; }
                m_r1 = cm1;
            }
        }

        // PV MMA (all 8 warps)
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
                for (int nt = 0; nt < TQ3P_N_TILES_PER_WARP_512; nt++) {
                    unsigned int nc = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = ks * 16 + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * 512 + nc] << 16) |
                                      (unsigned int)sV[k0 * 512 + nc];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * 512 + nc] << 16) |
                                      (unsigned int)sV[k1 * 512 + nc];
                    FLASH_MMA_K16(acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3]);
                }
            }
        }

        __syncthreads();

        if (kv_block + 1 < num_kv_blocks) {
            LOAD_TQ3_K_512(K_absmax, K_quant, block_table, smem_K, (kv_block + 1) * TQ3P_BC_512, kv_len, kv_head, tid, TQ3P_NUM_THREADS_512);
            FLASH_ASYNC_COMMIT();
            FLASH_ASYNC_WAIT();
            __syncthreads();
        }
    }

    // Final normalization and store
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
        for (int nt = 0; nt < TQ3P_N_TILES_PER_WARP_512; nt++) {
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

#undef LOAD_TQ3_K_512
#undef LOAD_TQ3_V_512
#undef TQ3P_BR_512
#undef TQ3P_BC_512
#undef TQ3P_PAD_P_512
#undef TQ3P_N_TILES_PER_WARP_512
#undef TQ3P_TILE_CHUNKS_512
#undef TQ3P_NUM_THREADS_512
#undef TQ3P_K_BPH_512

#endif // TQ3P_HDIM > 256

#undef LOAD_TQ3_K_TILE
#undef LOAD_TQ3_V_TILE
#undef TQ3P_BR
#undef TQ3P_BC
#undef TQ3P_HDIM
#undef TQ3P_PAD_KV
#undef TQ3P_HDIM_PAD
#undef TQ3P_PAD_P
#undef TQ3P_N_TILES_PER_WARP
#undef TQ3P_TILE_CHUNKS
#undef TQ3P_NUM_THREADS
#undef TQ3P_K_BYTES_PER_HEAD
#undef TQ3P_V_BYTES_PER_HEAD
