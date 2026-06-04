// MoE GEMM/GEMV kernel for GGUF quantized weights on Metal.
//
// Supports Q8_0, Q2_K, Q4_K, Q5_K, Q6_K weight formats.
//
// Two kernel variants:
//   1. GEMV: One simdgroup per output element — optimal for decode (small M)
//   2. GEMM: Tiled with threadgroup dequantization — optimal for prefill (larger M)
//
// Weight layout: [num_experts, N, K/qk * block_size_bytes] (packed GGUF)
// Input layout:  [num_input_tokens, K] (half or bfloat16)
// Output layout: [size_m, N] (float)
//
// Copyright (c) 2025, Guoqing Bao. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "metal_dtype.metal"
#include "gguf_utils.metal"
#include <metal_stdlib>

using namespace metal;

#define MOE_WARP_SIZE 32

// ============================================================
// GEMV Kernel — one simdgroup per (m, n) output element
//
// Each simdgroup streams over the K dimension, dequantizing
// weight blocks on-the-fly and computing the dot product with
// the corresponding input row.
//
// Grid:  (ceil(N/4), size_m, 1)
// Block: (32, 4, 1) — 4 warps per threadgroup
// ============================================================

template <typename T>
[[kernel]] void moe_gguf_gemv_kernel(
    device const T            *input            [[ buffer(0) ]],   // [num_input_tokens, K]
    device const uint8_t      *weights          [[ buffer(1) ]],   // packed GGUF [num_experts, N, K/qk * block_bytes]
    device const int          *sorted_token_ids [[ buffer(2) ]],
    device const int          *expert_ids       [[ buffer(3) ]],
    device const float        *topk_weights     [[ buffer(4) ]],
    device       float        *output           [[ buffer(5) ]],   // [size_m, N] float
    constant     int          &num_experts      [[ buffer(6) ]],
    constant     int          &topk             [[ buffer(7) ]],
    constant     int          &size_m           [[ buffer(8) ]],
    constant     int          &size_n           [[ buffer(9) ]],
    constant     int          &size_k           [[ buffer(10) ]],
    constant     int          &has_topk_weights [[ buffer(11) ]],
    constant     int          &gguf_type        [[ buffer(12) ]],   // 0=Q8_0, 1=Q4K, 2=Q2K, 3=Q3K, 4=Q5K, 5=Q6K
    constant     int          &block_size_bytes [[ buffer(13) ]],
    constant     int          &qk               [[ buffer(14) ]],   // 32 for Q8_0, 256 for K-quants
    uint3 gid             [[ threadgroup_position_in_grid ]],
    uint  simd_lane_id    [[ thread_index_in_simdgroup ]],
    uint  simd_group_id   [[ simdgroup_index_in_threadgroup ]]
) {
    const int n_out = int(gid.x) * 4 + int(simd_group_id);
    const int m_out = int(gid.y);

    if (n_out >= size_n || m_out >= size_m) return;

    const int token_id = sorted_token_ids[m_out];
    const int expert_id = expert_ids[m_out];
    const int input_token = has_topk_weights ? token_id : (token_id / topk);

    if (expert_id < 0 || expert_id >= num_experts) {
        if (simd_lane_id == 0) {
            output[m_out * size_n + n_out] = 0.0f;
        }
        return;
    }

    // Input pointer for this token
    device const T *x_ptr = input + long(input_token) * long(size_k);

    // Weight pointer for this expert's row n_out
    const long blocks_per_row = long(size_k) / long(qk);
    const long expert_stride = long(size_n) * blocks_per_row * long(block_size_bytes);
    device const uint8_t *w_expert = weights + long(expert_id) * expert_stride;
    device const uint8_t *w_row = w_expert + long(n_out) * blocks_per_row * long(block_size_bytes);

    float sum_f = 0.0f;

    // Iterate over K in chunks of qk, with each lane handling elements stride-apart
    for (int k_block = 0; k_block < int(blocks_per_row); ++k_block) {
        device const uint8_t *block_ptr = w_row + long(k_block) * long(block_size_bytes);
        int k_base = k_block * qk;

        // Dequantize and dot-product within this block
        // Each lane handles elements at stride MOE_WARP_SIZE
        switch (gguf_type) {
            case 0: { // Q8_0 — 32 elements per block
                device const block_q8_0 *blk = reinterpret_cast<device const block_q8_0 *>(block_ptr);
                float d_val = float(blk->d);
                for (int i = int(simd_lane_id); i < QK8_0; i += MOE_WARP_SIZE) {
                    int k = k_base + i;
                    if (k < size_k) {
                        sum_f += float(x_ptr[k]) * d_val * float(blk->qs[i]);
                    }
                }
                break;
            }
            case 1: { // Q4_K — 256 elements per block
                device const block_q4_K *blk = reinterpret_cast<device const block_q4_K *>(block_ptr);
                float dall = float(blk->dm[0]);
                float dmin = float(blk->dm[1]);

                for (int i = int(simd_lane_id); i < QK_K; i += MOE_WARP_SIZE) {
                    int k = k_base + i;
                    if (k < size_k) {
                        // Q4_K: 4 groups of 64, each with low/high nibble halves
                        // qs[128] stores pairs: qs[32*il + j] has low nibble for
                        // element 64*il+j and high nibble for element 64*il+j+32
                        int il = i / 64;          // 64-element group (0..3)
                        int pos = i % 64;         // position within group

                        // Scale: 2 scales per 64-element group
                        int scale_idx = 2 * il + (pos >= 32 ? 1 : 0);
                        uint8_t sc_v, m_v;
                        if (scale_idx < 4) {
                            sc_v = blk->scales[scale_idx] & 63;
                            m_v = blk->scales[scale_idx + 4] & 63;
                        } else {
                            sc_v = (blk->scales[scale_idx + 4] & 0xF) | ((blk->scales[scale_idx - 4] >> 6) << 4);
                            m_v = (blk->scales[scale_idx + 4] >> 4) | ((blk->scales[scale_idx] >> 6) << 4);
                        }

                        int nibble;
                        if (pos < 32) {
                            nibble = int(blk->qs[32 * il + pos]) & 0xF;
                        } else {
                            nibble = int(blk->qs[32 * il + (pos - 32)]) >> 4;
                        }

                        float w_val = dall * float(sc_v) * float(nibble) - dmin * float(m_v);
                        sum_f += float(x_ptr[k]) * w_val;
                    }
                }
                break;
            }
            case 2: { // Q2_K — 256 elements per block
                device const block_q2_K *blk = reinterpret_cast<device const block_q2_K *>(block_ptr);
                float dall = float(blk->dm[0]);
                float dmin = float(blk->dm[1]);

                for (int i = int(simd_lane_id); i < QK_K; i += MOE_WARP_SIZE) {
                    int k = k_base + i;
                    if (k < size_k) {
                        // Q2_K: qs[64] with interleaved packing
                        // qs[32*n + l] contains 4 2-bit values for elements
                        // at positions 128*n + l + {0, 32, 64, 96}
                        int n2 = i / 128;
                        int wh = i % 128;
                        int qs_byte = 32 * n2 + (wh % 32);
                        int qs_shift = (wh / 32) * 2;
                        uint8_t q = (blk->qs[qs_byte] >> qs_shift) & 3;

                        int group = i / 16;
                        uint8_t sc = blk->scales[group];
                        float w_val = dall * float(sc & 0xF) * float(q) - dmin * float(sc >> 4);
                        sum_f += float(x_ptr[k]) * w_val;
                    }
                }
                break;
            }
            case 3: { // Q3_K — 256 elements per block
                device const block_q3_K *blk = reinterpret_cast<device const block_q3_K *>(block_ptr);
                float d_all = float(blk->d);

                for (int i = int(simd_lane_id); i < QK_K; i += MOE_WARP_SIZE) {
                    int k = k_base + i;
                    if (k < size_k) {
                        // Q3_K packing: 256 values in 2 halves of 128.
                        // Each half has 4 sub-blocks of 32.
                        // qs[64] stores 2-bit quants in interleaved order:
                        //   byte qs[32*n + l] contains 4 quants at shifts 0,2,4,6
                        //   for elements at positions 128*n + {0,32,64,96} + l
                        int n = i / 128;          // half (0 or 1)
                        int j = (i % 128) / 32;   // sub-block within half (0-3)
                        int l = i % 32;            // element within sub-block

                        // Scale: 16 scales for 16 groups of 16 elements
                        int is = i / 16;

                        int sc_val;
                        if (is < 4) {
                            sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 8] >> 0) & 3) << 4)) - 32;
                        } else if (is < 8) {
                            sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 4] >> 2) & 3) << 4)) - 32;
                        } else if (is < 12) {
                            sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is] >> 4) & 3) << 4)) - 32;
                        } else {
                            sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is - 4] >> 6) & 3) << 4)) - 32;
                        }

                        // 2-bit quant: interleaved packing
                        int qs_byte_idx = 32 * n + l;
                        int qs_shift = 2 * j;
                        int ql_val = (int(blk->qs[qs_byte_idx]) >> qs_shift) & 3;

                        // High bit from hmask[32]: byte = i%32, bit = i/32
                        int high_bit = (int(blk->hmask[i % 32]) >> (i / 32)) & 1;
                        int qval = ql_val - (high_bit ? 0 : 4);

                        float w_val = d_all * float(sc_val) * float(qval);
                        sum_f += float(x_ptr[k]) * w_val;
                    }
                }
                break;
            }
            case 4: { // Q5_K — 256 elements per block
                device const block_q5_K *blk = reinterpret_cast<device const block_q5_K *>(block_ptr);
                float dall = float(blk->dm[0]);
                float dmin = float(blk->dm[1]);

                for (int i = int(simd_lane_id); i < QK_K; i += MOE_WARP_SIZE) {
                    int k = k_base + i;
                    if (k < size_k) {
                        // Q5_K: same 64-element group packing as Q4_K plus high bits
                        int il = i / 64;
                        int pos = i % 64;
                        int scale_idx = 2 * il + (pos >= 32 ? 1 : 0);

                        uint8_t sc_v, m_v;
                        if (scale_idx < 4) {
                            sc_v = blk->scales[scale_idx] & 63;
                            m_v = blk->scales[scale_idx + 4] & 63;
                        } else {
                            sc_v = (blk->scales[scale_idx + 4] & 0xF) | ((blk->scales[scale_idx - 4] >> 6) << 4);
                            m_v = (blk->scales[scale_idx + 4] >> 4) | ((blk->scales[scale_idx] >> 6) << 4);
                        }

                        int low_nibble;
                        if (pos < 32) {
                            low_nibble = int(blk->qs[32 * il + pos]) & 0xF;
                        } else {
                            low_nibble = int(blk->qs[32 * il + (pos - 32)]) >> 4;
                        }

                        // qh[32]: byte i%32, bit at position i/32
                        // But from CUDA: qh accessed at 2*ir, hm=1<<(2*il)
                        // For element i: qh_byte = i%32, hm_bit = 2*il + (pos>=32?1:0)
                        int hm_bit = 2 * il + (pos >= 32 ? 1 : 0);
                        int high_bit = (int(blk->qh[i % 32]) >> hm_bit) & 1;

                        int val = low_nibble + (high_bit << 4);
                        float w_val = dall * float(sc_v) * float(val) - dmin * float(m_v);
                        sum_f += float(x_ptr[k]) * w_val;
                    }
                }
                break;
            }
            case 5: { // Q6_K — 256 elements per block
                device const block_q6_K *blk = reinterpret_cast<device const block_q6_K *>(block_ptr);
                float d = float(blk->d);

                for (int i = int(simd_lane_id); i < QK_K; i += MOE_WARP_SIZE) {
                    int k = k_base + i;
                    if (k < size_k) {
                        int group = i / 16;
                        int8_t sc = blk->scales[group];

                        int half_idx = i / 128;
                        int within_half = i % 128;

                        int ql_val, qh_val;
                        if (within_half < 64) {
                            int ql_idx = half_idx * 64 + within_half;
                            int qh_idx = half_idx * 32 + (within_half % 32);
                            ql_val = int(blk->ql[ql_idx]) & 0xF;
                            qh_val = (int(blk->qh[qh_idx]) >> ((within_half / 32) * 2)) & 3;
                        } else {
                            int ql_idx = half_idx * 64 + (within_half - 64);
                            int qh_idx = half_idx * 32 + (within_half % 32);
                            ql_val = int(blk->ql[ql_idx]) >> 4;
                            int qh_shift = ((within_half - 64) / 32) * 2 + 4;
                            qh_val = (int(blk->qh[qh_idx]) >> qh_shift) & 3;
                        }

                        int val = ql_val | (qh_val << 4);
                        float w_val = d * float(sc) * float(int8_t(val) - 32);
                        sum_f += float(x_ptr[k]) * w_val;
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    // Warp reduction
    sum_f = simd_sum(sum_f);

    if (simd_lane_id == 0) {
        float tw = 1.0f;
        if (has_topk_weights != 0) {
            tw = topk_weights[token_id];
        }
        output[m_out * size_n + n_out] = sum_f * tw;
    }
}


// ============================================================
// GEMM Kernel — tiled with threadgroup dequantization
//
// Uses shared memory to cache dequantized weight tiles and
// input tiles, then performs tiled multiply-accumulate.
//
// Grid:  (ceil(N/BLOCK_N), ceil(size_m/BLOCK_M), 1)
// Block: (32, 1, 1) — one simdgroup
// BLOCK_M=32, BLOCK_N=32, BLOCK_K=qk (process one quant block per K-step)
// ============================================================

constant constexpr int GEMM_BLOCK_M = 32;
constant constexpr int GEMM_BLOCK_N = 32;

template <typename T>
[[kernel]] void moe_gguf_gemm_kernel(
    device const T            *input            [[ buffer(0) ]],
    device const uint8_t      *weights          [[ buffer(1) ]],
    device const int          *sorted_token_ids [[ buffer(2) ]],
    device const int          *expert_ids       [[ buffer(3) ]],
    device const float        *topk_weights     [[ buffer(4) ]],
    device       float        *output           [[ buffer(5) ]],
    constant     int          &num_experts      [[ buffer(6) ]],
    constant     int          &topk             [[ buffer(7) ]],
    constant     int          &size_m           [[ buffer(8) ]],
    constant     int          &size_n           [[ buffer(9) ]],
    constant     int          &size_k           [[ buffer(10) ]],
    constant     int          &has_topk_weights [[ buffer(11) ]],
    constant     int          &gguf_type        [[ buffer(12) ]],
    constant     int          &block_size_bytes [[ buffer(13) ]],
    constant     int          &qk               [[ buffer(14) ]],
    uint2 gid             [[ threadgroup_position_in_grid ]],
    uint  lane_id         [[ thread_index_in_simdgroup ]]
) {
    // Output tile position
    const int col_base = int(gid.x) * GEMM_BLOCK_N;
    const int row_base = int(gid.y) * GEMM_BLOCK_M;

    // Precompute token/expert mapping for this tile's rows
    int token_ids_local[GEMM_BLOCK_M];
    int expert_ids_local[GEMM_BLOCK_M];
    int input_tokens_local[GEMM_BLOCK_M];

    for (int i = 0; i < GEMM_BLOCK_M; ++i) {
        int m = row_base + i;
        if (m < size_m) {
            int tid = sorted_token_ids[m];
            token_ids_local[i] = tid;
            expert_ids_local[i] = expert_ids[m];
            input_tokens_local[i] = has_topk_weights ? tid : (tid / topk);
        } else {
            token_ids_local[i] = 0;
            expert_ids_local[i] = -1;
            input_tokens_local[i] = 0;
        }
    }

    // Find reference expert for weight loading (sorted tokens share same expert in a tile)
    int ref_expert = -1;
    for (int i = 0; i < GEMM_BLOCK_M; ++i) {
        if (expert_ids_local[i] >= 0) {
            ref_expert = expert_ids_local[i];
            break;
        }
    }

    // Weight stride calculations
    const long blocks_per_row = long(size_k) / long(qk);
    const long expert_stride = long(size_n) * blocks_per_row * long(block_size_bytes);

    // Accumulators — use simdgroup_matrix for 8x8 tiles
    threadgroup half s_a[GEMM_BLOCK_M][32]; // input tile (max BLOCK_K=32 per step)
    threadgroup half s_b[32][GEMM_BLOCK_N]; // weight tile (dequantized)
    simdgroup_matrix<float, 8, 8> acc[GEMM_BLOCK_M / 8][GEMM_BLOCK_N / 8];

    for (int i = 0; i < GEMM_BLOCK_M / 8; ++i) {
        for (int j = 0; j < GEMM_BLOCK_N / 8; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    int tid = int(lane_id);

    // K-loop: process qk elements at a time
    const int k_step = min(qk, 32);

    for (int k_block = 0; k_block < int(blocks_per_row); ++k_block) {
        int k_base = k_block * qk;

        // For K-quant blocks (qk=256), we need multiple sub-iterations
        int sub_steps = qk / k_step;

        for (int sub = 0; sub < sub_steps; ++sub) {
            int k_sub_base = k_base + sub * k_step;

            // Load A tile: input rows
            for (int i = 0; i < GEMM_BLOCK_M; ++i) {
                int gc = k_sub_base + tid;
                half val = half(0);
                if (expert_ids_local[i] >= 0 && gc < size_k) {
                    val = static_cast<half>(float(input[long(input_tokens_local[i]) * long(size_k) + gc]));
                }
                s_a[i][tid] = val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load B tile: dequantize weight column for this K sub-block
            if (ref_expert >= 0) {
                device const uint8_t *w_expert = weights + long(ref_expert) * expert_stride;

                for (int n_local = 0; n_local < GEMM_BLOCK_N; ++n_local) {
                    int gn = col_base + n_local;
                    if (gn < size_n) {
                        device const uint8_t *w_block = w_expert + long(gn) * blocks_per_row * long(block_size_bytes) + long(k_block) * long(block_size_bytes);

                        // Each lane dequantizes elements at stride
                        for (int i = tid; i < k_step; i += 32) {
                            int elem_in_block = sub * k_step + i;
                            float dq_val = 0.0f;

                            switch (gguf_type) {
                                case 0: { // Q8_0
                                    device const block_q8_0 *blk = reinterpret_cast<device const block_q8_0 *>(w_block);
                                    dq_val = float(blk->d) * float(blk->qs[elem_in_block]);
                                    break;
                                }
                                case 1: { // Q4_K
                                    device const block_q4_K *blk = reinterpret_cast<device const block_q4_K *>(w_block);
                                    int il4 = elem_in_block / 64;
                                    int pos4 = elem_in_block % 64;
                                    int si4 = 2 * il4 + (pos4 >= 32 ? 1 : 0);
                                    uint8_t sc_v, m_v;
                                    if (si4 < 4) { sc_v = blk->scales[si4] & 63; m_v = blk->scales[si4 + 4] & 63; }
                                    else { sc_v = (blk->scales[si4 + 4] & 0xF) | ((blk->scales[si4 - 4] >> 6) << 4); m_v = (blk->scales[si4 + 4] >> 4) | ((blk->scales[si4] >> 6) << 4); }
                                    int nibble;
                                    if (pos4 < 32) { nibble = int(blk->qs[32 * il4 + pos4]) & 0xF; }
                                    else { nibble = int(blk->qs[32 * il4 + (pos4 - 32)]) >> 4; }
                                    dq_val = float(blk->dm[0]) * float(sc_v) * float(nibble) - float(blk->dm[1]) * float(m_v);
                                    break;
                                }
                                case 2: { // Q2_K
                                    device const block_q2_K *blk = reinterpret_cast<device const block_q2_K *>(w_block);
                                    int n2 = elem_in_block / 128;
                                    int wh2 = elem_in_block % 128;
                                    uint8_t q = (blk->qs[32 * n2 + (wh2 % 32)] >> ((wh2 / 32) * 2)) & 3;
                                    uint8_t sc = blk->scales[elem_in_block / 16];
                                    dq_val = float(blk->dm[0]) * float(sc & 0xF) * float(q) - float(blk->dm[1]) * float(sc >> 4);
                                    break;
                                }
                                case 3: { // Q3_K
                                    device const block_q3_K *blk = reinterpret_cast<device const block_q3_K *>(w_block);
                                    int is = elem_in_block / 16;
                                    int sc_val;
                                    if (is < 4) { sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 8] >> 0) & 3) << 4)) - 32; }
                                    else if (is < 8) { sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 4] >> 2) & 3) << 4)) - 32; }
                                    else if (is < 12) { sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is] >> 4) & 3) << 4)) - 32; }
                                    else { sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is - 4] >> 6) & 3) << 4)) - 32; }
                                    int n3 = elem_in_block / 128;
                                    int j3 = (elem_in_block % 128) / 32;
                                    int l3 = elem_in_block % 32;
                                    int ql_val = (int(blk->qs[32 * n3 + l3]) >> (2 * j3)) & 3;
                                    int high_bit = (int(blk->hmask[elem_in_block % 32]) >> (elem_in_block / 32)) & 1;
                                    dq_val = float(blk->d) * float(sc_val) * float(ql_val - (high_bit ? 0 : 4));
                                    break;
                                }
                                case 4: { // Q5_K
                                    device const block_q5_K *blk = reinterpret_cast<device const block_q5_K *>(w_block);
                                    int il5 = elem_in_block / 64;
                                    int pos5 = elem_in_block % 64;
                                    int si5 = 2 * il5 + (pos5 >= 32 ? 1 : 0);
                                    uint8_t sc_v, m_v;
                                    if (si5 < 4) { sc_v = blk->scales[si5] & 63; m_v = blk->scales[si5 + 4] & 63; }
                                    else { sc_v = (blk->scales[si5 + 4] & 0xF) | ((blk->scales[si5 - 4] >> 6) << 4); m_v = (blk->scales[si5 + 4] >> 4) | ((blk->scales[si5] >> 6) << 4); }
                                    int low_nibble;
                                    if (pos5 < 32) { low_nibble = int(blk->qs[32 * il5 + pos5]) & 0xF; }
                                    else { low_nibble = int(blk->qs[32 * il5 + (pos5 - 32)]) >> 4; }
                                    int hm_bit = 2 * il5 + (pos5 >= 32 ? 1 : 0);
                                    int high_bit = (int(blk->qh[elem_in_block % 32]) >> hm_bit) & 1;
                                    dq_val = float(blk->dm[0]) * float(sc_v) * float(low_nibble + (high_bit << 4)) - float(blk->dm[1]) * float(m_v);
                                    break;
                                }
                                case 5: { // Q6_K
                                    device const block_q6_K *blk = reinterpret_cast<device const block_q6_K *>(w_block);
                                    int group = elem_in_block / 16;
                                    int8_t sc = blk->scales[group];
                                    int half_idx = elem_in_block / 128;
                                    int within_half = elem_in_block % 128;
                                    int ql_val, qh_val;
                                    if (within_half < 64) {
                                        ql_val = int(blk->ql[half_idx * 64 + within_half]) & 0xF;
                                        qh_val = (int(blk->qh[half_idx * 32 + (within_half % 32)]) >> ((within_half / 32) * 2)) & 3;
                                    } else {
                                        ql_val = int(blk->ql[half_idx * 64 + (within_half - 64)]) >> 4;
                                        int qh_shift = ((within_half - 64) / 32) * 2 + 4;
                                        qh_val = (int(blk->qh[half_idx * 32 + (within_half % 32)]) >> qh_shift) & 3;
                                    }
                                    dq_val = float(blk->d) * float(sc) * float(int8_t((ql_val | (qh_val << 4))) - 32);
                                    break;
                                }
                                default:
                                    break;
                            }

                            s_b[i][n_local] = half(dq_val);
                        }
                    } else {
                        for (int i = tid; i < k_step; i += 32) {
                            s_b[i][n_local] = half(0);
                        }
                    }
                }
            } else {
                for (int r = 0; r < k_step; ++r) {
                    for (int c = tid; c < GEMM_BLOCK_N; c += 32) {
                        s_b[r][c] = half(0);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute via simdgroup_matrix MMA — 8x8 tiles
            for (int k_tile = 0; k_tile < k_step; k_tile += 8) {
                simdgroup_matrix<half, 8, 8> fragA;
                simdgroup_matrix<half, 8, 8> fragB;

                for (int row_tile = 0; row_tile < GEMM_BLOCK_M / 8; ++row_tile) {
                    for (int col_tile = 0; col_tile < GEMM_BLOCK_N / 8; ++col_tile) {
                        simdgroup_load(fragA, &s_a[row_tile * 8][k_tile], 32, ulong2(0, 0), false);
                        simdgroup_load(fragB, &s_b[k_tile][col_tile * 8], GEMM_BLOCK_N, ulong2(0, 0), false);
                        simdgroup_multiply_accumulate(acc[row_tile][col_tile], fragA, fragB, acc[row_tile][col_tile]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        } // sub-step loop
    } // k_block loop

    // Store results
    threadgroup float s_out[GEMM_BLOCK_M][GEMM_BLOCK_N];

    for (int row_tile = 0; row_tile < GEMM_BLOCK_M / 8; ++row_tile) {
        for (int col_tile = 0; col_tile < GEMM_BLOCK_N / 8; ++col_tile) {
            simdgroup_store(acc[row_tile][col_tile], &s_out[row_tile * 8][col_tile * 8], GEMM_BLOCK_N, ulong2(0, 0), false);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int local_r = tid;
    int global_r = row_base + local_r;

    if (local_r < GEMM_BLOCK_M && global_r < size_m) {
        int token_id = sorted_token_ids[global_r];
        int eid = expert_ids[global_r];
        if (eid >= 0 && eid < num_experts) {
            float tw = 1.0f;
            if (has_topk_weights != 0) {
                tw = topk_weights[token_id];
            }
            for (int local_c = 0; local_c < GEMM_BLOCK_N; ++local_c) {
                int global_c = col_base + local_c;
                if (global_c < size_n) {
                    float val = s_out[local_r][local_c] * tw;
                    output[global_r * size_n + global_c] = val;
                }
            }
        }
    }
}


// ============================================================
// Template instantiations
// ============================================================

// GEMV instantiations
template [[host_name("moe_gguf_gemv_half")]] [[kernel]]
void moe_gguf_gemv_kernel<half>(
    device const half*, device const uint8_t*,
    device const int*, device const int*, device const float*,
    device float*,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    uint3, uint, uint);

#if defined(__HAVE_BFLOAT__)
template [[host_name("moe_gguf_gemv_bfloat16")]] [[kernel]]
void moe_gguf_gemv_kernel<bfloat16_t>(
    device const bfloat16_t*, device const uint8_t*,
    device const int*, device const int*, device const float*,
    device float*,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    uint3, uint, uint);
#endif

// GEMM instantiations
template [[host_name("moe_gguf_gemm_half")]] [[kernel]]
void moe_gguf_gemm_kernel<half>(
    device const half*, device const uint8_t*,
    device const int*, device const int*, device const float*,
    device float*,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    uint2, uint);

#if defined(__HAVE_BFLOAT__)
template [[host_name("moe_gguf_gemm_bfloat16")]] [[kernel]]
void moe_gguf_gemm_kernel<bfloat16_t>(
    device const bfloat16_t*, device const uint8_t*,
    device const int*, device const int*, device const float*,
    device float*,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    uint2, uint);
#endif
