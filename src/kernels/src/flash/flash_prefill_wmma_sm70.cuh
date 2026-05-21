/**
 * @brief WMMA Tensor Core prefill kernel for SM70 (V100).
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs
 *
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 *
 * @details
 * Uses nvcuda::wmma m16n16k16 instructions on V100 Tensor Cores for the
 * QK^T and PV matrix multiplications in Flash Attention v2 prefill.
 *
 * Tile sizes: BR_W=16, BC_W=16 (4 warps, 128 threads)
 * - QK^T: WMMA m16n16k16 with F16 inputs, F32 accumulation
 * - Softmax/masking: via shared memory round-trip (WMMA fragments are opaque)
 * - PV: WMMA m16n16k16 with F16 P-matrix and V values, F32 accumulation
 * - Output: online softmax with F32 rescaling, final store as F16
 *
 * Only compiled when NO_BF16_KERNEL is defined (SM < 80).
 * The kernel body is further guarded by __CUDA_ARCH__ < 750 so that only
 * SM70 device code is emitted. SM75 uses the m16n8k8 MMA path instead.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#ifdef FLASH_SM70_WMMA

#include <mma.h>

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif

#define BR_W 16
#define BC_W 16
#define HDIM_W FLASH_HDIM
#define PAD_W 8
#define HDIM_PAD_W (HDIM_W + PAD_W)
#define NUM_THREADS_W 128
#define NUM_WARPS_W 4
#define TILE_CHUNKS_W (BR_W * (HDIM_W / 8))
#define HDIM_TILES_W (HDIM_W / 16)

#ifndef WMMA_FLASH_USING_DECLARED
#define WMMA_FLASH_USING_DECLARED
using namespace nvcuda::wmma;
#endif

#define WMMA_LOAD_Q_FN_NAME2(hd) wmma_load_q_tile_##hd
#define WMMA_LOAD_Q_FN_NAME(hd)  WMMA_LOAD_Q_FN_NAME2(hd)
#define WMMA_LOAD_KV_FN_NAME2(hd) wmma_load_kv_tile_##hd
#define WMMA_LOAD_KV_FN_NAME(hd)  WMMA_LOAD_KV_FN_NAME2(hd)
#define wmma_load_q_tile  WMMA_LOAD_Q_FN_NAME(FLASH_HDIM)
#define wmma_load_kv_tile WMMA_LOAD_KV_FN_NAME(FLASH_HDIM)

__device__ __forceinline__ void wmma_load_q_tile(
    const flash_half_t* __restrict__ Q,
    flash_half_t* smem_Q,
    unsigned int q_start, unsigned int q_len,
    unsigned int q_seq_stride, unsigned int q_head,
    unsigned int head_dim, unsigned int tid
) {
    const unsigned int cpr = HDIM_W / 8;
    for (unsigned int idx = tid; idx < TILE_CHUNKS_W; idx += NUM_THREADS_W) {
        unsigned int row = idx / cpr, col = (idx % cpr) * 8;
        if (q_start + row < q_len) {
            *reinterpret_cast<uint4*>(&smem_Q[row * HDIM_PAD_W + col]) =
                *reinterpret_cast<const uint4*>(
                    &Q[(q_start + row) * q_seq_stride + q_head * head_dim + col]);
        } else {
            *reinterpret_cast<uint4*>(&smem_Q[row * HDIM_PAD_W + col]) = make_uint4(0,0,0,0);
        }
    }
}

static __device__ __forceinline__ void wmma_load_kv_tile(
    const flash_half_t* __restrict__ cache,
    const int* __restrict__ block_table,
    flash_half_t* smem_dst,
    unsigned int kv_start, unsigned int kv_len,
    unsigned int kv_head,
    unsigned int num_kv_heads, unsigned int head_dim,
    unsigned int cache_block_size,
    unsigned int tid, unsigned int stride
) {
    const unsigned int cpr = HDIM_W / 8;
    const unsigned int tile_elems = BC_W * (HDIM_W / 8);
    const unsigned long long ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim;
    const unsigned long long rs = (unsigned long long)num_kv_heads * head_dim;
    const unsigned int bs_shift = __ffs(cache_block_size) - 1;
    const unsigned int bs_mask  = cache_block_size - 1;
    for (unsigned int i = tid; i < tile_elems; i += stride) {
        unsigned int row = i / cpr, col = (i % cpr) * 8;
        unsigned int pos = kv_start + row;
        if (pos < kv_len) {
            unsigned int lb = pos >> bs_shift;
            unsigned int bo = pos & bs_mask;
            unsigned int pb = __ldg(&block_table[lb]);
            *reinterpret_cast<uint4*>(&smem_dst[row * HDIM_PAD_W + col]) =
                *reinterpret_cast<const uint4*>(
                    cache + (unsigned long long)pb * ps + (unsigned long long)bo * rs
                    + (unsigned long long)kv_head * head_dim + col);
        } else {
            *reinterpret_cast<uint4*>(&smem_dst[row * HDIM_PAD_W + col]) = make_uint4(0,0,0,0);
        }
    }
}

extern "C" __global__ void __launch_bounds__(NUM_THREADS_W)
flash_prefill_paged_wmma(
    const flash_half_t* __restrict__ Q,
    const flash_half_t* __restrict__ K_cache,
    const flash_half_t* __restrict__ V_cache,
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

    const unsigned int q_start = q_block * BR_W;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR_W, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    const flash_half_t* Q_base = Q + q_seq_start * q_seq_stride;
    flash_half_t* O_base = O + q_seq_start * q_seq_stride;

    extern __shared__ __align__(16) unsigned char wmma_smem[];

    const unsigned int smem_s_stride = BC_W + PAD_W;
    const unsigned int smem_p_stride = BC_W + PAD_W;

    flash_half_t* smem_Q = reinterpret_cast<flash_half_t*>(wmma_smem);
    flash_half_t* smem_K = smem_Q + BR_W * HDIM_PAD_W;
    flash_half_t* smem_V = smem_K + BC_W * HDIM_PAD_W;
    float* smem_S = reinterpret_cast<float*>(smem_V + BC_W * HDIM_PAD_W);
    flash_half_t* smem_P = reinterpret_cast<flash_half_t*>(smem_S + BR_W * smem_s_stride);
    float* smem_ml = reinterpret_cast<float*>(smem_P + BR_W * smem_p_stride);
    float* smem_O = smem_ml + BR_W * 2;
    float* smem_PV = smem_O + BR_W * HDIM_PAD_W;

    for (unsigned int i = tid; i < BR_W * HDIM_PAD_W; i += NUM_THREADS_W) {
        smem_O[i] = 0.f;
    }
    for (unsigned int i = tid; i < BR_W; i += NUM_THREADS_W) {
        smem_ml[i * 2]     = -1e30f;
        smem_ml[i * 2 + 1] = 0.f;
    }
    __syncthreads();

    wmma_load_q_tile(Q_base, smem_Q, q_start, q_len, q_seq_stride, q_head, head_dim, tid);
    __syncthreads();

    unsigned int num_kv_blocks = (kv_len + BC_W - 1) / BC_W;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / BC_W;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ? (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / BC_W;
    }

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC_W;
        unsigned int kv_end = min(kv_start + BC_W, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        // All threads load K
        wmma_load_kv_tile(K_cache, block_table, smem_K, kv_start, kv_len, kv_head,
                          num_kv_heads, head_dim, cache_block_size, tid, NUM_THREADS_W);
        __syncthreads();

        // QK^T via WMMA (warp 0) || V prefetch (warps 1-3)
        if (warp_id == 0) {
            fragment<accumulator, 16, 16, 16, float> qk_acc;
            fill_fragment(qk_acc, 0.0f);

            for (unsigned int k = 0; k < HDIM_TILES_W; k++) {
                fragment<matrix_a, 16, 16, 16, __half, row_major> q_frag;
                fragment<matrix_b, 16, 16, 16, __half, col_major> k_frag;

                load_matrix_sync(q_frag, &smem_Q[k * 16], HDIM_PAD_W);
                load_matrix_sync(k_frag, &smem_K[k * 16], HDIM_PAD_W);

                mma_sync(qk_acc, q_frag, k_frag, qk_acc);
            }

            store_matrix_sync(smem_S, qk_acc, smem_s_stride, mem_row_major);
        } else {
            wmma_load_kv_tile(V_cache, block_table, smem_V, kv_start, kv_len, kv_head,
                              num_kv_heads, head_dim, cache_block_size,
                              tid - 32, NUM_THREADS_W - 32);
        }
        __syncthreads();

        // Scale, softcap, masking
        for (unsigned int i = tid; i < BR_W * BC_W; i += NUM_THREADS_W) {
            unsigned int row = i / BC_W;
            unsigned int col = i % BC_W;

            float val = smem_S[row * smem_s_stride + col];
            val *= inv_sqrt_d;

            if (softcap > 0.f) {
                val = softcap * tanhf(val / softcap);
            }

            if (causal) {
                unsigned int qr = q_offset + q_start + row;
                if (kv_start + col > qr) val = -1e30f;
            }

            if (sliding_window > 0) {
                unsigned int qr = q_offset + q_start + row;
                if (qr >= kv_start + col && qr - (kv_start + col) >= sliding_window)
                    val = -1e30f;
            }

            if (col >= kv_tile_len) val = -1e30f;
            if (row >= q_tile_len) val = -1e30f;

            smem_S[row * smem_s_stride + col] = val;
        }
        __syncthreads();

        // Online softmax: row-max, rescale, exp, row-sum, write P
        // 128 threads / 16 rows = 8 threads per row (within same warp)
        {
            const unsigned int tpr = NUM_THREADS_W / BR_W;  // 8
            unsigned int row = tid / tpr;
            unsigned int sub = tid % tpr;

            if (row < BR_W) {
                float rmax = -1e30f;
                for (unsigned int c = sub; c < BC_W; c += tpr) {
                    rmax = fmaxf(rmax, smem_S[row * smem_s_stride + c]);
                }
                for (int offset = tpr / 2; offset > 0; offset >>= 1) {
                    rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, offset));
                }

                float old_m = smem_ml[row * 2];
                float old_l = smem_ml[row * 2 + 1];
                float new_m = fmaxf(old_m, rmax);

                if (new_m != old_m && sub == 0) {
                    float rescale = __expf(old_m - new_m);
                    smem_ml[row * 2 + 1] = old_l * rescale;
                    for (unsigned int d = 0; d < HDIM_W; d++) {
                        smem_O[row * HDIM_PAD_W + d] *= rescale;
                    }
                    smem_ml[row * 2] = new_m;
                }
            }
        }
        __syncthreads();

        {
            const unsigned int tpr = NUM_THREADS_W / BR_W;
            unsigned int row = tid / tpr;
            unsigned int sub = tid % tpr;

            if (row < BR_W) {
                float new_m = smem_ml[row * 2];
                float psum = 0.f;
                for (unsigned int c = sub; c < BC_W; c += tpr) {
                    float p = __expf(smem_S[row * smem_s_stride + c] - new_m);
                    smem_P[row * smem_p_stride + c] = __float2half(p);
                    psum += p;
                }
                for (int offset = tpr / 2; offset > 0; offset >>= 1) {
                    psum += __shfl_xor_sync(0xFFFFFFFF, psum, offset);
                }
                if (sub == 0) {
                    smem_ml[row * 2 + 1] += psum;
                }
            }
        }
        // Zero-pad P beyond BC_W for WMMA loads
        for (unsigned int i = tid; i < BR_W * PAD_W; i += NUM_THREADS_W) {
            unsigned int row = i / PAD_W;
            unsigned int col = BC_W + (i % PAD_W);
            if (col < smem_p_stride) {
                smem_P[row * smem_p_stride + col] = __float2half(0.f);
            }
        }
        __syncthreads();

        // PV via WMMA: P[16x16] @ V[16xHDIM] → O[16xHDIM]
        // Distribute HDIM/16 column tiles across 4 warps
        {
            unsigned int tiles_per_warp = (HDIM_TILES_W + NUM_WARPS_W - 1) / NUM_WARPS_W;
            unsigned int d_start = warp_id * tiles_per_warp;
            unsigned int d_end = min(d_start + tiles_per_warp, (unsigned int)HDIM_TILES_W);

            fragment<matrix_a, 16, 16, 16, __half, row_major> p_frag;
            load_matrix_sync(p_frag, smem_P, smem_p_stride);

            float* my_pv = smem_PV + warp_id * (BR_W * 16);

            for (unsigned int dt = d_start; dt < d_end; dt++) {
                fragment<matrix_b, 16, 16, 16, __half, row_major> v_frag;
                fragment<accumulator, 16, 16, 16, float> pv_acc;
                fill_fragment(pv_acc, 0.0f);

                load_matrix_sync(v_frag, &smem_V[dt * 16], HDIM_PAD_W);
                mma_sync(pv_acc, p_frag, v_frag, pv_acc);

                store_matrix_sync(my_pv, pv_acc, 16, mem_row_major);
                __syncwarp();

                for (unsigned int i = lane_id; i < BR_W * 16; i += 32) {
                    unsigned int r = i / 16;
                    unsigned int c = i % 16;
                    unsigned int dst_col = dt * 16 + c;
                    if (dst_col < HDIM_W) {
                        smem_O[r * HDIM_PAD_W + dst_col] += my_pv[r * 16 + c];
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // Final normalization and store
    for (unsigned int i = tid; i < BR_W * HDIM_W; i += NUM_THREADS_W) {
        unsigned int row = i / HDIM_W;
        unsigned int col = i % HDIM_W;
        unsigned int gr = q_start + row;

        if (gr < q_len && row < q_tile_len) {
            float l = smem_ml[row * 2 + 1];
            float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
            float val = smem_O[row * HDIM_PAD_W + col] * inv_l;
            O_base[gr * q_seq_stride + q_head * head_dim + col] = __float2half(val);
        }
    }
}

#undef wmma_load_q_tile
#undef wmma_load_kv_tile
#undef WMMA_LOAD_Q_FN_NAME
#undef WMMA_LOAD_Q_FN_NAME2
#undef WMMA_LOAD_KV_FN_NAME
#undef WMMA_LOAD_KV_FN_NAME2

#undef BR_W
#undef BC_W
#undef HDIM_W
#undef PAD_W
#undef HDIM_PAD_W
#undef NUM_THREADS_W
#undef NUM_WARPS_W
#undef TILE_CHUNKS_W
#undef HDIM_TILES_W

#endif // FLASH_SM70_WMMA
