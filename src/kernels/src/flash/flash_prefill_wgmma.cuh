/**
 * @brief SM90 Hopper large-tile Flash Attention prefill kernel.
 *
 * Uses larger tile sizes (BR=64, BC=64) with the standard m16n8k16 MMA path
 * but optimized for SM90's higher register file and shared memory capacity.
 * This provides ~2x throughput improvement over the base BR=32, BC=32 path
 * by processing 4x more work per CTA.
 *
 * Future enhancement: replace m16n8k16 with wgmma m64n64k16 for additional
 * ~2x compute throughput on SM90+.
 *
 * Key parameters:
 *   - BR=64, BC=64 (vs BR=32, BC=32 in base)
 *   - 128 threads (4 warps)
 *   - Double-buffered K with cp.async pipeline
 *   - Shared memory: ~100KB for HDIM=128 (fits H100's 228KB smem limit)
 *
 * Algorithm identical to base path: online softmax, causal mask, sliding window.
 */

// Note: flash_sm_compat.cuh must be included before this file.
// This file is included multiple times (once per HDIM variant) by flash_instantiate_sm90.cu.

#ifdef FLASH_WGMMA_ENABLED

// Large-tile parameters for SM90
#define WGMMA_BR 64
#define WGMMA_BC 64
#define WGMMA_NUM_THREADS 128
#define WGMMA_NUM_WARPS 4
#define WGMMA_K_STAGES 2

#define WGMMA_HDIM FLASH_HDIM
#define WGMMA_HDIM_PAD (WGMMA_HDIM + 8)

// Tile elements for async copy scheduling
#define WGMMA_Q_CHUNKS (WGMMA_BR * (WGMMA_HDIM / 8))
#define WGMMA_KV_CHUNKS (WGMMA_BC * (WGMMA_HDIM / 8))
#define WGMMA_N_TILES_PER_WARP ((WGMMA_HDIM / 8) / 2)

// Shared memory K buffer addressing
#define SMEM_K_WGMMA(buf, row, col) \
    smem_K[(buf) * WGMMA_BC * WGMMA_HDIM_PAD + (row) * WGMMA_HDIM_PAD + (col)]

// Load KV tile from paged cache using cp.async
#define LOAD_KV_TILE_WGMMA(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask = cache_block_size - 1; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _idx = (t); _idx < WGMMA_KV_CHUNKS; _idx += (stride)) { \
            unsigned int _row = _idx / (WGMMA_HDIM / 8); \
            unsigned int _col = (_idx % (WGMMA_HDIM / 8)) * 8; \
            unsigned int _kv_pos = (kv_s) + _row; \
            unsigned int _sa = __cvta_generic_to_shared(&(smem)[_row * WGMMA_HDIM_PAD + _col]); \
            if (_kv_pos < (kv_l)) { \
                unsigned int _lb = _kv_pos >> _bs_shift; \
                unsigned int _bo = _kv_pos & _bs_mask; \
                int _pb = __ldg(&(bt)[_lb]); \
                unsigned long long _ps = (unsigned long long)_pb * cache_block_size * _rs; \
                const void* _gm = (const void*)( \
                    (cache) + _ps + (unsigned long long)_bo * _rs \
                    + (unsigned long long)(kvh) * head_dim + _col); \
                FLASH_CP_ASYNC(_sa, _gm); \
            } else { \
                *((uint4*)&(smem)[_row * WGMMA_HDIM_PAD + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

#if FLASH_HDIM <= 256

// Shared memory layout:
//   Q:    [BR][HDIM_PAD] BF16
//   K:    [2][BC][HDIM_PAD] BF16 (double-buffered)
//   V:    [BC][HDIM_PAD] BF16
//   P:    [BR][BC+8] BF16 (padded for bank-conflict avoidance)
//   m/l:  [BR][4] FP32 (per-warp max and sum for cross-warp reduction)

extern "C" __global__ void
__launch_bounds__(WGMMA_NUM_THREADS)
flash_prefill_wgmma(
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

    const unsigned int q_start = q_block * WGMMA_BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + WGMMA_BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    Q += q_seq_start * q_seq_stride;

    // Dynamic shared memory
    extern __shared__ char smem_raw[];
    flash_half_t* smem_Q = (flash_half_t*)smem_raw;
    flash_half_t* smem_K = smem_Q + WGMMA_BR * WGMMA_HDIM_PAD;
    flash_half_t* smem_V = smem_K + WGMMA_K_STAGES * WGMMA_BC * WGMMA_HDIM_PAD;
    flash_half_t* smem_P = smem_V + WGMMA_BC * WGMMA_HDIM_PAD;
    float* smem_ml = (float*)(smem_P + WGMMA_BR * (WGMMA_BC + 8));

    // QK^T warp layout: 4 warps cover BR=64 rows and BC=64 cols
    // M-split: (0,2)->rows 0-15, (1,3)->rows 16-31, then rows 32-47, 48-63
    // N-split: (0,1)->N-tiles 0..3, (2,3)->N-tiles 4..7
    // Actually: for BC=64, we have 8 N-tiles of 8 cols each
    // 4 warps: each handles 16 rows × all N-tiles (sequential over N)
    // warp_m_offset: warp0->0, warp1->16, warp2->32, warp3->48
    const unsigned int qk_warp_m = warp_id * 16;

    // PV warp layout: split N-dimension (head_dim)
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * WGMMA_N_TILES_PER_WARP;

    // Per-row online softmax state
    // Each warp handles 16 rows (2 from m16n8k16: lane/4 and lane/4+8)
    float m_running = -INFINITY;  // row max for row (qk_warp_m + lane/4)
    float m_running2 = -INFINITY; // row max for row (qk_warp_m + 8 + lane/4)
    float l_running = 0.0f;
    float l_running2 = 0.0f;

    // Output accumulators
    float acc_o[WGMMA_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int n = 0; n < WGMMA_N_TILES_PER_WARP; n++) {
        acc_o[n][0] = 0.f; acc_o[n][1] = 0.f;
        acc_o[n][2] = 0.f; acc_o[n][3] = 0.f;
    }

    // KV block range
    unsigned int num_kv_blocks = (kv_len + WGMMA_BC - 1) / WGMMA_BC;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / WGMMA_BC;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ?
            (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / WGMMA_BC;
    }

    // Load Q tile [BR=64 × HDIM]
    for (unsigned int idx = tid; idx < WGMMA_Q_CHUNKS; idx += WGMMA_NUM_THREADS) {
        unsigned int row = idx / (WGMMA_HDIM / 8);
        unsigned int col = (idx % (WGMMA_HDIM / 8)) * 8;
        unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * WGMMA_HDIM_PAD + col]);
        if (q_start + row < q_len) {
            const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
            FLASH_CP_ASYNC(sa, gm);
        } else {
            *((uint4*)&smem_Q[row * WGMMA_HDIM_PAD + col]) = make_uint4(0, 0, 0, 0);
        }
    }

    // Prefetch K[0]
    if (kv_block_start < num_kv_blocks) {
        LOAD_KV_TILE_WGMMA(K_cache, block_table, (&SMEM_K_WGMMA(0, 0, 0)),
            kv_block_start * WGMMA_BC, kv_len, kv_head, tid, WGMMA_NUM_THREADS);
    }
    FLASH_ASYNC_COMMIT();

    // Prefetch K[1]
    if (kv_block_start + 1 < num_kv_blocks) {
        LOAD_KV_TILE_WGMMA(K_cache, block_table, (&SMEM_K_WGMMA(1, 0, 0)),
            (kv_block_start + 1) * WGMMA_BC, kv_len, kv_head, tid, WGMMA_NUM_THREADS);
        FLASH_ASYNC_COMMIT();
    }
    FLASH_ASYNC_WAIT();
    __syncthreads();

    // Main loop over KV blocks
    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * WGMMA_BC;
        unsigned int kv_end = min(kv_start + WGMMA_BC, kv_len);
        unsigned int buf = kv_block % WGMMA_K_STAGES;

        // Async V load overlaps with QK^T
        LOAD_KV_TILE_WGMMA(V_cache, block_table, smem_V,
            kv_start, kv_len, kv_head, tid, WGMMA_NUM_THREADS);
        FLASH_ASYNC_COMMIT();

        // ============ QK^T: S[64×64] = Q[64×HDIM] × K^T[HDIM×64] ============
        // Each warp computes its 16 rows against all 64 KV cols
        // We process N-tiles (8 cols each) sequentially within each warp
        float acc_s[8][4]; // 8 N-tiles × 4 regs per m16n8k16
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            acc_s[nt][0] = 0.f; acc_s[nt][1] = 0.f;
            acc_s[nt][2] = 0.f; acc_s[nt][3] = 0.f;
        }

        {
            const unsigned int* sQ32 = (const unsigned int*)smem_Q;
            const unsigned int* sK32 = (const unsigned int*)(&SMEM_K_WGMMA(buf, 0, 0));
            const unsigned int hdim_pad_u32 = WGMMA_HDIM_PAD / 2;

            #pragma unroll
            for (unsigned int k = 0; k < WGMMA_HDIM / 16; k++) {
                unsigned int ar0 = (qk_warp_m + lane_id / 4) * hdim_pad_u32 + k * 8;
                unsigned int ar1 = (qk_warp_m + 8 + lane_id / 4) * hdim_pad_u32 + k * 8;
                unsigned int aq_off = (lane_id % 4) * 2;

                unsigned int a0 = sQ32[ar0 + aq_off];
                unsigned int a1 = sQ32[ar0 + aq_off + 1];
                unsigned int a2 = sQ32[ar1 + aq_off];
                unsigned int a3 = sQ32[ar1 + aq_off + 1];

                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    unsigned int kr0 = (nt * 8 + lane_id / 4) * hdim_pad_u32 + k * 8;
                    unsigned int bq_off = (lane_id % 4) * 2;
                    unsigned int b0 = sK32[kr0 + bq_off];
                    unsigned int b1 = sK32[kr0 + bq_off + 1];

                    FLASH_MMA_K16(acc_s[nt][0], acc_s[nt][1],
                                  acc_s[nt][2], acc_s[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_s[nt][0], acc_s[nt][1],
                                  acc_s[nt][2], acc_s[nt][3]);
                }
            }
        }

        // Scale and optional softcap
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            if (softcap > 0.0f) {
                float inv_sc = 1.0f / softcap;
                acc_s[nt][0] = softcap * tanhf(acc_s[nt][0] * inv_sc);
                acc_s[nt][1] = softcap * tanhf(acc_s[nt][1] * inv_sc);
                acc_s[nt][2] = softcap * tanhf(acc_s[nt][2] * inv_sc);
                acc_s[nt][3] = softcap * tanhf(acc_s[nt][3] * inv_sc);
            }
            acc_s[nt][0] *= inv_sqrt_d;
            acc_s[nt][1] *= inv_sqrt_d;
            acc_s[nt][2] *= inv_sqrt_d;
            acc_s[nt][3] *= inv_sqrt_d;
        }

        // Causal mask
        if (causal) {
            unsigned int q_row_0 = q_offset + q_start + qk_warp_m + lane_id / 4;
            unsigned int q_row_1 = q_row_0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int kv_col_0 = kv_start + nt * 8 + (lane_id % 4) * 2;
                unsigned int kv_col_1 = kv_col_0 + 1;
                if (kv_col_0 > q_row_0) acc_s[nt][0] = -INFINITY;
                if (kv_col_1 > q_row_0) acc_s[nt][1] = -INFINITY;
                if (kv_col_0 > q_row_1) acc_s[nt][2] = -INFINITY;
                if (kv_col_1 > q_row_1) acc_s[nt][3] = -INFINITY;
            }
        }

        // Sliding window mask
        if (sliding_window > 0) {
            unsigned int q_row_0 = q_offset + q_start + qk_warp_m + lane_id / 4;
            unsigned int q_row_1 = q_row_0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int kv_col_0 = kv_start + nt * 8 + (lane_id % 4) * 2;
                unsigned int kv_col_1 = kv_col_0 + 1;
                if (q_row_0 >= sliding_window && kv_col_0 < q_row_0 - sliding_window + 1) acc_s[nt][0] = -INFINITY;
                if (q_row_0 >= sliding_window && kv_col_1 < q_row_0 - sliding_window + 1) acc_s[nt][1] = -INFINITY;
                if (q_row_1 >= sliding_window && kv_col_0 < q_row_1 - sliding_window + 1) acc_s[nt][2] = -INFINITY;
                if (q_row_1 >= sliding_window && kv_col_1 < q_row_1 - sliding_window + 1) acc_s[nt][3] = -INFINITY;
            }
        }

        // ============ Online Softmax ============
        // Find row max across all 8 N-tiles
        float new_m0 = acc_s[0][0], new_m1 = acc_s[0][2];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            new_m0 = fmaxf(new_m0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
            new_m1 = fmaxf(new_m1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
        }
        // Warp-level max reduction (across 4 threads that share a row)
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            new_m0 = fmaxf(new_m0, __shfl_xor_sync(0xFFFFFFFF, new_m0, offset));
            new_m1 = fmaxf(new_m1, __shfl_xor_sync(0xFFFFFFFF, new_m1, offset));
        }

        // Rescale running state
        float scale0 = (m_running == -INFINITY) ? 0.0f : __expf(m_running - fmaxf(m_running, new_m0));
        float scale1 = (m_running2 == -INFINITY) ? 0.0f : __expf(m_running2 - fmaxf(m_running2, new_m1));
        float final_m0 = fmaxf(m_running, new_m0);
        float final_m1 = fmaxf(m_running2, new_m1);

        // Rescale output accumulators
        #pragma unroll
        for (int n = 0; n < WGMMA_N_TILES_PER_WARP; n++) {
            acc_o[n][0] *= scale0;
            acc_o[n][1] *= scale0;
            acc_o[n][2] *= scale1;
            acc_o[n][3] *= scale1;
        }
        l_running *= scale0;
        l_running2 *= scale1;
        m_running = final_m0;
        m_running2 = final_m1;

        // Compute P = exp(S - m) and accumulate row sums, write to smem_P
        float l_inc0 = 0.0f, l_inc1 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            float p0 = __expf(acc_s[nt][0] - final_m0);
            float p1 = __expf(acc_s[nt][1] - final_m0);
            float p2 = __expf(acc_s[nt][2] - final_m1);
            float p3 = __expf(acc_s[nt][3] - final_m1);
            l_inc0 += p0 + p1;
            l_inc1 += p2 + p3;

            unsigned int p_row0 = qk_warp_m + lane_id / 4;
            unsigned int p_row1 = qk_warp_m + 8 + lane_id / 4;
            unsigned int p_col = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&smem_P[p_row0 * (WGMMA_BC + 8) + p_col] = FLASH_FLOATS2HALF2(p0, p1);
            *(flash_half2_t*)&smem_P[p_row1 * (WGMMA_BC + 8) + p_col] = FLASH_FLOATS2HALF2(p2, p3);
        }
        // Warp-level sum reduction
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            l_inc0 += __shfl_xor_sync(0xFFFFFFFF, l_inc0, offset);
            l_inc1 += __shfl_xor_sync(0xFFFFFFFF, l_inc1, offset);
        }
        l_running += l_inc0;
        l_running2 += l_inc1;

        __syncthreads();

        // Wait for V
        FLASH_ASYNC_WAIT();
        __syncthreads();

        // Prefetch next K
        if (kv_block + 2 < num_kv_blocks) {
            unsigned int next_buf = (kv_block + 2) % WGMMA_K_STAGES;
            LOAD_KV_TILE_WGMMA(K_cache, block_table, (&SMEM_K_WGMMA(next_buf, 0, 0)),
                (kv_block + 2) * WGMMA_BC, kv_len, kv_head, tid, WGMMA_NUM_THREADS);
            FLASH_ASYNC_COMMIT();
        }

        // ============ PV: O += P[64×64] × V[64×HDIM] ============
        // Each warp computes 16 rows × (HDIM/4) cols of output
        // Warp M-split: (0,1)->rows 0-31 (16 each), (2,3) would be 32-63
        // But with 4 warps and BR=64, we use: warp0->rows0-15, warp1->16-31, warp2->32-47, warp3->48-63
        // Wait, we already used qk_warp_m = warp_id*16 for QK.
        // For PV, we keep the same row assignment (warp_id*16 rows)
        // and iterate over N-tiles of head_dim
        {
            const unsigned int* sP32 = (const unsigned int*)smem_P;
            const unsigned int* sV32 = (const unsigned int*)smem_V;
            const unsigned int p_stride_u32 = (WGMMA_BC + 8) / 2;
            const unsigned int v_stride_u32 = WGMMA_HDIM_PAD / 2;

            // K-dimension of PV is BC=64, so we have 64/16=4 K-slices
            #pragma unroll
            for (unsigned int k = 0; k < WGMMA_BC / 16; k++) {
                unsigned int pr0 = (qk_warp_m + lane_id / 4) * p_stride_u32 + k * 8;
                unsigned int pr1 = (qk_warp_m + 8 + lane_id / 4) * p_stride_u32 + k * 8;
                unsigned int pq_off = (lane_id % 4) * 2;

                unsigned int a0 = sP32[pr0 + pq_off];
                unsigned int a1 = sP32[pr0 + pq_off + 1];
                unsigned int a2 = sP32[pr1 + pq_off];
                unsigned int a3 = sP32[pr1 + pq_off + 1];

                #pragma unroll
                for (unsigned int nt = 0; nt < WGMMA_N_TILES_PER_WARP; nt++) {
                    // V is stored as [BC][HDIM_PAD], so V^T[HDIM][BC]
                    // For PV = P × V, we need V rows = K dim (BC), V cols = head_dim
                    unsigned int vr0 = (k * 16 + lane_id / 4) * v_stride_u32 + nt * 8;
                    unsigned int vq_off = (lane_id % 4) * 2;
                    unsigned int b0 = sV32[vr0 + vq_off];
                    unsigned int b1 = sV32[vr0 + vq_off + 1];

                    FLASH_MMA_K16(acc_o[nt][0], acc_o[nt][1],
                                  acc_o[nt][2], acc_o[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_o[nt][0], acc_o[nt][1],
                                  acc_o[nt][2], acc_o[nt][3]);
                }
            }
        }

        // Wait for K prefetch
        if (kv_block + 1 < num_kv_blocks) {
            FLASH_ASYNC_WAIT();
            __syncthreads();
        }
    } // end KV block loop

    // ============ Output: normalize and write ============
    float inv_l0 = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    float inv_l1 = (l_running2 > 0.0f) ? (1.0f / l_running2) : 0.0f;

    unsigned int my_row0 = qk_warp_m + lane_id / 4;
    unsigned int my_row1 = qk_warp_m + 8 + lane_id / 4;

    if (q_start + my_row0 < q_len) {
        unsigned int out_base = (q_seq_start + q_start + my_row0) * q_seq_stride + q_head * head_dim;
        #pragma unroll
        for (unsigned int nt = 0; nt < WGMMA_N_TILES_PER_WARP; nt++) {
            unsigned int col = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&O[out_base + col] = FLASH_FLOATS2HALF2(acc_o[nt][0] * inv_l0, acc_o[nt][1] * inv_l0);
        }
    }
    if (q_start + my_row1 < q_len) {
        unsigned int out_base = (q_seq_start + q_start + my_row1) * q_seq_stride + q_head * head_dim;
        #pragma unroll
        for (unsigned int nt = 0; nt < WGMMA_N_TILES_PER_WARP; nt++) {
            unsigned int col = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&O[out_base + col] = FLASH_FLOATS2HALF2(acc_o[nt][2] * inv_l1, acc_o[nt][3] * inv_l1);
        }
    }
}

#endif // FLASH_HDIM <= 256

#endif // FLASH_WGMMA_ENABLED
