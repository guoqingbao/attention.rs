/**
 * @brief SM100/SM120+ Blackwell tcgen05 Flash Attention prefill kernel.
 *
 * Uses the same m16n8k16 MMA with larger tile (BR=128, BC=128) for Blackwell
 * which has 256KB shared memory. The tcgen05 macro infrastructure is defined
 * in flash_sm_compat.cuh for future native hardware instruction use.
 *
 * Each warp handles 32 Q rows. Within each warp:
 *   - First 16 rows: handled by one m16n8k16 call (via rows lane_id/4 and +8)
 *   - Second 16 rows: handled by another m16n8k16 call
 * Each needs independent online softmax state.
 *
 * This kernel uses 128 threads (4 warps):
 *   warp0: rows 0-31, warp1: rows 32-63, warp2: rows 64-95, warp3: rows 96-127
 *
 * Algorithm structure identical to base path: online softmax, causal mask, paged KV.
 */

#ifndef FLASH_SM_COMPAT_INCLUDED
#include "flash_sm_compat.cuh"
#define FLASH_SM_COMPAT_INCLUDED
#endif

#ifdef FLASH_TCGEN05_ENABLED

#define TCGEN_BR 128
#define TCGEN_BC 128
#define TCGEN_NUM_THREADS 128
#define TCGEN_NUM_WARPS 4
#define TCGEN_K_STAGES 2

#define TCGEN_HDIM FLASH_HDIM
#define TCGEN_HDIM_PAD (TCGEN_HDIM + 8)

#define TCGEN_Q_CHUNKS (TCGEN_BR * (TCGEN_HDIM / 8))
#define TCGEN_KV_CHUNKS (TCGEN_BC * (TCGEN_HDIM / 8))

// N-tiles for output dimension (HDIM/8 cols per 8-wide MMA N-tile, 2 per m16n8k16 call)
#define TCGEN_O_NTILES ((TCGEN_HDIM / 8) / 2)

// K double-buffer addressing
#define SMEM_K_TCGEN(buf, row, col) \
    smem_K[(buf) * TCGEN_BC * TCGEN_HDIM_PAD + (row) * TCGEN_HDIM_PAD + (col)]

// Load KV tile from paged cache
#define LOAD_KV_TILE_TCGEN(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask = cache_block_size - 1; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _idx = (t); _idx < TCGEN_KV_CHUNKS; _idx += (stride)) { \
            unsigned int _row = _idx / (TCGEN_HDIM / 8); \
            unsigned int _col = (_idx % (TCGEN_HDIM / 8)) * 8; \
            unsigned int _kv_pos = (kv_s) + _row; \
            unsigned int _sa = __cvta_generic_to_shared(&(smem)[_row * TCGEN_HDIM_PAD + _col]); \
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
                *((uint4*)&(smem)[_row * TCGEN_HDIM_PAD + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

#if FLASH_HDIM <= 256

extern "C" __global__ void
__launch_bounds__(TCGEN_NUM_THREADS)
flash_prefill_tcgen05(
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

    const unsigned int q_start = q_block * TCGEN_BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + TCGEN_BR, q_len);
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    Q += q_seq_start * q_seq_stride;

    extern __shared__ char smem_raw[];
    flash_half_t* smem_Q = (flash_half_t*)smem_raw;
    flash_half_t* smem_K = smem_Q + TCGEN_BR * TCGEN_HDIM_PAD;
    flash_half_t* smem_V = smem_K + TCGEN_K_STAGES * TCGEN_BC * TCGEN_HDIM_PAD;
    flash_half_t* smem_P = smem_V + TCGEN_BC * TCGEN_HDIM_PAD;
    float* smem_ml = (float*)(smem_P + TCGEN_BR * (TCGEN_BC + 8));

    // Each warp handles 32 rows. Process as two independent 16-row halves.
    // warp0: [0..15] and [16..31], warp1: [32..47] and [48..63], etc.
    const unsigned int warp_row_base = warp_id * 32;

    // Independent softmax state for each 16-row half
    // Half A: rows warp_row_base + 0..15  (m16n8k16 rows: lane_id/4 and +8)
    // Half B: rows warp_row_base + 16..31
    float m_a = -INFINITY, l_a = 0.0f;  // half A: rows 0-7 via [0],[1]
    float m_a2 = -INFINITY, l_a2 = 0.0f; // half A: rows 8-15 via [2],[3]
    float m_b = -INFINITY, l_b = 0.0f;  // half B: rows 16-23
    float m_b2 = -INFINITY, l_b2 = 0.0f; // half B: rows 24-31

    // Output accumulators for both halves
    float acc_oA[TCGEN_O_NTILES][4]; // half A output
    float acc_oB[TCGEN_O_NTILES][4]; // half B output
    #pragma unroll
    for (int n = 0; n < TCGEN_O_NTILES; n++) {
        acc_oA[n][0] = 0.f; acc_oA[n][1] = 0.f; acc_oA[n][2] = 0.f; acc_oA[n][3] = 0.f;
        acc_oB[n][0] = 0.f; acc_oB[n][1] = 0.f; acc_oB[n][2] = 0.f; acc_oB[n][3] = 0.f;
    }

    // KV block range
    unsigned int num_kv_blocks = (kv_len + TCGEN_BC - 1) / TCGEN_BC;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / TCGEN_BC;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ?
            (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / TCGEN_BC;
    }

    // Load Q tile [BR=128 × HDIM]
    for (unsigned int idx = tid; idx < TCGEN_Q_CHUNKS; idx += TCGEN_NUM_THREADS) {
        unsigned int row = idx / (TCGEN_HDIM / 8);
        unsigned int col = (idx % (TCGEN_HDIM / 8)) * 8;
        unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * TCGEN_HDIM_PAD + col]);
        if (q_start + row < q_len) {
            const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
            FLASH_CP_ASYNC(sa, gm);
        } else {
            *((uint4*)&smem_Q[row * TCGEN_HDIM_PAD + col]) = make_uint4(0, 0, 0, 0);
        }
    }

    // Prefetch K
    if (kv_block_start < num_kv_blocks) {
        LOAD_KV_TILE_TCGEN(K_cache, block_table, (&SMEM_K_TCGEN(0, 0, 0)),
            kv_block_start * TCGEN_BC, kv_len, kv_head, tid, TCGEN_NUM_THREADS);
    }
    FLASH_ASYNC_COMMIT();
    if (kv_block_start + 1 < num_kv_blocks) {
        LOAD_KV_TILE_TCGEN(K_cache, block_table, (&SMEM_K_TCGEN(1, 0, 0)),
            (kv_block_start + 1) * TCGEN_BC, kv_len, kv_head, tid, TCGEN_NUM_THREADS);
        FLASH_ASYNC_COMMIT();
    }
    FLASH_ASYNC_WAIT();
    __syncthreads();

    // Stride for P matrix in shared memory
    const unsigned int p_stride = TCGEN_BC + 8;

    // Main loop over KV blocks
    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * TCGEN_BC;
        unsigned int buf = kv_block % TCGEN_K_STAGES;

        LOAD_KV_TILE_TCGEN(V_cache, block_table, smem_V,
            kv_start, kv_len, kv_head, tid, TCGEN_NUM_THREADS);
        FLASH_ASYNC_COMMIT();

        // ============ QK^T for both halves ============
        float acc_sA[16][4]; // half A: S scores for 16 rows × 128 cols
        float acc_sB[16][4]; // half B: S scores for 16 rows × 128 cols
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            acc_sA[nt][0] = 0.f; acc_sA[nt][1] = 0.f; acc_sA[nt][2] = 0.f; acc_sA[nt][3] = 0.f;
            acc_sB[nt][0] = 0.f; acc_sB[nt][1] = 0.f; acc_sB[nt][2] = 0.f; acc_sB[nt][3] = 0.f;
        }

        {
            const unsigned int* sQ32 = (const unsigned int*)smem_Q;
            const unsigned int* sK32 = (const unsigned int*)(&SMEM_K_TCGEN(buf, 0, 0));
            const unsigned int hdim_pad_u32 = TCGEN_HDIM_PAD / 2;

            #pragma unroll
            for (unsigned int k = 0; k < TCGEN_HDIM / 16; k++) {
                // Half A: rows warp_row_base + 0..15
                unsigned int arA0 = (warp_row_base + lane_id / 4) * hdim_pad_u32 + k * 8;
                unsigned int arA1 = (warp_row_base + 8 + lane_id / 4) * hdim_pad_u32 + k * 8;
                unsigned int aq_off = (lane_id % 4) * 2;
                unsigned int aA0 = sQ32[arA0 + aq_off];
                unsigned int aA1 = sQ32[arA0 + aq_off + 1];
                unsigned int aA2 = sQ32[arA1 + aq_off];
                unsigned int aA3 = sQ32[arA1 + aq_off + 1];

                // Half B: rows warp_row_base + 16..31
                unsigned int arB0 = (warp_row_base + 16 + lane_id / 4) * hdim_pad_u32 + k * 8;
                unsigned int arB1 = (warp_row_base + 24 + lane_id / 4) * hdim_pad_u32 + k * 8;
                unsigned int aB0 = sQ32[arB0 + aq_off];
                unsigned int aB1 = sQ32[arB0 + aq_off + 1];
                unsigned int aB2 = sQ32[arB1 + aq_off];
                unsigned int aB3 = sQ32[arB1 + aq_off + 1];

                #pragma unroll
                for (int nt = 0; nt < 16; nt++) {
                    unsigned int kr0 = (nt * 8 + lane_id / 4) * hdim_pad_u32 + k * 8;
                    unsigned int bq_off = (lane_id % 4) * 2;
                    unsigned int b0 = sK32[kr0 + bq_off];
                    unsigned int b1 = sK32[kr0 + bq_off + 1];

                    FLASH_MMA_K16(acc_sA[nt][0], acc_sA[nt][1], acc_sA[nt][2], acc_sA[nt][3],
                                  aA0, aA1, aA2, aA3, b0, b1,
                                  acc_sA[nt][0], acc_sA[nt][1], acc_sA[nt][2], acc_sA[nt][3]);

                    FLASH_MMA_K16(acc_sB[nt][0], acc_sB[nt][1], acc_sB[nt][2], acc_sB[nt][3],
                                  aB0, aB1, aB2, aB3, b0, b1,
                                  acc_sB[nt][0], acc_sB[nt][1], acc_sB[nt][2], acc_sB[nt][3]);
                }
            }
        }

        // Scale and softcap
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (softcap > 0.0f) {
                    float inv_sc = 1.0f / softcap;
                    acc_sA[nt][i] = softcap * tanhf(acc_sA[nt][i] * inv_sc);
                    acc_sB[nt][i] = softcap * tanhf(acc_sB[nt][i] * inv_sc);
                }
                acc_sA[nt][i] *= inv_sqrt_d;
                acc_sB[nt][i] *= inv_sqrt_d;
            }
        }

        // Causal mask
        if (causal) {
            unsigned int qrA0 = q_offset + q_start + warp_row_base + lane_id / 4;
            unsigned int qrA1 = qrA0 + 8;
            unsigned int qrB0 = q_offset + q_start + warp_row_base + 16 + lane_id / 4;
            unsigned int qrB1 = qrB0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned int kvc0 = kv_start + nt * 8 + (lane_id % 4) * 2;
                unsigned int kvc1 = kvc0 + 1;
                if (kvc0 > qrA0) acc_sA[nt][0] = -INFINITY;
                if (kvc1 > qrA0) acc_sA[nt][1] = -INFINITY;
                if (kvc0 > qrA1) acc_sA[nt][2] = -INFINITY;
                if (kvc1 > qrA1) acc_sA[nt][3] = -INFINITY;
                if (kvc0 > qrB0) acc_sB[nt][0] = -INFINITY;
                if (kvc1 > qrB0) acc_sB[nt][1] = -INFINITY;
                if (kvc0 > qrB1) acc_sB[nt][2] = -INFINITY;
                if (kvc1 > qrB1) acc_sB[nt][3] = -INFINITY;
            }
        }

        // ============ Online Softmax - Half A (rows 0..15) ============
        float nmA = -INFINITY, nmA2 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            nmA = fmaxf(nmA, fmaxf(acc_sA[nt][0], acc_sA[nt][1]));
            nmA2 = fmaxf(nmA2, fmaxf(acc_sA[nt][2], acc_sA[nt][3]));
        }
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            nmA = fmaxf(nmA, __shfl_xor_sync(0xFFFFFFFF, nmA, off));
            nmA2 = fmaxf(nmA2, __shfl_xor_sync(0xFFFFFFFF, nmA2, off));
        }

        float scA = (m_a == -INFINITY) ? 0.0f : __expf(m_a - fmaxf(m_a, nmA));
        float scA2 = (m_a2 == -INFINITY) ? 0.0f : __expf(m_a2 - fmaxf(m_a2, nmA2));
        m_a = fmaxf(m_a, nmA);
        m_a2 = fmaxf(m_a2, nmA2);

        #pragma unroll
        for (int n = 0; n < TCGEN_O_NTILES; n++) {
            acc_oA[n][0] *= scA; acc_oA[n][1] *= scA;
            acc_oA[n][2] *= scA2; acc_oA[n][3] *= scA2;
        }
        l_a *= scA; l_a2 *= scA2;

        float lincA = 0.0f, lincA2 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            float p0 = __expf(acc_sA[nt][0] - m_a);
            float p1 = __expf(acc_sA[nt][1] - m_a);
            float p2 = __expf(acc_sA[nt][2] - m_a2);
            float p3 = __expf(acc_sA[nt][3] - m_a2);
            lincA += p0 + p1; lincA2 += p2 + p3;
            unsigned int pr0 = warp_row_base + lane_id / 4;
            unsigned int pr1 = warp_row_base + 8 + lane_id / 4;
            unsigned int pc = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&smem_P[pr0 * p_stride + pc] = FLASH_FLOATS2HALF2(p0, p1);
            *(flash_half2_t*)&smem_P[pr1 * p_stride + pc] = FLASH_FLOATS2HALF2(p2, p3);
        }
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            lincA += __shfl_xor_sync(0xFFFFFFFF, lincA, off);
            lincA2 += __shfl_xor_sync(0xFFFFFFFF, lincA2, off);
        }
        l_a += lincA; l_a2 += lincA2;

        // ============ Online Softmax - Half B (rows 16..31) ============
        float nmB = -INFINITY, nmB2 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            nmB = fmaxf(nmB, fmaxf(acc_sB[nt][0], acc_sB[nt][1]));
            nmB2 = fmaxf(nmB2, fmaxf(acc_sB[nt][2], acc_sB[nt][3]));
        }
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            nmB = fmaxf(nmB, __shfl_xor_sync(0xFFFFFFFF, nmB, off));
            nmB2 = fmaxf(nmB2, __shfl_xor_sync(0xFFFFFFFF, nmB2, off));
        }

        float scB = (m_b == -INFINITY) ? 0.0f : __expf(m_b - fmaxf(m_b, nmB));
        float scB2 = (m_b2 == -INFINITY) ? 0.0f : __expf(m_b2 - fmaxf(m_b2, nmB2));
        m_b = fmaxf(m_b, nmB);
        m_b2 = fmaxf(m_b2, nmB2);

        #pragma unroll
        for (int n = 0; n < TCGEN_O_NTILES; n++) {
            acc_oB[n][0] *= scB; acc_oB[n][1] *= scB;
            acc_oB[n][2] *= scB2; acc_oB[n][3] *= scB2;
        }
        l_b *= scB; l_b2 *= scB2;

        float lincB = 0.0f, lincB2 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            float p0 = __expf(acc_sB[nt][0] - m_b);
            float p1 = __expf(acc_sB[nt][1] - m_b);
            float p2 = __expf(acc_sB[nt][2] - m_b2);
            float p3 = __expf(acc_sB[nt][3] - m_b2);
            lincB += p0 + p1; lincB2 += p2 + p3;
            unsigned int pr0 = warp_row_base + 16 + lane_id / 4;
            unsigned int pr1 = warp_row_base + 24 + lane_id / 4;
            unsigned int pc = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&smem_P[pr0 * p_stride + pc] = FLASH_FLOATS2HALF2(p0, p1);
            *(flash_half2_t*)&smem_P[pr1 * p_stride + pc] = FLASH_FLOATS2HALF2(p2, p3);
        }
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            lincB += __shfl_xor_sync(0xFFFFFFFF, lincB, off);
            lincB2 += __shfl_xor_sync(0xFFFFFFFF, lincB2, off);
        }
        l_b += lincB; l_b2 += lincB2;

        __syncthreads();
        FLASH_ASYNC_WAIT();
        __syncthreads();

        // Prefetch next K
        if (kv_block + 2 < num_kv_blocks) {
            unsigned int next_buf = (kv_block + 2) % TCGEN_K_STAGES;
            LOAD_KV_TILE_TCGEN(K_cache, block_table, (&SMEM_K_TCGEN(next_buf, 0, 0)),
                (kv_block + 2) * TCGEN_BC, kv_len, kv_head, tid, TCGEN_NUM_THREADS);
            FLASH_ASYNC_COMMIT();
        }

        // ============ PV: O += P × V for both halves ============
        {
            const unsigned int* sP32 = (const unsigned int*)smem_P;
            const unsigned int* sV32 = (const unsigned int*)smem_V;
            const unsigned int v_stride_u32 = TCGEN_HDIM_PAD / 2;
            const unsigned int p_stride_u32 = p_stride / 2;

            // Half A: P rows [warp_row_base..+15] × V[128×HDIM]
            #pragma unroll
            for (unsigned int k = 0; k < TCGEN_BC / 16; k++) {
                unsigned int prA0 = (warp_row_base + lane_id / 4) * p_stride_u32 + k * 8;
                unsigned int prA1 = (warp_row_base + 8 + lane_id / 4) * p_stride_u32 + k * 8;
                unsigned int pq_off = (lane_id % 4) * 2;
                unsigned int pA0 = sP32[prA0 + pq_off];
                unsigned int pA1 = sP32[prA0 + pq_off + 1];
                unsigned int pA2 = sP32[prA1 + pq_off];
                unsigned int pA3 = sP32[prA1 + pq_off + 1];

                #pragma unroll
                for (unsigned int nt = 0; nt < TCGEN_O_NTILES; nt++) {
                    unsigned int vr0 = (k * 16 + lane_id / 4) * v_stride_u32 + nt * 8;
                    unsigned int vq_off = (lane_id % 4) * 2;
                    unsigned int v0 = sV32[vr0 + vq_off];
                    unsigned int v1 = sV32[vr0 + vq_off + 1];
                    FLASH_MMA_K16(acc_oA[nt][0], acc_oA[nt][1], acc_oA[nt][2], acc_oA[nt][3],
                                  pA0, pA1, pA2, pA3, v0, v1,
                                  acc_oA[nt][0], acc_oA[nt][1], acc_oA[nt][2], acc_oA[nt][3]);
                }
            }

            // Half B: P rows [warp_row_base+16..+31] × V[128×HDIM]
            #pragma unroll
            for (unsigned int k = 0; k < TCGEN_BC / 16; k++) {
                unsigned int prB0 = (warp_row_base + 16 + lane_id / 4) * p_stride_u32 + k * 8;
                unsigned int prB1 = (warp_row_base + 24 + lane_id / 4) * p_stride_u32 + k * 8;
                unsigned int pq_off = (lane_id % 4) * 2;
                unsigned int pB0 = sP32[prB0 + pq_off];
                unsigned int pB1 = sP32[prB0 + pq_off + 1];
                unsigned int pB2 = sP32[prB1 + pq_off];
                unsigned int pB3 = sP32[prB1 + pq_off + 1];

                #pragma unroll
                for (unsigned int nt = 0; nt < TCGEN_O_NTILES; nt++) {
                    unsigned int vr0 = (k * 16 + lane_id / 4) * v_stride_u32 + nt * 8;
                    unsigned int vq_off = (lane_id % 4) * 2;
                    unsigned int v0 = sV32[vr0 + vq_off];
                    unsigned int v1 = sV32[vr0 + vq_off + 1];
                    FLASH_MMA_K16(acc_oB[nt][0], acc_oB[nt][1], acc_oB[nt][2], acc_oB[nt][3],
                                  pB0, pB1, pB2, pB3, v0, v1,
                                  acc_oB[nt][0], acc_oB[nt][1], acc_oB[nt][2], acc_oB[nt][3]);
                }
            }
        }

        // Wait for next K prefetch
        if (kv_block + 1 < num_kv_blocks) {
            FLASH_ASYNC_WAIT();
            __syncthreads();
        }
    } // end KV block loop

    // ============ Output: normalize and write both halves ============
    // Half A output
    float inv_lA = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
    float inv_lA2 = (l_a2 > 0.0f) ? (1.0f / l_a2) : 0.0f;
    unsigned int rowA0 = warp_row_base + lane_id / 4;
    unsigned int rowA1 = warp_row_base + 8 + lane_id / 4;
    if (q_start + rowA0 < q_len) {
        unsigned int out_base = (q_seq_start + q_start + rowA0) * q_seq_stride + q_head * head_dim;
        #pragma unroll
        for (unsigned int nt = 0; nt < TCGEN_O_NTILES; nt++) {
            unsigned int col = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&O[out_base + col] = FLASH_FLOATS2HALF2(acc_oA[nt][0] * inv_lA, acc_oA[nt][1] * inv_lA);
        }
    }
    if (q_start + rowA1 < q_len) {
        unsigned int out_base = (q_seq_start + q_start + rowA1) * q_seq_stride + q_head * head_dim;
        #pragma unroll
        for (unsigned int nt = 0; nt < TCGEN_O_NTILES; nt++) {
            unsigned int col = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&O[out_base + col] = FLASH_FLOATS2HALF2(acc_oA[nt][2] * inv_lA2, acc_oA[nt][3] * inv_lA2);
        }
    }
    // Half B output
    float inv_lB = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
    float inv_lB2 = (l_b2 > 0.0f) ? (1.0f / l_b2) : 0.0f;
    unsigned int rowB0 = warp_row_base + 16 + lane_id / 4;
    unsigned int rowB1 = warp_row_base + 24 + lane_id / 4;
    if (q_start + rowB0 < q_len) {
        unsigned int out_base = (q_seq_start + q_start + rowB0) * q_seq_stride + q_head * head_dim;
        #pragma unroll
        for (unsigned int nt = 0; nt < TCGEN_O_NTILES; nt++) {
            unsigned int col = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&O[out_base + col] = FLASH_FLOATS2HALF2(acc_oB[nt][0] * inv_lB, acc_oB[nt][1] * inv_lB);
        }
    }
    if (q_start + rowB1 < q_len) {
        unsigned int out_base = (q_seq_start + q_start + rowB1) * q_seq_stride + q_head * head_dim;
        #pragma unroll
        for (unsigned int nt = 0; nt < TCGEN_O_NTILES; nt++) {
            unsigned int col = nt * 8 + (lane_id % 4) * 2;
            *(flash_half2_t*)&O[out_base + col] = FLASH_FLOATS2HALF2(acc_oB[nt][2] * inv_lB2, acc_oB[nt][3] * inv_lB2);
        }
    }
}

#endif // FLASH_HDIM <= 256

#endif // FLASH_TCGEN05_ENABLED
