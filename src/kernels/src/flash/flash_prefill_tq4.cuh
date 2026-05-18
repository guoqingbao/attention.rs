// TurboQuant-4bit aware Flash Attention v2 Paged Prefill, SM80+.
//
// Same algorithm as flash_prefill_paged.cuh but reads K/V from 4-bit
// TurboQuant buffers instead of BF16 paged cache.
//
// K is stored as WHT(sign_flip(K_original)) in 4-bit. To compute correct
// Q·K_original, we apply sign_flip+WHT to Q (once per Q tile), so that:
//   Q_rot · K_stored = WHT(sign_flip(Q)) · WHT(sign_flip(K)) = Q · K  (WHT is unitary)
//
// V is stored as plain 4-bit (no rotation).
//
// K layout: K_quant [num_blocks, block_size, num_kv_heads, head_dim/2] U8
//           K_absmax [num_blocks, block_size, num_kv_heads] F32
// V layout: V_quant [num_blocks, block_size, num_kv_heads, head_dim/2] U8
//           V_absmax [num_blocks, block_size, num_kv_heads] F32

#include <cuda_bf16.h>

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif

#define TQ4P_BR 32
#define TQ4P_BC 32
#define TQ4P_HDIM FLASH_HDIM

#if TQ4P_HDIM <= 256
#define TQ4P_PAD_KV 8
#else
#define TQ4P_PAD_KV 0
#endif

#define TQ4P_HDIM_PAD (TQ4P_HDIM + TQ4P_PAD_KV)
#define TQ4P_PAD_P 8
#define TQ4P_N_TILES_PER_WARP ((TQ4P_HDIM / 8) / 2)
#define TQ4P_TILE_CHUNKS (TQ4P_BR * (TQ4P_HDIM / 8))
#define TQ4P_NUM_THREADS 128

// Load a KV tile from 4-bit TQ buffers, dequantize to BF16 in smem.
// Each token in the tile has absmax (f32) and packed nibbles (hd/2 bytes).
#define LOAD_TQ4_KV_TILE(absmax_buf, quant_buf, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _hd_half = TQ4P_HDIM / 2; \
        const unsigned int _cpr = TQ4P_HDIM / 8; \
        for (unsigned int _i = (t); _i < TQ4P_TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr; \
            unsigned int _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                unsigned long long _am_off = (unsigned long long)_pb * cache_block_size * num_kv_heads \
                    + (unsigned long long)_bo * num_kv_heads + (kvh); \
                float _scale = (absmax_buf)[_am_off]; \
                unsigned long long _q_base = (unsigned long long)_pb * cache_block_size * num_kv_heads * _hd_half \
                    + (unsigned long long)_bo * num_kv_heads * _hd_half \
                    + (unsigned long long)(kvh) * _hd_half; \
                const unsigned char* _qp = (quant_buf) + _q_base; \
                unsigned int _byte_off = _col / 2; \
                __nv_bfloat16 _tmp[8]; \
                for (int _b = 0; _b < 4; _b++) { \
                    unsigned char _packed = _qp[_byte_off + _b]; \
                    float _lo_val = (((float)(_packed & 0xF) - 7.5f) / 7.5f) * _scale; \
                    float _hi_val = (((float)(_packed >> 4) - 7.5f) / 7.5f) * _scale; \
                    _tmp[_b * 2]     = __float2bfloat16(_lo_val); \
                    _tmp[_b * 2 + 1] = __float2bfloat16(_hi_val); \
                } \
                *((uint4*)&(smem)[_row * TQ4P_HDIM_PAD + _col]) = *((uint4*)_tmp); \
            } else { \
                *((uint4*)&(smem)[_row * TQ4P_HDIM_PAD + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)


#if TQ4P_HDIM <= 256

#if TQ4P_HDIM > 128
#define TQ4P_USE_DYNAMIC_SMEM 1
#else
#define TQ4P_USE_DYNAMIC_SMEM 0
#endif

extern "C" __global__ void
#if TQ4P_USE_DYNAMIC_SMEM
__launch_bounds__(TQ4P_NUM_THREADS)
#endif
flash_tq4_prefill(
    const __nv_bfloat16* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
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
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * TQ4P_BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + TQ4P_BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

#if TQ4P_USE_DYNAMIC_SMEM
    extern __shared__ __align__(16) unsigned char tq4p_smem_dyn[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(tq4p_smem_dyn);
    __nv_bfloat16* smem_K_flat = smem_Q + TQ4P_BR * TQ4P_HDIM_PAD;
    __nv_bfloat16* smem_V = smem_K_flat + 2 * TQ4P_BC * TQ4P_HDIM_PAD;
    __nv_bfloat16* smem_P = smem_V + TQ4P_BC * TQ4P_HDIM_PAD;
    float* smem_ml = reinterpret_cast<float*>(smem_P + TQ4P_BR * (TQ4P_BC + TQ4P_PAD_P));
    #define SMEM_K_TQ4P(buf, idx) smem_K_flat[(buf) * TQ4P_BC * TQ4P_HDIM_PAD + (idx)]
#else
    __shared__ __nv_bfloat16 smem_Q[TQ4P_BR * TQ4P_HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K_arr[2][TQ4P_BC * TQ4P_HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[TQ4P_BC * TQ4P_HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[TQ4P_BR * (TQ4P_BC + TQ4P_PAD_P)];
    __shared__ float smem_ml[TQ4P_BR * 2];
    #define SMEM_K_TQ4P(buf, idx) smem_K_arr[buf][idx]
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * TQ4P_N_TILES_PER_WARP;
    const unsigned int p_stride = TQ4P_BC + TQ4P_PAD_P;

    float acc_o[TQ4P_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < TQ4P_N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f;
        acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + TQ4P_BC - 1) / TQ4P_BC;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / TQ4P_BC;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }

    // Load Q (BF16 from model output, cp.async)
    {
        const unsigned int cpr = TQ4P_HDIM / 8;
        for (unsigned int idx = tid; idx < TQ4P_TILE_CHUNKS; idx += TQ4P_NUM_THREADS) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * TQ4P_HDIM_PAD + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm));
            } else {
                *((uint4*)&smem_Q[row * TQ4P_HDIM_PAD + col]) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    // Apply sign_flip + WHT to Q rows so Q_rot · K_stored = Q · K_original.
    // Each warp transforms TQ4P_BR/4 rows (8 rows each for 4 warps).
    {
        const unsigned int tq_vec = TQ4P_HDIM / WARP_SIZE;
        for (unsigned int row = warp_id; row < TQ4P_BR; row += 4) {
            if (q_start + row >= q_len) continue;
            float qr[TQ4P_HDIM / WARP_SIZE];
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                qr[i] = __bfloat162float(smem_Q[row * TQ4P_HDIM_PAD + ch]);
                qr[i] *= get_sign_flip(kv_head, ch);
            }
            wht_transform(qr, lane_id);
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                smem_Q[row * TQ4P_HDIM_PAD + ch] = __float2bfloat16(qr[i]);
            }
        }
    }
    __syncthreads();

    // Load K[0] from TQ4 buffers
    if (num_kv_blocks > 0) {
        LOAD_TQ4_KV_TILE(K_absmax, K_quant, block_table, (&SMEM_K_TQ4P(0, 0)), 0, kv_len, kv_head, tid, TQ4P_NUM_THREADS);
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * TQ4P_BC;
        unsigned int kv_end = min(kv_start + TQ4P_BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // Load V from TQ4 (overlaps with QK^T conceptually, but no cp.async)
        LOAD_TQ4_KV_TILE(V_absmax, V_quant, block_table, smem_V, kv_start, kv_len, kv_head, tid, TQ4P_NUM_THREADS);
        __syncthreads();

        // QK^T (warps 0-1)
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.f; acc_s[i][1] = 0.f;
                acc_s[i][2] = 0.f; acc_s[i][3] = 0.f;
            }

            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)(&SMEM_K_TQ4P(buf, 0));

            #pragma unroll
            for (unsigned int ks = 0; ks < (TQ4P_HDIM / 16); ks++) {
                unsigned int kb = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int ac0 = kb + tid_in_group * 2, ac1 = ac0 + 8;

                unsigned int sa_q0 = __cvta_generic_to_shared(&sQ[(ar0 * TQ4P_HDIM_PAD + ac0)]);
                unsigned int sa_q1 = __cvta_generic_to_shared(&sQ[(ar1 * TQ4P_HDIM_PAD + ac0)]);
                unsigned int sa_q2 = __cvta_generic_to_shared(&sQ[(ar0 * TQ4P_HDIM_PAD + ac1)]);
                unsigned int sa_q3 = __cvta_generic_to_shared(&sQ[(ar1 * TQ4P_HDIM_PAD + ac1)]);

                unsigned int a0, a1, a2, a3;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(a0) : "r"(sa_q0));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(a1) : "r"(sa_q1));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(a2) : "r"(sa_q2));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(a3) : "r"(sa_q3));

                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int nc = nt * 8 + group_id;
                    unsigned int k0 = kb + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sK[nc * TQ4P_HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK[nc * TQ4P_HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK[nc * TQ4P_HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK[nc * TQ4P_HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3]));
                }
            }

            // Scale, softcap, mask, softmax
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
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
                m_r0 = mn0;
            }
            float mn1 = fmaxf(m_r1, rmax1);
            if (mn1 != m_r1) {
                float eo1 = __expf(m_r1 - mn1); l_r1 *= eo1;
                #pragma unroll
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
                m_r1 = mn1;
            }

            float sum0 = 0.f, sum1 = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01; sum1 += p10 + p11;
                unsigned int c0 = nt * 8 + tid_in_group * 2;
                smem_P[row0 * p_stride + c0]     = __float2bfloat16(p00);
                smem_P[row0 * p_stride + c0 + 1] = __float2bfloat16(p01);
                smem_P[row1 * p_stride + c0]     = __float2bfloat16(p10);
                smem_P[row1 * p_stride + c0 + 1] = __float2bfloat16(p11);
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
        }
        __syncthreads();

        // Warps 2-3: rescale accumulators
        if (warp_id >= 2) {
            unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8;
            float cm0 = smem_ml[r0 * 2], cm1 = smem_ml[r1 * 2];
            if (cm0 != m_r0) {
                float er0 = __expf(m_r0 - cm0);
                #pragma unroll
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP; i++) { acc_o[i][0] *= er0; acc_o[i][1] *= er0; }
                m_r0 = cm0;
            }
            if (cm1 != m_r1) {
                float er1 = __expf(m_r1 - cm1);
                #pragma unroll
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP; i++) { acc_o[i][2] *= er1; acc_o[i][3] *= er1; }
                m_r1 = cm1;
            }
        }

        // Preload K[i+1] from TQ4
        if (kv_block + 1 < num_kv_blocks) {
            LOAD_TQ4_KV_TILE(K_absmax, K_quant, block_table, (&SMEM_K_TQ4P(1 - buf, 0)),
                (kv_block + 1) * TQ4P_BC, kv_len, kv_head, tid, TQ4P_NUM_THREADS);
        }

        // PV MMA (all 4 warps)
        {
            const unsigned short* sP = (const unsigned short*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_V;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int ko = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int ac0 = ko + tid_in_group * 2, ac1 = ac0 + 8;
                unsigned int a0 = *(const unsigned int*)&sP[ar0 * p_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * p_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * p_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * p_stride + ac1];
                #pragma unroll
                for (int nt = 0; nt < TQ4P_N_TILES_PER_WARP; nt++) {
                    unsigned int nc = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = ko + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * TQ4P_HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k0 * TQ4P_HDIM_PAD + nc];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * TQ4P_HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k1 * TQ4P_HDIM_PAD + nc];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3]));
                }
            }
        }

        __syncthreads();
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

        __nv_bfloat16* ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < TQ4P_N_TILES_PER_WARP; nt++) {
            unsigned int c0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + r0, gr1 = q_start + r1;
            if (gr0 < q_len && r0 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * il0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * il0));
                *(unsigned int*)&ob[gr0 * q_seq_stride + c0] = lo | (hi << 16);
            }
            if (gr1 < q_len && r1 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * il1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * il1));
                *(unsigned int*)&ob[gr1 * q_seq_stride + c0] = lo | (hi << 16);
            }
        }
    }
}

#undef SMEM_K_TQ4P
#undef TQ4P_USE_DYNAMIC_SMEM

#else // TQ4P_HDIM == 512

// HDIM=512 variant with dynamic shared memory
#define TQ4P_BR_512 32
#define TQ4P_BC_512 32
#define TQ4P_PAD_P_512 8
#define TQ4P_N_TILES_PER_WARP_512 16
#define TQ4P_TILE_CHUNKS_512 (TQ4P_BR_512 * (512 / 8))
#define TQ4P_NUM_THREADS_512 256

#define LOAD_TQ4_KV_512(absmax_buf, quant_buf, bt, dst, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _hd_half = 256; \
        const unsigned int _cpr = 64; \
        for (unsigned int _i = (t); _i < TQ4P_TILE_CHUNKS_512; _i += (stride)) { \
            unsigned int _row = _i / _cpr; \
            unsigned int _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                unsigned long long _am_off = (unsigned long long)_pb * cache_block_size * num_kv_heads \
                    + (unsigned long long)_bo * num_kv_heads + (kvh); \
                float _scale = (absmax_buf)[_am_off]; \
                unsigned long long _q_base = (unsigned long long)_pb * cache_block_size * num_kv_heads * _hd_half \
                    + (unsigned long long)_bo * num_kv_heads * _hd_half \
                    + (unsigned long long)(kvh) * _hd_half; \
                const unsigned char* _qp = (quant_buf) + _q_base; \
                unsigned int _byte_off = _col / 2; \
                __nv_bfloat16 _tmp[8]; \
                for (int _b = 0; _b < 4; _b++) { \
                    unsigned char _packed = _qp[_byte_off + _b]; \
                    float _lo_val = (((float)(_packed & 0xF) - 7.5f) / 7.5f) * _scale; \
                    float _hi_val = (((float)(_packed >> 4) - 7.5f) / 7.5f) * _scale; \
                    _tmp[_b * 2]     = __float2bfloat16(_lo_val); \
                    _tmp[_b * 2 + 1] = __float2bfloat16(_hi_val); \
                } \
                *((uint4*)&(dst)[_row * 512 + _col]) = *((uint4*)_tmp); \
            } else { *((uint4*)&(dst)[_row * 512 + _col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

extern "C" __global__ void flash_tq4_prefill(
    const __nv_bfloat16* __restrict__ Q,
    const float* __restrict__ K_absmax,
    const unsigned char* __restrict__ K_quant,
    const float* __restrict__ V_absmax,
    const unsigned char* __restrict__ V_quant,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
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
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * TQ4P_BR_512;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + TQ4P_BR_512, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    extern __shared__ __align__(16) unsigned char smem_dyn[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_dyn);
    __nv_bfloat16* smem_K = smem_Q + TQ4P_BR_512 * 512;
    __nv_bfloat16* smem_V = smem_K + TQ4P_BC_512 * 512;
    __nv_bfloat16* smem_P = smem_V + TQ4P_BC_512 * 512;
    float* smem_ml = reinterpret_cast<float*>(smem_P + TQ4P_BR_512 * (TQ4P_BC_512 + TQ4P_PAD_P_512));

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * TQ4P_N_TILES_PER_WARP_512;
    const unsigned int p_stride_512 = TQ4P_BC_512 + TQ4P_PAD_P_512;

    float acc_o[TQ4P_N_TILES_PER_WARP_512][4];
    #pragma unroll
    for (int i = 0; i < TQ4P_N_TILES_PER_WARP_512; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f;
        acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + TQ4P_BC_512 - 1) / TQ4P_BC_512;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / TQ4P_BC_512;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }

    // Load Q (cp.async)
    {
        const unsigned int cpr = 512 / 8;
        for (unsigned int idx = tid; idx < TQ4P_TILE_CHUNKS_512; idx += TQ4P_NUM_THREADS_512) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * 512 + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm));
            } else {
                *((uint4*)&smem_Q[row * 512 + col]) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    // Apply sign_flip + WHT to Q rows (8 warps can handle 8 rows at a time)
    {
        const unsigned int tq_vec = 512 / WARP_SIZE;
        for (unsigned int row = warp_id; row < TQ4P_BR_512; row += 8) {
            if (q_start + row >= q_len) continue;
            float qr[512 / WARP_SIZE];
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                qr[i] = __bfloat162float(smem_Q[row * 512 + ch]);
                qr[i] *= get_sign_flip(kv_head, ch);
            }
            wht_transform(qr, lane_id);
            #pragma unroll
            for (unsigned int i = 0; i < tq_vec; i++) {
                unsigned int ch = lane_id * tq_vec + i;
                smem_Q[row * 512 + ch] = __float2bfloat16(qr[i]);
            }
        }
    }
    __syncthreads();

    if (num_kv_blocks > 0) {
        LOAD_TQ4_KV_512(K_absmax, K_quant, block_table, smem_K, 0, kv_len, kv_head, tid, TQ4P_NUM_THREADS_512);
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * TQ4P_BC_512;
        unsigned int kv_end = min(kv_start + TQ4P_BC_512, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }
            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K;

            #pragma unroll
            for (unsigned int ks = 0; ks < (512/16); ks++) {
                unsigned int kb = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int ac0 = kb + tid_in_group * 2, ac1 = ac0 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ[ar0 * 512 + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ[ar1 * 512 + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ[ar0 * 512 + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ[ar1 * 512 + ac1];
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int nc = nt * 8 + group_id;
                    unsigned int k0 = kb + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sK[nc * 512 + k0 + 1] << 16) |
                                      (unsigned int)sK[nc * 512 + k0];
                    unsigned int b1 = ((unsigned int)sK[nc * 512 + k1 + 1] << 16) |
                                      (unsigned int)sK[nc * 512 + k1];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3]));
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
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP_512; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
                m_r0 = mn0;
            }
            float mn1 = fmaxf(m_r1, rmax1);
            if (mn1 != m_r1) {
                float eo1 = __expf(m_r1 - mn1); l_r1 *= eo1;
                #pragma unroll
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP_512; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
                m_r1 = mn1;
            }

            float sum0 = 0.f, sum1 = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01; sum1 += p10 + p11;
                unsigned int c0 = nt * 8 + tid_in_group * 2;
                smem_P[row0 * p_stride_512 + c0]     = __float2bfloat16(p00);
                smem_P[row0 * p_stride_512 + c0 + 1] = __float2bfloat16(p01);
                smem_P[row1 * p_stride_512 + c0]     = __float2bfloat16(p10);
                smem_P[row1 * p_stride_512 + c0 + 1] = __float2bfloat16(p11);
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
            asm volatile("cp.async.commit_group;");
        } else {
            LOAD_TQ4_KV_512(V_absmax, V_quant, block_table, smem_V, kv_start, kv_len, kv_head, tid - 64, 192);
            asm volatile("cp.async.commit_group;");
        }

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        if (warp_id >= 2) {
            unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8;
            float cm0 = smem_ml[r0 * 2], cm1 = smem_ml[r1 * 2];
            if (cm0 != m_r0) {
                float er0 = __expf(m_r0 - cm0);
                #pragma unroll
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP_512; i++) { acc_o[i][0] *= er0; acc_o[i][1] *= er0; }
                m_r0 = cm0;
            }
            if (cm1 != m_r1) {
                float er1 = __expf(m_r1 - cm1);
                #pragma unroll
                for (int i = 0; i < TQ4P_N_TILES_PER_WARP_512; i++) { acc_o[i][2] *= er1; acc_o[i][3] *= er1; }
                m_r1 = cm1;
            }
        }

        // PV MMA (all 8 warps)
        {
            const unsigned short* sP = (const unsigned short*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_V;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int ko = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int ac0 = ko + tid_in_group * 2, ac1 = ac0 + 8;
                unsigned int a0 = *(const unsigned int*)&sP[ar0 * p_stride_512 + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * p_stride_512 + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * p_stride_512 + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * p_stride_512 + ac1];
                #pragma unroll
                for (int nt = 0; nt < TQ4P_N_TILES_PER_WARP_512; nt++) {
                    unsigned int nc = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = ko + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * 512 + nc] << 16) |
                                      (unsigned int)sV[k0 * 512 + nc];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * 512 + nc] << 16) |
                                      (unsigned int)sV[k1 * 512 + nc];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3]));
                }
            }
        }

        __syncthreads();

        if (kv_block + 1 < num_kv_blocks) {
            LOAD_TQ4_KV_512(K_absmax, K_quant, block_table, smem_K, (kv_block + 1) * TQ4P_BC_512, kv_len, kv_head, tid, TQ4P_NUM_THREADS_512);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
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

        __nv_bfloat16* ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < TQ4P_N_TILES_PER_WARP_512; nt++) {
            unsigned int c0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + r0, gr1 = q_start + r1;
            if (gr0 < q_len && r0 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * il0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * il0));
                *(unsigned int*)&ob[gr0 * q_seq_stride + c0] = lo | (hi << 16);
            }
            if (gr1 < q_len && r1 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * il1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * il1));
                *(unsigned int*)&ob[gr1 * q_seq_stride + c0] = lo | (hi << 16);
            }
        }
    }
}

#undef LOAD_TQ4_KV_512
#undef TQ4P_BR_512
#undef TQ4P_BC_512
#undef TQ4P_PAD_P_512
#undef TQ4P_N_TILES_PER_WARP_512
#undef TQ4P_TILE_CHUNKS_512
#undef TQ4P_NUM_THREADS_512

#endif // TQ4P_HDIM > 256

#undef LOAD_TQ4_KV_TILE
#undef TQ4P_BR
#undef TQ4P_BC
#undef TQ4P_HDIM
#undef TQ4P_PAD_KV
#undef TQ4P_HDIM_PAD
#undef TQ4P_PAD_P
#undef TQ4P_N_TILES_PER_WARP
#undef TQ4P_TILE_CHUNKS
#undef TQ4P_NUM_THREADS
