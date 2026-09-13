/**
 * NVFP4 paged KV prefill: software dequant (E2M1 LUT * E4M3 scale) + paged attention.
 *
 * Ragged multi-token prefill: each block handles ONE query token (q_local) for ONE
 * q_head of ONE sequence, attending over the causal KV prefix. The 4 warps split
 * the KV sequence (online softmax + inter-warp reduction).
 *
 * Paged layout (matching flash_nvfp4_kv_store):
 *   K_fp4: [num_blocks, block_size, num_kv_heads, head_dim/2] U8
 *   K_sf:  [num_blocks, block_size, num_kv_heads, head_dim/16] U8 (E4M3)
 *   V_fp4, V_sf: same shapes
 *
 * Q/O layout (ragged): [total_q, num_q_heads, head_dim]; cu_seqlens_q[seq] gives
 * the first query-token index of sequence `seq`.
 *
 * Query is WHT-rotated on-the-fly (if rotate=true) to match rotated K in cache.
 * V is unrotated.
 *
 * Grid: (max_q_len, num_q_heads, num_seqs)
 * Block: 128 threads (4 warps)
 */

#include "flash_sm_compat.cuh"
// wht_transform, get_sign_flip from flash_turboquant.cuh
// nvfp4_e2m1_to_float, e4m3_to_float_direct from flash_nvfp4_kv_store.cuh
// TQ4_NUM_WARPS, TQ4_BC from flash_turboquant_lowbit.cuh

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

#ifndef TQ4_NUM_WARPS
#define TQ4_NUM_WARPS 4
#endif
#ifndef TQ4_BC
#define TQ4_BC 8
#endif

#define NVDEC_VEC (HDIM / WARP_SIZE)
#define NVDEC_LANES_PER_GROUP (16 / NVDEC_VEC)
#define NVFP4_PREFILL_THREADS (TQ4_NUM_WARPS * WARP_SIZE)

template<typename HalfT>
__global__ void flash_nvfp4_kv_prefill(
    const HalfT* __restrict__ Q,
    const unsigned char* __restrict__ K_fp4,
    const unsigned char* __restrict__ K_sf,
    const unsigned char* __restrict__ V_fp4,
    const unsigned char* __restrict__ V_sf,
    HalfT* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int* __restrict__ cu_seqlens_q,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_seqs,
    const unsigned int q_stride,
    const unsigned int kv_stride,
    const float softcap,
    const unsigned int sliding_window,
    const bool rotate
) {
    const unsigned int q_local = blockIdx.x;   // query token index its sequence
    const unsigned int q_head  = blockIdx.y;   // 0..num_q_heads
    const unsigned int seq_idx = blockIdx.z;   // 0..num_seqs
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    const unsigned int n_rep = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / n_rep;

    // Ragged query offsets.
    const unsigned int q_seq_start = cu_seqlens_q[seq_idx];
    const unsigned int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
    if (q_local >= q_len) return;

    const unsigned int kv_len = (unsigned int)seq_lens[seq_idx];
    if (kv_len == 0) return;
    // Causal: query token q_local (at KV position q_local + q_offset) attends to
    // KV positions [0, q_local + q_offset].
    const unsigned int q_offset = (kv_len > q_len) ? (kv_len - q_len) : 0u;
    const unsigned int attend_limit = q_local + q_offset;
    const unsigned int kv_end = (kv_len < attend_limit + 1) ? kv_len : (attend_limit + 1);

    const unsigned int window_start =
        (sliding_window > 0 && kv_end > sliding_window) ? (kv_end - sliding_window) : 0u;

    // This warp's chunk of the KV sequence [window_start, kv_end).
    const unsigned int attended = kv_end - window_start;
    unsigned int chunk_size = (attended + TQ4_NUM_WARPS - 1) / TQ4_NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    // Load Q for this (q_local, q_head), apply WHT rotation.
    const unsigned int q_vec_off = lane_id * NVDEC_VEC;
    float q_reg[NVDEC_VEC];
    {
        const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)(q_seq_start + q_local) * q_stride
                                                            + (unsigned long long)q_head * head_dim + q_vec_off);
        #pragma unroll
        for (int i = 0; i < NVDEC_VEC / 2; i++) {
            unsigned int packed = __ldg(q32 + i);
            const HalfT* hp = reinterpret_cast<const HalfT*>(&packed);
            q_reg[2*i]   = FLASH_TO_FLOAT(hp[0]);
            q_reg[2*i+1] = FLASH_TO_FLOAT(hp[1]);
        }
        if (rotate) {
            #pragma unroll
            for (int i = 0; i < NVDEC_VEC; i++) {
                unsigned int ch = lane_id * NVDEC_VEC + i;
                q_reg[i] *= get_sign_flip(kv_head, ch);
            }
            wht_transform(q_reg, lane_id);
        }
    }

    // Paged strides
    const unsigned int hd_half = head_dim / 2;
    const unsigned int hd_groups = head_dim / 16;
    const unsigned long long fp4_page = (unsigned long long)block_size * num_kv_heads * hd_half;
    const unsigned long long sf_page  = (unsigned long long)block_size * num_kv_heads * hd_groups;
    const unsigned int lane_fp4_off = lane_id * (NVDEC_VEC / 2);
    const unsigned int lane_sf_off  = lane_id / NVDEC_LANES_PER_GROUP;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[NVDEC_VEC];
    #pragma unroll
    for (int i = 0; i < NVDEC_VEC; i++) o_reg[i] = 0.f;

    // Iterate over paged KV [my_start, my_end)
    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        const unsigned char* k_fp4_base = K_fp4 + (unsigned long long)physical_block * fp4_page
                                          + (unsigned long long)kv_head * hd_half;
        const unsigned char* k_sf_base  = K_sf  + (unsigned long long)physical_block * sf_page
                                          + (unsigned long long)kv_head * hd_groups;
        const unsigned char* v_fp4_base = V_fp4 + (unsigned long long)physical_block * fp4_page
                                          + (unsigned long long)kv_head * hd_half;
        const unsigned char* v_sf_base  = V_sf  + (unsigned long long)physical_block * sf_page
                                          + (unsigned long long)kv_head * hd_groups;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / TQ4_BC) * TQ4_BC;

        for (; processed < aligned; processed += TQ4_BC) {
            float scores[TQ4_BC];

            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                const unsigned char* kp = k_fp4_base + (unsigned long long)bo * num_kv_heads * hd_half + lane_fp4_off;
                const unsigned char* ks = k_sf_base  + (unsigned long long)bo * num_kv_heads * hd_groups + lane_sf_off;

                float k_scale = e4m3_to_float_direct(*ks);
                unsigned char packed0 = *kp;
                unsigned char p1 = *(kp + 1);
                float k0 = nvfp4_e2m1_to_float(packed0 & 0xF) * k_scale;
                float k1 = nvfp4_e2m1_to_float((packed0 >> 4) & 0xF) * k_scale;
                float k2 = nvfp4_e2m1_to_float(p1 & 0xF) * k_scale;
                float k3 = nvfp4_e2m1_to_float((p1 >> 4) & 0xF) * k_scale;

                float dot = q_reg[0]*k0 + q_reg[1]*k1 + q_reg[2]*k2 + q_reg[3]*k3;
                #pragma unroll
                for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, off);

                scores[b] = dot * inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            // Online softmax
            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) m_new = fmaxf(m_new, scores[b]);
            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < NVDEC_VEC; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            float exp_factors[TQ4_BC];
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l_val += exp_factors[b];
            }

            // PV accumulate: dequant V, multiply by attention weight
            #pragma unroll
            for (int b = 0; b < TQ4_BC; b++) {
                unsigned int bo = block_offset + processed + b;
                const unsigned char* vp = v_fp4_base + (unsigned long long)bo * num_kv_heads * hd_half + lane_fp4_off;
                const unsigned char* vs = v_sf_base  + (unsigned long long)bo * num_kv_heads * hd_groups + lane_sf_off;

                float v_scale = e4m3_to_float_direct(*vs);
                float w = exp_factors[b] * v_scale;
                unsigned char p0 = *vp;
                unsigned char p1 = *(vp + 1);
                o_reg[0] += w * nvfp4_e2m1_to_float(p0 & 0xF);
                o_reg[1] += w * nvfp4_e2m1_to_float((p0 >> 4) & 0xF);
                o_reg[2] += w * nvfp4_e2m1_to_float(p1 & 0xF);
                o_reg[3] += w * nvfp4_e2m1_to_float((p1 >> 4) & 0xF);
            }
            m_val = m_new;
        }

        // Remainder (positions not aligned to TQ4_BC)
        for (; processed < batch_count; processed++) {
            unsigned int bo = block_offset + processed;
            const unsigned char* kp = k_fp4_base + (unsigned long long)bo * num_kv_heads * hd_half + lane_fp4_off;
            const unsigned char* ks = k_sf_base  + (unsigned long long)bo * num_kv_heads * hd_groups + lane_sf_off;

            float k_scale = e4m3_to_float_direct(*ks);
            unsigned char p0 = *kp;
            unsigned char p1 = *(kp + 1);
            float k0 = nvfp4_e2m1_to_float(p0 & 0xF) * k_scale;
            float k1 = nvfp4_e2m1_to_float((p0 >> 4) & 0xF) * k_scale;
            float k2 = nvfp4_e2m1_to_float(p1 & 0xF) * k_scale;
            float k3 = nvfp4_e2m1_to_float((p1 >> 4) & 0xF) * k_scale;

            float dot = q_reg[0]*k0 + q_reg[1]*k1 + q_reg[2]*k2 + q_reg[3]*k3;
            #pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, off);

            float score = dot * inv_sqrt_d;
            if (softcap > 0.f) score = softcap * tanhf(score / softcap);

            float m_new = fmaxf(m_val, score);
            float exp_old = __expf(m_val - m_new), exp_new = __expf(score - m_new);
            l_val = l_val * exp_old + exp_new;
            #pragma unroll
            for (int i = 0; i < NVDEC_VEC; i++) o_reg[i] *= exp_old;

            const unsigned char* vp = v_fp4_base + (unsigned long long)bo * num_kv_heads * hd_half + lane_fp4_off;
            const unsigned char* vs = v_sf_base  + (unsigned long long)bo * num_kv_heads * hd_groups + lane_sf_off;
            float v_scale = e4m3_to_float_direct(*vs);
            float w = exp_new * v_scale;
            unsigned char vp0 = *vp;
            unsigned char vp1 = *(vp + 1);
            o_reg[0] += w * nvfp4_e2m1_to_float(vp0 & 0xF);
            o_reg[1] += w * nvfp4_e2m1_to_float((vp0 >> 4) & 0xF);
            o_reg[2] += w * nvfp4_e2m1_to_float(vp1 & 0xF);
            o_reg[3] += w * nvfp4_e2m1_to_float((vp1 >> 4) & 0xF);

            m_val = m_new;
        }

        pos += batch_count;
    }

    // Inter-warp reduction
    __shared__ float smem_m[TQ4_NUM_WARPS];
    __shared__ float smem_l[TQ4_NUM_WARPS];
    __shared__ float smem_o[TQ4_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < NVDEC_VEC; i++) smem_o[warp_id][q_vec_off + i] = o_reg[i];
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
                for (int i = 0; i < NVDEC_VEC; i++)
                    smem_o[warp_id][q_vec_off + i] =
                        smem_o[warp_id][q_vec_off + i] * scale_me +
                        smem_o[other][q_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.f) ? (1.f / final_l) : 0.f;
        HalfT* o_ptr = O + (unsigned long long)(q_seq_start + q_local) * q_stride
                        + (unsigned long long)q_head * head_dim
                        + q_vec_off;
        #pragma unroll
        for (int i = 0; i < NVDEC_VEC; i++) {
            o_ptr[i] = FLASH_FROM_FLOAT(smem_o[0][q_vec_off + i] * inv_l);
        }
    }
}