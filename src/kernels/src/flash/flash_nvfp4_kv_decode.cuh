/**
 * NVFP4 paged KV decode: software dequant (E2M1 LUT * E4M3 scale) + paged attention.
 *
 * Proven SM120 pattern (hikarioyama/vllm-nvfp4-kv-sm120):
 *   - Dequantize FP4 E2M1 codes to BF16 in registers using E4M3 group-16 scales
 *   - Compute attention with standard paged iteration + online softmax
 *   - 91-100% of FP8 decode throughput, 1.78x KV capacity
 *
 * Paged layout (matching flash_nvfp4_kv_store):
 *   K_fp4: [num_blocks, block_size, num_kv_heads, head_dim/2] U8 (packed FP4)
 *   K_sf:  [num_blocks, block_size, num_kv_heads, head_dim/16] U8 (E4M3 scales)
 *   V_fp4, V_sf: same shapes
 *
 * Lane-to-data mapping (HDIM=128, NVDEC_VEC=4):
 *   Lane L holds elements [L*4, L*4+3]
 *   Group G = L/4 covers elements [G*16, (G+1)*16)
 *   FP4 byte offset for lane L: L*2 (2 bytes = 4 codes)
 *   SF byte offset for group G: G (1 E4M3 scale per 16 elements)
 *
 * Q is WHT-rotated on-the-fly (if rotate=true) to match rotated K in cache.
 * V is unrotated.
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
#define TQ4_NUM_WARPS 8
#endif
#ifndef TQ4_BC
#define TQ4_BC 8
#endif

#define NVDEC_VEC (HDIM / WARP_SIZE)
// Each lane holds NVDEC_VEC elements. Group of 16 is handled by 16/NVDEC_VEC lanes.
#define NVDEC_LANES_PER_GROUP (16 / NVDEC_VEC)
// Lane L's group index: L / NVDEC_LANES_PER_GROUP
// Lane L's FP4 byte offset within its group: (L % NVDEC_LANES_PER_GROUP) * (NVDEC_VEC / 2)
// Lane L's SF byte offset: L / NVDEC_LANES_PER_GROUP

template<typename HalfT>
__global__ void flash_nvfp4_kv_decode(
    const HalfT* __restrict__ Q,
    const unsigned char* __restrict__ K_fp4,
    const unsigned char* __restrict__ K_sf,
    const unsigned char* __restrict__ V_fp4,
    const unsigned char* __restrict__ V_sf,
    HalfT* __restrict__ O,
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
    const unsigned int sliding_window,
    const bool rotate
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

    // Load Q, apply WHT rotation
    const unsigned int bf16_vec_off = lane_id * NVDEC_VEC;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[NVDEC_VEC];
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

    // Paged KV iteration
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + TQ4_NUM_WARPS - 1) / TQ4_NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[NVDEC_VEC];
    #pragma unroll
    for (int i = 0; i < NVDEC_VEC; i++) o_reg[i] = 0.f;

    // Paged strides (byte offsets)
    const unsigned int hd_half = head_dim / 2;
    const unsigned int hd_groups = head_dim / 16;
    const unsigned long long fp4_page = (unsigned long long)block_size * num_kv_heads * hd_half;
    const unsigned long long sf_page  = (unsigned long long)block_size * num_kv_heads * hd_groups;

    // This lane's fixed offsets within a (position, head) row
    const unsigned int lane_fp4_off = lane_id * (NVDEC_VEC / 2);  // byte offset in FP4 row
    const unsigned int lane_sf_off  = lane_id / NVDEC_LANES_PER_GROUP;  // group index = SF byte offset

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        // Base for for this physical page + kv_head
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
                // Per-position byte offsets
                const unsigned char* kp = k_fp4_base + (unsigned long long)bo * num_kv_heads * hd_half + lane_fp4_off;
                const unsigned char* ks = k_sf_base  + (unsigned long long)bo * num_kv_heads * hd_groups + lane_sf_off;

                // Dequant K: 2 bytes of FP4 = 4 codes, 1 byte of E4M3 scale
                float k_scale = e4m3_to_float_direct(*ks);
                unsigned char packed0 = *kp;
                unsigned char p1 = *(kp + 1);
                float k0 = nvfp4_e2m1_to_float(packed0 & 0xF) * k_scale;
                float k1 = nvfp4_e2m1_to_float((packed0 >> 4) & 0xF) * k_scale;
                float k2 = nvfp4_e2m1_to_float(p1 & 0xF) * k_scale;
                float k3 = nvfp4_e2m1_to_float((p1 >> 4) & 0xF) * k_scale;

                float dot = q_reg[0]*k0 + q_reg[1]*k1 + q_reg[2]*k2 + q_reg[3]*k3;

                // Full warp reduction (all 32 lanes contribute to the dot)
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

            // V accumulate
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

    // Inter-warp reduction (same pattern as TQ4 decode)
    __shared__ float smem_m[TQ4_NUM_WARPS];
    __shared__ float smem_l[TQ4_NUM_WARPS];
    __shared__ float smem_o[TQ4_NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < NVDEC_VEC; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
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
        HalfT* o_ptr = O + (unsigned long long)seq_idx * num_q_heads * head_dim
                       + (unsigned long long)q_head * head_dim + bf16_vec_off;
        #pragma unroll
        for (int i = 0; i < NVDEC_VEC; i++) {
            o_ptr[i] = FLASH_FROM_FLOAT(smem_o[0][bf16_vec_off + i] * inv_l);
        }
    }
}