/**
 * FP8 paged decode with WHT rotation on Q.
 *
 * Pairs with flash_fp8_rot_store: K in cache is pre-rotated,
 * V is unrotated. This kernel rotates Q on-the-fly so that
 * (H Q) . (H K)^T = Q . K^T (orthogonality).
 *
 * k_scale and v_scale are the c_k*absmax and c_v*absmax values
 * stored by flash_fp8_rot_store. The decode kernel does not
 * need to know about c separately.
 *
 * rotate=false reproduces the existing flash_decode_paged_fp8.
 */

#include "flash_sm_compat.cuh"
// wht_transform, get_sign_flip, fp8x4_to_f32x4, VEC_FP8, VEC_BF16,
// BC, NUM_WARPS already included via flash_turboquant.cuh and
// flash_decode_paged_fp8.cuh in flash_instantiate.cu before this file

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

template<typename HalfT>
__global__ void flash_fp8_rot_decode(
    const HalfT* __restrict__ Q,
    const void* __restrict__ K_cache,
    const void* __restrict__ V_cache,
    HalfT* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const float softcap,
    const unsigned int sliding_window,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr,
    const unsigned long long fp8_cache_stride,
    const bool rotate
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const float k_scale = k_scale_ptr[kv_head];
    const float v_scale = v_scale_ptr[kv_head];

    // Load Q and optionally rotate
    const unsigned int bf16_vec_off = lane_id * VEC_BF16;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[VEC_BF16];
    {
        unsigned int qp[VEC_U32];
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) qp[i] = __ldg(q32 + i);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++)
            unpack2_bf16_d<HalfT>(qp[i], q_reg[2*i], q_reg[2*i+1]);
    }

    if (rotate) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            unsigned int ch = lane_id * VEC_BF16 + i;
            q_reg[i] *= get_sign_flip(kv_head, ch);
        }
        wht_transform(q_reg, lane_id);
    }

    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.f;

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const unsigned int vec_offset = lane_id * VEC_FP8;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset_start = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned int remaining_in_block = block_size - block_offset_start;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = (remaining_in_block < remaining_total) ? remaining_in_block : remaining_total;

        unsigned long long page_base = (unsigned long long)physical_block * page_stride
            + (unsigned long long)kv_head * head_dim + vec_offset;

        unsigned int processed = 0;
        unsigned int aligned = (batch_count / BC) * BC;

        for (; processed < aligned; processed += BC) {
            float scores[BC];
            unsigned long long kv_off[BC];

            #pragma unroll
            for (int b = 0; b < BC; b++)
                kv_off[b] = page_base + (unsigned long long)(block_offset_start + processed + b) * head_stride_kv;

            float dots[BC] = {0.f};
            #pragma unroll
            for (int g = 0; g < VEC_FP8 / 4; g++) {
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    const unsigned int* k32 = (const unsigned int*)((const __nv_fp8_storage_t*)K_cache + kv_off[b]);
                    float kf0, kf1, kf2, kf3;
                    fp8x4_to_f32x4(__ldg(k32 + g), k_scale, kf0, kf1, kf2, kf3);
                    dots[b] += q_reg[g*4]*kf0 + q_reg[g*4+1]*kf1 + q_reg[g*4+2]*kf2 + q_reg[g*4+3]*kf3;
                }
            }
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                #pragma unroll
                for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                    dots[b] += __shfl_xor_sync(0xffffffff, dots[b], off);
                scores[b] = dots[b] * inv_sqrt_d;
                if (softcap > 0.f) scores[b] = softcap * tanhf(scores[b] / softcap);
            }

            float m_new = m_val;
            #pragma unroll
            for (int b = 0; b < BC; b++) m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m_val - m_new);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
            l_val *= exp_old;

            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float w = __expf(scores[b] - m_new);
                l_val += w;
                const unsigned int* v32 = (const unsigned int*)((const __nv_fp8_storage_t*)V_cache + kv_off[b]);
                float ws = w * v_scale;
                #pragma unroll
                for (int g = 0; g < VEC_FP8 / 4; g++) {
                    float vf0, vf1, vf2, vf3;
                    fp8x4_to_f32x4(__ldg(v32 + g), 1.0f, vf0, vf1, vf2, vf3);
                    o_reg[g*4]   += ws * vf0;
                    o_reg[g*4+1] += ws * vf1;
                    o_reg[g*4+2] += ws * vf2;
                    o_reg[g*4+3] += ws * vf3;
                }
            }
            m_val = m_new;
        }

        for (; processed < batch_count; processed++) {
            unsigned long long kv_base = page_base + (unsigned long long)(block_offset_start + processed) * head_stride_kv;
            const unsigned int* k32 = (const unsigned int*)((const __nv_fp8_storage_t*)K_cache + kv_base);
            float dot = 0.f;
            #pragma unroll
            for (int g = 0; g < VEC_FP8 / 4; g++) {
                float kf0, kf1, kf2, kf3;
                fp8x4_to_f32x4(__ldg(k32 + g), k_scale, kf0, kf1, kf2, kf3);
                dot += q_reg[g*4]*kf0 + q_reg[g*4+1]*kf1 + q_reg[g*4+2]*kf2 + q_reg[g*4+3]*kf3;
            }
            #pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, off);
            float score = dot * inv_sqrt_d;
            if (softcap > 0.f) score = softcap * tanhf(score / softcap);

            float m_new = fmaxf(m_val, score);
            float exp_old = __expf(m_val - m_new), exp_new = __expf(score - m_new);
            l_val = l_val * exp_old + exp_new;
            const unsigned int* v32 = (const unsigned int*)((const __nv_fp8_storage_t*)V_cache + kv_base);
            float ew_vs = exp_new * v_scale;
            #pragma unroll
            for (int g = 0; g < VEC_FP8 / 4; g++) {
                float vf0, vf1, vf2, vf3;
                fp8x4_to_f32x4(__ldg(v32 + g), 1.0f, vf0, vf1, vf2, vf3);
                o_reg[g*4]   = o_reg[g*4]   * exp_old + ew_vs * vf0;
                o_reg[g*4+1] = o_reg[g*4+1] * exp_old + ew_vs * vf1;
                o_reg[g*4+2] = o_reg[g*4+2] * exp_old + ew_vs * vf2;
                o_reg[g*4+3] = o_reg[g*4+3] * exp_old + ew_vs * vf3;
            }
            m_val = m_new;
        }

        pos += batch_count;
    }

    // Inter-warp reduction
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) { smem_m[warp_id] = m_val; smem_l[warp_id] = l_val; }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) smem_o[warp_id][bf16_vec_off + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS/2; stride > 0; stride >>= 1) {
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
                for (int i = 0; i < VEC_BF16; i++)
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
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + bf16_vec_off);
        #pragma unroll
        for (int i = 0; i < VEC_BF16 / 2; i++) {
            float v0 = smem_o[0][bf16_vec_off + 2*i]     * inv_l;
            float v1 = smem_o[0][bf16_vec_off + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)FLASH_AS_USHORT(FLASH_FROM_FLOAT(v0));
            unsigned int hi = (unsigned int)FLASH_AS_USHORT(FLASH_FROM_FLOAT(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}