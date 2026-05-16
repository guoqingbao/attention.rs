// Native Flash Decode Attention — Paged FP8 E4M3 KV cache, SM80+.
//
// Same algorithm as flash_decode_paged.cu but KV stored as FP8 E4M3.
// Dequantizes to float32 on-the-fly in registers.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif
#ifndef VEC_BF16
#define VEC_BF16 (HDIM / WARP_SIZE)
#endif
#ifndef VEC_U32
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#endif
#ifndef VEC_FP8
#define VEC_FP8  (HDIM / WARP_SIZE)
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif
#ifndef BC
#define BC 4
#endif

// fp8_to_f32_d is unique to this file.
// unpack2_bf16_d is already defined in flash_decode_paged.cu (included before us
// in flash_instantiate.cu) with the same per-HDIM rename.

__device__ __forceinline__ float fp8_to_f32_d(__nv_fp8_storage_t b, float scale) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
}

extern "C" __global__ void flash_decode_paged_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const void* __restrict__ K_cache,
    const void* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int sliding_window,
    const float softcap,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr,
    const unsigned long long fp8_cache_stride
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
    const unsigned int vec_offset = lane_id * VEC_FP8;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    const unsigned int bf16_vec_off = lane_id * VEC_BF16;
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) unpack2_bf16_d(q32[i], q_reg[2*i], q_reg[2*i+1]);

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

    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const __nv_fp8_storage_t* k_ptr = (const __nv_fp8_storage_t*)K_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset;

        float dot = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC_FP8; i++) {
            float kv = fp8_to_f32_d(k_ptr[i], k_scale);
            dot += q_reg[i] * kv;
        }
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        float score = dot * inv_sqrt_d;
        if (softcap > 0.f) score = softcap * tanhf(score / softcap);

        float m_new = fmaxf(m_val, score);
        float exp_old = __expf(m_val - m_new), exp_new = __expf(score - m_new);
        l_val = l_val * exp_old + exp_new;

        const __nv_fp8_storage_t* v_ptr = (const __nv_fp8_storage_t*)V_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset;
        #pragma unroll
        for (int i = 0; i < VEC_FP8; i++) {
            float vv = fp8_to_f32_d(v_ptr[i], v_scale);
            o_reg[i] = o_reg[i] * exp_old + exp_new * vv;
        }
        m_val = m_new;
    }

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
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new), scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
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
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][bf16_vec_off + 2*i]     * inv_l;
            float v1 = smem_o[0][bf16_vec_off + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// Split-K variant
extern "C" __global__ void flash_decode_paged_splitk_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const void* __restrict__ K_cache,
    const void* __restrict__ V_cache,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const float softcap,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr,
    const unsigned long long fp8_cache_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const float k_scale = k_scale_ptr[kv_head];
    const float v_scale = v_scale_ptr[kv_head];
    const unsigned int vec_offset = lane_id * VEC_FP8;
    const unsigned int bf16_vec_off = lane_id * VEC_BF16;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + bf16_vec_off);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) unpack2_bf16_d(q32[i], q_reg[2*i], q_reg[2*i+1]);

    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f, l_val = 0.f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.f;

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;

    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const __nv_fp8_storage_t* k_ptr = (const __nv_fp8_storage_t*)K_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset;

        float dot = 0.f;
        #pragma unroll
        for (int i = 0; i < VEC_FP8; i++) dot += q_reg[i] * fp8_to_f32_d(k_ptr[i], k_scale);
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        float score = dot * inv_sqrt_d;
        if (softcap > 0.f) score = softcap * tanhf(score / softcap);

        float m_new = fmaxf(m_val, score);
        float exp_old = __expf(m_val - m_new), exp_new = __expf(score - m_new);
        l_val = l_val * exp_old + exp_new;

        const __nv_fp8_storage_t* v_ptr = (const __nv_fp8_storage_t*)V_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset;
        #pragma unroll
        for (int i = 0; i < VEC_FP8; i++) {
            o_reg[i] = o_reg[i] * exp_old + exp_new * fp8_to_f32_d(v_ptr[i], v_scale);
        }
        m_val = m_new;
    }

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
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new), scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    smem_o[warp_id][bf16_vec_off + i] =
                        smem_o[warp_id][bf16_vec_off + i] * scale_me +
                        smem_o[other][bf16_vec_off + i] * scale_w;
            }
        }
        __syncthreads();
    }

    unsigned int ws_stride = head_dim + 2;
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) ws_base[bf16_vec_off + i] = smem_o[0][bf16_vec_off + i];
        if (lane_id == 0) { ws_base[head_dim] = smem_m[0]; ws_base[head_dim + 1] = smem_l[0]; }
    }
}
