/**
 * FP8 KV cache store with WHT rotation on K and per-tensor scale constants.
 *
 * K: sign_flip -> WHT -> -> per-head absmax -> c_k * absmax -> FP8 E4M3
 * V: per-head absmax -> c_v * absmax -> FP8 E4M3 (no rotation)
 *
 * rotate=false reproduces the existing flash_reshape_and_cache_fp8 behavior.
 */

#include "flash_sm_compat.cuh"
// wht_transform, get_sign_flip already included via flash_turboquant.cuh
// in flash_instantiate.cu before this file

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

#define FP8ROT_VEC (HDIM / WARP_SIZE)

template<typename HalfT>
__global__ void flash_fp8_rot_store(
    const HalfT* __restrict__ key,
    const HalfT* __restrict__ value,
    void* __restrict__ key_cache,
    void* __restrict__ value_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const float c_k,
    const float c_v,
    const bool rotate
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;
    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    unsigned int block_idx = (unsigned int)(slot / cache_block_size);
    unsigned int block_off = (unsigned int)(slot % cache_block_size);

    unsigned long long src_offset = (unsigned long long)token_idx * num_kv_heads * head_dim
                                   + (unsigned long long)head_idx * head_dim;
    unsigned long long dst_offset = (unsigned long long)block_idx * cache_block_size * num_kv_heads * head_dim
                                   + (unsigned long long)block_off * num_kv_heads * head_dim
                                   + (unsigned long long)head_idx * head_dim;

    __nv_fp8_storage_t* k_dst = (__nv_fp8_storage_t*)key_cache + dst_offset;
    __nv_fp8_storage_t* v_dst = (__nv_fp8_storage_t*)value_cache + dst_offset;

    // K: load, optionally rotate, compute absmax, quantize to FP8
    float k_reg[FP8ROT_VEC];
    #pragma unroll
    for (int i = 0; i < FP8ROT_VEC; i++) {
        unsigned int ch = lane_id * FP8ROT_VEC + i;
        k_reg[i] = FLASH_TO_FLOAT(key[src_offset + ch]);
        if (rotate) {
            k_reg[i] *= get_sign_flip(head_idx, ch);
        }
    }
    if (rotate) {
        wht_transform(k_reg, lane_id);
    }

    float k_absmax = 0.f;
    #pragma unroll
    for (int i = 0; i < FP8ROT_VEC; i++) k_absmax = fmaxf(k_absmax, fabsf(k_reg[i]));
    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        k_absmax = fmaxf(k_absmax, __shfl_xor_sync(0xffffffff, k_absmax, off));
    k_absmax = __shfl_sync(0xffffffff, k_absmax, 0);

    float k_scale = c_k * k_absmax;
    float k_inv = (k_scale > 1e-12f) ? (448.0f / k_scale) : 1.0f;

    #pragma unroll
    for (int i = 0; i < FP8ROT_VEC; i++) {
        unsigned int ch = lane_id * FP8ROT_VEC + i;
        float kf = k_reg[i] * k_inv;
        k_dst[ch] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
    }

    // V: load, compute absmax, quantize to FP8 (no rotation)
    float v_reg[FP8ROT_VEC];
    #pragma unroll
    for (int i = 0; i < FP8ROT_VEC; i++) {
        unsigned int ch = lane_id * FP8ROT_VEC + i;
        v_reg[i] = FLASH_TO_FLOAT(value[src_offset + ch]);
    }

    float v_absmax = 0.f;
    #pragma unroll
    for (int i = 0; i < FP8ROT_VEC; i++) v_absmax = fmaxf(v_absmax, fabsf(v_reg[i]));
    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        v_absmax = fmaxf(v_absmax, __shfl_xor_sync(0xffffffff, v_absmax, off));
    v_absmax = __shfl_sync(0xffffffff, v_absmax, 0);

    float v_scale = c_v * v_absmax;
    float v_inv = (v_scale > 1e-12f) ? (448.0f / v_scale) : 1.0f;

    #pragma unroll
    for (int i = 0; i < FP8ROT_VEC; i++) {
        unsigned int ch = lane_id * FP8ROT_VEC + i;
        float vf = v_reg[i] * v_inv;
        v_dst[ch] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, __NV_E4M3);
    }
}