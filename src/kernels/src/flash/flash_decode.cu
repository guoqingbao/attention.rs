/**
 * @brief Native flash decode instantiation — dual F16/BF16 templates.
 *
 * Instantiates templated paged decode kernels for HDIM 128/256/512.
 * F16 always; BF16 only when !NO_BF16_KERNEL (SM80+).
 */

#include <cuda_runtime.h>
#include <cstdio>
#include "flash_sm_compat.cuh"
#include <cuda_fp8.h>

// ============================================================================
// HDIM=128, GQA_RATIO=1 (per-Q-head; kv_head computed inside kernel)
// ============================================================================
#undef FLASH_HDIM
#undef GQA_RATIO
#undef WARP_SIZE
#undef HDIM
#undef VEC_BF16
#undef VEC_U32
#undef NUM_WARPS
#undef BC
#define FLASH_HDIM 128
#define GQA_RATIO 1
#define flash_decode_paged      flash_decode_paged_128
#define flash_decode_paged_splitk flash_decode_paged_splitk_128
#define flash_decode_paged_reduce flash_decode_paged_reduce_128
#include "flash_decode_paged.cuh"
#undef flash_decode_paged
#undef flash_decode_paged_splitk
#undef flash_decode_paged_reduce
#undef FLASH_HDIM
#undef GQA_RATIO
#undef WARP_SIZE
#undef HDIM
#undef VEC_BF16
#undef VEC_U32
#undef NUM_WARPS
#undef BC
#undef LDG_VEC_DEFINED
#undef LDG_VEC_LOAD
// Keep FLASH_DECODE_UNPACK_DEFINED: unpack2_bf16_d is HDIM-independent.

#define FLASH_HDIM 256
#define GQA_RATIO 1
#define flash_decode_paged      flash_decode_paged_256
#define flash_decode_paged_splitk flash_decode_paged_splitk_256
#define flash_decode_paged_reduce flash_decode_paged_reduce_256
#include "flash_decode_paged.cuh"
#undef flash_decode_paged
#undef flash_decode_paged_splitk
#undef flash_decode_paged_reduce
#undef FLASH_HDIM
#undef GQA_RATIO
#undef WARP_SIZE
#undef HDIM
#undef VEC_BF16
#undef VEC_U32
#undef NUM_WARPS
#undef BC
#undef LDG_VEC_DEFINED
#undef LDG_VEC_LOAD

#define FLASH_HDIM 512
#define GQA_RATIO 1
#define flash_decode_paged      flash_decode_paged_512
#define flash_decode_paged_splitk flash_decode_paged_splitk_512
#define flash_decode_paged_reduce flash_decode_paged_reduce_512
#include "flash_decode_paged.cuh"
#undef flash_decode_paged
#undef flash_decode_paged_splitk
#undef flash_decode_paged_reduce
#undef FLASH_HDIM
#undef GQA_RATIO
#undef WARP_SIZE
#undef HDIM
#undef VEC_BF16
#undef VEC_U32
#undef NUM_WARPS
#undef BC
#undef LDG_VEC_DEFINED
#undef LDG_VEC_LOAD
#undef FLASH_DECODE_UNPACK_DEFINED

// ============================================================================
// Launch helpers — unified dtype: 0=f16, 1=bf16
// ============================================================================

#define LAUNCH_DECODE_T(HALF, HD) \
    flash_decode_paged_##HD<HALF><<<grid, 256, 0, s>>>( \
        (const HALF*)Q, (const HALF*)K_cache, (const HALF*)V_cache, (HALF*)O, \
        block_tables, seq_lens, max_blocks_per_seq, \
        num_q_heads, num_kv_heads, head_dim, block_size, \
        inv_sqrt_d, q_stride, sliding_window, softcap)

#define LAUNCH_DECODE_SK_T(HALF, HD) \
    flash_decode_paged_splitk_##HD<HALF><<<grid, 256, 0, s>>>( \
        (const HALF*)Q, (const HALF*)K_cache, (const HALF*)V_cache, (float*)workspace, \
        block_tables, seq_lens, max_blocks_per_seq, \
        num_q_heads, num_kv_heads, head_dim, block_size, \
        inv_sqrt_d, num_splits, q_stride, softcap, sliding_window)

extern "C" void call_flash_decode_paged(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d, unsigned int num_seqs, unsigned int q_stride,
    unsigned int sliding_window, float softcap, unsigned int gqa_ratio,
    int dtype, int64_t stream
) {
    (void)gqa_ratio;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);
    if (dtype == 0) {
        if (head_dim <= 128)      { LAUNCH_DECODE_T(__half, 128); }
        else if (head_dim <= 256) { LAUNCH_DECODE_T(__half, 256); }
        else                      { LAUNCH_DECODE_T(__half, 512); }
    }
#ifndef NO_BF16_KERNEL
    else if (dtype == 1) {
        if (head_dim <= 128)      { LAUNCH_DECODE_T(__nv_bfloat16, 128); }
        else if (head_dim <= 256) { LAUNCH_DECODE_T(__nv_bfloat16, 256); }
        else                      { LAUNCH_DECODE_T(__nv_bfloat16, 512); }
    }
#endif
    else {
        printf("call_flash_decode_paged: unsupported dtype %d (0=f16, 1=bf16)\n", (int)dtype);
    }
}

extern "C" void call_flash_decode_paged_splitk(
    const void* Q, const void* K_cache, const void* V_cache, void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size, float inv_sqrt_d,
    unsigned int num_seqs, unsigned int num_splits, unsigned int q_stride,
    float softcap, unsigned int sliding_window, unsigned int gqa_ratio,
    int dtype, int64_t stream
) {
    (void)gqa_ratio;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_splits, num_seqs);
    if (dtype == 0) {
        if (head_dim <= 128)      { LAUNCH_DECODE_SK_T(__half, 128); }
        else if (head_dim <= 256) { LAUNCH_DECODE_SK_T(__half, 256); }
        else                      { LAUNCH_DECODE_SK_T(__half, 512); }
    }
#ifndef NO_BF16_KERNEL
    else if (dtype == 1) {
        if (head_dim <= 128)      { LAUNCH_DECODE_SK_T(__nv_bfloat16, 128); }
        else if (head_dim <= 256) { LAUNCH_DECODE_SK_T(__nv_bfloat16, 256); }
        else                      { LAUNCH_DECODE_SK_T(__nv_bfloat16, 512); }
    }
#endif
    else {
        printf("call_flash_decode_paged_splitk: unsupported dtype %d\n", (int)dtype);
    }
}

extern "C" void call_flash_decode_paged_reduce(
    const void* workspace, void* O,
    unsigned int num_q_heads, unsigned int head_dim,
    unsigned int num_splits, unsigned int num_seqs,
    int dtype, int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);
#define LAUNCH_REDUCE(HALF) \
    do { \
        if (head_dim <= 128) \
            flash_decode_paged_reduce_128<HALF><<<grid, 32, 0, s>>>( \
                (const float*)workspace, (HALF*)O, num_q_heads, head_dim, num_splits); \
        else if (head_dim <= 256) \
            flash_decode_paged_reduce_256<HALF><<<grid, 32, 0, s>>>( \
                (const float*)workspace, (HALF*)O, num_q_heads, head_dim, num_splits); \
        else \
            flash_decode_paged_reduce_512<HALF><<<grid, 32, 0, s>>>( \
                (const float*)workspace, (HALF*)O, num_q_heads, head_dim, num_splits); \
    } while (0)
    if (dtype == 0) { LAUNCH_REDUCE(__half); }
#ifndef NO_BF16_KERNEL
    else if (dtype == 1) { LAUNCH_REDUCE(__nv_bfloat16); }
#endif
    else {
        printf("call_flash_decode_paged_reduce: unsupported dtype %d\n", (int)dtype);
    }
#undef LAUNCH_REDUCE
}

#undef LAUNCH_DECODE_T
#undef LAUNCH_DECODE_SK_T
