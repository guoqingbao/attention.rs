// Instantiate flash attention kernels for all supported head dimensions.
// Each HDIM variant compiles to uniquely-named kernel symbols.
// The launcher dispatches based on runtime head_dim.
//
// We use #include with redefined FLASH_HDIM to instantiate each variant.
// Kernel names get HDIM suffix via token pasting.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Helper to force unique kernel names per HDIM
// We undef FLASH_HDIM before each include and rename via preprocessor

// ============================================================================
// HDIM=128 variants
// ============================================================================
#define FLASH_HDIM 128
#define flash_prefill_paged flash_prefill_paged_128
#define flash_prefill_paged_fp8 flash_prefill_paged_fp8_128
#define flash_decode_paged flash_decode_paged_128
#define flash_decode_paged_splitk flash_decode_paged_splitk_128
#define flash_decode_paged_reduce flash_decode_paged_reduce_128
#define flash_decode_paged_fp8 flash_decode_paged_fp8_128
#define flash_decode_paged_splitk_fp8 flash_decode_paged_splitk_fp8_128
#define flash_reshape_and_cache flash_reshape_and_cache_128
#define flash_reshape_and_cache_fp8 flash_reshape_and_cache_fp8_128
#define flash_bf16_absmax flash_bf16_absmax_128
#define flash_tq_store_k8v4 flash_tq_store_k8v4_128
#define flash_tq_decode_k8v4 flash_tq_decode_k8v4_128

// Need to also rename helper to avoid ODR violations
#define fp8_to_bf16_s fp8_to_bf16_s_128
#define fp8x4_to_bf16x4 fp8x4_to_bf16x4_128
#define fp8_to_f32_d fp8_to_f32_d_128
#define fp8x4_to_f32x4 fp8x4_to_f32x4_128
#define unpack2_bf16_d unpack2_bf16_d_128
#define unpack2_bf16_tq unpack2_bf16_tq_128
#define wht_intra_thread wht_intra_thread_128
#define wht_cross_thread wht_cross_thread_128
#define wht_transform wht_transform_128
#define get_sign_flip get_sign_flip_128
#define quantize_4bit quantize_4bit_128
#define dequantize_4bit dequantize_4bit_128
#define pack_4bit pack_4bit_128
#define unpack_4bit_lo unpack_4bit_lo_128
#define unpack_4bit_hi unpack_4bit_hi_128
#define flash_tq4_store flash_tq4_store_128
#define flash_tq4_decode flash_tq4_decode_128
#define flash_tq4_decode_splitk flash_tq4_decode_splitk_128
#define flash_tq4_prefill flash_tq4_prefill_128
#define unpack2_bf16_tq4 unpack2_bf16_tq4_128
#define flash_tq3_store flash_tq3_store_128
#define flash_tq3_decode flash_tq3_decode_128
#define flash_tq3_decode_splitk flash_tq3_decode_splitk_128
#define quantize_3bit quantize_3bit_128
#define dequantize_3bit dequantize_3bit_128
#define pack_3bit_x8 pack_3bit_x8_128
#define unpack_3bit_x8 unpack_3bit_x8_128

#include "flash_prefill_paged.cuh"
#include "flash_prefill_paged_fp8.cuh"
#include "flash_decode_paged.cuh"
#include "flash_decode_paged_fp8.cuh"
#include "flash_reshape_cache.cuh"
#include "flash_turboquant.cuh"
#include "flash_turboquant_lowbit.cuh"
#include "flash_prefill_tq4.cuh"

#undef FLASH_HDIM
#undef flash_prefill_paged
#undef flash_prefill_paged_fp8
#undef flash_decode_paged
#undef flash_decode_paged_splitk
#undef flash_decode_paged_reduce
#undef flash_decode_paged_fp8
#undef flash_decode_paged_splitk_fp8
#undef flash_reshape_and_cache
#undef flash_reshape_and_cache_fp8
#undef flash_bf16_absmax
#undef flash_tq_store_k8v4
#undef flash_tq_decode_k8v4
#undef fp8_to_bf16_s
#undef fp8x4_to_bf16x4
#undef fp8_to_f32_d
#undef fp8x4_to_f32x4
#undef unpack2_bf16_d
#undef unpack2_bf16_tq
#undef wht_intra_thread
#undef wht_cross_thread
#undef wht_transform
#undef get_sign_flip
#undef quantize_4bit
#undef dequantize_4bit
#undef pack_4bit
#undef unpack_4bit_lo
#undef unpack_4bit_hi
// Cleanup per-HDIM defines
#undef BR
#undef BC
#undef PAD_KV
#undef HDIM
#undef HDIM_PAD
#undef PAD_P
#undef N_TILES_PER_WARP
#undef TILE_CHUNKS
#undef NUM_THREADS
#undef WARP_SIZE
#undef VEC_BF16
#undef VEC_U32
#undef VEC_FP8
#undef NUM_WARPS
#undef FLASH_DECODE_UNPACK_DEFINED
#undef LOAD_KV_TILE_BF16
#undef LOAD_KV_TILE_FP8
#undef FP8_DEQUANT_HELPERS_DEFINED
#undef TQ_VEC
#undef TQ_NUM_WARPS
#undef TQ_BC
#undef TQ_VEC_U32
#undef flash_tq4_store
#undef flash_tq4_decode
#undef flash_tq4_decode_splitk
#undef flash_tq4_prefill
#undef unpack2_bf16_tq4
#undef TQ4_VEC
#undef TQ4_NUM_WARPS
#undef TQ4_BC
#undef TQ4_VEC_U32
#undef UNPACK2_BF16_TQ4_DEFINED
#undef flash_tq3_store
#undef flash_tq3_decode
#undef flash_tq3_decode_splitk
#undef quantize_3bit
#undef dequantize_3bit
#undef pack_3bit_x8
#undef unpack_3bit_x8
#undef TQ3_K_BYTES_PER_HEAD
#undef TQ3_QUANT_HELPERS_DEFINED

// ============================================================================
// HDIM=256 variants
// ============================================================================
#define FLASH_HDIM 256
#define flash_prefill_paged flash_prefill_paged_256
#define flash_prefill_paged_fp8 flash_prefill_paged_fp8_256
#define flash_decode_paged flash_decode_paged_256
#define flash_decode_paged_splitk flash_decode_paged_splitk_256
#define flash_decode_paged_reduce flash_decode_paged_reduce_256
#define flash_decode_paged_fp8 flash_decode_paged_fp8_256
#define flash_decode_paged_splitk_fp8 flash_decode_paged_splitk_fp8_256
#define flash_reshape_and_cache flash_reshape_and_cache_256
#define flash_reshape_and_cache_fp8 flash_reshape_and_cache_fp8_256
#define flash_bf16_absmax flash_bf16_absmax_256
#define flash_tq_store_k8v4 flash_tq_store_k8v4_256
#define flash_tq_decode_k8v4 flash_tq_decode_k8v4_256
#define fp8_to_bf16_s fp8_to_bf16_s_256
#define fp8x4_to_bf16x4 fp8x4_to_bf16x4_256
#define fp8_to_f32_d fp8_to_f32_d_256
#define fp8x4_to_f32x4 fp8x4_to_f32x4_256
#define unpack2_bf16_d unpack2_bf16_d_256
#define unpack2_bf16_tq unpack2_bf16_tq_256
#define wht_intra_thread wht_intra_thread_256
#define wht_cross_thread wht_cross_thread_256
#define wht_transform wht_transform_256
#define get_sign_flip get_sign_flip_256
#define quantize_4bit quantize_4bit_256
#define dequantize_4bit dequantize_4bit_256
#define pack_4bit pack_4bit_256
#define unpack_4bit_lo unpack_4bit_lo_256
#define unpack_4bit_hi unpack_4bit_hi_256
#define flash_tq4_store flash_tq4_store_256
#define flash_tq4_decode flash_tq4_decode_256
#define flash_tq4_decode_splitk flash_tq4_decode_splitk_256
#define flash_tq4_prefill flash_tq4_prefill_256
#define unpack2_bf16_tq4 unpack2_bf16_tq4_256
#define flash_tq3_store flash_tq3_store_256
#define flash_tq3_decode flash_tq3_decode_256
#define flash_tq3_decode_splitk flash_tq3_decode_splitk_256
#define quantize_3bit quantize_3bit_256
#define dequantize_3bit dequantize_3bit_256
#define pack_3bit_x8 pack_3bit_x8_256
#define unpack_3bit_x8 unpack_3bit_x8_256

#include "flash_prefill_paged.cuh"
#include "flash_prefill_paged_fp8.cuh"
#include "flash_decode_paged.cuh"
#include "flash_decode_paged_fp8.cuh"
#include "flash_reshape_cache.cuh"
#include "flash_turboquant.cuh"
#include "flash_turboquant_lowbit.cuh"
#include "flash_prefill_tq4.cuh"

#undef FLASH_HDIM
#undef flash_prefill_paged
#undef flash_prefill_paged_fp8
#undef flash_decode_paged
#undef flash_decode_paged_splitk
#undef flash_decode_paged_reduce
#undef flash_decode_paged_fp8
#undef flash_decode_paged_splitk_fp8
#undef flash_reshape_and_cache
#undef flash_reshape_and_cache_fp8
#undef flash_bf16_absmax
#undef flash_tq_store_k8v4
#undef flash_tq_decode_k8v4
#undef fp8_to_bf16_s
#undef fp8x4_to_bf16x4
#undef fp8_to_f32_d
#undef fp8x4_to_f32x4
#undef unpack2_bf16_d
#undef unpack2_bf16_tq
#undef wht_intra_thread
#undef wht_cross_thread
#undef wht_transform
#undef get_sign_flip
#undef quantize_4bit
#undef dequantize_4bit
#undef pack_4bit
#undef unpack_4bit_lo
#undef unpack_4bit_hi
#undef BR
#undef BC
#undef PAD_KV
#undef HDIM
#undef HDIM_PAD
#undef PAD_P
#undef N_TILES_PER_WARP
#undef TILE_CHUNKS
#undef NUM_THREADS
#undef WARP_SIZE
#undef VEC_BF16
#undef VEC_U32
#undef VEC_FP8
#undef NUM_WARPS
#undef FLASH_DECODE_UNPACK_DEFINED
#undef LOAD_KV_TILE_BF16
#undef LOAD_KV_TILE_FP8
#undef FP8_DEQUANT_HELPERS_DEFINED
#undef TQ_VEC
#undef TQ_NUM_WARPS
#undef TQ_BC
#undef TQ_VEC_U32
#undef flash_tq4_store
#undef flash_tq4_decode
#undef flash_tq4_decode_splitk
#undef flash_tq4_prefill
#undef unpack2_bf16_tq4
#undef TQ4_VEC
#undef TQ4_NUM_WARPS
#undef TQ4_BC
#undef TQ4_VEC_U32
#undef UNPACK2_BF16_TQ4_DEFINED
#undef flash_tq3_store
#undef flash_tq3_decode
#undef flash_tq3_decode_splitk
#undef quantize_3bit
#undef dequantize_3bit
#undef pack_3bit_x8
#undef unpack_3bit_x8
#undef TQ3_K_BYTES_PER_HEAD
#undef TQ3_QUANT_HELPERS_DEFINED

// ============================================================================
// HDIM=512 variants
// ============================================================================
#define FLASH_HDIM 512
#define flash_prefill_paged flash_prefill_paged_512
#define flash_prefill_paged_fp8 flash_prefill_paged_fp8_512
#define flash_decode_paged flash_decode_paged_512
#define flash_decode_paged_splitk flash_decode_paged_splitk_512
#define flash_decode_paged_reduce flash_decode_paged_reduce_512
#define flash_decode_paged_fp8 flash_decode_paged_fp8_512
#define flash_decode_paged_splitk_fp8 flash_decode_paged_splitk_fp8_512
#define flash_reshape_and_cache flash_reshape_and_cache_512
#define flash_reshape_and_cache_fp8 flash_reshape_and_cache_fp8_512
#define flash_bf16_absmax flash_bf16_absmax_512
#define flash_tq_store_k8v4 flash_tq_store_k8v4_512
#define flash_tq_decode_k8v4 flash_tq_decode_k8v4_512
#define fp8_to_bf16_s fp8_to_bf16_s_512
#define fp8x4_to_bf16x4 fp8x4_to_bf16x4_512
#define fp8_to_f32_d fp8_to_f32_d_512
#define fp8x4_to_f32x4 fp8x4_to_f32x4_512
#define unpack2_bf16_d unpack2_bf16_d_512
#define unpack2_bf16_tq unpack2_bf16_tq_512
#define wht_intra_thread wht_intra_thread_512
#define wht_cross_thread wht_cross_thread_512
#define wht_transform wht_transform_512
#define get_sign_flip get_sign_flip_512
#define quantize_4bit quantize_4bit_512
#define dequantize_4bit dequantize_4bit_512
#define pack_4bit pack_4bit_512
#define unpack_4bit_lo unpack_4bit_lo_512
#define unpack_4bit_hi unpack_4bit_hi_512
#define flash_tq4_store flash_tq4_store_512
#define flash_tq4_decode flash_tq4_decode_512
#define flash_tq4_decode_splitk flash_tq4_decode_splitk_512
#define flash_tq4_prefill flash_tq4_prefill_512
#define unpack2_bf16_tq4 unpack2_bf16_tq4_512
#define flash_tq3_store flash_tq3_store_512
#define flash_tq3_decode flash_tq3_decode_512
#define flash_tq3_decode_splitk flash_tq3_decode_splitk_512
#define quantize_3bit quantize_3bit_512
#define dequantize_3bit dequantize_3bit_512
#define pack_3bit_x8 pack_3bit_x8_512
#define unpack_3bit_x8 unpack_3bit_x8_512

// For 512, we need the specific defines
#define BR_512 32
#define BC_512 32
#define PAD_P_512 8
#define N_TILES_PER_WARP_512 16
#define TILE_CHUNKS_512 (BR_512 * (512 / 8))
#define NUM_THREADS_512 256

#include "flash_prefill_paged.cuh"
#include "flash_prefill_paged_fp8.cuh"
#include "flash_decode_paged.cuh"
#include "flash_decode_paged_fp8.cuh"
#include "flash_reshape_cache.cuh"
#include "flash_turboquant.cuh"
#include "flash_turboquant_lowbit.cuh"
#include "flash_prefill_tq4.cuh"

#undef FLASH_HDIM
#undef flash_prefill_paged
#undef flash_prefill_paged_fp8
#undef flash_decode_paged
#undef flash_decode_paged_splitk
#undef flash_decode_paged_reduce
#undef flash_decode_paged_fp8
#undef flash_decode_paged_splitk_fp8
#undef flash_reshape_and_cache
#undef flash_reshape_and_cache_fp8
#undef flash_bf16_absmax
#undef flash_tq_store_k8v4
#undef flash_tq_decode_k8v4
#undef flash_tq4_store
#undef flash_tq4_decode
#undef flash_tq4_decode_splitk
#undef flash_tq4_prefill
#undef fp8_to_bf16_s
#undef fp8_to_f32_d
#undef unpack2_bf16_d
#undef unpack2_bf16_tq
#undef unpack2_bf16_tq4
#undef wht_intra_thread
#undef wht_cross_thread
#undef wht_transform
#undef get_sign_flip
#undef quantize_4bit
#undef dequantize_4bit
#undef pack_4bit
#undef unpack_4bit_lo
#undef unpack_4bit_hi
#undef flash_tq3_store
#undef flash_tq3_decode
#undef flash_tq3_decode_splitk
#undef quantize_3bit
#undef dequantize_3bit
#undef pack_3bit_x8
#undef unpack_3bit_x8
#undef TQ3_K_BYTES_PER_HEAD
#undef TQ3_QUANT_HELPERS_DEFINED

// ============================================================================
// Dispatch launchers — called from Rust FFI
// ============================================================================

#define DISPATCH_PREFILL(HDIM_VAL, ...) flash_prefill_paged_##HDIM_VAL<<<__VA_ARGS__>>>
#define DISPATCH_PREFILL_FP8(HDIM_VAL, ...) flash_prefill_paged_fp8_##HDIM_VAL<<<__VA_ARGS__>>>
#define DISPATCH_DECODE(HDIM_VAL, ...) flash_decode_paged_##HDIM_VAL<<<__VA_ARGS__>>>
#define DISPATCH_DECODE_SPLITK(HDIM_VAL, ...) flash_decode_paged_splitk_##HDIM_VAL<<<__VA_ARGS__>>>
#define DISPATCH_DECODE_REDUCE(HDIM_VAL, ...) flash_decode_paged_reduce_##HDIM_VAL<<<__VA_ARGS__>>>
#define DISPATCH_DECODE_FP8(HDIM_VAL, ...) flash_decode_paged_fp8_##HDIM_VAL<<<__VA_ARGS__>>>
#define DISPATCH_DECODE_SPLITK_FP8(HDIM_VAL, ...) flash_decode_paged_splitk_fp8_##HDIM_VAL<<<__VA_ARGS__>>>

// Prefill BF16 launcher
extern "C" void call_flash_prefill_paged(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_table,
    unsigned int q_len, unsigned int kv_len, unsigned int q_offset,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    unsigned int sliding_window, unsigned int causal,
    float inv_sqrt_d, float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    unsigned int br = 32;
    dim3 grid(num_q_heads, (q_len + br - 1) / br);

    #define LAUNCH_PREFILL(HD, THREADS, SMEM) \
        flash_prefill_paged_##HD<<<grid, THREADS, SMEM, s>>>( \
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, \
            (const __nv_bfloat16*)V_cache, (__nv_bfloat16*)O, \
            block_table, q_len, kv_len, q_offset, num_q_heads, num_kv_heads, \
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap)

    if (head_dim <= 128) {
        LAUNCH_PREFILL(128, 128, 0);
    } else if (head_dim <= 256) {
        unsigned int smem = (32*264 + 2*32*264 + 32*264) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_paged_256,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_PREFILL(256, 128, smem);
    } else {
        // HDIM=512: dynamic smem
        // Q[32*512] + K[32*512] + V[32*512] + P[32*40] + ml[32*2] in bf16/f32
        unsigned int smem = (32*512 + 32*512 + 32*512) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_paged_512,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_PREFILL(512, 256, smem);
    }
    #undef LAUNCH_PREFILL
}

// Prefill FP8 launcher
extern "C" void call_flash_prefill_paged_fp8(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_table,
    unsigned int q_len, unsigned int kv_len, unsigned int q_offset,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    unsigned int sliding_window, unsigned int causal,
    float inv_sqrt_d, float softcap,
    const float* k_scale_ptr, const float* v_scale_ptr, unsigned long long fp8_cache_stride,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    unsigned int br = 32;
    dim3 grid(num_q_heads, (q_len + br - 1) / br);

    #define LAUNCH_PREFILL_FP8(HD, THREADS, SMEM) \
        flash_prefill_paged_fp8_##HD<<<grid, THREADS, SMEM, s>>>( \
            (const __nv_bfloat16*)Q, K_cache, V_cache, (__nv_bfloat16*)O, \
            block_table, q_len, kv_len, q_offset, num_q_heads, num_kv_heads, \
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap, \
            k_scale_ptr, v_scale_ptr, fp8_cache_stride)

    if (head_dim <= 128) {
        LAUNCH_PREFILL_FP8(128, 128, 0);
    } else if (head_dim <= 256) {
        unsigned int smem = (32*264 + 2*32*264 + 32*264) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_paged_fp8_256,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_PREFILL_FP8(256, 128, smem);
    } else {
        unsigned int smem = (32*512 + 32*512 + 32*512) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_paged_fp8_512,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_PREFILL_FP8(512, 256, smem);
    }
    #undef LAUNCH_PREFILL_FP8
}

// Decode BF16 launcher
extern "C" void call_flash_decode_paged(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs,
    unsigned int q_stride,
    unsigned int sliding_window, float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    #define LAUNCH_DECODE(HD) \
        flash_decode_paged_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, \
            (const __nv_bfloat16*)V_cache, (__nv_bfloat16*)O, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, q_stride, sliding_window, softcap)

    if (head_dim <= 128) { LAUNCH_DECODE(128); }
    else if (head_dim <= 256) { LAUNCH_DECODE(256); }
    else { LAUNCH_DECODE(512); }
    #undef LAUNCH_DECODE
}

// Decode BF16 split-K launcher
extern "C" void call_flash_decode_paged_splitk(
    const void* Q, const void* K_cache, const void* V_cache,
    void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs, unsigned int num_splits,
    unsigned int q_stride, float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_splits, num_seqs);

    #define LAUNCH_DECODE_SK(HD) \
        flash_decode_paged_splitk_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K_cache, \
            (const __nv_bfloat16*)V_cache, (float*)workspace, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_splits, q_stride, softcap)

    if (head_dim <= 128) { LAUNCH_DECODE_SK(128); }
    else if (head_dim <= 256) { LAUNCH_DECODE_SK(256); }
    else { LAUNCH_DECODE_SK(512); }
    #undef LAUNCH_DECODE_SK
}

// Reduce split-K partials
extern "C" void call_flash_decode_paged_reduce(
    const void* workspace, void* O,
    unsigned int num_q_heads, unsigned int head_dim,
    unsigned int num_splits, unsigned int num_seqs,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    #define LAUNCH_REDUCE(HD) \
        flash_decode_paged_reduce_##HD<<<grid, 32, 0, s>>>( \
            (const float*)workspace, (__nv_bfloat16*)O, \
            num_q_heads, head_dim, num_splits)

    if (head_dim <= 128) { LAUNCH_REDUCE(128); }
    else if (head_dim <= 256) { LAUNCH_REDUCE(256); }
    else { LAUNCH_REDUCE(512); }
    #undef LAUNCH_REDUCE
}

// Decode FP8 launcher
extern "C" void call_flash_decode_paged_fp8(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs,
    unsigned int q_stride,
    unsigned int sliding_window, float softcap,
    const float* k_scale_ptr, const float* v_scale_ptr, unsigned long long fp8_cache_stride,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    #define LAUNCH_DECODE_FP8(HD) \
        flash_decode_paged_fp8_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, K_cache, V_cache, (__nv_bfloat16*)O, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, q_stride, sliding_window, softcap, \
            k_scale_ptr, v_scale_ptr, fp8_cache_stride)

    if (head_dim <= 128) { LAUNCH_DECODE_FP8(128); }
    else if (head_dim <= 256) { LAUNCH_DECODE_FP8(256); }
    else { LAUNCH_DECODE_FP8(512); }
    #undef LAUNCH_DECODE_FP8
}

// Decode FP8 split-K launcher
extern "C" void call_flash_decode_paged_splitk_fp8(
    const void* Q, const void* K_cache, const void* V_cache,
    void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs, unsigned int num_splits,
    unsigned int q_stride, float softcap,
    const float* k_scale_ptr, const float* v_scale_ptr, unsigned long long fp8_cache_stride,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_splits, num_seqs);

    #define LAUNCH_DECODE_SK_FP8(HD) \
        flash_decode_paged_splitk_fp8_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, K_cache, V_cache, (float*)workspace, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_splits, q_stride, softcap, \
            k_scale_ptr, v_scale_ptr, fp8_cache_stride)

    if (head_dim <= 128) { LAUNCH_DECODE_SK_FP8(128); }
    else if (head_dim <= 256) { LAUNCH_DECODE_SK_FP8(256); }
    else { LAUNCH_DECODE_SK_FP8(512); }
    #undef LAUNCH_DECODE_SK_FP8
}

// Reshape & cache BF16 launcher
extern "C" void call_flash_reshape_and_cache_bf16(
    const void* key, const void* value, void* key_cache, void* value_cache,
    const long long* slot_mapping,
    unsigned int num_tokens, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_tokens, num_kv_heads);
    unsigned int threads = (head_dim + 7) / 8;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;

    #define LAUNCH_CACHE(HD) \
        flash_reshape_and_cache_##HD<<<grid, threads, 0, s>>>( \
            (const __nv_bfloat16*)key, (const __nv_bfloat16*)value, \
            (__nv_bfloat16*)key_cache, (__nv_bfloat16*)value_cache, \
            slot_mapping, num_tokens, num_kv_heads, head_dim, cache_block_size)

    if (head_dim <= 128) { LAUNCH_CACHE(128); }
    else if (head_dim <= 256) { LAUNCH_CACHE(256); }
    else { LAUNCH_CACHE(512); }
    #undef LAUNCH_CACHE
}

// Reshape & cache FP8 launcher
extern "C" void call_flash_reshape_and_cache_fp8_kv(
    const void* key, const void* value, void* key_cache, void* value_cache,
    const long long* slot_mapping,
    unsigned int num_tokens, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    const float* k_scale_ptr, const float* v_scale_ptr,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_tokens, num_kv_heads);
    unsigned int threads = (head_dim + 31) / 32 * 32;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;

    #define LAUNCH_CACHE_FP8(HD) \
        flash_reshape_and_cache_fp8_##HD<<<grid, threads, 0, s>>>( \
            (const __nv_bfloat16*)key, (const __nv_bfloat16*)value, \
            key_cache, value_cache, slot_mapping, \
            num_tokens, num_kv_heads, head_dim, cache_block_size, \
            k_scale_ptr, v_scale_ptr)

    if (head_dim <= 128) { LAUNCH_CACHE_FP8(128); }
    else if (head_dim <= 256) { LAUNCH_CACHE_FP8(256); }
    else { LAUNCH_CACHE_FP8(512); }
    #undef LAUNCH_CACHE_FP8
}

// ============================================================================
// TurboQuant k8v4 launchers
// ============================================================================

// TurboQuant store: K → WHT rotate → FP8, V → 4-bit uniform
extern "C" void call_flash_tq_store_k8v4(
    const void* K, const void* V,
    void* K_cache, void* V_absmax, void* V_quant,
    const long long* slot_mapping,
    unsigned int num_tokens, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    const float* k_scale_ptr,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_tokens, num_kv_heads);
    unsigned int threads = 32; // single warp per head

    #define LAUNCH_TQ_STORE(HD) \
        flash_tq_store_k8v4_##HD<<<grid, threads, 0, s>>>( \
            (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, \
            K_cache, (float*)V_absmax, (unsigned char*)V_quant, \
            slot_mapping, num_tokens, num_kv_heads, head_dim, block_size, \
            k_scale_ptr)

    if (head_dim <= 128) { LAUNCH_TQ_STORE(128); }
    else if (head_dim <= 256) { LAUNCH_TQ_STORE(256); }
    else { LAUNCH_TQ_STORE(512); }
    #undef LAUNCH_TQ_STORE
}

// TurboQuant decode: FP8 keys + 4-bit values → attention output
extern "C" void call_flash_tq_decode_k8v4(
    const void* Q, const void* K_cache,
    const void* V_absmax, const void* V_quant,
    void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs,
    unsigned int q_stride,
    float softcap,
    const float* k_scale_ptr,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);
    unsigned int threads = 8 * 32; // TQ_NUM_WARPS * WARP_SIZE

    #define LAUNCH_TQ_DECODE(HD) \
        flash_tq_decode_k8v4_##HD<<<grid, threads, 0, s>>>( \
            (const __nv_bfloat16*)Q, K_cache, \
            (const float*)V_absmax, (const unsigned char*)V_quant, \
            (__nv_bfloat16*)O, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_seqs, q_stride, softcap, k_scale_ptr)

    if (head_dim <= 128) { LAUNCH_TQ_DECODE(128); }
    else if (head_dim <= 256) { LAUNCH_TQ_DECODE(256); }
    else { LAUNCH_TQ_DECODE(512); }
    #undef LAUNCH_TQ_DECODE
}

// TurboQuant turbo4 store: K → WHT → 4-bit, V → 4-bit
extern "C" void call_flash_tq4_store(
    const void* K, const void* V,
    void* K_absmax, void* K_quant,
    void* V_absmax, void* V_quant,
    const long long* slot_mapping,
    unsigned int num_tokens, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_tokens, num_kv_heads);

    #define LAUNCH_TQ4_STORE(HD) \
        flash_tq4_store_##HD<<<grid, 32, 0, s>>>( \
            (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, \
            (float*)K_absmax, (unsigned char*)K_quant, \
            (float*)V_absmax, (unsigned char*)V_quant, \
            slot_mapping, num_tokens, num_kv_heads, head_dim, block_size)

    if (head_dim <= 128) { LAUNCH_TQ4_STORE(128); }
    else if (head_dim <= 256) { LAUNCH_TQ4_STORE(256); }
    else { LAUNCH_TQ4_STORE(512); }
    #undef LAUNCH_TQ4_STORE
}

// TurboQuant turbo4 decode: 4-bit K + 4-bit V → attention
extern "C" void call_flash_tq4_decode(
    const void* Q,
    const void* K_absmax, const void* K_quant,
    const void* V_absmax, const void* V_quant,
    void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs,
    unsigned int q_stride,
    float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    #define LAUNCH_TQ4_DECODE(HD) \
        flash_tq4_decode_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, \
            (const float*)K_absmax, (const unsigned char*)K_quant, \
            (const float*)V_absmax, (const unsigned char*)V_quant, \
            (__nv_bfloat16*)O, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_seqs, q_stride, softcap)

    if (head_dim <= 128) { LAUNCH_TQ4_DECODE(128); }
    else if (head_dim <= 256) { LAUNCH_TQ4_DECODE(256); }
    else { LAUNCH_TQ4_DECODE(512); }
    #undef LAUNCH_TQ4_DECODE
}

// TurboQuant turbo4 decode split-K: long sequences
extern "C" void call_flash_tq4_decode_splitk(
    const void* Q,
    const void* K_absmax, const void* K_quant,
    const void* V_absmax, const void* V_quant,
    void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_splits,
    unsigned int num_seqs,
    unsigned int q_stride,
    float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_splits, num_seqs);

    #define LAUNCH_TQ4_DECODE_SK(HD) \
        flash_tq4_decode_splitk_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, \
            (const float*)K_absmax, (const unsigned char*)K_quant, \
            (const float*)V_absmax, (const unsigned char*)V_quant, \
            (float*)workspace, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_splits, num_seqs, q_stride, softcap)

    if (head_dim <= 128) { LAUNCH_TQ4_DECODE_SK(128); }
    else if (head_dim <= 256) { LAUNCH_TQ4_DECODE_SK(256); }
    else { LAUNCH_TQ4_DECODE_SK(512); }
    #undef LAUNCH_TQ4_DECODE_SK
}

// TurboQuant turbo3 store: K → WHT → 3-bit, V → 4-bit
extern "C" void call_flash_tq3_store(
    const void* K, const void* V,
    void* K_absmax, void* K_quant,
    void* V_absmax, void* V_quant,
    const long long* slot_mapping,
    unsigned int num_tokens, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_tokens, num_kv_heads);

    #define LAUNCH_TQ3_STORE(HD) \
        flash_tq3_store_##HD<<<grid, 32, 0, s>>>( \
            (const __nv_bfloat16*)K, (const __nv_bfloat16*)V, \
            (float*)K_absmax, (unsigned char*)K_quant, \
            (float*)V_absmax, (unsigned char*)V_quant, \
            slot_mapping, num_tokens, num_kv_heads, head_dim, block_size)

    if (head_dim <= 128) { LAUNCH_TQ3_STORE(128); }
    else if (head_dim <= 256) { LAUNCH_TQ3_STORE(256); }
    else { LAUNCH_TQ3_STORE(512); }
    #undef LAUNCH_TQ3_STORE
}

// TurboQuant turbo3 decode: 3-bit K + 4-bit V → attention
extern "C" void call_flash_tq3_decode(
    const void* Q,
    const void* K_absmax, const void* K_quant,
    const void* V_absmax, const void* V_quant,
    void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs,
    unsigned int q_stride,
    float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    #define LAUNCH_TQ3_DECODE(HD) \
        flash_tq3_decode_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, \
            (const float*)K_absmax, (const unsigned char*)K_quant, \
            (const float*)V_absmax, (const unsigned char*)V_quant, \
            (__nv_bfloat16*)O, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_seqs, q_stride, softcap)

    if (head_dim <= 128) { LAUNCH_TQ3_DECODE(128); }
    else if (head_dim <= 256) { LAUNCH_TQ3_DECODE(256); }
    else { LAUNCH_TQ3_DECODE(512); }
    #undef LAUNCH_TQ3_DECODE
}

// TurboQuant turbo3 decode split-K: long sequences
extern "C" void call_flash_tq3_decode_splitk(
    const void* Q,
    const void* K_absmax, const void* K_quant,
    const void* V_absmax, const void* V_quant,
    void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_splits,
    unsigned int num_seqs,
    unsigned int q_stride,
    float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_splits, num_seqs);

    #define LAUNCH_TQ3_DECODE_SK(HD) \
        flash_tq3_decode_splitk_##HD<<<grid, 256, 0, s>>>( \
            (const __nv_bfloat16*)Q, \
            (const float*)K_absmax, (const unsigned char*)K_quant, \
            (const float*)V_absmax, (const unsigned char*)V_quant, \
            (float*)workspace, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_splits, num_seqs, q_stride, softcap)

    if (head_dim <= 128) { LAUNCH_TQ3_DECODE_SK(128); }
    else if (head_dim <= 256) { LAUNCH_TQ3_DECODE_SK(256); }
    else { LAUNCH_TQ3_DECODE_SK(512); }
    #undef LAUNCH_TQ3_DECODE_SK
}

// TurboQuant 4-bit prefill: reads K/V from 4-bit TQ buffers, dequant to BF16 in smem
extern "C" void call_flash_tq4_prefill(
    const void* Q,
    const void* K_absmax, const void* K_quant,
    const void* V_absmax, const void* V_quant,
    void* O,
    const int* block_table,
    unsigned int q_len, unsigned int kv_len, unsigned int q_offset,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    unsigned int sliding_window, unsigned int causal,
    float inv_sqrt_d, float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    unsigned int br = 32;
    dim3 grid(num_q_heads, (q_len + br - 1) / br);

    #define LAUNCH_TQ4_PREFILL(HD, THREADS, SMEM) \
        flash_tq4_prefill_##HD<<<grid, THREADS, SMEM, s>>>( \
            (const __nv_bfloat16*)Q, \
            (const float*)K_absmax, (const unsigned char*)K_quant, \
            (const float*)V_absmax, (const unsigned char*)V_quant, \
            (__nv_bfloat16*)O, \
            block_table, q_len, kv_len, q_offset, num_q_heads, num_kv_heads, \
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap)

    if (head_dim <= 128) {
        LAUNCH_TQ4_PREFILL(128, 128, 0);
    } else if (head_dim <= 256) {
        unsigned int smem = (32*264 + 2*32*264 + 32*264) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_tq4_prefill_256,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_TQ4_PREFILL(256, 128, smem);
    } else {
        unsigned int smem = (32*512 + 32*512 + 32*512) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_tq4_prefill_512,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_TQ4_PREFILL(512, 256, smem);
    }
    #undef LAUNCH_TQ4_PREFILL
}
