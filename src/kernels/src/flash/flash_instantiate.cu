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

// Need to also rename helper to avoid ODR violations
#define fp8_to_bf16_s fp8_to_bf16_s_128
#define fp8_to_f32_d fp8_to_f32_d_128
#define unpack2_bf16_d unpack2_bf16_d_128

#include "flash_prefill_paged.cuh"
#include "flash_prefill_paged_fp8.cuh"
#include "flash_decode_paged.cuh"
#include "flash_decode_paged_fp8.cuh"
#include "flash_reshape_cache.cuh"

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
#undef fp8_to_bf16_s
#undef fp8_to_f32_d
#undef unpack2_bf16_d
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
#define fp8_to_bf16_s fp8_to_bf16_s_256
#define fp8_to_f32_d fp8_to_f32_d_256
#define unpack2_bf16_d unpack2_bf16_d_256

#include "flash_prefill_paged.cuh"
#include "flash_prefill_paged_fp8.cuh"
#include "flash_decode_paged.cuh"
#include "flash_decode_paged_fp8.cuh"
#include "flash_reshape_cache.cuh"

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
#undef fp8_to_bf16_s
#undef fp8_to_f32_d
#undef unpack2_bf16_d
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
#define fp8_to_bf16_s fp8_to_bf16_s_512
#define fp8_to_f32_d fp8_to_f32_d_512
#define unpack2_bf16_d unpack2_bf16_d_512

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
#undef fp8_to_bf16_s
#undef fp8_to_f32_d
#undef unpack2_bf16_d

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
        LAUNCH_PREFILL(256, 128, 0);
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
        LAUNCH_PREFILL_FP8(256, 128, 0);
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
