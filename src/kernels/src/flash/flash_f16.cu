/**
 * @brief SM80+ native flash F16 instantiations (alongside BF16 in flash_decode.cu /
 * flash_instantiate.cu).
 *
 * On SM70/75 (NO_BF16_KERNEL), native flash is already F16 — the *_f16 entry
 * points forward to the existing launchers so symbols always link.
 */

#include <cuda_runtime.h>
#include <cstdio>

#ifdef NO_BF16_KERNEL

// ---------------------------------------------------------------------------
// SM70/75: native flash is already F16 — alias the F16 API to the default one.
// ---------------------------------------------------------------------------

extern "C" void call_flash_decode_paged(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d, unsigned int num_seqs, unsigned int q_stride,
    unsigned int sliding_window, float softcap, unsigned int gqa_ratio,
    int64_t stream);

extern "C" void call_flash_decode_paged_splitk(
    const void* Q, const void* K_cache, const void* V_cache, void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size, float inv_sqrt_d,
    unsigned int num_seqs, unsigned int num_splits, unsigned int q_stride,
    float softcap, unsigned int sliding_window, unsigned int gqa_ratio,
    int64_t stream);

extern "C" void call_flash_decode_paged_reduce(
    const void* workspace, void* O,
    unsigned int num_q_heads, unsigned int head_dim,
    unsigned int num_splits, unsigned int num_seqs, int64_t stream);

extern "C" void call_flash_prefill_paged(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, unsigned int block_table_stride,
    const unsigned int* cu_seqlens_q, const unsigned int* context_lens,
    unsigned int num_seqs, unsigned int max_q_len,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    unsigned int sliding_window, unsigned int causal,
    float inv_sqrt_d, float softcap, int64_t stream);

extern "C" void call_flash_reshape_and_cache_bf16(
    const void* key, const void* value, void* key_cache, void* value_cache,
    const long long* slot_mapping,
    unsigned int num_tokens, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size, int64_t stream);

extern "C" void call_flash_decode_paged_f16(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d, unsigned int num_seqs, unsigned int q_stride,
    unsigned int sliding_window, float softcap, unsigned int gqa_ratio,
    int64_t stream) {
    call_flash_decode_paged(
        Q, K_cache, V_cache, O, block_tables, seq_lens, max_blocks_per_seq,
        num_q_heads, num_kv_heads, head_dim, block_size, inv_sqrt_d, num_seqs,
        q_stride, sliding_window, softcap, gqa_ratio, stream);
}

extern "C" void call_flash_decode_paged_splitk_f16(
    const void* Q, const void* K_cache, const void* V_cache, void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size, float inv_sqrt_d,
    unsigned int num_seqs, unsigned int num_splits, unsigned int q_stride,
    float softcap, unsigned int sliding_window, unsigned int gqa_ratio,
    int64_t stream) {
    call_flash_decode_paged_splitk(
        Q, K_cache, V_cache, workspace, block_tables, seq_lens,
        max_blocks_per_seq, num_q_heads, num_kv_heads, head_dim, block_size,
        inv_sqrt_d, num_seqs, num_splits, q_stride, softcap, sliding_window,
        gqa_ratio, stream);
}

extern "C" void call_flash_decode_paged_reduce_f16(
    const void* workspace, void* O,
    unsigned int num_q_heads, unsigned int head_dim,
    unsigned int num_splits, unsigned int num_seqs, int64_t stream) {
    call_flash_decode_paged_reduce(
        workspace, O, num_q_heads, head_dim, num_splits, num_seqs, stream);
}

extern "C" void call_flash_prefill_paged_f16(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, unsigned int block_table_stride,
    const unsigned int* cu_seqlens_q, const unsigned int* context_lens,
    unsigned int num_seqs, unsigned int max_q_len,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    unsigned int sliding_window, unsigned int causal,
    float inv_sqrt_d, float softcap, int64_t stream) {
    call_flash_prefill_paged(
        Q, K_cache, V_cache, O, block_tables, block_table_stride, cu_seqlens_q,
        context_lens, num_seqs, max_q_len, num_q_heads, num_kv_heads, head_dim,
        cache_block_size, sliding_window, causal, inv_sqrt_d, softcap, stream);
}

extern "C" void call_flash_reshape_and_cache_f16(
    const void* key, const void* value, void* key_cache, void* value_cache,
    const long long* slot_mapping,
    unsigned int num_tokens, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size, int64_t stream) {
    call_flash_reshape_and_cache_bf16(
        key, value, key_cache, value_cache, slot_mapping, num_tokens,
        num_kv_heads, head_dim, cache_block_size, stream);
}

#else // !NO_BF16_KERNEL — SM80+: real F16 Tensor Core instantiations

#define FLASH_FORCE_F16
#define FLASH_F16_RESHAPE_ONLY
#include "flash_sm_compat.cuh"

// ============================================================================
// Decode HDIM=128/256/512
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
#define flash_decode_paged flash_decode_paged_f16_128
#define flash_decode_paged_splitk flash_decode_paged_splitk_f16_128
#define flash_decode_paged_reduce flash_decode_paged_reduce_f16_128
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
// Keep FLASH_DECODE_UNPACK_DEFINED — unpack helper is dtype-agnostic via macros

#define FLASH_HDIM 256
#define GQA_RATIO 1
#define flash_decode_paged flash_decode_paged_f16_256
#define flash_decode_paged_splitk flash_decode_paged_splitk_f16_256
#define flash_decode_paged_reduce flash_decode_paged_reduce_f16_256
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
#define flash_decode_paged flash_decode_paged_f16_512
#define flash_decode_paged_splitk flash_decode_paged_splitk_f16_512
#define flash_decode_paged_reduce flash_decode_paged_reduce_f16_512
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

// ============================================================================
// Prefill HDIM=128/256/512
// ============================================================================
#define FLASH_HDIM 128
#define flash_prefill_paged flash_prefill_paged_f16_128
#include "flash_prefill_paged.cuh"
#undef FLASH_HDIM
#undef flash_prefill_paged
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
#undef LOAD_KV_TILE_BF16
#undef PREFILL_USE_DYNAMIC_SMEM
#undef BR_512
#undef BC_512
#undef PAD_P_512
#undef N_TILES_PER_WARP_512
#undef TILE_CHUNKS_512
#undef NUM_THREADS_512

#define FLASH_HDIM 256
#define flash_prefill_paged flash_prefill_paged_f16_256
#include "flash_prefill_paged.cuh"
#undef FLASH_HDIM
#undef flash_prefill_paged
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
#undef LOAD_KV_TILE_BF16
#undef PREFILL_USE_DYNAMIC_SMEM
#undef BR_512
#undef BC_512
#undef PAD_P_512
#undef N_TILES_PER_WARP_512
#undef TILE_CHUNKS_512
#undef NUM_THREADS_512

#define FLASH_HDIM 512
#define flash_prefill_paged flash_prefill_paged_f16_512
#include "flash_prefill_paged.cuh"
#undef FLASH_HDIM
#undef flash_prefill_paged
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
#undef LOAD_KV_TILE_BF16
#undef PREFILL_USE_DYNAMIC_SMEM
#undef BR_512
#undef BC_512
#undef PAD_P_512
#undef N_TILES_PER_WARP_512
#undef TILE_CHUNKS_512
#undef NUM_THREADS_512

// ============================================================================
// Reshape-and-cache (F16 → F16)
// ============================================================================
#define FLASH_HDIM 128
#define flash_reshape_and_cache flash_reshape_and_cache_f16_128
#include "flash_reshape_cache.cuh"
#undef FLASH_HDIM
#undef flash_reshape_and_cache
#undef HDIM
#undef WARP_SIZE

#define FLASH_HDIM 256
#define flash_reshape_and_cache flash_reshape_and_cache_f16_256
#include "flash_reshape_cache.cuh"
#undef FLASH_HDIM
#undef flash_reshape_and_cache
#undef HDIM
#undef WARP_SIZE

#define FLASH_HDIM 512
#define flash_reshape_and_cache flash_reshape_and_cache_f16_512
#include "flash_reshape_cache.cuh"
#undef FLASH_HDIM
#undef flash_reshape_and_cache
#undef HDIM
#undef WARP_SIZE

// ============================================================================
// Launchers
// ============================================================================

extern "C" void call_flash_decode_paged_f16(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d, unsigned int num_seqs, unsigned int q_stride,
    unsigned int sliding_window, float softcap, unsigned int gqa_ratio,
    int64_t stream
) {
    (void)gqa_ratio;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    #define DO_LAUNCH(HD) \
        flash_decode_paged_f16_##HD<<<grid, 256, 0, s>>>( \
            (const flash_half_t*)Q, (const flash_half_t*)K_cache, \
            (const flash_half_t*)V_cache, (flash_half_t*)O, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, q_stride, sliding_window, softcap)

    if (head_dim <= 128)      { DO_LAUNCH(128); }
    else if (head_dim <= 256) { DO_LAUNCH(256); }
    else                      { DO_LAUNCH(512); }
    #undef DO_LAUNCH
}

extern "C" void call_flash_decode_paged_splitk_f16(
    const void* Q, const void* K_cache, const void* V_cache,
    void* workspace,
    const int* block_tables, const int* seq_lens,
    unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d,
    unsigned int num_seqs, unsigned int num_splits,
    unsigned int q_stride, float softcap,
    unsigned int sliding_window,
    unsigned int gqa_ratio,
    int64_t stream
) {
    (void)gqa_ratio;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_splits, num_seqs);

    #define DO_LAUNCH_SK(HD) \
        flash_decode_paged_splitk_f16_##HD<<<grid, 256, 0, s>>>( \
            (const flash_half_t*)Q, (const flash_half_t*)K_cache, \
            (const flash_half_t*)V_cache, (float*)workspace, \
            block_tables, seq_lens, max_blocks_per_seq, \
            num_q_heads, num_kv_heads, head_dim, block_size, \
            inv_sqrt_d, num_splits, q_stride, softcap, sliding_window)

    if (head_dim <= 128)      { DO_LAUNCH_SK(128); }
    else if (head_dim <= 256) { DO_LAUNCH_SK(256); }
    else                      { DO_LAUNCH_SK(512); }
    #undef DO_LAUNCH_SK
}

extern "C" void call_flash_decode_paged_reduce_f16(
    const void* workspace, void* O,
    unsigned int num_q_heads, unsigned int head_dim,
    unsigned int num_splits, unsigned int num_seqs,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    if (head_dim <= 128)
        flash_decode_paged_reduce_f16_128<<<grid, 32, 0, s>>>(
            (const float*)workspace, (flash_half_t*)O, num_q_heads, head_dim, num_splits);
    else if (head_dim <= 256)
        flash_decode_paged_reduce_f16_256<<<grid, 32, 0, s>>>(
            (const float*)workspace, (flash_half_t*)O, num_q_heads, head_dim, num_splits);
    else
        flash_decode_paged_reduce_f16_512<<<grid, 32, 0, s>>>(
            (const float*)workspace, (flash_half_t*)O, num_q_heads, head_dim, num_splits);
}

extern "C" void call_flash_prefill_paged_f16(
    const void* Q, const void* K_cache, const void* V_cache, void* O,
    const int* block_tables, unsigned int block_table_stride,
    const unsigned int* cu_seqlens_q, const unsigned int* context_lens,
    unsigned int num_seqs, unsigned int max_q_len,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int cache_block_size,
    unsigned int sliding_window, unsigned int causal,
    float inv_sqrt_d, float softcap,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    unsigned int br = 32;
    dim3 grid(num_q_heads, (max_q_len + br - 1) / br, num_seqs);

    #define LAUNCH_PREFILL(HD, THREADS, SMEM) \
        flash_prefill_paged_f16_##HD<<<grid, THREADS, SMEM, s>>>( \
            (const flash_half_t*)Q, (const flash_half_t*)K_cache, \
            (const flash_half_t*)V_cache, (flash_half_t*)O, \
            block_tables, block_table_stride, cu_seqlens_q, context_lens, \
            num_q_heads, num_kv_heads, \
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap)

    if (head_dim <= 128) {
        LAUNCH_PREFILL(128, 128, 0);
    } else if (head_dim <= 256) {
        unsigned int smem = (32*264 + 2*32*264 + 32*264) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_paged_f16_256,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_PREFILL(256, 128, smem);
    } else {
        unsigned int smem = (32*512 + 32*512 + 32*512) * 2 + 32*40*2 + 32*2*4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_paged_f16_512,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        LAUNCH_PREFILL(512, 256, smem);
    }
    #undef LAUNCH_PREFILL
}

extern "C" void call_flash_reshape_and_cache_f16(
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
        flash_reshape_and_cache_f16_##HD<<<grid, threads, 0, s>>>( \
            (const flash_half_t*)key, (const flash_half_t*)value, \
            (flash_half_t*)key_cache, (flash_half_t*)value_cache, \
            slot_mapping, num_tokens, num_kv_heads, head_dim, cache_block_size)

    if (head_dim <= 128) { LAUNCH_CACHE(128); }
    else if (head_dim <= 256) { LAUNCH_CACHE(256); }
    else { LAUNCH_CACHE(512); }
    #undef LAUNCH_CACHE
}

#endif // NO_BF16_KERNEL
