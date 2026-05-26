/**
 * @brief SM90+ instantiation file for wgmma and tcgen05-based Flash Attention kernels.
 *
 * Conditionally compiled only when FLASH_WGMMA_ENABLED (SM90) or DFLASH_TCGEN05_ENABLED (SM100+) is defined .
 * Instantiates the large-tile prefill kernel with wgmma or tcgen05 for HDIM 128/256.
 * The FP8 variant uses the same kernel with fp8→bf16 dequant in shared memory.
 *
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 */

#include <cuda_runtime.h>
#include "flash_sm_compat.cuh"

#ifdef FLASH_WGMMA_ENABLED

#include <cuda_fp8.h>

// ============================================================================
// HDIM=128 wgmma prefill
// ============================================================================
#define FLASH_HDIM 128
#define flash_prefill_wgmma flash_prefill_wgmma_128

#include "flash_prefill_wgmma.cuh"

#undef FLASH_HDIM
#undef flash_prefill_wgmma
#undef WGMMA_BR
#undef WGMMA_BC
#undef WGMMA_NUM_THREADS
#undef WGMMA_NUM_WARPS
#undef WGMMA_K_STAGES
#undef WGMMA_HDIM
#undef WGMMA_HDIM_PAD
#undef WGMMA_Q_CHUNKS
#undef WGMMA_KV_CHUNKS
#undef WGMMA_N_TILES_PER_WARP
#undef SMEM_K_WGMMA
#undef LOAD_KV_TILE_WGMMA

// ============================================================================
// HDIM=256 wgmma prefill
// ============================================================================
#define FLASH_HDIM 256
#define flash_prefill_wgmma flash_prefill_wgmma_256

#include "flash_prefill_wgmma.cuh"

#undef FLASH_HDIM
#undef flash_prefill_wgmma
#undef WGMMA_BR
#undef WGMMA_BC
#undef WGMMA_NUM_THREADS
#undef WGMMA_NUM_WARPS
#undef WGMMA_K_STAGES
#undef WGMMA_HDIM
#undef WGMMA_HDIM_PAD
#undef WGMMA_Q_CHUNKS
#undef WGMMA_KV_CHUNKS
#undef WGMMA_N_TILES_PER_WARP
#undef SMEM_K_WGMMA
#undef LOAD_KV_TILE_WGMMA

// ============================================================================
// SM90 wgmma prefill launcher
// ============================================================================

extern "C" void call_flash_prefill_wgmma(
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
    const unsigned int br = 64; // WGMMA_BR
    dim3 grid(num_q_heads, (max_q_len + br - 1) / br, num_seqs);
    const unsigned int threads = 128; // WGMMA_NUM_THREADS

    // Shared memory calculation:
    // Q: BR * HDIM_PAD * sizeof(bf16) = 64 * (HDIM+8) * 2
    // K: 2 * BC * HDIM_PAD * sizeof(bf16) = 2 * 64 * (HDIM+8) * 2
    // V: BC * HDIM_PAD * sizeof(bf16) = 64 * (HDIM+8) * 2
    // P: BR * (BC+8) * sizeof(bf16) = 64 * 72 * 2
    // ml: BR * 4 * sizeof(float) = 64 * 4 * 4
    // Total for HDIM=128: 64*136*2 + 2*64*136*2 + 64*136*2 + 64*72*2 + 64*4*4
    //                    = 17408 + 34816 + 17408 + 9216 + 1024 = 79872 bytes

    if (head_dim <= 128) {
        unsigned int hdim_pad = 128 + 8;
        unsigned int smem = 64 * hdim_pad * 2       // Q
                          + 2 * 64 * hdim_pad * 2   // K (double-buffered)
                          + 64 * hdim_pad * 2       // V
                          + 64 * 72 * 2             // P
                          + 64 * 4 * 4;             // ml
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_wgmma_128,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        flash_prefill_wgmma_128<<<grid, threads, smem, s>>>(
            (const flash_half_t*)Q, (const flash_half_t*)K_cache,
            (const flash_half_t*)V_cache, (flash_half_t*)O,
            block_tables, block_table_stride, cu_seqlens_q, context_lens,
            num_q_heads, num_kv_heads,
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap);
    } else {
        unsigned int hdim_pad = 256 + 8;
        unsigned int smem = 64 * hdim_pad * 2
                          + 2 * 64 * hdim_pad * 2
                          + 64 * hdim_pad * 2
                          + 64 * 72 * 2
                          + 64 * 4 * 4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_wgmma_256,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        flash_prefill_wgmma_256<<<grid, threads, smem, s>>>(
            (const flash_half_t*)Q, (const flash_half_t*)K_cache,
            (const flash_half_t*)V_cache, (flash_half_t*)O,
            block_tables, block_table_stride, cu_seqlens_q, context_lens,
            num_q_heads, num_kv_heads,
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap);
    }
}

#endif // FLASH_WGMMA_ENABLED

// ============================================================================
// SM100/SM120+ tcgen05 prefill kernels
// ============================================================================
#ifdef FLASH_TCGEN05_ENABLED

// HDIM=128 tcgen05 prefill
#define FLASH_HDIM 128
#define flash_prefill_tcgen05 flash_prefill_tcgen05_128

#include "flash_prefill_tcgen05.cuh"

#undef FLASH_HDIM
#undef flash_prefill_tcgen05
#undef TCGEN_BR
#undef TCGEN_BC
#undef TCGEN_NUM_THREADS
#undef TCGEN_NUM_WARPS
#undef TCGEN_K_STAGES
#undef TCGEN_HDIM
#undef TCGEN_HDIM_PAD
#undef TCGEN_Q_CHUNKS
#undef TCGEN_KV_CHUNKS
#undef TCGEN_N_TILES_PER_WARP
#undef SMEM_K_TCGEN
#undef LOAD_KV_TILE_TCGEN

// HDIM=256 tcgen05 prefill
#define FLASH_HDIM 256
#define flash_prefill_tcgen05 flash_prefill_tcgen05_256

#include "flash_prefill_tcgen05.cuh"

#undef FLASH_HDIM
#undef flash_prefill_tcgen05
#undef TCGEN_BR
#undef TCGEN_BC
#undef TCGEN_NUM_THREADS
#undef TCGEN_NUM_WARPS
#undef TCGEN_K_STAGES
#undef TCGEN_HDIM
#undef TCGEN_HDIM_PAD
#undef TCGEN_Q_CHUNKS
#undef TCGEN_KV_CHUNKS
#undef TCGEN_N_TILES_PER_WARP
#undef SMEM_K_TCGEN
#undef LOAD_KV_TILE_TCGEN

// SM100+ tcgen05 prefill launcher
extern "C" void call_flash_prefill_tcgen05(
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
    const unsigned int br = 128; // TCGEN_BR
    dim3 grid(num_q_heads, (max_q_len + br - 1) / br, num_seqs);
    const unsigned int threads = 128; // TCGEN_NUM_THREADS

    // Shared memory for SM100+ (larger BC=128):
    // Q: BR * HDIM_PAD * 2
    // K: 2 * BC * HDIM_PAD * 2
    // V: BC * HDIM_PAD * 2
    // P: BR * (BC+8) * 2
    // ml: BR * 4 * 4

    if (head_dim <= 128) {
        unsigned int hdim_pad = 128 + 8;
        unsigned int smem = 128 * hdim_pad * 2       // Q
                          + 2 * 128 * hdim_pad * 2   // K (double-buffered)
                          + 128 * hdim_pad * 2       // V
                          + 128 * 136 * 2            // P (BC+8=136)
                          + 128 * 4 * 4;             // ml
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_tcgen05_128,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        flash_prefill_tcgen05_128<<<grid, threads, smem, s>>>(
            (const flash_half_t*)Q, (const flash_half_t*)K_cache,
            (const flash_half_t*)V_cache, (flash_half_t*)O,
            block_tables, block_table_stride, cu_seqlens_q, context_lens,
            num_q_heads, num_kv_heads,
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap);
    } else {
        unsigned int hdim_pad = 256 + 8;
        unsigned int smem = 128 * hdim_pad * 2
                          + 2 * 128 * hdim_pad * 2
                          + 128 * hdim_pad * 2
                          + 128 * 136 * 2
                          + 128 * 4 * 4;
        smem = (smem + 255) & ~255u;
        cudaFuncSetAttribute(flash_prefill_tcgen05_256,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        flash_prefill_tcgen05_256<<<grid, threads, smem, s>>>(
            (const flash_half_t*)Q, (const flash_half_t*)K_cache,
            (const flash_half_t*)V_cache, (flash_half_t*)O,
            block_tables, block_table_stride, cu_seqlens_q, context_lens,
            num_q_heads, num_kv_heads,
            head_dim, cache_block_size, sliding_window, causal, inv_sqrt_d, softcap);
    }
}

#endif // FLASH_TCGEN05_ENABLED
