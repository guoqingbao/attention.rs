/* ldg_vec fix: support VEC_U32 > 2 for HDIM > 128 */
/**
 * @brief BF16 paged decode kernel instantiation for native flash attention.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_decode.cu
 *
 * @details
 * Instantiates BF16 paged decode kernels (main + split-K + reduce) for HDIM
 * 128/256/512. Each CTA handles one Q head; GQA kv_head mapping is computed
 * inside the kernel. 8 warps split the KV sequence with online softmax and
 * batched (BC=4) score/V accumulation for reduced __expf overhead.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
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

// ============================================================================
// HDIM=256, GQA_RATIO=1
// ============================================================================
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

// ============================================================================
// HDIM=512, GQA_RATIO=1
// ============================================================================
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

// ============================================================================
// BF16 Decode Dispatcher — per-Q-head, no GQA grouping
// ============================================================================

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
    unsigned int gqa_ratio,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    #define DO_LAUNCH(HD) \
        flash_decode_paged_##HD<<<grid, 256, 0, s>>>( \
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
    unsigned int sliding_window,
    unsigned int gqa_ratio,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_splits, num_seqs);

    #define DO_LAUNCH_SK(HD) \
        flash_decode_paged_splitk_##HD<<<grid, 256, 0, s>>>( \
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

extern "C" void call_flash_decode_paged_reduce(
    const void* workspace, void* O,
    unsigned int num_q_heads, unsigned int head_dim,
    unsigned int num_splits, unsigned int num_seqs,
    int64_t stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(num_q_heads, num_seqs);

    if (head_dim <= 128)      flash_decode_paged_reduce_128<<<grid, 32, 0, s>>>((const float*)workspace, (flash_half_t*)O, num_q_heads, head_dim, num_splits);
    else if (head_dim <= 256) flash_decode_paged_reduce_256<<<grid, 32, 0, s>>>((const float*)workspace, (flash_half_t*)O, num_q_heads, head_dim, num_splits);
    else                      flash_decode_paged_reduce_512<<<grid, 32, 0, s>>>((const float*)workspace, (flash_half_t*)O, num_q_heads, head_dim, num_splits);
}
