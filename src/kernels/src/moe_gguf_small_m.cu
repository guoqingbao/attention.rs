/**
 * @brief Optimized CUDA kernel for Mixture-of-Experts (MoE) GEMM for small M (batch size 1-8).
 *
 * This kernel is optimized for decoding scenarios with small batch sizes where
 * caching the input vector in shared memory is more efficient than caching weights.
 *
 * Strategy:
 * 1. Cooperative Load: Cache reusable Input (y) in Shared Memory.
 * 2. Streaming: Stream Weights (w) directly from Global Memory (L2) to Registers.
 * 3. Warp Reduction: Fast parallel summation.
 *
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
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
#include "gguf/gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <cassert>
#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"

constexpr int MATRIX_ROW_PADDING_SMALL_M = 512;

constexpr int pad_small_m(int size, int padding) {
    if (padding == 0) return size;
    return ((size + padding - 1) / padding) * padding;
}

constexpr int ceil_div_small_m(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * Optimized MoE GEMM kernel for small M (batch size ~1-8)
 *
 * Template Parameters:
 * @tparam qk             Quantization block size for weights (e.g., 32, 256)
 * @tparam qi             Number of int values per quantized weight block
 * @tparam block_q_t      Type of quantized weight block (e.g., block_q8_0, block_q4_K)
 * @tparam vdr            Vectorization factor
 * @tparam vec_dot_q_cuda Function for computing vectorized dot-product
 *
 * Kernel Parameters:
 * @param all_weights         Pointer to all expert weight matrices [num_experts, N, K]
 * @param all_inputs          Pointer to all input tokens [M_total, K] (quantized Q8_1)
 * @param sorted_token_ids    Sorted token indices for batch processing
 * @param expert_ids          Expert ID for each token
 * @param topk_weights        Optional top-k MoE weight per token
 * @param all_outputs         Output buffer [M_total, N] (float)
 * @param num_experts         Number of experts
 * @param topk                Top-k experts selected per token
 * @param size_m              Number of tokens processed (M dimension)
 * @param size_n              Output feature dimension (N dimension)
 * @param size_k              Input feature dimension (K dimension)
 * @param k_padded            Padded K dimension for GGUF stride
 */
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void moe_gemm_gguf_small_m_kernel(
    const void * __restrict__ all_weights,       // [num_experts, N, K] (quantized)
    const void * __restrict__ all_inputs,        // [M_total, K] (quantized Q8_1)
    const int32_t* __restrict__ sorted_token_ids,// [M] 
    const int32_t* __restrict__ expert_ids,      // [M]
    const float* __restrict__ topk_weights,      // [M]
    float * __restrict__ all_outputs,            // [M_total, N] (float)
    int num_experts,
    int topk,
    int size_m, int size_n, int size_k, 
    int k_padded 
) {
    // --- 1. Addressing & Setup ---
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int m_idx  = blockIdx.y; // Token index (Batch dimension)
    
    // Bounds check for the batch dimension (Whole block works on one token)
    if (m_idx >= size_m) return;

    // Identify Token and Expert
    const int token_id = sorted_token_ids[m_idx];
    const int expert = expert_ids[m_idx];

    // Safety: If expert is invalid, the whole block exits
    if (expert < 0 || expert >= num_experts) return;

    // Setup Memory Pointers
    const size_t weight_expert_stride_bytes = (size_t)(size_n * size_k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    
    // Base pointer for this expert's weights
    const block_q_t * __restrict__ w_expert_base = 
        (const block_q_t *)((const char *)all_weights + (size_t)expert * weight_expert_stride_bytes);

    // Pointer for this token's input
    const int input_index = topk_weights ? token_id : (token_id / topk);
    const block_q8_1 * __restrict__ y_src = 
        (const block_q8_1 *)((const char *)all_inputs + (size_t)input_index * input_task_stride_bytes);

    // --- 2. Cooperative Input Loading (Input Caching) ---
    // Allocate shared memory for the Input Vector (y)
    extern __shared__ char shared_mem[];
    block_q8_1* y_shared = (block_q8_1*)shared_mem;

    const int num_k_blocks = k_padded / QK8_1;
    const int tid_in_block = wrapId * blockDim.x + laneId; // Flat thread ID
    const int total_threads = blockDim.y * blockDim.x;

    // Parallel copy from Global -> Shared
    for (int i = tid_in_block; i < num_k_blocks; i += total_threads) {
        y_shared[i] = y_src[i];
    }

    // BARRIER: Ensure Input is fully loaded before any math starts
    __syncthreads(); 

    // --- 3. Compute Phase (Streaming Weights) ---
    // Each Warp calculates one Output Row (N dimension)
    
    // Calculate global output row index
    const int row = blockIdx.x * blockDim.y + wrapId;

    if (row < size_n) {
        // Get the scaling factor for this token/expert pair
        const float scale = (topk_weights) ? topk_weights[token_id] : 1.0f;

        // Pointer to the specific row of weights
        const int blocks_per_row = size_k / qk;
        const block_q_t * __restrict__ w_row = w_expert_base + row * blocks_per_row;

        float acc = 0.0f;

        // Loop Setup (Standard GGML/GGUF Stride Logic)
        const int blocks_per_iter = vdr * WARP_SIZE / qi;
        const int k_start = laneId / (qi / vdr);

        // Main Dot Product Loop
        // Weights are read from Global (Streaming), Inputs from Shared (Cached)
        #pragma unroll 4
        for (int kbx = k_start; kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kby = kbx * (qk / QK8_1); // Index for Input (different block size)
            const int kqs = vdr * (laneId % (qi / vdr)); 

            acc += vec_dot_q_cuda(
                &w_row[kbx],    // Global Memory Read (Cached in L2)
                &y_shared[kby], // Shared Memory Read (Fast)
                kqs
            );
        }

        // --- 4. Warp Reduction ---
        // Sum partial results across the warp
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }

        // --- 5. Output Write ---
        if (laneId == 0) {
            const size_t output_task_stride_elems = (size_t)size_n;
            float * __restrict__ out_ptr = all_outputs + ((size_t)token_id) * output_task_stride_elems;
            out_ptr[row] = acc * scale;
        }
    }
}

// Launch macro for different quantization types
#define LAUNCH_MOE_GGUF_SMALL_M(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    /* Calculate Shared Memory needed for the Input Vector (y) */ \
    moe_gemm_gguf_small_m_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
        <<<grid_dim, block_dim, shared_bytes, stream>>>(\
        weights, y_q8_1,\
        sorted_token_ids, expert_ids, topk_weights,\
        outputs,\
        num_experts, topk,\
        size_m, size_n, size_k,\
        kx_padded\
    );\


extern "C" void moe_gemm_gguf_small_m(
    const float* inputs, //must be float
    const void* weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    float* outputs,
    int num_experts,
    int topk,
    int size_m,         // M (num tokens to process)
    int size_n,         // N (output dim)
    int size_k,         // K (input dim)
    int quant_type,     // Q8_0: 0, Q4K: 1, etc.
    cudaStream_t stream
) {
    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int kx_padded = pad_small_m(size_k, MATRIX_ROW_PADDING_SMALL_M);
    const int num_blocks = ceil_div_small_m(kx_padded, QUANTIZE_BLOCK_SIZE);
    
    // TopK adjustment for M dimension logic
    int m = topk_weights ? size_m : size_m / topk;

    // 1. Quantize Inputs (Float -> Q8_1)
    dim3 grid_dim_quant(num_blocks, m, 1);
    dim3 block_dim_quant(QUANTIZE_BLOCK_SIZE, 1, 1);
    int y_size_in_bytes = m * (kx_padded / QK8_1 * sizeof(block_q8_1));
    
    void* y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(inputs, y_q8_1, size_k, kx_padded);

    // 2. Compute GEMM
    const int nWraps = 4; // Warps per block
    dim3 grid_dim(ceil_div_small_m(size_n, nWraps), size_m, 1);
    dim3 block_dim(WARP_SIZE, nWraps, 1);
    const int shared_bytes = (kx_padded / QK8_1) * sizeof(block_q8_1); \

    // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3, Q5K: 4, Q6K: 5
    switch (quant_type) {
        case 0: // Q8_0
            LAUNCH_MOE_GGUF_SMALL_M(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1);
            break;
        case 1: // Q4K
            LAUNCH_MOE_GGUF_SMALL_M(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1);
            break;
        case 2: // Q2_K
            LAUNCH_MOE_GGUF_SMALL_M(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1);
            break;
        case 3: // Q3_K
            LAUNCH_MOE_GGUF_SMALL_M(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1);
            break;
        case 4: // Q5_K
            LAUNCH_MOE_GGUF_SMALL_M(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1);
            break;
        case 5: // Q6K
            LAUNCH_MOE_GGUF_SMALL_M(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1);
            break;
        default:
            break;
    }
    
    cudaFreeAsync(y_q8_1, stream);
}
