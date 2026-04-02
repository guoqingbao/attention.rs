/**
 * @brief Optimized CUDA kernel for MoE GEMV (General Matrix-Vector Multiplication)
 * for the decode phase.
 *
 * This kernel is optimized for small batch sizes (M <= 8, typically M = 1 for decode).
 * Based on llama.cpp's approach, it uses warp-level reductions instead of tensor cores,
 * which provides better performance for small batches due to lower overhead.
 *
 * @details
 * - Each CUDA block computes ONE output element for ONE token
 * - Grid configuration: (N, M) where N = output dimension, M = num_tokens
 * - Uses warp-level reductions via __shfl_xor_sync
 * - Minimal shared memory usage (32 bytes for 8 warps)
 * - Vectorized loads using half2/bfloat162 for memory bandwidth
 */

#include "moe/moe_utils.cuh"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include "attention/attention_dtypes.h"
#include "attention/dtype_fp8.cuh"

namespace vllm {

inline __device__ void from_float(half& dst, float src) {
  dst = static_cast<half>(float_to_half(src));
}

inline __device__ float to_float(half u) {
  return half_to_float(static_cast<uint16_t>(u));
}
}

namespace vllm_rs {

// Warp reduction sum using shuffle instructions
template <int WARP_SIZE = 32>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return x;
}

inline __device__ void zero(__nv_bfloat162& dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __ushort_as_bfloat16((unsigned short)0x0000U);
  dst.y = dst.x;
#endif
}
inline __device__ void zero(half2& dst) {
  dst.x = __half_as_ushort(__float2half(0));
  dst.y = __half_as_ushort(__float2half(0));
}

} // namespace vllm_rs

/**
 * @brief MoE GEMV kernel for standard weight layout [E, N, K].
 *
 * Optimized version using:
 * - float4 loads (128-bit, 8 half values at once) for better memory bandwidth
 * - __hfma2 for native half2 fused multiply-add accumulation
 * - Only converts to float at the end for reduction
 *
 * @tparam T Data type: half or nv_bfloat16
 * @tparam BLOCK_SIZE Number of threads per block (default 256 = 8 warps)
 *
 * @param input             [M, K] - Input activations for all tokens
 * @param weights           [num_experts, N, K] - Expert weight matrices
 * @param sorted_token_ids  [M] - Indices of tokens sorted by expert assignment
 * @param expert_ids        [M] - Expert ID for each token
 * @param topk_weights      [M] (optional) - Per-token gating weights (nullptr if
 * not used)
 * @param output            [M, N] - Output activations for all tokens
 * @param num_experts       Total number of experts
 * @param topk              Number of experts selected per token
 * @param M                 Number of tokens (work items)
 * @param N                 Output dimension per expert
 * @param K                 Input dimension per expert
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_kernel(
    const T *__restrict__ input,                  // [M, K]
    const T *__restrict__ weights,                // [num_experts, N, K]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights, // [M] optional, can be nullptr
    T *__restrict__ output,                 // [M, N]
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  // blockIdx.x = output row (N dimension)
  // blockIdx.y = token index
  const int row = blockIdx.x;
  const int token_idx = blockIdx.y;

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  // Get input and weight pointers
  // If topk_weights is provided, tokens are NOT replicated (one entry per
  // token) If topk_weights is nullptr, tokens are replicated topk times
  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  const T *weight_row = weights + (size_t)expert * N * K + (size_t)row * K;

  const int tid = threadIdx.x;

  // Use float4 for 128-bit loads (8 elements per load for half/bf16)
  // This provides better memory bandwidth than smaller loads
  constexpr int LOAD_VEC_SIZE = 8; // 8 half/bf16 values = 16 bytes = float4
  const int k_vec = K / LOAD_VEC_SIZE;

  const float4 *in_vec = reinterpret_cast<const float4 *>(input_row);
  const float4 *w_vec = reinterpret_cast<const float4 *>(weight_row);

  // Use the appropriate vector type for the data type
  using Vec2T =
      typename std::conditional<std::is_same<T, half>::value, half2,
                                nv_bfloat162>::type;

  float sum = 0.0f;

  #ifndef NO_BF16_KERNEL
    __nv_bfloat162 prod;
    vllm_rs::zero(prod);
  #endif
  // Main vectorized loop - process 8 elements at a time
  for (int k = tid; k < k_vec; k += BLOCK_SIZE) {
    float4 in_val = in_vec[k];
    float4 w_val = w_vec[k];

    // Reinterpret as 4 half2/bfloat162 pairs and accumulate
    const Vec2T *in_v2 = reinterpret_cast<const Vec2T *>(&in_val);
    const Vec2T *w_v2 = reinterpret_cast<const Vec2T *>(&w_val);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Use native vector multiply, then convert to float for accumulation
      // For half2: uses __hmul2, for bfloat162: uses equivalent intrinsics
      if constexpr (std::is_same<T, half>::value) {
        float2 in_f = __half22float2(in_v2[i]);
        float2 w_f = __half22float2(w_v2[i]);
        sum = fmaf(in_f.x, w_f.x, sum);
        sum = fmaf(in_f.y, w_f.y, sum);
      } else {
        // For BF16, multiply in bf16 and accumulate in f32.
#ifndef NO_BF16_KERNEL
        prod = __hadd2(__hmul2(in_v2[i], w_v2[i]), prod);
#endif
      }
    }
  }

  #ifndef NO_BF16_KERNEL
    float2 f = vllm::bf1622float2(prod);
    sum += f.x + f.y;
  #endif
  // Handle remainder if K is not divisible by LOAD_VEC_SIZE
  const int remainder_start = k_vec * LOAD_VEC_SIZE;
  for (int k = remainder_start + tid; k < K; k += BLOCK_SIZE) {
    sum = __fmaf_rn(vllm::to_float(input_row[k]), vllm::to_float(weight_row[k]),
                    sum);
  }

  // Warp-level reduction
  sum = vllm_rs::warp_reduce_sum(sum);

  // Inter-warp reduction using shared memory
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) {
    smem[warp_id] = sum;
  }
  __syncthreads();

  // Final reduction in the first warp
  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;

// Reduce across the first warp
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // Thread 0 writes the final result
    if (lane_id == 0) {
      if (topk_weights) {
        sum *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, sum);
      output[(size_t)token_id * N + row] = out_val;
    }
  }
}

/**
 * @brief MoE GEMV kernel for transposed weight layout [E, K, N].
 *
 * Same algorithm as moe_gemv_kernel but with different weight access pattern.
 * For transposed layout, weights have stride N so vectorized weight loads
 * aren't possible, but we still use __fmaf_rn for better accumulation.
 *
 * @param weights [num_experts, K, N] - Expert weight matrices (transposed)
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_transposed_kernel(
    const T *__restrict__ input,   // [M, K]
    const T *__restrict__ weights, // [num_experts, K, N] - transposed layout
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights, // [M] optional, can be nullptr
    T *__restrict__ output,                 // [M, N]
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  const int row = blockIdx.x;       // Output N dimension
  const int token_idx = blockIdx.y; // Token index

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  // For transposed layout [E, K, N]: weight[k, n] = weights[expert * K * N + k
  // * N + n]
  const T *weight_expert = weights + (size_t)expert * K * N;

  float sum = 0.0f;
  const int tid = threadIdx.x;

  // For transposed layout, weights are accessed with stride N
  // This is less efficient for memory coalescing, but still faster than
  // moe_gemm for small M. Use __fmaf_rn for fused multiply-add.
  for (int k = tid; k < K; k += BLOCK_SIZE) {
    // weight[k, row] = weight_expert[k * N + row]
    sum = __fmaf_rn(vllm::to_float(input_row[k]),
                    vllm::to_float(weight_expert[(size_t)k * N + row]), sum);
  }

  // Warp-level reduction
  sum = vllm_rs::warp_reduce_sum(sum);

  // Inter-warp reduction
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) {
    smem[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;

#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
      if (topk_weights) {
        sum *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, sum);
      output[(size_t)token_id * N + row] = out_val;
    }
  }
}

extern "C" void moe_gemv(
    const void *input,   // input [size_m or size_m / topk, size_k]
    const void *weights, // weights [num_experts, size_n, size_k]
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights, // device ptr or nullptr
    void *output,              // output [size_m, size_n]
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int dtype, // 0=float16, 1=bf16
    cudaStream_t stream) {

  constexpr int BLOCK_SIZE = 256;

  // Grid: (N, M) - one block per output element per token
  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_kernel<nv_bfloat16, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(input),
        reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
        expert_ids, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
        num_experts, topk, size_m, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv: unsupported dtype.\n");
  }
}

extern "C" void moe_gemv_transposed(
    const void *input, // input [size_m or size_m / topk, size_k]
    const void *weights, // weights [num_experts, size_k, size_n] - transposed layout
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights, // device ptr or nullptr
    void *output,              // output [size_m, size_n]
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int dtype, // 0=float16, 1=bf16
    cudaStream_t stream) {

  constexpr int BLOCK_SIZE = 256;

  // Grid: (N, M) - one block per output element per token
  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_transposed_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_transposed_kernel<nv_bfloat16, BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_ids, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
            num_experts, topk, size_m, size_n, size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv_transposed: unsupported dtype.\n");
  }
}

#define CEILDIV(x,y) (((x) + (y) - 1) / (y))


/**
 * @brief MoE GEMV kernel for FP8 weights with block-wise scales.
 *
 * Uses uint8_t FP8 weights and converts to T (half/bf16) using scaled_convert.
 * Each block computes one output element for one token.
 *
 * @tparam T Data type for input/output: half or nv_bfloat16
 * @tparam BLOCK_SIZE Number of threads per block
 */
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_kernel_fp8(
    const T *__restrict__ input,                  // [M, K]
    const uint8_t *__restrict__ weights,          // [num_experts, N, K] FP8
    const float *__restrict__ weight_scales,      // [num_experts, scale_n_dim, scale_k_dim]
    const int32_t *__restrict__ sorted_token_ids, // [M]
    const int32_t *__restrict__ expert_ids,       // [M]
    const float *__restrict__ topk_weights,       // [M] optional, can be nullptr
    T *__restrict__ output,                       // [M, N]
    const int num_experts, const int topk, const int M, const int N,
    const int K, const int block_size_n, const int block_size_k) {
  
  const int row = blockIdx.x;         // N dimension
  const int token_idx = blockIdx.y;   // Token index

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  // Get input pointer
  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  
  // FP8 weight row for this expert and output row
  const uint8_t *weight_row = weights + (size_t)expert * N * K + (size_t)row * K;
  
  // Scale layout: [num_experts, scale_n_dim, scale_k_dim]
  const int scale_n_dim = CEILDIV(N, block_size_n);
  const int scale_k_dim = CEILDIV(K, block_size_k);
  const float *expert_scales = weight_scales + (size_t)expert * scale_n_dim * scale_k_dim;
  const int scale_n_idx = row / block_size_n;

  const int tid = threadIdx.x;

  float sum = 0.0f;

  // Process 4 FP8 values at a time using uint32_t loads
  const int k_vec4 = K / 4;
  for (int k = tid; k < k_vec4; k += BLOCK_SIZE) {
    int k_base = k * 4;
    
    // Load 4 FP8 weights as uint32_t
    uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(&weight_row[k_base]));
    
    // Get scale for this K block
    int scale_k_idx = k_base / block_size_k;
    float scale = expert_scales[scale_n_idx * scale_k_dim + scale_k_idx];
    
    uint8_t w0 = (w4 >> 0) & 0xFF;
    uint8_t w1 = (w4 >> 8) & 0xFF;
    uint8_t w2 = (w4 >> 16) & 0xFF;
    uint8_t w3 = (w4 >> 24) & 0xFF;

    float4 w = make_float4(
        vllm::fp8::dispatch_fp8_to_float(w0),
        vllm::fp8::dispatch_fp8_to_float(w1),
        vllm::fp8::dispatch_fp8_to_float(w2),
        vllm::fp8::dispatch_fp8_to_float(w3));

    float2 i01;
    float2 i23;
    if (std::is_same<T, half>::value) {
      const __half2* in2 = reinterpret_cast<const __half2*>(&input_row[k_base]);
      i01 = __half22float2(in2[0]);
      i23 = __half22float2(in2[1]);
    } else {
#ifndef NO_BF16_KERNEL
      const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(&input_row[k_base]);
      i01 = vllm::bf1622float2(in2[0]);
      i23 = vllm::bf1622float2(in2[1]);
#endif
    }

    float partial_sum = 0.0f;
    partial_sum = fmaf(i01.x, w.x, partial_sum);
    partial_sum = fmaf(i01.y, w.y, partial_sum);
    partial_sum = fmaf(i23.x, w.z, partial_sum);
    partial_sum = fmaf(i23.y, w.w, partial_sum);

    sum += scale * partial_sum;
  }

  // Handle remainder
  const int remainder_start = k_vec4 * 4;
  for (int k = remainder_start + tid; k < K; k += BLOCK_SIZE) {
    uint8_t w = weight_row[k];
    int scale_k_idx = k / block_size_k;
    float scale = expert_scales[scale_n_idx * scale_k_dim + scale_k_idx];
    float wf = vllm::fp8::dispatch_fp8_to_float(w) * scale;
    sum += vllm::to_float(input_row[k]) * wf;
  }

  // Warp-level reduction
  sum = vllm_rs::warp_reduce_sum(sum);

  // Inter-warp reduction using shared memory
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float smem[NUM_WARPS];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) {
    smem[warp_id] = sum;
  }
  __syncthreads();

  // Final reduction in the first warp
  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;

#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // Thread 0 writes the final result
    if (lane_id == 0) {
      if (topk_weights) {
        sum *= topk_weights[token_id];
      }
      T out_val;
      vllm::from_float(out_val, sum);
      output[(size_t)token_id * N + row] = out_val;
    }
  }
}

extern "C" void moe_gemv_fp8(
    const void *input,                // input [size_m or size_m / topk, size_k]
    const uint8_t *weights,           // weights [num_experts, size_n, size_k] FP8
    const float *weight_scales,       // [num_experts, scale_n_dim, scale_k_dim]
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,        // device ptr or nullptr
    void *output,                     // output [size_m, size_n]
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int block_size_n,
    int block_size_k,
    int dtype,                        // 0=float16, 1=bf16 (for input/output)
    cudaStream_t stream) {

  constexpr int BLOCK_SIZE = 256;

  // Grid: (N, M) - one block per output element per token
  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) { // FP16
    moe_gemv_kernel_fp8<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        weights, weight_scales, sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k, block_size_n, block_size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) { // BF16
    moe_gemv_kernel_fp8<nv_bfloat16, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(input),
        weights, weight_scales, sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<nv_bfloat16 *>(output), num_experts, topk,
        size_m, size_n, size_k, block_size_n, block_size_k);
  }
#endif
  else {
    fprintf(stderr, "moe_gemv_fp8: unsupported dtype.\n");
  }
}