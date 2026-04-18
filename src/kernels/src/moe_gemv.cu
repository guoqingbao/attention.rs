/**
 * @brief Optimized CUDA kernels for MoE GEMV (General Matrix-Vector Multiplication)
 * for the decode phase.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemv.cu
 *
 * @details
 * Multiple kernel strategies are provided:
 * - moe_gemv_kernel: standard bf16/f16 weights, one block per output element
 * - moe_gemv_kernel_fp8: FP8 weights with block-wise scales, warp-per-row design
 *   with shared memory input caching and 128-bit vectorized loads
 *
 * SM89+ (Hopper/Ada) uses hardware FP8 dequantization intrinsics.
 * SM100+ (Blackwell) uses __nv_cvt_fp8x2_to_halfraw2 for paired FP8 conversion.
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

// FP8 dequantization: converts 4 packed FP8 values (uint32) to 4 floats.
// On SM90+, uses __nv_cvt_fp8x2_to_halfraw2 for paired conversion.
// On SM80+ uses scalar __nv_cvt_fp8_to_halfraw.
// On older archs, uses software conversion.
__device__ __forceinline__ void fp8x4_to_float4(
    uint32_t packed, float &f0, float &f1, float &f2, float &f3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(NO_HARDWARE_FP8)
  // Blackwell: convert 2 FP8 values at a time
  __half2_raw pair0 = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(packed & 0xFFFF), __NV_E4M3);
  __half2_raw pair1 = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>((packed >> 16) & 0xFFFF), __NV_E4M3);
  f0 = __half2float(*reinterpret_cast<__half*>(&pair0.x));
  f1 = __half2float(*reinterpret_cast<__half*>(&pair0.y));
  f2 = __half2float(*reinterpret_cast<__half*>(&pair1.x));
  f3 = __half2float(*reinterpret_cast<__half*>(&pair1.y));
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(NO_HARDWARE_FP8)
  // Ampere/Ada: scalar conversion
  __half_raw h0 = __nv_cvt_fp8_to_halfraw((packed >>  0) & 0xFF, __NV_E4M3);
  __half_raw h1 = __nv_cvt_fp8_to_halfraw((packed >>  8) & 0xFF, __NV_E4M3);
  __half_raw h2 = __nv_cvt_fp8_to_halfraw((packed >> 16) & 0xFF, __NV_E4M3);
  __half_raw h3 = __nv_cvt_fp8_to_halfraw((packed >> 24) & 0xFF, __NV_E4M3);
  f0 = __half2float(*reinterpret_cast<__half*>(&h0));
  f1 = __half2float(*reinterpret_cast<__half*>(&h1));
  f2 = __half2float(*reinterpret_cast<__half*>(&h2));
  f3 = __half2float(*reinterpret_cast<__half*>(&h3));
#else
  f0 = vllm::fp8::softmax_fp8_to_float_e4m3((packed >>  0) & 0xFF);
  f1 = vllm::fp8::softmax_fp8_to_float_e4m3((packed >>  8) & 0xFF);
  f2 = vllm::fp8::softmax_fp8_to_float_e4m3((packed >> 16) & 0xFF);
  f3 = vllm::fp8::softmax_fp8_to_float_e4m3((packed >> 24) & 0xFF);
#endif
}

} // namespace vllm_rs

// ==========================================================================
// Standard bf16/f16 GEMV kernels
// ==========================================================================

template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_kernel(
    const T *__restrict__ input,
    const T *__restrict__ weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const float *__restrict__ topk_weights,
    T *__restrict__ output,
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  const int row = blockIdx.x;
  const int token_idx = blockIdx.y;

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  const T *weight_row = weights + (size_t)expert * N * K + (size_t)row * K;

  const int tid = threadIdx.x;

  constexpr int LOAD_VEC_SIZE = 8;
  const int k_vec = K / LOAD_VEC_SIZE;

  const float4 *in_vec = reinterpret_cast<const float4 *>(input_row);
  const float4 *w_vec = reinterpret_cast<const float4 *>(weight_row);

  using Vec2T =
      typename std::conditional<std::is_same<T, half>::value, half2,
                                nv_bfloat162>::type;

  float sum = 0.0f;

  #ifndef NO_BF16_KERNEL
    __nv_bfloat162 prod;
    vllm_rs::zero(prod);
  #endif
  for (int k = tid; k < k_vec; k += BLOCK_SIZE) {
    float4 in_val = in_vec[k];
    float4 w_val = w_vec[k];

    const Vec2T *in_v2 = reinterpret_cast<const Vec2T *>(&in_val);
    const Vec2T *w_v2 = reinterpret_cast<const Vec2T *>(&w_val);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (std::is_same<T, half>::value) {
        float2 in_f = __half22float2(in_v2[i]);
        float2 w_f = __half22float2(w_v2[i]);
        sum = fmaf(in_f.x, w_f.x, sum);
        sum = fmaf(in_f.y, w_f.y, sum);
      } else {
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
  const int remainder_start = k_vec * LOAD_VEC_SIZE;
  for (int k = remainder_start + tid; k < K; k += BLOCK_SIZE) {
    sum = __fmaf_rn(vllm::to_float(input_row[k]), vllm::to_float(weight_row[k]),
                    sum);
  }

  sum = vllm_rs::warp_reduce_sum(sum);

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

template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_transposed_kernel(
    const T *__restrict__ input,
    const T *__restrict__ weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const float *__restrict__ topk_weights,
    T *__restrict__ output,
    const int num_experts, const int topk, const int M, const int N,
    const int K) {
  const int row = blockIdx.x;
  const int token_idx = blockIdx.y;

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  const T *weight_expert = weights + (size_t)expert * K * N;

  float sum = 0.0f;
  const int tid = threadIdx.x;

  for (int k = tid; k < K; k += BLOCK_SIZE) {
    sum = __fmaf_rn(vllm::to_float(input_row[k]),
                    vllm::to_float(weight_expert[(size_t)k * N + row]), sum);
  }

  sum = vllm_rs::warp_reduce_sum(sum);

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
    const void *input,
    const void *weights,
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,
    void *output,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int dtype,
    cudaStream_t stream) {

  constexpr int BLOCK_SIZE = 256;

  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) {
    moe_gemv_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) {
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
    const void *input,
    const void *weights,
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,
    void *output,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int dtype,
    cudaStream_t stream) {

  constexpr int BLOCK_SIZE = 256;

  dim3 grid(size_n, size_m);
  dim3 block(BLOCK_SIZE);

  if (dtype == 0) {
    moe_gemv_transposed_kernel<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids, expert_ids,
        topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
        size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (dtype == 1) {
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

// ==========================================================================
// FP8 GEMV — warp-per-row design with shared memory input caching
// ==========================================================================
//
// Each block processes ROWS_PER_BLOCK output rows for one token.
// Each warp is assigned one output row and performs the full K-reduction.
// The input vector is loaded once into shared memory and reused by all warps.
//
// Grid: (ceil(N / ROWS_PER_BLOCK), M)
// Block: ROWS_PER_BLOCK * 32 threads (one warp per row)
//
// Benefits over one-block-per-row:
// - Amortizes input loading across ROWS_PER_BLOCK rows
// - Fewer blocks → less launch overhead
// - Better L2 locality for weight reads

// Warp-per-row FP8 GEMV. Input cached in shared memory (as float).
// Each warp computes dot(input, weight_row) for one output row.
template <typename T, int ROWS_PER_BLOCK>
__global__ void moe_gemv_kernel_fp8(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weights,
    const float *__restrict__ weight_scales,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const float *__restrict__ topk_weights,
    T *__restrict__ output,
    const int num_experts, const int topk, const int M, const int N,
    const int K, const int block_size_n, const int block_size_k) {

  const int row_base = blockIdx.x * ROWS_PER_BLOCK;
  const int token_idx = blockIdx.y;

  if (token_idx >= M)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;

  const int scale_k_dim = CEILDIV(K, block_size_k);
  const int scale_n_dim = CEILDIV(N, block_size_n);
  const float *expert_scales = weight_scales + (size_t)expert * scale_n_dim * scale_k_dim;

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  extern __shared__ float smem_input[];

  constexpr int THREADS = ROWS_PER_BLOCK * 32;

  // Vectorized load of input into shared memory as float.
  constexpr int LOAD_VEC = 8;
  const int k_vec_loads = K / LOAD_VEC;
  for (int i = tid; i < k_vec_loads; i += THREADS) {
    const int base = i * LOAD_VEC;
    float4 packed = __ldg(reinterpret_cast<const float4 *>(&input_row[base]));

    if constexpr (std::is_same<T, half>::value) {
      const half2 *h2 = reinterpret_cast<const half2 *>(&packed);
      float2 f0 = __half22float2(h2[0]);
      float2 f1 = __half22float2(h2[1]);
      float2 f2 = __half22float2(h2[2]);
      float2 f3 = __half22float2(h2[3]);
      smem_input[base + 0] = f0.x; smem_input[base + 1] = f0.y;
      smem_input[base + 2] = f1.x; smem_input[base + 3] = f1.y;
      smem_input[base + 4] = f2.x; smem_input[base + 5] = f2.y;
      smem_input[base + 6] = f3.x; smem_input[base + 7] = f3.y;
    } else {
#ifndef NO_BF16_KERNEL
      const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&packed);
      float2 f0 = vllm::bf1622float2(b2[0]);
      float2 f1 = vllm::bf1622float2(b2[1]);
      float2 f2 = vllm::bf1622float2(b2[2]);
      float2 f3 = vllm::bf1622float2(b2[3]);
      smem_input[base + 0] = f0.x; smem_input[base + 1] = f0.y;
      smem_input[base + 2] = f1.x; smem_input[base + 3] = f1.y;
      smem_input[base + 4] = f2.x; smem_input[base + 5] = f2.y;
      smem_input[base + 6] = f3.x; smem_input[base + 7] = f3.y;
#endif
    }
  }
  const int vec_end = k_vec_loads * LOAD_VEC;
  for (int i = vec_end + tid; i < K; i += THREADS) {
    smem_input[i] = vllm::to_float(input_row[i]);
  }
  __syncthreads();

  const int row = row_base + warp_id;
  if (row >= N)
    return;

  const uint8_t *weight_row = weights + (size_t)expert * N * K + (size_t)row * K;
  const int scale_n_idx = row / block_size_n;
  const float *row_scales = expert_scales + scale_n_idx * scale_k_dim;

  float sum = 0.0f;

  constexpr int VEC = 16;
  const int k_vec = K / VEC;

#pragma unroll 4
  for (int vi = lane_id; vi < k_vec; vi += 32) {
    const int k_base = vi * VEC;
    uint4 w16 = __ldg(reinterpret_cast<const uint4 *>(&weight_row[k_base]));
    const float scale = __ldg(&row_scales[k_base / block_size_k]);

    float partial = 0.0f;
    float wf0, wf1, wf2, wf3;

    vllm_rs::fp8x4_to_float4(w16.x, wf0, wf1, wf2, wf3);
    partial = fmaf(smem_input[k_base +  0], wf0, partial);
    partial = fmaf(smem_input[k_base +  1], wf1, partial);
    partial = fmaf(smem_input[k_base +  2], wf2, partial);
    partial = fmaf(smem_input[k_base +  3], wf3, partial);

    vllm_rs::fp8x4_to_float4(w16.y, wf0, wf1, wf2, wf3);
    partial = fmaf(smem_input[k_base +  4], wf0, partial);
    partial = fmaf(smem_input[k_base +  5], wf1, partial);
    partial = fmaf(smem_input[k_base +  6], wf2, partial);
    partial = fmaf(smem_input[k_base +  7], wf3, partial);

    vllm_rs::fp8x4_to_float4(w16.z, wf0, wf1, wf2, wf3);
    partial = fmaf(smem_input[k_base +  8], wf0, partial);
    partial = fmaf(smem_input[k_base +  9], wf1, partial);
    partial = fmaf(smem_input[k_base + 10], wf2, partial);
    partial = fmaf(smem_input[k_base + 11], wf3, partial);

    vllm_rs::fp8x4_to_float4(w16.w, wf0, wf1, wf2, wf3);
    partial = fmaf(smem_input[k_base + 12], wf0, partial);
    partial = fmaf(smem_input[k_base + 13], wf1, partial);
    partial = fmaf(smem_input[k_base + 14], wf2, partial);
    partial = fmaf(smem_input[k_base + 15], wf3, partial);

    sum = fmaf(scale, partial, sum);
  }

  const int k_remainder_start = k_vec * VEC;
  for (int k = k_remainder_start + lane_id; k < K; k += 32) {
    float wf;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(NO_HARDWARE_FP8)
    __half_raw h = __nv_cvt_fp8_to_halfraw(weight_row[k], __NV_E4M3);
    wf = __half2float(*reinterpret_cast<__half*>(&h));
#else
    wf = vllm::fp8::softmax_fp8_to_float_e4m3(weight_row[k]);
#endif
    float scale = __ldg(&row_scales[k / block_size_k]);
    sum = fmaf(smem_input[k] * wf, scale, sum);
  }

  sum = vllm_rs::warp_reduce_sum(sum);

  if (lane_id == 0) {
    if (topk_weights) {
      sum *= topk_weights[token_id];
    }
    T out_val;
    vllm::from_float(out_val, sum);
    output[(size_t)token_id * N + row] = out_val;
  }
}

// One-block-per-row fallback for large N when RPB doesn't evenly divide.
// Uses all BLOCK_SIZE threads in a single block for K-reduction on one output row.
template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_gemv_kernel_fp8_single(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weights,
    const float *__restrict__ weight_scales,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const float *__restrict__ topk_weights,
    T *__restrict__ output,
    const int num_experts, const int topk, const int M, const int N,
    const int K, const int block_size_n, const int block_size_k) {

  const int row = blockIdx.x;
  const int token_idx = blockIdx.y;

  if (token_idx >= M || row >= N)
    return;

  const int token_id = sorted_token_ids[token_idx];
  const int expert = expert_ids[token_idx];
  if (expert < 0 || expert >= num_experts)
    return;

  const int input_idx = token_id / (topk_weights ? 1 : topk);
  const T *input_row = input + (size_t)input_idx * K;
  const uint8_t *weight_row = weights + (size_t)expert * N * K + (size_t)row * K;

  const int scale_k_dim = CEILDIV(K, block_size_k);
  const int scale_n_dim = CEILDIV(N, block_size_n);
  const float *expert_scales = weight_scales + (size_t)expert * scale_n_dim * scale_k_dim;
  const int scale_n_idx = row / block_size_n;
  const float *row_scales = expert_scales + scale_n_idx * scale_k_dim;

  const int tid = threadIdx.x;
  float sum = 0.0f;

  constexpr int VEC = 16;
  const int k_vec = K / VEC;

#pragma unroll 4
  for (int vi = tid; vi < k_vec; vi += BLOCK_SIZE) {
    const int k_base = vi * VEC;
    uint4 w16 = __ldg(reinterpret_cast<const uint4 *>(&weight_row[k_base]));
    const float scale = __ldg(&row_scales[k_base / block_size_k]);

    // Load 16 input values
    float in_vals[16];
    if constexpr (std::is_same<T, half>::value) {
      const half2 *h2 = reinterpret_cast<const half2 *>(&input_row[k_base]);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float2 f = __half22float2(__ldg(&h2[j]));
        in_vals[j*2] = f.x;
        in_vals[j*2+1] = f.y;
      }
    } else {
#ifndef NO_BF16_KERNEL
      const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&input_row[k_base]);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float2 f = vllm::bf1622float2(__ldg(&b2[j]));
        in_vals[j*2] = f.x;
        in_vals[j*2+1] = f.y;
      }
#endif
    }

    float partial = 0.0f;
    float wf0, wf1, wf2, wf3;
    vllm_rs::fp8x4_to_float4(w16.x, wf0, wf1, wf2, wf3);
    partial = fmaf(in_vals[0], wf0, partial);
    partial = fmaf(in_vals[1], wf1, partial);
    partial = fmaf(in_vals[2], wf2, partial);
    partial = fmaf(in_vals[3], wf3, partial);
    vllm_rs::fp8x4_to_float4(w16.y, wf0, wf1, wf2, wf3);
    partial = fmaf(in_vals[4], wf0, partial);
    partial = fmaf(in_vals[5], wf1, partial);
    partial = fmaf(in_vals[6], wf2, partial);
    partial = fmaf(in_vals[7], wf3, partial);
    vllm_rs::fp8x4_to_float4(w16.z, wf0, wf1, wf2, wf3);
    partial = fmaf(in_vals[8], wf0, partial);
    partial = fmaf(in_vals[9], wf1, partial);
    partial = fmaf(in_vals[10], wf2, partial);
    partial = fmaf(in_vals[11], wf3, partial);
    vllm_rs::fp8x4_to_float4(w16.w, wf0, wf1, wf2, wf3);
    partial = fmaf(in_vals[12], wf0, partial);
    partial = fmaf(in_vals[13], wf1, partial);
    partial = fmaf(in_vals[14], wf2, partial);
    partial = fmaf(in_vals[15], wf3, partial);
    sum = fmaf(scale, partial, sum);
  }

  const int k_remainder_start = k_vec * VEC;
  for (int k = k_remainder_start + tid; k < K; k += BLOCK_SIZE) {
    float wf;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(NO_HARDWARE_FP8)
    __half_raw h = __nv_cvt_fp8_to_halfraw(weight_row[k], __NV_E4M3);
    wf = __half2float(*reinterpret_cast<__half*>(&h));
#else
    wf = vllm::fp8::softmax_fp8_to_float_e4m3(weight_row[k]);
#endif
    float scale = __ldg(&row_scales[k / block_size_k]);
    sum = fmaf(vllm::to_float(input_row[k]) * wf, scale, sum);
  }

  sum = vllm_rs::warp_reduce_sum(sum);

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

extern "C" void moe_gemv_fp8(
    const void *input,
    const uint8_t *weights,
    const float *weight_scales,
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,
    void *output,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int block_size_n,
    int block_size_k,
    int dtype,
    cudaStream_t stream) {

  // Shared memory: K floats for input caching
  int smem_bytes = size_k * sizeof(float);

  // Choose ROWS_PER_BLOCK (= warps per block) based on N dimension.
  // More rows → fewer blocks, better input reuse, but need enough N to fill.
  // Thread count = ROWS_PER_BLOCK * 32.
  //
  // For N=512: RPB=8 → 64 blocks per token, 256 threads/block
  // For N=2048: RPB=8 → 256 blocks per token, 256 threads/block
  // For large N: RPB=4 → more blocks for better SM utilization

  auto launch = [&]<int RPB>() {
    dim3 grid(CEILDIV(size_n, RPB), size_m);
    dim3 block(RPB * 32);

    if (dtype == 0) {
      moe_gemv_kernel_fp8<half, RPB><<<grid, block, smem_bytes, stream>>>(
          reinterpret_cast<const half *>(input),
          weights, weight_scales, sorted_token_ids, expert_ids,
          topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
          size_m, size_n, size_k, block_size_n, block_size_k);
    }
#ifndef NO_BF16_KERNEL
    else if (dtype == 1) {
      moe_gemv_kernel_fp8<nv_bfloat16, RPB><<<grid, block, smem_bytes, stream>>>(
          reinterpret_cast<const nv_bfloat16 *>(input),
          weights, weight_scales, sorted_token_ids, expert_ids,
          topk_weights, reinterpret_cast<nv_bfloat16 *>(output), num_experts, topk,
          size_m, size_n, size_k, block_size_n, block_size_k);
    }
#endif
  };

  // For small-to-medium N: use warp-per-row kernel with shared memory input caching.
  // For large N (>2048): use one-block-per-row with 256 threads for more K parallelism.
  if (size_n <= 512) {
    launch.template operator()<16>();
  } else if (size_n <= 2048) {
    launch.template operator()<8>();
  } else {
    // Large N: use single-row kernel (256 threads per output element)
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(size_n, size_m);
    dim3 block(BLOCK_SIZE);

    if (dtype == 0) {
      moe_gemv_kernel_fp8_single<half, BLOCK_SIZE><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half *>(input),
          weights, weight_scales, sorted_token_ids, expert_ids,
          topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
          size_m, size_n, size_k, block_size_n, block_size_k);
    }
#ifndef NO_BF16_KERNEL
    else if (dtype == 1) {
      moe_gemv_kernel_fp8_single<nv_bfloat16, BLOCK_SIZE><<<grid, block, 0, stream>>>(
          reinterpret_cast<const nv_bfloat16 *>(input),
          weights, weight_scales, sorted_token_ids, expert_ids,
          topk_weights, reinterpret_cast<nv_bfloat16 *>(output), num_experts, topk,
          size_m, size_n, size_k, block_size_n, block_size_k);
    }
#endif
  }
}