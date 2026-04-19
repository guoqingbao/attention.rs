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
 * SM80+ (Ampere/Ada) uses FP8 dequantization intrinsics.
 * SM90+ (Hopper/Blackwell) uses __nv_cvt_fp8x2_to_halfraw2 for paired FP8 conversion.
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && !defined(NO_HARDWARE_FP8)
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
// Optimizations for Hopper (SM90) decode path:
// - Input cached in shared memory as native half/bf16, halving smem bandwidth
// - half2 FMA accumulation to exploit SM90 half2 throughput
// - 32-byte vectorized weight loads (uint4) to saturate memory bandwidth
// - Paired FP8→half2 conversion via __nv_cvt_fp8x2_to_halfraw2 (SM90 native)
// - Dual-accumulator technique to hide FMA pipeline latency
// - Tuned ROWS_PER_BLOCK for better occupancy and input reuse

__device__ __forceinline__ float smem_to_float(half v) {
  return __half2float(v);
}
#ifndef NO_BF16_KERNEL
__device__ __forceinline__ float smem_to_float(nv_bfloat16 v) {
  return __bfloat162float(v);
}
#endif

// Dequant + FMA helper: converts 4 packed FP8 values and accumulates dot product
// against 4 input values from shared memory.
// SM90+ half path: native half2 FMA for double throughput.
// SM90+ bf16 path: native __nv_bfloat162 FMA for double throughput.
// Pre-SM90: scalar float FMA.
template <typename T>
__device__ __forceinline__ void fp8x4_dot_half2(
    uint32_t packed, const T *smem, int offset, float &acc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && !defined(NO_HARDWARE_FP8)
  __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(packed & 0xFFFF), __NV_E4M3);
  __half2_raw pair2 = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>((packed >> 16) & 0xFFFF), __NV_E4M3);
  if constexpr (std::is_same<T, half>::value) {
    half2 in01 = *reinterpret_cast<const half2*>(&smem[offset]);
    half2 in23 = *reinterpret_cast<const half2*>(&smem[offset + 2]);
    half2 w01 = *reinterpret_cast<half2*>(&pair);
    half2 w23 = *reinterpret_cast<half2*>(&pair2);
    half2 prod1 = __hmul2(in01, w01);
    half2 prod2 = __hmul2(in23, w23);
    half2 sum2 = __hadd2(prod1, prod2);
    float2 f = __half22float2(sum2);
    acc += f.x + f.y;
  } else {
#ifndef NO_BF16_KERNEL
    // Convert FP8→half pairs to bf16 pairs for native bf162 FMA
    __nv_bfloat162 in01 = *reinterpret_cast<const __nv_bfloat162*>(&smem[offset]);
    __nv_bfloat162 in23 = *reinterpret_cast<const __nv_bfloat162*>(&smem[offset + 2]);
    // Convert half2 weights to bf16 pairs
    float2 wf01 = __half22float2(*reinterpret_cast<half2*>(&pair));
    float2 wf23 = __half22float2(*reinterpret_cast<half2*>(&pair2));
    __nv_bfloat162 w01 = __floats2bfloat162_rn(wf01.x, wf01.y);
    __nv_bfloat162 w23 = __floats2bfloat162_rn(wf23.x, wf23.y);
    __nv_bfloat162 prod1 = __hmul2(in01, w01);
    __nv_bfloat162 prod2 = __hmul2(in23, w23);
    __nv_bfloat162 sum2 = __hadd2(prod1, prod2);
    float2 f = vllm::bf1622float2(sum2);
    acc += f.x + f.y;
#endif
  }
#else
  float wf0, wf1, wf2, wf3;
  vllm_rs::fp8x4_to_float4(packed, wf0, wf1, wf2, wf3);
  acc = fmaf(smem_to_float(smem[offset + 0]), wf0, acc);
  acc = fmaf(smem_to_float(smem[offset + 1]), wf1, acc);
  acc = fmaf(smem_to_float(smem[offset + 2]), wf2, acc);
  acc = fmaf(smem_to_float(smem[offset + 3]), wf3, acc);
#endif
}

// Warp-per-row FP8 GEMV. Input cached in shared memory in native T format.
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

  // Shared memory stores input in native T format (half or bf16), halving bandwidth
  extern __shared__ char smem_raw[];
  T *smem_input = reinterpret_cast<T*>(smem_raw);

  constexpr int THREADS = ROWS_PER_BLOCK * 32;

  // Load input into shared memory using 16-byte vectorized loads
  constexpr int LOAD_VEC = 8; // 8 T elements = 16 bytes
  const int k_vec_loads = K / LOAD_VEC;
  for (int i = tid; i < k_vec_loads; i += THREADS) {
    const int base = i * LOAD_VEC;
    // 16-byte load: float4 = 16 bytes = 8 half values
    *reinterpret_cast<float4*>(&smem_input[base]) =
        __ldg(reinterpret_cast<const float4*>(&input_row[base]));
  }
  const int vec_end = k_vec_loads * LOAD_VEC;
  for (int i = vec_end + tid; i < K; i += THREADS) {
    smem_input[i] = input_row[i];
  }
  __syncthreads();

  const int row = row_base + warp_id;
  if (row >= N)
    return;

  const uint8_t *weight_row = weights + (size_t)expert * N * K + (size_t)row * K;
  const int scale_n_idx = row / block_size_n;
  const float *row_scales = expert_scales + scale_n_idx * scale_k_dim;

  float sum0 = 0.0f;
  float sum1 = 0.0f;

  constexpr int VEC = 16;
  const int k_vec = K / VEC;

#pragma unroll 4
  for (int vi = lane_id; vi < k_vec; vi += 32) {
    const int k_base = vi * VEC;
    uint4 w16 = __ldg(reinterpret_cast<const uint4*>(&weight_row[k_base]));
    const float scale = __ldg(&row_scales[k_base / block_size_k]);

    float partial0 = 0.0f, partial1 = 0.0f;
    fp8x4_dot_half2<T>(w16.x, smem_input, k_base + 0, partial0);
    fp8x4_dot_half2<T>(w16.y, smem_input, k_base + 4, partial0);
    fp8x4_dot_half2<T>(w16.z, smem_input, k_base + 8, partial1);
    fp8x4_dot_half2<T>(w16.w, smem_input, k_base + 12, partial1);
    sum0 = fmaf(scale, partial0, sum0);
    sum1 = fmaf(scale, partial1, sum1);
  }

  float sum = sum0 + sum1;

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
    sum = fmaf(smem_to_float(smem_input[k]) * wf, scale, sum);
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

// One-block-per-row FP8 GEMV with shared memory input caching (native T format).
// Uses all BLOCK_SIZE threads for K-reduction on one output row.
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

  // Shared memory in native T format + space for warp reduction
  extern __shared__ char smem_single_raw[];
  T *smem_input = reinterpret_cast<T*>(smem_single_raw);

  // Load input into shared memory using 16-byte vectorized loads
  constexpr int LOAD_VEC = 8;
  const int k_vec_loads = K / LOAD_VEC;
  for (int i = tid; i < k_vec_loads; i += BLOCK_SIZE) {
    const int base = i * LOAD_VEC;
    *reinterpret_cast<float4*>(&smem_input[base]) =
        __ldg(reinterpret_cast<const float4*>(&input_row[base]));
  }
  const int vec_end = k_vec_loads * LOAD_VEC;
  for (int i = vec_end + tid; i < K; i += BLOCK_SIZE) {
    smem_input[i] = input_row[i];
  }
  __syncthreads();

  float sum0 = 0.0f;
  float sum1 = 0.0f;

  constexpr int VEC = 16;
  const int k_vec = K / VEC;

#pragma unroll 4
  for (int vi = tid; vi < k_vec; vi += BLOCK_SIZE) {
    const int k_base = vi * VEC;
    uint4 w16 = __ldg(reinterpret_cast<const uint4*>(&weight_row[k_base]));
    const float scale = __ldg(&row_scales[k_base / block_size_k]);

    float partial0 = 0.0f, partial1 = 0.0f;
    fp8x4_dot_half2<T>(w16.x, smem_input, k_base + 0, partial0);
    fp8x4_dot_half2<T>(w16.y, smem_input, k_base + 4, partial0);
    fp8x4_dot_half2<T>(w16.z, smem_input, k_base + 8, partial1);
    fp8x4_dot_half2<T>(w16.w, smem_input, k_base + 12, partial1);
    sum0 = fmaf(scale, partial0, sum0);
    sum1 = fmaf(scale, partial1, sum1);
  }

  float sum = sum0 + sum1;

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
    sum = fmaf(smem_to_float(smem_input[k]) * wf, scale, sum);
  }

  sum = vllm_rs::warp_reduce_sum(sum);

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  // Warp reduction area after input data in shared memory
  float *smem_reduce = reinterpret_cast<float*>(smem_single_raw + K * sizeof(T));
  // Align to float boundary
  smem_reduce = reinterpret_cast<float*>(
    (reinterpret_cast<uintptr_t>(smem_reduce) + 3) & ~uintptr_t(3));
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) {
    smem_reduce[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
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

  // Shared memory: K elements in native T format (half/bf16 = 2 bytes each)
  int elem_size = (dtype == 0) ? sizeof(half) : sizeof(nv_bfloat16);
  int smem_bytes = size_k * elem_size;

  // Choose ROWS_PER_BLOCK (= warps per block) based on N dimension.
  // More rows → fewer blocks, better input reuse, but need enough N to fill.
  // Thread count = ROWS_PER_BLOCK * 32.
  //
  // For N=512:  RPB=16 → 32 blocks per token, 512 threads/block
  // For N=2048: RPB=8  → 256 blocks per token, 256 threads/block
  // For N=4096: RPB=4  → 1024 blocks per token, 128 threads/block

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
  // For large N (>2048): use single-row kernel with shared memory + 256 threads.
  if (size_n <= 512) {
    launch.template operator()<16>();
  } else if (size_n <= 2048) {
    launch.template operator()<8>();
  } else {
    // Large N: single-row kernel with shared memory input caching
    constexpr int BLOCK_SIZE = 256;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    // smem for input (K elements as T) + warp reduction (NUM_WARPS floats, aligned)
    int smem_single = size_k * elem_size + 16 + NUM_WARPS * sizeof(float);
    dim3 grid(size_n, size_m);
    dim3 block(BLOCK_SIZE);

    if (dtype == 0) {
      moe_gemv_kernel_fp8_single<half, BLOCK_SIZE><<<grid, block, smem_single, stream>>>(
          reinterpret_cast<const half *>(input),
          weights, weight_scales, sorted_token_ids, expert_ids,
          topk_weights, reinterpret_cast<half *>(output), num_experts, topk,
          size_m, size_n, size_k, block_size_n, block_size_k);
    }
#ifndef NO_BF16_KERNEL
    else if (dtype == 1) {
      moe_gemv_kernel_fp8_single<nv_bfloat16, BLOCK_SIZE><<<grid, block, smem_single, stream>>>(
          reinterpret_cast<const nv_bfloat16 *>(input),
          weights, weight_scales, sorted_token_ids, expert_ids,
          topk_weights, reinterpret_cast<nv_bfloat16 *>(output), num_experts, topk,
          size_m, size_n, size_k, block_size_n, block_size_k);
    }
#endif
  }
}