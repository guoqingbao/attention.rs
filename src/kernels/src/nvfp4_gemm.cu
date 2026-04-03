/**
 * @brief CUDA kernels for NVFP4 (NVIDIA FP4) GEMM, small-M dot-product GEMM,
 *        and indexed Mixture-of-Experts (MoE) GEMM with LUT-based dequantization.
 *
 * This file implements three kernel families for NVFP4 quantized weight matrices:
 *   1. nvfp4_matmul_smallm_kernel   – Dot-product kernel optimized for decode
 *      (M < 32), one thread-row per output row, no shared memory tiles.
 *   2. nvfp4_matmul_tiled           – Tiled GEMM for larger M (prefill), using
 *      shared memory tiles with configurable BM/BN/BK and thread-level tiling.
 *   3. nvfp4_moe_gemm               – Indexed Mixture-of-Experts GEMM with
 *      top-k expert selection and per-expert global scales.
 *
 * NVFP4 Format (NVIDIA FP4 / modelopt):
 * - FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
 * - Block size: 16 elements per scale (vs. 32 for MXFP4)
 * - Block scale: FP8 E4M3 format (stored as u8), converted via hardware-
 *   accelerated dispatch_fp8_to_float (SM89+) or software fallback
 * - Global scale: FP32 scalar per tensor (hierarchical two-level scaling)
 * - 2 FP4 values packed per byte (nibbles)
 * - Dequantization: x = LUT[nibble] * fp8_to_float(block_scale) * global_scale
 *
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/nvfp4_gemm.cu
 *
 * Notes:
 * - LUT-based FP4 E2M1 dequantization via byte_perm intrinsics (shared with MXFP4)
 * - FP8 E4M3 block scale conversion uses dtype_fp8.cuh dispatch_fp8_to_float
 *   for hardware intrinsics on SM89+ with software fallback on older GPUs
 * - Small-M kernel uses warp-stride loops for memory-bound decode workloads
 * - MoE kernel takes per-expert global_scales array (float[num_experts])
 * - BF16 dummy stubs provided for V100 (NO_BF16_KERNEL)
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

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "attention/dtype_fp8.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define NVFP4_BLOCK_SIZE 16
#define MOE_BLOCK_N 8
#define WARP_SIZE 32

namespace nvfp4_gemm {

using vllm::fp8::dispatch_fp8_to_float;

__device__ __forceinline__ int2 get_int_from_table_16(const int q4,
                                                      const uint32_t table0,
                                                      const uint32_t table1,
                                                      const uint32_t table2,
                                                      const uint32_t table3) {
  uint32_t tmp[2];
  const uint32_t low_high_selection = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = __byte_perm(table0, table1, q4 >> shift);
    const uint32_t high = __byte_perm(table2, table3, q4 >> shift);
    tmp[i] = __byte_perm(low, high, low_high_selection >> shift);
  }
  return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420),
                   __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ void dequant_store_8(int q4, float scale,
                                                uint32_t LUT0, uint32_t LUT1,
                                                uint32_t LUT2, uint32_t LUT3,
                                                float *dst) {
  int2 w = get_int_from_table_16(q4, LUT0, LUT1, LUT2, LUT3);
  dst[0] = (float)(int8_t)(w.x) * scale;
  dst[1] = (float)(int8_t)(w.y) * scale;
  dst[2] = (float)(int8_t)(w.x >> 8) * scale;
  dst[3] = (float)(int8_t)(w.y >> 8) * scale;
  dst[4] = (float)(int8_t)(w.x >> 16) * scale;
  dst[5] = (float)(int8_t)(w.y >> 16) * scale;
  dst[6] = (float)(int8_t)(w.x >> 24) * scale;
  dst[7] = (float)(int8_t)(w.y >> 24) * scale;
}

// Small-M matmul for NVFP4: dot-product per output element, one warp per N.
// NVFP4 uses block_size=16 and FP8 E4M3 block scales + FP32 global scale.
// Grid: (ceil(N/BLOCK_N_SM), M)  Block: (BLOCK_N_SM * 32)
constexpr int BLOCK_N_SM = 8;

template <typename T>
__launch_bounds__(BLOCK_N_SM * WARP_SIZE) __global__
    void nvfp4_matmul_smallm_kernel(const T *__restrict__ input,
                                    const uint8_t *__restrict__ weight,
                                    const uint8_t *__restrict__ weight_scale,
                                    float weight_global_scale,
                                    const T *__restrict__ bias,
                                    T *__restrict__ output, int M, int N,
                                    int K, bool has_bias) {
  extern __shared__ float s_input[];

  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int row = blockIdx.y;
  const int n_base = blockIdx.x * BLOCK_N_SM;
  const int n_idx = n_base + warp_id;
  const int weight_row_stride = K / 2;
  const int scale_stride = CEILDIV(K, NVFP4_BLOCK_SIZE);

  if (row >= M) return;

  const T *in_row = input + (size_t)row * K;

  for (int k = tid; k < K; k += block_size) {
    const int smem_idx = k + (k / WARP_SIZE);
    if constexpr (std::is_same_v<T, half>) {
      s_input[smem_idx] = __half2float(__ldg(&in_row[k]));
    } else {
      s_input[smem_idx] = __bfloat162float(__ldg(&in_row[k]));
    }
  }
  __syncthreads();

  if (n_idx >= N) return;

  const uint8_t *w_row = weight + (size_t)n_idx * weight_row_stride;
  const uint8_t *w_scale_row = weight_scale + (size_t)n_idx * scale_stride;

  float acc = 0.0f;

  for (int k = lane_id * NVFP4_BLOCK_SIZE; k < K;
       k += WARP_SIZE * NVFP4_BLOCK_SIZE) {
    float block_scale =
        dispatch_fp8_to_float(__ldg(&w_scale_row[k / NVFP4_BLOCK_SIZE])) *
        weight_global_scale * 0.5f;

    // 16 elements = 8 bytes
    uint2 w_vec = *reinterpret_cast<const uint2 *>(w_row + k / 2);
    const float *in = s_input + (k + (k / WARP_SIZE));

    {
      int2 w_int8 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
      acc = fmaf(in[0], (float)(int8_t)(w_int8.x) * block_scale, acc);
      acc = fmaf(in[1], (float)(int8_t)(w_int8.y) * block_scale, acc);
      acc = fmaf(in[2], (float)(int8_t)(w_int8.x >> 8) * block_scale, acc);
      acc = fmaf(in[3], (float)(int8_t)(w_int8.y >> 8) * block_scale, acc);
      acc = fmaf(in[4], (float)(int8_t)(w_int8.x >> 16) * block_scale, acc);
      acc = fmaf(in[5], (float)(int8_t)(w_int8.y >> 16) * block_scale, acc);
      acc = fmaf(in[6], (float)(int8_t)(w_int8.x >> 24) * block_scale, acc);
      acc = fmaf(in[7], (float)(int8_t)(w_int8.y >> 24) * block_scale, acc);
    }
    {
      int2 w_int8 = get_int_from_table_16(w_vec.y, LUT0, LUT1, LUT2, LUT3);
      acc = fmaf(in[8], (float)(int8_t)(w_int8.x) * block_scale, acc);
      acc = fmaf(in[9], (float)(int8_t)(w_int8.y) * block_scale, acc);
      acc = fmaf(in[10], (float)(int8_t)(w_int8.x >> 8) * block_scale, acc);
      acc = fmaf(in[11], (float)(int8_t)(w_int8.y >> 8) * block_scale, acc);
      acc = fmaf(in[12], (float)(int8_t)(w_int8.x >> 16) * block_scale, acc);
      acc = fmaf(in[13], (float)(int8_t)(w_int8.y >> 16) * block_scale, acc);
      acc = fmaf(in[14], (float)(int8_t)(w_int8.x >> 24) * block_scale, acc);
      acc = fmaf(in[15], (float)(int8_t)(w_int8.y >> 24) * block_scale, acc);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }

  if (lane_id == 0) {
    if (has_bias && bias != nullptr) {
      if constexpr (std::is_same_v<T, half>) {
        acc += __half2float(__ldg(&bias[n_idx]));
      } else {
        acc += __bfloat162float(__ldg(&bias[n_idx]));
      }
    }
    if constexpr (std::is_same_v<T, half>) {
      output[(size_t)row * N + n_idx] = __float2half(acc);
    } else {
      output[(size_t)row * N + n_idx] = __float2bfloat16(acc);
    }
  }
}

// Tiled matmul for larger M: NVFP4 version
template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void nvfp4_matmul_tiled(const T *__restrict__ input,
                                   const uint8_t *__restrict__ weight,
                                   const uint8_t *__restrict__ weight_scale,
                                   float weight_global_scale,
                                   const T *__restrict__ bias,
                                   T *__restrict__ output, int M, int N, int K,
                                   bool has_bias) {
  constexpr int THREADS_N = BLOCK_N / TN;
  constexpr int THREADS_M = BLOCK_M / TM;
  constexpr int NUM_THREADS = THREADS_N * THREADS_M;

  __shared__ float s_input[BLOCK_M][BLOCK_K + 1];
  __shared__ float s_weight[BLOCK_N][BLOCK_K + 1];

  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.y * THREADS_N + threadIdx.x;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int scale_stride = CEILDIV(K, NVFP4_BLOCK_SIZE);

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++)
      acc[i][j] = 0.0f;

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
      const int lm = idx / BLOCK_K;
      const int lk = idx % BLOCK_K;
      const int gm = by * BLOCK_M + lm;
      const int gk = k_tile + lk;
      float val = 0.0f;
      if (gm < M && gk < K) {
        if constexpr (std::is_same_v<T, half>) {
          val = __half2float(__ldg(&input[(size_t)gm * K + gk]));
        } else {
          val = __bfloat162float(__ldg(&input[(size_t)gm * K + gk]));
        }
      }
      s_input[lm][lk] = val;
    }

    // NVFP4 uses block_size=16, so for a BLOCK_K=16 tile, one scale per row
    for (int ln = tid; ln < BLOCK_N; ln += NUM_THREADS) {
      const int gn = bx * BLOCK_N + ln;
      if (gn < N) {
        float block_scale =
            dispatch_fp8_to_float(__ldg(&weight_scale[(size_t)gn * scale_stride +
                                                   k_tile / NVFP4_BLOCK_SIZE])) *
            weight_global_scale * 0.5f;

        uint2 w_vec = *reinterpret_cast<const uint2 *>(
            &weight[(size_t)gn * (K / 2) + k_tile / 2]);

        dequant_store_8(w_vec.x, block_scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][0]);
        dequant_store_8(w_vec.y, block_scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][8]);
      } else {
#pragma unroll
        for (int k = 0; k < BLOCK_K; k++)
          s_weight[ln][k] = 0.0f;
      }
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
      float a_frag[TM];
      float b_frag[TN];
#pragma unroll
      for (int i = 0; i < TM; i++)
        a_frag[i] = s_input[threadIdx.y * TM + i][k];
#pragma unroll
      for (int j = 0; j < TN; j++)
        b_frag[j] = s_weight[threadIdx.x * TN + j][k];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = by * BLOCK_M + threadIdx.y * TM + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int col = bx * BLOCK_N + threadIdx.x * TN + j;
        if (col < N) {
          float val = acc[i][j];
          if (has_bias && bias != nullptr) {
            if constexpr (std::is_same_v<T, half>) {
              val += __half2float(__ldg(&bias[col]));
            } else {
              val += __bfloat162float(__ldg(&bias[col]));
            }
          }
          if constexpr (std::is_same_v<T, half>) {
            output[(size_t)row * N + col] = __float2half(val);
          } else {
            output[(size_t)row * N + col] = __float2bfloat16(val);
          }
        }
      }
    }
  }
}

// Per-token MoE GEMM for NVFP4
template <typename T>
__launch_bounds__(MOE_BLOCK_N *WARP_SIZE) __global__
    void nvfp4_moe_gemm(const T *__restrict__ input,
                        const uint8_t *__restrict__ weights,
                        const uint8_t *__restrict__ weight_scales,
                        const float *__restrict__ weight_global_scales,
                        const T *__restrict__ biases,
                        const uint32_t *__restrict__ indices,
                        T *__restrict__ output, int num_tokens, int topk,
                        int num_experts, int N, int K, bool has_bias,
                        bool input_has_topk_dim) {
  extern __shared__ float s_input_padded[];

  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int n_chunks = CEILDIV(N, MOE_BLOCK_N);
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int weight_row_stride = K / 2;
  const int scale_stride = CEILDIV(K, NVFP4_BLOCK_SIZE);

  int token_idx, expert_slot_start, expert_slot_end, n_base;

  if (!input_has_topk_dim) {
    n_base = (blockIdx.x % n_chunks) * MOE_BLOCK_N;
    token_idx = blockIdx.x / n_chunks;
    expert_slot_start = 0;
    expert_slot_end = topk;
  } else {
    n_base = (blockIdx.x % n_chunks) * MOE_BLOCK_N;
    int temp = blockIdx.x / n_chunks;
    expert_slot_start = temp % topk;
    expert_slot_end = expert_slot_start + 1;
    token_idx = temp / topk;
  }

  if (token_idx >= num_tokens) return;

  const int n_idx = n_base + warp_id;
  if (n_idx >= N) return;

  const T *in_row;
  if (!input_has_topk_dim) {
    in_row = input + (size_t)token_idx * K;
  } else {
    in_row =
        input + (size_t)token_idx * topk * K + (size_t)expert_slot_start * K;
  }

  for (int k = tid; k < K; k += block_size) {
    const int smem_idx = k + (k / WARP_SIZE);
    if constexpr (std::is_same_v<T, half>) {
      s_input_padded[smem_idx] = __half2float(__ldg(&in_row[k]));
    } else {
      s_input_padded[smem_idx] = __bfloat162float(__ldg(&in_row[k]));
    }
  }
  __syncthreads();

  for (int expert_slot = expert_slot_start; expert_slot < expert_slot_end;
       expert_slot++) {
    const uint32_t expert_idx = __ldg(&indices[token_idx * topk + expert_slot]);
    if (expert_idx >= (uint32_t)num_experts) continue;

    const float global_scale = weight_global_scales[expert_idx];

    const uint8_t *w_row = weights +
                           (size_t)expert_idx * N * weight_row_stride +
                           (size_t)n_idx * weight_row_stride;
    const uint8_t *w_scale_row = weight_scales +
                                 (size_t)expert_idx * N * scale_stride +
                                 (size_t)n_idx * scale_stride;

    float acc = 0.0f;

    for (int k = lane_id * NVFP4_BLOCK_SIZE; k < K;
         k += WARP_SIZE * NVFP4_BLOCK_SIZE) {
      float block_scale =
          dispatch_fp8_to_float(__ldg(&w_scale_row[k / NVFP4_BLOCK_SIZE])) *
          global_scale * 0.5f;

      uint2 w_vec = *reinterpret_cast<const uint2 *>(w_row + k / 2);
      const float *in = s_input_padded + (k + (k / WARP_SIZE));

      {
        int2 w_int8 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
        acc = fmaf(in[0], (float)(int8_t)(w_int8.x) * block_scale, acc);
        acc = fmaf(in[1], (float)(int8_t)(w_int8.y) * block_scale, acc);
        acc = fmaf(in[2], (float)(int8_t)(w_int8.x >> 8) * block_scale, acc);
        acc = fmaf(in[3], (float)(int8_t)(w_int8.y >> 8) * block_scale, acc);
        acc = fmaf(in[4], (float)(int8_t)(w_int8.x >> 16) * block_scale, acc);
        acc = fmaf(in[5], (float)(int8_t)(w_int8.y >> 16) * block_scale, acc);
        acc = fmaf(in[6], (float)(int8_t)(w_int8.x >> 24) * block_scale, acc);
        acc = fmaf(in[7], (float)(int8_t)(w_int8.y >> 24) * block_scale, acc);
      }
      {
        int2 w_int8 = get_int_from_table_16(w_vec.y, LUT0, LUT1, LUT2, LUT3);
        acc = fmaf(in[8], (float)(int8_t)(w_int8.x) * block_scale, acc);
        acc = fmaf(in[9], (float)(int8_t)(w_int8.y) * block_scale, acc);
        acc = fmaf(in[10], (float)(int8_t)(w_int8.x >> 8) * block_scale, acc);
        acc = fmaf(in[11], (float)(int8_t)(w_int8.y >> 8) * block_scale, acc);
        acc = fmaf(in[12], (float)(int8_t)(w_int8.x >> 16) * block_scale, acc);
        acc = fmaf(in[13], (float)(int8_t)(w_int8.y >> 16) * block_scale, acc);
        acc = fmaf(in[14], (float)(int8_t)(w_int8.x >> 24) * block_scale, acc);
        acc = fmaf(in[15], (float)(int8_t)(w_int8.y >> 24) * block_scale, acc);
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
      if (has_bias && biases) {
        const T *bias_row = biases + (size_t)expert_idx * N;
        if constexpr (std::is_same_v<T, half>) {
          acc += __half2float(__ldg(&bias_row[n_idx]));
        } else {
          acc += __bfloat162float(__ldg(&bias_row[n_idx]));
        }
      }

      size_t out_idx =
          (size_t)token_idx * topk * N + (size_t)expert_slot * N + n_idx;
      if constexpr (std::is_same_v<T, half>) {
        output[out_idx] = __float2half(acc);
      } else {
        output[out_idx] = __float2bfloat16(acc);
      }
    }
  }
}

} // namespace nvfp4_gemm

// ============================================================================
// C API
// ============================================================================

extern "C" void nvfp4_matmul_smallm_f16(const __half *input,
                                         const uint8_t *weight,
                                         const uint8_t *weight_scale,
                                         float weight_global_scale,
                                         const __half *bias, __half *output,
                                         int M, int N, int K, bool has_bias,
                                         cudaStream_t stream) {
  using namespace nvfp4_gemm;
  constexpr int THREADS = BLOCK_N_SM * WARP_SIZE;
  dim3 block(THREADS);
  dim3 grid(CEILDIV(N, BLOCK_N_SM), M);
  size_t smem = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);
  nvfp4_gemm::nvfp4_matmul_smallm_kernel<half>
      <<<grid, block, smem, stream>>>(input, weight, weight_scale,
                                      weight_global_scale, bias, output, M, N,
                                      K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_BF16_KERNEL
extern "C" void nvfp4_matmul_smallm_bf16(const __nv_bfloat16 *input,
                                          const uint8_t *weight,
                                          const uint8_t *weight_scale,
                                          float weight_global_scale,
                                          const __nv_bfloat16 *bias,
                                          __nv_bfloat16 *output,
                                          int M, int N, int K, bool has_bias,
                                          cudaStream_t stream) {
  using namespace nvfp4_gemm;
  constexpr int THREADS = BLOCK_N_SM * WARP_SIZE;
  dim3 block(THREADS);
  dim3 grid(CEILDIV(N, BLOCK_N_SM), M);
  size_t smem = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);
  nvfp4_gemm::nvfp4_matmul_smallm_kernel<__nv_bfloat16>
      <<<grid, block, smem, stream>>>(input, weight, weight_scale,
                                      weight_global_scale, bias, output, M, N,
                                      K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}
#else
extern "C" void nvfp4_matmul_smallm_bf16(const void *, const uint8_t *,
                                          const uint8_t *, float, const void *,
                                          void *, int, int, int, bool,
                                          cudaStream_t) {}
#endif

extern "C" void nvfp4_matmul_f16(const __half *input, const uint8_t *weight,
                                  const uint8_t *weight_scale,
                                  float weight_global_scale,
                                  const __half *bias, __half *output, int M,
                                  int N, int K, bool has_bias,
                                  cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_M = BM / TM;
  dim3 block(THREADS_N, THREADS_M);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
  nvfp4_gemm::nvfp4_matmul_tiled<half, BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale,
                                   weight_global_scale, bias, output, M, N, K,
                                   has_bias);
  CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_BF16_KERNEL
extern "C" void nvfp4_matmul_bf16(const __nv_bfloat16 *input,
                                   const uint8_t *weight,
                                   const uint8_t *weight_scale,
                                   float weight_global_scale,
                                   const __nv_bfloat16 *bias,
                                   __nv_bfloat16 *output, int M, int N, int K,
                                   bool has_bias, cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_M = BM / TM;
  dim3 block(THREADS_N, THREADS_M);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
  nvfp4_gemm::nvfp4_matmul_tiled<__nv_bfloat16, BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale,
                                   weight_global_scale, bias, output, M, N, K,
                                   has_bias);
  CUDA_CHECK(cudaGetLastError());
}
#else
extern "C" void nvfp4_matmul_bf16(const void *, const uint8_t *,
                                   const uint8_t *, float, const void *,
                                   void *, int, int, int, bool,
                                   cudaStream_t) {}
#endif

extern "C" void nvfp4_indexed_moe_gemm_f16(
    const __half *input, const uint8_t *weights, const uint8_t *weight_scales,
    const float *weight_global_scales, const __half *biases,
    const uint32_t *indices, __half *output, int num_tokens, int topk,
    int num_experts, int N, int K, bool has_bias, bool input_has_topk_dim,
    cudaStream_t stream) {
  constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
  int n_chunks = CEILDIV(N, MOE_BLOCK_N);
  int total_blocks =
      input_has_topk_dim ? num_tokens * topk * n_chunks : num_tokens * n_chunks;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(total_blocks);
  size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);
  nvfp4_gemm::nvfp4_moe_gemm<half><<<grid, block, shared_mem_size, stream>>>(
      input, weights, weight_scales, weight_global_scales, biases, indices,
      output, num_tokens, topk, num_experts, N, K, has_bias,
      input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_BF16_KERNEL
extern "C" void nvfp4_indexed_moe_gemm_bf16(
    const __nv_bfloat16 *input, const uint8_t *weights,
    const uint8_t *weight_scales, const float *weight_global_scales,
    const __nv_bfloat16 *biases, const uint32_t *indices,
    __nv_bfloat16 *output, int num_tokens, int topk, int num_experts, int N,
    int K, bool has_bias, bool input_has_topk_dim, cudaStream_t stream) {
  constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
  int n_chunks = CEILDIV(N, MOE_BLOCK_N);
  int total_blocks =
      input_has_topk_dim ? num_tokens * topk * n_chunks : num_tokens * n_chunks;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(total_blocks);
  size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);
  nvfp4_gemm::nvfp4_moe_gemm<__nv_bfloat16>
      <<<grid, block, shared_mem_size, stream>>>(
          input, weights, weight_scales, weight_global_scales, biases, indices,
          output, num_tokens, topk, num_experts, N, K, has_bias,
          input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}
#else
extern "C" void nvfp4_indexed_moe_gemm_bf16(const void *, const uint8_t *,
                                             const uint8_t *, const float *,
                                             const void *, const uint32_t *,
                                             void *, int, int, int, int, int,
                                             bool, bool, cudaStream_t) {}
#endif
