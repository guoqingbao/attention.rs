/**
 * @brief CUDA kernels for MXFP4 (OCP Microscaling FP4) GEMM, MoE GEMM, and
 *        small-M dot-product GEMM with LUT-based dequantization.
 *
 * This file implements three kernel families for MXFP4 quantized weight matrices:
 *   1. (Newly added) mxfp4_matmul_smallm_kernel  – Dot-product kernel optimized for decode
 *      (M < 32), one thread-row per output row, no shared memory tiles.
 *   2. mxfp4_matmul_tiled          – Tiled GEMM for larger M (prefill), using
 *      shared memory tiles with configurable BM/BN/BK and thread-level tiling.
 *   3. mxfp4_moe_grouped_gemm_tiled – Indexed Mixture-of-Experts GEMM with
 *      top-k expert selection, segmented per-expert weight layouts.
 *
 * MXFP4 Format (OCP Microscaling):
 * - FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
 * - Block size: 32 elements per scale
 * - Scale: E8M0 format (8-bit exponent, stored as u8 with bias 127)
 * - 2 FP4 values packed per byte (nibbles)
 * - Dequantization: x = LUT[nibble] * e8m0_to_float(scale)
 *
 * Copyright (c) 2025, Eric L Buehler and Guoqing Bao.  All rights reserved.
 *
 * Notes:
 * - LUT-based FP4 E2M1 dequantization via byte_perm intrinsics for throughput
 * - Small-M kernel uses warp-stride loops for memory-bound decode workloads
 * - Tiled GEMM supports FP16 and BF16 output via C++ templates
 * - MoE kernel handles per-expert indexing with dynamic shared memory for
 *   segment offsets
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

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define MXFP4_BLOCK_SIZE 32
#define MOE_BLOCK_N 8
#define WARP_SIZE 32

namespace mxfp4_gemm {

__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
  return __uint_as_float((uint32_t)e << 23);
}

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

// Tiled matmul: input [M, K] @ weight^T [N, K/2] -> output [M, N]
template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void mxfp4_matmul_tiled(const T *__restrict__ input,
                                   const uint8_t *__restrict__ weight,
                                   const uint8_t *__restrict__ weight_scale,
                                   const T *__restrict__ bias,
                                   T *__restrict__ output, int M, int N, int K,
                                   bool has_bias) {
  constexpr int THREADS_N = BLOCK_N / TN;
  constexpr int THREADS_M = BLOCK_M / TM;
  constexpr int NUM_THREADS = THREADS_N * THREADS_M;
  constexpr int BK_PAD = BLOCK_K + 1;

  __shared__ float s_input[BLOCK_M][BK_PAD];
  __shared__ float s_weight[BLOCK_N][BK_PAD];

  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.y * THREADS_N + threadIdx.x;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

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

    for (int ln = tid; ln < BLOCK_N; ln += NUM_THREADS) {
      const int gn = bx * BLOCK_N + ln;
      if (gn < N) {
        uint4 w_vec = *reinterpret_cast<const uint4 *>(
            &weight[(size_t)gn * (K / 2) + k_tile / 2]);
        float scale =
            e8m0_to_float(__ldg(&weight_scale[(size_t)gn * scale_stride +
                                              k_tile / MXFP4_BLOCK_SIZE])) *
            0.5f;

        dequant_store_8(w_vec.x, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][0]);
        dequant_store_8(w_vec.y, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][8]);
        dequant_store_8(w_vec.z, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][16]);
        dequant_store_8(w_vec.w, scale, LUT0, LUT1, LUT2, LUT3,
                        &s_weight[ln][24]);
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

// Per-token MoE GEMM: one block per (token × N-chunk), loops over topk experts
template <typename T>
__launch_bounds__(MOE_BLOCK_N *WARP_SIZE) __global__
    void mxfp4_moe_gemm(const T *__restrict__ input,
                        const uint8_t *__restrict__ weights,
                        const uint8_t *__restrict__ weight_scales,
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
  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

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

  if (token_idx >= num_tokens)
    return;

  const int n_idx = n_base + warp_id;
  if (n_idx >= N)
    return;

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
    if (expert_idx >= (uint32_t)num_experts)
      continue;

    const uint8_t *w_row = weights +
                           (size_t)expert_idx * N * weight_row_stride +
                           (size_t)n_idx * weight_row_stride;
    const uint8_t *w_scale_row = weight_scales +
                                 (size_t)expert_idx * N * scale_stride +
                                 (size_t)n_idx * scale_stride;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = lane_id * 32; k < K; k += 32 * 32) {
      float w_scale =
          e8m0_to_float(__ldg(&w_scale_row[k / MXFP4_BLOCK_SIZE])) * 0.5f;

      uint4 w_vec = *reinterpret_cast<const uint4 *>(w_row + k / 2);
      const float *in = s_input_padded + (k + (k / WARP_SIZE));

      {
        int2 w_int8 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
        const int w_even = w_int8.x;
        const int w_odd = w_int8.y;
        acc0 = fmaf(in[0], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[1], (float)(int8_t)(w_odd)*w_scale, acc1);
        acc0 = fmaf(in[2], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
        acc1 = fmaf(in[3], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
        acc0 = fmaf(in[4], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
        acc1 = fmaf(in[5], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
        acc0 = fmaf(in[6], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
        acc1 = fmaf(in[7], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
      }
      {
        int2 w_int8 = get_int_from_table_16(w_vec.y, LUT0, LUT1, LUT2, LUT3);
        const int w_even = w_int8.x;
        const int w_odd = w_int8.y;
        acc0 = fmaf(in[8], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[9], (float)(int8_t)(w_odd)*w_scale, acc1);
        acc0 = fmaf(in[10], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
        acc1 = fmaf(in[11], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
        acc0 = fmaf(in[12], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
        acc1 = fmaf(in[13], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
        acc0 = fmaf(in[14], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
        acc1 = fmaf(in[15], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
      }
      {
        int2 w_int8 = get_int_from_table_16(w_vec.z, LUT0, LUT1, LUT2, LUT3);
        const int w_even = w_int8.x;
        const int w_odd = w_int8.y;
        acc0 = fmaf(in[16], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[17], (float)(int8_t)(w_odd)*w_scale, acc1);
        acc0 = fmaf(in[18], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
        acc1 = fmaf(in[19], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
        acc0 = fmaf(in[20], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
        acc1 = fmaf(in[21], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
        acc0 = fmaf(in[22], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
        acc1 = fmaf(in[23], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
      }
      {
        int2 w_int8 = get_int_from_table_16(w_vec.w, LUT0, LUT1, LUT2, LUT3);
        const int w_even = w_int8.x;
        const int w_odd = w_int8.y;
        acc0 = fmaf(in[24], (float)(int8_t)(w_even)*w_scale, acc0);
        acc1 = fmaf(in[25], (float)(int8_t)(w_odd)*w_scale, acc1);
        acc0 = fmaf(in[26], (float)(int8_t)(w_even >> 8) * w_scale, acc0);
        acc1 = fmaf(in[27], (float)(int8_t)(w_odd >> 8) * w_scale, acc1);
        acc0 = fmaf(in[28], (float)(int8_t)(w_even >> 16) * w_scale, acc0);
        acc1 = fmaf(in[29], (float)(int8_t)(w_odd >> 16) * w_scale, acc1);
        acc0 = fmaf(in[30], (float)(int8_t)(w_even >> 24) * w_scale, acc0);
        acc1 = fmaf(in[31], (float)(int8_t)(w_odd >> 24) * w_scale, acc1);
      }
    }

    float acc = acc0 + acc1;

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

// Phase 1: Build per-expert token lists in global memory.
__global__ void mxfp4_moe_scatter_tokens_tiled(
    const uint32_t *__restrict__ indices,
    int *__restrict__ g_token_lists,
    int *__restrict__ g_expert_counts,
    int total_work, int num_experts, int list_stride) {
  const int expert_id = blockIdx.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_size = blockDim.x * blockDim.y;
  int *my_list = g_token_lists + (size_t)expert_id * list_stride;
  int *my_count = g_expert_counts + expert_id;

  for (int i = tid; i < total_work; i += block_size) {
    if (__ldg(&indices[i]) == (uint32_t)expert_id) {
      int pos = atomicAdd(my_count, 1);
      if (pos < list_stride)
        my_list[pos] = i;
    }
  }
}

// Phase 2: Tiled GEMM reading per-expert token lists from global memory.
template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
__global__ void mxfp4_moe_grouped_gemm_tiled(
    const T *__restrict__ input, const uint8_t *__restrict__ weights,
    const uint8_t *__restrict__ weight_scales, const T *__restrict__ biases,
    const uint32_t *__restrict__ indices, T *__restrict__ output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim,
    const int *__restrict__ g_token_lists,
    const int *__restrict__ g_expert_counts,
    int list_stride) {
  constexpr int THREADS_N = BLOCK_N / TN;
  constexpr int THREADS_M = BLOCK_M / TM;
  constexpr int NUM_THREADS = THREADS_N * THREADS_M;
  constexpr int BK_PAD = BLOCK_K + 1;

  __shared__ float s_input[BLOCK_M][BK_PAD];
  __shared__ float s_weight[BLOCK_N][BK_PAD];

  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.y * THREADS_N + threadIdx.x;
  const int expert_id = blockIdx.y;
  const int n_base = blockIdx.x * BLOCK_N;
  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

  const uint8_t *expert_weight = weights + (size_t)expert_id * N * (K / 2);
  const uint8_t *expert_scale =
      weight_scales + (size_t)expert_id * N * scale_stride;

  const int *token_list = g_token_lists + (size_t)expert_id * list_stride;
  const int M_expert = g_expert_counts[expert_id];
  if (M_expert == 0)
    return;

  for (int m_tile = 0; m_tile < CEILDIV(M_expert, BLOCK_M); m_tile++) {
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
        const int work_pos = m_tile * BLOCK_M + lm;
        const int gk = k_tile + lk;

        float val = 0.0f;
        if (work_pos < M_expert && gk < K) {
          const int work_idx = token_list[work_pos];
          const int input_row =
              input_has_topk_dim ? work_idx : (work_idx / topk);
          if constexpr (std::is_same_v<T, half>) {
            val = __half2float(__ldg(&input[(size_t)input_row * K + gk]));
          } else {
            val = __bfloat162float(__ldg(&input[(size_t)input_row * K + gk]));
          }
        }
        s_input[lm][lk] = val;
      }

      for (int ln = tid; ln < BLOCK_N; ln += NUM_THREADS) {
        const int gn = n_base + ln;
        if (gn < N) {
          uint4 w_vec = *reinterpret_cast<const uint4 *>(
              &expert_weight[(size_t)gn * (K / 2) + k_tile / 2]);
          float scale =
              e8m0_to_float(__ldg(&expert_scale[(size_t)gn * scale_stride +
                                                k_tile / MXFP4_BLOCK_SIZE])) *
              0.5f;
          dequant_store_8(w_vec.x, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][0]);
          dequant_store_8(w_vec.y, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][8]);
          dequant_store_8(w_vec.z, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][16]);
          dequant_store_8(w_vec.w, scale, LUT0, LUT1, LUT2, LUT3,
                          &s_weight[ln][24]);
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
      const int work_pos = m_tile * BLOCK_M + threadIdx.y * TM + i;
      if (work_pos < M_expert) {
        const int work_idx = token_list[work_pos];
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const int col = n_base + threadIdx.x * TN + j;
          if (col < N) {
            float val = acc[i][j];
            if (has_bias && biases != nullptr) {
              if constexpr (std::is_same_v<T, half>) {
                val +=
                    __half2float(__ldg(&biases[(size_t)expert_id * N + col]));
              } else {
                val += __bfloat162float(
                    __ldg(&biases[(size_t)expert_id * N + col]));
              }
            }
            if constexpr (std::is_same_v<T, half>) {
              output[(size_t)work_idx * N + col] = __float2half(val);
            } else {
              output[(size_t)work_idx * N + col] = __float2bfloat16(val);
            }
          }
        }
      }
    }
  }
}

// Small-M matmul: dot-product per output element, one warp per N, M rows per block.
// Grid: (ceil(N/BLOCK_N_SM), M)  Block: (BLOCK_N_SM * 32)
// Each warp computes one dot product over K, then warp-reduces.
constexpr int BLOCK_N_SM = 8;

template <typename T>
__launch_bounds__(BLOCK_N_SM * WARP_SIZE) __global__
    void mxfp4_matmul_smallm_kernel(const T *__restrict__ input,
                                    const uint8_t *__restrict__ weight,
                                    const uint8_t *__restrict__ weight_scale,
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
  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

  if (row >= M) return;

  const T *in_row = input + (size_t)row * K;
  const int smem_stride = K + CEILDIV(K, WARP_SIZE);

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

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int k = lane_id * MXFP4_BLOCK_SIZE; k < K;
       k += WARP_SIZE * MXFP4_BLOCK_SIZE) {
    float w_scale =
        e8m0_to_float(__ldg(&w_scale_row[k / MXFP4_BLOCK_SIZE])) * 0.5f;

    uint4 w_vec = *reinterpret_cast<const uint4 *>(w_row + k / 2);
    const float *in = s_input + (k + (k / WARP_SIZE));

    {
      int2 w_int8 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
      acc0 = fmaf(in[0], (float)(int8_t)(w_int8.x) * w_scale, acc0);
      acc1 = fmaf(in[1], (float)(int8_t)(w_int8.y) * w_scale, acc1);
      acc0 = fmaf(in[2], (float)(int8_t)(w_int8.x >> 8) * w_scale, acc0);
      acc1 = fmaf(in[3], (float)(int8_t)(w_int8.y >> 8) * w_scale, acc1);
      acc0 = fmaf(in[4], (float)(int8_t)(w_int8.x >> 16) * w_scale, acc0);
      acc1 = fmaf(in[5], (float)(int8_t)(w_int8.y >> 16) * w_scale, acc1);
      acc0 = fmaf(in[6], (float)(int8_t)(w_int8.x >> 24) * w_scale, acc0);
      acc1 = fmaf(in[7], (float)(int8_t)(w_int8.y >> 24) * w_scale, acc1);
    }
    {
      int2 w_int8 = get_int_from_table_16(w_vec.y, LUT0, LUT1, LUT2, LUT3);
      acc0 = fmaf(in[8], (float)(int8_t)(w_int8.x) * w_scale, acc0);
      acc1 = fmaf(in[9], (float)(int8_t)(w_int8.y) * w_scale, acc1);
      acc0 = fmaf(in[10], (float)(int8_t)(w_int8.x >> 8) * w_scale, acc0);
      acc1 = fmaf(in[11], (float)(int8_t)(w_int8.y >> 8) * w_scale, acc1);
      acc0 = fmaf(in[12], (float)(int8_t)(w_int8.x >> 16) * w_scale, acc0);
      acc1 = fmaf(in[13], (float)(int8_t)(w_int8.y >> 16) * w_scale, acc1);
      acc0 = fmaf(in[14], (float)(int8_t)(w_int8.x >> 24) * w_scale, acc0);
      acc1 = fmaf(in[15], (float)(int8_t)(w_int8.y >> 24) * w_scale, acc1);
    }
    {
      int2 w_int8 = get_int_from_table_16(w_vec.z, LUT0, LUT1, LUT2, LUT3);
      acc0 = fmaf(in[16], (float)(int8_t)(w_int8.x) * w_scale, acc0);
      acc1 = fmaf(in[17], (float)(int8_t)(w_int8.y) * w_scale, acc1);
      acc0 = fmaf(in[18], (float)(int8_t)(w_int8.x >> 8) * w_scale, acc0);
      acc1 = fmaf(in[19], (float)(int8_t)(w_int8.y >> 8) * w_scale, acc1);
      acc0 = fmaf(in[20], (float)(int8_t)(w_int8.x >> 16) * w_scale, acc0);
      acc1 = fmaf(in[21], (float)(int8_t)(w_int8.y >> 16) * w_scale, acc1);
      acc0 = fmaf(in[22], (float)(int8_t)(w_int8.x >> 24) * w_scale, acc0);
      acc1 = fmaf(in[23], (float)(int8_t)(w_int8.y >> 24) * w_scale, acc1);
    }
    {
      int2 w_int8 = get_int_from_table_16(w_vec.w, LUT0, LUT1, LUT2, LUT3);
      acc0 = fmaf(in[24], (float)(int8_t)(w_int8.x) * w_scale, acc0);
      acc1 = fmaf(in[25], (float)(int8_t)(w_int8.y) * w_scale, acc1);
      acc0 = fmaf(in[26], (float)(int8_t)(w_int8.x >> 8) * w_scale, acc0);
      acc1 = fmaf(in[27], (float)(int8_t)(w_int8.y >> 8) * w_scale, acc1);
      acc0 = fmaf(in[28], (float)(int8_t)(w_int8.x >> 16) * w_scale, acc0);
      acc1 = fmaf(in[29], (float)(int8_t)(w_int8.y >> 16) * w_scale, acc1);
      acc0 = fmaf(in[30], (float)(int8_t)(w_int8.x >> 24) * w_scale, acc0);
      acc1 = fmaf(in[31], (float)(int8_t)(w_int8.y >> 24) * w_scale, acc1);
    }
  }

  float acc = acc0 + acc1;
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

} // namespace mxfp4_gemm

// ============================================================================
// C API
// ============================================================================

extern "C" void mxfp4_matmul_smallm_f16(const __half *input,
                                         const uint8_t *weight,
                                         const uint8_t *weight_scale,
                                         const __half *bias, __half *output,
                                         int M, int N, int K, bool has_bias,
                                         cudaStream_t stream) {
  using namespace mxfp4_gemm;
  constexpr int THREADS = BLOCK_N_SM * WARP_SIZE;
  dim3 block(THREADS);
  dim3 grid(CEILDIV(N, BLOCK_N_SM), M);
  size_t smem = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

  mxfp4_gemm::mxfp4_matmul_smallm_kernel<half>
      <<<grid, block, smem, stream>>>(input, weight, weight_scale, bias,
                                      output, M, N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_BF16_KERNEL
extern "C" void mxfp4_matmul_smallm_bf16(const __nv_bfloat16 *input,
                                          const uint8_t *weight,
                                          const uint8_t *weight_scale,
                                          const __nv_bfloat16 *bias,
                                          __nv_bfloat16 *output,
                                          int M, int N, int K, bool has_bias,
                                          cudaStream_t stream) {
  using namespace mxfp4_gemm;
  constexpr int THREADS = BLOCK_N_SM * WARP_SIZE;
  dim3 block(THREADS);
  dim3 grid(CEILDIV(N, BLOCK_N_SM), M);
  size_t smem = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

  mxfp4_gemm::mxfp4_matmul_smallm_kernel<__nv_bfloat16>
      <<<grid, block, smem, stream>>>(input, weight, weight_scale, bias,
                                      output, M, N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}
#endif

extern "C" void mxfp4_matmul_f16(const __half *input,
                                  const uint8_t *weight,
                                  const uint8_t *weight_scale,
                                  const __half *bias, __half *output,
                                  int M, int N, int K, bool has_bias,
                                  cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_M = BM / TM;

  dim3 block(THREADS_N, THREADS_M);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));

  mxfp4_gemm::mxfp4_matmul_tiled<half, BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, bias, output, M,
                                   N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_BF16_KERNEL
extern "C" void
mxfp4_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                   const uint8_t *weight_scale, const __nv_bfloat16 *bias,
                   __nv_bfloat16 *output, int M, int N, int K,
                   bool has_bias, cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_M = BM / TM;

  dim3 block(THREADS_N, THREADS_M);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));

  mxfp4_gemm::mxfp4_matmul_tiled<__nv_bfloat16, BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, bias, output, M,
                                   N, K, has_bias);
  CUDA_CHECK(cudaGetLastError());
}
#endif

extern "C" void mxfp4_indexed_moe_gemm_f16(
    const __half *input, const uint8_t *weights, const uint8_t *weight_scales,
    const __half *biases, const uint32_t *indices, __half *output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim, cudaStream_t stream) {
  constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
  int n_chunks = CEILDIV(N, MOE_BLOCK_N);

  int total_blocks =
      input_has_topk_dim ? num_tokens * topk * n_chunks : num_tokens * n_chunks;

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(total_blocks);
  size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

  mxfp4_gemm::mxfp4_moe_gemm<half><<<grid, block, shared_mem_size, stream>>>(
      input, weights, weight_scales, biases, indices, output, num_tokens, topk,
      num_experts, N, K, has_bias, input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_BF16_KERNEL
extern "C" void mxfp4_indexed_moe_gemm_bf16(
    const __nv_bfloat16 *input, const uint8_t *weights,
    const uint8_t *weight_scales, const __nv_bfloat16 *biases,
    const uint32_t *indices, __nv_bfloat16 *output, int num_tokens, int topk,
    int num_experts, int N, int K, bool has_bias, bool input_has_topk_dim,
    cudaStream_t stream) {
  constexpr int THREADS_PER_BLOCK = MOE_BLOCK_N * 32;
  int n_chunks = CEILDIV(N, MOE_BLOCK_N);

  int total_blocks =
      input_has_topk_dim ? num_tokens * topk * n_chunks : num_tokens * n_chunks;

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(total_blocks);
  size_t shared_mem_size = (K + CEILDIV(K, WARP_SIZE)) * sizeof(float);

  mxfp4_gemm::mxfp4_moe_gemm<__nv_bfloat16>
      <<<grid, block, shared_mem_size, stream>>>(
          input, weights, weight_scales, biases, indices, output, num_tokens,
          topk, num_experts, N, K, has_bias, input_has_topk_dim);
  CUDA_CHECK(cudaGetLastError());
}
#endif

extern "C" int mxfp4_get_max_smem_optin() {
  int max_smem = 0;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         dev);
  return max_smem;
}

template <typename T, int BM, int BN, int BK, int TM, int TN>
static void launch_moe_tiled(
    const T *input, const uint8_t *weights, const uint8_t *weight_scales,
    const T *biases, const uint32_t *indices, T *output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim, cudaStream_t stream) {

  int total_work = num_tokens * topk;
  int list_stride = total_work;

  int *g_token_lists = nullptr;
  int *g_expert_counts = nullptr;
  size_t lists_bytes = (size_t)num_experts * list_stride * sizeof(int);
  size_t counts_bytes = (size_t)num_experts * sizeof(int);
  CUDA_CHECK(cudaMallocAsync(&g_token_lists, lists_bytes, stream));
  CUDA_CHECK(cudaMallocAsync(&g_expert_counts, counts_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(g_expert_counts, 0, counts_bytes, stream));

  // Phase 1: scatter tokens to per-expert lists
  {
    dim3 scatter_block(BN / TN, BM / TM);
    dim3 scatter_grid(1, num_experts);
    mxfp4_gemm::mxfp4_moe_scatter_tokens_tiled<<<scatter_grid, scatter_block, 0, stream>>>(
        indices, g_token_lists, g_expert_counts,
        total_work, num_experts, list_stride);
    CUDA_CHECK(cudaGetLastError());
  }

  // Phase 2: tiled GEMM (no dynamic shared memory needed for token lists)
  {
    dim3 block(BN / TN, BM / TM);
    dim3 grid(CEILDIV(N, BN), num_experts);

    mxfp4_gemm::mxfp4_moe_grouped_gemm_tiled<T, BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(
            input, weights, weight_scales, biases, indices, output,
            num_tokens, topk, num_experts, N, K, has_bias, input_has_topk_dim,
            g_token_lists, g_expert_counts, list_stride);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaFreeAsync(g_token_lists, stream));
  CUDA_CHECK(cudaFreeAsync(g_expert_counts, stream));
}

extern "C" void mxfp4_moe_grouped_gemm_f16(
    const __half *input, const uint8_t *weights, const uint8_t *weight_scales,
    const __half *biases, const uint32_t *indices, __half *output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim, cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  launch_moe_tiled<half, BM, BN, BK, TM, TN>(
      input, weights, weight_scales, biases, indices, output,
      num_tokens, topk, num_experts, N, K, has_bias, input_has_topk_dim, stream);
}

#ifndef NO_BF16_KERNEL
extern "C" void mxfp4_moe_grouped_gemm_bf16(
    const __nv_bfloat16 *input, const uint8_t *weights,
    const uint8_t *weight_scales, const __nv_bfloat16 *biases,
    const uint32_t *indices, __nv_bfloat16 *output, int num_tokens, int topk,
    int num_experts, int N, int K, bool has_bias, bool input_has_topk_dim,
    cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  launch_moe_tiled<__nv_bfloat16, BM, BN, BK, TM, TN>(
      input, weights, weight_scales, biases, indices, output,
      num_tokens, topk, num_experts, N, K, has_bias, input_has_topk_dim, stream);
}
#endif
