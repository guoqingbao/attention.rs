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
#include <mma.h>
#include "attention/dtype_fp8.cuh"

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
//
// Each lane processes 2 NVFP4 blocks (32 elements) per stride iteration
// to improve ILP and amortise the scale-fetch overhead.
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

  // Load input row into shared memory with padding to avoid bank conflicts.
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

  // Each lane processes 2 consecutive NVFP4 blocks (32 elements) per step.
  // stride = warp_size * 32 = 1024 elements/iteration.
  constexpr int ELEMS_PER_LANE = 2 * NVFP4_BLOCK_SIZE; // 32
  for (int k = lane_id * ELEMS_PER_LANE; k < K;
       k += WARP_SIZE * ELEMS_PER_LANE) {

    // --- First NVFP4 block of 16 ---
    {
      float block_scale =
          dispatch_fp8_to_float(__ldg(&w_scale_row[k / NVFP4_BLOCK_SIZE])) *
          weight_global_scale * 0.5f;

      uint2 w_vec = *reinterpret_cast<const uint2 *>(w_row + k / 2);
      const float *in = s_input + (k + (k / WARP_SIZE));

      int2 w0 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
      acc = fmaf(in[0], (float)(int8_t)(w0.x) * block_scale, acc);
      acc = fmaf(in[1], (float)(int8_t)(w0.y) * block_scale, acc);
      acc = fmaf(in[2], (float)(int8_t)(w0.x >> 8) * block_scale, acc);
      acc = fmaf(in[3], (float)(int8_t)(w0.y >> 8) * block_scale, acc);
      acc = fmaf(in[4], (float)(int8_t)(w0.x >> 16) * block_scale, acc);
      acc = fmaf(in[5], (float)(int8_t)(w0.y >> 16) * block_scale, acc);
      acc = fmaf(in[6], (float)(int8_t)(w0.x >> 24) * block_scale, acc);
      acc = fmaf(in[7], (float)(int8_t)(w0.y >> 24) * block_scale, acc);

      int2 w1 = get_int_from_table_16(w_vec.y, LUT0, LUT1, LUT2, LUT3);
      acc = fmaf(in[8], (float)(int8_t)(w1.x) * block_scale, acc);
      acc = fmaf(in[9], (float)(int8_t)(w1.y) * block_scale, acc);
      acc = fmaf(in[10], (float)(int8_t)(w1.x >> 8) * block_scale, acc);
      acc = fmaf(in[11], (float)(int8_t)(w1.y >> 8) * block_scale, acc);
      acc = fmaf(in[12], (float)(int8_t)(w1.x >> 16) * block_scale, acc);
      acc = fmaf(in[13], (float)(int8_t)(w1.y >> 16) * block_scale, acc);
      acc = fmaf(in[14], (float)(int8_t)(w1.x >> 24) * block_scale, acc);
      acc = fmaf(in[15], (float)(int8_t)(w1.y >> 24) * block_scale, acc);
    }

    // --- Second NVFP4 block of 16 ---
    int k2 = k + NVFP4_BLOCK_SIZE;
    if (k2 < K) {
      float block_scale2 =
          dispatch_fp8_to_float(__ldg(&w_scale_row[k2 / NVFP4_BLOCK_SIZE])) *
          weight_global_scale * 0.5f;

      uint2 w_vec2 = *reinterpret_cast<const uint2 *>(w_row + k2 / 2);
      const float *in2 = s_input + (k2 + (k2 / WARP_SIZE));

      int2 w2a = get_int_from_table_16(w_vec2.x, LUT0, LUT1, LUT2, LUT3);
      acc = fmaf(in2[0], (float)(int8_t)(w2a.x) * block_scale2, acc);
      acc = fmaf(in2[1], (float)(int8_t)(w2a.y) * block_scale2, acc);
      acc = fmaf(in2[2], (float)(int8_t)(w2a.x >> 8) * block_scale2, acc);
      acc = fmaf(in2[3], (float)(int8_t)(w2a.y >> 8) * block_scale2, acc);
      acc = fmaf(in2[4], (float)(int8_t)(w2a.x >> 16) * block_scale2, acc);
      acc = fmaf(in2[5], (float)(int8_t)(w2a.y >> 16) * block_scale2, acc);
      acc = fmaf(in2[6], (float)(int8_t)(w2a.x >> 24) * block_scale2, acc);
      acc = fmaf(in2[7], (float)(int8_t)(w2a.y >> 24) * block_scale2, acc);

      int2 w2b = get_int_from_table_16(w_vec2.y, LUT0, LUT1, LUT2, LUT3);
      acc = fmaf(in2[8], (float)(int8_t)(w2b.x) * block_scale2, acc);
      acc = fmaf(in2[9], (float)(int8_t)(w2b.y) * block_scale2, acc);
      acc = fmaf(in2[10], (float)(int8_t)(w2b.x >> 8) * block_scale2, acc);
      acc = fmaf(in2[11], (float)(int8_t)(w2b.y >> 8) * block_scale2, acc);
      acc = fmaf(in2[12], (float)(int8_t)(w2b.x >> 16) * block_scale2, acc);
      acc = fmaf(in2[13], (float)(int8_t)(w2b.y >> 16) * block_scale2, acc);
      acc = fmaf(in2[14], (float)(int8_t)(w2b.x >> 24) * block_scale2, acc);
      acc = fmaf(in2[15], (float)(int8_t)(w2b.y >> 24) * block_scale2, acc);
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

    constexpr int MOE_ELEMS_PER_LANE = 2 * NVFP4_BLOCK_SIZE; // 32
    for (int k = lane_id * MOE_ELEMS_PER_LANE; k < K;
         k += WARP_SIZE * MOE_ELEMS_PER_LANE) {
      // --- First NVFP4 block of 16 ---
      {
        float block_scale =
            dispatch_fp8_to_float(__ldg(&w_scale_row[k / NVFP4_BLOCK_SIZE])) *
            global_scale * 0.5f;

        uint2 w_vec = *reinterpret_cast<const uint2 *>(w_row + k / 2);
        const float *in = s_input_padded + (k + (k / WARP_SIZE));

        int2 w0 = get_int_from_table_16(w_vec.x, LUT0, LUT1, LUT2, LUT3);
        acc = fmaf(in[0], (float)(int8_t)(w0.x) * block_scale, acc);
        acc = fmaf(in[1], (float)(int8_t)(w0.y) * block_scale, acc);
        acc = fmaf(in[2], (float)(int8_t)(w0.x >> 8) * block_scale, acc);
        acc = fmaf(in[3], (float)(int8_t)(w0.y >> 8) * block_scale, acc);
        acc = fmaf(in[4], (float)(int8_t)(w0.x >> 16) * block_scale, acc);
        acc = fmaf(in[5], (float)(int8_t)(w0.y >> 16) * block_scale, acc);
        acc = fmaf(in[6], (float)(int8_t)(w0.x >> 24) * block_scale, acc);
        acc = fmaf(in[7], (float)(int8_t)(w0.y >> 24) * block_scale, acc);

        int2 w1 = get_int_from_table_16(w_vec.y, LUT0, LUT1, LUT2, LUT3);
        acc = fmaf(in[8], (float)(int8_t)(w1.x) * block_scale, acc);
        acc = fmaf(in[9], (float)(int8_t)(w1.y) * block_scale, acc);
        acc = fmaf(in[10], (float)(int8_t)(w1.x >> 8) * block_scale, acc);
        acc = fmaf(in[11], (float)(int8_t)(w1.y >> 8) * block_scale, acc);
        acc = fmaf(in[12], (float)(int8_t)(w1.x >> 16) * block_scale, acc);
        acc = fmaf(in[13], (float)(int8_t)(w1.y >> 16) * block_scale, acc);
        acc = fmaf(in[14], (float)(int8_t)(w1.x >> 24) * block_scale, acc);
        acc = fmaf(in[15], (float)(int8_t)(w1.y >> 24) * block_scale, acc);
      }

      // --- Second NVFP4 block of 16 ---
      int k2 = k + NVFP4_BLOCK_SIZE;
      if (k2 < K) {
        float block_scale2 =
            dispatch_fp8_to_float(__ldg(&w_scale_row[k2 / NVFP4_BLOCK_SIZE])) *
            global_scale * 0.5f;

        uint2 w_vec2 = *reinterpret_cast<const uint2 *>(w_row + k2 / 2);
        const float *in2 = s_input_padded + (k2 + (k2 / WARP_SIZE));

        int2 w2a = get_int_from_table_16(w_vec2.x, LUT0, LUT1, LUT2, LUT3);
        acc = fmaf(in2[0], (float)(int8_t)(w2a.x) * block_scale2, acc);
        acc = fmaf(in2[1], (float)(int8_t)(w2a.y) * block_scale2, acc);
        acc = fmaf(in2[2], (float)(int8_t)(w2a.x >> 8) * block_scale2, acc);
        acc = fmaf(in2[3], (float)(int8_t)(w2a.y >> 8) * block_scale2, acc);
        acc = fmaf(in2[4], (float)(int8_t)(w2a.x >> 16) * block_scale2, acc);
        acc = fmaf(in2[5], (float)(int8_t)(w2a.y >> 16) * block_scale2, acc);
        acc = fmaf(in2[6], (float)(int8_t)(w2a.x >> 24) * block_scale2, acc);
        acc = fmaf(in2[7], (float)(int8_t)(w2a.y >> 24) * block_scale2, acc);

        int2 w2b = get_int_from_table_16(w_vec2.y, LUT0, LUT1, LUT2, LUT3);
        acc = fmaf(in2[8], (float)(int8_t)(w2b.x) * block_scale2, acc);
        acc = fmaf(in2[9], (float)(int8_t)(w2b.y) * block_scale2, acc);
        acc = fmaf(in2[10], (float)(int8_t)(w2b.x >> 8) * block_scale2, acc);
        acc = fmaf(in2[11], (float)(int8_t)(w2b.y >> 8) * block_scale2, acc);
        acc = fmaf(in2[12], (float)(int8_t)(w2b.x >> 16) * block_scale2, acc);
        acc = fmaf(in2[13], (float)(int8_t)(w2b.y >> 16) * block_scale2, acc);
        acc = fmaf(in2[14], (float)(int8_t)(w2b.x >> 24) * block_scale2, acc);
        acc = fmaf(in2[15], (float)(int8_t)(w2b.y >> 24) * block_scale2, acc);
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

// ============================================================================
// WMMA-based NVFP4 GEMM for SM80+ (Tensor Cores)
//
// BK=16 (matches NVFP4_BLOCK_SIZE exactly: 1 scale per row per K-tile).
// BM=64, BN=64, 128 threads (4 warps).
// Each warp computes a 32×32 sub-tile via 2×2 WMMA 16×16×16 fragments.
//
// B-tile dequant: 64 rows × 16 elements = 1024 elements.
// Each of 64 threads handles one row (16 elements = one NVFP4 block).
// Remaining 64 threads help load A-tile.
//
// Double-buffering for shared memory to overlap dequant with compute.
// ============================================================================

#define WMMA_BK 16

template <typename T, int BM, int BN>
__global__ void nvfp4_wmma_matmul_kernel(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ weight_scale,
    float weight_global_scale,
    const T *__restrict__ bias,
    T *__restrict__ output,
    int M, int N, int K, bool has_bias) {

  using namespace nvcuda::wmma;
  using namespace nvfp4_gemm;

  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int scale_stride = CEILDIV(K, NVFP4_BLOCK_SIZE);
  const int half_k = K / 2;

  const int warp_row = (warp_id / 2) * 32;
  const int warp_col = (warp_id % 2) * 32;

  fragment<accumulator, 16, 16, 16, float> acc[2][2];
  #pragma unroll
  for (int i = 0; i < 2; i++)
    #pragma unroll
    for (int j = 0; j < 2; j++)
      fill_fragment(acc[i][j], 0.0f);

  __shared__ T s_a[BM][WMMA_BK + 4]; // +4 padding
  __shared__ T s_b[BN][WMMA_BK + 4];

  for (int k_step = 0; k_step < K; k_step += WMMA_BK) {

    // --- Load A [BM × 16]: 128 threads, 64*16=1024 elements, 8 per thread ---
    #pragma unroll
    for (int i = tid; i < BM * WMMA_BK; i += 128) {
      int r = i / WMMA_BK;
      int c = i % WMMA_BK;
      int gr = by * BM + r;
      int gc = k_step + c;
      s_a[r][c] = (gr < M && gc < K) ? input[(size_t)gr * K + gc] : T(0);
    }

    // --- Dequant B [BN × 16]: distribute 64 rows across 128 threads ---
    // Each row = one NVFP4 block of 16 packed in 8 bytes (uint2)
    // 128 threads, 64 rows → 2 threads per row (each handles 8 elements)
    {
      int row = tid / 2;           // 0..63
      int sub = tid & 1;           // 0: low 8 elements, 1: high 8 elements
      int gn = bx * BN + row;

      if (row < BN && gn < N && k_step < K) {
        float block_scale = dispatch_fp8_to_float(
            __ldg(&weight_scale[(size_t)gn * scale_stride + k_step / NVFP4_BLOCK_SIZE]))
            * weight_global_scale * 0.5f;

        uint2 w_vec = *reinterpret_cast<const uint2 *>(
            &weight[(size_t)gn * half_k + k_step / 2]);

        uint32_t word = sub ? w_vec.y : w_vec.x;
        int col_base = sub * 8;
        float dq[8];
        dequant_store_8(word, block_scale, LUT0, LUT1, LUT2, LUT3, dq);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          if constexpr (std::is_same_v<T, __half>)
            s_b[row][col_base + j] = __float2half(dq[j]);
          else
            s_b[row][col_base + j] = __float2bfloat16(dq[j]);
        }
      } else if (row < BN) {
        int col_base = sub * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++)
          s_b[row][col_base + j] = T(0);
      }
    }

    __syncthreads();

    // --- WMMA compute ---
    fragment<matrix_a, 16, 16, 16, T, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, T, col_major> b_frag;

    #pragma unroll
    for (int fi = 0; fi < 2; fi++) {
      #pragma unroll
      for (int fj = 0; fj < 2; fj++) {
        load_matrix_sync(a_frag, &s_a[warp_row + fi * 16][0], WMMA_BK + 4);
        load_matrix_sync(b_frag, &s_b[warp_col + fj * 16][0], WMMA_BK + 4);
        mma_sync(acc[fi][fj], a_frag, b_frag, acc[fi][fj]);
      }
    }
    __syncthreads();
  }

  // --- Store output: from accumulators through shared memory ---
  __shared__ float s_out[BM][BN + 4];

  #pragma unroll
  for (int fi = 0; fi < 2; fi++)
    #pragma unroll
    for (int fj = 0; fj < 2; fj++)
      store_matrix_sync(&s_out[warp_row + fi * 16][warp_col + fj * 16],
                        acc[fi][fj], BN + 4, mem_row_major);
  __syncthreads();

  for (int i = tid; i < BM * BN; i += 128) {
    int r = i / BN;
    int c = i % BN;
    int gr = by * BM + r;
    int gc = bx * BN + c;
    if (gr < M && gc < N) {
      float val = s_out[r][c];
      if (has_bias && bias != nullptr) {
        if constexpr (std::is_same_v<T, __half>)
          val += __half2float(__ldg(&bias[gc]));
        else
          val += __bfloat162float(__ldg(&bias[gc]));
      }
      if constexpr (std::is_same_v<T, __half>)
        output[(size_t)gr * N + gc] = __float2half(val);
      else
        output[(size_t)gr * N + gc] = __float2bfloat16(val);
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
  auto kernel = nvfp4_gemm::nvfp4_matmul_smallm_kernel<half>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kernel<<<grid, block, smem, stream>>>(input, weight, weight_scale,
                                        weight_global_scale, bias, output, M, N,
                                        K, has_bias);
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
  auto kernel = nvfp4_gemm::nvfp4_matmul_smallm_kernel<__nv_bfloat16>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kernel<<<grid, block, smem, stream>>>(input, weight, weight_scale,
                                        weight_global_scale, bias, output, M, N,
                                        K, has_bias);
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
#ifndef NO_BF16_KERNEL
  constexpr int BM = 64, BN = 64;
  dim3 block(128);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
  nvfp4_gemm::nvfp4_wmma_matmul_kernel<half, BM, BN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale,
                                   weight_global_scale, bias, output, M, N, K,
                                   has_bias);
#else
  constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_M = BM / TM;
  dim3 block(THREADS_N, THREADS_M);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
  nvfp4_gemm::nvfp4_matmul_tiled<half, BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale,
                                   weight_global_scale, bias, output, M, N, K,
                                   has_bias);
#endif
}

#ifndef NO_BF16_KERNEL
extern "C" void nvfp4_matmul_bf16(const __nv_bfloat16 *input,
                                   const uint8_t *weight,
                                   const uint8_t *weight_scale,
                                   float weight_global_scale,
                                   const __nv_bfloat16 *bias,
                                   __nv_bfloat16 *output, int M, int N, int K,
                                   bool has_bias, cudaStream_t stream) {
  constexpr int BM = 64, BN = 64;
  dim3 block(128);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
  nvfp4_gemm::nvfp4_wmma_matmul_kernel<__nv_bfloat16, BM, BN>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale,
                                   weight_global_scale, bias, output, M, N, K,
                                   has_bias);
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
  auto kernel = nvfp4_gemm::nvfp4_moe_gemm<half>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
  kernel<<<grid, block, shared_mem_size, stream>>>(
      input, weights, weight_scales, weight_global_scales, biases, indices,
      output, num_tokens, topk, num_experts, N, K, has_bias,
      input_has_topk_dim);
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
  auto kernel = nvfp4_gemm::nvfp4_moe_gemm<__nv_bfloat16>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
  kernel<<<grid, block, shared_mem_size, stream>>>(
      input, weights, weight_scales, weight_global_scales, biases, indices,
      output, num_tokens, topk, num_experts, N, K, has_bias,
      input_has_topk_dim);
}
#else
extern "C" void nvfp4_indexed_moe_gemm_bf16(const void *, const uint8_t *,
                                             const uint8_t *, const float *,
                                             const void *, const uint32_t *,
                                             void *, int, int, int, int, int,
                                             bool, bool, cudaStream_t) {}
#endif


// ============================================================================
// WMMA-based grouped MoE GEMM for NVFP4 (SM80+)
//
// Each block handles one expert segment (sorted tokens) × one N-tile.
// B-tile loads dequantize NVFP4 weights into shared memory, then uses WMMA.
//
// Grid: (num_experts, ceil(N/N_BLK))
// Block: 128 threads (4 warps), 2×2 warp layout → 32×32 output tile.
// ============================================================================
namespace nvfp4_gemm {

constexpr int MOE_WMMA_K = 16;
constexpr int MOE_M_BLK = 32;
constexpr int MOE_N_BLK = 32;
constexpr int MOE_WARPS = 4;
constexpr int MOE_THREADS = MOE_WARPS * 32;

template<typename T>
__global__ void nvfp4_moe_gemm_wmma_kernel(
    const T* __restrict__ input,               // [num_input_rows, K]
    const uint8_t* __restrict__ weights,       // [E, N, K/2] packed FP4
    const uint8_t* __restrict__ weight_scales,  // [E, N, K/16] FP8 E4M3
    const float* __restrict__ weight_global_scales, // [E]
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    T* __restrict__ output,                    // [num_input_rows, N] (zero-init)
    const int num_experts, const int topk,
    const int32_t size_m, const int32_t size_n, const int32_t size_k,
    const bool input_has_topk_dim
) {
    using namespace nvcuda::wmma;

    const uint32_t LUT0 = 0x03020100;
    const uint32_t LUT1 = 0x0C080604;
    const uint32_t LUT2 = 0xFDFEFF00;
    const uint32_t LUT3 = 0xF4F8FAFC;

    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows = segment_end - segment_start;
    if (num_rows == 0) return;

    const int n_base = n_tile_idx * MOE_N_BLK;
    if (n_base >= size_n) return;

    const float global_scale = weight_global_scales[expert_id];
    const int half_k = size_k / 2;
    const int scale_stride = CEILDIV(size_k, NVFP4_BLOCK_SIZE);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m_idx = warp_id / 2;
    const int warp_n_idx = warp_id % 2;

    extern __shared__ uint8_t smem_raw[];
    T* s_a = reinterpret_cast<T*>(smem_raw);                       // [M_BLK, K_BLK]
    T* s_b = s_a + MOE_M_BLK * MOE_WMMA_K;                        // [N_BLK, K_BLK + 4]
    float* s_c = reinterpret_cast<float*>(s_b + MOE_N_BLK * (MOE_WMMA_K + 4));  // [M_BLK, N_BLK]

    for (int m_base = 0; m_base < num_rows; m_base += MOE_M_BLK) {
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k_base = 0; k_base < size_k; k_base += MOE_WMMA_K) {
            // Load A tile: inputs for this expert segment
            // sorted_token_ids maps to flat [num_tokens * topk] indices.
            // The actual input row = tok_idx / topk (input shape: [num_input_rows, K])
            constexpr int A_ELEMS = MOE_M_BLK * MOE_WMMA_K;
            for (int i = tid; i < A_ELEMS; i += MOE_THREADS) {
                int m_local = i / MOE_WMMA_K;
                int k_local = i % MOE_WMMA_K;
                int m_seg = m_base + m_local;
                int k_global = k_base + k_local;
                if (m_seg < num_rows && k_global < size_k) {
                    int tok_pair_idx = segment_start + m_seg;
                    int tok_idx = sorted_token_ids[tok_pair_idx];
                    int input_row = input_has_topk_dim ? tok_idx : (tok_idx / topk);
                    s_a[m_local * MOE_WMMA_K + k_local] = input[(size_t)input_row * size_k + k_global];
                } else {
                    s_a[m_local * MOE_WMMA_K + k_local] = T(0);
                }
            }

            // Dequant B tile: NVFP4 weights for this expert
            // 32 N-rows × 16 K-elements = 32 dequant units, 128 threads → 4 threads/row
            // Each row = 1 NVFP4 block of 16 packed in uint2 (8 bytes)
            // Split: 2 threads per row (each handles 8 elements via one dequant_store_8 call)
            constexpr int B_ROWS = MOE_N_BLK;
            constexpr int B_STRIDE = MOE_WMMA_K + 4;
            for (int i = tid; i < B_ROWS * 2; i += MOE_THREADS) {
                int row = i / 2;
                int sub = i & 1;  // 0: low 8, 1: high 8
                int gn = n_base + row;
                if (gn < size_n && k_base < size_k) {
                    float block_scale = dispatch_fp8_to_float(
                        __ldg(&weight_scales[(size_t)expert_id * size_n * scale_stride +
                                             (size_t)gn * scale_stride + k_base / NVFP4_BLOCK_SIZE]))
                        * global_scale * 0.5f;

                    uint2 w_vec = *reinterpret_cast<const uint2*>(
                        &weights[(size_t)expert_id * size_n * half_k + (size_t)gn * half_k + k_base / 2]);

                    uint32_t word = sub ? w_vec.y : w_vec.x;
                    int col_base = sub * 8;
                    float dq[8];
                    dequant_store_8(word, block_scale, LUT0, LUT1, LUT2, LUT3, dq);
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        if constexpr (std::is_same_v<T, __half>)
                            s_b[row * B_STRIDE + col_base + j] = __float2half(dq[j]);
                        else
                            s_b[row * B_STRIDE + col_base + j] = __float2bfloat16(dq[j]);
                    }
                } else {
                    int col_base = sub * 8;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        s_b[row * B_STRIDE + col_base + j] = T(0);
                }
            }

            __syncthreads();

            fragment<matrix_a, 16, 16, 16, T, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, T, col_major> b_frag;

            load_matrix_sync(a_frag, s_a + warp_m_idx * 16 * MOE_WMMA_K, MOE_WMMA_K);
            load_matrix_sync(b_frag, s_b + warp_n_idx * 16 * B_STRIDE, B_STRIDE);
            mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }

        // Store accumulated results
        store_matrix_sync(s_c + warp_m_idx * 16 * MOE_N_BLK + warp_n_idx * 16,
                          c_frag, MOE_N_BLK, mem_row_major);
        __syncthreads();

        constexpr int C_ELEMS = MOE_M_BLK * MOE_N_BLK;
        for (int i = tid; i < C_ELEMS; i += MOE_THREADS) {
            int m_local = i / MOE_N_BLK;
            int n_local = i % MOE_N_BLK;
            int m_seg = m_base + m_local;
            int n_global = n_base + n_local;
            if (m_seg < num_rows && n_global < size_n) {
                int tok_pair_idx = segment_start + m_seg;
                if (tok_pair_idx < size_m) {
                    int tok_idx = sorted_token_ids[tok_pair_idx];
                    float val = s_c[m_local * MOE_N_BLK + n_local];
                    if (topk_weights) val *= topk_weights[tok_idx];
                    if constexpr (std::is_same_v<T, __half>)
                        output[(size_t)tok_idx * size_n + n_global] = __float2half(val);
                    else
                        output[(size_t)tok_idx * size_n + n_global] = __float2bfloat16(val);
                }
            }
        }
        __syncthreads();
    }
}

} // namespace nvfp4_gemm

// C API for NVFP4 MoE WMMA grouped GEMM
extern "C" void nvfp4_moe_gemm_wmma_f16(
    const __half *input, const uint8_t *weights, const uint8_t *weight_scales,
    const float *weight_global_scales,
    const int32_t *sorted_token_ids, const int32_t *expert_offsets,
    const float *topk_weights,
    __half *output,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    bool input_has_topk_dim,
    cudaStream_t stream) {
  using namespace nvfp4_gemm;
  dim3 grid(num_experts, CEILDIV(size_n, MOE_N_BLK));
  dim3 block(MOE_THREADS);
  size_t smem = MOE_M_BLK * MOE_WMMA_K * sizeof(__half)
              + MOE_N_BLK * (MOE_WMMA_K + 4) * sizeof(__half)
              + MOE_M_BLK * MOE_N_BLK * sizeof(float);
  nvfp4_moe_gemm_wmma_kernel<__half><<<grid, block, smem, stream>>>(
      input, weights, weight_scales, weight_global_scales,
      sorted_token_ids, expert_offsets, topk_weights,
      output, num_experts, topk, size_m, size_n, size_k,
      input_has_topk_dim);
}

#ifndef NO_BF16_KERNEL
extern "C" void nvfp4_moe_gemm_wmma_bf16(
    const __nv_bfloat16 *input, const uint8_t *weights, const uint8_t *weight_scales,
    const float *weight_global_scales,
    const int32_t *sorted_token_ids, const int32_t *expert_offsets,
    const float *topk_weights,
    __nv_bfloat16 *output,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    bool input_has_topk_dim,
    cudaStream_t stream) {
  using namespace nvfp4_gemm;
  dim3 grid(num_experts, CEILDIV(size_n, MOE_N_BLK));
  dim3 block(MOE_THREADS);
  size_t smem = MOE_M_BLK * MOE_WMMA_K * sizeof(__nv_bfloat16)
              + MOE_N_BLK * (MOE_WMMA_K + 4) * sizeof(__nv_bfloat16)
              + MOE_M_BLK * MOE_N_BLK * sizeof(float);
  nvfp4_moe_gemm_wmma_kernel<__nv_bfloat16><<<grid, block, smem, stream>>>(
      input, weights, weight_scales, weight_global_scales,
      sorted_token_ids, expert_offsets, topk_weights,
      output, num_experts, topk, size_m, size_n, size_k,
      input_has_topk_dim);
}
#else
extern "C" void nvfp4_moe_gemm_wmma_bf16(
    const void *, const uint8_t *, const uint8_t *, const float *,
    const int32_t *, const int32_t *, const float *,
    void *, int, int, int, int, int, bool, cudaStream_t) {}
#endif
