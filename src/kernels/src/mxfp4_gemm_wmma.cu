/**
 * @brief CUDA kernels for MXFP4 GEMM and MoE GEMM using WMMA (Warp Matrix
 *        Multiply-Accumulate) Tensor Core acceleration (Ampere+ / SM >= 80).
 *
 * This file implements two WMMA-accelerated kernel families:
 *   1. mxfp4_matmul_wmma_kernel            – Dense GEMM for medium-to-large M
 *      (prefill), dequantizing MXFP4 weights to FP16/BF16 in shared memory
 *      then accumulating via 16x16x16 WMMA fragments.
 *   2. mxfp4_moe_grouped_gemm_wmma_kernel  – Mixture-of-Experts GEMM with
 *      top-k expert selection, using the same WMMA tiled approach with
 *      per-expert segment indexing and dynamic shared memory.
 *
 * MXFP4 Format (OCP Microscaling):
 * - FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
 * - Block size: 32 elements per scale
 * - Scale: E8M0 format (8-bit exponent, stored as u8 with bias 127)
 * - 2 FP4 values packed per byte (nibbles)
 *
 * Copyright (c) 2025, Eric L Buehler.  All rights reserved.
 * https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-quant/kernels/mxfp4/mxfp4_gemm_wmma.cu
 * Notes:
 * - Block tile: 64x64x32 (M_BLK x N_BLK x K_BLK), 8 warps (4x2), 256 threads
 * - Vectorized uint4 loads for weights, LUT-based FP4 dequantization to FP16/BF16
 *   in shared memory before WMMA fragment loads
 * - Shared memory layout: tiles for A (input), B (dequantized weights), and C
 *   (output accumulator)
 * - MoE kernel uses dynamic shared memory for segment offset tables
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

using namespace nvcuda::wmma;

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define MXFP4_BLOCK_SIZE 32

namespace mxfp4_wmma {

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

__device__ __forceinline__ void dequant_store_8_f16(int q4, float scale,
                                                    uint32_t L0, uint32_t L1,
                                                    uint32_t L2, uint32_t L3,
                                                    half *dst) {
  int2 w = get_int_from_table_16(q4, L0, L1, L2, L3);
  dst[0] = __float2half((float)(int8_t)(w.x) * scale);
  dst[1] = __float2half((float)(int8_t)(w.y) * scale);
  dst[2] = __float2half((float)(int8_t)(w.x >> 8) * scale);
  dst[3] = __float2half((float)(int8_t)(w.y >> 8) * scale);
  dst[4] = __float2half((float)(int8_t)(w.x >> 16) * scale);
  dst[5] = __float2half((float)(int8_t)(w.y >> 16) * scale);
  dst[6] = __float2half((float)(int8_t)(w.x >> 24) * scale);
  dst[7] = __float2half((float)(int8_t)(w.y >> 24) * scale);
}

#ifndef NO_BF16_KERNEL
__device__ __forceinline__ void dequant_store_8_bf16(int q4, float scale,
                                                     uint32_t L0, uint32_t L1,
                                                     uint32_t L2, uint32_t L3,
                                                     __nv_bfloat16 *dst) {
  int2 w = get_int_from_table_16(q4, L0, L1, L2, L3);
  dst[0] = __float2bfloat16((float)(int8_t)(w.x) * scale);
  dst[1] = __float2bfloat16((float)(int8_t)(w.y) * scale);
  dst[2] = __float2bfloat16((float)(int8_t)(w.x >> 8) * scale);
  dst[3] = __float2bfloat16((float)(int8_t)(w.y >> 8) * scale);
  dst[4] = __float2bfloat16((float)(int8_t)(w.x >> 16) * scale);
  dst[5] = __float2bfloat16((float)(int8_t)(w.y >> 16) * scale);
  dst[6] = __float2bfloat16((float)(int8_t)(w.x >> 24) * scale);
  dst[7] = __float2bfloat16((float)(int8_t)(w.y >> 24) * scale);
}
#endif

template <typename T>
__device__ __forceinline__ void
dequant_store_8(int q4, float scale, uint32_t L0, uint32_t L1, uint32_t L2,
                uint32_t L3, T *dst);

template <>
__device__ __forceinline__ void
dequant_store_8<half>(int q4, float scale, uint32_t L0, uint32_t L1,
                      uint32_t L2, uint32_t L3, half *dst) {
  dequant_store_8_f16(q4, scale, L0, L1, L2, L3, dst);
}

#ifndef NO_BF16_KERNEL
template <>
__device__ __forceinline__ void
dequant_store_8<__nv_bfloat16>(int q4, float scale, uint32_t L0, uint32_t L1,
                               uint32_t L2, uint32_t L3, __nv_bfloat16 *dst) {
  dequant_store_8_bf16(q4, scale, L0, L1, L2, L3, dst);
}
#endif

constexpr int WMMA_M_DIM = 16;
constexpr int WMMA_N_DIM = 16;
constexpr int WMMA_K_DIM = 16;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32;

constexpr int M_BLK = WARPS_M * WMMA_M_DIM;
constexpr int N_BLK = WARPS_N * 2 * WMMA_N_DIM;
constexpr int K_BLK = MXFP4_BLOCK_SIZE;
constexpr int WMMA_K_STEPS = K_BLK / WMMA_K_DIM;

using VecT = float4;
constexpr int VEC_SIZE = 8;

template <typename T>
__launch_bounds__(BLOCK_THREADS) __global__
    void mxfp4_matmul_wmma_kernel(const T *__restrict__ input,
                                  const uint8_t *__restrict__ weight,
                                  const uint8_t *__restrict__ weight_scale,
                                  const T *__restrict__ bias,
                                  T *__restrict__ output, int M, int N, int K,
                                  bool has_bias) {
  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);

  extern __shared__ uint8_t smem_bytes[];

  T *A_sh = reinterpret_cast<T *>(smem_bytes);
  T *B_sh = A_sh + M_BLK * K_BLK;
  uint8_t *C_raw = reinterpret_cast<uint8_t *>(B_sh + N_BLK * K_BLK);
  size_t align_off = reinterpret_cast<uintptr_t>(C_raw) % alignof(float);
  if (align_off != 0)
    C_raw += (alignof(float) - align_off);
  float *C_sh = reinterpret_cast<float *>(C_raw);

  const int threadId = threadIdx.x;
  const int warpId = threadId / 32;
  const int warp_m_idx = warpId / WARPS_N;
  const int warp_n_idx = warpId % WARPS_N;

  const int m_base = blockIdx.y * M_BLK;
  const int n_base = blockIdx.x * N_BLK;

  VecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  fragment<accumulator, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, float> c_frag[2];
  fill_fragment(c_frag[0], 0.0f);
  fill_fragment(c_frag[1], 0.0f);

  for (int k_base = 0; k_base < K; k_base += K_BLK) {
    constexpr int A_VEC_ELEMS = M_BLK * K_BLK / VEC_SIZE;
    for (int i = threadId; i < A_VEC_ELEMS; i += BLOCK_THREADS) {
      const int idx = i * VEC_SIZE;
      const int lm = idx / K_BLK;
      const int lk = idx % K_BLK;
      const int gm = m_base + lm;
      const int gk = k_base + lk;

      if (gm < M && gk < K) {
        *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) =
            *reinterpret_cast<const VecT *>(&input[(size_t)gm * K + gk]);
      } else {
        *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) = zero_vec;
      }
    }

    for (int ln = threadId; ln < N_BLK; ln += BLOCK_THREADS) {
      const int gn = n_base + ln;
      if (gn < N) {
        uint4 w_vec = *reinterpret_cast<const uint4 *>(
            &weight[(size_t)gn * (K / 2) + k_base / 2]);
        float scale =
            e8m0_to_float(__ldg(&weight_scale[(size_t)gn * scale_stride +
                                              k_base / MXFP4_BLOCK_SIZE])) *
            0.5f;

        T *dst = &B_sh[ln * K_BLK];
        dequant_store_8<T>(w_vec.x, scale, LUT0, LUT1, LUT2, LUT3, dst);
        dequant_store_8<T>(w_vec.y, scale, LUT0, LUT1, LUT2, LUT3, dst + 8);
        dequant_store_8<T>(w_vec.z, scale, LUT0, LUT1, LUT2, LUT3, dst + 16);
        dequant_store_8<T>(w_vec.w, scale, LUT0, LUT1, LUT2, LUT3, dst + 24);
      } else {
        T *dst = &B_sh[ln * K_BLK];
#pragma unroll
        for (int k = 0; k < K_BLK; k++)
          dst[k] = T(0);
      }
    }

    __syncthreads();

#pragma unroll
    for (int k_step = 0; k_step < WMMA_K_STEPS; k_step++) {
      fragment<matrix_a, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, row_major>
          a_frag;
      const T *A_ptr =
          A_sh + warp_m_idx * WMMA_M_DIM * K_BLK + k_step * WMMA_K_DIM;
      load_matrix_sync(a_frag, A_ptr, K_BLK);

#pragma unroll
      for (int n_sub = 0; n_sub < 2; n_sub++) {
        fragment<matrix_b, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, col_major>
            b_frag;
        const T *B_ptr = B_sh + (warp_n_idx * 2 + n_sub) * WMMA_N_DIM * K_BLK +
                         k_step * WMMA_K_DIM;
        load_matrix_sync(b_frag, B_ptr, K_BLK);
        mma_sync(c_frag[n_sub], a_frag, b_frag, c_frag[n_sub]);
      }
    }

    __syncthreads();
  }

  for (int n_sub = 0; n_sub < 2; n_sub++) {
    float *C_ptr = C_sh + warp_m_idx * WMMA_M_DIM * N_BLK +
                   (warp_n_idx * 2 + n_sub) * WMMA_N_DIM;
    store_matrix_sync(C_ptr, c_frag[n_sub], N_BLK, mem_row_major);
  }
  __syncthreads();

  constexpr int C_ELEMS = M_BLK * N_BLK;
  for (int i = threadId; i < C_ELEMS; i += BLOCK_THREADS) {
    const int lm = i / N_BLK;
    const int ln = i % N_BLK;
    const int gm = m_base + lm;
    const int gn = n_base + ln;

    if (gm < M && gn < N) {
      float val = C_sh[lm * N_BLK + ln];
      if (has_bias && bias != nullptr) {
        if constexpr (std::is_same_v<T, half>) {
          val += __half2float(__ldg(&bias[gn]));
        } else {
          val += __bfloat162float(__ldg(&bias[gn]));
        }
      }
      if constexpr (std::is_same_v<T, half>) {
        output[(size_t)gm * N + gn] = __float2half(val);
      } else {
        output[(size_t)gm * N + gn] = __float2bfloat16(val);
      }
    }
  }
}

// Phase 1: Build per-expert token lists in global memory.
// Grid: (1, num_experts), Block: 256 threads.
// g_token_lists: [num_experts, max_tokens_per_expert] — pre-allocated
// g_expert_counts: [num_experts] — pre-zero'd
__global__ void mxfp4_moe_scatter_tokens(
    const uint32_t *__restrict__ indices,
    int *__restrict__ g_token_lists,
    int *__restrict__ g_expert_counts,
    int total_work, int num_experts, int list_stride) {
  const int expert_id = blockIdx.y;
  int *my_list = g_token_lists + (size_t)expert_id * list_stride;
  int *my_count = g_expert_counts + expert_id;

  for (int i = threadIdx.x; i < total_work; i += blockDim.x) {
    if (__ldg(&indices[i]) == (uint32_t)expert_id) {
      int pos = atomicAdd(my_count, 1);
      if (pos < list_stride)
        my_list[pos] = i;
    }
  }
}

// Phase 2: WMMA GEMM kernel reading per-expert token lists from global memory.
// Shared memory is now fixed-size (WMMA tiles only, ~24 KB).
template <typename T>
__launch_bounds__(BLOCK_THREADS) __global__
    void mxfp4_moe_grouped_gemm_wmma_kernel(
        const T *__restrict__ input, const uint8_t *__restrict__ weights,
        const uint8_t *__restrict__ weight_scales, const T *__restrict__ biases,
        const uint32_t *__restrict__ indices, T *__restrict__ output,
        int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
        bool input_has_topk_dim,
        const int *__restrict__ g_token_lists,
        const int *__restrict__ g_expert_counts,
        int list_stride) {
  const uint32_t LUT0 = 0x03020100;
  const uint32_t LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00;
  const uint32_t LUT3 = 0xF4F8FAFC;

  const int scale_stride = CEILDIV(K, MXFP4_BLOCK_SIZE);
  const int expert_id = blockIdx.y;
  const int n_base = blockIdx.x * N_BLK;
  const int threadId = threadIdx.x;

  const uint8_t *expert_weight = weights + (size_t)expert_id * N * (K / 2);
  const uint8_t *expert_scale =
      weight_scales + (size_t)expert_id * N * scale_stride;

  const int *token_list = g_token_lists + (size_t)expert_id * list_stride;
  const int M_expert = g_expert_counts[expert_id];
  if (M_expert == 0)
    return;

  extern __shared__ uint8_t smem_bytes[];
  T *A_sh = reinterpret_cast<T *>(smem_bytes);
  T *B_sh = A_sh + M_BLK * K_BLK;
  uint8_t *C_raw = reinterpret_cast<uint8_t *>(B_sh + N_BLK * K_BLK);
  size_t align_off = reinterpret_cast<uintptr_t>(C_raw) % alignof(float);
  if (align_off != 0)
    C_raw += (alignof(float) - align_off);
  float *C_sh = reinterpret_cast<float *>(C_raw);

  const int warpId = threadId / 32;
  const int warp_m_idx = warpId / WARPS_N;
  const int warp_n_idx = warpId % WARPS_N;

  VecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  for (int m_tile = 0; m_tile < CEILDIV(M_expert, M_BLK); m_tile++) {
    const int m_base = m_tile * M_BLK;

    fragment<accumulator, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, float> c_frag[2];
    fill_fragment(c_frag[0], 0.0f);
    fill_fragment(c_frag[1], 0.0f);

    for (int k_base = 0; k_base < K; k_base += K_BLK) {
      constexpr int A_VEC_ELEMS = M_BLK * K_BLK / VEC_SIZE;
      for (int i = threadId; i < A_VEC_ELEMS; i += BLOCK_THREADS) {
        const int idx = i * VEC_SIZE;
        const int lm = idx / K_BLK;
        const int lk = idx % K_BLK;
        const int work_pos = m_base + lm;
        const int gk = k_base + lk;

        if (work_pos < M_expert && gk < K) {
          const int work_idx = token_list[work_pos];
          const int input_row =
              input_has_topk_dim ? work_idx : (work_idx / topk);
          *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) =
              *reinterpret_cast<const VecT *>(
                  &input[(size_t)input_row * K + gk]);
        } else {
          *reinterpret_cast<VecT *>(&A_sh[lm * K_BLK + lk]) = zero_vec;
        }
      }

      for (int ln = threadId; ln < N_BLK; ln += BLOCK_THREADS) {
        const int gn = n_base + ln;
        if (gn < N) {
          uint4 w_vec = *reinterpret_cast<const uint4 *>(
              &expert_weight[(size_t)gn * (K / 2) + k_base / 2]);
          float scale =
              e8m0_to_float(__ldg(&expert_scale[(size_t)gn * scale_stride +
                                                k_base / MXFP4_BLOCK_SIZE])) *
              0.5f;
          T *dst = &B_sh[ln * K_BLK];
          dequant_store_8<T>(w_vec.x, scale, LUT0, LUT1, LUT2, LUT3, dst);
          dequant_store_8<T>(w_vec.y, scale, LUT0, LUT1, LUT2, LUT3, dst + 8);
          dequant_store_8<T>(w_vec.z, scale, LUT0, LUT1, LUT2, LUT3, dst + 16);
          dequant_store_8<T>(w_vec.w, scale, LUT0, LUT1, LUT2, LUT3, dst + 24);
        } else {
          T *dst = &B_sh[ln * K_BLK];
#pragma unroll
          for (int k = 0; k < K_BLK; k++)
            dst[k] = T(0);
        }
      }

      __syncthreads();

#pragma unroll
      for (int k_step = 0; k_step < WMMA_K_STEPS; k_step++) {
        fragment<matrix_a, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, row_major>
            a_frag;
        const T *A_ptr =
            A_sh + warp_m_idx * WMMA_M_DIM * K_BLK + k_step * WMMA_K_DIM;
        load_matrix_sync(a_frag, A_ptr, K_BLK);

#pragma unroll
        for (int n_sub = 0; n_sub < 2; n_sub++) {
          fragment<matrix_b, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, T, col_major>
              b_frag;
          const T *B_ptr = B_sh +
                           (warp_n_idx * 2 + n_sub) * WMMA_N_DIM * K_BLK +
                           k_step * WMMA_K_DIM;
          load_matrix_sync(b_frag, B_ptr, K_BLK);
          mma_sync(c_frag[n_sub], a_frag, b_frag, c_frag[n_sub]);
        }
      }

      __syncthreads();
    }

    for (int n_sub = 0; n_sub < 2; n_sub++) {
      float *C_ptr = C_sh + warp_m_idx * WMMA_M_DIM * N_BLK +
                     (warp_n_idx * 2 + n_sub) * WMMA_N_DIM;
      store_matrix_sync(C_ptr, c_frag[n_sub], N_BLK, mem_row_major);
    }
    __syncthreads();

    constexpr int C_ELEMS = M_BLK * N_BLK;
    for (int i = threadId; i < C_ELEMS; i += BLOCK_THREADS) {
      const int lm = i / N_BLK;
      const int ln = i % N_BLK;
      const int work_pos = m_base + lm;
      const int gn = n_base + ln;

      if (work_pos < M_expert && gn < N) {
        const int work_idx = token_list[work_pos];
        float val = C_sh[lm * N_BLK + ln];
        if (has_bias && biases != nullptr) {
          if constexpr (std::is_same_v<T, half>) {
            val += __half2float(__ldg(&biases[(size_t)expert_id * N + gn]));
          } else {
            val += __bfloat162float(__ldg(&biases[(size_t)expert_id * N + gn]));
          }
        }
        if constexpr (std::is_same_v<T, half>) {
          output[(size_t)work_idx * N + gn] = __float2half(val);
        } else {
          output[(size_t)work_idx * N + gn] = __float2bfloat16(val);
        }
      }
    }

    __syncthreads();
  }
}

} // namespace mxfp4_wmma

static size_t wmma_smem_bytes() {
  using namespace mxfp4_wmma;
  size_t AB = (M_BLK * K_BLK + N_BLK * K_BLK) * 2;
  size_t pad = (16 - (AB % 16)) % 16;
  size_t C = M_BLK * N_BLK * sizeof(float);
  return AB + pad + C;
}

extern "C" void mxfp4_matmul_wmma_f16(const __half *input,
                                       const uint8_t *weight,
                                       const uint8_t *weight_scale,
                                       const __half *bias, __half *output,
                                       int M, int N, int K, bool has_bias,
                                       cudaStream_t stream) {
  using namespace mxfp4_wmma;

  dim3 grid(CEILDIV(N, N_BLK), CEILDIV(M, M_BLK));
  dim3 block(BLOCK_THREADS);
  size_t smem = wmma_smem_bytes();

  auto kernel = mxfp4_wmma::mxfp4_matmul_wmma_kernel<half>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kernel<<<grid, block, smem, stream>>>(
      input, weight, weight_scale, bias, output, M, N, K, has_bias);
}

#ifndef NO_BF16_KERNEL
extern "C" void mxfp4_matmul_wmma_bf16(const __nv_bfloat16 *input,
                                        const uint8_t *weight,
                                        const uint8_t *weight_scale,
                                        const __nv_bfloat16 *bias,
                                        __nv_bfloat16 *output, int M,
                                        int N, int K, bool has_bias,
                                        cudaStream_t stream) {
  using namespace mxfp4_wmma;

  dim3 grid(CEILDIV(N, N_BLK), CEILDIV(M, M_BLK));
  dim3 block(BLOCK_THREADS);
  size_t smem = wmma_smem_bytes();

  auto kernel = mxfp4_wmma::mxfp4_matmul_wmma_kernel<__nv_bfloat16>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kernel<<<grid, block, smem, stream>>>(input, weight, weight_scale, bias, output,
                                        M, N, K, has_bias);
}
#else
extern "C" void mxfp4_matmul_wmma_bf16(const void *, const uint8_t *,
                                         const uint8_t *, const void *,
                                         void *, int, int, int, bool,
                                         cudaStream_t) {}
#endif

// Helper: allocate scatter buffers, launch scatter kernel, then GEMM kernel.
template <typename T, typename KernelT>
static void launch_moe_wmma(
    KernelT kernel_fn,
    const T *input, const uint8_t *weights, const uint8_t *weight_scales,
    const T *biases, const uint32_t *indices, T *output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim, cudaStream_t stream) {
  using namespace mxfp4_wmma;

  int total_work = num_tokens * topk;
  int list_stride = total_work;

  int *g_token_lists = nullptr;
  int *g_expert_counts = nullptr;
  size_t lists_bytes = (size_t)num_experts * list_stride * sizeof(int);
  size_t counts_bytes = (size_t)num_experts * sizeof(int);
  cudaMallocAsync(&g_token_lists, lists_bytes, stream);
  cudaMallocAsync(&g_expert_counts, counts_bytes, stream);
  cudaMemsetAsync(g_expert_counts, 0, counts_bytes, stream);

  {
    dim3 scatter_grid(1, num_experts);
    dim3 scatter_block(256);
    mxfp4_wmma::mxfp4_moe_scatter_tokens<<<scatter_grid, scatter_block, 0, stream>>>(
        indices, g_token_lists, g_expert_counts,
        total_work, num_experts, list_stride);
  }

  {
    dim3 grid(CEILDIV(N, N_BLK), num_experts);
    dim3 block(BLOCK_THREADS);
    size_t smem = wmma_smem_bytes();

    cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    kernel_fn<<<grid, block, smem, stream>>>(
        input, weights, weight_scales, biases, indices, output,
        num_tokens, topk, num_experts, N, K, has_bias, input_has_topk_dim,
        g_token_lists, g_expert_counts, list_stride);
  }

  cudaFreeAsync(g_token_lists, stream);
  cudaFreeAsync(g_expert_counts, stream);
}

extern "C" void mxfp4_moe_grouped_gemm_wmma_f16(
    const __half *input, const uint8_t *weights, const uint8_t *weight_scales,
    const __half *biases, const uint32_t *indices, __half *output,
    int num_tokens, int topk, int num_experts, int N, int K, bool has_bias,
    bool input_has_topk_dim, cudaStream_t stream) {
  launch_moe_wmma<half>(
      mxfp4_wmma::mxfp4_moe_grouped_gemm_wmma_kernel<half>,
      input, weights, weight_scales, biases, indices, output,
      num_tokens, topk, num_experts, N, K, has_bias, input_has_topk_dim, stream);
}

#ifndef NO_BF16_KERNEL
extern "C" void mxfp4_moe_grouped_gemm_wmma_bf16(
    const __nv_bfloat16 *input, const uint8_t *weights,
    const uint8_t *weight_scales, const __nv_bfloat16 *biases,
    const uint32_t *indices, __nv_bfloat16 *output, int num_tokens, int topk,
    int num_experts, int N, int K, bool has_bias, bool input_has_topk_dim,
    cudaStream_t stream) {
  launch_moe_wmma<__nv_bfloat16>(
      mxfp4_wmma::mxfp4_moe_grouped_gemm_wmma_kernel<__nv_bfloat16>,
      input, weights, weight_scales, biases, indices, output,
      num_tokens, topk, num_experts, N, K, has_bias, input_has_topk_dim, stream);
}
#else
extern "C" void mxfp4_moe_grouped_gemm_wmma_bf16(
    const void *, const uint8_t *, const uint8_t *, const void *,
    const uint32_t *, void *, int, int, int, int, int, bool, bool,
    cudaStream_t) {}
#endif
