#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <cassert>
#include <cstring>
#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"
#include "attention/dtype_fp8.cuh"
#include "attention/dtype_e8m0.cuh"
using namespace nvcuda::wmma;

namespace vllm {
    __forceinline__ __device__ void from_float(half& out, float in) {
        out = __float2half(in);
    }
} // namespace vllm

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

template <typename ScaleT>
__device__ __forceinline__ float get_scale(const ScaleT *__restrict__ scale,
                                           int n, int k, int scale_stride,
                                           int block_size_y, int block_size_x) {
  int sr = n / block_size_y;
  int sc = k / block_size_x;
  return attention_rs::e8m0::read_scale(scale, sr * scale_stride + sc);
}

template <typename T, typename ScaleT, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_kernel(const T *__restrict__ input,
                                 const uint8_t *__restrict__ weight,
                                 const ScaleT *__restrict__ weight_scale,
                                 T *__restrict__ output, int M, int N, int K,
                                 int scale_row_stride, int block_size_y,
                                 int block_size_x) {
  __shared__ float s_input[BLOCK_M][BLOCK_K + 4];
  __shared__ float s_weight[BLOCK_N][BLOCK_K + 4];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = by * BLOCK_M + ty;
  const int col = bx * BLOCK_N + tx;

  float acc = 0.0f;

  const int num_threads = BLOCK_M * BLOCK_N;
  const int tid = ty * BLOCK_N + tx;

  for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    const bool tile_scale_uniform =
        block_size_x >= BLOCK_K &&
        ((k_tile % block_size_x) + BLOCK_K <= block_size_x);
    const int scale_k_idx_tile = tile_scale_uniform ? (k_tile / block_size_x) : 0;

    for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
      int lm = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gm = by * BLOCK_M + lm;
      int gk = k_tile + lk;

      float val = 0.0f;
      if (gm < M && gk < K) {
        if constexpr (std::is_same_v<T, half>) {
          val = __half2float(__ldg(&input[gm * K + gk]));
        } else {
#ifndef NO_BF16_KERNEL
          val = __bfloat162float(__ldg(&input[gm * K + gk]));
#endif
        }
      }
      s_input[lm][lk] = val;
    }

    // Vectorized FP8 weight loading: process 4 bytes per iteration
    constexpr int WEIGHT_ELEMS = BLOCK_N * BLOCK_K;
    constexpr int VEC4_ITERS = WEIGHT_ELEMS / 4;
    for (int i = tid; i < VEC4_ITERS; i += num_threads) {
      int flat = i * 4;
      int ln = flat / BLOCK_K;
      int lk = flat % BLOCK_K;
      int gn = bx * BLOCK_N + ln;
      int gk_base = k_tile + lk;

      if (gn < N && gk_base + 3 < K) {
        uint32_t w4 = __ldg(reinterpret_cast<const uint32_t*>(&weight[gn * K + gk_base]));
        float s;
        if (tile_scale_uniform) {
          int scale_row = gn / block_size_y;
          s = attention_rs::e8m0::read_scale(
              weight_scale, scale_row * scale_row_stride + scale_k_idx_tile);
        } else {
          s = get_scale(weight_scale, gn, gk_base, scale_row_stride, block_size_y,
                        block_size_x);
        }
        s_weight[ln][lk]     = vllm::fp8::dispatch_fp8_to_float((uint8_t)(w4 & 0xFF)) * s;
        s_weight[ln][lk + 1] = vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 8) & 0xFF)) * s;
        s_weight[ln][lk + 2] = vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 16) & 0xFF)) * s;
        s_weight[ln][lk + 3] = vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 24) & 0xFF)) * s;
      } else {
        for (int d = 0; d < 4; d++) {
          int cur_lk = lk + d;
          int cur_gk = gk_base + d;
          float val = 0.0f;
          if (gn < N && cur_gk < K) {
            uint8_t w_raw = __ldg(&weight[gn * K + cur_gk]);
            float s = 0.0f;
            if (tile_scale_uniform) {
              int scale_row = gn / block_size_y;
              s = attention_rs::e8m0::read_scale(
                  weight_scale, scale_row * scale_row_stride + scale_k_idx_tile);
            } else {
              s = get_scale(weight_scale, gn, cur_gk, scale_row_stride, block_size_y,
                            block_size_x);
            }
            val = vllm::fp8::dispatch_fp8_to_float(w_raw) * s;
          }
          s_weight[ln][cur_lk] = val;
        }
      }
    }
    // Handle remaining elements (when WEIGHT_ELEMS not divisible by 4)
    for (int i = VEC4_ITERS * 4 + tid; i < WEIGHT_ELEMS; i += num_threads) {
      int ln = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gn = bx * BLOCK_N + ln;
      int gk = k_tile + lk;
      float val = 0.0f;
      if (gn < N && gk < K) {
        uint8_t w_raw = __ldg(&weight[gn * K + gk]);
        float s = tile_scale_uniform
            ? attention_rs::e8m0::read_scale(
                  weight_scale, (gn / block_size_y) * scale_row_stride + scale_k_idx_tile)
            : get_scale(weight_scale, gn, gk, scale_row_stride, block_size_y, block_size_x);
        val = vllm::fp8::dispatch_fp8_to_float(w_raw) * s;
      }
      s_weight[ln][lk] = val;
    }

    __syncthreads();

    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < BLOCK_K; k += 4) {
        float4 in4 = *reinterpret_cast<float4*>(&s_input[ty][k]);
        float4 w4 = *reinterpret_cast<float4*>(&s_weight[tx][k]);
        acc = fmaf(in4.x, w4.x, acc);
        acc = fmaf(in4.y, w4.y, acc);
        acc = fmaf(in4.z, w4.z, acc);
        acc = fmaf(in4.w, w4.w, acc);
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    vllm::from_float(output[row * N + col], acc);
  }
}

#define BK 32 

template <typename T, typename ScaleT, int BM, int BN, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void fp8_wmma_matmul(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weight,
    const ScaleT *__restrict__ weight_scale,
    T *__restrict__ output,
    int M, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x) 
{
    // Warps layout:
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Specialized warp mapping
    int warp_row_px, warp_col_px;
    
    // Accumulators
    // Fragments count depends on (BM, BN) vs (WMMA_M, WMMA_N)
    constexpr int FRAGS_M = BM / WMMA_M;
    constexpr int FRAGS_N_PER_WARP = (BN / 4) / WMMA_N; // Assuming 1x4 Grid?
    
    // Let's make the grid explicit per config
    if constexpr (BM == 64 && BN == 64) {
        // Grid 2x2. Warp covers 32x32.
        // WMMA 16x16x16.
        // Frags: 32/16 x 32/16 = 2x2.
        warp_row_px = (warp_id / 2) * 32;
        warp_col_px = (warp_id % 2) * 32;
    } else if constexpr (BM == 16 && BN == 64) {
        // Grid 1x4. Warp covers 16x16. WMMA 16x16. Frags 1x1.
        warp_row_px = 0;             
        warp_col_px = warp_id * 16;  
    } else if constexpr (BM == 16 && BN == 128) {
        // Grid 1x4. Warp covers 16x32. WMMA 16x16. Frags 1x2.
        warp_row_px = 0;
        warp_col_px = warp_id * 32;
    } else if constexpr (BM == 8 && BN == 128) {
        // Grid 1x4. Warp covers 8x32. 
        // using WMMA 8x32. Frags 1x1.
        warp_row_px = 0;
        warp_col_px = warp_id * 32;
    } else {
        warp_row_px = 0; 
        warp_col_px = 0;
    }

    // Number of fragments per thread
    constexpr int NUM_FRAGS_M = (BM == 64) ? 2 : 1;
    constexpr int NUM_FRAGS_N = (BM == 64) ? 2 : (BN == 128 ? 2 : 1);
    // Special case for BM=8, BN=128 with WMMA 8x32 => 1 fragment N
    constexpr int ACTUAL_FRAGS_N = (BM == 8 && BN == 128) ? 1 : NUM_FRAGS_N;
    
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[NUM_FRAGS_M][ACTUAL_FRAGS_N];

    #pragma unroll
    for(int i=0; i<NUM_FRAGS_M; ++i)
        #pragma unroll
        for(int j=0; j<ACTUAL_FRAGS_N; ++j)
            fill_fragment(acc[i][j], 0.0f);

    // Shared Memory
    __shared__ T s_a[BM][BK]; 
    // B is now loaded as row-major chunks, essentially directly copying tiles.
    // Padding to avoid bank conflicts.
    // padding for bank conflicts (must maintain 128-bit/16-byte alignment of rows)
    // BK=32 (aligned). Padding should be 8.
    __shared__ T s_b[BN][BK + 8]; 
    // Output shared memory
    __shared__ float s_out[BM][BN + 8]; 

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tid = threadIdx.x;

    // Loop over K in chunks of BK
    for (int k_step = 0; k_step < K; k_step += BK) {
        // Cooperative Load A -> Smem ---
        #pragma unroll
        for(int i=0; i < (BM * BK) / 128; ++i) {
             int current_idx = i * 128 + tid;
             if (current_idx < BM * BK) {
                 int r = current_idx / BK;
                 int c = current_idx % BK;
                 int gr = by * BM + r;
                 int gc = k_step + c;
                 if(gr < M && gc < K) {
                    s_a[r][c] = input[gr * K + gc];
                 } else {
                    s_a[r][c] = T(0.0);
                 }
             }
        }

        // Weight is N x K. Loading tile [BN, BK].
        // Vectorized: load 4 FP8 bytes at once, dequant and store as half/bf16.
        
        constexpr int TILE_ELEMS = BN * BK;
        constexpr int VEC4_TILE = TILE_ELEMS / 4;
        
        #pragma unroll
        for (int i = tid; i < VEC4_TILE; i += blockDim.x) {
             int flat = i * 4;
             int n_rel = flat / BK;
             int k_rel = flat % BK;
             
             int gn = bx * BN + n_rel;
             int gk_base = k_step + k_rel;
             
             if (gn < N && gk_base + 3 < K) {
                 uint32_t w4 = __ldg(reinterpret_cast<const uint32_t*>(&weight[gn * K + gk_base]));
                 
                 int sr = gn / block_size_y;
                 int sc = gk_base / block_size_x;
                 float s = attention_rs::e8m0::read_scale(
                     weight_scale, sr * scale_row_stride + sc);
                 
                 vllm::from_float(s_b[n_rel][k_rel],     vllm::fp8::dispatch_fp8_to_float((uint8_t)(w4 & 0xFF)) * s);
                 vllm::from_float(s_b[n_rel][k_rel + 1], vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 8) & 0xFF)) * s);
                 vllm::from_float(s_b[n_rel][k_rel + 2], vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 16) & 0xFF)) * s);
                 vllm::from_float(s_b[n_rel][k_rel + 3], vllm::fp8::dispatch_fp8_to_float((uint8_t)((w4 >> 24) & 0xFF)) * s);
             } else {
                 for (int d = 0; d < 4; d++) {
                     int cur_k = k_rel + d;
                     int gk = gk_base + d;
                     float val = 0.0f;
                     if (gn < N && gk < K) {
                         uint8_t w = weight[gn * K + gk];
                         int sr2 = gn / block_size_y;
                         int sc2 = gk / block_size_x;
                         float s2 = attention_rs::e8m0::read_scale(
                             weight_scale, sr2 * scale_row_stride + sc2);
                         val = vllm::fp8::dispatch_fp8_to_float(w) * s2;
                     }
                     vllm::from_float(s_b[n_rel][cur_k], val);
                 }
             }
        }
        // Handle remaining elements
        for (int i = VEC4_TILE * 4 + tid; i < TILE_ELEMS; i += blockDim.x) {
             int n_rel = i / BK;
             int k_rel = i % BK;
             int gn = bx * BN + n_rel;
             int gk = k_step + k_rel;
             float val = 0.0f;
             if (gn < N && gk < K) {
                 uint8_t w = weight[gn * K + gk];
                 int sr = gn / block_size_y;
                 int sc = gk / block_size_x;
                 float s = attention_rs::e8m0::read_scale(
                     weight_scale, sr * scale_row_stride + sc);
                 val = vllm::fp8::dispatch_fp8_to_float(w) * s;
             }
             vllm::from_float(s_b[n_rel][k_rel], val);
        }
        
        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag; 

        // Iterate over the loaded K tile (BK=32) in steps of WMMA_K
        #pragma unroll
        for (int k_sub = 0; k_sub < BK; k_sub += WMMA_K) {
             #pragma unroll
            for (int i = 0; i < NUM_FRAGS_M; ++i) {     // Rows of sub-tiles (M)
                #pragma unroll
                for (int j = 0; j < ACTUAL_FRAGS_N; ++j) { // Cols of sub-tiles (N)
                    
                    load_matrix_sync(a_frag, &s_a[warp_row_px + i*WMMA_M][k_sub], BK);

                    // B fragment: Load from s_b[warp_col + j*WMMA_N][k_sub]
                    // s_b is padded [BN][BK+8].
                    load_matrix_sync(b_frag, &s_b[warp_col_px + j*WMMA_N][k_sub], BK + 8);

                    mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NUM_FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < ACTUAL_FRAGS_N; ++j) {
            float *ptr = &s_out[warp_row_px + i * WMMA_M][warp_col_px + j * WMMA_N];
            store_matrix_sync(ptr, acc[i][j], BN + 8, mem_row_major);
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int r = i / BN;
        int c = i % BN;
            
        int global_r = by * BM + r;
        int global_c = bx * BN + c;
            
        if (global_r < M && global_c < N) {
             float val = s_out[r][c];
             vllm::from_float(output[global_r * N + global_c], val);
        }
    }
}

// Channel-wise compressed-tensors FP8 uses one F32 scale for every output
// row: scale[n] applies to all K values of weight[n, :].  The WMMA path above
// first converts each dequantized weight to T in shared memory.  That is a
// precision loss for this format (especially when T is BF16), and it is
// needlessly repeated for every K tile.  Keep this path on the FP32
// dequantizing kernel, whose accumulator already stays in FP32, and only
// round once when writing the requested output dtype.
template <typename T>
static void fp8_matmul_channelwise_impl(
    const T *input, const uint8_t *weight, const float *weight_scale,
    T *output, int M, int N, int K, cudaStream_t stream) {
  constexpr int BM = 8;
  constexpr int BN = 64;
  constexpr int BK_S = 32;
  dim3 block(BN, BM);
  dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
  fp8_matmul_kernel<T, float, BM, BN, BK_S>
      <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                   1, 1, K);
}

extern "C" void fp8_matmul_f16_channelwise(
    const __half *input, const uint8_t *weight, const float *weight_scale,
    __half *output, int M, int N, int K, cudaStream_t stream) {
  fp8_matmul_channelwise_impl(input, weight, weight_scale, output, M, N, K, stream);
}

extern "C" void fp8_matmul_f16(const __half *input, const uint8_t *weight,
                        const void *weight_scale, __half *output, int M,
                        int N, int K, int scale_row_stride, int block_size_y,
                        int block_size_x, int scale_dtype, cudaStream_t stream) {

  if (scale_dtype == 1) { // e8m0
    const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__half, uint8_t, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__half, uint8_t, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  } else { // f32
    const float* ws = reinterpret_cast<const float*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__half, float, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__half, float, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  }
}

extern "C" void fp8_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                        const void *weight_scale, __nv_bfloat16 *output, int M,
                        int N, int K, int scale_row_stride, int block_size_y,
                        int block_size_x, int scale_dtype, cudaStream_t stream) {

#ifndef NO_BF16_KERNEL
  if (scale_dtype == 1) { // e8m0
    const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__nv_bfloat16, uint8_t, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__nv_bfloat16, uint8_t, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  } else { // f32
    const float* ws = reinterpret_cast<const float*>(weight_scale);
    if (M <= 32) {
      constexpr int BM = 8;
      constexpr int BN = 64;
      constexpr int BK_S = 32;
      dim3 block(BN, BM);
      dim3 grid(CEILDIV(N, BN), CEILDIV(M, BM));
      fp8_matmul_kernel<__nv_bfloat16, float, BM, BN, BK_S>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__nv_bfloat16, float, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, ws, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
    }
  }
#endif
}

extern "C" void fp8_matmul_bf16_channelwise(
    const __nv_bfloat16 *input, const uint8_t *weight, const float *weight_scale,
    __nv_bfloat16 *output, int M, int N, int K, cudaStream_t stream) {
#ifndef NO_BF16_KERNEL
  fp8_matmul_channelwise_impl(input, weight, weight_scale, output, M, N, K, stream);
#endif
}
