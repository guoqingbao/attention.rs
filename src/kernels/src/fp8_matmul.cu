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
using namespace nvcuda::wmma;

namespace vllm {
    __forceinline__ __device__ void from_float(half& out, float in) {
        out = __float2half(in);
    }
} // namespace vllm

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

__device__ __forceinline__ float get_scale(const float *__restrict__ scale,
                                           int n, int k, int scale_stride,
                                           int block_size_y, int block_size_x) {
  int sr = n / block_size_y;
  int sc = k / block_size_x;
  return __ldg(&scale[sr * scale_stride + sc]);
}

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void fp8_matmul_kernel(const T *__restrict__ input,
                                 const uint8_t *__restrict__ weight,
                                 const float *__restrict__ weight_scale,
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

    for (int i = tid; i < BLOCK_N * BLOCK_K; i += num_threads) {
      int ln = i / BLOCK_K;
      int lk = i % BLOCK_K;
      int gn = bx * BLOCK_N + ln;
      int gk = k_tile + lk;

      float val = 0.0f;
      if (gn < N && gk < K) {
        uint8_t w_raw = __ldg(&weight[gn * K + gk]);
        float s = 0.0f;
        if (tile_scale_uniform) {
          int scale_row = gn / block_size_y;
          s = __ldg(&weight_scale[scale_row * scale_row_stride + scale_k_idx_tile]);
        } else {
          s = get_scale(weight_scale, gn, gk, scale_row_stride, block_size_y,
                        block_size_x);
        }
        val = vllm::fp8::dispatch_fp8_to_float(w_raw) * s;
      }
      s_weight[ln][lk] = val;
    }

    __syncthreads();

    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < BLOCK_K; k += 2) {
        float2 in2 = *reinterpret_cast<float2*>(&s_input[ty][k]);
        float2 w2 = *reinterpret_cast<float2*>(&s_weight[tx][k]);
        acc = fmaf(in2.x, w2.x, acc);
        acc = fmaf(in2.y, w2.y, acc);
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    vllm::from_float(output[row * N + col], acc);
  }
}

#define BK 32 

template <typename T, int BM, int BN, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void fp8_wmma_matmul(
    const T *__restrict__ input,
    const uint8_t *__restrict__ weight,
    const float *__restrict__ weight_scale,
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
        // Store into s_b[BN][BK] directly (no transpose).
        // WMMA will load as col_major.
        
        constexpr int TILE_ELEMS = BN * BK;
        
        #pragma unroll
        for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
             int n_rel = i / BK;
             int k_rel = i % BK;
             
             int gn = bx * BN + n_rel;
             int gk = k_step + k_rel;
             
             float val = 0.0f;
             if (gn < N && gk < K) {
                 uint8_t w = weight[gn * K + gk];
                 
                 // Scale
                 int sr = gn / block_size_y;
                 int sc = gk / block_size_x;
                 float s = weight_scale[sr * scale_row_stride + sc];
                 
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

extern "C" void fp8_matmul_f16(const __half *input, const uint8_t *weight,
                        const float *weight_scale, __half *output, int M,
                        int N, int K, int scale_row_stride, int block_size_y,
                        int block_size_x, cudaStream_t stream) {

  
  if (M <= 32) {
      constexpr int TILE = 32; 
      constexpr int TILE_K = 32; 
      
      dim3 block(TILE, TILE);
      dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));
      
      fp8_matmul_kernel<__half, TILE, TILE, TILE_K>
       <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
  } else {
      // Default / Prefill
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      
      fp8_wmma_matmul<__half, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
  }
}

extern "C" void fp8_matmul_bf16(const __nv_bfloat16 *input, const uint8_t *weight,
                        const float *weight_scale, __nv_bfloat16 *output, int M,
                        int N, int K, int scale_row_stride, int block_size_y,
                        int block_size_x, cudaStream_t stream) {


#ifndef NO_BF16_KERNEL
  if (M <= 32) {
      constexpr int TILE = 32; 
      constexpr int TILE_K = 32; 
      dim3 block(TILE, TILE);
      dim3 grid(CEILDIV(N, TILE), CEILDIV(M, TILE));
      
      fp8_matmul_kernel<__nv_bfloat16, TILE, TILE, TILE_K>
       <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
  } else {
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 64;
      dim3 block(128, 1);
      dim3 grid(CEILDIV(N, BLOCK_N), CEILDIV(M, BLOCK_M));
      fp8_wmma_matmul<__nv_bfloat16, BLOCK_M, BLOCK_N, 16, 16, 16>
       <<<grid, block, 0, stream>>>(input, weight, weight_scale, output, M, N, K,
                                    scale_row_stride, block_size_y, block_size_x);
  }
#endif
}