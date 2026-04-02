/**
 * @brief CUDA kernel for GPU-accelerated top-k/top-p sampling with softmax.
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 *
 * This kernel performs efficient sampling from logits by:
 *  - Stage A: Tiled local top-k extraction per vocabulary chunk
 *  - Stage B: Global top-k merge, softmax, top-p filtering, and sampling
 * Uses Philox RNG for deterministic, reproducible sampling across runs.
 * Supports float32, float16, and bfloat16 input logits.
 *
 * This CUDA kernel is developed for vLLM.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/gpu_sampling.cu
 *
 * @details
 * - Two-stage algorithm: local top-k per tile, then global merge and sample.
 * - Supports top-k values of 32, 64, 128, or 256 via template instantiation.
 * - Uses CUB block radix sort for efficient in-block sorting.
 * - Temperature scaling and top-p (nucleus) sampling within top-k.
 * - Philox4x32-10 PRNG for reproducible sampling with (seed, token_pos) keys.
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

#include "gpu_sampling.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Helper to convert from input type to float
template<typename T>
__device__ __forceinline__ float to_float(T x);

template<>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template<>
__device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

static __device__ __forceinline__ uint4 philox4x32_10(uint2 key, uint4 ctr) {
  const uint32_t M0 = 0xD2511F53u;
  const uint32_t M1 = 0xCD9E8D57u;
  const uint32_t W0 = 0x9E3779B9u;
  const uint32_t W1 = 0xBB67AE85u;

  uint32_t k0 = key.x, k1 = key.y;
  #pragma unroll
  for (int r = 0; r < 10; ++r) {
    uint64_t p0 = (uint64_t)M0 * (uint64_t)ctr.x;
    uint64_t p1 = (uint64_t)M1 * (uint64_t)ctr.z;

    uint32_t hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo0 = (uint32_t)(p0 & 0xffffffffu);
    uint32_t hi1 = (uint32_t)(p1 >> 32);
    uint32_t lo1 = (uint32_t)(p1 & 0xffffffffu);

    uint32_t nx = hi1 ^ ctr.y ^ k0;
    uint32_t ny = lo1;
    uint32_t nz = hi0 ^ ctr.w ^ k1;
    uint32_t nw = lo0;

    ctr = make_uint4(nx, ny, nz, nw);
    k0 += W0; k1 += W1;
  }
  return ctr;
}

static __device__ __forceinline__ float u01_from_u32(uint32_t x) {
  return ((float)x + 0.5f) * (1.0f / 4294967296.0f);
}

template<int K>
static __device__ __forceinline__ void merge_topk_desc_safe(
    const float* aV, const int* aI, // sorted desc, len K
    const float* bV, const int* bI, // sorted desc, len K
    float* outV, int* outI          // sorted desc, len K
) {
  int ia = 0, ib = 0;
  #pragma unroll
  for (int t = 0; t < K; ++t) {
    float va = (ia < K) ? aV[ia] : -CUDART_INF_F;
    float vb = (ib < K) ? bV[ib] : -CUDART_INF_F;
    bool take_b = (vb > va);
    outV[t] = take_b ? vb : va;
    outI[t] = take_b ? bI[ib] : aI[ia];
    if (take_b) ib++; else ia++;
  }
}

// ---------------- Stage A: tiled local topK per (batch, tile) ----------------
//
// One block handles one tile of the vocab for one batch element.
// It sorts a CHUNK of logits within the block (descending) and writes top-K.
// Template parameter T: input logit type (float, __half, __nv_bfloat16)
template<int K, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T>
__global__ void stageA_local_topk(
    const T* __restrict__ logits, // [B,V]
    int B, int V,
    float temperature,
    int tiles_per_row,                // number of tiles along vocab
    float* __restrict__ out_vals,      // [B, tiles, K]
    int* __restrict__ out_idx          // [B, tiles, K]
) {
  int b = blockIdx.x;
  int tile = blockIdx.y;
  if (b >= B) return;

  constexpr int CHUNK = BLOCK_THREADS * ITEMS_PER_THREAD;
  int base = tile * CHUNK;

  const T* row = logits + (size_t)b * (size_t)V;

  float invT = (temperature > 1e-6f) ? (1.0f / temperature) : 1e6f;

  float v[ITEMS_PER_THREAD];
  int   i[ITEMS_PER_THREAD];

  #pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    int idx = base + threadIdx.x + it * BLOCK_THREADS;
    if (idx < V) {
      v[it] = to_float(row[idx]) * invT;
      i[it] = idx;
    } else {
      v[it] = -CUDART_INF_F;
      i[it] = -1;
    }
  }

  using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockSort::TempStorage temp;
  BlockSort(temp).SortDescending(v, i);
  __syncthreads();

  // Write out the first K ranks from the sorted block.
  // Rank = threadIdx.x + it*BLOCK_THREADS
  int out_base = (b * tiles_per_row + tile) * K;

  #pragma unroll
  for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
    int rank = threadIdx.x + it * BLOCK_THREADS;
    if (rank < K) {
      out_vals[out_base + rank] = v[it];
      out_idx[out_base + rank]  = i[it];
    }
  }
}

// ---------------- Stage B: reduce tiles -> global topK -> softmax -> top-p -> sample ----------------
//
// One block per batch element.
// Merges (tiles_per_row) lists of K candidates into one global K.
// Then samples.
template<int K, int BLOCK_THREADS>
__global__ void stageB_reduce_and_sample(
    int B,
    int tiles_per_row,
    const float* __restrict__ tile_vals, // [B, tiles, K] sorted desc per tile
    const int* __restrict__ tile_idx,    // [B, tiles, K]
    float top_p,
    int top_k,
    uint64_t seed,
    uint64_t token_pos,
    int* __restrict__ out_tokens         // [B]
) {
  int b = blockIdx.x;
  if (b >= B) return;
  int tid = threadIdx.x;

  __shared__ float topv[K];
  __shared__ int   topi[K];
  __shared__ float tmpv[K];
  __shared__ int   tmpi[K];

  // init with -inf
  for (int t = tid; t < K; t += BLOCK_THREADS) {
    topv[t] = -CUDART_INF_F;
    topi[t] = -1;
  }
  __syncthreads();

  // Sequentially merge each tile's topK into running topK.
  // K is small (32/64/128/256), tiles_per_row is modest; this is fast in practice.
  for (int tile = 0; tile < tiles_per_row; ++tile) {
    const int base = (b * tiles_per_row + tile) * K;

    // Load tile candidates into tmp (shared)
    for (int t = tid; t < K; t += BLOCK_THREADS) {
      tmpv[t] = tile_vals[base + t];
      tmpi[t] = tile_idx [base + t];
    }
    __syncthreads();

    if (tid == 0) {
      float outV[K];
      int   outI[K];
      merge_topk_desc_safe<K>(topv, topi, tmpv, tmpi, outV, outI);
      #pragma unroll
      for (int t = 0; t < K; ++t) { topv[t] = outV[t]; topi[t] = outI[t]; }
    }
    __syncthreads();
  }

  int k_eff = top_k > 0 ? (top_k < K ? top_k : K) : K;
  if (k_eff < 1) k_eff = 1;

  // Softmax over topK (stable). Use one thread (K is small) for robustness.
  __shared__ float probs[K];
  if (tid == 0) {
    float mx = topv[0];
    #pragma unroll
    for (int t = 1; t < K; ++t) {
      if (t < k_eff) {
        mx = fmaxf(mx, topv[t]);
      }
    }

    float sum = 0.0f;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
      if (t < k_eff) {
        float e = __expf(topv[t] - mx);
        probs[t] = e;
        sum += e;
      } else {
        probs[t] = 0.0f;
      }
    }
    sum = fmaxf(sum, 1e-20f);
    #pragma unroll
    for (int t = 0; t < K; ++t) probs[t] /= sum;

    // top-p within top-k
    int cutoff = k_eff;
    if (top_p > 0.0f && top_p < 1.0f) {
      double cum = 0.0;
      for (int t = 0; t < K; ++t) {
        cum += (double)probs[t];
        if (cum >= (double)top_p) { cutoff = t + 1; break; }
      }
    }

    float psum = 0.0f;
    for (int t = 0; t < cutoff; ++t) psum += probs[t];
    psum = fmaxf(psum, 1e-20f);

    // Deterministic RNG: per (b, token_pos)
    uint2 key = make_uint2((uint32_t)(seed ^ (0x9E3779B97f4A7C15ULL + (uint64_t)b)),
                           (uint32_t)((seed >> 32) + 0xD1B54A32D192ED03ULL));
    uint4 ctr = make_uint4((uint32_t)token_pos, (uint32_t)(token_pos >> 32),
                           (uint32_t)b, 0x12345678u);
    uint4 r = philox4x32_10(key, ctr);
    float u = u01_from_u32(r.x);

    float acc = 0.0f;
    int picked = topi[0]; // fallback
    for (int t = 0; t < cutoff; ++t) {
      acc += probs[t] / psum;
      if (u <= acc) { picked = topi[t]; break; }
    }
    out_tokens[b] = picked;
  }
}

template<int K, typename T>
void gpu_topk_topp_sample(
    const T* logits_d,
    int* out_tokens_d,
    const SamplerParams& p,
    cudaStream_t stream
) {
  // Best-practice defaults:
  // - 256 threads is a good balance for scanning vocab.
  // - ITEMS_PER_THREAD controls chunk size. 4 is safe; 8 may be faster on some GPUs.
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int CHUNK = BLOCK_THREADS * ITEMS_PER_THREAD;

  // tiles along vocab; ensure at least 1
  int tiles = (p.V + CHUNK - 1) / CHUNK;

  // Allocate intermediate buffers: [B, tiles, K]
  // In production, allocate once and reuse.
  // For now, consistent with user request, we allocate here.
  float* tile_vals_d = nullptr;
  int*   tile_idx_d  = nullptr;

  size_t vals_bytes = (size_t)p.B * (size_t)tiles * (size_t)K * sizeof(float);
  size_t idx_bytes  = (size_t)p.B * (size_t)tiles * (size_t)K * sizeof(int);

  CUDA_CHECK(cudaMallocAsync(&tile_vals_d, vals_bytes, stream));
  CUDA_CHECK(cudaMallocAsync(&tile_idx_d,  idx_bytes,  stream));

  // Stage A: grid = (B, tiles)
  dim3 gridA(p.B, tiles, 1);
  dim3 blockA(BLOCK_THREADS, 1, 1);
  stageA_local_topk<K, BLOCK_THREADS, ITEMS_PER_THREAD, T>
      <<<gridA, blockA, 0, stream>>>(
          logits_d, p.B, p.V, p.temperature, tiles, tile_vals_d, tile_idx_d);

  // Stage B: one block per batch element
  dim3 gridB(p.B, 1, 1);
  dim3 blockB(256, 1, 1);
  stageB_reduce_and_sample<K, 256>
      <<<gridB, blockB, 0, stream>>>(
          p.B, tiles, tile_vals_d, tile_idx_d, p.top_p, p.top_k, p.seed, p.token_pos, out_tokens_d);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(tile_vals_d, stream));
  CUDA_CHECK(cudaFreeAsync(tile_idx_d,  stream));
}

// Explicit instantiations for float
template void gpu_topk_topp_sample<32, float>(const float*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<64, float>(const float*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<128, float>(const float*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<256, float>(const float*, int*, const SamplerParams&, cudaStream_t);

// Explicit instantiations for __half (f16)
template void gpu_topk_topp_sample<32, __half>(const __half*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<64, __half>(const __half*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<128, __half>(const __half*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<256, __half>(const __half*, int*, const SamplerParams&, cudaStream_t);

// Explicit instantiations for __nv_bfloat16 (bf16) - requires sm_80+
#ifndef NO_BF16_KERNEL
template void gpu_topk_topp_sample<32, __nv_bfloat16>(const __nv_bfloat16*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<64, __nv_bfloat16>(const __nv_bfloat16*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<128, __nv_bfloat16>(const __nv_bfloat16*, int*, const SamplerParams&, cudaStream_t);
template void gpu_topk_topp_sample<256, __nv_bfloat16>(const __nv_bfloat16*, int*, const SamplerParams&, cudaStream_t);
#endif


extern "C" void sampling_f32(
    const float* logits_d,
    int* out_tokens_d,
    int B,
    int V,
    int K,
    float temperature,
    float top_p,
    uint64_t seed,
    uint64_t token_pos,
    int64_t stream_ptr) 
{
    SamplerParams p;
    p.B = B;
    p.V = V;
    p.temperature = temperature;
    p.top_p = top_p;
    int k_eff = K <= 0 ? V : K;
    if (k_eff > V) k_eff = V;
    if (k_eff < 1) k_eff = 1;
    if (k_eff > 256) k_eff = 256;
    p.top_k = k_eff;
    p.seed = seed;
    p.token_pos = token_pos;

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    if (k_eff <= 32) {
        gpu_topk_topp_sample<32, float>(logits_d, out_tokens_d, p, stream);
    } else if (k_eff <= 64) {
        gpu_topk_topp_sample<64, float>(logits_d, out_tokens_d, p, stream);
    } else if (k_eff <= 128) {
        gpu_topk_topp_sample<128, float>(logits_d, out_tokens_d, p, stream);
    } else {
        gpu_topk_topp_sample<256, float>(logits_d, out_tokens_d, p, stream);
    }
}

extern "C" void sampling_f16(
    const void* logits_d,
    int* out_tokens_d,
    int B,
    int V,
    int K,
    float temperature,
    float top_p,
    uint64_t seed,
    uint64_t token_pos,
    int64_t stream_ptr) 
{
    SamplerParams p;
    p.B = B;
    p.V = V;
    p.temperature = temperature;
    p.top_p = top_p;
    int k_eff = K <= 0 ? V : K;
    if (k_eff > V) k_eff = V;
    if (k_eff < 1) k_eff = 1;
    if (k_eff > 256) k_eff = 256;
    p.top_k = k_eff;
    p.seed = seed;
    p.token_pos = token_pos;

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const __half* logits = reinterpret_cast<const __half*>(logits_d);

    if (k_eff <= 32) {
        gpu_topk_topp_sample<32, __half>(logits, out_tokens_d, p, stream);
    } else if (k_eff <= 64) {
        gpu_topk_topp_sample<64, __half>(logits, out_tokens_d, p, stream);
    } else if (k_eff <= 128) {
        gpu_topk_topp_sample<128, __half>(logits, out_tokens_d, p, stream);
    } else {
        gpu_topk_topp_sample<256, __half>(logits, out_tokens_d, p, stream);
    }
}

extern "C" void sampling_bf16(
    const void* logits_d,
    int* out_tokens_d,
    int B,
    int V,
    int K,
    float temperature,
    float top_p,
    uint64_t seed,
    uint64_t token_pos,
    int64_t stream_ptr) 
{
    SamplerParams p;
    p.B = B;
    p.V = V;
    p.temperature = temperature;
    p.top_p = top_p;
    int k_eff = K <= 0 ? V : K;
    if (k_eff > V) k_eff = V;
    if (k_eff < 1) k_eff = 1;
    if (k_eff > 256) k_eff = 256;
    p.top_k = k_eff;
    p.seed = seed;
    p.token_pos = token_pos;

    cudaStream_t stream = (cudaStream_t)stream_ptr;
  #ifndef NO_BF16_KERNEL
    const __nv_bfloat16* logits = reinterpret_cast<const __nv_bfloat16*>(logits_d);

    if (k_eff <= 32) {
        gpu_topk_topp_sample<32, __nv_bfloat16>(logits, out_tokens_d, p, stream);
    } else if (k_eff <= 64) {
        gpu_topk_topp_sample<64, __nv_bfloat16>(logits, out_tokens_d, p, stream);
    } else if (k_eff <= 128) {
        gpu_topk_topp_sample<128, __nv_bfloat16>(logits, out_tokens_d, p, stream);
    } else {
        gpu_topk_topp_sample<256, __nv_bfloat16>(logits, out_tokens_d, p, stream);
    }
  #endif
}
