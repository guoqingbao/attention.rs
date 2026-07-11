/*
 * MiniMax M3 block indexer.
 *
 * M3 differs from DeepSeek DSA: its index branch has one query per KV group
 * and scores 128-token blocks with a max reduction over the tokens in each
 * block.  This kernel is intentionally small and conservative; the SM100 MSA
 * implementation can replace it later without changing the Rust API.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>

template <typename T> __device__ inline float m3_load(const T* p) {
  return static_cast<float>(*p);
}
template <> __device__ inline float m3_load<__half>(const __half* p) {
  return __half2float(*p);
}
template <> __device__ inline float m3_load<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}

__device__ inline int m3_find_sequence(const uint32_t* cu_seqlens, int batch_size,
                                        int token) {
  int lo = 0;
  int hi = batch_size;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (cu_seqlens[mid + 1] <= static_cast<uint32_t>(token)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

template <typename T>
__global__ void minimax_m3_indexer_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    int32_t* __restrict__ topk_out, const uint32_t* __restrict__ cu_seqlens,
    int total_tokens, int batch_size, int n_heads, int head_dim, int topk,
    int block_size, float scale) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= total_tokens * n_heads) return;

  const int token = row / n_heads;
  const int head = row % n_heads;
  const int seq = m3_find_sequence(cu_seqlens, batch_size, token);
  const int seq_start = static_cast<int>(cu_seqlens[seq]);
  const int seq_end = static_cast<int>(cu_seqlens[seq + 1]);
  const int seq_len = seq_end - seq_start;
  const int local_token = token - seq_start;
  const int num_blocks = (seq_len + block_size - 1) / block_size;
  const int q_base = (token * n_heads + head) * head_dim;

  extern __shared__ float scores[];
  for (int block = tid; block < num_blocks; block += blockDim.x) {
    const int begin = block * block_size;
    const int end = min(begin + block_size, local_token + 1);
    float best = -FLT_MAX;
    for (int pos = begin; pos < end; ++pos) {
      float dot = 0.0f;
      const int k_base = (seq_start + pos) * head_dim;
      for (int d = 0; d < head_dim; ++d)
        dot += m3_load(q + q_base + d) * m3_load(k + k_base + d);
      best = fmaxf(best, dot);
    }
    scores[block] = best * scale;
    if (block == local_token / block_size) scores[block] = FLT_MAX;
  }
  __syncthreads();

  if (tid == 0) {
    const int actual_topk = min(topk, num_blocks);
    int32_t* out = topk_out + (static_cast<size_t>(token) * n_heads * topk) + head * topk;
    for (int i = 0; i < actual_topk; ++i) {
      float best = -FLT_MAX;
      int best_idx = -1;
      for (int block = 0; block < num_blocks; ++block) {
        if (scores[block] > best) {
          best = scores[block];
          best_idx = block;
        }
      }
      out[i] = best_idx;
      if (best_idx >= 0) scores[best_idx] = -FLT_MAX;
    }
    for (int i = actual_topk; i < topk; ++i) out[i] = -1;
  }
}

// Hopper/Blackwell tensor-core path. A CTA computes a complete 128-query tile
// against one 128-key block at a time. The eight warps each issue a 16x16x128
// WMMA and keep every query row's block top-k resident in shared memory. The
// grid maps (sequence, query-tile, head), so ragged batches are processed by a
// single launch without per-sequence host dispatch.
template <typename T>
__global__ __launch_bounds__(256, 1) void minimax_m3_indexer_sm90_wmma_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    int32_t* __restrict__ topk_out, const uint32_t* __restrict__ cu_seqlens,
    int total_tokens, int batch_size, int n_heads, int head_dim, int topk,
    int block_size, int max_seq_len, float scale) {
  constexpr int TILE = 128;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 16;
  constexpr int TOPK = 16;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int tiles_per_seq = (max_seq_len + TILE - 1) / TILE;
  const int work = blockIdx.x;
  const int head = blockIdx.y;
  const int seq = work / tiles_per_seq;
  const int q_tile = work - seq * tiles_per_seq;
  if (seq >= batch_size) return;
  const int seq_start = static_cast<int>(cu_seqlens[seq]);
  const int seq_end = static_cast<int>(cu_seqlens[seq + 1]);
  const int seq_len = seq_end - seq_start;
  const int q_begin = q_tile * TILE;
  if (q_begin >= seq_len) return;
  const int q_count = min(TILE, seq_len - q_begin);

  extern __shared__ __align__(16) unsigned char storage[];
  T* q_smem = reinterpret_cast<T*>(storage);
  T* k_smem = q_smem + TILE * TILE;
  float* mma_scores = reinterpret_cast<float*>(k_smem + MMA_N * TILE);
  float* block_best = mma_scores + TILE * MMA_N;
  float* top_scores = block_best + TILE;
  int32_t* top_indices = reinterpret_cast<int32_t*>(top_scores + TILE * TOPK);

  for (int i = tid; i < TILE * TILE; i += blockDim.x) {
    const int row = i / TILE;
    const int d = i - row * TILE;
    q_smem[i] = row < q_count
        ? q[((seq_start + q_begin + row) * n_heads + head) * head_dim + d]
        : static_cast<T>(0.0f);
  }
  for (int i = tid; i < TILE * TOPK; i += blockDim.x) {
    top_scores[i] = -FLT_MAX;
    top_indices[i] = -1;
  }
  __syncthreads();

  // Only blocks that can be causal for at least one row in this query tile.
  for (int key_block = 0; key_block <= q_tile; ++key_block) {
    if (tid < TILE) block_best[tid] = -FLT_MAX;
    __syncthreads();

    #pragma unroll
    for (int key_subtile = 0; key_subtile < TILE / MMA_N; ++key_subtile) {
      for (int i = tid; i < MMA_N * TILE; i += blockDim.x) {
        const int key_row = i / TILE;
        const int d = i - key_row * TILE;
        const int key_pos = key_block * TILE + key_subtile * MMA_N + key_row;
        k_smem[i] = key_pos < seq_len ? k[(seq_start + key_pos) * head_dim + d]
                                       : static_cast<T>(0.0f);
      }
      __syncthreads();

      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, TILE, T,
                     wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, TILE, T,
                     wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, MMA_M, MMA_N, TILE, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      wmma::load_matrix_sync(a_frag, q_smem + warp * MMA_M * TILE, TILE);
      // K is [16, 128] row-major; viewing the same bytes as [128, 16]
      // column-major supplies K^T without a shared-memory transpose.
      wmma::load_matrix_sync(b_frag, k_smem, TILE);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::store_matrix_sync(mma_scores + warp * MMA_M * MMA_N, c_frag,
                              MMA_N, wmma::mem_row_major);
      __syncthreads();

      if (tid < TILE && tid < q_count) {
        const int query_pos = q_begin + tid;
        float best = block_best[tid];
        #pragma unroll
        for (int col = 0; col < MMA_N; ++col) {
          const int key_pos = key_block * TILE + key_subtile * MMA_N + col;
          if (key_pos <= query_pos && key_pos < seq_len)
            best = fmaxf(best, mma_scores[tid * MMA_N + col]);
        }
        block_best[tid] = best;
      }
      __syncthreads();
    }

    if (tid < TILE && tid < q_count) {
      const int query_pos = q_begin + tid;
      const float candidate = key_block == query_pos / TILE
                                  ? FLT_MAX
                                  : block_best[tid] * scale;
      float* row_scores = top_scores + tid * TOPK;
      int32_t* row_indices = top_indices + tid * TOPK;
      int min_slot = 0;
      #pragma unroll
      for (int slot = 1; slot < TOPK; ++slot)
        if (row_scores[slot] < row_scores[min_slot]) min_slot = slot;
      if (candidate > row_scores[min_slot]) {
        row_scores[min_slot] = candidate;
        row_indices[min_slot] = key_block;
      }
    }
    __syncthreads();
  }

  if (tid < TILE && tid < q_count) {
    const int token = seq_start + q_begin + tid;
    int32_t* out = topk_out +
        (static_cast<size_t>(token) * n_heads + head) * topk;
    float* row_scores = top_scores + tid * TOPK;
    int32_t* row_indices = top_indices + tid * TOPK;
    const int actual_topk = min(topk, TOPK);
    for (int out_slot = 0; out_slot < actual_topk; ++out_slot) {
      int best_slot = 0;
      #pragma unroll
      for (int slot = 1; slot < TOPK; ++slot)
        if (row_scores[slot] > row_scores[best_slot]) best_slot = slot;
      out[out_slot] = row_indices[best_slot];
      row_scores[best_slot] = -FLT_MAX;
    }
    for (int out_slot = actual_topk; out_slot < topk; ++out_slot)
      out[out_slot] = -1;
  }
}

extern "C" cudaError_t minimax_m3_indexer_prefill(
    const void* q, const void* k, int32_t* topk_out, const void* cu_seqlens_,
    int total_tokens, int batch_size, int n_heads, int head_dim, int topk,
    int block_size, int max_seq_len, float scale, uint32_t dtype, int64_t stream_) {
  if (total_tokens <= 0 || batch_size <= 0 || n_heads <= 0 || head_dim <= 0 ||
      topk <= 0 || block_size <= 0 || max_seq_len <= 0)
    return cudaErrorInvalidValue;
  const size_t smem = static_cast<size_t>((max_seq_len + block_size - 1) / block_size) * sizeof(float);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  dim3 grid(total_tokens * n_heads);
  dim3 block(256);
  const auto* cu_seqlens = reinterpret_cast<const uint32_t*>(cu_seqlens_);
  int device = 0;
  int major = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  const bool sm90_fast = major >= 9 && head_dim == 128 && block_size == 128 && topk == 16;
  const int tiles_per_seq = (max_seq_len + block_size - 1) / block_size;
  dim3 sm90_grid(batch_size * tiles_per_seq, n_heads);
  const size_t sm90_smem =
      (128 * 128 + 16 * 128) * sizeof(__half) +
      (128 * 16 + 128 + 128 * 16) * sizeof(float) +
      128 * 16 * sizeof(int32_t);
  switch (dtype) {
    case 0:
      if (sm90_fast) {
        cudaFuncSetAttribute(minimax_m3_indexer_sm90_wmma_kernel<__half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm90_smem));
        minimax_m3_indexer_sm90_wmma_kernel<<<sm90_grid, block, sm90_smem, stream>>>(
          reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k), topk_out,
          cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size,
          max_seq_len, scale);
      } else minimax_m3_indexer_kernel<<<grid, block, smem, stream>>>(
          reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k), topk_out,
          cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size, scale);
      break;
    case 1:
      if (sm90_fast) {
        cudaFuncSetAttribute(minimax_m3_indexer_sm90_wmma_kernel<__nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm90_smem));
        minimax_m3_indexer_sm90_wmma_kernel<<<sm90_grid, block, sm90_smem, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k), topk_out,
          cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size,
          max_seq_len, scale);
      } else minimax_m3_indexer_kernel<<<grid, block, smem, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k), topk_out,
          cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size, scale);
      break;
    case 2:
      minimax_m3_indexer_kernel<<<grid, block, smem, stream>>>(
          reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k), topk_out,
          cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size, scale);
      break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
