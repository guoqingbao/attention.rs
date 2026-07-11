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
  switch (dtype) {
    case 0: minimax_m3_indexer_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k), topk_out,
        cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size, scale); break;
    case 1: minimax_m3_indexer_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k), topk_out,
        cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size, scale); break;
    case 2: minimax_m3_indexer_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k), topk_out,
        cu_seqlens, total_tokens, batch_size, n_heads, head_dim, topk, block_size, scale); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
