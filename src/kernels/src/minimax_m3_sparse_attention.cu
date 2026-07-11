/* Correctness-first MiniMax M3 sparse GQA prefill kernel. */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

template <typename T> __device__ inline float m3a_load(const T* p) {
  return static_cast<float>(*p);
}
template <> __device__ inline float m3a_load<__half>(const __half* p) {
  return __half2float(*p);
}
template <> __device__ inline float m3a_load<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}
template <typename T> __device__ inline void m3a_store(T* p, float v) { *p = static_cast<T>(v); }
template <> __device__ inline void m3a_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void m3a_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ inline int m3a_find_sequence(const uint32_t* cu_seqlens, int batch_size,
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
__global__ void minimax_m3_sparse_attention_kernel(
    T* __restrict__ out, const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, const int32_t* __restrict__ topk,
    const uint32_t* __restrict__ cu_seqlens, int total_tokens, int batch_size,
    int num_heads, int num_kv_heads, int head_dim, int topk_blocks,
    int block_size, float scale) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= total_tokens * num_heads) return;
  const int token = row / num_heads;
  const int head = row % num_heads;
  const int seq = m3a_find_sequence(cu_seqlens, batch_size, token);
  const int seq_start = static_cast<int>(cu_seqlens[seq]);
  const int local_token = token - seq_start;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int q_base = (token * num_heads + head) * head_dim;
  const int topk_base = (token * num_kv_heads + kv_head) * topk_blocks;

  __shared__ float qk_max;
  __shared__ float denom;
  __shared__ float reduce[256];

  if (tid == 0) {
    float best = -FLT_MAX;
    for (int slot = 0; slot < topk_blocks; ++slot) {
      const int block = topk[topk_base + slot];
      if (block < 0) continue;
      const int begin = block * block_size;
      const int end = min(begin + block_size, local_token + 1);
      for (int pos = begin; pos < end; ++pos) {
        const int k_base = ((seq_start + pos) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += m3a_load(q + q_base + d) * m3a_load(k + k_base + d);
        best = fmaxf(best, dot * scale);
      }
    }
    qk_max = best;
  }
  __syncthreads();

  float partial = 0.0f;
  for (int slot = 0; slot < topk_blocks; ++slot) {
    const int block = topk[topk_base + slot];
    if (block < 0) continue;
    const int begin = block * block_size;
    const int end = min(begin + block_size, local_token + 1);
    for (int pos = begin + tid; pos < end; pos += blockDim.x) {
      const int k_base = ((seq_start + pos) * num_kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d)
        dot += m3a_load(q + q_base + d) * m3a_load(k + k_base + d);
      partial += expf(dot * scale - qk_max);
    }
  }
  reduce[tid] = partial;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }
  if (tid == 0) denom = reduce[0];
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int slot = 0; slot < topk_blocks; ++slot) {
      const int block = topk[topk_base + slot];
      if (block < 0) continue;
      const int begin = block * block_size;
      const int end = min(begin + block_size, local_token + 1);
      for (int pos = begin; pos < end; ++pos) {
        const int base = ((seq_start + pos) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int j = 0; j < head_dim; ++j)
          dot += m3a_load(q + q_base + j) * m3a_load(k + base + j);
        acc += expf(dot * scale - qk_max) * m3a_load(v + base + d);
      }
    }
    m3a_store(out + q_base + d, acc / denom);
  }
}

extern "C" cudaError_t minimax_m3_sparse_attention_prefill(
    void* out, const void* q, const void* k, const void* v, const int32_t* topk,
    const void* cu_seqlens_, int total_tokens, int batch_size, int num_heads,
    int num_kv_heads, int head_dim, int topk_blocks, int block_size,
    int max_seq_len, float scale, uint32_t dtype, int64_t stream_) {
  if (total_tokens <= 0 || batch_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
      head_dim <= 0 || topk_blocks <= 0 || block_size <= 0 || max_seq_len <= 0 ||
      num_heads % num_kv_heads != 0)
    return cudaErrorInvalidValue;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  dim3 grid(total_tokens * num_heads);
  dim3 block(256);
  const auto* cu_seqlens = reinterpret_cast<const uint32_t*>(cu_seqlens_);
  switch (dtype) {
    case 0: minimax_m3_sparse_attention_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k), reinterpret_cast<const __half*>(v), topk,
        cu_seqlens, total_tokens, batch_size, num_heads, num_kv_heads, head_dim,
        topk_blocks, block_size, scale); break;
    case 1: minimax_m3_sparse_attention_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out), reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k), reinterpret_cast<const __nv_bfloat16*>(v), topk,
        cu_seqlens, total_tokens, batch_size, num_heads, num_kv_heads, head_dim,
        topk_blocks, block_size, scale); break;
    case 2: minimax_m3_sparse_attention_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<const float*>(q),
        reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v), topk,
        cu_seqlens, total_tokens, batch_size, num_heads, num_kv_heads, head_dim,
        topk_blocks, block_size, scale); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
