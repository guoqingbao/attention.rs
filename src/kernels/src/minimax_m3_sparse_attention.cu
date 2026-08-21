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

__device__ inline float m3a_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// Hopper/Blackwell path. Producer warps compute independent QK positions while
// four consumer warps own the output vector. Blackwell uses the official MSA
// 16-warp CTA shape and twice the producer width; Hopper uses an 8-warp CTA to
// preserve occupancy. Q and probabilities remain resident in shared memory.
template <typename T, int THREADS, int PRODUCER_WARPS, int CONSUMER_WARP>
__global__ __launch_bounds__(THREADS, 1) void minimax_m3_sparse_attention_sm90_kernel(
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
  const int candidates = topk_blocks * block_size;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  extern __shared__ float smem[];
  float* q_smem = smem;
  float* probs = q_smem + head_dim;
  float* reduce = probs + candidates;
  for (int d = tid; d < head_dim; d += blockDim.x)
    q_smem[d] = m3a_load(q + q_base + d);
  __syncthreads();

  for (int base = 0; base < candidates; base += PRODUCER_WARPS) {
    const int candidate = base + warp;
    if (warp < PRODUCER_WARPS && candidate < candidates) {
      const int slot = candidate / block_size;
      const int within_block = candidate - slot * block_size;
      const int selected_block = topk[topk_base + slot];
      const int pos = selected_block * block_size + within_block;
      float dot = 0.0f;
      if (selected_block >= 0 && pos <= local_token) {
        const int k_base = ((seq_start + pos) * num_kv_heads + kv_head) * head_dim;
        #pragma unroll 4
        for (int d = lane; d < head_dim; d += 32)
          dot += q_smem[d] * m3a_load(k + k_base + d);
        dot = m3a_warp_sum(dot) * scale;
      } else {
        dot = -FLT_MAX;
      }
      if (lane == 0) probs[candidate] = dot;
    }
  }
  __syncthreads();

  float local_max = -FLT_MAX;
  for (int candidate = tid; candidate < candidates; candidate += blockDim.x)
    local_max = fmaxf(local_max, probs[candidate]);
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    __syncthreads();
  }
  const float qk_max = reduce[0];

  float local_sum = 0.0f;
  for (int candidate = tid; candidate < candidates; candidate += blockDim.x) {
    const float raw = probs[candidate];
    const float prob = raw == -FLT_MAX ? 0.0f : expf(raw - qk_max);
    probs[candidate] = prob;
    local_sum += prob;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }
  const float denom = reduce[0];

  // Exactly four consumer warps own the 128 output elements.
  if (warp >= CONSUMER_WARP && warp < CONSUMER_WARP + 4) {
    const int d = (warp - CONSUMER_WARP) * 32 + lane;
    if (d < head_dim) {
      float acc = 0.0f;
      for (int candidate = 0; candidate < candidates; ++candidate) {
        const float prob = probs[candidate];
        if (prob == 0.0f) continue;
        const int slot = candidate / block_size;
        const int within_block = candidate - slot * block_size;
        const int pos = topk[topk_base + slot] * block_size + within_block;
        const int v_base = ((seq_start + pos) * num_kv_heads + kv_head) * head_dim;
        acc += prob * m3a_load(v + v_base + d);
      }
      m3a_store(out + q_base + d, acc / denom);
    }
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
  const auto* cu_seqlens = reinterpret_cast<const uint32_t*>(cu_seqlens_);
  int device = 0;
  int major = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  const bool sm90_fast = major >= 9 && head_dim == 128 && block_size == 128;
  const bool sm100_fast = major >= 10 && head_dim == 128 && block_size == 128;
  const int fast_threads = sm100_fast ? 512 : 256;
  dim3 block(fast_threads);
  const size_t sm90_smem = (static_cast<size_t>(head_dim) +
      static_cast<size_t>(topk_blocks) * block_size + fast_threads) * sizeof(float);
#define LAUNCH_M3_FAST(T)                                                        \
  do {                                                                           \
    if (sm100_fast)                                                              \
      minimax_m3_sparse_attention_sm90_kernel<T, 512, 8, 8>                     \
          <<<grid, block, sm90_smem, stream>>>(                                  \
              reinterpret_cast<T*>(out), reinterpret_cast<const T*>(q),          \
              reinterpret_cast<const T*>(k), reinterpret_cast<const T*>(v),      \
              topk, cu_seqlens, total_tokens, batch_size, num_heads,             \
              num_kv_heads, head_dim, topk_blocks, block_size, scale);           \
    else                                                                         \
      minimax_m3_sparse_attention_sm90_kernel<T, 256, 4, 4>                     \
          <<<grid, block, sm90_smem, stream>>>(                                  \
              reinterpret_cast<T*>(out), reinterpret_cast<const T*>(q),          \
              reinterpret_cast<const T*>(k), reinterpret_cast<const T*>(v),      \
              topk, cu_seqlens, total_tokens, batch_size, num_heads,             \
              num_kv_heads, head_dim, topk_blocks, block_size, scale);           \
  } while (0)
  switch (dtype) {
    case 0:
      if (sm90_fast) LAUNCH_M3_FAST(__half);
      else minimax_m3_sparse_attention_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(q),
          reinterpret_cast<const __half*>(k), reinterpret_cast<const __half*>(v), topk,
          cu_seqlens, total_tokens, batch_size, num_heads, num_kv_heads, head_dim,
          topk_blocks, block_size, scale);
      break;
    case 1:
      if (sm90_fast) LAUNCH_M3_FAST(__nv_bfloat16);
      else minimax_m3_sparse_attention_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(out), reinterpret_cast<const __nv_bfloat16*>(q),
          reinterpret_cast<const __nv_bfloat16*>(k), reinterpret_cast<const __nv_bfloat16*>(v), topk,
          cu_seqlens, total_tokens, batch_size, num_heads, num_kv_heads, head_dim,
          topk_blocks, block_size, scale);
      break;
    case 2:
      if (sm90_fast) LAUNCH_M3_FAST(float);
      else minimax_m3_sparse_attention_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<float*>(out), reinterpret_cast<const float*>(q),
          reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v), topk,
          cu_seqlens, total_tokens, batch_size, num_heads, num_kv_heads, head_dim,
          topk_blocks, block_size, scale);
      break;
    default: return cudaErrorInvalidValue;
  }
#undef LAUNCH_M3_FAST
  return cudaGetLastError();
}
