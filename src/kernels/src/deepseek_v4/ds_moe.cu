// DeepSeek V4 fused hash-gate routing (tid2eid lookup + per-expert gate dots).
// Ported from openinfer deepseek_moe.cu — avoids full router matmul + candle gather.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__global__ void ds_v4_hash_gate_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const long long *__restrict__ tid2eid,
    const unsigned int *__restrict__ token_ids,
    float *__restrict__ route_weights,
    unsigned int *__restrict__ route_indices,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    int vocab_size) {
  int route = blockIdx.x;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  if (route >= topk || token >= seq_len) return;

  extern __shared__ float scratch[];
  unsigned int token_id = token_ids[token];
  int expert = -1;
  if (vocab_size > 0 && token_id < (unsigned int)vocab_size) {
    long long expert_i64 = tid2eid[(size_t)token_id * (size_t)topk + (size_t)route];
    expert = (int)expert_i64;
  }

  float partial = 0.0f;
  if (expert >= 0 && expert < n_experts) {
    for (int k = tid; k < hidden_dim; k += blockDim.x) {
      float xv = __bfloat162float(x[(size_t)token * (size_t)hidden_dim + (size_t)k]);
      float wv = __bfloat162float(
          gate_weight[(size_t)expert * (size_t)hidden_dim + (size_t)k]);
      partial += xv * wv;
    }
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float score = scratch[0];
    float softplus = score > 20.0f ? score : log1pf(expf(score));
    float routed = sqrtf(softplus);
    route_weights[(size_t)token * (size_t)topk + (size_t)route] = routed;
    route_indices[(size_t)token * (size_t)topk + (size_t)route] =
        (expert >= 0 && expert < n_experts) ? (unsigned int)expert : 0u;
  }
}

__global__ void ds_v4_route_normalize_kernel(
    float *__restrict__ route_weights,
    int seq_len,
    int topk,
    float route_scale) {
  int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= seq_len) return;

  float sum = 0.0f;
  for (int route = 0; route < topk; ++route) {
    sum += route_weights[(size_t)token * (size_t)topk + (size_t)route];
  }
  float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
  for (int route = 0; route < topk; ++route) {
    route_weights[(size_t)token * (size_t)topk + (size_t)route] =
        route_weights[(size_t)token * (size_t)topk + (size_t)route] * inv_sum *
        route_scale;
  }
}

}  // namespace

extern "C" {

cudaError_t ds_v4_hash_gate(
    const void *x,
    const void *gate_weight,
    const void *tid2eid,
    const void *token_ids,
    void *route_weights,
    void *route_indices,
    int seq_len,
    int hidden_dim,
    int n_experts,
    int topk,
    int vocab_size,
    float route_scale,
    long long stream) {
  if (seq_len <= 0 || hidden_dim <= 0 || n_experts <= 0 || topk <= 0) {
    return cudaErrorInvalidValue;
  }
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
  constexpr int threads = 256;
  dim3 score_grid(topk, seq_len);
  size_t shared_bytes = threads * sizeof(float);
  ds_v4_hash_gate_kernel<<<score_grid, threads, shared_bytes, cu_stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(x),
      reinterpret_cast<const __nv_bfloat16 *>(gate_weight),
      reinterpret_cast<const long long *>(tid2eid),
      reinterpret_cast<const unsigned int *>(token_ids),
      reinterpret_cast<float *>(route_weights),
      reinterpret_cast<unsigned int *>(route_indices),
      seq_len,
      hidden_dim,
      n_experts,
      topk,
      vocab_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (seq_len + threads - 1) / threads;
  ds_v4_route_normalize_kernel<<<norm_blocks, threads, 0, cu_stream>>>(
      reinterpret_cast<float *>(route_weights), seq_len, topk, route_scale);
  return cudaGetLastError();
}

}  // extern "C"
